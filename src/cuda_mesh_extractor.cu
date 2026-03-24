#include "cuda_mesh_extractor.h"
#include "cuda_support.h"

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace tsdfmc {

namespace {

constexpr int kCudaBlockSize = 8;
constexpr int kCudaBlockVolume = kCudaBlockSize * kCudaBlockSize * kCudaBlockSize;
constexpr int kMaxTrianglesPerCube = 12;
constexpr int kEmptyLookupIndex = -1;

static_assert(VoxelHashTSDF::kBlockSize == kCudaBlockSize, "CUDA mesh extractor assumes 8x8x8 voxel blocks.");
static_assert(VoxelHashTSDF::kBlockVolume == kCudaBlockVolume, "CUDA mesh extractor block volume mismatch.");

struct MeshLookupEntry {
  BlockKey key;
  int index = kEmptyLookupIndex;
};

struct GpuMeshParams {
  float voxel_size;
  float iso_level;
  unsigned int num_blocks;
  unsigned int lookup_table_size;
};

__constant__ GpuMeshParams c_mesh_params;

__device__ __constant__ int c_corner_offsets[8][3] = {
    {0, 0, 0},
    {1, 0, 0},
    {1, 1, 0},
    {0, 1, 0},
    {0, 0, 1},
    {1, 0, 1},
    {1, 1, 1},
    {0, 1, 1},
};

__device__ __constant__ int c_cube_tetrahedra[6][4] = {
    {0, 5, 1, 6},
    {0, 1, 2, 6},
    {0, 2, 3, 6},
    {0, 3, 7, 6},
    {0, 7, 4, 6},
    {0, 4, 5, 6},
};

struct MeshExtractorWorkspace {
  ReusableDeviceBuffer<VoxelHashTSDF::FlatBlockRecord> flat_blocks;
  ReusableDeviceBuffer<MeshLookupEntry> lookup_table;
  ReusableDeviceBuffer<unsigned int> triangle_count;
  ReusableDeviceBuffer<Triangle> triangles;
};

MeshExtractorWorkspace& meshExtractorWorkspace() {
  static MeshExtractorWorkspace workspace;
  return workspace;
}

unsigned int hashBlockKeyHost(const BlockKey& key, const unsigned int table_size) {
  const int h0 = key.x * 73856093;
  const int h1 = key.y * 19349663;
  const int h2 = key.z * 83492791;
  int hash_value = (h0 ^ h1 ^ h2) % static_cast<int>(table_size);
  if (hash_value < 0) {
    hash_value += static_cast<int>(table_size);
  }
  return static_cast<unsigned int>(hash_value);
}

std::vector<MeshLookupEntry> buildLookupTable(const std::vector<VoxelHashTSDF::FlatBlockRecord>& flat_blocks) {
  unsigned int table_size = 1;
  while (table_size < flat_blocks.size() * 2U) {
    table_size <<= 1U;
  }
  if (table_size == 0U) {
    table_size = 1U;
  }

  std::vector<MeshLookupEntry> lookup_table(static_cast<std::size_t>(table_size));
  for (MeshLookupEntry& entry : lookup_table) {
    entry.index = kEmptyLookupIndex;
  }

  for (std::size_t i = 0; i < flat_blocks.size(); ++i) {
    const BlockKey key = flat_blocks[i].key;
    unsigned int slot = hashBlockKeyHost(key, table_size);
    while (lookup_table[slot].index != kEmptyLookupIndex) {
      slot = (slot + 1U) % table_size;
    }
    lookup_table[slot].key = key;
    lookup_table[slot].index = static_cast<int>(i);
  }

  return lookup_table;
}

__device__ unsigned int hashBlockKeyDevice(const BlockKey& key, const unsigned int table_size) {
  const int h0 = key.x * 73856093;
  const int h1 = key.y * 19349663;
  const int h2 = key.z * 83492791;
  int hash_value = (h0 ^ h1 ^ h2) % static_cast<int>(table_size);
  if (hash_value < 0) {
    hash_value += static_cast<int>(table_size);
  }
  return static_cast<unsigned int>(hash_value);
}

__device__ int floorDivDevice(const int value, const int divisor) {
  int quotient = value / divisor;
  const int remainder = value % divisor;
  if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
    --quotient;
  }
  return quotient;
}

__device__ int positiveModDevice(const int value, const int divisor) {
  int remainder = value % divisor;
  if (remainder < 0) {
    remainder += divisor;
  }
  return remainder;
}

__device__ bool lookupBlockIndex(const MeshLookupEntry* lookup_table, const BlockKey& key, int& index) {
  unsigned int slot = hashBlockKeyDevice(key, c_mesh_params.lookup_table_size);
  for (unsigned int probe = 0; probe < c_mesh_params.lookup_table_size; ++probe) {
    const MeshLookupEntry& entry = lookup_table[slot];
    if (entry.index == kEmptyLookupIndex) {
      return false;
    }
    if (entry.key.x == key.x && entry.key.y == key.y && entry.key.z == key.z) {
      index = entry.index;
      return true;
    }
    slot = (slot + 1U) % c_mesh_params.lookup_table_size;
  }
  return false;
}

__device__ bool fetchVoxel(const VoxelHashTSDF::FlatBlockRecord* flat_blocks,
                           const MeshLookupEntry* lookup_table,
                           const int vx,
                           const int vy,
                           const int vz,
                           Voxel& voxel_out) {
  const BlockKey key{
      floorDivDevice(vx, kCudaBlockSize),
      floorDivDevice(vy, kCudaBlockSize),
      floorDivDevice(vz, kCudaBlockSize),
  };

  int block_index = kEmptyLookupIndex;
  if (!lookupBlockIndex(lookup_table, key, block_index)) {
    return false;
  }

  const int lx = positiveModDevice(vx, kCudaBlockSize);
  const int ly = positiveModDevice(vy, kCudaBlockSize);
  const int lz = positiveModDevice(vz, kCudaBlockSize);
  const int local_index = lx + kCudaBlockSize * (ly + kCudaBlockSize * lz);
  voxel_out = flat_blocks[block_index].voxels[local_index];
  return voxel_out.weight > 0.0f;
}

__device__ Vec3f voxelCoordToWorld(const int vx, const int vy, const int vz) {
  return {
      (static_cast<float>(vx) + 0.5f) * c_mesh_params.voxel_size,
      (static_cast<float>(vy) + 0.5f) * c_mesh_params.voxel_size,
      (static_cast<float>(vz) + 0.5f) * c_mesh_params.voxel_size,
  };
}

__device__ Vec3f interpolateVertexGpu(const Vec3f& p0,
                                      const Vec3f& p1,
                                      const float v0,
                                      const float v1,
                                      const float iso_level) {
  const float delta = v1 - v0;
  if (fabsf(delta) < 1e-6f) {
    return p0;
  }

  const float t = fminf(1.0f, fmaxf(0.0f, (iso_level - v0) / delta));
  return {
      p0.x + (p1.x - p0.x) * t,
      p0.y + (p1.y - p0.y) * t,
      p0.z + (p1.z - p0.z) * t,
  };
}

__device__ int countTetraTriangles(const float values[8], const int tetra[4], const float iso_level) {
  int inside_count = 0;
  for (int i = 0; i < 4; ++i) {
    inside_count += values[tetra[i]] < iso_level ? 1 : 0;
  }

  if (inside_count == 0 || inside_count == 4) {
    return 0;
  }
  if (inside_count == 2) {
    return 2;
  }
  return 1;
}

__device__ int countCubeTriangles(const float values[8], const float iso_level) {
  int triangle_count = 0;
  for (int tetra_index = 0; tetra_index < 6; ++tetra_index) {
    triangle_count += countTetraTriangles(values, c_cube_tetrahedra[tetra_index], iso_level);
  }
  return triangle_count;
}

__device__ int emitTetraTriangles(const Vec3f positions[8],
                                  const float values[8],
                                  const int tetra[4],
                                  const float iso_level,
                                  Triangle* out_triangles) {
  int inside_vertices[4];
  int outside_vertices[4];
  int inside_count = 0;
  int outside_count = 0;

  for (int i = 0; i < 4; ++i) {
    const int vertex_index = tetra[i];
    if (values[vertex_index] < iso_level) {
      inside_vertices[inside_count++] = vertex_index;
    } else {
      outside_vertices[outside_count++] = vertex_index;
    }
  }

  if (inside_count == 0 || inside_count == 4) {
    return 0;
  }

  if (inside_count == 1) {
    const int a = inside_vertices[0];
    const int b = outside_vertices[0];
    const int c = outside_vertices[1];
    const int d = outside_vertices[2];

    out_triangles[0] = {
        interpolateVertexGpu(positions[a], positions[b], values[a], values[b], iso_level),
        interpolateVertexGpu(positions[a], positions[c], values[a], values[c], iso_level),
        interpolateVertexGpu(positions[a], positions[d], values[a], values[d], iso_level),
    };
    return 1;
  }

  if (inside_count == 3) {
    const int a = outside_vertices[0];
    const int b = inside_vertices[0];
    const int c = inside_vertices[1];
    const int d = inside_vertices[2];

    out_triangles[0] = {
        interpolateVertexGpu(positions[a], positions[b], values[a], values[b], iso_level),
        interpolateVertexGpu(positions[a], positions[d], values[a], values[d], iso_level),
        interpolateVertexGpu(positions[a], positions[c], values[a], values[c], iso_level),
    };
    return 1;
  }

  const int a = inside_vertices[0];
  const int b = inside_vertices[1];
  const int c = outside_vertices[0];
  const int d = outside_vertices[1];

  const Vec3f p0 = interpolateVertexGpu(positions[a], positions[c], values[a], values[c], iso_level);
  const Vec3f p1 = interpolateVertexGpu(positions[a], positions[d], values[a], values[d], iso_level);
  const Vec3f p2 = interpolateVertexGpu(positions[b], positions[c], values[b], values[c], iso_level);
  const Vec3f p3 = interpolateVertexGpu(positions[b], positions[d], values[b], values[d], iso_level);

  out_triangles[0] = {p0, p1, p2};
  out_triangles[1] = {p1, p3, p2};
  return 2;
}

__device__ int emitCubeTriangles(const Vec3f positions[8],
                                 const float values[8],
                                 const float iso_level,
                                 Triangle* out_triangles) {
  int emitted = 0;
  for (int tetra_index = 0; tetra_index < 6; ++tetra_index) {
    emitted += emitTetraTriangles(
        positions, values, c_cube_tetrahedra[tetra_index], iso_level, out_triangles + emitted);
  }
  return emitted;
}

__device__ bool loadCube(const VoxelHashTSDF::FlatBlockRecord* flat_blocks,
                         const MeshLookupEntry* lookup_table,
                         const BlockKey& block_key,
                         const int lx,
                         const int ly,
                         const int lz,
                         Vec3f positions[8],
                         float values[8]) {
  const int base_x = block_key.x * kCudaBlockSize + lx;
  const int base_y = block_key.y * kCudaBlockSize + ly;
  const int base_z = block_key.z * kCudaBlockSize + lz;

  for (int corner = 0; corner < 8; ++corner) {
    const int vx = base_x + c_corner_offsets[corner][0];
    const int vy = base_y + c_corner_offsets[corner][1];
    const int vz = base_z + c_corner_offsets[corner][2];

    Voxel voxel{};
    if (!fetchVoxel(flat_blocks, lookup_table, vx, vy, vz, voxel)) {
      return false;
    }

    positions[corner] = voxelCoordToWorld(vx, vy, vz);
    values[corner] = voxel.tsdf;
  }

  return true;
}

__global__ void countTrianglesKernel(const VoxelHashTSDF::FlatBlockRecord* flat_blocks,
                                     const MeshLookupEntry* lookup_table,
                                     unsigned int* total_triangles) {
  const unsigned int block_index = blockIdx.x;
  if (block_index >= c_mesh_params.num_blocks) {
    return;
  }

  const int lx = static_cast<int>(threadIdx.x);
  const int ly = static_cast<int>(threadIdx.y);
  const int lz = static_cast<int>(threadIdx.z);

  Vec3f positions[8];
  float values[8];
  const BlockKey block_key = flat_blocks[block_index].key;
  if (!loadCube(flat_blocks, lookup_table, block_key, lx, ly, lz, positions, values)) {
    return;
  }

  const int triangle_count = countCubeTriangles(values, c_mesh_params.iso_level);
  if (triangle_count > 0) {
    atomicAdd(total_triangles, static_cast<unsigned int>(triangle_count));
  }
}

__global__ void extractTrianglesKernel(const VoxelHashTSDF::FlatBlockRecord* flat_blocks,
                                       const MeshLookupEntry* lookup_table,
                                       Triangle* triangles,
                                       unsigned int* triangle_counter,
                                       const unsigned int max_triangles) {
  const unsigned int block_index = blockIdx.x;
  if (block_index >= c_mesh_params.num_blocks) {
    return;
  }

  const int lx = static_cast<int>(threadIdx.x);
  const int ly = static_cast<int>(threadIdx.y);
  const int lz = static_cast<int>(threadIdx.z);

  Vec3f positions[8];
  float values[8];
  const BlockKey block_key = flat_blocks[block_index].key;
  if (!loadCube(flat_blocks, lookup_table, block_key, lx, ly, lz, positions, values)) {
    return;
  }

  Triangle local_triangles[kMaxTrianglesPerCube];
  const int triangle_count = emitCubeTriangles(positions, values, c_mesh_params.iso_level, local_triangles);
  if (triangle_count <= 0) {
    return;
  }

  const unsigned int base = atomicAdd(triangle_counter, static_cast<unsigned int>(triangle_count));
  if (base >= max_triangles) {
    return;
  }

  const unsigned int remaining = max_triangles - base;
  const unsigned int writable =
      static_cast<unsigned int>(triangle_count) < remaining ? static_cast<unsigned int>(triangle_count) : remaining;
  for (unsigned int i = 0; i < writable; ++i) {
    triangles[base + i] = local_triangles[i];
  }
}

}  // namespace

std::vector<Triangle> extractMeshCuda(const VoxelHashTSDF& volume, const float iso_level) {
  std::vector<VoxelHashTSDF::FlatBlockRecord> flat_blocks = volume.copyObservedBlocksToFlat();
  if (flat_blocks.empty()) {
    return {};
  }

  const std::vector<MeshLookupEntry> lookup_table = buildLookupTable(flat_blocks);
  MeshExtractorWorkspace& workspace = meshExtractorWorkspace();

  workspace.flat_blocks.copyFromHost(flat_blocks.data(), flat_blocks.size());
  workspace.lookup_table.copyFromHost(lookup_table.data(), lookup_table.size());

  unsigned int zero = 0;
  workspace.triangle_count.copyFromHost(&zero, 1);

  GpuMeshParams params{};
  params.voxel_size = volume.voxelSize();
  params.iso_level = iso_level;
  params.num_blocks = static_cast<unsigned int>(flat_blocks.size());
  params.lookup_table_size = static_cast<unsigned int>(lookup_table.size());
  throwCudaError(cudaMemcpyToSymbol(c_mesh_params, &params, sizeof(GpuMeshParams)), "cudaMemcpyToSymbol");

  const dim3 block_size(kCudaBlockSize, kCudaBlockSize, kCudaBlockSize);
  const dim3 grid_size(static_cast<unsigned int>(flat_blocks.size()), 1, 1);

  countTrianglesKernel<<<grid_size, block_size>>>(
      workspace.flat_blocks.data(), workspace.lookup_table.data(), workspace.triangle_count.data());
  throwCudaError(cudaGetLastError(), "CUDA mesh count kernel launch");
  throwCudaError(cudaDeviceSynchronize(), "CUDA mesh count kernel sync");

  unsigned int triangle_count = 0;
  workspace.triangle_count.copyToHost(&triangle_count, 1);
  if (triangle_count == 0U) {
    return {};
  }

  workspace.triangles.ensureCapacity(triangle_count);
  workspace.triangle_count.copyFromHost(&zero, 1);

  extractTrianglesKernel<<<grid_size, block_size>>>(
      workspace.flat_blocks.data(),
      workspace.lookup_table.data(),
      workspace.triangles.data(),
      workspace.triangle_count.data(),
      triangle_count);
  throwCudaError(cudaGetLastError(), "CUDA mesh extract kernel launch");
  throwCudaError(cudaDeviceSynchronize(), "CUDA mesh extract kernel sync");

  unsigned int written_triangles = 0;
  workspace.triangle_count.copyToHost(&written_triangles, 1);
  if (written_triangles > triangle_count) {
    written_triangles = triangle_count;
  }

  std::vector<Triangle> triangles(static_cast<std::size_t>(written_triangles));
  workspace.triangles.copyToHost(triangles.data(), triangles.size());
  return triangles;
}

}  // namespace tsdfmc
