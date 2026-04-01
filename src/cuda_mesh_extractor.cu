#include "cuda_mesh_extractor.h"
#include "cuda_support.h"

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace tsdfmc {

namespace {

constexpr int kCudaBlockSize = 8;
constexpr int kCudaBlockVolume = kCudaBlockSize * kCudaBlockSize * kCudaBlockSize;
constexpr int kMaxTrianglesPerCube = 5;
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

__device__ __constant__ int c_edge_corners[12][2] = {
    {0, 1},
    {1, 2},
    {2, 3},
    {3, 0},
    {4, 5},
    {5, 6},
    {6, 7},
    {7, 4},
    {0, 4},
    {1, 5},
    {2, 6},
    {3, 7},
};

__device__ __constant__ int c_mc_edge_table[256];
__device__ __constant__ int c_mc_tri_table[256 * 16];

struct MeshExtractorWorkspace {
  ReusableDeviceBuffer<VoxelHashTSDF::FlatBlockRecord> flat_blocks;
  ReusableDeviceBuffer<MeshLookupEntry> lookup_table;
  ReusableDeviceBuffer<unsigned int> triangle_count;
  ReusableDeviceBuffer<Triangle> triangles;
  bool lookup_tables_initialized = false;
};

MeshExtractorWorkspace& meshExtractorWorkspace() {
  static MeshExtractorWorkspace workspace;
  return workspace;
}

void ensureDeviceLookupTablesInitialized(MeshExtractorWorkspace& workspace) {
  if (workspace.lookup_tables_initialized) {
    return;
  }

  throwCudaError(
      cudaMemcpyToSymbol(
          c_mc_edge_table,
          kMarchingCubesEdgeTable.data(),
          kMarchingCubesEdgeTable.size() * sizeof(int)),
      "cudaMemcpyToSymbol(c_mc_edge_table)");
  throwCudaError(
      cudaMemcpyToSymbol(
          c_mc_tri_table,
          kMarchingCubesTriTable.data(),
          kMarchingCubesTriTable.size() * sizeof(kMarchingCubesTriTable[0])),
      "cudaMemcpyToSymbol(c_mc_tri_table)");
  workspace.lookup_tables_initialized = true;
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

__device__ int computeCubeIndex(const float values[8], const float iso_level) {
  int cube_index = 0;
  for (int corner = 0; corner < 8; ++corner) {
    if (values[corner] < iso_level) {
      cube_index |= (1 << corner);
    }
  }
  return cube_index;
}

__device__ int countCubeTriangles(const float values[8], const float iso_level) {
  const int cube_index = computeCubeIndex(values, iso_level);
  const int edge_mask = c_mc_edge_table[cube_index];
  if (edge_mask == 0) {
    return 0;
  }

  const int tri_offset = cube_index * 16;
  int triangle_count = 0;
  for (int i = 0; i < 16; i += 3) {
    if (c_mc_tri_table[tri_offset + i] < 0) {
      break;
    }
    ++triangle_count;
  }
  return triangle_count;
}

__device__ int emitCubeTriangles(const Vec3f positions[8],
                                 const float values[8],
                                 const float iso_level,
                                 Triangle* out_triangles) {
  const int cube_index = computeCubeIndex(values, iso_level);
  const int edge_mask = c_mc_edge_table[cube_index];
  if (edge_mask == 0) {
    return 0;
  }

  Vec3f edge_vertices[12];
  for (int edge = 0; edge < 12; ++edge) {
    if ((edge_mask & (1 << edge)) == 0) {
      continue;
    }

    const int c0 = c_edge_corners[edge][0];
    const int c1 = c_edge_corners[edge][1];
    edge_vertices[edge] = interpolateVertexGpu(positions[c0], positions[c1], values[c0], values[c1], iso_level);
  }

  const int tri_offset = cube_index * 16;
  int emitted = 0;
  for (int i = 0; i < 16; i += 3) {
    const int e0 = c_mc_tri_table[tri_offset + i];
    if (e0 < 0) {
      break;
    }
    const int e1 = c_mc_tri_table[tri_offset + i + 1];
    const int e2 = c_mc_tri_table[tri_offset + i + 2];
    out_triangles[emitted++] = {edge_vertices[e0], edge_vertices[e1], edge_vertices[e2]};
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
  ensureDeviceLookupTablesInitialized(workspace);

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
