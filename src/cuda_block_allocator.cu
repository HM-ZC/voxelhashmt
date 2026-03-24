#include "cuda_block_allocator.h"
#include "cuda_support.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

namespace tsdfmc {

namespace {

constexpr int kCudaBlockSize = VoxelHashTSDF::kBlockSize;
constexpr unsigned int kAllocationThreadsPerBlock = 128U;
constexpr unsigned int kCompactThreadsPerBlock = 256U;
constexpr unsigned long long kEmptyPackedBlockKey = ~0ULL;
constexpr int kPackedCoordinateBits = 21;
constexpr int kPackedCoordinateBias = 1 << (kPackedCoordinateBits - 1);
constexpr unsigned long long kPackedCoordinateMask = (1ULL << kPackedCoordinateBits) - 1ULL;

static_assert(std::is_trivially_copyable<BlockKey>::value, "BlockKey must be trivially copyable for CUDA.");

struct GpuMat3f {
  float data[9];
};

struct GpuVec3f {
  float x;
  float y;
  float z;
};

struct GpuPose {
  GpuMat3f R;
  GpuVec3f t;
};

struct GpuIntrinsics {
  int width;
  int height;
  float fx;
  float fy;
  float cx;
  float cy;
};

struct GpuBlockAllocationParams {
  GpuPose T_wc;
  GpuIntrinsics intrinsics;
  float truncation_distance;
  float block_world_size;
  int allocation_stride;
  unsigned int sample_width;
  unsigned int sample_height;
  unsigned int hash_capacity;
  unsigned int hash_mask;
};

__constant__ GpuBlockAllocationParams c_params;

GpuPose toGpuPose(const Pose& pose) {
  GpuPose result{};
  for (std::size_t i = 0; i < pose.R.data.size(); ++i) {
    result.R.data[i] = pose.R.data[i];
  }
  result.t = {pose.t.x, pose.t.y, pose.t.z};
  return result;
}

GpuIntrinsics toGpuIntrinsics(const Intrinsics& intrinsics) {
  return {
      intrinsics.width,
      intrinsics.height,
      intrinsics.fx,
      intrinsics.fy,
      intrinsics.cx,
      intrinsics.cy,
  };
}

std::size_t nextPowerOfTwo(std::size_t value) {
  if (value <= 1U) {
    return 1U;
  }

  --value;
  for (std::size_t shift = 1; shift < sizeof(std::size_t) * 8U; shift <<= 1U) {
    value |= value >> shift;
  }
  return value + 1U;
}

BlockKey unpackBlockKey(const unsigned long long packed_key) {
  const auto unpack_coord = [](const unsigned long long packed, const int shift) {
    const int encoded =
        static_cast<int>((packed >> shift) & static_cast<unsigned long long>(kPackedCoordinateMask));
    return encoded - kPackedCoordinateBias;
  };

  return {
      unpack_coord(packed_key, 0),
      unpack_coord(packed_key, kPackedCoordinateBits),
      unpack_coord(packed_key, 2 * kPackedCoordinateBits),
  };
}

struct BlockAllocatorWorkspace {
  ReusableDeviceBuffer<float> depth;
  ReusableDeviceBuffer<unsigned long long> hash_table;
  ReusableDeviceBuffer<unsigned int> overflow_flag;
  ReusableDeviceBuffer<unsigned long long> compact_keys;
  ReusableDeviceBuffer<unsigned int> compact_count;
};

BlockAllocatorWorkspace& blockAllocatorWorkspace() {
  static BlockAllocatorWorkspace workspace;
  return workspace;
}

__device__ float vecNorm(const GpuVec3f& v) {
  return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ GpuVec3f normalizeVec(const GpuVec3f& v) {
  const float length = vecNorm(v);
  if (length <= 1e-6f) {
    return {0.0f, 0.0f, 0.0f};
  }

  const float inverse_length = 1.0f / length;
  return {v.x * inverse_length, v.y * inverse_length, v.z * inverse_length};
}

__device__ GpuVec3f transformPoint(const GpuPose& pose, const GpuVec3f& p) {
  return {
      pose.R.data[0] * p.x + pose.R.data[1] * p.y + pose.R.data[2] * p.z + pose.t.x,
      pose.R.data[3] * p.x + pose.R.data[4] * p.y + pose.R.data[5] * p.z + pose.t.y,
      pose.R.data[6] * p.x + pose.R.data[7] * p.y + pose.R.data[8] * p.z + pose.t.z,
  };
}

__device__ GpuVec3f backProject(const int u, const int v, const float depth, const GpuIntrinsics& intrinsics) {
  return {
      (static_cast<float>(u) - intrinsics.cx) * depth / intrinsics.fx,
      (static_cast<float>(v) - intrinsics.cy) * depth / intrinsics.fy,
      depth,
  };
}

__device__ unsigned int hashBlockKey(const int x, const int y, const int z) {
  unsigned int seed = 0U;
  seed ^= static_cast<unsigned int>(x * 73856093) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
  seed ^= static_cast<unsigned int>(y * 19349663) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
  seed ^= static_cast<unsigned int>(z * 83492791) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
  return seed;
}

__device__ bool packBlockKey(const int x, const int y, const int z, unsigned long long& packed_key) {
  const int min_coordinate = -kPackedCoordinateBias;
  const int max_coordinate = kPackedCoordinateBias - 1;
  if (x < min_coordinate || x > max_coordinate || y < min_coordinate || y > max_coordinate ||
      z < min_coordinate || z > max_coordinate) {
    return false;
  }

  const unsigned long long px = static_cast<unsigned long long>(x + kPackedCoordinateBias);
  const unsigned long long py = static_cast<unsigned long long>(y + kPackedCoordinateBias);
  const unsigned long long pz = static_cast<unsigned long long>(z + kPackedCoordinateBias);
  packed_key = px | (py << kPackedCoordinateBits) | (pz << (2 * kPackedCoordinateBits));
  return true;
}

__device__ void setOverflow(unsigned int* overflow_flag) {
  atomicExch(overflow_flag, 1U);
}

__device__ void insertPackedBlockKey(const unsigned long long packed_key,
                                     const int x,
                                     const int y,
                                     const int z,
                                     unsigned long long* hash_table,
                                     unsigned int* overflow_flag) {
  const unsigned int start_slot = hashBlockKey(x, y, z) & c_params.hash_mask;
  for (unsigned int probe = 0; probe < c_params.hash_capacity; ++probe) {
    const unsigned int slot = (start_slot + probe) & c_params.hash_mask;
    const unsigned long long previous = atomicCAS(&hash_table[slot], kEmptyPackedBlockKey, packed_key);
    if (previous == kEmptyPackedBlockKey || previous == packed_key) {
      return;
    }
  }

  setOverflow(overflow_flag);
}

__device__ void insertNeighborhood(const int center_x,
                                   const int center_y,
                                   const int center_z,
                                   unsigned long long* hash_table,
                                   unsigned int* overflow_flag) {
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const int x = center_x + dx;
        const int y = center_y + dy;
        const int z = center_z + dz;

        unsigned long long packed_key = 0ULL;
        if (!packBlockKey(x, y, z, packed_key)) {
          setOverflow(overflow_flag);
          return;
        }

        insertPackedBlockKey(packed_key, x, y, z, hash_table, overflow_flag);
      }
    }
  }
}

__device__ void insertWorldPointNeighborhood(const GpuVec3f& point_w,
                                             unsigned long long* hash_table,
                                             unsigned int* overflow_flag) {
  const int block_x = static_cast<int>(floorf(point_w.x / c_params.block_world_size));
  const int block_y = static_cast<int>(floorf(point_w.y / c_params.block_world_size));
  const int block_z = static_cast<int>(floorf(point_w.z / c_params.block_world_size));
  insertNeighborhood(block_x, block_y, block_z, hash_table, overflow_flag);
}

__global__ void buildActiveBlockHashKernel(const float* depth,
                                           unsigned long long* hash_table,
                                           unsigned int* overflow_flag) {
  const unsigned int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int sample_count = c_params.sample_width * c_params.sample_height;
  if (sample_index >= sample_count || atomicAdd(overflow_flag, 0U) != 0U) {
    return;
  }

  const unsigned int sample_x = sample_index % c_params.sample_width;
  const unsigned int sample_y = sample_index / c_params.sample_width;
  const int u = static_cast<int>(sample_x) * c_params.allocation_stride;
  const int v = static_cast<int>(sample_y) * c_params.allocation_stride;
  if (u >= c_params.intrinsics.width || v >= c_params.intrinsics.height) {
    return;
  }

  const float depth_value = depth[static_cast<std::size_t>(v * c_params.intrinsics.width + u)];
  if (depth_value <= 0.0f) {
    return;
  }

  const GpuVec3f point_c = backProject(u, v, depth_value, c_params.intrinsics);
  const GpuVec3f point_w = transformPoint(c_params.T_wc, point_c);
  const GpuVec3f camera_origin = c_params.T_wc.t;
  const GpuVec3f ray_w = normalizeVec(
      {point_w.x - camera_origin.x, point_w.y - camera_origin.y, point_w.z - camera_origin.z});
  const GpuVec3f front_point{
      point_w.x - ray_w.x * c_params.truncation_distance,
      point_w.y - ray_w.y * c_params.truncation_distance,
      point_w.z - ray_w.z * c_params.truncation_distance,
  };
  const GpuVec3f back_point{
      point_w.x + ray_w.x * c_params.truncation_distance,
      point_w.y + ray_w.y * c_params.truncation_distance,
      point_w.z + ray_w.z * c_params.truncation_distance,
  };

  insertWorldPointNeighborhood(point_w, hash_table, overflow_flag);
  insertWorldPointNeighborhood(front_point, hash_table, overflow_flag);
  insertWorldPointNeighborhood(back_point, hash_table, overflow_flag);
}

__global__ void compactifyActiveBlockHashKernel(const unsigned long long* hash_table,
                                                unsigned long long* compact_keys,
                                                unsigned int* compact_count) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ unsigned int local_count;
  if (threadIdx.x == 0) {
    local_count = 0U;
  }
  __syncthreads();

  int local_offset = -1;
  if (index < c_params.hash_capacity && hash_table[index] != kEmptyPackedBlockKey) {
    local_offset = static_cast<int>(atomicAdd(&local_count, 1U));
  }
  __syncthreads();

  __shared__ unsigned int global_base;
  if (threadIdx.x == 0 && local_count > 0U) {
    global_base = atomicAdd(compact_count, local_count);
  }
  __syncthreads();

  if (local_offset >= 0) {
    compact_keys[global_base + static_cast<unsigned int>(local_offset)] = hash_table[index];
  }
}

}  // namespace

std::vector<BlockKey> allocateActiveBlocksCuda(const DepthFrame& frame,
                                               const Pose& T_wc,
                                               const float voxel_size,
                                               const float truncation_distance,
                                               const int allocation_stride,
                                               const float* device_depth) {
  if (frame.depth.empty()) {
    return {};
  }

  const int stride = std::max(1, allocation_stride);
  const unsigned int sample_width = static_cast<unsigned int>((frame.intrinsics.width + stride - 1) / stride);
  const unsigned int sample_height = static_cast<unsigned int>((frame.intrinsics.height + stride - 1) / stride);
  const std::size_t sample_count = static_cast<std::size_t>(sample_width) * sample_height;
  if (sample_count == 0U) {
    return {};
  }

  const std::size_t estimated_unique_blocks = std::max<std::size_t>(2048U, sample_count * 16U);
  const std::size_t hash_capacity = nextPowerOfTwo(estimated_unique_blocks * 2U);
  if (hash_capacity > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
    throw CudaActiveBlockAllocationFallback("GPU active block allocation requested an oversized device hash table.");
  }

  BlockAllocatorWorkspace& workspace = blockAllocatorWorkspace();
  if (device_depth == nullptr) {
    workspace.depth.copyFromHost(frame.depth.data(), frame.depth.size());
    device_depth = workspace.depth.data();
  }

  workspace.hash_table.memsetBytes(0xFF, hash_capacity);
  const unsigned int zero = 0U;
  workspace.overflow_flag.copyFromHost(&zero, 1U);

  GpuBlockAllocationParams params{};
  params.T_wc = toGpuPose(T_wc);
  params.intrinsics = toGpuIntrinsics(frame.intrinsics);
  params.truncation_distance = truncation_distance;
  params.block_world_size = voxel_size * static_cast<float>(kCudaBlockSize);
  params.allocation_stride = stride;
  params.sample_width = sample_width;
  params.sample_height = sample_height;
  params.hash_capacity = static_cast<unsigned int>(hash_capacity);
  params.hash_mask = static_cast<unsigned int>(hash_capacity - 1U);

  throwCudaError(cudaMemcpyToSymbol(c_params, &params, sizeof(GpuBlockAllocationParams)), "cudaMemcpyToSymbol");

  const dim3 allocation_block_size(kAllocationThreadsPerBlock, 1, 1);
  const dim3 allocation_grid_size(
      static_cast<unsigned int>((sample_count + kAllocationThreadsPerBlock - 1U) / kAllocationThreadsPerBlock), 1, 1);
  buildActiveBlockHashKernel<<<allocation_grid_size, allocation_block_size>>>(
      device_depth, workspace.hash_table.data(), workspace.overflow_flag.data());
  throwCudaError(cudaGetLastError(), "CUDA active block allocation kernel launch");
  throwCudaError(cudaDeviceSynchronize(), "CUDA active block allocation kernel sync");

  unsigned int overflow_flag = 0U;
  workspace.overflow_flag.copyToHost(&overflow_flag, 1U);
  if (overflow_flag != 0U) {
    throw CudaActiveBlockAllocationFallback(
        "GPU active block allocation overflowed its temporary device hash table; falling back to CPU block "
        "materialization is recommended for this frame.");
  }

  workspace.compact_keys.ensureCapacity(hash_capacity);
  workspace.compact_count.copyFromHost(&zero, 1U);

  const dim3 compact_block_size(kCompactThreadsPerBlock, 1, 1);
  const dim3 compact_grid_size(
      static_cast<unsigned int>((hash_capacity + kCompactThreadsPerBlock - 1U) / kCompactThreadsPerBlock), 1, 1);
  compactifyActiveBlockHashKernel<<<compact_grid_size, compact_block_size>>>(
      workspace.hash_table.data(), workspace.compact_keys.data(), workspace.compact_count.data());
  throwCudaError(cudaGetLastError(), "CUDA active block compactify kernel launch");
  throwCudaError(cudaDeviceSynchronize(), "CUDA active block compactify kernel sync");

  unsigned int compact_count = 0U;
  workspace.compact_count.copyToHost(&compact_count, 1U);
  if (compact_count == 0U) {
    return {};
  }

  std::vector<unsigned long long> packed_keys(compact_count);
  workspace.compact_keys.copyToHost(packed_keys.data(), packed_keys.size());

  std::vector<BlockKey> block_keys;
  block_keys.reserve(packed_keys.size());
  for (const unsigned long long packed_key : packed_keys) {
    block_keys.push_back(unpackBlockKey(packed_key));
  }

  return block_keys;
}

}  // namespace tsdfmc
