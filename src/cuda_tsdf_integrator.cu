#include "cuda_block_allocator.h"
#include "cuda_support.h"
#include "cuda_tsdf_integrator.h"

#include <algorithm>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

namespace tsdfmc {

namespace {

constexpr int kCudaBlockSize = 8;
constexpr int kCudaBlockVolume = kCudaBlockSize * kCudaBlockSize * kCudaBlockSize;
constexpr float kMaxVoxelWeight = 100.0f;

static_assert(VoxelHashTSDF::kBlockSize == kCudaBlockSize, "CUDA kernel assumes 8x8x8 voxel blocks.");
static_assert(VoxelHashTSDF::kBlockVolume == kCudaBlockVolume, "CUDA kernel block volume mismatch.");
static_assert(std::is_trivially_copyable<VoxelHashTSDF::FlatBlockRecord>::value,
              "FlatBlockRecord must be trivially copyable for CUDA transfers.");

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

struct GpuIntegrationParams {
  GpuPose T_cw;
  GpuIntrinsics intrinsics;
  float voxel_size;
  float truncation_distance;
  unsigned int num_blocks;
};

__constant__ GpuIntegrationParams c_params;

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

struct IntegratorWorkspace {
  ReusableDeviceBuffer<VoxelHashTSDF::FlatBlockRecord> flat_blocks;
  ReusableDeviceBuffer<float> depth;
};

IntegratorWorkspace& integratorWorkspace() {
  static IntegratorWorkspace workspace;
  return workspace;
}

__device__ float clampGpu(const float value, const float min_value, const float max_value) {
  return fminf(max_value, fmaxf(min_value, value));
}

__device__ GpuVec3f transformPoint(const GpuPose& pose, const GpuVec3f& p) {
  return {
      pose.R.data[0] * p.x + pose.R.data[1] * p.y + pose.R.data[2] * p.z + pose.t.x,
      pose.R.data[3] * p.x + pose.R.data[4] * p.y + pose.R.data[5] * p.z + pose.t.y,
      pose.R.data[6] * p.x + pose.R.data[7] * p.y + pose.R.data[8] * p.z + pose.t.z,
  };
}

__device__ bool projectPoint(const GpuVec3f& p_c, const GpuIntrinsics& intrinsics, float& u, float& v) {
  if (p_c.z <= 1e-6f) {
    return false;
  }

  u = intrinsics.fx * p_c.x / p_c.z + intrinsics.cx;
  v = intrinsics.fy * p_c.y / p_c.z + intrinsics.cy;
  return u >= 0.0f && u <= static_cast<float>(intrinsics.width - 1) &&
         v >= 0.0f && v <= static_cast<float>(intrinsics.height - 1);
}

__device__ bool sampleDepthNearest(const float* depth, const GpuIntrinsics& intrinsics, const float u, const float v,
                                   float& depth_value) {
  const int ui = static_cast<int>(floorf(u + 0.5f));
  const int vi = static_cast<int>(floorf(v + 0.5f));
  if (ui < 0 || ui >= intrinsics.width || vi < 0 || vi >= intrinsics.height) {
    return false;
  }

  depth_value = depth[static_cast<std::size_t>(vi * intrinsics.width + ui)];
  return depth_value > 0.0f;
}

__device__ bool sampleDepthBilinear(const float* depth, const GpuIntrinsics& intrinsics, const float u, const float v,
                                    float& depth_value) {
  if (intrinsics.width <= 0 || intrinsics.height <= 0) {
    return false;
  }

  if (u < 0.0f || v < 0.0f || u > static_cast<float>(intrinsics.width - 1) ||
      v > static_cast<float>(intrinsics.height - 1)) {
    return false;
  }

  const int x0 = static_cast<int>(floorf(u));
  const int y0 = static_cast<int>(floorf(v));
  const int x1 = min(x0 + 1, intrinsics.width - 1);
  const int y1 = min(y0 + 1, intrinsics.height - 1);
  const float tx = u - static_cast<float>(x0);
  const float ty = v - static_cast<float>(y0);

  const float d00 = depth[static_cast<std::size_t>(y0 * intrinsics.width + x0)];
  const float d10 = depth[static_cast<std::size_t>(y0 * intrinsics.width + x1)];
  const float d01 = depth[static_cast<std::size_t>(y1 * intrinsics.width + x0)];
  const float d11 = depth[static_cast<std::size_t>(y1 * intrinsics.width + x1)];

  if (d00 > 0.0f && d10 > 0.0f && d01 > 0.0f && d11 > 0.0f) {
    const float top = d00 + (d10 - d00) * tx;
    const float bottom = d01 + (d11 - d01) * tx;
    depth_value = top + (bottom - top) * ty;
    return true;
  }

  return sampleDepthNearest(depth, intrinsics, u, v, depth_value);
}

__global__ void integrateActiveBlocksKernel(VoxelHashTSDF::FlatBlockRecord* flat_blocks, const float* depth) {
  const unsigned int block_index = blockIdx.x;
  if (block_index >= c_params.num_blocks) {
    return;
  }

  const int lx = static_cast<int>(threadIdx.x);
  const int ly = static_cast<int>(threadIdx.y);
  const int lz = static_cast<int>(threadIdx.z);
  const int local_index = lx + kCudaBlockSize * (ly + kCudaBlockSize * lz);

  VoxelHashTSDF::FlatBlockRecord& flat_block = flat_blocks[block_index];
  const int base_x = flat_block.key.x * kCudaBlockSize;
  const int base_y = flat_block.key.y * kCudaBlockSize;
  const int base_z = flat_block.key.z * kCudaBlockSize;

  const GpuVec3f point_w{
      (static_cast<float>(base_x + lx) + 0.5f) * c_params.voxel_size,
      (static_cast<float>(base_y + ly) + 0.5f) * c_params.voxel_size,
      (static_cast<float>(base_z + lz) + 0.5f) * c_params.voxel_size,
  };
  const GpuVec3f point_c = transformPoint(c_params.T_cw, point_w);
  if (point_c.z <= 0.0f) {
    return;
  }

  float u = 0.0f;
  float v = 0.0f;
  if (!projectPoint(point_c, c_params.intrinsics, u, v)) {
    return;
  }

  float depth_value = 0.0f;
  if (!sampleDepthBilinear(depth, c_params.intrinsics, u, v, depth_value)) {
    return;
  }

  const float signed_distance = depth_value - point_c.z;
  if (signed_distance <= -c_params.truncation_distance) {
    return;
  }

  const float tsdf = clampGpu(signed_distance / c_params.truncation_distance, -1.0f, 1.0f);
  Voxel& voxel = flat_block.voxels[local_index];
  const float new_weight = fminf(kMaxVoxelWeight, voxel.weight + 1.0f);
  voxel.tsdf = (voxel.tsdf * voxel.weight + tsdf) / new_weight;
  voxel.weight = new_weight;
}

}  // namespace

CudaBackendStatus queryCudaBackendStatus() {
  int device_count = 0;
  const cudaError_t error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess) {
    return {false, cudaGetErrorString(error)};
  }
  if (device_count <= 0) {
    return {false, "no CUDA device detected"};
  }
  return {true, std::to_string(device_count) + " CUDA device(s) detected"};
}

void integrateDepthFrameCuda(VoxelHashTSDF& volume,
                             const DepthFrame& frame,
                             const Pose& T_wc,
                             const int allocation_stride) {
  std::vector<BlockKey> active_blocks;
  IntegratorWorkspace& workspace = integratorWorkspace();
  workspace.depth.copyFromHost(frame.depth.data(), frame.depth.size());
  try {
    active_blocks = allocateActiveBlocksCuda(
        frame, T_wc, volume.voxelSize(), volume.truncationDistance(), allocation_stride, workspace.depth.data());
    volume.ensureBlocksExist(active_blocks);
  } catch (const CudaActiveBlockAllocationFallback&) {
    active_blocks = volume.prepareFrameIntegration(frame, T_wc, allocation_stride);
  }
  if (active_blocks.empty()) {
    return;
  }

  std::vector<VoxelHashTSDF::FlatBlockRecord> flat_blocks = volume.copyBlocksToFlat(active_blocks);
  if (flat_blocks.empty() || frame.depth.empty()) {
    return;
  }

  workspace.flat_blocks.copyFromHost(flat_blocks.data(), flat_blocks.size());

  GpuIntegrationParams params{};
  params.T_cw = toGpuPose(T_wc.inverse());
  params.intrinsics = toGpuIntrinsics(frame.intrinsics);
  params.voxel_size = volume.voxelSize();
  params.truncation_distance = volume.truncationDistance();
  params.num_blocks = static_cast<unsigned int>(flat_blocks.size());

  throwCudaError(cudaMemcpyToSymbol(c_params, &params, sizeof(GpuIntegrationParams)), "cudaMemcpyToSymbol");

  const dim3 block_size(kCudaBlockSize, kCudaBlockSize, kCudaBlockSize);
  const dim3 grid_size(static_cast<unsigned int>(flat_blocks.size()), 1, 1);
  integrateActiveBlocksKernel<<<grid_size, block_size>>>(workspace.flat_blocks.data(), workspace.depth.data());
  throwCudaError(cudaGetLastError(), "CUDA integrate kernel launch");
  throwCudaError(cudaDeviceSynchronize(), "CUDA integrate kernel sync");

  workspace.flat_blocks.copyToHost(flat_blocks.data(), flat_blocks.size());
  volume.applyFlatBlocks(flat_blocks);
}

}  // namespace tsdfmc
