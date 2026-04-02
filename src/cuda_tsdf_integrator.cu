#include "cuda_block_allocator.h"
#include "cuda_support.h"
#include "cuda_tsdf_integrator.h"

#include <algorithm>
#include <unordered_map>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

namespace tsdfmc {

namespace {

constexpr int kCudaBlockSize = 8;
constexpr int kCudaBlockVolume = kCudaBlockSize * kCudaBlockSize * kCudaBlockSize;
constexpr float kMaxVoxelWeight = 100.0f;
constexpr float kDepthSigma = 1.5f;
constexpr float kHuberDelta = 0.35f;
constexpr float kMinObservationWeight = 0.05f;
constexpr float kMinViewAngleWeight = 0.15f;
constexpr int kNormalBilateralRadius = 1;
constexpr float kNormalBilateralSigmaSpatial = 1.0f;
constexpr float kNormalBilateralSigmaRange = 0.0012f;
constexpr float kNormalBilateralSigmaAngular = 0.35f;
constexpr float kDepthDiscontinuityBase = 0.0012f;
constexpr float kDepthDiscontinuityRelative = 0.0025f;
constexpr float kLayerMergeVoxelScale = 0.25f;
constexpr float kLayerNormalConsistencyCosine = 0.75f;
constexpr int kDepthBilateralRadius = 2;
constexpr float kDepthBilateralSigmaSpatial = 1.5f;
constexpr float kDepthBilateralSigmaRange = 0.0012f;
constexpr float kDepthBilateralSigmaAngular = 0.35f;
constexpr float kMinSurfaceConsistencyCosine = 0.6f;
constexpr unsigned int kFilterThreadsX = 16U;
constexpr unsigned int kFilterThreadsY = 16U;

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
  GpuPose T_wc;
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
  ReusableDeviceBuffer<float> filtered_depth;
  ReusableDeviceBuffer<GpuVec3f> normals;
  ReusableDeviceBuffer<unsigned char> normal_valid;
  ReusableDeviceBuffer<int> active_block_indices;
  ReusableDeviceBuffer<unsigned long long> observed_voxel_count;
};

IntegratorWorkspace& integratorWorkspace() {
  static IntegratorWorkspace workspace;
  return workspace;
}

struct GpuVolumeCache {
  const VoxelHashTSDF* attached_volume = nullptr;
  ReusableDeviceBuffer<VoxelHashTSDF::FlatBlockRecord> flat_blocks;
  std::vector<BlockKey> block_keys;
  std::unordered_map<BlockKey, int, BlockKeyHash> block_to_index;
};

GpuVolumeCache& gpuVolumeCache() {
  static GpuVolumeCache cache;
  return cache;
}

void resetGpuVolumeCache(GpuVolumeCache& cache, const VoxelHashTSDF& volume) {
  cache.attached_volume = &volume;
  cache.block_keys.clear();
  cache.block_to_index.clear();
}

std::vector<int> ensureGpuVolumeCacheBlocks(const std::vector<BlockKey>& active_blocks, GpuVolumeCache& cache) {
  std::vector<int> active_indices;
  active_indices.reserve(active_blocks.size());

  std::vector<VoxelHashTSDF::FlatBlockRecord> new_records;
  for (const BlockKey& key : active_blocks) {
    const auto it = cache.block_to_index.find(key);
    if (it != cache.block_to_index.end()) {
      active_indices.push_back(it->second);
      continue;
    }

    const int new_index = static_cast<int>(cache.block_keys.size());
    cache.block_keys.push_back(key);
    cache.block_to_index.emplace(key, new_index);
    active_indices.push_back(new_index);

    VoxelHashTSDF::FlatBlockRecord record{};
    record.key = key;
    new_records.push_back(record);
  }

  if (!new_records.empty()) {
    const std::size_t old_count = cache.block_keys.size() - new_records.size();
    cache.flat_blocks.ensureCapacity(cache.block_keys.size());
    throwCudaError(
        cudaMemcpy(
            cache.flat_blocks.data() + old_count,
            new_records.data(),
            new_records.size() * sizeof(VoxelHashTSDF::FlatBlockRecord),
            cudaMemcpyHostToDevice),
        "cudaMemcpy H2D append flat blocks");
  }

  return active_indices;
}

__device__ float clampGpu(const float value, const float min_value, const float max_value) {
  return fminf(max_value, fmaxf(min_value, value));
}

__device__ float gaussianWeightGpu(const float squared_value, const float inv_two_sigma_sq) {
  return expf(-squared_value * inv_two_sigma_sq);
}

__device__ float depthDiscontinuityThresholdGpu(const float reference_depth) {
  return fmaxf(kDepthDiscontinuityBase, kDepthDiscontinuityRelative * fmaxf(reference_depth, 0.0f));
}

__device__ GpuVec3f subVec(const GpuVec3f& a, const GpuVec3f& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ float dotVec(const GpuVec3f& a, const GpuVec3f& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ GpuVec3f crossVec(const GpuVec3f& a, const GpuVec3f& b) {
  return {
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x,
  };
}

__device__ float normVec(const GpuVec3f& v) {
  return sqrtf(dotVec(v, v));
}

__device__ GpuVec3f normalizeVec(const GpuVec3f& v) {
  const float length = normVec(v);
  if (length <= 1e-6f) {
    return {0.0f, 0.0f, 0.0f};
  }

  const float inv = 1.0f / length;
  return {v.x * inv, v.y * inv, v.z * inv};
}

__device__ GpuVec3f mulVec(const GpuVec3f& v, const float scale) {
  return {v.x * scale, v.y * scale, v.z * scale};
}

__device__ GpuVec3f addVec(const GpuVec3f& a, const GpuVec3f& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ GpuVec3f orientNormalTowardCameraGpu(const GpuVec3f& normal, const GpuVec3f& point_c) {
  if (normVec(normal) <= 1e-6f || normVec(point_c) <= 1e-6f) {
    return normal;
  }

  const GpuVec3f view_ray = normalizeVec(point_c);
  return dotVec(normal, view_ray) > 0.0f ? mulVec(normal, -1.0f) : normal;
}

__device__ int dominantDirectionIndexGpu(const GpuVec3f& normal) {
  const float nx = fabsf(normal.x);
  const float ny = fabsf(normal.y);
  const float nz = fabsf(normal.z);
  if (nx <= 1e-6f && ny <= 1e-6f && nz <= 1e-6f) {
    return -1;
  }

  if (nx >= ny && nx >= nz) {
    return normal.x >= 0.0f ? 0 : 1;
  }
  if (ny >= nx && ny >= nz) {
    return normal.y >= 0.0f ? 2 : 3;
  }
  return normal.z >= 0.0f ? 4 : 5;
}

__device__ float layerMergeToleranceGpu() {
  return fmaxf(1e-6f, c_params.voxel_size * kLayerMergeVoxelScale);
}

__device__ float signNotZeroGpu(const float value) {
  return value >= 0.0f ? 1.0f : -1.0f;
}

__device__ unsigned int quantizeOctahedralComponentGpu(const float value) {
  const float clamped = fminf(1.0f, fmaxf(0.0f, value * 0.5f + 0.5f));
  return static_cast<unsigned int>(floorf(clamped * 65535.0f + 0.5f));
}

__device__ float dequantizeOctahedralComponentGpu(const unsigned int packed_component) {
  return (static_cast<float>(packed_component & 0xffffU) / 65535.0f) * 2.0f - 1.0f;
}

__device__ unsigned int packNormalOctahedralGpu(const GpuVec3f& normal_raw) {
  const GpuVec3f normal = normalizeVec(normal_raw);
  const float abs_sum = fabsf(normal.x) + fabsf(normal.y) + fabsf(normal.z);
  if (abs_sum <= 1e-6f) {
    return 0U;
  }

  GpuVec3f projected{normal.x / abs_sum, normal.y / abs_sum, normal.z / abs_sum};
  if (projected.z < 0.0f) {
    const float px = (1.0f - fabsf(projected.y)) * signNotZeroGpu(projected.x);
    const float py = (1.0f - fabsf(projected.x)) * signNotZeroGpu(projected.y);
    projected.x = px;
    projected.y = py;
  }

  return quantizeOctahedralComponentGpu(projected.x) | (quantizeOctahedralComponentGpu(projected.y) << 16U);
}

__device__ GpuVec3f unpackNormalOctahedralGpu(const unsigned int packed_normal) {
  GpuVec3f normal{
      dequantizeOctahedralComponentGpu(packed_normal),
      dequantizeOctahedralComponentGpu(packed_normal >> 16U),
      0.0f,
  };
  normal.z = 1.0f - fabsf(normal.x) - fabsf(normal.y);
  if (normal.z < 0.0f) {
    const float px = (1.0f - fabsf(normal.y)) * signNotZeroGpu(normal.x);
    const float py = (1.0f - fabsf(normal.x)) * signNotZeroGpu(normal.y);
    normal.x = px;
    normal.y = py;
  }
  return normalizeVec(normal);
}

__device__ float coordinateAlongDirectionGpu(const GpuVec3f& point_w, const int direction_index) {
  if (direction_index == 0) {
    return point_w.x;
  }
  if (direction_index == 1) {
    return -point_w.x;
  }
  if (direction_index == 2) {
    return point_w.y;
  }
  if (direction_index == 3) {
    return -point_w.y;
  }
  if (direction_index == 4) {
    return point_w.z;
  }
  return -point_w.z;
}

__device__ bool isLayerObservedGpu(const DirectionalTsdfLayer& layer) {
  return layer.weight > 0.0f;
}

__device__ void sortDirectionalLayersGpu(Voxel& voxel, const int direction_index) {
  for (int pass = 0; pass < Voxel::kLayersPerDirection - 1; ++pass) {
    for (int i = 0; i < Voxel::kLayersPerDirection - pass - 1; ++i) {
      DirectionalTsdfLayer& left = voxel.layers[direction_index][i];
      DirectionalTsdfLayer& right = voxel.layers[direction_index][i + 1];
      const bool left_valid = isLayerObservedGpu(left);
      const bool right_valid = isLayerObservedGpu(right);
      const bool should_swap =
          (!left_valid && right_valid) ||
          (left_valid && right_valid && left.surface_coord > right.surface_coord);
      if (should_swap) {
        const DirectionalTsdfLayer temp = left;
        left = right;
        right = temp;
      }
    }
  }
}

__device__ int findLayerSlotForObservationGpu(const Voxel& voxel,
                                              const int direction_index,
                                              const float surface_coord,
                                              const GpuVec3f& surface_normal,
                                              const float merge_tolerance,
                                              const float min_normal_cosine) {
  int empty_index = -1;
  int best_index = -1;
  float best_delta = 1e30f;
  for (int layer_index = 0; layer_index < Voxel::kLayersPerDirection; ++layer_index) {
    const DirectionalTsdfLayer& layer = voxel.layers[direction_index][layer_index];
    if (!isLayerObservedGpu(layer)) {
      if (empty_index < 0) {
        empty_index = layer_index;
      }
      continue;
    }

    const GpuVec3f layer_normal = unpackNormalOctahedralGpu(layer.packed_normal);
    if (dotVec(layer_normal, surface_normal) < min_normal_cosine) {
      continue;
    }

    const float delta = fabsf(layer.surface_coord - surface_coord);
    if (delta < best_delta) {
      best_delta = delta;
      best_index = layer_index;
    }
  }

  if (best_index >= 0 && best_delta <= merge_tolerance) {
    return best_index;
  }
  return empty_index;
}

__device__ void integrateDirectionalSampleGpu(Voxel& voxel,
                                              const int direction_index,
                                              const float surface_coord,
                                              const GpuVec3f& surface_point_w,
                                              const GpuVec3f& surface_normal_w,
                                              const float observation_weight) {
  if (direction_index < 0 || direction_index >= Voxel::kDirectionalBins || observation_weight <= 0.0f) {
    return;
  }

  const GpuVec3f normal = normalizeVec(surface_normal_w);
  if (normVec(normal) <= 1e-6f) {
    return;
  }

  const float plane_offset = dotVec(normal, surface_point_w);
  const int layer_index = findLayerSlotForObservationGpu(
      voxel,
      direction_index,
      surface_coord,
      normal,
      layerMergeToleranceGpu(),
      kLayerNormalConsistencyCosine);
  if (layer_index < 0) {
    return;
  }

  DirectionalTsdfLayer& layer = voxel.layers[direction_index][layer_index];
  const float previous_weight = layer.weight;
  const float effective_weight = fminf(observation_weight, fmaxf(0.0f, kMaxVoxelWeight - previous_weight));
  if (effective_weight <= 0.0f) {
    return;
  }

  const float new_weight = previous_weight + effective_weight;
  GpuVec3f blended_normal = normal;
  layer.surface_coord =
      previous_weight > 0.0f ? ((layer.surface_coord * previous_weight + surface_coord * effective_weight) / new_weight)
                             : surface_coord;
  layer.plane_offset =
      previous_weight > 0.0f ? ((layer.plane_offset * previous_weight + plane_offset * effective_weight) / new_weight)
                             : plane_offset;
  if (previous_weight > 0.0f) {
    const GpuVec3f previous_normal = unpackNormalOctahedralGpu(layer.packed_normal);
    blended_normal = normalizeVec(
        mulVec(addVec(mulVec(previous_normal, previous_weight), mulVec(normal, effective_weight)), 1.0f / new_weight));
    if (normVec(blended_normal) <= 1e-6f) {
      blended_normal = normal;
    }
  }
  layer.weight = new_weight;
  layer.packed_normal = packNormalOctahedralGpu(blended_normal);
  sortDirectionalLayersGpu(voxel, direction_index);
}

__device__ float computeObservationWeightGpu(const float depth,
                                             const float signed_distance,
                                             const float truncation_distance,
                                             const float view_angle_weight = 1.0f) {
  if (depth <= 0.0f || truncation_distance <= 1e-6f) {
    return 0.0f;
  }

  const float depth_ratio = depth / kDepthSigma;
  const float depth_weight = 1.0f / (1.0f + depth_ratio * depth_ratio);

  const float normalized_residual = fabsf(signed_distance) / truncation_distance;
  float robust_weight = 1.0f;
  if (normalized_residual > kHuberDelta) {
    robust_weight = kHuberDelta / normalized_residual;
  }

  const float band_weight = 1.0f - 0.5f * clampGpu(normalized_residual, 0.0f, 1.0f);
  const float base_weight = fmaxf(kMinObservationWeight, depth_weight * robust_weight * band_weight);
  const float angle_weight = clampGpu(view_angle_weight, kMinViewAngleWeight, 1.0f);
  return base_weight * angle_weight;
}

__device__ GpuVec3f transformPoint(const GpuPose& pose, const GpuVec3f& p) {
  return {
      pose.R.data[0] * p.x + pose.R.data[1] * p.y + pose.R.data[2] * p.z + pose.t.x,
      pose.R.data[3] * p.x + pose.R.data[4] * p.y + pose.R.data[5] * p.z + pose.t.y,
      pose.R.data[6] * p.x + pose.R.data[7] * p.y + pose.R.data[8] * p.z + pose.t.z,
  };
}

__device__ GpuVec3f transformVector(const GpuPose& pose, const GpuVec3f& v) {
  return {
      pose.R.data[0] * v.x + pose.R.data[1] * v.y + pose.R.data[2] * v.z,
      pose.R.data[3] * v.x + pose.R.data[4] * v.y + pose.R.data[5] * v.z,
      pose.R.data[6] * v.x + pose.R.data[7] * v.y + pose.R.data[8] * v.z,
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

__device__ bool sampleDepthPixel(const float* depth,
                                 const GpuIntrinsics& intrinsics,
                                 const int u,
                                 const int v,
                                 float& depth_value) {
  if (u < 0 || u >= intrinsics.width || v < 0 || v >= intrinsics.height) {
    return false;
  }

  depth_value = depth[static_cast<std::size_t>(v * intrinsics.width + u)];
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
    float reference_depth = 0.0f;
    if (!sampleDepthNearest(depth, intrinsics, u, v, reference_depth)) {
      reference_depth = d00;
    }

    const float threshold = depthDiscontinuityThresholdGpu(reference_depth);
    const float min_depth = fminf(fminf(d00, d10), fminf(d01, d11));
    const float max_depth = fmaxf(fmaxf(d00, d10), fmaxf(d01, d11));
    const float top = d00 + (d10 - d00) * tx;
    const float bottom = d01 + (d11 - d01) * tx;
    if (max_depth - min_depth <= threshold) {
      depth_value = top + (bottom - top) * ty;
      return true;
    }

    const float bilinear_weights[4] = {
        (1.0f - tx) * (1.0f - ty),
        tx * (1.0f - ty),
        (1.0f - tx) * ty,
        tx * ty,
    };
    const float neighbor_depths[4] = {d00, d10, d01, d11};

    float weighted_depth_sum = 0.0f;
    float weight_sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
      if (fabsf(neighbor_depths[i] - reference_depth) > threshold) {
        continue;
      }
      weighted_depth_sum += bilinear_weights[i] * neighbor_depths[i];
      weight_sum += bilinear_weights[i];
    }

    if (weight_sum > 1e-6f) {
      depth_value = weighted_depth_sum / weight_sum;
      return true;
    }
  }

  return sampleDepthNearest(depth, intrinsics, u, v, depth_value);
}

__device__ GpuVec3f backProjectContinuous(const float u,
                                          const float v,
                                          const float depth,
                                          const GpuIntrinsics& intrinsics) {
  return {
      (u - intrinsics.cx) * depth / intrinsics.fx,
      (v - intrinsics.cy) * depth / intrinsics.fy,
      depth,
  };
}

__device__ bool estimateSurfaceNormalAtPixel(const float* depth,
                                             const GpuIntrinsics& intrinsics,
                                             const int u,
                                             const int v,
                                             GpuVec3f& normal) {
  if (u <= 0 || v <= 0 || u + 1 >= intrinsics.width || v + 1 >= intrinsics.height) {
    return false;
  }

  float center_depth = 0.0f;
  float depth_left = 0.0f;
  float depth_right = 0.0f;
  float depth_up = 0.0f;
  float depth_down = 0.0f;
  if (!sampleDepthPixel(depth, intrinsics, u, v, center_depth) ||
      !sampleDepthPixel(depth, intrinsics, u - 1, v, depth_left) ||
      !sampleDepthPixel(depth, intrinsics, u + 1, v, depth_right) ||
      !sampleDepthPixel(depth, intrinsics, u, v - 1, depth_up) ||
      !sampleDepthPixel(depth, intrinsics, u, v + 1, depth_down)) {
    return false;
  }

  const float threshold = depthDiscontinuityThresholdGpu(center_depth);
  if (fabsf(depth_left - center_depth) > threshold ||
      fabsf(depth_right - center_depth) > threshold ||
      fabsf(depth_up - center_depth) > threshold ||
      fabsf(depth_down - center_depth) > threshold) {
    return false;
  }

  const GpuVec3f p_left = backProjectContinuous(static_cast<float>(u - 1), static_cast<float>(v), depth_left, intrinsics);
  const GpuVec3f p_right = backProjectContinuous(static_cast<float>(u + 1), static_cast<float>(v), depth_right, intrinsics);
  const GpuVec3f p_up = backProjectContinuous(static_cast<float>(u), static_cast<float>(v - 1), depth_up, intrinsics);
  const GpuVec3f p_down = backProjectContinuous(static_cast<float>(u), static_cast<float>(v + 1), depth_down, intrinsics);

  const GpuVec3f dx = subVec(p_right, p_left);
  const GpuVec3f dy = subVec(p_down, p_up);
  normal = normalizeVec(crossVec(dy, dx));
  if (normVec(normal) <= 1e-6f) {
    return false;
  }

  const GpuVec3f center_point =
      backProjectContinuous(static_cast<float>(u), static_cast<float>(v), center_depth, intrinsics);
  normal = orientNormalTowardCameraGpu(normal, center_point);
  return normVec(normal) > 1e-6f;
}

__global__ void estimateDepthNormalsKernel(const float* depth,
                                           GpuVec3f* normals,
                                           unsigned char* normal_valid) {
  const int u = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int v = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (u >= c_params.intrinsics.width || v >= c_params.intrinsics.height) {
    return;
  }

  const std::size_t index = static_cast<std::size_t>(v * c_params.intrinsics.width + u);
  GpuVec3f normal{0.0f, 0.0f, 0.0f};
  if (estimateSurfaceNormalAtPixel(depth, c_params.intrinsics, u, v, normal)) {
    normals[index] = normal;
    normal_valid[index] = 1U;
    return;
  }

  normals[index] = {0.0f, 0.0f, 0.0f};
  normal_valid[index] = 0U;
}

__global__ void bilateralFilterDepthKernel(const float* input_depth,
                                           const GpuVec3f* normals,
                                           const unsigned char* normal_valid,
                                           float* filtered_depth) {
  const int u = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int v = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (u >= c_params.intrinsics.width || v >= c_params.intrinsics.height) {
    return;
  }

  const std::size_t center_index = static_cast<std::size_t>(v * c_params.intrinsics.width + u);
  const float center_depth = input_depth[center_index];
  if (center_depth <= 0.0f) {
    filtered_depth[center_index] = 0.0f;
    return;
  }

  constexpr float kEpsilon = 1e-6f;
  constexpr float kInvTwoSpatialSigmaSq =
      1.0f / (2.0f * kDepthBilateralSigmaSpatial * kDepthBilateralSigmaSpatial);
  constexpr float kInvTwoRangeSigmaSq =
      1.0f / (2.0f * kDepthBilateralSigmaRange * kDepthBilateralSigmaRange);
  constexpr float kInvTwoAngularSigmaSq =
      1.0f / (2.0f * kDepthBilateralSigmaAngular * kDepthBilateralSigmaAngular);

  const bool has_center_normal = normal_valid[center_index] != 0U;
  const GpuVec3f center_normal = has_center_normal ? normals[center_index] : GpuVec3f{0.0f, 0.0f, 0.0f};
  const float discontinuity_threshold = depthDiscontinuityThresholdGpu(center_depth);

  float weighted_depth_sum = 0.0f;
  float weight_sum = 0.0f;
  for (int dv = -kDepthBilateralRadius; dv <= kDepthBilateralRadius; ++dv) {
    const int sv = v + dv;
    if (sv < 0 || sv >= c_params.intrinsics.height) {
      continue;
    }

    for (int du = -kDepthBilateralRadius; du <= kDepthBilateralRadius; ++du) {
      const int su = u + du;
      if (su < 0 || su >= c_params.intrinsics.width) {
        continue;
      }

      const std::size_t sample_index = static_cast<std::size_t>(sv * c_params.intrinsics.width + su);
      const float sample_depth = input_depth[sample_index];
      if (sample_depth <= 0.0f || fabsf(sample_depth - center_depth) > discontinuity_threshold) {
        continue;
      }

      const float spatial_squared = static_cast<float>(du * du + dv * dv);
      const float depth_delta = sample_depth - center_depth;
      const float range_squared = depth_delta * depth_delta;
      float weight =
          gaussianWeightGpu(spatial_squared, kInvTwoSpatialSigmaSq) *
          gaussianWeightGpu(range_squared, kInvTwoRangeSigmaSq);
      if (has_center_normal) {
        if (normal_valid[sample_index] == 0U) {
          continue;
        }

        const float cosine = clampGpu(dotVec(center_normal, normals[sample_index]), -1.0f, 1.0f);
        if (cosine < kMinSurfaceConsistencyCosine) {
          continue;
        }

        const float angular_delta = 1.0f - cosine;
        weight *= gaussianWeightGpu(angular_delta * angular_delta, kInvTwoAngularSigmaSq);
      }

      weighted_depth_sum += weight * sample_depth;
      weight_sum += weight;
    }
  }

  filtered_depth[center_index] = weight_sum > kEpsilon ? (weighted_depth_sum / weight_sum) : center_depth;
}

__device__ bool estimateSurfaceNormalBaseCamera(const float* depth,
                                                const GpuIntrinsics& intrinsics,
                                                const float u,
                                                const float v,
                                                GpuVec3f& normal) {
  float depth_left = 0.0f;
  float depth_right = 0.0f;
  float depth_up = 0.0f;
  float depth_down = 0.0f;
  if (!sampleDepthBilinear(depth, intrinsics, u - 1.0f, v, depth_left) ||
      !sampleDepthBilinear(depth, intrinsics, u + 1.0f, v, depth_right) ||
      !sampleDepthBilinear(depth, intrinsics, u, v - 1.0f, depth_up) ||
      !sampleDepthBilinear(depth, intrinsics, u, v + 1.0f, depth_down)) {
    return false;
  }

  const GpuVec3f p_left = backProjectContinuous(u - 1.0f, v, depth_left, intrinsics);
  const GpuVec3f p_right = backProjectContinuous(u + 1.0f, v, depth_right, intrinsics);
  const GpuVec3f p_up = backProjectContinuous(u, v - 1.0f, depth_up, intrinsics);
  const GpuVec3f p_down = backProjectContinuous(u, v + 1.0f, depth_down, intrinsics);

  const GpuVec3f dx = subVec(p_right, p_left);
  const GpuVec3f dy = subVec(p_down, p_up);
  normal = normalizeVec(crossVec(dy, dx));
  return normVec(normal) > 1e-6f;
}

__device__ bool estimateSurfaceNormalCamera(const float* depth,
                                            const GpuIntrinsics& intrinsics,
                                            const float u,
                                            const float v,
                                            GpuVec3f& normal) {
  GpuVec3f center_normal{};
  if (!estimateSurfaceNormalBaseCamera(depth, intrinsics, u, v, center_normal)) {
    return false;
  }

  float center_depth = 0.0f;
  if (!sampleDepthBilinear(depth, intrinsics, u, v, center_depth)) {
    return false;
  }
  const GpuVec3f center_point = backProjectContinuous(u, v, center_depth, intrinsics);
  center_normal = orientNormalTowardCameraGpu(center_normal, center_point);

  constexpr float kEpsilon = 1e-6f;
  constexpr float kInvTwoSpatialSigmaSq =
      1.0f / (2.0f * kNormalBilateralSigmaSpatial * kNormalBilateralSigmaSpatial);
  constexpr float kInvTwoRangeSigmaSq =
      1.0f / (2.0f * kNormalBilateralSigmaRange * kNormalBilateralSigmaRange);
  constexpr float kInvTwoAngularSigmaSq =
      1.0f / (2.0f * kNormalBilateralSigmaAngular * kNormalBilateralSigmaAngular);

  GpuVec3f weighted_normal_sum{0.0f, 0.0f, 0.0f};
  float weight_sum = 0.0f;
  for (int dv = -kNormalBilateralRadius; dv <= kNormalBilateralRadius; ++dv) {
    for (int du = -kNormalBilateralRadius; du <= kNormalBilateralRadius; ++du) {
      const float sample_u = u + static_cast<float>(du);
      const float sample_v = v + static_cast<float>(dv);

      GpuVec3f sample_normal{};
      if (!estimateSurfaceNormalBaseCamera(depth, intrinsics, sample_u, sample_v, sample_normal)) {
        continue;
      }

      float sample_depth = 0.0f;
      if (!sampleDepthBilinear(depth, intrinsics, sample_u, sample_v, sample_depth)) {
        continue;
      }
      const GpuVec3f sample_point = backProjectContinuous(sample_u, sample_v, sample_depth, intrinsics);
      sample_normal = orientNormalTowardCameraGpu(sample_normal, sample_point);

      const float spatial_squared = static_cast<float>(du * du + dv * dv);
      const float depth_delta = sample_depth - center_depth;
      const float range_squared = depth_delta * depth_delta;
      const float cosine = clampGpu(dotVec(center_normal, sample_normal), -1.0f, 1.0f);
      const float angular_delta = 1.0f - cosine;
      const float angular_squared = angular_delta * angular_delta;
      const float weight =
          gaussianWeightGpu(spatial_squared, kInvTwoSpatialSigmaSq) *
          gaussianWeightGpu(range_squared, kInvTwoRangeSigmaSq) *
          gaussianWeightGpu(angular_squared, kInvTwoAngularSigmaSq);

      weighted_normal_sum.x += sample_normal.x * weight;
      weighted_normal_sum.y += sample_normal.y * weight;
      weighted_normal_sum.z += sample_normal.z * weight;
      weight_sum += weight;
    }
  }

  if (weight_sum <= kEpsilon) {
    normal = center_normal;
    return true;
  }

  const float inv_weight_sum = 1.0f / weight_sum;
  normal = normalizeVec({
      weighted_normal_sum.x * inv_weight_sum,
      weighted_normal_sum.y * inv_weight_sum,
      weighted_normal_sum.z * inv_weight_sum,
  });
  normal = orientNormalTowardCameraGpu(normal, center_point);
  if (normVec(normal) <= 1e-6f) {
    normal = center_normal;
  }
  return normVec(normal) > 1e-6f;
}

__global__ void integrateActiveBlocksKernel(VoxelHashTSDF::FlatBlockRecord* flat_blocks,
                                            const int* active_block_indices,
                                            const float* depth) {
  const unsigned int active_block_index = blockIdx.x;
  if (active_block_index >= c_params.num_blocks) {
    return;
  }
  const unsigned int block_index = static_cast<unsigned int>(active_block_indices[active_block_index]);

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

  const GpuVec3f view_ray_c = normalizeVec(point_c);
  float view_angle_weight = 1.0f;
  GpuVec3f normal_c = mulVec(view_ray_c, -1.0f);
  if (estimateSurfaceNormalCamera(depth, c_params.intrinsics, u, v, normal_c)) {
    view_angle_weight = fabsf(dotVec(normal_c, view_ray_c));
  }

  const float observation_weight =
      computeObservationWeightGpu(depth_value, signed_distance, c_params.truncation_distance, view_angle_weight);
  if (observation_weight <= 0.0f) {
    return;
  }

  const GpuVec3f surface_point_c = backProjectContinuous(u, v, depth_value, c_params.intrinsics);
  const GpuVec3f surface_point_w = transformPoint(c_params.T_wc, surface_point_c);
  const GpuVec3f normal_w = normalizeVec(transformVector(c_params.T_wc, normal_c));
  const int direction_index = dominantDirectionIndexGpu(normal_w);
  if (direction_index < 0 || direction_index >= Voxel::kDirectionalBins) {
    return;
  }
  const float surface_coord = coordinateAlongDirectionGpu(surface_point_w, direction_index);

  Voxel& voxel = flat_block.voxels[local_index];
  integrateDirectionalSampleGpu(voxel, direction_index, surface_coord, surface_point_w, normal_w, observation_weight);
}

__device__ bool hasAnyObservationGpu(const Voxel& voxel) {
  for (int direction_index = 0; direction_index < Voxel::kDirectionalBins; ++direction_index) {
    for (int layer_index = 0; layer_index < Voxel::kLayersPerDirection; ++layer_index) {
      if (voxel.layers[direction_index][layer_index].weight > 0.0f) {
        return true;
      }
    }
  }
  return false;
}

__global__ void countObservedVoxelsKernel(const VoxelHashTSDF::FlatBlockRecord* flat_blocks,
                                          const unsigned int num_blocks,
                                          unsigned long long* total_count) {
  const unsigned int voxel_index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total_voxels = num_blocks * static_cast<unsigned int>(VoxelHashTSDF::kBlockVolume);
  if (voxel_index >= total_voxels) {
    return;
  }

  const unsigned int block_index = voxel_index / static_cast<unsigned int>(VoxelHashTSDF::kBlockVolume);
  const unsigned int local_index = voxel_index % static_cast<unsigned int>(VoxelHashTSDF::kBlockVolume);
  if (hasAnyObservationGpu(flat_blocks[block_index].voxels[local_index])) {
    atomicAdd(total_count, 1ULL);
  }
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
  std::vector<int> active_block_indices;
  IntegratorWorkspace& workspace = integratorWorkspace();
  GpuVolumeCache& volume_cache = gpuVolumeCache();
  if (volume_cache.attached_volume != &volume) {
    resetGpuVolumeCache(volume_cache, volume);
  }

  workspace.depth.copyFromHost(frame.depth.data(), frame.depth.size());
  workspace.filtered_depth.ensureCapacity(frame.depth.size());
  workspace.normals.ensureCapacity(frame.depth.size());
  workspace.normal_valid.ensureCapacity(frame.depth.size());

  GpuIntegrationParams params{};
  params.T_cw = toGpuPose(T_wc.inverse());
  params.T_wc = toGpuPose(T_wc);
  params.intrinsics = toGpuIntrinsics(frame.intrinsics);
  params.voxel_size = volume.voxelSize();
  params.truncation_distance = volume.truncationDistance();
  throwCudaError(cudaMemcpyToSymbol(c_params, &params, sizeof(GpuIntegrationParams)), "cudaMemcpyToSymbol");

  const dim3 filter_block_size(kFilterThreadsX, kFilterThreadsY, 1);
  const dim3 filter_grid_size(
      static_cast<unsigned int>((frame.intrinsics.width + static_cast<int>(kFilterThreadsX) - 1) / static_cast<int>(kFilterThreadsX)),
      static_cast<unsigned int>((frame.intrinsics.height + static_cast<int>(kFilterThreadsY) - 1) / static_cast<int>(kFilterThreadsY)),
      1);
  estimateDepthNormalsKernel<<<filter_grid_size, filter_block_size>>>(
      workspace.depth.data(), workspace.normals.data(), workspace.normal_valid.data());
  throwCudaError(cudaGetLastError(), "CUDA depth normal kernel launch");

  bilateralFilterDepthKernel<<<filter_grid_size, filter_block_size>>>(
      workspace.depth.data(), workspace.normals.data(), workspace.normal_valid.data(), workspace.filtered_depth.data());
  throwCudaError(cudaGetLastError(), "CUDA depth bilateral filter kernel launch");
  throwCudaError(cudaDeviceSynchronize(), "CUDA depth filter kernel sync");
  try {
    active_blocks = allocateActiveBlocksCuda(
        frame, T_wc, volume.voxelSize(), volume.truncationDistance(), allocation_stride, workspace.filtered_depth.data());
  } catch (const CudaActiveBlockAllocationFallback&) {
    active_blocks = volume.prepareFrameIntegration(frame, T_wc, allocation_stride);
  }
  if (active_blocks.empty()) {
    return;
  }

  active_block_indices = ensureGpuVolumeCacheBlocks(active_blocks, volume_cache);
  if (active_block_indices.empty()) {
    return;
  }

  workspace.active_block_indices.copyFromHost(active_block_indices.data(), active_block_indices.size());
  params.num_blocks = static_cast<unsigned int>(active_block_indices.size());

  throwCudaError(cudaMemcpyToSymbol(c_params, &params, sizeof(GpuIntegrationParams)), "cudaMemcpyToSymbol");

  const dim3 block_size(kCudaBlockSize, kCudaBlockSize, kCudaBlockSize);
  const dim3 grid_size(static_cast<unsigned int>(active_block_indices.size()), 1, 1);
  integrateActiveBlocksKernel<<<grid_size, block_size>>>(
      volume_cache.flat_blocks.data(), workspace.active_block_indices.data(), workspace.filtered_depth.data());
  throwCudaError(cudaGetLastError(), "CUDA integrate kernel launch");
  throwCudaError(cudaDeviceSynchronize(), "CUDA integrate kernel sync");
}

bool hasGpuVolumeCache(const VoxelHashTSDF& volume) {
  const GpuVolumeCache& cache = gpuVolumeCache();
  return cache.attached_volume == &volume && !cache.block_keys.empty();
}

std::size_t gpuVolumeCacheBlockCount(const VoxelHashTSDF& volume) {
  return hasGpuVolumeCache(volume) ? gpuVolumeCache().block_keys.size() : volume.blockCount();
}

std::size_t gpuVolumeCacheObservedVoxelCount(const VoxelHashTSDF& volume) {
  if (!hasGpuVolumeCache(volume)) {
    return volume.observedVoxelCount();
  }

  const GpuVolumeCache& cache = gpuVolumeCache();
  if (cache.block_keys.empty()) {
    return 0;
  }

  IntegratorWorkspace& workspace = integratorWorkspace();
  const unsigned long long zero = 0ULL;
  workspace.observed_voxel_count.copyFromHost(&zero, 1);

  constexpr unsigned int kCountThreadsPerBlock = 256U;
  const std::size_t total_voxels = cache.block_keys.size() * static_cast<std::size_t>(VoxelHashTSDF::kBlockVolume);
  const dim3 grid_size(
      static_cast<unsigned int>((total_voxels + kCountThreadsPerBlock - 1U) / kCountThreadsPerBlock),
      1,
      1);
  const dim3 block_size(kCountThreadsPerBlock, 1, 1);
  countObservedVoxelsKernel<<<grid_size, block_size>>>(
      cache.flat_blocks.data(),
      static_cast<unsigned int>(cache.block_keys.size()),
      workspace.observed_voxel_count.data());
  throwCudaError(cudaGetLastError(), "CUDA observed voxel count kernel launch");
  throwCudaError(cudaDeviceSynchronize(), "CUDA observed voxel count kernel sync");

  unsigned long long observed_voxels = 0ULL;
  workspace.observed_voxel_count.copyToHost(&observed_voxels, 1);
  return static_cast<std::size_t>(observed_voxels);
}

const std::vector<BlockKey>& gpuVolumeCacheBlockKeys(const VoxelHashTSDF& volume) {
  static const std::vector<BlockKey> kEmptyKeys;
  return hasGpuVolumeCache(volume) ? gpuVolumeCache().block_keys : kEmptyKeys;
}

const VoxelHashTSDF::FlatBlockRecord* gpuVolumeCacheFlatBlocks(const VoxelHashTSDF& volume) {
  if (!hasGpuVolumeCache(volume)) {
    return nullptr;
  }
  return gpuVolumeCache().flat_blocks.data();
}

}  // namespace tsdfmc
