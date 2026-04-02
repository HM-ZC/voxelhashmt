#include "tsdfmc/reconstruction_backend.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>

#if TSDFMC_HAVE_CUDA
#include "cuda_mesh_extractor.h"
#include "cuda_tsdf_integrator.h"
#endif

namespace tsdfmc {

namespace {

std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

constexpr int kDepthBilateralRadius = 2;
constexpr float kDepthBilateralSigmaSpatial = 1.5f;
constexpr float kDepthBilateralSigmaRange = 0.0012f;
constexpr float kDepthBilateralSigmaAngular = 0.35f;
constexpr float kDepthDiscontinuityBase = 0.0012f;
constexpr float kDepthDiscontinuityRelative = 0.0025f;
constexpr float kMinSurfaceConsistencyCosine = 0.6f;

float gaussianKernel(const float squared_value, const float inv_two_sigma_sq) {
  return std::exp(-squared_value * inv_two_sigma_sq);
}

float depthDiscontinuityThreshold(const float reference_depth) {
  return std::max(kDepthDiscontinuityBase, kDepthDiscontinuityRelative * std::max(reference_depth, 0.0f));
}

Vec3f backProjectContinuous(const float u, const float v, const float depth, const Intrinsics& K) {
  const float x = (u - K.cx) * depth / K.fx;
  const float y = (v - K.cy) * depth / K.fy;
  return {x, y, depth};
}

Vec3f orientNormalTowardCamera(const Vec3f& normal, const Vec3f& point_c) {
  if (norm(normal) <= 1e-6f || norm(point_c) <= 1e-6f) {
    return normal;
  }

  const Vec3f view_ray = normalized(point_c);
  return dot(normal, view_ray) > 0.0f ? (normal * -1.0f) : normal;
}

bool estimateSurfaceNormalAtPixel(const DepthFrame& input, const int u, const int v, Vec3f& normal) {
  if (u <= 0 || v <= 0 || u + 1 >= input.intrinsics.width || v + 1 >= input.intrinsics.height) {
    return false;
  }

  const float center_depth = input.at(u, v);
  const float depth_left = input.at(u - 1, v);
  const float depth_right = input.at(u + 1, v);
  const float depth_up = input.at(u, v - 1);
  const float depth_down = input.at(u, v + 1);
  if (center_depth <= 0.0f || depth_left <= 0.0f || depth_right <= 0.0f || depth_up <= 0.0f || depth_down <= 0.0f) {
    return false;
  }

  const float threshold = depthDiscontinuityThreshold(center_depth);
  if (std::fabs(depth_left - center_depth) > threshold ||
      std::fabs(depth_right - center_depth) > threshold ||
      std::fabs(depth_up - center_depth) > threshold ||
      std::fabs(depth_down - center_depth) > threshold) {
    return false;
  }

  const Vec3f p_left = backProjectContinuous(static_cast<float>(u - 1), static_cast<float>(v), depth_left, input.intrinsics);
  const Vec3f p_right = backProjectContinuous(static_cast<float>(u + 1), static_cast<float>(v), depth_right, input.intrinsics);
  const Vec3f p_up = backProjectContinuous(static_cast<float>(u), static_cast<float>(v - 1), depth_up, input.intrinsics);
  const Vec3f p_down = backProjectContinuous(static_cast<float>(u), static_cast<float>(v + 1), depth_down, input.intrinsics);

  const Vec3f dx = p_right - p_left;
  const Vec3f dy = p_down - p_up;
  normal = normalized(cross(dy, dx));
  if (norm(normal) <= 1e-6f) {
    return false;
  }

  const Vec3f center_point = backProjectContinuous(static_cast<float>(u), static_cast<float>(v), center_depth, input.intrinsics);
  normal = orientNormalTowardCamera(normal, center_point);
  return norm(normal) > 1e-6f;
}

DepthFrame applyDepthBilateralFilter(const DepthFrame& input) {
  DepthFrame filtered(input.intrinsics);
  if (input.intrinsics.width <= 0 || input.intrinsics.height <= 0 || input.depth.empty()) {
    return filtered;
  }

  constexpr float kEpsilon = 1e-6f;
  constexpr float kInvTwoSpatialSigmaSq =
      1.0f / (2.0f * kDepthBilateralSigmaSpatial * kDepthBilateralSigmaSpatial);
  constexpr float kInvTwoRangeSigmaSq =
      1.0f / (2.0f * kDepthBilateralSigmaRange * kDepthBilateralSigmaRange);
  constexpr float kInvTwoAngularSigmaSq =
      1.0f / (2.0f * kDepthBilateralSigmaAngular * kDepthBilateralSigmaAngular);

  const std::size_t pixel_count = static_cast<std::size_t>(input.intrinsics.width) *
                                  static_cast<std::size_t>(input.intrinsics.height);
  std::vector<Vec3f> normals(pixel_count);
  std::vector<std::uint8_t> normal_valid(pixel_count, 0U);
  for (int v = 0; v < input.intrinsics.height; ++v) {
    for (int u = 0; u < input.intrinsics.width; ++u) {
      Vec3f normal{};
      if (estimateSurfaceNormalAtPixel(input, u, v, normal)) {
        const std::size_t index = static_cast<std::size_t>(v * input.intrinsics.width + u);
        normals[index] = normal;
        normal_valid[index] = 1U;
      }
    }
  }

  for (int v = 0; v < input.intrinsics.height; ++v) {
    for (int u = 0; u < input.intrinsics.width; ++u) {
      const float center_depth = input.at(u, v);
      if (center_depth <= 0.0f) {
        filtered.at(u, v) = 0.0f;
        continue;
      }

      const std::size_t center_index = static_cast<std::size_t>(v * input.intrinsics.width + u);
      const bool has_center_normal = normal_valid[center_index] != 0U;
      const Vec3f center_normal = has_center_normal ? normals[center_index] : Vec3f{};
      const float discontinuity_threshold = depthDiscontinuityThreshold(center_depth);
      float weighted_depth_sum = 0.0f;
      float weight_sum = 0.0f;
      for (int dv = -kDepthBilateralRadius; dv <= kDepthBilateralRadius; ++dv) {
        const int sv = v + dv;
        if (sv < 0 || sv >= input.intrinsics.height) {
          continue;
        }

        for (int du = -kDepthBilateralRadius; du <= kDepthBilateralRadius; ++du) {
          const int su = u + du;
          if (su < 0 || su >= input.intrinsics.width) {
            continue;
          }

          const float sample_depth = input.at(su, sv);
          if (sample_depth <= 0.0f) {
            continue;
          }
          if (std::fabs(sample_depth - center_depth) > discontinuity_threshold) {
            continue;
          }

          const float spatial_squared = static_cast<float>(du * du + dv * dv);
          const float depth_delta = sample_depth - center_depth;
          const float range_squared = depth_delta * depth_delta;
          float weight =
              gaussianKernel(spatial_squared, kInvTwoSpatialSigmaSq) *
              gaussianKernel(range_squared, kInvTwoRangeSigmaSq);
          if (has_center_normal) {
            const std::size_t sample_index = static_cast<std::size_t>(sv * input.intrinsics.width + su);
            if (normal_valid[sample_index] == 0U) {
              continue;
            }

            const float cosine = clampf(dot(center_normal, normals[sample_index]), -1.0f, 1.0f);
            if (cosine < kMinSurfaceConsistencyCosine) {
              continue;
            }

            const float angular_delta = 1.0f - cosine;
            weight *= gaussianKernel(angular_delta * angular_delta, kInvTwoAngularSigmaSq);
          }
          weighted_depth_sum += weight * sample_depth;
          weight_sum += weight;
        }
      }

      filtered.at(u, v) = weight_sum > kEpsilon ? (weighted_depth_sum / weight_sum) : center_depth;
    }
  }

  return filtered;
}

}  // namespace

const char* toString(const ComputeBackendPreference preference) {
  switch (preference) {
    case ComputeBackendPreference::kAuto:
      return "auto";
    case ComputeBackendPreference::kCpu:
      return "cpu";
    case ComputeBackendPreference::kGpu:
      return "gpu";
  }
  return "unknown";
}

bool parseComputeBackendPreference(const std::string& value, ComputeBackendPreference& out) {
  const std::string lowered = toLower(value);
  if (lowered == "auto") {
    out = ComputeBackendPreference::kAuto;
    return true;
  }
  if (lowered == "cpu") {
    out = ComputeBackendPreference::kCpu;
    return true;
  }
  if (lowered == "gpu") {
    out = ComputeBackendPreference::kGpu;
    return true;
  }
  return false;
}

TsdfIntegrationBackend::TsdfIntegrationBackend(const ComputeBackendPreference preference) {
  switch (preference) {
    case ComputeBackendPreference::kCpu:
      use_gpu_ = false;
      description_ = "CPU";
      return;
    case ComputeBackendPreference::kGpu:
#if TSDFMC_HAVE_CUDA
    {
      const CudaBackendStatus status = queryCudaBackendStatus();
      if (!status.runtime_available) {
        throw std::runtime_error("GPU backend requested but CUDA is unavailable: " + status.message);
      }
      use_gpu_ = true;
      description_ = "GPU (CUDA)";
      return;
    }
#else
      throw std::runtime_error("GPU backend requested but this build was configured without CUDA support.");
#endif
    case ComputeBackendPreference::kAuto:
#if TSDFMC_HAVE_CUDA
    {
      const CudaBackendStatus status = queryCudaBackendStatus();
      use_gpu_ = status.runtime_available;
      description_ = use_gpu_ ? "GPU (CUDA, auto-selected)"
                              : "CPU (CUDA backend built but unavailable: " + status.message + ")";
      return;
    }
#else
      use_gpu_ = false;
      description_ = "CPU (CUDA toolkit not found at configure time)";
      return;
#endif
  }
}

void TsdfIntegrationBackend::integrate(VoxelHashTSDF& volume,
                                       const DepthFrame& frame,
                                       const Pose& T_wc,
                                       const int allocation_stride) const {
  if (!use_gpu_) {
    const DepthFrame filtered_frame = applyDepthBilateralFilter(frame);
    volume.integrate(filtered_frame, T_wc, allocation_stride);
    return;
  }

#if TSDFMC_HAVE_CUDA
  integrateDepthFrameCuda(volume, frame, T_wc, allocation_stride);
#else
  (void)frame;
  (void)T_wc;
  (void)allocation_stride;
  throw std::runtime_error("CUDA backend dispatch reached a CPU-only build.");
#endif
}

std::vector<Triangle> TsdfIntegrationBackend::extractMesh(const VoxelHashTSDF& volume, const float iso_level) const {
  if (!use_gpu_) {
    return volume.extractMesh(iso_level);
  }

#if TSDFMC_HAVE_CUDA
  return extractMeshCuda(volume, iso_level);
#else
  (void)volume;
  (void)iso_level;
  throw std::runtime_error("CUDA mesh extraction dispatch reached a CPU-only build.");
#endif
}

std::size_t TsdfIntegrationBackend::blockCount(const VoxelHashTSDF& volume) const {
  if (!use_gpu_) {
    return volume.blockCount();
  }

#if TSDFMC_HAVE_CUDA
  return gpuVolumeCacheBlockCount(volume);
#else
  return volume.blockCount();
#endif
}

std::size_t TsdfIntegrationBackend::observedVoxelCount(const VoxelHashTSDF& volume) const {
  if (!use_gpu_) {
    return volume.observedVoxelCount();
  }

#if TSDFMC_HAVE_CUDA
  return gpuVolumeCacheObservedVoxelCount(volume);
#else
  return volume.observedVoxelCount();
#endif
}

}  // namespace tsdfmc
