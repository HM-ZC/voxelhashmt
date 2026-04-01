#include "tsdfmc/reconstruction_backend.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

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

float gaussianKernel(const float squared_value, const float inv_two_sigma_sq) {
  return std::exp(-squared_value * inv_two_sigma_sq);
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

  for (int v = 0; v < input.intrinsics.height; ++v) {
    for (int u = 0; u < input.intrinsics.width; ++u) {
      const float center_depth = input.at(u, v);
      if (center_depth <= 0.0f) {
        filtered.at(u, v) = 0.0f;
        continue;
      }

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

          const float spatial_squared = static_cast<float>(du * du + dv * dv);
          const float depth_delta = sample_depth - center_depth;
          const float range_squared = depth_delta * depth_delta;
          const float weight =
              gaussianKernel(spatial_squared, kInvTwoSpatialSigmaSq) *
              gaussianKernel(range_squared, kInvTwoRangeSigmaSq);
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
  const DepthFrame filtered_frame = applyDepthBilateralFilter(frame);
  if (!use_gpu_) {
    volume.integrate(filtered_frame, T_wc, allocation_stride);
    return;
  }

#if TSDFMC_HAVE_CUDA
  integrateDepthFrameCuda(volume, filtered_frame, T_wc, allocation_stride);
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

}  // namespace tsdfmc
