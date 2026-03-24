#include "tsdfmc/reconstruction_backend.h"

#include <algorithm>
#include <cctype>
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
    volume.integrate(frame, T_wc, allocation_stride);
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

}  // namespace tsdfmc
