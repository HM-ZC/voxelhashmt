#pragma once

#include <string>

#include "tsdfmc/voxel_hash_tsdf.h"

namespace tsdfmc {

enum class ComputeBackendPreference {
  kAuto,
  kCpu,
  kGpu,
};

const char* toString(const ComputeBackendPreference preference);
bool parseComputeBackendPreference(const std::string& value, ComputeBackendPreference& out);

class TsdfIntegrationBackend {
 public:
  explicit TsdfIntegrationBackend(ComputeBackendPreference preference);

  const std::string& description() const {
    return description_;
  }

  bool usesGpu() const {
    return use_gpu_;
  }

  void integrate(VoxelHashTSDF& volume,
                 const DepthFrame& frame,
                 const Pose& T_wc,
                 const int allocation_stride) const;

  std::vector<Triangle> extractMesh(const VoxelHashTSDF& volume, float iso_level = 0.0f) const;

 private:
  bool use_gpu_ = false;
  std::string description_;
};

}  // namespace tsdfmc
