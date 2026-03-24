#pragma once

#include <string>

#include "tsdfmc/voxel_hash_tsdf.h"

namespace tsdfmc {

struct CudaBackendStatus {
  bool runtime_available = false;
  std::string message;
};

CudaBackendStatus queryCudaBackendStatus();

void integrateDepthFrameCuda(VoxelHashTSDF& volume,
                             const DepthFrame& frame,
                             const Pose& T_wc,
                             const int allocation_stride);

}  // namespace tsdfmc
