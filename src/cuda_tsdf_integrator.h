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

bool hasGpuVolumeCache(const VoxelHashTSDF& volume);
std::size_t gpuVolumeCacheBlockCount(const VoxelHashTSDF& volume);
std::size_t gpuVolumeCacheObservedVoxelCount(const VoxelHashTSDF& volume);
const std::vector<BlockKey>& gpuVolumeCacheBlockKeys(const VoxelHashTSDF& volume);
const VoxelHashTSDF::FlatBlockRecord* gpuVolumeCacheFlatBlocks(const VoxelHashTSDF& volume);

}  // namespace tsdfmc
