#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "tsdfmc/voxel_hash_tsdf.h"

namespace tsdfmc {

class CudaActiveBlockAllocationFallback : public std::runtime_error {
 public:
  explicit CudaActiveBlockAllocationFallback(const std::string& message)
      : std::runtime_error(message) {}
};

std::vector<BlockKey> allocateActiveBlocksCuda(const DepthFrame& frame,
                                               const Pose& T_wc,
                                               float voxel_size,
                                               float truncation_distance,
                                               int allocation_stride,
                                               const float* device_depth = nullptr);

}  // namespace tsdfmc
