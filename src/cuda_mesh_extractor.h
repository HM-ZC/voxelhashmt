#pragma once

#include <vector>

#include "tsdfmc/voxel_hash_tsdf.h"

namespace tsdfmc {

std::vector<Triangle> extractMeshCuda(const VoxelHashTSDF& volume, float iso_level = 0.0f);

}  // namespace tsdfmc
