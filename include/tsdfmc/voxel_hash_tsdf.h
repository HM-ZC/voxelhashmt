#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tsdfmc/marching_cubes.h"
#include "tsdfmc/math_types.h"

namespace tsdfmc {

struct Voxel {
  float tsdf = 1.0f;
  float weight = 0.0f;
};

struct BlockKey {
  int x = 0;
  int y = 0;
  int z = 0;

  bool operator==(const BlockKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct BlockKeyHash {
  std::size_t operator()(const BlockKey& key) const {
    std::size_t seed = 0;
    seed ^= std::hash<int>{}(key.x * 73856093) + 0x9e3779b9 + (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<int>{}(key.y * 19349663) + 0x9e3779b9 + (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<int>{}(key.z * 83492791) + 0x9e3779b9 + (seed << 6U) + (seed >> 2U);
    return seed;
  }
};

class VoxelHashTSDF {
 public:
  static constexpr int kBlockSize = 8;
  static constexpr int kBlockVolume = kBlockSize * kBlockSize * kBlockSize;

  struct Block {
    std::array<Voxel, kBlockVolume> voxels{};
  };

  struct FlatBlockRecord {
    BlockKey key;
    Voxel voxels[kBlockVolume];
  };

  VoxelHashTSDF(const float voxel_size, const float truncation_distance)
      : voxel_size_(voxel_size), truncation_distance_(truncation_distance) {}

  void integrate(const DepthFrame& frame, const Pose& T_wc, const int allocation_stride = 1) {
    const std::vector<BlockKey> active_blocks = prepareFrameIntegration(frame, T_wc, allocation_stride);
    integrateBlocks(active_blocks, frame, T_wc);
  }

  void integratePointCloud(const std::vector<Vec3f>& points_c, const Pose& T_wc) {
    const Vec3f camera_origin_w = T_wc.t;
    const float ray_step = std::max(0.5f * voxel_size_, 0.0025f);

    for (const Vec3f& point_c : points_c) {
      const float range = norm(point_c);
      if (range <= 1e-6f) {
        continue;
      }

      const Vec3f ray_c = point_c / range;
      const Vec3f ray_w = normalized(T_wc.transformVector(ray_c));
      const float start_range = std::max(0.0f, range - truncation_distance_);
      const float end_range = range + truncation_distance_;

      for (float sample_range = start_range; sample_range <= end_range; sample_range += ray_step) {
        const float signed_distance = range - sample_range;
        const float tsdf = clampf(signed_distance / truncation_distance_, -1.0f, 1.0f);
        const Vec3f sample_w = camera_origin_w + ray_w * sample_range;
        updateVoxelAtWorld(sample_w, tsdf, range, signed_distance);
      }
    }
  }

  std::vector<Triangle> extractMesh(const float iso_level = 0.0f) const {
    std::vector<Triangle> triangles;
    triangles.reserve(blocks_.size() * 64U);

    for (const auto& entry : blocks_) {
      const BlockKey& block_key = entry.first;
      const int base_x = block_key.x * kBlockSize;
      const int base_y = block_key.y * kBlockSize;
      const int base_z = block_key.z * kBlockSize;

      for (int lz = 0; lz < kBlockSize; ++lz) {
        for (int ly = 0; ly < kBlockSize; ++ly) {
          for (int lx = 0; lx < kBlockSize; ++lx) {
            const Vec3i cell_origin{base_x + lx, base_y + ly, base_z + lz};
            std::array<Vec3f, 8> positions;
            std::array<float, 8> values;
            bool valid_cube = true;

            for (std::size_t corner = 0; corner < kCubeCornerOffsets.size(); ++corner) {
              const Vec3i sample_coord = cell_origin + kCubeCornerOffsets[corner];
              const Voxel* voxel = findVoxel(sample_coord);
              if (voxel == nullptr || voxel->weight <= 0.0f) {
                valid_cube = false;
                break;
              }

              positions[corner] = voxelToWorld(sample_coord);
              values[corner] = voxel->tsdf;
            }

            if (!valid_cube) {
              continue;
            }

            polygoniseCube(positions, values, iso_level, triangles);
          }
        }
      }
    }

    return triangles;
  }

  std::size_t blockCount() const {
    return blocks_.size();
  }

  std::size_t observedVoxelCount() const {
    std::size_t count = 0;
    for (const auto& entry : blocks_) {
      for (const Voxel& voxel : entry.second.voxels) {
        if (voxel.weight > 0.0f) {
          ++count;
        }
      }
    }
    return count;
  }

  float voxelSize() const {
    return voxel_size_;
  }

  float truncationDistance() const {
    return truncation_distance_;
  }

  std::vector<BlockKey> prepareFrameIntegration(const DepthFrame& frame,
                                                const Pose& T_wc,
                                                const int allocation_stride = 1) {
    const std::vector<BlockKey> active_blocks = collectBlocksForFrame(frame, T_wc, allocation_stride);
    ensureBlocksExist(active_blocks);
    return active_blocks;
  }

  void ensureBlocksExist(const std::vector<BlockKey>& block_keys) {
    for (const BlockKey& key : block_keys) {
      blocks_.try_emplace(key);
    }
  }

  std::vector<FlatBlockRecord> copyBlocksToFlat(const std::vector<BlockKey>& block_keys) const {
    std::vector<FlatBlockRecord> flat_blocks;
    flat_blocks.reserve(block_keys.size());

    for (const BlockKey& key : block_keys) {
      const auto it = blocks_.find(key);
      if (it == blocks_.end()) {
        continue;
      }

      FlatBlockRecord record{};
      record.key = key;
      std::copy_n(it->second.voxels.begin(), kBlockVolume, record.voxels);
      flat_blocks.push_back(record);
    }

    return flat_blocks;
  }

  std::vector<BlockKey> observedBlockKeys() const {
    std::vector<BlockKey> keys;
    keys.reserve(blocks_.size());

    for (const auto& entry : blocks_) {
      if (blockHasObservedVoxels(entry.second)) {
        keys.push_back(entry.first);
      }
    }

    return keys;
  }

  std::vector<FlatBlockRecord> copyObservedBlocksToFlat() const {
    return copyBlocksToFlat(observedBlockKeys());
  }

  void applyFlatBlocks(const std::vector<FlatBlockRecord>& flat_blocks) {
    for (const FlatBlockRecord& record : flat_blocks) {
      Block& block = blocks_[record.key];
      std::copy_n(record.voxels, kBlockVolume, block.voxels.begin());
    }
  }

 private:
 float voxel_size_ = 0.02f;
  float truncation_distance_ = 0.06f;
  std::unordered_map<BlockKey, Block, BlockKeyHash> blocks_;
  static constexpr float kMaxVoxelWeight = 100.0f;

  static bool blockHasObservedVoxels(const Block& block) {
    for (const Voxel& voxel : block.voxels) {
      if (voxel.weight > 0.0f) {
        return true;
      }
    }
    return false;
  }

  static int floorDiv(const int value, const int divisor) {
    int quotient = value / divisor;
    const int remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
      --quotient;
    }
    return quotient;
  }

  static int positiveMod(const int value, const int divisor) {
    int remainder = value % divisor;
    if (remainder < 0) {
      remainder += divisor;
    }
    return remainder;
  }

  static int localIndexFromCoords(const int lx, const int ly, const int lz) {
    return lx + kBlockSize * (ly + kBlockSize * lz);
  }

  static float computeObservationWeight(const float depth,
                                        const float signed_distance,
                                        const float truncation_distance) {
    if (depth <= 0.0f || truncation_distance <= 1e-6f) {
      return 0.0f;
    }

    // Depth uncertainty grows with range: down-weight farther samples.
    constexpr float kDepthSigma = 1.5f;
    const float depth_ratio = depth / kDepthSigma;
    const float depth_weight = 1.0f / (1.0f + depth_ratio * depth_ratio);

    // Robust kernel against outliers inside the truncation band.
    const float normalized_residual = std::fabs(signed_distance) / truncation_distance;
    constexpr float kHuberDelta = 0.35f;
    float robust_weight = 1.0f;
    if (normalized_residual > kHuberDelta) {
      robust_weight = kHuberDelta / normalized_residual;
    }

    // Favor voxels closer to the zero-crossing surface.
    const float band_weight = 1.0f - 0.5f * clampf(normalized_residual, 0.0f, 1.0f);

    constexpr float kMinObservationWeight = 0.05f;
    return std::max(kMinObservationWeight, depth_weight * robust_weight * band_weight);
  }

  Vec3i worldToVoxel(const Vec3f& point) const {
    return {
        static_cast<int>(std::floor(point.x / voxel_size_)),
        static_cast<int>(std::floor(point.y / voxel_size_)),
        static_cast<int>(std::floor(point.z / voxel_size_)),
    };
  }

  Vec3f voxelToWorld(const Vec3i& voxel_coord) const {
    return {
        (static_cast<float>(voxel_coord.x) + 0.5f) * voxel_size_,
        (static_cast<float>(voxel_coord.y) + 0.5f) * voxel_size_,
        (static_cast<float>(voxel_coord.z) + 0.5f) * voxel_size_,
    };
  }

  BlockKey voxelToBlockKey(const Vec3i& voxel_coord) const {
    return {
        floorDiv(voxel_coord.x, kBlockSize),
        floorDiv(voxel_coord.y, kBlockSize),
        floorDiv(voxel_coord.z, kBlockSize),
    };
  }

  int voxelToLocalIndex(const Vec3i& voxel_coord) const {
    const int lx = positiveMod(voxel_coord.x, kBlockSize);
    const int ly = positiveMod(voxel_coord.y, kBlockSize);
    const int lz = positiveMod(voxel_coord.z, kBlockSize);
    return localIndexFromCoords(lx, ly, lz);
  }

  void collectNeighborhood(const BlockKey& center_key,
                           std::unordered_set<BlockKey, BlockKeyHash>& active_blocks) const {
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const BlockKey key{center_key.x + dx, center_key.y + dy, center_key.z + dz};
          active_blocks.insert(key);
        }
      }
    }
  }

  std::vector<BlockKey> collectBlocksForFrame(const DepthFrame& frame,
                                              const Pose& T_wc,
                                              const int allocation_stride) const {
    const int stride = std::max(1, allocation_stride);
    const Vec3f camera_origin = T_wc.t;
    std::unordered_set<BlockKey, BlockKeyHash> active_blocks;

    for (int v = 0; v < frame.intrinsics.height; v += stride) {
      for (int u = 0; u < frame.intrinsics.width; u += stride) {
        const float depth = frame.at(u, v);
        if (depth <= 0.0f) {
          continue;
        }

        const Vec3f point_c = backProject(u, v, depth, frame.intrinsics);
        const Vec3f point_w = T_wc.transformPoint(point_c);
        const Vec3f ray_world = normalized(point_w - camera_origin);

        const Vec3f front_point = point_w - ray_world * truncation_distance_;
        const Vec3f back_point = point_w + ray_world * truncation_distance_;

        collectNeighborhood(voxelToBlockKey(worldToVoxel(point_w)), active_blocks);
        collectNeighborhood(voxelToBlockKey(worldToVoxel(front_point)), active_blocks);
        collectNeighborhood(voxelToBlockKey(worldToVoxel(back_point)), active_blocks);
      }
    }

    std::vector<BlockKey> keys;
    keys.reserve(active_blocks.size());
    for (const BlockKey& key : active_blocks) {
      keys.push_back(key);
    }
    return keys;
  }

  bool sampleDepthNearest(const DepthFrame& frame, const float u, const float v, float& depth) const {
    const int ui = static_cast<int>(std::lround(u));
    const int vi = static_cast<int>(std::lround(v));
    if (ui < 0 || ui >= frame.intrinsics.width || vi < 0 || vi >= frame.intrinsics.height) {
      return false;
    }

    depth = frame.at(ui, vi);
    return depth > 0.0f;
  }

  bool sampleDepthBilinear(const DepthFrame& frame, const float u, const float v, float& depth) const {
    const int width = frame.intrinsics.width;
    const int height = frame.intrinsics.height;
    if (width <= 0 || height <= 0) {
      return false;
    }

    if (u < 0.0f || v < 0.0f || u > static_cast<float>(width - 1) || v > static_cast<float>(height - 1)) {
      return false;
    }

    const int x0 = static_cast<int>(std::floor(u));
    const int y0 = static_cast<int>(std::floor(v));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const float tx = u - static_cast<float>(x0);
    const float ty = v - static_cast<float>(y0);

    const float d00 = frame.at(x0, y0);
    const float d10 = frame.at(x1, y0);
    const float d01 = frame.at(x0, y1);
    const float d11 = frame.at(x1, y1);

    if (d00 > 0.0f && d10 > 0.0f && d01 > 0.0f && d11 > 0.0f) {
      const float top = d00 + (d10 - d00) * tx;
      const float bottom = d01 + (d11 - d01) * tx;
      depth = top + (bottom - top) * ty;
      return true;
    }

    return sampleDepthNearest(frame, u, v, depth);
  }

  const Voxel* findVoxel(const Vec3i& voxel_coord) const {
    const BlockKey key = voxelToBlockKey(voxel_coord);
    const auto it = blocks_.find(key);
    if (it == blocks_.end()) {
      return nullptr;
    }

    const int index = voxelToLocalIndex(voxel_coord);
    return &it->second.voxels[static_cast<std::size_t>(index)];
  }

  void updateVoxelAtWorld(const Vec3f& point_w,
                          const float tsdf,
                          const float measurement_depth,
                          const float signed_distance) {
    const Vec3i voxel_coord = worldToVoxel(point_w);
    const BlockKey key = voxelToBlockKey(voxel_coord);
    Block& block = blocks_[key];
    const int index = voxelToLocalIndex(voxel_coord);
    Voxel& voxel = block.voxels[static_cast<std::size_t>(index)];

    const float observation_weight =
        computeObservationWeight(measurement_depth, signed_distance, truncation_distance_);
    if (observation_weight <= 0.0f) {
      return;
    }

    const float new_weight = std::min(kMaxVoxelWeight, voxel.weight + observation_weight);
    voxel.tsdf = (voxel.tsdf * voxel.weight + tsdf * observation_weight) / new_weight;
    voxel.weight = new_weight;
  }

  void integrateBlocks(const std::vector<BlockKey>& active_blocks, const DepthFrame& frame, const Pose& T_wc) {
    const Pose T_cw = T_wc.inverse();
    for (const BlockKey& block_key : active_blocks) {
      auto it = blocks_.find(block_key);
      if (it == blocks_.end()) {
        continue;
      }

      Block& block = it->second;
      const int base_x = block_key.x * kBlockSize;
      const int base_y = block_key.y * kBlockSize;
      const int base_z = block_key.z * kBlockSize;

      for (int lz = 0; lz < kBlockSize; ++lz) {
        for (int ly = 0; ly < kBlockSize; ++ly) {
          for (int lx = 0; lx < kBlockSize; ++lx) {
            const Vec3i voxel_coord{base_x + lx, base_y + ly, base_z + lz};
            const Vec3f point_w = voxelToWorld(voxel_coord);
            const Vec3f point_c = T_cw.transformPoint(point_w);
            if (point_c.z <= 0.0f) {
              continue;
            }

            float u = 0.0f;
            float v = 0.0f;
            if (!projectPoint(point_c, frame.intrinsics, u, v)) {
              continue;
            }

            float depth = 0.0f;
            if (!sampleDepthBilinear(frame, u, v, depth)) {
              continue;
            }

            const float signed_distance = depth - point_c.z;
            if (signed_distance <= -truncation_distance_) {
              continue;
            }

            const float tsdf = clampf(signed_distance / truncation_distance_, -1.0f, 1.0f);
            const int index = localIndexFromCoords(lx, ly, lz);
            Voxel& voxel = block.voxels[static_cast<std::size_t>(index)];
            const float observation_weight = computeObservationWeight(depth, signed_distance, truncation_distance_);
            if (observation_weight <= 0.0f) {
              continue;
            }

            const float new_weight = std::min(kMaxVoxelWeight, voxel.weight + observation_weight);
            voxel.tsdf = (voxel.tsdf * voxel.weight + tsdf * observation_weight) / new_weight;
            voxel.weight = new_weight;
          }
        }
      }
    }
  }
};

}  // namespace tsdfmc
