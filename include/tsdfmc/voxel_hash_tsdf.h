#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
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

  VoxelHashTSDF(const float voxel_size, const float truncation_distance)
      : voxel_size_(voxel_size), truncation_distance_(truncation_distance) {}

  void integrate(const DepthFrame& frame, const Pose& T_wc, const int allocation_stride = 2) {
    allocateBlocksForFrame(frame, T_wc, allocation_stride);

    const Pose T_cw = T_wc.inverse();
    for (auto& entry : blocks_) {
      const BlockKey& block_key = entry.first;
      Block& block = entry.second;
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
            if (!sampleDepthNearest(frame, u, v, depth)) {
              continue;
            }

            const float signed_distance = depth - point_c.z;
            if (signed_distance <= -truncation_distance_) {
              continue;
            }

            const float tsdf = clampf(signed_distance / truncation_distance_, -1.0f, 1.0f);
            const int index = localIndexFromCoords(lx, ly, lz);
            Voxel& voxel = block.voxels[static_cast<std::size_t>(index)];

            constexpr float observation_weight = 1.0f;
            const float new_weight = std::min(100.0f, voxel.weight + observation_weight);
            voxel.tsdf = (voxel.tsdf * voxel.weight + tsdf * observation_weight) / new_weight;
            voxel.weight = new_weight;
          }
        }
      }
    }
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
        updateVoxelAtWorld(sample_w, tsdf);
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

 private:
  float voxel_size_ = 0.02f;
  float truncation_distance_ = 0.06f;
  std::unordered_map<BlockKey, Block, BlockKeyHash> blocks_;

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

  void allocateNeighborhood(const BlockKey& center_key) {
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const BlockKey key{center_key.x + dx, center_key.y + dy, center_key.z + dz};
          blocks_.try_emplace(key);
        }
      }
    }
  }

  void allocateBlocksForFrame(const DepthFrame& frame, const Pose& T_wc, const int allocation_stride) {
    const int stride = std::max(1, allocation_stride);
    const Vec3f camera_origin = T_wc.t;

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

        allocateNeighborhood(voxelToBlockKey(worldToVoxel(point_w)));
        allocateNeighborhood(voxelToBlockKey(worldToVoxel(front_point)));
        allocateNeighborhood(voxelToBlockKey(worldToVoxel(back_point)));
      }
    }
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

  const Voxel* findVoxel(const Vec3i& voxel_coord) const {
    const BlockKey key = voxelToBlockKey(voxel_coord);
    const auto it = blocks_.find(key);
    if (it == blocks_.end()) {
      return nullptr;
    }

    const int index = voxelToLocalIndex(voxel_coord);
    return &it->second.voxels[static_cast<std::size_t>(index)];
  }

  void updateVoxelAtWorld(const Vec3f& point_w, const float tsdf) {
    const Vec3i voxel_coord = worldToVoxel(point_w);
    const BlockKey key = voxelToBlockKey(voxel_coord);
    Block& block = blocks_[key];
    const int index = voxelToLocalIndex(voxel_coord);
    Voxel& voxel = block.voxels[static_cast<std::size_t>(index)];

    constexpr float observation_weight = 1.0f;
    const float new_weight = std::min(100.0f, voxel.weight + observation_weight);
    voxel.tsdf = (voxel.tsdf * voxel.weight + tsdf * observation_weight) / new_weight;
    voxel.weight = new_weight;
  }
};

}  // namespace tsdfmc
