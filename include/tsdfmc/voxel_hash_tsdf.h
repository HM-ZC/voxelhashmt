#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tsdfmc/marching_cubes.h"
#include "tsdfmc/math_types.h"

namespace tsdfmc {

struct DirectionalTsdfLayer {
  float weight = 0.0f;
  float surface_coord = 0.0f;
  float plane_offset = 0.0f;
  std::uint32_t packed_normal = 0U;
};

struct Voxel {
  // Keep the total slot count unchanged versus the previous 3x4 layout:
  // we now separate opposite-facing surfaces into signed dominant-direction bins.
  static constexpr int kDirectionalBins = 6;
  static constexpr int kLayersPerDirection = 2;
  DirectionalTsdfLayer layers[kDirectionalBins][kLayersPerDirection]{};
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
      const Vec3f surface_point_w = T_wc.transformPoint(point_c);
      const float start_range = std::max(0.0f, range - truncation_distance_);
      const float end_range = range + truncation_distance_;

      for (float sample_range = start_range; sample_range <= end_range; sample_range += ray_step) {
        const float signed_distance = range - sample_range;
        const Vec3f sample_w = camera_origin_w + ray_w * sample_range;
        updateVoxelAtWorld(sample_w, range, signed_distance, surface_point_w, ray_w * -1.0f);
      }
    }
  }

  std::vector<Triangle> extractMesh(const float iso_level = 0.0f) const {
    std::vector<Triangle> triangles;
    triangles.reserve(blocks_.size() * 128U);
    for (int direction_index = 0; direction_index < Voxel::kDirectionalBins; ++direction_index) {
      extractDirectionalMultiLayerMesh(direction_index, iso_level, triangles);
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
        if (hasAnyObservation(voxel)) {
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
  static constexpr float kMinViewAngleWeight = 0.15f;
  static constexpr int kNormalBilateralRadius = 1;
  static constexpr float kNormalBilateralSigmaSpatial = 1.0f;
  static constexpr float kNormalBilateralSigmaRange = 0.0012f;
  static constexpr float kNormalBilateralSigmaAngular = 0.35f;
  static constexpr float kDepthDiscontinuityBase = 0.0012f;
  static constexpr float kDepthDiscontinuityRelative = 0.0025f;
  static constexpr float kLayerMergeVoxelScale = 0.25f;
  static constexpr float kLayerExtractVoxelScale = 0.35f;
  static constexpr float kLayerNormalConsistencyCosine = 0.75f;

  static bool blockHasObservedVoxels(const Block& block) {
    for (const Voxel& voxel : block.voxels) {
      if (hasAnyObservation(voxel)) {
        return true;
      }
    }
    return false;
  }

  static bool hasAnyObservation(const Voxel& voxel) {
    for (int direction_index = 0; direction_index < Voxel::kDirectionalBins; ++direction_index) {
      for (int layer_index = 0; layer_index < Voxel::kLayersPerDirection; ++layer_index) {
        if (voxel.layers[direction_index][layer_index].weight > 0.0f) {
          return true;
        }
      }
    }
    return false;
  }

  static float depthDiscontinuityThreshold(const float reference_depth) {
    return std::max(kDepthDiscontinuityBase, kDepthDiscontinuityRelative * std::max(reference_depth, 0.0f));
  }

  static Vec3f orientNormalTowardCamera(const Vec3f& normal, const Vec3f& point_c) {
    if (norm(normal) <= 1e-6f || norm(point_c) <= 1e-6f) {
      return normal;
    }

    const Vec3f view_ray = normalized(point_c);
    return dot(normal, view_ray) > 0.0f ? (normal * -1.0f) : normal;
  }

  static int dominantDirectionIndex(const Vec3f& normal) {
    const float nx = std::fabs(normal.x);
    const float ny = std::fabs(normal.y);
    const float nz = std::fabs(normal.z);
    if (nx <= 1e-6f && ny <= 1e-6f && nz <= 1e-6f) {
      return -1;
    }

    if (nx >= ny && nx >= nz) {
      return normal.x >= 0.0f ? 0 : 1;
    }
    if (ny >= nx && ny >= nz) {
      return normal.y >= 0.0f ? 2 : 3;
    }
    return normal.z >= 0.0f ? 4 : 5;
  }

  float layerMergeTolerance() const {
    return std::max(1e-6f, voxel_size_ * kLayerMergeVoxelScale);
  }

  float layerExtractionTolerance() const {
    return std::max(1e-6f, voxel_size_ * kLayerExtractVoxelScale);
  }

  static float signNotZero(const float value) {
    return value >= 0.0f ? 1.0f : -1.0f;
  }

  static std::uint32_t packNormalOctahedral(const Vec3f& normal_raw) {
    const Vec3f normal = normalized(normal_raw);
    const float abs_sum = std::fabs(normal.x) + std::fabs(normal.y) + std::fabs(normal.z);
    if (abs_sum <= 1e-6f) {
      return 0U;
    }

    Vec3f projected = normal / abs_sum;
    if (projected.z < 0.0f) {
      const float px = (1.0f - std::fabs(projected.y)) * signNotZero(projected.x);
      const float py = (1.0f - std::fabs(projected.x)) * signNotZero(projected.y);
      projected.x = px;
      projected.y = py;
    }

    const auto quantize = [](const float value) -> std::uint32_t {
      const float clamped = clampf(value * 0.5f + 0.5f, 0.0f, 1.0f);
      return static_cast<std::uint32_t>(std::lround(clamped * 65535.0f));
    };

    return quantize(projected.x) | (quantize(projected.y) << 16U);
  }

  static Vec3f unpackNormalOctahedral(const std::uint32_t packed_normal) {
    const auto dequantize = [](const std::uint32_t packed_component) {
      return (static_cast<float>(packed_component & 0xffffU) / 65535.0f) * 2.0f - 1.0f;
    };

    Vec3f normal{
        dequantize(packed_normal),
        dequantize(packed_normal >> 16U),
        0.0f,
    };
    normal.z = 1.0f - std::fabs(normal.x) - std::fabs(normal.y);
    if (normal.z < 0.0f) {
      const float px = (1.0f - std::fabs(normal.y)) * signNotZero(normal.x);
      const float py = (1.0f - std::fabs(normal.x)) * signNotZero(normal.y);
      normal.x = px;
      normal.y = py;
    }
    return normalized(normal);
  }

  static Vec3f directionUnitNormal(const int direction_index) {
    switch (direction_index) {
      case 0:
        return {1.0f, 0.0f, 0.0f};
      case 1:
        return {-1.0f, 0.0f, 0.0f};
      case 2:
        return {0.0f, 1.0f, 0.0f};
      case 3:
        return {0.0f, -1.0f, 0.0f};
      case 4:
        return {0.0f, 0.0f, 1.0f};
      default:
        return {0.0f, 0.0f, -1.0f};
    }
  }

  static float coordinateAlongDirection(const Vec3f& point_w, const int direction_index) {
    return dot(point_w, directionUnitNormal(direction_index));
  }

  static bool isLayerObserved(const DirectionalTsdfLayer& layer) {
    return layer.weight > 0.0f;
  }

  static float planeSignedDistance(const DirectionalTsdfLayer& layer, const Vec3f& point_w) {
    const Vec3f normal = unpackNormalOctahedral(layer.packed_normal);
    return dot(normal, point_w) - layer.plane_offset;
  }

  static void sortDirectionalLayers(Voxel& voxel, const int direction_index) {
    DirectionalTsdfLayer* layers = voxel.layers[direction_index];
    for (int pass = 0; pass < Voxel::kLayersPerDirection - 1; ++pass) {
      for (int i = 0; i < Voxel::kLayersPerDirection - pass - 1; ++i) {
        const bool left_valid = isLayerObserved(layers[i]);
        const bool right_valid = isLayerObserved(layers[i + 1]);
        const bool should_swap =
            (!left_valid && right_valid) ||
            (left_valid && right_valid && layers[i].surface_coord > layers[i + 1].surface_coord);
        if (should_swap) {
          std::swap(layers[i], layers[i + 1]);
        }
      }
    }
  }

  static int findLayerSlotForObservation(const Voxel& voxel,
                                         const int direction_index,
                                         const float surface_coord,
                                         const Vec3f& surface_normal,
                                         const float merge_tolerance,
                                         const float min_normal_cosine) {
    const DirectionalTsdfLayer* layers = voxel.layers[direction_index];
    int empty_index = -1;
    int best_index = -1;
    float best_delta = std::numeric_limits<float>::max();
    for (int layer_index = 0; layer_index < Voxel::kLayersPerDirection; ++layer_index) {
      const DirectionalTsdfLayer& layer = layers[layer_index];
      if (!isLayerObserved(layer)) {
        if (empty_index < 0) {
          empty_index = layer_index;
        }
        continue;
      }

      const Vec3f layer_normal = unpackNormalOctahedral(layer.packed_normal);
      if (dot(layer_normal, surface_normal) < min_normal_cosine) {
        continue;
      }

      const float delta = std::fabs(layer.surface_coord - surface_coord);
      if (delta < best_delta) {
        best_delta = delta;
        best_index = layer_index;
      }
    }

    if (best_index >= 0 && best_delta <= merge_tolerance) {
      return best_index;
    }
    return empty_index;
  }

  static bool selectClosestLayerForSurface(const Voxel& voxel,
                                           const int direction_index,
                                           const float surface_key,
                                           const Vec3f& reference_point,
                                           const float match_tolerance,
                                           DirectionalTsdfLayer& layer_out) {
    const DirectionalTsdfLayer* layers = voxel.layers[direction_index];
    int best_index = -1;
    float best_delta = std::numeric_limits<float>::max();
    for (int layer_index = 0; layer_index < Voxel::kLayersPerDirection; ++layer_index) {
      const DirectionalTsdfLayer& layer = layers[layer_index];
      if (!isLayerObserved(layer)) {
        continue;
      }

      const float delta = std::fabs(planeSignedDistance(layer, reference_point) - surface_key);
      if (delta <= match_tolerance && delta < best_delta) {
        best_delta = delta;
        best_index = layer_index;
      }
    }

    if (best_index < 0) {
      return false;
    }
    layer_out = layers[best_index];
    return true;
  }

  static void integrateDirectionalSample(Voxel& voxel,
                                         const int direction_index,
                                         const float surface_coord,
                                         const Vec3f& surface_point_w,
                                         const Vec3f& surface_normal_w,
                                         const float observation_weight,
                                         const float merge_tolerance,
                                         const float min_normal_cosine) {
    if (direction_index < 0 || direction_index >= Voxel::kDirectionalBins || observation_weight <= 0.0f) {
      return;
    }

    const Vec3f normal = normalized(surface_normal_w);
    if (norm(normal) <= 1e-6f) {
      return;
    }

    const float plane_offset = dot(normal, surface_point_w);
    const int layer_index =
        findLayerSlotForObservation(voxel, direction_index, surface_coord, normal, merge_tolerance, min_normal_cosine);
    if (layer_index < 0) {
      return;
    }

    DirectionalTsdfLayer& layer = voxel.layers[direction_index][layer_index];
    const float previous_weight = layer.weight;
    const float effective_weight = std::min(observation_weight, std::max(0.0f, kMaxVoxelWeight - previous_weight));
    if (effective_weight <= 0.0f) {
      return;
    }

    const float new_weight = previous_weight + effective_weight;
    Vec3f blended_normal = normal;
    layer.surface_coord =
        previous_weight > 0.0f ? ((layer.surface_coord * previous_weight + surface_coord * effective_weight) / new_weight)
                               : surface_coord;
    layer.plane_offset =
        previous_weight > 0.0f ? ((layer.plane_offset * previous_weight + plane_offset * effective_weight) / new_weight)
                               : plane_offset;
    if (previous_weight > 0.0f) {
      const Vec3f previous_normal = unpackNormalOctahedral(layer.packed_normal);
      blended_normal =
          normalized((previous_normal * previous_weight + normal * effective_weight) / new_weight);
      if (norm(blended_normal) <= 1e-6f) {
        blended_normal = normal;
      }
    }
    layer.weight = new_weight;
    layer.packed_normal = packNormalOctahedral(blended_normal);
    sortDirectionalLayers(voxel, direction_index);
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

  static float gaussianWeight(const float squared_value, const float inv_two_sigma_sq) {
    return std::exp(-squared_value * inv_two_sigma_sq);
  }

  static float computeObservationWeight(const float depth,
                                        const float signed_distance,
                                        const float truncation_distance,
                                        const float view_angle_weight = 1.0f) {
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
    const float base_weight = std::max(kMinObservationWeight, depth_weight * robust_weight * band_weight);
    const float angle_weight = clampf(view_angle_weight, kMinViewAngleWeight, 1.0f);
    return base_weight * angle_weight;
  }

  static Vec3f backProjectContinuous(const float u,
                                     const float v,
                                     const float depth,
                                     const Intrinsics& K) {
    const float x = (u - K.cx) * depth / K.fx;
    const float y = (v - K.cy) * depth / K.fy;
    return {x, y, depth};
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
      float reference_depth = 0.0f;
      if (!sampleDepthNearest(frame, u, v, reference_depth)) {
        reference_depth = d00;
      }

      const float threshold = depthDiscontinuityThreshold(reference_depth);
      const float min_depth = std::min(std::min(d00, d10), std::min(d01, d11));
      const float max_depth = std::max(std::max(d00, d10), std::max(d01, d11));
      const float top = d00 + (d10 - d00) * tx;
      const float bottom = d01 + (d11 - d01) * tx;
      if (max_depth - min_depth <= threshold) {
        depth = top + (bottom - top) * ty;
        return true;
      }

      const std::array<float, 4> neighbor_depths = {d00, d10, d01, d11};
      const std::array<float, 4> bilinear_weights = {
          (1.0f - tx) * (1.0f - ty),
          tx * (1.0f - ty),
          (1.0f - tx) * ty,
          tx * ty,
      };

      float weighted_depth_sum = 0.0f;
      float weight_sum = 0.0f;
      for (std::size_t i = 0; i < neighbor_depths.size(); ++i) {
        if (std::fabs(neighbor_depths[i] - reference_depth) > threshold) {
          continue;
        }
        weighted_depth_sum += bilinear_weights[i] * neighbor_depths[i];
        weight_sum += bilinear_weights[i];
      }

      if (weight_sum > 1e-6f) {
        depth = weighted_depth_sum / weight_sum;
        return true;
      }
    }

    return sampleDepthNearest(frame, u, v, depth);
  }

  bool estimateSurfaceNormalBaseCamera(const DepthFrame& frame, const float u, const float v, Vec3f& normal) const {
    float depth_left = 0.0f;
    float depth_right = 0.0f;
    float depth_up = 0.0f;
    float depth_down = 0.0f;
    if (!sampleDepthBilinear(frame, u - 1.0f, v, depth_left) ||
        !sampleDepthBilinear(frame, u + 1.0f, v, depth_right) ||
        !sampleDepthBilinear(frame, u, v - 1.0f, depth_up) ||
        !sampleDepthBilinear(frame, u, v + 1.0f, depth_down)) {
      return false;
    }

    const Vec3f p_left = backProjectContinuous(u - 1.0f, v, depth_left, frame.intrinsics);
    const Vec3f p_right = backProjectContinuous(u + 1.0f, v, depth_right, frame.intrinsics);
    const Vec3f p_up = backProjectContinuous(u, v - 1.0f, depth_up, frame.intrinsics);
    const Vec3f p_down = backProjectContinuous(u, v + 1.0f, depth_down, frame.intrinsics);

    const Vec3f dx = p_right - p_left;
    const Vec3f dy = p_down - p_up;
    normal = normalized(cross(dy, dx));
    return norm(normal) > 1e-6f;
  }

  bool estimateSurfaceNormalCamera(const DepthFrame& frame, const float u, const float v, Vec3f& normal) const {
    Vec3f center_normal{};
    if (!estimateSurfaceNormalBaseCamera(frame, u, v, center_normal)) {
      return false;
    }

    float center_depth = 0.0f;
    if (!sampleDepthBilinear(frame, u, v, center_depth)) {
      return false;
    }
    const Vec3f center_point = backProjectContinuous(u, v, center_depth, frame.intrinsics);
    center_normal = orientNormalTowardCamera(center_normal, center_point);

    constexpr float kEpsilon = 1e-6f;
    constexpr float kInvTwoSpatialSigmaSq =
        1.0f / (2.0f * kNormalBilateralSigmaSpatial * kNormalBilateralSigmaSpatial);
    constexpr float kInvTwoRangeSigmaSq =
        1.0f / (2.0f * kNormalBilateralSigmaRange * kNormalBilateralSigmaRange);
    constexpr float kInvTwoAngularSigmaSq =
        1.0f / (2.0f * kNormalBilateralSigmaAngular * kNormalBilateralSigmaAngular);

    Vec3f weighted_normal_sum{};
    float weight_sum = 0.0f;
    for (int dv = -kNormalBilateralRadius; dv <= kNormalBilateralRadius; ++dv) {
      for (int du = -kNormalBilateralRadius; du <= kNormalBilateralRadius; ++du) {
        const float sample_u = u + static_cast<float>(du);
        const float sample_v = v + static_cast<float>(dv);

        Vec3f sample_normal{};
        if (!estimateSurfaceNormalBaseCamera(frame, sample_u, sample_v, sample_normal)) {
          continue;
        }

        float sample_depth = 0.0f;
        if (!sampleDepthBilinear(frame, sample_u, sample_v, sample_depth)) {
          continue;
        }
        const Vec3f sample_point = backProjectContinuous(sample_u, sample_v, sample_depth, frame.intrinsics);
        sample_normal = orientNormalTowardCamera(sample_normal, sample_point);

        const float spatial_squared = static_cast<float>(du * du + dv * dv);
        const float depth_delta = sample_depth - center_depth;
        const float range_squared = depth_delta * depth_delta;
        const float cosine = clampf(dot(center_normal, sample_normal), -1.0f, 1.0f);
        const float angular_delta = 1.0f - cosine;
        const float angular_squared = angular_delta * angular_delta;
        const float weight =
            gaussianWeight(spatial_squared, kInvTwoSpatialSigmaSq) *
            gaussianWeight(range_squared, kInvTwoRangeSigmaSq) *
            gaussianWeight(angular_squared, kInvTwoAngularSigmaSq);

        weighted_normal_sum += sample_normal * weight;
        weight_sum += weight;
      }
    }

    if (weight_sum <= kEpsilon) {
      normal = center_normal;
      return true;
    }

    normal = normalized(weighted_normal_sum * (1.0f / weight_sum));
    normal = orientNormalTowardCamera(normal, center_point);
    if (norm(normal) <= 1e-6f) {
      normal = center_normal;
    }
    return norm(normal) > 1e-6f;
  }

  void extractDirectionalMultiLayerMesh(const int direction_index,
                                        const float iso_level,
                                        std::vector<Triangle>& triangles) const {
    constexpr int kMaxSurfaceCandidates = 8 * Voxel::kLayersPerDirection;
    const float match_tolerance = layerExtractionTolerance();
    for (const auto& entry : blocks_) {
      const BlockKey& block_key = entry.first;
      const int base_x = block_key.x * kBlockSize;
      const int base_y = block_key.y * kBlockSize;
      const int base_z = block_key.z * kBlockSize;

      for (int lz = 0; lz < kBlockSize; ++lz) {
        for (int ly = 0; ly < kBlockSize; ++ly) {
          for (int lx = 0; lx < kBlockSize; ++lx) {
            const Vec3i cell_origin{base_x + lx, base_y + ly, base_z + lz};
            std::array<Vec3f, 8> positions{};
            const Vec3f cell_center = voxelToWorld(cell_origin) + Vec3f{0.5f * voxel_size_, 0.5f * voxel_size_,
                                                                        0.5f * voxel_size_};
            std::array<float, kMaxSurfaceCandidates> candidate_keys{};
            std::array<int, kMaxSurfaceCandidates> cluster_counts{};
            int candidate_count = 0;
            for (std::size_t corner = 0; corner < kCubeCornerOffsets.size(); ++corner) {
              const Vec3i sample_coord = cell_origin + kCubeCornerOffsets[corner];
              const Voxel* voxel = findVoxel(sample_coord);
              positions[corner] = voxelToWorld(sample_coord);
              if (voxel == nullptr) {
                continue;
              }

              for (int layer_index = 0; layer_index < Voxel::kLayersPerDirection; ++layer_index) {
                const DirectionalTsdfLayer& layer = voxel->layers[direction_index][layer_index];
                if (!isLayerObserved(layer)) {
                  continue;
                }

                if (candidate_count < static_cast<int>(candidate_keys.size())) {
                  candidate_keys[static_cast<std::size_t>(candidate_count++)] = planeSignedDistance(layer, cell_center);
                }
              }
            }

            if (candidate_count <= 0) {
              continue;
            }

            std::sort(candidate_keys.begin(), candidate_keys.begin() + candidate_count);
            std::array<float, kMaxSurfaceCandidates> cluster_centers{};
            int cluster_count = 0;
            for (int candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
              const float surface_key = candidate_keys[static_cast<std::size_t>(candidate_index)];
              if (cluster_count == 0 ||
                  std::fabs(surface_key - cluster_centers[static_cast<std::size_t>(cluster_count - 1)]) >
                      match_tolerance) {
                cluster_centers[static_cast<std::size_t>(cluster_count)] = surface_key;
                cluster_counts[static_cast<std::size_t>(cluster_count)] = 1;
                ++cluster_count;
                continue;
              }

              const int last_cluster = cluster_count - 1;
              const int count = cluster_counts[static_cast<std::size_t>(last_cluster)];
              cluster_centers[static_cast<std::size_t>(last_cluster)] =
                  (cluster_centers[static_cast<std::size_t>(last_cluster)] * static_cast<float>(count) + surface_key) /
                  static_cast<float>(count + 1);
              cluster_counts[static_cast<std::size_t>(last_cluster)] = count + 1;
            }

            for (int cluster_index = 0; cluster_index < cluster_count; ++cluster_index) {
              const float surface_key = cluster_centers[static_cast<std::size_t>(cluster_index)];
              std::array<float, 8> values{};
              bool valid_cube = true;
              for (std::size_t corner = 0; corner < kCubeCornerOffsets.size(); ++corner) {
                const Vec3i sample_coord = cell_origin + kCubeCornerOffsets[corner];
                const Voxel* voxel = findVoxel(sample_coord);
                if (voxel == nullptr) {
                  valid_cube = false;
                  break;
                }

                DirectionalTsdfLayer layer{};
                if (!selectClosestLayerForSurface(*voxel, direction_index, surface_key, cell_center, match_tolerance,
                                                 layer)) {
                  valid_cube = false;
                  break;
                }
                values[corner] = planeSignedDistance(layer, positions[corner]);
              }

              if (!valid_cube) {
                continue;
              }

              polygoniseCube(positions, values, iso_level, triangles);
            }
          }
        }
      }
    }
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
                          const float measurement_depth,
                          const float signed_distance,
                          const Vec3f& surface_point_w,
                          const Vec3f& surface_normal_w) {
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

    const int direction_index = dominantDirectionIndex(surface_normal_w);
    if (direction_index < 0) {
      return;
    }
    const float surface_coord = coordinateAlongDirection(surface_point_w, direction_index);
    integrateDirectionalSample(
        voxel,
        direction_index,
        surface_coord,
        surface_point_w,
        surface_normal_w,
        observation_weight,
        layerMergeTolerance(),
        kLayerNormalConsistencyCosine);
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

            const Vec3f view_ray_c = normalized(point_c);
            float view_angle_weight = 1.0f;
            Vec3f surface_normal_c = view_ray_c * -1.0f;
            if (estimateSurfaceNormalCamera(frame, u, v, surface_normal_c)) {
              view_angle_weight = std::fabs(dot(surface_normal_c, view_ray_c));
            }

            const float observation_weight =
                computeObservationWeight(depth, signed_distance, truncation_distance_, view_angle_weight);
            if (observation_weight <= 0.0f) {
              continue;
            }

            const Vec3f surface_point_c = backProjectContinuous(u, v, depth, frame.intrinsics);
            const Vec3f surface_point_w = T_wc.transformPoint(surface_point_c);
            const int index = localIndexFromCoords(lx, ly, lz);
            Voxel& voxel = block.voxels[static_cast<std::size_t>(index)];
            const Vec3f surface_normal_w = normalized(T_wc.transformVector(surface_normal_c));
            const int direction_index = dominantDirectionIndex(surface_normal_w);
            if (direction_index < 0) {
              continue;
            }
            const float surface_coord = coordinateAlongDirection(surface_point_w, direction_index);
            integrateDirectionalSample(
                voxel,
                direction_index,
                surface_coord,
                surface_point_w,
                surface_normal_w,
                observation_weight,
                layerMergeTolerance(),
                kLayerNormalConsistencyCosine);
          }
        }
      }
    }
  }
};

}  // namespace tsdfmc
