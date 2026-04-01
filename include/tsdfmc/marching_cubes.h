#pragma once

#include <array>
#include <vector>

#include "tsdfmc/math_types.h"
#include "tsdfmc/marching_cubes_tables.h"

namespace tsdfmc {

struct Triangle {
  Vec3f v0;
  Vec3f v1;
  Vec3f v2;
};

inline constexpr std::array<Vec3i, 8> kCubeCornerOffsets = {
    Vec3i{0, 0, 0},
    Vec3i{1, 0, 0},
    Vec3i{1, 1, 0},
    Vec3i{0, 1, 0},
    Vec3i{0, 0, 1},
    Vec3i{1, 0, 1},
    Vec3i{1, 1, 1},
    Vec3i{0, 1, 1},
};

inline constexpr int kEdgeCorners[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},
    {4, 5}, {5, 6}, {6, 7}, {7, 4},
    {0, 4}, {1, 5}, {2, 6}, {3, 7},
};

inline Vec3f interpolateVertex(const Vec3f& p0,
                               const Vec3f& p1,
                               const float v0,
                               const float v1,
                               const float iso_level) {
  const float delta = v1 - v0;
  if (std::fabs(delta) < 1e-6f) {
    return p0;
  }

  const float t = clampf((iso_level - v0) / delta, 0.0f, 1.0f);
  return p0 + (p1 - p0) * t;
}

inline void polygoniseCube(const std::array<Vec3f, 8>& positions,
                           const std::array<float, 8>& values,
                           const float iso_level,
                           std::vector<Triangle>& triangles) {
  int cube_index = 0;
  for (int corner = 0; corner < 8; ++corner) {
    if (values[static_cast<std::size_t>(corner)] < iso_level) {
      cube_index |= (1 << corner);
    }
  }

  const int edge_mask = kMarchingCubesEdgeTable[static_cast<std::size_t>(cube_index)];
  if (edge_mask == 0) {
    return;
  }

  std::array<Vec3f, 12> edge_vertices{};
  for (int edge = 0; edge < 12; ++edge) {
    if ((edge_mask & (1 << edge)) == 0) {
      continue;
    }

    const int c0 = kEdgeCorners[edge][0];
    const int c1 = kEdgeCorners[edge][1];
    edge_vertices[static_cast<std::size_t>(edge)] = interpolateVertex(
        positions[static_cast<std::size_t>(c0)],
        positions[static_cast<std::size_t>(c1)],
        values[static_cast<std::size_t>(c0)],
        values[static_cast<std::size_t>(c1)],
        iso_level);
  }

  const std::array<int, 16>& tri_row = kMarchingCubesTriTable[static_cast<std::size_t>(cube_index)];
  for (int i = 0; i < 16; i += 3) {
    const int e0 = tri_row[static_cast<std::size_t>(i)];
    if (e0 < 0) {
      break;
    }
    const int e1 = tri_row[static_cast<std::size_t>(i + 1)];
    const int e2 = tri_row[static_cast<std::size_t>(i + 2)];
    triangles.push_back({
        edge_vertices[static_cast<std::size_t>(e0)],
        edge_vertices[static_cast<std::size_t>(e1)],
        edge_vertices[static_cast<std::size_t>(e2)],
    });
  }
}

}  // namespace tsdfmc
