#pragma once

#include <array>
#include <vector>

#include "tsdfmc/math_types.h"

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

inline void polygoniseTetra(const std::array<Vec3f, 8>& positions,
                            const std::array<float, 8>& values,
                            const std::array<int, 4>& tetra,
                            const float iso_level,
                            std::vector<Triangle>& triangles) {
  std::array<int, 4> inside_vertices{};
  std::array<int, 4> outside_vertices{};
  int inside_count = 0;
  int outside_count = 0;

  for (int i = 0; i < 4; ++i) {
    const int vi = tetra[static_cast<std::size_t>(i)];
    if (values[static_cast<std::size_t>(vi)] < iso_level) {
      inside_vertices[static_cast<std::size_t>(inside_count++)] = vi;
    } else {
      outside_vertices[static_cast<std::size_t>(outside_count++)] = vi;
    }
  }

  if (inside_count == 0 || inside_count == 4) {
    return;
  }

  if (inside_count == 1) {
    const int a = inside_vertices[0];
    const int b = outside_vertices[0];
    const int c = outside_vertices[1];
    const int d = outside_vertices[2];

    const Vec3f p0 = interpolateVertex(positions[static_cast<std::size_t>(a)],
                                       positions[static_cast<std::size_t>(b)],
                                       values[static_cast<std::size_t>(a)],
                                       values[static_cast<std::size_t>(b)],
                                       iso_level);
    const Vec3f p1 = interpolateVertex(positions[static_cast<std::size_t>(a)],
                                       positions[static_cast<std::size_t>(c)],
                                       values[static_cast<std::size_t>(a)],
                                       values[static_cast<std::size_t>(c)],
                                       iso_level);
    const Vec3f p2 = interpolateVertex(positions[static_cast<std::size_t>(a)],
                                       positions[static_cast<std::size_t>(d)],
                                       values[static_cast<std::size_t>(a)],
                                       values[static_cast<std::size_t>(d)],
                                       iso_level);
    triangles.push_back({p0, p1, p2});
    return;
  }

  if (inside_count == 3) {
    const int a = outside_vertices[0];
    const int b = inside_vertices[0];
    const int c = inside_vertices[1];
    const int d = inside_vertices[2];

    const Vec3f p0 = interpolateVertex(positions[static_cast<std::size_t>(a)],
                                       positions[static_cast<std::size_t>(b)],
                                       values[static_cast<std::size_t>(a)],
                                       values[static_cast<std::size_t>(b)],
                                       iso_level);
    const Vec3f p1 = interpolateVertex(positions[static_cast<std::size_t>(a)],
                                       positions[static_cast<std::size_t>(d)],
                                       values[static_cast<std::size_t>(a)],
                                       values[static_cast<std::size_t>(d)],
                                       iso_level);
    const Vec3f p2 = interpolateVertex(positions[static_cast<std::size_t>(a)],
                                       positions[static_cast<std::size_t>(c)],
                                       values[static_cast<std::size_t>(a)],
                                       values[static_cast<std::size_t>(c)],
                                       iso_level);
    triangles.push_back({p0, p1, p2});
    return;
  }

  // inside_count == 2: produce a quad split into two triangles.
  const int a = inside_vertices[0];
  const int b = inside_vertices[1];
  const int c = outside_vertices[0];
  const int d = outside_vertices[1];

  const Vec3f p0 = interpolateVertex(positions[static_cast<std::size_t>(a)],
                                     positions[static_cast<std::size_t>(c)],
                                     values[static_cast<std::size_t>(a)],
                                     values[static_cast<std::size_t>(c)],
                                     iso_level);
  const Vec3f p1 = interpolateVertex(positions[static_cast<std::size_t>(a)],
                                     positions[static_cast<std::size_t>(d)],
                                     values[static_cast<std::size_t>(a)],
                                     values[static_cast<std::size_t>(d)],
                                     iso_level);
  const Vec3f p2 = interpolateVertex(positions[static_cast<std::size_t>(b)],
                                     positions[static_cast<std::size_t>(c)],
                                     values[static_cast<std::size_t>(b)],
                                     values[static_cast<std::size_t>(c)],
                                     iso_level);
  const Vec3f p3 = interpolateVertex(positions[static_cast<std::size_t>(b)],
                                     positions[static_cast<std::size_t>(d)],
                                     values[static_cast<std::size_t>(b)],
                                     values[static_cast<std::size_t>(d)],
                                     iso_level);

  triangles.push_back({p0, p1, p2});
  triangles.push_back({p1, p3, p2});
}

inline void polygoniseCube(const std::array<Vec3f, 8>& positions,
                           const std::array<float, 8>& values,
                           const float iso_level,
                           std::vector<Triangle>& triangles) {
  static constexpr std::array<std::array<int, 4>, 6> kCubeTetrahedra = {{
      {{0, 5, 1, 6}},
      {{0, 1, 2, 6}},
      {{0, 2, 3, 6}},
      {{0, 3, 7, 6}},
      {{0, 7, 4, 6}},
      {{0, 4, 5, 6}},
  }};

  for (const auto& tetra : kCubeTetrahedra) {
    polygoniseTetra(positions, values, tetra, iso_level, triangles);
  }
}

}  // namespace tsdfmc
