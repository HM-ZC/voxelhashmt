#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tsdfmc {

inline float clampf(const float value, const float min_value, const float max_value) {
  return std::max(min_value, std::min(max_value, value));
}

struct Vec3f {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;

  float& operator[](const std::size_t index) {
    return index == 0 ? x : (index == 1 ? y : z);
  }

  const float& operator[](const std::size_t index) const {
    return index == 0 ? x : (index == 1 ? y : z);
  }
};

struct Vec3i {
  int x = 0;
  int y = 0;
  int z = 0;

  int& operator[](const std::size_t index) {
    return index == 0 ? x : (index == 1 ? y : z);
  }

  const int& operator[](const std::size_t index) const {
    return index == 0 ? x : (index == 1 ? y : z);
  }
};

inline Vec3f operator+(const Vec3f& a, const Vec3f& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3f operator-(const Vec3f& a, const Vec3f& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3f operator*(const Vec3f& v, const float scale) {
  return {v.x * scale, v.y * scale, v.z * scale};
}

inline Vec3f operator*(const float scale, const Vec3f& v) {
  return v * scale;
}

inline Vec3f operator/(const Vec3f& v, const float scale) {
  return {v.x / scale, v.y / scale, v.z / scale};
}

inline Vec3f& operator+=(Vec3f& a, const Vec3f& b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

inline Vec3i operator+(const Vec3i& a, const Vec3i& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3i operator-(const Vec3i& a, const Vec3i& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline float dot(const Vec3f& a, const Vec3f& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3f cross(const Vec3f& a, const Vec3f& b) {
  return {
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x,
  };
}

inline float norm(const Vec3f& v) {
  return std::sqrt(dot(v, v));
}

inline Vec3f normalized(const Vec3f& v) {
  const float length = norm(v);
  if (length <= 1e-8f) {
    return {0.0f, 0.0f, 0.0f};
  }
  return v / length;
}

struct Mat3f {
  std::array<float, 9> data = {1.0f, 0.0f, 0.0f,
                               0.0f, 1.0f, 0.0f,
                               0.0f, 0.0f, 1.0f};

  static Mat3f identity() {
    return {};
  }

  static Mat3f fromColumns(const Vec3f& c0, const Vec3f& c1, const Vec3f& c2) {
    Mat3f result;
    result.data = {c0.x, c1.x, c2.x,
                   c0.y, c1.y, c2.y,
                   c0.z, c1.z, c2.z};
    return result;
  }

  Vec3f column(const std::size_t index) const {
    return {data[index], data[index + 3], data[index + 6]};
  }
};

struct Quaternionf {
  float w = 1.0f;
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

inline Vec3f operator*(const Mat3f& m, const Vec3f& v) {
  return {
      m.data[0] * v.x + m.data[1] * v.y + m.data[2] * v.z,
      m.data[3] * v.x + m.data[4] * v.y + m.data[5] * v.z,
      m.data[6] * v.x + m.data[7] * v.y + m.data[8] * v.z,
  };
}

inline Mat3f transpose(const Mat3f& m) {
  Mat3f result;
  result.data = {m.data[0], m.data[3], m.data[6],
                 m.data[1], m.data[4], m.data[7],
                 m.data[2], m.data[5], m.data[8]};
  return result;
}

inline Quaternionf normalized(const Quaternionf& q) {
  const float length =
      std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (length <= 1e-8f) {
    return {};
  }

  return {q.w / length, q.x / length, q.y / length, q.z / length};
}

inline Mat3f quaternionToMatrix(const Quaternionf& q_raw) {
  const Quaternionf q = normalized(q_raw);
  const float ww = q.w * q.w;
  const float xx = q.x * q.x;
  const float yy = q.y * q.y;
  const float zz = q.z * q.z;
  const float wx = q.w * q.x;
  const float wy = q.w * q.y;
  const float wz = q.w * q.z;
  const float xy = q.x * q.y;
  const float xz = q.x * q.z;
  const float yz = q.y * q.z;

  Mat3f result;
  result.data = {
      ww + xx - yy - zz, 2.0f * (xy - wz),     2.0f * (xz + wy),
      2.0f * (xy + wz),  ww - xx + yy - zz,    2.0f * (yz - wx),
      2.0f * (xz - wy),  2.0f * (yz + wx),     ww - xx - yy + zz,
  };
  return result;
}

struct Pose {
  Mat3f R = Mat3f::identity();
  Vec3f t;

  Vec3f transformPoint(const Vec3f& p) const {
    return (R * p) + t;
  }

  Vec3f transformVector(const Vec3f& v) const {
    return R * v;
  }

  Pose inverse() const {
    Pose result;
    result.R = transpose(R);
    result.t = result.R * (t * -1.0f);
    return result;
  }
};

struct Intrinsics {
  int width = 0;
  int height = 0;
  float fx = 0.0f;
  float fy = 0.0f;
  float cx = 0.0f;
  float cy = 0.0f;
};

struct DepthFrame {
  Intrinsics intrinsics;
  std::vector<float> depth;

  DepthFrame() = default;

  explicit DepthFrame(const Intrinsics& intrinsics_)
      : intrinsics(intrinsics_),
        depth(static_cast<std::size_t>(intrinsics_.width * intrinsics_.height), 0.0f) {}

  float& at(const int u, const int v) {
    return depth[static_cast<std::size_t>(v * intrinsics.width + u)];
  }

  float at(const int u, const int v) const {
    return depth[static_cast<std::size_t>(v * intrinsics.width + u)];
  }
};

inline Vec3f backProject(const int u, const int v, const float depth, const Intrinsics& K) {
  const float x = (static_cast<float>(u) - K.cx) * depth / K.fx;
  const float y = (static_cast<float>(v) - K.cy) * depth / K.fy;
  return {x, y, depth};
}

inline bool projectPoint(const Vec3f& p_c, const Intrinsics& K, float& u, float& v) {
  if (p_c.z <= 1e-6f) {
    return false;
  }

  u = K.fx * p_c.x / p_c.z + K.cx;
  v = K.fy * p_c.y / p_c.z + K.cy;

  return u >= 0.0f && u <= static_cast<float>(K.width - 1) &&
         v >= 0.0f && v <= static_cast<float>(K.height - 1);
}

}  // namespace tsdfmc
