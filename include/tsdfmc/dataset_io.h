#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tsdfmc/math_types.h"

namespace tsdfmc {

namespace fs = std::filesystem;

struct LaserScanFrame {
  int frame_id = -1;
  Pose T_wc;
  std::vector<Vec3f> points_c;
};

inline Intrinsics loadDatasetIntrinsics(const fs::path& path, float& depth_scale) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Failed to open intrinsic file: " + path.string());
  }

  float width = 0.0f;
  float height = 0.0f;
  Intrinsics intrinsics;
  if (!(input >> intrinsics.fx >> intrinsics.fy >> intrinsics.cx >> intrinsics.cy >> width >> height >>
        depth_scale)) {
    throw std::runtime_error("Failed to parse intrinsic file: " + path.string());
  }

  intrinsics.width = static_cast<int>(width);
  intrinsics.height = static_cast<int>(height);
  return intrinsics;
}

inline std::vector<int> listDatasetFrameIds(const fs::path& opt_result_dir) {
  std::vector<int> frame_ids;
  for (const auto& entry : fs::directory_iterator(opt_result_dir)) {
    if (!entry.is_directory()) {
      continue;
    }

    const std::string name = entry.path().filename().string();
    if (!std::all_of(name.begin(), name.end(), [](const unsigned char c) { return std::isdigit(c) != 0; })) {
      continue;
    }

    frame_ids.push_back(std::stoi(name));
  }

  std::sort(frame_ids.begin(), frame_ids.end());
  return frame_ids;
}

inline Pose loadPoseFromFile(const fs::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Failed to open pose file: " + path.string());
  }

  Quaternionf q;
  Pose pose;
  if (!(input >> q.w >> q.x >> q.y >> q.z >> pose.t.x >> pose.t.y >> pose.t.z)) {
    throw std::runtime_error("Failed to parse pose file: " + path.string());
  }

  pose.R = quaternionToMatrix(q);
  return pose;
}

inline std::vector<Vec3f> loadLaserPoints(const fs::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Failed to open laser point file: " + path.string());
  }

  std::vector<Vec3f> points;
  points.reserve(16384U);

  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }

    std::istringstream iss(line);
    Vec3f point;
    if (!(iss >> point.x >> point.y >> point.z)) {
      continue;
    }

    if (point.z > 0.0f) {
      points.push_back(point);
    }
  }

  return points;
}

inline LaserScanFrame loadLaserScanFrame(const fs::path& opt_result_dir,
                                         const int frame_id,
                                         const std::string& pose_filename) {
  const fs::path frame_dir = opt_result_dir / std::to_string(frame_id);

  LaserScanFrame frame;
  frame.frame_id = frame_id;
  frame.T_wc = loadPoseFromFile(frame_dir / pose_filename);
  frame.points_c = loadLaserPoints(frame_dir / "laser_points.txt");
  return frame;
}

}  // namespace tsdfmc
