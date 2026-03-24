#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "tsdfmc/dataset_io.h"
#include "tsdfmc/reconstruction_backend.h"
#include "tsdfmc/voxel_hash_tsdf.h"

namespace fs = std::filesystem;
using namespace tsdfmc;

namespace {

#ifndef TSDFMC_DEFAULT_DATASET_ROOT
#define TSDFMC_DEFAULT_DATASET_ROOT "datasets"
#endif

struct Options {
  std::string output_path = "output/reconstruction.ply";
  int frames = 28;
  int width = 128;
  int height = 96;
  float voxel_size = 0.003f;
  float truncation = 0.009f;
  std::string dataset_root = TSDFMC_DEFAULT_DATASET_ROOT;
  int frame_start = 0;
  int frame_end = -1;
  int frame_step = 1;
  std::string pose_file = "pose.txt";
  ComputeBackendPreference backend = ComputeBackendPreference::kAuto;
  bool synthetic = false;
};

void printUsage(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--output path] [--frames N] [--width W] [--height H]"
            << " [--voxel-size s] [--truncation t]\n"
            << "       " << argv0
            << " [--dataset-root path] [--frame-start N] [--frame-end N] [--frame-step N]"
            << " [--pose-file pose.txt|speckle_pose.txt] [--backend auto|cpu|gpu] [--output path]\n";
  std::cout << "Default dataset root: " << TSDFMC_DEFAULT_DATASET_ROOT << '\n';
  std::cout << "Use --synthetic to run the synthetic demo instead of dataset reconstruction.\n";
}

bool parseIntArg(const std::string& value, int& out) {
  try {
    out = std::stoi(value);
    return true;
  } catch (...) {
    return false;
  }
}

bool parseFloatArg(const std::string& value, float& out) {
  try {
    out = std::stof(value);
    return true;
  } catch (...) {
    return false;
  }
}

Options parseArgs(const int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      std::exit(0);
    }

    if (arg == "--synthetic") {
      options.synthetic = true;
      continue;
    }

    if (i + 1 >= argc) {
      printUsage(argv[0]);
      throw std::runtime_error("Missing value for argument: " + arg);
    }

    const std::string value = argv[++i];
    if (arg == "--output") {
      options.output_path = value;
    } else if (arg == "--frames") {
      if (!parseIntArg(value, options.frames)) {
        throw std::runtime_error("Invalid integer for --frames");
      }
    } else if (arg == "--width") {
      if (!parseIntArg(value, options.width)) {
        throw std::runtime_error("Invalid integer for --width");
      }
    } else if (arg == "--height") {
      if (!parseIntArg(value, options.height)) {
        throw std::runtime_error("Invalid integer for --height");
      }
    } else if (arg == "--voxel-size") {
      if (!parseFloatArg(value, options.voxel_size)) {
        throw std::runtime_error("Invalid float for --voxel-size");
      }
    } else if (arg == "--truncation") {
      if (!parseFloatArg(value, options.truncation)) {
        throw std::runtime_error("Invalid float for --truncation");
      }
    } else if (arg == "--dataset-root") {
      options.dataset_root = value;
    } else if (arg == "--frame-start") {
      if (!parseIntArg(value, options.frame_start)) {
        throw std::runtime_error("Invalid integer for --frame-start");
      }
    } else if (arg == "--frame-end") {
      if (!parseIntArg(value, options.frame_end)) {
        throw std::runtime_error("Invalid integer for --frame-end");
      }
    } else if (arg == "--frame-step") {
      if (!parseIntArg(value, options.frame_step) || options.frame_step <= 0) {
        throw std::runtime_error("Invalid integer for --frame-step");
      }
    } else if (arg == "--pose-file") {
      options.pose_file = value;
    } else if (arg == "--backend") {
      if (!parseComputeBackendPreference(value, options.backend)) {
        throw std::runtime_error("Invalid value for --backend (expected auto|cpu|gpu)");
      }
    } else {
      printUsage(argv[0]);
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  return options;
}

#if defined(_WIN32)
class ScopedComInitializer {
 public:
  ScopedComInitializer() {
    hr_ = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr_) && hr_ != RPC_E_CHANGED_MODE) {
      std::ostringstream oss;
      oss << "CoInitializeEx failed with HRESULT=0x" << std::hex << static_cast<unsigned long>(hr_);
      throw std::runtime_error(oss.str());
    }
  }

  ~ScopedComInitializer() {
    if (hr_ == S_OK || hr_ == S_FALSE) {
      CoUninitialize();
    }
  }

 private:
  HRESULT hr_ = S_OK;
};
#endif

Pose makeLookAtPose(const Vec3f& eye, const Vec3f& target) {
  const Vec3f world_up{0.0f, 1.0f, 0.0f};
  const Vec3f forward = normalized(target - eye);
  const Vec3f right = normalized(cross(world_up, forward));
  const Vec3f down = cross(forward, right);

  Pose pose;
  pose.R = Mat3f::fromColumns(right, down, forward);
  pose.t = eye;
  return pose;
}

bool intersectSphere(const Vec3f& origin,
                     const Vec3f& direction,
                     const Vec3f& center,
                     const float radius,
                     float& hit_t) {
  const Vec3f oc = origin - center;
  const float b = dot(oc, direction);
  const float c = dot(oc, oc) - radius * radius;
  const float discriminant = b * b - c;
  if (discriminant < 0.0f) {
    return false;
  }

  const float sqrt_discriminant = std::sqrt(discriminant);
  const float t0 = -b - sqrt_discriminant;
  const float t1 = -b + sqrt_discriminant;
  const float candidate = t0 > 1e-4f ? t0 : t1;
  if (candidate <= 1e-4f) {
    return false;
  }

  hit_t = candidate;
  return true;
}

bool intersectAabb(const Vec3f& origin,
                   const Vec3f& direction,
                   const Vec3f& min_corner,
                   const Vec3f& max_corner,
                   float& hit_t) {
  float t_min = 0.0f;
  float t_max = std::numeric_limits<float>::max();

  for (std::size_t axis = 0; axis < 3; ++axis) {
    const float dir = direction[axis];
    const float ori = origin[axis];

    if (std::fabs(dir) < 1e-6f) {
      if (ori < min_corner[axis] || ori > max_corner[axis]) {
        return false;
      }
      continue;
    }

    const float inv_dir = 1.0f / dir;
    float t0 = (min_corner[axis] - ori) * inv_dir;
    float t1 = (max_corner[axis] - ori) * inv_dir;
    if (t0 > t1) {
      std::swap(t0, t1);
    }

    t_min = std::max(t_min, t0);
    t_max = std::min(t_max, t1);
    if (t_max < t_min) {
      return false;
    }
  }

  if (t_max <= 1e-4f) {
    return false;
  }

  hit_t = t_min > 1e-4f ? t_min : t_max;
  return true;
}

bool traceScene(const Vec3f& origin, const Vec3f& direction, float& hit_t) {
  hit_t = std::numeric_limits<float>::max();

  float t = 0.0f;
  if (intersectSphere(origin, direction, Vec3f{-0.18f, 0.02f, 0.0f}, 0.34f, t) && t < hit_t) {
    hit_t = t;
  }
  if (intersectAabb(origin,
                    direction,
                    Vec3f{0.02f, -0.26f, -0.20f},
                    Vec3f{0.42f, 0.14f, 0.18f},
                    t) &&
      t < hit_t) {
    hit_t = t;
  }

  return hit_t < std::numeric_limits<float>::max();
}

DepthFrame renderSyntheticDepth(const Intrinsics& intrinsics,
                                const Pose& T_wc,
                                const int frame_index,
                                std::mt19937& rng) {
  DepthFrame frame(intrinsics);
  const Pose T_cw = T_wc.inverse();

  std::normal_distribution<float> gaussian_noise(0.0f, 0.0025f);
  std::uniform_real_distribution<float> uniform_01(0.0f, 1.0f);

  for (int v = 0; v < intrinsics.height; ++v) {
    for (int u = 0; u < intrinsics.width; ++u) {
      const Vec3f direction_c = normalized(backProject(u, v, 1.0f, intrinsics));
      const Vec3f direction_w = normalized(T_wc.transformVector(direction_c));

      float hit_t = 0.0f;
      if (!traceScene(T_wc.t, direction_w, hit_t)) {
        continue;
      }

      const Vec3f point_w = T_wc.t + direction_w * hit_t;
      const Vec3f point_c = T_cw.transformPoint(point_w);
      float depth = point_c.z;

      const float patterned_bias =
          0.0015f * std::sin(0.12f * static_cast<float>(u) + 0.17f * static_cast<float>(frame_index)) *
          std::cos(0.10f * static_cast<float>(v) - 0.11f * static_cast<float>(frame_index));
      depth += patterned_bias + gaussian_noise(rng);

      if (uniform_01(rng) < 0.03f) {
        depth = 0.0f;
      }

      if (depth > 0.1f && depth < 4.0f) {
        frame.at(u, v) = depth;
      }
    }
  }

  return frame;
}

bool writeMeshAsPly(const fs::path& path, const std::vector<Triangle>& triangles) {
  std::vector<Triangle> filtered;
  filtered.reserve(triangles.size());
  for (const Triangle& triangle : triangles) {
    const Vec3f normal = cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0);
    if (norm(normal) > 1e-8f) {
      filtered.push_back(triangle);
    }
  }

  std::ofstream output(path);
  if (!output) {
    return false;
  }

  const std::size_t vertex_count = filtered.size() * 3U;
  output << "ply\n";
  output << "format ascii 1.0\n";
  output << "comment generated by tsdf_voxel_hash_demo\n";
  output << "element vertex " << vertex_count << "\n";
  output << "property float x\n";
  output << "property float y\n";
  output << "property float z\n";
  output << "property float nx\n";
  output << "property float ny\n";
  output << "property float nz\n";
  output << "element face " << filtered.size() << "\n";
  output << "property list uchar int vertex_indices\n";
  output << "end_header\n";

  for (const Triangle& triangle : filtered) {
    const Vec3f unit_normal = normalized(cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0));
    const Vec3f vertices[3] = {triangle.v0, triangle.v1, triangle.v2};
    for (const Vec3f& vertex : vertices) {
      output << vertex.x << ' ' << vertex.y << ' ' << vertex.z << ' '
             << unit_normal.x << ' ' << unit_normal.y << ' ' << unit_normal.z << '\n';
    }
  }

  for (std::size_t face_index = 0; face_index < filtered.size(); ++face_index) {
    const std::size_t base = face_index * 3U;
    output << "3 " << base << ' ' << base + 1U << ' ' << base + 2U << '\n';
  }

  return true;
}

int runSyntheticDemo(const Options& options) {
  constexpr float kPi = 3.14159265358979323846f;

  Intrinsics intrinsics;
  intrinsics.width = options.width;
  intrinsics.height = options.height;
  intrinsics.fx = static_cast<float>(options.width) * 1.05f;
  intrinsics.fy = static_cast<float>(options.height) * 1.15f;
  intrinsics.cx = (static_cast<float>(options.width) - 1.0f) * 0.5f;
  intrinsics.cy = (static_cast<float>(options.height) - 1.0f) * 0.5f;

  VoxelHashTSDF volume(options.voxel_size, options.truncation);
  const TsdfIntegrationBackend integration_backend(options.backend);
  std::mt19937 rng(42U);

  std::cout << "Mode: synthetic demo\n";
  std::cout << "Frames=" << options.frames << ", resolution=" << options.width << "x" << options.height
            << ", voxel_size=" << options.voxel_size << ", truncation=" << options.truncation << '\n';
  std::cout << "Integration backend: " << integration_backend.description() << '\n';

  for (int i = 0; i < options.frames; ++i) {
    const float angle = 2.0f * kPi * static_cast<float>(i) /
                        static_cast<float>(std::max(1, options.frames));
    const float radius = 1.15f;
    const Vec3f eye{
        radius * std::cos(angle),
        0.18f + 0.10f * std::sin(angle * 0.5f),
        radius * std::sin(angle),
    };
    const Pose T_wc = makeLookAtPose(eye, Vec3f{0.08f, -0.02f, 0.0f});
    DepthFrame frame = renderSyntheticDepth(intrinsics, T_wc, i, rng);
    integration_backend.integrate(volume, frame, T_wc, 2);

    std::cout << "Frame " << (i + 1) << "/" << options.frames
              << " integrated, active blocks=" << volume.blockCount()
              << ", observed voxels=" << volume.observedVoxelCount() << '\n';
  }

  const std::vector<Triangle> triangles = integration_backend.extractMesh(volume);
  fs::path output_path = options.output_path;
  if (output_path.has_parent_path()) {
    fs::create_directories(output_path.parent_path());
  }

  if (!writeMeshAsPly(output_path, triangles)) {
    std::cerr << "Failed to write mesh to " << output_path << '\n';
    return 1;
  }

  std::cout << "Mesh written to " << output_path << '\n';
  std::cout << "Final stats: blocks=" << volume.blockCount()
            << ", observed voxels=" << volume.observedVoxelCount()
            << ", raw triangles=" << triangles.size() << '\n';
  return 0;
}

int runDatasetReconstruction(const Options& options) {
  const fs::path dataset_root = options.dataset_root;
  const fs::path intrinsic_path = dataset_root / "intrinsic.txt";

  if (!fs::exists(dataset_root)) {
    std::cerr << "Dataset directory not found: " << dataset_root << '\n';
    return 1;
  }

  float depth_scale = 0.0f;
  const Intrinsics intrinsics = loadDatasetIntrinsics(intrinsic_path, depth_scale);
  const std::vector<int> frame_ids = listDatasetFrameIds(dataset_root);
  if (frame_ids.empty()) {
    std::cerr << "No frame directories found under " << dataset_root << '\n';
    return 1;
  }

  const int last_frame_id = frame_ids.back();
  const int start_frame = std::max(frame_ids.front(), options.frame_start);
  const int end_frame = options.frame_end >= 0 ? std::min(last_frame_id, options.frame_end) : last_frame_id;

  std::vector<int> selected_frame_ids;
  selected_frame_ids.reserve(frame_ids.size());
  for (const int frame_id : frame_ids) {
    if (frame_id < start_frame || frame_id > end_frame) {
      continue;
    }
    if (((frame_id - start_frame) % options.frame_step) != 0) {
      continue;
    }
    selected_frame_ids.push_back(frame_id);
  }

  if (selected_frame_ids.empty()) {
    std::cerr << "No frames selected with the current start/end/step options.\n";
    return 1;
  }

  VoxelHashTSDF volume(options.voxel_size, options.truncation);
  const TsdfIntegrationBackend integration_backend(options.backend);
  std::cout << "Mode: dataset reconstruction\n";
  std::cout << "Dataset root: " << dataset_root << '\n';
  std::cout << "Intrinsics: fx=" << intrinsics.fx << ", fy=" << intrinsics.fy << ", cx=" << intrinsics.cx
            << ", cy=" << intrinsics.cy << ", width=" << intrinsics.width << ", height=" << intrinsics.height
            << ", depth_scale=" << depth_scale << '\n';
  std::cout << "Frame selection: start=" << start_frame << ", end=" << end_frame
            << ", step=" << options.frame_step << ", count=" << selected_frame_ids.size() << '\n';
  std::cout << "Pose file: " << options.pose_file << '\n';
  std::cout << "Integration backend: " << integration_backend.description() << '\n';
  std::cout << "Depth input: <frame_id>/depth.png\n";

  const std::size_t total_pixels =
      static_cast<std::size_t>(intrinsics.width) * static_cast<std::size_t>(intrinsics.height);

  for (std::size_t i = 0; i < selected_frame_ids.size(); ++i) {
    const int frame_id = selected_frame_ids[i];
    const DatasetFrame frame =
        loadDatasetDepthFrame(dataset_root, frame_id, options.pose_file, intrinsics, depth_scale);

    std::cout << "Loading frame " << frame_id << " (" << (i + 1) << "/" << selected_frame_ids.size()
              << "), valid_depth=" << frame.valid_depth_samples << "/" << total_pixels << '\n';
    integration_backend.integrate(volume, frame.depth_frame, frame.T_wc, 2);
    std::cout << "Integrated frame " << frame_id
              << ", active blocks=" << volume.blockCount()
              << ", observed voxels=" << volume.observedVoxelCount() << '\n';
  }

  std::cout << "Extracting mesh with marching cubes...\n";
  const std::vector<Triangle> triangles = integration_backend.extractMesh(volume);
  fs::path output_path = options.output_path;
  if (output_path.has_parent_path()) {
    fs::create_directories(output_path.parent_path());
  }

  if (!writeMeshAsPly(output_path, triangles)) {
    std::cerr << "Failed to write mesh to " << output_path << '\n';
    return 1;
  }

  std::cout << "Mesh written to " << output_path << '\n';
  std::cout << "Final stats: blocks=" << volume.blockCount()
            << ", observed voxels=" << volume.observedVoxelCount()
            << ", raw triangles=" << triangles.size() << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parseArgs(argc, argv);
#if defined(_WIN32)
    if (!options.synthetic) {
      const ScopedComInitializer com_initializer;
      return runDatasetReconstruction(options);
    }
#else
    if (!options.synthetic) {
      return runDatasetReconstruction(options);
    }
#endif
    return runSyntheticDemo(options);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
