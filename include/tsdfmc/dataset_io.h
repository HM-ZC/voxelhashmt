#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <combaseapi.h>
#include <objbase.h>
#include <wincodec.h>
#endif

#include "tsdfmc/math_types.h"

namespace tsdfmc {

namespace fs = std::filesystem;

struct DatasetFrame {
  int frame_id = -1;
  Pose T_wc;
  DepthFrame depth_frame;
  std::size_t valid_depth_samples = 0;
};

inline std::vector<int> listDatasetFrameIds(const fs::path& dataset_root);

inline std::vector<double> readNumericValuesFromFile(const fs::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Failed to open metadata file: " + path.string());
  }

  std::vector<double> values;
  double value = 0.0;
  while (input >> value) {
    values.push_back(value);
  }

  if (values.empty()) {
    throw std::runtime_error("Failed to parse numeric metadata from file: " + path.string());
  }

  return values;
}

inline fs::path resolveDatasetRootFromMetadataPath(const fs::path& metadata_path) {
  if (metadata_path.filename() == "params.txt") {
    return metadata_path.parent_path().parent_path();
  }
  return metadata_path.parent_path();
}

inline fs::path findDatasetIntrinsicsMetadataPath(const fs::path& preferred_path) {
  if (fs::exists(preferred_path)) {
    return preferred_path;
  }

  const fs::path dataset_root = preferred_path.parent_path();
  for (const int frame_id : listDatasetFrameIds(dataset_root)) {
    const fs::path params_path = dataset_root / std::to_string(frame_id) / "params.txt";
    if (fs::exists(params_path)) {
      return params_path;
    }
  }

  throw std::runtime_error("Failed to locate dataset intrinsics metadata. Expected " + preferred_path.string() +
                           " or <frame>/params.txt under " + dataset_root.string());
}

inline fs::path findFirstDatasetDepthImage(const fs::path& dataset_root) {
  for (const int frame_id : listDatasetFrameIds(dataset_root)) {
    const fs::path depth_path = dataset_root / std::to_string(frame_id) / "depth.png";
    if (fs::exists(depth_path)) {
      return depth_path;
    }
  }

  throw std::runtime_error("Failed to locate any dataset depth.png under " + dataset_root.string());
}

inline bool parseIntrinsicsMetadataValues(const std::vector<double>& values,
                                          const fs::path& source_path,
                                          Intrinsics& intrinsics,
                                          float& depth_scale,
                                          bool& has_resolution) {
  has_resolution = false;
  if (values.size() == 7U) {
    intrinsics.fx = static_cast<float>(values[0]);
    intrinsics.fy = static_cast<float>(values[1]);
    intrinsics.cx = static_cast<float>(values[2]);
    intrinsics.cy = static_cast<float>(values[3]);
    intrinsics.width = static_cast<int>(values[4]);
    intrinsics.height = static_cast<int>(values[5]);
    depth_scale = static_cast<float>(values[6]);
    has_resolution = true;
    return true;
  }

  if (values.size() == 5U) {
    intrinsics.fx = static_cast<float>(values[0]);
    intrinsics.fy = static_cast<float>(values[1]);
    intrinsics.cx = static_cast<float>(values[2]);
    intrinsics.cy = static_cast<float>(values[3]);
    depth_scale = static_cast<float>(values[4]);
    return true;
  }

  if (values.size() == 6U) {
    intrinsics.fx = static_cast<float>(values[0]);
    intrinsics.fy = static_cast<float>(values[1]);
    intrinsics.cx = static_cast<float>(values[2]);
    intrinsics.cy = static_cast<float>(values[3]);
    intrinsics.width = static_cast<int>(values[4]);
    intrinsics.height = static_cast<int>(values[5]);
    depth_scale = 1.0f;
    has_resolution = true;
    return true;
  }

  throw std::runtime_error("Unsupported intrinsics metadata format in " + source_path.string() +
                           " (expected 5, 6, or 7 numeric values)");
}

inline std::vector<int> listDatasetFrameIds(const fs::path& dataset_root) {
  std::vector<int> frame_ids;
  for (const auto& entry : fs::directory_iterator(dataset_root)) {
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
  const std::vector<double> values = readNumericValuesFromFile(path);

  if (values.size() == 7U) {
    Quaternionf q;
    Pose pose;
    q.w = static_cast<float>(values[0]);
    q.x = static_cast<float>(values[1]);
    q.y = static_cast<float>(values[2]);
    q.z = static_cast<float>(values[3]);
    pose.t.x = static_cast<float>(values[4]);
    pose.t.y = static_cast<float>(values[5]);
    pose.t.z = static_cast<float>(values[6]);
    pose.R = quaternionToMatrix(q);
    return pose;
  }

  if (values.size() == 12U || values.size() == 16U) {
    Pose pose;
    pose.R.data = {
        static_cast<float>(values[0]), static_cast<float>(values[1]), static_cast<float>(values[2]),
        static_cast<float>(values[4]), static_cast<float>(values[5]), static_cast<float>(values[6]),
        static_cast<float>(values[8]), static_cast<float>(values[9]), static_cast<float>(values[10]),
    };
    pose.t = {
        static_cast<float>(values[3]),
        static_cast<float>(values[7]),
        static_cast<float>(values[11]),
    };
    return pose;
  }

  throw std::runtime_error("Unsupported pose format in " + path.string() +
                           " (expected 7, 12, or 16 numeric values)");
}

template <typename T>
inline std::string makeDecodeError(const std::string& action, const fs::path& path, const T detail) {
  std::ostringstream oss;
  oss << action << ": " << path.string() << " (detail=" << detail << ")";
  return oss.str();
}

#if defined(_WIN32)

template <typename T>
class ScopedComPtr {
 public:
  ScopedComPtr() = default;
  ScopedComPtr(const ScopedComPtr&) = delete;
  ScopedComPtr& operator=(const ScopedComPtr&) = delete;

  ~ScopedComPtr() {
    reset();
  }

  T* get() const {
    return ptr_;
  }

  T** put() {
    reset();
    return &ptr_;
  }

  void reset() {
    if (ptr_ != nullptr) {
      ptr_->Release();
      ptr_ = nullptr;
    }
  }

 private:
  T* ptr_ = nullptr;
};

inline std::string formatHresult(const HRESULT hr) {
  std::ostringstream oss;
  oss << "0x" << std::hex << static_cast<unsigned long>(hr);
  return oss.str();
}

inline void throwWicError(const std::string& action, const fs::path& path, const HRESULT hr) {
  throw std::runtime_error(makeDecodeError(action, path, formatHresult(hr)));
}

inline Intrinsics completeIntrinsicsResolutionFromDepth(const Intrinsics& intrinsics_hint, const fs::path& depth_path) {
  ScopedComPtr<IWICImagingFactory> factory;
  HRESULT hr = CoCreateInstance(
      CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, __uuidof(IWICImagingFactory),
      reinterpret_cast<void**>(factory.put()));
  if (FAILED(hr)) {
    throwWicError("Failed to create WIC imaging factory", depth_path, hr);
  }

  ScopedComPtr<IWICBitmapDecoder> decoder;
  const std::wstring wide_path = depth_path.wstring();
  hr = factory.get()->CreateDecoderFromFilename(
      wide_path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad, decoder.put());
  if (FAILED(hr)) {
    throwWicError("Failed to open depth PNG", depth_path, hr);
  }

  ScopedComPtr<IWICBitmapFrameDecode> source_frame;
  hr = decoder.get()->GetFrame(0, source_frame.put());
  if (FAILED(hr)) {
    throwWicError("Failed to access depth PNG frame", depth_path, hr);
  }

  UINT width = 0;
  UINT height = 0;
  hr = source_frame.get()->GetSize(&width, &height);
  if (FAILED(hr)) {
    throwWicError("Failed to read depth PNG size", depth_path, hr);
  }

  Intrinsics intrinsics = intrinsics_hint;
  intrinsics.width = static_cast<int>(width);
  intrinsics.height = static_cast<int>(height);
  return intrinsics;
}

inline DepthFrame loadDepthPng(const fs::path& path,
                               const Intrinsics& intrinsics,
                               const float depth_scale,
                               std::size_t& valid_depth_samples) {
  valid_depth_samples = 0;

  ScopedComPtr<IWICImagingFactory> factory;
  HRESULT hr = CoCreateInstance(
      CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, __uuidof(IWICImagingFactory),
      reinterpret_cast<void**>(factory.put()));
  if (FAILED(hr)) {
    throwWicError("Failed to create WIC imaging factory", path, hr);
  }

  ScopedComPtr<IWICBitmapDecoder> decoder;
  const std::wstring wide_path = path.wstring();
  hr = factory.get()->CreateDecoderFromFilename(
      wide_path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad, decoder.put());
  if (FAILED(hr)) {
    throwWicError("Failed to open depth PNG", path, hr);
  }

  ScopedComPtr<IWICBitmapFrameDecode> source_frame;
  hr = decoder.get()->GetFrame(0, source_frame.put());
  if (FAILED(hr)) {
    throwWicError("Failed to access depth PNG frame", path, hr);
  }

  UINT width = 0;
  UINT height = 0;
  hr = source_frame.get()->GetSize(&width, &height);
  if (FAILED(hr)) {
    throwWicError("Failed to read depth PNG size", path, hr);
  }

  if (static_cast<int>(width) != intrinsics.width || static_cast<int>(height) != intrinsics.height) {
    throw std::runtime_error(makeDecodeError(
        "Depth image resolution does not match dataset intrinsics metadata", path,
        std::to_string(width) + "x" + std::to_string(height) + " vs " + std::to_string(intrinsics.width) + "x" +
            std::to_string(intrinsics.height)));
  }

  ScopedComPtr<IWICFormatConverter> converter;
  hr = factory.get()->CreateFormatConverter(converter.put());
  if (FAILED(hr)) {
    throwWicError("Failed to create WIC format converter", path, hr);
  }

  hr = converter.get()->Initialize(source_frame.get(),
                                   GUID_WICPixelFormat16bppGray,
                                   WICBitmapDitherTypeNone,
                                   nullptr,
                                   0.0,
                                   WICBitmapPaletteTypeCustom);
  if (FAILED(hr)) {
    throwWicError("Failed to convert depth PNG to 16-bit grayscale", path, hr);
  }

  std::vector<std::uint16_t> raw_depth(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0U);
  const UINT stride = width * static_cast<UINT>(sizeof(std::uint16_t));
  const UINT buffer_size = static_cast<UINT>(raw_depth.size() * sizeof(std::uint16_t));
  hr = converter.get()->CopyPixels(nullptr, stride, buffer_size, reinterpret_cast<BYTE*>(raw_depth.data()));
  if (FAILED(hr)) {
    throwWicError("Failed to copy depth PNG pixels", path, hr);
  }

  DepthFrame frame(intrinsics);
  for (std::size_t i = 0; i < raw_depth.size(); ++i) {
    const std::uint16_t raw_value = raw_depth[i];
    if (raw_value == 0U) {
      continue;
    }

    const float depth = static_cast<float>(raw_value) * depth_scale;
    if (depth <= 0.0f) {
      continue;
    }

    frame.depth[i] = depth;
    ++valid_depth_samples;
  }

  return frame;
}

#else

inline DepthFrame loadDepthPng(const fs::path& path,
                               const Intrinsics& intrinsics,
                               const float depth_scale,
                               std::size_t& valid_depth_samples) {
  static_cast<void>(path);
  static_cast<void>(intrinsics);
  static_cast<void>(depth_scale);
  static_cast<void>(valid_depth_samples);
  throw std::runtime_error("Depth PNG decoding is only implemented for Windows builds via WIC.");
}

inline Intrinsics completeIntrinsicsResolutionFromDepth(const Intrinsics& intrinsics_hint, const fs::path& depth_path) {
  static_cast<void>(intrinsics_hint);
  static_cast<void>(depth_path);
  throw std::runtime_error("Depth PNG size probing is only implemented for Windows builds via WIC.");
}

#endif

inline Intrinsics loadDatasetIntrinsics(const fs::path& path, float& depth_scale) {
  const fs::path metadata_path = findDatasetIntrinsicsMetadataPath(path);
  const std::vector<double> values = readNumericValuesFromFile(metadata_path);
  Intrinsics intrinsics;
  bool has_resolution = false;
  parseIntrinsicsMetadataValues(values, metadata_path, intrinsics, depth_scale, has_resolution);

  if (!has_resolution) {
    const fs::path dataset_root = resolveDatasetRootFromMetadataPath(metadata_path);
    intrinsics = completeIntrinsicsResolutionFromDepth(intrinsics, findFirstDatasetDepthImage(dataset_root));
  }

  return intrinsics;
}

inline DatasetFrame loadDatasetDepthFrame(const fs::path& dataset_root,
                                          const int frame_id,
                                          const std::string& pose_filename,
                                          const Intrinsics& intrinsics,
                                          const float depth_scale) {
  const fs::path frame_dir = dataset_root / std::to_string(frame_id);

  DatasetFrame frame;
  frame.frame_id = frame_id;
  frame.T_wc = loadPoseFromFile(frame_dir / pose_filename);
  frame.depth_frame = loadDepthPng(frame_dir / "depth.png", intrinsics, depth_scale, frame.valid_depth_samples);
  return frame;
}

}  // namespace tsdfmc
