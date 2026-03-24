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
        "Depth image resolution does not match intrinsic.txt", path,
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

#endif

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
