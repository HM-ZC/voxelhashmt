#pragma once

#include <cstddef>
#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>

namespace tsdfmc {

inline void throwCudaError(const cudaError_t error, const char* action) {
  if (error == cudaSuccess) {
    return;
  }

  std::ostringstream oss;
  oss << action << " failed: " << cudaGetErrorString(error);
  throw std::runtime_error(oss.str());
}

template <typename T>
class ReusableDeviceBuffer {
 public:
  ReusableDeviceBuffer() = default;
  ReusableDeviceBuffer(const ReusableDeviceBuffer&) = delete;
  ReusableDeviceBuffer& operator=(const ReusableDeviceBuffer&) = delete;

  ~ReusableDeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  void ensureCapacity(const std::size_t count) {
    if (count <= capacity_) {
      return;
    }

    if (data_ != nullptr) {
      cudaFree(data_);
      data_ = nullptr;
      capacity_ = 0;
    }

    throwCudaError(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc");
    capacity_ = count;
  }

  void copyFromHost(const T* host_data, const std::size_t count) {
    if (count == 0) {
      return;
    }

    ensureCapacity(count);
    throwCudaError(cudaMemcpy(data_, host_data, count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
  }

  void copyToHost(T* host_data, const std::size_t count) const {
    if (count == 0) {
      return;
    }

    throwCudaError(cudaMemcpy(host_data, data_, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
  }

  void memsetBytes(const int value, const std::size_t count) {
    if (count == 0) {
      return;
    }

    ensureCapacity(count);
    throwCudaError(cudaMemset(data_, value, count * sizeof(T)), "cudaMemset");
  }

  T* data() {
    return data_;
  }

  const T* data() const {
    return data_;
  }

  std::size_t capacity() const {
    return capacity_;
  }

 private:
  T* data_ = nullptr;
  std::size_t capacity_ = 0;
};

}  // namespace tsdfmc
