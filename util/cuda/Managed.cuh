#pragma once
#include <cuda_runtime.h>

#include "CheckCudaRuntime.cuh"

namespace utils::cuda {
class Managed {
public:
  void *operator new(size_t len)  {
    // printf("managed new\n");
    void *ptr;
    CHECK_CUDA_RUNTIME(cudaMallocManaged(&ptr, len));
    return ptr;
  }

  void operator delete(void *ptr) {
    CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());
    CHECK_CUDA_RUNTIME(cudaFree(ptr));
  }

  void* operator new[] (size_t len) {
    // printf("managed new array\n");
    void *ptr; 
    CHECK_CUDA_RUNTIME(cudaMallocManaged(&ptr, len));
    return ptr;
  }
  void operator delete[] (void* ptr) {
    CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());
    CHECK_CUDA_RUNTIME(cudaFree(ptr));
  }
};
}
