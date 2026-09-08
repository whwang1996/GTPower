#include <cstdio>
#include <cassert>

#include "CheckCudaRuntime.cuh"

namespace utils::cuda {
int __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
  if (code != cudaSuccess) {
    const char* err_name = cudaGetErrorName(code);
    const char* err_message = cudaGetErrorString(code);
    printf("CUDA runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
    exit(EXIT_FAILURE);
    return GENERAL_ERROR;  // TODO CUDA decide whether exit the program?
  }
  return OK;
}

__device__ void __handle_cuda_exception(const char* err_msg, const char* file, int line) {
  printf("Exception %s:%d, message = %s\n", file, line, err_msg);
  assert(false);
}
}