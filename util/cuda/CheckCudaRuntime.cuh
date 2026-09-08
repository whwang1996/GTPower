#pragma once

#include "ErrorCode.hh"

namespace utils::cuda {
int __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);
__device__ void __handle_cuda_exception(const char* err_msg, const char* file, int line);

#define CHECK_CUDA_RUNTIME(op) utils::cuda::__check_cuda_runtime((op), #op, __FILE__, __LINE__)
#define CUDA_ERR_RET(op, ret) \
  do {  \
    if (utils::cuda::__check_cuda_runtime((op), #op, __FILE__, __LINE__) != OK) return ret; \
  } while(0)
 
#define HANDLE_CUDA_EXCEPTION(msg) utils::cuda::__handle_cuda_exception(msg, __FILE__, __LINE__)
}