#include "DeviceSetting.cuh"
#include "Log.hh"
#include "Defines.hh"

namespace utils::cuda{
void setDevice(int* current_device) {
  int device_count = -1;
  CHECK_CUDA_RUNTIME(cudaGetDeviceCount(&device_count));

  if (*current_device == INVALID_DEVICE_ID) { // no selected device
    *current_device = device_count - 1; // select the last one
  }
  LOG_INFO << "device_count: " << device_count << " selecting device " << *current_device << " as working device";
  CHECK_CUDA_RUNTIME(cudaSetDevice(*current_device));
}
}