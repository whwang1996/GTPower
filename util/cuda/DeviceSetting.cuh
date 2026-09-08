#pragma once
#include <cassert>
#include "CheckCudaRuntime.cuh"

namespace utils::cuda{
void setDevice(int* current_device);
}