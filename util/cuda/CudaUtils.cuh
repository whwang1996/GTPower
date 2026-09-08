#pragma once

namespace utils::cuda {
// ------------------------------my_min------------------------------
__device__ int my_min(const int a, const int b)
{
  return (a < b) ? a : b;
}

__device__ long long int my_min(const long long int a, const long long int b)
{
  return (a < b) ? a : b;
}

__device__ int64_t my_min(const int64_t a, const int64_t b)
{
  return (a < b) ? a : b;
}

__device__ size_t my_min(const size_t a, const size_t b)
{
  return (a < b) ? a : b;
}

__device__ float my_min(const float a, const float b)
{
  return (a < b) ? a : b;
}

__device__ double my_min(const double a, const double b)
{
  return (a < b) ? a : b;
}

template<typename T>
__device__ T my_min(const T, const T) = delete;
// ------------------------------end of my_min------------------------------

// ------------------------------my_max------------------------------
__device__ int my_max(const int a, const int b)
{
  return (a > b) ? a : b;
}

__device__ long long int my_max(const long long int a, const long long int b)
{
  return (a > b) ? a : b;
}

__device__ int64_t my_max(const int64_t a, const int64_t b)
{
  return (a > b) ? a : b;
}

__device__ size_t my_max(const size_t a, const size_t b)
{
  return (a > b) ? a : b;
}

__device__ float my_max(const float a, const float b)
{
  return (a > b) ? a : b;
}

__device__ double my_max(const double a, const double b)
{
  return (a > b) ? a : b;
}

template<typename T>
__device__ T my_max(const T, const T) = delete;
// ------------------------------end of my_max------------------------------

__host__ __device__ long long int ceil_div(const long long int a, const long long int b) {
    return (a + b - 1) / b;
}

__host__ __device__ size_t ceil_div(const size_t a, const size_t b) {
    return (a + b - 1) / b;
}

__host__ __device__ int64_t ceil_div(const int64_t a, const int64_t b) {
    return (a + b - 1) / b;
}

template<typename T>
__host__ __device__ T ceil_div(const T, const T) = delete;
}