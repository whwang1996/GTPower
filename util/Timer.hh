#pragma once

#include <iostream>
#include <chrono>


#define TIMERSTART(tag) auto tag##_start = std::chrono::steady_clock::now(),tag##_end = tag##_start;
#define TIMEREND(tag) tag##_end = std::chrono::steady_clock::now();
#define DURATION_s(tag) std::cout << #tag << " time cost: " << std::chrono::duration_cast<std::chrono::seconds>(tag##_end - tag##_start).count() << "s" << std::endl;
#define DURATION_ms(tag) std::cout << #tag << " time cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(tag##_end - tag##_start).count() << "ms" << std::endl;
#define DURATION_us(tag) std::cout << #tag << " time cost: " << std::chrono::duration_cast<std::chrono::microseconds>(tag##_end - tag##_start).count() << "us" << std::endl;
#define DURATION_ns(tag) std::cout << #tag << " time cost: " << std::chrono::duration_cast<std::chrono::nanoseconds>(tag##_end - tag##_start).count() << "ns" << std::endl;


namespace utils {
double CpuTime();

double WallTime();
}