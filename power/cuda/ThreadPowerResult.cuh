#pragma once

#include "Types.hh"

struct ThreadPowerResult
{
  PowerVal leakage_power = 0.0;
  PowerVal switching_power = 0.0;
  PowerVal glitch_switching_power = 0.0;
  PowerVal internal_power = 0.0;
  PowerVal glitch_internal_power = 0.0;

  __device__ explicit ThreadPowerResult():
  leakage_power(0.0),
  switching_power(0.0),
  glitch_switching_power(0.0),
  internal_power(0.0),
  glitch_internal_power(0.0) {

  }
};
