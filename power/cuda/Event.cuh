#pragma once

#include "Types.hh"
#include "Managed.cuh"

namespace sta::power {
// We use AoS instead of SoA since the event access of power analysis cannot be coalesced since the threads within the same warp do not access adjacent events
// And the AoS storage stores the events sequentially, so the time and val can be cached by cacheline memchanism
// reference: https://zhuanlan.zhihu.com/p/552884861
struct Event {
  VcdEventTime time;
  VcdEventVal val;
  // bool is_glitch;
};
}  // end of namespace sta::power