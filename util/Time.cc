#include <sys/time.h>
// #include <ctime>
// #include <sys/resource.h>
// #include <unistd.h>

#include "Timer.hh"

namespace utils {
double CpuTime()
{
  // struct rusage usage;
  // int who = RUSAGE_SELF;
  // // int who = RUSAGE_CHILDREN;
  // // int who = RUSAGE_BOTH;
  // const int status = getrusage(who, &usage);

  // Seconds userTime;
  // if (status == 0) {
  //   userTime = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec * 1e-6;
  //   // sysTime  = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec * 1e-6;
  // } else {
  //   userTime = 0.0;
  // }
  // return userTime;

  return 1.0 * std::clock() / CLOCKS_PER_SEC;
}

double WallTime()
{
  struct timeval tv;
  const int status = gettimeofday(&tv, NULL);

  double tee;
  if (status == 0)
    tee = (double)(tv.tv_sec) + ((double)(tv.tv_usec)) * 1.0e-6;
  else
    tee = 0.0;
  return tee;
}
}