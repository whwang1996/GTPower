#pragma once
#include <memory>

namespace utils {
template <typename T>
class Singleton {
 public:
  static T& instance();

  Singleton(const Singleton&) = delete;
  Singleton(Singleton&&) = delete;
  Singleton& operator=(const Singleton&) = delete;
  Singleton& operator=(Singleton&&) = delete;

 protected:
  struct token {};
  Singleton() {}
};

template <typename T>
T& Singleton<T>::instance() {
  //static const std::unique_ptr<T> instance{new T{token{}}};
  static T* const instance = new T{token{}};
  return *instance;
}
}
