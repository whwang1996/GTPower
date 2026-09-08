#pragma once

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

namespace utils {
class Logger : public std::stringstream {
  private:
  std::string level_;
  std::string func_;
  std::string file_;
  int line_;

  public:
#ifdef NDEBUG
  static inline bool is_debug = false;
#else
  static inline bool is_debug = false;
#endif

  Logger(const std::string& level, const std::string& func, const std::string& file,
    int line)
    : level_(level)
    , func_(func)
    , file_(file)
    , line_(line) {}

  template <typename T>
  friend Logger& operator<<(Logger& mylog, const T& v) {
    static_cast<std::stringstream&>(mylog) << v;

    return mylog;
  }

  void OutputStr(const std::string& msg)
  {
    std::cout << msg << std::endl;
  }

  ~Logger() {
    std::string msg;
    msg = file_ + ":" + std::to_string(line_) + "] ";

    if (level_ == "DEBUG") {
      assert(is_debug);
      msg = "D" + msg + str();
    } else if (level_ == "INFO") {
      msg = "I" + msg + str();
    } else if (level_ == "ERROR" || level_ == "WARN") {
      msg = "[" + level_ + "]" + msg + str();
    } else {
      msg = level_ + msg + str();
    }
    OutputStr(msg);

    if (level_ == "ERROR") {
      exit(EXIT_FAILURE);
    }
  }
};
  
} // end of namespace utils

#define LOG(level) LOG_##level
#define LOG_INFO ::utils::Logger("INFO", __FUNCTION__, __FILE__, __LINE__)
#define LOG_DEBUG ::utils::Logger("DEBUG", __FUNCTION__, __FILE__, __LINE__)
#define LOG_WARN ::utils::Logger("WARN", __FUNCTION__, __FILE__, __LINE__)
#define LOG_ERROR ::utils::Logger("ERROR", __FUNCTION__, __FILE__, __LINE__)

#define LOG_BEGIN(level, title) LOG(level) << ">>>>>>> " << title << " >>>>>>>"
#define LOG_END(level, title) LOG(level) << "<<<<<<< " << title << " <<<<<<<"

#define LOG_DEBUG_FLAG ::utils::Logger::is_debug
