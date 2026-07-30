#pragma once

#include <chrono>
#include <iostream>
#include <string>

class ScopedTimer {
 public:
  explicit ScopedTimer(const std::string& name)
      : name_(name), start_(std::chrono::steady_clock::now()) {}

  ~ScopedTimer() {
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now() - start_)
                          .count();
    std::cout << "[TIME] " << name_ << ": " << (elapsed_us / 1000.0) << " ms"
              << std::endl;
  }

 private:
  std::string name_;
  std::chrono::steady_clock::time_point start_;
};

// 简单的单次计时函数：直接返回毫秒，不自动打印
inline double ElapsedMs(const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now() - start)
             .count() /
         1000.0;
}

inline double ElapsedMs(const std::chrono::steady_clock::time_point& start,
                        const std::chrono::steady_clock::time_point& end) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
             .count() /
         1000.0;
}
