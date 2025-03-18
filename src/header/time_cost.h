#ifndef HEADER_TIME_COST_H_
#define HEADER_TIME_COST_H_

#include <chrono>
#include <iostream>
#include <string>

struct TimerRecorder {
  TimerRecorder(const std::string &name) {
    this->name_ = name;
    // 获取开始时间点
    start_ = std::chrono::high_resolution_clock::now();
  }

  ~TimerRecorder() {
    auto end = std::chrono::high_resolution_clock::now();
    // 计算持续时间
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);

    // 输出结果
    std::cout << name_ << "\t" << duration.count() << "\n";
    start_ = std::chrono::high_resolution_clock::now();
  }

  std::string name_ = "";
#ifdef _WIN32
  std::chrono::steady_clock::time_point start_;
#else
  std::chrono::_V2::system_clock::time_point start_;
#endif
};

#endif  // HEADER_TIME_COST_H_