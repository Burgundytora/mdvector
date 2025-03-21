#ifndef HEADER_TIME_COST_H_
#define HEADER_TIME_COST_H_

#include <chrono>
#include <iostream>
#include <map>
#include <string>

using std::map;
using std::string;
using std::to_string;

//
#include "src/test/test_set.h"

struct TimerRecorder {
  TimerRecorder(const std::string &name) {
    this->name_ = name;

    if (speed_recorder_.find(name) != speed_recorder_.end()) {
      speed_recorder_.insert({name, vector<double>{}});
    }

    if (test_name_.size() == 0) {
      for (const auto &it : all_test_points) {
        test_name_.push_back(to_string(it.dim1_) + "*" + to_string(it.dim2_));
      }
    }

    // 获取开始时间点
    start_ = std::chrono::high_resolution_clock::now();
  }

  ~TimerRecorder() {
    auto end = std::chrono::high_resolution_clock::now();

    // 计算持续时间
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
    double speed = total_cal * 1e-5 / duration.count();

    // 输出结果
    std::cout << name_ << "\t\t" << speed << "\n";

    // 记录
    speed_recorder_[name_].push_back(speed);
  }

  string name_ = "";
#ifdef _WIN32
  std::chrono::steady_clock::time_point start_;
#else
  std::chrono::_V2::system_clock::time_point start_;
#endif

  static inline vector<string> test_name_;
  static inline map<string, vector<double>> speed_recorder_;
};

#endif  // HEADER_TIME_COST_H_