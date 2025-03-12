#ifndef __TEST_TYPE_H__
#define __TEST_TYPE_H__

#include <iostream>

// 四元数 符合内存对齐
struct quat {
  double e0;
  double e1;
  double e2;
  double e3;

  void Show() { std::cout << e0 << "\n"; }
};

// 三维坐标 不符合内存对齐
struct pos {
  double e0;
  double e1;
  double e2;
};

// 三维坐标 手动内存对齐
struct alignas(32) posSIMD {
  double e0;
  double e1;
  double e2;
};

#endif  // __TEST_TYPE_H__