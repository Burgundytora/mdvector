
#include <iostream>
#include <format>

#include "src/header/allocator.h"
#include "src/header/avx2.h"
#include "src/header/intel_mkl.h"
#include "src/header/normal.h"
#include "src/header/time_cost.h"
#include "src/header/mdvector.hpp"

#include "Eigen/Dense"

constexpr bool do_add = true;
constexpr bool do_sub = true;
constexpr bool do_mul = true;
constexpr bool do_div = true;
constexpr size_t loop = 100000000;
constexpr size_t total_element = 10;

template <class T>
void test_norm() {
  T* data1_ = new T[total_element];
  T* data2_ = new T[total_element];
  T* data3_ = new T[total_element];

  // 赋值
  for (size_t i = 0; i < total_element; i++) {
    data1_[i] = 1;
    data2_[i] = 2;
  }

  TimerRecorder a(std::string(typeid(T).name()) + ": norm");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      norm_add<T>(data1_, data2_, data3_, total_element);
    }

    if constexpr (do_sub) {
      norm_sub<T>(data1_, data2_, data3_, total_element);
    }

    if constexpr (do_mul) {
      norm_mul<T>(data1_, data2_, data3_, total_element);
    }

    if constexpr (do_div) {
      norm_div<T>(data1_, data2_, data3_, total_element);
    }
  }

  delete[] data1_;
  delete[] data2_;
  delete[] data3_;
}

template <class T>
void test_avx2() {
  AlignedAllocator<T> allocator_;

  T* data1_ = allocator_.allocate(total_element);
  T* data2_ = allocator_.allocate(total_element);
  T* data3_ = allocator_.allocate(total_element);

  // 赋值
  for (size_t i = 0; i < total_element; i++) {
    data1_[i] = 1;
    data2_[i] = 2;
  }

  TimerRecorder a(std::string(typeid(T).name()) + ": avx2");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      avx2_add<T>(data1_, data2_, data3_, total_element);
    }

    if constexpr (do_sub) {
      avx2_sub<T>(data1_, data2_, data3_, total_element);
    }

    if constexpr (do_mul) {
      avx2_mul<T>(data1_, data2_, data3_, total_element);
    }

    if constexpr (do_div) {
      avx2_div<T>(data1_, data2_, data3_, total_element);
    }
  }

  allocator_.deallocate(data1_);
  allocator_.deallocate(data2_);
  allocator_.deallocate(data3_);
}

template <class T>
void test_mkl_avx2() {
  MklAlignedAllocator<T> allocator_;
  T* data1_ = allocator_.allocate(total_element);
  T* data2_ = allocator_.allocate(total_element);
  T* data3_ = allocator_.allocate(total_element);

  // 赋值
  for (size_t i = 0; i < total_element; i++) {
    data1_[i] = 1;
    data2_[i] = 2;
  }

  TimerRecorder a(std::string(typeid(T).name()) + ": mkl");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      mkl_add<T>(data1_, data2_, data3_, total_element);
    }

    if constexpr (do_sub) {
      mkl_sub<T>(data1_, data2_, data3_, total_element);
    }

    if constexpr (do_mul) {
      mkl_mul<T>(data1_, data2_, data3_, total_element);
    }

    if constexpr (do_div) {
      mkl_div<T>(data1_, data2_, data3_, total_element);
    }
  }

  allocator_.deallocate(data1_);
  allocator_.deallocate(data2_);
  allocator_.deallocate(data3_);
}

void test_eigen_matrixf() {
  Eigen::MatrixXf data1_ = Eigen::MatrixXf::Zero(1, total_element);
  Eigen::MatrixXf data2_ = Eigen::MatrixXf::Zero(1, total_element);
  Eigen::MatrixXf data3_ = Eigen::MatrixXf::Zero(1, total_element);

  // 赋值
  for (size_t i = 0; i < total_element; i++) {
    data1_(i, 0) = 1;
    data2_(i, 0) = 2;
  }

  TimerRecorder a("float: eigen matrix");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      data3_ = data1_ + data2_;
    }

    if constexpr (do_sub) {
      data3_ = data1_ - data2_;
    }

    if constexpr (do_mul) {
      data3_ = data1_.cwiseProduct(data2_);
    }

    if constexpr (do_div) {
      data3_ = data1_.cwiseQuotient(data2_);
    }
  }

  return;
}

void test_eigen_matrixd() {
  Eigen::MatrixXd data1_ = Eigen::MatrixXd::Zero(total_element, 1);
  Eigen::MatrixXd data2_ = Eigen::MatrixXd::Zero(total_element, 1);
  Eigen::MatrixXd data3_ = Eigen::MatrixXd::Zero(total_element, 1);

  // 赋值
  for (size_t i = 0; i < total_element; i++) {
    data1_(i, 0) = 1;
    data2_(i, 0) = 2;
  }

  TimerRecorder a("double: eigen matrix");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      data3_ = data1_ + data2_;
    }

    if constexpr (do_sub) {
      data3_ = data1_ - data2_;
    }

    if constexpr (do_mul) {
      data3_ = data1_.cwiseProduct(data2_);
    }

    if constexpr (do_div) {
      data3_ = data1_.cwiseQuotient(data2_);
    }
  }

  return;
}

void test_eigen_vectorxd() {
  Eigen::VectorXd data1_ = Eigen::VectorXd::Zero(total_element);
  Eigen::VectorXd data2_ = Eigen::VectorXd::Zero(total_element);
  Eigen::VectorXd data3_ = Eigen::VectorXd::Zero(total_element);

  // 赋值
  for (size_t i = 0; i < total_element; i++) {
    data1_(i) = 1;
    data2_(i) = 2;
  }

  TimerRecorder a("double: eigen vector");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      data3_ = data1_ + data2_;
    }

    if constexpr (do_sub) {
      data3_ = data1_ - data2_;
    }

    if constexpr (do_mul) {
      data3_ = data1_.cwiseProduct(data2_);
    }

    if constexpr (do_div) {
      data3_ = data1_.cwiseQuotient(data2_);
    }
  }

  return;
}

void test_eigen_vectorxf() {
  Eigen::VectorXf data1_ = Eigen::VectorXf::Zero(total_element);
  Eigen::VectorXf data2_ = Eigen::VectorXf::Zero(total_element);
  Eigen::VectorXf data3_ = Eigen::VectorXf::Zero(total_element);

  // 赋值
  for (size_t i = 0; i < total_element; i++) {
    data1_(i) = 1;
    data2_(i) = 2;
  }

  TimerRecorder a("float: eigen vector");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      data3_ = data1_ + data2_;
    }

    if constexpr (do_sub) {
      data3_ = data1_ - data2_;
    }

    if constexpr (do_mul) {
      data3_ = data1_.cwiseProduct(data2_);
    }

    if constexpr (do_div) {
      data3_ = data1_.cwiseQuotient(data2_);
    }
  }

  return;
}

int main(int args, char* argv[]) {
  set_mkl_avx2_sequential_mode();

  // float
  test_norm<float>();
  test_avx2<float>();
  test_mkl_avx2<float>();
  test_eigen_matrixf();
  test_eigen_vectorxf();

  // double
  test_norm<double>();
  test_avx2<double>();
  test_mkl_avx2<double>();
  test_eigen_matrixd();
  test_eigen_vectorxd();

  std::cout << "test complete" << std::endl;

  return 0;
}