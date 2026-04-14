#ifndef __MDVECTOR_ITERATOR_FILL__
#define __MDVECTOR_ITERATOR_FILL__

#include "iterator_concept.h"

#include <algorithm>
#include <random>

namespace md {

template <typename Derived, typename T>
class fill_ops {
 protected:
  Derived& derived() noexcept { return static_cast<Derived&>(*this); }
  const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

 public:
  // ------------------------------------------------------------------------
  // 基础填充
  // ------------------------------------------------------------------------
  void fill(T val)
    requires Iterable<Derived>
  {
    std::fill(derived().begin(), derived().end(), val);
  }

  void set_zeros()
    requires Iterable<Derived> && Numeric<T>
  {
    fill(static_cast<T>(0));
  }

  void set_ones()
    requires Iterable<Derived> && Numeric<T>
  {
    fill(static_cast<T>(1));
  }

  // ------------------------------------------------------------------------
  // 序列生成
  // ------------------------------------------------------------------------
  void set_arange(T start = 0, T step = 1)
    requires Iterable<Derived> && Numeric<T>
  {
    T current = start;
    for (auto& val : derived()) {
      val = current;
      current += step;
    }
  }

  // ------------------------------------------------------------------------
  // 随机数生成
  // ------------------------------------------------------------------------
  void set_random_uniform(T min_val = 0, T max_val = 1)
    requires Iterable<Derived> && Numeric<T>
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    if constexpr (std::is_floating_point_v<T>) {
      std::uniform_real_distribution<T> dis(min_val, max_val);
      for (auto& val : derived()) val = dis(gen);
    } else {
      std::uniform_int_distribution<T> dis(static_cast<int>(min_val), static_cast<int>(max_val));
      for (auto& val : derived()) val = dis(gen);
    }
  }

  void set_random_normal(T mean = 0, T stddev = 1)
    requires Iterable<Derived> && Numeric<T> && std::is_floating_point_v<T>
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dis(mean, stddev);
    for (auto& val : derived()) val = dis(gen);
  }
};

}  // namespace md

#endif  // __MDVECTOR_ITERATOR_FILL__