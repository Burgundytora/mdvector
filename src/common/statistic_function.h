#ifndef __MDVECTOR_STATISTIC_FUNCTION_H__
#define __MDVECTOR_STATISTIC_FUNCTION_H__

#include <concepts>
#include <type_traits>
#include <vector>

#include "type_concept.h"

// 独立统计函数模板
namespace md {

template <StatisticContainer Container>
auto sum(const Container& c) {
  return std::reduce(c.begin(), c.end());
}

template <StatisticContainer Container>
auto prod(const Container& c) {
  return std::reduce(c.begin(), c.end(), typename Container::value_type(1), std::multiplies<>());
}

template <StatisticContainer Container>
auto max(const Container& c) {
  return *std::max_element(c.begin(), c.end());
}

template <StatisticContainer Container>
auto min(const Container& c) {
  return *std::min_element(c.begin(), c.end());
}

template <StatisticContainer Container>
auto mean(const Container& c) {
  return sum(c) / static_cast<typename Container::value_type>(c.size());
}

template <StatisticContainer Container>
auto variance(const Container& c) {
  auto m = mean(c);
  double sum_sq = std::accumulate(c.begin(), c.end(), 0.0, [m](double acc, auto val) {
    double diff = static_cast<double>(val) - static_cast<double>(m);
    return acc + diff * diff;
  });
  return static_cast<typename Container::value_type>(sum_sq / (c.size() - 1));
}

template <StatisticContainer Container>
auto standard_deviation(const Container& c) {
  return static_cast<typename Container::value_type>(std::sqrt(variance(c)));
}

template <StatisticContainer Container>
auto median(const Container& c) {  // 按值传递以进行排序
  std::vector<typename Container::value_type> vec(c.size());
  vec.assign(std::begin(c), std::end(c));
  std::sort(vec.begin(), vec.end());
  size_t n = vec.size();
  if (n % 2 == 0) {
    return (vec[n / 2 - 1] + vec[n / 2]) / typename Container::value_type(2);
  } else {
    return vec[n / 2];
  }
}

// 最大值索引
template <StatisticContainer Container>
size_t max_index(const Container& c) {
  return std::distance(c.begin(), std::max_element(c.begin(), c.end()));
}

// 最小值索引
template <StatisticContainer Container>
size_t min_index(const Container& c) {
  return std::distance(c.begin(), std::min_element(c.begin(), c.end()));
}

// 绝对值最大值
template <StatisticContainer Container>
auto abs_max(const Container& c) {
  if (c.empty()) return typename Container::value_type(0);
  return *std::max_element(c.begin(), c.end(), [](auto a, auto b) { return std::abs(a) < std::abs(b); });
}

// 绝对值最小值
template <StatisticContainer Container>
auto abs_min(const Container& c) {
  if (c.empty()) return typename Container::value_type(0);
  return *std::min_element(c.begin(), c.end(), [](auto a, auto b) { return std::abs(a) < std::abs(b); });
}

}  // namespace md

#endif  // __MDVECTOR_STATISTIC_FUNCTION_H__