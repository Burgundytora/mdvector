#ifndef __MDVECTOR_STATISTIC_FUNCTION_H__
#define __MDVECTOR_STATISTIC_FUNCTION_H__

#include <concepts>
#include <type_traits>

// 定义容器概念
template <typename C>
concept StatisticContainer = requires(C c) {
  typename C::value_type;
  { c.begin() } -> std::input_iterator;
  { c.end() } -> std::input_iterator;
};

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
  if (c.size() <= 1) return typename Container::value_type(0);

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
auto median(Container c) {  // 按值传递以进行排序
  if (c.empty()) return typename Container::value_type(0);

  std::sort(c.begin(), c.end());
  size_t n = c.size();
  if (n % 2 == 0) {
    return (c[n / 2 - 1] + c[n / 2]) / typename Container::value_type(2);
  } else {
    return c[n / 2];
  }
}

}  // namespace md

#endif  // __MDVECTOR_STATISTIC_FUNCTION_H__