#ifndef __MDVECTOR_EXTRACT_LAYOUT__
#define __MDVECTOR_EXTRACT_LAYOUT__

#include "scalar_wrapper.h"

namespace md {

template <typename T, typename = void>
struct tensor_scalar_type {
  using type = const T&;
  static constexpr size_t rank_ = T::rank_;
  using layout_type = typename T::layout_type;
  static constexpr bool is_scalar = false;
};

template <typename T>
struct tensor_scalar_type<T, std::enable_if_t<std::is_arithmetic_v<T>>> {
  using type = scalar_wrapper<T>;
  static constexpr size_t rank_ = 0;      // 标量 rank 应该是 0
  using layout_type = std::layout_right;  // 标量使用默认布局
  static constexpr bool is_scalar = true;
};

template <typename T>
using AutoType = typename tensor_scalar_type<T>::type;

// 辅助 trait 来检查是否为 layout_stride
template <typename Layout>
struct is_layout_stride : std::false_type {};

template <>
struct is_layout_stride<std::layout_stride> : std::true_type {};

template <typename Layout>
constexpr bool is_layout_stride_v = is_layout_stride<Layout>::value;

// 布局类型转换：如果是 layout_stride 则转换为 layout_right
template <typename Layout>
struct normalize_layout {
  using type = std::conditional_t<is_layout_stride_v<Layout>, std::layout_right, Layout>;
};

template <typename Layout>
using normalize_layout_t = typename normalize_layout<Layout>::type;

// 推导主要操作数的布局类型 - 优先使用非标量类型的layout
template <typename L, typename R>
struct derived_layout_type {
  // 如果 L 不是标量，使用 L 的 layout
  static constexpr bool l_is_scalar = tensor_scalar_type<L>::is_scalar;
  static constexpr bool r_is_scalar = tensor_scalar_type<R>::is_scalar;

  using raw_layout_type =
      std::conditional_t<!l_is_scalar, typename tensor_scalar_type<L>::layout_type,
                         std::conditional_t<!r_is_scalar, typename tensor_scalar_type<R>::layout_type,
                                            std::layout_right>>;  // 如果都是标量，使用默认布局

  // 规范化布局：如果是 layout_stride 则转换为 layout_right
  using type = normalize_layout_t<raw_layout_type>;
};

template <typename L, typename R>
using layout_type_t = typename derived_layout_type<L, R>::type;
// 获取主要操作数的 rank
template <typename L, typename R>
constexpr size_t derived_rank() {
  static constexpr size_t l_rank = tensor_scalar_type<L>::rank_;
  static constexpr size_t r_rank = tensor_scalar_type<R>::rank_;
  return (l_rank > r_rank) ? l_rank : r_rank;
}

}  // namespace md

#endif  // __MDVECTOR_EXTRACT_LAYOUT__