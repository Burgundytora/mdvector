#pragma once

#include "scalar_wrapper.h"

namespace md {

template <typename T, typename L, typename R, typename Cal>
class binary_expr;

template <typename Op, typename SubExpr, typename T>
class unary_expr;

template <typename T, typename Condition, typename TrueExpr, typename FalseExpr>
class ternary_expr;

namespace detail {

// Expression nodes own their lightweight child nodes, while containers and
// views remain referenced. This keeps temporary expression trees alive without
// copying any array data.
template <typename>
struct is_expression_node : std::false_type {};

template <typename T, typename L, typename R, typename Cal>
struct is_expression_node<binary_expr<T, L, R, Cal>> : std::true_type {};

template <typename Op, typename SubExpr, typename T>
struct is_expression_node<unary_expr<Op, SubExpr, T>> : std::true_type {};

template <typename T, typename Condition, typename TrueExpr, typename FalseExpr>
struct is_expression_node<ternary_expr<T, Condition, TrueExpr, FalseExpr>> : std::true_type {};

template <typename T>
inline constexpr bool is_expression_node_v = is_expression_node<std::remove_cvref_t<T>>::value;

template <typename T>
using expression_storage_t =
    std::conditional_t<is_expression_node_v<T>, std::remove_cvref_t<T>, const std::remove_cvref_t<T>&>;

}  // namespace detail

template <typename T, typename = void>
struct tensor_scalar_type {
  using raw_type = std::remove_cvref_t<T>;
  using type = detail::expression_storage_t<raw_type>;
  static constexpr size_t rank_ = raw_type::rank_;
  using layout_type = typename raw_type::layout_type;
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
