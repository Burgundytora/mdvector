#ifndef __MDARRAY_MULTI_DIM_CONCEPT__
#define __MDARRAY_MULTI_DIM_CONCEPT__

#include "base_concept.h"
#include <concepts>
#include <cstddef>
#include <array>

namespace md {

// ============================================================================
// 基础 MultiDim 概念 - 所有多维类必须满足
// ============================================================================
template <typename M, typename T, size_t Rank>
concept BasicMultiDim = requires(M& m, const M& cm, size_t dim) {
  // 形状信息
  { m.extents() } -> std::convertible_to<std::array<size_t, Rank>>;
  { cm.extents() } -> std::convertible_to<std::array<size_t, Rank>>;
  { m.extent(dim) } -> std::convertible_to<size_t>;
  { cm.extent(dim) } -> std::convertible_to<size_t>;
  { m.rank() } -> std::convertible_to<size_t>;
  { cm.rank() } -> std::convertible_to<size_t>;

  //   // mdspan 访问
  //   { m.mdspan() } -> std::same_as<std::mdspan<T, std::dextents<size_t, Rank>>&>;
  //   { cm.mdspan() } -> std::same_as<const std::mdspan<T, std::dextents<size_t, Rank>>&>;
};

// ============================================================================
// 多维索引概念
// ============================================================================
template <typename M, typename T, size_t Rank, typename... Indices>
concept MultiDimIndexable = requires(M& m, const M& cm, Indices... idx) {
  requires sizeof...(Indices) == Rank;

  { m(idx...) } -> std::same_as<T&>;
  { cm(idx...) } -> std::same_as<const T&>;
  { m[idx...] } -> std::same_as<T&>;
  { cm[idx...] } -> std::same_as<const T&>;
};

// ============================================================================
// 边界检查概念
// ============================================================================
template <typename M, typename T, size_t Rank, typename... Indices>
concept MultiDimBoundsCheck = requires(M& m, const M& cm, Indices... idx) {
  requires sizeof...(Indices) == Rank;

  { m.at(idx...) } -> std::same_as<T&>;
  { cm.at(idx...) } -> std::same_as<const T&>;
};

// ============================================================================
// 索引转换概念
// ============================================================================
template <typename M, size_t Rank, typename... Indices>
concept MultiDimIndexConvert = requires(const M& cm, size_t linear_idx, size_t dim, Indices... idx) {
  requires sizeof...(Indices) == Rank;

  { cm.get_1d_index(idx...) } -> std::convertible_to<size_t>;
  { cm.get_md_index(linear_idx) } -> std::same_as<std::array<size_t, Rank>>;
  { cm.get_dim_index(linear_idx, dim) } -> std::convertible_to<size_t>;
};

// ============================================================================
// 形状修改概念（仅动态）
// ============================================================================
template <typename M, size_t Rank>
concept MultiDimShapeMutable = requires(M& m, const std::array<size_t, Rank>& shape) {
  { m.set_shape(shape) } -> std::same_as<void>;
};

// ============================================================================
// 步长访问概念（仅 stride）
// ============================================================================
template <typename M, size_t Rank>
concept MultiDimStrideAccess = requires(const M& cm, size_t dim) {
  { cm.strides() } -> std::same_as<std::array<size_t, Rank>>;
  { cm.stride(dim) } -> std::convertible_to<size_t>;
};

// ============================================================================
// 静态维度概念（仅 static）
// ============================================================================
template <typename M, size_t Rank>
concept MultiDimStatic = requires {
  { M::extents() } -> std::convertible_to<std::array<size_t, Rank>>;
  { M::extent(size_t{}) } -> std::convertible_to<size_t>;
  requires std::is_default_constructible_v<M>;
};

}  // namespace md

#endif  // __MDARRAY_MULTI_DIM_CONCEPT__