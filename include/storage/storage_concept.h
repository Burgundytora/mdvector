// core/storage_concept.h
#ifndef __MDVECTOR_STORAGE_CONCEPT__
#define __MDVECTOR_STORAGE_CONCEPT__

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace md {

// ============================================================================
// 数据所有权标签
// ============================================================================
struct owns_data_tag {};
struct view_data_tag {};

// ============================================================================
// 基础存储概念 - 所有存储必须满足
// ============================================================================
template <typename S>
concept BasicStorage = requires(S& s, const S& cs) {
  // 类型定义
  typename S::value_type;

  // 数据访问
  { s.data() } -> std::same_as<typename S::value_type*>;
  { cs.data() } -> std::same_as<const typename S::value_type*>;

  // 大小信息
  { s.size() } -> std::convertible_to<size_t>;
  { s.capacity() } -> std::convertible_to<size_t>;
  { s.used_size() } -> std::convertible_to<size_t>;
};

// ============================================================================
// 可调整大小概念
// ============================================================================
template <typename S>
concept ResizableStorage = BasicStorage<S> && requires(S& s, size_t n) {
  { s.resize(n) } -> std::same_as<void>;
};

// ============================================================================
// 静态大小概念（编译期已知大小）
// ============================================================================
template <typename S>
concept StaticSizedStorage = BasicStorage<S> && requires {
  { S::static_size() } -> std::convertible_to<size_t>;
};

// ============================================================================
// 拥有数据的存储概念 - 通过标签判断
// ============================================================================
template <typename S>
concept OwningStorage = BasicStorage<S> && requires {
  typename S::ownership;
  requires std::same_as<typename S::ownership, owns_data_tag>;
};

// ============================================================================
// 视图存储概念 - 通过标签判断
// ============================================================================
template <typename S>
concept ViewStorage = BasicStorage<S> && requires {
  typename S::ownership;
  requires std::same_as<typename S::ownership, view_data_tag>;
};

}  // namespace md

#endif