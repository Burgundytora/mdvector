#ifndef __CHECK_SIMD_H__
#define __CHECK_SIMD_H__

#include <array>
#include <string_view>
#include <type_traits>

#include "aligned_allocator.h"

// 编译期获取类型名
// TODO: 怎么加到静态断言中???
#if defined(__clang__) || defined(__GNUC__)
template <typename T>
constexpr std::string_view type_name() {
  constexpr std::string_view name = __PRETTY_FUNCTION__;
  constexpr size_t prefix = name.find("T = ") + 4;
  constexpr size_t suffix = name.find(";", prefix);
  return name.substr(prefix, suffix - prefix);
}
#elif defined(_MSC_VER)
template <typename T>
constexpr std::string_view type_name() {
  constexpr std::string_view name = __FUNCSIG__;
  constexpr size_t prefix = name.find("type_name<") + 10;
  constexpr size_t suffix = name.find(">(void)");
  return name.substr(prefix, suffix - prefix);
}
#endif

// 用于编译期显示类型名的模板
template <typename>
struct this_type_is_not_aligned;
template <typename>
struct this_type_is_not_POD;

template <typename T>
constexpr void CheckTypeSizeAligned() {
  constexpr size_t size = sizeof(T);
  static_assert((size != 0) && ((size & (size - 1)) == 0), "type is not aligned!");
  if constexpr (!(size != 0) || !((size & (size - 1)) == 0)) {
    this_type_is_not_aligned<T>{};
  }
}

template <typename T>
constexpr void CheckTypePOD() {
  static_assert(std::is_trivial_v<T>, "is not trivial type!");
  static_assert(std::is_standard_layout_v<T>, "is not standard layout type!");
  if constexpr (!std::is_trivial_v<T> || !std::is_standard_layout_v<T>) {
    this_type_is_not_POD<T>{};
  }
}

// 检查类型是否满足SIMD指令集要求
// 1.要求T为POD类型
// 2.要求T为2的幂次大小 满足内存对齐条件
template <typename T>
constexpr void CheckTypeSIMD() {
  CheckTypeSizeAligned<T>();
  CheckTypePOD<T>();
}
#endif  // __CHECK_SIMD_H__