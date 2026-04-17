#ifndef __MDARRAY_STORAGE__
#define __MDARRAY_STORAGE__

#include "stack_storage.h"
#include "heap_storage.h"
#include "view_storage.h"

namespace md {

namespace detail {

// 编译期验证容器特性
template <typename T>
struct storage_checks {
  static_assert(BasicStorage<stack_storage<T, 3, 3>>);
  static_assert(BasicStorage<heap_storage<T>>);
  static_assert(BasicStorage<view_storage<T>>);

  static_assert(OwningStorage<stack_storage<T, 3, 3>>);
  static_assert(OwningStorage<heap_storage<T>>);
  static_assert(!OwningStorage<view_storage<T>>);

  static_assert(!ViewStorage<stack_storage<T, 3, 3>>);
  static_assert(!ViewStorage<heap_storage<T>>);
  static_assert(ViewStorage<view_storage<T>>);
};

// 验证数值类型
template struct storage_checks<float>;
template struct storage_checks<double>;
template struct storage_checks<int>;

}  // namespace detail

}  // namespace md

#endif  // __MDARRAY_STORAGE__