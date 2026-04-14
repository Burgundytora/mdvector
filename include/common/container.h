#ifndef __MDVECTOR_CONTAINER__
#define __MDVECTOR_CONTAINER__

#include <array>
#include <variant>
#include <vector>

#include "simd/allocator.h"
#include "simd/simd_function.h"
#include "base_concept.h"

// 用于mdvector的栈数组优化
// 经测试 性能相比直接vector有下降
// 可能是variant有开销 弃用

namespace md {

// 对齐的 array 包装器
template <typename T, size_t N>
struct aligned_array {
  alignas(64) std::array<T, N> data_;

  // 提供 array 的接口
  auto begin() noexcept { return data_.begin(); }
  auto end() noexcept { return data_.end(); }
  auto begin() const noexcept { return data_.begin(); }
  auto end() const noexcept { return data_.end(); }

  T& operator[](size_t i) noexcept { return data_[i]; }
  const T& operator[](size_t i) const noexcept { return data_[i]; }

  T& at(size_t i) noexcept { return data_.at(i); }
  const T& at(size_t i) const noexcept { return data_.at(i); }

  T* data() noexcept { return data_.data(); }
  const T* data() const noexcept { return data_.data(); }

  static constexpr size_t size() noexcept { return N; }
};

constexpr size_t ShortSize = 64;

template <typename T>
class variant_container {
 private:
  using stack_type = aligned_array<T, ShortSize>;
  using heap_type = std::vector<T, auto_allocator<T>>;

  std::variant<stack_type, heap_type> storage_;
  size_t size_ = 0;

 public:
  variant_container() = default;

  variant_container(size_t size) { allocate(size); }

  // 分配存储
  void allocate(size_t size) {
    size_ = size;
    if constexpr (!Numeric<T>) {
      storage_.template emplace<stack_type>();
    } else {
      if (size <= ShortSize) {
        storage_.template emplace<stack_type>();
      } else {
        storage_.template emplace<heap_type>(size);
      }
    }
  }

  // 访问数据
  T* data() noexcept {
    return std::visit(
        [](auto& storage) -> T* {
          if constexpr (std::is_same_v<std::decay_t<decltype(storage)>, stack_type>) {
            return storage.data();
          } else {
            return storage.data();
          }
        },
        storage_);
  }

  const T* data() const noexcept {
    return std::visit(
        [](const auto& storage) -> const T* {
          if constexpr (std::is_same_v<std::decay_t<decltype(storage)>, stack_type>) {
            return storage.data();
          } else {
            return storage.data();
          }
        },
        storage_);
  }

  // 状态查询
  bool using_stack() const noexcept { return std::holds_alternative<stack_type>(storage_); }

  // 调整大小
  void resize(size_t new_size) {
    if (new_size == size_) {
      return;
    }

    if (new_size <= ShortSize && using_stack()) {
      // 栈内调整
      size_ = new_size;
    } else if (new_size <= ShortSize && !using_stack()) {
      // 从堆切换到栈
      auto& heap_storage = std::get<heap_type>(storage_);
      stack_type new_stack;
      std::copy_n(heap_storage.data(), std::min(new_size, size_), new_stack.data());
      storage_.template emplace<stack_type>(std::move(new_stack));
      size_ = new_size;
    } else if (new_size > ShortSize && using_stack()) {
      // 从栈切换到堆
      auto& stack_storage = std::get<stack_type>(storage_);
      heap_type new_heap(new_size);
      std::copy_n(stack_storage.data(), size_, new_heap.data());
      storage_.template emplace<heap_type>(std::move(new_heap));
      size_ = new_size;
    } else {
      // 堆内调整
      std::get<heap_type>(storage_).resize(new_size);
      size_ = new_size;
    }
  }
};

}  // namespace md

#endif  // __MDVECTOR_CONTAINER__