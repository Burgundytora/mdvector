#pragma once

#include "../simd/simd.h"
#include "../parallel/parallel.h"

namespace md {

template <typename Derived, typename T, typename ExecutionPolicy = sequential_t>
class base_expr {
 public:
  Derived& derived() noexcept { return static_cast<Derived&>(*this); }
  const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

  size_t used_size() const noexcept { return derived().used_size(); }
  size_t size() const noexcept { return derived().size(); }

  auto extents() const noexcept { return derived().extents(); }

  // Linear scalar access shared by containers, views, and expression nodes.
  // It intentionally goes through the existing SIMD load interface so rank-N
  // containers do not need a separate one-dimensional operator[].
  T scalar_at(size_t i) const noexcept {
    constexpr size_t pack_size = simd<T>::pack_size;
    alignas(simd<T>::alignment) std::array<T, pack_size> values{};
    const size_t pack_begin = (i / pack_size) * pack_size;
    simd<T>::store(values.data(), derived().template load_simd<T>(pack_begin));
    return values[i - pack_begin];
  }

  // 默认顺序求值
  template <typename Dest>
  void eval_to(Dest& dest) const noexcept {
    eval_to_impl(dest, sequential_t{});
  }

  // 带执行策略的求值
  template <typename Dest, typename Policy>
  void eval_to(Dest& dest, Policy&& policy) const noexcept {
    eval_to_impl(dest, std::forward<Policy>(policy));
  }

 private:
  template <typename Dest>
  void eval_to_impl(Dest& dest, sequential_t) const noexcept {
    const size_t n = used_size();
    constexpr size_t pack_size = simd<T>::pack_size;

    for (size_t i = 0; i + pack_size <= n; i += pack_size) {
      auto simd_val = derived().template load_simd<T>(i);
      dest.template store_simd<T>(i, simd_val);
    }
  }

  // 并行实现（使用 jthread 线程池）
  template <typename Dest>
  void eval_to_impl(Dest& dest, parallel_t policy) const noexcept {
    const size_t n = used_size();

    if (n < parallel_threshold) {
      eval_to_impl(dest, sequential_t{});
      return;
    }

    constexpr size_t pack_size = simd<T>::pack_size;
    const size_t num_packs = n / pack_size;

    // 确定 chunk 大小
    size_t chunk_packs = policy.chunk_size;
    if (chunk_packs == 0) {
      const size_t num_workers = thread_pool::instance().num_workers();
      chunk_packs = (num_packs + num_workers - 1) / num_workers;
      // 至少保证每个 chunk 有一定的工作量
      constexpr size_t min_packs_per_chunk = 64;
      chunk_packs = std::max(chunk_packs, min_packs_per_chunk);
    }

    // 并行处理所有完整的 SIMD packs
    thread_pool::instance().parallel_for(
        num_packs,
        [this, &dest, pack_size](size_t /*worker_id*/, size_t start_pack, size_t end_pack) {
          for (size_t p = start_pack; p < end_pack; ++p) {
            size_t i = p * pack_size;
            auto simd_val = this->derived().template load_simd<T>(i);
            dest.template store_simd<T>(i, simd_val);
          }
        },
        chunk_packs);
  }
};

}  // namespace md
