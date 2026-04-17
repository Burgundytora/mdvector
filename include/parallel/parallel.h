#ifndef __MDARRAY_PARALLEL__
#define __MDARRAY_PARALLEL__

#include "thread_pool.h"
#include "cuda.h"

namespace md {

// 数组太小，不值得并行化
constexpr size_t parallel_threshold = 5000000;

// ============================================
// 执行策略标签
// ============================================
struct sequential_t {};
struct parallel_t {
  size_t chunk_size = 0;  // 0 表示自动最大线程
};
struct cuda_t {};

inline constexpr sequential_t seq{};
inline constexpr parallel_t par{};
inline constexpr cuda_t cu{};

}  // namespace md

#endif  //__MDARRAY_PARALLEL__