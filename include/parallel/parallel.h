#ifndef __MDVECTOR_PARALLEL__
#define __MDVECTOR_PARALLEL__

#include "thread_pool.h"
#include "cuda.h"

namespace md {

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

#endif  //__MDVECTOR_PARALLEL__