#ifndef __MDVECTOR_UNROLL__
#define __MDVECTOR_UNROLL__

namespace md {

// 循环展开策略
struct unroll_1 {
  static constexpr size_t factor = 1;
};
struct unroll_2 {
  static constexpr size_t factor = 2;
};
struct unroll_4 {
  static constexpr size_t factor = 4;
};
struct unroll_8 {
  static constexpr size_t factor = 8;
};
struct unroll_16 {
  static constexpr size_t factor = 16;
};

// 默认策略：无循环展开
struct no_unroll {
  static constexpr size_t factor = 1;
};

// 自动选择循环展开因子
template <size_t Size>
struct auto_unroll {
  static constexpr size_t factor = (Size >= 1024) ? 8 : (Size >= 512) ? 4 : (Size >= 256) ? 2 : 1;
};

}  // namespace md

#endif  // __MDVECTOR_UNROLL__