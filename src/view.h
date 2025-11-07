#ifndef __MDVECTOR_VIEW_H__
#define __MDVECTOR_VIEW_H__

#include "common/detail.h"
#include "common/type_concept.h"
#include "expression_template/operator.h"
#include "simd/simd_function.h"

namespace md {

// 带步长功能 不要求内存连续视图 基于std::layout_stride
template <typename T, size_t Rank>
class view : public md::tensor_expr<view<T, Rank>, T> {
 public:
  using Policy = md::unaligned_policy;
  using value_type = T;
  using layout_type = Layout;
  static constexpr size_t rank_ = Rank;

 protected:
  std::mdspan<T, std::dextents<size_t, Rank>, std::layout_stride> mdspan_;
  std::array<size_t, Rank> shape_;

 public:
 private:
};

}  // namespace md

#endif  //__MDVECTOR_VIEW_H__