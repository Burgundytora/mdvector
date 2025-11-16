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
  using layout_type = std::layout_stride;
  static constexpr size_t rank_ = Rank;

 protected:
  std::mdspan<T, std::dextents<size_t, Rank>, std::layout_stride> mdspan_;
  std::array<size_t, Rank> shape_;
  size_t size_;

 public:
  ///////////////////////////////////////////////////////////////////////////////////////
  /// 构造函数
  constexpr view() noexcept = default;

  view(T* data, const std::array<std::size_t, Rank>& shape, const std::array<std::size_t, Rank>& stride)
      : mdspan_(create_mdspan(data, shape, stride, std::make_index_sequence<Rank>{})),
        shape_(shape),
        size_(md::calculate_size(shape)) {}

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 打印
  void print()
    requires Printable<T>
  {
    if (!mdspan_.empty()) {
      md::print_mdspan(mdspan_);
    } else {
      throw std::logic_error("md::span need to be initialized before print!!!");
    }
  }

 private:
  ///////////////////////////////////////////////////////////////////////////////////////
  /// 内部函数
  template <size_t... Is>
  auto create_mdspan(T* data, const std::array<size_t, Rank>& shape, const std::array<std::size_t, Rank>& stride,
                     std::index_sequence<Is...>) {
    return std::mdspan<T, std::dextents<size_t, Rank>, std::layout_stride>(
        data,
        std::layout_stride::mapping<std::dextents<size_t, Rank>>(std::dextents<size_t, Rank>(shape[Is]...), stride));
  }
};

}  // namespace md

#endif  //__MDVECTOR_VIEW_H__