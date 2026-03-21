#ifndef __MDVECTOR_MDSPAN_LITTLE__
#define __MDVECTOR_MDSPAN_LITTLE__

#include <version>

// 检查标准库 mdspan 支持
#if __has_include(<mdspan>) && defined(__cpp_lib_mdspan)

#include <mdspan>

#else

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace std {

// 内存布局策略
struct layout_right {
  template <typename Extents>
  class mapping;
};

struct layout_left {
  template <typename Extents>
  class mapping;
};

struct layout_stride {
  template <typename Extents>
  class mapping;
};

// extents 实现
template <typename IndexType, size_t... Extents>
class extents {
 public:
  using index_type = IndexType;
  using size_type = std::size_t;
  using rank_type = size_t;

  static constexpr rank_type rank() { return sizeof...(Extents); }

  constexpr extents() noexcept = default;

  constexpr index_type extent(size_t r) const noexcept { return extents_[r]; }

  constexpr auto get_shape() { return extents_; }

 private:
  std::array<index_type, rank()> extents_{static_cast<index_type>(Extents)...};
};

// dextents 实现
template <typename IndexType, size_t Rank>
class dextents {
 public:
  using index_type = IndexType;
  using size_type = std::size_t;
  using rank_type = size_t;

  static constexpr rank_type rank() { return Rank; }

  constexpr dextents() = default;

  template <typename... Extents>
  constexpr dextents(Extents... ext) {
    static_assert(Rank == sizeof...(Extents), "Number of indices must match rank");
    extents_ = {static_cast<index_type>(ext)...};
  }

  constexpr index_type extent(size_t r) const noexcept { return extents_[r]; }

  template <size_t... Extents>
  void set_extents() {
    static_assert(Rank == sizeof...(Extents), "Number of indices must match rank");
    extents_ = {static_cast<index_type>(Extents)...};
  }

  constexpr auto get_shape() { return extents_; }

 private:
  std::array<index_type, Rank> extents_{};
};

// Layout 映射
template <typename Extents>
class layout_right::mapping {
 public:
  using extents_type = Extents;
  using index_type = typename extents_type::index_type;
  using size_type = typename extents_type::size_type;
  using rank_type = typename extents_type::rank_type;

  constexpr mapping() noexcept = default;
  constexpr mapping(const extents_type& ext) noexcept : extents_(ext) {}

  constexpr const extents_type& extents() const noexcept { return extents_; }

  template <typename... Indices>
  constexpr index_type operator()(Indices... indices) const noexcept {
    static_assert(sizeof...(Indices) == extents_type::rank(), "Number of indices must match rank");
    return compute_index(indices...);
  }

 private:
  extents_type extents_;

  template <typename... Indices>
  constexpr index_type compute_index(Indices... indices) const {
    const std::array<index_type, sizeof...(Indices)> idxs{static_cast<index_type>(indices)...};
    index_type result = 0;
    index_type stride = 1;

    for (rank_type i = extents_type::rank(); i > 0; --i) {
      result += idxs[i - 1] * stride;
      stride *= extents_.extent(i - 1);
    }

    return result;
  }
};

template <typename Extents>
class layout_left::mapping {
 public:
  using extents_type = Extents;
  using index_type = typename extents_type::index_type;
  using size_type = typename extents_type::size_type;
  using rank_type = typename extents_type::rank_type;

  constexpr mapping() noexcept = default;
  constexpr mapping(const extents_type& ext) noexcept : extents_(ext) {}

  constexpr const extents_type& extents() const noexcept { return extents_; }

  template <typename... Indices>
  constexpr index_type operator()(Indices... indices) const noexcept {
    static_assert(sizeof...(Indices) == extents_type::rank(), "Number of indices must match rank");
    return compute_index(indices...);
  }

 private:
  extents_type extents_;

  template <typename... Indices>
  constexpr index_type compute_index(Indices... indices) const {
    const std::array<index_type, sizeof...(Indices)> idxs{static_cast<index_type>(indices)...};
    index_type result = 0;
    index_type stride = 1;

    for (rank_type i = 0; i < extents_type::rank(); ++i) {
      result += idxs[i] * stride;
      stride *= extents_.extent(i);
    }

    return result;
  }
};

// layout_stride 映射实现
template <typename Extents>
class layout_stride::mapping {
 public:
  using extents_type = Extents;
  using index_type = typename extents_type::index_type;
  using size_type = typename extents_type::size_type;
  using rank_type = typename extents_type::rank_type;

  constexpr mapping() noexcept = default;

  // 从 extents 和 strides 构造
  constexpr mapping(const extents_type& ext, const std::array<index_type, extents_type::rank()>& strides) noexcept
      : extents_(ext), strides_(strides) {}

  // 从其他映射构造（简化版本）
  template <typename OtherMapping>
  constexpr mapping(const OtherMapping& other) noexcept : extents_(other.extents()) {
    for (rank_type i = 0; i < extents_type::rank(); ++i) {
      strides_[i] = other.stride(i);
    }
  }

  constexpr const extents_type& extents() const noexcept { return extents_; }
  constexpr const std::array<index_type, extents_type::rank()>& strides() const noexcept { return strides_; }

  template <typename... Indices>
  constexpr index_type operator()(Indices... indices) const noexcept {
    static_assert(sizeof...(Indices) == extents_type::rank(), "Number of indices must match rank");
    return compute_index(indices...);
  }

  constexpr index_type required_span_size() const noexcept {
    index_type max_index = 0;
    for (rank_type i = 0; i < extents_type::rank(); ++i) {
      max_index += (extents_.extent(i) - 1) * strides_[i];
    }
    return max_index + 1;
  }

  static constexpr bool is_always_unique() noexcept { return true; }
  static constexpr bool is_always_exhaustive() noexcept { return false; }
  static constexpr bool is_always_strided() noexcept { return true; }

  constexpr bool is_unique() const noexcept { return true; }
  constexpr bool is_exhaustive() const noexcept {
    // 简化检查：如果步长是连续的，则是 exhaustive
    index_type expected_stride = 1;
    for (rank_type i = extents_type::rank() - 1; i < extents_type::rank(); --i) {
      if (strides_[i] != expected_stride) {
        return false;
      }
      expected_stride *= extents_.extent(i);
    }
    return true;
  }
  constexpr bool is_strided() const noexcept { return true; }

  constexpr index_type stride(rank_type r) const noexcept { return strides_[r]; }

 private:
  extents_type extents_;
  std::array<index_type, extents_type::rank()> strides_{};

  template <typename... Indices>
  constexpr index_type compute_index(Indices... indices) const {
    const std::array<index_type, sizeof...(Indices)> idxs{static_cast<index_type>(indices)...};
    index_type result = 0;

    for (rank_type i = 0; i < extents_type::rank(); ++i) {
      result += idxs[i] * strides_[i];
    }

    return result;
  }
};

// 默认访问器
template <typename ElementType>
class default_accessor {
 public:
  using offset_policy = default_accessor;
  using element_type = ElementType;
  using reference = ElementType&;
  using data_handle_type = ElementType*;

  constexpr default_accessor() noexcept = default;

  template <typename OtherElementType>
  constexpr default_accessor(const default_accessor<OtherElementType>&) noexcept {}

  constexpr reference access(data_handle_type p, size_t i) const noexcept { return p[i]; }

  constexpr data_handle_type offset(data_handle_type p, size_t i) const noexcept { return p + i; }
};

// 主要的 mdspan 类
template <typename ElementType, typename Extents, typename LayoutPolicy = layout_right,
          typename AccessorPolicy = default_accessor<ElementType> >
class mdspan {
 public:
  using element_type = ElementType;
  using reference = ElementType&;
  using data_handle_type = ElementType*;
  using value_type = std::remove_cv_t<ElementType>;
  using index_type = typename Extents::index_type;
  using size_type = typename Extents::size_type;
  using rank_type = typename Extents::rank_type;
  using layout_type = LayoutPolicy;
  using accessor_type = AccessorPolicy;
  using mapping_type = typename layout_type::template mapping<Extents>;
  using extents_type = Extents;

  // 构造函数
  constexpr mdspan() noexcept : ptr_(nullptr), mapping_(), accessor_() {}

  // dextent
  template <typename... OtherIndexTypes>
  constexpr mdspan(data_handle_type p, OtherIndexTypes... exts) noexcept
      : ptr_(p), mapping_(extents_type(exts...)), accessor_() {}

  // extent
  constexpr mdspan(data_handle_type p) noexcept : ptr_(p), mapping_(extents_type()), accessor_() {}

  constexpr mdspan(data_handle_type p, const mapping_type& m) noexcept : ptr_(p), mapping_(m), accessor_() {}

  constexpr mdspan(data_handle_type p, const mapping_type& m, const accessor_type& a) noexcept
      : ptr_(p), mapping_(m), accessor_(a) {}

  // 访问操作
  template <typename... Indices>
  constexpr reference operator()(Indices... indices) const noexcept {
    return accessor_.access(ptr_, mapping_(indices...));
  }

  template <typename... Indices>
  constexpr reference operator[](Indices... indices) const noexcept {
    return (*this)(indices...);
  }

  // 属性访问
  static constexpr rank_type rank() noexcept { return extents_type::rank(); }
  static constexpr rank_type rank_dynamic() noexcept { return extents_type::rank_dynamic(); }
  static constexpr size_type static_extent(rank_type r) noexcept { return extents_type::static_extent(r); }

  constexpr const extents_type& extents() const noexcept { return mapping_.extents(); }
  constexpr index_type extent(rank_type r) const noexcept { return mapping_.extents().extent(r); }
  constexpr size_type size() const noexcept {
    size_type result = 1;
    for (rank_type i = 0; i < rank(); ++i) {
      result *= extent(i);
    }
    return result;
  }

  constexpr const mapping_type& mapping() const noexcept { return mapping_; }
  constexpr const accessor_type& accessor() const noexcept { return accessor_; }
  constexpr data_handle_type data_handle() const noexcept { return ptr_; }

  constexpr bool empty() const noexcept { return size() == 0; }

  // // 步长相关方法（对于 strided 布局）
  // constexpr index_type stride(rank_type r) const requires { mapping_.stride(r); } { return mapping_.stride(r); }

 private:
  data_handle_type ptr_;
  mapping_type mapping_;
  accessor_type accessor_;
};

}  // namespace std

#endif

#endif  // __MDVECTOR_MDSPAN_LITTLE__