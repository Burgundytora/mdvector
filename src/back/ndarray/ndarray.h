#ifndef __NDARRAY_H__
#define __NDARRAY_H__

#include <algorithm>
#include <array>
#include <execution>
#include <functional>
#include <initializer_list>
#include <iosfwd>
#include <iostream>
#include <mdspan>
#include <numeric>
#include <vector>

using std::array;
using std::vector;

// STL执行策略 默认为串行乱序
// 对于特殊场景 上万级别数据 可以设置为并行乱序 par_unseq
// TODO: 构造和运行时可设置策略
#define exec_policy std::execution::unseq

// 多维POD+SIMD数据一维化
// 内存顺序IJK
// 构造之后无法更改维度 需重新构造
// TODO: 实现其他layout策略可设置
template <class T, size_t Dims>
struct _nd_array {
  // 类型别名定义
  using extents_type = std::dextents<size_t, Dims>;
  using layout_type = std::layout_right;
  using mdspan_type = std::mdspan<T, extents_type, layout_type>;

  // 成员
  vector<T, AlignedAllocator<T, alignment_>> data_;  // 一维数据存储 首地址对齐
  mdspan_type view_;                                 // 多维视图
  array<size_t, Dims> len_info_;                     // 纬度信息

  // ========================================================
  // 默认构造函数 检查类型SIMD
  _nd_array() { CheckTypeSIMD<T>(); }

  // 1. array构造函数优先使用 执行编译期维度检查 推荐通过类型别名 如_2d_shape
  explicit _nd_array(const array<size_t, Dims>& dims) {
    CheckTypeSIMD<T>();
    static_assert(sizeof(dims) / sizeof(size_t) == Dims, "_nd_array dimention error!");
    InitFromContainer(dims);
  }

  // // 2. 初始化列表构造函数  弃用 无法执行编译期维度检查
  // explicit _nd_array(std::initializer_list<size_t> dims) {
  //   // 无法编译期检查维度
  //   InitFromContainer(dims);
  // }

  //  3. 可变参数构造, 弃用 与其他构造函数存在冲突 且UB概率相比其他构造更高
  // template <typename... Sizes>
  // _nd_array(Sizes... lens) {
  //   // 维度匹配
  //   static_assert(sizeof...(lens) == Dims, "data dimention error!");
  //   InitFromContainer(array<size_t, Dims>{static_cast<size_t>(lens)...});
  // }
  // ========================================================

  // 构造后设置维度长度
  void SetShape(const array<size_t, Dims>& dims) {
    static_assert(sizeof(dims) / sizeof(size_t) == Dims, "_nd_array dimention error!");
    InitFromContainer(dims);
  }

  // ========================================================
  template <size_t... I>
  extents_type CreateExtents(std::index_sequence<I...>) {
    return extents_type{len_info_[I]...};
  }

  void InitFromContainer(const array<size_t, Dims>& dims) {
    // 存储维度信息
    len_info_ = dims;

    // 计算总长度
    const size_t total_len = std::accumulate(len_info_.begin(), len_info_.end(), 1, std::multiplies<>());

    // 初始化数据存储
    data_.resize(total_len);

    // 检查内存对齐
    VerifyAlignment();

    // 构造 extents（使用索引序列展开）
    auto extents = CreateExtents(std::make_index_sequence<Dims>{});
    view_ = mdspan_type(this->data_.data(), extents);
  }

  // 验证内存对齐
  void VerifyAlignment() const {
    if (reinterpret_cast<uintptr_t>(data_.data()) % alignment_ != 0) {
      std::cerr << "Memory alignment failed!" << std::endl;
      std::abort();
    }
  }

  void CheckDataDimSame(const _nd_array& a, const _nd_array& b) {
    if (a.len_info_.size() != b.len_info_.size()) {
      std::cerr << "data dim not same!\t left:" << a.len_info_.size() << "\t right:" << b.len_info_.size();
      std::abort();
    } else {
      for (size_t i = 0; i < a.len_info_.size(); i++) {
        if (a.len_info_[i] != b.len_info_[i]) {
          std::cerr << " dim no." << i + 1 << " length not same!\t left:" << a.len_info_[i]
                    << "\t right:" << b.len_info_[i];
          std::abort();
        }
      }
    }
  }
  // ========================================================

  // ========================================================
  // 用同一个值填充数据
  void SetValue(T val) { std::fill(data_.begin(), data_.end(), val); }

  // 显示各维度长度
  void ShowDimInfo() {
    std::cout << "dim len: ";
    for (auto it : len_info_) {
      std::cout << it << " ";
    }
    std::cout << "\n";
  }

  void ShowDataArrayStyle() {
    // std::cout << "data in array style:\n";
    for (const auto& it : this->data_) {
      std::cout << it << " ";
    }
    std::cout << "\n";
  }

  void ShowDataMatrixStyle() {
    if (Dims == 0) return;

    const size_t cols = len_info_.back();
    const size_t rows = data_.size() / cols;

    // std::cout << "data in matrix style:\n";
    for (size_t i = 0; i < rows; ++i) {
      const T* row_start = data_.data() + i * cols;
      for (size_t j = 0; j < cols; ++j) {
        std::cout << row_start[j] << " ";
      }
      std::cout << "\n";
    }
  }
  // ========================================================

  // ========================================================
  // 访问运算符 提供safe 和 unsafe两种方式
  // (i, j, k) unsafe style
  template <typename... Indices>
  T& operator()(Indices... indices) {
    static_assert(sizeof...(Indices) == Dims, "Number of indices must match dimensions");
    return view_[indices...];
  }

  // [i, j, k] unsafe style
  template <typename... Indices>
  T& operator[](Indices... indices) {
    static_assert(sizeof...(Indices) == Dims, "Number of indices must match dimensions");
    return view_[indices...];
  }

  // .at(i, j, k) safe style
  template <typename... Indices>
  T& at(Indices... indices) {
    static_assert(sizeof...(Indices) == Dims, "Number of indices must match dimensions");
    size_t i = 0;
    for (auto len : array<size_t, Dims>{static_cast<size_t>(indices)...}) {
      if (len > view_.extent(i)) {
        std::cerr << "mdspan out-of-range error: " << len << ">" << view_.extent(i) << "\n";
        std::abort();
      }
      i++;
    }
    return view_[indices...];
  }

  // 拷贝构造 使用size信息 重新构造成员
  _nd_array(const _nd_array& other) {
    data_.resize(other.data_.size());
    len_info_ = other.len_info_;
    // 重新创建mdsapn
    view_ = mdspan_type(data_.data(), CreateExtents(std::make_index_sequence<Dims>{}));
  }

  // 赋值构造 STL容器深拷贝 mspan重新构造
  _nd_array& operator=(const _nd_array& other) {
    data_ = other.data_;
    len_info_ = other.len_info_;
    // 重新创建mdsapn
    view_ = mdspan_type(data_.data(), CreateExtents(std::make_index_sequence<Dims>{}));
    return *this;
  }

  // ========================================================
  // 操作符重载 ndarray unsafe
  _nd_array operator+(const _nd_array& other) {
    _nd_array result(*this);
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), result.data_.begin(), std::plus<>());
    return result;
  }
  _nd_array& operator+=(const _nd_array& other) {
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), data_.begin(), std::plus<>());
    return *this;
  }

  _nd_array operator-(const _nd_array& other) {
    _nd_array result(*this);
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), result.data_.begin(), std::minus<>());
    return result;
  }
  _nd_array& operator-=(const _nd_array& other) {
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), data_.begin(), std::minus<>());
    return *this;
  }

  _nd_array operator*(const _nd_array& other) {
    _nd_array result(*this);
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), result.data_.begin(),
                   std::multiplies<>());
    return result;
  }
  _nd_array& operator*=(const _nd_array& other) {
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), data_.begin(), std::multiplies<>());
    return *this;
  }

  _nd_array operator/(const _nd_array& other) {
    _nd_array result(*this);
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), result.data_.begin(),
                   std::divides<>());
    return result;
  }
  _nd_array& operator/=(const _nd_array& other) {
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), data_.begin(), std::divides<>());
    return *this;
  }

  _nd_array operator%(const _nd_array& other) {
    _nd_array result(*this);
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), result.data_.begin(),
                   std::modulus<>());
    return result;
  }
  _nd_array& operator%=(const _nd_array& other) {
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), data_.begin(), std::modulus<>());
    return *this;
  }
  // ========================================================

  // ========================================================
  // 基础运算 函数形式 safe style   eg. auto data3 = data1.Plus(data2)
  _nd_array Plus(const _nd_array& other) {
    CheckDataDimSame(*this, other);
    _nd_array result(*this);
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), result.data_.begin(), std::plus<>());
    return result;
  }

  void PlusEqual(const _nd_array& other) {
    CheckDataDimSame(*this, other);
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), data_.begin(), std::plus<>());
    return;
  }

  _nd_array Minus(const _nd_array& other) {
    CheckDataDimSame(*this, other);
    _nd_array result(*this);
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), result.data_.begin(), std::minus<>());
    return result;
  }

  void MinusEqual(const _nd_array& other) {
    CheckDataDimSame(*this, other);
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), data_.begin(), std::minus<>());
    return;
  }

  _nd_array Multiplies(const _nd_array& other) {
    CheckDataDimSame(*this, other);
    _nd_array result(*this);
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), result.data_.begin(),
                   std::multiplies<>());
    return result;
  }

  void MultipliesEqual(const _nd_array& other) {
    CheckDataDimSame(*this, other);
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), data_.begin(), std::multiplies<>());
    return;
  }

  _nd_array Divides(const _nd_array& other) {
    CheckDataDimSame(*this, other);
    _nd_array result(*this);
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), result.data_.begin(),
                   std::divides<>());
    return result;
  }

  void DividesEqual(const _nd_array& other) {
    CheckDataDimSame(*this, other);
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), data_.begin(), std::divides<>());
    return;
  }

  _nd_array Modulus(const _nd_array& other) {
    CheckDataDimSame(*this, other);
    _nd_array result(*this);
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), result.data_.begin(),
                   std::modulus<>());
    return result;
  }

  void ModulusEqual(const _nd_array& other) {
    CheckDataDimSame(*this, other);
    std::transform(exec_policy, data_.begin(), data_.end(), other.data_.begin(), data_.begin(), std::modulus<>());
    return;
  }
  // ========================================================

  // ========================================================
  // 操作符重载 基础类型 ，天然safe 因此不提供函数形式
  _nd_array operator+(const T& val) {
    _nd_array result = *this;
    result.data_ = this->data_;
    int index = 0;
    for (auto& it : result.data_) {
      it += val;
      index++;
    }
    return result;
  }

  _nd_array& operator+=(const T& val) {
    for (auto& it : data_) {
      it += val;
    }
    return *this;
  }

  _nd_array operator-(const T& val) {
    _nd_array result = *this;
    result.data_ = this->data_;
    int index = 0;
    for (auto& it : result.data_) {
      it -= val;
      index++;
    }
    return result;
  }

  _nd_array& operator-=(const T& val) {
    for (auto& it : data_) {
      it -= val;
    }
    return *this;
  }

  _nd_array operator*(const T& val) {
    _nd_array result = *this;
    result.data_ = this->data_;
    int index = 0;
    for (auto& it : result.data_) {
      it *= val;
      index++;
    }
    return result;
  }

  _nd_array& operator*=(const T& val) {
    for (auto& it : data_) {
      it *= val;
    }
    return *this;
  }

  _nd_array operator/(const T& val) {
    _nd_array result = *this;
    result.data_ = this->data_;
    int index = 0;
    for (auto& it : result.data_) {
      it /= val;
      index++;
    }
    return result;
  }

  _nd_array& operator/=(const T& val) {
    for (auto& it : data_) {
      it /= val;
    }
    return *this;
  }

  _nd_array operator%(const T& val) {
    _nd_array result = *this;
    result.data_ = this->data_;
    int index = 0;
    for (auto& it : result.data_) {
      it %= val;
      index++;
    }
    return result;
  }

  _nd_array& operator%=(const T& val) {
    for (auto& it : data_) {
      it %= val;
    }
    return *this;
  }
};

// ========================================================
// 维度  array别名
template <size_t n>
using _nd_shape = array<size_t, n>;

// 常用长度 一维到四维
template <typename T>
using _1d_array = _nd_array<T, 1>;
using _1d_shape = _nd_shape<1>;

template <typename T>
using _2d_array = _nd_array<T, 2>;
using _2d_shape = _nd_shape<2>;

template <typename T>
using _3d_array = _nd_array<T, 3>;
using _3d_shape = _nd_shape<3>;

template <typename T>
using _4d_array = _nd_array<T, 4>;
using _4d_shape = _nd_shape<4>;

// ========================================================
// 常用类型 1~4维
using _1d_array_double = _1d_array<double>;
using _1d_array_float = _1d_array<float>;
using _1d_array_int = _1d_array<int>;
using _1d_array_size = _1d_array<size_t>;
using _1d_array_bool = _1d_array<bool>;
using _1d_array_char = _1d_array<char>;
using _1d_array_wchar = _1d_array<wchar_t>;

using _2d_array_double = _2d_array<double>;
using _2d_array_float = _2d_array<float>;
using _2d_array_int = _2d_array<int>;
using _2d_array_size = _2d_array<size_t>;
using _2d_array_bool = _2d_array<bool>;
using _2d_array_char = _2d_array<char>;
using _2d_array_wchar = _2d_array<wchar_t>;

using _3d_array_double = _3d_array<double>;
using _3d_array_float = _3d_array<float>;
using _3d_array_int = _3d_array<int>;
using _3d_array_size = _3d_array<size_t>;
using _3d_array_bool = _3d_array<bool>;
using _3d_array_char = _3d_array<char>;
using _3d_array_wchar = _3d_array<wchar_t>;

using _4d_array_double = _4d_array<double>;
using _4d_array_float = _4d_array<float>;
using _4d_array_int = _4d_array<int>;
using _4d_array_size = _4d_array<size_t>;
using _4d_array_bool = _4d_array<bool>;
using _4d_array_char = _4d_array<char>;
using _4d_array_wchar = _4d_array<wchar_t>;
// ========================================================

#endif  // __NDARRAY_H__