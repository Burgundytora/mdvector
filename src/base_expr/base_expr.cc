#include <cassert>
#include <cstddef>
#include <iostream>

using std::size_t;

template <class T>
struct Scalar;

template <class T>
struct Traits {
  using ExprRef = T const&;
};

template <class T>
struct Traits<Scalar<T>> {
  using ExprRef = Scalar<T>;
};

template <class T, class OP1, class OP2>
struct Add {
 private:
  typename Traits<OP1>::ExprRef op1_;
  typename Traits<OP2>::ExprRef op2_;

 public:
  Add(OP1 const& l, OP2 const& r) : op1_(l), op2_(r) {}

  T operator[](size_t idx) const { return op1_[idx] + op2_[idx]; }

  size_t size() const { return op1_.size(); }
};

template <class T, class OP1, class OP2>
struct Sub {
 private:
  typename Traits<OP1>::ExprRef op1_;
  typename Traits<OP2>::ExprRef op2_;

 public:
  Sub(OP1 const& l, OP2 const& r) : op1_(l), op2_(r) {}

  T operator[](size_t idx) const { return op1_[idx] - op2_[idx]; }

  size_t size() const { return op1_.size(); }
};

template <class T, class OP1, class OP2>
struct Mul {
 private:
  typename Traits<OP1>::ExprRef op1_;
  typename Traits<OP2>::ExprRef op2_;

 public:
  Mul(OP1 const& l, OP2 const& r) : op1_(l), op2_(r) {}

  T operator[](size_t idx) const { return op1_[idx] * op2_[idx]; }

  size_t size() const { return op1_.size(); }
};

template <class T, class OP1, class OP2>
struct Div {
 private:
  typename Traits<OP1>::ExprRef op1_;
  typename Traits<OP2>::ExprRef op2_;

 public:
  Div(OP1 const& l, OP2 const& r) : op1_(l), op2_(r) {}

  T operator[](size_t idx) const { return op1_[idx] / op2_[idx]; }

  size_t size() const { return op1_.size(); }
};

// scalar
template <class T>
struct Scalar {
 private:
  T const& s;

 public:
  Scalar(T const& v) : s(v) {}

  constexpr T const& operator[](size_t) const { return s; }

  constexpr size_t size() const { return 0; }
};

template <class T>
struct SArray {
  T* storage_;
  size_t storage_size_;

  SArray() = delete;

  SArray(size_t s) : storage_(new T[s]), storage_size_(s) { init(); }

  void init() {
    for (size_t idx = 0; idx < size(); idx++) {
      storage_[idx] = T();
    }
  }

  ~SArray() { delete[] storage_; }

  //   // copy construct
  //   SArray(SArray<T> const& orig) : storage_(new T[orig.size()]), storage_size_(orig.size()) { copy(orig); }

  SArray<T> operator=(SArray<T> const orig) {
    if (&orig != this) {
      copy(orig);
    }
    return *this;
  }

  size_t size() const { return storage_size_; }

  T const& operator[](size_t idx) const { return storage_[idx]; }

  T& operator[](size_t idx) { return storage_[idx]; }

  void copy(SArray<T> const orig) {
    for (size_t idx = 0; idx < orig.size(); idx++) {
      storage_[idx] = orig.storage_[idx];
    }
  }
};

#include <vector>
using std::vector;

template <class T, class Rep = SArray<T>>
struct Array {
 private:
  Rep expr_rep_;

 public:
  Array() = delete;

  explicit Array(Rep const& expr) : expr_rep_(expr) {}  // 新增

  explicit Array(size_t s) : expr_rep_(s) {}

  size_t size() const { return expr_rep_.size(); }

  T* data() { return expr_rep_.data(); }

  Array& operator=(Array const& b) {
    for (size_t idx = 0; idx < b.size(); idx++) {
      expr_rep_[idx] = b[idx];
    }
    return *this;
  }

  template <class Rep2>
  Array& operator=(Array<T, Rep2> const& b) {
    for (size_t idx = 0; idx < b.size(); idx++) {
      expr_rep_[idx] = b[idx];
    }
    return *this;
  }

  decltype(auto) operator[](size_t idx) const { return expr_rep_[idx]; }

  T& operator[](size_t idx) { return expr_rep_[idx]; }

  Rep const& rep() const { return expr_rep_; }

  Rep& rep() { return expr_rep_; }
};

template <class T, class R1, class R2>
Array<T, Add<T, R1, R2>> operator+(Array<T, R1> const& a, Array<T, R2> const& b) {
  return Array<T, Add<T, R1, R2>>(Add<T, R1, R2>(a.rep(), b.rep()));
}

template <class T, class R1, class R2>
Array<T, Sub<T, R1, R2>> operator-(Array<T, R1> const& a, Array<T, R2> const& b) {
  return Array<T, Sub<T, R1, R2>>(Sub<T, R1, R2>(a.rep(), b.rep()));
}

template <class T, class R1, class R2>
Array<T, Mul<T, R1, R2>> operator*(Array<T, R1> const& a, Array<T, R2> const& b) {
  return Array<T, Mul<T, R1, R2>>(Mul<T, R1, R2>(a.rep(), b.rep()));
}

template <class T, class R1, class R2>
Array<T, Div<T, R1, R2>> operator/(Array<T, R1> const& a, Array<T, R2> const& b) {
  return Array<T, Div<T, R1, R2>>(Div<T, R1, R2>(a.rep(), b.rep()));
}

int main(int args, char* argv[]) {
  size_t total_element = 1000;

  Array<double> data1_(total_element);
  Array<double> data2_(total_element);
  Array<double> data3_(total_element);

  // 赋值
  for (size_t i = 0; i < total_element; i++) {
    data1_[i] = 1;
    data2_[i] = 2;
    data3_[i] = 3;
  }

  data3_ = data1_ + data2_;
  data3_ = data1_ - data2_;
  data3_ = data1_ * data2_;
  data3_ = data1_ / data2_;

  std::cout << "data1: " << data1_[0] << "\n";
  std::cout << "data2: " << data2_[0] << "\n";
  std::cout << "data3: " << data3_[0] << "\n";

  return 0;
}
