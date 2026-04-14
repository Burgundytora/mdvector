#ifndef __MDVECTOR_ITERATOR_MIXIN__
#define __MDVECTOR_ITERATOR_MIXIN__

namespace md {

// 迭代器Mixin模板
template <typename Derived, typename T>
class iterator_mixin {
 public:
  Derived& derived() noexcept { return static_cast<Derived&>(*this); }
  const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

  // 假设派生类有 data() 和 size() 方法
  T* data() { return derived().data(); }
  const T* data() const { return derived().data(); }
  size_t size() const { return derived().size(); }

 public:
  using iterator = T*;
  using const_iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  iterator begin() noexcept { return derived().data(); }
  iterator end() noexcept { return derived().data() + derived().size(); }
  const_iterator begin() const noexcept { return derived().data(); }
  const_iterator end() const noexcept { return derived().data() + derived().size(); }
  const_iterator cbegin() const noexcept { return derived().data(); }
  const_iterator cend() const noexcept { return derived().data() + derived().size(); }
  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
  const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }
};

}  // namespace md

#endif  // __MDVECTOR_ITERATOR_MIXIN__