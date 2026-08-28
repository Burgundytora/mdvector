#pragma once

#include "base_concept.h"

namespace md {

// ============================================================================
// 可迭代概念
// ============================================================================
template <typename S>
concept Iterable = requires(S& s, const S& cs) {
  { s.begin() } -> std::same_as<typename S::value_type*>;
  { s.end() } -> std::same_as<typename S::value_type*>;
  { cs.begin() } -> std::same_as<const typename S::value_type*>;
  { cs.end() } -> std::same_as<const typename S::value_type*>;
};

}  // namespace md
