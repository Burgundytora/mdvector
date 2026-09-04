#pragma once

#include "../concepts/storage_concept.h"

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace md::detail {

template <typename Object>
std::pair<std::uintptr_t, std::uintptr_t> mapped_memory_region(const Object& object) {
  const auto base = reinterpret_cast<std::uintptr_t>(object.data());
  if (object.size() == 0) return {base, base};

  std::ptrdiff_t minimum_offset = 0;
  std::ptrdiff_t maximum_offset = 0;
  if constexpr (requires { object.strides(); }) {
    const auto extents = object.extents();
    const auto strides = object.strides();
    for (size_t i = 0; i < Object::rank_; ++i) {
      const auto delta = static_cast<std::ptrdiff_t>(extents[i] - 1) * strides[i];
      minimum_offset += std::min<std::ptrdiff_t>(0, delta);
      maximum_offset += std::max<std::ptrdiff_t>(0, delta);
    }
  } else {
    maximum_offset = static_cast<std::ptrdiff_t>(object.size() - 1);
  }

  constexpr size_t element_size = sizeof(typename Object::value_type);
  const auto address_at = [base](std::ptrdiff_t offset) {
    if (offset < 0)
      return base - static_cast<std::uintptr_t>(-offset) * element_size;
    return base + static_cast<std::uintptr_t>(offset) * element_size;
  };
  return {address_at(minimum_offset), address_at(maximum_offset) + element_size};
}

template <typename Source, typename Dest>
bool memory_regions_overlap(const Source& source, const Dest& dest) {
  if constexpr (!(requires {
                  source.data();
                  source.size();
                  dest.data();
                  dest.size();
                })) {
    return false;
  } else {
    const auto [source_begin, source_end] = mapped_memory_region(source);
    const auto [dest_begin, dest_end] = mapped_memory_region(dest);
    return source_begin < dest_end && dest_begin < source_end;
  }
}

template <typename Source, typename Dest>
bool has_same_mapping(const Source& source, const Dest& dest) {
  if constexpr (!(requires {
                  source.data();
                  source.extents();
                  dest.data();
                  dest.extents();
                })) {
    return false;
  } else if constexpr (Source::rank_ != Dest::rank_ ||
                       !std::is_same_v<typename Source::value_type, typename Dest::value_type>) {
    return false;
  } else {
    if (reinterpret_cast<const void*>(source.data()) != reinterpret_cast<const void*>(dest.data()) ||
        source.extents() != dest.extents())
      return false;
    if constexpr (requires { source.strides(); } || requires { dest.strides(); }) {
      if constexpr (requires {
                      source.strides();
                      dest.strides();
                    }) {
        return source.strides() == dest.strides();
      } else {
        return false;
      }
    } else if constexpr (Source::rank_ > 1 &&
                         !std::is_same_v<typename Source::layout_type, typename Dest::layout_type>) {
      return false;
    }
    return true;
  }
}

template <typename Source, typename Dest>
bool leaf_requires_temporary(const Source& source, const Dest& dest) noexcept {
  // Two owning containers cannot partially overlap: each allocation is either
  // independent, or source and destination are the same complete container.
  // The latter is safe for element-wise expressions. Keeping this as a
  // compile-time branch removes all alias bookkeeping from the common
  // vector-to-vector evaluation path, which is important for tiny arrays.
  if constexpr (OwningStorage<Source> && OwningStorage<Dest>) {
    return false;
  } else {
    return memory_regions_overlap(source, dest) && !has_same_mapping(source, dest);
  }
}

}  // namespace md::detail
