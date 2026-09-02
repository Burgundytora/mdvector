#pragma once

#include "../concepts/storage_concept.h"

#include <cstdint>
#include <type_traits>

namespace md::detail {

template <typename Object>
size_t mapped_element_count(const Object& object) {
  if constexpr (requires { object.strides(); }) {
    const auto extents = object.extents();
    const auto strides = object.strides();
    size_t count = object.size() == 0 ? 0 : 1;
    for (size_t i = 0; i < Object::rank_; ++i) {
      if (extents[i] != 0) count += (extents[i] - 1) * strides[i];
    }
    return count;
  } else {
    return object.size();
  }
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
    const auto source_begin = reinterpret_cast<std::uintptr_t>(source.data());
    const auto dest_begin = reinterpret_cast<std::uintptr_t>(dest.data());
    const auto source_end = source_begin + mapped_element_count(source) * sizeof(typename Source::value_type);
    const auto dest_end = dest_begin + mapped_element_count(dest) * sizeof(typename Dest::value_type);
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
