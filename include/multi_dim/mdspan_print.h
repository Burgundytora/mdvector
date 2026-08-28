#pragma once

#include "mdspan_impl.h"

#include <print>

namespace md {

template <typename T, typename Extents, typename Layout>
void print_mdspan(std::mdspan<T, Extents, Layout> mdspan_) {
  constexpr size_t Rank = mdspan_.rank();
  if constexpr (Rank == 1) {
    // 1D 输出
    std::cout << "[";
    for (int i = 0; i < mdspan_.extent(0); ++i) {
      std::cout << mdspan_[i];
      if (i < mdspan_.extent(0) - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  } else if constexpr (Rank == 2) {
    // 2D 矩阵输出
    std::cout << "[\n";
    for (int i = 0; i < mdspan_.extent(0); ++i) {
      std::cout << "  [";
      for (size_t j = 0; j < mdspan_.extent(1); ++j) {
        std::cout << std::format("{:3}", mdspan_[i, j]);
        if (j < mdspan_.extent(1) - 1) {
          std::cout << ", ";
        }
      }
      std::cout << "]\n";
    }
    std::cout << "]" << std::endl;
  } else if constexpr (Rank == 3) {
    // 3D 张量输出
    std::cout << std::format("3D Tensor [{} x {} x {}]:\n", mdspan_.extent(0), mdspan_.extent(1), mdspan_.extent(2));

    for (int i = 0; i < mdspan_.extent(0); ++i) {
      std::cout << std::format("Layer {}:\n", i);
      std::cout << "  [\n";
      for (size_t j = 0; j < mdspan_.extent(1); ++j) {
        std::cout << "    [";
        for (size_t k = 0; k < mdspan_.extent(2); ++k) {
          std::cout << std::format("{:3}", mdspan_[i, j, k]);
          if (k < mdspan_.extent(2) - 1) {
            std::cout << ", ";
          }
        }
        std::cout << "]";
        if (j < mdspan_.extent(1) - 1) {
          std::cout << ",";
        }
        std::cout << "\n";
      }
      std::cout << "  ]";
      if (i < mdspan_.extent(0) - 1) {
        std::cout << ",";
      }
      std::cout << std::endl;
    }
  } else {
    // 更高维度输出
    std::cout << std::format("<{}D Tensor>: [", Rank);
    for (int i = 0; i < Rank; ++i) {
      std::cout << mdspan_.extent(i);
      if (i < Rank - 1) {
        std::cout << " x ";
      }
    }
    std::cout << "]" << std::endl;

    // 对于高维张量，显示前几个元素作为示例
    std::cout << "First few elements: ";
    size_t count = 0;
    constexpr size_t max_elements = 6;

    // 简单的扁平化遍历显示前几个元素
    for (int i = 0; i < mdspan_.size() && count < max_elements; ++i, ++count) {
      std::cout << *(mdspan_.data_handle() + i);
      if (i < mdspan_.size() - 1 && count < max_elements - 1) {
        std::cout << ", ";
      }
    }
    if (mdspan_.size() > max_elements) {
      std::cout << ", ...";
    }
    std::cout << std::endl;
  }
}

}  // namespace md
