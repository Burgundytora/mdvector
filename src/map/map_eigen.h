#ifndef __MDVECTOR_TO_EIGEN_H__
#define __MDVECTOR_TO_EIGEN_H__

#include <Eigen/Dense>

#include "src/mdvector/mdvector.h"

// mdvector转eigen 目前只能是row-major / layout-left
template <class T, size_t Rank>
auto map_mdvector_to_eigen_matrix(const mdvector<T, Rank> &mdv) {
  static_assert(Rank == 2, "only 2d mdvector can be mapped to Eigen matrix");
  size_t rows = mdv.extent(0);
  size_t cols = mdv.extent(1);
  return Eigen::Map<Eigen::>
}

template <class T, size_t Rank>
auto map_mdvector_to_eigen_tensor(const mdvector<T, Rank> &mdv) {
  //
}

#endif  // __MDVECTOR_TO_EIGEN_H__