#ifndef __MDVECTOR_SPAN_SUBVIEW_H__
#define __MDVECTOR_SPAN_SUBVIEW_H__

#include "mdspan.h"

// 子视图 不支持simd表达式计算 但支持跨步等python风格切片
template <typename T, size_t Rank, typename Layout = layout_right>
class subview {};

#endif  // MDVECTOR_SPAN_SUBVIEW_H_