#ifndef MDVECTOR_SPAN_SUBVIEW_H_
#define MDVECTOR_SPAN_SUBVIEW_H_

#include "mdspan.h"

// 子视图 不支持表达式计算 但支持跨步等python风格切片
template <typename T, size_t Rank, typename Layout = layout_right>
class subview : public mdspan {};

#endif  // MDVECTOR_SPAN_SUBVIEW_H_