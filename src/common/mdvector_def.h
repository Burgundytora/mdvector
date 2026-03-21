#ifndef __MDVECTOR_MDVECTOR_DEF__
#define __MDVECTOR_MDVECTOR_DEF__

#include <vector>

#include "mdspan_little.h"
#include "type_concept.h"

namespace md {

// 前向声明
template <typename T, size_t Rank, typename Layout = std::layout_right>
class vector;

}  // namespace md

#endif  // __MDVECTOR_MDVECTOR_DEF__