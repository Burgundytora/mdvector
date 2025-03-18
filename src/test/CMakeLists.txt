
include_directories("${PROJECT_SOURCE_DIR}")

add_executable(avx2 avx2.cc)
add_executable(eigen eigen.cc)
add_executable(vector vector.cc)
add_executable(norm norm.cc)
add_executable(mdvector_fun mdvector_fun.cc)
add_executable(mdvector_expr mdvector_expr.cc)
add_executable(test_2d test_2d.cc)