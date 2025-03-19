
include_directories("${PROJECT_SOURCE_DIR}")

add_executable(test_2d test_2d.cc)
add_executable(base_expr "${PROJECT_SOURCE_DIR}/src/base_expr/base_expr.cc")

add_subdirectory(single)
