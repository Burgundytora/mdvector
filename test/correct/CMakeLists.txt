
include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(test_base test_base.cc)
add_executable(test_scalar test_scalar.cc)
add_executable(test_span test_span.cc)
add_executable(test_stl test_stl.cc)
add_executable(test_math test_math.cc)
add_executable(test_layout test_layout.cc)
