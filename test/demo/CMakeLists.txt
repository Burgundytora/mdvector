
include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(test_cpp23 test_cpp23.cc)
target_compile_features(test_cpp23 PRIVATE cxx_std_23)
