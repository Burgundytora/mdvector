cmake_minimum_required(VERSION 3.15)
project(mdvector VERSION 1.0.0 LANGUAGES CXX)

# 基础配置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含自定义模块
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PlatformSetting)
include(CompilerOption)
include(SelectSimd)

# 添加子目录
include_directories(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(test/correct)

# 测试
option(BUILD_SPEED_TEST "" ON)
if(BUILD_SPEED_TEST)
    if(WIN32)
        set(INCLUDE_DIR "G:/third_party/include")
    else()
        set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/include")
    endif()
    include_directories(
        "${INCLUDE_DIR}/eigen"
        "${INCLUDE_DIR}/highway"
        "${INCLUDE_DIR}/xtensor/include"
        "${INCLUDE_DIR}/xtl/include"
        "${INCLUDE_DIR}/xsimd/include"
    )
    include_directories(${PROJECT_SOURCE_DIR}/test)
    add_subdirectory(test/speed)
endif()
