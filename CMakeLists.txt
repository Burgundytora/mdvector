cmake_minimum_required(VERSION 3.15)
project(mdvector VERSION 0.1.0 LANGUAGES CXX)

# 基础配置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含自定义模块
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PlatformSetting)
include(DetectSIMD)
include(CompilerOption)

# 依赖项配置
if(WIN32)
    set(INCLUDE_DIR "D:/dependency/include")
else()
    set(INCLUDE_DIR "/mnt/goldflex/dependency/include")
endif()

# 用于比较性能
include_directories(
    "${INCLUDE_DIR}/eigen-3.4.0"
    "${INCLUDE_DIR}/highway"
    "${INCLUDE_DIR}/xtensor/include"
    "${INCLUDE_DIR}/xtl/include"
)

# 添加子目录
add_subdirectory(src)
