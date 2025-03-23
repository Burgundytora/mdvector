cmake_minimum_required(VERSION 3.15)
project(mdvector VERSION 0.1.0 LANGUAGES CXX)

# 设置标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    set(INCLUDE_DIR "D:/dependency/include")
else()
    set(INCLUDE_DIR "/mnt/goldflex/dependency/include")
endif()

include_directories("${INCLUDE_DIR}/eigen-3.4.0")
include_directories("${INCLUDE_DIR}/highway")
include_directories("${INCLUDE_DIR}/xtensor/include")
include_directories("${INCLUDE_DIR}/xtl/include")

# 指令集选项: X86: AVX2 / AVX512   ARM: NEON
set(MY_OPTION "AVX2" CACHE STRING "Choose between AVX2, AVX512, NEON")
set_property(CACHE MY_OPTION PROPERTY STRINGS "AVX2" "AVX512" "NEON")

# 根据选项定义宏
if(MY_OPTION STREQUAL "AVX2")
    add_compile_definitions(USE_AVX2)
elseif(MY_OPTION STREQUAL "AVX512")
    add_compile_definitions(USE_AVX512)
elseif(MY_OPTION STREQUAL "NEON")
    add_compile_definitions(USE_NEON)
endif()

# 平台信息
if(WIN32)
    if(CMAKE_CL_64)
        set(PLATFORM_INFO "Win64")
    else()
        set(PLATFORM_INFO "Win32")
    endif()
elseif(UNIX)
    set(PLATFORM_INFO "Linux")
elseif(IOS)
    set(PLATFORM_INFO "macOS")
endif()

message(STATUS "platform info: ${PLATFORM_INFO}")
message(STATUS "build type: ${CMAKE_BUILD_TYPE}")
set(GENERATOR_PATH "${PLATFORM_INFO}_${CMAKE_BUILD_TYPE}")
set(PROJECT_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/out/${GENERATOR_PATH}")

# 设置程序编译生成路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# vs 编译时关闭 msvc 自定义的 max/min 宏，使用标准库中的函数
if(MSVC)
    add_compile_options(/w)
    add_compile_options(/bigobj)
    add_compile_options(/arch:AVX2)
    add_definitions("-D_CRT_SECURE_NO_DEPRECATE") # avoids deprecation warnings
    add_definitions("-D_SCL_SECURE_NO_DEPRECATE") # avoids deprecation warnings
    add_definitions("-DNOMINMAX") # do not use MSVC's min/max macros
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /MANIFEST:NO")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /MANIFEST:NO")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /MANIFEST:NO")
else()
    add_compile_options("-Werror")
    add_definitions("-mavx2 -mfma -flto")
endif()

add_subdirectory(src)
