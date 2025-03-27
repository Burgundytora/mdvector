# 指令集选项
set(SIMD_OPTION "AUTO" CACHE STRING "Choose between AVX2, AVX512, SSE, NEON, RISC-V, AUTO")
set_property(CACHE SIMD_OPTION PROPERTY STRINGS "AVX2" "AVX512" "SSE" "NEON" "RISC-V" "AUTO")
message(STATUS "Selected SIMD type: ${SIMD_OPTION}")

function(detect_simd_extension)
	if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86|x86_64|AMD64")
		include(CheckCXXSourceCompiles)

		# 检测AVX512
		set(CMAKE_REQUIRED_FLAGS "-mavx512f")
		check_cxx_source_compiles("
            #include <immintrin.h>
            int main() { __m512i v = _mm512_setzero_si512(); return 0; }
        " HAVE_AVX512)

		# 检测AVX2
		set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma")
		check_cxx_source_compiles("
            #include <immintrin.h>
            int main() { __m256i v = _mm256_setzero_si256(); return 0; }
        " HAVE_AVX2)

		# 检测SSE4
		set(CMAKE_REQUIRED_FLAGS "-msse4.1")
		check_cxx_source_compiles("
            #include <smmintrin.h>
            int main() { __m128i v = _mm_setzero_si128(); return 0; }
        " HAVE_SSE4)

		if(HAVE_AVX512)
			set(SIMD_AUTO_DETECTED "AVX512" PARENT_SCOPE)
		elseif(HAVE_AVX2)
			set(SIMD_AUTO_DETECTED "AVX2" PARENT_SCOPE)
		elseif(HAVE_SSE4)
			set(SIMD_AUTO_DETECTED "SSE" PARENT_SCOPE)
		else()
			set(SIMD_AUTO_DETECTED "NONE" PARENT_SCOPE)
		endif()
	elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
		include(CheckCXXSourceCompiles)
		set(CMAKE_REQUIRED_FLAGS "-march=armv8-a+simd")
		check_cxx_source_compiles("
            #include <arm_neon.h>
            int main() { float32x4_t v = vdupq_n_f32(0.0f); return 0; }
        " HAVE_NEON)

		if(HAVE_NEON)
			set(SIMD_AUTO_DETECTED "NEON" PARENT_SCOPE)
		else()
			set(SIMD_AUTO_DETECTED "NONE" PARENT_SCOPE)
		endif()
	elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
		include(CheckCXXSourceCompiles)
		set(CMAKE_REQUIRED_FLAGS "-march=rv64gcv")
		check_cxx_source_compiles("
            #include <riscv_vector.h>
            int main() { vint32m1_t v; return 0; }
        " HAVE_RVV)

		if(HAVE_RVV)
			set(SIMD_AUTO_DETECTED "RISC-V" PARENT_SCOPE)
		else()
			set(SIMD_AUTO_DETECTED "NONE" PARENT_SCOPE)
		endif()
	else()
		set(SIMD_AUTO_DETECTED "NONE" PARENT_SCOPE)
	endif()
endfunction()

# 处理自动检测
if(SIMD_OPTION STREQUAL "AUTO")
	detect_simd_extension()
	message(STATUS "Auto-detected SIMD extension: ${SIMD_AUTO_DETECTED}")
	set(SIMD_OPTION ${SIMD_AUTO_DETECTED})
endif()

# 设置指令集选项
if(SIMD_OPTION STREQUAL "AVX2")
	add_compile_definitions(USE_AVX2)

	if(MSVC)
		add_compile_options(/arch:AVX2)
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
	endif()

	message(STATUS "Enabled AVX2 instructions")
elseif(SIMD_OPTION STREQUAL "AVX512")
	add_compile_definitions(USE_AVX512)

	if(MSVC)
		add_compile_options(/arch:AVX512)
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mfma")
	endif()

	message(STATUS "Enabled AVX512 instructions")
elseif(SIMD_OPTION STREQUAL "SSE")
	add_compile_definitions(USE_SSE)

	if(MSVC)
		add_compile_options(/arch:SSE4.1)
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
	endif()

	message(STATUS "Enabled SSE instructions")
elseif(SIMD_OPTION STREQUAL "NEON")
	add_compile_definitions(USE_NEON)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
	message(STATUS "Enabled NEON instructions")
elseif(SIMD_OPTION STREQUAL "RISC-V")
	add_compile_definitions(USE_RVV)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv")
	message(STATUS "Enabled RISC-V Vector instructions")
elseif(SIMD_OPTION STREQUAL "NONE")
	message(STATUS "No SIMD extensions enabled")
else()
	message(FATAL_ERROR "Unknown SIMD option: ${SIMD_OPTION}")
endif()