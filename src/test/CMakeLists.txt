
include_directories("${PROJECT_SOURCE_DIR}")

if(CMAKE_CL_64)
    file(GLOB MKL_DLL
        "${MKL_DIR}/bin/mkl_sequential.2.dll"
        "${MKL_DIR}/bin/mkl_core.2.dll"
        "${MKL_DIR}/bin/mkl_avx2.2.dll"
        "${MKL_DIR}/bin/mkl_rt.2.dll"
        "${MKL_DIR}/bin/mkl_intel_thread.2.dll"
        "${MKL_DIR}/bin/mkl_vml_def.2.dll"
    )
    file(GLOB MKL_LIB
        "${MKL_DIR}/lib/mkl_sequential.lib"
        "${MKL_DIR}/lib/mkl_core.lib"
        "${MKL_DIR}/lib/mkl_rt.lib"
    )
else()
    file(GLOB MKL_DLL
        "${MKL_DIR}/bin32/mkl_sequential.2.dll"
        "${MKL_DIR}/bin32/mkl_core.2.dll"
        "${MKL_DIR}/bin32/mkl_avx2.2.dll"
        "${MKL_DIR}/bin32/mkl_rt.2.dll"
        "${MKL_DIR}/bin32/mkl_p4.2.dll"
        "${MKL_DIR}/bin32/mkl_intel_thread.2.dll"
        "${MKL_DIR}/bin32/mkl_vml_p4.2.dll"
    )
    file(GLOB MKL_LIB
        "${MKL_DIR}/lib32/mkl_sequential.lib"
        "${MKL_DIR}/lib32/mkl_core.lib"
        "${MKL_DIR}/lib32/mkl_rt.lib"
    )
endif()

file(COPY ${MKL_DLL} DESTINATION ${PROJECT_BINARY_DIR})

add_executable(test_1d test_1d.cc)
add_executable(test_2d test_2d.cc)

target_link_libraries(test_1d ${MKL_LIB})
target_link_libraries(test_2d ${MKL_LIB})
