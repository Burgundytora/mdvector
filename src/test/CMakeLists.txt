
include_directories("${PROJECT_SOURCE_DIR}")

add_executable(test test.cc)

if(USE_MKL)
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
    target_link_libraries(test ${MKL_LIB})
    file(COPY ${MKL_DLL} DESTINATION ${PROJECT_BINARY_DIR})
endif()
