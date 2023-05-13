set(CUDA_MIN_VERSION "9.0")
find_package(CUDA ${CUDA_MIN_VERSION})

if(CUDA_PROFILE)
    set(EXTRA_LIBS  -lnvToolsExt)
endif()

if(CUDA_FOUND)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES -Wno-deprecated-gpu-targets -w ${EXTRA_LIBS}")
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O0")
    else()
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
    endif()


    IF ((CUDA_VERSION VERSION_GREATER "9.0") OR (CUDA_VERSION VERSION_EQUAL "9.0"))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
    ENDIF()

    IF ((CUDA_VERSION VERSION_GREATER "10.0") OR (CUDA_VERSION VERSION_EQUAL "10.0"))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
    ENDIF()

    IF ((CUDA_VERSION VERSION_GREATER "11.0") OR (CUDA_VERSION VERSION_EQUAL "11.0"))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
    ENDIF()

    message(STATUS "Enabling CUDA support (version: ${CUDA_VERSION_STRING},"
                    " archs: ${CUDA_ARCH_FLAGS_readable})")

    else()
    message(FATAL_ERROR "CUDA not found >= ${CUDA_MIN_VERSION} required)")
endif()

