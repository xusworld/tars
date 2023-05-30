cmake_minimum_required(VERSION 3.0)

# third party path
set(ACE_ROOT ${CMAKE_CURRENT_LIST_DIR})

set (CMAKE_C_COMPILER "/usr/bin/gcc")
set (CMAKE_CXX_COMPILER "/usr/bin/g++")


message(STATUS "Ace v2 root dir: " ${ACE_ROOT})
set(ACE_THIRD_PARTY_PATH ${ACE_ROOT}/third-party)
set(ACE_BUILD_THIRD_PARTY_PATH ${CMAKE_BINARY_DIR}/third-party)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/gflags.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/glog.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/gtest.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/protobuf.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/flatbuffers.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/xbyak.cmake)

set(VERSION_MAJOR "0")
set(VERSION_MINOR "1")
set(VERSION_PATCH "0")
set(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

cmake_policy(SET CMP0048 NEW)

# Ace is cool deep learning inference framework.
project(ACE VERSION ${VERSION} LANGUAGES C CXX ASM)

# complier options
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_LIST_DIR}/cmake"
)

# build options, move options into cmake/options.cmake
option(ACE_BUILD_HARD "Build -mfloat-abi=hard or not" OFF)
option(ACE_BUILD_SHARED_LIBS "ACE build shared or static lib" ON)
option(ACE_FORBID_MULTI_THREAD "Disable Multi Thread" OFF)
option(ACE_USE_THREAD_POOL "Use ACE's own thread pool implementation" ON)
option(ACE_BUILD_CONVERTER "Build Converter" ON)
option(ACE_DEBUG_MEMORY "ACE Debug Memory Access" OFF)
option(ACE_DEBUG_TENSOR_SIZE "Enable Tensor Size" OFF)

option(ACE_USE_ONEDNN "Enable oneDNN" OFF)
option(ACE_OPENMP "Use OpenMP" OFF)
option(ACE_USE_SSE "Use SSE optimization for x86 if possiable" ON)
option(ACE_AVX2 "Enable AVX2" OFF)
option(ACE_AVX512 "Enable AVX512" OFF)

option(ACE_DEBUG "Debug " ON)

# Options for Nvidia gpus
option(ACE_USE_CUDA "Enable CUDA" ON)
option(ACE_USE_CUDNN "Enable CUDNN" OFF)
option(ACE_TENSORRT "Enable TensorRT" OFF)

if(ACE_USE_CUDA) 
 include(${CMAKE_CURRENT_LIST_DIR}/cmake/cuda.cmake)
 include_directories(/usr/local/cuda/include/)
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_LIST_DIR}/tars)
include_directories(${CMAKE_CURRENT_LIST_DIR}/ir)
include_directories(${CMAKE_CURRENT_LIST_DIR}/third-party)
include_directories(${CMAKE_CURRENT_LIST_DIR}/third-party/vector)
include_directories(${CMAKE_CURRENT_LIST_DIR}/third-party/flatbuffers/include)

SET(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -rdynamic -ldl  -g3 -std=c++17 -m64 -mavx2 -mavx512f -mfma -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -rdynamic -ldl -g3 -std=c++17 -m64 -mavx2 -mavx512f -mfma -fopenmp -L/usr/local/cuda/lib64/ -lcudart")


add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tars)