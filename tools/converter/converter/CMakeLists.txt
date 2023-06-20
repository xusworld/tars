project(ace-converter C CXX)

message(STATUS "Ace Model Converter, only support onnx model now..." )
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1 -O3 -std=c++17 -m64  ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1 -O3 -std=c++17 -m64 ")

# move binary file to build dir
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# onnx 
include(${CMAKE_CURRENT_LIST_DIR}/src/onnx/CMakeLists.txt)

add_executable(ace-converter converter.cc)
target_link_libraries(ace-converter  gflags glog protobuf flatbuffers ConverterONNX)
