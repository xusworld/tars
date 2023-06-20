message(STATUS "Generate onnx proto files...")


message(STATUS "onnx CMAKE_CURRENT_SOURCE_DIR" ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "IMPL_SCRS" ${IMPL_SCRS})

# header files 
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
# include_directories(${ACE_ROOT})

# source files
file(GLOB IMPL_SCRS   ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx/*.cc)
message(STATUS "IMPL_SCRS" ${IMPL_SCRS})

file(GLOB ONNX_OP_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx/operator/*.cc)

set(ONNX_PROTO_SRCS   ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx/proto/onnx.pb.cc)

add_library(ConverterONNX OBJECT ${IMPL_SCRS} ${ONNX_OP_SRC} ${ONNX_PROTO_SRCS})
target_link_libraries(ConverterONNX  protobuf )
