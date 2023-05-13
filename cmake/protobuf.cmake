include(ExternalProject)


set(PROTOBUF_PROJECT       "external_protobuf")
set(PROTOBUF_SOURCE_DIR    ${ACE_TEMP_THIRD_PARTY_PATH}/protobuf)
set(PROTOBUF_INSTALL_DIR   ${ACE_THIRD_PARTY_PATH}/protobuf)


# ExternalProject_Add(
#     ${PROTOBUF_PROJECT}
#     GIT_REPOSITORY      "https://github.com/protocolbuffers/protobuf.git"
#     GIT_TAG             "v3.20.3"
#     PREFIX              ${PROTOBUF_SOURCE_DIR}
#     UPDATE_COMMAND      ""
#     # CMAKE_CACHE_ARGS
#     #   "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}"
#     #   "-Dprotobuf_BUILD_TESTS:BOOL=OFF"
#     #   "-Dprotobuf_BUILD_EXAMPLES:BOOL=OFF"
#     #   "-Dprotobuf_WITH_ZLIB:BOOL=OFF"
#     #   "-DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}" 
#     # CMAKE_ARGS          -DCMAKE_INSTALL_PREFIX=${PROTOBUF_INSTALL_DIR}
#     # CMAKE_ARGS          -DCMAKE_INSTALL_LIBDIR=lib64
#     # SOURCE_SUBDIR cmake
#     UPDATE_DISCONNECTED 1
#     BUILD_IN_SOURCE 1
#     CONFIGURE_COMMAND ./autogen.sh COMMAND ./configure --prefix=${PROTOBUF_INSTALL_DIR}
#     INSTALL_COMMAND make install
# )


set(PROTOBUF_INCLUDE_DIRS ${PROTOBUF_INSTALL_DIR}/include CACHE PATH "Local glog header dir.")
set(PROTOBUF_LIBRARY_PATH ${PROTOBUF_INSTALL_DIR}/lib CACHE PATH "Local glog lib dir.")
set(PROTOBUF_LIBRARIES protobuf)
add_library(protobuf SHARED IMPORTED GLOBAL)

message(STATUS "Protobuf header dir: " ${PROTOBUF_INCLUDE_DIRS})
message(STATUS "Protobuf library path: " ${PROTOBUF_LIBRARY_PATH})

set_target_properties(
    protobuf PROPERTIES 
    IMPORTED_LOCATION ${PROTOBUF_LIBRARY_PATH}/libprotobuf.so
    INCLUDE_DIRECTORIES ${PROTOBUF_INCLUDE_DIRS})

add_dependencies(protobuf ${PROTOBUF_PROJECT})

include_directories(${PROTOBUF_INCLUDE_DIRS})
link_directories(${PROTOBUF_LIBRARY_PATH})

