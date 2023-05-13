include(ExternalProject)

message(STATUS "Ace third party path is " ${ACE_THIRD_PARTY_PATH})

set(GLOG_PROJECT       "external_glog")
set(GLOG_SOURCE_DIR    ${ACE_TEMP_THIRD_PARTY_PATH}/glog)
set(GLOG_INSTALL_DIR   ${ACE_THIRD_PARTY_PATH}/glog)

# ExternalProject_Add(
#     ${GLOG_PROJECT}
#     DEPENDS             gflags
#     GIT_REPOSITORY      "https://github.com/google/glog.git"
#     GIT_TAG             "v0.6.0"
#     PREFIX              ${GLOG_SOURCE_DIR}
#     UPDATE_COMMAND      ""
#     #CMAKE_ARGS          -DCMAKE_INSTALL_PREFIX=${GLOG_INSTALL_DIR} -DCMAKE_INSTALL_LIBDIR=lib64
#     CMAKE_ARGS          -DCMAKE_INSTALL_PREFIX=${GLOG_INSTALL_DIR} -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON
#     INSTALL_COMMAND make install
#     LOG_CONFIGURE 1
#     LOG_INSTALL 1
# )

set(GLOG_INCLUDE_DIRS ${GLOG_INSTALL_DIR}/include CACHE PATH "Local glog header dir.")
set(GLOG_LIBRARY_PATH ${GLOG_INSTALL_DIR}/lib64  CACHE PATH "Local glog lib dir.")
set(GLOG_LIBRARIES glog)
add_library(glog SHARED IMPORTED GLOBAL)

set_target_properties(
    glog PROPERTIES 
    IMPORTED_LOCATION ${GLOG_INSTALL_DIR}/lib64/libglog.so
    INCLUDE_DIRECTORIES ${GLOG_INCLUDE_DIRS})

add_dependencies(glog ${GLOG_PROJECT})

message(STATUS "GLOG_INCLUDE_DIRS: " ${GLOG_INCLUDE_DIRS})
message(STATUS "GLOG_LIBRARY_PATH: " ${GLOG_LIBRARY_PATH})

include_directories(${GLOG_INCLUDE_DIRS})
link_directories(${GLOG_LIBRARY_PATH})
