include(ExternalProject)

set(GFLAGS_PROJECT       "external_gflags")
set(GFLAGS_SOURCE_DIR    ${ACE_TEMP_THIRD_PARTY_PATH}/gflags)
set(GFLAGS_INSTALL_DIR   ${ACE_THIRD_PARTY_PATH}/gflags)


# ExternalProject_Add(
#     ${GFLAGS_PROJECT}
#     GIT_REPOSITORY      "https://github.com/gflags/gflags.git"
#     GIT_TAG             "v2.2.2"
#     PREFIX              ${GFLAGS_SOURCE_DIR}
#     DOWNLOAD_DIR        ${GFLAGS_SOURCE_DIR}
#     UPDATE_COMMAND      ""
#     CMAKE_ARGS          -DCMAKE_INSTALL_PREFIX=${GFLAGS_INSTALL_DIR} -DBUILD_SHARED_LIBS=ON  -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_nothreads_LIB=OFF -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
#     BUILD_COMMAND       make
#     INSTALL_COMMAND     make install
#     LOG_CONFIGURE 1
#     LOG_INSTALL 1
# )

set(GFLAGS_INCLUDE_DIRS ${GFLAGS_INSTALL_DIR}/include CACHE PATH "Local Gflags header dir.")
set(GFLAGS_LIBRARY_PATH ${GFLAGS_INSTALL_DIR}/lib CACHE PATH "Local Gflags lib dir.")

set(GFLAGS_LIBRARIES gflags)
add_library(gflags SHARED IMPORTED GLOBAL)
set_target_properties(gflags PROPERTIES IMPORTED_LOCATION ${GFLAGS_INSTALL_DIR}/lib/libgflags.a)
add_dependencies(gflags ${GFLAGS_PROJECT})

include_directories(${GFLAGS_INCLUDE_DIRS})
link_directories(${GFLAGS_LIBRARY_PATH})