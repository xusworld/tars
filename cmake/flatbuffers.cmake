include(ExternalProject)

set(FLATBUFFERS_PROJECT       "external_flatbuffers")
set(FLATBUFFERS_SOURCE_DIR    ${ACE_TEMP_THIRD_PARTY_PATH}/flatbuffers)
set(FLATBUFFERS_INSTALL_DIR   ${ACE_THIRD_PARTY_PATH}/flatbuffers)

# ExternalProject_Add(
#     ${FLATBUFFERS_PROJECT}
#     GIT_REPOSITORY      "https://github.com/google/flatbuffers.git"
#     GIT_TAG             "v2.0.8"
#     PREFIX              ${FLATBUFFERS_SOURCE_DIR}
#     UPDATE_COMMAND      ""
#     CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${FLATBUFFERS_INSTALL_DIR} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DFLATBUFFERS_BUILD_FLATC=ON
#     LOG_UPDATE          ON
#     LOG_CONFIGURE       ON
#     LOG_BUILD           ON 
#     INSTALL_COMMAND     make install
# )

set(FLATBUFFERS_INCLUDE_DIRS ${FLATBUFFERS_INSTALL_DIR}/include CACHE PATH "Local flatbuffers header dir.")
set(FLATBUFFERS_LIBRARY_PATH ${FLATBUFFERS_INSTALL_DIR}/lib64   CACHE PATH "Local flatbuffers lib path.")
set(FLATBUFFERS_BIN_PATH     ${FLATBUFFERS_INSTALL_DIR}/bin     CACHE PATH "Local flatbuffers bin path.")


message(STATUS "Flatbuffers  header dirs: " ${FLATBUFFERS_INCLUDE_DIRS})
message(STATUS "Flatbuffers library path: " ${FLATBUFFERS_LIBRARY_PATH})

set(FLATBUFFERS_LIBRARIES flatbuffers)
add_library(flatbuffers SHARED IMPORTED GLOBAL)

set_target_properties(
    flatbuffers PROPERTIES 
    IMPORTED_LOCATION ${FLATBUFFERS_INSTALL_DIR}/lib64/libflatbuffers.a
    INCLUDE_DIRECTORIES ${FLATBUFFERS_INCLUDE_DIRS})

add_dependencies(flatbuffers ${FLATBUFFERS_PROJECT})

include_directories(${FLATBUFFERS_INCLUDE_DIRS})
link_directories(${FLATBUFFERS_LIBRARY_PATH})
