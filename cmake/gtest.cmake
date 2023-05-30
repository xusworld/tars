include(ExternalProject)

message(STATUS "Ace third party path is " ${ACE_THIRD_PARTY_PATH})

set(GTEST_PROJECT       "external_gtest")
set(GTEST_SOURCE_DIR    ${ACE_TEMP_THIRD_PARTY_PATH}/gtest)
set(GTEST_INSTALL_DIR   ${ACE_THIRD_PARTY_PATH}/gtest)

ExternalProject_Add(
    ${GTEST_PROJECT}
    GIT_REPOSITORY      "https://github.com/google/googletest.git"
    GIT_TAG             "v1.12.0"
    PREFIX              ${GTEST_SOURCE_DIR}
    UPDATE_COMMAND      ""
    CMAKE_ARGS          -DCMAKE_INSTALL_PREFIX=${GTEST_INSTALL_DIR} -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON
    INSTALL_COMMAND make install
    LOG_CONFIGURE 1
    LOG_INSTALL 1
)

set(GTEST_INCLUDE_DIRS ${GTEST_INSTALL_DIR}/include CACHE PATH "Local gtest header dir.")
set(GTEST_LIBRARY_PATH ${GTEST_INSTALL_DIR}/lib64  CACHE PATH "Local gtest lib dir.")
set(GTEST_LIBRARIES gtest)

add_library(gtest SHARED IMPORTED GLOBAL)

set_target_properties(
    gtest PROPERTIES 
    IMPORTED_LOCATION ${GTEST_INSTALL_DIR}/lib64/libgtest.so
    INCLUDE_DIRECTORIES ${GTEST_INCLUDE_DIRS})

add_dependencies(gtest ${GTEST_PROJECT})

include_directories(${GTEST_INCLUDE_DIRS})
link_directories(${GTEST_LIBRARY_PATH})
