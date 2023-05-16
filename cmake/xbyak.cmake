include(ExternalProject)

set(XBYAK_PROJECT       extern_xbyak)
set(XBYAK_PREFIX_DIR    ${ANAKIN_TEMP_THIRD_PARTY_PATH}/xbyak)
set(XBYAK_CLONE_DIR     ${XBYAK_PREFIX_DIR}/src/${XBYAK_PROJECT})
set(XBYAK_INSTALL_ROOT  ${ANAKIN_TEMP_THIRD_PARTY_PATH}/xbyak)
set(XBYAK_INC_DIR       ${XBYAK_INSTALL_ROOT}/include)

set(XBYAK_PROJECT       "extern_xbyak")
set(XBYAK_SOURCE_DIR    ${ACE_TEMP_THIRD_PARTY_PATH}/xbyak)
set(XBYAK_INSTALL_DIR   ${ACE_THIRD_PARTY_PATH}/xbyak)

if(USE_SGX)
    set(SGX_PATCH_CMD "cd ${XBYAK_INSTALL_DIR} && patch -p1 <${ACE_TEMP_THIRD_PARTY_PATH}/xbyak.patch")
else()
    # use a whitespace as nop so that sh won't complain about missing argument
    set(SGX_PATCH_CMD " ")
endif()

ExternalProject_Add(
    ${XBYAK_PROJECT}
    DEPENDS             ""
    GIT_REPOSITORY      "https://github.com/herumi/xbyak.git"
    GIT_TAG             "v5.661"  # Jul 26th
    PREFIX              ${XBYAK_SOURCE_DIR}
    UPDATE_COMMAND      ""
    CMAKE_ARGS          -DCMAKE_INSTALL_PREFIX=${XBYAK_INSTALL_DIR}
    INSTALL_COMMAND     make install
    COMMAND             sh -c "${SGX_PATCH_CMD}"
    VERBATIM
)

add_library(xbyak SHARED IMPORTED GLOBAL)
add_dependencies(xbyak ${XBYAK_PROJECT})
set(XBYAK_INCLUDE_DIRS ${XBYAK_INSTALL_DIR}/include CACHE PATH "Local flatbuffers header dir.")
include_directories(${XBYAK_INCLUDE_DIRS})
