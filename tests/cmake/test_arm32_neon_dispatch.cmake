cmake_minimum_required (VERSION 3.18)

if (NOT DEFINED PROJECT_SOURCE_DIR OR NOT DEFINED TEST_BINARY_DIR OR
    NOT DEFINED TEST_CXX_COMPILER)
  message (FATAL_ERROR "The configure test arguments are incomplete")
endif (NOT DEFINED PROJECT_SOURCE_DIR OR NOT DEFINED TEST_BINARY_DIR OR
       NOT DEFINED TEST_CXX_COMPILER)

file (MAKE_DIRECTORY "${TEST_BINARY_DIR}")
set (_toolchain_file "${TEST_BINARY_DIR}/toolchain.cmake")
file (WRITE "${_toolchain_file}"
  "set (CMAKE_SYSTEM_NAME Linux)\n"
  "set (CMAKE_SYSTEM_PROCESSOR arm)\n"
  "set (CMAKE_CROSSCOMPILING_EMULATOR /usr/bin/env)\n"
  "set (CMAKE_CXX_COMPILER \"${TEST_CXX_COMPILER}\")\n"
)

execute_process (
  COMMAND ${CMAKE_COMMAND} -E env CCACHE_DISABLE=1 ${CMAKE_COMMAND}
    -S "${PROJECT_SOURCE_DIR}"
    -B "${TEST_BINARY_DIR}"
    -DCMAKE_TOOLCHAIN_FILE=${_toolchain_file}
    -DCMAKE_CXX_COMPILER_LAUNCHER=
    -DBUILD_TESTING=OFF
    -DWITH_DISPATCH=ON
    -DCMAKE_DISABLE_FIND_PACKAGE_OpenCV=ON
    -DCMAKE_DISABLE_FIND_PACKAGE_Sphinx=ON
  RESULT_VARIABLE _configure_result
  OUTPUT_VARIABLE _configure_output
  ERROR_VARIABLE _configure_error
)

if (_configure_result)
  message (FATAL_ERROR
    "The ARM configure test failed:\n${_configure_output}${_configure_error}")
endif (_configure_result)

file (READ "${TEST_BINARY_DIR}/python/isaconfig.hpp" _isa_config)
if (_isa_config MATCHES "#define HAVE_ISA_NEON")
  message (FATAL_ERROR
    "NEON was registered without compiler support:\n${_isa_config}")
endif (_isa_config MATCHES "#define HAVE_ISA_NEON")
