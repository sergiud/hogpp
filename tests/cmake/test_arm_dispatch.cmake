cmake_minimum_required (VERSION 3.18)

# Configures the project for an ARM processor named TEST_SYSTEM_PROCESSOR and
# checks whether NEON dispatch is registered as TEST_EXPECT_NEON states. With
# TEST_EXPECT_SVE, the project is configured for TEST_SYSTEM_NAME as if the
# compiler supported SVE, which the host compiler usually does not, and SVE
# dispatch must be registered as TEST_EXPECT_SVE states.
if (NOT DEFINED PROJECT_SOURCE_DIR OR NOT DEFINED TEST_BINARY_DIR OR
    NOT DEFINED TEST_CXX_COMPILER OR NOT DEFINED TEST_SYSTEM_PROCESSOR OR
    NOT DEFINED TEST_EXPECT_NEON OR NOT DEFINED TEST_GENERATOR)
  message (FATAL_ERROR "The configure test arguments are incomplete")
endif (NOT DEFINED PROJECT_SOURCE_DIR OR NOT DEFINED TEST_BINARY_DIR OR
       NOT DEFINED TEST_CXX_COMPILER OR NOT DEFINED TEST_SYSTEM_PROCESSOR OR
       NOT DEFINED TEST_EXPECT_NEON OR NOT DEFINED TEST_GENERATOR)

if (NOT DEFINED TEST_SYSTEM_NAME)
  set (TEST_SYSTEM_NAME Linux)
endif (NOT DEFINED TEST_SYSTEM_NAME)

set (_sve_args)

if (DEFINED TEST_EXPECT_SVE)
  set (_sve_args -DHAVE_SVE=ON)
endif (DEFINED TEST_EXPECT_SVE)

file (MAKE_DIRECTORY "${TEST_BINARY_DIR}")
set (_toolchain_file "${TEST_BINARY_DIR}/toolchain.cmake")
file (WRITE "${_toolchain_file}"
  "set (CMAKE_SYSTEM_NAME ${TEST_SYSTEM_NAME})\n"
  "set (CMAKE_SYSTEM_PROCESSOR ${TEST_SYSTEM_PROCESSOR})\n"
  "set (CMAKE_CROSSCOMPILING_EMULATOR /usr/bin/env)\n"
  "set (CMAKE_CXX_COMPILER \"${TEST_CXX_COMPILER}\")\n"
  # The host compiler cannot link for every system the tests configure for.
  "set (CMAKE_CXX_COMPILER_WORKS ON)\n"
)

execute_process (
  COMMAND ${CMAKE_COMMAND} -E env CCACHE_DISABLE=1 ${CMAKE_COMMAND}
    -S "${PROJECT_SOURCE_DIR}"
    -B "${TEST_BINARY_DIR}"
    # The nested project must use the generator of the enclosing one, since
    # the default generator may not be installed.
    -G "${TEST_GENERATOR}"
    -DCMAKE_MAKE_PROGRAM=${TEST_MAKE_PROGRAM}
    -DCMAKE_TOOLCHAIN_FILE=${_toolchain_file}
    -DCMAKE_CXX_COMPILER_LAUNCHER=
    -DBUILD_TESTING=OFF
    -DWITH_DISPATCH=ON
    -DCMAKE_DISABLE_FIND_PACKAGE_OpenCV=ON
    -DCMAKE_DISABLE_FIND_PACKAGE_Sphinx=ON
    ${_sve_args}
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
  set (_has_neon ON)
else (_isa_config MATCHES "#define HAVE_ISA_NEON")
  set (_has_neon OFF)
endif (_isa_config MATCHES "#define HAVE_ISA_NEON")

if (NOT _has_neon STREQUAL TEST_EXPECT_NEON)
  message (FATAL_ERROR
    "NEON dispatch for the ${TEST_SYSTEM_PROCESSOR} processor is "
    "${_has_neon} but expected to be ${TEST_EXPECT_NEON}:\n${_isa_config}")
endif (NOT _has_neon STREQUAL TEST_EXPECT_NEON)

if (DEFINED TEST_EXPECT_SVE)
  if (_isa_config MATCHES "#define HAVE_ISA_SVE128")
    set (_has_sve ON)
  else (_isa_config MATCHES "#define HAVE_ISA_SVE128")
    set (_has_sve OFF)
  endif (_isa_config MATCHES "#define HAVE_ISA_SVE128")

  if (NOT _has_sve STREQUAL TEST_EXPECT_SVE)
    message (FATAL_ERROR
      "SVE dispatch on ${TEST_SYSTEM_NAME} is ${_has_sve} but expected to be "
      "${TEST_EXPECT_SVE}:\n${_isa_config}")
  endif (NOT _has_sve STREQUAL TEST_EXPECT_SVE)
endif (DEFINED TEST_EXPECT_SVE)
