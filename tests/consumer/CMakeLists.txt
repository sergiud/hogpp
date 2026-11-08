# SPDX-FileCopyrightText: 2026 Sergiu Deitsch
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required (VERSION 3.28...4.4)
project (hogpp_consumer LANGUAGES CXX)

# The source tree is added by consumers that build hogpp along with their own
# targets, whereas the installed package is found otherwise.
set (HOGPP_SOURCE_DIR "" CACHE PATH
  "Source directory of hogpp to add instead of finding the installed package")
set (HOGPP_REQUIRED_VERSION "" CACHE STRING
  "Version of the installed package to require")

if (HOGPP_SOURCE_DIR)
  add_subdirectory (${HOGPP_SOURCE_DIR} hogpp)

  # Projects building hogpp along with their own targets get its library
  # alone, without its tests, documentation or Python module.
  get_property (hogpp_targets DIRECTORY ${HOGPP_SOURCE_DIR}
    PROPERTY BUILDSYSTEM_TARGETS)

  if (NOT hogpp_targets STREQUAL "hogpp")
    message (FATAL_ERROR "hogpp added the targets “${hogpp_targets}”, "
      "expected “hogpp”")
  endif (NOT hogpp_targets STREQUAL "hogpp")
else (HOGPP_SOURCE_DIR)
  find_package (hogpp ${HOGPP_REQUIRED_VERSION} REQUIRED)
endif (HOGPP_SOURCE_DIR)

include (CTest)

add_executable (consumer main.cpp)
target_link_libraries (consumer PRIVATE hogpp::hogpp)

add_test (NAME consumer COMMAND consumer)
