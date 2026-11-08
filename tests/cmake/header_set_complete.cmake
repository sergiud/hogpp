# SPDX-FileCopyrightText: 2026 Sergiu Deitsch
#
# SPDX-License-Identifier: Apache-2.0

# Fails unless the header set HEADERS, separated by vertical bars, lists every
# header below the directory INCLUDE_DIR, since headers missing from the set
# are neither installed nor verified to compile on their own. Relative paths
# in the set are relative to SOURCE_DIR.
#
# Usage: cmake -D HEADERS=<a|b|...> -D INCLUDE_DIR=<dir> -D SOURCE_DIR=<dir>
#   -P header_set_complete.cmake

file (GLOB_RECURSE expected RELATIVE "${INCLUDE_DIR}" "${INCLUDE_DIR}/*.hpp")

string (REPLACE "|" ";" headers "${HEADERS}")
set (actual)

foreach (header IN LISTS headers)
  cmake_path (ABSOLUTE_PATH header BASE_DIRECTORY "${SOURCE_DIR}")
  cmake_path (RELATIVE_PATH header BASE_DIRECTORY "${INCLUDE_DIR}")
  list (APPEND actual "${header}")
endforeach (header)

list (SORT expected)
list (SORT actual)

if (NOT actual STREQUAL expected)
  set (missing ${expected})
  list (REMOVE_ITEM missing ${actual})
  set (extra ${actual})
  list (REMOVE_ITEM extra ${expected})
  message (FATAL_ERROR "The header set lists “${actual}”, expected "
    "“${expected}”, missing “${missing}” and extra “${extra}”")
endif (NOT actual STREQUAL expected)
