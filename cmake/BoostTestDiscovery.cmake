# SPDX-FileCopyrightText: 2026 Sergiu Deitsch
#
# SPDX-License-Identifier: Apache-2.0

#[=======================================================================[.rst:
BoostTestDiscovery
------------------

Registers each test case of a program of Boost.Test as a test of its own,
whose result CTest reports apart from the others.

.. command:: boost_test_discover_tests

  .. code-block:: cmake

    boost_test_discover_tests (<target> [TEST_PREFIX <prefix>])

  Lists the test cases of the executable ``<target>`` when CTest runs, so
  that building the program leaves the tests up to date, and registers the
  test ``<prefix><path>`` for each, which runs the program restricted to the
  test case at ``<path>``. Samples of data test cases are test cases of their
  own. Disabled test cases are registered as disabled tests, and test cases
  skipped because a precondition failed are reported as skipped.

  CTest runs the program through the launcher of the target given by
  ``TEST_LAUNCHER``, and through its emulator given by
  ``CROSSCOMPILING_EMULATOR`` if cross compiling, both when listing the test
  cases and when running them.
#]=======================================================================]

include_guard (GLOBAL)

set (_boost_test_discovery_module "${CMAKE_CURRENT_LIST_FILE}")

# Quotes value as a bracket argument.
function (_boost_test_quote value output)
  set (${output} "[==[${value}]==]" PARENT_SCOPE)
endfunction (_boost_test_quote)

function (boost_test_discover_tests target)
  cmake_parse_arguments (PARSE_ARGV 1 arg "" "TEST_PREFIX" "")

  if (arg_UNPARSED_ARGUMENTS)
    message (FATAL_ERROR "boost_test_discover_tests (${target}) was given "
      "the unknown arguments “${arg_UNPARSED_ARGUMENTS}”, expected "
      "TEST_PREFIX")
  endif (arg_UNPARSED_ARGUMENTS)

  set (launcher "$<TARGET_PROPERTY:${target},TEST_LAUNCHER>")

  # Tests registered by CTest itself name no target the emulator would be
  # taken from, which add_test applies when cross compiling only.
  if (CMAKE_CROSSCOMPILING)
    string (APPEND launcher
      "$<SEMICOLON>$<TARGET_PROPERTY:${target},CROSSCOMPILING_EMULATOR>")
  endif (CMAKE_CROSSCOMPILING)

  get_property (multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

  set (base "${CMAKE_CURRENT_BINARY_DIR}/${target}")
  set (suffix)

  if (multi_config)
    set (suffix "-$<CONFIG>")
  endif (multi_config)

  set (script "${base}_discovery${suffix}.cmake")
  set (tests "${base}_tests${suffix}.cmake")

  _boost_test_quote ("${_boost_test_discovery_module}" quoted_module)
  _boost_test_quote ("$<TARGET_FILE:${target}>" quoted_executable)
  _boost_test_quote ("${launcher}" quoted_launcher)
  _boost_test_quote ("${tests}" quoted_tests)
  _boost_test_quote ("${arg_TEST_PREFIX}" quoted_prefix)

  file (GENERATE OUTPUT "${script}" CONTENT
"include (${quoted_module})
_boost_test_write_tests (
  ${quoted_executable}
  ${quoted_launcher}
  ${quoted_prefix}
  ${quoted_tests}
)
include (${quoted_tests})
")

  if (multi_config)
    # CTest runs the tests of the configuration it is given.
    set (include "${base}_discovery.cmake")
    string (REPLACE "$<CONFIG>" "\${CTEST_CONFIGURATION_TYPE}" script
      "${script}")
    file (WRITE "${include}"
"if (EXISTS \"${script}\")
  include (\"${script}\")
else (EXISTS \"${script}\")
  add_test (${arg_TEST_PREFIX}NOT_BUILT ${target}_NOT_BUILT)
endif (EXISTS \"${script}\")
")
  else (multi_config)
    set (include "${script}")
  endif (multi_config)

  set_property (DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${include}")
endfunction (boost_test_discover_tests)

# Writes the file tests registering the test cases executable lists, run
# through the commands launcher, unless the file is newer than the executable
# and the scripts producing the file.
function (_boost_test_write_tests executable launcher prefix tests)
  list (REMOVE_ITEM launcher "")

  if (NOT EXISTS "${executable}")
    _boost_test_quote ("${prefix}NOT_BUILT" name)
    file (WRITE "${tests}" "add_test (${name} ${name})\n")
    return ()
  endif (NOT EXISTS "${executable}")

  set (outdated FALSE)

  if (NOT EXISTS "${tests}")
    set (outdated TRUE)
  endif (NOT EXISTS "${tests}")

  foreach (source IN ITEMS "${executable}" "${CMAKE_CURRENT_LIST_FILE}"
      "${_boost_test_discovery_module}")
    if (NOT "${tests}" IS_NEWER_THAN "${source}")
      set (outdated TRUE)
    endif (NOT "${tests}" IS_NEWER_THAN "${source}")
  endforeach (source)

  if (NOT outdated)
    return ()
  endif (NOT outdated)

  # Boost.Test writes the tree to the standard error unless redirected, which
  # keeps it apart from what emulators write there.
  execute_process (
    COMMAND ${launcher} "${executable}" --list_content --report_sink=stdout
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
  )

  if (NOT result EQUAL 0)
    message (FATAL_ERROR "Listing the test cases of ${executable} failed "
      "with ${result}, expected 0:\n${output}${error}")
  endif (NOT result EQUAL 0)

  # Units are indented by four spaces for each suite containing them, and
  # enabled ones end with an asterisk.
  set (indentation 4)
  set (unit_pattern "^( *)([^ *]|[^ ].*[^*])(\\*?)$")

  string (REPLACE "\n" ";" lines "${output}")
  set (units)

  foreach (line IN LISTS lines)
    if (line MATCHES "${unit_pattern}")
      string (LENGTH "${CMAKE_MATCH_1}" spaces)
      math (EXPR depth "${spaces} / ${indentation}")
      set (enabled TRUE)

      if (CMAKE_MATCH_3 STREQUAL "")
        set (enabled FALSE)
      endif (CMAKE_MATCH_3 STREQUAL "")

      list (APPEND units "${depth}|${CMAKE_MATCH_2}|${enabled}")
    elseif (NOT line STREQUAL "")
      message (FATAL_ERROR "${executable} listed “${line}”, expected a "
        "test unit")
    endif (line MATCHES "${unit_pattern}")
  endforeach (line)

  # Boost.Test reports the units it does not run as skipped because they are
  # disabled, so that only failed preconditions skip a test. Those of suites
  # skip the test cases the suites contain.
  _boost_test_quote ("is skipped because precondition failed"
    quoted_skip_pattern)

  list (LENGTH units count)
  set (suites)
  set (content)

  if (count EQUAL 0)
    file (WRITE "${tests}" "")
    return ()
  endif (count EQUAL 0)

  # Units deeper than their successor contain it and are thus suites, whereas
  # the others are test cases.
  foreach (index RANGE 1 ${count})
    math (EXPR current "${index} - 1")
    list (GET units ${current} unit)
    string (REPLACE "|" ";" unit "${unit}")
    list (GET unit 0 depth)
    list (GET unit 1 name)
    list (GET unit 2 enabled)
    list (SUBLIST suites 0 ${depth} suites)
    set (next_depth 0)

    if (index LESS count)
      list (GET units ${index} next)
      string (REGEX REPLACE "\\|.*" "" next_depth "${next}")
    endif (index LESS count)

    if (next_depth GREATER depth)
      list (APPEND suites "${name}")
      continue ()
    endif (next_depth GREATER depth)

    set (path ${suites} "${name}")
    list (JOIN path "/" path)

    _boost_test_quote ("${prefix}${path}" quoted_name)
    string (APPEND content "add_test (${quoted_name}")

    # Boost.Test exits successfully if a precondition skips the test case,
    # which it states only at the log level of test units.
    foreach (argument IN LISTS launcher ITEMS "${executable}"
        "--run_test=${path}" "--log_level=test_suite")
      _boost_test_quote ("${argument}" quoted_argument)
      string (APPEND content " ${quoted_argument}")
    endforeach (argument)

    string (APPEND content ")\n")
    string (APPEND content "set_tests_properties (${quoted_name} PROPERTIES "
      "SKIP_REGULAR_EXPRESSION ${quoted_skip_pattern})\n")

    if (NOT enabled)
      string (APPEND content
        "set_tests_properties (${quoted_name} PROPERTIES DISABLED TRUE)\n")
    endif (NOT enabled)
  endforeach (index)

  file (WRITE "${tests}" "${content}")
endfunction (_boost_test_write_tests)
