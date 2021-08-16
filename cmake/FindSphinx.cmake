# Copyright 2021 Sergiu Deitsch <sergiu.deitsch@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

include (FindPackageHandleStandardArgs)

find_program (Sphinx_BUILD_EXECUTABLE
  NAMES sphinx-build
  DOC "Sphinx build executable"
)

mark_as_advanced (Sphinx_BUILD_EXECUTABLE)

if (NOT TARGET Sphinx::build)
  add_executable (Sphinx::build IMPORTED)
endif (NOT TARGET Sphinx::build)

if (Sphinx_BUILD_EXECUTABLE)
  set_property (TARGET Sphinx::build PROPERTY IMPORTED_LOCATION
    ${Sphinx_BUILD_EXECUTABLE})

  # Extract version information
  execute_process (COMMAND ${Sphinx_BUILD_EXECUTABLE} --version
    RESULT_VARIABLE _Sphinx_RESULT
    OUTPUT_VARIABLE _Sphinx_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if (NOT _Sphinx_RESULT EQUAL 0)
    message (FATAL_ERROR "Failed to obtain sphinx-build version")
  endif (NOT _Sphinx_RESULT EQUAL 0)

  if (_Sphinx_VERSION)
    string (REGEX MATCH "sphinx-build[ \t]+([0-9]+)\.([0-9]+)\.([0-9]+)"
      _Sphinx_MATCH "${_Sphinx_VERSION}")

    if (_Sphinx_MATCH)
      set (Sphinx_VERSION_MAJOR ${CMAKE_MATCH_1})
      set (Sphinx_VERSION_MINOR ${CMAKE_MATCH_2})
      set (Sphinx_VERSION_PATCH ${CMAKE_MATCH_3})
      set (Sphinx_VERSION
        ${Sphinx_VERSION_MAJOR}.${Sphinx_VERSION_MINOR}.${Sphinx_VERSION_PATCH})
      set (Sphinx_VERSION_COMPONENTS 3)
    endif (_Sphinx_MATCH)
  endif (_Sphinx_VERSION)
endif (Sphinx_BUILD_EXECUTABLE)

find_package_handle_standard_args (Sphinx
  REQUIRED_VARS Sphinx_BUILD_EXECUTABLE
  VERSION_VAR Sphinx_VERSION
)
