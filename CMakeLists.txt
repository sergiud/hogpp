cmake_minimum_required (VERSION 3.28...4.4)
project (hogpp
  VERSION 0.3.0
  LANGUAGES CXX
  DESCRIPTION "Fast histogram of oriented gradients computation using integral histogram"
  HOMEPAGE_URL https://github.com/sergiud/hogpp
)

# Disable module support since under MinGW Clang compiling fails with the error
# '"CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS-NOTFOUND"' is not recognized as an
# internal or external command'. On Linux, the following Clang bug additionally
# prevents compilation: https://github.com/llvm/llvm-project/issues/88896
set (CMAKE_CXX_SCAN_FOR_MODULES OFF)
set (CMAKE_CXX_VISIBILITY_PRESET hidden)
set (CMAKE_POSITION_INDEPENDENT_CODE ON)
set (CMAKE_VISIBILITY_INLINES_HIDDEN ON)

list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Eigen 3.4.1 and later accept version ranges, whereas earlier releases of
# Eigen 3 are found by their minimum version alone. The installed package
# configuration searches for the dependencies the same way.
set (hogpp_Eigen3_VERSION_RANGE 3.4.1...5)
set (hogpp_Eigen3_MIN_VERSION 3.3.7)
set (hogpp_fmt_MIN_VERSION 6.0)

include (CMakeDependentOption)

# Projects that add hogpp as a subdirectory define their own tests.
if (PROJECT_IS_TOP_LEVEL)
  include (CTest)
endif (PROJECT_IS_TOP_LEVEL)

option (WITH_PYTHON "Build the Python module" ${PROJECT_IS_TOP_LEVEL})
option (BUILD_DEPS
  "Download and build Eigen and fmt instead of using the installed packages" OFF)
# The exported targets require the installed packages of the dependencies, and
# Python wheels contain the module only.
cmake_dependent_option (WITH_INSTALL
  "Install the headers along with the CMake package configuration"
  ${PROJECT_IS_TOP_LEVEL} "NOT BUILD_DEPS;NOT SKBUILD" OFF)

if (PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)
  set (hogpp_BUILD_TESTS ON)
else (PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)
  set (hogpp_BUILD_TESTS OFF)
endif (PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)

if (hogpp_BUILD_TESTS)
  find_package (Boost 1.70 CONFIG)
endif (hogpp_BUILD_TESTS)

if (BUILD_DEPS)
  include (FetchContent)

  # Use EXCLUDE_FROM_ALL to avoid installing headers and libraries into the
  # wheels.
  FetchContent_Declare (eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG bc3b39870ecb690a623a3f49149a358b95c5781d # 5.0.1
    EXCLUDE_FROM_ALL
    SYSTEM
  )

  FetchContent_Declare (fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 407c905e45ad75fc29bf0f9bb7c5c2fd3475976f # 12.1.0
    EXCLUDE_FROM_ALL
    SYSTEM
  )

  FetchContent_MakeAvailable (eigen3 fmt)
else (BUILD_DEPS)
  find_package (Eigen3 ${hogpp_Eigen3_VERSION_RANGE} CONFIG)

  if (NOT Eigen3_FOUND)
    find_package (Eigen3 ${hogpp_Eigen3_MIN_VERSION} REQUIRED CONFIG)
  endif (NOT Eigen3_FOUND)

  find_package (fmt ${hogpp_fmt_MIN_VERSION} REQUIRED CONFIG)
endif (BUILD_DEPS)

find_package (OpenCV 4.0 CONFIG COMPONENTS core)

if (WITH_PYTHON)
  # Development.SABIModule provides the Python::SABIModule target nanobind
  # needs to build stable ABI (abi3) extension modules. The component is not
  # available for PyPy.
  find_package (Python 3.11 COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)

  # nanobind's config computes the extension suffix itself (querying
  # sysconfig directly when SKBUILD_SOABI/Python_SOABI is unavailable or, as
  # on PyPy, missing the platform tag), so no manual SOABI handling is
  # needed here.
  if (Python_FOUND)
    find_package (nanobind 2.2.0 CONFIG)
  endif (Python_FOUND)

  find_package (Sphinx 4.0)
endif (WITH_PYTHON)

include (BoostTestDiscovery)
include (CheckCXXCompilerFlag)
include (CheckCXXSourceCompiles)
include (CheckCXXSymbolExists)
include (CMakePackageConfigHelpers)
include (GNUInstallDirs)

# Header-only packages are independent of the architecture.
set (hogpp_package_dir ${CMAKE_INSTALL_DATADIR}/cmake/hogpp)

check_cxx_compiler_flag (-Wa,-mbig-obj HAVE_MBIG_OBJ)
check_cxx_compiler_flag (/bigobj HAVE_BIGOBJ)
check_cxx_symbol_exists (std::execution::par execution HAVE_EXECUTION)

# Replacing the allocation functions of glibc allows tests to observe the heap.
check_cxx_source_compiles ("
#include <cstddef>
extern \"C\" void* __libc_malloc(std::size_t size);
int main() { return __libc_malloc(1) == nullptr; }
" HAVE_LIBC_MALLOC)

# Lifts the limit on the number of sections of an object file that heavily
# templated sources exceed.
function (hogpp_enable_big_obj target)
  if (HAVE_MBIG_OBJ)
    target_compile_options (${target} PRIVATE -Wa,-mbig-obj)
  elseif (HAVE_BIGOBJ)
    target_compile_options (${target} PRIVATE /bigobj)
  endif (HAVE_MBIG_OBJ)
endfunction (hogpp_enable_big_obj)

if (WIN32)
  add_compile_definitions (NOMINMAX)
  add_compile_definitions (WIN32_LEAN_AND_MEAN)
endif (WIN32)

add_library (hogpp INTERFACE)
add_library (hogpp::hogpp ALIAS hogpp)

target_sources (hogpp INTERFACE FILE_SET HEADERS
  BASE_DIRS include
  FILES
    include/hogpp/assume.hpp
    include/hogpp/axis.hpp
    include/hogpp/bounds.hpp
    include/hogpp/cartesianproduct.hpp
    include/hogpp/constants.hpp
    include/hogpp/gradient.hpp
    include/hogpp/gradientmagnitude.hpp
    include/hogpp/gradientsqrtmagnitude.hpp
    include/hogpp/gradientsquaremagnitude.hpp
    include/hogpp/integralhistogram.hpp
    include/hogpp/integralhogdescriptor.hpp
    include/hogpp/l1hys.hpp
    include/hogpp/l1norm.hpp
    include/hogpp/l1sqrt.hpp
    include/hogpp/l2hys.hpp
    include/hogpp/l2norm.hpp
    include/hogpp/multiplyadd.hpp
    include/hogpp/normalize.hpp
    include/hogpp/normtraits.hpp
    include/hogpp/nouniqueaddress.hpp
    include/hogpp/precision.hpp
    include/hogpp/promote.hpp
    include/hogpp/repeatborder.hpp
    include/hogpp/signedgradient.hpp
    include/hogpp/spatialinterpolation.hpp
    include/hogpp/spatialweighting.hpp
    include/hogpp/unsignedgradient.hpp
)
# The prefix and the suffix enclose the contents of other headers and do not
# compile on their own.
target_sources (hogpp INTERFACE FILE_SET fragments TYPE HEADERS
  BASE_DIRS include
  FILES
    include/hogpp/prefix.hpp
    include/hogpp/suffix.hpp
)

target_compile_features (hogpp INTERFACE cxx_std_20)
target_link_libraries (hogpp INTERFACE
  Eigen3::Eigen
  fmt::fmt
)

if (WITH_PYTHON AND Python_FOUND AND nanobind_FOUND)
  # Shared between the pyhogpp extension module and the type_caster_test
  # module used by the internal type caster tests. Each nanobind_add_module()
  # call below compiles its own private copy: nanobind extension modules are
  # self-contained shared objects, so there is no ODR concern, and it avoids
  # having to keep a helper static library in lockstep with whichever
  # STABLE_ABI/FREE_THREADED nanobind core library flavor each module ends up
  # linking against.
  set (HOGPP_PYTHON_TYPE_CASTER_SOURCES
    python/type_caster/bounds.cpp
    python/type_caster/maskingref.cpp
  )
  set (HOGPP_PYTHON_TYPE_CASTER_HEADERS
    python/type_caster/array2i.hpp
    python/type_caster/bounds.hpp
    python/type_caster/maskingref.hpp
    python/type_caster/stride.hpp
    python/type_caster/tensor.hpp
    python/type_caster/typesequence.hpp
  )

  if (OpenCV_FOUND)
    list (APPEND HOGPP_PYTHON_TYPE_CASTER_SOURCES python/type_caster/opencv.cpp)
    list (APPEND HOGPP_PYTHON_TYPE_CASTER_HEADERS python/type_caster/opencv.hpp)
  endif (OpenCV_FOUND)

  # STABLE_ABI/FREE_THREADED are no-ops (silently ignored) when the
  # configured Python interpreter cannot support them, e.g. below Python
  # 3.12, PyPy, or a free-threaded build for STABLE_ABI; a non-free-threaded
  # build for FREE_THREADED. NB_SUPPRESS_WARNINGS marks nanobind's (and
  # transitively Python's) include directories as SYSTEM, so warnings
  # originating in their headers, rather than in hogpp's own code, don't
  # trip project-level -Werror/-Wx builds.
  nanobind_add_module (pyhogpp STABLE_ABI FREE_THREADED NB_SUPPRESS_WARNINGS
    ${HOGPP_PYTHON_TYPE_CASTER_SOURCES}
    python/binning.cpp
    python/blocknormalizer.cpp
    python/hogpp.cpp
    python/integralhogdescriptor.cpp
    python/interpolation.cpp
    python/magnitude.cpp
  )

  target_sources (pyhogpp PRIVATE FILE_SET HEADERS
    BASE_DIRS python
    FILES
      ${HOGPP_PYTHON_TYPE_CASTER_HEADERS}
      python/binning.hpp
      python/blocknormalizer.hpp
      python/formatter.hpp
      python/hogpp.hpp
      python/integralhogdescriptor.hpp
      python/interpolation.hpp
      python/magnitude.hpp
  )

  hogpp_enable_big_obj (pyhogpp)

  if (OpenCV_FOUND)
    target_compile_definitions (pyhogpp PRIVATE HAVE_OPENCV)
    target_link_libraries (pyhogpp PRIVATE opencv_core)
  endif (OpenCV_FOUND)

  if (HAVE_EXECUTION)
    target_compile_definitions (pyhogpp PRIVATE HAVE_EXECUTION)
  endif (HAVE_EXECUTION)

  if (SKBUILD)
    set_target_properties (pyhogpp PROPERTIES OUTPUT_NAME _hogpp)
    set_property (SOURCE python/hogpp.cpp APPEND PROPERTY COMPILE_DEFINITIONS
      HOGPP_SKBUILD)
  else (SKBUILD)
    set_target_properties (pyhogpp PROPERTIES OUTPUT_NAME hogpp)
  endif (SKBUILD)

  target_link_libraries (pyhogpp PRIVATE
    hogpp::hogpp
    Eigen3::Eigen
  )

  nanobind_add_module (type_caster_test STABLE_ABI FREE_THREADED NB_SUPPRESS_WARNINGS
    ${HOGPP_PYTHON_TYPE_CASTER_SOURCES}
    python/type_caster_test.cpp
  )

  target_sources (type_caster_test PRIVATE FILE_SET HEADERS
    BASE_DIRS python
    FILES ${HOGPP_PYTHON_TYPE_CASTER_HEADERS}
  )

  hogpp_enable_big_obj (type_caster_test)

  if (OpenCV_FOUND)
    target_compile_definitions (type_caster_test PRIVATE HAVE_OPENCV)
    target_link_libraries (type_caster_test PRIVATE opencv_core)
  endif (OpenCV_FOUND)

  target_link_libraries (type_caster_test PRIVATE
    hogpp::hogpp
    Eigen3::Eigen
  )

  if (MINGW)
    # MinGW's libstdc++-6.dll import library does not export every
    # vague-linkage symbol (e.g. the out-of-line instantiation of
    # std::string's move constructor pulled in through fmt::format() and
    # std::vector<std::string>::push_back() in repr()), which otherwise
    # fails at link time with an undefined reference. Statically linking
    # the runtime instead pulls those symbols in directly, the same
    # workaround nanobind itself applies for musllinux wheels via
    # nanobind_musl_static_libcpp().
    target_link_options (pyhogpp PRIVATE -static-libstdc++ -static-libgcc)
    target_link_options (type_caster_test PRIVATE -static-libstdc++
      -static-libgcc)
  endif (MINGW)

  # NB_SUPPRESS_WARNINGS above only silences warnings that originate in
  # headers (e.g. Python's or nanobind's own), since those are reached via
  # a SYSTEM include directory. It cannot silence warnings from code
  # written directly in nanobind's own .cpp sources, e.g. a sign
  # conversion in nb_func.cpp. That source isn't ours to fix, so treating
  # warnings as errors is disabled for nanobind's compiled library
  # specifically, without weakening warnings for hogpp's own targets.
  foreach (_hogpp_nanobind_lib nanobind-static nanobind-static-abi3 nanobind-static-ft)
    if (TARGET ${_hogpp_nanobind_lib})
      if (MSVC)
        target_compile_options (${_hogpp_nanobind_lib} PRIVATE /WX-)
      else (MSVC)
        target_compile_options (${_hogpp_nanobind_lib} PRIVATE -Wno-error)
      endif (MSVC)
    endif (TARGET ${_hogpp_nanobind_lib})
  endforeach (_hogpp_nanobind_lib)
  unset (_hogpp_nanobind_lib)

  # macOS resolves Python symbols at load time only if the linker is told about
  # them. nanobind selects a list of PyPy-prefixed C API names for PyPy, but
  # PyPy 3.12 exports most of its C API under the names CPython uses. A module
  # references names from both lists, and names it does not use are ignored.
  if (APPLE AND Python_INTERPRETER_ID STREQUAL "PyPy")
    foreach (_hogpp_python_module IN ITEMS pyhogpp type_caster_test)
      target_link_options (${_hogpp_python_module} PRIVATE
        "-Wl,@${NB_DIR}/cmake/darwin-ld-cpython.sym")
    endforeach (_hogpp_python_module IN ITEMS pyhogpp type_caster_test)
  endif (APPLE AND Python_INTERPRETER_ID STREQUAL "PyPy")
endif (WITH_PYTHON AND Python_FOUND AND nanobind_FOUND)

if (hogpp_BUILD_TESTS)
  # Each public header compiles on its own.
  set_property (TARGET hogpp PROPERTY VERIFY_INTERFACE_HEADER_SETS ON)
  set_property (TARGET hogpp PROPERTY INTERFACE_HEADER_SETS_TO_VERIFY HEADERS)

  add_test (NAME hogpp_verify_interface_header_sets
    COMMAND ${CMAKE_COMMAND} --build ${PROJECT_BINARY_DIR}
      --target hogpp_verify_interface_header_sets --config $<CONFIG>
  )
  add_test (NAME hogpp_header_set_complete
    COMMAND ${CMAKE_COMMAND}
      -D "HEADERS=$<JOIN:$<TARGET_PROPERTY:hogpp,HEADER_SET>;$<TARGET_PROPERTY:hogpp,HEADER_SET_fragments>,|>"
      -D INCLUDE_DIR=${PROJECT_SOURCE_DIR}/include
      -D SOURCE_DIR=${PROJECT_SOURCE_DIR}
      -P ${PROJECT_SOURCE_DIR}/tests/cmake/header_set_complete.cmake
  )

  # The tests include the header-only variant of Boost.Test.
  if (Boost_FOUND)
    set (cpp_tests binning block_norm cartesian_product descriptor gradient
      integral_histogram maskingref multiply_add spatialweighting)

    if (HAVE_LIBC_MALLOC)
      list (APPEND cpp_tests allocations)
    endif (HAVE_LIBC_MALLOC)

    foreach (test IN LISTS cpp_tests)
      add_executable (test_${test} tests/cpp/test_${test}.cpp)
      target_link_libraries (test_${test} PRIVATE
        Boost::headers
        hogpp::hogpp
      )
      boost_test_discover_tests (test_${test} TEST_PREFIX test_${test}/)
    endforeach (test)

    hogpp_enable_big_obj (test_descriptor)
    hogpp_enable_big_obj (test_spatialweighting)
  endif (Boost_FOUND)

  # Consumers find the installed package or add the source tree as a
  # subdirectory, and build with the compiler and generator of this build.
  set (consumer_options
    --build-generator ${CMAKE_GENERATOR}
    --build-config $<CONFIG>
  )

  if (CMAKE_GENERATOR_PLATFORM)
    list (APPEND consumer_options
      --build-generator-platform ${CMAKE_GENERATOR_PLATFORM})
  endif (CMAKE_GENERATOR_PLATFORM)

  if (CMAKE_GENERATOR_TOOLSET)
    list (APPEND consumer_options
      --build-generator-toolset ${CMAKE_GENERATOR_TOOLSET})
  endif (CMAKE_GENERATOR_TOOLSET)

  set (consumer_build_options -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})

  if (CMAKE_TOOLCHAIN_FILE)
    list (APPEND consumer_build_options
      -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
  endif (CMAKE_TOOLCHAIN_FILE)

  set (consumer_test_command
    --test-command ${CMAKE_CTEST_COMMAND} --output-on-failure
      --build-config $<CONFIG>
  )

  if (WITH_INSTALL)
    set (consumer_prefix ${PROJECT_BINARY_DIR}/consumer-prefix)

    # Files an earlier run left in the prefix would hide those the install
    # rules no longer install.
    add_test (NAME package_clean
      COMMAND ${CMAKE_COMMAND} -E rm -rf ${consumer_prefix}
    )
    add_test (NAME package_install
      COMMAND ${CMAKE_COMMAND} --install ${PROJECT_BINARY_DIR}
        --prefix ${consumer_prefix} --config $<CONFIG>
        --component Development
    )
    # Toolchains for cross compiling restrict the search for packages to the
    # sysroot, which the prefix is not part of.
    if (CMAKE_CROSSCOMPILING)
      set (consumer_find_option -Dhogpp_DIR=${consumer_prefix}/${hogpp_package_dir})
    else (CMAKE_CROSSCOMPILING)
      set (consumer_find_option -DCMAKE_PREFIX_PATH=${consumer_prefix})
    endif (CMAKE_CROSSCOMPILING)

    add_test (NAME package_consumer
      COMMAND ${CMAKE_CTEST_COMMAND}
        --build-and-test ${PROJECT_SOURCE_DIR}/tests/consumer
          ${PROJECT_BINARY_DIR}/consumer-package
        ${consumer_options}
        --build-options ${consumer_build_options} ${consumer_find_option}
          -DHOGPP_REQUIRED_VERSION=${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
        ${consumer_test_command}
    )

    set_tests_properties (package_clean PROPERTIES
      FIXTURES_SETUP hogpp_prefix
    )
    set_tests_properties (package_install PROPERTIES
      FIXTURES_REQUIRED hogpp_prefix
      FIXTURES_SETUP hogpp_package
    )
    set_tests_properties (package_consumer PROPERTIES
      FIXTURES_REQUIRED hogpp_package
    )

    # The package rejects requests for an earlier minor version, since
    # releases before 1.0 may break the interface with each minor version.
    if (PROJECT_VERSION_MINOR GREATER 0)
      math (EXPR previous_minor "${PROJECT_VERSION_MINOR} - 1")

      add_test (NAME package_previous_minor_version
        COMMAND ${CMAKE_CTEST_COMMAND}
          --build-and-test ${PROJECT_SOURCE_DIR}/tests/consumer
            ${PROJECT_BINARY_DIR}/consumer-previous-minor-version
          ${consumer_options}
          --build-options ${consumer_build_options} ${consumer_find_option}
            -DHOGPP_REQUIRED_VERSION=${PROJECT_VERSION_MAJOR}.${previous_minor}
      )

      set_tests_properties (package_previous_minor_version PROPERTIES
        FIXTURES_REQUIRED hogpp_package
        PASS_REGULAR_EXPRESSION
          "considered but not accepted:\n\n *[^\n]*/hogpp-config.cmake, version: ${PROJECT_VERSION}\n"
      )
    endif (PROJECT_VERSION_MINOR GREATER 0)
  endif (WITH_INSTALL)

  # The consumer runs its own test only, since hogpp registers its tests as
  # the top-level project alone.
  if (NOT BUILD_DEPS)
    add_test (NAME subdirectory_consumer
      COMMAND ${CMAKE_CTEST_COMMAND}
        --build-and-test ${PROJECT_SOURCE_DIR}/tests/consumer
          ${PROJECT_BINARY_DIR}/consumer-subdirectory
        ${consumer_options}
        --build-options ${consumer_build_options}
          -DHOGPP_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        ${consumer_test_command}
    )

    # CTest 4.4 omits the count of failed tests from the summary if there are
    # none.
    set_tests_properties (subdirectory_consumer PROPERTIES
      PASS_REGULAR_EXPRESSION "100% tests passed(, 0 tests failed)? out of 1\n"
    )
  endif (NOT BUILD_DEPS)

  # pytest-xdist running on PyPy with extension modules compiled using MSVC runs
  # extremely slow. Avoid using xdist.
  if (Python_INTERPRETER_ID STREQUAL "PyPy" AND MSVC)
    set (pytest_xdist_args)
  else (Python_INTERPRETER_ID STREQUAL "PyPy" AND MSVC)
    set (pytest_xdist_args -n auto)
  endif (Python_INTERPRETER_ID STREQUAL "PyPy" AND MSVC)

  set (pytest_command Python::Interpreter -m pytest -W error ${pytest_xdist_args})

  if (TARGET pyhogpp AND TARGET Python::Interpreter)
    add_test (NAME python_hogpp
      COMMAND ${pytest_command}
        ${PROJECT_SOURCE_DIR}/tests/python/module/test_hogpp.py
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
    add_test (NAME python_tensor_type_caster
      COMMAND ${pytest_command}
        ${PROJECT_SOURCE_DIR}/tests/python/internal/test_tensor_type_caster.py
    )
    add_test (NAME python_repr
      COMMAND ${pytest_command}
        ${PROJECT_SOURCE_DIR}/tests/python/module/test_repr.py
    )

    set_tests_properties (python_hogpp python_tensor_type_caster python_repr
      PROPERTIES
        ENVIRONMENT_MODIFICATION
          PYTHONPATH=path_list_prepend:$<TARGET_FILE_DIR:pyhogpp>
    )
  endif (TARGET pyhogpp AND TARGET Python::Interpreter)

  if (TARGET type_caster_test AND TARGET Python::Interpreter AND OpenCV_FOUND)
    add_test (NAME python_opencv_type_caster
      COMMAND ${pytest_command}
        ${PROJECT_SOURCE_DIR}/tests/python/internal/test_opencv_type_caster.py
    )

    set_tests_properties (python_opencv_type_caster PROPERTIES
      ENVIRONMENT_MODIFICATION
        PYTHONPATH=path_list_prepend:$<TARGET_FILE_DIR:type_caster_test>
    )
  endif (TARGET type_caster_test AND TARGET Python::Interpreter AND OpenCV_FOUND)
endif (hogpp_BUILD_TESTS)

if (TARGET pyhogpp AND Sphinx_FOUND)
  add_custom_target (sphinx
    ${CMAKE_COMMAND} -E env
      PYTHONPATH=$<TARGET_FILE_DIR:pyhogpp>
        $<TARGET_FILE:Sphinx::build> -M html
        ${PROJECT_SOURCE_DIR}/docs
        ${PROJECT_BINARY_DIR}/docs/_build
        -W --keep-going
    DEPENDS pyhogpp
    BYPRODUCTS docs/_build/html/.buildinfo
    COMMENT "Generating Sphinx HTML documentation"
    USES_TERMINAL
  )

  set_property (DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES docs/_build)
endif (TARGET pyhogpp AND Sphinx_FOUND)

if (TARGET pyhogpp AND SKBUILD)
  install (TARGETS pyhogpp
    LIBRARY DESTINATION ${PROJECT_NAME} COMPONENT Runtime
    RUNTIME DESTINATION ${PROJECT_NAME} COMPONENT Runtime
  )
endif (TARGET pyhogpp AND SKBUILD)

if (WITH_INSTALL)
  install (TARGETS hogpp
    EXPORT hogpp-targets
    FILE_SET HEADERS COMPONENT Development
    FILE_SET fragments COMPONENT Development
  )
  install (EXPORT hogpp-targets
    NAMESPACE hogpp::
    DESTINATION ${hogpp_package_dir}
    COMPONENT Development
  )

  configure_package_config_file (cmake/hogpp-config.cmake.in
    ${PROJECT_BINARY_DIR}/hogpp-config.cmake
    INSTALL_DESTINATION ${hogpp_package_dir}
  )
  # Releases before 1.0 may break the interface with each minor version.
  write_basic_package_version_file (${PROJECT_BINARY_DIR}/hogpp-config-version.cmake
    COMPATIBILITY SameMinorVersion
    ARCH_INDEPENDENT
  )

  install (FILES
    ${PROJECT_BINARY_DIR}/hogpp-config.cmake
    ${PROJECT_BINARY_DIR}/hogpp-config-version.cmake
    DESTINATION ${hogpp_package_dir}
    COMPONENT Development
  )
endif (WITH_INSTALL)
