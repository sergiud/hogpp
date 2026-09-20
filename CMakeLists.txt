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
option (WITH_DISPATCH
  "Build one pyhogpp variant per supported instruction set and pick the \
fastest at runtime. When OFF, build a single variant using the compiler's \
default target settings instead (e.g. -march=native if the user set it)."
  ON
)
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
  set (PYBIND11_NOPYTHON ON)

  # Python precedes pybind11 whose package configuration otherwise searches
  # for Python itself.
  find_package (Python 3.11 COMPONENTS Development.Module
    OPTIONAL_COMPONENTS Interpreter)

  if (SKBUILD_SOABI)
    set (Python_SOABI ${SKBUILD_SOABI})
  endif (SKBUILD_SOABI)

  if (MSVC AND MSVC_VERSION GREATER_EQUAL 1930)
    find_package (pybind11 2.9.0 CONFIG)
  else (MSVC AND MSVC_VERSION GREATER_EQUAL 1930)
    find_package (pybind11 2.6.2 CONFIG)
  endif (MSVC AND MSVC_VERSION GREATER_EQUAL 1930)

  find_package (Sphinx 4.0)
endif (WITH_PYTHON)

include (AddISA)
include (BoostTestDiscovery)
include (CheckCXXCompilerFlag)
include (CheckCXXSourceCompiles)
include (CheckCXXSymbolExists)
include (CheckIncludeFileCXX)
include (CMakePackageConfigHelpers)
include (GNUInstallDirs)

# Header-only packages are independent of the architecture.
set (hogpp_package_dir ${CMAKE_INSTALL_DATADIR}/cmake/hogpp)

check_cxx_compiler_flag (-Wa,-mbig-obj HAVE_MBIG_OBJ)
check_cxx_compiler_flag (/bigobj HAVE_BIGOBJ)
check_cxx_compiler_flag (/Zc:preprocessor HAVE_ZC_PREPROCESSOR)
check_cxx_compiler_flag (-mtune=generic HAVE_MTUNE_GENERIC)
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
    include/hogpp/fastatan.hpp
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

if (WITH_PYTHON AND Python_FOUND AND pybind11_FOUND)
  check_cxx_compiler_flag (-mfpu=neon HAVE_MFPU_NEON)
  check_cxx_compiler_flag (-msve-vector-bits=scalable HAVE_MSVE_VECTOR_BITS)

  check_cxx_symbol_exists (__check_isa_support "immintrin.h" HAVE___CHECK_ISA_SUPPORT)
  check_cxx_symbol_exists (getauxval "sys/auxv.h" HAVE_GETAUXVAL)
  check_cxx_symbol_exists (prctl "sys/prctl.h" HAVE_PRCTL)
  check_include_file_cxx ("asm/hwcap.h" HAVE_ASM_HWCAP_H)
  check_include_file_cxx ("sys/auxv.h" HAVE_SYS_AUXV_H)
  check_include_file_cxx ("sys/prctl.h" HAVE_SYS_PRCTL_H)

  set (HOGPP_ISAS)

  if (WITH_DISPATCH)
    # Each hogpp_add_isa() call below tests both the GNU/Clang and MSVC flag
    # spelling for the same canonical ISA name, appending it to HOGPP_ISAS on
    # success. The canonical name (used for the HAVE_ISA_* macro, the
    # pyhogpp_<name> target/init_hogpp_<name> symbol, and the Eigen
    # vectorization define) is independent of the compiler flag spelling,
    # e.g. AVX512 pairs with -mavx512f on GNU/Clang and /arch:AVX512 on MSVC.
    hogpp_add_isa (SSE2 GNU_FLAG -msse2 MSVC_FLAG /arch:SSE2)
    hogpp_add_isa (SSE3 GNU_FLAG -msse3)
    hogpp_add_isa (SSSE3 GNU_FLAG -mssse3)
    hogpp_add_isa (SSE4.1 GNU_FLAG -msse4.1)
    hogpp_add_isa (SSE4.2 GNU_FLAG -msse4.2 MSVC_FLAG /arch:SSE4.2)
    hogpp_add_isa (AVX GNU_FLAG -mavx MSVC_FLAG /arch:AVX)
    # Every AVX2-capable CPU (Haswell and later on Intel, all AVX2-capable
    # AMD chips) also implements FMA3: the two were introduced together
    # and no shipped CPU has one without the other. Unlike plain AVX
    # (Sandy/Ivy Bridge predate FMA3), requiring FMA here needs no
    # additional runtime check.
    hogpp_add_isa (AVX2 GNU_FLAG -mavx2 MSVC_FLAG /arch:AVX2 REQUIRES_FMA)
    # Required for AVX512 (and above) by Eigen 5.x: compiling with -mavx512f
    # alone without SSE/AVX FMA is not supported (bug 1638)
    # https://gitlab.com/libeigen/eigen/-/issues/1638
    hogpp_add_isa (AVX512 GNU_FLAG -mavx512f MSVC_FLAG /arch:AVX512 REQUIRES_FMA)

    # Avoid the following Eigen error when compiling using AppleClang:
    #   error: invalid output size for constraint '=v'
    #     360 |   __asm__("vbroadcastss %[mem], %[dst]" : [dst] "=v"(ret) : [mem] "m"(*from));
    # when compiling using -mavx10.1 by explicitly setting the word size.
    # Eigen assumes 512 bit vectors while AppleClang -mavx10.1 defaults to
    # 256 bits.
    if (NOT (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"))
      hogpp_add_isa (AVX10.1 GNU_FLAG -mavx10.1 MSVC_FLAG /arch:AVX10.1 REQUIRES_FMA)
      hogpp_add_isa (AVX10.2 GNU_FLAG -mavx10.2 REQUIRES_FMA)
    endif (NOT (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"))

    # CMAKE_SYSTEM_PROCESSOR alone is not reliable for the Visual Studio
    # generator: with an explicit -A arm64 (as used for cross-generating an
    # ARM64 solution), it does not necessarily get updated to reflect the
    # requested target architecture. CMAKE_GENERATOR_PLATFORM always mirrors
    # the -A value verbatim for VS generators, so check both.
    string (TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _hogpp_system_processor)
    string (TOLOWER "${CMAKE_GENERATOR_PLATFORM}" _hogpp_generator_platform)

    if (_hogpp_system_processor MATCHES "^(arm|aarch64)" OR
        _hogpp_generator_platform MATCHES "^arm")
      if (HAVE_MFPU_NEON OR
          _hogpp_system_processor MATCHES "^aarch64" OR
          _hogpp_generator_platform MATCHES "^arm64")
        list (APPEND HOGPP_ISAS NEON)
        set (HAVE_ISA_NEON 1)
      endif (HAVE_MFPU_NEON OR
             _hogpp_system_processor MATCHES "^aarch64" OR
             _hogpp_generator_platform MATCHES "^arm64")

      if (HAVE_MSVE_VECTOR_BITS)
        set (HAVE_ISA_SVE128 1)
        set (HAVE_ISA_SVE256 1)
        set (HAVE_ISA_SVE512 1)
        list (APPEND HOGPP_ISAS SVE128 SVE256 SVE512)
      endif (HAVE_MSVE_VECTOR_BITS)
    endif (_hogpp_system_processor MATCHES "^(arm|aarch64)" OR
           _hogpp_generator_platform MATCHES "^arm")

    unset (_hogpp_system_processor)
    unset (_hogpp_generator_platform)
  endif (WITH_DISPATCH)

  configure_file (
    python/isaconfig.hpp.cmake.in
    python/isaconfig.hpp
    @ONLY
  )

  add_library (python_type_caster STATIC
    python/type_caster/bounds.cpp
    python/type_caster/maskingref.cpp
  )

  target_sources (python_type_caster PRIVATE FILE_SET HEADERS
    BASE_DIRS python
    FILES
      python/type_caster/bounds.hpp
      python/type_caster/maskingref.hpp
      python/type_caster/stride.hpp
      python/type_caster/tensor.hpp
      python/type_caster/typesequence.hpp
  )

  if (OpenCV_FOUND)
    target_sources (python_type_caster PRIVATE python/type_caster/opencv.cpp)
    target_sources (python_type_caster PRIVATE FILE_SET HEADERS
      FILES python/type_caster/opencv.hpp
    )
    target_link_libraries (python_type_caster PUBLIC opencv_core)
  endif (OpenCV_FOUND)

  target_link_libraries (python_type_caster
    PUBLIC
      Eigen3::Eigen
      hogpp::hogpp
      pybind11::headers
    PRIVATE
      Python::Module
  )

  python_add_library (pyhogpp MODULE WITH_SOABI
    python/cpufeature.cpp
    python/cpufeature.hpp
    python/cpufeatures.cpp
    python/cpufeatures.hpp
    python/dispatchinit.cpp
    python/isa.hpp
    python/levenshtein.hpp
    python/moduledispatch.cpp
    python/moduledispatch.hpp
    python/moduleinitializer.cpp
    python/moduleinitializer.hpp
  )
  target_compile_features (pyhogpp PRIVATE cxx_std_20)

  target_include_directories (pyhogpp PRIVATE
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/python>
  )

  # The mask is converted once before dispatching, so its type caster does not
  # depend on the instruction set and is shared by all dispatch modules.
  target_link_libraries (pyhogpp PRIVATE
    fmt::fmt
    pybind11::headers
    Python::Module
    python_type_caster
  )

  set (_arch
    generic
    ${HOGPP_ISAS}
  )

  foreach (arch IN LISTS _arch)
    string (TOLOWER "${arch}" _lower_arch)
    string (MAKE_C_IDENTIFIER "${_lower_arch}" _suffix)
    string (MAKE_C_IDENTIFIER "${arch}" _eigen_suffix)
    set (target_name pyhogpp_${_suffix})

    add_library (${target_name} OBJECT
      python/binning.cpp
      python/binning.hpp
      python/blocknormalizer.cpp
      python/blocknormalizer.hpp
      python/formatter.hpp
      python/hogpp.cpp
      python/hogpp.hpp
      python/integralhogdescriptor.cpp
      python/integralhogdescriptor.hpp
      python/interpolation.cpp
      python/interpolation.hpp
      python/magnitude.cpp
      python/magnitude.hpp
      python/typetraits.hpp
      # We cannot reuse the python_type_caster library since we need to rename
      # namespaces.
      #
      # While bounds.[ch]pp do not depend on a specific ABI, we rename the
      # namespace which requires treating the sources as ABI specific.
      python/type_caster/bounds.cpp
      python/type_caster/bounds.hpp
      python/type_caster/stride.hpp
      python/type_caster/tensor.hpp
      python/type_caster/typesequence.hpp
    )

    if (arch STREQUAL generic)
      target_compile_definitions (${target_name} PRIVATE
        HOGPP_TARGET=${_suffix}
      )

      if (WITH_DISPATCH)
        # This is specifically the runtime fallback used when no other
        # compiled-in ISA matches the CPU it ends up running on, so it must
        # stay portable: don't vectorize it (avoiding the same code
        # generation as one of the other modules) and don't tune for the
        # build machine's microarchitecture.
        target_compile_definitions (${target_name} PRIVATE EIGEN_DONT_VECTORIZE)

        if (HAVE_MTUNE_GENERIC)
          target_compile_options (${target_name} PRIVATE -mtune=generic)
        endif (HAVE_MTUNE_GENERIC)
      endif (WITH_DISPATCH)
    else (arch STREQUAL generic)
      target_compile_definitions (${target_name} PRIVATE
        HOGPP_TARGET=${_suffix}
      )

      # Real ISA-specific dispatch objects (as opposed to the portable
      # generic fallback above) may replace exact but non-vectorizable
      # libm calls in per-pixel hot paths with cheaper approximations,
      # such as the atan2/atan calls used for gradient binning (see
      # include/hogpp/signedgradient.hpp and unsignedgradient.hpp). The
      # generic fallback keeps the exact std::atan2/std::atan behavior.
      target_compile_definitions (${target_name} PRIVATE HOGPP_FAST_MATH)

      if (arch STREQUAL NEON)
        target_compile_definitions (${target_name} PRIVATE EIGEN_VECTORIZE_NEON)

        if (HAVE_MFPU_NEON)
          target_compile_options (${target_name} PRIVATE -mfpu=neon)
        endif (HAVE_MFPU_NEON)
      elseif (arch MATCHES "^SVE([0-9]+)$")
        target_compile_definitions (${target_name} PRIVATE EIGEN_VECTORIZE_SVE)
        target_compile_options (${target_name} PRIVATE -msve-vector-bits=${CMAKE_MATCH_1})
      else (arch STREQUAL NEON)
        target_compile_options (${target_name} PRIVATE
          ${hogpp_isa_${_eigen_suffix}_flags}
        )

        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
          if (hogpp_isa_${_eigen_suffix}_requires_fma)
            target_compile_options (${target_name} PRIVATE -mfma)
          endif (hogpp_isa_${_eigen_suffix}_requires_fma)
        endif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      endif (arch STREQUAL NEON)

      # Rename the Eigen namespace to avoid One Definition Rule (ODR) violations
      target_compile_definitions (${target_name} PRIVATE
        EIGEN_VECTORIZE_${_eigen_suffix}
        Eigen=Eigen_${_suffix}
        hogpp=hogpp_${_suffix}
      )
    endif (arch STREQUAL generic)

    hogpp_enable_big_obj (${target_name})

    # The conforming preprocessor of MSVC evaluates the variadic macros of the
    # free-threading support.
    if (HAVE_ZC_PREPROCESSOR)
      target_compile_options (${target_name} PRIVATE /Zc:preprocessor)
    endif (HAVE_ZC_PREPROCESSOR)

    if (OpenCV_FOUND)
      target_compile_definitions (${target_name} PRIVATE HAVE_OPENCV)
      target_link_libraries (${target_name} PRIVATE opencv_core)
    endif (OpenCV_FOUND)

    if (HAVE_EXECUTION)
      target_compile_definitions (${target_name} PRIVATE HAVE_EXECUTION)
    endif (HAVE_EXECUTION)

    target_link_libraries (${target_name} PRIVATE
      hogpp::hogpp
      pybind11::headers
      Python::Module
    )

    target_link_libraries (pyhogpp PRIVATE ${target_name})
  endforeach (arch)

  unset (target_name)

  if (HAVE_ZC_PREPROCESSOR)
    target_compile_options (pyhogpp PRIVATE /Zc:preprocessor)
  endif (HAVE_ZC_PREPROCESSOR)

  if (SKBUILD)
    set_target_properties (pyhogpp PROPERTIES OUTPUT_NAME _hogpp)
    set_property (SOURCE python/dispatchinit.cpp APPEND PROPERTY
      COMPILE_DEFINITIONS HOGPP_SKBUILD)
  else (SKBUILD)
    set_target_properties (pyhogpp PROPERTIES OUTPUT_NAME hogpp)
  endif (SKBUILD)

  python_add_library (type_caster_test MODULE WITH_SOABI
    python/type_caster_test.cpp
  )

  if (OpenCV_FOUND)
    target_compile_definitions (type_caster_test PRIVATE HAVE_OPENCV)
  endif (OpenCV_FOUND)

  target_link_libraries (type_caster_test PRIVATE
    Eigen3::Eigen
    python_type_caster
  )
endif (WITH_PYTHON AND Python_FOUND AND pybind11_FOUND)

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
      integral_histogram levenshtein maskingref multiply_add spatialweighting)

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

  if (TARGET pyhogpp AND UNIX)
    add_test (NAME cmake_arm32_neon_requires_compiler_support
      COMMAND ${CMAKE_COMMAND}
        -DPROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -DTEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/cmake_arm32_neon
        -DTEST_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -P ${PROJECT_SOURCE_DIR}/tests/cmake/test_arm32_neon_dispatch.cmake
    )
  endif (TARGET pyhogpp AND UNIX)

  # Benchmarks run on their own, so their plugin stays disabled in the tests.
  set (pytest_command Python::Interpreter -m pytest -p no:benchmark -W error
    ${pytest_xdist_args})

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

    # Each dispatch test imports the module in subprocesses of its own.
    add_test (NAME python_dispatch
      COMMAND Python::Interpreter -m pytest -p no:benchmark -W error
        ${PROJECT_SOURCE_DIR}/tests/python/module/test_dispatch.py
    )

    set_tests_properties (python_hogpp python_tensor_type_caster python_repr
      python_dispatch PROPERTIES
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
