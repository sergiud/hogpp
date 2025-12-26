cmake_minimum_required (VERSION 3.18...4.2.1)
project (hogpp
  VERSION 0.3.0
  LANGUAGES CXX
  DESCRIPTION "Fast histogram of oriented gradients computation using integral histogram"
  HOMEPAGE_URL https://github.com/sergiud/hogpp
)

option (WITH_DISPATCH "Dynamic ISA dispatch" ON)

# Disable module support since under MinGW Clang compiling fails with the error
# '"CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS-NOTFOUND"' is not recognized as an
# internal or external command'. On Linux, the following Clang bug additionally
# prevents compilation: https://github.com/llvm/llvm-project/issues/88896
set (CMAKE_CXX_SCAN_FOR_MODULES OFF)
set (CMAKE_CXX_VISIBILITY_PRESET hidden)
set (CMAKE_POSITION_INDEPENDENT_CODE ON)
set (CMAKE_VISIBILITY_INLINES_HIDDEN ON)

list (APPEND CMAKE_MODULE_PATH ${hogpp_SOURCE_DIR}/cmake)

include (CheckCXXCompilerFlag)
include (CheckCXXSymbolExists)
include (CheckIncludeFileCXX)
include (CTest)
include (GNUInstallDirs)

check_cxx_compiler_flag (-Wa,-mbig-obj HAVE_MBIG_OBJ)
check_cxx_compiler_flag (/bigobj HAVE_BIGOBJ)
check_cxx_compiler_flag (/Zc:preprocessor HAVE_ZC_PREPROCESSOR)
check_cxx_compiler_flag (-mtune=generic HAVE_MTUNE_GENERIC)

set (PYBIND11_NOPYTHON ON)

find_package (Boost 1.70 COMPONENTS unit_test_framework NO_MODULE)
find_package (OpenCV 4.0 COMPONENTS core NO_MODULE)
find_package (Python 3.0 COMPONENTS Development.Module OPTIONAL_COMPONENTS Interpreter)
find_package (Sphinx 4.0)

if ("$ENV{CIBUILDWHEEL}" EQUAL 1)
  set (CIBUILDWHEEL ON)
else ("$ENV{CIBUILDWHEEL}" EQUAL 1)
  set (CIBUILDWHEEL OFF)
endif ("$ENV{CIBUILDWHEEL}" EQUAL 1)

if (NOT CIBUILDWHEEL)
  find_package (Eigen3 3.4.1...5 NO_MODULE)

  if (NOT Eigen3_FOUND)
    find_package (Eigen3 3.3.7 REQUIRED NO_MODULE)
  endif (NOT Eigen3_FOUND)

  find_package (fmt 6.0 REQUIRED)
else (NOT CIBUILDWHEEL)
  include (FetchContent)

  # Use EXCLUDE_FROM_ALL to avoid installing headers and libraries into the
  # wheels.

  FetchContent_Declare (eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0.1
    EXCLUDE_FROM_ALL
  )

  FetchContent_Declare (fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 12.1.0
    EXCLUDE_FROM_ALL
  )

  FetchContent_MakeAvailable (eigen3)
  FetchContent_MakeAvailable (fmt)
endif (NOT CIBUILDWHEEL)

if (SKBUILD_SOABI)
  set (Python_SOABI ${SKBUILD_SOABI})
else (SKBUILD_SOABI)
  if (Python_INTERPRETER_ID STREQUAL "PyPy")
    # Correct SOABI which is missing the platform tag. See
    # https://github.com/pypy/pypy/issues/3815.
    execute_process (COMMAND
      ${Python_EXECUTABLE} -c
      "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX').split('.')[1])"
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE Python_SOABI
      # FIXME: Use COMMAND_ERROR_IS_FATAL ANY when CMake >= 3.19
    )
  endif (Python_INTERPRETER_ID STREQUAL "PyPy")
endif (SKBUILD_SOABI)

if (MSVC AND MSVC_VERSION GREATER_EQUAL 1930)
  find_package (pybind11 2.9.0 NO_MODULE)
else (MSVC AND MSVC_VERSION GREATER_EQUAL 1930)
  find_package (pybind11 2.6.2 NO_MODULE)
endif (MSVC AND MSVC_VERSION GREATER_EQUAL 1930)

check_cxx_symbol_exists (std::execution::par execution HAVE_EXECUTION)

if (WIN32)
  add_compile_definitions (NOMINMAX)
  add_compile_definitions (WIN32_LEAN_AND_MEAN)
endif (WIN32)

if (HAVE_ZC_PREPROCESSOR)
  add_compile_options (/Zc:preprocessor)
endif (HAVE_ZC_PREPROCESSOR)

add_library (hogpp INTERFACE)
add_library (hogpp::hogpp ALIAS hogpp)

target_include_directories (hogpp INTERFACE
  $<BUILD_INTERFACE:${hogpp_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features (hogpp INTERFACE cxx_std_20)
target_link_libraries (hogpp INTERFACE
  Eigen3::Eigen
  fmt::fmt
)

if (pybind11_FOUND AND Python_Development.Module_FOUND)
  check_cxx_symbol_exists (__aarch64__ "" HAVE_ARCH_AARCH64)
  check_cxx_symbol_exists (__arm__ "" HAVE_ARCH_ARM)
  check_cxx_compiler_flag (-mfpu=neon HAVE_MFPU_NEON)
  check_cxx_compiler_flag (-msve-vector-bits=scalable HAVE_MSVE_VECTOR_BITS)

  if (HAVE_ARCH_AARCH64 OR HAVE_ARCH_ARM)
    set (HAVE_ISA_NEON 1)
  endif (HAVE_ARCH_AARCH64 OR HAVE_ARCH_ARM)

  check_cxx_symbol_exists (__check_isa_support "immintrin.h" HAVE___CHECK_ISA_SUPPORT)
  check_cxx_symbol_exists (getauxval "sys/auxv.h" HAVE_GETAUXVAL)
  check_include_file_cxx ("asm/hwcap.h" HAVE_ASM_HWCAP_H)
  check_include_file_cxx ("sys/auxv.h" HAVE_SYS_AUXV_H)

  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set (_supported
      SSE2
      SSE3
      SSSE3
      SSE4.1
      SSE4.2
      AVX
      AVX2
      AVX512F
      AVX10.1
      AVX10.2
    )

    check_cxx_compiler_flag (-msse2 HAVE_ISA_SSE2)
    check_cxx_compiler_flag (-msse3 HAVE_ISA_SSE3)
    check_cxx_compiler_flag (-mssse3 HAVE_ISA_SSSE3)
    check_cxx_compiler_flag (-msse4.1 HAVE_ISA_SSE4_1)
    check_cxx_compiler_flag (-msse4.2 HAVE_ISA_SSE4_2)
    check_cxx_compiler_flag (-mavx HAVE_ISA_AVX)
    check_cxx_compiler_flag (-mavx2 HAVE_ISA_AVX2)
    check_cxx_compiler_flag (-mavx512f HAVE_ISA_AVX512F)
    check_cxx_compiler_flag (-mavx10.1 HAVE_ISA_AVX10_1)
    check_cxx_compiler_flag (-mavx10.2 HAVE_ISA_AVX10_2)
  elseif (MSVC)
    set (_supported
      SSE2
      SSE4.2
      AVX
      AVX2
      AVX512
      AVX10.1
    )

    check_cxx_compiler_flag (/arch:SSE2 HAVE_ISA_SSE2)
    check_cxx_compiler_flag (/arch:SSE4.2 HAVE_ISA_SSE4_2)
    check_cxx_compiler_flag (/arch:AVX HAVE_ISA_AVX)
    check_cxx_compiler_flag (/arch:AVX2 HAVE_ISA_AVX2)
    check_cxx_compiler_flag (/arch:AVX512 HAVE_ISA_AVX512)
    check_cxx_compiler_flag (/arch:AVX10.1 HAVE_ISA_AVX10_1)
  endif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")

  list (APPEND _supported NEON)

  if (HAVE_MSVE_VECTOR_BITS)
    set (HAVE_ISA_SVE128 1)
    set (HAVE_ISA_SVE256 1)
    set (HAVE_ISA_SVE512 1)
    list (APPEND _supported SVE128)
    list (APPEND _supported SVE256)
    list (APPEND _supported SVE512)
  endif (HAVE_MSVE_VECTOR_BITS)

  configure_file (
    python/isaconfig.hpp.cmake.in
    python/isaconfig.hpp
    @ONLY
  )

  add_library (python_type_caster STATIC
    python/type_caster/bounds.cpp
    python/type_caster/bounds.hpp
    python/type_caster/stride.hpp
    python/type_caster/tensor.hpp
    python/type_caster/typesequence.hpp
  )

  if (OpenCV_FOUND)
    target_sources (python_type_caster PRIVATE
      python/type_caster/opencv.cpp
      python/type_caster/opencv.hpp
    )
    target_link_libraries (python_type_caster PUBLIC opencv_core)
  endif (OpenCV_FOUND)

  target_link_libraries (python_type_caster PUBLIC
    Eigen3::Eigen
    hogpp::hogpp
    pybind11::headers
    PRIVATE Python::Module
  )

  python_add_library (pyhogpp WITH_SOABI
    python/cpufeature.cpp
    python/cpufeature.hpp
    python/cpufeatures.hpp
    python/cpufeatures.cpp
    python/dispatchinit.cpp
    python/isa.hpp
    python/moduledispatch.cpp
    python/moduledispatch.hpp
    python/moduleinitializer.cpp
    python/moduleinitializer.hpp
  )
  target_compile_features (pyhogpp PRIVATE cxx_std_20)

  target_include_directories (pyhogpp PRIVATE
    $<BUILD_INTERFACE:${hogpp_BINARY_DIR}/python>
  )

  target_link_libraries (pyhogpp PRIVATE
    fmt::fmt
    pybind11::headers
    Python::Module
  )

  # TODO aarch64 Eigen 3.4.0 EIGEN_ARM64_USE_SVE -msve-vector-bits=N

  set (_arch
    generic
  )

  foreach (isa IN LISTS _supported)
    string (TOLOWER "${isa}" _lower_isa)
    string (MAKE_C_IDENTIFIER "${isa}" _suffix)

    if (HAVE_ISA_${_suffix})
      list (APPEND _arch ${isa})
    endif (HAVE_ISA_${_suffix})
  endforeach (isa)

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
      python/magnitude.cpp
      python/magnitude.hpp
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
        # Don't vector the generic dispatch to avoid same code generation for
        # different modules
        EIGEN_DONT_VECTORIZE
        HOGPP_TARGET=${_suffix}
      )
      # set_property (TARGET ${target_name} APPEND PROPERTY COMPILE_DEFINITIONS
      #   $<$<TARGET_NAME:${target_name}>:HOGPP_TARGET="${arch}">
      # )

      if (HAVE_MTUNE_GENERIC)
        target_compile_options (${target_name} PRIVATE -mtune=generic)
      endif (HAVE_MTUNE_GENERIC)
    else (arch STREQUAL generic)
      #set_source_files_properties (python/hogpp.cpp
      #  TARGET_DIRECTORY ${target_name}
      #  PROPERTIES COMPILE_DEFINITIONS HOGPP_TARGET=${_suffix}
      #)
      target_compile_definitions (${target_name} PRIVATE
        HOGPP_TARGET=${_suffix}
      )

      if (arch STREQUAL NEON)
        target_compile_definitions (${target_name} PRIVATE EIGEN_VECTORIZE_NEON)

        if (HAVE_MFPU_NEON)
          target_compile_options (${target_name} PRIVATE -mfpu=neon)
        endif (HAVE_MFPU_NEON)
      elseif (arch MATCHES "^SVE([0-9]+)$")
        target_compile_definitions (${target_name} PRIVATE EIGEN_VECTORIZE_SVE)
        target_compile_options (${target_name} PRIVATE -msve-vector-bits=${CMAKE_MATCH_1})
      else (arch STREQUAL NEON)
        if (NOT MSVC)
          target_compile_options (${target_name} PRIVATE -m${_lower_arch})
        else (NOT MSVC)
          target_compile_options (${target_name} PRIVATE /arch:${arch})
        endif (NOT MSVC)
      endif (arch STREQUAL NEON)

      if (arch STREQUAL AVX512F OR arch MATCHES "^AVX10")
        # Required for AVX512F (and above) by Eigen 5.x
        # compiling with -mavx512f alone without SSE/AVX FMA is not supported (bug 1638)
        # https://gitlab.com/libeigen/eigen/-/issues/1638
        target_compile_options (${target_name} PRIVATE -mfma)
      endif (arch STREQUAL AVX512F OR arch MATCHES "^AVX10")

      # Rename the Eigen namespace to avoid One Definition Rule (ODR) violations
      target_compile_definitions (${target_name} PRIVATE
        EIGEN_VECTORIZE_${_eigen_suffix}
        Eigen=Eigen_${_suffix}
        hogpp=hogpp_${_suffix}
      )
    endif (arch STREQUAL generic)

    if (HAVE_MBIG_OBJ)
      target_compile_options (${target_name} PRIVATE -Wa,-mbig-obj)
    elseif (HAVE_BIGOBJ)
      target_compile_options (${target_name} PRIVATE /bigobj)
    endif (HAVE_MBIG_OBJ)

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

  python_add_library (type_caster_test WITH_SOABI
    python/type_caster_test.cpp
  )

  target_link_libraries (type_caster_test PRIVATE
    python_type_caster
    Eigen3::Eigen
  )

  if (OpenCV_FOUND)
    target_compile_definitions (type_caster_test PRIVATE HAVE_OPENCV)
  endif (OpenCV_FOUND)

  if (SKBUILD)
    set_target_properties (pyhogpp PROPERTIES OUTPUT_NAME _hogpp)
    set_property (SOURCE python/dispatchinit.cpp APPEND PROPERTY
      COMPILE_DEFINITIONS HOGPP_SKBUILD)
  else (SKBUILD)
    set_target_properties (pyhogpp PROPERTIES OUTPUT_NAME hogpp)
  endif (SKBUILD)
endif (pybind11_FOUND AND Python_Development.Module_FOUND)

if (BUILD_TESTING)
  if (Boost_unit_test_framework_FOUND)
    add_executable (test_integral_histogram
      tests/cpp/test_integral_histogram.cpp
    )
    target_link_libraries (test_integral_histogram PRIVATE
      Boost::unit_test_framework
      Eigen3::Eigen
      hogpp::hogpp
    )

    add_test (NAME integral_histogram_full COMMAND test_integral_histogram -t full)
    add_test (NAME integral_histogram_sub COMMAND test_integral_histogram -t sub)

    add_executable (test_binning tests/cpp/test_binning.cpp)
    target_link_libraries (test_binning PRIVATE hogpp::hogpp
      Boost::unit_test_framework)

    add_test (NAME binning_signed_gradient_float COMMAND test_binning -t "signed_gradient<float>")
    add_test (NAME binning_signed_gradient_double COMMAND test_binning -t "signed_gradient<double>")
    add_test (NAME binning_signed_gradient_long_double COMMAND test_binning -t "signed_gradient<long double>")

    add_test (NAME binning_unsigned_gradient_float COMMAND test_binning -t "unsigned_gradient<float>")
    add_test (NAME binning_unsigned_gradient_double COMMAND test_binning -t "unsigned_gradient<double>")
    add_test (NAME binning_unsigned_gradient_long_double COMMAND test_binning -t "unsigned_gradient<long double>")

    add_executable (test_descriptor tests/cpp/test_descriptor.cpp)
    target_link_libraries (test_descriptor PRIVATE hogpp::hogpp
      Boost::unit_test_framework)

    add_test (NAME descriptor_empty_float COMMAND test_descriptor -t "empty<float>")
    add_test (NAME descriptor_empty_double COMMAND test_descriptor -t "empty<double>")
    add_test (NAME descriptor_empty_long_double COMMAND test_descriptor -t "empty<long double>")

    add_test (NAME descriptor_void_gradient_float COMMAND test_descriptor -t "void_gradient<float>")
    add_test (NAME descriptor_void_gradient_double COMMAND test_descriptor -t "void_gradient<double>")
    add_test (NAME descriptor_void_gradient_long_double COMMAND test_descriptor -t "void_gradient<long double>")

    add_executable (test_block_norm tests/cpp/test_block_norm.cpp)
    target_link_libraries (test_block_norm PRIVATE hogpp::hogpp
      Boost::unit_test_framework)

    add_test (NAME block_norm_zero_l1hys_float COMMAND test_block_norm -t "zero<hogpp__L1Hys<float*")
    add_test (NAME block_norm_zero_l1hys_double COMMAND test_block_norm -t "zero<hogpp__L1Hys<double*")
    add_test (NAME block_norm_zero_l1hys_long_double COMMAND test_block_norm -t "zero<hogpp__L1Hys<long double*")
    add_test (NAME block_norm_zero_l1norm_float COMMAND test_block_norm -t "zero<hogpp__L1Norm<float*")
    add_test (NAME block_norm_zero_l1norm_double COMMAND test_block_norm -t "zero<hogpp__L1Norm<double*")
    add_test (NAME block_norm_zero_l1norm_long_double COMMAND test_block_norm -t "zero<hogpp__L1Norm<long double*")
    add_test (NAME block_norm_zero_l1sqrt_float COMMAND test_block_norm -t "zero<hogpp__L1Sqrt<float*")
    add_test (NAME block_norm_zero_l1sqrt_double COMMAND test_block_norm -t "zero<hogpp__L1Sqrt<double*")
    add_test (NAME block_norm_zero_l1sqrt_long_double COMMAND test_block_norm -t "zero<hogpp__L1Sqrt<long double*")
    add_test (NAME block_norm_zero_l2hys_float COMMAND test_block_norm -t "zero<hogpp__L2Hys<float*")
    add_test (NAME block_norm_zero_l2hys_double COMMAND test_block_norm -t "zero<hogpp__L2Hys<double*")
    add_test (NAME block_norm_zero_l2hys_long_double COMMAND test_block_norm -t "zero<hogpp__L2Hys<long double*")
    add_test (NAME block_norm_zero_l2norm_float COMMAND test_block_norm -t "zero<hogpp__L2Norm<float*")
    add_test (NAME block_norm_zero_l2norm_double COMMAND test_block_norm -t "zero<hogpp__L2Norm<double*")
    add_test (NAME block_norm_zero_l2norm_long_double COMMAND test_block_norm -t "zero<hogpp__L2Norm<long double*")

    add_test (NAME block_norm_negative_near_zero_l1hys_float COMMAND test_block_norm -t "negative_near_zero<hogpp__L1Hys<float*")
    add_test (NAME block_norm_negative_near_zero_l1hys_double COMMAND test_block_norm -t "negative_near_zero<hogpp__L1Hys<double*")
    add_test (NAME block_norm_negative_near_zero_l1hys_long_double COMMAND test_block_norm -t "negative_near_zero<hogpp__L1Hys<long double*")
    add_test (NAME block_norm_negative_near_zero_l1norm_float COMMAND test_block_norm -t "negative_near_zero<hogpp__L1Norm<float*")
    add_test (NAME block_norm_negative_near_zero_l1norm_double COMMAND test_block_norm -t "negative_near_zero<hogpp__L1Norm<double*")
    add_test (NAME block_norm_negative_near_zero_l1norm_long_double COMMAND test_block_norm -t "negative_near_zero<hogpp__L1Norm<long double*")
    add_test (NAME block_norm_negative_near_zero_l1sqrt_float COMMAND test_block_norm -t "negative_near_zero<hogpp__L1Sqrt<float*")
    add_test (NAME block_norm_negative_near_zero_l1sqrt_double COMMAND test_block_norm -t "negative_near_zero<hogpp__L1Sqrt<double*")
    add_test (NAME block_norm_negative_near_zero_l1sqrt_long_double COMMAND test_block_norm -t "negative_near_zero<hogpp__L1Sqrt<long double*")
    add_test (NAME block_norm_negative_near_zero_l2hys_float COMMAND test_block_norm -t "negative_near_zero<hogpp__L2Hys<float*")
    add_test (NAME block_norm_negative_near_zero_l2hys_double COMMAND test_block_norm -t "negative_near_zero<hogpp__L2Hys<double*")
    add_test (NAME block_norm_negative_near_zero_l2hys_long_double COMMAND test_block_norm -t "negative_near_zero<hogpp__L2Hys<long double*")
    add_test (NAME block_norm_negative_near_zero_l2norm_float COMMAND test_block_norm -t "negative_near_zero<hogpp__L2Norm<float*")
    add_test (NAME block_norm_negative_near_zero_l2norm_double COMMAND test_block_norm -t "negative_near_zero<hogpp__L2Norm<double*")
    add_test (NAME block_norm_negative_near_zero_l2norm_long_double COMMAND test_block_norm -t "negative_near_zero<hogpp__L2Norm<long double*")

    add_test (NAME block_norm_positive_near_zero_l1hys_float COMMAND test_block_norm -t "positive_near_zero<hogpp__L1Hys<float*")
    add_test (NAME block_norm_positive_near_zero_l1hys_double COMMAND test_block_norm -t "positive_near_zero<hogpp__L1Hys<double*")
    add_test (NAME block_norm_positive_near_zero_l1hys_long_double COMMAND test_block_norm -t "positive_near_zero<hogpp__L1Hys<long double*")
    add_test (NAME block_norm_positive_near_zero_l1norm_float COMMAND test_block_norm -t "positive_near_zero<hogpp__L1Norm<float*")
    add_test (NAME block_norm_positive_near_zero_l1norm_double COMMAND test_block_norm -t "positive_near_zero<hogpp__L1Norm<double*")
    add_test (NAME block_norm_positive_near_zero_l1norm_long_double COMMAND test_block_norm -t "positive_near_zero<hogpp__L1Norm<long double*")
    add_test (NAME block_norm_positive_near_zero_l1sqrt_float COMMAND test_block_norm -t "positive_near_zero<hogpp__L1Sqrt<float*")
    add_test (NAME block_norm_positive_near_zero_l1sqrt_double COMMAND test_block_norm -t "positive_near_zero<hogpp__L1Sqrt<double*")
    add_test (NAME block_norm_positive_near_zero_l1sqrt_long_double COMMAND test_block_norm -t "positive_near_zero<hogpp__L1Sqrt<long double*")
    add_test (NAME block_norm_positive_near_zero_l2hys_float COMMAND test_block_norm -t "positive_near_zero<hogpp__L2Hys<float*")
    add_test (NAME block_norm_positive_near_zero_l2hys_double COMMAND test_block_norm -t "positive_near_zero<hogpp__L2Hys<double*")
    add_test (NAME block_norm_positive_near_zero_l2hys_long_double COMMAND test_block_norm -t "positive_near_zero<hogpp__L2Hys<long double*")
    add_test (NAME block_norm_positive_near_zero_l2norm_float COMMAND test_block_norm -t "positive_near_zero<hogpp__L2Norm<float*")
    add_test (NAME block_norm_positive_near_zero_l2norm_double COMMAND test_block_norm -t "positive_near_zero<hogpp__L2Norm<double*")
    add_test (NAME block_norm_positive_near_zero_l2norm_long_double COMMAND test_block_norm -t "positive_near_zero<hogpp__L2Norm<long double*")
  endif (Boost_unit_test_framework_FOUND)

  # pytest-xdist running on PyPy with extension modules compiled using MSVC runs
  # extremely slow. Avoid using xdist.
  if (Python_INTERPRETER_ID STREQUAL "PyPy" AND MSVC)
    set (_pytest_xdist_ARGS)
  else (Python_INTERPRETER_ID STREQUAL "PyPy" AND MSVC)
    set (_pytest_xdist_ARGS -n auto)
  endif (Python_INTERPRETER_ID STREQUAL "PyPy" AND MSVC)

  if (TARGET pyhogpp)
    add_test (NAME python_hogpp COMMAND
      ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:pyhogpp>
      ${Python_EXECUTABLE} -m pytest -W error ${_pytest_xdist_ARGS}
      ${hogpp_SOURCE_DIR}/tests/python/module/test_hogpp.py
      WORKING_DIRECTORY ${hogpp_SOURCE_DIR}
    )

    add_test (NAME python_tensor_type_caster COMMAND
      ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:pyhogpp>
      ${Python_EXECUTABLE} -m pytest -W error ${_pytest_xdist_ARGS}
      ${hogpp_SOURCE_DIR}/tests/python/internal/test_tensor_type_caster.py)

    add_test (NAME python_repr COMMAND
      ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:pyhogpp>
      ${Python_EXECUTABLE} -m pytest -W error ${_pytest_xdist_ARGS}
      ${hogpp_SOURCE_DIR}/tests/python/module/test_repr.py)
  endif (TARGET pyhogpp)

  if (TARGET type_caster_test AND OpenCV_FOUND)
    add_test (NAME python_opencv_type_caster COMMAND
      ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:type_caster_test>
      ${Python_EXECUTABLE} -m pytest -W error ${_pytest_xdist_ARGS}
      ${hogpp_SOURCE_DIR}/tests/python/internal/test_opencv_type_caster.py)
  endif (TARGET type_caster_test AND OpenCV_FOUND)
endif (BUILD_TESTING)

if (TARGET pyhogpp AND Sphinx_FOUND)
  add_custom_target (sphinx
    ${CMAKE_COMMAND} -E env
      PYTHONPATH=$<TARGET_FILE_DIR:pyhogpp>
        $<TARGET_FILE:Sphinx::build> -M html
        ${hogpp_SOURCE_DIR}/docs
        ${hogpp_BINARY_DIR}/docs/_build
        -W --keep-going
    DEPENDS pyhogpp
    BYPRODUCTS docs/_build/html/.buildinfo
    COMMENT "Generating Sphinx HTML documentation"
    USES_TERMINAL
  )

  set_property (DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES docs/_build)
endif (TARGET pyhogpp AND Sphinx_FOUND)

if (TARGET pyhogpp)
  if (SKBUILD)
    install (TARGETS pyhogpp
      LIBRARY DESTINATION ${PROJECT_NAME} COMPONENT Runtime
      RUNTIME DESTINATION ${PROJECT_NAME} COMPONENT Runtime
    )
  endif (SKBUILD)
endif (TARGET pyhogpp)
