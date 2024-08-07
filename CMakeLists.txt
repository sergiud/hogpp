cmake_minimum_required (VERSION 3.15)
project (hogpp
  VERSION 0.1.0
  LANGUAGES CXX
  DESCRIPTION "Fast histogram of oriented gradients computation using integral histogram"
  HOMEPAGE_URL https://github.com/sergiud/hogpp
)

set (CMAKE_CXX_VISIBILITY_PRESET hidden)
set (CMAKE_VISIBILITY_INLINES_HIDDEN ON)

include (CheckCXXCompilerFlag)
include (CheckCXXSymbolExists)
include (CTest)

check_cxx_compiler_flag (-Wa,-mbig-obj HAVE_MBIG_OBJ)

find_package (Boost 1.70 COMPONENTS unit_test_framework NO_MODULE)
find_package (Eigen3 3.4.0 REQUIRED NO_MODULE)
find_package (fmt 6.0 REQUIRED)
find_package (OpenCV 4.0 REQUIRED)
find_package (Python 3.0 COMPONENTS Development Interpreter)

if (NOT Boost_FOUND)
  set (CMAKE_DISABLE_FIND_PACKAGE_pybind11 ON)
endif (NOT Boost_FOUND)

if (MSVC AND MSVC_VERSION GREATER_EQUAL 1930)
  find_package (pybind11 2.9.0 NO_MODULE)
else (MSVC AND MSVC_VERSION GREATER_EQUAL 1930)
  find_package (pybind11 2.6.2 NO_MODULE)
endif (MSVC AND MSVC_VERSION GREATER_EQUAL 1930)

check_cxx_symbol_exists (std::execution::par execution HAVE_EXECUTION)

add_library (hogpp INTERFACE)
add_library (hogpp::hogpp ALIAS hogpp)

target_include_directories (hogpp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLURDIR}>
)

target_compile_features (hogpp INTERFACE cxx_std_17)
target_link_libraries (hogpp INTERFACE
  Eigen3::Eigen
  fmt::fmt
  opencv_core
)

if (pybind11_FOUND)
  add_library (python_type_caster STATIC
    python/type_caster/cartesianproduct.hpp
    python/type_caster/opencv.cpp
    python/type_caster/opencv.hpp
    python/type_caster/stride.hpp
    python/type_caster/tensor.hpp
    python/type_caster/typesequence.hpp
  )

  target_compile_features (python_type_caster PUBLIC cxx_std_17)
  target_link_libraries (python_type_caster PUBLIC
    Eigen3::Eigen
    opencv_core
    pybind11::pybind11
  )

  set_target_properties (python_type_caster PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

  pybind11_add_module (pyhogpp
    python/binning.cpp
    python/binning.hpp
    python/blocknormalizer.cpp
    python/blocknormalizer.hpp
    python/formatter.hpp
    python/hogpp.cpp
    python/integralhogdescriptor.cpp
    python/integralhogdescriptor.hpp
    python/magnitude.cpp
    python/magnitude.hpp
  )

  if (HAVE_MBIG_OBJ)
    set_property (TARGET pyhogpp APPEND PROPERTY COMPILE_OPTIONS -Wa,-mbig-obj)
  endif (HAVE_MBIG_OBJ)

  if (HAVE_EXECUTION)
    target_compile_definitions (pyhogpp PRIVATE HAVE_EXECUTION)
  endif (HAVE_EXECUTION)

  target_link_libraries (pyhogpp PRIVATE
    Boost::boost
    hogpp::hogpp
    python_type_caster
  )

  pybind11_add_module (type_caster_test
    python/type_caster_test.cpp
  )

  target_link_libraries (type_caster_test PRIVATE
    python_type_caster
    Eigen3::Eigen
  )

  set_target_properties (pyhogpp PROPERTIES OUTPUT_NAME hogpp)
endif (pybind11_FOUND)

if (BUILD_TESTING)
  if (Boost_unit_test_framework_FOUND)
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

  if (pybind11_FOUND AND Python_Interpreter_FOUND)
    add_test (NAME python_hogpp COMMAND
      ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:pyhogpp>
      ${Python_EXECUTABLE} -m pytest -n auto
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_hogpp.py)

    add_test (NAME python_tensor_type_caster COMMAND
      ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:pyhogpp>
      ${Python_EXECUTABLE} -m pytest -n auto
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_tensor_type_caster.py)

    add_test (NAME python_opencv_type_caster COMMAND
      ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:pyhogpp>
      ${Python_EXECUTABLE} -m pytest -n auto
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_opencv_type_caster.py)

    add_test (NAME python_repr COMMAND
      ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:pyhogpp>
      ${Python_EXECUTABLE} -m pytest -n auto
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_repr.py)
  endif (pybind11_FOUND AND Python_Interpreter_FOUND)
endif (BUILD_TESTING)
