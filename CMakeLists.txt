cmake_minimum_required (VERSION 3.5)
project (hogpp
  VERSION 0.1.0
  LANGUAGES CXX
  DESCRIPTION "Fast histogram of oriented gradients computation using integral histogram"
)

include (CTest)

find_package (Boost 1.70 COMPONENTS unit_test_framework)
find_package (Eigen3 3.3.7 REQUIRED NO_MODULE)
find_package (fmt 6.0 REQUIRED)
find_package (OpenCV 4.0 REQUIRED)
find_package (Python 3.0 COMPONENTS Development Interpreter)
find_package (pybind11 2.2.4 NO_MODULE)

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
  opencv_imgproc
)

if (pybind11_FOUND)
  add_library (python_type_caster STATIC
    python/type_caster/opencv.cpp
    python/type_caster/opencv.hpp
    python/type_caster/tensor.hpp
  )

  target_compile_features (python_type_caster PUBLIC cxx_std_17)
  target_link_libraries (python_type_caster PUBLIC
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
    python/hogpp.cpp
    python/integralhogdescriptor.cpp
    python/integralhogdescriptor.hpp
    python/magnitude.cpp
    python/magnitude.hpp
  )

  target_link_libraries (pyhogpp PRIVATE
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
  endif (Boost_unit_test_framework_FOUND)

  if (pybind11_FOUND AND Python_Interpreter_FOUND)
    add_test (NAME python_hogpp COMMAND
      ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:pyhogpp>
      ${Python_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_hogpp.py)

    add_test (NAME python_tensor_type_caster COMMAND
      ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:pyhogpp>
      ${Python_EXECUTABLE} -m pytest
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_tensor_type_caster.py)

    add_test (NAME python_opencv_type_caster COMMAND
      ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:pyhogpp>
      ${Python_EXECUTABLE} -m pytest
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_opencv_type_caster.py)
  endif (pybind11_FOUND AND Python_Interpreter_FOUND)
endif (BUILD_TESTING)
