cmake_minimum_required (VERSION 3.5)
project (hogpp
  VERSION 0.1.0
  LANGUAGES CXX
  DESCRIPTION "Fast histogram of oriented gradients computation using integral histogram"
)

find_package (pybind11 2.2.4 NO_MODULE)
find_package (Eigen3 3.3.7 REQUIRED NO_MODULE)
find_package (OpenCV 4.0 REQUIRED)

add_library (hogpp INTERFACE)
add_library (hogpp::hogpp ALIAS hogpp)

target_include_directories (hogpp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLURDIR}>
)

target_compile_features (hogpp INTERFACE cxx_std_17)
target_link_libraries (hogpp INTERFACE Eigen3::Eigen opencv_core opencv_imgproc)

if (pybind11_FOUND)
  pybind11_add_module (pyhogpp
    python/hogpp.cpp
  )

  target_link_libraries (pyhogpp PRIVATE
    hogpp::hogpp
  )

  set_target_properties (pyhogpp PROPERTIES OUTPUT_NAME hogpp)
endif (pybind11_FOUND)
