cmake_minimum_required (VERSION 3.26...3.30)
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

list (APPEND CMAKE_MODULE_PATH ${hogpp_SOURCE_DIR}/cmake)

include (CheckCXXCompilerFlag)
include (CheckCXXSourceCompiles)
include (CheckCXXSymbolExists)
include (CTest)
include (GNUInstallDirs)

check_cxx_compiler_flag (-Wa,-mbig-obj HAVE_MBIG_OBJ)
check_cxx_compiler_flag (/bigobj HAVE_BIGOBJ)
check_cxx_compiler_flag (/Zc:preprocessor HAVE_ZC_PREPROCESSOR)

# Lifts the limit on the number of sections of an object file that heavily
# templated sources exceed.
function (hogpp_enable_big_obj target)
  if (HAVE_MBIG_OBJ)
    set_property (TARGET ${target} APPEND PROPERTY COMPILE_OPTIONS -Wa,-mbig-obj)
  elseif (HAVE_BIGOBJ)
    set_property (TARGET ${target} APPEND PROPERTY COMPILE_OPTIONS /bigobj)
  endif (HAVE_MBIG_OBJ)
endfunction (hogpp_enable_big_obj)

find_package (Boost 1.70 COMPONENTS unit_test_framework NO_MODULE)
find_package (OpenCV 4.0 COMPONENTS core NO_MODULE)

# Development.SABIModule provides the Python::SABIModule target nanobind needs
# to build stable ABI (abi3) extension modules.
find_package (Python 3.10 COMPONENTS Interpreter Development.Module  Development.SABIModule)
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

if (Python_Interpreter_FOUND AND Python_Development.Module_FOUND)
  # nanobind's config computes the extension suffix itself (querying
  # sysconfig directly when SKBUILD_SOABI/Python_SOABI is unavailable or, as
  # on PyPy, missing the platform tag), so no manual SOABI handling is
  # needed here.
  find_package (nanobind 2.2.0 CONFIG)
endif (Python_Interpreter_FOUND AND Python_Development.Module_FOUND)

check_cxx_symbol_exists (std::execution::par execution HAVE_EXECUTION)

# Replacing the allocation functions of glibc allows tests to observe the heap.
check_cxx_source_compiles ("
#include <cstddef>
extern \"C\" void* __libc_malloc(std::size_t size);
int main() { return __libc_malloc(1) == nullptr; }
" HAVE_LIBC_MALLOC)

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

if (nanobind_FOUND)
  # Shared between the pyhogpp extension module and the type_caster_test
  # module used by the internal type caster tests. Each nanobind_add_module()
  # call below compiles its own private copy: nanobind extension modules are
  # self-contained shared objects, so there is no ODR concern, and it avoids
  # having to keep a helper static library in lockstep with whichever
  # STABLE_ABI/FREE_THREADED nanobind core library flavor each module ends up
  # linking against.
  set (HOGPP_PYTHON_TYPE_CASTER_SOURCES
    python/type_caster/bounds.cpp
    python/type_caster/bounds.hpp
    python/type_caster/maskingref.cpp
    python/type_caster/maskingref.hpp
    python/type_caster/stride.hpp
    python/type_caster/tensor.hpp
    python/type_caster/typesequence.hpp
  )

  if (OpenCV_FOUND)
    list (APPEND HOGPP_PYTHON_TYPE_CASTER_SOURCES
      python/type_caster/opencv.cpp
      python/type_caster/opencv.hpp
    )
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
  )

  hogpp_enable_big_obj (pyhogpp)

  if (OpenCV_FOUND)
    target_compile_definitions (pyhogpp PRIVATE HAVE_OPENCV)
    target_link_libraries (pyhogpp PRIVATE opencv_core)
  endif (OpenCV_FOUND)

  if (HAVE_EXECUTION)
    target_compile_definitions (pyhogpp PRIVATE HAVE_EXECUTION)
  endif (HAVE_EXECUTION)

  target_link_libraries (pyhogpp PRIVATE
    hogpp::hogpp
    Eigen3::Eigen
  )

  nanobind_add_module (type_caster_test STABLE_ABI FREE_THREADED NB_SUPPRESS_WARNINGS
    ${HOGPP_PYTHON_TYPE_CASTER_SOURCES}
    python/type_caster_test.cpp
  )

  if (HAVE_MBIG_OBJ)
    set_property (TARGET type_caster_test APPEND PROPERTY COMPILE_OPTIONS -Wa,-mbig-obj)
  endif (HAVE_MBIG_OBJ)

  target_link_libraries (type_caster_test PRIVATE
    hogpp::hogpp
    Eigen3::Eigen
  )

  if (OpenCV_FOUND)
    target_compile_definitions (type_caster_test PRIVATE HAVE_OPENCV)
    target_link_libraries (type_caster_test PRIVATE opencv_core)
  endif (OpenCV_FOUND)

  if (MINGW)
    # Statically linking the runtime, the same workaround nanobind itself
    # applies for musllinux wheels via nanobind_musl_static_libcpp(), makes
    # the module self-contained and avoids depending on the matching
    # libstdc++-6.dll/libgcc_s_seh-1.dll being present at import time.
    target_link_options (pyhogpp PRIVATE -static-libstdc++ -static-libgcc)
    target_link_options (type_caster_test PRIVATE -static-libstdc++
      -static-libgcc)
  endif (MINGW)

  # NB_SUPPRESS_WARNINGS above only silences warnings that originate in
  # headers (e.g. Python's or nanobind's own), since those are reached via
  # a SYSTEM include directory. It cannot silence warnings from code
  # written directly in nanobind's own .cpp sources, e.g. a sign
  # conversion in nb_func.cpp. That source isn't ours to fix, so warnings
  # are disabled outright for nanobind's compiled library specifically,
  # without weakening warnings for hogpp's own targets. This also covers
  # warnings a newer compiler than what nanobind was written against might
  # introduce, since none are emitted for that code in the first place.
  foreach (_hogpp_nanobind_lib nanobind-static nanobind-static-abi3 nanobind-static-ft)
    if (TARGET ${_hogpp_nanobind_lib})
      if (MSVC)
        target_compile_options (${_hogpp_nanobind_lib} PRIVATE /w)
      else (MSVC)
        target_compile_options (${_hogpp_nanobind_lib} PRIVATE -w)
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

  if (SKBUILD)
    set_target_properties (pyhogpp PROPERTIES OUTPUT_NAME _hogpp)
    set_property (SOURCE python/hogpp.cpp APPEND PROPERTY COMPILE_DEFINITIONS HOGPP_SKBUILD)
  else (SKBUILD)
    set_target_properties (pyhogpp PROPERTIES OUTPUT_NAME hogpp)
  endif (SKBUILD)
endif (nanobind_FOUND)

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
    add_test (NAME integral_histogram_volumetric COMMAND test_integral_histogram -t volumetric)
    add_test (NAME integral_histogram_cascade_matches_closed_form_for_constant_field COMMAND test_integral_histogram -t cascade_matches_closed_form_for_constant_field)
    add_test (NAME integral_histogram_interpolate_matches_brute_force_bilinear_splat COMMAND test_integral_histogram -t interpolate_matches_brute_force_bilinear_splat)
    add_test (NAME integral_histogram_interpolate_row_major_matches_column_major COMMAND test_integral_histogram -t interpolate_row_major_matches_column_major)
    add_test (NAME integral_histogram_interpolate_unsigned_coordinates_near_lower_boundary COMMAND test_integral_histogram -t interpolate_unsigned_coordinates_near_lower_boundary)

    add_executable (test_cartesian_product tests/cpp/test_cartesian_product.cpp)
    target_link_libraries (test_cartesian_product PRIVATE hogpp::hogpp
      Boost::unit_test_framework)

    if (MSVC)
      set (_long_long_name __int64)
    else (MSVC)
      set (_long_long_name "long long")
    endif (MSVC)

    add_test (NAME cartesian_product_loop_counter_type_schar COMMAND test_cartesian_product -t "loop_counter_type<signed char>")
    add_test (NAME cartesian_product_loop_counter_type_uchar COMMAND test_cartesian_product -t "loop_counter_type<unsigned char>")
    add_test (NAME cartesian_product_loop_counter_type_short COMMAND test_cartesian_product -t "loop_counter_type<short>")
    add_test (NAME cartesian_product_loop_counter_type_ushort COMMAND test_cartesian_product -t "loop_counter_type<unsigned short>")
    add_test (NAME cartesian_product_loop_counter_type_int COMMAND test_cartesian_product -t "loop_counter_type<int>")
    add_test (NAME cartesian_product_loop_counter_type_uint COMMAND test_cartesian_product -t "loop_counter_type<unsigned int>")
    add_test (NAME cartesian_product_loop_counter_type_long COMMAND test_cartesian_product -t "loop_counter_type<long>")
    add_test (NAME cartesian_product_loop_counter_type_ulong COMMAND test_cartesian_product -t "loop_counter_type<unsigned long>")
    add_test (NAME cartesian_product_loop_counter_type_longlong COMMAND test_cartesian_product -t "loop_counter_type<${_long_long_name}>")
    add_test (NAME cartesian_product_loop_counter_type_ulonglong COMMAND test_cartesian_product -t "loop_counter_type<unsigned ${_long_long_name}>")

    unset (_long_long_name)

    add_executable (test_gradient tests/cpp/test_gradient.cpp)
    target_link_libraries (test_gradient PRIVATE hogpp::hogpp
      Boost::unit_test_framework)

    add_test (NAME gradient_interior_border_magnitude_comparable_float COMMAND test_gradient -t "interior_border_magnitude_comparable<float>")
    add_test (NAME gradient_interior_border_magnitude_comparable_double COMMAND test_gradient -t "interior_border_magnitude_comparable<double>")
    add_test (NAME gradient_interior_border_magnitude_comparable_long_double COMMAND test_gradient -t "interior_border_magnitude_comparable<long double>")

    add_test (NAME gradient_single_row_float COMMAND test_gradient -t "single_row<float>")
    add_test (NAME gradient_single_row_double COMMAND test_gradient -t "single_row<double>")
    add_test (NAME gradient_single_row_long_double COMMAND test_gradient -t "single_row<long double>")

    add_test (NAME gradient_single_column_float COMMAND test_gradient -t "single_column<float>")
    add_test (NAME gradient_single_column_double COMMAND test_gradient -t "single_column<double>")
    add_test (NAME gradient_single_column_long_double COMMAND test_gradient -t "single_column<long double>")

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
    hogpp_enable_big_obj (test_descriptor)

    add_test (NAME descriptor_empty_float COMMAND test_descriptor -t "empty<float>")
    add_test (NAME descriptor_empty_double COMMAND test_descriptor -t "empty<double>")
    add_test (NAME descriptor_empty_long_double COMMAND test_descriptor -t "empty<long double>")

    add_test (NAME descriptor_void_gradient_float COMMAND test_descriptor -t "void_gradient<float>")
    add_test (NAME descriptor_void_gradient_double COMMAND test_descriptor -t "void_gradient<double>")
    add_test (NAME descriptor_void_gradient_long_double COMMAND test_descriptor -t "void_gradient<long double>")

    add_test (NAME descriptor_features_block_larger_than_region_float COMMAND test_descriptor -t "features_block_larger_than_region<float>")
    add_test (NAME descriptor_features_block_larger_than_region_double COMMAND test_descriptor -t "features_block_larger_than_region<double>")
    add_test (NAME descriptor_features_block_larger_than_region_long_double COMMAND test_descriptor -t "features_block_larger_than_region<long double>")

    add_test (NAME descriptor_features_cover_region_border_float COMMAND test_descriptor -t "features_cover_region_border<float>")
    add_test (NAME descriptor_features_cover_region_border_double COMMAND test_descriptor -t "features_cover_region_border<double>")
    add_test (NAME descriptor_features_cover_region_border_long_double COMMAND test_descriptor -t "features_cover_region_border<long double>")

    add_test (NAME descriptor_features_largest_block_stride_float COMMAND test_descriptor -t "features_largest_block_stride<float>")
    add_test (NAME descriptor_features_largest_block_stride_double COMMAND test_descriptor -t "features_largest_block_stride<double>")
    add_test (NAME descriptor_features_largest_block_stride_long_double COMMAND test_descriptor -t "features_largest_block_stride<long double>")

    add_test (NAME descriptor_features_clip_block_to_small_region_float COMMAND test_descriptor -t "features_clip_block_to_small_region<float>")
    add_test (NAME descriptor_features_clip_block_to_small_region_double COMMAND test_descriptor -t "features_clip_block_to_small_region<double>")
    add_test (NAME descriptor_features_clip_block_to_small_region_long_double COMMAND test_descriptor -t "features_clip_block_to_small_region<long double>")

    add_test (NAME descriptor_nan_pixel_float COMMAND test_descriptor -t "nan_pixel<float>")
    add_test (NAME descriptor_nan_pixel_double COMMAND test_descriptor -t "nan_pixel<double>")
    add_test (NAME descriptor_nan_pixel_long_double COMMAND test_descriptor -t "nan_pixel<long double>")

    add_test (NAME descriptor_default_descriptor_has_no_spatial_interpolation_float COMMAND test_descriptor -t "default_descriptor_has_no_spatial_interpolation<float>")
    add_test (NAME descriptor_default_descriptor_has_no_spatial_interpolation_double COMMAND test_descriptor -t "default_descriptor_has_no_spatial_interpolation<double>")
    add_test (NAME descriptor_default_descriptor_has_no_spatial_interpolation_long_double COMMAND test_descriptor -t "default_descriptor_has_no_spatial_interpolation<long double>")

    add_test (NAME descriptor_trilinear_interpolation_even_cell_size_float COMMAND test_descriptor -t "trilinear_interpolation_even_cell_size<float>")
    add_test (NAME descriptor_trilinear_interpolation_even_cell_size_double COMMAND test_descriptor -t "trilinear_interpolation_even_cell_size<double>")
    add_test (NAME descriptor_trilinear_interpolation_even_cell_size_long_double COMMAND test_descriptor -t "trilinear_interpolation_even_cell_size<long double>")
    add_test (NAME descriptor_trilinear_interpolation_odd_cell_size_float COMMAND test_descriptor -t "trilinear_interpolation_odd_cell_size<float>")
    add_test (NAME descriptor_trilinear_interpolation_odd_cell_size_double COMMAND test_descriptor -t "trilinear_interpolation_odd_cell_size<double>")
    add_test (NAME descriptor_trilinear_interpolation_odd_cell_size_long_double COMMAND test_descriptor -t "trilinear_interpolation_odd_cell_size<long double>")
    add_test (NAME descriptor_trilinear_interpolation_large_float_image COMMAND test_descriptor -t trilinear_interpolation_large_float_image)

    add_test (NAME descriptor_trilinear_interpolation_with_gaussian_weighting_float COMMAND test_descriptor -t "trilinear_interpolation_with_gaussian_weighting<float>")
    add_test (NAME descriptor_trilinear_interpolation_with_gaussian_weighting_double COMMAND test_descriptor -t "trilinear_interpolation_with_gaussian_weighting<double>")
    add_test (NAME descriptor_trilinear_interpolation_with_gaussian_weighting_long_double COMMAND test_descriptor -t "trilinear_interpolation_with_gaussian_weighting<long double>")
    add_test (NAME descriptor_trilinear_interpolation_with_gaussian_weighting_large_float_image COMMAND test_descriptor -t trilinear_interpolation_with_gaussian_weighting_large_float_image)

    if (HAVE_LIBC_MALLOC)
      add_executable (test_allocations tests/cpp/test_allocations.cpp)
      target_link_libraries (test_allocations PRIVATE hogpp::hogpp
        Boost::unit_test_framework)

      foreach (_type IN ITEMS float double)
        add_test (NAME allocations_trilinear_features_allocate_no_more_per_cell_${_type} COMMAND test_allocations -t "trilinear_features_allocate_no_more_per_cell<${_type}>")
        add_test (NAME allocations_weighted_trilinear_features_allocate_no_more_per_cell_${_type} COMMAND test_allocations -t "weighted_trilinear_features_allocate_no_more_per_cell<${_type}>")
        add_test (NAME allocations_trilinear_recompute_keeps_one_cascade_${_type} COMMAND test_allocations -t "trilinear_recompute_keeps_one_cascade<${_type}>")
      endforeach (_type IN ITEMS float double)
    endif (HAVE_LIBC_MALLOC)

    add_executable (test_multiply_add tests/cpp/test_multiply_add.cpp)
    target_link_libraries (test_multiply_add PRIVATE hogpp::hogpp
      Boost::unit_test_framework)

    add_test (NAME multiply_add_fused_only_if_fast_float COMMAND test_multiply_add -t "fused_only_if_fast<float>")
    add_test (NAME multiply_add_fused_only_if_fast_double COMMAND test_multiply_add -t "fused_only_if_fast<double>")
    add_test (NAME multiply_add_fused_only_if_fast_long_double COMMAND test_multiply_add -t "fused_only_if_fast<long double>")

    add_executable (test_spatialweighting tests/cpp/test_spatialweighting.cpp)
    target_link_libraries (test_spatialweighting PRIVATE hogpp::hogpp
      Boost::unit_test_framework)
    hogpp_enable_big_obj (test_spatialweighting)

    add_test (NAME spatialweighting_gaussian_matches_reference_float COMMAND test_spatialweighting -t "gaussian_matches_reference<float>")
    add_test (NAME spatialweighting_gaussian_matches_reference_double COMMAND test_spatialweighting -t "gaussian_matches_reference<double>")
    add_test (NAME spatialweighting_gaussian_matches_reference_long_double COMMAND test_spatialweighting -t "gaussian_matches_reference<long double>")

    add_test (NAME spatialweighting_recompute_replaces_previous_image_float COMMAND test_spatialweighting -t "recompute_replaces_previous_image<float>")
    add_test (NAME spatialweighting_recompute_replaces_previous_image_double COMMAND test_spatialweighting -t "recompute_replaces_previous_image<double>")
    add_test (NAME spatialweighting_recompute_replaces_previous_image_long_double COMMAND test_spatialweighting -t "recompute_replaces_previous_image<long double>")

    add_test (NAME spatialweighting_set_histogram_replaces_previous_image_float COMMAND test_spatialweighting -t "set_histogram_replaces_previous_image<float>")
    add_test (NAME spatialweighting_set_histogram_replaces_previous_image_double COMMAND test_spatialweighting -t "set_histogram_replaces_previous_image<double>")
    add_test (NAME spatialweighting_set_histogram_replaces_previous_image_long_double COMMAND test_spatialweighting -t "set_histogram_replaces_previous_image<long double>")

    add_test (NAME spatialweighting_set_num_bins_after_compute_float COMMAND test_spatialweighting -t "set_num_bins_after_compute<float>")
    add_test (NAME spatialweighting_set_num_bins_after_compute_double COMMAND test_spatialweighting -t "set_num_bins_after_compute<double>")
    add_test (NAME spatialweighting_set_num_bins_after_compute_long_double COMMAND test_spatialweighting -t "set_num_bins_after_compute<long double>")

    add_test (NAME spatialweighting_geometry_change_after_compute_float COMMAND test_spatialweighting -t "geometry_change_after_compute<float>")
    add_test (NAME spatialweighting_geometry_change_after_compute_double COMMAND test_spatialweighting -t "geometry_change_after_compute<double>")
    add_test (NAME spatialweighting_geometry_change_after_compute_long_double COMMAND test_spatialweighting -t "geometry_change_after_compute<long double>")

    add_test (NAME spatialweighting_sigma_change_after_compute_float COMMAND test_spatialweighting -t "sigma_change_after_compute<float>")
    add_test (NAME spatialweighting_sigma_change_after_compute_double COMMAND test_spatialweighting -t "sigma_change_after_compute<double>")
    add_test (NAME spatialweighting_sigma_change_after_compute_long_double COMMAND test_spatialweighting -t "sigma_change_after_compute<long double>")

    add_test (NAME spatialweighting_tiny_sigma_keeps_closest_pixels_float COMMAND test_spatialweighting -t "tiny_sigma_keeps_closest_pixels<float>")
    add_test (NAME spatialweighting_tiny_sigma_keeps_closest_pixels_double COMMAND test_spatialweighting -t "tiny_sigma_keeps_closest_pixels<double>")
    add_test (NAME spatialweighting_tiny_sigma_keeps_closest_pixels_long_double COMMAND test_spatialweighting -t "tiny_sigma_keeps_closest_pixels<long double>")

    add_test (NAME spatialweighting_invalid_sigma_rejected_float COMMAND test_spatialweighting -t "invalid_sigma_rejected<float>")
    add_test (NAME spatialweighting_invalid_sigma_rejected_double COMMAND test_spatialweighting -t "invalid_sigma_rejected<double>")
    add_test (NAME spatialweighting_invalid_sigma_rejected_long_double COMMAND test_spatialweighting -t "invalid_sigma_rejected<long double>")

    add_test (NAME spatialweighting_largest_sigma_matches_reference_float COMMAND test_spatialweighting -t "largest_sigma_matches_reference<float>")
    add_test (NAME spatialweighting_largest_sigma_matches_reference_double COMMAND test_spatialweighting -t "largest_sigma_matches_reference<double>")
    add_test (NAME spatialweighting_largest_sigma_matches_reference_long_double COMMAND test_spatialweighting -t "largest_sigma_matches_reference<long double>")

    add_test (NAME spatialweighting_disabled_matches_default_float COMMAND test_spatialweighting -t "disabled_matches_default<float>")
    add_test (NAME spatialweighting_disabled_matches_default_double COMMAND test_spatialweighting -t "disabled_matches_default<double>")
    add_test (NAME spatialweighting_disabled_matches_default_long_double COMMAND test_spatialweighting -t "disabled_matches_default<long double>")

    add_test (NAME spatialweighting_disabled_weighting_adds_no_storage_float COMMAND test_spatialweighting -t "disabled_weighting_adds_no_storage<float>")
    add_test (NAME spatialweighting_disabled_weighting_adds_no_storage_double COMMAND test_spatialweighting -t "disabled_weighting_adds_no_storage<double>")
    add_test (NAME spatialweighting_disabled_weighting_adds_no_storage_long_double COMMAND test_spatialweighting -t "disabled_weighting_adds_no_storage<long double>")

    add_test (NAME spatialweighting_masked_pixels_excluded_float COMMAND test_spatialweighting -t "masked_pixels_excluded<float>")
    add_test (NAME spatialweighting_masked_pixels_excluded_double COMMAND test_spatialweighting -t "masked_pixels_excluded<double>")
    add_test (NAME spatialweighting_masked_pixels_excluded_long_double COMMAND test_spatialweighting -t "masked_pixels_excluded<long double>")

    add_test (NAME spatialweighting_fast_path_selection_float COMMAND test_spatialweighting -t "fast_path_selection<float>")
    add_test (NAME spatialweighting_fast_path_selection_double COMMAND test_spatialweighting -t "fast_path_selection<double>")
    add_test (NAME spatialweighting_fast_path_selection_long_double COMMAND test_spatialweighting -t "fast_path_selection<long double>")

    add_test (NAME spatialweighting_failed_prepare_discards_weighting_float COMMAND test_spatialweighting -t "failed_prepare_discards_weighting<float>")
    add_test (NAME spatialweighting_failed_prepare_discards_weighting_double COMMAND test_spatialweighting -t "failed_prepare_discards_weighting<double>")
    add_test (NAME spatialweighting_failed_prepare_discards_weighting_long_double COMMAND test_spatialweighting -t "failed_prepare_discards_weighting<long double>")

    add_test (NAME spatialweighting_restored_weighting_matches_exactly_float COMMAND test_spatialweighting -t "restored_weighting_matches_exactly<float>")
    add_test (NAME spatialweighting_restored_weighting_matches_exactly_double COMMAND test_spatialweighting -t "restored_weighting_matches_exactly<double>")
    add_test (NAME spatialweighting_restored_weighting_matches_exactly_long_double COMMAND test_spatialweighting -t "restored_weighting_matches_exactly<long double>")

    add_test (NAME spatialweighting_mismatched_weighting_rejected_float COMMAND test_spatialweighting -t "mismatched_weighting_rejected<float>")
    add_test (NAME spatialweighting_mismatched_weighting_rejected_double COMMAND test_spatialweighting -t "mismatched_weighting_rejected<double>")
    add_test (NAME spatialweighting_mismatched_weighting_rejected_long_double COMMAND test_spatialweighting -t "mismatched_weighting_rejected<long double>")

    add_test (NAME spatialweighting_descriptor_prepares_weighting_float COMMAND test_spatialweighting -t "descriptor_prepares_weighting<float>")
    add_test (NAME spatialweighting_descriptor_prepares_weighting_double COMMAND test_spatialweighting -t "descriptor_prepares_weighting<double>")
    add_test (NAME spatialweighting_descriptor_prepares_weighting_long_double COMMAND test_spatialweighting -t "descriptor_prepares_weighting<long double>")

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

    add_executable (test_maskingref tests/cpp/test_maskingref.cpp)
    target_compile_features (test_maskingref PRIVATE cxx_std_20)
    target_link_libraries (test_maskingref PRIVATE
      Boost::unit_test_framework
      hogpp::hogpp
    )

    add_test (NAME maskingref_copy_refers_to_callable COMMAND test_maskingref -t copy_refers_to_callable)
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
