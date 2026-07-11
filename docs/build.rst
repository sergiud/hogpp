Compiling from Source
=====================

Requirements
------------

-  C++20 compiler
-  `CMake <https://gitlab.kitware.com/cmake/cmake>`__ 3.28
-  `Eigen <https://gitlab.com/libeigen/eigen>`__ 3.3.7
-  `fmt <https://github.com/fmtlib/fmt>`__ 6.0

Optional dependencies:

-  `OpenCV <https://github.com/opencv/opencv>`__ 4.0
-  `Boost <https://www.boost.org>`__ 1.70
-  `Python <https://www.python.org>`__ 3.11
-  `nanobind <https://github.com/wjakob/nanobind>`__ 2.2.0

More recent versions of the above are expected to work as well.

Options
-------

``WITH_PYTHON``
   Builds the Python module if Python and nanobind are found. Defaults to
   ``ON`` unless hogpp is added to another project as a subdirectory.

``BUILD_DEPS``
   Downloads and builds Eigen and fmt instead of using the installed
   packages. Defaults to ``OFF``.

``WITH_INSTALL``
   Installs the headers along with the CMake package configuration.
   Defaults to ``ON`` unless hogpp is added to another project as a
   subdirectory, and is unavailable if ``BUILD_DEPS`` is
   enabled.

Using hogpp in CMake Projects
-----------------------------

The installed package provides the header-only library as the target
``hogpp::hogpp``:

.. code-block:: cmake

   find_package (hogpp 0.3 REQUIRED)
   target_link_libraries (app PRIVATE hogpp::hogpp)

Projects that add the source tree with ``add_subdirectory`` link to the
same target.
