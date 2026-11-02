Compiling from Source
=====================

Requirements
------------

-  C++20 compiler
-  `CMake <https://gitlab.kitware.com/cmake/cmake>`__ 3.18
-  `Eigen <https://gitlab.com/libeigen/eigen>`__ 3.3.7
-  `fmt <https://github.com/fmtlib/fmt>`__ 6.0

Optional dependencies:

-  `OpenCV <https://github.com/opencv/opencv>`__ 4.0
-  `Boost <https://www.boost.org>`__ 1.70
-  `pybind11 <https://github.com/pybind/pybind11>`__ 2.6.2 (version
   2.9.0 is required for use with Visual Studio 17 2022 and above)

More recent versions of the above are expected to work as well.

Instruction Set Dispatch
------------------------

By default, the Python module contains one variant per instruction set the
compiler supports, such as SSE2, AVX2 or NEON, besides a generic variant. When
the module is imported, it initializes the most capable variant the processor
supports. Configuring with ``-DWITH_DISPATCH=OFF`` builds a single variant for
the compiler's target settings instead.

The environment variable ``HOGPP_DISPATCH`` overrides the selection with the
name of an instruction set, such as ``AVX2``. An empty value selects the
generic variant. Importing the module fails if the processor does not support
the requested instruction set or the build does not contain it.

The generic variant bins gradient orientations exactly. The other variants
approximate the arctangent by a polynomial, which shifts the normalized bin
weight of a gradient by at most :math:`2.12 \times 10^{-5}` for signed and
:math:`4.24 \times 10^{-5}` for unsigned gradients. The features computed by
different variants therefore differ slightly, in the order of
:math:`10^{-4}`, in single as well as double precision. Set
``HOGPP_DISPATCH`` to an empty value where the features must not depend on the
processor.
