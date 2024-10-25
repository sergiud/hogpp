//
// HOGpp - Fast histogram of oriented gradients computation using integral
// histograms
//
// Copyright 2024 Sergiu Deitsch <sergiu.deitsch@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef PYTHON_HOGPP_HPP
#define PYTHON_HOGPP_HPP

#include <pybind11/pybind11.h>

#define HOGPP_PYBIND11_MAKE_VERSION(major, minor, patch) \
    ((major) << 16 | (minor) << 8 | (patch))

#define HOGPP_PYBIND11_VERSION_AT_LEAST(...)          \
    (((PYBIND11_VERSION_HEX & 0xFF'FF'FF'00) >> 8) >= \
     HOGPP_PYBIND11_MAKE_VERSION(__VA_ARGS__))

#if defined(Py_GIL_DISABLED) && HOGPP_PYBIND11_VERSION_AT_LEAST(2, 13, 0)
#    define HOGPP_GIL_DISABLED
#endif // (defined(Py_GIL_DISABLED) && HOGPP_PYBIND11_AT_LEAST(2, 13, 0))

#endif // PYTHON_HOGPP_HPP
