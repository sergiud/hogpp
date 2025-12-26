//
// HOGpp - Fast histogram of oriented gradients computation using integral
// histograms
//
// Copyright 2025 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

#ifndef PYTHON_MODULEDISPATCH_HPP
#define PYTHON_MODULEDISPATCH_HPP

#include <pybind11/pybind11.h>

#include "isa.hpp"
#include "isaconfig.hpp"

namespace pyhogpp {

/**
 * @brief Provides dispatch for the module functionality compiled using the
 * specified instruction set architecture (ISA).
 */
template<ISA Type>
struct ModuleDispatch;

template<>
struct ModuleDispatch<ISA::Generic>
{
    static void initialize(pybind11::module& m);
};

#if defined(HAVE_ISA_SSE2)
template<>
struct ModuleDispatch<ISA::SSE2>
{
    static void initialize(pybind11::module& m);
};
#endif

#if defined(HAVE_ISA_SSE3)
template<>
struct ModuleDispatch<ISA::SSE3>
{
    static void initialize(pybind11::module& m);
};
#endif

#if defined(HAVE_ISA_SSSE3)
template<>
struct ModuleDispatch<ISA::SSSE3>
{
    static void initialize(pybind11::module& m);
};
#endif

#if defined(HAVE_ISA_SSE4_1)
template<>
struct ModuleDispatch<ISA::SSE4_1>
{
    static void initialize(pybind11::module& m);
};
#endif

#if defined(HAVE_ISA_SSE4_2)
template<>
struct ModuleDispatch<ISA::SSE4_2>
{
    static void initialize(pybind11::module& m);
};
#endif

#if defined(HAVE_ISA_AVX2)
template<>
struct ModuleDispatch<ISA::AVX>
{
    static void initialize(pybind11::module& m);
};
#endif

#if defined(HAVE_ISA_AVX2)
template<>
struct ModuleDispatch<ISA::AVX2>
{
    static void initialize(pybind11::module& m);
};
#endif

#if defined(HAVE_ISA_AVX512F)
template<>
struct ModuleDispatch<ISA::AVX512>
{
    static void initialize(pybind11::module& m);
};
#endif

#if defined(HAVE_ISA_AVX10_1)
template<>
struct ModuleDispatch<ISA::AVX10_1>
{
    static void initialize(pybind11::module& m);
};
#endif

#if defined(HAVE_ISA_AVX10_2)
template<>
struct ModuleDispatch<ISA::AVX10_2>
{
    static void initialize(pybind11::module& m);
};
#endif

#if defined(HAVE_ISA_NEON)
template<>
struct ModuleDispatch<ISA::NEON>
{
    static void initialize(pybind11::module& m);
};
#endif

#if defined(HAVE_ISA_SVE128)
template<>
struct ModuleDispatch<ISA::SVE128>
{
    static void initialize(pybind11::module& m);
};
#endif

#if defined(HAVE_ISA_SVE256)
template<>
struct ModuleDispatch<ISA::SVE256>
{
    static void initialize(pybind11::module& m);
};
#endif

#if defined(HAVE_ISA_SVE512)
template<>
struct ModuleDispatch<ISA::SVE512>
{
    static void initialize(pybind11::module& m);
};
#endif

template<ISA Type>
concept ModuleDispatchSupported =
    requires(pybind11::module& m) { ModuleDispatch<Type>::initialize(m); };

} // namespace pyhogpp

#endif
