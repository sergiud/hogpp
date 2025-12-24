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

#ifndef PYTHON_MODULE_HPP
#define PYTHON_MODULE_HPP

#include <string_view>
#include <vector>

#include <pybind11/pybind11.h>

#include "isa.hpp"
#include "isaconfig.hpp"

template<ISA Type>
struct HOGppModule;

template<>
struct HOGppModule<ISA::Default>
{
    static void initialize(pybind11::module& m);
};

#if defined(HAVE_ISA_SSE2)
template<>
struct HOGppModule<ISA::SSE2>
{
    static void initialize(pybind11::module& m)
    {
        void init_hogpp_sse2(pybind11::module & m);
        init_hogpp_sse2(m);
    }
};
#endif

#if defined(HAVE_ISA_SSE3)
template<>
struct HOGppModule<ISA::SSE3>
{
    static void initialize(pybind11::module& m)
    {
        void init_hogpp_sse3(pybind11::module & m);
        init_hogpp_sse3(m);
    }
};
#endif

#if defined(HAVE_ISA_SSSE3)
template<>
struct HOGppModule<ISA::SSSE3>
{
    static void initialize(pybind11::module& m)
    {
        void init_hogpp_ssse3(pybind11::module & m);
        init_hogpp_ssse3(m);
    }
};
#endif

#if defined(HAVE_ISA_SSE4_1)
template<>
struct HOGppModule<ISA::SSE4_1>
{
    static void initialize(pybind11::module& m)
    {
        void init_hogpp_sse4_1(pybind11::module & m);
        init_hogpp_sse4_1(m);
    }
};
#endif

#if defined(HAVE_ISA_SSE4_2)
template<>
struct HOGppModule<ISA::SSE4_2>
{
    static void initialize(pybind11::module& m)
    {
        void init_hogpp_sse4_2(pybind11::module & m);
        init_hogpp_sse4_2(m);
    }
};
#endif

#if defined(HAVE_ISA_AVX2)
template<>
struct HOGppModule<ISA::AVX>
{
    static void initialize(pybind11::module& m)
    {
        void init_hogpp_avx(pybind11::module & m);
        init_hogpp_avx(m);
    }
};
#endif

#if defined(HAVE_ISA_AVX2)
template<>
struct HOGppModule<ISA::AVX2>
{
    static void initialize(pybind11::module& m)
    {
        void init_hogpp_avx2(pybind11::module & m);
        init_hogpp_avx2(m);
    }
};
#endif

#if defined(HAVE_ISA_AVX512F)
template<>
struct HOGppModule<ISA::AVX512>
{
    static void initialize(pybind11::module& m)
    {
        void init_hogpp_avx512f(pybind11::module & m);
        init_hogpp_avx512f(m);
    }
};
#endif

#if defined(HAVE_ISA_AVX10_1)
template<>
struct HOGppModule<ISA::AVX10_1>
{
    static void initialize(pybind11::module& m)
    {
        void init_hogpp_avx10_1(pybind11::module & m);
        init_hogpp_avx10_1(m);
    }
};
#endif

#if defined(HAVE_ISA_AVX10_2)
template<>
struct HOGppModule<ISA::AVX10_2>
{
    static void initialize(pybind11::module& m)
    {
        void init_hogpp_avx10_2(pybind11::module & m);
        init_hogpp_avx10_2(m);
    }
};
#endif

#if defined(HAVE_ISA_NEON)
template<>
struct HOGppModule<ISA::NEON>
{
    static void initialize(pybind11::module& m)
    {
        void init_hogpp_neon(pybind11::module & m);
        init_hogpp_neon(m);
    }
};
#endif

template<ISA Type>
concept HOGppModuleSupported =
    requires(pybind11::module& m) { HOGppModule<Type>::initialize(m); };

[[nodiscard]] std::vector<std::string_view> supportedCPUFeatureNames();

#endif
