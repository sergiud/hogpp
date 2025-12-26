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

#ifndef PYTHON_CPUFEATURE_HPP
#define PYTHON_CPUFEATURE_HPP

#include <string_view>

#include "isa.hpp"

namespace pyhogpp {

/**
 * @brief Identifies a CPU feature given an instruction set architecture (ISA)
 * at runtime.
 */
template<ISA Type>
struct CPUFeature;

template<>
struct CPUFeature<ISA::SSE2>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "SSE2";
    }

    [[nodiscard]] static bool supported() noexcept;
};

template<>
struct CPUFeature<ISA::SSE3>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "SSE3";
    }

    [[nodiscard]] static bool supported() noexcept;
};

template<>
struct CPUFeature<ISA::SSSE3>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "SSSE3";
    }

    [[nodiscard]] static bool supported() noexcept;
};

template<>
struct CPUFeature<ISA::SSE4_2>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "SSE4.2";
    }

    [[nodiscard]] static bool supported() noexcept;
};

template<>
struct CPUFeature<ISA::SSE4_1>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "SSE4.1";
    }

    [[nodiscard]] static bool supported() noexcept;
};

template<>
struct CPUFeature<ISA::AVX>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "AVX";
    }

    [[nodiscard]] static bool supported() noexcept;
};

template<>
struct CPUFeature<ISA::AVX2>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "AVX2";
    }

    [[nodiscard]] static bool supported() noexcept;
};

template<>
struct CPUFeature<ISA::AVX512>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "AVX512";
    }

    [[nodiscard]] static bool supported() noexcept;
};

template<>
struct CPUFeature<ISA::AVX10_1>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "AVX10.1";
    }

    [[nodiscard]] static bool supported() noexcept;
};

template<>
struct CPUFeature<ISA::AVX10_2>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "AVX10.2";
    }

    [[nodiscard]] static bool supported() noexcept;
};

template<>
struct CPUFeature<ISA::NEON>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "NEON";
    }

    [[nodiscard]] static bool supported() noexcept;
};

template<>
struct CPUFeature<ISA::SVE>
{
    [[nodiscard]] static bool supported() noexcept;
};

template<>
struct CPUFeature<ISA::SVE128> : CPUFeature<ISA::SVE>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "SVE128";
    }
};

template<>
struct CPUFeature<ISA::SVE256> : CPUFeature<ISA::SVE>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "SVE256";
    }
};

template<>
struct CPUFeature<ISA::SVE512> : CPUFeature<ISA::SVE>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "SVE512";
    }
};

} // namespace pyhogpp

#endif
