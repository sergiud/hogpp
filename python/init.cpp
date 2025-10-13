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

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <locale>
#include <ranges>
#include <string_view>
#include <vector>

#include <pybind11/pybind11.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "isa.hpp"

enum class ISA : std::uint8_t
{
    Default,
    SSE2,
    SSE3,
    SSSE3,
    SSE4_1,
    SSE4_2,
    AVX,
    AVX2,
    AVX512,
    AVX10_1,
    AVX10_2,
};

template<ISA Type>
struct CPUFeature;

template<ISA Type>
struct HOGppModule
{
    static void initialize(pybind11::module& m)
    {
        void init_hogpp_default(pybind11::module & m);
        init_hogpp_default(m);
    }
};

template<>
struct HOGppModule<ISA::SSE2>
{
    static void initialize(pybind11::module& m)
    {
#if defined(HAVE_ISA_SSE2)
        void init_hogpp_sse2(pybind11::module & m);
        init_hogpp_sse2(m);
#else
#endif
    }
};

template<>
struct CPUFeature<ISA::SSE2>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "SSE2";
    }

    [[nodiscard]] static bool supported() noexcept
    {
#if defined(HAVE_ISA_SSE2)
        return __builtin_cpu_supports("sse2");
#else
        return false;
#endif
    }
};

template<>
struct HOGppModule<ISA::SSE3>
{
    static void initialize(pybind11::module& m)
    {
#if defined(HAVE_ISA_SSE3)
        void init_hogpp_sse3(pybind11::module & m);
        init_hogpp_sse3(m);
#else
#endif
    }
};

template<>
struct CPUFeature<ISA::SSE3>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "SSE3";
    }

    [[nodiscard]] static bool supported() noexcept
    {
#if defined(HAVE_ISA_SSE3)
        return __builtin_cpu_supports("sse3");
#else
        return false;
#endif
    }
};

template<>
struct HOGppModule<ISA::SSSE3>
{
    static void initialize(pybind11::module& m)
    {
#if defined(HAVE_ISA_SSSE3)
        void init_hogpp_ssse3(pybind11::module & m);
        init_hogpp_ssse3(m);
#else
#endif
    }
};

template<>
struct CPUFeature<ISA::SSSE3>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "SSSE3";
    }

    [[nodiscard]] static bool supported() noexcept
    {
#if defined(HAVE_ISA_SSSE3)
        return __builtin_cpu_supports("ssse3");
#else
        return false;
#endif
    }
};

template<>
struct HOGppModule<ISA::SSE4_1>
{
    static void initialize(pybind11::module& m)
    {
#if defined(HAVE_ISA_SSE4_1)
        void init_hogpp_sse4_1(pybind11::module & m);
        init_hogpp_sse4_1(m);
#else
#endif
    }
};

template<>
struct CPUFeature<ISA::SSE4_1>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "SSE4.1";
    }

    [[nodiscard]] static bool supported() noexcept
    {
#if defined(HAVE_ISA_SSE4_1)
        return __builtin_cpu_supports("sse4.1");
#else
        return false;
#endif
    }
};

template<>
struct HOGppModule<ISA::SSE4_2>
{
    static void initialize(pybind11::module& m)
    {
#if defined(HAVE_ISA_SSE4_2)
        void init_hogpp_sse4_2(pybind11::module & m);
        init_hogpp_sse4_2(m);
#else
#endif
    }
};

template<>
struct CPUFeature<ISA::SSE4_2>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "SSE4.2";
    }

    [[nodiscard]] static bool supported() noexcept
    {
#if defined(HAVE_ISA_SSE4_2)
        return __builtin_cpu_supports("sse4.2");
#else
        return false;
#endif
    }
};

template<>
struct HOGppModule<ISA::AVX>
{
    static void initialize(pybind11::module& m)
    {
#if defined(HAVE_ISA_AVX2)
        void init_hogpp_avx(pybind11::module & m);
        init_hogpp_avx(m);
#else
        // TODO Throw
#endif
    }
};

template<>
struct CPUFeature<ISA::AVX>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "AVX";
    }

    [[nodiscard]] static bool supported() noexcept
    {
#if defined(HAVE_ISA_AVX2)
        return __builtin_cpu_supports("avx");
#else
        return false;
#endif
    }
};

template<>
struct HOGppModule<ISA::AVX2>
{
    static void initialize(pybind11::module& m)
    {
#if defined(HAVE_ISA_AVX2)
        void init_hogpp_avx2(pybind11::module & m);
        init_hogpp_avx2(m);
#else
#endif
    }
};

template<>
struct CPUFeature<ISA::AVX2>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "AVX2";
    }

    [[nodiscard]] static bool supported() noexcept
    {
#if defined(HAVE_ISA_AVX2)
        return __builtin_cpu_supports("avx2");
#else
        return false;
#endif
    }
};

template<>
struct HOGppModule<ISA::AVX512>
{
    static void initialize(pybind11::module& m)
    {
#if defined(HAVE_ISA_AVX512F)
        void init_hogpp_avx512f(pybind11::module & m);
        init_hogpp_avx512f(m);
#else
        // TODO Trhow
#endif
    }
};

template<>
struct CPUFeature<ISA::AVX512>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "AVX512";
    }

    [[nodiscard]] static bool supported() noexcept
    {
#if defined(HAVE_ISA_AVX512F)
        return __builtin_cpu_supports("avx512f");
#else
        return false;
#endif
    }
};

template<>
struct HOGppModule<ISA::AVX10_1>
{
    static void initialize(pybind11::module& m)
    {
#if defined(HAVE_ISA_AVX10_1)
        void init_hogpp_avx10_1(pybind11::module & m);
        init_hogpp_avx10_1(m);
#else
        // TODO Throw
#endif
    }
};

template<>
struct CPUFeature<ISA::AVX10_1>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "AVX10.1";
    }

    [[nodiscard]] static bool supported() noexcept
    {
#if defined(HAVE_ISA_AVX10_1)
        return __builtin_cpu_supports("avx10.1");
#else
        return false;
#endif
    }
};

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

template<>
struct CPUFeature<ISA::AVX10_2>
{
    [[nodiscard]] constexpr static std::string_view name() noexcept
    {
        return "AVX10.2";
    }

    [[nodiscard]] static bool supported() noexcept
    {
        return __builtin_cpu_supports("avx10.2");
    }
};
#endif

template<ISA... Types>
struct CPUFeatures
{
};

using AvailableCPUFeatures = CPUFeatures
    // clang-format off
<
      ISA::AVX10_2
    , ISA::AVX10_1
    , ISA::AVX512
    , ISA::AVX2
    , ISA::AVX
    , ISA::SSE4_2
    , ISA::SSE4_1
    , ISA::SSSE3
    , ISA::SSE3
    , ISA::SSE2
>;
// clang-format on

namespace {

void supportedCPUFeatureNames(std::vector<std::string_view>& /*names*/,
                              CPUFeatures<> /*unused*/)
{
}

template<ISA Type, ISA... Types>
void supportedCPUFeatureNames(std::vector<std::string_view>& names,
                              CPUFeatures<Type, Types...> /*unused*/)
{
    if (CPUFeature<Type>::supported()) {
        names.push_back(CPUFeature<Type>::name());
    }

    supportedCPUFeatureNames(names, CPUFeatures<Types...>{});
}

[[nodiscard]] std::vector<std::string_view> supportedCPUFeatureNames()
{
    std::vector<std::string_view> names;
    supportedCPUFeatureNames(names, AvailableCPUFeatures{});
    std::ranges::sort(names);
    return names;
}

void init_hogpp(pybind11::module& m, std::string_view isa,
                const std::locale* /*loc*/, CPUFeatures<> /*unused*/)
{
    if (!isa.empty()) {
        using namespace fmt::literals;

        throw pybind11::import_error{fmt::format(
            FMT_STRING(
                "ISA specified by the HOGPP_DISPATCH environment "
                "variable (\"{isa}\") is neither available nor supported. The "
                "following CPU features are supported: {features}."),
            "isa"_a = isa,
            "features"_a = fmt::join(supportedCPUFeatureNames(), ", "))};
    }

    HOGppModule<ISA::Default>::initialize(m);
}

template<ISA Type, ISA... Types>
void init_hogpp(pybind11::module& m, std::string_view isa,
                const std::locale* loc, CPUFeatures<Type, Types...> /*unused*/)
{
    bool initialize = false;

    if (!isa.empty()) {
        const auto proj = [loc = *loc](char ch) {
            return std::tolower(ch, loc);
        };

        constexpr std::string_view name = CPUFeature<Type>::name();

        if (std::ranges::equal(isa, name, {}, proj, proj)) {
            if (!CPUFeature<Type>::supported()) {
                using namespace fmt::literals;

                throw pybind11::import_error{fmt::format(
                    FMT_STRING(
                        "ISA specified by the HOGPP_DISPATCH environment "
                        "variable (\"{isa}\") is not supported by the CPU. The "
                        "following CPU features are supported: {features}."),
                    "isa"_a = isa,
                    "features"_a =
                        fmt::join(supportedCPUFeatureNames(), ", "))};
            }

            initialize = true;
        }
    }
    else {
        initialize = CPUFeature<Type>::supported();
    }

    if (initialize) {
        HOGppModule<Type>::initialize(m);
    }
    else {
        init_hogpp(m, isa, loc, CPUFeatures<Types...>{});
    }
}

} // namespace

#if defined(HOGPP_GIL_DISABLED)
#    define HOGPP_MODULE(name, module, ...) \
        PYBIND11_MODULE(name, module, pybind11::mod_gil_not_used())
#else // !defined(HOGPP_GIL_DISABLED)
#    define HOGPP_MODULE PYBIND11_MODULE
#endif // defined(HOGPP_GIL_DISABLED)

#if defined(HOGPP_SKBUILD)
#    define HOGPP_MODULE_NAME _hogpp
#else // !defined(HOGPP_SKBUILD)
#    define HOGPP_MODULE_NAME hogpp
#endif // defined(HOGPP_SKBUILD)

HOGPP_MODULE(HOGPP_MODULE_NAME, m)
{
    // TODO Log detected dispatch
    if (const char* const isa = std::getenv("HOGPP_DISPATCH")) {
        // Allow to override the dispatch using HOGPP_DISPATCH environment
        // variable
        init_hogpp(m, isa, &std::locale::classic(), AvailableCPUFeatures{});
    }
    else {
        using namespace std::string_view_literals;
        init_hogpp(m, ""sv, nullptr, AvailableCPUFeatures{});
    }
}
