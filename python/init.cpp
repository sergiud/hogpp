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

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <locale>
#include <numeric>
#include <ranges>
#include <string_view>
#include <vector>

#include <pybind11/pybind11.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "isa.hpp"

template<std::ranges::random_access_range R1,
         std::ranges::random_access_range R2,
         class Pred = std::ranges::equal_to, class Proj1 = std::identity,
         class Proj2 = std::identity>
    requires std::indirectly_comparable<std::ranges::iterator_t<R1>,
                                        std::ranges::iterator_t<R2>, Pred,
                                        Proj1, Proj2>
[[nodiscard]] std::size_t levenshteinDistance(const R1& a, const R2& b,
                                              Pred pred = {}, Proj1 proj1 = {},
                                              Proj2 proj2 = {})
{
    const auto m = std::ranges::size(a);
    const auto n = std::ranges::size(b);

    std::vector<std::size_t> v0(n + 1);
    std::vector<std::size_t> v1(n + 1);

    std::iota(v0.begin(), v0.end() - 1, 0);

    for (std::size_t i = 0; i != m; ++i) {
        v1[0] = i + 1;

        for (std::size_t j = 0; j != n; ++j) {
            const auto deletion = v0[j + 1] + 1;
            const auto insertion = v1[j] + 1;
            auto substitution = v0[j];

            if (!std::invoke(pred, std::invoke(proj1, a[i]),
                             std::invoke(proj2, b[j]))) {
                ++substitution;
            }

            v1[j + 1] = std::min({deletion, insertion, substitution});
        }

        swap(v0, v1);
    }

    return v0.back();
}

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
    [[noreturn]] static void initialize(pybind11::module& /*m*/)
    {
        // TODO CPU features not available
        throw pybind11::import_error{"cannot initialize"};
    }
};

template<>
struct HOGppModule<ISA::Default>
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
        (void)m;
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
        (void)m;
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
        (void)m;
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
        (void)m;
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
        (void)m;
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
        (void)m;
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
        (void)m;
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
        (void)m;
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
        (void)m;
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

template<>
struct HOGppModule<ISA::AVX10_2>
{
    static void initialize(pybind11::module& m)
    {
#if defined(HAVE_ISA_AVX10_2)
        void init_hogpp_avx10_2(pybind11::module & m);
        init_hogpp_avx10_2(m);
#else
        (void)m;
#endif
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
#if defined(HAVE_ISA_AVX10_2)
        return __builtin_cpu_supports("avx10.2");
#else
        return false;
#endif
    }
};

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

[[nodiscard]] std::string_view getenv(const char* key) noexcept
{
    if (const char* const value = std::getenv(key)) {
        return value;
    }

    return {}; // Avoid invoking std::strlen on a nullptr
}

struct Initializer
{
    [[nodiscard]] Initializer(pybind11::module& m)
        : m{m} // Allow to override the dispatch using HOGPP_DISPATCH
               // environment variable
        , isa{getenv("HOGPP_DISPATCH")}
        , logging{pybind11::module::import("logging")}
        , getLogger{pybind11::getattr(logging, "getLogger")}
        , moduleName{pybind11::getattr(m, "__name__")}
        , logger{getLogger(moduleName)}
        , debug{pybind11::getattr(logger, "debug")}
        , loc{isa.empty() ? nullptr : &std::locale::classic()}
    {
    }

    void run(CPUFeatures<> /*unused*/) const
    {
        if (!isa.empty()) {
            const auto& supported = supportedCPUFeatureNames();

            const auto proj = [loc = *loc](char ch) {
                return std::tolower(ch, loc);
            };

            auto pos = std::ranges::min_element(
                supported, {}, [&isa = isa, &proj](std::string_view test) {
                    return levenshteinDistance(isa, test, {}, proj, proj);
                });

            using namespace fmt::literals;

            if (pos != supported.end()) {
                throw pybind11::import_error{fmt::format(
                    FMT_STRING(
                        "The instruction set specified by the HOGPP_DISPATCH "
                        "environment variable (\"{isa}\") is neither available "
                        "nor supported. The following CPU features are "
                        "supported: {features}. Did you mean {match}?"),
                    "isa"_a = isa, "features"_a = fmt::join(supported, ", "),
                    "match"_a = *pos)};
            }
            else {
                throw pybind11::import_error{fmt::format(
                    FMT_STRING(
                        "The instruction set specified by the "
                        "HOGPP_DISPATCH environment variable (\"{isa}\") is "
                        "neither available nor supported. The following CPU "
                        "features are supported: {features}."),
                    "isa"_a = isa, "features"_a = fmt::join(supported, ", "))};
            }
        }

        HOGppModule<ISA::Default>::initialize(m);
    }

    template<ISA Type, ISA... Types>
    void run(CPUFeatures<Type, Types...> /*unused*/) const
    {
        using namespace fmt::literals;
        bool initialize = false;

        if (!isa.empty()) {
            const auto proj = [loc = *loc](char ch) {
                return std::tolower(ch, loc);
            };

            constexpr std::string_view name = CPUFeature<Type>::name();

            if (std::ranges::equal(isa, name, {}, proj, proj)) {
                if (!CPUFeature<Type>::supported()) {
                    throw pybind11::import_error{fmt::format(
                        FMT_STRING(
                            "ISA specified by the HOGPP_DISPATCH environment "
                            "variable (\"{isa}\") is not supported by the CPU. "
                            "The following CPU features are supported: "
                            "{features}."),
                        "isa"_a = isa,
                        "features"_a =
                            fmt::join(supportedCPUFeatureNames(), ", "))};
                }

                debug(fmt::format(FMT_STRING("found requested ISA {isa}"),
                                  "isa"_a = name));
                initialize = true;
            }
        }
        else {
            initialize = CPUFeature<Type>::supported();
        }

        if (initialize) {
            debug(fmt::format(FMT_STRING("initializing using ISA {isa}"),
                              "isa"_a = CPUFeature<Type>::name()));
            HOGppModule<Type>::initialize(m);
        }
        else {
            run(CPUFeatures<Types...>{});
        }
    }

    pybind11::module& m;
    const std::string_view isa;
    pybind11::object logging;
    pybind11::object getLogger;
    pybind11::object moduleName;
    pybind11::object logger;
    pybind11::object debug;
    const std::locale* loc;
};

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
    const Initializer initializer{m};
    initializer.run(AvailableCPUFeatures{});
}
