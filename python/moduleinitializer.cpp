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
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <ranges>
#include <string_view>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "cpufeature.hpp"
#include "moduledispatch.hpp"
#include "moduleinitializer.hpp"

namespace pyhogpp {
namespace {

[[nodiscard]] std::string_view getenv(const char* key) noexcept
{
    if (const char* const value = std::getenv(key)) {
        return value;
    }

    return {}; // Avoid invoking std::strlen on a nullptr
}

} // namespace

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

ModuleInitializer::ModuleInitializer(pybind11::module& m)
    : m{m} // Allow to override the dispatch using the HOGPP_DISPATCH
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

void ModuleInitializer::run() const
{
    run(AvailableCPUFeatures{});
}

void ModuleInitializer::run(CPUFeatures<> /*unused*/) const
{
    if (!isa.empty()) {
        using namespace fmt::literals;

        const auto& supported = supportedCPUFeatureNames();

        if (supported.empty()) {
            // No dispatch for additional instruction sets defined: nothing to
            // do
            throw pybind11::import_error{fmt::format(
                FMT_STRING(
                    "The instruction set specified by the HOGPP_DISPATCH "
                    "environment variable (\"{isa}\") is neither available "
                    "nor supported."),
                "isa"_a = isa)};
        }

        const auto proj = [loc = *loc](char ch) {
            return std::tolower(ch, loc);
        };

        auto pos = std::ranges::min_element(
            supported, {}, [&isa = isa, &proj](std::string_view test) {
                return levenshteinDistance(isa, test, {}, proj, proj);
            });

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

        throw pybind11::import_error{fmt::format(
            FMT_STRING("The instruction set specified by the "
                       "HOGPP_DISPATCH environment variable (\"{isa}\") is "
                       "neither available nor supported. The following CPU "
                       "features are supported: {features}."),
            "isa"_a = isa, "features"_a = fmt::join(supported, ", "))};
    }

    ModuleDispatch<ISA::Default>::initialize(m);
}

template<ISA Type, ISA... Types>
void ModuleInitializer::run(CPUFeatures<Type, Types...> /*unused*/) const
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
        // Make sure the module initialization is defined at compile-time
        if constexpr (ModuleDispatchSupported<Type>) {
            debug(fmt::format(FMT_STRING("initializing using ISA {isa}"),
                              "isa"_a = CPUFeature<Type>::name()));
            ModuleDispatch<Type>::initialize(m);
        }
        else {
            // Otherwise continue searching
            initialize = false;
        }
    }

    if (!initialize) {
        run(CPUFeatures<Types...>{});
    }
}

} // namespace pyhogpp
