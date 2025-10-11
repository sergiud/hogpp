//
// HOGpp - Fast histogram of oriented gradients computation using integral
// histograms
//
// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

#include <cstdlib>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "cpufeature.hpp"
#include "levenshtein.hpp"
#include "moduledispatch.hpp"
#include "moduleinitializer.hpp"

namespace pyhogpp {
namespace {

[[nodiscard]] std::optional<std::string_view> getenv(const char* key) noexcept
{
    if (const char* const value = std::getenv(key)) {
        return value;
    }

    return std::nullopt; // Avoid invoking std::strlen on a nullptr
}

// pybind11::import_error is translated to a Python ImportError at the
// module-init boundary. nanobind raises the equivalent via nb::import_error
// (or PyErr_SetString directly), so these throw sites will need updating on
// migration.
[[noreturn]] void reportUnsupportedDispatch(std::string_view isa)
{
    using namespace fmt::literals;

    throw pybind11::import_error{fmt::format(
        FMT_STRING(
            "The instruction set specified by the HOGPP_DISPATCH environment "
            "variable (\"{isa}\") is neither available nor supported. Only "
            "the generic (non-vectorized) dispatch is available in this "
            "build."),
        "isa"_a = isa)};
}

[[noreturn]] void reportUnsupportedDispatch(
    std::string_view isa, const std::vector<std::string_view>& supported)
{
    using namespace fmt::literals;

    throw pybind11::import_error{fmt::format(
        FMT_STRING(
            "The instruction set specified by the HOGPP_DISPATCH environment "
            "variable (\"{isa}\") is neither available nor supported. The "
            "following CPU features are supported: {features}."),
        "isa"_a = isa, "features"_a = fmt::join(supported, ", "))};
}

[[noreturn]] void reportUnsupportedDispatch(
    std::string_view isa, std::string_view match,
    const std::vector<std::string_view>& supported)
{
    using namespace fmt::literals;

    throw pybind11::import_error{fmt::format(
        FMT_STRING("The instruction set specified by the HOGPP_DISPATCH "
                   "environment variable (\"{isa}\") is neither available nor "
                   "supported. The following CPU features are supported: "
                   "{features}. Did you mean {match}?"),
        "isa"_a = isa, "features"_a = fmt::join(supported, ", "),
        "match"_a = match)};
}

} // namespace

ModuleInitializer::ModuleInitializer(pybind11::module& m)
    : m{m} // Allow to override the dispatch using the HOGPP_DISPATCH
           // environment variable
    , isa{getenv("HOGPP_DISPATCH")}
    , logging{pybind11::module::import("logging")}
    , getLogger{pybind11::getattr(logging, "getLogger")}
    , moduleName{pybind11::getattr(m, "__name__")}
    , logger{getLogger(moduleName)}
    , debug{pybind11::getattr(logger, "debug")}
    , loc{!isa ? nullptr : &std::locale::classic()}
{
}

void ModuleInitializer::run() const
{
    if (isa && isa->empty()) {
        // Setting the environment variable to an empty string implies generic
        // dispatch: take a shortcut
        debug("using generic instruction set");
        ModuleDispatch<ISA::Generic>::initialize(m);
    }
    else {
        // Cycle through available CPU features while taking the dispatch set
        // using the environment variable if any
        run(AvailableCPUFeatures{});
    }
}

void ModuleInitializer::run(CPUFeatures<> /*unused*/) const
{
    if (isa) {
        const auto& supported = supportedCPUFeatureNames();

        if (supported.empty()) {
            // No dispatch for additional instruction sets defined: nothing to
            // do
            reportUnsupportedDispatch(*isa);
        }

        if (const auto match = findClosestMatch(*isa, supported, *loc)) {
            reportUnsupportedDispatch(*isa, *match, supported);
        }

        reportUnsupportedDispatch(*isa, supported);
    }

    // Exhaused all SIMD features: fallback to generic dispatch
    ModuleDispatch<ISA::Generic>::initialize(m);
}

template<ISA Type, ISA... Types>
void ModuleInitializer::run(CPUFeatures<Type, Types...> /*unused*/) const
{
    using namespace fmt::literals;
    bool initialize = false;

    if (isa) {
        const auto proj = [loc = *loc](char ch) {
            return std::tolower(ch, loc);
        };

        constexpr std::string_view name = CPUFeature<Type>::name();

        if (std::ranges::equal(*isa, name, {}, proj, proj)) {
            if (!CPUFeature<Type>::supported()) {
                const auto& supported = supportedCPUFeatureNames();

                if (supported.empty()) {
                    // No dispatch for additional instruction sets defined:
                    // nothing to do
                    reportUnsupportedDispatch(*isa);
                }

                reportUnsupportedDispatch(*isa, supported);
            }

            debug(
                fmt::format(FMT_STRING("found requested instruction set {isa}"),
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
            debug(fmt::format(
                FMT_STRING("initializing using {isa} instruction set"),
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
