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

#ifndef PYTHON_MODULEINITIALIZER_HPP
#define PYTHON_MODULEINITIALIZER_HPP

#include <locale>
#include <optional>
#include <string_view>

#include <pybind11/pybind11.h>

#include "cpufeatures.hpp"

namespace pyhogpp {

struct ModuleInitializer
{
    [[nodiscard]] explicit ModuleInitializer(pybind11::module& m);

    void run() const;

private:
    void run(CPUFeatures<> /*unused*/) const;
    template<ISA Type, ISA... Types>
    void run(CPUFeatures<Type, Types...> /*unused*/) const;

    pybind11::module& m;
    std::optional<const std::string_view> isa;
    pybind11::object logging;
    pybind11::object getLogger;
    pybind11::object moduleName;
    pybind11::object logger;
    pybind11::object debug;
    const std::locale* const loc;
};

} // namespace pyhogpp

#endif
