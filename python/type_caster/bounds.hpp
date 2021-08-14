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

#ifndef PYTHON_TYPE_CASTER_BOUNDS_HPP
#define PYTHON_TYPE_CASTER_BOUNDS_HPP

// FIXME GCC 14.x workaround for https://github.com/pybind/pybind11/pull/5208
#include <algorithm>

#include <pybind11/pybind11.h>

#include <hogpp/bounds.hpp>

template<>
class pybind11::detail::type_caster<hogpp::Bounds>
{
public:
    PYBIND11_TYPE_CASTER(hogpp::Bounds, _("Bounds"));

    bool load(handle src, bool /*unused*/);

    static handle cast(const hogpp::Bounds& in, return_value_policy /*policy*/,
                       handle /*parent*/);
};

#endif // PYTHON_TYPE_CASTER_BOUNDS_HPP
