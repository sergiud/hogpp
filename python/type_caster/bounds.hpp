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

#include <nanobind/nanobind.h>

#include <hogpp/bounds.hpp>

template<>
class nanobind::detail::type_caster<hogpp::Bounds>
{
public:
    NB_TYPE_CASTER(hogpp::Bounds, const_name("Bounds"))

    bool from_python(handle src, std::uint8_t /*flags*/,
                     cleanup_list* /*cleanup*/) noexcept;

    static handle from_cpp(const hogpp::Bounds& in, rv_policy /*policy*/,
                           cleanup_list* /*cleanup*/);
};

#endif // PYTHON_TYPE_CASTER_BOUNDS_HPP
