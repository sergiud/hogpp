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

#ifndef PYTHON_HOGPP_INTERPOLATION_HPP
#define PYTHON_HOGPP_INTERPOLATION_HPP

#include <cstdint>

#include <nanobind/nanobind.h>

// Selects the hogpp::IntegralHOGDescriptor spatial interpolation trait.
// A compile-time template parameter of the descriptor, not a runtime
// functor, so this only steers which precompiled variant is picked.
enum class InterpolationType
{
    Orientation,
    Trilinear
};

namespace nanobind::detail {

template<>
class type_caster<InterpolationType>
{
public:
    NB_TYPE_CASTER(InterpolationType, const_name("Interpolation"))

    bool from_python(handle src, std::uint8_t flags,
                     cleanup_list* cleanup) noexcept;
    static handle from_cpp(InterpolationType in, rv_policy /*policy*/,
                           cleanup_list* /*cleanup*/);
};

} // namespace nanobind::detail

#endif // PYTHON_HOGPP_INTERPOLATION_HPP
