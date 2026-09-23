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

#include <pybind11/pybind11.h>

// Selects the hogpp::IntegralHOGDescriptor spatial interpolation trait.
// A compile-time template parameter of the descriptor, not a runtime
// functor, so this only steers which precompiled variant is picked.
enum class InterpolationType
{
    Orientation,
    Trilinear
};

namespace pybind11::detail {

template<>
class type_caster<InterpolationType>
{
public:
    bool load(handle src, bool);
    static handle cast(InterpolationType in, return_value_policy /*policy*/,
                       handle /*parent*/);

private:
    PYBIND11_TYPE_CASTER(InterpolationType, _("Interpolation"));
};

} // namespace pybind11::detail

#endif // PYTHON_HOGPP_INTERPOLATION_HPP
