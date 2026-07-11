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

#include <hogpp/prefix.hpp>

#include <nanobind/stl/string.h>

#include "magnitude.hpp"

namespace nanobind::detail {

bool type_caster<MagnitudeType>::from_python(handle src, std::uint8_t flags,
                                             cleanup_list* /*cleanup*/) noexcept
{
    std::string name;

    if (!try_cast(src, name,
                  (flags & static_cast<std::uint8_t>(cast_flags::convert)) != 0)) {
        return false;
    }

    if (name == "identity") {
        value = MagnitudeType::Identity;
    }
    else if (name == "square") {
        value = MagnitudeType::Square;
    }
    else if (name == "sqrt") {
        value = MagnitudeType::Sqrt;
    }
    else {
        return false;
    }

    return true;
}

handle type_caster<MagnitudeType>::from_cpp(MagnitudeType in,
                                            rv_policy /*policy*/,
                                            cleanup_list* /*cleanup*/)
{
    str result;

    switch (in) {
        case MagnitudeType::Identity:
            result = str{"identity"};
            break;
        case MagnitudeType::Square:
            result = str{"square"};
            break;
        case MagnitudeType::Sqrt:
            result = str{"sqrt"};
            break;
    }

    return result.release();
}

} // namespace nanobind::detail

template class Magnitude<float>;
template class Magnitude<double>;

#include <hogpp/suffix.hpp>
