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

#include "bounds.hpp"

#include <tuple>

#include <nanobind/stl/tuple.h>

namespace nanobind::detail {

bool type_caster<hogpp::Bounds>::from_python(handle src, std::uint8_t /*flags*/,
                                             cleanup_list* /*cleanup*/) noexcept
{
    try {
        std::tie(value.y, value.x, value.height, value.width) =
            nanobind::cast<std::tuple<int, int, int, int>>(src);
    }
    catch (const nanobind::python_error&) {
        return false;
    }
    catch (const nanobind::cast_error&) {
        return false;
    }

    return true;
}

handle type_caster<hogpp::Bounds>::from_cpp(const hogpp::Bounds& in,
                                            rv_policy /*policy*/,
                                            cleanup_list* /*cleanup*/)
{
    return make_tuple(in.y, in.x, in.height, in.width).release();
}

} // namespace nanobind::detail
