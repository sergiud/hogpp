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

#ifndef PYTHON_TYPE_CASTER_ARRAY2I_HPP
#define PYTHON_TYPE_CASTER_ARRAY2I_HPP

#include <Eigen/Core>

#include <cstdint>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>

// nanobind's own <nanobind/eigen/dense.h> caster only accepts inputs that
// already support the buffer/DLPack protocol (e.g., numpy.ndarray), unlike
// pybind11's Eigen support, which additionally accepted arbitrary Python
// sequences by first converting them into a numpy array. Since cell_size,
// block_size, and block_stride are documented and tested to accept plain
// 2-tuples (e.g., an image's shape slice), this full specialization replaces
// nanobind's Eigen caster for exactly this type, delegating to the
// std::pair<int, int> caster, which accepts any 2-element sequence.
template<>
class nanobind::detail::type_caster<Eigen::Array2i>
{
public:
    NB_TYPE_CASTER(Eigen::Array2i, const_name("tuple[int, int]"))

    bool from_python(handle src, std::uint8_t flags,
                     cleanup_list* /*cleanup*/) noexcept
    {
        std::pair<int, int> p;

        if (!try_cast(src, p,
                      (flags & static_cast<std::uint8_t>(cast_flags::convert)) != 0)) {
            return false;
        }

        value = Eigen::Array2i{p.first, p.second};

        return true;
    }

    static handle from_cpp(const Eigen::Array2i& in, rv_policy policy,
                           cleanup_list* cleanup)
    {
        return make_caster<std::pair<int, int>>::from_cpp(
            std::pair<int, int>{in.x(), in.y()}, policy, cleanup);
    }
};

#endif // PYTHON_TYPE_CASTER_ARRAY2I_HPP
