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

#ifndef PYTHON_TYPE_CASTER_MASKINGREF_HPP
#define PYTHON_TYPE_CASTER_MASKINGREF_HPP

// FIXME GCC 14.x workaround for https://github.com/pybind/pybind11/pull/5208
#include <algorithm>
#include <optional>

#include <Eigen/Core>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../maskingref.hpp"

/**
 * @brief Converts a Python mask into a reference to a masking callable.
 *
 * The mask is either `None`, an object indexable by a 2-tuple of pixel
 * coordinates, such as a `numpy.ndarray`, or a callable that accepts such a
 * 2-tuple. Any other object raises a `TypeError` naming its type.
 *
 * This specialization replaces the generic `std::optional` caster of
 * pybind11, which loads the contained value through a temporary caster and
 * would leave the MaskingRef referring to a destroyed indexer. Here, the
 * MaskingRef refers to the indexer owned by this caster, so the converted
 * value is valid only as long as the caster, which pybind11 keeps alive for
 * the duration of the bound function call. The converted value must therefore
 * not be obtained through pybind11::cast, which returns it after destroying
 * the caster.
 *
 * @note The lifetime of argument casters is specific to pybind11 and must be
 * revisited for other binding libraries.
 */
template<>
class pybind11::detail::type_caster<std::optional<pyhogpp::MaskingRef>>
{
public:
    PYBIND11_TYPE_CASTER(
        std::optional<pyhogpp::MaskingRef>,
        _("Optional[Union[Callable[[Tuple[int, int]], bool], numpy.ndarray]]"));

    type_caster() = default;
    type_caster(const type_caster& other) = delete;
    type_caster(type_caster&& other) = delete;
    type_caster& operator=(const type_caster& other) = delete;
    type_caster& operator=(type_caster&& other) = delete;
    ~type_caster() = default;

    bool load(handle src, bool /*unused*/);

private:
    class Indexer
    {
    public:
        explicit Indexer(object getitem) noexcept;

        [[nodiscard]] bool operator()(Eigen::DenseIndex i,
                                      Eigen::DenseIndex j) const;

    private:
        object getitem_;
    };

    std::optional<Indexer> indexer_;
};

#endif // PYTHON_TYPE_CASTER_MASKINGREF_HPP
