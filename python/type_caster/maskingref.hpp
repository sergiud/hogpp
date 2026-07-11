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

#include <cstdint>
#include <optional>

#include <Eigen/Core>

#include <nanobind/nanobind.h>

#include "../maskingref.hpp"

/**
 * @brief Converts a Python mask into a reference to a masking callable.
 *
 * The mask is either `None`, an object indexable by a 2-tuple of pixel
 * coordinates, such as a `numpy.ndarray`, or a callable that accepts such a
 * 2-tuple. Any other object raises a `TypeError` naming its type.
 *
 * This specialization replaces the generic `std::optional` caster of
 * nanobind, which loads the contained value through a nested caster. Here, the
 * MaskingRef refers to the indexer owned by this caster, so the converted
 * value is valid only as long as the caster, which nanobind keeps alive for
 * the duration of the bound function call. The converted value must therefore
 * not be obtained through nanobind::cast, which returns it after destroying
 * the caster.
 *
 * nanobind requires from_python() to be noexcept. An invalid mask is therefore
 * accepted by from_python() and rejected with a `TypeError` once the bound
 * function retrieves the converted value. Rejecting it in from_python()
 * instead would only produce the generic overload resolution error of
 * nanobind, which does not name the type of the mask.
 *
 * @note The lifetime of argument casters is specific to nanobind and must be
 * revisited for other binding libraries.
 */
template<>
class nanobind::detail::type_caster<std::optional<pyhogpp::MaskingRef>>
{
public:
    using Value = std::optional<pyhogpp::MaskingRef>;

    static constexpr auto Name = const_name(
        "collections.abc.Callable[[tuple[int, int]], bool] | numpy.ndarray | "
        "None");

    template<class T>
    using Cast = movable_cast_t<T>;

    template<class T>
    [[nodiscard]] static constexpr bool can_cast() noexcept
    {
        return true;
    }

    type_caster() = default;
    type_caster(const type_caster& other) = delete;
    type_caster(type_caster&& other) = delete;
    type_caster& operator=(const type_caster& other) = delete;
    type_caster& operator=(type_caster&& other) = delete;
    ~type_caster() = default;

    bool from_python(handle src, std::uint8_t flags,
                     cleanup_list* cleanup) noexcept;

    explicit operator Value*()
    {
        return &validated();
    }

    explicit operator Value&()
    {
        return validated();
    }

    explicit operator Value&&()
    {
        return std::move(validated());
    }

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

    [[nodiscard]] Value& validated();

    std::optional<Indexer> indexer_;
    // Mask that provides neither an indexer nor a call operator.
    object invalid_;
    Value value_;
};

#endif // PYTHON_TYPE_CASTER_MASKINGREF_HPP
