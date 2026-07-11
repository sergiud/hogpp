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

#ifndef PYTHON_HOGPP_MAGNITUDE_HPP
#define PYTHON_HOGPP_MAGNITUDE_HPP

#include <cstdint>
#include <type_traits>
#include <variant>

#include <nanobind/nanobind.h>

#include <hogpp/gradientmagnitude.hpp>
#include <hogpp/gradientsqrtmagnitude.hpp>
#include <hogpp/gradientsquaremagnitude.hpp>
#include <hogpp/prefix.hpp>

enum class MagnitudeType
{
    Identity,
    Square,
    Sqrt
};

namespace nanobind::detail {

template<>
class type_caster<MagnitudeType>
{
public:
    NB_TYPE_CASTER(MagnitudeType, const_name("Magnitude"))

    bool from_python(handle src, std::uint8_t flags,
                     cleanup_list* cleanup) noexcept;
    static handle from_cpp(MagnitudeType in, rv_policy /*policy*/,
                           cleanup_list* /*cleanup*/);
};

} // namespace nanobind::detail

template<class T>
class Magnitude
{
public:
    [[nodiscard]] explicit Magnitude(
        MagnitudeType type = MagnitudeType::Identity)
    {
        switch (type) {
            case MagnitudeType::Identity:
                voting_ = hogpp::GradientMagnitude<T>{};
                break;
            case MagnitudeType::Square:
                voting_ = hogpp::GradientSquareMagnitude<T>{};
                break;
            case MagnitudeType::Sqrt:
                voting_ = hogpp::GradientSqrtMagnitude<T>{};
                break;
        }
    }

    template<class Derived1, class Derived2>
    [[nodiscard]] constexpr decltype(auto) operator()(
        const Eigen::TensorBase<Derived1, Eigen::ReadOnlyAccessors>& dx,
        const Eigen::TensorBase<Derived2, Eigen::ReadOnlyAccessors>& dy) const
    {
        return std::visit(
            [&dx, &dy](auto& voting) -> std::common_type_t<Derived1, Derived2> {
                return voting(dx, dy);
            },
            voting_);
    }

    [[nodiscard]] MagnitudeType type() const noexcept
    {
        return std::visit(MagnitudeVisitor{}, voting_);
    }

private:
    struct MagnitudeVisitor
    {
        [[nodiscard]] constexpr MagnitudeType operator()(
            const hogpp::GradientMagnitude<T>& /*unused*/) const noexcept
        {
            return MagnitudeType::Identity;
        }

        [[nodiscard]] constexpr MagnitudeType operator()(
            const hogpp::GradientSquareMagnitude<T>& /*unused*/) const noexcept
        {
            return MagnitudeType::Square;
        }

        [[nodiscard]] constexpr MagnitudeType operator()(
            const hogpp::GradientSqrtMagnitude<T>& /*unused*/) const noexcept
        {
            return MagnitudeType::Sqrt;
        }
    };

    // clang-format off
    std::variant
    <
          hogpp::GradientMagnitude<T>
        , hogpp::GradientSquareMagnitude<T>
        , hogpp::GradientSqrtMagnitude<T>
    >
    voting_;
    // clang-format on
};

extern template class Magnitude<float>;
extern template class Magnitude<double>;

#include <hogpp/suffix.hpp>

#endif // PYTHON_HOGPP_MAGNITUDE_HPP
