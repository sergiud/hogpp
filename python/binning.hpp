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

#ifndef PYTHON_HOGPP_BINNING_HPP
#define PYTHON_HOGPP_BINNING_HPP

// FIXME GCC 14.x workaround for https://github.com/pybind/pybind11/pull/5208
#include <algorithm>
#include <variant>

#include <unsupported/Eigen/CXX11/Tensor>

#include <pybind11/pybind11.h>

#include <hogpp/prefix.hpp>
#include <hogpp/signedgradient.hpp>
#include <hogpp/unsignedgradient.hpp>

namespace pyhogpp::inline HOGPP_TARGET {

enum class BinningType
{
    Signed,
    Unsigned
};

} // namespace pyhogpp::inline HOGPP_TARGET

namespace pybind11::detail {

template<>
class type_caster<pyhogpp::BinningType>
{
public:
    bool load(handle src, bool);
    static handle cast(pyhogpp::BinningType in, return_value_policy /*policy*/,
                       handle /*parent*/);

private:
    PYBIND11_TYPE_CASTER(pyhogpp::BinningType, _("Binning"));
};

} // namespace pybind11::detail

namespace pyhogpp::inline HOGPP_TARGET {

// The Fast binning profile is only wired in for the ISA-specific dispatch
// object libraries (see the non-generic branch in CMakeLists.txt, which
// defines HOGPP_FAST_MATH); the generic dispatch fallback and the
// plain non-dispatch build keep the exact Accurate profile.
#if defined(HOGPP_FAST_MATH)
using BinningProfile = hogpp::Fast;
#else
using BinningProfile = hogpp::Accurate;
#endif // defined(HOGPP_FAST_MATH)

template<class T>
class Binning
{
public:
    [[nodiscard]] explicit Binning(BinningType type = BinningType::Unsigned)
    {
        switch (type) {
            case BinningType::Signed:
                binning_ = hogpp::SignedGradient<T, BinningProfile>{};
                break;
            case BinningType::Unsigned:
                binning_ = hogpp::UnsignedGradient<T, BinningProfile>{};
                break;
        }
    }

    [[nodiscard]] constexpr T operator()(T dx, T dy) const
    {
        return std::visit([dx, dy](auto& binning) { return binning(dx, dy); },
                          binning_);
    }

#if defined(HOGPP_FAST_MATH)
    // Only meaningful when BinningProfile is Fast, which is the only
    // profile providing a tensor-expression operator() overload (see
    // SignedGradient<T, Fast> / UnsignedGradient<T, Fast>). std::visit
    // requires a common return type across variant alternatives, so the
    // lazy expression each alternative returns is materialized into a
    // concrete tensor here rather than left lazy.
    template<class Derived1, class Derived2>
    [[nodiscard]] decltype(auto) operator()(
        const Eigen::TensorBase<Derived1, Eigen::ReadOnlyAccessors>& dx,
        const Eigen::TensorBase<Derived2, Eigen::ReadOnlyAccessors>& dy) const
    {
        using ResultTensor =
            Eigen::Tensor<T, Derived1::NumDimensions, Derived1::Layout>;

        return std::visit(
            [&dx, &dy](const auto& binning) -> ResultTensor {
                return binning(dx, dy);
            },
            binning_);
    }
#endif // defined(HOGPP_FAST_MATH)

    [[nodiscard]] BinningType type() const noexcept
    {
        return std::visit(BinningVisitor{}, binning_);
    }

private:
    struct BinningVisitor
    {
        [[nodiscard]] constexpr BinningType operator()(
            const hogpp::SignedGradient<T, BinningProfile>& /*unused*/)
            const noexcept
        {
            return BinningType::Signed;
        }

        [[nodiscard]] constexpr BinningType operator()(
            const hogpp::UnsignedGradient<T, BinningProfile>& /*unused*/)
            const noexcept
        {
            return BinningType::Unsigned;
        }
    };

    // clang-format off
    std::variant
    <
          hogpp::SignedGradient<T, BinningProfile>
        , hogpp::UnsignedGradient<T, BinningProfile>
    >
    binning_;
    // clang-format on
};

extern template class Binning<float>;
extern template class Binning<double>;

} // namespace pyhogpp::inline HOGPP_TARGET

#include <hogpp/suffix.hpp>

#endif // PYTHON_HOGPP_BINNING_HPP
