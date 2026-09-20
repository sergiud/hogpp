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

#ifndef HOGPP_UNSIGNEDGRADIENT_HPP
#define HOGPP_UNSIGNEDGRADIENT_HPP

#include <cmath>

#include <unsupported/Eigen/CXX11/Tensor>

#include <hogpp/constants.hpp>
#include <hogpp/fastatan.hpp>

namespace hogpp {

struct Fast;
struct Accurate;

template<class Scalar, class Profile = Accurate>
struct UnsignedGradient;

template<class Scalar>
struct UnsignedGradient<Scalar, Accurate>
{
    [[nodiscard]] constexpr Scalar operator()(Scalar dx,
                                              Scalar dy) const noexcept
    {
        using std::atan;
        using std::copysign;
        using std::fpclassify;

        Scalar angle = fpclassify(dx) == FP_ZERO && fpclassify(dy) == FP_ZERO
                           ? 0
                       : fpclassify(dx) != FP_ZERO
                           ? atan(dy / dx)
                           : copysign(constants::half_pi<Scalar>, dy);

        // Map [-π/2, +π/2) to [0, 1)
        return (angle + constants::half_pi<Scalar>) / constants::pi<Scalar>;
    }
};

/**
 * @brief Approximates @c atan(dy / dx) with a branch-light minimax
 * polynomial instead of the exact libm call.
 *
 * Built on hogpp::detail::fastAtanUnit(), evaluated on the
 * smaller-magnitude ratio, avoiding the exact but non-vectorizable libm
 * @c atan call entirely. Maximum absolute angular error is approximately
 * @f$1.3312 \times 10^{-4}@f$ rad (the polynomial's minimax bound),
 * verified numerically against @c std::atan across the full domain.
 * After mapping to the normalized [0, 1) bin weight this is at most
 * @f$4.24 \times 10^{-5}@f$, several orders of magnitude below the width
 * of a single bin in a typical histogram, so bin assignment is
 * unaffected.
 */
template<class Scalar>
struct UnsignedGradient<Scalar, Fast>
{
    [[nodiscard]] constexpr Scalar operator()(Scalar dx,
                                              Scalar dy) const noexcept
    {
        using std::abs;
        using std::copysign;
        using std::fpclassify;

        if (fpclassify(dx) == FP_ZERO && fpclassify(dy) == FP_ZERO) {
            return Scalar(0.5);
        }

        if (fpclassify(dx) == FP_ZERO) {
            return (copysign(constants::half_pi<Scalar>, dy) +
                    constants::half_pi<Scalar>) /
                   constants::pi<Scalar>;
        }

        const Scalar ratio = dy / dx;
        const Scalar absRatio = abs(ratio);

        Scalar angle;

        if (absRatio <= Scalar{1}) {
            angle = copysign(detail::fastAtanUnit(absRatio), ratio);
        }
        else {
            const Scalar reciprocal = Scalar{1} / absRatio;
            angle = copysign(
                constants::half_pi<Scalar> - detail::fastAtanUnit(reciprocal),
                ratio);
        }

        // Map [-π/2, +π/2) to [0, 1)
        return (angle + constants::half_pi<Scalar>) / constants::pi<Scalar>;
    }

    /**
     * @brief Tensor-expression overload evaluating the same approximation
     * as a lazy, elementwise Eigen expression instead of once per pixel.
     *
     * See SignedGradient<Scalar, Fast>'s tensor overload: this is not
     * provided for the Accurate profile, which keeps the exact,
     * unbatched per-pixel behavior.
     */
    template<class Derived1, class Derived2>
    [[nodiscard]] constexpr decltype(auto) operator()(
        const Eigen::TensorBase<Derived1, Eigen::ReadOnlyAccessors>& dx,
        const Eigen::TensorBase<Derived2, Eigen::ReadOnlyAccessors>& dy) const
    {
        const auto& dxDerived = dx.derived();
        const auto& dyDerived = dy.derived();

        const auto dxIsZero = dxDerived == dxDerived.constant(Scalar{0});
        const auto bothZero =
            dxIsZero && (dyDerived == dyDerived.constant(Scalar{0}));

        // At dx == 0 lanes the general-case expression below is discarded
        // by select() regardless of its value, so dividing by exactly
        // zero there (producing +-inf) is safe.
        const auto ratio = dyDerived / dxDerived;
        const auto absRatio = ratio.abs();
        const auto ratioSmall = absRatio <= absRatio.constant(Scalar{1});
        const auto reciprocal = absRatio.constant(Scalar{1}) / absRatio;

        const auto magnitudeAngle = ratioSmall.select(
            detail::fastAtanUnit(absRatio),
            absRatio.constant(constants::half_pi<Scalar>) -
                detail::fastAtanUnit(reciprocal));
        const auto generalAngle =
            (ratio < ratio.constant(Scalar{0}))
                .select(-magnitudeAngle, magnitudeAngle);

        const auto dxZeroAngle =
            (dyDerived < dyDerived.constant(Scalar{0}))
                .select(dyDerived.constant(-constants::half_pi<Scalar>),
                        dyDerived.constant(constants::half_pi<Scalar>));

        const auto angle = bothZero.select(
            generalAngle.constant(Scalar{0}),
            dxIsZero.select(dxZeroAngle, generalAngle));

        // Map [-π/2, +π/2) to [0, 1)
        return (angle + angle.constant(constants::half_pi<Scalar>)) /
               angle.constant(constants::pi<Scalar>);
    }
};

} // namespace hogpp

#endif // HOGPP_UNSIGNEDGRADIENT_HPP
