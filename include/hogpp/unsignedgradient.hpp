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
 * of a single bin in a typical histogram. A vote therefore changes its
 * lower bin only if the exact weight lies within this bound of a bin
 * boundary.
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
     * @brief Evaluates the approximation on every lane of an Eigen packet.
     *
     * Mirrors the scalar overload using lane masks, so vectorized
     * evaluations of the functor, such as by the tensor overload, agree
     * with the scalar one.
     */
    template<class Packet>
    [[nodiscard]] Packet packetOp(const Packet& dx,
                                  const Packet& dy) const noexcept
    {
        using Eigen::internal::pabs;
        using Eigen::internal::padd;
        using Eigen::internal::pand;
        using Eigen::internal::pcmp_eq;
        using Eigen::internal::pcmp_le;
        using Eigen::internal::pdiv;
        using Eigen::internal::por;
        using Eigen::internal::pselect;
        using Eigen::internal::pset1;
        using Eigen::internal::psub;
        using Eigen::internal::pzero;

        const Packet zero = pzero(dx);
        const Packet one = pset1<Packet>(Scalar{1});
        const Packet signMask = pset1<Packet>(Scalar{-0.0});
        const Packet halfPi = pset1<Packet>(constants::half_pi<Scalar>);
        const Packet dxIsZero = pcmp_eq(dx, zero);
        const Packet bothZero = pand(dxIsZero, pcmp_eq(dy, zero));

        // The lanes with dx = 0 divide by zero here. Their result is
        // replaced below.
        const Packet ratio = pdiv(dy, dx);
        const Packet absRatio = pabs(ratio);
        const Packet ratioSmall = pcmp_le(absRatio, one);
        // A single evaluation of the polynomial covers both branches of the
        // scalar overload, which differ only in its argument.
        const Packet unitAngle = detail::fastAtanUnitPacket(
            pselect(ratioSmall, absRatio, pdiv(one, absRatio)));
        const Packet magnitudeAngle =
            pselect(ratioSmall, unitAngle, psub(halfPi, unitAngle));
        // The angles are not negative, so setting their sign bit computes
        // copysign.
        const Packet generalAngle = por(magnitudeAngle, pand(ratio, signMask));
        const Packet dxZeroAngle = por(halfPi, pand(dy, signMask));
        const Packet angle = pselect(dxIsZero, dxZeroAngle, generalAngle);

        // Map [-π/2, +π/2) to [0, 1)
        const Packet weight =
            pdiv(padd(angle, halfPi), pset1<Packet>(constants::pi<Scalar>));

        return pselect(bothZero, pset1<Packet>(Scalar(0.5)), weight);
    }

    /**
     * @brief Evaluates the approximation over whole tensors as a lazy
     * elementwise Eigen expression instead of once per pixel.
     *
     * See the tensor overload of SignedGradient<Scalar, Fast>. The Accurate
     * profile deliberately lacks this overload and keeps the exact
     * per-pixel behavior.
     */
    template<class Derived1, class Derived2>
    [[nodiscard]] auto operator()(
        const Eigen::TensorBase<Derived1, Eigen::ReadOnlyAccessors>& dx,
        const Eigen::TensorBase<Derived2, Eigen::ReadOnlyAccessors>& dy) const
    {
        return static_cast<const Derived1&>(dx).binaryExpr(
            static_cast<const Derived2&>(dy), *this);
    }
};

} // namespace hogpp

namespace Eigen::internal {

//! Enables the vectorized evaluation of the Fast UnsignedGradient.
template<class Scalar>
struct functor_traits<hogpp::UnsignedGradient<Scalar, hogpp::Fast>>
{
    static constexpr int Divisions = 3;
    static constexpr int OtherOperations = 18;
    static constexpr int DivisionCost = scalar_div_cost<Scalar, true>::value;

    static constexpr int Cost =
        Divisions * DivisionCost + OtherOperations * NumTraits<Scalar>::AddCost;
    static constexpr bool PacketAccess = packet_traits<Scalar>::Vectorizable &&
                                         packet_traits<Scalar>::HasAbs &&
                                         packet_traits<Scalar>::HasDiv;
};

} // namespace Eigen::internal

#endif // HOGPP_UNSIGNEDGRADIENT_HPP
