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

#ifndef HOGPP_SIGNEDGRADIENT_HPP
#define HOGPP_SIGNEDGRADIENT_HPP

#include <cmath>

#include <unsupported/Eigen/CXX11/Tensor>

#include <hogpp/constants.hpp>
#include <hogpp/fastatan.hpp>

namespace hogpp {

struct Fast;
struct Accurate;

template<class Scalar, class Profile = Accurate>
struct SignedGradient;

template<class Scalar>
struct SignedGradient<Scalar, Accurate>
{
    [[nodiscard]] constexpr Scalar operator()(Scalar dx,
                                              Scalar dy) const noexcept
    {
        using std::atan2;
        Scalar angle = atan2(dy, dx);

        // Map [-π, +π) to [0, 1)
        return (angle + constants::pi<Scalar>) / constants::two_pi<Scalar>;
    }
};

/**
 * @brief Approximates @c atan2(dy, dx) with a branch-light minimax
 * polynomial instead of the exact libm call.
 *
 * Built on hogpp::detail::fastAtanUnit(), evaluated on the
 * smaller-magnitude ratio, avoiding the exact but non-vectorizable libm
 * @c atan2 call entirely. Maximum absolute angular error is
 * approximately @f$1.3312 \times 10^{-4}@f$ rad (the polynomial's
 * minimax bound), verified numerically against @c std::atan2 across the
 * full angular range. After mapping to the normalized [0, 1) bin weight
 * this is at most @f$2.12 \times 10^{-5}@f$, several orders of magnitude
 * below the width of a single bin in a typical histogram. A vote therefore
 * changes its lower bin only if the exact weight lies within this bound of
 * a bin boundary.
 */
template<class Scalar>
struct SignedGradient<Scalar, Fast>
{
    [[nodiscard]] constexpr Scalar operator()(Scalar dx,
                                              Scalar dy) const noexcept
    {
        using std::abs;
        using std::copysign;

        const Scalar absDx = abs(dx);
        const Scalar absDy = abs(dy);
        const Scalar largerMagnitude = absDx > absDy ? absDx : absDy;
        const Scalar smallerMagnitude = absDx > absDy ? absDy : absDx;
        const Scalar ratio = largerMagnitude > Scalar{0}
                                 ? smallerMagnitude / largerMagnitude
                                 : Scalar{0};

        Scalar angle = detail::fastAtanUnit(ratio);

        if (absDy > absDx) {
            angle = constants::half_pi<Scalar> - angle;
        }
        if (dx < Scalar{0}) {
            angle = constants::pi<Scalar> - angle;
        }
        angle = copysign(angle, dy);

        // Map [-π, +π) to [0, 1)
        return (angle + constants::pi<Scalar>) / constants::two_pi<Scalar>;
    }

    /**
     * @brief Evaluates the approximation on every lane of an Eigen packet.
     *
     * Mirrors the scalar overload branch for branch using lane masks, so
     * vectorized evaluations of the functor, such as by the tensor overload,
     * agree with the scalar one.
     */
    template<class Packet>
    [[nodiscard]] Packet packetOp(const Packet& dx,
                                  const Packet& dy) const noexcept
    {
        using Eigen::internal::pabs;
        using Eigen::internal::padd;
        using Eigen::internal::pand;
        using Eigen::internal::pcmp_lt;
        using Eigen::internal::pdiv;
        using Eigen::internal::por;
        using Eigen::internal::pselect;
        using Eigen::internal::pset1;
        using Eigen::internal::psub;
        using Eigen::internal::pzero;

        const Packet zero = pzero(dx);
        const Packet absDx = pabs(dx);
        const Packet absDy = pabs(dy);
        const Packet dyLarger = pcmp_lt(absDx, absDy);
        const Packet largerMagnitude = pselect(dyLarger, absDy, absDx);
        const Packet smallerMagnitude = pselect(dyLarger, absDx, absDy);
        const Packet ratio =
            pselect(pcmp_lt(zero, largerMagnitude),
                    pdiv(smallerMagnitude, largerMagnitude), zero);

        Packet angle = detail::fastAtanUnitPacket(ratio);
        angle = pselect(dyLarger,
                        psub(pset1<Packet>(constants::half_pi<Scalar>), angle),
                        angle);
        angle =
            pselect(pcmp_lt(dx, zero),
                    psub(pset1<Packet>(constants::pi<Scalar>), angle), angle);
        // The angle is not negative, so setting its sign bit to that of dy
        // computes copysign(angle, dy), including for dy = -0.
        angle = por(angle, pand(dy, pset1<Packet>(Scalar{-0.0})));

        // Map [-π, +π) to [0, 1)
        return pdiv(padd(angle, pset1<Packet>(constants::pi<Scalar>)),
                    pset1<Packet>(constants::two_pi<Scalar>));
    }

    /**
     * @brief Evaluates the approximation over whole tensors as a lazy
     * elementwise Eigen expression instead of once per pixel.
     *
     * hogpp::IntegralHOGDescriptor::compute() uses this, when available,
     * to precompute the bin weight for the whole image as a single
     * vectorized pass ahead of the histogram scan, which itself ends in
     * a data-dependent scatter write and can never be vectorized. The
     * expression applies packetOp() and evaluates every intermediate once
     * per packet. The Accurate profile deliberately lacks this overload,
     * so the generic dispatch fallback and the plain non-dispatch build
     * keep the exact per-pixel behavior.
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

//! Enables the vectorized evaluation of the Fast SignedGradient.
template<class Scalar>
struct functor_traits<hogpp::SignedGradient<Scalar, hogpp::Fast>>
{
    static constexpr int Divisions = 2;
    static constexpr int OtherOperations = 16;
    static constexpr int DivisionCost = scalar_div_cost<Scalar, true>::value;

    static constexpr int Cost =
        Divisions * DivisionCost + OtherOperations * NumTraits<Scalar>::AddCost;
    static constexpr bool PacketAccess = packet_traits<Scalar>::Vectorizable &&
                                         packet_traits<Scalar>::HasAbs &&
                                         packet_traits<Scalar>::HasDiv;
};

} // namespace Eigen::internal

#endif // HOGPP_SIGNEDGRADIENT_HPP
