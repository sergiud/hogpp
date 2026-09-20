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
 * below the width of a single bin in a typical histogram, so bin
 * assignment is unaffected.
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
};

} // namespace hogpp

#endif // HOGPP_SIGNEDGRADIENT_HPP
