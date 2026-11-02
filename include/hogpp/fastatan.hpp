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

#ifndef HOGPP_FASTATAN_HPP
#define HOGPP_FASTATAN_HPP

#include <type_traits>

#include <Eigen/Core>

namespace hogpp::detail {

// Coefficients of the polynomial, rounded to double by scripts/remez_atan.py.
template<class T>
inline constexpr T fastAtanDegree3 = static_cast<T>(-0.3262382105724919);
template<class T>
inline constexpr T fastAtanDegree5 = static_cast<T>(0.1553161994609051);
template<class T>
inline constexpr T fastAtanDegree7 = static_cast<T>(-0.04381294810998689);

/**
 * @brief Degree-7 minimax polynomial approximation of
 * @f$\atan(\mathit{ratio})@f$ for @f$\mathit{ratio} \in [0, 1]@f$.
 *
 * Evaluated with plain multiply-add arithmetic only, deliberately
 * avoiding both the libm transcendental call and std::fma (not inlined
 * as a single instruction without -mfma). Intended for hot paths where
 * the branchy, non-vectorizable libm @c atan / @c atan2 implementation
 * would otherwise dominate the runtime regardless of the target
 * instruction set.
 *
 * The polynomial has the form
 * @f[
 *   \atan(x) \approx x + a x^3 + b x^5 + c x^7
 * @f]
 * with @f$a@f$, @f$b@f$, @f$c@f$ chosen as the minimax (equioscillating)
 * solution on @f$[0, 1]@f$, computed to 80 decimal digits with the Remez
 * exchange algorithm using mpmath and rounded to the nearest
 * double-precision value. The verified maximum absolute error over
 * @f$[0, 1]@f$ is approximately @f$1.3312 \times 10^{-4}@f$ rad.
 *
 * The coefficients were produced by the Remez exchange algorithm
 * implemented in scripts/remez_atan.py, which prints a maximum error of
 * @f$1.3312261902197300292 \times 10^{-4}@f$ and
 *
 * @code{.py}
 * a = -0.3262382105724919
 * b = 0.1553161994609051
 * c = -0.04381294810998689
 * @endcode
 */
template<class Scalar>
    requires std::is_floating_point_v<Scalar>
[[nodiscard]] constexpr Scalar fastAtanUnit(Scalar ratio) noexcept
{
    // Plain multiply-add rather than std::fma. On instruction sets without
    // FMA, std::fma calls into libm and reintroduces the non-vectorizable
    // call this approximation avoids. hogpp::multiplyAdd() selects the
    // fused operation where it is fast but is not constexpr. The rounding
    // difference from not fusing is negligible next to the approximation
    // error of the polynomial.
    constexpr Scalar degree3Coefficient = fastAtanDegree3<Scalar>;
    constexpr Scalar degree5Coefficient = fastAtanDegree5<Scalar>;
    constexpr Scalar degree7Coefficient = fastAtanDegree7<Scalar>;

    const Scalar ratioSquared = ratio * ratio;
    Scalar poly = (degree7Coefficient * ratioSquared) + degree5Coefficient;
    poly = (poly * ratioSquared) + degree3Coefficient;

    return (poly * ratioSquared * ratio) + ratio;
}

/**
 * @brief Packet overload of fastAtanUnit() evaluating the same polynomial on
 * every lane of an Eigen packet.
 *
 * Evaluates each term once per packet with the same separate multiplications
 * and additions as the scalar overload.
 */
template<class Packet>
[[nodiscard]] inline Packet fastAtanUnitPacket(const Packet& ratio) noexcept
{
    using Scalar = typename Eigen::internal::unpacket_traits<Packet>::type;
    using Eigen::internal::padd;
    using Eigen::internal::pmul;
    using Eigen::internal::pset1;

    const Packet ratioSquared = pmul(ratio, ratio);
    Packet poly =
        padd(pmul(pset1<Packet>(fastAtanDegree7<Scalar>), ratioSquared),
             pset1<Packet>(fastAtanDegree5<Scalar>));
    poly =
        padd(pmul(poly, ratioSquared), pset1<Packet>(fastAtanDegree3<Scalar>));

    return padd(pmul(pmul(poly, ratioSquared), ratio), ratio);
}

} // namespace hogpp::detail

#endif // HOGPP_FASTATAN_HPP
