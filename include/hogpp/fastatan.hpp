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

#include <unsupported/Eigen/CXX11/Tensor>

namespace hogpp::detail {

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
    // Plain multiply-add rather than std::fma: without -mfma (e.g. the
    // AVX2 dispatch object, which this project builds without requiring
    // FMA3 support), std::fma is not inlined as a single hardware
    // instruction and instead calls into libm, reintroducing exactly the
    // non-vectorizable call overhead this approximation exists to avoid.
    // The sub-ULP rounding difference from not fusing is negligible next
    // to the polynomial's own approximation error.
    constexpr Scalar degree3Coefficient{-0.3262382105724919};
    constexpr Scalar degree5Coefficient{0.1553161994609051};
    constexpr Scalar degree7Coefficient{-0.04381294810998689};

    const Scalar ratioSquared = ratio * ratio;
    Scalar poly = (degree7Coefficient * ratioSquared) + degree5Coefficient;
    poly = (poly * ratioSquared) + degree3Coefficient;

    return (poly * ratioSquared * ratio) + ratio;
}

/**
 * @brief Tensor-expression overload of fastAtanUnit(), evaluating the same
 * minimax polynomial as a lazy, elementwise Eigen expression.
 *
 * Unlike the scalar overload, this is meant to be evaluated over a whole
 * tensor at once (e.g. the full image) rather than once per pixel inside
 * a scalar hot loop, so Eigen can pack it into the target ISA's native
 * vector width instead of being forced through scalar code regardless of
 * the compiled instruction set.
 */
template<class Derived>
[[nodiscard]] constexpr decltype(auto) fastAtanUnit(
    const Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors>& ratio)
{
    using Scalar = typename Derived::Scalar;

    constexpr Scalar degree3Coefficient{-0.3262382105724919};
    constexpr Scalar degree5Coefficient{0.1553161994609051};
    constexpr Scalar degree7Coefficient{-0.04381294810998689};

    const auto ratioSquared = ratio.square();

    // Eigen's Tensor arithmetic operators only re-derive the concrete
    // expression type of the object a method is called on, not of a
    // same-typed argument passed to it: ratio's declared parameter type
    // is the TensorBase reference itself, so using it bare as the
    // right-hand side of an operator below (rather than as the object a
    // method is called on, as above) would embed that reference type
    // into the expression tree instead of the concrete one, and fail to
    // compile. Calling a no-op left-hand operation on it first, as done
    // throughout this codebase (e.g. GradientSquareMagnitude), recovers
    // the concrete type.
    const auto ratioIdentity = ratio + ratio.constant(Scalar{0});

    return (((ratioSquared * ratioSquared.constant(degree7Coefficient)) +
             ratioSquared.constant(degree5Coefficient)) *
                ratioSquared +
            ratioSquared.constant(degree3Coefficient)) *
               ratioSquared * ratioIdentity +
           ratioIdentity;
}

} // namespace hogpp::detail

#endif // HOGPP_FASTATAN_HPP
