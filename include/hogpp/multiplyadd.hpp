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

#ifndef HOGPP_MULTIPLYADD_HPP
#define HOGPP_MULTIPLYADD_HPP

#include <cmath>

namespace hogpp {

/**
 * @brief Determines whether the target performs a fused multiply-add of @p T
 * at least as fast as a separate multiplication and addition.
 *
 * Follows the @c FP_FAST_FMAF, @c FP_FAST_FMA and @c FP_FAST_FMAL macros of
 * @c <cmath> which compilers define if the instruction set provides the
 * operation, such as with @c -mfma.
 */
template<class T>
inline constexpr bool HasFastFma = false;

#if defined(FP_FAST_FMAF)
template<>
inline constexpr bool HasFastFma<float> = true;
#endif // defined(FP_FAST_FMAF)

#if defined(FP_FAST_FMA)
template<>
inline constexpr bool HasFastFma<double> = true;
#endif // defined(FP_FAST_FMA)

#if defined(FP_FAST_FMAL)
template<>
inline constexpr bool HasFastFma<long double> = true;
#endif // defined(FP_FAST_FMAL)

/**
 * @brief Computes @f$ab+c@f$, rounded once if the target provides a fast
 * fused multiply-add.
 *
 * Otherwise, @c std::fma is emulated in software at several times the cost of
 * a separate multiplication and addition, and prevents vectorization.
 */
template<class T>
[[nodiscard]] inline T multiplyAdd(T a, T b, T c) noexcept
{
    if constexpr (HasFastFma<T>) {
        using std::fma;
        return fma(a, b, c);
    }
    else {
        return a * b + c;
    }
}

} // namespace hogpp

#endif // HOGPP_MULTIPLYADD_HPP
