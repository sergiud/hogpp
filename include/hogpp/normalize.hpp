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

#ifndef HOGPP_NORMALIZE_HPP
#define HOGPP_NORMALIZE_HPP

#include <cmath>

namespace hogpp {

/**
 * @brief Normalizes a block by performing a safe division by the given
 * value.
 *
 * @param block[in,out] The block to be normalized.
 * @param den The normalization scalar. Unless the corresponding is exactly 0,
 * each coefficient of the block tensor is divided by this value.
 */
template<class Tensor>
constexpr void normalize(Tensor& block, typename Tensor::Scalar den) noexcept(
    noexcept(block = block / den))
{
    using std::fpclassify;

    // NOTE Due to rounding errors, block values can be negative but close to
    // zero.
    if (fpclassify(den) != FP_ZERO) {
        block = block / den;
    }
}

} // namespace hogpp

#endif // HOGPP_NORMALIZE_HPP
