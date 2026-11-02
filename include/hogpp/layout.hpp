//
// HOGpp - Fast histogram of oriented gradients computation using integral
// histograms
//
// Copyright 2021 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

#ifndef HOGPP_LAYOUT_HPP
#define HOGPP_LAYOUT_HPP

#include <Eigen/Core>

namespace hogpp {

/**
 * @brief Determines whether @p Tensor stores its elements in column-major
 * order.
 *
 * Eigen 3.4 declares the layout of a tensor as an unnamed enumerator, whose
 * comparison with Eigen::StorageOptions compares different enumeration types.
 * Both are therefore compared as integers.
 */
template<class Tensor>
inline constexpr bool IsColMajor =
    static_cast<int>(Tensor::Layout) == static_cast<int>(Eigen::ColMajor);

} // namespace hogpp

#endif // HOGPP_LAYOUT_HPP
