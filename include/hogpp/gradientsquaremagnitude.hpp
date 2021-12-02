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

#ifndef HOGPP_GRADIENTSQUAREMAGNITUDE_HPP
#define HOGPP_GRADIENTSQUAREMAGNITUDE_HPP

#include <unsupported/Eigen/CXX11/Tensor>

#include <cmath>

namespace hogpp {

struct Fast;
struct Accurate;

template<class T, class Profile = Fast>
struct GradientSquareMagnitude;

template<class T>
struct GradientSquareMagnitude<T, Fast>
{
    template<class Derived1, class Derived2>
    [[nodiscard]] constexpr decltype(auto) operator()(
        const Eigen::TensorBase<Derived1, Eigen::ReadOnlyAccessors>& dx,
        const Eigen::TensorBase<Derived2, Eigen::ReadOnlyAccessors>& dy) const
        noexcept(noexcept(dx.square() + dy.square()))
    {
        return dx.square() + dy.square();
    }
};

} // namespace hogpp

#endif // HOGPP_GRADIENTSQUAREMAGNITUDE_HPP
