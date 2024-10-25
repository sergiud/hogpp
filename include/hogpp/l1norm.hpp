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

#ifndef HOGPP_L1NORM_HPP
#define HOGPP_L1NORM_HPP

#include <unsupported/Eigen/CXX11/Tensor>

#include <type_traits>

#include <hogpp/normalize.hpp>
#include <hogpp/normtraits.hpp>

namespace hogpp {

template<class T, class TraitsType = NormTraits<T>>
class L1Norm
{
public:
    using Scalar = T;
    using Traits = TraitsType;

    [[nodiscard]] constexpr explicit L1Norm(
        Scalar regularization =
            TraitsType::regularization()) noexcept(std::is_nothrow_constructible_v<Scalar,
                                                                                   Scalar>)
        : eps_{regularization}
    {
    }

    template<class Tensor>
    constexpr void operator()(Tensor& block) const
    {
        // L² norm
        const Eigen::Tensor<Scalar, 0> v = block.abs().sum() + eps_;
        normalize(block, v(0));
    }

    [[nodiscard]] constexpr T regularization() const noexcept
    {
        return eps_;
    }

private:
    Scalar eps_;
};

template<class T>
explicit L1Norm(T) -> L1Norm<T>;

} // namespace hogpp

#endif // HOGPP_L1NORM_HPP
