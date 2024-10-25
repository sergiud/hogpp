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

#ifndef HOGPP_L1SQRT_HPP
#define HOGPP_L1SQRT_HPP

#include <unsupported/Eigen/CXX11/Tensor>

#include <type_traits>

#include <hogpp/l1norm.hpp>
#include <hogpp/normtraits.hpp>

namespace hogpp {

template<class T, class TraitsType = NormTraits<T>>
class L1Sqrt
{
public:
    using Scalar = T;
    using Traits = TraitsType;

    [[nodiscard]] constexpr explicit L1Sqrt(Scalar regularization = TraitsType::regularization()) noexcept(
        std::is_nothrow_constructible_v<L1Norm<Scalar, TraitsType>, Scalar>)
        : l1_{regularization}
    {
    }

    template<class Tensor>
    constexpr void operator()(Tensor& block) const
    {
        // L1-sqrt block normalization
        l1_(block);
        // Due to rounding errors, the L¹ block normalization can produce small
        // negative values. To avoid NaNs, clamp negative values to zero.
        block =
            (block < Scalar{0}).select(block.constant(Scalar{0}), block.sqrt());
    }

    [[nodiscard]] constexpr const L1Norm<Scalar>& norm() const noexcept
    {
        return l1_;
    }

private:
    L1Norm<Scalar, TraitsType> l1_;
};

template<class T>
explicit L1Sqrt(T) -> L1Sqrt<T>;

} // namespace hogpp

#endif // HOGPP_L1SQRT_HPP
