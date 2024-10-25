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

#ifndef HOGPP_L1HYS_HPP
#define HOGPP_L1HYS_HPP

#include <type_traits>

#include <hogpp/l1norm.hpp>
#include <hogpp/normtraits.hpp>

namespace hogpp {

template<class T, class TraitsType = NormTraits<T>>
class L1Hys
{
public:
    using Scalar = T;
    using Traits = TraitsType;

    [[nodiscard]] constexpr explicit L1Hys(
        Scalar clip = TraitsType::clip(),
        Scalar regularization =
            TraitsType::regularization()) noexcept(std::is_nothrow_constructible_v<Scalar,
                                                                                   Scalar> &&
                                                   std::is_nothrow_constructible_v<
                                                       L1Norm<Scalar,
                                                              TraitsType>>)
        : clip_{clip}
        , l1_{regularization}
    {
    }

    template<class Tensor>
    constexpr void operator()(Tensor& block) const
    {
        // L¹-Hys block normalization
        // L¹ norm
        l1_(block);
        // Clipping
        block = block.cwiseMin(clip_);
        // Renormalization
        l1_(block);
    }

    [[nodiscard]] constexpr T clip() const noexcept
    {
        return clip_;
    }

    [[nodiscard]] constexpr const L1Norm<Scalar>& norm() const noexcept
    {
        return l1_;
    }

private:
    Scalar clip_;
    L1Norm<Scalar, TraitsType> l1_;
};

template<class T>
explicit L1Hys(T) -> L1Hys<T>;

template<class T1, class T2>
explicit L1Hys(T1, T2) -> L1Hys<std::common_type_t<T1, T2>>;

} // namespace hogpp

#endif // HOGPP_L1HYS_HPP
