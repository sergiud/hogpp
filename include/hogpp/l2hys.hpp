//
// HOGpp - Fast histogram of oriented gradients computation using integral
// histograms
//
// Copyright 2020 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

#ifndef HOGPP_L2HYS_HPP
#define HOGPP_L2HYS_HPP

#include <unsupported/Eigen/CXX11/Tensor>

#include <type_traits>

namespace hogpp {

template<class T>
class L2Hys
{
public:
    using Scalar = T;

    [[nodiscard]] explicit L2Hys(
        Scalar clip = Scalar(0.2),
        Scalar regularization = std::numeric_limits<Scalar>::epsilon())
        : clip_{clip}
        , eps2_{regularization * regularization}
    {
    }

    template<class Tensor>
    void operator()(Tensor&& block) const
    {
        // L²-Hys block normalization
        // L² norm
        const Eigen::Tensor<Scalar, 0> v1 =
            (block.square().sum() + eps2_).sqrt();
        block = block / v1(0);
        // Clipping
        block = block.cwiseMin(clip_);
        // Renormalization
        const Eigen::Tensor<Scalar, 0> v2 =
            (block.square().sum() + eps2_).sqrt();
        block = block / v2(0);
    }

private:
    Scalar clip_;
    Scalar eps2_;
};

template<class T>
explicit L2Hys(T) -> L2Hys<T>;

template<class T1, class T2>
explicit L2Hys(T1, T2) -> L2Hys<std::common_type_t<T1, T2> >;

} // namespace hogpp

#endif // HOGPP_L2HYS_HPP
