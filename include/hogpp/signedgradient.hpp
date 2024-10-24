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

#ifndef HOGPP_SIGNEDGRADIENT_HPP
#define HOGPP_SIGNEDGRADIENT_HPP

#include <cmath>

#include <hogpp/constants.hpp>

namespace hogpp {

template<class Scalar>
struct SignedGradient
{
    [[nodiscard]] constexpr Scalar operator()(Scalar dx,
                                              Scalar dy) const noexcept
    {
        using std::atan2;
        Scalar angle = atan2(dy, dx);

        // Map [-π, +π) to [0, 1)
        return (angle + constants::pi<Scalar>) / constants::two_pi<Scalar>;
    }
};

} // namespace hogpp

#endif // HOGPP_SIGNEDGRADIENT_HPP
