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

#ifndef HOGPP_UNSIGNEDGRADIENT_HPP
#define HOGPP_UNSIGNEDGRADIENT_HPP

#include <cmath>

#include <hogpp/constants.hpp>

namespace hogpp {

template<class Scalar>
struct UnsignedGradient
{
    [[nodiscard]] constexpr Scalar operator()(Scalar dx,
                                              Scalar dy) const noexcept
    {
        using std::atan;
        using std::copysign;
        using std::fpclassify;

        Scalar angle = fpclassify(dx) == FP_ZERO && fpclassify(dy) == FP_ZERO
                           ? 0
                       : fpclassify(dx) != FP_ZERO
                           ? atan(dy / dx)
                           : copysign(constants::half_pi<Scalar>, dy);

        // Map [-π/2, +π/2) to [0, 1)
        return (angle + constants::half_pi<Scalar>) / constants::pi<Scalar>;
    }
};

} // namespace hogpp

#endif // HOGPP_UNSIGNEDGRADIENT_HPP
