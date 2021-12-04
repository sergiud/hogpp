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

#ifndef HOGPP_AXIS_HPP
#define HOGPP_AXIS_HPP

#include <Eigen/Core>

#include <type_traits>

namespace hogpp {

template<Eigen::DenseIndex Index>
struct Axis_t : std::integral_constant<Eigen::DenseIndex, Index>
{
};

using Vertical_t = Axis_t<0>;
using Horizontal_t = Axis_t<1>;
using NoAxis_t = Axis_t<-1>;

} // namespace hogpp

#endif // HOGPP_AXIS_HPP
