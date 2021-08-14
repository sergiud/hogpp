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

#ifndef HOGPP_CONSTANTS_HPP
#define HOGPP_CONSTANTS_HPP

#include <numbers>

namespace hogpp::constants {

template<class T>
inline constexpr T pi = std::numbers::pi_v<T>;

template<class T>
inline constexpr T two_pi = 2 * pi<T>;

template<class T>
inline constexpr T half_pi = pi<T> / 2;

} // namespace hogpp::constants

#endif // HOGPP_CONSTANTS_HPP
