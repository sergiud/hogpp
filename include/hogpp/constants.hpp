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

namespace hogpp::constants {

template<class T>
inline constexpr T pi = T(3.1415926535897932384626433832795028841971693993751);

template<class T>
inline constexpr T two_pi =
    T(6.28318530717958623199592693708837032318115234375);

template<class T>
inline constexpr T half_pi =
    T(1.5707963267948965579989817342720925807952880859375);

} // namespace hogpp::constants

#endif // HOGPP_CONSTANTS_HPP
