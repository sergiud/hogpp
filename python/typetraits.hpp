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

#ifndef PYTHON_HOGPP_TYPETRAITS_HPP
#define PYTHON_HOGPP_TYPETRAITS_HPP

#include <type_traits>

template<class Norm, class Enable = void>
struct HasClip : std::false_type
{
};

// clang-format off
template<class Norm>
struct HasClip
<
      Norm
    , std::void_t<decltype(std::declval<Norm>().clip())>
>
: std::true_type
// clang-format on
{
};

template<class Norm>
constexpr bool HasClip_v = HasClip<Norm>::value;

template<class Norm, class Enable = void>
struct HasNorm : std::false_type
{
};

// clang-format off
template<class Norm>
struct HasNorm
<
      Norm
    , std::void_t<decltype(std::declval<Norm>().norm())>
>
: std::true_type
// clang-format on
{
};

template<class Norm>
constexpr bool HasNorm_v = HasNorm<Norm>::value;

template<class Norm, class Enable = void>
struct HasRegularization : std::false_type
{
};

// clang-format off
template<class Norm>
struct HasRegularization
<
      Norm
    , std::void_t<decltype(std::declval<Norm>().regularization())>
>
: std::true_type
// clang-format on
{
};

template<class Norm>
constexpr bool HasRegularization_v = HasRegularization<Norm>::value;

#endif // PYTHON_HOGPP_TYPETRAITS_HPP
