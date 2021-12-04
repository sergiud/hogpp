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

#ifndef HOGPP_PRECISION_HPP
#define HOGPP_PRECISION_HPP

#include <type_traits>

#include <hogpp/promote.hpp>

namespace hogpp {

template<class T, class E = void>
struct MakePrecisionType
{
    using Scalar = T;
    using type = std::common_type_t<Promote_t<T>, T>;
};

template<class Tensor>
struct MakePrecisionType
    // clang-format off
<
      Tensor
    , std::void_t
    <
        typename Tensor::Scalar
    >
>
// clang-format on
{
    using Scalar = typename Tensor::Scalar;
    using type = std::common_type_t<Promote_t<Scalar>, Scalar>;
};

template<class T>
using PrecisionType_t = typename MakePrecisionType<T>::type;

} // namespace hogpp

#endif // HOGPP_PRECISION_HPP
