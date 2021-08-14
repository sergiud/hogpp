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

#ifndef PYTHON_TYPE_CASTER_STRIDE_HPP
#define PYTHON_TYPE_CASTER_STRIDE_HPP

#include <tuple>
#include <type_traits>

template<class... Loops, class... Strides, std::size_t... Indices>
constexpr auto reduceIndices(const std::tuple<Loops...>& i,
                             const std::tuple<Strides...>& strides,
                             std::index_sequence<Indices...> /*unused*/)
{
    return ((std::get<Indices>(i) * std::get<Indices>(strides)) + ... + 0);
}

template<class... Loops, class... Strides>
constexpr auto reduceIndices(const std::tuple<Loops...>& i,
                             const std::tuple<Strides...>& strides)
{
    return reduceIndices(i, strides, std::index_sequence_for<Loops...>{});
}

#endif // PYTHON_TYPE_CASTER_STRIDE_HPP
