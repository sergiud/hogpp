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

#ifndef PYTHON_TYPE_CASTER_CARTESIANPRODUCT_HPP
#define PYTHON_TYPE_CASTER_CARTESIANPRODUCT_HPP

#include <tuple>
#include <type_traits>
#include <utility>

template<class... Sizes, class Function, class... Loops>
constexpr void cartesianProduct(
    const std::tuple<Sizes...>& /*s*/, const std::tuple<Loops...>& l,
    Function&& func,
    std::index_sequence<> /*unused*/) noexcept(noexcept(func(l)))
{
    func(l);
}

template<class... Sizes, class Function, class... Loops, std::size_t Index,
         std::size_t... Indices>
constexpr void cartesianProduct(
    const std::tuple<Sizes...>& s, const std::tuple<Loops...>& l,
    Function&& func,
    std::index_sequence<
        Index,
        Indices...> /*unused*/) noexcept(noexcept(func(std::declval<std::tuple<Sizes...>>())))
{
    for (auto i = 0; i < std::get<Index>(s); ++i) {
        cartesianProduct(s, std::tuple_cat(l, std::make_tuple(i)),
                         std::forward<Function>(func),
                         std::index_sequence<Indices...>{});
    }
}

template<class... Sizes, class Function>
constexpr void
cartesianProduct(const std::tuple<Sizes...>& s, Function&& func) noexcept(
    noexcept(func(std::declval<std::tuple<Sizes...>>())))
{
    cartesianProduct(s, std::tuple<>{}, std::forward<Function>(func),
                     std::index_sequence_for<Sizes...>{});
}

#endif // PYTHON_TYPE_CASTER_CARTESIANPRODUCT_HPP
