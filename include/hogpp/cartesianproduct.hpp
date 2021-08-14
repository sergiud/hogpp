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

#ifndef HOGPP_CARTESIANPRODUCT_HPP
#define HOGPP_CARTESIANPRODUCT_HPP

#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

namespace hogpp {

template<std::integral... Sizes, class Body, std::integral... Loops>
constexpr void cartesianProduct(
    [[maybe_unused]] const std::tuple<Sizes...>& s,
    const std::tuple<Loops...>& l, Body&& body,
    std::index_sequence<> /*unused*/) noexcept(noexcept(body(l)))
{
    body(l);
}

template<std::integral... Sizes, class Body, std::integral... Loops,
         std::size_t Index, std::size_t... Indices>
constexpr void cartesianProduct(
    const std::tuple<Sizes...>& s, const std::tuple<Loops...>& l, Body&& body,
    std::index_sequence<
        Index,
        Indices...> /*unused*/) noexcept(noexcept(body(std::declval<std::tuple<Sizes...>>())))
{
    for (auto i = 0; i < std::get<Index>(s); ++i) {
        cartesianProduct(s, std::tuple_cat(l, std::make_tuple(i)),
                         std::forward<Body>(body),
                         std::index_sequence<Indices...>{});
    }
}

template<std::integral... Sizes, class Body>
constexpr void cartesianProduct(
    const std::tuple<Sizes...>& s,
    Body&& body) noexcept(noexcept(body(std::declval<std::tuple<Sizes...>>())))
{
    cartesianProduct(s, std::tuple<>{}, std::forward<Body>(body),
                     std::index_sequence_for<Sizes...>{});
}

} // namespace hogpp

#endif // HOGPP_CARTESIANPRODUCT_HPP
