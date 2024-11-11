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

#ifndef HOGPP_INTEGRALHISTOGRAM_HPP
#define HOGPP_INTEGRALHISTOGRAM_HPP

#include <bit>
#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

#include <unsupported/Eigen/CXX11/Tensor>

#include <hogpp/cartesianproduct.hpp>

namespace hogpp {

template<class T>
concept Arithmetic = std::is_arithmetic_v<T>;

namespace detail {

template<class... T>
concept TupleOfIntegrals = requires {
    std::tuple<T...>{};
    (std::is_integral_v<T> && ...);
};

template<std::size_t K, class... Types, std::size_t... Indices>
[[nodiscard]] constexpr decltype(auto) neighbor(
    const std::tuple<Types...>& i,
    std::index_sequence<Indices...> /*unused*/) noexcept
{
    return std::make_tuple(
        (std::get<Indices>(i) + (((1U << Indices) & K) != 0))...);
}

template<std::size_t K, class... Types>
[[nodiscard]] constexpr decltype(auto) neighbor(
    const std::tuple<Types...>& i) noexcept
{
    return neighbor<K>(i, std::index_sequence_for<Types...>{});
}

template<class Tensor, class... Types>
[[nodiscard]] constexpr decltype(auto) chip(
    Tensor&& t, [[maybe_unused]] const std::tuple<Types...>& i,
    std::integer_sequence<Eigen::DenseIndex> /*unused*/) noexcept
{
    return std::forward<Tensor>(t);
}

template<class Tensor, class... Types, Eigen::DenseIndex Index,
         Eigen::DenseIndex... Indices>
[[nodiscard]] constexpr auto chip(
    Tensor&& t, const std::tuple<Types...>& i,
    std::integer_sequence<Eigen::DenseIndex, Index,
                          Indices...> /*unused*/) noexcept
{
    return chip(t.template chip<0>(std::get<Index>(i)), i,
                std::integer_sequence<Eigen::DenseIndex, Indices...>{});
}

template<class Tensor, class... Types>
[[nodiscard]] constexpr decltype(auto) chip(
    Tensor&& t, const std::tuple<Types...>& i) noexcept
{
    return chip(std::forward<Tensor>(t), i,
                std::make_integer_sequence<Eigen::DenseIndex,
                                           static_cast<Eigen::DenseIndex>(
                                               sizeof...(Types))>{});
}

template<Eigen::DenseIndex Offset, class... Types, Eigen::DenseIndex... Indices>
[[nodiscard]] constexpr decltype(auto) offset(
    const std::tuple<Types...>& i,
    std::integer_sequence<Eigen::DenseIndex, Indices...> /*unused*/) noexcept
{
    return std::make_tuple((std::get<Indices>(i) + Offset)...);
}

template<Eigen::DenseIndex Offset, class... Types>
[[nodiscard]] constexpr decltype(auto) offset(
    const std::tuple<Types...>& i) noexcept
{
    return offset<Offset>(
        i, std::make_integer_sequence<Eigen::DenseIndex,
                                      static_cast<Eigen::DenseIndex>(
                                          sizeof...(Types))>{});
}

template<bool Sign, class Tensor, std::enable_if_t<Sign>* = nullptr>
[[nodiscard]] constexpr decltype(auto) negate(Tensor&& t) noexcept(noexcept(-t))
{
    return -t;
}

template<bool Sign, class Tensor, std::enable_if_t<!Sign>* = nullptr>
[[nodiscard]] constexpr decltype(auto) negate(Tensor&& t) noexcept
{
    return std::forward<Tensor>(t);
}

template<class Tensor, class... Types, std::size_t... Combinations>
[[nodiscard]] constexpr decltype(auto) propagate(
    Tensor&& t, const std::tuple<Types...>& i,
    std::index_sequence<Combinations...> /*unused*/) noexcept
{
    return (negate<Combinations == 0>(
                chip(std::forward<Tensor>(t), neighbor<Combinations>(i))) +
            ...);
}

template<class Tensor, class... Types>
[[nodiscard]] constexpr decltype(auto) propagate(
    Tensor&& t, const std::tuple<Types...>& i) noexcept
{
    constexpr std::size_t N = (1U << sizeof...(Types)) - 1U;
    return propagate(std::forward<Tensor>(t), i, std::make_index_sequence<N>{});
}

template<bool Sign, class... Types1, class... Types2,
         std::enable_if_t<Sign>* = nullptr>
[[nodiscard]] constexpr decltype(auto) select(
    const std::tuple<Types1...>& a,
    [[maybe_unused]] const std::tuple<Types2...>& b)
{
    return a;
}

template<bool Sign, class... Types1, class... Types2,
         std::enable_if_t<!Sign>* = nullptr>
[[nodiscard]] constexpr decltype(auto) select(
    [[maybe_unused]] const std::tuple<Types1...>& a,
    const std::tuple<Types2...>& b)
{
    return b;
}

template<std::size_t K, std::integral... Types1, std::integral... Types2,
         std::size_t... Indices>
[[nodiscard]] constexpr decltype(auto) cross1(
    const std::tuple<Types1...>& a, const std::tuple<Types2...>& b,
    std::index_sequence<Indices...> /*unused*/) noexcept
{
    return std::make_tuple(
        std::get<Indices>(select<(((1U << Indices) & K) != 0)>(a, b))...);
}

template<std::integral... Types1, std::integral... Types2,
         std::size_t... Combinations>
[[nodiscard]] constexpr decltype(auto) combinations(
    const std::tuple<Types1...>& a, const std::tuple<Types2...>& b,
    std::index_sequence<Combinations...> /*unused*/) noexcept
{
    static_assert(sizeof...(Types1) == sizeof...(Types2),
                  "combinations tuple size does not match");

    return std::make_tuple(
        cross1<Combinations>(a, b, std::index_sequence_for<Types1...>{})...);
}

template<std::integral... Types1, std::integral... Types2>
[[nodiscard]] constexpr decltype(auto) combinations(
    const std::tuple<Types1...>& a, const std::tuple<Types2...>& b) noexcept
{
    static_assert(sizeof...(Types1) == sizeof...(Types2),
                  "combinations tuple size does not match");

    constexpr std::size_t N = (1U << sizeof...(Types1));
    return combinations(a, b, std::make_index_sequence<N>{});
}

template<std::size_t K, class Tensor>
[[nodiscard]] constexpr decltype(auto) sign(Tensor&& t)
{
    // Cannot use std::bitset::count because it's not constexpr.
    constexpr std::size_t N = std::popcount(K);
    // Bit set: + otherwise -
    // If the number of set bits does not match the number of total bits, we
    // need to negate the term.
    return negate<(N & 1) != 0>(t);
}

template<class Tensor, class... Types, std::size_t... Indices>
[[nodiscard]] constexpr decltype(auto) intersect(
    Tensor&& t, const std::tuple<Types...>& ab,
    std::index_sequence<Indices...> /*unused*/)
{
    return (
        sign<Indices>(chip(std::forward<Tensor>(t), std::get<Indices>(ab))) +
        ...);
}

template<class Tensor, TupleOfIntegrals... Types>
[[nodiscard]] constexpr decltype(auto) intersect(Tensor&& t,
                                                 const std::tuple<Types...>& ab)
{
    return intersect(std::forward<Tensor>(t), ab,
                     std::index_sequence_for<Types...>{});
}

template<class Tensor, std::integral... Types1, std::integral... Types2>
[[nodiscard]] constexpr decltype(auto) intersect(Tensor&& t,
                                                 const std::tuple<Types1...>& a,
                                                 const std::tuple<Types2...>& b)
{
    return intersect(std::forward<Tensor>(t), combinations(a, b));
}

} // namespace detail

template<Arithmetic T, Eigen::DenseIndex N, Eigen::DenseIndex K = 1,
         int Options = 0>
class IntegralHistogram
{
public:
    static_assert(
        N > 0,
        "IntegralHistogram number of input space dimensions must be positive");
    static_assert(K > 0,
                  "IntegralHistogram number of histogram space dimensions must "
                  "be positive");

    using Tensor = Eigen::Tensor<T, N + K, Options>;

    template<std::integral... InputDims, std::integral... BinDims>
    void resize(const std::tuple<InputDims...>& dims,
                const std::tuple<BinDims...>& bins)
    {
        static_assert(sizeof...(InputDims) == N,
                      "IntegralHistogram resize number of provided input space "
                      "dimensions does not match the actual number of input "
                      "space dimensions");
        static_assert(
            sizeof...(BinDims) == K,
            "IntegralHistogram resize number of provided histogram space "
            "dimensions does not match the actual number of histogram "
            "space dimensions");

        resize(std::tuple_cat(detail::offset<1>(dims), bins));
    }

    template<std::integral... InputDims, std::integral Bins,
             std::enable_if_t<K == 1>* = nullptr>
    void resize(const std::tuple<InputDims...>& dims, Bins bins)
    {
        resize(dims, std::make_tuple(bins));
    }

    // TODO Tensor of compatible dimension
    template<class Binning>
    void scan(Binning binning) // requires std::invocable<Binning,
    {
        scan(histogram_.dimensions(),
             std::make_integer_sequence<Eigen::DenseIndex, N>{},
             std::move(binning));
    }

    template<std::integral... Types1, std::integral... Types2>
    [[nodiscard]] decltype(auto) intersect(const std::tuple<Types1...>& a,
                                           const std::tuple<Types2...>& b) const
        noexcept(noexcept(detail::intersect(std::declval<Tensor>(), a, b)))
    {
        static_assert(sizeof...(Types1) == sizeof...(Types2),
                      "IntegralHistogram intersect coordinates dimensionality "
                      "does not match");
        return detail::intersect(histogram_, a, b);
    }

    [[nodiscard]] const Tensor& histogram() const noexcept
    {
        return histogram_;
    }

    template<class Derived>
    void setHistogram(
        const Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors>& value)
    {
        histogram_ = value.eval();
    }

    [[nodiscard]] bool isEmpty() const noexcept
    {
        return histogram_.size() == 0;
    }

private:
    template<Eigen::DenseIndex... Indices, class Binning>
    void scan(
        const std::array<Eigen::DenseIndex, static_cast<std::size_t>(N + K)>&
            dims,
        std::integer_sequence<Eigen::DenseIndex, Indices...> /*unused*/,
        Binning
            binning) // noexcept(noexcept(binning(std::declval<Eigen::Tensor<T,
                     // N> >()
    {
        histogram_.setZero();

        // Wavefront scan
        cartesianProduct(std::make_tuple((dims[Indices] - 1)...),
                         [this, &binning](const auto& i) constexpr {
                             auto current = detail::offset<1>(i);
                             auto&& H = detail::chip(histogram_, current);
                             H = detail::propagate(histogram_, i);
                             binning(H, i);
                         });
    }

    template<std::integral... Types>
    void resize(const std::tuple<Types...>& dims)
    {
        resize(dims, std::make_integer_sequence<Eigen::DenseIndex,
                                                static_cast<Eigen::DenseIndex>(
                                                    sizeof...(Types))>{});
    }

    template<std::integral... Types, Eigen::DenseIndex... Indices>
    void resize(const std::tuple<Types...>& dims,
                std::integer_sequence<Eigen::DenseIndex, Indices...> /*unused*/)
    {
        histogram_.resize(std::get<Indices>(dims)...);
    }

    Tensor histogram_;
};

} // namespace hogpp

#endif // HOGPP_INTEGRALHISTOGRAM_HPP
