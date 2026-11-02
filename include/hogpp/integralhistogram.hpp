//
// HOGpp - Fast histogram of oriented gradients computation using integral
// histograms
//
// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
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

template<class T>
struct IsTupleOfIntegrals : std::false_type
{
};

template<class... T>
struct IsTupleOfIntegrals<std::tuple<T...>>
    : std::bool_constant<(std::is_integral_v<T> && ...)>
{
};

// Satisfied by a std::tuple whose own elements are all integral, e.g. a
// single corner's per-axis coordinates. Used to constrain each element
// of an outer tuple of corners (a tuple of such coordinate tuples).
template<class T>
concept TupleOfIntegrals = IsTupleOfIntegrals<std::decay_t<T>>::value;

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

// Inclusion-exclusion sign for the corner reached by offsetting the
// dimensions selected by the K bit mask, while building the wavefront
// recurrence over a D-dimensional input space: (-1)^(D - 1 +
// popcount(K)). Negating only the zero-offset corner (as if D were
// always 2) is wrong for D == 1 or D >= 3.
template<std::size_t D>
[[nodiscard]] constexpr bool cornerSignbit(std::size_t k) noexcept
{
    return (D + static_cast<std::size_t>(std::popcount(k))) % 2 == 0;
}

template<class Tensor, class... Types, std::size_t... Combinations>
[[nodiscard]] constexpr decltype(auto) propagate(
    Tensor&& t, const std::tuple<Types...>& i,
    std::index_sequence<Combinations...> /*unused*/) noexcept
{
    constexpr std::size_t D = sizeof...(Types);
    return (negate<cornerSignbit<D>(Combinations)>(
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

template<class... Types, std::size_t... Indices>
[[nodiscard]] constexpr auto reverse(
    const std::tuple<Types...>& t,
    std::index_sequence<Indices...> /*unused*/) noexcept
{
    return std::make_tuple(std::get<sizeof...(Types) - 1 - Indices>(t)...);
}

template<class... Types>
[[nodiscard]] constexpr auto reverse(const std::tuple<Types...>& t) noexcept
{
    return reverse(t, std::index_sequence_for<Types...>{});
}

// Histogram space elements of a single input space position. They are evenly
// spaced in memory: a column-major histogram stores the histogram space
// dimensions last, so consecutive elements lie as far apart as the input space
// has positions, while a row-major histogram stores them contiguously. The
// elements follow the order of the linear indices of the corresponding chip.
template<class T>
struct StridedBins
{
    T* data;
    Eigen::DenseIndex stride;
    Eigen::DenseIndex size;
};

template<class Tensor, class... Types>
[[nodiscard]] auto stridedBins(Tensor& t,
                               const std::tuple<Types...>& i) noexcept
{
    using Scalar = std::remove_reference_t<decltype(*t.data())>;

    constexpr std::size_t N = sizeof...(Types);
    constexpr auto Rank = static_cast<std::size_t>(Tensor::NumIndices);

    std::array<Eigen::DenseIndex, Rank> index{};

    std::apply(
        [&index](auto... j) {
            std::size_t d = 0;
            ((index[d++] = static_cast<Eigen::DenseIndex>(j)), ...);
        },
        i);

    Eigen::DenseIndex positions = 1;
    Eigen::DenseIndex size = 1;

    for (std::size_t d = 0; d < N; ++d) {
        positions *= t.dimension(d);
    }

    for (std::size_t d = N; d < Rank; ++d) {
        size *= t.dimension(d);
    }

    const Eigen::DenseIndex stride =
        Tensor::Layout == Eigen::ColMajor ? positions : 1;

    return StridedBins<Scalar>{&t(index), stride, size};
}

// Element-wise counterpart of propagate() that writes the histogram space
// elements of the position offset by one along every input space dimension
// contiguously to bins. The right fold adds the corners in the same order as
// the tensor expression, so both round identically.
//
// The histogram space elements of a column-major histogram are strided, and
// there are typically few of them, e.g., 9 orientation bins. Evaluating
// propagate() as a vectorized tensor expression for every position gathers
// each packet one coefficient at a time through a call to the tensor executor,
// which made every SIMD instruction set slower than scalar code.
template<class Tensor, class... Types, std::size_t... Combinations>
void propagateBins(const Tensor& t, const std::tuple<Types...>& i,
                   typename Tensor::Scalar* bins,
                   std::index_sequence<Combinations...> /*unused*/) noexcept
{
    using Scalar = typename Tensor::Scalar;

    constexpr std::size_t D = sizeof...(Types);

    const StridedBins<const Scalar> origin = stridedBins(t, i);
    const std::array<const Scalar*, sizeof...(Combinations)> corners{
        stridedBins(t, neighbor<Combinations>(i)).data...};

    for (Eigen::DenseIndex m = 0; m < origin.size; ++m) {
        const Eigen::DenseIndex position = m * origin.stride;
        bins[m] = (negate<cornerSignbit<D>(Combinations)>(
                       corners[Combinations][position]) +
                   ...);
    }
}

template<class Tensor, class... Types>
void propagateBins(const Tensor& t, const std::tuple<Types...>& i,
                   typename Tensor::Scalar* bins) noexcept
{
    constexpr std::size_t N = (1U << sizeof...(Types)) - 1U;
    propagateBins(t, i, bins, std::make_index_sequence<N>{});
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

    using Scalar = T;
    using Tensor = Eigen::Tensor<T, N + K, Options>;
    //! Histogram space elements of a single input space position.
    using Bins = Eigen::TensorMap<Eigen::Tensor<T, K, Options>>;

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

    /**
     * @brief Builds the integral histogram from the votes of each input space
     * position.
     *
     * Invokes @p binning once for every input space position @c i as
     * @c binning(bins, i), where @c bins is a view of type Bins of the
     * histogram space elements of that position. On entry, the view holds the
     * sums over all positions @f$p \le i@f$ componentwise except
     * @c i itself. The votes of @c i are added to it in place. The order in
     * which the positions are visited follows the memory layout of the
     * histogram and is otherwise unspecified.
     *
     * The view refers to a buffer that is reused for every position, so it
     * must not be retained beyond the invocation. Taking it as Bins rather
     * than as an @c Eigen::TensorRef avoids an allocation for every position.
     */
    // TODO Tensor of compatible dimension
    template<class Binning>
    void scan(Binning binning) // requires std::invocable<Binning,
    {
        scan(histogram_.dimensions(),
             std::make_integer_sequence<Eigen::DenseIndex, N>{},
             std::move(binning));
    }

    /**
     * @brief Computes the second-order integral histogram.
     *
     * Applies the summed-area table recurrence to the first-order integral
     * histogram built by scan(). The result has one additional slot along
     * each input space dimension to retain the complete first-order sums at
     * the upper domain boundary which interpolate() requires to extrapolate
     * beyond the domain.
     *
     * The magnitude of the second-order integral histogram grows
     * quadratically with the input domain volume while its queries subtract
     * nearly equal values. A narrow @p T such as `float` therefore requires
     * a wider @p U to avoid catastrophic cancellation.
     *
     * @tparam U Arithmetic type of the second-order integral histogram.
     */
    template<Arithmetic U = T>
    [[nodiscard]] IntegralHistogram<U, N, K, Options> cascade() const
    {
        IntegralHistogram<U, N, K, Options> result;
        cascade(result.histogram_, histogram_.dimensions(),
                std::make_integer_sequence<Eigen::DenseIndex, N>{});
        return result;
    }

    /**
     * @brief Bilinearly interpolated histogram of the cell spanning
     * @f$[a, b)@f$.
     *
     * Evaluates the spatial part of the trilinear interpolation by Dalal and
     * Triggs on a second-order integral histogram obtained by cascade(). Each
     * input position @f$(x, y)@f$ contributes with the weight
     * @f$\max(0, 1 - |x - c_x| / s_x) \max(0, 1 - |y - c_y| / s_y)@f$ where
     * @f$c = (a + b - 1) / 2@f$ denotes the geometric center of the cell and
     * @f$s = b - a@f$ its size. Positions outside the input domain do not
     * contribute.
     *
     * Convolving a box filter with itself yields a tent. The weights are
     * therefore obtained by the second finite difference of the second-order
     * integral histogram. An even cell size places the center between two
     * positions in which case the stencils of both positions are averaged.
     */
    template<std::integral X, std::integral Y, Eigen::DenseIndex M = N>
        requires(M == 2)
    [[nodiscard]] Eigen::Tensor<T, K, Options> interpolate(
        const std::tuple<X, Y>& a, const std::tuple<X, Y>& b) const
    {
        std::array<Eigen::DenseIndex, static_cast<std::size_t>(K)> bins;
        std::copy_n(histogram_.dimensions().begin() + N, K, bins.begin());

        Eigen::Tensor<T, K, Options> result{bins};
        interpolate(a, b, result);
        return result;
    }

    /**
     * @brief Evaluates interpolate() into @p result instead of allocating the
     * histogram.
     *
     * Accumulating each bin directly avoids the temporaries of tensor
     * expressions, which allocate on every evaluation.
     *
     * @param result Histogram whose dimensions equal the histogram space
     * dimensions.
     */
    template<std::integral X, std::integral Y, Eigen::DenseIndex M = N>
        requires(M == 2)
    void interpolate(const std::tuple<X, Y>& a, const std::tuple<X, Y>& b,
                     Eigen::Tensor<T, K, Options>& result) const
    {
        const Stencil xs =
            stencil(static_cast<Eigen::DenseIndex>(std::get<0>(a)),
                    static_cast<Eigen::DenseIndex>(std::get<0>(b)),
                    histogram_.dimension(0));
        const Stencil ys =
            stencil(static_cast<Eigen::DenseIndex>(std::get<1>(a)),
                    static_cast<Eigen::DenseIndex>(std::get<1>(b)),
                    histogram_.dimension(1));

        const Eigen::DenseIndex rows = histogram_.dimension(0);
        const Eigen::DenseIndex cols = histogram_.dimension(1);
        const Eigen::DenseIndex bins = result.size();

        // Distances between consecutive slots along each axis and between
        // consecutive bins in the storage of the histogram
        constexpr bool RowMajor = (Options & Eigen::RowMajor) != 0;
        const Eigen::DenseIndex xStride = RowMajor ? cols * bins : 1;
        const Eigen::DenseIndex yStride = RowMajor ? bins : rows;
        const Eigen::DenseIndex stride = RowMajor ? 1 : rows * cols;

        const T normalization =
            static_cast<T>(xs.normalization) * static_cast<T>(ys.normalization);
        const T* const data = histogram_.data();

        for (Eigen::DenseIndex bin = 0; bin < bins; ++bin) {
            T sum{0};

            for (std::size_t i = 0; i < xs.size; ++i) {
                for (std::size_t j = 0; j < ys.size; ++j) {
                    sum += static_cast<T>(xs.weights[i] * ys.weights[j]) *
                           data[xs.indices[i] * xStride +
                                ys.indices[j] * yStride + bin * stride];
                }
            }

            result.data()[bin] = sum / normalization;
        }
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

        // The votes of each position are accumulated in a contiguous buffer
        // that is allocated once for the whole scan and stored to the
        // histogram afterwards. Passing a view of the histogram itself would
        // require an Eigen::TensorRef for the strided elements, whose
        // construction allocates for every position.
        std::array<Eigen::DenseIndex, static_cast<std::size_t>(K)> binDims{};
        std::copy(dims.begin() + N, dims.end(), binDims.begin());

        Eigen::Tensor<T, K, Options> buffer{binDims};
        Bins bins{buffer.data(), binDims};

        auto visit = [this, &binning, &bins](const auto& i) constexpr {
            detail::propagateBins(histogram_, i, bins.data());
            binning(bins, i);

            const detail::StridedBins<T> target =
                detail::stridedBins(histogram_, detail::offset<1>(i));

            for (Eigen::DenseIndex m = 0; m < target.size; ++m) {
                target.data[m * target.stride] = bins.data()[m];
            }
        };

        // Wavefront scan in the order of the memory layout. Each position
        // depends on the position stored right before it. Along the
        // contiguous dimension, that position lies next to the one being
        // stored, which no corner reads. Across it, the stored position lies
        // next to another corner, and compilers merge the loads of both into
        // a single wide load that cannot be forwarded from the narrow store
        // still in flight, which stalled the scan on wide SIMD instruction
        // sets.
        if constexpr (Tensor::Layout == Eigen::ColMajor) {
            cartesianProduct(std::make_tuple((dims[N - 1 - Indices] - 1)...),
                             [&visit](const auto& reversed) constexpr {
                                 visit(detail::reverse(reversed));
                             });
        }
        else {
            cartesianProduct(std::make_tuple((dims[Indices] - 1)...), visit);
        }
    }

    template<class Result, Eigen::DenseIndex... Indices>
    void cascade(
        Result& result,
        const std::array<Eigen::DenseIndex, static_cast<std::size_t>(N + K)>&
            dims,
        std::integer_sequence<Eigen::DenseIndex, Indices...> /*unused*/) const
    {
        using U = typename Result::Scalar;

        std::array<Eigen::DenseIndex, static_cast<std::size_t>(N + K)>
            extended = dims;

        for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i) {
            ++extended[i];
        }

        result.resize(extended);
        result.setZero();

        // Unlike scan(), the wavefront also covers the last slot of the
        // first-order integral histogram, i.e., the complete sums along each
        // dimension.
        cartesianProduct(
            std::make_tuple(dims[Indices]...), [this, &result](const auto& i) {
                auto&& H = detail::chip(result, detail::offset<1>(i));
                H = detail::propagate(result, i) +
                    detail::chip(histogram_, i).template cast<U>();
            });
    }

    // Two neighboring cell centers with three second finite difference taps
    // each. Taps beyond the upper domain boundary are extrapolated from the
    // last two slots, which can add at most two distinct slots.
    static constexpr std::size_t MaxStencilSize = 8;

    // Sparse weights of the second-order integral histogram slots along a
    // single dimension.
    struct Stencil
    {
        std::array<Eigen::DenseIndex, MaxStencilSize> indices{};
        std::array<Eigen::DenseIndex, MaxStencilSize> weights{};
        std::size_t size = 0;
        Eigen::DenseIndex normalization = 1;

        void add(Eigen::DenseIndex index, Eigen::DenseIndex weight) noexcept
        {
            const auto last = indices.begin() + size;
            const auto pos = std::find(indices.begin(), last, index);

            if (pos != last) {
                weights[static_cast<std::size_t>(pos - indices.begin())] +=
                    weight;
            }
            else {
                assert(size < MaxStencilSize);
                indices[size] = index;
                weights[size] = weight;
                ++size;
            }
        }
    };

    [[nodiscard]] static Stencil stencil(Eigen::DenseIndex first,
                                         Eigen::DenseIndex last,
                                         Eigen::DenseIndex extent) noexcept
    {
        const Eigen::DenseIndex size = last - first;

        assert(size > 0);

        // Positions next to the geometric center of the cell. Both coincide
        // for odd cell sizes.
        const std::array<Eigen::DenseIndex, 2> centers{first + (size - 1) / 2,
                                                       first + size / 2};
        // Second finite difference coefficients
        constexpr std::array<Eigen::DenseIndex, 3> coeffs{1, -2, 1};

        // Besides one slot per input position, the second-order integral
        // histogram has a leading zero slot and a trailing slot. The slots
        // grow linearly starting at the last input position since no input
        // follows it.
        constexpr Eigen::DenseIndex ExtraSlots = 2;
        const Eigen::DenseIndex boundary = extent - ExtraSlots;

        Stencil result;
        result.normalization =
            static_cast<Eigen::DenseIndex>(centers.size()) * size;

        for (const Eigen::DenseIndex center : centers) {
            for (std::size_t k = 0; k < coeffs.size(); ++k) {
                // A position contributes to the slot following it.
                const Eigen::DenseIndex tap =
                    center + 1 + (static_cast<Eigen::DenseIndex>(k) - 1) * size;

                if (tap <= 0) {
                    // Nothing precedes the input domain
                    continue;
                }

                if (tap <= boundary) {
                    result.add(tap, coeffs[k]);
                }
                else {
                    // Linear extrapolation from the last two slots
                    const Eigen::DenseIndex steps = tap - boundary;
                    result.add(boundary, coeffs[k] * (1 - steps));
                    result.add(boundary + 1, coeffs[k] * steps);
                }
            }
        }

        return result;
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

    template<Arithmetic U, Eigen::DenseIndex M, Eigen::DenseIndex L, int O>
    friend class IntegralHistogram;

    Tensor histogram_;
};

} // namespace hogpp

#endif // HOGPP_INTEGRALHISTOGRAM_HPP
