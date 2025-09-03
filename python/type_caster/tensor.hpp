//
// HOGpp - Fast histogram of oriented gradients computation using integral
// histograms
//
// Copyright 2025 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

#ifndef PYTHON_TYPE_CASTER_TENSOR_HPP
#define PYTHON_TYPE_CASTER_TENSOR_HPP

#include <unsupported/Eigen/CXX11/Tensor>

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>

#include <hogpp/cartesianproduct.hpp>
#include <hogpp/prefix.hpp>

#include "stride.hpp"

template<class T, std::size_t N, std::size_t... Indices>
[[nodiscard]] constexpr Eigen::DenseIndex prod(
    const std::array<Eigen::DenseIndex, N>& values,
    std::index_sequence<Indices...> /*unused*/) noexcept
{
    return (static_cast<Eigen::DenseIndex>(sizeof(T)) * ... * values[Indices]);
}

template<class T, std::size_t N, std::size_t... Indices>
[[nodiscard]] constexpr decltype(auto) f_strides(
    const std::array<Eigen::DenseIndex, N>& values,
    std::index_sequence<Indices...> /*unused*/) noexcept
{
    return std::array<Eigen::DenseIndex, N>{
        prod<T>(values, std::make_index_sequence<Indices>{})...};
}

template<class T, std::size_t N>
[[nodiscard]] constexpr decltype(auto) f_strides(
    const std::array<Eigen::DenseIndex, N>& values) noexcept
{
    return f_strides<T>(values, std::make_index_sequence<N>{});
}

template<class T, std::size_t N>
[[nodiscard]] constexpr decltype(auto) c_strides(
    const std::array<Eigen::DenseIndex, N>& values) noexcept
{
    std::array<Eigen::DenseIndex, N> reversed = values;
    std::reverse(reversed.begin(), reversed.end());

    std::array<Eigen::DenseIndex, N> result = f_strides<T>(reversed);
    std::reverse(result.begin(), result.end());

    return result;
}

template<int Options>
struct RowMajor_t
    : std::bool_constant<(Options & Eigen::RowMajor) == Eigen::RowMajor>
{
};

template<int Options>
struct ColMajor_t : std::bool_constant<!RowMajor_t<Options>::value>
{
};

template<int Options>
inline constexpr bool RowMajor_v = RowMajor_t<Options>::value;

template<int Options>
inline constexpr bool ColMajor_v = ColMajor_t<Options>::value;

namespace pybind11::detail {

template<class Tensor>
class type_caster
    // clang-format off
<
      Tensor
    , std::void_t
    <
          typename Tensor::Scalar
        // Make sure the data method is available
        , decltype(std::declval<Tensor>().data())
        // Make sure the tensor defines NumDimensions
        , decltype(std::decay_t<Tensor>::NumDimensions)
        // Make sure the tensor provides the dimensions method
        , decltype(std::declval<Tensor>().dimensions())
    >
>
// clang-format on
{
public:
    PYBIND11_TYPE_CASTER(Tensor, _("numpy.ndarray"));

    bool load(handle in, bool /*unused*/)
    {
        using Scalar = typename Tensor::Scalar;

        // TODO Support float32
        if ((Tensor::NumDimensions > 0 && !isinstance<array>(in)) ||
            (Tensor::NumDimensions == 0 && (!isinstance<pybind11::float_>(in) ||
                                            !std::is_same_v<Scalar, double>))) {
            return false;
        }

        if (Tensor::NumDimensions == 0 && isinstance<pybind11::float_>(in)) {
            value(0) = pybind11::cast<Scalar>(in);
        }
        else {
            auto a = reinterpret_borrow<buffer>(in);
            auto info = a.request();

            if (!dtype{info}.equal(dtype::of<Scalar>())) {
                return false;
            }

            if (info.ndim != Tensor::NumDimensions) {
                return false;
            }

            bool fallback = true;

            if constexpr (Tensor::NumDimensions > 0) {
                fallback = !contiguous(info, a);
            }

            if (fallback) {
                // Non-contiguous memory requires taking the strides into
                // consideration
                copy(static_cast<const std::uint8_t*>(info.ptr), info.shape,
                     info.strides,
                     std::make_index_sequence<Tensor::NumDimensions>{});
            }
        }

        return true;
    }

    template<class TensorType>
    static handle cast(TensorType&& in, return_value_policy policy,
                       handle /* parent */)
    {
        switch (policy) {
            case return_value_policy::automatic:
                if constexpr (std::is_rvalue_reference_v<TensorType>) {
                    return move(std::forward<TensorType>(in));
                }
                // copy l-value references
                break;
            case return_value_policy::automatic_reference:
            case return_value_policy::reference:
            case return_value_policy::reference_internal:
                return reference(std::forward<TensorType>(in));
            case return_value_policy::move:
            case return_value_policy::take_ownership:
                if constexpr (!std::is_const_v<TensorType>) {
                    return move(std::forward<TensorType>(in));
                }
            case return_value_policy::copy:
                // Delegate to default handler
                break;
        }

        // automatic & copy
        return copy(std::forward<TensorType>(in)); // LCOV_EXCL_LINE
    }

private:
    // clang-format off
    template<bool RowMajor>
    static constexpr auto Style_v = std::conditional_t
    <
          RowMajor
        , std::integral_constant<int, array::c_style>
        , std::integral_constant<int, array::f_style>
    >::value;
    // clang-format on

    template<bool RowMajor>
    using Comparer_t =
        std::conditional_t<RowMajor, std::greater<>, std::less<>>;

    [[nodiscard]] bool contiguous(const pybind11::buffer_info& info,
                                  const pybind11::array& a) noexcept
    {
        using Scalar = typename Tensor::Scalar;

        constexpr auto style = Style_v<RowMajor_v<Tensor::Options>>;
        constexpr auto otherStyle = Style_v<ColMajor_v<Tensor::Options>>;

        // Contiguous layout of C-style arrays implies strides in descending
        // order
        using ThisComparer = Comparer_t<RowMajor_v<Tensor::Options>>;
        // Contiguous layout of F-style arrays implies strides in ascending
        // order
        using OtherComparer = Comparer_t<ColMajor_v<Tensor::Options>>;

        // Arrays can be both C-style and Fortran-style contiguous
        // simultaneously. This is evidently true for 1-dimensional arrays, but
        // can also be true for higher dimensional arrays. At the same time,
        // contiguous layout does not necessarily guarantee the strides to be in
        // the expected order. Therefore, when matching the layout, we need to
        // ensure that the strides are not arbitrary but sorted in the order
        // given by the corresponding memory layout.

        if ((a.flags() & style) == style &&
            std::is_sorted(info.strides.begin(), info.strides.end(),
                           ThisComparer{})) {
            // Take a shortcut
            value = map(static_cast<const Scalar*>(info.ptr), info.shape);
        }
        else if ((a.flags() & otherStyle) == otherStyle &&
                 std::is_sorted(info.strides.begin(), info.strides.end(),
                                OtherComparer{})) {
            // If incompatible contiguous memory layout (i.e., compatible to
            // the opposite layout, we can still construct a tensor and
            // change the layout post hoc.
            using OtherOptions_t = std::conditional_t<
                ColMajor_v<Tensor::Options>,
                std::integral_constant<int, Eigen::RowMajor>,
                std::integral_constant<int, Eigen::ColMajor>>;

            using MappedTensor = Eigen::Tensor<Scalar, Tensor::NumDimensions,
                                               OtherOptions_t::value>;

            value = reverseMap<MappedTensor>(
                static_cast<const Scalar*>(info.ptr), info.shape);
        }
        else {
            // Not contiguous layout
            return false;
        }

        // Processed contiguous layout
        return true;
    }

    template<class MappedTensor, class Ptr, std::size_t... Indices>
    [[nodiscard]] constexpr static decltype(auto) map(
        Ptr p, const std::vector<pybind11::ssize_t>& shape,
        std::index_sequence<Indices...> /*unused*/) noexcept
    {
        static_assert(std::is_pointer_v<Ptr>, "Ptr must be a pointer");

        using T =
            std::conditional_t<std::is_const_v<std::remove_pointer_t<Ptr>>,
                               std::add_const_t<MappedTensor>, MappedTensor>;

        return Eigen::TensorMap<T>{
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
            p,
#else  // !EIGEN_VERSION_AT_LEAST(3, 4, 0)
            const_cast<std::remove_const_t<std::remove_pointer_t<Ptr>>*>(p),
#endif // EIGEN_VERSION_AT_LEAST(3, 4, 0)
            shape[Indices]...};
    }

    template<class MappedTensor = Tensor, class Ptr>
    [[nodiscard]] constexpr static decltype(auto) map(
        Ptr p, const std::vector<pybind11::ssize_t>& shape) noexcept
    {
        return map<MappedTensor>(
            p, shape, std::make_index_sequence<Tensor::NumDimensions>{});
    }

    template<class MappedTensor, class Ptr, std::size_t... Indices>
    [[nodiscard]] constexpr static
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
        decltype(auto)
#else  // !EIGEN_VERSION_AT_LEAST(3, 4, 0)
        Tensor // Avoid assertion in lazy assignment
#endif // EIGEN_VERSION_AT_LEAST(3, 4, 0)
        reverseMap(Ptr p, const std::vector<pybind11::ssize_t>& shape,
                   std::index_sequence<Indices...> /*unused*/) noexcept
    {
        auto&& mapped = map<MappedTensor>(p, shape);
        return mapped.swap_layout().shuffle(
            std::array{(sizeof...(Indices) - Indices - 1)...});
    }

    template<class MappedTensor = Tensor, class Ptr>
    [[nodiscard]] constexpr static decltype(auto) reverseMap(
        Ptr p, const std::vector<pybind11::ssize_t>& shape) noexcept
    {
        return reverseMap<MappedTensor>(
            p, shape, std::make_index_sequence<Tensor::NumDimensions>{});
    }

    template<class TensorType>
    [[nodiscard]] static handle reference(TensorType&& t)
    {
        using NonRefTensor = std::decay_t<TensorType>;
        using Scalar = typename NonRefTensor::Scalar;
        constexpr auto N =
            static_cast<std::size_t>(NonRefTensor::NumDimensions);

        const std::array<Eigen::DenseIndex, N> s =
            strides<Scalar, NonRefTensor::NumDimensions, NonRefTensor::Options>(
                t.dimensions());

        buffer_info info{const_cast<Scalar*>(t.data()), t.dimensions(),
                         std::vector<pybind11::ssize_t>{s.begin(), s.end()},
                         std::is_const_v<TensorType>};

        return array{info}.release();
    }

    template<class TensorType>
    [[nodiscard]] static handle move(TensorType&& t)
    {
        using NonRefTensor = std::decay_t<TensorType>;
        using Scalar = typename NonRefTensor::Scalar;

        constexpr auto N = NonRefTensor::NumDimensions;
        constexpr auto Options = NonRefTensor::Options;

        auto* p = new Tensor{std::forward<TensorType>(t)};
        return take<Scalar, N, Options>(p);
    }

    template<class Scalar, int N, int Options>
    [[nodiscard]] static handle copy(const Eigen::Tensor<Scalar, N, Options>& t)
    {
        auto* p = new Tensor{t};
        return take<Scalar, N, Options>(p);
    }

    template<class Scalar, std::size_t N, int Options,
             std::enable_if_t<RowMajor_v<Options>>* = nullptr>
    [[nodiscard]] constexpr static decltype(auto) strides(
        const std::array<Eigen::DenseIndex, N>& dimensions) noexcept
    {
        return ::c_strides<Scalar>(dimensions);
    }

    template<class Scalar, std::size_t N, int Options,
             std::enable_if_t<ColMajor_v<Options>>* = nullptr>
    [[nodiscard]] constexpr static decltype(auto) strides(
        const std::array<Eigen::DenseIndex, N>& dimensions) noexcept
    {
        return ::f_strides<Scalar>(dimensions);
    }

    template<class Scalar, std::size_t N, int Options>
    [[nodiscard]] static handle take(Tensor* p)
    {
        capsule cleanup{p, [](void* p) { delete static_cast<Tensor*>(p); }};

        const std::array<Eigen::DenseIndex, N> s =
            strides<Scalar, N, Options>(p->dimensions());

        array result{p->dimensions(),
                     std::vector<pybind11::ssize_t>{s.begin(), s.end()},
                     p->data(), cleanup};

        return result.release();
    }

    template<class... Loops, class... Strides, std::size_t... Indices>
    void assign(const std::uint8_t* ptr, const std::tuple<Loops...>& i,
                const std::tuple<Strides...>& strides,
                std::index_sequence<Indices...> /*unused*/)
    {
        using Scalar = typename Tensor::Scalar;

        value(std::get<Indices>(i)...) =
            *reinterpret_cast<const Scalar*>(ptr + reduceIndices(i, strides));
    }

    template<class... Loops, class... Strides>
    void assign(const std::uint8_t* ptr, const std::tuple<Loops...>& i,
                const std::tuple<Strides...>& strides)
    {
        assign(ptr, i, strides, std::index_sequence_for<Loops...>{});
    }

    template<class... Sizes, class... Strides>
    void copy(const std::uint8_t* ptr, const std::tuple<Sizes...>& sizes,
              const std::tuple<Strides...>& strides)
    {
        hogpp::cartesianProduct(
            sizes, [this, ptr, strides](const auto& i) constexpr {
                (void)this; // Avoid (incorrect) Clang -Wunused-lambda-capture
                            // warning
                assign(ptr, i, strides);
            });
    }

    template<std::size_t... Indices>
    void copy(const std::uint8_t* ptr, const std::vector<ssize_t>& sizes,
              const std::vector<ssize_t>& strides,
              std::index_sequence<Indices...> /*unused*/)
    {
        value.resize(sizes[Indices]...);
        copy(ptr, std::make_tuple(sizes[Indices]...),
             std::make_tuple(strides[Indices]...));
    }
};

} // namespace pybind11::detail

#include <hogpp/suffix.hpp>

#endif // PYTHON_TYPE_CASTER_TENSOR_HPP
