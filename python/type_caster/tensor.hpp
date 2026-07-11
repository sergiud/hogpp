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
#include <cstdint>
#include <type_traits>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

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

template<class T, class = void>
struct IsEigenTensorLike : std::false_type
{
};

template<class T>
struct IsEigenTensorLike<T, std::void_t
                         // clang-format off
<
      typename T::Scalar
    // Make sure the data method is available
    , decltype(std::declval<T>().data())
    // Make sure the tensor defines NumDimensions
    , decltype(std::decay_t<T>::NumDimensions)
    // Make sure the tensor provides the dimensions method
    , decltype(std::declval<T>().dimensions())
>
>
    // clang-format on
    : std::true_type
{
};

template<class T>
inline constexpr bool IsEigenTensorLike_v = IsEigenTensorLike<T>::value;

namespace nanobind::detail {

// nanobind's primary type_caster template defaults its SFINAE parameter to
// int (unlike pybind11, which defaults to void), so the condition must be
// expressed via enable_if_t rather than void_t directly.
template<class Tensor>
class type_caster<Tensor, enable_if_t<IsEigenTensorLike_v<Tensor>>>
{
public:
    NB_TYPE_CASTER(Tensor, const_name("numpy.ndarray"));

    using Scalar = typename Tensor::Scalar;

    bool from_python(handle src, std::uint8_t flags,
                     cleanup_list* /*cleanup*/) noexcept
    {
        // TODO Support float32
        const bool convert = (flags & (std::uint8_t)cast_flags::convert) != 0;

        if constexpr (Tensor::NumDimensions == 0) {
            if (!isinstance<nanobind::float_>(src)) {
                return false;
            }

            Scalar s{};

            if (!try_cast<Scalar>(src, s, convert)) {
                return false;
            }

            value(0) = s;

            return true;
        }
        else {
            // const so read-only arrays (e.g., ones loaded from an image
            // file) are accepted too; this caster never writes back into
            // the source array.
            nanobind::ndarray<const Scalar,
                              nanobind::ndim<Tensor::NumDimensions>>
                a;

            if (!try_cast(src, a, convert)) {
                return false;
            }

            const Shape shape = extentsOf(a);
            const Shape byteStrides = byteStridesOf(a);

            if (!contiguous(shape, byteStrides,
                            static_cast<const Scalar*>(a.data()))) {
                // Non-contiguous memory requires taking the strides into
                // consideration
                copy(reinterpret_cast<const std::uint8_t*>(a.data()), shape,
                     byteStrides, std::make_index_sequence<NumDimensions>{});
            }

            return true;
        }
    }

    template<class TensorType>
    static handle from_cpp(TensorType&& in, rv_policy policy,
                           cleanup_list* /*cleanup*/)
    {
        switch (policy) {
            case rv_policy::automatic:
                if constexpr (std::is_rvalue_reference_v<TensorType>) {
                    return move(std::forward<TensorType>(in));
                }
                // copy l-value references
                break;
            case rv_policy::automatic_reference:
            case rv_policy::reference:
            case rv_policy::reference_internal:
                return reference(std::forward<TensorType>(in));
            case rv_policy::move:
            case rv_policy::take_ownership:
                if constexpr (!std::is_const_v<
                                  std::remove_reference_t<TensorType>>) {
                    return move(std::forward<TensorType>(in));
                }
                [[fallthrough]];
            case rv_policy::copy:
            case rv_policy::none:
                // Delegate to default handler
                break;
        }

        // automatic & copy
        return copyOut(std::forward<TensorType>(in)); // LCOV_EXCL_LINE
    }

private:
    static constexpr std::size_t NumDimensions =
        static_cast<std::size_t>(Tensor::NumDimensions);
    using Shape = std::array<Eigen::DenseIndex, NumDimensions>;

    template<class Array>
    [[nodiscard]] static Shape extentsOf(const Array& a) noexcept
    {
        Shape shape{};

        for (std::size_t i = 0; i != NumDimensions; ++i) {
            shape[i] = static_cast<Eigen::DenseIndex>(a.shape(i));
        }

        return shape;
    }

    template<class Array>
    [[nodiscard]] static Shape byteStridesOf(const Array& a) noexcept
    {
        Shape byteStrides{};

        for (std::size_t i = 0; i != NumDimensions; ++i) {
            byteStrides[i] = static_cast<Eigen::DenseIndex>(a.stride(i)) *
                             static_cast<Eigen::DenseIndex>(sizeof(Scalar));
        }

        return byteStrides;
    }

    [[nodiscard]] bool contiguous(const Shape& shape, const Shape& byteStrides,
                                  const Scalar* data) noexcept
    {
        // Contiguous layout of C-style (row-major) arrays implies strides
        // matching ::c_strides, and similarly for Fortran-style (column
        // major) arrays and ::f_strides. Arrays can be both C-style and
        // Fortran-style contiguous simultaneously, e.g., trivially for
        // 1-dimensional arrays.

        if (byteStrides == ::c_strides<Scalar>(shape)) {
            if constexpr (RowMajor_v<Tensor::Options>) {
                value = map(data, shape);
            }
            else {
                value = reverseMap<Eigen::Tensor<Scalar, Tensor::NumDimensions,
                                                 Eigen::RowMajor>>(data, shape);
            }

            return true;
        }

        if (byteStrides == ::f_strides<Scalar>(shape)) {
            if constexpr (ColMajor_v<Tensor::Options>) {
                value = map(data, shape);
            }
            else {
                value = reverseMap<Eigen::Tensor<Scalar, Tensor::NumDimensions,
                                                 Eigen::ColMajor>>(data, shape);
            }

            return true;
        }

        // Not a contiguous layout
        return false;
    }

    template<class MappedTensor, class Ptr, std::size_t... Indices>
    [[nodiscard]] constexpr static decltype(auto) map(
        Ptr p, const Shape& shape,
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
        Ptr p, const Shape& shape) noexcept
    {
        return map<MappedTensor>(p, shape,
                                 std::make_index_sequence<NumDimensions>{});
    }

    template<class MappedTensor, class Ptr, std::size_t... Indices>
    [[nodiscard]] constexpr static
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
        decltype(auto)
#else  // !EIGEN_VERSION_AT_LEAST(3, 4, 0)
        Tensor // Avoid assertion in lazy assignment
#endif // EIGEN_VERSION_AT_LEAST(3, 4, 0)
        reverseMap(Ptr p, const Shape& shape,
                   std::index_sequence<Indices...> /*unused*/) noexcept
    {
        auto&& mapped = map<MappedTensor>(p, shape);
        return mapped.swap_layout().shuffle(
            std::array{(sizeof...(Indices) - Indices - 1)...});
    }

    template<class MappedTensor = Tensor, class Ptr>
    [[nodiscard]] constexpr static decltype(auto) reverseMap(
        Ptr p, const Shape& shape) noexcept
    {
        return reverseMap<MappedTensor>(
            p, shape, std::make_index_sequence<NumDimensions>{});
    }

    template<class TensorType>
    [[nodiscard]] static handle reference(TensorType&& t)
    {
        using NonRefTensor = std::decay_t<TensorType>;
        using TensorScalar = typename NonRefTensor::Scalar;
        constexpr auto N =
            static_cast<std::size_t>(NonRefTensor::NumDimensions);

        const std::array<Eigen::DenseIndex, N> byteStrides =
            strides<TensorScalar, NonRefTensor::NumDimensions,
                    NonRefTensor::Options>(t.dimensions());

        std::array<std::size_t, N> shape{};
        std::array<std::int64_t, N> elementStrides{};

        for (std::size_t i = 0; i != N; ++i) {
            shape[i] = static_cast<std::size_t>(t.dimension(i));
            elementStrides[i] = static_cast<std::int64_t>(
                byteStrides[i] /
                static_cast<Eigen::DenseIndex>(sizeof(TensorScalar)));
        }

        nanobind::ndarray<nanobind::numpy, TensorScalar> result{
            const_cast<std::remove_const_t<TensorScalar>*>(t.data()), N,
            shape.data(), nanobind::handle{}, elementStrides.data()};

        return result.cast().release();
    }

    template<class TensorType>
    [[nodiscard]] static handle move(TensorType&& t)
    {
        auto* p = new Tensor{std::forward<TensorType>(t)};
        return take(p);
    }

    template<class Scalar_, int N, int Options>
    [[nodiscard]] static handle copyOut(
        const Eigen::Tensor<Scalar_, N, Options>& t)
    {
        auto* p = new Tensor{t};
        return take(p);
    }

    template<class Scalar_, std::size_t N, int Options,
             std::enable_if_t<RowMajor_v<Options>>* = nullptr>
    [[nodiscard]] constexpr static decltype(auto) strides(
        const std::array<Eigen::DenseIndex, N>& dimensions) noexcept
    {
        return ::c_strides<Scalar_>(dimensions);
    }

    template<class Scalar_, std::size_t N, int Options,
             std::enable_if_t<ColMajor_v<Options>>* = nullptr>
    [[nodiscard]] constexpr static decltype(auto) strides(
        const std::array<Eigen::DenseIndex, N>& dimensions) noexcept
    {
        return ::f_strides<Scalar_>(dimensions);
    }

    [[nodiscard]] static handle take(Tensor* p)
    {
        nanobind::capsule owner{
            p, [](void* p) noexcept { delete static_cast<Tensor*>(p); }};

        const std::array<Eigen::DenseIndex, NumDimensions> byteStrides =
            strides<Scalar, Tensor::NumDimensions, Tensor::Options>(
                p->dimensions());

        std::array<std::size_t, NumDimensions> shape{};
        std::array<std::int64_t, NumDimensions> elementStrides{};

        for (std::size_t i = 0; i != NumDimensions; ++i) {
            shape[i] = static_cast<std::size_t>(p->dimension(i));
            elementStrides[i] = static_cast<std::int64_t>(
                byteStrides[i] /
                static_cast<Eigen::DenseIndex>(sizeof(Scalar)));
        }

        nanobind::ndarray<nanobind::numpy, Scalar> result{
            p->data(), NumDimensions, shape.data(), owner,
            elementStrides.data()};

        return result.cast().release();
    }

    template<class... Loops, class... Strides, std::size_t... Indices>
    void assign(const std::uint8_t* ptr, const std::tuple<Loops...>& i,
                const std::tuple<Strides...>& strides,
                std::index_sequence<Indices...> /*unused*/)
    {
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
    void copy(const std::uint8_t* ptr, const Shape& sizes,
              const Shape& byteStrides,
              std::index_sequence<Indices...> /*unused*/)
    {
        value.resize(sizes[Indices]...);
        copy(ptr, std::make_tuple(sizes[Indices]...),
             std::make_tuple(byteStrides[Indices]...));
    }
};

} // namespace nanobind::detail

#include <hogpp/suffix.hpp>

#endif // PYTHON_TYPE_CASTER_TENSOR_HPP
