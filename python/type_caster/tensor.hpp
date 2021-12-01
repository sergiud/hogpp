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

#ifndef PYTHON_TYPE_CASTER_TENSOR_HPP
#define PYTHON_TYPE_CASTER_TENSOR_HPP

#include <unsupported/Eigen/CXX11/Tensor>

#include <array>
#include <cstddef>
#include <type_traits>

#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>

#include "cartesianproduct.hpp"
#include "stride.hpp"

template<class T, std::size_t N, std::size_t... Indices>
[[nodiscard]] constexpr Eigen::DenseIndex prod(
    const std::array<Eigen::DenseIndex, N>& values,
    std::index_sequence<Indices...> /*unused*/) noexcept
{
    return (static_cast<Eigen::DenseIndex>(sizeof(T)) * ... * values[Indices]);
}

template<class T, std::size_t N, std::size_t... Indices>
[[nodiscard]] constexpr auto strides(
    const std::array<Eigen::DenseIndex, N>& values,
    std::index_sequence<Indices...> /*unused*/) noexcept
{
    return std::array<Eigen::DenseIndex, N>{
        prod<T>(values, std::make_index_sequence<Indices>{})...};
}

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
            auto a = reinterpret_borrow<array>(in);
            auto info = a.request();

            if (info.format != format_descriptor<Scalar>::format()) {
                return false;
            }

            if (info.ndim != Tensor::NumDimensions) {
                return false;
            }

            copy(static_cast<const std::uint8_t*>(info.ptr), info.shape,
                 info.strides,
                 std::make_index_sequence<Tensor::NumDimensions>{});
        }

        return true;
    }

    static handle cast(Tensor in, return_value_policy /* policy */,
                       handle /* parent */)
    {
        return convert(in);
    }

private:
    template<class Scalar, int Dims, int Options>
    static handle convert(Eigen::Tensor<Scalar, Dims, Options> t)
    {
        auto* p = new Tensor{std::move(t)};

        capsule cleanup{p, [](void* p) { delete static_cast<Tensor*>(p); }};
        constexpr auto N = static_cast<std::size_t>(Dims);

        const std::array<Eigen::DenseIndex, N> s =
            strides<Scalar>(p->dimensions(), std::make_index_sequence<N>{});

        constexpr auto style = (Options & Eigen::ColMajor) == Eigen::ColMajor
                                   ? array::f_style
                                   : array::c_style;

        array_t<Scalar, style> result{
            p->dimensions(), std::vector<Eigen::DenseIndex>{s.begin(), s.end()},
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
        cartesianProduct(sizes, [this, ptr, strides](const auto& i) constexpr {
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

#endif // PYTHON_TYPE_CASTER_TENSOR_HPP
