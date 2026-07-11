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

#ifndef PYTHON_TYPE_CASTER_OPENCV_HPP
#define PYTHON_TYPE_CASTER_OPENCV_HPP

#include <opencv2/core/core.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include <hogpp/cartesianproduct.hpp>
#include <hogpp/prefix.hpp>

#include "stride.hpp"
#include "typesequence.hpp"

[[noreturn]] inline nanobind::dlpack::dtype depthToDtype(
    [[maybe_unused]] int depth, [[maybe_unused]] TypeSequence<> /*unused*/)
{
    throw std::invalid_argument{
        "cannot find the appropriate OpenCV depth to NumPy dtype mapping"};
}

template<class Type, class... Types>
[[nodiscard]] constexpr nanobind::dlpack::dtype
depthToDtype(int depth, TypeSequence<Type, Types...> /*unused*/) noexcept(
    noexcept(depthToDtype(depth, TypeSequence<Types...>{})))
{
    if (depth == cv::DataDepth<Type>::value) {
        return nanobind::dtype<Type>();
    }

    return depthToDtype(depth, TypeSequence<Types...>{});
}

namespace nanobind::detail {

template<>
class type_caster<cv::Mat>
{
public:
    NB_TYPE_CASTER(cv::Mat, const_name("numpy.ndarray"))

    bool from_python(handle src, std::uint8_t flags,
                     cleanup_list* /*cleanup*/) noexcept
    {
        const bool convert =
            (flags & static_cast<std::uint8_t>(cast_flags::convert)) != 0;

        nanobind::ndarray<nanobind::ro> a;

        if (!try_cast(src, a, convert)) {
            return false;
        }

        return tryConvert(a, OpenCVTypes{});
    }

    static handle from_cpp(const cv::Mat& in, rv_policy /*policy*/,
                           cleanup_list* /*cleanup*/)
    {
        if (in.total() == 0) {
            return nanobind::none().release();
        }

        // Deep-copy into a freshly allocated, contiguous buffer so the
        // returned array is independent of the source cv::Mat, matching the
        // value semantics NumPy callers expect.
        auto* clone = new cv::Mat{in.clone()}; // NOLINT

        nanobind::capsule owner{
            clone, [](void* p) noexcept { delete static_cast<cv::Mat*>(p); }};

        const nanobind::dlpack::dtype dt =
            depthToDtype(clone->depth(), OpenCVTypes{});

        std::size_t ndim = 2;
        std::array<std::size_t, 3> shape{static_cast<std::size_t>(clone->rows),
                                         static_cast<std::size_t>(clone->cols),
                                         1};

        if (clone->channels() > 1) {
            ndim = 3;
            shape[2] = static_cast<std::size_t>(clone->channels());
        }

        nanobind::ndarray<nanobind::numpy> result{
            clone->ptr(), ndim, shape.data(), owner, nullptr, dt};

        return result.cast().release();
    }

private:
    using OpenCVTypes = TypeSequence<std::uint8_t, std::int8_t, std::uint16_t,
                                     std::int16_t, std::int32_t, float, double>;

    template<class T>
    static decltype(auto) make(const nanobind::ndarray<nanobind::ro>& a)
    {
        const std::size_t ndim = a.ndim();

        std::vector<std::int64_t> shape(ndim);
        std::vector<std::int64_t> byteStrides(ndim);

        for (std::size_t i = 0; i != ndim; ++i) {
            shape[i] = static_cast<std::int64_t>(a.shape(i));
            byteStrides[i] = static_cast<std::int64_t>(a.stride(i)) *
                             static_cast<std::int64_t>(sizeof(T));
        }

        const int channels =
            shape.size() == 3 ? static_cast<int>(shape.back()) : 1;
        constexpr int depth = cv::DataDepth<T>::value;

        const int type = CV_MAKETYPE(depth, channels);

        cv::Mat in;
        bool success = true;

        // TODO Wrap the buffer only if the last stride is 1.

        switch (shape.size()) {
            case 2: {
                if (byteStrides[0] > 0 &&
                    byteStrides[1] == static_cast<std::int64_t>(sizeof(T))) {
                    std::vector<int> sizes;
                    sizes.resize(shape.size());

                    std::transform(shape.begin(), shape.end(), sizes.begin(),
                                   [](std::int64_t value) constexpr {
                                       return static_cast<int>(value);
                                   });

                    std::vector<std::size_t> steps{byteStrides.begin(),
                                                   byteStrides.end()};

                    in = cv::Mat{2, sizes.data(), type,
                                 const_cast<void*>(a.data()), steps.data()};
                }
                else {
                    const auto* const p =
                        static_cast<const std::uint8_t*>(a.data());

                    in.create(static_cast<int>(shape[0]),
                              static_cast<int>(shape[1]), type);

                    hogpp::cartesianProduct(
                        std::make_tuple(shape[0], shape[1]),
                        [&in, p, &byteStrides](const auto& i) constexpr {
                            assign<T>(in, p, i, byteStrides);
                        });
                }
            } break;
            case 3: {
                if (!(channels == 1 || channels == 3 || channels == 4)) {
                    success = false;
                }
                else {
                    if (byteStrides[0] > 0 &&
                        byteStrides[1] ==
                            static_cast<std::int64_t>(sizeof(T)) * channels &&
                        byteStrides[2] ==
                            static_cast<std::int64_t>(sizeof(T))) {
                        std::vector<int> sizes;
                        sizes.resize(shape.size());

                        std::transform(shape.begin(), shape.end(),
                                       sizes.begin(),
                                       [](std::int64_t value) constexpr {
                                           return static_cast<int>(value);
                                       });
                        std::vector<std::size_t> steps{byteStrides.begin(),
                                                       byteStrides.end()};

                        in = cv::Mat{2, sizes.data(), type,
                                     const_cast<void*>(a.data()), steps.data()};
                    }
                    else {
                        const auto* const p =
                            static_cast<const std::uint8_t*>(a.data());

                        in.create(static_cast<int>(shape[0]),
                                  static_cast<int>(shape[1]), type);

                        hogpp::cartesianProduct(
                            std::make_tuple(shape[0], shape[1], shape[2]),
                            [&in, p, &byteStrides](const auto& i) constexpr {
                                assign<T>(in, p, i, byteStrides);
                            });
                    }
                }
            } break;
        }

        return std::make_tuple(in, success);
    }

    template<class T, class... Loops, class... Strides, std::size_t... Indices,
             std::enable_if_t<sizeof...(Loops) == 2>* = nullptr>
    constexpr static void assign(cv::Mat& in, const std::uint8_t* p,
                                 const std::tuple<Loops...>& i,
                                 const std::tuple<Strides...>& strides,
                                 std::index_sequence<Indices...> /*unused*/)
    {
        in.at<T>(static_cast<int>(std::get<Indices>(i))...) =
            *reinterpret_cast<const T*>(p + reduceIndices(i, strides));
    }

    template<class T, class... Loops, class... Strides, std::size_t Index0,
             std::size_t Index1, std::size_t Index2,
             std::enable_if_t<sizeof...(Loops) == 3>* = nullptr>
    constexpr static void assign(
        cv::Mat& in, const std::uint8_t* p, const std::tuple<Loops...>& i,
        const std::tuple<Strides...>& strides,
        std::index_sequence<Index0, Index1, Index2> /*unused*/)
    {
        in.ptr<T>(static_cast<int>(std::get<Index0>(i)),
                  static_cast<int>(std::get<Index1>(i)))[std::get<Index2>(i)] =
            *reinterpret_cast<const T*>(p + reduceIndices(i, strides));
    }

    template<class T, class... Loops, std::size_t... Indices>
    constexpr static void assign(cv::Mat& in, const std::uint8_t* p,
                                 const std::tuple<Loops...>& i,
                                 const std::vector<std::int64_t>& strides,
                                 std::index_sequence<Indices...> idxs)
    {
        assign<T>(in, p, i, std::make_tuple(strides[Indices]...), idxs);
    }

    template<class T, class... Loops, class Strides>
    constexpr static void assign(cv::Mat& in, const std::uint8_t* p,
                                 const std::tuple<Loops...>& i,
                                 const Strides& strides)
    {
        assign<T>(in, p, i, strides, std::index_sequence_for<Loops...>{});
    }

    template<class T>
    bool convert(const nanobind::ndarray<nanobind::ro>& a)
    {
        if (nanobind::dtype<T>() != a.dtype()) {
            return false;
        }

        if (a.ndim() == 1) {
            cv::Mat_<T> in{1, static_cast<int>(a.shape(0)),
                           const_cast<T*>(static_cast<const T*>(a.data()))};

            value = in;
        }
        else if (a.ndim() == 2 || a.ndim() == 3) {
            auto&& [in, success] = make<T>(a);

            if (!success) {
                return false;
            }

            value = in;
        }
        else {
            return false;
        }

        return true;
    }

    template<class... Types>
    bool tryConvert(const nanobind::ndarray<nanobind::ro>& a,
                    TypeSequence<Types...> /*unused*/)
    {
        return (convert<Types>(a) || ...);
    }
};

template<class T>
class type_caster<cv::Mat_<T>> : public type_caster<cv::Mat>
{
};

template<class T>
class type_caster<cv::Rect_<T>>
{
public:
    NB_TYPE_CASTER(cv::Rect_<T>, const_name("Rect"))

    bool from_python(handle src, std::uint8_t /*flags*/,
                     cleanup_list* /*cleanup*/) noexcept
    {
        try {
            std::tie(value.y, value.x, value.height, value.width) =
                nanobind::cast<std::tuple<T, T, T, T>>(src);
        }
        catch (const nanobind::python_error&) {
            return false;
        }
        catch (const nanobind::cast_error&) {
            return false;
        }

        return true;
    }

    static handle from_cpp(const cv::Rect_<T>& in, rv_policy /*policy*/,
                           cleanup_list* /*cleanup*/)
    {
        return make_tuple(in.y, in.x, in.height, in.width).release();
    }
};

extern template class type_caster<cv::Rect>;

} // namespace nanobind::detail

#include <hogpp/suffix.hpp>

#endif // PYTHON_TYPE_CASTER_OPENCV_HPP
