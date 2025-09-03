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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

#include <pybind11/cast.h>
#include <pybind11/numpy.h>

#include <hogpp/cartesianproduct.hpp>
#include <hogpp/prefix.hpp>

#include "stride.hpp"
#include "typesequence.hpp"

[[noreturn]] inline std::string depthToFormat(
    [[maybe_unused]] int depth, [[maybe_unused]] TypeSequence<> /*unused*/)
{
    throw std::invalid_argument{
        "cannot find the appropriate OpenCV depth to NumPy format mapping"};
}

template<class Type, class... Types>
[[nodiscard]] constexpr auto
depthToFormat(int depth, TypeSequence<Type, Types...> /*unused*/) noexcept(
    noexcept(depthToFormat(depth, TypeSequence<Types...>{})))
{
    if (depth == cv::DataDepth<Type>::value) {
        return pybind11::format_descriptor<Type>::format();
    }

    return depthToFormat(depth, TypeSequence<Types...>{});
}

namespace pybind11::detail {

template<>
class type_caster<cv::Mat>
{
public:
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

    bool load(handle in, bool /*unused*/)
    {
        auto a = reinterpret_borrow<array>(in);
        auto info = a.request();

        return tryConvert(info, OpenCVTypes{});
    }

    static handle cast(const cv::Mat& in, return_value_policy /*policy*/,
                       handle /*parent*/)
    {
        array result;

        const pybind11::dtype dt{depthToFormat(in.depth(), OpenCVTypes{})};

        if (in.total() != 0) {
            array::ShapeContainer shape{{in.rows, in.cols}};
            array::StridesContainer strides{std::begin(in.step.buf),
                                            std::end(in.step.buf)};

            if (in.channels() > 1) {
                shape->push_back(in.channels());
                strides->push_back(static_cast<ssize_t>(in.elemSize1()));
            }

            result = array(dt, shape, strides, in.ptr());
        }

        return result.release();
    }

private:
    using OpenCVTypes = TypeSequence<std::uint8_t, std::int8_t, std::uint16_t,
                                     std::int16_t, std::int32_t, float, double>;

    template<class T>
    static decltype(auto) make(const buffer_info& info)
    {
        const auto& shape = info.shape;
        const auto& strides = info.strides;

        const int channels = shape.size() == 3
                                 ? static_cast<int>(shape.back())
                                 : 1; // static_cast<int>(info.strides.back());
        constexpr int depth = cv::DataDepth<T>::value;

        const int type = CV_MAKETYPE(depth, channels);

        cv::Mat in;
        bool success = true;

        // TODO Wrap the buffer only if the last stride is 1.

        switch (shape.size()) {
            case 2: {
                if (strides[0] > 0 && strides[1] == sizeof(T)) {
                    std::vector<int> sizes;
                    sizes.resize(shape.size());

                    std::transform(shape.begin(), shape.end(), sizes.begin(),
                                   [](pybind11::ssize_t value) constexpr {
                                       return static_cast<int>(value);
                                   });

                    std::vector<std::size_t> steps{strides.begin(),
                                                   strides.end()};

                    in = cv::Mat{2, sizes.data(), type, info.ptr, steps.data()};
                }
                else {
                    const auto* const p =
                        static_cast<const std::uint8_t*>(info.ptr);

                    in.create(static_cast<int>(shape[0]),
                              static_cast<int>(shape[1]), type);

                    hogpp::cartesianProduct(
                        std::make_tuple(shape[0], shape[1]),
                        [&in, p, strides](const auto& i) constexpr {
                            assign<T>(in, p, i, strides);
                        });
                }
            } break;
            case 3: {
                if (!(channels == 1 || channels == 3 || channels == 4)) {
                    success = false;
                }
                else {
                    if (strides[0] > 0 &&
                        strides[1] == static_cast<long>(sizeof(T)) * channels &&
                        strides[2] == static_cast<long>(sizeof(T))) {
                        std::vector<int> sizes;
                        sizes.resize(shape.size());

                        std::transform(shape.begin(), shape.end(),
                                       sizes.begin(),
                                       [](pybind11::ssize_t value) constexpr {
                                           return static_cast<int>(value);
                                       });
                        std::vector<std::size_t> steps{strides.begin(),
                                                       strides.end()};

                        in = cv::Mat{2, sizes.data(), type, info.ptr,
                                     steps.data()};
                    }
                    else {
                        const auto* const p =
                            static_cast<const std::uint8_t*>(info.ptr);

                        in.create(static_cast<int>(shape[0]),
                                  static_cast<int>(shape[1]), type);

                        hogpp::cartesianProduct(
                            std::make_tuple(shape[0], shape[1], shape[2]),
                            [&in, p, strides](const auto& i) constexpr {
                                assign<T>(in, p, i, strides);
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
                                 const std::vector<ssize_t>& strides,
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
    bool convert(const buffer_info& info)
    {
        if (dtype{info}.equal(dtype::of<T>())) {
            if (info.ndim == 1) {
                cv::Mat_<T> in{1, static_cast<int>(info.shape[0]),
                               static_cast<T*>(info.ptr)};

                value = in;
            }
            else if (info.ndim == 2 || info.ndim == 3) {
                auto&& [in, success] = make<T>(info);

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

        return false;
    }

    template<class... Types>
    bool tryConvert(const buffer_info& info, TypeSequence<Types...> /*unused*/)
    {
        return (convert<Types>(info) || ...);
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
    PYBIND11_TYPE_CASTER(cv::Rect_<T>, _("Rect"));

    bool load(handle src, bool /*unused*/)
    {
        try {
            std::tie(value.y, value.x, value.height, value.width) =
                pybind11::cast<std::tuple<T, T, T, T>>(src);
        }
        catch (const pybind11::builtin_exception&) {
            return false;
        }

        return true;
    }

    static handle cast(const cv::Rect_<T>& in, return_value_policy /*policy*/,
                       handle /*parent*/)
    {
        return make_tuple(in.y, in.x, in.height, in.width).release();
    }
};

extern template class type_caster<cv::Rect>;

} // namespace pybind11::detail

#include <hogpp/suffix.hpp>

#endif // PYTHON_TYPE_CASTER_OPENCV_HPP
