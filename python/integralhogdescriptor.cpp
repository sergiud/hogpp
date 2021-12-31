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

#include <algorithm>
#include <cstddef>

#ifdef HAVE_EXECUTION
#include <execution>
#endif // HAVE_EXECUTION

#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/iterator_range.hpp>

#include <fmt/format.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include "formatter.hpp"
#include "integralhogdescriptor.hpp"
#include "type_caster/opencv.hpp"
#include "type_caster/tensor.hpp"

namespace {

template<class T>
struct NativeScalar : NativeScalar<std::decay_t<T> >
{
};

template<>
struct NativeScalar<pybind11::float_>
{
    using type = double;
};

template<>
struct NativeScalar<pybind11::int_>
{
    using type = int;
};

template<class T>
using NativeScalar_t = typename NativeScalar<T>::type;

template<class Function>
[[maybe_unused]] constexpr void matToTensorRank2(
    [[maybe_unused]] const cv::Mat& image, [[maybe_unused]] Function&& func,
    Types<> /*unused*/)
{
}

template<class Function, class Arg, class... Args>
[[maybe_unused]] void matToTensorRank2(const cv::Mat& image, Function&& func,
                                       Types<Arg, Args...> /*unused*/)
{
    if (image.depth() != cv::DataDepth<Arg>::value) {
        matToTensorRank2(image, std::forward<Function>(func), Types<Args...>{});
    }
    else {
        Eigen::TensorMap<const Eigen::Tensor<Arg, 2, Eigen::RowMajor> > t{
            image.ptr<Arg>(), image.rows, image.cols};
        func(t.reshape(std::array<long, 3>{{image.rows, image.cols, 1}})
                 .swap_layout()
                 .shuffle(std::array<int, 3>{{2, 1, 0}}));
    }
}

template<class Function>
[[maybe_unused]] constexpr void matToTensorRank3(
    [[maybe_unused]] const cv::Mat& image, [[maybe_unused]] Function&& func,
    Types<> /*unused*/)
{
}

template<class Function, class Arg, class... Args>
[[maybe_unused]] void matToTensorRank3(const cv::Mat& image, Function&& func,
                                       Types<Arg, Args...> /*unused*/)
{
    if (image.depth() != cv::DataDepth<Arg>::value) {
        matToTensorRank3(image, std::forward<Function>(func), Types<Args...>{});
    }
    else {
        Eigen::TensorMap<const Eigen::Tensor<Arg, 3, Eigen::RowMajor> > t{
            image.ptr<Arg>(), 1, 1, image.rows * image.cols * image.channels()};
        func(t.reshape(std::array<int, 3>{
                           {image.rows, image.cols, image.channels()}})
                 .swap_layout()
                 .shuffle(std::array<int, 3>{{2, 1, 0}}));
    }
}

template<class Function>
void bufferToTensor( // LCOV_EXCL_LINE
    [[maybe_unused]] const pybind11::buffer& image,
    [[maybe_unused]] const pybind11::buffer_info& info,
    [[maybe_unused]] Function&& func, Types<> /*unused*/)
{
}

template<class Arg>
[[nodiscard]] Eigen::Tensor<Arg, 3> extend(const pybind11::buffer& image,
                                           const pybind11::buffer_info& info)
{
    using Tensor2 = Eigen::Tensor<Arg, 2>;
    using Tensor3 = Eigen::Tensor<Arg, 3>;

    if (info.ndim == 2) {
        auto tmp = pybind11::cast<Tensor2>(image);
        // Extend rank-2 arrays by a third dimension.
        return tmp.reshape(std::array{tmp.dimension(0), tmp.dimension(1),
                                      typename Tensor2::Index{1}});
    }

    return pybind11::cast<Tensor3>(image);
}

template<class Function, class Arg, class... Args>
void bufferToTensor(const pybind11::buffer& image,
                    const pybind11::buffer_info& info, Function&& func,
                    Types<Arg, Args...> /*unused*/)
{
    if (info.format != pybind11::format_descriptor<Arg>::format()) {
        bufferToTensor(image, info, std::forward<Function>(func),
                       Types<Args...>{});
    }
    else {
        func(extend<Arg>(image, info));
    }
}

template<long... Ranks, class Function, class... Args>
void bufferToTensor(const RankNTensorPair<Ranks...>& p,
                    const pybind11::buffer_info& info, Function&& func,
                    Types<Args...> types)
{
    auto convert = [&p, &func, &info](const auto& t) {
        using T = typename std::decay_t<decltype(t)>::Scalar;
        func(t, extend<T>(p.buf1.buf, info));
    };

    // Assume [dys, dxs] ordering
    bufferToTensor(p.buf2.buf, info, convert, types);
}

} // namespace

IntegralHOGDescriptor::IntegralHOGDescriptor(
    const std::optional<Eigen::Array2i>& cellSize,
    const std::optional<Eigen::Array2i>& blockSize,
    const std::optional<Eigen::Array2i>& blockStride,
    const std::optional<pybind11::int_>& numBins,
    const std::optional<MagnitudeType>& magnitude,
    const std::optional<BinningType>& binning,
    const std::optional<BlockNormalizerType>& blockNorm,
    const std::optional<std::variant<pybind11::int_, pybind11::float_> >&
        clipNorm,
    const std::optional<std::variant<pybind11::int_, pybind11::float_> >&
        epsilon)
    : cellSize_{cellSize}
    , blockSize_{blockSize}
    , blockStride_{blockStride}
    , numBins_{numBins}
    , magnitudeType_{magnitude}
    , binningType_{binning}
    , blockNormalizerType_{blockNorm}
    , clipNorm_{clipNorm}
    , epsilon_{epsilon}
{
    // Delay setting parameters until we know the floating point type we
    // will be working with.

    if (clipNorm) {
        // Allow None
        std::visit(
            [](const auto& value) {
                if (!value.is_none()) {
                    try {
                        if (!(value >
                              std::decay_t<decltype(value)>{
                                  NativeScalar_t<decltype(value)>{0}})) {
                            throw pybind11::cast_error{
                                "clip_norm is 0 or negative"};
                        }
                    }
                    catch (const pybind11::cast_error&) {
                        throw std::invalid_argument{fmt::format(
                            "IntegralHOGDescriptor clip_norm can only "
                            "be a positive floating point value but {} was "
                            "given",
                            value)};
                    }
                }
            },
            *clipNorm);
    }

    if (epsilon) {
        // Allow None
        std::visit(
            [](const auto& value) {
                if (!value.is_none()) {
                    try {
                        if (value < std::decay_t<decltype(value)>{
                                        NativeScalar_t<decltype(value)>{0}}) {
                            throw pybind11::cast_error{
                                "clip_norm is 0 or negative"};
                        }
                    }
                    catch (const pybind11::cast_error&) {
                        throw std::invalid_argument{fmt::format(
                            "IntegralHOGDescriptor epsilon can be either 0 or "
                            "a positive floating point value but {} was "
                            "given",
                            value)};
                    }
                }
            },
            *epsilon);
    }

    if (!blockNormalizerType_ && (clipNorm_ || epsilon_)) {
        // In case block normalizer arguments were provided, we must ensure
        // these are actually forwarded to the corresponding instance. We
        // assume the default to be L2-Hys.
        blockNormalizerType_ = BlockNormalizerType::L2Hys;
    }

    update();
}

template<class... Args>
[[nodiscard]] DescriptorVariant makeDescriptor(
    [[maybe_unused]] const pybind11::buffer_info& info, Types<> /*unused*/,
    Args&&... args)
    noexcept(std::is_nothrow_constructible_v<Descriptor<double>, Args...>)
{
    // Default descriptor instance if floating point was not matched against
    // supported types.
    return Descriptor<double>(std::forward<Args>(args)...);
}

template<class Scalar, class... Scalars, class... Args>
[[nodiscard]] DescriptorVariant makeDescriptor(
    const pybind11::buffer_info& info, Types<Scalar, Scalars...> /*unused*/,
    Args&&... args)
{
    if (info.format == pybind11::format_descriptor<Scalar>::format()) {
        return Descriptor<Scalar>(std::forward<Args>(args)...);
    }

    return makeDescriptor(info, Types<Scalars...>{},
                          std::forward<Args>(args)...);
}

void IntegralHOGDescriptor::compute(const Rank2Or3Tensor& t,
                                    const pybind11::handle& mask)
{
    // TODO Maybe support dtype parameter?

    pybind11::buffer image = t.buf;
    pybind11::buffer_info info = image.request();

    descriptor_ = makeDescriptor(info, PrecisionTypes{});

    update();

    std::visit(
        [&image, &info, &mask](auto& descriptor) {
            if (mask.is_none()) {
                auto convert = [&descriptor](const auto& t) {
                    descriptor.compute(t);
                };

                bufferToTensor(image, info, convert, SupportedTypes{});
            }
            else {
                pybind11::object getitem;

                if (pybind11::hasattr(mask, "__getitem__")) {
                    // mask can be indexed, e.g., it's a numpy.ndarray
                    getitem = pybind11::getattr(mask, "__getitem__");
                }
                else if (pybind11::hasattr(mask, "__call__")) {
                    // mask is a callable
                    getitem =
                        pybind11::reinterpret_borrow<pybind11::object>(mask);
                }
                else {
                    throw std::invalid_argument{fmt::format(
                        FMT_STRING(
                            "IntegralHOGDescriptor.compute mask must be "
                            "either a callable or provide an indexer in terms "
                            "of a __getitem__ method that accepts a 2-tuple, "
                            "e.g., a numpy.ndarray instance, but a {} object "
                            "was given"),
                        mask.get_type())};
                }

                auto masking = [&getitem](Eigen::DenseIndex i,
                                          Eigen::DenseIndex j) {
                    return pybind11::bool_(getitem(pybind11::make_tuple(i, j)));
                };

                auto convert = [&descriptor, &masking](const auto& t) {
                    descriptor.compute(t, masking);
                };

                bufferToTensor(image, info, convert, SupportedTypes{});
            }
        },
        descriptor_);
}

void IntegralHOGDescriptor::compute(const Rank2Or3TensorPair& dydx,
                                    const pybind11::handle& mask)
{
    pybind11::buffer_info info = dydx.buf1.buf.request();

    descriptor_ = makeDescriptor(info, PrecisionTypes{});

    update();

    std::visit(
        [&dydx, &info, &mask](auto& descriptor) {
            using Scalar = typename std::decay_t<decltype(descriptor)>::Scalar;

            if (mask.is_none()) {
                auto convert = [&descriptor](const auto& dx, const auto& dy) {
                    Eigen::Tensor<Scalar, 3> dxs = dx.template cast<Scalar>();
                    Eigen::Tensor<Scalar, 3> dys = dy.template cast<Scalar>();

                    descriptor.compute(dxs, dys, nullptr);
                };

                bufferToTensor(dydx, info, convert, SupportedTypes{});
            }
            else {
                pybind11::object getitem;

                if (pybind11::hasattr(mask, "__getitem__")) {
                    // mask can be indexed, e.g., it's a numpy.ndarray
                    getitem = pybind11::getattr(mask, "__getitem__");
                }
                else if (pybind11::hasattr(mask, "__call__")) {
                    // mask is a callable
                    getitem =
                        pybind11::reinterpret_borrow<pybind11::object>(mask);
                }
                else {
                    throw std::invalid_argument{fmt::format(
                        FMT_STRING(
                            "IntegralHOGDescriptor.compute mask must be "
                            "either a callable or provide an indexer in terms "
                            "of a __getitem__ method that accepts a 2-tuple, "
                            "e.g., a numpy.ndarray instance, but a {} object "
                            "was given"),
                        mask.get_type())};
                }

                auto masking = [&getitem](Eigen::DenseIndex i,
                                          Eigen::DenseIndex j) {
                    return pybind11::bool_(getitem(pybind11::make_tuple(i, j)));
                };

                auto convert = [&descriptor, &masking](const auto& dx,
                                                       const auto& dy) {
                    Eigen::Tensor<Scalar, 3> dxs = dx.template cast<Scalar>();
                    Eigen::Tensor<Scalar, 3> dys = dy.template cast<Scalar>();

                    descriptor.compute(dxs, dys, masking);
                };

                bufferToTensor(dydx, info, convert, SupportedTypes{});
            }
        },
        descriptor_);
}

pybind11::object IntegralHOGDescriptor::features() const
{
    return isEmpty() ? pybind11::none{}
                     : std::visit(
                           [](auto& descriptor) {
                               return pybind11::cast(descriptor.features());
                           },
                           descriptor_);
}

pybind11::object IntegralHOGDescriptor::featuresROI(const cv::Rect& rect) const
{
    return isEmpty() ? pybind11::none{}
           : rect.area() == 0
               ? pybind11::array{}
               : std::visit(
                     [&rect](auto& descriptor) {
                         return pybind11::cast(descriptor.features(rect));
                     },
                     descriptor_);
}

pybind11::object IntegralHOGDescriptor::featuresROIs(
    const pybind11::iterable& rects) const
{
    auto extract = [&rects](auto& descriptor) {
        using Scalar = typename std::decay_t<decltype(descriptor)>::Scalar;
        using Tensor = std::decay_t<decltype(
            descriptor.features(std::declval<cv::Rect>()))>;
        using Dimensions =
            std::decay_t<decltype(std::declval<Tensor>().dimensions())>;
        constexpr auto NumDimensions = Tensor::NumDimensions;

        const std::size_t n = pybind11::len(rects);
        Eigen::Tensor<Scalar, NumDimensions + 1> features;

        // TODO Replace by C++20 ranges
        auto idxs =
            boost::make_iterator_range(rects.begin(), rects.end()) |
            boost::adaptors::transformed([](const pybind11::handle& rect) {
                return pybind11::cast<cv::Rect>(rect);
            }) |
            boost::adaptors::indexed();

        auto first = idxs.begin();
        Dimensions one;
        cv::Rect firstBounds;

        if (n > 0) {
            // Allocate memory once the bounds of the first element are known.
            // We cannot perform the allocation within the for_each lambda
            // because it will be possibly executed multiple times at the same
            // time causing in a race condition. Therefore, we process the first
            // element independently from the remaining ones.
            firstBounds = first->value();
            auto X = descriptor.features(firstBounds);
            // Store the dimensions of a single tensor to ensure the bounds
            // produce compatible tensors
            one = X.dimensions();
            std::array<Eigen::DenseIndex,
                       static_cast<std::size_t>(NumDimensions) + 1>
                dims;
            dims.front() = static_cast<Eigen::DenseIndex>(n);
            std::copy(X.dimensions().begin(), X.dimensions().end(),
                      std::next(dims.begin()));

            features.resize(dims);
            features.template chip<0>(
                static_cast<Eigen::DenseIndex>(first->index())) = X;

            ++first;
        }

        auto assign = [&one, &firstBounds, &features, &descriptor](auto value) {
            auto X = descriptor.features(value.value());

            if (X.dimensions() != one) {
                throw pybind11::value_error{fmt::format(
                    "IntegralHOGDescriptor extraction of features from "
                    "multiple regions requires all bounds to be of the same "
                    "dimensions. however, the bounds at index 0 are different "
                    "from those at index {} ({} vs. {})",
                    value.index(), pybind11::cast(firstBounds),
                    pybind11::cast(value.value()))};
            }

            features.template chip<0>(
                static_cast<Eigen::DenseIndex>(value.index())) = X;
        };

        // Process the remaining bounds
        std::for_each(
#ifdef HAVE_EXECUTION
            std::execution::par,
#endif // HAVE_EXECUTION
            first, idxs.end(), assign);

        return pybind11::cast(features);
    };

    return isEmpty() ? pybind11::none{} : std::visit(extract, descriptor_);
}

std::tuple<int, int> IntegralHOGDescriptor::cellSize() const
{
    return std::visit(
        [](auto& descriptor) {
            const auto& value = descriptor.cellSize();
            return std::make_tuple(value.x(), value.y());
        },
        descriptor_);
}

std::tuple<int, int> IntegralHOGDescriptor::blockSize() const
{
    return std::visit(
        [](auto& descriptor) {
            const auto& value = descriptor.blockSize();
            return std::make_tuple(value.x(), value.y());
        },
        descriptor_);
}

std::tuple<int, int> IntegralHOGDescriptor::blockStride() const
{
    return std::visit(
        [](auto& descriptor) {
            const auto& value = descriptor.blockStride();
            return std::make_tuple(value.x(), value.y());
        },
        descriptor_);
}

Eigen::DenseIndex IntegralHOGDescriptor::numBins() const
{
    return std::visit([](auto& descriptor) { return descriptor.numBins(); },
                      descriptor_);
}

pybind11::object IntegralHOGDescriptor::histogram() const
{
    return isEmpty() ? pybind11::none{}
                     : std::visit(
                           [](auto& descriptor) {
                               return pybind11::cast(descriptor.histogram());
                           },
                           descriptor_);
}

void IntegralHOGDescriptor::update()
{
    std::visit(
        [this](auto& descriptor) {
            if (cellSize_) {
                descriptor.setCellSize(*cellSize_);
            }

            if (blockSize_) {
                descriptor.setBlockSize(*blockSize_);
            }

            if (blockStride_) {
                descriptor.setBlockStride(*blockStride_);
            }

            if (numBins_) {
                descriptor.setNumBins(*numBins_);
            }

            using Descriptor = std::decay_t<decltype(descriptor)>;

            if (magnitudeType_) {
                typename Descriptor::Magnitude magnitude{*magnitudeType_};
                descriptor.setMagnitude(std::move(magnitude));
            }

            if (binningType_) {
                typename Descriptor::Binning binning{*binningType_};
                descriptor.setBinning(std::move(binning));
            }

            if (blockNormalizerType_) {
                auto optionalNumber =
                    [](const auto& value) -> std::optional<pybind11::float_> {
                    return pybind11::cast<pybind11::float_>(value);
                };

                std::optional<pybind11::float_> clipNorm =
                    !clipNorm_ ? std::nullopt
                               : std::visit(optionalNumber, *clipNorm_);
                std::optional<pybind11::float_> epsilon =
                    !epsilon_ ? std::nullopt
                              : std::visit(optionalNumber, *epsilon_);

                typename Descriptor::BlockNormalizer blockNormalizer{
                    *blockNormalizerType_, clipNorm, epsilon};
                descriptor.setBlockNormalizer(std::move(blockNormalizer));
            }
        },
        descriptor_);
}

BinningType IntegralHOGDescriptor::binning() const
{
    return std::visit(
        [](const auto& descriptor) { return descriptor.binning().type(); },
        descriptor_);
}

BlockNormalizerType IntegralHOGDescriptor::blockNormalizer() const
{
    return std::visit(
        [](const auto& descriptor) {
            return descriptor.blockNormalizer().type();
        },
        descriptor_);
}

MagnitudeType IntegralHOGDescriptor::magnitude() const
{
    return std::visit(
        [](const auto& descriptor) { return descriptor.magnitude().type(); },
        descriptor_);
}

bool IntegralHOGDescriptor::isEmpty() const noexcept
{
    return std::visit(
        [](const auto& descriptor) { return descriptor.isEmpty(); },
        descriptor_);
}

pybind11::object IntegralHOGDescriptor::clipNorm() const noexcept
{
    pybind11::object result = std::visit(
        [](const auto& descriptor) {
            return descriptor.blockNormalizer().clip();
        },
        descriptor_);

    return !result.is_none() ? result
           : clipNorm_       ? std::visit(
                                   [](const auto& value) {
                                 return pybind11::cast<pybind11::object>(value);
                                   },
                                   *clipNorm_)
                       : pybind11::none{};
}

pybind11::object IntegralHOGDescriptor::epsilon() const noexcept
{
    // Every block normalizer provides regularization; no additional checks
    // required unless a different block normalizer is implemented.
    return std::visit(
        [](const auto& descriptor) {
            return descriptor.blockNormalizer().epsilon();
        },
        descriptor_);
}

IntegralHOGDescriptor::operator bool() const noexcept
{
    return !isEmpty();
}
