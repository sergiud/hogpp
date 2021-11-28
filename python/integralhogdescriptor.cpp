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

#include <fmt/format.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include "formatter.hpp"
#include "integralhogdescriptor.hpp"
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

template<class... T>
[[nodiscard]] constexpr decltype(auto) possible(T... values)
{
    return std::array<std::string, sizeof...(T)>{
        {pybind11::repr(pybind11::cast(values))...}};
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
    // Prefer keyword arguments over dedicated signature to delegate definition
    // of defaults to the native implementation.

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

void IntegralHOGDescriptor::compute(const cv::Mat& image,
                                    const pybind11::handle& mask)
{
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    // Default to 64 bit float
    switch (image.depth()) {
        case CV_32F:
            descriptor_ = Descriptor<float>{};
            break;
        default:
            descriptor_ = Descriptor<double>{};
            break;
    }

    update();

    std::visit(
        [&channels, &mask](auto& descriptor) {
            if (mask.is_none()) {
                descriptor.compute(channels.begin(), channels.end());
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

                descriptor.compute(channels.begin(), channels.end(), masking);
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
            using Scalar = typename Descriptor::Scalar;

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
