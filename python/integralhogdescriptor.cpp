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

#include "integralhogdescriptor.hpp"
#include "type_caster/tensor.hpp"

template<>
struct fmt::formatter<pybind11::str> : formatter<string_view>
{
    template<class FormatContext>
    auto format(const pybind11::str& s, FormatContext& ctx)
    {
        return formatter<string_view>::format(std::string{s}, ctx);
    }
};

template<>
struct fmt::formatter<pybind11::handle> : formatter<pybind11::str>
{
    template<class FormatContext>
    auto format(const pybind11::handle& o, FormatContext& ctx)
    {
        return formatter<pybind11::str>::format(pybind11::repr(o), ctx);
    }
};

namespace {

template<class... T>
[[nodiscard]] static constexpr decltype(auto) possible(T... values)
{
    return std::array<std::string, sizeof...(T)>{
        {pybind11::repr(pybind11::cast(values))...}};
}

} // namespace

IntegralHOGDescriptor::IntegralHOGDescriptor(const pybind11::kwargs& args)
{
    // Prefer keyword arguments over dedicated signature to delegate definition
    // of defaults to the native implementation.

    // Delay setting parameters until we know the floating point type we
    // will be working with.
    for (auto&& [key, value] : args) {
        const std::string& name = pybind11::cast<std::string>(key);

        if (name == "cell_size") {
            cellSize_ = pybind11::cast<Eigen::Array2i>(value);
        }
        else if (name == "block_size") {
            blockSize_ = pybind11::cast<Eigen::Array2i>(value);
        }
        else if (name == "block_stride") {
            blockStride_ = pybind11::cast<Eigen::Array2i>(value);
        }
        else if (name == "num_bins") {
            numBins_ = pybind11::cast<int>(value);
        }
        else if (name == "magnitude") {
            try {
                magnitudeType_ = pybind11::cast<MagnitudeType>(value);
            }
            catch (const pybind11::cast_error&) {
                throw std::invalid_argument{fmt::format(
                    "IntegralHOGDescriptor magnitude can be one "
                    "of the values {} "
                    "but the unknown value {} was given",
                    fmt::join(
                        possible(MagnitudeType::Identity, MagnitudeType::Square,
                                 MagnitudeType::Sqrt),
                        ", "),
                    value)};
            }
        }
        else if (name == "binning") {
            try {
                binningType_ = pybind11::cast<BinningType>(value);
            }
            catch (const pybind11::cast_error&) {
                throw std::invalid_argument{fmt::format(
                    "IntegralHOGDescriptor binning can be one of the values {} "
                    "but the unknown value {} was given",
                    fmt::join(
                        possible(BinningType::Signed, BinningType::Unsigned),
                        ", "),
                    value)};
            }
        }
        else if (name == "block_norm") {
            try {
                blockNormalizerType_ =
                    pybind11::cast<BlockNormalizerType>(value);
            }
            catch (const pybind11::cast_error&) {
                throw std::invalid_argument{
                    fmt::format("IntegralHOGDescriptor block_norm can be one "
                                "of the values {} "
                                "but the unknown value {} was given",
                                fmt::join(possible(BlockNormalizerType::L1,
                                                   BlockNormalizerType::L2,
                                                   BlockNormalizerType::L2Hys,
                                                   BlockNormalizerType::L1sqrt),
                                          ", "),
                                value)};
            }
        }
        else if (name == "clip_norm") {
            // Allow None
            if (!value.is_none()) {
                try {
                    auto clipNorm = pybind11::cast<pybind11::float_>(value);

                    if (!(clipNorm > pybind11::float_{0.0f})) {
                        throw pybind11::cast_error{
                            "clip_norm is 0 or negative"};
                    }

                    clipNorm_ = clipNorm;
                }
                catch (const pybind11::cast_error&) {
                    throw std::invalid_argument{
                        fmt::format("IntegralHOGDescriptor clip_norm can only "
                                    "be a positive "
                                    "floating point value but {} was given",
                                    value)};
                }
            }
        }
        else if (name == "epsilon") {
            // Allow None
            if (!value.is_none()) {
                try {
                    auto epsilon = pybind11::cast<pybind11::float_>(value);

                    if (epsilon < pybind11::float_{0.0f}) {
                        throw pybind11::cast_error{"epsilon is negative"};
                    }

                    epsilon_ = epsilon;
                }
                catch (const pybind11::cast_error&) {
                    throw std::invalid_argument{fmt::format(
                        "IntegralHOGDescriptor epsilon can be either 0 or a "
                        "positive floating point value but {} was given",
                        value)};
                }
            }
        }
        else {
            throw std::invalid_argument{fmt::format(
                "unsupported IntegralHOGDescriptor parameter \"{}\"", name)};
        }
    }

    if (!blockNormalizerType_ && (clipNorm_ || epsilon_)) {
        // In case block normalizer arguments were provided, we must ensure
        // these are actually forwarded to the corresponding instance. We assume
        // the default to be L2-Hys.
        blockNormalizerType_ = BlockNormalizerType::L2Hys;
    }

    update();
}

void IntegralHOGDescriptor::compute(const cv::Mat& image)
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
        [&channels](auto& descriptor) {
            descriptor.compute(channels.begin(), channels.end());
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

            if (magnitudeType_) {
                typename std::decay_t<decltype(descriptor)>::Magnitude
                    magnitude{*magnitudeType_};
                descriptor.setMagnitude(std::move(magnitude));
            }

            if (binningType_) {
                typename std::decay_t<decltype(descriptor)>::Binning binning{
                    *binningType_};
                descriptor.setBinning(std::move(binning));
            }

            if (blockNormalizerType_) {
                typename std::decay_t<decltype(descriptor)>::BlockNormalizer
                    blockNormalizer{*blockNormalizerType_, clipNorm_, epsilon_};
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
           : clipNorm_       ? pybind11::object{*clipNorm_}
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
