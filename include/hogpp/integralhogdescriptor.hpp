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

#ifndef HOGPP_INTEGRALHOGDESCRIPTOR_HPP
#define HOGPP_INTEGRALHOGDESCRIPTOR_HPP

#include <unsupported/Eigen/CXX11/Tensor>

#include <cassert>
#include <cmath>
#include <utility>

#include <fmt/format.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <hogpp/gradientmagnitude.hpp>
#include <hogpp/l2hys.hpp>
#include <hogpp/unsignedgradient.hpp>

namespace hogpp {

// clang-format off
template
<
      class T
    , class MagnitudeType = GradientMagnitude<T>
    , class BinningType = UnsignedGradient<T>
    , class BlockNormalizerType = L2Hys<T>
>
// clang-format on
class IntegralHOGDescriptor
{
public:
    using Scalar = T;
    using Tensor5 = Eigen::Tensor<Scalar, 5>;
    using Magnitude = MagnitudeType;
    using Binning = BinningType;
    using BlockNormalizer = BlockNormalizerType;

    [[nodiscard]] explicit IntegralHOGDescriptor(
        MagnitudeType magnitude = MagnitudeType{},
        BinningType binning = BinningType{},
        BlockNormalizerType normalization = BlockNormalizerType{})
        : vote_{std::move(magnitude)}
        , binning_{std::move(binning)}
        , normalize_{std::move(normalization)}
    {
    }

    template<class InputIterator, class Masking = std::nullptr_t>
    void compute(InputIterator first, InputIterator last,
                 Masking&& masked = nullptr)
    {
        const cv::Matx<Scalar, 1, 3> kx{Scalar(-1), Scalar(0), Scalar(1)};
        const cv::Matx<Scalar, 3, 1> ky = kx.t();

        const cv::Mat& image = *first;
        const Eigen::Array2i dims{image.rows, image.cols};

        std::ptrdiff_t n = std::distance(first, last);
        Tensor3 dxs{static_cast<Eigen::DenseIndex>(n), dims.x(), dims.y()};
        Tensor3 dys{dxs.dimensions()};

        cv::Mat dx;
        cv::Mat dy;

        const cv::Point anchor{-1, -1};
        const cv::BorderTypes borderMode = cv::BORDER_REFLECT101;

        for (InputIterator start = first; first != last; ++first) {
            std::ptrdiff_t i = std::distance(start, first);
            const cv::Mat& channel = *first;

            cv::filter2D(channel, dx, cv::DataDepth<Scalar>::value, kx, anchor,
                         0, borderMode);
            cv::filter2D(channel, dy, cv::DataDepth<Scalar>::value, ky, anchor,
                         0, borderMode);

            dxs.template chip<0>(static_cast<Eigen::DenseIndex>(i)) =
                Eigen::TensorMap<const Tensor2>{dx.ptr<Scalar>(), dx.rows,
                                                dx.cols};
            dys.template chip<0>(static_cast<Eigen::DenseIndex>(i)) =
                Eigen::TensorMap<const Tensor2>{dy.ptr<Scalar>(), dy.rows,
                                                dy.cols};
        }

        Tensor3 mags = vote_(dxs, dys);

        Eigen::Array2i dims1 = dims + 1;

        histogram_.resize(dims1.x(), dims1.y(), bins_);
        histogram_.setZero();

        const Eigen::Tensor<Eigen::DenseIndex, 2, Eigen::RowMajor>& k =
            mags.argmax(0);

        const auto scale = static_cast<Scalar>(bins_ - 1);

        // Wavefront scan
        for (Eigen::DenseIndex i = 0; i < dims.x(); ++i) {
            for (Eigen::DenseIndex j = 0; j < dims.y(); ++j) {
                // Histogram propagation
                const auto& a =
                    histogram_.template chip<0>(i).template chip<0>(j + 1);
                const auto& b =
                    histogram_.template chip<0>(i + 1).template chip<0>(j);
                const auto& c =
                    histogram_.template chip<0>(i).template chip<0>(j);

                histogram_.template chip<0>(i + 1).template chip<0>(j + 1) =
                    a + b - c;

                if constexpr (!std::is_null_pointer_v<Masking>) {
                    if (masked(i, j)) {
                        // Skip masked out pixels
                        continue;
                    }
                }

                // Select a channel with the maximum magnitude
                Eigen::DenseIndex kk = k(i, j);
                Scalar mag = mags(kk, i, j);

                assert(mag >= 0);

                if (!(mag > 0)) {
                    // No gradient; take a shortcut
                    continue;
                }

                Scalar dx = dxs(kk, i, j);
                Scalar dy = dys(kk, i, j);

                // Gradient binning
                Scalar weight = binning_(dx, dy);

                assert(weight >= 0 && weight <= 1);

                using std::floor;
                using std::min;

                // Uniformly distribute the weight across [0, 1, ..., n - 1]
                // where n is the number of bins.
                Scalar center = weight * scale;
                Scalar lower = floor(center);
                // The upper bin will overflow n-1 iff the weight is 1. However,
                // in that case the last bin obtains all the votes anyway.
                Scalar upper = min(lower + 1, scale);

                // Soft binning: distribute magnitude to neighboring bins
                // proportionally to the gradient distance to each center.
                Scalar alpha = center - lower;

                assert(alpha >= 0 && alpha <= 1);

                const auto bin1 = static_cast<Eigen::DenseIndex>(lower);
                const auto bin2 = static_cast<Eigen::DenseIndex>(upper);

                assert(bin1 <= bin2);

                // The bin closest to the target orientation obtains
                // proportionally a higher magnitude.
                Scalar weightedMag = alpha * mag;
                // Use an expansion of (1 - alpha) * mag to avoid numerical
                // inaccuracy.
                Scalar value1 = mag - weightedMag;
                Scalar value2 = weightedMag;
                // Distribute weighted values to neighboring bins.
                histogram_(i + 1, j + 1, bin1) += value1;
                histogram_(i + 1, j + 1, bin2) += value2;
            }
        }
    }

    [[nodiscard]] Tensor5 features() const
    {
        if (histogram_.size() == 0) {
            return Tensor5{};
        }

        return features(
            cv::Rect{0, 0, static_cast<int>(histogram_.dimension(1) - 1),
                     static_cast<int>(histogram_.dimension(0) - 1)});
    }

    [[nodiscard]] Tensor5 features(const cv::Rect& roi) const
    {
        if (roi.area() == 0) {
            return Tensor5{};
        }

        const Eigen::Array2i dims{roi.height, roi.width};

        if (dims.x() < 0) {
            throw std::invalid_argument{
                fmt::format("IntegralHOGDescriptor features region row count "
                            "must be positive but is {}",
                            dims.x())};
        }

        if (dims.y() < 0) {
            throw std::invalid_argument{
                fmt::format("IntegralHOGDescriptor features region column "
                            "count must be positive but is {}",
                            dims.y())};
        }

        const Eigen::Array2i offset{roi.y, roi.x};

        if (offset.x() < 0 || dims.x() - offset.x() < 0) {
            throw std::invalid_argument{fmt::format(
                "IntegralHOGDescriptor features cannot be extracted from a "
                "region outside of the input domain specified by the row {}",
                offset.x())};
        }

        if (offset.y() < 0 || dims.y() - offset.x() < 0) {
            throw std::invalid_argument{fmt::format(
                "IntegralHOGDescriptor features cannot be extracted from a "
                "region outside of the input domain specified by the column {}",
                offset.y())};
        }

        const Eigen::Array2i br = offset + dims;

        if (br.x() > histogram_.dimension(0) - 1) {
            throw std::invalid_argument{fmt::format(
                "IntegralHOGDescriptor features cannot be extracted from a "
                "region larger than the input domain with the bottom row {}",
                br.x() - 1)};
        }

        if (br.y() > histogram_.dimension(1) - 1) {
            throw std::invalid_argument{fmt::format(
                "IntegralHOGDescriptor features cannot be extracted from a "
                "region larger than the input domain with the right column {}",
                br.y() - 1)};
        }

        const Eigen::Array2i numBlocks = (dims - blockSize_) / blockStride_ + 1;
        const Eigen::Array2i numCells = blockSize_ / cellSize_;

        // Organize the features in a 5-D tensor (blocks (x,y), cells (x,y) and
        // bins).
        Tensor5 X;

        X.resize(numBlocks.x(), numBlocks.y(), numCells.x(), numCells.y(),
                 histogram_.dimension(2));
        X.setZero();

        for (int i = 0; i < numBlocks.x(); ++i) {
            for (int j = 0; j < numBlocks.y(); ++j) {
                const Eigen::Array2i blockOffset =
                    offset + Eigen::Array2i{i, j} * blockStride_;

                for (int k = 0; k < numCells.x(); ++k) {
                    for (int l = 0; l < numCells.y(); ++l) {
                        const Eigen::Array2i cellOffset{k, l};
                        const Eigen::Array2i& offset1 =
                            blockOffset + cellOffset * cellSize_;
                        const Eigen::Array2i& offset2 = offset1 + cellSize_;

                        const auto& a = histogram_.template chip<0>(offset2.x())
                                            .template chip<0>(offset2.y());
                        const auto& b = histogram_.template chip<0>(offset1.x())
                                            .template chip<0>(offset2.y());
                        const auto& c = histogram_.template chip<0>(offset2.x())
                                            .template chip<0>(offset1.y());
                        const auto& d = histogram_.template chip<0>(offset1.x())
                                            .template chip<0>(offset1.y());

                        // Extract the histogram from the integral histogram
                        // as an intersection.
                        const Eigen::Tensor<Scalar, 1> h = a - b - c + d;

                        X.template chip<0>(i)
                            .template chip<0>(j)
                            .template chip<0>(k)
                            .template chip<0>(l) = h;
                    }
                }

                // Block normalization
                auto block = X.template chip<0>(i).template chip<0>(j);

                normalize_(block);
            }
        }

        return X;
    }

    void setNumBins(Eigen::DenseIndex value)
    {
        if (value <= 0) {
            throw std::invalid_argument{fmt::format(
                "IntegralHOGDescriptor number of histogram bins must be a "
                "positive number but {} was given",
                value)};
        }

        bins_ = value;
    }

    [[nodiscard]] Eigen::DenseIndex numBins() const noexcept
    {
        return bins_;
    }

    void setCellSize(const Eigen::Ref<const Eigen::Array2i>& value)
    {
        if ((value <= 0).any()) {
            throw std::invalid_argument{fmt::format(
                "IntegralHOGDescriptor cell size cannot be zero or negative "
                "but [{}] was given",
                fmt::join(value.data(), value.data() + value.size(), ", "))};
        }

        cellSize_ = value;
    }

    [[nodiscard]] const Eigen::Array2i& cellSize() const noexcept
    {
        return cellSize_;
    }

    void setBlockSize(const Eigen::Ref<const Eigen::Array2i>& value)
    {
        if ((value <= 0).any()) {
            throw std::invalid_argument{fmt::format(
                "IntegralHOGDescriptor block size cannot be zero or negative "
                "but [{}] was given",
                fmt::join(value.data(), value.data() + value.size(), ", "))};
        }

        blockSize_ = value;
    }

    [[nodiscard]] const Eigen::Array2i& blockSize() const noexcept
    {
        return blockSize_;
    }

    void setBlockStride(const Eigen::Ref<const Eigen::Array2i>& value)
    {
        if ((value <= 0).any()) {
            throw std::invalid_argument{fmt::format(
                "IntegralHOGDescriptor block stride cannot be zero or negative "
                "but [{}] was given",
                fmt::join(value.data(), value.data() + value.size(), ", "))};
        }

        blockStride_ = value;
    }

    [[nodiscard]] const Eigen::Array2i& blockStride() const noexcept
    {
        return blockStride_;
    }

    [[nodiscard]] const Eigen::Tensor<Scalar, 3>& histogram() const noexcept
    {
        return histogram_;
    }

    void setBinning(BinningType value)
    {
        binning_ = std::move(value);
    }

    [[nodiscard]] const BinningType& binning() const noexcept
    {
        return binning_;
    }

    [[nodiscard]] BinningType& binning() noexcept
    {
        return binning_;
    }

    void setBlockNormalizer(BlockNormalizerType value)
    {
        normalize_ = std::move(value);
    }

    [[nodiscard]] const BlockNormalizerType& blockNormalizer() const noexcept
    {
        return normalize_;
    }

    [[nodiscard]] BlockNormalizerType& blockNormalizer() noexcept
    {
        return normalize_;
    }

    void setMagnitude(MagnitudeType value)
    {
        vote_ = std::move(value);
    }

    [[nodiscard]] const MagnitudeType& magnitude() const noexcept
    {
        return vote_;
    }

    [[nodiscard]] MagnitudeType& magnitude() noexcept
    {
        return vote_;
    }

    [[nodiscard]] bool isEmpty() const noexcept
    {
        return histogram_.size() == 0;
    }

private:
    using Tensor2 = Eigen::Tensor<Scalar, 2, Eigen::RowMajor>;
    using Tensor3 = Eigen::Tensor<Scalar, 3, Eigen::RowMajor>;

    MagnitudeType vote_;
    BinningType binning_;
    BlockNormalizerType normalize_;
    Eigen::Tensor<Scalar, 3> histogram_;
    Eigen::DenseIndex bins_ = 9;
    Eigen::Array2i cellSize_{8, 8};
    Eigen::Array2i blockSize_ = cellSize_ * 2;
    Eigen::Array2i blockStride_ = cellSize_;
};

template<class BlockNormalizerType>
explicit IntegralHOGDescriptor(BlockNormalizerType)
    -> IntegralHOGDescriptor<typename BlockNormalizerType::Scalar>;

} // namespace hogpp

#endif // HOGPP_INTEGRALHOGDESCRIPTOR_HPP
