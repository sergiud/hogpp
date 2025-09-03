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

#ifndef HOGPP_INTEGRALHOGDESCRIPTOR_HPP
#define HOGPP_INTEGRALHOGDESCRIPTOR_HPP

#include <unsupported/Eigen/CXX11/Tensor>

#include <cassert>
#include <cmath>
#include <utility>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <hogpp/assume.hpp>
#include <hogpp/bounds.hpp>
#include <hogpp/gradient.hpp>
#include <hogpp/gradientmagnitude.hpp>
#include <hogpp/integralhistogram.hpp>
#include <hogpp/l2hys.hpp>
#include <hogpp/prefix.hpp>
#include <hogpp/unsignedgradient.hpp>

namespace hogpp {

template<class T>
struct IntegralHOGDescriptorTraits;

// We want to allow to specify 'void' as Gradient functor type. Since 'void'
// cannot be instantiated, we need to provide different definitions of
// IntegralHOGDescriptor with a different set of member variables depending on
// the specified Gradient type.
//
// For this purpose, we separate member variables definition by moving them into
// a (non-virtual) base class which can be partially specialized. This
// IntegralHOGDescriptorBase is partially specialized with respect to the
// Gradient type. A general definition of the class defines a minimal set of
// member variables without a Gradient member variable. Another variant of the
// class is partially specialized with respect to the Gradient type. If the
// latter is non-void, an additional Gradient member variable is defined.
//
// In case Gradient is void, the user is required to explicitly provide the
// gradients and IntegralHOGDescriptor::compute overload that accepts an image
// tensor is disabled.

template<class Derived, class E = void>
class IntegralHOGDescriptorBase
{
public:
    using Magnitude = typename IntegralHOGDescriptorTraits<Derived>::Magnitude;
    using Binning = typename IntegralHOGDescriptorTraits<Derived>::Binning;
    using BlockNormalizer =
        typename IntegralHOGDescriptorTraits<Derived>::BlockNormalizer;

    [[nodiscard]] explicit IntegralHOGDescriptorBase(
        Magnitude magnitude = Magnitude{}, Binning binning = Binning{},
        BlockNormalizer normalization = BlockNormalizer{})
        : vote_{std::move(magnitude)}
        , binning_{std::move(binning)}
        , normalize_{std::move(normalization)}
    {
    }

    void setBinning(Binning value)
    {
        binning_ = std::move(value);
    }

    [[nodiscard]] const Binning& binning() const noexcept
    {
        return binning_;
    }

    [[nodiscard]] Binning& binning() noexcept
    {
        return binning_;
    }

    void setBlockNormalizer(BlockNormalizer value)
    {
        normalize_ = std::move(value);
    }

    [[nodiscard]] const BlockNormalizer& blockNormalizer() const noexcept
    {
        return normalize_;
    }

    [[nodiscard]] BlockNormalizer& blockNormalizer() noexcept
    {
        return normalize_;
    }

    void setMagnitude(Magnitude value)
    {
        vote_ = std::move(value);
    }

    [[nodiscard]] const Magnitude& magnitude() const noexcept
    {
        return vote_;
    }

    [[nodiscard]] Magnitude& magnitude() noexcept
    {
        return vote_;
    }

protected:
    Magnitude vote_;
    Binning binning_;
    BlockNormalizer normalize_;
};

template<class Derived>
class IntegralHOGDescriptorBase
    // clang-format off
<
      Derived
    , std::void_t
    <
        typename IntegralHOGDescriptorTraits<Derived>::Gradient
    >
>
    // clang-format on
    : public IntegralHOGDescriptorBase<Derived, Derived>
{
public:
    using Gradient = typename IntegralHOGDescriptorTraits<Derived>::Gradient;

    template<class... Args>
    [[nodiscard]] explicit IntegralHOGDescriptorBase(
        Gradient gradient = Gradient{}, Args&&... args)
        : IntegralHOGDescriptorBase<Derived, Derived>{std::forward<Args>(
              args)...}
        , gradient_{std::move(gradient)}
    {
    }

protected:
    Gradient gradient_;
};

// clang-format off
template
<
      class T
    , class GradientType = Gradient<T>
    , class MagnitudeType = GradientMagnitude<T>
    , class BinningType = UnsignedGradient<T>
    , class BlockNormalizerType = L2Hys<T>
>
// clang-format on
class IntegralHOGDescriptor
    : public IntegralHOGDescriptorBase
      // clang-format off
<
    IntegralHOGDescriptor
    <
          T
        , GradientType
        , MagnitudeType
        , BinningType
        , BlockNormalizerType
    >
>
// clang-format on
{
    // clang-format off
    using Base = IntegralHOGDescriptorBase
    <
        IntegralHOGDescriptor
        <
              T
            , GradientType
            , MagnitudeType
            , BinningType
            , BlockNormalizerType
        >
    >
    ;
    // clang-format on
public:
    using Scalar = T;
    using Tensor5 = Eigen::Tensor<Scalar, 5>;
    using Gradient = GradientType;
    using Magnitude = MagnitudeType;
    using Binning = BinningType;
    using BlockNormalizer = BlockNormalizerType;

    [[nodiscard]] IntegralHOGDescriptor() = default;

    // clang-format off
    template
    <
          class ...Args
        , class B = Base
        , std::enable_if_t
        <
            std::is_constructible_v<Base, Args...>
        >* = nullptr
    >
    // clang-format on
    [[nodiscard]] explicit IntegralHOGDescriptor(Args&&... args)
        : Base{std::forward<Args>(args)...}
    {
    }

    // clang-format off
    template
    <
          class U
        , int DataLayout
        , class Masking = std::nullptr_t
        , class G = GradientType
        // Enable overload only if Gradient is not void
        , std::enable_if_t<!std::is_void_v<G> >* = nullptr
    >
    // clang-format on
    void compute(const Eigen::Tensor<U, 3, DataLayout>& image,
                 Masking&& masked = nullptr)
    {
        Eigen::Tensor<Scalar, 3, DataLayout> dxs;
        Eigen::Tensor<Scalar, 3, DataLayout> dys;

        std::tie(dxs, dys) = this->gradient_(image);

        assert(dxs.dimension(0) == image.dimension(0));
        assert(dxs.dimension(1) == image.dimension(1));
        assert(dxs.dimension(2) == image.dimension(2));

        assert(dys.dimension(0) == image.dimension(0));
        assert(dys.dimension(1) == image.dimension(1));
        assert(dys.dimension(2) == image.dimension(2));

        compute(dxs, dys, std::forward<Masking>(masked));
    }

    template<int DataLayout, class Masking = std::nullptr_t>
    void compute(const Eigen::Tensor<Scalar, 3, DataLayout>& dxs,
                 const Eigen::Tensor<Scalar, 3, DataLayout>& dys,
                 Masking&& masked)
    {
        Tensor3 mags =
            this->vote_(dxs, dys).swap_layout().shuffle(std::array{2, 1, 0});

        if (mags.size() == 0) {
            return; // Nothing to do
        }

        const Eigen::Array2i dims{mags.dimension(0), mags.dimension(1)};
        histogram_.resize(std::make_tuple(dims.x(), dims.y()), bins_);

        const Eigen::Tensor<Eigen::DenseIndex, 2, Eigen::RowMajor>& k =
            mags.argmax(2);

        const auto scale = static_cast<Scalar>(bins_ - 1);

        histogram_.scan(
            [this, &k, &dxs, &dys, &mags, scale, &masked](
                Eigen::TensorRef<Eigen::Tensor<Scalar, 1, DataLayout>> bins,
                const auto& ij) {
                (void)masked; // Avoid error: lambda capture 'masked' is not
                              // used [-Werror,-Wunused-lambda-capture] on
                              // AppleClang
                if constexpr (!std::is_null_pointer_v<Masking>) {
                    if (std::apply(masked, ij)) {
                        // Skip masked out pixels
                        return;
                    }
                }

                // Select a channel with the maximum magnitude
                Eigen::DenseIndex kk = std::apply(k, ij);
                const auto ijk = std::tuple_cat(ij, std::make_tuple(kk));

                Scalar mag = std::apply(mags, ijk);

                using std::fpclassify;

                if (fpclassify(mag) == FP_ZERO) {
                    // No gradient; take a shortcut
                    return;
                }

                // The gradient magnitude cannot be negative (or zero at this
                // point)
                HOGPP_ASSUME(mag > 0);

                Scalar dx = std::apply(dxs, ijk);
                Scalar dy = std::apply(dys, ijk);

                // Gradient binning
                Scalar weight = this->binning_(dx, dy);

                HOGPP_ASSUME(weight >= 0 && weight <= 1);

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

                HOGPP_ASSUME(alpha >= 0 && alpha <= 1);

                const auto bin1 = static_cast<Eigen::DenseIndex>(lower);
                const auto bin2 = static_cast<Eigen::DenseIndex>(upper);

                HOGPP_ASSUME(bin1 <= bin2);

                // Distribute weighted values to neighboring bins.
                Scalar& value1 = bins.coeffRef(bin1);
                Scalar& value2 = bins.coeffRef(bin2);

                using std::fma;
                // The bin closest to the target orientation obtains
                // proportionally a higher magnitude.
                value1 = fma(1 - alpha, mag, value1);
                value2 = fma(alpha, mag, value2);
            });
    }

    [[nodiscard]] Tensor5 features() const
    {
        if (histogram_.isEmpty()) {
            return Tensor5{};
        }

        return features(Bounds{0, 0,
                               static_cast<int>(histogram().dimension(1) - 1),
                               static_cast<int>(histogram().dimension(0) - 1)});
    }

    [[nodiscard]] Tensor5 features(const Bounds& roi) const
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

        const auto dimX = histogram().dimension(0) - 1;
        const auto dimY = histogram().dimension(1) - 1;
        const Eigen::Array2i offset{roi.y, roi.x};

        if (offset.x() < 0 || dimX - offset.x() < 0) {
            throw std::invalid_argument{fmt::format(
                "IntegralHOGDescriptor features cannot be extracted from a "
                "region outside of the input domain specified by the row {}",
                offset.x())};
        }

        if (offset.y() < 0 || dimY - offset.y() < 0) {
            throw std::invalid_argument{fmt::format(
                "IntegralHOGDescriptor features cannot be extracted from a "
                "region outside of the input domain specified by the column {}",
                offset.y())};
        }

        const Eigen::Array2i br = offset + dims;

        if (br.x() > dimX) {
            throw std::invalid_argument{fmt::format(
                "IntegralHOGDescriptor features cannot be extracted from a "
                "region larger than the input domain with the bottom row {}",
                br.x() - 1)};
        }

        if (br.y() > dimY) {
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
                 histogram_.histogram().dimension(2));
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

                        // Extract the histogram from the integral histogram
                        // as an intersection.
                        const Eigen::Tensor<Scalar, 1> h = histogram_.intersect(
                            std::make_tuple(offset1.x(), offset1.y()),
                            std::make_tuple(offset2.x(), offset2.y()));

                        X.template chip<0>(i)
                            .template chip<0>(j)
                            .template chip<0>(k)
                            .template chip<0>(l) = h;
                    }
                }

                // Block normalization
                auto block = X.template chip<0>(i).template chip<0>(j);

                this->normalize_(block);
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
        return histogram_.histogram();
    }

    template<class Derived>
    void setHistogram(
        const Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors>& value)
    {
        histogram_.setHistogram(value);
    }

    [[nodiscard]] bool isEmpty() const noexcept
    {
        return histogram_.isEmpty();
    }

private:
    using Tensor2 = Eigen::Tensor<Scalar, 2, Eigen::RowMajor>;
    using Tensor3 = Eigen::Tensor<Scalar, 3, Eigen::RowMajor>;

    IntegralHistogram<Scalar, 2, 1> histogram_;
    Eigen::DenseIndex bins_ = 9;
    Eigen::Array2i cellSize_{8, 8};
    Eigen::Array2i blockSize_ = cellSize_ * 2;
    Eigen::Array2i blockStride_ = cellSize_;
};

template<class BlockNormalizerType>
explicit IntegralHOGDescriptor(BlockNormalizerType)
    -> IntegralHOGDescriptor<typename BlockNormalizerType::Scalar>;

// clang-format off
template
<
      class T
    , class GradientType
    , class MagnitudeType
    , class BinningType
    , class BlockNormalizerType
>
// clang-format on
struct IntegralHOGDescriptorTraits
    // clang-format off
<
    IntegralHOGDescriptor
    <
        T
        , GradientType
        , MagnitudeType
        , BinningType
        , BlockNormalizerType
    >
>
// clang-format on
{
    using Gradient = GradientType;
    using Magnitude = MagnitudeType;
    using Binning = BinningType;
    using BlockNormalizer = BlockNormalizerType;
};

// clang-format off
template
<
      class T
    , class MagnitudeType
    , class BinningType
    , class BlockNormalizerType
>
// clang-format on
struct IntegralHOGDescriptorTraits
    // clang-format off
<
    IntegralHOGDescriptor
    <
        T
        , void
        , MagnitudeType
        , BinningType
        , BlockNormalizerType
    >
>
// clang-format on
{
    using Magnitude = MagnitudeType;
    using Binning = BinningType;
    using BlockNormalizer = BlockNormalizerType;
};

} // namespace hogpp

#include <hogpp/suffix.hpp>

#endif // HOGPP_INTEGRALHOGDESCRIPTOR_HPP
