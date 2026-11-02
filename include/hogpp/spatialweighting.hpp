//
// HOGpp - Fast histogram of oriented gradients computation using integral
// histograms
//
// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

#ifndef HOGPP_SPATIALWEIGHTING_HPP
#define HOGPP_SPATIALWEIGHTING_HPP

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <hogpp/integralhistogram.hpp>
#include <hogpp/multiplyadd.hpp>

namespace hogpp {

//! Block layout the spatial weighting depends on.
struct BlockGeometry
{
    Eigen::Array2i cellSize = Eigen::Array2i::Zero();
    Eigen::Array2i blockSize = Eigen::Array2i::Zero();
    Eigen::Array2i blockStride = Eigen::Array2i::Zero();
};

[[nodiscard]] inline bool operator==(const BlockGeometry& lhs,
                                     const BlockGeometry& rhs) noexcept
{
    return (lhs.cellSize == rhs.cellSize).all() &&
           (lhs.blockSize == rhs.blockSize).all() &&
           (lhs.blockStride == rhs.blockStride).all();
}

/**
 * @brief Extracts cell histograms without any spatial weighting.
 *
 * A spatial weighting policy provides a @c State that the descriptor keeps
 * alongside the integral histogram, a @c prepare() function the descriptor
 * invokes after the integral histogram was replaced, and a @c cellQuery()
 * function that returns a callable extracting a single cell histogram of a
 * block.
 *
 * The callable passed to @c prepare() as @p visitVotes is invoked as
 * @c visitVotes(ij,emit) and calls @c emit(bin,value) for every vote of the
 * pixel @c ij.
 */
struct NoSpatialWeighting
{
    template<class Scalar>
    struct State
    {
    };

    template<class Scalar, class VisitVotes>
    constexpr void prepare(State<Scalar>& /*state*/,
                           const BlockGeometry& /*geometry*/,
                           const Eigen::Array2i& /*dims*/,
                           Eigen::DenseIndex /*bins*/,
                           VisitVotes&& /*visitVotes*/) const noexcept
    {
    }

    template<class Scalar, Eigen::DenseIndex K, int Options>
    [[nodiscard]] auto cellQuery(
        const State<Scalar>& /*state*/,
        const IntegralHistogram<Scalar, 2, K, Options>& histogram,
        const BlockGeometry& /*geometry*/) const noexcept
    {
        return [&histogram](auto&& cell, const Eigen::Array2i& /*index*/,
                            const Eigen::Array2i& first,
                            const Eigen::Array2i& last) {
            cell = histogram.intersect(std::make_tuple(first.x(), first.y()),
                                       std::make_tuple(last.x(), last.y()));
        };
    }
};

namespace detail {

/**
 * @brief Computes the separable weighted sum
 * @f$\sum_{q=0}^{c-1} w_c(q) \sum_{p=0}^{r-1} w_r(p) v(p, q)@f$.
 *
 * Each inner sum accumulates over @f$p@f$ in ascending order, and the outer
 * sum over @f$q@f$, so the rounding does not depend on how the sums are
 * scheduled.
 *
 * @param rows Number of terms @f$r@f$ of each inner sum.
 * @param cols Number of inner sums @f$c@f$.
 * @param rowWeight Weight @f$w_r(p)@f$ of the term @f$p@f$ of each inner sum.
 * @param colWeight Weight @f$w_c(q)@f$ of the inner sum @f$q@f$.
 * @param value Value @f$v(p, q)@f$.
 */
template<class S, class RowWeight, class ColWeight, class Value>
[[nodiscard]] S separableSum(int rows, int cols, RowWeight&& rowWeight,
                             ColWeight&& colWeight, Value&& value)
{
    // Each inner sum is a chain of dependent multiply-adds whose latency, not
    // its throughput, bounds a single sum. Accumulating several inner sums at
    // once interleaves independent chains. This matters most with fused
    // multiply-adds, whose latency exceeds that of an addition.
    constexpr int Lanes = 4;

    S total{0};
    int q = 0;

    for (; q + Lanes <= cols; q += Lanes) {
        std::array<S, Lanes> rowSums{};

        for (int p = 0; p < rows; ++p) {
            const S weight = rowWeight(p);

            for (int lane = 0; lane < Lanes; ++lane) {
                S& sum = rowSums[static_cast<std::size_t>(lane)];
                sum = multiplyAdd(weight, value(p, q + lane), sum);
            }
        }

        for (int lane = 0; lane < Lanes; ++lane) {
            total = multiplyAdd(colWeight(q + lane),
                                rowSums[static_cast<std::size_t>(lane)], total);
        }
    }

    for (; q < cols; ++q) {
        S row{0};

        for (int p = 0; p < rows; ++p) {
            row = multiplyAdd(rowWeight(p), value(p, q), row);
        }

        total = multiplyAdd(colWeight(q), row, total);
    }

    return total;
}

} // namespace detail

/**
 * @brief Down-weights pixels by a Gaussian of their distance to the block
 * center.
 *
 * A pixel at the block-relative position @f$(u,v)@f$ of a block with the
 * extent @f$B_r \times B_c@f$ contributes to the block with the weight
 * @f$w_r(u)\,w_c(v)@f$ where
 *
 * @f[
 *  w(u) = \exp\left(-\frac{(u-\frac{B-1}{2})^2-d_{\min}^2}{2\sigma^2}\right)
 * @f]
 *
 * and @f$d_{\min}@f$ denotes the distance of the pixel closest to the block
 * center. The normalization by the closest pixel is a constant per block that
 * cancels in block normalization. It guarantees a weight of 1 for the closest
 * pixels regardless of how small @f$\sigma@f$ is.
 *
 * The weighting is always enabled. Use NoSpatialWeighting instead to disable
 * it without any overhead.
 *
 * If the block stride is not smaller than the cell size along both axes, the
 * weight of a pixel for a given cell does not depend on the block containing
 * it but only on the pixel position modulo the block stride. The weights are
 * then folded into the votes of an additional integral histogram with a
 * separate set of channels per cell, and each weighted cell histogram becomes
 * an O(1) query. Otherwise, or for blocks whose origin is not a multiple of
 * the block stride, the weighted cell histograms are summed from the per-pixel
 * votes in O(cell area) per cell. The latter includes every block of a region
 * whose offset is not a multiple of the block stride and the last block that
 * is placed flush with the region border.
 */
template<class T>
class GaussianWeighting
{
public:
    using Scalar = T;

    template<class S>
    class State
    {
        friend class GaussianWeighting;

    public:
        /**
         * @brief Returns the integral histogram of the weighted votes.
         *
         * The histogram is empty if the block layout does not allow to
         * integrate the weighted votes upfront. Pass it to
         * GaussianWeighting::restore() to reproduce the weighted cell
         * histograms exactly.
         */
        [[nodiscard]] const Eigen::Tensor<S, 3>& weightedHistogram()
            const noexcept
        {
            return weighted_.histogram();
        }

    private:
        IntegralHistogram<S, 2, 1> weighted_;
        BlockGeometry geometry_;
        S sigma_{};
        Eigen::DenseIndex bins_ = 0;
    };

    /**
     * @brief Initializes the weighting using the specified standard deviation
     * in pixels.
     *
     * @throws std::invalid_argument @p sigma lies outside of @f$(0,M]@f$ with
     * @f$M@f$ denoting the largest finite value of @p T.
     */
    [[nodiscard]] explicit GaussianWeighting(T sigma)
        : sigma_{validate(sigma)}
    {
    }

    //! @throws std::invalid_argument @p value lies outside of the valid range.
    void setSigma(T value)
    {
        sigma_ = validate(value);
    }

    [[nodiscard]] T sigma() const noexcept
    {
        return sigma_;
    }

    template<class S, class VisitVotes>
    void prepare(State<S>& state, const BlockGeometry& geometry,
                 const Eigen::Array2i& dims, Eigen::DenseIndex bins,
                 VisitVotes&& visitVotes) const
    {
        // Release the previous table before building the new one. This
        // lowers the peak memory usage and leaves no stale table behind if
        // building the new one fails.
        state.weighted_ = IntegralHistogram<S, 2, 1>{};
        state.geometry_ = geometry;
        state.sigma_ = static_cast<S>(sigma_);
        state.bins_ = bins;

        if (!integrable(geometry)) {
            return;
        }

        const Eigen::Array2i numCells = geometry.blockSize / geometry.cellSize;
        const Eigen::DenseIndex numBlockCells = numCells.prod();
        const Eigen::Array2i& stride = geometry.blockStride;

        const std::vector<S> rowWeights =
            residueWeights(geometry.cellSize.x(), stride.x(),
                           geometry.blockSize.x(), numCells.x(), state.sigma_);
        const std::vector<S> colWeights =
            residueWeights(geometry.cellSize.y(), stride.y(),
                           geometry.blockSize.y(), numCells.y(), state.sigma_);

        // Product weights indexed by the row residue, the column residue and
        // the cell.
        std::vector<S> weights(static_cast<std::size_t>(
            stride.prod() * static_cast<int>(numBlockCells)));

        for (int rr = 0; rr < stride.x(); ++rr) {
            for (int rc = 0; rc < stride.y(); ++rc) {
                for (int k = 0; k < numCells.x(); ++k) {
                    for (int l = 0; l < numCells.y(); ++l) {
                        weights[static_cast<std::size_t>(
                            ((rr * stride.y() + rc) * numCells.x() + k) *
                                numCells.y() +
                            l)] =
                            rowWeights[static_cast<std::size_t>(k * stride.x() +
                                                                rr)] *
                            colWeights[static_cast<std::size_t>(l * stride.y() +
                                                                rc)];
                    }
                }
            }
        }

        IntegralHistogram<S, 2, 1> weighted;
        weighted.resize(std::make_tuple(dims.x(), dims.y()),
                        numBlockCells * bins);
        weighted.scan([&weights, &visitVotes, &stride, numBlockCells, bins](
                          typename IntegralHistogram<S, 2, 1>::Bins channels,
                          const auto& ij) {
            const auto [i, j] = ij;
            const S* const w = weights.data() + ((i % stride.x()) * stride.y() +
                                                 j % stride.y()) *
                                                    numBlockCells;

            visitVotes(ij, [&channels, w, numBlockCells, bins](
                               Eigen::DenseIndex bin, S vote) {
                for (Eigen::DenseIndex kl = 0; kl < numBlockCells; ++kl) {
                    S& value = channels(kl * bins + bin);
                    value = multiplyAdd(w[kl], vote, value);
                }
            });
        });

        state.weighted_ = std::move(weighted);
    }

    /**
     * @brief Restores the state from the integral histogram of the weighted
     * votes obtained from State::weightedHistogram().
     *
     * Unlike prepare(), the weighted votes are not recovered from the
     * regular integral histogram, which loses precision.
     *
     * @throws std::invalid_argument @p weighted does not have the dimensions
     * expected for the block @p geometry, the input @p dims and the number of
     * @p bins. @p state is left unchanged in this case.
     */
    template<class S>
    void restore(State<S>& state, const BlockGeometry& geometry,
                 const Eigen::Array2i& dims, Eigen::DenseIndex bins,
                 const Eigen::Tensor<S, 3>& weighted) const
    {
        std::array<Eigen::DenseIndex, 3> expected{};

        if (integrable(geometry)) {
            const Eigen::Array2i numCells =
                geometry.blockSize / geometry.cellSize;

            expected = {dims.x() + 1, dims.y() + 1, numCells.prod() * bins};
        }

        if (weighted.dimensions() != expected) {
            using namespace fmt::literals;

            throw std::invalid_argument{fmt::format(
                "GaussianWeighting weighted integral histogram must have the "
                "dimensions [{expected}] but [{actual}] were given",
                "expected"_a = fmt::join(expected, ", "),
                "actual"_a = fmt::join(weighted.dimensions(), ", "))};
        }

        state.weighted_ = IntegralHistogram<S, 2, 1>{};
        state.geometry_ = geometry;
        state.sigma_ = static_cast<S>(sigma_);
        state.bins_ = bins;

        if (weighted.size() > 0) {
            state.weighted_.setHistogram(weighted);
        }
    }

    template<class S, int Options>
    [[nodiscard]] auto cellQuery(
        const State<S>& state,
        const IntegralHistogram<S, 2, 1, Options>& histogram,
        const BlockGeometry& geometry) const
    {
        const S sigma = static_cast<S>(sigma_);

        const auto& H = histogram.histogram();
        const auto& W = state.weighted_.histogram();

        // The prepared weights only apply to the sigma they were built for,
        // so only an exactly equal sigma may reuse them.
        const bool valid =
            !state.weighted_.isEmpty() && state.geometry_ == geometry &&
            std::equal_to<S>{}(state.sigma_, sigma) &&
            state.bins_ == H.dimension(2) && W.dimension(0) == H.dimension(0) &&
            W.dimension(1) == H.dimension(1);

        return Query<S, Options>{histogram, valid ? &state.weighted_ : nullptr,
                                 geometry, sigma};
    }

    /**
     * @brief Computes the weight of the block-relative position @p u along an
     * axis with the block @p extent.
     *
     * The squared distance to the block center in excess of the squared
     * distance of the closest pixel is an integer computed exactly. Dividing
     * it by @p sigma twice avoids the underflow of @f$\sigma^2@f$ for tiny
     * values. An overflow to infinity correctly yields a zero weight.
     */
    template<class S>
    [[nodiscard]] static S weight(int u, int extent, S sigma)
    {
        const auto e = 2 * static_cast<std::int64_t>(u) - (extent - 1);
        const std::int64_t emin = extent % 2 == 0 ? 1 : 0;
        const std::int64_t excess = (e * e - emin * emin) / 4;

        if (excess == 0) {
            return S{1};
        }

        using std::exp;
        return exp(-((static_cast<S>(excess) / sigma) / sigma) / 2);
    }

private:
    template<class S, int Options>
    class Query
    {
    public:
        [[nodiscard]] Query(
            const IntegralHistogram<S, 2, 1, Options>& histogram,
            const IntegralHistogram<S, 2, 1>* weighted,
            const BlockGeometry& geometry, S sigma)
            : histogram_{&histogram}
            , weighted_{weighted}
            , cellSize_{geometry.cellSize}
            , blockStride_{geometry.blockStride}
            , blockSize_{geometry.blockSize}
            , numCellCols_{geometry.blockSize.y() / geometry.cellSize.y()}
            , bins_{histogram.histogram().dimension(2)}
            , sigma_{sigma}
            , rowWeights_{blockWeights(geometry.blockSize.x(), sigma)}
            , colWeights_{blockWeights(geometry.blockSize.y(), sigma)}
            , cell_(bins_)
        {
        }

        template<class Cell>
        void operator()(Cell&& cell, const Eigen::Array2i& index,
                        const Eigen::Array2i& first, const Eigen::Array2i& last)
        {
            const Eigen::Array2i origin = first - index * cellSize_;

            // The weighted integral histogram assigns the weights by the
            // pixel position modulo the block stride. This only holds for
            // blocks whose origin is a multiple of the stride, which excludes
            // blocks of unaligned regions and blocks placed flush with the
            // region border.
            if (weighted_ != nullptr &&
                (origin - origin / blockStride_ * blockStride_ == 0).all()) {
                const Eigen::DenseIndex kl =
                    static_cast<Eigen::DenseIndex>(index.x()) * numCellCols_ +
                    index.y();

                cell = weighted_
                           ->intersect(std::make_tuple(first.x(), first.y()),
                                       std::make_tuple(last.x(), last.y()))
                           .slice(std::array{kl * bins_}, std::array{bins_});
                return;
            }

            sum(first, last, origin);
            cell = cell_;
        }

        /**
         * @brief Extracts the cell @f$[a,b)@f$ with the index @p index of a
         * block with each vote bilinearly interpolated between neighboring
         * cell centers.
         *
         * A vote at the position @f$x@f$ is weighted by the tent
         * @f$\max(0, 1 - |x - c| / s)@f$ with the cell center
         * @f$c = (a + b - 1) / 2@f$ and the cell size @f$s = b - a@f$ along
         * each axis, and by the Gaussian of its position relative to the
         * block origin. Votes outside of the input domain do not contribute.
         *
         * The tent extends one cell beyond the cell and therefore beyond the
         * block. Unlike Dalal and Triggs, who only consider the pixels of
         * the block, pixels outside of the block are not excluded but
         * weighted by the same Gaussian evaluated at their block-relative
         * position. This keeps the weights continuous across the block
         * border.
         *
         * The weighted votes depend on the block for every block layout, so
         * they are summed in @f$O(s_r s_c)@f$ per cell. Summation by parts
         *
         * @f[
         *  \sum_{x=x_0}^{x_1-1} w_x \left(H_{x+1} - H_x\right)
         *  = \sum_{x=x_0}^{x_1} \left(w_{x-1} - w_x\right) H_x
         * @f]
         *
         * with @f$w_{x_0-1} = w_{x_1} = 0@f$ along each axis reads every
         * entry of the integral histogram @f$H@f$ once instead of recovering
         * each vote from four entries.
         */
        template<class Cell>
        void interpolate(Cell&& cell, const Eigen::Array2i& index,
                         const Eigen::Array2i& first,
                         const Eigen::Array2i& last)
        {
            const Eigen::Array2i origin = first - index * cellSize_;
            const auto& H = histogram_->histogram();

            const int rowBegin = kernelDifferences(
                rowKernel_, first.x(), last.x(), origin.x(), blockSize_.x(),
                static_cast<int>(H.dimension(0) - 1));
            const int colBegin = kernelDifferences(
                colKernel_, first.y(), last.y(), origin.y(), blockSize_.y(),
                static_cast<int>(H.dimension(1) - 1));

            const auto rows = static_cast<int>(rowKernel_.size());
            const auto cols = static_cast<int>(colKernel_.size());

            const auto rowWeight = [this](int p) {
                return rowKernel_[static_cast<std::size_t>(p)];
            };
            const auto colWeight = [this](int q) {
                return colKernel_[static_cast<std::size_t>(q)];
            };

            for (Eigen::DenseIndex b = 0; b < bins_; ++b) {
                cell_(b) = detail::separableSum<S>(
                    rows, cols, rowWeight, colWeight,
                    [&H, rowBegin, colBegin, b](int p, int q) {
                        return H(rowBegin + p, colBegin + q, b);
                    });
            }

            cell = cell_;
        }

    private:
        /**
         * Stores the differences @f$w_{x-1} - w_x@f$ of the products
         * @f$w_x@f$ of the tent of the cell @f$[a,b)@f$ and the Gaussian
         * weight along an axis in @p differences. The positions @f$x@f$ range
         * over the support of the tent within the input domain @f$[0,n)@f$
         * and one position beyond, and the weights vanish outside of it.
         * Returns the first position.
         */
        [[nodiscard]] int kernelDifferences(std::vector<S>& differences, int a,
                                            int b, int origin, int extent,
                                            int n) const
        {
            const int size = b - a;
            // Twice the cell center keeps the distances integral for even
            // cell sizes.
            const int center2 = a + b - 1;
            const int begin = std::max(a - size, 0);
            const int end = std::min(b + size, n);

            differences.clear();

            S previous{0};

            for (int x = begin; x < end; ++x) {
                const int distance2 = std::abs(2 * x - center2);
                const S tent = distance2 < 2 * size
                                   ? static_cast<S>(2 * size - distance2) /
                                         static_cast<S>(2 * size)
                                   : S{0};
                const S current = tent * weight(x - origin, extent, sigma_);

                differences.push_back(previous - current);
                previous = current;
            }

            differences.push_back(previous);

            return begin;
        }

        // Sums the weighted votes recovered from the integral histogram.
        void sum(const Eigen::Array2i& first, const Eigen::Array2i& last,
                 const Eigen::Array2i& origin)
        {
            const auto& H = histogram_->histogram();

            const auto rowWeight = [this,
                                    offset = first.x() - origin.x()](int p) {
                return rowWeights_[static_cast<std::size_t>(offset + p)];
            };
            const auto colWeight = [this,
                                    offset = first.y() - origin.y()](int q) {
                return colWeights_[static_cast<std::size_t>(offset + q)];
            };

            for (Eigen::DenseIndex b = 0; b < bins_; ++b) {
                cell_(b) = detail::separableSum<S>(
                    last.x() - first.x(), last.y() - first.y(), rowWeight,
                    colWeight, [&H, &first, b](int p, int q) {
                        const int x = first.x() + p;
                        const int y = first.y() + q;
                        return H(x + 1, y + 1, b) - H(x, y + 1, b) -
                               H(x + 1, y, b) + H(x, y, b);
                    });
            }
        }

        const IntegralHistogram<S, 2, 1, Options>* histogram_;
        //! Weighted integral histogram, or @c nullptr if it is unusable.
        const IntegralHistogram<S, 2, 1>* weighted_;
        Eigen::Array2i cellSize_;
        Eigen::Array2i blockStride_;
        Eigen::Array2i blockSize_;
        Eigen::DenseIndex numCellCols_;
        Eigen::DenseIndex bins_;
        S sigma_;
        std::vector<S> rowWeights_;
        std::vector<S> colWeights_;
        //! Interpolation weights reused across cells.
        std::vector<S> rowKernel_;
        std::vector<S> colKernel_;
        Eigen::Tensor<S, 1> cell_;
    };

    /**
     * @brief Determines whether the weighted votes can be integrated upfront.
     *
     * A pixel can occupy different positions within the same cell of
     * overlapping blocks if the block stride is smaller than the cell size.
     * The weighted votes then depend on the block.
     */
    [[nodiscard]] static bool integrable(const BlockGeometry& geometry)
    {
        return (geometry.blockStride >= geometry.cellSize).all();
    }

    [[nodiscard]] static T validate(T value)
    {
        constexpr T max = std::numeric_limits<T>::max();

        // The inverted comparison also rejects NaN.
        if (!(value > 0 && value <= max)) {
            using namespace fmt::literals;

            throw std::invalid_argument{fmt::format(
                "GaussianWeighting sigma must lie within (0, {max}] but "
                "{value} was given",
                "max"_a = max, "value"_a = value)};
        }

        return value;
    }

    // Weights of all positions along a block axis.
    template<class S>
    [[nodiscard]] static std::vector<S> blockWeights(int extent, S sigma)
    {
        std::vector<S> result(static_cast<std::size_t>(extent));

        for (int u = 0; u < extent; ++u) {
            result[static_cast<std::size_t>(u)] = weight(u, extent, sigma);
        }

        return result;
    }

    /**
     * Weights along a block axis indexed by the cell @f$k@f$ and the residue
     * @f$\rho@f$ of the pixel position modulo the block stride @f$s@f$. For
     * @f$s \geq c@f$ with the cell size @f$c@f$, a pixel with the residue
     * @f$\rho@f$ occupies the block-relative position @f$kc+t@f$ with
     * @f$t=(\rho-kc) \bmod s@f$ in cell @f$k@f$ if @f$t<c@f$ and lies outside
     * of cell @f$k@f$ of every block otherwise.
     */
    template<class S>
    [[nodiscard]] static std::vector<S> residueWeights(int cellSize, int stride,
                                                       int extent, int numCells,
                                                       S sigma)
    {
        std::vector<S> result(static_cast<std::size_t>(numCells * stride));

        for (int k = 0; k < numCells; ++k) {
            for (int rho = 0; rho < stride; ++rho) {
                const int t = ((rho - k * cellSize) % stride + stride) % stride;

                result[static_cast<std::size_t>(k * stride + rho)] =
                    t < cellSize ? weight(k * cellSize + t, extent, sigma)
                                 : S{0};
            }
        }

        return result;
    }

    T sigma_;
};

} // namespace hogpp

#endif // HOGPP_SPATIALWEIGHTING_HPP
