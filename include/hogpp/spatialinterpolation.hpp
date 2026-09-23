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

#ifndef HOGPP_SPATIALINTERPOLATION_HPP
#define HOGPP_SPATIALINTERPOLATION_HPP

#include <tuple>
#include <type_traits>
#include <utility>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <hogpp/integralhistogram.hpp>

namespace hogpp {

// Default trait: a pixel's vote stays in its own cell (plain box sum).
// Each trait owns the state it derives from the integral histogram and the
// extraction of cell histograms, so IntegralHOGDescriptor itself never
// branches on which is active.
struct NoSpatialInterpolation
{
    static constexpr bool value = false;

    template<class Scalar>
    struct Cache
    {
    };

    template<class IntegralHistogram, class CacheType>
    static void update(const IntegralHistogram& /*histogram*/,
                       CacheType& /*cache*/) noexcept
    {
        // The first-order integral histogram suffices.
    }

    /**
     * @brief Returns a callable that extracts the cell histogram with the
     * index @c index of a block starting at @c offset1 using @p query.
     *
     * The callable is created once per extraction of features and is invoked
     * as @c extract(cell,index,offset1,cellSize,br). It clips the cell to the
     * region ending at @c br so that a block larger than the region
     * accumulates only the votes of the pixels inside of it.
     */
    template<class Query, class IntegralHistogram, class CacheType>
    [[nodiscard]] static auto extractor(Query& query,
                                        const IntegralHistogram& /*histogram*/,
                                        const CacheType& /*cache*/) noexcept
    {
        return
            [&query](auto&& cell, const Eigen::Array2i& index,
                     const Eigen::Array2i& offset1,
                     const Eigen::Array2i& cellSize, const Eigen::Array2i& br) {
                const Eigen::Array2i first = offset1.min(br);
                const Eigen::Array2i last = (first + cellSize).min(br);
                query(std::forward<decltype(cell)>(cell), index, first, last);
            };
    }
};

// The spatial half of Dalal & Triggs' trilinear interpolation:
// bilinearly splits a pixel's vote across neighboring cell centers via
// a second-order cascaded integral histogram (IntegralHistogram::
// cascade()/interpolate()) rather than a fixed cell grid.
struct TrilinearInterpolation
{
    static constexpr bool value = true;

    // At least double precision avoids catastrophic cancellation in the
    // second-order integral histogram of single precision votes.
    template<class Scalar>
    using Cache = IntegralHistogram<std::common_type_t<Scalar, double>, 2, 1>;

    template<class IntegralHistogram, class CacheType>
    static void update(const IntegralHistogram& histogram, CacheType& cache)
    {
        // Release the previous table before building the new one. This halves
        // the peak memory usage of the table and leaves no stale table behind
        // if building the new one fails.
        cache = CacheType{};
        cache = histogram.template cascade<typename CacheType::Scalar>();
    }

    /**
     * @brief Returns a callable that extracts the bilinearly interpolated
     * cell histogram with the index @c index of a block starting at
     * @c offset1.
     *
     * The callable is invoked as @c extract(cell,index,offset1,cellSize,br).
     * The cell keeps its extent even if it exceeds the region ending at
     * @c br, since the interpolation weights depend on the cell size. A
     * spatial weighting whose @p query interpolates its weighted votes itself
     * does not use the second-order integral histogram in @p cache.
     * Otherwise, the callable owns the buffer the interpolation is evaluated
     * into, which avoids allocating it per cell.
     */
    template<class Query, class IntegralHistogram, class CacheType>
    [[nodiscard]] static auto extractor(Query& query,
                                        const IntegralHistogram& /*histogram*/,
                                        const CacheType& cache)
    {
        using Scalar = typename IntegralHistogram::Scalar;

        if constexpr (requires(Eigen::Tensor<Scalar, 1>& cell,
                               const Eigen::Array2i& offset) {
                          query.interpolate(cell, offset, offset, offset);
                      }) {
            return [&query](auto&& cell, const Eigen::Array2i& index,
                            const Eigen::Array2i& offset1,
                            const Eigen::Array2i& cellSize,
                            const Eigen::Array2i& /*br*/) {
                query.interpolate(std::forward<decltype(cell)>(cell), index,
                                  offset1, offset1 + cellSize);
            };
        }
        else {
            return [&cache, buffer =
                                Eigen::Tensor<typename CacheType::Scalar, 1>{
                                    cache.histogram().dimension(2)}](
                       auto&& cell, const Eigen::Array2i& /*index*/,
                       const Eigen::Array2i& offset1,
                       const Eigen::Array2i& cellSize,
                       const Eigen::Array2i& /*br*/) mutable {
                const Eigen::Array2i offset2 = offset1 + cellSize;

                cache.interpolate(std::make_tuple(offset1.x(), offset1.y()),
                                  std::make_tuple(offset2.x(), offset2.y()),
                                  buffer);
                // A conversion to the same type evaluates into a temporary.
                if constexpr (std::is_same_v<Scalar,
                                             typename CacheType::Scalar>) {
                    cell = buffer;
                }
                else {
                    cell = buffer.template cast<Scalar>();
                }
            };
        }
    }
};

} // namespace hogpp

#endif // HOGPP_SPATIALINTERPOLATION_HPP
