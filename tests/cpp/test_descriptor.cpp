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

#define BOOST_TEST_MODULE hogpp

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>

#include <hogpp/integralhogdescriptor.hpp>
#include <hogpp/signedgradient.hpp>
#include <hogpp/spatialinterpolation.hpp>
#include <hogpp/spatialweighting.hpp>

#include <boost/mpl/list.hpp>
#include <boost/test/included/unit_test.hpp>

using Scalars = boost::mpl::list<float, double, long double>;

template<class Scalar>
auto makeImage()
{
    Eigen::TensorFixedSize<Scalar, Eigen::Sizes<3, 3, 1>> image;
    image.setValues({{{Scalar(0)}, {Scalar(1)}, {Scalar(2)}},
                     {{Scalar(3)}, {Scalar(4)}, {Scalar(5)}},
                     {{Scalar(6)}, {Scalar(7)}, {Scalar(8)}}});
    return image;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(empty, Scalar, Scalars)
{
    BOOST_TEST(hogpp::IntegralHOGDescriptor<Scalar>{}.isEmpty());
    BOOST_TEST(hogpp::IntegralHOGDescriptor<Scalar>{}.features().size() == 0);
    BOOST_TEST(hogpp::IntegralHOGDescriptor<Scalar>{}
                   .features(hogpp::Bounds{})
                   .size() == 0);
    BOOST_TEST(hogpp::IntegralHOGDescriptor<Scalar>{}.histogram().size() == 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(void_gradient, Scalar, Scalars)
{
    hogpp::IntegralHOGDescriptor<Scalar, void> d;

    Eigen::Tensor<Scalar, 3> dxs;
    Eigen::Tensor<Scalar, 3> dys;

    d.compute(dxs, dys, nullptr);

    BOOST_TEST(d.isEmpty());
}

// A block many strides larger than the region still yields exactly one
// block that covers the region.
BOOST_AUTO_TEST_CASE_TEMPLATE(features_block_larger_than_region, Scalar,
                              Scalars)
{
    auto image = makeImage<Scalar>();

    hogpp::IntegralHOGDescriptor<Scalar> d;
    d.setBlockSize(Eigen::Array2i{100, 100});
    d.setBlockStride(Eigen::Array2i{1, 1});
    d.compute(image);

    const typename hogpp::IntegralHOGDescriptor<Scalar>::Tensor5 X =
        d.features(hogpp::Bounds{0, 0, 3, 3});

    BOOST_TEST(X.dimension(0) == 1);
    BOOST_TEST(X.dimension(1) == 1);
}

namespace {

// Indices of the cell and bin dimensions of the features tensor
constexpr int CellRowDimension = 2;
constexpr int CellColDimension = 3;
constexpr int BinDimension = 4;

template<class Scalar>
using Features = typename hogpp::IntegralHOGDescriptor<Scalar>::Tensor5;

// Gradient tensors are indexed by row, column and channel.
constexpr int GradientRank = 3;
constexpr int NumChannels = 1;

constexpr int GradientRows = 17;
constexpr int GradientCols = 15;
constexpr unsigned GradientSeed = 42;

// Cell, block and stride sizes are given as rows and columns. The block is
// not square and the stride equals the cell size with swapped axes, so that
// confusing the axes or the stride with the cell size changes the block
// layout.
const Eigen::Array2i CellSize{2, 3};
const Eigen::Array2i BlockSize{4, 6};
const Eigen::Array2i BlockStride{3, 2};

// Both region extents exceed a multiple of the block stride by one pixel.
constexpr int RegionRows = 14;
constexpr int RegionCols = 11;

template<class Scalar>
struct Gradients
{
    Eigen::Tensor<Scalar, GradientRank> dxs;
    Eigen::Tensor<Scalar, GradientRank> dys;
};

template<class Scalar>
Gradients<Scalar> makeRandomGradients(int rows, int cols)
{
    constexpr Scalar MaxGradient{1};

    std::mt19937 rng{GradientSeed};
    std::uniform_real_distribution<Scalar> dist{-MaxGradient, MaxGradient};

    Gradients<Scalar> g{
        Eigen::Tensor<Scalar, GradientRank>(rows, cols, NumChannels),
        Eigen::Tensor<Scalar, GradientRank>(rows, cols, NumChannels)};

    for (Eigen::Index i = 0; i < g.dxs.size(); ++i) {
        g.dxs.data()[i] = dist(rng);
        g.dys.data()[i] = dist(rng);
    }

    return g;
}

template<class Scalar>
hogpp::IntegralHOGDescriptor<Scalar> makeDescriptor(const Gradients<Scalar>& g)
{
    hogpp::IntegralHOGDescriptor<Scalar> d;
    d.setCellSize(CellSize);
    d.setBlockSize(BlockSize);
    d.setBlockStride(BlockStride);
    d.compute(g.dxs, g.dys, nullptr);
    return d;
}

template<class Scalar>
void checkBlockEqual(const Features<Scalar>& actual, Eigen::Index i,
                     Eigen::Index j, const Features<Scalar>& expected)
{
    BOOST_TEST_REQUIRE(expected.dimension(0) == 1);
    BOOST_TEST_REQUIRE(expected.dimension(1) == 1);

    for (std::size_t k = CellRowDimension; k < Features<Scalar>::NumDimensions;
         ++k) {
        BOOST_TEST_REQUIRE(actual.dimension(k) == expected.dimension(k));
    }

    // Block normalized features are of unit scale. Bins without votes are
    // differences of equal prefix sums which cancel only up to rounding.
    // Hence, the comparison uses an absolute rather than a relative tolerance.
    using std::sqrt;
    const Scalar tolerance = sqrt(std::numeric_limits<Scalar>::epsilon());

    for (Eigen::Index k = 0; k < actual.dimension(CellRowDimension); ++k) {
        for (Eigen::Index l = 0; l < actual.dimension(CellColDimension); ++l) {
            for (Eigen::Index m = 0; m < actual.dimension(BinDimension); ++m) {
                const Scalar a = actual(i, j, k, l, m);
                const Scalar b = expected(0, 0, k, l, m);

                BOOST_TEST_CONTEXT("block (" << i << ", " << j << "), cell ("
                                             << k << ", " << l << "), bin " << m
                                             << ", actual " << a
                                             << ", expected " << b)
                {
                    BOOST_CHECK_SMALL(a - b, tolerance);
                }
            }
        }
    }
}

} // namespace

// When the region extent minus the block extent is not a multiple of the
// block stride, the last block is placed flush with the bottom and right
// border of the region, so that every pixel contributes.
BOOST_AUTO_TEST_CASE_TEMPLATE(features_cover_region_border, Scalar, Scalars)
{
    const Gradients<Scalar> g =
        makeRandomGradients<Scalar>(GradientRows, GradientCols);
    const hogpp::IntegralHOGDescriptor<Scalar> d = makeDescriptor(g);

    constexpr int RegionRow = 2;
    constexpr int RegionCol = 1;

    const std::array expectedRowStarts{0, 3, 6, 9, 10};
    const std::array expectedColStarts{0, 2, 4, 5};

    const Features<Scalar> X =
        d.features(hogpp::Bounds{RegionCol, RegionRow, RegionCols, RegionRows});

    BOOST_TEST_REQUIRE(X.dimension(0) == static_cast<Eigen::DenseIndex>(
                                             expectedRowStarts.size()));
    BOOST_TEST_REQUIRE(X.dimension(1) == static_cast<Eigen::DenseIndex>(
                                             expectedColStarts.size()));

    for (std::size_t i = 0; i != expectedRowStarts.size(); ++i) {
        for (std::size_t j = 0; j != expectedColStarts.size(); ++j) {
            const Features<Scalar> expected =
                d.features(hogpp::Bounds{RegionCol + expectedColStarts[j],
                                         RegionRow + expectedRowStarts[i],
                                         BlockSize.y(), BlockSize.x()});

            checkBlockEqual<Scalar>(X, static_cast<Eigen::Index>(i),
                                    static_cast<Eigen::Index>(j), expected);
        }
    }
}

// The largest representable block stride exceeds any region. Rounding the
// number of strides up must not overflow, so that the region is still covered
// by one block at its origin and one block flush with its border.
BOOST_AUTO_TEST_CASE_TEMPLATE(features_largest_block_stride, Scalar, Scalars)
{
    const Gradients<Scalar> g =
        makeRandomGradients<Scalar>(GradientRows, GradientCols);
    hogpp::IntegralHOGDescriptor<Scalar> d = makeDescriptor(g);
    d.setBlockStride(Eigen::Array2i::Constant(std::numeric_limits<int>::max()));

    const Features<Scalar> X =
        d.features(hogpp::Bounds{0, 0, RegionCols, RegionRows});

    BOOST_TEST_REQUIRE(X.dimension(0) == 2);
    BOOST_TEST_REQUIRE(X.dimension(1) == 2);

    const Features<Scalar> expected = d.features(
        hogpp::Bounds{RegionCols - BlockSize.y(), RegionRows - BlockSize.x(),
                      BlockSize.y(), BlockSize.x()});

    checkBlockEqual<Scalar>(X, 1, 1, expected);
}

// A region smaller than a block yields a single block at the region origin
// whose cells are clipped to the region. Its features equal those of the
// region padded with zero gradients to the block size. The regions touch the
// bottom right image corner, so that an unclipped block reaches beyond the
// image.
BOOST_AUTO_TEST_CASE_TEMPLATE(features_clip_block_to_small_region, Scalar,
                              Scalars)
{
    const Gradients<Scalar> g =
        makeRandomGradients<Scalar>(GradientRows, GradientCols);
    const hogpp::IntegralHOGDescriptor<Scalar> d = makeDescriptor(g);

    const std::array regionSizes{
        Eigen::Array2i{BlockSize},
        Eigen::Array2i{BlockSize - Eigen::Array2i{1, 0}},
        Eigen::Array2i{BlockSize - Eigen::Array2i{0, 1}},
        Eigen::Array2i{BlockSize - Eigen::Array2i{1, 1}},
        Eigen::Array2i{BlockSize - Eigen::Array2i{BlockStride.x(), 0}},
        Eigen::Array2i{BlockSize - Eigen::Array2i{0, BlockStride.y()}},
    };

    for (const Eigen::Array2i& size : regionSizes) {
        BOOST_TEST_CONTEXT("region size " << size.x() << "x" << size.y())
        {
            const int row = GradientRows - size.x();
            const int col = GradientCols - size.y();

            Gradients<Scalar> padded{
                Eigen::Tensor<Scalar, GradientRank>(BlockSize.x(),
                                                    BlockSize.y(), NumChannels),
                Eigen::Tensor<Scalar, GradientRank>(
                    BlockSize.x(), BlockSize.y(), NumChannels)};
            padded.dxs.setZero();
            padded.dys.setZero();

            const std::array<Eigen::Index, GradientRank> offsets{row, col, 0};
            const std::array<Eigen::Index, GradientRank> extents{
                size.x(), size.y(), NumChannels};
            const std::array<Eigen::Index, GradientRank> origin{0, 0, 0};

            padded.dxs.slice(origin, extents) = g.dxs.slice(offsets, extents);
            padded.dys.slice(origin, extents) = g.dys.slice(offsets, extents);

            const Features<Scalar> expected = makeDescriptor(padded).features(
                hogpp::Bounds{0, 0, BlockSize.y(), BlockSize.x()});

            const Features<Scalar> X =
                d.features(hogpp::Bounds{col, row, size.y(), size.x()});

            BOOST_TEST_REQUIRE(X.dimension(0) == 1);
            BOOST_TEST_REQUIRE(X.dimension(1) == 1);

            checkBlockEqual<Scalar>(X, 0, 0, expected);
        }
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(default_descriptor_has_no_spatial_interpolation,
                              Scalar, Scalars)
{
    using Descriptor = hogpp::IntegralHOGDescriptor<Scalar>;

    BOOST_TEST(!Descriptor::spatialInterpolation());

    auto image = makeImage<Scalar>();

    Descriptor d;
    d.setCellSize(Eigen::Array2i{3, 3});
    d.setBlockSize(Eigen::Array2i{3, 3});
    d.setBlockStride(Eigen::Array2i{3, 3});
    d.compute(image);

    BOOST_CHECK_NO_THROW((void)d.features());
}

namespace {

// Leaves block histograms untouched so that features() exposes the raw cell
// histograms.
struct IdentityBlockNormalizer
{
    template<class Tensor>
    constexpr void operator()(Tensor& /*block*/) const noexcept
    {
    }
};

template<class Scalar, class SpatialInterpolation>
using UnnormalizedDescriptor = hogpp::IntegralHOGDescriptor<
    Scalar, hogpp::Gradient<Scalar>, hogpp::GradientMagnitude<Scalar>,
    hogpp::UnsignedGradient<Scalar>, IdentityBlockNormalizer,
    hogpp::NoSpatialWeighting, SpatialInterpolation>;

[[nodiscard]] Eigen::Tensor<double, 3> makeRandomImage(Eigen::DenseIndex rows,
                                                       Eigen::DenseIndex cols)
{
    std::default_random_engine g{1};
    std::uniform_real_distribution<double> intensity{0.0, 1.0};

    Eigen::Tensor<double, 3> image(rows, cols, 1);

    for (Eigen::DenseIndex i = 0; i < image.size(); ++i) {
        image.data()[i] = intensity(g);
    }

    return image;
}

// Orientation votes of each individual pixel as a rows x cols x bins tensor.
// Cells and blocks of a single pixel do not mix votes of neighboring pixels.
[[nodiscard]] Eigen::Tensor<double, 3> pixelVotes(
    const Eigen::Tensor<double, 3>& image)
{
    const Eigen::Array2i pixel{1, 1};

    UnnormalizedDescriptor<double, hogpp::NoSpatialInterpolation> d;
    d.setCellSize(pixel);
    d.setBlockSize(pixel);
    d.setBlockStride(pixel);
    d.compute(image);

    const Eigen::Tensor<double, 5> X = d.features();
    return X.chip<3>(0).chip<2>(0);
}

// Spatial half of the trilinear interpolation by Dalal and Triggs: a pixel
// votes into a cell with the weight (1 - |dx| / sx) (1 - |dy| / sy) where
// (dx, dy) is the distance of the pixel to the geometric center of the cell
// and (sx, sy) the cell size. Pixels outside the image do not vote.
[[nodiscard]] Eigen::Tensor<double, 1> bruteForceBilinearSplat(
    const Eigen::Tensor<double, 3>& votes, const Eigen::Array2i& offset,
    const Eigen::Array2i& cellSize)
{
    const Eigen::Array2d size = cellSize.cast<double>();
    const Eigen::Array2d center = offset.cast<double>() + (size - 1) / 2;

    const Eigen::Array2i first = (offset - cellSize).max(0);
    const Eigen::Array2i last =
        (offset + 2 * cellSize)
            .min(Eigen::Array2i{votes.dimension(0), votes.dimension(1)});

    Eigen::Tensor<double, 1> h(votes.dimension(2));
    h.setZero();

    for (int i = first.x(); i < last.x(); ++i) {
        const double wx =
            std::max(0.0, 1 - std::abs(i - center.x()) / size.x());

        for (int j = first.y(); j < last.y(); ++j) {
            const double wy =
                std::max(0.0, 1 - std::abs(j - center.y()) / size.y());

            h += votes.chip<0>(i).chip<0>(j) * (wx * wy);
        }
    }

    return h;
}

// Plain sum of the votes within the cell.
[[nodiscard]] Eigen::Tensor<double, 1> bruteForceBoxSum(
    const Eigen::Tensor<double, 3>& votes, const Eigen::Array2i& offset,
    const Eigen::Array2i& cellSize)
{
    Eigen::Tensor<double, 1> h(votes.dimension(2));
    h.setZero();

    for (int i = offset.x(); i < offset.x() + cellSize.x(); ++i) {
        for (int j = offset.y(); j < offset.y() + cellSize.y(); ++j) {
            h += votes.chip<0>(i).chip<0>(j);
        }
    }

    return h;
}

// Largest absolute deviation of the cell histograms from the reference
// relative to the largest absolute reference value.
template<class Scalar, class SpatialInterpolation, class Reference>
[[nodiscard]] double relativeError(const Eigen::Tensor<double, 3>& image,
                                   const Eigen::Array2i& cellSize,
                                   Reference reference)
{
    const Eigen::Tensor<double, 3> votes = pixelVotes(image);

    UnnormalizedDescriptor<Scalar, SpatialInterpolation> d;
    d.setCellSize(cellSize);
    d.setBlockSize(cellSize);
    d.setBlockStride(cellSize);
    d.compute(Eigen::Tensor<Scalar, 3>{image.cast<Scalar>()});

    const Eigen::Tensor<Scalar, 5> X = d.features();

    // Cells must tile the whole image for the border cells to be covered.
    BOOST_TEST(X.dimension(0) * cellSize.x() == image.dimension(0));
    BOOST_TEST(X.dimension(1) * cellSize.y() == image.dimension(1));

    double maxAbsError = 0;
    double maxAbsExpected = 0;

    for (int i = 0; i < X.dimension(0); ++i) {
        for (int j = 0; j < X.dimension(1); ++j) {
            const Eigen::Tensor<double, 1> expected =
                reference(votes, Eigen::Array2i{i, j} * cellSize, cellSize);

            for (Eigen::DenseIndex k = 0; k < expected.dimension(0); ++k) {
                const auto actual = static_cast<double>(X(i, j, 0, 0, k));
                maxAbsError =
                    std::max(maxAbsError, std::abs(actual - expected(k)));
                maxAbsExpected =
                    std::max(maxAbsExpected, std::abs(expected(k)));
            }
        }
    }

    return maxAbsError / maxAbsExpected;
}

// Tolerance of the relative error against a double precision reference. The
// reference rounds to double precision and therefore cannot verify a more
// precise Scalar beyond the accuracy of double.
template<class Scalar>
[[nodiscard]] double referenceTolerance()
{
    using Verifiable = std::conditional_t<(std::numeric_limits<Scalar>::digits <
                                           std::numeric_limits<double>::digits),
                                          Scalar, double>;

    using std::sqrt;
    return static_cast<double>(
        sqrt(std::numeric_limits<Verifiable>::epsilon()));
}

template<class Scalar>
void checkTrilinearInterpolationAgainstBruteForce(
    Eigen::DenseIndex rows, Eigen::DenseIndex cols,
    const Eigen::Array2i& cellSize)
{
    const double error = relativeError<Scalar, hogpp::TrilinearInterpolation>(
        makeRandomImage(rows, cols), cellSize, bruteForceBilinearSplat);

    const double tolerance = referenceTolerance<Scalar>();

    BOOST_TEST(error <= tolerance);
}

} // namespace

BOOST_AUTO_TEST_CASE_TEMPLATE(trilinear_interpolation_even_cell_size, Scalar,
                              Scalars)
{
    constexpr Eigen::DenseIndex Rows = 40;
    constexpr Eigen::DenseIndex Cols = 48;

    checkTrilinearInterpolationAgainstBruteForce<Scalar>(Rows, Cols,
                                                         Eigen::Array2i{8, 6});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(trilinear_interpolation_odd_cell_size, Scalar,
                              Scalars)
{
    constexpr Eigen::DenseIndex Rows = 35;
    constexpr Eigen::DenseIndex Cols = 40;

    checkTrilinearInterpolationAgainstBruteForce<Scalar>(Rows, Cols,
                                                         Eigen::Array2i{7, 5});
}

// Single precision loses accuracy with the image size already in the
// first-order integral histogram. Trilinear interpolation must not lose
// substantially more than the plain box sum on top of that.
BOOST_AUTO_TEST_CASE(trilinear_interpolation_large_float_image)
{
    constexpr Eigen::DenseIndex Size = 512;
    // Admissible ratio of both relative errors
    constexpr double Slack = 2;

    const Eigen::Tensor<double, 3> image = makeRandomImage(Size, Size);
    const Eigen::Array2i cellSize{8, 8};

    const double boxError = relativeError<float, hogpp::NoSpatialInterpolation>(
        image, cellSize, bruteForceBoxSum);
    const double trilinearError =
        relativeError<float, hogpp::TrilinearInterpolation>(
            image, cellSize, bruteForceBilinearSplat);

    BOOST_TEST(trilinearError <= Slack * boxError);
}

namespace {

// Gaussian weight of the block-relative position u along an axis with the
// block extent normalized by the weight of the position closest to the block
// center. Positions outside of the block are weighted by the same formula.
[[nodiscard]] double gaussianWeight(int u, int extent, double sigma)
{
    const double distance = u - (extent - 1) / 2.0;
    const double closest = extent % 2 == 0 ? 0.5 : 0.0;

    return std::exp(-(distance * distance - closest * closest) /
                    (2 * sigma * sigma));
}

// Bilinear splat of the votes into the cell at the offset with each vote
// additionally weighted by the Gaussian of its position relative to the block
// origin.
[[nodiscard]] Eigen::Tensor<double, 1> bruteForceWeightedBilinearSplat(
    const Eigen::Tensor<double, 3>& votes, const Eigen::Array2i& blockOrigin,
    const Eigen::Array2i& blockSize, double sigma, const Eigen::Array2i& offset,
    const Eigen::Array2i& cellSize)
{
    const Eigen::Array2d size = cellSize.cast<double>();
    const Eigen::Array2d center = offset.cast<double>() + (size - 1) / 2;

    const Eigen::Array2i first = (offset - cellSize).max(0);
    const Eigen::Array2i last =
        (offset + 2 * cellSize)
            .min(Eigen::Array2i{votes.dimension(0), votes.dimension(1)});

    Eigen::Tensor<double, 1> h(votes.dimension(2));
    h.setZero();

    for (int i = first.x(); i < last.x(); ++i) {
        const double wx =
            std::max(0.0, 1 - std::abs(i - center.x()) / size.x()) *
            gaussianWeight(i - blockOrigin.x(), blockSize.x(), sigma);

        for (int j = first.y(); j < last.y(); ++j) {
            const double wy =
                std::max(0.0, 1 - std::abs(j - center.y()) / size.y()) *
                gaussianWeight(j - blockOrigin.y(), blockSize.y(), sigma);

            h += votes.chip<0>(i).chip<0>(j) * (wx * wy);
        }
    }

    return h;
}

template<class Scalar>
using WeightedTrilinearDescriptor = hogpp::IntegralHOGDescriptor<
    Scalar, hogpp::Gradient<Scalar>, hogpp::GradientMagnitude<Scalar>,
    hogpp::UnsignedGradient<Scalar>, IdentityBlockNormalizer,
    hogpp::GaussianWeighting<Scalar>, hogpp::TrilinearInterpolation>;

// Largest absolute deviation of the weighted and interpolated cell histograms
// of all blocks from the reference relative to the largest absolute reference
// value.
template<class Scalar>
[[nodiscard]] double weightedRelativeError(
    const Eigen::Tensor<double, 3>& image, const Eigen::Array2i& cellSize,
    const Eigen::Array2i& blockSize, const Eigen::Array2i& stride, double sigma)
{
    const Eigen::Tensor<double, 3> votes = pixelVotes(image);

    WeightedTrilinearDescriptor<Scalar> d{
        hogpp::Gradient<Scalar>{}, hogpp::GradientMagnitude<Scalar>{},
        hogpp::UnsignedGradient<Scalar>{}, IdentityBlockNormalizer{},
        hogpp::GaussianWeighting<Scalar>{static_cast<Scalar>(sigma)}};
    d.setCellSize(cellSize);
    d.setBlockSize(blockSize);
    d.setBlockStride(stride);
    d.compute(Eigen::Tensor<Scalar, 3>{image.cast<Scalar>()});

    const Eigen::Tensor<Scalar, 5> X = d.features();
    const Eigen::Array2i lastBlockOffset =
        Eigen::Array2i{image.dimension(0), image.dimension(1)} - blockSize;

    double maxAbsError = 0;
    double maxAbsExpected = 0;

    for (int i = 0; i < X.dimension(0); ++i) {
        for (int j = 0; j < X.dimension(1); ++j) {
            const Eigen::Array2i blockOrigin =
                (Eigen::Array2i{i, j} * stride).min(lastBlockOffset);

            for (int k = 0; k < X.dimension(2); ++k) {
                for (int l = 0; l < X.dimension(3); ++l) {
                    const Eigen::Tensor<double, 1> expected =
                        bruteForceWeightedBilinearSplat(
                            votes, blockOrigin, blockSize, sigma,
                            blockOrigin + Eigen::Array2i{k, l} * cellSize,
                            cellSize);

                    for (Eigen::DenseIndex b = 0; b < expected.dimension(0);
                         ++b) {
                        const auto actual =
                            static_cast<double>(X(i, j, k, l, b));
                        maxAbsError = std::max(maxAbsError,
                                               std::abs(actual - expected(b)));
                        maxAbsExpected =
                            std::max(maxAbsExpected, std::abs(expected(b)));
                    }
                }
            }
        }
    }

    return maxAbsError / maxAbsExpected;
}

} // namespace

// Combines the Gaussian down-weighting with the trilinear interpolation for
// block strides at and below the cell size, which select different
// evaluations of the weighting, and for blocks placed flush with the border.
BOOST_AUTO_TEST_CASE_TEMPLATE(trilinear_interpolation_with_gaussian_weighting,
                              Scalar, Scalars)
{
    constexpr Eigen::DenseIndex Rows = 30;
    constexpr Eigen::DenseIndex Cols = 37;
    constexpr double Sigma = 5;

    const Eigen::Array2i cellSize{4, 5};
    const Eigen::Array2i blockSize = 2 * cellSize;
    const std::array strides{cellSize, Eigen::Array2i{2, 3}};

    const Eigen::Tensor<double, 3> image = makeRandomImage(Rows, Cols);

    const double tolerance = referenceTolerance<Scalar>();

    for (const Eigen::Array2i& stride : strides) {
        BOOST_TEST(weightedRelativeError<Scalar>(image, cellSize, blockSize,
                                                 stride, Sigma) <= tolerance);
    }
}

// The weighted votes are recovered from the single precision integral
// histogram, which loses accuracy with the image size. The weighting must not
// lose substantially more than the unweighted interpolation on top of that.
BOOST_AUTO_TEST_CASE(
    trilinear_interpolation_with_gaussian_weighting_large_float_image)
{
    constexpr Eigen::DenseIndex Size = 256;
    constexpr double Sigma = 8;
    // Admissible ratio of both relative errors
    constexpr double Slack = 2;

    const Eigen::Tensor<double, 3> image = makeRandomImage(Size, Size);
    const Eigen::Array2i cellSize{8, 8};

    const double trilinearError =
        relativeError<float, hogpp::TrilinearInterpolation>(
            image, cellSize, bruteForceBilinearSplat);
    const double weightedError = weightedRelativeError<float>(
        image, cellSize, 2 * cellSize, cellSize, Sigma);

    BOOST_TEST(weightedError <= Slack * trilinearError);
}

// A single non-finite pixel (e.g. from masked/invalid input data) must
// not violate the internal "gradient magnitude is strictly positive at
// this point" invariant used while voting, since that invariant
// degrades to undefined behavior in release builds.
BOOST_AUTO_TEST_CASE_TEMPLATE(nan_pixel, Scalar, Scalars)
{
    auto image = makeImage<Scalar>();
    image(1, 1, 0) = std::numeric_limits<Scalar>::quiet_NaN();

    hogpp::IntegralHOGDescriptor<Scalar> d;
    d.compute(image);

    const Eigen::Tensor<Scalar, 3>& h = d.histogram();

    bool allFinite = true;

    for (Eigen::DenseIndex i = 0; i < h.size(); ++i) {
        using std::isfinite;

        if (!isfinite(h.data()[i])) {
            allFinite = false;
            break;
        }
    }

    BOOST_TEST(allFinite);
}

// SignedGradient<Scalar, Fast> (and UnsignedGradient) provide a
// tensor-expression operator() overload that IntegralHOGDescriptor::compute()
// uses to precompute the per-pixel bin weight for the whole image as a
// single vectorized pass ahead of the histogram scan (see
// detail::HasTensorBinning in integralhogdescriptor.hpp), instead of
// calling the approximation once per pixel inside the scan. This exercises
// that wiring end to end: the precomputed-weight result must agree with
// calling the exact reference implementation per pixel, within the same
// tolerance already established for the Fast binning approximation itself.
BOOST_AUTO_TEST_CASE_TEMPLATE(precomputed_binning_matches_reference, Scalar,
                              Scalars)
{
    namespace tt = boost::test_tools;

    Eigen::Tensor<Scalar, 3> image(16, 16, 1);

    for (Eigen::DenseIndex i = 0; i < image.dimension(0); ++i) {
        for (Eigen::DenseIndex j = 0; j < image.dimension(1); ++j) {
            image(i, j, 0) =
                static_cast<Scalar>((i * image.dimension(1) + j) % 7);
        }
    }

    hogpp::IntegralHOGDescriptor<Scalar, hogpp::Gradient<Scalar>,
                                 hogpp::GradientMagnitude<Scalar>,
                                 hogpp::SignedGradient<Scalar, hogpp::Accurate>>
        referenceDescriptor;
    hogpp::IntegralHOGDescriptor<Scalar, hogpp::Gradient<Scalar>,
                                 hogpp::GradientMagnitude<Scalar>,
                                 hogpp::SignedGradient<Scalar, hogpp::Fast>>
        precomputedDescriptor;

    referenceDescriptor.compute(image);
    precomputedDescriptor.compute(image);

    const auto referenceFeatures = referenceDescriptor.features();
    const auto precomputedFeatures = precomputedDescriptor.features();

    BOOST_TEST_REQUIRE(referenceFeatures.size() == precomputedFeatures.size());

    // boost::unit_test::tolerance() compares relative to the operands'
    // magnitude, which is unstable for feature values close to zero
    // (L2Hys-normalized entries are often small). Both operands are
    // shifted away from zero by a fixed offset first, exactly as in
    // tests/cpp/test_binning.cpp's *_fast tests, so the same relative
    // tolerance enforces a tight absolute bound instead.
    for (Eigen::DenseIndex i = 0; i < referenceFeatures.size(); ++i) {
        const Scalar precomputedShifted =
            precomputedFeatures.data()[i] + Scalar{1};
        const Scalar referenceShifted = referenceFeatures.data()[i] + Scalar{1};

        BOOST_TEST(precomputedShifted == referenceShifted,
                   tt::tolerance(Scalar(1e-4L)));
    }
}

// With a single channel, IntegralHOGDescriptor::compute() skips
// Eigen's argmax(2) reduction entirely (see channels > 1 in
// integralhogdescriptor.hpp) since the maximum over one element is
// trivially that element. This guards the channels > 1 path still
// genuinely selects the channel with the largest gradient magnitude,
// rather than e.g. always defaulting to channel 0: a multi-channel
// image whose first channel is constant (zero gradient everywhere,
// so it can never be the argmax) and whose second channel carries
// the same data as a single-channel reference image must produce the
// same descriptor as that reference image.
BOOST_AUTO_TEST_CASE_TEMPLATE(multi_channel_selects_max_magnitude_channel,
                              Scalar, Scalars)
{
    namespace tt = boost::test_tools;

    constexpr Eigen::DenseIndex rows = 16;
    constexpr Eigen::DenseIndex cols = 16;

    Eigen::Tensor<Scalar, 3> referenceImage(rows, cols, 1);
    Eigen::Tensor<Scalar, 3> multiChannelImage(rows, cols, 2);

    for (Eigen::DenseIndex i = 0; i < rows; ++i) {
        for (Eigen::DenseIndex j = 0; j < cols; ++j) {
            const auto value = static_cast<Scalar>((i * cols + j) % 7);
            referenceImage(i, j, 0) = value;
            multiChannelImage(i, j, 0) = Scalar{3}; // constant: zero gradient
            multiChannelImage(i, j, 1) = value;
        }
    }

    hogpp::IntegralHOGDescriptor<Scalar> referenceDescriptor;
    hogpp::IntegralHOGDescriptor<Scalar> multiChannelDescriptor;

    referenceDescriptor.compute(referenceImage);
    multiChannelDescriptor.compute(multiChannelImage);

    const auto referenceFeatures = referenceDescriptor.features();
    const auto multiChannelFeatures = multiChannelDescriptor.features();

    BOOST_TEST_REQUIRE(referenceFeatures.size() == multiChannelFeatures.size());

    for (Eigen::DenseIndex i = 0; i < referenceFeatures.size(); ++i) {
        const Scalar multiChannelShifted =
            multiChannelFeatures.data()[i] + Scalar{1};
        const Scalar referenceShifted = referenceFeatures.data()[i] + Scalar{1};

        BOOST_TEST(multiChannelShifted == referenceShifted,
                   tt::tolerance(Scalar(1e-4L)));
    }
}
