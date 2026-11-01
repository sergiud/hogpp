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

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>

#include <hogpp/integralhogdescriptor.hpp>

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
