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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <tuple>
#include <utility>

#include <boost/histogram.hpp>
#include <boost/test/included/unit_test.hpp>

#include <hogpp/integralhistogram.hpp>

namespace bh = boost::histogram;
using Axis = bh::axis::integer<int, bh::use_default, bh::axis::option::none_t>;

struct RandomImage
{
    static constexpr Eigen::DenseIndex M = 113;
    static constexpr Eigen::DenseIndex N = 152;
    static constexpr Eigen::DenseIndex y1 = 45;
    static constexpr Eigen::DenseIndex x1 = 35;
    static constexpr Eigen::DenseIndex Rows = 25;
    static constexpr Eigen::DenseIndex Cols = 30;
    static constexpr Eigen::DenseIndex y2 = y1 + Rows;
    static constexpr Eigen::DenseIndex x2 = x1 + Cols;
    static constexpr int Bins = std::numeric_limits<std::uint8_t>::max() + 1;

    RandomImage()
    {
        std::generate_n(image.data(), image.size(), [&x = x, &g = g] {
            return static_cast<std::uint8_t>(
                static_cast<int>(std::round(x(g) * 127)) + 128);
        });

        auto h1 = bh::make_histogram(Axis{0, Bins});
        std::for_each(image.data(), image.data() + image.size(), std::ref(h1));

        subimage = image.block(y1, x1, Rows, Cols);

        auto h2 = bh::make_histogram(Axis{0, Bins});
        std::for_each(subimage.data(), subimage.data() + subimage.size(),
                      std::ref(h2));

        ih.resize(std::make_tuple(image.rows(), image.cols()), Bins);

        ih.scan([&image = image](Eigen::TensorRef<Eigen::Tensor<int, 1>> h,
                                 const std::tuple<int, int>& i) {
            std::uint8_t value = std::apply(std::ref(image), i);
            ++h.coeffRef(value);
        });

        Eigen::Tensor<int, 1> ih1 =
            ih.intersect(std::make_tuple(0, 0), std::make_tuple(M, N));
        Eigen::Tensor<int, 1> ih2 =
            ih.intersect(std::make_tuple(y1, x1), std::make_tuple(y2, x2));

        std::copy(h1.cbegin(), h1.cend(), ref_hist1.begin());
        std::copy(h2.cbegin(), h2.cend(), ref_hist2.begin());

        std::copy(ih1.data(), ih1.data() + ih1.size(), computed_hist1.begin());
        std::copy(ih2.data(), ih2.data() + ih2.size(), computed_hist2.begin());
    }

    std::normal_distribution<double> x;
    std::default_random_engine g;
    Eigen::Matrix<std::uint8_t, M, N> image;
    Eigen::Matrix<std::uint8_t, Rows, Cols> subimage;

    hogpp::IntegralHistogram<int, 2, 1> ih;

    std::array<int, Bins> ref_hist1;
    std::array<int, Bins> ref_hist2;

    std::array<int, Bins> computed_hist1;
    std::array<int, Bins> computed_hist2;
};

BOOST_FIXTURE_TEST_CASE(full, RandomImage)
{
    BOOST_TEST(ref_hist1 == computed_hist1, boost::test_tools::per_element());
}

BOOST_FIXTURE_TEST_CASE(sub, RandomImage)
{
    BOOST_TEST(ref_hist2 == computed_hist2, boost::test_tools::per_element());
}

// The sign of the corners combined while propagating the wavefront
// scan generalizes to any number of input dimensions.
BOOST_AUTO_TEST_CASE(volumetric)
{
    constexpr Eigen::DenseIndex Nx = 4;
    constexpr Eigen::DenseIndex Ny = 5;
    constexpr Eigen::DenseIndex Nz = 6;

    hogpp::IntegralHistogram<long long, 3, 1> ih;
    ih.resize(std::make_tuple(Nx, Ny, Nz), 1);

    ih.scan([](Eigen::TensorRef<Eigen::Tensor<long long, 1>> h,
               const std::tuple<Eigen::DenseIndex, Eigen::DenseIndex,
                                Eigen::DenseIndex>& /*unused*/) {
        ++h.coeffRef(0);
    });

    const Eigen::Tensor<long long, 1> full =
        ih.intersect(std::make_tuple(0, 0, 0), std::make_tuple(Nx, Ny, Nz));
    BOOST_TEST(full(0) == Nx * Ny * Nz);

    const Eigen::Tensor<long long, 1> sub =
        ih.intersect(std::make_tuple(1, 1, 1), std::make_tuple(3, 4, 5));
    BOOST_TEST(sub(0) == (3 - 1) * (4 - 1) * (5 - 1));
}

// Compile-time check of the corner sign used by the wavefront
// recurrence, for D == 1, 2 and 3, against a hand-derived truth table
// (K enumerates every corner but the all-bits-set one, which is the
// cell being computed rather than a previously computed neighbor).
static_assert(!hogpp::detail::cornerSignbit<1>(0),
              "D=1, K=0: the only corner is a positive term");

static_assert(hogpp::detail::cornerSignbit<2>(0),
              "D=2, K=0: the diagonal corner is a negative term");
static_assert(!hogpp::detail::cornerSignbit<2>(1),
              "D=2, K=1: a single-axis corner is a positive term");
static_assert(!hogpp::detail::cornerSignbit<2>(2),
              "D=2, K=2: a single-axis corner is a positive term");

static_assert(!hogpp::detail::cornerSignbit<3>(0),
              "D=3, K=0: the diagonal corner is a positive term");
static_assert(hogpp::detail::cornerSignbit<3>(1),
              "D=3, K=1: a single-axis corner is a negative term");
static_assert(hogpp::detail::cornerSignbit<3>(2),
              "D=3, K=2: a single-axis corner is a negative term");
static_assert(!hogpp::detail::cornerSignbit<3>(3),
              "D=3, K=3: a two-axis corner is a positive term");
static_assert(hogpp::detail::cornerSignbit<3>(4),
              "D=3, K=4: a single-axis corner is a negative term");
static_assert(!hogpp::detail::cornerSignbit<3>(5),
              "D=3, K=5: a two-axis corner is a positive term");
static_assert(!hogpp::detail::cornerSignbit<3>(6),
              "D=3, K=6: a two-axis corner is a positive term");

// cascade() must be the prefix sum of scan()'s first-order histogram:
// for votes(i, j) == 1 everywhere, H1(i, j) == i * j, so its prefix sum
// H2(i, j) == (i * (i - 1) / 2) * (j * (j - 1) / 2). The second-order
// histogram has one more slot per input dimension than the first-order one.
BOOST_AUTO_TEST_CASE(cascade_matches_closed_form_for_constant_field)
{
    constexpr Eigen::DenseIndex Rows = 12;
    constexpr Eigen::DenseIndex Cols = 9;

    hogpp::IntegralHistogram<long long, 2, 1> ih;
    ih.resize(std::make_tuple(Rows, Cols), 1);

    ih.scan([](Eigen::TensorRef<Eigen::Tensor<long long, 1>> h,
               const std::tuple<Eigen::DenseIndex, Eigen::DenseIndex>&
               /*unused*/) { h.coeffRef(0) += 1; });

    const auto cascaded = ih.cascade();
    const Eigen::Tensor<long long, 3>& h2 = cascaded.histogram();

    BOOST_TEST(h2.dimension(0) == Rows + 2);
    BOOST_TEST(h2.dimension(1) == Cols + 2);

    for (Eigen::DenseIndex i = 0; i < h2.dimension(0); ++i) {
        for (Eigen::DenseIndex j = 0; j < h2.dimension(1); ++j) {
            const long long expected = (i * (i - 1) / 2) * (j * (j - 1) / 2);
            BOOST_TEST(h2(i, j, 0) == expected);
        }
    }
}

// Reference bilinear splat of the votes into the cell [a, b) computed
// independently of the integral histogram machinery. Each position is
// weighted by its distance to the geometric center of the cell.
template<class Votes>
[[nodiscard]] double bruteForceBilinearSplat(
    const Votes& votes, const std::array<Eigen::DenseIndex, 2>& a,
    const std::array<Eigen::DenseIndex, 2>& b)
{
    const auto sx = static_cast<double>(b[0] - a[0]);
    const auto sy = static_cast<double>(b[1] - a[1]);
    const double cx = static_cast<double>(a[0] + b[0] - 1) / 2;
    const double cy = static_cast<double>(a[1] + b[1] - 1) / 2;

    double total = 0.0;

    for (Eigen::DenseIndex i = 0; i < votes.dimension(0); ++i) {
        const double wx =
            std::max(0.0, 1.0 - std::abs(static_cast<double>(i) - cx) / sx);

        for (Eigen::DenseIndex j = 0; j < votes.dimension(1); ++j) {
            const double wy =
                std::max(0.0, 1.0 - std::abs(static_cast<double>(j) - cy) / sy);
            total += votes(i, j) * wx * wy;
        }
    }

    return total;
}

// interpolate() must reproduce a brute-force per-pixel bilinear splat for
// every cell of a grid tiling the field, including the cells at the domain
// boundary, for both even and odd cell sizes.
BOOST_AUTO_TEST_CASE(interpolate_matches_brute_force_bilinear_splat)
{
    constexpr Eigen::DenseIndex Rows = 40;
    constexpr Eigen::DenseIndex Cols = 42;

    std::default_random_engine g{1};
    std::uniform_real_distribution<double> value{0.0, 1.0};
    std::bernoulli_distribution keep{0.6};

    Eigen::Tensor<double, 2> votes(Rows, Cols);
    votes.setZero();

    for (Eigen::DenseIndex i = 0; i < Rows; ++i) {
        for (Eigen::DenseIndex j = 0; j < Cols; ++j) {
            if (keep(g)) {
                votes(i, j) = value(g);
            }
        }
    }

    hogpp::IntegralHistogram<double, 2, 1> ih;
    ih.resize(std::make_tuple(Rows, Cols), 1);

    ih.scan(
        [&votes](Eigen::TensorRef<Eigen::Tensor<double, 1>> h,
                 const std::tuple<Eigen::DenseIndex, Eigen::DenseIndex>& i) {
            h.coeffRef(0) += std::apply(votes, i);
        });

    const auto h2 = ih.cascade();

    constexpr double Tolerance = 1e-9;

    for (const auto& [sx, sy] :
         {std::pair<Eigen::DenseIndex, Eigen::DenseIndex>{8, 6},
          std::pair<Eigen::DenseIndex, Eigen::DenseIndex>{5, 7}}) {
        double maxAbsError = 0.0;

        for (Eigen::DenseIndex x = 0; x + sx <= Rows; x += sx) {
            for (Eigen::DenseIndex y = 0; y + sy <= Cols; y += sy) {
                const Eigen::Tensor<double, 1> interpolated = h2.interpolate(
                    std::make_tuple(x, y), std::make_tuple(x + sx, y + sy));

                const double expected =
                    bruteForceBilinearSplat(votes, {x, y}, {x + sx, y + sy});

                maxAbsError =
                    std::max(maxAbsError, std::abs(interpolated(0) - expected));
            }
        }

        BOOST_TEST(maxAbsError < Tolerance);
    }
}

// Unsigned coordinates of a cell close to the lower domain boundary must not
// wrap around while computing the taps preceding the cell.
BOOST_AUTO_TEST_CASE(interpolate_unsigned_coordinates_near_lower_boundary)
{
    constexpr Eigen::DenseIndex Size = 16;
    constexpr std::size_t CellSize = 8;

    hogpp::IntegralHistogram<double, 2, 1> ih;
    ih.resize(std::make_tuple(Size, Size), 1);

    ih.scan([](Eigen::TensorRef<Eigen::Tensor<double, 1>> h,
               const std::tuple<Eigen::DenseIndex, Eigen::DenseIndex>&
               /*unused*/) { h.coeffRef(0) += 1; });

    const Eigen::Tensor<double, 1> actual = ih.cascade().interpolate(
        std::make_tuple(std::size_t{0}, std::size_t{0}),
        std::make_tuple(CellSize, CellSize));

    Eigen::Tensor<double, 2> votes(Size, Size);
    votes.setConstant(1);

    const double expected =
        bruteForceBilinearSplat(votes, {0, 0},
                                {static_cast<Eigen::DenseIndex>(CellSize),
                                 static_cast<Eigen::DenseIndex>(CellSize)});

    constexpr double Tolerance = 1e-12;

    BOOST_TEST(actual(0) == expected, boost::test_tools::tolerance(Tolerance));
}
