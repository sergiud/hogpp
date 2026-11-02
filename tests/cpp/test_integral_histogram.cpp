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
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/histogram.hpp>
#include <boost/mp11/list.hpp>
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

// Interpolating must not depend on the memory layout of the histogram. The
// taps are summed in the same order for both layouts, so the results must
// agree exactly. Multiple bins and a non-square domain make each stride
// distinguishable.
BOOST_AUTO_TEST_CASE(interpolate_row_major_matches_column_major)
{
    constexpr Eigen::DenseIndex Rows = 11;
    constexpr Eigen::DenseIndex Cols = 14;
    constexpr Eigen::DenseIndex Bins = 3;

    std::default_random_engine g{1};
    std::uniform_real_distribution<double> value{-1.0, 1.0};

    Eigen::Tensor<double, 3> colMajor(Rows, Cols, Bins);
    Eigen::Tensor<double, 3, Eigen::RowMajor> rowMajor(Rows, Cols, Bins);

    for (Eigen::DenseIndex b = 0; b < Bins; ++b) {
        for (Eigen::DenseIndex j = 0; j < Cols; ++j) {
            for (Eigen::DenseIndex i = 0; i < Rows; ++i) {
                colMajor(i, j, b) = value(g);
                rowMajor(i, j, b) = colMajor(i, j, b);
            }
        }
    }

    hogpp::IntegralHistogram<double, 2, 1> colMajorHistogram;
    colMajorHistogram.setHistogram(colMajor);
    hogpp::IntegralHistogram<double, 2, 1, Eigen::RowMajor> rowMajorHistogram;
    rowMajorHistogram.setHistogram(rowMajor);

    for (const auto& [sx, sy] :
         {std::pair<Eigen::DenseIndex, Eigen::DenseIndex>{4, 6},
          std::pair<Eigen::DenseIndex, Eigen::DenseIndex>{3, 5}}) {
        for (Eigen::DenseIndex x = 0; x + sx < Rows; x += sx) {
            for (Eigen::DenseIndex y = 0; y + sy < Cols; y += sy) {
                const auto a = std::make_tuple(x, y);
                const auto b = std::make_tuple(x + sx, y + sy);

                const Eigen::Tensor<double, 1> expected =
                    colMajorHistogram.interpolate(a, b);
                const Eigen::Tensor<double, 1, Eigen::RowMajor> actual =
                    rowMajorHistogram.interpolate(a, b);

                for (Eigen::DenseIndex k = 0; k < Bins; ++k) {
                    BOOST_TEST(actual(k) == expected(k),
                               "bin " << k << " of the cell at (" << x << ", "
                                      << y << ") of size (" << sx << ", " << sy
                                      << ")");
                }
            }
        }
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

// Compile-time check that TupleOfIntegrals actually evaluates its
// constraint instead of merely checking the expression compiles.
static_assert(hogpp::detail::TupleOfIntegrals<std::tuple<int, int>>,
              "a tuple of integral coordinates must satisfy the concept");
static_assert(
    !hogpp::detail::TupleOfIntegrals<std::tuple<double, double>>,
    "a tuple of non-integral coordinates must not satisfy the concept");
static_assert(!hogpp::detail::TupleOfIntegrals<int>,
              "a type that is not a tuple must not satisfy the concept");

// Input and histogram space extents of the integral histograms whose scan is
// compared with the tensor expression recurrence, in either data layout.
template<int Layout>
struct Plane
{
    static constexpr int Options = Layout;
    static constexpr std::array<Eigen::DenseIndex, 2> Input{13, 11};
    static constexpr std::array<Eigen::DenseIndex, 1> Bins{9};
};

template<int Layout>
struct Volume
{
    static constexpr int Options = Layout;
    static constexpr std::array<Eigen::DenseIndex, 3> Input{5, 6, 7};
    static constexpr std::array<Eigen::DenseIndex, 1> Bins{4};
};

using ScanShapes =
    boost::mp11::mp_list<Plane<Eigen::ColMajor>, Plane<Eigen::RowMajor>,
                         Volume<Eigen::ColMajor>, Volume<Eigen::RowMajor>>;

namespace {

template<class T, std::size_t Size>
[[nodiscard]] auto toTuple(const std::array<T, Size>& a)
{
    return std::apply([](auto... v) { return std::make_tuple(v...); }, a);
}

// Bit patterns compare floating-point values exactly, including the sign of
// zero, without tripping -Wfloat-equal.
template<class Tensor>
[[nodiscard]] std::vector<std::uint32_t> bitPatterns(const Tensor& t)
{
    std::vector<std::uint32_t> result(static_cast<std::size_t>(t.size()));
    std::transform(
        t.data(), t.data() + t.size(), result.begin(),
        [](float value) { return std::bit_cast<std::uint32_t>(value); });
    return result;
}

} // namespace

// The scan must reproduce the recurrence of the tensor expressions exactly,
// i.e., add the same values in the same order. Votes spanning many orders of
// magnitude with both signs make the rounding of every addition depend on
// that order.
BOOST_AUTO_TEST_CASE_TEMPLATE(scan_matches_tensor_expression, Shape, ScanShapes)
{
    constexpr auto N = static_cast<Eigen::DenseIndex>(Shape::Input.size());
    constexpr auto K = static_cast<Eigen::DenseIndex>(Shape::Bins.size());
    constexpr float MinExponent = -20;
    constexpr float MaxExponent = 20;

    constexpr std::size_t Rank = Shape::Input.size() + Shape::Bins.size();

    using Tensor = Eigen::Tensor<float, N + K, Shape::Options>;
    using Bins = Eigen::Tensor<float, K, Shape::Options>;

    std::array<Eigen::DenseIndex, Rank> votesDims{};
    std::copy(Shape::Input.begin(), Shape::Input.end(), votesDims.begin());
    std::copy(Shape::Bins.begin(), Shape::Bins.end(), votesDims.begin() + N);

    std::default_random_engine g{1};
    std::uniform_real_distribution<float> exponent{MinExponent, MaxExponent};
    std::bernoulli_distribution negative;

    Tensor votes(votesDims);
    std::generate_n(votes.data(), votes.size(), [&g, &exponent, &negative] {
        const float magnitude = std::exp2(exponent(g));
        return negative(g) ? -magnitude : magnitude;
    });

    hogpp::IntegralHistogram<float, N, K, Shape::Options> ih;
    ih.resize(toTuple(Shape::Input), toTuple(Shape::Bins));
    ih.scan([&votes](Eigen::TensorRef<Bins> h, const auto& i) {
        const Bins v = hogpp::detail::chip(votes, i);

        for (Eigen::DenseIndex m = 0; m < v.size(); ++m) {
            h.coeffRef(m) += v.data()[m];
        }
    });

    std::array<Eigen::DenseIndex, Rank> expectedDims = votesDims;

    for (Eigen::DenseIndex d = 0; d < N; ++d) {
        ++expectedDims[static_cast<std::size_t>(d)];
    }

    Tensor expected(expectedDims);
    expected.setZero();

    hogpp::cartesianProduct(
        toTuple(Shape::Input), [&expected, &votes](const auto& i) {
            hogpp::detail::chip(expected, hogpp::detail::offset<1>(i)) =
                hogpp::detail::propagate(expected, i) +
                hogpp::detail::chip(votes, i);
        });

    BOOST_TEST(bitPatterns(ih.histogram()) == bitPatterns(expected),
               boost::test_tools::per_element());
}
