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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include <hogpp/gradient.hpp>

#include <boost/mpl/list.hpp>
#include <boost/test/included/unit_test.hpp>

using Scalars = boost::mpl::list<float, double, long double>;

// A constant-slope signal must produce the same gradient magnitude at
// every column, border columns included, since the border and
// interior samplers estimate the same per-pixel-distance derivative.
BOOST_AUTO_TEST_CASE_TEMPLATE(interior_border_magnitude_comparable, Scalar,
                              Scalars)
{
    Eigen::TensorFixedSize<Scalar, Eigen::Sizes<1, 6, 1>> image;

    // Initialize the entire tensor
    for (Eigen::DenseIndex j = 0; j < image.dimension(1); ++j) {
        image(0, j, 0) = static_cast<Scalar>(j);
    }

    hogpp::Gradient<Scalar> gradient;

    Eigen::Tensor<Scalar, 3> dxs;
    Eigen::Tensor<Scalar, 3> dys;

    std::tie(dxs, dys) = gradient(image);

    for (Eigen::DenseIndex j = 0; j < image.dimension(1); ++j) {
        BOOST_TEST(dxs(0, j, 0) == Scalar(1));
    }

    using Derived = decltype(image);
    const Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors>& base = image;
    const hogpp::CentralDifferences<Scalar, hogpp::Horizontal_t> sampler;

    BOOST_TEST(sampler(base, 0, 2, 0) == Scalar(1));
}

// An axis of size 1 has no well-defined finite difference (the lower- and
// upper-border conditions are both satisfied by the single valid index at
// once), and must not read past the end of that axis.
BOOST_AUTO_TEST_CASE_TEMPLATE(single_row, Scalar, Scalars)
{
    Eigen::Tensor<Scalar, 3> image(1, 4, 1);
    image.setValues({{{Scalar(10)}, {Scalar(20)}, {Scalar(30)}, {Scalar(40)}}});

    hogpp::Gradient<Scalar> gradient;

    Eigen::Tensor<Scalar, 3> dxs;
    Eigen::Tensor<Scalar, 3> dys;

    std::tie(dxs, dys) = gradient(image);

    for (Eigen::DenseIndex j = 0; j < image.dimension(1); ++j) {
        BOOST_TEST(dys(0, j, 0) == Scalar(0));
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(single_column, Scalar, Scalars)
{
    Eigen::Tensor<Scalar, 3> image(4, 1, 1);
    image.setValues(
        {{{Scalar(10)}}, {{Scalar(20)}}, {{Scalar(30)}}, {{Scalar(40)}}});

    hogpp::Gradient<Scalar> gradient;

    Eigen::Tensor<Scalar, 3> dxs;
    Eigen::Tensor<Scalar, 3> dys;

    std::tie(dxs, dys) = gradient(image);

    for (Eigen::DenseIndex i = 0; i < image.dimension(0); ++i) {
        BOOST_TEST(dxs(i, 0, 0) == Scalar(0));
    }
}

namespace {

template<class Scalar>
[[nodiscard]] std::vector<Scalar> values(const Eigen::Tensor<Scalar, 3>& t)
{
    return {t.data(), t.data() + t.size()};
}

// Derivative along Axis computed for every pixel on its own with the sampler
// the gradient selects for the pixel's position along that axis.
template<class Axis, class Scalar>
[[nodiscard]] Eigen::Tensor<Scalar, 3> sampledDerivative(
    const Eigen::Tensor<Scalar, 3>& image)
{
    constexpr int Dimension = std::is_same_v<Axis, hogpp::Vertical_t> ? 0 : 1;

    const hogpp::CentralDifferences<Scalar, Axis> interior;
    const hogpp::ForwardDifferences<Scalar, Axis> lower;
    const hogpp::BackwardDifferences<Scalar, Axis> upper;
    const Eigen::TensorBase<Eigen::Tensor<Scalar, 3>, Eigen::ReadOnlyAccessors>&
        base = image;

    const Eigen::DenseIndex n = image.dimension(Dimension);
    Eigen::Tensor<Scalar, 3> result(image.dimensions());

    for (Eigen::DenseIndex k = 0; k < image.dimension(2); ++k) {
        for (Eigen::DenseIndex j = 0; j < image.dimension(1); ++j) {
            for (Eigen::DenseIndex i = 0; i < image.dimension(0); ++i) {
                const Eigen::DenseIndex p = Dimension == 0 ? i : j;

                if (n == 1) {
                    result(i, j, k) = Scalar{0};
                }
                else if (p == 0) {
                    result(i, j, k) = lower(image, i, j, k);
                }
                else if (p == n - 1) {
                    result(i, j, k) = upper(image, i, j, k);
                }
                else {
                    result(i, j, k) = interior(base, i, j, k);
                }
            }
        }
    }

    return result;
}

} // namespace

// Each derivative must be the one of the sampler that applies at the pixel's
// position along the derivative's axis, regardless of the order in which the
// gradient visits the pixels. Single rows and columns have no neighbors to
// differentiate.
BOOST_AUTO_TEST_CASE_TEMPLATE(gradient_matches_samplers, Scalar, Scalars)
{
    constexpr Eigen::DenseIndex Channels = 3;
    constexpr std::array<std::array<Eigen::DenseIndex, 2>, 5> Shapes{
        {{13, 17}, {17, 13}, {1, 5}, {5, 1}, {1, 1}}};

    std::default_random_engine g{1};
    std::uniform_real_distribution<Scalar> intensity{0, 1};
    const hogpp::Gradient<Scalar> gradient;

    for (const auto& [rows, cols] : Shapes) {
        BOOST_TEST_CONTEXT("image of " << rows << "x" << cols << " pixels")
        {
            Eigen::Tensor<Scalar, 3> image(rows, cols, Channels);
            std::generate_n(image.data(), image.size(),
                            [&g, &intensity] { return intensity(g); });

            const auto [dxs, dys] = gradient(image);

            // Exact comparison, since the same samplers must yield the same
            // values.
            BOOST_TEST(
                values<Scalar>(dxs) ==
                    values(sampledDerivative<hogpp::Horizontal_t>(image)),
                boost::test_tools::per_element());
            BOOST_TEST(values<Scalar>(dys) ==
                           values(sampledDerivative<hogpp::Vertical_t>(image)),
                       boost::test_tools::per_element());
        }
    }
}

namespace {

// Central difference along Axis at the center of a 3x3 image whose neighbors
// along that axis are a and b.
template<class Axis, class Scalar, class Value>
[[nodiscard]] Scalar centralDifference(Value a, Value b)
{
    Eigen::Tensor<Value, 3> image(3, 3, 1);
    image.setZero();

    if constexpr (std::is_same_v<Axis, hogpp::Vertical_t>) {
        image(0, 1, 0) = a;
        image(2, 1, 0) = b;
    }
    else {
        image(1, 0, 0) = a;
        image(1, 2, 0) = b;
    }

    const Eigen::TensorBase<Eigen::Tensor<Value, 3>, Eigen::ReadOnlyAccessors>&
        base = image;
    return hogpp::CentralDifferences<Scalar, Axis>{}(base, 1, 1, 0);
}

using Integers = boost::mpl::list<std::uint8_t, std::int16_t, int>;

} // namespace

// The central difference of floating-point images must round exactly like
// std::midpoint of the negated lower and the upper neighbor. The neighbors
// span many orders of magnitude, but stay normal when halved.
BOOST_AUTO_TEST_CASE_TEMPLATE(central_differences_match_midpoint, Scalar,
                              Scalars)
{
    constexpr int Samples = 1000;
    constexpr Scalar MinExponent = -100;
    constexpr Scalar MaxExponent = 100;

    std::default_random_engine g{1};
    std::uniform_real_distribution<Scalar> exponent{MinExponent, MaxExponent};
    std::bernoulli_distribution negative;

    auto sample = [&g, &exponent, &negative] {
        using std::exp2;
        const Scalar magnitude = exp2(exponent(g));
        return negative(g) ? -magnitude : magnitude;
    };

    for (int n = 0; n < Samples; ++n) {
        const Scalar a = sample();
        const Scalar b = sample();
        const Scalar expected = std::midpoint(-a, b);

        BOOST_TEST_CONTEXT("lower neighbor " << a << ", upper neighbor " << b)
        {
            BOOST_TEST((centralDifference<hogpp::Horizontal_t, Scalar>(a, b)) ==
                       expected);
            BOOST_TEST((centralDifference<hogpp::Vertical_t, Scalar>(a, b)) ==
                       expected);
        }
    }
}

// Neighbors at opposite ends of the representable range have a difference
// that exceeds it, but half of it does not.
BOOST_AUTO_TEST_CASE_TEMPLATE(central_differences_do_not_overflow, Scalar,
                              Scalars)
{
    constexpr Scalar Largest = std::numeric_limits<Scalar>::max();

    BOOST_TEST((centralDifference<hogpp::Horizontal_t, Scalar>(
                   -Largest, Largest)) == Largest);
    BOOST_TEST((centralDifference<hogpp::Vertical_t, Scalar>(
                   Largest, -Largest)) == -Largest);
}

// Integer images keep the rounding of std::midpoint, which rounds towards its
// first argument rather than towards zero.
BOOST_AUTO_TEST_CASE_TEMPLATE(
    central_differences_of_integer_images_match_midpoint, Value, Integers)
{
    using Precision = hogpp::PrecisionType_t<Eigen::Tensor<Value, 3>>;

    constexpr std::array<std::array<int, 2>, 4> Neighbors{
        {{0, 1}, {1, 0}, {3, 8}, {8, 3}}};

    for (const auto& [a, b] : Neighbors) {
        const auto lower = static_cast<Value>(a);
        const auto upper = static_cast<Value>(b);
        // The unary operators promote the neighbors, exactly like the sampler
        // does.
        const auto expected = static_cast<float>(std::midpoint(
            -static_cast<Precision>(lower), +static_cast<Precision>(upper)));

        BOOST_TEST_CONTEXT("lower neighbor " << a << ", upper neighbor " << b)
        {
            BOOST_TEST((centralDifference<hogpp::Horizontal_t, float>(
                           lower, upper)) == expected);
            BOOST_TEST((centralDifference<hogpp::Vertical_t, float>(
                           lower, upper)) == expected);
        }
    }
}
