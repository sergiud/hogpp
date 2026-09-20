//
// HOGpp - Fast histogram of oriented gradients computation using integral
// histograms
//
// Copyright 2024 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

#include <cmath>
#include <utility>

#include <hogpp/constants.hpp>
#include <hogpp/signedgradient.hpp>
#include <hogpp/unsignedgradient.hpp>

#include <boost/mpl/list.hpp>
#include <boost/test/included/unit_test.hpp>

using Scalars = boost::mpl::list<float, double, long double>;

// clang-format off
BOOST_TEST_DECORATOR
(
    *boost::unit_test::tolerance(5.96048e-08f)
    *boost::unit_test::tolerance(3.90314e-17l)
)
// clang-format on
BOOST_AUTO_TEST_CASE_TEMPLATE(signed_gradient, Scalar, Scalars)
{
    using std::fpclassify;
    using std::nextafter;

    hogpp::SignedGradient<Scalar> binning;

    BOOST_TEST(binning(+1, 0) == Scalar(0.5));
    BOOST_TEST(binning(-1, 0) == Scalar{1});
    BOOST_TEST(binning(0, +1) == Scalar(0.75));
    BOOST_TEST(binning(0, -1) == Scalar(0.25));
    BOOST_TEST(binning(0, 0) == Scalar(0.5));
    BOOST_TEST(binning(+1, nextafter(Scalar{0}, Scalar{-1})) == Scalar(0.5));
    BOOST_TEST(binning(-1, nextafter(Scalar{0}, Scalar{+1})) == Scalar{1});
}

// clang-format off
BOOST_TEST_DECORATOR
(
    *boost::unit_test::tolerance(5.96048e-08f)
    *boost::unit_test::tolerance(3.90314e-17l)
)
// clang-format on
BOOST_AUTO_TEST_CASE_TEMPLATE(unsigned_gradient, Scalar, Scalars)
{
    using std::fpclassify;
    using std::nextafter;

    hogpp::UnsignedGradient<Scalar> binning;

    BOOST_TEST(binning(+1, 0) == Scalar(0.5));
    BOOST_TEST(binning(-1, 0) == Scalar(0.5));
    BOOST_TEST(binning(0, +1) == Scalar{1});
    BOOST_TEST(fpclassify(binning(0, -1)) == FP_ZERO);
    BOOST_TEST(binning(0, 0) == Scalar(0.5));
    BOOST_TEST(binning(+1, nextafter(Scalar{0}, Scalar{-1})) == Scalar(0.5));
    BOOST_TEST(binning(-1, nextafter(Scalar{0}, Scalar{+1})) == Scalar(0.5));
}

// The Fast binning profile replaces the exact (but branchy,
// non-vectorizable) libm atan2/atan call with a branch-light polynomial
// approximation intended for ISA-dispatched builds. Its maximum angular
// error is approximately 1.3312e-4 rad (the minimax bound of
// hogpp::detail::fastAtanUnit()), verified numerically against
// std::atan2/std::atan across the full angular range. After mapping to
// the normalized [0, 1) bin weight this is at most ~2.13e-5 (signed) /
// ~4.24e-5 (unsigned), more than three orders of magnitude below the
// width of a single bin in the default 9-bin histogram (1/9 ~ 0.111), so
// it cannot change bin assignment.
//
// boost::unit_test::tolerance() compares relative to the operands'
// magnitude, which is unstable for weight values close to zero. Both
// operands are shifted away from zero by a fixed offset before the
// comparison so the same relative tolerance enforces the tight,
// evidence-based absolute bound above regardless of where in [0, 1) the
// weight falls (shifted operands lie in [1, 2), so the strong relative
// check reduces to an absolute one).
template<class Scalar>
inline constexpr Scalar signedFastBinningTolerance = Scalar(2.5e-5L);

template<class Scalar>
inline constexpr Scalar unsignedFastBinningTolerance = Scalar(5e-5L);

BOOST_AUTO_TEST_CASE_TEMPLATE(signed_gradient_fast, Scalar, Scalars)
{
    namespace tt = boost::test_tools;

    using std::cos;
    using std::sin;

    hogpp::SignedGradient<Scalar, hogpp::Accurate> exact;
    hogpp::SignedGradient<Scalar, hogpp::Fast> approx;

    const auto shifted = [&](Scalar dx, Scalar dy) {
        return std::pair{approx(dx, dy) + Scalar{1}, exact(dx, dy) + Scalar{1}};
    };

    constexpr int samples = 3601;

    for (int i = 0; i < samples; ++i) {
        const auto theta = (static_cast<Scalar>(i) /
                             static_cast<Scalar>(samples - 1) *
                             hogpp::constants::two_pi<Scalar>) -
                            hogpp::constants::pi<Scalar>;
        const Scalar dx = cos(theta);
        const Scalar dy = sin(theta);
        const auto [approxWeight, exactWeight] = shifted(dx, dy);

        BOOST_TEST(approxWeight == exactWeight,
                  tt::tolerance(signedFastBinningTolerance<Scalar>));
    }

    for (auto [dx, dy] : {std::pair<Scalar, Scalar>{+1, 0},
                          std::pair<Scalar, Scalar>{-1, 0},
                          std::pair<Scalar, Scalar>{0, +1},
                          std::pair<Scalar, Scalar>{0, -1},
                          std::pair<Scalar, Scalar>{0, 0}}) {
        const auto [approxWeight, exactWeight] = shifted(dx, dy);

        BOOST_TEST(approxWeight == exactWeight,
                  tt::tolerance(signedFastBinningTolerance<Scalar>));
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(unsigned_gradient_fast, Scalar, Scalars)
{
    namespace tt = boost::test_tools;

    using std::cos;
    using std::sin;

    hogpp::UnsignedGradient<Scalar, hogpp::Accurate> exact;
    hogpp::UnsignedGradient<Scalar, hogpp::Fast> approx;

    const auto shifted = [&](Scalar dx, Scalar dy) {
        return std::pair{approx(dx, dy) + Scalar{1}, exact(dx, dy) + Scalar{1}};
    };

    constexpr int samples = 3601;

    for (int i = 0; i < samples; ++i) {
        const auto theta = (static_cast<Scalar>(i) /
                             static_cast<Scalar>(samples - 1) *
                             hogpp::constants::two_pi<Scalar>) -
                            hogpp::constants::pi<Scalar>;
        const Scalar dx = cos(theta);
        const Scalar dy = sin(theta);
        const auto [approxWeight, exactWeight] = shifted(dx, dy);

        BOOST_TEST(approxWeight == exactWeight,
                  tt::tolerance(unsignedFastBinningTolerance<Scalar>));
    }

    for (auto [dx, dy] : {std::pair<Scalar, Scalar>{+1, 0},
                          std::pair<Scalar, Scalar>{-1, 0},
                          std::pair<Scalar, Scalar>{0, +1},
                          std::pair<Scalar, Scalar>{0, -1},
                          std::pair<Scalar, Scalar>{0, 0}}) {
        const auto [approxWeight, exactWeight] = shifted(dx, dy);

        BOOST_TEST(approxWeight == exactWeight,
                  tt::tolerance(unsignedFastBinningTolerance<Scalar>));
    }
}

// The tensor-expression overloads of SignedGradient<Scalar, Fast> and
// UnsignedGradient<Scalar, Fast> re-express the same formula as lazy,
// elementwise Eigen operations so IntegralHOGDescriptor::compute() can
// precompute the bin weight for a whole image in one vectorized pass
// instead of once per pixel. They must agree with the scalar overload at
// every element: this is not a second, independent approximation, just a
// batched evaluation of the identical polynomial.
template<class Scalar>
inline constexpr Scalar tensorBinningAgreementTolerance = Scalar(1e-6L);

BOOST_AUTO_TEST_CASE_TEMPLATE(signed_gradient_fast_tensor, Scalar, Scalars)
{
    namespace tt = boost::test_tools;

    using std::cos;
    using std::sin;

    hogpp::SignedGradient<Scalar, hogpp::Fast> approx;

    constexpr int circleSamples = 3601;
    constexpr int samples = circleSamples + 1;
    Eigen::Tensor<Scalar, 1> dx(samples);
    Eigen::Tensor<Scalar, 1> dy(samples);

    for (int i = 0; i < circleSamples; ++i) {
        const auto theta = (static_cast<Scalar>(i) /
                             static_cast<Scalar>(circleSamples - 1) *
                             hogpp::constants::two_pi<Scalar>) -
                            hogpp::constants::pi<Scalar>;
        dx(i) = cos(theta);
        dy(i) = sin(theta);
    }
    // dx == dy == 0 never occurs on the circle sweep above, but is an
    // important degenerate case handled explicitly by the formula.
    dx(circleSamples) = Scalar{0};
    dy(circleSamples) = Scalar{0};

    const Eigen::Tensor<Scalar, 1> batched = approx(dx, dy);

    for (int i = 0; i < samples; ++i) {
        BOOST_TEST(batched(i) == approx(dx(i), dy(i)),
                  tt::tolerance(tensorBinningAgreementTolerance<Scalar>));
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(unsigned_gradient_fast_tensor, Scalar, Scalars)
{
    namespace tt = boost::test_tools;

    using std::cos;
    using std::sin;

    hogpp::UnsignedGradient<Scalar, hogpp::Fast> approx;

    constexpr int circleSamples = 3601;
    constexpr int samples = circleSamples + 1;
    Eigen::Tensor<Scalar, 1> dx(samples);
    Eigen::Tensor<Scalar, 1> dy(samples);

    for (int i = 0; i < circleSamples; ++i) {
        const auto theta = (static_cast<Scalar>(i) /
                             static_cast<Scalar>(circleSamples - 1) *
                             hogpp::constants::two_pi<Scalar>) -
                            hogpp::constants::pi<Scalar>;
        dx(i) = cos(theta);
        dy(i) = sin(theta);
    }
    // dx == dy == 0 never occurs on the circle sweep above, but is an
    // important degenerate case handled explicitly by the formula.
    dx(circleSamples) = Scalar{0};
    dy(circleSamples) = Scalar{0};

    const Eigen::Tensor<Scalar, 1> batched = approx(dx, dy);

    for (int i = 0; i < samples; ++i) {
        BOOST_TEST(batched(i) == approx(dx(i), dy(i)),
                  tt::tolerance(tensorBinningAgreementTolerance<Scalar>));
    }
}
