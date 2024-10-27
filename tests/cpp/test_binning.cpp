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
