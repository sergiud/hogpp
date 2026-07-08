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
    Eigen::Tensor<Scalar, 3> image(1, 6, 1);
    image.setZero();

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
}
