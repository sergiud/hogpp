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

#include <stdexcept>

#include <hogpp/integralhogdescriptor.hpp>

#include <boost/mpl/list.hpp>
#include <boost/test/included/unit_test.hpp>

using Scalars = boost::mpl::list<float, double, long double>;

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

// A block size far larger than the region, combined with a small
// stride, drives numBlocks negative rather than just to zero, and
// must be rejected rather than passed on to a Tensor resize with a
// negative dimension.
BOOST_AUTO_TEST_CASE_TEMPLATE(features_block_larger_than_region, Scalar,
                              Scalars)
{
    Eigen::Tensor<Scalar, 3> image(3, 3, 1);
    image.setValues({{{Scalar(0)}, {Scalar(1)}, {Scalar(2)}},
                     {{Scalar(3)}, {Scalar(4)}, {Scalar(5)}},
                     {{Scalar(6)}, {Scalar(7)}, {Scalar(8)}}});

    hogpp::IntegralHOGDescriptor<Scalar> d;
    d.setBlockSize(Eigen::Array2i{100, 100});
    d.setBlockStride(Eigen::Array2i{1, 1});
    d.compute(image);

    BOOST_CHECK_THROW((void)d.features(hogpp::Bounds{0, 0, 3, 3}),
                      std::invalid_argument);
}

// A region smaller than the (default) block size is valid on its own:
// it simply yields no blocks, and must not be rejected.
BOOST_AUTO_TEST_CASE_TEMPLATE(features_block_larger_than_region_but_valid,
                              Scalar, Scalars)
{
    Eigen::Tensor<Scalar, 3> image(3, 3, 1);
    image.setValues({{{Scalar(0)}, {Scalar(1)}, {Scalar(2)}},
                     {{Scalar(3)}, {Scalar(4)}, {Scalar(5)}},
                     {{Scalar(6)}, {Scalar(7)}, {Scalar(8)}}});

    hogpp::IntegralHOGDescriptor<Scalar> d;
    d.compute(image);

    const auto X = d.features(hogpp::Bounds{0, 0, 3, 3});
    BOOST_TEST(X.size() == 0);
}
