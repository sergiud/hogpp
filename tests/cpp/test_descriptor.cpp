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
