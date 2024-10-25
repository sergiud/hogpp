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

#include <hogpp/l1hys.hpp>
#include <hogpp/l1norm.hpp>
#include <hogpp/l1sqrt.hpp>
#include <hogpp/l2hys.hpp>
#include <hogpp/l2norm.hpp>

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>
#include <boost/test/included/unit_test.hpp>

using Scalars = boost::mp11::mp_list<float, double, long double>;

// clang-format off
using Norms = boost::mp11::mp_list
<
      boost::mp11::mp_quote<hogpp::L1Hys>
    , boost::mp11::mp_quote<hogpp::L1Norm>
    , boost::mp11::mp_quote<hogpp::L1Sqrt>
    , boost::mp11::mp_quote<hogpp::L2Hys>
    , boost::mp11::mp_quote<hogpp::L2Norm>
>;
// clang-format on

using PrecisionNorms =
    boost::mp11::mp_product<boost::mp11::mp_invoke_q, Norms, Scalars>;

BOOST_TEST_DECORATOR(*boost::unit_test::label("zero"))
BOOST_AUTO_TEST_CASE_TEMPLATE(zero, Norm, PrecisionNorms)
{
    using Scalar = typename Norm::Scalar;

    Eigen::TensorFixedSize<Scalar, Eigen::Sizes<8, 8>> block;
    block.setZero();

    Norm{}(block);

    const Eigen::Tensor<Scalar, 0> s = block.sum();
    BOOST_TEST(s(0) == 0);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("negative_near_zero"))
BOOST_AUTO_TEST_CASE_TEMPLATE(negative_near_zero, Norm, PrecisionNorms)
{
    using Scalar = typename Norm::Scalar;

    Eigen::TensorFixedSize<Scalar, Eigen::Sizes<8, 8>> block;
    block.setConstant(-Eigen::NumTraits<Scalar>::dummy_precision());

    Norm{}(block);

    const Eigen::Tensor<Scalar, 0> s = block.abs().sum();
    BOOST_TEST(s(0) >= 0);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("positive_near_zero"))
BOOST_AUTO_TEST_CASE_TEMPLATE(positive_near_zero, Norm, PrecisionNorms)
{
    using Scalar = typename Norm::Scalar;

    Eigen::TensorFixedSize<Scalar, Eigen::Sizes<8, 8>> block;
    block.setConstant(+Eigen::NumTraits<Scalar>::dummy_precision());

    Norm{}(block);

    const Eigen::Tensor<Scalar, 0> s = block.abs().sum();
    BOOST_TEST(s(0) >= 0);
}
