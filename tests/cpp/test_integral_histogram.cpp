//
// HOGpp - Fast histogram of oriented gradients computation using integral
// histograms
//
// Copyright 2021 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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
