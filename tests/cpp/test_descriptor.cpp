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

#include <cmath>
#include <limits>
#include <stdexcept>

#include <hogpp/integralhogdescriptor.hpp>
#include <hogpp/signedgradient.hpp>

#include <boost/mpl/list.hpp>
#include <boost/test/included/unit_test.hpp>

using Scalars = boost::mpl::list<float, double, long double>;

template<class Scalar>
auto makeImage()
{
    Eigen::TensorFixedSize<Scalar, Eigen::Sizes<3, 3, 1>> image;
    image.setValues({{{Scalar(0)}, {Scalar(1)}, {Scalar(2)}},
                     {{Scalar(3)}, {Scalar(4)}, {Scalar(5)}},
                     {{Scalar(6)}, {Scalar(7)}, {Scalar(8)}}});
    return image;
}

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
    auto image = makeImage<Scalar>();

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
    auto image = makeImage<Scalar>();

    hogpp::IntegralHOGDescriptor<Scalar> d;
    d.compute(image);

    const auto X = d.features(hogpp::Bounds{0, 0, 3, 3});
    BOOST_TEST(X.size() == 0);
}

// A single non-finite pixel (e.g. from masked/invalid input data) must
// not violate the internal "gradient magnitude is strictly positive at
// this point" invariant used while voting, since that invariant
// degrades to undefined behavior in release builds.
BOOST_AUTO_TEST_CASE_TEMPLATE(nan_pixel, Scalar, Scalars)
{
    auto image = makeImage<Scalar>();
    image(1, 1, 0) = std::numeric_limits<Scalar>::quiet_NaN();

    hogpp::IntegralHOGDescriptor<Scalar> d;
    d.compute(image);

    const Eigen::Tensor<Scalar, 3>& h = d.histogram();

    bool allFinite = true;

    for (Eigen::DenseIndex i = 0; i < h.size(); ++i) {
        using std::isfinite;

        if (!isfinite(h.data()[i])) {
            allFinite = false;
            break;
        }
    }

    BOOST_TEST(allFinite);
}

// SignedGradient<Scalar, Fast> (and UnsignedGradient) provide a
// tensor-expression operator() overload that IntegralHOGDescriptor::compute()
// uses to precompute the per-pixel bin weight for the whole image as a
// single vectorized pass ahead of the histogram scan (see
// detail::HasTensorBinning in integralhogdescriptor.hpp), instead of
// calling the approximation once per pixel inside the scan. This exercises
// that wiring end to end: the precomputed-weight result must agree with
// calling the exact reference implementation per pixel, within the same
// tolerance already established for the Fast binning approximation itself.
BOOST_AUTO_TEST_CASE_TEMPLATE(precomputed_binning_matches_reference, Scalar,
                              Scalars)
{
    namespace tt = boost::test_tools;

    Eigen::Tensor<Scalar, 3> image(16, 16, 1);

    for (Eigen::DenseIndex i = 0; i < image.dimension(0); ++i) {
        for (Eigen::DenseIndex j = 0; j < image.dimension(1); ++j) {
            image(i, j, 0) =
                static_cast<Scalar>((i * image.dimension(1) + j) % 7);
        }
    }

    hogpp::IntegralHOGDescriptor<Scalar, hogpp::Gradient<Scalar>,
                                 hogpp::GradientMagnitude<Scalar>,
                                 hogpp::SignedGradient<Scalar, hogpp::Accurate>>
        referenceDescriptor;
    hogpp::IntegralHOGDescriptor<Scalar, hogpp::Gradient<Scalar>,
                                 hogpp::GradientMagnitude<Scalar>,
                                 hogpp::SignedGradient<Scalar, hogpp::Fast>>
        precomputedDescriptor;

    referenceDescriptor.compute(image);
    precomputedDescriptor.compute(image);

    const auto referenceFeatures = referenceDescriptor.features();
    const auto precomputedFeatures = precomputedDescriptor.features();

    BOOST_TEST_REQUIRE(referenceFeatures.size() == precomputedFeatures.size());

    // boost::unit_test::tolerance() compares relative to the operands'
    // magnitude, which is unstable for feature values close to zero
    // (L2Hys-normalized entries are often small). Both operands are
    // shifted away from zero by a fixed offset first, exactly as in
    // tests/cpp/test_binning.cpp's *_fast tests, so the same relative
    // tolerance enforces a tight absolute bound instead.
    for (Eigen::DenseIndex i = 0; i < referenceFeatures.size(); ++i) {
        const Scalar precomputedShifted =
            precomputedFeatures.data()[i] + Scalar{1};
        const Scalar referenceShifted = referenceFeatures.data()[i] + Scalar{1};

        BOOST_TEST(precomputedShifted == referenceShifted,
                   tt::tolerance(Scalar(1e-4L)));
    }
}

// With a single channel, IntegralHOGDescriptor::compute() skips
// Eigen's argmax(2) reduction entirely (see channels > 1 in
// integralhogdescriptor.hpp) since the maximum over one element is
// trivially that element. This guards the channels > 1 path still
// genuinely selects the channel with the largest gradient magnitude,
// rather than e.g. always defaulting to channel 0: a multi-channel
// image whose first channel is constant (zero gradient everywhere,
// so it can never be the argmax) and whose second channel carries
// the same data as a single-channel reference image must produce the
// same descriptor as that reference image.
BOOST_AUTO_TEST_CASE_TEMPLATE(multi_channel_selects_max_magnitude_channel,
                              Scalar, Scalars)
{
    namespace tt = boost::test_tools;

    constexpr Eigen::DenseIndex rows = 16;
    constexpr Eigen::DenseIndex cols = 16;

    Eigen::Tensor<Scalar, 3> referenceImage(rows, cols, 1);
    Eigen::Tensor<Scalar, 3> multiChannelImage(rows, cols, 2);

    for (Eigen::DenseIndex i = 0; i < rows; ++i) {
        for (Eigen::DenseIndex j = 0; j < cols; ++j) {
            const auto value = static_cast<Scalar>((i * cols + j) % 7);
            referenceImage(i, j, 0) = value;
            multiChannelImage(i, j, 0) = Scalar{3}; // constant: zero gradient
            multiChannelImage(i, j, 1) = value;
        }
    }

    hogpp::IntegralHOGDescriptor<Scalar> referenceDescriptor;
    hogpp::IntegralHOGDescriptor<Scalar> multiChannelDescriptor;

    referenceDescriptor.compute(referenceImage);
    multiChannelDescriptor.compute(multiChannelImage);

    const auto referenceFeatures = referenceDescriptor.features();
    const auto multiChannelFeatures = multiChannelDescriptor.features();

    BOOST_TEST_REQUIRE(referenceFeatures.size() == multiChannelFeatures.size());

    for (Eigen::DenseIndex i = 0; i < referenceFeatures.size(); ++i) {
        const Scalar multiChannelShifted =
            multiChannelFeatures.data()[i] + Scalar{1};
        const Scalar referenceShifted = referenceFeatures.data()[i] + Scalar{1};

        BOOST_TEST(multiChannelShifted == referenceShifted,
                   tt::tolerance(Scalar(1e-4L)));
    }
}
