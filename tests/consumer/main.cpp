// SPDX-FileCopyrightText: 2026 Sergiu Deitsch
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <iostream>

#include <unsupported/Eigen/CXX11/Tensor>

#include <hogpp/integralhogdescriptor.hpp>

namespace {

// The image covers a single block of the default descriptor, whose cells are
// 8 by 8 pixels and whose blocks consist of 2 by 2 cells.
constexpr Eigen::Index ImageRows = 16;
constexpr Eigen::Index ImageCols = 16;
constexpr Eigen::Index NumChannels = 1;

} // namespace

int main()
{
    Eigen::Tensor<float, 3> image{ImageRows, ImageCols, NumChannels};
    image.setRandom();

    hogpp::IntegralHOGDescriptor<float> descriptor;
    descriptor.compute(image);

    const auto features = descriptor.features();

    if (features.size() == 0) {
        std::cerr << "expected features of a " << ImageRows << "x" << ImageCols
                  << " image, but got none\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
