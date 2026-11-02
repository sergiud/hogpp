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

#include <optional>
#include <utility>

#include <Eigen/Core>

#include "../../python/maskingref.hpp"

#include <boost/test/included/unit_test.hpp>

// Constructing a MaskingRef from another one must copy the reference to the
// callable rather than refer to the other MaskingRef, which may not outlive
// the copy. Replacing the original in place reveals what the copy refers to.
BOOST_AUTO_TEST_CASE(copy_refers_to_callable)
{
    auto masked = [](Eigen::DenseIndex /*i*/, Eigen::DenseIndex /*j*/) {
        return true;
    };
    auto unmasked = [](Eigen::DenseIndex /*i*/, Eigen::DenseIndex /*j*/) {
        return false;
    };

    std::optional<pyhogpp::MaskingRef> original{std::in_place, masked};
    pyhogpp::MaskingRef copy{*original};
    original.emplace(unmasked);

    BOOST_TEST(copy(0, 0));
}
