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

#include <boost/mpl/list.hpp>
#include <boost/test/included/unit_test.hpp>

#include <hogpp/multiplyadd.hpp>

using Scalars = boost::mpl::list<float, double, long double>;

static_assert(hogpp::HasFastFma<float> ==
#if defined(FP_FAST_FMAF)
                  true,
#else  // !defined(FP_FAST_FMAF)
                  false,
#endif // defined(FP_FAST_FMAF)
              "HasFastFma<float> must follow FP_FAST_FMAF");

static_assert(hogpp::HasFastFma<double> ==
#if defined(FP_FAST_FMA)
                  true,
#else  // !defined(FP_FAST_FMA)
                  false,
#endif // defined(FP_FAST_FMA)
              "HasFastFma<double> must follow FP_FAST_FMA");

static_assert(hogpp::HasFastFma<long double> ==
#if defined(FP_FAST_FMAL)
                  true,
#else  // !defined(FP_FAST_FMAL)
                  false,
#endif // defined(FP_FAST_FMAL)
              "HasFastFma<long double> must follow FP_FAST_FMAL");

// The product (1 + eps)(1 - eps) = 1 - eps^2 rounds to 1, so only the fused
// operation retains -eps^2 after adding -1.
BOOST_AUTO_TEST_CASE_TEMPLATE(fused_only_if_fast, T, Scalars)
{
    constexpr T eps = std::numeric_limits<T>::epsilon();
    const T a = T(1) + eps;
    const T b = T(1) - eps;
    const T c = T(-1);

    const T expected = hogpp::HasFastFma<T> ? -(eps * eps) : T(0);

    BOOST_TEST(hogpp::multiplyAdd(a, b, c) == expected);
}
