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

#include <tuple>
#include <type_traits>

#include <hogpp/cartesianproduct.hpp>

#include <boost/mp11/list.hpp>
#include <boost/test/included/unit_test.hpp>

// clang-format off
using IntegerTypes = boost::mp11::mp_list
<
      signed char
    , unsigned char
    , short
    , unsigned short
    , int
    , unsigned int
    , long
    , unsigned long
    , long long
    , unsigned long long
>;
// clang-format on

// The traversal must carry the caller's own index type through every
// loop, not narrow it to int, since a tensor dimension can exceed what
// int can represent. The property is purely a compile-time fact about
// the deduced type, so it is checked with static_assert rather than a
// runtime comparison.
BOOST_AUTO_TEST_CASE_TEMPLATE(loop_counter_type, T, IntegerTypes)
{
    hogpp::cartesianProduct(std::make_tuple(T{2}, T{2}), [](const auto& i) {
        static_assert(
            std::is_same_v<std::decay_t<decltype(std::get<0>(i))>, T>,
            "cartesianProduct must not narrow the loop counter's type");
        static_assert(
            std::is_same_v<std::decay_t<decltype(std::get<1>(i))>, T>,
            "cartesianProduct must not narrow the loop counter's type");
    });
}
