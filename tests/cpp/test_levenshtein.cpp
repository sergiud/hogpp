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

#include <array>
#include <locale>
#include <string_view>

#include "../../python/levenshtein.hpp"

#include <boost/test/included/unit_test.hpp>

using pyhogpp::findClosestMatch;
using pyhogpp::levenshteinDistance;

BOOST_AUTO_TEST_CASE(identical_strings)
{
    using namespace std::string_view_literals;

    BOOST_TEST(levenshteinDistance("AVX2"sv, "AVX2"sv) == 0U);
    BOOST_TEST(levenshteinDistance(""sv, ""sv) == 0U);
}

BOOST_AUTO_TEST_CASE(single_edits)
{
    using namespace std::string_view_literals;

    BOOST_TEST(levenshteinDistance("AVX2"sv,
                                   "AVX"sv) == 1U); // deletion
    BOOST_TEST(levenshteinDistance("AVX"sv,
                                   "AVX2"sv) == 1U); // insertion
    BOOST_TEST(levenshteinDistance("AVX2"sv,
                                   "AVXZ"sv) == 1U); // substitution
}

// The first DP row must be fully initialized up to and including its last
// element, particularly when the second string is longer than the first.
BOOST_AUTO_TEST_CASE(empty_versus_non_empty_regression)
{
    using namespace std::string_view_literals;

    BOOST_TEST(levenshteinDistance(""sv, "a"sv) == 1U);
    BOOST_TEST(levenshteinDistance(""sv, "bb"sv) == 2U);
    BOOST_TEST(levenshteinDistance("a"sv, "cc"sv) == 2U);
}

BOOST_AUTO_TEST_CASE(distance_is_symmetric)
{
    using namespace std::string_view_literals;

    BOOST_TEST(levenshteinDistance("kitten"sv, "sitting"sv) == 3U);
    BOOST_TEST(levenshteinDistance("sitting"sv, "kitten"sv) == 3U);
}

BOOST_AUTO_TEST_CASE(closest_match_suggests_plausible_typo)
{
    using namespace std::string_view_literals;

    const std::array supported{"SSE2"sv, "AVX2"sv, "AVX512"sv};

    const auto match =
        findClosestMatch("AVX"sv, supported, std::locale::classic());

    BOOST_TEST_REQUIRE(match.has_value());
    BOOST_TEST(*match == "AVX2");
}

// No suggestion should be returned when every candidate is too dissimilar
// from the requested name to plausibly be a typo.
BOOST_AUTO_TEST_CASE(closest_match_rejects_unrelated_name)
{
    using namespace std::string_view_literals;

    constexpr std::array supported{"SSE2"sv, "AVX2"sv, "AVX512"sv};

    const auto match = findClosestMatch("zzzzzzzzzzzzzzzzzzzz", supported,
                                        std::locale::classic());

    BOOST_TEST(!match.has_value());
}

BOOST_AUTO_TEST_CASE(closest_match_empty_supported_list)
{
    using namespace std::string_view_literals;

    constexpr std::array<std::string_view, 0> supported{};

    const auto match =
        findClosestMatch("AVX2"sv, supported, std::locale::classic());

    BOOST_TEST(!match.has_value());
}
