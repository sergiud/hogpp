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

#ifndef PYTHON_LEVENSHTEIN_HPP
#define PYTHON_LEVENSHTEIN_HPP

#include <algorithm>
#include <cctype>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <locale>
#include <numeric>
#include <optional>
#include <ranges>
#include <string_view>
#include <vector>

namespace pyhogpp {

template<std::ranges::random_access_range R1,
         std::ranges::random_access_range R2,
         class Pred = std::ranges::equal_to, class Proj1 = std::identity,
         class Proj2 = std::identity>
    requires std::indirectly_comparable<std::ranges::iterator_t<R1>,
                                        std::ranges::iterator_t<R2>, Pred,
                                        Proj1, Proj2>
[[nodiscard]] inline std::size_t levenshteinDistance(const R1& a, const R2& b,
                                                     Pred pred = {},
                                                     Proj1 proj1 = {},
                                                     Proj2 proj2 = {})
{
    const auto m = std::ranges::size(a);
    const auto n = std::ranges::size(b);

    std::vector<std::size_t> v0(n + 1);
    std::vector<std::size_t> v1(n + 1);

    std::iota(v0.begin(), v0.end(), 0);

    for (std::size_t i = 0; i != m; ++i) {
        v1[0] = i + 1;

        for (std::size_t j = 0; j != n; ++j) {
            const auto deletion = v0[j + 1] + 1;
            const auto insertion = v1[j] + 1;
            auto substitution = v0[j];

            if (!std::invoke(pred, std::invoke(proj1, a[i]),
                             std::invoke(proj2, b[j]))) {
                ++substitution;
            }

            v1[j + 1] = std::min({deletion, insertion, substitution});
        }

        swap(v0, v1);
    }

    return v0.back();
}

// Only suggest a match that is close enough to plausibly be a typo. Without
// this bound the closest of an unrelated set of names would always be
// suggested, however dissimilar it is to the requested one.
template<std::ranges::random_access_range R>
    requires std::convertible_to<typename R::value_type, std::string_view>
[[nodiscard]] inline std::optional<std::string_view> findClosestMatch(
    std::string_view isa, const R& supported, const std::locale& loc)
{
    const auto proj = [&loc](char ch) { return std::tolower(ch, loc); };

    std::string_view best;
    std::size_t bestDistance = std::numeric_limits<std::size_t>::max();

    for (std::string_view candidate : supported) {
        const auto distance =
            levenshteinDistance(isa, candidate, {}, proj, proj);

        if (distance < bestDistance) {
            bestDistance = distance;
            best = candidate;
        }
    }

    const auto threshold = std::max<std::size_t>(2, isa.size() / 2);

    if (bestDistance <= threshold) {
        return best;
    }

    return std::nullopt;
}

} // namespace pyhogpp

#endif // PYTHON_LEVENSHTEIN_HPP
