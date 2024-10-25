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

#ifndef HOGPP_CONDITIONALBORDER_HPP
#define HOGPP_CONDITIONALBORDER_HPP

#include <unsupported/Eigen/CXX11/Tensor>

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace hogpp {

struct ConditionalBorder
{
    template<class... Elements>
    struct Size_t : std::integral_constant<Eigen::DenseIndex,
                                           static_cast<Eigen::DenseIndex>(
                                               sizeof...(Elements))>
    {
    };

    template<class... Elements>
    static constexpr auto Size_v = Size_t<Elements...>::value;

    template<class Axis, class Scalar, int DataLayout, class Interior,
             class AtLowerBorder, class AtUpperBorder, class... Elements>
    constexpr decltype(auto) select(
        const Eigen::Tensor<Scalar, Size_v<Elements...>, DataLayout>& image,
        Interior&& interior, AtLowerBorder&& atLowerBorder,
        AtUpperBorder&& atUpperBorder, std::tuple<Elements...> idxs) const
    {
        using Index =
            std::tuple_element_t<Axis::value, std::tuple<Elements...>>;

        auto& i = std::get<Axis::value>(idxs);
        auto&& args = std::tuple_cat(std::forward_as_tuple(image), idxs);

        if (i <= 0) {
            return std::apply(atLowerBorder, args);
        }

        if (i >= static_cast<Index>(image.dimension(Axis::value)) - 1) {
            return std::apply(atUpperBorder, args);
        }

        return std::apply(interior, args);
    }

    template<class Axis, class Scalar, int DataLayout, class Interior,
             class AtLowerBorder, class AtUpperBorder, class... Args>
    constexpr decltype(auto) compute(
        const Eigen::Tensor<Scalar, Size_v<Args...>, DataLayout>& image,
        Interior&& interior, AtLowerBorder&& atLowerBorder,
        AtUpperBorder&& atUpperBorder, Args&&... args) const
    {
        return select<Axis>(image, std::forward<Interior>(interior),
                            std::forward<AtLowerBorder>(atLowerBorder),
                            std::forward<AtUpperBorder>(atUpperBorder),
                            std::make_tuple(args...));
    }
};

} // namespace hogpp

#endif // HOGPP_CONDITIONALBORDER_HPP
