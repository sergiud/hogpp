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

#ifndef HOGPP_REPEATBORDER_HPP
#define HOGPP_REPEATBORDER_HPP

#include <unsupported/Eigen/CXX11/Tensor>

#include <algorithm>
#include <tuple>

namespace hogpp {

struct RepeatBorder
{
    template<std::size_t Axis, class Scalar, int DataLayout, class... Elements>
    [[nodiscard]] constexpr Scalar clamp(
        const Eigen::Tensor<Scalar, sizeof...(Elements), DataLayout>& image,
        std::tuple<Elements...> idxs) const
    {
        using std::clamp;
        using Index = std::tuple_element_t<Axis, std::tuple<Elements...>>;

        auto& i = std::get<Axis>(idxs);

        i = clamp(i, Index{0}, static_cast<Index>(image.dimension(Axis)) - 1);

        return std::apply(image, idxs);
    }

    template<std::size_t Axis, class Scalar, int DataLayout, class... Args>
    [[nodiscard]] constexpr auto compute(
        const Eigen::Tensor<Scalar, sizeof...(Args), DataLayout>& image,
        Args... args) const
    {
        return clamp<Axis>(image, std::make_tuple(args...));
    }
};

} // namespace hogpp

#endif // HOGPP_REPEATBORDER_HPP
