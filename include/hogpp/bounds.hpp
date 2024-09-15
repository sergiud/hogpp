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

#ifndef HOGPP_BOUNDS_HPP
#define HOGPP_BOUNDS_HPP

namespace hogpp {

struct Bounds
{
    struct Size
    {
        int width;
        int height;
    };

    int x;
    int y;
    int width;
    int height;

    [[nodiscard]] constexpr int area() const noexcept
    {
        return width * height;
    }

    [[nodiscard]] constexpr Size size() const noexcept
    {
        return {width, height};
    }
};

[[nodiscard]] constexpr bool operator==(const Bounds::Size& lhs,
                                        const Bounds::Size& rhs) noexcept
{
    return lhs.width == rhs.width && lhs.height == rhs.height;
}

} // namespace hogpp

#endif // HOGPP_BOUNDS_HPP
