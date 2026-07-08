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

#ifndef HOGPP_GRADIENT_HPP
#define HOGPP_GRADIENT_HPP

#include <numeric>
#include <utility>

#include <unsupported/Eigen/CXX11/Tensor>

#include <hogpp/axis.hpp>
#include <hogpp/precision.hpp>

namespace hogpp {

template<class T, class Axis = NoAxis_t>
struct ForwardDifferences;

template<class T>
struct ForwardDifferences<T, Vertical_t>
{
    using Scalar = T;

    template<class Tensor, class PrecisionType = PrecisionType_t<Tensor>>
    [[nodiscard]] constexpr decltype(auto) operator()(const Tensor& image,
                                                      Eigen::DenseIndex i,
                                                      Eigen::DenseIndex j,
                                                      Eigen::DenseIndex k) const
    {
        return Scalar(PrecisionType(image(i + 1, j, k)) -
                      PrecisionType(image(i, j, k)));
    }
};

template<class T>
struct ForwardDifferences<T, Horizontal_t>
{
    using Scalar = T;

    template<class Tensor, class PrecisionType = PrecisionType_t<Tensor>>
    [[nodiscard]] constexpr decltype(auto) operator()(const Tensor& image,
                                                      Eigen::DenseIndex i,
                                                      Eigen::DenseIndex j,
                                                      Eigen::DenseIndex k) const
    {
        return Scalar(PrecisionType(image(i, j + 1, k)) -
                      PrecisionType(image(i, j, k)));
    }
};

template<class T, class Axis = NoAxis_t>
struct BackwardDifferences;

template<class T>
struct BackwardDifferences<T, Vertical_t>
{
    using Scalar = T;

    template<class Tensor, class PrecisionType = PrecisionType_t<Tensor>>
    [[nodiscard]] constexpr decltype(auto) operator()(const Tensor& image,
                                                      Eigen::DenseIndex i,
                                                      Eigen::DenseIndex j,
                                                      Eigen::DenseIndex k) const
    {
        return Scalar(PrecisionType(image(i, j, k)) -
                      PrecisionType(image(i - 1, j, k)));
    }
};

template<class T>
struct BackwardDifferences<T, Horizontal_t>
{
    using Scalar = T;

    template<class Tensor, class PrecisionType = PrecisionType_t<Tensor>>
    [[nodiscard]] constexpr decltype(auto) operator()(const Tensor& image,
                                                      Eigen::DenseIndex i,
                                                      Eigen::DenseIndex j,
                                                      Eigen::DenseIndex k) const
    {
        return Scalar(PrecisionType(image(i, j, k)) -
                      PrecisionType(image(i, j - 1, k)));
    }
};

template<class T, class Axis = NoAxis_t>
struct CentralDifferences;

template<class T>
struct CentralDifferences<T, Vertical_t>
{
    using Scalar = T;

    template<class Tensor, class PrecisionType = PrecisionType_t<Tensor>>
    [[nodiscard]] constexpr decltype(auto) operator()(const Tensor& image,
                                                      Eigen::DenseIndex i,
                                                      Eigen::DenseIndex j,
                                                      Eigen::DenseIndex k) const
    {
        using std::midpoint;
        return Scalar(midpoint(-PrecisionType(image(i - 1, j, k)),
                               +PrecisionType(image(i + 1, j, k))));
    }
};

template<class T>
struct CentralDifferences<T, Horizontal_t>
{
    using Scalar = T;

    template<class Tensor, class PrecisionType = PrecisionType_t<Tensor>>
    [[nodiscard]] constexpr decltype(auto) operator()(const Tensor& image,
                                                      Eigen::DenseIndex i,
                                                      Eigen::DenseIndex j,
                                                      Eigen::DenseIndex k) const
    {
        using std::midpoint;
        // Compute the dot product beween the kernel [-1 0 +1] and
        // the corresponding row (neighbor) pixels.
        return Scalar(midpoint(-PrecisionType(image(i, j - 1, k)),
                               +PrecisionType(image(i, j + 1, k))));
    }
};

template<class T, class Axis = NoAxis_t>
struct DiscretePointDifferences;

template<class T>
struct DiscretePointDifferences<T, Vertical_t>
{
    using Scalar = T;

    template<class Tensor, class PrecisionType = PrecisionType_t<Tensor>>
    [[nodiscard]] constexpr decltype(auto) operator()(const Tensor& image,
                                                      Eigen::DenseIndex i,
                                                      Eigen::DenseIndex j,
                                                      Eigen::DenseIndex k) const
    {
        return Scalar(PrecisionType(image(i + 1, j, k)) -
                      PrecisionType(image(i - 1, j, k)));
    }
};

template<class T>
struct DiscretePointDifferences<T, Horizontal_t>
{
    using Scalar = T;

    template<class Tensor, class PrecisionType = PrecisionType_t<Tensor>>
    [[nodiscard]] constexpr decltype(auto) operator()(const Tensor& image,
                                                      Eigen::DenseIndex i,
                                                      Eigen::DenseIndex j,
                                                      Eigen::DenseIndex k) const
    {
        return Scalar(PrecisionType(image(i, j + 1, k)) -
                      PrecisionType(image(i, j - 1, k)));
    }
};

template<class T, class Axis, class E = void>
struct MakeSampler
{
    using type = T;
};

template<class T, class Axis0, class Axis>
struct MakeSampler<CentralDifferences<T, Axis0>, Axis>
{
    using type = CentralDifferences<T, Axis>;
};

template<class T, class Axis0, class Axis>
struct MakeSampler<DiscretePointDifferences<T, Axis0>, Axis>
{
    using type = DiscretePointDifferences<T, Axis>;
};

template<class T, class Axis0, class Axis>
struct MakeSampler<ForwardDifferences<T, Axis0>, Axis>
{
    using type = ForwardDifferences<T, Axis>;
};

template<class T, class Axis0, class Axis>
struct MakeSampler<BackwardDifferences<T, Axis0>, Axis>
{
    using type = BackwardDifferences<T, Axis>;
};

template<class T>
using MakeVerticalSampler_t = typename MakeSampler<T, Vertical_t>::type;

template<class T>
using MakeHorizontalSampler_t = typename MakeSampler<T, Horizontal_t>::type;

template<class T, class InteriorSampler = CentralDifferences<T>,
         class LowerBoundSampler = ForwardDifferences<T>,
         class UpperBoundSampler = BackwardDifferences<T>>
class Gradient
{
public:
    using Scalar = T;

    template<class U, int DataLayout>
    [[nodiscard]] constexpr decltype(auto) operator()(
        const Eigen::Tensor<U, 3, DataLayout>& image) const
    {
        Eigen::Tensor<Scalar, 3, DataLayout> dxs;
        Eigen::Tensor<Scalar, 3, DataLayout> dys;

        dxs.resize(image.dimensions());
        dys.resize(image.dimensions());

        // Each derivative is swept as a lower-border, an interior, and an
        // upper-border pass along its own axis, so every call site resolves to
        // a single sampler at compile time instead of dispatching on a runtime
        // border check for every pixel.
        sweep<Horizontal_t>(image, dxs, interiorX_, atLowerBorderX_,
                            atUpperBorderX_);
        sweep<Vertical_t>(image, dys, interiorY_, atLowerBorderY_,
                          atUpperBorderY_);

        return std::make_tuple(dxs, dys);
    }

private:
    using InteriorSamplerX = MakeHorizontalSampler_t<InteriorSampler>;
    using InteriorSamplerY = MakeVerticalSampler_t<InteriorSampler>;
    using LowerBoundSamplerX = MakeHorizontalSampler_t<LowerBoundSampler>;
    using LowerBoundSamplerY = MakeVerticalSampler_t<LowerBoundSampler>;
    using UpperBoundSamplerX = MakeHorizontalSampler_t<UpperBoundSampler>;
    using UpperBoundSamplerY = MakeVerticalSampler_t<UpperBoundSampler>;

    // Sweeps the derivative along Axis (the dimension being differentiated),
    // holding the other dimension fixed as the outer loop. coord() maps the
    // (primary, secondary) pair used by the sweep to the (i, j) order
    // Eigen::Tensor::operator() expects, resolved entirely at compile time via
    // Axis::value.
    template<class Axis, class U, int DataLayout, class Interior,
             class LowerBound, class UpperBound>
    constexpr void sweep(const Eigen::Tensor<U, 3, DataLayout>& image,
                         Eigen::Tensor<Scalar, 3, DataLayout>& out,
                         const Interior& interior, const LowerBound& lower,
                         const UpperBound& upper) const
    {
        constexpr Eigen::DenseIndex Secondary = 1 - Axis::value;

        const Eigen::DenseIndex n = image.dimension(Axis::value);
        const Eigen::DenseIndex m = image.dimension(Secondary);
        const Eigen::DenseIndex channels = image.dimension(2);

        auto coord = [](Eigen::DenseIndex primary,
                        Eigen::DenseIndex secondary) {
            if constexpr (Axis::value == 0) {
                return std::pair{primary, secondary};
            }

            return std::pair{secondary, primary};
        };

        for (Eigen::DenseIndex k = 0; k < channels; ++k) {
            for (Eigen::DenseIndex s = 0; s < m; ++s) {
                if (n == 0) {
                    continue;
                }

                if (n == 1) {
                    const auto [i, j] = coord(0, s);
                    out(i, j, k) = Scalar{0};
                    continue;
                }

                {
                    const auto [i, j] = coord(0, s);
                    out(i, j, k) = lower(image, i, j, k);
                }

                for (Eigen::DenseIndex p = 1; p + 1 < n; ++p) {
                    const auto [i, j] = coord(p, s);
                    out(i, j, k) = interior(image, i, j, k);
                }

                {
                    const auto [i, j] = coord(n - 1, s);
                    out(i, j, k) = upper(image, i, j, k);
                }
            }
        }
    }

    InteriorSamplerX interiorX_;
    InteriorSamplerY interiorY_;
    LowerBoundSamplerX atLowerBorderX_;
    LowerBoundSamplerY atLowerBorderY_;
    UpperBoundSamplerX atUpperBorderX_;
    UpperBoundSamplerY atUpperBorderY_;
};

} // namespace hogpp

#endif // HOGPP_GRADIENT_HPP
