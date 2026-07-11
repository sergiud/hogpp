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

#ifndef PYTHON_HOGPP_INTEGRALHOGDESCRIPTOR_HPP
#define PYTHON_HOGPP_INTEGRALHOGDESCRIPTOR_HPP

#include <Eigen/Core>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <hogpp/bounds.hpp>
#include <hogpp/integralhogdescriptor.hpp>
#include <hogpp/spatialinterpolation.hpp>
#include <hogpp/spatialweighting.hpp>

#include "binning.hpp"
#include "blocknormalizer.hpp"
#include "interpolation.hpp"
#include "magnitude.hpp"
#include "maskingref.hpp"
#include "type_caster/array2i.hpp"
#include "type_caster/maskingref.hpp"
#include "type_caster/typesequence.hpp"

template<class T, class SpatialWeighting = hogpp::NoSpatialWeighting,
         class SpatialInterpolation = hogpp::NoSpatialInterpolation>
using Descriptor =
    hogpp::IntegralHOGDescriptor<T, hogpp::Gradient<T>, Magnitude<T>,
                                 Binning<T>, BlockNormalizer<T>,
                                 SpatialWeighting, SpatialInterpolation>;

template<class T>
using GaussianDescriptor = Descriptor<T, hogpp::GaussianWeighting<T>>;

template<class T>
using TrilinearDescriptor =
    Descriptor<T, hogpp::NoSpatialWeighting, hogpp::TrilinearInterpolation>;

template<class T>
using GaussianTrilinearDescriptor =
    Descriptor<T, hogpp::GaussianWeighting<T>, hogpp::TrilinearInterpolation>;

// Every precision is combined with every spatial weighting and interpolation.
// The traits are compile-time parameters, so runtime selection has to pick
// among precompiled types.
template<class... T>
struct MakeDescriptorVariant_t
{
    // clang-format off
    using type = std::variant
    <
          Descriptor<T>...
        , GaussianDescriptor<T>...
        , TrilinearDescriptor<T>...
        , GaussianTrilinearDescriptor<T>...
    >;
    // clang-format on
};

template<class... T>
struct MakeDescriptorVariant_t<TypeSequence<T...>>
    : MakeDescriptorVariant_t<T...>
{
};

template<class... T>
using MakeDescriptorVariant = typename MakeDescriptorVariant_t<T...>::type;

// Supported underlying floating point types. Internal computations are
// performed using one of the type which is determined from the input.
using PrecisionTypes = TypeSequence<float, double, long double>;
// std::variant of descriptor of the supported precision types
using DescriptorVariant = MakeDescriptorVariant<PrecisionTypes>;
// Supported input types. Avoid using <cstdint> types because they can differ
// between compilers. Instead, rely on standard types.
// clang-format off
using SupportedTypes = TypeSequence
<
      bool
    , double
    , float
    , long double
    , char
    , signed char
    , unsigned char
    , short
    , signed short
    , unsigned short
    , int
    , signed int
    , unsigned int
    , long
    , signed long
    , unsigned long
    , long long
    , signed long long
    , unsigned long long
>;
// clang-format on

template<long... Ranks>
struct RankNTensor
{
    // The original Python object is retained (rather than an already-parsed
    // nanobind::ndarray<nanobind::ro>) so that consumers can independently
    // re-derive a properly Scalar-typed view via their own caster, mirroring
    // how the descriptor's compute() overloads dispatch across SupportedTypes.
    nanobind::object buf;
};

template<long... Ranks>
struct RankNTensorPair
{
    RankNTensor<Ranks...> buf1;
    RankNTensor<Ranks...> buf2;
};

template<long... Ranks>
class nanobind::detail::type_caster<RankNTensor<Ranks...>>
{
public:
    NB_TYPE_CASTER(RankNTensor<Ranks...>,
                   const_name("numpy.ndarray[n, m[, o]]"))

    [[nodiscard]] bool from_python(handle in, std::uint8_t flags,
                                   cleanup_list* /*cleanup*/) noexcept
    {
        const bool convert = (flags & static_cast<std::uint8_t>(cast_flags::convert)) != 0;

        nanobind::ndarray<nanobind::ro> a;

        if (!try_cast(in, a, convert)) {
            return false;
        }

        const bool supported =
            ((static_cast<long>(a.ndim()) == Ranks) || ...) &&
            compatible(a.dtype(), SupportedTypes{});

        if (supported) {
            value.buf = nanobind::borrow<nanobind::object>(in);
        }

        return supported;
    }

    [[nodiscard]] static handle from_cpp(const RankNTensor<Ranks...>& in,
                                         rv_policy /*policy*/,
                                         cleanup_list* /*cleanup*/)
    {
        return nanobind::object{in.buf}.release();
    }

private:
    template<class... T>
    [[nodiscard]] static constexpr bool compatible(
        const nanobind::dlpack::dtype& dt,
        TypeSequence<T...> /*unused*/) noexcept
    {
        return ((nanobind::dtype<T>() == dt) || ...);
    }
};

template<long... Ranks>
class nanobind::detail::type_caster<RankNTensorPair<Ranks...>>
{
public:
    NB_TYPE_CASTER(
        RankNTensorPair<Ranks...>,
        const_name(
            "Tuple[numpy.ndarray[n, m[, o]], numpy.ndarray[n, m[, o]]]"))

    [[nodiscard]] bool from_python(handle in, std::uint8_t flags,
                                   cleanup_list* /*cleanup*/) noexcept
    {
        const bool convert = (flags & static_cast<std::uint8_t>(cast_flags::convert)) != 0;

        std::tuple<RankNTensor<Ranks...>, RankNTensor<Ranks...>> pair;

        if (!try_cast(in, pair, convert)) {
            return false;
        }

        auto& [buf1, buf2] = pair;

        nanobind::ndarray<nanobind::ro> a1;
        nanobind::ndarray<nanobind::ro> a2;

        const bool supported = try_cast(buf1.buf, a1) &&
                               try_cast(buf2.buf, a2) &&
                               a1.ndim() == a2.ndim() &&
                               a1.dtype() == a2.dtype() && shapesEqual(a1, a2);

        if (supported) {
            value.buf1 = std::move(buf1);
            value.buf2 = std::move(buf2);
        }

        return supported;
    }

    [[nodiscard]] static handle from_cpp(const RankNTensorPair<Ranks...>& in,
                                         rv_policy policy,
                                         cleanup_list* cleanup)
    {
        return nanobind::make_tuple(
                   nanobind::detail::make_caster<
                       RankNTensor<Ranks...>>::from_cpp(in.buf1, policy,
                                                        cleanup),
                   nanobind::detail::make_caster<
                       RankNTensor<Ranks...>>::from_cpp(in.buf2, policy,
                                                        cleanup))
            .release();
    }

private:
    [[nodiscard]] static bool shapesEqual(
        const nanobind::ndarray<nanobind::ro>& a,
        const nanobind::ndarray<nanobind::ro>& b) noexcept
    {
        for (std::size_t i = 0; i != a.ndim(); ++i) {
            if (a.shape(i) != b.shape(i)) {
                return false;
            }
        }

        return true;
    }
};

using Rank2Or3Tensor = RankNTensor<2, 3>;
using Rank2Or3TensorPair = RankNTensorPair<2, 3>;

class IntegralHOGDescriptor
{
public:
    // Unfortunately, it is not possible to extract the argument types of a
    // constructor (particularly, because a class may define multiple
    // constructors implicitly). We therefore need to write the constructor
    // signature explicitly.

    // Pickled state of versions without spatial weighting
    // clang-format off
    using LegacyState = std::tuple
    <
          std::optional<Eigen::Array2i>
        , std::optional<Eigen::Array2i>
        , std::optional<Eigen::Array2i>
        , std::optional<nanobind::int_>
        , std::optional<MagnitudeType>
        , std::optional<BinningType>
        , std::optional<BlockNormalizerType>
        , std::optional<std::variant<nanobind::int_, nanobind::float_> >
        , std::optional<std::variant<nanobind::int_, nanobind::float_> >
        , nanobind::object
    >;
    // clang-format on

    // Appends sigma and the integral histogram of the spatially weighted votes
    // which restores the weighted features exactly. Pickled state of versions
    // without spatial interpolation.
    using WeightedState = decltype(std::tuple_cat(
        std::declval<LegacyState>(),
        std::declval<std::tuple<double, nanobind::object>>()));

    // Appends the spatial interpolation.
    using State = decltype(std::tuple_cat(
        std::declval<WeightedState>(),
        std::declval<std::tuple<std::optional<InterpolationType>>>()));

    [[nodiscard]] explicit IntegralHOGDescriptor(
        const std::optional<Eigen::Array2i>& cellSize,
        const std::optional<Eigen::Array2i>& blockSize,
        const std::optional<Eigen::Array2i>& blockStride,
        const std::optional<nanobind::int_>& numBins,
        const std::optional<MagnitudeType>& magnitude,
        const std::optional<BinningType>& binning,
        const std::optional<BlockNormalizerType>& blockNorm,
        const std::optional<std::variant<nanobind::int_, nanobind::float_>>&
            clipNorm,
        const std::optional<std::variant<nanobind::int_, nanobind::float_>>&
            epsilon,
        double sigma, const std::optional<InterpolationType>& interpolation);

    void compute(const Rank2Or3Tensor& image,
                 const std::optional<pyhogpp::MaskingRef>& mask);
    void compute(const Rank2Or3TensorPair& dydx,
                 const std::optional<pyhogpp::MaskingRef>& mask);

    [[nodiscard]] nanobind::object features() const;
    [[nodiscard]] nanobind::object featuresROI(const hogpp::Bounds& rect) const;
    [[nodiscard]] nanobind::object featuresROIs(
        const nanobind::iterable& rects) const;
    [[nodiscard]] std::tuple<int, int> cellSize() const;
    [[nodiscard]] std::tuple<int, int> blockSize() const;
    [[nodiscard]] std::tuple<int, int> blockStride() const;
    [[nodiscard]] Eigen::DenseIndex numBins() const;
    [[nodiscard]] nanobind::object histogram() const;
    [[nodiscard]] nanobind::object weightedHistogram() const;
    [[nodiscard]] BinningType binning() const;
    [[nodiscard]] BlockNormalizerType blockNormalizer() const;
    [[nodiscard]] MagnitudeType magnitude() const;
    [[nodiscard]] nanobind::object clipNorm() const noexcept;
    [[nodiscard]] nanobind::object epsilon() const noexcept;
    [[nodiscard]] double sigma() const noexcept;
    [[nodiscard]] InterpolationType interpolation() const;
    [[nodiscard]] explicit operator bool() const noexcept;

    [[nodiscard]] State state() const;
    [[nodiscard]] static IntegralHOGDescriptor fromState(
        const nanobind::tuple& value);
    [[nodiscard]] static IntegralHOGDescriptor fromState(const State& value);

    [[nodiscard]] std::string repr() const;

private:
    [[nodiscard]] bool isEmpty() const noexcept;
    void update();
    void update(const nanobind::dlpack::dtype& dt);

    std::optional<Eigen::Array2i> cellSize_;
    std::optional<Eigen::Array2i> blockSize_;
    std::optional<Eigen::Array2i> blockStride_;
    std::optional<nanobind::int_> numBins_;
    std::optional<MagnitudeType> magnitudeType_;
    std::optional<BinningType> binningType_;
    std::optional<BlockNormalizerType> blockNormalizerType_;
    std::optional<std::variant<nanobind::int_, nanobind::float_>> clipNorm_;
    std::optional<std::variant<nanobind::int_, nanobind::float_>> epsilon_;
    // Standard deviation of the Gaussian spatial weighting in pixels. 0
    // disables the weighting.
    double sigma_ = 0;
    std::optional<InterpolationType> interpolationType_;

    DescriptorVariant descriptor_;
};

#endif // PYTHON_HOGPP_INTEGRALHOGDESCRIPTOR_HPP
