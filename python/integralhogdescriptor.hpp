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

#include <optional>
#include <string>
#include <utility>
#include <variant>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <hogpp/bounds.hpp>
#include <hogpp/integralhogdescriptor.hpp>

#include "binning.hpp"
#include "blocknormalizer.hpp"
#include "magnitude.hpp"
#include "type_caster/typesequence.hpp"

template<class T>
using Descriptor =
    hogpp::IntegralHOGDescriptor<T, hogpp::Gradient<T>, Magnitude<T>,
                                 Binning<T>, BlockNormalizer<T>>;

template<class... T>
struct MakeDescriptorVariant_t
{
    using type = std::variant<Descriptor<T>...>;
};

template<class... T>
struct MakeDescriptorVariant_t<TypeSequence<T...>>
{
    using type = std::variant<Descriptor<T>...>;
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
    pybind11::buffer buf;
};

template<long... Ranks>
struct RankNTensorPair
{
    RankNTensor<Ranks...> buf1;
    RankNTensor<Ranks...> buf2;
};

template<long... Ranks>
class pybind11::detail::type_caster<RankNTensor<Ranks...>>
{
public:
    PYBIND11_TYPE_CASTER(RankNTensor<Ranks...>, _("numpy.ndarray[n, m[, o]]"));

    [[nodiscard]] bool load(handle in, bool /*unused*/)
    {
        try {
            auto a = pybind11::cast<pybind11::buffer>(in);
            auto info = a.request();

            bool supported = ((info.ndim == Ranks) || ...) &&
                             compatible(dtype{info}, SupportedTypes{});

            if (supported) {
                value.buf = a;
            }

            return supported;
        }
        catch (const pybind11::builtin_exception&) {
        }

        return false;
    }

    [[nodiscard]] static handle cast(const RankNTensor<Ranks...>& in,
                                     return_value_policy /*policy*/,
                                     handle /*parent*/)
    {
        return in.buf.release();
    }

private:
    template<class... T>
    [[nodiscard]] static constexpr bool compatible(
        const dtype& dt, TypeSequence<T...> /*unused*/) noexcept
    {
        return (dt.equal(dtype::of<T>()) || ...);
    }
};

template<long... Ranks>
class pybind11::detail::type_caster<RankNTensorPair<Ranks...>>
{
public:
    PYBIND11_TYPE_CASTER(
        RankNTensorPair<Ranks...>,
        _("Tuple[numpy.ndarray[n, m[, o]], numpy.ndarray[n, m[, o]]]"));

    [[nodiscard]] bool load(handle in, bool /*unused*/)
    {
        try {
            auto [buf1, buf2] = pybind11::cast<
                std::tuple<RankNTensor<Ranks...>, RankNTensor<Ranks...>>>(in);
            auto info1 = buf1.buf.request();
            auto info2 = buf2.buf.request();

            bool supported = info1.ndim == info2.ndim &&
                             info1.format == info2.format &&
                             info1.shape == info2.shape;

            if (supported) {
                value.buf1 = std::move(buf1);
                value.buf2 = std::move(buf2);
            }

            return supported;
        }
        catch (const pybind11::builtin_exception&) {
        }

        return false;
    }

    [[nodiscard]] static handle cast(const RankNTensorPair<Ranks...>& in,
                                     return_value_policy /*policy*/,
                                     handle /*parent*/)
    {
        return pybind11::make_tuple(in.buf1, in.buf2).release();
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

    // clang-format off
    using State = std::tuple
    <
          std::optional<Eigen::Array2i>
        , std::optional<Eigen::Array2i>
        , std::optional<Eigen::Array2i>
        , std::optional<pybind11::int_>
        , std::optional<MagnitudeType>
        , std::optional<BinningType>
        , std::optional<BlockNormalizerType>
        , std::optional<std::variant<pybind11::int_, pybind11::float_> >
        , std::optional<std::variant<pybind11::int_, pybind11::float_> >
        , pybind11::object
    >;
    // clang-format on

    [[nodiscard]] explicit IntegralHOGDescriptor(
        const std::optional<Eigen::Array2i>& cellSize,
        const std::optional<Eigen::Array2i>& blockSize,
        const std::optional<Eigen::Array2i>& blockStride,
        const std::optional<pybind11::int_>& numBins,
        const std::optional<MagnitudeType>& magnitude,
        const std::optional<BinningType>& binning,
        const std::optional<BlockNormalizerType>& blockNorm,
        const std::optional<std::variant<pybind11::int_, pybind11::float_>>&
            clipNorm,
        const std::optional<std::variant<pybind11::int_, pybind11::float_>>&
            epsilon);

    void compute(const Rank2Or3Tensor& image, const pybind11::handle& mask);
    void compute(const Rank2Or3TensorPair& dydx, const pybind11::handle& mask);

    [[nodiscard]] pybind11::object features() const;
    [[nodiscard]] pybind11::object featuresROI(const hogpp::Bounds& rect) const;
    [[nodiscard]] pybind11::object featuresROIs(
        const pybind11::iterable& rects) const;
    [[nodiscard]] std::tuple<int, int> cellSize() const;
    [[nodiscard]] std::tuple<int, int> blockSize() const;
    [[nodiscard]] std::tuple<int, int> blockStride() const;
    [[nodiscard]] Eigen::DenseIndex numBins() const;
    [[nodiscard]] pybind11::object histogram() const;
    [[nodiscard]] BinningType binning() const;
    [[nodiscard]] BlockNormalizerType blockNormalizer() const;
    [[nodiscard]] MagnitudeType magnitude() const;
    [[nodiscard]] pybind11::object clipNorm() const noexcept;
    [[nodiscard]] pybind11::object epsilon() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

    [[nodiscard]] State state() const;
    [[nodiscard]] static IntegralHOGDescriptor fromState(const State& value);

    [[nodiscard]] std::string repr() const;

private:
    [[nodiscard]] bool isEmpty() const noexcept;
    void update();
    void update(const pybind11::dtype& dt);

    std::optional<Eigen::Array2i> cellSize_;
    std::optional<Eigen::Array2i> blockSize_;
    std::optional<Eigen::Array2i> blockStride_;
    std::optional<pybind11::int_> numBins_;
    std::optional<MagnitudeType> magnitudeType_;
    std::optional<BinningType> binningType_;
    std::optional<BlockNormalizerType> blockNormalizerType_;
    std::optional<std::variant<pybind11::int_, pybind11::float_>> clipNorm_;
    std::optional<std::variant<pybind11::int_, pybind11::float_>> epsilon_;

    DescriptorVariant descriptor_;
};

#endif // PYTHON_HOGPP_INTEGRALHOGDESCRIPTOR_HPP
