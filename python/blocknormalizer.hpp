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

#ifndef PYTHON_HOGPP_BLOCKNORMALIZER_HPP
#define PYTHON_HOGPP_BLOCKNORMALIZER_HPP

// FIXME GCC 14.x workaround for https://github.com/pybind/pybind11/pull/5208
#include <algorithm>
#include <optional>
#include <type_traits>
#include <variant>

#include <pybind11/pybind11.h>

#include <hogpp/l1hys.hpp>
#include <hogpp/l1norm.hpp>
#include <hogpp/l1sqrt.hpp>
#include <hogpp/l2hys.hpp>
#include <hogpp/l2norm.hpp>
#include <hogpp/prefix.hpp>

#include "typetraits.hpp"

enum class BlockNormalizerType
{
    L1,
    L1Hys,
    L1sqrt,
    L2,
    L2Hys
};

namespace pybind11::detail {

template<>
class type_caster<BlockNormalizerType>
{
public:
    bool load(handle src, bool);
    static handle cast(BlockNormalizerType in, return_value_policy /*policy*/,
                       handle /*parent*/);

private:
    PYBIND11_TYPE_CASTER(BlockNormalizerType, _("BlockNormalizer"));
};

} // namespace pybind11::detail

template<class T>
class BlockNormalizer
{
public:
    using L1 = hogpp::L1Norm<T>;
    using L1Hys = hogpp::L1Hys<T>;
    using L1Sqrt = hogpp::L1Sqrt<T>;
    using L2 = hogpp::L2Norm<T>;
    using L2Hys = hogpp::L2Hys<T>;

    [[nodiscard]] explicit BlockNormalizer(
        BlockNormalizerType type = BlockNormalizerType::L2Hys,
        const std::optional<pybind11::float_>& clip = std::nullopt,
        const std::optional<pybind11::float_>& epsilon = std::nullopt)
    {
        // TODO L1|L2Norm -> L1|L2
        switch (type) {
            case BlockNormalizerType::L1:
                norm_ = L1{epsilon ? pybind11::cast<T>(*epsilon)
                                   : L1::Traits::regularization()};
                break;
            case BlockNormalizerType::L1Hys:
                norm_ = L1Hys{
                    clip ? pybind11::cast<T>(*clip) : L1Hys::Traits::clip(),
                    epsilon ? pybind11::cast<T>(*epsilon)
                            : L1Hys::Traits::regularization()};
                break;
            case BlockNormalizerType::L2:
                norm_ = L2{epsilon ? pybind11::cast<T>(*epsilon)
                                   : L2::Traits::regularization()};
                break;
            case BlockNormalizerType::L2Hys:
                norm_ = L2Hys{
                    clip ? pybind11::cast<T>(*clip) : L2Hys::Traits::clip(),
                    epsilon ? pybind11::cast<T>(*epsilon)
                            : L2Hys::Traits::regularization()};
                break;
            case BlockNormalizerType::L1sqrt:
                norm_ = L1Sqrt{epsilon ? pybind11::cast<T>(*epsilon)
                                       : L1Sqrt::Traits::regularization()};
                break;
        }
    }

    template<class Tensor>
    void operator()(Tensor& block) const
    {
        std::visit([&block](auto& norm) { norm(block); }, norm_);
    }

    [[nodiscard]] BlockNormalizerType type() const noexcept
    {
        return std::visit(BlockNormalizerVisitor{}, norm_);
    }

    [[nodiscard]] pybind11::object clip() const noexcept
    {
        return std::visit(ClipVisitor{}, norm_);
    }

    [[nodiscard]] pybind11::object epsilon() const noexcept
    {
        return std::visit(RegularizationVisitor{}, norm_);
    }

private:
    struct BlockNormalizerVisitor
    {
        [[nodiscard]] constexpr BlockNormalizerType operator()(
            const L1& /*unused*/) const noexcept
        {
            return BlockNormalizerType::L1;
        }

        [[nodiscard]] constexpr BlockNormalizerType operator()(
            const L1Hys& /*unused*/) const noexcept
        {
            return BlockNormalizerType::L1Hys;
        }

        [[nodiscard]] constexpr BlockNormalizerType operator()(
            const L2& /*unused*/) const noexcept
        {
            return BlockNormalizerType::L2;
        }

        [[nodiscard]] constexpr BlockNormalizerType operator()(
            const L2Hys& /*unused*/) const noexcept
        {
            return BlockNormalizerType::L2Hys;
        }

        [[nodiscard]] constexpr BlockNormalizerType operator()(
            const L1Sqrt& /*unused*/) const noexcept
        {
            return BlockNormalizerType::L1sqrt;
        }
    };

    // clang-format off
    template<class U>
    using Demote_t = std::conditional_t
    <
          std::is_same_v<U, long double>
        , double
        , U
    >;
    // clang-format on

    struct ClipVisitor
    {
        template<class Norm, std::enable_if_t<HasClip_v<Norm>>* = nullptr>
        [[nodiscard]] pybind11::object operator()(
            const Norm& norm) const noexcept
        {
            using Scalar = Demote_t<typename Norm::Scalar>;
            return pybind11::float_{Scalar(norm.clip())};
        }

        template<class Norm, std::enable_if_t<!HasClip_v<Norm>>* = nullptr>
        [[nodiscard]] pybind11::object operator()(
            [[maybe_unused]] const Norm& norm) const noexcept
        {
            return pybind11::none{};
        }
    };

    struct RegularizationVisitor
    {
        template<class Norm, std::enable_if_t<HasNorm_v<Norm>>* = nullptr>
        [[nodiscard]] pybind11::object operator()(
            const Norm& norm) const noexcept
        {
            return (*this)(norm.norm());
        }

        template<class Norm,
                 std::enable_if_t<HasRegularization_v<Norm>>* = nullptr>
        [[nodiscard]] pybind11::object operator()(
            [[maybe_unused]] const Norm& norm) const noexcept
        {
            using Scalar = Demote_t<typename Norm::Scalar>;
            return pybind11::float_{Scalar(norm.regularization())};
        }

        template<class Norm,
                 std::enable_if_t<!HasNorm_v<Norm> &&
                                  !HasRegularization_v<Norm>>* = nullptr>
        [[nodiscard]] pybind11::object operator()(
            [[maybe_unused]] const Norm& norm) const noexcept
        {
            return pybind11::none{};
        }
    };

    // clang-format off
    std::variant
    <
          hogpp::L1Hys<T>
        , hogpp::L1Norm<T>
        , hogpp::L1Sqrt<T>
        , hogpp::L2Hys<T>
        , hogpp::L2Norm<T>
    >
    norm_;
    // clang-format on
};

extern template class BlockNormalizer<float>;
extern template class BlockNormalizer<double>;

#include <hogpp/suffix.hpp>

#endif // PYTHON_HOGPP_BLOCKNORMALIZER_HPP
