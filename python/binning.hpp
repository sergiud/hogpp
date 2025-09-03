//
// HOGpp - Fast histogram of oriented gradients computation using integral
// histograms
//
// Copyright 2025 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

#ifndef PYTHON_HOGPP_BINNING_HPP
#define PYTHON_HOGPP_BINNING_HPP

// FIXME GCC 14.x workaround for https://github.com/pybind/pybind11/pull/5208
#include <algorithm>
#include <variant>

#include <pybind11/pybind11.h>

#include <hogpp/prefix.hpp>
#include <hogpp/signedgradient.hpp>
#include <hogpp/unsignedgradient.hpp>

enum class BinningType
{
    Signed,
    Unsigned
};

namespace pybind11::detail {

template<>
class type_caster<BinningType>
{
public:
    bool load(handle src, bool);
    static handle cast(BinningType in, return_value_policy /*policy*/,
                       handle /*parent*/);

private:
    PYBIND11_TYPE_CASTER(BinningType, _("Binning"));
};

} // namespace pybind11::detail

template<class T>
class Binning
{
public:
    [[nodiscard]] explicit Binning(BinningType type = BinningType::Unsigned)
    {
        switch (type) {
            case BinningType::Signed:
                binning_ = hogpp::SignedGradient<T>{};
                break;
            case BinningType::Unsigned:
                binning_ = hogpp::UnsignedGradient<T>{};
                break;
        }
    }

    [[nodiscard]] constexpr T operator()(T dx, T dy) const
    {
        return std::visit([dx, dy](auto& binning) { return binning(dx, dy); },
                          binning_);
    }

    [[nodiscard]] BinningType type() const noexcept
    {
        return std::visit(BinningVisitor{}, binning_);
    }

private:
    struct BinningVisitor
    {
        [[nodiscard]] constexpr BinningType operator()(
            const hogpp::SignedGradient<T>& /*unused*/) const noexcept
        {
            return BinningType::Signed;
        }

        [[nodiscard]] constexpr BinningType operator()(
            const hogpp::UnsignedGradient<T>& /*unused*/) const noexcept
        {
            return BinningType::Unsigned;
        }
    };

    // clang-format off
    std::variant
    <
          hogpp::SignedGradient<T>
        , hogpp::UnsignedGradient<T>
    >
    binning_;
    // clang-format on
};

extern template class Binning<float>;
extern template class Binning<double>;

#include <hogpp/suffix.hpp>

#endif // PYTHON_HOGPP_BINNING_HPP
