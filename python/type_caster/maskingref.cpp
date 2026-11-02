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

#include <string>
#include <utility>

#include <fmt/format.h>

#include "../formatter.hpp"
#include "maskingref.hpp"

namespace {

// Defined outside of pybind11, whose inline literals namespace would otherwise
// hide the named argument literal of fmt.
[[nodiscard]] std::string invalidMaskMessage(pybind11::handle mask)
{
    using namespace fmt::literals;

    return fmt::format(
        FMT_STRING("IntegralHOGDescriptor.compute mask must be either a "
                   "callable or provide an indexer in terms of a __getitem__ "
                   "method that accepts a 2-tuple, e.g., a numpy.ndarray "
                   "instance, but a {type} object was given"),
        "type"_a = pybind11::type::handle_of(mask));
}

} // namespace

namespace pybind11::detail {

type_caster<std::optional<pyhogpp::MaskingRef>>::Indexer::Indexer(
    object getitem) noexcept
    : getitem_{std::move(getitem)}
{
}

bool type_caster<std::optional<pyhogpp::MaskingRef>>::Indexer::operator()(
    Eigen::DenseIndex i, Eigen::DenseIndex j) const
{
    return bool_(getitem_(make_tuple(i, j)));
}

bool type_caster<std::optional<pyhogpp::MaskingRef>>::load(handle src,
                                                           bool /*unused*/)
{
    if (src.is_none()) {
        return true;
    }

    if (hasattr(src, "__getitem__")) {
        // mask can be indexed, e.g., it's a numpy.ndarray
        indexer_.emplace(getattr(src, "__getitem__"));
    }
    else if (hasattr(src, "__call__")) {
        // mask is a callable
        indexer_.emplace(reinterpret_borrow<object>(src));
    }
    else {
        throw type_error{invalidMaskMessage(src)};
    }

    value.emplace(*indexer_);

    return true;
}

} // namespace pybind11::detail
