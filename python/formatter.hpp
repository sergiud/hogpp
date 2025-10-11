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

#ifndef PYTHON_HOGPP_FORMATTER_HPP
#define PYTHON_HOGPP_FORMATTER_HPP

#include <string_view>

#include <fmt/format.h>

#include <pybind11/cast.h>

#if defined(FMT_VERSION) && FMT_VERSION >= 110000
#    define HOGPP_FORMATTER_FORMAT_CONST const
#else // !(defined(FMT_VERSION) && FMT_VERSION >= 110000)
#    define HOGPP_FORMATTER_FORMAT_CONST
#endif // defined(FMT_VERSION) && FMT_VERSION >= 110000

template<>
struct fmt::formatter<pybind11::str> : formatter<string_view>
{
    template<class FormatContext>
    [[nodiscard]] constexpr auto format(
        const pybind11::str& s, FormatContext& ctx) HOGPP_FORMATTER_FORMAT_CONST
    {
        return formatter<string_view>::format(s.cast<std::string_view>(), ctx);
    }
};

template<>
struct fmt::formatter<pybind11::handle> : formatter<pybind11::str>
{
    template<class FormatContext>
    [[nodiscard]] constexpr auto format(const pybind11::handle& o,
                                        FormatContext& ctx)
        HOGPP_FORMATTER_FORMAT_CONST
    {
        return formatter<pybind11::str>::format(pybind11::repr(o), ctx);
    }
};

template<>
struct fmt::formatter<pybind11::object> : formatter<pybind11::handle>
{
};

template<>
struct fmt::formatter<pybind11::float_> : formatter<pybind11::object>
{
};

template<>
struct fmt::formatter<pybind11::int_> : formatter<pybind11::object>
{
};

#undef HOGPP_FORMATTER_FORMAT_CONST

#endif // PYTHON_HOGPP_FORMATTER_HPP
