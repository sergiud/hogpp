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

#include <nanobind/nanobind.h>

#if defined(FMT_VERSION) && FMT_VERSION >= 110000
#    define HOGPP_FORMATTER_FORMAT_CONST const
#else // !(defined(FMT_VERSION) && FMT_VERSION >= 110000)
#    define HOGPP_FORMATTER_FORMAT_CONST
#endif // defined(FMT_VERSION) && FMT_VERSION >= 110000

template<>
struct fmt::formatter<nanobind::str> : formatter<string_view>
{
    template<class FormatContext>
    [[nodiscard]] constexpr auto format(
        const nanobind::str& s, FormatContext& ctx) HOGPP_FORMATTER_FORMAT_CONST
    {
        return formatter<string_view>::format(std::string_view{s.c_str()}, ctx);
    }
};

template<>
struct fmt::formatter<nanobind::handle> : formatter<nanobind::str>
{
    template<class FormatContext>
    [[nodiscard]] constexpr auto format(const nanobind::handle& o,
                                        FormatContext& ctx)
        HOGPP_FORMATTER_FORMAT_CONST
    {
        return formatter<nanobind::str>::format(nanobind::repr(o), ctx);
    }
};

template<>
struct fmt::formatter<nanobind::object> : formatter<nanobind::handle>
{
};

template<>
struct fmt::formatter<nanobind::float_> : formatter<nanobind::object>
{
};

template<>
struct fmt::formatter<nanobind::int_> : formatter<nanobind::object>
{
};

#undef HOGPP_FORMATTER_FORMAT_CONST

#endif // PYTHON_HOGPP_FORMATTER_HPP
