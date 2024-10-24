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

#ifndef HOGPP_ASSUME_HPP
#define HOGPP_ASSUME_HPP

#include <cassert>

#if !defined(HOGPP_ASSUME_ONLY)
#    if defined(__has_cpp_attribute)
#        if __has_cpp_attribute(assume) >= 202207L
#            define HOGPP_ASSUME_ONLY(expr) [[assume(expr)]]
#        endif // __has_cpp_attribute(assume) >= 202207L
#    endif     // defined(__has_cpp_attribute)
#endif         // !defined(HOGPP_ASSUME_ONLY)

#if !defined(HOGPP_ASSUME_ONLY)
#    if defined(_MSC_VER)
#        define HOGPP_ASSUME_ONLY(expr) __assume(expr)
#    endif //  defined(_MSC_VER)
#endif     // !defined(HOGPP_ASSUME_ONLY)

// Clang builtin
#if !defined(HOGPP_ASSUME_ONLY)
#    if defined(__has_builtin)
#        if __has_builtin(__builtin_assume)
#            define HOGPP_ASSUME_ONLY(expr) __builtin_assume(expr)
#        endif //  __has_builtin(__builtin_assume)
#    endif     // defined(__has_builtin)
#endif         // !defined(HOGPP_ASSUME_ONLY)

// GCC attribute
#if !defined(HOGPP_ASSUME_ONLY)
#    if defined(__has_attribute)
#        if __has_attribute(assume)
#            define HOGPP_ASSUME_ONLY(expr) __attribute__((assume(expr)))
#        endif //  __has_attribute(assume)
#    endif     // defined(__has_attribute)
#endif         // !defined(HOGPP_ASSUME_ONLY)

// Fallback
#if !defined(HOGPP_ASSUME_ONLY)
#    define HOGPP_ASSUME_ONLY(expr) (void)(expr)
#endif // !defined(HOGPP_ASSUME_ONLY)

// NOTE: Use the comma operator instead if a semicolon causes an "expected
// identifier before [" error when compiling using GCC.
#define HOGPP_ASSUME(expr)       \
    do {                         \
        assert(expr);            \
        HOGPP_ASSUME_ONLY(expr); \
    } while (0)

#endif // HOGPP_ASSUME_HPP
