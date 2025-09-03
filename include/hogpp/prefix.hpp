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

#if defined(__clang__)
#    pragma clang diagnostic push
#elif defined(__GNUG__)
#    pragma GCC diagnostic push
#endif // defined(__clang__)

// -Wswitch-default was first added in Clang 18:
// https://github.com/llvm/llvm-project/commit/c28178298513f99dc869daa301fc25257df81688
// and in GCC 3.3.0. However, we require much more recent versions of GCC thus
// allowing us to omit the version check.
#if defined(__GNUC__) || defined(__clang__)
#    if !defined(__clang__) ||                              \
        ((defined(__APPLE__) && (__clang_major__ >= 17)) || \
         (__clang_major__ >= 18))
#        pragma GCC diagnostic ignored "-Wswitch-default"
#    endif // !defined(__clang__) || ((defined(__APPLE__) && (__clang_major__ >=
           // 17)) || (__clang_major__ >= 18))
#endif     // defined(__GNUC__) || defined(__clang__)

// -Wc++23-attribute-extensions was first added in Clang 19:
// https://github.com/llvm/llvm-project/commit/2b5f68a5f63d2342a056bf9f86bd116c100fd81a
#if defined(__clang__)
#    if (defined(__APPLE__) && (__clang_major__ >= 17)) || \
        (__clang_major__ >= 18)
#        pragma clang diagnostic ignored "-Wc++23-attribute-extensions"
#    endif // (defined(__APPLE__) && (__clang_major__ >= 17)) ||
           // (__clang_major__ >= 18)
#endif     // defined(__clang__)
