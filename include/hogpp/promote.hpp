//
// HOGpp - Fast histogram of oriented gradients computation using integral
// histograms
//
// Copyright 2021 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

#ifndef HOGPP_PROMOTE_HPP
#define HOGPP_PROMOTE_HPP

#include <type_traits>

namespace hogpp {

// clang-format off
template<class T>
using Promote_t = std::conditional_t
<
      std::is_same_v<T, bool>
    , char
    , std::conditional_t
    <
          std::is_unsigned_v<T>
        , std::conditional_t
        <
              std::is_same_v<T, char> || std::is_same_v<T, signed char>
            , short
            , std::conditional_t
            <
                  std::is_same_v<T, short>
                , int
                , std::conditional_t
                <
                      std::is_same_v<T, int> || std::is_same_v<T, long>
                    , long long
                    , T
                >
            >
        >
        , T
    >
>;
// clang-format on

} // namespace hogpp

#endif // HOGPP_PROMOTE_HPP
