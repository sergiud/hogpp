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

#ifndef HOGPP_NOUNIQUEADDRESS_HPP
#define HOGPP_NOUNIQUEADDRESS_HPP

// Allows members of an empty type to occupy no storage. MSVC and clang-cl
// accept the standard attribute but ignore it to retain ABI compatibility.
#if !defined(HOGPP_NO_UNIQUE_ADDRESS)
#    if defined(_MSC_VER) && defined(__has_cpp_attribute)
#        if __has_cpp_attribute(msvc::no_unique_address)
#            define HOGPP_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#        endif // __has_cpp_attribute(msvc::no_unique_address)
#    endif     // defined(_MSC_VER) && defined(__has_cpp_attribute)
#endif         // !defined(HOGPP_NO_UNIQUE_ADDRESS)

#if !defined(HOGPP_NO_UNIQUE_ADDRESS)
#    if !defined(_MSC_VER) && defined(__has_cpp_attribute)
#        if __has_cpp_attribute(no_unique_address)
#            define HOGPP_NO_UNIQUE_ADDRESS [[no_unique_address]]
#        endif // __has_cpp_attribute(no_unique_address)
#    endif     // !defined(_MSC_VER) && defined(__has_cpp_attribute)
#endif         // !defined(HOGPP_NO_UNIQUE_ADDRESS)

// Fallback
#if !defined(HOGPP_NO_UNIQUE_ADDRESS)
#    define HOGPP_NO_UNIQUE_ADDRESS
#endif // !defined(HOGPP_NO_UNIQUE_ADDRESS)

#endif // HOGPP_NOUNIQUEADDRESS_HPP
