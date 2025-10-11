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

#include <pybind11/pybind11.h>

void init_hogpp_x86_64(pybind11::module& m);
void init_hogpp_x86_64_v2(pybind11::module& m);
void init_hogpp_x86_64_v3(pybind11::module& m);
void init_hogpp_x86_64_v4(pybind11::module& m);

#if defined(HOGPP_GIL_DISABLED)
#    define HOGPP_MODULE(name, module, ...) \
        PYBIND11_MODULE(name, module, pybind11::mod_gil_not_used())
#else // !defined(HOGPP_GIL_DISABLED)
#    define HOGPP_MODULE PYBIND11_MODULE
#endif // defined(HOGPP_GIL_DISABLED)

#if defined(HOGPP_SKBUILD)
#    define HOGPP_MODULE_NAME _hogpp
#else // !defined(HOGPP_SKBUILD)
#    define HOGPP_MODULE_NAME hogpp
#endif // defined(HOGPP_SKBUILD)

HOGPP_MODULE(HOGPP_MODULE_NAME, m)
{
    if (__builtin_cpu_supports("x86-64-v4")) {
        init_hogpp_x86_64_v4(m);
    }
    else if (__builtin_cpu_supports("x86-64-v3")) {
        init_hogpp_x86_64_v3(m);
    }
    else if (__builtin_cpu_supports("x86-64-v2")) {
        init_hogpp_x86_64_v2(m);
    }
    else {
        init_hogpp_x86_64(m);
    }
}
