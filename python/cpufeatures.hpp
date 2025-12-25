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

#ifndef PYTHON_CPUFEATURES_HPP
#define PYTHON_CPUFEATURES_HPP

#include "isa.hpp"

namespace pyhogpp {

template<ISA... Types>
struct CPUFeatures
{
};

using AvailableCPUFeatures = CPUFeatures
    // clang-format off
<
      ISA::AVX10_2
    , ISA::AVX10_1
    , ISA::AVX512
    , ISA::AVX2
    , ISA::AVX
    , ISA::SSE4_2
    , ISA::SSE4_1
    , ISA::SSSE3
    , ISA::SSE3
    , ISA::SSE2
    , ISA::SVE512
    , ISA::SVE256
    , ISA::SVE128
    , ISA::NEON
>;
// clang-format on

} // namespace pyhogpp

#endif
