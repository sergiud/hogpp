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

#include "cpufeature.hpp"
#include "isaconfig.hpp"

#if defined(HAVE_SYS_AUXV_H)
#    include <sys/auxv.h>
#endif

#if defined(HAVE_ASM_HWCAP_H)
#    include <asm/hwcap.h>
#endif

#if defined(HAVE___CHECK_ISA_SUPPORT)
#    include <immintrin.h>
#    include <isa_availability.h>
#endif

#if defined(_M_ARM64)
#    include <intrin.h>
#endif

// Cacade macro definition by constraining it to x86/x86-64 instead of checking
// for the platform inline to avoid unused macro definition warnings.
#if defined(__i386__) || defined(__x86_64__)
#    if defined(__has_builtin)
#        if __has_builtin(__builtin_cpu_supports)
#            define HOGPP_USE_BUILTIN_CPU_SUPPORTS
#        endif
#    endif
#endif

namespace pyhogpp {

#if defined(_M_ARM64) && defined(_MSC_VER)
//  Test whether Value >= 0 && Value < Max in terms of bits sets except for Max
//  LSB.
template<auto Value, decltype(Value) Max>
concept MaxBits = (Value & ~((1 << Max) - 1)) == 0;

template<unsigned char op0, unsigned char op1, unsigned char CRn,
         unsigned char CRm, unsigned char op2>
    requires(MaxBits<op0, 2> && //
             MaxBits<op1, 3> && //
             MaxBits<CRn, 4> && //
             MaxBits<CRm, 4> && //
             MaxBits<op2, 3>    //
             )
inline constexpr unsigned short SysReg = op0 << 14 | //
                                         op1 << 11 | //
                                         CRn << 7 |  //
                                         CRm << 3 |  //
                                         op2         //
    ;
#endif

bool CPUFeature<ISA::SSE2>::supported() noexcept
{
#if defined(HAVE___CHECK_ISA_SUPPORT)
    return __check_isa_support(__IA_SUPPORT_VECTOR128, 0);
#elif defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return __builtin_cpu_supports("sse2");
#else
    return false;
#endif
}

bool CPUFeature<ISA::SSE3>::supported() noexcept
{
#if defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return __builtin_cpu_supports("sse3");
#else
    return false;
#endif
}

bool CPUFeature<ISA::SSSE3>::supported() noexcept
{
#if defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return __builtin_cpu_supports("ssse3");
#else
    return false;
#endif
}

bool CPUFeature<ISA::SSE4_2>::supported() noexcept
{
#if defined(HAVE___CHECK_ISA_SUPPORT)
    return __check_isa_support(__IA_SUPPORT_SSE42, 0);
#elif defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return __builtin_cpu_supports("sse4.2");
#else
    return false;
#endif
}

bool CPUFeature<ISA::SSE4_1>::supported() noexcept
{
#if defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return __builtin_cpu_supports("sse4.1");
#else
    return false;
#endif
}

bool CPUFeature<ISA::AVX>::supported() noexcept
{
#if defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return __builtin_cpu_supports("avx");
#else
    return false;
#endif
}

bool CPUFeature<ISA::AVX2>::supported() noexcept
{
#if defined(HAVE___CHECK_ISA_SUPPORT)
    return __check_isa_support(__IA_SUPPORT_VECTOR256, 0);
#elif defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

bool CPUFeature<ISA::AVX512>::supported() noexcept
{
#if defined(HAVE___CHECK_ISA_SUPPORT)
    return __check_isa_support(__IA_SUPPORT_VECTOR512, 0);
#elif defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}

bool CPUFeature<ISA::AVX10_1>::supported() noexcept
{
#if defined(HAVE___CHECK_ISA_SUPPORT)
    return __check_isa_support(__IA_SUPPORT_VECTOR512, 1);
#elif defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS) && defined(__GNUC__) && \
    (__GNUC__ >= 15)
    return __builtin_cpu_supports("avx10.1");
#else
    return false;
#endif
}

bool CPUFeature<ISA::AVX10_2>::supported() noexcept
{
#if defined(HAVE___CHECK_ISA_SUPPORT)
    return __check_isa_support(__IA_SUPPORT_AVX10_2, 0);
#elif defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS) && defined(__GNUC__) && \
    (__GNUC__ >= 15)
    return __builtin_cpu_supports("avx10.2");
#else
    return false;
#endif
}

bool CPUFeature<ISA::NEON>::supported() noexcept
{
#if defined(__aarch64__) || defined(__ARM_NEON__) || defined(_M_ARM64)
    return true; // NEON always available
#elif defined(HAVE_GETAUXVAL) && defined(HAVE_ASM_HWCAP_H) && \
    defined(HWCAP_NEON)
    return (getauxval(AT_HWCAP) & HWCAP_NEON) == HWCAP_NEON;
#else
    return false;
#endif
}

bool CPUFeature<ISA::SVE>::supported() noexcept
{
#if defined(_M_ARM64) && defined(_MSC_VER)
    constexpr int ID_AA64PFR0_EL1 = SysReg<0b11, 0b000, 0b0000, 0b0100, 0b000>;
    // AArch64 Processor Feature Register 0
    const auto value = _ReadStatusReg(ID_AA64PFR0_EL1 & ((1 << 15) - 1));
    constexpr int FEAT_SVE = 0b0001;
    return ((value >> 32) & 0b1111) == FEAT_SVE;
#elif defined(HAVE_GETAUXVAL) && defined(HAVE_ASM_HWCAP_H) && defined(HWCAP_SVE)
    return (getauxval(AT_HWCAP) & HWCAP_SVE) == HWCAP_SVE;
#else
    return false;
#endif
}

} // namespace pyhogpp
