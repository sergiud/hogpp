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

#include "cpufeature.hpp"
#include "isaconfig.hpp"

#if defined(HAVE_SYS_AUXV_H)
#    include <sys/auxv.h>
#endif // defined(HAVE_SYS_AUXV_H)

#if defined(HAVE_ASM_HWCAP_H)
#    include <asm/hwcap.h>
#endif // defined(HAVE_ASM_HWCAP_H)

#if defined(HAVE_SYS_PRCTL_H)
#    include <sys/prctl.h>
#endif // defined(HAVE_SYS_PRCTL_H)

#if defined(HAVE___CHECK_ISA_SUPPORT)
#    include <immintrin.h>
#    include <isa_availability.h>
#endif // defined(HAVE___CHECK_ISA_SUPPORT)

// Cacade macro definition by constraining it to x86/x86-64 instead of checking
// for the platform inline to avoid unused macro definition warnings.
#if defined(__i386__) || defined(__x86_64__)
#    if defined(__has_builtin)
#        if __has_builtin(__builtin_cpu_supports)
#            define HOGPP_USE_BUILTIN_CPU_SUPPORTS
#        endif // __has_builtin(__builtin_cpu_supports)
#    endif     // defined(__has_builtin)
#endif         // defined(__i386__) || defined(__x86_64__)

namespace pyhogpp {

bool CPUFeature<ISA::SSE2>::supported() noexcept
{
#if defined(HAVE___CHECK_ISA_SUPPORT)
    return __check_isa_support(__IA_SUPPORT_VECTOR128, 0);
#elif defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return __builtin_cpu_supports("sse2");
#else  // !defined(HAVE___CHECK_ISA_SUPPORT)
    return false;
#endif // defined(HAVE___CHECK_ISA_SUPPORT)
}

bool CPUFeature<ISA::SSE3>::supported() noexcept
{
#if defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return __builtin_cpu_supports("sse3");
#else  // !defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return false;
#endif // defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
}

bool CPUFeature<ISA::SSSE3>::supported() noexcept
{
#if defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return __builtin_cpu_supports("ssse3");
#else  // !defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return false;
#endif // defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
}

bool CPUFeature<ISA::SSE4_2>::supported() noexcept
{
#if defined(HAVE___CHECK_ISA_SUPPORT)
    return __check_isa_support(__IA_SUPPORT_SSE42, 0);
#elif defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return __builtin_cpu_supports("sse4.2");
#else  // !defined(HAVE___CHECK_ISA_SUPPORT)
    return false;
#endif // defined(HAVE___CHECK_ISA_SUPPORT)
}

bool CPUFeature<ISA::SSE4_1>::supported() noexcept
{
#if defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return __builtin_cpu_supports("sse4.1");
#else  // !defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return false;
#endif // defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
}

bool CPUFeature<ISA::AVX>::supported() noexcept
{
#if defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return __builtin_cpu_supports("avx");
#else  // !defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return false;
#endif // defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
}

bool CPUFeature<ISA::AVX2>::supported() noexcept
{
#if defined(HAVE___CHECK_ISA_SUPPORT)
    return __check_isa_support(__IA_SUPPORT_VECTOR256, 0);
#elif defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return __builtin_cpu_supports("avx2");
#else  // !defined(HAVE___CHECK_ISA_SUPPORT)
    return false;
#endif // defined(HAVE___CHECK_ISA_SUPPORT)
}

bool CPUFeature<ISA::AVX512>::supported() noexcept
{
#if defined(HAVE___CHECK_ISA_SUPPORT)
    return __check_isa_support(__IA_SUPPORT_VECTOR512, 0);
#elif defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS)
    return __builtin_cpu_supports("avx512f");
#else  // !defined(HAVE___CHECK_ISA_SUPPORT)
    return false;
#endif // defined(HAVE___CHECK_ISA_SUPPORT)
}

bool CPUFeature<ISA::AVX10_1>::supported() noexcept
{
#if defined(HAVE___CHECK_ISA_SUPPORT)
    return __check_isa_support(__IA_SUPPORT_VECTOR512, 1);
#elif defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS) && defined(__GNUC__) && \
    (__GNUC__ >= 15)
    return __builtin_cpu_supports("avx10.1");
#else  // !defined(HAVE___CHECK_ISA_SUPPORT)
    return false;
#endif // defined(HAVE___CHECK_ISA_SUPPORT)
}

bool CPUFeature<ISA::AVX10_2>::supported() noexcept
{
#if defined(HAVE___CHECK_ISA_SUPPORT)
    return __check_isa_support(__IA_SUPPORT_AVX10_2, 0);
#elif defined(HOGPP_USE_BUILTIN_CPU_SUPPORTS) && defined(__GNUC__) && \
    (__GNUC__ >= 15)
    return __builtin_cpu_supports("avx10.2");
#else  // !defined(HAVE___CHECK_ISA_SUPPORT)
    return false;
#endif // defined(HAVE___CHECK_ISA_SUPPORT)
}

bool CPUFeature<ISA::NEON>::supported() noexcept
{
#if defined(__aarch64__) || defined(__ARM_NEON__) || defined(_M_ARM64)
    return true; // NEON always available
#elif defined(HAVE_GETAUXVAL) && defined(HAVE_ASM_HWCAP_H) && \
    defined(HWCAP_NEON)
    return (getauxval(AT_HWCAP) & HWCAP_NEON) == HWCAP_NEON;
#else  // !defined(__aarch64__) || defined(__ARM_NEON__) || defined(_M_ARM64)
    return false;
#endif // defined(__aarch64__) || defined(__ARM_NEON__) || defined(_M_ARM64)
}

bool CPUFeature<ISA::SVE>::supported() noexcept
{
#if defined(HAVE_GETAUXVAL) && defined(HAVE_ASM_HWCAP_H) && defined(HWCAP_SVE)
    return (getauxval(AT_HWCAP) & HWCAP_SVE) == HWCAP_SVE;
#else  // !defined(HAVE_GETAUXVAL) && defined(HAVE_ASM_HWCAP_H) &&
       // defined(HWCAP_SVE)
    return false;
#endif // defined(HAVE_GETAUXVAL) && defined(HAVE_ASM_HWCAP_H) &&
       // defined(HWCAP_SVE)
}

namespace {

// Returns the CPU's actual SVE vector length in bits, or 0 if it cannot be
// determined. A fixed-width SVE build (-msve-vector-bits=N) requires the
// runtime vector length to match N exactly. Querying only HWCAP_SVE (as
// CPUFeature<ISA::SVE>::supported() does) cannot tell apart hardware
// implementing a different width.
[[nodiscard]] long sveVectorLengthBits() noexcept
{
#if defined(HAVE_PRCTL) && defined(HAVE_SYS_PRCTL_H) && defined(PR_SVE_GET_VL)
    const long vl = prctl(PR_SVE_GET_VL);

    if (vl > 0) {
        return (vl & PR_SVE_VL_LEN_MASK) * 8;
    }
#endif // defined(HAVE_PRCTL) && defined(HAVE_SYS_PRCTL_H) &&
       // defined(PR_SVE_GET_VL)

    return 0;
}

} // namespace

bool CPUFeature<ISA::SVE128>::supported() noexcept
{
    return CPUFeature<ISA::SVE>::supported() && sveVectorLengthBits() == 128;
}

bool CPUFeature<ISA::SVE256>::supported() noexcept
{
    return CPUFeature<ISA::SVE>::supported() && sveVectorLengthBits() == 256;
}

bool CPUFeature<ISA::SVE512>::supported() noexcept
{
    return CPUFeature<ISA::SVE>::supported() && sveVectorLengthBits() == 512;
}

} // namespace pyhogpp
