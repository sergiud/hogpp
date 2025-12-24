#include "cpufeature.hpp"
#include "isaconfig.hpp"

#if defined(HAVE_SYS_AUXV_H)
#    include <sys/auxv.h>
#endif

#if defined(HAVE_ASM_HWCAP_H)
#    include <asm/hwcap.h>
#endif

bool CPUFeature<ISA::SSE2>::supported() noexcept
{
#if defined(HAVE_ISA_SSE2)
    return __builtin_cpu_supports("sse2");
#else
    return false;
#endif
}

bool CPUFeature<ISA::SSE3>::supported() noexcept
{
#if defined(HAVE_ISA_SSE3)
    return __builtin_cpu_supports("sse3");
#else
    return false;
#endif
}

bool CPUFeature<ISA::SSSE3>::supported() noexcept
{
#if defined(HAVE_ISA_SSSE3)
    return __builtin_cpu_supports("ssse3");
#else
    return false;
#endif
}

bool CPUFeature<ISA::SSE4_2>::supported() noexcept
{
#if defined(HAVE_ISA_SSE4_2)
    return __builtin_cpu_supports("sse4.2");
#else
    return false;
#endif
}

bool CPUFeature<ISA::SSE4_1>::supported() noexcept
{
#if defined(HAVE_ISA_SSE4_1)
    return __builtin_cpu_supports("sse4.1");
#else
    return false;
#endif
}

bool CPUFeature<ISA::AVX>::supported() noexcept
{
#if defined(HAVE_ISA_AVX)
    return __builtin_cpu_supports("avx");
#else
    return false;
#endif
}

bool CPUFeature<ISA::AVX2>::supported() noexcept
{
#if defined(HAVE_ISA_AVX2)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

bool CPUFeature<ISA::AVX512>::supported() noexcept
{
#if defined(HAVE_ISA_AVX512F)
    return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}

bool CPUFeature<ISA::AVX10_1>::supported() noexcept
{
#if defined(HAVE_ISA_AVX10_1)
    return __builtin_cpu_supports("avx10.1");
#else
    return false;
#endif
}

bool CPUFeature<ISA::AVX10_2>::supported() noexcept
{
#if defined(HAVE_ISA_AVX10_2)
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
#if defined(HAVE_GETAUXVAL) && defined(HAVE_ASM_HWCAP_H) && defined(HWCAP_SVE)
    return (getauxval(AT_HWCAP) & HWCAP_SVE) == HWCAP_SVE;
#else
    return false;
#endif
}
