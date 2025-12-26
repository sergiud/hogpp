#include "moduledispatch.hpp"

void init_hogpp_default(pybind11::module& m);
void init_hogpp_sse2(pybind11::module& m);
void init_hogpp_sse3(pybind11::module& m);
void init_hogpp_ssse3(pybind11::module& m);
void init_hogpp_sse4_1(pybind11::module& m);
void init_hogpp_sse4_2(pybind11::module& m);
void init_hogpp_avx(pybind11::module& m);
void init_hogpp_avx2(pybind11::module& m);
void init_hogpp_avx512f(pybind11::module& m);
void init_hogpp_avx10_1(pybind11::module& m);
void init_hogpp_avx10_2(pybind11::module& m);
void init_hogpp_neon(pybind11::module& m);
void init_hogpp_sve128(pybind11::module& m);
void init_hogpp_sve256(pybind11::module& m);
void init_hogpp_sve512(pybind11::module& m);

namespace pyhogpp {

void ModuleDispatch<ISA::Default>::initialize(pybind11::module& m)
{
    init_hogpp_default(m);
}

#if defined(HAVE_ISA_SSE2)
void ModuleDispatch<ISA::SSE2>::initialize(pybind11::module& m)
{
    init_hogpp_sse2(m);
}
#endif

#if defined(HAVE_ISA_SSE3)
void ModuleDispatch<ISA::SSE3>::initialize(pybind11::module& m)
{
    init_hogpp_sse3(m);
}
#endif

#if defined(HAVE_ISA_SSSE3)
void ModuleDispatch<ISA::SSSE3>::initialize(pybind11::module& m)
{
    init_hogpp_ssse3(m);
}
#endif

#if defined(HAVE_ISA_SSE4_1)
void ModuleDispatch<ISA::SSE4_1>::initialize(pybind11::module& m)
{
    init_hogpp_sse4_1(m);
}
#endif

#if defined(HAVE_ISA_SSE4_2)
void ModuleDispatch<ISA::SSE4_2>::initialize(pybind11::module& m)
{
    init_hogpp_sse4_2(m);
}
#endif

#if defined(HAVE_ISA_AVX2)
void ModuleDispatch<ISA::AVX>::initialize(pybind11::module& m)
{
    init_hogpp_avx(m);
}
#endif

#if defined(HAVE_ISA_AVX2)
void ModuleDispatch<ISA::AVX2>::initialize(pybind11::module& m)
{
    init_hogpp_avx2(m);
}
#endif

#if defined(HAVE_ISA_AVX512F)
void ModuleDispatch<ISA::AVX512>::initialize(pybind11::module& m)
{
    init_hogpp_avx512f(m);
}
#endif

#if defined(HAVE_ISA_AVX10_1)
void ModuleDispatch<ISA::AVX10_1>::initialize(pybind11::module& m)
{
    init_hogpp_avx10_1(m);
}
#endif

#if defined(HAVE_ISA_AVX10_2)
void ModuleDispatch<ISA::AVX10_2>::initialize(pybind11::module& m)
{
    init_hogpp_avx10_2(m);
}
#endif

#if defined(HAVE_ISA_NEON)
void ModuleDispatch<ISA::NEON>::initialize(pybind11::module& m)
{
    init_hogpp_neon(m);
}
#endif

#if defined(HAVE_ISA_SVE128)
void ModuleDispatch<ISA::SVE128>::initialize(pybind11::module& m)
{
    init_hogpp_sve128(m);
}
#endif

#if defined(HAVE_ISA_SVE256)
void ModuleDispatch<ISA::SVE256>::initialize(pybind11::module& m)
{
    init_hogpp_sve256(m);
}
#endif

#if defined(HAVE_ISA_SVE512)
void ModuleDispatch<ISA::SVE512>::initialize(pybind11::module& m)
{
    init_hogpp_sve512(m);
}
#endif

} // namespace pyhogpp
