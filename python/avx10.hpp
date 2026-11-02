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

#ifndef PYTHON_AVX10_HPP
#define PYTHON_AVX10_HPP

#include <array>
#include <cstddef>
#include <cstdint>

namespace pyhogpp {

//! EAX, EBX, ECX and EDX of a CPUID leaf
using CPUIDRegisters = std::array<std::uint32_t, 4>;

/**
 * @brief Processor and operating system state that determines AVX10 support.
 *
 * Each member holds the register of the CPUID leaf, or the XCR0 register,
 * that the AVX10 enumeration reads.
 */
struct AVX10Enumeration
{
    //! Leaf that enumerates the AVX10 version
    static constexpr std::uint32_t AVX10Leaf = 0x24;
    //! Leaf that enumerates the extended features
    static constexpr std::uint32_t ExtendedFeaturesLeaf = 7;
    //! Bit of CPUID.(EAX=1):ECX set if XGETBV can read XCR0
    static constexpr std::uint32_t OSXSAVE = 1U << 27;
    //! Bit of CPUID.(EAX=7,ECX=1):EDX set if the processor implements AVX10
    static constexpr std::uint32_t AVX10 = 1U << 19;
    //! Bits of CPUID.(EAX=24h,ECX=0):EBX holding the AVX10 version
    static constexpr std::uint32_t VersionMask = 0xff;
    //! Bits of XCR0 for the SSE, YMM, opmask and ZMM state, which AVX10 shares
    //! with AVX-512
    static constexpr std::uint64_t AVX512State = 0xe6;

    //! Highest standard CPUID leaf, CPUID.(EAX=0):EAX
    std::uint32_t maxLeaf = 0;
    //! Feature flags including OSXSAVE, CPUID.(EAX=1):ECX
    std::uint32_t features = 0;
    //! Highest subleaf of the extended features, CPUID.(EAX=7,ECX=0):EAX
    std::uint32_t maxExtendedSubleaf = 0;
    //! Extended features including AVX10, CPUID.(EAX=7,ECX=1):EDX
    std::uint32_t extendedFeatures = 0;
    //! AVX10 version in bits 7:0, CPUID.(EAX=24h,ECX=0):EBX
    std::uint32_t avx10Features = 0;
    //! Register state enabled by the operating system
    std::uint64_t xcr0 = 0;
};

/**
 * @brief Determines the AVX10 version a processor supports and the operating
 * system enables.
 *
 * @return The AVX10 version, or 0 if AVX10 is not usable.
 */
[[nodiscard]] constexpr unsigned avx10Version(
    const AVX10Enumeration& enumeration) noexcept
{
    using E = AVX10Enumeration;

    if (enumeration.maxLeaf < E::AVX10Leaf ||
        (enumeration.features & E::OSXSAVE) == 0 ||
        (enumeration.xcr0 & E::AVX512State) != E::AVX512State ||
        enumeration.maxExtendedSubleaf < 1 ||
        (enumeration.extendedFeatures & E::AVX10) == 0) {
        return 0;
    }

    return enumeration.avx10Features & E::VersionMask;
}

/**
 * @brief Reads the state that determines AVX10 support.
 *
 * Leaves beyond the highest one the processor implements are not queried,
 * and XCR0 is read only if the operating system enables XGETBV.
 *
 * @param cpuid Returns the CPUIDRegisters of a leaf and subleaf.
 * @param xgetbv Returns XCR0.
 */
template<class CPUID, class XGETBV>
[[nodiscard]] AVX10Enumeration enumerateAVX10(CPUID&& cpuid, XGETBV&& xgetbv)
{
    constexpr std::size_t EAX = 0;
    constexpr std::size_t EBX = 1;
    constexpr std::size_t ECX = 2;
    constexpr std::size_t EDX = 3;

    AVX10Enumeration result;
    result.maxLeaf = cpuid(0U, 0U)[EAX];

    if (result.maxLeaf >= 1) {
        result.features = cpuid(1U, 0U)[ECX];
    }

    if ((result.features & AVX10Enumeration::OSXSAVE) != 0) {
        result.xcr0 = xgetbv();
    }

    if (result.maxLeaf >= AVX10Enumeration::ExtendedFeaturesLeaf) {
        result.maxExtendedSubleaf =
            cpuid(AVX10Enumeration::ExtendedFeaturesLeaf, 0U)[EAX];

        if (result.maxExtendedSubleaf >= 1) {
            result.extendedFeatures =
                cpuid(AVX10Enumeration::ExtendedFeaturesLeaf, 1U)[EDX];
        }
    }

    if (result.maxLeaf >= AVX10Enumeration::AVX10Leaf) {
        result.avx10Features = cpuid(AVX10Enumeration::AVX10Leaf, 0U)[EBX];
    }

    return result;
}

} // namespace pyhogpp

#endif // PYTHON_AVX10_HPP
