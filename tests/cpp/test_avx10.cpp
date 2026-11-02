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

#define BOOST_TEST_MODULE hogpp

#include <algorithm>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "../../python/avx10.hpp"

#include <boost/test/included/unit_test.hpp>

namespace {

// Register values as enumerated by the Intel AVX10 Architecture Specification
// and checked by GCC's CPU detection.
constexpr std::uint32_t AVX10Leaf = 0x24;
constexpr std::uint32_t OSXSAVE = 1U << 27;
constexpr std::uint32_t AVX10 = 1U << 19;
// x87, SSE, YMM, opmask, upper halves of ZMM0-15 and ZMM16-31
constexpr std::uint64_t AVX512State = 0xe7;
// x87, SSE and YMM only
constexpr std::uint64_t AVXState = 0x7;

// A processor implementing AVX10 of the given version with all of its
// register state enabled by the operating system.
[[nodiscard]] constexpr pyhogpp::AVX10Enumeration avx10Processor(
    std::uint32_t version) noexcept
{
    pyhogpp::AVX10Enumeration result;
    result.maxLeaf = AVX10Leaf;
    result.features = OSXSAVE;
    result.maxExtendedSubleaf = 1;
    result.extendedFeatures = AVX10;
    result.avx10Features = version;
    result.xcr0 = AVX512State;
    return result;
}

} // namespace

BOOST_AUTO_TEST_CASE(avx10_1_processor)
{
    BOOST_TEST(pyhogpp::avx10Version(avx10Processor(1)) == 1U);
}

BOOST_AUTO_TEST_CASE(avx10_2_processor)
{
    BOOST_TEST(pyhogpp::avx10Version(avx10Processor(2)) == 2U);
}

// Bits above the version are reserved or enumerate other capabilities.
BOOST_AUTO_TEST_CASE(version_ignores_upper_bits)
{
    constexpr std::uint32_t UpperBits = 0xffffff00;

    BOOST_TEST(pyhogpp::avx10Version(avx10Processor(UpperBits | 1)) == 1U);
}

BOOST_AUTO_TEST_CASE(avx512_without_avx10)
{
    pyhogpp::AVX10Enumeration enumeration = avx10Processor(1);
    enumeration.extendedFeatures = 0;

    BOOST_TEST(pyhogpp::avx10Version(enumeration) == 0U);
}

// AVX10 uses the register state of AVX-512, which the operating system may
// leave disabled.
BOOST_AUTO_TEST_CASE(avx512_state_disabled)
{
    pyhogpp::AVX10Enumeration enumeration = avx10Processor(1);
    enumeration.xcr0 = AVXState;

    BOOST_TEST(pyhogpp::avx10Version(enumeration) == 0U);
}

// Without OSXSAVE, XCR0 cannot be read and its value is meaningless.
BOOST_AUTO_TEST_CASE(osxsave_disabled)
{
    pyhogpp::AVX10Enumeration enumeration = avx10Processor(1);
    enumeration.features = 0;

    BOOST_TEST(pyhogpp::avx10Version(enumeration) == 0U);
}

// The version is enumerated by leaf 24h, which the processor must implement.
BOOST_AUTO_TEST_CASE(avx10_leaf_unavailable)
{
    pyhogpp::AVX10Enumeration enumeration = avx10Processor(1);
    enumeration.maxLeaf = AVX10Leaf - 1;

    BOOST_TEST(pyhogpp::avx10Version(enumeration) == 0U);
}

// The AVX10 flag is enumerated by subleaf 1 of leaf 7, which the processor
// must implement.
BOOST_AUTO_TEST_CASE(extended_subleaf_unavailable)
{
    pyhogpp::AVX10Enumeration enumeration = avx10Processor(1);
    enumeration.maxExtendedSubleaf = 0;

    BOOST_TEST(pyhogpp::avx10Version(enumeration) == 0U);
}

namespace {

// Self-snoop flag of CPUID.(EAX=1):EDX at the bit position OSXSAVE has in ECX
constexpr std::uint32_t SelfSnoop = 1U << 27;

using Leaf = std::pair<std::uint32_t, std::uint32_t>;

// Processor with a CPUID table that records which leaves are queried and how
// often XCR0 is read.
struct FakeProcessor
{
    [[nodiscard]] pyhogpp::AVX10Enumeration enumerate()
    {
        return pyhogpp::enumerateAVX10(
            [this](std::uint32_t leaf, std::uint32_t subleaf) {
                queried.emplace_back(leaf, subleaf);
                const auto pos = leaves.find(Leaf{leaf, subleaf});
                return pos != leaves.end() ? pos->second
                                           : pyhogpp::CPUIDRegisters{};
            },
            [this] {
                ++xcr0Reads;
                return xcr0;
            });
    }

    [[nodiscard]] bool wasQueried(std::uint32_t leaf,
                                  std::uint32_t subleaf) const
    {
        return std::ranges::find(queried, Leaf{leaf, subleaf}) != queried.end();
    }

    std::map<Leaf, pyhogpp::CPUIDRegisters> leaves;
    std::uint64_t xcr0 = 0;
    std::vector<Leaf> queried;
    int xcr0Reads = 0;
};

// CPUID table of a processor implementing AVX10.1 with all of its register
// state enabled by the operating system.
[[nodiscard]] FakeProcessor avx10FakeProcessor()
{
    FakeProcessor result;
    result.leaves[{0, 0}] = {AVX10Leaf, 0, 0, 0};
    result.leaves[{1, 0}] = {0, 0, OSXSAVE, 0};
    result.leaves[{7, 0}] = {1, 0, 0, 0};
    result.leaves[{7, 1}] = {0, 0, 0, AVX10};
    result.leaves[{AVX10Leaf, 0}] = {0, 1, 0, 0};
    result.xcr0 = AVX512State;
    return result;
}

} // namespace

BOOST_AUTO_TEST_CASE(enumerates_avx10_processor)
{
    FakeProcessor processor = avx10FakeProcessor();

    BOOST_TEST(pyhogpp::avx10Version(processor.enumerate()) == 1U);
    BOOST_TEST(processor.xcr0Reads == 1);
}

// OSXSAVE is a flag of ECX. The flag at the same position of EDX denotes
// self-snooping, which is unrelated.
BOOST_AUTO_TEST_CASE(enumerates_osxsave_from_ecx)
{
    FakeProcessor processor = avx10FakeProcessor();
    processor.leaves[{1, 0}] = {0, 0, 0, SelfSnoop};

    BOOST_TEST(pyhogpp::avx10Version(processor.enumerate()) == 0U);
    // XGETBV faults if the operating system does not enable it.
    BOOST_TEST(processor.xcr0Reads == 0);
}

// The AVX10 flag is enumerated by subleaf 1 of leaf 7. The flag at the same
// position of subleaf 0 is unrelated.
BOOST_AUTO_TEST_CASE(enumerates_avx10_flag_from_subleaf_1)
{
    FakeProcessor processor = avx10FakeProcessor();
    processor.leaves[{7, 0}] = {1, 0, 0, AVX10};
    processor.leaves[{7, 1}] = {};

    BOOST_TEST(pyhogpp::avx10Version(processor.enumerate()) == 0U);
}

BOOST_AUTO_TEST_CASE(enumerates_version_from_ebx)
{
    FakeProcessor processor = avx10FakeProcessor();
    processor.leaves[{AVX10Leaf, 0}] = {2, 0, 2, 2};

    BOOST_TEST(pyhogpp::avx10Version(processor.enumerate()) == 0U);
}

// Leaves beyond the highest one the processor implements return the values of
// another leaf and must not be queried.
BOOST_AUTO_TEST_CASE(does_not_query_unimplemented_leaves)
{
    FakeProcessor processor = avx10FakeProcessor();
    processor.leaves[{0, 0}] = {AVX10Leaf - 1, 0, 0, 0};
    processor.leaves[{7, 0}] = {0, 0, 0, 0};

    BOOST_TEST(pyhogpp::avx10Version(processor.enumerate()) == 0U);
    BOOST_TEST(!processor.wasQueried(AVX10Leaf, 0));
    BOOST_TEST(!processor.wasQueried(7, 1));
}
