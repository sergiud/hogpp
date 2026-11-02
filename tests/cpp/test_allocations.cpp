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

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <random>
#include <type_traits>

#include <malloc.h>

#include <hogpp/integralhogdescriptor.hpp>
#include <hogpp/spatialinterpolation.hpp>
#include <hogpp/spatialweighting.hpp>

#include <boost/mpl/list.hpp>
#include <boost/test/included/unit_test.hpp>

// The heap of the whole process is observed by replacing the allocation
// functions of glibc. This captures allocations of Eigen and the standard
// library alike.
extern "C" {
void* __libc_malloc(std::size_t size);
void* __libc_calloc(std::size_t count, std::size_t size);
void* __libc_realloc(void* ptr, std::size_t size);
void* __libc_memalign(std::size_t alignment, std::size_t size);
void __libc_free(void* ptr);
}

namespace {

std::atomic<long long> allocationCount{0};
std::atomic<long long> liveBytes{0};
std::atomic<long long> peakBytes{0};

void* recordAllocation(void* ptr) noexcept
{
    if (ptr != nullptr) {
        allocationCount.fetch_add(1, std::memory_order_relaxed);

        const long long live =
            liveBytes.fetch_add(static_cast<long long>(malloc_usable_size(ptr)),
                                std::memory_order_relaxed) +
            static_cast<long long>(malloc_usable_size(ptr));

        long long peak = peakBytes.load(std::memory_order_relaxed);

        while (live > peak && !peakBytes.compare_exchange_weak(
                                  peak, live, std::memory_order_relaxed)) {
        }
    }

    return ptr;
}

void recordDeallocation(void* ptr) noexcept
{
    if (ptr != nullptr) {
        liveBytes.fetch_sub(static_cast<long long>(malloc_usable_size(ptr)),
                            std::memory_order_relaxed);
    }
}

// Heap usage since construction. Blocks allocated before and freed during
// the observation lower the live bytes, so the peak is relative to the heap
// usage at construction.
class HeapProbe
{
public:
    HeapProbe() noexcept
        : allocations_{allocationCount.load()}
        , live_{liveBytes.load()}
    {
        peakBytes.store(live_);
    }

    [[nodiscard]] long long allocations() const noexcept
    {
        return allocationCount.load() - allocations_;
    }

    [[nodiscard]] long long peakIncrease() const noexcept
    {
        return peakBytes.load() - live_;
    }

private:
    long long allocations_;
    long long live_;
};

} // namespace

extern "C" {

void* malloc(std::size_t size)
{
    return recordAllocation(__libc_malloc(size));
}

void* calloc(std::size_t count, std::size_t size)
{
    return recordAllocation(__libc_calloc(count, size));
}

void* realloc(void* ptr, std::size_t size)
{
    recordDeallocation(ptr);
    return recordAllocation(__libc_realloc(ptr, size));
}

void* memalign(std::size_t alignment, std::size_t size)
{
    return recordAllocation(__libc_memalign(alignment, size));
}

void* aligned_alloc(std::size_t alignment, std::size_t size)
{
    return recordAllocation(__libc_memalign(alignment, size));
}

int posix_memalign(void** ptr, std::size_t alignment, std::size_t size)
{
    void* const result = recordAllocation(__libc_memalign(alignment, size));

    if (result == nullptr) {
        return ENOMEM;
    }

    *ptr = result;
    return 0;
}

void free(void* ptr)
{
    recordDeallocation(ptr);
    __libc_free(ptr);
}

} // extern "C"

namespace {

using Scalars = boost::mpl::list<float, double>;

template<class Scalar>
[[nodiscard]] Eigen::Tensor<Scalar, 3> makeImage(Eigen::DenseIndex rows,
                                                 Eigen::DenseIndex cols)
{
    std::default_random_engine g{1};
    std::uniform_real_distribution<Scalar> intensity{0, 1};

    Eigen::Tensor<Scalar, 3> image(rows, cols, 1);

    for (Eigen::DenseIndex i = 0; i < image.size(); ++i) {
        image.data()[i] = intensity(g);
    }

    return image;
}

template<class Scalar, class SpatialWeighting, class SpatialInterpolation>
using Descriptor = hogpp::IntegralHOGDescriptor<
    Scalar, hogpp::Gradient<Scalar>, hogpp::GradientMagnitude<Scalar>,
    hogpp::UnsignedGradient<Scalar>, hogpp::L2Hys<Scalar>, SpatialWeighting,
    SpatialInterpolation>;

template<class Descriptor>
[[nodiscard]] Descriptor makeDescriptor()
{
    using Scalar = typename Descriptor::Scalar;
    using SpatialWeighting = typename Descriptor::SpatialWeighting;

    constexpr Scalar Sigma = 4;

    if constexpr (std::is_same_v<SpatialWeighting, hogpp::NoSpatialWeighting>) {
        return Descriptor{};
    }
    else {
        return Descriptor{hogpp::Gradient<Scalar>{},
                          hogpp::GradientMagnitude<Scalar>{},
                          hogpp::UnsignedGradient<Scalar>{},
                          hogpp::L2Hys<Scalar>{}, SpatialWeighting{Sigma}};
    }
}

template<class Descriptor>
[[nodiscard]] long long featuresAllocations(Eigen::DenseIndex rows,
                                            Eigen::DenseIndex cols)
{
    using Scalar = typename Descriptor::Scalar;

    Descriptor d = makeDescriptor<Descriptor>();
    d.compute(makeImage<Scalar>(rows, cols));

    const HeapProbe probe;
    const typename Descriptor::Tensor5 X = d.features();

    return probe.allocations();
}

// Additional allocations of features() on a larger input over a smaller one.
template<class Descriptor>
[[nodiscard]] long long featuresAllocationGrowth()
{
    constexpr Eigen::DenseIndex SmallRows = 32;
    constexpr Eigen::DenseIndex SmallCols = 40;
    constexpr Eigen::DenseIndex LargeRows = 96;
    constexpr Eigen::DenseIndex LargeCols = 120;

    return featuresAllocations<Descriptor>(LargeRows, LargeCols) -
           featuresAllocations<Descriptor>(SmallRows, SmallCols);
}

// The interpolation must not allocate per cell. Allocations per block of the
// block normalization are shared with the descriptor without interpolation.
template<class Scalar, class SpatialWeighting>
void checkInterpolationAllocatesNoMorePerCell()
{
    BOOST_TEST(
        (featuresAllocationGrowth<Descriptor<
             Scalar, SpatialWeighting, hogpp::TrilinearInterpolation>>()) ==
        (featuresAllocationGrowth<Descriptor<
             Scalar, SpatialWeighting, hogpp::NoSpatialInterpolation>>()));
}

} // namespace

BOOST_AUTO_TEST_CASE_TEMPLATE(trilinear_features_allocate_no_more_per_cell,
                              Scalar, Scalars)
{
    checkInterpolationAllocatesNoMorePerCell<Scalar,
                                             hogpp::NoSpatialWeighting>();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(
    weighted_trilinear_features_allocate_no_more_per_cell, Scalar, Scalars)
{
    checkInterpolationAllocatesNoMorePerCell<
        Scalar, hogpp::GaussianWeighting<Scalar>>();
}

namespace {

template<class Descriptor>
[[nodiscard]] long long computeAllocations(Eigen::DenseIndex rows,
                                           Eigen::DenseIndex cols)
{
    using Scalar = typename Descriptor::Scalar;

    Descriptor d = makeDescriptor<Descriptor>();
    const Eigen::Tensor<Scalar, 3> image = makeImage<Scalar>(rows, cols);

    const HeapProbe probe;
    d.compute(image);

    return probe.allocations();
}

// Additional allocations of compute() on a larger image over a smaller one.
template<class Descriptor>
[[nodiscard]] long long computeAllocationGrowth()
{
    constexpr Eigen::DenseIndex SmallRows = 32;
    constexpr Eigen::DenseIndex SmallCols = 40;
    constexpr Eigen::DenseIndex LargeRows = 96;
    constexpr Eigen::DenseIndex LargeCols = 120;

    return computeAllocations<Descriptor>(LargeRows, LargeCols) -
           computeAllocations<Descriptor>(SmallRows, SmallCols);
}

} // namespace

// Voting into the integral histogram must not allocate per pixel. The
// allocations of compute() are therefore independent of the image size.
BOOST_AUTO_TEST_CASE_TEMPLATE(compute_allocates_no_more_per_pixel, Scalar,
                              Scalars)
{
    BOOST_TEST((computeAllocationGrowth<
                   Descriptor<Scalar, hogpp::NoSpatialWeighting,
                              hogpp::NoSpatialInterpolation>>()) == 0);
}

// The spatial weighting repeats the voting of every pixel and accumulates the
// weighted votes in an integral histogram of its own, neither of which may
// allocate per pixel.
BOOST_AUTO_TEST_CASE_TEMPLATE(weighted_compute_allocates_no_more_per_pixel,
                              Scalar, Scalars)
{
    BOOST_TEST((computeAllocationGrowth<
                   Descriptor<Scalar, hogpp::GaussianWeighting<Scalar>,
                              hogpp::NoSpatialInterpolation>>()) == 0);
}

// Recomputing replaces the second-order integral histogram of the
// interpolation. The previous one must be released first rather than being
// kept alive next to its replacement.
BOOST_AUTO_TEST_CASE_TEMPLATE(trilinear_recompute_keeps_one_cascade, Scalar,
                              Scalars)
{
    constexpr Eigen::DenseIndex Rows = 128;
    constexpr Eigen::DenseIndex Cols = 160;
    constexpr Eigen::DenseIndex Bins = 9;

    // Size of the second-order integral histogram in at least double
    // precision with one additional slot along each axis
    constexpr auto CascadeBytes =
        static_cast<long long>((Rows + 2) * (Cols + 2) * Bins *
                               static_cast<Eigen::DenseIndex>(
                                   sizeof(std::common_type_t<Scalar, double>)));

    auto d = makeDescriptor<Descriptor<Scalar, hogpp::NoSpatialWeighting,
                                       hogpp::TrilinearInterpolation>>();
    const Eigen::Tensor<Scalar, 3> image = makeImage<Scalar>(Rows, Cols);
    d.compute(image);

    const HeapProbe probe;
    d.compute(image);

    BOOST_TEST(probe.peakIncrease() < CascadeBytes);
}
