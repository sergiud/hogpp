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
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <fmt/format.h>

#include <hogpp/integralhogdescriptor.hpp>
#include <hogpp/nouniqueaddress.hpp>
#include <hogpp/spatialweighting.hpp>

#include <boost/mpl/list.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/test/included/unit_test.hpp>

using Scalars = boost::mpl::list<float, double, long double>;

namespace {

// Extended precision used by the brute-force reference.
using Reference = boost::multiprecision::cpp_bin_float_quad;

// Maximum deviation from the reference in multiples of the machine epsilon
// relative to the vote mass of the summed-area table prefix the result is
// derived from.
constexpr double Tolerance = 1024;

constexpr Eigen::DenseIndex DefaultNumBins = 5;
constexpr int ImageRows = 29;
constexpr int ImageCols = 31;
constexpr unsigned DefaultSeed = 42;
constexpr unsigned OtherSeed = 7;
// Size difference of a smaller second image
constexpr int ShrinkRows = 7;
constexpr int ShrinkCols = 5;

// Every ZeroMagnitudePeriod-th pixel receives no vote.
constexpr int ZeroMagnitudePeriod = 11;
// Every MaskPeriod-th pixel is masked out.
constexpr int MaskPeriod = 3;

// Votes the vertical gradient as magnitude so that the votes are known
// exactly.
struct TestMagnitude
{
    template<class Tensor>
    [[nodiscard]] auto operator()(const Tensor& /*dx*/, const Tensor& dy) const
    {
        return dy.abs();
    }
};

// Interprets the horizontal gradient directly as the normalized orientation.
struct TestBinning
{
    template<class Scalar>
    [[nodiscard]] Scalar operator()(Scalar dx, Scalar /*dy*/) const noexcept
    {
        return dx;
    }
};

// Keeps the weighted cell histograms unnormalized so that they can be compared
// against the reference directly.
struct IdentityNormalizer
{
    template<class Block>
    void operator()(Block& /*block*/) const noexcept
    {
    }
};

struct PrepareLog
{
    int calls = 0;
    Eigen::Array2i dims = Eigen::Array2i::Zero();
    Eigen::DenseIndex bins = 0;
    Reference total{0};
};

// Records the votes passed to prepare() and otherwise behaves like no
// weighting.
class RecordingWeighting : public hogpp::NoSpatialWeighting
{
public:
    explicit RecordingWeighting(PrepareLog* log) noexcept
        : log_{log}
    {
    }

    template<class Scalar, class VisitVotes>
    void prepare(State<Scalar>& /*state*/,
                 const hogpp::BlockGeometry& /*geometry*/,
                 const Eigen::Array2i& dims, Eigen::DenseIndex bins,
                 VisitVotes&& visitVotes) const
    {
        ++log_->calls;
        log_->dims = dims;
        log_->bins = bins;
        log_->total = 0;

        for (Eigen::DenseIndex i = 0; i < dims.x(); ++i) {
            for (Eigen::DenseIndex j = 0; j < dims.y(); ++j) {
                visitVotes(
                    std::make_tuple(i, j),
                    [log = log_](Eigen::DenseIndex /*bin*/, Scalar value) {
                        log->total += Reference{value};
                    });
            }
        }
    }

private:
    PrepareLog* log_ = nullptr;
};

template<class T, class SpatialWeighting = hogpp::GaussianWeighting<T>>
using TestDescriptor =
    hogpp::IntegralHOGDescriptor<T, void, TestMagnitude, TestBinning,
                                 IdentityNormalizer, SpatialWeighting>;

template<class T>
struct Sample
{
    Eigen::Tensor<T, 3> dxs;
    Eigen::Tensor<T, 3> dys;
};

template<class T>
[[nodiscard]] Sample<T> makeSample(int rows, int cols, unsigned seed)
{
    std::mt19937 rng{seed};
    std::uniform_real_distribution<T> orientation{T(0), T(1)};
    std::uniform_real_distribution<T> magnitude{T(0.25), T(1)};

    Sample<T> result;
    result.dxs.resize(rows, cols, 1);
    result.dys.resize(rows, cols, 1);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.dxs(i, j, 0) = orientation(rng);
            result.dys(i, j, 0) = (i * cols + j) % ZeroMagnitudePeriod == 0
                                      ? T(0)
                                      : magnitude(rng);
        }
    }

    return result;
}

[[nodiscard]] bool isMasked(Eigen::DenseIndex i, Eigen::DenseIndex j) noexcept
{
    return (i + 2 * j) % MaskPeriod == 0;
}

// Per-pixel votes computed in the working precision exactly like the
// descriptor does and stored in extended precision.
class Votes
{
public:
    Votes(int rows, int cols, Eigen::DenseIndex bins)
        : rows_{rows}
        , cols_{cols}
        , bins_{bins}
        , values_(static_cast<std::size_t>(rows * cols * bins))
    {
    }

    [[nodiscard]] Reference& operator()(int i, int j, Eigen::DenseIndex k)
    {
        return values_[index(i, j, k)];
    }

    [[nodiscard]] const Reference& operator()(int i, int j,
                                              Eigen::DenseIndex k) const
    {
        return values_[index(i, j, k)];
    }

    [[nodiscard]] Eigen::DenseIndex bins() const noexcept
    {
        return bins_;
    }

    // Returns the sum of the votes of the bin k over all pixels in the rows
    // [0, i) and the columns [0, j).
    [[nodiscard]] const Reference& prefix(int i, int j,
                                          Eigen::DenseIndex k) const
    {
        return prefix_[static_cast<std::size_t>((i * (cols_ + 1) + j) * bins_ +
                                                k)];
    }

    void integrate()
    {
        prefix_.assign(
            static_cast<std::size_t>((rows_ + 1) * (cols_ + 1) * bins_),
            Reference{0});

        auto at = [this](int i, int j, Eigen::DenseIndex k) -> Reference& {
            return prefix_[static_cast<std::size_t>(
                (i * (cols_ + 1) + j) * bins_ + k)];
        };

        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                for (Eigen::DenseIndex k = 0; k < bins_; ++k) {
                    at(i + 1, j + 1, k) = at(i, j + 1, k) + at(i + 1, j, k) -
                                          at(i, j, k) + (*this)(i, j, k);
                }
            }
        }
    }

private:
    [[nodiscard]] std::size_t index(int i, int j,
                                    Eigen::DenseIndex k) const noexcept
    {
        return static_cast<std::size_t>((i * cols_ + j) * bins_ + k);
    }

    int rows_;
    int cols_;
    Eigen::DenseIndex bins_;
    std::vector<Reference> values_;
    std::vector<Reference> prefix_;
};

template<class T>
[[nodiscard]] Votes makeVotes(const Sample<T>& sample, Eigen::DenseIndex bins,
                              bool masked)
{
    const auto rows = static_cast<int>(sample.dxs.dimension(0));
    const auto cols = static_cast<int>(sample.dxs.dimension(1));

    Votes votes{rows, cols, bins};

    const auto scale = static_cast<T>(bins - 1);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const T mag = sample.dys(i, j, 0);

            if ((masked && isMasked(i, j)) || !(mag > 0)) {
                continue;
            }

            using std::floor;
            using std::min;

            const T center = sample.dxs(i, j, 0) * scale;
            const T lower = floor(center);
            const T upper = min(lower + 1, scale);
            const T alpha = center - lower;

            votes(i, j, static_cast<Eigen::DenseIndex>(lower)) +=
                Reference{T(1) - alpha} * Reference{mag};
            votes(i, j, static_cast<Eigen::DenseIndex>(upper)) +=
                Reference{alpha} * Reference{mag};
        }
    }

    votes.integrate();

    return votes;
}

// Gaussian weight of the block-relative pixel position u normalized to the
// pixels closest to the block center. An infinite sigma weights all pixels
// equally. A zero sigma is not a valid input because the reference type
// flushes subnormal values such as the long double denorm_min to 0.
[[nodiscard]] Reference referenceWeight(int u, int extent,
                                        const Reference& sigma)
{
    const int e = 2 * u - (extent - 1);
    const int emin = extent % 2 == 0 ? 1 : 0;
    const int excess = (e * e - emin * emin) / 4;

    if (excess == 0) {
        return Reference{1};
    }

    return exp(-((Reference{excess} / sigma) / sigma) / 2);
}

struct Geometry
{
    Eigen::Array2i cellSize;
    Eigen::Array2i blockSize;
    Eigen::Array2i blockStride;
};

[[nodiscard]] std::string describe(const Geometry& g)
{
    using namespace fmt::literals;

    return fmt::format("cell {cr}x{cc} block {br}x{bc} stride {sr}x{sc}",
                       "cr"_a = g.cellSize.x(), "cc"_a = g.cellSize.y(),
                       "br"_a = g.blockSize.x(), "bc"_a = g.blockSize.y(),
                       "sr"_a = g.blockStride.x(), "sc"_a = g.blockStride.y());
}

// Reference features along with the vote mass of the summed-area table prefix
// each value is derived from.
struct ReferenceFeatures
{
    std::array<Eigen::DenseIndex, 5> dims{};
    std::vector<Reference> values;
    std::vector<Reference> scales;

    [[nodiscard]] std::size_t index(
        const std::array<Eigen::DenseIndex, 5>& idx) const noexcept
    {
        std::size_t result = 0;

        for (std::size_t n = 0; n < idx.size(); ++n) {
            result = result * static_cast<std::size_t>(dims[n]) +
                     static_cast<std::size_t>(idx[n]);
        }

        return result;
    }
};

// Region-relative block origins along an axis. Blocks advance by the stride
// while they fit and a final block is placed flush with the region border
// unless the previous one already ends there.
[[nodiscard]] std::vector<int> blockOrigins(int extent, int blockSize,
                                            int stride)
{
    const int last = std::max(extent - blockSize, 0);
    std::vector<int> result;

    for (int origin = 0; origin < last; origin += stride) {
        result.push_back(origin);
    }

    result.push_back(last);

    return result;
}

[[nodiscard]] ReferenceFeatures referenceFeatures(const Votes& votes,
                                                  const Geometry& g,
                                                  const hogpp::Bounds& roi,
                                                  const Reference& sigma)
{
    const Eigen::Array2i dims{roi.height, roi.width};
    const Eigen::Array2i offset{roi.y, roi.x};
    const Eigen::Array2i br = offset + dims;
    const std::vector<int> rowOrigins =
        blockOrigins(dims.x(), g.blockSize.x(), g.blockStride.x());
    const std::vector<int> colOrigins =
        blockOrigins(dims.y(), g.blockSize.y(), g.blockStride.y());
    const Eigen::Array2i numCells = g.blockSize / g.cellSize;

    ReferenceFeatures result;
    result.dims = {static_cast<Eigen::DenseIndex>(rowOrigins.size()),
                   static_cast<Eigen::DenseIndex>(colOrigins.size()),
                   numCells.x(), numCells.y(), votes.bins()};

    std::size_t size = 1;

    for (auto d : result.dims) {
        size *= static_cast<std::size_t>(d);
    }

    result.values.resize(size);
    result.scales.resize(size);

    std::vector<Reference> rowWeights;
    std::vector<Reference> colWeights;

    for (int u = 0; u < g.blockSize.x(); ++u) {
        rowWeights.push_back(referenceWeight(u, g.blockSize.x(), sigma));
    }

    for (int v = 0; v < g.blockSize.y(); ++v) {
        colWeights.push_back(referenceWeight(v, g.blockSize.y(), sigma));
    }

    for (int i = 0; i < static_cast<int>(rowOrigins.size()); ++i) {
        for (int j = 0; j < static_cast<int>(colOrigins.size()); ++j) {
            const Eigen::Array2i origin =
                offset +
                Eigen::Array2i{rowOrigins[static_cast<std::size_t>(i)],
                               colOrigins[static_cast<std::size_t>(j)]};

            for (int k = 0; k < numCells.x(); ++k) {
                for (int l = 0; l < numCells.y(); ++l) {
                    // Cells of a block larger than the region only cover the
                    // pixels inside of it.
                    const Eigen::Array2i first =
                        (origin + Eigen::Array2i{k, l} * g.cellSize).min(br);
                    const Eigen::Array2i last = (first + g.cellSize).min(br);

                    for (Eigen::DenseIndex b = 0; b < votes.bins(); ++b) {
                        const std::size_t n = result.index({i, j, k, l, b});

                        for (int x = first.x(); x < last.x(); ++x) {
                            const Reference& wr =
                                rowWeights[static_cast<std::size_t>(
                                    x - origin.x())];

                            for (int y = first.y(); y < last.y(); ++y) {
                                const Reference& wc =
                                    colWeights[static_cast<std::size_t>(
                                        y - origin.y())];

                                result.values[n] += wr * wc * votes(x, y, b);
                            }
                        }

                        result.scales[n] = votes.prefix(last.x(), last.y(), b);
                    }
                }
            }
        }
    }

    return result;
}

// Returns the maximum deviation from the reference in multiples of the machine
// epsilon relative to the corresponding vote mass.
template<class T>
[[nodiscard]] double maxDeviation(const Eigen::Tensor<T, 5>& actual,
                                  const ReferenceFeatures& expected)
{
    for (std::size_t n = 0; n < expected.dims.size(); ++n) {
        BOOST_TEST_REQUIRE(actual.dimension(n) == expected.dims[n]);
    }

    const Reference eps{std::numeric_limits<T>::epsilon()};
    Reference result{0};

    for (Eigen::DenseIndex i = 0; i < expected.dims[0]; ++i) {
        for (Eigen::DenseIndex j = 0; j < expected.dims[1]; ++j) {
            for (Eigen::DenseIndex k = 0; k < expected.dims[2]; ++k) {
                for (Eigen::DenseIndex l = 0; l < expected.dims[3]; ++l) {
                    for (Eigen::DenseIndex b = 0; b < expected.dims[4]; ++b) {
                        const std::size_t n = expected.index({i, j, k, l, b});
                        const T value = actual(i, j, k, l, b);

                        using std::abs;

                        // The inverted comparison also catches NaN.
                        if (!(abs(value) <= std::numeric_limits<T>::max())) {
                            return std::numeric_limits<double>::infinity();
                        }

                        const Reference error =
                            abs(Reference{value} - expected.values[n]);

                        if (error > 0) {
                            if (!(expected.scales[n] > 0)) {
                                return std::numeric_limits<double>::infinity();
                            }

                            result = std::max(
                                result, error / (eps * expected.scales[n]));
                        }
                    }
                }
            }
        }
    }

    return static_cast<double>(result);
}

template<class Descriptor>
void applyGeometry(Descriptor& d, const Geometry& g)
{
    d.setCellSize(g.cellSize);
    d.setBlockSize(g.blockSize);
    d.setBlockStride(g.blockStride);
}

[[nodiscard]] hogpp::Bounds fullRegion(int rows, int cols) noexcept
{
    return hogpp::Bounds{0, 0, cols, rows};
}

[[nodiscard]] bool isFastPath(const Geometry& g,
                              const hogpp::Bounds& roi) noexcept
{
    const Eigen::Array2i offset{roi.y, roi.x};
    return (g.blockStride >= g.cellSize).all() &&
           (offset - offset / g.blockStride * g.blockStride == 0).all();
}

template<class T>
[[nodiscard]] TestDescriptor<T> makeDescriptor(const Geometry& g, T sigma,
                                               Eigen::DenseIndex bins)
{
    TestDescriptor<T> d{TestMagnitude{}, TestBinning{}, IdentityNormalizer{},
                        hogpp::GaussianWeighting<T>{sigma}};
    applyGeometry(d, g);
    d.setNumBins(bins);
    return d;
}

// Stride equal to, larger than and smaller than the cell size, differing cell
// counts per axis and odd block sizes.
const std::array Geometries{
    Geometry{{4, 4}, {8, 8}, {4, 4}}, Geometry{{3, 4}, {6, 12}, {5, 6}},
    Geometry{{3, 3}, {9, 9}, {3, 3}}, Geometry{{2, 3}, {6, 3}, {2, 5}},
    Geometry{{4, 4}, {8, 8}, {2, 2}}, Geometry{{4, 2}, {8, 4}, {3, 2}},
    Geometry{{5, 3}, {5, 9}, {7, 3}},
};

constexpr std::array Sigmas{1.5, 4.0};

[[nodiscard]] std::vector<hogpp::Bounds> regionsFor(const Geometry& g)
{
    const hogpp::Bounds full = fullRegion(ImageRows, ImageCols);
    // Offsets that are multiples of the block stride
    const Eigen::Array2i aligned = g.blockStride * Eigen::Array2i{1, 2};
    // Offsets that are not multiples of the block stride unless the stride
    // is 1
    const Eigen::Array2i unaligned{1, 2};

    std::vector<hogpp::Bounds> result{full};

    for (const Eigen::Array2i& offset : {aligned, unaligned}) {
        const Eigen::Array2i size =
            Eigen::Array2i{ImageRows, ImageCols} - offset - 1;

        if ((size >= g.blockSize).all()) {
            result.push_back(
                hogpp::Bounds{offset.y(), offset.x(), size.y(), size.x()});
        }
    }

    return result;
}

} // namespace

BOOST_AUTO_TEST_CASE_TEMPLATE(gaussian_matches_reference, T, Scalars)
{
    const Sample<T> sample = makeSample<T>(ImageRows, ImageCols, DefaultSeed);
    const Votes votes = makeVotes(sample, DefaultNumBins, false);

    double maxFast = 0;
    double maxFallback = 0;

    for (const Geometry& g : Geometries) {
        for (double s : Sigmas) {
            const auto sigma = static_cast<T>(s);

            TestDescriptor<T> d = makeDescriptor(g, sigma, DefaultNumBins);
            d.compute(sample.dxs, sample.dys, nullptr);

            for (const hogpp::Bounds& roi : regionsFor(g)) {
                const double deviation = maxDeviation(
                    d.features(roi),
                    referenceFeatures(votes, g, roi, Reference{sigma}));

                BOOST_TEST_INFO("geometry " << describe(g) << " sigma " << s
                                            << " region " << roi.x << ","
                                            << roi.y);
                BOOST_TEST(deviation <= Tolerance);

                (isFastPath(g, roi) ? maxFast : maxFallback) = std::max(
                    isFastPath(g, roi) ? maxFast : maxFallback, deviation);
            }
        }
    }

    BOOST_TEST_MESSAGE("maximum deviation in eps: fast path "
                       << maxFast << ", fallback " << maxFallback);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(recompute_replaces_previous_image, T, Scalars)
{
    const Geometry g = Geometries.front();
    const auto sigma = static_cast<T>(Sigmas.front());

    const Sample<T> first = makeSample<T>(ImageRows, ImageCols, DefaultSeed);

    for (const Eigen::Array2i& size :
         {Eigen::Array2i{ImageRows, ImageCols},
          Eigen::Array2i{ImageRows - ShrinkRows, ImageCols - ShrinkCols}}) {
        const Sample<T> second = makeSample<T>(size.x(), size.y(), OtherSeed);

        TestDescriptor<T> d = makeDescriptor(g, sigma, DefaultNumBins);
        d.compute(first.dxs, first.dys, nullptr);
        (void)d.features();
        d.compute(second.dxs, second.dys, nullptr);

        const double deviation =
            maxDeviation(d.features(),
                         referenceFeatures(
                             makeVotes(second, DefaultNumBins, false), g,
                             fullRegion(size.x(), size.y()), Reference{sigma}));

        BOOST_TEST(deviation <= Tolerance);
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(set_histogram_replaces_previous_image, T, Scalars)
{
    const auto sigma = static_cast<T>(Sigmas.front());

    const Sample<T> first = makeSample<T>(ImageRows, ImageCols, DefaultSeed);
    const Sample<T> second = makeSample<T>(ImageRows, ImageCols, OtherSeed);
    const Votes votes = makeVotes(second, DefaultNumBins, false);

    for (const Geometry& g : Geometries) {
        TestDescriptor<T> source = makeDescriptor(g, sigma, DefaultNumBins);
        source.compute(second.dxs, second.dys, nullptr);

        TestDescriptor<T> d = makeDescriptor(g, sigma, DefaultNumBins);
        d.compute(first.dxs, first.dys, nullptr);
        (void)d.features();
        d.setHistogram(source.histogram());

        for (const hogpp::Bounds& roi : regionsFor(g)) {
            const double deviation = maxDeviation(
                d.features(roi),
                referenceFeatures(votes, g, roi, Reference{sigma}));

            BOOST_TEST_INFO("geometry " << describe(g) << " region " << roi.x
                                        << "," << roi.y);
            BOOST_TEST(deviation <= Tolerance);
            BOOST_TEST_MESSAGE("recovered votes, geometry "
                               << describe(g) << ", region " << roi.x << ","
                               << roi.y
                               << ", maximum deviation in eps: " << deviation);
        }
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(set_num_bins_after_compute, T, Scalars)
{
    constexpr Eigen::DenseIndex MoreBins = 9;

    const Geometry g = Geometries.front();
    const auto sigma = static_cast<T>(Sigmas.front());
    const Sample<T> sample = makeSample<T>(ImageRows, ImageCols, DefaultSeed);

    TestDescriptor<T> d = makeDescriptor(g, sigma, DefaultNumBins);
    d.compute(sample.dxs, sample.dys, nullptr);
    d.setNumBins(MoreBins);

    for (const hogpp::Bounds& roi : regionsFor(g)) {
        const Eigen::Tensor<T, 5> X = d.features(roi);

        BOOST_TEST(X.dimension(4) == DefaultNumBins);
        BOOST_TEST(maxDeviation(X, referenceFeatures(
                                       makeVotes(sample, DefaultNumBins, false),
                                       g, roi, Reference{sigma})) <= Tolerance);
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(geometry_change_after_compute, T, Scalars)
{
    const auto sigma = static_cast<T>(Sigmas.front());
    const Sample<T> sample = makeSample<T>(ImageRows, ImageCols, DefaultSeed);
    const Votes votes = makeVotes(sample, DefaultNumBins, false);

    for (const Geometry& before : Geometries) {
        for (const Geometry& after : Geometries) {
            TestDescriptor<T> d = makeDescriptor(before, sigma, DefaultNumBins);
            d.compute(sample.dxs, sample.dys, nullptr);
            applyGeometry(d, after);

            for (const hogpp::Bounds& roi : regionsFor(after)) {
                BOOST_TEST_INFO("geometry " << describe(before) << " -> "
                                            << describe(after));
                BOOST_TEST(maxDeviation(d.features(roi),
                                        referenceFeatures(votes, after, roi,
                                                          Reference{sigma})) <=
                           Tolerance);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sigma_change_after_compute, T, Scalars)
{
    const Sample<T> sample = makeSample<T>(ImageRows, ImageCols, DefaultSeed);
    const Votes votes = makeVotes(sample, DefaultNumBins, false);
    const auto before = static_cast<T>(Sigmas.front());
    const auto after = static_cast<T>(Sigmas.back());

    for (const Geometry& g : Geometries) {
        TestDescriptor<T> d = makeDescriptor(g, before, DefaultNumBins);
        d.compute(sample.dxs, sample.dys, nullptr);
        d.setSpatialWeighting(hogpp::GaussianWeighting<T>{after});

        BOOST_TEST(d.spatialWeighting().sigma() == after);

        for (const hogpp::Bounds& roi : regionsFor(g)) {
            BOOST_TEST_INFO("geometry " << describe(g));
            BOOST_TEST(maxDeviation(d.features(roi),
                                    referenceFeatures(votes, g, roi,
                                                      Reference{after})) <=
                       Tolerance);
        }

        // Restoring the original value must yield the original result.
        d.setSpatialWeighting(hogpp::GaussianWeighting<T>{before});

        BOOST_TEST(maxDeviation(d.features(),
                                referenceFeatures(
                                    votes, g, fullRegion(ImageRows, ImageCols),
                                    Reference{before})) <= Tolerance);
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(tiny_sigma_keeps_closest_pixels, T, Scalars)
{
    constexpr T sigma = std::numeric_limits<T>::denorm_min();

    const Sample<T> sample = makeSample<T>(ImageRows, ImageCols, DefaultSeed);
    const Votes votes = makeVotes(sample, DefaultNumBins, false);

    // Even and odd block sizes, and both the fast path and the fallback
    for (const Geometry& g :
         {Geometries[0], Geometries[2], Geometries[4], Geometries[6]}) {
        TestDescriptor<T> d = makeDescriptor(g, sigma, DefaultNumBins);
        d.compute(sample.dxs, sample.dys, nullptr);

        const Eigen::Tensor<T, 5> X = d.features();
        const Eigen::Tensor<T, 0> total = X.sum();

        BOOST_TEST_INFO("geometry " << describe(g));
        BOOST_TEST(total() > 0);
        BOOST_TEST(total() <= std::numeric_limits<T>::max());
        BOOST_TEST(
            maxDeviation(
                X, referenceFeatures(votes, g, fullRegion(ImageRows, ImageCols),
                                     Reference{sigma})) <= Tolerance);

        // Block normalization must not turn the sparse weights into NaN or
        // infinity.
        hogpp::IntegralHOGDescriptor<T, void, TestMagnitude, TestBinning,
                                     hogpp::L2Hys<T>,
                                     hogpp::GaussianWeighting<T>>
            normalized{TestMagnitude{}, TestBinning{}, hogpp::L2Hys<T>{},
                       hogpp::GaussianWeighting<T>{sigma}};
        applyGeometry(normalized, g);
        normalized.setNumBins(DefaultNumBins);
        normalized.compute(sample.dxs, sample.dys, nullptr);

        const Eigen::Tensor<T, 5> Y = normalized.features();
        // The inverted comparison also catches NaN.
        const Eigen::Tensor<bool, 0> finite =
            (Y.abs() <= std::numeric_limits<T>::max()).all();

        BOOST_TEST(finite());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(invalid_sigma_rejected, T, Scalars)
{
    // Disabling the weighting is a compile time choice of NoSpatialWeighting.
    // An enabled weighting therefore always requires an explicit sigma.
    static_assert(!std::is_default_constructible_v<hogpp::GaussianWeighting<T>>,
                  "GaussianWeighting must require an explicit sigma");

    for (T sigma : {T(0), T(-0.0), T(-1), -std::numeric_limits<T>::denorm_min(),
                    std::numeric_limits<T>::quiet_NaN(),
                    std::numeric_limits<T>::infinity(),
                    -std::numeric_limits<T>::infinity()}) {
        BOOST_CHECK_THROW((void)hogpp::GaussianWeighting<T>{sigma},
                          std::invalid_argument);

        hogpp::GaussianWeighting<T> w{T(1)};

        BOOST_CHECK_THROW(w.setSigma(sigma), std::invalid_argument);
        // A rejected value leaves the previous one intact.
        BOOST_TEST(w.sigma() == T(1));
    }

    for (T sigma : {std::numeric_limits<T>::denorm_min(),
                    std::numeric_limits<T>::max()}) {
        BOOST_CHECK_NO_THROW((void)hogpp::GaussianWeighting<T>{sigma});
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(largest_sigma_matches_reference, T, Scalars)
{
    const T sigma = std::numeric_limits<T>::max();
    const Sample<T> sample = makeSample<T>(ImageRows, ImageCols, DefaultSeed);
    const Votes votes = makeVotes(sample, DefaultNumBins, false);

    for (const Geometry& g : Geometries) {
        TestDescriptor<T> d = makeDescriptor(g, sigma, DefaultNumBins);
        d.compute(sample.dxs, sample.dys, nullptr);

        for (const hogpp::Bounds& roi : regionsFor(g)) {
            BOOST_TEST_INFO("geometry " << describe(g));
            BOOST_TEST(maxDeviation(d.features(roi),
                                    referenceFeatures(votes, g, roi,
                                                      Reference{sigma})) <=
                       Tolerance);
        }
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(disabled_matches_default, T, Scalars)
{
    static_assert(
        std::is_same_v<hogpp::IntegralHOGDescriptor<T>,
                       hogpp::IntegralHOGDescriptor<
                           T, hogpp::Gradient<T>, hogpp::GradientMagnitude<T>,
                           hogpp::UnsignedGradient<T>, hogpp::L2Hys<T>,
                           hogpp::NoSpatialWeighting>>,
        "IntegralHOGDescriptor must default to no spatial weighting");

    const Sample<T> sample = makeSample<T>(ImageRows, ImageCols, DefaultSeed);
    const Votes votes = makeVotes(sample, DefaultNumBins, false);

    for (const Geometry& g : Geometries) {
        TestDescriptor<T, hogpp::NoSpatialWeighting> unweighted{
            TestMagnitude{}, TestBinning{}, IdentityNormalizer{}};
        applyGeometry(unweighted, g);
        unweighted.setNumBins(DefaultNumBins);
        unweighted.compute(sample.dxs, sample.dys, nullptr);

        for (const hogpp::Bounds& roi : regionsFor(g)) {
            BOOST_TEST_INFO("geometry " << describe(g));
            // An infinite reference sigma weights all pixels equally.
            BOOST_TEST(
                maxDeviation(unweighted.features(roi),
                             referenceFeatures(
                                 votes, g, roi,
                                 std::numeric_limits<Reference>::infinity())) <=
                Tolerance);
        }
    }
}

namespace {

// Mirrors the members of a descriptor that has no notion of spatial
// weighting in the order of their declaration.
template<class T>
struct UnweightedBase
{
    hogpp::GradientMagnitude<T> vote;
    hogpp::UnsignedGradient<T> binning;
    hogpp::L2Hys<T> normalize;
};

template<class T>
struct UnweightedGradientBase : UnweightedBase<T>
{
    hogpp::Gradient<T> gradient;
};

template<class T>
struct UnweightedDescriptor : UnweightedGradientBase<T>
{
    hogpp::IntegralHistogram<T, 2, 1> histogram;
    Eigen::DenseIndex bins;
    Eigen::Array2i cellSize;
    Eigen::Array2i blockSize;
    Eigen::Array2i blockStride;
};

} // namespace

BOOST_AUTO_TEST_CASE_TEMPLATE(disabled_weighting_adds_no_storage, T, Scalars)
{
    static_assert(std::is_empty_v<hogpp::NoSpatialWeighting>,
                  "NoSpatialWeighting must be stateless");
    static_assert(
        std::is_empty_v<hogpp::NoSpatialWeighting::State<T>>,
        "NoSpatialWeighting must not keep any state alongside the histogram");

    // Members must never overlap, so the descriptor cannot be smaller than one
    // without spatial weighting. It is only as small if empty members can
    // occupy no storage.
#if HOGPP_HAS_NO_UNIQUE_ADDRESS
    BOOST_TEST(sizeof(hogpp::IntegralHOGDescriptor<T>) ==
               sizeof(UnweightedDescriptor<T>));
#else  // !HOGPP_HAS_NO_UNIQUE_ADDRESS
    BOOST_TEST(sizeof(hogpp::IntegralHOGDescriptor<T>) >=
               sizeof(UnweightedDescriptor<T>));
#endif // HOGPP_HAS_NO_UNIQUE_ADDRESS
}

BOOST_AUTO_TEST_CASE_TEMPLATE(masked_pixels_excluded, T, Scalars)
{
    const auto sigma = static_cast<T>(Sigmas.front());
    const Sample<T> sample = makeSample<T>(ImageRows, ImageCols, DefaultSeed);
    const Votes votes = makeVotes(sample, DefaultNumBins, true);

    for (const Geometry& g : Geometries) {
        TestDescriptor<T> d = makeDescriptor(g, sigma, DefaultNumBins);
        d.compute(sample.dxs, sample.dys, isMasked);

        for (const hogpp::Bounds& roi : regionsFor(g)) {
            BOOST_TEST_INFO("geometry " << describe(g));
            BOOST_TEST(maxDeviation(d.features(roi),
                                    referenceFeatures(votes, g, roi,
                                                      Reference{sigma})) <=
                       Tolerance);
        }
    }
}

// The fast path reads only the weighted table while the fallback sums the
// votes of the regular integral histogram. Pairing a weighted table built from
// nonzero votes with an all-zero integral histogram reveals which of the two
// was selected.
BOOST_AUTO_TEST_CASE_TEMPLATE(fast_path_selection, T, Scalars)
{
    const Geometry& g = Geometries.front();
    const hogpp::BlockGeometry geometry{g.cellSize, g.blockSize, g.blockStride};
    const Eigen::Array2i dims{ImageRows, ImageCols};
    const hogpp::GaussianWeighting<T> weighting{T(2)};

    auto voteOnce = [](const auto& /*ij*/, auto&& emit) {
        emit(Eigen::DenseIndex{0}, T(1));
    };

    Eigen::Tensor<T, 3> zeros(ImageRows + 1, ImageCols + 1, DefaultNumBins);
    zeros.setZero();

    hogpp::IntegralHistogram<T, 2, 1> histogram;
    histogram.setHistogram(zeros);

    auto extract = [&histogram](const hogpp::GaussianWeighting<T>& w,
                                const auto& state,
                                const hogpp::BlockGeometry& geom,
                                const Eigen::Array2i& offset) {
        auto query = w.cellQuery(state, histogram, geom);
        Eigen::Tensor<T, 1> cell(DefaultNumBins);
        const Eigen::Array2i last = offset + geom.cellSize;
        query(cell, Eigen::Array2i{0, 0}, offset, last);
        return cell(0);
    };

    typename hogpp::GaussianWeighting<T>::template State<T> state;
    weighting.prepare(state, geometry, dims, DefaultNumBins, voteOnce);

    // Offsets that are multiples of the block stride select the fast path.
    const Eigen::Array2i aligned = g.blockStride * Eigen::Array2i{1, 2};

    BOOST_TEST(extract(weighting, state, geometry, Eigen::Array2i{0, 0}) > 0);
    BOOST_TEST(extract(weighting, state, geometry, aligned) > 0);

    // An unaligned offset, a different sigma or a different geometry select
    // the fallback.
    BOOST_TEST(extract(weighting, state, geometry, Eigen::Array2i{1, 2}) == 0);
    BOOST_TEST(extract(hogpp::GaussianWeighting<T>{T(3)}, state, geometry,
                       Eigen::Array2i{0, 0}) == 0);

    hogpp::BlockGeometry other = geometry;
    other.blockStride = g.blockStride + 1;

    BOOST_TEST(extract(weighting, state, other, Eigen::Array2i{0, 0}) == 0);

    // A block stride smaller than the cell size always selects the fallback.
    hogpp::BlockGeometry overlapping = geometry;
    overlapping.blockStride = g.cellSize - 1;

    typename hogpp::GaussianWeighting<T>::template State<T> released;
    weighting.prepare(released, overlapping, dims, DefaultNumBins, voteOnce);

    BOOST_TEST(
        extract(weighting, released, overlapping, Eigen::Array2i{0, 0}) == 0);
}

// A prepare() that fails while visiting the votes must not leave behind a
// partially populated weighted table that a later query accepts as valid. The
// all-zero integral histogram reveals whether the query reads the table.
BOOST_AUTO_TEST_CASE_TEMPLATE(failed_prepare_discards_weighting, T, Scalars)
{
    const Geometry& g = Geometries.front();
    const hogpp::BlockGeometry geometry{g.cellSize, g.blockSize, g.blockStride};
    const Eigen::Array2i dims{ImageRows, ImageCols};
    const hogpp::GaussianWeighting<T> weighting{T(2)};

    auto voteOnce = [](const auto& /*ij*/, auto&& emit) {
        emit(Eigen::DenseIndex{0}, T(1));
    };

    // Votes every pixel except the last one, for which it fails.
    auto failAtLastPixel = [](const auto& ij, auto&& emit) {
        if (std::get<0>(ij) == ImageRows - 1 &&
            std::get<1>(ij) == ImageCols - 1) {
            throw std::runtime_error{"vote visitor failed"};
        }

        emit(Eigen::DenseIndex{0}, T(1));
    };

    Eigen::Tensor<T, 3> zeros(ImageRows + 1, ImageCols + 1, DefaultNumBins);
    zeros.setZero();

    hogpp::IntegralHistogram<T, 2, 1> histogram;
    histogram.setHistogram(zeros);

    typename hogpp::GaussianWeighting<T>::template State<T> state;
    weighting.prepare(state, geometry, dims, DefaultNumBins, voteOnce);

    BOOST_CHECK_THROW(weighting.prepare(state, geometry, dims, DefaultNumBins,
                                        failAtLastPixel),
                      std::runtime_error);

    auto query = weighting.cellQuery(state, histogram, geometry);
    Eigen::Tensor<T, 1> cell(DefaultNumBins);
    query(cell, Eigen::Array2i{0, 0}, Eigen::Array2i{0, 0}, g.cellSize);

    BOOST_TEST(cell(0) == 0);
}

// Restoring the weighted table along with the integral histogram reproduces
// the features exactly instead of recovering the votes from the latter.
BOOST_AUTO_TEST_CASE_TEMPLATE(restored_weighting_matches_exactly, T, Scalars)
{
    const auto sigma = static_cast<T>(Sigmas.front());
    const Sample<T> sample = makeSample<T>(ImageRows, ImageCols, DefaultSeed);

    for (const Geometry& g : Geometries) {
        TestDescriptor<T> source = makeDescriptor(g, sigma, DefaultNumBins);
        source.compute(sample.dxs, sample.dys, nullptr);

        TestDescriptor<T> d = makeDescriptor(g, sigma, DefaultNumBins);
        d.setHistogram(source.histogram(),
                       source.spatialWeightingState().weightedHistogram());

        BOOST_TEST_INFO("geometry " << describe(g));
        BOOST_TEST(d.spatialWeightingState().weightedHistogram().size() ==
                   source.spatialWeightingState().weightedHistogram().size());

        for (const hogpp::Bounds& roi : regionsFor(g)) {
            const Eigen::Tensor<T, 5> expected = source.features(roi);
            const Eigen::Tensor<T, 5> actual = d.features(roi);
            const Eigen::Tensor<bool, 0> equal = (actual == expected).all();

            BOOST_TEST(equal());
        }
    }
}

// A weighted table that does not match the integral histogram is rejected and
// leaves the descriptor unchanged.
BOOST_AUTO_TEST_CASE_TEMPLATE(mismatched_weighting_rejected, T, Scalars)
{
    const Geometry& g = Geometries.front();
    const auto sigma = static_cast<T>(Sigmas.front());
    const Sample<T> sample = makeSample<T>(ImageRows, ImageCols, DefaultSeed);

    TestDescriptor<T> source = makeDescriptor(g, sigma, DefaultNumBins);
    source.compute(sample.dxs, sample.dys, nullptr);

    const Eigen::Tensor<T, 3>& W =
        source.spatialWeightingState().weightedHistogram();

    BOOST_TEST_REQUIRE(W.size() > 0);

    // One channel short of the expected count
    const Eigen::Tensor<T, 3> truncated =
        W.slice(std::array<Eigen::DenseIndex, 3>{0, 0, 0},
                std::array<Eigen::DenseIndex, 3>{W.dimension(0), W.dimension(1),
                                                 W.dimension(2) - 1});

    TestDescriptor<T> d = makeDescriptor(g, sigma, DefaultNumBins);
    d.compute(sample.dxs, sample.dys, nullptr);
    const Eigen::Tensor<T, 5> before = d.features();

    BOOST_CHECK_THROW(d.setHistogram(source.histogram(), truncated),
                      std::invalid_argument);

    // A table is not expected if the stride is smaller than the cell size.
    Geometry overlapping = g;
    overlapping.blockStride = g.cellSize - 1;
    applyGeometry(d, overlapping);

    BOOST_CHECK_THROW(d.setHistogram(source.histogram(), W),
                      std::invalid_argument);

    applyGeometry(d, g);

    const Eigen::Tensor<bool, 0> unchanged = (d.features() == before).all();

    BOOST_TEST(unchanged());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(descriptor_prepares_weighting, T, Scalars)
{
    const Sample<T> sample = makeSample<T>(ImageRows, ImageCols, DefaultSeed);

    for (bool masked : {false, true}) {
        const Votes votes = makeVotes(sample, DefaultNumBins, masked);
        Reference expected{0};

        for (Eigen::DenseIndex b = 0; b < DefaultNumBins; ++b) {
            expected += votes.prefix(ImageRows, ImageCols, b);
        }

        const Reference eps{std::numeric_limits<T>::epsilon()};

        PrepareLog log;
        TestDescriptor<T, RecordingWeighting> d{TestMagnitude{}, TestBinning{},
                                                IdentityNormalizer{},
                                                RecordingWeighting{&log}};
        d.setNumBins(DefaultNumBins);

        if (masked) {
            d.compute(sample.dxs, sample.dys, isMasked);
        }
        else {
            d.compute(sample.dxs, sample.dys, nullptr);
        }

        // compute() passes the exact votes.
        BOOST_TEST(log.calls == 1);
        BOOST_TEST((log.dims == Eigen::Array2i{ImageRows, ImageCols}).all());
        BOOST_TEST(log.bins == DefaultNumBins);
        BOOST_TEST(static_cast<double>(abs(log.total - expected) /
                                       (eps * expected)) <= Tolerance);

        // setHistogram() passes the votes recovered from the integral
        // histogram.
        const Eigen::Tensor<T, 3> H = d.histogram();
        d.setHistogram(H);

        BOOST_TEST(log.calls == 2);
        BOOST_TEST((log.dims == Eigen::Array2i{ImageRows, ImageCols}).all());
        BOOST_TEST(log.bins == DefaultNumBins);
        BOOST_TEST(static_cast<double>(abs(log.total - expected) /
                                       (eps * expected)) <= Tolerance);
    }
}

// The separable sum must add the terms of each inner sum in ascending order,
// and the inner sums in ascending order, regardless of how many inner sums it
// accumulates at once. Terms spanning many orders of magnitude with both signs
// make every rounding depend on that order.
BOOST_AUTO_TEST_CASE_TEMPLATE(separable_sum_matches_sequential_sum, T, Scalars)
{
    constexpr std::array<int, 4> RowCounts{0, 1, 5, 17};
    constexpr std::array<int, 9> ColCounts{0, 1, 2, 3, 4, 5, 8, 9, 13};
    constexpr T MinExponent = -20;
    constexpr T MaxExponent = 20;

    std::default_random_engine g{DefaultSeed};
    std::uniform_real_distribution<T> exponent{MinExponent, MaxExponent};
    std::bernoulli_distribution negative;

    auto sample = [&g, &exponent, &negative] {
        using std::exp2;
        const T magnitude = exp2(exponent(g));
        return negative(g) ? -magnitude : magnitude;
    };

    for (const int rows : RowCounts) {
        for (const int cols : ColCounts) {
            std::vector<T> rowWeights(static_cast<std::size_t>(rows));
            std::vector<T> colWeights(static_cast<std::size_t>(cols));
            std::vector<T> values(static_cast<std::size_t>(rows * cols));

            std::generate(rowWeights.begin(), rowWeights.end(), sample);
            std::generate(colWeights.begin(), colWeights.end(), sample);
            std::generate(values.begin(), values.end(), sample);

            auto value = [&values, rows](int p, int q) {
                return values[static_cast<std::size_t>(q * rows + p)];
            };

            T expected{0};

            for (int q = 0; q < cols; ++q) {
                T row{0};

                for (int p = 0; p < rows; ++p) {
                    row = hogpp::multiplyAdd(
                        rowWeights[static_cast<std::size_t>(p)], value(p, q),
                        row);
                }

                expected = hogpp::multiplyAdd(
                    colWeights[static_cast<std::size_t>(q)], row, expected);
            }

            const T actual = hogpp::detail::separableSum<T>(
                rows, cols,
                [&rowWeights](int p) {
                    return rowWeights[static_cast<std::size_t>(p)];
                },
                [&colWeights](int q) {
                    return colWeights[static_cast<std::size_t>(q)];
                },
                value);

            BOOST_TEST_CONTEXT(rows << " rows and " << cols << " columns")
            {
                // Exact comparison, since both sums add the same terms in the
                // same order.
                BOOST_TEST(actual == expected);
            }
        }
    }
}

// The weighted cells must not depend on the memory layout of the integral
// histogram, down to the rounding. Entries spanning many orders of magnitude
// with both signs make every rounding depend on the summation order.
BOOST_AUTO_TEST_CASE_TEMPLATE(row_major_histogram_matches_column_major, T,
                              Scalars)
{
    constexpr Eigen::DenseIndex Bins = 9;
    constexpr T MinExponent = -20;
    constexpr T MaxExponent = 20;

    std::default_random_engine g{DefaultSeed};
    std::uniform_real_distribution<T> exponent{MinExponent, MaxExponent};
    std::bernoulli_distribution negative;

    Eigen::Tensor<T, 3> colMajor(ImageRows + 1, ImageCols + 1, Bins);
    Eigen::Tensor<T, 3, Eigen::RowMajor> rowMajor(ImageRows + 1, ImageCols + 1,
                                                  Bins);

    for (Eigen::DenseIndex b = 0; b < Bins; ++b) {
        for (Eigen::DenseIndex j = 0; j <= ImageCols; ++j) {
            for (Eigen::DenseIndex i = 0; i <= ImageRows; ++i) {
                using std::exp2;
                const T magnitude = exp2(exponent(g));
                colMajor(i, j, b) = negative(g) ? -magnitude : magnitude;
                rowMajor(i, j, b) = colMajor(i, j, b);
            }
        }
    }

    hogpp::IntegralHistogram<T, 2, 1> colMajorHistogram;
    colMajorHistogram.setHistogram(colMajor);
    hogpp::IntegralHistogram<T, 2, 1, Eigen::RowMajor> rowMajorHistogram;
    rowMajorHistogram.setHistogram(rowMajor);

    const hogpp::GaussianWeighting<T> weighting{T(2)};
    // Without prepared weights, every query sums the votes.
    const typename hogpp::GaussianWeighting<T>::template State<T> state;

    for (const Geometry& geom : Geometries) {
        const hogpp::BlockGeometry geometry{geom.cellSize, geom.blockSize,
                                            geom.blockStride};
        auto colMajorQuery =
            weighting.cellQuery(state, colMajorHistogram, geometry);
        auto rowMajorQuery =
            weighting.cellQuery(state, rowMajorHistogram, geometry);

        const Eigen::Array2i last = Eigen::Array2i{ImageRows, ImageCols} -
                                    geom.cellSize - Eigen::Array2i{1, 1};

        for (int x = 0; x <= last.x(); x += geom.cellSize.x()) {
            for (int y = 0; y <= last.y(); y += geom.cellSize.y()) {
                const Eigen::Array2i first{x, y};
                const Eigen::Array2i index{0, 0};
                const Eigen::Array2i end = first + geom.cellSize;

                Eigen::Tensor<T, 1> expected(Bins);
                Eigen::Tensor<T, 1> actual(Bins);

                colMajorQuery(expected, index, first, end);
                rowMajorQuery(actual, index, first, end);

                for (Eigen::DenseIndex b = 0; b < Bins; ++b) {
                    BOOST_TEST_CONTEXT("sum at " << x << ", " << y << " bin "
                                                 << b)
                    {
                        BOOST_TEST(actual(b) == expected(b));
                    }
                }

                colMajorQuery.interpolate(expected, index, first, end);
                rowMajorQuery.interpolate(actual, index, first, end);

                for (Eigen::DenseIndex b = 0; b < Bins; ++b) {
                    BOOST_TEST_CONTEXT("interpolation at " << x << ", " << y
                                                           << " bin " << b)
                    {
                        BOOST_TEST(actual(b) == expected(b));
                    }
                }
            }
        }
    }
}
