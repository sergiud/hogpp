//
// HOGpp - Fast histogram of oriented gradients computation using integral
// histograms
//
// Copyright 2024 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#ifdef HAVE_EXECUTION
#    include <execution>
#endif // HAVE_EXECUTION

#include <fmt/format.h>

#include "type_caster/array2i.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>

#include "formatter.hpp"
#include "integralhogdescriptor.hpp"
#include "type_caster/bounds.hpp"
#if defined(HAVE_OPENCV)
#    include "type_caster/opencv.hpp"
#endif // defined(HAVE_OPENCV)
#include "hogpp.hpp"
#include "type_caster/tensor.hpp"

namespace {

template<class... Elements, std::size_t... Indices>
[[nodiscard]] constexpr decltype(auto) tuple_head(
    const std::tuple<Elements...>& t, std::index_sequence<Indices...>)
{
    return std::forward_as_tuple(std::get<Indices>(t)...);
}

template<std::size_t N, class... Elements>
[[nodiscard]] constexpr decltype(auto) tuple_head(
    const std::tuple<Elements...>& t)
{
    return tuple_head(t, std::make_index_sequence<N>{});
}

template<class T>
struct NativeScalar : NativeScalar<std::decay_t<T>>
{
};

template<>
struct NativeScalar<nanobind::float_>
{
    using type = double;
};

template<>
struct NativeScalar<nanobind::int_>
{
    using type = int;
};

template<class T>
using NativeScalar_t = typename NativeScalar<T>::type;

#if defined(HAVE_OPENCV)
template<class Function>
[[maybe_unused]] constexpr void matToTensorRank2(
    [[maybe_unused]] const cv::Mat& image, [[maybe_unused]] Function&& func,
    TypeSequence<> /*unused*/)
{
}

template<class Function, class Arg, class... Args>
[[maybe_unused]] void matToTensorRank2(const cv::Mat& image, Function&& func,
                                       TypeSequence<Arg, Args...> /*unused*/)
{
    if (image.depth() != cv::DataDepth<Arg>::value) {
        matToTensorRank2(image, std::forward<Function>(func),
                         TypeSequence<Args...>{});
    }
    else {
        Eigen::TensorMap<const Eigen::Tensor<Arg, 2, Eigen::RowMajor>> t{
            image.ptr<Arg>(), image.rows, image.cols};
        func(t.reshape(std::array<long, 3>{{image.rows, image.cols, 1}})
                 .swap_layout()
                 .shuffle(std::array<int, 3>{{2, 1, 0}}));
    }
}

template<class Function>
[[maybe_unused]] constexpr void matToTensorRank3(
    [[maybe_unused]] const cv::Mat& image, [[maybe_unused]] Function&& func,
    TypeSequence<> /*unused*/)
{
}

template<class Function, class Arg, class... Args>
[[maybe_unused]] void matToTensorRank3(const cv::Mat& image, Function&& func,
                                       TypeSequence<Arg, Args...> /*unused*/)
{
    if (image.depth() != cv::DataDepth<Arg>::value) {
        matToTensorRank3(image, std::forward<Function>(func),
                         TypeSequence<Args...>{});
    }
    else {
        Eigen::TensorMap<const Eigen::Tensor<Arg, 3, Eigen::RowMajor>> t{
            image.ptr<Arg>(), 1, 1, image.rows * image.cols * image.channels()};
        func(t.reshape(std::array<int, 3>{
                           {image.rows, image.cols, image.channels()}})
                 .swap_layout()
                 .shuffle(std::array<int, 3>{{2, 1, 0}}));
    }
}

#endif // defined(HAVE_OPENCV)

template<class Function>
void objectToTensor( // LCOV_EXCL_LINE
    [[maybe_unused]] const nanobind::object& image,
    [[maybe_unused]] std::size_t ndim,
    [[maybe_unused]] const nanobind::dlpack::dtype& dt,
    [[maybe_unused]] Function&& func, TypeSequence<> /*unused*/)
{
}

template<class Arg>
[[nodiscard]] Eigen::Tensor<Arg, 3> extend(const nanobind::object& image,
                                           std::size_t ndim)
{
    using Tensor2 = Eigen::Tensor<Arg, 2>;
    using Tensor3 = Eigen::Tensor<Arg, 3>;

    if (ndim == 2) {
        auto tmp = nanobind::cast<Tensor2>(image);
        // Extend rank-2 arrays by a third dimension.
        return tmp.reshape(std::array{tmp.dimension(0), tmp.dimension(1),
                                      typename Tensor2::Index{1}});
    }

    return nanobind::cast<Tensor3>(image);
}

template<class Function, class Arg, class... Args>
void objectToTensor(const nanobind::object& image, std::size_t ndim,
                    const nanobind::dlpack::dtype& dt, Function&& func,
                    TypeSequence<Arg, Args...> /*unused*/)
{
    if (nanobind::dtype<Arg>() != dt) {
        objectToTensor(image, ndim, dt, std::forward<Function>(func),
                       TypeSequence<Args...>{});
    }
    else {
        func(extend<Arg>(image, ndim));
    }
}

template<long... Ranks, class Function, class... Args>
void objectToTensor(const RankNTensorPair<Ranks...>& p, std::size_t ndim,
                    const nanobind::dlpack::dtype& dt, Function&& func,
                    TypeSequence<Args...> types)
{
    auto convert = [&p, &func, ndim](const auto& t) {
        using T = typename std::decay_t<decltype(t)>::Scalar;
        func(t, extend<T>(p.buf1.buf, ndim));
    };

    // Assume [dys, dxs] ordering
    objectToTensor(p.buf2.buf, ndim, dt, convert, types);
}

[[nodiscard]] nanobind::ndarray<nanobind::ro> probe(const nanobind::object& buf)
{
    nanobind::ndarray<nanobind::ro> a;

    // The buffer was already validated by RankNTensor's own caster when the
    // argument was bound, so this cannot fail in practice.
    nanobind::try_cast(buf, a);

    return a;
}

template<class T>
[[nodiscard]] nanobind::object asTuple(const std::optional<T>& value)
{
    if (value) {
        nanobind::object tmp = nanobind::cast(value);
        return nanobind::tuple(tmp);
    }

    return nanobind::none();
}

} // namespace

IntegralHOGDescriptor::IntegralHOGDescriptor(
    const std::optional<Eigen::Array2i>& cellSize,
    const std::optional<Eigen::Array2i>& blockSize,
    const std::optional<Eigen::Array2i>& blockStride,
    const std::optional<nanobind::int_>& numBins,
    const std::optional<MagnitudeType>& magnitude,
    const std::optional<BinningType>& binning,
    const std::optional<BlockNormalizerType>& blockNorm,
    const std::optional<std::variant<nanobind::int_, nanobind::float_>>&
        clipNorm,
    const std::optional<std::variant<nanobind::int_, nanobind::float_>>&
        epsilon)
    : cellSize_{cellSize}
    , blockSize_{blockSize}
    , blockStride_{blockStride}
    , numBins_{numBins}
    , magnitudeType_{magnitude}
    , binningType_{binning}
    , blockNormalizerType_{blockNorm}
    , clipNorm_{clipNorm}
    , epsilon_{epsilon}
{
    // Delay setting parameters until we know the floating point type we
    // will be working with.

    if (clipNorm) {
        // Allow None
        std::visit(
            [](const auto& value) {
                if (!value.is_none()) {
                    try {
                        if (!(value >
                              std::decay_t<decltype(value)>{
                                  NativeScalar_t<decltype(value)>{0}})) {
                            throw nanobind::cast_error{};
                        }
                    }
                    catch (const nanobind::cast_error&) {
                        throw std::invalid_argument{fmt::format(
                            "IntegralHOGDescriptor clip_norm can only "
                            "be a positive floating point value but {} was "
                            "given",
                            value)};
                    }
                }
            },
            *clipNorm);
    }

    if (epsilon) {
        // Allow None
        std::visit(
            [](const auto& value) {
                if (!value.is_none()) {
                    try {
                        if (value < std::decay_t<decltype(value)>{
                                        NativeScalar_t<decltype(value)>{0}}) {
                            throw nanobind::cast_error{};
                        }
                    }
                    catch (const nanobind::cast_error&) {
                        throw std::invalid_argument{fmt::format(
                            "IntegralHOGDescriptor epsilon can be either 0 or "
                            "a positive floating point value but {} was "
                            "given",
                            value)};
                    }
                }
            },
            *epsilon);
    }

    if (!blockNormalizerType_ && (clipNorm_ || epsilon_)) {
        // In case block normalizer arguments were provided, we must ensure
        // these are actually forwarded to the corresponding instance. We
        // assume the default to be L2-Hys.
        blockNormalizerType_ = BlockNormalizerType::L2Hys;
    }

    update();
}

template<class... Args>
[[nodiscard]] DescriptorVariant makeDescriptor(
    [[maybe_unused]] const nanobind::dlpack::dtype& dt,
    TypeSequence<> /*unused*/,
    Args&&... args) noexcept(std::is_nothrow_constructible_v<Descriptor<double>,
                                                             Args...>)
{
    // Default descriptor instance if floating point was not matched against
    // supported types.
    return Descriptor<double>(std::forward<Args>(args)...);
}

template<class Scalar, class... Scalars, class... Args>
[[nodiscard]] DescriptorVariant makeDescriptor(
    const nanobind::dlpack::dtype& dt,
    TypeSequence<Scalar, Scalars...> /*unused*/, Args&&... args)
{
    if (nanobind::dtype<Scalar>() == dt) {
        return Descriptor<Scalar>(std::forward<Args>(args)...);
    }

    return makeDescriptor(dt, TypeSequence<Scalars...>{},
                          std::forward<Args>(args)...);
}

void IntegralHOGDescriptor::compute(const Rank2Or3Tensor& t,
                                    const nanobind::handle& mask)
{
    // TODO Maybe support dtype parameter?

    const nanobind::ndarray<nanobind::ro> image = probe(t.buf);

    update(image.dtype());

    std::visit(
        [&t, &image, &mask](auto& descriptor) {
            if (mask.is_none()) {
                auto convert = [&descriptor](const auto& t) {
                    descriptor.compute(t);
                };

                objectToTensor(t.buf, image.ndim(), image.dtype(), convert,
                               SupportedTypes{});
            }
            else {
                nanobind::object getitem;

                if (nanobind::hasattr(mask, "__getitem__")) {
                    // mask can be indexed, e.g., it's a numpy.ndarray
                    getitem = nanobind::getattr(mask, "__getitem__");
                }
                else if (nanobind::hasattr(mask, "__call__")) {
                    // mask is a callable
                    getitem = nanobind::borrow<nanobind::object>(mask);
                }
                else {
                    throw std::invalid_argument{fmt::format(
                        FMT_STRING(
                            "IntegralHOGDescriptor.compute mask must be "
                            "either a callable or provide an indexer in terms "
                            "of a __getitem__ method that accepts a 2-tuple, "
                            "e.g., a numpy.ndarray instance, but a {} object "
                            "was given"),
                        mask.type())};
                }

                auto masking = [&getitem](Eigen::DenseIndex i,
                                          Eigen::DenseIndex j) {
                    return nanobind::bool_(getitem(nanobind::make_tuple(i, j)));
                };

                auto convert = [&descriptor, &masking](const auto& t) {
                    descriptor.compute(t, masking);
                };

                objectToTensor(t.buf, image.ndim(), image.dtype(), convert,
                               SupportedTypes{});
            }
        },
        descriptor_);
}

void IntegralHOGDescriptor::compute(const Rank2Or3TensorPair& dydx,
                                    const nanobind::handle& mask)
{
    const nanobind::ndarray<nanobind::ro> image = probe(dydx.buf1.buf);

    update(image.dtype());

    std::visit(
        [&dydx, &image, &mask](auto& descriptor) {
            using Scalar = typename std::decay_t<decltype(descriptor)>::Scalar;

            if (mask.is_none()) {
                auto convert = [&descriptor](const auto& dx, const auto& dy) {
                    Eigen::Tensor<Scalar, 3> dxs = dx.template cast<Scalar>();
                    Eigen::Tensor<Scalar, 3> dys = dy.template cast<Scalar>();

                    descriptor.compute(dxs, dys, nullptr);
                };

                objectToTensor(dydx, image.ndim(), image.dtype(), convert,
                               SupportedTypes{});
            }
            else {
                nanobind::object getitem;

                if (nanobind::hasattr(mask, "__getitem__")) {
                    // mask can be indexed, e.g., it's a numpy.ndarray
                    getitem = nanobind::getattr(mask, "__getitem__");
                }
                else if (nanobind::hasattr(mask, "__call__")) {
                    // mask is a callable
                    getitem = nanobind::borrow<nanobind::object>(mask);
                }
                else {
                    throw std::invalid_argument{fmt::format(
                        FMT_STRING(
                            "IntegralHOGDescriptor.compute mask must be "
                            "either a callable or provide an indexer in terms "
                            "of a __getitem__ method that accepts a 2-tuple, "
                            "e.g., a numpy.ndarray instance, but a {} object "
                            "was given"),
                        mask.type())};
                }

                auto masking = [&getitem](Eigen::DenseIndex i,
                                          Eigen::DenseIndex j) {
                    return nanobind::bool_(getitem(nanobind::make_tuple(i, j)));
                };

                auto convert = [&descriptor, &masking](const auto& dx,
                                                       const auto& dy) {
                    Eigen::Tensor<Scalar, 3> dxs = dx.template cast<Scalar>();
                    Eigen::Tensor<Scalar, 3> dys = dy.template cast<Scalar>();

                    descriptor.compute(dxs, dys, masking);
                };

                objectToTensor(dydx, image.ndim(), image.dtype(), convert,
                               SupportedTypes{});
            }
        },
        descriptor_);
}

nanobind::object IntegralHOGDescriptor::features() const
{
    return isEmpty() ? nanobind::none()
                     : std::visit(
                           [](auto& descriptor) {
                               return nanobind::cast(
                                   descriptor.features(),
                                   nanobind::rv_policy::reference_internal);
                           },
                           descriptor_);
}

nanobind::object IntegralHOGDescriptor::featuresROI(
    const hogpp::Bounds& rect) const
{
    if (isEmpty()) {
        return nanobind::none();
    }

    return std::visit(
        [&rect](auto& descriptor) -> nanobind::object {
            using Tensor = std::decay_t<decltype(descriptor.features(rect))>;

            if (rect.area() == 0) {
                // A default-constructed dynamic-size Eigen::Tensor has all
                // dimensions set to 0, i.e., it is an empty tensor of the
                // correct rank.
                return nanobind::cast(Tensor{});
            }

            return nanobind::cast(descriptor.features(rect),
                                  nanobind::rv_policy::move);
        },
        descriptor_);
}

nanobind::object IntegralHOGDescriptor::featuresROIs(
    const nanobind::iterable& rects) const
{
    auto extract = [&rects](auto& descriptor) {
        using Scalar = typename std::decay_t<decltype(descriptor)>::Scalar;
        using Tensor = std::decay_t<decltype(descriptor.features(
            std::declval<hogpp::Bounds>()))>;
        constexpr auto NumDimensions = Tensor::NumDimensions;

        const std::size_t n = nanobind::len(rects);
        Eigen::Tensor<Scalar, NumDimensions + 1> features;

        // Greedily convert bounds to be able to release the GIL for
        // multithreaded processing
        std::vector<hogpp::Bounds> bounds;
        bounds.resize(n);

        std::transform(rects.begin(), rects.end(), bounds.begin(),
                       [](const nanobind::handle& rect) {
                           return nanobind::cast<hogpp::Bounds>(rect);
                       });

        hogpp::Bounds firstBounds;

        // Check for compatible bounds upfront. We cannot perform the check in
        // the for_each lambda because throwing an exception from the lambda
        // when using the parallel version of for_each causes segmentation
        // faults.
        if (!bounds.empty()) {
            firstBounds = bounds.front();

            auto pos = std::find_if_not(std::next(bounds.begin()), bounds.end(),
                                        [&bounds](const hogpp::Bounds& value) {
                                            return bounds.front().size() ==
                                                   value.size();
                                        });

            if (pos != bounds.end()) {
                auto index = std::distance(bounds.begin(), pos);

                throw nanobind::value_error(
                    fmt::format(
                        "IntegralHOGDescriptor extraction of features from "
                        "multiple regions requires all bounds to be of the "
                        "same dimensions. however, the bounds at index 0 are "
                        "different from those at index {} ({} vs. {})",
                        index, nanobind::cast(bounds.front()),
                        nanobind::cast(*pos))
                        .c_str());
            }
        }

        // Unfortunately, we also need to greedily evaluate the indexed sequence
        // because std::for_each will not run in parallel with a forward
        // iterator.
        std::vector<std::pair<std::size_t, hogpp::Bounds>> idxs;
        idxs.reserve(bounds.size());

        // TODO Replace by C++20 ranges
        for (std::size_t i = 0; i != bounds.size(); ++i) {
            idxs.emplace_back(i, bounds[i]);
        }

        auto first = idxs.cbegin();

        if (n > 0) {
            // Allocate memory once the bounds of the first element are known.
            // We cannot perform the allocation within the for_each lambda
            // because it will be possibly executed multiple times at the same
            // time causing in a race condition. Therefore, we process the first
            // element independently from the remaining ones.
            auto X = descriptor.features(firstBounds);
            // Store the dimensions of a single tensor to ensure the bounds
            // produce compatible tensors
            std::array<Eigen::DenseIndex,
                       static_cast<std::size_t>(NumDimensions) + 1>
                dims;
            dims.front() = static_cast<Eigen::DenseIndex>(n);
            std::copy(X.dimensions().begin(), X.dimensions().end(),
                      std::next(dims.begin()));

            features.resize(dims);
            features.template chip<0>(
                static_cast<Eigen::DenseIndex>(first->first)) = X;

            ++first;
        }

        auto assign = [&features, &descriptor](auto value) {
            auto X = descriptor.features(value.second);

            features.template chip<0>(
                static_cast<Eigen::DenseIndex>(value.first)) = X;
        };

        {
#if !defined(NB_FREE_THREADED)
            nanobind::gil_scoped_release release;
#endif // !defined(NB_FREE_THREADED)

            // Process the remaining bounds
            std::for_each(
#ifdef HAVE_EXECUTION
                std::execution::par,
#endif // HAVE_EXECUTION
                first, idxs.cend(), assign);
        }

        return nanobind::cast(std::move(features));
    };

    return isEmpty() ? nanobind::none() : std::visit(extract, descriptor_);
}

std::tuple<int, int> IntegralHOGDescriptor::cellSize() const
{
    return std::visit(
        [](auto& descriptor) {
            const auto& value = descriptor.cellSize();
            return std::make_tuple(value.x(), value.y());
        },
        descriptor_);
}

std::tuple<int, int> IntegralHOGDescriptor::blockSize() const
{
    return std::visit(
        [](auto& descriptor) {
            const auto& value = descriptor.blockSize();
            return std::make_tuple(value.x(), value.y());
        },
        descriptor_);
}

std::tuple<int, int> IntegralHOGDescriptor::blockStride() const
{
    return std::visit(
        [](auto& descriptor) {
            const auto& value = descriptor.blockStride();
            return std::make_tuple(value.x(), value.y());
        },
        descriptor_);
}

Eigen::DenseIndex IntegralHOGDescriptor::numBins() const
{
    return std::visit([](auto& descriptor) { return descriptor.numBins(); },
                      descriptor_);
}

nanobind::object IntegralHOGDescriptor::histogram() const
{
    return isEmpty() ? nanobind::none()
                     : std::visit(
                           [](auto& descriptor) {
                               return nanobind::cast(descriptor.histogram());
                           },
                           descriptor_);
}

void IntegralHOGDescriptor::update()
{
    std::visit(
        [this](auto& descriptor) {
            if (cellSize_) {
                descriptor.setCellSize(*cellSize_);
            }

            if (blockSize_) {
                descriptor.setBlockSize(*blockSize_);
            }

            if (blockStride_) {
                descriptor.setBlockStride(*blockStride_);
            }

            if (numBins_) {
                descriptor.setNumBins(
                    static_cast<Eigen::DenseIndex>(*numBins_));
            }

            using Descriptor = std::decay_t<decltype(descriptor)>;

            if (magnitudeType_) {
                typename Descriptor::Magnitude magnitude{*magnitudeType_};
                descriptor.setMagnitude(std::move(magnitude));
            }

            if (binningType_) {
                typename Descriptor::Binning binning{*binningType_};
                descriptor.setBinning(std::move(binning));
            }

            if (blockNormalizerType_) {
                auto optionalNumber =
                    [](const auto& value) -> std::optional<nanobind::float_> {
                    // nanobind::float_'s caster (unlike pybind11::float_'s)
                    // requires an exact Python float and does not implicitly
                    // convert an int, and int_'s own conversion operator
                    // disables numeric conversion. nanobind::cast<double>
                    // enables it, correctly handling both int_ and float_.
                    return nanobind::float_{nanobind::cast<double>(value)};
                };

                std::optional<nanobind::float_> clipNorm =
                    !clipNorm_ ? std::nullopt
                               : std::visit(optionalNumber, *clipNorm_);
                std::optional<nanobind::float_> epsilon =
                    !epsilon_ ? std::nullopt
                              : std::visit(optionalNumber, *epsilon_);

                typename Descriptor::BlockNormalizer blockNormalizer{
                    *blockNormalizerType_, clipNorm, epsilon};
                descriptor.setBlockNormalizer(std::move(blockNormalizer));
            }
        },
        descriptor_);
}

void IntegralHOGDescriptor::update(const nanobind::dlpack::dtype& dt)
{
    descriptor_ = makeDescriptor(dt, PrecisionTypes{});
    update();
}

BinningType IntegralHOGDescriptor::binning() const
{
    return std::visit(
        [](const auto& descriptor) { return descriptor.binning().type(); },
        descriptor_);
}

BlockNormalizerType IntegralHOGDescriptor::blockNormalizer() const
{
    return std::visit(
        [](const auto& descriptor) {
            return descriptor.blockNormalizer().type();
        },
        descriptor_);
}

MagnitudeType IntegralHOGDescriptor::magnitude() const
{
    return std::visit(
        [](const auto& descriptor) { return descriptor.magnitude().type(); },
        descriptor_);
}

bool IntegralHOGDescriptor::isEmpty() const noexcept
{
    return std::visit(
        [](const auto& descriptor) { return descriptor.isEmpty(); },
        descriptor_);
}

nanobind::object IntegralHOGDescriptor::clipNorm() const noexcept
{
    nanobind::object result = std::visit(
        [](const auto& descriptor) {
            return descriptor.blockNormalizer().clip();
        },
        descriptor_);

    return !result.is_none() ? result
           : clipNorm_       ? std::visit(
                                   [](const auto& value) {
                                 return nanobind::cast<nanobind::object>(value);
                                   },
                                   *clipNorm_)
                             : nanobind::none();
}

nanobind::object IntegralHOGDescriptor::epsilon() const noexcept
{
    // Every block normalizer provides regularization; no additional checks
    // required unless a different block normalizer is implemented.
    return std::visit(
        [](const auto& descriptor) {
            return descriptor.blockNormalizer().epsilon();
        },
        descriptor_);
}

IntegralHOGDescriptor::operator bool() const noexcept
{
    return !isEmpty();
}

IntegralHOGDescriptor::State IntegralHOGDescriptor::state() const
{
    // clang-format off
    return std::make_tuple
    (
          cellSize_
        , blockSize_
        , blockStride_
        , numBins_
        , magnitudeType_
        , binningType_
        , blockNormalizerType_
        , clipNorm_
        , epsilon_
        , histogram()
    )
    ;
    // clang-format on
}

IntegralHOGDescriptor IntegralHOGDescriptor::fromState(
    const IntegralHOGDescriptor::State& value)
{
    constexpr auto N = std::tuple_size_v<State>;
    constexpr auto N_minus_one = N - 1;

    auto result = std::make_from_tuple<IntegralHOGDescriptor>(
        tuple_head<N_minus_one>(value));

    nanobind::object histogram = std::get<N_minus_one>(value);

    if (!histogram.is_none()) {
        nanobind::ndarray<nanobind::ro> a;

        if (!nanobind::try_cast(histogram, a)) {
            throw std::invalid_argument{
                "IntegralHOGDescriptor histogram state must be a "
                "numpy.ndarray-like buffer"};
        }

        // Histogram floating point type defines descriptor's working precision
        result.update(a.dtype());

        std::visit(
            [&histogram](auto& descriptor) {
                using Tensor = std::decay_t<decltype(descriptor.histogram())>;
                descriptor.setHistogram(nanobind::cast<Tensor>(histogram));
            },
            result.descriptor_);
    }

    return result;
}

std::string IntegralHOGDescriptor::repr() const
{
    auto val = [](const auto& v) -> nanobind::object { return v; };

    nanobind::object clipNorm;
    nanobind::object epsilon;

    if (clipNorm_) {
        clipNorm = std::visit(val, *clipNorm_);
    }

    if (epsilon_) {
        epsilon = std::visit(val, *epsilon_);
    }

    constexpr std::size_t NumCtorArgs = 9;
    const std::array<std::pair<std::string_view, nanobind::object>, NumCtorArgs>
        args{{
            {"cell_size", asTuple(cellSize_)},
            {"block_size", asTuple(blockSize_)},
            {"block_stride", asTuple(blockStride_)},
            {"n_bins", nanobind::cast(numBins_)},
            {"magnitude", nanobind::cast(magnitudeType_)},
            {"binning", nanobind::cast(binningType_)},
            {"block_norm", nanobind::cast(blockNormalizerType_)},
            {"clip_norm", nanobind::cast(clipNorm_)},
            {"epsilon", nanobind::cast(epsilon_)},
        }};

    std::vector<std::string> argvals;
    argvals.reserve(args.size());

    for (auto [name, value] : args) {
        if (!value.is_none()) {
            argvals.push_back(fmt::format("{}={}", name, value));
        }
    }

    return fmt::format("IntegralHOGDescriptor({})", fmt::join(argvals, ", "));
}
