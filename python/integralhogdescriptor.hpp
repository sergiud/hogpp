//
// HOGpp - Fast histogram of oriented gradients computation using integral
// histograms
//
// Copyright 2021 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

#ifndef PYTHON_HOGPP_INTEGRALHOGDESCRIPTOR_HPP
#define PYTHON_HOGPP_INTEGRALHOGDESCRIPTOR_HPP

#include <Eigen/Core>

#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>

#include <hogpp/integralhogdescriptor.hpp>

#include "binning.hpp"
#include "blocknormalizer.hpp"
#include "magnitude.hpp"

class IntegralHOGDescriptor
{
public:
    [[nodiscard]] explicit IntegralHOGDescriptor(
        const std::optional<Eigen::Array2i>& cellSize,
        const std::optional<Eigen::Array2i>& blockSize,
        const std::optional<Eigen::Array2i>& blockStride,
        const std::optional<pybind11::int_>& numBins,
        const std::optional<MagnitudeType>& magnitude,
        const std::optional<BinningType>& binning,
        const std::optional<BlockNormalizerType>& blockNorm,
        const std::optional<std::variant<pybind11::int_, pybind11::float_> >&
            clipNorm,
        const std::optional<std::variant<pybind11::int_, pybind11::float_> >&
            epsilon);

    void compute(const cv::Mat& image, const pybind11::handle& mask);

    [[nodiscard]] pybind11::object features() const;
    [[nodiscard]] pybind11::object featuresROI(const cv::Rect& rect) const;
    [[nodiscard]] std::tuple<int, int> cellSize() const;
    [[nodiscard]] std::tuple<int, int> blockSize() const;
    [[nodiscard]] std::tuple<int, int> blockStride() const;
    [[nodiscard]] Eigen::DenseIndex numBins() const;
    [[nodiscard]] pybind11::object histogram() const;
    [[nodiscard]] BinningType binning() const;
    [[nodiscard]] BlockNormalizerType blockNormalizer() const;
    [[nodiscard]] MagnitudeType magnitude() const;
    [[nodiscard]] pybind11::object clipNorm() const noexcept;
    [[nodiscard]] pybind11::object epsilon() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    template<class T>
    using Descriptor = hogpp::IntegralHOGDescriptor<T, Magnitude<T>, Binning<T>,
                                                    BlockNormalizer<T> >;

    [[nodiscard]] bool isEmpty() const noexcept;
    void update();

    std::optional<Eigen::Array2i> cellSize_;
    std::optional<Eigen::Array2i> blockSize_;
    std::optional<Eigen::Array2i> blockStride_;
    std::optional<int> numBins_;
    std::optional<MagnitudeType> magnitudeType_;
    std::optional<BinningType> binningType_;
    std::optional<BlockNormalizerType> blockNormalizerType_;

    // clang-format off
    std::variant
    <
          Descriptor<float>
        , Descriptor<double>
        //, Descriptor<long double>
    >
    descriptor_;
    // clang-format on

    std::optional<std::variant<pybind11::int_, pybind11::float_> > clipNorm_;
    std::optional<std::variant<pybind11::int_, pybind11::float_> > epsilon_;
};

#endif // PYTHON_HOGPP_INTEGRALHOGDESCRIPTOR_HPP
