//
// HOGpp - Fast histogram of oriented gradients computation using integral
// histograms
//
// Copyright 2020 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <variant>

#include <hogpp/integralhogdescriptor.hpp>

namespace {

template<class T>
struct BlockNormalization
{
};

struct IntegralHOGDescriptor
{
    hogpp::IntegralHOGDescriptor<double> descriptor_;
    //// clang-format off
    //boost::variant
    //<
    //      hogpp::IntegralHOGDescriptor<float, hogpp::L2Hys<float> >
    //    , hogpp::IntegralHOGDescriptor<double, hogpp::L2Hys<double> >
    //    , hogpp::IntegralHOGDescriptor<long double, hogpp::L2Hys<long double> >
    //>
    //// clang-format on
    //descriptor_;
};

} // namespace

PYBIND11_MODULE(hogpp, m)
{
    namespace py = pybind11;

    py::options opts;
    opts.disable_function_signatures();

    py::class_<IntegralHOGDescriptor> cls{m, "IntegralHOGDescriptor"};

    cls.def
    (
        py::init()
    )
    .def
    (
          "compute"
        , [] (IntegralHOGDescriptor& cls, py::buffer image)
        {
            auto info = image.request();

            if (info.format == py::format_descriptor<std::uint8_t>::format()) {
                if (info.ndim == 1) {
                    cv::Mat1b in{static_cast<int>(info.shape[0]),
                        static_cast<int>(info.shape[1]),
                        static_cast<std::uint8_t*>(info.ptr),
                        static_cast<std::size_t>(info.strides[0])};
                    cls.descriptor_.compute(&in, &in + 1);
                }
                else if (info.ndim == 3) {
                    std::vector<int> shape{info.shape.begin(), info.shape.end() - 1};
                    std::vector<std::size_t> steps{info.strides.begin(), info.strides.end() - 1};

                    cv::Mat3b in{static_cast<int>(info.ndim) - 1,
                        shape.data(),
                        static_cast<cv::Vec3b*>(info.ptr),
                        steps.data()};

                    //cv::imshow("foo", in);
                    //cv::waitKey();

                    std::vector<cv::Mat> channels;
                    cv::split(in, channels);

                    cls.descriptor_.compute(channels.begin(), channels.end());
                }
                else
                    throw std::invalid_argument("unsupported format");
            }
            else if (info.format == py::format_descriptor<double>::format()) {
                if (info.ndim == 2) {
                    cv::Mat1d in{static_cast<int>(info.shape[0]),
                        static_cast<int>(info.shape[1]),
                        static_cast<double*>(info.ptr),
                        static_cast<std::size_t>(info.strides[0])};
                    cls.descriptor_.compute(&in, &in + 1);
                }
                else
                    throw std::invalid_argument("unsupported format double");
            }
            else
                throw std::invalid_argument("unsupported format");

            //if (info.format == py::format_descriptor<float>::format()) {
            //    cls.descriptor_ = hogpp::IntegralHOGDescriptor<float>{};
            //}
            //else if (info.format == py::format_descriptor<double>::format()) {
            //    cls.descriptor_ = hogpp::IntegralHOGDescriptor<double>{};
            //}
            //else if (info.format == py::format_descriptor<long double>::format()) {
            //    cls.descriptor_ = hogpp::IntegralHOGDescriptor<long double>{};
            //}
            //else
            //    throw std::invalid_argument("unsupported image type");
        }
        , py::arg("image")
    )
    .def
    (
        "features"
        , [] (IntegralHOGDescriptor& cls, const std::tuple<int, int, int, int>& roi)
        {
            cv::Rect rect;
            std::tie(rect.x, rect.y, rect.width, rect.height) = roi;

            auto X = cls.descriptor_.features(rect);
            //auto XX = X.swap_layout().shuffle(std::array<int, 5>{4, 3, 2, 1, 0});

            using Tensor5 = Eigen::Tensor<double, 5>;
            auto* p = new Tensor5{std::move(X)};

            py::capsule cleanup{p, [] (void* p)
            {
                delete static_cast<Tensor5*>(p);
            }};

            const Eigen::DenseIndex n = sizeof(double);
            const Eigen::DenseIndex n0 = n;
            const Eigen::DenseIndex n1 = p->dimension(0) * n0;
            const Eigen::DenseIndex n2 = p->dimension(1) * n1;
            const Eigen::DenseIndex n3 = p->dimension(2) * n2;
            const Eigen::DenseIndex n4 = p->dimension(3) * n3;

            return py::array_t<double, py::array::f_style | py::array::forcecast>{p->dimensions(), {n0, n1, n2, n3, n4}, p->data(), cleanup};
        }
    )
    .def_property
    (
        "block_stride"
        , [] (IntegralHOGDescriptor& cls)
        {
            const auto& value = cls.descriptor_.blockStride();
            return std::make_tuple(value.x(), value.y());
        }
        , [] (IntegralHOGDescriptor& cls, const std::tuple<int, int>& value)
        {
            cls.descriptor_.setBlockStride(Eigen::Array2i{std::get<0>(value), std::get<1>(value)});
        }
    )
    .def_property
    (
        "block_size"
        , [] (IntegralHOGDescriptor& cls)
        {
            const auto& value = cls.descriptor_.blockSize();
            return std::make_tuple(value.x(), value.y());
        }
        , [] (IntegralHOGDescriptor& cls, const std::tuple<int, int>& value)
        {
            cls.descriptor_.setBlockSize(Eigen::Array2i{std::get<0>(value), std::get<1>(value)});
        }
    )
    .def_property_readonly
    (
        "block_size"
        , [] (IntegralHOGDescriptor& cls)
        {
            const auto& value = cls.descriptor_.blockSize();
            return std::make_tuple(value.x(), value.y());
        }
    )
    .def_property
    (
        "num_bins"
        , [] (IntegralHOGDescriptor& cls)
        {
            return cls.descriptor_.numBins();
        }
        , [] (IntegralHOGDescriptor& cls, Eigen::DenseIndex value)
        {
            cls.descriptor_.setNumBins(value);
        }
    )
    ;
}
