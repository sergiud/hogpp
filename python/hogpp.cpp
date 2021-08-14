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

#include <functional>

#include <pybind11/pybind11.h>

#include "integralhogdescriptor.hpp"
#include "type_caster/opencv.hpp"

PYBIND11_MODULE(hogpp, m)
{
    namespace py = pybind11;

    py::options opts;
    opts.disable_function_signatures();

    py::class_<IntegralHOGDescriptor> cls{m, "IntegralHOGDescriptor"};

    // clang-format off
    cls.def
    (
        py::init<py::kwargs>()
    )
    .def
    (
          "compute"
        , &IntegralHOGDescriptor::compute
        , py::arg("image")
    )
    .def
    (
          "__call__"
        , &IntegralHOGDescriptor::featuresROI
        , py::arg("roi")
    )
    .def
    (
          "__bool__"
        , std::not1(std::mem_fn(&IntegralHOGDescriptor::isEmpty))
    )
    .def_property_readonly
    (
          "features_"
        , &IntegralHOGDescriptor::features
    )
    .def_property_readonly
    (
          "cell_size_"
        , &IntegralHOGDescriptor::cellSize
    )
    .def_property_readonly
    (
          "block_stride_"
        , &IntegralHOGDescriptor::blockStride
    )
    .def_property_readonly
    (
          "block_size_"
        , &IntegralHOGDescriptor::blockSize
    )
    .def_property_readonly
    (
          "num_bins_"
        , &IntegralHOGDescriptor::numBins
    )
    .def_property_readonly
    (
          "histogram_"
        , &IntegralHOGDescriptor::histogram
    )
    .def_property_readonly
    (
          "binning_"
        , &IntegralHOGDescriptor::binning
    )
    .def_property_readonly
    (
          "block_norm_"
        , &IntegralHOGDescriptor::blockNormalizer
    )
    .def_property_readonly
    (
          "magnitude_"
        , &IntegralHOGDescriptor::magnitude
    )
    .def_property_readonly
    (
          "clip_norm_"
        , &IntegralHOGDescriptor::clipNorm
    )
    .def_property_readonly
    (
          "epsilon_"
        , &IntegralHOGDescriptor::epsilon
    )
    ;
    // clang-format off
}
