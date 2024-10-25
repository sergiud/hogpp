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

#include <optional>
#include <variant>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "binning.hpp"
#include "blocknormalizer.hpp"
#include "hogpp.hpp"
#include "integralhogdescriptor.hpp"
#include "magnitude.hpp"
#include "type_caster/bounds.hpp"

#if defined(HOGPP_GIL_DISABLED)
#    define HOGPP_MODULE(name, module, ...) \
        PYBIND11_MODULE(name, module, pybind11::mod_gil_not_used())
#else // !defined(HOGPP_GIL_DISABLED)
#    define HOGPP_MODULE PYBIND11_MODULE
#endif // defined(HOGPP_GIL_DISABLED)

HOGPP_MODULE(hogpp, m)
{
    namespace py = pybind11;

    py::options opts;
    opts.disable_function_signatures();

    py::class_<IntegralHOGDescriptor> cls{m, "IntegralHOGDescriptor"};

    // clang-format off
    cls.def
    (
          py::init
          <
              std::optional<Eigen::Array2i>
            , std::optional<Eigen::Array2i>
            , std::optional<Eigen::Array2i>
            , std::optional<pybind11::int_>
            , std::optional<MagnitudeType>
            , std::optional<BinningType>
            , std::optional<BlockNormalizerType>
            , std::optional<std::variant<pybind11::int_, pybind11::float_> >
            , std::optional<std::variant<pybind11::int_, pybind11::float_> >
          >()
        , py::kw_only()
        , py::arg("cell_size") = std::nullopt
        , py::arg("block_size") = std::nullopt
        , py::arg("block_stride") = std::nullopt
        , py::arg("n_bins") = std::nullopt
        , py::arg("magnitude") = std::nullopt
        , py::arg("binning") = std::nullopt
        , py::arg("block_norm") = std::nullopt
        , py::arg("clip_norm") = std::nullopt
        , py::arg("epsilon") = std::nullopt
    )
    .def
    (
          "compute"
        , py::overload_cast<const Rank2Or3Tensor&, const py::handle&>(&IntegralHOGDescriptor::compute)
        , py::arg("image")
        , py::pos_only() // 'image' can only be provided as positional argument
        , py::kw_only() // All following arguments are keyword-only
        , py::arg("mask") = py::none{}
    )
    .def
    (
          "compute"
        , py::overload_cast<const Rank2Or3TensorPair&, const py::handle&>(&IntegralHOGDescriptor::compute)
        , py::arg("dydx")
        , py::pos_only() // 'dxdy' can only be provided as positional argument
        , py::kw_only() // All following arguments are keyword-only
        , py::arg("mask") = py::none{}
    )
    .def
    (
          "__call__"
        , &IntegralHOGDescriptor::featuresROI
        , py::arg("roi")
    )
    .def
    (
          "__call__"
        , &IntegralHOGDescriptor::featuresROIs
        , py::arg("rois")
    )
    .def
    (
          "__bool__"
        , &IntegralHOGDescriptor::operator bool
    )
    .def
    (
          "__repr__"
        , &IntegralHOGDescriptor::repr
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
          "n_bins_"
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
    .def
    (
        py::pickle
        (
            [] (const IntegralHOGDescriptor& d)
            {
                return d.state();
            }
            , &IntegralHOGDescriptor::fromState
        )
    )
    .def
    (
        "__deepcopy__"
        , [] (const IntegralHOGDescriptor& d, py::object /*memo*/)
        {
            return IntegralHOGDescriptor{d};
        }
        , py::arg("memo")
    )
    ;
    // clang-format off
}
