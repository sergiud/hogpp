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

#include <new>
#include <variant>

#include "type_caster/array2i.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>

#include "binning.hpp"
#include "blocknormalizer.hpp"
#include "hogpp.hpp"
#include "integralhogdescriptor.hpp"
#include "magnitude.hpp"
#include "type_caster/bounds.hpp"

#if defined(HOGPP_SKBUILD)
#    define HOGPP_MODULE_NAME _hogpp
#else // !defined(HOGPP_SKBUILD)
#    define HOGPP_MODULE_NAME hogpp
#endif // defined(HOGPP_SKBUILD)

NB_MODULE(HOGPP_MODULE_NAME, m)
{
    namespace py = nanobind;
    py::class_<IntegralHOGDescriptor> cls{m, "IntegralHOGDescriptor"};

    // clang-format off
    cls.def
    (
          py::init
          <
              std::optional<Eigen::Array2i>
            , std::optional<Eigen::Array2i>
            , std::optional<Eigen::Array2i>
            , std::optional<nanobind::int_>
            , std::optional<MagnitudeType>
            , std::optional<BinningType>
            , std::optional<BlockNormalizerType>
            , std::optional<std::variant<nanobind::int_, nanobind::float_> >
            , std::optional<std::variant<nanobind::int_, nanobind::float_> >
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
        // 'image' is left unnamed so it can only be passed positionally.
        , py::arg()
        , py::kw_only() // All following arguments are keyword-only
        , py::arg("mask") = py::none()
    )
    .def
    (
          "compute"
        , py::overload_cast<const Rank2Or3TensorPair&, const py::handle&>(&IntegralHOGDescriptor::compute)
        // 'dydx' is left unnamed so it can only be passed positionally.
        , py::arg()
        , py::kw_only() // All following arguments are keyword-only
        , py::arg("mask") = py::none()
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
    .def_prop_ro
    (
          "features_"
        , &IntegralHOGDescriptor::features
    )
    .def_prop_ro
    (
          "cell_size_"
        , &IntegralHOGDescriptor::cellSize
    )
    .def_prop_ro
    (
          "block_stride_"
        , &IntegralHOGDescriptor::blockStride
    )
    .def_prop_ro
    (
          "block_size_"
        , &IntegralHOGDescriptor::blockSize
    )
    .def_prop_ro
    (
          "n_bins_"
        , &IntegralHOGDescriptor::numBins
    )
    .def_prop_ro
    (
          "histogram_"
        , &IntegralHOGDescriptor::histogram
    )
    .def_prop_ro
    (
          "binning_"
        , &IntegralHOGDescriptor::binning
    )
    .def_prop_ro
    (
          "block_norm_"
        , &IntegralHOGDescriptor::blockNormalizer
    )
    .def_prop_ro
    (
          "magnitude_"
        , &IntegralHOGDescriptor::magnitude
    )
    .def_prop_ro
    (
          "clip_norm_"
        , &IntegralHOGDescriptor::clipNorm
    )
    .def_prop_ro
    (
          "epsilon_"
        , &IntegralHOGDescriptor::epsilon
    )
    .def
    (
          "__getstate__"
        , &IntegralHOGDescriptor::state
    )
    .def
    (
          "__setstate__"
        , [] (IntegralHOGDescriptor& self, const IntegralHOGDescriptor::State& state)
        {
            new (&self) IntegralHOGDescriptor{IntegralHOGDescriptor::fromState(state)};
        }
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
