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

// FIXME GCC 14.x workaround for https://github.com/pybind/pybind11/pull/5208
#include <algorithm>
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

#if defined(HOGPP_SKBUILD)
#    define HOGPP_MODULE_NAME _hogpp
#else // !defined(HOGPP_SKBUILD)
#    define HOGPP_MODULE_NAME hogpp
#endif // defined(HOGPP_SKBUILD)

HOGPP_MODULE(HOGPP_MODULE_NAME, m)
{
    namespace py = pybind11;

    py::options opts;
    opts.disable_function_signatures();

    py::class_<IntegralHOGDescriptor> cls{m, "IntegralHOGDescriptor"};
    cls.doc() = R"|(
IntegralHOGDescriptor(*, n_bins=9, binning='unsigned', cell_size=(8, 8), block_size=(16, 16), block_stride=(8, 8), magnitude='identity', block_norm='l2-hys', clip_norm=0.2, epsilon=1e-12)

Rectangular Histogram of Oriented Gradiens (R-HOG) feature descriptor
:cite:`Dalal2005` implementend in terms of an integral histogram
:cite:`porikli2005`. Employing an integral histogram allows to efficiently
compute the feature descriptor in overlapping image regions, e.g., in sliding
window object detection approaches.

Computing feature descriptors involves two stages:

1. The representation of a (possibly large) image is precomputed in an initial
   step using :meth:`IntegralHOGDescriptor.compute`.
2. After the preprocessing step, feature descriptors of individual image
   subregions can be repeatedly extracted using a function call on an
   :class:`IntegralHOGDescriptor` instance, i.e., using
   :meth:`IntegralHOGDescriptor.__call__`.

Note
----
To ensure maximum performance when extracting features, do not compute the
feature descriptor on individual images patches of a larger image. Instead, the
initial computation should be performed on the original image first. After that,
the feature descriptors of individual patches can be extracted much more
efficiently than using the naive approach.

Parameters
----------
n_bins : int, optional
    Number of histogram bins.
binning : str, optional
    Gradient orientation binning method. Possible choices are:

    'unsigned'
        The orientation bins are evenly spaced over :math:`[0^\circ,180^\circ]`
        with the sign of the gradient ignored. Gradient orientations falling into
        quadrants of the Cartesian plane with negative orientation are mapped to
        their positive quadrant counterparts.

        Given an image gradient :math:`\vec g = (g_x,g_y)^\top =
        \left[\frac{\partial I}{\partial x}, \frac{\partial I}{\partial y}\right]^\top`,
        its orientation :math:`\alpha=\tan^{-1} \frac{g_y}{g_x} \in
        \left[-\frac{\pi}{2}, \frac{\pi}{2}\right)` within the first and fourth
        quadrants of the Cartesian plane is computed. Using the mapping
        :math:`\angle_u\colon \left[-\frac{\pi}{2}, \frac{\pi}{2}\right) \to [0,\pi)`
        given by

        .. math::

            \angle_u(\alpha) \coloneqq \alpha+\frac{\pi}{2}

        negative angles are mapped to their corresponding positive counterparts
        in the second quadrant.
    'signed'
        The orientation bins are evenly spaced over :math:`[0^\circ,360^\circ]`,
        i.e., the sign of the gradient in the quadrants of the Cartesian plane are
        considered.

        Given an image gradient :math:`\vec g = (g_x,g_y)^\top =
        \left[\frac{\partial I}{\partial x}, \frac{\partial I}{\partial y}\right]^\top`,
        its orientation :math:`\alpha=\arctan_2 (g_y, g_x) \in [-\pi,\pi)`
        across the Cartesian plane is computed. The corresponding mapping
        :math:`\angle_s \colon [-\pi,\pi) \to [0,2\pi)` is then

        .. math::

            \angle_u(\alpha) \coloneqq \alpha+\pi
            \enspace .

cell_size : tuple (2, ), optional
    The size of a single block cell in pixels.
block_size : tuple (2, ), optional
    The size of a single block in pixels.
block_stride : tuple (2, ), optional
    The shift amount between neighboring blocks in pixels.
magnitude : str, optional
    Function of the image gradient :math:`\vec g=(g_x,g_y)^\top` that computes
    the value voted into each orientation bin. Possible choices are:

    'identity'
        Computes the magnitude in terms of the gradient's :math:`\ell^2` norm,
        i.e., as :math:`\lVert\vec g\rVert_2`.
    'sqrt'
        Computes the square root of the magnitude, i.e., :math:`\sqrt{\lVert\vec g\rVert_2}`.
    'square'
        Computes the magnitude in terms of a squared :math:`\ell^2` norm, i.e.,
        as :math:`\lVert\vec g\rVert_2^2`.

block_norm : str, optional
    Contrast normalization applied to individual blocks :math:`\vec v`. Possible
    choices are:

    'l1-sqrt'
        Computes the square root of the :math:`\ell^1` normalized block as

        .. math::

            \vec v \gets \sqrt{\frac{\vec v}{\lVert \vec v \rVert_1 + \epsilon}}
    'l1'
        Normalizes the blocks using the :math:`\ell^1` as

        .. math::

            \vec v \gets \frac{\vec v}{\lVert \vec v \rVert_1 + \epsilon}
    'l1-hys'
        Similar to `l1` normalization but additionally followed by clipping of
        values larger than `clip_norm`.
    'l2'
        Normalizes the blocks using the :math:`\ell^2` as

        .. math::

            \vec v \gets \sqrt{\frac{\vec v}{\lVert \vec v \rVert_2^2 + \epsilon^2}}
    'l2-hys'
        Similar to `l2` normalization but additionally followed by clipping of
        values larger than `clip_norm`.

clip_norm : float, optional
    Maximum block norm. Applicable only to 'l1-hys' and 'l2-hys' block
    normalization.
epsilon : float, optional
    The regularization amount.


Attributes
----------
num_bins_ : int
    The number of histogram bins being used.
binning_ : str
    Gradient orientation binning method.
cell_size_ : tuple (2, )
    The size of a single block cell in pixels.
block_size_ : tuple (2, )
    The size of a single block in pixels.
block_stride_ : tuple (2, )
    The shift amount between neighboring blocks in pixels.
magnitude_ : str
    Magnitude function that determines the voted value.
block_norm_ : str
    Contrast normalization applied to individual blocks.
clip_norm_ : float
    Maximum block norm. Norm values above are clipped to the specified value.
    Applicable only to `l2-hys` block normalization.
epsilon_ : float
    The regularization amount.
)|";

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
        ,
R"|(
compute(image, /, *, mask=None)

Computes the feature descriptor of the specified `image`.

Parameters
----------
image : array_like (m, n, (3, ))
    2-D or 3-D tensor representing the image whose feature descriptor shall be
    computed.
mask : collections.abc.Callable, array_like (m, n, (3, ))
    A callable that indicates whether the pixel at the coordinate passed to the
    callable as a tuple is masked or not. Alternatively, the mask can be specified
    in terms of a tensor with the same rank and dimensions as the specified `image`.
)|"
        , py::arg("image")
        // LCOV_EXCL_START
        , py::pos_only() // 'image' can only be provided as positional argument
        , py::kw_only() // All following arguments are keyword-only
        // LCOV_EXCL_STOP
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
        ,
R"|(
__call__(self, roi)

Extracts the features of the specified region of interest `roi`.

Parameters
----------
roi : array_like (4, )
    An array specifying the top-left coordinate and the size of the image region
    whose feature descriptor will be exracted.


Returns
-------
numpy.ndarray
    A 5-D array whose first two dimensions represent the block, the following
    two dimensions the cell, and the final dimension represents the
    orientation bins.

Raises
------
ValueError
    Thrown if `roi` describes a negative area.
)|"
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
        ,
R"|(
__bool__(self)

Determines whether the descriptor was initialized in terms of a previous :meth:`compute` call.

Returns
-------
bool
    Returns `True` if :py:meth:`compute` was previously called and the input was
    not empty, and `False` otherwise.
)|"
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
