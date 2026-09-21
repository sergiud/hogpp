# HOGpp - Fast histogram of oriented gradients computation using integral
# histograms
#
# Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from . import _hogpp
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from typing import Any
from typing import Callable
from typing import cast
from typing import Iterable
from typing import Literal
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Union

__all__ = ('IntegralHOGDescriptor',)

Binning = Literal['signed', 'unsigned']
Magnitude = Literal['identity', 'sqrt', 'square']
BlockNorm = Literal['l1', 'l1-sqrt', 'l1-hys', 'l2', 'l2-hys']
Size = Tuple[int, int]
Number = Union[int, float]
Mask = Union[Callable[[int, int], bool], ArrayLike]


class IntegralHOGDescriptor(_hogpp.IntegralHOGDescriptor):
    r"""Rectangular Histogram of Oriented Gradiens (R-HOG) feature descriptor
    :cite:`Dalal2005` implementend in terms of an integral histogram
    :cite:`porikli2005`. Employing an integral histogram allows to efficiently
    compute the feature descriptor in overlapping image regions, e.g., in
    sliding window object detection approaches.

    Computing feature descriptors involves two stages:

    1. The representation of a (possibly large) image is precomputed in an
       initial step using :meth:`IntegralHOGDescriptor.compute`.
    2. After the preprocessing step, feature descriptors of individual image
       subregions can be repeatedly extracted using a function call on an
       :class:`IntegralHOGDescriptor` instance, i.e., using
       :meth:`IntegralHOGDescriptor.__call__`.

    Note
    ----
    To ensure maximum performance when extracting features, do not compute
    the feature descriptor on individual images patches of a larger image.
    Instead, the initial computation should be performed on the original
    image first. After that, the feature descriptors of individual patches
    can be extracted much more efficiently than using the naive approach.

    Parameters
    ----------
    n_bins : int, optional
        Number of histogram bins. Default is 9.
    binning : str, optional
        Gradient orientation binning method. Default is 'unsigned'. Possible
        choices are:

        'unsigned'
            The orientation bins are evenly spaced over
            :math:`[0^\circ,180^\circ]` with the sign of the gradient
            ignored. Gradient orientations falling into quadrants of the
            Cartesian plane with negative orientation are mapped to their
            positive quadrant counterparts.

            Given an image gradient :math:`\vec g = (g_x,g_y)^\top =
            \left[\frac{\partial I}{\partial x}, \frac{\partial I}{\partial y}\right]^\top`,
            its orientation :math:`\alpha=\tan^{-1} \frac{g_y}{g_x} \in
            \left[-\frac{\pi}{2}, \frac{\pi}{2}\right)` within the first and
            fourth quadrants of the Cartesian plane is computed. Using the
            mapping :math:`\angle_u\colon \left[-\frac{\pi}{2},
            \frac{\pi}{2}\right) \to [0,\pi)` given by

            .. math::

                \angle_u(\alpha) \coloneqq \alpha+\frac{\pi}{2}

            negative angles are mapped to their corresponding positive
            counterparts in the second quadrant.
        'signed'
            The orientation bins are evenly spaced over
            :math:`[0^\circ,360^\circ]`, i.e., the sign of the gradient in
            the quadrants of the Cartesian plane are considered.

            Given an image gradient :math:`\vec g = (g_x,g_y)^\top =
            \left[\frac{\partial I}{\partial x}, \frac{\partial I}{\partial y}\right]^\top`,
            its orientation :math:`\alpha=\arctan_2 (g_y, g_x) \in [-\pi,\pi)`
            across the Cartesian plane is computed. The corresponding
            mapping :math:`\angle_s \colon [-\pi,\pi) \to [0,2\pi)` is then

            .. math::

                \angle_u(\alpha) \coloneqq \alpha+\pi
                \enspace .

    cell_size : tuple (2, ), optional
        The size of a single block cell in pixels. Default is (8, 8).
    block_size : tuple (2, ), optional
        The size of a single block in pixels. Default is (16, 16).
    block_stride : tuple (2, ), optional
        The shift amount between neighboring blocks in pixels. Default is
        (8, 8).
    magnitude : str, optional
        Function of the image gradient :math:`\vec g=(g_x,g_y)^\top` that
        computes the value voted into each orientation bin. Default is
        'identity'. Possible choices are:

        'identity'
            Computes the magnitude in terms of the gradient's
            :math:`\ell^2` norm, i.e., as :math:`\lVert\vec g\rVert_2`.
        'sqrt'
            Computes the square root of the magnitude, i.e.,
            :math:`\sqrt{\lVert\vec g\rVert_2}`.
        'square'
            Computes the magnitude in terms of a squared :math:`\ell^2`
            norm, i.e., as :math:`\lVert\vec g\rVert_2^2`.

    block_norm : str, optional
        Contrast normalization applied to individual blocks :math:`\vec v`.
        Default is 'l2-hys'. Possible choices are:

        'l1-sqrt'
            Computes the square root of the :math:`\ell^1` normalized block
            as

            .. math::

                \vec v \gets \sqrt{\frac{\vec v}{\lVert \vec v \rVert_1 + \epsilon}}
        'l1'
            Normalizes the blocks using the :math:`\ell^1` as

            .. math::

                \vec v \gets \frac{\vec v}{\lVert \vec v \rVert_1 + \epsilon}
        'l1-hys'
            Similar to `l1` normalization but additionally followed by
            clipping of values larger than `clip_norm`.
        'l2'
            Normalizes the blocks using the :math:`\ell^2` as

            .. math::

                \vec v \gets \sqrt{\frac{\vec v}{\lVert \vec v \rVert_2^2 + \epsilon^2}}
        'l2-hys'
            Similar to `l2` normalization but additionally followed by
            clipping of values larger than `clip_norm`.

    clip_norm : float, optional
        Maximum block norm. Applicable only to 'l1-hys' and 'l2-hys' block
        normalization. Default is 0.2.
    epsilon : float, optional
        The regularization amount. Default is 1e-12.
    """

    def __init__(
        self,
        *,
        n_bins: Optional[int] = None,
        binning: Optional[Binning] = None,
        cell_size: Optional[Size] = None,
        block_size: Optional[Size] = None,
        block_stride: Optional[Size] = None,
        magnitude: Optional[Magnitude] = None,
        block_norm: Optional[BlockNorm] = None,
        clip_norm: Optional[Number] = None,
        epsilon: Optional[Number] = None,
    ) -> None:
        try:
            super().__init__(
                cell_size=cell_size,
                block_size=block_size,
                block_stride=block_stride,
                n_bins=n_bins,
                magnitude=magnitude,
                binning=binning,
                block_norm=block_norm,
                clip_norm=clip_norm,
                epsilon=epsilon,
            )
        except BaseException:
            # The underlying C++ instance is never constructed when the
            # arguments do not match any constructor overload. Drop the
            # reference from this frame so a traceback formatter (e.g.,
            # pytest) does not call repr() on the uninitialized instance,
            # which would otherwise crash the interpreter.
            del self
            raise

    @overload
    def compute(self, image: ArrayLike, /, *, mask: Optional[Mask] = None) -> None: ...

    @overload
    def compute(
        self,
        dydx: Tuple[ArrayLike, ArrayLike],
        /,
        *,
        mask: Optional[Mask] = None,
    ) -> None: ...

    def compute(self, image_or_dydx, /, *, mask=None) -> None:
        """compute(image, /, *, mask=None)
        compute(dydx, /, *, mask=None)

        Computes the feature descriptor of the specified `image`, or,
        alternatively, of the specified pair `dydx` of image gradients along
        the vertical and horizontal axes.

        Parameters
        ----------
        image : array_like (m, n, (3, ))
            2-D or 3-D tensor representing the image whose feature
            descriptor shall be computed.
        dydx : tuple of array_like (m, n, (3, ))
            A 2-tuple of 2-D or 3-D tensors representing the vertical and
            horizontal image gradients, e.g., as returned by
            :func:`numpy.gradient`.
        mask : collections.abc.Callable, array_like (m, n, (3, ))
            A callable that indicates whether the pixel at the coordinate
            passed to the callable as a tuple is masked or not.
            Alternatively, the mask can be specified in terms of a tensor
            with the same rank and dimensions as the specified `image`.
        """
        super().compute(image_or_dydx, mask=mask)

    @overload
    def __call__(self, roi: ArrayLike, /) -> Optional[NDArray[Any]]: ...

    @overload
    def __call__(self, rois: Iterable[ArrayLike], /) -> Optional[NDArray[Any]]: ...

    def __call__(self, roi_or_rois: Any) -> Optional[NDArray[Any]]:
        """__call__(roi)
        __call__(rois)

        Extracts the features of the specified region of interest `roi`, or,
        alternatively, of the specified regions of interest `rois`.

        Parameters
        ----------
        roi : array_like (4, )
            An array specifying the top-left coordinate and the size of the
            image region whose feature descriptor will be exracted.
        rois : iterable of array_like (4, )
            An iterable of arrays each specifying the top-left coordinate
            and the size of an image region whose feature descriptor will be
            exracted.

        Returns
        -------
        numpy.ndarray
            A 5-D array whose first two dimensions represent the block, the
            following two dimensions the cell, and the final dimension
            represents the orientation bins. If `rois` was specified, an
            additional leading dimension enumerates the regions.

        Raises
        ------
        ValueError
            Thrown if `roi` describes a negative area, or, if `rois` was
            specified, if not all regions have the same dimensions.
        """
        return super().__call__(roi_or_rois)

    def __bool__(self) -> bool:
        """__bool__(self)

        Determines whether the descriptor was initialized in terms of a
        previous :meth:`compute` call.

        Returns
        -------
        bool
            Returns `True` if :py:meth:`compute` was previously called and
            the input was not empty, and `False` otherwise.
        """
        return super().__bool__()

    def __deepcopy__(self, memo: dict) -> 'IntegralHOGDescriptor':
        result = self.__class__.__new__(self.__class__)
        result.__setstate__(self.__getstate__())
        return result

    @property
    def features_(self) -> Optional[NDArray[Any]]:
        """numpy.ndarray or None: The features extracted from the entire
        image previously passed to :meth:`compute`, or `None` if
        :meth:`compute` was not called yet. See :meth:`__call__` for a
        description of the array layout.
        """
        return super().features_

    @property
    def cell_size_(self) -> Size:
        """tuple (2, ): The size of a single block cell in pixels."""
        return super().cell_size_

    @property
    def block_size_(self) -> Size:
        """tuple (2, ): The size of a single block in pixels."""
        return super().block_size_

    @property
    def block_stride_(self) -> Size:
        """tuple (2, ): The shift amount between neighboring blocks in
        pixels.
        """
        return super().block_stride_

    @property
    def n_bins_(self) -> int:
        """int: The number of histogram bins being used."""
        return super().n_bins_

    @property
    def histogram_(self) -> Optional[NDArray[Any]]:
        """numpy.ndarray or None: The raw integral histogram accumulated by
        the previous :meth:`compute` call, or `None` if :meth:`compute` was
        not called yet. This is the array returned and consumed when
        pickling a descriptor.
        """
        return super().histogram_

    @property
    def binning_(self) -> Binning:
        """str: Gradient orientation binning method."""
        return cast(Binning, super().binning_)

    @property
    def block_norm_(self) -> BlockNorm:
        """str: Contrast normalization applied to individual blocks."""
        return cast(BlockNorm, super().block_norm_)

    @property
    def magnitude_(self) -> Magnitude:
        """str: Magnitude function that determines the voted value."""
        return cast(Magnitude, super().magnitude_)

    @property
    def clip_norm_(self) -> Optional[float]:
        """float or None: Maximum block norm. Norm values above are clipped
        to the specified value. Applicable only to `l2-hys` block
        normalization.
        """
        return super().clip_norm_

    @property
    def epsilon_(self) -> float:
        """float: The regularization amount."""
        return super().epsilon_
