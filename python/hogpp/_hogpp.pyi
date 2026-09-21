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

from numpy.typing import ArrayLike
from numpy.typing import NDArray
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Union

Binning = Literal['signed', 'unsigned']
Magnitude = Literal['identity', 'sqrt', 'square']
BlockNorm = Literal['l1', 'l1-sqrt', 'l1-hys', 'l2', 'l2-hys']
Size = Tuple[int, int]
Number = Union[int, float]
Mask = Union[Callable[[int, int], bool], ArrayLike]

class IntegralHOGDescriptor:
    def __init__(
        self,
        *,
        cell_size: Optional[Size] = ...,
        block_size: Optional[Size] = ...,
        block_stride: Optional[Size] = ...,
        n_bins: Optional[int] = ...,
        magnitude: Optional[Magnitude] = ...,
        binning: Optional[Binning] = ...,
        block_norm: Optional[BlockNorm] = ...,
        clip_norm: Optional[Number] = ...,
        epsilon: Optional[Number] = ...,
    ) -> None: ...
    @overload
    def compute(self, image: ArrayLike, /, *, mask: Optional[Mask] = ...) -> None: ...
    @overload
    def compute(
        self,
        dydx: Tuple[ArrayLike, ArrayLike],
        /,
        *,
        mask: Optional[Mask] = ...,
    ) -> None: ...
    @overload
    def __call__(self, roi: ArrayLike, /) -> Optional[NDArray[Any]]: ...
    @overload
    def __call__(self, rois: Iterable[ArrayLike], /) -> Optional[NDArray[Any]]: ...
    def __bool__(self) -> bool: ...
    def __repr__(self) -> str: ...
    def __deepcopy__(self, memo: Any) -> 'IntegralHOGDescriptor': ...
    def __reduce__(self) -> Tuple[Any, ...]: ...
    @property
    def features_(self) -> Optional[NDArray[Any]]: ...
    @property
    def cell_size_(self) -> Size: ...
    @property
    def block_size_(self) -> Size: ...
    @property
    def block_stride_(self) -> Size: ...
    @property
    def n_bins_(self) -> int: ...
    @property
    def histogram_(self) -> Optional[NDArray[Any]]: ...
    @property
    def binning_(self) -> Binning: ...
    @property
    def block_norm_(self) -> BlockNorm: ...
    @property
    def magnitude_(self) -> Magnitude: ...
    @property
    def clip_norm_(self) -> Optional[float]: ...
    @property
    def epsilon_(self) -> float: ...
