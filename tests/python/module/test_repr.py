# HOGpp - Fast histogram of oriented gradients computation using integral
# histograms
#
# Copyright 2024 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

import numpy as np  # noqa: F401
import os

if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
    path = os.getenv('HOGPPPATH')
    if path:
        os.add_dll_directory(path)

from hogpp import IntegralHOGDescriptor
import pytest

all_args = [
    dict(),
    dict(block_norm='l1-hys', clip_norm=1),
    dict(block_norm='l2-hys', clip_norm=1e-4),
    dict(binning='signed', clip_norm=1e-2),
    dict(magnitude='identity', binning='unsigned', epsilon=1e-3),
    dict(cell_size=(2, 2), n_bins=7),
    dict(block_size=(1, 1)),
    dict(block_size=(1, 2), block_stride=(2, 3)),
]


@pytest.mark.parametrize('args', all_args)
def test_repr(args):
    desc = IntegralHOGDescriptor(**args)
    text = repr(desc)
    # Make sure we can exactly reconstruct the descriptor from its textual
    # representation.
    desc2 = eval(text, None, {})
    text2 = repr(desc2)

    assert text == text2
