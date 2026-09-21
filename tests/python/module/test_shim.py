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

import copy
import inspect
import os
import pickle
import typing

if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
    path = os.getenv('HOGPPPATH')
    if path:
        os.add_dll_directory(path)

from hogpp import IntegralHOGDescriptor
import pytest


def test_descriptor_is_defined_in_python():
    # The docstrings and type hints live in a pure Python shim rather than
    # being embedded in the compiled extension, so editing them does not
    # require recompiling the C++ sources.
    assert IntegralHOGDescriptor.__module__ == 'hogpp'
    assert inspect.getsourcefile(IntegralHOGDescriptor) is not None
    assert inspect.getsourcefile(IntegralHOGDescriptor).endswith('.py')


@pytest.mark.parametrize(
    'member',
    [
        IntegralHOGDescriptor.__init__,
        IntegralHOGDescriptor.compute,
        IntegralHOGDescriptor.__call__,
    ],
)
def test_members_carry_type_hints(member):
    hints = typing.get_type_hints(member)

    assert hints


@pytest.mark.parametrize(
    'name',
    [
        'cell_size_',
        'block_size_',
        'block_stride_',
        'n_bins_',
        'binning_',
        'block_norm_',
        'magnitude_',
        'clip_norm_',
        'epsilon_',
        'features_',
        'histogram_',
    ],
)
def test_properties_have_docstrings(name):
    prop = inspect.getattr_static(IntegralHOGDescriptor, name)

    assert isinstance(prop, property)
    assert prop.__doc__


def test_deepcopy_preserves_the_public_type():
    desc = IntegralHOGDescriptor()
    desc1 = copy.deepcopy(desc)

    assert type(desc1) is IntegralHOGDescriptor


def test_pickle_roundtrip_preserves_the_public_type():
    desc = IntegralHOGDescriptor()
    desc1 = pickle.loads(pickle.dumps(desc))

    assert type(desc1) is IntegralHOGDescriptor
