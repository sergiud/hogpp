# HOGpp - Fast histogram of oriented gradients computation using integral
# histograms
#
# Copyright 2021 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

import os

if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
    path = os.getenv('HOGPPPATH')
    if path:
        os.add_dll_directory(path)

from itertools import repeat
import numpy as np
import pytest
import type_caster_test.opencv as ttt

unsupported_dtypes = [np.float16, np.int64, np.uint64, np.object_]
supported_dtypes = [
    np.uint8,
    np.int8,
    np.int16,
    np.uint16,
    np.int32,
    np.float32,
    np.float64,
]

try:
    unsupported_dtypes += [np.float128]
except AttributeError:
    pass


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('size0', list(range(2, 10, 2)) + list(range(1, 10, 2)))
def test_rank_1_tensor(dtype, size0):
    a = np.random.rand(size0).astype(dtype)
    np.testing.assert_array_equal(a, ttt.pass_(a).ravel())


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('size0', list(range(2, 10, 2)) + list(range(1, 10, 2)))
@pytest.mark.parametrize('size1', list(range(2, 10, 2)) + list(range(1, 10, 2)))
def test_rank_2_tensor(dtype, size0, size1):
    a = np.random.rand(size0, size1).astype(dtype)
    np.testing.assert_array_equal(a, ttt.pass_(a))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('size0', list(range(2, 10, 2)) + list(range(1, 10, 2)))
@pytest.mark.parametrize('size1', list(range(2, 10, 2)) + list(range(1, 10, 2)))
def test_rank_2_negative_strides_1(dtype, size0, size1):
    a = np.random.rand(size0, size1).astype(dtype)[..., ::-1]
    np.testing.assert_array_equal(a, ttt.pass_(a))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('size0', list(range(2, 10, 2)) + list(range(1, 10, 2)))
@pytest.mark.parametrize('size1', list(range(2, 10, 2)) + list(range(1, 10, 2)))
def test_rank_2_negative_strides_2(dtype, size0, size1):
    a = np.random.rand(size0, size1).astype(dtype)[::-1]
    np.testing.assert_array_equal(a, ttt.pass_(a))


@pytest.mark.parametrize('dtype', unsupported_dtypes)
@pytest.mark.xfail(raises=TypeError)
def test_rank_2_negative_strides(dtype):
    ttt.pass_(np.empty((0, 0), dtype=dtype))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('size0', list(range(2, 10, 2)) + list(range(1, 10, 2)))
@pytest.mark.parametrize('size1', list(range(2, 10, 2)) + list(range(1, 10, 2)))
@pytest.mark.parametrize('size2', [3, 4])
def test_rank_3_tensor(dtype, size0, size1, size2):
    a = np.random.rand(size0, size1, size2).astype(dtype)
    np.testing.assert_array_equal(a, ttt.pass_(a))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('size0', list(range(2, 10, 2)) + list(range(1, 10, 2)))
@pytest.mark.parametrize('size1', list(range(2, 10, 2)) + list(range(1, 10, 2)))
@pytest.mark.parametrize('size2', [3, 4])
def test_rank_3_negative_strides_1(dtype, size0, size1, size2):
    a = np.random.rand(size0, size1, size2).astype(dtype)[..., ::-1]
    np.testing.assert_array_equal(a, ttt.pass_(a))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('size0', list(range(2, 10, 2)) + list(range(1, 10, 2)))
@pytest.mark.parametrize('size1', list(range(2, 10, 2)) + list(range(1, 10, 2)))
@pytest.mark.parametrize('size2', [3, 4])
def test_rank_3_negative_strides_2(dtype, size0, size1, size2):
    a = np.random.rand(size0, size1, size2).astype(dtype)[::-1]
    np.testing.assert_array_equal(a, ttt.pass_(a))


@pytest.mark.parametrize('dtype', unsupported_dtypes)
@pytest.mark.xfail(raises=TypeError)
def test_rank_3_negative_strides(dtype):
    ttt.pass_(np.empty((0, 0, 0), dtype=dtype))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('size', [2, 5, 6])
@pytest.mark.xfail(raises=TypeError)
def test_rank_3_unsupported(dtype, size):
    ttt.pass_(np.empty((0, 0, size), dtype=dtype))


@pytest.mark.parametrize(
    'bounds1', [[1, 2, 3, 4], [1.0, 2, 3, 4], [1.0, 2.0, 3.0, 4.0]]
)
def test_bounds(bounds1):
    bounds2 = ttt.pass_bounds_(bounds1)

    np.testing.assert_array_equal(bounds1, bounds2)


@pytest.mark.parametrize(
    'bounds1', [['f', 2, 3, 4], [1.0, None, 3, 4], [1.0, 2.0, 4.0]]
)
@pytest.mark.xfail(raises=TypeError)
def test_invalid_bounds(bounds1):
    ttt.pass_bounds_(bounds1)


@pytest.mark.parametrize('ndim', [0, 4, 5, 6])
@pytest.mark.parametrize('dtype', supported_dtypes)
@pytest.mark.xfail(raises=TypeError)
def test_unsupported_ndim(ndim, dtype):
    ttt.pass_(np.empty(tuple(repeat(0, ndim)), dtype=dtype))
