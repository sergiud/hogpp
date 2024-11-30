# HOGpp - Fast histogram of oriented gradients computation using integral
# histograms
#
# Copyright 2022 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

import numpy as np
import pytest
import type_caster_test.tensor.c_style as ttt_c
import type_caster_test.tensor.f_style as ttt_f

styles = [ttt_f, ttt_c]
layouts = [np.ascontiguousarray, np.asfortranarray]


@pytest.mark.parametrize('style', styles)
def test_rank_0_tensor(style):
    a = np.random.rand()
    np.testing.assert_equal(a, style.pass_(a))


@pytest.mark.parametrize('style', styles)
@pytest.mark.parametrize('layout', layouts)
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('size0', list(range(0, 10, 2)) + list(range(1, 10, 2)))
def test_rank_1_tensor(style, layout, dtype, size0):
    a = layout(np.random.rand(size0).astype(dtype))
    np.testing.assert_array_equal(a, style.pass_(a))


@pytest.mark.parametrize('style', styles)
@pytest.mark.parametrize('layout', layouts)
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('size0', list(range(0, 10, 2)) + list(range(1, 10, 2)))
@pytest.mark.parametrize('size1', list(range(0, 10, 2)) + list(range(1, 10, 2)))
def test_rank_2_tensor(style, layout, dtype, size0, size1):
    a = layout(np.random.rand(size0, size1).astype(dtype))
    b = a[..., ::-1]
    c = a[::-1, ...]

    np.testing.assert_array_equal(a, style.pass_(a))
    np.testing.assert_array_equal(b, style.pass_(b))
    np.testing.assert_array_equal(c, style.pass_(c))


@pytest.mark.parametrize('style', styles)
@pytest.mark.parametrize('layout', layouts)
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('size0', list(range(0, 10, 2)) + list(range(1, 10, 2)))
@pytest.mark.parametrize('size1', list(range(5, 10, 2)) + list(range(6, 10, 2)))
@pytest.mark.parametrize('size2', list(range(5, 10, 2)) + list(range(6, 10, 2)))
def test_rank_3_tensor(style, layout, dtype, size0, size1, size2):
    a = layout(np.random.rand(size0, size1, size2).astype(dtype))
    b = a[..., ::-1]
    c = a[::-1, ...]

    np.testing.assert_array_equal(a, style.pass_(a))
    np.testing.assert_array_equal(b, style.pass_(b))
    np.testing.assert_array_equal(c, style.pass_(c))


@pytest.mark.parametrize('style', styles)
@pytest.mark.parametrize('layout', layouts)
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('size0', list(range(0, 10, 2)) + list(range(1, 10, 2)))
@pytest.mark.parametrize('size1', list(range(5, 10, 2)) + list(range(6, 10, 2)))
@pytest.mark.parametrize('size2', list(range(5, 10, 2)) + list(range(6, 10, 2)))
@pytest.mark.parametrize('size3', list(range(5, 10, 2)) + list(range(6, 10, 2)))
def test_rank_4_tensor(style, layout, dtype, size0, size1, size2, size3):
    a = layout(np.random.rand(size0, size1, size2, size3).astype(dtype))
    b = a[..., ::-1]
    c = a[::-1, ...]

    np.testing.assert_array_equal(a, style.pass_(a))
    np.testing.assert_array_equal(b, style.pass_(b))
    np.testing.assert_array_equal(c, style.pass_(c))


@pytest.mark.parametrize('style', styles)
@pytest.mark.parametrize('layout', layouts)
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('size0', list(range(0, 10, 2)) + list(range(1, 10, 2)))
@pytest.mark.parametrize('size1', list(range(0, 10, 2)) + list(range(1, 10, 2)))
@pytest.mark.parametrize('size2', list(range(5, 10, 2)) + list(range(6, 10, 2)))
@pytest.mark.parametrize('size3', list(range(5, 10, 2)) + list(range(6, 10, 2)))
@pytest.mark.parametrize('size4', list(range(5, 10, 2)) + list(range(6, 10, 2)))
def test_rank_5_tensor(style, layout, dtype, size0, size1, size2, size3, size4):
    a = layout(np.random.rand(size0, size1, size2, size3, size4).astype(dtype))
    b = a[..., ::-1]
    c = a[::-1, ...]

    np.testing.assert_array_equal(a, style.pass_(a))
    np.testing.assert_array_equal(b, style.pass_(b))
    np.testing.assert_array_equal(c, style.pass_(c))


@pytest.mark.parametrize('style', styles)
@pytest.mark.parametrize('dtype', [np.object_])
@pytest.mark.xfail(raises=TypeError)
def test_unsupported_dtype(style, dtype):
    style.pass_(np.empty((0, 0), dtype=dtype))
