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

import io
import os
import pickle

if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
    path = os.getenv('HOGPPPATH')
    if path:
        os.add_dll_directory(path)

from hogpp import IntegralHOGDescriptor
from PIL import Image
import copy
import numpy as np
import pytest


@pytest.fixture(params=['gray', 'color'])
def circle_image(request):
    # Obtain an absolute to the image file to avoid requiring to run pytest from
    # the project root.
    image_fn = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            '..',
            '..',
            'data',
            f'circle-{request.param}.png',
        )
    )

    with Image.open(image_fn, 'r') as im:
        return np.asarray(im)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
def test_descriptor_size(dtype):
    desc = IntegralHOGDescriptor()

    bounds = (128, 64)

    image = np.random.rand(*bounds).astype(dtype)
    desc.compute(image)

    assert desc.features_.dtype == dtype

    X = desc.features_

    np.testing.assert_array_equal(X, desc([0, 0, *bounds]))
    np.testing.assert_array_equal(X[np.newaxis], desc([[0, 0, *bounds]]))
    assert desc.features_.size == 3780


@pytest.mark.parametrize('magnitude', ['identity', 'square', 'sqrt'])
def test_magnitude_attribute(magnitude):
    desc = IntegralHOGDescriptor(magnitude=magnitude)
    assert desc.magnitude_ == magnitude


@pytest.mark.parametrize('binning', ['unsigned', 'signed'])
def test_binning_attribute(binning):
    desc = IntegralHOGDescriptor(binning=binning)
    assert desc.binning_ == binning


@pytest.mark.parametrize('block_norm', ['l1', 'l1-hys', 'l2', 'l2-hys', 'l1-sqrt'])
def test_block_norm_attribute(block_norm):
    desc = IntegralHOGDescriptor(block_norm=block_norm)
    assert desc.block_norm_ == block_norm


def test_default_attributes():
    desc = IntegralHOGDescriptor(
        n_bins=8,
        cell_size=(4, 4),
        block_size=(12, 12),
        block_stride=(8, 8),
        binning='unsigned',
        block_norm='l1',
    )

    assert not desc
    assert desc.n_bins_ == 8
    assert desc.histogram_ is None
    assert desc.features_ is None
    assert desc([0, 0, 0, 0]) is None
    np.testing.assert_equal(desc.cell_size_, (4, 4))
    np.testing.assert_equal(desc.block_size_, (12, 12))
    np.testing.assert_equal(desc.block_stride_, (8, 8))
    assert desc.clip_norm_ is None
    assert desc.epsilon_ is not None


@pytest.mark.xfail(raises=TypeError)
@pytest.mark.parametrize('binning', ['signed1', 'unsigned1'])
def test_invalid_binning(binning):
    IntegralHOGDescriptor(binning=binning)


@pytest.mark.xfail(raises=TypeError)
@pytest.mark.parametrize('block_norm', ['l11', 'foo'])
def test_invalid_block_norm(block_norm):
    IntegralHOGDescriptor(block_norm=block_norm)


@pytest.mark.xfail(raises=TypeError)
@pytest.mark.parametrize('magnitude', ['l11', 'foo'])
def test_invalid_magnitude(magnitude):
    IntegralHOGDescriptor(magnitude=magnitude)


@pytest.mark.xfail(raises=TypeError)
def test_unsupported_paramater():
    IntegralHOGDescriptor(some_parameter=True)


@pytest.mark.xfail(raises=ValueError)
def test_negative_bins():
    IntegralHOGDescriptor(n_bins=-1)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize('cell_size', [[0, 0], [-1, 0], [0, -1]])
def test_invalid_cell_size(cell_size):
    IntegralHOGDescriptor(cell_size=cell_size)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize('block_size', [[0, 0], [-1, 0], [0, -1]])
def test_invalid_block_size(block_size):
    IntegralHOGDescriptor(block_size=block_size)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize('block_stride', [[0, 0], [-1, 0], [0, -1]])
def test_invalid_block_stride(block_stride):
    IntegralHOGDescriptor(block_stride=block_stride)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
@pytest.mark.parametrize('block_norm', ['l1', 'l1-hys', 'l2', 'l2-hys', 'l1-sqrt'])
@pytest.mark.parametrize('magnitude', ['identity', 'square', 'sqrt'])
def test_vertical_gradient(dtype, block_norm, magnitude):
    image = np.empty((16, 16), dtype=dtype)
    image[:, :8] = 1
    image[:, 8:] = 0

    desc = IntegralHOGDescriptor(block_norm=block_norm, magnitude=magnitude)
    desc.compute(image)

    assert desc([0, 0, 0, 0]).size == 0
    assert desc([[0, 0, 0, 0]]).size == 0
    assert desc.histogram_.size != 0

    X = desc.features_.ravel()
    assert X.dtype == dtype

    (idxs,) = np.nonzero(X)
    diff = np.diff(idxs)

    # Ensure the vote is at the beginning of each block
    np.testing.assert_array_equal(
        idxs - desc.n_bins_ // 2, np.arange(X.size, step=desc.n_bins_)
    )
    np.testing.assert_array_equal(diff, desc.n_bins_)

    XX = desc([0, 0, *image.shape[::-1]]).ravel()
    assert XX.dtype == dtype

    np.testing.assert_array_equal(X, XX)

    desc1 = copy.deepcopy(desc)
    desc1.compute(np.gradient(image, axis=(0, 1)))

    np.testing.assert_array_almost_equal(X, desc1.features_.ravel())


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
@pytest.mark.parametrize('block_norm', ['l1', 'l1-hys', 'l2', 'l2-hys', 'l1-sqrt'])
@pytest.mark.parametrize('magnitude', ['identity', 'square', 'sqrt'])
def test_horizontal_gradient(dtype, block_norm, magnitude):
    image = np.empty((16, 16), dtype=dtype)
    image[:8] = 1
    image[8:] = 0

    desc = IntegralHOGDescriptor(block_norm=block_norm, magnitude=magnitude)
    desc.compute(image)

    assert desc([0, 0, 0, 0]).size == 0
    assert desc.histogram_.size != 0

    X = desc.features_.ravel()
    assert X.dtype == dtype

    (idxs,) = np.nonzero(X)
    diff = np.diff(idxs)

    # Ensure the vote is in the middle of each block
    np.testing.assert_array_equal(idxs, np.arange(X.size, step=desc.n_bins_))
    np.testing.assert_array_equal(diff, desc.n_bins_)

    XX = desc([0, 0, *image.shape[::-1]]).ravel()
    assert XX.dtype == dtype

    np.testing.assert_array_equal(X, XX)


@pytest.mark.xfail(raises=TypeError)
@pytest.mark.parametrize('clip_norm', ['foo'])
def test_invalid_clip_norm_type(clip_norm):
    IntegralHOGDescriptor(clip_norm=clip_norm)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize('clip_norm', [-1, 0.0])
def test_invalid_clip_norm_value(clip_norm):
    IntegralHOGDescriptor(clip_norm=clip_norm)


@pytest.mark.xfail(raises=TypeError)
@pytest.mark.parametrize('epsilon', ['foo'])
def test_invalid_epsilon_type(epsilon):
    IntegralHOGDescriptor(epsilon=epsilon)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize('epsilon', [-1, -2.0])
def test_invalid_epsilon_value(epsilon):
    IntegralHOGDescriptor(epsilon=epsilon)


@pytest.mark.parametrize('block_norm', ['l1-hys', 'l2-hys'])
@pytest.mark.parametrize('clip_norm', [0.2, 0.5, 1, 1e3])
def test_clip_norm_init(block_norm, clip_norm):
    desc = IntegralHOGDescriptor(block_norm=block_norm, clip_norm=clip_norm)
    np.testing.assert_almost_equal(desc.clip_norm_, clip_norm)


@pytest.mark.parametrize('epsilon', [0, 1e-5, 1])
def test_epsilon_init(epsilon):
    desc = IntegralHOGDescriptor(epsilon=epsilon)
    np.testing.assert_almost_equal(desc.epsilon_, epsilon)


@pytest.mark.parametrize('block_norm', ['l1', 'l1-sqrt', 'l2'])
@pytest.mark.parametrize('clip_norm', [0.2, 0.5, 1, 1e3, None])
def test_no_clip_norm(block_norm, clip_norm):
    desc = IntegralHOGDescriptor(block_norm=block_norm, clip_norm=clip_norm)

    if clip_norm is not None:
        np.testing.assert_almost_equal(desc.clip_norm_, clip_norm)

    assert desc.block_norm_ == block_norm


@pytest.mark.parametrize('block_norm', ['l1', 'l1-hys', 'l1-sqrt', 'l2', 'l2-hys'])
@pytest.mark.parametrize('epsilon', [0, 1e-5, 1, None])
def test_ensure_epsilon(block_norm, epsilon):
    desc = IntegralHOGDescriptor(block_norm=block_norm, epsilon=epsilon)

    if epsilon is not None:
        np.testing.assert_almost_equal(desc.epsilon_, epsilon)

    assert desc.block_norm_ == block_norm


@pytest.mark.parametrize(
    'bounds',
    [
        [0, 0, 129, 2],
        [0, 0, 128, 65],
        [-1, 0, 128, 64],
        [0, -1, 128, 64],
        [0, 0, -1, 2],
        [0, 0, 2, -1],
    ],
)
@pytest.mark.xfail(raises=ValueError)
def test_invalid_bounds(bounds):
    desc = IntegralHOGDescriptor()

    image = np.random.rand(128, 64)
    desc.compute(image)

    desc(bounds)


@pytest.mark.parametrize('bounds', [[64, 32, 32, 32], [64, 32, 64, 16]])
def test_valid_bounds(bounds):
    desc = IntegralHOGDescriptor()

    image = np.random.rand(128, 64)
    desc.compute(image)

    X = desc(bounds)

    assert X.size > 0


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
@pytest.mark.parametrize('channels', [0, 1, 3, 4])
@pytest.mark.parametrize('block_norm', ['l1', 'l1-hys', 'l2', 'l2-hys', 'l1-sqrt'])
def test_zero_gradient(dtype, channels, block_norm):
    desc = IntegralHOGDescriptor(block_norm=block_norm, epsilon=0)

    shape = (128, 64)

    if channels > 0:
        shape = (*shape, channels)

    image = np.zeros(shape, dtype=dtype)
    desc.compute(image)

    assert desc.features_.size > 0
    assert not np.any(np.isinf(desc.features_))
    np.testing.assert_array_almost_equal(desc.features_, 0)


@pytest.fixture(params=[np.float32, np.float64, np.longdouble])
def horizontal_gradient_image(request):
    image = np.zeros((128, 64), dtype=request.param)

    i = image.shape[0] // 2
    image[i:, ...] = 1

    return image


@pytest.mark.parametrize('dtype', [np.bool_, np.uint8])
@pytest.mark.parametrize('channels', [0, 1, 3, 4])
def test_compute_mask_ndarray(dtype, channels, horizontal_gradient_image):
    desc = IntegralHOGDescriptor()

    image = horizontal_gradient_image

    i = image.shape[0] // 2
    mask = np.zeros_like(image, dtype=dtype)
    mask[i - 1 : i + 1, ...] = True

    if channels > 0:
        image = np.repeat(np.atleast_3d(image), channels, axis=-1)

    desc.compute(image, mask=mask)

    assert desc.features_.size > 0
    assert not np.any(np.isinf(desc.features_))
    np.testing.assert_array_equal(desc.features_[desc.features_ != 0], 0)

    desc1 = IntegralHOGDescriptor()
    desc1.compute(np.gradient(image, axis=(0, 1)), mask=mask)

    np.testing.assert_array_almost_equal(desc.features_, desc1.features_)


def test_compute_mask_callable(horizontal_gradient_image):
    desc = IntegralHOGDescriptor()

    image = horizontal_gradient_image
    i = image.shape[0] // 2

    def mask(key):
        k, l = key
        return k >= i - 1 and k <= i + 1

    desc.compute(image, mask=mask)

    assert not np.any(np.isinf(desc.features_))
    assert desc.features_.size > 0
    np.testing.assert_array_equal(desc.features_[desc.features_ != 0], 0)

    desc1 = IntegralHOGDescriptor()
    desc1.compute(np.gradient(image, axis=(0, 1)), mask=mask)

    np.testing.assert_array_almost_equal(desc.features_, desc1.features_)


@pytest.mark.parametrize('mask', [0, False, 0.0])
@pytest.mark.xfail(raises=ValueError)
def test_compute_image_mask_invalid(mask, horizontal_gradient_image):
    desc = IntegralHOGDescriptor()
    desc.compute(horizontal_gradient_image, mask=mask)


@pytest.mark.parametrize('mask', [0, False, 0.0])
@pytest.mark.xfail(raises=ValueError)
def test_compute_gradient_mask_invalid(mask, horizontal_gradient_image):
    desc = IntegralHOGDescriptor()
    desc.compute(np.gradient(horizontal_gradient_image, axis=(0, 1)), mask=mask)


@pytest.mark.xfail(raises=TypeError)
def test_compute_image_positional(horizontal_gradient_image):
    desc = IntegralHOGDescriptor()
    desc.compute(image=horizontal_gradient_image)


@pytest.mark.xfail(raises=TypeError)
def test_compute_mask_keyword(horizontal_gradient_image):
    desc = IntegralHOGDescriptor()
    desc.compute(horizontal_gradient_image, horizontal_gradient_image > 0)


@pytest.mark.xfail(raises=TypeError)
@pytest.mark.parametrize('dims', [0, 1, 4, 5])
def test_compute_image_invalid_dim(dims):
    desc = IntegralHOGDescriptor()
    desc.compute(np.empty((0,) * dims))


@pytest.mark.parametrize(
    'dtype',
    [
        np.bool_,
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uintc,
        np.intc,
        np.longlong,
        np.ulonglong,
    ],
)
def test_decay_to_double(dtype, horizontal_gradient_image):
    desc = IntegralHOGDescriptor()
    desc.compute(horizontal_gradient_image.astype(dtype))

    assert desc.features_.dtype == np.float64


def test_empty_multiple_bounds(horizontal_gradient_image):
    desc = IntegralHOGDescriptor()
    desc.compute(horizontal_gradient_image)

    X = desc([])
    np.testing.assert_array_equal(X.shape, (0, 0, 0, 0, 0, 0))


@pytest.mark.xfail(raises=ValueError)
def test_different_multiple_bounds(horizontal_gradient_image):
    desc = IntegralHOGDescriptor()
    desc.compute(horizontal_gradient_image)

    desc([[0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 3, 4]])


def test_multiple_bounds(horizontal_gradient_image):
    desc = IntegralHOGDescriptor()
    desc.compute(horizontal_gradient_image)

    X1, X2 = desc([[0, 0, 0, 0], [0, 0, 0, 0]])

    np.testing.assert_array_equal(X1, X2)

    Y1, Y2 = desc([[0, 0, 3, 4], [0, 0, 3, 4]])

    np.testing.assert_array_equal(Y1, Y2)


@pytest.mark.parametrize('block_norm', ['l1', 'l1-hys', 'l2', 'l2-hys', 'l1-sqrt'])
@pytest.mark.parametrize('magnitude', ['identity', 'square', 'sqrt'])
@pytest.mark.parametrize('clip_norm', [0.2, 0.5, 1, 1e3])
@pytest.mark.parametrize('epsilon', [0, 1e-5, 1, None])
def test_pickle_empty_descriptor(block_norm, magnitude, clip_norm, epsilon):
    desc = IntegralHOGDescriptor(
        block_norm=block_norm, magnitude=magnitude, clip_norm=clip_norm, epsilon=epsilon
    )

    desc1 = copy.deepcopy(desc)

    assert not desc
    assert desc.block_norm_ == desc1.block_norm_
    assert desc.magnitude_ == desc1.magnitude_
    assert desc.clip_norm_ == desc1.clip_norm_
    assert desc.epsilon_ == desc1.epsilon_

    np.testing.assert_array_equal(desc.features_, desc1.features_)
    np.testing.assert_array_equal(desc.histogram_, desc1.histogram_)


@pytest.mark.parametrize('block_norm', ['l1', 'l1-hys', 'l2', 'l2-hys', 'l1-sqrt'])
@pytest.mark.parametrize('magnitude', ['identity', 'square', 'sqrt'])
@pytest.mark.parametrize('clip_norm', [0.2, 0.5, 1, 1e3])
@pytest.mark.parametrize('epsilon', [0, 1e-5, 1, None])
def test_pickle_descriptor(
    block_norm, magnitude, clip_norm, epsilon, horizontal_gradient_image
):
    desc = IntegralHOGDescriptor(
        block_norm=block_norm, magnitude=magnitude, clip_norm=clip_norm, epsilon=epsilon
    )
    desc.compute(horizontal_gradient_image)

    with io.BytesIO() as f:
        pickle.dump(desc, f)
        b = bytes(f.getbuffer())

    with io.BytesIO(b) as f:
        desc1 = pickle.load(f)

    assert desc
    assert desc.block_norm_ == desc1.block_norm_
    assert desc.magnitude_ == desc1.magnitude_
    assert desc.clip_norm_ == desc1.clip_norm_
    assert desc.epsilon_ == desc1.epsilon_

    np.testing.assert_array_equal(desc.features_, desc1.features_)
    np.testing.assert_array_equal(desc.histogram_, desc1.histogram_)


def test_uniform_gradients(circle_image):
    size = circle_image.shape[:2][::-1]

    desc = IntegralHOGDescriptor(
        block_size=size,
        block_stride=size,
        cell_size=size,
        binning='unsigned',
        block_norm='l2-hys',
        magnitude='identity',
        n_bins=9,
    )
    desc.compute(circle_image)

    diff = np.diff(desc.features_.ravel())

    np.testing.assert_array_equal(diff, 0)
