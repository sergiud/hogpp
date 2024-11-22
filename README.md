# HOGpp

![Linux](https://github.com/sergiud/hogpp/actions/workflows/linux.yml/badge.svg)
![macOS](https://github.com/sergiud/hogpp/actions/workflows/macos.yml/badge.svg)
![Windows](https://github.com/sergiud/hogpp/actions/workflows/windows.yml/badge.svg)
[![codecov](https://codecov.io/gh/sergiud/hogpp/branch/master/graph/badge.svg?token=PQ3WKQGKC4)](https://codecov.io/gh/sergiud/hogpp)
[![Documentation Status](https://readthedocs.org/projects/hogpp/badge/?version=latest)](https://hogpp.readthedocs.io/en/latest/?badge=latest)
![PyPI - Version](https://img.shields.io/pypi/v/hogpp)

This repository contains an implementation of the rectangular histogram of
oriented gradients feature descriptor (R-HOG) using integral histograms. The
integral histogram representation allows to quickly compute HOG features in
subregions of an image in constant time. This is particularly useful if the
features in an image must be computed repeatedly, e.g., in a sliding window
manner.

## Features

* C++ templated implementation
* Python support for 32, 64, and 80 bit floating point precision
* Unrestricted input size (e.g., OpenCV as of version 4.5.5 requires the input
  to be a [multiple of the block
  size](https://github.com/opencv/opencv/blob/5f249a3e67bfe3627e184bf5535da64daeaeb1c8/modules/objdetect/src/hog.cpp#L95-L96))
* Support for arbitrary integer (8 bit to 64 bit, both signed and unsigned) and
  floating point input (e.g., OpenCV requires 8-bit unsigned integer input)
* Masking support (i.e., spatial exclusion of gradient magnitudes from
  contributing to features)

For a complete summary of differences between HOGpp and existing
implementations, refer to the [feature
matrix](https://hogpp.readthedocs.io/en/stable/index.html#comparison-to-existing-libraries).

## Getting Started

In Python:

```python
from hogpp import IntegralHOGDescriptor

desc = IntegralHOGDescriptor()
# Load image
image = # ...
# Precompute the gradient histograms. This needs to be done only once for each image.
desc.compute(image)
# Extract the feature descriptor of a region of interest. The method can be
# called multiple times for different subregions of the above image. Note the
# use of matrix indexing along each axis opposed to Cartesian coordinates.
roi = (0, 0, 128, 64) # top left (row, column) size (height, width)
X = desc(roi)
```

## License

HOGpp is provided under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).
