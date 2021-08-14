# HOGpp

![Linux](https://github.com/sergiud/hogpp/actions/workflows/linux.yml/badge.svg)
![Windows](https://github.com/sergiud/hogpp/actions/workflows/windows.yml/badge.svg)
[![codecov](https://codecov.io/gh/sergiud/hogpp/branch/master/graph/badge.svg?token=PQ3WKQGKC4)](https://codecov.io/gh/sergiud/hogpp)

This repository contains an implementation of the rectangular histogram of
oriented gradients feature descriptor (R-HOG) using integral histograms. The
integral histogram representation allows to quickly compute HOG features in
subregions of an image in constant time. This is particularly useful if the
features in an image must be computed repeatedly, e.g., in a sliding window
manner.

HOG features may be seen as a special case of the Scale-invariant Feature
Transform (SIFT) computed over a dense grid of keypoints where each block is
additionally contrast-normalized.

## Features

* C++ templated implementation
* Python support with 32 and 64 bit floating point precision
* Unrestricted input size (e.g., compared to OpenCV that requires powers of two
  input)

## Requirements

* C++17 compiler
* [CMake](https://gitlab.kitware.com/cmake/cmake) 3.5
* [Eigen](https://gitlab.com/libeigen/eigen) 3.3.7
* [fmt](https://github.com/fmtlib/fmt) 6.0
* [OpenCV](https://github.com/opencv/opencv) 4.0
* [pybind11](https://github.com/pybind/pybind11) 2.2.4

More recent versions of the above are expected to work as well.

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
# called multiple times for different subregions of the above image.
roi = (0, 0, 64, 128)
X = desc(roi)
```

## Differences to Dalal & Triggs Formulation

When using HOGpp, one should be aware of subtle differences between the integral
histogram implementation and the one originally proposed by Dalal & Triggs.

In general, computing R-HOG consists of the following steps:

1. (optional) gamma correction
2. gradient computation
3. orientation binning within a cell
    * down-weighting of pixels using a Gaussian with respect to their position
      within a block
    * trilinear interpolation of magnitude votes between neighboring bins in
      both orientation and position
4. block normalization

Provided these steps, R-HOG extracted using an integral histogram is slightly
inferior to the original formulation. The reason for this being that neither
pixel down-weighting using a Gaussian nor trilinear interpolation can be
performed efficiently within the integral histogram framework. However, the
integral histogram R-HOG formulation is substantially faster while being
sufficiently close approximation to the original R-HOG formulation.

For a comparison of both approaches, the interested reader should refer to:

> Qiang Zhu, Mei-Chen Yeh, Kwang-Ting Cheng, & Avidan, S. (2006). Fast Human Detection Using a Cascade of Histograms of Oriented Gradients. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (Vol. 2, pp. 1491–1498). IEEE. DOI: 10.1109/CVPR.2006.119

## Further Reading

> Porikli, F. (2005). Integral histogram: a fast way to extract histograms in Cartesian spaces. In IEEE Computer Society Conference on Computer Vision and Pattern Recognition (Vol. 1, pp. 829–836). IEEE. DOI: 10.1109/CVPR.2005.188

> Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. In IEEE Computer Society Conference on Computer Vision and Pattern Recognition (Vol. 1, pp. 886–893). IEEE. DOI: 10.1109/CVPR.2005.177

> Dollar, P., Tu, Z., Perona, P., & Belongie, S. (2009). Integral Channel Features. In Proceedings of the British Machine Vision Conference 2009 (Vol. 30, pp. 91.1-91.11). British Machine Vision Association. DOI: 10.5244/C.23.91
