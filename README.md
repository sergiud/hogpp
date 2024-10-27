# HOGpp

![Linux](https://github.com/sergiud/hogpp/actions/workflows/linux.yml/badge.svg)
![macOS](https://github.com/sergiud/hogpp/actions/workflows/macos.yml/badge.svg)
![Windows](https://github.com/sergiud/hogpp/actions/workflows/windows.yml/badge.svg)
[![codecov](https://codecov.io/gh/sergiud/hogpp/branch/master/graph/badge.svg?token=PQ3WKQGKC4)](https://codecov.io/gh/sergiud/hogpp)
[![Documentation Status](https://readthedocs.org/projects/hogpp/badge/?version=latest)](https://hogpp.readthedocs.io/en/latest/?badge=latest)

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
matrix](#comparison-to-existing-libraries) below.

## Requirements

* C++17 compiler
* [Boost](https://www.boost.org) 1.70
* [CMake](https://gitlab.kitware.com/cmake/cmake) 3.15
* [Eigen](https://gitlab.com/libeigen/eigen) 3.4.0
* [fmt](https://github.com/fmtlib/fmt) 6.0
* [OpenCV](https://github.com/opencv/opencv) 4.0
* [pybind11](https://github.com/pybind/pybind11) 2.6.2 (version 2.9.0 is
  required for use with Visual Studio 17 2022 and above)

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
# called multiple times for different subregions of the above image. Note the
# use of matrix indexing along each axis opposed to Cartesian coordinates.
roi = (0, 0, 128, 64) # top left (row, column) size (height, width)
X = desc(roi)
```

## Comparison to Existing Libraries

The following feature matrix summarizes the differences between existing
implementations.

| Library      | Signed Orientations | Custom Gradients |  Masking  | Arbitrary Input Size | Implementation |
| :---         |        :---:        |       :---:      |   :---:   |        :---:         |     :---:      |
| HOGpp        | ✔️                   | ✔️                | ✔️         | ✔️                    | C++            |
| OpenCV       | ✔️                   | ✖                | ✖         | ✖                    | C++            |
| scikit-image | ✖                   | ✖                | ✖         | ✔️                    | Cython/Python  |


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
a sufficiently close approximation to the original R-HOG formulation.

Despite the above limitations, our evaluation on the [INRIA person
dataset](#performance-on-the-inria-person-dataset) and the comparison against
OpenCV's `HOGDescriptor` indicates that particularly the Gaussian down-weighting
[does not necessarily improve](#quantitative-results) the generalization ability
of the associated classifiers.

For a comparison of both approaches, the interested reader should refer to:

> Qiang Zhu, Mei-Chen Yeh, Kwang-Ting Cheng, & Avidan, S. (2006). Fast Human Detection Using a Cascade of Histograms of Oriented Gradients. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (Vol. 2, pp. 1491–1498). IEEE. DOI: 10.1109/CVPR.2006.119

## Performance on the INRIA Person Dataset

HOGpp implementation was validated by applying it to the task of pedestrian
detection. For the most part, the experiments by Dalal & Triggs were replicated
with few alterations.

More specifically, we trained a linear support vector machine (SVM) in the
primal using stochastic gradient descent (SGD) on features extracted from
[cropped annotations](https://github.com/sergiud/pascal-annotation-v1-extractor)
of the INRIA person dataset. We then quantitatively compared the performance of
the obtained classifier against models trained on descriptors extracted using
[OpenCV](https://docs.opencv.org/4.5.5/d5/d33/structcv_1_1HOGDescriptor.html)
and
[scikit-image](https://scikit-image.org/docs/0.19.x/api/skimage.feature.html?highlight=hog#skimage.feature.hog).

The following figure provides an intuition of the steps involved in training a
pedestrian classifier and its use on HOG features for predicting the
corresponding class.

<p align="center">
<img src="./docs/experiments/inria/overview-light.svg#gh-light-mode-only" />
<img src="./docs/experiments/inria/overview-dark.svg#gh-dark-mode-only" />
</p>

On a high level, HOG features describe the silhouette of a pedestrian which is
eventually used in a way that is similar to how template matching works albeit
accounting for some pose variations.

### Training

For comparison purposes, we trained all the classifiers using same fixed set of
HOG parameters producing a 3780-dimensional feature vector. Specifically, the
parameters employed were:

* 9 orientation bins constructed from `unsigned` gradients
* cell size of 8×8 pixels
* overlapping blocks consisting of 16×16 pixels (or equivalently, 2×2 cells)
* `l2-hys` block normalization clipped at 0.2

We then trained an initial SVM classifier using 5-fold stratified inner
cross-validation while optimizing the regularization term penalty using grid
search. 20% of the samples of each training split were additionally used as a
validation split to allow for early stopping.

After obtaining the initial model, we used each classifier to perform an
exhaustive search for false positives (i.e., hard mining) and retrained the
classifiers by including the hard mined samples.

We used confidence based sampling opposed to random sampling to subsample the
large set of false positives. Specifically, up to 30 most confident false
positives (i.e., samples farthest away from the decision boundary) were selected
as hard negatives.

### Quantitative Results

The following plot summarizes the performance of refined models at various
thresholds.

<p align="center">
<img src="./docs/experiments/inria/roc-light.svg#gh-light-mode-only" />
<img src="./docs/experiments/inria/roc-dark.svg#gh-dark-mode-only" />
</p>

Overall, the HOGpp based model outperforms models that use OpenCV and
scikit-image HOG descriptors.

A detailed look at additional classification metrics, however, shows that HOGpp
achieves a lower precision compared to other two implementations. Yet, the
recall and consequently the F₁ score are considerably higher thereby
outperforming both implementations.

| Implementation | Precision     | Recall     | F₁ score     |  Accuracy     |
| :---           | ---:          | ---:       | ---:         |  ---:         |
| hogpp          | 95.45%        | **90.75%** | **93.04%**   |  **97.20%**   |
| skimage        | 96.95%        | 83.53%     | 89.74%       |  96.06%       |
| cv2            | **98.32%**    | 79.46%     | 87.89%       |  95.48%       |

#### Hard Negatives

It is also important to consider the number of hard negatives produced by each
of the HOG descriptor implementations. The following table provides an overview
of the corresponding absolute numbers.

| Implementation | Hard negatives |
| :---           | ---:           |
| hogpp          | **30584**      |
| cv2            | 31113          |
| skimage        | 33433          |

In this specific application, the initial model obtained from HOGpp descriptors
generates the least number of false positives usable for further refinement.
While the overall number of training samples is lowest, the HOGpp model still
achieves the best performance in terms of the F₁ score and ROC AUC. At the same
time, this indicates that the initial HOGpp model already generalizes better
than OpenCV and scikit-image based models.

Due to the probabilistic nature of the learning process, particularly the number
of hard negatives can vary depending on the chosen seed. Therefore, the
corresponding numbers should be taken with a grain of salt because at times the
OpenCV based model can produce fewer hard negatives than HOGpp. This
observation, however, does not affect the generalization ability of the refined
models on this task.

#### Runtime Performance

The following bar plot summarizes the average runtime of individual HOG
implementations for extracting the descriptor of a single 128×64 (height×width)
region of interest (ROI) within a larger image as performed during hard mining.

<p align="center">
<img src="./docs/experiments/inria/runtime-light.svg#gh-light-mode-only" />
<img src="./docs/experiments/inria/runtime-dark.svg#gh-dark-mode-only" />
</p>

The runtime of the `precompute` stage applicable only to HOGpp is negligible and
can therefore be hardly observed in the bar plot. As such, the `extract` stage
is computationally more expensive. Nevertheless, HOGpp outperforms both
implementations in terms of the average cumulative runtime for a single ROI
consuming around 32 μs.

The speed up factor achieved by HOGpp with respect to OpenCV and scikit-image
implementations is as follows:

|       | cv2   | skimage |
| :---  | :---: | :---:   |
| hogpp | ×2.4  | ×7.3    |

### Final Remarks

As always, the provided results are specific to the described experiment,
environment, and the setup used to evaluate the models, and therefore should not
be extrapolated to different tasks without validation.

## Further Reading

> Porikli, F. (2005). Integral histogram: a fast way to extract histograms in Cartesian spaces. In IEEE Computer Society Conference on Computer Vision and Pattern Recognition (Vol. 1, pp. 829–836). IEEE. DOI: 10.1109/CVPR.2005.188

> Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. In IEEE Computer Society Conference on Computer Vision and Pattern Recognition (Vol. 1, pp. 886–893). IEEE. DOI: 10.1109/CVPR.2005.177

> Dollar, P., Tu, Z., Perona, P., & Belongie, S. (2009). Integral Channel Features. In Proceedings of the British Machine Vision Conference 2009 (Vol. 30, pp. 91.1-91.11). British Machine Vision Association. DOI: 10.5244/C.23.91

## License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This document and all figures are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

HOGpp is provided under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).
