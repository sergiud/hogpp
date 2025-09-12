.. HOGpp documentation master file, created by
   sphinx-quickstart on Mon Aug 16 01:00:53 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HOGpp
=====

HOGpp implements the rectangular histogram of oriented gradients feature
descriptor (R-HOG) using integral histograms. The integral histogram
representation allows to quickly compute HOG features in subregions of an image
in constant time. This is particularly useful if the features in an image must
be computed repeatedly, e.g., in a sliding window manner.

HOG features may be seen as a special case of the Scale-invariant Feature
Transform (SIFT) computed over a dense grid of keypoints where each block is
additionally contrast-normalized.

Features
--------

-  C++ templated implementation
-  Python support for 32, 64, and 80 bit floating point precision
-  Unrestricted input size (e.g., OpenCV as of version 4.12.0 requires
   the input to be a `multiple of the block
   size <https://github.com/opencv/opencv/blob/49486f61fb25722cbcf586b7f4320921d46fb38e/modules/objdetect/src/hog.cpp#L94-L95>`__)
-  Support for arbitrary integer (8 bit to 64 bit, both signed and
   unsigned) and floating point input (e.g., OpenCV requires 8-bit
   unsigned integer input)
-  Masking support (i.e., spatial exclusion of gradient magnitudes from
   contributing to features)

Comparison to Existing Libraries
--------------------------------

The following feature matrix summarizes the differences between existing
implementations.

============= =================== ================ ======= ==================== ==============
 Library      Signed Orientations Custom Gradients Masking Arbitrary Input Size Implementation
============= =================== ================ ======= ==================== ==============
 HOGpp        ✔️                   ✔️                ✔️       ✔️                    C++
 OpenCV       ✔️                   ✖                ✖       ✖                    C++
 scikit-image ✖                   ✖                ✖       ✔️                    Cython/Python
============= =================== ================ ======= ==================== ==============

.. toctree::
   :hidden:

   quickstart
   build
   usage
   differences
   inria
   bibliography
   license


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
