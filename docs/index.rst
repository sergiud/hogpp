.. HOGpp documentation master file, created by
   sphinx-quickstart on Mon Aug 16 01:00:53 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HOGpp
=====

Overview
--------

HOGpp implements the rectangular histogram of oriented gradients feature
descriptor (R-HOG) using integral histograms. The integral histogram
representation allows to quickly compute HOG features in subregions of an image
in constant time. This is particularly useful if the features in an image must
be computed repeatedly, e.g., in a sliding window manner.

Features
--------

* C++ templated implementation
* Python support with 32 and 64 bit floating point precision
* Unrestricted input size (e.g., OpenCV requires the input to be a `multiple of
  the block size <https://github.com/opencv/opencv/blob/d24befa0bc7ef5e73bf8b1402fa1facbdbf9febb/modules/objdetect/src/hog.cpp#L93-L96>`__)
* Masking support

Getting Started
---------------

1. Load the module and instantiate the descriptor:

   .. code-block:: python

        from hogpp import IntegralHOGDescriptor

        desc = IntegralHOGDescriptor()


2. Load the `image` and precompute its integral histogram R-HOG representation.
   This needs to be done only once per image:

   .. code-block:: python

        desc.compute(image)

3. Extract the feature descriptor of a region of interest using a function call
   on a :class:`hogpp.IntegralHOGDescriptor` instance, i.e., by invoking
   :meth:`hogpp.IntegralHOGDescriptor.__call__`. The method can be called
   multiple times for different subregions of the image whose integral histogram
   representation was previously precomputed.

   .. code-block:: python

        # top left (row, column) size (height, width)
        roi = (0, 0, 128, 64)
        X = desc(roi)

   .. note::

      :class:`hogpp.IntegralHOGDescriptor` uses matrix indexing along each axis
      as opposed to Cartesian coordinates, i.e., the first index corresponds to
      the vertical (:math:`y`) coordinate, the second index to the horizontal
      (:math:`x`) coordinate, etc.

.. toctree::
   :hidden:

   usage


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
