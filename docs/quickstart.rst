Getting Started
===============

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
