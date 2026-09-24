Differences to Dalal & Triggs Formulation
-----------------------------------------

When using HOGpp, one should be aware of subtle differences between the integral
histogram implementation and the one originally proposed by Dalal & Triggs.

In general, computing R-HOG consists of the following steps:

1. (optional) gamma correction
2. gradient computation
3. orientation binning within a cell

   - down-weighting of pixels using a Gaussian with respect to their position
     within a block
   - trilinear interpolation of magnitude votes between neighboring bins in
     both orientation and position

4. block normalization

Provided these steps, R-HOG extracted using an integral histogram is slightly
inferior to the original formulation. The reason for this being that trilinear
interpolation cannot be performed efficiently within the integral histogram
framework. However, the integral histogram R-HOG formulation is substantially
faster while being a sufficiently close approximation to the original R-HOG
formulation.

Gaussian down-weighting is available through the ``sigma`` parameter of
:class:`hogpp.IntegralHOGDescriptor` and disabled by default. The weighted cell
histograms are exact up to rounding. Their cost depends on the block layout:

- If the block stride is not smaller than the cell size along both axes, the
  weight of a pixel within a cell does not depend on the block containing it.
  The weights are then folded into an additional integral histogram and each
  weighted cell histogram is extracted in :math:`O(1)`, provided the block
  origin is a multiple of the block stride. This includes the default block
  layout, except for the last block that is placed flush with the region
  border if the region extent minus the block extent is not a multiple of the
  block stride. The additional integral histogram requires as much memory as
  the regular one times the number of cells per block, e.g., 4 times for 2x2
  cell blocks.
- Otherwise, each weighted cell histogram is summed from the per-pixel votes
  in :math:`O(c)` where :math:`c` denotes the number of pixels in a cell.

Our evaluation on the :ref:`INRIA person dataset <inria-performance>` was
performed without Gaussian down-weighting. The comparison against OpenCV’s
``HOGDescriptor``, which applies it, indicates that particularly the Gaussian
down-weighting :ref:`does not necessarily improve <inria-quantitative-results>`
the generalization ability of the associated classifiers.

For a comparison of both approaches, the interested reader should refer to
:cite:t:`Zhu2006`. Additional evaluation of related approaches can be found in
:cite:`Dollar2009`.
