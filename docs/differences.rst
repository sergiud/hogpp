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

HOGpp performs all of these steps. The Gaussian down-weighting and the spatial
part of the trilinear interpolation are disabled by default, however, since
both make the extraction slower. Without them, a pixel votes only into its own
cell, while its vote is still interpolated between neighboring orientation
bins.

Both steps are computed exactly from the integral histogram, up to rounding,
for any cell size and any placement of the blocks. Unlike the original
formulation, cells are therefore not restricted to a fixed grid.

The only remaining difference arises when both steps are enabled. The
interpolation spreads the vote of a pixel up to one cell beyond its own cell
and thus possibly beyond the block. Dalal & Triggs only consider the pixels
inside the block. HOGpp instead weights such pixels by the same Gaussian
evaluated at their position relative to the block, which keeps the weights
continuous across the block border.

Each step adds to the cost of the extraction. The interpolation keeps an
additional table in at least double precision, which for single precision
input is about twice the size of the integral histogram. If the block stride is
not smaller than the cell size, the Gaussian down-weighting keeps an additional
table as large as the integral histogram times the number of cells per block.
Otherwise, and in combination with the interpolation, the weighted cells are
summed from the votes of their pixels instead.

Our evaluation on the :ref:`INRIA person dataset <inria-performance>` was
performed without either step. The comparison against OpenCV’s
``HOGDescriptor``, which applies the Gaussian down-weighting, indicates that it
:ref:`does not necessarily improve <inria-quantitative-results>` the
generalization ability of the associated classifiers.

For a comparison of both approaches, the interested reader should refer to
:cite:t:`Zhu2006`. Additional evaluation of related approaches can be found in
:cite:`Dollar2009`.
