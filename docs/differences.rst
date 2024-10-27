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
inferior to the original formulation. The reason for this being that neither
pixel down-weighting using a Gaussian nor trilinear interpolation can be
performed efficiently within the integral histogram framework. However, the
integral histogram R-HOG formulation is substantially faster while being a
sufficiently close approximation to the original R-HOG formulation.

Despite the above limitations, our evaluation on the :ref:`INRIA person dataset
<inria-performance>` and the comparison against OpenCV’s ``HOGDescriptor``
indicates that particularly the Gaussian down-weighting :ref:`does not
necessarily improve <inria-quantitative-results>` the generalization ability of
the associated classifiers.

For a comparison of both approaches, the interested reader should refer to
:cite:t:`Zhu2006`. Additional evaluation of related approaches can be found in
:cite:`Dollar2009`.
