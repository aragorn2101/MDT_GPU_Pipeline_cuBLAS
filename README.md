# MDT_GPU_Pipeline_cuBLAS

The Mauritius Deuterium Telescope GPU pipeline implemented with a custom
F-engine and a cuBLAS-based X-engine.

The F-engine consists of a PFB. A first kernel calculates the window
coefficients for the FIR filters of the PFB and stores them in GPU global
memory. Then, the PPFBatch() kernel is run. The latter reads the window
coefficients, applies the FIR filters to the data and accumulates. Finally the
Nvidia cuFFT library is called to perform Fourier transform. Header file
**cufft.h** is needed in the preamble.

The X-engine is actually a matrix multiplier from the Nvidia cuBLAS API. Since
these kernels take data in column-major format and output data from the
F-engine is in frequency channel order, the data will need to be re-ordered
before input to the cuBLAS kernel.  This is done using the ReorderXInput()
kernel. The function used from cuBLAS is cugemmBatched() and we need to include
**cublas_v2.h** in the preamble.
