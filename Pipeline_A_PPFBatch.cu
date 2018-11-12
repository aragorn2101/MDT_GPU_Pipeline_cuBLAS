/*
 *  CUDA kernel for polyphase filter kernel in F-engine working with
 *  custom X-engine.
 *
 *  Copyright (C) 2018 Nitish Ragoomundun
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Affero General Public License as
 *  published by the Free Software Foundation, either version 3 of the
 *  License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Affero General Public License for more details.
 *
 *  You should have received a copy of the GNU Affero General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include <cuda.h>
#include <cufft.h>


/*
 *  Kernel to compute polyphase structure.
 *
 *  Grid dimensions:
 *
 *  NumThreadx = Number of channels computed per block,
 *  NumThready = 1,
 *  NumThreadz = 1,
 *
 *  NumBlockx = Nchannels / NumThreadx,
 *  NumBlocky = Nspectra,
 *  NumBlockz = Nelements
 *
 *  Ntaps: number of taps,
 *  Nchannels: number of channels in output,
 *  Filter: array of size Ntaps x Nchannels containing FIR filter,
 *  InSignal: array of size 2 x Ntaps x Nchannels containing
 *            interleaved IQ samples of the input signal in an
 *            an array of float2 vectors (I:x, Q:y),
 *  PolyStruct: array of size Nchannels which will hold output.
 *
 */
__global__ void PPFBatch(int Ntaps, long Nchannels, float *Filter, float2 *InSignal, cufftComplex *PolyStruct)
{
  int i;
  int channelIdx = threadIdx.x + blockIdx.x*blockDim.x;
  long stride_element, stride_spect, stride_taps;
  float2 tmp_input;
  float tmp_filter;
  cufftComplex tmp_product;

  /*  Stride wrt element index  */
  stride_element = blockIdx.z * (gridDim.y - 1 + Ntaps) * Nchannels;

  /*  Stride wrt number of previous spectra  */
  stride_spect = blockIdx.y * Nchannels;

  tmp_product.x = 0.0f;
  tmp_product.y = 0.0f;

  for ( i=0 ; i<Ntaps ; i++ )
  {
    /*  Stride in spectrum wrt previous number of taps  */
    stride_taps = i*Nchannels;

    /*  Read input signal data and filter coefficient  */
    tmp_input = InSignal[stride_element + stride_spect + stride_taps + channelIdx];
    tmp_filter = Filter[stride_taps + channelIdx];

    /*  Accumulate FIR  */
    tmp_product.x = fmaf(tmp_filter, tmp_input.x, tmp_product.x);  // I
    tmp_product.y = fmaf(tmp_filter, tmp_input.y, tmp_product.y);  // Q
  }

  /*  Write output to array in global memory  */
  PolyStruct[(blockIdx.z * gridDim.y * Nchannels) + stride_spect + channelIdx] = tmp_product;
}
