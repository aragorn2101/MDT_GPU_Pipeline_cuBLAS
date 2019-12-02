/*
 *  CUDA kernels for pre-filters and for calculating the polyphase
 *  structure prior to the FFT kernels, for cuBLAS-based pipeline.
 *
 *  Copyright (C) 2019 Nitish Ragoomundun
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

/*
 *  GPU kernel to compute FIR filter coefficients.
 *
 *  Ntaps: number of taps,
 *  Nchannels: number of frequency channels expected in output spectrum,
 *  Filter: array of size Ntaps x Nchannels which will hold coefficients
 *          of FIR filter.
 *
 *  Example call:
 *
 *  NumThreadx = MaxThreadsPerBlock;
 *  NumThready = 1;
 *  NumThreadz = 1;
 *  NumBlockx  = (Ntaps*Nchannels)/NumThreadx + (((Ntaps*Nchannels)%NumThreadx != 0) ? 1 : 0);
 *  NumBlocky  = 1;
 *  NumBlockz  = 1;
 *
 *  WindowedSinc<<< dim3(NumBlockx,NumBlocky,NumBlockz), dim3(NumThreadx,NumThready,NumThreadz) >>>(Ntaps, Nchannels, d_Filter);
 *
 */
__global__ void WindowedSinc(int Ntaps, long Nchannels, float *Filter)
{
  long idx = threadIdx.x + blockIdx.x*blockDim.x;

  /*  Temporary variables to prevent redundant computation  */
  float tmp1, tmp2, tmp_filter, tmp_window;

  if (idx < Ntaps*Nchannels)
  {

    /*
     *  Filter: Sinc
     *
     *  sinc( ( channel - (Ntaps x Nchannels)/2 ) / Nchannels )
     *
     */
    tmp1 = (idx - 0.5f*Ntaps*Nchannels) / Nchannels;

    if ( tmp1 == 0.0f )  /*  To prevent division by 0  */
      tmp_filter = 1.0f ;
    else
      tmp_filter = sinpif( tmp1 ) / ( d_PI * tmp1 );


    /*
     *  Window: Exact Blackman
     *
     *  a0 - a1cos( 2 x PI x i / (Ntaps x Nchannels) ) + a2cos( 4 x PI x i / (Ntaps x Nchannels) )
     *
     */
    tmp2 = 2.0f*idx / (Ntaps*Nchannels);
    tmp_window = 0.42659071f - 0.49656062f*cospif(tmp2) + 0.07684867f*cospif(2.0f*tmp2);


    /*  Write Windowed Sinc to global memory array  */
    Filter[idx] = tmp_filter * tmp_window;
  }
}



/*
 *  GPU kernel to compute polyphase structure.
 *
 *  Ntaps: number of taps,
 *  Nchannels: number of channels in output,
 *  Filter: array of size Ntaps x Nchannels containing FIR filter,
 *  InSignal: array of size (Nspectra - 1 + Ntaps) x Nchannels containing
 *            interleaved IQ samples of the input signal in an
 *            an array of cufftComplex vectors (I:x, Q:y),
 *  PolyStruct: array of size Nchannels which will hold output.
 *
 *  Example call:
 *
 *  NumThreadx = MaxThreadsPerBlock;
 *  NumThready = 1;
 *  NumThreadz = 1;
 *  NumBlockx  = (Ntaps*Nchannels)/NumThreadx + (((Ntaps*Nchannels)%NumThreadx != 0) ? 1 : 0);
 *  NumBlocky  = Nspectra;
 *  NumBlockz  = Npol*Nelements;
 *
 *  PPFBatch<<< dim3(NumBlockx,NumBlocky,NumBlockz), dim3(NumThreadx,NumThready,NumThreadz) >>>(Ntaps, Nchannels, d_Filter, d_PPFInput, d_FOutput);
 *
 */
__global__ void PPFBatch(int Ntaps, long Nchannels, float *Filter, cufftComplex *InSignal, cufftComplex *PolyStruct)
{
  int i;
  int channelIdx = threadIdx.x + blockIdx.x*blockDim.x;
  long stride_elem, stride_spec;
  float tmp_filter;
  cufftComplex tmp_input, tmp_product;


  if (channelIdx < Nchannels)
  {
    /*  Input strides  */
    /*  wrt element index  */
    stride_elem = blockIdx.z * (gridDim.y - 1 + Ntaps) * Nchannels;

    /*  wrt spectrum index  */
    stride_spec = blockIdx.y * Nchannels;

    tmp_product.x = 0.0f;
    tmp_product.y = 0.0f;

    for ( i=0 ; i<Ntaps ; i++ )
    {
      /*  Read input signal data and filter coefficient  */
      tmp_input  = InSignal[stride_elem + stride_spec + i*Nchannels + channelIdx];
      tmp_filter = Filter[i*Nchannels + channelIdx];

      /*  Accumulate FIR  */
      tmp_product.x = fmaf(tmp_filter, tmp_input.x, tmp_product.x);  // I
      tmp_product.y = fmaf(tmp_filter, tmp_input.y, tmp_product.y);  // Q
    }

    /*  Write output to array in global memory  */
    //PolyStruct[(blockIdx.z*gridDim.y + blockIdx.y)*Nchannels + channelIdx] = tmp_product;

    // NOTE: in order to output specific order for use with Reorder kernel for cuBLAS:
    PolyStruct[(blockIdx.y*gridDim.z + blockIdx.z)*Nchannels + channelIdx] = tmp_product;
  }
}
