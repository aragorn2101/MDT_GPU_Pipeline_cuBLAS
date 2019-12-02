/*
 *  CUDA kernel for re-ordering the data before running cuBLAS kernel.
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
 *  GPU kernel to re-order array data
 *
 *  Nelements: number of elements in the array x number of polarisations
 *             this flexibility allows to process for a single
 *             polarisation if needed,
 *  Npol: number of polarisations,
 *  Nchannels: number of frequency channels in each spectrum,
 *  FOutput: array output from F-engine,
 *  XInput: array to be input to cuBLAS kernel.
 *
 *  Example call:
 *
 *  NumThreadx = (Npol*Nelements >= 32) ? 32 : (Npol*Nelements);
 *  NumThready = 32;
 *  NumThreadz = 1;
 *  NumBlockx  = Nspectra;
 *  NumBlocky  = Nchannels/NumThready + ((Nchannels%NumThready != 0) ? 1 : 0);
 *  NumBlockz  = (Npol*Nelements)/NumThreadx + (((Npol*Nelements)%NumThreadx != 0) ? 1 : 0);
 *
 *  ReorderXInput<<< dim3(NumBlockx,NumBlocky,NumBlockz), dim3(NumThreadx,NumThready,NumThreadz) >>>(Npol*Nelements, Nchannels, d_FOutput, d_XInput);
 *
 */
__global__ void ReorderXInput(int Nelements, int Nchannels, cufftComplex *FOutput, cuComplex *XInput)
{
  __shared__ cufftComplex sh_Temp[32][32];

  int channelIdx = blockIdx.y*blockDim.y + threadIdx.y;
  int elementIdx = blockIdx.z*blockDim.x + threadIdx.x;
  int FOutputIdx = (blockIdx.x*Nelements + elementIdx)*Nchannels + channelIdx;
  int XInputIdx  = (channelIdx*gridDim.x + blockIdx.x)*Nelements + elementIdx;

  if (channelIdx < Nchannels && elementIdx < Nelements)
  {
    // Copy data from output of F-engine
    sh_Temp[threadIdx.x][threadIdx.y] = FOutput[FOutputIdx];

    // Copy data to input for X-engine
    XInput[XInputIdx] = sh_Temp[threadIdx.x][threadIdx.y];
  }
}
