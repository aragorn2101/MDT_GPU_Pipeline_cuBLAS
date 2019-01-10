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
#include <cufft.h>


/*
 *  Re-ordering kernel
 *
 *  Grid dimensions for running this kernel:
 *
 *  NumThreadx = (Nelements >= 32) ? 32 : Nelements;
 *  NumThready = 32;
 *  NumThreadz = 1;
 *
 *  NumBlockx = Nspectra;
 *  NumBlocky = Nchannels/NumThready + ((Nchannels%NumThready != 0) ? 1 : 0);
 *  NumBlockz = Nelements/NumThreadx + ((Nelements%NumThreadx != 0) ? 1 : 0);
 *
 *
 */
__global__ void ReorderXInput(int Nelements, int Nchannels, cufftComplex *FOutput, cuComplex *XInput)
{
  int channelIdx = blockIdx.y*blockDim.y + threadIdx.y;
  int elementIdx = blockIdx.z*blockDim.x + threadIdx.x;
  int FOutputIdx = (blockIdx.x*Nelements + elementIdx)*Nchannels + channelIdx;
  int XInputIdx  = (channelIdx*gridDim.x + blockIdx.x)*Nelements + elementIdx;

  if (channelIdx < Nchannels && elementIdx < Nelements)
    XInput[XInputIdx] = FOutput[FOutputIdx];
}
