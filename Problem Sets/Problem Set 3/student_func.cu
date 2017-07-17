/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>

__global__ void findMin(const float* const d_logLuminance, 
                        float* d_out)
{
    extern __shared__ float logLuminance[];

    unsigned int myId = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tid = threadIdx.x;
    
    logLuminance[tid] = d_logLuminance[myId];
    __syncthreads();
    
    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
        logLuminance[tid] = min(logLuminance[tid + s], logLuminance[tid]);
        }
        __syncthreads();
    }
    
    if(tid == 0)
    {
        d_out[blockIdx.x] = logLuminance[tid];
    }
    __syncthreads();
}

__global__ void finalMinReduce(float* d_in,
                                float *d_out)
{
    extern __shared__ float logLuminance[];

    unsigned int tid = threadIdx.x;
    
    logLuminance[tid] = d_in[tid];
    __syncthreads();
    
    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
        logLuminance[tid] = min(logLuminance[tid + s], logLuminance[tid]);
        }
        __syncthreads();
    }
    if(tid == 0)
    d_out[blockIdx.x] = logLuminance[0];
    
    __syncthreads();
}
__global__ void finalMaxReduce(float* d_in,
                                float *d_out)
{
    extern __shared__ float logLuminance[];

    unsigned int tid = threadIdx.x;
    
    
    logLuminance[tid] = d_in[tid];
    __syncthreads();
    
    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
        logLuminance[tid] = max(logLuminance[tid + s], logLuminance[tid]);
        }
        __syncthreads();
    }
    if(tid == 0)
    d_out[blockIdx.x] = logLuminance[0];
    
    __syncthreads();
}
__global__ void findMax(const float* const d_logLuminance, 
                                float * d_out)
{
    extern __shared__ float logLuminance[];

    unsigned int myId = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tid = threadIdx.x;
    
    
    logLuminance[tid] = d_logLuminance[myId];
    __syncthreads();
    
    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
        logLuminance[tid] = max(logLuminance[tid + s], logLuminance[tid]);
        }
        __syncthreads();
    }
    
    if(tid == 0)
    {
        d_out[blockIdx.x] = logLuminance[tid];
    }
    __syncthreads();
}

__global__ void findHist(const float * const d_logLuminance,
                                unsigned int * d_hist,
                                const float RANGE,
                                float min_logLum,
                                const size_t numRows,
                                const size_t numCols,
                                const size_t numBins)
{
    int2 thread_2D_pos = make_int2(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y);
    unsigned int myIdx = thread_2D_pos.y * numCols + thread_2D_pos.x;
    
    d_hist[myIdx] = 0;
    __syncthreads();
    unsigned int bin = min(static_cast<unsigned int>(numBins - 1),
                           static_cast<unsigned int>((d_logLuminance[myIdx] - min_logLum) / RANGE * numBins));
                           
    atomicAdd(&d_hist[bin], 1);
}

__global__ void prescan(unsigned int* const g_odata, unsigned int *g_idata, const size_t n, const size_t numCols, const size_t numRows)
{
extern __shared__ unsigned int temp[];  // allocated on invocation

unsigned int thid = threadIdx.x;
int offset = 1;

temp[2*thid] = g_idata[2*thid]; // load input into shared memory
temp[2*thid+1] = g_idata[2*thid+1];
__syncthreads();
 	
for (int d = n>>1; d > 0; d >>= 1)                    // build sum in place up the tree
{ 
__syncthreads();
   if (thid < d)
   {

    int ai = offset*(2*thid+1)-1;
    int bi = offset*(2*thid+2)-1;
    
 	
      temp[bi] += temp[ai];
   }
   offset *= 2;
}

   
if (thid == 0) { temp[n - 1] = 0; } // clear the last element
               
 	
for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
{
     offset >>= 1;
     __syncthreads();
     if (thid < d)                     
     {

       
    int ai = offset*(2*thid+1)-1;
    int bi = offset*(2*thid+2)-1;
 	
       
float t = temp[ai];
temp[ai] = temp[bi];
temp[bi] += t; 
      }
}
 __syncthreads();
     g_odata[2*thid] = temp[2*thid]; // write results to device memory
     g_odata[2*thid+1] = temp[2*thid+1];
 	
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
    float *d_out, *d_intermediate;
    float h_out;
    const dim3 blockSize(32, 16, 1);
    const dim3 gridSize( (numCols + blockSize.x - 1) / blockSize.x,
                       (numRows + blockSize.y - 1) / blockSize.y );
    checkCudaErrors(cudaMalloc(&d_intermediate, sizeof(float) * numCols));
    checkCudaErrors(cudaMalloc(&d_out, sizeof(float) * numRows));
    printf("numRows: %d\n", numRows);
	printf("numCols: %d\n", numCols);
	
    findMin<<<numRows, numCols, sizeof(float) * numCols>>>(d_logLuminance, d_intermediate);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    finalMinReduce<<<1, numRows, sizeof(float) * numRows>>>(d_intermediate, d_out);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    checkCudaErrors(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    min_logLum = h_out;
    
    findMax<<<numRows, numCols, sizeof(float) * numCols>>>(d_logLuminance, d_intermediate);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    finalMaxReduce<<<1, numRows, sizeof(float) * numRows>>>(d_intermediate, d_out);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    max_logLum = h_out;
    //max_logLum = d_out[0];
    
	printf("Min: %f\n", min_logLum);
	printf("Max: %f\n", max_logLum);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    const float RANGE = max_logLum - min_logLum;
    
    unsigned int *d_hist;
    checkCudaErrors(cudaMalloc(&d_hist, sizeof(unsigned int) * numBins));
    findHist<<<gridSize, blockSize>>>(d_logLuminance, d_hist, RANGE, min_logLum, numRows, numCols, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  /*Here are the steps you need to implement
    
    
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
       
       
       
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
       
    
    prescan<<<1, numBins, sizeof(unsigned int) * numBins * 2>>>(d_cdf, d_hist, numBins, numCols, numRows);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
       
}
