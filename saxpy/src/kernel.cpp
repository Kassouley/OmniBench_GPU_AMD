#include <hip/hip_runtime.h>

#ifdef BASIC
__global__ void saxpy_kernel(const float a, const float* d_x, float* d_y, const unsigned int size)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < size)
    {
        d_y[i] = a * d_x[i] + d_y[i];
    }
}
#else
__global__ void saxpy_kernel(const float a, const float* d_x, float* d_y, const unsigned int size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < size; 
         i += blockDim.x * gridDim.x) 
      {
          d_y[i] = a * d_x[i] + d_y[i];
      }
}
#endif