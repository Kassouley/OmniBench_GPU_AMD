#include <hip/hip_runtime.h>

#ifdef BASIC
__global__ void sgemm_kernel (unsigned int N, const float* a, const float* b, float* c)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N)
    {
        float sum = 0.0f;
        for (unsigned int i = 0; i < N; i++)
        {
            sum += a[row * N + i ] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

#endif