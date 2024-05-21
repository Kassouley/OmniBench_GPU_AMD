#include <hip/hip_runtime.h>

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

void sgemm_cpu (unsigned int N, const float* a, const float* b, float* c)
{
    for(unsigned int i = 0; i < N; i++)
    {
        for(unsigned int j = 0; j < N; j++)
        {
            c[i*N+j] = 0;
            for(unsigned int k = 0; k < N; k++)
            {
                c[i*N+j] += a[i*N+k] * b[k*N+j];
            }
        }
    }
}