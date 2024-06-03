#include <hip/hip_runtime.h>

__global__ void svmm_kernel (unsigned int N, const float* a, const float* b, float* c)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float sum = 0.0f;
        for (unsigned int i = 0; i < N; i++)
        {
            sum += a[idx * N + i ] * b[i];
        }
        c[idx] = sum;
    }
}