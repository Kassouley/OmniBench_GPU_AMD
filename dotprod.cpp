#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>

#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}

__global__ void kernel_gpu (unsigned int n, const float* a, const float* b, float* c)
{
    float sum = 0.0f;
    for (unsigned int i = 0; i < n; i++)
    {
        sum += a[i] * b[i];
    }
    *c = sum;
}

void initializeMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[j*rows+i] = (float)(rand() % 10); // random values
        }
    }
}

int main() {
    const int N = 512; // dimension of the square matrices
    int i = 0, j = 0;
    float *h_A, *h_B, *h_C_gpu, *h_C_cpu;
    float *d_A, *d_B, *d_C;

    // Allocate memory on host
    h_A = (float *)malloc(N * N * sizeof(float));
    h_B = (float *)malloc(N * N * sizeof(float));
    h_C_gpu = 0;

    // Initialize input matrices
    initializeMatrix(h_A, N, N);
    initializeMatrix(h_B, N, N);

    // Allocate memory on device
    CHECK(hipMalloc(&d_A, N * N * sizeof(float)));
    CHECK(hipMalloc(&d_B, N * N * sizeof(float)));
    CHECK(hipMalloc(&d_C, 1 * sizeof(float)));

    // Copy matrices from host to device
    CHECK(hipMemcpy(d_A, h_A, N * N * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_B, h_B, N * N * sizeof(float), hipMemcpyHostToDevice));

    // Perform kernel
    float alpha = 1.0f;
    float beta = 0.0f;

    dim3 blockDim (1, 1);
    dim3 gridDim (1,1);

    hipLaunchKernelGGL(kernel_gpu, gridDim, blockDim, 0, 0, N, d_A, d_B, d_C);
    
    // Copy result matrix from device to host
    CHECK(hipMemcpy(h_C_gpu, d_C, 1 * sizeof(float), hipMemcpyDeviceToHost));

    CHECK(hipFree(d_A));
    CHECK(hipFree(d_B));
    CHECK(hipFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}


