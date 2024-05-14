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

__global__ void kernel_gpu (unsigned int m, unsigned int n, unsigned int p,
                                    const float* a, const float* b, float* c)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < p)
    {
        float sum = 0.0f;
        for (unsigned int i = 0; i < n; i++)
        {
            sum += a[row * n + i ] * b[i * p + col];
        }
        c[row * p + col] = sum;
    }
}

void kernel_cpu (unsigned int m, unsigned int n, unsigned int p, 
                    const float* a, const float* b, float* c)
{
    for(unsigned int i = 0; i < m; i++)
    {
        for(unsigned int j = 0; j < p; j++)
        {
            c[i*p+j] = 0;
            for(unsigned int k = 0; k < n; k++)
            {
                c[i*p+j] += a[i*n+k] * b[k*p+j];
            }
        }
    }
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
    h_C_gpu = (float *)malloc(N * N * sizeof(float));
    h_C_cpu = (float *)malloc(N * N * sizeof(float));

    // Initialize input matrices
    initializeMatrix(h_A, N, N);
    initializeMatrix(h_B, N, N);

    // Allocate memory on device
    CHECK(hipMalloc(&d_A, N * N * sizeof(float)));
    CHECK(hipMalloc(&d_B, N * N * sizeof(float)));
    CHECK(hipMalloc(&d_C, N * N * sizeof(float)));

    // Copy matrices from host to device
    CHECK(hipMemcpy(d_A, h_A, N * N * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_B, h_B, N * N * sizeof(float), hipMemcpyHostToDevice));

    // Perform kernel
    float alpha = 1.0f;
    float beta = 0.0f;

    dim3 blockDim (N, N);
    dim3 gridDim (1,1);
    if ( N > 32 )
    {
        blockDim.x = 32;
        gridDim.x = ceil(double(N)/double(blockDim.x));
    }
    if ( N > 32 )
    {
        blockDim.y = 32;
        gridDim.y = ceil(double(N)/double(blockDim.y));
    }

    hipLaunchKernelGGL(kernel_gpu, gridDim, blockDim, 0, 0, N, N, N, d_A, d_B, d_C);
    kernel_cpu(N, N, N, h_A, h_B, h_C_cpu);
    
    // Copy result matrix from device to host
    CHECK(hipMemcpy(h_C_gpu, d_C, N * N * sizeof(float), hipMemcpyDeviceToHost));

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            if (h_C_cpu[i*N+j] != h_C_gpu[i*N+j])
            {
                printf("CHECK NOT OK\n");
                CHECK(hipFree(d_A));
                CHECK(hipFree(d_B));
                CHECK(hipFree(d_C));
                free(h_A);
                free(h_B);
                free(h_C_cpu);
                free(h_C_gpu);
                return EXIT_FAILURE;
            }
        }
    }
    printf("CHECK OK\n");

    CHECK(hipFree(d_A));
    CHECK(hipFree(d_B));
    CHECK(hipFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}


