#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <string.h>
#include <math.h>
#include "kernel.h"
#include "common.h"

int driver_check_cpu(unsigned int size, char* file_name)
{
    srand(0);
    const size_t size_bytes = size * size * sizeof(float);

    float* A = (float *)malloc(size_bytes);
    float* B = (float *)malloc(size_bytes);
    float* C = (float *)malloc(size_bytes);

    init_matrix(size, size, A);
    init_matrix(size, size, B);

    #pragma omp parallel for schedule(dynamic)
    for(unsigned int i = 0; i < size; i++)
    {
        for(unsigned int j = 0; j < size; j++)
        {
            C[i*size+j] = 0;
            for(unsigned int k = 0; k < size; k++)
            {
                C[i*size+j] += A[i*size+k] * B[k*size+j];
            }
        }
    }

    FILE * output = NULL;
    output = fopen(file_name, "w");
    for (unsigned int i = 0; i < size; i++)
    {
        for (unsigned int j = 0; j < size; j++)
        {
            fprintf(output, "%f ", C[i*size+j]);
        }
    }
    fclose(output);
    
    free(A);
    free(B);
    free(C);
    return EXIT_SUCCESS;
}
int driver_check_gpu(unsigned int size, dim3 blockDim, dim3 gridDim, char* file_name)
{
    srand(0);
    const size_t size_bytes = size * size * sizeof(float);
    float* A = (float *)malloc(size_bytes);
    float* B = (float *)malloc(size_bytes);
    float* C = (float *)malloc(size_bytes);
    init_matrix(size, size, A);
    init_matrix(size, size, B);

    float* d_A = NULL;
    float* d_B = NULL;
    float *d_C = NULL;
    HIP_CHECK_CALL(hipMalloc(&d_A, size_bytes));
    HIP_CHECK_CALL(hipMalloc(&d_B, size_bytes));
    HIP_CHECK_CALL(hipMalloc(&d_C, size_bytes));
    
    HIP_CHECK_CALL(hipMemcpy(d_A, A, size_bytes, hipMemcpyHostToDevice));
    HIP_CHECK_CALL(hipMemcpy(d_B, B, size_bytes, hipMemcpyHostToDevice));

    #ifdef ROCBLAS
        matrixMultiply(size, d_A, d_B, d_C);
    #else
        matrixMultiply<<<gridDim, blockDim, 0, hipStreamDefault>>>(size, d_A, d_B, d_C);
    #endif

    HIP_CHECK_CALL(hipMemcpy(C, d_C, size_bytes, hipMemcpyDeviceToHost));


    FILE * output = NULL;
    output = fopen(file_name, "w");
    for (unsigned int i = 0; i < size; i++)
    {
        for (unsigned int j = 0; j < size; j++)
        {
            fprintf(output, "%f ", C[i*size+j]);
        }
    }
    fclose(output);
    
    HIP_CHECK_CALL(hipFree(d_A));
    HIP_CHECK_CALL(hipFree(d_B));
    HIP_CHECK_CALL(hipFree(d_C));
    free(A);
    free(B);
    free(C);
    return EXIT_SUCCESS;
}
