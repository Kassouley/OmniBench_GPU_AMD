#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <string.h>
#include <math.h>
#include "kernel.h"
#include "common.h"

int driver_check(unsigned int size, float** out_cpu, float** out_gpu, unsigned int* size_out_ptr,
                dim3 blockDim, dim3 gridDim)
{
    srand(0);
    const size_t size_bytes = size * size * sizeof(float);

    *out_cpu = (float*) malloc(size_bytes);
    *out_gpu = (float*) malloc(size_bytes);
    *size_out_ptr = size * size;
    
    float* A = (float*) malloc(size_bytes);
    float* B = (float*) malloc(size_bytes);
    init_matrix(size, size, A);
    init_matrix(size, size, B);

    float* d_A = NULL;
    float* d_B = NULL;
    float *d_C = NULL;
    HIP_CHECK_CALL(hipMalloc(&d_A, size_bytes));
    HIP_CHECK_CALL(hipMalloc(&d_B, size_bytes));
    HIP_CHECK_CALL(hipMalloc(&d_C, size_bytes));

    printf("=== Copy data CPU > GPU . . .\n");
    HIP_CHECK_CALL(hipMemcpy(d_A, A, size_bytes, hipMemcpyHostToDevice));
    HIP_CHECK_CALL(hipMemcpy(d_B, B, size_bytes, hipMemcpyHostToDevice));
    HIP_CHECK_CALL(hipDeviceSynchronize());
    printf("=== Copy data CPU > GPU done.\n");
    
    printf("=== Check kernel on GPU for size=%d, blockDim(%d, %d, %d), gridDim(%d, %d, %d)\n", 
                size, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);

    #ifdef ROCBLAS
        matrixMultiply(size, d_A, d_B, d_C);
    #else
        matrixMultiply<<<gridDim, blockDim, 0, hipStreamDefault>>>(size, d_A, d_B, d_C);
    #endif
    printf("=== Check kernel on CPU for size=%d . . .\n",size);
    // #pragma omp parallel for schedule(dynamic)
    for(unsigned int i = 0; i < size; i++)
    {
        for(unsigned int j = 0; j < size; j++)
        {
            (*out_cpu)[i*size+j] = 0;
            for(unsigned int k = 0; k < size; k++)
            {
                (*out_cpu)[i*size+j] += A[i*size+k] * B[k*size+j];
            }
        }
    }
    printf("=== Check kernel on CPU done.\n");

    HIP_CHECK_CALL(hipDeviceSynchronize());
    printf("=== Check kernel on GPU done.\n");

    printf("=== Copy data GPU > CPU . . .\n");
    HIP_CHECK_CALL(hipMemcpy(*out_gpu, d_C, size_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK_CALL(hipDeviceSynchronize());
    printf("=== Copy data GPU > CPU done.\n");

    HIP_CHECK_CALL(hipFree(d_A));
    HIP_CHECK_CALL(hipFree(d_B));
    HIP_CHECK_CALL(hipFree(d_C));
    free(A);
    free(B);
    return EXIT_SUCCESS;
}