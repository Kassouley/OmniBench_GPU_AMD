#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include "kernel.h"
#include "driver.h"
#include "common.h"

int driver (const unsigned int size, dim3 blockDim, dim3 gridDim, const unsigned int nb_rep, const unsigned int nwu, float tdiff[NB_META])
{
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

    for (unsigned int i = 0; i < nb_rep; i++)
    {
        #ifdef ROCBLAS
            matrixMultiply(size, d_A, d_B, d_C);
        #else
            matrixMultiply<<<gridDim, blockDim, 0, hipStreamDefault>>>(size, d_A, d_B, d_C);
        #endif
        HIP_CHECK_CALL(hipDeviceSynchronize());
    }

    HIP_CHECK_CALL(hipMemcpy(C, d_C, size_bytes, hipMemcpyDeviceToHost));

    HIP_CHECK_CALL(hipFree(d_A));
    HIP_CHECK_CALL(hipFree(d_B));
    HIP_CHECK_CALL(hipFree(d_C));
    free(A);
    free(B);
    free(C);

    return EXIT_SUCCESS;
}