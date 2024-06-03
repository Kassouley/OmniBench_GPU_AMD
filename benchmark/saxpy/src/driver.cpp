#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <omp.h>
#include "kernel.h"
#include "common.h"
#include "driver.h"

int driver (const unsigned int size, dim3 blockDim, dim3 gridDim, const unsigned int nrep, float tdiff[NB_META])
{    
    int nwu = 5;
    const size_t size_bytes = size * sizeof(float);
    const float a = 2.0f;

    float* x = (float*)malloc(size_bytes);
    float* y = (float*)malloc(size_bytes);
    init_vector(x, size);
    init_vector(y, size);

    float* d_x = NULL;
    float* d_y = NULL;
    HIP_CHECK_CALL(hipMalloc(&d_x, size_bytes));
    HIP_CHECK_CALL(hipMalloc(&d_y, size_bytes));
    HIP_CHECK_CALL(hipMemcpy(d_x, x, size_bytes, hipMemcpyHostToDevice));
    HIP_CHECK_CALL(hipMemcpy(d_y, y, size_bytes, hipMemcpyHostToDevice));

    kernelBenchmark(tdiff, nwu, nrep, 
                    saxpy, 
                    gridDim, blockDim, 0, hipStreamDefault, 
                    a, d_x, d_y, size);

    HIP_CHECK_CALL(hipMemcpy(y, d_y, size_bytes, hipMemcpyDeviceToHost));

    HIP_CHECK_CALL(hipFree(d_x));
    HIP_CHECK_CALL(hipFree(d_y));

    free(x);
    free(y);

    return EXIT_SUCCESS;
}