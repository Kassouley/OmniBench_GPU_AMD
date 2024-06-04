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
    const size_t size_bytes = size * sizeof(float);
    const float a = 2.0f;

    *out_cpu =( float *)malloc(size_bytes);
    *out_gpu = (float *)malloc(size_bytes);
    *size_out_ptr = size;

    float* x = (float *)malloc(size_bytes);
    float* y = (float *)malloc(size_bytes);
    init_vector(x, size);
    init_vector(y, size);

    float* d_x = NULL;
    float* d_y = NULL;
    HIP_CHECK_CALL(hipMalloc(&d_x, size_bytes));
    HIP_CHECK_CALL(hipMalloc(&d_y, size_bytes));

    printf("=== Copy data CPU > GPU . . .\n");
    HIP_CHECK_CALL(hipMemcpy(d_x, x, size_bytes, hipMemcpyHostToDevice));
    HIP_CHECK_CALL(hipMemcpy(d_y, y, size_bytes, hipMemcpyHostToDevice));
    HIP_CHECK_CALL(hipDeviceSynchronize());
    printf("=== Copy data CPU > GPU done.\n");
    
    printf("=== Check kernel on GPU for size=%d . . .\n",size);
    saxpy<<<gridDim, blockDim, 0, hipStreamDefault>>>(a, d_x, d_y, size);

    printf("=== Check kernel on CPU for size=%d . . .\n",size);
    for (unsigned int i = 0; i < size; i++)
    {
        (*out_cpu)[i] = a * x[i] + y[i];
    }
    printf("=== Check kernel on CPU done.\n");

    HIP_CHECK_CALL(hipDeviceSynchronize());
    printf("=== Check kernel on GPU done.\n");

    printf("=== Copy data GPU > CPU . . .\n");
    HIP_CHECK_CALL(hipMemcpy(*out_gpu, d_y, size_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK_CALL(hipDeviceSynchronize());
    printf("=== Copy data GPU > CPU done.\n");

    HIP_CHECK_CALL(hipFree(d_x))
    HIP_CHECK_CALL(hipFree(d_y))
    free(x);
    free(y);
    return EXIT_SUCCESS;
}