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
    const size_t size_bytes = size * sizeof(float);
    const float a = 2.0f;

    float* x = (float *)malloc(size_bytes);
    float* y = (float *)malloc(size_bytes);
    init_vector(x, size);
    init_vector(y, size);

    for (unsigned int i = 0; i < size; i++)
    {
        y[i] = a * x[i] + y[i];
    }
    
    FILE * output = NULL;
    output = fopen(file_name, "w");
    for (unsigned int i = 0; i < size; i++)
    {
        fprintf(output, "%f ", y[i]);
    }
    fclose(output);
    
    free(x);
    free(y);
    return EXIT_SUCCESS;
}
int driver_check_gpu(unsigned int size, dim3 blockDim, dim3 gridDim, char* file_name)
{
    srand(0);
    const size_t size_bytes = size * sizeof(float);
    const float a = 2.0f;

    float* x = (float *)malloc(size_bytes);
    float* y = (float *)malloc(size_bytes);
    init_vector(x, size);
    init_vector(y, size);

    float* d_x = NULL;
    float* d_y = NULL;
    HIP_CHECK_CALL(hipMalloc(&d_x, size_bytes));
    HIP_CHECK_CALL(hipMalloc(&d_y, size_bytes));

    HIP_CHECK_CALL(hipMemcpy(d_x, x, size_bytes, hipMemcpyHostToDevice));
    HIP_CHECK_CALL(hipMemcpy(d_y, y, size_bytes, hipMemcpyHostToDevice));

    saxpy<<<gridDim, blockDim, 0, hipStreamDefault>>>(a, d_x, d_y, size);

    HIP_CHECK_CALL(hipMemcpy(y, d_y, size_bytes, hipMemcpyDeviceToHost));

    FILE * output = NULL;
    output = fopen(file_name, "w");
    for (unsigned int i = 0; i < size; i++)
    {
        fprintf(output, "%f ", y[i]);
    }
    fclose(output);
    
    HIP_CHECK_CALL(hipFree(d_x))
    HIP_CHECK_CALL(hipFree(d_y))
    free(x);
    free(y);
    return EXIT_SUCCESS;
}
