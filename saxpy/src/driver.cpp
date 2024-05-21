#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <omp.h>
#include "kernel.h"
#include "common.h"

int main(int argc, char* argv[])
{
    unsigned int size = 0;
    unsigned int nrep, block_size;
    if (argc != 4) 
    {
        fprintf (stderr, "Usage: %s <size> <block size> <nb rep>\n", argv[0]);
        return 1;
    }
    else
    {
        size = atoi(argv[1]);
        block_size = atoi(argv[2]);
        nrep = atoi(argv[3]);
    }

    double tdiff = 0.0;

    srand(0);
    
    const size_t size_bytes = size * sizeof(float);
    const unsigned int grid_size = (size + block_size - 1) / block_size;
    const float a = 2.0f;

    float* x = NULL;
    float* y = NULL;
    float* y_check = NULL;
    x = (float *)malloc(size_bytes);
    y = (float *)malloc(size_bytes);
    y_check = (float *)malloc(size_bytes);
    init_vector(x, size);
    init_vector(y, size);

    float* d_x = NULL;
    float* d_y = NULL;
    CHECK(hipMalloc(&d_x, size_bytes));
    CHECK(hipMalloc(&d_y, size_bytes));
    CHECK(hipMemcpy(d_x, x, size_bytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_y, y, size_bytes, hipMemcpyHostToDevice));
    CHECK(hipDeviceSynchronize());
        
    printf("Check of saxpy (size=%d) with %d threads/block and %d blocks\n",size, block_size, grid_size);
    saxpy_kernel<<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(a, d_x, d_y, size);
    CHECK(hipMemcpy(y_check, d_y, size_bytes, hipMemcpyDeviceToHost));
    saxpy_cpu(a, x, &y, size);
    for (unsigned int i = 0; i < size; i++)
    {
        if (y_check[i] != y[i])
        {
            printf("Check KO, i=%d, gpu_y=%f, cpu_y=%f\n",i,y_check[i],y[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("Check OK\n");

    printf("Execution of saxpy with %d threads/block and %d blocks\n",block_size, grid_size);
    const double t1 = omp_get_wtime();
    for (unsigned int i = 0; i < nrep; i++)
    {
        saxpy_kernel<<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(a, d_x, d_y, size);
        CHECK(hipDeviceSynchronize());
    }
    const double t2 = omp_get_wtime();
    tdiff = (t2 - t1) / nrep;

    CHECK(hipMemcpy(y, d_y, size_bytes, hipMemcpyDeviceToHost));

    printf("Average Time : %.0f ns\n",tdiff*10e9);
        
    CHECK(hipFree(d_x));
    CHECK(hipFree(d_y));

    free(x);
    free(y);

    return EXIT_SUCCESS;
}