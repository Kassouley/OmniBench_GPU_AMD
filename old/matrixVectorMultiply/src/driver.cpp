#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <omp.h>
#include <math.h>
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
    
    const size_t size_bytes_matrix = size * size * sizeof(float);
    const size_t size_bytes_vector = size * sizeof(float);
    block_size = sqrt(block_size);
    const unsigned int grid_size = (size + block_size - 1) / block_size;
    dim3 blockDim (block_size, block_size);
    dim3 gridDim (grid_size, grid_size);

    float* A = NULL;
    float* B = NULL;
    float* C = NULL;
    A = (float *)malloc(size_bytes_matrix);
    B = (float *)malloc(size_bytes_vector);
    C = (float *)malloc(size_bytes_vector);
    init_matrix(size, size, A);
    init_vector(B, size);

    float* d_A = NULL;
    float* d_B = NULL;
    float *d_C = NULL;

    CHECK(hipMalloc(&d_A, size_bytes_matrix));
    CHECK(hipMalloc(&d_B, size_bytes_vector));
    CHECK(hipMalloc(&d_C, size_bytes_vector));
    CHECK(hipMemcpy(d_A, A, size_bytes_matrix, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_B, B, size_bytes_vector, hipMemcpyHostToDevice));
    CHECK(hipDeviceSynchronize());
    const double t1 = omp_get_wtime();
    for (unsigned int i = 0; i < nrep; i++)
    {
        svmm_kernel<<<gridDim, blockDim, 0, 0>>>(size, d_A, d_B, d_C);
        CHECK(hipDeviceSynchronize());
    }
    const double t2 = omp_get_wtime();
    tdiff = (t2 - t1) / nrep;

    printf("Average Time : %.0f ns\n",tdiff*10e9);

    CHECK(hipMemcpy(C, d_C, size_bytes_vector, hipMemcpyDeviceToHost));

    CHECK(hipFree(d_A));
    CHECK(hipFree(d_B));
    CHECK(hipFree(d_C));
    free(A);
    free(B);
    free(C);

    return EXIT_SUCCESS;
}