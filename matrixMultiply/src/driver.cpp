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
    unsigned int nwu, nrep, block_size;
    if (argc != 5) 
    {
        fprintf (stderr, "Usage: %s <size> <block size> <nb warmup> <nb rep>\n", argv[0]);
        return 1;
    }
    else
    {
        size = atoi(argv[1]);
        block_size = atoi(argv[2]);
        nwu = atoi(argv[3]);
        nrep = atoi(argv[4]);
    }

    double tdiff[NB_META];

    srand(0);
    
    const size_t size_bytes = size * size * sizeof(float);
    block_size = sqrt(block_size);
    const unsigned int grid_size = (size + block_size - 1) / block_size;
    dim3 blockDim (block_size, block_size);
    dim3 gridDim (grid_size, grid_size);

    float* A = NULL;
    float* B = NULL;
    float* C = NULL;
    A = (float *)malloc(size_bytes);
    B = (float *)malloc(size_bytes);
    C = (float *)malloc(size_bytes);
    init_matrix(size, size, A);
    init_matrix(size, size, B);

    float* d_A = NULL;
    float* d_B = NULL;
    float *d_C = NULL;

    CHECK(hipMalloc(&d_A, size_bytes));
    CHECK(hipMalloc(&d_B, size_bytes));
    CHECK(hipMalloc(&d_C, size_bytes));
    CHECK(hipMemcpy(d_A, A, size_bytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_B, B, size_bytes, hipMemcpyHostToDevice));

    for (unsigned int i_meta = 0; i_meta < NB_META; i_meta++)
    {
        if ( i_meta == 0 )
        {
            for (unsigned int i = 0; i < nwu; i++)
            {
                sgemm_kernel<<<gridDim, blockDim, 0, hipStreamDefault>>>(size, A, B, C);
            }
        }
        else
        {
            sgemm_kernel<<<gridDim, blockDim, 0, hipStreamDefault>>>(size, A, B, C);
        }

        const double t1 = omp_get_wtime();
        for (unsigned int i = 0; i < nrep; i++)
        {
            sgemm_kernel<<<gridDim, blockDim, 0, hipStreamDefault>>>(size, A, B, C);
        }
        const double t2 = omp_get_wtime();

        tdiff[i_meta] = t2 - t1;
        
    }

    CHECK(hipMemcpy(C, d_C, size_bytes, hipMemcpyDeviceToHost));

    CHECK(hipFree(d_A));
    CHECK(hipFree(d_B));
    CHECK(hipFree(d_C));
    free(A);
    free(B);
    free(C);

    print_measure(block_size, size, nrep, tdiff);
    
    return EXIT_SUCCESS;
}