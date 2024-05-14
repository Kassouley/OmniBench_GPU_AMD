#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <omp.h>
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
    
    const size_t size_bytes = size * sizeof(float);
    const unsigned int grid_size = (size + block_size - 1) / block_size;
    const float a = 2.0f;

    float* x = NULL;
    float* y = NULL;
    x = (float *)malloc(size_bytes);
    y = (float *)malloc(size_bytes);
    init_vector(x, size);
    init_vector(y, size);

    float* d_x = NULL;
    float* d_y = NULL;
    CHECK(hipMalloc(&d_x, size_bytes));
    CHECK(hipMalloc(&d_y, size_bytes));
    CHECK(hipMemcpy(d_x, x, size_bytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_y, y, size_bytes, hipMemcpyHostToDevice));
    

    for (unsigned int i_meta = 0; i_meta < NB_META; i_meta++)
    {
        if ( i_meta == 0 )
        {
            for (unsigned int i = 0; i < nwu; i++)
            {
                saxpy_kernel<<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(a, d_x, d_y, size);
            }
        }
        else
        {
            saxpy_kernel<<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(a, d_x, d_y, size);
        }

        const double t1 = omp_get_wtime();
        for (unsigned int i = 0; i < nrep; i++)
        {
            saxpy_kernel<<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(a, d_x, d_y, size);
        }
        const double t2 = omp_get_wtime();

        tdiff[i_meta] = t2 - t1;
        
    }

    CHECK(hipFree(d_x));
    CHECK(hipFree(d_y));

    free(x);
    free(y);

    print_measure(block_size, size, nrep, tdiff);
    
    return EXIT_SUCCESS;
}