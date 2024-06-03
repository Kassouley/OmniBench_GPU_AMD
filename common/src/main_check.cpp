#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include "driver_check.h"

#define MAX_ERROR_ALLOWED -2

int main (int argc, char* argv[])
{
    unsigned int size;
    unsigned int block_dim;
    unsigned int grid_dim;

    if (argc != 3 && argc != 4)
    {
        fprintf (stderr, "Usage: %s <size> <block dim> [grid dim]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 3)
    {
        size = atoi(argv[1]);
        block_dim = atoi(argv[2]);
        grid_dim = (size + block_dim - 1) / block_dim;
    }
    else 
    {
        size = atoi(argv[1]);
        block_dim = atoi(argv[2]);
        grid_dim = atoi(argv[3]);
    }

    dim3 blockDim = GET_BLOCK_DIM(block_dim);
    dim3 gridDim = GET_GRID_DIM(grid_dim);

    float* out_cpu = NULL;
    float* out_gpu = NULL;
    unsigned int size_out = 0;
    float max_error = -1;
    float error = 0;
    double exponent = 0;

    driver_check(size, &out_cpu, &out_gpu, &size_out, blockDim, gridDim);

    printf("=== Compare output . . .\n");
    fflush(stdout);
    for(unsigned int i = 0; i < size_out; i++)
    {
        error = fabs(out_cpu[i] - out_gpu[i]);
        if (error > max_error)
        {
            max_error = error;
            if (floor(log10(max_error)) >= MAX_ERROR_ALLOWED)
                break;
        }
    }
    if (max_error == 0) 
    {
        printf("=== Kernel output is \u001b[42mcorrect\033[0m.\n");
    } 
    else if (max_error == -1)
    {
        printf("=== Error : No compare done.\n");
        return EXIT_FAILURE;
    }
    else 
    {
        exponent = floor(log10(max_error));
        if (exponent < MAX_ERROR_ALLOWED) {
            printf("=== Kernel output is \u001b[42mcorrect\033[0m with a max error of 10^%.0f\n", exponent);
        } else {
            printf("=== Kernel output is \u001b[41mincorrect\033[0m (error max of 10^%.0f)\n", exponent);
        }
    }

    free(out_cpu);
    free(out_gpu);
}