#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include "driver_check.h"

int main (int argc, char* argv[])
{
    unsigned int block_dim;
    unsigned int grid_dim;
    unsigned int size;
    unsigned int only_gpu;
    char*        file_name_cpu = NULL;
    char*        file_name_gpu = NULL;

    if (argc != 5 && argc != 6)
    {
        fprintf (stderr, "Usage: %s <size> <block dim> [grid dim] <output_file> <only_gpu>\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 5)
    {
        size = atoi(argv[1]);
        block_dim = atoi(argv[2]);
        grid_dim = (size + block_dim - 1) / block_dim;
        file_name_cpu = (char*)malloc(256*sizeof(char));
        file_name_gpu = (char*)malloc(256*sizeof(char));
        strcpy(file_name_cpu, argv[3]);
        strcat(file_name_cpu, "_cpu.check_out");
        strcpy(file_name_gpu, argv[3]);
        strcat(file_name_gpu, "_gpu.check_out");
        only_gpu = atoi(argv[4]);
    }
    else 
    {
        size = atoi(argv[1]);
        block_dim = atoi(argv[2]);
        grid_dim = atoi(argv[3]);
        file_name_cpu = (char*)malloc(256*sizeof(char));
        file_name_gpu = (char*)malloc(256*sizeof(char));
        strcpy(file_name_cpu, argv[4]);
        strcat(file_name_cpu, "_cpu.check_out");
        strcpy(file_name_gpu, argv[4]);
        strcat(file_name_gpu, "_gpu.check_out");
        only_gpu = atoi(argv[5]);
    }

    dim3 blockDim = GET_BLOCK_DIM(block_dim);
    dim3 gridDim = GET_GRID_DIM(grid_dim);

    if (only_gpu == 0)
    {
        printf("Check kernel on CPU for size=%d . . .\n",size);
        driver_check_cpu(size, file_name_cpu);
    }
    printf("Check kernel on GPU for size=%d . . .\n",size);
    driver_check_gpu(size, blockDim, gridDim, file_name_gpu);

    free(file_name_cpu);
    free(file_name_gpu);
}