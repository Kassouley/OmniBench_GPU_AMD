#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include "driver.h"

int main (int argc, char* argv[])
{
    unsigned int block_dim;
    unsigned int grid_dim;
    unsigned int nb_rep;
    unsigned int size;

    if (argc != 5 && argc != 4) 
    {
        fprintf (stderr, "Usage: %s <size> <block dim> [grid dim] <nb rep>\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 4)
    {
        size = atoi(argv[1]);
        block_dim = atoi(argv[2]);
        grid_dim = (size + block_dim - 1) / block_dim;
        nb_rep = atoi(argv[3]); 
    }
    else 
    {
        size = atoi(argv[1]);
        block_dim = atoi(argv[2]);
        grid_dim = atoi(argv[3]);
        nb_rep = atoi(argv[4]); 
    }

    srand(0);

    dim3 blockDim = GET_BLOCK_DIM(block_dim);
    dim3 gridDim = GET_GRID_DIM(grid_dim);

    driver(size, blockDim, gridDim, nb_rep);
}