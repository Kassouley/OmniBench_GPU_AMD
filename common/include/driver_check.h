#ifndef DRIVER_H
#define DRIVER_H

#ifdef ONE_DIM
#define GET_BLOCK_DIM(block_dim) {block_dim}
#define GET_GRID_DIM(grid_dim) {grid_dim}
#elif defined(TWO_DIM)
#define GET_BLOCK_DIM(block_dim) {block_dim, block_dim}
#define GET_GRID_DIM(grid_dim) {grid_dim, grid_dim}
#else
#define GET_BLOCK_DIM(block_dim) { exit(EXIT_FAILURE) }
#define GET_GRID_DIM(grid_dim) { exit(EXIT_FAILURE) }
    
    
#endif

int driver_check(unsigned int size, float** out_cpu, float** out_gpu, unsigned int* size_out_ptr,
                dim3 blockDim, dim3 gridDim);

#endif