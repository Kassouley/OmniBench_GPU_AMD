#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#ifdef TILE
#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif
__global__ void matrixMultiply (unsigned int N, const float* a, const float* b, float* c)
{
    __shared__ float sub_tile_N[TILE_SIZE][TILE_SIZE];
    __shared__ float sub_tile_M[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx;

    float sum = 0.0;
    for (int i = 0; i < ((N-1)/TILE_SIZE+1); i++)
    {
        int tile_row = i * TILE_SIZE + tx;
        int tile_col = i * TILE_SIZE + ty;
        int curr_l = row * N + tile_row;
        int curr_r = tile_col * N + col;
        if ( tile_row < N &&  row < N )
            sub_tile_M[ty][tx] = a[curr_l];
        else
            sub_tile_M[ty][tx] = 0.0;
        
        if ( tile_col < N &&  col < N )
            sub_tile_N[ty][tx] = b[curr_r];
        else
            sub_tile_N[ty][tx] = 0.0;

        __syncthreads();    

        for (int j = 0; j < TILE_SIZE; j++)
        {
            if ( j + TILE_SIZE * i < N )
            {
                sum += sub_tile_M[ty][j] * sub_tile_N[j][tx];
            }
        }
        __syncthreads();    
    }

    if ( row < N && col < N )
    {
        c[row * N + col] = sum;
    }
}

#elif defined(ROCBLAS)
void matrixMultiply (unsigned int N, const float* a, const float* b, float* c)
{
    const float alpha = 1.0; 
    const float beta = 0.0;
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
            N, N, N, &alpha, b, N, a, N, &beta, c, N);
    rocblas_destroy_handle(handle);
}

#else
__global__ void matrixMultiply (unsigned int N, const float* a, const float* b, float* c)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N)
    {
        float sum = 0.0f;
        for (unsigned int i = 0; i < N; i++)
        {
            sum += a[row * N + i ] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

#endif

