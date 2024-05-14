#include <hip/hip_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
   
#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}

#ifdef BASIC
__global__ void saxpy_kernel(const float a, const float* d_x, float* d_y, const unsigned int size)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < size)
    {
        d_y[i] = a * d_x[i] + d_y[i];
    }
}
#else
__global__ void saxpy_kernel(const float a, const float* d_x, float* d_y, const unsigned int size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < size; 
         i += blockDim.x * gridDim.x) 
      {
          d_y[i] = a * d_x[i] + d_y[i];
      }
}
#endif


void init_vector(float *v, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) 
    {
        v[i] = (float)(rand() % 10); // random values
    }
}

int main(int argc, char** argv)
{
    unsigned int size = 10000000;
    unsigned int block_size = 1024;
    if (argc == 3)
    {
        size = atoi(argv[1]);
        block_size = atoi(argv[2]);
    }
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
    
    float* yf = NULL;
    yf = (float *)malloc(size_bytes);

    printf("Start : nb thread : %d, nb block : %d\n", block_size, grid_size);

    saxpy_kernel<<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(a, d_x, d_y, size);
    CHECK(hipGetLastError());
    CHECK(hipMemcpy(yf, d_y, size_bytes, hipMemcpyDeviceToHost));

    for( int i = 0; i < size; i++ )
    {
        y[i] = a * x[i] + y[i];
    }

    for( int i = 0; i < size; i++ )
    {
        if (y[i] != yf[i])
            printf("WRONG %d %f %f\n",i, y[i], yf[i]);
            break;
    }

    // for( int i = 0; i < 10000; i++ )
    // {
    //     saxpy_kernel<<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(a, d_x, d_y, size);
    //     CHECK(hipGetLastError());
    // }

    printf("End\n");

    CHECK(hipFree(d_x));
    CHECK(hipFree(d_y));
}