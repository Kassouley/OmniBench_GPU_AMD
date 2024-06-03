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

#define NB_META 31

#define kernelBenchmark(tdiff, nwu, nrep, KERNEL, ...)   \
    hipEvent_t start, stop;  \
    HIP_CHECK_CALL(hipEventCreate(&start));  \
    HIP_CHECK_CALL(hipEventCreate(&stop));  \
    for (unsigned int i_meta = 0; i_meta < NB_META; i_meta++)  \
    {  \
        if ( i_meta == 0 )  \
        {  \
            for (unsigned int i = 0; i < nwu; i++)  \
            {  \
                hipLaunchKernelGGL(KERNEL, __VA_ARGS__);  \
            }  \
        }  \
        else  \
        {  \
            hipLaunchKernelGGL(KERNEL, __VA_ARGS__);  \
        }  \
        HIP_CHECK_CALL(hipDeviceSynchronize());  \
        HIP_CHECK_CALL(hipEventRecord(start, 0));  \
        for (unsigned int i = 0; i < nrep; i++)  \
        {  \
            hipLaunchKernelGGL(KERNEL, __VA_ARGS__);  \
        }  \
        HIP_CHECK_CALL(hipDeviceSynchronize());  \
        HIP_CHECK_CALL(hipEventRecord(stop, 0));  \
        HIP_CHECK_CALL(hipEventSynchronize(stop));  \
        float milliseconds = 0.0;  \
        HIP_CHECK_CALL(hipEventElapsedTime(&milliseconds, start, stop));  \
        tdiff[i_meta] = milliseconds;  \
    } \
    HIP_CHECK_CALL(hipEventDestroy(start));\
    HIP_CHECK_CALL(hipEventDestroy(stop));


int driver (const unsigned int size, dim3 blockDim, dim3 gridDim, const unsigned int nrep, float tdiff[NB_META]);

#endif