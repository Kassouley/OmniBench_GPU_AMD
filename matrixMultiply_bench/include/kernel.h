#ifndef KERNEL_H
#define KERNEL_H

#ifdef ROCBLAS
void matrixMultiply (unsigned int N, const float* a, const float* b, float* c);
#else
__global__ void matrixMultiply (unsigned int N, const float* a, const float* b, float* c);
#endif
#endif