#ifndef KERNEL_H
#define KERNEL_H

__global__ void sgemm_kernel (unsigned int N, const float* a, const float* b, float* c);
void sgemm_cpu (unsigned int N, const float* a, const float* b, float* c);
#endif