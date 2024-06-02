#ifndef KERNEL_H
#define KERNEL_H

__global__ void saxpy(const float a, const float* d_x, float* d_y, const unsigned int size);
#endif