#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include "common.h"

void init_vector(float *v, int size) 
{
    for (int i = 0; i < size; ++i) 
    {
        v[i] = (float)rand() / (float)RAND_MAX; // random values
    }
}

void init_matrix(unsigned int row, unsigned int col, float* array)
{
    for (unsigned int i = 0; i < row; i++)
    {
        for (unsigned int j = 0; j < col; j++)
        {
            array[i*col+j] = (float)rand() / (float)RAND_MAX;
        }
    }
}