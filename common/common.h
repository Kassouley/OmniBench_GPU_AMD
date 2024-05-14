#ifndef COMMON_H
#define COMMON_H

#define NB_META 31
   
#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}

#define OUTPUT_FILE "measure_tmp.out"

void init_vector(float *v, int size);
void init_matrix(unsigned int row, unsigned int col, float* array);
static int cmp_uint64 (const void *a, const void *b);
void print_measure(unsigned int block_size, unsigned int data_size, unsigned int nrep, double tdiff[NB_META]);

#endif