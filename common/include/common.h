#ifndef COMMON_H
#define COMMON_H

   
#define HIP_CHECK_CALL(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}

#define OUTPUT_FILE "tmp/measure_tmp.out"

void init_vector(float *v, int size);
void init_matrix(unsigned int row, unsigned int col, float* array);

void current_datetime(char *buffer, size_t size);
void log_printf(const char *format, ...);

#endif