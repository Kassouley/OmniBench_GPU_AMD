#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include "common.h"

void init_vector(float *v, int size) 
{
    #pragma omp parallel for
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

static int cmp_uint64 (const void *a, const void *b)
{
    const uint64_t va = *((uint64_t *) a);
    const uint64_t vb = *((uint64_t *) b);

    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

void print_measure(unsigned int block_size, unsigned int data_size, unsigned int nrep, double tdiff[NB_META])
{
    FILE * output = NULL;

    qsort (tdiff, NB_META, sizeof tdiff[0], cmp_uint64);

    const double time_min  = (double)tdiff[0]/(double)nrep*10e9;
    const double time_med  = (double)tdiff[NB_META/2]/(double)nrep*10e9;
    const float stabilite  = (tdiff[NB_META/2] - tdiff[0]) * 100.0f / tdiff[0];

    double rate = 0.0, drate = 0.0;
    double t_tmp = 0.0;
    for (unsigned int i = 0; i < NB_META; i++)
    {
        t_tmp = (double)tdiff[i]/(double)nrep * 10e9;
        rate += t_tmp;
        drate += t_tmp*t_tmp;
    }
    rate /= (double)(NB_META);
    drate = sqrt(drate / (double)(NB_META) - (rate * rate));
  
    printf("-----------------------------------------------------\n");

    printf("Time (minimum, ns): %13s %10.5f ns\n", "", time_min);
    printf("Time (median, ns):  %13s %10.5f ns\n", "", time_med);
    
    if (stabilite >= 10)
        printf("Bad Stability: %18s %10.2f %%\n", "", stabilite);
    else if ( stabilite >= 5 )
        printf("Average Stability: %14s %10.2f %%\n", "", stabilite);
    else
        printf("Good Stability: %17s %10.2f %%\n", "", stabilite);

    printf("\033[1m%s %9s \033[42m%10.0lf +- %.0lf ns\033[0m\n",
        "Average time:", "", rate, drate);
    printf("-----------------------------------------------------\n");
    

    output = fopen(OUTPUT_FILE, "a");
    if (output != NULL) 
    {
        fprintf(output, "%d, %d, %f, %f, %f, %f\n", 
                data_size,
                block_size,
                rate, 
                time_min, 
                time_med,
                stabilite);
        fclose(output);
    }
    else
    {
        char cwd[1028];
        if (getcwd(cwd, sizeof(cwd)) != NULL) 
        {
            printf("Couldn't open '%s/%s' file\n Measure not saved\n", cwd, OUTPUT_FILE);
        }
    }
}