#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include "driver.h"
#include "common.h"

static int cmp_float (const void *a, const void *b)
{
    const float va = *((float *) a);
    const float vb = *((float *) b);

    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

void print_measure(unsigned int nrep, float tdiff[NB_META])
{
    qsort (tdiff, NB_META, sizeof tdiff[0], cmp_float);

    const float time_min  = (float)tdiff[0]/(float)nrep;
    const float time_med  = (float)tdiff[NB_META/2]/(float)nrep;
    const float stabilite  = (time_med - time_min) * 100.0f / time_min;

    printf("=== Result:\n");
    printf("Time (minimum, ns): %13s %10.0f ns\n", "", time_min * 1e6);
    printf("Time (median, ns):  %13s %10.0f ns\n", "", time_med * 1e6);
    
    if (stabilite >= 10)
        printf("Bad Stability: %18s %10.2f %%\n", "", stabilite);
    else if ( stabilite >= 5 )
        printf("Average Stability: %14s %10.2f %%\n", "", stabilite);
    else
        printf("Good Stability: %17s %10.2f %%\n", "", stabilite);
    
    FILE * output = NULL;
    output = fopen(OUTPUT_FILE, "w");
    if (output != NULL) 
    {
        fprintf(output, "time_min: %.0f\n", time_min * 1e6);
        fprintf(output, "time_med: %.0f\n", time_med * 1e6);
        fprintf(output, "stability: %.2f\n", stabilite);
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

int main (int argc, char* argv[])
{
    unsigned int block_dim;
    unsigned int grid_dim;
    unsigned int nrep;
    unsigned int size;
    float tdiff[NB_META];

    if (argc != 5 && argc != 4) 
    {
        fprintf (stderr, "Usage: %s <size> <block dim> [grid dim] <nb rep>\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 4)
    {
        size = atoi(argv[1]);
        block_dim = atoi(argv[2]);
        grid_dim = (size + block_dim - 1) / block_dim;
        nrep = atoi(argv[3]); 
    }
    else 
    {
        size = atoi(argv[1]);
        block_dim = atoi(argv[2]);
        grid_dim = atoi(argv[3]);
        nrep = atoi(argv[4]); 
    }

    srand(0);

    dim3 blockDim = GET_BLOCK_DIM(block_dim);
    dim3 gridDim = GET_GRID_DIM(grid_dim);

    log_printf("=== Run benchmark with size: %d, blockDim(%d, %d, %d), gridDim(%d, %d, %d), nrep: %d\n", 
                size, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z, nrep);

    driver(size, blockDim, gridDim, nrep, tdiff);

    print_measure(nrep, tdiff);
}