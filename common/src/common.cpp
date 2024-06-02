#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <stdarg.h>
#include <time.h>
#include <fcntl.h>
#include <string.h>
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


void current_datetime(char *buffer, size_t size) 
{
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    strftime(buffer, size, "%Y-%m-%d %H:%M:%S", t);
}

void log_printf(const char *format, ...) 
{
    int file_fd = open("output.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (file_fd == -1) 
    {
        perror("Erreur lors de l'ouverture du fichier");
        exit(EXIT_FAILURE);
    }

    char message_buffer[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(message_buffer, sizeof(message_buffer), format, args);
    va_end(args);

    char datetime_buffer[64];
    current_datetime(datetime_buffer, sizeof(datetime_buffer));

    char log_buffer[1100];
    snprintf(log_buffer, sizeof(log_buffer), "[%s] %s", datetime_buffer, message_buffer);

    write(file_fd, log_buffer, strlen(log_buffer));

    printf("%s", log_buffer);

    close(file_fd);
}