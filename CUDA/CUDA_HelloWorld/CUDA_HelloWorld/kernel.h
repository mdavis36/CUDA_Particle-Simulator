#ifndef KERNEL_H
#define KERNEL_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t findMaxValue(int *i_arr, int *o_arr, unsigned int size, unsigned int block_size, unsigned int grid_size);
#endif
