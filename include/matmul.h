#ifndef MATMUL_H
#define MATMUL_H

#include <stdbool.h>

// Function should return void since implementation doesn't return anything
void matmul_2d_2d(float* A, float* B, int m, int n, int p, bool transpose_B);
void matmul_2d_1d(float* A, float* B, float* C, int m, int n);

#endif
