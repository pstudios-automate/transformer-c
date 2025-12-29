// matrix_operations.h
#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <stdio.h>

#define MATRIX_SIZE 2

// Matrix addition functions
void add_matrices(float matrix1[][MATRIX_SIZE], 
                  double matrix2[][MATRIX_SIZE], 
                  double result_matrix[][MATRIX_SIZE], 
                  int rows, int cols);

void add_double_matrices(double matrix1[][MATRIX_SIZE], 
                        double matrix2[][MATRIX_SIZE], 
                        double result_matrix[][MATRIX_SIZE], 
                        int rows, int cols);

void add_float_matrices(float matrix1[][MATRIX_SIZE], 
                       float matrix2[][MATRIX_SIZE], 
                       float result_matrix[][MATRIX_SIZE], 
                       int rows, int cols);

// Utility function to print addition results
void print_addition_result(const char* name, double matrix[][MATRIX_SIZE], int rows, int cols);

#endif