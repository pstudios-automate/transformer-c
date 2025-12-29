// matrix_operations.c
#include "matrix_operations.h"
#include <stdio.h>

/////////////////////// MATRIX ADDITION FUNCTIONS ////////////////////////////

void add_matrices(float matrix1[][MATRIX_SIZE], 
                  double matrix2[][MATRIX_SIZE], 
                  double result_matrix[][MATRIX_SIZE], 
                  int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_matrix[i][j] = (double)matrix1[i][j] + matrix2[i][j];
        }
    }
}

void add_double_matrices(double matrix1[][MATRIX_SIZE], 
                        double matrix2[][MATRIX_SIZE], 
                        double result_matrix[][MATRIX_SIZE], 
                        int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_matrix[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
}

void add_float_matrices(float matrix1[][MATRIX_SIZE], 
                       float matrix2[][MATRIX_SIZE], 
                       float result_matrix[][MATRIX_SIZE], 
                       int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_matrix[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
}

// Utility function to print addition results
void print_addition_result(const char* name, double matrix[][MATRIX_SIZE], int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.4f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}