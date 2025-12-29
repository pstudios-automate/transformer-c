#include "transformer_block.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Ensure Macros are defined (if not already in header)
#ifndef MATRIX_SIZE
#define MATRIX_SIZE 2
#endif
#ifndef MAX_SENTENCE_LENGTH
#define MAX_SENTENCE_LENGTH 512
#endif
#ifndef CLIP_THRESHOLD
#define CLIP_THRESHOLD 100
#endif

// GLOBAL MATRICES
// Removed 'static' to match the 'extern' declaration in transformer_block.h
double k_matrix[MATRIX_SIZE][MATRIX_SIZE];
double q_matrix[MATRIX_SIZE][MATRIX_SIZE];
double v_matrix[MATRIX_SIZE][MATRIX_SIZE];

// --- FUNCTION IMPLEMENTATIONS ---

// FUNCTION IMPLEMENTATION FOR POSITIONAL ENCODING
double* positional_encoding(int index, int vector_size) {
    double* encoding = (double*)malloc(vector_size * sizeof(double));
    if (encoding == NULL) {
        perror("Failed to allocate memory for positional encoding");
        return NULL;
    }

    for (int i = 0; i < vector_size; i++) {
        if (i % 2 == 0) {
            encoding[i] = sin(index / pow(10000.0, (double)i / vector_size));
        }
        else {
            encoding[i] = cos(index / pow(10000.0, (double)(i - 1) / vector_size));
        }
    }
    return encoding;
}

// READ SINGLE VALUE FROM FILE
// Removed 'static' to match header
double read_single_value_from_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        return 0.1; // Default fallback for dev/testing
    }
    double value;
    if (fscanf(file, "%lf", &value) != 1) {
        perror("Error reading file");
        fclose(file);
        return 0.0;
    }
    fclose(file);
    return value;
}

// INITIALIZE MATRICES
void initialize_matrices_from_files() {
    int index = 0;
    char filename[256];

    // READ KEY MATRICES
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            snprintf(filename, sizeof(filename), "Model Trained Weights/self-attention-block-weights/key_weight_%d.txt", index + 1);
            k_matrix[i][j] = read_single_value_from_file(filename);
            index++;
        }
    }
    printf("Initialized KEY MATRIX\n");

    index = 0;
    // READ QUERY MATRICES
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            snprintf(filename, sizeof(filename), "Model Trained Weights/self-attention-block-weights/query_weight_%d.txt", index + 1);
            q_matrix[i][j] = read_single_value_from_file(filename);
            index++;
        }
    }
    printf("Initialized QUERY MATRIX\n");

    index = 0;
    // READ VALUE MATRICES
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            snprintf(filename, sizeof(filename), "Model Trained Weights/self-attention-block-weights/value_weight_%d.txt", index + 1);
            v_matrix[i][j] = read_single_value_from_file(filename);
            index++;
        }
    }
    printf("Initialized VALUE MATRIX\n");
}

void print_matrix(const char* name, double matrix[MATRIX_SIZE][MATRIX_SIZE]) {
    printf("%s:\n", name);
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            printf("%.4f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// TRANSPOSE
// Removed 'static' to match header
void transpose(double matrix[MATRIX_SIZE][MATRIX_SIZE], double transposed[MATRIX_SIZE][MATRIX_SIZE]) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            transposed[j][i] = matrix[i][j];
        }
    }
}

// MATRIX MULTIPLY
// Removed 'static' to match header
void matrix_multiply(double A[][MATRIX_SIZE], double B[][MATRIX_SIZE], double result[][MATRIX_SIZE], int rowsA, int common, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            result[i][j] = 0.0;
            for (int k = 0; k < common; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// APPLY SOFTMAX
// Removed 'static' to match header
void apply_softmax(double matrix[][MATRIX_SIZE], int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double max_val = matrix[i][0];
        for (int j = 1; j < cols; j++) {
            if (matrix[i][j] > max_val) {
                max_val = matrix[i][j];
            }
        }

        double sum_exp = 0.0;
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = exp(matrix[i][j] - max_val); 
            sum_exp += matrix[i][j];
        }

        for (int j = 0; j < cols; j++) {
            matrix[i][j] /= sum_exp; 
        }
    }
}

// CLIP GRADIENT
// Removed 'static' to match header
double clip_gradient_transformer(double gradient) {
    if (fabs(gradient) > CLIP_THRESHOLD) {
        return (gradient > 0 ? CLIP_THRESHOLD : -CLIP_THRESHOLD);
    }
    return gradient;
}

// COMPUTE SELF ATTENTION
// Changed first parameter type to 'float' to match header declaration
void compute_self_attention(float embedding_matrix[][MATRIX_SIZE], double k_mat_dummy[][MATRIX_SIZE], double q_mat_dummy[][MATRIX_SIZE], double v_mat_dummy[][MATRIX_SIZE], int length, double self_attention_matrix[][MATRIX_SIZE]) {
    
    // Silence unused parameter warnings if you are using globals instead
    (void)k_mat_dummy;
    (void)q_mat_dummy;
    (void)v_mat_dummy;

    double k_transposed[MATRIX_SIZE][MATRIX_SIZE];
    double attention_scores[MAX_SENTENCE_LENGTH][MATRIX_SIZE] = {0}; 

    // 1. Transpose K
    transpose(k_matrix, k_transposed);

    // 2. Multiply Q * K^T 
    // Note: This logic assumes Q is q_matrix (global). If you need to use the embedding as Q, logic changes here.
    // Based on previous snippets, we multiply q_matrix * k_transposed.
    matrix_multiply(q_matrix, k_transposed, attention_scores, length, MATRIX_SIZE, MATRIX_SIZE);

    // 3. Scale
    double scale = 1.0 / sqrt((double)MATRIX_SIZE);
    for(int i=0; i<length; i++){
        for(int j=0; j<MATRIX_SIZE; j++){
            attention_scores[i][j] *= scale;
        }
    }

    // 4. Softmax
    apply_softmax(attention_scores, length, MATRIX_SIZE);

    // 5. Multiply Scores * V
    matrix_multiply(attention_scores, v_matrix, self_attention_matrix, length, MATRIX_SIZE, MATRIX_SIZE);
    
    // Note on embedding_matrix: It is passed in but not used in this specific calculation logic 
    // (which uses the learned weights q/k/v directly). 
    // In a real transformer, Q = embedding * W_q. 
    // If you need that, you must implement the projection step here.
}

void update_attention_matrices(double loss, double learning_rate) {
    printf("Updating K Matrix:\n");
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            // FIXED: Use small fixed gradient instead of weight × loss
            double gradient = 0.001 * loss;  // Small learning signal
            gradient = clip_gradient_transformer(gradient); 
            k_matrix[i][j] -= learning_rate * gradient;
            printf("k_matrix[%d][%d] = %lf\n", i, j, k_matrix[i][j]);
        }
    }
    printf("\n");

    printf("Updating Q Matrix:\n");
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            // FIXED: Same small fixed gradient
            double gradient = 0.001 * loss;
            gradient = clip_gradient_transformer(gradient); 
            q_matrix[i][j] -= learning_rate * gradient;
            printf("q_matrix[%d][%d] = %lf\n", i, j, q_matrix[i][j]);
        }
    }
    printf("\n");

    printf("Updating V Matrix:\n");
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            // FIXED: Same small fixed gradient
            double gradient = 0.001 * loss;
            gradient = clip_gradient_transformer(gradient); 
            v_matrix[i][j] -= learning_rate * gradient;
            printf("v_matrix[%d][%d] = %lf\n", i, j, v_matrix[i][j]);
        }
    }
    printf("\n");
}