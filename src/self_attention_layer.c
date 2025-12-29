#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "transformer_block.h"

#define VOCAB_SIZE 1000
#define EMBEDDING_DIM_LOCAL 512  // Renamed to avoid conflict with header
#define MAX_SEQ_LENGTH 128

// FUNCTION TO COMPUTE DOT PRODUCT
float dot_product_local(float *a, float *b, int dim) {  // Renamed
    float result = 0.0f;
    for (int i = 0; i < dim; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// FUNCTION TO COMPUTE SOFTMAX
void softmax_local(float *input, float *output, int length) {  // Renamed
    float max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < length; i++) output[i] /= sum;
}

// FUNCTION TO INITIALIZE THE EMBEDDING MATRIX
void initialize_embedding(float embedding[VOCAB_SIZE][EMBEDDING_DIM_LOCAL]) {
    printf("Initializing Token Embedding Matrix:\n\n");
    for (int i = 0; i < VOCAB_SIZE; i++) {
        for (int j = 0; j < EMBEDDING_DIM_LOCAL; j++) {
            embedding[i][j] = ((float) rand() / RAND_MAX) - 0.5f;
        }
    }
    printf("Token Embedding Matrix Initialized!\n\n");
}

// FUNCTION TO GENERATE POSITIONAL ENCODING
void generate_positional_encoding(float positional_encoding[VOCAB_SIZE][EMBEDDING_DIM_LOCAL]) {
    for (int pos = 0; pos < VOCAB_SIZE; pos++) {
        for (int i = 0; i < EMBEDDING_DIM_LOCAL; i++) {
            if (i % 2 == 0)
                positional_encoding[pos][i] = sinf(pos / powf(10000.0f, (2.0f * i / EMBEDDING_DIM_LOCAL)));
            else
                positional_encoding[pos][i] = cosf(pos / powf(10000.0f, (2.0f * (i - 1) / EMBEDDING_DIM_LOCAL)));
        }
    }
}

// FUNCTION TO LOOKUP EMBEDDING FOR A GIVEN TOKEN INDEX
void lookup_embedding(float embedding[VOCAB_SIZE][EMBEDDING_DIM_LOCAL], int token_index, float *output) {
    for (int i = 0; i < EMBEDDING_DIM_LOCAL; i++) output[i] = embedding[token_index][i];
}

// FUNCTION TO CONCATENATE TOKEN EMBEDDING AND POSITIONAL EMBEDDING
void concatenate_embeddings(float token_embedding[EMBEDDING_DIM_LOCAL],
                            float positional_embedding[EMBEDDING_DIM_LOCAL],
                            float *output) {
    for (int i = 0; i < EMBEDDING_DIM_LOCAL; i++)
        output[i] = token_embedding[i] + positional_embedding[i];
}

// GENERALIZED WEIGHT INITIALIZER
void initialize_weight_matrix(float *weight, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        weight[i] = ((float) rand() / RAND_MAX) - 0.5f;
}

// FLEXIBLE FLOAT VERSION OF MATRIX MULTIPLY
static void matrix_multiply_float_flexible(float *A, float *B, float *result,
                                          int rowsA, int colsA, int colsB,
                                          int strideA, int strideB, int strideResult) {
    // Initialize result matrix to 0
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            result[i * strideResult + j] = 0.0f;
        }
    }

    // Perform matrix multiplication
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            float sum = 0.0f;
            for (int k = 0; k < colsA; k++) {
                sum += A[i * strideA + k] * B[k * strideB + j];
            }
            result[i * strideResult + j] = sum;
        }
    }
}

// Convenience wrapper for square matrices (512x512)
static void matrix_multiply_float_512x512(float A[][EMBEDDING_DIM_LOCAL], 
                                         float B[][EMBEDDING_DIM_LOCAL], 
                                         float result[][EMBEDDING_DIM_LOCAL],
                                         int rowsA, int colsA, int colsB) {
    matrix_multiply_float_flexible(&A[0][0], &B[0][0], &result[0][0],
                                   rowsA, colsA, colsB,
                                   EMBEDDING_DIM_LOCAL, EMBEDDING_DIM_LOCAL, EMBEDDING_DIM_LOCAL);
}

// FUNCTION TO COMPUTE SELF-ATTENTION WITH TRAINABLE K, Q, V
void self_attention(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM_LOCAL],
                    float output[MAX_SEQ_LENGTH][EMBEDDING_DIM_LOCAL],
                    int seq_length) {
    float W_Q[EMBEDDING_DIM_LOCAL][EMBEDDING_DIM_LOCAL];
    float W_K[EMBEDDING_DIM_LOCAL][EMBEDDING_DIM_LOCAL];
    float W_V[EMBEDDING_DIM_LOCAL][EMBEDDING_DIM_LOCAL];

    initialize_weight_matrix(&W_Q[0][0], EMBEDDING_DIM_LOCAL, EMBEDDING_DIM_LOCAL);
    initialize_weight_matrix(&W_K[0][0], EMBEDDING_DIM_LOCAL, EMBEDDING_DIM_LOCAL);
    initialize_weight_matrix(&W_V[0][0], EMBEDDING_DIM_LOCAL, EMBEDDING_DIM_LOCAL);

    float Q[MAX_SEQ_LENGTH][EMBEDDING_DIM_LOCAL];
    float K[MAX_SEQ_LENGTH][EMBEDDING_DIM_LOCAL];
    float V[MAX_SEQ_LENGTH][EMBEDDING_DIM_LOCAL];

    // Use the 512x512 version for self-attention
    matrix_multiply_float_512x512(input, W_Q, Q, seq_length, EMBEDDING_DIM_LOCAL, EMBEDDING_DIM_LOCAL);
    matrix_multiply_float_512x512(input, W_K, K, seq_length, EMBEDDING_DIM_LOCAL, EMBEDDING_DIM_LOCAL);
    matrix_multiply_float_512x512(input, W_V, V, seq_length, EMBEDDING_DIM_LOCAL, EMBEDDING_DIM_LOCAL);

    float attention_scores[MAX_SEQ_LENGTH][MAX_SEQ_LENGTH] = {{0}};
    float attention_weights[MAX_SEQ_LENGTH][MAX_SEQ_LENGTH] = {{0}};

    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < seq_length; j++) {
            attention_scores[i][j] = dot_product_local(Q[i], K[j], EMBEDDING_DIM_LOCAL) / sqrtf(EMBEDDING_DIM_LOCAL);
        }
    }

    for (int i = 0; i < seq_length; i++) {
        softmax_local(attention_scores[i], attention_weights[i], seq_length);
    }

    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < EMBEDDING_DIM_LOCAL; j++) {
            output[i][j] = 0.0f;
            for (int k = 0; k < seq_length; k++) {
                output[i][j] += attention_weights[i][k] * V[k][j];
            }
        }
    }
}

// FIXED FEED-FORWARD NETWORK
void feed_forward(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM_LOCAL],
                  float output[MAX_SEQ_LENGTH][EMBEDDING_DIM_LOCAL],
                  int seq_length) {
    float W1[EMBEDDING_DIM_LOCAL][EMBEDDING_DIM_LOCAL * 4];
    float W2[EMBEDDING_DIM_LOCAL * 4][EMBEDDING_DIM_LOCAL];

    initialize_weight_matrix(&W1[0][0], EMBEDDING_DIM_LOCAL, EMBEDDING_DIM_LOCAL * 4);
    initialize_weight_matrix(&W2[0][0], EMBEDDING_DIM_LOCAL * 4, EMBEDDING_DIM_LOCAL);

    float intermediate[MAX_SEQ_LENGTH][EMBEDDING_DIM_LOCAL * 4];

    // First multiplication: input (seq_length x 512) * W1 (512 x 2048) = intermediate (seq_length x 2048)
    matrix_multiply_float_flexible(&input[0][0], &W1[0][0], &intermediate[0][0],
                                   seq_length, EMBEDDING_DIM_LOCAL, EMBEDDING_DIM_LOCAL * 4,
                                   EMBEDDING_DIM_LOCAL, EMBEDDING_DIM_LOCAL * 4, EMBEDDING_DIM_LOCAL * 4);

    // Apply ReLU activation
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < EMBEDDING_DIM_LOCAL * 4; j++) {
            intermediate[i][j] = fmaxf(0.0f, intermediate[i][j]);
        }
    }

    // Second multiplication: intermediate (seq_length x 2048) * W2 (2048 x 512) = output (seq_length x 512)
    matrix_multiply_float_flexible(&intermediate[0][0], &W2[0][0], &output[0][0],
                                   seq_length, EMBEDDING_DIM_LOCAL * 4, EMBEDDING_DIM_LOCAL,
                                   EMBEDDING_DIM_LOCAL * 4, EMBEDDING_DIM_LOCAL, EMBEDDING_DIM_LOCAL);
}

// LAYER NORMALIZATION
void layer_normalization(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM_LOCAL],
                         float output[MAX_SEQ_LENGTH][EMBEDDING_DIM_LOCAL],
                         int seq_length) {
    for (int i = 0; i < seq_length; i++) {
        float mean = 0.0f, variance = 0.0f;
        
        // Calculate mean
        for (int j = 0; j < EMBEDDING_DIM_LOCAL; j++) {
            mean += input[i][j];
        }
        mean /= EMBEDDING_DIM_LOCAL;
        
        // Calculate variance
        for (int j = 0; j < EMBEDDING_DIM_LOCAL; j++) {
            variance += (input[i][j] - mean) * (input[i][j] - mean);
        }
        variance /= EMBEDDING_DIM_LOCAL;
        
        // Normalize
        for (int j = 0; j < EMBEDDING_DIM_LOCAL; j++) {
            output[i][j] = (input[i][j] - mean) / sqrtf(variance + 1e-6f);
        }
    }
}