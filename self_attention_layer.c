#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VOCAB_SIZE 1000
#define EMBEDDING_DIM 512
#define MAX_SEQ_LENGTH 128

// FUNCTION TO COMPUTE DOT PRODUCT
float dot_product(float *a, float *b, int dim) {
    float result = 0.0f;
    for (int i = 0; i < dim; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// FUNCTION TO COMPUTE SOFTMAX
void softmax(float *input, float *output, int length) {
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
void initialize_embedding(float embedding[VOCAB_SIZE][EMBEDDING_DIM]) {
    printf("Initializing Token Embedding Matrix:\n\n");
    for (int i = 0; i < VOCAB_SIZE; i++) {
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            embedding[i][j] = ((float) rand() / RAND_MAX) - 0.5f;
        }
    }
    printf("Token Embedding Matrix Initialized!\n\n");
}

// FUNCTION TO GENERATE POSITIONAL ENCODING
void generate_positional_encoding(float positional_encoding[VOCAB_SIZE][EMBEDDING_DIM]) {
    for (int pos = 0; pos < VOCAB_SIZE; pos++) {
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            if (i % 2 == 0)
                positional_encoding[pos][i] = sinf(pos / powf(10000.0f, (2.0f * i / EMBEDDING_DIM)));
            else
                positional_encoding[pos][i] = cosf(pos / powf(10000.0f, (2.0f * (i - 1) / EMBEDDING_DIM)));
        }
    }
}

// FUNCTION TO LOOKUP EMBEDDING FOR A GIVEN TOKEN INDEX
void lookup_embedding(float embedding[VOCAB_SIZE][EMBEDDING_DIM], int token_index, float *output) {
    for (int i = 0; i < EMBEDDING_DIM; i++) output[i] = embedding[token_index][i];
}

// FUNCTION TO CONCATENATE TOKEN EMBEDDING AND POSITIONAL EMBEDDING
void concatenate_embeddings(float token_embedding[EMBEDDING_DIM],
                            float positional_embedding[EMBEDDING_DIM],
                            float *output) {
    for (int i = 0; i < EMBEDDING_DIM; i++)
        output[i] = token_embedding[i] + positional_embedding[i];
}

/* ----------------------------------------------------------------------
   FIXED GENERALIZED FUNCTIONS BELOW
   ---------------------------------------------------------------------- */

// GENERALIZED MATRIX MULTIPLICATION (supports any M×N × N×P)
void matrix_multiply(float *A, float *B, float *C,
                     int rows_A, int cols_A, int cols_B) {
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            float sum = 0.0f;
            for (int k = 0; k < cols_A; k++) {
                sum += A[i * cols_A + k] * B[k * cols_B + j];
            }
            C[i * cols_B + j] = sum;
        }
    }
}

// GENERALIZED WEIGHT INITIALIZER
void initialize_weight_matrix(float *weight, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        weight[i] = ((float) rand() / RAND_MAX) - 0.5f;
}

/* ---------------------------------------------------------------------- */

// FUNCTION TO COMPUTE SELF-ATTENTION WITH TRAINABLE K, Q, V
void self_attention(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM],
                    float output[MAX_SEQ_LENGTH][EMBEDDING_DIM],
                    int seq_length) {
    float W_Q[EMBEDDING_DIM][EMBEDDING_DIM];
    float W_K[EMBEDDING_DIM][EMBEDDING_DIM];
    float W_V[EMBEDDING_DIM][EMBEDDING_DIM];

    initialize_weight_matrix(&W_Q[0][0], EMBEDDING_DIM, EMBEDDING_DIM);
    initialize_weight_matrix(&W_K[0][0], EMBEDDING_DIM, EMBEDDING_DIM);
    initialize_weight_matrix(&W_V[0][0], EMBEDDING_DIM, EMBEDDING_DIM);

    float Q[MAX_SEQ_LENGTH][EMBEDDING_DIM];
    float K[MAX_SEQ_LENGTH][EMBEDDING_DIM];
    float V[MAX_SEQ_LENGTH][EMBEDDING_DIM];

    matrix_multiply(&input[0][0], &W_Q[0][0], &Q[0][0], seq_length, EMBEDDING_DIM, EMBEDDING_DIM);
    matrix_multiply(&input[0][0], &W_K[0][0], &K[0][0], seq_length, EMBEDDING_DIM, EMBEDDING_DIM);
    matrix_multiply(&input[0][0], &W_V[0][0], &V[0][0], seq_length, EMBEDDING_DIM, EMBEDDING_DIM);

    float attention_scores[MAX_SEQ_LENGTH][MAX_SEQ_LENGTH] = {0};
    float attention_weights[MAX_SEQ_LENGTH][MAX_SEQ_LENGTH] = {0};

    for (int i = 0; i < seq_length; i++)
        for (int j = 0; j < seq_length; j++)
            attention_scores[i][j] = dot_product(Q[i], K[j], EMBEDDING_DIM) / sqrtf(EMBEDDING_DIM);

    for (int i = 0; i < seq_length; i++)
        softmax(attention_scores[i], attention_weights[i], seq_length);

    for (int i = 0; i < seq_length; i++)
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            output[i][j] = 0.0f;
            for (int k = 0; k < seq_length; k++)
                output[i][j] += attention_weights[i][k] * V[k][j];
        }
}

// FIXED FEED-FORWARD NETWORK
void feed_forward(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM],
                  float output[MAX_SEQ_LENGTH][EMBEDDING_DIM],
                  int seq_length) {
    float W1[EMBEDDING_DIM][EMBEDDING_DIM * 4];
    float W2[EMBEDDING_DIM * 4][EMBEDDING_DIM];

    initialize_weight_matrix(&W1[0][0], EMBEDDING_DIM, EMBEDDING_DIM * 4);
    initialize_weight_matrix(&W2[0][0], EMBEDDING_DIM * 4, EMBEDDING_DIM);

    float intermediate[MAX_SEQ_LENGTH][EMBEDDING_DIM * 4];

    matrix_multiply(&input[0][0], &W1[0][0], &intermediate[0][0],
                    seq_length, EMBEDDING_DIM, EMBEDDING_DIM * 4);

    for (int i = 0; i < seq_length; i++)
        for (int j = 0; j < EMBEDDING_DIM * 4; j++)
            intermediate[i][j] = fmaxf(0.0f, intermediate[i][j]); // ReLU

    matrix_multiply(&intermediate[0][0], &W2[0][0], &output[0][0],
                    seq_length, EMBEDDING_DIM * 4, EMBEDDING_DIM);
}

// LAYER NORMALIZATION
void layer_normalization(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM],
                         float output[MAX_SEQ_LENGTH][EMBEDDING_DIM],
                         int seq_length) {
    for (int i = 0; i < seq_length; i++) {
        float mean = 0.0f, variance = 0.0f;
        for (int j = 0; j < EMBEDDING_DIM; j++) mean += input[i][j];
        mean /= EMBEDDING_DIM;
        for (int j = 0; j < EMBEDDING_DIM; j++)
            variance += (input[i][j] - mean) * (input[i][j] - mean);
        variance /= EMBEDDING_DIM;
        for (int j = 0; j < EMBEDDING_DIM; j++)
            output[i][j] = (input[i][j] - mean) / sqrtf(variance + 1e-6f);
    }
}

// MAIN FUNCTION FOR TESTING
int main() {
    float embedding[VOCAB_SIZE][EMBEDDING_DIM];
    float positional_encoding[VOCAB_SIZE][EMBEDDING_DIM];

    initialize_embedding(embedding);
    generate_positional_encoding(positional_encoding);

    int seq_length = 4;
    float input[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};

    for (int i = 0; i < seq_length; i++) {
        float token_embedding[EMBEDDING_DIM];
        float positional_embedding[EMBEDDING_DIM];
        lookup_embedding(embedding, i, token_embedding);
        lookup_embedding(positional_encoding, i, positional_embedding);

        float final_embedding[EMBEDDING_DIM];
        concatenate_embeddings(token_embedding, positional_embedding, final_embedding);
        for (int j = 0; j < EMBEDDING_DIM; j++)
            input[i][j] = final_embedding[j];
    }

    float output[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};
    self_attention(input, output, seq_length);
    feed_forward(output, output, seq_length);

    printf("\n✅ Self-Attention + Feed-Forward completed successfully.\n");
    return 0;
}
