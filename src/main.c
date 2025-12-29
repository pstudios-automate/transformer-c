// src/main.c
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>

#define MAX_SENTENCE_LENGTH 512
#define MATRIX_SIZE 2
#define EMBEDDING_DIM 2
#define LEARNING_RATE 0.01

/* SELF CREATED HEADER FILES */
#include "Data_Loading_Cleaning.h"
#include "Tokenizer.h"
#include "Data_Preprocessing.h"
#include "transformer_block.h"
#include "feed_forward_layer.h"
#include "activation_functions.h"
#include "backpropagation.h"
#include "matrix_operations.h"
#include "model_manager.h"
#include "interactive_repl.h"  // <-- ONLY ONE COPY!

// ============================================================================
// ============================================================================
// REPL DISPLAY FUNCTIONS
// ============================================================================

void print_header(const char* title) {
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║ %-60s ║\n", title);
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
}

void print_section(const char* title) {
    printf("\n┌──────────────────────────────────────────────────────────────────┐\n");
    printf("│ %-60s │\n", title);
    printf("└──────────────────────────────────────────────────────────────────┘\n");
}

void print_subsection(const char* title) {
    printf("\n  ┌──────────────────────────────────────────────────────────────┐\n");
    printf("  │ %-58s │\n", title);
    printf("  └──────────────────────────────────────────────────────────────┘\n");
}

void print_info(const char* format, ...) {
    va_list args;
    va_start(args, format);
    printf("  ℹ️  ");
    vprintf(format, args);
    printf("\n");
    va_end(args);
}

void print_success(const char* format, ...) {
    va_list args;
    va_start(args, format);
    printf("  ✅ ");
    vprintf(format, args);
    printf("\n");
    va_end(args);
}

void print_warning(const char* format, ...) {
    va_list args;
    va_start(args, format);
    printf("  ⚠️  ");
    vprintf(format, args);
    printf("\n");
    va_end(args);
}

void print_error(const char* format, ...) {
    va_list args;
    va_start(args, format);
    printf("  ❌ ");
    vprintf(format, args);
    printf("\n");
    va_end(args);
}

void print_progress_bar(int current, int total, const char* label) {
    int bar_width = 50;
    float progress = (float)current / total;
    int pos = bar_width * progress;
    
    printf("  ");
    printf("%s [", label);
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) printf("█");
        else if (i == pos) printf("▌");
        else printf(" ");
    }
    printf("] %d/%d (%.1f%%)\n", current, total, progress * 100.0);
}

void print_matrix_preview(const char* name, float matrix[][2], int rows_to_show, int cols) {
    printf("\n  📊 %s (first %d rows):\n", name, rows_to_show);
    printf("  ┌─────────────────────────────────────┐\n");
    for (int i = 0; i < rows_to_show && i < 10; i++) {
        printf("  │ ");
        for (int j = 0; j < cols; j++) {
            printf("%8.4f ", matrix[i][j]);
        }
        printf(" │\n");
    }
    printf("  └─────────────────────────────────────┘\n");
}

void print_token_mapping(const char* token, int id, double embedding[2]) {
    printf("  🏷️  %-15s → ID: %-4d → Embedding: [%8.4f, %8.4f]\n", 
           token, id, embedding[0], embedding[1]);
}

// ============================================================================
// WEIGHT MANAGEMENT
// ============================================================================

void create_weight_directories_and_files() {
    print_section("WEIGHT INITIALIZATION");
    
    // Seed random number generator
    srand(time(NULL));
    
    // Create directories
    #ifdef _WIN32
        system("mkdir \"Model Trained Weights\" 2>nul");
        system("mkdir \"Model Trained Weights\\self-attention-block-weights\" 2>nul");
        system("mkdir \"Model Trained Weights\\Semi_Final Multi-layered perceptron Weights\" 2>nul");
        system("mkdir \"Model Trained Weights\\Final Multi-layered perceptron weights\" 2>nul");
        print_success("Created weight directories");
    #else
        system("mkdir -p \"Model Trained Weights/self-attention-block-weights\" 2>/dev/null");
        system("mkdir -p \"Model Trained Weights/Semi_Final Multi-layered perceptron Weights\" 2>/dev/null");
        system("mkdir -p \"Model Trained Weights/Final Multi-layered perceptron weights\" 2>/dev/null");
        print_success("Created weight directories");
    #endif
    
    // Create self-attention weights (2x2 matrices = 4 weights each)
    print_info("Creating self-attention weights...");
    for (int matrix_type = 0; matrix_type < 3; matrix_type++) {
        const char* types[] = {"key", "query", "value"};
        for (int i = 1; i <= 4; i++) {
            char filename[256];
            snprintf(filename, sizeof(filename), 
                    "Model Trained Weights/self-attention-block-weights/%s_weight_%d.txt", 
                    types[matrix_type], i);
            
            FILE* file = fopen(filename, "r");
            if (file == NULL) {
                file = fopen(filename, "w");
                if (file) {
                    double weight = ((double)rand() / RAND_MAX - 0.5) * 0.1;
                    fprintf(file, "%.6f", weight);
                    fclose(file);
                }
            } else {
                fclose(file);
            }
        }
    }
    print_success("Self-attention weights ready");
    
    // Create semi-final layer weights (512 * 65 = 33280 weights)
    print_info("Creating semi-final layer weights...");
    int semi_final_count = 512 * 65;
    for (int i = 1; i <= semi_final_count; i++) {
        if (i % 5000 == 0) {
            print_progress_bar(i, semi_final_count, "Semi-final weights");
        }
        
        char filename[256];
        snprintf(filename, sizeof(filename), 
                "Model Trained Weights/Semi_Final Multi-layered perceptron Weights/weight_%d.txt", i);
        
        FILE* file = fopen(filename, "r");
        if (file == NULL) {
            file = fopen(filename, "w");
            if (file) {
                double weight = ((double)rand() / RAND_MAX - 0.5) * 0.01;
                fprintf(file, "%.6f", weight);
                fclose(file);
            }
        } else {
            fclose(file);
        }
    }
    print_success("Semi-final layer weights ready (%d weights)", semi_final_count);
    
    // Create final layer weights (65 * 2 = 130 weights)
    print_info("Creating final layer weights...");
    int final_count = 65 * 2;
    for (int i = 1; i <= final_count; i++) {
        if (i % 10 == 0) {
            print_progress_bar(i, final_count, "Final layer weights");
        }
        
        char filename[256];
        snprintf(filename, sizeof(filename), 
                "Model Trained Weights/Final Multi-layered perceptron weights/weight_%d.txt", i);
        
        FILE* file = fopen(filename, "r");
        if (file == NULL) {
            file = fopen(filename, "w");
            if (file) {
                double weight = ((double)rand() / RAND_MAX - 0.5) * 0.01;
                fprintf(file, "%.6f", weight);
                fclose(file);
            }
        } else {
            fclose(file);
        }
    }
    print_success("Final layer weights ready (%d weights)", final_count);
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

void process_training_data(char **sentences, int num_sentences, int ***training_data, int *training_data_count) {
    *training_data = malloc(num_sentences * sizeof(int*));
    *training_data_count = 0;

    for(int i = 0; i < num_sentences; i++) {
        char* sentence = sentences[i];
        
        // 1) CONVERT SENTENCE INTO AN ARRAY OF STRINGS
        char* words[MAX_SENTENCE_LENGTH];
        int word_count = 0;
        
        char* word = strtok(sentence, " ");
        while(word != NULL && word_count < MAX_SENTENCE_LENGTH) {
            words[word_count++] = word;
            word = strtok(NULL, " ");
        }

        // 2) CONVERT SENTENCE INTO AN ARRAY OF INTEGERS
        int* token_array = malloc(MAX_SENTENCE_LENGTH * sizeof(int));
        for(int j = 0; j < word_count; j++) {
            token_array[j] = getTokenId(words[j]);
        }

        // 3) TRIM OR PAD THE ARRAY TO EXACTLY 512 ELEMENTS
        if(word_count > MAX_SENTENCE_LENGTH) {
            word_count = MAX_SENTENCE_LENGTH;
        } else if(word_count < MAX_SENTENCE_LENGTH) {
            for(int j = word_count; j < MAX_SENTENCE_LENGTH; j++) {
                token_array[j] = 0;
            }
        }

        // APPEND THE ARRAY TO TRAINING DATA
        (*training_data)[(*training_data_count)++] = token_array;
    }
}

void print_training_samples(int **training_data, int training_data_count) {
    printf("\n  📋 Training Samples Preview:\n");
    printf("  ┌─────┬──────────────────────────────────────────────────────┐\n");
    int limit = (training_data_count < 5) ? training_data_count : 5;
    
    for(int i = 0; i < limit; i++) {
        printf("  │ %3d │ ", i + 1);
        for(int j = 0; j < 8; j++) {
            printf("%3d ", training_data[i][j]);
        }
        if (limit > 8) printf("...");
        printf(" │\n");
    }
    printf("  └─────┴──────────────────────────────────────────────────────┘\n");
}

void free_training_data(int **training_data, int training_data_count) {
    if (training_data) {
        for (int i = 0; i < training_data_count; i++) {
            if (training_data[i]) free(training_data[i]);
        }
        free(training_data);
    }
}

// ============================================================================
// MAIN FUNCTION WITH REPL LAYOUT
// ============================================================================

int main() {
    print_header("TRANSFORMER-C NEURAL NETWORK");
    print_info("Starting training session at %s", __TIME__);
    
    // ==================== LEVEL 1: DATA LOADING ====================
    print_section("1. DATA LOADING & PREPROCESSING");
    
    // LOAD RAW TEXT DATA
    print_info("Loading text data from 'text_data.txt'...");
    char *raw_text = readFileToString("text_data.txt");
    if (!raw_text) {
        print_error("Failed to load text_data.txt");
        return 1;
    }
    
    print_success("Loaded %zu characters", strlen(raw_text));
    
    // Show preview
    print_info("Text preview (first 100 chars):");
    printf("  ┌──────────────────────────────────────────────────────────────────┐\n");
    printf("  │ ");
    for(int i = 0; i < 100 && raw_text[i] != '\0'; i++) {
        if (raw_text[i] == '\n') printf("\\n");
        else putchar(raw_text[i]);
        if ((i+1) % 50 == 0 && i < 99) printf("\n  │ ");
    }
    printf("\n  └──────────────────────────────────────────────────────────────────┘\n");
    
    // SPLIT TO SENTENCES
    print_info("Splitting text into sentences...");
    char **sentences = SplitSentences(raw_text);
    if (!sentences) {
        print_error("Failed to split sentences");
        free(raw_text);
        return 1;
    }
    
    // Count sentences
    int num_sentences = 0;
    while(sentences[num_sentences] != NULL) num_sentences++;
    print_success("Extracted %d sentences", num_sentences);
    
    // PRINT SENTENCES
    if (num_sentences > 0) {
        print_info("First 3 sentences:");
        int show_count = (num_sentences < 3) ? num_sentences : 3;
        for (int i = 0; i < show_count; i++) {
            printf("  %d. %s\n", i + 1, sentences[i]);
        }
    }
    
    // DATA CLEANING
    print_info("Cleaning text (lowercase, remove punctuation)...");
    for (int i = 0; i < num_sentences; i++) {
        if (sentences[i]) {
            char *cleaned_sentence = Cleaned_Text(sentences[i]);
            if (cleaned_sentence) {
                free(sentences[i]);
                sentences[i] = cleaned_sentence;
            }
        }
    }
    print_success("Text cleaned");
    
    // ==================== LEVEL 2: TOKENIZATION ====================
    print_section("2. TOKENIZATION & VOCABULARY");
    
    print_info("Extracting unique words and creating vocabulary...");
    extractUniqueWords(sentences);
    print_success("Vocabulary created");
    
    // Show sample tokens
    print_info("Sample token mappings:");
    const char* sample_tokens[] = {"martin", "luther", "king", "civil", "rights", NULL};
    for (int i = 0; sample_tokens[i] != NULL; i++) {
        int token_id = getTokenId(sample_tokens[i]);
        if (token_id != -1) {
            double embedding[2];
            getEmbeddingByTokenId(token_id, embedding);
            print_token_mapping(sample_tokens[i], token_id, embedding);
        }
    }
    
    // ==================== LEVEL 3: TRAINING DATA PREPARATION ====================
    print_section("3. TRAINING DATA PREPARATION");
    
    // Prepare training data
    print_info("Preparing training data...");
    int **training_data = NULL;
    int training_data_count = 0;
    
    process_training_data(sentences, num_sentences, &training_data, &training_data_count);
    
    if (training_data_count > 0) {
        print_success("Prepared %d training samples", training_data_count);
        print_training_samples(training_data, training_data_count);
    } else {
        print_error("No training data prepared");
        // Cleanup and exit
        free(raw_text);
        for (int i = 0; i < num_sentences; i++) {
            if (sentences[i]) free(sentences[i]);
        }
        free(sentences);
        return 1;
    }
    
    // ==================== LEVEL 4: MODEL INITIALIZATION ====================
    print_section("4. MODEL INITIALIZATION");
    
    print_info("Creating/checking weight files...");
    create_weight_directories_and_files();
    
    print_info("Loading self-attention matrices...");
    initialize_matrices_from_files();
    print_success("Self-attention matrices loaded");
    
    // Show attention matrices
    print_info("Attention matrices:");
    printf("  K Matrix: [%8.4f, %8.4f]\n", k_matrix[0][0], k_matrix[0][1]);
    printf("           [%8.4f, %8.4f]\n", k_matrix[1][0], k_matrix[1][1]);
    printf("  Q Matrix: [%8.4f, %8.4f]\n", q_matrix[0][0], q_matrix[0][1]);
    printf("           [%8.4f, %8.4f]\n", q_matrix[1][0], q_matrix[1][1]);
    printf("  V Matrix: [%8.4f, %8.4f]\n", v_matrix[0][0], v_matrix[0][1]);
    printf("           [%8.4f, %8.4f]\n", v_matrix[1][0], v_matrix[1][1]);
    
    // Load MLP weights
    print_info("Loading multi-layer perceptron weights...");
    double semi_final_layer_weights[512 * 65] = {0.0};
    double final_layer_weights[65 * 2] = {0.0};
    
    read_weights("Model Trained Weights/Semi_Final Multi-layered perceptron Weights/", 
                 semi_final_layer_weights, 512 * 65);
    read_weights("Model Trained Weights/Final Multi-layered perceptron weights/", 
                 final_layer_weights, 65 * 2);
    print_success("MLP weights loaded");
    
    // ==================== LEVEL 5: TRAINING LOOP ====================
    print_section("5. TRAINING LOOP");
    
    int epochs = 5;  // Reduced for demo
    printf("\n  🎯 Training Configuration:\n");
    printf("    Epochs: %d\n", epochs);
    printf("    Samples: %d\n", training_data_count);
    printf("    Learning Rate: %.4f\n", LEARNING_RATE);
    printf("    Batch Size: 1 (online learning)\n");
    
    printf("\n  ⚙️  Starting training...\n");
    printf("  ┌──────────────────────────────────────────────────────────────────┐\n");
    
    double best_loss = 1e9;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("  │ Epoch %3d/%d ", epoch + 1, epochs);
        fflush(stdout);
        
        double epoch_loss = 0;
        
        // Process each sample
        for (int sample_index = 0; sample_index < training_data_count; sample_index++) {
            // Process each sample
            float sentence[MAX_SENTENCE_LENGTH];
            for (int i = 0; i < MAX_SENTENCE_LENGTH; i++) {
                sentence[i] = training_data[sample_index][i];
            }
            
            // Get target (last non-zero token)
            int y_actual = 0;
            for (int k = MAX_SENTENCE_LENGTH - 1; k >= 0; k--) {
                if (sentence[k] != 0) {
                    y_actual = sentence[k];
                    sentence[k] = 0;  // Remove for prediction
                    break;
                }
            }
            
            // Create embedding matrix
            float embedding_matrix[MAX_SENTENCE_LENGTH][2] = {0};
            for (int i = 0; i < MAX_SENTENCE_LENGTH; i++) {
                if (sentence[i] != 0) {
                    unsigned int token_id = (unsigned int)sentence[i];
                    double curr_vector_embedding[2];
                    getEmbeddingByTokenId(token_id, curr_vector_embedding);
                    embedding_matrix[i][0] = (float)curr_vector_embedding[0];
                    embedding_matrix[i][1] = (float)curr_vector_embedding[1];
                }
            }
            
            // Scale and add positional encoding
            scale_matrix(embedding_matrix);
            Add_Positional_Encoding(embedding_matrix, MAX_SENTENCE_LENGTH);
            
            // Self-attention
            double final_k_matrix[MAX_SENTENCE_LENGTH][MATRIX_SIZE] = {0};
            double final_q_matrix[MAX_SENTENCE_LENGTH][MATRIX_SIZE] = {0};
            double final_v_matrix[MAX_SENTENCE_LENGTH][MATRIX_SIZE] = {0};
            
            for (int i = 0; i < MAX_SENTENCE_LENGTH; i++) {
                for (int j = 0; j < MATRIX_SIZE; j++) {
                    for (int k = 0; k < MATRIX_SIZE; k++) {
                        final_k_matrix[i][j] += embedding_matrix[i][k] * k_matrix[k][j];
                        final_q_matrix[i][j] += embedding_matrix[i][k] * q_matrix[k][j];
                        final_v_matrix[i][j] += embedding_matrix[i][k] * v_matrix[k][j];
                    }
                }
            }
            
            double self_attention_matrix[MAX_SENTENCE_LENGTH][MATRIX_SIZE] = {0};
            compute_self_attention(embedding_matrix, final_k_matrix, final_q_matrix, 
                                 final_v_matrix, MAX_SENTENCE_LENGTH, self_attention_matrix);
            
            double context_matrix[MAX_SENTENCE_LENGTH][2] = {0};
            add_matrices(embedding_matrix, self_attention_matrix, context_matrix, 
                        MAX_SENTENCE_LENGTH, MATRIX_SIZE);
            
            // Forward pass through MLP - FIXED: Correct dimensions
            double semi_final_layer_nodes[65] = {0.0};  // FIXED: 65 nodes, not 512*65
            for(int node = 0; node < 65; node++) {
                double total = 0.0;
                for(int row = 0; row < 512; row++) {
                    int weight_index = (row * 65) + node;
                    double weight = semi_final_layer_weights[weight_index];
                    total += embedding_matrix[row][0] * weight + embedding_matrix[row][1] * weight;
                }
                semi_final_layer_nodes[node] = leaky_relu(total, 0.01);
            }
            
            // Output layer - FIXED: Correct dimensions
            double total_value_node_1 = 0, total_value_node_2 = 0;
            for(int i = 0; i < 65; i++) {  // FIXED: 65 nodes, not 130
                total_value_node_1 += semi_final_layer_nodes[i] * final_layer_weights[i];
                total_value_node_2 += semi_final_layer_nodes[i] * final_layer_weights[i + 65];
            }
            
            double output_embedding[2] = {swish(total_value_node_1), swish(total_value_node_2)};
            
            // Compute loss
            double expected_embedding[2];
            getEmbeddingByTokenId(y_actual, expected_embedding);
            double loss = calculate_mse(output_embedding, expected_embedding, 2);
            epoch_loss += loss;
            
            // Backpropagation - FIXED: Pass correct parameters
            double learning_rate = (double)LEARNING_RATE;
            
            // FIXED: Pass semi_final_layer_NODES (activations), not weights!
            update_weights_last_layer(loss, learning_rate, final_layer_weights, 
                                    semi_final_layer_nodes, 130, 65, 100);  // 130 weights, 65 nodes
            
            update_semi_final_layer_weights(loss, learning_rate, semi_final_layer_weights, 
                                          33280, 100);  // 512*65 = 33280 weights
            
            // This function is in transformer_block.c
            update_attention_matrices(loss, learning_rate);
        }
        
        epoch_loss /= training_data_count;
        
        if (epoch_loss < best_loss) {
            best_loss = epoch_loss;
            printf("   Loss: %.6f (✨ BEST)", epoch_loss);
        } else {
            printf("   Loss: %.6f", epoch_loss);
        }
        
        if (epoch == epochs - 1) {
            printf(" │\n");
        } else {
            printf(" │\n  │");
        }
    }
    
    printf("  └──────────────────────────────────────────────────────────────────┘\n");
    print_success("Training complete! Best loss: %.6f", best_loss);
    
    // ==================== SAVE TRAINED MODEL ====================
    print_section("MODEL SAVING");
    
    // Save the trained model
    save_model("trained_model_v1", 
               k_matrix, q_matrix, v_matrix,
               semi_final_layer_weights, final_layer_weights,
               512 * 65, 65 * 2,
               best_loss, epochs,
               NULL);  // Vocabulary path (you can add this later)

    // ==================== INTERACTIVE REPL ====================
    print_section("INTERACTIVE REPL");
    printf("\n  🎮 Starting interactive REPL...\n");
    printf("  Type 'help' for available commands\n\n");
    
    // Start interactive session
    start_interactive_repl();
    
    // ==================== CLEANUP ====================
    print_section("CLEANUP");
    
    print_info("Freeing memory...");
    free_training_data(training_data, training_data_count);
    free(raw_text);
    
    for (int i = 0; i < num_sentences; i++) {
        if (sentences[i]) free(sentences[i]);
    }
    free(sentences);
    
    print_success("Memory freed");
    print_header("TRANSFORMER-C TRAINING COMPLETE");
    
    return 0;
}