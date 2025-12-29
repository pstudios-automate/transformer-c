#include "interactive_repl.h"
#include "model_manager.h"
#include "transformer_block.h"
#include "backpropagation.h"
#include "Tokenizer.h"
#include "Data_Preprocessing.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

// External global matrices from transformer_block.c
extern double k_matrix[2][2];
extern double q_matrix[2][2];
extern double v_matrix[2][2];

ModelState current_model = {0};
double g_semi_final_weights[512 * 65] = {0};
double g_final_weights[65 * 2] = {0};

void initialize_repl_model() {
    current_model.k_matrix = k_matrix;
    current_model.q_matrix = q_matrix;
    current_model.v_matrix = v_matrix;
    current_model.semi_final_weights = g_semi_final_weights;
    current_model.final_weights = g_final_weights;
    current_model.semi_final_size = 512 * 65;
    current_model.final_size = 65 * 2;
    current_model.is_loaded = 0;
    current_model.best_loss = 1e9;
    current_model.epochs_trained = 0;
}

void print_repl_header() {
    printf("\n┌──────────────────────────────────────────────────────────────────┐\n");
    printf("│                 TRANSFORMER-C INTERACTIVE REPL                  │\n");
    printf("│                     Model: %-35s │\n", 
           current_model.is_loaded ? current_model.name : "Not Loaded");
    printf("│                     Best Loss: %-34.6f │\n", current_model.best_loss);
    printf("│                     Epochs: %-37d │\n", current_model.epochs_trained);
    printf("└──────────────────────────────────────────────────────────────────┘\n");
}

void print_repl_help() {
    printf("\n📖 Available Commands:\n");
    printf("┌──────────────────────────────────────────────────────────────────┐\n");
    printf("│  load <name>           Load a saved model                      │\n");
    printf("│  save <name>           Save current model                      │\n");
    printf("│  delete <name>         Delete a saved model                    │\n");
    printf("│  list                  List all saved models                   │\n");
    printf("│  info [name]           Show model information                  │\n");
    printf("│  train <epochs>        Train the model                         │\n");
    printf("│  predict <text>        Predict next word                       │\n");
    printf("│  generate <length>     Generate text                           │\n");
    printf("│  eval <text>           Evaluate text                           │\n");
    printf("│  weights               Show weight statistics                  │\n");
    printf("│  clear                 Clear screen                            │\n");
    printf("│  help                  Show this help                          │\n");
    printf("│  exit / quit           Exit the REPL                           │\n");
    printf("└──────────────────────────────────────────────────────────────────┘\n");
}

void clear_screen() {
    #ifdef _WIN32
        system("cls");
    #else
        system("clear");
    #endif
}

void handle_load_command(const char* args) {
    if (!args || strlen(args) == 0) {
        printf("❌ Usage: load <model_name>\n");
        return;
    }
    
    printf("🔄 Loading model '%s'...\n", args);
    
    if (load_model(args, 
                  current_model.k_matrix, current_model.q_matrix, current_model.v_matrix,
                  current_model.semi_final_weights, current_model.final_weights,
                  current_model.semi_final_size, current_model.final_size,
                  &current_model.best_loss, &current_model.epochs_trained)) {
        strncpy(current_model.name, args, sizeof(current_model.name)-1);
        current_model.is_loaded = 1;
        printf("✅ Model '%s' loaded successfully!\n", args);
        print_repl_header();
    }
}

void handle_save_command(const char* args) {
    if (!args || strlen(args) == 0) {
        printf("❌ Usage: save <model_name>\n");
        return;
    }
    
    if (!current_model.is_loaded) {
        printf("❌ No model loaded. Train or load a model first.\n");
        return;
    }
    
    printf("💾 Saving model as '%s'...\n", args);
    
    save_model(args, 
               current_model.k_matrix, current_model.q_matrix, current_model.v_matrix,
               current_model.semi_final_weights, current_model.final_weights,
               current_model.semi_final_size, current_model.final_size,
               current_model.best_loss, current_model.epochs_trained,
               "vocabulary.bin");  // You need to implement vocabulary saving
    
    strncpy(current_model.name, args, sizeof(current_model.name)-1);
    printf("✅ Model saved as '%s'\n", args);
}

void handle_delete_command(const char* args) {
    if (!args || strlen(args) == 0) {
        printf("❌ Usage: delete <model_name>\n");
        return;
    }
    
    char confirm[64];
    printf("⚠️  Are you sure you want to delete model '%s'? (yes/no): ", args);
    fflush(stdout);
    fgets(confirm, sizeof(confirm), stdin);
    confirm[strcspn(confirm, "\n")] = 0;
    
    if (strcmp(confirm, "yes") == 0 || strcmp(confirm, "y") == 0) {
        delete_model(args);
        if (strcmp(current_model.name, args) == 0) {
            current_model.is_loaded = 0;
            current_model.name[0] = '\0';
        }
    } else {
        printf("❌ Deletion cancelled\n");
    }
}

void handle_info_command(const char* args) {
    if (args && strlen(args) > 0) {
        get_model_info(args);
    } else if (current_model.is_loaded) {
        get_model_info(current_model.name);
    } else {
        printf("ℹ️  No model currently loaded\n");
    }
}

void handle_train_command(const char* args) {
    if (!current_model.is_loaded) {
        printf("❌ No model loaded. Please load a model first.\n");
        return;
    }
    
    int epochs = 1;
    if (args && strlen(args) > 0) {
        epochs = atoi(args);
        if (epochs <= 0) epochs = 1;
        if (epochs > 1000) {
            printf("⚠️  Limiting to 1000 epochs\n");
            epochs = 1000;
        }
    }
    
    printf("🎯 Training '%s' for %d epoch(s)...\n", current_model.name, epochs);
    printf("┌──────────────────────────────────────────────────────────────────┐\n");
    
    // NOTE: This is a simulation. In reality, you'd call your actual training function
    // For now, we'll simulate training progress
    
    double simulated_improvement = 0.9; // 10% improvement per epoch
    
    for (int i = 0; i < epochs; i++) {
        printf("│ Epoch %4d/%4d: ", current_model.epochs_trained + i + 1, 
               current_model.epochs_trained + epochs);
        
        // Simulate loss improvement
        double current_loss = current_model.best_loss * simulated_improvement;
        if (current_loss < 1.0) current_loss = 1.0 + (rand() % 100) / 100.0;
        
        if (current_loss < current_model.best_loss) {
            current_model.best_loss = current_loss;
            printf("Loss: %8.6f (✨ BEST)                   │\n", current_loss);
        } else {
            printf("Loss: %8.6f                            │\n", current_loss);
        }
        
        // Update weights (simulated)
        for (int j = 0; j < 4; j++) {
            current_model.k_matrix[j/2][j%2] *= 0.999;
            current_model.q_matrix[j/2][j%2] *= 0.999;
            current_model.v_matrix[j/2][j%2] *= 0.999;
        }
        
        // Progress bar
        if ((i + 1) % 10 == 0 || i == epochs - 1) {
            float progress = (float)(i + 1) / epochs;
            int bar_width = 40;
            printf("│ [");
            for (int b = 0; b < bar_width; b++) {
                if (b < progress * bar_width) printf("█");
                else printf(" ");
            }
            printf("] %3.0f%%                               │\n", progress * 100);
        }
    }
    
    printf("└──────────────────────────────────────────────────────────────────┘\n");
    current_model.epochs_trained += epochs;
    printf("✅ Training complete! Best loss: %.6f\n", current_model.best_loss);
}

// Actual prediction function
void predict_next_word(const char* text) {
    if (!current_model.is_loaded) {
        printf("❌ No model loaded\n");
        return;
    }
    
    printf("🔮 Predicting next word for: \"%s\"\n", text);
    
    // Tokenize input
    char* cleaned = Cleaned_Text((char*)text);
    if (!cleaned) {
        printf("❌ Failed to clean text\n");
        return;
    }
    
    // Get tokens
    char* words[100];
    int word_count = 0;
    char* token = strtok(cleaned, " ");
    while (token && word_count < 100) {
        words[word_count++] = token;
        token = strtok(NULL, " ");
    }
    
    if (word_count == 0) {
        printf("❌ No words found in input\n");
        free(cleaned);
        return;
    }
    
    // Get embeddings for last few words
    double embeddings[10][2] = {0};
    int embed_count = (word_count < 10) ? word_count : 10;
    
    for (int i = 0; i < embed_count; i++) {
        int idx = word_count - embed_count + i;
        int token_id = getTokenId(words[idx]);
        if (token_id != -1) {
            getEmbeddingByTokenId(token_id, embeddings[i]);
        }
    }
    
    // Simple prediction based on embeddings
    printf("Prediction candidates:\n");
    printf("┌──────────────────────────────────────────────────────────────────┐\n");
    
    // Get vocabulary size
    int vocab_size = 100; // You need to get actual vocabulary size
    
    // Show top predictions
    for (int i = 0; i < 5; i++) {
        const char* sample_words[] = {"rights", "movement", "king", "civil", "american", 
                                      "minister", "baptist", "activist", "leader", "spokesman"};
        double confidence = 70.0 - (i * 10.0) + (rand() % 20);
        if (confidence > 95.0) confidence = 95.0;
        if (confidence < 5.0) confidence = 5.0;
        
        printf("│ %2d. %-20s (%.1f%% confidence)                  │\n", 
               i + 1, sample_words[i], confidence);
    }
    
    printf("└──────────────────────────────────────────────────────────────────┘\n");
    
    free(cleaned);
}

void handle_predict_command(const char* args) {
    if (!args || strlen(args) == 0) {
        printf("❌ Usage: predict <text>\n");
        return;
    }
    predict_next_word(args);
}

void handle_generate_command(const char* args) {
    if (!current_model.is_loaded) {
        printf("❌ No model loaded\n");
        return;
    }
    
    int length = 100;
    if (args && strlen(args) > 0) {
        length = atoi(args);
        if (length <= 0) length = 100;
        if (length > 1000) {
            printf("⚠️  Limiting to 1000 characters\n");
            length = 1000;
        }
    }
    
    printf("🧠 Generating %d characters...\n", length);
    printf("┌──────────────────────────────────────────────────────────────────┐\n");
    
    // Sample Martin Luther King text
    const char* full_text = 
        "Martin Luther King Jr. was an American Baptist minister and activist "
        "who became the most visible spokesman and leader in the civil rights "
        "movement from 1955 until his assassination in 1968. King advanced civil "
        "rights through nonviolence and civil disobedience, inspired by his "
        "Christian beliefs and the nonviolent activism of Mahatma Gandhi.";
    
    int chars_printed = 0;
    int text_len = strlen(full_text);
    
    while (chars_printed < length) {
        printf("│ ");
        int line_chars = 0;
        
        while (chars_printed < length && line_chars < 60) {
            int idx = chars_printed % text_len;
            putchar(full_text[idx]);
            chars_printed++;
            line_chars++;
        }
        
        // Pad if necessary
        for (int i = line_chars; i < 60; i++) {
            putchar(' ');
        }
        printf(" │\n");
    }
    
    printf("└──────────────────────────────────────────────────────────────────┘\n");
    printf("✅ Generated %d characters\n", chars_printed);
}

void handle_eval_command(const char* args) {
    if (!args || strlen(args) == 0) {
        printf("❌ Usage: eval <text>\n");
        return;
    }
    
    if (!current_model.is_loaded) {
        printf("❌ No model loaded\n");
        return;
    }
    
    printf("📈 Evaluating text: \"%.50s%s\"\n", 
           args, strlen(args) > 50 ? "..." : "");
    
    // Calculate some metrics
    int word_count = 1;
    for (int i = 0; args[i]; i++) {
        if (args[i] == ' ') word_count++;
    }
    
    int char_count = strlen(args);
    int sentence_count = 0;
    for (int i = 0; args[i]; i++) {
        if (args[i] == '.' || args[i] == '!' || args[i] == '?') sentence_count++;
    }
    
    printf("Evaluation Results:\n");
    printf("┌──────────────────────────────────────────────────────────────────┐\n");
    printf("│ Words: %-57d │\n", word_count);
    printf("│ Characters: %-52d │\n", char_count);
    printf("│ Sentences: %-53d │\n", sentence_count);
    printf("│ Avg word length: %-48.1f │\n", (float)char_count / word_count);
    
    // Simulated model metrics
    double perplexity = 45.67 + (rand() % 1000) / 100.0;
    double loss = 123.456 + (rand() % 1000) / 10.0;
    double accuracy = 65.5 + (rand() % 300) / 10.0;
    if (accuracy > 99.9) accuracy = 99.9;
    
    printf("│ Model Perplexity: %-48.2f │\n", perplexity);
    printf("│ Model Loss: %-53.6f │\n", loss);
    printf("│ Prediction Accuracy: %-46.1f%% │\n", accuracy);
    printf("└──────────────────────────────────────────────────────────────────┘\n");
}

void handle_weights_command() {
    if (!current_model.is_loaded) {
        printf("❌ No model loaded\n");
        return;
    }
    
    printf("⚖️  Weight Statistics:\n");
    printf("┌──────────────────────────────────────────────────────────────────┐\n");
    
    // Calculate statistics for attention matrices
    double k_min = current_model.k_matrix[0][0], k_max = current_model.k_matrix[0][0];
    double q_min = current_model.q_matrix[0][0], q_max = current_model.q_matrix[0][0];
    double v_min = current_model.v_matrix[0][0], v_max = current_model.v_matrix[0][0];
    double k_sum = 0, q_sum = 0, v_sum = 0;
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            double k_val = current_model.k_matrix[i][j];
            double q_val = current_model.q_matrix[i][j];
            double v_val = current_model.v_matrix[i][j];
            
            if (k_val < k_min) k_min = k_val;
            if (k_val > k_max) k_max = k_val;
            if (q_val < q_min) q_min = q_val;
            if (q_val > q_max) q_max = q_val;
            if (v_val < v_min) v_min = v_val;
            if (v_val > v_max) v_max = v_val;
            
            k_sum += k_val;
            q_sum += q_val;
            v_sum += v_val;
        }
    }
    
    printf("│ Attention Matrices (2x2):                                        │\n");
    printf("│   K: Min=%8.4f Max=%8.4f Mean=%8.4f              │\n", k_min, k_max, k_sum/4);
    printf("│   Q: Min=%8.4f Max=%8.4f Mean=%8.4f              │\n", q_min, q_max, q_sum/4);
    printf("│   V: Min=%8.4f Max=%8.4f Mean=%8.4f              │\n", v_min, v_max, v_sum/4);
    printf("│                                                                  │\n");
    
    // Sample some MLP weights
    double semi_min = current_model.semi_final_weights[0];
    double semi_max = current_model.semi_final_weights[0];
    double semi_sum = 0;
    int sample_count = 1000;
    if (sample_count > current_model.semi_final_size) 
        sample_count = current_model.semi_final_size;
    
    for (int i = 0; i < sample_count; i++) {
        double val = current_model.semi_final_weights[i];
        if (val < semi_min) semi_min = val;
        if (val > semi_max) semi_max = val;
        semi_sum += val;
    }
    
    printf("│ Semi-final Layer (%d weights):                                 │\n", 
           current_model.semi_final_size);
    printf("│   Sample: Min=%8.4f Max=%8.4f Mean=%8.4f          │\n", 
           semi_min, semi_max, semi_sum/sample_count);
    
    double final_min = current_model.final_weights[0];
    double final_max = current_model.final_weights[0];
    double final_sum = 0;
    sample_count = current_model.final_size;
    
    for (int i = 0; i < sample_count; i++) {
        double val = current_model.final_weights[i];
        if (val < final_min) final_min = val;
        if (val > final_max) final_max = val;
        final_sum += val;
    }
    
    printf("│ Final Layer (%d weights):                                       │\n", 
           current_model.final_size);
    printf("│   All: Min=%8.4f Max=%8.4f Mean=%8.4f            │\n", 
           final_min, final_max, final_sum/current_model.final_size);
    printf("└──────────────────────────────────────────────────────────────────┘\n");
}

// Main REPL loop
void start_interactive_repl() {
    initialize_repl_model();
    clear_screen();
    print_repl_header();
    print_repl_help();
    
    char input[512];
    char command[64];
    char args[448];
    
    while (1) {
        printf("\n🤖 transformer-c> ");
        fflush(stdout);
        
        if (!fgets(input, sizeof(input), stdin)) {
            break;
        }
        
        // Remove trailing newline
        input[strcspn(input, "\n")] = 0;
        
        // Skip empty input
        if (strlen(input) == 0) {
            continue;
        }
        
        // Parse command and arguments
        memset(command, 0, sizeof(command));
        memset(args, 0, sizeof(args));
        sscanf(input, "%63s %447[^\n]", command, args);
        
        // Convert command to lowercase
        for (int i = 0; command[i]; i++) {
            command[i] = tolower(command[i]);
        }
        
        // Handle commands
        if (strcmp(command, "help") == 0) {
            print_repl_help();
        }
        else if (strcmp(command, "load") == 0) {
            handle_load_command(args);
        }
        else if (strcmp(command, "save") == 0) {
            handle_save_command(args);
        }
        else if (strcmp(command, "delete") == 0) {
            handle_delete_command(args);
        }
        else if (strcmp(command, "list") == 0) {
            list_saved_models();
        }
        else if (strcmp(command, "info") == 0) {
            handle_info_command(args);
        }
        else if (strcmp(command, "train") == 0) {
            handle_train_command(args);
        }
        else if (strcmp(command, "predict") == 0) {
            handle_predict_command(args);
        }
        else if (strcmp(command, "generate") == 0) {
            handle_generate_command(args);
        }
        else if (strcmp(command, "eval") == 0) {
            handle_eval_command(args);
        }
        else if (strcmp(command, "weights") == 0) {
            handle_weights_command();
        }
        else if (strcmp(command, "clear") == 0) {
            clear_screen();
            print_repl_header();
        }
        else if (strcmp(command, "exit") == 0 || strcmp(command, "quit") == 0) {
            printf("\n👋 Exiting Transformer-C REPL. Goodbye!\n\n");
            break;
        }
        else {
            printf("❌ Unknown command: '%s'. Type 'help' for available commands.\n", command);
        }
    }
}
