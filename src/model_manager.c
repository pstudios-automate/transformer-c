#include "model_manager.h"
#include "transformer_block.h"
#include "backpropagation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#define mkdir(path, mode) _mkdir(path)
#define access _access
#else
#include <unistd.h>
#endif

#define MODEL_DIR "saved_models"
#define MAX_MODELS 100

// Create directory if it doesn't exist
static int create_directory(const char* path) {
    #ifdef _WIN32
        return _mkdir(path);
    #else
        return mkdir(path, 0755);
    #endif
}

// Check if directory exists
static int directory_exists(const char* path) {
    #ifdef _WIN32
        struct _stat info;
        if (_stat(path, &info) != 0) return 0;
        return (info.st_mode & _S_IFDIR);
    #else
        struct stat info;
        if (stat(path, &info) != 0) return 0;
        return S_ISDIR(info.st_mode);
    #endif
}

// Save complete model state
void save_model(const char* model_name, 
                double k_matrix[][2], double q_matrix[][2], double v_matrix[][2],
                double* semi_final_weights, double* final_weights,
                int semi_final_size, int final_size,
                double best_loss, int epochs_trained, const char* vocabulary_path) {
    
    // Ensure base directory exists
    if (!directory_exists(MODEL_DIR)) {
        if (create_directory(MODEL_DIR) != 0) {
            printf("❌ Failed to create model directory\n");
            return;
        }
    }
    
    // Create model directory
    char model_path[512];
    snprintf(model_path, sizeof(model_path), "%s/%s", MODEL_DIR, model_name);
    
    if (!directory_exists(model_path)) {
        if (create_directory(model_path) != 0) {
            printf("❌ Failed to create model directory: %s\n", model_path);
            return;
        }
    }
    
    printf("\n💾 Saving model '%s' to %s/\n", model_name, model_path);
    
    // Save attention matrices
    char attn_path[512];
    snprintf(attn_path, sizeof(attn_path), "%s/attention.bin", model_path);
    FILE* attn_file = fopen(attn_path, "wb");
    if (attn_file) {
        // Save all 4 elements of each 2x2 matrix
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                fwrite(&k_matrix[i][j], sizeof(double), 1, attn_file);
                fwrite(&q_matrix[i][j], sizeof(double), 1, attn_file);
                fwrite(&v_matrix[i][j], sizeof(double), 1, attn_file);
            }
        }
        fclose(attn_file);
        printf("✅ Attention matrices saved to %s\n", attn_path);
    } else {
        printf("❌ Failed to save attention matrices\n");
    }
    
    // Save semi-final weights
    char semi_path[512];
    snprintf(semi_path, sizeof(semi_path), "%s/semi_final.bin", model_path);
    FILE* semi_file = fopen(semi_path, "wb");
    if (semi_file) {
        size_t written = fwrite(semi_final_weights, sizeof(double), semi_final_size, semi_file);
        fclose(semi_file);
        if (written == semi_final_size) {
            printf("✅ Semi-final weights saved (%zu weights)\n", written);
        } else {
            printf("⚠️  Partial save of semi-final weights (%zu of %d)\n", written, semi_final_size);
        }
    } else {
        printf("❌ Failed to save semi-final weights\n");
    }
    
    // Save final weights
    char final_path[512];
    snprintf(final_path, sizeof(final_path), "%s/final.bin", model_path);
    FILE* final_file = fopen(final_path, "wb");
    if (final_file) {
        size_t written = fwrite(final_weights, sizeof(double), final_size, final_file);
        fclose(final_file);
        if (written == final_size) {
            printf("✅ Final weights saved (%zu weights)\n", written);
        } else {
            printf("⚠️  Partial save of final weights (%zu of %d)\n", written, final_size);
        }
    } else {
        printf("❌ Failed to save final weights\n");
    }
    
    // Save metadata
    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s/metadata.txt", model_path);
    FILE* meta_file = fopen(meta_path, "w");
    if (meta_file) {
        time_t now = time(NULL);
        fprintf(meta_file, "MODEL METADATA\n");
        fprintf(meta_file, "===============\n");
        fprintf(meta_file, "Name: %s\n", model_name);
        fprintf(meta_file, "Saved: %s", ctime(&now));
        fprintf(meta_file, "Best Loss: %.6f\n", best_loss);
        fprintf(meta_file, "Epochs Trained: %d\n", epochs_trained);
        fprintf(meta_file, "Semi-final layer size: %d\n", semi_final_size);
        fprintf(meta_file, "Final layer size: %d\n", final_size);
        fprintf(meta_file, "Matrix size: 2x2\n");
        fprintf(meta_file, "Learning Rate: 0.01\n");
        fprintf(meta_file, "Embedding Dimension: 2\n");
        fprintf(meta_file, "Max Sentence Length: 512\n");
        if (vocabulary_path) {
            fprintf(meta_file, "Vocabulary: %s\n", vocabulary_path);
        }
        fclose(meta_file);
        printf("✅ Metadata saved to %s\n", meta_path);
    }
    
    // Save vocabulary state if path provided
    if (vocabulary_path && access(vocabulary_path, 0) == 0) {
        char vocab_dest[512];
        snprintf(vocab_dest, sizeof(vocab_dest), "%s/vocabulary.bin", model_path);
        
        #ifdef _WIN32
            char cmd[1024];
            snprintf(cmd, sizeof(cmd), "copy \"%s\" \"%s\" >nul", vocabulary_path, vocab_dest);
            system(cmd);
        #else
            char cmd[1024];
            snprintf(cmd, sizeof(cmd), "cp \"%s\" \"%s\" 2>/dev/null", vocabulary_path, vocab_dest);
            system(cmd);
        #endif
        printf("✅ Vocabulary state saved\n");
    }
    
    printf("✨ Model '%s' saved successfully!\n", model_name);
}

// Load complete model state
int load_model(const char* model_name,
               double k_matrix[][2], double q_matrix[][2], double v_matrix[][2],
               double* semi_final_weights, double* final_weights,
               int semi_final_size, int final_size,
               double* best_loss, int* epochs_trained) {
    
    char model_path[512];
    snprintf(model_path, sizeof(model_path), "%s/%s", MODEL_DIR, model_name);
    
    if (!directory_exists(model_path)) {
        printf("❌ Model '%s' not found at %s\n", model_name, model_path);
        return 0;
    }
    
    printf("\n📂 Loading model '%s' from %s/\n", model_name, model_path);
    
    // Load attention matrices
    char attn_path[512];
    snprintf(attn_path, sizeof(attn_path), "%s/attention.bin", model_path);
    FILE* attn_file = fopen(attn_path, "rb");
    if (attn_file) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                if (fread(&k_matrix[i][j], sizeof(double), 1, attn_file) != 1) goto attn_error;
                if (fread(&q_matrix[i][j], sizeof(double), 1, attn_file) != 1) goto attn_error;
                if (fread(&v_matrix[i][j], sizeof(double), 1, attn_file) != 1) goto attn_error;
            }
        }
        fclose(attn_file);
        printf("✅ Attention matrices loaded\n");
        printf("   K: [%8.4f, %8.4f] [%8.4f, %8.4f]\n", 
               k_matrix[0][0], k_matrix[0][1], k_matrix[1][0], k_matrix[1][1]);
        printf("   Q: [%8.4f, %8.4f] [%8.4f, %8.4f]\n", 
               q_matrix[0][0], q_matrix[0][1], q_matrix[1][0], q_matrix[1][1]);
        printf("   V: [%8.4f, %8.4f] [%8.4f, %8.4f]\n", 
               v_matrix[0][0], v_matrix[0][1], v_matrix[1][0], v_matrix[1][1]);
    } else {
        printf("❌ Failed to load attention matrices\n");
        return 0;
    }
    
    // Load semi-final weights
    char semi_path[512];
    snprintf(semi_path, sizeof(semi_path), "%s/semi_final.bin", model_path);
    FILE* semi_file = fopen(semi_path, "rb");
    if (semi_file) {
        size_t read = fread(semi_final_weights, sizeof(double), semi_final_size, semi_file);
        fclose(semi_file);
        if (read == semi_final_size) {
            printf("✅ Semi-final weights loaded (%zu weights)\n", read);
            // Show sample weights
            printf("   Sample: [%8.4f, %8.4f, %8.4f, %8.4f, %8.4f]\n",
                   semi_final_weights[0], semi_final_weights[1], semi_final_weights[2],
                   semi_final_weights[3], semi_final_weights[4]);
        } else {
            printf("❌ Failed to load semi-final weights (expected %d, got %zu)\n", 
                   semi_final_size, read);
            return 0;
        }
    } else {
        printf("❌ Failed to load semi-final weights\n");
        return 0;
    }
    
    // Load final weights
    char final_path[512];
    snprintf(final_path, sizeof(final_path), "%s/final.bin", model_path);
    FILE* final_file = fopen(final_path, "rb");
    if (final_file) {
        size_t read = fread(final_weights, sizeof(double), final_size, final_file);
        fclose(final_file);
        if (read == final_size) {
            printf("✅ Final weights loaded (%zu weights)\n", read);
            // Show sample weights
            printf("   Sample: [%8.4f, %8.4f, %8.4f, %8.4f, %8.4f]\n",
                   final_weights[0], final_weights[1], final_weights[2],
                   final_weights[3], final_weights[4]);
        } else {
            printf("❌ Failed to load final weights (expected %d, got %zu)\n", 
                   final_size, read);
            return 0;
        }
    } else {
        printf("❌ Failed to load final weights\n");
        return 0;
    }
    
    // Load metadata
    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s/metadata.txt", model_path);
    FILE* meta_file = fopen(meta_path, "r");
    if (meta_file) {
        char line[256];
        while (fgets(line, sizeof(line), meta_file)) {
            if (strstr(line, "Best Loss:")) {
                sscanf(line, "Best Loss: %lf", best_loss);
            } else if (strstr(line, "Epochs Trained:")) {
                sscanf(line, "Epochs Trained: %d", epochs_trained);
            }
        }
        fclose(meta_file);
        printf("✅ Metadata loaded (Loss: %.6f, Epochs: %d)\n", 
               *best_loss, *epochs_trained);
    } else {
        printf("⚠️  Metadata not found, using defaults\n");
        if (best_loss) *best_loss = 0.0;
        if (epochs_trained) *epochs_trained = 0;
    }
    
    printf("✨ Model '%s' loaded successfully!\n", model_name);
    return 1;

attn_error:
    fclose(attn_file);
    printf("❌ Error reading attention matrices\n");
    return 0;
}

// List saved models with details
void list_saved_models() {
    printf("\n📁 Saved Models in '%s/':\n", MODEL_DIR);
    printf("┌──────────────────────────────────────────────────────────────────┐\n");
    
    if (!directory_exists(MODEL_DIR)) {
        printf("│ No saved models directory found                                │\n");
        printf("└──────────────────────────────────────────────────────────────────┘\n");
        return;
    }
    
    #ifdef _WIN32
        // Windows implementation
        struct _finddata_t fileinfo;
        intptr_t handle;
        char search_path[512];
        snprintf(search_path, sizeof(search_path), "%s\\*", MODEL_DIR);
        
        handle = _findfirst(search_path, &fileinfo);
        if (handle == -1) {
            printf("│ No models found                                              │\n");
        } else {
            int count = 0;
            do {
                if (fileinfo.attrib & _A_SUBDIR) {
                    if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0) {
                        char model_path[512];
                        snprintf(model_path, sizeof(model_path), "%s/%s/metadata.txt", MODEL_DIR, fileinfo.name);
                        FILE* meta = fopen(model_path, "r");
                        if (meta) {
                            char line[256];
                            double loss = 0.0;
                            int epochs = 0;
                            while (fgets(line, sizeof(line), meta)) {
                                if (strstr(line, "Best Loss:")) sscanf(line, "Best Loss: %lf", &loss);
                                if (strstr(line, "Epochs Trained:")) sscanf(line, "Epochs Trained: %d", &epochs);
                            }
                            fclose(meta);
                            printf("│ %-30s │ Loss: %8.4f │ Epochs: %4d │\n", 
                                   fileinfo.name, loss, epochs);
                        } else {
                            printf("│ %-30s │ (no metadata)                     │\n", fileinfo.name);
                        }
                        count++;
                    }
                }
            } while (_findnext(handle, &fileinfo) == 0);
            _findclose(handle);
            
            if (count == 0) {
                printf("│ No valid models found                                      │\n");
            }
        }
    #else
        // Linux/macOS implementation
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "find %s -maxdepth 1 -type d 2>/dev/null | grep -v '^%s$' | sed 's|%s/||' | while read model; do "
                  "if [ -f \"%s/$model/metadata.txt\" ]; then "
                  "loss=$(grep 'Best Loss:' \"%s/$model/metadata.txt\" | cut -d' ' -f3); "
                  "epochs=$(grep 'Epochs Trained:' \"%s/$model/metadata.txt\" | cut -d' ' -f3); "
                  "printf \"│ %%-30s │ Loss: %%8.4f │ Epochs: %%4d │\\\\n\" \"$model\" \"$loss\" \"$epochs\"; "
                  "else printf \"│ %%-30s │ (no metadata)                     │\\\\n\" \"$model\"; "
                  "fi; done 2>/dev/null || echo '│ No models found                                      │'", 
                  MODEL_DIR, MODEL_DIR, MODEL_DIR, MODEL_DIR, MODEL_DIR, MODEL_DIR);
        
        FILE* pipe = popen(cmd, "r");
        if (pipe) {
            char buffer[256];
            int count = 0;
            while (fgets(buffer, sizeof(buffer), pipe)) {
                printf("%s", buffer);
                count++;
            }
            pclose(pipe);
            if (count == 0) {
                printf("│ No models found                                              │\n");
            }
        } else {
            printf("│ Error listing models                                         │\n");
        }
    #endif
    
    printf("└──────────────────────────────────────────────────────────────────┘\n");
}

// Delete a saved model
int delete_model(const char* model_name) {
    char model_path[512];
    snprintf(model_path, sizeof(model_path), "%s/%s", MODEL_DIR, model_name);
    
    if (!directory_exists(model_path)) {
        printf("❌ Model '%s' not found\n", model_name);
        return 0;
    }
    
    printf("🗑️  Deleting model '%s'...\n", model_name);
    
    #ifdef _WIN32
        char cmd[1024];
        snprintf(cmd, sizeof(cmd), "rmdir /s /q \"%s\" 2>nul", model_path);
        int result = system(cmd);
    #else
        char cmd[1024];
        snprintf(cmd, sizeof(cmd), "rm -rf \"%s\" 2>/dev/null", model_path);
        int result = system(cmd);
    #endif
    
    if (result == 0) {
        printf("✅ Model '%s' deleted successfully\n", model_name);
        return 1;
    } else {
        printf("❌ Failed to delete model '%s'\n", model_name);
        return 0;
    }
}

// Get model info
void get_model_info(const char* model_name) {
    char model_path[512];
    snprintf(model_path, sizeof(model_path), "%s/%s", MODEL_DIR, model_name);
    
    if (!directory_exists(model_path)) {
        printf("❌ Model '%s' not found\n", model_name);
        return;
    }
    
    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s/metadata.txt", model_path);
    FILE* meta_file = fopen(meta_path, "r");
    if (!meta_file) {
        printf("❌ No metadata found for model '%s'\n", model_name);
        return;
    }
    
    printf("\n📊 Model Information: %s\n", model_name);
    printf("┌──────────────────────────────────────────────────────────────────┐\n");
    
    char line[256];
    while (fgets(line, sizeof(line), meta_file)) {
        line[strcspn(line, "\n")] = 0; // Remove newline
        printf("│ %-64s │\n", line);
    }
    fclose(meta_file);
    
    // Check file sizes
    struct stat st;
    char semi_path[512], final_path[512], attn_path[512];
    snprintf(semi_path, sizeof(semi_path), "%s/semi_final.bin", model_path);
    snprintf(final_path, sizeof(final_path), "%s/final.bin", model_path);
    snprintf(attn_path, sizeof(attn_path), "%s/attention.bin", model_path);
    
    printf("│                                                                  │\n");
    
    if (stat(semi_path, &st) == 0) {
        printf("│ Semi-final weights: %-42zu bytes │\n", st.st_size);
    }
    if (stat(final_path, &st) == 0) {
        printf("│ Final weights: %-48zu bytes │\n", st.st_size);
    }
    if (stat(attn_path, &st) == 0) {
        printf("│ Attention weights: %-45zu bytes │\n", st.st_size);
    }
    
    printf("└──────────────────────────────────────────────────────────────────┘\n");
}