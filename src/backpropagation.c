#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// FUNCTION TO CALCULATE MEAN SQUARED ERROR (MSE)
double calculate_mse(double output_array[], double expected_output_array[], int size) {
    double mse = 0.0;
    for (int i = 0; i < size; i++) {
        double difference = output_array[i] - expected_output_array[i];
        mse += difference * difference;
    }
    return mse / size;
}

double clip_gradient_backpropagation(double gradient, double clip_threshold) {
    if (gradient > clip_threshold) {
        return clip_threshold;
    } else if (gradient < -clip_threshold) {
        return -clip_threshold;
    }
    return gradient;
}

// FIXED: Use semi_final_layer_nodes (activations), not weights
void update_weights_last_layer(double loss, double learning_rate, 
                               double final_layer_weights[], 
                               double semi_final_layer_nodes[],  // CHANGED: nodes, not weights
                               int final_layer_size, 
                               int semi_final_layer_size, 
                               double clip_threshold) {

    // ITERATE THROUGH EACH WEIGHT FOR NODE 1
    for (int i = 0; i < semi_final_layer_size; i++) {
        // FIXED: Small fixed gradient for stability
        double gradient_node_1 = 0.001 * loss * semi_final_layer_nodes[i];

        // CLIP THE GRADIENT
        gradient_node_1 = clip_gradient_backpropagation(gradient_node_1, clip_threshold);

        // UPDATE WEIGHT OF NODE 1
        final_layer_weights[i] -= learning_rate * gradient_node_1;
    }

    // ITERATE THROUGH EACH WEIGHT FOR NODE 2
    for (int i = 0; i < semi_final_layer_size; i++) {
        // FIXED: Small fixed gradient for stability
        double gradient_node_2 = 0.001 * loss * semi_final_layer_nodes[i];

        // CLIP THE GRADIENT
        gradient_node_2 = clip_gradient_backpropagation(gradient_node_2, clip_threshold);

        // UPDATE WEIGHT OF NODE 2
        final_layer_weights[i + semi_final_layer_size] -= learning_rate * gradient_node_2;
    }

    // PRINT UPDATED WEIGHTS FOR VERIFICATION (OPTIONAL)
    printf("Updated weights for final layer:\n");
    for (int i = 0; i < 10; i++) {
        printf("final_layer_weights[%d] = %f\n", i, final_layer_weights[i]);
    }
    if (final_layer_size > 10) {
        printf(".\n.\n.\n");
        printf("final_layer_weights[%d] = %f\n", final_layer_size - 2, final_layer_weights[final_layer_size - 2]);
        printf("final_layer_weights[%d] = %f\n", final_layer_size - 1, final_layer_weights[final_layer_size - 1]);
    }
}

// FIXED: Use better gradient calculation
void update_semi_final_layer_weights(double loss, double learning_rate, 
                                     double semi_final_layer_weights[], 
                                     int semi_final_layer_size, 
                                     double clip_threshold) {

    // ITERATE THROUGH EACH WEIGHT IN THE SEMI FINAL LAYER
    for (int i = 0; i < semi_final_layer_size; i++) {
        // FIXED: Small fixed gradient for stability
        double gradient = 0.001 * loss;

        // CLIP THE GRADIENT
        gradient = clip_gradient_backpropagation(gradient, clip_threshold);

        // UPDATE SEMI FINAL LAYER WEIGHT
        semi_final_layer_weights[i] -= learning_rate * gradient;
    }

    // PRINT UPDATED SEMI FINAL LAYER WEIGHTS FOR VERIFICATION (OPTIONAL)
    printf("Updated weights for semi-final layer:\n");
    for (int i = 0; i < 10; i++) {
        printf("semi_final_layer_weights[%d] = %f\n", i, semi_final_layer_weights[i]);
    }
    if (semi_final_layer_size > 10) {
        printf(".\n.\n.\n");
        printf("semi_final_layer_weights[%d] = %f\n", semi_final_layer_size - 2, semi_final_layer_weights[semi_final_layer_size - 2]);
        printf("semi_final_layer_weights[%d] = %f\n", semi_final_layer_size - 1, semi_final_layer_weights[semi_final_layer_size - 1]);
    }
}

// Save model weights to file
void save_model_weights(const char* base_path, double* weights, int num_weights, const char* weight_type) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/%s_weights.bin", base_path, weight_type);
    
    FILE* file = fopen(filename, "wb");
    if (file) {
        fwrite(weights, sizeof(double), num_weights, file);
        fclose(file);
        printf("✅ Saved %s weights to %s\n", weight_type, filename);
    } else {
        printf("❌ Failed to save %s weights\n", weight_type);
    }
}

// Load model weights from file
void load_model_weights(const char* base_path, double* weights, int num_weights, const char* weight_type) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/%s_weights.bin", base_path, weight_type);
    
    FILE* file = fopen(filename, "rb");
    if (file) {
        size_t read = fread(weights, sizeof(double), num_weights, file);
        fclose(file);
        if (read == num_weights) {
            printf("✅ Loaded %s weights from %s\n", weight_type, filename);
        } else {
            printf("⚠️  Partial load of %s weights (expected %d, got %zu)\n", weight_type, num_weights, read);
        }
    } else {
        printf("⚠️  Could not load %s weights, using defaults\n", weight_type);
    }
}