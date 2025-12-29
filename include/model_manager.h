#ifndef MODEL_MANAGER_H
#define MODEL_MANAGER_H

#include <stddef.h>

// ============================================================================
// MODEL SAVE/LOAD FUNCTIONS
// ============================================================================

/**
 * @brief Save complete model state to disk
 */
void save_model(const char* model_name, 
                double k_matrix[][2], double q_matrix[][2], double v_matrix[][2],
                double* semi_final_weights, double* final_weights,
                int semi_final_size, int final_size,
                double best_loss, int epochs_trained, const char* vocabulary_path);

/**
 * @brief Load complete model state from disk
 */
int load_model(const char* model_name,
               double k_matrix[][2], double q_matrix[][2], double v_matrix[][2],
               double* semi_final_weights, double* final_weights,
               int semi_final_size, int final_size,
               double* best_loss, int* epochs_trained);

/**
 * @brief List all saved models
 */
void list_saved_models(void);

/**
 * @brief Delete a saved model
 */
int delete_model(const char* model_name);

/**
 * @brief Get detailed information about a model
 */
void get_model_info(const char* model_name);

#endif // MODEL_MANAGER_H