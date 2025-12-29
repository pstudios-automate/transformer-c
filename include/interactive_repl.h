#ifndef INTERACTIVE_REPL_H
#define INTERACTIVE_REPL_H

#include <stddef.h>

// ============================================================================
// REPL CORE FUNCTIONS
// ============================================================================

/**
 * @brief Start the interactive REPL (Read-Eval-Print Loop)
 */
void start_interactive_repl(void);

/**
 * @brief Initialize the REPL model state
 */
void initialize_repl_model(void);

// ============================================================================
// COMMAND HANDLERS
// ============================================================================

/**
 * @brief Load a saved model into memory
 */
void handle_load_command(const char* args);

/**
 * @brief Save the current model to disk
 */
void handle_save_command(const char* args);

/**
 * @brief Delete a saved model
 */
void handle_delete_command(const char* args);

/**
 * @brief Get information about a model
 */
void handle_info_command(const char* args);

/**
 * @brief Train the model for specified number of epochs
 */
void handle_train_command(const char* args);

/**
 * @brief Predict the next word given input text
 */
void handle_predict_command(const char* args);

/**
 * @brief Generate text of specified length
 */
void handle_generate_command(const char* args);

/**
 * @brief Evaluate text using the current model
 */
void handle_eval_command(const char* args);

/**
 * @brief Show weight statistics for the current model
 */
void handle_weights_command(void);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Print the REPL header with current model info
 */
void print_repl_header(void);

/**
 * @brief Print the help message with available commands
 */
void print_repl_help(void);

/**
 * @brief List all saved models in the saved_models directory
 */
void list_saved_models(void);

/**
 * @brief Clear the terminal screen
 */
void clear_screen(void);

// ============================================================================
// PREDICTION FUNCTIONS
// ============================================================================

/**
 * @brief Predict the next word given input text
 */
void predict_next_word(const char* text);

// ============================================================================
// MODEL STATE STRUCTURE
// ============================================================================

/**
 * @brief Structure representing the current model state
 */
typedef struct {
    char name[64];                    ///< Model name
    int is_loaded;                    ///< Whether a model is currently loaded
    double best_loss;                 ///< Best loss achieved during training
    int epochs_trained;               ///< Total epochs trained
    
    // Actual model weights (pointers to existing arrays)
    double (*k_matrix)[2];            ///< Key matrix pointer
    double (*q_matrix)[2];            ///< Query matrix pointer
    double (*v_matrix)[2];            ///< Value matrix pointer
    double* semi_final_weights;       ///< Semi-final layer weights pointer
    double* final_weights;            ///< Final layer weights pointer
    int semi_final_size;              ///< Size of semi-final weights array
    int final_size;                   ///< Size of final weights array
} ModelState;

// ============================================================================
// GLOBAL VARIABLE DECLARATIONS
// ============================================================================

/**
 * @brief Global variable holding the current model state
 */
extern ModelState current_model;

/**
 * @brief Global semi-final weights array for REPL
 */
extern double g_semi_final_weights[512 * 65];

/**
 * @brief Global final weights array for REPL
 */
extern double g_final_weights[65 * 2];

// ============================================================================
// EXTERNAL GLOBAL DECLARATIONS
// ============================================================================

/**
 * @brief External declaration of attention matrices from transformer_block.c
 */
extern double k_matrix[2][2];   ///< Key matrix from transformer_block.c
extern double q_matrix[2][2];   ///< Query matrix from transformer_block.c
extern double v_matrix[2][2];   ///< Value matrix from transformer_block.c

#endif // INTERACTIVE_REPL_H