#ifndef NOVEL_CONVERSATION_H
#define NOVEL_CONVERSATION_H

#include <time.h>
#include <stdbool.h>

// ================================================
// ARCHITECTURE CONSTANTS
// ================================================
#define VOCAB_SIZE 50257
#define MAX_SEQ_LEN 1024
#define D_MODEL 768
#define D_FF 3072
#define N_HEADS 12
#define N_LAYERS 12
#define D_HEAD 64
#define TOP_K 50
#define TOP_P 0.9
#define TEMPERATURE 0.8
#define NUM_EXPERTS 8
#define EXPERT_TOP_K 2
#define END_TOKEN -1

// ================================================
// STRUCT DEFINITIONS
// ================================================

typedef struct {
    int* token_ids;
    int length;
    int max_length;
    float* hidden_states;
    float* attention_mask;
    float temperature;
    float top_p;
    int top_k;
    bool use_nucleus;
} GenerationContext;

typedef struct {
    float* novelty_matrix;
    float* token_counts;
    int* recent_tokens;
    float* novelty_scores;
    int recent_index;
    int novelty_window;
    float novelty_threshold;
    int vocab_size;
} NoveltyDetector;

typedef struct {
    float* router_weights;
    float* expert_weights[8];
    float* expert_biases[8];
    float* output_weights[8];
    float* output_biases[8];
    int active_experts;
} MoELayer;

// ================================================
// FUNCTION DECLARATIONS
// ================================================

// Core transformer
float* transformer_forward(float* hidden_states, int seq_len, int layer_idx);
float* multi_head_attention(float* input, int seq_len, int d_model, int n_heads, float* mask, bool training);
float* scaled_dot_product_attention(float* Q, float* K, float* V, int seq_len, int d_k, float* mask);
float* layer_norm(float* x, int size, float eps);
float* feed_forward(float* x, int d_model, int d_ff);

// Mixture of Experts
MoELayer* create_moe_layer(int d_model, int d_ff, int num_experts, int top_k);
float* moe_forward(MoELayer* layer, float* input, int seq_len, int d_model);
float* router_forward(MoELayer* layer, float* input, int seq_len);
void get_top_k_experts(float* router_logits, int num_experts, int top_k, int* indices, float* weights);
float* expert_forward(MoELayer* layer, int expert_idx, float* input, int seq_len);
void free_moe_layer(MoELayer* layer);

// Novelty detection
NoveltyDetector* create_novelty_detector(int vocab_size, int window_size);
void update_novelty_detector(NoveltyDetector* detector, int prev_token, int curr_token);
float calculate_token_novelty(NoveltyDetector* detector, int token_id);
float xor_novelty_pattern(int prev_token, int curr_token);
void free_novelty_detector(NoveltyDetector* detector);

// Generation
int sample_token(float* logits, GenerationContext* ctx);
void apply_top_p_sampling(float* probs, int size, float top_p);
void apply_top_k_sampling(float* probs, int size, int top_k);
bool should_stop_generation(int token, int* context, int length, float* hidden_states);

// Conversation
char* generate_response(const char* prompt, int max_new_tokens, float temperature, float top_p, int top_k);
char* build_conversation_prompt(const char* input, int turn_count, char** history, int history_len);
void interactive_conversation(void);

// Utilities
void print_novelty_stats(NoveltyDetector* detector);
void print_generation_stats(int turn_count, time_t session_start, float avg_novelty, float avg_attention);
float calculate_average_novelty(NoveltyDetector* detector);

// Matrix operations
void matmul(float* A, float* B, float* C, int m, int n, int p);
void matmul_transpose(float* A, float* B, float* C, int m, int n, int p);
void matadd(float* A, float* B, float* C, int size);

#endif
