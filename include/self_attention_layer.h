#ifndef SELF_ATTENTION_LAYER_H
#define SELF_ATTENTION_LAYER_H

#include <stdbool.h>

float* multi_head_attention_12(float* input, int seq_len, int d_model, int n_heads);
float* scaled_dot_product_attention(float* Q, float* K, float* V, int seq_len, int d_k);
float* project_head(float* input, int head_idx, int d_k, int seq_len, int d_model, bool is_query);
float* concatenate_heads(float** heads, int seq_len, int d_model, int n_heads);
float* attention_output_projection(float* output, int seq_len, int d_model);
void apply_causal_mask(float* scores, int seq_len);

#endif
