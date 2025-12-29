#ifndef NOVELTY_H
#define NOVELTY_H

#include <stdbool.h>

#define NOVELTY_WINDOW 20
#define NOVELTY_THRESHOLD 0.3

typedef struct {
    float* novelty_matrix;
    float novelty_threshold;
    int novelty_window;
    int* recent_tokens;
    int recent_index;
} NoveltyDetector;

NoveltyDetector* init_novelty_detector(int vocab_size);
void update_novelty_detector(NoveltyDetector* detector, int token_id);
float calculate_novelty_score(int token_id, NoveltyDetector* detector);
float xor_novelty_pattern(int prev_token, int current_token);
void free_novelty_detector(NoveltyDetector* detector);

#endif