#ifndef TOKENIZER_UTILS_H
#define TOKENIZER_UTILS_H

#include <stdbool.h>

#define END_TOKEN -1
#define VOCAB_SIZE 50257

int* tokenize_text(const char* text);
int count_tokens(int* tokens);
char* token_to_text(int token_id);
char* detokenize(int* tokens, int length);
float* get_token_embedding(int token_id);

#endif
