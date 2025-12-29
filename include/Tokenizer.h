#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdbool.h>

#define TABLE_SIZE 100000  // SIZE OF THE HASH TABLE

// STRUCTURE FOR THE HASH TABLE NODE
typedef struct word_token_node {
    char *word;              // THE WORD
    unsigned int token_id;   // THE UNIQUE TOKEN ID
    struct word_token_node *next; // POINTER TO THE NEXT NODE IN THE LINKED LIST
    float embedding[2];      // THE 2-VALUE WORD EMBEDDING
} word_token_node;

// ============================================
// HASH TABLE FUNCTIONS
// ============================================

// HASH FUNCTION USING DJB2 ALGORITHM
unsigned int hash(const char* word);

// INSERT WORD INTO THE HASH TABLE
void insertWord(const char* word);

// CHECK IF WORD IS ALREADY IN THE HASH TABLE
int isWordPresent(const char* word);

// GET TOKEN ID FOR A WORD
unsigned int getTokenId(const char* word);

// PRINT ALL UNIQUE WORDS, THEIR TOKEN IDS, AND EMBEDDINGS
void Print_Tokens_And_Ids();

// ============================================
// EMBEDDING FUNCTIONS
// ============================================

// GENERATE RANDOM FLOAT BETWEEN -50 to 50
float generate_random();

// GENERATE WORD EMBEDDING FOR A GIVEN TOKEN ID
float** Word_Embedding_Generation(int token_id);

// GET EMBEDDING FOR A GIVEN TOKEN ID
void getEmbeddingByTokenId(unsigned int token_id, double expected_embedding[2]);

// ============================================
// TEXT PROCESSING FUNCTIONS
// ============================================

// EXTRACT UNIQUE WORDS FROM SENTENCES AND GENERATE EMBEDDINGS
void extractUniqueWords(char** sentences);

// TEXT CLEANING FUNCTION
char* Cleaned_Text(const char* text);

// ============================================
// NOVEL CONVERSATION FUNCTIONS (FOR COMPATIBILITY)
// ============================================

// TOKENIZE TEXT (simplified version for novel conversation)
int* tokenize_text(const char* text);

// COUNT TOKENS IN ARRAY
int count_tokens(int* tokens);

// CONVERT TOKEN ID TO TEXT
char* token_to_text(int token_id);

// DETOKENIZE ARRAY BACK TO TEXT
char* detokenize(int* tokens, int length);

// GET TOKEN EMBEDDING (for novel conversation)
float* get_token_embedding(int token_id);

#endif // TOKENIZER_H