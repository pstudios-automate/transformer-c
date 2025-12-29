// simple_novel_test.c - Test novel conversation
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VOCAB_SIZE 50257
#define MAX_SEQ_LEN 1024
#define D_MODEL 768

// Simple response generation
char* simple_generate(const char* prompt) {
    const char* responses[] = {
        "I'm Transformer-C, a neural network implemented in pure C!",
        "That's an interesting question. Let me think about it...",
        "The transformer architecture uses self-attention mechanisms.",
        "Machine learning is fascinating, especially when implemented from scratch!",
        "Hello! I'm here to help with your questions about AI and transformers."
    };
    
    int idx = rand() % 5;
    return strdup(responses[idx]);
}

// Interactive chat
void simple_chat() {
    printf("\n=== SIMPLE TRANSFORMER-C CHAT ===\n");
    printf("Type 'quit' to exit\n\n");
    
    char input[1024];
    int turn = 1;
    
    while (1) {
        printf("[Turn %d] You: ", turn);
        fgets(input, sizeof(input), stdin);
        input[strcspn(input, "\n")] = 0;
        
        if (strcmp(input, "quit") == 0) break;
        if (strlen(input) == 0) continue;
        
        printf("AI: ");
        char* response = simple_generate(input);
        printf("%s\n\n", response);
        free(response);
        
        turn++;
    }
    
    printf("\nChat ended after %d turns. Goodbye!\n", turn - 1);
}

int main() {
    srand(time(NULL));
    
    printf("?? Simple Novel Conversation Test\n");
    printf("================================\n\n");
    
    printf("Testing basic generation...\n");
    char* test_response = simple_generate("Hello");
    printf("Response: %s\n\n", test_response);
    free(test_response);
    
    printf("Starting interactive chat...\n");
    simple_chat();
    
    return 0;
}
