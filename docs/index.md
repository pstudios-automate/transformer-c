# Transformer-C Documentation 
 
## Architecture 
 
### Model Components 
1. Self-Attention Layer (12 heads) 
2. Feed Forward Network 
3. Training System 
4. Model Management 
 
### File Structure 
- src/main.c - REPL interface 
- src/self_attention_layer.c - Attention mechanism 
- src/backpropagation.c - Training 
- include/transformer_block.h - Main architecture 
 
## Usage Examples 
 
### Training 
\`\`\`bash 
transformer-c> load trained_model_v1 
transformer-c> train 10 
transformer-c> save improved_model 
\`\`\` 
 
### Generation 
\`\`\`bash 
transformer-c> generate 50 
\`\`\` 
 
## Troubleshooting 
 
### Common Issues 
1. "Model not loaded" - Load or train a model first 
2. Low accuracy - Train for more epochs 
3. Memory issues - Models are ~270KB each 
