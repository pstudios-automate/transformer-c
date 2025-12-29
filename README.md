# Transformer-C 
 
A complete transformer neural network implementation in C with interactive training. 
 
## Features 
- Transformer architecture with 12 attention heads 
- Interactive REPL for training and testing 
- Model saving and loading 
- Text generation and prediction 
- Built-in analysis tools 
 
## Quick Start 
 
### Build the project 
\`\`\`bash 
make 
\`\`\` 
 
### Run the REPL 
\`\`\`bash 
.\bin\transformer_c.exe 
\`\`\` 
 
## REPL Commands 
- load <model>      - Load saved model 
- train <epochs>    - Train model 
- predict <text>    - Predict next word 
- generate <length> - Generate text 
- save <name>       - Save model 
- info              - Show model info 
- weights           - Show weights 
- list              - List models 
- help              - Show help 
- exit              - Exit REPL 
 
## Model Architecture 
- Attention Heads: 12 (4 layers) 
- Embedding Dimension: 2 
- Matrix Size: 2x2 per head 
- Semi-final layer: 33,280 parameters 
- Final layer: 130 parameters 
- Max sequence length: 512 tokens 
 
## Building 
\`\`\`bash 
make clean 
make 
make test 
\`\`\` 
 
## License 
MIT License 
