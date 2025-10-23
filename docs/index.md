# 🧠 Transformer-C Documentation

Welcome to the documentation site for **Transformer-C**,  
a low-level, high-performance implementation of a Transformer neural network built entirely in **C**.

> Repository: [pstudios-automate/transformer-c](https://github.com/pstudios-automate/transformer-c)

---

## 🚀 Quick Start

# Clone the repository
git clone https://github.com/pstudios-automate/transformer-c.git
cd transformer-c

# Build and run
gcc -O2 -Wall -Wextra -fopenmp \
    main.c activation_functions.c backpropagation.c Data_Loading_Cleaning.c \
    Data_Preprocessing.c feed_forward_layer.c Tokenizer.c transformer_block.c \
    self_attention_layer.c -lm -o transformer_main
