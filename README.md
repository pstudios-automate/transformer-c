# 🧠 transformer-c

A foundational **Transformer neural network implemented purely in C**, built for transparency, learning, and performance.  
This project reconstructs the architecture of modern attention-based models (like GPT and BERT) from scratch —  
revealing every computational step under the hood of deep learning.

---

## 🚀 Overview
`transformer-c` is a **minimal, educational, and modular implementation** of a Transformer neural network,  
written entirely in **C** to expose the true mechanics of sequence modeling and attention.

It demonstrates:
- Modular, low-level implementations of every Transformer subsystem.  
- Self-attention, feed-forward layers, normalization, and backpropagation.  
- Step-by-step numerical transparency for debugging or embedded experimentation.  
- Extensible design for OpenCL/Vulkan compute or hardware-level deployment.  

---

## 🧩 Architecture Layout
```mermaid
graph TD
  A[Input Text] -->|Tokenize| B[Tokenizer.c]
  B --> C[Embedding + Positional Encoding]
  C --> D[Self-Attention Layer]
  D --> E[Feed Forward Layer]
  E --> F[Output Probabilities]
  F --> G[Backpropagation + Weight Update]
````

---

## 📂 Project Layout

| File                        | Description                                             |
| --------------------------- | ------------------------------------------------------- |
| `main.c`                    | Entry point; runs the model and orchestrates all stages |
| `activation_functions.c/h`  | Core activations (`sigmoid`, `swish`, `relu`)           |
| `backpropagation.c/h`       | Loss functions, gradient clipping                       |
| `Data_Loading_Cleaning.c/h` | Handles text preprocessing                              |
| `Data_Preprocessing.c/h`    | Normalization and positional encoding                   |
| `feed_forward_layer.c/h`    | Fully connected neural layer                            |
| `self_attention_layer.c`    | Q/K/V attention, softmax, weighting                     |
| `transformer_block.c/h`     | Integrates all core modules                             |
| `Tokenizer.c/h`             | Token-to-ID mapping, embeddings                         |
| `text_data.txt`             | Demo dataset for sentence encoding                      |

---

## ⚙️ Build

```bash
gcc -O2 -Wall -Wextra -fopenmp \
    main.c activation_functions.c backpropagation.c Data_Loading_Cleaning.c \
    Data_Preprocessing.c feed_forward_layer.c Tokenizer.c transformer_block.c \
    self_attention_layer.c -lm -o transformer_main
./transformer_main
```

---

## 🧮 Math Overview

**Self-Attention:**
[
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
]
**Feed-Forward:**
[
FFN(x)=max(0,xW_1+b_1)W_2+b_2
]
**Positional Encoding:**
[
PE(pos,2i)=sin(\frac{pos}{10000^{2i/d_{model}}})
]
[
PE(pos,2i+1)=cos(\frac{pos}{10000^{2i/d_{model}}})
]

---

## 🧠 Mermaid Diagram — Full Transformer Pass

```mermaid
sequenceDiagram
    participant X as Input Text
    participant T as Tokenizer
    participant E as Embedding
    participant A as Self-Attention
    participant F as Feed Forward
    participant O as Output
    X->>T: Tokenize text
    T->>E: Generate embeddings
    E->>A: Compute Q,K,V and attention weights
    A->>F: Pass context vector
    F->>O: Output predictions

MIT License © 2025 **PStudios Automate**

---

## 💬 Credits

Part of **Karma-Cortex** research:

* Symbolic + Neural hybrid cognition
* GPU-native AI runtime design
* C-based introspective architecture

Made with ❤️ by **PStudios Automate**
