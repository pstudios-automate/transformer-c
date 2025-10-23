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

```

┌───────────────────────────────┐
│ main.c                        │
│  ├── Data Loading & Cleaning   │
│  ├── Tokenizer (Word → ID)     │
│  ├── Preprocessing (Normalize) │
│  ├── Transformer Block         │
│  │    ├── Self-Attention       │
│  │    ├── Feed-Forward         │
│  │    └── Positional Encoding  │
│  ├── Backpropagation           │
│  └── Activation Functions      │
└───────────────────────────────┘

````

---

## 📂 Project Layout

| File | Description |
|------|--------------|
| `main.c` | Entry point. Runs the full pipeline and orchestrates training/inference. |
| `activation_functions.c/h` | Defines core activations (`sigmoid`, `leaky_relu`, `swish`). |
| `backpropagation.c/h` | Handles loss, gradient clipping, and weight updates. |
| `Data_Loading_Cleaning.c/h` | Loads text and performs basic string cleaning/splitting. |
| `Data_Preprocessing.c/h` | Handles matrix scaling, normalization, and positional encoding. |
| `feed_forward_layer.c/h` | Dense layer and weight initialization utilities. |
| `self_attention_layer.c` | Implements self-attention: `dot_product`, `softmax`, and embeddings. |
| `transformer_block.c/h` | Central module connecting all math ops and attention logic. |
| `Tokenizer.c/h` | Vocabulary handling, hashing, embedding lookup, and token ID assignment. |
| `test.c` | Matrix diagnostics and standalone testing. |
| `text_data.txt` | Sample dataset for preprocessing and tokenization tests. |

---

## ⚙️ Build Instructions

To compile:
```bash
gcc -O2 -Wall -Wextra -fopenmp \
    main.c activation_functions.c backpropagation.c Data_Loading_Cleaning.c \
    Data_Preprocessing.c feed_forward_layer.c Tokenizer.c transformer_block.c \
    self_attention_layer.c -lm -o transformer_main
````

Run the model:

```bash
./transformer_main
```

Optional:

```bash
./transformer_main | tee output.log
```

---

## 🧠 Transformer Core Concepts

### 1️⃣ Self-Attention

The self-attention mechanism calculates how strongly each token in a sequence relates to the others.
It does so using three matrices:

* **Q (Query)** — the active word asking for context
* **K (Key)** — the passive word providing context
* **V (Value)** — the encoded information being transferred

Mathematically:

```
Attention(Q, K, V) = softmax( (Q × Kᵀ) / √dₖ ) × V
```

Where:

* `Q × Kᵀ` produces similarity scores
* Division by √dₖ prevents large gradients
* `softmax()` normalizes attention weights across the sequence

---

### 2️⃣ Feed-Forward Network

Each token passes through an independent dense neural layer to increase non-linearity:

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

---

### 3️⃣ Positional Encoding

Since attention is order-agnostic, we add deterministic sinusoidal position vectors:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

This encodes relative distance directly into the embeddings.

---

### 4️⃣ Backpropagation & Gradient Clipping

Loss is computed using **Mean Squared Error (MSE)**:

```
L = (1/n) * Σ (ŷᵢ - yᵢ)²
```

Gradients are clipped to prevent overflow:

```
if (|∂L/∂W| > threshold)
    ∂L/∂W = sign(∂L/∂W) * threshold
```

---

## 🧮 Mathematical Appendix

### A. Forward Pass Equations

Given an input matrix `X`:

1. Compute queries, keys, and values:

   ```
   Q = XW_Q
   K = XW_K
   V = XW_V
   ```
2. Attention weights:

   ```
   A = softmax(QKᵀ / √dₖ)
   ```
3. Weighted sum:

   ```
   Z = A × V
   ```
4. Feed-forward projection:

   ```
   H₁ = ReLU(ZW₁ + b₁)
   H₂ = H₁W₂ + b₂
   Output = LayerNorm(H₂ + X)
   ```

---

### B. Gradient Flow (Simplified)

For each layer, compute partial derivatives:

1. **Loss → Output Gradient**

   ```
   dL/dO = 2(ŷ - y)
   ```

2. **Feed-Forward Gradients**

   ```
   dL/dW₂ = H₁ᵀ × dL/dO
   dL/dH₁ = dL/dO × W₂ᵀ
   dL/dW₁ = Xᵀ × d(ReLU(Z))
   ```

3. **Attention Gradients**

   ```
   dL/dV = Aᵀ × dL/dZ
   dL/dA = dL/dZ × Vᵀ
   dL/dQ = dL/dA × K
   dL/dK = dL/dAᵀ × Q
   ```

4. **Gradient Clipping**

   ```
   if (|grad| > clip_threshold)
       grad = sign(grad) * clip_threshold
   ```

---

### C. Numerical Stability Considerations

* Use `exp(x - max(x))` inside `softmax` to prevent overflow.
* Normalize embedding magnitudes to prevent drift.
* Apply clipping during backpropagation to maintain gradient control.
* When integrating OpenMP, ensure thread-safe operations on weight updates.

---

## 📊 Example Execution

Example log excerpt:

```
[cortex] heartbeat started (5.0 Hz)
[cortex] output snapshot:
 0.317  0.433  0.548  0.664
[reflect] fluctuation detected (Δ=0.478)
```

Indicates forward propagation and reflection checks (stability diagnostics).

---

## 🧩 Future Roadmap

| Feature                          | Status         |
| -------------------------------- | -------------- |
| Multi-head attention             | 🕓 Planned     |
| Layer normalization              | 🕓 Planned     |
| Encoder-decoder model            | 🕓 Planned     |
| GPU acceleration (OpenCL/Vulkan) | 🕓 In Research |
| Persistent model saving          | 🕓 Pending     |
| Attention visualization          | 🕓 Pending     |

---

## 🧾 License

MIT License © 2025 **PStudios Automate**

You are free to use, modify, and redistribute under the MIT License.

---

## 💬 Acknowledgements

This project is part of the **Karma-Cortex** research suite exploring:

* Symbolic + Neural hybrid cognition models
* Pure C-based AI runtime architectures
* Low-level interpretability of attention-driven systems

Created with ❤️ and curiosity by **PStudios Automate**.

