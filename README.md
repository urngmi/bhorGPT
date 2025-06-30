## Simple ASCII function calling tree for `gpu.cu`

```text
main()
│
├── cublasCreate()
├── model.load()
│   ├── fread(...) x5         <-- Loads constants: C, E, D, H, O
│   ├── tra[i].load()         <-- Transformer blocks
│   │   ├── a.load()
│   │   ├── b.load()
│   │   └── c.load()
│   └── out.load()
│
├── model.generate(input, n)
│   ├── s.push_back(...)                <-- Initial char append
│   ├── memset(vs, 0)
│   ├── loop: for each char
│   │   └── sample(x, p)
│   │       ├── emb.data = _wyhash64()  <-- Initializes input embedding
│   │       ├── tra[d].fw(...)          <-- Transformer forward
│   │       │   ├── a.fw()
│   │       │   │   ├── u.fw()
│   │       │   │   │   ├── linear::fw()
│   │       │   │   │   │   ├── _s16() (kernel)
│   │       │   │   │   │   ├── cublasTSSgemvStridedBatched()
│   │       │   │   │   │   └── _l16() (kernel)
│   │       │   │   ├── _layernorm() (kernel)
│   │       │   │   ├── _selffsuv() (kernel)
│   │       │   │   └── o.fw()
│   │       │   ├── b.fw()
│   │       │   │   ├── x.fw()
│   │       │   │   ├── _layernorm() (kernel)
│   │       │   │   ├── _s16() x3 (kernel)
│   │       │   │   ├── cublasTSSgemvStridedBatched() x2
│   │       │   │   ├── _sexyfp() (kernel)
│   │       │   │   ├── _sexyfsuv() (kernel)
│   │       │   │   ├── _layernorm() (kernel)
│   │       │   │   ├── o.fw()
│   │       │   │   └── _sexyadd() (kernel)
│   │       │   └── c.fw()
│   │       │       ├── u.fw()
│   │       │       │   └── linear::fw() (same as above)
│   │       │       ├── _layernorm() (kernel)
│   │       │       ├── _selffsuv() (kernel)
│   │       │       ├── o.fw()
│   │       │       └── _sexyadd() (kernel)
│   │       ├── _layernorm() (kernel)
│   │       └── out.fw()
│   │           └── linear::fw()
│   ├── cudaDeviceSynchronize()
│   ├── softmax()
│   ├── wy2u01(wyrand(&prng))   <-- Sampling next token
│   └── s.push_back(c)          <-- Append next token
│
├── gettimeofday() x2
└── cublasDestroy()

Miscellaneous
├── _wyhash64(A, B)            <-- Custom 64-bit hash
│   └── _wymum(A, B)
├── wy2u01(r)                  <-- Int → Float conversion [0,1)
├── wyrand(&seed)             <-- Pseudo-random number gen
├── _s16() / _l16()           <-- Float ↔ bfloat16 (CUDA kernels)
├── _layernorm()              <-- Standard transformer normalization (CUDA)
├── _sexyfp(), _sexyfsuv(), _sexyadd(), _selffsuv()  <-- Custom fused ops
```

---

## 🧠 SYSTEM CHAIN: Transformer LLM Training (Chronological)

---

### 🟢 PHASE 0: Data Processing

#### 🔸 Step 0.1 — **Raw Text Input**

* **Input**: Large corpus (books, code, forums, web pages, etc.)

#### 🔸 Step 0.2 — **Tokenization**

* **Task**: Split text into token IDs
* **Tool**: Byte Pair Encoding (BPE), SentencePiece, etc.
* **Example**:

  ```
  "The cat sat" → [1212, 389, 789]
  ```

#### 🔸 Step 0.3 — **Create Training Samples**

* Format: Pairs of `(input_tokens, target_tokens)`
* Example:

  ```
  Input: [1212, 389] → Predict: [389, 789]
  ```

---

## 🔁 REPEATED FOR EACH TRAINING BATCH

---

### 🟦 1. Embedding + Positional Encoding

#### 🔸 Step 1.1 — **Token Embedding**

* **Map**: Token IDs → Dense vectors
  `token_embeddings = Embedding(input_tokens)`

#### 🔸 Step 1.2 — **Add Positional Embeddings**

* Final input to model:
  `x = token_embeddings + positional_embeddings`

---

### 🟦 2. Transformer Blocks (N layers deep)

Repeat the below for each Transformer layer.

#### 🔷 a. **Layer Norm 1**

#### 🔷 b. **QKV Projections**

* Compute:
  `Q = xW_Q`, `K = xW_K`, `V = xW_V`

#### 🔷 c. **Scaled Dot-Product Attention**

* Compute:
  `scores = QKᵀ / √d_k`

#### 🔷 d. **Causal Masking**

* Prevents tokens from attending to the future.

#### 🔷 e. ✅ **Softmax Over Scores**

* Task: Attention weights
  `weights = softmax(scores)`

#### 🔷 f. **Attention Output**

* `attn_output = weights × V`

#### 🔷 g. **Output Projection + Residual + LayerNorm**

#### 🔷 h. **Feedforward Network (MLP)**

* `FFN(x) = Linear2(Activation(Linear1(x)))`

#### 🔷 i. **Residual + LayerNorm Again**

---

### 🟦 3. Final Output Processing

#### 🔸 Step 3.1 — **Final Layer Norm**

#### 🔸 Step 3.2 — **Linear Output Head**

* Projects to vocab size:

  ```
  logits = x × W_vocabᵗ
  # Shape: [batch_size, seq_len, vocab_size]
  ```

---

### 🟦 4. ✅ **Softmax + Loss (Key Training Step)**

#### 🔸 Step 4.1 — **Cross-Entropy Loss**

* Ground truth next tokens: `y = [389, 789]`
* Compute per-token loss:

  ```
  loss = CrossEntropy(softmax(logits), y)
  ```
* Or more efficiently:

  ```
  loss = CrossEntropyWithLogits(logits, y)
  ```

---

### 🟦 5. ✅ **Backpropagation**

#### 🔸 Step 5.1 — Compute Gradients

* Call `.backward()` on loss
* Uses automatic differentiation (autograd)

---

### 🟦 6. ✅ **Optimizer Step**

#### 🔸 Step 6.1 — Weight Update

* Use optimizer (e.g., **AdamW**):

  ```
  optimizer.step()
  ```

#### 🔸 Step 6.2 — Zero Gradients

* Clear gradients for next batch:

  ```
  optimizer.zero_grad()
  ```

---

### 🟦 7. Optional Training Utilities

| Component                           | Purpose                             |
| ----------------------------------- | ----------------------------------- |
| **Gradient Clipping**               | Prevent exploding gradients         |
| **Learning Rate Scheduler**         | Warmup and cosine decay             |
| **Mixed Precision Training (FP16)** | Speeds up training with less memory |
| **Checkpointing**                   | Save model state                    |
| **Logging**                         | Track loss, perplexity, LR, etc.    |

---

## ✅ Summary: Chronological Key Tasks

| Stage | Task                  | Module            |
| ----- | --------------------- | ----------------- |
| 0.2   | Tokenization          | Tokenizer         |
| 1.1   | Embedding Lookup      | Embedding Matrix  |
| 2.b   | QKV Computation       | Linear Layers     |
| 2.c   | Attention Scores      | Dot Product       |
| 2.e   | **Softmax #1**        | Attention Weights |
| 3.2   | Output Logits         | Output Head       |
| 4.1   | **Softmax #2 + Loss** | CrossEntropy      |
| 5     | **Backpropagation**   | Autograd          |
| 6     | **Optimizer Step**    | AdamW             |

---

### 🧪 Final Note:

During **inference**, only **steps 1–3** are used, and **no gradients, losses, or optimizer steps** are run. Training is vastly more computationally expensive because it includes the full **forward + backward pass**.

---

Here is a **chronological system-level chain** of how a **Transformer-based LLM (like GPT)** processes a single input prompt, **step-by-step**. I’ll explicitly mark **key tasks**, like `Softmax`, and what happens at each stage of the **forward pass**.

---

### 🧠 System Chain: Transformer LLM (Autoregressive, Decoder-only like GPT)

---

#### 🟦 0. **Input Tokenization**

* **Task**: Convert raw input string to tokens (IDs).
* **Example**: `"The cat sat"` → `[1212, 389, 789]`
* **Module**: `Tokenizer` (e.g., Byte-Pair Encoding, SentencePiece)

---

#### 🟦 1. **Embedding Lookup**

* **Task**: Map token IDs to vectors.
* **Example**: `[1212, 389, 789]` → `[v₁, v₂, v₃] ∈ ℝᵈ`
* **Module**: `Embedding Layer` (learned matrix of shape `[Vocab_Size × d_model]`)

---

#### 🟦 2. **Add Positional Encoding**

* **Task**: Add position information to token embeddings.
* **Module**: Sinusoidal or learned `Positional Embeddings`
* **Operation**:

  ```
  x = TokenEmbedding + PositionalEmbedding
  ```

---

#### 🟦 3. **Input to First Transformer Block**

* **Task**: Start of deep computation.
* **Input**: `x ∈ ℝ^{seq_len × d_model}`

---

### 🔁 Transformer Block (repeated N times, usually 12 to 96+)

#### 🔷 a. **Layer Norm 1**

* **Task**: Normalize input for stability.
* **Operation**:

  ```
  x_norm = LayerNorm(x)
  ```

---

#### 🔷 b. **Self-Attention Mechanism**

##### 🔸 Step 1: Linear Projections

* **Task**: Compute Q, K, V
* **Module**: 3 separate Linear layers
* **Operation**:

  ```
  Q = xW_Q, K = xW_K, V = xW_V
  ```

##### 🔸 Step 2: Scaled Dot Product

* **Task**: Compute attention scores
* **Operation**:

  ```
  scores = QKᵀ / √d_k
  ```

##### 🔸 Step 3: **Causal Masking**

* **Task**: Prevent peeking at future tokens.
* **Operation**: Set upper triangle of `scores` to `-∞`.

##### 🔸 Step 4: **Softmax**

* ✅ **This is where Softmax happens**
* **Task**: Turn scores into probabilities
* **Operation**:

  ```
  attn_weights = softmax(scores)
  ```

##### 🔸 Step 5: Apply Weights

* **Task**: Weighted sum over values
* **Operation**:

  ```
  attn_output = attn_weights × V
  ```

##### 🔸 Step 6: Output Projection

* **Task**: Project attention output back
* **Operation**:

  ```
  attn_output = attn_output × W_O
  ```

---

#### 🔷 c. **Residual Connection**

* **Task**: Add input to output.
* **Operation**:

  ```
  x = x + attn_output
  ```

---

#### 🔷 d. **Layer Norm 2**

* **Task**: Normalize again before feedforward.
* **Operation**:

  ```
  x_norm = LayerNorm(x)
  ```

---

#### 🔷 e. **Feedforward Network (MLP)**

* **Task**: Non-linear transformation.
* **Structure**:

  * Linear → GELU/ReLU → Linear
* **Operation**:

  ```
  x_ffn = Linear₂(Activation(Linear₁(x_norm)))
  ```

---

#### 🔷 f. **Residual Connection**

* **Task**: Add again.
* **Operation**:

  ```
  x = x + x_ffn
  ```

---

### 🔁 End of Transformer Block (repeat N times)

---

#### 🟦 4. **Final Layer Norm**

* **Task**: Normalize before output.
* **Module**: `LayerNorm`

---

#### 🟦 5. **Output Projection (Language Modeling Head)**

* **Task**: Convert to vocabulary logits.
* **Operation**:

  ```
  logits = x × W_vocabᵗ  # shape = [seq_len × Vocab_Size]
  ```

---

#### 🟦 6. **Softmax (Again)**

* ✅ **Another Softmax here**
* **Task**: Convert logits to probabilities over vocabulary.
* **Used during inference only**, not training (where logits go to cross-entropy).
* **Operation**:

  ```
  probs = softmax(logits)
  ```

---

#### 🟦 7. **Sampling / Argmax**

* **Task**: Choose the next token.

  * **Greedy**: `argmax(probs)`
  * **Sampling**: Based on distribution
  * **Top-k / Top-p**: Constrained sampling

---

#### 🟦 8. **Repeat (Autoregressive Loop)**

* Feed the next token back to step 1.
* Continue until:

  * End-of-text token
  * Max length reached

---

### ✅ Summary Table

| Step | Module              | Key Task               |
| ---- | ------------------- | ---------------------- |
| 0    | Tokenizer           | Tokenize input         |
| 1    | Embedding           | Convert ID to vector   |
| 2    | Positional Encoding | Inject position        |
| 3    | Transformer Block   | Main processing        |
| a    | QKV Projection      | Linear layers          |
| b    | Attention Scores    | `QKᵀ / √d_k`           |
| c    | **Softmax**         | Turns scores → weights |
| d    | Weighted Sum        | Output attention       |
| e    | MLP                 | Deep transformation    |
| 4    | Final Norm          | Normalize again        |
| 5    | Output Head         | Map to vocab logits    |
| 6    | **Softmax**         | Final probabilities    |
| 7    | Sampling            | Decide next token      |
| 8    | Loop                | Generate next token    |

---

Let me know if you want this with visual diagrams or annotated source code.
