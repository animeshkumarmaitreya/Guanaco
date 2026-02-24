# Guanaco Framework: Comprehensive Feature Catalog

This document provides an exhaustive, granular inventory of every single feature, data structure, and technical capability currently implemented in the **Guanaco** LLM Inference engine.

---

## 1. Loader & GGUF Parser (`src/engine/loader.c`)
* **Dynamic Binary Parsing**: Reads the raw binary bytes of GGUF formatted files according to the official specifications.
* **Format Support**:
  * ✅ Support for `v2` GGUF binary format.
  * ✅ Support for `v3` GGUF binary format.
* **Metadata Extraction**:
  * Extracts nested Key-Value pairs natively (Strings, `uint32`, `uint64`, `float32`, arrays).
  * Auto-extracts Model Configuration (Hidden dimension size, attention head counts, KV head counts for GQA, number of transformer layers, feed-forward dimensions).
  * Auto-extracts RoPE (Rotary Positional Embedding) configurations (e.g., base dimensions).
  * Auto-extracts Tokenizer dictionaries (`tokenizer.ggml.tokens` and `tokenizer.ggml.scores`) directly from the binary header without needing a separate `tokenizer.json` file.
* **Tensor Registry & Mapping**:
  * Maps string names from the GGUF file (e.g., `blk.0.attn_q.weight`) to the exact pointer layer in the `ModelWeights` C struct.
  * ✅ Tied Embedding Support: Detects if the model ties its input word embeddings to its final output classifier weights (`output.weight`), preventing null-pointer crashes.
* **Zero-Copy Execution (`mmap`)**:
  * Uses POSIX `mmap` with `MAP_SHARED` and `MADV_RANDOM`.
  * Allows the instant mapping of 20GB+ models into virtual memory instantly. The OS dynamically pages these weights from your SSD into RAM, bypassing the need to load the file manually.

---

## 2. Transformer Engine & Generation Loop (`src/engine/engine.c` & `generate.c`)
* **Transformer Forward Pass**:
  * A mathematically precise implementation of the LLaMA/Mistral transformer block.
  * Steps dynamically executed per token:
    1. Token Embedding Lookup.
    2. Optional Pre-Attention RMSNorm.
    3. Query, Key, and Value Projection (QKV).
    4. Grouped-Query Attention (GQA) mapping.
    5. RoPE Positional Encoding application to Q and K.
    6. KV Cache Append (writes new Key/Values to history).
    7. Multi-Head Attention Scoring ($Q \times K^T / \sqrt{d}$) with Causal Masking (ignoring future tokens).
    8. Softmax across attention scores.
    9. Attention Output projection ($Scores \times V$).
    10. Final Output Projection ($Output \times W_{o}$)
    11. Residual Connection Additions.
    12. Pre-MLP RMSNorm.
    13. SwiGLU Feed-Forward Network ($Swish(x \times W_{gate}) \times (x \times W_{up}) \times W_{down}$).
    14. Final RMSNorm.
    15. Logits projection mapping hidden states back to vocabulary size.
* **Causal Masking**: Automatically masks future tokens with $-INFINITY$ during the Prefill phase, ensuring the model only pays attention to previous words.
* **Autoregressive Generate Loop**:
  * A continuous loop in `generate.c` that loops the forward pass until the model emits an End-Of-Sequence (`EOS`) token or reaches `max_tokens`.
* **Runtime Profiling**: Output prints the performance metric of `Tokens/sec`.

---

## 3. Kernels & Mathematics (`src/kernels/kernels.c`)
* **Hardware Acceleration**:
  * Heavily utilizes **AVX2 (Advanced Vector Extensions 2)** and **FMA (Fused Multiply-Add)** compiler intrinsics.
  * Processes 8 `FP32` numbers simultaneously per CPU clock cycle.
* **General Matrix Multiplication (`gemm_f32`)**:
  * Custom $C = A \times B$ mathematical engine.
  * Capable of handling arbitrary shapes, handling both gigantic matrix-matrix prefill multiplies, and narrow matrix-vector decode hot-paths.
* **RMSNorm (Root Mean Square Normalization)**:
  * Calculates variance securely, adding user-defined Epsilon (e.g., `1e-6`) to prevent division by zero, before scaling by the learned weight layer.
* **Softmax**:
  * Implements `softmax_inplace` with Max-Subtraction. Subtracts the maximum value from all elements before calculating `exp(x)` to prevent catastrophic floating-point infinity overflow.
* **SiLU (Swish-1)**:
  * Implements $x \times Sigmoid(x)$ in-place for the MLP activation functions.
* **RoPE (Rotary Positional Embeddings)**:
  * Applies complex-number rotation math to the Query and Key tensors based on their absolute sequence position natively, required for the model to understand the order of words.

---

## 4. Memory Allocators & Caching (`src/memory/`)
* **The Arena Allocator (`arena.c`)**:
  * A bump-allocator that assigns permanent memory at startup. Used for the `ModelConfig`, pointers, and the layout of the `ModelWeights` struct. Completely bypasses native OS `malloc` fragmentation.
* **The Scratch Allocator (`scratch.c`)**:
  * A ring-buffer style dynamic allocator used during the live forward pass.
  * Used for temporary intermediate matrices (like the $Q$, $K$, and $V$ tensors). Fast pointer resets (`scratch_reset()`) drastically speed up layer iterations by reusing the exact same L2/L3 CPU cache layout block.
* **The KV Cache (`kv_cache.c`)**:
  * A system that stores historical Key and Value tensors.
  * Eliminates exponential recomputation. Without this, processing the 100th word would require re-calculating the matrix products for the previous 99 words.
  * Dynamically fetches historical Context arrays contiguous in memory for fast AVX2 dot-products during Attention.

---

## 5. Tokenizer & Sampler (`src/tokenizer/tokenizer.c`)
* **Byte-Pair Encoding (BPE) Simulation**:
  * Takes user string input, and parses it byte-by-byte via a Longest-Prefix Match against the 32,000+ vocabulary tokens extracted from the GGUF metadata.
* **Detokenizer**:
  * Handles reversing token IDs back into printable text.
  * Explicitly handles HuggingFace / SentencePiece space character mapping (translates the `\xe2\x96\x81` / `\u2581` block character back into standard ASCII spaces ` ` before printing to the terminal).
* **Sampling Suite**:
  * Consumes the `FP32` Logit probability array that the engine outputs.
  * **Greedy Sampling (`sample_argmax`)**: Scans 32,000 logits to find the absolute maximum mathematical probability (temperature = 0).
  * **Temperature Sampling (`sample_temperature`)**:
    * Adjusts the probability distribution dynamically.
    * Lower temperature (`0.1`) makes the model more deterministic and confident.
    * Higher temperature (`1.0`) makes the math flatter, allowing the model to randomly select slightly less probable words for creative outcomes.

---

## 6. Testing & Quality Assurance
* 33 Hand-built, isolated Unit Tests natively written in C.
* **Kernel Tests**: Mathematically verifies that AVX2 outputs identical results to Python across extreme edge-cases (e.g., NaNs, -Infinities, single vector lines, and massive matrix bounds).
* **Memory Tests**: Validates that Arena bump allocation structurally guarantees pointer alignment constraints, and proves the Scratch allocator safely overwrites data upon reset.
* **Engine Tests**: Verifies total architectural wiring, creating fake GGUF configurations and running dummy forward passes across 20 layers without segfaulting.
* **Golden Data Script**: Includes a local Python/PyTorch HuggingFace script (`gen_golden.py`) to scrape Ground-Truth mathematically perfect intermediate tensors from TinyLlama to compare against engine output.
