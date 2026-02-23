# LLM Inference Engine — Implementation Status & Next Steps

This document outlines the current state of the LLM Inference Framework developed by the 4-person team. It details the extent to which the required "MUST" features have been implemented, architectural choices made, and future directions, specifically addressing the CPU-only (no GPU) limitation.

## 1. Feature Implementation Status

The objective was to build a complete, dependency-free C inference runtime capable of loading and running a Llama-architecture model from a GGUF file. **This objective has been successfully met.** All 4 core modules are fully integrated into the `main` branch.

### ✅ Module A: Kernels & Math (Person A)
* **Status**: Fully Implemented & Integrated
* **Details**: 
  * Replaced stub kernels with highly optimized raw AVX2 (+ FMA) implementations in `src/kernels/kernels.c`.
  * Implemented `gemm_f32`, `softmax`, `rmsnorm`, `silu`, and `rope`.
  * **Integration Note**: Person A's original functions were written to process raw `float*` arrays. The integrator wrapped these to accept the agreed-upon `Tensor*` struct API, preserving the AVX2 performance while maintaining the strict module contract.

### ✅ Module B: Memory Management (Person B)
* **Status**: Fully Implemented & Integrated
* **Details**: 
  * Implemented a fast bump-allocator `Arena` for permanent metadata and model structs.
  * Implemented a reusable `Scratch` allocator for intermediate tensor activations during the forward pass.
  * Implemented a `KVCache` to store historical Key/Value arrays, preventing recomputation during text generation/decode.
  * Extensively tested and verified against memory leaks.

### ✅ Module C: Model Loader & Engine (Person B/D)
* **Status**: Fully Implemented & Integrated
* **Details**: 
  * Written from scratch to dynamically parse `v2` and `v3` GGUF binary files (`src/engine/loader.c`).
  * Uses zero-copy `mmap` to map the multi-gigabyte weight tensors directly into memory instantly.
  * Implements the full 20-step Transformer forward pass (`src/engine/engine.c`) supporting Grouped-Query Attention (GQA), RoPE, and SwiGLU.

### ✅ Module D: Tokenizer & Sampler (Person C/D)
* **Status**: Fully Implemented & Integrated
* **Details**: 
  * Initially provided only as stubs.
  * The integration team parsed the GGUF metadata to extract the array of strings and scores for the tokenizer (`tokenizer.ggml.tokens` and `tokenizer.ggml.scores`).
  * Implemented a naive longest-prefix BPE-style scanner to convert strings into token IDs and detokenize IDs back into strings.
  * Implemented greedy and Temperature-based sampling logic to select the next generated token from the output logits.

---

## 2. Limitations (CPU-Only execution)

Currently, the engine relies entirely on the host CPU. Because there is no GPU or CUDA backend implemented, the system faces several hard limitations:

1. **Throughput (Tokens/sec)**:
   * While AVX2/FMA vectorization provides a massive speedup over scalar C code, matrix multiplications (GEMMs) for layers like the MLP up/down projections are heavily memory-bandwidth bound. A typical CPU can only fetch 50-100 GB/s from RAM, whereas a GPU fetches 1000-2000+ GB/s from VRAM. 
   * Expect single-digit tokens/sec on models >1 Billion parameters.
2. **Context Length**:
   * The KV cache grows linearly with sequence length. Without GPU tensor cores to parallelize attention over thousands of tokens, prefilling a large prompt (e.g., 2048+ tokens) will take several seconds to minutes on the CPU.
3. **Quantization Requirements**:
   * Currently, the engine only supports `FP32` math. 
   * A 1.1B parameter Llama model in FP32 requires ~4.4GB of RAM. A 7B model requires ~28GB of RAM. Without `INT8` or `INT4` weight quantization, running standard 7B models on consumer laptops is computationally impossible or prohibitively slow due to cache-thrashing.

---

## 3. What Can Be Done Next (Without a GPU)

Despite lacking a GPU, several critical optimizations and features can be added purely in software (C):

### A. Weight Quantization (High Priority)
* **What**: Implement 4-bit (`Q4_0`) or 8-bit (`Q8_0`) block quantization logic.
* **Why**: LLM inference is constrained by Memory Bandwidth, not Compute. By shrinking the weights 4x-8x in RAM, the CPU cache hit rate skyrockets, and AVX2 decode loops speed up drastically. This is the single biggest performance gain possible on a CPU.

### B. Threading / OpenMP
* **What**: Parallelize the loop over attention heads and rows of matrix multiplications using `pthread` or `#pragma omp parallel for`.
* **Why**: The current AVX2 AVX kernels utilize a single CPU core. Modern CPUs have 8-16+ cores that are sitting idle during the forward pass.

### C. True BPE Tokenizer
* **What**: Replace the current longest-prefix scanner with a true Byte-Pair Encoding (BPE) or SentencePiece algorithm using a Trie data structure.
* **Why**: The current fallback tokenizer is functional but naive. It may not precisely match Llama's tokenizations on complex unicode or multi-byte sequences.

### D. Flash Attention (CPU Algorithm)
* **What**: Implement a tiling-based attention algorithm to minimize memory reads/writes during the Softmax computation.
* **Why**: While famous on GPUs, Flash Attention principles (tiling) also improve L1/L2 cache locality on CPUs during large prompt prefills.

### E. Golden Data Automated Testing
* **What**: Wire the `test_golden.c` file to automatically load a downloaded GGUF file and automatically assert mathematical equivalence (<1e-4 MAE) against the PyTorch tensor dumps we generated in Phase 4.

---

## 4. What CANNOT Be Done (Until GPU Support is Added)

Certain features are fundamentally unfeasible to implement in this codebase until a backend like CUDA, Metal (Apple), or Vulkan is added to `src/kernels/`:

* **Training / Backpropagation**: The engine is purely forward-pass inference. Building autograd, optimizers, and gradient buffers on a CPU is too slow to be practical.
* **Running >13B Models Interactively**: Even with perfect CPU threading and 4-bit quantization, running a 70B parameter model on a CPU yields less than 1 token/sec. Interactive streaming is impossible.
* **Continuous Batching for Servers**: A major feature of runtimes like vLLM is batching 50+ user requests simultaneously. On a CPU, the context switching and lack of parallel SIMD blocks make high-throughput concurrent batching extremely inefficient compared to a GPU's thousands of streaming multiprocessors. 

## Conclusion
The architecture is completely modular (`engine.h` vs `kernels.h`). The correct path forward is to optimize the CPU kernels (threading & quantization) to build a fast local executable, providing a perfect baseline before bridging the `kernels.h` API to CUDA in the future.
