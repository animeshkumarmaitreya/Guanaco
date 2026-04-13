# Guanaco LLM Inference Runtime — Feature Extensions Roadmap

This document proposes technically-scoped extensions for the Guanaco local LLM inference runtime (pure C, CPU-first), with optional CUDA acceleration and minimal external dependencies.

The goal is to extend functionality in ways that are strongly aligned with a systems programming course: memory layout, backend dispatch, kernel implementation choices, profiling, and correctness validation.

---

## 0) Current baseline (what exists today)

**Entry points & flow**
- CLI: [src/main.c](src/main.c) calls `cli_parse()` and then `generate()`.
- CLI parsing + samplers: [src/tokenizer/tokenizer.c](src/tokenizer/tokenizer.c).
- Orchestration: [src/engine/generate.c](src/engine/generate.c) (load model → allocators → KV → tokenize → prefill → decode loop).
- Forward pass & transformer layer: [src/engine/engine.c](src/engine/engine.c) (`forward()` and `transformer_layer()`).
- GGUF loader: [src/engine/loader.c](src/engine/loader.c) mmaps weights and builds `ModelWeights` + `Tensor` views.
- Kernels: stateless ops declared in [src/include/kernels.h](src/include/kernels.h), implemented in [src/kernels/kernels.c](src/kernels/kernels.c).
- Memory/KV: Arena + Scratch + KV cache in [src/include/memory.h](src/include/memory.h) and [src/memory/](src/memory/).

**Important implementation details (affect extensions)**
- `forward(model, kv, scr, token_ids, n_tokens, pos)` already supports “incremental position” semantics: RoPE and KV append happen at `pos + t`. This is perfect for chat / streaming.
- Sampler flags exist (`--top-k`, `--top-p`) but `sample_top_k()` and `sample_top_p()` currently fall back to `sample_temperature()` (no true top-k/top-p yet).
- Loader sets `Tensor.dtype` for quantized types (Q4/Q8) but the compute path assumes FP32 tensors. True quantized inference requires either load-time dequantization or quantized kernels.

---

## 1) Backend selection (CPU vs CUDA) — shared concept

### Proposed CLI

Add one flag:
- `--device auto|cpu|cuda`
  - `auto` (default): use CUDA if the binary was built with CUDA support and a device is present; otherwise use CPU.
  - `cpu`: force CPU backend.
  - `cuda`: force CUDA backend; error if CUDA not available.

Optionally add:
- `--backend-report` (prints which backend is selected and why)

### Runtime detection (auto)

- If compiled without CUDA support: always CPU.
- If compiled with CUDA support: call `cudaGetDeviceCount()`; if count > 0, select CUDA.

### Key performance caution

Offloading *only* GEMM/matvec while keeping activations on CPU can be slower due to PCIe copies per layer. The recommended first CUDA milestone is **GPU-accelerated prefill only** (larger GEMMs amortize transfer costs). Decode offload is a second milestone.

### Backend architecture sketch (keeps the codebase clean)

To keep GPU support from turning into a web of `#ifdef USE_CUDA`, add a small backend abstraction that owns *how* kernels are executed:

- `BackendKind`: CPU / CUDA
- `KernelVTable`: function pointers matching the ops in [src/include/kernels.h](src/include/kernels.h)
- `BackendContext`: optional state (CUDA streams, handles, scratch workspace)

The engine selects a backend once (based on `--device`) and then calls `backend->kernels.gemm_f32(...)` etc.

Critical note for your current engine: attention in [src/engine/engine.c](src/engine/engine.c) is computed using explicit dot-product loops. A “GEMM-only” CUDA backend will not accelerate that portion unless you refactor attention to express:

- scores = Q · Kᵀ (per head)
- context = softmax(scores) · V (per head)

via `gemm_f32()` (CPU first). This refactor tends to improve CPU performance *and* makes CUDA GEMM offload much more likely to win.

---

## 2) Feature: Conditional GPU inference (CUDA) with CPU fallback

### Goal

Run inference on the RTX 3050 (laptop) when present, but preserve CPU inference as the default fallback.

### Phased plan

**Phase 0 (high leverage): attention-as-GEMM refactor (CPU first)**
- Rewrite attention score and attention×V computations in terms of GEMM calls.
- Keep the numerically stable softmax and masking behavior unchanged.

**Phase A (recommended): CUDA prefill only**
- Use CUDA for the prompt prefill (T > 1) where GEMMs are larger.
- Keep decode (T = 1) on CPU.

**Phase B: CUDA decode matvec**
- Offload the M=1 GEMM shapes that dominate decode.
- Mitigate overhead with pinned memory and async copies (optional).

**Phase C: full CUDA forward**
- Maintain hidden activations and KV cache on GPU across steps.
- Requires more refactoring but eliminates repeated transfers.

### Implementation approach choices

**Option 1: custom CUDA kernels (preferred first if feasible)**
- Implement a small set of CUDA kernels that mirror your existing kernel API.
- Start with GEMM/matvec; then consider RMSNorm and simple elementwise ops.

**Option 2: pivot to CUDA Toolkit libraries if custom GEMM is too hard**
- Use cuBLAS for GEMM/matvec.
- Still “dependency-minimal” in the sense that it relies only on CUDA Toolkit, not PyTorch/TensorRT.

### Where to integrate

- CLI plumbing: [src/tokenizer/tokenizer.c](src/tokenizer/tokenizer.c), [src/include/tokenizer.h](src/include/tokenizer.h)
- Orchestration / device selection: [src/main.c](src/main.c) and/or [src/engine/generate.c](src/engine/generate.c)
- Kernel dispatch: [src/engine/engine.c](src/engine/engine.c) (calls to `gemm_f32`, etc.)
- CUDA kernels/wrappers: new files under [src/kernels/](src/kernels/) (e.g., `cuda_gemm.cu`, `cuda_backend.c`)
- Build: [Makefile](Makefile) add `USE_CUDA=1` mode

### Verification

- **Correctness**: a debug mode that runs CPU and CUDA back-to-back for a tiny prompt and compares logits with a tolerance.
- **Performance**: compare TTFT and tok/s printed by [src/engine/generate.c](src/engine/generate.c).

---

## 3) Feature: Quantized model support (Q4_0 / Q8_0 GGUF)

### Goal

Support quantized GGUF models while keeping the runtime simple.

### Phased plan

**Phase A (fastest): load-time dequantization to FP32**
- In [src/engine/loader.c](src/engine/loader.c), if a tensor is Q4_0 or Q8_0:
  - Allocate an FP32 buffer.
  - Dequantize into FP32.
  - Point `Tensor.data` to FP32 and set dtype to FP32 for runtime compute.
- Pros: minimal changes to engine/kernels.
- Cons: higher RAM use.

**Phase B: true quantized decode matvec**
- Keep weights quantized and implement a decode matvec kernel that dequantizes blocks on-the-fly.
- This targets the decode hot path.

**Phase C: quantized GEMM for prefill**
- Stretch goal.

### Where to integrate

- Dequant path: [src/engine/loader.c](src/engine/loader.c)
- Optional quant kernels: [src/kernels/kernels.c](src/kernels/kernels.c)

### Verification

- Add a unit test for Q4/Q8 dequantization correctness on a few blocks.
- Run end-to-end generation on a Q4 model and ensure no NaNs and stable output.

---

## 4) Feature: Context-based chat (interactive REPL) with KV reuse

### Goal

Implement a multi-turn interactive chat CLI that preserves context efficiently.

### Recommended UX

- `--chat` to enter a REPL.
- User types a message; engine generates assistant tokens; repeat.

### Design (maps well to current code)

- Keep these alive for the chat session:
  - `ModelWeights*`, `Tokenizer*`, `Arena*`, `Scratch*`, `KVCache*`
- Maintain:
  - A token history buffer
  - `current_pos` (monotonic position)

Incremental turn processing:
- Tokenize only the new user text.
- Call `forward(model, kv, scr, new_tokens, new_len, current_pos)`.
- Generate tokens, append them to history, advance `current_pos`.

Context window management:
- When `current_pos` approaches `max_seq_len`:
  - simplest: reset KV + re-prefill last W tokens (“sliding window via re-prefill”)

### Where to integrate

- CLI parsing: [src/tokenizer/tokenizer.c](src/tokenizer/tokenizer.c)
- Chat loop: best added in [src/engine/generate.c](src/engine/generate.c)
- Mode selection: [src/main.c](src/main.c)

### Verification

- Manual: multi-turn conversation references earlier turns.
- Debug: print `current_pos` and ensure no overflow.

---

## 5) Feature: Real top-k / top-p sampling + reproducibility

### Goal

Make sampler flags real and testable.

### Implementation

- Implement true top-k (partial selection):
  - select k largest logits (no full sort needed)
  - compute normalized probs over those k
  - sample
- Implement true top-p:
  - compute probs
  - sort candidates by prob desc
  - keep minimal prefix where cumulative prob ≥ p
  - renormalize, sample
- Add `--seed` and call `srand(seed)` for deterministic runs.

### Where to integrate

- Sampling code + CLI: [src/tokenizer/tokenizer.c](src/tokenizer/tokenizer.c)
- Tests: extend [tests/test_tokenizer.c](tests/test_tokenizer.c)

---

## 6) Feature: Tokenizer upgrade (still dependency-free)

### Goals

- Improve performance and robustness.
- Better match LLaMA-family tokenization behavior.

### Implementation options

- Build a trie for `cfg->vocab_strings` at init; greedy scan using trie rather than scanning full vocab each position.
- Add byte-fallback for unknown bytes.
- (Optional) Use vocab scores to implement a closer-to-BPE merge procedure.

### Where to integrate

- [src/tokenizer/tokenizer.c](src/tokenizer/tokenizer.c)

### Verification

- Round-trip tests for representative strings (ASCII + UTF-8)
- Microbench mode to measure tokens/s

---

## 7) Feature: CPU throughput improvements with threads

### Goal

Scale throughput on multicore CPUs without large dependencies.

### Implementation

- Add `--threads N`.
- Parallelize:
  - GEMM tiles
  - attention across heads
- Use pthreads.

### Where to integrate

- Kernels: [src/kernels/kernels.c](src/kernels/kernels.c)
- Attention loop: [src/engine/engine.c](src/engine/engine.c)
- Build: [Makefile](Makefile) add `-pthread`

---

## 8) Feature: Long-context behavior (sliding window KV)

### Goal

Avoid hard failure at `max_seq_len` during long chats.

### Implementation

- Add a window size W.
- In attention, attend only to the last W tokens.
- When position overflows:
  - simplest: reset KV + re-prefill last W tokens

### Where to integrate

- KV handling: [src/memory/kv_cache.c](src/memory/kv_cache.c)
- Attention masking/seq_len logic: [src/engine/engine.c](src/engine/engine.c)

---

## 9) Feature: Profiling + golden-data dumps

### Goal

Make correctness and performance measurable and reportable.

### Implementation

- Add per-layer timers (wall clock) and aggregate printout.
- Add flags to dump logits / selected intermediate activations.
- Compare against [golden_data/](golden_data/) references.

### Where to integrate

- [src/engine/engine.c](src/engine/engine.c)
- [src/engine/generate.c](src/engine/generate.c)

---

## 10) Suggested milestone order

1. Real top-k/top-p + `--seed`
2. Chat REPL with KV reuse
3. Load-time dequant (Q4/Q8)
4. Profiling + dumps
5. CUDA prefill acceleration (then decode)
