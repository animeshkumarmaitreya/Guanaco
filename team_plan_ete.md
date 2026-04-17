# Team of 4 — End-to-End Extensions (ETE) Plan

This plan implements the **accepted** features from FEATURE_EXTENSIONS.md and **explicitly excludes** rejected items.
It is written to be directly actionable in this repo’s current structure (C11, Makefile, `src/include/*` APIs).

## Decisions (as provided)

### Accepted (MUST implement)
1. **Backend Selection (CPU/CUDA)**
   - Add CLI flag: `--device auto|cpu|cuda` (default: `auto`).
   - **No** `--backend-report`.
   - Introduce a backend abstraction (vtable) so the engine calls kernels via a selected backend.
   - Keep **CPU kernels** and **GPU kernels** in separate codepaths/files.
   - Refactor attention to use matrix ops (GEMM-style), not explicit dot-product loops, enabling CPU threading.

2. **Conditional GPU inference**
   - Implement only:
     - **Phase A (MUST)**: GPU prefill (prompt processing) when `--device cuda` (or `auto` selects CUDA).
     - **Phase C (OPTIONAL)**: keep the complete forward pass on GPU.
   - CUDA approach: **custom kernels first**; if infeasible for performance/correctness, **fallback to CUDA toolkit libraries** (cuBLAS) for GEMM.

3. **Quantization (MUST)**
   - Implement **Phase B**: true quantized decode matvec (on-the-fly dequant) with low memory overhead.
   - Target GGUF models:
     - `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (default MUST)
     - `Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf` (optional alternative; uses `Q8_0` for embedding + output weights)

4. **Context-based chat (MUST)**

7. **CPU threads (MUST)**

9. **Verification / validation infra (MUST, lightweight)**

### Optional (implement only if time allows)
- **Real sampling**: true top-k / top-p *(DONE 2026-04-18)* + `--seed` reproducibility *(still optional)*.
- **Tokenizer change**: optional trie for vocab lookup (no other tokenizer behavior changes).
- **Long context behavior**: sliding window / KV reset + reprefill.

### Explicitly Rejected (not in plan)
- `--backend-report`
- GPU Phase B (CUDA decode-only offload)
- Quantization Phase A (load-time full dequantization)
- Any tokenizer redesign beyond optional trie

---

## Team Split (4 nearly equal parts)

Each person owns a coherent subsystem and ships tests from Day 1. Person D also owns integration.

| Person | Dedicated Component | Integration? | Why this split is balanced |
|---|---|---|---|
| **A** | CUDA backend (Phase A) + optional Phase C | No | GPU work is large, self-contained, testable vs CPU |
| **B** | Quantization runtime (Q4_K / Q8_0) + loader dtype plumbing | No | Deep but bounded: loader + quant matvec + tests |
| **C** | Chat REPL + CLI UX plumbing + optional long-context + optional real sampling | No | Product-facing features + correctness controls |
| **D (Integrator)** | Backend abstraction + CPU attention-as-GEMM refactor + CPU threading + validation infra | Yes | Cross-cutting engine refactors + tests + final wiring |

---

## Day 0 — Interfaces (2–4 hours, non-negotiable)

### 0.0) Repo reality check (prevents rework)
- In the current codebase, `gemm_f32()` (implemented in `src/kernels/kernels.c`) computes:
  - $C(M,N) = A(M,K) \times B(N,K)^T$ where `B` is stored row-major as `(N, K)` (this matches how GGUF weights are loaded: rows = out_features).
  - Status: `src/include/kernels.h` docs have been aligned to this contract.
- Attention is still written with explicit dot-product loops in `src/engine/engine.c` (scores + context). The “attention-as-GEMM” refactor will:
  - use `gemm_f32()` for the score step ($Q\times K^T$), and
  - add **one** additional GEMM variant for the context step ($\text{scores}\times V$): standard $A\times B$.
- Token IDs are now parsed from GGUF metadata: `bos_token_id`/`eos_token_id` live in `ModelConfig`, tokenizer uses `bos_token_id`, and generation terminates on `eos_token_id`. Chat turns must use `tokenize_no_bos()`.

### 0.0b) Pre-flight commits (must merge before parallel work)
- **Kernel contract fix (DONE 2026-04-18):** updated `src/include/kernels.h` to correctly document `gemm_f32` as $A\times B^T$ with `B(N,K)` layout; updated kernel GEMM tests so `make test_kernels` validates the real contract; added debug asserts to `gemm_f32()` to enforce shapes/strides/dtypes in debug builds.
- **Token IDs in config (DONE 2026-04-18):** add `bos_token_id` / `eos_token_id` to `ModelConfig`, populate from GGUF metadata, and use `cfg->eos_token_id` for EOS termination in generation.
- **Chat-safe tokenization (DONE 2026-04-18):** added `tokenize_no_bos()` so chat turns don’t repeatedly inject BOS.
- **Sampler correctness (DONE 2026-04-18):** `sample_top_k()` / `sample_top_p()` are real implementations (no silent fallback), and `--help` documents `--top-k/--top-p`.
- **Tensor byte sizing + dtype safety (DONE 2026-04-18):** `Tensor.byte_size`/`tensor_nbytes()` added; loader computes block-aware `byte_size` and fails fast on unsupported dtypes (prevents treating quant bytes as F32). Runtime remains FP32-only until Phase B quant.
- **Extra debug asserts (DONE 2026-04-18):** added shape/dtype asserts in engine embedding lookup, KV cache append, and non-GEMM kernels.
- **CLI plumbing (DONE 2026-04-18):** extend `CLIArgs` with `--device`, `--threads`, `--chat`; keep existing flags unchanged.
- **Build switches (DONE 2026-04-18):** add `USE_CUDA ?= 0` and Makefile pthread wiring (via `USE_PTHREAD ?= 0`) so CUDA/threads work doesn’t fork the build system.
- **MUST scaffolding + tests (DONE 2026-04-18):** add MUST-only stub folder layout (`src/backend/`, `src/kernels/cpu/`, `src/kernels/cuda/`, `src/threadpool/`) plus stub headers/sources for backend/quant/threadpool/chat/gpu-prefill/CUDA wrappers; wire new coverage tests into `make test` so the APIs stay buildable as real implementations land.

### 0.1) Agree on new/updated headers
We will add one new header and minimally extend existing ones.

#### New: `src/include/backend.h`
**Purpose:** Select backend (CPU/CUDA), expose a kernel vtable and backend-owned resources.

```c
// backend.h
#pragma once
#include "types.h"

typedef enum {
    BACKEND_CPU = 0,
    BACKEND_CUDA = 1,
} BackendKind;

typedef struct Backend Backend;

typedef struct {
    // Core math
  // Existing repo GEMM semantics (matches weight layout):
  // C(M,N) = A(M,K) * B(N,K)^T   (B stored row-major as (N,K))
    void (*gemm_f32)(const Tensor* A, const Tensor* B, Tensor* C);
  // NEW: standard GEMM for cases where B is stored as (K,N)
  // C(M,N) = A(M,K) * B(K,N)
  void (*gemm_f32_nn)(const Tensor* A, const Tensor* B, Tensor* C);

    void (*rmsnorm)(const Tensor* input, const Tensor* weight, Tensor* output, float eps);
    void (*softmax_inplace)(Tensor* scores, int seq_len);
    void (*silu_inplace)(Tensor* x);
    void (*rope)(Tensor* q, Tensor* k, int pos, int head_dim);
    void (*residual_add)(Tensor* x, const Tensor* residual);
    void (*elemwise_mul)(Tensor* a, const Tensor* b);

    // Quantized decode hot path (Phase B)
    // y(N) = W(N,K) * x(K)
    int (*matvec_q4k_f32)(const Tensor* W_q4k, const float* x, float* y);
    int (*matvec_q8_0_f32)(const Tensor* W_q8,  const float* x, float* y);
} KernelVTable;

typedef struct {
    BackendKind kind;
    int threads;      // CPU threads or CUDA host-side threading
    int device_id;    // CUDA device id; ignored for CPU
} BackendConfig;

Backend* backend_create(const BackendConfig* cfg);
void     backend_destroy(Backend* b);

const KernelVTable* backend_kernels(const Backend* b);
BackendKind         backend_kind(const Backend* b);
```

**Why we add `gemm_f32_nn`:** attention context computation needs $\text{ctx}(T,d) = \text{scores}(T,\text{seq}) \times V(\text{seq},d)$ where V is naturally stored row-major as `(seq_len, head_dim)` and must be treated as `(K,N)` for standard GEMM. The existing `gemm_f32()` computes $A\times B^T$ and cannot express `scores × V` without materializing `V^T` into scratch every head.

#### Update: `src/include/tokenizer.h`
Add CLI flag fields:
- `--device auto|cpu|cuda` → `args->device`
- `--threads N` → `args->threads`
- `--chat` → `args->chat`

Optional later:
- `--seed` (sampling)
- `--ctx-window` (sliding window)

**Must-add nuance for Llama-3.1 correctness:**
- Add `bos_token_id` and `eos_token_id` into `ModelConfig` (in `src/include/types.h`) and populate them in `src/engine/loader.c` from GGUF metadata keys (commonly `tokenizer.ggml.bos_token_id` / `tokenizer.ggml.eos_token_id`).
- Chat mode must not inject BOS every turn. Keep the existing `tokenize()` behavior for non-chat generation, but add a second entry point (e.g., `tokenize_no_bos()` or `tokenize_into(buf, add_bos)`) for incremental chat turns.

#### Update: `src/include/engine.h`
Engine APIs will accept a backend context (non-optional for maintainability):

```c
// engine.h (new signatures)
ModelWeights* load_model(const char* path, Arena* arena);
void free_model(ModelWeights* model);

Tensor* forward(ModelWeights* model, KVCache* kv, Scratch* scr,
                int* token_ids, int n_tokens, int pos,
                const KernelVTable* k);

void generate(const char* model_path, const char* prompt, int max_tokens,
              float temperature, int top_k, float top_p,
              const KernelVTable* k,
              BackendKind backend_kind);

void chat_repl(const char* model_path, int max_tokens, float temperature,
               int top_k, float top_p, const KernelVTable* k,
               BackendKind backend_kind);
```

> Important: we pass `KernelVTable*` rather than `Backend*` into hot functions to keep call overhead minimal (one pointer deref).

#### Update: `src/include/types.h` for quantization
Extend `DataType` to represent K-quant types we will actually see:
- `DTYPE_Q4_K`
- `DTYPE_Q8_0`
…and add fields needed for quant kernels:

```c
// types.h additions (minimal)
// - keep Tensor as-is but add ggml_type + byte_size for quant tensors

typedef struct {
    void*    data;
    int      shape[MAX_DIMS];
    int      stride[MAX_DIMS];
    int      ndim;
    DataType dtype;

    // NEW (needed for quant tensors, where "stride in elements" is ambiguous)
    uint32_t ggml_type;   // raw GGML type from GGUF tensor info
    size_t   byte_size;   // exact bytes mapped for this tensor
} Tensor;
```

**Notes (important for correctness):**
- For quantized tensors, `tensor_numel()` is the logical element count, but compute must treat `data` as a sequence of blocks. `byte_size` is the authoritative bound.
- Do not rely on `dtype_size()` for quantized tensors (it is approximate today). Either update it or add a dedicated helper like `tensor_nbytes(const Tensor*)`.

---

## Person A — CUDA Backend (Phase A MUST, Phase C optional)

### Scope
- Add CUDA backend implementation behind `Backend` abstraction.
- Keep CUDA code **separate** from CPU kernels.
- Implement GPU prefill path (Phase A) with CPU decode fallback.
- Optional: full GPU forward (Phase C).

### Files owned

Proposed files (minimize disruptive moves; keep existing CPU kernels in place):
```
src/include/backend.h
src/backend/backend.c                 (backend_create/destroy, selects cpu/cuda)
src/backend/cpu_backend.c             (vtable = existing CPU kernels)
src/backend/cuda_backend.c            (vtable = CUDA wrappers)

src/kernels/cuda/kernels_cuda.cu      (custom CUDA kernels)
src/kernels/cuda/cuda_gemm.cu         (GEMM/matvec kernels)
src/kernels/cuda/cuda_norm.cu         (rmsnorm/silu/elementwise)
src/kernels/cuda/cuda_softmax.cu      (softmax)

src/engine/gpu_prefill.c              (optional helper: prefill on GPU)
```

**Explicit non-goal:** do not move/rename existing CPU kernel file(s) unless strictly necessary. “Separate codepaths/files” is satisfied by adding CUDA kernels in distinct `.cu` files and keeping CPU kernels in the current C sources.

### Task A1: Build system — `USE_CUDA=1`
**Goal:** Build either CPU-only or CPU+CUDA with identical public headers.

- Makefile changes:
  - Add `NVCC`, `CUDA_CFLAGS`, `CUDA_LDFLAGS`.
  - Compile `.cu` with `nvcc`.
  - Link with `-lcudart` (and optionally `-lcublas` for fallback).
  - Guard CUDA compilation behind `USE_CUDA ?= 0`.

**Pass criteria:**
- `make` works on CPU-only machine.
- `make USE_CUDA=1` produces a binary that links and prints selected backend.

### Task A2: CUDA backend selection semantics
Implement:
- `--device cpu` → force CPU.
- `--device cuda` → require CUDA build + GPU present, otherwise error.
- `--device auto` → choose CUDA only if built with CUDA and `cudaGetDeviceCount()>0`.

**Warnings:**
- Do not silently fall back from `--device cuda` to CPU; error loudly.
- For `auto`, fallback is allowed and must be explicit in stdout once.

### Task A3 (MUST): Phase A — GPU Prefill
**Definition:** Only the prompt processing (`n_tokens > 1`) runs on GPU.
Decode (`n_tokens == 1`) remains CPU.

**Design constraints (must satisfy):**
- KV cache ultimately must exist in CPU layout for decode.
- To avoid per-layer PCIe copies, prefill should keep activations and KV on GPU during prefill.
- After prefill completes, copy KV cache (for `seq_len = prompt_len`) from GPU → CPU once.

**Concrete plan:**
1. Add a GPU KV cache structure mirroring CPU KV layout:
   - `K[layer][kv_head][pos][head_dim]` contiguous.
2. During GPU prefill:
   - Upload token embeddings (or embedding matrix) and the prompt token IDs.
   - Run layer loop entirely on GPU.
   - Append K/V into GPU KV cache.
3. After prefill:
   - Download the populated KV region for `seq_len=prompt_len` into CPU KV cache.

**Implementation approach (custom-first, with a hard fallback):**
- First attempt: implement custom CUDA kernels for the minimal operator set needed by prefill (GEMM/matvec + a few elementwise ops). Keep kernels in separate `.cu` compilation units.
- Pivot condition: if custom GEMM is not correct+stable by the agreed checkpoint, switch GEMM/matvec to cuBLAS while retaining custom kernels for the non-GEMM pieces.

**Pass criteria:**
- With a small model (TinyLlama), `--device cuda` produces the same greedy next token as CPU for 3 fixed prompts.
- Prefill time (TTFT) improves for prompts of length ≥ 256 vs CPU on the same machine.

**Warnings:**
- VRAM constraints: Llama-3.1-8B GGUF weights are ~5GB; many RTX 3050 laptops cannot fit full weights in VRAM. Phase A must therefore support **streamed weights** (host→device per layer) or fallback to cuBLAS with pageable memory. If prefill cannot fit, `--device auto` must revert to CPU; `--device cuda` must error with a clear message.

**Clarification (quantized weights + GPU prefill):**
- The accepted quantization approach is “Phase B” (CPU decode matvec with on-the-fly dequant). That does **not** automatically make GPU prefill work for Q4_K models.
- For Phase A correctness, implement and validate GPU prefill first on a small FP16/FP32 model (e.g., TinyLlama). For Llama-3.1-8B Q4_K_{M|L}, Phase A must either:
  - Stream and dequantize per-layer weights into a temporary FP16/FP32 device buffer for GEMM (no full-model dequant in RAM/VRAM), or
  - Detect “not supported/insufficient memory” and fail clearly (or auto-fallback when allowed).

### Task A4 (OPTIONAL): Phase C — Full GPU forward
**Definition:** Keep hidden activations and KV cache on GPU across both prefill and decode.

Required changes:
- GPU KV cache persists across decode iterations.
- CPU decode loop becomes GPU decode loop.
- Sampling + detokenize stays on CPU (small).

**Pass criteria:**
- Generation of 32 greedy tokens matches CPU token IDs for TinyLlama.

---

## Person B — Quantization (Phase B MUST)

### Scope
- Enable end-to-end inference with Q4_K_M / Q4_K_L models with **low RAM overhead**.
- Implement quantized **decode matvec** (M=1) with on-the-fly dequantization.
- Add minimal loader/type plumbing so tensors expose their true dtype.

### Files owned

```
src/engine/loader.c                    (dtype mapping, tensor byte_size, ggml_type)
src/include/types.h                    (DTYPE_Q4_K, Tensor fields)
src/include/backend.h                  (matvec_q4k_f32, matvec_q8_0_f32)

src/kernels/cpu/quant_matvec_q4k.c     (Q4_K matvec)
src/kernels/cpu/quant_matvec_q8_0.c    (Q8_0 matvec)

tests/test_quant.c                     (new)
```

### Task B1: Loader — correct dtype mapping
Extend `ggml_to_dtype()` so:
- GGML_TYPE_Q4_K → DTYPE_Q4_K
- GGML_TYPE_Q8_0 → DTYPE_Q8_0
- GGML_TYPE_F16/F32 remain.

Also set in each Tensor:
- `t->ggml_type = ti->ggml_type`
- `t->byte_size = computed_from_dims_and_type`

**Warning:** For quantized tensors, “num elements” is *logical* element count; the byte size is block-based:
$$\text{bytes} = \frac{\text{numel}}{\text{block\_size}} \cdot \text{bytes\_per\_block}$$

### Task B2: CPU quantized matvec — Q8_0 and Q4_K

#### Q8_0 matvec
- Block size: 32.
- Each block stores scale (f16) + 32 int8 values.
- For each output row:
  - Iterate blocks, dequant to float, accumulate dot.

#### Q4_K matvec
- Block size: 256.
- Q4_K is more complex (multiple scales and packed 4-bit values).

**Authoritative layout (re-derived, keep in sync with upstream GGML):**
- Q4_K stores 256 logical values per block (a “super-block”).
- Each block contains:
  - two FP16 scalars: `d` and `dmin` (super-block scale factors)
  - `12` bytes of packed 6-bit values encoding 8 per-subblock scales and 8 per-subblock mins (16 total u6 values)
  - `128` bytes of packed 4-bit quant values (2 values per byte)

This is enough to implement the dequant path, but the exact u6 packing/unpacking order must be validated with an oracle.

Implementation strategy:
1. Define a local C struct matching GGML’s q4_k block layout (do **not** copy code; re-derive layout from documentation and validate).
2. Implement a `dequant_q4k_block(float out[256], const void* block)`.
3. In matvec:
   - For each block: dequant into a small stack/temporary buffer, dot with `x`.

**Validation strategy (must be in tests):**
- Compare dequantized blocks against llama.cpp (oracle) for 100 random blocks extracted from the real GGUF file.
  - This can be a developer-only tool (`tools/dump_qblock.c`) plus a python script to diff.

**Pass criteria:**
- Quant matvec matches oracle within max abs error < 1e-3 for Q4_K and < 1e-5 for Q8_0.

### Task B3: Engine integration for quant decode
The engine must route linear layers through a helper that chooses:
- If `T==1` and weight is Q4_K → `matvec_q4k_f32`
- If `T==1` and weight is Q8_0 → `matvec_q8_0_f32`
- Else (prefill) either:
  - (MVP) run a loop of matvec across T rows (slower but correct), OR
  - (stretch) implement a quantized GEMM.

Also required for end-to-end correctness on the selected model variants:
- `embedding_lookup()` must handle `DTYPE_Q8_0` embeddings (Q4_K_L) by dequantizing the chosen row into the FP32 hidden buffer.
- Final logits projection (`lm_head`) must dispatch to quant matvec when it is `DTYPE_Q8_0` / `DTYPE_Q4_K` (at least for `T==1`).

**MUST requirement:** correctness over speed; prefill can be slower.

### Task B4: Model support checklist (Llama-3.1-8B)
- Confirm loader maps these required tensor names:
  - `token_embd.weight`, `output.weight`, all `blk.*` tensors.
- Ensure tensors may have mixed types:
  - Q4_K for most linear weights
  - Q8_0 for embedding/lm_head in Q4_K_L variant

**Warning:** memory sizing
- CPU KV cache in FP32 can be extremely large at Llama-3 sizes. If RAM becomes a blocker, we keep KV as FP16 as a follow-up; not in scope unless required.

---

## Person C — Chat + CLI UX + Optional Sampling/Tokenizer/Long-context

### Scope
- Implement a context-preserving chat REPL.
- Add CLI UX that works with backend selection, threads, and chat.
- Optional last: real top-k/top-p + seed, tokenizer trie, long-context sliding window.

### Files owned

```
src/main.c
src/tokenizer/tokenizer.c              (cli_parse + samplers)
src/engine/generate.c                  (add chat mode entry)

tests/test_chat.c                      (new, lightweight)
tests/test_sampling.c                  (new, optional)
```

### Task C1 (MUST): CLI extensions
Add:
- `--device auto|cpu|cuda`
- `--threads N`
- `--chat`

Constraints:
- No new UX beyond flags above.
- Keep existing flags working.

### Task C2 (MUST): Chat REPL with KV reuse
Add `chat_repl()` (declared in engine.h).

Behavior:
- Load model once.
- Create Arena/Scratch/KV once.
- Maintain:
  - `int* history_tokens` buffer
  - `int history_len`
  - `int current_pos`

Loop:
1. Read a user line.
2. Tokenize only new user content (plus a minimal template/prefix if needed).
3. **Prefill only new tokens** by calling `forward(..., new_tokens, new_len, current_pos)`.
4. Decode assistant tokens, appending to history and advancing `current_pos`.

**Pass criteria:**
- Multi-turn run does not reset KV.
- `current_pos` increments monotonically.

### Task C3 (OPTIONAL): Real top-k/top-p + `--seed`
Implement true top-k and top-p (nucleus) with deterministic RNG.

- `--seed N` sets `srand(N)` at startup.
- top-k: partial select k-largest logits (no full sort required).
- top-p: compute softmax probs, sort candidates by prob desc, take prefix with cumulative ≥ p.

### Task C4 (OPTIONAL): Tokenizer trie
Add an optional trie to accelerate vocab lookup in the existing greedy longest-prefix tokenizer.

Constraints:
- Must not change tokenizer output compared to current implementation.
- Only change how it finds the longest prefix.

### Task C5 (OPTIONAL): Long context sliding window
When `current_pos` approaches `max_seq_len`:
- Reset KV cache + re-prefill the last `W` tokens from history.
- `W` default: `max_seq_len/2` (or a fixed number), gated behind `--ctx-window W`.

---

## Person D — Backend Abstraction + CPU Refactor + Threads + Validation (Integration owner)

### Scope
- Introduce backend abstraction end-to-end in CPU build.
- Refactor attention implementation to use GEMM-style kernels.
- Implement CPU threading (`--threads`) in compute hot paths.
- Provide lightweight validation tooling and tests.

### Files owned

```
src/include/backend.h
src/backend/*

src/engine/engine.c                    (attention-as-GEMM refactor, matvec dispatch)
src/engine/generate.c                  (wire backend + threads)

src/kernels/cpu/*                      (threaded GEMM + gemm_f32_nn)

tests/test_e2e_smoke.c                 (new)
```

### Task D1 (MUST): Attention-as-GEMM refactor
Replace the explicit loops in transformer attention with two GEMM calls per head:

1) **Scores**
- Input: `Q_h` shape `(T, head_dim)`
- Input: `K_h` shape `(seq_len, head_dim)`
- Output: `scores` shape `(T, seq_len)`
- Compute using existing GEMM semantics ($A\times B^T$):
  - `scores = gemm_f32(Q_h(T,d), K_h(seq,d))` producing `(T, seq)`
  - scale by `1/sqrt(head_dim)`
  - apply causal mask
  - softmax rowwise

2) **Context**
- Input: `scores` shape `(T, seq_len)`
- Input: `V_h` shape `(seq_len, head_dim)`
- Output: `ctx` shape `(T, head_dim)`
- Requires **standard** GEMM: `ctx = scores(T,seq) × V_h(seq,d)` via `gemm_f32_nn()`

Finally scatter `ctx` into `attn_output`.

**Warnings:**
- Be meticulous with strides. The KV cache tensors use `stride[]` in *elements*.
- Softmax rows must handle all-masked rows (decode masking edge cases).
- **Important:** the current `gemm_f32` / `gemm_f32_nn` implementations assume contiguous row-major buffers and effectively ignore `Tensor.stride[]`. Because `Q` is stored as `(T, H)`, a per-head slice is strided by `H` across rows; pack `Q_h` into a contiguous `(T, head_dim)` scratch matrix before calling GEMM.

### Task D2 (MUST): CPU threads
Add `--threads N`:
- Default: `N = number of CPU cores` (or 1 for deterministic).
- Implement a small pthread threadpool used by GEMM kernels.

Threading targets (in order):
1. `gemm_f32` and `gemm_f32_nn` tile loops
2. Optional: attention heads parallelization (only after scratch safety is addressed)

**Critical warning:** `Scratch` is not thread-safe. Do not allocate per-head temporaries from shared scratch inside parallel regions.

### Task D3 (MUST): Lightweight validation infra
Add:
- A deterministic e2e smoke test that runs:
  - `--temperature 0` greedy
  - fixed prompt
  - asserts next token ID equals expected for TinyLlama

- Keep golden-data comparison optional:
  - Use `golden_data/gen_golden.py` for TinyLlama float reference.
  - Add a C helper to load `*.bin` and compute max-abs-diff for layer 0 tensors.

For quantized Llama-3.1 models:
- Validation oracle is llama.cpp greedy output for the same GGUF.
- We only require that the **first N greedy tokens** match (N=8 or 16), to keep it lightweight.

**Status (DONE 2026-04-18: scaffold-level coverage exists; correctness TODOs remain):**
- The repo now contains stub-level tests wired into `make test` so MUST feature APIs are exercised early:
  - `tests/test_quant.c` (contract/link coverage; real dequant+matvec golden tests pending)
  - `tests/test_e2e_smoke.c` (placeholder; real GGUF-based greedy oracle test pending)
  - `tests/test_backend.c` (backend abstraction coverage)
  - `tests/test_threadpool.c` (threadpool API coverage; currently serial)
  - `tests/test_chat_stub.c` (chat entrypoint stub coverage)
  - `tests/test_gpu_prefill_stub.c` (GPU prefill hook stub coverage)

**Remaining MUST validation suites (non-stub correctness):**
- Quant Phase B: golden/oracle tests for Q8_0 + Q4_K dequant and matvec (block-accurate)
- E2E smoke: TinyLlama greedy “next token id” oracle test from a real GGUF
- GPU Phase A: CUDA prefill correctness parity vs CPU greedy (first few tokens)
- Chat: scripted multi-turn test that asserts KV reuse + monotonic `current_pos`
- Threads: run same path with `--threads 1` vs `--threads >1`, assert identical outputs

### Task D4: Integration
- Ensure CLI selects backend and threads.
- Ensure forward/generate/chat all accept `KernelVTable*`.
- Ensure CPU-only build still runs fully.

---

## Integration Milestones (priority-ordered)

### Milestone 1 — Backend abstraction + CPU correctness (enables everything)
- Backend vtable created.
- CPU backend wired.
- Attention-as-GEMM refactor passes existing tests.

### Milestone 2 — Quantized CPU decode (must for chosen models)
- Loader recognizes Q4_K and Q8_0.
- Matvec kernels work.
- Engine routes decode linear layers through quant matvec.

### Milestone 3 — Chat
- REPL works with KV reuse.

### Milestone 4 — CPU threads
- `--threads` threads GEMM.

### Milestone 5 — GPU prefill (Phase A)
- `--device cuda` prefill works on a small model.

### Optional milestones
- Phase C full GPU forward
- Real top-k/top-p + seed
- Tokenizer trie
- Long context sliding window

---

## Risk Matrix (what to watch out for)

| Risk | Severity | What it looks like | Mitigation |
|---|---:|---|---|
| Stride/layout bug in attention GEMM | High | NaNs or nonsense output after refactor | Add shape/stride asserts; compare against old loop path on small prompts |
| Q4_K block layout misinterpreted | High | Quant outputs wildly off vs oracle | Build a standalone block dequant tester against llama.cpp |
| GPU VRAM insufficient for 8B weights | High | CUDA OOM at model init | Phase A must allow auto fallback; test GPU on TinyLlama first |
| Threading + scratch allocator races | High | Sporadic crashes or ASAN failures | Keep scratch allocations out of parallel regions; thread-local temporaries |
| Tokenizer mismatch for Llama-3 | Medium | Output differs from llama.cpp early | Validate tokenization for a few prompts; do not change behavior beyond allowed scope |

---

## Definition of “Done” (reliably complete)

MUST features are considered complete when:
- `make test` passes.
- CPU path runs `--device cpu --threads 1` deterministically.
- Quantized model runs end-to-end on CPU (greedy generation).
- Chat REPL preserves KV across turns.
- GPU prefill works for a small model and auto-falls back safely.

Optional features are complete only if their tests pass and do not regress MUST features.
