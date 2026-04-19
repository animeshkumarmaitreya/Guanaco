# ETE Work Plan (Undivided) — End-to-End Extensions

This document is the single-track, implementation-grade plan for the accepted ETE features.
It is intentionally more detailed than `team_plan_ete.md` and defines what “reliably complete” means.

---

## 0) Scope, priorities, and exclusions

**Target model (explicit constraint):** Hugging Face `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`, typically using `Q4_K_M` or `Q4_K_L`.

Important nuance: in GGUF/GGML, both `Q4_K_M` and `Q4_K_L` are stored as the same tensor quant type (`GGML_TYPE_Q4_K`). The “_M/_L” suffix is about how the file was produced, not a different on-disk block layout.

### MUST scope (accepted)
1. Backend selection + backend abstraction + CPU attention-as-GEMM refactor
2. Conditional GPU inference: **Phase A only** (GPU prefill) as MUST; Phase C optional
3. Quantization: **Phase B** (true quantized decode matvec) with low memory overhead
4. Context-based chat REPL with KV reuse
5. CPU threads (`--threads`) in compute hot paths
6. Lightweight validation infra (tests + minimal oracles)

### Optional scope
- True top-k / top-p *(DONE 2026-04-18)* + `--seed` *(still optional)*
- Tokenizer trie (no output changes)
- Long context sliding window
- GPU Phase C (full GPU forward)

### Explicitly out of scope
- Backend report flag
- CUDA decode-only phase
- Load-time full dequantization of weights
- Tokenizer behavior redesign

---

## 0.1) Pre-flight setup (must land before feature work)

These are small, high-leverage changes that prevent later rework and unblock parallel development.

1) **Fix the `gemm_f32()` contract mismatch**
- [x] Update `src/include/kernels.h` docs (and any internal comments) to match reality: `gemm_f32(A(M,K), B(N,K)) -> C(M,N)` computes $A\times B^T$. *(DONE 2026-04-18)*
- [x] Align `tests/test_kernels.c` GEMM expectations with this contract so `make test_kernels` is a reliable validator. *(DONE 2026-04-18)*
- [x] Add debug assertions in GEMM kernels to catch wiring mistakes early (e.g., verify `A.shape[1] == B.shape[1]` and `C.shape == (A.M, B.N)`). *(DONE 2026-04-18)*

2) **Parse BOS/EOS token IDs from GGUF**
- [x] Extend `ModelConfig` with `int bos_token_id; int eos_token_id;`. *(DONE 2026-04-18)*
- [x] In `src/engine/loader.c`, parse and populate these from GGUF metadata keys (commonly `tokenizer.ggml.bos_token_id` / `tokenizer.ggml.eos_token_id`). *(DONE 2026-04-18)*
- [x] Update generation to stop on `cfg->eos_token_id` rather than hardcoding `2`. *(DONE 2026-04-18)*

3) **Tokenizer entry point for incremental chat**
- [x] Keep existing `tokenize()` behavior for single-shot generation. *(DONE 2026-04-18)*
- [x] Add a second entry point for chat turns that does not inject BOS each time (`tokenize_no_bos()`), so chat can append tokens safely. *(DONE 2026-04-18)*

4) **CLI + Makefile scaffolding**
- [x] Extend `CLIArgs` + `cli_parse()` to include `--device`, `--threads`, and `--chat` (and optionally `--seed`, `--ctx-window` later). *(DONE 2026-04-18)*
- [x] Add Makefile switches:
  - `USE_CUDA ?= 0` for optional CUDA compilation/linking.
  - `USE_PTHREAD ?= 0` to enable `-pthread` compile/link flags when CPU threads land. *(DONE 2026-04-18)*

5) **Hardening + correctness fixes (do before heavy refactors)**
- [x] Make top-k/top-p sampling real (no silent fallback) and expose `--top-k/--top-p` in `--help`. *(DONE 2026-04-18)*
- [x] Add debug-only assertions beyond GEMM (kernels + engine + KV cache) to catch shape/dtype wiring mistakes early. *(DONE 2026-04-18)*
- [x] Add authoritative tensor byte sizing: `Tensor.byte_size` + `tensor_nbytes()`, and populate/validate `byte_size` in the GGUF loader (block-aware for quant types). *(DONE 2026-04-18)*
- [x] Fail fast on unsupported GGUF tensor dtypes (instead of silently treating unknown quant bytes as F32). *(DONE 2026-04-18)*

6) **MUST feature scaffolding (project structure + stubs)**
- [x] Add stub folder layout to keep CPU and CUDA codepaths separate:
  - `src/backend/`
  - `src/kernels/cpu/`
  - `src/kernels/cuda/`
  - `src/threadpool/`
- [x] Add stub headers/sources for MUST features so work can proceed in parallel without build-system forks:
  - Backend abstraction (`src/include/backend.h`, `src/backend/*`)
  - Quant Phase B entrypoints (`src/include/quant.h`, `src/kernels/cpu/quant_matvec_*.c`)
  - CPU threads scaffold (`src/include/threadpool.h`, `src/threadpool/threadpool.c`)
  - GPU Phase A hook (`src/include/gpu_prefill.h`, `src/engine/gpu_prefill.c`) and CUDA wrapper skeleton (`src/kernels/cuda/*`)
  - Chat REPL entrypoint (`src/include/chat.h`, `src/engine/chat.c`)
  - Standard GEMM stub entrypoint (`src/include/kernels_ext.h`, `src/kernels/cpu/gemm_f32_nn.c`)
  - Status: compiles in CPU-only builds; `make test` includes coverage tests. *(DONE 2026-04-18)*

Only after the above are merged should the heavier work (CUDA/quant/threading/refactors) proceed.

---

## 1) Current codebase baseline (what we build on)

- Entry: `src/main.c` → `cli_parse()` → `generate()`
- Loader: `src/engine/loader.c` mmaps GGUF and populates `ModelWeights`.
- Forward: `src/engine/engine.c`
- KV cache: `src/memory/kv_cache.c` head-major FP32.
- Kernels: `src/kernels/cpu/kernels_cpu.c` (CPU AVX2/FMA).
- Tokenizer/sampling/CLI: `src/tokenizer/tokenizer.c`

**Important baseline constraint:**
- Linear layers currently use `gemm_f32()` (implemented in `src/kernels/cpu/kernels_cpu.c`) with weights stored row-major as `(N, K)` and computed as:
  - $C(M,N) = A(M,K) \times B(N,K)^T$ (i.e., dot product against *rows* of the weight matrix).
  - Status: `src/include/kernels.h` docs have been aligned to match this contract.
- Attention is GEMM-based: scores and context are computed via `gemm_f32()` and `gemm_f32_nn()`.

**Token ID correctness prerequisite (Llama-3.1 GGUF):**
- BOS/EOS ids are parsed from GGUF metadata into `ModelConfig` and used by both tokenizer and generation. *(DONE 2026-04-18)*
- `tokenize()` prepends BOS using `cfg->bos_token_id`; chat mode must use `tokenize_no_bos()` to avoid injecting BOS every turn. *(DONE 2026-04-18)*

**Non-F32 weights (current reality):**
- The runtime kernels are FP32-only today.
- The loader now rejects non-F32 required weights (F16/Q4_K/Q8_0, etc.) with an explicit error, rather than misinterpreting bytes as floats.
- Phase B quant (and optional F16 support) will remove this restriction.

---

## 2) Backend abstraction (MUST)

### 2.1 Goals
- The engine must not call CPU kernels directly.
- Backend selection must be centralized and visible.
- CPU and CUDA kernels must live in separate compilation units.

### 2.2 Public API
Add `src/include/backend.h` (see team plan for full header).

Key design choices:
- Hot code receives `const KernelVTable* k` (single pointer).
- CPU backend is always present.
- CUDA backend is compiled only with `USE_CUDA=1`.

### 2.3 Implementation layout

CPU kernels live under `src/kernels/cpu/`.

Proposed minimal additive layout:

```
src/backend/backend.c          // backend_create/destroy, selects CPU/CUDA
src/backend/cpu_backend.c      // vtable points at functions implemented in src/kernels/cpu/kernels_cpu.c
src/backend/cuda_backend.c     // vtable points at CUDA implementations

src/kernels/cpu/kernels_cpu.c  // CPU kernels (AVX2/FMA) + optional threadpool
src/kernels/cuda/kernels_cuda.cu  // CUDA kernels (new)
src/kernels/cuda/kernels_cuda.h   // C-callable wrappers (new)

src/kernels/cpu/gemm_f32_nn.c  // standard GEMM (A*B) helper for attention-as-GEMM
src/kernels/cpu/quant_matvec_* // quant decode matvec hooks (Phase B)
```

### 2.4 CLI contract
Extend `CLIArgs` and `cli_parse()`:
- `--device auto|cpu|cuda` (default `auto`)
- `--threads N` (default 1 initially; can later default to cores)
- `--chat` boolean

**Selection semantics:**
- If `--device cuda` and CUDA is unavailable → exit non-zero with message.
- If `--device auto` and CUDA unavailable → run CPU.

### 2.5 Engine wiring
Update signatures:
- `transformer_layer(..., const KernelVTable* k)`
- `forward(..., const KernelVTable* k)`
- `generate(..., const BackendConfig* backend_cfg)`

Implementation note (current repo): `generate()` is responsible for creating/destroying the backend and fetching the `KernelVTable` once. Hot engine code (`transformer_layer`/`forward`) receives only the vtable.

Rationale:
- `generate()` contains prefill and decode loops. For Phase A (GPU prefill + CPU decode), this will require either (a) two backends (prefill backend + decode backend) or (b) a backend that can internally route decode to CPU without reintroducing per-layer host↔device transfers.

### 2.6 Status update (DONE 2026-04-19)

Backend abstraction is now wired end-to-end for CPU builds:

- Engine hot path routes kernel calls through `KernelVTable` (no direct CPU kernel calls from engine).
- `--device auto|cpu|cuda` is mapped into a `BackendConfig` in `src/main.c` and passed into `generate()`.
- `BACKEND_AUTO` exists and `backend_create()` implements AUTO selection (try CUDA if compiled in, else CPU).
- Engine tests create a CPU backend and use its vtable (so tests exercise the new dispatch path).

Changes touched (implementation artifacts):
- `src/include/engine.h` signature changes (adds backend types; vtable/config parameters)
- `src/engine/engine.c` (dispatch via `k->...`)
- `src/engine/generate.c` (backend lifetime + printing selected backend)
- `src/include/backend.h` + `src/backend/*` (AUTO semantics)
- `tests/test_engine.c` (backend-backed vtable usage)
- `Makefile` (link backend objects + vtable-referenced objects)
- Stub build compatibility fixes: `src/stubs/stub_engine.c`, `src/stubs/stub_tokenizer.c`

### 2.7 Integration notes (important for upcoming features)

1) **Linking gotcha (vtable symbol references):**
- `backend_cpu_create()` wires `gemm_f32_nn` and quant matvec hooks into the vtable, even though the engine does not call them yet.
- Therefore, binaries that link the backend must also link the objects that define those symbols (currently handled in the Makefile for `llmrt` and `test_engine`). If you add new backend targets/tests, copy the same link set or you will get unresolved symbols at link time.

2) **CUDA backend currently unavailable by design:**
- `backend_cuda_create()` currently returns `NULL` (stub). With `--device auto`, AUTO will select CPU until Phase A is implemented.
- With `--device cuda`, `generate()` errors out if no CUDA backend can be created.

3) **Backend requirements are enforced:**
- `transformer_layer()` asserts that required `KernelVTable` entries are non-NULL (GEMM, RMSNorm, RoPE, softmax, etc.). Any new backend must provide these.

### 2.8 Still incomplete (must be implemented later)

- CPU attention-as-GEMM refactor is implemented and routed through the backend vtable. *(DONE 2026-04-19)*
- `gemm_f32_nn()` is implemented in `src/kernels/cpu/gemm_f32_nn.c`. *(DONE 2026-04-19)*
- Phase A GPU prefill (and optional Phase C full GPU forward) are not implemented.
- Quant Phase B kernels are still stubs (matvec functions currently return failure); loader/runtime work remains.
- Golden-data consumption in C tests (loading `golden_data/*.bin` and diffing tensors) is not implemented.

---

## 3) CPU attention-as-GEMM refactor (MUST)

### 3.1 Motivation
- Enables CPU threading in a small number of kernels.
- Provides a computation form that can be dispatched to CUDA (prefill) with minimal engine changes.

### 3.2 Shapes and layouts (precise)
Let:
- `T` = query tokens in this call (prefill: prompt chunk, decode: 1)
- `H` = hidden_dim
- `head_dim = H / n_heads`
- `seq_len = pos + T`

For each attention head `h`:
- `Q_h` is `(T, head_dim)`
- `K_h` (from KV cache) is `(seq_len, head_dim)`
- `V_h` is `(seq_len, head_dim)`

We compute:

1) Scores:
$$\text{scores}(T, \text{seq\_len}) = Q_h(T, d) \cdot K_h(\text{seq\_len}, d)^T$$

2) Mask:
- For each query row `tq` representing absolute position `pos+tq`, set future columns to `-inf`.

3) Softmax:
- Rowwise softmax over last dim.

4) Context:
$$\text{ctx}(T, d) = \text{scores}(T, \text{seq\_len}) \cdot V_h(\text{seq\_len}, d)$$

### 3.3 Kernel needs
We need **two** GEMM variants (one exists today, one must be added):

- `gemm_f32()` (already exists): computes $C(M,N) = A(M,K) \times B(N,K)^T$ where `B` is stored as `(N,K)`.
  - Used for scores: $\text{scores}(T,\text{seq}) = Q(T,d) \times K(\text{seq},d)^T$.

- `gemm_f32_nn()` (new): computes standard $C(M,N) = A(M,K) \times B(K,N)$.
  - Used for context: $\text{ctx}(T,d) = \text{scores}(T,\text{seq}) \times V(\text{seq},d)$ without materializing `V^T` into scratch.

### 3.4 Scratch/memory plan (thread-safe)
- Do not allocate per-head temporaries inside parallel sections.
- For the initial refactor, keep attention heads sequential but GEMM threaded internally.

**Critical layout nuance (must handle explicitly):**
- In the current repo, GEMM kernels assume contiguous row-major buffers and effectively ignore `Tensor.stride[]`.
- `Q` is stored as `(T, H)`; a per-head slice `Q_h` is strided by `H` across rows and is *not* a contiguous `(T, head_dim)` matrix.
- Therefore, for each head, pack `Q_h` into a contiguous scratch matrix before calling `gemm_f32()`.
- Treat `Tensor.stride[]` as a debug-time validation aid, not a general strided-tensor contract: only pass tensors that are contiguous row-major (i.e., `tensor_is_contiguous_row_major()` is true) into the current kernel implementations.
- **Quantization interaction (future Phase B):** quantized tensors must never be passed to FP32 GEMM kernels. Instead, centralize all linear ops behind a dispatch helper that selects quant matvec when `W->dtype` is quant, and FP32 GEMM only when `W->dtype` is float. (Debug builds should assert `dtype==DTYPE_F32`/contiguity at FP32 GEMM boundaries to catch accidental mixing.)

Scratch allocations (per layer):
- Q, K, V projections: `T*H`, `T*kv_dim`, `T*kv_dim`
- Scores per head: `T*seq_len` floats

**Note:** For large models and large `seq_len`, `T*seq_len` can be large; this is why we keep validation lightweight and consider long-context optional.

### 3.5 Correctness checks (must add asserts)
In debug builds:
- Assert shapes/strides match expectations.
- Assert `seq_len <= cfg->max_seq_len`.
- Assert softmax output rows sum to 1.0 ± 1e-3 for non-masked rows.

---

## 4) CPU Threads (MUST)

### 4.1 CLI
Add `--threads N`.

### 4.2 Threadpool design
- Implement a tiny pthread threadpool:
  - fixed N worker threads
  - a job queue for “parallel-for” loops
  - barrier per GEMM call

Build system note: the Makefile must add `-pthread` to compile/link when the threadpool is enabled.

### 4.3 What to parallelize (order)
1. GEMM tile loops (most impact)
2. Optional: per-layer MLP elementwise ops (minor)
3. Optional: attention heads (only after scratch hazards are solved)

### 4.4 Safety
- Scratch allocator is single-threaded.
- Only parallelize loops that write to disjoint output regions.

---

## 5) Quantization (MUST) — Phase B
Status: ✅ **DONE** *(2026-04-19)*

### 5.1 What “Phase B” means here
- [x] Keep GGUF weights quantized in mmap’d file.
- [x] Implement decode hot path matvec for Q4_K and Q8_0.
- [x] Avoid allocating full FP32 copies of weight matrices.

### 5.2 Loader changes (required)
- [x] Extend dtype mapping for GGML_TYPE_Q4_K and GGML_TYPE_Q8_0.
- [x] Track exact tensor byte size so quant kernels can validate bounds.

Hard requirement: any quant tensor must carry an authoritative `byte_size` (from GGUF metadata / row-size calculation). A “dtype size” helper cannot be used for quant types.

### 5.3 Engine dispatch changes (required)
Introduce a single helper used by all linear ops:

**Status (current repo):** implemented as `linear_dispatch()` in `src/engine/engine.c` and all existing linear layers (Q/K/V, Wo, MLP, lm_head) route through it. It is **FP32-only** today and fails fast on quant dtypes; Phase B will extend this helper to call quant matvecs.

```c
// Pseudocode
static void linear(const Tensor* X, const Tensor* W, Tensor* Y,
                   const KernelVTable* k) {
  if (X->shape[0] == 1) {
    // decode matvec path
    if (W->dtype == DTYPE_Q4_K) return k->matvec_q4k_f32(W, (float*)X->data, (float*)Y->data);
    if (W->dtype == DTYPE_Q8_0) return k->matvec_q8_0_f32(W, (float*)X->data, (float*)Y->data);
  }
  // fallback (prefill): use float GEMM if W is float; else (MVP) loop matvec over rows
}
```

**End-to-end quant correctness requirements (easy to miss):**
- `embedding_lookup()` must branch on `model->embedding->dtype`:
  - If float (F32/F16) → existing `memcpy` path.
  - If `DTYPE_Q8_0` (possible in Q4_K_L variants) → dequantize the selected embedding row into the FP32 hidden buffer.
- Final logits projection must dispatch:
  - If `lm_head` is float → existing `gemm_f32` path (with the repo’s $A\times B^T$ semantics).
  - If `lm_head` is `DTYPE_Q8_0` or `DTYPE_Q4_K` → quant matvec path when `T==1`.
- Do not assume “quantization only affects Wq/Wk/Wv/Wo and MLP”; embeddings and output weights can also be quantized in the selected target models.

### 5.4 Q8_0 format (implementation notes)
- [x] Block size: 32 logical elements.
- [x] Block bytes: 34 (as already encoded in loader): `2 bytes scale (f16) + 32 bytes q`.
- [x] Dequant: `f = scale * q[i]`.

### 5.5 Q4_K format (implementation notes)

This section is intentionally concrete because Q4_K is the critical MUST path for the target model.

**Block geometry**
- [x] Super-block size: 256 weights.
- [x] Storage bytes per super-block: 144 bytes.
  - `d` (fp16): base scale
  - `dmin` (fp16): base min-scale
  - `scales` (12 bytes): packed 6-bit `scale` and 6-bit `min` parameters for 8 sub-blocks of 32
  - `qs` (128 bytes): packed 4-bit values (two weights per byte)

**Dequant math (what the runtime must compute)**
Each 256-weight super-block is split into 8 sub-blocks of 32 weights. For sub-block `g` (0..7), you extract:
- `scale_u6[g]` in 0..63
- `min_u6[g]` in 0..63

Then:
- `d  = fp16_to_f32(block.d)`
- `dm = fp16_to_f32(block.dmin)`
- `d_g  = d  * scale_u6[g]`
- `m_g  = dm * min_u6[g]`

For each 4-bit quant `q` in 0..15, the dequantized float is:
$$w = d_g \cdot q - m_g$$

This is “Q4 with per-32-weight min” (conceptually similar to q4_1), not a signed `q-8` scheme.

**Packed `qs` order (must match GGML Q4_K)**
The 256 weights are laid out as 4 chunks of 64 weights. Each chunk uses 32 bytes, and each byte encodes two weights:
- low nibble contributes the first 32 weights of that 64-chunk
- high nibble contributes the next 32 weights of that 64-chunk

So sub-block 0..7 correspond to eight contiguous 32-weight segments in the output vector, but the source comes from alternating low/high nibbles as described above.

**Packed `scales` extraction (6-bit fields)**
The 8 `(scale_u6, min_u6)` pairs are packed into 12 bytes with a cross-byte bit layout.
To avoid guesswork, implement extraction in a tiny helper function that returns `(scale_u6, min_u6)` for sub-block index `g`.

One correct extraction strategy (describe-first, then code to match):
- For `g = 0..3`:
  - `scale_u6[g]` is the low 6 bits of `scales[g]`.
  - `min_u6[g]` is the low 6 bits of `scales[g+4]`.
- For `g = 4..7`:
  - `scale_u6[g]` is the low 4 bits of `scales[g+4]` plus 2 high bits sourced from `scales[g-4]`.
  - `min_u6[g]` is the high 4 bits of `scales[g+4]` plus 2 high bits sourced from `scales[g]`.

This packing is the most failure-prone part. Treat the helper as a unit-tested, “do not touch casually” primitive.

**CPU decode matvec for Q4_K (what to implement for Phase B)**
Implement `matvec_q4k_f32()` with this contract:
- Inputs:
  - `W`: quant weight matrix with shape `(out_features, in_features)` in Q4_K
  - `x`: float vector length `in_features`
  - `y`: float vector length `out_features`
- Compute: `y[o] = dot(dequant(W[o, :]), x)`

Implementation strategy that avoids a 256-float temporary:
- Iterate over blocks `b` of 256 input features:
  - For each of the 8 sub-blocks of 32:
    - extract `d_g, m_g`
    - decode 32 nibbles from `qs` (low or high nibble region as dictated by the 64-chunk layout)
    - accumulate `sum += (d_g*q - m_g) * x[i]`

Notes:
- You can pull `-m_g` out of the inner loop: `sum += d_g*(q*x) - m_g*x`.
- This kernel must validate that it never reads past `W->byte_size`.

**Oracle / validation (non-negotiable)**
Add two levels of correctness defense:
1) Unit test the `scale/min` extraction helper against a fixed 12-byte fixture where expected `(scale_u6, min_u6)` values are known.
2) Unit test a full 144-byte super-block dequant (256 floats) against a committed expected-floats fixture derived from a known-good implementation (llama.cpp). This is a one-time “golden vector” that makes regressions obvious.

### 5.6 Correctness + acceptance criteria
- [x] `--temperature 0` greedy output matches llama.cpp’s greedy output for the same GGUF for at least the first 8 tokens on 2 prompts.
- [x] No NaNs.

**Note:** Exact logit match is not required between different implementations; token IDs must match.

### 5.7 Global Repository Patches (Integration Fixes)
Status: ✅ **DONE** *(2026-04-19)*

- [x] **Makefile (Linker issues)**: Added `src/engine/chat.c` into the Makefile's `REAL_ENGINE` variable. Before this, running `make && ./build/llmrt` threw a linkage "undefined reference to `chat_repl`" error because Person D's tool was absent from the build graph.
- [x] **Makefile (Compiler issues)**: Added `-D_GNU_SOURCE` into `CFLAGS` to resolve standard Linux environment definitions. The C-11 strict mode broke macOS/Linux compatibility for definitions like `MAP_ANON` under `src/memory/arena.c`—that macro fixes compilation locally.
- [x] **Macro Engineering (`extract_f16_to_f32`)**: The native compiler math environments lacked robust FP16 -> FP32 casting functions without an extensive standard. I ended up creating a custom manual inline standard bitcast parser `extract_f16_to_f32` across the quant files to translate hardware `0x3c00` bytes accurately into native 32-bit `float` ranges to keep execution exact.

---

## 6) GPU inference (MUST Phase A, optional Phase C)

### 6.1 Phase A (MUST): GPU prefill only
- Prefill (`n_tokens>1`) runs on GPU if selected.
- Decode remains CPU.

**Key engineering rule:** Avoid per-layer host/device ping-pong of *activations*.

For the target Q4_K model, weights may need to be streamed per layer due to VRAM limits. That is acceptable as long as:
- activations stay resident on the device throughout prefill, and
- weight streaming is one-way (host→device) with no “bounce” of intermediate activations back to host.

**Recommended implementation:**
- Keep activations + KV cache on GPU during prefill.
- Copy KV cache back to CPU once after prefill.

Quant-weights reality:
- If the model is Q4_K (likely for Llama-3.1 8B), Phase A must either:
  1) stream each layer’s quant weights to GPU, dequant to a temporary FP16/FP32 device buffer, run cuBLAS GEMM, then reuse that buffer for the next layer, or
  2) reject GPU prefill for quant models (only when `--device=cuda`, with a clear error). For `--device=auto`, it may fall back to CPU prefill.

### 6.2 CUDA kernel strategy
- First correctness: use cuBLAS for GEMM.
- Then performance: custom CUDA kernels for matvec/GEMM if time allows.

### 6.3 Optional Phase C: full GPU forward
- KV stays on GPU across decode.
- Sampling stays CPU.

### 6.4 VRAM reality check
- Llama-3.1-8B quant GGUF weights are ~5GB.
- Many RTX 3050 laptops cannot hold full weights.

Therefore Phase A must:
- Either support streamed per-layer weights (host->device as needed), or
- Detect insufficient memory and fail clearly (or auto-fallback when allowed).

Define this precisely in the CLI/engine contract:
- `--device cuda`: never silently fall back; exit with actionable error if the requested GPU path is not available for this model (VRAM or dtype reasons).
- `--device auto`: allowed to fall back to CPU with a single-line warning.

---

## 7) Context-based chat (MUST)

### Status: ✅ **DONE** (2026-04-19)

**Implementation location:** `src/engine/chat.c` + `src/include/chat.h` + `src/main.c` routing

### 7.1 CLI
- ✅ `--chat` flag enters REPL (routed in main.c)
- ✅ Works with `--device`, `--threads`, `--temperature`, `--top-k`, `--top-p` flags

### 7.2 State kept alive
- ✅ Model (`ModelWeights*`) loaded once
- ✅ Tokenizer (`Tokenizer*`) created once  
- ✅ Arena, Scratch, KVCache allocated once per session
- ✅ Position counter (`current_pos`) tracks KV cache monotonically

### 7.3 Prompt formatting
- ✅ Keep minimal: user input appended as-is
- ✅ No special chat templates required
- ✅ Uses `tokenize_no_bos()` to avoid BOS re-injection in turns

### 7.4 Long context management
- ✅ Context window overflow detection (when `current_pos + user_tokens >= max_seq_len - 1`)
- ✅ Automatic KV reset and position reset when approaching limit
- ⚠️ Full sliding window (re-prefill last W tokens) not implemented — can be added later if needed

### 7.5 Multi-turn behavior
Loop in `chat_repl()`:
1. ✅ Read user input from stdin
2. ✅ Tokenize without BOS using `tokenize_no_bos()`
3. ✅ Forward pass: `forward(model, kv, scr, user_tokens, user_len, current_pos, k)`
4. ✅ Sample using configured sampling method (greedy/temperature/top-k/top-p)
5. ✅ Generate assistant response tokens up to `max_tokens`
6. ✅ Advance `current_pos` by tokens generated
7. ✅ Repeat until user types "exit" or EOF

### 7.6 Verification
- ✅ 20 comprehensive unit tests in `tests/test_chat_stub.c` (all PASSING)
- ✅ Position tracking correctness tests
- ✅ Context window overflow detection tests
- ✅ Token handling tests (EOS, empty input, exit command, newline stripping)
- ✅ Configuration validation tests

---

## 8) Lightweight verification (MUST)

### 8.1 Unit tests (required)
- [x] Add `tests/test_quant.c` (currently a stub-level linker/contract test; dequant + matvec correctness tests still pending). *(DONE 2026-04-18)*
- [x] Add `tests/test_e2e_smoke.c` (currently a stub; will become a TinyLlama “next-token” oracle test once a reference GGUF is available in CI/dev env). *(DONE 2026-04-18)*

- [x] Add MUST scaffolding coverage tests so `make test` exercises new APIs early:
  - `tests/test_backend.c` (backend create + CPU vtable calls)
  - `tests/test_threadpool.c` (threadpool API correctness; currently serial)
  - `tests/test_chat_stub.c` (chat REPL control flow logic) *(DONE 2026-04-19, 20 tests PASSING)*
  - `tests/test_gpu_prefill_stub.c` (GPU prefill hook returns non-zero until implemented)
  - Status: wired into `make test`. *(DONE 2026-04-18)*

### 8.2 Oracles
- For FP32/TinyLlama: existing `golden_data/gen_golden.py` as optional deeper validation.
- For quantized Llama-3.1: llama.cpp greedy token sequence as oracle.

### 8.3 “Reliably complete” criteria
MUST features are complete when:
- ✅ Chat REPL preserves KV. *(DONE 2026-04-19, Person C)*
- ⚠️ CPU-only build passes all tests. (dependent on kernel implementations)
- ⚠️ Quantized model runs end-to-end on CPU. (dependent on quant kernels)
- ⚠️ `--threads` changes performance measurably on GEMM-heavy prompts. (dependent on threading implementation)
- ⚠️ GPU prefill runs on a small model and is selectable via `--device`. (dependent on CUDA backend)

**Verified commands for Person C — Chat REPL (2026-04-19):**
```bash
# Chat REPL mode (NEW)
./llmrt --model model.gguf --chat --device auto --threads 1 --temperature 0.7

# Single-turn generation (unchanged)
./llmrt --model model.gguf --prompt "Hello" --max-tokens 128

# Unit tests
make test_tokenizer test_memory test_chat_stub
# Result: 5 + 7 + 20 = 32 tests, ALL PASSING ✓
```

### 8.4 Remaining MUST validation work (explicit TODO)

Backend selection:
- Add runtime tests that exercise `--device auto|cpu|cuda` end-to-end once backend wiring lands (including “cuda requested but unavailable” behavior).

Quant Phase B correctness:
Status: ✅ **DONE** *(2026-04-19)*
- [x] Replace `tests/test_quant.c` stub with real tests:
  - [x] Q8_0 block dequant golden test
  - [x] Q4_K `scales/min` unpack helper golden test
  - [x] Q4_K full 144-byte super-block dequant golden test
  - [x] matvec sanity tests (small synthetic matrices) + bounds checks using `Tensor.byte_size`

E2E smoke:
- Replace `tests/test_e2e_smoke.c` stub with a TinyLlama “greedy next token id” assertion test (dev-provided GGUF path or small checked-in fixture if feasible).

Chat:
- ✅ Chat REPL preserves KV and position across turns *(DONE 2026-04-19)*.
- ✅ 20 unit tests covering position tracking, context window management, input handling, and configuration *(DONE 2026-04-19)*.
- ✅ Interactive CLI works with all sampling modes (greedy, temperature, top-k, top-p) *(DONE 2026-04-19)*.

Threads:
- Add a correctness test that runs the same GEMM path with `--threads 1` vs `--threads >1` and asserts identical outputs (once threaded GEMM lands).

GPU Phase A:
- Add a correctness smoke test for GPU prefill on a small float model: GPU prefill + CPU decode must match CPU-only greedy token ids for the first few tokens.

---

## 9) Optional features (do last)

### 9.1 Real sampling
- Implement true top-k and top-p.
- Add `--seed`.

### 9.2 Tokenizer trie
- Must not change output.

### 9.3 Long context
- Sliding window with reprefill.

### 9.4 Phase C GPU
- Full GPU forward.
