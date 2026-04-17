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
- Kernels: `src/kernels/kernels.c` (CPU AVX2).
- Tokenizer/sampling/CLI: `src/tokenizer/tokenizer.c`

**Important baseline constraint:**
- Linear layers currently use `gemm_f32()` (implemented in `src/kernels/kernels.c`) with weights stored row-major as `(N, K)` and computed as:
  - $C(M,N) = A(M,K) \times B(N,K)^T$ (i.e., dot product against *rows* of the weight matrix).
  - Status: `src/include/kernels.h` docs have been aligned to match this contract.
- Attention is *not* GEMM-based yet: it uses explicit dot-product loops for scores and context.

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

Keep existing CPU kernels where they are (today: `src/kernels/kernels.c`) and add new files around them. Do **not** move or rename existing kernel files as part of this feature set.

Proposed minimal additive layout:

```
src/backend/backend.c          // backend_create/destroy, selects CPU/CUDA
src/backend/cpu_backend.c      // vtable points at functions implemented in src/kernels/kernels.c
src/backend/cuda_backend.c     // vtable points at CUDA implementations

src/kernels/kernels.c          // CPU kernels (existing) + new CPU quant matvecs + optional threadpool
src/kernels/kernels_cuda.cu    // CUDA kernels (new)
src/kernels/kernels_cuda.h     // C-callable wrappers (new)
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
- `forward(..., const KernelVTable* k)`
- `generate(..., const KernelVTable* k, BackendKind kind)`

Rationale:
- `generate()` contains prefill and decode loops and must pick CPU decode even when Phase A uses GPU prefill.

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

### 5.1 What “Phase B” means here
- Keep GGUF weights quantized in mmap’d file.
- Implement decode hot path matvec for Q4_K and Q8_0.
- Avoid allocating full FP32 copies of weight matrices.

### 5.2 Loader changes (required)
- Extend dtype mapping for GGML_TYPE_Q4_K and GGML_TYPE_Q8_0.
- Track exact tensor byte size so quant kernels can validate bounds.

Hard requirement: any quant tensor must carry an authoritative `byte_size` (from GGUF metadata / row-size calculation). A “dtype size” helper cannot be used for quant types.

### 5.3 Engine dispatch changes (required)
Introduce a single helper used by all linear ops:

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
- Block size: 32 logical elements.
- Block bytes: 34 (as already encoded in loader): `2 bytes scale (f16) + 32 bytes q`.
- Dequant: `f = scale * q[i]`.

### 5.5 Q4_K format (implementation notes)

This section is intentionally concrete because Q4_K is the critical MUST path for the target model.

**Block geometry**
- Super-block size: 256 weights.
- Storage bytes per super-block: 144 bytes.
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
- `--temperature 0` greedy output matches llama.cpp’s greedy output for the same GGUF for at least the first 8 tokens on 2 prompts.
- No NaNs.

**Note:** Exact logit match is not required between different implementations; token IDs must match.

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

### 7.1 CLI
- `--chat` enters REPL.

### 7.2 State kept alive
- Model, Tokenizer, Arena, Scratch, KVCache.
- `history_tokens[]` and `current_pos`.

### 7.3 Prompt formatting
- Keep minimal: user input is appended as-is.
- Avoid introducing new chat templates unless required.

### 7.4 Long context (optional)
- When near max context, reset KV and reprefill last W tokens.

---

## 8) Lightweight verification (MUST)

### 8.1 Unit tests (required)
- [x] Add `tests/test_quant.c` (currently a stub-level linker/contract test; dequant + matvec correctness tests still pending). *(DONE 2026-04-18)*
- [x] Add `tests/test_e2e_smoke.c` (currently a stub; will become a TinyLlama “next-token” oracle test once a reference GGUF is available in CI/dev env). *(DONE 2026-04-18)*

- [x] Add MUST scaffolding coverage tests so `make test` exercises new APIs early:
  - `tests/test_backend.c` (backend create + CPU vtable calls)
  - `tests/test_threadpool.c` (threadpool API correctness; currently serial)
  - `tests/test_chat_stub.c` (chat entrypoint returns non-zero until implemented)
  - `tests/test_gpu_prefill_stub.c` (GPU prefill hook returns non-zero until implemented)
  - Status: wired into `make test`. *(DONE 2026-04-18)*

### 8.2 Oracles
- For FP32/TinyLlama: existing `golden_data/gen_golden.py` as optional deeper validation.
- For quantized Llama-3.1: llama.cpp greedy token sequence as oracle.

### 8.3 “Reliably complete” criteria
MUST features are complete when:
- CPU-only build passes all tests.
- Quantized model runs end-to-end on CPU.
- Chat REPL preserves KV.
- `--threads` changes performance measurably on GEMM-heavy prompts.
- GPU prefill runs on a small model and is selectable via `--device`.

Make these verifiable with concrete commands (update as CLI evolves):
- Tests: `make clean && make test`
- CPU quant run (Llama-3.1 8B Q4_K): `./build/llmrt --model /path/to/model.gguf --prompt "Hello" --max-tokens 8 --temperature 0`
- Thread scaling check (CPU): run the same command with `--threads 1` and `--threads 4` and confirm wall-time decreases for prompt lengths where prefill dominates.
- GPU prefill smoke (small float model): `make USE_CUDA=1` then run with `--device cuda` and a prompt long enough to trigger prefill work.

### 8.4 Remaining MUST validation work (explicit TODO)

Backend selection:
- Add runtime tests that exercise `--device auto|cpu|cuda` end-to-end once backend wiring lands (including “cuda requested but unavailable” behavior).

Quant Phase B correctness:
- Replace `tests/test_quant.c` stub with real tests:
  - Q8_0 block dequant golden test
  - Q4_K `scales/min` unpack helper golden test
  - Q4_K full 144-byte super-block dequant golden test
  - matvec sanity tests (small synthetic matrices) + bounds checks using `Tensor.byte_size`

E2E smoke:
- Replace `tests/test_e2e_smoke.c` stub with a TinyLlama “greedy next token id” assertion test (dev-provided GGUF path or small checked-in fixture if feasible).

Chat:
- Add a lightweight non-interactive test that calls the chat loop logic with scripted inputs and asserts `current_pos` monotonicity + KV reuse (once chat REPL is implemented).

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
