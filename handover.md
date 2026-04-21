# 🏁 MEGA HANDOVER: Heterogeneous LLM Acceleration (RTX 3050)

This document is a comprehensive guide for the next developer. It consolidates the implementation plan, bottleneck analysis, and current task status into a single master reference.

## � Executive Summary: Progress & Gains
| Stage | Decode Speed | TTFT (Prefill) | Status |
|---|---|---|---|
| **Baseline** | 0.5 tok/s | 15.7s | — |
| **Phase 0 (Offload Fix)** | 0.6 tok/s | 15.7s | ✅ Completed |
| **Phase 3 (AVX2 + Prefetch)** | 0.7 tok/s | 6.9s | ✅ Completed |
| **Phase 1 (Fused GPU Layer)** | **1.3 tok/s** | **3.1s** | ✅ Completed (v1) |

---

## 🏛️ System Architecture

### 1. The Heterogeneous Split
The system dynamically splits the 32-layer Llama model. 
- **Auto-Default**: 26 layers on GPU, 6 on CPU.
- **VRAM Budget**: ~3.3GB for weights, ~0.7GB for KV cache/activations.
- **Gating Logic**: In [loader.c](file:///home/animesh/Desktop/llmframework/src/engine/loader.c), only `Q4_K` weights are offloaded. Non-quantized or higher-precision weights stay on CPU to prevent kernel crashes.

### 2. The Fused GPU Pipeline ([cuda_layer_kernels.cu](file:///home/animesh/Desktop/llmframework/src/kernels/cuda/cuda_layer_kernels.cu))
- **Innovation**: Eliminates 14 PCIe round-trips per layer. Activations stay in VRAM.
- **Workflow**: 
  1. H→D hidden state upload (once).
  2. GPU: RMSNorm → QKV Proj → RoPE.
  3. D→H Q/K/V download for CPU Attention (Numerical Safety).
  4. H→D Attn-out upload.
  5. GPU: Out-Proj → Residual → RMSNorm → MLP (Gate/Up/SiLU/Down) → Residual.
  6. D→H Final hidden state download (once).

### 3. CPU Vectorization ([kernels_cpu.c](file:///home/animesh/Desktop/llmframework/src/kernels/cpu/kernels_cpu.c))
- All fallback kernels (`rmsnorm`, `silu`, `residual_add`, `elemwise_mul`) use 8-wide AVX2 intrinsics.
- Prefetching added in `quant_matvec_q4_k.c` to L1-cache next blocks.

---

## �️ War Stories: Solved Critical Bugs

### 🌊 The mmap DMA Segfault
- **Symptom**: Instant crash on weight upload.
- **Lesson**: CUDA DMA engines cannot trigger page faults on lazy `MAP_PRIVATE` mmap regions.
- **Solution**: Implemented a **heap staging buffer**. Always `malloc` -> `memcpy(mmap)` -> `cudaMemcpy(malloc)` -> `free()`.

### 👻 The Ghost Layer Segfault
- **Symptom**: CPU kernels crashing while accessing weights.
- **Cause**: Loader blindly marked VRAM weights as "device-resident" even for dtypes (Q6_K) without CUDA kernels. CPU tried to read 0x7f... pointers and died.
- **Solution**: Explicit check: `if (dtype == DTYPE_Q4_K) residency = 1`.

---

## 📅 Future Roadmap (Phases 1.5 - 6)

### Phase 1.5: Fused GPU Attention (High Priority)
- The current path falls back to CPU attention for numerical correctness.
- **Goal**: Fix the `cuda_rope_kernel` and `softmax` logic in `cuda_layer_kernels.cu` to keep attention in VRAM. This will push us toward **1.8+ tok/s**.

### Phase 2: Shared Memory Matvec
- Update the Q4_K kernel to load scale/min blocks into `__shared__` memory.
- Current kernel is warp-reduced but metadata reads are uncoalesced.

### Phase 4: Async Double Buffering
- Overlap GPU execution of Layer N with CPU processing of the remaining layers OR the next token's embedding preparation.

### Phase 5: GQA Optimization
- Implement a dedicated GQA kernel for the 4:1 Llama 3.1 ratio to maximize occupancy.

---

## ⚠️ Developer Checklist
- [ ] Always build with `make USE_CUDA=1 USE_PTHREAD=1`.
- [ ] Verify GPU kernels via `test_quant` parity test.
- [ ] Check `metrics.log` if performance drops (Thermal throttling @ 87°C).
- [ ] Keep tensors 32-byte aligned for AVX2 and CUDA memory access rules.
