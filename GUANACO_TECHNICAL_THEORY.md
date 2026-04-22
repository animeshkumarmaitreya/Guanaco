# Guanaco — Theory of Operations & Engineering Manual

This document provides a comprehensive technical breakdown of the Guanaco LLM inference engine. It covers everything from hardware-level SIMD and CUDA optimizations to high-level session management and heterogeneous scheduling.

---

## 1. Architectural Philosophy

Guanaco is designed as a **dependency-minimal, high-performance C runtime** for Transformer-based LLMs. Its core design principles are:

- **Zero-Allocation Hot Path**: No `malloc` or `free` calls occur during inference. All memory is pre-allocated or managed via high-performance bump-pointers.
- **Backend Heterogeneity**: Seamlessly dispatching computation between CPU (SIMD) and GPU (CUDA) based on hardware availability and model configuration.
- **Quantization First**: Native support for various block-wise quantization formats (Q4_0, Q8_0, Q4_K_M, Q6_K) to minimize memory bandwidth bottlenecks.
- **Stateless Kernels**: Computational kernels are pure functions, making the engine highly testable and deterministic.

---

## 2. Memory Subsystem: The 3-Tier Strategy

Guanaco eliminates memory fragmentation and overhead by using three distinct allocation layers:

| Tier | Lifetime | Backing | Usage |
|---|---|---|---|
| **Persistent** | Global (Engine) | `mmap` / `malloc` | Model weights, static metadata, vocabulary. |
| **Arena** | Session | Large pre-allocated block | KV cache, token history, session-specific state. |
| **Scratch** | Forward Pass | Reusable pool | Intermediate activations, attention scores, logits. |

### The Scratch Buffer Recycle
After each `forward()` pass, the `scratch_reset()` function is called. This simply resets the bump-pointer to the start of the scratch pool. This ensures that memory is "freed" in $O(1)$ time without any platform-level syscalls.

---

## 3. Tensor Foundations

Guanaco uses a standard row-major tensor representation.

```c
typedef struct {
    void* data;          // Data pointer (host or device)
    float* d_data;       // Device-specific pointer
    dtype_t dtype;       // F32, Q4_0, Q8_0, etc.
    int ndim;            // Number of dimensions
    int shape[4];        // Dimension sizes
    int stride[4];       // Strides for indexing
    size_t byte_size;    // Total size in bytes
    int device_residency; // 0 = CPU, 1 = GPU
} Tensor;
```

### The GGUF Format
Guanaco natively parses GGUF files. This involves:
1. Reading the **Header** (Magic number, version, tensor count, metadata count).
2. Parsing **Metadata** entries (Model architecture, head counts, dimensions).
3. Indexing the **Tensor Info** (Name, offset, type, shape).
4. `mmap`-ing the weight data for demand-paged loading.

---

## 4. Quantization Theory

LLM inference is typically **memory-bandwidth bound**. Quantization reduces the number of bits per weight, effectively increasing the "computational intensity" of the hardware.

### Block-wise Quantization
Instead of quantizing the entire matrix with one scale, Guanaco uses block-wise quantization (e.g., 32 elements per block for Q8_0).

- **Q8_0**: 8-bit integers with a single FP16 scale per 32 elements.
  $$w = q \times \text{scale}$$
- **Q4_K**: Uses super-blocks (256 elements) with multiple scales and mins to minimize perplexity loss while maintaining 4-bit weights.
- **Q6_K**: 6-bit weights for a balance between speed and precision.

### Dequantization kernels
On CPUs, Guanaco uses AVX2/AVX-512 to dequantize blocks into SIMD registers on-the-fly during matrix multiplication.

### Dequantization Mathematics

#### Q8_0 (8-bit linear)
Each block of 32 elements has one 16-bit float scale ($s$).
$$w_i = q_i \cdot s$$
Where $q_i \in [-128, 127]$.

#### Q4_0 (4-bit linear)
Each block of 32 elements has one 16-bit float scale ($s$). Each byte stores two 4-bit values.
$$w_i = (q_i - 8) \cdot s$$
Where $q_i \in [0, 15]$.

#### Q4_K_M (K-Quants)
Uses a specialized "super-block" of 256 elements, divided into 8 sub-blocks of 32.
- 1 super-scale (6 bits)
- 1 super-min (6 bits)
- Per-sub-block scales and mins (6 bits each)
This structure significantly reduces the residual error compared to linear quantization.

---

## 5. Transformer Internal Phases

Each forward pass consists of two distinct regimes:

### A. Prefill (Prompt Processing)
- **Input**: Many tokens ($T > 1$).
- **Compute**: Large Matrix-Matrix multiplications (GEMM).
- **Bound**: Often **compute-bound** on modern CPUs/GPUs.
- **KV Cache**: Large chunks of Key and Value vectors are appended to the cache at once.

### B. Decoding (Token Generation)
- **Input**: Exactly one token ($T = 1$).
- **Compute**: Matrix-Vector multiplications (GEMV).
- **Bound**: Strictly **memory-bandwidth bound**.
- **KV Cache**: Reads the *entire* past history ($pos$) for every new token generated.

---

## 6. Computational Kernels

### CPU Backend (SIMD)
Guanaco leverages Intel AVX2 and AMD FMA instructions.
- **Tiled GEMM**: Matrix multiplications are broken into small tiles (e.g., 8x32) that fit into L1/L2 cache.
- **Multithreading**: Uses `pthreads` to distribute rows of the weight matrix across cores.

### SIMD Intrinsics (AVX2)
In the CPU GEMM kernel, we process 8 FP32 elements at a time using `__m256` registers. For quantized weights, the inner loop looks like:
1. Load 32 quantized bytes.
2. Unpack into two `__m256` registers.
3. Multiply by the scale using `_mm256_mul_ps`.
4. Perform Fused Multiply-Add (FMA) using `_mm256_fmadd_ps`.

### GPU Backend (CUDA)
Guanaco offloads entire layers to NVIDIA GPUs.
- **Warp-Level Reduction**: Uses `__shfl_down_sync` for fast horizontal sums within a warp (32 threads).
- **Shared Memory**: Softmax and attention kernels use `__shared__` memory to cache scores and eliminate redundant VRAM reads.
- **Kernel Fusion**: Operations like SiLU + Elementwise Mul are fused into a single kernel:
```cuda
__global__ void silu_fused_kernel(float* x, const float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = x[i];
        x[i] = (val / (1.0f + expf(-val))) * y[i];
    }
}
```

---

## 7. Sampling & Output Generation

The **Sampler** is responsible for picking the next token from the logit distribution.

### Strategies:
1. **Temperature Scaling**: $P_i = \exp(L_i / T) / \sum \exp(L_j / T)$. Low $T$ makes output deterministic.
2. **Top-K**: Only considers the $K$ most likely tokens.
3. **Top-P (Nucleus)**: Considers the smallest set of tokens whose cumulative probability exceeds $P$.
4. **Repetition Penalty**: Penalizes tokens that have appeared recently in the context to prevent "looping".

---

## 8. Heterogeneous Inference & Scheduling

Guanaco can split a model between CPU and GPU. Usually, the first $N$ layers are put on the GPU (VRAM permitting), and the rest on the CPU.

### Dispatch Logic
The `transformer_layer` function acts as a router:
```c
if (weights->wq->device_residency == 1 && T == 1) {
    cuda_transformer_layer_gpu(...);
} else {
    // Standard CPU path
}
```

### Thermal Mangement (NVML)
On Laptops (like RTX 3050), Guanaco uses NVML to monitor GPU temperature. If the temperature exceeds a threshold (e.g., 85°C), the engine can dynamically insert delays or throttle the frequency to prevent hardware damage.

### Load Balancing Algorithm
At startup, Guanaco benchmarks the CPU and GPU. It then decides the `n_gpu_layers` based on:
1. **Available VRAM**: Each layer needs $\sim H^2 \times 7 \times \text{bytes\_per\_param}$.
2. **Transfer Overhead**: If $T=1$, the time to transfer the hidden state to/from GPU must be less than the compute savings.

---

## 9. Session & KV Cache Management

### Head-Major Layout
To maximize L1/L2 cache locality, the KV cache is stored in a head-major layout:
`[layer][head][token][dim]`

This ensures that the dot product for a single head reads a contiguous block of memory.

### Persistence & Serialization
Guanaco supports **Zero-Overhead Serialization**. The entire Arena (containing the KV cache and session history) can be dumped to disk and `mmap`-ed back. This allows:
1. **Persistent Sessions**: Chat history is preserved across restarts.
2. **Infinite Context (Ring Buffers)**: When `max_seq_len` is reached, the engine can evict the oldest tokens (using a sliding window) to allow continuous generation.
3. **KV Cache Sharing**: Multiple sessions can share a common prefix (e.g., a system message) to save VRAM.

---

## 10. Performance Modeling: Roofline Analysis

- **Memory Bandwidth (BW)**: The peak speed data can be moved from RAM to CPU/GPU.
- **Arithmetic Intensity**: The ratio of FLOPs to Bytes ($FLOP/Byte$).

In the **Decode phase**, the intensity is very low (~1.0). Therefore:
$$\text{Tokens/Sec} \approx \frac{\text{Memory Bandwidth}}{\text{Model Size in Bytes}}$$

For a 7B Q4 model (~4GB) on a system with 50GB/s bandwidth:
$$\text{Tokens/Sec} \approx 50 / 4 = 12.5 \text{ tok/s}$$

---

## 10. Summary Checklist for Making an Engine from Scratch

1. **Model Loader**: Parse GGUF/Safetensors.
2. **Memory Manager**: Implement Arena/Scratch allocators.
3. **Foundation Kernels**: Matmul, Softmax, Norm, SiLU.
4. **Transformer Logic**: Multi-head attention and MLP.
5. **Positional Encoding**: RoPE.
6. **Sampler**: Top-K, Top-P, Temperature.
7. **Optimization**: SIMD (CPU) or CUDA (GPU).
8. **Heterogeneous Dispatch**: Load balance between backends.
