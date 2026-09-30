# Guanaco: High-Performance Local LLM Inference Engine

[![Language: C11](https://img.shields.io/badge/Language-C11%20%7C%20CUDA-green.svg)]()
[![Platform: Linux](https://img.shields.io/badge/Platform-Linux-orange.svg)]()
[![SIMD: AVX2 / FMA](https://img.shields.io/badge/SIMD-AVX2%20%2F%20FMA-purple.svg)]()

**Guanaco** is a dependency-minimal, bare-metal C11 and CUDA inference runtime engineered from scratch for local Large Language Model (LLM) execution. It bypasses heavy frameworks (PyTorch, ONNX, LibTorch) to prioritize system-level efficiency, zero-allocation hot paths, hardware-level vectorization, and memory safety.

Guanaco natively parses GGUF models (e.g., **Meta Llama 3.1 8B**, **TinyLlama**) and features a dynamic heterogeneous engine that splits model layers across GPU VRAM and CPU system RAM, enabling 8B parameter models to run smoothly even on consumer laptops with limited VRAM (e.g., 4GB RTX 3050).

---

## ⚡ Key Highlights & Capabilities

* **Zero-Allocation Hot Path:** Employs a strict 3-tier memory model (Persistent `mmap`, bump-pointer Arena, $\mathcal{O}(1)$ recycled Scratch pool) guaranteeing **0 dynamic heap allocations (`malloc`/`free`) per token generated**.
* **Heterogeneous CPU + GPU Inference:** Overcomes GPU Out-Of-Memory (OOM) barriers by dynamically partitioning transformer layers across NVIDIA VRAM and host RAM (`--n-gpu-layers`).
* **Hand-Crafted SIMD Kernels:** Custom x86 AVX2 and FMA vectorized inner loops calculating 8 single-precision floating-point operations per CPU cycle, with on-the-fly dequantization.
* **Block-Wise Quantization Support:** Native execution for GGUF K-Quants (`Q4_K_M`, `Q4_K_L`), `Q8_0`, and `FP32` tensors directly from disk.
* **Instant Session Persistence (`.lctx`):** Zero-overhead KV cache serialization allows saving and resuming chat conversations instantly without re-processing prompt history.
* **Frozen Prefix Caching:** Eliminates prefill latency across multiple sessions by caching common system prompts in memory.
* **Custom Pthread Threadpool:** Low-latency worker thread pool with tiled Matrix-Multiplication (GEMM) scheduling and automatic CPU core topology discovery.
* **Hardware Telemetry & Thermal Safety:** Real-time NVML GPU monitoring that injects throttling backoffs if hardware thermals exceed safe operational limits (85°C).
* **Interactive CLI & Wizard:** Real-time token streaming with TTFT (Time-To-First-Token) and tokens/sec latency telemetry, plus an interactive setup wizard (`./run.sh`).

---

## 🚀 Quickstart

### Prerequisites
* **Operating System:** Linux (x86_64)
* **Compiler:** GCC 9+ or Clang 11+ with C11 support
* **Build System:** Make
* **Optional (GPU):** NVIDIA CUDA Toolkit (nvcc) and NVIDIA driver for CUDA acceleration

### 1. Build the Engine
Compile the binary using the automated build script:

```bash
# Standard CPU build (AVX2 + Pthreads enabled)
./build.sh

# GPU-accelerated build (requires CUDA toolkit)
./build.sh --cuda

# Clean rebuild
./build.sh --clean
```

Alternatively, use `make` directly:
```bash
make -j$(nproc)                          # CPU build
make USE_CUDA=1 USE_PTHREAD=1 -j$(nproc) # CUDA + Multithreaded build
```

### 2. Run with the Interactive Wizard
The easiest way to start inference is the built-in wizard:
```bash
./run.sh
```
The wizard guides you through selecting a model, configuring CPU threads, offloading GPU layers, and toggling chat mode.

### 3. Run via CLI Directly
```bash
# Single-shot prompt generation
./build/llmrt --model models/llama-3.1-8b-instruct-q4_k_m.gguf --prompt "Explain the concept of memory alignment in C."

# Interactive Chat REPL
./build/llmrt --model models/llama-3.1-8b-instruct-q4_k_m.gguf --chat --threads 8

# Heterogeneous GPU Offloading (20 layers on GPU, remaining on CPU)
./build/llmrt --model models/llama-3.1-8b-instruct-q4_k_m.gguf --device cuda --n-gpu-layers 20 --chat
```

---

## 🛠️ CLI Reference (`llmrt`)

| Flag | Argument | Description | Default |
|:---|:---|:---|:---|
| `--model` | `<path>` | **[Required]** Path to GGUF model file. | - |
| `--prompt` | `<text>` | Input text for single-shot generation. | `"Hello"` |
| `--chat` | - | Launch the multi-turn **Interactive Chat REPL**. | Off |
| `--device` | `auto\|cpu\|cuda`| Computational backend device. | `auto` |
| `--n-gpu-layers` | `<N>` | Number of transformer layers to offload to GPU VRAM. | `0` |
| `--threads` | `<N>` | Number of CPU worker threads (defaults to system cores). | `1` |
| `--ctx` | `<N>` | Context buffer token limit. | `8192` |
| `--max-tokens` | `<N>` | Maximum new tokens to generate per turn. | `128` |
| `--session` | `<path>` | Save or resume session KV-cache state (`.lctx`). | - |
| `--prompt-cache` | `<path>` | Pre-load frozen prompt prefix cache. | - |
| `--temperature` | `<float>` | Softmax temperature scaling (lower = more deterministic).| `0.70` |
| `--top-k` | `<int>` | Top-K sampling cutoff (0 to disable). | `40` |
| `--top-p` | `<float>` | Nucleus (Top-P) probability threshold (1.0 to disable). | `0.90` |

---

## 🧩 Codebase Architecture & Tour

The codebase is organized into cleanly decoupled, testable subsystems:

```
llmframework/
├── build.sh                  # Automated compiler detection & build runner
├── run.sh                    # Interactive CLI runner and execution wizard
├── Makefile                  # Granular compilation flags (AVX2, CUDA, ASAN)
├── scripts/
│   └── system_monitor.py     # Live CPU/GPU VRAM & thermal hardware telemetry
├── src/
│   ├── main.c               # Program entrypoint & CLI argument parsing
│   ├── backend/             # Hardware backend abstraction & virtual dispatch
│   │   ├── backend.c        # Backend factory (`auto`, `cpu`, `cuda`)
│   │   ├── cpu_backend.c    # CPU vtable binding & system thermal hooks
│   │   └── cuda_backend.c   # CUDA layer offload & memory upload wrappers
│   ├── engine/              # Core Transformer inference pipeline
│   │   ├── loader.c         # GGUF parser, metadata extraction & mmap loader
│   │   ├── engine.c         # Forward pass & layer execution (RMSNorm, RoPE, MHA, SwiGLU)
│   │   ├── generate.c       # Prefill and autoregressive decode token loop
│   │   ├── chat.c           # Multi-turn conversational REPL & BOS management
│   │   ├── session.c        # KV-cache serialization & prompt prefix caching
│   │   └── gpu_prefill.c    # CUDA prompt prefill dispatch hooks
│   ├── memory/              # High-performance memory management
│   │   ├── arena.c          # Bump-pointer memory allocator for session state
│   │   ├── scratch.c        # O(1) recycled activation memory workspace
│   │   └── kv_cache.c       # Head-major KV cache ring buffer
│   ├── kernels/             # Computational math kernels
│   │   ├── cpu/
│   │   │   ├── kernels_cpu.c         # SIMD AVX2/FMA GEMM, RMSNorm, Softmax, RoPE
│   │   │   ├── quant_matvec_q4_k.c   # AVX2 on-the-fly Q4_K_M dequantization & GEMV
│   │   │   ├── quant_matvec_q6_k.c   # Q6_K quantized GEMV
│   │   │   └── quant_matvec_q8_0.c   # AVX2 Q8_0 linear dequantization & GEMV
│   │   └── cuda/
│   │       ├── cuda_layer_kernels.cu # Complete GPU transformer layer offload
│   │       ├── matvec_q4k_cuda.cu    # CUDA warp-shuffle Q4_K matrix-vector kernel
│   │       └── kernels_cuda.cu       # CUDA runtime entrypoints
│   ├── threadpool/          # POSIX multi-core parallel scheduler
│   │   └── threadpool.c     # Worker thread queue, mutexes, parallel-for
│   ├── tokenizer/           # Text processing & sampling
│   │   └── tokenizer.c      # BPE tokenizer, vocab lookup, Top-K/Top-P samplers
│   └── include/             # Public type definitions and kernel API headers
└── tests/                   # Comprehensive unit and integration test suite
    ├── test_engine.c        # Transformer forward pass & layer shape tests
    ├── test_quant.c         # Quantization dequant math & bit-bounds tests
    ├── test_backend.c       # Backend virtual dispatch & fallback verification
    ├── test_threadpool.c    # Multithreading concurrency tests
    └── test_chat_stub.c     # Multi-turn context management & token tracking
```

---

## 🔬 Core Engineering Deep Dives

### 1. The 3-Tier Zero-Allocation Memory Model
Standard memory allocators introduce non-deterministic latency jitter, OS context switches, and cache fragmentation during token generation. Guanaco isolates memory into three distinct tiers:
1. **Persistent Tier:** Static weights and vocabulary are mapped directly from disk using POSIX `mmap()`, allowing OS-level page cache eviction without swapping.
2. **Arena Tier:** A contiguous pre-allocated slab managed by a fast bump-pointer for session-scoped data (KV Cache, conversational history).
3. **Scratch Pool Tier:** A reusable activation buffer recycled after every single token in $\mathcal{O}(1)$ time via `scratch_reset()`, guaranteeing **zero heap allocations on the hot path**.

### 2. Heterogeneous CPU/GPU Scheduling (4GB VRAM Optimization)
Quantized 8B parameter models (~4.9GB) physically exceed the 4.0GB VRAM ceiling of common laptop GPUs (like the NVIDIA RTX 3050). Guanaco solves this by:
* Storing a `device_residency` tag on every tensor.
* Uploading the first $N$ layers (e.g., 20 layers ~3.0GB) to VRAM via PCIe during initialization.
* Dynamically dispatching each layer through a `KernelVTable`: GPU-resident layers run CUDA warp-shuffle kernels, while host layers run multithreaded AVX2 SIMD kernels.
* Exchanging activation hidden states across PCIe strictly at boundary layers, eliminating inter-layer copy overhead.

### 3. Cache-Friendly Head-Major KV Cache
To maximize hardware prefetching and L1/L2 data cache hit rates during scaled dot-product attention, the KV cache is arranged in a head-major layout:
```
[layer][head][seq_len][head_dim]
```
This guarantees that vector dot-products between query vectors and past key history access contiguous virtual memory addresses.

---

## 🧪 Testing & Verification

Guanaco includes a comprehensive test suite covering mathematical kernel parity, quantization bounds, multi-threading, and context tracking:

```bash
# Run all unit and integration test suites
make test
```

To run with memory leak and undefined behavior diagnostics:
```bash
make debug
make test
```

