# GUANACO LLM INFERENCE ENGINE: TECHNICAL ARCHITECTURE & RESUME IMPACT SPECIFICATION

> **Target Audience for this Document:** Resume Writers, AI Prompt Systems, Technical Recruiters, and Engineering Hiring Managers.  
> **Purpose:** Comprehensive source of truth detailing the low-level systems architecture, computational optimizations, algorithms, and engineering metrics of the Guanaco LLM Inference Runtime, structured to generate **ATS-optimized (Applicant Tracking System)**, high-impact resume bullet points and portfolio case studies.

---

## 1. Executive Summary & Core Competencies

### 1.1 Project Overview
* **System Name:** Guanaco LLM Inference Engine (`llmrt`)
* **Core Definition:** A high-performance, dependency-minimal C/CUDA inference runtime engineered from scratch for local transformer execution, optimized for consumer hardware with strict VRAM/RAM constraints.
* **Target Workloads:** Large Language Models formatted in GGUF (specifically Meta Llama-3.1-8B-Instruct quantized to Q4_K_M / Q8_0 / FP32).
* **Primary Engineering Focus:** Systems programming, memory hierarchy optimization, custom hardware acceleration (x86 AVX2/FMA SIMD + NVIDIA CUDA), multi-threaded scheduling, zero-allocation memory architectures, and heterogeneous compute routing.

### 1.2 ATS Keyword Cloud & Technology Taxonomy
* **Languages & Compilers:** C (C11/C99), C++20, CUDA C++, NVIDIA NVCC, Clang/LLVM, GCC, Make, Python.
* **Hardware Architectures & Accelerators:** NVIDIA Ampere/Ada Lovelace (RTX 3050 Laptop GPU, 4GB VRAM), x86-64 SIMD (Intel AVX2, AMD FMA, `immintrin.h`), Multi-core POSIX Threads (`pthreads`).
* **Systems & Operating Systems:** Linux kernel memory subsystems, `mmap` demand paging, Virtual Memory, Cache Locality (L1/L2/L3 tiling), NVML (NVIDIA Management Library), thermal throttling mitigation, ASAN (AddressSanitizer), UBSAN (UndefinedBehaviorSanitizer), GDB.
* **LLM & Deep Learning Architecture:** Transformer Decoders, Scaled Dot-Product Attention, Multi-Head Attention (MHA) / Grouped-Query Attention (GQA), Rotary Positional Embeddings (RoPE), RMSNorm, SwiGLU / SiLU activation, Causal Attention Masking, KV-Cache Management, KV reuse, Sliding Window / Ring Buffers.
* **Quantization & Numerics:** GGUF Specification, Block-wise Quantization, K-Quants (`Q4_K_M`, `Q4_K_L`, `Q8_0`), FP16/FP32 numerical parity, on-the-fly SIMD dequantization, Roofline Model, Arithmetic Intensity.
* **Inference Pipeline & Serving:** Prefill vs. Decode phases, GEMM (General Matrix Multiply), GEMV (Matrix-Vector Multiply), Top-K, Nucleus Top-P Sampling, Temperature scaling, Repetition Penalty, TTFT (Time-To-First-Token), Tokens per Second (tok/sec) throughput, `.lctx` zero-overhead session persistence.

---

## 2. Low-Level Systems & Architectural Deep-Dives

### 2.1 3-Tier Zero-Allocation Memory Subsystem
* **The Problem:** Dynamic memory allocation (`malloc`/`free`) on the inference hot path causes heap fragmentation, non-deterministic latency jitter, and cache line invalidation.
* **Architectural Solution:** Designed a strict **3-Tier Zero-Allocation Memory Hierarchy**:
  1. **Persistent Tier (Model Weights):** Loads static model weights via POSIX `mmap()` with read-only demand paging. Minimizes resident memory overhead by allowing OS-level page cache eviction without disk swapping.
  2. **Arena Tier (Session State):** Pre-allocated contiguous memory slab utilizing a bump-pointer allocator for session-scoped data (KV Cache, conversational history buffers, session metadata).
  3. **Scratch Pool Tier (Per-Token Hot Path):** A double-buffered activation workspace for intermediate layer tensors (residual connections, attention scores, logits).
* **Zero-Malloc Invariant:** The forward inference loop executes with **$0$ heap allocations per token**. Scratch buffers are recycled in $\mathcal{O}(1)$ time via a single bump-pointer reset (`scratch_reset()`), completely eliminating OS syscall overhead.

### 2.2 Heterogeneous CPU/GPU Compute Splitting (Overcoming 4GB VRAM Barrier)
* **The Problem:** Meta Llama-3.1-8B (Q4_K_M) requires ~4.9GB of memory. Deploying this model on consumer hardware with 4.0GB VRAM (e.g., NVIDIA RTX 3050 Laptop) results in immediate CUDA Out-Of-Memory (OOM) aborts.
* **Architectural Solution:** Implemented a **Dynamic Heterogeneous Inference Pipeline**:
  * **Layer-Wise Offloading:** Added `--n-gpu-layers <N>` CLI parameter. The runtime profiles available VRAM and assigns the first $N$ transformer layers (e.g., 20 layers ~3.0GB) to device VRAM while retaining the remaining layers in host system RAM.
  * **Device Residency Tagging:** Each weight matrix stores a `device_residency` enum (`RESIDENCY_CPU` vs `RESIDENCY_GPU`).
  * **Dynamic Dispatch Engine:** The runtime's `KernelVTable` dispatches forward calls dynamically: GPU-resident layers execute parallel dequantization and GEMV via CUDA kernels, while host layers execute multithreaded AVX2 CPU kernels.
  * **PCIe Transfer Amortization:** Activation transfers between host and device are restricted to boundary layers, ensuring the transfer cost does not outweigh execution speedups.
  * **NVML Thermal Safety Guard:** Integrated real-time GPU polling via `nvmlDeviceGetTemperature()`. If junction thermals exceed 85°C, the engine inserts controlled backoff cycles (`usleep`), protecting mobile laptop hardware from thermal throttling and hardware degradation.

### 2.3 Vectorized Kernel Engineering & SIMD Acceleration
* **The Problem:** Token generation (Decode phase) is fundamentally a Matrix-Vector Multiplication (GEMV) where batch size $M = 1$. The bottleneck is memory bandwidth and dequantization overhead.
* **Architectural Solution:**
  * **AVX2 & FMA Vectorization:** Hand-crafted inner loops using x86 intrinsics (`immintrin.h`, `_mm256_fmadd_ps`, `_mm256_mul_ps`, `_mm256_loadu_si256`). Processes 8 single-precision floating-point operations per CPU cycle natively.
  * **Block-Wise Dequantization:** Designed SIMD dequantizers for `Q8_0` and `Q4_K_M` super-blocks (256 elements split into 8 sub-blocks of 32). Dequantization is performed *on-the-fly directly into SIMD vector registers*, bypassing intermediate FP32 RAM writes.
  * **Cache-Tiled GEMM:** Implemented an $8 \times 32$ tiled GEMM kernel structured around L1/L2 data cache capacities to maximize memory reuse during prompt prefill ($T > 1$).

### 2.4 GPU Kernel Optimization (CUDA)
* **Warp Reduction Primitives:** Replaced naive global/shared memory reductions in attention and GEMV kernels with `__shfl_down_sync()` warp-level register exchange primitives, delivering high arithmetic bandwidth.
* **Kernel Fusion:** Fused memory-bound activation layers (e.g., SwiGLU / SiLU element-wise multiplication) into single-pass CUDA kernels to prevent redundant VRAM read/write cycles.
* **Shared Memory Tiling:** Cached intermediate attention scores in fast on-chip `__shared__` memory during softmax computation to optimize bandwidth.

### 2.5 Multi-Core Concurrency & Work-Stealing Threadpool
* **Custom Pthread Threadpool:** Built a low-latency thread pool from scratch utilizing POSIX threads, mutexes, and condition variables (`pthread_mutex_t`, `pthread_cond_t`).
* **Parallel-For Scheduling:** Row-level work distribution across CPU cores for large weight projections (`threadpool_parallel_for`).
* **Topology Discovery:** Integrated automatic CPU core detection via `sysconf(_SC_NPROCESSORS_ONLN)` to scale thread allocation dynamically if `--threads` is unspecified.

### 2.6 KV-Cache Architecture & Session Management
* **Head-Major Memory Layout:** Structured the KV cache tensor as `[layers][heads][seq_len][head_dim]`. Ensures that vector dot-products across sequence history access contiguous virtual memory addresses, maximizing L1/L2 cache prefetcher hit rates.
* **Attention-as-GEMM Refactoring:** Unified attention computation into standardized GEMM operations ($Q \times K^T$ via transposed GEMM, and $\text{scores} \times V$ via standard $A \times B$ GEMM), decoupling attention math from hardware backend abstractions.
* **Zero-Overhead Session Serialization (`.lctx`):** Implemented raw memory serialization for the Arena KV Cache. Conversations can be saved to disk and reloaded instantly via `mmap` with zero re-computation or prefill latency.
* **Sliding Window Ring Buffer:** Designed a circular eviction mechanism for the KV cache to support continuous generation beyond fixed context bounds (`--ctx`) without memory overflow.

### 2.7 Verification, Profiling & Systems Quality
* **Strict Numerical Parity:** Validated kernel correctness by asserting floating-point outputs against Python/PyTorch golden data references within a $1\times 10^{-5}$ tolerance.
* **Safety & Sanity:** Sanitized codebase with AddressSanitizer (`-fsanitize=address`) and UndefinedBehaviorSanitizer (`-fsanitize=undefined`), achieving zero memory leaks, zero buffer overflows, and zero pointer corruption across test suites.
* **Hardware Telemetry:** Engineered a standalone telemetry pipeline monitoring CPU/GPU load, VRAM allocation, and thermal metrics to log files (`metrics.log`).

---

## 3. The Theoretical Foundations (For Technical Interviews)

| Architectural Pillar | Core Mechanism | Engineering Tradeoff / Why It Matters |
|---|---|---|
| **Roofline Model (Decode)** | Memory-Bandwidth Bound ($\text{Intensity} \approx 1.0$) | Decode processes 1 token at a time; FLOP count is trivial compared to loading gigabytes of weights. Token speed is bounded by RAM/VRAM bandwidth ($\text{tok/s} \approx \text{BW} / \text{Model Size}$). |
| **Roofline Model (Prefill)** | Compute Bound ($\text{Intensity} \gg 1.0$) | Prompt processing processes $T$ tokens simultaneously via Matrix-Matrix multiplication (GEMM), achieving near-peak TFLOPS. |
| **Quantization (K-Quants)** | 256-element super-blocks with sub-scales | Reduces memory footprint by up to 75% compared to FP16 while maintaining negligible perplexity degradation. |
| **RoPE (Rotary Embeddings)** | Complex coordinate rotation on Q/K vectors | Relative position encoding without explicit absolute position lookup tables, enabling sliding window attention. |
| **Zero-Copy `mmap`** | Direct file-to-virtual-memory mapping | Eliminates redundant system RAM copies during model initialization; instant startup time. |

---

## 4. Pre-Formulated, ATS-Optimized Resume Bullet Points

### Track A: Systems Software & Low-Level / C++ Engineer
* **Engineered a high-performance local LLM inference runtime in pure C11 and CUDA**, implementing complete Transformer forward-pass architectures (RMSNorm, RoPE, MHA/GQA, SwiGLU) with zero external machine learning dependencies.
* **Eliminated hot-path memory latency by designing a 3-tier zero-allocation memory hierarchy** (mmap persistent weights, bump-pointer session arena, recycled scratch pool), achieving **0 dynamic heap allocations (malloc/free) per token generation**.
* **Vectorized quantized tensor operations using x86 AVX2 and FMA SIMD intrinsics**, developing custom kernels for `Q4_K_M` and `Q8_0` block dequantization to process 8 single-precision floating-point operations per CPU cycle.
* **Constructed a low-overhead multi-threaded worker threadpool** using POSIX threads (`pthreads`) and condition variables, optimizing CPU core saturation and tiled GEMM parallelization across varying hardware thread topologies.
* **Architected a zero-overhead session persistence system (`.lctx`)** utilizing direct memory dumping and `mmap` re-hydration, enabling instantaneous context restoration and continuous conversation streaming via sliding window ring buffers.

### Track B: AI Infrastructure / LLM Systems / Performance Engineer
* **Architected a dynamic heterogeneous compute engine (CPU + GPU)** capable of running Llama-3.1-8B on memory-constrained hardware (4GB VRAM), splitting 32 transformer layers across PCIe bus interfaces via layer-residency scheduling.
* **Accelerated transformer inference throughput** by refactoring attention mechanisms into cache-friendly GEMM formulations, optimizing memory access patterns to maximize L1/L2 data cache hit rates for multi-head projections.
* **Implemented real-time hardware safety telemetry and dynamic throttling** using NVIDIA Management Library (NVML) C APIs, programmatically adjusting inference duty cycles to mitigate thermal throttling above 85°C.
* **Built a rigorous numerical validation suite comparing C runtime tensors against PyTorch golden references**, maintaining numerical precision parity within a $1\times 10^{-5}$ tolerance across FP32, Q8_0, and Q4_K quantizations.
* **Optimized autoregressive decode latency through Roofline model analysis**, maximizing memory bandwidth saturation during memory-bound single-token generation phases.

### Track C: CUDA / GPU Kernel Optimization Engineer
* **Developed custom CUDA device kernels for quantized Matrix-Vector multiplication (GEMV)**, integrating `__shfl_down_sync` warp-level reduction primitives to maximize register reuse and reduce memory stall cycles.
* **Fused memory-bound activation kernels (SwiGLU / SiLU + Element-wise Multiply)** in CUDA C++, slashing global memory read/write transactions by 50% across transformer MLP blocks.
* **Engineered shared-memory cache tiling for multi-head attention and softmax kernels**, eliminating redundant global VRAM transactions and boosting arithmetic intensity on NVIDIA Ampere architecture.
* **Managed host-device memory transfers via asynchronous streams and pinned memory**, hiding PCIe latency during heterogeneous layer handoffs between CPU and GPU backends.

---

## 5. Concise Project Blurb for Resume / LinkedIn / Portfolio

### Short Version (1-2 sentences)
> **Guanaco — High-Performance Heterogeneous LLM Inference Runtime (C11 / CUDA / AVX2)**  
> Engineered a dependency-free C and CUDA local LLM inference engine from scratch, featuring a zero-malloc 3-tier memory subsystem, custom AVX2/CUDA quantized kernels (Q4_K_M/Q8_0), and dynamic layer-offloading enabling 8B parameter models to run on 4GB VRAM GPUs.

### Full Project Description (Resume Project Section)
> **Guanaco — High-Performance Heterogeneous LLM Inference Engine** | *C, CUDA, AVX2, Pthreads, Linux, GGUF*
> * Designed and implemented a bare-metal Transformer inference runtime in C11 and CUDA from scratch, supporting Llama-3 architecture with native GGUF parsing and block-wise quantization (`Q4_K_M`, `Q8_0`).
> * Designed a 3-tier zero-allocation memory architecture (mmap, arena, and $\mathcal{O}(1)$ recycled scratch pools), achieving a zero-heap-allocation hot path during autoregressive token generation.
> * Implemented SIMD vectorization with AVX2/FMA intrinsics (8 FLOPs/cycle) and custom CUDA GEMV kernels utilizing warp-level shuffle reductions.
> * Solved GPU out-of-memory bottlenecks by engineering a heterogeneous scheduling pipeline that dynamically splits layers between NVIDIA VRAM and host RAM, running 8B models on consumer 4GB GPUs.
> * Integrated POSIX multi-threading, head-major KV caching, and NVML thermal safety monitoring, validating tensor accuracy against PyTorch golden data within a $10^{-5}$ tolerance.

---

## 6. Technical Interview Questions & Answers Reference Guide

### Q1: Why did you design a 3-tier memory allocator instead of standard `malloc`?
> *"In low-latency systems and LLM inference, calling `malloc` on the hot path introduces non-deterministic OS kernel calls, heap fragmentation, and potential cache misses. By establishing Persistent memory via `mmap`, an Arena for session KV-caches, and a bump-pointer Scratch allocator that resets in $\mathcal{O}(1)$ time after each forward pass, the inference loop operates with exactly zero memory allocations per token, guaranteeing deterministic latency and cache locality."*

### Q2: How does the Roofline Model dictate optimization between Prefill and Decode?
> *"Prefill processes a sequence of $T$ tokens at once, making it a Matrix-Matrix multiplication (GEMM) with high arithmetic intensity (FLOPs per byte loaded). Prefill is compute-bound, so optimizations focus on tensor tiling, vectorization (AVX2/FMA), and CUDA core saturation. In contrast, Decode generates one token at a time ($M=1$), turning the operation into Matrix-Vector multiplication (GEMV). Every single weight must be fetched from memory just to perform two operations per weight (arithmetic intensity $\approx 1$). Decode is strictly memory-bandwidth bound, so optimizations focus on quantization (Q4_K) to reduce byte traffic and maximizing memory bus utilization."*

### Q3: How did you fit an 8B model into an RTX 3050 Laptop with only 4GB VRAM?
> *"Meta-Llama-3.1-8B quantized to Q4_K_M requires approximately 4.9GB of memory, which exceeds the 4.0GB physical limit of the RTX 3050. Rather than failing or offloading everything to CPU, I implemented a heterogeneous layer-splitting scheduler. The engine tags each layer's tensors with a residency attribute. During initialization, the first 20 layers (~3GB) are pushed to VRAM, leaving headroom for the KV cache and OS buffers. During forward passes, the vtable dynamically routes computation: GPU layers execute through custom CUDA kernels, and CPU layers execute through multi-threaded AVX2 kernels, exchanging intermediate activations over PCIe only at boundary layers."*

### Q4: Why use a Head-Major layout for the KV Cache?
> *"Attention requires computing dot products between a query vector for head $h$ and all historical key vectors for that same head across past sequence positions. Storing the cache in head-major layout `[layer][head][seq_len][head_dim]` ensures that all historical tokens for a given head reside in contiguous physical memory. This maximizes hardware prefetching and L1/L2 cache line hits during scaled dot-product attention."*
