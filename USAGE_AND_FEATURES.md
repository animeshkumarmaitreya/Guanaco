# Guanaco LLM Inference Runtime — Usage & Features

The **Guanaco LLM Inference Runtime** is a high-performance, dependency-minimal C framework designed for local LLM inference. It prioritizes system-level efficiency, memory safety, and modularity, making it an ideal demonstration tool for understanding LLM internals.

---

## 🚀 Core Features

### 1. High-Performance CPU Kernels
- **SIMD Optimized**: Built-in support for **AVX2** and **FMA** instruction sets for rapid matrix multiplications and activations.
- **Multithreaded Execution**: Optional `pthreads` backend allows scaling inference across multiple CPU cores via tiled GEMM distribution.
- **Stateless Design**: Kernels are decoupled from memory management, ensuring deterministic behavior and ease of testing.

### 2. Advanced Memory System
- **Arena Allocation**: Uses a bump-pointer Arena for per-session memory (KV cache, history), eliminating fragmentation.
- **Scratch Pool**: Per-forward pass scratch memory that is recycled instantly, maintaining a zero-malloc hot path.
- **Efficient KV Cache**: Optimized head-major layout for the KV cache to maximize L1/L2 cache locality during scaled dot-product attention.

### 3. Comprehensive Model Support
- **GGUF Format**: Native support for loading GGUF models, including metadata parsing and weight mapping.
- **Quantization Ready**: Infrastructure for **Q4_0**, **Q8_0**, and **FP32** weight tensors, enabling high-quality inference with reduced RAM footprints.
- **Incremental Decoding**: Fully implements positional embeddings (RoPE) and causal masking for stable autoregressive generation.

### 4. Robust Sampling System
- **Top-K & Top-P**: Advanced samplers to balance diversity and coherence.
- **Temperature Scaling**: Controls output randomness for various creative or deterministic tasks.
- **Greedy Sampling**: Optimized fallback for deterministic "best-fit" token selection.

### 5. Interactive Interfaces
- **Streaming CLI**: Real-time token output with latency metrics.
- **Chat REPL**: A multi-turn interactive mode that preserves context efficiently across conversations.

---

## 🛠 Usage Guide

### 1. Building the Runtime
The runtime can be built using the provided `Makefile`.

```bash
# Standard build (AVX2 enabled)
make

# Build with multithreading support (recommended for multi-core CPUs)
make USE_PTHREAD=1

# Build with debug instrumentation (ASAN/UBSAN)
make debug
```

### 2. CLI Basic Usage
All operations are routed through the `llmrt` binary found in the `build/` directory.

```bash
./build/llmrt --model <path_to_gguf> --prompt "Explain quantum physics like I'm five."
```

### 3. Detailed CLI Flags

| Flag | Description | Default |
|---|---|---|
| `--model <path>` | **[Required]** Path to your GGUF model file. | - |
| `--prompt <text>` | Input text to start generation. | "Hello" |
| `--chat` | Enter **Interactive Chat Mode** (REPL). | False |
| `--max-tokens <N>` | Maximum number of tokens to generate. | 128 |
| `--threads <N>` | Number of CPU threads (requires `USE_PTHREAD=1`).| 1 |
| `--temperature <F>`| Controls randomness (Lower = more focused). | 0.70 |
| `--top-k <N>` | Limits sampling to top K tokens (0 to disable). | 40 |
| `--top-p <F>` | Nucleus sampling threshold (1.0 to disable). | 0.90 |
| `--device <kind>`| Backend device: `auto`, `cpu`, or `cuda`. | auto |

---

## 💡 Demonstration Scenarios

### A. Performance Benchmarking
Test how threading impacts your token throughput:
```bash
./build/llmrt --model models/llama-3-8b-q4.gguf --prompt "Write a short story." --threads 8
```

### B. Interactive Conversation
Use the chat mode to test multi-turn reasoning and context retention:
```bash
./build/llmrt --model models/tinyllama-1.1b-fp32.gguf --chat
```

### C. Creative Generation tuning
Adjust sampling parameters to see how the model's "personality" changes:
```bash
# Creative/Unpredictable
./build/llmrt --model models/model.gguf --temperature 1.2 --top_p 0.95

# Deterministic/Factual
./build/llmrt --model models/model.gguf --temperature 0.2 --top_k 1
```

---

## 📊 Technical Specifications & Metrics

When running, the CLI provides real-time feedback including:
- **TTFT (Time To First Token)**: Measures prompt prefill latency.
- **Tokens/sec**: Average throughput during full generation.
- **Model Metadata**: Prints architecture details, quantization level, and vocabulary size on load.
