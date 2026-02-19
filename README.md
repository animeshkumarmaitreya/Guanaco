# LLM Inference Runtime

A CPU-first LLM inference engine in pure C. Zero framework dependencies. Targets x86_64 Linux.

## Quick Start

```bash
# Build (with stubs — compiles immediately)
make

# Run all tests
make test

# Run a specific role's tests
make test_kernels    # Person A
make test_memory     # Person B
make test_tokenizer  # Person C

# Build with ASAN (debug)
make debug
```

## Project Structure

```
src/
├── include/           ← Shared headers (DO NOT modify without group consent)
│   ├── types.h        ← Tensor, ModelConfig, ModelWeights, DataType
│   ├── kernels.h      ← Person A's API
│   ├── memory.h       ← Person B's API
│   ├── tokenizer.h    ← Person C's API
│   └── engine.h       ← Animesh's API
├── stubs/             ← Stub implementations (everyone starts here)
│   ├── stub_kernels.c
│   ├── stub_memory.c
│   ├── stub_tokenizer.c
│   └── stub_engine.c
├── kernels/           ← Person A writes real kernels here
├── memory/            ← Person B writes real allocators + KV cache here
├── tokenizer/         ← Person C writes real tokenizer + sampler here
├── engine/            ← Animesh writes real loader + engine here
└── main.c             ← Entry point

tests/
├── test_kernels.c     ← Person A's test harness
├── test_memory.c      ← Person B's test harness
└── test_tokenizer.c   ← Person C's test harness

golden_data/           ← PyTorch-generated reference tensors (Animesh creates)
```

## Team Roles

| Person | Owns | Branch |
|---|---|---|
| **Person A** | Kernels (GEMM, softmax, RMSNorm, SiLU, RoPE) | `feat/kernels` |
| **Person B** | Memory (Arena, Scratch, KV Cache) | `feat/memory-kv` |
| **Person C** | Tokenizer, Sampler, CLI | `feat/tok-sampler` |
| **Animesh** | IO (GGUF loader), Engine (forward pass, generate) | `feat/io-engine` |

## Git Workflow

1. Everyone starts from `main` (this repo — Day 0 interfaces + stubs)
2. Create your feature branch: `git checkout -b feat/<your-branch>`
3. Build and test your module independently using stubs
4. Push your branch — CI runs your tests
5. Day 8: merge all branches for integration

## How to Switch from Stubs to Real Code

Edit the `Makefile` — uncomment the `REAL_*` lines and comment the `STUB_*` lines for the modules that are ready:

```makefile
# KERNELS = $(STUB_KERNELS)       # ← comment this out
KERNELS = $(REAL_KERNELS)          # ← uncomment this
```
