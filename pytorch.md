# PyTorch & This Runtime — Relationship, Roles, and Why Not

## Where PyTorch Fits (and Doesn't)

| Context | PyTorch used? | Why |
|---|---|---|
| **Your runtime (inference engine)** | ❌ Never | The whole point is pure C/C++, zero ML framework dependencies. PyTorch is ~2 GB of library; you're building a ~1 MB binary. |
| **Golden data generation** (testing) | ✅ Yes, as a dev tool | You run a forward pass in PyTorch, save intermediate tensors (per-layer outputs), and compare your C++ runtime's output against them. This is how you *validate correctness*. |
| **Model weight conversion** | ✅ Yes, one-time | Models are trained in PyTorch. Someone (you or the community) converts PyTorch `.safetensors` / `.pt` files → GGUF format. After conversion, PyTorch is no longer needed. |
| **Training** | 🚫 Out of scope | You're building inference only. Training stays in PyTorch/JAX. |

---

## What Happens If You Use PyTorch *In the Runtime*?

If you embed PyTorch (e.g., via `libtorch` / TorchScript):

- **Binary size:** ~500 MB–2 GB (vs your target of <10 MB)
- **Startup time:** ~2–5 seconds just to initialize the PyTorch runtime (vs your target of <100 ms)
- **Memory overhead:** ~200–500 MB of framework overhead before model weights even load
- **Control lost:** You can't control memory layout, cache tiling, KV cache strategy, or allocation — PyTorch makes those decisions for you
- **Deployment:** Users must install Python/conda or a massive C++ library — defeats the "single binary, zero deps" goal
- **Performance:** PyTorch's CPU inference is not optimized for autoregressive decode (it's designed for training with GPU). Your hand-tuned kernels will beat it on CPU decode by 2–5×

**In short:** Using PyTorch in the runtime would destroy every advantage of this project.

---

## The Correct Relationship

```
PyTorch (Python, dev machine only)
    │
    ├── Train model (someone else does this)
    ├── Export weights → GGUF (one-time conversion script)
    └── Generate golden test data (save per-layer tensors for correctness testing)
         │
         ▼
Your C/C++ Runtime (ships to users)
    ├── Loads GGUF directly (no PyTorch needed)
    ├── Runs inference with your own kernels
    └── Validates against golden data in CI (the .npy files, not PyTorch itself)
```

PyTorch is a **development dependency** (like a compiler or test framework) — it helps you build and verify the runtime, but it never ships with it and never runs alongside it.

---

## "But PyTorch Is So Advanced — Won't It Make Work Easier?"

PyTorch is incredibly advanced, but it's built for a **different job**.

### The Analogy

| | PyTorch | Your runtime |
|---|---|---|
| **Analogy** | A full commercial kitchen with 500 appliances | A perfectly sharpened knife |
| **Designed for** | Research, training, experimentation, prototyping | One thing: run a trained model as fast as physically possible |
| **Flexibility** | Can do anything (vision, audio, RL, training, inference, custom ops) | Does exactly one thing — LLM inference — and does it optimally |

### Why PyTorch Is *Worse* for This Specific Job

1. **PyTorch doesn't know about KV caching natively.** You'd have to implement KV cache management yourself *on top of* PyTorch's tensor API anyway. So you're doing the hard work regardless, but now with abstraction overhead.

2. **PyTorch's memory allocator is general-purpose.** It calls `malloc`/`free` constantly, has CUDA-aware pooling you don't need, and has no concept of your 3-tier arena/scratch model. Your pointer-bump scratch allocator does a memory "allocation" in **1 nanosecond**. PyTorch's allocator takes **hundreds of nanoseconds** per allocation.

3. **PyTorch dispatches every operation dynamically.** When you do `torch.matmul(Q, K)`, PyTorch goes through ~15 levels of dispatch (autograd check → dtype check → device check → layout check → kernel selection). Your runtime calls `gemm_avx512(Q, K, out)` directly. That dispatch overhead adds **microseconds per op** × thousands of ops per forward pass.

4. **PyTorch can't fuse your way.** You want to fuse RMSNorm + Q projection into one kernel to avoid writing and re-reading the normalized tensor. PyTorch doesn't do this automatically for CPU inference. `torch.compile` helps for GPU, but CPU fusion is limited.

5. **PyTorch ships the whole kitchen.** Even if you only use `matmul` and `softmax`, you're linking ~2 GB of libraries covering autograd, distributed training, JIT compiler, CUDA, MPS, etc.

### Where PyTorch *Would* Make Work Easier

| Task | PyTorch advantage | But... |
|---|---|---|
| Prototyping quickly | `model(input)` in 3 lines | You'd hit a wall at ~5 tok/s on CPU and have no way to improve it |
| Correctness | Battle-tested ops | You use it for golden data generation — best of both worlds |
| Supporting many architectures | `transformers` library has them all | You're targeting Llama-family only (for now), not 500 architectures |
| Autograd / training | Free gradients | You're not training — inference only |

---

## Real-World Proof

**llama.cpp** took exactly this approach — pure C/C++, no PyTorch — and it runs on phones, Raspberry Pis, laptops, and servers. It's the most widely deployed local LLM runtime in the world. If they had used PyTorch, it would need a 2 GB Python environment on every device.

---

## Bottom Line

> PyTorch is the best tool for **building and training models**.
> Your runtime is the best tool for **deploying and running them**.
>
> They're not competing — they serve different stages of the pipeline. Using PyTorch for inference is like using a forklift to deliver a letter. It *works*, but a bicycle is faster, lighter, and fits through the door.
