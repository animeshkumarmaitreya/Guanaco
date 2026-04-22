# Guanaco: Master Engineering & Usage Manual

**Guanaco** is a high-performance, dependency-minimal C framework designed for local LLM inference. It prioritizes system-level efficiency, memory safety, and modularity, offering a state-of-the-art runtime for Transformer-based models.

---

## 🚀 Getting Started

### 1. Build the Runtime
Compile the engine using the automated `build.sh` script:
```bash
./build.sh --cuda    # Enable GPU acceleration
./build.sh --clean   # Perform a fresh rebuild
```

### 2. Run with the Wizard
The easiest way to start is the **Interactive Wizard Mode**:
```bash
./run.sh
```
The wizard will guide you through selecting a model, setting threads, offloading GPU layers, and configuring context limits.

---

## 🛠️ Detailed CLI Reference (`llmrt`)

If you prefer to run the binary directly (`./build/llmrt`), the following flags are available:

| Flag | Description | Default |
|:---|:---|:---|
| `--model <path>` | **[Required]** Path to GGUF model file. | - |
| `--prompt <text>` | Input text for inference. | "Hello" |
| `--chat` | Enter **Interactive Chat REPL** mode. | False |
| `--threads <N>` | Number of CPU threads to use. | 1 |
| `--device <kind>` | Computing backend: `auto`, `cpu`, or `cuda`. | `auto` |
| `--n-gpu-layers <N>` | Number of transformer layers to offload to GPU. | 0 |
| `--ctx <N>` | Maximum context buffer size (token limit). | 8192 |
| `--max-tokens <N>` | Maximum number of new tokens to generate. | 128 |
| `--temperature <F>`| Controls randomness (Lower = more focused). | 0.70 |
| `--top-k <N>` | Limits sampling to top K tokens. | 40 |
| `--top-p <F>` | Nucleus sampling threshold. | 0.90 |

---

## 🧠 Core Engineering Concepts

### 1. The 3-Tier Memory Strategy
Guanaco uses a specialized allocation system to ensure a **Zero-Malloc Hot Path**:

*   **Persistent Tier**: Stores model weights and static engine metadata. Loaded once via `mmap`.
*   **Arena Tier**: High-speed bump-pointer memory for session-specific state (KV Cache, history).
*   **Scratch Tier**: Temporary per-forward-pass memory, recycled instantly after every token.

### 2. Session & Context Management
Guanaco implements three primary strategies for handling long-running interactions:

*   **Persistence**: The entire internal KV cache state can be saved as a `.lctx` file and reloaded later, resuming conversations instantly without re-processing.
*   **Sliding Window (Ring Buffer)**: When the context limit (`--ctx`) is reached, Guanaco evicts the oldest tokens to make room for new ones, allowing "infinite" chat sessions.
*   **Prefix Caching**: Common prompt prefixes (like system instructions) can be cached in memory to speed up multi-session inference.

### 3. Backend Backends & Heterogeneous Scheduling
Guanaco is designed to bridge CPu and GPU compute:

*   **CPU SIMD**: Uses **AVX2 & FMA** instructions for high-bandwidth matrix multiplication.
*   **CUDA Core**: Massively parallel offloading of transformer layers to NVIDIA GPUs.
*   **Heterogeneous Mode**: Automatically splits the model between GPU and CPU based on VRAM availability, ensuring even large models run on consumer hardware.

---

## 💬 Interactive Chat Mode

Enter Chat mode with `./run.sh --chat`. This mode features:
- **Multi-turn reasoning**: Remembrance of previous user inputs.
- **Latency Monitoring**: Real-time display of **TTFT** (Time To First Token) and **Tokens/sec**.
- **Special Command Support**: Use custom escape sequences (if implemented) to reset memory or save sessions.

---

## 📡 Safe Hardware Telemetry (System Monitor)

Guanaco includes a built-in monitoring script to ensure your hardware stays within safe operating parameters during intense heterogeneous inference.

### Running the Monitor
In a separate terminal, execute:
```bash
python3 scripts/system_monitor.py
```

### Key Metrics Tracked
- **CPU/GPU Temperature**: Real-time thermal monitoring with built-in **Safe/Warning/Danger** alerts.
- **VRAM Utilization**: Tracks exactly how much memory is consumed by offloaded transformer layers.
- **Load Logger**: Automatically saves all telemetry data to `metrics.log` for post-inference performance analysis.

---

## 📊 Performance Benchmarking

To demonstrate the full power of Guanaco, try testing your hardware's limits:
```bash
# Benchmark CPU Multithreading (test 1, 4, 8 threads)
./run.sh --model model.gguf --threads 8 --prompt "Explain quantum decoherence."

# Benchmark GPU Acceleration
./run.sh --model model.gguf --n-gpu-layers 32 --device cuda
```

### Metrics to Watch:
- **Tokens/sec**: The raw generation throughput.
- **TTFT**: How quickly the model "understands" your prompt.
- **Memory Pressure**: Monitor system RAM/VRAM usage during high `--ctx` sessions.

---

A "Zero-Dependency" achievement in high-performance C engineering.
