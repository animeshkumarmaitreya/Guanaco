# GDB Debugging Guide for LLM Inference

When writing low-level C code—especially dealing with gigantic memory buffers and complex math like in our Large Language Model runtime—crashes (like Segmentation Faults) and silent logical bugs are inevitable. 

This document explains how we used **GDB (GNU Debugger)** to safely investigate and fix the exact crashes we encountered while building this framework.

---

## 1. Preparing for Debugging: Building with Symbols
By default, the `Makefile` compiles code with aggressive optimizations (`-O2` or `-O3`) to make the math run as fast as possible. However, the compiler strips away variable names and reorganizes code, making it impossible to step through line-by-line.

To use GDB, we first had to disable optimizations and include "Debug Symbols" (the `-g` flag).
```bash
# How we originally compiled for debugging:
make clean
make CFLAGS="-g -O0 -Wall -Wextra -Isrc/include"
```
*   `-g`: Embeds human-readable variable names and line numbers into the binary.
*   `-O0`: Disables code optimizations so the execution flow matches the C code exactly.

---

## 2. Starting the Debugger
Instead of running `./build/llmrt` directly, you pass the executable into GDB, along with all the arguments the program expects:

```bash
gdb --args ./build/llmrt --model /path/to/Llama-3.2.gguf --prompt "Hello" --max-tokens 50
```

Once inside the `(gdb)` prompt, simply type `run` (or `r`) and press Enter to start the engine.

---

## 3. Real-World Example: Debugging the Segmentation Fault
During development, the engine crashed instantly with a `Segmentation fault (core dumped)` when attempting to generate the first token. Here is exactly how GDB pinpointed the problem for us:

### Step A: Catching the Crash
We typed `run` inside GDB. The program outputted the model loading logs, but then GDB intercepted the crash and paused the program immediately:
```
Program received signal SIGSEGV, Segmentation fault.
0x00007ffff7a4a9c6 in __memset_avx2_unaligned_erms () from /lib64/libc.so.6
```
GDB told us the crash occurred inside a `memset` function deep in the C standard library. But *where* in our code did we call that `memset`?

### Step B: The Backtrace (`bt`)
We used the `backtrace` (or `bt`) command to see the "call stack"—the list of functions that led to the crash.
```
(gdb) bt
#0  0x00007ffff7a4a9c6 in __memset_avx2_unaligned_erms () from /lib64/libc.so.6
#1  0x0000000000403f0b in gemm_f32 (A=0x7ffff7fa3010, B=0x7ffff7fa3060, C=0x7ffff7fa30b0) at src/kernels/kernels.c:13
#2  0x00000000004052a2 in transformer_layer (...) at src/engine/engine.c:203
#3  0x00000000004068f0 in forward (...) at src/engine/generate.c:120
```
**Conclusion:** GDB instantly proved that `kernels.c` line 13 inside `gemm_f32()` was causing the crash.

### Step C: Inspecting the Scene (`frame` and `print`)
We typed `frame 1` (or `f 1`) to jump up into the `gemm_f32` environment, so we could look at our own variables:
```
(gdb) frame 1
#1  0x0000000000403f0b in gemm_f32 (A=0x7ffff... , B=0x7ffff... , C=0x7ffff...) at src/kernels/kernels.c:13
13          memset(c_data, 0, (size_t)M * N * sizeof(float));
```

Now we wanted to see *why* the `memset` was accessing forbidden memory. We used the `print` (or `p`) command to dump the values of the variables:
```
(gdb) print M
$1 = 1
(gdb) print N
$2 = 14336
(gdb) print C->data
$3 = (void *) 0x0   <--- BINGO!
```
**The Discovery:** GDB showed us that `C->data` was a **NULL pointer**. 

### Step D: Finding the Root Cause
Knowing that `C` was strictly NULL, we just had to look up the call stack to see where `C` was created. We realized `C` was allocated by the `Scratch` memory allocator in `generate.c`. 
Our scratch allocator had a hardcoded limit of `256 MiB`. But LLaMA 3.2's massive `14336` dimension matrices required more space, causing the allocator to silently fail and return NULL. 
*   **The Fix:** We increased the scratch space to `512 MiB`, and the crash vanished!

---

## 4. Cheat Sheet: Most Useful GDB Commands

Here are the critical commands we used throughout the framework's development:

| Command | Shorthand | What it does |
| :--- | :--- | :--- |
| `run` | `r` | Starts the program. You can also pass arguments here (e.g. `run --model x.gguf`). |
| `break [file:line]` | `b` | Sets a breakpoint. GDB will pause execution when it reaches this exact line. (e.g. `b src/kernels/kernels.c:50`) |
| `continue` | `c` | If paused at a breakpoint, this resumes execution until the next breakpoint or crash. |
| `step` | `s` | Moves forward exactly 1 line of code. If it's a function call, it steps *into* the function. |
| `next` | `n` | Moves forward 1 line of code, but steps *over* function calls without diving into them. |
| `backtrace` | `bt` | Prints the Call Stack. Critical for finding out how the program got to the crash. |
| `frame [number]`| `f [num]` | Jumps to a specific level in the backtrace so you can inspect its variables. |
| `print [var]` | `p [var]` | Prints the value of a variable, struct, or pointer (e.g. `p A->shape[0]`). |
| `quit` | `q` | Exits the debugger. |

## 5. Summary
Using GDB transforms debugging from "adding hundreds of `printf` lines and guessing" into a surgical operation. Whenever the engine generates garbage tokens, segfaults, or hangs, compiling with `CFLAGS="-g -O0"` and running `gdb --args ./build/llmrt ...` is always step one.
