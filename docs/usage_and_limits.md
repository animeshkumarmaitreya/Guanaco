# Guanaco Intercept Engine: Usage, Tokens & Hardware Limits

This document serves as a comprehensive guide for using the `llmrt` command-line interface, understanding the mechanics of token generation, and the theoretical limits you can hit when switching from CPU inference to GPU hardware acceleration.

---

## 1. Complete Usage Guide & CLI Flags

The `llmrt` executable provides several flags to control both *what* the model processes and *how* it decides on the output text. Every time the engine generates a word, it calculates the probability for all 128,000+ words in its vocabulary. The flags below dictate how the engine selects the winning word from those probabilities.

### Basic Setup Flags
* **`--model <path>`** *(Required)*
  * **What it does**: Specifies the path to your `.gguf` weight file (e.g., LLaMA 3.2).
  * **What happens under the hood**: The `loader.c` module reads the header of this file to determine the architecture (hidden dimensions, layers, vocabulary size), and uses `mmap` to load the 20GB+ tensor matrices directly into your RAM for the execution loop.

* **`--prompt <text>`** *(Default: "Hello")*
  * **What it does**: The starting input string that gives the model context.
  * **What happens under the hood**: The tokenizer breaks this string down into a series of integer IDs. The engine then runs a massive "Prefill" matrix multiplication on these IDs simultaneously to build up the internal conversation memory (the **KV Cache**) before it even tries to guess the first new word.

* **`--max-tokens <N>`** *(Default: 128)*
  * **What it does**: Sets a hard limit on the total number of new tokens the model is allowed to generate.
  * **What happens under the hood**: The main decode `for` loop in `generate.c` will terminate if it hits this number. It acts as a safety against infinite generation loops. The process can end earlier if the model outputs an organic End-of-Sequence (EOS) token or if the engine hits its maximum context window size.

### Advanced Sampling & Creativity Flags
These flags manipulate the likelihood arrays *before* the model selects a token.

* **`--temperature <float>`** *(Default: `0.7`)*
  * **What it does**: Controls how random or creative the generated text is. 
  * **What happens under the hood**: All the raw logit scores outputted by the final projection layer are divided by this number.
    * **`0.0` (Greedy Search)**: The math collapses. The model becomes entirely deterministic, picking the #1 highest probability token 100% of the time. Use this for coding, structured JSON, or factual Q&A.
    * **`0.1 - 0.5`**: Low randomness. The model stays highly focused but avoids strict repetition routines.
    * **`0.7 - 0.9`**: Normal/High randomness. Allows the model to naturally pick the 2nd or 3rd best choices, leading to more conversational, "human-like" text.
    * **`> 1.0`**: Chaos. The probability spread flattens out, meaning the 500th best word starts looking almost as good as the 1st best word, resulting in disjointed gibberish.

* **`--top-k <N>`** *(Default: `40`)*
  * **What it does**: A hard cutoff that ignores any token outside the top `N` most likely choices.
  * **What happens under the hood**: The engine sorts the 128,000 likelihoods and immediately deletes the bottom 127,960. The temperature randomness is only applied to the remaining 40 coherent words, heavily reducing "hallucinations" (bizarre words) when using high temperatures.

* **`--top-p <float>`** *(Default: `0.9`)*
  * **What it does**: Dynamic Nucleus Sampling. Limits the word pool based on cumulative probability instead of a hard numbered cutoff limit.
  * **What happens under the hood**: The engine ranks words and adds them to a shrinking pool until their probabilities add up to `90%` (or `0.9`). If the model is extremely confident, the pool might only consist of 1 word. If the model is unsure, the pool dynamically expands to 50+ possible words. This is mathematically superior to `--top-k` for capping hallucinations while maintaining creativity.

---

## 2. Tokenization: Vocabularies, Dictionaries, and Tokens

To understand how an LLM thinks, you must realize that **AI models cannot read English letters**. They can only process massive arrays of mathematics. We must convert human text into numbers before the engine can understand it, and we must convert the engine's output numbers back into text so humans can read it.

### What is a Vocabulary / Dictionary?
Just like a human dictionary contains every word you know, an AI model has a hardcoded **Vocabulary** (also called a Dictionary). 
* When Meta trained LLaMA 3.2, they defined exactly **128,256** unique text fragments it is allowed to use. 
* This Vocabulary is literally just a giant numbered list saved inside the `.gguf` file. 
* **Example of a Vocabulary List:**
  * `0` = [Unknown]
  * `1` = [Start of Text]
  * `2` = [End of Text]
  * ...
  * `8492` = " Apple"
  * `10243` = "ing"

### What Exactly IS a "Token"?
A token is a single entry from that numbered Vocabulary list. It is **not** necessarily a full word. 

Because the dictionary size is limited to 128,256 slots, the AI cannot possibly store every single rare word, medical term, or foreign language string in the entire world. Instead, the dictionary contains:
1. **Full common words:** (e.g., `"The"`, `"Apple"`, `"France"`)
2. **Sub-word chunks:** (e.g., `"Un"`, `"belie"`, `"vable"`)
3. **Individual letters/bytes:** (For when a word is so rare it must be spelled out letter-by-letter).

**How the Tokenizer works:**
When you type the prompt: **"I ate an Unbelievable Apple."**, the `src/tokenizer/tokenizer.c` script looks at the model's Dictionary and chops your sentence into the biggest matching chunks it can find:
* `["I"]`, `[" ate"]`, `[" an"]`, `[" Un"]`, `["belie"]`, `["vable"]`, `[" Apple"]`, `["."]`.

It then converts these chunks into integers (Tokens): `[42, 902, 38, 9211, 4022, 112, 8492, 13]`. 
**These numbers are what the Engine actually processes!** 

### The Token Ratio:
Because many complex words are split into 2 or 3 tokens, on average across the English language:
* **1 Token ≈ 0.75 Words** (or about 4 characters).
* If you set `--max-tokens 512`, the AI will generate about a **380-word essay**.

When the engine "generates a token", it isn't thinking of a sentence. It is executing 30+ billion mathematical operations simply to predict the single next most statistically probable sub-word chunk to append to your prompt.

---

## 3. Theoretical Hardware Limits: CPU vs. GPU Architecture

The Guanaco framework is currently a pure **CPU-bound** C11 engine utilizing AVX2 logic. If you were to implement a `gemm_f32_cuda` backend (moving the matrix math to an NVIDIA GPU), the performance ceiling shatters. 

Here are the strict physical limitations dictated by hardware:

### The CPU Limit (Current State)
Generating tokens (the Decode Phase) is strictly bottlenecked by **Memory Bandwidth**. A single token generation requires the engine to load the *entire weight of the model* from your RAM into your CPU's L3 cache, multiply it against your prompt's context state, and throw it out.

* **Limit:** High-end consumer DDR5 RAM physically maxes out around **80 to 100 GB/s** of data transfer.
* **Math:** If you use a massive 7 Billion Parameter model in pure FP32 format (which is 28 Gigabytes of data), your CPU can physically only move that data across the motherboard ~3 times a second:
  * `100 GB/s \ 28 GB = ~3.5 tokens per second absolute maximum`.
* **The Reality:** Without moving to compressed 4-bit/8-bit Quantized models (shrinking the 28GB model down to 4GB), pure FP32 execution on consumer CPUs will permanently hover between 1 to 5 tokens per second.

### The GPU Limit (Future State)
NVIDIA (and Apple Silicon) GPUs do not use standard DDR5 RAM. They use GDDR6X VRAM soldered directly alongside thousands of specialized Tensor matrix cores, unlocking massive memory highways.

* **Limit:** An consumer-grade NVIDIA RTX 4090 possesses a memory bandwidth of **~1,008 GB/s**. Enterprise H100s exceed **3,000+ GB/s**.
* **Math:** If we move the exact same 28GB FP32 LLaMA model into VRAM:
  * `1008 GB/s \ 28 GB = ~36 tokens per second generated limit`.
* **The Context Limit (VRAM Constraints):** While GPUs decimate CPUs in math speed, their fatal flaw is total capacity. 
  * A CPU has access to 64GB+ of system memory, allowing for massive 100,000+ token KV Cache contexts (whole books).
  * An RTX 4090 only possesses 24GB of VRAM. A 14B parameter model will completely fill the card, leaving almost 0 bytes leftover for context memory. Attempting to converse with the model will repeatedly trigger Out-Of-Memory (OOM) driver crashes.

**Conclusion:** 
Moving the Guanaco engine to a GPU backend would instantly catapult generation speeds from unreadable ~2 tok/s into fluid, real-time ~35+ tok/s speeds. However, you would be strictly limited to smaller models (1B to 7B sizes) to prevent VRAM exhaustion during long conversations.
