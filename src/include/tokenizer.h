#ifndef LLMRT_TOKENIZER_H
#define LLMRT_TOKENIZER_H

#include "types.h"

/*============================================================================
 * Tokenizer & Sampler API — Person C owns the implementation
 *============================================================================*/

/* ---- Tokenizer ---- */

typedef struct Tokenizer Tokenizer;

/* Load tokenizer vocabulary from the model configuration array.
 * Returns NULL on failure. */
Tokenizer* tokenizer_create(const ModelConfig* cfg);

/* Encode UTF-8 text into token IDs.
 * Returns malloc'd array of token IDs. Caller must free().
 * *out_len is set to the number of tokens. */
int* tokenize(Tokenizer* tok, const char* text, int* out_len);

/* Encode UTF-8 text into token IDs, without injecting BOS.
 * Intended for incremental chat turns where BOS must not be re-added.
 * Returns malloc'd array of token IDs. Caller must free().
 * *out_len is set to the number of tokens. */
int* tokenize_no_bos(Tokenizer* tok, const char* text, int* out_len);

/* Decode a single token ID to its string representation.
 * Returns pointer to internal buffer — do NOT free. Valid until next call. */
const char* detokenize(Tokenizer* tok, int token_id);

/* Free tokenizer resources. */
void tokenizer_destroy(Tokenizer* tok);

/* ---- Sampler ---- */

/* Greedy: return argmax of logits. */
int sample_greedy(const float* logits, int vocab_size);

/* Temperature sampling: scale logits by 1/temperature, softmax, sample.
 * temperature=0 → greedy. */
int sample_temperature(const float* logits, int vocab_size, float temperature);

/* Top-k sampling: keep only the top k logits, zero rest, then temperature sample. */
int sample_top_k(const float* logits, int vocab_size, int k, float temperature);

/* Top-p (nucleus) sampling: keep tokens until cumulative prob >= p. */
int sample_top_p(const float* logits, int vocab_size, float p, float temperature);

/* ---- CLI ---- */

typedef enum {
    DEVICE_AUTO = 0,
    DEVICE_CPU  = 1,
    DEVICE_CUDA = 2,
} DeviceKind;

typedef struct {
    const char* model_path;
    const char* prompt;
    int         max_tokens;
    float       temperature;
    int         top_k;
    float       top_p;
    int         ctx_len;
    int         n_gpu_layers;

    /* Pre-flight scaffolding: used by later backend/chat/thread work. */
    DeviceKind  device;
    int         threads;
    int         chat;

    /* Session management (zero hot-path cost — runs outside decode loop) */
    const char* session_path;       /* --session <file>       Save/load KV state */
    const char* prompt_cache_path;  /* --prompt-cache <file>  Frozen prefix cache */
} CLIArgs;

/* Parse command-line arguments. Returns 0 on success, -1 on error.
 * Prints usage on --help. */
int cli_parse(int argc, char** argv, CLIArgs* args);

#endif /* LLMRT_TOKENIZER_H */
