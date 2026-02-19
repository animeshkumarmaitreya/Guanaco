/*============================================================================
 * Tokenizer & Sampler Stubs — Person C replaces these
 *============================================================================*/

#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>

/* ---- Tokenizer stub ---- */

struct Tokenizer {
    int vocab_size;
};

Tokenizer* tokenizer_create(const char* model_path) {
    (void)model_path;
    Tokenizer* tok = (Tokenizer*)malloc(sizeof(Tokenizer));
    if (!tok) return NULL;
    tok->vocab_size = 32000;  /* default LLaMA vocab size */
    return tok;
}

int* tokenize(Tokenizer* tok, const char* text, int* out_len) {
    (void)tok;
    /* Stub: one token per character (NOT correct BPE — placeholder only) */
    int len = (int)strlen(text);
    if (len == 0) { *out_len = 0; return NULL; }
    int* ids = (int*)malloc(len * sizeof(int));
    for (int i = 0; i < len; i++) {
        ids[i] = (unsigned char)text[i];  /* ASCII as token ID */
    }
    *out_len = len;
    return ids;
}

static char detok_buf[16];
const char* detokenize(Tokenizer* tok, int token_id) {
    (void)tok;
    /* Stub: print the token ID as a character if ASCII, else "?" */
    if (token_id >= 32 && token_id < 127) {
        detok_buf[0] = (char)token_id;
        detok_buf[1] = '\0';
    } else {
        snprintf(detok_buf, sizeof(detok_buf), "[%d]", token_id);
    }
    return detok_buf;
}

void tokenizer_destroy(Tokenizer* tok) {
    free(tok);
}

/* ---- Sampler stubs ---- */

int sample_greedy(const float* logits, int vocab_size) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

int sample_temperature(const float* logits, int vocab_size, float temperature) {
    /* Stub: just use greedy */
    (void)temperature;
    return sample_greedy(logits, vocab_size);
}

int sample_top_k(const float* logits, int vocab_size, int k, float temperature) {
    (void)k; (void)temperature;
    return sample_greedy(logits, vocab_size);
}

int sample_top_p(const float* logits, int vocab_size, float p, float temperature) {
    (void)p; (void)temperature;
    return sample_greedy(logits, vocab_size);
}

/* ---- CLI stub ---- */

int cli_parse(int argc, char** argv, CLIArgs* args) {
    /* Set defaults */
    args->model_path  = NULL;
    args->prompt      = "Hello";
    args->max_tokens  = 128;
    args->temperature = 0.7f;
    args->top_k       = 40;
    args->top_p       = 0.9f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --model <path>        Path to GGUF model file\n");
            printf("  --prompt <text>       Input prompt\n");
            printf("  --max-tokens <N>      Max tokens to generate (default: 128)\n");
            printf("  --temperature <float> Sampling temperature (default: 0.7)\n");
            printf("  --top-k <int>         Top-k sampling (default: 40)\n");
            printf("  --top-p <float>       Top-p sampling (default: 0.9)\n");
            printf("  --help                Show this help\n");
            return 1;  /* signal help printed */
        }
        else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            args->model_path = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc)
            args->prompt = argv[++i];
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc)
            args->max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc)
            args->temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc)
            args->top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc)
            args->top_p = (float)atof(argv[++i]);
        else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return -1;
        }
    }

    if (!args->model_path) {
        fprintf(stderr, "Error: --model is required\n");
        return -1;
    }
    return 0;
}
