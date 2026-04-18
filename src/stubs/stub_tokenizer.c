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
    int bos_token_id;
};

Tokenizer* tokenizer_create(const ModelConfig* cfg) {
    (void)cfg;
    Tokenizer* tok = (Tokenizer*)malloc(sizeof(Tokenizer));
    if (!tok) return NULL;
    tok->vocab_size = (cfg && cfg->vocab_size > 0) ? cfg->vocab_size : 32000;
    tok->bos_token_id = (cfg && cfg->bos_token_id > 0) ? cfg->bos_token_id : 1;
    return tok;
}

int* tokenize(Tokenizer* tok, const char* text, int* out_len) {
    (void)tok;
    /* Stub: one token per character (NOT correct BPE — placeholder only) */
    int len = (int)strlen(text);
    if (len == 0) { *out_len = 0; return NULL; }
    int add_bos = (tok && tok->bos_token_id > 0) ? 1 : 0;
    int* ids = (int*)malloc((size_t)(len + add_bos) * sizeof(int));
    int n = 0;
    if (add_bos) ids[n++] = tok->bos_token_id;
    for (int i = 0; i < len; i++) ids[n++] = (unsigned char)text[i];
    *out_len = n;
    return ids;
}

int* tokenize_no_bos(Tokenizer* tok, const char* text, int* out_len) {
    (void)tok;
    int len = (int)strlen(text);
    if (len == 0) { *out_len = 0; return NULL; }
    int* ids = (int*)malloc((size_t)len * sizeof(int));
    for (int i = 0; i < len; i++) ids[i] = (unsigned char)text[i];
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

static DeviceKind parse_device_kind(const char* s) {
    if (!s) return DEVICE_AUTO;
    if (strcmp(s, "auto") == 0) return DEVICE_AUTO;
    if (strcmp(s, "cpu") == 0) return DEVICE_CPU;
    if (strcmp(s, "cuda") == 0) return DEVICE_CUDA;
    return DEVICE_AUTO;
}

int cli_parse(int argc, char** argv, CLIArgs* args) {
    /* Set defaults */
    args->model_path  = NULL;
    args->prompt      = "Hello";
    args->max_tokens  = 128;
    args->temperature = 0.7f;
    args->top_k       = 40;
    args->top_p       = 0.9f;
    args->device      = DEVICE_AUTO;
    args->threads     = 1;
    args->chat        = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --model <path>        Path to GGUF model file\n");
            printf("  --prompt <text>       Input prompt\n");
            printf("  --max-tokens <N>      Max tokens to generate (default: 128)\n");
            printf("  --temperature <float> Sampling temperature (default: 0.7)\n");
            printf("  --top-k <int>         Top-k sampling (default: 40)\n");
            printf("  --top-p <float>       Top-p sampling (default: 0.9)\n");
            printf("  --device <kind>       Device: auto|cpu|cuda (default: auto)\n");
            printf("  --threads <N>         CPU threads (default: 1)\n");
            printf("  --chat                Chat REPL mode\n");
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
        else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            const char* kind = argv[++i];
            args->device = parse_device_kind(kind);
            if (strcmp(kind, "auto") != 0 && strcmp(kind, "cpu") != 0 && strcmp(kind, "cuda") != 0) {
                fprintf(stderr, "Error: invalid --device '%s' (expected auto|cpu|cuda)\n", kind);
                return -1;
            }
        }
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            args->threads = atoi(argv[++i]);
            if (args->threads < 1) {
                fprintf(stderr, "Error: --threads must be >= 1\n");
                return -1;
            }
        }
        else if (strcmp(argv[i], "--chat") == 0) {
            args->chat = 1;
        }
        else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return -1;
        }
    }

    if (args->chat) {
        fprintf(stderr, "Error: --chat is not implemented yet\n");
        return -1;
    }

    if (args->device == DEVICE_CUDA) {
        fprintf(stderr, "Error: --device cuda requested but CUDA is not enabled in this build (try: make USE_CUDA=1)\n");
        return -1;
    }

    if (!args->model_path) {
        fprintf(stderr, "Error: --model is required\n");
        return -1;
    }
    return 0;
}
