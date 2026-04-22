/*============================================================================
 * Tokenizer & Sampler implementation (Replacing Person C's stubs)
 *============================================================================*/

#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <unistd.h>

/* ---- Tokenizer ---- */

struct Tokenizer {
    int vocab_size;
    char** vocab;
    float* scores;
    int bos_token_id;
};

typedef struct {
    int idx;
    float logit;
} LogitIndex;

static int logit_index_cmp_desc(const void* a, const void* b) {
    float va = ((const LogitIndex*)a)->logit;
    float vb = ((const LogitIndex*)b)->logit;
    return (va < vb) ? 1 : (va > vb) ? -1 : 0;
}

Tokenizer* tokenizer_create(const ModelConfig* cfg) {
    if (!cfg || !cfg->vocab_strings) return NULL;
    Tokenizer* tok = (Tokenizer*)calloc(1, sizeof(Tokenizer));
    if (!tok) return NULL;
    tok->vocab_size = cfg->vocab_size;
    tok->vocab = cfg->vocab_strings;
    tok->scores = cfg->vocab_scores;
    /* Loader sets a default BOS; if a model legitimately uses BOS=0, respect it. */
    tok->bos_token_id = (cfg->bos_token_id >= 0) ? cfg->bos_token_id : 1;
    return tok;
}

    static int* tokenize_impl(Tokenizer* tok, const char* text, int add_bos, int* out_len) {
        if (!text || text[0] == '\0') {
            *out_len = 0;
            return NULL;
        }

        /* For Llama, text usually gets a space prepended (" ") mapped to U+2581 "_" */
        /* We'll do a very basic eager greedy matching for brevity in this engine,
           rather than strict BPE, just to ensure it's functional.
           A true BPE splits to chars, finds best merge pairs iteratively. */

        int len = (int)strlen(text);
        int* ids = (int*)malloc((len + 2) * sizeof(int));
        int n_tokens = 0;

        if (add_bos) {
            /* Add BOS token (from GGUF; fallback preserves prior behavior) */
            ids[n_tokens++] = tok->bos_token_id;
        }

        /* Very naive longest-prefix match tokenizer (not true BPE, but works decently) */
        int pos = 0;
        while (pos < len) {
            int best_id = -1;
            int best_len = 0;

            for (int i = 0; i < tok->vocab_size; i++) {
                char* v = tok->vocab[i];
                if (!v) continue;
                int vlen = (int)strlen(v);
                if (vlen == 0) continue;

                /* Handle special whitespace prefixes */
                int match = 1;
                int t_pos = pos;
                for (int k = 0; k < vlen; k++) {
                    if (t_pos >= len) { match = 0; break; }
                    char c = v[k];
                    /* SentencePiece U+2581 */
                    if ((unsigned char)c == 0xe2 && k+2 < vlen && (unsigned char)v[k+1] == 0x96 && (unsigned char)v[k+2] == 0x81) {
                        if (text[t_pos] != ' ') { match = 0; break; }
                        k += 2;
                    }
                    /* LLaMA 3 BPE Ġ */
                    else if ((unsigned char)c == 0xc4 && k+1 < vlen && (unsigned char)v[k+1] == 0xa0) {
                        if (text[t_pos] != ' ') { match = 0; break; }
                        k += 1;
                    }
                    else if (c != text[t_pos]) {
                        match = 0; break;
                    }
                    t_pos++;
                }
                if (match && (t_pos - pos) > best_len) {
                    best_len = t_pos - pos;
                    best_id = i;
                }
            }

            if (best_id != -1) {
                ids[n_tokens++] = best_id;
                pos += best_len;
            } else {
                /* Fallback to unknown byte */
                pos++;
            }
        }

        *out_len = n_tokens;
        return ids;
    }



int* tokenize(Tokenizer* tok, const char* text, int* out_len) {
        return tokenize_impl(tok, text, 1, out_len);
}

    int* tokenize_no_bos(Tokenizer* tok, const char* text, int* out_len) {
        return tokenize_impl(tok, text, 0, out_len);
    }

const char* detokenize(Tokenizer* tok, int token_id) {
    if (token_id < 0 || token_id >= tok->vocab_size) return "";
    char* s = tok->vocab[token_id];
    if (!s) return "";

    /* Handle spaces:
     * - SentencePiece uses U+2581 ( ) [0xe2, 0x96, 0x81]
     * - LLaMA 3 BPE uses 'Ġ' (U+0120) [0xc4, 0xa0]
     */
    static char buf[256];
    int j = 0;
    for (int i = 0; s[i] != '\0' && j < 255; i++) {
        /* Old LLaMA SentencePiece */
        if ((unsigned char)s[i] == 0xe2 && (unsigned char)s[i+1] == 0x96 && (unsigned char)s[i+2] == 0x81) {
            buf[j++] = ' ';
            i += 2;
        } 
        /* LLaMA 3 BPE Byte Decoder */
        else if ((unsigned char)s[i] == 0xc4) {
            /* 0xC4 maps to base offsets for bytes 0-63 */
            buf[j++] = (char)((unsigned char)s[i+1] - 0x80);
            i += 1;
        } 
        else if ((unsigned char)s[i] == 0xc5) {
            /* 0xC5 maps to base offsets for bytes 64-127 */
            buf[j++] = (char)((unsigned char)s[i+1] - 0x80 + 64);
            i += 1;
        }
        else if ((unsigned char)s[i] == 0xc3) {
            /* 0xC3 maps to extended control mappings */
            buf[j++] = (char)((unsigned char)s[i+1] + 64);
            i += 1;
        }
        /* Fallback standard ASCII/UTF-8 passthrough */
        else {
            buf[j++] = s[i];
        }
    }
    buf[j] = '\0';
    return buf;
}

void tokenizer_destroy(Tokenizer* tok) {
    free(tok);
}

/* ---- Sampler ---- */

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
    if (temperature <= 0.0f) return sample_greedy(logits, vocab_size);

    /* Compute exp(logit/temp) safely */
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    float sum = 0.0f;
    float* probs = (float*)malloc(vocab_size * sizeof(float));
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_val) / temperature);
        sum += probs[i];
    }

    float r = ((float)rand() / (float)RAND_MAX) * sum;
    float acc = 0.0f;
    int picked = vocab_size - 1;
    for (int i = 0; i < vocab_size; i++) {
        acc += probs[i];
        if (r <= acc) {
            picked = i;
            break;
        }
    }
    
    free(probs);
    return picked;
}

int sample_top_k(const float* logits, int vocab_size, int k, float temperature) {
    if (vocab_size <= 0) return 0;
    if (temperature <= 0.0f) return sample_greedy(logits, vocab_size);
    if (k <= 0 || k >= vocab_size) return sample_temperature(logits, vocab_size, temperature);

    /* Track top-k logits without sorting the entire vocabulary. */
    int* top_idx = (int*)malloc((size_t)k * sizeof(int));
    float* top_val = (float*)malloc((size_t)k * sizeof(float));
    if (!top_idx || !top_val) {
        free(top_idx);
        free(top_val);
        return sample_temperature(logits, vocab_size, temperature);
    }

    int count = 0;
    for (int i = 0; i < vocab_size; i++) {
        float v = logits[i];
        if (count < k) {
            top_idx[count] = i;
            top_val[count] = v;
            count++;
            continue;
        }

        int min_j = 0;
        float min_v = top_val[0];
        for (int j = 1; j < k; j++) {
            if (top_val[j] < min_v) { min_v = top_val[j]; min_j = j; }
        }
        if (v > min_v) {
            top_idx[min_j] = i;
            top_val[min_j] = v;
        }
    }

    /* Sample from the top-k set using temperature-softmax. */
    float max_val = top_val[0];
    for (int j = 1; j < k; j++) if (top_val[j] > max_val) max_val = top_val[j];

    float sum = 0.0f;
    float* probs = (float*)malloc((size_t)k * sizeof(float));
    if (!probs) {
        free(top_idx);
        free(top_val);
        return sample_temperature(logits, vocab_size, temperature);
    }

    for (int j = 0; j < k; j++) {
        probs[j] = expf((top_val[j] - max_val) / temperature);
        sum += probs[j];
    }

    float r = ((float)rand() / (float)RAND_MAX) * sum;
    float acc = 0.0f;
    int picked_j = k - 1;
    for (int j = 0; j < k; j++) {
        acc += probs[j];
        if (r <= acc) { picked_j = j; break; }
    }

    int picked = top_idx[picked_j];

    free(probs);
    free(top_idx);
    free(top_val);
    return picked;
}

int sample_top_p(const float* logits, int vocab_size, float p, float temperature) {
    if (vocab_size <= 0) return 0;
    if (temperature <= 0.0f) return sample_greedy(logits, vocab_size);
    if (p >= 1.0f) return sample_temperature(logits, vocab_size, temperature);
    if (p <= 0.0f) return sample_greedy(logits, vocab_size);

    LogitIndex* arr = (LogitIndex*)malloc((size_t)vocab_size * sizeof(LogitIndex));
    if (!arr) return sample_temperature(logits, vocab_size, temperature);
    for (int i = 0; i < vocab_size; i++) {
        arr[i].idx = i;
        arr[i].logit = logits[i];
    }

    qsort(arr, (size_t)vocab_size, sizeof(LogitIndex), logit_index_cmp_desc);

    float max_val = arr[0].logit;
    double sum_total = 0.0;
    for (int i = 0; i < vocab_size; i++) {
        sum_total += (double)expf((arr[i].logit - max_val) / temperature);
    }

    double cum = 0.0;
    int cutoff = 0;
    for (int i = 0; i < vocab_size; i++) {
        double w = (double)expf((arr[i].logit - max_val) / temperature);
        cum += w / sum_total;
        cutoff++;
        if (cum >= (double)p) break;
    }
    if (cutoff < 1) cutoff = 1;

    double sum_cut = 0.0;
    for (int i = 0; i < cutoff; i++) {
        sum_cut += (double)expf((arr[i].logit - max_val) / temperature);
    }

    double r = ((double)rand() / (double)RAND_MAX) * sum_cut;
    double acc = 0.0;
    int picked_i = cutoff - 1;
    for (int i = 0; i < cutoff; i++) {
        acc += (double)expf((arr[i].logit - max_val) / temperature);
        if (r <= acc) { picked_i = i; break; }
    }

    int picked = arr[picked_i].idx;
    free(arr);
    return picked;
}

/* ---- CLI ---- */

static DeviceKind parse_device_kind(const char* s) {
    if (!s) return DEVICE_AUTO;
    if (strcmp(s, "auto") == 0) return DEVICE_AUTO;
    if (strcmp(s, "cpu") == 0) return DEVICE_CPU;
    if (strcmp(s, "cuda") == 0) return DEVICE_CUDA;
    return DEVICE_AUTO;
}

int cli_parse(int argc, char** argv, CLIArgs* args) {
    args->model_path  = NULL;
    args->prompt      = "Hello";
    args->max_tokens  = 128;
    args->temperature = 0.7f;
    args->top_k       = 40;
    args->top_p       = 0.9f;
    args->ctx_len     = 0;
    args->n_gpu_layers = 0;
    args->device      = DEVICE_AUTO;
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    args->threads     = (cores > 1) ? (int)(cores <= 8 ? cores : 8) : 1;
    args->chat        = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --model <path>        Path to GGUF model file\n");
            printf("  --prompt <text>       Input prompt\n");
            printf("  --max-tokens <N>      Max tokens to generate (default: 128)\n");
            printf("  --ctx <N>             Context buffer size limit (default: min(model, 8192))\n");
            printf("  --temperature <float> Sampling temperature (default: 0.7)\n");
            printf("  --top-k <N>           Top-k sampling (default: 40; 0 disables)\n");
            printf("  --top-p <p>           Top-p sampling (default: 0.9; 1 disables)\n");
            printf("  --device <kind>       Device: auto|cpu|cuda (default: auto)\n");
            printf("  --n-gpu-layers <N>    Number of layers to offload to GPU\n");
            printf("  --threads <N>         CPU threads (default: 1)\n");
            printf("  --chat                Chat REPL mode\n");
            return 1;
        }
        else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) args->model_path = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) args->prompt = argv[++i];
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) args->max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n-gpu-layers") == 0 && i + 1 < argc) args->n_gpu_layers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) args->ctx_len = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) args->temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) args->top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) args->top_p = (float)atof(argv[++i]);
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
        /* Chat mode is implemented in src/engine/chat.c; parsing should
         * succeed and main.c will route to chat_repl(). */
    }

    if (args->device == DEVICE_CUDA) {
#if !defined(USE_CUDA) || (USE_CUDA == 0)
        fprintf(stderr, "Error: --device cuda requested but CUDA is not enabled in this build (try: make USE_CUDA=1)\n");
        return -1;
#else
        /* Auto-detect GPU layers: if user didn't set --n-gpu-layers, default to
         * 20 to accommodate KV cache in VRAM. */
        if (args->n_gpu_layers == 0) {
            printf("CUDA device detected. Defaulting n_gpu_layers=20 to accommodate KV cache in VRAM.\n");
            args->n_gpu_layers = 20;
            fprintf(stderr, "[auto] --n-gpu-layers not set, defaulting to %d for CUDA device\n", args->n_gpu_layers);
        }
#endif
    }

    if (!args->model_path) {
        fprintf(stderr, "Error: --model is required\n");
        return -1;
    }
    return 0;
}
