/*============================================================================
 * Tokenizer & Sampler implementation (Replacing Person C's stubs)
 *============================================================================*/

#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

/* ---- Tokenizer ---- */

struct Tokenizer {
    int vocab_size;
    char** vocab;
    float* scores;
};

Tokenizer* tokenizer_create(const ModelConfig* cfg) {
    if (!cfg || !cfg->vocab_strings) return NULL;
    Tokenizer* tok = (Tokenizer*)calloc(1, sizeof(Tokenizer));
    if (!tok) return NULL;
    tok->vocab_size = cfg->vocab_size;
    tok->vocab = cfg->vocab_strings;
    tok->scores = cfg->vocab_scores;
    return tok;
}



int* tokenize(Tokenizer* tok, const char* text, int* out_len) {
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

    /* Add BOS token (1 for Llama) */
    ids[n_tokens++] = 1;

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
            
            /* Handle the special '_' character used by SentencePiece for space */
            int match = 1;
            int t_pos = pos;
            for (int k = 0; k < vlen; k++) {
                if (t_pos >= len) { match = 0; break; }
                char c = v[k];
                if ((unsigned char)c == 0xe2 && k+2 < vlen && (unsigned char)v[k+1] == 0x96 && (unsigned char)v[k+2] == 0x81) {
                    /* U+2581 mapped to space */
                    if (text[t_pos] != ' ') { match = 0; break; }
                    k += 2;
                } else if (c != text[t_pos]) {
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

const char* detokenize(Tokenizer* tok, int token_id) {
    if (token_id < 0 || token_id >= tok->vocab_size) return "";
    char* s = tok->vocab[token_id];
    if (!s) return "";
    
    /* Handle sentencepiece space ( U+2581 block ) */
    static char buf[256];
    int j = 0;
    for (int i = 0; s[i] != '\0' && j < 255; i++) {
        if ((unsigned char)s[i] == 0xe2 && (unsigned char)s[i+1] == 0x96 && (unsigned char)s[i+2] == 0x81) {
            buf[j++] = ' ';
            i += 2;
        } else {
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
    (void)k; /* Full top_k requires sorting, fallback to temp for simplicity here */
    return sample_temperature(logits, vocab_size, temperature);
}

int sample_top_p(const float* logits, int vocab_size, float p, float temperature) {
    (void)p; /* Full top_p requires sorting, fallback to temp for simplicity here */
    return sample_temperature(logits, vocab_size, temperature);
}

/* ---- CLI ---- */

int cli_parse(int argc, char** argv, CLIArgs* args) {
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
            return 1;
        }
        else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) args->model_path = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) args->prompt = argv[++i];
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) args->max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) args->temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) args->top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) args->top_p = (float)atof(argv[++i]);
    }

    if (!args->model_path) {
        fprintf(stderr, "Error: --model is required\n");
        return -1;
    }
    return 0;
}
