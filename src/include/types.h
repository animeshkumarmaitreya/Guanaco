#ifndef LLMRT_TYPES_H
#define LLMRT_TYPES_H

#include <stddef.h>
#include <stdint.h>

/*============================================================================
 * Shared Types — Agreed on Day 0
 * 
 * These are the core data structures that cross module boundaries.
 * Every team member includes this header. DO NOT modify without group consent.
 *============================================================================*/

/* ---------- Data type enum ---------- */

typedef enum {
    DTYPE_F32   = 0,
    DTYPE_F16   = 1,
    DTYPE_Q8_0  = 2,
    DTYPE_Q4_0  = 3,
} DataType;

/* Returns bytes per element for a given dtype (for F32/F16 only; quantized types are block-based) */
static inline size_t dtype_size(DataType dt) {
    switch (dt) {
        case DTYPE_F32:  return 4;
        case DTYPE_F16:  return 2;
        case DTYPE_Q8_0: return 1;  /* approximate */
        case DTYPE_Q4_0: return 1;  /* approximate — real size is block-based */
        default:         return 0;
    }
}

/* ---------- Tensor ---------- */

#define MAX_DIMS 4

typedef struct {
    void*    data;
    int      shape[MAX_DIMS];   /* shape[0] = outermost dimension */
    int      stride[MAX_DIMS];  /* stride in elements (not bytes) */
    int      ndim;              /* number of active dimensions (1-4) */
    DataType dtype;
} Tensor;

/* Convenience: total number of elements */
static inline int tensor_numel(const Tensor* t) {
    int n = 1;
    for (int i = 0; i < t->ndim; i++) {
        n *= t->shape[i];
    }
    return n;
}

/* ---------- Model config ---------- */

typedef struct {
    int hidden_dim;     /* H — e.g. 2048, 4096 */
    int n_heads;        /* number of query attention heads */
    int n_kv_heads;     /* number of key/value heads (GQA) */
    int head_dim;       /* d_k = hidden_dim / n_heads */
    int n_layers;       /* number of transformer layers */
    int vocab_size;     /* vocabulary size */
    int ff_dim;         /* MLP intermediate dimension (H_ff) */
    int max_seq_len;    /* maximum sequence length */
    char** vocab_strings;
    float* vocab_scores;
} ModelConfig;

/* ---------- Per-layer weights ---------- */

typedef struct {
    Tensor* wq;         /* Q projection:     (H, H) */
    Tensor* wk;         /* K projection:     (H, n_kv_heads * head_dim) */
    Tensor* wv;         /* V projection:     (H, n_kv_heads * head_dim) */
    Tensor* wo;         /* Output projection: (H, H) */
    Tensor* w_gate;     /* MLP gate (SwiGLU): (H, H_ff) */
    Tensor* w_up;       /* MLP up:            (H, H_ff) */
    Tensor* w_down;     /* MLP down:          (H_ff, H) */
    Tensor* rms_att;    /* Pre-attention RMSNorm weight: (H,) */
    Tensor* rms_ffn;    /* Pre-MLP RMSNorm weight:       (H,) */
} LayerWeights;

/* ---------- Full model weights ---------- */

typedef struct {
    ModelConfig   config;
    LayerWeights* layers;       /* array of n_layers */
    Tensor*       embedding;    /* token embedding: (vocab_size, H) */
    Tensor*       rms_final;    /* final RMSNorm weight: (H,) */
    Tensor*       lm_head;      /* output projection: (H, vocab_size) */
} ModelWeights;

/* ---------- Opaque handles (defined in their respective .c files) ---------- */

typedef struct Arena   Arena;
typedef struct Scratch Scratch;
typedef struct KVCache KVCache;

#endif /* LLMRT_TYPES_H */
