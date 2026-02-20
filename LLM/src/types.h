#ifndef TYPES_H
#define TYPES_H

// types.h — THE contract. Touch this, everyone recompiles.

typedef enum { F32, F16, Q8_0, Q4_0, Q4_K_M } DataType;

typedef struct {
    void*    data;
    int      shape[4];   // [dim0, dim1, dim2, dim3], unused dims = 1
    int      stride[4];  // in elements, not bytes
    DataType dtype;
    int      ndim;       // 1, 2, 3, or 4
} Tensor;

typedef struct {
    int hidden_dim;      // H (e.g., 2048 for TinyLlama)
    int n_heads;         // number of attention heads
    int n_kv_heads;      // number of KV heads (GQA)
    int head_dim;        // H / n_heads
    int n_layers;        // number of transformer layers
    int vocab_size;      // vocabulary size
    int ff_dim;          // MLP intermediate dim (H_ff)
    int max_seq_len;     // maximum sequence length
} ModelConfig;

typedef struct {
    Tensor* token_embedding;   // (vocab_size × H)
    Tensor* final_norm_weight; // (H,)
    Tensor* output_proj;       // (H × vocab_size) or tied to embedding

    // Per-layer weights [n_layers]:
    Tensor** attn_norm;    // (H,) per layer
    Tensor** wq;           // (H × H) per layer
    Tensor** wk;           // (H × n_kv_heads*head_dim) per layer
    Tensor** wv;           // same as wk
    Tensor** wo;           // (H × H) per layer
    Tensor** mlp_norm;     // (H,) per layer
    Tensor** w_gate;       // (H × H_ff) per layer
    Tensor** w_up;         // (H × H_ff) per layer
    Tensor** w_down;       // (H_ff × H) per layer
} ModelWeights;

// Opaque handles — implementation hidden from other modules
typedef struct Arena Arena;
typedef struct Scratch Scratch;
typedef struct KVCache KVCache;
typedef struct Tokenizer Tokenizer;

#endif // TYPES_H
