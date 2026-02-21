#ifndef LLMRT_ENGINE_H
#define LLMRT_ENGINE_H

#include "types.h"

/*============================================================================
 * Engine & IO API — Animesh owns the implementation
 * 
 * Model loading (GGUF), transformer forward pass, and generation loop.
 *============================================================================*/

/* ---- Model Loader ---- */

/* Load a GGUF model file. Parses header, validates tensors, mmaps weights.
 * Uses `arena` for metadata allocations (or malloc if arena is NULL).
 * Returns NULL on failure. */
ModelWeights* load_model(const char* path, Arena* arena);

/* Free model weights (unmap, close file, etc.) */
void free_model(ModelWeights* model);

/* ---- Forward Pass ---- */

/* Run one transformer layer.
 * hidden:  input hidden state (T_q, H) — modified in-place with residual
 * weights: this layer's weight tensors
 * kv:      KV cache handle
 * layer:   layer index (0-based)
 * pos:     token position (for decode: position of the single new token)
 * scr:     scratch allocator for temporaries
 * cfg:     model config */
void transformer_layer(Tensor* hidden, LayerWeights* weights, KVCache* kv,
                       int layer, int pos, Scratch* scr, ModelConfig* cfg);

/* Run full forward pass: embed → N layers → final norm → logits.
 * token_ids: array of token IDs to process
 * n_tokens:  number of tokens (1 for decode, T for prefill)
 * pos:       starting position
 * Returns:   logits tensor (1, vocab_size) for the last token.
 *            Allocated in scratch — valid until scratch_reset(). */
Tensor* forward(ModelWeights* model, KVCache* kv, Scratch* scr,
                int* token_ids, int n_tokens, int pos);

/* ---- Generation ---- */

/* Full generation loop: tokenize → prefill → decode loop → print output.
 * model_path:  path to GGUF file
 * prompt:      input text
 * max_tokens:  maximum tokens to generate
 * temperature: sampling temperature (0 = greedy)
 * top_k:       top-k sampling parameter (0 = disabled)
 * top_p:       top-p sampling parameter (1.0 = disabled) */
void generate(const char* model_path, const char* prompt, int max_tokens,
              float temperature, int top_k, float top_p);

#endif /* LLMRT_ENGINE_H */
