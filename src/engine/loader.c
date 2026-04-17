/*============================================================================
 * GGUF Model Loader
 * 
 * Parses the GGUF binary format, mmaps weight data, and builds a
 * ModelWeights struct for use by the forward pass.
 *
 * GGUF v3 binary layout (little-endian):
 *   [Header]   magic(4) | version(u32) | tensor_count(u64) | kv_count(u64)
 *   [Metadata] kv_count × { key_string, value_type(u32), value }
 *   [TensorInfo] tensor_count × { name_string, ndims(u32), dims[ndims](u64),
 *                                  ggml_type(u32), offset(u64) }
 *   [Padding]  align to `general.alignment` (default 32)
 *   [TensorData] contiguous block of all tensor data
 *============================================================================*/

#define _GNU_SOURCE
#include "engine.h"
#include "memory.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

/* ---------- GGUF constants ---------- */

#define GGUF_MAGIC 0x46554747  /* "GGUF" in little-endian */
#define GGUF_DEFAULT_ALIGNMENT 32

/* GGUF metadata value types */
enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

/* ggml_type values (from ggml.h) */
enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
};

/* Block sizes for quantized types (elements per block) */
static int ggml_block_size(int ggml_type) {
    switch (ggml_type) {
        case GGML_TYPE_F32:  return 1;
        case GGML_TYPE_F16:  return 1;
        case GGML_TYPE_Q4_0: return 32;
        case GGML_TYPE_Q4_1: return 32;
        case GGML_TYPE_Q5_0: return 32;
        case GGML_TYPE_Q5_1: return 32;
        case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_Q8_1: return 32;
        case GGML_TYPE_Q2_K: return 256;
        case GGML_TYPE_Q3_K: return 256;
        case GGML_TYPE_Q4_K: return 256;
        case GGML_TYPE_Q5_K: return 256;
        case GGML_TYPE_Q6_K: return 256;
        default: return 1;
    }
}

/* Bytes per block for quantized types */
static size_t ggml_type_size(int ggml_type) {
    switch (ggml_type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_Q4_0: return 18;   /* 32 × 4bit + 1 × f16 scale = 16 + 2 */
        case GGML_TYPE_Q4_1: return 20;   /* 32 × 4bit + f16 scale + f16 min = 16 + 2 + 2 */
        case GGML_TYPE_Q5_0: return 22;   /* 32 × 5bit + 4 byte bits + 2 scale */
        case GGML_TYPE_Q5_1: return 24;
        case GGML_TYPE_Q8_0: return 34;   /* 32 × 8bit + f16 scale = 32 + 2 */
        case GGML_TYPE_Q8_1: return 36;
        case GGML_TYPE_Q2_K: return 84;
        case GGML_TYPE_Q3_K: return 110;
        case GGML_TYPE_Q4_K: return 144;
        case GGML_TYPE_Q5_K: return 176;
        case GGML_TYPE_Q6_K: return 210;
        case GGML_TYPE_BF16: return 2;
        default: return 0;
    }
}

static size_t ggml_tensor_nbytes(uint32_t ggml_type, const uint64_t* dims, int ndims) {
    if (ndims <= 0) return 0;
    int blck = ggml_block_size((int)ggml_type);
    size_t tsize = ggml_type_size((int)ggml_type);
    if (blck <= 0 || tsize == 0) return 0;

    /* GGUF/ggml dims: dims[0] is the fastest-changing dimension (ne0).
     * Quantization blocks are along ne0. */
    uint64_t ne0 = dims[0];
    uint64_t blocks0 = (ne0 + (uint64_t)blck - 1ULL) / (uint64_t)blck;
    size_t row_bytes = (size_t)blocks0 * tsize;

    uint64_t rows = 1;
    for (int d = 1; d < ndims; d++) {
        rows *= dims[d];
    }
    return row_bytes * (size_t)rows;
}

/* Convert ggml_type to our DataType enum */
static DataType ggml_to_dtype(int ggml_type) {
    switch (ggml_type) {
        case GGML_TYPE_F32:  return DTYPE_F32;
        case GGML_TYPE_F16:  return DTYPE_F16;
        case GGML_TYPE_Q8_0: return DTYPE_Q8_0;
        case GGML_TYPE_Q4_0: return DTYPE_Q4_0;
        case GGML_TYPE_Q4_K: return DTYPE_Q4_K;
        default:             return DTYPE_UNKNOWN;
    }
}

/* ---------- Buffered reader for parsing ---------- */

typedef struct {
    const uint8_t* data;     /* mmap'd file content */
    size_t         size;     /* total file size */
    size_t         pos;      /* current read position */
} Reader;

static int reader_check(Reader* r, size_t bytes) {
    return (r->pos + bytes <= r->size);
}

static uint8_t read_u8(Reader* r) {
    uint8_t v = r->data[r->pos];
    r->pos += 1;
    return v;
}

static uint32_t read_u32(Reader* r) {
    uint32_t v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static uint64_t read_u64(Reader* r) {
    uint64_t v;
    memcpy(&v, r->data + r->pos, 8);
    r->pos += 8;
    return v;
}

static int32_t read_i32(Reader* r) {
    int32_t v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static float read_f32(Reader* r) {
    float v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

/* GGUF strings: u64 length + raw bytes (NOT null-terminated in file) */
typedef struct {
    char*  str;
    size_t len;
} GGUFString;

static GGUFString read_gguf_string(Reader* r) {
    GGUFString s;
    s.len = (size_t)read_u64(r);
    s.str = (char*)malloc(s.len + 1);
    memcpy(s.str, r->data + r->pos, s.len);
    s.str[s.len] = '\0';
    r->pos += s.len;
    return s;
}

/* Skip a metadata value (we don't need most metadata) */
static void skip_metadata_value(Reader* r, uint32_t vtype) {
    switch (vtype) {
        case GGUF_TYPE_UINT8:   r->pos += 1; break;
        case GGUF_TYPE_INT8:    r->pos += 1; break;
        case GGUF_TYPE_UINT16:  r->pos += 2; break;
        case GGUF_TYPE_INT16:   r->pos += 2; break;
        case GGUF_TYPE_UINT32:  r->pos += 4; break;
        case GGUF_TYPE_INT32:   r->pos += 4; break;
        case GGUF_TYPE_FLOAT32: r->pos += 4; break;
        case GGUF_TYPE_BOOL:    r->pos += 1; break;
        case GGUF_TYPE_UINT64:  r->pos += 8; break;
        case GGUF_TYPE_INT64:   r->pos += 8; break;
        case GGUF_TYPE_FLOAT64: r->pos += 8; break;
        case GGUF_TYPE_STRING: {
            GGUFString s = read_gguf_string(r);
            free(s.str);
            break;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t arr_type = read_u32(r);
            uint64_t arr_len  = read_u64(r);
            for (uint64_t i = 0; i < arr_len; i++) {
                skip_metadata_value(r, arr_type);
            }
            break;
        }
        default:
            fprintf(stderr, "Warning: unknown metadata type %u\n", vtype);
            break;
    }
}

/* ---------- Tensor info parsed from file ---------- */

typedef struct {
    char*    name;
    int      ndims;
    uint64_t dims[MAX_DIMS];
    uint32_t ggml_type;
    uint64_t offset;     /* offset from start of tensor data block */
} TensorInfo;

/* ---------- Main loader ---------- */

/* Internal state for a loaded model */
typedef struct {
    int         fd;
    void*       mmap_base;
    size_t      mmap_size;
} LoaderState;

static LoaderState* g_loader_state = NULL;  /* single model for now */

ModelWeights* load_model(const char* path, Arena* arena) {
    /* ---- Open and mmap the entire file ---- */
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "load_model: cannot open '%s': %s\n", path, strerror(errno));
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        fprintf(stderr, "load_model: fstat failed: %s\n", strerror(errno));
        close(fd);
        return NULL;
    }
    size_t file_size = (size_t)st.st_size;

    void* file_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file_data == MAP_FAILED) {
        fprintf(stderr, "load_model: mmap failed: %s\n", strerror(errno));
        close(fd);
        return NULL;
    }

    /* Advise sequential access for initial parse */
    madvise(file_data, file_size, MADV_SEQUENTIAL);

    Reader r = { .data = (const uint8_t*)file_data, .size = file_size, .pos = 0 };

    /* ---- Parse header ---- */
    if (!reader_check(&r, 4 + 4 + 8 + 8)) {
        fprintf(stderr, "load_model: file too small for GGUF header\n");
        goto fail;
    }

    uint32_t magic = read_u32(&r);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "load_model: bad magic 0x%08X (expected 0x%08X)\n", magic, GGUF_MAGIC);
        goto fail;
    }

    uint32_t version      = read_u32(&r);
    uint64_t tensor_count = read_u64(&r);
    uint64_t kv_count     = read_u64(&r);

    printf("GGUF v%u: %lu tensors, %lu metadata KVs\n",
           version, (unsigned long)tensor_count, (unsigned long)kv_count);

    if (version < 2 || version > 3) {
        fprintf(stderr, "load_model: unsupported GGUF version %u\n", version);
        goto fail;
    }

    /* ---- Parse metadata — extract model config values ---- */
    ModelConfig cfg = {0};
    /* Preserve prior runtime behavior if GGUF does not specify these. */
    cfg.bos_token_id = 1;
    cfg.eos_token_id = 2;
    uint32_t alignment = GGUF_DEFAULT_ALIGNMENT;

    for (uint64_t i = 0; i < kv_count; i++) {
        GGUFString key = read_gguf_string(&r);
        uint32_t vtype  = read_u32(&r);

        /* Extract the config values we need */
        int matched = 0;

        if (vtype == GGUF_TYPE_UINT32 || vtype == GGUF_TYPE_INT32) {
            int32_t val = read_i32(&r);
            matched = 1;

            if (strcmp(key.str, "llama.embedding_length") == 0)
                cfg.hidden_dim = val;
            else if (strcmp(key.str, "llama.attention.head_count") == 0)
                cfg.n_heads = val;
            else if (strcmp(key.str, "llama.attention.head_count_kv") == 0)
                cfg.n_kv_heads = val;
            else if (strcmp(key.str, "llama.block_count") == 0)
                cfg.n_layers = val;
            else if (strcmp(key.str, "llama.feed_forward_length") == 0)
                cfg.ff_dim = val;
            else if (strcmp(key.str, "llama.context_length") == 0)
                cfg.max_seq_len = val;
            else if (strcmp(key.str, "general.alignment") == 0)
                alignment = (uint32_t)val;
            else if (strcmp(key.str, "tokenizer.ggml.bos_token_id") == 0)
                cfg.bos_token_id = val;
            else if (strcmp(key.str, "tokenizer.ggml.eos_token_id") == 0)
                cfg.eos_token_id = val;
            /* else: skip — we don't need this KV */
        }
        else if (vtype == GGUF_TYPE_ARRAY && strcmp(key.str, "tokenizer.ggml.tokens") == 0) {
            uint32_t arr_type = read_u32(&r);
            uint64_t arr_len  = read_u64(&r);
            if (arr_type == GGUF_TYPE_STRING) {
                cfg.vocab_size = (int)arr_len;
                cfg.vocab_strings = (char**)calloc(arr_len, sizeof(char*));
                for (uint64_t v = 0; v < arr_len; v++) {
                    GGUFString s = read_gguf_string(&r);
                    cfg.vocab_strings[v] = s.str; /* keep allocated string */
                }
            } else {
                for (uint64_t v = 0; v < arr_len; v++) skip_metadata_value(&r, arr_type);
            }
            matched = 1;
        }
        else if (vtype == GGUF_TYPE_ARRAY && strcmp(key.str, "tokenizer.ggml.scores") == 0) {
            uint32_t arr_type = read_u32(&r);
            uint64_t arr_len  = read_u64(&r);
            if (arr_type == GGUF_TYPE_FLOAT32) {
                cfg.vocab_scores = (float*)calloc(arr_len, sizeof(float));
                for (uint64_t v = 0; v < arr_len; v++) {
                    cfg.vocab_scores[v] = read_f32(&r);
                }
            } else {
                for (uint64_t v = 0; v < arr_len; v++) skip_metadata_value(&r, arr_type);
            }
            matched = 1;
        }

        if (!matched) {
            /* Read as u64 for matching against uint64 keys, or skip */
            if (vtype == GGUF_TYPE_UINT64 || vtype == GGUF_TYPE_INT64) {
                uint64_t val = read_u64(&r);
                if (strcmp(key.str, "llama.embedding_length") == 0)
                    cfg.hidden_dim = (int)val;
                else if (strcmp(key.str, "llama.attention.head_count") == 0)
                    cfg.n_heads = (int)val;
                else if (strcmp(key.str, "llama.attention.head_count_kv") == 0)
                    cfg.n_kv_heads = (int)val;
                else if (strcmp(key.str, "llama.block_count") == 0)
                    cfg.n_layers = (int)val;
                else if (strcmp(key.str, "llama.feed_forward_length") == 0)
                    cfg.ff_dim = (int)val;
                else if (strcmp(key.str, "llama.context_length") == 0)
                    cfg.max_seq_len = (int)val;
                else if (strcmp(key.str, "tokenizer.ggml.bos_token_id") == 0)
                    cfg.bos_token_id = (int)val;
                else if (strcmp(key.str, "tokenizer.ggml.eos_token_id") == 0)
                    cfg.eos_token_id = (int)val;
            } else {
                skip_metadata_value(&r, vtype);
            }
        }

        free(key.str);
    }

    /* Derive head_dim and vocab_size (vocab_size from token_embedding shape) */
    if (cfg.n_heads > 0) {
        cfg.head_dim = cfg.hidden_dim / cfg.n_heads;
    }

    printf("Model config: H=%d, heads=%d, kv_heads=%d, head_dim=%d, layers=%d, ff=%d, ctx=%d, bos=%d, eos=%d\n",
           cfg.hidden_dim, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
           cfg.n_layers, cfg.ff_dim, cfg.max_seq_len, cfg.bos_token_id, cfg.eos_token_id);

    /* ---- Parse tensor info ---- */
    TensorInfo* tensor_infos = (TensorInfo*)calloc(tensor_count, sizeof(TensorInfo));
    if (!tensor_infos) goto fail;

    for (uint64_t i = 0; i < tensor_count; i++) {
        GGUFString name = read_gguf_string(&r);
        tensor_infos[i].name = name.str;  /* take ownership */

        tensor_infos[i].ndims = (int)read_u32(&r);
        for (int d = 0; d < tensor_infos[i].ndims; d++) {
            tensor_infos[i].dims[d] = read_u64(&r);
        }

        tensor_infos[i].ggml_type = read_u32(&r);
        tensor_infos[i].offset    = read_u64(&r);
    }

    /* ---- Compute tensor data start ---- */
    /* Tensor data begins at the current position, aligned to `alignment` */
    size_t data_start = r.pos;
    data_start = (data_start + alignment - 1) & ~((size_t)alignment - 1);

    printf("Tensor data starts at offset 0x%lx (alignment=%u)\n",
           (unsigned long)data_start, alignment);

    /* Switch madvise to random access for weight reads */
    madvise(file_data, file_size, MADV_RANDOM);

    /* ---- Build ModelWeights ---- */
    ModelWeights* model = (ModelWeights*)calloc(1, sizeof(ModelWeights));
    if (!model) goto fail_tensors;

    model->config = cfg;
    model->layers = (LayerWeights*)calloc(cfg.n_layers, sizeof(LayerWeights));
    if (!model->layers) goto fail_model;

    /* Helper: create a Tensor from a TensorInfo, pointing into mmap'd data */
    #define MAKE_TENSOR(ti) _make_tensor(&(ti), (uint8_t*)file_data + data_start)

    /* We also allocate Tensor structs — use malloc for now (arena later) */
    (void)arena;

    for (uint64_t i = 0; i < tensor_count; i++) {
        TensorInfo* ti = &tensor_infos[i];

        /* Compute the pointer into mmap'd data */
        void* tdata = (uint8_t*)file_data + data_start + ti->offset;

        size_t tbytes = ggml_tensor_nbytes(ti->ggml_type, ti->dims, ti->ndims);
        if (tbytes == 0) {
            fprintf(stderr, "load_model: unsupported ggml_type=%u for tensor '%s'\n",
                ti->ggml_type, ti->name);
            goto fail_model;
        }
        if (data_start + (size_t)ti->offset + tbytes > file_size) {
            fprintf(stderr, "load_model: tensor '%s' overruns file (offset=%lu, bytes=%zu)\n",
                ti->name, (unsigned long)ti->offset, tbytes);
            goto fail_model;
        }

        /* Allocate a Tensor struct */
        Tensor* t = (Tensor*)calloc(1, sizeof(Tensor));
        t->data = tdata;
        t->dtype = ggml_to_dtype(ti->ggml_type);
        t->ndim = ti->ndims;
        t->byte_size = tbytes;

        if (t->dtype == DTYPE_UNKNOWN) {
            fprintf(stderr, "load_model: unsupported dtype for tensor '%s' (ggml_type=%u)\n",
                ti->name, ti->ggml_type);
            free(t);
            goto fail_model;
        }

        /* GGUF dimensions are stored in ggml order (row-major, first dim varies fastest).
         * For a weight matrix (out_features, in_features) in PyTorch:
         *   gguf dims[0] = in_features (columns)
         *   gguf dims[1] = out_features (rows)
         * We store as shape[0] = rows, shape[1] = cols for our row-major convention. */
        if (ti->ndims == 1) {
            t->shape[0] = (int)ti->dims[0];
            t->stride[0] = 1;
        } else if (ti->ndims == 2) {
            t->shape[0] = (int)ti->dims[1];  /* rows = out_features */
            t->shape[1] = (int)ti->dims[0];  /* cols = in_features */
            t->stride[0] = (int)ti->dims[0];
            t->stride[1] = 1;
        } else {
            /* Higher dims — store as-is (rare for LLM weights) */
            for (int d = 0; d < ti->ndims; d++) {
                t->shape[d] = (int)ti->dims[ti->ndims - 1 - d];
            }
            /* Compute strides */
            t->stride[ti->ndims - 1] = 1;
            for (int d = ti->ndims - 2; d >= 0; d--) {
                t->stride[d] = t->stride[d+1] * t->shape[d+1];
            }
        }

        /* ---- Map tensor name to ModelWeights field ---- */
        const char* name = ti->name;

        /* Global tensors */
        if (strcmp(name, "token_embd.weight") == 0) {
            model->embedding = t;
            /* Extract vocab_size from embedding shape */
            cfg.vocab_size = t->shape[0];
            model->config.vocab_size = t->shape[0];
        }
        else if (strcmp(name, "output_norm.weight") == 0) {
            model->rms_final = t;
        }
        else if (strcmp(name, "output.weight") == 0) {
            model->lm_head = t;
        }
        /* Per-layer tensors: blk.{layer}.{component}.weight */
        else if (strncmp(name, "blk.", 4) == 0) {
            /* Parse layer number */
            int layer = 0;
            const char* p = name + 4;
            while (*p >= '0' && *p <= '9') {
                layer = layer * 10 + (*p - '0');
                p++;
            }
            if (*p != '.') { free(t); continue; }
            p++;  /* skip '.' */

            if (layer < 0 || layer >= cfg.n_layers) { free(t); continue; }

            LayerWeights* lw = &model->layers[layer];

            if (strcmp(p, "attn_q.weight") == 0)      lw->wq = t;
            else if (strcmp(p, "attn_k.weight") == 0)  lw->wk = t;
            else if (strcmp(p, "attn_v.weight") == 0)  lw->wv = t;
            else if (strcmp(p, "attn_output.weight") == 0) lw->wo = t;
            else if (strcmp(p, "ffn_gate.weight") == 0) lw->w_gate = t;
            else if (strcmp(p, "ffn_up.weight") == 0)   lw->w_up = t;
            else if (strcmp(p, "ffn_down.weight") == 0) lw->w_down = t;
            else if (strcmp(p, "attn_norm.weight") == 0) lw->rms_att = t;
            else if (strcmp(p, "ffn_norm.weight") == 0)  lw->rms_ffn = t;
            else {
                /* Unknown tensor — skip */
                free(t);
            }
        }
        else {
            /* Unknown global tensor — skip */
            free(t);
        }
    }

    /* If no separate lm_head, it's tied with embedding */
    if (!model->lm_head && model->embedding) {
        model->lm_head = model->embedding;
        printf("Note: output.weight not found, using tied embeddings\n");
    }

    /* Print validation summary */
    printf("Loaded %lu tensors. vocab_size=%d\n",
           (unsigned long)tensor_count, model->config.vocab_size);

    /* Validate that all layers have all required weights */
    int valid = 1;
    for (int l = 0; l < cfg.n_layers; l++) {
        LayerWeights* lw = &model->layers[l];
        if (!lw->wq || !lw->wk || !lw->wv || !lw->wo) {
            fprintf(stderr, "Warning: layer %d missing attention weights\n", l);
            valid = 0;
        }
        if ((lw->wq && lw->wq->dtype != DTYPE_F32) ||
            (lw->wk && lw->wk->dtype != DTYPE_F32) ||
            (lw->wv && lw->wv->dtype != DTYPE_F32) ||
            (lw->wo && lw->wo->dtype != DTYPE_F32)) {
            fprintf(stderr, "Error: layer %d has non-F32 attention weights (not supported yet)\n", l);
            valid = 0;
        }
        if (!lw->w_gate || !lw->w_up || !lw->w_down) {
            fprintf(stderr, "Warning: layer %d missing MLP weights\n", l);
            valid = 0;
        }
        if ((lw->w_gate && lw->w_gate->dtype != DTYPE_F32) ||
            (lw->w_up && lw->w_up->dtype != DTYPE_F32) ||
            (lw->w_down && lw->w_down->dtype != DTYPE_F32)) {
            fprintf(stderr, "Error: layer %d has non-F32 MLP weights (not supported yet)\n", l);
            valid = 0;
        }
        if (!lw->rms_att || !lw->rms_ffn) {
            fprintf(stderr, "Warning: layer %d missing norm weights\n", l);
            valid = 0;
        }
        if ((lw->rms_att && lw->rms_att->dtype != DTYPE_F32) ||
            (lw->rms_ffn && lw->rms_ffn->dtype != DTYPE_F32)) {
            fprintf(stderr, "Error: layer %d has non-F32 norm weights (not supported yet)\n", l);
            valid = 0;
        }
    }
    if (!model->embedding) {
        fprintf(stderr, "Warning: missing embedding weights\n");
        valid = 0;
    }
    if (model->embedding && model->embedding->dtype != DTYPE_F32) {
        fprintf(stderr, "Error: embedding is non-F32 (not supported yet)\n");
        valid = 0;
    }
    if (!model->rms_final) {
        fprintf(stderr, "Warning: missing final norm weights\n");
        valid = 0;
    }
    if (model->rms_final && model->rms_final->dtype != DTYPE_F32) {
        fprintf(stderr, "Error: final norm is non-F32 (not supported yet)\n");
        valid = 0;
    }
    if (model->lm_head && model->lm_head->dtype != DTYPE_F32) {
        fprintf(stderr, "Error: lm_head is non-F32 (not supported yet)\n");
        valid = 0;
    }

    if (!valid) {
        fprintf(stderr, "load_model: model contains unsupported tensor dtypes; this runtime currently supports F32 weights only\n");
        goto fail_model;
    }

    if (valid) {
        printf("All layer weights validated ✓\n");
    }

    /* Save loader state for cleanup */
    g_loader_state = (LoaderState*)malloc(sizeof(LoaderState));
    g_loader_state->fd = fd;
    g_loader_state->mmap_base = file_data;
    g_loader_state->mmap_size = file_size;

    /* Free tensor info names */
    for (uint64_t i = 0; i < tensor_count; i++) {
        free(tensor_infos[i].name);
    }
    free(tensor_infos);

    return model;

fail_model:
    free(model);
fail_tensors:
    for (uint64_t i = 0; i < tensor_count; i++) {
        free(tensor_infos[i].name);
    }
    free(tensor_infos);
fail:
    munmap(file_data, file_size);
    close(fd);
    return NULL;
}

void free_model(ModelWeights* model) {
    if (!model) return;

    /* Free individual Tensor structs (data is in mmap, not alloc'd) */
    if (model->embedding) free(model->embedding);
    if (model->rms_final) free(model->rms_final);
    /* Only free lm_head if it's not tied to embedding */
    if (model->lm_head && model->lm_head != model->embedding)
        free(model->lm_head);

    for (int l = 0; l < model->config.n_layers; l++) {
        LayerWeights* lw = &model->layers[l];
        free(lw->wq);
        free(lw->wk);
        free(lw->wv);
        free(lw->wo);
        free(lw->w_gate);
        free(lw->w_up);
        free(lw->w_down);
        free(lw->rms_att);
        free(lw->rms_ffn);
    }
    free(model->layers);

    if (model->config.vocab_strings) {
        for (int i = 0; i < model->config.vocab_size; i++) {
            free(model->config.vocab_strings[i]);
        }
        free(model->config.vocab_strings);
    }
    if (model->config.vocab_scores) {
        free(model->config.vocab_scores);
    }

    /* Unmap and close file */
    if (g_loader_state) {
        munmap(g_loader_state->mmap_base, g_loader_state->mmap_size);
        close(g_loader_state->fd);
        free(g_loader_state);
        g_loader_state = NULL;
    }

    free(model);
}
