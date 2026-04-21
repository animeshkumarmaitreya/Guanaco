#============================================================================
# LLM Inference Runtime — Makefile
#
# Targets:
#   make              — build main binary (with stubs for non-engine parts)
#   make test_kernels — build + run Person A's tests
#   make test_memory  — build + run Person B's tests
#   make test_tokenizer — build + run Person C's tests
#   make test_engine  — build + run Animesh's engine tests
#   make test         — run all tests
#   make clean        — remove build artifacts
#   make debug        — build with ASAN/UBSAN
#============================================================================

CC       = gcc
CFLAGS   = -Wall -Wextra -Wpedantic -std=c11 -D_GNU_SOURCE -O2 -Isrc/include -mavx2 -mfma
LDFLAGS  = -lm
DEBUG_FLAGS = -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer
BUILD    = build

# Build switches (scaffolding for upcoming work)
USE_CUDA ?= 0
USE_PTHREAD ?= 0

CUDA_SRCS = src/kernels/cuda/matvec_q4k_cuda.cu
CUDA_OBJS = $(patsubst %.cu, $(BUILD)/%.o, $(CUDA_SRCS))

ifeq ($(USE_CUDA),1)
	CFLAGS += -DUSE_CUDA=1
	LDFLAGS += -L/usr/local/cuda/lib64 -lcudart -lnvidia-ml
endif

ifeq ($(USE_PTHREAD),1)
	CFLAGS += -DUSE_PTHREAD=1 -pthread
	LDFLAGS += -pthread
endif

# Source files — stubs (replace with real implementations per person)
STUB_KERNELS   = src/stubs/stub_kernels.c
STUB_MEMORY    = src/stubs/stub_memory.c
STUB_TOKENIZER = src/stubs/stub_tokenizer.c
STUB_ENGINE    = src/stubs/stub_engine.c

# Real implementations
REAL_ENGINE    = src/engine/loader.c src/engine/engine.c src/engine/generate.c src/engine/chat.c
REAL_KERNELS   = src/kernels/cpu/kernels_cpu.c
REAL_MEMORY    = src/memory/arena.c src/memory/scratch.c src/memory/kv_cache.c
REAL_TOKENIZER = src/tokenizer/tokenizer.c

# Quant stubs (Phase B scaffolding)
REAL_QUANT     = src/kernels/cpu/quant_matvec_q8_0.c src/kernels/cpu/quant_matvec_q4_k.c src/kernels/cpu/quant_matvec_q6_k.c

# Backend + threading scaffolding
REAL_BACKEND   = src/backend/backend.c src/backend/cpu_backend.c src/backend/cuda_backend.c
REAL_THREADPOOL = src/threadpool/threadpool.c

# Active per module — swap STUB ↔ REAL as each person finishes
KERNELS   = $(REAL_KERNELS)
MEMORY    = $(REAL_MEMORY)
TOKENIZER = $(REAL_TOKENIZER)
ENGINE    = $(REAL_ENGINE)

.PHONY: all clean test test_kernels test_memory test_tokenizer test_engine test_quant test_e2e_smoke test_backend test_threadpool test_chat_stub test_gpu_prefill_stub debug

all: $(BUILD)/llmrt

$(BUILD):
	mkdir -p $(BUILD)

# ---- CUDA Kernels ----
$(BUILD)/%.o: %.cu | $(BUILD)
	@mkdir -p $(dir $@)
	nvcc -O3 -arch=sm_86 -Isrc/include -c $< -o $@

# ---- Main binary ----
ifeq ($(USE_CUDA),1)
$(BUILD)/llmrt: src/main.c $(KERNELS) $(MEMORY) $(TOKENIZER) $(ENGINE) $(REAL_BACKEND) $(REAL_THREADPOOL) $(REAL_QUANT) $(CUDA_OBJS) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
else
$(BUILD)/llmrt: src/main.c $(KERNELS) $(MEMORY) $(TOKENIZER) $(ENGINE) $(REAL_BACKEND) $(REAL_THREADPOOL) $(REAL_QUANT) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
endif

# ---- Test binaries ----
$(BUILD)/test_kernels: tests/test_kernels.c $(REAL_KERNELS) $(REAL_THREADPOOL) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD)/test_memory: tests/test_memory.c $(REAL_MEMORY) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD)/test_tokenizer: tests/test_tokenizer.c $(REAL_TOKENIZER) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD)/test_engine: tests/test_engine.c $(REAL_ENGINE) $(REAL_KERNELS) $(REAL_THREADPOOL) $(REAL_MEMORY) $(REAL_TOKENIZER) $(REAL_BACKEND) $(REAL_QUANT) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

ifeq ($(USE_CUDA),1)
$(BUILD)/test_quant: tests/test_quant.c $(REAL_QUANT) $(REAL_THREADPOOL) $(REAL_KERNELS) $(CUDA_OBJS) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
else
$(BUILD)/test_quant: tests/test_quant.c $(REAL_QUANT) $(REAL_THREADPOOL) $(REAL_KERNELS) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
endif

$(BUILD)/test_e2e_smoke: tests/test_e2e_smoke.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD)/test_backend: tests/test_backend.c $(REAL_BACKEND) $(REAL_KERNELS) $(REAL_THREADPOOL) $(REAL_QUANT) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD)/test_threadpool: tests/test_threadpool.c $(REAL_THREADPOOL) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD)/test_chat_stub: tests/test_chat_stub.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD)/test_gpu_prefill_stub: tests/test_gpu_prefill_stub.c src/engine/gpu_prefill.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# ---- Run tests ----
test_kernels: $(BUILD)/test_kernels
	@echo "--- Running kernel tests ---"
	@./$(BUILD)/test_kernels

test_memory: $(BUILD)/test_memory
	@echo "--- Running memory tests ---"
	@./$(BUILD)/test_memory

test_tokenizer: $(BUILD)/test_tokenizer
	@echo "--- Running tokenizer/sampler tests ---"
	@./$(BUILD)/test_tokenizer

test_engine: $(BUILD)/test_engine
	@echo "--- Running engine tests ---"
	@./$(BUILD)/test_engine

test_quant: $(BUILD)/test_quant
	@echo "--- Running quantization stub tests ---"
	@./$(BUILD)/test_quant

test_e2e_smoke: $(BUILD)/test_e2e_smoke
	@echo "--- Running E2E smoke stub tests ---"
	@./$(BUILD)/test_e2e_smoke

test_backend: $(BUILD)/test_backend
	@echo "--- Running backend scaffolding tests ---"
	@./$(BUILD)/test_backend

test_threadpool: $(BUILD)/test_threadpool
	@echo "--- Running threadpool scaffolding tests ---"
	@./$(BUILD)/test_threadpool

test_chat_stub: $(BUILD)/test_chat_stub
	@echo "--- Running chat stub tests ---"
	@./$(BUILD)/test_chat_stub

test_gpu_prefill_stub: $(BUILD)/test_gpu_prefill_stub
	@echo "--- Running GPU prefill stub tests ---"
	@./$(BUILD)/test_gpu_prefill_stub

test: test_kernels test_memory test_tokenizer test_engine test_quant test_e2e_smoke test_backend test_threadpool test_chat_stub test_gpu_prefill_stub
	@echo ""
	@echo "=== All test suites complete ==="

# ---- Debug build (ASAN + UBSAN) ----
debug: CFLAGS += $(DEBUG_FLAGS)
debug: LDFLAGS += -fsanitize=address,undefined
debug: clean all
	@echo "Debug build complete (ASAN + UBSAN enabled)"

# ---- Clean ----
clean:
	rm -rf $(BUILD)
