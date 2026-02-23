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
CFLAGS   = -Wall -Wextra -Wpedantic -std=c11 -O2 -Isrc/include
LDFLAGS  = -lm
DEBUG_FLAGS = -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer
BUILD    = build

# Source files — stubs (replace with real implementations per person)
STUB_KERNELS   = src/stubs/stub_kernels.c
STUB_MEMORY    = src/stubs/stub_memory.c
STUB_TOKENIZER = src/stubs/stub_tokenizer.c
STUB_ENGINE    = src/stubs/stub_engine.c

# Real implementations
REAL_ENGINE    = src/engine/loader.c src/engine/engine.c src/engine/generate.c
# REAL_KERNELS   = src/kernels/kernels.c
# REAL_MEMORY    = src/memory/arena.c src/memory/scratch.c src/memory/kv_cache.c
# REAL_TOKENIZER = src/tokenizer/tokenizer.c src/tokenizer/sampler.c src/tokenizer/cli.c

# Active per module — swap STUB ↔ REAL as each person finishes
KERNELS   = $(STUB_KERNELS)
MEMORY    = $(STUB_MEMORY)
TOKENIZER = $(STUB_TOKENIZER)
ENGINE    = $(REAL_ENGINE)

.PHONY: all clean test test_kernels test_memory test_tokenizer test_engine debug

all: $(BUILD)/llmrt

$(BUILD):
	mkdir -p $(BUILD)

# ---- Main binary ----
$(BUILD)/llmrt: src/main.c $(KERNELS) $(MEMORY) $(TOKENIZER) $(ENGINE) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# ---- Test binaries ----
$(BUILD)/test_kernels: tests/test_kernels.c $(STUB_KERNELS) | $(BUILD)
	$(CC) $(CFLAGS) -DSTUB_MODE=1 -o $@ $^ $(LDFLAGS)

$(BUILD)/test_memory: tests/test_memory.c $(STUB_MEMORY) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD)/test_tokenizer: tests/test_tokenizer.c $(STUB_TOKENIZER) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD)/test_engine: tests/test_engine.c $(REAL_ENGINE) $(STUB_KERNELS) $(STUB_MEMORY) $(STUB_TOKENIZER) | $(BUILD)
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

test: test_kernels test_memory test_tokenizer test_engine
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
