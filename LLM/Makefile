CC = gcc
CFLAGS = -Wall -Wextra -O3 -mavx2 -mfma -g -fsanitize=address,undefined -Isrc
LDFLAGS = -lm -fsanitize=address,undefined

SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build

KER_SRCS = $(SRC_DIR)/kernels.c
KER_OBJS = $(BUILD_DIR)/kernels.o

TEST_KER_SRCS = $(TEST_DIR)/test_kernels.c
TEST_KER_OBJS = $(BUILD_DIR)/test_kernels.o

all: test_kernels

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

test_kernels: $(TEST_KER_OBJS) $(KER_OBJS)
	$(CC) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR) test_kernels

.PHONY: all clean
