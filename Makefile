# Transformer-C Makefile 
CC = gcc 
CFLAGS = -Wall -Wextra -O2 -std=c99 -I./include 
TARGET = bin/transformer_c.exe 
 
# Source files 
SRC = $(wildcard src/*.c) 
OBJ = $(patsubst src/.o, $(SRC)) 
 
# Default target 
all: $(TARGET) 
 
# Link executable 
$(TARGET): $(OBJ) 
    @mkdir -p bin 
    $(CC) $(CFLAGS) -o $@ $ 
 
# Compile objects 
obj/ src/.c 
    @mkdir -p obj 
 
# Clean build 
clean: 
    rm -rf obj bin/*.exe 
 
# Run tests 
test: $(TARGET) 
    ./$(TARGET) --test 
 
.PHONY: all clean test 
