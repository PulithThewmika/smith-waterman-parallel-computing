# Makefile for the OpenMP Smith-Waterman implementation
CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11 -fopenmp
TARGET  = omp_sw

all: $(TARGET)

$(TARGET): omp_sw.c sw_common.h
	$(CC) $(CFLAGS) -o $(TARGET) omp_sw.c

# Quick run with 8 threads
run: $(TARGET)
	OMP_NUM_THREADS=8 ./$(TARGET) 8000 8000 12345

clean:
	rm -f $(TARGET)

.PHONY: all run clean
