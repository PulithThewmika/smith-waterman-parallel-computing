# Makefile for the MPI Smith-Waterman implementation
MPICC   = mpicc
CFLAGS  = -O2 -Wall -Wextra -std=c11
TARGET  = mpi_sw

all: $(TARGET)

$(TARGET): mpi_sw.c sw_common.h
	$(MPICC) $(CFLAGS) -o $(TARGET) mpi_sw.c

# Quick run with 4 processes
run: $(TARGET)
	mpirun -np 4 ./$(TARGET) 8000 8000 12345

clean:
	rm -f $(TARGET)

.PHONY: all run clean
