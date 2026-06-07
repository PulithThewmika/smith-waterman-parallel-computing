# Makefile for the CUDA Smith-Waterman implementation
#
# GPU ARCH: RTX 4060 is Ada Lovelace = compute capability 8.9 -> sm_89.
# If nvcc complains about sm_89 (older CUDA toolkit), drop the -arch line and
# let nvcc pick a default, or use a value your toolkit supports (e.g. sm_86).
NVCC    = nvcc
ARCH    = -arch=sm_89
NVFLAGS = -O2 $(ARCH)
TARGET  = cuda_sw

all: $(TARGET)

$(TARGET): cuda_sw.cu sw_common.h
	$(NVCC) $(NVFLAGS) -o $(TARGET) cuda_sw.cu

# Quick run with 256 threads/block
run: $(TARGET)
	./$(TARGET) 8000 8000 12345 256

clean:
	rm -f $(TARGET)

.PHONY: all run clean
