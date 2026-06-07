# Top-level convenience Makefile -- builds every implementation.
# Individual versions also have their own Makefile in source/<impl>/.

SRC = source

all: serial openmp mpi cuda

serial:
	$(MAKE) -C $(SRC)/serial

openmp:
	$(MAKE) -C $(SRC)/openmp

mpi:
	$(MAKE) -C $(SRC)/mpi

cuda:
	$(MAKE) -C $(SRC)/cuda

# Build only the CPU versions (handy on machines without CUDA/MPI yet)
cpu: serial openmp

clean:
	-$(MAKE) -C $(SRC)/serial clean
	-$(MAKE) -C $(SRC)/openmp clean
	-$(MAKE) -C $(SRC)/mpi    clean
	-$(MAKE) -C $(SRC)/cuda   clean

.PHONY: all serial openmp mpi cuda cpu clean
