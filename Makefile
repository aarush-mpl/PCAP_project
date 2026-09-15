# -----------------------------------------------------------------------
# Makefile — Scalable Parallel K-Means (PS-5)
#
# Targets:
#   make serial   -> bin/kmeans_serial     (CPU reference baseline)
#   make mpi      -> bin/kmeans_mpi        (Tirth's MPI layer, CPU assignment step)
#   make hybrid   -> bin/kmeans_hybrid     (MPI + Aarush's CUDA kernel; needs
#                                            cuda_assignment_kernel.cu in src/)
#   make clean
# -----------------------------------------------------------------------
CXX      = g++
MPICXX   = mpicxx
NVCC     = nvcc
CXXFLAGS = -O3 -std=c++17 -Wall -Iinclude
NVCCFLAGS = -O3 -std=c++17 -Iinclude

BIN_DIR = bin

.PHONY: all serial mpi hybrid clean dirs

all: serial mpi

dirs:
	mkdir -p $(BIN_DIR)

serial: dirs
	$(CXX) $(CXXFLAGS) src/kmeans_serial.cpp -o $(BIN_DIR)/kmeans_serial

mpi: dirs
	$(MPICXX) $(CXXFLAGS) src/kmeans_mpi.cpp -o $(BIN_DIR)/kmeans_mpi

# Hybrid target left wired up for Aarush's CUDA kernel (src/cuda_assignment_kernel.cu,
# expected to define extern "C" launch_assignment_kernel(...) per the contract
# documented in src/kmeans_mpi.cpp). Not built by `make all` since CUDA/nvcc
# may not be present on every dev machine.
hybrid: dirs
	$(NVCC) $(NVCCFLAGS) -c src/cuda_assignment_kernel.cu -o $(BIN_DIR)/cuda_assignment_kernel.o
	$(MPICXX) $(CXXFLAGS) -DUSE_CUDA_KERNEL src/kmeans_mpi.cpp $(BIN_DIR)/cuda_assignment_kernel.o \
		-L$(CUDA_HOME)/lib64 -lcudart -o $(BIN_DIR)/kmeans_hybrid

clean:
	rm -rf $(BIN_DIR)
