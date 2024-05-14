ROCM_PATH ?= /opt/rocm

CC		   = $(ROCM_PATH)/llvm/bin/clang++
# CFLAGS     = -fopenmp=libomp -target x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1030
CC = hipcc
CFLAGS = -O3 


kernels = matrixMultiply saxpy VmatrixMultiply Vsaxpy

all : $(kernels)

matrixMultiply: matrixMultiply.cpp
	$(CC) -o $@ $^ $(CFLAGS)
	@roc-obj -d $@

VmatrixMultiply: matrixMultiply.cpp
	$(CC) -o $@ $^ $(CFLAGS) -fvectorize
	@roc-obj -d $@

saxpy: saxpy.cpp
	$(CC) -o $@ $^ $(CFLAGS)
	@roc-obj -d $@

Vsaxpy: saxpy.cpp
	$(CC) -o $@ $^ $(CFLAGS) -fvectorize
	@roc-obj -d $@

clean :
	rm *.s $(kernels) *:1*
