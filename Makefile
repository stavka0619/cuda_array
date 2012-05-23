SHELL = /bin/sh

# Suffixes:
.SUFFIXES: .cu .o

# Compilers and linker:
CC = nvcc


# Include and library paths:
CUDA_PATH = /usr/local/cuda/include/
CUDA_LIB_PATH = /usr/local/cuda/lib64/
CUDA_LIB_PATH2 = /home/xwang/NVIDIA_GPU_Computing_SDK/C/lib/
CUDA_LIB_PATH3 = /home/xwang/NVIDIA_GPU_Computing_SDK/shared/lib/
CUDA_SHR_PATH = /home/xwang/NVIDIA_GPU_Computing_SDK/shared/inc/
CUDA_SHR_PATH2 = /home/xwang/NVIDIA_GPU_Computing_SDK/C/common/inc/

# Compiler's flags:
CFLAGS = -pg -g  -I./ -I$(CUDA_PATH) -I$(CUDA_SHR_PATH) -I$(CUDA_SHR_PATH2) -L$(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH2) -L$(CUDA_LIB_PATH3) -lcutil_x86_64 -lcublas -lcuda -lcudart -lshrutil_x86_64 -lm --ptxas-options=-v -gencode=arch=compute_20,code=sm_20 
 

# Command to erase files:
RM = /bin/rm -vf

EXECUTABLE	:= test
# Program's name:
# Cuda source files (compiled with cudacc)
CUFILES		:= test.cu
# C/C++ source files (compiled with gcc / c++)
OBJS = $(CUFILES).o


# Building the application:
default: $(OBJS)
	$(CC) $(CFLAGS) -o $(EXECUTABLE) $(OBJS)
$(CUFILES).o:$(CUFILES) *.h
	$(CC) $(CFLAGS) -c $(CUFILES) -o $(OBJS)

# Rule for cleaning re-compilable files
clean:
	$(RM) $(OBJS) $(EXECUTABLE)



