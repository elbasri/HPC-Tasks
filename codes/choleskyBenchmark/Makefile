# Include paths for MKL and CUDA
MKLINCLUDE := -m64 -I/srv/software/easybuild/software/imkl/2022.1.0/mkl/2022.1.0/include
MKLLIB := -m64 -L/srv/software/easybuild/software/imkl/2022.1.0/mkl/2022.1.0/lib/intel64 \
          -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
CUDA_LIB := -L/usr/local/cuda/lib64 -lcudart -lcusolver

# Compiler and flags
CXX = g++
CXXFLAGS = -fopenmp -O3 $(MKLINCLUDE)

# Source files and executable target
SOURCES = main.cpp utils.h helper_string.h helper_cuda.h
TARGET = cholesky

# Rules for building
all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) -o $@ $(SOURCES) $(CXXFLAGS) $(MKLLIB) $(CUDA_LIB)

clean:
	rm -f $(TARGET) *.o
