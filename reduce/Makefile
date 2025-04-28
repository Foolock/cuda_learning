# Makefile

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3 -Xcompiler "-O3 -fopenmp"

# Target
TARGET = reduce 

# Source files
SRC = reduce.cu

# Default rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

# Clean rule
clean:
	rm -f $(TARGET)

