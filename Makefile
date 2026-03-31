# Compiler
NVCC = nvcc

# Compiler Flags (-O3 for maximum optimization, -I to include the header folder)
NVCC_FLAGS = -O3 -I./include

# Target executable name
TARGET = augment_pipeline

# Build rules
all: $(TARGET)

$(TARGET): src/main.cu src/kernels.cu
	$(NVCC) $(NVCC_FLAGS) src/main.cu src/kernels.cu -o $(TARGET)

clean:
	rm -f $(TARGET)