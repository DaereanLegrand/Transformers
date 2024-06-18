NVCC = nvcc
NVCCFLAGS = -arch=sm_75
CUDA_LIBS = -lcudart -lcublas -lcurand

# Project files (modify to match your project structure)
TARGET = transformers
CUDA_SOURCES = InputEmbeddings.cu LayerNormalization.cu PositionalEncoding.cu Utilities.cu
OBJECTS = $(CUDA_SOURCES:.cu=.o)

# Default target
all: $(TARGET)

# Build rules
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ main.cu -o $@ $(CUDA_LIBS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Cleaning rule
clean:
	rm -f $(OBJECTS) $(TARGET)
