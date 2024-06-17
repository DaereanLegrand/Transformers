main:
	nvcc -o output main.cu Utilities.cu InputEmbeddings.cu PositionalEncoding.cu -lcublas -lcurand -arch=sm_75
