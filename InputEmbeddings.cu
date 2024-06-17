#include "InputEmbeddings.h"
#include "Utilities.h"
#include <cstdio>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>

__global__ void initializeEmbeddingMatrix(float* embedding_matrix, int vocab_size, int d_model, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size * d_model) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        embedding_matrix[idx] = curand_uniform(&state);
    }
}

InputEmbeddings::InputEmbeddings(int d_model, int vocab_size)
    : d_model(d_model), vocab_size(vocab_size) {
    // Allocate memory 
    checkCudaErrors(cudaMalloc(&embedding_matrix, vocab_size * d_model * sizeof(float)));

    // Random values for each word
    int blockSize = 256;
    int numBlocks = (vocab_size * d_model + blockSize - 1) / blockSize;
    initializeEmbeddingMatrix<<<numBlocks, blockSize>>>(embedding_matrix, vocab_size, d_model, time(0));
    checkCudaErrors(cudaDeviceSynchronize());

    // Create cuBLAS handle
    checkCublasErrors(cublasCreate(&cublas_handle));
}

InputEmbeddings::~InputEmbeddings() {
    // Free the embedding matrix
    checkCudaErrors(cudaFree(embedding_matrix));

    // Destroy cuBLAS handle
    checkCublasErrors(cublasDestroy(cublas_handle));
}

void InputEmbeddings::forward(const int* input, float* output, int batch_size, int seq_len) {
    // Scale factor
    float scale = std::sqrt(d_model);

    // Loop over each element in the batch and sequence length
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            int idx = b * seq_len + s;
            int word_id = input[idx];

            if (word_id < 0 || word_id >= vocab_size) {
                std::cerr << "Word ID out of range: " << word_id << std::endl;
                exit(EXIT_FAILURE);
            }
            // Use cuBLAS to copy the embedding vector and scale it
            checkCublasErrors(cublasScopy(cublas_handle, d_model, embedding_matrix + word_id * d_model, 1, output + idx * d_model, 1));
            checkCublasErrors(cublasSscal(cublas_handle, d_model, &scale, output + idx * d_model, 1));
        }
    }
}
