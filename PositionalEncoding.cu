#include "PositionalEncoding.h"
#include "Utilities.h"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

__global__ void 
initializePositionalEncodingKernel(float* pe, int d_model, int seq_len) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len * d_model) {
        int position = idx / d_model;
        int i = idx % d_model;
        float div_term = expf(-logf(10000.0f) * (i / d_model));
        if (i % 2 == 0) {
            pe[idx] = sinf(position * div_term);
        } else {
            pe[idx] = cosf(position * div_term);
        }
        // printf("pe[%d]: %f\n", idx, pe[idx]);
    }
}

__global__ void 
dropoutKernel(float* x, float* random_values, int size, float dropout) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (random_values[idx] < dropout) {
            x[idx] = 0.0f;
        }
    }
}

__global__ void 
addPositionalEncodingKernel(float* x, const float* pe, int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] += pe[idx];
    }
}

PositionalEncoding::PositionalEncoding(int d_model, int seq_len, float dropout)
    : d_model(d_model), seq_len(seq_len), dropout(dropout) 
{
    checkCudaErrors(cudaMalloc(&pe, seq_len * d_model * sizeof(float)));
    
    int blockSize = 256;
    int numBlocks = (seq_len * d_model + blockSize - 1) / blockSize;
    initializePositionalEncodingKernel<<<numBlocks, blockSize>>>(pe, d_model, seq_len);
    checkCudaErrors(cudaDeviceSynchronize());

    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 1234ULL);
}

PositionalEncoding::~PositionalEncoding() 
{
    checkCudaErrors(cudaFree(pe));
    curandDestroyGenerator(curand_gen);
}

void
PositionalEncoding::applyDropout(float* x, int size) 
{
    thrust::device_vector<float> random_values(size);
    curandGenerateUniform(curand_gen, thrust::raw_pointer_cast(random_values.data()), size);

    // Apply dropout 
    float* d_random_values = thrust::raw_pointer_cast(random_values.data());
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    dropoutKernel<<<numBlocks, blockSize>>>(x, d_random_values, size, dropout);
    checkCudaErrors(cudaDeviceSynchronize());
}

void 
PositionalEncoding::forward(float* x, int batch_size, int seq_len) 
{
    int size = batch_size * seq_len * d_model;

    // Copy of pe for PositionalEncoding 
    thrust::device_vector<float> pos_enc(size);
    for (int b = 0; b < batch_size; ++b) {
        checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(pos_enc.data()) + b * seq_len * d_model,
                                   pe, seq_len * d_model * sizeof(float),
                                   cudaMemcpyDeviceToDevice));
    }

    // Apply dropout (random values will get zeroed)
    applyDropout(thrust::raw_pointer_cast(pos_enc.data()), size);
    
    // Adding Pos enc to original data
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    addPositionalEncodingKernel<<<numBlocks, blockSize>>>(x, thrust::raw_pointer_cast(pos_enc.data()), size);
    checkCudaErrors(cudaDeviceSynchronize());
}
