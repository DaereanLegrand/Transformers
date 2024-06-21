#include "FeedForwardBlock.h"
#include "Utilities.h"
#include <time.h>
#include <cmath>
#include <thrust/device_vector.h>

__global__ void
computeGradientsKernel(float* grad_output, float* grad_w1, float* grad_b1, float* grad_w2, float* grad_b2, float* input, int d_model, int d_ff, int batch_size, int seq_len) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < d_ff) {
        float grad_b2_sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                int pos = b * seq_len * d_ff + s * d_ff + idx;
                grad_b2_sum += grad_output[pos];
            }
        }
        grad_b2[idx] = grad_b2_sum / (batch_size * seq_len);
    }

    if (idx < d_model * d_ff) {
        int w2_row = idx / d_ff;
        int w2_col = idx % d_ff;
        float grad_w2_sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                int pos = b * seq_len * d_ff + s * d_ff + w2_col;
                grad_w2_sum += grad_output[pos] * input[b * seq_len * d_model + s * d_model + w2_row];
            }
        }
        grad_w2[idx] = grad_w2_sum / (batch_size * seq_len);
    }

    // Computing gradients for the first linear layer
    if (idx < d_model) {
        for (int i = 0; i < d_ff; ++i) {
            float grad_b1_sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                for (int s = 0; s < seq_len; ++s) {
                    int pos = b * seq_len * d_ff + s * d_ff + i;
                    grad_b1_sum += grad_output[pos];
                }
            }
            grad_b1[i] = grad_b1_sum / (batch_size * seq_len);
        }

        for (int i = 0; i < d_ff * d_model; ++i) {
            int w1_row = i / d_model;
            int w1_col = i % d_model;
            float grad_w1_sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                for (int s = 0; s < seq_len; ++s) {
                    int pos = b * seq_len * d_model + s * d_model + w1_col;
                    grad_w1_sum += grad_output[w1_row] * input[pos];
                }
            }
            grad_w1[i] = grad_w1_sum / (batch_size * seq_len);
        }
    }
}

__global__ void
relu(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

FeedForwardBlock::FeedForwardBlock(int d_model, int d_ff, float dropout)
    : d_model(d_model), d_ff(d_ff), dropout(dropout)
{
    checkCudaErrors(cudaMalloc(&w1, d_model * d_ff * sizeof(float)));
    checkCudaErrors(cudaMalloc(&b1, d_ff * sizeof(float)));
    checkCudaErrors(cudaMalloc(&w2, d_ff * d_model * sizeof(float)));
    checkCudaErrors(cudaMalloc(&b2, d_model * sizeof(float)));

    checkCudaErrors(cudaMalloc(&grad_w1, d_model * d_ff * sizeof(float)));
    checkCudaErrors(cudaMalloc(&grad_b1, d_ff * sizeof(float)));
    checkCudaErrors(cudaMalloc(&grad_w2, d_ff * d_model * sizeof(float)));
    checkCudaErrors(cudaMalloc(&grad_b2, d_model * sizeof(float)));

    checkCublasErrors(cublasCreate(&cublas_handle));
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, time(0));

    initializeWeights();
}

FeedForwardBlock::~FeedForwardBlock() 
{
    checkCudaErrors(cudaFree(w1));
    checkCudaErrors(cudaFree(b1));
    checkCudaErrors(cudaFree(w2));
    checkCudaErrors(cudaFree(b2));
    checkCudaErrors(cudaFree(grad_w1));
    checkCudaErrors(cudaFree(grad_b1));
    checkCudaErrors(cudaFree(grad_w2));
    checkCudaErrors(cudaFree(grad_b2));

    checkCublasErrors(cublasDestroy(cublas_handle));
    curandDestroyGenerator(curand_gen);
}

void
FeedForwardBlock::initializeWeights()
{
    int blockSize = 256;
    int numBlocks_w1 = (d_model * d_ff + blockSize - 1) / blockSize;
    int numBlocks_w2 = (d_ff * d_model + blockSize - 1) / blockSize;
    int numBlocks_b1 = (d_ff + blockSize - 1) / blockSize;
    int numBlocks_b2 = (d_model + blockSize - 1) / blockSize;

    initializeWeightsKernel<<<numBlocks_w1, blockSize>>>(w1, d_model * d_ff, time(0));
    initializeWeightsKernel<<<numBlocks_w2, blockSize>>>(w2, d_ff * d_model, time(0));
    initializeWeightsKernel<<<numBlocks_b1, blockSize>>>(b1, d_ff, time(0));
    initializeWeightsKernel<<<numBlocks_b2, blockSize>>>(b2, d_model, time(0));

    checkCudaErrors(cudaDeviceSynchronize());
}

void
FeedForwardBlock::applyDropout(float* x, int size) 
{
    thrust::device_vector<float> random_values(size);
    curandGenerateUniform(curand_gen, thrust::raw_pointer_cast(random_values.data()), size);

    float* d_random_values = thrust::raw_pointer_cast(random_values.data());
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    dropoutKernel<<<numBlocks, blockSize>>>(x, d_random_values, size, dropout);
    checkCudaErrors(cudaDeviceSynchronize());
}

void
FeedForwardBlock::forward(float* input, float* output, int batch_size, int seq_len)
{
    float alpha = 1.0f;
    float beta = 0.0f;

    // First linear transformation (input -> d_ff)
    checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, d_ff, batch_size * seq_len, d_model, &alpha, w1, d_ff, input, d_model, &beta, output, d_ff));

    // Apply ReLU activation
    int total_elements = batch_size * seq_len * d_ff;
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    relu<<<numBlocks, blockSize>>>(output, total_elements);
    checkCudaErrors(cudaDeviceSynchronize());

    // Apply Dropout (random values will get zeroed)
    applyDropout(output, total_elements);

    // Second linear transformation (d_ff -> d_model)
    checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, d_model, batch_size * seq_len, d_ff, &alpha, w2, d_model, output, d_ff, &beta, output, d_model));
}

void
FeedForwardBlock::backward(float* grad_output, float* input, int batch_size, int seq_len)
{
    int total_elements = batch_size * seq_len * d_ff;
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;

    // Compute gradients for w1, b1, w2, b2
    computeGradientsKernel<<<numBlocks, blockSize>>>(grad_output, grad_w1, grad_b1, grad_w2, grad_b2, input, d_model, d_ff, batch_size, seq_len);
    checkCudaErrors(cudaDeviceSynchronize());
}

void
FeedForwardBlock::updateParameters(float learning_rate)
{
    int blockSize = 256;
    int numBlocks_w1 = (d_model * d_ff + blockSize - 1) / blockSize;
    int numBlocks_w2 = (d_ff * d_model + blockSize - 1) / blockSize;
    int numBlocks_b1 = (d_ff + blockSize - 1) / blockSize;
    int numBlocks_b2 = (d_model + blockSize - 1) / blockSize;

    updateParametersKernel<<<numBlocks_w1, blockSize>>>(w1, grad_w1, learning_rate, d_model * d_ff);
    updateParametersKernel<<<numBlocks_w2, blockSize>>>(w2, grad_w2, learning_rate, d_ff * d_model);
    updateParametersKernel<<<numBlocks_b1, blockSize>>>(b1, grad_b1, learning_rate, d_ff);
    updateParametersKernel<<<numBlocks_b2, blockSize>>>(b2, grad_b2, learning_rate, d_model);

    checkCudaErrors(cudaDeviceSynchronize());
}
