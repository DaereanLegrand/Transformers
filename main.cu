#include <cstdio>
#include <iostream>
#include <iterator>
#include <vector>
#include "FeedForwardBlock.h"
#include "InputEmbeddings.h"
#include "LayerNormalization.h"
#include "MultiHeadAttentionBlock.h"
#include "PositionalEncoding.h"

using std::cout;
using std::endl;

void 
printGPUInfo() 
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount returned " << static_cast<int>(error_id) << " -> " << cudaGetErrorString(error_id) << std::endl;
        return;
    }

    if (deviceCount == 0) {
        std::cout << "There are no available CUDA devices." << std::endl;
    } else {
        std::cout << "Detected " << deviceCount << " CUDA Capable device(s)." << std::endl;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << ": " << deviceProp.name << std::endl;
        std::cout << "  CUDA Capability Major/Minor version number: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total amount of global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Number of multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Total amount of constant memory: " << deviceProp.totalConstMem << " bytes" << std::endl;
        std::cout << "  Total amount of shared memory per block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << "  Total number of registers available per block: " << deviceProp.regsPerBlock << std::endl;
    }
}

void
printTensor(const float* tensor, int batch_size, int seq_len, int d_model) 
{
    for (int b = 0; b < batch_size; ++b) {
        std::cout << "Batch " << b << ":" << std::endl;
        for (int s = 0; s < seq_len; ++s) {
            std::cout << "  Seq " << s << ": ";
            for (int d = 0; d < d_model; ++d) {
                std::cout << tensor[b * seq_len * d_model + s * d_model + d] << " ";
            }
            std::cout << std::endl;
        }
    }
}

int 
main() 
{
    cout << "Transformer Model C++ Basic Implementation" << endl;
    printGPUInfo();

    // Parameters
    // Multihead 
    int d_model = 8; // in paper 512
    // FeedFoward inner layer dimensionality
    int d_ff = 16; // in paper 2048
    // Size of dict
    int vocab_size = 10;
    // input will have...
    int batch_size = 2;
    // length of input
    int seq_len = 3;
    // likelyhood of dropout
    float dropout = 0.1;
    // h meaning heads
    int h = 4;

    std::vector<int> input = {1, 2, 3, 4, 5, 6};
    std::vector<float> output(batch_size * seq_len * d_model);
    float* d_output;   
    checkCudaErrors(cudaMalloc(&d_output, batch_size * seq_len * d_model * sizeof(float)));
    checkCudaErrors(cudaMemset(d_output, 0, batch_size * seq_len * d_model * sizeof(float)));

    //InputEmbeddings embeddings(d_model, vocab_size);
    //PositionalEncoding positional_encoding(d_model, seq_len, dropout);

    //printf("Creating Embeddings \n");
    //embeddings.forward(input.data(), d_output, batch_size, seq_len);
    //printf("Postional Encoding \n");
    //positional_encoding.forward(d_output, batch_size, seq_len);

    LayerNormalization layer_norm_mha(d_model);
    MultiHeadAttentionBlock multiHeadAttentionBlock(d_model, h, dropout);
    LayerNormalization layer_norm_ff(d_model);
    FeedForwardBlock feedForwardBlock(d_model, d_ff, dropout);

    printf("Creation of Q, K, V matrices: \n");

    int size = batch_size * seq_len * d_model;
    float *q, *k, *v;
    checkCudaErrors(cudaMalloc(&q, size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&k, size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&v, size * sizeof(float)));

    checkCudaErrors(cudaMemcpy(q, d_output, size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(k, d_output, size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(v, d_output, size * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    printTensor(output.data(), batch_size, seq_len, d_model);

    printf("Layer Normalization: \n");
    layer_norm_mha.forward(d_output, batch_size, seq_len);

    checkCudaErrors(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    printTensor(output.data(), batch_size, seq_len, d_model);

    printf("MultiHead Attention Block: \n");
    multiHeadAttentionBlock.forward(q, k, v, nullptr, d_output, batch_size, seq_len);

    checkCudaErrors(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    printTensor(output.data(), batch_size, seq_len, d_model);

    printf("Layer Normalization: \n");
    layer_norm_ff.forward(d_output, batch_size, seq_len);

    checkCudaErrors(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    printTensor(output.data(), batch_size, seq_len, d_model);

    printf("Feed Forward: \n");
    feedForwardBlock.forward(d_output, d_output, batch_size, seq_len);

    checkCudaErrors(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    printTensor(output.data(), batch_size, seq_len, d_model);
    return 0;
}
