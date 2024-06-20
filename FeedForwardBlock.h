#ifndef FEED_FORWARD_BLOCK_H
#define FEED_FORWARD_BLOCK_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>

class FeedForwardBlock {
public:
    FeedForwardBlock(int d_model, int d_ff, float dropout);
    ~FeedForwardBlock();
    void initializeWeights();
    void applyDropout(float* x, int size);
    void forward(float* input, float* output, int batch_size, int seq_len);
    void backward(float* grad_output, float* input, int batch_size, int seq_len);
    void updateParameters(float learning_rate);

private:
    int d_model;
    int d_ff;
    float dropout;

    float* w1;
    float* b1;
    float* w2;
    float* b2;

    float* grad_w1;
    float* grad_b1;
    float* grad_w2;
    float* grad_b2;

    cublasHandle_t cublas_handle;
    curandGenerator_t curand_gen;
};

#endif // FEED_FORWARD_BLOCK_H

