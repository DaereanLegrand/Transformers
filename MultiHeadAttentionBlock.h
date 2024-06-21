#ifndef MULTIHEADATTENTIONBLOCK_H
#define MULTIHEADATTENTIONBLOCK_H

#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cmath>
#include <iostream>

class MultiHeadAttentionBlock {
public:
    MultiHeadAttentionBlock(int d_model, int h, float dropout);
    ~MultiHeadAttentionBlock();
    void forward(float* q, float* k, float* v, float* mask, float* output, int batch_size, int seq_len);
    void backward(float* grad_output, float* q, float* k, float* v, float* mask, int batch_size, int seq_len);
    void updateParameters(float learning_rate);

private:
    int d_model;
    int h;
    int d_k;
    float dropout;

    float* w_q;
    float* w_k;
    float* w_v;
    float* w_o;

    float *grad_w_q, *grad_w_k, *grad_w_v, *grad_w_o;
    float *attention_scores;  // Store this from the forward pass

    cublasHandle_t cublas_handle;

    void attention(float* query, float* key, float* value, float* mask, float* output, int batch_size, int h, int seq_len, int d_k, float dropout);
};

#endif // MULTIHEADATTENTIONBLOCK_H
