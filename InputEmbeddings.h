#ifndef INPUTEMBEDDINGS_H
#define INPUTEMBEDDINGS_H

#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "cublas_v2.h"

void checkCudaErrors(cudaError_t err);
void checkCublasErrors(cublasStatus_t status);

class InputEmbeddings {
public:
    InputEmbeddings(int d_model, int vocab_size);
    ~InputEmbeddings();

    void forward(const int* input, float* output, int batch_size, int seq_len);

private:
    int d_model;
    int vocab_size;
    float* embedding_matrix;
    cublasHandle_t cublas_handle;
};

#endif
