#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector>

class PositionalEncoding {
public:
    PositionalEncoding(int d_model, int seq_len, float dropout);
    ~PositionalEncoding();
    void forward(float* x, int batch_size, int seq_len);

private:
    int d_model;
    int seq_len;
    float dropout;
    float* pe;
    curandGenerator_t curand_gen;

    void initializePositionalEncoding();
    void applyDropout(float* x, int size);
};

#endif // POSITIONAL_ENCODING_H

