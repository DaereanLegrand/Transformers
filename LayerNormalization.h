#ifndef LAYER_NORMALIZATION_H
#define LAYER_NORMALIZATION_H

#include <cuda_runtime.h>
#include <vector>

class LayerNormalization {
public:
    LayerNormalization(int features, float eps = 1e-6);
    ~LayerNormalization();
    void forward(float* x, int batch_size, int seq_len);
    void backward(float* grad_output, float* x, int batch_size, int seq_len, float learning_rate);

private:
    int features;
    float eps;
    float* alpha;
    float* bias;
    float* grad_alpha;
    float* grad_bias;

    void initializeParameters();
    void updateParameters(float learning_rate);
};

#endif // LAYER_NORMALIZATION_H

