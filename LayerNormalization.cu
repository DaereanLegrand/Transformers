#include "LayerNormalization.h"
#include "Utilities.h"
#include <cmath>
#include <thrust/device_vector.h>

__global__ void 
initializeParametersKernel(float* alpha, float* bias, int features) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < features) {
        alpha[idx] = 1.0f;
        bias[idx] = 0.0f;
    }
}

__global__ void 
computeMeanAndVariance(const float* x, float* mean, float* variance, int batch_size, int seq_len, int features) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = idx / seq_len;
    int seq = idx % seq_len;

    if (batch < batch_size && seq < seq_len) {
        float sum = 0.0f;
        float sq_sum = 0.0f;
        for (int i = 0; i < features; ++i) {
            float val = x[batch * seq_len * features + seq * features + i];
            sum += val;
            sq_sum += val * val;
        }
        mean[idx] = sum / features;
        variance[idx] = sq_sum / features - mean[idx] * mean[idx];
    }
}

__global__ void 
normalize(float* x, const float* mean, const float* variance, const float* alpha, const float* bias, float eps, int batch_size, int seq_len, int features) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = idx / (seq_len * features);
    int seq = (idx / features) % seq_len;
    int feature = idx % features;

    if (batch < batch_size && seq < seq_len && feature < features) {
        float mean_val = mean[batch * seq_len + seq];
        float var_val = variance[batch * seq_len + seq];
        x[idx] = alpha[feature] * (x[idx] - mean_val) / sqrtf(var_val + eps) + bias[feature];
    }
}

__global__ void 
computeGradients(const float* grad_output, const float* x, const float* mean, const float* variance, float* grad_alpha, float* grad_bias, int batch_size, int seq_len, int features) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = idx / (seq_len * features);
    int seq = (idx / features) % seq_len;
    int feature = idx % features;

    if (batch < batch_size && seq < seq_len && feature < features) {
        float mean_val = mean[batch * seq_len + seq];
        float var_val = variance[batch * seq_len + seq];
        float normalized = (x[idx] - mean_val) / sqrtf(var_val + 1e-6);
        atomicAdd(&grad_alpha[feature], grad_output[idx] * normalized);
        atomicAdd(&grad_bias[feature], grad_output[idx]);
    }
}

__global__ void 
updateParametersKernel(float* alpha, float* bias, const float* grad_alpha, const float* grad_bias, float learning_rate, int features) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < features) {
        alpha[idx] -= learning_rate * grad_alpha[idx];
        bias[idx] -= learning_rate * grad_bias[idx];
    }
}

LayerNormalization::LayerNormalization(int features, float eps)
    : features(features), eps(eps) 
{
    checkCudaErrors(cudaMalloc(&alpha, features * sizeof(float)));
    checkCudaErrors(cudaMalloc(&bias, features * sizeof(float)));
    checkCudaErrors(cudaMalloc(&grad_alpha, features * sizeof(float)));
    checkCudaErrors(cudaMalloc(&grad_bias, features * sizeof(float)));

    initializeParameters();

    int blockSize = 256;
    int numBlocks = (features + blockSize - 1) / blockSize;
    initializeParametersKernel<<<numBlocks, blockSize>>>(alpha, bias, features);
    checkCudaErrors(cudaDeviceSynchronize());
}

LayerNormalization::~LayerNormalization() 
{
    checkCudaErrors(cudaFree(alpha));
    checkCudaErrors(cudaFree(bias));
    checkCudaErrors(cudaFree(grad_alpha));
    checkCudaErrors(cudaFree(grad_bias));
}

void 
LayerNormalization::initializeParameters() 
{
    int blockSize = 256;
    int numBlocks = (features + blockSize - 1) / blockSize;
    initializeParametersKernel<<<numBlocks, blockSize>>>(alpha, bias, features);
    checkCudaErrors(cudaDeviceSynchronize());
}

void 
LayerNormalization::forward(float* x, int batch_size, int seq_len) 
{
    int total_elements = batch_size * seq_len;
    thrust::device_vector<float> mean(total_elements);
    thrust::device_vector<float> variance(total_elements);

    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    computeMeanAndVariance<<<numBlocks, blockSize>>>(x, thrust::raw_pointer_cast(mean.data()), thrust::raw_pointer_cast(variance.data()), batch_size, seq_len, features);
    checkCudaErrors(cudaDeviceSynchronize());

    total_elements *= features;
    numBlocks = (total_elements + blockSize - 1) / blockSize;
    normalize<<<numBlocks, blockSize>>>(x, thrust::raw_pointer_cast(mean.data()), thrust::raw_pointer_cast(variance.data()), alpha, bias, eps, batch_size, seq_len, features);
    checkCudaErrors(cudaDeviceSynchronize());
}

void 
LayerNormalization::backward(float* grad_output, float* x, int batch_size, int seq_len, float learning_rate) 
{
    int total_elements = batch_size * seq_len;
    thrust::device_vector<float> mean(total_elements);
    thrust::device_vector<float> variance(total_elements);

    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    computeMeanAndVariance<<<numBlocks, blockSize>>>(x, thrust::raw_pointer_cast(mean.data()), thrust::raw_pointer_cast(variance.data()), batch_size, seq_len, features);
    checkCudaErrors(cudaDeviceSynchronize());

    total_elements *= features;
    numBlocks = (total_elements + blockSize - 1) / blockSize;
    computeGradients<<<numBlocks, blockSize>>>(grad_output, x, thrust::raw_pointer_cast(mean.data()), thrust::raw_pointer_cast(variance.data()), grad_alpha, grad_bias, batch_size, seq_len, features);
    checkCudaErrors(cudaDeviceSynchronize());

    numBlocks = (features + blockSize - 1) / blockSize;
    updateParametersKernel<<<numBlocks, blockSize>>>(alpha, bias, grad_alpha, grad_bias, learning_rate, features);
    checkCudaErrors(cudaDeviceSynchronize());
}
