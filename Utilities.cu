#include "Utilities.h"
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>

void 
checkCudaErrors(cudaError_t err) 
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
}

void 
checkCublasErrors(cublasStatus_t status) 
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << status << std::endl;
        exit(status);
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
initializeWeightsKernel(float* weights, int size, unsigned long seed) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        weights[idx] = curand_uniform(&state) * 0.02f - 0.01f;
    }
}

__global__ void 
updateParametersKernel(float* param, float* grad_param, float learning_rate, int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        param[idx] -= learning_rate * grad_param[idx];
    }
}


