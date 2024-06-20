#include "Utilities.h"
#include <iostream>

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


