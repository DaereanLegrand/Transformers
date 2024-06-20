#include <cuda_runtime.h>
#include "cublas_v2.h"

void checkCudaErrors(cudaError_t);
void checkCublasErrors(cublasStatus_t);
__global__ void dropoutKernel(float* x, float* random_values, int size, float dropout);
