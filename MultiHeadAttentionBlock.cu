#include "MultiHeadAttentionBlock.h"
#include "Utilities.h"
#include <cstdio>

__global__ void
combineHeadsKernel(float* attention_output, float* combined_output, int batch_size, int seq_len, int h, int d_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * h * d_k;

    if (idx < total_elements) {
        int b = idx / (seq_len * h * d_k);
        int s = (idx % (seq_len * h * d_k)) / (h * d_k);
        int head = (idx % (h * d_k)) / d_k;
        int k = idx % d_k;

        int combined_idx = b * (seq_len * h * d_k) + s * (h * d_k) + head * d_k + k;
        combined_output[combined_idx] = attention_output[idx];
    }
}

__global__ void
reshapeAndTranspose(float* input, float* output, int batch_size, int seq_len, int h, int d_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * h * d_k;

    if (idx < total_elements) {
        int b = idx / (seq_len * h * d_k);
        int s = (idx % (seq_len * h * d_k)) / (h * d_k);
        int head = (idx % (h * d_k)) / d_k;
        int k = idx % d_k;

        int input_idx = b * (seq_len * h * d_k) + s * h * d_k + head * d_k + k;
        output[idx] = input[input_idx];
    }
}

__global__ void
softmaxBackwardKernel(float* grad_out, const float* softmax_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sg = softmax_output[idx] * grad_out[idx];

        __shared__ float shared_sum[32];
        float thread_sum = sg;
        for (int stride = 16; stride > 0; stride >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, stride);
        }
        if (threadIdx.x % 32 == 0) {
            shared_sum[threadIdx.x / 32] = thread_sum;
        }
        __syncthreads();

        if (threadIdx.x < 32) {
            thread_sum = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_sum[threadIdx.x] : 0;
            for (int stride = 16; stride > 0; stride >>= 1) {
                thread_sum += __shfl_down_sync(0xffffffff, thread_sum, stride);
            }
            if (threadIdx.x == 0) {
                shared_sum[0] = thread_sum;
            }
        }
        __syncthreads();

        float sum_sg = shared_sum[0];

        grad_out[idx] = softmax_output[idx] * (grad_out[idx] - sum_sg);
    }
}

MultiHeadAttentionBlock::MultiHeadAttentionBlock(int d_model, int h, float dropout)
    : d_model(d_model), h(h), dropout(dropout) {
    if (d_model % h != 0) {
        throw std::invalid_argument("d_model is not divisible by h");
    }
    d_k = d_model / h;

    checkCudaErrors(cudaMalloc(&w_q, d_model * d_model * sizeof(float)));
    checkCudaErrors(cudaMalloc(&w_k, d_model * d_model * sizeof(float)));
    checkCudaErrors(cudaMalloc(&w_v, d_model * d_model * sizeof(float)));
    checkCudaErrors(cudaMalloc(&w_o, d_model * d_model * sizeof(float)));

    checkCudaErrors(cudaMalloc(&grad_w_q, d_model * d_model * sizeof(float)));
    checkCudaErrors(cudaMalloc(&grad_w_k, d_model * d_model * sizeof(float)));
    checkCudaErrors(cudaMalloc(&grad_w_v, d_model * d_model * sizeof(float)));
    checkCudaErrors(cudaMalloc(&grad_w_o, d_model * d_model * sizeof(float)));

    int blockSize = 256;
    int numBlocks = (d_model * d_model + blockSize - 1) / blockSize;
    initializeWeightsKernel<<<numBlocks, blockSize>>>(w_q, d_model * d_model, time(0));
    initializeWeightsKernel<<<numBlocks, blockSize>>>(w_k, d_model * d_model, time(0) + 1);
    initializeWeightsKernel<<<numBlocks, blockSize>>>(w_v, d_model * d_model, time(0) + 2);
    initializeWeightsKernel<<<numBlocks, blockSize>>>(w_o, d_model * d_model, time(0) + 3);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCublasErrors(cublasCreate(&cublas_handle));
}

MultiHeadAttentionBlock::~MultiHeadAttentionBlock() {
    checkCudaErrors(cudaFree(w_q));
    checkCudaErrors(cudaFree(w_k));
    checkCudaErrors(cudaFree(w_v));
    checkCudaErrors(cudaFree(w_o));

    checkCublasErrors(cublasDestroy(cublas_handle));
}

__global__ void
maskedFillKernel(float* scores, float* mask, int size, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (mask[idx] == 0) {
            scores[idx] = value;
        }
    }
}

__global__ void
softmaxKernel(float* scores, int batch_size, int h, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / (h * seq_len * seq_len);
    int h_idx = (idx % (h * seq_len * seq_len)) / (seq_len * seq_len);
    int i = (idx % (seq_len * seq_len)) / seq_len;

    if (idx < batch_size * h * seq_len * seq_len) {
        float max_val = -1e9;
        for (int k = 0; k < seq_len; ++k) {
            max_val = max(max_val, scores[b * h * seq_len * seq_len + h_idx * seq_len * seq_len + i * seq_len + k]);
        }

        float sum_exp = 0.0f;
        for (int k = 0; k < seq_len; ++k) {
            sum_exp += exp(scores[b * h * seq_len * seq_len + h_idx * seq_len * seq_len + i * seq_len + k] - max_val);
        }

        scores[idx] = exp(scores[idx] - max_val) / sum_exp;
    }
}

__global__ void
dropoutKernel(float* scores, float dropout_prob, int size, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float rand_val = curand_uniform(&state);
        scores[idx] = (rand_val > dropout_prob) ? scores[idx] / (1.0f - dropout_prob) : 0.0f;
    }
}

void
MultiHeadAttentionBlock::attention(float* query, float* key, float* value, float* mask, float* output, int batch_size, int h, int seq_len, int d_k, float dropout) {
    float* attention_scores;
    checkCudaErrors(cudaMalloc(&attention_scores, batch_size * h * seq_len * seq_len * sizeof(float)));

    float scale = 1.0f / sqrtf(d_k);

    float alpha = 1.0f;
    float beta = 0.0f;
    checkCublasErrors(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                                seq_len, seq_len, d_k,
                                                &scale,
                                                key, d_k, seq_len * d_k,
                                                query, d_k, seq_len * d_k,
                                                &beta,
                                                attention_scores, seq_len, seq_len * seq_len,
                                                batch_size * h));

    if (mask != nullptr) {
        int blockSize = 256;
        int numBlocks = (batch_size * h * seq_len * seq_len + blockSize - 1) / blockSize;
        maskedFillKernel<<<numBlocks, blockSize>>>(attention_scores, mask, batch_size * h * seq_len * seq_len, -1e9f);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    int blockSize = 256;
    int numBlocks = (batch_size * h * seq_len * seq_len + blockSize - 1) / blockSize;
    softmaxKernel<<<numBlocks, blockSize>>>(attention_scores, batch_size, h, seq_len);
    checkCudaErrors(cudaDeviceSynchronize());

    if (dropout > 0.0f) {
        dropoutKernel<<<numBlocks, blockSize>>>(attention_scores, dropout, batch_size * h * seq_len * seq_len, time(0));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    //printf("Attention Output: \n");
    //printf("Batch size: %d\n h: %d\n seq_len: %d\n d_k: %d\n", batch_size, h, seq_len, d_k);
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < h; ++i) {
            int score_offset = (b * h + i) * seq_len * seq_len;
            int value_offset = (b * h + i) * seq_len * d_k;
            int output_offset = (b * h + i) * seq_len * d_k;

            checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                          d_k, seq_len, seq_len,
                                          &alpha,
                                          value + value_offset, d_k,
                                          attention_scores + score_offset, seq_len,
                                          &beta,
                                          output + output_offset, d_k));
        }
    }

    checkCudaErrors(cudaFree(attention_scores));
}


void 
MultiHeadAttentionBlock::forward(float* q, float* k, float* v, float* mask, float* output, int batch_size, int seq_len) {
    float* query;
    float* key;
    float* value;
    checkCudaErrors(cudaMalloc(&query, batch_size * seq_len * d_model * sizeof(float)));
    checkCudaErrors(cudaMalloc(&key, batch_size * seq_len * d_model * sizeof(float)));
    checkCudaErrors(cudaMalloc(&value, batch_size * seq_len * d_model * sizeof(float)));

    float alpha = 1.0f;
    float beta = 0.0f;
    checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, d_model, batch_size * seq_len, d_model, &alpha, w_q, d_model, q, d_model, &beta, query, d_model));
    checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, d_model, batch_size * seq_len, d_model, &alpha, w_k, d_model, k, d_model, &beta, key, d_model));
    checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, d_model, batch_size * seq_len, d_model, &alpha, w_v, d_model, v, d_model, &beta, value, d_model));

    float* reshaped_query;
    float* reshaped_key;
    float* reshaped_value;
    checkCudaErrors(cudaMalloc(&reshaped_query, batch_size * h * seq_len * d_k * sizeof(float)));
    checkCudaErrors(cudaMalloc(&reshaped_key, batch_size * h * seq_len * d_k * sizeof(float)));
    checkCudaErrors(cudaMalloc(&reshaped_value, batch_size * h * seq_len * d_k * sizeof(float)));

    int blockSize = 256;
    int numBlocks = (batch_size * seq_len * d_model + blockSize - 1) / blockSize;
    reshapeAndTranspose<<<numBlocks, blockSize>>>(query, reshaped_query, batch_size, seq_len, h, d_k);
    reshapeAndTranspose<<<numBlocks, blockSize>>>(key, reshaped_key, batch_size, seq_len, h, d_k);
    reshapeAndTranspose<<<numBlocks, blockSize>>>(value, reshaped_value, batch_size, seq_len, h, d_k);
    checkCudaErrors(cudaDeviceSynchronize());

    float* attention_output;
    checkCudaErrors(cudaMalloc(&attention_output, batch_size * h * seq_len * d_k * sizeof(float)));
    attention(reshaped_query, reshaped_key, reshaped_value, mask, attention_output, batch_size, h, seq_len, d_k, dropout);

    float* combined_output;
    checkCudaErrors(cudaMalloc(&combined_output, batch_size * seq_len * d_model * sizeof(float)));

    blockSize = 256;
    numBlocks = (batch_size * seq_len * d_model + blockSize - 1) / blockSize;
    combineHeadsKernel<<<numBlocks, blockSize>>>(attention_output, combined_output, batch_size, seq_len, h, d_k);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, d_model, batch_size * seq_len, d_model, &alpha, w_o, d_model, combined_output, d_model, &beta, output, d_model));

    checkCudaErrors(cudaFree(query));
    checkCudaErrors(cudaFree(key));
    checkCudaErrors(cudaFree(value));
    checkCudaErrors(cudaFree(reshaped_query));
    checkCudaErrors(cudaFree(reshaped_key));
    checkCudaErrors(cudaFree(reshaped_value));
    checkCudaErrors(cudaFree(attention_output));
    checkCudaErrors(cudaFree(combined_output));
}

void 
MultiHeadAttentionBlock::backward(float* grad_output, float* q, float* k, float* v, float* mask, int batch_size, int seq_len) {
    float *grad_q, *grad_k, *grad_v, *grad_attention_scores;
    checkCudaErrors(cudaMalloc(&grad_q, batch_size * seq_len * d_model * sizeof(float)));
    checkCudaErrors(cudaMalloc(&grad_k, batch_size * seq_len * d_model * sizeof(float)));
    checkCudaErrors(cudaMalloc(&grad_v, batch_size * seq_len * d_model * sizeof(float)));
    checkCudaErrors(cudaMalloc(&grad_attention_scores, batch_size * h * seq_len * seq_len * sizeof(float)));

    float alpha = 1.0f;
    float beta = 0.0f;

    float* reshaped_grad_output;
    checkCudaErrors(cudaMalloc(&reshaped_grad_output, batch_size * h * seq_len * d_k * sizeof(float)));
    int blockSize = 256;
    int numBlocks = (batch_size * seq_len * d_model + blockSize - 1) / blockSize;
    reshapeAndTranspose<<<numBlocks, blockSize>>>(grad_output, reshaped_grad_output, batch_size, seq_len, h, d_k);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < h; ++i) {
            int offset = (b * h + i) * seq_len * d_k;
            int score_offset = (b * h + i) * seq_len * seq_len;

            checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                          d_k, seq_len, seq_len,
                                          &alpha,
                                          reshaped_grad_output + offset, d_k,
                                          attention_scores + score_offset, seq_len,
                                          &beta,
                                          grad_v + offset, d_k));

            checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                          seq_len, seq_len, d_k,
                                          &alpha,
                                          v + offset, d_k,
                                          reshaped_grad_output + offset, d_k,
                                          &beta,
                                          grad_attention_scores + score_offset, seq_len));

            softmaxBackwardKernel<<<numBlocks, blockSize>>>(grad_attention_scores + score_offset, 
                                                            attention_scores + score_offset, 
                                                            seq_len * seq_len);

            float scale = 1.0f / sqrtf(d_k);
            
            checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                          d_k, seq_len, seq_len,
                                          &scale,
                                          k + offset, d_k,
                                          grad_attention_scores + score_offset, seq_len,
                                          &beta,
                                          grad_q + offset, d_k));

            checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                          d_k, seq_len, seq_len,
                                          &scale,
                                          q + offset, d_k,
                                          grad_attention_scores + score_offset, seq_len,
                                          &beta,
                                          grad_k + offset, d_k));
        }
    }

    checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                  d_model, d_model, batch_size * seq_len,
                                  &alpha,
                                  q, d_model,
                                  grad_q, d_model,
                                  &beta,
                                  grad_w_q, d_model));

    checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                  d_model, d_model, batch_size * seq_len,
                                  &alpha,
                                  k, d_model,
                                  grad_k, d_model,
                                  &beta,
                                  grad_w_k, d_model));

    checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                  d_model, d_model, batch_size * seq_len,
                                  &alpha,
                                  v, d_model,
                                  grad_v, d_model,
                                  &beta,
                                  grad_w_v, d_model));

    checkCudaErrors(cudaFree(grad_q));
    checkCudaErrors(cudaFree(grad_k));
    checkCudaErrors(cudaFree(grad_v));
    checkCudaErrors(cudaFree(grad_attention_scores));
    checkCudaErrors(cudaFree(reshaped_grad_output));
}

void
MultiHeadAttentionBlock::updateParameters(float learning_rate) {
    int size = d_model * d_model;
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    updateParametersKernel<<<numBlocks, blockSize>>>(w_q, grad_w_q, size, learning_rate);
    updateParametersKernel<<<numBlocks, blockSize>>>(w_k, grad_w_k, size, learning_rate);
    updateParametersKernel<<<numBlocks, blockSize>>>(w_v, grad_w_v, size, learning_rate);
    updateParametersKernel<<<numBlocks, blockSize>>>(w_o, grad_w_o, size, learning_rate);

    checkCudaErrors(cudaDeviceSynchronize());
}
