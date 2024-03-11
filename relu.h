#pragma once

__global__ void reluForwardKernel(float* input, float* output, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void reluBackwardKernel(float* input, float* output, float* delta, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        delta[idx] *= (input[idx] > 0) ? 1.0f : 0.0f;
    }
}

class ReLU {
public:
    void forward(float* input, float* output, int batch_size, int feature_size) {
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((feature_size + blockSize.x - 1) / blockSize.x, (batch_size + blockSize.y - 1) / blockSize.y);

        reluForwardKernel <<<gridSize, blockSize>>> (input, output, size);
        cudaDeviceSynchronize();
    }

    void backward(float* input, float* output, float* delta, int size) {
        dim3 blocks((size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 threads(BLOCK_SIZE);

        reluBackwardKernel << <blocks, threads >> > (input, output, delta, size);
        cudaDeviceSynchronize();
    }
};