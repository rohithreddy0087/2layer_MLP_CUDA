#pragma once
#include <curand_kernel.h>


__global__ void matrixMul(float* a, float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

__global__ void matrixVecAdd(float* a, float* b, int m, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        a[row * k + col] += b[col];
    }
}

__global__ void initializeWeightsKernel(float* weights, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < k) {
        curandState state;
        curand_init(0, row * k + col, 0, &state);
        weights[row * k + col] = 2 * curand_uniform(&state) - 1;
    }
}

__global__ void initializeBiasKernel(float* bias, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < k) {
        curandState state;
        curand_init(0, col, 0, &state);
        bias[col] = 2 * curand_uniform(&state) - 1;
    }
}

class FullyConnectedLayer {
private:
    float* d_weights;
    float* d_biases;
    int input_size;
    int output_size;

public:
    FullyConnectedLayer(int input_size, int output_size) : input_size(input_size), output_size(output_size) {
        cudaMalloc(&d_weights, input_size * output_size * sizeof(float));
        cudaMalloc(&d_biases, output_size * sizeof(float));
        initializeWeights();
    }

    ~FullyConnectedLayer() {
        cudaFree(d_weights);
        cudaFree(d_biases);
    }


    void initializeWeights() {
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((input_size + blockSize.x - 1) / blockSize.x, (output_size + blockSize.y - 1) / blockSize.y);
        initializeWeightsKernel <<<gridSize, blockSize>>> (d_weights, output_size, input_size);

        /*float output[4 * 5];
        cudaMemcpy(output, d_weights, 20*sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 20; i++) {
            printf("%.6f ", output[i]);
            if ((i + 1) % input_size == 0) {
                printf("\n");
            }
        }*/

        dim3 gridSizeb((output_size + blockSize.x - 1) / blockSize.x);
        initializeBiasKernel <<<gridSizeb, BLOCK_SIZE >>> (d_biases, output_size);

        cudaDeviceSynchronize();
    }

    void forward(float* input, float* output, int batch_size) {
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x, (batch_size + blockSize.y - 1) / blockSize.y);

        matrixMul <<< gridSize, blockSize >>> (d_weights, input, output, batch_size, input_size, output_size);
        matrixVecAdd <<< gridSize, blockSize >>> (output, d_biases, batch_size, output_size);
    }

    void backward(float* input, float* delta_z, float* input_gradients, float* weights_gradients, float* biases_gradients, int batch_size, float learning_rate) {
        
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x, (batch_size + blockSize.y - 1) / blockSize.y);

        matrixMul <<< blockSize, gridSize >>> (delta_z, input, weights_gradients, batch_size, input_size, output_size);
        cudaDeviceSynchronize();

        matrixMul <<< blockSize, gridSize >>> (d_weights, delta_z, input_gradients, batch_size, input_size, output_size);
        cudaDeviceSynchronize();

        cudaMemcpy(biases_gradients, delta_z, output_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
};