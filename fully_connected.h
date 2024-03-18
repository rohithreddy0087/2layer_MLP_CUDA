#pragma once
#include "kernels.h"

void updateWeights(float* A, float* B, int m, int n, float learning_rate){
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
    matrixAddKernel <<< gridSize, blockSize >>> (A, B, m, n, -1*learning_rate);
}

void updateBias(float* a, float* b, int m, float learning_rate){
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((m + blockSize.y - 1) / blockSize.y);
    vectorAddKernel <<< gridSize, blockSize >>> (a, b, m, -1*learning_rate);
}

void printMatrix(float* W, int m, int n){
    float output [m * n];
    cudaMemcpy(output, W, m*n*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < m*n; i++) {
        printf("%f ", output[i]);
        if ((i + 1) % n == 0) {
            printf("\n");
        }
    }
}

void printVector(float*b, int m){
    float output [m];
    cudaMemcpy(output, b, m*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < m; i++) {
        printf("%f ", output[i]);
        printf("\n");
    }
}

void initializeWeights(float* W, float* b, int m, int n) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
    initializeWeightsKernel <<<gridSize, blockSize>>> (W, m, n);
    cudaDeviceSynchronize();

    dim3 gridSizeb((m + blockSize.x - 1) / blockSize.x);
    initializeBiasKernel <<<gridSizeb, BLOCK_SIZE >>> (b, m);
    cudaDeviceSynchronize();
}

class FullyConnectedLayer {

public:

    FullyConnectedLayer(int input_size, int output_size, int abatch_size, int index) : L_1(input_size), L(output_size), batch_size(abatch_size), index(index) {
        cudaMalloc((void**)&W_L, L_1 * L * sizeof(float));
        cudaMalloc((void**)&dW_L, L_1 * L * sizeof(float));
        cudaMalloc((void**)&b_L, L * sizeof(float));
        cudaMalloc((void**)&db_L, L * sizeof(float));
        cudaMalloc((void**)&A_L_1, L_1 * batch_size * sizeof(float));
        initializeWeights(W_L, b_L, L, L_1);
    }

    ~FullyConnectedLayer() {
        cudaFree(W_L);
        cudaFree(b_L);
        cudaFree(dW_L);
        cudaFree(db_L);
        cudaFree(A_L_1);
    }

    void forward(float* input, float* Z_L) {
        // Storing input for using it in backprop calculation
        cudaMemcpy(A_L_1, input, L_1*batch_size*sizeof(float ), cudaMemcpyHostToDevice);
        // Z_tmp[l] = W[l]*A[l-1]
        float* Z_tmp_L;
        cudaMalloc((void**)&Z_tmp_L, L* batch_size * sizeof(float));
        matrixMultiply(W_L, A_L_1, Z_tmp_L, L, L_1, batch_size);
        // Z[l] = Z_tmp[l] + b[l]
        matrixVectorAdd(Z_tmp_L, b_L, Z_L, L, batch_size);
        cudaFree(Z_tmp_L);
    }

    void backward(float* dZ_L, float* dA_L_1, float learning_rate) {

        // dW_L = (1/batch_size)*dZ_L*(A_L_1)^T
        float* A_L_1_transpose;
        cudaMalloc((void**)&A_L_1_transpose, L_1 * batch_size * sizeof(float));
        matrixTranspose(A_L_1, A_L_1_transpose, L_1, batch_size);
        matrixMultiply(dZ_L, A_L_1_transpose, dW_L, L, batch_size, L_1, batch_size);

        // db_L = (1/batch_size)*sum(dZ_L, axis=1)
        matrixAvgAlongAxis(dZ_L, db_L, L, batch_size);

        if(index>1) {
            // dA_L_1 = (W_L)^T.dZ_L
            float *W_L_transpose;
            cudaMalloc((void**)&W_L_transpose, L * L_1 * sizeof(float));
            matrixTranspose(W_L, W_L_transpose, L, L_1);
            matrixMultiply(W_L_transpose, dZ_L, dA_L_1, L_1, L, batch_size);
            cudaFree(W_L_transpose);
        }

        // W_L = W_L - learning_rate*dW_L;
        updateWeights(W_L, dW_L, L, L_1, learning_rate);

        // b_L = b_L - learning_rate*db_L;
        updateBias(b_L, db_L, L, learning_rate);

        cudaFree(A_L_1_transpose);
    }

    float* getWeights() const {
        return W_L;
    }

    float* getBiases() const {
        return b_L;
    }

    int getInputSize() const {
        return L_1;
    }

    int getOutputSize() const {
        return L;
    }

private:
    int L;
    int L_1;
    int batch_size;
    int index;
    float* W_L;
    float* b_L;
    float* A_L_1;
    float* dW_L;
    float* db_L;

};
