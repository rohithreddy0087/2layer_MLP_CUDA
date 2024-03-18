#pragma once

__global__ void reluForwardKernel(float* input, float* output, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        output[row*n+col] = fmaxf(0.0f, input[row*n+col]);
    }
}

__global__ void reluBackwardKernel(float* input, float* delta, float* output, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        output[row*n+col] = delta[row*n+col] * ((input[row*n+col] > 0) ? 1.0f : 0.0f);
    }
}

class ReLU {
public:

    ReLU(int input_size, int abatch_size):L(input_size), batch_size(abatch_size){
        cudaMalloc((void**)&Z_L, L * batch_size * sizeof(float));
    }

    ~ReLU() = default;

    void forward(float* input_Z_L, float* A_L) const{

        cudaMemcpy(Z_L, input_Z_L, L*batch_size*sizeof(float ), cudaMemcpyDeviceToDevice);

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((batch_size + blockSize.x - 1) / blockSize.x, (L + blockSize.y - 1) / blockSize.y);

        // A_L = relu(Z_L)
        reluForwardKernel <<<gridSize, blockSize>>> (Z_L, A_L, L, batch_size);
        cudaDeviceSynchronize();
    }

    void backward(float* dA_L, float* dZ_L) {
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((batch_size + blockSize.x - 1) / blockSize.x, (L + blockSize.y - 1) / blockSize.y);

        // dZ_L = dA_L*dRelu(Z_L)
        reluBackwardKernel <<<gridSize, blockSize>>> (Z_L, dA_L, dZ_L, L, batch_size);
        cudaDeviceSynchronize();
    }

private:
    int L;
    int batch_size;
    float* Z_L;
};