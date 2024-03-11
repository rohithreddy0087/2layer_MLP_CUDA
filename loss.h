#pragma once

__device__ void softmax(float* input, float* output, int num_classes) {
    float max_val = -INFINITY;
    for (int i = 0; i < num_classes; ++i) {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < num_classes; ++i) {
        output[i] /= sum;
    }
}


__global__ void crossEntropyLossKernel(float* logits, unsigned char* labels, float* activations, float* loss, int batch_size, int num_classes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < batch_size) {
        softmax(&logits[idx * num_classes], &activations[idx * num_classes], num_classes);
        int label = labels[idx];
        loss[idx] = (-logf(activations[idx * num_classes + label] + 1e-10f));
    }
}

__global__ void computeGradientKernel(float* activations, unsigned char* labels, float* delta, int batch_size, int num_classes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < batch_size) {
        int label = labels[idx];
        for (int i = 0; i < num_classes; ++i) {
            delta[idx * num_classes + i] = activations[idx * num_classes + i] - (i == label ? 1.0f : 0.0f);
        }
    }
}

class CrossEntropyLoss {
public:
    void computeLoss(float* logits, unsigned char* labels, float* activations, float* loss, int batch_size, int num_classes){ 
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((num_classes + blockSize.x - 1) / blockSize.x, (batch_size + blockSize.y - 1)/blockSize.y);
        crossEntropyLossKernel <<<gridSize, blockSize>>> (logits, labels, activations, loss, batch_size, num_classes);
        cudaDeviceSynchronize();
    }

    void computeGradient(float* activations, unsigned char* labels, float* delta, int batch_size, int num_classes) {
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((num_classes + blockSize.x - 1) / blockSize.x, (batch_size + blockSize.y - 1) / blockSize.y);
        computeGradientKernel <<< gridSize, blockSize >>> (activations, labels, delta, batch_size, num_classes);
        cudaDeviceSynchronize();
    }
};