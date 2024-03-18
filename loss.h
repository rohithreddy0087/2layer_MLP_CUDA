#pragma once

__global__ void crossEntropyLossKernel(float* logits, int* labels, float* activations, float* loss, int batch_size, int num_classes) {

    int batch = threadIdx.x + blockIdx.x * blockDim.x;

    if (batch < batch_size) {
        int label = labels[batch];
        float max_val = -INFINITY;
        for (int i = 0; i < num_classes; ++i) {
            float logit = logits[i * batch_size + batch];
            if (logit > max_val) {
                max_val = logit;
            }
        }

        float sum_exp = 0.0;
        for (int i = 0; i < num_classes; ++i) {
            sum_exp += expf(logits[i * batch_size + batch] - max_val);
        }

        float prob;
        for (int i = 0; i < num_classes; ++i) {
            prob = expf(logits[i * batch_size + batch] - max_val) / sum_exp;
            activations[i * batch_size + batch] = prob; // Store softmax activations for backpropagation
            if (i == label) {
                loss[batch] = -logf(prob + 1e-10f); // Add epsilon to avoid log(0)
            }
        };
    }
}

__global__ void computeGradientKernel(float* activations, int* labels, float* delta, int batch_size, int num_classes) {
    int batch = threadIdx.x + blockIdx.x * blockDim.x;

    if (batch < batch_size) {
        int label = labels[batch];
        for (int i = 0; i < num_classes; ++i) {
            delta[i * batch_size + batch] = (activations[i * batch_size + batch] - (i == label ? 1.0f : 0.0f))/batch_size;
        }
    }
}

class CrossEntropyLoss {
public:

    CrossEntropyLoss(int anum_classes, int abatch_size): num_classes(anum_classes), batch_size(abatch_size){
        cudaMalloc((void**)&loss_classes, batch_size* sizeof(float));
    }

    ~CrossEntropyLoss(){
        cudaFree(loss_classes);
    }

    // Logits (num_class, batch_size)
    // Labels (batch_size)
    // Activations (num_classes, batch_size)
    float computeLoss(float* logits, int* labels, float* activations) const{
        dim3 blockSize(BLOCK_SIZE);
        dim3 gridSize((batch_size + blockSize.x - 1)/blockSize.x);
        crossEntropyLossKernel <<<gridSize, blockSize>>> (logits, labels, activations, loss_classes, batch_size, num_classes);
        cudaDeviceSynchronize();

        // Later parallelize it using reduction algorithms
        float loss_host[batch_size];
        float loss{0};
        cudaMemcpy(loss_host, loss_classes, batch_size*sizeof(float ), cudaMemcpyDeviceToHost);
        for(int i = 0; i < batch_size; ++i){
            loss += loss_host[i];
        }
        return loss/batch_size;
    }

    void computeGradient(float* activations, int* labels, float* delta) {
        dim3 blockSize(BLOCK_SIZE);
        dim3 gridSize((batch_size + blockSize.x - 1)/blockSize.x);
        computeGradientKernel <<< gridSize, blockSize >>> (activations, labels, delta, batch_size, num_classes);
        cudaDeviceSynchronize();
    }

private:
    int num_classes;
    int batch_size;
    float* loss_classes;
};