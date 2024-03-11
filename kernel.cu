
#include <iostream>
#include <cmath>
#include <cstdlib>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 16

#include <stdio.h>
#include "fully_connected.h"
#include "relu.h"
#include "loss.h"

void testLoss();
void testfc();

void trainMLP(int num_epochs, int input_size, int hidden_size, int output_size, float learning_rate) {
    const std::string imagesFile = "train-images.idx3-ubyte";
    const std::string labelsFile = "train-labels.idx1-ubyte";
    const int num_examples = 60000; 

    std::vector<std::vector<float>> imagesData;
    std::vector<std::vector<unsigned char>> labelsData;

    loadMNISTData(imagesFile, labelsFile, num_examples, 1, imagesData, labelsData);
    

    float* d_input;
    unsigned char* d_labels;
    float* d_logits;
    float* d_activations;
    float* d_loss;
    float* d_delta1;
    float* d_delta2;
    float* d_input_gradients_fc1;
    float* d_input_gradients_fc2;
    float* d_weights_gradients_fc1;
    float* d_weights_gradients_fc2;
    float* d_biases_gradients_fc1;
    float* d_biases_gradients_fc2;

    cudaMalloc(&d_input, num_examples * input_size * sizeof(float));
    cudaMalloc(&d_labels, num_examples * sizeof(unsigned char));
    cudaMalloc(&d_logits, num_examples * output_size * sizeof(float));
    cudaMalloc(&d_activations, num_examples * output_size * sizeof(float));
    cudaMalloc(&d_loss, num_examples * sizeof(float));
    cudaMalloc(&d_delta1, num_examples * hidden_size * sizeof(float));
    cudaMalloc(&d_delta2, num_examples * output_size * sizeof(float));
    cudaMalloc(&d_input_gradients_fc1, num_examples * input_size * sizeof(float));
    cudaMalloc(&d_input_gradients_fc2, num_examples * hidden_size * sizeof(float));
    cudaMalloc(&d_weights_gradients_fc1, input_size * hidden_size * sizeof(float));
    cudaMalloc(&d_weights_gradients_fc2, hidden_size * output_size * sizeof(float));
    cudaMalloc(&d_biases_gradients_fc1, hidden_size * sizeof(float));
    cudaMalloc(&d_biases_gradients_fc2, output_size * sizeof(float));

    ReLU relu;
    FullyConnectedLayer fc1(input_size, hidden_size);
    FullyConnectedLayer fc2(hidden_size, output_size);
    CrossEntropyLoss loss;

    for (int epoch = 0; epoch < num_epochs; ++epoch){
        fc1.forward(d_input, d_logits, num_examples);
        relu.forward(d_logits, d_activations, num_examples * output_size);
        fc2.forward(d_activations, d_logits, num_examples);

        // Compute loss
        loss.computeLoss(d_logits, d_labels, d_activations, d_loss, num_examples, output_size);

        // Backward pass
        loss.computeGradient(d_activations, d_labels, d_delta2, num_examples, output_size);
        fc2.backward(d_activations, d_delta2, d_input_gradients_fc2, d_weights_gradients_fc2, d_biases_gradients_fc2, num_examples, learning_rate);
        relu.backward(d_logits, d_activations, d_delta2, num_examples * output_size);
        fc1.backward(d_input, d_delta1, d_input_gradients_fc1, d_weights_gradients_fc1, d_biases_gradients_fc1, num_examples, learning_rate);

        // Update weights and biases
        cudaMemcpy(fc1.getWeights(), d_weights_gradients_fc1, input_size * hidden_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(fc1.getBiases(), d_biases_gradients_fc1, hidden_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(fc2.getWeights(), d_weights_gradients_fc2, hidden_size * output_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(fc2.getBiases(), d_biases_gradients_fc2, output_size * sizeof(float), cudaMemcpyDeviceToDevice);
}

// Free device memory
cudaFree(d_input);
cudaFree(d_labels);
cudaFree(d_logits);
cudaFree(d_activations);
cudaFree(d_loss);
cudaFree(d_delta1);
cudaFree(d_delta2);
cudaFree(d_input_gradients_fc1);
cudaFree(d_input_gradients_fc2);
cudaFree(d_weights_gradients_fc1);
cudaFree(d_weights_gradients_fc2);
cudaFree(d_biases_gradients_fc1);
cudaFree(d_biases_gradients_fc2);
}

int main()
{
    testfc();
    return 0;
}

void printMatrix(float* mat, int n, int k) {
    for (int i = 0; i < n * k; i++) {
        printf("%.6f ", mat[i]);
        if ((i + 1) % k == 0) {
            printf("\n");
        }
    }
}

void testfc() {
    const int batch_size = 4;
    const int input_size = 5;
    const int output_size = 4;

    float input[batch_size*input_size] = {
        1.0, -1.0, 1.0, 1.0, 1.0,
        1.0, -1.0, 1.0, 1.0, 1.0,
        1.0, -1.0, 1.0, 1.0, 1.0,
        1.0, -1.0, 1.0, 1.0, 1.0
    };

    float output[batch_size * output_size];

    float* input_d, *output_d;
    int inSize = batch_size * input_size * sizeof(float);
    int outSize = batch_size * output_size * sizeof(float);
    cudaMalloc((void**)&input_d, inSize);
    cudaMalloc((void**)&output_d, inSize);

    cudaMemcpy(input_d, input, inSize, cudaMemcpyHostToDevice);

    FullyConnectedLayer fc1(input_size, output_size);

    fc1.forward(input_d, output_d, batch_size);

    cudaMemcpy(output, output_d, outSize, cudaMemcpyDeviceToHost);
    printMatrix(output, output_size, batch_size);


}

void testLoss() {
    const int batch_size = 4;
    const int num_classes = 4;
    const int input_size = batch_size * num_classes;

    float logits[input_size] = {
        2.0, 1.0, 0.1, 3.0,
        0.1, 3.0, 0.2, 2.5,
        1.5, 2.0, 0.5, 1.2,
        0.8, 1.5, 2.0, 1.0
    };

    unsigned char labels[batch_size] = {
        0,1,2,3
    };

    float activations[input_size], loss[batch_size], delta[input_size];

    float* logits_d, * activations_d, * loss_d, *delta_d;
    unsigned char* labels_d;

    int logitSize = input_size * sizeof(float);
    int labelSize = input_size * sizeof(unsigned char);
    int lossSize = batch_size * sizeof(float);
    cudaMalloc((void**)&logits_d, logitSize);
    cudaMalloc((void**)&labels_d, labelSize);
    cudaMalloc((void**)&activations_d, logitSize);
    cudaMalloc((void**)&delta_d, logitSize);
    cudaMalloc((void**)&loss_d, lossSize);

    cudaMemcpy(logits_d, logits, logitSize, cudaMemcpyHostToDevice);
    cudaMemcpy(labels_d, labels, labelSize, cudaMemcpyHostToDevice);
    
    CrossEntropyLoss lossLayer;
    lossLayer.computeLoss(logits_d, labels_d, activations_d, loss_d, batch_size, num_classes);
    lossLayer.computeGradient(activations_d, labels_d, delta_d, batch_size, num_classes);

    cudaMemcpy(activations, activations_d, logitSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(loss, loss_d, lossSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(delta, delta_d, logitSize, cudaMemcpyDeviceToHost);

    printf("Softmax output:\n");
    for (int i = 0; i < input_size; i++) {
        printf("%.6f ", activations[i]);
        if ((i + 1) % num_classes == 0) {
            printf("\n");
        }
    }

    printf("Loss individual is: \n");
    float total_loss = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        printf("%.6f ", loss[i]);
        total_loss += loss[i];
    }
    printf("\nLoss: %.6f\n", total_loss/batch_size);

    printf("Delta output:\n");
    for (int i = 0; i < input_size; i++) {
        printf("%.6f ", delta[i]);
        if ((i + 1) % num_classes == 0) {
            printf("\n");
        }
    }
}