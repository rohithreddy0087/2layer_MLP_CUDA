# CUDA Multi-Layer Perceptron (MLP) for MNIST Classification

## Overview

This project implements a multi-layer perceptron (MLP) for classifying MNIST handwritten digit images using CUDA (Compute Unified Device Architecture). The MLP consists of two fully connected layers with a ReLU activation function in between. It utilizes parallelism provided by CUDA to accelerate computation on GPU.

## Code Structure

The code is organized into several classes:

### DataLoader

- `void loadMNISTData(const std::string& imagesFile, const std::string& labelsFile, float* images, unsigned char* labels)`: Loads MNIST data from files into arrays.

### ReLU

- `void forward(float* input, float* output, int size)`: Applies ReLU activation function forward pass.
- `void backward(float* input, float* output, float* delta, int size)`: Computes gradients using backward pass.

### FullyConnectedLayer

- `FullyConnectedLayer(int input_size, int output_size)`: Constructor to initialize fully connected layer.
- `void initializeWeights()`: Initializes weights and biases.
- `void forward(float* input, float* output, int batch_size)`: Performs forward pass through the layer.
- `void backward(float* input, float* delta_z, float* input_gradients, float* weights_gradients, float* biases_gradients, int batch_size, float learning_rate)`: Performs backward pass and computes gradients.

### CrossEntropyLoss

- `void computeLoss(float* logits, unsigned char* labels, float* activations, float* loss, int batch_size, int num_classes)`: Computes cross entropy loss.
- `void computeGradient(float* activations, unsigned char* labels, float* delta, int batch_size, int num_classes)`: Computes gradients of loss function.

### Main Functionality

- `trainMLP(int num_epochs, int num_examples, int input_size, int hidden_size, int output_size, float learning_rate)`: Trains the MLP for a specified number of epochs.

## Parallelization

- Matrix multiplication and vector addition kernels (`matrixMul` and `matrixVecAdd`) are parallelized to utilize GPU cores for faster computation.
- ReLU activation function and its backward pass (`reluForwardKernel` and `reluBackwardKernel`) are parallelized to process large arrays efficiently.
- Cross-entropy loss computation (`crossEntropyLossKernel`) and gradient computation (`computeGradientKernel`) are parallelized to process multiple samples simultaneously.

## Kernels

### matrixMul

This kernel performs matrix multiplication of input and weights matrices.

### matrixVecAdd

This kernel adds a vector to each row of a matrix.

### reluForwardKernel

This kernel applies ReLU activation function element-wise to an input array.

### reluBackwardKernel

This kernel computes gradients using ReLU backward pass.

### crossEntropyLossKernel

This kernel computes cross-entropy loss for each sample in a batch.

### computeGradientKernel

This kernel computes gradients of the loss function.

## Usage

To train the MLP model, call the `trainMLP` function with desired parameters.

## Dependencies

- CUDA Toolkit
- C++ compiler (supporting C++11)

## References

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/

