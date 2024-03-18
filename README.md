# CUDA Multi-Layer Perceptron (MLP) for MNIST Classification

## Overview

This project is an implementation of a 2-layer Multilayer Perceptron (MLP) for classifying MNIST handwritten digit images, written from scratch in CUDA and C++. Designed with modularity and scalability in mind, it showcases the efficient utilization of CUDA kernels for deep learning tasks. This MLP model is capable of handling datasets like MNIST for digit recognition and can be easily scaled or modified for various deep learning applications.

## Features

- **Modular Design:** Each component of the MLP (fully connected layers, ReLU activation, and cross-entropy loss) is implemented in separate modules, making the codebase clean, easy to navigate, and extend.
- **CUDA-accelerated Computation:** Utilizes CUDA kernels for efficient parallel computation of operations such as matrix multiplication, addition, and activation functions, significantly speeding up the training and inference processes.
- **Fully Connected Layers:** Implemented with forward and backward propagation capabilities, supporting gradient descent optimization.
- **Activation Functions:** Includes a ReLU (Rectified Linear Unit) activation layer for introducing non-linearity.
- **Loss Calculation:** Incorporates Cross-Entropy Loss for evaluating model performance and guiding the training process.
- **Data Handling:** Features a data loader for MNIST dataset, facilitating easy data preprocessing and batch management.


## Implementation Details

- **Kernels and Parallelization:** The core of the performance lies in CUDA kernels such as `matrixMulKernel`, `reluForwardKernel`, and `crossEntropyLossKernel`, which execute matrix operations, activation functions, and loss computations in parallel on the GPU.
- **Modular Architecture:** Components like `FullyConnectedLayer`, `ReLU`, and `CrossEntropyLoss` encapsulate specific functionalities, adhering to the principle of single responsibility. This approach not only makes the code more readable but also facilitates easy updates and scalability.
- **Efficient Memory Management:** The code demonstrates effective management of GPU memory, including allocation, transfer between host and device, and deallocation, ensuring optimal resource utilization.
- **Data Loading and Preprocessing:** The `Dataloader` class efficiently handles the loading and preprocessing of the MNIST dataset, converting image data into a format suitable for neural network processing.

## MLP Architecture

The MLP architecture consists of two fully connected layers followed by a softmax layer and cross-entropy loss function:

- Before feeding the images into the MLP, they are flattened into a 1D array. Each 28x28 pixel image is flattened into a 784-dimensional vector.
- Input Layer: 784 neurons (flattened image data)
- First Hidden Layer: 64 neurons with ReLU activation
- Second Hidden Layer: 10 neurons (output layer) with softmax activation
- Loss Function: Cross-entropy loss

Weights are updated using simple gradient descent.

## Parallelization

- Matrix multiplication and vector addition kernels (`matrixMul` and `matrixVecAdd`) are parallelized to utilize GPU cores for faster computation.
- ReLU activation function and its backward pass (`reluForwardKernel` and `reluBackwardKernel`) are parallelized to process large arrays efficiently.
- Cross-entropy loss computation (`crossEntropyLossKernel`) and gradient computation (`computeGradientKernel`) are parallelized to process multiple samples simultaneously.


## Prerequisites

Before you begin, ensure you have met the following requirements:

- NVIDIA CUDA Toolkit (10.x or newer recommended)
- C++ Compiler with C++11 support
- An NVIDIA GPU capable of CUDA computation

## References

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/

