
#include <iostream>
#include <cmath>
#include <cstdlib>
#include "cuda_runtime.h"

#define BLOCK_SIZE 32

#include <stdio.h>
#include "fully_connected.h"
#include "relu.h"
#include "loss.h"
#include "testing.h"
#include "load_data.h"

class Model {
public:
    Model(int ainput_size, int ahidden_size, int anum_classes, int abatch_size, float alearning_rate):
        input_size(ainput_size), hidden_size(ahidden_size), num_classes(anum_classes), batch_size(abatch_size), learning_rate(alearning_rate),
        fc1(ainput_size, ahidden_size, abatch_size, 1), relu(ahidden_size, batch_size), fc2(ahidden_size, num_classes, abatch_size, 2),
        lossLayer(anum_classes, abatch_size){
        size_t size_1{hidden_size*batch_size*sizeof(float)};
        size_t size_2{num_classes*batch_size*sizeof(float)};

        cudaMalloc((void**)&Z_1, size_1);
        cudaMalloc((void**)&dZ_1, size_1);
        cudaMalloc((void**)&A_1, size_1);
        cudaMalloc((void**)&dA_1, size_1);

        cudaMalloc((void**)&Z_2, size_2);
        cudaMalloc((void**)&dZ_2, size_2);
        cudaMalloc((void**)&A_2, size_2);
    }

    ~Model(){
        cudaFree(Z_1);
        cudaFree(dZ_1);
        cudaFree(A_1);
        cudaFree(dA_1);

        cudaFree(Z_2);
        cudaFree(dZ_2);
        cudaFree(A_2);
    }

    void train(float* A_0, int* labels, float& lossVal, float& accuracy) {

        int* labels_d;
        cudaMalloc((void**)&labels_d, batch_size* sizeof(int));
        cudaMemcpy(labels_d, labels, batch_size* sizeof(int), cudaMemcpyHostToDevice);

        // Forward Propagation
        fc1.forward(A_0, Z_1);
        relu.forward(Z_1, A_1);
        fc2.forward(A_1, Z_2);
        lossVal += lossLayer.computeLoss(Z_2, labels_d, A_2);

        // Back Propagation
        lossLayer.computeGradient(A_2, labels_d, dZ_2);
        fc2.backward(dZ_2, dA_1, learning_rate);
        relu.backward(dA_1, dZ_1);
        fc1.backward(dZ_1, nullptr, learning_rate);

        cudaFree(labels_d);

        accuracy += compute_accuracy(A_2, labels);

    }

    float test(float* A_0, int* labels) {

        fc1.forward(A_0, Z_1);
        relu.forward(Z_1, A_1);
        fc2.forward(A_1, Z_2);

        return compute_accuracy(Z_2, labels);
    }

    float compute_accuracy(float* activations, int* labels) const{

        float correct_predictions = 0;
        float A_2_h[batch_size * num_classes];
        cudaMemcpy(A_2_h, activations, batch_size * num_classes * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < batch_size; ++i) {
            int max_index = 0;
            float max_activation = A_2_h[i];
            for (int j = 1; j < num_classes; ++j) {
                if (A_2_h[j * batch_size + i] > max_activation) {
                    max_index = j;
                    max_activation = A_2_h[j * batch_size + i];
                }
            }
            if (labels[i] == max_index) {
                correct_predictions++;
            }
        }

        return correct_predictions / static_cast<float>(batch_size);
    }

private:
    int input_size;
    int hidden_size;
    int num_classes;
    int batch_size;
    float learning_rate;

    float* Z_1;
    float* dZ_1;
    float* A_1;
    float* dA_1;

    float* Z_2;
    float* dZ_2;
    float* A_2;

    FullyConnectedLayer fc1;
    ReLU relu;
    FullyConnectedLayer fc2;
    CrossEntropyLoss lossLayer;
};


int main()
{
    std::string trainImagesFile = "/home/rrr/MLP/MLP/train-images.idx3-ubyte";
    std::string trainLabelsFile = "/home/rrr/MLP/MLP/train-labels.idx1-ubyte";

    std::string testImagesFile = "/home/rrr/MLP/MLP/t10k-images.idx3-ubyte";
    std::string testLabelsFile = "/home/rrr/MLP/MLP/t10k-labels.idx1-ubyte";

    int train_batch_size = 64;
    int test_batch_size = 64;

    Dataloader trainDataLoader(trainImagesFile, trainLabelsFile, train_batch_size);
    Dataloader testDataLoader(testImagesFile, testLabelsFile, test_batch_size);

    int trainExamples = trainDataLoader.getNumExamples();
    int testExamples = testDataLoader.getNumExamples();
    int inputSize = trainDataLoader.getInputSize();
    int num_classes = trainDataLoader.getNumClasses();

    std::cout<<"Loaded Training and Testing data\n";
    std::cout<<"Number of Training Examples: "<<trainExamples<<"\n";
    std::cout<<"Number of Testing Examples: "<<testExamples<<"\n";
    std::cout<<"Input Size: "<<inputSize<<"\n";
    std::cout<<"Number of Classes: "<<num_classes<<"\n";

    int epochs = 100;
    float learning_rate = 0.01;
    int hiddenSize = 256;
    Model model(inputSize, hiddenSize, num_classes, train_batch_size, learning_rate);

    for(int epoch = 0; epoch < epochs; ++epoch){
        float loss_epoch = 0;
        float acc_epoch = 0;
        int num_batches_train = trainExamples / train_batch_size;
        for (int batch = 0; batch < num_batches_train-1; ++batch) {
            float input[train_batch_size*inputSize];
            int label[train_batch_size];
            trainDataLoader.getBatch(batch, input, label);
            model.train(input, label, loss_epoch, acc_epoch);
        }
        loss_epoch /= num_batches_train;
        acc_epoch /= num_batches_train;

        float test_acc_epoch = 0;
        int num_batches_test = testExamples / test_batch_size;
        for (int batch = 0; batch < num_batches_test-1; ++batch) {
            float input[test_batch_size*inputSize];
            int label[test_batch_size];
            testDataLoader.getBatch(batch, input, label);
            float acc_batch = model.test(input, label);
            test_acc_epoch += acc_batch;
        }
        test_acc_epoch /= num_batches_test;

        std::cout<<"Epoch "<<epoch+1<<"/"<<epochs<<" Loss: "<<loss_epoch<<" Test Accuracy: "<<test_acc_epoch<<"\n";
    }

    return 0;
}


//int main(){
//    testLoss();
//    return 0;
//}