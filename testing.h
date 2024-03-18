//
// Created by rrr on 3/16/24.
//

#ifndef MLP_TESTING_H
#define MLP_TESTING_H

void printMatrixHost(float* mat, int n, int k) {
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

//    float output[batch_size * output_size];

    float* input_d, *output_d;
    int inSize = batch_size * input_size * sizeof(float);
    int outSize = batch_size * output_size * sizeof(float);
    cudaMalloc((void**)&input_d, inSize);
    cudaMalloc((void**)&output_d, inSize);

    cudaMemcpy(input_d, input, inSize, cudaMemcpyHostToDevice);

    FullyConnectedLayer fc1(input_size, output_size, batch_size, 2);

    fc1.forward(input_d, output_d);

    float a_l[batch_size*output_size] = {
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0
    };
    float* al_d;
    cudaMalloc((void**)&al_d, outSize);
    cudaMemcpy(al_d, a_l, outSize, cudaMemcpyHostToDevice);
    fc1.backward(al_d, output_d,  0.001);
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

    int labels[batch_size] = {
            0,1,2,3
    };

    float activations[input_size], delta[input_size];

    float* logits_d, * activations_d, *delta_d;
    int* labels_d;

    int logitSize = input_size * sizeof(float);
    int labelSize = input_size * sizeof(unsigned char);
//    int lossSize = batch_size * sizeof(float);
    cudaMalloc((void**)&logits_d, logitSize);
    cudaMalloc((void**)&labels_d, labelSize);
    cudaMalloc((void**)&activations_d, logitSize);
    cudaMalloc((void**)&delta_d, logitSize);

    cudaMemcpy(logits_d, logits, logitSize, cudaMemcpyHostToDevice);
    cudaMemcpy(labels_d, labels, labelSize, cudaMemcpyHostToDevice);

    CrossEntropyLoss lossLayer(num_classes, batch_size);
    float loss_d{0};
    loss_d = lossLayer.computeLoss(logits_d, labels_d, activations_d );
    lossLayer.computeGradient(activations_d, labels_d, delta_d);

    cudaMemcpy(activations, activations_d, logitSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(delta, delta_d, logitSize, cudaMemcpyDeviceToHost);

    printf("Activations:\n");
    for (int i = 0; i < input_size; i++) {
        printf("%.6f ", activations[i]);
        if ((i + 1) % num_classes == 0) {
            printf("\n");
        }
    }

//    printf("Loss individual is: \n");
//    float total_loss = 0.0f;
//    for (int i = 0; i < batch_size; i++) {
//        printf("%.6f ", loss[i]);
//        total_loss += loss[i];
//    }
    printf("\nLoss: %.6f\n", loss_d);

    printf("Delta output:\n");
    for (int i = 0; i < input_size; i++) {
        printf("%.6f ", delta[i]);
        if ((i + 1) % num_classes == 0) {
            printf("\n");
        }
    }
}
#endif //MLP_TESTING_H
