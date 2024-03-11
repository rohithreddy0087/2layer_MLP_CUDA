#pragma once


#include <iostream>
#include <fstream>
#include <vector>

const int IMAGE_SIZE = 28 * 28;

void loadMNISTData(const std::string& imagesFile, const std::string& labelsFile, int numExamples, int numDatasets,
    std::vector<std::vector<float>>& imagesData, std::vector<std::vector<unsigned char>>& labelsData) {
    std::ifstream imagesStream(imagesFile, std::ios::binary);
    if (!imagesStream.is_open()) {
        std::cerr << "Failed to open images file: " << imagesFile << std::endl;
        return;
    }

    std::ifstream labelsStream(labelsFile, std::ios::binary);
    if (!labelsStream.is_open()) {
        std::cerr << "Failed to open labels file: " << labelsFile << std::endl;
        return;
    }

    for (int n = 0; n < numDatasets; ++n) {
        std::vector<float> images(numExamples * IMAGE_SIZE);
        std::vector<unsigned char> labels(numExamples);

        for (int i = 0; i < numExamples; ++i) {
            imagesStream.seekg(16 + n * numExamples * IMAGE_SIZE + i * IMAGE_SIZE);
            imagesStream.read(reinterpret_cast<char*>(&images[i * IMAGE_SIZE]), IMAGE_SIZE);

            labelsStream.seekg(8 + n * numExamples + i);
            labelsStream.read(reinterpret_cast<char*>(&labels[i]), sizeof(unsigned char));
        }

        imagesData.push_back(images);
        labelsData.push_back(labels);
    }

    imagesStream.close();
    labelsStream.close();
}