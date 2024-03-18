//#pragma once
//
//
#include <iostream>
#include <fstream>
#include <vector>

void plotImage(const std::vector<unsigned char>& image_vec) {
    const int image_size = 28;

    for (int i = 0; i < image_size; ++i) {
        for (int j = 0; j < image_size; ++j) {
            char pixel_char;
            if (image_vec[i * image_size + j] > 0.5) {
                pixel_char = '#';
            } else {
                pixel_char = '.';
            }
            std::cout << pixel_char << ' ';
        }
        std::cout << std::endl;
    }
}

class Dataloader{
public:
    Dataloader(std::string& aimgfilename, std::string& alabelfilename, int abatch_size):
            imgfilename(aimgfilename), labelfilename(alabelfilename), batch_size(abatch_size){
        loadMNISTData();
    }

    void getBatch(int batch, float* input, int* label){
        int start = batch * batch_size;
        int end = (batch + 1) * batch_size;

        for (int example = start; example < end; ++example) {
            for (int i = 0; i < inputSize; ++i) {
                input[(example - start) * inputSize + i] = static_cast<float>(images[example][i]) / 255.0f;
            }
//            plotImage(images[example]);
//            std::cout<<"Label: \n"<<static_cast<int>(labels[example]);
            label[example - start] = static_cast<int>(labels[example]);
        }
    }

    int getNumExamples() const {
        return numExamples;
    }

    int getInputSize() const {
        return inputSize;
    }

    int getNumClasses() const {
        return num_classes;
    }

private:
    std::string& imgfilename;
    std::string& labelfilename;
    int batch_size{0};
    int numExamples{0};

    int inputSize{0};
    int num_classes{10};

    std::vector<std::vector<unsigned char>> images;
    std::vector<unsigned char> labels;

    void loadMNISTData() {

        std::ifstream imgfile(imgfilename, std::ios::binary);
        if (!imgfile) {
            std::cerr << "Failed to open file: " << imgfilename << std::endl;
            exit(1);
        }

        std::ifstream lblfile(labelfilename, std::ios::binary);
        if (!lblfile) {
            std::cerr << "Failed to open file: " << labelfilename << std::endl;
            exit(1);
        }

        int magic_number, num_rows, num_cols;
        imgfile.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        imgfile.read(reinterpret_cast<char*>(&numExamples), sizeof(numExamples));
        imgfile.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
        imgfile.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));

        magic_number = __builtin_bswap32(magic_number);
        numExamples = __builtin_bswap32(numExamples);
        num_rows = __builtin_bswap32(num_rows);
        num_cols = __builtin_bswap32(num_cols);
        inputSize = num_rows*num_cols;
        int lbl_magic_number, num_labels;
        lblfile.read(reinterpret_cast<char*>(&lbl_magic_number), sizeof(lbl_magic_number));
        lblfile.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));

        lbl_magic_number = __builtin_bswap32(lbl_magic_number);
        num_labels = __builtin_bswap32(num_labels);

        images.resize(numExamples, std::vector<unsigned char>(inputSize));
        labels.resize(num_labels);
        for (int i = 0; i < numExamples; ++i) {
            imgfile.read(reinterpret_cast<char*>(images[i].data()), inputSize);
        }
        lblfile.read(reinterpret_cast<char*>(labels.data()), num_labels);
    }

};

