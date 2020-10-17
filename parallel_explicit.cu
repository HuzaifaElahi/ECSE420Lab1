#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <device_launch_parameters.h>
#include <string.h>

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define NXOR 5
#define THREADS_PER_BLOCK 1024

inline cudaError_t checkCudaErr(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime error at %s: %s\n", msg, cudaGetErrorString(err));
    }
    return err;
}


__global__ void doLogicGates(char* inputBuffer, int inputLength, char* outputBuffer, int inputLineLength, int outputLineLength) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < inputLength) {
        char* logicLine = inputBuffer + inputLineLength * (index);

        int operand1 = logicLine[0] - '0';
        int operand2 = logicLine[2] - '0';
        int operatorr = logicLine[4] - '0';
        int output;

        switch (operatorr) {
        case 0:
            output = operand1 & operand2;
            break;
        case 1:
            output = operand1 | operand2;
            break;
        case 2:
            output = !(operand1 & operand2);
            break;
        case 3:
            output = !(operand1 | operand2);
            break;
        case 4:
            output = operand1 ^ operand2;
            break;
        case 5:
            output = !(operand1 ^ operand2);
            break;
        }



        char* outputLocation = outputBuffer + index * outputLineLength;
        char outputValue = output + '0';


        outputLocation[0] = outputValue;
        outputLocation[1] = '\0';
        outputLocation[2] = '\n';
    }
}

//int main(int argc, char* argv[]) {
//
//    FILE* inputFile;
//    FILE* outputFile;
//
//    int inputLineLength = 7;
//    int outputLineLength = 3;
//
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    float milliseconds = 0;
//    float copyMilliseconds = 0;
//
//    if (argc != 4) {
//        printf("Error: Please enter the input file path, input file length and output file path when running.\n");
//        return 1;
//    }
//
//    char* input_fileName = argv[1];
//    int inputLength = atoi(argv[2]);
//    char* output_fileName = argv[3];
//
//    inputFile = fopen(input_fileName, "r");
//    if (inputFile == NULL) {
//        fprintf(stderr, "Error opening file.\n");
//        return 1;
//    }
//
//    int inputSize = inputLength * inputLineLength * sizeof(unsigned char);
//    int outputSize = inputLength * outputLineLength * sizeof(unsigned char);
//
//    char* inputBuffer = (char*)malloc(inputSize);
//
//    char buf[7];
//    int addressLocation = 0;
//    while (fgets(buf, sizeof buf, inputFile) != NULL) {
//        strcpy(inputBuffer + addressLocation, buf);
//        addressLocation += inputLineLength;
//    }
//
//    fclose(inputFile);
//
//    char* cudaBuffer;
//    char* outputBuffer;
//    char* returnBuffer;
//    returnBuffer = (char*)malloc(outputSize * sizeof(char));
//
//    cudaMalloc(&cudaBuffer, inputSize);
//    cudaMalloc(&outputBuffer, outputSize);
//
//
//    clock_t start_copy = clock();
//    cudaMemcpy(cudaBuffer, inputBuffer, inputSize, cudaMemcpyHostToDevice);
//    clock_t end_copy = clock();
//    float time_spent = ((end_copy - start_copy) * 1000.0) / CLOCKS_PER_SEC;
//    printf("Copying Time elapsed: %.6fms\n", time_spent);
//
//    // Run kernel and record time
//    cudaEventRecord(start);
//    doLogicGates << < (inputLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (cudaBuffer, inputLength, outputBuffer, inputLineLength, outputLineLength);
//
//    cudaEventRecord(stop);
//    cudaMemcpy(returnBuffer, outputBuffer, outputSize, cudaMemcpyDeviceToHost);
//    cudaEventSynchronize(stop);
//
//    checkCudaErr(cudaDeviceSynchronize(), "Syncronization");
//    checkCudaErr(cudaGetLastError(), "GPU");
//
//    cudaEventElapsedTime(&milliseconds, start, stop);
//    cudaEventDestroy(start);
//    cudaEventDestroy(stop);
//
//    outputFile = fopen(output_fileName, "w");
//    int counter = 0;
//    addressLocation = 0;
//
//    while (counter < inputLength) {
//        if (counter == inputLength - 1) {
//            fprintf(outputFile, "%s", returnBuffer + addressLocation);
//        }
//        else {
//            fprintf(outputFile, "%s\n", returnBuffer + addressLocation);
//        }
//        addressLocation += outputLineLength;
//        counter++;
//    }
//
//    fclose(outputFile);
//
//
//    printf("Time elapsed: %.6fms\n", milliseconds);
//    printf("Success!\n");
//
//    return 0;
//}