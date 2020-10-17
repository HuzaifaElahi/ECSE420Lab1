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

__global__ void evaluateLogicGate(char* inputBuffer, int inputLength, char* outputBuffer, int inputLineLength, int outputLineLength) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < inputLength) {
        char* logicLine = inputBuffer + inputLineLength * (index);

        int operand1 = logicLine[0] - '0';
        int operand2 = logicLine[2] - '0';
        int gateType = logicLine[4] - '0';
        char result;

        switch (gateType) {
        case NOR:
            result = !(operand1 | operand2);
            break;
        case XOR:
            result = operand1 ^ operand2;
            break;
        case NAND:
            result = !(operand1 & operand2);
            break;
        case AND:
            result = operand1 & operand2;
            break;
        case OR:
            result = operand1 | operand2;
            break;
        case NXOR:
            result = !(operand1 ^ operand2);
            break;
        }

        char* outputLocation = outputBuffer + index * outputLineLength;
        char outputValue = result + '0';

        outputLocation[0] = outputValue;
        outputLocation[1] = '\0';
        outputLocation[2] = '\n';
    }
}

int main(int argc, char* argv[]) {

    if (argc != 4) {
        printf("Error: Please enter the input file path, input file length and output file path when running.\n");
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int inputLineLength = 7;
    int outputLineLength = 3;
    char* inputFileName = argv[1];
    int inputLength = atoi(argv[2]);
    char* outputFileName = argv[3];

    FILE* inputFile = fopen(inputFileName, "r");
    if (inputFile == NULL) {
        fprintf(stderr, "Error opening file.\n");
        return 1;
    }

    int inputSize = inputLength * inputLineLength * sizeof(unsigned char);
    int outputSize = inputLength * outputLineLength * sizeof(unsigned char);

    char* inputBuffer = (char*)malloc(inputSize);

    char buf[7];
    int addressLocation = 0;
    while (fgets(buf, sizeof buf, inputFile) != NULL) {
        strcpy(inputBuffer + addressLocation, buf);
        addressLocation += inputLineLength;
    }

    fclose(inputFile);

    char* cudaBuffer;
    char* outputBuffer;

    cudaMallocManaged(&cudaBuffer, inputSize);
    cudaMallocManaged(&outputBuffer, outputSize);

    for (int i = 0; i < inputSize; i++) {
        cudaBuffer[i] = inputBuffer[i];
    }

    cudaEventRecord(start);
    evaluateLogicGate<<< (inputLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (cudaBuffer, inputLength, outputBuffer, inputLineLength, outputLineLength);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    checkCudaErr(cudaDeviceSynchronize(), "Syncronization");
    checkCudaErr(cudaGetLastError(), "GPU");

    float duration = 0;
    cudaEventElapsedTime(&duration, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    FILE* outputFile = fopen(outputFileName, "w");
    int counter = 0;
    addressLocation = 0;

    while (counter < inputLength) {
        if (counter == inputLength - 1) {
            fprintf(outputFile, "%s", outputBuffer + addressLocation);
        }
        else {
            fprintf(outputFile, "%s\n", outputBuffer + addressLocation);
        }
        addressLocation += outputLineLength;
        counter++;
    }

    fclose(outputFile);

    printf("Completed!\n");
    printf("Time: %.6fms\n", duration);

    return 0;
}