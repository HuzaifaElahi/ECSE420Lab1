
#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <cstring>
#include "device_launch_parameters.h"

__global__ void logic_gate(int gate1, int gate2, int op) {
    /*
    while () {
        // Call the logic_gate function
        int gate1 = data[0] - '0';
        int gate2 = data[2] - '0';
        int op = data[4] - '0';
        logic_gate << <1, 1 >> > (gate1, gate2, op);

    }*/
}


int process(char* input_filename, int file_length, char* output_filename) {

    // Allocate space for the data
    char** data = (char**) malloc(file_length * sizeof(char*));

    FILE* input = fopen(input_filename, "r");
    FILE* output = fopen(output_filename, "w");

    for (int i = 0; i < file_length; i++) {
        data[i] = (char*)malloc(16);
        fgets(data[i], sizeof(16), input);
    }
   
    for (int i = 0; i < file_length; i++) {
        free(data[i]);
    }
    free(data);


    fclose(input);
    fclose(output);
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc == 4) {
        // User should pass absolute file paths
        char* input_filename = argv[1];
        int file_length = atoi(argv[2]);
        char* output_filename = argv[3];

        int error = process(input_filename, file_length, output_filename);

        if (error != 0) {
            printf("Error %d \n", error);

        }
        else {
            printf("Success .\n");
        }
    }
    else {
        printf("Expected three arguments : input_filename file_length output_filename \n");
    }
    return 0;
}