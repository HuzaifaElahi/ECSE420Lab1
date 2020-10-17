#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define NXOR 5

void getOperands(char* line, int* operands) {

    /* get operands */
    operands[0] = atoi(line);
    operands[1] = atoi(line+2);
    operands[2] = atoi(line+4);
    printf("op1: %d, op2: %d, op3: %d\n", operands[0], operands[1], operands[2]);

}

char get_LogicGate_Output(int ops[]) {

    char output;
    int result;

    switch (ops[2]) {

    case AND:
        result = ops[0] & ops[1];
        break;
    case OR:
        result = ops[0] | ops[1];
        break;
    case NAND:
        result = !(ops[0] & ops[1]);
        break;
    case NOR:
        result = !(ops[0] | ops[1]);
        break;
    case XOR:
        result = ops[0] ^ ops[1];
        break;
    case NXOR:
        result = !(ops[0] ^ ops[1]);
        break;
    }

    output = result + '0';
    return output;

}

//int main(int argc, char* argv[]) {
//    FILE* inputFile;
//    FILE* outputFile;
//    char line[100];
//    int len = 6;
//    int operands[3] = { 0 };
//    char output = 'a';
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
//    outputFile = fopen(output_fileName, "w");
//
//    if (inputFile == NULL) {
//        fprintf(stderr, "Error opening file.\n");
//        return 1;
//    }
//
//    clock_t start = clock();
//    printf("%s\n", input_fileName);
//    printf("%s\n", fgets(line, len, inputFile));
//    while (fgets(line, 100, inputFile)) {
//        printf("%s\n", line);
//        getOperands(line, operands);
//        output = get_LogicGate_Output(operands);
//        fprintf(outputFile, "%c\n", output);
//
//    }
//    clock_t end = clock();
//    int time_spent = ((end - start) * 1000) / CLOCKS_PER_SEC;
//
//
//    //free(line);
//    fclose(outputFile);
//    fclose(inputFile);
//
//    printf("Success!, Time: %ds %dms\n", time_spent / 1000, time_spent % 1000);
//
//    return 0;
//
//}