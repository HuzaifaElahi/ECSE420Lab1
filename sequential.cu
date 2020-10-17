//#include <string>
//#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>
//
//#define AND 0
//#define OR 1
//#define NAND 2
//#define NOR 3
//#define XOR 4
//#define NXOR 5
//
///*
//* Evaluates logical expression.
//*/
//char evaluateLogicGate(int ops[]) {
//    int result=0;
//
//    switch (ops[2]) {
//    case NOR:
//        result = !(ops[0] | ops[1]);
//        break;
//    case XOR:
//        result = ops[0] ^ ops[1];
//        break;
//    case NAND:
//        result = !(ops[0] & ops[1]);
//        break;
//    case AND:
//        result = ops[0] & ops[1];
//        break;
//    case OR:
//        result = ops[0] | ops[1];
//        break;
//    case NXOR:
//        result = !(ops[0] ^ ops[1]);
//        break;
//    }
//
//    char output = result + '0';
//    return output;
//
//}
//
///*
//* Sequential computation of list of logical expressions.
//* Takes input file, file length and output file name as arguments.
//*/
//int main(int argc, char* argv[]) {
//    if (argc != 4) {
//        printf("Error: Please enter the input file path, input file length and output file path when running.\n");
//        return 1;
//    }
//    char line[10];
//    char* inputFileName = argv[1];
//    int inputLength = atoi(argv[2]);
//    char* outputFileName = argv[3];
//
//    FILE* inputFile = fopen(inputFileName, "r");
//    FILE* outputFile = fopen(outputFileName, "w");
//
//    if (inputFile == NULL) {
//        fprintf(stderr, "Error opening file.\n");
//        return 1;
//    }
//
//    clock_t start = clock();
//
//    int operands[3] = { 0 };
//    char result = NULL;
//    while (fgets(line, 10, inputFile)) {
//        operands[0] = atoi(line);
//        operands[1] = atoi(line + 2);
//        operands[2] = atoi(line + 4);
//        result = evaluateLogicGate(operands);
//        fprintf(outputFile, "%c\n", result);
//
//    }
//    clock_t end = clock();
//    int duration = ((end - start) * 1000) / CLOCKS_PER_SEC;
//
//    fclose(outputFile);
//    fclose(inputFile);
//
//    printf("Completed\nTime: %ds %dms\n", duration / 1000, duration % 1000);
//
//    return 0;
//
//}