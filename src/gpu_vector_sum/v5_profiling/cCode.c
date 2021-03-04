/*================================================*/
/*==================== cCode.h ===================*/
/*================================================*/
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "cCode.h"

void dataInitializer(float *inputArray, int size) {
    /* Generating float-type random numbers 
     * between 0.0 and 1.0
     */
    time_t t;
    srand( (unsigned int) time(&t) );

    for (int i = 0; i < size; i++) {
        inputArray[i] = ( (float)rand() / (float)(RAND_MAX) ) * 1.0;
    }

    return;
}
/*-----------------------------------------------*/
void arraySumOnHost(float *A, float *B, float *C, const int size) {
    for (int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}
/*-----------------------------------------------*/
void arrayEqualityCheck(float *hostPtr, float *devicePtr, const int size) {
    double tolerance = 1.0E-8;
    bool isEqual = true;

    for (int i = 0; i < size; i++) {
        if (abs(hostPtr[i] - devicePtr[i]) > tolerance) {
            isEqual = false;
            printf("Arrays are NOT equal because:\n");
            printf("at %dth index: hostPtr[%d] = %5.2f \
            and devicePtr[%d] = %5.2f;\n", \
            i, i, hostPtr[i], i, devicePtr[i]);
            break;
        }
    }

    if (isEqual) {
        printf("Arrays are equal.\n\n");
    }

    return;
}
