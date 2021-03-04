/*================================================*/
/*================== cudaCode.cu =================*/
/*================================================*/
#include "cudaCode.h"
#include <stdio.h>

/*-----------------------------------------------*/
__global__ void arraySumOnDevice(float *A, float *B, float *C, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { 
        C[idx] = A[idx] + B[idx];
    }
}
/*-----------------------------------------------*/
__host__ void deviceProperties(int deviceIdx) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("GPU device %s with index (%d) is set!\n\n", \
    deviceProp.name, deviceIdx);   
}
