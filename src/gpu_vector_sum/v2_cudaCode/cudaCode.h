/*================================================*/
/*================== cudaCode.h ==================*/
/*================================================*/
#ifndef CUDACODE_H
#define CUDACODE_H

__global__ void arraySumOnDevice(float *A, float *B, float *C, const int size);

#endif // CUDACODE_H

