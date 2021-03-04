/*================================================*/
/*================== cudaCode.h ==================*/
/*================================================*/
#ifndef CUDACODE_H
#define CUDACODE_H

__global__ void arraySumOnDevice(float *A, float *B, float *C, const int size);
__host__ void deviceProperties(int deviceIdx);

#endif // CUDACODE_H
