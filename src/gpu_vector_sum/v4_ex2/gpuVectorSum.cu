/*================================================*/
/*================ gpuVectorSum.cu ===============*/
/*================================================*/
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "cudaCode.h"
extern "C" {
    #include "cCode.h"
}

/*************************************************/
int main(int argc, char **argv) {
    printf("Kicking off %s\n\n", argv[0]);

    /* Device setup */
    int deviceIdx = 0;
    ERRORHANDLER(cudaSetDevice(deviceIdx));

    /* Device properties */
    deviceProperties(deviceIdx);
/*-----------------------------------------------*/
    /* Fixing the vector size to 1 * 2^24 = 16777216 (64 MB) */
    int vecSize = 1 << 24;
    size_t vecSizeInBytes = vecSize * sizeof(float);
    printf("Vector size: %d floats (%lu MB)\n\n", vecSize, vecSizeInBytes/1024/1024);

    /* Memory allocation on the host */
    float *h_A, *h_B, *hostPtr, *devicePtr;
    h_A     = (float *)malloc(vecSizeInBytes);
    h_B     = (float *)malloc(vecSizeInBytes);
    hostPtr = (float *)malloc(vecSizeInBytes);
    devicePtr  = (float *)malloc(vecSizeInBytes);

    double tStart, tElapsed;

    /* Vector initialization on the host */
    tStart = chronometer();
    dataInitializer(h_A, vecSize);
    dataInitializer(h_B, vecSize);
    tElapsed = chronometer() - tStart;
    printf("Elapsed time for dataInitializer: %f second(s)\n", tElapsed);
    memset(hostPtr, 0, vecSizeInBytes);
    memset(devicePtr,  0, vecSizeInBytes);

    /* Vector summation on the host */
    tStart = chronometer();
    arraySumOnHost(h_A, h_B, hostPtr, vecSize);
    tElapsed = chronometer() - tStart;
    printf("Elapsed time for arraySumOnHost: %f second(s)\n", tElapsed);
/*-----------------------------------------------*/
    /* (Global) memory allocation on the device */
    float *d_A, *d_B, *d_C;
    ERRORHANDLER(cudaMalloc((float**)&d_A, vecSizeInBytes));
    ERRORHANDLER(cudaMalloc((float**)&d_B, vecSizeInBytes));
    ERRORHANDLER(cudaMalloc((float**)&d_C, vecSizeInBytes));

    /* Data transfer from host to device */
    ERRORHANDLER(cudaMemcpy(d_A, h_A, vecSizeInBytes, cudaMemcpyHostToDevice));
    ERRORHANDLER(cudaMemcpy(d_B, h_B, vecSizeInBytes, cudaMemcpyHostToDevice));
    ERRORHANDLER(cudaMemcpy(d_C, devicePtr, vecSizeInBytes, cudaMemcpyHostToDevice));

    /* Organizing grids and blocks */
    int numThreadsInBlocks = 1024;
    dim3 block (numThreadsInBlocks);
    dim3 grid  ((vecSize + block.x - 1) / block.x);

    /* Execute the kernel from the host*/
    tStart = chronometer();
    arraySumOnDevice<<<grid, block>>>(d_A, d_B, d_C, vecSize);
    ERRORHANDLER(cudaDeviceSynchronize());
    tElapsed = chronometer() - tStart;
    printf("Elapsed time for arraySumOnDevice <<< %d, %d >>>: %f second(s) \n\n", \
    grid.x, block.x, tElapsed);
/*-----------------------------------------------*/
    /* Returning the last error from a runtime call */
    ERRORHANDLER(cudaGetLastError());

    /* Data transfer back from device to host */
    ERRORHANDLER(cudaMemcpy(devicePtr, d_C, vecSizeInBytes, cudaMemcpyDeviceToHost));

    /* Check to see if the array summations on 
     * CPU and GPU yield the same results 
     */
    arrayEqualityCheck(hostPtr, devicePtr, vecSize);
/*-----------------------------------------------*/
    /* Free the allocated memory on the device */
    ERRORHANDLER(cudaFree(d_A));
    ERRORHANDLER(cudaFree(d_B));
    ERRORHANDLER(cudaFree(d_C));

    /* Free the allocated memory on the host */
    free(h_A);
    free(h_B);
    free(hostPtr);
    free(devicePtr);

    return(EXIT_SUCCESS);
}
