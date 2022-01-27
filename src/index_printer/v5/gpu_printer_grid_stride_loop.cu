#include <stdlib.h>
#include <stdio.h>

#define N 8

void cpuPrinter(int nlim) {
    for (int idx = 0; idx < nlim; idx++)
        printf("CPU Prints Idx: %d\n", idx);

    printf("\n");
}

__global__ void gpuPrinter(int nlim) {
    int idx = threadIdx.x +  blockIdx.x * blockDim.x;
    int stride = 2;                                 /* Manually initialized. Not good! */
    
    for(int i = idx; i < nlim; i+=stride)           /* grid-stride loop */
        printf("GPU Prints Idx: %d\n", i);
}

int main(int argc, char **argv) {
    cpuPrinter(N);

    gpuPrinter<<<1, 2>>>(N);      /*  Launch the kernel 2 threads which is less than 8 */

    cudaDeviceSynchronize();

    return(EXIT_SUCCESS);
}