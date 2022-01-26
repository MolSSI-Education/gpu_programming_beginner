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
    
    if(idx < nlim)                 /* Make sure the global index does not go beyond the limit */
        printf("GPU Prints Idx: %d\n", idx);
}

int main(int argc, char **argv) {
    cpuPrinter(N);

    gpuPrinter<<<4,N>>>(N);      /*  Launch the kernel with 32 threads (4 blocks with 8 threads) */
                                /*  The number of dispatched threads (32) is greater than N */
    cudaDeviceSynchronize();

    return(EXIT_SUCCESS);
}