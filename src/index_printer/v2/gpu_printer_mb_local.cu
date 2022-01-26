#include <stdlib.h>
#include <stdio.h>

#define N 8

void cpuPrinter(int nlim) {
    for (int idx = 0; idx < nlim; idx++)
        printf("CPU Prints Idx: %d\n", idx);

    printf("\n");
}

__global__ void gpuPrinter(void) {
    int idx = threadIdx.x;                                  /* Note that the local index (threadIdx.x) is used */
    printf("GPU Prints Idx: %d\n", idx);                    /* Write the kernel for individual threads */
}

int main(int argc, char **argv) {
    cpuPrinter(N);

    gpuPrinter<<<2,N/2>>>();      /*  Launch the kernel with two blocks threads */
                                  
    cudaDeviceSynchronize();

    return(EXIT_SUCCESS);
}