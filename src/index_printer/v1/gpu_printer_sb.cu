#include <stdlib.h>
#include <stdio.h>

#define N 8

void cpuPrinter(int nlim) {
    for (int idx = 0; idx < nlim; idx++)
        printf("CPU Prints Idx: %d\n", idx);

    printf("\n");
}

__global__ void gpuPrinter(void) {
    int idx = threadIdx.x;
    printf("GPU Prints Idx: %d\n", idx);        /* Write the kernel for individual threads */
}

int main(int argc, char **argv) {
    cpuPrinter(N);

    gpuPrinter<<<1,N>>>();      /*  Launch the kernel with a single block of threads */
                                /*  The kernel launch will be fine as long as N <= 1024 */
    cudaDeviceSynchronize();

    return(EXIT_SUCCESS);
}
