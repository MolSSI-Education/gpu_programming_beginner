#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void printThreadID() {
    /* For each thread, the kernel prints
     * the threadIdx, blockIdx, blockDim,
     * and gridDim, respectively.
     */
    printf("threadIdx:(%d, %d, %d), \
            blockIdx:(%d, %d, %d), \
            blockDim:(%d, %d, %d), \
            gridDim:(%d, %d, %d)\n", \
            threadIdx.x, threadIdx.y, threadIdx.z, \
            blockIdx.x, blockIdx.y, blockIdx.z, \
            blockDim.x, blockDim.y, blockDim.z, \
            gridDim.x, gridDim.y, gridDim.z);

}

int main(int argc, char **argv)
{
    /* Array size */
    int numArray  = 6;

    /* Number of threads in blocks */
    int numBlocks = 2;

    /* Organizing grids and blocks */
    dim3 block(numBlocks);
    dim3 grid((numArray + block.x - 1) / block.x);

    /* Let the user know that the dimensions will be printed from the host */
    printf("Printing from the host!\n");

    /* Print the grid and block dimensions from the host */
    printf("[grid.x, grid.y, grid.z]:    [%d, %d, %d]\n", grid.x, grid.y, grid.z);
    printf("[block.x, block.y, block.z]: [%d, %d, %d]\n\n", block.x, block.y, block.z);

    /* Indicate that the dimensions will now be printed from the device */
    printf("Printing from the device!\n");

    /* Print the grid and block dimensions from the device */
    printThreadID<<<grid, block>>>();

    /* Performing house-keeping for the device */
    cudaDeviceReset();

    return(EXIT_SUCCESS);
}