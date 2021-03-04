#include <stdlib.h>                       /* For status marcos */
#include <stdio.h>                        /* For printf() function */

/**********************************************/

void helloFromCPU(void) {                 /* This function runs on the host */
    printf("Hello World from CPU!\n");  
}

__global__ void helloFromGPU() {          /* This function runs on the device */
    printf("Hello World from GPU!\n");
}

/**********************************************/

int main(int argc, char **argv) {
    helloFromCPU();                       /* Calling from host */

    helloFromGPU<<<1, 8>>>();             /* Calling from device */

    cudaDeviceReset();                    /* House-keeping on device */

    return(EXIT_SUCCESS);
}