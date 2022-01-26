#include <stdlib.h>
#include <stdio.h>

#define N 8

void cpuPrinter(int nlim) {
    for (int idx = 0; idx < nlim; idx++)
        printf("CPU Prints Idx: %d\n", idx);
}

int main(int argc, char **argv) {
    cpuPrinter(N);

    return(EXIT_SUCCESS);
}
