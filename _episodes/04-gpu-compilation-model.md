---
title: "CUDA GPU Compilation Model"
teaching: 30
exercises: 0
questions:
- "Container"
objectives:
- "Container"
keypoints:
- "Container"
---

- [1. NVIDIA's CUDA Compiler](#1-nvidias-cuda-compiler)
- [2. Compiling Separate Source Files using NVCC](#2-compiling-separate-source-files-using-nvcc)
- [3. Error Handling](#3-error-handling)

## 1. NVIDIA's CUDA Compiler

*NVIDIA's CUDA compiler (NVCC)* is distributed as part of CUDA Toolkit and 
is based upon poplar [*LLVM*](https://llvm.org/) open source infrastructure.
Each CUDA program is a combination of host code written in C/C++ standard semantics with some extensions within CUDA API as well as the GPU device 
kernel functions. The nvcc compiler driver separates the host code from that of the device. The host code is then preprocessed and compiled with the 
host [C++ compilers supported by nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#supported-host-compilers).
The nvcc compiler also compiles the device kernel functions using the 
proprietary NVIDIA assembler and compilers. Then, nvcc embeds 
the GPU kernels as [*fatbinary*](https://en.wikipedia.org/wiki/Fat_binary) images into the host object files. Finally, 
during the linking stage, CUDA runtime libraries are added for kernel
procedure calls and memory and data transfer management.
The description of the exact details of the [compilation phases](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#supported-phases) 
is beyond the scope and intended level of this tutorial. The interested
reader is referred to [CUDA Toolkit documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#introduction) and [*parallel thread execution (PTX)* compiler API](https://docs.nvidia.com/cuda/ptx-compiler-api/index.html) and [*instruction set architecture (ISA)*](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) for
further details.

## 2. Compiling Separate Source Files using NVCC

In the previous lesson, where the summation of arrays on GPUs example was
presented, we mentioned that since the source code was a long one. Therefore, we
need to break it into different smaller source files according to the 
logical structure of our code.

Before CUDA 5.0, breaking the CUDA program code into separate files and 
[multiple-source file compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#using-separate-compilation-in-cuda) was not possible and the only available option was the whole-code compilation approach. Since we are using CUDA 11.2 here, this should not be any problem.

Getting back to our 
[Summation of Arrays on GPUs]({{site.baseurl}}{% link _episodes/03-cuda-program-model.md %}#3-summation-of-arrays-on-gpus)
example, let's start breaking the code into separate source files by copying the C function
signatures and pasting them into an empty file. Then rename the file to ***cCode.h*** and add the necessary include header preprocessor expressions and header guards to it. The resulting header file's content should be the same as the following code block

~~~
#ifndef CCODE_H
#define CCODE_H

#include <time.h>
#include <sys/time.h>

/*************************************************/
inline double chronometer() {
    struct timezone tzp;
    struct timeval tp;
    int tmp = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
/*-----------------------------------------------*/
void dataInitializer(float *inputArray, int size);
void arraySumOnHost(float *A, float *B, float *C, const int size);
void arrayEqualityCheck(float *hostPtr, float *devicePtr, const int size);

#endif // CCODE_H
~~~
{: .language-c}

Next, copy the c function definitions from the ***gpuVectorSum.cu*** file to a new file, add the include preprocessor statements and save it as ***cCode.c***. The resulting source file should look like the following

~~~
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "cCode.h"

void dataInitializer(float *inputArray, int size) {
    /* Generating float-type random numbers 
     * between 0.0 and 1.0
     */
    time_t t;
    srand( (unsigned int) time(&t) );

    for (int i = 0; i < size; i++) {
        inputArray[i] = ( (float)rand() / (float)(RAND_MAX) ) * 1.0;
    }

    return;
}
/*-----------------------------------------------*/
void arraySumOnHost(float *A, float *B, float *C, const int size) {
    for (int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}
/*-----------------------------------------------*/
void arrayEqualityCheck(float *hostPtr, float *devicePtr, const int size) {
    double tolerance = 1.0E-8;
    bool isEqual = true;

    for (int i = 0; i < size; i++) {
        if (abs(hostPtr[i] - devicePtr[i]) > tolerance) {
            isEqual = false;
            printf("Arrays are NOT equal because:\n");
            printf("at %dth index: hostPtr[%d] = %5.2f \
            and devicePtr[%d] = %5.2f;\n", \
            i, i, hostPtr[i], i, devicePtr[i]);
            break;
        }
    }

    if (isEqual) {
        printf("Arrays are equal.\n\n");
    }

    return;
}
~~~
{: .language-c}

After moving all C-based functions to separate source and 
header files, we should add the function declarations in 
***cCode.h*** to both ***cCode.c*** and ***gpuVectorSum.cu*** files.
The latter file now should be similar to the following

~~~
/*================================================*/
/*================ gpuVectorSum.cu ===============*/
/*================================================*/
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "cCode.h"

/*-----------------------------------------------*/
__global__ void arraySumOnDevice(float *A, float *B, float *C, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { 
        C[idx] = A[idx] + B[idx];
    }
}
/*************************************************/
int main(int argc, char **argv) {
    printf("Kicking off %s\n\n", argv[0]);

    /* Device setup */
    int deviceIdx = 0;
    cudaSetDevice(deviceIdx);

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
    cudaMalloc((float**)&d_A, vecSizeInBytes);
    cudaMalloc((float**)&d_B, vecSizeInBytes);
    cudaMalloc((float**)&d_C, vecSizeInBytes);

    /* Data transfer from host to device */
    cudaMemcpy(d_A, h_A, vecSizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, vecSizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, devicePtr, vecSizeInBytes, cudaMemcpyHostToDevice);

    /* Organizing grids and blocks */
    int numThreadsInBlocks = 1024;
    dim3 block (numThreadsInBlocks);
    dim3 grid  ((vecSize + block.x - 1) / block.x);

    /* Execute the kernel from the host*/
    tStart = chronometer();
    arraySumOnDevice<<<grid, block>>>(d_A, d_B, d_C, vecSize);
    cudaDeviceSynchronize();
    tElapsed = chronometer() - tStart;
    printf("Elapsed time for arraySumOnDevice <<< %d, %d >>>: %f second(s)\n\n", \
    grid.x, block.x, tElapsed);
/*-----------------------------------------------*/
    /* Returning the last error from a runtime call */
    cudaGetLastError();

    /* Data transfer back from device to host */
    cudaMemcpy(devicePtr, d_C, vecSizeInBytes, cudaMemcpyDeviceToHost);

    /* Check to see if the array summations on 
     * CPU and GPU yield the same results 
     */
    arrayEqualityCheck(hostPtr, devicePtr, vecSize);
/*-----------------------------------------------*/
    /* Free the allocated memory on the device */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    /* Free the allocated memory on the host */
    free(h_A);
    free(h_B);
    free(hostPtr);
    free(devicePtr);

    return(EXIT_SUCCESS);
}
~~~
{: .language-cuda}

Now let's try and compile our multiple source files into an executable
using nvcc and running the following command

~~~
$ nvcc gpuVectorSum.cu cCode.c -o gpuVectorSum
$ ./gpuVectorSum
~~~
{: .language-bash}

After running the aforementioned command, it is likely that you've got
some error messages like the following to deal with

~~~
/tmp/tmpxft_000050f6_00000000-11_test.o: In function `main':
tmpxft_000050f6_00000000-6_test.cudafe1.cpp:(.text+0x16a): undefined reference to `dataInitializer(float*, int)'
tmpxft_000050f6_00000000-6_test.cudafe1.cpp:(.text+0x181): undefined reference to `dataInitializer(float*, int)'
tmpxft_000050f6_00000000-6_test.cudafe1.cpp:(.text+0x227): undefined reference to `arraySumOnHost(float*, float*, float*, int)'
tmpxft_000050f6_00000000-6_test.cudafe1.cpp:(.text+0x477): undefined reference to `arrayEqualityCheck(float*, float*, int)'
collect2: error: ld returned 1 exit status
~~~
{: .output}

The error comes from the `ld` (short for "load") GNU linker which
complains about receiving undefined references to a list of functions
which we have put in ***cCode.c*** source file. Can you guess what is
the problem?

The main job of `ld` linker is to combine the object and archive files, rearrange their data and manage their symbol references. Calling `ld` is usually the last step of the compilation process. So, you can guess there
should not be any issues or bugs withing your code and it should be something
else. The source of the problem might not be that trivial to many of the
readers: nvcc compiler uses the host's C++ compiler for compiling all
non-GPU code. Therefore, when you try to link the symbol references in
***cCode.c*** file, the C++ compiler (and nvcc) do not know you are trying
to embed the C (not C++) function definitions in your program.

The solution to this problem is familiar to programmers who in the past
tried to call C functions within C++ programs: using `extern "C" { ... }`
snippet. Use this snippet around `#include cCode.h` preprocessor statement
and you should be able to successfully compile your code using the 
same command without any issues.



~~~
/*================================================*/
/*==================== cCode.h ===================*/
/*================================================*/
#ifndef CCODE_H
#define CCODE_H

#include <time.h>
#include <sys/time.h>

/*************************************************/
inline double chronometer() {
    struct timezone tzp;
    struct timeval tp;
    int tmp = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
/*-----------------------------------------------*/
void dataInitializer(float *inputArray, int size);
void arraySumOnHost(float *A, float *B, float *C, const int size);
void arrayEqualityCheck(float *hostPtr, float *devicePtr, const int size);

#endif // CCODE_H
~~~
{: .language-c}

~~~
/*================================================*/
/*==================== cCode.c ===================*/
/*================================================*/
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "cCode.h"

void dataInitializer(float *inputArray, int size) {
    /* Generating float-type random numbers 
     * between 0.0 and 1.0
     */
    time_t t;
    srand( (unsigned int) time(&t) );

    for (int i = 0; i < size; i++) {
        inputArray[i] = ( (float)rand() / (float)(RAND_MAX) ) * 1.0;
    }

    return;
}
/*-----------------------------------------------*/
void arraySumOnHost(float *A, float *B, float *C, const int size) {
    for (int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}
/*-----------------------------------------------*/
void arrayEqualityCheck(float *hostPtr, float *devicePtr, const int size) {
    double tolerance = 1.0E-8;
    bool isEqual = true;

    for (int i = 0; i < size; i++) {
        if (abs(hostPtr[i] - devicePtr[i]) > tolerance) {
            isEqual = false;
            printf("Arrays are NOT equal because:\n");
            printf("at %dth index: hostPtr[%d] = %5.2f \
            and devicePtr[%d] = %5.2f;\n", \
            i, i, hostPtr[i], i, devicePtr[i]);
            break;
        }
    }

    if (isEqual) {
        printf("Arrays are equal.\n\n");
    }

    return;
}
~~~
{: .language-c}

~~~
/*================================================*/
/*================== cudaCode.h ==================*/
/*================================================*/
ifndef CUDACODE_H
#define CUDACODE_H

__global__ void arraySumOnDevice(float *A, float *B, float *C, const int size);
__host__ void deviceProperties(int deviceIdx);

#endif // CUDACODE_H
~~~
{: .language-cuda}

~~~
/*================================================*/
/*================== cudaCode.cu =================*/
/*================================================*/
#include "cudaCode.h"
#include <stdio.h>
// #include <cuda_runtime.h>

/*-----------------------------------------------*/
__global__ void arraySumOnDevice(float *A, float *B, float *C, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { 
        C[idx] = A[idx] + B[idx];
    }
}

__host__ void deviceProperties(int deviceIdx) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("GPU device %s with index (%d) is set!\n\n", \
    deviceProp.name, deviceIdx);   
}
~~~
{: .language-cuda}

## 3. Error Handling