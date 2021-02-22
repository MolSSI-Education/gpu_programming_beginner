---
title: "CUDA GPU Compilation Model"
teaching: 45
exercises: 2
questions:
- "What is NVCC compiler and why do we need it?"
- "Can multiple GPU and CPU source code files be simultaneously compiled with NVCC?"
- "How does NVCC distinguish between the host and device code domains and handle the compilation process?"
- "How can runtime errors be handled during a CUDA program execution?"
objectives:
- "Understanding the basic mechanism of NVCC compilation phases"
- "Learning about multiple source code compilation mode in NVCC compiler"
- "Mastering the basics of error handling in a CUDA program using C/C++ wrapper marcos"
keypoints:
- "The NVCC compiler"
- "Compilation phases"
- "Compiling multiple CPU and GPU source code files simultaneously"
- "Error handling in a CUDA program"
---

> ## Table of Contents
> - [1. NVIDIA's CUDA Compiler](#1-nvidias-cuda-compiler)
> - [2. Compiling Separate Source Files using NVCC](#2-compiling-separate-source-files-using-nvcc)
> - [3. Error Handling](#3-error-handling)
{: .prereq}

## 1. NVIDIA's CUDA Compiler

**NVIDIA's CUDA compiler (NVCC)** is distributed as part of CUDA Toolkit and 
is based upon the poplar [*LLVM*](https://llvm.org/) open-source infrastructure.
Each CUDA program is a combination of host code written in C/C++ standard 
semantics with some extensions within CUDA API as well as the GPU device 
kernel functions. The nvcc compiler driver separates the host code from 
that of the device. The host code is then pre-processed and compiled with
host's [C++ compilers supported by nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#supported-host-compilers).
The nvcc compiler also pre-processes and compiles the device kernel functions
using the proprietary NVIDIA assemblers and compilers. Then, nvcc embeds 
the GPU kernels as [*fatbinary*](https://en.wikipedia.org/wiki/Fat_binary) 
images into the host object files. Finally, during the linking stage, CUDA
runtime libraries are added for kernel procedure calls as well as memory and data 
transfer managements. The description of the exact details of the 
[compilation phases](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#supported-phases) 
is beyond the scope of this tutorial. The interested reader is referred to 
[CUDA Toolkit documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#introduction),
[*parallel thread execution (PTX)* compiler API](https://docs.nvidia.com/cuda/ptx-compiler-api/index.html)
and [*instruction set architecture (ISA)*](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
for further details.

## 2. Compiling Separate Source Files using NVCC

In the [Summation of Arrays on GPUs]({{site.baseurl}}{% link _episodes/03-cuda-program-model.md %}#3-summation-of-arrays-on-gpus)
example, we criticized the length of the source code. Therefore,
we need to break it into smaller source files according to the 
logical structure of our code.
Before CUDA 5.0, splitting the CUDA source code into separate files and 
[multiple-source-file compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#using-separate-compilation-in-cuda) 
was not possible and the only available option was the whole-code 
compilation approach. Since we are using CUDA 11.2 here, this should
not be a problem anymore.
Let us now start breaking the code into separate source files by copying 
the C *function signatures* and pasting them into an empty file. Name this 
file as ***cCode.h*** and add the necessary include header pre-processor 
directives and header guards to it. The resulting header file's content 
should be the same as the following code block

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

Next, copy the C *function definitions* from the ***gpuVectorSum.cu*** 
file to a new file, add the include pre-processor directives and save it
as ***cCode.c***. The resulting source file should look like the following

~~~
/*================================================*/
/*==================== cCode.h ===================*/
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

After moving all C-based functions into separate source and 
header files, we should add the function declarations in 
***cCode.h*** to both ***cCode.c*** and ***gpuVectorSum.cu*** files.
The latter file now should contain the following code

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
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("GPU device %s with index (%d) is set!\n\n", \
    deviceProp.name, deviceIdx);
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

Now let's try to compile our multiple source files into an executable
using nvcc and run it using the following commands

~~~
$ nvcc gpuVectorSum.cu cCode.c -o gpuVectorSum
$ ./gpuVectorSum
~~~
{: .language-bash}

After running the aforementioned commands, it is likely that you will get
some error messages like the following

~~~
/tmp/tmpxft_000050f6_00000000-11_test.o: In function `main':
tmpxft_000050f6_00000000-6_test.cudafe1.cpp:(.text+0x16a): undefined reference to `dataInitializer(float*, int)'
tmpxft_000050f6_00000000-6_test.cudafe1.cpp:(.text+0x181): undefined reference to `dataInitializer(float*, int)'
tmpxft_000050f6_00000000-6_test.cudafe1.cpp:(.text+0x227): undefined reference to `arraySumOnHost(float*, float*, float*, int)'
tmpxft_000050f6_00000000-6_test.cudafe1.cpp:(.text+0x477): undefined reference to `arrayEqualityCheck(float*, float*, int)'
collect2: error: ld returned 1 exit status
~~~
{: .error}

The error comes from the `ld` (short for "load") GNU linker which
complains about receiving undefined references to a list of functions
which we have put in ***cCode.c*** source file. Can you guess what is
the problem?

The main job of `ld` linker is to combine the object and archive files,
rearrange their data and manage their symbol references. Calling `ld` 
is usually the last step of the compilation process. So, you can guess there
should not be any issues or bugs withing your code and it should be something
else. The source of the problem might not be that trivial to many of the
readers: nvcc compiler uses the host's C++ compiler by default for compiling all
non-GPU code. Therefore, when you try to link the symbol references in
***cCode.c*** file, the C++ compiler (and nvcc) do not know you that your are trying
to embed the C (not C++) function definitions in your program.

The solution to this problem is familiar to programmers who have experience
in calling C functions within C++ programs using the `extern "C" { ... }`
snippet. By wrapping thin snippet around the `#include cCode.h` pre-processor
directive, you should be able to successfully compile your code using the 
same command without any issues.

We have now, localized the C function definitions into their corresponding
implementation source files and headers. However, we still have a CUDA kernel
function definition left in the ***gpuVectorSum.cu*** file. Similar to what
we have done for C function definitions, CUDA kernel definitions can also be
moved to their pertinent source files. Therefore, the first step is to
create a header file, ***cudaCode.h***, which should include the declaration
signature of the `arraySumOnDevice()` kernel function and look like 
the following code block

~~~
/*================================================*/
/*================== cudaCode.h ==================*/
/*================================================*/
#ifndef CUDACODE_H
#define CUDACODE_H

__global__ void arraySumOnDevice(float *A, float *B, float *C, const int size);

#endif // CUDACODE_H
~~~
{: .language-cuda}


Next, create a source file for kernel definition and save it as
***cudaCode.cu***. The contents of the ***cudaCode.cu*** file should be
the same as the following code block

~~~
/*================================================*/
/*================== cudaCode.cu =================*/
/*================================================*/
#include "cudaCode.h"
#include <stdio.h>

/*-----------------------------------------------*/
__global__ void arraySumOnDevice(float *A, float *B, float *C, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { 
        C[idx] = A[idx] + B[idx];
    }
}
~~~
{: .language-cuda}

Do not forget to add the ***cudaCode.h*** pre-processor `#include` directive
in both ***cudaCode.cu*** and ***gpuVectorSum.cu*** files. The latter
file now does not include any function or kernel function definitions 
for either host or device side code domains, respectively. 
The ***gpuVectorSum.cu*** file now has the following structure

~~~
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
    cudaSetDevice(deviceIdx);

    /* Device properties */
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("GPU device %s with index (%d) is set!\n\n", \
    deviceProp.name, deviceIdx);
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

The localization of each code domain into its own implementation source files
allows for regimenting a logical hierarchy within our package structure which
makes the debugging process more convenient and efficient as the pertinent 
source files are now shorter as well.

In order to compile our code which now includes an additional source file,
we should run the following commands

~~~
$ nvcc gpuVectorSum.cu cCode.c cudaCode.cu -o gpuVectorSum
$ ./gpuVectorSum
~~~
{: .language-bash}

There are still opportunities for us in the `main()` function within the 
***gpuVectorSum.cu*** file for further encapsulation of code into new functions 
that can be subsequently transferred to the ***cCode.c*** or 
***cudaCode.cu*** source files and their corresponding headers.
The following exercise asks you to find these opportunities and
use them to make the code even shorter and therefore more readable and
easier to work with.

> ## Exercise: 
> Parts of our code can still be made shorter through
> encapsulation of groups of expressions and statements that can be
> incorporated into a function. See if you can find these
> code parts within the ***gpuVectorSum.cu*** file
> and transfer them into ***cCode.c*** or 
> ***cudaCode.cu*** source files.
>> ## Solution
>> The three lines under `/* Device properties */` comment
>> can be transferred into a function called `deviceProperties(deviceIdx)`
>> which takes the device index `deviceIdx` as its input argument.
>> So, the new ***cudaCode.cu*** file should take the following form
>> ~~~
>> /*================================================*/
>> /*================== cudaCode.cu =================*/
>> /*================================================*/
>> #include "cudaCode.h"
>> #include <stdio.h>
>> 
>> /*-----------------------------------------------*/
>> __global__ void arraySumOnDevice(float *A, float *B, float *C, const int size) {
>>     int idx = blockIdx.x * blockDim.x + threadIdx.x;
>>     if (idx < size) { 
>>         C[idx] = A[idx] + B[idx];
>>     }
>> }
>> /*-----------------------------------------------*/
>> __host__ void deviceProperties(int deviceIdx) {
>>     cudaDeviceProp deviceProp;
>>     cudaGetDeviceProperties(&deviceProp, deviceIdx);
>>     printf("GPU device %s with index (%d) is set!\n\n", \
>>     deviceProp.name, deviceIdx);   
>> }
>> ~~~
>> {: .language-cuda}
>> 
>> and the corresponding header file, ***cudaCode.h***, should be
>> the same as the following code block
>> ~~~
>> /*================================================*/
>> /*================== cudaCode.h ==================*/
>> /*================================================*/
>> #ifndef CUDACODE_H
>> #define CUDACODE_H
>> 
>> __global__ void arraySumOnDevice(float *A, float *B, float *C, const int size);
>> __host__ void deviceProperties(int deviceIdx);
>> 
>> #endif // CUDACODE_H
>> ~~~
>> {: .language-cuda}
>>
>> Note that we have used `__host__` declaration expression qualifier
>> for our device function, `deviceProperties()`, since it will run from
>> host and therefore, no specific memory management on the device will
>> required.
>> With the aforementioned changes, the ***gpuVectorSum.cu*** file
>> takes the following form 
>>
>> ~~~
>> /*================================================*/
>> /*================ gpuVectorSum.cu ===============*/
>> /*================================================*/
>> #include <stdlib.h>
>> #include <stdio.h>
>> #include <cuda_runtime.h>
>> #include "cudaCode.h"
>> extern "C" {
>>     #include "cCode.h"
>> }
>> 
>> /*************************************************/
>> int main(int argc, char **argv) {
>>     printf("Kicking off %s\n\n", argv[0]);
>> 
>>     /* Device setup */
>>     int deviceIdx = 0;
>>     cudaSetDevice(deviceIdx);
>> 
>>     /* Device properties */
>>     deviceProperties(deviceIdx);
>> /*-----------------------------------------------*/
>>     /* Fixing the vector size to 1 * 2^24 = 16777216 (64 MB) */
>>     int vecSize = 1 << 24;
>>     size_t vecSizeInBytes = vecSize * sizeof(float);
>>     printf("Vector size: %d floats (%lu MB)\n\n", vecSize, vecSizeInBytes/1024/1024);
>> 
>>     /* Memory allocation on the host */
>>     float *h_A, *h_B, *hostPtr, *devicePtr;
>>     h_A     = (float *)malloc(vecSizeInBytes);
>>     h_B     = (float *)malloc(vecSizeInBytes);
>>     hostPtr = (float *)malloc(vecSizeInBytes);
>>     devicePtr  = (float *)malloc(vecSizeInBytes);
>> 
>>     double tStart, tElapsed;
>> 
>>     /* Vector initialization on the host */
>>     tStart = chronometer();
>>     dataInitializer(h_A, vecSize);
>>     dataInitializer(h_B, vecSize);
>>     tElapsed = chronometer() - tStart;
>>     printf("Elapsed time for dataInitializer: %f second(s)\n", tElapsed);
>>     memset(hostPtr, 0, vecSizeInBytes);
>>     memset(devicePtr,  0, vecSizeInBytes);
>> 
>>     /* Vector summation on the host */
>>     tStart = chronometer();
>>     arraySumOnHost(h_A, h_B, hostPtr, vecSize);
>>     tElapsed = chronometer() - tStart;
>>     printf("Elapsed time for arraySumOnHost: %f second(s)\n", tElapsed);
>> /*-----------------------------------------------*/
>>     /* (Global) memory allocation on the device */
>>     float *d_A, *d_B, *d_C;
>>     cudaMalloc((float**)&d_A, vecSizeInBytes);
>>     cudaMalloc((float**)&d_B, vecSizeInBytes);
>>     cudaMalloc((float**)&d_C, vecSizeInBytes);
>> 
>>     /* Data transfer from host to device */
>>     cudaMemcpy(d_A, h_A, vecSizeInBytes, cudaMemcpyHostToDevice);
>>     cudaMemcpy(d_B, h_B, vecSizeInBytes, cudaMemcpyHostToDevice);
>>     cudaMemcpy(d_C, devicePtr, vecSizeInBytes, cudaMemcpyHostToDevice);
>> 
>>     /* Organizing grids and blocks */
>>     int numThreadsInBlocks = 1024;
>>     dim3 block (numThreadsInBlocks);
>>     dim3 grid  ((vecSize + block.x - 1) / block.x);
>> 
>>     /* Execute the kernel from the host*/
>>     tStart = chronometer();
>>     arraySumOnDevice<<<grid, block>>>(d_A, d_B, d_C, vecSize);
>>     cudaDeviceSynchronize();
>>     tElapsed = chronometer() - tStart;
>>     printf("Elapsed time for arraySumOnDevice <<< %d, %d >>>: %f second(s) \n\n", \
>>     grid.x, block.x, tElapsed);
>> /*-----------------------------------------------*/
>>     /* Returning the last error from a runtime call */
>>     cudaGetLastError();
>> 
>>     /* Data transfer back from device to host */
>>     cudaMemcpy(devicePtr, d_C, vecSizeInBytes, cudaMemcpyDeviceToHost);
>> 
>>     /* Check to see if the array summations on 
>>      * CPU and GPU yield the same results 
>>      */
>>     arrayEqualityCheck(hostPtr, devicePtr, vecSize);
>> /*-----------------------------------------------*/
>>     /* Free the allocated memory on the device */
>>     cudaFree(d_A);
>>     cudaFree(d_B);
>>     cudaFree(d_C);
>> 
>>     /* Free the allocated memory on the host */
>>     free(h_A);
>>     free(h_B);
>>     free(hostPtr);
>>     free(devicePtr);
>> 
>>     return(EXIT_SUCCESS);
>> }
>> ~~~
>> {: .language-cuda}
>>
> {: .solution}
{: .challenge}

## 3. Error Handling

Many of the CUDA function calls within each CUDA program are asynchronous--
the execution flow returns to the host immediately after the function call.
The asynchronous nature of these function calls makes it difficult to identify
and troubleshoot the source of errors if several CUDA functions have been
called consecutively. Fortunately, with the exception of kernel executions,
CUDA functions return error codes of [`cudaError_t`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) 
enumerated type. As such, we can define error handling macros to wrap
around the CUDA function calls and check them for any possible errors.

In order to include a macro definition within our 
[Summation of Arrays on GPUs]({{site.baseurl}}{% link _episodes/03-cuda-program-model.md %}#3-summation-of-arrays-on-gpus)
code, open the ***cCode.h*** header file and add the following macro definition to it

~~~
/*================================================*/
/*==================== cCode.h ===================*/
/*================================================*/
#ifndef CCODE_H
#define CCODE_H

#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

#define ERRORHANDLER(funcCall) { \
    const cudaError_t error = funcCall; \
    char *errorMessage = cudaGetErrorString(error); \
    if (error != cudaSuccess) { \
        printf("Error in file %s, line %d, code %d,  Message %s\n", \
        __FILE__, __LINE__, error, errorMessage); \
        exit(EXIT_FAILURE); \
    } \
}
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

Note that backslashes (`\`) are used as the 
*line continuation escape character* since marcos should be defined 
in one line. Each backslash must be the last character on the line 
otherwise you will get an error.
We can now wrap our CUDA function calls within `ERRORHANDLER()` macro
in order to capture the errors. If an error happens, `ERRORHANDLER()` 
macro prints the error code in a human-readable format and terminates 
the program by calling the `exit(EXIT_FAILURE)` function.

> ## Exercise:
> Try to wrap every CUDA function call in ***gpuVectorSum.cu*** file
> with the `ERRORHANDLER()` macro.
>> ## Solution:
>> ~~~
>> /*================================================*/
>> /*================ gpuVectorSum.cu ===============*/
>> /*================================================*/
>> #include <stdlib.h>
>> #include <stdio.h>
>> #include <cuda_runtime.h>
>> #include "cudaCode.h"
>> extern "C" {
>>     #include "cCode.h"
>> }
>> 
>> /*************************************************/
>> int main(int argc, char **argv) {
>>     printf("Kicking off %s\n\n", argv[0]);
>> 
>>     /* Device setup */
>>     int deviceIdx = 0;
>>     ERRORHANDLER(cudaSetDevice(deviceIdx));
>> 
>>     /* Device properties */
>>     deviceProperties(deviceIdx);
>> /*-----------------------------------------------*/
>>     /* Fixing the vector size to 1 * 2^24 = 16777216 (64 MB) */
>>     int vecSize = 1 << 24;
>>     size_t vecSizeInBytes = vecSize * sizeof(float);
>>     printf("Vector size: %d floats (%lu MB)\n\n", vecSize, vecSizeInBytes/1024/1024);
>> 
>>     /* Memory allocation on the host */
>>     float *h_A, *h_B, *hostPtr, *devicePtr;
>>     h_A     = (float *)malloc(vecSizeInBytes);
>>     h_B     = (float *)malloc(vecSizeInBytes);
>>     hostPtr = (float *)malloc(vecSizeInBytes);
>>     devicePtr  = (float *)malloc(vecSizeInBytes);
>> 
>>     double tStart, tElapsed;
>> 
>>     /* Vector initialization on the host */
>>     tStart = chronometer();
>>     dataInitializer(h_A, vecSize);
>>     dataInitializer(h_B, vecSize);
>>     tElapsed = chronometer() - tStart;
>>     printf("Elapsed time for dataInitializer: %f second(s)\n", tElapsed);
>>     memset(hostPtr, 0, vecSizeInBytes);
>>     memset(devicePtr,  0, vecSizeInBytes);
>> 
>>     /* Vector summation on the host */
>>     tStart = chronometer();
>>     arraySumOnHost(h_A, h_B, hostPtr, vecSize);
>>     tElapsed = chronometer() - tStart;
>>     printf("Elapsed time for arraySumOnHost: %f second(s)\n", tElapsed);
>> /*-----------------------------------------------*/
>>     /* (Global) memory allocation on the device */
>>     float *d_A, *d_B, *d_C;
>>     ERRORHANDLER(cudaMalloc((float**)&d_A, vecSizeInBytes));
>>     ERRORHANDLER(cudaMalloc((float**)&d_B, vecSizeInBytes));
>>     ERRORHANDLER(cudaMalloc((float**)&d_C, vecSizeInBytes));
>> 
>>     /* Data transfer from host to device */
>>     ERRORHANDLER(cudaMemcpy(d_A, h_A, vecSizeInBytes, cudaMemcpyHostToDevice));
>>     ERRORHANDLER(cudaMemcpy(d_B, h_B, vecSizeInBytes, cudaMemcpyHostToDevice));
>>     ERRORHANDLER(cudaMemcpy(d_C, devicePtr, vecSizeInBytes, cudaMemcpyHostToDevice));
>> 
>>     /* Organizing grids and blocks */
>>     int numThreadsInBlocks = 1024;
>>     dim3 block (numThreadsInBlocks);
>>     dim3 grid  ((vecSize + block.x - 1) / block.x);
>> 
>>     /* Execute the kernel from the host*/
>>     tStart = chronometer();
>>     arraySumOnDevice<<<grid, block>>>(d_A, d_B, d_C, vecSize);
>>     ERRORHANDLER(cudaDeviceSynchronize());
>>     tElapsed = chronometer() - tStart;
>>     printf("Elapsed time for arraySumOnDevice <<< %d, %d >>>: %f second(s) \n\n", \
>>     grid.x, block.x, tElapsed);
>> /*-----------------------------------------------*/
>>     /* Returning the last error from a runtime call */
>>     ERRORHANDLER(cudaGetLastError());
>> 
>>     /* Data transfer back from device to host */
>>     ERRORHANDLER(cudaMemcpy(devicePtr, d_C, vecSizeInBytes, cudaMemcpyDeviceToHost));
>> 
>>     /* Check to see if the array summations on 
>>      * CPU and GPU yield the same results 
>>      */
>>     arrayEqualityCheck(hostPtr, devicePtr, vecSize);
>> /*-----------------------------------------------*/
>>     /* Free the allocated memory on the device */
>>     ERRORHANDLER(cudaFree(d_A));
>>     ERRORHANDLER(cudaFree(d_B));
>>     ERRORHANDLER(cudaFree(d_C));
>> 
>>     /* Free the allocated memory on the host */
>>     free(h_A);
>>     free(h_B);
>>     free(hostPtr);
>>     free(devicePtr);
>> 
>>     return(EXIT_SUCCESS);
>> }
>> ~~~
>> {: .language-cuda}
>>
>> We should point out that our `deviceProperties()` function is not
>> a CUDA API function. Since it encapsulates the `cudaGetDeviceProperties()`
>> CUDA function within its implementation, we could wrap the 
>> `ERRORHANDLER()`macro directly around it within the `deviceProperties()` 
>> function definition. However, this will add a C-based header file within
>> our device-based code which is in contradiction with our first intention
>> of separating the device and the host side code domains within different source
>> files. This is one of those situations that we might have to compromise
>> somehow-- either by adding the ***cCode.h*** header into our 
>> ***cudaCode.cu*** file or leaving the `cudaGetDeviceProperties()`
>> CUDA function un-encapsulated within the `main()` function.
> {: .solution}
{: .challenge}

{% include links.md %}