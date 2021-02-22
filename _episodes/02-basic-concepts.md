---
title: "Basic Concepts in CUDA Programming"
teaching: 30
exercises: 0
questions:
- "How to write, compile and run a basic CUDA program?"
- "What is the structure of a CUDA program?"
- "How to write and launch a CUDA kernel function?"
objectives:
- "Understanding the basics of the CUDA programming model"
- "The ability to write, compile and run a basic CUDA program"
- "Recognition of similarities between the semantics of C and those of CUDA C"
keypoints:
- "CUDA programming model"
- "Preliminary structure of a CUDA program"
- "Implementing and launching CUDA kernels"
---

> ## Table of Contents
> - [1. Writing Our First CUDA Program](#1-writing-our-first-cuda-program)
> - [2. Structure of a CUDA Program](#2-structure-of-a-cuda-program)
>   - [2.1. Writing a CUDA Kernel](#21-writing-a-cuda-kernel)
>   - [2.2. Kernel Execution in CUDA](#22-kernel-execution-in-cuda)
{: .prereq}

## 1. Writing Our First CUDA Program

In this section, we plan to write our first CUDA program that runs
on a GPU device. 

> ## Note:
> In heterogeneous parallel programming, GPU works as an **accelerator** or **co-processor**
> to CPU (and not as a stand-alone computational device) in order to improve the 
> overall performance of the parallel code. As such, heterogeneous codes often consist of
> two separate domains: (i) host code, which runs on CPUs, and (ii) device code, which runs on
> GPUs. The heterogeneous applications are often initialized by the host (CPU and its memory)
> which manages the application environment, code processing and data transfer between the host
> and the device (GPU and its memory).
{: .discussion}

A classic example in learning any programming language is to print the "Hello World" string
to the output stream. We start by implementing two functions `helloFromCPU()` and `helloFromGPU()`.
Each function, as the names suggest, prints the "Hello World" message from the host or the device.
First, open a new file with the name ***hello.cu*** copy the following code into it.

~~~
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
~~~
{: .language-cuda}

After copying the code, save and close the file and run the following 
commands in the same folder within a terminal in order to compile the code:

~~~
$ nvcc hello.cu -o hello
~~~
{: .language-bash}

If your are familiar with the GNU compilers, you will probably notice
the similarities in the adopted syntax between *gcc* and
 *nvcc* compilers. Running the executable using

~~~
$ ./hello
~~~
{: .language-bash}

will print the following output

~~~
Hello World from CPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
~~~
{: .output}

The first line in the output has been printed by the CPU and the next 8 lines 
have been printed by GPU. Readers with sharp eyes might get curious about a possible
relation between the number of lines printed by GPU and the number 8, in the 
triple angular brackets in `helloFromGPU<<...>>()` kernel function call. 
We are going to find out this relation shortly.

## 2. Structure of a CUDA Program

As mentioned earlier, within the CUDA programming model, each program has 
host and device sections. Let us review our code and analyze it piece by piece
to distinguish the host and device sections.
Just like a simple code written in C, we include necessary headers using 
[*preprocessor directives*](https://en.cppreference.com/w/c/preprocessor) 
to be able to access the standard libraries that provide
us with functions/macros we need to construct our application.

~~~
#include <stdlib.h>
#include <stdio.h>
~~~
{: .language-c}

Here, we have included ***stdlib.h*** header for greater portability because
it includes `EXIT_SUCCESS` (and `EXIT_FAILURE`) macros which are used in 
`return` expression in the `main()` function to show the status 
(success or failure) of our application. The ***stdio.h*** header file provides
access to the formatted print function, `printf()`.

The second part of the code describes the implementations of 
the `helloFromCPU()` and `helloFromGPU()` functions which are written in pure 
C and CUDA C, respectively. In C, a function definition often takes
the following form

~~~
returnType functionName( parameterList ) {
    functionImplementation
}
~~~
{: .language-c}

The `helloFromCPU()` function has no return type and no input parameters (`void`);
All it does is to call the `printf()` function to print the "Hello World" message
on the screen.

### 2.1. Writing a CUDA Kernel

The next definition is that of `helloFromGPU()` kernel function which will
be executed on the device. The kernel functions are often written for individual
threads. However, they are called on many CUDA threads to perform their task,
concurrently. The CUDA C 
[syntax](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model) 
for kernel function definition is as follows

~~~
__declarationSpecification__ void kernelName( parameterList ) {
    kernelImplementation
}
~~~
{: .language-cuda}
 
The kernel **declaration specification qualifier**, `__global__`, in the kernel 
function definition, `helloFromGPU()`, indicates that this function executes on
the device but is callable from host or device (with compute capabilities 
greater than 3.0). The `__device__` qualifier is used when the kernel should be executed
on the device and should be called from the device only. The `__host__` qualifier
should be used when the kernel should be called and executed from the host, only.
The `__host__` and `__device__` qualifiers can be used simultaneously in the kernel declaration if the kernel function should be compiled for both host and
device.

> ## Note:
> A kernel function must have a `void` return type.
{: .discussion}

### 2.2. Kernel Execution in CUDA

In the previous section, we have seen the existing similarities
in the syntax adopted by C and CUDA C programming languages for 
the implementation of functions and kernel functions, respectively.
Not surprisingly, there is a connection between C and CUDA C 
programming languages' semantics adopted for (kernel) function launches.
In CUDA C, a kernel function can be executed using the following syntax

~~~
kernelName<<< grid, block >>>( parameterList );
~~~
{: .language-cuda}

The CUDA C kernel function call syntax extends the C programming
language's semantics used for simple function executions through adding
**execution configuration** within triple angular brackets `<<< ... >>>`.
The execution configuration exposes a great amount of control over 
*thread hierarchy* which enables the programmer to organize the threads for 
kernel launches. This is one of the most powerful, critical and unique 
aspects of CUDA programming model. In the next lesson, you will learn about
thread hierarchy and the meaning of each of the two arguments, 
`grid` and `block` within the execution configuration layout.

> ## Note:
> An important difference between conventional function calls in C 
> and CUDA kernel function launches is that the latter is *asynchronous*
> with respect to the control flow. In an asynchronous kernel launch, 
> the control flow returns back to the CPU (host) right after the
> CUDA kernel call.
{: .discussion}

{% include links.md %}