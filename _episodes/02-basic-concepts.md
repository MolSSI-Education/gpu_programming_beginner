---
title: "Basic Concepts in CUDA Programming"
teaching: 45
exercises: 0
questions:
- "How to write a basic CUDA program and then compile and run it?"
- "What is the structure of a CUDA program?"
- "How to write a CUDA kernel function?"
- "How to launch a CUDA kernel?"
objectives:
- "Understanding the basics of the CUDA programming model"
- "The ability to write, compile and run a basic heterogeneous parallel program within the CUDA platform"
- "Recognition of similarities in semantics between **C** programming language syntax and that of **CUDA C** extension"
keypoints:
- "CUDA programming model"
- "Structure of a CUDA program"
- "Writing and launching CUDA kernels"
---

> ## Table of Contents
> - [1. Writing Our First CUDA Program](#1-writing-our-first-cuda-program)
> - [2. Structure of a CUDA Program](#2-structure-of-a-cuda-program)
>   - [2.1. Writing a CUDA Kernel](#21-writing-a-cuda-kernel)
>   - [2.2. Kernel Execution in CUDA](#22-kernel-execution-in-cuda)
{: .prereq}

## 1. Writing Our First CUDA Program

In this section, we are going to write our first CUDA-enabled code that runs
on a GPU. 

> ## Note:
> In heterogeneous parallel programming, GPU works as an **accelerator** or **co-processor**
> to CPU (and not as a stand-alone computational device) in order to improve the 
> overall performance of the parallel code. As such, heterogeneous codes are constructed
> form two parts: (i) host code, which runs on CPUs, and (ii) device code, which runs on
> GPUs. The heterogeneous applications are often initialized by the host CPU and its memory)
> which manages the application environment, code processing and data transfer between the host
> and the device (GPU and its memory).
{: .discussion}

A classic example in learning any programming language is to print the "Hello World" string
to the stream output. We will start by implementing two functions `helloFromCPU()` 
and `helloFromGPU()`. Each function, as the names suggest, prints the "Hello World" message 
from the host or the device. First, open a new file with the name `hello.cu`
copy the following code into it.

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
{: .language-c}

After copying the code, save and close the file and run the following commands in the same folder within a terminal in order to compile the code:

~~~
nvcc hello.cu -o hello
~~~
{: .language-bash}

If your are familiar with the GNU compilers, you will probably notice
the similarities in the adopted syntax between *gcc* and
 *nvcc* compilers. Running the executable using

~~~
./hello
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
have been printed by GPU. A sharp pair of eyes might get curious about a possible
relation between the number of lines printed by GPU and the number 8, in the 
triple angular brackets in `helloFromGPU<<...>>()` call. We are going to find out this relation shortly.

## 2. Structure of a CUDA Program

As mentioned earlier, within the CUDA programming model, each program has  
host and device sections. Let us review our code and analyze it piece by piece
to distinguish the host and device sections.

Just like a simple code written in **C**, we include necessary headers to be able 
to access standard libraries that provide us with functions/macros we need to 
construct our application.

~~~
#include <stdlib.h>
#include <stdio.h>
~~~
{: .language-c}

Here, we have included ***stdlib.h*** header for greater portability because
it includes `EXIT_SUCCESS` (and `EXIT_FAILURE`) macros which are used in 
`return` expression for the `main()` function to show the status 
(success or failure) of our application. The ***stdio.h*** header file provides
access to the formatted print, `printf()`, function.

The second part of the code describes the implementations of 
the `helloFromCPU()` and `helloFromGPU()` functions which are written in pure 
**C** and **CUDA C**, respectively. In **C**, a function definition often takes
the following form

~~~
returnType functionName( parameterList ) {
    functionImplementation
}
~~~
{: .language-c}

The `helloFromCPU()` function has no return type and no input parameters (`void`);
All it does is to call the `printf()` function to print the "Hello World" message on the screen.

### 2.1. Writing a CUDA Kernel

The next definition is that of `helloFromGPU()` kernel function which will
be executed on the device. The kernel functions are often defined for individual
threads. However, they are called on many CUDA threads to perform their task,
concurrently. The **CUDA C** [syntax](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model) for kernel function definition is as follows

~~~
__declarationSpecification__ void kernelName( parameterList ) {
    kernelImplementation
}
~~~
{: .language-cuda}
 
The kernel declaration specification qualifier `__global__` in the kernel 
function definition `helloFromGPU()` indicates that this function executes on
the device but is callable from host or device (with compute capabilities greater than 3.0). The `__device__` qualifier is used when the kernel should be executed
on the device and should be called from the device only. The `__host__` qualifier
should be used when the kernel should be called and executed from the host, only.
The `__host__` and `__device__` qualifiers can be used simultaneously in the kernel declaration if the kernel function should be compiled for both host and
device.

> ## Note:
> A kernel function must have a `void` return type.
{: .discussion}

### 2.2. Kernel Execution in CUDA

In the previous section, we have seen the existing similarities
in the syntax adopted by **C** and **CUDA C** programming languages for 
the implementation of functions and kernel functions, respectively.
Not surprisingly, there is a close similarity between **C** and **CUDA C** 
programming language semantics adopted for (kernel) function launches.
In **CUDA C**, a kernel function can be executed as follows

~~~
kernelName<<< grid, block >>>( parameterList );
~~~
{: .language-cuda}

The **CUDA C** kernel function call syntax extends the **C** programming
language's semantics used for simple function executions through adding
**execution configuration** within triple anglular brackets `<<< ... >>>`.
The execution configuration exposes a great amount of control over *thread 
hierarchy* which enables the programmer to organize the threads for 
kernel launch. This is one of the most powerful, critical and unique 
aspects of CUDA programming model. In the next lesson, you will learn about
thread hierarchy and the meaning of each of the two arguments, `grid` and `block` within the execution configuration layout.

> ## Note:
> An important difference between conventional function calls in C 
> and CUDA kernel function launches is that the latter is *asynchronous*.
> In an asynchronous kernel launch, the control flow returns back to the CPU
> (host) right after the CUDA kernel is called.
{: .discussion}

{% include links.md %}

