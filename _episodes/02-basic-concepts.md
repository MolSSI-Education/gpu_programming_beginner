---
title: "Basic Concepts in CUDA Programming"
teaching: 30
exercises: 0
questions:
- "How to write, compile and run a basic CUDA program?"
- "What is the structure of a CUDA program?"
- "How to write and launch a CUDA kernel function?"
- "How to capture runtime errors in CUDA?"
objectives:
- "Understanding the basics of the CUDA programming model"
- "The ability to write, compile and run a basic CUDA program"
- "Recognition of similarities between the semantics of C and those of CUDA C"
- "Debugging and catching runtime errors"
- "Handling multiple GPUs"
keypoints:
- "CUDA programming model"
- "Preliminary structure of a CUDA program"
- "Implementing and launching CUDA kernels"
- "Good practices in handling errors and multiple GPU devices"
---

> ## Table of Contents
> - [1. Writing Our First CUDA Program](#1-writing-our-first-cuda-program)
> - [2. Structure of a CUDA Program](#2-structure-of-a-cuda-program)
>   - [2.1. Writing a CUDA Kernel](#21-writing-a-cuda-kernel)
>   - [2.2. Kernel Execution in CUDA](#22-kernel-execution-in-cuda)
> - [3. Debugging and Error Checking](#3-debugging-and-error-checking)
>   - [3.1 Runtime API functions and cudaError](#31-runtime-api-functions-and-cudaerror)
>   - [3.2 Dealing with errors and kernel launches](#32-dealing-with-errors-and-kernel-launches)
> - [4. Multiple GPUs and Multithreading Bugs](#4-multiple-gpus-and-multithreading-bugs)
]
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

If you got output only from the CPU and none from the GPU, see [section 3](#3-debugging-and-error-checking).
The first line in the output has been printed by the CPU and the next 8 lines 
have been printed by the GPU. Readers with sharp eyes might get curious about a possible
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
All it does is call the `printf()` function to print the "Hello World" message
on the screen.

### 2.1. Writing a CUDA Kernel
The CUDA kernel, such as `helloFromGPU()`, is a function that gets executed on the GPU device. Every CUDA kernel 
starts with a declaration specifier. The CUDA C [syntax](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model) 
for kernel function definition is as follows:

~~~
__declarationSpecification__ void kernelName( parameterList ) {
    kernelImplementation
}
~~~
{: .language-cuda}

The 3 types of qualifiers (`__declarationSpecification__`) we will be discussing here are `__host__`, `__device__`, and `__global__`.

The __host__ execution qualifier declares a function to be a kernel that is:
- Executed on the host
- Callable from the host only

> ## Note:
> Any function declared without any qualifier is by default assumed to compile only for the CPU
> i.e. `__host__` is implicitly the default function qualifier in CUDA C.
{: .discussion}

The __device__ execution qualifier declares a function to be a kernel that is:
- Executed on the device
- Callable from the device only

> ## Note:
> The `__host__` and `__device__` qualifiers can be used simultaneously in the kernel declaration 
> if the kernel function should be compiled for both host and device.
{: .discussion}

The __global__ execution qualifier declares a function to be a kernel that:
- Is executed on the device
- Is callable from the host
- Is callable from the device for devices of compute capability 3.2 or higher
- Must have void return type
- Cannot be a member of a class

> ## Note:
> The `__global__` qualifier cannot be used with either the `__device__` or 
> `__host__` qualifiers.
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


## 3. Debugging and Error Checking

### 3.1 Runtime API functions and cudaError

In general, it is considered a good practice to check for errors every time an API CUDA function is called.
This makes debugging much easier and less time consuming. The CUDA API provides several [functions](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html) for handling errors. In the modified code below for ***hello.cu***, we introduced a macro (`checkCudaErrors`) that wraps
the function `checkErr(cudaError_t error, ...)`. The latter captures any runtime 
error in a kernel launch and subsequently exits on failure (or for instance an exception can be thrown instead).

CUDA API runtime functions use the enum `cudaError_t` (or `cudaError`) to describe the execution return code.
For instance, if `cudaError_t` is `cudaSuccess` (0), execution is considered to be successful. Otherwise, `cudaError_t` is a positive, non-zero integer
that describes different kinds of failures e.g. if `cudaError_t` is `cudaErrorInvalidValue` (1), then this implies one or more of the parameters passed to the API call is not within an acceptable range of values. For a list of all the possible return codes and their meaning, see the [documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038).

> ## Note:
> All API runtime functions in CUDA return the enum cudaError, which describes CUDA error types
{: .discussion}

The modified ***hello.cu*** code is shown below.

~~~
#include <stdio.h>                        /* For printf() function */

/**********************************************/

/* Define a macro function that returns the runtime error, file name, and line number */
#define checkCudaErrors(val) checkErr((val), #val, __FILE__, __LINE__)

void checkErr(cudaError_t error, char const *const func, const char *const file,
           int const line) {
  if (error) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(error), cudaGetErrorName(error), func);
    exit(EXIT_FAILURE);
  }
}

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

    checkCudaErrors( cudaDeviceReset() );         /* House-keeping on device */

    return(EXIT_SUCCESS);
}
~~~
{: .language-cuda}


### 3.2 Dealing with errors and kernel launches

There are numerous problems that could arise when using CUDA kernels. Some problems may not be related to any custom code
but to hardware or software incomptability problems. For instance, if running
the compiled `hello.cu` binary file did not yield any stdout from the GPU, then this might be
due to an incomptabile driver. To uncover such problems, we need to further improve the code in `hello.cu`.

Unfortunately, CUDA kernel launches do not return any error code, so we will need to make use of the runtime API functions such as
`cudaPeekAtLastError()` and `cudaGetLastError ` to catch any prelaunch errors. Both functions are similar in functionality in returning the last error that has been produced by any of the runtime calls in the same host thread, but the latter does the additional step of resetting the error to `cudaSuccess`. For the sake of demonstration, we will only need `cudaPeekAtLastError` here.

Since kernel launches are asynchronous, the application must 1st synchronize inbetween the kernel launch and then call cudaPeekAtLastError(), 
which we achieve by calling another runtime API function: `cudaDeviceSynchronize`. Synchronisation is not always necessary if it is followed by a blocking API call. In this case, we will synchornize all threads between the host and the device, and then exit. The modified code is shown below.

~~~
/**********************************************/

int main(int argc, char **argv) {
    helloFromCPU();                       /* Calling from host */

    helloFromGPU<<<1, 8>>>();             /* Calling from device */

    checkCudaErrors( cudaPeekAtLastError() );
    checkCudaErrors( cudaDeviceSynchronize() );    /* Force the host to wait for the device to check for error */ 

    /* Do more stuff ... */

    checkCudaErrors( cudaDeviceReset() );         /* House-keeping on device */

    return(EXIT_SUCCESS);
}
~~~
{: .language-cuda}

We can compile this code just like before. A common problem encountered when setting up CUDA infrastructure for the 1st is driver incomptability. 
If we run `hello.cu` without checking for kernel errors, we get stdout only from the CPU. With error checking for kernel launches taken into account, we can
decipher the problem by catching the runtime error:
~~~
$ ./hello
~~~
{: .language-bash}

will print the following output

~~~
Hello World from CPU!
GPUassert: CUDA driver version is insufficient for CUDA runtime version hello.cu 32
~~~
{: .output}

In this case, the runtime error has informed us that we need to install a newer driver for successful code execution
on the device. This is a typical error beginners encounter on the GNU/Linux OS.

For more info on error handling, see the more detailed [Error Handling](/04-gpu-compilation-model#3-error-handling) section.

## 4. Multiple GPUs and Multithreading Bugs

Apart from always checking for errors returned from CUDA API runtime functions, it is a good practice to always check for
multiple devices available on the system and then specify which device to run the computations on. To demonstrate this, we're
going to launch the same "hello world" kernel on multiple cores using [OpenMP](https://www.openmp.org).

The modified `hello.cu` code is shown below.

~~~

int main(int argc, char **argv) {

    int numGPUs;

    checkCudaErrors(cudaGetDeviceCount(&numGPUs));

    printf("CUDA-capable device count: %i\n", numGPUs);
        
    helloFromCPU();                       /* Calling from host */

    #pragma omp parallel 
    {
        helloFromGPU<<<1, 8>>>();             /* Calling from device */
    }

    checkCudaErrors( cudaDeviceSynchronize() );    /* Force the host to wait for the device to check for error */ 

    /* Do more stuff ... */

    checkCudaErrors( cudaDeviceReset() );         /* House-keeping on device */

    return(EXIT_SUCCESS);
}
~~~
{: .language-cuda}

While this code would work fawlessly for a single GPU device, it hides a potential multi-threading bug when more than a single GPU device
is available. The reason is we used OpenMP to spawn more threads without specifying the device. CUDA assumes device 0 is the default device, which 
might lead to unexpected behavior. To fix this problem, we need to use the API function: `cudaError_t cudaSetDevice (int  device)` to specify
which device to run multi-processing on. The improved code is shown below.

~~~
#include <stdio.h>                        /* For printf() function */

/**********************************************/


int main(int argc, char **argv) {

    int device = atoi(sys.argv[1]), numGPUs;

    checkCudaErrors(cudaGetDeviceCount(&numGPUs));

    printf("CUDA-capable device count: %i\n", numGPUs);
        
    helloFromCPU();                       /* Calling from host */

    #pragma omp parallel 
    {
        checkCudaErrors(cudaSetDevice(device));    /* Run OpenMP only on device 1 */
        helloFromGPU<<<1, 8>>>();             /* Calling from device */
    }

    checkCudaErrors( cudaDeviceSynchronize() );    /* Force the host to wait for the device to check for error */ 

    /* Do more stuff ... */

    checkCudaErrors( cudaDeviceReset() );         /* House-keeping on device */

    return(EXIT_SUCCESS);
}
~~~
{: .language-cuda}

Compiling the new ***hello.cu*** code via 

~~~
$ nvcc hello.cu -Xcompiler -fopenmp -o hello
~~~
{: .language-bash}

Running the executable on device 0 works without any problems:

~~~
$ ./hello 0
~~~
{: .language-bash}

will print "Hello World from GPU!" `nCPUcores` times:

~~~
Hello World from CPU!
Hello World from GPU!
...
Hello World from GPU!
~~~
{: .output}

However, running the executable on device 1 should return error if only a single CUDA-capable GPU device is available.

~~~
$ ./hello 1
~~~
{: .language-bash}

will print the following output:

~~~
CUDA-capable device count: 1
Hello World from CPU!
CUDA error at test.cu:37 code=101(cudaErrorInvalidDevice) "cudaSetDevice(1)" 
...
~~~
{: .output}

If we remove the error checking function (checkCudaErrors) before cudaSetDevice(1) and recompile, the exectuable runs successfully, so we couldn't
have known in runtime that there's something flawed with our code. As you can see, finding such bugs can be quite difficult without error checking. This is because in principle, we can launch multiple kernels that run on device 0 but access memory allocated on device 1. This could cause invalid memory access or corruption errors if we're using pointers to allocate and access memory. Therefore, you should always make it a habit to set the device whenever your 
code could potentially spawn new host threads, and check for runtime errors every single time you make an API function call.

> ## Note:
> Always set the GPU device when creating new host threads
{: .discussion}


{% include links.md %}
