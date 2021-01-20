---
title: "Basic Concepts in Heterogeneous Parallel Programming"
teaching: 0
exercises: 0
questions:
- "Key question (FIXME)"
objectives:
- "First learning objective. (FIXME)"
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---

- [1. Writing Our First CUDA Program](#1-writing-our-first-cuda-program)
- [2. Structure of a CUDA Program](#2-structure-of-a-cuda-program)

## 1. Writing Our First CUDA Program

In this section, we are going to write our first CUDA-enabled code that runs
on a GPU. 

> ## Note:
> In heterogeneous parallel programming, GPU works as an **accelerator** or **co-processor**
> to CPU (and not as a stand-alone computational device) in order to improve the 
> overall performance of the parallel code. As such, heterogeneous codes are constructed
> form two parts: (i) host code, which runs on CPUs, and (ii) device code, which runs on
> GPUs. The heterogeneous applications are often initialized by the CPU (host) which manages
> the application environment, code processing and data transfer between the host
> and the device.
{: .callout}

A classic example in learning any programming language is to print the "Hello World" string
to the stream output. We will start by implementing two functions `helloFromCPU()` 
and `helloFromGPU()`. Each function, as the names suggest, prints the "Hello World" message 
from the CPU (host) or GPU (device). First, open a new file with the name `hello.cu`
copy the following code into it.

~~~
#include <stdlib.h>                       /* for EXIT_SUCCESS macro */
#include <stdio.h>                        /* for printf() definition */

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
triple angular brackets in `helloFromGPU<<...>>()` call. We are going to find out this
relation shortly.

## 2. Structure of a CUDA Program

As mentioned earlier, within the CUDA programming model, each program has  
host and device sections. Let us review our code and analyze it piece by piece
to distinguish the host and device sections.

Just like a simple code written in **C**, we include necessary headers to be able 
to access standard libraries that provide us with functions/macros we need to 
construct our application.

~~~
#include <stdlib.h>                       /* for EXIT_SUCCESS macro */
#include <stdio.h>                        /* for printf() definition */
~~~
{: .language-c}

Here, we have included **stdlib.h** header for greater 
portability because it includes `EXIT_SUCCESS` (and `EXIT_FAILURE`) macros which 
are used in `return` expression for the `main()` function to show the status 
(success or failure) of our application.

The second part of the code describes the implementation of 
the function `helloFromCPU()` which is written in **C**:

~~~
void helloFromCPU(void) {
    printf("Hello World from CPU!\n");  
}
~~~
{: .language-c}





{% include links.md %}

