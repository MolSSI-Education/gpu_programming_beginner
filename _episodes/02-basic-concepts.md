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

## Writing Our First CUDA Program

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
// This function runs on the host
void helloFromCPU(void) {
    printf("Hello World from CPU!\n\n");  
}

// This function runs on the device
__global__ void helloFromGPU() {
    printf("Hello World from GPU!\n");
}

int main(int argc, char **argv) {
    // Calling from host
    helloFromCPU();  

    // Calling from device
    helloFromGPU<<<1, 8>>>();

    // House-keeping on device
    cudaDeviceReset();

    return(EXIT_SUCCESS);
}
~~~
{: .language-c}

After copying the code, save and close the file and run the following commands in the same folder within a terminal in order to compile the code:

~~~
nvcc hello.cu -o hello
~~~
{: .language-bash}

If your are familiar with the GNU compilers, you will notice
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

[Done] exited with code=0 in 1.185 seconds
~~~
{: .output}

{% include links.md %}

