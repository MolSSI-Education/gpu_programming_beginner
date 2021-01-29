---
title: "CUDA Programming Model"
teaching: 45
exercises: 0
questions:
- "Container"
objectives:
- "Container"
keypoints:
- "Container"
---

- [1. Basics of the Device Memory Management in CUDA](#1-basics-of-the-device-memory-management-in-cuda)
- [2. Thread Hierarchy in CUDA](#2-thread-hierarchy-in-cuda)

Our Hello World example from previous lesson lacks two important aspects of a CUDA
program that are crucial for programmers in heterogeneous parallel programming within CUDA platform:
memory and thread hierarchies. Our next example will demonstrate the summation of two arrays on GPU.
Before proceeding with our example, we need to learn two important techniques in CUDA programming:
(i) memory management and (ii) thread organization. In the following, we present some of the
most frequently used functions from CUDA runtime API collection that will be used in our array summation
case study.

## 1. Basics of the Device Memory Management in CUDA

In our array summation example, (and in many scientific applications, in general), we will follow a typical pattern 
in CUDA programming which can be formulated in a series of steps as follows:

1. Transfering the data from host to device
2. Kernel execution on the device
3. Moving the results back from device to host

As we mentioned previously, most CUDA programs have at least two code domains:
(i) the host code domain which runs on the host (CPU and its memory), and (ii) the device code
domain which is performed on the device (GPU and its memory). The separation (and localization)
of data processes in each domain with different architecture type requires a specific strategy for
memory management and data transfer between the two processing units. As such, CUDA provides
convenient runtime APIs that allow the user to _allocate_ or _deallocate_ the device memory and _transfer_
data between host and device memories.

|                           **C/C++**                            |                                                                     **CUDA**                                                                      |        **Description**        |
| :------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------: |
|   [malloc()](https://en.cppreference.com/w/c/memory/malloc)    | [cudaMalloc()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356) | Allocate uninitialized memory |
| [memset()](https://en.cppreference.com/w/c/string/byte/memset) | [cudaMemset()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gf7338650f7683c51ee26aadc6973c63a) |       Initialize memory       |
|     [free()](https://en.cppreference.com/w/c/memory/free)      |  [cudaFree()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ga042655cbbf3408f01061652a075e094)  |       Deallocate memory       |
| [memcpy()](https://en.cppreference.com/w/c/string/byte/memcpy) | [cudaMemcpy()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8) |          Copy memory          |

As the table above demonstrates, CUDA adopts a convenient naming style for its C/C++ function names
and syntax extensions making it easier for the programmer to manage memory on GPU devices.
NVIDIA adopts `lowerCamelCase` (Java style) naming style for its CUDA C/C++ extention APIs.
Here, the `cudaMalloc()` function with the following syntax

```
__host__ __device__ cudaError_t cudaMalloc(void** devPtr, size_t size)
```
{: .language-cuda}

allocates `size` bytes of linear memory on the device pointed to by the `devPtr` double-pointer
variable. As mentioned previously, the `__host__` and `__device__` qualifiers can be used 
together should the kernel be compiled for both host and device.

> ## Note:
> All CUDA function APIs (except kernel launches) return an error value of 
> [enumerated type](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038), [`cudaError_t`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6).
{: .discussion}

With the memory being allocated on the device, the `cudaMemcpy()` function, with 
the following signature,

```
__host__ cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
```
{: .language-cuda}

can be adopted to transfer `count` bytes of data from source memory, pointed to by `src`
pointer, to the destination memory address, pointed to by `dst`. The direction of data
transfer is inferred from the value of the variable, `kind`, of cuda memory enumeration type,
[`cudaMemcpyKind`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g18fa99055ee694244a270e4d5101e95b) which can take one of
the following values:

- cudaMemcpyHostToHost
- cudaMemcpyHostToDevice
- cudaMemcpyDeviceToHost
- cudaMemcpyDeviceToDevice
- cudaMemcpyDefault

CUDA recommends passing `cudaMemcpyDefault` to `cudaMemcpy()` function call, in which case the
transfer direction is automatically chosen based upon the pointer values `scr` and `dst`. 
Note that `cudaMemcpyDefault` should only be adopted when [*unified virtual 
addressing (UVA)*](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiJ9rD787ruAhXkQ98KHVYMAI0QFjAAegQIARAC&url=https%3A%2F%2Fdeveloper.download.nvidia.com%2FCUDA%2Ftraining%2Fcuda_webinars_GPUDirect_uva.pdf&usg=AOvVaw0h8XB32gYKtSmfEwEaFcbQ) is supported.

> ## Note:
> Most kernel launches we consider in this tutorial are *asynchronous* in their behavior in which
> case the control flow is immediately returned to the host after kernel execution. However,
> some function calls, such as `cudaMemcpy()`, are *synchronous*-- the host application stops until
> the function completes its task.
{: .discussion}

## 2. Thread Hierarchy in CUDA

CUDA exposes a two-level thread hierarchy, consisting of **block of threads** and 
**grids of blocks**, to the programmer in order to allow for thread organization
on GPU devices.

![figure]()

As figure demonstrates, each grid is often constructed from many thread blocks.
Each block is a group of threads invoked by kernel to perform a specific task
in parallel. Each thread in a block has its own private local memory space.
However, threads in a block can cooperate to perform the same task in parallel
thanks to the shared memory space in the block which makes data visible to all
threads in the block for the life time of that block. The cooperation between threads
not only can happen in terms of sharing the data and access to it within the block-local
shared memory space but also can be realized in the form of block-level thread synchronization. 

Within the aforementioned two-level thread hierarchy, each thread can be identified with
two coordinates:

- `threadIdx`: which refers to the thread index within each block
- `blockIdx`: which stands for the block index within each grid

Both `threadIdx` and `blockIdx` identifiers are 
[built-in structure variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-variables)
of integer-based vector-type, `uint3`, assigned to each thread by CUDA 
runtime application. The internal assignment of these variables are driven by kernel 
execution which makes them accessible to that kernel. Components of the `threadIdx` or `blockIdx`
structure variables, *i.e.*, `threadIdx.x`, `threadIdx.y`, and `threadIdx.z` as well as 
`blockIdx.x`, `blockIdx.y`, and `blockIdx.z` allow for a three-dimensional organization of
blocks and grids in CUDA. The dimensions of grids of blocks and bocks of threads can be
controlled via the following CUDA built-in variables, respectively

- `blockDim`: which indicates the block of threads' dimension
- `gridDim`: which refers to the grids of block object dimension

The `blockDim` and `gridDim` variables are structures of `dim3` type with x, y, z fields
for Cartesian components. 

Let's write a simple kernel that shows how blocks of threads and 
grids of blocks can be organized and identified in an CUDA program:

```
#include <cuda_runtime.h>
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

    /* Number of thread blocks in each grid */
    int numBlocks = 2;

    /* Organizing grids and blocks */
    dim3 block(numBlocks);
    dim3 grid((numArray + block.x - 1) / block.x);

    /* Indicate that the dimensions will be printed from the host */
    printf("Printing from the host!\n"); fflush(stdout);

    /* Print the grid and block dimensions from the host */
    printf("[grid.x, grid.y, grid.z]:    [%d, %d, %d]\n", grid.x, grid.y, grid.z);
    printf("[block.x, block.y, block.z]: [%d, %d, %d]\n\n", block.x, block.y, block.z);

    /* Indicate that the dimensions will be printed from the host */
    printf("Printing from the device!\n"); fflush(stdout);

    /* Print the grid and block dimensions from the device */
    printThreadID<<<grid, block>>>();

    /* Performing house-keeping for the device */
    cudaDeviceReset();

    return(EXIT_SUCCESS);
}
```
{: .language-cuda}



{% include links.md %}