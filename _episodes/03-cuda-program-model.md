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

## 1. Basics of the Device Memory Management in CUDA

Our Hello World example from previous lesson lacks two important aspects of a CUDA
program that are crucial for programmers in heterogeneous parallel programming within CUDA platform:
memory and thread hierarchies. Our next example demonstrates the summation of two arrays on GPU.
In this example, (and in many scientific applications, in general), a typical pattern of a CUDA program can
be formulated in a series of steps as follows:

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

{% include links.md %}
