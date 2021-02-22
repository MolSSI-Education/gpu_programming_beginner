---
title: Setup
---

> ## Table of Contents
> - [1. Linux](#1-linux)
> - [2. Windows](#2-windows)
> - [3. Mac OS](#3-mac-os)
{: .prereq}

In this section, we briefly overview the necessary steps for setting up a CUDA development
environment. At the time of writing this tutorial, **CUDA Toolkit v11.2** is the latest 
official release. Therefore, this version will be the center of our focus throughout the tutorial.

## 1. Linux

Depending on the flavor of the Linux OS on the host machine, NVIDIA offers three
options for installation of CUDA Toolkit: *RPM*, *Debian* or *Runfile* packages.
Each of these packages are provided as *Local* or *Network* installers.
Network installers are ideal for users with high-speed internet connection and
low local disk storage capacity. Network installers also allow users to
download only those applications from CUDA Toolkit that they need. Local installers,
on the other hand, offer a stand-alone large-size installer file that should be downloaded
to the host machine once. Future installations using this installer file will not require
any internet connection. Runfiles are Local installers but depending on the type of
Linux OS on the host machine, RPM and Debian packages can be Local or Network installers.

Managing the dependencies and prerequisites in various operating systems can be
very different depending on the chosen installation method. In comparison with 
Debian and RPM packages, Runfiles offer a cleaner and more independent method 
with more control over the installation process. Meanwhile, the installed CUDA 
Toolkit and its dependent software will not automatically update. On the other hand,
Debian and RPM packages provide a native and straightforward way to install the CUDA
Toolkit. However, resolving dependencies, conflicts and broken packages will often
be an inseparable part of the process. Take a look at CUDA Toolkit 
[documentation](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#linux)
for further information and details on installers.

Before spending time on the installation of the CUDA Toolkit on your Linux machine,
consider the following set of actions.

> ## Pre-installation Steps   {#pre-installation-steps}
> - Make sure your system has a CUDA-capable graphics processing unit (GPU) device.
> There are multiple ways to do this task:
>
>    &#9824; For a minimalist, a simple bash command will do the trick
>
>    ~~~
>    $ ls -l /dev/nv*   
>    ~~~
>    {: .language-bash} 
> 
>    if the system is armed with a GPU accelerator device, a typical output of the 
>    command above would be:
>
>    ~~~
>    crw-rw-rw- 1 root root 195,   0 Jan 10 09:43 /dev/nvidia0        <--   This line corresponds to your active GPU accelerator device
>    crw-rw-rw- 1 root root 195, 255 Jan 10 09:43 /dev/nvidiactl
>    crw-rw-rw- 1 root root 195, 254 Jan 10 09:43 /dev/nvidia-modeset
>    crw-rw-rw- 1 root root 236,   0 Jan 10 09:43 /dev/nvidia-uvm
>    crw-rw-rw- 1 root root 236,   1 Jan 10 09:43 /dev/nvidia-uvm-tools
>    crw------- 1 root root 243,   0 Jan 10 09:43 /dev/nvme0
>    brw-rw---- 1 root disk 259,   0 Jan 10 09:43 /dev/nvme0n1
>    brw-rw---- 1 root disk 259,   1 Jan 10 09:43 /dev/nvme0n1p1
>    brw-rw---- 1 root disk 259,   2 Jan 10 09:43 /dev/nvme0n1p2
>    brw-rw---- 1 root disk 259,   3 Jan 10 09:43 /dev/nvme0n1p3
>    brw-rw---- 1 root disk 259,   4 Jan 10 09:43 /dev/nvme0n1p4
>    ~~~
>    {: .output}
>   
>    &#9829; Helpful information about active hardware on the host machine (including graphics card) can be obtained from
>    **About This Computer** panel which can be accessed from the top-right gear icon at the top 
>    corner of the Ubuntu (Unity) desktop screen or through **Settings/Details** icon that can be looked up from the search bar.
>
>    &#9827; NVIDIA website provides [tables](https://developer.nvidia.com/cuda-gpus) of CUDA-enabled GPUs along side
>    their [***compute capabilities***](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability).
>    Compute capability (or *streaming multiprocessor version*) consisting of a version number (M.N) where M and N stand for 
>    major and minor digits, respectively, specifies features that the GPU hardware can support. GPU devices with the same major 
>    number (M) belong to the same core architecture: 8 for devices based on the *Ampere* architecture, 7 for devices based 
>    on the *Volta* architecture, 6 for devices based on the *Pascal* architecture, 5 for devices based on the *Maxwell* 
>    architecture, 3 for devices based on the *Kepler* architecture, 2 for devices based on the *Fermi* architecture, and 1 
>    for devices based on the *Tesla* architecture. Older CUDA-enabled GPUs (legacy GPUs) are listed 
>    [here](https://developer.nvidia.com/cuda-legacy-gpus).
>      
>    &#9830; The *NVIDIA System Management Interface* (`nvidia-smi`) is a command-line tool which is derived
>    from *NVIDIA Management Library (NVML)* and designed to provide control and monitoring
>    capabilities over NVIDIA CUDA-enabled GPU devices. `nvidia-smi` ships with NVIDIA GPU display drivers
>    on Linux and some versions of Microsoft Windows. For further details, see
>    [here](https://developer.nvidia.com/nvidia-system-management-interface). In order to run `nvidia-smi`,
>    simply call it through a terminal:
>
>    ~~~
>    $ nvidia-smi   
>    ~~~
>    {: .language-bash} 
>    
>    A typical output would look like:
> 
>    ~~~
>    +-----------------------------------------------------------------------------+
>    | NVIDIA-SMI 455.38       Driver Version: 455.38       CUDA Version: 11.2     |
>    |-------------------------------+----------------------+----------------------+
>    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
>    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
>    |                               |                      |               MIG M. |
>    |===============================+======================+======================|
>    |   0  GeForce GTX 1650    Off  | 00000000:01:00.0 Off |                  N/A |
>    | N/A   42C    P8     2W /  N/A |    438MiB /  3911MiB |     10%      Default |
>    |                               |                      |                  N/A |
>    +-------------------------------+----------------------+----------------------+
>    ...                                                                               
>    ~~~
>    {: .output}
>
>    where unnecessary information from the output are replaced with ellipses. 
>    The result shows the driver version (455.38), CUDA version (11.2), and the
>    CUDA-enabled GPU device name (GeForce GTX 1650). Since multiple GPUs might be
>    available on each machine, applications such as `nvidia-smi` often adopt
>    integer indices, starting from zero, for referencing the GPU devices. 
>
> - Because the present tutorial is based on the CUDA C/C++ programming language extensions,
> check to see if the version of Linux on the host machine is supported by CUDA.
> To do so, take a
> glance at [this](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements) page.
> In order to verify the availability of GNU compilers (gcc and g++) on the system, try
>
>    ~~~
>    $ <gnu-compiler> --version
>    ~~~
>    {: .language-bash} 
>    
>    where `<gnu-compiler>` placeholder should be replaced with either `gcc` or `g++`.
> 
> - Download the NVIDIA CUDA Toolkit from [here](https://developer.nvidia.com/cuda-downloads).
>   Once the CUDA Toolkit installer is downloaded, follow the instructions
>   [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile) based on the type of 
>   your Linux OS platform.
{: .prereq}

> ## **Known Issues**:
>
> It is a very common issue that a previously installed version of CUDA conflicts with a
> newer version that is intended to be installed. In order to resolve the conflict, check the compatibility
> [matrices](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#handle-uninstallation)
> and follow the instructions provided thereafter.
{: .callout}

## 2. Windows

Basic instructions on using Local or Network installers can be found on CUDA Toolkit's
[documentation](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).
NVIDIA CUDA Toolkit supports specific version combinations of Microsoft Windows OSs,
compilers and Microsoft Visual Studio environments. For further details, see
[here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/#system-requirements).

> ## WSL Users
>
> After following directions in the [Pre-installation Steps](#pre-installation-steps) section, 
> the *Windows Subsystem for Linux (WSL)* users can refer to CUDA Toolkit
> [documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#wsl-installation)
> for setting up CUDA Toolkit and following the instructions.
{: .prereq}

## 3. Mac OS

> ## **Note**:
>
> CUDA Toolkit v10.2.x is the last release that supports Mac OS as a target platform
> for heterogeneous parallel code development with CUDA. However, NVIDIA still provides
> support for launching CUDA debugger and profiler application sessions for Mac OS as a host platform.
>
> Since the present tutorial is based on the latest
> version (v11.2.0), the Mac OS will not be the subject of our further consideration.
{: .discussion}

{% include links.md %}
