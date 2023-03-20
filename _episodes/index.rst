
Fundamentals of Heterogeneous Parallel Programming with CUDA C/C++
====================================================================

This course by `The Molecular Sciences Software Institute <https://molssi.org/>`_ (MolSSI)
overviews the fundamentals of heterogeneous parallel programming with CUDA C/C++ at the 
beginner level.


.. admonition:: Prerequisites
   :class: attention

   * Previous knowledge of High-performance Computing (HPC) basic concepts are helpful but not required for starting this course.
   Nevertheless, we encourage students to take a glance at our `Parallel Programming <https://education.molssi.org/parallel-programming>`_
   tutorial, specifically, Chapters 1, 2 and 5 for a brief overview of some of the fundamental concepts in HPC.

   * Basic familiarity with Bash, C and C++ programming languages is required.

.. admonition:: Software/Hardware Specifications
   :class: note


   The following NVIDIA CUDA-enabled GPU devices have been used throughout this tutorial:

   * Device 0: `GeForce GTX 1650 <https://www.nvidia.com/en-us/geforce/graphics-cards/gtx-1650>`_ with Turing architecture (Compute Capability = 7.5)
   
   * Device 1: `GeForce GT 740M <https://www.techpowerup.com/gpu-specs/geforce-gt-740m.c2299>`_ with Kepler architecture (Compute Capability = 3.5)

   Linux 18.04 (Bionic Beaver) OS is the target platform for CUDA Toolkit v11.2.0 on the two host
   machines armed with devices 0 and 1.

   Workshop Lessons
   ----------------

.. csv-table:: 
  :file: csv_tables/cuda_lessons.csv
  :header-rows: 1


.. toctree::
   :maxdepth: 2
   :hidden:
   :titlesonly:

   setup
   01-introduction
   02-basic-concepts
   03-cuda-program-model
   04-gpu-compilation-model
   05-cuda-execution-model
