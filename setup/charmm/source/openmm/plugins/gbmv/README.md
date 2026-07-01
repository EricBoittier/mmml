
# OpenMM GBMV2 plugin

This is an OpenMM plugin that accelerates the [GBMV2/SA implicit solvent model using the GPU][GPU-GBMV2/SA].

The plugin only support the CUDA platform, and its further optimization and development is also in progress.

# Descriptions on files

1. CMakeLists.txt 

CMake compile file and used to create the OpenMMGBMV shared library.

2. wrappers

This folder includes the C/Fortran API.

3. openmmapi

This folder include the API between OpenMM and GBMV plugin platforms.

4. platforms

This folder has implementations in four platforms. Currently, the CUDA platform can work, but other
platforms cannot and the CPU version can be found in the CHARMM module "gbmv".

platforms/cuda/CudaGBMVKernels.cpp: implement how to call the CUDA kernels.
platforms/cuda/kernels: include the GBMV2/SA CUDA kernels.

# Tutorials

Please look at [these examples for GBMV2/SA calculations][GBMVGitHub]

# Credits

This plugin is based on the implementation of [GPU-GBSW OpenMM plugin][GPU-GBSW], and 
is currently maintained by [Jianhan Chen group][JianhanChenGroup].

This work was supported by National Science Foundation (MCB 1817332).


[GPU-GBMV2/SA]: https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.26133
[GPU-GBSW]: https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.24280
[JianhanChenGroup]: https://people.chem.umass.edu/jchenlab/
[GBMVGitHub]: https://github.com/XipingGong/gbmvtutorial

