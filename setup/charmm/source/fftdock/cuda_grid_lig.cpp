#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cuda_driver_api.h"
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <kernels.h>

#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {fprintf(stderr, "Error %s at %s:%d\n in generating ligand grids using GPU\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);}

/* ---- nvrtc compilation of shared generateLigGrid kernel ---- */

static CUmodule   s_lig_module = NULL;
static CUfunction s_generateLigGrid = NULL;

static void compile_lig_kernel() {
  if (s_lig_module) return;

  if (cuda_driver_load() != 0) {
    fprintf(stderr, "FFTDOCK: GPU kernels unavailable — "
                    "CUDA driver not loaded.\n");
    exit(1);
  }

  /* Force CUDA runtime to initialize a context before driver API calls */
  cudaFree(0);

  std::string src = Kernels::gpu_compat + "\n" + Kernels::generateLigGrid + "\n";

  nvrtcProgram prog;
  nvrtcResult nres = nvrtcCreateProgram(&prog, src.c_str(),
                                        "generateLigGrid.cu", 0, NULL, NULL);
  if (nres != NVRTC_SUCCESS) {
    fprintf(stderr, "nvrtc error: %s (createProgram lig)\n",
            nvrtcGetErrorString(nres));
    exit(-1);
  }

  CUdevice device;
  cuCtxGetDevice(&device);
  int major = 0, minor = 0;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  char archFlag[64];
  snprintf(archFlag, sizeof(archFlag), "--gpu-architecture=compute_%d%d",
           major, minor);
  const char * opts[] = { archFlag };

  nres = nvrtcCompileProgram(prog, 1, opts);
  if (nres != NVRTC_SUCCESS) {
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    char * log = new char[logSize];
    nvrtcGetProgramLog(prog, log);
    fprintf(stderr, "nvrtc compile log (lig):\n%s\n", log);
    delete[] log;
    exit(-1);
  }

  size_t ptxSize;
  nvrtcGetPTXSize(prog, &ptxSize);
  char * ptx = new char[ptxSize];
  nvrtcGetPTX(prog, ptx);
  nvrtcDestroyProgram(&prog);

  CUresult cres = cuModuleLoadData(&s_lig_module, ptx);
  delete[] ptx;
  if (cres != CUDA_SUCCESS) {
    const char * errStr = NULL;
    cuGetErrorString(cres, &errStr);
    fprintf(stderr, "cuModuleLoadData failed (lig): %s\n",
            errStr ? errStr : "unknown");
    exit(-1);
  }

  cres = cuModuleGetFunction(&s_generateLigGrid, s_lig_module,
                             "generateLigGrid");
  if (cres != CUDA_SUCCESS) {
    const char * errStr = NULL;
    cuGetErrorString(cres, &errStr);
    fprintf(stderr, "cuModuleGetFunction failed (generateLigGrid): %s\n",
            errStr ? errStr : "unknown");
    exit(-1);
  }
}

/* ---- calcLigGrid — host function called from Fortran ---- */

extern "C"
void calcLigGrid(const int BatchIdx, const int BatchSize,
        const int NumRotamers, const int NumAtoms,
        const int NumVdwGridUsed,
        const float DGrid, const int XGridNum, const int YGridNum,
        const int ZGridNum,
        float *SelectAtomsParameters, float *LigGrid,
        float *LigRotamerCoors,
        float *LigRotamerMinCoors, void* &d_LigGrid_F)
{
    compile_lig_kernel();

    const int BlockDim = 128;
    const int NumGrids = NumVdwGridUsed + 1;
    int RealBatchSize = BatchSize;
    if (BatchIdx * BatchSize > NumRotamers) {
        RealBatchSize = NumRotamers % BatchSize;
    }
    int GridDim = (BatchSize + BlockDim - 1) / BlockDim;

    int NumGridPoints = 1;
    int GridNum[3];
    GridNum[0] = XGridNum;
    GridNum[1] = YGridNum;
    GridNum[2] = ZGridNum;
    NumGridPoints = XGridNum * YGridNum * ZGridNum;

    float *d_rotamersCoor;
    float *d_GridMinCoor;
    cudaMalloc((void**)&d_rotamersCoor, BatchSize * NumAtoms * 3 * sizeof(float));
    cudaMalloc((void**)&d_GridMinCoor, BatchSize * 3 * sizeof(float));
    CUDA_CHECK();
    cudaMemcpy(d_rotamersCoor, LigRotamerCoors,
               BatchSize * NumAtoms * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_GridMinCoor, LigRotamerMinCoors,
               BatchSize * 3 * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK();

    float *d_LigGrid;
    int LigGridSize = BatchSize * NumGrids * NumGridPoints * sizeof(float);
    if (BatchIdx == 1) {
        cudaMalloc((void**)&d_LigGrid, LigGridSize);
        d_LigGrid_F = (void*)d_LigGrid;
        CUDA_CHECK();
    } else {
        d_LigGrid = (float*)(d_LigGrid_F);
    }

    cudaMemset(d_LigGrid, 0, LigGridSize);
    CUDA_CHECK();

    float *d_par;
    cudaMalloc((void**)&d_par, NumAtoms * 4 * sizeof(float));
    cudaMemcpy(d_par, SelectAtomsParameters,
               NumAtoms * 4 * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK();

    int *d_GridNum;
    cudaMalloc((void**)&d_GridNum, 3 * sizeof(int));
    cudaMemcpy(d_GridNum, GridNum, 3 * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_CHECK();

    /* Launch shared kernel via driver API */
    int numGrids = NumGrids;
    int numAtoms = NumAtoms;
    float dGrid = DGrid;

    void * args[] = {
      &RealBatchSize, &numAtoms, &numGrids,
      &d_GridNum, &dGrid,
      &d_rotamersCoor, &d_par, &d_GridMinCoor,
      &d_LigGrid
    };

    CUresult cres = cuLaunchKernel(s_generateLigGrid,
                                   GridDim, 1, 1,
                                   BlockDim, 1, 1,
                                   0, 0, args, NULL);
    if (cres != CUDA_SUCCESS) {
      const char * errStr = NULL;
      cuGetErrorString(cres, &errStr);
      fprintf(stderr, "cuLaunchKernel failed (generateLigGrid): %s\n",
              errStr ? errStr : "unknown");
      exit(-1);
    }
    CUDA_CHECK();

    cudaFree(d_par);
    cudaFree(d_rotamersCoor);
    cudaFree(d_GridMinCoor);
    cudaFree(d_GridNum);
    CUDA_CHECK();
}
