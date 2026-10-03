#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cuda_driver_api.h"
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <kernels.h>

#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {fprintf(stderr, "Error %s at %s:%d\n in generating protein grid using GPU.\n",\
          cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);}

/* ---- nvrtc compilation of shared generateProtGrid kernel ---- */

static CUmodule   s_pot_module = NULL;
static CUfunction s_generateProtGrid = NULL;

static void compile_pot_kernel() {
  if (s_pot_module) return;

  if (cuda_driver_load() != 0) {
    fprintf(stderr, "FFTDOCK: GPU kernels unavailable — "
                    "CUDA driver not loaded.\n");
    exit(1);
  }

  /* Initialize driver API and force runtime context creation */
  cuInit(0);
  cudaFree(0);

  std::string src = Kernels::gpu_compat + "\n" + Kernels::generateProtGrid + "\n";

  nvrtcProgram prog;
  nvrtcResult nres = nvrtcCreateProgram(&prog, src.c_str(),
                                        "generateProtGrid.cu", 0, NULL, NULL);
  if (nres != NVRTC_SUCCESS) {
    fprintf(stderr, "nvrtc error: %s (createProgram pot)\n",
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
    fprintf(stderr, "nvrtc compile log (pot):\n%s\n", log);
    delete[] log;
    exit(-1);
  }

  size_t ptxSize;
  nvrtcGetPTXSize(prog, &ptxSize);
  char * ptx = new char[ptxSize];
  nvrtcGetPTX(prog, ptx);
  nvrtcDestroyProgram(&prog);

  CUresult cres = cuModuleLoadData(&s_pot_module, ptx);
  delete[] ptx;
  if (cres != CUDA_SUCCESS) {
    const char * errStr = NULL;
    cuGetErrorString(cres, &errStr);
    fprintf(stderr, "cuModuleLoadData failed (pot): %s\n",
            errStr ? errStr : "unknown");
    exit(-1);
  }

  cres = cuModuleGetFunction(&s_generateProtGrid, s_pot_module,
                             "generateProtGrid");
  if (cres != CUDA_SUCCESS) {
    const char * errStr = NULL;
    cuGetErrorString(cres, &errStr);
    fprintf(stderr, "cuModuleGetFunction failed (generateProtGrid): %s\n",
            errStr ? errStr : "unknown");
    exit(-1);
  }
}

/* ---- calcPotGrid — host function called from Fortran ---- */

extern "C"
void calcPotGrid(const int NumGrids, const int NumAtoms,
        const float DGrid, const int XGridLen, const int YGridLen,
        const int ZGridLen, const float XMin, const float YMin, const float ZMin,
        const float Fa, const float Fb, const float Gmax,
        const float VdwEmax, const float ElecAttrEmax,
        const float ElecReplEmax, const float CCELEC, const int ElecMode,
        const float Dielec, float *SelectAtomsParameters, float *GridPot,
        float *GridRadii)
{
    compile_pot_kernel();

    const int NUM_PARAMS_PER_ATOM = 8;

    int GridNum[3];
    int NumGridPoints = 1;
    float GridMinCoor[3];
    GridNum[0] = XGridLen;
    GridNum[1] = YGridLen;
    GridNum[2] = ZGridLen;
    GridMinCoor[0] = XMin;
    GridMinCoor[1] = YMin;
    GridMinCoor[2] = ZMin;
    NumGridPoints = GridNum[0]*GridNum[1]*GridNum[2];

    int paramMemSize = NumAtoms*NUM_PARAMS_PER_ATOM*sizeof(float);
    int gridMemSize = NumGridPoints*NumGrids*sizeof(float);
    int probesMemSize = (NumGrids-3)*sizeof(float);

    float* d_parameter;
    float* d_GridPot;
    float* d_probes;
    int* d_GridNum;
    float* d_GridMinCoor;
    cudaMalloc((void**)&d_parameter, paramMemSize);
    cudaMalloc((void**)&d_GridNum, 3*sizeof(int));
    cudaMalloc((void**)&d_GridMinCoor, 3*sizeof(float));
    cudaMalloc((void**)&d_GridPot, gridMemSize);
    cudaMalloc((void**)&d_probes, probesMemSize);
    CUDA_CHECK()

    cudaMemcpy(d_parameter, SelectAtomsParameters, paramMemSize,
            cudaMemcpyHostToDevice);
    cudaMemcpy(d_probes, GridRadii, probesMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_GridNum, GridNum, 3*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_GridMinCoor, GridMinCoor, 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_GridPot, 0, gridMemSize);
    CUDA_CHECK()

    int BlockDim = 512;
    int GridDim = (NumGridPoints + BlockDim - 1) / BlockDim;

    /* Launch shared kernel via driver API */
    int nGrids = NumGrids;
    int nAtoms = NumAtoms;
    float dGrid = DGrid;
    float fa = Fa, fb = Fb, gmax = Gmax;
    float vdwEmax = VdwEmax;
    float elecReplEmax = ElecReplEmax;
    float elecAttrEmax = ElecAttrEmax;
    float ccelec = CCELEC;
    int elecMode = ElecMode;
    float dielec = Dielec;

    void * args[] = {
      &d_probes, &d_parameter, &d_GridPot,
      &nGrids, &nAtoms,
      &d_GridNum, &d_GridMinCoor,
      &fa, &fb, &gmax, &dGrid, &vdwEmax,
      &elecReplEmax, &elecAttrEmax,
      &ccelec, &elecMode, &dielec
    };

    CUresult cres = cuLaunchKernel(s_generateProtGrid,
                                   GridDim, 1, 1,
                                   BlockDim, 1, 1,
                                   0, 0, args, NULL);
    if (cres != CUDA_SUCCESS) {
      const char * errStr = NULL;
      cuGetErrorString(cres, &errStr);
      fprintf(stderr, "cuLaunchKernel failed (generateProtGrid): %s\n",
              errStr ? errStr : "unknown");
      exit(-1);
    }
    CUDA_CHECK()

    cudaMemcpy(GridPot, d_GridPot, gridMemSize, cudaMemcpyDeviceToHost);
    CUDA_CHECK()

    cudaFree(d_parameter);
    cudaFree(d_GridNum);
    cudaFree(d_GridMinCoor);
    cudaFree(d_GridPot);
    cudaFree(d_probes);
    CUDA_CHECK()
}
