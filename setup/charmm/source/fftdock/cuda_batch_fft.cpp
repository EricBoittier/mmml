#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cuda.h>
#include "cuda_driver_api.h"
#include <nvrtc.h>
#include <stdio.h>
#include <string>
#include <kernels.h>

#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {fprintf(stderr, "Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);}

#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {fprintf(stderr, "Error %s at %s:%d\n You can try to fix this by reducing SIZB parameter in command FFTG.\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);}

/* ================================================================== */
/*  nvrtc kernel compilation from shared kernel source strings         */
/* ================================================================== */

static CUmodule   s_fftdock_module = NULL;
static CUfunction s_correctEnergy  = NULL;
static CUfunction s_conjMult       = NULL;
static CUfunction s_sumGrids       = NULL;

static void nvrtc_check(nvrtcResult res, const char * msg) {
  if (res != NVRTC_SUCCESS) {
    fprintf(stderr, "nvrtc error: %s (%s)\n", nvrtcGetErrorString(res), msg);
    exit(-1);
  }
}

static void cu_check(CUresult res, const char * msg) {
  if (res != CUDA_SUCCESS) {
    const char * errStr = NULL;
    cuGetErrorString(res, &errStr);
    fprintf(stderr, "CUDA driver error: %s (%s)\n",
            errStr ? errStr : "unknown", msg);
    exit(-1);
  }
}

/*
 * Compile all 3 shared fftdock kernels from the embedded source strings
 * (gpu_compat.h + conjMult.cl + sumGrids.cl + correctEnergy.cl) using
 * nvrtc, and cache the resulting CUfunction handles.
 *
 * Called lazily on first invocation of rigid_FFT_dock.
 */
static void compile_shared_kernels() {
  if (s_fftdock_module) return;  /* already compiled */

  if (cuda_driver_load() != 0) {
    fprintf(stderr, "FFTDOCK: GPU kernels unavailable — "
                    "CUDA driver not loaded.\n");
    exit(1);
  }

  /* Force CUDA runtime to initialize a context before driver API calls */
  cudaFree(0);

  /* Combine all kernel sources with the compat header into one program */
  std::string src = Kernels::gpu_compat + "\n"
                  + Kernels::correctEnergy + "\n"
                  + Kernels::conjMult + "\n"
                  + Kernels::sumGrids + "\n";

  nvrtcProgram prog;
  nvrtcResult nres;
  nres = nvrtcCreateProgram(&prog, src.c_str(), "fftdock_kernels.cu",
                            0, NULL, NULL);
  nvrtc_check(nres, "nvrtcCreateProgram");

  /* Compile for the current device architecture (driver API) */
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
    fprintf(stderr, "nvrtc compile log:\n%s\n", log);
    delete[] log;
    nvrtc_check(nres, "nvrtcCompileProgram");
  }

  /* Extract PTX */
  size_t ptxSize;
  nres = nvrtcGetPTXSize(prog, &ptxSize);
  nvrtc_check(nres, "nvrtcGetPTXSize");
  char * ptx = new char[ptxSize];
  nres = nvrtcGetPTX(prog, ptx);
  nvrtc_check(nres, "nvrtcGetPTX");

  nvrtcDestroyProgram(&prog);

  /* Load PTX into a CUDA module */
  CUresult cres;
  cres = cuModuleLoadData(&s_fftdock_module, ptx);
  cu_check(cres, "cuModuleLoadData");
  delete[] ptx;

  /* Get kernel function handles */
  cres = cuModuleGetFunction(&s_correctEnergy, s_fftdock_module,
                             "correctEnergy");
  cu_check(cres, "cuModuleGetFunction(correctEnergy)");

  cres = cuModuleGetFunction(&s_conjMult, s_fftdock_module, "conjMult");
  cu_check(cres, "cuModuleGetFunction(conjMult)");

  cres = cuModuleGetFunction(&s_sumGrids, s_fftdock_module, "sumGrids");
  cu_check(cres, "cuModuleGetFunction(sumGrids)");

  fprintf(stdout, "FFTDOCK: shared kernels compiled via nvrtc\n");
}

/* ================================================================== */
/*  rigid_FFT_dock — main FFT docking pipeline                         */
/* ================================================================== */

extern "C"
void rigid_FFT_dock(const int xdim, const int ydim, const int zdim,
		    const int batch_size, const int idx_batch,
		    const int num_quaternions, const int num_grid,
		    cufftHandle* potential_R2C_plan,
		    cufftHandle* lig_R2C_plan,
		    cufftHandle* C2R_plan,
		    float* grid_potential, float* LigGrid, float* EnergyGrid,
            void* &d_LigGrid_Fort,
            void* &d_LigGrid_FFT_Fort,
            void* &d_GridPot_Fort,
            void* &d_GridPot_FFT_Fort,
            void* &d_LigSum_Fort,
            void* &d_LigSum_FFT_Fort)
{
  compile_shared_kernels();

  int idist = xdim * ydim * zdim;
  int odist = xdim * ydim * (zdim / 2 + 1);

  /* ------ allocate GPU buffers (first batch only) ------ */

  cufftReal* d_potential_f;
  cufftComplex *d_potential_F;
  cufftComplex *d_lig_F;
  cufftComplex * d_lig_sum_F;
  cufftReal *d_lig_sum_f;
  if(idx_batch == 1){
      cudaMalloc((void **)&d_potential_f, sizeof(cufftReal)*num_grid*idist);
      CUDA_CHECK();
      cudaMemcpy(d_potential_f, grid_potential,
              sizeof(cufftReal)*num_grid*idist,
              cudaMemcpyHostToDevice);
      CUDA_CHECK();
      d_GridPot_Fort = (void*) d_potential_f;

      cudaMalloc((void **)&d_potential_F, sizeof(cufftComplex)*num_grid*odist);
      CUDA_CHECK();
      d_GridPot_FFT_Fort = (void*) d_potential_F;

      cudaMalloc((void **)&d_lig_F, sizeof(cufftComplex)*num_grid*batch_size*odist);
      CUDA_CHECK();
      d_LigGrid_FFT_Fort = (void*) d_lig_F;

      cudaMalloc((void **)&d_lig_sum_F, sizeof(cufftComplex)*batch_size*odist);
      CUDA_CHECK();
      d_LigSum_FFT_Fort = (void*) d_lig_sum_F;

      cudaMalloc((void **)&d_lig_sum_f, sizeof(cufftReal)*batch_size*idist);
      CUDA_CHECK();
      d_LigSum_Fort = (void*) d_lig_sum_f;
  }else{
      d_potential_f = (cufftReal*) d_GridPot_Fort;
      d_potential_F = (cufftComplex*) d_GridPot_FFT_Fort;
      d_lig_F = (cufftComplex*) d_LigGrid_FFT_Fort;
      d_lig_sum_F = (cufftComplex*) d_LigSum_FFT_Fort;
      d_lig_sum_f = (cufftReal*) d_LigSum_Fort;
  }

  /* ------ 1. Forward R2C: potential grids ------ */

  cufftResult potentialRes = cufftExecR2C(*potential_R2C_plan, d_potential_f, d_potential_F);
  CUDA_CHECK();
  if (potentialRes != CUFFT_SUCCESS)
    fprintf(stderr, "Potential transform failed!\n");

  /* ------ 2. Forward R2C: ligand grids ------ */

  cufftReal* d_lig_f = (cufftReal*)d_LigGrid_Fort;
  cufftResult ligRes = cufftExecR2C(*lig_R2C_plan, d_lig_f, d_lig_F);
  if (ligRes != CUFFT_SUCCESS)
    fprintf(stderr, "Lig transform failed!\n");

  /* ------ 3. Conjugate multiplication (shared kernel via nvrtc) ------ */
  /*
   * The shared kernel takes float* (interleaved real/imag pairs).
   * cufftComplex is a struct of two floats, so casting is safe.
   */
  {
    int conj_N = batch_size * num_grid * odist;
    int conj_odist = odist;
    int conj_ngrids = num_grid;
    float * pot_f = (float *)d_potential_F;
    float * lig_f_ptr = (float *)d_lig_F;
    void * args[] = { &conj_N, &pot_f, &lig_f_ptr, &conj_odist, &conj_ngrids };
    CUresult cres = cuLaunchKernel(s_conjMult,
                                   1024, 1, 1,   /* grid */
                                   256, 1, 1,    /* block */
                                   0, 0,         /* shared mem, stream */
                                   args, NULL);
    cu_check(cres, "launch conjMult");
  }
  CUDA_CHECK();

  /* ------ 4. Sum grids across grid types (shared kernel via nvrtc) ------ */
  {
    int sum_N = batch_size * odist;
    int sum_ngrids = num_grid;
    int sum_odist = odist;
    int sum_idist = idist;
    float * lig_f_ptr = (float *)d_lig_F;
    float * sum_f_ptr = (float *)d_lig_sum_F;
    void * args[] = { &sum_N, &lig_f_ptr, &sum_f_ptr,
                      &sum_ngrids, &sum_odist, &sum_idist };
    CUresult cres = cuLaunchKernel(s_sumGrids,
                                   1024, 1, 1,
                                   256, 1, 1,
                                   0, 0,
                                   args, NULL);
    cu_check(cres, "launch sumGrids");
  }
  CUDA_CHECK();

  /* ------ 5. Inverse C2R: energy grids ------ */

  cufftResult fftRes = cufftExecC2R(*C2R_plan, d_lig_sum_F, d_lig_sum_f);
  if (fftRes != CUFFT_SUCCESS)
    fprintf(stderr, "Reverse transform failed!\n");

  /* ------ 6. Correct energy (shared kernel via nvrtc) ------ */
  {
    int ener_N = batch_size * idist;
    int ener_idist = idist;
    float * sum_real = (float *)d_lig_sum_f;
    void * args[] = { &ener_N, &ener_idist, &sum_real };
    CUresult cres = cuLaunchKernel(s_correctEnergy,
                                   1024, 1, 1,
                                   256, 1, 1,
                                   0, 0,
                                   args, NULL);
    cu_check(cres, "launch correctEnergy");
  }

  /* ------ 7. Copy results back to host ------ */

  cudaMemcpy(EnergyGrid, d_lig_sum_f, sizeof(float)*batch_size*idist,
	     cudaMemcpyDeviceToHost);
}

/* ================================================================== */
/*  Plan management and GPU setup (unchanged)                          */
/* ================================================================== */

extern "C"
void destroy_cufft_plan(cufftHandle* plan)
{
  cufftDestroy(*plan);
}

extern "C"
void allocate_GPU_id(const int gpuid)
{
  if (cuda_driver_load() != 0) {
    fprintf(stderr, "FFTDOCK: Cannot allocate GPU — "
                    "CUDA driver not available on this machine.\n"
                    "FFTDOCK: Submit your job to a GPU compute node.\n");
    exit(1);
  }

  /* Initialize the CUDA driver API (required before any cu* calls) */
  cuInit(0);

  int nDevices;
  cudaGetDeviceCount(&nDevices);
  fprintf(stdout, "Num of GPU Devices: %d\n", nDevices);
  fprintf(stdout, "The device %d is used. \n", gpuid);
  cudaSetDevice(gpuid);

  CUdevice cudev;
  cuDeviceGet(&cudev, gpuid);
  char devName[256];
  cuDeviceGetName(devName, sizeof(devName), cudev);
  size_t totalMem = 0;
  cuDeviceTotalMem(&totalMem, cudev);
  fprintf(stdout, "  GPU Devices Name: %s\n", devName);
  fprintf(stdout, "  total global devices memory: %zu MB\n", totalMem / (1024*1024));
};

extern "C"
void make_cufft_R2C_plan(const int xdim, const int ydim, const int zdim,
		         const int batch_size, cufftHandle* plan)
{
  int n[3];
  n[0] = xdim;
  n[1] = ydim;
  n[2] = zdim;

  int inembed[3];
  inembed[0] = xdim;
  inembed[1] = ydim;
  inembed[2] = zdim;
  int idist = inembed[0] * inembed[1] * inembed[2];
  int istride = 1;

  int onembed[3];
  onembed[0] = xdim;
  onembed[1] = ydim;
  onembed[2] = zdim/2 + 1;
  int odist = onembed[0] * onembed[1] * onembed[2];
  int ostride = 1;

  cufftResult potentialRes = cufftPlanMany(plan, 3, n,
  					   inembed, istride, idist,
  					   onembed, ostride, odist,
  					   CUFFT_R2C, batch_size);
  size_t grid_size;
  cufftResult sizeRes = cufftEstimateMany(3, n, inembed, istride, idist,
                                          onembed, ostride, odist, CUFFT_R2C,
                                          batch_size, &grid_size);

  if (potentialRes != CUFFT_SUCCESS)
  {
    fprintf(stderr, "%s", "make cufft R2C plan failed!");
  }
  printf("Batch size is %d\n", batch_size);
  printf("CuFFT result is %d\n", potentialRes);
  printf("Estimated grid memory is %d\n", grid_size / (1024 * 1024));
};

extern "C"
void make_cufft_C2R_plan(const int xdim, const int ydim, const int zdim,
		         const int batch_size, cufftHandle* plan)
{

  int n[3];
  n[0] = xdim;
  n[1] = ydim;
  n[2] = zdim;

  int inembed[3];
  inembed[0] = xdim;
  inembed[1] = ydim;
  inembed[2] = zdim;
  int idist = inembed[0] * inembed[1] * inembed[2];
  int istride = 1;

  int onembed[3];
  onembed[0] = xdim;
  onembed[1] = ydim;
  onembed[2] = zdim/2 + 1;
  int odist = onembed[0] * onembed[1] * onembed[2];
  int ostride = 1;

  cufftResult potentialRes = cufftPlanMany(plan, 3, n,
  					   onembed, ostride, odist,
  					   inembed, istride, idist,
  					   CUFFT_C2R, batch_size);

  if (potentialRes != CUFFT_SUCCESS)
  {
    fprintf(stderr, "%s", "make cuttf C2R plan failed!");
  }
};

extern "C"
void batchFFT(const int xdim, const int ydim, const int zdim,
		  const int batch_size,
		  cufftHandle* plan, float *grid_potential)
{
  int inembed[3];
  inembed[0] = xdim;
  inembed[1] = ydim;
  inembed[2] = zdim;
  int idist = inembed[0] * inembed[1] * inembed[2];

  int onembed[3];
  onembed[0] = xdim;
  onembed[1] = ydim;
  onembed[2] = zdim/2 + 1;
  int odist = onembed[0] * onembed[1] * onembed[2];

  cufftReal* d_potential_f;

  cudaMalloc((void **)&d_potential_f, sizeof(cufftReal)*batch_size*idist);
  cudaMemcpy(d_potential_f, grid_potential,
  	     sizeof(cufftReal)*batch_size*idist,
  	     cudaMemcpyHostToDevice);
  cufftComplex *d_potential_F;
  cudaMalloc((void **)&d_potential_F, sizeof(cufftComplex)*batch_size*odist);

  cufftResult potentialRes = cufftExecR2C(*plan, d_potential_f, d_potential_F);

  if (potentialRes != CUFFT_SUCCESS)
  {
    fprintf(stderr, "%s", "Potential transform failed!");
  }
  cudaFree(d_potential_f);
  cudaFree(d_potential_F);
}

extern "C"
void clean_FFTDock_GPU(
        void* &d_LigGrid_Fort,
        void* &d_LigGrid_FFT_Fort,
        void* &d_GridPot_Fort,
        void* &d_GridPot_FFT_Fort,
        void* &d_LigSum_Fort,
        void* &d_LigSum_FFT_Fort){
    cufftReal* d_lig_f = (cufftReal*)d_LigGrid_Fort;
    cufftComplex* d_lig_F = (cufftComplex*)d_LigGrid_FFT_Fort;
    cufftReal* d_potential_f = (cufftReal*) d_GridPot_Fort;
    cufftComplex* d_potential_F = (cufftComplex*) d_GridPot_FFT_Fort;
    cufftComplex* d_lig_sum_F = (cufftComplex*) d_LigSum_FFT_Fort;
    cufftReal* d_lig_sum_f = (cufftReal*) d_LigSum_Fort;
    cudaFree(d_lig_f);
    cudaFree(d_lig_F);
    cudaFree(d_potential_f);
    cudaFree(d_potential_F);
    cudaFree(d_lig_sum_f);
    cudaFree(d_lig_sum_F);
    CUDA_CHECK();

    /* Clean up nvrtc module */
    if (s_fftdock_module) {
      cuModuleUnload(s_fftdock_module);
      s_fftdock_module = NULL;
      s_correctEnergy = NULL;
      s_conjMult = NULL;
      s_sumGrids = NULL;
    }
}
