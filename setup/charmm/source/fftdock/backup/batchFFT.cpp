#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "cu2cl_util.h"




/*CU2CL Unhandled -- No main() found
CU2CL Boilerplate inserted here:
CU2CL Initialization:
__cu2cl_Init();


CU2CL Cleanup:
__cu2cl_Cleanup();
*/
cl_kernel __cu2cl_Kernel_CorrectEnergy;
cl_kernel __cu2cl_Kernel_ConjMult;
cl_kernel __cu2cl_Kernel_SumGrids;
cl_program __cu2cl_Program_batchFFT_cu;
extern const char *progSrc;
extern size_t progLen;

extern cl_device_id * __cu2cl_AllDevices;
extern cl_uint __cu2cl_AllDevices_curr_idx;
extern cl_uint __cu2cl_AllDevices_size;
extern cl_platform_id __cu2cl_Platform;
extern cl_device_id __cu2cl_Device;
extern cl_context __cu2cl_Context;
extern cl_command_queue __cu2cl_CommandQueue;

extern size_t globalWorkSize[3];
extern size_t localWorkSize[3];
void __cu2cl_Cleanup_batchFFT_cu() {
    clReleaseKernel(__cu2cl_Kernel_CorrectEnergy);
    clReleaseKernel(__cu2cl_Kernel_ConjMult);
    clReleaseKernel(__cu2cl_Kernel_SumGrids);
    clReleaseProgram(__cu2cl_Program_batchFFT_cu);
}
void __cu2cl_Init_batchFFT_cu() {
    #ifdef WITH_ALTERA
    progLen = __cu2cl_LoadProgramSource("batchFFT_cu_cl.aocx", &progSrc);
    __cu2cl_Program_batchFFT_cu = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, NULL);
    #else
    progLen = __cu2cl_LoadProgramSource("batchFFT.cu-cl.cl", &progSrc);
    __cu2cl_Program_batchFFT_cu = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, NULL);
    #endif
    free((void *) progSrc);
    clBuildProgram(__cu2cl_Program_batchFFT_cu, 1, &__cu2cl_Device, "-I . ", NULL, NULL);
    __cu2cl_Kernel_CorrectEnergy = clCreateKernel(__cu2cl_Program_batchFFT_cu, "CorrectEnergy", NULL);
    __cu2cl_Kernel_ConjMult = clCreateKernel(__cu2cl_Program_batchFFT_cu, "ConjMult", NULL);
    __cu2cl_Kernel_SumGrids = clCreateKernel(__cu2cl_Program_batchFFT_cu, "SumGrids", NULL);
}

#include <cufft.h>
#include <stdio.h>

#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {fprintf(stderr, "Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);} 

#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {fprintf(stderr, "Error %s at %s:%d\n You can try to fix this by reducing SIZB parameter in command FFTG.\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);} 


/////  GPU kernels /////

;

;

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

  // FFT transformation for potential grids
  cl_mem d_potential_f;
  cl_mem d_potential_F;
  cl_mem d_lig_F;
  cl_mem d_lig_sum_F;
  cl_mem d_lig_sum_f;
  if(idx_batch == 1){
      *(void **)&d_potential_f = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(cufftReal)*num_grid*idist, NULL, NULL);
/*CU2CL Unsupported -- Unsupported CUDA call: cudaGetErrorString*/
/*CU2CL Unsupported -- Unsupported CUDA call: cudaPeekAtLastError*/
      CUDA_CHECK();  
      clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_potential_f, CL_TRUE, 0, sizeof(cufftReal)*num_grid*idist, grid_potential, 0, NULL, NULL);
/*CU2CL Unsupported -- Unsupported CUDA call: cudaGetErrorString*/
/*CU2CL Unsupported -- Unsupported CUDA call: cudaPeekAtLastError*/
      CUDA_CHECK();
      d_GridPot_Fort = (void*) d_potential_f;

      *(void **)&d_potential_F = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(cufftComplex)*num_grid*odist, NULL, NULL);
/*CU2CL Unsupported -- Unsupported CUDA call: cudaGetErrorString*/
/*CU2CL Unsupported -- Unsupported CUDA call: cudaPeekAtLastError*/
      CUDA_CHECK();
      d_GridPot_FFT_Fort = (void*) d_potential_F;

      *(void **)&d_lig_F = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(cufftComplex)*num_grid*batch_size*odist, NULL, NULL);
/*CU2CL Unsupported -- Unsupported CUDA call: cudaGetErrorString*/
/*CU2CL Unsupported -- Unsupported CUDA call: cudaPeekAtLastError*/
      CUDA_CHECK();
      d_LigGrid_FFT_Fort = (void*) d_lig_F;

      *(void **)&d_lig_sum_F = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(cufftComplex)*batch_size*odist, NULL, NULL);
/*CU2CL Unsupported -- Unsupported CUDA call: cudaGetErrorString*/
/*CU2CL Unsupported -- Unsupported CUDA call: cudaPeekAtLastError*/
      CUDA_CHECK();
      d_LigSum_FFT_Fort = (void*) d_lig_sum_F;

      *(void **)&d_lig_sum_f = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(cufftReal)*batch_size*idist, NULL, NULL);
/*CU2CL Unsupported -- Unsupported CUDA call: cudaGetErrorString*/
/*CU2CL Unsupported -- Unsupported CUDA call: cudaPeekAtLastError*/
      CUDA_CHECK();
      d_LigSum_Fort = (void*) d_lig_sum_f;
  }else{
      d_potential_f = (cufftReal*) d_GridPot_Fort;
      d_potential_F = (cufftComplex*) d_GridPot_FFT_Fort;
      d_lig_F = (cufftComplex*) d_LigGrid_FFT_Fort;
      d_lig_sum_F = (cufftComplex*) d_LigSum_FFT_Fort;
      d_lig_sum_f = (cufftReal*) d_LigSum_Fort;
  }
  
/*CU2CL Unsupported -- Unsupported CUDA call: cufftExecR2C*/
  cufftResult potentialRes = cufftExecR2C(*potential_R2C_plan, d_potential_f, d_potential_F);
/*CU2CL Unsupported -- Unsupported CUDA call: cudaGetErrorString*/
/*CU2CL Unsupported -- Unsupported CUDA call: cudaPeekAtLastError*/
  CUDA_CHECK();
  
  if (potentialRes != CUFFT_SUCCESS)
  {
    fprintf(stderr, "%s", "Potential transform failed!\n");
  }
  
  // FFT transform for ligand grids
  //printf("FFT memory=%d\n",d_LigGrid_Fort);
  cufftReal* d_lig_f = (cufftReal*)d_LigGrid_Fort;
  //cudaMalloc((void **)&d_lig_f, sizeof(cufftReal)*num_grid*batch_size*idist);  
  //CUDA_CHECK();    
  //cudaMemcpy(d_lig_f, LigGrid,
  // 	     sizeof(cufftReal)*num_grid*batch_size*idist,
  //     cudaMemcpyHostToDevice);

/*CU2CL Unsupported -- Unsupported CUDA call: cufftExecR2C*/
  cufftResult ligRes = cufftExecR2C(*lig_R2C_plan, d_lig_f, d_lig_F);
  
  if (ligRes != CUFFT_SUCCESS)
  {
    fprintf(stderr, "%s", "Lig transform failed!");
  }

  // Inverse FFT transform to calcualte energy grids
  
  ConjMult<<<1024, 256>>>(batch_size*num_grid*odist, d_potential_F, d_lig_F, odist, num_grid);
/*CU2CL Unsupported -- Unsupported CUDA call: cudaGetErrorString*/
/*CU2CL Unsupported -- Unsupported CUDA call: cudaPeekAtLastError*/
  CUDA_CHECK();
  
  SumGrids<<<1024, 256>>>(batch_size*odist, d_lig_F, d_lig_sum_F, num_grid, odist, idist);
/*CU2CL Unsupported -- Unsupported CUDA call: cudaGetErrorString*/
/*CU2CL Unsupported -- Unsupported CUDA call: cudaPeekAtLastError*/
  CUDA_CHECK();
  
/*CU2CL Unsupported -- Unsupported CUDA call: cufftExecC2R*/
  cufftResult fftRes = cufftExecC2R(*C2R_plan, d_lig_sum_F, d_lig_sum_f);
    
  if (fftRes != CUFFT_SUCCESS)
  {
    fprintf(stderr, "%s", "Reverse transform failed!");
  }
  
  // copy energy grid from GPU to CPU
  //printf("batch_size: %d \n", batch_size);
  //printf("idist: %d \n", idist);  

  CorrectEnergy<<<1024, 256>>>(batch_size*idist, idist, d_lig_sum_f);

  clEnqueueReadBuffer(__cu2cl_CommandQueue, d_lig_sum_f, CL_TRUE, 0, sizeof(float)*batch_size*idist, EnergyGrid, 0, NULL, NULL);

  //removed for efficiency
  //for (int i = 0; i < batch_size*idist; i++)
  //{
  //  EnergyGrid[i] = EnergyGrid[i] / sqrt(idist);
  //}
 
  //GPU memory deallocation was moved to clean_FFTDock_GPU
  //cudaFree(d_potential_f);
  //cudaFree(d_potential_F);
  //cudaFree(d_lig_f);
  //cudaFree(d_lig_F);
  //cudaFree(d_lig_sum_F);
  //cudaFree(d_lig_sum_f);  
}

extern "C"
void destroy_cufft_plan(cufftHandle* plan)
{
/*CU2CL Unsupported -- Unsupported CUDA call: cufftDestroy*/
  cufftDestroy(*plan);
/*CU2CL Unsupported -- Unsupported CUDA call: cudaDeviceReset*/
  cudaDeviceReset();
}

extern "C"
void allocate_GPU_id(const int gpuid)
{ 
  int nDevices;
  clGetDeviceIDs(__cu2cl_Platform, CL_DEVICE_TYPE_GPU, 0, NULL, (cl_uint *) &nDevices);
  fprintf(stdout, "Num of GPU Devices: %d\n", nDevices);
  fprintf(stdout, "The device %d is used. \n", gpuid);
/*CU2CL Warning -- CU2CL Identified cudaSetDevice usage*/
  __cu2cl_SetDevice(gpuid);
  __cu2cl_DeviceProp prop;
  __cu2cl_GetDeviceProperties(&prop, gpuid);  
  fprintf(stdout, "  GPU Devices Name: %s\n", prop.name);
  fprintf(stdout, "  total global devices memory: %zu MB\n", prop.totalGlobalMem / (1024*1024));
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

/*CU2CL Unsupported -- Unsupported CUDA call: cufftPlanMany*/
  cufftResult potentialRes = cufftPlanMany(plan, 3, n,
  					   inembed, istride, idist,
  					   onembed, ostride, odist,
  					   CUFFT_R2C, batch_size);  
  if (potentialRes != CUFFT_SUCCESS)
  {
    fprintf(stderr, "%s", "make cufft R2C plan failed!");
  }  
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
  
/*CU2CL Unsupported -- Unsupported CUDA call: cufftPlanMany*/
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
    
  cl_mem d_potential_f;

  *(void **)&d_potential_f = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(cufftReal)*batch_size*idist, NULL, NULL);
  clEnqueueWriteBuffer(__cu2cl_CommandQueue, d_potential_f, CL_TRUE, 0, sizeof(cufftReal)*batch_size*idist, grid_potential, 0, NULL, NULL);
  cl_mem d_potential_F;
  *(void **)&d_potential_F = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, sizeof(cufftComplex)*batch_size*odist, NULL, NULL);

/*CU2CL Unsupported -- Unsupported CUDA call: cufftExecR2C*/
  cufftResult potentialRes = cufftExecR2C(*plan, d_potential_f, d_potential_F);
  
  if (potentialRes != CUFFT_SUCCESS)
  {
    fprintf(stderr, "%s", "Potential transform failed!");
  }
  clReleaseMemObject(d_potential_f);
  clReleaseMemObject(d_potential_F);
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
    clReleaseMemObject(d_lig_f);
    clReleaseMemObject(d_lig_F);
    clReleaseMemObject(d_potential_f);
    clReleaseMemObject(d_potential_F);
    clReleaseMemObject(d_lig_sum_f);
    clReleaseMemObject(d_lig_sum_F);
/*CU2CL Unsupported -- Unsupported CUDA call: cudaGetErrorString*/
/*CU2CL Unsupported -- Unsupported CUDA call: cudaPeekAtLastError*/
    CUDA_CHECK();
}

