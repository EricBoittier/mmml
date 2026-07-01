#include <cufft.h>
#include <stdio.h>

#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {fprintf(stderr, "Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);} 

#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {fprintf(stderr, "Error %s at %s:%d\n You can try to fix this by reducing SIZB parameter in command FFTG.\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);} 


/////  GPU kernels /////
__global__ void CorrectEnergy(const int N, const int idist, cufftReal *d_lig_sum_f){ 
    for (int idx = blockIdx.x*blockDim.x + threadIdx.x;
            idx < N;
            idx += blockDim.x*gridDim.x ){
        d_lig_sum_f[idx] = d_lig_sum_f[idx] / idist;
    }
}
__global__ void ConjMult(int N,
			 cufftComplex *d_potential_F,
			 cufftComplex *d_ligand_F,
			 int odist, int numOfGridsUsed)
{
  int dist = odist * numOfGridsUsed;
  int idx_p;
  float x, y;
  for (int idx_l = blockIdx.x * blockDim.x + threadIdx.x;
       idx_l < N;
       idx_l += blockDim.x * gridDim.x)
  {
    idx_p = idx_l % dist;
    x = d_ligand_F[idx_l].x;
    y = d_ligand_F[idx_l].y;
    d_ligand_F[idx_l].x = x * d_potential_F[idx_p].x + y * d_potential_F[idx_p].y;
    d_ligand_F[idx_l].y = x * d_potential_F[idx_p].y - y * d_potential_F[idx_p].x;
  }
};

__global__ void SumGrids(int N_sum_F,
			 cufftComplex *d_ligand_F, cufftComplex *d_ligand_sum_F,
			 int numOfGridsUsed, int odist, int idist)
{
  for (int idx_sum_F = blockIdx.x * blockDim.x + threadIdx.x;
       idx_sum_F < N_sum_F;
       idx_sum_F += blockDim.x * gridDim.x)
  {
    int idx_rotamer = idx_sum_F / odist;
    int idx_point = idx_sum_F % odist;
    d_ligand_sum_F[idx_sum_F].x = 0;
    d_ligand_sum_F[idx_sum_F].y = 0;
    for (int idx_grid_type = 0; idx_grid_type < numOfGridsUsed; idx_grid_type++)
    {
      int idx_F = (idx_rotamer * numOfGridsUsed + idx_grid_type) * odist + idx_point;
      d_ligand_sum_F[idx_sum_F].x += d_ligand_F[idx_F].x;
      d_ligand_sum_F[idx_sum_F].y += d_ligand_F[idx_F].y;
    }
    d_ligand_sum_F[idx_sum_F].x = d_ligand_sum_F[idx_sum_F].x;
    d_ligand_sum_F[idx_sum_F].y = d_ligand_sum_F[idx_sum_F].y;
  }  
};

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
  
  cufftResult potentialRes = cufftExecR2C(*potential_R2C_plan, d_potential_f, d_potential_F);
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

  cufftResult ligRes = cufftExecR2C(*lig_R2C_plan, d_lig_f, d_lig_F);
  
  if (ligRes != CUFFT_SUCCESS)
  {
    fprintf(stderr, "%s", "Lig transform failed!");
  }

  // Inverse FFT transform to calcualte energy grids
  
  ConjMult<<<1024, 256>>>(batch_size*num_grid*odist, d_potential_F, d_lig_F, odist, num_grid);
  CUDA_CHECK();
  
  SumGrids<<<1024, 256>>>(batch_size*odist, d_lig_F, d_lig_sum_F, num_grid, odist, idist);
  CUDA_CHECK();
  
  cufftResult fftRes = cufftExecC2R(*C2R_plan, d_lig_sum_F, d_lig_sum_f);
    
  if (fftRes != CUFFT_SUCCESS)
  {
    fprintf(stderr, "%s", "Reverse transform failed!");
  }
  
  // copy energy grid from GPU to CPU
  //printf("batch_size: %d \n", batch_size);
  //printf("idist: %d \n", idist);  

  CorrectEnergy<<<1024, 256>>>(batch_size*idist, idist, d_lig_sum_f);

  cudaMemcpy(EnergyGrid, d_lig_sum_f, sizeof(float)*batch_size*idist,
	     cudaMemcpyDeviceToHost);

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
  cufftDestroy(*plan);
  cudaDeviceReset();
}

extern "C"
void allocate_GPU_id(const int gpuid)
{ 
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  fprintf(stdout, "Num of GPU Devices: %d\n", nDevices);
  fprintf(stdout, "The device %d is used. \n", gpuid);
  cudaSetDevice(gpuid);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, gpuid);  
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
}

