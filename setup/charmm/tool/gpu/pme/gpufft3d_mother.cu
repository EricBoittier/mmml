#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <gpuheader.h>
#include <cufft.h>
extern "C"
void gpufft3d_  (float*, int, int, int, int, int);
void gpufft3d__ (float*, int, int, int, int, int);
extern "C"
void
gpufft3d_ (float* x, int size_x, int size_y, int size_z, int fftflag, int idevice)
{
   /* CUDA_SAFE_CALL(cudaSetDevice(idevice)); */
   unsigned int mem_size_A = sizeof(cufftComplex) * size_x * size_y * size_z;
   cufftHandle plan;
   cufftComplex *idata, *odata;
   CUDA_SAFE_CALL(cudaMalloc((void**) &idata, mem_size_A));
   CUDA_SAFE_CALL(cudaMalloc((void**) &odata, mem_size_A));
   CUDA_SAFE_CALL(cudaMemcpy(idata, x, mem_size_A, cudaMemcpyHostToDevice));
   cufftPlan3d(&plan, size_x, size_y, size_z, CUFFT_C2C);
   if (fftflag == 1){
     cufftExecC2C(plan, idata, odata, CUFFT_FORWARD);
   }
   if (fftflag == -1){
     cufftExecC2C(plan, idata, odata, CUFFT_INVERSE);
   }
   CUDA_SAFE_CALL(cudaMemcpy(x, odata, mem_size_A,cudaMemcpyDeviceToHost));
   cufftDestroy(plan);
   CUDA_SAFE_CALL(cudaFree(idata));
   CUDA_SAFE_CALL(cudaFree(odata));
}
extern "C"
void
gpufft3d__ (float* x, int* size_x, int* size_y, int* size_z, int* flagfft, int* idevice)
{
  gpufft3d_ (x, *size_x, *size_y, *size_z, *flagfft, *idevice);
}
