#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include <gpuheader.h>
#include <cufft.h>

////////////////////////////////////////////////////////////////////////////////
extern "C"
void gpufft3d_  (double*, double*, int, int, int, int);
void gpufft3d__ (double*, double*, int, int, int, int);
////////////////////////////////////////////////////////////////////////////////
//! 3D-FFT
////////////////////////////////////////////////////////////////////////////////
extern "C"
void
gpufft3d_ (double* xd_re, double* xd_im, int size_x, int size_y, int size_z, int fftflag)
{
   CUT_CHECK_DEVICE();
   int itmp;
   unsigned int size_A = size_x * size_y * size_z;
   unsigned int mem_size_A = sizeof(float) * size_A * 2;
   float* x_float = (float*) malloc(mem_size_A);
   for (int i = 0; i < size_x; i++)
     for (int j = 0; j < size_y; j++)
       for (int k = 0; k < size_z; k++){
	 itmp = i + j*size_x + k*size_x*size_y;
	 x_float[itmp*2]   = (float)xd_re[itmp];
	 x_float[itmp*2+1] = (float)xd_im[itmp];
       }
   cufftHandle plan;
   cufftComplex *data;
   CUDA_SAFE_CALL(cudaMalloc((void**) &data, sizeof(cufftComplex)*size_x*size_y*size_z));
   CUDA_SAFE_CALL(cudaMemcpy(data, x_float, mem_size_A, cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cufftPlan3d(&plan, size_x, size_y, size_z, CUFFT_C2C));
   if (fftflag == 1){
     CUDA_SAFE_CALL(cufftExecC2C(plan, data, data, CUFFT_FORWARD));
   }
   if (fftflag == -1){
     CUDA_SAFE_CALL(cufftExecC2C(plan, data, data, CUFFT_INVERSE));
   }
   CUDA_SAFE_CALL(cudaMemcpy(x_float, data, mem_size_A,cudaMemcpyDeviceToHost));
   for (int i = 0; i < size_x; i++)
     for (int j = 0; j < size_y; j++)
       for (int k = 0; k < size_z; k++){
	 itmp = i + j*size_x + k*size_x*size_y;
	 xd_re[itmp] = (double)x_float[itmp*2];
	 xd_im[itmp] = (double)x_float[itmp*2+1];
       }
   CUDA_SAFE_CALL(cufftDestroy(plan));
   CUDA_SAFE_CALL(cudaFree(data));
   free(x_float);
}

extern "C"
void
gpufft3d__ (double* xd_re, double* xd_im, int* size_x, int* size_y, int* size_z, int* flagfft)
{
  gpufft3d_ (xd_re, xd_im, *size_x, *size_y, *size_z, *flagfft);
}
