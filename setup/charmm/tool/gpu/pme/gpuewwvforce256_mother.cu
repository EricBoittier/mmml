#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include <gpuheader.h>

#include <gpuewwvforce256_kernel.cu>
////////////////////////////////////////////////////////////////////////////////
extern "C"
void gpuewwvforce256_  (double*, int, double*, int, double*);
void gpuewwvforce256__ (double*, int, double*, int, double*);
//extern "C"
//void computeGold_d(double*, const double*, const double*, int , int);
extern "C"
void gpuewwvforce256_ (double* xq, int num_atm, double* g_vec, int num_k, double* force)
{
   CUT_CHECK_DEVICE();

   int num_atm256 = ((int)((float)(num_atm-1)/(float)THD)+1)*THD;
   int num_k256   = ((int)((float)(num_k-1)  /(float)THD)+1)*THD;

   unsigned int size_A = num_atm256 * 4;
   unsigned int mem_size_A = sizeof(float) * size_A;
   float* xq_float = (float*) malloc(mem_size_A);
   unsigned int size_B = num_k256 * 4;
   unsigned int mem_size_B = sizeof(float) * size_B;
   float* gv_float = (float*) malloc(mem_size_B);
   unsigned int size_C = num_k256 * 3;
   unsigned int mem_size_C = sizeof(float) * size_C;
   float* pot_float = (float*) malloc(mem_size_C);
   unsigned int size_D = num_atm256 * 3;
   unsigned int mem_size_D = sizeof(float) * size_D;
   float* f_float = (float*) malloc(mem_size_D);

   for (int i = 0; i < num_atm ; i++){
     xq_float[i*4]   = (float)xq[i*4];
     xq_float[i*4+1] = (float)xq[i*4+1];
     xq_float[i*4+2] = (float)xq[i*4+2];
     xq_float[i*4+3] = (float)xq[i*4+3];
   }
   for (int i = num_atm; i < num_atm256 ; i++){
     xq_float[i*4]   = 0.e0;
     xq_float[i*4+1] = 0.e0;
     xq_float[i*4+2] = 0.e0;
     xq_float[i*4+3] = 0.e0;
   }
   for (int i = 0; i < num_k; i++){
     gv_float[i*4]   = (float)g_vec[i*4];
     gv_float[i*4+1] = (float)g_vec[i*4+1];
     gv_float[i*4+2] = (float)g_vec[i*4+2];
     gv_float[i*4+3] = (float)g_vec[i*4+3];
   }
   for (int i = num_k; i < num_k256; i++){
     gv_float[i*4]   = 0.e0;
     gv_float[i*4+1] = 0.e0;
     gv_float[i*4+2] = 0.e0;
     gv_float[i*4+3] = 0.e0;
   }

   float* d_A;
   CUDA_SAFE_CALL(cudaMalloc((void**) &d_A, mem_size_A));
   float* d_B;
   CUDA_SAFE_CALL(cudaMalloc((void**) &d_B, mem_size_B));
   float* d_C;
   CUDA_SAFE_CALL(cudaMalloc((void**) &d_C, mem_size_C));
   float* d_D;
   CUDA_SAFE_CALL(cudaMalloc((void**) &d_D, mem_size_D));

   CUDA_SAFE_CALL(cudaMemcpy(d_A, xq_float, mem_size_A,cudaMemcpyHostToDevice) );
   CUDA_SAFE_CALL(cudaMemcpy(d_B, gv_float, mem_size_B,cudaMemcpyHostToDevice) );

   dim3 threads(THD);
   dim3 grid(num_k256 / THD);
   gpuewwvforce256_kernel<<< grid, threads >>>(d_C, d_A, d_B, num_atm);
   dim3 grid2(num_atm256 / THD);
   gpuewwvforce256_kernel2<<< grid2, threads >>>(d_C, d_B, d_A, d_D, num_k);
   CUT_CHECK_ERROR("Kernel execution failed");

   CUDA_SAFE_CALL(cudaMemcpy(f_float, d_D, mem_size_D,cudaMemcpyDeviceToHost) );

   //computeGold_d(force_double, g_vec, xq, num_atm, num_k);
   for (int i = 0; i < num_atm; ++i){
     //printf("%16.6f %16.6f %d \n",f_float[i*3],force_double[i*3],i);
     force[i*3]   = (double)f_float[i*3];
     force[i*3+1] = (double)f_float[i*3+1];
     force[i*3+2] = (double)f_float[i*3+2];
     //force[i*3]   = force_double[i*3];
     //force[i*3+1] = force_double[i*3+1];
     //force[i*3+2] = force_double[i*3+2];
   }
   //printf("GPU : %20.8f \n",force[0]);
   //printf("HOST: %20.8f \n",force_double[0]);

   CUDA_SAFE_CALL(cudaFree(d_A));
   CUDA_SAFE_CALL(cudaFree(d_B));
   CUDA_SAFE_CALL(cudaFree(d_C));
   CUDA_SAFE_CALL(cudaFree(d_D));

   free(xq_float);
   free(gv_float);
   free(f_float);
   free(pot_float);
   //free(force_double);
}

extern "C"
void
gpuewwvforce256__ (double* xq, int* num_atm, double* g_vec, int* num_k, double* force)
{
  gpuewwvforce256_ (xq,*num_atm,g_vec,*num_k,force);
}

extern "C"
void
computeGold_d(double* C, const double* A, const double* B, int num_atm, int num_k)
{
  //A : g_vec
  //B : xq

  double kr, qsin, qcos, tmp;

  unsigned int size_sc = num_atm;
  unsigned int mem_size_sc = sizeof(double) * size_sc;
  double* sin_theta = (double*) malloc(mem_size_sc);
  double* cos_theta = (double*) malloc(mem_size_sc);

  for (unsigned int i = 0; i < num_atm; ++i){
    C[i*3]   = 0.e0;
    C[i*3+1] = 0.e0;
    C[i*3+2] = 0.e0;
  }

  for (unsigned int i = 0; i < num_k; ++i){
    qsin = 0.e0;
    qcos = 0.e0;
    for (unsigned int j = 0; j < num_atm; ++j){
      kr = A[i*4]   * B[j*4]
	 + A[i*4+1] * B[j*4+1]
	 + A[i*4+2] * B[j*4+2];
      sin_theta[j] = sin(kr);
      cos_theta[j] = cos(kr);
      qsin += B[j*4+3] * sin_theta[j];
      qcos += B[j*4+3] * cos_theta[j];
    }
    //C[i*3]   = qsin;
    //C[i*3+1] = qcos;
    //C[i*3+2] = A[i*4+3] * ( qsin * qsin + qcos * qcos );
    qsin *= A[i*4+3];
    qcos *= A[i*4+3];
    for (unsigned int j = 0; j < num_atm; ++j){
      tmp = qcos * sin_theta[j] - qsin * cos_theta[j];
      C[j*3]   += tmp * A[i*4];
      C[j*3+1] += tmp * A[i*4+1];
      C[j*3+2] += tmp * A[i*4+2];
    }
  }

  for (unsigned int i = 0; i < num_atm; ++i){
    C[i*3]   *= B[i*4+3];
    C[i*3+1] *= B[i*4+3];
    C[i*3+2] *= B[i*4+3];
  }

  free(sin_theta);
  free(cos_theta);

}
