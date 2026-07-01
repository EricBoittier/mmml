#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <gpuheader.h>
#include <gpuewwvpot256_kernel.cu>
////////////////////////////////////////////////////////////////////////////////
extern "C"
void gpuewwvpot256_  (double*, int, double*, int, double*);
void gpuewwvpot256__ (double*, int, double*, int, double*);
extern "C"
void gpuewwvpot256_ (double* xq, int num_atm, double* g_vec, int num_k, double* force)
{
   CUT_CHECK_DEVICE();

   int num_atm256 = ((int)((float)num_atm/(float)THD)+1)*THD;
   int num_k256   = ((int)((float)num_k  /(float)THD)+1)*THD;

   unsigned int size_A = num_atm256 * 4;
   unsigned int mem_size_A = sizeof(float) * size_A;
   float* xq_float = (float*) malloc(mem_size_A);
   unsigned int size_B = num_k256 * 4;
   unsigned int mem_size_B = sizeof(float) * size_B;
   float* gv_float = (float*) malloc(mem_size_B);
   unsigned int size_C = num_k256 * 3;
   unsigned int mem_size_C = sizeof(float) * size_C;
   float* f_float = (float*) malloc(mem_size_C);

   for (int i = 0; i < num_atm ; i++){
     xq_float[i*4]   = (float)xq[i*4];
     xq_float[i*4+1] = (float)xq[i*4+1];
     xq_float[i*4+2] = (float)xq[i*4+2];
     xq_float[i*4+3] = (float)xq[i*4+3];
   }
   for (int i = num_atm; i < num_atm256 ; i++){
     xq_float[i*4]   = 0.f;
     xq_float[i*4+1] = 0.f;
     xq_float[i*4+2] = 0.f;
     xq_float[i*4+3] = 0.f;
   }
   for (int i = 0; i < num_k; i++){
     gv_float[i*4]   = (float)g_vec[i*4];
     gv_float[i*4+1] = (float)g_vec[i*4+1];
     gv_float[i*4+2] = (float)g_vec[i*4+2];
     gv_float[i*4+3] = (float)g_vec[i*4+3];
   }
   for (int i = num_k; i < num_k256; i++){
     gv_float[i*4]   = 0.f;
     gv_float[i*4+1] = 0.f;
     gv_float[i*4+2] = 0.f;
     gv_float[i*4+3] = 0.f;
   }

   float* d_A;
   CUDA_SAFE_CALL(cudaMalloc((void**) &d_A, mem_size_A));
   float* d_B;
   CUDA_SAFE_CALL(cudaMalloc((void**) &d_B, mem_size_B));
   float* d_C;
   CUDA_SAFE_CALL(cudaMalloc((void**) &d_C, mem_size_C));

   CUDA_SAFE_CALL(cudaMemcpy(d_A, xq_float, mem_size_A,cudaMemcpyHostToDevice) );
   CUDA_SAFE_CALL(cudaMemcpy(d_B, gv_float, mem_size_B,cudaMemcpyHostToDevice) );

   dim3 threads(THD);
   dim3 grid(num_k256 / THD);
   gpuewwvpot256_kernel<<< grid, threads >>>(d_C, d_A, d_B, num_atm256);
   CUT_CHECK_ERROR("Kernel execution failed");

   CUDA_SAFE_CALL(cudaMemcpy(f_float, d_C, mem_size_C,cudaMemcpyDeviceToHost) );

   /*
   unsigned int size_E = num_atm * 3;
   unsigned int mem_size_E = size_E * sizeof(double);
   double* force_double = (double*) malloc(mem_size_E);
   for (int i = 0; i < num_atm; ++i){
     force_double[i*3]   = 0.e0;
     force_double[i*3+1] = 0.e0;
     force_double[i*3+2] = 0.e0;
     }
   computeGold_d2(force_double, g_vec, xq, num_atm, num_k);
   */
   for (int i = 0; i < num_k; ++i){
     //printf("%16.6f %16.6f %d \n",f_float[i*3+2],force_double[i*3+2],i);
     force[i*3] = (double)f_float[i*3+2];
   }

   free(xq_float);
   free(gv_float);
   free(f_float);
   //free(force_double);
   CUDA_SAFE_CALL(cudaFree(d_A));
   CUDA_SAFE_CALL(cudaFree(d_B));
   CUDA_SAFE_CALL(cudaFree(d_C));
}
extern "C"
void
gpuewwvpot256__ (double* xq, int* num_atm, double* g_vec, int* num_k, double* force)
{
  gpuewwvpot256_ (xq,*num_atm,g_vec,*num_k,force);
}

extern "C"
void
computeGold_d2(double* C, const double* A, const double* B, int num_atm, int num_k)
{
  double kr;
  double qsin = 0.e0;
  double qcos = 0.e0;

  unsigned int size_sc = num_atm;
  unsigned int mem_size_sc = sizeof(double) * size_sc;
  double* sin_theta = (double*) malloc(mem_size_sc);
  double* cos_theta = (double*) malloc(mem_size_sc);
  

  /*  for (unsigned int i = 0; i < num_atm*3; ++i)
    printf("%d %f\n",i,C[i]);
  for (unsigned int i = 0; i < num_atm*4; ++i)
  printf("%d %f\n",i,B[i]);*/

  double sum_pot;
  for (unsigned int i = 0; i < num_k; ++i){
    qsin = 0.e0;
    qcos = 0.e0;
    for (unsigned int j = 0; j < num_atm; ++j){
      kr = A[i*4]   * B[j*4]
	+  A[i*4+1] * B[j*4+1]
	+  A[i*4+2] * B[j*4+2];
      //theta = kr * pi2;
      sin_theta[j] = sin(kr);
      cos_theta[j] = cos(kr);
      qsin += B[j*4+3] * sin_theta[j];
      qcos += B[j*4+3] * cos_theta[j];
    }
    C[i*3]   = qsin;
    C[i*3+1] = qcos;
    C[i*3+2] = A[i*4+3] * ( qsin * qsin + qcos * qcos );
    sum_pot += C[i*3+2];
  }
  //printf("%f\n",sum_pot);

  free(sin_theta);
  free(cos_theta);

}
