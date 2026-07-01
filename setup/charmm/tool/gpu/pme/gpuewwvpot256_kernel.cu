#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_
#include <stdio.h>
#include "matrixMul.h"
#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) CUT_BANK_CHECKER(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) CUT_BANK_CHECKER(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif

extern "C"
__global__ void
gpuewwvpot256_kernel( float* C, float* B, float* A, int num_atm)
{
  // A : k vertor
  // B : x and q

  int by = blockIdx.x;
  int ty = threadIdx.x;
  int jty4, bty3;
  int bty4 = THD * by * 4 + ty*4;
  int jEnd = num_atm / THD;
  float kr;
  float sin_theta, cos_theta;
  float qsin = 0.f;
  float qcos = 0.f;
  float xi[4];

  xi[0] = A[bty4];
  xi[1] = A[bty4+1];
  xi[2] = A[bty4+2];
  xi[3] = A[bty4+3];
  for (int j = 0; j < jEnd; j++){
    jty4 = j*THD*4 + ty*4;
    __shared__ float Bs[1][THD*4];
    BS(0,ty*4)   = B[jty4];
    BS(0,ty*4+1) = B[jty4+1];
    BS(0,ty*4+2) = B[jty4+2];
    BS(0,ty*4+3) = B[jty4+3];
    __syncthreads();
    for (int i = 0; i < THD; i++){
      kr = xi[0] * BS(0,i*4) 
	 + xi[1] * BS(0,i*4+1) 
	 + xi[2] * BS(0,i*4+2);
      sin_theta = sin(kr);
      cos_theta = cos(kr);
      qsin += sin_theta * BS(0,i*4+3);
      qcos += cos_theta * BS(0,i*4+3);
    }
    __syncthreads();
  }
  bty3 = THD*by*3 + ty*3;
  C[bty3]   = qsin;
  C[bty3+1] = qcos;
  //tmp2 = tex1D(tex,(float)(by*THD+ty)/(10000));
  //C[bty3]   = tmp2.x;
  //C[bty3+1] = tmp2.y;
  C[bty3+2] = xi[3] * (qsin*qsin + qcos*qcos);
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
