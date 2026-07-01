#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_
#include <stdio.h>
#include "matrixMul.h"
#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) CUT_BANK_CHECKER(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) CUT_BANK_CHECKER(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#define DS(i, j) CUT_BANK_CHECKER(((float*)&Ds[0][0]), (BLOCK_SIZE * i + j))
#define ES(i, j) CUT_BANK_CHECKER(((float*)&Es[0][0]), (BLOCK_SIZE * i + j))
#define FS(i, j) CUT_BANK_CHECKER(((float*)&Fs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#define DS(i, j) Ds[i][j]
#define ES(i, j) Es[i][j]
#define FS(i, j) Fs[i][j]
#endif
__constant__ float d_B[THD];
texture<float,1,cudaReadModeElementType> tex;

extern "C"
__global__ void
gpuvdwpot256_kernel( float* C, float* A, int* D, int num, float xmax, int nat)
{
  int by = blockIdx.x;
  int ty = threadIdx.x;
  int aBegin = THD*by*3 + ty*3;
  int jEnd = num / THD;
  int itype, itmp;
  float al = 1.f / xmax;
  //float exclude_radius2 = 0.0001f;
  //float cutoff_radius2 = 10000.f;
  float dn2, dn6;
  float dx, dy, dz;
  float xi,yi,zi;
  int ai;
  float Csub = 0.f;

  xi = A[aBegin];
  yi = A[aBegin+1];
  zi = A[aBegin+2];
  ai = D[THD*by+ty]-1;
  for (int j = 0; j < jEnd; j++){
    itmp = j*THD*3 + ty*3;
    __shared__ float Bs[1][THD*3];
    BS(0,ty*3)   = A[itmp];
    BS(0,ty*3+1) = A[itmp+1];
    BS(0,ty*3+2) = A[itmp+2];
    itmp = j*THD+ty;
    __shared__ int Es[1][THD];
    ES(0,ty)   = D[itmp]  -1;
    __syncthreads();
    for (int i = 0; i < THD; i++){
      dx = xi - BS(0,i*3);
      dy = yi - BS(0,i*3+1);
      dz = zi - BS(0,i*3+2);
      dx = dx - rintf(dx*al)*xmax;
      dy = dy - rintf(dy*al)*xmax;
      dz = dz - rintf(dz*al)*xmax;
      itype = ai*nat+ES(0,i);
      dn2 = dx*dx+dy*dy+dz*dz;
      dn6 = d_B[itype*2+1];
      dn6 *= tex1D(tex,1.f/dn2);
      dn6 *= dn6 - 1.f;
      Csub += d_B[itype*2] * dn6;
    }
    __syncthreads();
  }
  C[aBegin] = Csub;
  //C[aBegin] = (float)(by*THD+ty+1);
  C[aBegin] = tex1D(tex,(float)(by*THD+ty+1)/(float)(num));
  //C[aBegin] = (float)(by*THD+ty+1)/(float)(num);
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
