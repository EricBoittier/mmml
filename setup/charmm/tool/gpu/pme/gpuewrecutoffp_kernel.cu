#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_
#include <stdio.h>
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
texture<float2,1,cudaReadModeElementType> tex;
__shared__ float4 Bs[1][BEWRDIRC];
extern "C"
__global__ void gpuewrecutoffp_kernel
(float4* C, float4* A, float4* Ai, int num, float xmax0, float xmax1, float xmax2,
 float al0, float al1, float al2, float cut2, float rcut)
{
  int ty = threadIdx.x;
  float2 dn2;
  float dx[4];
  float4 xi, Csub;
  SET_FLOAT4_ZERO(Csub)
  xi = Ai[BEWRDIRC*blockIdx.x+ty];
  for (int j = 0; j < num/BEWRDIRC; j++){
    BS(0,ty) = A[j*BEWRDIRC+ty];
    __syncthreads();
    for (int i = 0; i < BEWRDIRC; i++){
      dx[0] = xi.x - BS(0,i).x;
      dx[1] = xi.y - BS(0,i).y;
      dx[2] = xi.z - BS(0,i).z;
      dx[0] -= rintf(dx[0] * al0) * xmax0;
      dx[1] -= rintf(dx[1] * al1) * xmax1;
      dx[2] -= rintf(dx[2] * al2) * xmax2;
      GET_SQUARE_AND_R2(dx)
      if (dx[3] < cut2){
	dn2 = tex1D(tex,sqrt(dx[3])*rcut);
	dx[3] = BS(0,i).w * dn2.x;
	Csub.x += dx[3] * dx[0];
	Csub.y += dx[3] * dx[1];
	Csub.z += dx[3] * dx[2];
	Csub.w += BS(0,i).w * dn2.y;
      }
    }
    __syncthreads();
  }
  Csub.x *= xi.w;
  Csub.y *= xi.w;
  Csub.z *= xi.w;
  Csub.w *= xi.w;
  C[BEWRDIRC*blockIdx.x+ty] = Csub;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
