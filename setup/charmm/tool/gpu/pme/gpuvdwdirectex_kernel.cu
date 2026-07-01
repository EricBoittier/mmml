#include <stdio.h>
#include "matrixMul.h"
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#define DS(i, j) Ds[i][j]
#define ES(i, j) Es[i][j]
#define FS(i, j) Fs[i][j]
texture<float4,1,cudaReadModeElementType> tex;
__constant__ float4 d_B[THD/4];
__shared__ vdwatmex Bs[1][BSIZEVDW];

extern "C"
__global__ void gpuvdwdirectex_kernel
(float4* C, vdwatmex* A, int num, float xmax0, float xmax1, float xmax2, short nat,
 float al0, float al1, float al2)
{
  int ty = threadIdx.x;
  short itmp;
  float4 Csub;
  vdwatmex xi;
  float dx[4], dn6;

  SET_FLOAT4_ZERO(Csub)
  xi = A[BSIZEVDW*blockIdx.x+ty];
  for (int j = 0; j < num/BSIZEVDW; j++){
    BS(0,ty) = A[j*BSIZEVDW+ty];
    __syncthreads();
    for (int i = 0; i < BSIZEVDW; i++){
      dx[0] = xi.x - BS(0,i).x;
      dx[1] = xi.y - BS(0,i).y;
      dx[2] = xi.z - BS(0,i).z;
      dx[0] -= rintf(dx[0] * al0) * xmax0;
      dx[1] -= rintf(dx[1] * al1) * xmax1;
      dx[2] -= rintf(dx[2] * al2) * xmax2;
      dx[3] = dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2];
      if (xi.m != BS(0,i).m){
	dx[3] = 1.f / dx[3];
      }else{
	dx[3] = 0.f;
      }
      itmp = nat*xi.a + BS(0,i).a;
      dn6 = dx[3]*dx[3]*dx[3];
      Csub.w += dn6 * (d_B[itmp].x * dn6 - d_B[itmp].y);
      dn6 = dx[3] * dn6 * (d_B[itmp].z * dn6 - d_B[itmp].w);
      Csub.x += dn6 * dx[0];
      Csub.y += dn6 * dx[1];
      Csub.z += dn6 * dx[2];
    }
    __syncthreads();
  }
  C[BSIZEVDW*blockIdx.x+ty] = Csub;
}

