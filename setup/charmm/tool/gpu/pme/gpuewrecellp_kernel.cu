#include <stdio.h>
#define BS(i, j) Bs[i][j]
#define DS(i, j) Ds[i][j]
#define FORCELOOP(iatm)				\
  dx[0] = xi.x - BS(0,iatm).x;			\
  dx[1] = xi.y - BS(0,iatm).y;			\
  dx[2] = xi.z - BS(0,iatm).z;			\
  GET_SQUARE_AND_R2(dx)				\
  if (dx[3] < radius2){				\
    dn2 = tex1D(tex,sqrt(dx[3])*rcut);		\
    dx[3] = BS(0,iatm).w * dn2.x;		\
    Csub.x += dx[3] * dx[0];			\
    Csub.y += dx[3] * dx[1];			\
    Csub.z += dx[3] * dx[2];			\
  }

texture<float2,1,cudaReadModeElementType> tex;
__shared__ float4 Bs[1][BEWRCELL];
__shared__ int Ds[1][BEWRCELL];

extern "C"
__global__ void gpuewrecellp_kernel
(int* D, float4* C, float4* A, float4* E, int num_round_cell, int num_per_cell,
 float radius2, float rcut)
{
  float4 Csub;
  float dx[4];
  float2 dn2;
  float4 xi;

  SET_FLOAT4_ZERO(Csub)
  DS(0,threadIdx.x) = D[BEWRCELL*blockIdx.x+threadIdx.x]; // jcelllist
  xi = A[BEWRCELL*blockIdx.x+threadIdx.x]; // x_float
  for (int j = 0; j < num_round_cell*num_per_cell; j++){
    BS(0,threadIdx.x) = E[DS(0,j)+threadIdx.x]; // x_float_psd:
    __syncthreads();
    for (int i = 0; i < BEWRCELL/8; i++){
      FORCELOOP(i*8+0)
      FORCELOOP(i*8+1)
      FORCELOOP(i*8+2)
      FORCELOOP(i*8+3)
      FORCELOOP(i*8+4)
      FORCELOOP(i*8+5)
      FORCELOOP(i*8+6)
      FORCELOOP(i*8+7)
    }
    __syncthreads();
  }
  Csub.x *= xi.w;
  Csub.y *= xi.w;
  Csub.z *= xi.w;
  Csub.w *= xi.w;
  C[BEWRCELL*blockIdx.x+threadIdx.x] = Csub;
}
