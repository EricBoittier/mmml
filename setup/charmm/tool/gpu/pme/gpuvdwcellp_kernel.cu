#include <stdio.h>
#define BS(i, j) Bs[i][j]
#define DS(i, j) Ds[i][j]
__constant__ float4 d_B[THD/4];
__shared__ vdwatm Bs[1][BVDWCELL];
__shared__ int Ds[1][BVDWCELL];

extern "C"
__global__ void gpuvdwcellp_kernel
(int* D, float4* C, vdwatm* A, vdwatm* E, int num_round_cell, int num_per_cell, float radius2)
{
  int itmp;
  float4 Csub;
  float dx[4], dn6;
  vdwatm xi;

  SET_FLOAT4_ZERO(Csub)
    //DS(0,threadIdx.x*2+0) = D[BVDWCELL*2*blockIdx.x+2*threadIdx.x+0]; // jcelllist
  //DS(0,threadIdx.x*2+1) = D[BVDWCELL*2*blockIdx.x+2*threadIdx.x+1]; // jcelllist
  DS(0,threadIdx.x) = D[BVDWCELL*blockIdx.x+threadIdx.x]; // jcelllist
  xi = A[BVDWCELL*blockIdx.x+threadIdx.x]; // x_float
  for (int j = 0; j < num_per_cell; j++){
    BS(0,threadIdx.x) = E[DS(0,j)+threadIdx.x]; // x_float_psd:
    __syncthreads();
    for (int i = 0; i < BVDWCELL; i++){
      dx[0] = xi.x - BS(0,i).x;
      dx[1] = xi.y - BS(0,i).y;
      dx[2] = xi.z - BS(0,i).z;
      GET_SQUARE_AND_R2(dx)
      if (dx[3] < radius2){
	if (dx[3] != 0){
	  dx[3] = 1.f / dx[3];
	}else{
	  dx[3] = 0.f;
	}
	itmp = xi.a + BS(0,i).a;
	dn6 = dx[3]*dx[3]*dx[3];
	//Csub.w += dn6 * (d_B[itmp].x * dn6 - d_B[itmp].y);
	dn6 *= dx[3] * (d_B[itmp].z * dn6 - d_B[itmp].w);
	Csub.x += dn6 * dx[0];
	Csub.y += dn6 * dx[1];
	Csub.z += dn6 * dx[2];
      }
    }
    __syncthreads();
  }
  for (int j = num_per_cell; j < num_round_cell*num_per_cell; j++){
    BS(0,threadIdx.x) = E[DS(0,j)+threadIdx.x]; // x_float_psd:
    __syncthreads();
    for (int i = 0; i < BVDWCELL; i++){
      dx[0] = xi.x - BS(0,i).x;
      dx[1] = xi.y - BS(0,i).y;
      dx[2] = xi.z - BS(0,i).z;
      GET_SQUARE_AND_R2(dx)
      if (dx[3] < radius2){
	dx[3] = 1.f / dx[3];
	itmp = xi.a + BS(0,i).a;
	dn6 = dx[3]*dx[3]*dx[3];
	//Csub.w += dn6 * (d_B[itmp].x * dn6 - d_B[itmp].y);
	dn6 *= dx[3] * (d_B[itmp].z * dn6 - d_B[itmp].w);
	Csub.x += dn6 * dx[0];
	Csub.y += dn6 * dx[1];
	Csub.z += dn6 * dx[2];
      }
    }
    __syncthreads();
  }
  C[BVDWCELL*blockIdx.x+threadIdx.x] = Csub;
}
/*
extern "C"
__global__ void gpuvdwcell_kernel
(int* D, float4* C, vdwatm* A, vdwatm* E, int num_round_cell, int num_per_cell, float radius2)
{
  int ty = threadIdx.x;
  int itmp;
  float4 Csub;
  float dx[4], dn6;
  vdwatm xi;

  SET_FLOAT4_ZERO(Csub)
  DS(0,ty) = D[BVDWCELL*blockIdx.x+ty]; // jcelllist
  xi = A[BVDWCELL*blockIdx.x+ty]; // x_float
  for (int j = 0; j < num_per_cell; j++){
    BS(0,ty) = E[DS(0,j)+ty]; // x_float_psd:
    __syncthreads();
    for (int i = 0; i < BVDWCELL; i++){
      dx[0] = xi.x - BS(0,i).x;
      dx[1] = xi.y - BS(0,i).y;
      dx[2] = xi.z - BS(0,i).z;
      dx[3] = dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2];
      if (dx[3] != 0){
	dx[3] = 1.f / dx[3];
      }else{
	dx[3] = 0.f;
      }
      itmp = xi.a + BS(0,i).a;
      dn6 = dx[3]*dx[3]*dx[3];
      Csub.w += dn6 * (d_B[itmp].x * dn6 - d_B[itmp].y);
      dn6 *= dx[3] * (d_B[itmp].z * dn6 - d_B[itmp].w);
      Csub.x += dn6 * dx[0];
      Csub.y += dn6 * dx[1];
      Csub.z += dn6 * dx[2];
    }
    __syncthreads();
  }
  for (int j = num_per_cell; j < num_round_cell*num_per_cell; j++){
    BS(0,ty) = E[DS(0,j)+ty]; // x_float_psd:
    __syncthreads();
    for (int i = 0; i < BVDWCELL; i++){
      dx[0] = xi.x - BS(0,i).x;
      dx[1] = xi.y - BS(0,i).y;
      dx[2] = xi.z - BS(0,i).z;
      dx[3] = dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2];
      if (dx[3] < radius2){
	dx[3] = 1.f / dx[3];
	itmp = xi.a + BS(0,i).a;
	dn6 = dx[3]*dx[3]*dx[3];
	Csub.w += dn6 * (d_B[itmp].x * dn6 - d_B[itmp].y);
	dn6 *= dx[3] * (d_B[itmp].z * dn6 - d_B[itmp].w);
	Csub.x += dn6 * dx[0];
	Csub.y += dn6 * dx[1];
	Csub.z += dn6 * dx[2];
      }
    }
    __syncthreads();
  }
  C[BVDWCELL*blockIdx.x+ty] = Csub;
}*/
  /*
  for (int j = 0; j < num_round_cell*num_per_cell; j++){
    BS(0,ty) = E[DS(0,j)+ty]; // x_float_psd:
    __syncthreads();
    for (int i = 0; i < BVDWCELL; i++){
      dx[0] = xi.x - BS(0,i).x;
      dx[1] = xi.y - BS(0,i).y;
      dx[2] = xi.z - BS(0,i).z;
      dx[3] = dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2];
      if (dx[3] < radius2){
	if (dx[3] != 0){
	  dx[3] = 1.f / dx[3];
	}else{
	  dx[3] = 0.f;
	}
	itmp = xi.a + BS(0,i).a;
	dn6 = dx[3]*dx[3]*dx[3];
	Csub.w += dn6 * (d_B[itmp].x * dn6 - d_B[itmp].y);
	dn6 = dx[3] * dn6 * (d_B[itmp].z * dn6 - d_B[itmp].w);
	Csub.x += dn6 * dx[0];
	Csub.y += dn6 * dx[1];
	Csub.z += dn6 * dx[2];
      }
    }
    __syncthreads();
  }
  */
