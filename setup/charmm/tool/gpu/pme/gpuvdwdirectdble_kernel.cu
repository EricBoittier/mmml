#include <stdio.h>
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#define DS(i, j) Ds[i][j]
#define ES(i, j) Es[i][j]
#define FS(i, j) Fs[i][j]
texture<float4,1,cudaReadModeElementType> tex;
__constant__ float4 d_B[THD/4];
__shared__ vdwatm Bs[1][BVDWDIRC];

extern "C"
__global__ void gpuvdwdirectdble_kernel
(float4* C, vdwatm* A, vdwatm* Ai, int num, float xmax0, float xmax1, float xmax2, int nat,
 float al0, float al1, float al2 )
{
  int ty = threadIdx.x;
  int itmp;
  float4 Csub1, Csub2, coef;
  vdwatm xi;
  float dx[4], dn6, tmp0, tmp1, tmp2, tmp3;

  SET_FLOAT4_ZERO(Csub1)
  SET_FLOAT4_ZERO(Csub2)
  xi = Ai[BVDWDIRC*blockIdx.x+ty];
  xi.a *= nat;
  for (int j = 0; j < num/BVDWDIRC; j++){
    BS(0,ty) = A[j*BVDWDIRC+ty];
    __syncthreads();
    for (int i = 0; i < BVDWDIRC; i++){
      dx[0] = xi.x - BS(0,i).x;
      dx[1] = xi.y - BS(0,i).y;
      dx[2] = xi.z - BS(0,i).z;
      dx[0] -= rintf(dx[0] * al0) * xmax0;
      dx[1] -= rintf(dx[1] * al1) * xmax1;
      dx[2] -= rintf(dx[2] * al2) * xmax2;
      dx[3] = dx[0]*dx[0];
      dx[3] += dx[1]*dx[1];
      dx[3] += dx[2]*dx[2];
      if (dx[3] != 0.f){
      	dx[3] = 1.f / dx[3];
      }else{
      	dx[3] = 0.f;
      }
      itmp = xi.a + BS(0,i).a;
      coef = d_B[itmp];
      dn6 = dx[3]*dx[3]*dx[3];
      tmp0 = dn6 * (coef.x * dn6 - coef.y);
      ADD_DOUBLEFLOAT(tmp0,tmp1,tmp2,tmp3,Csub2.z,Csub2.w)
      //Csub.w += dn6 * (coef.x * dn6 - coef.y);
      dn6 = dx[3] * dn6 * (coef.z * dn6 - coef.w);
      tmp0 = dn6 * dx[0];
      ADD_DOUBLEFLOAT(tmp0,tmp1,tmp2,tmp3,Csub1.x,Csub1.y)
      tmp0 = dn6 * dx[1];
      ADD_DOUBLEFLOAT(tmp0,tmp1,tmp2,tmp3,Csub1.z,Csub1.w)
      tmp0 = dn6 * dx[2];
      ADD_DOUBLEFLOAT(tmp0,tmp1,tmp2,tmp3,Csub2.x,Csub2.y)
      //Csub.x += dn6 * dx[0];
      //Csub.y += dn6 * dx[1];
      //Csub.z += dn6 * dx[2];
    }
    __syncthreads();
  }
  C[BVDWDIRC*blockIdx.x*2+ty*2+0] = Csub1;
  C[BVDWDIRC*blockIdx.x*2+ty*2+1] = Csub2;
}

