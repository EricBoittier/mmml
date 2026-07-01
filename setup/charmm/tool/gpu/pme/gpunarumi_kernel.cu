#include <stdio.h>
#include "matrixMul.h"
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#define DS(i, j) Ds[i][j]
#define ES(i, j) Es[i][j]
#define FS(i, j) Fs[i][j]
struct __align__(16) vdwatm {float x;  float y;  float z;  int a;};

texture<float4,1,cudaReadModeElementType> tex;
__constant__ float4 d_B[THD/4];
__shared__ vdwatm Bs[1][BSIZEVDW];
//__shared__ float Bs[1][THD*4];

extern "C"
__global__ void gpuvdwdirect_kernel
(float4* C, vdwatm* A, vdwatm* Ai, int num, float xmax0, float xmax1, float xmax2, int nat)
{
  int ty = threadIdx.x;
  int itmp;
  float4 Csub, coef;
  //float4 dn4;
  vdwatm xi;
  float dx[4], al[3], dn6;
  float *BSp=(float *)Bs,*Ap=(float *)A;
  al[0]   = 1.f / xmax0;
  al[1]   = 1.f / xmax1;
  al[2]   = 1.f / xmax2;
  Csub.x = 0.f;
  Csub.y = 0.f;
  Csub.z = 0.f;
  Csub.w = 0.f;
  xi = Ai[BSIZEVDW*blockIdx.x+ty];
  //printf("dev %f",xi.x);
  for (int j = 0; j < num/BSIZEVDW; j++){
    BSp[ty*4  ] = Ap[(j*BSIZEVDW+ty)*4  ];
    BSp[ty*4+1] = Ap[(j*BSIZEVDW+ty)*4+1];
    BSp[ty*4+2] = Ap[(j*BSIZEVDW+ty)*4+2];
    BSp[ty*4+3] = Ap[(j*BSIZEVDW+ty)*4+3];
    __syncthreads();
    for (int i = 0; i < BSIZEVDW; i++){
      //      dx[0] = xi.x;
      //      dx[1] = xi.y;
      //      dx[2] = xi.z;
      dx[0] = xi.x - BS(0,i).x;
      dx[1] = xi.y - BS(0,i).y;
      dx[2] = xi.z - BS(0,i).z;
      //dx[0] -= rintf(dx[0] * al[0]) * xmax0;
      //dx[1] -= rintf(dx[1] * al[1]) * xmax1;
      //dx[2] -= rintf(dx[2] * al[2]) * xmax2;
      dx[3] = dx[0]*dx[0];
      dx[3] += dx[1]*dx[1];
      dx[3] += dx[2]*dx[2];
      dx[3] += 2.f;
      //if (dx[3] != 0){
      //	dx[3] = 1.f / dx[3];
      //	itmp = nat*xi.a + BS(0,i).a;
      //	coef = d_B[itmp];
      //	dn6 = dx[3]*dx[3]*dx[3];
      //	Csub.w += dn6 * (coef.x * dn6 - coef.y);
      //	dn6 = dx[3] * dn6 * (coef.z * dn6 - coef.w);
        dn6 = rsqrtf(dx[3]);
	dn6 *= dn6 * dn6;
	Csub.x += dn6 * dx[0];
	Csub.y += dn6 * dx[1];
	Csub.z += dn6 * dx[2];
	//}
    }
    __syncthreads();
  }
  C[BSIZEVDW*blockIdx.x+ty] = Csub;
}

/*
extern "C"
__global__ void gpuvdwdirect_kernel2
(float* C, float* A, int num, float xmax0, float xmax1, float xmax2, int nat,
 int num_max_exclude, int* exclude)
{
  int id = THD*blockIdx.x+threadIdx.x;
  int itmp;
  float4 dn4;
  float dx[4], xi[4], Csub[4], al[3], xj[4], dn6;
  al[0]   = 1.f / xmax0;
  al[1]   = 1.f / xmax1;
  al[2]   = 1.f / xmax2;
  Csub[0] = 0.f;
  Csub[1] = 0.f;
  Csub[2] = 0.f;
  Csub[3] = 0.f;
  xi[0] = A[id*4+0];
  xi[1] = A[id*4+1];
  xi[2] = A[id*4+2];
  xi[3] = __int_as_float((int)A[id*4+3]);
  for (int j = 0; j < num_max_exclude; j++){
    itmp = exclude[id*num_max_exclude+j];
    xj[0] = A[itmp*4+0];
    xj[1] = A[itmp*4+1];
    xj[2] = A[itmp*4+2];
    xj[3] = __int_as_float((int)A[itmp*4+3]);
    dx[0] = xi[0] - xj[0];
    dx[1] = xi[1] - xj[1];
    dx[2] = xi[2] - xj[2];
    dx[0] = dx[0] - rintf(dx[0] * al[0]) * xmax0;
    dx[1] = dx[1] - rintf(dx[1] * al[1]) * xmax1;
    dx[2] = dx[2] - rintf(dx[2] * al[2]) * xmax2;
    itmp  = nat*__float_as_int(xi[3]) + __float_as_int(xj[3]);
    dx[3] = (dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]);
    dn4 = tex1D(tex,rsqrt(dx[3]));
    dn6 = d_B[itmp*4+2] * dn4.z - d_B[itmp*4+3] * dn4.w;
    Csub[0] -= dn6 * dx[0];
    Csub[1] -= dn6 * dx[1];
    Csub[2] -= dn6 * dx[2];
    //Csub[3] -= dn6 * dx[3]; // atmic virial
    Csub[3] -= d_B[itmp*4+0] * dn4.x - d_B[itmp*4+1] * dn4.y;
  }
  C[id*4+0] += Csub[0];
  C[id*4+1] += Csub[1];
  C[id*4+2] += Csub[2];
  C[id*4+3] += Csub[3];
  __syncthreads();
}

*/
