#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#define BSIZEVDW 256
#include <gpuheader.h>
#include <gpuvdwdirectex_kernel.cu>
////////////////////////////////////////////////////////////////////////////////

extern "C"
void gpuvdwdirectex_ 
(double*, int, int*, int, double*, double*, int, double*, int, int, double*, double*, int, int*);
void gpuvdwdirectex__
(double*, int, int*, int, double*, double*, int, double*, int, int, double*, double*, int, int*);

extern "C"
void gpuvdwdirectex_
(double* x, int num_atmall, int* atype, int nat, double* epsilon4, double* sigma2, int tblno,
 double* xmax, int periodicflag, int natchangeflag, double* force, double* pot,
 int idevice, int* moleid)
{
   CUT_CHECK_DEVICE();
   CUDA_SAFE_CALL(cudaSetDevice(idevice));
   int tex_l = 512;
   int num_atm;
   int num_site;
   int itmp, itmp1;
   //if (atype[3] == 1 && atype[4] != 1) num_site = 3;
   //if (atype[3] != 1 && atype[4] == 1) num_site = 4;
   //if (atype[3] == 1 && atype[4] == 1) num_site = 1;
   num_site = 1;
   num_atm = num_atmall / num_site;
   int n = ((int)((float)(num_atm-1)/(float)BSIZEVDW)+1)*BSIZEVDW;
   float xmax_float0 = (float)xmax[0];
   float xmax_float1 = (float)xmax[1];
   float xmax_float2 = (float)xmax[2];
   float rcp_xmax0 = (float)(1.e0/xmax[0]);
   float rcp_xmax1 = (float)(1.e0/xmax[1]);
   float rcp_xmax2 = (float)(1.e0/xmax[2]);
   int thd2 = (int)sqrt((float)(THD/4));
   if (thd2 < nat)
     printf("WARNING:gpuvdwforce256 TOO MUCH NUMBER OF ATM TYPE");
   unsigned int memsize_x_float = sizeof(vdwatmex) * n;
   vdwatmex* x_float = (vdwatmex*) malloc(memsize_x_float);
   unsigned int memsize_gr_float = sizeof(float4) * THD/4;
   float4* gr_float = (float4*) malloc(memsize_gr_float);
   unsigned int memsize_f_float = sizeof(float4) * n;
   float4* f_float = (float4*) malloc(memsize_f_float);
   unsigned int memsize_tex = sizeof(float4) * tex_l;
   float4* tex_data = (float4*) malloc(memsize_tex);
   
   for (int i = 0; i < num_atm; i++){
     COPY_DOUBLE_TO_VDWATMEX(x_float,x,atype,moleid,i,i*num_site)
   }
   for (int i = num_atm; i < n; i++){
     SET_DUMMY_VDWATMEX(x_float[i],nat)
   }
   for (int i = 0; i < THD/4; i++){
     SET_FLOAT4_ZERO(gr_float[i])
   }
   for (int i = 0; i < nat; i++)
     for (int j = 0; j < nat; j++){
       itmp = i*(nat+1)+j;
       itmp1 = i*(nat)+j;
       COPY_DOUBLE_TO_VDWCOEF(gr_float,epsilon4,sigma2,itmp,itmp1)
     }
   
   CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_B, gr_float , memsize_gr_float) );
   vdwatmex* x_dev;
   CUDA_SAFE_CALL(cudaMalloc((void**)&x_dev, memsize_x_float));
   CUDA_SAFE_CALL(cudaMemcpy(x_dev, x_float  , memsize_x_float, cudaMemcpyHostToDevice) );
   float4* f_dev;
   CUDA_SAFE_CALL(cudaMalloc((void**)&f_dev, memsize_f_float));
   dim3 threads(BSIZEVDW);
   dim3 grid(n / BSIZEVDW);
   short nats = (short)nat+1;
   gpuvdwdirectex_kernel<<< grid, threads >>>
     (f_dev, x_dev, n, xmax_float0, xmax_float1, xmax_float2, nats, rcp_xmax0, rcp_xmax1, rcp_xmax2);
   CUT_CHECK_ERROR("Kernel execution failed");
   CUDA_SAFE_CALL(cudaMemcpy(f_float, f_dev, memsize_f_float,cudaMemcpyDeviceToHost) );
   for (int i = 0; i < num_atm; ++i){
     force[i*num_site*3+0] += (double)f_float[i].x;
     force[i*num_site*3+1] += (double)f_float[i].y;
     force[i*num_site*3+2] += (double)f_float[i].z;
     pot[i*num_site] += (double)f_float[i].w;
   }
   
   free(x_float);
   free(f_float);
   free(gr_float);
   CUDA_SAFE_CALL(cudaFree(x_dev));
   CUDA_SAFE_CALL(cudaFree(f_dev));
}

extern "C"
void
gpuvdwdirectex__
(double* x, int* n, int* atype, int* nat, double* epsilon4, double* sigma2, int* tblno,
 double* xmax, int* periodicflag, int* natchangeflag, double* force, double* pot,
 int* idevice, int* moleid)
//(double* x, int* n, int* atype, int* nat, double* epsilon4, double* sigma2, int* tblno,
// double* xmax, int* periodicflag, int* natchangeflag, double* force, double* pot,
// int* numex, int* natex)
{
  gpuvdwdirectex_
    (x,*n,atype,*nat,epsilon4,sigma2,*tblno,xmax,*periodicflag,*natchangeflag,force,pot,
     *idevice,moleid);
  //(x,*n,atype,*nat,epsilon4,sigma2,*tblno,xmax,*periodicflag,*natchangeflag,force,pot,
  //   numex,natex);
}
