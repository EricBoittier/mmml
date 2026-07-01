#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <gpuheader.h>
#include <gpuvdwdirectp_kernel.cu>
////////////////////////////////////////////////////////////////////////////////

extern "C"
void gpuvdwdirectp_ 
(double*, int, int*, int, double*, double*, int, double*, int, int, double*, double*, int, int, int);
void gpuvdwdirectp__
(double*, int, int*, int, double*, double*, int, double*, int, int, double*, double*, int, int, int);

extern "C"
void gpuvdwdirectp_
(double* x, int num_atmall, int* atype, int nat, double* epsilon4, double* sigma2, int tblno,
 double* xmax, int periodicflag, int natchangeflag, double* force, double* pot, int idevice,
 int begin_numi, int num_atmi)
{
  CUDA_SAFE_CALL(cudaSetDevice(idevice));
  int itmp, itmp1;
   int tex_l = 512;
   int num_atm;
   int num_site;
   num_site = 1;
   num_atm = num_atmall / num_site;
   int num_atm_i = num_atmi;
   int iatm_begin = begin_numi - 1;
   int iatm_end = iatm_begin + num_atm_i - 1;
   if (num_site != 1){
     if (iatm_begin != 0)
       iatm_begin = 1+(int)((float)(iatm_begin-1)/(float)num_site);
     iatm_end = (int)((float)(iatm_end)/(float)num_site);
     num_atm_i = iatm_end - iatm_begin + 1;
   }
   int n  = ((int)((float)(num_atm-1)/(float)BVDWDIRC)+1)*BVDWDIRC;
   int ni = ((int)((float)(num_atm_i-1)/(float)BVDWDIRC)+1)*BVDWDIRC;
   float xmax_float0 = (float)xmax[0];
   float xmax_float1 = (float)xmax[1];
   float xmax_float2 = (float)xmax[2];
   float rcp_xmax0 = (float)(1.e0/xmax[0]);
   float rcp_xmax1 = (float)(1.e0/xmax[1]);
   float rcp_xmax2 = (float)(1.e0/xmax[2]);
   int thd2 = (int)sqrt((float)(THD/4));
   if (thd2 < nat)
     printf("WARNING:gpuvdwforce256 TOO MUCH NUMBER OF ATM TYPE");
   unsigned int memsize_x_float = sizeof(vdwatm) * n;
   vdwatm* x_float = (vdwatm*) malloc(memsize_x_float);
   unsigned int memsize_xi_float = sizeof(vdwatm) * ni;
   vdwatm* xi_float = (vdwatm*) malloc(memsize_xi_float);
   unsigned int memsize_gr_float = sizeof(float4) * THD/4;
   float4* gr_float = (float4*) malloc(memsize_gr_float);
   unsigned int memsize_f_float = sizeof(float4) * ni;
   float4* f_float = (float4*) malloc(memsize_f_float);
   unsigned int memsize_tex = sizeof(float4) * tex_l;
   float4* tex_data = (float4*) malloc(memsize_tex);
   
   for (int i = 0; i < num_atm; i++){
     COPY_DOUBLE_TO_VDWATM(x_float,x,atype,i,i*num_site)
   }
   for (int i = num_atm  ; i < n        ; i++){
     SET_DUMMY_VDWATM(x_float[i],nat)
   }
   for (int i = 0        ; i < num_atm_i; i++){
     xi_float[i] = x_float[i+iatm_begin];
   }
   for (int i = num_atm_i; i < ni       ; i++){
     SET_DUMMY_VDWATM(xi_float[i],nat)
   }
   for (int i = 0        ; i < THD/4    ; i++){
     SET_FLOAT4_ZERO(gr_float[i])
   }
   for (int i = 0; i < nat; i++)
     for (int j = 0; j < nat; j++){
       itmp = i*(nat+1)+j;
       itmp1 = i*(nat)+j;
       COPY_DOUBLE_TO_VDWCOEF(gr_float,epsilon4,sigma2,itmp,itmp1)
     }
   
   CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_B, gr_float , memsize_gr_float) );
   vdwatm* x_dev;
   CUDA_SAFE_CALL(cudaMalloc((void**)&x_dev, memsize_x_float));
   CUDA_SAFE_CALL(cudaMemcpy(x_dev, x_float  , memsize_x_float, cudaMemcpyHostToDevice) );
   vdwatm* xi_dev;
   CUDA_SAFE_CALL(cudaMalloc((void**)&xi_dev, memsize_xi_float));
   CUDA_SAFE_CALL(cudaMemcpy(xi_dev, xi_float  , memsize_xi_float, cudaMemcpyHostToDevice) );
   float4* f_dev;
   CUDA_SAFE_CALL(cudaMalloc((void**)&f_dev, memsize_f_float));
   dim3 threads(BVDWDIRC);
   dim3 grid(ni / BVDWDIRC);
   int nat2 = nat + 1;
   gpuvdwdirectp_kernel<<< grid, threads >>>
     (f_dev, x_dev, xi_dev, n, xmax_float0, xmax_float1, xmax_float2, nat2,rcp_xmax0,rcp_xmax1,rcp_xmax2);
   CUT_CHECK_ERROR("Kernel execution failed");
   CUDA_SAFE_CALL(cudaMemcpy(f_float, f_dev, memsize_f_float,cudaMemcpyDeviceToHost) );

#if 0
   double dx[6];
   for (int i = 0; i < num_atm; ++i){
     pot[i] = 0.e0;
     for (int k = 0; k < 3; ++k)
       force[i*3+k] = 0.e0;
     for (int j = 0; j < num_atm; ++j){
       for (int k = 0; k < 3; ++k){
	 dx[k] = x[i*3+k] - x[j*3+k];
	 //dx[k] -= rint(dx[k]/xmax[k])*xmax[k];
       }
       dx[3] = dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2];
       dx[3] += 2.e0;
       //if (dx[3] != 0.e0){
	 dx[3] = 1.e0/dx[3];
	 dx[4] = dx[3]*dx[3]*dx[3];
	 pot[i] += 4.e0 * dx[4] * (dx[4] - 1.e0);
	 dx[5] = 24.e0 * dx[3] * dx[4] * (2.e0 * dx[4] - 1.e0);
	 for (int k = 0; k < 3; ++k)
	   force[i*3+k] += dx[5] * dx[k];
	 //}
     }
   }
   double sum[3]={0.0,0.0,0.0};
   for (int i = 0; i < num_atm; ++i)
     for (int k = 0; k < 3; ++k)
       sum[k] += force[i*3+k]*force[i*3+k];
#endif 
   
   for (int i = 0; i < num_atm_i; ++i){
     itmp = (i+iatm_begin)*num_site;
     COPY_SINGLE_TO_FORCEPOT(f_float,force,pot,itmp,i)
   }
#if 0
   double sum0[3]={0.0,0.0,0.0};
   for (int i = 0; i < num_atm; ++i)
     for (int k = 0; k < 3; ++k)
       sum0[k] += force[i*3+k]*force[i*3+k];
   //printf("host %f %f %f\n",sum[0],sum[1],sum[2]);
   //printf("gpu  %f %f %f\n",sum0[0],sum0[1],sum0[2]);
#endif
   
   free(x_float);
   free(xi_float);
   free(f_float);
   free(gr_float);
   free(tex_data);
   CUDA_SAFE_CALL(cudaFree(x_dev));
   CUDA_SAFE_CALL(cudaFree(xi_dev));
   CUDA_SAFE_CALL(cudaFree(f_dev));
}

extern "C"
void
gpuvdwdirectp__
(double* x, int* n, int* atype, int* nat, double* epsilon4, double* sigma2, int* tblno,
 double* xmax, int* periodicflag, int* natchangeflag, double* force, double* pot, int* idevice,
 int* begin_numi, int* num_atmi)
{
  gpuvdwdirectp_
    (x,*n,atype,*nat,epsilon4,sigma2,*tblno,xmax,*periodicflag,*natchangeflag,force,pot,*idevice,
     *begin_numi,*num_atmi);
}
