#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#define BSIZEVDW 240
#include <gpuvdwdirect_kernel.cu>
#include <gpuheader.h>
////////////////////////////////////////////////////////////////////////////////

extern "C"
void gpuvdwdirect_ 
//(double*, int, int*, int, double*, double*, int, double*, int, int, double*, double*,
// int*, int*);
(double*, int, int*, int, double*, double*, int, double*, int, int, double*, double*, int, int, int);
void gpuvdwdirect__
//(double*, int*, int*, int*, double*, double*, int*, double*, int*, int*, double*, double*,
// int*, int*);
//(double*, int, int*, int, double*, double*, int, double*, int, int, double*, double*,
// int*, int*);
(double*, int, int*, int, double*, double*, int, double*, int, int, double*, double*, int, int, int);

extern "C"
void gpuvdwdirect_
(double* x, int num_atmall, int* atype, int nat, double* epsilon4, double* sigma2, int tblno,
 double* xmax, int periodicflag, int natchangeflag, double* force, double* pot, int idevice,
 int begin_numi, int num_atmi)
//(double* x, int num_atm, int* atype, int nat, double* epsilon4, double* sigma2, int tblno,
// double* xmax, int periodicflag, int natchangeflag, double* force, double* pot,
// int* numex, int* natex)
{
  int itmp, itmp1;
   CUT_CHECK_DEVICE();
   CUDA_SAFE_CALL(cudaSetDevice(idevice));
   int tex_l = 512;
   int num_atm;
   int num_site;
   if (atype[3] == 1 && atype[4] != 1) num_site = 3;
   if (atype[3] != 1 && atype[4] == 1) num_site = 4;
   if (atype[3] == 1 && atype[4] == 1) num_site = 1;
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
   int n  = ((int)((float)(num_atm-1)/(float)BSIZEVDW)+1)*BSIZEVDW;
   int ni = ((int)((float)(num_atm_i-1)/(float)BSIZEVDW)+1)*BSIZEVDW;
   float xmax_float0 = (float)xmax[0];
   float xmax_float1 = (float)xmax[1];
   float xmax_float2 = (float)xmax[2];
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
     x_float[i].x = (float)x[i*num_site*3+0];
     x_float[i].y = (float)x[i*num_site*3+1];
     x_float[i].z = (float)x[i*num_site*3+2];
     x_float[i].a = atype[i*num_site]-1;
   }
   for (int i = num_atm; i < n; i++){
     SET_DAMMY_VDWATM(x_float[i],nat)
   }
   for (int i = 0; i < num_atm_i; i++)
     xi_float[i] = x_float[i+iatm_begin];
   for (int i = num_atm_i; i < ni; i++){
     SET_DAMMY_VDWATM(xi_float[i],nat)
   }
   for (int i = 0; i < THD/4; i++){
     SET_FLOAT4_ZERO(gr_float[i])
   }
   for (int i = 0; i < nat; i++)
     for (int j = 0; j < nat; j++){
       itmp = i*(nat+1)+j;
       itmp1 = i*(nat)+j;
       gr_float[itmp].x = 
	 (float)(epsilon4[itmp1]/
		 (sigma2[itmp1]*sigma2[itmp1]*sigma2[itmp1]
		  *sigma2[itmp1]*sigma2[itmp1]*sigma2[itmp1]));
       gr_float[itmp].y = 
	 (float)(epsilon4[itmp1]/(sigma2[itmp1]*sigma2[itmp1]*sigma2[itmp1]));
       gr_float[itmp].z = gr_float[itmp].x*12.f;
       gr_float[itmp].w = gr_float[itmp].y*6.f;
     }
   /*
   double tmp;
   //double threshold = 1.0e0;
   //double threshold = 1.5e0;
   //double rcp_threshold = 1.e0 / threshold;
   for (int i = 0; i < tex_l; i++){
     tmp = ((double)i+0.5e0)/((double)tex_l);
     //if (rcp_threshold < tmp){
       //tex_data[i].x = (float)pow(rcp_threshold,12);
       //tex_data[i].y = (float)pow(rcp_threshold,6);
       //tex_data[i].z = (float)pow(rcp_threshold,14);
       //tex_data[i].w = (float)pow(rcp_threshold,8);
       //tex_data[i].x = 0.f;
       //tex_data[i].y = 0.f;
       //tex_data[i].z = 0.f;
       //tex_data[i].w = 0.f;
     //}else{
       //tex_data[i].x = (float)pow(tmp,12);
       //tex_data[i].y = (float)pow(tmp,6);
       //tex_data[i].z = (float)pow(tmp,14);
       //tex_data[i].w = (float)pow(tmp,8);
       tex_data[i].x = (float)(tmp*tmp*tmp*tmp*tmp*tmp);
       tex_data[i].y = (float)(tmp*tmp*tmp);
       tex_data[i].z = (float)(tmp*tmp*tmp*tmp*tmp*tmp*tmp);
       tex_data[i].w = (float)(tmp*tmp*tmp*tmp);
       //}
     //printf("tex %d %e %e %e %e\n",i,tex_data[i].x,tex_data[i].y,tex_data[i].z,tex_data[i].w);
   }
   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
   cudaArray* cu_array;
   cudaMallocArray(&cu_array, &channelDesc, tex_l);
   cudaMemcpyToArray(cu_array, 0, 0, tex_data, memsize_tex, cudaMemcpyHostToDevice);
   tex.normalized = true;
   tex.addressMode[0] = cudaAddressModeClamp;
   tex.filterMode = cudaFilterModeLinear;
   CUDA_SAFE_CALL(cudaBindTextureToArray(tex, cu_array, channelDesc));
   int num_max_exclude = 0;
   for (int i = 0; i < num_atm; i++)
     if (numex[i] > num_max_exclude)
       num_max_exclude = numex[i];
   unsigned int memsize_exclude = sizeof(int) * n * num_max_exclude;
   int* exclude = (int*) malloc(memsize_exclude);
   itmp = -1;
   for (int i = 0; i < num_atm; i++){
     for (int j = 0; j < numex[i]; j++){
       itmp += 1;
       exclude[i*num_max_exclude+j] = natex[itmp]-1;
     }
     for (int j = numex[i]; j < num_max_exclude; j++)
       exclude[i*num_max_exclude+j] = i;
   }
  for (int i = num_atm; i < n; i++)
    for (int j = 0; j < num_max_exclude; j++)
      exclude[i*num_max_exclude+j] = i;
   */
   
   CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_B, gr_float , memsize_gr_float) );
   vdwatm* x_dev;
   CUDA_SAFE_CALL(cudaMalloc((void**)&x_dev, memsize_x_float));
   CUDA_SAFE_CALL(cudaMemcpy(x_dev, x_float  , memsize_x_float, cudaMemcpyHostToDevice) );
   vdwatm* xi_dev;
   CUDA_SAFE_CALL(cudaMalloc((void**)&xi_dev, memsize_xi_float));
   CUDA_SAFE_CALL(cudaMemcpy(xi_dev, xi_float  , memsize_xi_float, cudaMemcpyHostToDevice) );
   float4* f_dev;
   CUDA_SAFE_CALL(cudaMalloc((void**)&f_dev, memsize_f_float));
   //int* ex_dev;
   //CUDA_SAFE_CALL(cudaMalloc((void**)&ex_dev, memsize_exclude));
   //CUDA_SAFE_CALL(cudaMemcpy(ex_dev, exclude , memsize_exclude, cudaMemcpyHostToDevice) );
   dim3 threads(BSIZEVDW);
   dim3 grid(ni / BSIZEVDW);
   int nat2 = nat + 1;
   gpuvdwdirect_kernel<<< grid, threads >>>
     (f_dev, x_dev, xi_dev, n, xmax_float0, xmax_float1, xmax_float2, nat2);
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
     force[itmp*3+0] = (double)f_float[i].x;
     force[itmp*3+1] = (double)f_float[i].y;
     force[itmp*3+2] = (double)f_float[i].z;
     pot[itmp] = (double)f_float[i].w;
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
   //free(exclude);
   CUDA_SAFE_CALL(cudaFree(x_dev));
   CUDA_SAFE_CALL(cudaFree(xi_dev));
   //CUDA_SAFE_CALL(cudaFree(ex_dev));
   CUDA_SAFE_CALL(cudaFree(f_dev));
   //CUDA_SAFE_CALL(cudaFreeArray(cu_array));
}

extern "C"
void
gpuvdwdirect__
(double* x, int* n, int* atype, int* nat, double* epsilon4, double* sigma2, int* tblno,
 double* xmax, int* periodicflag, int* natchangeflag, double* force, double* pot, int* idevice,
 int* begin_numi, int* num_atmi)
//(double* x, int* n, int* atype, int* nat, double* epsilon4, double* sigma2, int* tblno,
// double* xmax, int* periodicflag, int* natchangeflag, double* force, double* pot,
// int* numex, int* natex)
{
  gpuvdwdirect_
    (x,*n,atype,*nat,epsilon4,sigma2,*tblno,xmax,*periodicflag,*natchangeflag,force,pot,*idevice,
     *begin_numi,*num_atmi);
  //(x,*n,atype,*nat,epsilon4,sigma2,*tblno,xmax,*periodicflag,*natchangeflag,force,pot,
  //   numex,natex);
}
