#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#define BEWRDIRC 256
#include <gpuheader.h>
#include <gpuewrecutoff_kernel.cu>
////////////////////////////////////////////////////////////////////////////////
extern "C"
void gpuewrecutoff_ 
(double*, int, double*, double, int, double*, int, int, double*, double*, int, int, int, double);
void gpuewrecutoff__
(double*, int, double*, double, int, double*, int, int, double*, double*, int, int, int, double);

extern "C"
void gpuewrecutoff_
(double* x, int num_atm, double* q, double rscale, int tblno, double* xmax,
 int periodicflag, int natchangeflag, double* force, double* pot, int idevice,
 int begin_numi, int num_atmi, double radius_cut)
{
  CUDA_SAFE_CALL(cudaSetDevice(idevice));
  int tex_l = 1024;
  int itmp;
  int num_atm_i = num_atmi;
  int iatm_begin = begin_numi - 1;
  int n, ni;
  float xmax_float0, xmax_float1, xmax_float2, rcp_xmax0, rcp_xmax1, rcp_xmax2;
  ROLLING_FORWARD(n,num_atm,BEWRDIRC)
  ROLLING_FORWARD(ni,num_atm_i,BEWRDIRC)
  SET_XMAX_FLOAT(xmax_float0,xmax_float1,xmax_float2,xmax)
  SET_RCP_XMAX(rcp_xmax0,rcp_xmax1,rcp_xmax2,xmax)	
  unsigned int memsize_x_float = sizeof(float4) * n;
  float4* x_float = (float4*) malloc(memsize_x_float);
  unsigned int memsize_xi_float = sizeof(float4) * ni;
  float4* xi_float = (float4*) malloc(memsize_xi_float);
  unsigned int memsize_f_float = sizeof(float4) * ni;
  float4* f_float = (float4*) malloc(memsize_f_float);
  unsigned int memsize_tex = sizeof(float2) * tex_l;
  float2* tex_data = (float2*) malloc(memsize_tex);

  for (int i = 0; i < num_atm; i++){
    COPY_DOUBLE_TO_COULOMB(x_float,x,q,i,i)
  }
  for (int i = num_atm; i < n; i++){
    SET_FLOAT4_ZERO(x_float[i])
  }
  for (int i = 0; i < num_atm_i; i++)
    xi_float[i] = x_float[i+iatm_begin];
  for (int i = num_atm_i; i < ni; i++){
    SET_FLOAT4_ZERO(xi_float[i])
  }

  double tmp, tmp1, tmp2;
  for (int i = 0; i < tex_l; i++){
    tmp = ((double)i+0.5e0)/((double)tex_l);
    tmp = rscale * tmp * radius_cut;
    tmp1 = 1.e0 / tmp;
    tmp2 = erfc(tmp) * tmp1;
    tex_data[i].x = (float)
      (rscale*rscale*rscale*(tmp2+RSQRTPI2*exp(-tmp*tmp))*tmp1*tmp1);
    tex_data[i].y = (float)(rscale*tmp2);
  }
  tex_data[tex_l-1].x = 0.f;
  tex_data[tex_l-1].y = 0.f;
  tex_data[0].x = 0.f;
  tex_data[0].y = 0.f;

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float2>();
  cudaArray* cu_array;
  cudaMallocArray(&cu_array, &channelDesc, tex_l);
  cudaMemcpyToArray(cu_array, 0, 0, tex_data, memsize_tex, cudaMemcpyHostToDevice);
  tex.normalized = true;
  tex.addressMode[0] = cudaAddressModeClamp;
  tex.filterMode = cudaFilterModeLinear;
  CUDA_SAFE_CALL(cudaBindTextureToArray(tex, cu_array, channelDesc));
   
  float4* x_dev;
  CUDA_SAFE_CALL(cudaMalloc((void**)&x_dev, memsize_x_float));
  float4* xi_dev;
  CUDA_SAFE_CALL(cudaMalloc((void**)&xi_dev, memsize_xi_float));
  float4* f_dev;
  CUDA_SAFE_CALL(cudaMalloc((void**)&f_dev, memsize_f_float));
  CUDA_SAFE_CALL(cudaMemcpy(x_dev, x_float , memsize_x_float, cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL(cudaMemcpy(xi_dev, xi_float , memsize_xi_float, cudaMemcpyHostToDevice) );
  float cut2 = (float)(radius_cut*radius_cut);
  float rcut = (float)(1.e0/radius_cut);
  dim3 threads(BEWRDIRC);
  dim3 grid(ni/BEWRDIRC);
  gpuewrecutoff_kernel<<< grid, threads >>>
    (f_dev, x_dev, xi_dev, n, xmax_float0, xmax_float1, xmax_float2,
     rcp_xmax0, rcp_xmax1, rcp_xmax2, cut2, rcut);
  CUT_CHECK_ERROR("Kernel execution failed");
  CUDA_SAFE_CALL(cudaMemcpy(f_float, f_dev, memsize_f_float,cudaMemcpyDeviceToHost) );

  for (int i = 0; i < num_atm_i; ++i){
    itmp = i+iatm_begin;
    COPY_SINGLE_TO_FORCEPOT(f_float,force,pot,itmp,i)
  }
   
  free(x_float);
  free(xi_float);
  free(f_float);
  free(tex_data);
  CUDA_SAFE_CALL(cudaFree(x_dev));
  CUDA_SAFE_CALL(cudaFree(xi_dev));
  CUDA_SAFE_CALL(cudaFree(f_dev));
  CUDA_SAFE_CALL(cudaFreeArray(cu_array));
}

extern "C"
void
gpuewrecutoff__
(double* x, int* num_atm, double* q, double* rscale, int* tblno, double* xmax,
 int* periodicflag, int* natchangeflag, double* force, double* pot, int* idevice,
 int* begin_numi, int* num_atmi, double* radius_cut)
{
  gpuewrecutoff_
    (x,*num_atm,q,*rscale,*tblno,xmax,*periodicflag,*natchangeflag,force,pot,*idevice,
     *begin_numi,*num_atmi,*radius_cut);
}

