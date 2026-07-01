#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <gpuheader.h>
#include <gpuewrecell_kernel.cu>
////////////////////////////////////////////////////////////////////////////////
extern "C"
void gpuewrecell_ 
(double*, int, double*, double, int, double*, int, int, double*, double*,
 int, int, int, double, int, int*, int);
void gpuewrecell__
(double*, int, double*, double, int, double*, int, int, double*, double*,
 int, int, int, double, int, int*, int);

extern "C"
void gpuewrecell_
(double* x, int num_atm, double* q, double rscale, int tblno, double* xmax,
 int periodicflag, int natchangeflag, double* force, double* pot, int idevice,
 int begin_numi, int num_atmi, double radius_cut, int num_div_cell, int* num_calc,
 int disp)
{
  float tmp, tmp3[3];
  int itmp, itmp1, itmp2, itmp0, itmp33[3], itmp32[3], itmp3[3], i, j, k;
  double dtmp3[3];

  //CUT_CHECK_DEVICE();
  CUDA_SAFE_CALL(cudaSetDevice(idevice));
  int num_cell[3], num_cell_psd[3];
  double cell_width[3];
  float dom_width_float[3], rcell[3];

  for(i = 0; i < 3; i++){
    dom_width_float[i] = (float)xmax[i];
    num_cell[i] = (int)((double)num_div_cell*xmax[i]/radius_cut);
    if (num_cell[i] < num_div_cell*2+1)
      printf("ERROR gpuewrecell : too small domain size \n");
    cell_width[i] = xmax[i]/(double)num_cell[i];
    rcell[i] = 1.f / (float)(cell_width[i]);
    num_cell_psd[i] = num_cell[i] + 2*num_div_cell;
  }
  int total_num_cell = num_cell[0] * num_cell[1] * num_cell[2];
  int total_num_cell_psd = num_cell_psd[0] * num_cell_psd[1] * num_cell_psd[2];
  float* xinbox = (float*) malloc(sizeof(float)*num_atm*3);
  for(i = 0; i < num_atm; i++){
    SET_POS_TO_REGION(0,i,i,xmax,xinbox,x)
    SET_POS_TO_REGION(1,i,i,xmax,xinbox,x)
    SET_POS_TO_REGION(2,i,i,xmax,xinbox,x)
  }
  int* cell_base  = (int*) malloc(sizeof(int)*total_num_cell);
  int* cell_size  = (int*) malloc(sizeof(int)*total_num_cell);
  int* cell_link  = (int*) malloc(sizeof(int)*num_atm);
  int* cell_order = (int*) malloc(sizeof(int)*total_num_cell);
  for (i = 0; i < num_atm; i++)
    cell_link[i] = 0;
  for (i = 0; i < total_num_cell; i++){
    cell_size[i] = 0;
    cell_base[i] = 0;
  }
  for(i = num_atm-1; i > -1; i--){
    itmp3[0] = (int)(xinbox[i*3+0]*rcell[0]);
    itmp3[1] = (int)(xinbox[i*3+1]*rcell[1]);
    itmp3[2] = (int)(xinbox[i*3+2]*rcell[2]);
    GET_3DIM_TO_1DIM(itmp,itmp3,num_cell)
    cell_link[i] = cell_base[itmp];
    cell_size[itmp] += 1;
    cell_base[itmp] =  i;
  }
  int num_cell_valid = 0;
  int max_cell_size = 1;
  int sum_cell_size = 0;
  for (i = 0; i < total_num_cell; i++){
    sum_cell_size = sum_cell_size + cell_size[i];
    if (cell_size[i] > max_cell_size)
      max_cell_size = cell_size[i];
    if (cell_size[i] > 0){
      cell_order[i] = num_cell_valid;
      num_cell_valid += 1;
    }else{
      cell_order[i] = -1; // 0 ~ 
    }
  }
  int num_thread_per_cell = (int)((float)(max_cell_size-1)/(float)BEWRCELL)+1;
  int num_max_atm_per_cell = BEWRCELL*num_thread_per_cell;
  int num_atm_i = num_cell_valid*num_max_atm_per_cell;
  int num_atm_j = total_num_cell_psd*num_max_atm_per_cell;
  int memsize_x_float = sizeof(float4)*num_atm_i;
  float4* x_float = (float4*) malloc(memsize_x_float);
  int memsize_f_float = sizeof(float4)*num_atm_i;
  float4* f_float = (float4*) malloc(memsize_f_float);
  int memsize_x_float_psd = sizeof(float4)*num_atm_j;
  float4* x_float_psd = (float4*) malloc(memsize_x_float_psd);
  for (i = 0; i < total_num_cell; i++){
    if (cell_size[i] > 0){
      if (cell_size[i] > num_max_atm_per_cell)
	printf("ERROR gpuewrecell : too small num_max_atm_per_cell \n");
      itmp = cell_order[i]*num_max_atm_per_cell;
      itmp1 = cell_base[i];
      for (j = 0; j < cell_size[i]; j++){
	x_float[itmp+j].x = xinbox[itmp1*3+0];
	x_float[itmp+j].y = xinbox[itmp1*3+1];
	x_float[itmp+j].z = xinbox[itmp1*3+2];
	x_float[itmp+j].w = (float)q [itmp1];
	itmp1 = cell_link[itmp1];
      }
      for (j = cell_size[i]; j < num_max_atm_per_cell; j++){
	SET_FLOAT4_ZERO(x_float[itmp+j])
      }
    }
  }
  for (itmp3[0] = 0; itmp3[0] < num_cell_psd[0]; itmp3[0]++){
    SET_CELL_IMAGE(0,itmp3,num_div_cell,itmp32,num_cell,tmp3,num_cell_psd)
    for (itmp3[1] = 0; itmp3[1] < num_cell_psd[1]; itmp3[1]++){
      SET_CELL_IMAGE(1,itmp3,num_div_cell,itmp32,num_cell,tmp3,num_cell_psd)
      for (itmp3[2] = 0; itmp3[2] < num_cell_psd[2]; itmp3[2]++){
	SET_CELL_IMAGE(2,itmp3,num_div_cell,itmp32,num_cell,tmp3,num_cell_psd)
	GET_3DIM_TO_1DIM(itmp1,itmp3,num_cell_psd)
	GET_3DIM_TO_1DIM(itmp2,itmp32,num_cell)
	itmp = cell_order[itmp2] * num_max_atm_per_cell;
	itmp0 = itmp1 * num_max_atm_per_cell;
	if (cell_size[itmp2] > 0){
	  for (i = 0; i < num_max_atm_per_cell; i++){
	    x_float_psd[itmp0+i].x = x_float[itmp+i].x - tmp3[0] * dom_width_float[0];
	    x_float_psd[itmp0+i].y = x_float[itmp+i].y - tmp3[1] * dom_width_float[1];
	    x_float_psd[itmp0+i].z = x_float[itmp+i].z - tmp3[2] * dom_width_float[2];
	    x_float_psd[itmp0+i].w = x_float[itmp+i].w;
	  }
	}else{
	  for (i = 0; i < num_max_atm_per_cell; i++){
	    SET_FLOAT4_ZERO(x_float_psd[itmp0+i])
	  }
	}
      }
    }
  }
  itmp = num_div_cell*2+1;
  int* tmp_round = (int*)malloc(itmp*itmp*itmp*sizeof(int)*3);
  GET_ROUND_CELL(itmp,itmp3,num_div_cell,dtmp3,cell_width,tmp,radius_cut,tmp_round)
  int num_round_cell = itmp;
  if (itmp*num_thread_per_cell > BEWRCELL)
    printf("ERROR!!! gpuewrecell too much number of round cell num_round_cell:%d num_thread_per_cell:%d\n",num_round_cell,num_thread_per_cell);
  int memsize_jcelllist = sizeof(int) * num_atm_i;
  int* jcelllist = (int*) malloc(memsize_jcelllist);
  for (itmp3[0] = 0; itmp3[0] < num_cell[0]; itmp3[0]++){
    itmp32[0] = itmp3[0] + num_div_cell;
    for (itmp3[1] = 0; itmp3[1] < num_cell[1]; itmp3[1]++){
      itmp32[1] = itmp3[1] + num_div_cell;
      for (itmp3[2] = 0; itmp3[2] < num_cell[2]; itmp3[2]++){
	itmp32[2] = itmp3[2] + num_div_cell;
	GET_3DIM_TO_1DIM(itmp1,itmp3,num_cell)
	if (cell_size[itmp1] > 0)
	  itmp = cell_order[itmp1]*BEWRCELL;
	  for (i = 0; i < num_round_cell; i++){
	    itmp33[0] = itmp32[0] + tmp_round[i*3+0];
	    itmp33[1] = itmp32[1] + tmp_round[i*3+1];
	    itmp33[2] = itmp32[2] + tmp_round[i*3+2];
	    GET_3DIM_TO_1DIM(itmp2,itmp33,num_cell_psd)
	    for (k = 0; k < num_thread_per_cell; k++)
	      for (j = 0; j < num_thread_per_cell; j++){
		jcelllist[itmp+BEWRCELL*k+i*num_thread_per_cell+j]
		  = itmp2*num_max_atm_per_cell+BEWRCELL*j;
	      }
	  }
      }
    }
  }

  int tex_l = 1024;
  unsigned int memsize_tex = sizeof(float2) * tex_l;
  float2* tex_data = (float2*) malloc(memsize_tex);
  double dtmp0, dtmp1, dtmp2;
  for (int i = 0; i < tex_l; i++){
    dtmp0 = ((double)i+0.5e0)/((double)tex_l);
    dtmp0 = rscale * dtmp0 * radius_cut;
    dtmp1 = 1.e0 / dtmp0;
    dtmp2 = erfc(dtmp0) * dtmp1;
    tex_data[i].x = (float)
      (rscale*rscale*rscale*(dtmp2+RSQRTPI2*exp(-dtmp0*dtmp0))*dtmp1*dtmp1);
    tex_data[i].y = (float)(rscale*dtmp2);
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

  float4* d_A;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_A, memsize_x_float));
  float4* d_E;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_E, memsize_x_float_psd));
  float4* d_C;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_C, memsize_f_float));
  int* d_D;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_D, memsize_jcelllist));
  CUDA_SAFE_CALL(cudaMemcpy(d_A,x_float    ,memsize_x_float    ,cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_E,x_float_psd,memsize_x_float_psd,cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_D,jcelllist  ,memsize_jcelllist  ,cudaMemcpyHostToDevice));
  num_calc[0] = num_round_cell*num_thread_per_cell*num_thread_per_cell*num_cell_valid*BEWRCELL*BEWRCELL;

  dim3 threads(BEWRCELL);
  dim3 grid(num_thread_per_cell*num_cell_valid);
  float radius2 = (float)(radius_cut*radius_cut);
  float rcut = (float)(1.e0/radius_cut);
  if (disp == 1){
    printf("--ewrecell-----------------------------\n");
    printf("capacity number of atms per a cell %d\n",num_max_atm_per_cell);
    printf("number of total cells %d\n",total_num_cell);
    printf("number of cells around a cell %d\n",num_round_cell);
    printf("max number of atms in a cell %d\n",max_cell_size);
    printf("ave number of atms in a cell %f\n",(float)sum_cell_size/(float)total_num_cell);
  }
  gpuewrecell_kernel<<< grid , threads >>>
    (d_D,d_C,d_A,d_E,num_round_cell,num_thread_per_cell,radius2,rcut);
  CUT_CHECK_ERROR("Kernel execution failed");
  CUDA_SAFE_CALL(cudaMemcpy(f_float, d_C, memsize_f_float, cudaMemcpyDeviceToHost));

  for (i = 0; i < total_num_cell; i++)
    if (cell_size[i] > 0){
      itmp = cell_order[i]*num_max_atm_per_cell;
      itmp1 = cell_base[i];
      for (j = 0; j < cell_size[i]; j++){
	COPY_SINGLE_TO_FORCEPOT(f_float,force,pot,itmp1,itmp+j)
	itmp1 = cell_link[itmp1];
      }
    }

  free(xinbox);
  free(cell_base);
  free(cell_size);
  free(cell_link);
  free(cell_order);
  free(tmp_round);
  free(jcelllist);
  free(x_float);
  free(x_float_psd);
  free(f_float);
  free(tex_data);
  CUDA_SAFE_CALL(cudaFreeArray(cu_array));
  CUDA_SAFE_CALL(cudaFree(d_A));
  CUDA_SAFE_CALL(cudaFree(d_C));
  CUDA_SAFE_CALL(cudaFree(d_D));
  CUDA_SAFE_CALL(cudaFree(d_E));
}

extern "C"
void
gpuewrecell__
(double* x, int* num_atm, double* q, double* rscale, int* tblno, double* xmax,
 int* periodicflag, int* natchangeflag, double* force, double* pot, int* idevice,
 int* begin_numi, int* num_atmi, double* radius_cut, int* num_div_cell, int* num_calc,
 int* disp)
{
  gpuewrecell_
    (x,*num_atm,q,*rscale,*tblno,xmax,*periodicflag,*natchangeflag,force,pot,*idevice,
     *begin_numi,*num_atmi,*radius_cut,*num_div_cell,num_calc,*disp);
}
