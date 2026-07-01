#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <gpuheader.h>
#include <gpuvdwpot256_kernel.cu>
////////////////////////////////////////////////////////////////////////////////
extern "C"
void gpuvdwpot256_ (double*, int, int*, int, double*, double*, int, double, int, int, double*);
void gpuvdwpot256__ (double*, int, int*, int, double*, double*, int, double, int, int, double*);

extern "C"
void
gpuvdwpot256_ (double* x, int num_atm, int* atype, int nat, double* gscale, double* rscale, int tblno, double xmax, int periodicflag, int natchangeflag, double* force)
{
   CUT_CHECK_DEVICE();
   
   int tex_l = TEX;
   int n = ((int)((float)(num_atm-1)/(float)THD)+1)*THD;
   //printf("%d\n",n);
   float xmax_float = (float)xmax;
   //double cutoff = 10.e0;
   //float rcut = 1.f/(float)(cutoff*cutoff);
   int thd2 = (int)sqrt((float)(THD/2));
   if (thd2 < nat){
     printf("WARNING:gpuvdwforce256 TOO MUCH NUMBER OF ATM TYPE\n");
   }

   unsigned int size_A = n * 3;
   unsigned int mem_size_A = sizeof(float) * size_A;
   float* x_float = (float*) malloc(mem_size_A);
   unsigned int size_B = THD;
   unsigned int mem_size_B = sizeof(float) * size_B;
   float* gr_float = (float*) malloc(mem_size_B);
   unsigned int size_C = n * 3;
   unsigned int mem_size_C = sizeof(float) * size_C;
   float* f_float = (float*) malloc(mem_size_A);
   unsigned int size_D = n;
   unsigned int mem_size_D = sizeof(int) * size_D;
   int* atype_int = (int*) malloc(mem_size_D);
   unsigned int size_E = tex_l;
   unsigned int mem_size_E = sizeof(float) * size_E;
   float* tex_data = (float*) malloc(mem_size_E);

   for (int i = 0; i < num_atm; i++){
     x_float[i*3]   = (float)x[i*3];
     x_float[i*3+1] = (float)x[i*3+1];
     x_float[i*3+2] = (float)x[i*3+2];
     atype_int[i]   = atype[i];
   }
   if (num_atm != n)
     for (int i = num_atm; i < n; i++){
       x_float[i*3]   = 0.f;
       x_float[i*3+1] = 0.f;
       x_float[i*3+2] = 0.f;
       atype_int[i]   = thd2;
     }
   for (int i = 0; i < nat*nat; i++){
     gr_float[i*2]    = (float)gscale[i];
     //gr_float[i*2+1]  = (float)rscale[i];
     gr_float[i*2+1]  = (float)(1.e0/(rscale[i]*rscale[i]*rscale[i]));
   }
   for (int i = nat*nat; i < THD/2; i++){
     gr_float[i*2]   = 0.f;
     gr_float[i*2+1] = 1.f;
   }
   double tmp;
   for (int i = 0; i < tex_l; i++){
     tmp = ((double)(i)+0.5)/((double)tex_l);
     //tex_data[i] = (float)(tmp);
     tex_data[i] = (float)(tmp);
     //tex_data[i] = (float)(tmp*tmp*tmp);
   }
   //tex_data[tex_l-1] = 0.f;
   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
   cudaArray* cu_array;
   cudaMallocArray(&cu_array, &channelDesc, tex_l);
   cudaMemcpyToArray(cu_array, 0, 0, tex_data, mem_size_E, cudaMemcpyHostToDevice);
   //tex.addressMode[0] = cudaAddressModeWrap;
   tex.normalized = true;
   tex.addressMode[0] = cudaAddressModeClamp;
   tex.filterMode = cudaFilterModeLinear;
   CUDA_SAFE_CALL(cudaBindTextureToArray(tex, cu_array, channelDesc));

   float* d_A;
   CUDA_SAFE_CALL(cudaMalloc((void**) &d_A, mem_size_A));
   //float* d_B;
   //CUDA_SAFE_CALL(cudaMalloc((void**) &d_B, mem_size_B));
   float* d_C;
   CUDA_SAFE_CALL(cudaMalloc((void**) &d_C, mem_size_C));
   int* d_D;
   CUDA_SAFE_CALL(cudaMalloc((void**) &d_D, mem_size_D));

   CUDA_SAFE_CALL(cudaMemcpy(d_A, x_float  , mem_size_A, cudaMemcpyHostToDevice) );
   //CUDA_SAFE_CALL(cudaMemcpy(d_B, gr_float , mem_size_B, cudaMemcpyHostToDevice) );
   CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_B, gr_float , mem_size_B) );
   CUDA_SAFE_CALL(cudaMemcpy(d_D, atype_int, mem_size_D, cudaMemcpyHostToDevice) );

   dim3 threads(THD);
   dim3 grid(n / THD);
   gpuvdwpot256_kernel<<< grid, threads >>>(d_C, d_A, d_D, n, xmax_float, nat);
   //gpuvdwpot256_kernel<<< grid, threads >>>(d_C, d_A, d_B, d_D, n, xmax_float, nat);

   CUT_CHECK_ERROR("Kernel execution failed");
   CUDA_SAFE_CALL(cudaMemcpy(f_float, d_C, mem_size_C, cudaMemcpyDeviceToHost) );

   float ftmpin, ftmp2;
   int itmp1; 
   for(int i = 0; i < num_atm; ++i){
     // ftmpin is normalized dummy input of texture.
     ftmpin = (float)(i+1)/(float)num_atm;
     ftmp2 = ftmpin*(tex_l)-0.5f;
     itmp1 = (int)ftmp2;
     //f_float[i*3+1] = (float)ftmp1;
     //f_float[i*3+1] = ((float)(i+1)/(float)num_atm)*((float)(i+1)/(float)num_atm);
     f_float[i*3+1] = tex_data[itmp1]
       + (tex_data[itmp1+1] - tex_data[itmp1])
       * (ftmp2 - (float)itmp1);
     //f_float[i*3+1] = tex_data[itmp1];
     //f_float[i*3+1] = (ftmp2 - (float)itmp1);
   }
   // (y2-y1)/(x2-x1)*(x-x1)+y1

   for (int i = 0; i < num_atm; ++i){
     force[i*3]   += (double)f_float[i*3];
     force[i*3+1] += (double)f_float[i*3+1];
     //force[i*3+2] += (double)f_float[i*3+2];
   }

   free(x_float);
   free(gr_float);
   free(f_float);
   free(atype_int);
   free(tex_data);
   CUDA_SAFE_CALL(cudaFree(d_A));
   //CUDA_SAFE_CALL(cudaFree(d_B));
   CUDA_SAFE_CALL(cudaFree(d_C));
   CUDA_SAFE_CALL(cudaFree(d_D));
   CUDA_SAFE_CALL(cudaFreeArray(cu_array));
}

/*
extern "C"
void
gpuvdwpot_ (double* x, int *n, int* atype, int *nat, double* gscale, double* rscale, int *tblno, double *xmax, int *periodicflag, int *natchangeflag, double* force)
{
  gpuvdwpot_ (x,*n,atype,*nat,gscale,rscale,*tblno,*xmax,*periodicflag,*natchangeflag,force);
}
*/

extern "C"
void
gpuvdwpot256__ (double* x, int *n, int* atype, int *nat, double* gscale, double* rscale, int *tblno, double *xmax, int *periodicflag, int *natchangeflag, double* force)
{
  gpuvdwpot256_ (x,*n,atype,*nat,gscale,rscale,*tblno,*xmax,*periodicflag,*natchangeflag,force);
  }

/*
void
hostvdwpot256(float* C, const double* A, const double* B, unsigned int num_a, double xmax)
{

  double sum;
  double dn2;
  double dn6;
  double dx;
  double dy;
  double dz;
  double l2 = xmax * 0.5;
  double exclude_radius2 = 0.1;

  for (unsigned int i = 0; i < num_a; ++i){
    sum = 0;
    for (unsigned int j = 0; j < num_a; ++j) {

      dx = (A[i*3]   - A[j*3]  );
      dy = (A[i*3+1] - A[j*3+1]);
      dz = (A[i*3+2] - A[j*3+2]);

      if (!(dx < l2 && dx > -l2))
	if (dx > l2){
	  dx = dx - xmax;
	}else{
	  dx = dx + xmax;
	}
      if (!(dy < l2 && dy > -l2))
	if (dy > l2){
	  dy = dy - xmax;
	}else{
	  dy = dy + xmax;
	}
      if (!(dz < l2 && dz > -l2))
	if (dz > l2){
	  dz = dz - xmax;
	}else{
	  dz = dz + xmax;
	}

      dn2 = (dx * dx + dy * dy + dz * dz) * 1;
      if (dn2 > exclude_radius2){
	dn6 = 1.e0 / (dn2 * dn2 * dn2);
	sum += 4 * dn6 * (dn6 - 1.e0);
      }
    }
    C[i*3] = (float)sum;
  }
}
void
computeGold_d(double* C, const double* A, const double* B, unsigned int num_a, double xmax)
{

  double sum;
  double dn2;
  double dn6;
  double dx;
  double dy;
  double dz;
  double l2 = xmax * 0.5;
  //double exclude_radius2 = 0.01e0;
  double cutoff_radius2 = 9.e0;

  for (unsigned int i = 0; i < num_a; ++i){
    sum = 0.e0;
    for (unsigned int j = 0; j < num_a; ++j) {

      dx = (A[i*3]   - A[j*3]  );
      dy = (A[i*3+1] - A[j*3+1]);
      dz = (A[i*3+2] - A[j*3+2]);

      if (!(dx < l2 && dx > -l2))
	if (dx > l2){
	  dx = dx - xmax;
	}else{
	  dx = dx + xmax;
	}
      if (!(dy < l2 && dy > -l2))
	if (dy > l2){
	  dy = dy - xmax;
	}else{
	  dy = dy + xmax;
	}
      if (!(dz < l2 && dz > -l2))
	if (dz > l2){
	  dz = dz - xmax;
	}else{
	  dz = dz + xmax;
	}

      dn2 = dx * dx + dy * dy + dz * dz;
      if ((i != j) && dn2 < cutoff_radius2){
	dn6 = 1.e0 / dn2 / dn2 / dn2 / dn2 / dn2 / dn2
	  - 1.e0 / dn2 / dn2 / dn2;
	sum += dn6 * 4.e0;
      }
      
      dn2 = (dx * dx + dy * dy + dz * dz) * 1;
      if (dn2 > exclude_radius2 && dn2 < cutoff_radius2){
	dn6 = 1.e0 / (dn2 * dn2 * dn2);
	sum += 4 * dn6 * (dn6 - 1.e0);
	}
    }
    C[i*3] = sum;
  }
}
void
computeGold_f(float* C, const float* A, const float* B, unsigned int num_a, float xmax)
{

  float sum;
  float dn2;
  float dn6;
  float dx;
  float dy;
  float dz;
  float l2 = xmax * 0.5;
  //float exclude_radius2 = 0.01e0;
  float cutoff_radius2 = 9.e0;

  for (unsigned int i = 0; i < num_a; ++i){
    sum = 0.e0;
    for (unsigned int j = 0; j < num_a; ++j) {

      dx = (A[i*3]   - A[j*3]  );
      dy = (A[i*3+1] - A[j*3+1]);
      dz = (A[i*3+2] - A[j*3+2]);

      if (!(dx < l2 && dx > -l2))
	if (dx > l2){
	  dx = dx - xmax;
	}else{
	  dx = dx + xmax;
	}
      if (!(dy < l2 && dy > -l2))
	if (dy > l2){
	  dy = dy - xmax;
	}else{
	  dy = dy + xmax;
	}
      if (!(dz < l2 && dz > -l2))
	if (dz > l2){
	  dz = dz - xmax;
	}else{
	  dz = dz + xmax;
	}

      dn2 = dx * dx + dy * dy + dz * dz;
      if ((i != j) && dn2 < cutoff_radius2){
	dn6 = 1.e0 / dn2 / dn2 / dn2 / dn2 / dn2 / dn2
	  - 1.e0 / dn2 / dn2 / dn2;
	sum += dn6 * 4.e0;
      }
      
      dn2 = (dx * dx + dy * dy + dz * dz) * 1;
      if (dn2 > exclude_radius2 && dn2 < cutoff_radius2){
	dn6 = 1.e0 / (dn2 * dn2 * dn2);
	sum += 4 * dn6 * (dn6 - 1.e0);
	}
    }
    C[i*3] = sum;
  }
}
*/
