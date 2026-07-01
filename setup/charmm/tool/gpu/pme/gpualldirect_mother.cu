#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <gpuheader.h>
#include <gpuvdwdirect_kernel.cu>
////////////////////////////////////////////////////////////////////////////////
extern "C"
void gpuvdwdirect_ 
(double*, int, int*, int, double*, double*, int, double*, int, int, double*, double*,
 int*, int*);
void gpuvdwdirect__
(double*, int, int*, int, double*, double*, int, double*, int, int, double*, double*,
 int*, int*);

extern "C"
void gpuvdwdirect_
(double* x, int num_atm, int* atype, int nat, double* epsilon4, double* sigma2, int tblno,
 double* xmax, int periodicflag, int natchangeflag, double* force, double* pot,
 int* numex, int* natex)
{
   CUT_CHECK_DEVICE();
   int tex_l = 1024;
   int n = ((int)((float)num_atm/(float)THD)+1)*THD;
   float xmax_float0 = (float)xmax[0];
   float xmax_float1 = (float)xmax[1];
   float xmax_float2 = (float)xmax[2];
   int thd2 = (int)sqrt((float)(THD/4));
   if (thd2 < nat)
     printf("WARNING:gpuvdwforce256 TOO MUCH NUMBER OF ATM TYPE");
   unsigned int memsize_x_float = sizeof(float) * n * 4;
   float* x_float = (float*) malloc(memsize_x_float);
   unsigned int memsize_gr_float = sizeof(float) * THD;
   float* gr_float = (float*) malloc(memsize_gr_float);
   unsigned int memsize_f_float = sizeof(float) * n * 4;
   float* f_float = (float*) malloc(memsize_f_float);
   unsigned int memsize_tex = sizeof(float) * tex_l;
   float* tex_data = (float*) malloc(memsize_tex);

   for (int i = 0; i < num_atm; i++){
     for (int j = 0; j < 3; j++)
       x_float[i*4+j]   = (float)x[i*3+j];
     x_float[i*4+3]   = (float)atype[i]-1;
     //x_float[i*3+3] = (float*)(atype)[i];
   }
   for (int i = num_atm; i < n; i++){
     for (int j = 0; j < 3; j++)
       x_float[i*4+j]   = 0.f;
     x_float[i*4+3]   = (float)thd2-1;
   }
   for (int i = 0; i < nat*nat; i++){
     gr_float[i*4]   = (float)epsilon4[i];
     gr_float[i*4+1] = (float)sigma2[i];
     gr_float[i*4+2] = (float)(epsilon4[i]*6.e0*sigma2[i]);
     gr_float[i*4+3] = (float)sigma2[i];
   }
   for (int i = nat*nat; i < THD/4; i++){
     gr_float[i*4]   = 0.f;
     gr_float[i*4+1] = 1.f;
     gr_float[i*4+2] = 0.f;
     gr_float[i*4+3] = 0.f;
   }
   double tmp;
   for (int i = 0; i < tex_l; i++){
     tmp = ((double)i+0.5e0)/((double)tex_l);
     tex_data[i] = (double)(1.e0/tmp);
   }
   tex_data[tex_l-1] = 0.f;
   int num_max_exclude = 0;
   for (int i = 0; i < num_atm; i++)
     if (numex[i] > num_max_exclude)
       num_max_exclude = numex[i];
   unsigned int memsize_exclude = sizeof(int) * n * num_max_exclude;
   int* exclude = (int*) malloc(memsize_exclude);
   int itmp;
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

   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
   cudaArray* cu_array;
   cudaMallocArray(&cu_array, &channelDesc, tex_l);
   cudaMemcpyToArray(cu_array, 0, 0, tex_data, memsize_tex, cudaMemcpyHostToDevice);
   tex.normalized = true;
   tex.addressMode[0] = cudaAddressModeClamp;
   tex.filterMode = cudaFilterModeLinear;
   CUDA_SAFE_CALL(cudaBindTextureToArray(tex, cu_array, channelDesc));
   
   CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_B, gr_float , memsize_gr_float) );
   float* x_dev;
   CUDA_SAFE_CALL(cudaMalloc((void**)&x_dev, memsize_x_float));
   CUDA_SAFE_CALL(cudaMemcpy(x_dev, x_float  , memsize_x_float, cudaMemcpyHostToDevice) );
   float* f_dev;
   CUDA_SAFE_CALL(cudaMalloc((void**)&f_dev, memsize_f_float));
   int* ex_dev;
   CUDA_SAFE_CALL(cudaMalloc((void**)&ex_dev, memsize_exclude));
   CUDA_SAFE_CALL(cudaMemcpy(ex_dev, exclude , memsize_exclude, cudaMemcpyHostToDevice) );
   dim3 threads(THD);
   dim3 grid(n / THD);
   gpuvdwdirect_kernel<<< grid, threads >>>
     (f_dev, x_dev, n, xmax_float0, xmax_float1, xmax_float2, nat);
   if (num_max_exclude != 0)
     //gpuvdwdirect_kernel2<<< grid, threads >>>
     //  (f_dev, x_dev, n, xmax_float0, xmax_float1, xmax_float2, nat, num_max_exclude,
     //	ex_dev);
   CUT_CHECK_ERROR("Kernel execution failed");
   CUDA_SAFE_CALL(cudaMemcpy(f_float, f_dev, memsize_f_float,cudaMemcpyDeviceToHost) );

   double tmp_pot = 0.e0;
   for (int i = 0; i < num_atm; ++i){
     for (int j = 0; j < 3; ++j)
       force[i*3+j] += (double)f_float[i*4+j];
     tmp_pot += (double)f_float[i*4+3];
   }
   *pot = *pot + tmp_pot*0.5e0;
   
   free(x_float);
   free(f_float);
   free(gr_float);
   free(tex_data);
   free(exclude);
   CUDA_SAFE_CALL(cudaFree(x_dev));
   CUDA_SAFE_CALL(cudaFree(ex_dev));
   CUDA_SAFE_CALL(cudaFree(f_dev));
   CUDA_SAFE_CALL(cudaFreeArray(cu_array));
}

extern "C"
void
gpuvdwdirect__
(double* x, int* n, int* atype, int* nat, double* epsilon4, double* sigma2, int* tblno,
 double* xmax, int* periodicflag, int* natchangeflag, double* force, double* pot,
 int* numex, int* natex)
{
  gpuvdwdirect_
    (x,*n,atype,*nat,epsilon4,sigma2,*tblno,xmax,*periodicflag,*natchangeflag,force,pot,
     numex,natex);
}

void
computeGold2(float* C, const double* A, const double* B, unsigned int num_a, double xmax)
{

  double dn2, adn2, tmp2;
  double dx, dy, dz;
  double f, fx, fy, fz;
  double l2 = 0.5e0 * xmax;
  double exclude_radius2 = 0.1;

  for (unsigned int i = 0; i < num_a; ++i){
    fx = 0.e0;
    fy = 0.e0;
    fz = 0.e0;
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
	adn2 = 1.e0 / dn2;
	tmp2 = adn2 * adn2 * adn2;
	f = 24.e0 * adn2 * tmp2 * (2.e0 * tmp2 - 1.e0);
	fx += f * dx;
	fy += f * dy;
	fz += f * dz;
	
      }
    }
    C[i*3]   = (float)fx;
    C[i*3+1] = (float)fy;
    C[i*3+2] = (float)fz;
  }
}

void
computeGold2_d(double* C, const double* A, const double* B, unsigned int num_a, double xmax)
{

  double dn2, tmp2;
  double dx, dy, dz;
  double f, fx, fy, fz;
  double l2 = 0.5e0 * xmax;
  //double exclude_radius2 = 0.01e0;
  double cutoff_radius2 = 9.e0;

  for (unsigned int i = 0; i < num_a; ++i){
    fx = 0.e0;
    fy = 0.e0;
    fz = 0.e0;
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
	tmp2 = 2.e0 / dn2 / dn2 / dn2 / dn2 / dn2 / dn2 / dn2
	  - 1.e0 / dn2 / dn2 / dn2 / dn2;
	f = tmp2 * 24.e0;
	fx += f * dx;
	fy += f * dy;
	fz += f * dz;

	
      }
      /*
      dn2 = (dx * dx + dy * dy + dz * dz) * 1;
      if (dn2 > exclude_radius2 && dn2 < cutoff_radius2){
	adn2 = 1.e0 / dn2;
	tmp2 = adn2 * adn2 * adn2;
	f = 24.e0 * adn2 * tmp2 * (2.e0 * tmp2 - 1.e0);
	fx += f * dx;
	fy += f * dy;
	fz += f * dz;
	
	}*/
    }
    C[i*3]   = fx;
    C[i*3+1] = fy;
    C[i*3+2] = fz;
	if (fz > 100.e0)
	  printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  }
}
void
computeGold2_d2(double* C, const double* A, const double* B, unsigned int num_a, double xmax)
{

  double dn2, tmp2;
  double dx, dy, dz;
  double f;
  double l2 = 0.5e0 * xmax;
  //double exclude_radius2 = 0.01e0;
  double cutoff_radius2 = 9.e0;

  for (unsigned int i = 0; i < num_a-1; ++i){
    for (unsigned int j = i+1; j < num_a; ++j) {

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
      if (dn2 < cutoff_radius2){
	tmp2 = 2.e0 / dn2 / dn2 / dn2 / dn2 / dn2 / dn2 / dn2
	  - 1.e0 / dn2 / dn2 / dn2 / dn2;
	f = tmp2 * 24.e0;
	C[i*3]   += f * dx;
	C[i*3+1] += f * dy;
	C[i*3+2] += f * dz;
	C[j*3]   -= f * dx;
	C[j*3+1] -= f * dy;
	C[j*3+2] -= f * dz;
	
      }
      /*
      dn2 = (dx * dx + dy * dy + dz * dz) * 1;
      if (dn2 > exclude_radius2 && dn2 < cutoff_radius2){
	adn2 = 1.e0 / dn2;
	tmp2 = adn2 * adn2 * adn2;
	f = 24.e0 * adn2 * tmp2 * (2.e0 * tmp2 - 1.e0);
	fx += f * dx;
	fy += f * dy;
	fz += f * dz;
	
	}*/
    }
  }
}
void
computeGold2_f2(float* C, const float* A, const float* B, unsigned int num_a, float xmax)
{

  float dn2, tmp2;
  float dx, dy, dz;
  float f;
  float l2 = 0.5e0 * xmax;
  //double exclude_radius2 = 0.01e0;
  float cutoff_radius2 = 9.e0;

  for (unsigned int i = 0; i < num_a-1; ++i){
    for (unsigned int j = i+1; j < num_a; ++j) {

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
      if (dn2 < cutoff_radius2){
	tmp2 = 2.e0 / dn2 / dn2 / dn2 / dn2 / dn2 / dn2 / dn2
	  - 1.e0 / dn2 / dn2 / dn2 / dn2;
	f = tmp2 * 24.e0;
	C[i*3]   += f * dx;
	C[i*3+1] += f * dy;
	C[i*3+2] += f * dz;
	C[j*3]   -= f * dx;
	C[j*3+1] -= f * dy;
	C[j*3+2] -= f * dz;
	
      }
      /*
      dn2 = (dx * dx + dy * dy + dz * dz) * 1;
      if (dn2 > exclude_radius2 && dn2 < cutoff_radius2){
	adn2 = 1.e0 / dn2;
	tmp2 = adn2 * adn2 * adn2;
	f = 24.e0 * adn2 * tmp2 * (2.e0 * tmp2 - 1.e0);
	fx += f * dx;
	fy += f * dy;
	fz += f * dz;
	
	}*/
    }
  }
}
