
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../domdec/sse_defs.h"

#ifndef __SSE2__ /* __SSE2__ */

void spread_charge_kernel4u_sse(const int *natomlist_in, const int *atomlist, const int *grid_atom,
				const int *fr1, const int *fr2, const int *fr3, 
				const int *fr1_orig_in, const int *fr2_orig_in,
				const int *fr3_orig_in,
				const int *xsize_in, const int *xysize_in,
				const float *charge, 
				const float *theta1, const float *theta2, const float *theta3,
				float *q) {
  printf("colfft_sse.c: calling spread_charge_kernel4u_sse when it is not compiled\n");
  exit(1);
}

void gather_force_kernel4u_sse(const int *istart_in, const int *iend_in, const int *grid_atom,
			       const int *fr1, const int *fr2, const int *fr3, 
			       const int *fr2_orig_in, const int *fr3_orig_in,
			       const int *xsize_in, const int *ysize_in,
			       const float *recip,
			       const float *charge, 
			       const float *theta1, const float *theta2, const float *theta3,
			       const float *dtheta1, const float *dtheta2, const float *dtheta3,
			       const float *q,
			       double *forcex, double *forcey, double *forcez) {
  printf("colfft_sse.c: calling gather_force_kernel4u_sse when it is not compiled\n");
  exit(1);
}

void fill_bspline4_kernel_sse_ps(const int *jstart_in, const int *jend_in, const int *grid_atom,
				 const float *x, const float *y, const float *z,
				 const float *charge,
				 const float *recip_in,
				 const int *ydim_in, const int *zdim_in,
				 const int *nfft1_in, const int *nfft2_in, const int *nfft3_in,
				 int *fr1, int *fr2, int *fr3,
				 float *theta1, float *theta2, float *theta3,
				 float *dtheta1, float *dtheta2, float *dtheta3
				 ,const int *grid2tx, const int *grid2ty, const int *grid2tz,
				 const int *grid2tx_lo, const int *grid2ty_lo,
				 const int *grid2tz_lo,
				 int *natom_thread, int *thread_id_list
			      ) {
  printf("colfft_sse.c: calling fill_bspline4_kernel_sse_ps when it is not compiled\n");
  exit(1);
}

void fill_bspline4_kernel_sse_pd(const int *jstart_in, const int *jend_in, const int *grid_atom,
				 const double *x, const double *y, const double *z, 
				 const double *charge,
				 const double *recip_in,
				 const int *ydim_in, const int *zdim_in,
				 const int *nfft1_in, const int *nfft2_in, const int *nfft3_in,
				 int *fr1, int *fr2, int *fr3,
				 double *theta1, double *theta2, double *theta3,
				 double *dtheta1, double *dtheta2, double *dtheta3,
				 const int *grid2tx, const int *grid2ty, const int *grid2tz,
				 const int *grid2tx_lo, const int *grid2ty_lo, 
				 const int *grid2tz_lo,
				 int *natom_thread, int *thread_id_list
			      ) {
  printf("colfft_sse.c: calling fill_bspline4_kernel_sse_pd when it is not compiled\n");
  exit(1);
}

#else

#include "../domdec/sse_utils.h"

//
// Calculates forder=4 bspline for 4 atoms
//
static inline void calc_bspline4_sp(const __m128 w, float *theta, float *dtheta) {
  const __m128 zero  = _mm_set1_ps(0.0);
  const __m128 third = _mm_set1_ps(1.0f/3.0f);
  const __m128 half  = _mm_set1_ps(0.5f);
  const __m128 one   = _mm_set1_ps(1.0f);
  const __m128 two   = _mm_set1_ps(2.0f);
  const __m128 three = _mm_set1_ps(3.0f);

  __m128 array1, array2, array3, array4;
  __m128 darray1, darray2, darray3, darray4;

  array4 = _mm_setzero_ps();
  array2 = w;
  array1 = _mm_sub_ps(one, w);

  //--- compute standard b-spline recursion
  array3 = _mm_mul_ps(half,_mm_mul_ps(w,w));
  array2 = _mm_mul_ps(half,_mm_add_ps(_mm_mul_ps(_mm_add_ps(w,one),array1),
					 _mm_mul_ps(_mm_sub_ps(two,w),w)));
  array1 = _mm_mul_ps(half,_mm_mul_ps(_mm_sub_ps(one,w),array1));
       
  //--- perform standard b-spline differentiation
  darray1 = _mm_sub_ps(zero, array1);
  darray2 = _mm_sub_ps(array1, array2);
  darray3 = _mm_sub_ps(array2, array3);
  darray4 = _mm_sub_ps(array3, array4);

  //--- one more recursion
  array4 = _mm_mul_ps(third,_mm_mul_ps(w,array3));
  array3 = _mm_mul_ps(third,_mm_add_ps(_mm_mul_ps(_mm_add_ps(w,one),array2),
					  _mm_mul_ps(_mm_sub_ps(three,w),array3)));
  array2 = _mm_mul_ps(third,_mm_add_ps(_mm_mul_ps(_mm_add_ps(w,two),array1),
				       _mm_mul_ps(_mm_sub_ps(two,w),array2)));
  array1 = _mm_mul_ps(third,_mm_mul_ps(_mm_sub_ps(one,w),array1));

  // array1 = theta1(atom1) theta1(atom2) theta1(atom3) theta1(atom4)
  // array2 = theta2(atom1) theta2(atom2) theta2(atom3) theta2(atom4)
  // array3 = theta3(atom1) theta3(atom2) theta3(atom3) theta3(atom4)
  // array4 = theta4(atom1) theta4(atom2) theta4(atom3) theta4(atom4)

  // Transpose array -matrix
  transpose_4x4_m128(array1, array2, array3, array4);

  // array1 = theta1(atom1) theta2(atom1) theta3(atom1) theta4(atom1)
  // array2 = theta1(atom2) theta2(atom2) theta3(atom2) theta4(atom2)
  // array3 = theta1(atom3) theta2(atom3) theta3(atom3) theta4(atom3)
  // array4 = theta1(atom4) theta2(atom4) theta3(atom4) theta4(atom4)

  // Store the result
  _mm_storeu_ps(&theta[0], array1);
  _mm_storeu_ps(&theta[4], array2);
  _mm_storeu_ps(&theta[8], array3);
  _mm_storeu_ps(&theta[12], array4);

  transpose_4x4_m128(darray1, darray2, darray3, darray4);

  _mm_storeu_ps(&dtheta[0], darray1);
  _mm_storeu_ps(&dtheta[4], darray2);
  _mm_storeu_ps(&dtheta[8], darray3);
  _mm_storeu_ps(&dtheta[12], darray4);
}

//
// Calculates forder=4 bspline for 2 atoms
//
static inline void calc_bspline4_dp(__m128d w, double *theta, double *dtheta) {
  const __m128d zero  = _mm_set1_pd(0.0);
  const __m128d third = _mm_set1_pd(1.0/3.0);
  const __m128d half  = _mm_set1_pd(0.5);
  const __m128d one   = _mm_set1_pd(1.0);
  const __m128d two   = _mm_set1_pd(2.0);
  const __m128d three = _mm_set1_pd(3.0);

  __m128d array1, array2, array3, array4;
  __m128d darray1, darray2, darray3, darray4;

  array4 = _mm_setzero_pd();
  array2 = w;
  array1 = _mm_sub_pd(one, w);

  //--- compute standard b-spline recursion
  array3 = _mm_mul_pd(half,_mm_mul_pd(w,w));
  array2 = _mm_mul_pd(half,_mm_add_pd(_mm_mul_pd(_mm_add_pd(w,one),array1),
					 _mm_mul_pd(_mm_sub_pd(two,w),w)));
  array1 = _mm_mul_pd(half,_mm_mul_pd(_mm_sub_pd(one,w),array1));
       
  //--- perform standard b-spline differentiation
  darray1 = _mm_sub_pd(zero, array1);
  darray2 = _mm_sub_pd(array1, array2);
  darray3 = _mm_sub_pd(array2, array3);
  darray4 = _mm_sub_pd(array3, array4);

  //--- one more recursion
  array4 = _mm_mul_pd(third,_mm_mul_pd(w,array3));
  array3 = _mm_mul_pd(third,_mm_add_pd(_mm_mul_pd(_mm_add_pd(w,one),array2),
					  _mm_mul_pd(_mm_sub_pd(three,w),array3)));
  array2 = _mm_mul_pd(third,_mm_add_pd(_mm_mul_pd(_mm_add_pd(w,two),array1),
				       _mm_mul_pd(_mm_sub_pd(two,w),array2)));
  array1 = _mm_mul_pd(third,_mm_mul_pd(_mm_sub_pd(one,w),array1));

  // array1 = theta1(atom1) theta1(atom2)
  // array2 = theta2(atom1) theta2(atom2)
  // array3 = theta3(atom1) theta3(atom2)
  // array4 = theta4(atom1) theta4(atom2)

  // array1 = theta1(atom1) theta2(atom1)
  // array2 = theta3(atom1) theta4(atom1)
  // array3 = theta1(atom2) theta2(atom2)
  // array4 = theta3(atom2) theta4(atom2)
  // Transpose array -matrix
  transpose_4x2_m128d(array1, array2, array3, array4);

  // Store the result
  _mm_storeu_pd(&theta[0], array1);
  _mm_storeu_pd(&theta[2], array2);
  _mm_storeu_pd(&theta[4], array3);
  _mm_storeu_pd(&theta[6], array4);

  transpose_4x2_m128d(darray1, darray2, darray3, darray4);

  _mm_storeu_pd(&dtheta[0], darray1);
  _mm_storeu_pd(&dtheta[2], darray2);
  _mm_storeu_pd(&dtheta[4], darray3);
  _mm_storeu_pd(&dtheta[6], darray4);
}

//
// Spreads charge on 4x4x4 block using unaligned load and store
//
void spread_charge_kernel4u_sse(const int *natomlist_in, const int *atomlist, const int *grid_atom,
				const int *fr1, const int *fr2, const int *fr3, 
				const int *fr1_orig_in, const int *fr2_orig_in,
				const int *fr3_orig_in,
				const int *xsize_in, const int *xysize_in,
				const float *charge, 
				const float *theta1, const float *theta2, const float *theta3,
				float *q) {
  int fr1_orig = *fr1_orig_in;
  int fr2_orig = *fr2_orig_in;
  int fr3_orig = *fr3_orig_in;
  int xsize = *xsize_in;
  int xysize = *xysize_in;
  int natomlist = *natomlist_in;
  
  int j;
  for (j=0;j < natomlist;j++) {
    int i = atomlist[j] - 1;

    float chargef = charge[grid_atom[i]-1];
    //    if (chargef != 0.0f) {

      __m128 chargev = _mm_set1_ps(chargef);

      __m128 theta1v = _mm_load_ps(&theta1[4*i]);
      __m128 theta2v0 = _mm_load1_ps(&theta2[4*i]);
      __m128 theta2v1 = _mm_load1_ps(&theta2[4*i+1]);
      __m128 theta2v2 = _mm_load1_ps(&theta2[4*i+2]);
      __m128 theta2v3 = _mm_load1_ps(&theta2[4*i+3]);
      __m128 theta3v = _mm_mul_ps(_mm_load_ps(&theta3[4*i]), chargev);
      
      int fr1i = fr1[i];
      int fr2i = fr2[i];
      int fr3i = fr3[i];

      int ind = (fr1i - fr1_orig) + (fr2i - fr2_orig)*xsize + (fr3i - fr3_orig)*xysize;
      
      int ith3;
      for(ith3=0;ith3 < 4;ith3++) {
	__m128 theta3v0 = set1_m128(theta3v);
	__m128 theta1_theta3 = _mm_mul_ps(theta3v0, theta1v);
	
	// Load charge grid values
	__m128 q0 = _mm_loadu_ps(&q[ind]);
	__m128 q1 = _mm_loadu_ps(&q[ind+xsize]);
	__m128 q2 = _mm_loadu_ps(&q[ind+xsize*2]);
	__m128 q3 = _mm_loadu_ps(&q[ind+xsize*3]);
	
	q0 = _mm_add_ps(q0, _mm_mul_ps(theta1_theta3, theta2v0));
	q1 = _mm_add_ps(q1, _mm_mul_ps(theta1_theta3, theta2v1));
	q2 = _mm_add_ps(q2, _mm_mul_ps(theta1_theta3, theta2v2));
	q3 = _mm_add_ps(q3, _mm_mul_ps(theta1_theta3, theta2v3));
	
	// Store charge grid values
	_mm_storeu_ps(&q[ind], q0);
	_mm_storeu_ps(&q[ind+xsize], q1);
	_mm_storeu_ps(&q[ind+xsize*2], q2);
	_mm_storeu_ps(&q[ind+xsize*3], q3);
	
	ind += xysize;
	theta3v = sr_m128_4(theta3v);
      } // for(ith3=0;ith3 < 4;ith3++)
      //    } // if (chargef != 0.0f)
  } // for (j=0;j < natomlist;j++)

  return;
}

//
// Unaligned force gather kernel for forder=4
//
void gather_force_kernel4u_sse(const int *istart_in, const int *iend_in, const int *grid_atom,
			       const int *fr1, const int *fr2, const int *fr3, 
			       const int *fr2_orig_in, const int *fr3_orig_in,
			       const int *xsize_in, const int *ysize_in,
			       const float *recip,
			       const float *charge, 
			       const float *theta1, const float *theta2, const float *theta3,
			       const float *dtheta1, const float *dtheta2, const float *dtheta3,
			       const float *q,
			       double *forcex, double *forcey, double *forcez) {
  float dforcef[4];
  __m128 recip1 = _mm_setr_ps(recip[0], recip[1], recip[2], 0.0f);
  __m128 recip2 = _mm_setr_ps(recip[3], recip[4], recip[5], 0.0f);
  __m128 recip3 = _mm_setr_ps(recip[6], recip[7], recip[8], 0.0f);

  int istart = *istart_in - 1;
  int iend = *iend_in - 1;
  int fr2_orig = *fr2_orig_in;
  int fr3_orig = *fr3_orig_in;
  int xsize = *xsize_in;
  int xysize = (*xsize_in)*(*ysize_in);
  int ind_add2 = xysize - 4*xsize;

  int i;
  for (i=istart;i <= iend;i++) {
    int j = grid_atom[i] - 1;
    float chargef = charge[j];

    //    if (chargef != 0.0f) {

      int fr1i = fr1[i];
      int fr2i = fr2[i];
      int fr3i = fr3[i];

      __m128 f1 = _mm_setzero_ps();
      __m128 f2 = _mm_setzero_ps();
      __m128 f3 = _mm_setzero_ps();

      __m128 theta1v  = _mm_load_ps(&theta1[4*i]);
      __m128 dtheta1v = _mm_load_ps(&dtheta1[4*i]);

      __m128 theta2v  = _mm_load_ps(&theta2[4*i]);
      __m128 dtheta2v = _mm_load_ps(&dtheta2[4*i]);

      __m128 theta3v  = _mm_load_ps(&theta3[4*i]);
      __m128 dtheta3v = _mm_load_ps(&dtheta3[4*i]);

      int ind = fr1i + (fr2i - fr2_orig)*xsize + (fr3i - fr3_orig)*xysize;

      int ith3;
      for (ith3=0;ith3 < 4;ith3++) {
	__m128 theta3v0  = set1_m128(theta3v);
	__m128 dtheta3v0 = set1_m128(dtheta3v);
	int ith2;
#pragma unroll
	for (ith2=0;ith2 < 4;ith2++) {
	  __m128 theta2v0  = set1_m128(theta2v);
	  __m128 dtheta2v0 = set1_m128(dtheta2v);

	  // Load charge grid values
	  __m128 q0 = _mm_loadu_ps(&q[ind]);
	  
	  __m128 q_dtheta1 = _mm_mul_ps(q0, dtheta1v);
	  __m128 q_theta1  = _mm_mul_ps(q0, theta1v);
	  
	  f1 = _mm_sub_ps(f1, _mm_mul_ps(q_dtheta1, _mm_mul_ps(theta2v0, theta3v0)));
	  f2 = _mm_sub_ps(f2, _mm_mul_ps(q_theta1, _mm_mul_ps(dtheta2v0, theta3v0)));
	  f3 = _mm_sub_ps(f3, _mm_mul_ps(q_theta1, _mm_mul_ps(theta2v0, dtheta3v0)));
	  
	  ind += xsize;

	  theta2v = src_m128(theta2v);
	  dtheta2v = src_m128(dtheta2v);
	}
	ind += ind_add2;
	theta3v = sr_m128_4(theta3v);
	dtheta3v = sr_m128_4(dtheta3v);
      }

      __m128 f1sum = sum_m128(f1);
      __m128 f2sum = sum_m128(f2);
      __m128 f3sum = sum_m128(f3);

      __m128 f1sum_recip1 = _mm_mul_ps(f1sum, recip1);
      __m128 f2sum_recip2 = _mm_mul_ps(f2sum, recip2);
      __m128 f3sum_recip3 = _mm_mul_ps(f3sum, recip3);

      // dforce = [dforcex][dforcey][dforcez][0]
      __m128 chargev = _mm_set1_ps(chargef);
      __m128 dforce = _mm_mul_ps(chargev, 
				 _mm_add_ps(f1sum_recip1, _mm_add_ps(f2sum_recip2, f3sum_recip3)));
    
      _mm_storeu_ps(dforcef, dforce);

      forcex[j] -= (double)dforcef[0];
      forcey[j] -= (double)dforcef[1];
      forcez[j] -= (double)dforcef[2];
      //    }
  }

  return;
}

/*
//
// Force gather kernel for forder=4, takes 4 atoms at the time. SLOWER THAN kernel4u above
//
void gather_force_kernel4x_sse(const int *istart_in, const int *iend_in, const int *grid_atom,
			       const int *fr1, const int *fr2, const int *fr3, 
			       const int *fr2_orig_in, const int *fr3_orig_in,
			       const int *xsize_in, const int *ysize_in,
			       const float *recip,
			       const float *charge, 
			       const float *theta1, const float *theta2, const float *theta3,
			       const float *dtheta1, const float *dtheta2, const float *dtheta3,
			       const float *q,
			       double *forcex, double *forcey, double *forcez) {
  float dforcexf[4], dforceyf[4], dforcezf[4];

  __m128 recip1 = _mm_set1_ps(recip[0]);
  __m128 recip2 = _mm_set1_ps(recip[4]);
  __m128 recip3 = _mm_set1_ps(recip[8]);

  int istart = *istart_in - 1;
  int iend = *iend_in - 1;
  int fr2_orig = *fr2_orig_in;
  int fr3_orig = *fr3_orig_in;
  int xsize = *xsize_in;
  int xysize = (*xsize_in)*(*ysize_in);
  int ind_add1 = xsize - 4;
  int ind_add2 = xysize - 4*xsize;

  int n_round4 = ((iend - istart + 1)/4)*4;
  int iend_round4 = istart + n_round4 - 1;

  int i;
  for (i=istart;i <= iend_round4;i+=4) {

    __m128 f1 = _mm_setzero_ps();
    __m128 f2 = _mm_setzero_ps();
    __m128 f3 = _mm_setzero_ps();
      
    int ind0 = fr1[i]   + (fr2[i]   - fr2_orig)*xsize + (fr3[i]   - fr3_orig)*xysize;
    int ind1 = fr1[i+1] + (fr2[i+1] - fr2_orig)*xsize + (fr3[i+1] - fr3_orig)*xysize;
    int ind2 = fr1[i+2] + (fr2[i+2] - fr2_orig)*xsize + (fr3[i+2] - fr3_orig)*xysize;
    int ind3 = fr1[i+3] + (fr2[i+3] - fr2_orig)*xsize + (fr3[i+3] - fr3_orig)*xysize;

    int ind = 0;

    int ith3;
    for (ith3=0;ith3 < 4;ith3++) {
      __m128 theta3v = load_4sp(&theta3[4*i+ith3], &theta3[4*i+4+ith3], 
				&theta3[4*i+8+ith3], &theta3[4*i+12+ith3]);
      __m128 dtheta3v = load_4sp(&dtheta3[4*i+ith3], &dtheta3[4*i+4+ith3], 
				 &dtheta3[4*i+8+ith3], &dtheta3[4*i+12+ith3]);
      int ith2;
      for (ith2=0;ith2 < 4;ith2++) {
	__m128 theta2v = load_4sp(&theta2[4*i+ith2], &theta2[4*i+4+ith2], 
				  &theta2[4*i+8+ith2], &theta2[4*i+12+ith2]);
	__m128 dtheta2v = load_4sp(&dtheta2[4*i+ith2], &dtheta2[4*i+4+ith2], 
				   &dtheta2[4*i+8+ith2], &dtheta2[4*i+12+ith2]);
	int ith1;
#pragma unroll
	for (ith1=0;ith1 < 4;ith1++) {
	  __m128 theta1v = load_4sp(&theta1[4*i+ith1], &theta1[4*i+4+ith1], 
				    &theta1[4*i+8+ith1], &theta1[4*i+12+ith1]);

	  __m128 dtheta1v = load_4sp(&dtheta1[4*i+ith1], &dtheta1[4*i+4+ith1], 
				     &dtheta1[4*i+8+ith1], &dtheta1[4*i+12+ith1]);

	  __m128 qv = load_4sp(&q[ind0+ind], &q[ind1+ind], &q[ind2+ind], &q[ind3+ind]);
	  
	  __m128 q_dtheta1 = _mm_mul_ps(qv, dtheta1v);
	  __m128 q_theta1  = _mm_mul_ps(qv, theta1v);
	  
	  f1 = _mm_sub_ps(f1, _mm_mul_ps(q_dtheta1, _mm_mul_ps(theta2v, theta3v)));
	  f2 = _mm_sub_ps(f2, _mm_mul_ps(q_theta1, _mm_mul_ps(dtheta2v, theta3v)));
	  f3 = _mm_sub_ps(f3, _mm_mul_ps(q_theta1, _mm_mul_ps(theta2v, dtheta3v)));

	  ind++;
	}
	ind += ind_add1;
      }
      ind += ind_add2;
    }

    // f1 = [f1[i]][f1[i+1]][f1[i+2]][f1[i+3]]

    int j0 = grid_atom[i]   - 1;
    int j1 = grid_atom[i+1] - 1;
    int j2 = grid_atom[i+2] - 1;
    int j3 = grid_atom[i+3] - 1;

    __m128 chargev = load_4sp(&charge[j0], &charge[j1], &charge[j2], &charge[j3]);

    __m128 dforcex = _mm_mul_ps(chargev, _mm_mul_ps(f1, recip1));
    __m128 dforcey = _mm_mul_ps(chargev, _mm_mul_ps(f2, recip2));
    __m128 dforcez = _mm_mul_ps(chargev, _mm_mul_ps(f3, recip3));
    
    _mm_storeu_ps(dforcexf, dforcex);
    _mm_storeu_ps(dforceyf, dforcey);
    _mm_storeu_ps(dforcezf, dforcez);

    forcex[j0] -= (double)dforcexf[0];
    forcey[j0] -= (double)dforceyf[0];
    forcez[j0] -= (double)dforcezf[0];

    forcex[j1] -= (double)dforcexf[1];
    forcey[j1] -= (double)dforceyf[1];
    forcez[j1] -= (double)dforcezf[1];

    forcex[j2] -= (double)dforcexf[2];
    forcey[j2] -= (double)dforceyf[2];
    forcez[j2] -= (double)dforcezf[2];

    forcex[j3] -= (double)dforcexf[3];
    forcey[j3] -= (double)dforceyf[3];
    forcez[j3] -= (double)dforcezf[3];

  }

  recip1 = _mm_setr_ps(recip[0], recip[1], recip[2], 0.0f);
  recip2 = _mm_setr_ps(recip[3], recip[4], recip[5], 0.0f);
  recip3 = _mm_setr_ps(recip[6], recip[7], recip[8], 0.0f);

  for (;i <= iend;i++) {
    int j = grid_atom[i] - 1;
    float chargef = charge[j];

    int fr1i = fr1[i];
    int fr2i = fr2[i];
    int fr3i = fr3[i];
    
    __m128 f1 = _mm_setzero_ps();
    __m128 f2 = _mm_setzero_ps();
    __m128 f3 = _mm_setzero_ps();
      
    __m128 theta1v  = _mm_load_ps(&theta1[4*i]);
    __m128 dtheta1v = _mm_load_ps(&dtheta1[4*i]);

    __m128 theta2v  = _mm_load_ps(&theta2[4*i]);
    __m128 dtheta2v = _mm_load_ps(&dtheta2[4*i]);

    __m128 theta3v  = _mm_load_ps(&theta3[4*i]);
    __m128 dtheta3v = _mm_load_ps(&dtheta3[4*i]);

    int ind = fr1i + (fr2i - fr2_orig)*xsize + (fr3i - fr3_orig)*xysize;

    int ith3;
    for (ith3=0;ith3 < 4;ith3++) {
      __m128 theta3v0  = set1_m128(theta3v);
      __m128 dtheta3v0 = set1_m128(dtheta3v);
      int ith2;
#pragma unroll
      for (ith2=0;ith2 < 4;ith2++) {
	__m128 theta2v0  = set1_m128(theta2v);
	__m128 dtheta2v0 = set1_m128(dtheta2v);

	// Load charge grid values
	__m128 q0 = _mm_loadu_ps(&q[ind]);
	  
	__m128 q_dtheta1 = _mm_mul_ps(q0, dtheta1v);
	__m128 q_theta1  = _mm_mul_ps(q0, theta1v);
	  
	f1 = _mm_sub_ps(f1, _mm_mul_ps(q_dtheta1, _mm_mul_ps(theta2v0, theta3v0)));
	f2 = _mm_sub_ps(f2, _mm_mul_ps(q_theta1, _mm_mul_ps(dtheta2v0, theta3v0)));
	f3 = _mm_sub_ps(f3, _mm_mul_ps(q_theta1, _mm_mul_ps(theta2v0, dtheta3v0)));
	  
	ind += xsize;

	theta2v = src_m128(theta2v);
	dtheta2v = src_m128(dtheta2v);
      }
      ind += ind_add2;
      theta3v = sr_m128_4(theta3v);
      dtheta3v = sr_m128_4(dtheta3v);
    }

    __m128 f1sum = sum_m128(f1);
    __m128 f2sum = sum_m128(f2);
    __m128 f3sum = sum_m128(f3);

    __m128 f1sum_recip1 = _mm_mul_ps(f1sum, recip1);
    __m128 f2sum_recip2 = _mm_mul_ps(f2sum, recip2);
    __m128 f3sum_recip3 = _mm_mul_ps(f3sum, recip3);

    // dforce = [dforcex][dforcey][dforcez][0]
    __m128 chargev = _mm_set1_ps(chargef);
    __m128 dforce = _mm_mul_ps(chargev, 
			       _mm_add_ps(f1sum_recip1, _mm_add_ps(f2sum_recip2, f3sum_recip3)));
    
    _mm_storeu_ps(dforcexf, dforce);

    forcex[j] -= (double)dforcexf[0];
    forcey[j] -= (double)dforcexf[1];
    forcez[j] -= (double)dforcexf[2];

  }

  return;
}
*/

__m128 mask0[4], mask1[4];

void set_mask() {
  mask0[0] = _mm_castsi128_ps(_mm_setr_epi32(-1,-1,-1,-1));
  mask0[1] = _mm_castsi128_ps(_mm_setr_epi32( 0,-1,-1,-1));
  mask0[2] = _mm_castsi128_ps(_mm_setr_epi32( 0, 0,-1,-1));
  mask0[3] = _mm_castsi128_ps(_mm_setr_epi32( 0, 0, 0,-1));

  mask1[0] = _mm_castsi128_ps(_mm_setr_epi32( 0, 0, 0, 0));
  mask1[1] = _mm_castsi128_ps(_mm_setr_epi32(-1, 0, 0, 0));
  mask1[2] = _mm_castsi128_ps(_mm_setr_epi32(-1,-1, 0, 0));
  mask1[3] = _mm_castsi128_ps(_mm_setr_epi32(-1,-1,-1, 0));
}

/*
//
// Aligned force gather kernel for forder=4, SLOWER than the unaligned version!
//
void gather_force_kernel4a_sse(const int *istart_in, const int *iend_in, const int *grid_atom,
			       const int *fr1, const int *fr2, const int *fr3, 
			       const int *fr2_orig_in, const int *fr3_orig_in,
			       const int *xsize_in, const int *ysize_in,
			       const float *recip,
			       const float *charge, 
			       const float *theta1, const float *theta2, const float *theta3,
			       const float *dtheta1, const float *dtheta2, const float *dtheta3,
			       const float *q,
			       double *forcex, double *forcey, double *forcez) {
  float dforcef[4];
  __m128 recip1 = _mm_setr_ps(recip[0], recip[1], recip[2], 0.0f);
  __m128 recip2 = _mm_setr_ps(recip[3], recip[4], recip[5], 0.0f);
  __m128 recip3 = _mm_setr_ps(recip[6], recip[7], recip[8], 0.0f);

  int istart = *istart_in - 1;
  int iend = *iend_in - 1;
  int fr2_orig = *fr2_orig_in;
  int fr3_orig = *fr3_orig_in;
  int xsize = *xsize_in;
  int xysize = (*xsize_in)*(*ysize_in);
  int ind_add2 = xysize - 4*xsize;

  set_mask();

  int i;
  for (i=istart;i <= iend;i++) {
    int j = grid_atom[i] - 1;
    float chargef = charge[j];

    //    if (chargef != 0.0f) {

      int fr1i = fr1[i];
      int fr2i = fr2[i];
      int fr3i = fr3[i];

      __m128 f1 = _mm_setzero_ps();
      __m128 f2 = _mm_setzero_ps();
      __m128 f3 = _mm_setzero_ps();

      int ind = fr1i + (fr2i - fr2_orig)*xsize + (fr3i - fr3_orig)*xysize;
      int ind_offset = ind & 3;
      ind -= ind_offset;

      __m128 theta1v0  = _mm_loadu_ps(&theta1[4*i-ind_offset]);
      __m128 dtheta1v0 = _mm_loadu_ps(&dtheta1[4*i-ind_offset]);
      __m128 theta1v1  = _mm_loadu_ps(&theta1[4*i-ind_offset+4]);
      __m128 dtheta1v1 = _mm_loadu_ps(&dtheta1[4*i-ind_offset+4]);
      theta1v0  = _mm_and_ps(theta1v0,  mask0[ind_offset]);
      dtheta1v0 = _mm_and_ps(dtheta1v0, mask0[ind_offset]);
      theta1v1  = _mm_and_ps(theta1v1,  mask1[ind_offset]);
      dtheta1v1 = _mm_and_ps(dtheta1v1, mask1[ind_offset]);

      __m128 theta2v  = _mm_load_ps(&theta2[4*i]);
      __m128 dtheta2v = _mm_load_ps(&dtheta2[4*i]);

      __m128 theta3v  = _mm_load_ps(&theta3[4*i]);
      __m128 dtheta3v = _mm_load_ps(&dtheta3[4*i]);

      int ith3;
      for (ith3=0;ith3 < 4;ith3++) {
	__m128 theta3v0  = set1_m128(theta3v);
	__m128 dtheta3v0 = set1_m128(dtheta3v);
	int ith2;
#pragma unroll
	for (ith2=0;ith2 < 4;ith2++) {
	  __m128 theta2v0  = set1_m128(theta2v);
	  __m128 dtheta2v0 = set1_m128(dtheta2v);
	  
	  // Load charge grid values
	  __m128 q0 = _mm_load_ps(&q[ind]);
	  __m128 q1 = _mm_load_ps(&q[ind + 4]);

	  __m128 q_dtheta1 = _mm_add_ps(_mm_mul_ps(q0, dtheta1v0), _mm_mul_ps(q1, dtheta1v1));
	  __m128 q_theta1  = _mm_add_ps(_mm_mul_ps(q0, theta1v0), _mm_mul_ps(q1, theta1v1));
	  
	  f1 = _mm_sub_ps(f1, _mm_mul_ps(q_dtheta1, _mm_mul_ps(theta2v0, theta3v0)));
	  f2 = _mm_sub_ps(f2, _mm_mul_ps(q_theta1, _mm_mul_ps(dtheta2v0, theta3v0)));
	  f3 = _mm_sub_ps(f3, _mm_mul_ps(q_theta1, _mm_mul_ps(theta2v0, dtheta3v0)));
	  
	  ind += xsize;
	  
	  theta2v = src_m128(theta2v);
	  dtheta2v = src_m128(dtheta2v);
	}
	ind += ind_add2;
	theta3v = sr_m128_4(theta3v);
	dtheta3v = sr_m128_4(dtheta3v);
      }

      __m128 f1sum = sum_m128(f1);
      __m128 f2sum = sum_m128(f2);
      __m128 f3sum = sum_m128(f3);

      __m128 f1sum_recip1 = _mm_mul_ps(f1sum, recip1);
      __m128 f2sum_recip2 = _mm_mul_ps(f2sum, recip2);
      __m128 f3sum_recip3 = _mm_mul_ps(f3sum, recip3);

      // dforce = [dforcex][dforcey][dforcez][0]
      __m128 chargev = _mm_set1_ps(chargef);
      __m128 dforce = _mm_mul_ps(chargev, 
				 _mm_add_ps(f1sum_recip1, _mm_add_ps(f2sum_recip2, f3sum_recip3)));
    
      _mm_storeu_ps(dforcef, dforce);

      forcex[j] -= (double)dforcef[0];
      forcey[j] -= (double)dforcef[1];
      forcez[j] -= (double)dforcef[2];
      //    }
  }

  return;
}
*/

void fill_bspline4_kernel_sse_ps(const int *jstart_in, const int *jend_in, const int *grid_atom,
				 const float *x, const float *y, const float *z,
				 const float *charge,
				 const float *recip_in,
				 const int *ydim_in, const int *zdim_in,
				 const int *nfft1_in, const int *nfft2_in, const int *nfft3_in,
				 int *fr1, int *fr2, int *fr3,
				 float *theta1, float *theta2, float *theta3,
				 float *dtheta1, float *dtheta2, float *dtheta3
				 ,const int *grid2tx, const int *grid2ty, const int *grid2tz,
				 const int *grid2tx_lo, const int *grid2ty_lo,
				 const int *grid2tz_lo,
				 int *natom_thread, int *thread_id_list
			      ) {
  const __m128 zero = _mm_set1_ps(0.0f);
  const __m128 half = _mm_set1_ps(0.5f);
  const __m128 two  = _mm_set1_ps(2.0f);
  const __m128 three = _mm_set1_ps(3.0f);
  float theta1_tmp[4*4], theta2_tmp[4*4], theta3_tmp[4*4];
  float dtheta1_tmp[4*4], dtheta2_tmp[4*4], dtheta3_tmp[4*4];

  if (*jend_in < *jstart_in) return;

  __m128 ydim = _mm_set1_ps(*ydim_in);
  __m128 zdim = _mm_set1_ps(*zdim_in);

  __m128 recip_11 = _mm_set1_ps(recip_in[0]);
  __m128 recip_22 = _mm_set1_ps(recip_in[1]);
  __m128 recip_33 = _mm_set1_ps(recip_in[2]);

  __m128 nfft1 = _mm_set1_ps((float)*nfft1_in);
  __m128 nfft2 = _mm_set1_ps((float)*nfft2_in);
  __m128 nfft3 = _mm_set1_ps((float)*nfft3_in);

  int jstart = *jstart_in - 1;
  int jend = *jend_in - 1;
  int n_round4 = ((jend - jstart + 1)/4)*4;
  int jend_round4 = jstart + n_round4 - 1;

  int j;
  for (j=jstart;j <= jend_round4;j+=4) {

    int i0 = grid_atom[j]   - 1;
    int i1 = grid_atom[j+1] - 1;
    int i2 = grid_atom[j+2] - 1;
    int i3 = grid_atom[j+3] - 1;

    __m128 xi = load_4sp(&x[i0], &x[i1], &x[i2], &x[i3]);
    __m128 yi = load_4sp(&y[i0], &y[i1], &y[i2], &y[i3]);
    __m128 zi = load_4sp(&z[i0], &z[i1], &z[i2], &z[i3]);

    __m128 w;
    w = _mm_add_ps(_mm_mul_ps(xi,recip_11), two);
    __m128 fr1v = _mm_mul_ps(nfft1,_mm_sub_ps(w, _mm_sub_ps(round_m128(w), half)));
    
    w = _mm_add_ps(_mm_mul_ps(yi,recip_22), two);
    __m128 fr2v = _mm_mul_ps(nfft2,_mm_sub_ps(w, _mm_sub_ps(round_m128(w), half)));

    w = _mm_add_ps(_mm_mul_ps(zi,recip_33), two);
    __m128 fr3v = _mm_mul_ps(nfft3,_mm_sub_ps(w, _mm_sub_ps(round_m128(w), half)));

    __m128 fr1vi = floor_m128(fr1v);
    __m128 fr2vi = floor_m128(fr2v);
    __m128 fr3vi = floor_m128(fr3v);

    __m128 w1 = _mm_sub_ps(fr1v, fr1vi);
    __m128 w2 = _mm_sub_ps(fr2v, fr2vi);
    __m128 w3 = _mm_sub_ps(fr3v, fr3vi);

    // fr1i = 0...nfft1-1
    // fr2i = 0...nfft2-1
    // fr3i = 0...nfft3-1

    // Apply periodic boundaries
    // NOTE: we are doing the computation in floating point and rounding down to simulate
    //       integer arithmetic
    __m128 dfr2 = _mm_mul_ps(floor_m128(_mm_div_ps(_mm_add_ps(fr2vi, three), ydim)),ydim);
    __m128 dfr3 = _mm_mul_ps(floor_m128(_mm_div_ps(_mm_add_ps(fr3vi, three), zdim)),zdim);
    fr2vi = _mm_sub_ps(fr2vi, dfr2);
    fr3vi = _mm_sub_ps(fr3vi, dfr3);

    // For periodic systems:
    // fr1i = 0...nfft1-1
    // fr2i = -(forder-1)...nfft2-forder
    // fr3i = -(forder-1)...nfft3-forder
    // For non-periodic systems:
    // fr1i = 0...nfft1-1
    // fr2i = 0...nfft2-1
    // fr3i = 0...nfft3-1

    __m128i fr1i = _mm_cvttps_epi32(fr1vi);
    __m128i fr2i = _mm_cvttps_epi32(fr2vi);
    __m128i fr3i = _mm_cvttps_epi32(fr3vi);

    _mm_storeu_si128((__m128i *)&fr1[j], fr1i);
    _mm_storeu_si128((__m128i *)&fr2[j], fr2i);
    _mm_storeu_si128((__m128i *)&fr3[j], fr3i);

    int k;
#pragma unroll
    for (k=0;k < 4;k++) {
      int t1 = _mm_cvtsi128_si32(fr1i) - *grid2tx_lo;
      int t2 = _mm_cvtsi128_si32(fr2i) - *grid2ty_lo;
      int t3 = _mm_cvtsi128_si32(fr3i) - *grid2tz_lo;
      int thread_id = grid2tx[t1] + grid2ty[t2] + grid2tz[t3];
      thread_id_list[j+k] = thread_id;
      natom_thread[thread_id]++;
      fr1i = _mm_srli_si128(fr1i, 4);
      fr2i = _mm_srli_si128(fr2i, 4);
      fr3i = _mm_srli_si128(fr3i, 4);
    }

    calc_bspline4_sp(w1, &theta1[j*4], &dtheta1[j*4]);
    calc_bspline4_sp(w2, &theta2[j*4], &dtheta2[j*4]);
    calc_bspline4_sp(w3, &theta3[j*4], &dtheta3[j*4]);
  }

  int klen = jend - j + 1;

  if (klen > 0) {
    int i[4];

    int k;
    for (k=0;k < klen;k++) i[k] = grid_atom[j+k] - 1;
    for (;k < 4;k++) i[k] = i[0];

    __m128 xi = load_4sp(&x[i[0]], &x[i[1]], &x[i[2]], &x[i[3]]);
    __m128 yi = load_4sp(&y[i[0]], &y[i[1]], &y[i[2]], &y[i[3]]);
    __m128 zi = load_4sp(&z[i[0]], &z[i[1]], &z[i[2]], &z[i[3]]);

    __m128 w;
    w = _mm_add_ps(_mm_mul_ps(xi,recip_11), two);
    __m128 fr1v = _mm_mul_ps(nfft1,_mm_sub_ps(w, _mm_sub_ps(round_m128(w), half)));
    
    w = _mm_add_ps(_mm_mul_ps(yi,recip_22), two);
    __m128 fr2v = _mm_mul_ps(nfft2,_mm_sub_ps(w, _mm_sub_ps(round_m128(w), half)));
			     
    w = _mm_add_ps(_mm_mul_ps(zi,recip_33), two);
    __m128 fr3v = _mm_mul_ps(nfft3,_mm_sub_ps(w, _mm_sub_ps(round_m128(w), half)));

    __m128 fr1vi = floor_m128(fr1v);
    __m128 fr2vi = floor_m128(fr2v);
    __m128 fr3vi = floor_m128(fr3v);

    __m128 w1 = _mm_sub_ps(fr1v, fr1vi);
    __m128 w2 = _mm_sub_ps(fr2v, fr2vi);
    __m128 w3 = _mm_sub_ps(fr3v, fr3vi);

    // fr1i = 0...nfft1-1
    // fr2i = 0...nfft2-1
    // fr3i = 0...nfft3-1

    // Apply periodic boundaries
    // NOTE: we are doing the computation in floating point and rounding down to simulate
    //       integer arithmetic
    __m128 dfr2 = _mm_mul_ps(floor_m128(_mm_div_ps(_mm_add_ps(fr2vi, three), ydim)),ydim);
    __m128 dfr3 = _mm_mul_ps(floor_m128(_mm_div_ps(_mm_add_ps(fr3vi, three), zdim)),zdim);
    fr2vi = _mm_sub_ps(fr2vi, dfr2);
    fr3vi = _mm_sub_ps(fr3vi, dfr3);

    // For periodic systems:
    // fr1i = 0...nfft1-1
    // fr2i = -(forder-1)...nfft2-forder
    // fr3i = -(forder-1)...nfft3-forder
    // For non-periodic systems:
    // fr1i = 0...nfft1-1
    // fr2i = 0...nfft2-1
    // fr3i = 0...nfft3-1

    __m128i fr1i = _mm_cvttps_epi32(fr1vi);
    __m128i fr2i = _mm_cvttps_epi32(fr2vi);
    __m128i fr3i = _mm_cvttps_epi32(fr3vi);

    calc_bspline4_sp(w1, theta1_tmp, dtheta1_tmp);
    calc_bspline4_sp(w2, theta2_tmp, dtheta2_tmp);
    calc_bspline4_sp(w3, theta3_tmp, dtheta3_tmp);

    for (k=0;k < klen;k++) {
      int t1 = _mm_cvtsi128_si32(fr1i);
      int t2 = _mm_cvtsi128_si32(fr2i);
      int t3 = _mm_cvtsi128_si32(fr3i);
      fr1[j+k] = t1;
      fr2[j+k] = t2;
      fr3[j+k] = t3;
      int thread_id = grid2tx[t1 - *grid2tx_lo] + grid2ty[t2 - *grid2ty_lo] + 
	grid2tz[t3 - *grid2tz_lo];
      thread_id_list[j+k] = thread_id;
      natom_thread[thread_id]++;
      fr1i = _mm_srli_si128(fr1i, 4);
      fr2i = _mm_srli_si128(fr2i, 4);
      fr3i = _mm_srli_si128(fr3i, 4);
      _mm_store_ps(&theta1[j*4+k*4],_mm_loadu_ps(&theta1_tmp[k*4]));
      _mm_store_ps(&theta2[j*4+k*4],_mm_loadu_ps(&theta2_tmp[k*4]));
      _mm_store_ps(&theta3[j*4+k*4],_mm_loadu_ps(&theta3_tmp[k*4]));
      _mm_store_ps(&dtheta1[j*4+k*4],_mm_loadu_ps(&dtheta1_tmp[k*4]));
      _mm_store_ps(&dtheta2[j*4+k*4],_mm_loadu_ps(&dtheta2_tmp[k*4]));
      _mm_store_ps(&dtheta3[j*4+k*4],_mm_loadu_ps(&dtheta3_tmp[k*4]));
    }

  }

}

void fill_bspline4_kernel_sse_pd(const int *jstart_in, const int *jend_in, const int *grid_atom,
				 const double *x, const double *y, const double *z, 
				 const double *charge,
				 const double *recip_in,
				 const int *ydim_in, const int *zdim_in,
				 const int *nfft1_in, const int *nfft2_in, const int *nfft3_in,
				 int *fr1, int *fr2, int *fr3,
				 double *theta1, double *theta2, double *theta3,
				 double *dtheta1, double *dtheta2, double *dtheta3,
				 const int *grid2tx, const int *grid2ty, const int *grid2tz,
				 const int *grid2tx_lo, const int *grid2ty_lo, 
				 const int *grid2tz_lo,
				 int *natom_thread, int *thread_id_list
			      ) {
  const __m128d zero = _mm_set1_pd(0.0);
  const __m128d half = _mm_set1_pd(0.5);
  const __m128d two  = _mm_set1_pd(2.0);
  const __m128d three = _mm_set1_pd(3.0);
  double theta1_tmp[2*4], theta2_tmp[2*4], theta3_tmp[2*4];
  double dtheta1_tmp[2*4], dtheta2_tmp[2*4], dtheta3_tmp[2*4];

  if (*jend_in < *jstart_in) return;

  __m128d ydim = _mm_set1_pd((double)*ydim_in);
  __m128d zdim = _mm_set1_pd((double)*zdim_in);

  __m128d recip_11 = _mm_set1_pd(recip_in[0]);
  __m128d recip_22 = _mm_set1_pd(recip_in[1]);
  __m128d recip_33 = _mm_set1_pd(recip_in[2]);

  __m128d nfft1 = _mm_set1_pd((double)*nfft1_in);
  __m128d nfft2 = _mm_set1_pd((double)*nfft2_in);
  __m128d nfft3 = _mm_set1_pd((double)*nfft3_in);

  int jstart = *jstart_in - 1;
  int jend = *jend_in - 1;
  int n_round2 = ((jend - jstart + 1)/2)*2;
  int jend_round2 = jstart + n_round2 - 1;

  int j;
  for (j=jstart;j <= jend_round2;j+=2) {

    int i0 = grid_atom[j]   - 1;
    int i1 = grid_atom[j+1] - 1;

    __m128d xi = load_2dp(&x[i0], &x[i1]);
    __m128d yi = load_2dp(&y[i0], &y[i1]);
    __m128d zi = load_2dp(&z[i0], &z[i1]);

    __m128d w;
    w = _mm_add_pd(_mm_mul_pd(xi,recip_11), two);
    __m128d fr1v = _mm_mul_pd(nfft1,_mm_sub_pd(w, _mm_sub_pd(round_m128d(w), half)));
    
    w = _mm_add_pd(_mm_mul_pd(yi,recip_22), two);
    __m128d fr2v = _mm_mul_pd(nfft2,_mm_sub_pd(w, _mm_sub_pd(round_m128d(w), half)));
			     
    w = _mm_add_pd(_mm_mul_pd(zi,recip_33), two);
    __m128d fr3v = _mm_mul_pd(nfft3,_mm_sub_pd(w, _mm_sub_pd(round_m128d(w), half)));

    __m128d fr1vi = floor_m128d(fr1v);
    __m128d fr2vi = floor_m128d(fr2v);
    __m128d fr3vi = floor_m128d(fr3v);

    __m128d w1 = _mm_sub_pd(fr1v, fr1vi);
    __m128d w2 = _mm_sub_pd(fr2v, fr2vi);
    __m128d w3 = _mm_sub_pd(fr3v, fr3vi);

    // fr1i = 0...nfft1-1
    // fr2i = 0...nfft2-1
    // fr3i = 0...nfft3-1

    // Apply periodic boundaries
    // NOTE: we are doing the computation in floating point and rounding down to simulate
    //       integer arithmetic
    __m128d dfr2 = _mm_mul_pd(floor_m128d(_mm_div_pd(_mm_add_pd(fr2vi, three), ydim)),ydim);
    __m128d dfr3 = _mm_mul_pd(floor_m128d(_mm_div_pd(_mm_add_pd(fr3vi, three), zdim)),zdim);
    fr2vi = _mm_sub_pd(fr2vi, dfr2);
    fr3vi = _mm_sub_pd(fr3vi, dfr3);

    // For periodic systems:
    // fr1i = 0...nfft1-1
    // fr2i = -(forder-1)...nfft2-forder
    // fr3i = -(forder-1)...nfft3-forder
    // For non-periodic systems:
    // fr1i = 0...nfft1-1
    // fr2i = 0...nfft2-1
    // fr3i = 0...nfft3-1
    
    __m128i fr1i = _mm_cvttpd_epi32(fr1vi);
    __m128i fr2i = _mm_cvttpd_epi32(fr2vi);
    __m128i fr3i = _mm_cvttpd_epi32(fr3vi);

    _mm_storel_epi64((__m128i *)&fr1[j], fr1i);
    _mm_storel_epi64((__m128i *)&fr2[j], fr2i);
    _mm_storel_epi64((__m128i *)&fr3[j], fr3i);

    int k;
#pragma unroll
    for (k=0;k < 2;k++) {
      int t1 = _mm_cvtsi128_si32(fr1i) - *grid2tx_lo;
      int t2 = _mm_cvtsi128_si32(fr2i) - *grid2ty_lo;
      int t3 = _mm_cvtsi128_si32(fr3i) - *grid2tz_lo;
      int thread_id = grid2tx[t1] + grid2ty[t2] + grid2tz[t3];
      thread_id_list[j+k] = thread_id;
      natom_thread[thread_id]++;
      fr1i = _mm_srli_si128(fr1i, 4);
      fr2i = _mm_srli_si128(fr2i, 4);
      fr3i = _mm_srli_si128(fr3i, 4);
    }

    calc_bspline4_dp(w1, &theta1[j*4], &dtheta1[j*4]);
    calc_bspline4_dp(w2, &theta2[j*4], &dtheta2[j*4]);
    calc_bspline4_dp(w3, &theta3[j*4], &dtheta3[j*4]);
  }

  int klen = jend - j + 1;
  if (klen > 0) {
    int i = grid_atom[j] - 1;
    
    __m128d xi = load_2dp(&x[i], &x[i]);
    __m128d yi = load_2dp(&y[i], &y[i]);
    __m128d zi = load_2dp(&z[i], &z[i]);

    __m128d w;
    w = _mm_add_pd(_mm_mul_pd(xi,recip_11), two);
    __m128d fr1v = _mm_mul_pd(nfft1,_mm_sub_pd(w, _mm_sub_pd(round_m128d(w), half)));
    
    w = _mm_add_pd(_mm_mul_pd(yi,recip_22), two);
    __m128d fr2v = _mm_mul_pd(nfft2,_mm_sub_pd(w, _mm_sub_pd(round_m128d(w), half)));
			     
    w = _mm_add_pd(_mm_mul_pd(zi,recip_33), two);
    __m128d fr3v = _mm_mul_pd(nfft3,_mm_sub_pd(w, _mm_sub_pd(round_m128d(w), half)));

    __m128d fr1vi = floor_m128d(fr1v);
    __m128d fr2vi = floor_m128d(fr2v);
    __m128d fr3vi = floor_m128d(fr3v);

    __m128d w1 = _mm_sub_pd(fr1v, fr1vi);
    __m128d w2 = _mm_sub_pd(fr2v, fr2vi);
    __m128d w3 = _mm_sub_pd(fr3v, fr3vi);

    // fr1i = 0...nfft1-1
    // fr2i = 0...nfft2-1
    // fr3i = 0...nfft3-1

    // Apply periodic boundaries
    // NOTE: we are doing the computation in floating point and rounding down to simulate
    //       integer arithmetic
    __m128d dfr2 = _mm_mul_pd(floor_m128d(_mm_div_pd(_mm_add_pd(fr2vi, three), ydim)),ydim);
    __m128d dfr3 = _mm_mul_pd(floor_m128d(_mm_div_pd(_mm_add_pd(fr3vi, three), zdim)),zdim);
    fr2vi = _mm_sub_pd(fr2vi, dfr2);
    fr3vi = _mm_sub_pd(fr3vi, dfr3);

    // For periodic systems:
    // fr1i = 0...nfft1-1
    // fr2i = -(forder-1)...nfft2-forder
    // fr3i = -(forder-1)...nfft3-forder
    // For non-periodic systems:
    // fr1i = 0...nfft1-1
    // fr2i = 0...nfft2-1
    // fr3i = 0...nfft3-1

    __m128i fr1i = _mm_cvttpd_epi32(fr1vi);
    __m128i fr2i = _mm_cvttpd_epi32(fr2vi);
    __m128i fr3i = _mm_cvttpd_epi32(fr3vi);
    
    // theta1[0]...theta1[3] = atom1
    // theta1[4]...theta1[7] = atom2

    calc_bspline4_dp(w1, theta1_tmp, dtheta1_tmp);
    calc_bspline4_dp(w2, theta2_tmp, dtheta2_tmp);
    calc_bspline4_dp(w3, theta3_tmp, dtheta3_tmp);

    int t1 = _mm_cvtsi128_si32(fr1i);
    int t2 = _mm_cvtsi128_si32(fr2i);
    int t3 = _mm_cvtsi128_si32(fr3i);
    fr1[j] = t1;
    fr2[j] = t2;
    fr3[j] = t3;
    int thread_id = grid2tx[t1 - *grid2tx_lo] + grid2ty[t2 - *grid2ty_lo] + 
      grid2tz[t3 - *grid2tz_lo];
    thread_id_list[j] = thread_id;
    natom_thread[thread_id]++;

    _mm_store_pd(&theta1[j*4],  _mm_loadu_pd(&theta1_tmp[0]));
    _mm_store_pd(&theta1[j*4+2],_mm_loadu_pd(&theta1_tmp[2]));
    _mm_store_pd(&theta2[j*4],  _mm_loadu_pd(&theta2_tmp[0]));
    _mm_store_pd(&theta2[j*4+2],_mm_loadu_pd(&theta2_tmp[2]));
    _mm_store_pd(&theta3[j*4],  _mm_loadu_pd(&theta3_tmp[0]));
    _mm_store_pd(&theta3[j*4+2],_mm_loadu_pd(&theta3_tmp[2]));
    _mm_store_pd(&dtheta1[j*4],  _mm_loadu_pd(&dtheta1_tmp[0]));
    _mm_store_pd(&dtheta1[j*4+2],_mm_loadu_pd(&dtheta1_tmp[2]));
    _mm_store_pd(&dtheta2[j*4],  _mm_loadu_pd(&dtheta2_tmp[0]));
    _mm_store_pd(&dtheta2[j*4+2],_mm_loadu_pd(&dtheta2_tmp[2]));
    _mm_store_pd(&dtheta3[j*4],  _mm_loadu_pd(&dtheta3_tmp[0]));
    _mm_store_pd(&dtheta3[j*4+2],_mm_loadu_pd(&dtheta3_tmp[2]));

  }

}

#endif /* __SSE2__ */
