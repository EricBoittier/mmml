#include <iostream>
#include <cmath>
#include "enb_core_vec.h"
#include "sse_defs.h"

#ifdef __SSE2__
static enb_core_vec<double, __m128d, __m128d> enb_core_double_sse;
static enb_core_vec<float,  __m128,  __m128d> enb_core_single_sse;
#ifdef __AVX__
static enb_core_vec<double, __m256d, __m256d> enb_core_double_avx;
static enb_core_vec<float,  __m256,  __m256d> enb_core_single_avx;
#endif
#endif

//
// Calculates y = 1/sqrt(x)
//
extern "C" void invsqrt_approx(const float *x, float *y) {
#ifdef __SSE2__
  *y = enb_core_single_sse.invsqrt_approx(*x);
#else
  *y = 1.0f/sqrtf(*x);
#endif
}

extern "C" void enb_vv_charmm_dp_sse(int *ni, int *indi, int *indj, int *startj,
				     double *pftable, double *hinv,
				     xyzq_t<double> *xyzq, double *force, int *iscoord,
				     double *sforce, double *scoordtab,
				     double *coulpot, double *vdwpot,
				     double *c6OO, double *c12OO, double *c6OH,
				     double *c12OH, double *c6HH, double *c12HH,
#if KEY_LJPME==1
                     double *c6mult_OO, double *c6mult_OH, double *c6mult_HH,
#endif
                     double *qOO, double *qOH, double *qHH,
				     double *roff, double *ron, int *vdw_type) {
#ifdef __SSE2__
  enb_core_double_sse.calc_vv(*ni, indi, indj, startj,
			      pftable, *hinv,
			      xyzq, force, iscoord,
			      sforce, scoordtab,
			      coulpot, vdwpot, 
			      *c6OO, *c12OO, *c6OH, *c12OH, *c6HH, *c12HH,
#if KEY_LJPME==1
                  *c6mult_OO, *c6mult_OH, *c6mult_HH,
#endif
                  *qOO, *qOH, *qHH,
			      *roff, *ron, *vdw_type);
#endif
}

extern "C" void enb_uv_coul_vdw_dp_sse(int *ni, int *indi, int *indj, int *startj,
				       double *pftable, double *hinv,
				       int *vdwtype, double *vdwparam,
				       xyzq_t<double> *xyzq, double *force, int *iscoord,
				       double *sforce, double *scoordtab,
				       double *coulpot, double *vdwpot,
				       double *roff, double *ron, int *vdw_type) {
#ifdef __SSE2__
  enb_core_double_sse.calc_uv_coul_vdw(*ni, indi, indj, startj,
				       pftable, *hinv,
				       vdwtype, vdwparam,
				       xyzq, force, iscoord,
				       sforce, scoordtab,
				       coulpot, vdwpot, 
				       *roff, *ron, *vdw_type);
#endif
}

extern "C" void enb_uv_vdw_dp_sse(int *ni, int *indi, int *indj, int *startj,
				  double *pftable, double *hinv,
				  int *vdwtype, double *vdwparam,
				  xyzq_t<double> *xyzq, double *force, int *iscoord,
				  double *sforce, double *scoordtab,
				  double *vdwpot,
				  double *roff, double *ron, int *vdw_type) {
#ifdef __SSE2__
  enb_core_double_sse.calc_uv_vdw(*ni, indi, indj, startj,
				  pftable, *hinv,
				  vdwtype, vdwparam,
				  xyzq, force, iscoord,
				  sforce, scoordtab,
				  vdwpot, 
				  *roff, *ron, *vdw_type);
#endif
}

extern "C" void enb_uu_coul_vdw_dp_sse(int *ni, int *indi, int *indj, int *startj,
				       double *pftable, double *hinv,
				       int *vdwtype, double *vdwparam,
				       xyzq_t<double> *xyzq, double *force, int *iscoord,
				       double *sforce, double *scoordtab,
				       double *coulpot, double *vdwpot,
				       double *roff, double *ron, int *vdw_type) {
#ifdef __SSE2__
  enb_core_double_sse.calc_uu_coul_vdw(*ni, indi, indj, startj,
				       pftable, *hinv,
				       vdwtype, vdwparam,
				       xyzq, force, iscoord,
				       sforce, scoordtab,
				       coulpot, vdwpot, 
				       *roff, *ron, *vdw_type);
#endif
}

extern "C" void enb_uu_vdw_dp_sse(int *ni, int *indi, int *indj, int *startj,
				  double *pftable, double *hinv,
				  int *vdwtype, double *vdwparam,
				  xyzq_t<double> *xyzq, double *force, int *iscoord,
				  double *sforce, double *scoordtab,
				  double *vdwpot,
				  double *roff, double *ron, int *vdw_type) {
#ifdef __SSE2__
  enb_core_double_sse.calc_uu_vdw(*ni, indi, indj, startj,
				  pftable, *hinv,
				  vdwtype, vdwparam,
				  xyzq, force, iscoord,
				  sforce, scoordtab,
				  vdwpot, 
				  *roff, *ron, *vdw_type);
#endif
}

// ------------------------------- single precision ----------------------------------------

extern "C" void enb_vv_charmm_sp_sse(int *ni, int *indi, int *indj, int *startj,
				     float *pftable, double *hinv,
				     xyzq_t<float> *xyzq, double *force, int *iscoord,
				     double *sforce, float *scoordtab,
				     double *coulpot, double *vdwpot,
				     double *c6OO, double *c12OO, double *c6OH,
				     double *c12OH, double *c6HH, double *c12HH,
#if KEY_LJPME==1
                     double *c6mult_OO, double *c6mult_OH, double *c6mult_HH,
#endif
                     double *qOO, double *qOH, double *qHH,
				     double *roff, double *ron, int *vdw_type) {
#ifdef __SSE2__
  enb_core_single_sse.calc_vv(*ni, indi, indj, startj,
			      pftable, *hinv,
			      xyzq, force, iscoord,
			      sforce, scoordtab,
			      coulpot, vdwpot, 
			      *c6OO, *c12OO, *c6OH, *c12OH, *c6HH, *c12HH,
#if KEY_LJPME==1
                  *c6mult_OO, *c6mult_OH, *c6mult_HH,
#endif
                  *qOO, *qOH, *qHH,
			      *roff, *ron, *vdw_type);
#endif
}

extern "C" void enb_uv_coul_vdw_sp_sse(int *ni, int *indi, int *indj, int *startj,
				       float *pftable, double *hinv,
				       int *vdwtype, float *vdwparam,
				       xyzq_t<float> *xyzq, double *force, int *iscoord,
				       double *sforce, float *scoordtab,
				       double *coulpot, double *vdwpot,
				       double *roff, double *ron, int *vdw_type) {
#ifdef __SSE2__
  enb_core_single_sse.calc_uv_coul_vdw(*ni, indi, indj, startj,
				       pftable, *hinv,
				       vdwtype, vdwparam,
				       xyzq, force, iscoord,
				       sforce, scoordtab,
				       coulpot, vdwpot, 
				       *roff, *ron, *vdw_type);
#endif
}

extern "C" void enb_uv_vdw_sp_sse(int *ni, int *indi, int *indj, int *startj,
				  float *pftable, double *hinv,
				  int *vdwtype, float *vdwparam,
				  xyzq_t<float> *xyzq, double *force, int *iscoord,
				  double *sforce, float *scoordtab,
				  double *vdwpot,
				  double *roff, double *ron, int *vdw_type) {
#ifdef __SSE2__
  enb_core_single_sse.calc_uv_vdw(*ni, indi, indj, startj,
				  pftable, *hinv,
				  vdwtype, vdwparam,
				  xyzq, force, iscoord,
				  sforce, scoordtab,
				  vdwpot, 
				  *roff, *ron, *vdw_type);
#endif
}

extern "C" void enb_uu_coul_vdw_sp_sse(int *ni, int *indi, int *indj, int *startj,
				       float *pftable, double *hinv,
				       int *vdwtype, float *vdwparam,
				       xyzq_t<float> *xyzq, double *force, int *iscoord,
				       double *sforce, float *scoordtab,
				       double *coulpot, double *vdwpot,
				       double *roff, double *ron, int *vdw_type) {
#ifdef __SSE2__
  enb_core_single_sse.calc_uu_coul_vdw(*ni, indi, indj, startj,
				       pftable, *hinv,
				       vdwtype, vdwparam,
				       xyzq, force, iscoord,
				       sforce, scoordtab,
				       coulpot, vdwpot, 
				       *roff, *ron, *vdw_type);
#endif
}

extern "C" void enb_uu_vdw_sp_sse(int *ni, int *indi, int *indj, int *startj,
				  float *pftable, double *hinv,
				  int *vdwtype, float *vdwparam,
				  xyzq_t<float> *xyzq, double *force, int *iscoord,
				  double *sforce, float *scoordtab,
				  double *vdwpot,
				  double *roff, double *ron, int *vdw_type) {
#ifdef __SSE2__
  enb_core_single_sse.calc_uu_vdw(*ni, indi, indj, startj,
				  pftable, *hinv,
				  vdwtype, vdwparam,
				  xyzq, force, iscoord,
				  sforce, scoordtab,
				  vdwpot, 
				  *roff, *ron, *vdw_type);
#endif
}

//------------------------------------------------------------------------------------
//----------------------------- AVX --------------------------------------------------
//------------------------------------------------------------------------------------

extern "C" void enb_vv_charmm_dp_avx(int *ni, int *indi, int *indj, int *startj,
				     double *pftable, double *hinv,
				     xyzq_t<double> *xyzq, double *force, int *iscoord,
				     double *sforce, double *scoordtab,
				     double *coulpot, double *vdwpot,
				     double *c6OO, double *c12OO, double *c6OH,
				     double *c12OH, double *c6HH, double *c12HH,
#if KEY_LJPME==1
                     double *c6mult_OO, double *c6mult_OH, double *c6mult_HH,
#endif
                     double *qOO, double *qOH, double *qHH,
				     double *roff, double *ron, int *vdw_type) {
#ifdef __AVX__
  enb_core_double_avx.calc_vv(*ni, indi, indj, startj,
			      pftable, *hinv,
			      xyzq, force, iscoord,
			      sforce, scoordtab,
			      coulpot, vdwpot, 
			      *c6OO, *c12OO, *c6OH, *c12OH, *c6HH, *c12HH,
#if KEY_LJPME==1
                  *c6mult_OO, *c6mult_OH, *c6mult_HH,
#endif
                  *qOO, *qOH, *qHH,
			      *roff, *ron, *vdw_type);
#endif
}

extern "C" void enb_uv_coul_vdw_dp_avx(int *ni, int *indi, int *indj, int *startj,
				       double *pftable, double *hinv,
				       int *vdwtype, double *vdwparam,
				       xyzq_t<double> *xyzq, double *force, int *iscoord,
				       double *sforce, double *scoordtab,
				       double *coulpot, double *vdwpot,
				       double *roff, double *ron, int *vdw_type) {
#ifdef __AVX__
  enb_core_double_avx.calc_uv_coul_vdw(*ni, indi, indj, startj,
				       pftable, *hinv,
				       vdwtype, vdwparam,
				       xyzq, force, iscoord,
				       sforce, scoordtab,
				       coulpot, vdwpot, 
				       *roff, *ron, *vdw_type);
#endif
}

extern "C" void enb_uv_vdw_dp_avx(int *ni, int *indi, int *indj, int *startj,
				  double *pftable, double *hinv,
				  int *vdwtype, double *vdwparam,
				  xyzq_t<double> *xyzq, double *force, int *iscoord,
				  double *sforce, double *scoordtab,
				  double *vdwpot,
				  double *roff, double *ron, int *vdw_type) {
#ifdef __AVX__
  enb_core_double_avx.calc_uv_vdw(*ni, indi, indj, startj,
				  pftable, *hinv,
				  vdwtype, vdwparam,
				  xyzq, force, iscoord,
				  sforce, scoordtab,
				  vdwpot, 
				  *roff, *ron, *vdw_type);
#endif
}

extern "C" void enb_uu_coul_vdw_dp_avx(int *ni, int *indi, int *indj, int *startj,
				       double *pftable, double *hinv,
				       int *vdwtype, double *vdwparam,
				       xyzq_t<double> *xyzq, double *force, int *iscoord,
				       double *sforce, double *scoordtab,
				       double *coulpot, double *vdwpot,
				       double *roff, double *ron, int *vdw_type) {
#ifdef __AVX__
  enb_core_double_avx.calc_uu_coul_vdw(*ni, indi, indj, startj,
				       pftable, *hinv,
				       vdwtype, vdwparam,
				       xyzq, force, iscoord,
				       sforce, scoordtab,
				       coulpot, vdwpot, 
				       *roff, *ron, *vdw_type);
#endif
}

extern "C" void enb_uu_vdw_dp_avx(int *ni, int *indi, int *indj, int *startj,
				  double *pftable, double *hinv,
				  int *vdwtype, double *vdwparam,
				  xyzq_t<double> *xyzq, double *force, int *iscoord,
				  double *sforce, double *scoordtab,
				  double *vdwpot,
				  double *roff, double *ron, int *vdw_type) {
#ifdef __AVX__
  enb_core_double_avx.calc_uu_vdw(*ni, indi, indj, startj,
				  pftable, *hinv,
				  vdwtype, vdwparam,
				  xyzq, force, iscoord,
				  sforce, scoordtab,
				  vdwpot, 
				  *roff, *ron, *vdw_type);
#endif
}

// ------------------------------- single precision ----------------------------------------

extern "C" void enb_vv_charmm_sp_avx(int *ni, int *indi, int *indj, int *startj,
				     float *pftable, double *hinv,
				     xyzq_t<float> *xyzq, double *force, int *iscoord,
				     double *sforce, float *scoordtab,
				     double *coulpot, double *vdwpot,
				     double *c6OO, double *c12OO, double *c6OH,
				     double *c12OH, double *c6HH, double *c12HH,
#if KEY_LJPME==1
                     double *c6mult_OO, double *c6mult_OH, double *c6mult_HH,
#endif
                     double *qOO, double *qOH, double *qHH,
				     double *roff, double *ron, int *vdw_type) {
#ifdef __AVX__
  enb_core_single_avx.calc_vv(*ni, indi, indj, startj,
			      pftable, *hinv,
			      xyzq, force, iscoord,
			      sforce, scoordtab,
			      coulpot, vdwpot, 
			      *c6OO, *c12OO, *c6OH, *c12OH, *c6HH, *c12HH,
#if KEY_LJPME==1
                  *c6mult_OO, *c6mult_OH, *c6mult_HH,
#endif
                  *qOO, *qOH, *qHH,
			      *roff, *ron, *vdw_type);
#endif
}

extern "C" void enb_uv_coul_vdw_sp_avx(int *ni, int *indi, int *indj, int *startj,
				       float *pftable, double *hinv,
				       int *vdwtype, float *vdwparam,
				       xyzq_t<float> *xyzq, double *force, int *iscoord,
				       double *sforce, float *scoordtab,
				       double *coulpot, double *vdwpot,
				       double *roff, double *ron, int *vdw_type) {
#ifdef __AVX__
  enb_core_single_avx.calc_uv_coul_vdw(*ni, indi, indj, startj,
				       pftable, *hinv,
				       vdwtype, vdwparam,
				       xyzq, force, iscoord,
				       sforce, scoordtab,
				       coulpot, vdwpot, 
				       *roff, *ron, *vdw_type);
#endif
}

extern "C" void enb_uv_vdw_sp_avx(int *ni, int *indi, int *indj, int *startj,
				  float *pftable, double *hinv,
				  int *vdwtype, float *vdwparam,
				  xyzq_t<float> *xyzq, double *force, int *iscoord,
				  double *sforce, float *scoordtab,
				  double *vdwpot,
				  double *roff, double *ron, int *vdw_type) {
#ifdef __AVX__
  enb_core_single_avx.calc_uv_vdw(*ni, indi, indj, startj,
				  pftable, *hinv,
				  vdwtype, vdwparam,
				  xyzq, force, iscoord,
				  sforce, scoordtab,
				  vdwpot, 
				  *roff, *ron, *vdw_type);
#endif
}

extern "C" void enb_uu_coul_vdw_sp_avx(int *ni, int *indi, int *indj, int *startj,
				       float *pftable, double *hinv,
				       int *vdwtype, float *vdwparam,
				       xyzq_t<float> *xyzq, double *force, int *iscoord,
				       double *sforce, float *scoordtab,
				       double *coulpot, double *vdwpot,
				       double *roff, double *ron, int *vdw_type) {
#ifdef __AVX__
  enb_core_single_avx.calc_uu_coul_vdw(*ni, indi, indj, startj,
				       pftable, *hinv,
				       vdwtype, vdwparam,
				       xyzq, force, iscoord,
				       sforce, scoordtab,
				       coulpot, vdwpot, 
				       *roff, *ron, *vdw_type);
#endif
}

extern "C" void enb_uu_vdw_sp_avx(int *ni, int *indi, int *indj, int *startj,
				  float *pftable, double *hinv,
				  int *vdwtype, float *vdwparam,
				  xyzq_t<float> *xyzq, double *force, int *iscoord,
				  double *sforce, float *scoordtab,
				  double *vdwpot,
				  double *roff, double *ron, int *vdw_type) {
#ifdef __AVX__
  enb_core_single_avx.calc_uu_vdw(*ni, indi, indj, startj,
				  pftable, *hinv,
				  vdwtype, vdwparam,
				  xyzq, force, iscoord,
				  sforce, scoordtab,
				  vdwpot, 
				  *roff, *ron, *vdw_type);
#endif
}
