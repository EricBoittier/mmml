#include <iostream>
#include <typeinfo>
#include <stdlib.h>
#include <math.h>
#include "enb_core_vec.h"
#include "sse_defs.h"

#ifdef __SSE2__

//
// Class creator
//
template <typename T, typename VT, typename AVT>
enb_core_vec<T, VT, AVT>::enb_core_vec() {
}

//
// Class destructor
//
template <typename T, typename VT, typename AVT>
enb_core_vec<T, VT, AVT>::~enb_core_vec() {
}

//
// Determine VdW constants for analytical VdW function evaluation
//
template <typename T>
void vdw_const_vsh(double roff, T *aux) {
  double roff6 = roff*roff*roff*roff*roff*roff;
  double roff12 = roff6*roff6;
  double roff18 = roff12*roff6;
  aux[0] = roff*roff;
  aux[1] = 2.0/roff18;
  aux[2] = -3.0/roff12;
  aux[3] = 1.0/roff12;
  aux[4] = -2.0/roff6;
  aux[5] = -6.0/roff12;
  aux[6] = 12.0/roff18;
}

template <typename T>
void vdw_const_vsw(double roff, double ron, T *aux) {
  // aux[0] = ron^2
  // aux[1] = roff^2
  // aux[2] = roff^2 - 3*ron^2
  // aux[3] = 1/(roff^2 - ron^2)
  // aux[4] = 6/(roff^2 - ron^2)
  double ron2 = ron*ron;
  double roff2 = roff*roff;
  aux[0] = ron2;
  aux[1] = roff2;
  aux[2] = roff2 - 3.0*ron2;
  aux[3] = 1.0/((roff2 - ron2)*(roff2 - ron2)*(roff2 - ron2));
  aux[4] = 6.0/((roff2 - ron2)*(roff2 - ron2)*(roff2 - ron2));
}

template <typename T>
void vdw_const_vfsw(double roff, double ron, T *aux) {
  // aux[0] = ron^2
  // aux[1] = dv12
  // aux[2] = dv6
  // aux[3] = k12
  // aux[4] = k6
  // aux[5] = 1/roff^6
  // aux[6] = 1/roff^3
  // aux[7] = roff^2
  double ron2 = ron*ron;
  double roff2 = roff*roff;
  double ron3 = ron*ron*ron;
  double roff3 = roff*roff*roff;
  double ron6 = ron3*ron3;
  double roff6 = roff3*roff3;
  double roff12 = roff6*roff6;
  double dv12, dv6, k12, k6;
  // Constants for VdW
  if (ron < roff) {
    k6 = roff3/(roff3 - ron3);
    k12 = roff6/(roff6 - ron6);
    dv6 = -1.0/(ron3*roff3);
    dv12 = -1.0/(ron6*roff6);
  } else {
    k6 = 0.0;
    k12 = 0.0;
    dv6 = -1.0/roff6;
    dv12 = -1.0/roff12;
  }
  aux[0] = ron2;
  aux[1] = dv12;
  aux[2] = dv6;
  aux[3] = k12;
  aux[4] = k6;
  aux[5] = 1.0/roff6;
  aux[6] = 1.0/roff3;
  aux[7] = roff2;
}

static void mm_print(__m128d a) {
  double ad[2];
  _mm_storeu_pd(ad, a);
  //fprintf(stderr,"%lf %lf\n",ad[0],ad[1]);
  std::cerr << ad[0] << " " << ad[1] << std::endl;
  return;
}

static void mm_print(__m128 a) {
  float af[4];
  _mm_storeu_ps(af, a);
  //fprintf(stderr,"%13.9f %13.9f %13.9f %13.9f\n",af[0],af[1],af[2],af[3]);
  std::cerr << af[0] << " " << af[1] << " " << af[2] << " " << af[3] << std::endl;
  return;
}

inline __attribute__((always_inline))
void mm_setzero(__m128 &a) {
  a = _mm_setzero_ps();
}

inline __attribute__((always_inline))
void mm_setzero(__m128d &a) {
  a = _mm_setzero_pd();
}

#ifdef __AVX__
inline __attribute__((always_inline))
void mm_setzero(__m256 &a) {
  a = _mm256_setzero_ps();
}

inline __attribute__((always_inline))
void mm_setzero(__m256d &a) {
  a = _mm256_setzero_pd();
}
#endif

//----------------------------------------------------------
// Generic templates, can't be used for anything
template <typename T, typename VT>
VT mm_set1(const T a) {
  VT b;
  mm_setzero(b);
  std::cerr << "enb_core_vec::mm_set1, calling generic template not possible" << std::endl;
  exit(1);
  return b;
}

// Loads a single value to all registers
template <typename T, typename VT>
VT mm_load_one2all(const T *a) {
  VT b;
  mm_setzero(b);
  std::cerr << "enb_core_vec::mm_load_one2all, calling generic template not possible" << std::endl;
  exit(1);
  return b;
}

// Loads a single value to the lowest register and zeros the rest
template <typename T, typename VT>
VT mm_load_one(const T *a) {
  VT b;
  mm_setzero(b);
  std::cerr << "enb_core_vec::mm_load_one, calling generic template not possible" << std::endl;
  exit(1);
  return b;
}

//----------------------------------------------------------

template <>
inline __attribute__((always_inline))
__m128 mm_set1<float, __m128>(const float a) {
  return _mm_set1_ps(a);
}

template <>
inline __attribute__((always_inline))
__m128d mm_set1<double, __m128d>(const double a) {
  return _mm_set1_pd(a);
}

// Loads a single value to all registers
template <>
inline __attribute__((always_inline))
__m128 mm_load_one2all<float, __m128>(const float *a) {
  return _mm_load1_ps(a);
}

template <>
inline __attribute__((always_inline))
__m128d mm_load_one2all<double, __m128d>(const double *a) {
  return _mm_load1_pd(a);
}

// Loads a single value to the lowest register and zeros the rest
template <>
inline __attribute__((always_inline))
__m128d mm_load_one<double, __m128d>(const double *a) {
  return _mm_load_sd(a);
}

#ifdef __AVX__
template <>
inline __attribute__((always_inline))
__m256 mm_set1<float, __m256>(const float a) {
  return _mm256_set1_ps(a);
}

template <>
inline __attribute__((always_inline))
__m256d mm_set1<double, __m256d>(const double a) {
  return _mm256_set1_pd(a);
}

// Loads a single value to all registers
template <>
inline __attribute__((always_inline))
__m256 mm_load_one2all<float, __m256>(const float *a) {
  return _mm256_broadcast_ss(a);
}

template <>
inline __attribute__((always_inline))
__m256d mm_load_one2all<double, __m256d>(const double *a) {
  return _mm256_broadcast_sd(a);
}

// Loads a single value to the lowest register and zeros the rest
template <>
inline __attribute__((always_inline))
__m256d mm_load_one<double, __m256d>(const double *a) {
  return _mm256_maskload_pd(a,_mm256_set_epi64x(0,0,0,0xffffffffffffffff));
}

template <>
inline __attribute__((always_inline))
__m256 mm_load_one<float, __m256>(const float *a) {
  return _mm256_maskload_ps(a,_mm256_set_epi32(0,0,0,0,0,0,0,0xffffffff));
}
#endif

// ---------------------------------------------------------------------
//
// AVX functions
//
#ifdef __AVX__
static void mm_print(__m256d a) {
  double ad[4];
  _mm256_storeu_pd(ad, a);
  //fprintf(stderr,"%lf %lf %lf %lf\n",ad[0],ad[1],ad[2],ad[3]);
  std::cerr << ad[0] << " " << ad[1] << " " << ad[2] << " " << ad[3] << std::endl;
  return;
}

static void mm_print(__m256 a) {
  float af[8];
  _mm256_storeu_ps(af, a);
  //fprintf(stderr,"%f %f %f %f %f %f %f %f\n",
  //af[0],af[1],af[2],af[3],af[4],af[5],af[6],af[7]);
  std::cerr << af[0] << " " << af[1] << " " << af[2] << " " << af[3]
	    << " " << af[4] << " " << af[5] << " " << af[6] << " " << af[7] << std::endl;
  return;
}

static void mm_print(__m256i a) {
  int ai[8];
  _mm256_storeu_si256((__m256i *)ai, a);
  //fprintf(stderr,"%d %d %d %d %d %d %d %d\n",
  //	  ai[0],ai[1],ai[2],ai[3],ai[4],ai[5],ai[6],ai[7]);
  std::cerr << ai[0] << " " << ai[1] << " " << ai[2] << " " << ai[3]
	    << " " << ai[4] << " " << ai[5] << " " << ai[6] << " " << ai[7] << std::endl;
  return;
}

// Moves higher 128 bits down to lower 128 bits, top 128 bits are undefined
inline __attribute__((always_inline))
__m256d mm_movehl(const __m256d a) {
  return _mm256_castpd128_pd256(_mm256_extractf128_pd(a, 1));
}

inline __attribute__((always_inline))
__m256 mm_movehl(const __m256 a) {
  return _mm256_castps128_ps256(_mm256_extractf128_ps(a, 1));
}

inline __attribute__((always_inline))
__m256i mm_movehl(const __m256i a) {
  return _mm256_castsi128_si256(_mm256_extractf128_si256(a, 1));
}

//
// Performs reduction a0 + a1 + a2 + a3, and stores the result in the lowest 64 bits
//
inline __attribute__((always_inline))
__m256d mm_sumall(const __m256d a) {
  __m256d a0a1 = _mm256_hadd_pd(a,a);                   // a0+a1 a0+a1 a2+a3 a2+a3
  __m256d a2a3 = _mm256_permute2f128_pd(a0a1, a0a1, 3); // a2+a3 a2+a3 a0+a1 a0+a1
  return _mm256_add_pd(a0a1, a2a3);
}

// Stores all values to a double pointer by summing them
inline __attribute__((always_inline))
void mm_store_sumall(double *p, const __m256d a) {
  _mm_store_sd(p, _mm256_castpd256_pd128(mm_sumall(a)));
}

inline __attribute__((always_inline))
__m256 mm_add(const __m256 a, const __m256 b) {
  return _mm256_add_ps(a, b);
}

inline __attribute__((always_inline))
__m256d mm_add(const __m256d a, const __m256d b) {
  return _mm256_add_pd(a, b);
}

inline __attribute__((always_inline))
__m256 mm_sub(const __m256 a, const __m256 b) {
  return _mm256_sub_ps(a, b);
}

inline __attribute__((always_inline))
__m256d mm_sub(const __m256d a, const __m256d b) {
  return _mm256_sub_pd(a, b);
}

inline __attribute__((always_inline))
__m256 mm_mul(const __m256 a, const __m256 b) {
  return _mm256_mul_ps(a, b);
}

inline __attribute__((always_inline))
__m256d mm_mul(const __m256d a, const __m256d b) {
  return _mm256_mul_pd(a, b);
}

inline __attribute__((always_inline))
__m256 mm_and(const __m256 a, const __m256 b) {
  return _mm256_and_ps(a, b);
}

inline __attribute__((always_inline))
__m256d mm_and(const __m256d a, const __m256d b) {
  return _mm256_and_pd(a, b);
}

inline __attribute__((always_inline))
__m256 mm_min(const __m256 a, const __m256 b) {
  return _mm256_min_ps(a, b);
}

inline __attribute__((always_inline))
__m256d mm_min(const __m256d a, const __m256d b) {
  return _mm256_min_pd(a, b);
}

inline __attribute__((always_inline))
__m256 mm_cmplt(const __m256 a, const __m256 b) {
  return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
}

inline __attribute__((always_inline))
__m256d mm_cmplt(const __m256d a, const __m256d b) {
  return _mm256_cmp_pd(a, b, _CMP_LT_OQ);
}

inline __attribute__((always_inline))
int mm_movemask(const __m256 a) {
  return _mm256_movemask_ps(a);
}

inline __attribute__((always_inline))
int mm_movemask(const __m256d a) {
  return _mm256_movemask_pd(a);
}

inline __attribute__((always_inline))
__m256d mm_reduce(const __m256d a, const __m256 b) {
  __m256d ab = _mm256_add_pd(a, _mm256_cvtps_pd( _mm256_castps256_ps128(b)));
  return _mm256_add_pd(ab, _mm256_cvtps_pd(_mm256_extractf128_ps(b,1)));
}

inline __attribute__((always_inline))
__m256d mm_reduce(const __m256d a, const __m256d b) {
    return _mm256_add_pd(a, b);
}

inline __attribute__((always_inline))
void get_mask(__m256d &mask, const int jleft) {
  // jleft = 1, 2, 3
  const __m256d masktab[3] = {_mm256_castsi256_pd(_mm256_set_epi64x(0,   0,  0, -1)),
			      _mm256_castsi256_pd(_mm256_set_epi64x(0,   0, -1, -1)),
			      _mm256_castsi256_pd(_mm256_set_epi64x(0,  -1, -1, -1))};
  mask = masktab[jleft-1];
}

inline __attribute__((always_inline))
void get_mask(__m256 &mask, const int jleft) {
  // jleft = 1, 2, 3, 4, 5, 6, 7
  const __m256 masktab[7] = {_mm256_castsi256_ps(_mm256_set_epi32(0,  0,  0,  0,  0,  0,  0, -1)),
			     _mm256_castsi256_ps(_mm256_set_epi32(0,  0,  0,  0,  0,  0, -1, -1)),
			     _mm256_castsi256_ps(_mm256_set_epi32(0,  0,  0,  0,  0, -1, -1, -1)),
			     _mm256_castsi256_ps(_mm256_set_epi32(0,  0,  0,  0, -1, -1, -1, -1)),
			     _mm256_castsi256_ps(_mm256_set_epi32(0,  0,  0, -1, -1, -1, -1, -1)),
			     _mm256_castsi256_ps(_mm256_set_epi32(0,  0, -1, -1, -1, -1, -1, -1)),
			     _mm256_castsi256_ps(_mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1))};
  mask = masktab[jleft-1];
}

#endif //__AVX__
// ---------------------------------------------------------------------

inline __attribute__((always_inline))
__m128d mm_sumall(const __m128d a) {
#ifdef __SSE3__
  return _mm_hadd_pd(a,a);
#else
  return _mm_add_pd(a,_mm_unpackhi_pd(a,a));
#endif
}

// Stores all values to a double pointer by summing them
inline __attribute__((always_inline))
void mm_store_sumall(double *p, const __m128d a) {
  _mm_store_sd(p, mm_sumall(a));
}

inline __attribute__((always_inline))
__m128 mm_add(const __m128 a, const __m128 b) {
  return _mm_add_ps(a, b);
}

inline __attribute__((always_inline))
__m128d mm_add(const __m128d a, const __m128d b) {
  return _mm_add_pd(a, b);
}

inline __attribute__((always_inline))
__m128 mm_sub(const __m128 a, const __m128 b) {
  return _mm_sub_ps(a, b);
}

inline __attribute__((always_inline))
__m128d mm_sub(const __m128d a, const __m128d b) {
  return _mm_sub_pd(a, b);
}

inline __attribute__((always_inline))
__m128 mm_mul(const __m128 a, const __m128 b) {
  return _mm_mul_ps(a, b);
}

inline __attribute__((always_inline))
__m128d mm_mul(const __m128d a, const __m128d b) {
  return _mm_mul_pd(a, b);
}

inline __attribute__((always_inline))
__m128 mm_and(const __m128 a, const __m128 b) {
  return _mm_and_ps(a, b);
}

inline __attribute__((always_inline))
__m128d mm_and(const __m128d a, const __m128d b) {
  return _mm_and_pd(a, b);
}

inline __attribute__((always_inline))
__m128 mm_min(const __m128 a, const __m128 b) {
  return _mm_min_ps(a, b);
}

inline __attribute__((always_inline))
__m128d mm_min(const __m128d a, const __m128d b) {
  return _mm_min_pd(a, b);
}

inline __attribute__((always_inline))
__m128 mm_cmplt(const __m128 a, const __m128 b) {
  return _mm_cmplt_ps(a, b);
}

inline __attribute__((always_inline))
__m128d mm_cmplt(const __m128d a, const __m128d b) {
  return _mm_cmplt_pd(a, b);
}

inline __attribute__((always_inline))
int mm_movemask(const __m128 a) {
  return _mm_movemask_ps(a);
}

inline __attribute__((always_inline))
int mm_movemask(const __m128d a) {
  return _mm_movemask_pd(a);
}

inline __attribute__((always_inline))
__m128d mm_reduce(const __m128d a, const __m128 b) {
  __m128d ab = _mm_add_pd(a, _mm_cvtps_pd(b));
  return _mm_add_pd(ab, _mm_cvtps_pd(_mm_movehl_ps(b,b)));
}

inline __attribute__((always_inline))
__m128d mm_reduce(const __m128d a, const __m128d b) {
    return _mm_add_pd(a, b);
}

inline __attribute__((always_inline))
void get_mask(__m128d &mask, const int jleft) {
  mask = _mm_castsi128_pd(_mm_set_epi32(0, 0, 0xffffffff, 0xffffffff));
}

inline __attribute__((always_inline))
void get_mask(__m128 &mask, const int jleft) {
  // jleft = 1, 2, 3
  const __m128 masktab[3] = {
    _mm_castsi128_ps(_mm_set_epi32(0, 0,          0,          0xffffffff)),  // jleft = 1
    _mm_castsi128_ps(_mm_set_epi32(0, 0,          0xffffffff, 0xffffffff)),
    _mm_castsi128_ps(_mm_set_epi32(0, 0xffffffff, 0xffffffff, 0xffffffff))
  };
  mask = masktab[jleft-1];
}

//
// Loads 2 double precision numbers from two separate addresses
//
inline __attribute__((always_inline))
__m128d load_2dp(const double *dp1, const double *dp2) {
  return _mm_loadh_pd(_mm_load_sd(dp1), dp2);
}

//
// Loads full vector length of values 
//
inline __attribute__((always_inline))
void load_force(const double *force, const int *ind, const int i, const int t, __m128d &a) {
  a = load_2dp(&force[ind[i*2]*3+t], &force[ind[i*2+1]*3+t]); 
}

//
// Stores 2 double precision numbers into two separate addresses
// Note: We write the highest first (so that masked calculation work correctly)
//
static inline __attribute__((always_inline))
void store_2dp(const __m128d a, double *dp1, double *dp2) {
  _mm_store_sd(dp2, _mm_unpackhi_pd(a, a));
  _mm_store_sd(dp1, a);
}

inline __attribute__((always_inline))
void store_force(double *force, const int *ind, const int i, const int t, const __m128d a) {
  store_2dp(a, &force[ind[i*2]*3+t], &force[ind[i*2+1]*3+t]);
}

#ifdef __AVX__
static inline __attribute__((always_inline))
__m256d load_4dp(const double *dp1, const double *dp2, const double *dp3, const double *dp4) {
   __m128d lo = _mm_loadh_pd(_mm_load_sd(dp1), dp2);
   __m128d hi = _mm_loadh_pd(_mm_load_sd(dp3), dp4);
   return _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 1);
   //   return _mm256_set_m128d(hi, lo);
}

static inline __attribute__((always_inline))
void store_4dp(const __m256d a, double *dp1, double *dp2, double *dp3, double *dp4) {
  __m128d b;
  b = _mm256_castpd256_pd128( mm_movehl(a) );   // Take higher 128 bits
  _mm_store_sd(dp4, _mm_unpackhi_pd(b, b));
  _mm_store_sd(dp3, b);
  b = _mm256_castpd256_pd128(a);
  _mm_store_sd(dp2, _mm_unpackhi_pd(b, b));
  _mm_store_sd(dp1, b);
}

inline __attribute__((always_inline))
void store_force(double *force, const int *ind, const int i, const int t, const __m256d a) {
  store_4dp(a, &force[ind[i*4]*3+t], &force[ind[i*4+1]*3+t],
	    &force[ind[i*4+2]*3+t], &force[ind[i*4+3]*3+t]);
}

inline __attribute__((always_inline))
void load_force(const double *force, const int *ind, const int i, const int t, __m256d &a) {
  a = load_4dp(&force[ind[i*4]*3+t], &force[ind[i*4+1]*3+t],
	       &force[ind[i*4+2]*3+t], &force[ind[i*4+3]*3+t]);
}

inline __attribute__((always_inline))
void load_vdwparam(const double *vdwparam, const int *ivdw, __m256d &c6, __m256d &c12) {
  // [c6_0][c12_0] [c6_1][c12_1]
  __m256d tmp1 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_load_pd(&vdwparam[ivdw[0]])),
				      _mm_load_pd(&vdwparam[ivdw[1]]), 1);
  //__m256d tmp1 = _mm256_loadu2_m128d(&vdwparam[ivdw[1]], &vdwparam[ivdw[0]]);
  // [c6_2][c12_2] [c6_3][c12_3]
  __m256d tmp2 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_load_pd(&vdwparam[ivdw[2]])),
				      _mm_load_pd(&vdwparam[ivdw[3]]), 1);
  //__m256d tmp2 = _mm256_loadu2_m128d(&vdwparam[ivdw[3]], &vdwparam[ivdw[2]]);

  __m256d tmp1t, tmp2t, a, b;

  // [c6_1][c12_1] [c6_0][c12_0]
  tmp1t = _mm256_permute2f128_pd(tmp1, tmp1, 1);
  // [c6_3][c12_3] [c6_2][c12_2]
  tmp2t = _mm256_permute2f128_pd(tmp2, tmp2, 1);

  // [c6_0][c6_1] [c6_1][c6_0]
  a = _mm256_shuffle_pd(tmp1, tmp1t, 0*1 + 0*2 + 0*4 + 0*8);
  // [c6_2][c6_3] [c6_3][c6_2]
  b = _mm256_shuffle_pd(tmp2, tmp2t, 0*1 + 0*2 + 0*4 + 0*8);
  // [c6_0][c6_1] [c6_2][c6_3]
  c6 = _mm256_insertf128_pd(a, _mm256_castpd256_pd128(b), 1);

  // [c12_0][c12_1] [c12_1][c12_0]
  a = _mm256_shuffle_pd(tmp1, tmp1t, 1*1 + 1*2 + 1*4 + 1*8);
  // [c12_2][c12_3] [c12_3][c12_2]
  b = _mm256_shuffle_pd(tmp2, tmp2t, 1*1 + 1*2 + 1*4 + 1*8);
  // [c12_0][c12_1] [c12_2][c12_3]
  c12 = _mm256_insertf128_pd(a, _mm256_castpd256_pd128(b), 1);
}

inline __attribute__((always_inline))
void load_vdwparam(const float *vdwparam, const int *ivdw, __m256 &c6, __m256 &c12) {
  // tmp1 = [c6_0][c12_0][c6_1][c12_1]
  // tmp2 = [c6_2][c12_2][c6_3][c12_3]
  __m128 tmp1, tmp2;
  tmp1 = _mm_loadl_pi(_mm_loadh_pi(tmp1, (__m64 *)&vdwparam[ivdw[1]]), (__m64 *)&vdwparam[ivdw[0]]);
  tmp2 = _mm_loadl_pi(_mm_loadh_pi(tmp2, (__m64 *)&vdwparam[ivdw[3]]), (__m64 *)&vdwparam[ivdw[2]]);
  // tmp1 = [c6_0][c6_1][c12_0][c12_1]
  // tmp2 = [c6_2][c6_3][c12_2][c12_3]
  tmp1 = _mm_shuffle_ps(tmp1, tmp1, _MM_SHUFFLE(3,1,2,0));
  tmp2 = _mm_shuffle_ps(tmp2, tmp2, _MM_SHUFFLE(3,1,2,0));
  // c6 = [c6_0][c6_1][c6_2][c6_3]
  __m128 c6_lo = _mm_movelh_ps(tmp1, tmp2);
  // c12 = [c12_0][c12_1][c12_2][c12_3]
  __m128 c12_lo = _mm_movehl_ps(tmp2, tmp1);

  // tmp1 = [c6_0][c12_0][c6_1][c12_1]
  // tmp2 = [c6_2][c12_2][c6_3][c12_3]
  tmp1 = _mm_loadl_pi(_mm_loadh_pi(tmp1, (__m64 *)&vdwparam[ivdw[5]]), (__m64 *)&vdwparam[ivdw[4]]);
  tmp2 = _mm_loadl_pi(_mm_loadh_pi(tmp2, (__m64 *)&vdwparam[ivdw[7]]), (__m64 *)&vdwparam[ivdw[6]]);
  // tmp1 = [c6_0][c6_1][c12_0][c12_1]
  // tmp2 = [c6_2][c6_3][c12_2][c12_3]
  tmp1 = _mm_shuffle_ps(tmp1, tmp1, _MM_SHUFFLE(3,1,2,0));
  tmp2 = _mm_shuffle_ps(tmp2, tmp2, _MM_SHUFFLE(3,1,2,0));
  // c6 = [c6_0][c6_1][c6_2][c6_3]
  __m128 c6_hi = _mm_movelh_ps(tmp1, tmp2);
  // c12 = [c12_0][c12_1][c12_2][c12_3]
  __m128 c12_hi = _mm_movehl_ps(tmp2, tmp1);

  c6  = _mm256_insertf128_ps(_mm256_castps128_ps256(c6_lo),  c6_hi, 1);
  c12 = _mm256_insertf128_ps(_mm256_castps128_ps256(c12_lo), c12_hi, 1);
}

// Loads charges into a vector
inline __attribute__((always_inline))
void load_charge(const xyzq_t<double> *xyzq, const int *ind, __m256d &q) {
  __m128d tmp1 =  _mm_loadh_pd(_mm_load_sd(&xyzq[ind[0]].q), &xyzq[ind[1]].q);
  __m128d tmp2 =  _mm_loadh_pd(_mm_load_sd(&xyzq[ind[2]].q), &xyzq[ind[3]].q);
  q = _mm256_insertf128_pd(_mm256_castpd128_pd256(tmp1), tmp2, 1);  
}

inline __attribute__((always_inline))
void load_charge(const xyzq_t<float> *xyzq, const int *ind, __m256 &q) {
  __m128 tmp1 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[0]].q),_mm_load_ss(&xyzq[ind[1]].q));
  __m128 tmp2 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[2]].q),_mm_load_ss(&xyzq[ind[3]].q));
  __m128 q_lo = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1,0,1,0));
  tmp1 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[4]].q),_mm_load_ss(&xyzq[ind[5]].q));
  tmp2 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[6]].q),_mm_load_ss(&xyzq[ind[7]].q));
  __m128 q_hi = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1,0,1,0));
  q = _mm256_insertf128_ps(_mm256_castps128_ps256(q_lo), q_hi, 1);
}

#if KEY_LJPME==1
// Loads LJPME multiplicative C6 coefficients into a vector
inline __attribute__((always_inline))
void load_c6(const xyzq_t<double> *xyzq, const int *ind, __m256d &c6) {
  __m128d tmp1 =  _mm_loadh_pd(_mm_load_sd(&xyzq[ind[0]].c6), &xyzq[ind[1]].c6);
  __m128d tmp2 =  _mm_loadh_pd(_mm_load_sd(&xyzq[ind[2]].c6), &xyzq[ind[3]].c6);
  c6 = _mm256_insertf128_pd(_mm256_castpd128_pd256(tmp1), tmp2, 1);  
}

inline __attribute__((always_inline))
void load_c6(const xyzq_t<float> *xyzq, const int *ind, __m256 &c6) {
  __m128 tmp1 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[0]].c6),_mm_load_ss(&xyzq[ind[1]].c6));
  __m128 tmp2 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[2]].c6),_mm_load_ss(&xyzq[ind[3]].c6));
  __m128 c6_lo = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1,0,1,0));
  tmp1 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[4]].c6),_mm_load_ss(&xyzq[ind[5]].c6));
  tmp2 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[6]].c6),_mm_load_ss(&xyzq[ind[7]].c6));
  __m128 c6_hi = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1,0,1,0));
  c6 = _mm256_insertf128_ps(_mm256_castps128_ps256(c6_lo), c6_hi, 1);
}
#endif

#define MM256_SHUFFLE(x3, x2, x1, x0) ((x3 << 3) | (x2 << 2) | (x1 << 1) | (x0))

//
// Transposes a 4x3 matrix, macro version
//
#define TRANSPOSE_4X3(row1in, row2in, row3in, row4in, row1, row2, row3) \
  {									\
    __m256d tmp1 = _mm256_shuffle_pd(row1in, row2in, MM256_SHUFFLE(0, 0, 0, 0)); \
    __m256d tmp2 = _mm256_shuffle_pd(row3in, row4in, MM256_SHUFFLE(0, 0, 0, 0)); \
    tmp2 = _mm256_permute2f128_pd(tmp2, tmp2, 1);			\
    row1 = _mm256_blend_pd(tmp1, tmp2, MM256_SHUFFLE(1, 1, 0, 0));	\
    row3 = _mm256_blend_pd(tmp2, tmp1, MM256_SHUFFLE(1, 1, 0, 0));	\
    row3 = _mm256_permute2f128_pd(row3, row3, 1);			\
    tmp1 = _mm256_shuffle_pd(row1in, row2in, MM256_SHUFFLE(1, 1, 1, 1)); \
    tmp2 = _mm256_shuffle_pd(row3in, row4in, MM256_SHUFFLE(1, 1, 1, 1)); \
    tmp2 = _mm256_permute2f128_pd(tmp2, tmp2, 1);			\
    row2 = _mm256_blend_pd(tmp1, tmp2, MM256_SHUFFLE(1, 1, 0, 0));	\
  }

//
// Transposes a 4x4 matrix, macro version
//
#define TRANSPOSE_4X4(row1in, row2in, row3in, row4in, row1, row2, row3, row4) \
  {									\
    __m256d tmp1 = _mm256_shuffle_pd(row1in, row2in, MM256_SHUFFLE(0, 0, 0, 0)); \
    __m256d tmp2 = _mm256_shuffle_pd(row3in, row4in, MM256_SHUFFLE(0, 0, 0, 0)); \
    tmp2 = _mm256_permute2f128_pd(tmp2, tmp2, 1);			\
    row1 = _mm256_blend_pd(tmp1, tmp2, MM256_SHUFFLE(1, 1, 0, 0));	\
    row3 = _mm256_blend_pd(tmp2, tmp1, MM256_SHUFFLE(1, 1, 0, 0));	\
    row3 = _mm256_permute2f128_pd(row3, row3, 1);			\
    tmp1 = _mm256_shuffle_pd(row1in, row2in, MM256_SHUFFLE(1, 1, 1, 1)); \
    tmp2 = _mm256_shuffle_pd(row3in, row4in, MM256_SHUFFLE(1, 1, 1, 1)); \
    tmp2 = _mm256_permute2f128_pd(tmp2, tmp2, 1);			\
    row2 = _mm256_blend_pd(tmp1, tmp2, MM256_SHUFFLE(1, 1, 0, 0));	\
    row4 = _mm256_blend_pd(tmp2, tmp1, MM256_SHUFFLE(1, 1, 0, 0));	\
    row4 = _mm256_permute2f128_pd(row4, row4, 1);			\
  }

inline __attribute__((always_inline))
void mm_load_xyz(const xyzq_t<double> *xyzq, const int *ind, __m256d &x, __m256d &y, __m256d &z) {
  __m256d load1 = _mm256_loadu_pd(&xyzq[ind[0]].x);
  __m256d load2 = _mm256_loadu_pd(&xyzq[ind[1]].x);
  __m256d load3 = _mm256_loadu_pd(&xyzq[ind[2]].x);
  __m256d load4 = _mm256_loadu_pd(&xyzq[ind[3]].x);
  TRANSPOSE_4X3(load1, load2, load3, load4, x, y, z);
}

inline __attribute__((always_inline))
void mm_load_xyz(const xyzq_t<float> *xyzq, const int *ind, __m256 &x, __m256 &y, __m256 &z) {
  __m128 tmp1, tmp2;
  
  tmp1 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[0]].x),_mm_load_ss(&xyzq[ind[1]].x));
  tmp2 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[2]].x),_mm_load_ss(&xyzq[ind[3]].x));
  __m128 x_lo = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1,0,1,0));
  tmp1 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[4]].x),_mm_load_ss(&xyzq[ind[5]].x));
  tmp2 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[6]].x),_mm_load_ss(&xyzq[ind[7]].x));
  __m128 x_hi = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1,0,1,0));
  x = _mm256_insertf128_ps(_mm256_castps128_ps256(x_lo), x_hi, 1);

  tmp1 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[0]].y),_mm_load_ss(&xyzq[ind[1]].y));
  tmp2 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[2]].y),_mm_load_ss(&xyzq[ind[3]].y));
  __m128 y_lo = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1,0,1,0));
  tmp1 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[4]].y),_mm_load_ss(&xyzq[ind[5]].y));
  tmp2 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[6]].y),_mm_load_ss(&xyzq[ind[7]].y));
  __m128 y_hi = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1,0,1,0));
  y = _mm256_insertf128_ps(_mm256_castps128_ps256(y_lo), y_hi, 1);
  
  tmp1 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[0]].z),_mm_load_ss(&xyzq[ind[1]].z));
  tmp2 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[2]].z),_mm_load_ss(&xyzq[ind[3]].z));
  __m128 z_lo = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1,0,1,0));
  tmp1 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[4]].z),_mm_load_ss(&xyzq[ind[5]].z));
  tmp2 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[6]].z),_mm_load_ss(&xyzq[ind[7]].z));
  __m128 z_hi = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1,0,1,0));
  z = _mm256_insertf128_ps(_mm256_castps128_ps256(z_lo), z_hi, 1);
}

#endif //__AVX__

inline __attribute__((always_inline))
void load_vdwparam(const double *vdwparam, const int *ivdw, __m128d &c6, __m128d &c12) {
__m128d tmp1 = _mm_load_pd(&vdwparam[ivdw[0]]);   // [c6][c12]
  __m128d tmp2 = _mm_load_pd(&vdwparam[ivdw[1]]);
  c6  = _mm_unpacklo_pd(tmp1, tmp2);
  c12 = _mm_unpackhi_pd(tmp1, tmp2);
}

inline __attribute__((always_inline))
void load_vdwparam(const float *vdwparam, const int *ivdw, __m128 &c6, __m128 &c12) {
  // tmp1 = [c6_0][c12_0][c6_1][c12_1]
  // tmp2 = [c6_2][c12_2][c6_3][c12_3]
  __m128 tmp1 = _mm_loadl_pi(_mm_loadh_pi(c6, (__m64 *)&vdwparam[ivdw[1]]), (__m64 *)&vdwparam[ivdw[0]]);
  __m128 tmp2 = _mm_loadl_pi(_mm_loadh_pi(c6, (__m64 *)&vdwparam[ivdw[3]]), (__m64 *)&vdwparam[ivdw[2]]);
  // tmp1 = [c6_0][c6_1][c12_0][c12_1]
  // tmp2 = [c6_2][c6_3][c12_2][c12_3]
  tmp1 = _mm_shuffle_ps(tmp1, tmp1, _MM_SHUFFLE(3,1,2,0));
  tmp2 = _mm_shuffle_ps(tmp2, tmp2, _MM_SHUFFLE(3,1,2,0));
  // c6 = [c6_0][c6_1][c6_2][c6_3]
  c6 = _mm_movelh_ps(tmp1, tmp2);
  // c12 = [c12_0][c12_1][c12_2][c12_3]
  c12 = _mm_movehl_ps(tmp2, tmp1);
}

// Loads charges into a vector
inline __attribute__((always_inline))
void load_charge(const xyzq_t<double> *xyzq, const int *ind, __m128d &q) {
  q = _mm_loadh_pd(_mm_load_sd(&xyzq[ind[0]].q), &xyzq[ind[1]].q);
}

inline __attribute__((always_inline))
void load_charge(const xyzq_t<float> *xyzq, const int *ind, __m128 &q) {
  __m128 tmp1 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[0]].q),_mm_load_ss(&xyzq[ind[1]].q));
  __m128 tmp2 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[2]].q),_mm_load_ss(&xyzq[ind[3]].q));
  q = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1,0,1,0));
}

#if KEY_LJPME==1
// Loads c6s into a vector
inline __attribute__((always_inline))
void load_c6(const xyzq_t<double> *xyzq, const int *ind, __m128d &c6) {
  c6 = _mm_loadh_pd(_mm_load_sd(&xyzq[ind[0]].c6), &xyzq[ind[1]].c6);
}

inline __attribute__((always_inline))
void load_c6(const xyzq_t<float> *xyzq, const int *ind, __m128 &c6) {
  __m128 tmp1 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[0]].c6),_mm_load_ss(&xyzq[ind[1]].c6));
  __m128 tmp2 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[2]].c6),_mm_load_ss(&xyzq[ind[3]].c6));
  c6 = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1,0,1,0));
}
#endif

inline __attribute__((always_inline))
void mm_load_xyz(const xyzq_t<float> *xyzq, const int *ind, __m128 &x, __m128 &y, __m128 &z) {
  __m128 tmp1, tmp2;
  
  tmp1 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[0]].x),_mm_load_ss(&xyzq[ind[1]].x));
  tmp2 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[2]].x),_mm_load_ss(&xyzq[ind[3]].x));
  x = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1,0,1,0));
  
  tmp1 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[0]].y),_mm_load_ss(&xyzq[ind[1]].y));
  tmp2 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[2]].y),_mm_load_ss(&xyzq[ind[3]].y));
  y = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1,0,1,0));
  
  tmp1 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[0]].z),_mm_load_ss(&xyzq[ind[1]].z));
  tmp2 = _mm_unpacklo_ps(_mm_load_ss(&xyzq[ind[2]].z),_mm_load_ss(&xyzq[ind[3]].z));
  z = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1,0,1,0));
}

inline __attribute__((always_inline))
void mm_load_xyz(const xyzq_t<double> *xyzq, const int *ind, __m128d &x, __m128d &y, __m128d &z) {
  x = _mm_loadh_pd(_mm_load_sd(&xyzq[ind[0]].x), &xyzq[ind[1]].x);
  y = _mm_loadh_pd(_mm_load_sd(&xyzq[ind[0]].y), &xyzq[ind[1]].y);
  z = _mm_loadh_pd(_mm_load_sd(&xyzq[ind[0]].z), &xyzq[ind[1]].z);
}

inline __attribute__((always_inline))
__m128 mm_rsqrt(__m128 rsq) {
  return _mm_rsqrt_ps(rsq);
}

inline __attribute__((always_inline))
__m128d mm_rsqrt(__m128d rsq) {
  return _mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(rsq)));
}

//
// Calculates inverse square roots
// Two Newton iterations for double precision, one for single
//
inline __attribute__((always_inline))
void invsqrt(const __m128d rsq1, const __m128d rsq2, const __m128d rsq3,
	     __m128d &rinv1, __m128d &rinv2, __m128d &rinv3) {
  const __m128d half  = mm_set1<double, __m128d>(0.5);
  const __m128d three = mm_set1<double, __m128d>(3.0);
  __m128d lu1 = mm_rsqrt(rsq1);
  __m128d lu2 = mm_rsqrt(rsq2);
  __m128d lu3 = mm_rsqrt(rsq3);
  lu1 = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu1,lu1),rsq1)),lu1));
  lu2 = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu2,lu2),rsq2)),lu2));
  lu3 = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu3,lu3),rsq3)),lu3));
  rinv1 = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu1,lu1),rsq1)),lu1));
  rinv2 = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu2,lu2),rsq2)),lu2));
  rinv3 = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu3,lu3),rsq3)),lu3));
}

inline __attribute__((always_inline))
void invsqrt(const __m128 rsq1, const __m128 rsq2, const __m128 rsq3,
	     __m128 &rinv1, __m128 &rinv2, __m128 &rinv3) {
  const __m128 half  = mm_set1<float, __m128>(0.5f);
  const __m128 three = mm_set1<float, __m128>(3.0f);
  __m128 lu1 = mm_rsqrt(rsq1);
  __m128 lu2 = mm_rsqrt(rsq2);
  __m128 lu3 = mm_rsqrt(rsq3);
  rinv1 = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu1,lu1),rsq1)),lu1));
  rinv2 = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu2,lu2),rsq2)),lu2));
  rinv3 = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu3,lu3),rsq3)),lu3));
}

inline __attribute__((always_inline))
void invsqrt(const __m128d rsq, __m128d &rinv) {
  const __m128d half  = mm_set1<double, __m128d>(0.5);
  const __m128d three = mm_set1<double, __m128d>(3.0);
  __m128d lu = mm_rsqrt(rsq);
  lu = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu,lu),rsq)),lu));
  rinv = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu,lu),rsq)),lu));
}

inline __attribute__((always_inline))
void invsqrt(const __m128 rsq, __m128 &rinv) {
  const __m128 half  = mm_set1<float, __m128>(0.5f);
  const __m128 three = mm_set1<float, __m128>(3.0f);
  __m128 lu = mm_rsqrt(rsq);
  rinv = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu,lu),rsq)),lu));
}

#ifdef __AVX__
inline __attribute__((always_inline))
__m256 mm_rsqrt(__m256 rsq) {
  return _mm256_rsqrt_ps(rsq);
}

inline __attribute__((always_inline))
__m256d mm_rsqrt(__m256d rsq) {
  return _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(rsq)));
}

//
// Calculates inverse square roots
// Two Newton iterations for double precision, one for single
//
inline __attribute__((always_inline))
void invsqrt(const __m256d rsq1, const __m256d rsq2, const __m256d rsq3,
	     __m256d &rinv1, __m256d &rinv2, __m256d &rinv3) {
  const __m256d half  = mm_set1<double, __m256d>(0.5);
  const __m256d three = mm_set1<double, __m256d>(3.0);
  __m256d lu1 = mm_rsqrt(rsq1);
  __m256d lu2 = mm_rsqrt(rsq2);
  __m256d lu3 = mm_rsqrt(rsq3);
  lu1 = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu1,lu1),rsq1)),lu1));
  lu2 = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu2,lu2),rsq2)),lu2));
  lu3 = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu3,lu3),rsq3)),lu3));
  rinv1 = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu1,lu1),rsq1)),lu1));
  rinv2 = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu2,lu2),rsq2)),lu2));
  rinv3 = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu3,lu3),rsq3)),lu3));
}

inline __attribute__((always_inline))
void invsqrt(const __m256 rsq1, const __m256 rsq2, const __m256 rsq3,
	     __m256 &rinv1, __m256 &rinv2, __m256 &rinv3) {
  const __m256 half  = mm_set1<float, __m256>(0.5f);
  const __m256 three = mm_set1<float, __m256>(3.0f);
  __m256 lu1 = mm_rsqrt(rsq1);
  __m256 lu2 = mm_rsqrt(rsq2);
  __m256 lu3 = mm_rsqrt(rsq3);
  rinv1 = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu1,lu1),rsq1)),lu1));
  rinv2 = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu2,lu2),rsq2)),lu2));
  rinv3 = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu3,lu3),rsq3)),lu3));
}

inline __attribute__((always_inline))
void invsqrt(const __m256d rsq, __m256d &rinv) {
  const __m256d half  = mm_set1<double, __m256d>(0.5);
  const __m256d three = mm_set1<double, __m256d>(3.0);
  __m256d lu = mm_rsqrt(rsq);
  lu = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu,lu),rsq)),lu));
  rinv = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu,lu),rsq)),lu));
}

inline __attribute__((always_inline))
void invsqrt(const __m256 rsq, __m256 &rinv) {
  const __m256 half  = mm_set1<float, __m256>(0.5f);
  const __m256 three = mm_set1<float, __m256>(3.0f);
  __m256 lu = mm_rsqrt(rsq);
  rinv = mm_mul(half,mm_mul(mm_sub(three,mm_mul(mm_mul(lu,lu),rsq)),lu));
}

#endif //__AVX__

template<typename T, typename VT, typename AVT>
float enb_core_vec<T, VT, AVT>::invsqrt_approx(const float xf_in) {
  float xf = xf_in;
  __m128 x = _mm_load1_ps(&xf);
  __m128 y;
  invsqrt(x, y);
  float yf;
  _mm_store_ss(&yf, y);
  return yf;
}

/*
void test_invsqrt_ps() {
  for (int i=1;i < 100;i++) {
    float xf = (float)i;
    __m128 x, y;
    x = _mm_load1_ps(&xf);
    invsqrt<float, __m128>(x, y);
    float yf;
    _mm_store_ss(&yf, y);
    double ref = 1.0/sqrt((double)i);
    double error = fabs(ref-(double)yf);
    printf("%f %20.10f %20.10lf %e\n",xf,yf,ref,error/ref);
  }
}
*/

inline __attribute__((always_inline))
void load_lookup(const double* pftable, const int *t,
		 __m128d &a1, __m128d &a2, __m128d &a3, __m128d &a4) {
  __m128d a1t = _mm_load_pd(&pftable[t[0]]);
  a2  = _mm_load_pd(&pftable[t[1]]);
  __m128d a3t = _mm_load_pd(&pftable[t[0]+2]);
  a4  = _mm_load_pd(&pftable[t[1]+2]);
  a1 = _mm_unpacklo_pd(a1t,a2);
  a2 = _mm_unpackhi_pd(a1t,a2);
  a3 = _mm_unpacklo_pd(a3t,a4);
  a4 = _mm_unpackhi_pd(a3t,a4);
}

inline __attribute__((always_inline))
void load_lookup(const float* pftable, const int *t,
		 __m128 &a1, __m128 &a2, __m128 &a3, __m128 &a4) {
  __m128 load0 = _mm_load_ps(&pftable[t[0]]);    // pftable[t0] pftable[t0+1] pftable[t0+2] pftable[t0+3]
  __m128 load1 = _mm_load_ps(&pftable[t[1]]);    // pftable[t1] pftable[t1+1] pftable[t1+2] pftable[t1+3]
  __m128 load2 = _mm_load_ps(&pftable[t[2]]);    // pftable[t2] pftable[t2+1] pftable[t2+2] pftable[t2+3]
  __m128 load3 = _mm_load_ps(&pftable[t[3]]);    // pftable[t3] pftable[t3+1] pftable[t3+2] pftable[t3+3]
  __m128 load01 = _mm_unpacklo_ps(load0, load1);     // pftable[t0] pftable[t1] pftable[t0+1] pftable[t1+1]
  __m128 load23 = _mm_unpacklo_ps(load2, load3);     // pftable[t2] pftable[t3] pftable[t2+1] pftable[t3+1]
  a1 = _mm_movelh_ps(load01, load23);       // pftable[t0]   pftable[t1]   pftable[t2]   pftable[t3]
  a2 = _mm_movehl_ps(load23, load01);       // pftable[t0+1] pftable[t1+1] pftable[t2+1] pftable[t3+1]
  load01 = _mm_unpackhi_ps(load0, load1);            // pftable[t0+2] pftable[t1+2] pftable[t0+3] pftable[t1+3]
  load23 = _mm_unpackhi_ps(load2, load3);            // pftable[t2+2] pftable[t3+2] pftable[t2+3] pftable[t3+3]
  a3 = _mm_movelh_ps(load01, load23);       // pftable[t0+2] pftable[t1+2] pftable[t2+2] pftable[t3+2]
  a4 = _mm_movehl_ps(load23, load01);       // pftable[t0+3] pftable[t1+3] pftable[t2+3] pftable[t3+3]
}

#ifdef __AVX__
inline __attribute__((always_inline))
void load_lookup(const double* pftable, const int *t,
		 __m256d &a1, __m256d &a2, __m256d &a3, __m256d &a4) {
  __m256d load1 = _mm256_loadu_pd(&pftable[t[0]]);       // pftable[t1] pftable[t1+1] pftable[t1+2] pftable[t1+3]
  __m256d load2 = _mm256_loadu_pd(&pftable[t[1]]);       // pftable[t2] pftable[t2+1] pftable[t2+2] pftable[t2+3]
  __m256d load3 = _mm256_loadu_pd(&pftable[t[2]]);       // pftable[t3] pftable[t3+1] pftable[t3+2] pftable[t3+3]
  __m256d load4 = _mm256_loadu_pd(&pftable[t[3]]);       // pftable[t4] pftable[t4+1] pftable[t4+2] pftable[t4+3]
  TRANSPOSE_4X4(load1, load2, load3, load4, a1, a2, a3, a4);
}

inline __attribute__((always_inline))
void load_lookup(const float* pftable, const int *t,
		 __m256 &a1, __m256 &a2, __m256 &a3, __m256 &a4) {
  __m128 load0, load1, load2, load3, load01, load23;
  load0 = _mm_load_ps(&pftable[t[0]]);    // pftable[t0] pftable[t0+1] pftable[t0+2] pftable[t0+3]
  load1 = _mm_load_ps(&pftable[t[1]]);    // pftable[t1] pftable[t1+1] pftable[t1+2] pftable[t1+3]
  load2 = _mm_load_ps(&pftable[t[2]]);    // pftable[t2] pftable[t2+1] pftable[t2+2] pftable[t2+3]
  load3 = _mm_load_ps(&pftable[t[3]]);    // pftable[t3] pftable[t3+1] pftable[t3+2] pftable[t3+3]
  load01 = _mm_unpacklo_ps(load0, load1);     // pftable[t0] pftable[t1] pftable[t0+1] pftable[t1+1]
  load23 = _mm_unpacklo_ps(load2, load3);     // pftable[t2] pftable[t3] pftable[t2+1] pftable[t3+1]
  __m128 a1_lo = _mm_movelh_ps(load01, load23);       // pftable[t0]   pftable[t1]   pftable[t2]   pftable[t3]
  __m128 a2_lo = _mm_movehl_ps(load23, load01);       // pftable[t0+1] pftable[t1+1] pftable[t2+1] pftable[t3+1]
  load01 = _mm_unpackhi_ps(load0, load1);            // pftable[t0+2] pftable[t1+2] pftable[t0+3] pftable[t1+3]
  load23 = _mm_unpackhi_ps(load2, load3);            // pftable[t2+2] pftable[t3+2] pftable[t2+3] pftable[t3+3]
  __m128 a3_lo = _mm_movelh_ps(load01, load23);       // pftable[t0+2] pftable[t1+2] pftable[t2+2] pftable[t3+2]
  __m128 a4_lo = _mm_movehl_ps(load23, load01);       // pftable[t0+3] pftable[t1+3] pftable[t2+3] pftable[t3+3]
  load0 = _mm_load_ps(&pftable[t[4]]);    // pftable[t0] pftable[t0+1] pftable[t0+2] pftable[t0+3]
  load1 = _mm_load_ps(&pftable[t[5]]);    // pftable[t1] pftable[t1+1] pftable[t1+2] pftable[t1+3]
  load2 = _mm_load_ps(&pftable[t[6]]);    // pftable[t2] pftable[t2+1] pftable[t2+2] pftable[t2+3]
  load3 = _mm_load_ps(&pftable[t[7]]);    // pftable[t3] pftable[t3+1] pftable[t3+2] pftable[t3+3]
  load01 = _mm_unpacklo_ps(load0, load1);     // pftable[t0] pftable[t1] pftable[t0+1] pftable[t1+1]
  load23 = _mm_unpacklo_ps(load2, load3);     // pftable[t2] pftable[t3] pftable[t2+1] pftable[t3+1]
  __m128 a1_hi = _mm_movelh_ps(load01, load23);       // pftable[t0]   pftable[t1]   pftable[t2]   pftable[t3]
  __m128 a2_hi = _mm_movehl_ps(load23, load01);       // pftable[t0+1] pftable[t1+1] pftable[t2+1] pftable[t3+1]
  load01 = _mm_unpackhi_ps(load0, load1);            // pftable[t0+2] pftable[t1+2] pftable[t0+3] pftable[t1+3]
  load23 = _mm_unpackhi_ps(load2, load3);            // pftable[t2+2] pftable[t3+2] pftable[t2+3] pftable[t3+3]
  __m128 a3_hi = _mm_movelh_ps(load01, load23);       // pftable[t0+2] pftable[t1+2] pftable[t2+2] pftable[t3+2]
  __m128 a4_hi = _mm_movehl_ps(load23, load01);       // pftable[t0+3] pftable[t1+3] pftable[t2+3] pftable[t3+3]
  a1 = _mm256_insertf128_ps(_mm256_castps128_ps256(a1_lo), a1_hi, 1);
  a2 = _mm256_insertf128_ps(_mm256_castps128_ps256(a2_lo), a2_hi, 1);
  a3 = _mm256_insertf128_ps(_mm256_castps128_ps256(a3_lo), a3_hi, 1);
  a4 = _mm256_insertf128_ps(_mm256_castps128_ps256(a4_lo), a4_hi, 1);
}

#endif //__AVX__

//
// Calculates the lookup energy and force
//
#define LOOKUP3_ENERGY_FORCE(T, VT, pftable, t,				\
			     ep1, ep2, ep3,				\
			     pref1, pref2,				\
			     pot, fij1, fij2, fij3)			\
  {									\
    const VT two   = mm_set1<T, VT>((T)2.0);				\
    const VT three = mm_set1<T, VT>((T)3.0);				\
    VT a1, a2, a3, a4;							\
    load_lookup(pftable, &t[0],        a1, a2, a3, a4);			\
    a3 = mm_mul(ep1, a3);						\
    a4 = mm_mul(ep1,mm_mul(ep1, a4));					\
    VT pott = mm_mul(pref1,mm_add(a1,mm_mul(ep1,mm_add(a2,mm_add(a3,a4))))); \
    pot = mm_reduce(pot, pott);						\
    fij1 = mm_add(fij1, mm_mul(pref1,mm_add(a2,mm_add(mm_mul(a3,two),mm_mul(a4,three))))); \
    load_lookup(pftable, &t[veclen],   a1, a2, a3, a4);			\
    a3 = mm_mul(ep2, a3);						\
    a4 = mm_mul(ep2,mm_mul(ep2, a4));					\
    pott = mm_mul(pref2,mm_add(a1,mm_mul(ep2,mm_add(a2,mm_add(a3,a4))))); \
    pot = mm_reduce(pot, pott);						\
    fij2 = mm_add(fij2, mm_mul(pref2,mm_add(a2,mm_add(mm_mul(a3,two),mm_mul(a4,three))))); \
    load_lookup(pftable, &t[veclen*2], a1, a2, a3, a4);			\
    a3 = mm_mul(ep3, a3);						\
    a4 = mm_mul(ep3,mm_mul(ep3, a4));					\
    pott = mm_mul(pref2,mm_add(a1,mm_mul(ep3,mm_add(a2,mm_add(a3,a4))))); \
    pot = mm_reduce(pot, pott);						\
    fij3 = mm_add(fij3, mm_mul(pref2,mm_add(a2,mm_add(mm_mul(a3,two),mm_mul(a4,three))))); \
  }

#define LOOKUP1_ENERGY_FORCE(T, VT, pftable, t,				\
			     ep, pref,					\
			     pot, fij)					\
  {									\
    const VT two   = mm_set1<T, VT>((T)2.0);				\
    const VT three = mm_set1<T, VT>((T)3.0);				\
    VT a1, a2, a3, a4;							\
    load_lookup(pftable, t, a1, a2, a3, a4);				\
    a3 = mm_mul(ep, a3);						\
    a4 = mm_mul(ep, mm_mul(ep, a4));					\
    VT pot1 = mm_mul(pref,mm_add(a1,mm_mul(ep,mm_add(a2,mm_add(a3,a4))))); \
    pot = mm_reduce(pot, pot1);						\
    fij = mm_add(fij,mm_mul(pref,mm_add(a2,mm_add(mm_mul(a3,two),mm_mul(a4,three))))); \
  }

//
// Calculates lookup index
//
inline __attribute__((always_inline))
void calc_lookup_index(const __m128d rs, const int tablen, __m128d &ep, int *t) {
  __m64 ti = _mm_cvttpd_pi32(rs);
  ep = mm_sub(rs, _mm_cvtpi32_pd(ti));
  t[0] = tablen*(_mm_cvtsi64_si32(ti) - 1);
  t[1] = tablen*(_mm_cvtsi64_si32(_mm_srli_si64(ti,32)) - 1);
}

inline __attribute__((always_inline))
void calc_lookup_index(const __m128 rs, const int tablen, __m128 &ep, int *t) {
  __m128i ti = _mm_cvttps_epi32(rs);                     // [(int)rs][(int)rs][(int)rs][(int)rs]
  ep = _mm_sub_ps(rs, _mm_cvtepi32_ps(ti));      // [ep][ep][ep][ep]
  t[0] = tablen*(_mm_cvtsi128_si32(ti) - 1);
  ti = _mm_srli_si128(ti,4);
  t[1] = tablen*(_mm_cvtsi128_si32(ti) - 1);
  ti = _mm_srli_si128(ti,4);
  t[2] = tablen*(_mm_cvtsi128_si32(ti) - 1);
  ti = _mm_srli_si128(ti,4);
  t[3] = tablen*(_mm_cvtsi128_si32(ti) - 1);
}

#ifdef __AVX__
inline __attribute__((always_inline))
void calc_lookup_index(const __m256d rs, const int tablen, __m256d &ep, int *t) {
  __m128i ti = _mm256_cvttpd_epi32(rs);
  ep = _mm256_sub_pd(rs, _mm256_cvtepi32_pd(ti));
  t[0] = tablen*(_mm_cvtsi128_si32(ti) - 1);
  ti = _mm_srli_si128(ti,4);
  t[1] = tablen*(_mm_cvtsi128_si32(ti) - 1);
  ti = _mm_srli_si128(ti,4);
  t[2] = tablen*(_mm_cvtsi128_si32(ti) - 1);
  ti = _mm_srli_si128(ti,4);
  t[3] = tablen*(_mm_cvtsi128_si32(ti) - 1);
}

inline __attribute__((always_inline))
void calc_lookup_index(const __m256 rs, const int tablen, __m256 &ep, int *t) {
  __m256i ti = _mm256_cvttps_epi32(rs);
  ep = _mm256_sub_ps(rs, _mm256_cvtepi32_ps(ti));

  t[0] = tablen*(_mm_cvtsi128_si32(_mm256_castsi256_si128(ti)) - 1);
  ti = _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(ti), (1<<0) + (2<<2) + (3<<4)));
  t[1] = tablen*(_mm_cvtsi128_si32(_mm256_castsi256_si128(ti)) - 1);
  ti = _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(ti), (1<<0) + (2<<2) + (3<<4)));
  t[2] = tablen*(_mm_cvtsi128_si32(_mm256_castsi256_si128(ti)) - 1);
  ti = _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(ti), (1<<0) + (2<<2) + (3<<4)));
  t[3] = tablen*(_mm_cvtsi128_si32(_mm256_castsi256_si128(ti)) - 1);
  ti = _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(ti), (1<<0) + (2<<2) + (3<<4)));

  ti = mm_movehl(ti);

  t[4] = tablen*(_mm_cvtsi128_si32(_mm256_castsi256_si128(ti)) - 1);
  ti = _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(ti), (1<<0) + (2<<2) + (3<<4)));
  t[5] = tablen*(_mm_cvtsi128_si32(_mm256_castsi256_si128(ti)) - 1);
  ti = _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(ti), (1<<0) + (2<<2) + (3<<4)));
  t[6] = tablen*(_mm_cvtsi128_si32(_mm256_castsi256_si128(ti)) - 1);
  ti = _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(ti), (1<<0) + (2<<2) + (3<<4)));
  t[7] = tablen*(_mm_cvtsi128_si32(_mm256_castsi256_si128(ti)) - 1);
  ti = _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(ti), (1<<0) + (2<<2) + (3<<4)));

}
#endif

#define CALC_V_FULL_KERNEL(T, VT, coul_type, vdw_type, rsq1, rsq2, rsq3, \
			   hinv, pftable,				\
			   aux,						\
			   q1, q2,					\
			   c6_1, c6_2,					\
			   c12_1, c12_2,				\
               c6mult_1, c6mult_2, ljpme_shift, \
			   coul, vdw,					\
			   fij1, fij2, fij3)				\
  {									\
    const VT mintwo = mm_set1<T,VT>((T)-2.0);				\
    const VT half  = mm_set1<T,VT>((T)0.5);				\
    const VT two   = mm_set1<T,VT>((T)2.0);				\
    const VT three = mm_set1<T,VT>((T)3.0);				\
    const VT six    = mm_set1<T,VT>((T)6.0);				\
    const VT twelve = mm_set1<T,VT>((T)12.0);				\
    const int tablen = ljpme_shift + ((coul_type==LOOKUP) ? (4 + ((vdw_type==LOOKUP) ? 8 : 0)) : ((vdw_type==LOOKUP) ? 8 : 0)); \
    VT rinv1, rinv2, rinv3;						\
    invsqrt(rsq1, rsq2, rsq3, rinv1, rinv2, rinv3);			\
    VT r1 = mm_mul(rinv1, rsq1);					\
    VT r2 = mm_mul(rinv2, rsq2);					\
    VT r3 = mm_mul(rinv3, rsq3);					\
    int t[veclen*3];							\
    VT ep1, ep2, ep3;							\
    if ((coul_type == LOOKUP) || (vdw_type == LOOKUP)) {		\
      calc_lookup_index(mm_mul(r1, hinv), tablen, ep1, &t[0]);		\
      calc_lookup_index(mm_mul(r2, hinv), tablen, ep2, &t[veclen]);	\
      calc_lookup_index(mm_mul(r3, hinv), tablen, ep3, &t[veclen*2]);	\
    }									\
    mm_setzero(fij1);							\
    mm_setzero(fij2);							\
    mm_setzero(fij3);							\
    if (coul_type != NONE) mm_setzero(coul);				\
    mm_setzero(vdw);							\
    if (coul_type == LOOKUP) {						\
      LOOKUP3_ENERGY_FORCE(T, VT, pftable, t, ep1, ep2, ep3, q1, q2, coul, fij1, fij2, fij3); \
      if (vdw_type == LOOKUP) {						\
	for (int i=0;i < veclen*3;i++) t[i] += 4;			\
      } else {								\
	fij1 = mm_mul(fij1, hinv);					\
	fij2 = mm_mul(fij2, hinv);					\
	fij3 = mm_mul(fij3, hinv);					\
      }									\
    }									\
    if (vdw_type == LOOKUP) {						\
      LOOKUP3_ENERGY_FORCE(T, VT, pftable, t, ep1, ep2, ep3, c6_1, c6_2, vdw, fij1, fij2, fij3); \
      for (int i=0;i < veclen*3;i++) t[i] += 4;				\
      LOOKUP3_ENERGY_FORCE(T, VT, pftable, t, ep1, ep2, ep3, c12_1, c12_2, vdw, fij1, fij2, fij3); \
      if(ljpme_shift == 4) {                                      \
        for (int i=0;i < veclen*3;i++) t[i] += 4;				\
        LOOKUP3_ENERGY_FORCE(T, VT, pftable, t, ep1, ep2, ep3, c6mult_1, c6mult_2, vdw, fij1, fij2, fij3); \
      }                                                                     \
      fij1 = mm_mul(fij1, mm_mul(rinv1, hinv));				\
      fij2 = mm_mul(fij2, mm_mul(rinv2, hinv));				\
      fij3 = mm_mul(fij3, mm_mul(rinv3, hinv));				\
    } else if (vdw_type == VSH) {					\
    } else if (vdw_type == VSW) {					\
    } else if (vdw_type == VFSW) {					\
    }									\
  }

//
// Calculates solute forces for the full vector
//
#define CALC_U_FULL_KERNEL(T, VT, coul_type, vdw_type,			\
			   rsq, hinv, pftable, aux,			\
			   q, c6, c12, c6multij,				\
			   coul, vdw, fij, ljpme_shift)				\
  {									\
    const VT mintwo = mm_set1<T,VT>((T)-2.0);				\
    const VT half  = mm_set1<T,VT>((T)0.5);				\
    const VT two   = mm_set1<T,VT>((T)2.0);				\
    const VT three = mm_set1<T,VT>((T)3.0);				\
    const VT six    = mm_set1<T,VT>((T)6.0);				\
    const VT twelve = mm_set1<T,VT>((T)12.0);				\
    const int tablen = ljpme_shift + ((coul_type==LOOKUP) ? (4 + ((vdw_type==LOOKUP) ? 8 : 0)) : ((vdw_type==LOOKUP) ? 8 : 0));	\
    VT rinv;								\
    invsqrt(rsq, rinv);							\
    VT r = mm_mul(rinv, rsq);						\
    int t[veclen];							\
    VT ep;								\
    if ((coul_type == LOOKUP) || (vdw_type == LOOKUP)) {		\
      calc_lookup_index(mm_mul(r, hinv), tablen, ep, &t[0]);		\
    }									\
    mm_setzero(fij);							\
    if (coul_type != NONE) mm_setzero(coul);				\
    mm_setzero(vdw);							\
    if (coul_type == LOOKUP) {						\
      LOOKUP1_ENERGY_FORCE(T, VT, pftable, t, ep, q, coul, fij);	\
      if (vdw_type == LOOKUP) {						\
	for (int i=0;i < veclen;i++) t[i] += 4;				\
      } else {								\
	fij = mm_mul(fij, hinv);					\
      }									\
    }									\
    if (vdw_type == LOOKUP) {						\
      LOOKUP1_ENERGY_FORCE(T, VT, pftable, t, ep, c6, vdw, fij);	\
      for (int i=0;i < veclen;i++) t[i] += 4;				\
      LOOKUP1_ENERGY_FORCE(T, VT, pftable, t, ep, c12, vdw, fij);	\
      if(ljpme_shift == 4) {     /* This is the LJPME term*/          \
        for (int i=0;i < veclen;i++) t[i] += 4;				        \
        LOOKUP1_ENERGY_FORCE(T, VT, pftable, t, ep, c6multij, vdw, fij);	\
      }                                                             \
      fij = mm_mul(fij, mm_mul(rinv, hinv));				\
    } else if (vdw_type == VSH) {					\
    } else if (vdw_type == VSW) {					\
    } else if (vdw_type == VFSW) {					\
    }									\
  }

inline __attribute__((always_inline))
void add_force_component(const __m128d fijx, __m128d &fx1, __m128d *fjx_d) {
  fx1 = mm_add(fx1, fijx);
  fjx_d[0] = mm_sub(fjx_d[0], fijx);
}

inline __attribute__((always_inline))
void add_force_component(const __m128 fijx, __m128d &fx1, __m128d *fjx_d) {
  // Split fijx into two double precision vectors: fijx01 and fijx23
  __m128d fijx01 = _mm_cvtps_pd(fijx);                     // [fijx0] [fijx1]
  __m128d fijx23 = _mm_cvtps_pd(_mm_movehl_ps(fijx,fijx)); // [fijx2] [fijx3]
  fx1 = mm_add(fx1, _mm_add_pd(fijx01, fijx23));
  fjx_d[0] = mm_sub(fjx_d[0], fijx01);
  fjx_d[1] = mm_sub(fjx_d[1], fijx23);
}

#ifdef __AVX__
inline __attribute__((always_inline))
void add_force_component(const __m256d fijx, __m256d &fx1, __m256d *fjx_d) {
  fx1 = mm_add(fx1, fijx);
  fjx_d[0] = mm_sub(fjx_d[0], fijx);
}

inline __attribute__((always_inline))
void add_force_component(const __m256 fijx, __m256d &fx1, __m256d *fjx_d) {
  // Split fijx into two double precision vectors
  // [fijx0] [fijx1] [fijx2] [fijx3]
  __m256d fijx03 = _mm256_cvtps_pd(_mm256_castps256_ps128(fijx));
  // [fijx4] [fijx5] [fijx6] [fijx7]
  __m256d fijx47 = _mm256_cvtps_pd(_mm256_castps256_ps128(mm_movehl(fijx)));
  fx1 = mm_add(fx1, mm_add(fijx03, fijx47));
  fjx_d[0] = mm_sub(fjx_d[0], fijx03);
  fjx_d[1] = mm_sub(fjx_d[1], fijx47);
}
#endif

#define ADD_V_FORCE_FULL(T, VT, AVT, dx1, dx2, dx3,			\
			 ind, t,					\
			 fij1, fij2, fij3,				\
			 force, fx1, fx2, fx3)				\
  {									\
    const int n = sizeof(double)/sizeof(T);				\
    VT fijx;								\
    AVT fjx_d[n];							\
    for (int i=0;i < n;i++)						\
      load_force(force, ind, i, t, fjx_d[i]);				\
    fijx = mm_mul(dx1, fij1);						\
    add_force_component(fijx, fx1, fjx_d);				\
    fijx = mm_mul(dx2, fij2);						\
    add_force_component(fijx, fx2, fjx_d);				\
    fijx = mm_mul(dx3, fij3);						\
    add_force_component(fijx, fx3, fjx_d);				\
    for (int i=n-1;i >= 0;i--)						\
      store_force(force, ind, i, t, fjx_d[i]);				\
  }

//
// Adds forces
// t = 0 (x), 1 (y), 2 (z)
//
#define ADD_U_FORCE_FULL(T, VT, AVT, dx, ind, t, fij, force, fxi)	\
  {									\
    const int n = sizeof(double)/sizeof(T);				\
    VT fijx;								\
    AVT fjx_d[n];							\
    for (int i=0;i < n;i++)						\
      load_force(force, ind, i, t, fjx_d[i]);				\
    fijx = mm_mul(dx, fij);						\
    add_force_component(fijx, fxi, fjx_d);				\
    for (int i=n-1;i >= 0;i--)						\
      store_force(force, ind, i, t, fjx_d[i]);				\
  }

#define CALC_V_FULL(T, VT, AVT, coul_type, vdw_type,			\
		    x1, y1, z1, x2, y2, z2, x3, y3, z3, xj, yj, zj,	\
		    rcutsq, hinv, pftable, aux,				\
		    q1, q2, c6_1, c6_2,	c12_1, c12_2,       \
            c6mult_1, c6mult_2, ljpme_shift,   \
            coulpot, vdwpot,	\
		    jj, force,						\
		    fx1, fy1, fz1,					\
		    fx2, fy2, fz2,					\
		    fx3, fy3, fz3)					\
  {									\
    VT dx1 = mm_sub(x1, xj);						\
    VT dy1 = mm_sub(y1, yj);						\
    VT dz1 = mm_sub(z1, zj);						\
    VT dx2 = mm_sub(x2, xj);						\
    VT dy2 = mm_sub(y2, yj);						\
    VT dz2 = mm_sub(z2, zj);						\
    VT dx3 = mm_sub(x3, xj);						\
    VT dy3 = mm_sub(y3, yj);						\
    VT dz3 = mm_sub(z3, zj);						\
    const VT rsq1 = mm_add(mm_mul(dx1,dx1),mm_add(mm_mul(dy1,dy1),mm_mul(dz1,dz1))); \
    const VT rsq2 = mm_add(mm_mul(dx2,dx2),mm_add(mm_mul(dy2,dy2),mm_mul(dz2,dz2))); \
    const VT rsq3 = mm_add(mm_mul(dx3,dx3),mm_add(mm_mul(dy3,dy3),mm_mul(dz3,dz3))); \
    VT minrsq = mm_min(mm_min(rsq1,rsq2),rsq3);				\
    VT cmp_mask = mm_cmplt(minrsq, rcutsq);				\
    if (mm_movemask(cmp_mask)) {					\
      AVT coul, vdw;							\
      VT fij1, fij2, fij3;						\
      VT q1_mask = mm_and(q1, cmp_mask);				\
      VT q2_mask = mm_and(q2, cmp_mask);				\
      VT c6_1_mask = mm_and(c6_1, cmp_mask);				\
      VT c6_2_mask = mm_and(c6_2, cmp_mask);				\
      VT c12_1_mask = mm_and(c12_1, cmp_mask);				\
      VT c12_2_mask = mm_and(c12_2, cmp_mask);				\
      VT c6mult_1_mask, c6mult_2_mask;                      \
      if(ljpme_shift == 4) {                                \
        c6mult_1_mask = mm_and(c6mult_1, cmp_mask);      	\
        c6mult_2_mask = mm_and(c6mult_2, cmp_mask);	         \
      }                                                      \
      CALC_V_FULL_KERNEL(T, VT, coul_type, vdw_type,			\
			 rsq1, rsq2, rsq3,				\
			 hinv, pftable, aux,				\
			 q1_mask, q2_mask,				\
			 c6_1_mask, c6_2_mask,				\
			 c12_1_mask, c12_2_mask,			\
             c6mult_1_mask, c6mult_2_mask, ljpme_shift, \
			 coul, vdw,					\
			 fij1, fij2, fij3);				\
      coulpot = mm_add(coulpot, coul);					\
      vdwpot  = mm_add(vdwpot,  vdw);					\
      ADD_V_FORCE_FULL(T, VT, AVT, dx1, dx2, dx3, jj, 0, fij1, fij2, fij3, force, fx1, fx2, fx3); \
      ADD_V_FORCE_FULL(T, VT, AVT, dy1, dy2, dy3, jj, 1, fij1, fij2, fij3, force, fy1, fy2, fy3); \
      ADD_V_FORCE_FULL(T, VT, AVT, dz1, dz2, dz3, jj, 2, fij1, fij2, fij3, force, fz1, fz2, fz3); \
    }									\
  }

//
// Calculates energy and force using the full vector length
//
#define CALC_UV_FULL(T, VT, AVT, coul_type, vdw_type,			\
		     x1, y1, z1,					\
		     x2, y2, z2,					\
		     x3, y3, z3,					\
		     xj, yj, zj,					\
		     rcutsq, hinv,					\
		     pftable, aux,					\
		     qO, qH, xyzq, iaO, iaH,				\
		     vdwtype, vdwparam,					\
		     coulpot, vdwpot,					\
		     jj, force,						\
		     fx1, fy1, fz1,					\
		     fx2, fy2, fz2,					\
		     fx3, fy3, fz3)					\
  {									\
    VT dx1 = mm_sub(x1, xj);						\
    VT dy1 = mm_sub(y1, yj);						\
    VT dz1 = mm_sub(z1, zj);						\
    VT dx2 = mm_sub(x2, xj);						\
    VT dy2 = mm_sub(y2, yj);						\
    VT dz2 = mm_sub(z2, zj);						\
    VT dx3 = mm_sub(x3, xj);						\
    VT dy3 = mm_sub(y3, yj);						\
    VT dz3 = mm_sub(z3, zj);						\
    const VT rsq1 = mm_add(mm_mul(dx1,dx1),mm_add(mm_mul(dy1,dy1),mm_mul(dz1,dz1))); \
    const VT rsq2 = mm_add(mm_mul(dx2,dx2),mm_add(mm_mul(dy2,dy2),mm_mul(dz2,dz2))); \
    const VT rsq3 = mm_add(mm_mul(dx3,dx3),mm_add(mm_mul(dy3,dy3),mm_mul(dz3,dz3))); \
    VT minrsq = mm_min(mm_min(rsq1,rsq2),rsq3);				\
    VT cmp_mask = mm_cmplt(minrsq, rcutsq);				\
    if (mm_movemask(cmp_mask)) {					\
      int ivdwO[veclen];						\
      int ivdwH[veclen];						\
      for (int i=0;i < veclen;i++) {					\
	int ja = vdwtype[jj[i]];					\
	int aa = (ja > iaO) ? ja : iaO;					\
	ivdwO[i] = aa*(aa-3) + 2*(ja + iaO) - 2;			\
	aa = (ja > iaH) ? ja : iaH;					\
	ivdwH[i] = aa*(aa-3) + 2*(ja + iaH) - 2;			\
      }									\
      VT c6jO, c12jO;							\
      load_vdwparam(vdwparam, ivdwO, c6jO, c12jO);			\
      VT c6jH, c12jH;							\
      load_vdwparam(vdwparam, ivdwH, c6jH, c12jH);			\
      c6jO = mm_and(c6jO, cmp_mask);					\
      c6jH = mm_and(c6jH, cmp_mask);					\
      c12jO = mm_and(c12jO, cmp_mask);					\
      c12jH = mm_and(c12jH, cmp_mask);					\
      VT qjO, qjH;							\
      if (coul_type != NONE) {						\
	VT qj;								\
	load_charge(xyzq, jj, qj);					\
	qjO = mm_mul(qj, qO);						\
	qjH = mm_mul(qj, qH);						\
	qjO = mm_and(qjO, cmp_mask);					\
	qjH = mm_and(qjH, cmp_mask);					\
      }									\
      AVT coul, vdw;							\
      VT fij1, fij2, fij3;						\
      CALC_V_FULL_KERNEL(T, VT, coul_type, vdw_type,			\
			 rsq1, rsq2, rsq3,				\
			 hinv, pftable, aux,				\
			 qjO, qjH,					\
			 c6jO, c6jH,					\
			 c12jO, c12jH,					\
			 coul, vdw,					\
			 fij1, fij2, fij3);				\
      if (coul_type != NONE) coulpot = mm_add(coulpot, coul);		\
      vdwpot  = mm_add(vdwpot,  vdw);					\
      ADD_V_FORCE_FULL(T, VT, AVT, dx1, dx2, dx3, jj, 0, fij1, fij2, fij3, force, fx1, fx2, fx3); \
      ADD_V_FORCE_FULL(T, VT, AVT, dy1, dy2, dy3, jj, 1, fij1, fij2, fij3, force, fy1, fy2, fy3); \
      ADD_V_FORCE_FULL(T, VT, AVT, dz1, dz2, dz3, jj, 2, fij1, fij2, fij3, force, fz1, fz2, fz3); \
    }									\
  }

#define CALC_UV_MASK(T, VT, AVT, coul_type, vdw_type,			\
		     x1, y1, z1,					\
		     x2, y2, z2,					\
		     x3, y3, z3,					\
		     xj, yj, zj, mask,					\
		     rcutsq, hinv,					\
		     pftable, aux,					\
		     qO, qH, xyzq, iaO, iaH,				\
		     vdwtype, vdwparam,					\
		     coulpot, vdwpot,					\
		     jj, force,						\
		     fx1, fy1, fz1,					\
		     fx2, fy2, fz2,					\
		     fx3, fy3, fz3)					\
  {									\
    VT dx1 = mm_sub(x1, xj);						\
    VT dy1 = mm_sub(y1, yj);						\
    VT dz1 = mm_sub(z1, zj);						\
    VT dx2 = mm_sub(x2, xj);						\
    VT dy2 = mm_sub(y2, yj);						\
    VT dz2 = mm_sub(z2, zj);						\
    VT dx3 = mm_sub(x3, xj);						\
    VT dy3 = mm_sub(y3, yj);						\
    VT dz3 = mm_sub(z3, zj);						\
    const VT rsq1 = mm_add(mm_mul(dx1,dx1),mm_add(mm_mul(dy1,dy1),mm_mul(dz1,dz1))); \
    const VT rsq2 = mm_add(mm_mul(dx2,dx2),mm_add(mm_mul(dy2,dy2),mm_mul(dz2,dz2))); \
    const VT rsq3 = mm_add(mm_mul(dx3,dx3),mm_add(mm_mul(dy3,dy3),mm_mul(dz3,dz3))); \
    VT minrsq = mm_min(mm_min(rsq1,rsq2),rsq3);				\
    mask = mm_and(mask, mm_cmplt(minrsq, rcutsq));			\
    if (mm_movemask(mask)) {						\
      int ivdwO[veclen];						\
      int ivdwH[veclen];						\
      for (int i=0;i < veclen;i++) {					\
	int ja = vdwtype[jj[i]];					\
	int aa = (ja > iaO) ? ja : iaO;					\
	ivdwO[i] = aa*(aa-3) + 2*(ja + iaO) - 2;			\
	aa = (ja > iaH) ? ja : iaH;					\
	ivdwH[i] = aa*(aa-3) + 2*(ja + iaH) - 2;			\
      }									\
      VT c6jO, c12jO;							\
      load_vdwparam(vdwparam, ivdwO, c6jO, c12jO);			\
      VT c6jH, c12jH;							\
      load_vdwparam(vdwparam, ivdwH, c6jH, c12jH);			\
      c6jO  = mm_and(c6jO, mask);					\
      c12jO = mm_and(c12jO, mask);					\
      c6jH  = mm_and(c6jH, mask);					\
      c12jH = mm_and(c12jH, mask);					\
      VT qjO, qjH;							\
      if (coul_type != NONE) {						\
	VT qj;								\
	load_charge(xyzq, jj, qj);					\
	qjO = mm_mul(qj, qO);						\
	qjH = mm_mul(qj, qH);						\
	qjO = mm_and(qjO, mask);					\
	qjH = mm_and(qjH, mask);					\
      }									\
      AVT coul, vdw;							\
      VT fij1, fij2, fij3;						\
      CALC_V_FULL_KERNEL(T, VT, coul_type, vdw_type,			\
			 rsq1, rsq2, rsq3,				\
			 hinv, pftable, aux,				\
			 qjO, qjH,					\
			 c6jO, c6jH,					\
			 c12jO, c12jH,					\
			 coul, vdw,					\
			 fij1, fij2, fij3);				\
      if (coul_type != NONE) coulpot = mm_add(coulpot, coul);		\
      vdwpot  = mm_add(vdwpot,  vdw);					\
      ADD_V_FORCE_FULL(T, VT, AVT, dx1, dx2, dx3, jj, 0, fij1, fij2, fij3, force, fx1, fx2, fx3); \
      ADD_V_FORCE_FULL(T, VT, AVT, dy1, dy2, dy3, jj, 1, fij1, fij2, fij3, force, fy1, fy2, fy3); \
      ADD_V_FORCE_FULL(T, VT, AVT, dz1, dz2, dz3, jj, 2, fij1, fij2, fij3, force, fz1, fz2, fz3); \
    }									\
  }

//
// Calculates energy and force using the full vector length
//
#define CALC_U_FULL(T, VT, AVT, coul_type, vdw_type,			\
		    xi, yi, zi, xj, yj, zj,				\
		    rcutsq, hinv,					\
		    pftable, aux,					\
		    xyzq, ia,						\
		    vdwtype, vdwparam,					\
		    coulpot, vdwpot,					\
		    jj, force,						\
		    fxi, fyi, fzi)					\
  {									\
    VT dx = mm_sub(xi, xj);						\
    VT dy = mm_sub(yi, yj);						\
    VT dz = mm_sub(zi, zj);						\
    const VT rsq = mm_add(mm_mul(dx,dx),mm_add(mm_mul(dy,dy),mm_mul(dz,dz))); \
    VT cmp_mask = mm_cmplt(rsq, rcutsq);				\
    if (mm_movemask(cmp_mask)) {					\
      int ivdw[veclen];							\
      for (int i=0;i < veclen;i++) {					\
	int ja = vdwtype[jj[i]];					\
	int aa = (ja > ia) ? ja : ia;					\
	ivdw[i] = aa*(aa-3) + 2*(ja + ia) - 2;				\
      }									\
      VT c6, c12;							\
      load_vdwparam(vdwparam, ivdw, c6, c12);				\
      c6 = mm_and(c6, cmp_mask);					\
      c12 = mm_and(c12, cmp_mask);					\
      VT qij;								\
      if (coul_type != NONE) {						\
	VT qj;								\
	load_charge(xyzq, jj, qj);					\
	qij = mm_mul(qj, qi);						\
	qij = mm_and(qij, cmp_mask);					\
      }									\
      AVT coul, vdw;							\
      VT fij;								\
      CALC_U_FULL_KERNEL(T, VT, coul_type, vdw_type,			\
			 rsq, hinv, pftable, aux,			\
			 qij, c6, c12, coul, vdw, fij);			\
      if (coul_type != NONE) coulpot = mm_add(coulpot, coul);		\
      vdwpot  = mm_add(vdwpot,  vdw);					\
      ADD_U_FORCE_FULL(T, VT, AVT, dx, jj, 0, fij, force, fxi);		\
      ADD_U_FORCE_FULL(T, VT, AVT, dy, jj, 1, fij, force, fyi);		\
      ADD_U_FORCE_FULL(T, VT, AVT, dz, jj, 2, fij, force, fzi);		\
    }									\
  }

//      fprintf(stderr,"%d %d => %d\n",imask,jleft,imask & ((1 << jleft)-1)); \
//      nnnb += __builtin_popcount(imask & ((1 << jleft)-1));		\

#define CALC_U_MASK(T, VT, AVT, coul_type, vdw_type,			\
		    xi, yi, zi, xj, yj, zj, mask,			\
		    rcutsq, hinv,					\
		    pftable, aux,					\
		    xyzq, ia,						\
		    vdwtype, vdwparam,					\
		    coulpot, vdwpot,					\
		    jj, force,						\
		    fxi, fyi, fzi)					\
  {									\
    VT dx = mm_sub(xi, xj);						\
    VT dy = mm_sub(yi, yj);						\
    VT dz = mm_sub(zi, zj);						\
    const VT rsq = mm_add(mm_mul(dx,dx),mm_add(mm_mul(dy,dy),mm_mul(dz,dz))); \
    mask = mm_and(mask, mm_cmplt(rsq, rcutsq));				\
    if (mm_movemask(mask)) {						\
      int ivdw[veclen];							\
      for (int i=0;i < veclen;i++) {					\
	int ja = vdwtype[jj[i]];					\
	int aa = (ja > ia) ? ja : ia;					\
	ivdw[i] = aa*(aa-3) + 2*(ja + ia) - 2;				\
      }									\
      VT c6, c12;							\
      load_vdwparam(vdwparam, ivdw, c6, c12);				\
      c6  = mm_and(c6, mask);						\
      c12 = mm_and(c12, mask);						\
      VT qij;								\
      if (coul_type != NONE) {						\
	VT qj;								\
	load_charge(xyzq, jj, qj);					\
	qij = mm_mul(qj, qi);						\
	qij = mm_and(qij, mask);					\
      }									\
      AVT coul, vdw;							\
      VT fij;								\
      CALC_U_FULL_KERNEL(T, VT, coul_type, vdw_type,			\
			 rsq, hinv, pftable, aux,			\
			 qij, c6, c12, coul, vdw, fij);			\
      if (coul_type != NONE) coulpot = mm_add(coulpot, coul);		\
      vdwpot  = mm_add(vdwpot,  vdw);					\
      ADD_U_FORCE_FULL(T, VT, AVT, dx, jj, 0, fij, force, fxi);		\
      ADD_U_FORCE_FULL(T, VT, AVT, dy, jj, 1, fij, force, fyi);		\
      ADD_U_FORCE_FULL(T, VT, AVT, dz, jj, 2, fij, force, fzi);		\
    }									\
  }

template <typename AVT>
inline __attribute__((always_inline))
void store_v_force_full(const AVT fx1, const AVT fy1, const AVT fz1,
			const AVT fx2, const AVT fy2, const AVT fz2,
			const AVT fx3, const AVT fy3, const AVT fz3,
			double *force, const int ii, double *sforce, const int is) {
  double fx1d, fy1d, fz1d, fx2d, fy2d, fz2d, fx3d, fy3d, fz3d;
  mm_store_sumall(&fx1d, fx1);
  mm_store_sumall(&fy1d, fy1);
  mm_store_sumall(&fz1d, fz1);
  mm_store_sumall(&fx2d, fx2);
  mm_store_sumall(&fy2d, fy2);
  mm_store_sumall(&fz2d, fz2);
  mm_store_sumall(&fx3d, fx3);
  mm_store_sumall(&fy3d, fy3);
  mm_store_sumall(&fz3d, fz3);
  force[ii]   += fx1d;
  force[ii+1] += fy1d;
  force[ii+2] += fz1d;
  force[ii+3] += fx2d;
  force[ii+4] += fy2d;
  force[ii+5] += fz2d;
  force[ii+6] += fx3d;
  force[ii+7] += fy3d;
  force[ii+8] += fz3d;
  sforce[is]   += fx1d + fx2d + fx3d;
  sforce[is+1] += fy1d + fy2d + fy3d;
  sforce[is+2] += fz1d + fz2d + fz3d;
}

template <typename AVT>
inline __attribute__((always_inline))
void store_u_force_full(const AVT fx1, const AVT fy1, const AVT fz1,
			double *force, const int ii, double *sforce, const int is) {
  double fx1d, fy1d, fz1d;
  mm_store_sumall(&fx1d, fx1);
  mm_store_sumall(&fy1d, fy1);
  mm_store_sumall(&fz1d, fz1);
  force[ii]   += fx1d;
  force[ii+1] += fy1d;
  force[ii+2] += fz1d;
  sforce[is]   += fx1d;
  sforce[is+1] += fy1d;
  sforce[is+2] += fz1d;
}

//
// Setup VdW constants
//
template<typename T, typename VT>
void setup_vdw_const(const int vdw_type, const T roff, const T ron, VT *aux) {
  if (vdw_type == LOOKUP) {
  } else if (vdw_type == VSH) {
    T auxd[7];
    vdw_const_vsh<T>(roff, auxd);
    for (int i=0;i < 7;i++) aux[i] = mm_set1<T,VT>(auxd[i]);
  } else if (vdw_type == VSW) {
    T auxd[5];
    vdw_const_vsw<T>(roff, ron, auxd);
    for (int i=0;i < 5;i++) aux[i] = mm_set1<T,VT>(auxd[i]);
  } else if (vdw_type == VFSW) {
    T auxd[8];
    vdw_const_vfsw<T>(roff, ron, auxd);
    for (int i=0;i < 8;i++) aux[i] = mm_set1<T,VT>(auxd[i]);
  }
}

//
// Solvent - Solvent interactions
//
template <typename T, typename VT, typename AVT>
template <int coul_type, int vdw_type>
void enb_core_vec<T, VT, AVT>::enb_vv(const int ni, const int *indi, const int *indj, const int *startj,
				      const T *pftable, const T hinv,
				      const xyzq_t<T> *xyzq, double *force, const int *iscoord,
				      double *sforce, const T *scoordtab,
				      double *coulpot, double *vdwpot, 
				      const T c6OO, const T c12OO, const T c6OH, const T c12OH, const T c6HH, const T c12HH, 
#if KEY_LJPME==1
				      const T c6multOO, const T c6multOH, const T c6multHH,
#endif
                      const T qOO, const T qOH, const T qHH,
				      const T roff, const T ron) {
  if (coul_type != LOOKUP) {
    std::cerr << "void enb_core_vec<T, VT>::enb_vv Requires coul_type = LOOKUP" << std::endl;
    exit(1);
  }

  if (vdw_type != LOOKUP) {
    std::cerr << "void enb_core_vec<T, VT>::enb_vv Requires vdw_type = LOOKUP" << std::endl;
    exit(1);
  }

  VT aux[8];
  setup_vdw_const<T,VT>(vdw_type, roff, ron, aux);

  VT rcsq = mm_set1<T,VT>(roff*roff);

  VT qOOd   = mm_set1<T,VT>(qOO);
  VT c6OOd  = mm_set1<T,VT>(c6OO);
  VT c12OOd = mm_set1<T,VT>(c12OO);

  VT qOHd   = mm_set1<T,VT>(qOH);
  VT c6OHd  = mm_set1<T,VT>(c6OH);
  VT c12OHd = mm_set1<T,VT>(c12OH);

  VT qHHd   = mm_set1<T,VT>(qHH);
  VT c6HHd  = mm_set1<T,VT>(c6HH);
  VT c12HHd = mm_set1<T,VT>(c12HH);

  VT c6multOOd, c6multOHd, c6multHHd; // Undefined, and unused, if no LJPME
#if KEY_LJPME==1
  c6multOOd = mm_set1<T,VT>(c6multOO);
  c6multOHd = mm_set1<T,VT>(c6multOH);
  c6multHHd = mm_set1<T,VT>(c6multHH);
  int ljpme_shift = 4;
#else
  int ljpme_shift = 0;
#endif

  VT hinvd  = mm_set1<T,VT>(hinv);

  AVT coulpotd = mm_load_one<double, AVT>(coulpot);
  AVT vdwpotd  = mm_load_one<double, AVT>(vdwpot);

  int i;
#pragma omp barrier
#pragma omp for private(i)
  for (i=0;i < ni;i++) {
    // Shift coordinates
    int is = iscoord[i] - 1;
    VT sx = mm_load_one2all<T,VT>(&scoordtab[is]);
    VT sy = mm_load_one2all<T,VT>(&scoordtab[is+1]);
    VT sz = mm_load_one2all<T,VT>(&scoordtab[is+2]);
    // Atom i coordinates
    int ii = indi[i] - 1;
    // Coordinates for solvent i
    VT xO = mm_add(sx,  mm_load_one2all<T,VT>(&xyzq[ii].x));
    VT yO = mm_add(sy,  mm_load_one2all<T,VT>(&xyzq[ii].y));
    VT zO = mm_add(sz,  mm_load_one2all<T,VT>(&xyzq[ii].z));
    VT xH1 = mm_add(sx, mm_load_one2all<T,VT>(&xyzq[ii+1].x));
    VT yH1 = mm_add(sy, mm_load_one2all<T,VT>(&xyzq[ii+1].y));
    VT zH1 = mm_add(sz, mm_load_one2all<T,VT>(&xyzq[ii+1].z));
    VT xH2 = mm_add(sx, mm_load_one2all<T,VT>(&xyzq[ii+2].x));
    VT yH2 = mm_add(sy, mm_load_one2all<T,VT>(&xyzq[ii+2].y));
    VT zH2 = mm_add(sz, mm_load_one2all<T,VT>(&xyzq[ii+2].z));

    // j solvent loop limits
    int j = startj[i] - 1;
    int j1 = startj[i+1] - 1;

    // Zero solvent i forces
    AVT fxO, fyO, fzO;
    AVT fxH1, fyH1, fzH1;
    AVT fxH2, fyH2, fzH2;

    mm_setzero(fxO);
    mm_setzero(fyO);
    mm_setzero(fzO);
    mm_setzero(fxH1);
    mm_setzero(fyH1);
    mm_setzero(fzH1);
    mm_setzero(fxH2);
    mm_setzero(fyH2);
    mm_setzero(fzH2);

    for (;j < j1-(veclen-1);j += veclen) {
      int jj[veclen];
#pragma unroll
      for (int t=0;t < veclen;t++) jj[t] = indj[j+t] - 1;
      
      VT xj, yj, zj;
      
      // j O interaction with i O H1 H2
      mm_load_xyz(xyzq, jj, xj, yj, zj);
      CALC_V_FULL(T, VT, AVT, coul_type, vdw_type,
		  xO, yO, zO, xH1, yH1, zH1, xH2, yH2, zH2, xj, yj, zj,
		  rcsq, hinvd, pftable, aux, 
		  qOOd, qOHd, c6OOd, c6OHd, c12OOd, c12OHd, c6multOOd, c6multOHd, ljpme_shift,
		  coulpotd, vdwpotd, jj, force,
		  fxO, fyO, fzO, fxH1, fyH1, fzH1, fxH2, fyH2, fzH2);

      // j H1 interaction with i O H1 H2
#pragma unroll
      for (int t=0;t < veclen;t++) jj[t]++;
      mm_load_xyz(xyzq, jj, xj, yj, zj);
      CALC_V_FULL(T, VT, AVT, coul_type, vdw_type,
		  xO, yO, zO, xH1, yH1, zH1, xH2, yH2, zH2, xj, yj, zj,
		  rcsq, hinvd, pftable, aux, 
		  qOHd, qHHd, c6OHd, c6HHd, c12OHd, c12HHd, c6multOHd, c6multHHd, ljpme_shift,
		  coulpotd, vdwpotd, jj, force,
		  fxO, fyO, fzO, fxH1, fyH1, fzH1, fxH2, fyH2, fzH2);

      // j H2 interaction with i O H1 H2
#pragma unroll
      for (int t=0;t < veclen;t++) jj[t]++;
      mm_load_xyz(xyzq, jj, xj, yj, zj);
      CALC_V_FULL(T, VT, AVT, coul_type, vdw_type,
		  xO, yO, zO, xH1, yH1, zH1, xH2, yH2, zH2, xj, yj, zj,
		  rcsq, hinvd, pftable, aux, 
		  qOHd, qHHd, c6OHd, c6HHd, c12OHd, c12HHd, c6multOHd, c6multHHd, ljpme_shift,
		  coulpotd, vdwpotd, jj, force,
		  fxO, fyO, fzO, fxH1, fyH1, fzH1, fxH2, fyH2, fzH2);

    }

    if (j != j1) {
      if (j > j1) j -= veclen;    // In case we stepped over limit, step back by one
      int jleft = j1 - j;

      int jj[veclen];
      int t;
      for (t=0;t < jleft;t++) jj[t] = indj[j+t] - 1;
      for (;t < veclen;t++)   jj[t] = indj[j] - 1;    // Set the rest of jj to point to the first one

      VT mask;
      get_mask(mask, jleft);

      VT qOOm   = mm_and(qOOd, mask);
      VT c6OOm  = mm_and(c6OOd, mask);
      VT c12OOm = mm_and(c12OOd, mask);

      VT qOHm   = mm_and(qOHd, mask);
      VT qHHm   = mm_and(qHHd, mask);
      VT c6OHm  = mm_and(c6OHd, mask);
      VT c6HHm  = mm_and(c6HHd, mask);
      VT c12OHm = mm_and(c12OHd, mask);
      VT c12HHm = mm_and(c12HHd, mask);

#if KEY_LJPME==1
      VT c6multOOm   = mm_and(c6multOOd, mask);
      VT c6multOHm   = mm_and(c6multOHd, mask);
      VT c6multHHm   = mm_and(c6multHHd, mask);
#else
      VT c6multOOm, c6multOHm, c6multHHm;
#endif

      VT xj, yj, zj;
      // j O interaction with i O H1 H2
      mm_load_xyz(xyzq, jj, xj, yj, zj);
      CALC_V_FULL(T, VT, AVT, coul_type, vdw_type,
		  xO, yO, zO, xH1, yH1, zH1, xH2, yH2, zH2, xj, yj, zj,
		  rcsq, hinvd, pftable, aux, 
		  qOOm, qOHm, c6OOm, c6OHm, c12OOm, c12OHm, c6multOOm, c6multOHm, ljpme_shift,
		  coulpotd, vdwpotd, jj, force,
		  fxO, fyO, fzO, fxH1, fyH1, fzH1, fxH2, fyH2, fzH2);

      // j H1 interaction with i O H1 H2
#pragma unroll
      for (int t=0;t < veclen;t++) jj[t]++;
      mm_load_xyz(xyzq, jj, xj, yj, zj);
      CALC_V_FULL(T, VT, AVT, coul_type, vdw_type,
		  xO, yO, zO, xH1, yH1, zH1, xH2, yH2, zH2, xj, yj, zj,
		  rcsq, hinvd, pftable, aux, 
		  qOHm, qHHm, c6OHm, c6HHm, c12OHm, c12HHm, c6multOHm, c6multHHm, ljpme_shift,
		  coulpotd, vdwpotd, jj, force,
		  fxO, fyO, fzO, fxH1, fyH1, fzH1, fxH2, fyH2, fzH2);

      // j H2 interaction with i O H1 H2
#pragma unroll
      for (int t=0;t < veclen;t++) jj[t]++;
      mm_load_xyz(xyzq, jj, xj, yj, zj);
      CALC_V_FULL(T, VT, AVT, coul_type, vdw_type,
		  xO, yO, zO, xH1, yH1, zH1, xH2, yH2, zH2, xj, yj, zj,
		  rcsq, hinvd, pftable, aux, 
		  qOHm, qHHm, c6OHm, c6HHm, c12OHm, c12HHm, c6multOHm, c6multHHm, ljpme_shift,
		  coulpotd, vdwpotd, jj, force,
		  fxO, fyO, fzO, fxH1, fyH1, fzH1, fxH2, fyH2, fzH2);
    }

    // Store i-forces
    store_v_force_full<AVT>(fxO, fyO, fzO, fxH1, fyH1, fzH1, fxH2, fyH2, fzH2, force, ii*3, sforce, is);
  }

  mm_store_sumall(coulpot, coulpotd);
  mm_store_sumall(vdwpot, vdwpotd);

  _mm_empty();
}

//
// Calculates solvent-solute interactions
//
template<typename T, typename VT, typename AVT>
template <int coul_type, int vdw_type>
void enb_core_vec<T, VT, AVT>::enb_uv_coul_vdw(const int ni, const int *indi, const int *indj,
					       const int *startj,
					       const T *pftable, const T hinv,
					       const int *vdwtype, const T *vdwparam,
					       const xyzq_t<T> *xyzq, double *force, const int *iscoord,
					       double *sforce, const T *scoordtab,
					       double *coulpot, double *vdwpot, 
					       const T roff, const T ron) {

  if (coul_type != NONE && coul_type != LOOKUP) {
    std::cerr << "void enb_core_vec<T, VT>::enb_uv_coul_vdw Requires coul_type = LOOKUP or NONE" << std::endl;
    exit(1);
  }

  if (vdw_type != LOOKUP) {
    std::cerr << "void enb_core_vec<T, VT>::enb_uv_coul_vdw Requires vdw_type = LOOKUP" << std::endl;
    exit(1);
  }

  VT aux[8];
  setup_vdw_const<T,VT>(vdw_type, roff, ron, aux);

  VT rcsq = mm_set1<T,VT>(roff*roff);

  // Pre-load solvent charges and VdW parameters
  
  VT qO, qH;
  int ii = indi[0] - 1;
  if (coul_type != NONE) {
    qO = mm_load_one2all<T,VT>(&xyzq[ii].q);
    qH = mm_load_one2all<T,VT>(&xyzq[ii+1].q);
  }
  int iaO = vdwtype[ii];
  int iaH = vdwtype[ii+1];
#if KEY_LJPME==1
  VT c6multO = mm_load_one2all<T,VT>(&xyzq[ii].c6);
  VT c6multH = mm_load_one2all<T,VT>(&xyzq[ii+1].c6);
  int ljpme_shift = 4;
#else
  int ljpme_shift = 0;
#endif

  VT hinvd  = mm_set1<T,VT>(hinv);

  AVT coulpotd;
  if (coul_type != NONE) coulpotd = mm_load_one<double, AVT>(coulpot);
  AVT vdwpotd  = mm_load_one<double, AVT>(vdwpot);

  int i;
#pragma omp barrier
#pragma omp for private(i)
  for (i=0;i < ni;i++) {
    // Shift coordinates
    int is = iscoord[i] - 1;
    VT sx = mm_load_one2all<T,VT>(&scoordtab[is]);
    VT sy = mm_load_one2all<T,VT>(&scoordtab[is+1]);
    VT sz = mm_load_one2all<T,VT>(&scoordtab[is+2]);
    // Atom i coordinates
    int ii = indi[i] - 1;
    // Coordinates for solvent i
    VT xO = mm_add(sx,  mm_load_one2all<T,VT>(&xyzq[ii].x));
    VT yO = mm_add(sy,  mm_load_one2all<T,VT>(&xyzq[ii].y));
    VT zO = mm_add(sz,  mm_load_one2all<T,VT>(&xyzq[ii].z));
    VT xH1 = mm_add(sx, mm_load_one2all<T,VT>(&xyzq[ii+1].x));
    VT yH1 = mm_add(sy, mm_load_one2all<T,VT>(&xyzq[ii+1].y));
    VT zH1 = mm_add(sz, mm_load_one2all<T,VT>(&xyzq[ii+1].z));
    VT xH2 = mm_add(sx, mm_load_one2all<T,VT>(&xyzq[ii+2].x));
    VT yH2 = mm_add(sy, mm_load_one2all<T,VT>(&xyzq[ii+2].y));
    VT zH2 = mm_add(sz, mm_load_one2all<T,VT>(&xyzq[ii+2].z));

    // j solvent loop limits
    int j = startj[i] - 1;
    int j1 = startj[i+1] - 1;

    // Zero solvent i forces
    AVT fxO, fyO, fzO;
    AVT fxH1, fyH1, fzH1;
    AVT fxH2, fyH2, fzH2;

    mm_setzero(fxO);
    mm_setzero(fyO);
    mm_setzero(fzO);
    mm_setzero(fxH1);
    mm_setzero(fyH1);
    mm_setzero(fzH1);
    mm_setzero(fxH2);
    mm_setzero(fyH2);
    mm_setzero(fzH2);

    for (;j < j1-(veclen-1);j += veclen) {
      int jj[veclen];
#pragma unroll
      for (int t=0;t < veclen;t++) jj[t] = indj[j+t] - 1;
      
      VT xj, yj, zj;
      
      // j atom interaction with i O H1 H2
      mm_load_xyz(xyzq, jj, xj, yj, zj);
      VT dxO = mm_sub(xO, xj);
      VT dyO = mm_sub(yO, yj);
      VT dzO = mm_sub(zO, zj);
      VT dxH1 = mm_sub(xH1, xj);
      VT dyH1 = mm_sub(yH1, yj);
      VT dzH1 = mm_sub(zH1, zj);
      VT dxH2 = mm_sub(xH2, xj);
      VT dyH2 = mm_sub(yH2, yj);
      VT dzH2 = mm_sub(zH2, zj);
      const VT rsq1 = mm_add(mm_mul(dxO,dxO),mm_add(mm_mul(dyO,dyO),mm_mul(dzO,dzO)));
      const VT rsq2 = mm_add(mm_mul(dxH1,dxH1),mm_add(mm_mul(dyH1,dyH1),mm_mul(dzH1,dzH1)));
      const VT rsq3 = mm_add(mm_mul(dxH2,dxH2),mm_add(mm_mul(dyH2,dyH2),mm_mul(dzH2,dzH2)));
      VT minrsq = mm_min(mm_min(rsq1,rsq2),rsq3);
      VT cmp_mask = mm_cmplt(minrsq, rcsq);
      if (mm_movemask(cmp_mask)) {
        int ivdwO[veclen];
        int ivdwH[veclen];
        for (int i=0;i < veclen;i++) {
          int ja = vdwtype[jj[i]];
          int aa = (ja > iaO) ? ja : iaO;
          ivdwO[i] = aa*(aa-3) + 2*(ja + iaO) - 2;
          aa = (ja > iaH) ? ja : iaH;
          ivdwH[i] = aa*(aa-3) + 2*(ja + iaH) - 2;
        }
        VT c6jO, c12jO;
        load_vdwparam(vdwparam, ivdwO, c6jO, c12jO);
        VT c6jH, c12jH;
        load_vdwparam(vdwparam, ivdwH, c6jH, c12jH);
        c6jO = mm_and(c6jO, cmp_mask);
        c6jH = mm_and(c6jH, cmp_mask);
        c12jO = mm_and(c12jO, cmp_mask);
        c12jH = mm_and(c12jH, cmp_mask);
        VT qjO, qjH;
        if (coul_type != NONE) {
          VT qj;
          load_charge(xyzq, jj, qj);
          qjO = mm_mul(qj, qO);
          qjH = mm_mul(qj, qH);
          qjO = mm_and(qjO, cmp_mask);
          qjH = mm_and(qjH, cmp_mask);
        }
        VT c6multjO, c6multjH; // Undefined, and unused, if no LJPME
#if KEY_LJPME==1
        VT c6multj;
        load_c6(xyzq, jj, c6multj);
        c6multjO = mm_mul(c6multj, c6multO);
        c6multjH = mm_mul(c6multj, c6multH);
        c6multjO = mm_and(c6multjO, cmp_mask);
        c6multjH = mm_and(c6multjH, cmp_mask);
#endif
        AVT coul, vdw;
        VT fij1, fij2, fij3;
        CALC_V_FULL_KERNEL(T, VT, coul_type, vdw_type,
                           rsq1, rsq2, rsq3,
                           hinvd, pftable, aux,
                           qjO, qjH,
                           c6jO, c6jH,
                           c12jO, c12jH,
                           c6multjO, c6multjH, ljpme_shift,
                           coul, vdw,
                           fij1, fij2, fij3);
        if (coul_type != NONE) coulpotd = mm_add(coulpotd, coul);
        vdwpotd  = mm_add(vdwpotd,  vdw);
        ADD_V_FORCE_FULL(T, VT, AVT, dxO, dxH1, dxH2, jj, 0, fij1, fij2, fij3, force, fxO, fxH1, fxH2);
        ADD_V_FORCE_FULL(T, VT, AVT, dyO, dyH1, dyH2, jj, 1, fij1, fij2, fij3, force, fyO, fyH1, fyH2);
        ADD_V_FORCE_FULL(T, VT, AVT, dzO, dzH1, dzH2, jj, 2, fij1, fij2, fij3, force, fzO, fzH1, fzH2);
      }
    }

    if (j != j1) {
      if (j > j1) j -= veclen;    // In case we stepped over limit, step back by one
      int jleft = j1 - j;

      int jj[veclen];
      int t;
      for (t=0;t < jleft;t++) jj[t] = indj[j+t] - 1;
      for (;t < veclen;t++)   jj[t] = indj[j] - 1;    // Set the rest of jj to point to the first one

      VT mask;
      get_mask(mask, jleft);

      VT xj, yj, zj;
      // j atom interaction with i O H1 H2
      mm_load_xyz(xyzq, jj, xj, yj, zj);
      VT dxO = mm_sub(xO, xj);
      VT dyO = mm_sub(yO, yj);
      VT dzO = mm_sub(zO, zj);
      VT dxH1 = mm_sub(xH1, xj);
      VT dyH1 = mm_sub(yH1, yj);
      VT dzH1 = mm_sub(zH1, zj);
      VT dxH2 = mm_sub(xH2, xj);
      VT dyH2 = mm_sub(yH2, yj);
      VT dzH2 = mm_sub(zH2, zj);
      const VT rsq1 = mm_add(mm_mul(dxO,dxO),mm_add(mm_mul(dyO,dyO),mm_mul(dzO,dzO)));
      const VT rsq2 = mm_add(mm_mul(dxH1,dxH1),mm_add(mm_mul(dyH1,dyH1),mm_mul(dzH1,dzH1)));
      const VT rsq3 = mm_add(mm_mul(dxH2,dxH2),mm_add(mm_mul(dyH2,dyH2),mm_mul(dzH2,dzH2)));
      VT minrsq = mm_min(mm_min(rsq1,rsq2),rsq3);
      mask = mm_and(mask, mm_cmplt(minrsq, rcsq));
      if (mm_movemask(mask)) {
        int ivdwO[veclen];
        int ivdwH[veclen];
        for (int i=0;i < veclen;i++) {
          int ja = vdwtype[jj[i]];
          int aa = (ja > iaO) ? ja : iaO;
          ivdwO[i] = aa*(aa-3) + 2*(ja + iaO) - 2;
          aa = (ja > iaH) ? ja : iaH;
          ivdwH[i] = aa*(aa-3) + 2*(ja + iaH) - 2;
        }
        VT c6jO, c12jO;
        load_vdwparam(vdwparam, ivdwO, c6jO, c12jO);
        VT c6jH, c12jH;
        load_vdwparam(vdwparam, ivdwH, c6jH, c12jH);
        c6jO  = mm_and(c6jO, mask);
        c12jO = mm_and(c12jO, mask);
        c6jH  = mm_and(c6jH, mask);
        c12jH = mm_and(c12jH, mask);
        VT qjO, qjH;
        if (coul_type != NONE) {
          VT qj;
          load_charge(xyzq, jj, qj);
          qjO = mm_mul(qj, qO);
          qjH = mm_mul(qj, qH);
          qjO = mm_and(qjO, mask);
          qjH = mm_and(qjH, mask);
        }
        VT c6multjO, c6multjH;
#if KEY_LJPME==1
        VT c6multj;
        load_c6(xyzq, jj, c6multj);
        c6multjO = mm_mul(c6multj, c6multO);
        c6multjH = mm_mul(c6multj, c6multH);
        c6multjO = mm_and(c6multjO, mask);
        c6multjH = mm_and(c6multjH, mask);
#endif
        AVT coul, vdw;
        VT fij1, fij2, fij3;
        CALC_V_FULL_KERNEL(T, VT, coul_type, vdw_type,
                           rsq1, rsq2, rsq3,
                           hinvd, pftable, aux,
                           qjO, qjH,
                           c6jO, c6jH,
                           c12jO, c12jH,
                           c6multjO, c6multjH, ljpme_shift,
                           coul, vdw,
                           fij1, fij2, fij3);
        if (coul_type != NONE) coulpotd = mm_add(coulpotd, coul);
        vdwpotd  = mm_add(vdwpotd,  vdw);
        ADD_V_FORCE_FULL(T, VT, AVT, dxO, dxH1, dxH2, jj, 0, fij1, fij2, fij3, force, fxO, fxH1, fxH2);
        ADD_V_FORCE_FULL(T, VT, AVT, dyO, dyH1, dyH2, jj, 1, fij1, fij2, fij3, force, fyO, fyH1, fyH2);
        ADD_V_FORCE_FULL(T, VT, AVT, dzO, dzH1, dzH2, jj, 2, fij1, fij2, fij3, force, fzO, fzH1, fzH2);
      }

    }

    // Store i-forces
    store_v_force_full<AVT>(fxO, fyO, fzO, fxH1, fyH1, fzH1, fxH2, fyH2, fzH2, force, ii*3, sforce, is);
  }

  if (coul_type != NONE) mm_store_sumall(coulpot, coulpotd);
  mm_store_sumall(vdwpot, vdwpotd);

  _mm_empty();

}

//
// Calculates solute-solute interactions
//
template<typename T, typename VT, typename AVT>
template <int coul_type, int vdw_type>
void enb_core_vec<T, VT, AVT>::enb_uu_coul_vdw(const int ni, const int *indi, const int *indj,
					       const int *startj,
					       const T *pftable, const T hinv,
					       const int *vdwtype, const T *vdwparam,
					       const xyzq_t<T> *xyzq, double *force, const int *iscoord,
					       double *sforce, const T *scoordtab,
					       double *coulpot, double *vdwpot, 
					       const T roff, const T ron) {

  if (coul_type != NONE && coul_type != LOOKUP) {
    std::cerr << "void enb_core_vec<T, VT>::enb_uu_coul_vdw Requires coul_type = LOOKUP or NONE" << std::endl;
    exit(1);
  }

  if (vdw_type != LOOKUP) {
    std::cerr << "void enb_core_vec<T, VT>::enb_uu_coul_vdw Requires vdw_type = LOOKUP" << std::endl;
    exit(1);
  }

  VT aux[8];
  setup_vdw_const<T,VT>(vdw_type, roff, ron, aux);

  VT rcsq = mm_set1<T,VT>(roff*roff);

  VT hinvd  = mm_set1<T,VT>(hinv);

  AVT coulpotd;
  if (coul_type != NONE) coulpotd = mm_load_one<double, AVT>(coulpot);
  AVT vdwpotd  = mm_load_one<double, AVT>(vdwpot);
  //if (coul_type != NONE) mm_setzero(coulpotd);
  //AVT vdwpotd;
  //mm_setzero(vdwpotd);

  int i;
#pragma omp barrier
#pragma omp for private(i)
  for (i=0;i < ni;i++) {
    // Shift coordinates
    int is = iscoord[i] - 1;
    VT sx = mm_load_one2all<T,VT>(&scoordtab[is]);
    VT sy = mm_load_one2all<T,VT>(&scoordtab[is+1]);
    VT sz = mm_load_one2all<T,VT>(&scoordtab[is+2]);
    // Atom i coordinates
    int ii = indi[i] - 1;
    // Coordinates for solute i
    VT xi = mm_add(sx, mm_load_one2all<T,VT>(&xyzq[ii].x));
    VT yi = mm_add(sy, mm_load_one2all<T,VT>(&xyzq[ii].y));
    VT zi = mm_add(sz, mm_load_one2all<T,VT>(&xyzq[ii].z));
    VT qi = mm_load_one2all<T,VT>(&xyzq[ii].q);
#if KEY_LJPME==1
    VT c6multi = mm_load_one2all<T,VT>(&xyzq[ii].c6);
#endif
    int ia = vdwtype[ii];

    // j solvent loop limits
    int j = startj[i] - 1;
    int j1 = startj[i+1] - 1;

    // Zero solute i forces
    AVT fxi, fyi, fzi;

    mm_setzero(fxi);
    mm_setzero(fyi);
    mm_setzero(fzi);

    for (;j < j1-(veclen-1);j += veclen) {
      int jj[veclen];
#pragma unroll
      for (int t=0;t < veclen;t++) jj[t] = indj[j+t] - 1;
      
      VT xj, yj, zj;
      
      // j atom interaction with i atoms
      mm_load_xyz(xyzq, jj, xj, yj, zj);
      VT dx = mm_sub(xi, xj);
      VT dy = mm_sub(yi, yj);
      VT dz = mm_sub(zi, zj);
      const VT rsq = mm_add(mm_mul(dx,dx),mm_add(mm_mul(dy,dy),mm_mul(dz,dz)));
      VT cmp_mask = mm_cmplt(rsq, rcsq);
      if (mm_movemask(cmp_mask)) {
        int ivdw[veclen];
        for (int i=0;i < veclen;i++) {
          int ja = vdwtype[jj[i]];
          int aa = (ja > ia) ? ja : ia;
          ivdw[i] = aa*(aa-3) + 2*(ja + ia) - 2;
        }
        VT c6, c12;
        load_vdwparam(vdwparam, ivdw, c6, c12);
        c6 = mm_and(c6, cmp_mask);
        c12 = mm_and(c12, cmp_mask);
        VT qij;
        if (coul_type != NONE) {
          VT qj;
          load_charge(xyzq, jj, qj);
          qij = mm_mul(qj, qi);
          qij = mm_and(qij, cmp_mask);
        }
        AVT coul, vdw;
        VT fij;
#if KEY_LJPME==1
        VT c6multj, c6multij;
        load_c6(xyzq, jj, c6multj);
        c6multij = mm_mul(c6multj, c6multi);
        c6multij = mm_and(c6multij, cmp_mask);
        int ljpme_shift = 4;
#else
        VT c6multij; // no need to define this; it'll be unused and optimized away.
        int ljpme_shift = 0;
#endif
        CALC_U_FULL_KERNEL(T, VT, coul_type, vdw_type,
                      	   rsq, hinvd, pftable, aux,
                      	   qij, c6, c12, c6multij, coul, vdw, fij, ljpme_shift);
        if (coul_type != NONE) coulpotd = mm_add(coulpotd, coul);
        vdwpotd  = mm_add(vdwpotd,  vdw);
        ADD_U_FORCE_FULL(T, VT, AVT, dx, jj, 0, fij, force, fxi);
        ADD_U_FORCE_FULL(T, VT, AVT, dy, jj, 1, fij, force, fyi);
        ADD_U_FORCE_FULL(T, VT, AVT, dz, jj, 2, fij, force, fzi);
      }
    }

    if (j != j1) {
      if (j > j1) j -= veclen;    // In case we stepped over limit, step back by one
      int jleft = j1 - j;

      int jj[veclen];
      int t;
      for (t=0;t < jleft;t++) jj[t] = indj[j+t] - 1;
      for (;t < veclen;t++)   jj[t] = indj[j] - 1;    // Set the rest of jj to point to the first one

      VT mask;
      get_mask(mask, jleft);

      VT xj, yj, zj;
      // j atom interaction with i O H1 H2
      mm_load_xyz(xyzq, jj, xj, yj, zj);
      VT dx = mm_sub(xi, xj);
      VT dy = mm_sub(yi, yj);
      VT dz = mm_sub(zi, zj);
      const VT rsq = mm_add(mm_mul(dx,dx),mm_add(mm_mul(dy,dy),mm_mul(dz,dz)));
      mask = mm_and(mask, mm_cmplt(rsq, rcsq));
      if (mm_movemask(mask)) {
        int ivdw[veclen];
        for (int i=0;i < veclen;i++) {
          int ja = vdwtype[jj[i]];
          int aa = (ja > ia) ? ja : ia;
          ivdw[i] = aa*(aa-3) + 2*(ja + ia) - 2;
        }
        VT c6, c12;
        load_vdwparam(vdwparam, ivdw, c6, c12);
        c6  = mm_and(c6, mask);
        c12 = mm_and(c12, mask);
        VT qij;
        if (coul_type != NONE) {
          VT qj;
          load_charge(xyzq, jj, qj);
          qij = mm_mul(qj, qi);
          qij = mm_and(qij, mask);
        }
        AVT coul, vdw;
        VT fij;
#if KEY_LJPME==1
        VT c6multj, c6multij;
        load_c6(xyzq, jj, c6multj);
        c6multij = mm_mul(c6multj, c6multi);
        c6multij = mm_and(c6multij, mask);
        CALC_U_FULL_KERNEL(T, VT, coul_type, vdw_type,
                      	   rsq, hinvd, pftable, aux,
                      	   qij, c6, c12, c6multij, coul, vdw, fij, 4);
#else
        VT c6multij; // No need to define this; it'll be unused and optimized away.
        CALC_U_FULL_KERNEL(T, VT, coul_type, vdw_type,
                      	   rsq, hinvd, pftable, aux,
                      	   qij, c6, c12, c6multij, coul, vdw, fij, 0);
#endif
        if (coul_type != NONE) coulpotd = mm_add(coulpotd, coul);
        vdwpotd  = mm_add(vdwpotd,  vdw);
        ADD_U_FORCE_FULL(T, VT, AVT, dx, jj, 0, fij, force, fxi);
        ADD_U_FORCE_FULL(T, VT, AVT, dy, jj, 1, fij, force, fyi);
        ADD_U_FORCE_FULL(T, VT, AVT, dz, jj, 2, fij, force, fzi);
      }
    }

    // Store i-forces
    store_u_force_full<AVT>(fxi, fyi, fzi, force, ii*3, sforce, is);
  }

  if (coul_type != NONE) mm_store_sumall(coulpot, coulpotd);
  mm_store_sumall(vdwpot, vdwpotd);

  _mm_empty();

}

//
// Calculates solvent-solvent interactions
//
template<typename T, typename VT, typename AVT>
void enb_core_vec<T, VT, AVT>::calc_vv(const int ni, const int *indi, const int *indj,
				       const int *startj,
				       const T *pftable, const T hinv,
				       const xyzq_t<T> *xyzq, double *force,
				       const int *iscoord,
				       double *sforce, const T *scoordtab,
				       double *coulpot, double *vdwpot, 
				       const T c6OO, const T c12OO, const T c6OH, const T c12OH, const T c6HH, const T c12HH,
#if KEY_LJPME==1
                       const T c6multOO, const T c6multOH, const T c6multHH,
#endif
                       const T qOO, const T qOH, const T qHH,
				       const T roff, const T ron, const int vdw_type) {

  switch (vdw_type) {
  case 1:
    enb_vv<LOOKUP,LOOKUP>(ni, indi, indj, startj, pftable, hinv, xyzq, force,
			  iscoord, sforce, scoordtab, coulpot, vdwpot, c6OO, c12OO, c6OH, c12OH, c6HH, c12HH,
#if KEY_LJPME==1
              c6multOO, c6multOH, c6multHH,
#endif
              qOO, qOH, qHH, roff, ron);
    break;

  case 2:
    enb_vv<LOOKUP,VSH>(ni, indi, indj, startj, pftable, hinv, xyzq, force,
		       iscoord, sforce, scoordtab, coulpot, vdwpot, c6OO, c12OO, c6OH, c12OH, c6HH, c12HH,
#if KEY_LJPME==1
               c6multOO, c6multOH, c6multHH,
#endif
               qOO, qOH, qHH, roff, ron);
    break;

  case 3:
    enb_vv<LOOKUP,VSW>(ni, indi, indj, startj, pftable, hinv, xyzq, force,
		       iscoord, sforce, scoordtab, coulpot, vdwpot, c6OO, c12OO, c6OH, c12OH, c6HH, c12HH,
#if KEY_LJPME==1
               c6multOO, c6multOH, c6multHH,
#endif
               qOO, qOH, qHH, roff, ron);
    break;

  case 4:
    enb_vv<LOOKUP,VFSW>(ni, indi, indj, startj, pftable, hinv, xyzq, force,
			iscoord, sforce, scoordtab, coulpot, vdwpot, c6OO, c12OO, c6OH, c12OH, c6HH, c12HH,
#if KEY_LJPME==1
            c6multOO, c6multOH, c6multHH,
#endif
            qOO, qOH, qHH, roff, ron);
    break;

  default:
    std::cerr << "enb_core_vec.cpp: Invalid vdw_type" << std::endl;
    exit(1);
  }

}

//
// Calculates solvent-solute interactions with Coulomb and VdW
//
template<typename T, typename VT, typename AVT>
void enb_core_vec<T, VT, AVT>::calc_uv_coul_vdw(const int ni, const int *indi, const int *indj,
						const int *startj,
						const T *pftable, const T hinv,
						const int *vdwtype, const T *vdwparam, 
						const xyzq_t<T> *xyzq, double *force,
						const int *iscoord,
						double *sforce, const T *scoordtab,
						double *coulpot, double *vdwpot, 
						const T roff, const T ron, const int vdw_type) {

  switch (vdw_type) {
  case 1:
    enb_uv_coul_vdw<LOOKUP,LOOKUP>(ni, indi, indj, startj, pftable, hinv,
				   vdwtype, vdwparam, xyzq, force,
				   iscoord, sforce, scoordtab, coulpot, vdwpot, roff, ron);
    break;

  case 2:
    enb_uv_coul_vdw<LOOKUP,VSH>(ni, indi, indj, startj, pftable, hinv,
				vdwtype, vdwparam, xyzq, force,
				iscoord, sforce, scoordtab, coulpot, vdwpot, roff, ron);
    break;

  case 3:
    enb_uv_coul_vdw<LOOKUP,VSW>(ni, indi, indj, startj, pftable, hinv,
				vdwtype, vdwparam, xyzq, force,
				iscoord, sforce, scoordtab, coulpot, vdwpot, roff, ron);
    break;

  case 4:
    enb_uv_coul_vdw<LOOKUP,VFSW>(ni, indi, indj, startj, pftable, hinv,
				 vdwtype, vdwparam, xyzq, force,
				 iscoord, sforce, scoordtab, coulpot, vdwpot, roff, ron);
    break;

  default:
    std::cerr << "enb_core_vec.cpp: Invalid vdw_type" << std::endl;
    exit(1);
  }

}

//
// Calculates solvent-solute interactions with VdW
//
template<typename T, typename VT, typename AVT>
void enb_core_vec<T, VT, AVT>::calc_uv_vdw(const int ni, const int *indi, const int *indj,
					   const int *startj,
					   const T *pftable, const T hinv,
					   const int *vdwtype, const T *vdwparam, 
					   const xyzq_t<T> *xyzq, double *force,
					   const int *iscoord,
					   double *sforce, const T *scoordtab,
					   double *vdwpot, 
					   const T roff, const T ron, const int vdw_type) {
  double coulpot;

  switch (vdw_type) {
  case 1:
    enb_uv_coul_vdw<NONE,LOOKUP>(ni, indi, indj, startj, pftable, hinv,
				 vdwtype, vdwparam, xyzq, force,
				 iscoord, sforce, scoordtab, &coulpot, vdwpot, roff, ron);
    break;

  case 2:
    enb_uv_coul_vdw<NONE,VSH>(ni, indi, indj, startj, pftable, hinv,
			      vdwtype, vdwparam, xyzq, force,
			      iscoord, sforce, scoordtab, &coulpot, vdwpot, roff, ron);
    break;

  case 3:
    enb_uv_coul_vdw<NONE,VSW>(ni, indi, indj, startj, pftable, hinv,
			      vdwtype, vdwparam, xyzq, force,
			      iscoord, sforce, scoordtab, &coulpot, vdwpot, roff, ron);
    break;

  case 4:
    enb_uv_coul_vdw<NONE,VFSW>(ni, indi, indj, startj, pftable, hinv,
			       vdwtype, vdwparam, xyzq, force,
			       iscoord, sforce, scoordtab, &coulpot, vdwpot, roff, ron);
    break;

  default:
    std::cerr << "enb_core_vec.cpp: Invalid vdw_type" << std::endl;
    exit(1);
  }

}

//
// Calculates solute-solute interactions with Coulomb and VdW
//
template<typename T, typename VT, typename AVT>
void enb_core_vec<T, VT, AVT>::calc_uu_coul_vdw(const int ni, const int *indi, const int *indj,
						const int *startj,
						const T *pftable, const T hinv,
						const int *vdwtype, const T *vdwparam, 
						const xyzq_t<T> *xyzq, double *force,
						const int *iscoord,
						double *sforce, const T *scoordtab,
						double *coulpot, double *vdwpot, 
						const T roff, const T ron, const int vdw_type) {

  switch (vdw_type) {
  case 1:
    enb_uu_coul_vdw<LOOKUP,LOOKUP>(ni, indi, indj, startj, pftable, hinv,
				   vdwtype, vdwparam, xyzq, force,
				   iscoord, sforce, scoordtab, coulpot, vdwpot, roff, ron);
    break;

  case 2:
    enb_uu_coul_vdw<LOOKUP,VSH>(ni, indi, indj, startj, pftable, hinv,
				vdwtype, vdwparam, xyzq, force,
				iscoord, sforce, scoordtab, coulpot, vdwpot, roff, ron);
    break;

  case 3:
    enb_uu_coul_vdw<LOOKUP,VSW>(ni, indi, indj, startj, pftable, hinv,
				vdwtype, vdwparam, xyzq, force,
				iscoord, sforce, scoordtab, coulpot, vdwpot, roff, ron);
    break;

  case 4:
    enb_uu_coul_vdw<LOOKUP,VFSW>(ni, indi, indj, startj, pftable, hinv,
				 vdwtype, vdwparam, xyzq, force,
				 iscoord, sforce, scoordtab, coulpot, vdwpot, roff, ron);
    break;

  default:
    std::cerr << "enb_core_vec.cpp: Invalid vdw_type" << std::endl;
    exit(1);
  }

}

//
// Calculates solute-solute interactions with VdW
//
template<typename T, typename VT, typename AVT>
void enb_core_vec<T, VT, AVT>::calc_uu_vdw(const int ni, const int *indi, const int *indj,
					   const int *startj,
					   const T *pftable, const T hinv,
					   const int *vdwtype, const T *vdwparam, 
					   const xyzq_t<T> *xyzq, double *force,
					   const int *iscoord,
					   double *sforce, const T *scoordtab,
					   double *vdwpot, 
					   const T roff, const T ron, const int vdw_type) {

  double coulpot;

  switch (vdw_type) {
  case 1:
    enb_uu_coul_vdw<NONE,LOOKUP>(ni, indi, indj, startj, pftable, hinv,
				 vdwtype, vdwparam, xyzq, force,
				 iscoord, sforce, scoordtab, &coulpot, vdwpot, roff, ron);
    break;

  case 2:
    enb_uu_coul_vdw<NONE,VSH>(ni, indi, indj, startj, pftable, hinv,
			      vdwtype, vdwparam, xyzq, force,
			      iscoord, sforce, scoordtab, &coulpot, vdwpot, roff, ron);
    break;

  case 3:
    enb_uu_coul_vdw<NONE,VSW>(ni, indi, indj, startj, pftable, hinv,
			      vdwtype, vdwparam, xyzq, force,
			      iscoord, sforce, scoordtab, &coulpot, vdwpot, roff, ron);
    break;

  case 4:
    enb_uu_coul_vdw<NONE,VFSW>(ni, indi, indj, startj, pftable, hinv,
			       vdwtype, vdwparam, xyzq, force,
			       iscoord, sforce, scoordtab, &coulpot, vdwpot, roff, ron);
    break;

  default:
    std::cerr << "enb_core_vec.cpp: Invalid vdw_type" << std::endl;
    exit(1);
  }

}

//
// Explicit instances of enb_core_vec
//
template class enb_core_vec<float,  __m128,  __m128d>;
template class enb_core_vec<double, __m128d, __m128d>;
#ifdef __AVX__
template class enb_core_vec<float, __m256, __m256d>;
template class enb_core_vec<double, __m256d, __m256d>;
#endif

#endif // __SSE2__
