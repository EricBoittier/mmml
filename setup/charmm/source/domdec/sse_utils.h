//
// NOTE: "round" Rounds a positive number to the nearest integer value
//
static inline __attribute__((always_inline)) __m128 floor_m128(const __m128 a) {
#ifdef __SSE4_1__
  return _mm_floor_ps(a);
#else
  return _mm_cvtepi32_ps(_mm_cvttps_epi32(a));
#endif
}

static inline __attribute__((always_inline)) __m128 round_m128(const __m128 a) {
  const __m128 half  = _mm_set1_ps(0.5f);
  return floor_m128(_mm_add_ps(a,half));
}

static inline __attribute__((always_inline)) __m128d floor_m128d(const __m128d a) {
#ifdef __SSE4_1__
  return _mm_floor_pd(a);
#else
  return _mm_cvtepi32_pd(_mm_cvttpd_epi32(a));
#endif
}

static inline __attribute__((always_inline)) __m128d round_m128d(const __m128d a) {
  const __m128d half  = _mm_set1_pd(0.5);
  return floor_m128d(_mm_add_pd(a,half));
}

//
// shift __m128 to the right by n bytes
//
//static inline __attribute__((always_inline)) __m128 sr_m128(const __m128 a, const int n) {
//  return _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a),n));
//}

//
// shift __m128 to the right by 4 bytes
//
static inline __attribute__((always_inline)) __m128 sr_m128_4(const __m128 a) {
  return _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a),4));
}

//
// shift __m128 to the left by n bytes
//
//static inline __attribute__((always_inline)) __m128 sl_m128(const __m128 a, const int n) {
//  return _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(a),n));
//}

//
// Shifts __m128 to the right by 4 bytes, and cycles the lowest 4 bytes to the top
// Returns: [a1][a2][a3][a0]
// 
static inline __attribute__((always_inline)) __m128 src_m128(const __m128 a) {
  return _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(a), (int)(1+(2<<2)+(3<<4))));
}

// Takes [a0][a1][a2][a3] and returns [a0][a0][a0][a0]
static inline __attribute__((always_inline)) __m128 set1_m128(const __m128 a) {
  __m128 b = _mm_unpacklo_ps(a,a);  // b = [a0][a0][a1][a1]
  return _mm_unpacklo_ps(b,b);      // [a0][a0][a0][a0]
}

#ifdef __SSE3__
// Returns [a0 + a1 + a2 + a3][a0 + a1 + a2 + a3][a0 + a1 + a2 + a3][a0 + a1 + a2 + a3]
static inline __attribute__((always_inline)) __m128 sum_m128(const __m128 a) {
  __m128 b = _mm_hadd_ps(a,a); // [a0 + a1][a2 + a3][a0 + a1][a2 + a3]
  return _mm_hadd_ps(b,b); //[a0 + a1 + a2 + a3][a0 + a1 + a2 + a3][a0 + a1 + a2 + a3][a0 + a1 + a2 + a3]
}
#else
// Returns [a0 + a1 + a2 + a3][a0 + a1 + a2 + a3][a0 + a1 + a2 + a3][a0 + a1 + a2 + a3]
static inline __attribute__((always_inline)) __m128 sum_m128(const __m128 a) {
  __m128 b = a;
  // [a0 + a1][a1 + a2][a2 + a3][a3 + a0]
  b = _mm_add_ps(b, src_m128(b));
  // [a0 + a1 + a2 + a3][a1 + a2 + a3 + a0][a2 + a3 + a0 + a1][a3 + a0 + a1 + a2]
  b = _mm_add_ps(b, src_m128(src_m128(b)));
  return b;
}
#endif

//
// Loads 4 single precision numbers into 4 single precision slots:
// [sp1] [sp2] [sp3] [sp4]
//
static inline __attribute__((always_inline)) __m128 load_4sp(const float *sp1, const float *sp2,
							     const float *sp3, const float *sp4) {

  __m128 tmp1 = _mm_unpacklo_ps(_mm_load_ss(sp1),_mm_load_ss(sp2)); //[sp1][sp2][0][0]
  __m128 tmp2 = _mm_unpacklo_ps(_mm_load_ss(sp3),_mm_load_ss(sp4)); //[sp3][sp4][0][0]

  return _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1,0,1,0));
}

//
// Loads 4 double precision numbers into 4 single precision slots:
// [(float)*dp1] [(float)*dp2] [(float)*dp3] [(float)*dp4]
//
static inline __attribute__((always_inline)) __m128 load_4dp_4sp(const double *dp1, const double *dp2,
								 const double *dp3, const double *dp4) {
  return _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadh_pd(_mm_load_sd(dp1), dp2)),
		       _mm_cvtpd_ps(_mm_loadh_pd(_mm_load_sd(dp3), dp4)));
}

//
// Loads 3 double precision numbers into 4 single precision slots, repeating the
// last number:
// [(float)*dp1] [(float)*dp2] [(float)*dp3] [(float)*dp3]
//
static inline __attribute__((always_inline)) __m128 load_3dp_4sp(const double *dp1, const double *dp2,
								 const double *dp3) {
  return _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadh_pd(_mm_load_sd(dp1), dp2)),
		       _mm_cvtpd_ps(_mm_load1_pd(dp3)));
}

//
// Loads 1 double precision number into all 4 single precision slots
//
static inline __attribute__((always_inline)) __m128 load_dp_4sp(const double *dp) {
  __m128 a  = _mm_cvtpd_ps(_mm_load1_pd(dp));  // [(float)*dp][(float)*dp][0][0]
  return _mm_movelh_ps(a,a);                   // [(float)*dp][(float)*dp][(float)*dp][(float)*dp]
}

//
// Loads 1 double precision number into 1 single precision slot (upper three are zero)
//
static inline __attribute__((always_inline)) __m128 load_dp_sp(const double *dp) {
  return _mm_cvtpd_ps(_mm_load_sd(dp));  // [(float)*dp][0][0][0]
}

//
// Loads 2 double precision numbers
//
static inline __attribute__((always_inline)) __m128d load_2dp(const double *dp1, const double *dp2) {
  return _mm_loadh_pd(_mm_load_sd(dp1), dp2);
}

//
// Stores 2 double precision numbers
//
static inline __attribute__((always_inline)) void store_2dp(const __m128d a, double *dp1, double *dp2) {
  _mm_storel_pd(dp1, a);
  _mm_storeh_pd(dp2, a);
}

//
// Adds 4 single precision numbers and stores the result as double precision
//
static inline __attribute__((always_inline)) void add_4sp_store_dp(const __m128 a, double *dp) {
  __m128d b = _mm_cvtps_pd(sum_m128(a));
  _mm_storeu_pd(dp, b);
}

//
// Converts single precision [a0][a1][a2][a3] to double precision and stores it in dp[0:3]
//
static inline __attribute__((always_inline)) void store_4sp_4dp(const __m128 a, double *dp) {
  __m128d a01 = _mm_cvtps_pd(a);                  // [a0][a1]
  __m128d a23 = _mm_cvtps_pd(_mm_movehl_ps(a,a)); // [a2][a3]
  _mm_storeu_pd(&dp[0], a01);
  _mm_storeu_pd(&dp[2], a23);
}

//
// Adds: [dp0][dp1] = [dp0 + sp0 + sp2][dp1 + sp1 + sp3]
//
static inline __attribute__((always_inline)) __m128d add_sp2dp(__m128d dp, __m128 sp) {
  dp = _mm_add_pd(dp, _mm_cvtps_pd(sp));
  return _mm_add_pd(dp, _mm_cvtps_pd(_mm_movehl_ps(sp,sp)));
}

//
// Adds: [dp0][dp1] = [dp0 + sp0][dp1]
//
static inline __attribute__((always_inline)) __m128d add_ss2dp(__m128d dp, __m128 sp) {
  //_mm_cvtss_sd(dp, sp) = [sp0][dp1]
  return _mm_add_sd(dp,_mm_cvtss_sd(dp, sp));  // [dp0 + sp0][dp1]
}

//
// Packs 3 single precision numbers into one: [a0] [b0] [c0] [c0]
//
static inline __attribute__((always_inline)) __m128 pack_3sp_sp(__m128 a, __m128 b, __m128 c) {
  return _mm_move_ss(_mm_shuffle_ps(b,c,_MM_SHUFFLE(1,1,1,1)), a);
}

//
// Calculates rsq = x*x + y*y + z*z
//
static inline __attribute__((always_inline)) __m128 square_ps(__m128 x, __m128 y, __m128 z) {
  __m128 rsq = _mm_mul_ps(x,x);
  rsq = _mm_add_ps(rsq, _mm_mul_ps(y,y));
  rsq = _mm_add_ps(rsq, _mm_mul_ps(z,z));
  return rsq;
}

//
// Calculates rsq = x*x + y*y + z*z
//
static inline __attribute__((always_inline)) __m128 square_ss(__m128 x, __m128 y, __m128 z) {
  __m128 rsq = _mm_mul_ss(x,x);
  rsq = _mm_add_ss(rsq, _mm_mul_ss(y,y));
  rsq = _mm_add_ss(rsq, _mm_mul_ss(z,z));
  return rsq;
}

//
// Calculates 1/sqrt(x)
//
static inline __attribute__((always_inline)) __m128 invsqrt_ps(__m128 x) {
  const __m128 half  = _mm_set1_ps(0.5f);
  const __m128 three = _mm_set1_ps(3.0f);
  __m128 lu = _mm_rsqrt_ps(x);
  lu = _mm_mul_ps(half,_mm_mul_ps(_mm_sub_ps(three,_mm_mul_ps(_mm_mul_ps(lu,lu),x)),lu));
  return _mm_mul_ps(half,_mm_mul_ps(_mm_sub_ps(three,_mm_mul_ps(_mm_mul_ps(lu,lu),x)),lu));
}


//
// Faster ERFC approximation courtesy of Norbert Juffa. NVIDIA Corporation
//
static inline __attribute__((always_inline)) __m128 fasterfc(__m128 a) 
{
  // approximate log(erfc(a)) with rel. error < 7e-9
  const __m128 const1 = _mm_set1_ps((float)-1.6488499458192755E-006);
  const __m128 const2 = _mm_set1_ps((float)2.9524665006554534E-005);
  const __m128 const3 = _mm_set1_ps((float)-2.3341951153749626E-004);
  const __m128 const4 = _mm_set1_ps((float)1.0424943374047289E-003);
  const __m128 const5 = _mm_set1_ps((float)-2.5501426008983853E-003);
  const __m128 const6 = _mm_set1_ps((float)3.1979939710877236E-004);
  const __m128 const7 = _mm_set1_ps((float)2.7605379075746249E-002);
  const __m128 const8 = _mm_set1_ps((float)-1.4827402067461906E-001);
  const __m128 const9 = _mm_set1_ps((float)-9.1844764013203406E-001);
  const __m128 const10 = _mm_set1_ps((float)-1.6279070384382459E+000);

  __m128 x = a;
  __m128 t = const1;
  t = _mm_add_ps(_mm_mul_ps(t, x), const2);
  t = _mm_add_ps(_mm_mul_ps(t, x), const3);
  t = _mm_add_ps(_mm_mul_ps(t, x), const4);
  t = _mm_add_ps(_mm_mul_ps(t, x), const5);
  t = _mm_add_ps(_mm_mul_ps(t, x), const6);
  t = _mm_add_ps(_mm_mul_ps(t, x), const7);
  t = _mm_add_ps(_mm_mul_ps(t, x), const8);
  t = _mm_add_ps(_mm_mul_ps(t, x), const9);
  t = _mm_add_ps(_mm_mul_ps(t, x), const10);
  t = _mm_mul_ps(t, x);

  float t_float[4], exp2f_t_float[4];
  _mm_storeu_ps(t_float, t);
  int k;
#pragma simd assert
  for (k=0;k < 4;k++) exp2f_t_float[k] = exp2f(t_float[k]);

  return _mm_loadu_ps(exp2f_t_float);
}

//
// Prints 2 double precision numbers
//
static void print_dp(__m128d a) {
  double ad[2];

  _mm_storeu_pd(ad, a);

  fprintf(stderr,"%lf %lf\n",ad[0],ad[1]);

  return;
}

//
// Prints 4 single precision numbers
//
static void print_sp(__m128 a) {
  float af[4];

  _mm_storeu_ps(af, a);

  fprintf(stderr,"%f %f %f %f\n",af[0],af[1],af[2],af[3]);

  return;
}

//
// Prints 4 integers
//
static void print_int(__m128i a) {
  int ai[4];

  _mm_storeu_si128((__m128i *)ai, a);

  fprintf(stderr,"%d %d %d %d\n",ai[0],ai[1],ai[2],ai[3]);

  return;
}

#define transpose_4x4_m128i(in0, in1, in2, in3)				\
  {									\
    __m128i tmp0 = _mm_unpacklo_epi32(in0, in1);			\
    __m128i tmp1 = _mm_unpacklo_epi32(in2, in3);			\
    __m128i tmp2 = _mm_unpackhi_epi32(in0, in1);			\
    __m128i tmp3 = _mm_unpackhi_epi32(in2, in3);			\
    in0 = _mm_unpacklo_epi64(tmp0, tmp1);				\
    in1 = _mm_unpackhi_epi64(tmp0, tmp1);				\
    in2 = _mm_unpacklo_epi64(tmp2, tmp3);				\
    in3 = _mm_unpackhi_epi64(tmp2, tmp3);				\
  }

#define transpose_4x4_m128(in0, in1, in2, in3)				\
  {									\
    __m128 tmp0 = _mm_unpacklo_ps(in0, in1);				\
    __m128 tmp1 = _mm_unpacklo_ps(in2, in3);				\
    __m128 tmp2 = _mm_unpackhi_ps(in0, in1);				\
    __m128 tmp3 = _mm_unpackhi_ps(in2, in3);				\
    in0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(tmp0), _mm_castps_pd(tmp1))); \
    in1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(tmp0), _mm_castps_pd(tmp1))); \
    in2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(tmp2), _mm_castps_pd(tmp3))); \
    in3 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(tmp2), _mm_castps_pd(tmp3))); \
  }

  // array1 = theta1(atom1) theta1(atom2)
  // array2 = theta2(atom1) theta2(atom2)
  // array3 = theta3(atom1) theta3(atom2)
  // array4 = theta4(atom1) theta4(atom2)

  // array1 = theta1(atom1) theta2(atom1)
  // array2 = theta3(atom1) theta4(atom1)
  // array3 = theta1(atom2) theta2(atom2)
  // array4 = theta3(atom2) theta4(atom2)

  // Transpose array -matrix
#define transpose_4x2_m128d(in0, in1, in2, in3)			\
  {								\
  __m128d tmp0 = _mm_unpacklo_pd(in0, in1);			\
  __m128d tmp1 = _mm_unpacklo_pd(in2, in3);			\
  __m128d tmp2 = _mm_unpackhi_pd(in0, in1);			\
  __m128d tmp3 = _mm_unpackhi_pd(in2, in3);			\
  in0 = tmp0;							\
  in1 = tmp1;							\
  in2 = tmp2;							\
  in3 = tmp3;							\
  }
