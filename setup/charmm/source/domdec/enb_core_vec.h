#ifndef ENBCOREVEC_H
#define ENBCOREVEC_H

template<typename T>
struct xyzq_t {
  T x, y, z, q;
#if KEY_LJPME==1
  T c6;
#endif
};

#ifdef __SSE2__

//
// Non-bonded loops implemented using Intel (SSE/AVX) intrinsics
//
// (c) Antti-Pekka Hynninen, April 2014. aphynninen@hotmail.com
//

enum {NONE, LOOKUP, VSH, VSW, VFSW};

//
// T   = computation type (precision) = float, double
// VT  = vector type = __m128, __m128d, __m256, __m256d
// AVT = accumulation vector type = __m128d, __m256
//
template <typename T, typename VT, typename AVT>
class enb_core_vec {

private:
  static const int veclen = sizeof(VT)/sizeof(T);

public:
  enb_core_vec();
  ~enb_core_vec();

  float invsqrt_approx(const float xf_in);

  template <int coul_type, int vdw_type>
  void enb_vv(const int ni, const int *indi, const int *indj, const int *startj,
	      const T *pftable, const T hinv,
	      const xyzq_t<T> *xyzq, double *force, const int *iscoord,
	      double *sforce, const T *scoordtab,
	      double *coulpot, double *vdwpot, 
	      const T c6OO, const T c12OO, const T c6OH, const T c12OH, const T c6HH, const T c12HH,
#if KEY_LJPME==1
	      const T c6multOO, const T c6multOH, const T c6multHH,
#endif
          const T qOO, const T qOH, const T qHH,
	      const T roff, const T ron);

  template <int coul_type, int vdw_type>
  void enb_uv_coul_vdw(const int ni, const int *indi, const int *indj, const int *startj,
		       const T *pftable, const T hinv,
		       const int *vdwtype, const T *vdwparam,
		       const xyzq_t<T> *xyzq, double *force, const int *iscoord,
		       double *sforce, const T *scoordtab,
		       double *coulpot, double *vdwpot, 
		       const T roff, const T ron);

  template <int coul_type, int vdw_type>
  void enb_uu_coul_vdw(const int ni, const int *indi, const int *indj,
		       const int *startj,
		       const T *pftable, const T hinv,
		       const int *vdwtype, const T *vdwparam,
		       const xyzq_t<T> *xyzq, double *force, const int *iscoord,
		       double *sforce, const T *scoordtab,
		       double *coulpot, double *vdwpot, 
		       const T roff, const T ron);

  void calc_vv(const int ni, const int *indi, const int *indj, const int *startj,
	       const T *pftable, const T hinv,
	       const xyzq_t<T> *xyzq, double *force, const int *iscoord,
	       double *sforce, const T *scoordtab,
	       double *coulpot, double *vdwpot, 
	       const T c6OO, const T c12OO, const T c6OH, const T c12OH, const T c6HH, const T c12HH,
#if KEY_LJPME==1
           const T c6multOO, const T c6multOH, const T c6multHH,
#endif
           const T qOO, const T qOH, const T qHH,
	       const T roff, const T ron, const int vdw_type);

  void calc_uv_coul_vdw(const int ni, const int *indi, const int *indj,
			const int *startj,
			const T *pftable, const T hinv,
			const int *vdwtype, const T *vdwparam, 
			const xyzq_t<T> *xyzq, double *force,
			const int *iscoord,
			double *sforce, const T *scoordtab,
			double *coulpot, double *vdwpot, 
			const T roff, const T ron, const int vdw_type);

  void calc_uv_vdw(const int ni, const int *indi, const int *indj,
		   const int *startj,
		   const T *pftable, const T hinv,
		   const int *vdwtype, const T *vdwparam, 
		   const xyzq_t<T> *xyzq, double *force,
		   const int *iscoord,
		   double *sforce, const T *scoordtab,
		   double *vdwpot, 
		   const T roff, const T ron, const int vdw_type);

  void calc_uu_coul_vdw(const int ni, const int *indi, const int *indj,
			const int *startj,
			const T *pftable, const T hinv,
			const int *vdwtype, const T *vdwparam, 
			const xyzq_t<T> *xyzq, double *force,
			const int *iscoord,
			double *sforce, const T *scoordtab,
			double *coulpot, double *vdwpot, 
			const T roff, const T ron, const int vdw_type);

  void calc_uu_vdw(const int ni, const int *indi, const int *indj,
		   const int *startj,
		   const T *pftable, const T hinv,
		   const int *vdwtype, const T *vdwparam, 
		   const xyzq_t<T> *xyzq, double *force,
		   const int *iscoord,
		   double *sforce, const T *scoordtab,
		   double *vdwpot, 
		   const T roff, const T ron, const int vdw_type);
};

#endif // __SSE2__

#endif // ENBCOREVEC_H
