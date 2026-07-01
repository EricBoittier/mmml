#ifndef NOCUDAC
#ifndef XYZQ_H
#define XYZQ_H

//
// XYZQ class
//
// (c) Antti-Pekka Hynninen, 2013, aphynninen@hotmail.com
//
//

#include "gpu_utils.h"
#include "cudaXYZ.h"

class XYZQ {

private:
  int get_xyzq_len(const int ncoord_in);

public:
  int align;
  int ncoord;
  int xyzq_len;
  float4 *xyzq;

  XYZQ();
  XYZQ(int ncoord, int align=warpsize);
  XYZQ(const char *filename, int align=warpsize);
  ~XYZQ();

  void realloc(int ncoord_new, float fac=1.0f);
  void resize(int ncoord_new, float fac=1.0f);
  void set_xyzq(int ncopy, float4 *h_xyzq, size_t offset=0, cudaStream_t stream=0);

  void set_xyzq(const cudaXYZ<double>& coord, const float *q, cudaStream_t stream=0);
  void set_xyzq(const cudaXYZ<double>& coord, const float *q, const int *loc2glo,
		const float3 *xyz_shift, const double boxx, const double boxy, const double boxz,
		cudaStream_t stream=0);

  void set_xyz(const cudaXYZ<double>& coord, cudaStream_t stream=0);
  void set_xyz(const cudaXYZ<double>& coord, const int start, const int end, const float3 *xyz_shift,
	       const double boxx, const double boxy, const double boxz, cudaStream_t stream=0);

  bool compare(XYZQ& xyzq_in, const double tol, double& max_diff);

  void print(const int start, const int end, std::ostream& out);
  void save(const char* filename);
};

#endif // XYZQ_H
#endif //NOCUDAC
