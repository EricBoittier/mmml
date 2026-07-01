#ifndef NOCUDAC
#ifndef NBRECIP_H
#define NBRECIP_H
#include "../domdec_gpu/Force.h"
#include "../domdec_gpu/XYZQ.h"
#include "../domdec_gpu/CudaDomdecRecip.h"

//
// Storage class for non-bonded reciprocal calculation on the GPU
// (c) Antti-Pekka Hynninen, 2015
//

class MDsim;

class NBrecip {
  friend class MDsim;

private:

  //-------------------
  // Energy and virial
  //-------------------
  std::string strEwks;
  std::string strEwse;
  CudaEnergyVirial& energyVirial;
  
  //-------------
  // Coordinates
  //-------------
  XYZQ xyzq;
  int h_xyzq_len;
  float4 *h_xyzq;

  // ----------------------------
  // Reciprocal force calculator
  // ----------------------------
  CudaDomdecRecip recip;

  // Stream for force computation
  cudaStream_t stream;

  // Event for force computation
  cudaEvent_t force_done_event;
  
  //---------
  // Forces
  //---------
  Force<float> force;

  NBrecip(const int nfftx, const int nffty, const int nfftz,
	  const int forder, const double kappa, CudaEnergyVirial& energyVirial);
  ~NBrecip();
  void set_xyzq(const int ncoord, const double *h_x, const double *h_y, const double *h_z, const double *h_q);
  void calc_recip_force(const double inv_boxx, const double inv_boxy, const double inv_boxz,
			const bool calc_energy, const bool calc_virial, cudaEvent_t clear_done_event,
			XYZQ* xyzqp=NULL, Force<long long int>* forcep=NULL);
  
  void calc_recip_force_block(const double inv_boxx, const double inv_boxy, const double inv_boxz,
                              const bool calc_energy, const bool calc_virial,
                              cudaEvent_t clear_done_event,
                              float * bixlam,
                              double * biflam, const int * blockIndexes,
                              XYZQ * xyzqp = NULL,
                              Force<long long int> * forcep = NULL);
  
  void gpu_wait_force_done(cudaStream_t wait_stream);
  void get_energy(double& energy_ksum, double& energy_self);
};

#endif // NBRECIP_H
#endif //NOCUDAC
