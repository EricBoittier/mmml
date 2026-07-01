#ifndef NOCUDAC
#ifndef BONDED_H
#define BONDED_H

#include <string>
#include "../domdec_gpu/CudaBondedForce.h"
#include "../domdec_gpu/Force.h"
#include "../domdec_gpu/XYZQ.h"

//
// Interface class for bonded computation on the GPU
// (c) Antti-Pekka Hynninen 2015
//

class MDsim;

class Bonded {
  friend class MDsim;

private:

  //bool prev_calc_energy;
  //bool prev_calc_virial;

  // Energy and virial
  std::string strBond;
  std::string strUreyb;
  std::string strAngle;
  std::string strDihe;
  std::string strImdihe;
  std::string strCmap;
  CudaEnergyVirial& energyVirial;
  
  // Bonded force calculator
  CudaBondedForce<long long int, float> bondedforce;

  // Stream for force computation
  cudaStream_t stream;

  // Event for force computation
  cudaEvent_t force_done_event;

  Bonded(CudaEnergyVirial& energyVirial);
  ~Bonded();
  void setup_coef(const int nbondcoef, const float2 *h_bondcoef,
		  const int nureybcoef, const float2 *h_ureybcoef,
		  const int nanglecoef, const float2 *h_anglecoef,
		  const int ndihecoef, const float4 *h_dihecoef,
		  const int nimdihecoef, const float4 *h_imdihecoef,
		  const int ncmapcoef, const float2 *h_cmapcoef);
  void setup_list(const int nbondlist, const bondlist_t *h_bondlist, 
		  const int nureyblist, const bondlist_t *h_ureyblist,
		  const int nanglelist, const anglelist_t *h_anglelist,
		  const int ndihelist, const dihelist_t *h_dihelist,
		  const int nimdihelist, const dihelist_t *h_imdihelist,
		  const int ncmaplist, const cmaplist_t *h_cmaplist);
  void calc_bonded_force(const double boxx, const double boxy, const double boxz,
			 const bool calc_energy, const bool calc_virial,
			 const bool calc_bond, const bool calc_ureyb,
			 const bool calc_angle, const bool calc_dihe,
			 const bool calc_imdihe, const bool calc_cmap,
			 XYZQ& xyzq, Force<long long int>& force,
			 cudaEvent_t clear_done_event);
  void gpu_wait_force_done(cudaStream_t wait_stream);
  void get_energy(double& energy_bond, double& energy_ureyb,
		  double& energy_angle,
		  double& energy_dihe, double& energy_imdihe,
		  double& energy_cmap);
};

#endif // BONDED_H
#endif //NOCUDAC
