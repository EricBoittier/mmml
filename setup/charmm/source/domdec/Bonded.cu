#ifndef NOCUDAC
#include "Bonded.h"

//
// Class creator
//
Bonded::Bonded(CudaEnergyVirial& energyVirial) :
  strBond("bond"), strUreyb("ureyb"), strAngle("angle"),
  strDihe("dihe"), strImdihe("imdihe"), strCmap("cmap"),
  energyVirial(energyVirial),
  bondedforce(energyVirial, strBond.c_str(), strUreyb.c_str(), strAngle.c_str(), strDihe.c_str(),
	      strImdihe.c_str(), strCmap.c_str()) {

  // Energy terms
  energyVirial.insert(strBond);
  energyVirial.insert(strUreyb);
  energyVirial.insert(strAngle);
  energyVirial.insert(strDihe);
  energyVirial.insert(strImdihe);
  energyVirial.insert(strCmap);

  //prev_calc_energy = false;
  //prev_calc_virial = false;
  // Create stream
  cudaCheck(cudaStreamCreate(&stream));
  // Create event
  cudaCheck(cudaEventCreate(&force_done_event));
}

//
// Class destructor
//
Bonded::~Bonded() {
  // Destroy stream
  cudaCheck(cudaStreamDestroy(stream));
  // Destroy event
  cudaCheck(cudaEventDestroy(force_done_event));
}

//
// Setup bonded coefficients
//
void Bonded::setup_coef(const int nbondcoef, const float2 *h_bondcoef,
			const int nureybcoef, const float2 *h_ureybcoef,
			const int nanglecoef, const float2 *h_anglecoef,
			const int ndihecoef, const float4 *h_dihecoef,
			const int nimdihecoef, const float4 *h_imdihecoef,
			const int ncmapcoef, const float2 *h_cmapcoef) {
  bondedforce.setup_coef(nbondcoef, h_bondcoef,
			 nureybcoef, h_ureybcoef,
			 nanglecoef, h_anglecoef,
			 ndihecoef, h_dihecoef,
			 nimdihecoef, h_imdihecoef,
			 ncmapcoef, h_cmapcoef);
}

//
// Setup bonded lists
//
void Bonded::setup_list(const int nbondlist, const bondlist_t *h_bondlist, 
			const int nureyblist, const bondlist_t *h_ureyblist,
			const int nanglelist, const anglelist_t *h_anglelist,
			const int ndihelist, const dihelist_t *h_dihelist,
			const int nimdihelist, const dihelist_t *h_imdihelist,
			const int ncmaplist, const cmaplist_t *h_cmaplist) {
  bondedforce.setup_list(nbondlist, h_bondlist, 
			 nureyblist, h_ureyblist,
			 nanglelist, h_anglelist,
			 ndihelist, h_dihelist,
			 nimdihelist, h_imdihelist,
			 ncmaplist, h_cmaplist, stream);
}

//
// Calculates bonded forces
//
void Bonded::calc_bonded_force(const double boxx, const double boxy, const double boxz,
			       const bool calc_energy, const bool calc_virial,
			       const bool calc_bond, const bool calc_ureyb,
			       const bool calc_angle, const bool calc_dihe,
			       const bool calc_imdihe, const bool calc_cmap,
			       XYZQ& xyzq, Force<long long int>& force,
			       cudaEvent_t clear_done_event) {

  cudaCheck(cudaStreamWaitEvent(stream, clear_done_event, 0));
  
  bondedforce.calc_force(xyzq.xyzq, boxx, boxy, boxz, calc_energy, calc_virial,
			 force.stride(), force.xyz(), calc_bond, calc_ureyb,
			 calc_angle, calc_dihe, calc_imdihe, calc_cmap, stream);

  cudaCheck(cudaEventRecord(force_done_event, stream));
}

//
// GPU stream "wait_stream" waits until force computation is done
//
void Bonded::gpu_wait_force_done(cudaStream_t wait_stream) {
  cudaCheck(cudaStreamWaitEvent(wait_stream, force_done_event, 0));
}

//
// Get energies and virials
//
void Bonded::get_energy(double& energy_bond, double& energy_ureyb,
			double& energy_angle,
			double& energy_dihe, double& energy_imdihe,
			double& energy_cmap) {
  //bondedforce.get_energy_virial(prev_calc_energy, prev_calc_virial,
  //				energy_bond, energy_ureyb, energy_angle,
  //				energy_dihe, energy_imdihe, energy_cmap,
  //				sforce);
  energy_bond    = energyVirial.getEnergy(strBond);
  energy_ureyb   = energyVirial.getEnergy(strUreyb);
  energy_angle   = energyVirial.getEnergy(strAngle);
  energy_dihe    = energyVirial.getEnergy(strDihe);
  energy_imdihe  = energyVirial.getEnergy(strImdihe);
  energy_cmap    = energyVirial.getEnergy(strCmap);
}

#endif //NOCUDAC
