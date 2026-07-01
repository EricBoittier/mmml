#ifndef NOCUDAC
#ifndef CUDADOMDECRECIP_H
#define CUDADOMDECRECIP_H

#include "DomdecRecip.h"
#include "Force.h"
#include "XYZQ.h"
#include "CudaPMERecip.h"

class CudaDomdecRecip : public DomdecRecip {

 private:
  CudaPMERecip<int, float, float2> PMErecip;

  // Energy and virial
  CudaEnergyVirial& energyVirial;
  
  void solvePoisson(const double inv_boxx, const double inv_boxy, const double inv_boxz,
		    const float4* coord, const int ncoord,
		    const bool calc_energy, const bool calc_virial, double *recip) {
    for (int i=0;i < 9;i++) recip[i] = 0.0;
    recip[0] = inv_boxx;
    recip[4] = inv_boxy;
    recip[8] = inv_boxz;
    PMErecip.spread_charge(coord, ncoord, recip);
    PMErecip.r2c_fft();
    PMErecip.scalar_sum(recip, kappa, calc_energy, calc_virial);
    PMErecip.c2r_fft();
  }

  void solvePoissonBlock(const double inv_boxx, const double inv_boxy, const double inv_boxz,
		         const float4* coord, const int ncoord,
		         const bool calc_energy, const bool calc_virial,
                         const float * bixlam, const int * blockIndexes,
                         double *recip) {
    for (int i=0;i < 9;i++) recip[i] = 0.0;
    recip[0] = inv_boxx;
    recip[4] = inv_boxy;
    recip[8] = inv_boxz;
    PMErecip.spread_charge_block(coord, ncoord, bixlam, blockIndexes, recip);
    PMErecip.r2c_fft();
    PMErecip.scalar_sum(recip, kappa, calc_energy, calc_virial);
    PMErecip.c2r_fft();
  }

 public:
  CudaDomdecRecip(const int nfftx, const int nffty, const int nfftz, const int order, const double kappa,
		  CudaEnergyVirial& energyVirial, const char* nameRecip, const char* nameSelf) : 
    DomdecRecip(nfftx, nffty, nfftz, order, kappa), energyVirial(energyVirial),
    PMErecip(nfftx, nffty, nfftz, order, BOX, 1, 0, energyVirial, nameRecip, nameSelf) {}

  ~CudaDomdecRecip() {}

  void set_stream(cudaStream_t stream) {PMErecip.set_stream(stream);}

  //void clear_energy_virial() {grid.clear_energy_virial();}

  //void get_energy_virial(const bool calc_energy, const bool calc_virial,
  //			 double& energy, double& energy_self, double *virial) {
  // grid.get_energy_virial(kappa, calc_energy, calc_virial, energy, energy_self, virial);
  //}

  //
  // Strided add into Force<long long int>
  //
  void calc(const double inv_boxx, const double inv_boxy, const double inv_boxz,
	    const float4* coord, const int ncoord,
	    const bool calc_energy, const bool calc_virial, Force<long long int>& force) {
    double recip[9];
    solvePoisson(inv_boxx, inv_boxy, inv_boxz, coord, ncoord, calc_energy, calc_virial, recip);
    PMErecip.gather_force(coord, ncoord, recip, force.stride(), force.xyz());
    if (calc_energy) PMErecip.calc_self_energy(coord, ncoord, this->kappa);
  }

  //
  // Strided store into Force<float>
  //
  void calc(const double inv_boxx, const double inv_boxy, const double inv_boxz,
	    const float4* coord, const int ncoord,
	    const bool calc_energy, const bool calc_virial, Force<float>& force) {
    double recip[9];
    solvePoisson(inv_boxx, inv_boxy, inv_boxz, coord, ncoord, calc_energy, calc_virial, recip);
    PMErecip.gather_force(coord, ncoord, recip, force.stride(), force.xyz());
    if (calc_energy) PMErecip.calc_self_energy(coord, ncoord, this->kappa);
  }

  //
  // Non-strided store info XYZQ
  //
  void calc(const double inv_boxx, const double inv_boxy, const double inv_boxz,
	    const float4* coord, const int ncoord,
	    const bool calc_energy, const bool calc_virial,
            float3* force) {
    double recip[9];
    solvePoisson(inv_boxx, inv_boxy, inv_boxz, coord, ncoord, calc_energy, calc_virial, recip);
    PMErecip.gather_force(coord, ncoord, recip, 1, force);
    if (calc_energy) PMErecip.calc_self_energy(coord, ncoord, this->kappa);
  }

  // begin block calc
  
  // Strided add into Force<long long int>
  void calc_block(const double inv_boxx,
                  const double inv_boxy,
                  const double inv_boxz,
                  const float4 * coord, const int ncoord,
                  const float * bixlam,
                  const bool calc_energy, const bool calc_virial,
                  Force<long long int>& force, double * biflam, // outputs
                  const int * blockIndexes) {
    double recip[9];
    solvePoissonBlock(inv_boxx, inv_boxy, inv_boxz, coord, ncoord, calc_energy, calc_virial, bixlam, blockIndexes, recip);
    PMErecip.gather_force_block(coord, ncoord, recip, force.stride(),
                                force.xyz(), bixlam, biflam, blockIndexes);
    // if (calc_energy) PMErecip.calc_self_energy(coord, ncoord, this->kappa);
    // MSLDPME ->
    PMErecip.calc_self_energy_block(coord, ncoord, this->kappa, bixlam, biflam, blockIndexes);
  }

  // Strided store into Force<float>
  void calc_block(const double inv_boxx, const double inv_boxy, const double inv_boxz,
	    const float4* coord, const int ncoord,
            const float * bixlam,
	    const bool calc_energy, const bool calc_virial,
                  Force<float>& force, double * biflam, // outputs
                  const int * blockIndexes) {
    double recip[9];
    solvePoissonBlock(inv_boxx, inv_boxy, inv_boxz, coord, ncoord, calc_energy, calc_virial, bixlam, blockIndexes, recip);
    PMErecip.gather_force_block(coord, ncoord, recip, force.stride(),
                                force.xyz(), bixlam, biflam, blockIndexes);
    // if (calc_energy) PMErecip.calc_self_energy(coord, ncoord, this->kappa);
    // MSLDPME ->
    PMErecip.calc_self_energy_block(coord, ncoord, this->kappa, bixlam, biflam, blockIndexes);
  }

  // Non-strided store info XYZQ
  void calc_block(const double inv_boxx, const double inv_boxy, const double inv_boxz,
	    const float4* coord, const int ncoord,
            const float * bixlam,
	    const bool calc_energy, const bool calc_virial,
                  float3 * force, double * biflam, // outputs
                  const int * blockIndexes) {
    double recip[9];
    solvePoissonBlock(inv_boxx, inv_boxy, inv_boxz, coord, ncoord, calc_energy, calc_virial, bixlam, blockIndexes, recip);
    PMErecip.gather_force_block(coord, ncoord, recip, 1,
                                force, bixlam, biflam, blockIndexes);
    // if (calc_energy) PMErecip.calc_self_energy(coord, ncoord, this->kappa);
    // MSLDPME ->
    PMErecip.calc_self_energy_block(coord, ncoord, this->kappa, bixlam, biflam, blockIndexes);
  }

  // end block calc
};

#endif // CUDADOMDECRECIP_H
#endif //NOCUDAC
