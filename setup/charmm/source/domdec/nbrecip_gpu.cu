#ifndef NOCUDAC

#include "domdec_gpu.h"

#if defined(CUDA_VERSION) && (CUDA_VERSION < 4020)
#error "CUDA version 4.2 or newer required."
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#error "CUDA compute capability 2.0 or higher required."
#endif

//----------------------------------------------------------------------------------------
//
// Calculates reciprocal forces on the GPU (wrapper)
//
extern "C" void calc_recip_gpu(double *boxx, double *boxy, double *boxz,
			       int *h_q_calc_energy, int *h_q_calc_virial) {
  bool calc_energy = (*h_q_calc_energy != 0);
  bool calc_virial = (*h_q_calc_virial != 0);
  mdsim->set_box_size(*boxx, *boxy, *boxz);
  mdsim->calc_recip_force(calc_energy, calc_virial);
}

//----------------------------------------------------------------------------------------
//
// Calculates reciprocal forces on the GPU (wrapper)
//
extern "C" void calc_recip_block_gpu(double *boxx, double *boxy, double *boxz,
                                     int *h_q_calc_energy, int *h_q_calc_virial) {
  bool calc_energy = (*h_q_calc_energy != 0);
  bool calc_virial = (*h_q_calc_virial != 0);
  mdsim->set_box_size(*boxx, *boxy, *boxz);
  mdsim->calc_recip_force_block(calc_energy, calc_virial);
}


//----------------------------------------------------------------------------------------

//
// Reads the GPU-calculated reciprocal energies
//
extern "C" void read_recip_energy_gpu(double *h_ewksum, double *h_ewself) {
  double ewksum_tmp, ewself_tmp;
  mdsim->get_recip_energy(ewksum_tmp, ewself_tmp);
  *h_ewksum += ewksum_tmp;
  *h_ewself += ewself_tmp;
}

//----------------------------------------------------------------------------------------

//
// Reads the GPU-calculated reciprocal virial
//
extern "C" void read_recip_virial_gpu(double *h_ewvirial) {
  mdsim->get_recip_virial(h_ewvirial);
}

#endif // NOCUDAC
