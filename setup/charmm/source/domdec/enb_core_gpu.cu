#ifndef NOCUDAC

#include <cuda.h>
#include "../domdec_gpu/cuda_utils.h"
#include "../domdec_gpu/CudaPMEDirectForce.h"
#include "../domdec_gpu/Force.h"
#include "domdec_gpu.h"

#if defined(CUDA_VERSION) && (CUDA_VERSION < 4020)
#error "CUDA version 4.2 or newer required."
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#error "CUDA compute capability 2.0 or higher required."
#endif

//----------------------------------------------------------------------------------------
//
// Host wrapper for enb_tilex_kernel
//
extern "C" void enb_tilex_gpu(int *h_whichlist, int *h_q_calc_energy, int *h_q_calc_virial,
			      double *boxx, double *boxy, double *boxz) {

  bool calc_energy = (*h_q_calc_energy != 0);
  bool calc_virial = (*h_q_calc_virial != 0);
  mdsim->set_box_size(*boxx, *boxy, *boxz);  
  mdsim->calc_nonbond_force(*h_whichlist-1, calc_energy, calc_virial);

}

//----------------------------------------------------------------------------------------
//
// Sets parameters for the GPU non-bonded calculation
//
extern "C" void set_nonbond_param_gpu(double *h_kappa,
				      double *h_roff, double *h_ron,
				      double *h_e14fac,
				      int *h_vdwmodel, int *h_elecmodel,
				      int *h_qeterm_vdw, int *h_qeterm_elec) {
  float new_kappa = (float)(*h_kappa);
  float new_roff = (float)(*h_roff);
  float new_ron = (float)(*h_ron);
  float new_e14fac = (float)(*h_e14fac);
  int new_vdwmodel = *h_vdwmodel;
  int new_elecmodel = *h_elecmodel;
  bool new_qeterm_vdw = (*h_qeterm_vdw != 0);
  bool new_qeterm_elec = (*h_qeterm_elec != 0);
  mdsim->set_nonbond_param(new_kappa, new_roff, new_ron, new_e14fac,
			   new_vdwmodel, new_elecmodel, new_qeterm_vdw, new_qeterm_elec);
}

//----------------------------------------------------------------------------------------
//
// Sets box size
//
extern "C" void set_box_size_gpu(double *boxx, double *boxy, double *boxz) {
  mdsim->set_box_size(*boxx, *boxy, *boxz);
}
//----------------------------------------------------------------------------------------

//
// Reads the GPU-calculated non-bonded energies (vdwpot, coulpot) and virials and adds them
// NOTE: This can be safely called only after forces are communicated
//       (and hence force calculation is done)
//
extern "C" void read_nonbond_energy_gpu(double *h_vdwpot, double *h_coulpot,
					double *h_exclpot) {

  double vdwpot_tmp, coulpot_tmp, exclpot_tmp;
  
  mdsim->get_nonbond_energy(vdwpot_tmp, coulpot_tmp, exclpot_tmp);

  *h_vdwpot  += vdwpot_tmp;
  *h_coulpot += coulpot_tmp;
  *h_exclpot += exclpot_tmp;
}

//----------------------------------------------------------------------------------------

//
// Reads the GPU-calculated virials and adds them to the existing values
//
extern "C" void read_direct_virial_gpu(double *h_vir, double *h_virtensor) {

  double virtensor_tmp[9];

  mdsim->get_direct_virial(virtensor_tmp);

  for (int i=0;i < 9;i++) h_virtensor[i] += virtensor_tmp[i];

  *h_vir += (virtensor_tmp[0] + virtensor_tmp[4] + virtensor_tmp[8])/3.0;

}

//----------------------------------------------------------------------------------------
#endif

