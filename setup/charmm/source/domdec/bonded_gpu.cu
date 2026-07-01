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
// Setup bonded force coefficients on the GPU (wrapper)
//
extern "C" void setup_bonded_coef_gpu(int* nbond, double* cbb, double* cbc,
				      int* ntheta, double* ctb, double* ctc,
				      int* nureyb, double* ctub, double* ctuc,
				      int* nphi, int* cpd, double* cpc, double* cpsin, double* cpcos,
				      int* nimphi, int* cid, double* cic, double* cisin, double* cicos) {
  
  float2 *h_bondcoef = (*nbond > 0) ? new float2[*nbond] : NULL;
  float2 *h_ureybcoef = (*nureyb > 0) ?new float2[*nureyb] : NULL;
  float2 *h_anglecoef = (*ntheta > 0) ? new float2[*ntheta] : NULL;
  float4 *h_dihecoef = (*nphi > 0) ? new float4[*nphi] : NULL;
  float4 *h_imdihecoef = (*nimphi > 0) ? new float4[*nimphi] : NULL;

  for (int i=0;i < *nbond;i++) {
    h_bondcoef[i].x = cbb[i];
    h_bondcoef[i].y = cbc[i];
  }

  for (int i=0;i < *ntheta;i++) {
    h_anglecoef[i].x = ctb[i];
    h_anglecoef[i].y = ctc[i];
  }

  for (int i=0;i < *nureyb;i++) {
    h_ureybcoef[i].x = ctub[i];
    h_ureybcoef[i].y = ctuc[i];
  }

  for (int i=0;i < *nphi;i++) {
    h_dihecoef[i].x = cpd[i];
    h_dihecoef[i].y = cpc[i];
    h_dihecoef[i].z = cpsin[i];
    h_dihecoef[i].w = cpcos[i];
  }

  for (int i=0;i < *nimphi;i++) {
    h_imdihecoef[i].x = cid[i];
    h_imdihecoef[i].y = cic[i];
    h_imdihecoef[i].z = cisin[i];
    h_imdihecoef[i].w = cicos[i];
  }

  mdsim->setup_bonded_coef(*nbond, h_bondcoef,
			   *nureyb, h_ureybcoef,
			   *ntheta, h_anglecoef,
			   *nphi, h_dihecoef,
			   *nimphi, h_imdihecoef,
			   0, NULL);

  if (h_bondcoef != NULL) delete [] h_bondcoef;
  if (h_ureybcoef != NULL) delete [] h_ureybcoef;
  if (h_anglecoef != NULL) delete [] h_anglecoef;
  if (h_dihecoef != NULL) delete [] h_dihecoef;
  if (h_imdihecoef != NULL) delete [] h_imdihecoef;
  
}

//----------------------------------------------------------------------------------------
//
// Setup bonded lists on the GPU (wrapper)
//
extern "C" void setup_bonded_list_gpu(int* nbondtbl, bondlist_t* bondlist,
				      int* nangletbl, anglelist_t* anglelist,
				      int* nureybtbl, bondlist_t* ureyblist,
				      int* ndihetbl, dihelist_t* dihelist,
				      int* nimdihetbl, dihelist_t* imdihelist) {
  mdsim->setup_bonded_list(*nbondtbl, bondlist,
			   *nureybtbl, ureyblist,
			   *nangletbl, anglelist,
			   *ndihetbl, dihelist,
			   *nimdihetbl, imdihelist,
			   0, NULL);
}

//----------------------------------------------------------------------------------------
//
// Calculates bonded forces on the GPU (wrapper)
//
extern "C" void calc_bonded_gpu(double *boxx, double *boxy, double *boxz,
				int *q_calc_energy, int *q_calc_virial,
				int *q_calc_bond, int *q_calc_angle,
				int *q_calc_ureyb, int *q_calc_dihe,
				int *q_calc_imdihe) {
  bool calc_energy = (*q_calc_energy != 0);
  bool calc_virial = (*q_calc_virial != 0);
  bool calc_bond   = (*q_calc_bond != 0);
  bool calc_ureyb  = (*q_calc_ureyb != 0);
  bool calc_angle  = (*q_calc_angle != 0);
  bool calc_dihe   = (*q_calc_dihe != 0);
  bool calc_imdihe = (*q_calc_imdihe != 0);
  bool calc_cmap   = false;
  mdsim->set_box_size(*boxx, *boxy, *boxz);
  mdsim->calc_bonded_force(calc_energy, calc_virial,
			   calc_bond, calc_ureyb, calc_angle, calc_dihe,
			   calc_imdihe, calc_cmap);
}

//----------------------------------------------------------------------------------------

//
// Reads the GPU-calculated bonded energy
// NOTE: cmap is currently skipped
//
extern "C" void read_bonded_energy_gpu(double *energy_bond, double *energy_angle,
				       double *energy_ureyb, double *energy_dihe,
				       double *energy_imdihe) {
  double energy_bond_tmp, energy_angle_tmp, energy_ureyb_tmp, energy_dihe_tmp,
    energy_imdihe_tmp, energy_cmap_tmp;
  mdsim->get_bonded_energy(energy_bond_tmp, energy_ureyb_tmp, energy_angle_tmp,
			   energy_dihe_tmp, energy_imdihe_tmp, energy_cmap_tmp);
  *energy_bond   += energy_bond_tmp;
  *energy_ureyb  += energy_ureyb_tmp;
  *energy_angle  += energy_angle_tmp;
  *energy_dihe   += energy_dihe_tmp;
  *energy_imdihe += energy_imdihe_tmp;
}

#endif // NOCUDAC
