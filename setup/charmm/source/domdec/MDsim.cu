#ifndef NOCUDAC
#include <iostream>
#include <cassert>
#include <cuda.h>
#include "MDsim.h"
#include "../domdec_gpu/cuda_utils.h"
#include "../domdec_gpu/gpu_utils.h"

//
// Class creator
//
MDsim::MDsim(const int ndirect, const bool direct_on, const bool recip_on, const bool bonded_on,
	     const bool nlist_on, const int nx, const int ny, const int nz,
	     const int ncoord_global, const int *iblo14, const int *inb14,
	     const int nfftx, const int nffty, const int nfftz,
	     const int forder, const double kappa, const int numBlock,
             const int use_softcore, const int use_PMEL, const int seed) {
  nbdirect = NULL;
  nbrecip = NULL;
  bonded = NULL;
  cudaBlock = NULL;
  nlist = NULL;

  ncoord_home = 0;
  ncoord_import = 0;
  ncoord = 0;
  
  // Set impossible box sizes
  boxx = -1.0;
  boxy = -1.0;
  boxz = -1.0;
  
  // Sanity check
  if (!direct_on && !recip_on) {
    std::cerr << "MDsim::MDsim, Must have either direct or recip calculation" << std::endl;
    exit(1);
  }

  // Create block calculator, only for direct nodes
  if (direct_on && numBlock >= 1) cudaBlock = new CudaBlock(numBlock,use_softcore,use_PMEL);

  if (direct_on && recip_on && ndirect > 1) {
    std::cerr << "MDsim::MDsim, Currently no support for mixed direct/recip nodes when ndirect > 1" << std::endl;
    exit(1);
  }

  if (direct_on) nbdirect = new NBdirect(energyVirialDirect, cudaBlock);
  recip_uses_nbdirect_buffers = false;
  if (recip_on) {
    nbrecip = new NBrecip(nfftx, nffty, nfftz, forder, kappa, energyVirialRecip);
    // Cases:
    // Single direct+recip node:                      true
    // Single direct+recip node + other direct nodes: false
    // Single recip node + one direct node:           false
    recip_uses_nbdirect_buffers = (ndirect==1) && direct_on;
  }

  if (bonded_on) {
    // Quick sanity check
    if (!direct_on) {
      std::cerr << "MDsim::MDsim, If bonded are ON, direct must be ON as well" << std::endl;
      exit(1);
    }
    bonded = new Bonded(energyVirialDirect);
  }

  if (nlist_on) {
    // Quick sanity check
    if (!direct_on) {
      std::cerr << "MDsim::MDsim, If neighborlist is ON, direct must be ON as well" << std::endl;
      exit(1);
    }
    nlist = new Nlist(ncoord_global, iblo14, inb14, nx, ny, nz);
  }
  
  // Start streams
  cudaCheck(cudaStreamCreate(&mdsim_stream));

  // Create events
  cudaCheck(cudaEventCreate(&copy_done_event));
  cudaCheck(cudaEventCreate(&clear_done_event));

  cur_calc_energy = false;
  cur_calc_virial = false;
  
  // Initialize host force array to zero
  h_force_double_len = 0;
  h_force_double = NULL;
  h_force_float_len = 0;
  h_force_float = NULL;

  cudaRandom = new CudaRandom(seed);
}

//
// Class destructor
//
MDsim::~MDsim() {
  // Destroy force calculators
  if (nbdirect != NULL) delete nbdirect;
  if (nbrecip != NULL) delete nbrecip;
  if (bonded != NULL) delete bonded;
  // Destroy BLOCK calculator
  if (cudaBlock != NULL) delete cudaBlock;
  // Destroy neighborlist builder
  if (nlist != NULL) delete nlist;
  // Stop streams
  cudaCheck(cudaStreamDestroy(mdsim_stream));
  // Destroy events
  cudaCheck(cudaEventDestroy(copy_done_event));
  cudaCheck(cudaEventDestroy(clear_done_event));
  // De-allocate host force array
  if (h_force_double != NULL) deallocate_host<double>(&h_force_double);
  if (h_force_float != NULL) deallocate_host<float>(&h_force_float);
  if (cudaRandom != NULL ) delete cudaRandom;
}

//
// Enables / disables testing
//
void MDsim::setTest(const bool test) {
  if (nlist != NULL) nlist->setTest(test);
}

//
// Sort coordinates for neighborlist build
//
void MDsim::sortCoord(const int whichlist, const int* zonelist_atom, float4* h_xyzq, int* h_loc2glo_ind) {
  assert(nlist != NULL);
  nlist->sortCoord(whichlist, zonelist_atom, h_xyzq, h_loc2glo_ind, xyzq);
}

//
// Wait for coordinate sorting to finish
//
void MDsim::waitSortCoord(const int whichlist) {
  assert(nlist != NULL);
  nlist->waitSortCoord(whichlist);
}

//
// Build neighborlist 
//
void MDsim::buildNeighborList(const int whichlist, const int* zonelist_atom, const double rnl) {
  assert(nlist != NULL);
  nlist->buildNeighborList(whichlist, zonelist_atom, rnl, this->boxx, this->boxy, this->boxz, xyzq);
}

//
// Set the number of coordinates
//
void MDsim::set_ncoord(int ncoord_home, int ncoord_import) {
  this->ncoord_home = ncoord_home;
  this->ncoord_import = ncoord_import;
  this->ncoord = ncoord_home + ncoord_import;
  // (x, y, z, q) -coordinates
  xyzq.realloc(this->ncoord, 1.5f);
  // Fixed Precision force
  force.realloc(ncoord, 1.5f);
  force.clear();
  if (recip_uses_nbdirect_buffers) {
    recipforce.realloc(ncoord, 1.5f);
    recipforce.clear();
  }
}

//
// Clears the force buffer.
// NOTE: There is no need to clear the recip buffer because it is set, not accumulated.
//
void MDsim::clear_force_virial_energy(const bool calc_energy, const bool calc_virial) {
  cur_calc_virial = calc_virial;
  cur_calc_energy = calc_energy;
  // Clear force array
  force.clear(mdsim_stream);
  if (recip_uses_nbdirect_buffers) recipforce.clear(mdsim_stream);
  // Clear energy and virial
  if (calc_energy && calc_virial) {
    if (nbdirect != NULL) energyVirialDirect.clear(mdsim_stream);
    if (nbrecip != NULL) energyVirialRecip.clear(mdsim_stream);
  } else if (calc_energy) {
    if (nbdirect != NULL) energyVirialDirect.clearEnergy(mdsim_stream);
    if (nbrecip != NULL) {
      energyVirialRecip.clearEnergy(mdsim_stream);
      // recip virial and eterm(ewksum) are calculated together
      energyVirialRecip.clearVirial(mdsim_stream);
    }
  } else if (calc_virial) {
    if (nbdirect != NULL) energyVirialDirect.clearVirial(mdsim_stream);
    if (nbrecip != NULL) {
      energyVirialRecip.clearVirial(mdsim_stream);
      // recip virial and eterm(ewksum) are calculated together
      energyVirialRecip.clearEtermDevice(nbrecip->strEwks, mdsim_stream);
    }
  }
  cudaCheck(cudaEventRecord(clear_done_event, mdsim_stream));
}

//
// Reduces forces and calculates virial
//
void MDsim::reduce_force_virial() {
  // GPU stream "mdsim_stream" waits until force computation is done
  if (nbdirect != NULL) nbdirect->gpu_wait_force_done(mdsim_stream);
  if (nbrecip != NULL) nbrecip->gpu_wait_force_done(mdsim_stream);
  if (bonded != NULL) bonded->gpu_wait_force_done(mdsim_stream);
  // Reduce forces
  force.convert<double>(mdsim_stream);
  // Calculate direct-space virial
  // NOTE: reciprocal-space virial is calculated on the fly in "scalar_sum" -function
  if (cur_calc_virial && nbdirect != NULL) {
    nbdirect->calc_virial(ncoord, boxx, boxy, boxz, xyzq, force, mdsim_stream);
  }
  if (recip_uses_nbdirect_buffers) force.add<double>(recipforce, mdsim_stream);
}

//
// Copies force, virial, and energy to host
//
void MDsim::get_force_virial_energy() {
  if (nbdirect != NULL) {
    // Direct + possibly Recip
    reallocate_host<double>(&h_force_double, &h_force_double_len, 3*force.stride(), 1.5f);    
    copy_DtoH<double>((double *)force.xyz(), h_force_double, 3*force.stride(),
		      mdsim_stream);
    if (cur_calc_energy || cur_calc_virial) {
      energyVirialDirect.copyToHost(mdsim_stream);
      if (nbrecip != NULL) energyVirialRecip.copyToHost(mdsim_stream);
    }
  } else {
    // Recip
    reallocate_host<float>(&h_force_float, &h_force_float_len, 3*nbrecip->force.stride(), 1.0f);
    copy_DtoH<float>(nbrecip->force.xyz(), h_force_float, 3*nbrecip->force.stride(), mdsim_stream);
    if (cur_calc_energy || cur_calc_virial) energyVirialRecip.copyToHost(mdsim_stream);
  }
  cudaCheck(cudaEventRecord(copy_done_event, mdsim_stream));
}

//
// Returns (void *) pointer to host buffer
//
void* MDsim::get_force_pointer() {
  if (nbdirect != NULL) {
    return (void *)h_force_double;
  } else {
    return (void *)h_force_float;
  }
}

//
// Returns force array stride
//
int MDsim::get_force_stride() {
  if (nbdirect != NULL) {
    return force.stride();
  } else {
    return nbrecip->force.stride();
  }
}

//
// Returns host force array type: 0=double, 1=float
//
int MDsim::get_force_type() {
  if (nbdirect != NULL) {
    return 0;
  } else {
    return 1;
  }
}

void MDsim::set_ijlist(int h_whichlist, int h_n_ijlist, int *h_ijlist) {
  assert(nbdirect != NULL);
  nbdirect->set_ijlist(h_whichlist, h_n_ijlist, h_ijlist);
}

void MDsim::set_ientry(int h_whichlist, int h_n_ientry, ientry_t* h_ientry) {
  assert(nbdirect != NULL);
  nbdirect->set_ientry(h_whichlist, h_n_ientry, h_ientry);
}

void MDsim::set_cell_start(int ncell, int *h_cell_start) {
  assert(nbdirect != NULL);
  nbdirect->set_cell_start(ncell, h_cell_start);
}

void MDsim::build_excl(int h_whichlist, float roff) {
  assert(nbdirect != NULL);
  nbdirect->build_excl(h_whichlist, (float)this->boxx, (float)this->boxy, (float)this->boxz, roff, this->xyzq);
}

void MDsim::combine_tile_top(int h_whichlist, int h_ntile_top, int *h_tile_ind_top, 
			     tile_excl_t<32> *h_tile_excl_top) {
  assert(nbdirect != NULL);
  nbdirect->combine_tile_top(h_whichlist, h_ntile_top, h_tile_ind_top, h_tile_excl_top);
}

//
// CPU waits until force, virial and energy copying is done
// NOTE: we use cudaEventSynchronize here because the CPU has to wait
//
void MDsim::cpu_wait_copy_done() {
  cudaCheck(cudaEventSynchronize(copy_done_event));
}

void MDsim::setBlockType(int ncoord, int *h_blockType) {
  // Sanity checks
  assert(cudaBlock != NULL);
  assert(nbdirect != NULL);
  assert(this->ncoord == ncoord);
  // Copy to GPU
  cudaBlock->setBlockType(ncoord, h_blockType);
}

void MDsim::setBlockParam(float *h_blockParam) {
  // Sanity checks
  assert(cudaBlock != NULL);
  assert(nbdirect != NULL);
  // Copy to GPU
  cudaBlock->setBlockParam(h_blockParam);
}

void MDsim::setBixlam(float *h_bixlam) {
  assert(cudaBlock != NULL);
  cudaBlock->setBixlam(h_bixlam);
}

void MDsim::setSiteMLD(int *h_siteMLD) {
  assert(cudaBlock != NULL);
  cudaBlock->setSiteMLD(h_siteMLD);
}

void MDsim::getBiflam(double *h_biflam, double *h_biflam2) {
  // Sanity checks
  assert(cudaBlock != NULL);
  assert(nbdirect != NULL);
  // Copy biflam and biflam2 to CPU
  cudaBlock->getBiflam(h_biflam, h_biflam2);
}

void MDsim::setBiflam(double *h_biflam, double *h_biflam2) {
  assert(cudaBlock != NULL);
  cudaBlock->setBiflam(h_biflam, h_biflam2);
}

/*
void MDsim::get_box_size(double& boxx_out, double& boxy_out, double& boxz_out) {
  boxx_out = this->boxx;
  boxy_out = this->boxy;
  boxz_out = this->boxz;
}
*/

//
// Sets box size. NOTE: Box size is only set if it differs from set value
//
void MDsim::set_box_size(const double boxx_in, const double boxy_in, const double boxz_in) {
  if (this->boxx == boxx_in && this->boxy == boxy_in && this->boxz == boxz_in) return;
  this->boxx = boxx_in;
  this->boxy = boxy_in;
  this->boxz = boxz_in;
  if (nbdirect != NULL) nbdirect->set_box_size(boxx, boxy, boxz);
}

void MDsim::set_nonbond_param(float kappa, float roff, float ron, float e14fac,
			      int vdwmodel, int elecmodel,
			      bool qeterm_vdw, bool qeterm_elec) {
  assert(nbdirect != NULL);
  nbdirect->set_nonbond_param((float)boxx, (float)boxy, (float)boxz, kappa, roff, ron, e14fac,
			      vdwmodel, elecmodel, qeterm_vdw, qeterm_elec);
}

void MDsim::set_vdwparam(int h_nvdwparam, float *h_vdwparam) {
  assert(nbdirect != NULL);
  nbdirect->set_vdwparam(h_nvdwparam, h_vdwparam);
}

void MDsim::set_vdwparam14(int h_nvdwparam, float *h_vdwparam) {
  assert(nbdirect != NULL);
  nbdirect->set_vdwparam14(h_nvdwparam, h_vdwparam);
}

void MDsim::set_14_list(int nin14list, int nex14list, xx14list_t* h_in14list, xx14list_t* h_ex14list) {
  assert(nbdirect != NULL);
  nbdirect->set_14_list(nin14list, nex14list, h_in14list, h_ex14list);
}

void MDsim::set_14_block_pos(int* in14tbl_block_pos, int* ex14tbl_block_pos) {
  assert(nbdirect != NULL);
  assert(cudaBlock != NULL);
  nbdirect->set_14_block_pos(cudaBlock->getNumBlock(), in14tbl_block_pos, ex14tbl_block_pos);
}

void MDsim::set_home_xyzq(float4 *h_xyzq) {
  assert(nbdirect != NULL);
  xyzq.set_xyzq(ncoord_home, h_xyzq, 0, nbdirect->get_nonbond_stream(0));
}

void MDsim::set_import_xyzq(float4 *h_xyzq) {
  assert(nbdirect != NULL);
  if (ncoord_import > 0)
    xyzq.set_xyzq(ncoord_import, h_xyzq, ncoord_home, nbdirect->get_nonbond_stream(1));
}

void MDsim::set_vdwtype(int *h_vdwtype) {
  assert(nbdirect != NULL);
  nbdirect->set_vdwtype(this->ncoord, h_vdwtype);
}

void MDsim::set_recip_xyzq(const int ncoord_recip, const double *h_x, const double *h_y, const double *h_z, const double *h_q) {
  assert(nbrecip != NULL);
  nbrecip->set_xyzq(ncoord_recip, h_x, h_y, h_z, h_q);
}

void MDsim::calc_nonbond_force(int whichlist, bool calc_energy, bool calc_virial) {
  assert(nbdirect != NULL);
  if (nlist == NULL) {
    // Using CPU created neighborlist
    nbdirect->calc_nonbond_force(whichlist, calc_energy, calc_virial, xyzq, force, clear_done_event);
  } else {
    // Using GPU created neighborlist
    nlist->waitBuildDone(whichlist, nbdirect->get_nonbond_stream(whichlist));
    nbdirect->calc_nonbond_force(whichlist, calc_energy, calc_virial, xyzq, force, clear_done_event,
				 nlist->getNeighborList());
  }
}

void MDsim::calc_nonbond_14_force(bool calc_energy, bool calc_virial) {
  assert(nbdirect != NULL);
  nbdirect->calc_nonbond_14_force(calc_energy, calc_virial, xyzq, force, clear_done_event);
}

void MDsim::setup_bonded_coef(const int nbondcoef, const float2 *h_bondcoef,
			      const int nureybcoef, const float2 *h_ureybcoef,
			      const int nanglecoef, const float2 *h_anglecoef,
			      const int ndihecoef, const float4 *h_dihecoef,
			      const int nimdihecoef, const float4 *h_imdihecoef,
			      const int ncmapcoef, const float2 *h_cmapcoef) {
  assert(bonded != NULL);
  bonded->setup_coef(nbondcoef, h_bondcoef,
		     nureybcoef, h_ureybcoef,
		     nanglecoef, h_anglecoef,
		     ndihecoef, h_dihecoef,
		     nimdihecoef, h_imdihecoef,
		     ncmapcoef, h_cmapcoef);
}

void MDsim::setup_bonded_list(const int nbondlist, const bondlist_t *h_bondlist, 
			      const int nureyblist, const bondlist_t *h_ureyblist,
			      const int nanglelist, const anglelist_t *h_anglelist,
			      const int ndihelist, const dihelist_t *h_dihelist,
			      const int nimdihelist, const dihelist_t *h_imdihelist,
			      const int ncmaplist, const cmaplist_t *h_cmaplist) {
  assert(bonded != NULL);
  bonded->setup_list(nbondlist, h_bondlist, 
		     nureyblist, h_ureyblist,
		     nanglelist, h_anglelist,
		     ndihelist, h_dihelist,
		     nimdihelist, h_imdihelist,
		     ncmaplist, h_cmaplist);
}

void MDsim::calc_bonded_force(const bool calc_energy, const bool calc_virial,
			      const bool calc_bond, const bool calc_ureyb,
			      const bool calc_angle, const bool calc_dihe,
			      const bool calc_imdihe, const bool calc_cmap) {
  assert(bonded != NULL);
  bonded->calc_bonded_force(this->boxx, this->boxy, this->boxz,
			    calc_energy, calc_virial,
			    calc_bond, calc_ureyb,
			    calc_angle, calc_dihe,
			    calc_imdihe, calc_cmap,
			    xyzq, force, clear_done_event);
}

void MDsim::calc_recip_force(const bool calc_energy, const bool calc_virial) {
  assert(nbrecip != NULL);
  if (recip_uses_nbdirect_buffers) {
    nbrecip->calc_recip_force(1.0/this->boxx, 1.0/this->boxy, 1.0/this->boxz,
                              calc_energy, calc_virial, clear_done_event,
			      &xyzq, &recipforce);
  } else {
    energyVirialRecip.clearEtermDevice(nbrecip->strEwks, mdsim_stream);
    energyVirialRecip.clearEtermDevice(nbrecip->strEwse, mdsim_stream);
    nbrecip->calc_recip_force(1.0/this->boxx, 1.0/this->boxy, 1.0/this->boxz,
                              calc_energy, calc_virial, clear_done_event);
  }
  cur_calc_energy = calc_energy;
  cur_calc_virial = calc_virial;
}

void MDsim::calc_recip_force_block(const bool calc_energy, const bool calc_virial) {
  assert(nbrecip != NULL);

  if (recip_uses_nbdirect_buffers) {
    nbrecip->calc_recip_force_block(1.0 / this->boxx, 1.0 / this->boxy, 1.0 / this->boxz,
                                    calc_energy, calc_virial,
                                    clear_done_event,
                                    cudaBlock->getBixlam(),
                                    cudaBlock->getBiflam(), cudaBlock->getBlockType(),
                                    &xyzq, &recipforce);
  } else {
    energyVirialRecip.clearEtermDevice(nbrecip->strEwks, mdsim_stream);
    energyVirialRecip.clearEtermDevice(nbrecip->strEwse, mdsim_stream);
    nbrecip->calc_recip_force_block(1.0 / this->boxx, 1.0 / this->boxy, 1.0 / this->boxz,
                                    calc_energy, calc_virial,
                                    clear_done_event,
                                    cudaBlock->getBixlam(),
                                    cudaBlock->getBiflam(), cudaBlock->getBlockType());
  }
  cur_calc_energy = calc_energy;
  cur_calc_virial = calc_virial;
}

void MDsim::get_direct_virial(double *vir) {
  assert(nbdirect != NULL);
  energyVirialDirect.getVirial(vir);
}

void MDsim::get_recip_virial(double *vir) {
  assert(nbrecip != NULL);
  energyVirialRecip.getVirial(vir);
}

void MDsim::get_nonbond_energy(double& energy_vdw, double& energy_elec,
			       double& energy_excl) {
  assert(nbdirect != NULL);
  nbdirect->get_energy(energy_vdw, energy_elec, energy_excl);
}

void MDsim::get_bonded_energy(double& energy_bond, double& energy_ureyb,
			      double& energy_angle,
			      double& energy_dihe, double& energy_imdihe,
			      double& energy_cmap) {
  assert(bonded != NULL);
  bonded->get_energy(energy_bond, energy_ureyb,
		     energy_angle,
		     energy_dihe, energy_imdihe,
		     energy_cmap);
}

void MDsim::get_recip_energy(double& energy_ksum, double& energy_self) {
  assert(nbrecip != NULL);
  nbrecip->get_energy(energy_ksum, energy_self);
}

void MDsim::generate_rn_gpu(int natom,float* gaussian_rn) {
  assert(cudaRandom != NULL);
  cudaRandom->Generate(natom,gaussian_rn);
}

#endif //NOCUDAC
