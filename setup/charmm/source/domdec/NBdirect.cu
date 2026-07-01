#ifndef NOCUDAC
#include <iostream>
#include <cassert>
#include <cuda.h>
#include "NBdirect.h"
#include "../domdec_gpu/cuda_utils.h"
#include "../domdec_gpu/gpu_utils.h"

//
// Class creator
//
NBdirect::NBdirect(CudaEnergyVirial& energyVirial, CudaBlock *cudaBlock) :
  strVdw("vdw"), strElec("elec"), strEwex("ewex"), energyVirial(energyVirial) {
  
  // Energy terms
  energyVirial.insert(strVdw);
  energyVirial.insert(strElec);
  energyVirial.insert(strEwex);

  ncell = 0;
  cell_start_len = 0;
  cell_start = NULL;
  
  for (int i=0;i < 2;i++) {
    n_ijlist[i] = 0;
    ijlist_len[i] = 0;
    ijlist[i] = NULL;
  
    tile_indj_len[i] = 0;
    tile_indj[i] = NULL;

    ntile_top[i] = 0;
    tile_ind_top_len[i] = 0;
    tile_ind_top[i] = NULL;
    
    tile_excl_top_len[i] = 0;
    tile_excl_top[i] = NULL;
  }

  nlist[0] = new CudaNeighborListBuild<32>(0, 0, 0);
  nlist[1] = new CudaNeighborListBuild<32>(0, 0, 0);

  directforceBlock = NULL;
  
  if (cudaBlock == NULL) {
    directforce = new CudaPMEDirectForce<long long int, float>(energyVirial, strVdw.c_str(), strElec.c_str(), strEwex.c_str());
  } else {
    directforceBlock = new CudaPMEDirectForceBlock<long long int, float>(energyVirial, strVdw.c_str(), strElec.c_str(), strEwex.c_str(), *cudaBlock);
    directforce = directforceBlock;
  }
  
  //prev_calc_energy = false;
  //prev_calc_virial = false;

  // Start streams
  cudaCheck(cudaStreamCreate(&nonbond_stream[0]));
  cudaCheck(cudaStreamCreate(&nonbond_stream[1]));
  cudaCheck(cudaStreamCreate(&nonbond_14_stream));

  // Create events
  cudaCheck(cudaEventCreate(&nonbond_calc_done_event[0]));
  cudaCheck(cudaEventCreate(&nonbond_calc_done_event[1]));
  cudaCheck(cudaEventCreate(&nonbond_14_calc_done_event));
}

//
// Class destructor
//
NBdirect::~NBdirect() {
  if (cell_start != NULL) deallocate<int>(&cell_start);

  for (int i=0;i < 2;i++) {
    if (ijlist[i] != NULL) deallocate<int3>(&ijlist[i]);
    if (tile_indj[i] != NULL) deallocate<int>(&tile_indj[i]);
    if (tile_ind_top[i] != NULL) deallocate<int>(&tile_ind_top[i]);
    if (tile_excl_top[i] != NULL) deallocate< tile_excl_t<32> >(&tile_excl_top[i]);
  }

  delete nlist[0];
  delete nlist[1];

  delete directforce;
  
  // Destroy events
  cudaCheck(cudaEventDestroy(nonbond_calc_done_event[0]));
  cudaCheck(cudaEventDestroy(nonbond_calc_done_event[1]));
  cudaCheck(cudaEventDestroy(nonbond_14_calc_done_event));

  // Stop streams
  cudaCheck(cudaStreamDestroy(nonbond_stream[0]));
  cudaCheck(cudaStreamDestroy(nonbond_stream[1]));
  cudaCheck(cudaStreamDestroy(nonbond_14_stream));
}

//
// Calculates direct-space virial on GPU
//
void NBdirect::calc_virial(const int ncoord, const double boxx, const double boxy, const double boxz,
			   XYZQ& xyzq, Force<long long int>& force, cudaStream_t stream) {
  energyVirial.calcVirial(ncoord, xyzq.xyzq, boxx, boxy, boxz,
			  force.stride(), (double *)force.xyz(), stream);
}

//
// Copies vdwtype from CPU to GPU
//
void NBdirect::set_vdwtype(const int ncoord, int *h_vdwtype) {
  directforce->set_vdwtype(ncoord, h_vdwtype);
}

//
// Copies vdwparam from CPU to GPU
//
void NBdirect::set_vdwparam(int h_nvdwparam, float *h_vdwparam) {
  directforce->set_vdwparam(h_nvdwparam, h_vdwparam);
}

//
// Copies 1-4 exclusion/inclusion vdwparam from CPU to GPU
//
void NBdirect::set_vdwparam14(int h_nvdwparam, float *h_vdwparam) {
  directforce->set_vdwparam14(h_nvdwparam, h_vdwparam);
}

//
// Copies 1-4 exclusion/inclusion lists from CPU to GPU
//
void NBdirect::set_14_list(int nin14list, int nex14list, xx14list_t* h_in14list, xx14list_t* h_ex14list) {
  directforce->set_14_list(nin14list, nex14list, h_in14list, h_ex14list, nonbond_14_stream);
}

//
// Sets 1-4 block positions
//
void NBdirect::set_14_block_pos(int numBlock, int* in14tbl_block_pos, int* ex14tbl_block_pos) {
  assert(directforceBlock != NULL);
  int m = numBlock*(numBlock+1)/2;
  for (int k=0;k < m+1;k++) {
    in14tbl_block_pos[k]--;
    ex14tbl_block_pos[k]--;
  }
  directforceBlock->set14BlockPos(in14tbl_block_pos, ex14tbl_block_pos);
  for (int k=0;k < m+1;k++) {
    in14tbl_block_pos[k]++;
    ex14tbl_block_pos[k]++;
  }
}

//
// Copies cell_start to GPU
//
void NBdirect::set_cell_start(int ncell, int *h_cell_start) {

  this->ncell = ncell;

  // Allocate & reallocate cell_start as needed
  reallocate<int>(&cell_start, &cell_start_len, ncell, 1.5f);

  // Copy h_cell_start -> cell_start
  copy_HtoD<int>(h_cell_start, cell_start, ncell, nonbond_stream[0]);

}

//
// Copies jlist from CPU to GPU
// NOTE: h_ijlist is size (3,h_n_ijlist)
//
void NBdirect::set_ijlist(int h_whichlist, int h_n_ijlist, int *h_ijlist) {
  n_ijlist[h_whichlist] = h_n_ijlist;
  reallocate<int3>(&ijlist[h_whichlist], &ijlist_len[h_whichlist], h_n_ijlist, 1.5f);
  copy_HtoD<int3>((int3 *)h_ijlist, ijlist[h_whichlist], h_n_ijlist, nonbond_stream[h_whichlist]);
}

//
// Copies ientry from CPU to GPU
//
void NBdirect::set_ientry(int h_whichlist, int h_n_ientry, ientry_t* h_ientry) {
  nlist[h_whichlist]->set_ientry(h_n_ientry, h_ientry, nonbond_stream[h_whichlist]);
}

//
// Builds distance exclusion on GPU
//
void NBdirect::build_excl(int h_whichlist, float boxx, float boxy, float boxz, float roff, XYZQ& xyzq) {
  nlist[h_whichlist]->build_excl(boxx, boxy, boxz, roff,
				 n_ijlist[h_whichlist], ijlist[h_whichlist],
				 cell_start, xyzq.xyzq,
				 nonbond_stream[h_whichlist]);
}

//
// Combines topological exclusions from CPU with the distance exclusions from GPU.
// Work is done on the GPU
//
void NBdirect::combine_tile_top(int h_whichlist, int h_ntile_top, int *h_tile_ind_top, 
				tile_excl_t<32> *h_tile_excl_top) {
  ntile_top[h_whichlist] = h_ntile_top;
  // Allocate & re-allocate (tile_ind_top, tile_excl_top)
  reallocate<int>(&tile_ind_top[h_whichlist], &tile_ind_top_len[h_whichlist], h_ntile_top, 1.2f);
  reallocate< tile_excl_t<32> >(&tile_excl_top[h_whichlist], &tile_excl_top_len[h_whichlist],
				h_ntile_top, 1.2f);
  copy_HtoD<int>(h_tile_ind_top, tile_ind_top[h_whichlist], h_ntile_top,
		 nonbond_stream[h_whichlist]);
  copy_HtoD< tile_excl_t<32> >(h_tile_excl_top, tile_excl_top[h_whichlist], h_ntile_top,
			       nonbond_stream[h_whichlist]);
  nlist[h_whichlist]->add_tile_top(h_ntile_top, tile_ind_top[h_whichlist],
				   tile_excl_top[h_whichlist], nonbond_stream[h_whichlist]);
}

//
// Calculate nonbonded force
//
void NBdirect::calc_nonbond_force(int whichlist, bool calc_energy, bool calc_virial, XYZQ& xyzq,
				  Force<long long int>& force, cudaEvent_t clear_done_event,
				  CudaNeighborList<32>* neighborList) {

  assert(whichlist == 0 || whichlist == 1);

  // First wait for forces zeroing to be done
  cudaCheck(cudaStreamWaitEvent(nonbond_stream[whichlist], clear_done_event, 0));

  if (neighborList == NULL) {
    // Using CPU created neighborlist
    directforce->calc_force(whichlist, xyzq.xyzq, *nlist[whichlist], calc_energy, calc_virial,
			    force.stride(), force.xyz(), nonbond_stream[whichlist]);
  } else if (whichlist < neighborList->getNumList()) {
    // Using GPU created neighborlist
    directforce->calc_force(whichlist, xyzq.xyzq, neighborList->getBuilder(whichlist), calc_energy, calc_virial,
			    force.stride(), force.xyz(), nonbond_stream[whichlist]);
  }

  cudaCheck(cudaEventRecord(nonbond_calc_done_event[whichlist], nonbond_stream[whichlist]));
}

//
// Calculate nonbonded 1-4 exclusions and inclusions force
//
void NBdirect::calc_nonbond_14_force(bool calc_energy, bool calc_virial, XYZQ& xyzq, 
				     Force<long long int>& force, cudaEvent_t clear_done_event) {
  // First wait for forces zeroing to be done
  cudaCheck(cudaStreamWaitEvent(nonbond_14_stream, clear_done_event, 0));

  directforce->calc_14_force(xyzq.xyzq, calc_energy, calc_virial,
			     force.stride(), force.xyz(), nonbond_14_stream);
  
  cudaCheck(cudaEventRecord(nonbond_14_calc_done_event, nonbond_14_stream));
}

//
// GPU stream "stream" waits until calc_nonbond_force is done
//
void NBdirect::gpu_wait_force_done(cudaStream_t wait_stream) {
  cudaCheck(cudaStreamWaitEvent(wait_stream, nonbond_calc_done_event[0], 0));
  cudaCheck(cudaStreamWaitEvent(wait_stream, nonbond_calc_done_event[1], 0));
  cudaCheck(cudaStreamWaitEvent(wait_stream, nonbond_14_calc_done_event, 0));
}

//
// Get energies
//
void NBdirect::get_energy(double& energy_vdw, double& energy_elec, double& energy_excl) {
  energy_vdw  = energyVirial.getEnergy(strVdw);
  energy_elec = energyVirial.getEnergy(strElec);
  energy_excl = energyVirial.getEnergy(strEwex);
}

void NBdirect::set_nonbond_param(float boxx, float boxy, float boxz, float kappa,
				 float roff, float ron, float e14fac,
				 int vdwmodel, int elecmodel, bool qeterm_vdw, bool qeterm_elec) {
  directforce->setup(boxx, boxy, boxz, kappa, roff, ron, e14fac, vdwmodel, elecmodel);
  directforce->set_calc_vdw(qeterm_vdw);
  directforce->set_calc_elec(qeterm_elec);
}

void NBdirect::set_box_size(double boxx, double boxy, double boxz) {
  directforce->set_box_size(boxx, boxy, boxz);
}

cudaStream_t NBdirect::get_nonbond_stream(const int i) {
  assert(i == 0 || i == 1);
  return nonbond_stream[i];
}

#endif //NOCUDAC
