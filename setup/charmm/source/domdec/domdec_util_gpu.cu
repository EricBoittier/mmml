#ifndef NOCUDAC

#include <cuda.h>
#include "../domdec_gpu/cuda_utils.h"
#include "../domdec_gpu/gpu_utils.h"
#include "../domdec_gpu/CudaPMEDirectForce.h"
#include "../domdec_gpu/Force.h"
#include "domdec_gpu.h"

#if defined(CUDA_VERSION) && (CUDA_VERSION < 4020)
#error "CUDA version 4.2 or newer required."
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#error "CUDA compute capability 2.0 or higher required."
#endif

//     subroutine build_neighborlist_on_gpu(whichlist, zonelist_pcoord, cutnb, boxx, boxy, boxz) &
//----------------------------------------------------------------------------------------

//
// Builds neighborlist on GPU
//
extern "C" void build_neighborlist_on_gpu(int *whichlist, int *zonelist_atom, double* cutnb,
					  double* boxx, double* boxy, double* boxz) {
  mdsim->set_box_size(*boxx, *boxy, *boxz);
  mdsim->buildNeighborList(*whichlist-1, zonelist_atom, *cutnb);
}

//----------------------------------------------------------------------------------------

//
// Sorts coordinates on GPU
//
extern "C" void sort_xyzq_on_gpu(int *whichlist, int *zonelist_atom, float4* h_xyzq, int* h_loc2glo_ind) {
  mdsim->sortCoord(*whichlist-1, zonelist_atom, h_xyzq, h_loc2glo_ind);
}

//----------------------------------------------------------------------------------------

//
// Waits for coordinate sorting on GPU to finish
//
extern "C" void wait_sort_xyzq_on_gpu(int *whichlist) {
  mdsim->waitSortCoord(*whichlist-1);
}

//----------------------------------------------------------------------------------------

//
// Clears device force buffers
//
extern "C" void clear_force_virial_energy_gpu(int* q_calc_energy, int* q_calc_virial) {
  bool calc_energy = (*q_calc_energy != 0);
  bool calc_virial = (*q_calc_virial != 0);
  mdsim->clear_force_virial_energy(calc_energy, calc_virial);
}

//----------------------------------------------------------------------------------------

//
// Reduces device force buffers and virial (latter only if required)
//
extern "C" void reduce_force_virial_gpu() {

  // Reduce
  mdsim->reduce_force_virial();

}

//----------------------------------------------------------------------------------------

//
// Copies forces, virial, and energy to host
//
extern "C" void copy_force_virial_energy_from_gpu() {
  mdsim->get_force_virial_energy();
}
//----------------------------------------------------------------------------------------

//
// Waits for GPU forces to be ready for transfer
//
extern "C" void wait_force_virial_energy_from_gpu() {
  mdsim->cpu_wait_copy_done();
}

//----------------------------------------------------------------------------------------
//
// Returns pointer to host force array
//
extern "C" void get_force_pointer(void **h_force) {
  *h_force = mdsim->get_force_pointer();
}

//----------------------------------------------------------------------------------------
//
// Returns host force array stride
//
extern "C" void get_force_stride(int *stride) {
  *stride = mdsim->get_force_stride();
}

//----------------------------------------------------------------------------------------
//
// Returns host force type: 0=double, 1=float
//
extern "C" void get_force_type(int *ftype) {
  *ftype = mdsim->get_force_type();
}

//----------------------------------------------------------------------------------------

//
// Copies VdW parameter buffer (vdwparam) to GPU
// NOTE: This is done only when the system has changed
// NOTE: h_nvdwparam is the total size of the h_vdwparam
//       (i.e. h_nvdwparam = 2*number of vdw parameters)
//
extern "C" void copy_vdwparam_to_gpu(int *h_nvdwparam, float *h_vdwparam) {
  mdsim->set_vdwparam(*h_nvdwparam, h_vdwparam);
}

//----------------------------------------------------------------------------------------

//
// Copies 1-4 exclusion/inclusion VdW parameter buffer (vdwparam) to GPU
// NOTE: This is done only when the system has changed
// NOTE: h_nvdwparam is the total size of the h_vdwparam
//       (i.e. h_nvdwparam = 2*number of vdw parameters)
//
extern "C" void copy_vdwparam14_to_gpu(int *h_nvdwparam, float *h_vdwparam) {
  mdsim->set_vdwparam14(*h_nvdwparam, h_vdwparam);
}

//----------------------------------------------------------------------------------------

//
// Copies 1-4 exclusion/inclusion lists to GPU
//
extern "C" void copy_14_list_to_gpu(int *nin14list, int *nex14list, xx14list_t* h_in14list, xx14list_t* h_ex14list) {
  mdsim->set_14_list(*nin14list, *nex14list, h_in14list, h_ex14list);
}

//----------------------------------------------------------------------------------------

//
// Calculates 1-4 exclusion/inclusion forces on GPU
//
extern "C" void calc_14_force_gpu(int *q_calc_energy, int *q_calc_virial,
				  double *boxx, double *boxy, double *boxz) {
  bool calc_energy = (*q_calc_energy != 0);
  bool calc_virial = (*q_calc_virial != 0);
  mdsim->set_box_size(*boxx, *boxy, *boxz);
  mdsim->calc_nonbond_14_force(calc_energy, calc_virial);
}

//----------------------------------------------------------------------------------------

//
// Copies (x, y, z, q) to GPU
// NOTE: Only (x, y, z, q) are copied, (vdwtype) is left unchanged
//
extern "C" void copy_home_xyzq_to_gpu(float4 *h_xyzq) {
  mdsim->set_home_xyzq(h_xyzq);
}

//
// Copies (x, y, z, q) to GPU
// NOTE: Only (x, y, z, q) are copied, (vdwtype) is left unchanged
//
extern "C" void copy_import_xyzq_to_gpu(float4 *h_xyzq) {
    mdsim->set_import_xyzq(h_xyzq);
}

//
// Copies (x, y, z, q) to GPU
// NOTE: Only (x, y, z, q) are copied, (vdwtype) is left unchanged
//
extern "C" void copy_xyzq_to_gpu(int *ncoord, double *h_x, double *h_y, double *h_z, double *h_q) {
  mdsim->set_recip_xyzq(*ncoord, h_x, h_y, h_z, h_q);
}

//----------------------------------------------------------------------------------------

//
// Sets ncoord - allocate & re-allocates all related arrays as needed:
// d_xyzq, d_vdwtype, d_force, h_force
//
extern "C" void set_direct_ncoord_gpu(int *h_ncoord_home, int *h_ncoord_import) {
  mdsim->set_ncoord(*h_ncoord_home, *h_ncoord_import);
}

//----------------------------------------------------------------------------------------

//
// Copies vdwtype to GPU
//
extern "C" void copy_vdwtype_to_gpu(int *h_vdwtype) {
  mdsim->set_vdwtype(h_vdwtype);
}

//----------------------------------------------------------------------------------------

extern "C" void range_start(char *range_name) {
  gpu_range_start(range_name);
}

extern "C" void range_stop(char *range_name) {
  gpu_range_stop();
}

//----------------------------------------------------------------------------------------
//
// Copies blocktype (iblckp) to GPU
//
extern "C" void copy_blocktype_to_gpu(int *ncoord, int *h_blocktype) {
  mdsim->setBlockType(*ncoord, h_blocktype);
}

//----------------------------------------------------------------------------------------
//
// Copies blockparam (fullblcoep) to GPU
//
extern "C" void copy_blockparam_to_gpu(float *h_blockparam) {
  mdsim->setBlockParam(h_blockparam);
}

//----------------------------------------------------------------------------------------
//
// Copies bixlam to GPU
//
extern "C" void copy_bixlam_to_gpu(float *h_bixlam) {
  mdsim->setBixlam(h_bixlam);
}

//----------------------------------------------------------------------------------------
//
// Copies biflam and biflam2 on GPU
//
extern "C" void copy_biflam_to_gpu(double *h_biflam, double *h_biflam2) {
  mdsim->setBiflam(h_biflam, h_biflam2);
}

//----------------------------------------------------------------------------------------
//
// Copies isitemld to GPU
//
extern "C" void copy_isitemld_to_gpu(int *h_isitemld) {
  mdsim->setSiteMLD(h_isitemld);
}

//----------------------------------------------------------------------------------------
//
// Reads biflam and biflam2 from GPU
//
extern "C" void read_biflam_from_gpu(double *h_biflam, double *h_biflam2) {
  mdsim->getBiflam(h_biflam, h_biflam2);
}

//----------------------------------------------------------------------------------------
//
// Copies 1-4 block exclusion/inclusion lists to GPU
//
extern "C" void copy_14_block_pos_to_gpu(int *in14tbl_block_pos, int *ex14tbl_block_pos) {
  mdsim->set_14_block_pos(in14tbl_block_pos, ex14tbl_block_pos);
}

//----------------------------------------------------------------------------------------
//
// Generate 3 gaussian random numbers for natom atoms on the GPU
//
extern "C" void generate_rn_gpu(int *natom,float *gaussian_rn) {
  mdsim->generate_rn_gpu(*natom,gaussian_rn);
}

//----------------------------------------------------------------------------------------

#endif

