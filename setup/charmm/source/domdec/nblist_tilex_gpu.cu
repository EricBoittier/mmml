#ifndef NOCUDAC

#include <cuda.h>
#include "../domdec_gpu/cuda_utils.h"
#include "domdec_gpu.h"

#if defined(CUDA_VERSION) && (CUDA_VERSION < 4020)
#error "CUDA version 4.2 or newer required."
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#error "CUDA compute capability 2.0 or higher required."
#endif

//----------------------------------------------------------------------------------------
//
// Wrappers for C++ class methods
//
//----------------------------------------------------------------------------------------

//
// Copies cell_start to GPU
//
extern "C" void copy_cell_start_to_gpuB(int *h_ncell, int *h_cell_start) {
  mdsim->set_cell_start(*h_ncell, h_cell_start);
}

//
// Copies jlist to GPU
// NOTE: h_ijlist is size (3,h_n_ijlist)
//
extern "C" void copy_ijlist_to_gpu(int *h_whichlist, int *h_n_ijlist, int *h_ijlist) {
  mdsim->set_ijlist(*h_whichlist-1, *h_n_ijlist, h_ijlist);
}

//
// Copies ientry to GPU
//
extern "C" void copy_ientry_to_gpu(int *h_whichlist, int *h_n_ientry, int *h_ientry) {
  mdsim->set_ientry(*h_whichlist-1, *h_n_ientry, (ientry_t *)h_ientry);
}

//
// Host wrapper for build_tilex_kernel
// Builds exclusion mask based on atom-atom distance and index (i >= j excluded)
//
extern "C" void build_excl_dist_index(int *h_whichlist, double *h_boxx, double *h_boxy, double *h_boxz,
				      double *h_roff) {
  mdsim->set_box_size(*h_boxx, *h_boxy, *h_boxz);
  mdsim->build_excl(*h_whichlist-1, (float)*h_roff);
}

//
// Copies and combines topological exclusions (tile_ind_top, tile_excl_top) to GPU
//
extern "C" void combine_tile_top_to_gpu(int *h_whichlist, int *h_ntile_top, int *h_tile_ind_top,
					tile_excl_t<32> *h_tile_excl_top) {
  mdsim->combine_tile_top(*h_whichlist-1, *h_ntile_top, h_tile_ind_top, h_tile_excl_top);
}

#endif

