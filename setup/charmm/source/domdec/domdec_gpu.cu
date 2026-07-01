#ifndef NOCUDAC

#include <iostream>
#include <cuda.h>
#include "../domdec_gpu/cuda_utils.h"
#include "../domdec_gpu/gpu_utils.h"
//#include "../domdec_gpu/NeighborList.h"
#include "MDsim.h"

// MD simulation structure
// NOTE: this structure holds all the necessary parameters & data for the simulation run
MDsim *mdsim = NULL;
int local_rank = -2;
int local_size = -2;

//
// Sets q_test -flag for the GPU data structure
//
extern "C" void set_q_test_gpu(int *h_q_test) {
  if (mdsim == NULL) {
    std::cout << "set_q_test_gpu: mdsim object does not exist" << std::endl;
    exit(1);
  }
  mdsim->setTest(*h_q_test != 0);
}

//
// Returns true if device is running
//
extern "C" void is_device_on(int *q_device_on) {
  *q_device_on = (get_gpu_ind() == -1) ? 0 : 1;
}

//
// Stops GPU device
//
extern "C" void stop_device() {

  // Reset device
  if (get_gpu_ind() != -1) {
    stop_gpu();
  }
}

//
// Returns the local rank set by environment variable
// Returns -1 if no local rank found
//
int get_env_local_rank() {
  char *localRankStr = NULL;
  int rank;

  // We extract the local rank initialization using an environment variable
  if ((localRankStr = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) != NULL) {
    // OpenMPI found
    rank = atoi(localRankStr);
  } else if ((localRankStr = getenv("MV2_COMM_WORLD_LOCAL_RANK")) != NULL) {
    // MVAPICH found
    rank = atoi(localRankStr);
  } else {
    rank = -1;
  }

  return rank;
}

//
// Returns the local size set by environment variable
// Returns -1 if no local size found
//
int get_env_local_size() {
  char *localRankStr = NULL;
  int size;

  // We extract the local rank initialization using an environment variable
  if ((localRankStr = getenv("OMPI_COMM_WORLD_LOCAL_SIZE")) != NULL) {
    // OpenMPI found
    size = atoi(localRankStr);
  } else if ((localRankStr = getenv("MV2_COMM_WORLD_LOCAL_SIZE")) != NULL) {
    // MVAPICH found
    size = atoi(localRankStr);
  } else {
    size = -1;
  }

  return size;
}

//
// Starts GPU device
// if *gpuid >= 0, it is used to override default settings
//
extern "C" void start_device(int *prnlev, int *mynod, int *numnod, int *gpuid) {
  if (get_gpu_ind() > -1)  // device already selected
    return;

  if (local_rank == -2)  // local rank not yet set
    local_rank = get_env_local_rank();

  *gpuid = start_gpu(*prnlev, local_rank, *mynod, *gpuid);
}

//
// Start MDsim
//
extern "C" void start_mdsim(int *ndirect, int *recip_on, int *bonded_on, int *nlist_on,
			    int *q_direct_node, int *q_recip_node,
			    int *nx, int *ny, int *nz, int *ncoord, int *iblo14, int *inb14,
			    int *nfftx, int *nffty, int *nfftz, int *forder, double *kappa, int *numBlock,
                            int *use_softcore, int *use_PMEL, int *seed) {
  if (mdsim == NULL) {
    mdsim = new MDsim(*ndirect, *q_direct_node!=0, *q_recip_node!=0 && *recip_on!=0,
		      *q_direct_node!=0 && *bonded_on!=0, *q_direct_node!=0 && *nlist_on!=0,
		      *nx, *ny, *nz, *ncoord, iblo14, inb14,
		      *nfftx, *nffty, *nfftz, *forder, *kappa, *numBlock,
                      *use_softcore, *use_PMEL, *seed);
  }
}

//
// Stops MDsim
//
extern "C" void stop_mdsim() {
  if (mdsim != NULL) {
    delete mdsim;
    mdsim = NULL;
  }
}

//
// Allocates pinned host memory for float4
//
extern "C" void alloc_gpu_pinned_float4(void **h_float4, int *h_n) {
  allocate_host<float4>((float4 **)h_float4, *h_n);
  //  cudaCheck(cudaHostAlloc(h_float4, *h_n*sizeof(float4), cudaHostAllocPortable));
}

//
// Deallocates pinned host memory for float4
//
extern "C" void dealloc_gpu_pinned_float4(void **h_float4) {
  deallocate_host<float4>((float4 **)h_float4);
  //  cudaCheck(cudaFreeHost(*h_float4));
}

//
// Allocates pinned host memory for int
//
extern "C" void alloc_gpu_pinned_int(void **h_int, int *h_n) {
  allocate_host<int>((int **)h_int, *h_n);
  //  cudaCheck(cudaHostAlloc(h_int, *h_n*sizeof(int), cudaHostAllocPortable));
}

//
// Deallocates pinned host memory for int
//
extern "C" void dealloc_gpu_pinned_int(void **h_int) {
  deallocate_host<int>((int **)h_int);
  //  cudaCheck(cudaFreeHost(*h_int));
}

//
// Allocates pinned host memory for double
//
extern "C" void alloc_gpu_pinned_double(void **h_double, int *h_n) {
  allocate_host<double>((double **)h_double, *h_n);
}

//
// Deallocates pinned host memory for double
//
extern "C" void dealloc_gpu_pinned_double(void **h_double) {
  deallocate_host<double>((double **)h_double);
}

//
// Allocates pinned host memory for float
//
extern "C" void alloc_gpu_pinned_float(void **h_float, int *h_n) {
  allocate_host<float>((float **)h_float, *h_n);
}

//
// Deallocates pinned host memory for float
//
extern "C" void dealloc_gpu_pinned_float(void **h_float) {
  deallocate_host<float>((float **)h_float);
}

//
// Allocates pinned host memory for ientry
//
extern "C" void alloc_gpu_pinned_ientry(void **h_ientry, int *h_n) {
  allocate_host<ientry_t>((ientry_t **)h_ientry, *h_n);
  //cudaCheck(cudaHostAlloc(h_ientry, *h_n*sizeof(ientry_t), cudaHostAllocPortable));
}

//
// Deallocates pinned host memory for ientry
//
extern "C" void dealloc_gpu_pinned_ientry(void **h_ientry) {
  deallocate_host<ientry_t>((ientry_t **)h_ientry);
  //cudaCheck(cudaFreeHost(*h_ientry));
}

//
// Allocates pinned host memory for tile_excl
//
extern "C" void alloc_gpu_pinned_tile_excl(void **h_tile_excl, int *h_n) {
  allocate_host< tile_excl_t<32> >((tile_excl_t<32> **)h_tile_excl, *h_n);
  //cudaCheck(cudaHostAlloc(h_tile_excl, *h_n*sizeof(tile_excl_t), cudaHostAllocPortable));
}

//
// Deallocates pinned host memory for tile_excl
//
extern "C" void dealloc_gpu_pinned_tile_excl(void **h_tile_excl) {
  deallocate_host<tile_excl_t<32> >((tile_excl_t<32> **)h_tile_excl);
  //cudaCheck(cudaFreeHost(*h_tile_excl));
}

//
// Allocates pinned host memory for char
//
extern "C" void alloc_gpu_pinned_char(void **h_char, int *h_n) {
  allocate_host<char>((char **)h_char, *h_n);
}

//
// Deallocates pinned host memory for char
//
extern "C" void dealloc_gpu_pinned_char(void **h_char) {
  deallocate_host<char>((char **)h_char);
}

#endif

