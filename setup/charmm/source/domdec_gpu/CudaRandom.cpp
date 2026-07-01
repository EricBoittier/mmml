#if KEY_DOMDEC_GPU == 1

#include "CudaRandom.h"
#include <limits.h>
#include "cuda_utils.h"
#include <cstdio>
#include <cstdlib>

static const char *curandGetErrorString(curandStatus_t error)
{
  switch (error)
    {
    case CURAND_STATUS_SUCCESS:
      return "CURAND_STATUS_SUCCESS";

    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";

    case CURAND_STATUS_NOT_INITIALIZED:
      return "CURAND_STATUS_NOT_INITIALIZED";

    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";

    case CURAND_STATUS_TYPE_ERROR:
      return "CURAND_STATUS_TYPE_ERROR";

    case CURAND_STATUS_OUT_OF_RANGE:
      return "CURAND_STATUS_OUT_OF_RANGE";

    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

    case CURAND_STATUS_LAUNCH_FAILURE:
      return "CURAND_STATUS_LAUNCH_FAILURE";

    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";

    case CURAND_STATUS_ARCH_MISMATCH:
      return "CURAND_STATUS_ARCH_MISMATCH";

    case CURAND_STATUS_INTERNAL_ERROR:
      return "CURAND_STATUS_INTERNAL_ERROR";
    }

  return "<unknown>";
}

#define curandCheck(stmt) do {                                           \
        curandStatus_t err = stmt;                                       \
        if (err != CURAND_STATUS_SUCCESS) {                              \
	  fprintf(stderr, "Error running %s in file %s, function %s, line %d\n", \
                 #stmt, __FILE__, __FUNCTION__, __LINE__);               \
	  fprintf(stderr, "Error string: %s\n", curandGetErrorString(err)); \
	  exit(1);						         \
        }                                                                \
    } while(0)

CudaRandom::CudaRandom(int seed) {
  // Initialize space for storing random numbers
  n_gaussian_rn=0;
  d_gaussian_rn=NULL;

  /* Create pseudo-random number generator */
  curandCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32));
  
  /* Set seed */
  curandCheck(curandSetPseudoRandomGeneratorSeed(gen, seed));
}

CudaRandom::~CudaRandom() {

  // if (h_gaussian_rn != NULL) deallocate_host< float > (&h_gaussian_rn);
  // if (d_gaussian_rn != NULL) deallocate< float > (&d_gaussian_rn);
  // if (d_rng_state != NULL) deallocate< curandStateMtgp32 > (&d_rng_state);
  // if (d_rng_params != NULL) deallocate< mtgp32_kernel_params > (&d_rng_params);

    if (d_gaussian_rn != NULL) deallocate< float > (&d_gaussian_rn);
    
    curandCheck(curandDestroyGenerator(gen));
}

void CudaRandom::Generate(int natom,float* gaussian_rn) {
  int blocks;
  int nrandn;

  nrandn=3*natom;
  nrandn+=(nrandn & 1);
  blocks=(nrandn+256-1)/256;
  blocks=(blocks>MAXRNGBLOCK ? MAXRNGBLOCK : blocks);

  // Realloc data if necessary
  // reallocate_host< float > (&h_gaussian_rn, &n_gaussian_rn, nrandn, 1.4f);
  reallocate< float > (&d_gaussian_rn, &n_gaussian_rn, nrandn, 1.4f);
  n_gaussian_rn = nrandn;
  
  // Generate random numbers
  // generateKernel<<<blocks, 256>>>(d_rng_state, nrandn, d_gaussian_rn);
  curandCheck(curandGenerateNormal(gen,
                                   d_gaussian_rn, n_gaussian_rn,
                                   0.0f, 1.0f));
  
  // Copy random numbers back to host
  cudaCheck(cudaMemcpy(gaussian_rn, d_gaussian_rn, 3*natom * sizeof(float),
                       cudaMemcpyDeviceToHost));
}

#endif // KEY_DOMDEC_GPU
