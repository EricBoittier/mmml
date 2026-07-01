#if KEY_DOMDEC_GPU == 1
#ifndef CUDARANDOM_H
#define CUDARANDOM_H

#include <curand.h>

// Limit specified in curand documentation
#define MAXRNGBLOCK 200

class CudaRandom {

private:
  // curand library data structures
  // curandStateMtgp32 *d_rng_state;
  // mtgp32_kernel_params *d_rng_params;

  curandGenerator_t gen;
  
  // Amount of storage space
  int n_gaussian_rn;

  // Storage space on host and device
  // float *h_gaussian_rn;
  float *d_gaussian_rn;

public:
  // Creator
  CudaRandom(int seed);

  // Destructor
  ~CudaRandom();

  // Generate Gaussian random numbers for natom atoms. Pointer to float is returned
  void Generate(int natom,float* gaussian_rn);
};

#endif // CUDARANDOM_H
#endif // KEY_DOMDEC_GPU
