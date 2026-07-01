#ifndef NOCUDAC
#ifndef CUDAENERGYVIRIAL_H
#define CUDAENERGYVIRIAL_H
//
// Storage class for direct-space energies and virials in CUDA
// (c) Antti-Pekka Hynninen, February 2015
// aphynninen@hotmail.com
//
#include "EnergyVirial.h"
#include <cuda.h>

// Structure into which virials are stored
// NOTE: This structure is only used for computing addresses
struct Virial_t {
  double sforce_dp[27][3];
  long long int sforce_fp[27][3];
  double virmat[9];
  // Energies start here ...
};

class CudaEnergyVirial : public EnergyVirial {
private:
  
  // Host and device arrays for storing energies and sforce -arrays
  int h_buffer_len;
  char *h_buffer;

  int d_buffer_len;
  char *d_buffer;

  void reallocateBuffer();
  
public:
  CudaEnergyVirial();
  ~CudaEnergyVirial();


  void clear(cudaStream_t stream=0);
  void clearEnergy(cudaStream_t stream=0);
  void clearEtermDevice(std::string& nameEterm, cudaStream_t stream=0);
  void clearVirial(cudaStream_t stream=0);
  void calcVirial(const int ncoord, const float4 *xyzq,
		  const double boxx, const double boxy, const double boxz,
		  const int stride, const double *force,
		  cudaStream_t stream=0);
  void copyToHost(cudaStream_t stream=0);

  Virial_t* getVirialPointer();
  double* getEnergyPointer(std::string& name);

  double getEnergy(std::string& name);
  double getEnergy(const char* name);
  void getVirial(double *virmat);
  void getSforce(double *sforce);
};

#endif //CUDAENERGYVIRIAL_H
#endif //NOCUDAC
