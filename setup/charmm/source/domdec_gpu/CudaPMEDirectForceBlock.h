#ifndef NOCUDAC
#ifndef CUDAPMEDIRECTFORCEBLOCK_H
#define CUDAPMEDIRECTFORCEBLOCK_H
#include <cuda.h>
#include "CudaPMEDirectForce.h"
#include "CudaBlock.h"

//
// Calculates direct non-bonded interactions on GPU using BLOCK
//
// (c) Antti-Pekka Hynninen, 2014, aphynninen@hotmail.com
//
// AT = accumulation type
// CT = calculation type
//

template <typename AT, typename CT>
  class CudaPMEDirectForceBlock : public CudaPMEDirectForce<AT,CT> {

 private:

  CudaBlock &cudaBlock;
  
  // Local (fixed precision versions) force coefficients (size numBlock each)
  int biflamLen;
  AT *biflam[2];
  int biflam2Len;
  AT *biflam2[2];

  // Arrays (size nblock*(nblock+1)/2) for storing temporary results in calc_14_force
  // This is the main buffer that contains the actual allocated pointer
  double *energy14BlockBuffer;
  // These are pointers into the main buffer
  double *energyVdw14Block;
  double *energyElec14Block;
  double *energyExcl14Block;

  // Array that contains lower triangle matrix i and j indices (size nblock*(nblock+1)/2)
  // lowTriangleIJ = (jb << 16) | ib
  int *lowTriangleIJ;

  // Block table positions for 1-4 calculations (size nblock*(nblock+1)/2+1)
  int *h_in14TblBlockPos;
  int *h_ex14TblBlockPos;
  int *d_in14TblBlockPos;
  int *d_ex14TblBlockPos;
  
  int nin14BlockToBlock;
  int nex14BlockToBlock;
  int *h_in14BlockToBlock;
  int *h_ex14BlockToBlock;
  int *d_in14BlockToBlock;
  int *d_ex14BlockToBlock;
  
  int nin14BlockToTblPos;
  int nex14BlockToTblPos;
  int *h_in14BlockToTblPos;
  int *h_ex14BlockToTblPos;
  int *d_in14BlockToTblPos;
  int *d_ex14BlockToTblPos;
  
 public:

  CudaPMEDirectForceBlock(CudaEnergyVirial &energyVirial,
			  const char *nameVdw, const char *nameElec, const char *nameExcl,
			  CudaBlock &cudaBlock);
  ~CudaPMEDirectForceBlock();

  void set14BlockPos(int *h_in14TblBlockPos_in, int *h_ex14TblBlockPos_in);
  
  void calc_14_force(const float4 *xyzq,
  		     const bool calc_energy, const bool calc_virial,
  		     const int stride, AT *force, cudaStream_t stream=0);

  void calc_force(const int whichlist,
                  const float4 *xyzq,
		  const CudaNeighborListBuild<32>& nlist,
		  const bool calc_energy,
		  const bool calc_virial,
		  const int stride, AT *force,
		  cudaStream_t stream=0);

};

#endif // CUDAPMEDIRECTFORCEBLOCK_H
#endif //NOCUDAC
