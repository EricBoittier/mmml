#ifndef NOCUDAC
#ifndef NLIST_H
#define NLIST_H
//
// Neighborlist storage class
// By: Antti-Pekka Hynninen, Feb 2015
//
#include <vector>
#include <cuda.h>
#include "../domdec_gpu/XYZQ.h"
#include "../domdec_gpu/CudaNeighborList.h"

class Nlist {
private:
  // Topological exclusions
  CudaTopExcl topExcl;

  // Neighbor lists
  CudaNeighborList<32> neighborList;

  // Un-sorted coordinates
  XYZQ xyzq_unsorted;
  
  // Local -> Global index mapping
  int* loc2glo;
  int loc2glo_len;

  // Order of zones
  // I,FZ,FY,EX,FX,EZ,EY,C = 0,...7
  enum {I=0,FZ=1,FY=2,EX=3,FX=4,EZ=5,EY=6,C=7};

  // Stream
  cudaStream_t stream;

  // Events
  cudaEvent_t sortDoneEvent[2];
  cudaEvent_t buildDoneEvent[2];
  
  void getImportIntZones(std::vector<int>& numIntZone, std::vector< std::vector<int> >& intZones,
			 const int nx, const int ny, const int nz);
  
public:
  Nlist(const int ncoord_glo, const int* iblo14, const int* inb14,
	const int nx, const int ny, const int nz);
  ~Nlist();
  void setTest(const bool test) {neighborList.set_test(test);}
  void sortCoord(const int whichlist, const int *zonelist_atom, float4* h_xyzq, int* h_loc2glo_ind, XYZQ& xyzq);
  void waitSortCoord(const int whichlist);
  void buildNeighborList(const int whichlist, const int* zonelist_atom,
			 const double rnl, const double boxx, const double boxy, const double boxz,
			 XYZQ& xyzq);
  void waitBuildDone(const int whichlist, cudaStream_t stream_wait);
  CudaNeighborList<32>* getNeighborList() {return &neighborList;}
};

#endif // NLIST_H
#endif
