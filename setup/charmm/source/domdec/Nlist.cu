#ifndef NOCUDAC
#include <cassert>
#include "../domdec_gpu/cuda_utils.h"
#include "Nlist.h"

//
// Class creator
//
Nlist::Nlist(const int ncoord_glo, const int* iblo14, const int* inb14,
	     const int nx, const int ny, const int nz) :
  topExcl(ncoord_glo, iblo14, inb14), neighborList(topExcl, nx, ny, nz) {

  loc2glo = NULL;
  loc2glo_len = 0;

  // Neighborlists
  std::vector<int> numIntZone(8, 0);
  std::vector< std::vector<int> > intZones(8, std::vector<int>() );
  // Create I vs. I interaction
  numIntZone.at(0) = 1;
  intZones.at(0).push_back(I);
  neighborList.registerList(numIntZone, intZones);
  if (nx*ny*nz > 1) {
    // NOTE: getImportIntZones will clear contents of numIntZone & intZones
    this->getImportIntZones(numIntZone, intZones, nx, ny, nz);
    neighborList.registerList(numIntZone, intZones);
  }

  // Create streams
  cudaCheck(cudaStreamCreate(&stream));
  // Create events
  cudaCheck(cudaEventCreate(&sortDoneEvent[0]));
  cudaCheck(cudaEventCreate(&sortDoneEvent[1]));
  cudaCheck(cudaEventCreate(&buildDoneEvent[0]));
  cudaCheck(cudaEventCreate(&buildDoneEvent[1]));
}

//
// Class destructor
//
Nlist::~Nlist() {
  // Destroy streams
  cudaCheck(cudaStreamDestroy(stream));
  // Destroy events
  cudaCheck(cudaEventDestroy(sortDoneEvent[0]));
  cudaCheck(cudaEventDestroy(sortDoneEvent[1]));
  cudaCheck(cudaEventDestroy(buildDoneEvent[0]));
  cudaCheck(cudaEventDestroy(buildDoneEvent[1]));

  // Free memory to avoid leak
  deallocate<int>(&loc2glo);
}

//
// Build a list of interacting zones
//
void Nlist::getImportIntZones(std::vector<int>& numIntZone, std::vector< std::vector<int> >& intZones,
			      const int nx, const int ny, const int nz) {
  // Create a list of active zones
  bool activeZones[8];
  for (int izone=0;izone < 8;izone++) activeZones[izone] = false;
  activeZones[I] = true;
  if (nx > 1) activeZones[FX] = true;
  if (ny > 1) activeZones[FY] = true;
  if (nz > 1) activeZones[FZ] = true;
  if (ny > 1 && nz > 1) activeZones[EX] = true;
  if (nx > 1 && nz > 1) activeZones[EY] = true;
  if (nx > 1 && ny > 1) activeZones[EZ] = true;
  if (nx > 1 && ny > 1 && nz > 1) activeZones[C] = true;
  //
  const int zones[8][5] = { {I, -1, -1, -1, -1},  // I-I
			    {I, -1, -1, -1, -1},  // FZ-I
			    {I, FZ, -1, -1, -1},  // FY-I, FY-FZ
			    {I, -1, -1, -1, -1},  // EX-I
			    {I, FZ, FY, EX, -1},  // FX-I, FX-FZ, FX-FY, FX-EX
			    {I, FZ, -1, -1, -1},  // EZ-I, EZ-FZ
			    {I, FY, -1, -1, -1},  // EY-I, EY-FY
			    {I, -1, -1, -1, -1}}; // C-I
  // NOTE: we are skipping I vs. I interaction
  numIntZone.at(0) = 0;
  intZones.at(0).clear();
  for (int izone=1;izone < 8;izone++) {
    numIntZone.at(izone) = 0;
    intZones.at(izone).clear();
    if (activeZones[izone]) {
      int j = 0;
      while (zones[izone][j] > -1) {
	int jzone = zones[izone][j];
	if (activeZones[jzone]) {
	  intZones.at(izone).push_back(jzone);
	  numIntZone.at(izone)++;
	}
	j++;
      }
    }
  }
}

//
// Sort coordinates in preparation for neighborlist build
//
void Nlist::sortCoord(const int whichlist, const int *zonelist_atom, float4* h_xyzq, int* h_loc2glo_ind, XYZQ& xyzq) {
  // Sanity checks
  assert(whichlist == 0 || whichlist == 1);
  if (whichlist >= neighborList.getNumList()) return;

  int zone_patom[9];
  zone_patom[0] = 0;
  for (int i=1;i < 9;i++) zone_patom[i] = zonelist_atom[i-1];
  
  int start = (whichlist == 0) ? zone_patom[0] : zone_patom[1];
  int end = (whichlist == 0) ? zone_patom[1] : zone_patom[8];

  // Re-allocate memory as required
  // In case re-alloc is needed for the second sort, we need to synchronize first to make sure the first sort
  // has finished
  if (whichlist == 1 && (end > loc2glo_len || end > xyzq_unsorted.xyzq_len || end > xyzq.xyzq_len))
    cudaCheck(cudaEventSynchronize(sortDoneEvent[0]));
  if (whichlist == 0) {
    // First sort, non-content preserving realloc is OK
    reallocate<int>(&loc2glo, &loc2glo_len, end, 1.5f);
    xyzq_unsorted.realloc(end, 1.5f);
    xyzq.realloc(end, 1.5f);
  } else {
    // Second sort, need to preserve content, use resize
    resize<int>(&loc2glo, &loc2glo_len, start, end, 1.5f);
    xyzq_unsorted.resize(end, 1.5f);
    xyzq.resize(end, 1.5f);
  }

  // Copy coordinates to GPU
  xyzq_unsorted.set_xyzq(end-start, h_xyzq, start, stream);
  cudaCheck(cudaDeviceSynchronize());
  // Copy loc2glo to GPU
  copy_HtoD<int>(&h_loc2glo_ind[start], &loc2glo[start], end-start, stream);
  // Sort on GPU
  neighborList.sort(whichlist, zone_patom, xyzq_unsorted.xyzq, xyzq.xyzq, loc2glo, stream);
  // Copy sorted coordinates to CPU
  copy_DtoH<float4>(&xyzq.xyzq[start], &h_xyzq[start], end-start, stream);
  // Copy sorted loc2glo to CPU
  copy_DtoH<int>(&loc2glo[start], &h_loc2glo_ind[start], end-start, stream);
  cudaCheck(cudaEventRecord(sortDoneEvent[whichlist], stream));
}

//
// Wait for coordinate sorting to finish
//
void Nlist::waitSortCoord(const int whichlist) {
  // Sanity checks
  assert(whichlist == 0 || whichlist == 1);
  if (whichlist >= neighborList.getNumList()) return;
  // Make CPU wait
  cudaCheck(cudaEventSynchronize(sortDoneEvent[whichlist]));
}

//
// Build neighor list
//
void Nlist::buildNeighborList(const int whichlist, const int* zonelist_atom,
			      const double rnl, const double boxx, const double boxy, const double boxz,
			      XYZQ& xyzq) {
  // Sanity checks
  assert(whichlist == 0 || whichlist == 1);
  if (whichlist >= neighborList.getNumList()) return;

  int zone_patom[9];
  zone_patom[0] = 0;
  for (int i=1;i < 9;i++) zone_patom[i] = zonelist_atom[i-1];

  neighborList.build(whichlist, zone_patom, boxx, boxy, boxz, rnl, xyzq.xyzq, loc2glo, stream);
  cudaCheck(cudaEventRecord(buildDoneEvent[whichlist], stream));
}

//
// Make stream_wait until buildDoneEvent clears
//
void Nlist::waitBuildDone(const int whichlist, cudaStream_t stream_wait) {
  cudaCheck(cudaStreamWaitEvent(stream_wait, buildDoneEvent[whichlist], 0));
}
#endif
