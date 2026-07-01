#ifndef NOCUDAC
#include "NBrecip.h"

//
// Class creator
//
NBrecip::NBrecip(const int nfftx, const int nffty, const int nfftz,
		 const int forder, const double kappa, CudaEnergyVirial& energyVirial) :
  strEwks("ewks"), strEwse("ewse"),
  energyVirial(energyVirial),
  recip(nfftx, nffty, nfftz, forder, kappa, energyVirial, strEwks.c_str(), strEwse.c_str()) {
  
  h_xyzq_len = 0;
  h_xyzq = NULL;

  //prev_calc_energy = false;
  //prev_calc_virial = false;

  // Create stream
  cudaCheck(cudaStreamCreate(&stream));

  // Create event
  cudaCheck(cudaEventCreate(&force_done_event));

  // Set stream for reciprocal computation
  recip.set_stream(stream);
}

//
// Class destructor
//
NBrecip::~NBrecip() {
  if (h_xyzq != NULL) deallocate_host<float4>(&h_xyzq);
  // Destroy stream
  cudaCheck(cudaStreamDestroy(stream));
  // Destroy event
  cudaCheck(cudaEventDestroy(force_done_event));
}

//
// Copies host (x, y, z, q) to GPU
//
void NBrecip::set_xyzq(const int ncoord, const double *h_x, const double *h_y, const double *h_z, const double *h_q) {
  // (re)allocate h_xyzq
  if (h_xyzq != NULL && h_xyzq_len < ncoord) {
    deallocate_host<float4>(&h_xyzq);
    h_xyzq_len = 0;
    h_xyzq = NULL;
  }
  if (h_xyzq == NULL) {
    h_xyzq_len = ncoord;
    allocate_host<float4>(&h_xyzq, h_xyzq_len);
  }

  // (re)allocate xyzq
  xyzq.realloc(ncoord);

  // Pack (x, y, z, q) into h_xyzq
  int i;
#pragma omp parallel for private(i) schedule(static)
  for (i=0;i < ncoord;i++) {
    h_xyzq[i].x = (float)h_x[i];
    h_xyzq[i].y = (float)h_y[i];
    h_xyzq[i].z = (float)h_z[i];
    h_xyzq[i].w = (float)h_q[i];
  }

  // Copy to GPU
  xyzq.set_xyzq(ncoord, h_xyzq, 0, stream);
}

//
// Calculate reciprocal nonbonded force
//
void NBrecip::calc_recip_force(const double inv_boxx, const double inv_boxy, const double inv_boxz,
			       const bool calc_energy, const bool calc_virial,
                               cudaEvent_t clear_done_event,
			       XYZQ* xyzqp, Force<long long int>* forcep) {
  XYZQ* xyzq_tp = (xyzqp != NULL) ? xyzqp : &xyzq;
  if (xyzq_tp == NULL) {
    std::cerr << "NBrecip::calc_recip_force, xyzq_tp is NULL" << std::endl;
    exit(1);
  }

  cudaCheck(cudaStreamWaitEvent(stream, clear_done_event, 0));
  
  if (forcep != NULL) {
    recip.calc(inv_boxx, inv_boxy, inv_boxz, xyzq_tp->xyzq, xyzq_tp->ncoord,
	       calc_energy, calc_virial, *forcep);
  } else {
    force.realloc(xyzq_tp->ncoord, 1.5f);
    recip.calc(inv_boxx, inv_boxy, inv_boxz, xyzq_tp->xyzq, xyzq_tp->ncoord,
	       calc_energy, calc_virial, force);
  }
  cudaCheck(cudaEventRecord(force_done_event, stream));
}

// Block version:
// Calculate reciprocal nonbonded force
//
void NBrecip::calc_recip_force_block(const double inv_boxx,
                                     const double inv_boxy,
                                     const double inv_boxz,
                                     const bool calc_energy, const bool calc_virial,
                                     cudaEvent_t clear_done_event,
                                     float * bixlam,
                                     double * biflam, const int * blockIndexes,
                                     XYZQ* xyzqp,
                                     Force<long long int>* forcep) {
  XYZQ* xyzq_tp = (xyzqp != NULL) ? xyzqp : &xyzq;
  if (xyzq_tp == NULL) {
    std::cerr << "NBrecip::calc_recip_force, xyzq_tp is NULL" << std::endl;
    exit(1);
  }
  
  cudaCheck(cudaStreamWaitEvent(stream, clear_done_event, 0));
  
  if (forcep != NULL) {
    recip.calc_block(inv_boxx, inv_boxy, inv_boxz,
                     xyzq_tp->xyzq, xyzq_tp->ncoord,
                     bixlam,
                     calc_energy, calc_virial,
                     *forcep, biflam, blockIndexes);
  } else {
    force.realloc(xyzq_tp->ncoord, 1.5f);
    recip.calc_block(inv_boxx, inv_boxy, inv_boxz,
                     xyzq_tp->xyzq, xyzq_tp->ncoord,
                     bixlam,
                     calc_energy, calc_virial,
                     force, biflam, blockIndexes);
  }
  cudaCheck(cudaEventRecord(force_done_event, stream));
}

//
// GPU stream "wait_stream" waits until force computation is done
//
void NBrecip::gpu_wait_force_done(cudaStream_t wait_stream) {
  cudaCheck(cudaStreamWaitEvent(wait_stream, force_done_event, 0));
}

//
// Get energies and virials
//
void NBrecip::get_energy(double& energy_ksum, double& energy_self) {
  energy_ksum = energyVirial.getEnergy(strEwks);
  energy_self = energyVirial.getEnergy(strEwse);
}

#endif //NOCUDAC
