#if KEY_BLADE == 1
#define BLADE_IN_CHARMM
// #include <cuda_runtime.h>
#include <omp.h>
#include <string.h>

#include "run/run.h"
#include "system/system.h"
#include "system/coordinates.h"
#include "io/io.h"
#include "msld/msld.h"
#include "system/state.h"
#include "system/potential.h"
#include "system/selections.h"
#include "holonomic/rectify.h"
#include "holonomic/holonomic.h"
#include "domdec/domdec.h"

#if HAS_NVTX == 1
#include <nvToolsExtCuda.h>
#endif


extern "C"
void blade_set_step(System *system,int istep)
{
  system+=omp_get_thread_num();
  system->run->step=istep;
}

extern "C"
void blade_update_domdec(System *system)
{
  system+=omp_get_thread_num();
  system->domdec->update_domdec(system,(system->run->step%system->domdec->freqDomdec)==0);
}

extern "C"
void blade_rectify_holonomic(System *system)
{
  system+=omp_get_thread_num();
  if (system->id==0) {
    // Not sure it's optimal to call these twice...
    holonomic_rectify(system);
    holonomic_velocity(system);
  }
}

extern "C"
void blade_get_force(System *system,int report_energy)
{
  system+=omp_get_thread_num();
  system->run->freqNRG=1000;
  if(report_energy) {
    system->run->freqNRG=1;
  }
  system->potential->calc_force(system->run->step,system);
  if(report_energy) {
    system->state->kinetic_energy(system);
  }
}

extern "C"
void blade_update(System *system)
{
  system+=omp_get_thread_num();
  system->state->update(system->run->step,system);
}

extern "C"
void blade_check_gpu(System *system)
{
  system+=omp_get_thread_num();
  if (cudaPeekAtLastError() != cudaSuccess) {
    cudaError_t err=cudaPeekAtLastError();
    fatal(__FILE__,__LINE__,"GPU error code %d during run propogation of OMP rank %d\n%s\n",err,system->id,cudaGetErrorString(err));
  }
}

extern "C"
void blade_recv_state(System *system)
{
  system+=omp_get_thread_num();
  system->state->recv_state();
}

extern "C"
void blade_send_state(System *system)
{
  system+=omp_get_thread_num();
  system->state->send_state();
}

extern "C"
void blade_recv_position(System *system)
{
  system+=omp_get_thread_num();
  system->state->recv_position();
}

extern "C"
void blade_recv_theta(System *system)
{
  system+=omp_get_thread_num();
  cudaMemcpy(system->state->theta,system->state->theta_d,system->state->lambdaCount*sizeof(real_x),cudaMemcpyDeviceToHost);
}

extern "C"
void blade_recv_energy(System *system)
{
  system+=omp_get_thread_num();
  system->state->recv_energy();
}

extern "C"
void blade_recv_force(System *system)
{
  system+=omp_get_thread_num();
  cudaMemcpy(system->state->forceBuffer,
             system->state->forceBuffer_d,
             (2*system->state->lambdaCount+3*system->state->atomCount)*sizeof(real_f),
             cudaMemcpyDeviceToHost);
}

extern "C"
int blade_get_atom_count(System *system)
{
  system+=omp_get_thread_num();
  return system->state->atomCount;
}

extern "C"
int charmm_recv_position(System *system, double * out_pos)
{
  if (omp_get_thread_num()!=0) fatal(__FILE__,__LINE__,"Only master thread may call this function\n");
  int n = system->state->atomCount;
  //  memcpy(out_pos, system->state->position, n * sizeof(double));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++)
      out_pos[i * 3 + j] = system->state->position[i][j];
  }
  return n;
}

extern "C"
int charmm_send_position(System *system, double * out_pos)
{
  system+=omp_get_thread_num();
  int n = system->state->atomCount;
  //  memcpy(out_pos, system->state->position, n * sizeof(double));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++)
      system->state->position[i][j] = out_pos[i * 3 + j];
  }
  return n;
}

extern "C"
int charmm_recv_velocity(System *system, double * out_vel)
{
  if (omp_get_thread_num()!=0) fatal(__FILE__,__LINE__,"Only master thread may call this function\n");
  int n = system->state->atomCount;
  //  memcpy(out_pos, system->state->position, n * sizeof(double));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++)
      out_vel[i * 3 + j] = system->state->velocity[i][j];
  }
  return n;
}

extern "C"
int charmm_send_velocity(System *system, double * out_vel)
{
  system+=omp_get_thread_num();
  int n = system->state->atomCount;
  //  memcpy(out_pos, system->state->position, n * sizeof(double));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++)
      system->state->velocity[i][j] = out_vel[i * 3 + j];
  }
  return n;
}

extern "C"
void charmm_recv_box(System *system, double * out_box)
{
  if (omp_get_thread_num()!=0) fatal(__FILE__,__LINE__,"Only master thread may call this function\n");
  // system->state->box holds current state of box
  // system->coordinates->particleBoxABC and particleBoxAlBeGa are used for system setup
  out_box[0] = system->state->box.a.x;
  out_box[1] = system->state->box.a.y;
  out_box[2] = system->state->box.a.z;
  out_box[3] = system->state->box.b.x;
  out_box[4] = system->state->box.b.y;
  out_box[5] = system->state->box.b.z;
}

extern "C"
void charmm_send_box(System *system, double * out_box)
{
  system+=omp_get_thread_num();
  // system->state->box holds current state of box
  // system->coordinates->particleBoxABC and particleBoxAlBeGa are used for system setup
  system->coordinates->particleBoxABC.x = out_box[0];
  system->coordinates->particleBoxABC.y = out_box[1];
  system->coordinates->particleBoxABC.z = out_box[2];
  system->coordinates->particleBoxAlBeGa.x = out_box[3];
  system->coordinates->particleBoxAlBeGa.y = out_box[4];
  system->coordinates->particleBoxAlBeGa.z = out_box[5];
  if (system->state) {
    system->state->box.a.x = out_box[0];
    system->state->box.a.y = out_box[1];
    system->state->box.a.z = out_box[2];
    system->state->box.b.x = out_box[3];
    system->state->box.b.y = out_box[4];
    system->state->box.b.z = out_box[5];
  }
}

extern "C"
int blade_get_lambda_count(System *system)
{
  system+=omp_get_thread_num();
  return system->state->lambdaCount;
}

extern "C"
int charmm_recv_theta(System *system, double * out_the)
{
  if (omp_get_thread_num()!=0) fatal(__FILE__,__LINE__,"Only master thread may call this function\n");
  int n = system->state->lambdaCount;
  // memcpy(out_the, system->state->theta, n * sizeof(double));
  for (int i=0; i<n; i++) {
    out_the[i]=system->state->theta[i];
  }
  return n;
}

extern "C"
int charmm_send_theta(System *system, double * out_the)
{
  system+=omp_get_thread_num();
  int n = system->state->lambdaCount;
  // memcpy(out_the, system->state->theta, n * sizeof(double));
  system->state->theta[0] = 0;
  for (int i=1; i<n; i++) {
    system->state->theta[i] = out_the[i];
  }
  return n;
}

extern "C"
int charmm_recv_thetavelocity(System *system, double * out_the)
{
  if (omp_get_thread_num()!=0) fatal(__FILE__,__LINE__,"Only master thread may call this function\n");
  int n = system->state->lambdaCount;
  // memcpy(out_the, system->state->theta, n * sizeof(double));
  for (int i=0; i<n; i++) {
    out_the[i]=system->state->thetaVelocity[i];
  }
  return n;
}

extern "C"
int charmm_send_thetavelocity(System *system, double * out_the)
{
  system+=omp_get_thread_num();
  int n = system->state->lambdaCount;
  // memcpy(out_the, system->state->theta, n * sizeof(double));
  system->state->thetaVelocity[0] = 0;
  for (int i=1; i<n; i++) {
    system->state->thetaVelocity[i] = out_the[i];
  }
  return n;
}

extern "C"
int blade_get_energy_count(void)
{
  return eeend;
}

extern "C"
void charmm_recv_energy(System *system, double *out_energy)
{
  if (omp_get_thread_num()!=0) fatal(__FILE__,__LINE__,"Only master thread may call this function\n");
  int n = eeend;
  for (int i=0; i<n; i++) {
    out_energy[i]=system->state->energy[i];
  }
}

extern "C"
void charmm_recv_force(System *system, double *out_force, double *out_flambda)
{
  if (omp_get_thread_num()!=0) fatal(__FILE__,__LINE__,"Only master thread may call this function\n");
  int n;
  n=system->state->lambdaCount;
  for (int i=0; i<n; i++) {
    out_flambda[i]=system->state->lambdaForce[i];
  }
  n=system->state->atomCount;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++)
      out_force[i * 3 + j] = system->state->force[i][j];
  }
}

extern "C"
void blade_set_calctermflag(System *system,int term,int value)
{
  // removal of these commented out print statements causes an error why?
  //fprintf(stdout,"This is the current value %d\n",value);
  system+=omp_get_thread_num();
  if (term>=0 && term<eeend) {
    //fprintf(stdout,"This is the current value in system %d\n",system->run->calcTermFlag[term]);
    system->run->calcTermFlag[term]=value;
  }
}

extern "C"
void blade_calc_lambda_from_theta(System *system)
{
  system+=omp_get_thread_num();
  system->msld->calc_lambda_from_theta(0,system);
}

extern "C"
void blade_init_lambda_from_theta(System *system)
{
  system+=omp_get_thread_num();
  system->msld->init_lambda_from_theta(0,system);
}

extern "C"
void blade_prettify_position(System *system)
{
  system+=omp_get_thread_num();
  system->run->prettyXTC=true;
  system->state->prettify_position(system);
}

extern "C"
void blade_dynamics_initialize(System *system)
{
  system+=omp_get_thread_num();
  system->run->domdecHeuristic=false; // Hard code this in here, it's the more stable option
  // Finish setting up MSLD
  system->msld->initialize(system);

  // Set up update structures
  if (system->state) delete system->state;
  system->state=new State(system);
  system->state->initialize(system);

  // Set up potential structures
  if (system->potential) delete system->potential;
  system->potential=new Potential();
  system->potential->initialize(system);

  // Rectify bond constraints
  holonomic_rectify(system);

  // Read checkpoint
  // if (fnmCPI!="") {
  //   read_checkpoint_file(fnmCPI.c_str(),system);
  // }

  // Set up domain decomposition
  if (system->domdec) delete system->domdec;
  system->domdec=new Domdec();
  system->domdec->initialize(system);

  // NYI check gpu
  cudaDeviceSynchronize();
#pragma omp barrier
  if (cudaPeekAtLastError() != cudaSuccess) {
    cudaError_t err=cudaPeekAtLastError();
    fatal(__FILE__,__LINE__,"GPU error code %d during run initialization of OMP rank %d\n%s\n",err,system->id,cudaGetErrorString(err));
  }
#pragma omp barrier
}

extern "C"
void blade_range_begin(char *range_name)
{
#if HAS_NVTX == 1
  nvtxEventAttributes_t att = {0};
  att.messageType = NVTX_MESSAGE_TYPE_ASCII;
  att.message.ascii = range_name;
  nvtxRangePushEx(&att);
#endif
}

extern "C"
void blade_range_end()
{
#if HAS_NVTX == 1
  nvtxRangePop();
#endif
}
#endif /* KEY_BLADE == 1 */
