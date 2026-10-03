#if KEY_BLADE == 1
#define BLADE_IN_CHARMM
// #include <cuda_runtime.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <signal.h>

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
#include "main/gpu_check.h"

#if HAS_NVTX == 1
/* NVTX moved under nvtx3/ and the old top-level header was removed in
   CUDA 13.  Prefer the nvtx3 path where available (CUDA 10+), and fall
   back to the old location on older toolkits. */
#if defined(__has_include) && __has_include(<nvtx3/nvToolsExtCuda.h>)
#include <nvtx3/nvToolsExtCuda.h>
#else
#include <nvToolsExtCuda.h>
#endif
#endif


// Global interrupt flag for Ctrl+C handling
volatile sig_atomic_t blade_interrupt_flag = 0;
static struct sigaction blade_old_sigint_action;
static bool blade_signal_handler_installed = false;
static int blade_sigint_count = 0;
static const int BLADE_FORCE_EXIT_COUNT = 3;

// Signal handler for SIGINT (Ctrl+C)
static void blade_sigint_handler(int signum) {
  blade_sigint_count++;
  blade_interrupt_flag = 1;

  if (blade_sigint_count >= BLADE_FORCE_EXIT_COUNT) {
    fprintf(stderr, "\n[BLaDE] %dx Ctrl+C - FORCE EXIT!\n", blade_sigint_count);
    fflush(stderr);
    // Restore default handler and re-raise to exit immediately
    signal(SIGINT, SIG_DFL);
    raise(SIGINT);
    return;
  }

  int remaining = BLADE_FORCE_EXIT_COUNT - blade_sigint_count;
  fprintf(stderr, "\n[BLaDE] SIGINT received, requesting graceful stop... "
          "(press %dx more to force quit)\n", remaining);
  fflush(stderr);
}

// Interrupt handling API functions
extern "C"
void blade_set_interrupt(int value)
{
  blade_interrupt_flag = value;
}

extern "C"
int blade_check_interrupt()
{
  return blade_interrupt_flag;
}

extern "C"
void blade_install_signal_handler()
{
  if (!blade_signal_handler_installed) {
    struct sigaction new_action;
    new_action.sa_handler = blade_sigint_handler;
    sigemptyset(&new_action.sa_mask);
    new_action.sa_flags = 0;
    sigaction(SIGINT, &new_action, &blade_old_sigint_action);
    blade_signal_handler_installed = true;
    blade_interrupt_flag = 0;  // Reset flag when installing handler
    blade_sigint_count = 0;    // Reset rapid Ctrl+C counter
  }
}

extern "C"
void blade_restore_signal_handler()
{
  if (blade_signal_handler_installed) {
    sigaction(SIGINT, &blade_old_sigint_action, NULL);
    blade_signal_handler_installed = false;
  }
}

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
void blade_get_force(System *system,int report_energy,int refill_random)
{
  system+=omp_get_thread_num();
  system->run->freqNRG=1000;
  if(report_energy) {
    system->run->freqNRG=1;
  }
  system->potential->calc_force(system->run->step,system,refill_random);
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

static bool temperature_matches(double actual, double expected)
{
  return fabs(actual-expected) <= 1.0e-6*fmax(1.0,fabs(expected));
}

extern "C"
int blade_exchange_temperature(System *system, double expected, double target)
{
  if (!system || expected <= 0.0 || target <= 0.0) return -2;

  int systemCount=system->idCount > 0 ? system->idCount : 1;
  for (int id=0; id<systemCount; id++) {
    System *local=system+id;
    if (!local->run || !local->state || !local->domdec ||
        !local->state->leapParms1 || !local->state->leapParms2) {
      return -2;
    } else if (local->run->freqNPT > 0) {
      return -3;
    } else if (!temperature_matches(local->run->T,expected) ||
               !temperature_matches(local->state->leapParms1->kT/kB,expected)) {
      return -1;
    }
  }
  if (target == expected) return 1;

  double scale=sqrt(target/expected);
  for (int id=0; id<systemCount; id++) {
    System *local=system+id;
    gpuCheck(cudaSetDevice(local->gpu));
    State *state=local->state;
    int count=3*state->atomCount;
    gpuCheck(cudaMemcpy(state->velocityBuffer,state->velocityBuffer_d,
                        count*sizeof(real_v),cudaMemcpyDeviceToHost));
    for (int i=0; i<count; i++) state->velocityBuffer[i]*=scale;
    gpuCheck(cudaMemcpy(state->velocityBuffer_d,state->velocityBuffer,
                        count*sizeof(real_v),cudaMemcpyHostToDevice));

    local->run->T=target;
    state->leapParms1->kT=kB*target;
    state->leapParms2->noise=sqrt(
      (1-state->leapParms2->friction*state->leapParms2->friction)*kB*target);
    local->domdec->cullPad*=scale;
  }
  return 1;
}

extern "C"
void blade_send_coordinates(System *system)
{
  system+=omp_get_thread_num();
  gpuCheck(cudaMemcpy(system->state->position_d,system->state->position,
                      3*system->state->atomCount*sizeof(real_x),
                      cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy(system->state->theta_d,system->state->theta,
                      system->state->lambdaCount*sizeof(real_x),
                      cudaMemcpyHostToDevice));
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

  cudaDeviceSynchronize();
#pragma omp barrier
  gpuCheck(cudaPeekAtLastError());
#pragma omp barrier
}

extern "C"
int blade_minimizer(System *system,int nsteps,int mintype,double steplen)
{
  system+=omp_get_thread_num();
  Run *r=system->run;
  int status=0;
  system->run->nsteps=nsteps;
  system->run->minType=(EMin)mintype;
  system->run->dxRMSInit=steplen;

  system->state->min_init(system);
  
  for (r->step=0; r->step<r->nsteps; r->step++) {
    if (r->minType!=esdmd || r->step==0) {
      system->domdec->update_domdec(system,true); // true to always update neighbor list
      system->potential->calc_force(0,system,false); // step 0 to always calculate energy
    }
    if (!system->state->min_move(r->step,r->nsteps,system)) {
      status=1;
      break;
    }
    // print_dynamics_output(step,system);
    gpuCheck(cudaPeekAtLastError());
  }
  
  system->state->min_dest(system);
  return status;
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

// C interface to write to CHARMM output stream
extern "C"
void blade_charmm_write_output(const char* message);

#endif /* KEY_BLADE == 1 */
