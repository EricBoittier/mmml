#ifndef NOCUDAC
#ifndef NBDIRECT_H
#define NBDIRECT_H
#include <cuda.h>
#include "../domdec_gpu/Force.h"
#include "../domdec_gpu/XYZQ.h"
#include "../domdec_gpu/CudaNeighborList.h"
#include "../domdec_gpu/CudaPMEDirectForceBlock.h"
#include "../domdec_gpu/CudaEnergyVirial.h"

//
// Storage class to hold direct non-bonded calculation stuff
//

class MDsim;

class NBdirect {
  friend class MDsim;

private:

  //-------------------
  // Energy and virial
  //-------------------
  CudaEnergyVirial& energyVirial;
  std::string strVdw;
  std::string strElec;
  std::string strEwex;

  //---------------
  // Streams
  //---------------
  // nonbond_stream[0] = local non-bonded calculation
  // nonbond_stream[1] = global non-bonded calculation
  cudaStream_t nonbond_stream[2];
  cudaStream_t nonbond_14_stream;
  
  //---------------
  // Events
  //---------------
  cudaEvent_t nonbond_calc_done_event[2];
  cudaEvent_t nonbond_14_calc_done_event;

  // Direct non-bonded calculation
  // NOTE: we use the base class here. This allows us to call either
  //       the regular or the "block" version of the force calculation
  CudaPMEDirectForceBase<long long int, float> *directforce;

  // If block is used, this pointer will point to directforce
  CudaPMEDirectForceBlock<long long int, float> *directforceBlock;

  // -----------------------------------------
  //                Neighborlist
  // -----------------------------------------

  int ncell;
  int cell_start_len;
  int *cell_start;
  
  int n_ijlist[2];
  int ijlist_len[2];
  int3 *ijlist[2];

  int tile_indj_len[2];
  int *tile_indj[2];

  // List of topological exclusions
  int ntile_top[2];
  int tile_ind_top_len[2];
  int *tile_ind_top[2];

  // Topological exclusions
  int tile_excl_top_len[2];
  tile_excl_t<32> *tile_excl_top[2];

  CudaNeighborListBuild<32> *nlist[2];

  NBdirect(CudaEnergyVirial& energyVirial, CudaBlock *cudaBlock);
  ~NBdirect();

  void calc_virial(const int ncoord, const double boxx, const double boxy, const double boxz,
		   XYZQ& xyzq, Force<long long int>& force, cudaStream_t stream);

  void set_cell_start(int ncell, int *h_cell_start);

  void set_vdwtype(const int ncoord, int *h_vdwtype);
  void set_vdwparam(int h_nvdwparam, float *h_vdwparam);
  void set_vdwparam14(int h_nvdwparam, float *h_vdwparam);
  void set_14_list(int nin14list, int nex14list, xx14list_t* h_in14list, xx14list_t* h_ex14list);
  void set_14_block_pos(int numBlock, int* in14tbl_block_pos, int* ex14tbl_block_pos);
  void set_ijlist(int h_whichlist, int h_n_ijlist, int *h_ijlist);
  void set_ientry(int h_whichlist, int h_n_ientry, ientry_t* h_ientry);

  void build_excl(int h_whichlist, float boxx, float boxy, float boxz, float roff, XYZQ& xyzq);
  void combine_tile_top(int h_whichlist, int h_ntile_top, int *h_tile_ind_top, 
			tile_excl_t<32> *h_tile_excl_top);


  void calc_nonbond_force(int whichlist, bool calc_energy, bool calc_virial, XYZQ& xyzq,
			  Force<long long int>& force, cudaEvent_t clear_done_event,
			  CudaNeighborList<32>* neighborList=NULL);
  void calc_nonbond_14_force(bool calc_energy, bool calc_virial, XYZQ& xyzq,
			     Force<long long int>& force, cudaEvent_t clear_done_event);

  void gpu_wait_force_done(cudaStream_t wait_stream);
  
  void get_energy(double& energy_vdw, double& energy_elec, double& energy_excl);
  
  //int get_force_stride() {return force.stride();}
  //long long int* get_force_xyz() {return force.xyz();}

  //  void get_nonbond_param(float& kappa, float& roff, float& ron, float& e14fac,
  //			 int& vdwmodel, int& elecmodel, bool& qeterm_vdw, bool& qeterm_elec);

  void set_nonbond_param(float boxx, float boxy, float boxz, float kappa,
			 float roff, float ron, float e14fac,
			 int vdwmodel, int elecmodel, bool qeterm_vdw, bool qeterm_elec);
  
  void set_box_size(double boxx, double boxy, double boxz);
  //void reduce_force(cudaStream_t stream);

  //Force<long long int>* get_force() {return &force;}

  cudaStream_t get_nonbond_stream(const int i);
  
};

#endif // NBDIRECT_H
#endif //NOCUDAC
