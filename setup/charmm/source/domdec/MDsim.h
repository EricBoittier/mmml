#ifndef NOCUDAC
#ifndef MDSIM_H
#define MDSIM_H
#include <cuda.h>
#include "../domdec_gpu/XYZQ.h"
#include "../domdec_gpu/hostXYZ.h"
#include "../domdec_gpu/CudaEnergyVirial.h"
#include "../domdec_gpu/CudaRandom.h"
#include "NBdirect.h"
#include "NBrecip.h"
#include "Bonded.h"
#include "Nlist.h"

//
// GPU MD simulation structure that holds the required classes
//
class MDsim {

private:

  //-----------------------
  // Number of coordinates
  //-----------------------
  // Total: ncoord = ncoord_home + ncoord_import
  int ncoord;
  // Number of coordinates in the home box
  int ncoord_home;
  // Number of coordinates in the import region
  int ncoord_import;

  // Box size
  double boxx, boxy, boxz;
  
  //---------------
  // Stream
  //---------------
  cudaStream_t mdsim_stream;

  //----------
  // Event
  //----------
  cudaEvent_t copy_done_event;
  cudaEvent_t clear_done_event;

  //-------------------
  // Host force array:
  // NOTE: h_force_len is the length in BYTES
  //-------------------
  int h_force_double_len;
  double* h_force_double;

  int h_force_float_len;
  float* h_force_float;

  //-------------
  // Block class
  //-------------
  CudaBlock *cudaBlock;

  //---------------------------------------------------
  // Energy and virial for direct and reciprocal space
  //---------------------------------------------------
  CudaEnergyVirial energyVirialDirect;
  CudaEnergyVirial energyVirialRecip;

  // true if energy is calculated on this MD cycle
  bool cur_calc_energy;

  // true if virial is calculated on this MD cycle
  bool cur_calc_virial;

  //-------------------------------
  // Direct Non-bonded calculation
  //-------------------------------
  NBdirect *nbdirect;

  //-----------------------------------
  // Reciprocal Non-bonded calculation
  //-----------------------------------
  NBrecip *nbrecip;

  //--------------------
  // Bonded calculation
  //--------------------
  Bonded *bonded;

  // -----------------
  // Neighbor list
  // -----------------
  Nlist *nlist;

  //-------------
  // Coordinates
  //-------------
  // Temporary unsorted coordinate 
  XYZQ xyzq_unsorted;
  XYZQ xyzq;

  //---------
  // Forces:
  //---------
  Force<long long int> force;
  Force<long long int> recipforce;

  // Cases:
  // Single direct+recip node:                      true
  // Single direct+recip node + other direct nodes: false
  // Single recip node + one direct node:           false
  bool recip_uses_nbdirect_buffers;

  // ----------------------------
  // GPU Random Number Generator
  // ----------------------------
  CudaRandom *cudaRandom;

public:

  MDsim(const int ndirect, const bool direct_on, const bool recip_on, const bool bonded_on,
	const bool nlist_on, const int nx, const int ny, const int nz,
	const int ncoord_global, const int *iblo14, const int *inb14,
	const int nfftx, const int nffty, const int nfftz,
	const int forder, const double kappa, const int numBlock,
        const int use_softcore, const int use_PMEL, const int seed);
  ~MDsim();

  void setTest(const bool test);
  
  void sortCoord(const int whichlist, const int* zonelist_atom, float4* h_xyzq, int* h_loc2glo_ind);
  void waitSortCoord(const int whichlist);
  void buildNeighborList(const int whichlist, const int* zonelist_atom, const double rnl);

  void set_ncoord(int ncoord_home, int ncoord_import);

  void clear_force_virial_energy(const bool calc_energy, const bool calc_virial);

  void reduce_force_virial();

  void get_force_virial_energy();
  void set_homebox_force();

  void* get_force_pointer();
  int get_force_stride();
  int get_force_type();

  void gpu_wait_nonbond_done();
  void cpu_wait_copy_done();

  cudaStream_t get_stream() {return mdsim_stream;}

  void set_ijlist(int h_whichlist, int h_n_ijlist, int *h_ijlist);
  void set_ientry(int h_whichlist, int h_n_ientry, ientry_t* h_ientry);
  void set_cell_start(int ncell, int *h_cell_start);

  void setBlockType(int ncoord, int *h_blockType);
  void setBlockParam(float *h_blockParam);
  void setBixlam(float *h_bixlam);
  void setSiteMLD(int *h_siteMLD);
  void getBiflam(double *h_biflam, double *h_biflam2);
  void setBiflam(double *h_biflam, double *h_biflam2);

  void calc_nonbond_force(int whichlist, bool calc_energy, bool calc_virial);
  void calc_nonbond_14_force(bool calc_energy, bool calc_virial);

  void set_vdwparam(int h_nvdwparam, float *h_vdwparam);
  void set_vdwparam14(int h_nvdwparam, float *h_vdwparam);
  void set_14_list(int nin14list, int nex14list, xx14list_t* h_in14list, xx14list_t* h_ex14list);
  void set_14_block_pos(int* in14tbl_block_pos, int* ex14tbl_block_pos);
  void build_excl(int h_whichlist, float roff);
  void combine_tile_top(int h_whichlist, int h_ntile_top, int *h_tile_ind_top, 
			tile_excl_t<32> *h_tile_excl_top);

  void set_home_xyzq(float4 *h_xyzq);
  void set_import_xyzq(float4 *h_xyzq);
  void set_vdwtype(int *h_vdwtype);
  void set_recip_xyzq(const int ncoord, const double *h_x, const double *h_y, const double *h_z, const double *h_q);

  void set_box_size(const double boxx_in, const double boxy_in, const double boxz_in);

  void set_nonbond_param(float kappa, float roff, float ron, float e14fac,
			 int vdwmodel, int elecmodel,
			 bool qeterm_vdw, bool qeterm_elec);

  void setup_bonded_coef(const int nbondcoef, const float2 *h_bondcoef,
			 const int nureybcoef, const float2 *h_ureybcoef,
			 const int nanglecoef, const float2 *h_anglecoef,
			 const int ndihecoef, const float4 *h_dihecoef,
			 const int nimdihecoef, const float4 *h_imdihecoef,
			 const int ncmapcoef, const float2 *h_cmapcoef);

  void setup_bonded_list(const int nbondlist, const bondlist_t *h_bondlist, 
			 const int nureyblist, const bondlist_t *h_ureyblist,
			 const int nanglelist, const anglelist_t *h_anglelist,
			 const int ndihelist, const dihelist_t *h_dihelist,
			 const int nimdihelist, const dihelist_t *h_imdihelist,
			 const int ncmaplist, const cmaplist_t *h_cmaplist);
  
  void calc_bonded_force(const bool calc_energy, const bool calc_virial,
			 const bool calc_bond, const bool calc_ureyb,
			 const bool calc_angle, const bool calc_dihe,
			 const bool calc_imdihe, const bool calc_cmap);

  void calc_recip_force(const bool calc_energy, const bool calc_virial);
  void calc_recip_force_block(const bool calc_energy, const bool calc_virial);


  void get_direct_virial(double *vir);
  void get_recip_virial(double *vir);
  
  void get_nonbond_energy(double& energy_vdw, double& energy_elec,
			  double& energy_excl);

  void get_bonded_energy(double& energy_bond, double& energy_ureyb,
			 double& energy_angle,
			 double& energy_dihe, double& energy_imdihe,
			 double& energy_cmap);
  
  void get_recip_energy(double& energy_ksum, double& energy_self);

  void generate_rn_gpu(int natom, float* gaussian_rn);
};

#endif // MDSIM_H
#endif //NOCUDAC
