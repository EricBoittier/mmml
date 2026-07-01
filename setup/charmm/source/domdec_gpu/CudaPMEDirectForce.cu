#ifndef NOCUDAC
#include <iostream>
#include <fstream>
#include <cassert>
#include <cuda.h>
#include <math.h>
#include "gpu_utils.h"
#include "cuda_utils.h"
#include "CudaPMEDirectForce.h"
#include "CudaDirectForceKernels.h"

//
// Sets vdwtype from a global list
//
__global__ void set_vdwtype_kernel(const int ncoord, const int* __restrict__ glo_vdwtype,
				   const int* __restrict__ loc2glo, int* __restrict__ vdwtype) {
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < ncoord) {
    int j = loc2glo[i];
    vdwtype[i] = glo_vdwtype[j];
  }
}

__global__ void set_14_list_kernel(const int nin14_tbl, const int* __restrict__ in14_tbl,
				   const xx14_t* __restrict__ in14, xx14list_t* __restrict__ in14list,
				   const int nex14_tbl, const int* __restrict__ ex14_tbl,
				   const xx14_t* __restrict__ ex14, xx14list_t* __restrict__ ex14list,
				   const float4* __restrict__ xyzq,
				   const float3 half_box, const int*__restrict__ glo2loc_ind) {
  int pos = threadIdx.x + blockIdx.x*blockDim.x;
  if (pos < nin14_tbl) {
    int j = in14_tbl[pos];
    xx14_t in14v = in14[j];
    xx14list_t in14listv;
    in14listv.i = glo2loc_ind[in14v.i];
    in14listv.j = glo2loc_ind[in14v.j];
    float4 xyzq_i = xyzq[in14listv.i];
    float4 xyzq_j = xyzq[in14listv.j];
    in14listv.ishift = calc_ishift(xyzq_i, xyzq_j, half_box);
    in14list[pos] = in14listv;
  } else if (pos < nin14_tbl + nex14_tbl) {
    pos -= nin14_tbl;
    int j = ex14_tbl[pos];
    xx14_t ex14v = ex14[j];
    xx14list_t ex14listv;
    ex14listv.i = glo2loc_ind[ex14v.i];
    ex14listv.j = glo2loc_ind[ex14v.j];
    float4 xyzq_i = xyzq[ex14listv.i];
    float4 xyzq_j = xyzq[ex14listv.j];
    ex14listv.ishift = calc_ishift(xyzq_i, xyzq_j, half_box);
    ex14list[pos] = ex14listv;
  }
}

//########################################################################################
//########################################################################################
//########################################################################################

//
// Class creator
//
template <typename AT, typename CT>
CudaPMEDirectForce<AT, CT>::CudaPMEDirectForce(CudaEnergyVirial &energyVirial,
					       const char *nameVdw, const char *nameElec, const char *nameExcl) : 
  use_tex_vdwparam(true), use_tex_vdwparam14(true), energyVirial(energyVirial) {

  assert(nameVdw != NULL);
  assert(nameElec != NULL);
  assert(nameExcl != NULL);
  
  // Insert energy terms
  energyVirial.insert(nameVdw);
  strVdw = nameVdw;

  energyVirial.insert(nameElec);
  strElec = nameElec;

  energyVirial.insert(nameExcl);
  strExcl = nameExcl;
  
#ifdef USE_TEXTURE_OBJECTS
  vdwParamTexObjActive = false;
  vdwParam14TexObjActive = false;
#else
  // Assert that texture references must be unbound
  assert(!get_vdwparam_texref_bound());
  assert(!get_vdwparam14_texref_bound());
#endif 

  vdwparam = NULL;
  nvdwparam = 0;
  vdwparam_len = 0;

  vdwparam14 = NULL;
  nvdwparam14 = 0;
  vdwparam14_len = 0;

  nin14list = 0;
  in14list_len = 0;
  in14list = NULL;

  nex14list = 0;
  ex14list_len = 0;
  ex14list = NULL;

  vdwtype = NULL;
  vdwtype_len = 0;

  ewald_force = NULL;
  n_ewald_force = 0;

  set_calc_vdw(true);
  set_calc_elec(true);

  //allocate<DirectEnergyVirial_t>(&d_energy_virial, 1);
  //allocate_host<DirectEnergyVirial_t>(&h_energy_virial, 1);
  
  allocate_host<DirectSettings_t>(&h_setup, 1);

  //clear_energy_virial();
}

//
// Class destructor
//
template <typename AT, typename CT>
CudaPMEDirectForce<AT, CT>::~CudaPMEDirectForce() {
  this->clearTextures();
  if (vdwparam != NULL) deallocate<CT>(&vdwparam);
  if (vdwparam14 != NULL) deallocate<CT>(&vdwparam14);
  if (in14list != NULL) deallocate<xx14list_t>(&in14list);
  if (ex14list != NULL) deallocate<xx14list_t>(&ex14list);
  if (vdwtype != NULL) deallocate<int>(&vdwtype);
  if (ewald_force != NULL) deallocate<CT>(&ewald_force);
  //if (d_energy_virial != NULL) deallocate<DirectEnergyVirial_t>(&d_energy_virial);
  //if (h_energy_virial != NULL) deallocate_host<DirectEnergyVirial_t>(&h_energy_virial);
  if (h_setup != NULL) deallocate_host<DirectSettings_t>(&h_setup);
}

//
// Copies h_setup -> d_setup
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::update_setup() {
  updateDirectForceSetup(h_setup);
}

//
// Sets parameters for the nonbonded computation
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::setup(double boxx, double boxy, double boxz, double kappa,
				       double roff, double ron, double e14fac,
				       int vdw_model, int elec_model) {

  double ron2 = ron*ron;
  double ron4 = ron2*ron2;
  double ron3 = ron*ron*ron;
  double ron5 = ron3*ron2;
  double ron6 = ron3*ron3;

  double roff2 = roff*roff;
  double roff3 = roff*roff*roff;
  double roff4 = roff2*roff2;
  double roff5 = roff4*roff;
  double roff6 = roff3*roff3;
  double roff8 = roff6*roff2;
  double roff12 = roff6*roff6;
  double roff14 = roff12*roff2;

  h_setup->boxx = boxx;
  h_setup->boxy = boxy;
  h_setup->boxz = boxz;
  h_setup->kappa = kappa;
  h_setup->kappa2 = kappa*kappa;
  h_setup->roff = roff;
  h_setup->roff2 = roff2;
  h_setup->roff3 = roff3;
  h_setup->roff5 = roff5;
  h_setup->ron = ron;
  h_setup->ron2 = ron2;

  h_setup->roffinv =  ((CT)1.0)/roff;
  h_setup->roffinv2 =  ((CT)1.0)/roff2;
  h_setup->roffinv3 =  ((CT)1.0)/roff3;
  h_setup->roffinv4 = ((CT)1.0)/(roff2*roff2);
  h_setup->roffinv5 = ((CT)1.0)/(roff*roff2*roff2);
  h_setup->roffinv6 = ((CT)1.0)/(roff2*roff2*roff2);
  h_setup->roffinv12 = h_setup->roffinv6*h_setup->roffinv6;
  h_setup->roffinv18 = h_setup->roffinv12*h_setup->roffinv6;

  double roff2_min_ron2 = roff2 - ron2;
  double inv_roff2_ron2_3 = 1.0/(roff2_min_ron2*roff2_min_ron2*roff2_min_ron2);
  h_setup->inv_roff2_ron2_3 = (CT)inv_roff2_ron2_3;
 
  // Constants for VFSW
  if (ron < roff) {
    h_setup->k6 = roff3/(roff3 - ron3);
    h_setup->k12 = roff6/(roff6 - ron6);
    h_setup->dv6 = -((CT)1.0)/(ron3*roff3);
    h_setup->dv12 = -((CT)1.0)/(ron6*roff6);
  } else {
    h_setup->k6 = 1.0f;
    h_setup->k12 = 1.0f;
    h_setup->dv6 = -((CT)1.0)/(roff6);
    h_setup->dv12 = -((CT)1.0)/(roff6*roff6);
  }

  double g = (roff2 - ron2)*(roff2 - ron2)*(roff2 - ron2);
  h_setup->Aconst = roff4*(roff2 - 3.0*ron2)/g;
  h_setup->Bconst = 6.0*roff2*ron2/g;
  h_setup->Cconst = -(ron2 + roff2)/g;
  h_setup->Dconst = 2.0/(5.0*g);
  h_setup->dvc = 8.0*(ron2*roff2*(roff-ron) - (roff5 - ron5)/5.0)/g;
  if (ron < roff) {
    double Denom = 1.0/g;
    h_setup->Denom = Denom;
    double Acoef = roff4*(roff2 - 3.0*ron2)*Denom;
    h_setup->Acoef = Acoef;
    double Bcoef = 6.0*roff2*ron2*Denom;
    h_setup->Bcoef = Bcoef;
    double Ccoef = -3.0*(ron2 + roff2)*Denom;
    h_setup->Ccoef = Ccoef;
    h_setup->Constr = 2.0*Bcoef*log(roff) - Acoef/roff2 + Ccoef*roff2 + Denom*roff4;
    h_setup->Eaddr = (12.0*ron2*roff2*log(roff/ron) - 3.0*(roff4 - ron4))*Denom;
  } else {
    h_setup->Eaddr = -1.0/roff2;
  }
  
  // Constants for Gromacs-style potentials
  // Copy-pasted from Michael G. Lerner code
  h_setup->GAconst = (CT)(5.0/(3.0*roff));
  h_setup->GBcoef = (CT)(5.0/(3.0*roff2*roff2));
  double roff_ron = roff - ron;
  double roff_ron2 = roff_ron*roff_ron;
  double roff_ron3 = roff_ron*roff_ron2;
  double roff_ron4 = roff_ron2*roff_ron2;
  // NOTE: ga6, gb6, gc6 are divided by 6
  //       ga12, gb12, gc12 are divided by 12
  double ga6 = -(10.0*roff - 7.0*ron)/(roff8*roff_ron2);
  double gb6 =  ( 9.0*roff - 7.0*ron)/(roff8*roff_ron3);
  h_setup->ga6  = ga6;
  h_setup->gb6  = gb6;
  h_setup->gc6  = (1.0/roff6)/6.0 - (ga6*roff_ron3)/3.0 - (gb6*roff_ron4)/4.0;
  double ga12 = -(16.0*roff - 13.0*ron)/(roff14*roff_ron2);
  double gb12 =  (15.0*roff - 13.0*ron)/(roff14*roff_ron3);
  h_setup->ga12 = ga12;
  h_setup->gb12 = gb12;
  h_setup->gc12 = (1.0/roff12)/12.0 - (ga12*roff_ron3)/3.0 - (gb12*roff_ron4)/4.0;

  h_setup->e14fac = e14fac;

  // IPS parameters for constant dielectric coefficient
  double rips=roff;
  double rips2=rips*rips;
  double ripsr=(CT)(1.0/rips);
  double rips2r=ripsr*ripsr;
  h_setup->rips=rips;
  h_setup->rips2=rips2;
  h_setup->ripsr=ripsr;
  h_setup->rips2r=rips2r;
  h_setup->rips6r=rips2r*rips2r*rips2r;
  h_setup->rips12r=h_setup->rips6r*h_setup->rips6r;
  
  h_setup->pipsec=(CT)(35.0/16.0);
  h_setup->aipse0=(CT)(-35.0/16.0);
  //h_setup->aipse0=(CT)0.0;
  h_setup->aipse1=(CT)(35.0/16.0);
  h_setup->aipse2=(CT)(-21.0/16.0);
  h_setup->aipse3=(CT)(5.0/16.0);
  h_setup->aipse4=(CT)0.0;
  h_setup->aipse5=(CT)0.0;
  h_setup->aipse6=(CT)0.0;
  
  h_setup->bipse1=2.0*h_setup->aipse1;
  h_setup->bipse2=4.0*h_setup->aipse2;
  h_setup->bipse3=6.0*h_setup->aipse3;
  h_setup->bipse4=8.0*h_setup->aipse4;
  h_setup->bipse5=10.0*h_setup->aipse5;
  h_setup->bipse6=12.0*h_setup->aipse6;

  h_setup->pipsvcc=(CT)(1142321.0/629472.0);
  h_setup->pipsvcc=(CT)(2.8147289);
  h_setup->aipsvc0=(CT)(109.0/249.0)-h_setup->pipsvcc;
  //h_setup->aipsvc0=(CT)(0.437751);
  h_setup->aipsvc1=(CT)(43.0/79.0);
  h_setup->aipsvc2=(CT)(7.0/16.0);
  h_setup->aipsvc3=(CT)0.0;
  h_setup->aipsvc4=(CT)(999.0/2528.0);
  h_setup->aipsvc4=(CT)(0.395174);
  h_setup->aipsvc5=(CT)0.0;
  h_setup->aipsvc6=(CT)0.0;
  
  h_setup->bipsvc1=2.0*h_setup->aipsvc1;
  h_setup->bipsvc2=4.0*h_setup->aipsvc2;
  h_setup->bipsvc3=6.0*h_setup->aipsvc3;
  h_setup->bipsvc4=8.0*h_setup->aipsvc4;
  h_setup->bipsvc5=12.0*h_setup->aipsvc5;
  h_setup->bipsvc6=16.0*h_setup->aipsvc6;

  h_setup->pipsvac=(CT)(114769.0/98910.0);
  h_setup->pipsvac=(CT)(2.1603377);
  h_setup->aipsva0=(CT)(1.0/157.0)-h_setup->pipsvac;
  //h_setup->aipsva0=(CT)(0.0063694);
  h_setup->aipsva1=(CT)(1.0/20.0);
  h_setup->aipsva2=(CT)0.0;
  h_setup->aipsva3=(CT)(4.0/9.0);
  h_setup->aipsva4=(CT)0.0;
  h_setup->aipsva5=(CT)(277.0/420.0);
  h_setup->aipsva5=(CT)(0.65952381);
  h_setup->aipsva6=(CT)0.0;
  
  h_setup->bipsva1=2.0*h_setup->aipsva1;
  h_setup->bipsva2=4.0*h_setup->aipsva2;
  h_setup->bipsva3=6.0*h_setup->aipsva3;
  h_setup->bipsva4=10.0*h_setup->aipsva4;
  h_setup->bipsva5=14.0*h_setup->aipsva5;
  h_setup->bipsva6=18.0*h_setup->aipsva6;

  this->vdw_model = vdw_model;
  set_elec_model(elec_model);

  set_calc_vdw(true);
  set_calc_elec(true);

  update_setup();
}

//
// Returns parameters for the nonbonded computation
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::get_setup(float& boxx, float& boxy, float& boxz, float& kappa,
					   float& roff, float& ron, float& e14fac,
					   int& vdw_model, int& elec_model) {
  boxx       = h_setup->boxx;
  boxy       = h_setup->boxy;
  boxz       = h_setup->boxz;
  kappa      = h_setup->kappa;
  roff       = h_setup->roff;
  ron        = h_setup->ron;
  e14fac     = h_setup->e14fac;
  vdw_model  = this->vdw_model;
  elec_model = this->elec_model;
}

//
// Returns box sizes
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::get_box_size(CT &boxx, CT &boxy, CT &boxz) {
  boxx = h_setup->boxx;
  boxy = h_setup->boxy;
  boxz = h_setup->boxz;
}

//
// Sets box sizes
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_box_size(const CT boxx, const CT boxy, const CT boxz) {
  h_setup->boxx = boxx;
  h_setup->boxy = boxy;
  h_setup->boxz = boxz;
  update_setup();
}

//
// Sets "calc_vdw" flag
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_calc_vdw(const bool calc_vdw) {
  this->calc_vdw = calc_vdw;
}

//
// Sets "calc_elec" flag
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_calc_elec(const bool calc_elec) {
  this->calc_elec = calc_elec;
}

//
// Returns "calc_vdw" flag
//
template <typename AT, typename CT>
bool CudaPMEDirectForce<AT, CT>::get_calc_vdw() {
  return this->calc_vdw;
}

//
// Returns "calc_elec" flag
//
template <typename AT, typename CT>
bool CudaPMEDirectForce<AT, CT>::get_calc_elec() {
  return this->calc_elec;
}

//
// Unbinds textures if they're bound
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::clearTextures() {
#ifdef USE_TEXTURE_OBJECTS
  if (vdwParamTexObjActive) {
    cudaCheck(cudaDestroyTextureObject(vdwParamTexObjActive));
    vdwParamTexObjActive = false;
  }
  if (vdwParam14TexObjActive) {
    cudaCheck(cudaDestroyTextureObject(vdwParam14TexObjActive));
    vdwParam14TexObjActive = false;
  }
#else
  if (get_vdwparam_texref_bound()) {
    cudaCheck(cudaUnbindTexture(*get_vdwparam_texref()));
    set_vdwparam_texref_bound(false);
  }
  if (get_vdwparam14_texref_bound()) {
    cudaCheck(cudaUnbindTexture(*get_vdwparam14_texref()));
    set_vdwparam14_texref_bound(false);
  }
#endif
}

//
// Sets VdW parameters by copying them from CPU
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::setup_vdwparam(const int type, const int h_nvdwparam, const CT *h_vdwparam) {
  assert(type == VDW_MAIN || type == VDW_IN14);

  int *nvdwparam_loc;
  int *vdwparam_len_loc;
  CT **vdwparam_loc;

  if (type == VDW_MAIN) {
    nvdwparam_loc = &this->nvdwparam;
    vdwparam_len_loc = &this->vdwparam_len;
    vdwparam_loc = &this->vdwparam;
  } else {
    nvdwparam_loc = &this->nvdwparam14;
    vdwparam_len_loc = &this->vdwparam14_len;
    vdwparam_loc = &this->vdwparam14;
  }

  *nvdwparam_loc = h_nvdwparam;

  // "Fix" vdwparam by multiplying c6 by 6.0 and c12 by 12.0
  // NOTE: this is done in order to avoid the multiplication in the inner loop
  CT *h_vdwparam_fixed = new CT[*nvdwparam_loc];
  for(int i=0;i < *nvdwparam_loc/2;i++) {
    h_vdwparam_fixed[i*2]   = ((CT)6.0)*h_vdwparam[i*2];
    h_vdwparam_fixed[i*2+1] = ((CT)12.0)*h_vdwparam[i*2+1];
  }

  bool vdwparam_reallocated = false;
  if (*nvdwparam_loc > *vdwparam_len_loc) {
    reallocate<CT>(vdwparam_loc, vdwparam_len_loc, *nvdwparam_loc, 1.0f);
    vdwparam_reallocated = true;
  }
  copy_HtoD_sync<CT>(h_vdwparam_fixed, *vdwparam_loc, *nvdwparam_loc);
  delete [] h_vdwparam_fixed;

  const bool *use_tex_vdwparam_loc = (type == VDW_MAIN) ? &this->use_tex_vdwparam : &this->use_tex_vdwparam14;
#ifdef USE_TEXTURE_OBJECTS
  bool *texActive = (type == VDW_MAIN) ? &vdwParamTexObjActive : &vdwParam14TexObjActive;
  cudaTextureObject_t *tex = (type == VDW_MAIN) ? &vdwParamTexObj : &vdwParam14TexObj;
#else
  //bool* vdwparam_texref_bound_loc =
  //  (type == VDW_MAIN) ? get_vdwparam_texref_bound() : get_vdwparam14_texref_bound();
  texture<float2, 1, cudaReadModeElementType> *vdwparam_texref_loc = 
    (type == VDW_MAIN) ? get_vdwparam_texref() : get_vdwparam14_texref();
#endif

  if (*use_tex_vdwparam_loc && vdwparam_reallocated) {
#ifdef USE_TEXTURE_OBJECTS
    if (*texActive) {
      cudaDestroyTextureObject(*tex);
      *texActive = false;
    }
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = *vdwparam_loc;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = sizeof(CT)*8;
    resDesc.res.linear.desc.y = sizeof(CT)*8;
    resDesc.res.linear.sizeInBytes = (*nvdwparam_loc)*sizeof(CT);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    cudaCheck(cudaCreateTextureObject(tex, &resDesc, &texDesc, NULL));
    *texActive = true;
#else
    // Unbind texture
    if ((type == VDW_MAIN) ? get_vdwparam_texref_bound() : get_vdwparam14_texref_bound()) {
      cudaCheck(cudaUnbindTexture(*vdwparam_texref_loc));
      (type == VDW_MAIN) ? set_vdwparam_texref_bound(false) : set_vdwparam14_texref_bound(false);
    }
    // Bind texture
    memset(vdwparam_texref_loc, 0, sizeof(*vdwparam_texref_loc));
    vdwparam_texref_loc->normalized = 0;
    vdwparam_texref_loc->filterMode = cudaFilterModePoint;
    vdwparam_texref_loc->addressMode[0] = cudaAddressModeClamp;
    vdwparam_texref_loc->channelDesc.x = sizeof(CT)*8;
    vdwparam_texref_loc->channelDesc.y = sizeof(CT)*8;
    vdwparam_texref_loc->channelDesc.z = 0;
    vdwparam_texref_loc->channelDesc.w = 0;
    vdwparam_texref_loc->channelDesc.f = cudaChannelFormatKindFloat;
    cudaCheck(cudaBindTexture(NULL, *vdwparam_texref_loc, *vdwparam_loc, 
			      (*nvdwparam_loc)*sizeof(CT)));
    (type == VDW_MAIN) ? set_vdwparam_texref_bound(true) : set_vdwparam14_texref_bound(true);
#endif
  }
}

//
// Loads vdwparam from a file
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::load_vdwparam(const char *filename, const int nvdwparam, CT **h_vdwparam) {
  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    // Open file
    file.open(filename);
    *h_vdwparam = new float[nvdwparam];
    for (int i=0;i < nvdwparam;i++) {
      file >> (*h_vdwparam)[i];
    }
    file.close();
  }
  catch(std::ifstream::failure e) {
    std::cerr << "Error opening/reading/closing file " << filename << std::endl;
    exit(1);
  }
}

//
// Sets VdW parameters by copying them from CPU
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwparam(const int nvdwparam, const CT *h_vdwparam) {
  setup_vdwparam(VDW_MAIN, nvdwparam, h_vdwparam);
}

//
// Sets VdW parameters by loading them from a file
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwparam(const int nvdwparam, const char *filename) {  
  CT *h_vdwparam;
  load_vdwparam(filename, nvdwparam, &h_vdwparam);
  setup_vdwparam(VDW_MAIN, nvdwparam, h_vdwparam);
  delete [] h_vdwparam;
}

//
// Sets VdW parameters by copying them from CPU
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwparam14(const int nvdwparam, const CT *h_vdwparam) {
  setup_vdwparam(VDW_IN14, nvdwparam, h_vdwparam);
}

//
// Sets VdW parameters by loading them from a file
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwparam14(const int nvdwparam, const char *filename) {  
  CT *h_vdwparam;
  load_vdwparam(filename, nvdwparam, &h_vdwparam);
  setup_vdwparam(VDW_IN14, nvdwparam, h_vdwparam);
  delete [] h_vdwparam;
}

//
// Sets vdwtype array from global list in device memory memory
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwtype(const int ncoord, const int *glo_vdwtype,
					     const int *loc2glo, cudaStream_t stream) {
  // Align ncoord to warpsize
  int ncoord_aligned = ((ncoord-1)/warpsize+1)*warpsize;
  reallocate<int>(&vdwtype, &vdwtype_len, ncoord_aligned, 1.2f);

  int nthread = 512;
  int nblock = (ncoord - 1)/nthread + 1;
  set_vdwtype_kernel<<< nblock, nthread, 0, stream >>>
    (ncoord, glo_vdwtype, loc2glo, vdwtype);
  cudaCheck(cudaGetLastError());
}

//
// Sets vdwtype array from host memory
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwtype(const int ncoord, const int *h_vdwtype) {
  // Align ncoord to warpsize
  int ncoord_aligned = ((ncoord-1)/warpsize+1)*warpsize;
  reallocate<int>(&vdwtype, &vdwtype_len, ncoord_aligned, 1.2f);
  copy_HtoD_sync<int>(h_vdwtype, vdwtype, ncoord);
}

//
// Sets vdwtype array by loading it from a file
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_vdwtype(const int ncoord, const char *filename) {

  int *h_vdwtype;

  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    // Open file
    file.open(filename);

    h_vdwtype = new int[ncoord];

    for (int i=0;i < ncoord;i++) {
      file >> h_vdwtype[i];
    }

    file.close();
  }
  catch(std::ifstream::failure e) {
    std::cerr << "Error opening/reading/closing file " << filename << std::endl;
    exit(1);
  }

  set_vdwtype(ncoord, h_vdwtype);
  
  delete [] h_vdwtype;
}

//
// Sets 1-4 interaction and exclusion lists
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_14_list(int nin14list, int nex14list,
					     xx14list_t* h_in14list, xx14list_t* h_ex14list,
					     cudaStream_t stream) {

  this->nin14list = nin14list;
  this->nex14list = nex14list;

  if (nin14list > 0) {
    reallocate<xx14list_t>(&in14list, &in14list_len, nin14list);
    copy_HtoD<xx14list_t>(h_in14list, in14list, nin14list, stream);
  }

  if (nex14list > 0) {
    reallocate<xx14list_t>(&ex14list, &ex14list_len, nex14list);
    copy_HtoD<xx14list_t>(h_ex14list, ex14list, nex14list, stream);
  }

}

//
// Setup 1-4 interaction and exclusion lists from device memory using global data:
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_14_list(const float4 *xyzq,
					     const float boxx, const float boxy, const float boxz,
					     const int *glo2loc_ind,
					     const int nin14_tbl, const int *in14_tbl, const xx14_t *in14,
					     const int nex14_tbl, const int *ex14_tbl, const xx14_t *ex14,
					     cudaStream_t stream) {

  this->nin14list = nin14_tbl;
  if (nin14list > 0) reallocate<xx14list_t>(&in14list, &in14list_len, nin14list, 1.2f);

  this->nex14list = nex14_tbl;
  if (nex14list > 0) reallocate<xx14list_t>(&ex14list, &ex14list_len, nex14list, 1.2f);

  float3 half_box;
  half_box.x = boxx*0.5f;
  half_box.y = boxy*0.5f;
  half_box.z = boxz*0.5f;

  int nthread = 512;
  int nblock = (nin14_tbl + nex14_tbl - 1)/nthread + 1;
  set_14_list_kernel<<< nblock, nthread, 0, stream >>>
    (nin14_tbl, in14_tbl, in14, in14list,
     nex14_tbl, ex14_tbl, ex14, ex14list,
     xyzq, half_box, glo2loc_ind);
  cudaCheck(cudaGetLastError());
}

//
// Builds Ewald lookup table
// roff  = distance cut-off
// h     = the distance between interpolation points
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::setup_ewald_force(CT h) {
  h_setup->hinv = ((CT)1.0)/h;

  n_ewald_force = (int)(sqrt(h_setup->roff2)*h_setup->hinv) + 2;

  CT *h_ewald_force = new CT[n_ewald_force];

  for (int i=1;i < n_ewald_force;i++) {
    const CT two_sqrtpi = (CT)1.12837916709551;    // 2/sqrt(pi)
    CT r = i*h;
    CT r2 = r*r;
    h_ewald_force[i] = two_sqrtpi*((CT)h_setup->kappa)*exp(-((CT)h_setup->kappa2)*r2) + 
      erfc(((CT)h_setup->kappa)*r)/r;
  }
  h_ewald_force[0] = h_ewald_force[1];

  allocate<CT>(&ewald_force, n_ewald_force);
  copy_HtoD_sync<CT>(h_ewald_force, ewald_force, n_ewald_force);

  h_setup->ewald_force = ewald_force;

  delete [] h_ewald_force;
}

//
// Sets method for calculating electrostatic force and energy
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::set_elec_model(int elec_model, CT h) {
  this->elec_model = elec_model;
  
  if (elec_model == EWALD_LOOKUP) {
    setup_ewald_force(h);
  }
}

//
// Calculates 1-4 exclusions and interactions
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::calc_14_force(const float4 *xyzq,
					       const bool calc_energy, const bool calc_virial,
					       const int stride, AT *force, cudaStream_t stream) {
  if (use_tex_vdwparam14) {
#ifdef USE_TEXTURE_OBJECTS
    if (!vdwParam14TexObjActive) {
      std::cerr << "CudaPMEDirectForce<AT, CT>::calc_14_force, vdwParam14TexObj must be created" << std::endl;
      exit(1);
    }
    if (!vdwParamTexObjActive) {
      std::cerr << "CudaPMEDirectForce<AT, CT>::calc_14_force, vdwParamTexObj must be created" << std::endl;
      exit(1);
    }
#else
    if (!get_vdwparam14_texref_bound()) {
      std::cerr << "CudaPMEDirectForce<AT, CT>::calc_14_force, vdwparam14_texref must be bound" << std::endl;
      exit(1);
    }
    if (!get_vdwparam_texref_bound()) {
      std::cerr << "CudaPMEDirectForce<AT, CT>::calc_14_force, vdwparam_texref must be bound" << std::endl;
      exit(1);
    }
#endif
  }

  int nthread = 512;
  int nin14block = (nin14list + nthread - 1)/nthread;
  int nex14block = (nex14list + nthread - 1)/nthread;
  int nblock = nin14block + nex14block;
  int shmem_size = 0;
  if (calc_energy) {
    shmem_size = nthread*sizeof(double3); // only need double2 here. Compile can't seem to handle distinction, (in CudaPMEDirectForceBlock.cu), see note by shpot declaration in CudaDirectForce14_util.h
  }

  int vdw_model_loc = calc_vdw ? vdw_model : NONE;
  int elec_model_loc = calc_elec ? elec_model : NONE;
  if (elec_model_loc == NONE && vdw_model_loc == NONE) return;

  calcForce14KernelChoice<AT,CT>(nblock, nthread, shmem_size, stream,
				 vdw_model_loc, elec_model_loc, calc_energy, calc_virial,
				 this->nin14list, this->in14list, this->nex14list, this->ex14list,
				 nin14block,
                                 (int*) NULL, (int*) NULL, (int*) NULL,
                                 (int*) NULL, (int*) NULL, (int*) NULL,
                                 this->vdwtype, this->vdwparam, this->vdwparam14,
#ifdef USE_TEXTURE_OBJECTS
				 this->vdwParamTexObj, this->vdwParam14TexObj,
#endif
				 xyzq, 1.0f, stride, force,
				 energyVirial.getVirialPointer(),
				 energyVirial.getEnergyPointer(strVdw),
				 energyVirial.getEnergyPointer(strElec),
				 energyVirial.getEnergyPointer(strExcl),
                                 (CudaBlock*) NULL);
}

//
// Calculates direct force
//
template <typename AT, typename CT>
void CudaPMEDirectForce<AT, CT>::calc_force(const int whichlist,
                                            const float4 *xyzq,
					    const CudaNeighborListBuild<32>& nlist,
					    const bool calc_energy,
					    const bool calc_virial,
					    const int stride, AT *force, cudaStream_t stream) {

  if (use_tex_vdwparam) {
#ifdef USE_TEXTURE_OBJECTS
    if (!vdwParamTexObjActive) {
      std::cerr << "CudaPMEDirectForce<AT, CT>::calc_force, vdwParamTexObj must be created" << std::endl;
      exit(1);
    }
#else
    if (!get_vdwparam_texref_bound()) {
      std::cerr << "CudaPMEDirectForce<AT, CT>::calc_force, vdwparam_texref must be bound" << std::endl;
      exit(1);
    }
#endif
  }

  if (nlist.get_n_ientry() == 0) return;
  int vdw_model_loc = calc_vdw ? vdw_model : NONE;
  int elec_model_loc = calc_elec ? elec_model : NONE;
  if (elec_model_loc == NONE && vdw_model_loc == NONE) return;

  int nwarp = 2;
  if (get_cuda_arch() < 300) {
    nwarp = 2;
  } else {
    nwarp = 4;
  }
  int nthread = warpsize*nwarp;
  int nblock_tot = (nlist.get_n_ientry()-1)/(nthread/warpsize)+1;

  int shmem_size = 0;
  // (sh_xi, sh_yi, sh_zi, sh_qi, sh_vdwtypei)
  if (get_cuda_arch() < 300)
    shmem_size += (nthread/warpsize)*tilesize*(sizeof(float)*4 + sizeof(int));
  // (sh_fix, sh_fiy, sh_fiz)
  shmem_size += (nthread/warpsize)*warpsize*sizeof(AT)*3;
  // If no texture fetch for vdwparam:
  if (!use_tex_vdwparam) shmem_size += nvdwparam*sizeof(float);

  if (calc_energy) shmem_size = max(shmem_size, (int)(nthread*sizeof(double)*2));
  if (calc_virial) shmem_size = max(shmem_size, (int)(nthread*sizeof(double)*3));

  calcForceKernelChoice<AT,CT>(nblock_tot, nthread, shmem_size, stream,
			       vdw_model_loc, elec_model_loc, calc_energy, calc_virial,
			       nlist, this->vdwparam, this->nvdwparam, this->vdwtype,
#ifdef USE_TEXTURE_OBJECTS
			       this->vdwParamTexObj,
#endif
			       xyzq, stride, force,
			       energyVirial.getVirialPointer(),
			       energyVirial.getEnergyPointer(strVdw),
			       energyVirial.getEnergyPointer(strElec));
}

//
// Explicit instances of CudaPMEDirectForce
//
template class CudaPMEDirectForce<long long int, float>;
#endif //NOCUDAC
