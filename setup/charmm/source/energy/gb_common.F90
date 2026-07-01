module gb_common
  use chm_kinds
  implicit none

  Real(chm_real),save :: P(6), Eps_GB, CutAlph
  real(chm_real),allocatable,dimension(:),save :: &
       r_gb,alph_gb,vol_gb,SigX_GB, SigY_GB, SigZ_GB, T_GB, &
       gb_atm
  integer,save :: igentype
  Logical,save :: QAnalys
!
!  This common block file contains the control parameters
!  for generalized Born calculations using a modified version
!  if the Still GB treatment following:
!  B. Dominy and C.L. Brooks, III, JCC, in preparation.
!  The variables include:
!
!  Parameters for the GB equations
!  
!  P(6)        =   Vector of P parameters from linearized Still GB radii
!                  equation (1-5), P(6) is a scaling value for the vdW radii.
!  
!  EPS_GB      =   Value of dielectric constant for solvent region 
!                  in GB model.
!  
!  CutAlph     =   Maximum value of GB alpha allowed.
!
!  R_GB        =   Heap pointer to NATOM length array containing 
!                  vdW radius, Rmin.
!  
!  ALPH_GB     =   Heap pointer to NATOM length array containing effective 
!                  Born radii for a given configuration.
!  
!  VOL_GB      =   Heap pointer to NATOM length array containing atomic
!                  volume based on vdW Rmin value.
!
!  T_GB        =   Heap pointer to NATOM length array containing atomic
!                  coefficient needed in GB force calculation.
!
!
!  SIGX_GB     =   Heap pointer to NATOM length array containing gradient
!                  term for GB radius wrt molecular geometry - X-direction.
!
!  SIGY_GB     =   Heap pointer to NATOM length array containing gradient
!                  term for GB radius wrt molecular geometry - Y-direction.
!
!  SIGZ_GB     =   Heap pointer to NATOM length array containing gradient
!                  term for GB radius wrt molecular geometry - Z-direction.
!
!  QAnalys     =   Logical flag for inclusion of GB Atom based contributions
!
!  GB_Atm      =   Heap pointer to NATOM length array containing atomic
!                  contributions to the GB energy.
!
end module gb_common

