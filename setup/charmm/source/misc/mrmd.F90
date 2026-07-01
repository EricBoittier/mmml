!======================================================================
!  14-02-2014:developed by Tibor Nagy, Basel, Switzerland (Markus Meuwly)
!
!  14-10-2014:fix by Tibor Nagy, Budapest, Hungary (Markus Meuwly) 
!  1. setting intent(inout) for argument "array" 
!     in "MRMD_ALLOC_..." routines
!  2. max_n12,max_n13,max_n14,max_n15 upper estimates for array sizes
!     were modified
!  3. miscom was modified to call mrmd_init only on the
!     0th node in parallel runs
!  4. test case mrmd_h2so4.inp was modified to run only on the
!     0th node in parallel runs  
!  5. some comments were updated in mrmd.src
!
!  14-04-2016: addition by Oliver Unke, Basel, Switzerland (Markus Meuwly)
!     Added RKHS functionality.
!======================================================================
module MRMD_FCM

#if KEY_MRMD==1 /*mrmd*/

  use chm_kinds
  use memory
  implicit none

!======================================================================
! PRIVATE AND PUBLIC ROUTINES
!======================================================================
  ! set all functions and routines private  
  private
  ! set some routines public 
  public :: MRMD_INIT,EMRMD
   
!======================================================================
! 1. PUBLIC VARIABLES    
!======================================================================
  logical,public,save ::  & ! 
    MRMD_ACTIVE=.false.     ! is MRMD surface loaded ?
!======================================================================
! 2. FOR ALLOCATION ROUTINE    
!======================================================================
  character(len=8),parameter ::   & ! 
    filen='mrmd.src'      ! Filename

!======================================================================
! 3. INPUTS ARGUMENTS FROM RXMD KEYWORDS    
!======================================================================
  integer,private,save ::  &
    rx_paramfile_unit,     & ! INPUT ARGUMENT FOR UPAR: Unit number of MRMD parameter file.
    rx_crossgeom_unit,     & ! INPUT ARGUMENT FOR UGEO: Unit number for writing crossing geometries
    rx_dyna_print_steps,   & ! INPUT ARGUMENT FOR PRDY: print period in steps during dynamics
    rx_nondyna_print_calls   ! INPUT ARGUMENT FOR PRCA: print period in calls during non-dynamics

!======================================================================
! 4. COUNTERS, FLAGS FOR ROUTINE CALLS
!======================================================================
  logical,private,save :: &
    rx_calcgrad,          &  ! calculate gradient or not 
    rx_print_energy
    
  integer,private,save ::  &
    rx_ncall_emrmd,        & ! number of calls to EMRMD routine
    rx_isurf_lowest0         ! momentary lowest surface

!======================================================================
! 5. CROSSING RELATED VARIABLES  
!======================================================================
  ! integers 
  integer,private,save :: &  
    rx_nsurf, &                ! Number of user defined surfaces
    rx_nswch                   ! Number of switching rules

  ! switching
  character(len=10),private,save  :: &
    rx_swch_func    ! string of switching function  

  real(chm_real),private,save :: &
    rx_swch_dE      ! (>0) switching energy range in kcal/mol

  ! gauss*polynomial functions    
  integer,private,save :: &  
    rx_ngapo        ! Number of Gaussian-Polynomial functions for shaping the dividing surface

  integer,parameter,private  :: &
    rx_gapo_order_max=5   ! maximum allowed polynomial order in gaussian-polynomial
    
  integer,allocatable,dimension(:),private,save  :: &  
    rx_gapo_order   ! (1:rx_ngapo) ! polynomial order

  integer,allocatable,dimension(:,:),private,save  :: &
    rx_gapo_surfs   ! (1:rx_ngapo,1:2) index of surfaces whose dividing surface is adjusted

  real(chm_real),allocatable,dimension(:,:),private,save :: &
    rx_gapo_coefs   ! (1:rx_ngapo,-2:rx_gapo_order) parameters of gaussian*polynomials
                    !                 -2: center of the Gaussian   
                    !                 -1: standard deviation of the Gaussian   
                    !                  0,1,2,3...,rx_gapo_order : polynomial coefficients

!======================================================================
! 6. FORCE FIELD PARAMETERISATION
!======================================================================
!======================================================================
! CHARMM Force field parameterisation - external variables
!======================================================================
! BONDS:     IB,JB: (1:nbond) atom1 and atom2 in harmonic bonds
! ANGLES,UREY-BRADLEY:    IT,JT,KT: (1:ntheta) atom123 in harmonic bonds
! DIHEDRALS: IP,JP,KP,LP (1:nphi) atom1234 in dihedrals
! IMPROPERS: IM,JM,KM,LM (1:nimphi) atom1222 in dihedral
!======================================================================


  integer,private,parameter  ::  & ! 
    rx_ninte=10      ! Number of interactions
    ! 1: harm 2: mors 3: angl 4: dihe 5: impr 
    ! 6: elec 7:  vdw 8: gvdw 9: urey 10: rkhs
    
  integer,private,save  ::  & ! 
    rx_natom,     & ! Number of atoms defined in MRMD input     
    rx_nharm,     & ! Number of harmonic bonds defined in MRMD input
    rx_nmors,     & ! Number of Morse bonds defined in MRMD input
    rx_nrkhs,     & ! Number of RKHS bonds defined in MRMD input
    rx_nbond,     & ! Number of all bonds defined in MRMD input = rx_nharm+rx_nmors  
    rx_nangl,     & ! Number of angles (and Urey Bradleys) defined in MRMD input
    rx_ndihe,     & ! Number of proper dihedrals defined in MRMD input   
    rx_nimpr,     & ! Number of improper dihedrals defined in MRMD input 
    rx_ngvdw        ! Number of general-exponent van der Waals defined in MRMD input 

  integer,allocatable,dimension(:),private,save  :: &  ! if negative index=> new term not in CHARMM
    rx_atom,          & ! (1:rx_natom) Corresponding index for atom in CHARMM arrays     
    rx_atom_inv,      & ! (1:natom) 
                        ! Index array of CHARMM atoms in MRMD ATOM list,
                        ! equals -1 for atoms which are not define in MRMD file. 
    rx_bond,          & ! (1:rx_nbond) Corresponding index for bond in CHARMM arrays 
    rx_bond_inv,      & ! (1:nbond) index of CHARMM BOND's in MRMD unified bond array(HARM/MORS) 
    rx_angl,          & ! (1:rx_nangl) Corresponding index for angle in CHARMM arrays
    rx_angl_inv,      & ! (1:nangl) index of CHARMM ANGL's in MRMD ANGL
    rx_impr_inv         ! (1:nimpr) index of CHARMM IMPR's in MRMD IMPR
                        ! same four atoms and
                        ! same first atom and
                        ! same mid-two atoms (2,3 or 3,2) and
                        ! same last atoms and
                        ! same frequency

  logical,allocatable,dimension(:,:),private,save  ::   &
    rx_harm_exist,  & ! (0:rx_nsurf,1:rx_nharm)
                      ! (0         ,1:rx_nharm) 
                      !  Array of flags to indicate if a harmonic bond is present in CHARMM
                      ! (1:rx_nsurf,1:rx_nharm) 
                      !  Array of flags to indicate if a harmonic bond is present on a PES
    rx_mors_exist,  & ! (1:rx_nsurf,1:rx_nmors) 
                      !  Array of flags to indicate if a morse bond is present on a PES
    rx_rkhs_exist,  & ! (1:rx_nsurf,1:rx_nrkhs) 
                      !  Array of flags to indicate if a RKHS bond is present on a PES
    rx_bond_exist,  & ! (0:rx_nsurf,1:rx_nbond)
                      ! (0         ,1:rx_nbond) 
                      !  Array of flags to indicate if a bond is present in CHARMM
                      ! (1:rx_nsurf,1:rx_nbond) 
                      !  Array of flags to indicate if a bond is present on a PES
    rx_angl_exist,  & ! (0:rx_nsurf,1:rx_nangl)
                      ! (0         ,1:rx_nangl) 
                      !  Array of flags to indicate if a angle is present in CHARMM
                      ! (1:rx_nsurf,1:rx_nangl) 
                      ! Array of flags to indicate if a angle is present on a PES
    rx_urey_exist,  & ! (0:rx_nsurf,1:rx_nangl) ! only together with angle terms!!!!!!!
                      ! (0         ,1:rx_nangl) 
                      !  Array of flags to indicate if a Urey-Bradley is present in CHARMM
                      ! (1:rx_nsurf,1:rx_nangl) 
                      ! Array of flags to indicate if a Urey-Bradley is present on a PES
    rx_dihe_exist,  & ! (0:rx_nsurf,1:rx_ndihe)
                      ! (0         ,1:rx_ndihe) 
                      ! Array of flags to indicate if a   proper dihedral is present in CHARMM
                      ! (1:rx_nsurf,1:rx_ndihe) 
                      ! Array of flags to indicate if a   proper dihedral is present on a PES
    rx_impr_exist,  & ! (0:rx_nsurf,1:rx_nimpr)
                      ! (0         ,1:rx_nimpr) 
                      ! Array of flags to indicate if a   improper dihedral is present in CHARMM
                      ! (1:rx_nsurf,1:rx_nimpr) 
                      ! Array of flags to indicate if a   improper dihedral is present on a PES
    rx_gvdw_exist     ! (1:rx_nsurf,1:rx_ngvdw) 
                      ! Array of flags to indicate if a general van der Waals present on a PES
  
  integer,allocatable,dimension(:,:),private,save  :: &
    rx_dihe_inv,      & ! (1:ndihe,0:5) index of CHARMM DIHE's in MRMD DIHE 
                        !          0:5 for 0-5 allowed index increments for higher
                        !              multiplicities 
                        ! same four atoms and 
                        ! same mid atoms(2,3 or 3,2) and
                        ! same edge atoms (1,4 or 4,1) and
                        ! same frequency
    rx_harm_atoms,    & ! (1:rx_nharm,1:2) CHARMM atom number of the 2 atoms in harm in MRMD
    rx_mors_atoms,    & ! (1:rx_nmors,1:2) CHARMM atom number of the 2 atoms in mors in MRMD
    rx_rkhs_atoms,    & ! (1:rx_nrkhs,1:2) CHARMM atom number of the 2 atoms in rkhs in MRMD
    rx_bond_atoms,    & ! (1:rx_nbond,1:2) CHARMM atom number of the 2 atoms in bond in MRMD
    rx_angl_atoms,    & ! (1:rx_nangl,1:3) CHARMM atom number of the 3 atoms in angl in MRMD
    rx_dihe_atoms,    & ! (1:rx_ndihe,1:4) CHARMM atom number of the 4 atoms in dihe in MRMD  
    rx_impr_atoms,    & ! (1:rx_nimpr,1:4) CHARMM atom number of the 4 atoms in impr in MRMD  
    rx_gvdw_atoms,    & ! (1:rx_ngvdw,1:2) CHARMM atom number of the 2 atoms in gvdw in MRMD  
    rx_rkhs_ngrid       ! (1:rx_nrkhs,1:rx_nsurf) number of grid points on each surface

  real(chm_real),allocatable,dimension(:),private,save  ::  &
    rx_levelshift       ! (rx_nsurf) array of level shifts compared

  real(chm_real),allocatable,dimension(:,:),private,save  :: &
    rx_charges,       & ! (1:rx_nsurf,1:rx_natom) 
                        ! Coulomb potential: atomic charge: on each PES [e: atomic charge unit]  
    rx_vdw_eps,       & ! (1:rx_nsurf,1:rx_natom) 
                        ! Lennard-Jones potential: epsilon parameters on each PES [kcal/mol] 
    rx_vdw_rmin_half, & ! (1:rx_nsurf,1:rx_natom) 
                        ! Lennard-Jones potential: half rmin distance on each PES [kcal/mol]  
    rx_vdw_eps2,      & ! (1:rx_nsurf,1:rx_natom) SPECIAL VDW for 1-4 interaction
                        ! Lennard-Jones potential: epsilon parameters on each PES [kcal/mol] 
    rx_vdw_rmin_half2,& ! (1:rx_nsurf,1:rx_natom) SPECIAL VDW for 1-4 interaction
                        ! Lennard-Jones potential: half rmin distance on each PES [kcal/mol]  
    rx_harm_fc_half,  & ! (1:rx_nsurf,1:rx_nharm) 
                        ! Harmonic bond potential: half force const. on each PES [kcal/angstroem^2]
    rx_harm_re,       & ! (1:rx_nsurf,1:rx_natom) 
                        ! Harmonic bond potential: equilibrium bond lengths on each PES [angstroem]  
    rx_mors_De,       & ! (1:rx_nsurf,1:rx_nmors) 
                        ! Morse potential: dissociation energies on each [kcal/mol] 
    rx_mors_re,       & ! (1:rx_nsurf,1:rx_nmors) 
                        ! Morse potential: equilibrium bond lengths on each PES [angstroem]  
    rx_mors_beta,     & ! (1:rx_nsurf,1:rx_nmors) 
                        ! Morse potential: beta parameters on each PES [1/angstroem]
    rx_rkhs_asym,     & ! (1:rx_nrkhs,1:rx_nsurf) 
                        ! RKHS potential: value of the asymptote
    rx_angl_fc_half,  & ! (1:rx_nsurf,1:rx_nangl) 
                        ! Harmonic angle potential: force constants on each PES [kcal/mol/radian^2]
    rx_angl_phie,     & ! (1:rx_nsurf,1:rx_nangl) 
                        ! Harmonic angle potential: equilibrium angle on each PES [radian] 
                        ! (in degree in MRMD)
    rx_urey_fc_half, &  ! (1:rx_nsurf,1:rx_nangl) 
                        ! with angle potential:Urey-Bradley half force const. on each PES
                        ! [kcal/angstroem^2] 
    rx_urey_re,      &  ! (1:rx_nsurf,1:rx_nangl) 
                        ! with harmonic angle potential: Urey-Bradley eq. distance on each PES[A]
    rx_dihe_k,       &  ! (1:rx_nsurf,1:rx_ndihe) 
                        ! Proper dihedral angle potential: 
                        ! force constants on each PES [kcal/mol/radian^2]
    rx_dihe_phi0,   &   ! (1:rx_nsurf,1:rx_ndihe) 
                        ! Proper dihedral angle potential: 
                        ! phase shift on each PES [radian] 
                        ! (in degree in MRMD, then converted after input)
                        ! freq/=0 =>phi0/freq=angle at maximum-energy
                        ! freq=0 => phi0     =angle at minimum-energy 
    rx_dihe_cos0,   &   ! (1:rx_nsurf,1:rx_ndihe) Proper dihedral angle potential: 
                        ! cos(phase shift) on each PES [-]
    rx_dihe_sin0,   &   ! (1:rx_nsurf,1:rx_ndihe) 
                        ! Proper dihedral angle potential: sin(phase shift) on each PES [-] 
    rx_impr_k,      &   ! (1:rx_nsurf,1:rx_nimpr) 
                        ! Improper dihedral angle potential: 
                        ! force constants on each PES [kcal/mol/radian^2]
    rx_impr_phi0,   &   ! (1:rx_nsurf,1:rx_nimpr) 
                        ! Improper dihedral angle potential: phase shift on each PES [radian] 
                        ! (in degree in MRMD)
                        ! freq/=0 =>phi0/freq=angle at maximum-energy
                        ! freq=0 => phi0     =angle at minimum-energ
    rx_impr_cos0,   &   ! (1:rx_nsurf,1:rx_nimpr) Improper dihedral angle potential: 
                        ! cos(phase shift) on each PES [-]
    rx_impr_sin0,   &   ! (1:rx_nsurf,1:rx_nimpr) 
                        ! Improper Dihedral angle potential: sin(phase shift) on each PES [-] 
    rx_gvdw_eps,    &   ! (1:rx_nsurf,1:rx_ngvdw) 
                        ! general Lennard-Jones potential: epsilon parameters on each PES [kcal/mol] 
    rx_gvdw_rmin,   &   ! (1:rx_nsurf,1:rx_ngvdw) 
                        ! general Lennard-Jones potential: rmin distance on each PES [kcal/mol]  
    rx_gvdw_xatt,   &   ! (1:rx_nsurf,1:rx_ngvdw) SPECIAL VDW for 1-4 interaction
                        ! general Lennard-Jones potential:attractive exponent on each PES [kcal/mol]  
    rx_gvdw_xrep        ! (1:rx_nsurf,1:rx_ngvdw) SPECIAL VDW for 1-4 interaction
                        ! general Lennard-Jones potential:repulsive exponent on each PES [kcal/mol]  
                        
  real(chm_real),allocatable,dimension(:,:,:),private,save  :: &
    rx_rkhs_grid,     & ! (1:ngrid,1:rx_nrkhs,1:rx_nsurf)
                        ! RKHS potential: grid points on each PES [angstroem]         
    rx_rkhs_coef        ! (1:ngrid,1:rx_nrkhs,1:rx_nsurf)
                        ! RKHS potential: coefficients for each PES
                        
  integer,allocatable,dimension(:),private,save  :: &
    rx_dihe_mult, &   ! (1:rx_ndihe) Proper dihedral angle potential: 
                      !  multiplicity of the potential [-]
                      !  =0,1,2,3,4,5,6  
    rx_impr_mult      ! (1:rx_ndihe) improper dihedral angle potential: 
                      !  multiplicity of the potential [-]
                      !  =0,1,2,3,4,5,6 for 
                      
!======================================================================
! 7. EXCLUSION AND INCLUSION LISTS
!======================================================================

! BONDED

  ! size of inclusion and exclusion lists  
  ! exclusions =>   CHARMM interaction
  ! inclusions => MRMD interaction
  integer,private,save  ::   &
    rx_exc_nbond,   & ! (max.) number of bonds     to be removed  => same for all PES
    rx_exc_nangl0,  & !  max.  number of angles    to be removed
    rx_exc_ndihe0,  & !  max.  number of   proper dihedrals to be removed
    rx_exc_nimpr0,  & !  max.  number of improper dihedrals to be removed
    rx_inc_nharm0,  & ! max possible 
    rx_inc_nmors0,  & ! max possible 
    rx_inc_nrkhs0,  & ! max possible 
    rx_inc_nbond0,  & !  max.  number of bonds     to be added 
    rx_inc_nangl0,  & !  max.  number of angles    to be added
    rx_inc_ndihe0,  & !  max.  number of proper dihedrals to be added
    rx_inc_nimpr0     !  max.  number of improper dihedrals to be added

  integer,allocatable,dimension(:),private,save  :: &   
    rx_exc_nangl,   & ! (1:rx_nsurf) number of angles    to be removed
    rx_exc_ndihe,   & ! (1:rx_nsurf) number of proper dihedrals to be removed
    rx_exc_nimpr,   & ! (1:rx_nsurf) number of improper dihedrals to be removed
    rx_inc_nharm,   & ! (1:rx_nsurf) Number of harmonic bonds to be added 
    rx_inc_nmors,   & ! (1:rx_nsurf) Number of Morse bonds to be added 
    rx_inc_nrkhs,   & ! (1:rx_nsurf) Number of RKHS bonds to be added 
    rx_inc_nbond,   & ! (1:rx_nsurf) Number of bonds     to be added 
    rx_inc_nangl,   & ! (1:rx_nsurf) Number of angles    to be added
    rx_inc_ndihe,   & ! (1:rx_nsurf) Number of proper dihedrals to be added
    rx_inc_nimpr      ! (1:rx_nsurf) Number of improper dihedrals to be added

  ! inclusion and exclusion lists  
  ! exclusions =>   CHARMM interaction
  ! inclusions => MRMD interaction
  integer,allocatable,dimension(:),private,save  :: &
    rx_exc_bond      ! (1:rx_exc_nbond) List of bonds to be removed from CHARMM energy
                     ! => same for all PES  

  integer,allocatable,dimension(:,:),private,save  :: &
    rx_exc_angl,   & ! (1:rx_nsurf,1:rx_exc_nangl0) angles to be removed from CHARMM 
    rx_exc_impr,   & ! (1:rx_nsurf,1:rx_exc_nimpr0) improper dihedrals to be removed from CHARMM
    rx_inc_harm,   & ! (1:rx_nsurf,1:rx_inc_nharm0) harmonic bonds to be added on the PES
    rx_inc_mors,   & ! (1:rx_nsurf,1:rx_inc_nmors0) Morse bonds to be added on the PES
    rx_inc_rkhs,   & ! (1:rx_nsurf,1:rx_inc_nrkhs0) RKHS bonds to be added on the PES
    rx_inc_bond,   & ! (1:rx_nsurf,1:rx_inc_nbond0) harmonic bonds to be added on the PES
    rx_inc_angl,   & ! (1:rx_nsurf,1:rx_inc_nangl0) angles to be added on the PES
    rx_inc_dihe,   & ! (1:rx_nsurf,1:rx_inc_ndihe0) proper dihedrals to be added on the PES
    rx_inc_impr      ! (1:rx_nsurf,1:rx_inc_nimpr0) improper dihedrals to be added on the PES

  integer,allocatable,dimension(:,:,:),private,save  :: &
    rx_exc_dihe     ! (1:rx_nsurf,1:rx_exc_ndihe0,1:2) proper dihedrals to be removed from CHARMM
                    !                             1: CHARMM index of dihedral
                    !                             2: position of higher periodicities  +ic increase 

  integer,private,save   ::       & 
    rx_natom1234     ! number of all atoms within 1-3 distance of atoms of MRMD bonds
                     ! using all bonds in CHARMM and all bonds in surfaces  

  integer,allocatable,dimension(:),private,save  :: &
    rx_atom1234,   & ! (1:rx_natom1234) CHARMM index of all atoms within (1-4) of atoms listed in 
                     !  in all ATOM/GVDW/HARM/MORS record
    rx_atom1234_inv  ! (1:natom) inverse index (-1 for atoms not in previous list)


! NON-BONDED at various NBXMOD level
  integer,allocatable,dimension(:),private,save  :: &   
    rx_exc_n12, & ! (1:rx_nsurf) Number of 1-2 exclusions
    rx_exc_n13, & ! (1:rx_nsurf) Number of 1-3 exclusions
    rx_exc_n14, & ! (1:rx_nsurf) Number of 1-4 exclusions
    rx_exc_n15, & ! (1:rx_nsurf) Number of 1-(>4) exclusions
    rx_inc_n12, & ! (1:rx_nsurf) Number of 1-2 inclusions
    rx_inc_n13, & ! (1:rx_nsurf) Number of 1-3 inclusions
    rx_inc_n14, & ! (1:rx_nsurf) Number of 1-4 inclusions
    rx_inc_n15    ! (1:rx_nsurf) Number of 1-(>4) inclusions

  integer,allocatable,dimension(:,:,:),target,private,save  :: &
    rx_exc_12,  & ! (0:rx_nsurf,:,2)  List of 1-2 exclusions
                  ! (0         ,:,2)  List of 1-2 exclusions in the CHARMM
                  ! (1:rx_nsurf,:,2)  List of 1-2 exclusions generated for each PES
    rx_exc_13,  & ! (0:rx_nsurf,:,2)  List of 1-3 exclusions
                  ! (0         ,:,2)  List of 1-3 exclusions in the CHARMM
                  ! (1:rx_nsurf,:,2)  List of 1-3 exclusions generated for each PES
    rx_exc_14,  & ! (0:rx_nsurf,:,2)  List of 1-4 exclusions
                  ! (0         ,:,2)  List of 1-4 exclusions in the CHARMM
                  ! (1:rx_nsurf,:,2)  List of 1-4 exclusions generated for each PES 
    rx_exc_15,  & ! (0:rx_nsurf,:,2)  List of 1-(>4) exclusions
                  ! (0         ,:,2)  List of 1-(>4) exclusions in the CHARMM
                  ! (1:rx_nsurf,:,2)  List of 1-(>4) exclusions generated for each PES 
    rx_inc_12,  & ! (0:rx_nsurf,:,3)  List of 1-2 inclusions
                  ! (0         ,:,3)  List of 1-2 inclusions in the CHARMM
                  ! (1:rx_nsurf,:,3)  List of 1-2 inclusions generated for each PES
    rx_inc_13,  & ! (0:rx_nsurf,:,3)  List of 1-3 inclusions
                  ! (0         ,:,3)  List of 1-3 inclusions in the CHARMM
                  ! (1:rx_nsurf,:,3)  List of 1-3 inclusions generated for each PES
    rx_inc_14,  & ! (0:rx_nsurf,:,3)  List of 1-4 inclusions
                  ! (0         ,:,3)  List of 1-4 inclusions in the CHARMM
                  ! (1:rx_nsurf,:,3)  List of 1-4 inclusions generated for each PES 
    rx_inc_15     ! (0:rx_nsurf,:,3)  List of 1-(>4) inclusions
                  ! (0         ,:,3)  List of 1-(>4) inclusions in the CHARMM
                  ! (1:rx_nsurf,:,3)  List of 1-(>4) inclusions generated for each PES 

!===========================================================================
! GENERAL (DE)(RE)ALLOCATION WITHOUT KNOWING THE PREVIOUS SIZE
!===========================================================================
  INTERFACE MRMD_ALLOC
     module procedure MRMD_ALLOC_INTG1
     module procedure MRMD_ALLOC_INTG2
     module procedure MRMD_ALLOC_INTG3
     module procedure MRMD_ALLOC_REAL1
     module procedure MRMD_ALLOC_REAL2
     module procedure MRMD_ALLOC_REAL3 
     module procedure MRMD_ALLOC_LOGI1
     module procedure MRMD_ALLOC_LOGI2
     module procedure MRMD_ALLOC_CH16_1
  END INTERFACE MRMD_ALLOC
!===========================================================================
!===========================================================================
!
!
!
!                       SUBROUTINES
!
!
!
!===========================================================================
!===========================================================================
contains

!===========================================================================
!
! READING ARMD PARAMETRIZATION 
!
!===========================================================================
  subroutine MRMD_INIT
    !     
    !     Interprets the options MRMD command, reads the MRMD 
    !     parameter file into suitable arrays and generates lists  
    !     of exclusions/inclusion   
    !
    use dimens_fcm
    use comand
    use stream
    use string
    use coord
    use code
    use consta
    use psf
    use param
    use block_ltm
    use number
    use cnst_fcm
    use image

    implicit none

    ! FOR ALLOCATION ROUTINE    
    character(len=9),parameter ::   & ! 
       procn='MRMD_INIT'         ! procedure name

    logical changed,found(3),exists(3),error_found,lgvdw

    logical,allocatable,dimension(:) :: &
         lsurf ! flag for surface level shifts

    character(len=4) ::&
      keyword1,keyword2
   
    ! Loop counters, atom number storages, decision flags etc.
    integer ::      &
      isurf,     & ! surface index 
      isurf1,    & ! surface index
      isurf2,    & ! surface index
      isurfx,    & ! surface index 
      igvdw,     &
      i,j,k,l,N,M,  & 
      ib1,ib2,      & 
      n_old,     & ! old value of counters beyond the first call of cross
      max_n12,   & ! max number of 1-2   inclusions/exclusions -> for the size of temporary array
      max_n13,   & ! max number of 1-3   inclusions/exclusions -> for the size of temporary array     
      max_n14,   & ! max number of 1-4   inclusions/exclusions -> for the size of temporary array       
      max_n15,   & ! max number of 1-(>4)inclusions/exclusions -> for the size of temporary array       
      nbond_fix, & ! number of common bonds: all CHARMM bonds which are not in cross 
      nbond_all, & ! all the bonds in CHARMM and all surfaces  
      nbond1234, & ! num bonds within 2-bond dist from atoms involved in bonds listed in MRMD
      ngridrkhsmax ! max number of grid points for RKHS -> for the size of RKHS arrays              

    integer,allocatable,dimension(:) :: &
      atom_pos,     &  ! (1:natom) shortest relative position from ATOM/GVDW/HARM/MORS atoms        
      bond1234        ! (1:nbond1234) index of all bonds connecting atoms within (1-4)
                          ! distance from ATOM/GVDW/HARM/MORS atoms

!     bond1234_inv     ! (1:nbond) inverse index (-1 for bonds not in previous list)
    integer,allocatable,dimension(:,:) :: &
      bond_all       ! (1:nbond_all,1:2) all CHARMM and all cross bonds

    integer,allocatable,dimension(:,:,:) :: &
      atom_pos2, &   ! (1:rx_natom1234,1:rx_natom1234) shortest rel.pos. between 1-4 atoms    
      tmp_exc_12,   &   ! temporary 1-2    exclusion array in the rx_atom1234 list
      tmp_exc_13,   &   ! temporary 1-3    exclusion array in the rx_atom1234 list
      tmp_exc_14,   &   ! temporary 1-4    exclusion array in the rx_atom1234 list
      tmp_exc_15,   &   ! temporary 1-(>4) exclusion array in the rx_atom1234 list
      tmp_inc_12,   &   ! temporary 1-2    inclusion array in the rx_atom1234 list    
      tmp_inc_13,   &   ! temporary 1-3    inclusion array in the rx_atom1234 list  
      tmp_inc_14,   &   ! temporary 1-4    inclusion array in the rx_atom1234 list   
      tmp_inc_15        ! temporary 1-(>4) inclusion array in the rx_atom1234 list   
                                                         
    ! temporary storage for reading parameter file
    character(len=16),allocatable,dimension(:)    :: &
      tempstring  

    real(chm_real)   &
      tempreal0

    integer i1,i2,i3,j1,k1,ii
    
    !temporary variables used for RKHS 
    logical                     :: rkhs_calc_coef, rkhs_file_opened
    real(chm_real), allocatable :: rkhs_kernel_mat(:,:), &
                                   rkhs_kernel_diag(:),  &
                                   rkhs_energy_vec(:)
    integer :: rkhs_grid_unit, rkhs_coef_unit, rkhs_iostat
    
 
!===========================================================================
!
! Process RXMD keywords in CHARMM input
!
!===========================================================================

    !-------------------------------------------------------------------
    ! Reseting MRMD, deallocating array, deactivating cross module
    !-------------------------------------------------------------------
    if (INDXA(ComLyn,ComLen,'RSET')>0) then
        call MRMD_DEALLOC
        MRMD_ACTIVE=.false.
        return
    endif   

    !-------------------------------------------------------------------
    ! activate cross module
    !-------------------------------------------------------------------
    MRMD_ACTIVE = .true.

    !-------------------------------------------------------------------
    ! counters for number of calls to energy routines
    !-------------------------------------------------------------------
    rx_ncall_emrmd =0
    rx_isurf_lowest0=0  ! index of lowest energy surface

    !-------------------------------------------------------------------
    ! Unit for reading parameters - mandatory input
    !-------------------------------------------------------------------
    rx_paramfile_unit = GTRMI(ComLyn,ComLen,'UPAR',-1)
    if (rx_paramfile_unit==-1) call wrndie(-3,'<mrmd.src> mrmd_init', &  
         'No io unit defined for MRMD parameter file input.')
    if (PRNLEV>=5) &
    write(outu,'(A,I8)') &
    'MRMD_INIT>  Unit for reading MRMD parameter file: ',rx_paramfile_unit

    !-------------------------------------------------------------------
    ! Unit for writing crossing geometry - optional input
    !-------------------------------------------------------------------
    rx_crossgeom_unit = GTRMI(ComLyn,ComLen,'UCRG',-1) ! default=-1
    if (PRNLEV>=5) then
        write(outu,'(A)',advance='no') 'MRMD_INIT>  Unit for writing crossing geometries:'
        if (rx_crossgeom_unit>0) then
            write(outu,'(I8)') rx_crossgeom_unit
        else
            write(outu,'(A)')  'not written!'
        endif    
    endif

    !--------------------------------------------------------------------------
    ! Period of writing out energetics during dynamics 
    ! (in integrator steps)
    ! optional
    !--------------------------------------------------------------------------
    rx_dyna_print_steps = GTRMI(ComLyn,ComLen,'PRDY',-1)
    if (PRNLEV>=5) then
        write(outu,'(A)',advance='no') &
       'MRMD_INIT>  Number of integrator steps during dynamics after energetics is printed : '
        if (rx_dyna_print_steps>0) then
            write(outu,'(I8)') rx_dyna_print_steps
        else
            write(outu,'(A)') 'not written!'
        endif    
    endif

    !--------------------------------------------------------------------------
    ! Period of writing out energetics when dynamics is not active 
    ! (in number of EMRMD calls)
    ! optional
    !--------------------------------------------------------------------------
    rx_nondyna_print_calls = GTRMI(ComLyn,ComLen,'PRCA',-1)
    if (PRNLEV>=5) then
        write(outu,'(A)',advance='no') &
        'MRMD_INIT>  Number of not-in-dynamics energy calls after energetics is printed : '
        if (rx_nondyna_print_calls>0) then
            write(outu,'(I8)') rx_nondyna_print_calls
        else
            write(outu,'(A)' ) 'not written!'
        endif    
    endif
 
!===========================================================================
!
! READ AND INTERPRET MRMD PARAMETER FILE
!
!===========================================================================

!===========================================================================
!
!       READING NUMBER OF SURFACES 
!
!===========================================================================
    ! keyword SURF and number of surfaces
    if (PRNLEV>=5) write(outu,'(A)') 'MRMD_INIT> Reading SURF block...'
    read(rx_paramfile_unit,*) keyword1,rx_nsurf
    if (MRMD_UPPERCASE(keyword1)/='SURF') &
    call wrndie(-3,'<mrmd.src> mrmd_init', 'Missing SURF block in input')     

    ! Error testing
    if (rx_nsurf<1) &
    call wrndie(-3,'<mrmd.src> mrmd_init','At least one surface is needed !')
    if (rx_nsurf>99) &
    call wrndie(-3,'<mrmd.src> mrmd_init','At most 99 surfaces is allowed !')

!===========================================================================
!
!   READING AND PROCESSING LEVEL SHIFTS
!
!===========================================================================
    ! (re)(de)allocation (if size>0) 
    call MRMD_ALLOC(filen,procn,'lsurf',lsurf,1,rx_nsurf)    
    lsurf=.false.
    call MRMD_ALLOC(filen,procn,'rx_levelshift',rx_levelshift,1,rx_nsurf)    
    rx_levelshift=0._chm_real

    ! read surface shifts 
    do i=1, rx_nsurf
      read(rx_paramfile_unit,*) isurf,tempreal0

      ! range check
      if (isurf<1.or.rx_nsurf<isurf) &
      call wrndie(-3,'<mrmd.src> mrmd_init','Surface index is out of range (1<=isurf<=nsurf)')
      
      ! duplicate check  
      if (lsurf(isurf)) &      ! from here isurf1=1
      call wrndie(-3,'<mrmd.src> mrmd_init','Duplicate definition for level shifts!')

      lsurf(isurf)         =.true.
      rx_levelshift(isurf) =tempreal0
    enddo
    
    ! deallocation
    call MRMD_ALLOC(filen,procn,'lsurf',lsurf,1,0)    

!===========================================================================
!
!       READING SWITCHING PROTOCOL 
!
!===========================================================================
    ! keyword SWCH
    if (PRNLEV>=5) write(outu,'(A)') 'MRMD_INIT> Reading SWCH block...'
    read(rx_paramfile_unit,*) keyword1 !,rx_nswch 
    if (MRMD_UPPERCASE(keyword1)/='SWCH') &
    call wrndie(-3,'<mrmd.src> mrmd_init', 'Missing SWCH block in input')     

    ! set number of switching rules
    rx_nswch=1    
    
    ! read switching function and width of switching E range
    read(rx_paramfile_unit,*) rx_swch_func,rx_swch_dE

    ! setting switching function into uppercase
    rx_swch_func=MRMD_UPPERCASE(rx_swch_func)

    ! is it available switching function
    if (trim(rx_swch_func)/='JOHNSON7'.and.trim(rx_swch_func)/='EXPDECAY') then
    if (WRNLEV>=5) write(outu,'(A,A)') 'MRMD_INIT> Switching function:',trim(rx_swch_func)
    call wrndie(-3,'<mrmd.src> mrmd_init','Not allowed switching function!')
    endif
    
    ! switching delta must be positive
    if (rx_swch_dE<=0._chm_real) then
    if (WRNLEV>=5) write(outu,'(A,x,F12.4)') 'MRMD_INIT> Switching delta:',rx_swch_dE
    call wrndie(-3,'<mrmd.src> mrmd_init','Switching delta has to be positive!')
    endif
    
!===========================================================================
!
!       READING NUMBER OF GAUSS-POLYNOMIAL BARRIER SHAPING FUNCTIONS 
!
!===========================================================================
    ! keyword GAPO and number of gauss-polynomials
    if (PRNLEV>=5) write(outu,'(A)') 'MRMD_INIT> Reading SHAP GAPO block...'
    read(rx_paramfile_unit,*) keyword1,keyword2, rx_ngapo    
    if (MRMD_UPPERCASE(keyword1)/='SHAP'.or.MRMD_UPPERCASE(keyword2)/='GAPO') &
    call wrndie(-3,'<mrmd.src> mrmd_init', 'Missing SHAP GAPO block in input')     
    if (rx_ngapo<0) &
    call wrndie(-3,'<mrmd.src> mrmd_init','Number of Gauss-polynomials cannot be negative !')

    ! (re)(de)allocation (if size>0)
    call MRMD_ALLOC(filen,procn,'rx_gapo_surfs',rx_gapo_surfs,1,rx_ngapo,1,2)    
    call MRMD_ALLOC(filen,procn,'rx_gapo_order',rx_gapo_order,1,rx_ngapo)    
    call MRMD_ALLOC(filen,procn,'rx_gapo_coefs',rx_gapo_coefs,1,rx_ngapo,-2,rx_gapo_order_max)    

    if (rx_ngapo>0) then

        ! setting all coefficients to zero as not all read
        rx_gapo_coefs=0._chm_real
        
        ! read surface shifts 
        do i=1, rx_ngapo
            ! read GAPO parameters
        read(rx_paramfile_unit,*) rx_gapo_surfs(i,1:2),rx_gapo_order(i),&
            rx_gapo_coefs(i,-2:max(0,min(rx_gapo_order_max,rx_gapo_order(i))))

            ! check range of index of surfaces
            if (any(rx_gapo_surfs(i,1:2)<0.or.rx_nsurf<rx_gapo_surfs(i,1:2))) then
                if (WRNLEV>=5) then
                    write(outu,'(A,I3,3x,A,2(x,I2))') &
                    'MRMD_INIT> GAPO:',i,'between surfaces:',rx_gapo_surfs(i,1:2)
                    write(outu,'(A,I1,A,I2)') &
                    'MRMD_INIT> Surface indices have to be in the range of ',1,'-',rx_nsurf
                endif    
                call wrndie(-3,'<mrmd.src> mrmd_init','Surface index is out of range!')
            endif

            ! check whether the indices of the two surfaces are different
            if (rx_gapo_surfs(i,1)==rx_gapo_surfs(i,2)) then
                if (WRNLEV>=5) write(outu,'(A,I3,3x,A,2(x,I2))') &
                'MRMD_INIT> GAPO:',i,'between surfaces:',rx_gapo_surfs(i,1:2)
                call wrndie(-3,'<mrmd.src> mrmd_init','Surface indices have to be different!')
            endif
                
            ! check number of coeffs
            if (rx_gapo_order(i)<0.or.rx_gapo_order_max<rx_gapo_order(i)) then
                if (WRNLEV>=5) then
                    write(outu,'(A,I3,3x,A,2(x,I2))') &
                    'MRMD_INIT> GAPO:',i,'between surfaces:',rx_gapo_surfs(i,1:2)
                    write(outu,'(A,I1)') &
                    'MRMD_INIT> Its polynomial order is:',rx_gapo_order(i) 
                    write(outu,'(A,I1,A,I1)') &
                    'MRMD_INIT> It has to be in the range of ',0,'-',rx_gapo_order_max
                endif    
                call wrndie(-3,'<mrmd.src> mrmd_init','Polynomial order is out of range!')
            endif

            ! check the sign of the standard deviation parameter of the Gaussian
            if (rx_gapo_coefs(i,-1)<=0._chm_real) then
                if (WRNLEV>=5) then
                    write(outu,'(A,I3,3x,A,2(x,I2))') &
                    'MRMD_INIT> GAPO:',i,'between surfaces:',rx_gapo_surfs(i,1:2)
                    write(outu,'(A,G12.5)') &
                    'MRMD_INIT> Standard deviation parameter of the Gaussian is:'&
                    ,rx_gapo_coefs(i,-1) 
                    write(outu,'(A)') &
                    'MRMD_INIT> It has to be positive.'
                endif    
                call wrndie(-3,'<mrmd.src> mrmd_init',&
                'Non-positive standard deviation of the Gaussian function in GAPO!')
            endif
        enddo
    endif

!===========================================================================
!
!       READING ATOM PARAMETERS FOR MRMD NONBONDED INTERACTION 
!
!===========================================================================
    ! read new value 
    if (PRNLEV>=5) write(outu,'(A)') 'MRMD_INIT> Reading NBON ATOM block...'
    read(rx_paramfile_unit,*) keyword1,keyword2,rx_natom
    if (MRMD_UPPERCASE(keyword1)/='NBON'.or.MRMD_UPPERCASE(keyword2)/='ATOM') &    
    call wrndie(-3,'<mrmd.src> mrmd_init', 'Missing NBON ATOM block in input')     
    if (rx_natom<0) &
    call wrndie(-3,'<mrmd.src> mrmd_init','Number of atoms cannot be negative !')

    ! (re)(de)allocation (if size>0) 
    call MRMD_ALLOC(filen,procn,'rx_atom'    ,rx_atom    ,1,rx_natom)    
    call MRMD_ALLOC(filen,procn,'rx_atom_inv',rx_atom_inv,1,natom)    
    if (natom>0) rx_atom_inv(1:natom)=-1
    call MRMD_ALLOC(filen,procn,'rx_charges'       ,rx_charges       ,1,rx_nsurf,1,rx_natom)    
    call MRMD_ALLOC(filen,procn,'rx_vdw_eps'       ,rx_vdw_eps       ,1,rx_nsurf,1,rx_natom)    
    call MRMD_ALLOC(filen,procn,'rx_vdw_eps2'      ,rx_vdw_eps2      ,1,rx_nsurf,1,rx_natom)    
    call MRMD_ALLOC(filen,procn,'rx_vdw_rmin_half' ,rx_vdw_rmin_half ,1,rx_nsurf,1,rx_natom)    
    call MRMD_ALLOC(filen,procn,'rx_vdw_rmin_half2',rx_vdw_rmin_half2,1,rx_nsurf,1,rx_natom)    

    ! read data
    if (rx_natom>0) then

       ! read non-bonded parameters for atoms
       call MRMD_ALLOC(filen,procn,'tempstring',tempstring,1,5*rx_nsurf)
       do i = 1,rx_natom 
         read(rx_paramfile_unit,*) rx_atom(i),tempstring(1:5*rx_nsurf)

         ! error checking for out of range definitions    
         if (rx_atom(i)<1.or.natom<rx_atom(i)) &
         call wrndie(-3,'<mrmd.src> mrmd_init','ATOM atom has to be in range: 1<=iatom<=natom') 

         ! look for duplicate definitions
         if (rx_atom_inv(rx_atom(i))/=-1) &
         call wrndie(-3,'<mrmd.src> mrmd_init','Duplicate ATOM definition !') 

         ! processing parameters
         do j=1,rx_nsurf
            read(tempstring(5*j-4),*) rx_charges      (j,i)
            read(tempstring(5*j-3),*) rx_vdw_eps      (j,i)
            read(tempstring(5*j-2),*) rx_vdw_rmin_half(j,i)

            ! range check
            if (rx_vdw_eps(j,i)<0) then
                 if (WRNLEV>=5) then
                    write(outu,'(A,I3,3x,A,I2)') &
                    'MRMD_INIT> ATOM definition:',i,'on surface:',j
                    write(outu,'(A,x,G12.5)') &
                    'MRMD_INIT> - has negative a Lennard-Jones epsilon:',rx_vdw_eps(j,i)
                 endif
                 call wrndie(-3,'<mrmd.src> mrmd_init','Negative Lennard-Jones epsilon.') 
            endif
            if (rx_vdw_eps(j,i)>0) then
            if (rx_vdw_rmin_half(j,i)<=0) then
                 if (WRNLEV>=5) then
                    write(outu,'(A,I3,3x,A,I2)') &
                    'MRMD_INIT> ATOM definition:',i,'on surface:',j
                    write(outu,'(A,x,G12.5)') &
                    'MRMD_INIT> - has a non-positive Lennard-Jones rmin_half:',rx_vdw_rmin_half(j,i)
                 endif
                 call wrndie(-3,'<mrmd.src> mrmd_init','Non-positive Lennard-Jones rmin_half.') 
            endif
            endif

            ! second Lennard-Jones epsilon for 1-4 (special) vdw interaction
            if (tempstring(5*j-1)(1:1)=='x'.or.tempstring(5*j-1)(1:1)=='X') then
                ! if it is not given, then copy from first 
                rx_vdw_eps2(j,i)=rx_vdw_eps(j,i)        
            else
                read(tempstring(5*j-1),*) rx_vdw_eps2(j,i)
                if (rx_vdw_eps2(j,i)<0) then
                     if (WRNLEV>=5) then
                        write(outu,'(A,I3,3x,A,I2)') &
                        'MRMD_INIT> ATOM definition:',i,'on surface:',j
                        write(outu,'(A,x,G12.5)') &
                        'MRMD_INIT> - has a negative special(1,4) Lennard-Jones epsilon:',&
                        rx_vdw_eps2(j,i)
                     endif
                     call wrndie(-3,'<mrmd.src> mrmd_init', &
                     'Negative special Lennard-Jones epsilon !') 
                endif
            endif  

            ! second Lennard-Jones half vdw radius for 1-4 (special) vdw interaction
            if (tempstring(5*j)(1:1)=='x'.or.tempstring(5*j)(1:1)=='X') then
                ! if it is not given, then copy from first 
                rx_vdw_rmin_half2(j,i)=rx_vdw_rmin_half(j,i)        
            else
                read(tempstring(5*j),*) rx_vdw_rmin_half2(j,i)
                if (rx_vdw_eps2(j,i)>0) then
                if (rx_vdw_rmin_half2(j,i)<=0) then
                     if (WRNLEV>=5) then
                        write(outu,'(A,I3,3x,A,I2)') &
                        'MRMD_INIT> ATOM definition:',i,'on surface:',j
                        write(outu,'(A,x,G12.5)') &
                        'MRMD_INIT> - has a non-positive special(1,4) Lennard-Jones rmin_half:', &
                        rx_vdw_rmin_half2(j,i)
                     endif   
                     call wrndie(-3,'<mrmd.src> mrmd_init', &
                     'Non-positive special Lennard-Jones rmin_half.') 
                endif
                endif
            endif  
         enddo  

         ! inverse indexing
         rx_atom_inv(rx_atom(i))=i
         
       enddo

       ! deallocate
       call MRMD_ALLOC(filen,procn,'tempstring',tempstring,1,0)
    endif
    
!===========================================================================
! 
!  READING PARAMETERS FOR GENERAL VAN DER WAALS INTERACTION
!  V(r)=eps/(xrep/xatt-1)*[(rmin/r)**b-(rmin/r)**a] 
! 
!===========================================================================

    ! read value 
    if (PRNLEV>=5) write(outu,'(A)') 'MRMD_INIT> Reading NBON GVDW block...'
    read(rx_paramfile_unit,*) keyword1,keyword2,rx_ngvdw
    if (MRMD_UPPERCASE(keyword1)/='NBON'.or.MRMD_UPPERCASE(keyword2)/='GVDW') &    
    call wrndie(-3,'<mrmd.src> mrmd_init', 'Missing NBON GVDW block in input')     
    if (rx_ngvdw<0) &
    call wrndie(-3,'<mrmd.src> mrmd_init','Number of general-exponent vdW''s cannot be negative !')
                
    ! (re)(de)allocation (if size>0) 
    call MRMD_ALLOC(filen,procn,'rx_gvdw_exist',rx_gvdw_exist,1,rx_nsurf,1,rx_ngvdw)
    call MRMD_ALLOC(filen,procn,'rx_gvdw_atoms',rx_gvdw_atoms,1,rx_ngvdw,1,2)
    call MRMD_ALLOC(filen,procn,'rx_gvdw_eps'  ,rx_gvdw_eps  ,1,rx_nsurf,1,rx_ngvdw)
    call MRMD_ALLOC(filen,procn,'rx_gvdw_rmin' ,rx_gvdw_rmin ,1,rx_nsurf,1,rx_ngvdw)
    call MRMD_ALLOC(filen,procn,'rx_gvdw_xatt' ,rx_gvdw_xatt ,1,rx_nsurf,1,rx_ngvdw)
    call MRMD_ALLOC(filen,procn,'rx_gvdw_xrep' ,rx_gvdw_xrep ,1,rx_nsurf,1,rx_ngvdw)

    ! read data
    if (rx_ngvdw>0) then
        
        call MRMD_ALLOC(filen,procn,'tempstring',tempstring,1,4*rx_nsurf)
        ! read each improper dihedral
        do i=1,rx_ngvdw
             read(rx_paramfile_unit,*) rx_gvdw_atoms(i,1:2),tempstring(1:4*rx_nsurf)

            ! error checking for out of range definitions    
            if (any(rx_gvdw_atoms(i,1:2)<1.or.natom<rx_gvdw_atoms(i,1:2))) &
            call wrndie(-3,'<mrmd.src> mrmd_init','GVDW atom has to be in range: 1<=iatom<=natom') 

            ! error checking for different atoms
           if (rx_gvdw_atoms(i,1)==rx_gvdw_atoms(i,2))     &
           call wrndie(-3,'<mrmd.src> mrmd_init','GVDW atoms have to be different') 

            ! store values  
            do j=1,rx_nsurf
                ! new notation for deletion => if force constant string starts with x or X
                if (tempstring(4*j-3)(1:1)=='x'.or.tempstring(4*j-3)(1:1)=='X') then
                    rx_gvdw_eps (j,i)=-9999._chm_real
                    rx_gvdw_rmin(j,i)=-9999._chm_real
                    rx_gvdw_xatt(j,i)=-9999._chm_real
                    rx_gvdw_xrep(j,i)=-9999._chm_real
                    rx_gvdw_exist(j,i)=.false.
                else 
                    read(tempstring(4*j-3),*) rx_gvdw_eps (j,i)
                    read(tempstring(4*j-2),*) rx_gvdw_rmin(j,i)
                    read(tempstring(4*j-1),*) rx_gvdw_xatt(j,i)
                    read(tempstring(4*j  ),*) rx_gvdw_xrep(j,i)
                    rx_gvdw_exist(j,i)=.true.

                    ! eps sign check
                    if (rx_gvdw_eps(j,i)<0)  then
                        if (WRNLEV>=5) then
                            write(outu,'(A,x,I3,3x,A,I2)') &
                            'MRMD_INIT> GVDW',i,'on surface',j
                            write(outu,'(A,2(x,I6))') &
                            'MRMD_INIT> - between atoms',rx_gvdw_atoms(i,1:2)
                            write(outu,'(A,x,G12.5)') &
                            'MRMD_INIT> - has epsilon :',rx_gvdw_eps(j,i)
                        endif       
                        call wrndie(-3,'<mrmd.src> mrmd_init','GVDW epsilon must be non-negative') 
                    endif

                    ! rmin sign check
                    if (rx_gvdw_eps (j,i) >0)  then
                    if (rx_gvdw_rmin(j,i)<=0)  then
                        if (WRNLEV>=5) then
                            write(outu,'(A,x,I3,3x,A,I2)') &
                            'MRMD_INIT> GVDW',i,'on surface',j
                            write(outu,'(A,2(x,I6))') &
                            'MRMD_INIT> - between atoms',rx_gvdw_atoms(i,1:2)
                            write(outu,'(A,x,G12.5)') &
                            'MRMD_INIT> - has rmin :',rx_gvdw_rmin(j,i)   
                        endif
                        call wrndie(-3,'<mrmd.src> mrmd_init','GVDW rmin must be positive') 
                    endif
                    ! exponent check
                    if (rx_gvdw_xatt(j,i)<=0.or.rx_gvdw_xrep(j,i)<=rx_gvdw_xatt(j,i))  then
                        if (WRNLEV>=5) then
                            write(outu,'(A,x,I3,3x,A,I2)') &
                            'MRMD_INIT> GVDW',i,'on surface',j
                            write(outu,'(A,2(x,I6))') &
                            'MRMD_INIT> - between atoms',rx_gvdw_atoms(i,1:2)
                            write(outu,'(A,x,G12.5)') &
                            'MRMD_INIT> - has an attractive exponent :', rx_gvdw_xatt(j,i)   
                            write(outu,'(A,x,G12.5)') &
                            'MRMD_INIT> - has a repulsive  exponent :',rx_gvdw_xrep(j,i)
                        endif
                        call wrndie(-3,'<mrmd.src> mrmd_init',&
                        'GVDW exponents should fulfill: 0<=exp_att<exp_rep') 
                    endif
                    endif
                endif
            enddo

            ! duplicate check
            do k=1,i-1
                !------------------------------------------------------
                ! HAVING SAME ATOMS
                ! ab,ba
                !------------------------------------------------------
                if (all(rx_gvdw_atoms(i,1:2)==rx_gvdw_atoms(k,(/1,2/))).or.  &
                    all(rx_gvdw_atoms(i,1:2)==rx_gvdw_atoms(k,(/2,1/)))) then
                    if (WRNLEV>=5) then
                        write(outu,'(A,x,I3,x,A,x,I3)') &
                        'MRMD_INIT> GVDW',i,'and',k
                        write(outu,'(A,2(x,I6))') &
                        'MRMD_INIT> have the same set of atoms:',rx_gvdw_atoms(i,1:2)
                    endif    
                    call wrndie(-3,'<mrmd.src> mrmd_init','Duplicate GVDW for same set of atoms') 
                endif
            enddo    
        enddo
       call MRMD_ALLOC(filen,procn,'tempstring',tempstring,1,0)
    endif

!===========================================================================
!
!     READING PARAMETERS FOR MRMD HARMONIC BONDS  
!
!===========================================================================
    ! read new value 
    if (PRNLEV>=5) write(outu,'(A)') 'MRMD_INIT> Reading BOND HARM block...'
    read(rx_paramfile_unit,*) keyword1,keyword2,rx_nharm
    if (MRMD_UPPERCASE(keyword1)/='BOND'.or.MRMD_UPPERCASE(keyword2)/='HARM') &    
    call wrndie(-3,'<mrmd.src> mrmd_init', 'Missing BOND HARM block in input')     
    if (rx_nharm<0) &
    call wrndie(-3,'<mrmd.src> mrmd_init','Number of harmonic bonds cannot be negative !')

    ! (re)(de)allocation of arrays
    call MRMD_ALLOC(filen,procn,'rx_harm_exist'  ,rx_harm_exist  ,1,rx_nsurf,1,rx_nharm)
    call MRMD_ALLOC(filen,procn,'rx_harm_atoms'  ,rx_harm_atoms  ,1,rx_nharm,1,2)
    call MRMD_ALLOC(filen,procn,'rx_harm_fc_half',rx_harm_fc_half,1,rx_nsurf,1,rx_nharm)
    call MRMD_ALLOC(filen,procn,'rx_harm_re'     ,rx_harm_re     ,1,rx_nsurf,1,rx_nharm)

    ! read data
    if (rx_nharm>0) then
        ! allocation 
        call MRMD_ALLOC(filen,procn,'tempstring',tempstring,1,2*rx_nsurf)

        !  Fill BOND parameter arrays of every atom pair in a surface 
        do i = 1,rx_nharm
           read(rx_paramfile_unit,*) rx_harm_atoms(i,1:2),tempstring(1:2*rx_nsurf)

            ! error checking for out of range definitions    
            if (any(rx_harm_atoms(i,1:2)<1.or.natom<rx_harm_atoms(i,1:2))) &
            call wrndie(-3,'<mrmd.src> mrmd_init','BOND atom has to be in range: 1<=iatom<=natom') 

            ! error checking for being different
            if (rx_harm_atoms(i,1)==rx_harm_atoms(i,2))     &
            call wrndie(-3,'<mrmd.src> mrmd_init','BOND atoms have to be different') 

            ! order atom index
            if (rx_harm_atoms(i,1)>rx_harm_atoms(i,2)) &
                            rx_harm_atoms(i,1:2)=rx_harm_atoms(i,(/2,1/))

            ! look for duplicate definitions
            do j=1,i-1
              if (all(rx_harm_atoms(i,1:2)==rx_harm_atoms(j,1:2))) &
              call wrndie(-3,'<mrmd.src> mrmd_init','Duplicate BOND definition') 
            enddo

            ! process parameters
            do j=1,rx_nsurf
                if (tempstring(2*j-1)(1:1)=='x'.or.tempstring(2*j-1)(1:1)=='X') then
                    rx_harm_fc_half(j,i)= -9999._chm_real
                    rx_harm_re     (j,i)= -9999._chm_real
                    rx_harm_exist  (j,i)=.false.
                else
                    read(tempstring(2*j-1),*) rx_harm_fc_half(j,i)
                    read(tempstring(2*j  ),*) rx_harm_re     (j,i)
                    rx_harm_exist(j,i)  =.true.

                    ! range check
                    if (rx_harm_fc_half(j,i)<0) then
                         if (WRNLEV>=5) then
                            write(outu,'(2(A,I3,x),3x,A,2(x,I6))') &
                            'MRMD_INIT> HARM:',i,'on surface:',j,&
                            'between atoms',rx_harm_atoms(i,1:2)
                            write(outu,'(A,x,G12.5)') &
                            'MRMD_INIT> - has negative half force constant:',rx_harm_fc_half(j,i)
                         endif
                         call wrndie(-3,'<mrmd.src> mrmd_init', &
                         'Negative force constant for harmonic bond.') 
                    endif
                    if (rx_harm_fc_half(j,i) >0) then
                    if (rx_harm_re     (j,i)<=0) then
                         if (WRNLEV>=5) then
                            write(outu,'(2(A,I3,x),3x,A,2(x,I6))') &
                            'MRMD_INIT> HARM:',i,'on surface:',j,&
                            'between atoms',rx_harm_atoms(i,1:2)
                            write(outu,'(A,x,G12.5)') &
                            'MRMD_INIT> - has non-positive equilibrium distance :',rx_harm_re(j,i)
                         endif
                         call wrndie(-3,'<mrmd.src> mrmd_init', &
                         'Non-positive equilibrium distance for harmonic bond.') 
                    endif
                    endif
                endif
            enddo
        enddo
        call MRMD_ALLOC(filen,procn,'tempstring',tempstring,1,0)
    endif

!===========================================================================
!
!   READING PARAMETERS FOR MRMD MORSE BONDS  
!
!===========================================================================
    ! read new value 
    if (PRNLEV>=5) write(outu,'(A)') 'MRMD_INIT> Reading BOND MORS block...'
    read(rx_paramfile_unit,*) keyword1,keyword2,rx_nmors
    if (MRMD_UPPERCASE(keyword1)/='BOND'.or.MRMD_UPPERCASE(keyword2)/='MORS') &    
    call wrndie(-3,'<mrmd.src> mrmd_init', 'Missing BOND MORS block in input')     
    if (rx_nmors<0) &
    call wrndie(-3,'<mrmd.src> mrmd_init','Number of Morse bonds cannot be negative !')

    ! (re)(de)allocation (if size>0)
    call MRMD_ALLOC(filen,procn,'rx_mors_exist',rx_mors_exist,1,rx_nsurf,1,rx_nmors)
    call MRMD_ALLOC(filen,procn,'rx_mors_atoms',rx_mors_atoms,1,rx_nmors,1,2)
    call MRMD_ALLOC(filen,procn,'rx_mors_De'   ,rx_mors_De   ,1,rx_nsurf,1,rx_nmors)
    call MRMD_ALLOC(filen,procn,'rx_mors_re'   ,rx_mors_re   ,1,rx_nsurf,1,rx_nmors)
    call MRMD_ALLOC(filen,procn,'rx_mors_beta' ,rx_mors_beta ,1,rx_nsurf,1,rx_nmors)

    ! read data
    if (rx_nmors>0) then

      ! allocation
      call MRMD_ALLOC(filen,procn,'tempstring',tempstring,1,3*rx_nsurf)

      ! read Morse potential parameters
      do i = 1,rx_nmors
        read(rx_paramfile_unit,*) rx_mors_atoms(i,1:2),tempstring(1:3*rx_nsurf)

        ! error checking for out of range definitions    
        if (any(rx_mors_atoms(i,1:2)<1.or.natom<rx_mors_atoms(i,1:2))) &
        call wrndie(-3,'<mrmd.src> mrmd_init','MORS atom has to be in range: 1<=iatom<=natom')

        ! error checking for different atoms
        if (rx_mors_atoms(i,1)==rx_mors_atoms(i,2)) &
        call wrndie(-3,'<mrmd.src> mrmd_init', 'MORS atoms have to be different') 

        ! order atom index
        if (rx_mors_atoms(i,1)>rx_mors_atoms(i,2)) rx_mors_atoms(i,1:2)=rx_mors_atoms(i,(/2,1/))

        ! look for duplicate definitions
        do j=1,i-1
          if (all(rx_mors_atoms(i,1:2)==rx_mors_atoms(j,1:2))) &
            call wrndie(-3,'<mrmd.src> mrmd_init','Duplicate MORS definition') 
        enddo

        ! look for same bond definition in BOND     
        do j=1,rx_nharm
          if (all(rx_mors_atoms(i,1:2)==rx_harm_atoms(j,1:2))) &
            call wrndie(-3,'<mrmd.src> mrmd_init',&
            'Bond in MORS block is already defined as bond in BOND block') 
        enddo

        ! transfer values
        do j=1,rx_nsurf
            if (tempstring(3*j-2)(1:1)=='x'.or.tempstring(3*j-2)(1:1)=='X') then
                rx_mors_De   (j,i)= -9999._chm_real
                rx_mors_re   (j,i)= -9999._chm_real
                rx_mors_beta (j,i)= -9999._chm_real
                rx_mors_exist(j,i)=.false.
            else 
                read(tempstring(3*j-2),*) rx_mors_De  (j,i)
                read(tempstring(3*j-1),*) rx_mors_re  (j,i)
                read(tempstring(3*j  ),*) rx_mors_beta(j,i) 
                rx_mors_exist(j,i)=.true.

                ! range check
                if (rx_mors_De(j,i)<0) then
                     if (WRNLEV>=5) then
                        write(outu,'(2(A,I3,x),3x,A,2(x,I6))') &
                        'MRMD_INIT> MORS:',i,'on surface:',j,&
                        'between atoms',rx_mors_atoms(i,1:2)
                        write(outu,'(A,x,G12.5)') &
                        'MRMD_INIT> - has negative well-depth :',rx_mors_De(j,i)
                     endif
                     call wrndie(-3,'<mrmd.src> mrmd_init', &
                     'Negative well-depth of Morse potential.') 
                endif

                if (rx_mors_De  (j,i) >0) then
                if (rx_mors_beta(j,i)<=0) then
                    if (WRNLEV>=5) then
                        write(outu,'(2(A,I3,x),3x,A,2(x,I6))') &
                        'MRMD_INIT> MORS:',i,'on surface:',j,&
                        'between atoms',rx_mors_atoms(i,1:2)
                        write(outu,'(A,x,G12.5)') &
                        'MRMD_INIT> - has non-positive beta parameter :',rx_mors_beta(j,i)
                    endif
                    call wrndie(-3,'<mrmd.src> mrmd_init', &
                    'Non-positive beta parameter of Morse potential.') 
                endif

                if (rx_mors_re  (j,i)<=0) then
                     if (WRNLEV>=5) then
                        write(outu,'(2(A,I3,x),3x,A,2(x,I6))') &
                        'MRMD_INIT> MORS:',i,'on surface:',j,  &
                        'between atoms',rx_mors_atoms(i,1:2)
                        write(outu,'(A,x,G12.5)') &
                        'MRMD_INIT> - has non-positive equilibrium distance :',rx_mors_re(j,i)
                     endif
                     call wrndie(-3,'<mrmd.src> mrmd_init', &
                     'Non-positive equilibrium distance of Morse potential.') 
                endif
                endif
            endif
        enddo
      enddo
      call MRMD_ALLOC(filen,procn,'tempstring',tempstring,1,0)
    endif
    
!===========================================================================
!
!   READING PARAMETERS FOR MRMD RKHS BONDS  
!
!===========================================================================
    ! read new value 
    if (PRNLEV>=5) write(outu,'(A)') 'MRMD_INIT> Reading BOND RKHS block...'
    read(rx_paramfile_unit,*,iostat=rkhs_iostat) keyword1,keyword2,rx_nrkhs
    ! this is handled differently than HARM and MORS bonds for backwards compatibility
    if (MRMD_UPPERCASE(keyword1)/='BOND'.or.MRMD_UPPERCASE(keyword2)/='RKHS') then    
        call wrndie(-5,'<mrmd.src> mrmd_init', 'Missing BOND RKHS block in input')
        backspace(rx_paramfile_unit) !rewind unit to not cause errors
        rx_nrkhs = 0
    end if
    if (rx_nrkhs<0) &
    call wrndie(-3,'<mrmd.src> mrmd_init','Number of RKHS bonds cannot be negative !')

    ! (re)(de)allocation (if size>0)
    call MRMD_ALLOC(filen,procn,'rx_rkhs_exist',rx_rkhs_exist,1,rx_nsurf,1,rx_nrkhs)
    call MRMD_ALLOC(filen,procn,'rx_rkhs_atoms',rx_rkhs_atoms,1,rx_nrkhs,1,2)
    call MRMD_ALLOC(filen,procn,'rx_rkhs_ngrid',rx_rkhs_ngrid,1,rx_nrkhs,1,rx_nsurf)

    ! read data
    if (rx_nrkhs>0) then
      ! allocation
      call MRMD_ALLOC(filen,procn,'tempstring',tempstring,1,3*rx_nsurf) 
      
      ! find the maximum number of grid points
      ngridrkhsmax = 0
      rkhs_calc_coef = .false.
      do i = 1,rx_nrkhs 
        read(rx_paramfile_unit,*,iostat=rkhs_iostat) rx_rkhs_atoms(i,1:2),tempstring(1:3*rx_nsurf)
        if(rkhs_iostat /= 0) call wrndie(-3,'<mrmd.src> mrmd_init', 'Cannot read BOND RKHS block')
        do j = 1,3*rx_nsurf,3
            if((tempstring(j)(1:1) == 'x').OR.(tempstring(j)(1:1) == 'X')) cycle
            read(tempstring(j),'(I10)',iostat=rkhs_iostat) k
            if(rkhs_iostat /= 0) call wrndie(-3,'<mrmd.src> mrmd_init', 'Cannot read BOND RKHS block')
            if(k > ngridrkhsmax) ngridrkhsmax = k
            !check whether kernel coefficients need to be calculated
            if(.not.rkhs_calc_coef) then
                read(tempstring(j+2),'(I10)',iostat=rkhs_iostat) rkhs_coef_unit
                if(rkhs_iostat /= 0) call wrndie(-3,'<mrmd.src> mrmd_init', 'Cannot read BOND RKHS block')
                if(rkhs_coef_unit <= 0) rkhs_calc_coef = .true.
            end if
        end do
      end do
      do i = 1,rx_nrkhs
          backspace(rx_paramfile_unit)
      end do 
      
      ! (re)(de)allocation of grid and coefficient array (needs ngridrkhsmax)
      call MRMD_ALLOC(filen,procn,'rx_rkhs_grid',rx_rkhs_grid,1,ngridrkhsmax,1,rx_nrkhs,1,rx_nsurf)
      call MRMD_ALLOC(filen,procn,'rx_rkhs_coef',rx_rkhs_coef,1,ngridrkhsmax,1,rx_nrkhs,1,rx_nsurf)
      call MRMD_ALLOC(filen,procn,'rx_rkhs_asym',rx_rkhs_asym,1,rx_nrkhs,1,rx_nsurf)
      
      ! allocate temporary arrays in case the coefficients need to be calculated
      if(rkhs_calc_coef) then
        call MRMD_ALLOC(filen,procn,'rkhs_kernel_mat', rkhs_kernel_mat, 1,ngridrkhsmax,1,ngridrkhsmax)
        call MRMD_ALLOC(filen,procn,'rkhs_kernel_diag',rkhs_kernel_diag,1,ngridrkhsmax)
        call MRMD_ALLOC(filen,procn,'rkhs_energy_vec', rkhs_energy_vec, 1,ngridrkhsmax)
      end if

      ! read RKHS potential parameters
      do i = 1,rx_nrkhs
        read(rx_paramfile_unit,*,iostat=rkhs_iostat) rx_rkhs_atoms(i,1:2),tempstring(1:3*rx_nsurf)
        if(rkhs_iostat /= 0) call wrndie(-3,'<mrmd.src> mrmd_init', 'Cannot read BOND RKHS block')

        ! error checking for out of range definitions    
        if (any(rx_rkhs_atoms(i,1:2)<1.or.natom<rx_rkhs_atoms(i,1:2))) &
        call wrndie(-3,'<mrmd.src> mrmd_init','RKHS atom has to be in range: 1<=iatom<=natom')

        ! error checking for different atoms
        if (rx_rkhs_atoms(i,1)==rx_rkhs_atoms(i,2)) &
        call wrndie(-3,'<mrmd.src> mrmd_init', 'RKHS atoms have to be different') 

        ! order atom index
        if (rx_rkhs_atoms(i,1)>rx_rkhs_atoms(i,2)) rx_rkhs_atoms(i,1:2)=rx_rkhs_atoms(i,(/2,1/))

        ! look for duplicate definitions
        do j=1,i-1
          if (all(rx_rkhs_atoms(i,1:2)==rx_rkhs_atoms(j,1:2))) &
            call wrndie(-3,'<mrmd.src> mrmd_init','Duplicate RKHS definition') 
        enddo

        ! look for same bond definition in BOND     
        do j=1,rx_nharm
          if (all(rx_rkhs_atoms(i,1:2)==rx_harm_atoms(j,1:2))) &
            call wrndie(-3,'<mrmd.src> mrmd_init',&
            'Bond in RKHS block is already defined as bond in BOND block') 
        enddo
        
        do j=1,rx_nmors
          if (all(rx_rkhs_atoms(i,1:2)==rx_mors_atoms(j,1:2))) &
            call wrndie(-3,'<mrmd.src> mrmd_init',&
            'Bond in RKHS block is already defined as bond in BOND block') 
        enddo
        

        !read grid and read/calculate coefficients
        do j=1,rx_nsurf
            !skip missing surfaces
            if((tempstring(j*3-2)(1:1) == 'x').or.(tempstring(j*3-2)(1:1) == 'X')) then
                rx_rkhs_ngrid(i,j)  = 0
                rx_rkhs_grid(:,i,j) = 0._chm_real
                rx_rkhs_coef(:,i,j) = 0._chm_real
                rx_rkhs_exist(j,i) =.false.
            else
                !read number of grid points
                read(tempstring(j*3-2),'(I10)') rx_rkhs_ngrid(i,j)
                !read grid file unit
                read(tempstring(j*3-1),'(I10)') rkhs_grid_unit
                inquire(unit=rkhs_grid_unit,opened=rkhs_file_opened)
                if (.not.rkhs_file_opened) then
                    write(tempstring(1),'(I0)') rkhs_grid_unit
                    call wrndie(-3,'<mrmd.src> mrmd_init',&
                    'Unit '//trim(tempstring(1))// &
                    ' defined as grid file in BOND RKHS block was not opened in CHARMM input file!')
                end if
                !read coef file unit
                read(tempstring(j*3),'(I10)')   rkhs_coef_unit
                inquire(unit=abs(rkhs_coef_unit),opened=rkhs_file_opened)
                if((rkhs_coef_unit > 0).and.(.not.rkhs_file_opened)) then
                    write(tempstring(1),'(I0)') rkhs_coef_unit
                    call wrndie(-3,'<mrmd.src> mrmd_init',&
                    'Unit '//trim(tempstring(1))// &
                    ' defined as coefficient file in BOND RKHS block was not opened in CHARMM input file!')
                end if
                
                !coefficients are precalculated
                if(rkhs_coef_unit > 0) then
                    !read grid and coefficients
                    do k = 1,rx_rkhs_ngrid(i,j)
                        read(rkhs_grid_unit,*,iostat=rkhs_iostat) rx_rkhs_grid(k,i,j),tempreal0
                        if(rkhs_iostat /= 0) then
                            write(tempstring(1),'(I0)') rkhs_grid_unit
                            call wrndie(-3,'<mrmd.src> mrmd_init',&
                            'Unit '//trim(tempstring(1))// & 
                            ' defined as grid file in BOND RKHS block could not be read!') 
                        end if
                        read(rkhs_coef_unit,*,iostat=rkhs_iostat) rx_rkhs_coef(k,i,j)
                        if(rkhs_iostat /= 0) then
                            write(tempstring(1),'(I0)') rkhs_coef_unit
                            call wrndie(-3,'<mrmd.src> mrmd_init',&
                            'Unit '//trim(tempstring(1))// & 
                            ' defined as coefficient file in BOND RKHS block could not be read!') 
                        end if
                    end do
                    !read value of the asymptote (last entry in coef file
                    read(rkhs_coef_unit,*,iostat=rkhs_iostat) rx_rkhs_asym(i,j)
                    if(rkhs_iostat /= 0) then
                        write(tempstring(1),'(I0)') rkhs_coef_unit
                        call wrndie(-3,'<mrmd.src> mrmd_init',&
                        'Unit '//trim(tempstring(1))// & 
                        ' defined as coefficient file in BOND RKHS block could not be read (missing asymptote)!') 
                    end if
                !coefficients need to be calculated
                else
                  !read grid
                    do k = 1,rx_rkhs_ngrid(i,j)
                        read(rkhs_grid_unit,*,iostat=rkhs_iostat) rx_rkhs_grid(k,i,j),rkhs_energy_vec(k)
                        if(rkhs_iostat /= 0) then
                            write(tempstring(1),'(I0)') rkhs_grid_unit
                            call wrndie(-3,'<mrmd.src> mrmd_init',&
                            'Unit '//trim(tempstring(1))// & 
                            ' defined as grid file in BOND RKHS block could not be read') 
                        end if
                    end do
                  !calculate coefficients
                    !build kernel matrix
                    do k = 1,rx_rkhs_ngrid(i,j)
                        do l = k,rx_rkhs_ngrid(i,j)
                            rkhs_kernel_mat(k,l) = MRMD_RKHS_KERNEL(rx_rkhs_grid(k,i,j),rx_rkhs_grid(l,i,j))
                        end do
                    end do 
                    !solve for coefficients
                    call MRMD_CHOL_DECOMP(rkhs_kernel_mat (1:rx_rkhs_ngrid(i,j),1:rx_rkhs_ngrid(i,j)), &
                                          rkhs_kernel_diag(1:rx_rkhs_ngrid(i,j)))
                    call MRMD_CHOL_SOLVE (rkhs_kernel_mat (1:rx_rkhs_ngrid(i,j),1:rx_rkhs_ngrid(i,j)), &
                                          rkhs_kernel_diag(1:rx_rkhs_ngrid(i,j)),&
                                          rkhs_energy_vec (1:rx_rkhs_ngrid(i,j)),&
                                          rx_rkhs_coef(1:rx_rkhs_ngrid(i,j),i,j))
                    !calculate value of the asymptote (to be conform with Morse bonds)
                    call MRMD_RKHS_CALC_ASYM(rx_rkhs_ngrid(i,j),rx_rkhs_grid(1:rx_rkhs_ngrid(i,j),i,j),&
                                             rx_rkhs_coef(1:rx_rkhs_ngrid(i,j),i,j),&
                                             rkhs_energy_vec(1:rx_rkhs_ngrid(i,j)),rx_rkhs_asym(i,j))
                    !write coefficients if requested
                    if(rkhs_coef_unit < 0) then
                        rkhs_coef_unit = -rkhs_coef_unit !make it positive
                        do k = 1,rx_rkhs_ngrid(i,j)
                            write(rkhs_coef_unit,*,iostat=rkhs_iostat) rx_rkhs_coef(k,i,j)
                            if(rkhs_iostat /= 0) then
                                call wrndie(-5,'<mrmd.src> mrmd_init',&
                                    'could not write coefficient file, this should not cause problems') 
                                exit
                            end if
                        end do
                        !write asymptote value
                        write(rkhs_coef_unit,*,iostat=rkhs_iostat) rx_rkhs_asym(i,j)
                        if(rkhs_iostat /= 0) then
                            call wrndie(-5,'<mrmd.src> mrmd_init',&
                                'could not write asymptote to coefficient file, this should not cause problems') 
                        end if
                    end if
                end if
            end if
        end do
       enddo
       
       call MRMD_ALLOC(filen,procn,'tempstring',tempstring,1,0)
       ! deallocate temporary arrays in case the coefficients needed to be calculated
       if(rkhs_calc_coef) then
         call MRMD_ALLOC(filen,procn,'rkhs_kernel_mat', rkhs_kernel_mat, 1,0,1,0)
         call MRMD_ALLOC(filen,procn,'rkhs_kernel_diag',rkhs_kernel_diag,1,0)
         call MRMD_ALLOC(filen,procn,'rkhs_energy_vec', rkhs_energy_vec, 1,0)
       end if

    endif    

!===========================================================================
!
!             MAKING UNIFIED MRMD BOND ARRAYS   
!
!===========================================================================

    ! Calculate new value
    rx_nbond=rx_nharm+rx_nmors+rx_nrkhs

    ! (re)(de)allocation (if size>0)
    call MRMD_ALLOC(filen,procn,'rx_bond_exist',rx_bond_exist,0,rx_nsurf,1,rx_nbond)
    call MRMD_ALLOC(filen,procn,'rx_bond'      ,rx_bond      ,1,rx_nbond)
    call MRMD_ALLOC(filen,procn,'rx_bond_inv'  ,rx_bond_inv  ,1,nbond)
    call MRMD_ALLOC(filen,procn,'rx_bond_atoms',rx_bond_atoms,1,rx_nbond,1,2)
    call MRMD_ALLOC(filen,procn,'rx_inc_nharm' ,rx_inc_nharm ,1,rx_nsurf)
    call MRMD_ALLOC(filen,procn,'rx_inc_nmors' ,rx_inc_nmors ,1,rx_nsurf)
    call MRMD_ALLOC(filen,procn,'rx_inc_nrkhs' ,rx_inc_nrkhs ,1,rx_nsurf)
    call MRMD_ALLOC(filen,procn,'rx_inc_nbond' ,rx_inc_nbond ,1,rx_nsurf)

    call MRMD_ALLOC(filen,procn,'rx_exc_bond'  ,rx_exc_bond  ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_harm'  ,rx_inc_harm ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_mors'  ,rx_inc_mors ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_rkhs'  ,rx_inc_rkhs ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_bond'  ,rx_inc_bond ,1,0,1,0)

    rx_exc_nbond=0  ! same for all surface => not array
    rx_inc_nharm=0
    rx_inc_nmors=0
    rx_inc_nrkhs=0
    rx_inc_nbond=0
    if (nbond>0) rx_bond_inv=-1

    ! store data
    if (rx_nbond>0) then 

       ! Copy definitions from harmonic bond arrays to unified bond arrays
       do i=1, rx_nharm
         rx_bond_exist(1:rx_nsurf,i)    =rx_harm_exist(1:rx_nsurf,i)
         rx_bond_atoms(           i,1:2)=rx_harm_atoms(i,1:2)
       enddo

       ! Copy definitions from Morse bond arrays to unified bond arrays
       do i=1, rx_nmors
         k=rx_nharm+i
         rx_bond_exist(1:rx_nsurf,k    )=rx_mors_exist(1:rx_nsurf,i)
         rx_bond_atoms(           k,1:2)=rx_mors_atoms(i,1:2)
       enddo
       
       ! Copy definitions from RKHS bond arrays to unified bond arrays
       do i=1, rx_nrkhs
         k=rx_nharm+rx_nmors+i
         rx_bond_exist(1:rx_nsurf,k    )=rx_rkhs_exist(1:rx_nsurf,i)
         rx_bond_atoms(           k,1:2)=rx_rkhs_atoms(i,1:2)
       enddo
    
        ! Check if bond already exists in CHARMM  
        rx_bond_exist(0,:)=.false.
        rx_bond           =-1
        rx_exc_nbond      =0    ! it was set to 0 earlier,too
        do i=1,rx_nbond
            do j=1,nbond
                if (all(rx_bond_atoms(i,1:2)==(/IB(j),JB(j)/)).or. & 
                   all(rx_bond_atoms(i,1:2)==(/JB(j),IB(j)/))) then 
                    rx_bond_exist(0,i)=.true.
                    rx_bond        (i)=j
                    rx_bond_inv    (j)=i

                    ! increase counter and exit
                    rx_exc_nbond      =rx_exc_nbond+1
                    exit
                endif
            enddo
        enddo
    !------------------------------------------------------------
    !
    !       CREATE EXCLUSION LIST FOR CHARMM BONDS  - ALWAYS HARMONIC
    !   
    !------------------------------------------------------------
        if (rx_exc_nbond>0) then
            call MRMD_ALLOC(filen,procn,'rx_exc_bond',rx_exc_bond,1,rx_exc_nbond)
            rx_exc_nbond=0
            do i=1,rx_nbond
                ! if MRMD bond is listed in CHARMM 
                ! then it should be removed from CHARMM
                if (rx_bond_exist(0,i)) then
                    rx_exc_nbond=rx_exc_nbond+1                 
                    rx_exc_bond(rx_exc_nbond)=rx_bond(i)
                endif     
            enddo
        endif
    !------------------------------------------------------------
    !
    !    CREATE INCLUSION LIST FOR MRMD HARMONIC/MORSE/RKHS/ALL BONDS
    !
    !------------------------------------------------------------
        ! count how many bonds are redefined (made or reparameterized(bonded!)) 
        ! in each surfaces take their maximum
        ! harmonic
        if (rx_nharm>0) then
            rx_inc_nharm0=maxval(count(rx_bond_exist(1:rx_nsurf,         1:rx_nharm),2))
        else
            rx_inc_nharm0=0
        endif
        
        !Morse
        if (rx_nmors>0) then
            rx_inc_nmors0=maxval(count(rx_bond_exist(1:rx_nsurf,rx_nharm+1:rx_nharm+rx_nmors),2))
        else        
            rx_inc_nmors0=0
        endif     
        
        !RKHS
        if (rx_nrkhs>0) then
            rx_inc_nrkhs0=maxval(count(rx_bond_exist(1:rx_nsurf,rx_nharm+rx_nmors+1:rx_nbond),2))
        else        
            rx_inc_nrkhs0=0
        endif      
        
        !rx_nbond>0 here always
        rx_inc_nbond0=maxval(count(rx_bond_exist(1:rx_nsurf,         1:rx_nbond),2))

        ! (re)(de)allocation (if size>0)
        call MRMD_ALLOC(filen,procn,'rx_inc_harm',rx_inc_harm,1,rx_nsurf,1,rx_inc_nharm0)
        call MRMD_ALLOC(filen,procn,'rx_inc_mors',rx_inc_mors,1,rx_nsurf,1,rx_inc_nmors0)
        call MRMD_ALLOC(filen,procn,'rx_inc_rkhs',rx_inc_rkhs,1,rx_nsurf,1,rx_inc_nrkhs0)
        call MRMD_ALLOC(filen,procn,'rx_inc_bond',rx_inc_bond,1,rx_nsurf,1,rx_inc_nbond0)

        ! creating lists
        rx_inc_nharm=0  ! it was set to 0 earlier,too
        rx_inc_nmors=0  ! it was set to 0 earlier,too
        rx_inc_nrkhs=0  ! it was set to 0 earlier,too
        rx_inc_nbond=0  ! it was set to 0 earlier,too       
        do i=1,rx_nsurf
            do j=1,rx_nbond
                ! if it is bonded in MRMD
                ! it should be added from MRMD
                if (rx_bond_exist(i,j)) then
                    rx_inc_nbond(i)=rx_inc_nbond(i)+1
                    rx_inc_bond(i,rx_inc_nbond(i))=j

                    ! harmonic 1...rx_nharm
                    if (1<=j.and.j<=rx_nharm) then
                        rx_inc_nharm(i)=rx_inc_nharm(i)+1
                         rx_inc_harm(i, rx_inc_nharm(i))=j
                    endif
                    
                    ! Morse from rx_nharm+1...rx_nharm+rx_nmors
                    if (rx_nharm+1<=j.and.j<=rx_nharm+rx_nmors) then
                        rx_inc_nmors(i)=rx_inc_nmors(i)+1
                         rx_inc_mors(i, rx_inc_nmors(i))=j-rx_nharm
                    endif
                    
                    ! RKHS from rx_nharm+rx_nmors+1...rx_nharm+rx_nmors+rx_rkhs
                    if (rx_nharm+rx_nmors+1<=j.and.j<=rx_nharm+rx_nmors+rx_nrkhs) then
                        rx_inc_nrkhs(i)=rx_inc_nrkhs(i)+1
                         rx_inc_rkhs(i, rx_inc_nrkhs(i))=j-rx_nharm-rx_nmors
                    endif
                    
                    ! further bond types...
                    
                endif
            enddo                   
        enddo
    endif

!===========================================================================
!
!           READING PARAMETERS FOR MRMD ANGLES  
!
!===========================================================================
    ! read value 
    if (PRNLEV>=5) write(outu,'(A)') 'MRMD_INIT> Reading ANGL HARM block...'
    read(rx_paramfile_unit,*) keyword1,keyword2,rx_nangl
    if (MRMD_UPPERCASE(keyword1)/='ANGL'.or.MRMD_UPPERCASE(keyword2)/='HARM') &    
    call wrndie(-3,'<mrmd.src> mrmd_init', 'Missing ANGL HARM block in input')     
    if (rx_nangl<0) &
    call wrndie(-3,'<mrmd.src> mrmd_init','Number of harmonic angles cannot be negative !')

    ! (re)(de)allocation (if size>0) 
    call MRMD_ALLOC(filen,procn,'rx_angl_exist'  ,rx_angl_exist  ,0,rx_nsurf,1,rx_nangl)
    call MRMD_ALLOC(filen,procn,'rx_angl'        ,rx_angl        ,1,rx_nangl)
    call MRMD_ALLOC(filen,procn,'rx_angl_inv'    ,rx_angl_inv    ,1,ntheta)
    call MRMD_ALLOC(filen,procn,'rx_angl_atoms'  ,rx_angl_atoms  ,1,rx_nangl,1,3)
    call MRMD_ALLOC(filen,procn,'rx_angl_fc_half',rx_angl_fc_half,1,rx_nsurf,1,rx_nangl)
    call MRMD_ALLOC(filen,procn,'rx_angl_phie'   ,rx_angl_phie   ,1,rx_nsurf,1,rx_nangl)
    call MRMD_ALLOC(filen,procn,'rx_urey_exist'  ,rx_urey_exist  ,0,rx_nsurf,1,rx_nangl)
    call MRMD_ALLOC(filen,procn,'rx_urey_fc_half',rx_urey_fc_half,1,rx_nsurf,1,rx_nangl)
    call MRMD_ALLOC(filen,procn,'rx_urey_re'     ,rx_urey_re     ,1,rx_nsurf,1,rx_nangl)


    ! initialized inverse array
    if (ntheta>0) rx_angl_inv=-1

    ! read data
    if (rx_nangl>0) then
        call MRMD_ALLOC(filen,procn,'tempstring',tempstring,1,4*rx_nsurf)
        
        do i = 1,rx_nangl
            read(rx_paramfile_unit,*) rx_angl_atoms(i,1:3),tempstring(1:4*rx_nsurf)

            ! error checking for out of range definitions    
            if (any(rx_angl_atoms(i,1:3)<1.or.natom<rx_angl_atoms(i,1:3))) &
            call wrndie(-3,'<mrmd.src> mrmd_init','ANGL atom has to be in range: 1<=iatom<=natom') 

            ! error checking for same atoms
            do j=1,2
               do k=j+1,3
                   if (rx_angl_atoms(i,j)==rx_angl_atoms(i,k))     &
                   call wrndie(-3,'<mrmd.src> mrmd_init','ANGL atoms have to be different') 
               enddo
            enddo 

            ! look for duplicate definitions
            do j=1,i-1
              if (all(rx_angl_atoms(i,1:3)      ==rx_angl_atoms(j,1:3))  &
                  .or.                                                   &
                  all(rx_angl_atoms(i,(/3,2,1/))==rx_angl_atoms(j,1:3))) &
                call wrndie(-3,'<mrmd.src> mrmd_init','Duplicate ANGL definition') 
            enddo

            ! store values 
            do j=1,rx_nsurf
                ! deleted bond if force constant is x or X or starts with x or X
                if (tempstring(4*j-3)(1:1)=='x'.or.tempstring(4*j-3)(1:1)=='X') then
                    rx_angl_fc_half(j,i)= -9999._chm_real
                    rx_angl_phie   (j,i)= -9999._chm_real
                    rx_urey_fc_half(j,i)= -9999._chm_real
                    rx_urey_re     (j,i)= -9999._chm_real
                    rx_angl_exist  (j,i)=.false.
                    rx_urey_exist  (j,i)=.false.
                else 
                    read(tempstring(4*j-3),*) rx_angl_fc_half(j,i)
                    read(tempstring(4*j-2),*) rx_angl_phie   (j,i)
                    
                    ! Urey-Bradley parameters are optional
                    if (tempstring(4*j-1)(1:1)=='x'.or.tempstring(4*j-1)(1:1)=='X') then
                        rx_urey_fc_half(j,i)= -9999._chm_real
                        rx_urey_re     (j,i)= -9999._chm_real
                        rx_urey_exist  (j,i)=.false.
                    else
                        read(tempstring(4*j-1),*) rx_urey_fc_half(j,i)
                        read(tempstring(4*j  ),*) rx_urey_re     (j,i)
                        rx_urey_exist  (j,i)=.true.
                    endif 

                    rx_angl_phie (j,i)= rx_angl_phie(j,i)*degrad ! converting from degree to radian
                    rx_angl_exist(j,i)=.true.

                    if (rx_angl_fc_half(j,i) <0) then
                         if (WRNLEV>=5) then
                            write(outu,'(2(A,I3,x),3x,A,3(x,I6))') &
                            'MRMD_INIT> ANGL:',i,'on surface:',j,  &
                            'between atoms',rx_angl_atoms(i,1:3)
                            write(outu,'(A,x,G12.5)') &
                            'MRMD_INIT> - has negative half force constant:',rx_angl_fc_half(j,i)
                         endif
                         call wrndie(-3,'<mrmd.src> mrmd_init', &
                         'Negative force constant for angle potential.') 
                    endif
                    if (rx_angl_fc_half(j,i) >0) then
                        if (rx_angl_phie   (j,i)<=0) then
                            if (WRNLEV>=5) then
                                write(outu,'(2(A,I3,x),3x,A,3(x,I6))') &
                                'MRMD_INIT> ANGL:',i,'on surface:',j,  &
                                'between atoms',rx_angl_atoms(i,1:3)
                                write(outu,'(A,x,G12.5)') &
                                'MRMD_INIT> - has non-positive equilibrium angle:',rx_angl_phie(j,i)
                            endif
                            call wrndie(-3,'<mrmd.src> mrmd_init', &
                            'Non-positive equilibrium angle for angle potential.') 
                        endif

                        if (rx_urey_exist(j,i)) then
                            if (rx_urey_fc_half(j,i)<0) then
                                if (WRNLEV>=5) then
                                    write(outu,'(2(A,I3,x),3x,A,3(x,I6))') &
                                    'MRMD_INIT> ANGL:',i,'on surface:',j,  &
                                    'between atoms',rx_angl_atoms(i,1:3)
                                    write(outu,'(A,x,G12.5)') &
                                    'MRMD_INIT> has negative Urey-Bradley half force constant:',&
                                    rx_urey_fc_half(j,i)
                                endif
                                call wrndie(-3,'<mrmd.src> mrmd_init', &
                                'Negative force constant for Urey-Bradley potential.')
                            endif                     
                            if (rx_urey_fc_half(j,i) >0) then
                            if (rx_urey_re     (j,i)<=0) then
                                if (WRNLEV>=5) then
                                    write(outu,'(2(A,I3,x),3x,A,3(x,I6))') &
                                    'MRMD_INIT> ANGL:',i,'on surface:',j,  &
                                    'between atoms',rx_angl_atoms(i,1:3)
                                    write(outu,'(A,x,G12.5)') &
                                  'MRMD_INIT> has non-positive Urey-Bradley equilibrium distance:',&
                                    rx_urey_re(j,i)
                                endif
                                call wrndie(-3,'<mrmd.src> mrmd_init', &
                                'Non-positive equilibrium distance for Urey-Bradley potential.')
                            endif
                            endif                     
                        endif
                    endif
                endif
            enddo
        enddo
        call MRMD_ALLOC(filen,procn,'tempstring',tempstring,1,0)
        
        ! identifying it in CHARMM index
        rx_angl_exist(0,:)=.false.
        rx_urey_exist(0,:)=.false.
        rx_angl    =-1
        do i = 1,rx_nangl
            do j=1,ntheta
              if (all(rx_angl_atoms(i,1:3)==(/IT(j),JT(j),KT(j)/))     &
                 .or.                                                 &
                 all(rx_angl_atoms(i,1:3)==(/KT(j),JT(j),IT(j)/))) then
                    rx_angl_exist(0,i)=.true.
                    ! warning
                    ! can have multiple match as charmm CHARMM is stupid
                    ! and allows ABC and CBA at the same time (with some warning
                    rx_angl(i)=j
                    rx_angl_inv(j)=i
                    
                    ! Urey-Bradley force constant is positive
                    if (CTUC(ICT(j))>0._chm_real) rx_urey_exist(0,i)=.true.
              endif
            enddo
        enddo            
    endif

    !-------------------------------------------------------------------
    !      I. CONSISTENCY CHECKING FOR MRMD ANGLES:           
    !      DO THE MRMD-ANGLES HAVE
    !      CORRESPONDING BONDS PRESENT IN CHARMM&MRMD?
    !-------------------------------------------------------------------

    !                   ANGLE                 BOND1(2)          BOND2(1)               
    !1a(<=2a)        MRMD_PRESENT    =>    MRMD_PRESENT+    MRMD_PRESENT : checked   
    !1b(<=2b)        MRMD_PRESENT    =>    MRMD_PRESENT+CHARMM/MRMD_PRESENT : checked   
    !1c(<=2c)    CHARMM/MRMD_PRESENT =>  CHARMM/MRMD_PRESENT+CHARMM/MRMD_PRESENT : checked   
    !1d(<=2d)    CHARMM              =>  CHARMM/MRMD_PRESENT+CHARMM/MRMD_PRESENT: doesn't need  
    !         to be checked as if a bond removed then CHARMM angle will be removed automatically             
    !1e(<=2e)        MRMD_PRESENT    =>      MRMD_PRESENT+CHARMM     : checked   
    !1f(<=2f)    CHARMM/MRMD_PRESENT =>  CHARMM/MRMD_PRESENT+CHARMM  : checked   
    !1g(<=2g)    CHARMM              =>  CHARMM/MRMD_PRESENT+CHARMM  : doesn't need to be checked
    !                                                           as if a bond removed then
    !                                                           CHARMM angle will be removed
    !                                                           automatically             
    !1h(<=2h)    CHARMM              =>  CHARMM              +CHARMM  : it's okay 

    error_found=.false.
    do i=1,rx_nsurf
        !                   ANGLE                 BOND1(2)          BOND2(1)               
        !1a(<=2a)        MRMD_PRESENT    =>      MRMD_PRESENT+    MRMD_PRESENT        : checked   
        !1b(<=2b)        MRMD_PRESENT    =>      MRMD_PRESENT+CHARMM/MRMD_PRESENT     : checked   
        !1c(<=2c)    CHARMM/MRMD_PRESENT =>  CHARMM/MRMD_PRESENT+CHARMM/MRMD_PRESENT  : checked   
        !1e(<=2e)        MRMD_PRESENT    =>      MRMD_PRESENT+CHARMM                  : checked   
        !1f(<=2f)    CHARMM/MRMD_PRESENT =>  CHARMM/MRMD_PRESENT+CHARMM               : checked   
        do j=1,rx_nangl
            if (.not.rx_angl_exist(i,j)) cycle
                
            ! consistency checking = are their bonds are also present?
            ! assume first they are not
            !note, exists(3) and found(3) also exist but not used here
            ! use indices 1,2 everywhere to avoid referring the whole arrays
            found(1:2)=.false.  !IT,JT,KT                  
            exists(1:2)=.false.

            ! search for its bonds (m=1 or 2) - are they present?
            do m=1,2
                ! search for its bonds in MRMD
                ! for cases 1a,1b,1c,1e,1f
                do k=1,rx_nbond
                    ! if found exit
                    if (found(m)) exit

                    ! if atoms are not matched in any order => cycle
                    if (.not.(                                                       &
                        all(rx_angl_atoms(j,m:m+1)==rx_bond_atoms(k,(/1,2/))).or.   &
                        all(rx_angl_atoms(j,m:m+1)==rx_bond_atoms(k,(/2,1/)))    )) cycle

                    ! if matched => found
                    found(m)    =.true.
                    
                    ! if exist in the PES
                    exists(m)   =rx_bond_exist(i,k)
                    
                    ! found=> exit
                    exit

                enddo
            
                ! search for its bonds in CHARMM
                ! for cases 1b,1c,1e,1f
                do k=1,nbond
                    ! if it has been found (here or in MRMD) then quit
                    if (found(m)) exit

                   ! if the the bond in MRMD =>then skip, because it has already been investigated
                   !                       and was not found (that's why it could reach this point)
                    if (rx_bond_inv(k)>0) cycle
                    
                    ! if atoms are not matched in any order => cycle
                    if (.not.(   &
                         all(rx_angl_atoms(j,m:m+1)==(/IB(k),JB(k)/)).or.            &
                         all(rx_angl_atoms(j,m:m+1)==(/JB(k),IB(k)/))    )) cycle
                          
                    ! if matched => found
                    found(m)    =.true.
                    
                    ! if exists in CHARMM, but not listed in MRMD=>always present 
                    exists(m)   =.true.

                enddo
                
                ! not found or found but doesn't exist=> warning
                if (.not.found(m)) then
                    if (WRNLEV>=5) then 
                        write(outu,'(2(A,I3,x),3x,A,3(x,I6))') &
                        'MRMD_INIT> ANGL:',j,'on surface:',i,  &
                        'between atoms',rx_angl_atoms(j,1:3)
                        write(outu,'(A,2(x,I6),x,A)') &
                        'MRMD_INIT> - its bond between atoms',rx_angl_atoms(j,m:m+1),    &
                        'is listed neither in CHARMM nor in MRMD'
                    endif
                    error_found=.true.
                elseif (.not.exists(m)) then
                    if (WRNLEV>=5) then 
                        write(outu,'(2(A,I3,x),3x,A,3(x,I6))') &
                        'MRMD_INIT> ANGL:',j,'on surface:',i,  &
                        'between atoms',rx_angl_atoms(j,1:3)
                        write(outu,'(A,2(x,I6),x,A)') &
                        'MRMD_INIT> - its bond between atoms',rx_angl_atoms(j,m:m+1),    &
                        'is not present according to CHARMM&MRMD'
                    endif
                    error_found=.true.
                endif
            enddo
        enddo
    enddo
    if (error_found) call wrndie(-3,'<mrmd.src> mrmd_init','Errors were found!')

    !-------------------------------------------------------------------
    !      II. CONSISTENCY CHECKING FOR MRMD BONDS:           
    !      DO THE ANGLES FORMED DUE TO NEW MRMD-BONDS
    !      PRESENT IN CHARMM&MRMD?
    !-------------------------------------------------------------------

    !          BOND1(2)          BOND2(1)             ANGLE
    !2a     MRMD_PRESENT+    MRMD_PRESENT       =>     MRMD_PRESENT   : checked
    !2b     MRMD_PRESENT+CHARMM/MRMD_PRESENT    =>     MRMD_PRESENT   : checked
    !2c CHARMM/MRMD_PRESENT+CHARMM/MRMD_PRESENT => CHARMM/MRMD_PRESENT   : checked
    !2d CHARMM/MRMD_PRESENT+CHARMM/MRMD_PRESENT => CHARMM                 : checked
    !2e     MRMD_PRESENT+CHARMM                 =>     MRMD_PRESENT   : checked
    !2f CHARMM/MRMD_PRESENT+CHARMM              => CHARMM/MRMD_PRESENT   : checked
    !2g CHARMM/MRMD_PRESENT+CHARMM              => CHARMM                 : checked
    !2h CHARMM              +CHARMM             => CHARMM                 : it's okay

    error_found=.false.
    do i=1,rx_nsurf
        do j=1,rx_nbond
            ! ONE BOND(j) IS IN MRMD
            if (.not.rx_bond_exist(i,j)) cycle
            

            ! 2a-2d
            !          BOND1(2)          BOND2(1)             ANGLE
            !2a     MRMD_PRESENT+    MRMD_PRESENT =>     MRMD_PRESENT   : checked
            !2b     MRMD_PRESENT+CHARMM/MRMD_PRESENT =>     MRMD_PRESENT   : checked
            !2c CHARMM/MRMD_PRESENT+CHARMM/MRMD_PRESENT => CHARMM/MRMD_PRESENT   : checked
            !2d CHARMM/MRMD_PRESENT+CHARMM/MRMD_PRESENT => CHARMM                 : checked




            ! IF OTHER BOND(k) IS IN MRMD AND PRESENT
            do k=j+1,rx_nbond
                ! if not bonded in the PES then cycle
                if (.not.rx_bond_exist(i,k)) cycle 
                
                ! do they have one common atom? (can't have two!, filtered earlier)
                do j1=1,2
                    do k1=1,2
                        if (rx_bond_atoms(j,j1)==rx_bond_atoms(k,k1)) then
                            ! index of angle's edge atom in MRMD bond(j)
                            i1=rx_bond_atoms(j,3-j1)    !1=>2,2=>1
                            
                            ! connecting atom
                            i2=rx_bond_atoms(j,j1)
                            
                            ! index of angle's edge atom in MRMD bond(k)
                            i3=rx_bond_atoms(k,3-k1)    !1=>2,2=>1 
                            ! is there a corresponding angle present in MRMD or CHARMM?
                            found(1) =.false.
                            exists(1)=.false.
                            
                       !          BOND1(2)          BOND2(1)             ANGLE
                       !2a     MRMD_PRESENT+    MRMD_PRESENT =>     MRMD_PRESENT : checked
                       !2b     MRMD_PRESENT+CHARMM/MRMD_PRESENT =>     MRMD_PRESENT : checked
                       !2c CHARMM/MRMD_PRESENT+CHARMM/MRMD_PRESENT => CHARMM/MRMD_PRESENT : checked
                       ! IN MRMD: is there a corresponding angle present in MRMD?
                            do m=1,rx_nangl
                                if (found(1)) exit
                                if (all(rx_angl_atoms(m,1:3)==(/i1,i2,i3/)).or.    &
                                    all(rx_angl_atoms(m,1:3)==(/i3,i2,i1/))  ) then
                                    found(1)=.true.
                                    exists(1)=rx_angl_exist(i,m)
                                endif
                            enddo

                            ! 2d
                            !          BOND1(2)          BOND2(1)             ANGLE
                            !2d CHARMM/MRMD_PRESENT+CHARMM/MRMD_PRESENT => CHARMM      : checked
                            ! IN CHARMM: is there a corresponding angle present in CHARMM,
                                                                 ! but not in MRMD?
                            do m=1,ntheta
                                ! 'CHARMM angle',m,'atoms',IT(m),JT(M),KT(M)                                    
                                ! if found already (in MRMD or here) => exit
                                if (found(1)) exit
                                
                                ! if the the angl in MRMD 
                                ! =>then skip, because it has already been investigated
                                !   and was not found (that's why it could reach this point)
                                if (rx_angl_inv(m)>0) cycle
                                
                                ! If atoms are not matched in right order => cycle                                
                                if (.not.( all((/IT(m),JT(M),KT(M)/)==(/i1,i2,i3/)).or. &
                                           all((/IT(m),JT(M),KT(M)/)==(/i3,i2,i1/))    )  ) cycle
                                    
                                ! if matched => found
                                found(1)=.true.
                                                                
                                ! if exists in CHARMM, but not listed in MRMD=>always present 
                                exists(1)=.true.
                            enddo

                            ! not found or found but doesn't exist=> warning
                            if (.not.found(1)) then
                                if (WRNLEV>=5) then
                                   write(outu,'(A,I3,2(x,A,x,I6),x,A,3(x,I6))') &
                                   'MRMD_INIT> On surface:',i,'HARM&MORS:',j,'HARM&MORS:',k,  &
                                   'form an angle between atoms',i1,i2,i3
                                   write(outu,*)&
                                'MRMD_INIT> - but the angle is listed neither in CHARMM nor in MRMD'
                                endif
                                error_found=.true.
                            elseif (.not.exists(1)) then
                                if (WRNLEV>=5) then
                                   write(outu,'(A,I3,2(x,A,x,I6),x,A,3(x,I6))') &
                                   'MRMD_INIT> On surface:',i,'HARM&MORS:',j,'HARM&MORS:',k,  &
                                   'form an angle between atoms',i1,i2,i3
                                   write(outu,*)&
                                'MRMD_INIT> - but the angle is not present according to CHARMM&MRMD'
                                endif
                                error_found=.true.
                            endif
                        endif
                    enddo
                enddo                        
            enddo
            
            
            ! 2e-2g
            !          BOND1(2)          BOND2(1)             ANGLE
            !2e     MRMD_PRESENT+CHARMM               =>     MRMD_PRESENT   : checked
            !2f CHARMM/MRMD_PRESENT+CHARMM               => CHARMM/MRMD_PRESENT   : checked
            !2g CHARMM/MRMD_PRESENT+CHARMM               => CHARMM                 : checked
            ! IF OTHER BOND(k) IS IN ONLY THE CHARMM 
            ! = OLD BOND which was not removed
            do k=1,nbond
                ! if it is also in MRMD => cycle
                if (rx_bond_inv(k)>0) cycle 

                ! only in CHARMM => always present
                ! do they have one common atom? (can't have two!, filtered earlier)
                do j1=1,2
                    do k1=1,2
                        if ((k1==1.and.rx_bond_atoms(j,j1)==IB(k)).or. &
                            (k1==2.and.rx_bond_atoms(j,j1)==JB(k))) then
                            ! index of angle's edge atom in MRMD bond(j)
                            i1=rx_bond_atoms(j,3-j1)    !1=>2,2=>1
                            ! index of angle's connecting atom
                            i2=rx_bond_atoms(j,j1)
                            ! index of angle's edge atom in CHARMM bond(k)
                            if (k1==1) i3=JB(k)    !1=>2 
                            if (k1==2) i3=IB(k)    !2=>1 
                            
                            ! is there a corresponding angle present in MRMD or CHARMM?
                            found(1) =.false.
                            exists(1)=.false.
                            
                            ! 2e-2f
                            !          BOND1(2)          BOND2(1)             ANGLE
                            !2e     MRMD_PRESENT+CHARMM             =>     MRMD_PRESENT   : checked
                            !2f CHARMM/MRMD_PRESENT+CHARMM          => CHARMM/MRMD_PRESENT : checked
                            ! IN MRMD: is there a corresponding angle present in MRMD?
                            do m=1,rx_nangl
                                if (found(1)) exit
                                if (all(rx_angl_atoms(m,1:3)==(/i1,i2,i3/)).or.    &
                                    all(rx_angl_atoms(m,1:3)==(/i3,i2,i1/))  ) then
                                    found(1)=.true.
                                    exists(1)=rx_angl_exist(i,m)
                                endif
                            enddo

                            ! 2g
                            !         BOND1(2)          BOND2(1)             ANGLE
                            !2g CHARMM/MRMD_PRESENT+CHARMM           => CHARMM            : checked
                            ! IN CHARMM: is there a corresponding angle present in CHARMM, 
                            ! but not in MRMD?
                            do m=1,ntheta
                                ! if found already (in MRMD or here) => exit
                                if (found(1)) exit
                                
                                ! if the the angl in MRMD =>then skip, because 
                                ! it has already been investigated
                                !                            and was not found 
                                ! (that's why it could reach this point)
                                if (rx_angl_inv(m)>0) cycle
                                
                                ! If atoms are not matched in right order => cycle                                
                                if (.not.( all((/IT(m),JT(M),KT(M)/)==(/i1,i2,i3/)).or.     &
                                           all((/IT(m),JT(M),KT(M)/)==(/i3,i2,i1/))    )  ) cycle
                                    
                                ! if matched => found
                                found(1)=.true.
                                                                
                                ! if exists in CHARMM, but not listed in MRMD=>always present 
                                exists(1)=.true.
                            enddo

                            ! not found or found but doesn't exist=> warning
                            if (.not.found(1)) then
                                if (WRNLEV>=5) then
                                   write(outu,'(A,I3,2(x,A,x,I6),x,A,3(x,I6))') &
                                   'MRMD_INIT> On surface:',i,'HARM&MORS:',j,'CHARMM-BOND:',k,  &
                                   'form an angle between atoms',i1,i2,i3
                                   write(outu,*)&
                                'MRMD_INIT> - but the angle is listed neither in CHARMM nor in MRMD'
                                endif
                                error_found=.true.
                            elseif (.not.exists(1)) then
                                if (WRNLEV>=5) then
                                   write(outu,'(A,I3,2(x,A,x,I6),x,A,3(x,I6))') &
                                   'MRMD_INIT> On surface:',i,'HARM&MORS:',j,'CHARMM-BOND:',k,  &
                                   'form an angle between atoms',i1,i2,i3
                                   write(outu,*)&
                                'MRMD_INIT> - but the angle is not present according to CHARMM&MRMD'
                                endif
                                error_found=.true.
                            endif
                        endif
                    enddo
                enddo                        
            enddo                    
        enddo
    enddo
    if (error_found)  &
    call wrndie(-3,'<mrmd.src> mrmd_init','Bonds and angles suggest different connectivity.')


    !-------------------------------------------------------------------
    !
    !     CREATE EXCLUSIONS FOR CHARMM ANGLES - done
    !
    !-------------------------------------------------------------------
    ! (re)(de)allocation (if size>0)
    call MRMD_ALLOC(filen,procn,'rx_exc_nangl',rx_exc_nangl,1,rx_nsurf)
    call MRMD_ALLOC(filen,procn,'rx_exc_angl' ,rx_exc_angl ,1,0,1,0)

    ! 1. count how many angles of CHARMM was redefined in mrmd
    ! they should be removed from CHARMM energy
    ! 2. count how many FURTHER CHARMM angles have been removed due 
    ! to bond removal in MRMD

    !  first cycle(ii=1): determining array sizes 
    ! second cycle(ii=2): allocation and fill of arrays

    do ii=1,2
        ! CHARMM angles which are deleted or redefined in MRMD ANGL
        ! same for all surfaces!
        if (ii==1) then
            if (rx_nangl>0) then
                rx_exc_nangl=count(rx_angl_exist(0,1:rx_nangl))
            else
                rx_exc_nangl=0
                ! further angles could be removed due to bond removal
                ! thus do not exit now!
            endif        
        endif

        ! CHARMM angles in MRMD => should be removed in all surfaces
        if (ii==2) then

            ! maximum number of exclusions found
            rx_exc_nangl0=maxval(rx_exc_nangl)
            ! if zero =>exit
            if (rx_exc_nangl0==0) exit

            ! allocating arrays
            call MRMD_ALLOC(filen,procn,'rx_exc_angl',rx_exc_angl,1,rx_nsurf,1,rx_exc_nangl0)

            ! reset counters and count and store the angles to be removed from CHARMM
            rx_exc_nangl=0
            do j=1,rx_nangl
                if (rx_angl_exist(0,j)) then
                    ! increase counter
                    rx_exc_nangl(1:rx_nsurf)=rx_exc_nangl(1:rx_nsurf)+1                        

                    ! at this moment all rx_exc_nangl component has the same value
                    ! so take for example the first
                    rx_exc_angl(1:rx_nsurf,rx_exc_nangl(1))=rx_angl(j)                        
                endif
            enddo
        endif
                    
        ! CHARMM angles not in MRMD
        do j=1,ntheta
            ! is it in MRMD ANGL? => handled already
            if (rx_angl_inv(j)>0) cycle

            ! check wether bond is removed on each PES
            do i=1,rx_nsurf

                ! it is not in MRMD ANGL
                ! check wether any of its bond has been removed and not recreated on PES
                do k=1,rx_nbond
                
                    ! if bond present in surface => cannot remove angle =>no exclusion
                    if (rx_bond_exist(i,k)) cycle

                    ! if not in CHARMM => cannot affect CHARMM angle                    
                    if (.not.rx_bond_exist(0,k)) cycle

                    ! in CHARMM but removed by MRMD from PES
                    ! is it one of the bonds within the angle (IT-JT-KT)?
                    if (all(rx_bond_atoms(k,1:2)==(/IT(j),JT(j)/)).or.   &
                        all(rx_bond_atoms(k,1:2)==(/JT(j),IT(j)/)).or.   &
                        all(rx_bond_atoms(k,1:2)==(/JT(j),KT(j)/)).or.   &
                        all(rx_bond_atoms(k,1:2)==(/KT(j),JT(j)/))) then

                        ! rx_exc_nangl counters here can already be different
                        rx_exc_nangl(i)=rx_exc_nangl(i)+1
                        if (ii==2) rx_exc_angl(i,rx_exc_nangl(i))=j
                        
                        if (WRNLEV>=5) then
                            write(outu,'(A,I6,3x,A,3(x,I6))') &
                           'MRMD_INIT> CHARMM-ANGL:',j,'between atoms',IT(j),JT(j),KT(j) 
                            write(outu,'(A,x,I2)') 'MRMD_INIT> On surface',i
                            write(outu,'(A)')&
                            'MRMD_INIT> - it is removed due to HARM/MORS removal in MRMD'
                            write(outu,'(A)') &
                            'MRMD_INIT> - but its removal is not listed in MRMD parameter file.'  
                            write(outu,'(A)') &
                            'MRMD_INIT> - It will be automatically removed by MRMD.'  
                        endif
                        exit ! now it has been removed on this surface
                             ! should not be removed twice by its another removed bond =>exiting      
                    endif   
                enddo            
            enddo
        enddo
    enddo
        
    !------------------------------------------------------------
    !
    ! CREATE INCLUSION LIST FOR MRMD ANGLES
    !
    !------------------------------------------------------------
    rx_inc_nangl0=0
    if (rx_nangl>0) rx_inc_nangl0=maxval(count(rx_angl_exist(1:rx_nsurf,1:rx_nangl),2))

    ! (re)(de)allocation (if size>0)
    call MRMD_ALLOC(filen,procn,'rx_inc_nangl',rx_inc_nangl,1,rx_nsurf)
    call MRMD_ALLOC(filen,procn,'rx_inc_angl',rx_inc_angl,1,rx_nsurf,1,rx_inc_nangl0)

    ! fill arrays
    rx_inc_nangl=0
    do i=1,rx_nsurf
        do j=1,rx_nangl
            if (.not.rx_angl_exist(i,j)) cycle
            rx_inc_nangl(i)=rx_inc_nangl(i)+1
            rx_inc_angl(i,rx_inc_nangl(i))=j
        enddo
    enddo

!===========================================================================
!
!   READING PARAMETERS FOR MRMD PROPER DIHEDRALS 
!
!===========================================================================
    ! read value 
    if (PRNLEV>=5) write(outu,'(A)') 'MRMD_INIT> Reading DIHE FOUR block...'
    read(rx_paramfile_unit,*) keyword1,keyword2,rx_ndihe
    if (MRMD_UPPERCASE(keyword1)/='DIHE'.or.MRMD_UPPERCASE(keyword2)/='FOUR') &    
    call wrndie(-3,'<mrmd.src> mrmd_init', 'Missing DIHE FOUR block in input')     
 
    if (rx_ndihe<0) &
    call wrndie(-3,'<mrmd.src> mrmd_init','Number of proper dihedrals cannot be negative !')

    ! (re)(de)allocation (if size>0)
    call MRMD_ALLOC(filen,procn,'rx_dihe_exist',rx_dihe_exist,0,rx_nsurf,1,rx_ndihe)
    call MRMD_ALLOC(filen,procn,'rx_dihe_inv'  ,rx_dihe_inv  ,1,nphi,0,5)
    call MRMD_ALLOC(filen,procn,'rx_dihe_atoms',rx_dihe_atoms,1,rx_ndihe,1,4)
    call MRMD_ALLOC(filen,procn,'rx_dihe_mult' ,rx_dihe_mult ,1,rx_ndihe)
    call MRMD_ALLOC(filen,procn,'rx_dihe_k'    ,rx_dihe_k    ,1,rx_nsurf,1,rx_ndihe)
    call MRMD_ALLOC(filen,procn,'rx_dihe_phi0' ,rx_dihe_phi0 ,1,rx_nsurf,1,rx_ndihe)
    call MRMD_ALLOC(filen,procn,'rx_dihe_cos0' ,rx_dihe_cos0 ,1,rx_nsurf,1,rx_ndihe)
    call MRMD_ALLOC(filen,procn,'rx_dihe_sin0' ,rx_dihe_sin0 ,1,rx_nsurf,1,rx_ndihe)

    ! initialized inverse array
    if (nphi>0) rx_dihe_inv=-1

    ! read data
    if (rx_ndihe>0) then
        ! allocation 
        call MRMD_ALLOC(filen,procn,'tempstring',tempstring,1,2*rx_nsurf)
        
        ! read each dihedral
        do i=1,rx_ndihe
             read(rx_paramfile_unit,*) rx_dihe_atoms(i,1:4),rx_dihe_mult(i),tempstring(1:2*rx_nsurf)

            ! error checking for out of range definitions    
            if (any(rx_dihe_atoms(i,1:4)<1.or.natom<rx_dihe_atoms(i,1:4))) &
            call wrndie(-3,'<mrmd.src> mrmd_init','DIHE atom has to be in range: 1<=iatom<=natom') 

            ! error checking for different atoms
            do j=1,3
               do k=j+1,4
                   if (rx_dihe_atoms(i,j)==rx_dihe_atoms(i,k)) &
                   call wrndie(-3,'<mrmd.src> mrmd_init','DIHE atoms have to be different') 
               enddo
            enddo 
 
            ! frequency check
            if (rx_dihe_mult(i)<1.or.6<rx_dihe_mult(i))  then
                if (WRNLEV>=5) then
                   write(outu,'(A,I3,x,A,4(x,I6))') &
                   'MRMD_INIT> DIHE:',i,'formed by atoms',rx_dihe_atoms(i,1:4)
                   write(outu,*)&
                   'MRMD_INIT> - has a multiplicity of ',rx_dihe_mult(i)   
                endif
                call wrndie(-3,'<mrmd.src> mrmd_init','DIHE multiplicity is out of range 1-6.') 
            endif

            ! duplicate definition is not allowed: A-B-C-D=D-C-B-A 
            ! CHECKING FOR DUPLICATES
            do k=1,i-1
                ! HAVING SAME ATOMS BUT IN INVERSE ORDER SHOULD NOT HAPPEN
                if (all(rx_dihe_atoms(i,1:4)==rx_dihe_atoms(k,(/1,2,3,4/))).or.  &  !abcd
                    all(rx_dihe_atoms(i,1:4)==rx_dihe_atoms(k,(/4,3,2,1/)))) then   !dcba

                    ! DO THEY HAVE THE SAME FREQUENCY
                    if (rx_dihe_mult(i)==rx_dihe_mult(k)) then
                        if (WRNLEV>=5) then
                            write(outu,'(2(A,x,I3,x))')&
                            'MRMD_INIT> DIHE:',i,'and',k
                            write(outu,'(A,4(x,I6))')&
                            'MRMD_INIT> - formed by the same atoms:',rx_dihe_atoms(i,1:4)
                            write(outu,'(A)')&
                            'MRMD_INIT> - in the same or in reverse order' 
                            write(outu,'(A,x,I1)')&
                            'MRMD_INIT> - have the same multiplicity:',rx_dihe_mult(i)
                        endif
                        call wrndie(-3,'<mrmd.src> mrmd_init','Duplicate proper dihedral') 
                    endif
                endif
            enddo    

            ! store values  
            do j=1,rx_nsurf
                ! not present
                if (tempstring(2*j-1)(1:1)=='x'.or.tempstring(2*j-1)(1:1)=='X') then
                    rx_dihe_k   (j,i)=-9999._chm_real   
                    rx_dihe_phi0(j,i)=-9999._chm_real
                    rx_dihe_cos0(j,i)=-9999._chm_real
                    rx_dihe_sin0(j,i)=-9999._chm_real
                    rx_dihe_exist(j,i)=.false.
                else 
                    read(tempstring(2*j-1),*) rx_dihe_k   (j,i)
                    read(tempstring(2*j  ),*) rx_dihe_phi0(j,i)
                    rx_dihe_phi0(j,i)=rx_dihe_phi0(j,i)*degrad ! converting from degree to radian
                    rx_dihe_cos0(j,i) = cos(rx_dihe_phi0(j,i))
                    rx_dihe_sin0(j,i) = sin(rx_dihe_phi0(j,i))
                    rx_dihe_exist(j,i)=.true.
                    ! any k and phi0 values are allowed
                endif
            enddo
        enddo
        call MRMD_ALLOC(filen,procn,'tempstring',tempstring,1,0)

        !----------------------------------------------
        ! identifying it in CHARMM index and vice versa 
        !----------------------------------------------
        !  integer,allocatable,dimension(:,:),save :: &  ! if negative index=> new bond/angle/dihe,
        !                                                                      not in CHARMM
        ! rx_dihe_inv       ! (1:ndihe,1:2) index of CHARMM dihe's 
        !                   !  in MRMD DIHE for 0-5 index increments in CPD(ICP(j)+k)  k=0-5
        ! anything with same set of 4 atoms
        ! MRMD dihedral=> at most 6 CHARMM dihedral: 1,2,3,4,5,6

        do i=1,rx_ndihe
            rx_dihe_exist(0,i)=.false.  ! if one match found at least then setting it .true.
            do j=1,nphi
                ! HAVING SAME ATOMS BUT IN INVERSE ORDER SHOULD NOT HAPPEN
                if (all(rx_dihe_atoms(i,(/1,2,3,4/))==(/IP(j),JP(j),KP(j),LP(j)/)).or.  &   !abcd
                    all(rx_dihe_atoms(i,(/4,3,2,1/))==(/IP(j),JP(j),KP(j),LP(j)/))) then    !dcba

                    ! check various multiplicities
                    k=0
                    do
                        ! Only 1-6 multiplicity is allowed in CHARMM FOR PROPER DIHEDRAL
                        if (abs(CPD(ICP(j)+k))<1.or.6<abs(CPD(ICP(j)+k))) then
                            if (WRNLEV>=5) then
                                write(outu,'(A,I6,x,A,4(x,I6))') &
                                'CHARMM dihedral:',j,'formed by atoms:',IP(j),JP(j),KP(j),LP(j)
                                write(outu,'(A,x,I2)') &
                                'multiplicity:', abs(CPD(ICP(j)+k))
                            endif
                            call wrndie(-3,'<mrmd.src> mrmd_init', &
                            'CHARMM DIHE multiplicity is out of range 1-6.') 
                        endif

                        if (rx_dihe_mult(i)==abs(CPD(ICP(j)+k))) then 
                         !in CHARMM the freq can be negative=> which has no effect on energy/forces
                            rx_dihe_exist(0,i) =.true.
                            rx_dihe_inv(j,k)   =i    
                            
                            ! found => exit
                            exit
                        endif
                        
                        ! is it the highest multiplicity dihedral potential 
                        ! for this proper dihedral angle?
                        ! if multiplicity is positive (1-6) => yes => exit
                        if (CPD(ICP(j)+k)>0) exit
                        
                        ! no=> look for further potential definitions
                        k=k+1
                    enddo
                endif
            enddo 
        enddo
    endif

    !-------------------------------------------------------------------
    ! might be added in the future:
    !      I. CONSISTENCY CHECKING FOR MRMD DIHEDRALS:           
    !      DO THE MRMD-DIHEDRALS HAVE
    !      CORRESPONDING BONDS PRESENT IN CHARMM&MRMD?
    !-------------------------------------------------------------------

    !-------------------------------------------------------------------
    ! might be added in the future:
    !      II. CONSISTENCY CHECKING FOR MRMD BONDS:           
    !      DO THE DIHEDRALS FORMED DUE TO NEW MRMD-BONDS
    !      PRESENT IN CHARMM&MRMD?
    !-------------------------------------------------------------------

    !-------------------------------------------------------------------
    !     CREATE EXCLUSIONS FOR CHARMM DIHEDRALS 
    !     |note multiple definitions (~Fourier-series) possible | 
    !-------------------------------------------------------------------

    ! (re)(de)allocation (if size>0)
    call MRMD_ALLOC(filen,procn,'rx_exc_ndihe',rx_exc_ndihe,1,rx_nsurf)
    call MRMD_ALLOC(filen,procn,'rx_exc_dihe' ,rx_exc_dihe ,1,0,1,0,1,0)

    ! 1. count how many CHARMM dihedrals were redefined in mrmd
    ! they should be removed from CHARMM energy
    ! 2. count how many FURTHER CHARMM dihedrals have been removed due 
    ! to bond removal in MRMD

    !  first cycle(ii=1): to determine array sizes
    ! second cycle(ii=2):  allocation and fill of arrays
    do ii=1,2
        ! CHARMM dihedrals which are deleted or redefined in MRMD DIHE
        ! problem=> for one dihedral multiple frequencies and thus definitions
        ! are possible we should count in a different way
        if (ii==1) then
            if (nphi>0) then
                rx_exc_ndihe=count(rx_dihe_inv(1:nphi,0:5)>0)
            else
                rx_exc_ndihe=0
            endif
        endif    
        !------------------------------------------------------------
        ! CHARMM dihedrals in MRMD => should be removed in all surfaces
        if (ii==2) then

            ! maximum number of exclusions found
            rx_exc_ndihe0=maxval(rx_exc_ndihe)
            
            ! if none, then exit
            if (rx_exc_ndihe0==0) exit

            ! allocating arrays
            call MRMD_ALLOC(filen,procn,'rx_exc_dihe',rx_exc_dihe,&
            1,rx_nsurf,1,rx_exc_ndihe0,1,2)

            rx_exc_dihe=0

            ! count the dihedrals to be removed from CHARMM
            rx_exc_ndihe=0
            do j=1,nphi
                do k=0,5 ! index increment for higher multiplicities
                    ! is it listed (removed,reparameterized) in MRMD?
                    ! if not then exit
                    if (rx_dihe_inv(j,k)<0) exit

                    ! if yes=> store exclusion => increase exclusion counter
                    rx_exc_ndihe(1:rx_nsurf)=rx_exc_ndihe(1:rx_nsurf)+1
                    
                    ! at this moment both rx_exc_ndihe components 
                    ! for all surfaces have the same value
                    rx_exc_dihe(1:rx_nsurf,rx_exc_ndihe(1),1)=j
                    rx_exc_dihe(1:rx_nsurf,rx_exc_ndihe(1),2)=k
                enddo
            enddo
        endif
        
        !--------------------------------------------------------------------
        ! CHARMM dihedral not in MRMD=> should be removed on that surface where 
        ! any of its bonds is removed
        do j=1,nphi

            !------------------------
            ! IT IS NOT IN MRMD DIHE
            !------------------------
            ! check wether any of its bonds has been removed on PES's
            do i=1,rx_nsurf

                ! it is not in MRMD dihe
                ! check wether any of its bond has been removed and not recreated on PES
                do k=1,rx_nbond
                
                    ! if bond present in surface => cannot remove dihedral =>no exclusion
                    if (rx_bond_exist(i,k)) cycle

                    ! if not in CHARMM => cannot affect CHARMM dihedral                    
                    if (.not.rx_bond_exist(0,k)) cycle

                    ! in CHARMM but removed by MRMD from PES
                    ! is it one of the bonds within the dihedral
                    
                    !    (IP-JP, JP-KP, KP-LP)?
                    if (all(rx_bond_atoms(k,1:2)==(/IP(j),JP(j)/)).or.   &
                        all(rx_bond_atoms(k,1:2)==(/JP(j),IP(j)/)).or.   &
                        all(rx_bond_atoms(k,1:2)==(/JP(j),KP(j)/)).or.   &
                        all(rx_bond_atoms(k,1:2)==(/KP(j),JP(j)/)).or.   &
                        all(rx_bond_atoms(k,1:2)==(/KP(j),LP(j)/)).or.   &
                        all(rx_bond_atoms(k,1:2)==(/LP(j),KP(j)/))) then

                        ! check various multiplicities
                        do l=0,5
                            ! is it in MRMD DIHE? => handled already
                            if (rx_dihe_inv(j,l)>0) then

                                ! last multiplicity was found
                                if (CPD(ICP(j)+l)>0) exit

                                ! if not last=> check higher ones                
                                cycle
                            endif

                            ! dihedral was found, which is not removed/reparameterized in MRMD
                            ! but removed due to the removal of one of its bonds
                            ! add exclusion

                            ! rx_exc_ndihe counters here can already be different
                            rx_exc_ndihe(i)=rx_exc_ndihe(i)+1
                            if (ii==2) rx_exc_dihe(i,rx_exc_ndihe(i),1:2)=(/j,l/)
                            
                            if (WRNLEV>=5) then
                                write(outu,'(A,I6)') &
                                'MRMD_INIT> CHARMM-DIHE:',j
                                write(outu,'(A,I1)') &
                                'MRMD_INIT>  - has a multiplicity:',abs(CPD(ICP(j)+l))
                                write(outu,'(A,4(x,I6))') &
                                'MRMD_INIT>  - is formed by atoms:',IP(j),JP(j),KP(j),LP(j)
                                write(outu,'(A,I3)') &
                                'MRMD_INIT>  On surface',i
                                write(outu,'(A,I4)') &
                                'MRMD_INIT>  - it is removed due to removal of HARM&MORS:',k
                                write(outu,'(A,2(x,I6))') &
                                'MRMD_INIT>  - which is between atoms',rx_bond_atoms(k,1:2)
                                write(outu,'(A)') &
                                'MRMD_INIT>  - but it is neither removed nor listed as DIHE'  
                                write(outu,'(A)') &
                                'MRMD_INIT>  - It will be automatically removed by MRMD.'  
                            endif
                            ! last/highest multiplicity was found
                            if (CPD(ICP(j)+l)>0) exit
                        enddo

                        ! removal of dihedral potentials for this four atom with all multiplicities
                        ! have been done on this surface!
                        ! Should not be removed twice due to the removal of another bond of its 
                        ! =>exiting                          
                        exit 
                    endif
                enddo            
            enddo
        enddo !j=1,nphi
    enddo
        
    !------------------------------------------------------------
    !
    ! CREATE INCLUSION LIST FOR MRMD DIHEDRALS
    !
    !------------------------------------------------------------
    rx_inc_ndihe0=0
    if (rx_ndihe>0) rx_inc_ndihe0=maxval(count(rx_dihe_exist(1:rx_nsurf,1:rx_ndihe),2))

    ! (re)(de)allocation (if size>0)
    call MRMD_ALLOC(filen,procn,'rx_inc_ndihe',rx_inc_ndihe,1,rx_nsurf)
    call MRMD_ALLOC(filen,procn,'rx_inc_dihe' ,rx_inc_dihe ,1,rx_nsurf,1,rx_inc_ndihe0)

    ! fill arrays
    rx_inc_ndihe=0
    do i=1,rx_nsurf
        do j=1,rx_ndihe
            if (.not.rx_dihe_exist(i,j)) cycle
            rx_inc_ndihe(i)=rx_inc_ndihe(i)+1
            rx_inc_dihe(i,rx_inc_ndihe(i))=j
        enddo
    enddo

!===========================================================================
!
!   READING PARAMETERS FOR MRMD IMPROPER DIHEDRALS 
!
!===========================================================================
    ! read value 
    if (PRNLEV>=5) write(outu,'(A)') 'MRMD_INIT> Reading IMPR HARM block...'
    read(rx_paramfile_unit,*) keyword1,keyword2,rx_nimpr 
    if (MRMD_UPPERCASE(keyword1)/='IMPR'.and.MRMD_UPPERCASE(keyword2)/='HARM') &
    call wrndie(-3,'<mrmd.src> mrmd_init', 'Missing IMPR HARM block in input')     

    if (rx_nimpr<0) &
    call wrndie(-3,'<mrmd.src> mrmd_init','Number of improper dihedrals cannot be negative !')

    ! deallocate arrays if changing size from nonzero
    call MRMD_ALLOC(filen,procn,'rx_impr_exist',rx_impr_exist,0,rx_nsurf,1,rx_nimpr)
    call MRMD_ALLOC(filen,procn,'rx_impr_inv'  ,rx_impr_inv  ,1,nimphi)
    call MRMD_ALLOC(filen,procn,'rx_impr_atoms',rx_impr_atoms,1,rx_nimpr,1,4)
    call MRMD_ALLOC(filen,procn,'rx_impr_mult' ,rx_impr_mult ,1,rx_nimpr)
    call MRMD_ALLOC(filen,procn,'rx_impr_k'    ,rx_impr_k    ,1,rx_nsurf,1,rx_nimpr)
    call MRMD_ALLOC(filen,procn,'rx_impr_phi0' ,rx_impr_phi0 ,1,rx_nsurf,1,rx_nimpr)
    call MRMD_ALLOC(filen,procn,'rx_impr_cos0' ,rx_impr_cos0 ,1,rx_nsurf,1,rx_nimpr)
    call MRMD_ALLOC(filen,procn,'rx_impr_sin0' ,rx_impr_sin0 ,1,rx_nsurf,1,rx_nimpr)

    ! initialized inverse array
    if (nimphi>0) rx_impr_inv=-1

    ! read data
    if (rx_nimpr>0) then
        ! allocation 
        call MRMD_ALLOC(filen,procn,'tempstring',tempstring,1,2*rx_nsurf)
        
        ! read each improper dihedral
        do i=1,rx_nimpr
             read(rx_paramfile_unit,*) rx_impr_atoms(i,1:4),rx_impr_mult(i),tempstring(1:2*rx_nsurf)

            ! error checking for out of range definitions    
            if (any(rx_impr_atoms(i,1:4)<1.or.natom<rx_impr_atoms(i,1:4))) &
                call wrndie(-3,'<mrmd.src> mrmd_init',                              &
                'IMPR atom has to be in range: 1<=iatom<=natom') 

            ! error checking for different atoms
            do j=1,3
               do k=j+1,4
                   if (rx_impr_atoms(i,j)==rx_impr_atoms(i,k))     &
                   call wrndie(-3,'<mrmd.src> mrmd_init',                   &
                   'IMPR atoms have to be different') 
               enddo
            enddo 

            ! frequency check 0
            if (rx_impr_mult(i)/=0)  then
                if (WRNLEV>=5) then
                    write(outu,'(A,x,I3)') &
                    'MRMD_INIT> IMPR',i
                    write(outu,'(A,4(x,I6))') &
                    'MRMD_INIT> - formed by atoms:',rx_impr_atoms(i,1:4)
                    write(outu,'(A,x,I2)') &
                    'MRMD_INIT> - its multiplicity :',rx_impr_mult(i)   
                endif
                call wrndie(-3,'<mrmd.src> mrmd_init',           &
                   'IMPR multiplicity can be 0 only') 
            endif

            !----------------------------------------------------------------    
            ! CHECKING FOR DUPLICATES AND CONTRADICTIONS
            !-------------------------------------------------
            ! non-independent improper dihedrals with same connectivity
            !----------------------------------------------------
            !      B       potential form:A-B-C-D torsional potential 
            !       \                                                 !    
            !        A--C         
            !       /
            !      D 
            ! A is fixed due to connectivity
            ! duplicate definition is not allowed: A-B-C-D=D-C-B-A 
            !----------------------------------------------------
            do k=1,i-1
                ! abcd=dcba
                if (all(rx_impr_atoms(i,1:4)==rx_impr_atoms(k,(/1,2,3,4/))).or.  &
                    all(rx_impr_atoms(i,1:4)==rx_impr_atoms(k,(/4,3,2,1/)))) then
                    ! DUPLICATE
                    if (WRNLEV>=5) then
                        write(outu,'(2(A,x,I3,x))') &
                        'MRMD_INIT> IMPR:',i,'and',k
                        write(outu,'(A,4(x,I6))') &
                        'MRMD_INIT> - have the same atoms:',rx_impr_atoms(i,1:4)
                        write(outu,'(A)') &
                        'MRMD_INIT> - in the same or reverse order!' 
                    endif
                    call wrndie(-3,'<mrmd.src> mrmd_init','Duplicate improper dihedral') 
                endif
            enddo    

            ! store values  
            do j=1,rx_nsurf
                ! new notation for deletion => if force constant string starts with x or X
                if (tempstring(2*j-1)(1:1)=='x'.or.tempstring(2*j-1)(1:1)=='X') then
                    rx_impr_k   (j,i)=-9999._chm_real
                    rx_impr_phi0(j,i)=-9999._chm_real
                    rx_impr_cos0(j,i)=-9999._chm_real
                    rx_impr_sin0(j,i)=-9999._chm_real
                    rx_impr_exist(j,i)=.false.
                else 
                    read(tempstring(2*j-1),*) rx_impr_k   (j,i)
                    read(tempstring(2*j  ),*) rx_impr_phi0(j,i)
                    rx_impr_phi0(j,i)=rx_impr_phi0(j,i)*degrad ! converting from degree to radian
                    rx_impr_cos0(j,i) = cos(rx_impr_phi0(j,i))
                    rx_impr_sin0(j,i) = sin(rx_impr_phi0(j,i))
                    rx_impr_exist(j,i)=.true.
                    ! any k and phi0 values are allowed
                 endif
            enddo
        enddo
        ! deallocation 
        call MRMD_ALLOC(filen,procn,'tempstring',tempstring,1,0)

        !----------------------------------------------
        ! identifying it in CHARMM index and vice versa 
        !----------------------------------------------
        ! rx_impr_inv       ! (1:nimpr) index of CHARMM impr's in MRMD IMPR
        ! anything with same set of 4 atoms
        ! MRMD improper dihedral=> at most 7 CHARMM improper dihedral:0 or 1,2,3,4,5,6
        do i=1,rx_nimpr
            rx_impr_exist(0,i)=.false.  ! if one match found at least then setting it .true.
            do j=1,nimphi
                !------------------------------------------------------
                ! HAVING SAME OR REVERSE ORDER => same torsion angle
                ! ABCD<=>DCBA
                !------------------------------------------------------            
                ! looking for same dihe
                if (all(rx_impr_atoms(i,1:4)==(/IM(j),JM(j),KM(j),LM(j)/)).or.  &
                    all(rx_impr_atoms(i,1:4)==(/LM(j),KM(j),JM(j),IM(j)/))) then 

                    ! different frequency??
                    if (abs(rx_impr_mult(i))==abs(CID(ICI(j)))) then
                        !in CHARMM the freq can be negative=> which has no effect on energy/forces
                        rx_impr_exist(0,i)=.true.
                        rx_impr_inv(j)=i 
                    endif
                endif
            enddo 
        enddo
    endif

    !-------------------------------------------------------------------
    ! might be added in the future:
    !      I. CONSISTENCY CHECKING FOR MRMD IMPROPER DIHEDRALS:           
    !      DO THE MRMD IMPROPER DIHEDRALS HAVE
    !      CORRESPONDING BONDS PRESENT IN CHARMM&MRMD?
    !-------------------------------------------------------------------

    !-------------------------------------------------------------------
    ! might be added in the future:
    !      II. CONSISTENCY CHECKING FOR MRMD IMPROPER BONDS:           
    !      DO THE IMPROPER DIHEDRALS FORMED DUE TO NEW MRMD-BONDS
    !      PRESENT IN CHARMM&MRMD?
    !-------------------------------------------------------------------

!===========================================================================
!     CREATE EXCLUSIONS FOR CHARMM IMPROPER DIHEDRALS 
!     |note multiple definitions (~Fourier-series) possible | 
!===========================================================================

    ! (re)(de)allocation (if size>0)
    call MRMD_ALLOC(filen,procn,'rx_exc_nimpr',rx_exc_nimpr,1,rx_nsurf)
    call MRMD_ALLOC(filen,procn,'rx_exc_impr' ,rx_exc_impr ,1,0,1,0)

    ! 1. count how many CHARMM improper dihedrals were redefined in mrmd
    ! they should be removed from CHARMM energy
    ! 2. count how many FURTHER CHARMM improper dihedrals have been removed due 
    ! to bond removal in MRMD

    !  first cycle(ii=1): to determine array sizes
    ! second cycle(ii=2):  allocation and fill of arrays
    do ii=1,2
        ! CHARMM improper dihedrals which are deleted or redefined in MRMD IMPR
        ! problem


        ! redefinition means=changing force constant and phase, but not the frequency
        if (ii==1) then
            if (nimphi>0) then
                rx_exc_nimpr=count(rx_impr_inv(1:nimphi)>0)
            else
                rx_exc_nimpr=0
            endif
        endif

        ! CHARMM improper dihedrals in MRMD => should be removed in all surfaces
        if (ii==2) then

            ! maximum number of exclusions found
            rx_exc_nimpr0=maxval(rx_exc_nimpr)
            
            ! if none, then exit
            if (rx_exc_nimpr0==0) exit

            ! allocating arrays
            call MRMD_ALLOC(filen,procn,'rx_exc_impr',rx_exc_impr,1,rx_nsurf,1,rx_exc_nimpr0)

            ! count the improper dihedrals to be removed from CHARMM
            rx_exc_nimpr=0
            do j=1,nimphi
                if (rx_impr_inv(j)>0) then
                    ! increase counter
                    rx_exc_nimpr(1:rx_nsurf)=rx_exc_nimpr(1:rx_nsurf)+1
                    
                    ! at this moment all rx_exc_nimpr component has the same value
                    rx_exc_impr(1:rx_nsurf,rx_exc_nimpr(1))=j
                endif
            enddo
        endif
        
        !--------------------------------------------------------------------
        ! CHARMM improper dihedral not in MRMD=> should be removed on that surface where 
        ! any of its bonds is removed
        do j=1,nimphi
            ! is it in MRMD IMPR? => handled already
            if (rx_impr_inv(j)>0) cycle

            !------------------------
            ! IT IS NOT IN MRMD IMPR
            !------------------------

            ! check wether any of its bonds has been removed on PES's
            do i=1,rx_nsurf

                ! it is not in MRMD IMPR
                ! check wether any of its bond has been removed and not recreated on PES
                do k=1,rx_nbond
                
                    ! if bond present in surface => cannot remove improper dihedral =>no exclusion
                    if (rx_bond_exist(i,k)) cycle

                    ! if not in CHARMM => cannot affect CHARMM improper dihedral                    
                    if (.not.rx_bond_exist(0,k)) cycle

                    ! in CHARMM but removed by MRMD from PES
                    ! is it one of the bonds within the improper dihedral

                    ! improper (IM-JM, IM-KM, IM-LM)?
                    if (all(rx_bond_atoms(k,1:2)==(/IM(j),JM(j)/)).or.   &
                        all(rx_bond_atoms(k,1:2)==(/JM(j),IM(j)/)).or.   &
                        all(rx_bond_atoms(k,1:2)==(/IM(j),KM(j)/)).or.   &
                        all(rx_bond_atoms(k,1:2)==(/KM(j),IM(j)/)).or.   &
                        all(rx_bond_atoms(k,1:2)==(/IM(j),LM(j)/)).or.   &
                        all(rx_bond_atoms(k,1:2)==(/LM(j),IM(j)/))) then
                       
                        ! rx_exc_nimpr counters here can already be different
                        rx_exc_nimpr(i)=rx_exc_nimpr(i)+1
                        if (ii==2) rx_exc_impr(i,rx_exc_nimpr(i))=j
                        
                        if (WRNLEV>=5) then
                            write(outu,'(A,x,I6)')&
                            'MRMD_INIT> CHARMM IMPR',j
                            write(outu,'(A,4(x,I6))')&
                            'MRMD_INIT> - is formed by atoms:',IM(j),JM(j),KM(j),LM(j)
                            write(outu,'(A,x,I2)')&
                            'MRMD_INIT> On surface',i
                            write(outu,'(A,I4)')&
                            'MRMD_INIT> - it is removed due to removal of HARM/MORS',k
                            write(outu,'(A,2(x,I6))')&
                            'MRMD_INIT>   between atoms:',rx_bond_atoms(k,1:2)
                            write(outu,'(A)') &
                            'MRMD_INIT> - but it is neither removed nor listed as IMPR.'  
                            write(outu,'(A)') &
                            'MRMD_INIT> - It will be automatically removed by MRMD.'  
                        endif
                        exit ! already has been removed on this surface
                             ! should not be removed twice by its another removed bond =>exiting      
                    endif
                enddo            
            enddo
        enddo
    enddo
        
!===========================================================================
! CREATE INCLUSION LIST FOR MRMD IMPROPER DIHEDRALS
!===========================================================================

    rx_inc_nimpr0=0
    if (rx_nimpr>0) rx_inc_nimpr0=maxval(count(rx_impr_exist(1:rx_nsurf,1:rx_nimpr),2))

    ! (re)(de)allocation (if size>0)
    call MRMD_ALLOC(filen,procn,'rx_inc_nimpr',rx_inc_nimpr,1,rx_nsurf)
    call MRMD_ALLOC(filen,procn,'rx_inc_impr' ,rx_inc_impr ,1,rx_nsurf,1,rx_inc_nimpr0)

    ! fill arrays
    rx_inc_nimpr=0
    do i=1,rx_nsurf
        do j=1,rx_nimpr
            if (.not.rx_impr_exist(i,j)) cycle
            rx_inc_nimpr(i)=rx_inc_nimpr(i)+1
            rx_inc_impr(i,rx_inc_nimpr(i))=j
        enddo
    enddo


!===========================================================================
!
! write report on parameters read
!
!===========================================================================
    if (PRNLEV>=5) then
        !----------------------------------------
        ! NUMBER OF RECORD 
        !----------------------------------------
        write(outu,'(''MRMD_INIT>'',80(''=''))')
        write(outu,'(A)') 'MRMD_INIT> Number of records'
        write(outu,'(20(''MRMD_INIT> '',4(x,A11,x,I4)/))')         &
        'SURF:',rx_nsurf,     'SWCH:'     ,rx_nswch,'SHAP GAPO:',rx_ngapo,'NBON ATOM:',rx_natom, &
        'NBON GVDW:',rx_ngvdw,'BOND HARM:',rx_nharm,'BOND MORS:',rx_nmors,'ANGL HARM:',rx_nangl, & 
        'DIHE FOUR:',rx_ndihe,'IMPR HARM:',rx_nimpr        
    endif    
    
    !---------------------
    ! OUTPUT ON PARAMETERS
    !---------------------
    if (PRNLEV>=6) then                        
        write(outu,'(''MRMD_INIT>'',80(''=''))')  
        !---------------------
        ! Surfaces and shifts
        !---------------------
        write(outu,'(A)') 'MRMD_INIT> SURF# level_shift'
        do i=1, rx_nsurf
           write(outu,'(A,x,I5,x,F8.3)') 'MRMD_INIT> ',i,rx_levelshift(i)
        enddo

        !---------------------
        ! Switching rules
        !---------------------
        write(outu,'(A)') &
        'MRMD_INIT> SWCH# Surf1 Surf2 method parameters...'
        write(outu,'(A,I5,2(x,A5),x,A10,x,G12.5)') &
        'MRMD_INIT> ',1,'all','all',trim(rx_swch_func),rx_swch_dE

        !---------------------
        ! Gaussian*polynomials
        !---------------------
        if (rx_ngapo>0) then    
            write(outu,'(A)') &
            'MRMD_INIT> GAPO# Surf1 Surf2  Exp_shift    Exp_std  Poly_coeff''s:a0,a1,a2,...'
            do i=1,rx_ngapo
                write(outu,'(A,I5,2(x,I5),10(x,G12.5))') &
                'MRMD_INIT> ',i,rx_gapo_surfs(i,1:2),rx_gapo_coefs(i,-2:rx_gapo_order(i)) 
            enddo
        endif

        do j=1, rx_nsurf
            ! Standard nonbonded parameters for atoms
            write(outu,'(''MRMD_INIT>'',80(''=''))')  
            write(outu,'(A,x,I2)') &
            'MRMD_INIT>  Parameters read for surface',j
            write(outu,'(''MRMD_INIT>'',80(''-''))')  
            if (rx_natom>0) then
              ! header  
              write(outu,'(A)')&
               'MRMD_INIT> NBON ATOM#  Atom        Chg       Eps1     Sigma1       Eps2     Sigma2'
              do i=1,rx_natom   
                write(outu,'(A,5x,I5,x,I5,5(x,F10.4))') 'MRMD_INIT> ',i,rx_atom(i),rx_charges(j,i),&
                rx_vdw_eps(j,i),rx_vdw_rmin_half(j,i),rx_vdw_eps2(j,i),rx_vdw_rmin_half2(j,i) 
              enddo
              write(outu,'(''MRMD_INIT>'',80(''-''))')  
            endif

            ! General van der Waals
            if (rx_ngvdw>0) then     
                write(outu,'(A)') &
                'MRMD_INIT> NBON GVDW# Atom1 Atom2       Eps      Rmin    ExpAtt    ExpRep' 
                do i=1,rx_ngvdw
                  if (rx_gvdw_exist(j,i)) then
                    write(outu,'(A,5x,I5,2(x,I5),4(x,F9.3))') 'MRMD_INIT> ',i,rx_gvdw_atoms(i,1:2),&
                    rx_gvdw_eps(j,i), rx_gvdw_rmin(j,i),rx_gvdw_xatt(j,i),rx_gvdw_xrep(j,i) 
                  else 
                    write(outu,'(A,5x,I5,2(x,I5),4(x,A9))') 'MRMD_INIT> ',i,rx_gvdw_atoms(i,1:2),&
                    ('    -    ',k=1,4) 
                  endif
                enddo
                write(outu,'(''MRMD_INIT>'',80(''-''))')  
            endif

            ! Harmonic bonds
            if (rx_nharm>0) then     
                write(outu,'(A)') 'MRMD_INIT> BOND HARM# Atom1 Atom2      FC/2       Req  '
                do i=1,rx_nharm
                  if (rx_harm_exist(j,i)) then
                    write(outu,'(A,5x,I5,2(x,I5),2(x,F9.3))') &
                    'MRMD_INIT> ',i,rx_bond_atoms(i,1:2),rx_harm_fc_half(j,i),rx_harm_re(j,i) 
                  else 
                    write(outu,'(A,5x,I5,2(x,I5),2(x,A9))') 'MRMD_INIT> ',i,rx_bond_atoms(i,1:2),&
                    ('    -    ',k=1,2) 
                  endif
                enddo
                write(outu,'(''MRMD_INIT>'',80(''-''))')  
            endif

            ! Morse bonds
            if (rx_nmors>0) then   
                write(outu,'(A)') 'MRMD_INIT> BOND MORS# Atom1 Atom2        De       Req      beta'
                do i=1,rx_nmors
                  if (rx_bond_exist(j,i+rx_nharm)) then
                    write(outu,'(A,5x,I5,2(x,I5),3(x,F9.3))') 'MRMD_INIT> ', &
                    i,rx_bond_atoms(rx_nharm+i,1:2), rx_mors_De(j,i),rx_mors_re(j,i),&
                    rx_mors_beta(j,i) 
                  else
                    write(outu,'(A,5x,I5,2(x,I5),3(x,A9))') 'MRMD_INIT> ',&
                    i,rx_bond_atoms(rx_nharm+i,1:2),('    -    ',k=1,3) 
                  endif
                enddo 
                write(outu,'(''MRMD_INIT>'',80(''-''))')  
            endif
            
            ! RKHS bonds
            if (rx_nrkhs>0) then   
                write(outu,'(A)') 'MRMD_INIT> BOND RKHS# Atom1 Atom2'
                do i=1,rx_nrkhs
                  if (rx_bond_exist(j,i+rx_nharm+rx_nmors)) then
                    write(outu,'(A,5x,I5,2(x,I5))') 'MRMD_INIT> ', &
                    i,rx_bond_atoms(rx_nharm+rx_nmors+i,1:2)
                  else
                    write(outu,'(A,5x,I5,2(x,I5))') 'MRMD_INIT> ',&
                    i,rx_bond_atoms(rx_nharm+rx_nmors+i,1:2)
                  endif
                enddo 
                write(outu,'(''MRMD_INIT>'',80(''-''))')  
            endif

            ! Angle potentials
            if (rx_nangl>0) then 
                write(outu,'(A)') &
                'MRMD_INIT> ANGL HARM# Atom1 Atom2 Atom3      FC/2     PHIeq  UreyFC/2   UreyReq'
                do i=1,rx_nangl
                  if (rx_angl_exist(j,i)) then
                    write(outu,'(A,5x,I5,3(x,I5),4(x,F9.3))') &
                    'MRMD_INIT> ',i,rx_angl_atoms(i,1:3),            &
                    rx_angl_fc_half(j,i),rx_angl_phie(j,i)*raddeg, &
                    rx_urey_fc_half(j,i),rx_urey_re(j,i) 
                  else
                    write(outu,'(A,5x,I5,3(x,I5),4(x,A9))') 'MRMD_INIT> ',i, &
                    rx_angl_atoms(i,1:3),('    -    ',k=1,4) 
                  endif
                enddo 
                write(outu,'(''MRMD_INIT>'',80(''-''))')  
            endif 

            ! Proper dihedrals
            if (rx_ndihe>0) then
                write(outu,'(A)') &
                'MRMD_INIT> DIHE FOUR# Atom1 Atom2 Atom3 Atom4 Mult     Ampli      Phi0'
                do i=1,rx_ndihe
                  if (rx_dihe_exist(j,i)) then
                    write(outu,'(A,5x,I5,4(x,I5),x,I4,2(x,F9.3))') 'MRMD_INIT> ',i, &
                    rx_dihe_atoms(i,1:4),rx_dihe_mult(i),rx_dihe_k(j,i),rx_dihe_phi0(j,i)*raddeg
                  else
                    write(outu,'(A,5x,I5,4(x,I5),x,2(x,A9))') 'MRMD_INIT> ',i,rx_dihe_atoms(i,1:4),&
                    ('    -    ',k=1,2)
                  endif
                enddo
                write(outu,'(''MRMD_INIT>'',80(''-''))')  
            endif

            ! Improper dihedrals
            if (rx_nimpr>0) then
                write(outu,'(A)') &
                'MRMD_INIT> IMPR HARM# Atom1 Atom2 Atom3 Atom4 Mult      FC/2      Phi0'
                do i=1,rx_nimpr
                  if (rx_impr_exist(j,i)) then
                    write(outu,'(A,5x,I5,4(x,I5),x,I4,2(x,F9.3))') 'MRMD_INIT> ',i,  &
                    rx_impr_atoms(i,1:4),rx_impr_mult(i),rx_impr_k(j,i),rx_impr_phi0(j,i)*raddeg
                  else
                    write(outu,'(A,5x,I5,4(x,I5),2(x,A9))') 'MRMD_INIT> ',i,rx_impr_atoms(i,1:4), &
                    ('    -    ',k=1,2)
                  endif
                enddo
            endif
        enddo
        write(outu,'(''MRMD_INIT>'',80(''=''))')  
    endif

!===========================================================================
!
! EXCLUSION AND INCLUSION LIST FOR NONBONDED INTERACTIONS
! AT VARIOUS NBXMOD LEVELS (NBXMOD=1,2,3,4,5)
!
! exclusion: 
!    which 1-2, 1-3, 1-4, 1-(>4) neighbourships has disappeared 
!    due to MRMD HARM/MORS adding/removal
!
! inclusion: 
!    which new 1-2, 1-3, 1-4, 1-(>4) neighbourships appeared 
!    due to MRMD HARM/MORS adding/removal
!
! Idea: 
! reference:build 1-2,1-3,1-4,1-(>4) neighbourships lists for fixed bonds 
!                                   (in CHARMM and not listed in MRMD)
!       build 1-2,1-3,1-4,1-(>4) neighbourships lists for CHARMM (add CHARMM bonds listed in MRMD)
!       build 1-2,1-3,1-4,1-(>4) neighbourships lists for each PES in MRMD (add all MRMD bonds)
!       and find the differences between the list of CHARMM and PES's!
!
!===========================================================================

    ! (re)(de)allocation (if size>0) of arrays for exclusions and inclusions 
    call MRMD_ALLOC(filen,procn,'rx_exc_n12',rx_exc_n12,0,rx_nsurf)
    call MRMD_ALLOC(filen,procn,'rx_exc_n13',rx_exc_n13,0,rx_nsurf)
    call MRMD_ALLOC(filen,procn,'rx_exc_n14',rx_exc_n14,0,rx_nsurf)
    call MRMD_ALLOC(filen,procn,'rx_exc_n15',rx_exc_n15,0,rx_nsurf)
    call MRMD_ALLOC(filen,procn,'rx_inc_n12',rx_inc_n12,0,rx_nsurf)
    call MRMD_ALLOC(filen,procn,'rx_inc_n13',rx_inc_n13,0,rx_nsurf)
    call MRMD_ALLOC(filen,procn,'rx_inc_n14',rx_inc_n14,0,rx_nsurf)
    call MRMD_ALLOC(filen,procn,'rx_inc_n15',rx_inc_n15,0,rx_nsurf)

    ! reseting variables
    rx_exc_n12(0:rx_nsurf)  = 0
    rx_exc_n13(0:rx_nsurf)  = 0
    rx_exc_n14(0:rx_nsurf)  = 0
    rx_exc_n15(0:rx_nsurf)  = 0
    rx_inc_n12(0:rx_nsurf)  = 0
    rx_inc_n13(0:rx_nsurf)  = 0
    rx_inc_n14(0:rx_nsurf)  = 0
    rx_inc_n15(0:rx_nsurf)  = 0

!-----------------------------------------------------------
!
! GENERATE UNIFIED LIST OF CHARMM BONDS AND ALL BONDS IN MRMD
! making unified list:
!           1...bond_fix: bonds in CHARMM not listed in MRMD
!  bond_fix+1...bond_all: all bonds listed (removed/added/reparameterized) in MRMD
!
!-----------------------------------------------------------
    ! total number of CHARMM bonds which are not mentioned in MRMD parameter file
    nbond_fix=nbond-count(     rx_bond_exist(0,1:rx_nbond))  ! FIX BONDS = CHARMM - MRMD_IN_CHARMM     

    ! total number of different bonds in CHARMM+all MRMD surfaces
    nbond_all=nbond+count(.not.rx_bond_exist(0,1:rx_nbond))  ! ALL BONDS = CHARMM+MRMD_NOT_IN_CHARMM      

    ! (re)(de)allocation (if size>0)
    call MRMD_ALLOC(filen,procn,'bond_all',bond_all,0,nbond_all,1,2)

    ! filling bond_all
    ! bonds only in CHARMM, so not changed by mrmd
    nbond_all=0
    do i=1,nbond  ! going through CHARMM bonds
        if (rx_bond_inv(i)>0) cycle  ! if in mrmd then skip
        nbond_all=nbond_all+1
        bond_all(nbond_all,1:2)=(/IB(i),JB(i)/)
    enddo
    
    ! bonds in mrmd
    if (rx_nbond>0) then  ! bonds in mrmd
        bond_all(nbond_all+1:nbond_all+rx_nbond,1:2)=rx_bond_atoms(1:rx_nbond,1:2)
        nbond_all=nbond_all+rx_nbond
    endif


!-----------------------------------------------------------------------------
! FIND ALL ATOMS WITHIN 3-BOND DISTANCE FROM 
! - MRMD ATOMS :THESE WILL HAVE UPDATED charge and vdw parameter 
! - ATOMS OF MRMD GVDW
!    =>OTHER ATOMS FARTHER AWAY WILL HAVE 1-(>4) RELATIONSHIP
! - ATOMS OF MRMD BONDS: ONLY THESE ATOMS ARE AFFECTED REGARDING 1-2,1-3,1-4 
!                         NEIGHBOURSHIPS WHEN MRMD BONDS ARE FORMED AND BROKEN 
!    =>OTHER ATOMS WILL HAVE 1-(>4) RELATIONSHIP 
!-------------------------------------------------------------------------------
    call MRMD_ALLOC(filen,procn,'atom_pos',atom_pos,1,natom)

                                 ! HARM/MORS  atoms     
    atom_pos=max(5,natom+1)      ! assume everything is disconnected=>1-5 distance
 
    ! ATOMS LISTED IN MRMD ATOM
    do i=1,rx_natom
         atom_pos(rx_atom(i))=1           ! assign 1 for atoms with updated nonbonded parameters
    enddo

    ! ATOMS LISTED IN GENERAL VDW
    do i=1,rx_ngvdw
        atom_pos(rx_gvdw_atoms(i,1:2))=1
    enddo

    ! ATOMS OF MRMD BONDS
    do i=1,rx_nbond
     atom_pos(rx_bond_atoms(i,1:2))=1 ! assign 1 for atoms with mrmd bonds
    enddo

    !-----------------------------------------------------------
    ! Find distances of atoms within 1-2,1-3,1-4 distance from affected 
    ! atoms [MRMD ATOM, MRMD GVDW, MRMD HARM&MORS]
    ! iterative determination - first-breadth search        
    ! considering all CHARMM/MRMD bonds present at the same time!
    !-----------------------------------------------------------
    do j=1,3
        changed=.false.
        do i=1,nbond_all
            ! shorter variables names
            ib1=bond_all(i,1)
            ib2=bond_all(i,2)

            ! check bonds with an atom being 1-j distance from mrmd atoms
            !              and the another atom is 1-(j+2) or farther away     
            if (atom_pos(ib1)==j.and.atom_pos(ib2)>j+1) then
                atom_pos(ib2)=j+1
                changed=.true.
                cycle
            endif
            if (atom_pos(ib2)==j.and.atom_pos(ib1)>j+1) then
                atom_pos(ib1)=j+1
                changed=.true.
                cycle
            endif
        enddo

        ! if no improvement in index=> there won't be any later
        ! as all atoms with index larger than j have the same index = max(5,natom+1)
        ! so the =j.and. >j+1 conditions cannot be fulfilled 
        if (.not.changed) exit 
    enddo

    !-----------------------------------------------------------
    !
    ! index of atoms within 1-4 distance from directly affected 
    ! atoms [MRMD ATOM, MRMD GVDW, MRMD HARM&MORS]
    ! these are affected during mrmd bond-breakage/formation
    ! regarding change in FF interactions
    !
    !-----------------------------------------------------------
    rx_natom1234=count(atom_pos<=4)
     
    ! need to be allocated even if rx_natom1234=0
    call MRMD_ALLOC(filen,procn,'rx_atom1234_inv',rx_atom1234_inv,1,natom)
    rx_atom1234_inv =-1

    call MRMD_ALLOC(filen,procn,'rx_atom1234',rx_atom1234,1,rx_natom1234)

    if (rx_natom1234>0) then
        rx_natom1234    = 0
        do i=1, natom
           if (atom_pos(i)<=4) then
              rx_natom1234              =   rx_natom1234+1
              rx_atom1234(rx_natom1234) =   i  
              rx_atom1234_inv(i)        =   rx_natom1234
           endif
        enddo
    endif

    ! deallocate
    call MRMD_ALLOC(filen,procn,'atom_pos',atom_pos,1,0)
     
    !-----------------------------------------------------------
    ! Index of bonds in bond_all list which 
    ! are between atoms of list rx_natom1234.
    ! These bonds determine the distance in bond
    ! units between the atoms of list rx_natom1234.
    !-----------------------------------------------------------
    nbond1234=0  
    if (rx_natom1234>0) then
        call MRMD_ALLOC(filen,procn,'bond1234'    ,bond1234    ,1,nbond_all)
!       call MRMD_ALLOC(filen,procn,'bond1234_inv',bond1234_inv,1,nbond_all)
 
        do i=1, nbond_all
            ! shorter variables names
            ib1=bond_all(i,1)
            ib2=bond_all(i,2)        
            ! bonds with both atoms in the atomlist within 1-4 distance
            ! from mrmd atoms
            if (any(ib1==rx_atom1234).and.any(ib2==rx_atom1234)) then
               nbond1234              =   nbond1234+1
               bond1234(nbond1234) =   i  
!              bond1234_inv(i)        =   nbond1234
            endif
        enddo
    endif
    call MRMD_ALLOC(filen,procn,'atom_pos',atom_pos,1,0)

    !-----------------------------------------------------------
    ! all atom-atom relative position matrix based
    ! use k=-1 for relative positions based on common/fixed bonds (only in CHARMM)  
    ! use k=0  for relative positions based on all CHARMM bonds  
    ! use k=1  for relative positions based on bonds in the investigated surface 
    !                                         (overwritten for each new surface)
    !-----------------------------------------------------------
    call MRMD_ALLOC(filen,procn,'atom_pos2',atom_pos2,-1,1,1,rx_natom1234,1,rx_natom1234)

    ! initial values of relative positions
    do i=1,rx_natom1234
        atom_pos2(-1,i,i)=1    ! diagonal: all atom is 1-1 relative position to itself
        do j=i+1,rx_natom1234  ! initial assumption:none of them are connected 
            atom_pos2(-1,i,j)=max(5,natom+1)  ! upper triangle
            atom_pos2(-1,j,i)=max(5,natom+1)  ! lower triangle (same) 
        enddo
    enddo

    !----------------------------------------------------------------
    ! Allocating and reseting temporary memory for exclusion/inclusion
    ! The array sizes are temporarily set to the maximum possible number
    ! When the exact array sizes are evaluated this data is transfered 
    ! to rx_exc_12,rx_exc_13,rx_exc_14,rx_inc_12,rx_inc_13,rx_inc_14
    !----------------------------------------------------------------
    ! integer divisions, per 2 has to come last !
    max_n12=nbond1234
    max_n13=min(max(0,rx_natom1234-1)*(rx_natom1234-2)/2,      nbond1234   *(nbond1234-1)/2)  
    max_n14=min(max(0,rx_natom1234-2)*(rx_natom1234-3)/2,max(0,nbond1234-1)*(nbond1234-2)/2) 
    max_n15=rx_natom1234*(rx_natom1234-1)/2  

    if (max_n12>0) then
        call MRMD_ALLOC(filen,procn,'tmp_exc_12',tmp_exc_12,0,rx_nsurf,1,max_n12,1,2)
        call MRMD_ALLOC(filen,procn,'tmp_inc_12',tmp_inc_12,0,rx_nsurf,1,max_n12,1,3)
        tmp_exc_12=0            
        tmp_inc_12=0            
    endif
    
    if (max_n13>0) then
        call MRMD_ALLOC(filen,procn,'tmp_exc_13',tmp_exc_13,0,rx_nsurf,1,max_n13,1,2)
        call MRMD_ALLOC(filen,procn,'tmp_inc_13',tmp_inc_13,0,rx_nsurf,1,max_n13,1,3)
        tmp_exc_13=0            
        tmp_inc_13=0            
    endif
    
    if (max_n14>0) then
        call MRMD_ALLOC(filen,procn,'tmp_exc_14',tmp_exc_14,0,rx_nsurf,1,max_n14,1,2)
        call MRMD_ALLOC(filen,procn,'tmp_inc_14',tmp_inc_14,0,rx_nsurf,1,max_n14,1,3)
        tmp_exc_14=0            
        tmp_inc_14=0            
    endif

    if (max_n15>0) then
        call MRMD_ALLOC(filen,procn,'tmp_exc_15',tmp_exc_15,0,rx_nsurf,1,max_n15,1,2)
        call MRMD_ALLOC(filen,procn,'tmp_inc_15',tmp_inc_15,0,rx_nsurf,1,max_n15,1,3)
        tmp_exc_15=0            
        tmp_inc_15=0            
    endif

    do isurf=-1,rx_nsurf
        if (rx_natom1234<=0) exit
    
        ! storing isurf=-1 separately
        ! storing isurf= 0 separately 
        !         isurf>0  overwritten for each PES
        isurfx=min(isurf,1)
        ! For each surface (isurf>=0) as a starting point 
        ! transfer connectivities determined for common bonds(1-2,1-3,1-4)
        ! in cycle isurf=-1
        ! 
        ! Compared to that new bonds are formed, but none gets broken. 
        ! (it can save a lot of time)
        if (isurfx>-1) atom_pos2(isurfx,:,:)=atom_pos2(-1,:,:)

        ! First, find 1-2 relative positions based on common bonds 
        ! then also using additional bonds present in each surface
        ! write 2 into matrix elements for atoms with 1-2 position
        do j=1,nbond1234
          ! index in the bond_all list
          i=bond1234(j)
                  
          ! no more common bond=> exit
          if (i>nbond_fix) then
              ! only common bonds for the COMMON-BONDS SURFACE  


              if (isurf==-1) exit 
              ! FOR CHARMM isurf=0 => if it isn't in CHARMM =>skip 
              ! FOR PES isurf>0 => if it isn't present in the PES =>skip 
              if (.not.rx_bond_exist(isurf,i-nbond_fix)) cycle
          endif

          ! shorter notation for atom index in short list  
          ib1=rx_atom1234_inv(bond_all(i,1))
          ib2=rx_atom1234_inv(bond_all(i,2)) 
            
          ! UPDATE CLOSEST DISTANCE           
          ! for isurf=0 <=> common bonds => use l=0 array
          ! for isurf>0 <=> each PES     => use l=1 array
          atom_pos2(isurfx,ib1,ib2)=2 
          atom_pos2(isurfx,ib2,ib1)=2 
        enddo

        ! connections 1-2 have already been found
        ! now finding 1-3,1-4 relative positions 
        do i=3,4
          do j=1,rx_natom1234-1
            do k=j+1,rx_natom1234
               ! current shortest relative position of atoms j,k 
               m=atom_pos2(isurfx,j,k)  

               ! are they closer than what is currently updated
               if (m<=i) cycle
                  
               ! shortest route between j and k is obtained:
               ! - take the current minimum distances(+1) of atoms j and k from each atom 
               !   and add them up
               ! - and find the minimum of them
               ! - and subtract one to get the minimal distance of atoms j and k
               n=minval(sum(atom_pos2(isurfx,(/j,k/),:),1))-1   
               ! if a route found between atom j and k
               ! which has a length i and it is shorter 
               ! than their current relative position
               if (n<=i.and.n<m) then
                    atom_pos2(isurfx,j,k) = n
                    atom_pos2(isurfx,k,j) = n
               endif                    
            enddo
          enddo
        enddo
        
        ! inclusion and exclusion lists will be generated 
        ! for each PES (isurf>0) 
        ! compared to CHARMM (isurf=0) neighbourship list.
        if (isurf<=0) cycle

        !-----------------------------------------------------
        ! 
        ! generate inclusion and exclusion lists based on 
        ! the differences between atom_pos2(0,:,:) and
        !                         atom_pos2(1,:,:)  
        ! atom_pos2(0,:,:): is based on CHARMM connectivities
        !
        !-----------------------------------------------------
        
        do i=1,rx_natom1234
            do j=i+1,rx_natom1234
                ! * Even if there is no change in distance of a mrmd atoms and other atoms
                !   within rx_natom1234 list, the nonbonded interaction will still change 
                !   due to updated 
                !   vdw,q parameters of mrmd atom=> list should be generated for 
                !   atompairs having mrmd/mrmd or mrmd/nonmrmd atoms in rx_atom1234
                !   with nonchanged distance=> remove from 1-2    add to 1-2    with new parameters 
                !                           => remove from 1-3    add to 1-3    with new parameters
                !                           => remove from 1-4    add to 1-4    with new parameters 
                !                           => remove from 1-(4>) add to 1-(>4) with new parameters
                ! * FURTHERMORE 
                !      -     mrmd atoms in rx_atom1234 <=>  all atoms not in rx_atom1234
                !      - non-mrmd atoms in rx_atom1234 <=> mrmd atoms not in rx_atom1234
                !      -     mrmd atoms 
                  
                !   will also have updated vdw, electrostatic interaction
                !   As they all have unchanged 1-(>4) distance=> so all nonbonded
                !   should be updated 


                ! shorter variables for CHARMM index !
                ib1=rx_atom1234(i)
                ib2=rx_atom1234(j) 

                !-------------------------------------------------------------
                ! is it a generalized vdw pair, existing on surface isurf ?
                ! if yes => early exit => igvdw<=rx_ngvdw  
                ! if no  => late exit  => igvdw=rx_ngvdw+1
                ! igvdw will store the index of corresponding GVDW term!
                !-------------------------------------------------------------
                lgvdw=.false.
                do igvdw=1,rx_ngvdw
                    if ((all(rx_gvdw_atoms(igvdw,1:2)==(/ib1,ib2/)).or. &
                         all(rx_gvdw_atoms(igvdw,1:2)==(/ib2,ib1/))    )&
                         .and.rx_gvdw_exist(isurf,igvdw)) then
                         lgvdw=.true.
                         exit
                    endif                         
                enddo

                !----------------------------------------------
                ! if no change in group distance by mrmd bonds
                ! Differences between CHARMM and the surface
                !----------------------------------------------
                if (all(atom_pos2(0:1,i,j)==2)  .or.  &
                    all(atom_pos2(0:1,i,j)==3)  .or.  &
                    all(atom_pos2(0:1,i,j)==4)  .or.  &
                    all(atom_pos2(0:1,i,j)>=5)) then
                    
                    !--------------------------------------------
                    ! no change in neighbourship
                    ! and both atoms are NON-MRMD-ATOMS
                    ! and they do not form a general vdw pair
                    ! => no nonbonded interaction will be updated
                    !--------------------------------------------
                    if (all(rx_atom_inv((/ib1,ib2/))<=0).and..not.lgvdw) cycle
                    
                    !--------------------------------------------------------------
                    ! if there is at least one mrmd-atom in rx_atom1234 list
                    ! => their nonbonded interaction should be updated 
                    !    (depending on NBXMOD level and E14FAC)
                    ! => add them to exclusion list 1-n  (n=2,3,4,>4)
                    ! => add them to inclusion list 1-n  (n=2,3,4,>4)
                    ! Nonbonded interaction should be updated for general vdw pairs 
                    !--------------------------------------------------------------
                endif

                ! 1-2 exclusion
                if (atom_pos2(0,i,j)==2) then
                    rx_exc_n12(isurf)=rx_exc_n12(isurf)+1
                    tmp_exc_12(isurf,rx_exc_n12(isurf),1:2)= (/ib1,ib2/)
                endif

                ! 1-3 exclusion
                if (atom_pos2(0,i,j)==3) then
                    rx_exc_n13(isurf)=rx_exc_n13(isurf)+1
                    tmp_exc_13(isurf,rx_exc_n13(isurf),1:2)= (/ib1,ib2/)
                endif

                ! 1-4 exclusion
                if (atom_pos2(0,i,j)==4) then
                    rx_exc_n14(isurf)=rx_exc_n14(isurf)+1
                    tmp_exc_14(isurf,rx_exc_n14(isurf),1:2)= (/ib1,ib2/)
                endif

                ! 1-5 exclusion
                if (atom_pos2(0,i,j)>=5) then
                    rx_exc_n15(isurf)=rx_exc_n15(isurf)+1
                    tmp_exc_15(isurf,rx_exc_n15(isurf),1:2)= (/ib1,ib2/)
                endif

                ! 1-2 inclusion
                if (atom_pos2(1,i,j)==2) then
                    rx_inc_n12(isurf)=rx_inc_n12(isurf)+1
                    tmp_inc_12(isurf,rx_inc_n12(isurf),1:2)= (/ib1,ib2/)
                    ! storing index of general vdw interaction at 3rd position
                    if (lgvdw) tmp_inc_12(isurf,rx_inc_n12(isurf),3)=igvdw
                endif

                ! 1-3 inclusion   
                if (atom_pos2(1,i,j)==3) then
                    rx_inc_n13(isurf)=rx_inc_n13(isurf)+1
                    tmp_inc_13(isurf,rx_inc_n13(isurf),1:2)= (/ib1,ib2/)
                    ! storing index of general vdw interaction at 3rd position
                    if (lgvdw) tmp_inc_13(isurf,rx_inc_n13(isurf),3)=igvdw
                endif

                ! 1-4 inclusion
                if (atom_pos2(1,i,j)==4) then
                    rx_inc_n14(isurf)=rx_inc_n14(isurf)+1
                    tmp_inc_14(isurf,rx_inc_n14(isurf),1:2)= (/ib1,ib2/)
                    ! storing index of general vdw interaction at 3rd position
                    if (lgvdw) tmp_inc_14(isurf,rx_inc_n14(isurf),3)=igvdw
                endif

                ! 1-5 inclusion
                if (atom_pos2(1,i,j)>=5) then
                    rx_inc_n15(isurf)=rx_inc_n15(isurf)+1
                    tmp_inc_15(isurf,rx_inc_n15(isurf),1:2)= (/ib1,ib2/)
                    ! storing index of general vdw interaction at 3rd position
                    if (lgvdw) tmp_inc_15(isurf,rx_inc_n15(isurf),3)=igvdw
                endif
            enddo ! end of first atom cycle 
        enddo ! end of second atom cycle
    enddo ! end of isurf cycle 

    ! deallocate
    call MRMD_ALLOC(filen,procn,'atom_pos2',atom_pos2,1,0,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_atom1234',rx_atom1234,1,0)
    !rx_atom1234_inv is NEEDED LATER

!-------------------------------------------------------------------------
!
! Move neighbourship lists into global arrays of exact size.
!
!-------------------------------------------------------------------------

    !---------------------------------------------------
    ! inc12
    !---------------------------------------------------
    ! allocation
    j=maxval(rx_inc_n12(1:rx_nsurf))
    if (j>0) then
        call MRMD_ALLOC(filen,procn,'rx_inc_12',rx_inc_12,1,rx_nsurf,1,j,1,3)
        rx_inc_12=0
    endif
    ! transfering values
    do k=1,rx_nsurf
        if (rx_inc_n12(k)>0) &
        rx_inc_12(k,1:rx_inc_n12(k),1:3)=tmp_inc_12(k,1:rx_inc_n12(k),1:3)
    enddo  
    ! deallocation
    call MRMD_ALLOC(filen,procn,'tmp_inc_12',tmp_inc_12,1,0,1,0,1,0)

    !---------------------------------------------------
    ! exc12
    !---------------------------------------------------
    ! allocation
    j=maxval(rx_exc_n12(1:rx_nsurf))
    if (j>0) then
        call MRMD_ALLOC(filen,procn,'rx_exc_12',rx_exc_12,1,rx_nsurf,1,j,1,2)
        rx_exc_12=0
    endif
    ! transfering values
    do k=1,rx_nsurf
        if (rx_exc_n12(k)>0) &
        rx_exc_12(k,1:rx_exc_n12(k),1:2)=tmp_exc_12(k,1:rx_exc_n12(k),1:2)
    enddo  
    ! deallocation
    call MRMD_ALLOC(filen,procn,'tmp_exc_12',tmp_exc_12,1,0,1,0,1,0)

    !---------------------------------------------------
    ! inc13
    !---------------------------------------------------
    ! allocation
    j=maxval(rx_inc_n13(1:rx_nsurf))
    if (j>0) then
        call MRMD_ALLOC(filen,procn,'rx_inc_13',rx_inc_13,1,rx_nsurf,1,j,1,3)
        rx_inc_13=0
    endif
    ! transfering values
    do k=1,rx_nsurf
        if (rx_inc_n13(k)>0) &
        rx_inc_13(k,1:rx_inc_n13(k),1:3)=tmp_inc_13(k,1:rx_inc_n13(k),1:3)
    enddo  
    ! deallocation
    call MRMD_ALLOC(filen,procn,'tmp_inc_13',tmp_inc_13,1,0,1,0,1,0)

    !---------------------------------------------------
    ! exc13
    !---------------------------------------------------
    ! allocation
    j=maxval(rx_exc_n13(1:rx_nsurf))
    if (j>0) then
        call MRMD_ALLOC(filen,procn,'rx_exc_13',rx_exc_13,1,rx_nsurf,1,j,1,2)
        rx_exc_13=0
    endif
    ! transfering values
    do k=1,rx_nsurf
        if (rx_exc_n13(k)>0) &
        rx_exc_13(k,1:rx_exc_n13(k),1:2)=tmp_exc_13(k,1:rx_exc_n13(k),1:2)
    enddo  
    ! deallocation
    call MRMD_ALLOC(filen,procn,'tmp_exc_13',tmp_exc_13,1,0,1,0,1,0)

    !---------------------------------------------------
    ! inc14
    !---------------------------------------------------
    ! allocation
    j=maxval(rx_inc_n14(1:rx_nsurf))
    if (j>0) then
        call MRMD_ALLOC(filen,procn,'rx_inc_14',rx_inc_14,1,rx_nsurf,1,j,1,3)
        rx_inc_14=0
    endif
    ! transfering values
    do k=1,rx_nsurf
        if (rx_inc_n14(k)>0) &
        rx_inc_14(k,1:rx_inc_n14(k),1:3)=tmp_inc_14(k,1:rx_inc_n14(k),1:3)
    enddo  
    ! deallocation
    call MRMD_ALLOC(filen,procn,'tmp_inc_14',tmp_inc_14,1,0,1,0,1,0)

    !---------------------------------------------------
    ! exc14
    !---------------------------------------------------
    ! allocation
    j=maxval(rx_exc_n14(1:rx_nsurf))
    if (j>0) then
        call MRMD_ALLOC(filen,procn,'rx_exc_14',rx_exc_14,1,rx_nsurf,1,j,1,2)
        rx_exc_14=0
    endif
    ! transfering values
    do k=1,rx_nsurf
        if (rx_exc_n14(k)>0) &
        rx_exc_14(k,1:rx_exc_n14(k),1:2)=tmp_exc_14(k,1:rx_exc_n14(k),1:2)
    enddo  
    ! deallocation
    call MRMD_ALLOC(filen,procn,'tmp_exc_14',tmp_exc_14,1,0,1,0,1,0)   

    !---------------------------------------------------
    ! inc15
    !---------------------------------------------------
    ! allocation
    j=maxval(rx_inc_n15(1:rx_nsurf))
    if (j>0) then
        call MRMD_ALLOC(filen,procn,'rx_inc_15',rx_inc_15,1,rx_nsurf,1,j,1,3)   
        rx_inc_15=0
    endif
    ! transfering values
    do k=1,rx_nsurf
        if (rx_inc_n15(k)>0) &
        rx_inc_15(k,1:rx_inc_n15(k),1:3)=tmp_inc_15(k,1:rx_inc_n15(k),1:3)
    enddo   
    ! deallocation
    call MRMD_ALLOC(filen,procn,'tmp_inc_15',tmp_inc_15,1,0,1,0,1,0)

    !---------------------------------------------------
    ! exc15
    !---------------------------------------------------
    ! allocation
    j=maxval(rx_exc_n15(1:rx_nsurf))
    if (j>0) then
        call MRMD_ALLOC(filen,procn,'rx_exc_15',rx_exc_15,1,rx_nsurf,1,j,1,2)
        rx_exc_15=0
    endif
    ! transfering values
    do k=1,rx_nsurf
        if (rx_exc_n15(k)>0) &
        rx_exc_15(k,1:rx_exc_n15(k),1:2)=tmp_exc_15(k,1:rx_exc_n15(k),1:2)
    enddo  
    ! deallocation
    call MRMD_ALLOC(filen,procn,'tmp_exc_15',tmp_exc_15,1,0,1,0,1,0)   
  
    !==================================================================
    ! Print out terms which exist either only in exclusion or in only 
    ! inclusion list of same distance
    !==================================================================
    if (PRNLEV>=6) then 
        write(outu,'(''MRMD_INIT>'',80(''=''))')  
        write(outu,'(A)') 'MRMD_INIT>  Changes in neighbourship of atoms is generated:'
        write(outu,'(A)') 'MRMD_INIT>  This shows which 1-2,1-3,1-4,1-(>=5) relative'
        write(outu,'(A)') 'MRMD_INIT>  positions of atoms have been changed on each '
        write(outu,'(A)') 'MRMD_INIT>  surface due to redefinition of bonds in MRMD'

        do isurf=1, rx_nsurf
            write(outu,'(''MRMD_INIT>'',80(''-''))')  
            write(outu,'(A,x,I5)') 'MRMD_INIT>   On surface:',isurf

            ! 1-2 removed
            k=0
            do i=1,rx_exc_n12(isurf)
                do j=1,rx_inc_n12(isurf)
                    if ( all(rx_exc_12(isurf,i,1:2)==rx_inc_12(isurf,j,(/1,2/)))  &
                    .or. all(rx_exc_12(isurf,i,1:2)==rx_inc_12(isurf,j,(/2,1/)))) exit
                enddo
                ! if not found in the inclusion list
                if (j<=rx_inc_n12(isurf)) cycle
                ! print header
                if (k==0) write(outu,'(A,x,I5)') &
                'MRMD_INIT>    1-2    neighbourships removed:'
                ! start new line
                if (mod(k,4)==0) then
                    if (k>=4) write(outu,*) 
                    write(outu,'(A)',advance='no') 'MRMD_INIT>  '
                endif
                ! increase counter
                k=k+1
                ! write record
                write(outu,'(I6,x,I6,4x)',advance='no') rx_exc_12(isurf,i,1:2)
            enddo
            if (k>0) write(outu,*) 

            ! 1-2 added
            k=0
            do i=1,rx_inc_n12(isurf)
                do j=1,rx_exc_n12(isurf)
                    if ( all(rx_inc_12(isurf,i,1:2)==rx_exc_12(isurf,j,(/1,2/)))  &
                    .or. all(rx_inc_12(isurf,i,1:2)==rx_exc_12(isurf,j,(/2,1/)))) exit
                enddo
                ! if not found in the inclusion list
                if (j<=rx_exc_n12(isurf)) cycle
                ! print header
                if (k==0) write(outu,'(A,x,I5)') &
                'MRMD_INIT>    1-2    neighbourships   added:'
                ! start new line
                if (mod(k,4)==0) then
                    if (k>=4) write(outu,*) 
                    write(outu,'(A)',advance='no') 'MRMD_INIT>  '
                endif
                ! increase counter
                k=k+1
                ! write record
                write(outu,'(I6,x,I6,4x)',advance='no') rx_inc_12(isurf,i,1:2)
            enddo
            if (k>0) write(outu,*) 
            
            ! 1-3 removed
            k=0
            do i=1,rx_exc_n13(isurf)
                do j=1,rx_inc_n13(isurf)
                    if ( all(rx_exc_13(isurf,i,1:2)==rx_inc_13(isurf,j,(/1,2/)))  &
                    .or. all(rx_exc_13(isurf,i,1:2)==rx_inc_13(isurf,j,(/2,1/)))) exit
                enddo
                ! if not found in the inclusion list
                if (j<=rx_inc_n13(isurf)) cycle
                ! print header
                if (k==0) write(outu,'(A,x,I5)') &
                'MRMD_INIT>    1-3    neighbourships removed:'
                ! start new line
                if (mod(k,4)==0) then
                    if (k>=4) write(outu,*) 
                    write(outu,'(A)',advance='no') 'MRMD_INIT>  '
                endif
                ! increase counter
                k=k+1
                ! write record
                write(outu,'(I6,x,I6,4x)',advance='no') rx_exc_13(isurf,i,1:2)
            enddo
            if (k>0) write(outu,*) 

            ! 1-3 added
            k=0
            do i=1,rx_inc_n13(isurf)
                do j=1,rx_exc_n13(isurf)
                    if ( all(rx_inc_13(isurf,i,1:2)==rx_exc_13(isurf,j,(/1,2/)))  &
                    .or. all(rx_inc_13(isurf,i,1:2)==rx_exc_13(isurf,j,(/2,1/)))) exit
                enddo
                ! if not found in the inclusion list
                if (j<=rx_exc_n13(isurf)) cycle
                ! print header
                if (k==0) write(outu,'(A,x,I5)') &
                'MRMD_INIT>    1-3    neighbourships   added:'
                ! start new line
                if (mod(k,4)==0) then
                    if (k>=4) write(outu,*) 
                    write(outu,'(A)',advance='no') 'MRMD_INIT>  '
                endif
                ! increase counter
                k=k+1
                ! write record
                write(outu,'(I6,x,I6,4x)',advance='no') rx_inc_13(isurf,i,1:2)
            enddo
            if (k>0) write(outu,*) 
            
            
            ! 1-4 removed
            k=0
            do i=1,rx_exc_n14(isurf)
                do j=1,rx_inc_n14(isurf)
                    if ( all(rx_exc_14(isurf,i,1:2)==rx_inc_14(isurf,j,(/1,2/)))  &
                    .or. all(rx_exc_14(isurf,i,1:2)==rx_inc_14(isurf,j,(/2,1/)))) exit
                enddo
                ! if not found in the inclusion list
                if (j<=rx_inc_n14(isurf)) cycle
                ! print header
                if (k==0) write(outu,'(A,x,I5)') &
                'MRMD_INIT>    1-4    neighbourships removed:'
                ! start new line
                if (mod(k,4)==0) then
                    if (k>=4) write(outu,*) 
                    write(outu,'(A)',advance='no') 'MRMD_INIT>  '
                endif
                ! increase counter
                k=k+1
                ! write record
                write(outu,'(I6,x,I6,4x)',advance='no') rx_exc_14(isurf,i,1:2)
            enddo
            if (k>0) write(outu,*) 

            ! 1-4 added
            k=0
            do i=1,rx_inc_n14(isurf)
                do j=1,rx_exc_n14(isurf)
                    if ( all(rx_inc_14(isurf,i,1:2)==rx_exc_14(isurf,j,(/1,2/)))  &
                    .or. all(rx_inc_14(isurf,i,1:2)==rx_exc_14(isurf,j,(/2,1/)))) exit
                enddo
                ! if not found in the inclusion list
                if (j<=rx_exc_n14(isurf)) cycle
                ! print header
                if (k==0) write(outu,'(A,x,I5)') &
                'MRMD_INIT>    1-4    neighbourships   added:'
                ! start new line
                if (mod(k,4)==0) then
                    if (k>=4) write(outu,*) 
                    write(outu,'(A)',advance='no') 'MRMD_INIT>  '
                endif
                ! increase counter
                k=k+1
                ! write record
                write(outu,'(I6,x,I6,4x)',advance='no') rx_inc_14(isurf,i,1:2)
            enddo
            if (k>0) write(outu,*) 
            
            ! 1-(>4) removed
            k=0
            do i=1,rx_exc_n15(isurf)
                do j=1,rx_inc_n15(isurf)
                    if ( all(rx_exc_15(isurf,i,1:2)==rx_inc_15(isurf,j,(/1,2/)))  &
                    .or. all(rx_exc_15(isurf,i,1:2)==rx_inc_15(isurf,j,(/2,1/)))) exit
                enddo
                ! if not found in the inclusion list
                if (j<=rx_inc_n15(isurf)) cycle
                ! print header
                if (k==0) write(outu,'(A,x,I5)') &
                'MRMD_INIT>    1-(>4) neighbourships removed:'
                ! start new line
                if (mod(k,4)==0) then
                    if (k>=4) write(outu,*) 
                    write(outu,'(A)',advance='no') 'MRMD_INIT>  '
                endif
                ! increase counter
                k=k+1
                ! write record
                write(outu,'(I6,x,I6,4x)',advance='no') rx_exc_15(isurf,i,1:2)
            enddo
            if (k>0) write(outu,*) 

            ! 1-(>4) added
            k=0
            do i=1,rx_inc_n15(isurf)
                do j=1,rx_exc_n15(isurf)
                    if ( all(rx_inc_15(isurf,i,1:2)==rx_exc_15(isurf,j,(/1,2/)))  &
                    .or. all(rx_inc_15(isurf,i,1:2)==rx_exc_15(isurf,j,(/2,1/)))) exit
                enddo
                ! if not found in the inclusion list
                if (j<=rx_exc_n15(isurf)) cycle
                ! print header
                if (k==0) write(outu,'(A,x,I5)') &
                'MRMD_INIT>    1-(>4) neighbourships   added:'
                ! start new line
                if (mod(k,4)==0) then
                    if (k>=4) write(outu,*) 
                    write(outu,'(A)',advance='no') 'MRMD_INIT>  '
                endif
                ! increase counter
                k=k+1
                ! write record
                write(outu,'(I6,x,I6,4x)',advance='no') rx_inc_15(isurf,i,1:2)
            enddo
            if (k>0) write(outu,*) 
        enddo 
        write(outu,'(''MRMD_INIT>'',80(''=''))')  
    endif
end subroutine MRMD_INIT

!===========================================================================
!

!
!
!  
!                      ENERGY/FORCE CALCULATION
!
!
!
!
!===========================================================================
subroutine EMRMD(mrmd_energy,X,Y,Z,dEdx,dEdy,dEdz,natomx)     
    use dimens_fcm
    use number
    use stream
    use psf
    use inbnd
    use param
    use consta
    use contrl
    use reawri
    use energym
    implicit none

    !---------------------------------------------------------------------------
    ! input
    integer,intent(in):: natomx
    real(chm_real),intent(in) ::      &
        X(natomx),      &  ! x coordinate of all atoms
        Y(natomx),      &  ! y coordinate of all atoms
        Z(natomx)          ! z coordinate of all atoms   
    !---------------------------------------------------------------------------
    ! output 
    real(chm_real),intent(out) ::      &
        mrmd_energy         ! effective energy correction to CHARMM energy
    ! input-output 
    real(chm_real),intent(inout),optional ::     &
        dEdx(natomx),   &   ! x gradient of potential energy (-Fx) for all atoms              
        dEdy(natomx),   &   ! y gradient of potential energy (-Fy) for all atoms  
        dEdz(natomx)        ! z gradient of potential energy (-Fz) for all atoms
    !---------------------------------------------------------------------------
    ! local

    ! FOR ALLOCATION ROUTINE    
    character(len=5) ::   & ! 
       procn='EMRMD'         ! procedure name

    logical rx_print_energy0

    real(chm_real),allocatable,dimension(:,:)::  &
        rx_dEdx,    &   !              
        rx_dEdy,    & 
        rx_dEdz,    &
        dfdE          ! dfdE(i,j)=dfj/dfi: derivative of factors wrt. energies

    integer             &
        isurf_lowest,   &   ! Number of lowest energy surface
        i,j,k,          &
        isurf,          &   ! surface index
        isurf1,         &   ! surface index
        isurf2              ! surface index

    real(chm_real)                  &
        time,                       &
        energymin,                  &
        factors(rx_nsurf),          & ! weight factors for each PES
        energies(-rx_nsurf:rx_nsurf,-rx_ninte:rx_ninte),  & 
                        !
                        ! 1st index -rx_nsurf:-1, 2nd index=0 => GAPO corrected surfaces
                        !
                        ! 1st index -rx_nsurf:-1, 2nd index/=0 NOT USED!
                        !
                        ! energy correction of each PES by energy_type/inclusion/exclusion/in_total
                        ! 1st index>0 and 2nd index -rx_ninte:rx_ninte:
                        !  0: total - energies(1:isurf,-rx_ninte:rx_ninte)
                        ! -1: nbon elec removed from CHARMM, +1: nbon elec added on MRMD
                        ! -2: nbon  vdw removed from CHARMM, +2: nbon  vdw added on MRMD
                        ! -3: not used                     , +3: nbon gvdw added on MRMD
                        ! -4: bond harm removed from CHARMM, +4: bond harm added on MRMD
                        ! -5: not used                     , +5: bond mors added on MRMD
                        ! -6: angl harm removed from CHARMM, +6: angl harm added on MRMD
                        ! -7: angl urey removed from CHARMM, +7: angl urey added on MRMD
                        ! -8: nbon dihe removed from CHARMM, +8: nbon dihe added on MRMD
                        ! -9: nbon impr removed from CHARMM, +9: nbon impr added on MRMD
                        !-10: not used                     ,+10: bond rkhs added on MRMD
        deltaE,                     & ! energy difference of two surfaces  
        sumw12,                     & ! sum of weights as prefactor for GAPO 
        gapo,                       & ! gaussian*polynomial 
        dgapodDE                      ! derivative of gaussian*polynomial with respect to Ei-Ej
                                       

        real(chm_real),allocatable,dimension(:)::  &
            sum_dfGAPO_dE  ! sum_dfGAPO_dE(i,j)=d[sum_j((fj1+fj2)*GAPOj)]/dEi= 
                                       ! sum_j[( fj1    + fi2    )*dGAPOj/dEi]+
                                       ! sum_j[(dfj1/dEi+dfj2/dEi)* GAPOj]
                                                   
        ! check whether it is active
        if (.not.MRMD_ACTIVE) return

        ! number of calls counter
        rx_ncall_emrmd=rx_ncall_emrmd+1

        !==================================================================
        !  Check compatibility of non-bonded interaction handling
        !==================================================================
        if (.not.lElec )  &
                            call wrndie(-3,'<mrmd.src> emrmd', &
                            'ELECtrosatics: electrostatics is required')  

        if (    lgroup ) &
                            call wrndie(-3,'<mrmd.src> emrmd',&
                            'GROUP: group based electrostatics is not supported')   

        if (.not.lCons ) &
                            call wrndie(-3,'<mrmd.src> emrmd',&
                            'CDIElectr:constant dielectric is required')       
        
        if (.not.lShft ) &
                            call wrndie(-3,'<mrmd.src> emrmd',&
                            'SHIFt: shift is required for electrostatics')  

        if (      lfswt) &
                            call wrndie(-3,'<mrmd.src> emrmd',&
                            'FSWItched:force-switch for electrostatics is not supported')

        if (   EPS< 1._chm_real) &
                            call wrndie(-3,'<mrmd.src> emrmd',&
                            'EPS:relative permittivity should not be less than one')

        if (.not. lvdw ) &
                            call wrndie(-3,'<mrmd.src> emrmd',&
                            'VDW: van der Waals is required')

        if (.not.LVatom) &
                            call wrndie(-3,'<mrmd.src> emrmd',&
                            'VATOM:ATOM based van der Waals is required') 

        if (     lvshft) &
                            call wrndie(-3,'<mrmd.src> emrmd',&
                            'VSHIFted:shift for van der Waals is not supported')

        if (     lvfswt) &
                            call wrndie(-3,'<mrmd.src> emrmd',&
                            'FVSWitched:force-switch for van der Waals is not supported')

        if (  NBXMOD<=0)  &
                            call wrndie(-3,'<mrmd.src> emrmd',&
                            'NBXMOD 1-5 are supported')

        !====================================================================================
        ! CONDITION FOR PRINT OUTS
        !====================================================================================
        ! if dynamics is active and mdstep is divisable by rx_dyna_print_steps
        rx_print_energy=.false.                                    
        if (rx_dyna_print_steps>0) &
        rx_print_energy=rx_print_energy.or.(DYNAMQ.and.mod(MDSTEP,rx_dyna_print_steps)==0)

        ! if dynamics is inactive and callnumber is divisable by rx_nondyna_print_calls
        if (rx_nondyna_print_calls>0) &
        rx_print_energy=rx_print_energy.or.&
        (.not.DYNAMQ.and.mod(rx_ncall_emrmd,rx_nondyna_print_calls)==0) 
        
        ! if first call
        rx_print_energy=rx_print_energy.or.rx_isurf_lowest0==0
        
        !====================================================================================
        !
        ! CALCULATE ENERGY CORRECTION ON THE SURFACES
        !
        !====================================================================================
        ! separator
        if (PRNLEV>=7.and.rx_print_energy) write(outu,'(''EMRMD>'',80(''=''))')
        ! note: mrmd_energy=energies(0,0)+GAPO correction
        call MRMD_ENERG(natomx,x,y,z,energies,factors,mrmd_energy,isurf_lowest,energymin)

        !--------------------------------------------------
        ! FURTHER CONDITION FOR PRINT OUTS
        !--------------------------------------------------
        ! whenever crossing is detected [isurf_lowest/=rx_isurf_lowest0]
        rx_print_energy=rx_print_energy.or.(isurf_lowest/=rx_isurf_lowest0)

        !--------------------------------------------------
        ! GENERATE SUMMARY ON SURFACE ENERGY CORRECTION  
        !--------------------------------------------------
        if (rx_print_energy) then
            ! detailed output on each surfaces
            if (PRNLEV>=5) then
                write(outu,'(''EMRMD>'',80(''-''))')  

                ! print time
                if (DYNAMQ) &
                write(outu,'(A,F13.6,x,A)')'EMRMD> Time:',MDSTEP*TIMEST,'ps'

                ! print energy components
                write(outu,'(A6,x,A4,x,A10,x,A6,x,A10,x,A8,4(x,A10))') &
                'EMRMD>',     & !1  A6    
                'PES#',       & !2  A4  
                '    E-Emin', & !3  A10
                'Weight',     & !4  A6
                ' E-ECHarmm', & !5  A10
                '    GaPo',   & !6  A8
                'E+GaPo-ECH', & !7  A10
                '  W(E-ECH)', & !8  A10
                '    W*GAPO', & !9  A10
                'W(EGP-ECH)'    !10 A10
                              
    
                ! PES energies/factors for each surface
                do isurf=1,rx_nsurf
                    !energies        
                    write(outu,'(A6,x,I4.4,x,F10.3,x,F6.3,x,F10.3,x,F8.3,4(x,F10.3))',&
                    advance='no') &
                    'EMRMD>',                               &
                    isurf,                               &
                    energies(isurf,0)-energymin,   &            ! Ei-minE
                    factors(isurf),                   &            ! w 
                    energies(isurf,0),                &            ! Ei-E_CHARMM
                    energies(-isurf,0)-energies(isurf,0), &  ! GAPO
                    energies(-isurf,0),                         &  ! Ei+GAPOi-E_CHARMM
                    factors(isurf)* energies(isurf,0),    &  ! w*(Ei-E_CHARMM)
                    factors(isurf)*(energies(-isurf,0)-energies(isurf,0)), & !w*GAPO
                    factors(isurf)* energies(-isurf,0)       ! w*(Ei+GAPOi-E_CHARMM)
                
                    ! labeling lowest at the end of line
                    if (isurf==isurf_lowest) then
                        write(outu,'(A)') ' LOWEST'        
                    else
                        write(outu,*)
                    endif
                enddo   
            endif

            ! effective MRMD energy correction added to CHARMM
            if (PRNLEV>=4) then
                write(outu,'(''EMRMD>'',80(''-''))')  
                write(outu,'(A6,2(x,A10),x,A)') &
                'EMRMD>',    & !1 A6    
                'E-ECHaRMm', & !2 A10
                '    EGaPo', & !3 A10
                'EGP-ECHRM'    !4 A10
                write(outu,'(A6,3(x,F10.3),x,A)') &
                    'EMRMD>',   &
                    energies(0,0),             & ! E-E_CHARMM 
                    mrmd_energy-energies(0,0), & ! GAPO 
                    mrmd_energy,               & ! E+GAPO-E_CHARMM
                    '<=TOTAL MRMD ENERGY CORRECTION TO CHARMM'
                ! separator
                write(outu,'(''EMRMD>'',80(''-''))')  
            endif
        endif

        ! do not print energies during force calculations unless prnlev>=10
        rx_print_energy0=rx_print_energy  ! store it
        rx_print_energy=prnlev>=10.and.rx_print_energy0 ! for printing force of individual energy
                                                        ! terms                             
        
        !--------------------------------------------------
        ! GENERATE SUMMARY ON DETECTED CROSSING  
        !--------------------------------------------------
        if (rx_isurf_lowest0/=0.and.isurf_lowest/=rx_isurf_lowest0) then
            ! calculate time
            TIME = MDSTEP*TIMEST
            
            ! write pdb
            if (rx_crossgeom_unit>0) &
            call  MRMD_PDB(rx_crossgeom_unit,natomx,X,Y,Z,TIME,rx_nsurf,factors)

            ! write report on crossing
            if (PRNLEV>=4) write(outu,'(A,x,I2,x,A,x,I2,A,F13.6,A)')   &
            'EMRMD> CROSSING DETECTED FROM SURFACE',rx_isurf_lowest0,  &
            'TO SURFACE',isurf_lowest,                              &
            'AT TIME= ',TIME,'ps'
        endif

        !----------------------------
        ! new lowest energy surface
        !----------------------------
        rx_isurf_lowest0=isurf_lowest

        !====================================================================================
        ! return if force calculation is not requested
        !====================================================================================
        if (.not.(present(dEdx).and.present(dEdy).and.present(dEdz))) return
        !====================================================================================
        !
        ! CALCULATE GRADIENT CORRECTION ON THE SURFACES
        !
        ! Note, energy calculation is almost always done twice for each surfaces with EXPDECAY
        ! switching method.
        ! With JOHNSON7 switching function switching takes place in finite energy
        ! range, therefore forces for not all surfaces are needed.
        !
        ! Later there can be intelligent checking for acceleration whether
        ! energy should not be calculated for all surfaces for a few integration steps. 
        !====================================================================================
        !------------------------------------------------------------------------------
        ! calculate derivative of switching factors with respect to active PES energies
        ! dfdE(i,j)=  dfactor(j)/dE(i)  
        ! analytically 
        !------------------------------------------------------------------------------
        call MRMD_ALLOC(filen,procn,'dfdE',dfdE,1,rx_nsurf,1,rx_nsurf)              
        call MRMD_SWITCH_WEIGHTS(rx_nsurf,energies(1:rx_nsurf,0),rx_swch_dE,&
        trim(rx_swch_func),factors(1:rx_nsurf),dfdE)

        !---------------------------------------------------------------------------------
        ! calculate derivative of Gaussian*polynomials with respect 
        ! to active PES energies analytically 
        !---------------------------------------------------------------------------------
        call MRMD_ALLOC(filen,procn,'sum_dfGAPO_dE',sum_dfGAPO_dE,1,rx_nsurf)              
        sum_dfGAPO_dE(:)= 0._chm_real
        do i=1,rx_ngapo
            isurf1 = rx_gapo_surfs(i,1)
            isurf2 = rx_gapo_surfs(i,2)
            
            ! cycle if their weigths are zero
            sumw12=factors(isurf1)+factors(isurf2)
            if (sumw12==0._chm_real) cycle

            ! calculate gradient of Gauss*polynomials
            deltaE    = energies(isurf2,0)-energies(isurf1,0)
            call MRMD_GAPO(deltaE,rx_gapo_order(i),&
                 rx_gapo_coefs(i,-2:rx_gapo_order(i)),gapo,dgapodDE)

            ! sum_i[(dfi1/dE+dfi2/dE)*GAPOi]
            ! note: dfdE(i,j)=dfj/dEi
            sum_dfGAPO_dE(:)=&
            sum_dfGAPO_dE(:)+(dfdE(:,isurf1)+dfdE(:,isurf2))*gapo

            ! sum_i[(fi1+fi2)*dGAPOi/dEj]
            ! argument of Gauss*polynomials is x=E2-E1
            ! dx/dE2=1    dx/dE1=-1
            sum_dfGAPO_dE(isurf2)= sum_dfGAPO_dE(isurf2)+sumw12*dgapodDE
            sum_dfGAPO_dE(isurf1)= sum_dfGAPO_dE(isurf1)-sumw12*dgapodDE
        enddo

        !---------------------------------------------------------------------
        ! calculate gradient contribution for each active surface analytically
        !---------------------------------------------------------------------
        call MRMD_ALLOC(filen,procn,'rx_dEdx',rx_dEdx,0,1,1,natomx)              
        call MRMD_ALLOC(filen,procn,'rx_dEdy',rx_dEdy,0,1,1,natomx)              
        call MRMD_ALLOC(filen,procn,'rx_dEdz',rx_dEdz,0,1,1,natomx)              
        rx_dEdx(0,:)=0._chm_real
        rx_dEdy(0,:)=0._chm_real
        rx_dEdz(0,:)=0._chm_real
        do isurf=1,rx_nsurf

            ! skip surfaces with small factors
            if (factors(isurf)==0._chm_real.and.rx_ngapo>0) then
                ! is it involved in gapo?
                do j=1,rx_ngapo
                    ! if not in the j-th cycle
                    if (all(rx_gapo_surfs(j,1:2)/=isurf)) cycle
                    
                    ! involved in gausspolynomial dividing surface shaping  
                    ! if yes, is the other weigth also non-zero?
                    if (any(factors(rx_gapo_surfs(j,1:2))/=0._chm_real)) exit
                enddo
                ! if cycle terminated by exit command then j<=rx_ngapo       => grad needed   
                ! if cycle terminated by not exit command then j==rx_ngapo+1 => grad not needed
                if (j==rx_ngapo+1) cycle     
            endif
            !-----------------------------
            ! calculate potential gradient
            !-----------------------------
            rx_calcgrad=.true.  ! also forces=>global
            call MRMD_SURF(natomx,x(1:natomx),y(1:natomx),z(1:natomx),          &
                         isurf,energies(isurf,-rx_ninte:rx_ninte),              &
                         rx_dEdx(1,1:natomx),rx_dEdy(1,1:natomx),rx_dEdz(1,1:natomx))

            !--------------------------------------------------
            ! 1.
            ! add gradient coming from each PES energy derivation
            !
            ! gradient = sum_i(fi*(dEi/dR))+...
            !--------------------------------------------------
            rx_dEdx(0,:) = rx_dEdx(0,:) + factors(isurf)*rx_dEdx(1,:)
            rx_dEdy(0,:) = rx_dEdy(0,:) + factors(isurf)*rx_dEdy(1,:)
            rx_dEdz(0,:) = rx_dEdz(0,:) + factors(isurf)*rx_dEdz(1,:)

            !-------------------------------------------------------------------------------                        
            ! 2.
            ! add gradient coming from each switching factor derivation
            !
            ! chain rule...but through only one intermediate variable at a time
            ! gradient=...sum_j(sum_i(dfj/dEi*dEi/dR)*Ej)=sum_j(sum_i(dfj/dEi*dEi/dR*Ej))=
            !                                          =sum_i(sum_j(Ej*dfj/dEi)*dEi/dR )=
            !  j=isurf
            !  dfi/dEj= dfdE(j,i)  = derivative of factors w.r.t. the energy of surface isurf
            !  rx_dEdx(1,:)=dEj/dR = gradient of surface isurf
            !-------------------------------------------------------------------------------
            ! i summing
            rx_dEdx(0,:)= rx_dEdx(0,:) &
              +sum(energies(1:rx_nsurf,0)*dfdE(isurf,1:rx_nsurf))*rx_dEdx(1,:)

            rx_dEdy(0,:)= rx_dEdy(0,:) &
              +sum(energies(1:rx_nsurf,0)*dfdE(isurf,1:rx_nsurf))*rx_dEdy(1,:)

            rx_dEdz(0,:)= rx_dEdz(0,:) &
              +sum(energies(1:rx_nsurf,0)*dfdE(isurf,1:rx_nsurf))*rx_dEdz(1,:)

            !-------------------------------------------------------------------------------                        
            ! 3.
            ! add gradient coming from differentiating the sum of weight*Gauss*polynomial 
            ! functions: d[sum_j((fj1+fj2)*GAPOj)]/dEi*dEi/dR    - i=isurf summing
            !-------------------------------------------------------------------------------                        
            if (rx_ngapo>0) then
                if (sum_dfGAPO_dE(isurf)/=0._chm_real) then
                    rx_dEdx(0,:)= rx_dEdx(0,:)+sum_dfGAPO_dE(isurf)*rx_dEdx(1,:)
                    rx_dEdy(0,:)= rx_dEdy(0,:)+sum_dfGAPO_dE(isurf)*rx_dEdy(1,:)
                    rx_dEdz(0,:)= rx_dEdz(0,:)+sum_dfGAPO_dE(isurf)*rx_dEdz(1,:)
                endif
            endif
        enddo

        !-------------------------------------------
        ! ADD GRADIENT CORRECTION to CHARMM GRADIENT
        !-------------------------------------------
        dEdx = dEdx + rx_dEdx(0,:)
        dEdy = dEdy + rx_dEdy(0,:)
        dEdz = dEdz + rx_dEdz(0,:)

        !--------------------------------------------------
        ! GENERATE SUMMARY ON SURFACE FORCE CORRECTION  
        !--------------------------------------------------
        if (rx_print_energy0.and.PRNLEV>=9) then
             write(outu,'(A)') &
             'EMRMD> Non-zero corrections to CHARMM gradient on effective PES'
             write(outu,'(A6,x,4(A10,x))') &
             'EMRMD>','atom','+gradientX','+gradientY','+gradientZ'        
             do i=1,natomx
                 ! print only non-zero force corrections
                 if (rx_dEdx(0,i)/=0._chm_real.or. &
                     rx_dEdy(0,i)/=0._chm_real.or. &
                     rx_dEdx(0,i)/=0._chm_real)    &
                 write(outu,'(A6,x,I10,x,3(F10.4,x))') &
                 'EMRMD>',i,rx_dEdx(0,i),rx_dEdy(0,i),rx_dEdz(0,i)
             enddo
        endif

end subroutine EMRMD

!===========================================================================
!
!   CALCULATE POTENTIAL ENERGY CORRECTION (TO CHARMM) 
!
!===========================================================================
subroutine MRMD_ENERG(natomx,x,y,z,energies,factors,energy,isurf_lowest,energymin)
    use stream
    implicit none

    ! global
    ! rx_calcgrad
    ! rx_swch_dE
    ! rx_ninte
    ! rx_nsurf

    !---------------------------------------------------------------------------
    ! input
    integer,intent(in) :: &
        natomx ! number of atoms
    real(chm_real),intent(in):: &
        x(natomx),y(natomx),z(natomx)   ! x,y,z coordinates for all atoms 
    !---------------------------------------------------------------------------
    
    ! ouput
    real(chm_real),intent(out) :: &
        energies(-rx_nsurf:rx_nsurf,-rx_ninte:rx_ninte), & ! energy corrections for all surfaces
                        !
                        ! 1st index -rx_nsurf:-1, 2nd index=0 => GAPO corrected surfaces
                        !
                        ! 1st index -rx_nsurf:-1, 2nd index/=0 NOT USED!
                        !
                        ! energy correction of each PES by energy_type/inclusion/exclusion/in_total
                        ! 1st index>0 and 2nd index -rx_ninte:rx_ninte:
                        !  0: total - energies(1:isurf,-rx_ninte:rx_ninte)
                        ! -1: nbon elec removed from CHARMM, +1: nbon elec added on MRMD
                        ! -2: nbon  vdw removed from CHARMM, +2: nbon  vdw added on MRMD
                        ! -3: not used                     , +3: nbon gvdw added on MRMD
                        ! -4: bond harm removed from CHARMM, +4: bond harm added on MRMD
                        ! -5: not used                     , +5: bond mors added on MRMD
                        ! -6: angl harm removed from CHARMM, +6: angl harm added on MRMD
                        ! -7: angl urey removed from CHARMM, +7: angl urey added on MRMD
                        ! -8: nbon dihe removed from CHARMM, +8: nbon dihe added on MRMD
                        ! -9: nbon impr removed from CHARMM, +9: nbon impr added on MRMD   
        factors(rx_nsurf),       & ! switching weigths
        energy,                  & ! total energy correction with GAPO
        energymin                  ! energy correction for the minimum energy surface
    integer,intent(out) :: isurf_lowest ! index of the lowest energy surface
    !---------------------------------------------------------------------------
    ! local
    real(chm_real) sumw12,deltaE,Egapo
    integer i,rx_iinte,isurf,isurf1,isurf2
    
    !------------------------------------
    ! calculate energies for each surface
    ! and find the one with lowest energy
    !------------------------------------
        rx_calcgrad=.false.
        isurf_lowest=1
        do isurf=1,rx_nsurf
            ! call energy routine - shift included

            call MRMD_SURF(natomx,x(1:natomx),y(1:natomx),z(1:natomx),isurf,&
                            energies(isurf,-rx_ninte:rx_ninte))

            ! set the one with lower energy 
            if (energies(isurf,0)<energies(isurf_lowest,0)) &
            isurf_lowest =isurf
            
            ! store a copy of energy for later GAPO correction
            energies(-isurf,0)=energies(isurf,0)
        enddo
        ! lowest surface
        energymin=energies(isurf_lowest,0)

    !------------------------------------------------------------------------
    ! calculate mixed energy
    !------------------------------------------------------------------------
        !----------------------------------------------------------------------------------
        ! calculate energy mixing weights, all active surfaces
        !----------------------------------------------------------------------------------
        call MRMD_SWITCH_WEIGHTS(rx_nsurf,energies(1:rx_nsurf,0),rx_swch_dE, &
        trim(rx_swch_func),factors)

        !----------------------------------------------------------------------------------
        ! mix energy corrections to CHARMM energy for each energy terms
        !----------------------------------------------------------------------------------
        do rx_iinte=-rx_ninte,rx_ninte
            energies(0,rx_iinte)=sum(factors(1:rx_nsurf)*energies(1:rx_nsurf,rx_iinte))
        enddo

    !----------------------------------------------------------------------------------
    ! Add gaussian*polynomial corrections
    ! sum_i[ (fi1+fi2)*GAPO(Vi2-Vi1) ]
    !----------------------------------------------------------------------------------
        !--------------
        ! final energy
        !--------------
        ! energy        with GAPOs
        ! energies(0,0) without GAPOs
        energy=energies(0,0)
                
        if (rx_print_energy.and.PRNLEV>=7) &
        write(outu,'(A10,3(x,A5),4(x,A12))') &
        'MRMD_ENERG>','#GAPO','SURF1','SURF2',&
        '     W1     ','     W2     ',       &
        '   EGAPO   ','(W1+W2)EGAPO'

        do i=1,rx_ngapo
            isurf1 = rx_gapo_surfs(i,1)
            isurf2 = rx_gapo_surfs(i,2)

            ! if their weights are zero =>skip
            sumw12=factors(isurf1)+factors(isurf2)
            if (sumw12==0._chm_real) cycle

            ! calculate gauss*polynomial
            deltaE    = energies(isurf2,0)-energies(isurf1,0)
            call MRMD_GAPO(deltaE,rx_gapo_order(i),&
                 rx_gapo_coefs(i,-2:rx_gapo_order(i)),Egapo)
            
            ! add it to final energy
            energy=energy+sumw12*Egapo
            
            ! decompose into surface energies
            energies(-isurf1,0)=energies(-isurf1,0)+Egapo
            energies(-isurf2,0)=energies(-isurf2,0)+Egapo
            
            if (rx_print_energy.and.PRNLEV>=7) &
            write(outu,'(A10,3(x,I5),4(x,G12.5))') &
            'MRMD_ENERG>',i,rx_gapo_surfs(i,1:2), &
            factors(rx_gapo_surfs(i,1:2)),Egapo,sumw12*Egapo
        enddo
        
end subroutine MRMD_ENERG

!===========================================================================
!
!   CALCULATE POTENTIAL ENERGY CORRECTION (TO CHARMM) AND ITS GRADIENT FOR 
!   A SINGLE SURFACE  
!
!===========================================================================
!        call mrmd_surf(natomx,x,y,z,isurf,&
!                       energies(isurf,-rx_ninte:rx_ninte),   &
!                       dEdx(1,:),             &
!                       dEdy(1,:),             &
!                       dEdz(1,:))

subroutine MRMD_SURF(natomx,x,y,z,isurf,energies,dEdx,dEdy,dEdz)
  
    use dimens_fcm
    use number
    use reawri
    use stream
    use inbnd    ! nonbonded list
    use psf
    use cnst_fcm
    use contrl
    use consta
    use rtf
    use param
    use energym
    use code
    implicit none

    !---------------------------------------------------------------------------
    ! input
    integer,intent(in) :: &
        natomx,  & ! number of atoms
        isurf   ! index of requested surface 
        
    real(chm_real),intent(in) ::  &
        X(natomx),Y(natomx),Z(natomx) ! x,y,z coordinates for all atoms 
    !---------------------------------------------------------------------------
    ! output
    real(chm_real),optional,intent(inout):: & 
        energies(-rx_ninte:rx_ninte),    &   ! PES energy  correction on the surface
                        !
                        ! 1st index -rx_nsurf:-1, 2nd index=0 => GAPO corrected surfaces
                        !
                        ! 1st index -rx_nsurf:-1, 2nd index/=0 NOT USED!
                        !
                        ! energy correction of each PES by energy_type/inclusion/exclusion/in_total
                        ! 1st index>0 and 2nd index -rx_ninte:rx_ninte:
                        !  0: total - energies(1:isurf,-rx_ninte:rx_ninte)
                        ! -1: nbon elec removed from CHARMM, +1: nbon elec added on MRMD
                        ! -2: nbon  vdw removed from CHARMM, +2: nbon  vdw added on MRMD
                        ! -3: not used                     , +3: nbon gvdw added on MRMD
                        ! -4: bond harm removed from CHARMM, +4: bond harm added on MRMD
                        ! -5: not used                     , +5: bond mors added on MRMD
                        ! -6: angl harm removed from CHARMM, +6: angl harm added on MRMD
                        ! -7: angl urey removed from CHARMM, +7: angl urey added on MRMD
                        ! -8: nbon dihe removed from CHARMM, +8: nbon dihe added on MRMD
                        ! -9: nbon impr removed from CHARMM, +9: nbon impr added on MRMD
                        !-10: not used                     ,+10: bond rkhs added on MRMD
        dEdx(natomx),   &   ! x derivative of energy correction for the surface for all atoms
        dEdy(natomx),   &   ! x derivative of energy correction for the surface for all atoms
        dEdz(natomx)        ! x derivative of energy correction for the surface for all atoms
    !---------------------------------------------------------------------------
    !local

    ! FOR ALLOCATION ROUTINE    
    character(len=9),parameter ::   & ! 
       procn='MRMD_SURF'         ! procedure name

    real(chm_real) geompar

    integer i,j,k,l,II,JJ,KK,LL,IX,MULT

    real(chm_real) FCON,REQ,TEQ,PEQ,PEQC,PEQS,DE,BETA,R

    real(chm_real) C2OFNB,C2ONNB,CTOFNB1,CTOFNB2,CTOFNB4
    real(chm_real) S,rx,ry,rz,RUL3
    
    integer ibond,iharm,imors,irkhs,iangl,idihe,iimpr,igvdw
    integer iatom,psf_iatom
    
    integer,pointer :: xxc(:,:)

    integer      &
        pmone,   & 
        dist,    &
        nxxc,    &
        ixxc
    
    real(chm_real)  &
        rx_rminp2(2),   &
        rx_eps(2),      &
        rx_chrg(2)

    real(chm_real)              &
        ener_exc,ener_inc,      &
        denergy,                &   
        dfx(1:4),dfy(1:4),dfz(1:4)

    character spmone
    character(len=5) rx_nbdist(0:5) 

    ! reset energies
    energies(-rx_ninte:rx_ninte)=0._chm_real

    ! reset gradient
    if (rx_calcgrad) then
        dEdx=0._chm_real
        dEdy=0._chm_real
        dEdz=0._chm_real 
    endif
    
    if (rx_print_energy.and.PRNLEV>=7) then
        write(outu,'(''MRMD_SURF>'',80(''-''))')  
        write(outu,'(A10,x,A4,x,A10,x,A12,4(x,A5),4(x,A11))') &
        'MRMD_SURF>', &
        'surf','   type   ','   Energy  ','atom1','atom2','atom3','atom4',&
        'ener_param1','ener_param2','geom_param0','geom_param'
    endif  

    !==============================================================================================
    !==============================================================================================
    ! NON-BONDED INTERACTIONS
    !==============================================================================================
    !-------------------------------------------------------------------------------
    ! 1. add/remove changed vdw and electrostatics  
    !    between (all non-rx_atom1234)-(MRMD ATOMS(=>within rx_atom1234))
    !    - for all NBXMOD value ad their distance larger than 1-4 in CHARMM and all PES
    !    - remove with CHARMM parameters for the MRMD atom
    !    - add    with PES parameters for the MRMD atom
    ! 2. add/remove vdw and electrostatics between certain pairs of rx_atom1234 atoms
    !    - based on NBXMOD value
    !    - remove using arrays rx_exc_12, rx_exc_13, rx_exc_14, rx_exc_15 arrays
    !      using CHARMM parameters ! 
    !    -    add using arrays rx_inc_12, rx_inc_13, rx_inc_14, rx_inc_15 arrays
    !      using PES/CHARMM parameters !

    ! square of cut off distance of non-bonded interaction
    C2OFNB=CTOFNB*CTOFNB          ! roff^2  

    ! strings of atom positions in the connectivity network
    rx_nbdist(0)=' (1)-'
    rx_nbdist((/1,5/))='-(>4)'
    rx_nbdist(2)='-( 2)'
    rx_nbdist(3)='-( 3)'
    rx_nbdist(4)='-( 4)'
     
    do dist=-5,5
        
        ! association pointer with exclusion/inclusion arrays
        select case(dist)
            case(0)
            !  0: skip
                cycle
            case(-1)
            ! -1: 1-(>4) exclusion between MRMD_ATOM & non-rx_atom1234 atoms
                ! no mrmd atoms=>cycle
                if (rx_natom==0.or.rx_natom1234==0) cycle
            case( 1)
            !  1: 1-(>4) inclusion between MRMD_ATOM & non-rx_atom1234 atoms
                ! no mrmd atoms=>cycle
                if (rx_natom==0.or.rx_natom1234==0) cycle
            case(-2)
            ! -2: 1-2    exclusion among rx_atom1234 atoms
                nxxc=rx_exc_n12(isurf)  
                if (nxxc==0) cycle
                xxc=>rx_exc_12(isurf,1:nxxc,1:2)
            case( 2)
            !  2: 1-2    inclusion among rx_atom1234 atoms 
                nxxc=rx_inc_n12(isurf)  
                if (nxxc==0) cycle
                xxc=>rx_inc_12(isurf,1:nxxc,1:3)
            case(-3)
            ! -3: 1-3    exclusion among rx_atom1234 atoms
                nxxc=rx_exc_n13(isurf)  
                if (nxxc==0) cycle
                xxc=>rx_exc_13(isurf,1:nxxc,1:2)
            case( 3)
            !  3: 1-3    inclusion among rx_atom1234 atoms 
                nxxc=rx_inc_n13(isurf)  
                if (nxxc==0) cycle
                xxc=>rx_inc_13(isurf,1:nxxc,1:3)
            case(-4)
            ! -4: 1-4    exclusion among rx_atom1234 atoms
                nxxc=rx_exc_n14(isurf)  
                if (nxxc==0) cycle
                xxc=>rx_exc_14(isurf,1:nxxc,1:2)
            case( 4)
            !  4: 1-4    inclusion among rx_atom1234 atoms
                nxxc=rx_inc_n14(isurf)  
                if (nxxc==0) cycle
                xxc=>rx_inc_14(isurf,1:nxxc,1:3)
            case(-5)
            ! -5: 1-(>4) exclusion among rx_atom1234 atoms
                nxxc=rx_exc_n15(isurf)
                if (nxxc==0) cycle
                xxc=>rx_exc_15(isurf,1:nxxc,1:2)
            case( 5)
            !  5: 1-(>4) inclusion among rx_atom1234 atoms
                nxxc=rx_inc_n15(isurf)  
                if (nxxc==0) cycle
                xxc=>rx_inc_15(isurf,1:nxxc,1:3)
            case default
                ! self checking for further developments
                If (WRNLEV>=5) &
                write(outu,'(A10,A,I2)') 'MRMD_SURF>',' dist:',dist
                call wrndie(-3,'<mrmd.src> mrmd_surf','Not allowed value for dist')
        end select
        
        !------------------------------------------------------------------------------------------
        ! NBXMOD meaning
        ! http://www.charmm.org/ubbthreads/ubbthreads.php?ubb=showflat&Number=23646
        !------------------------------------------------------------------------------------------
        ! Standard CHARMM FORCEFIELD was determined with NBXMOD=5
        ! IF NBXMOD>0: causes the explicit exclusions in the CHARMM (inb array)
        !              note, in the topology file and thus in the CHARMM, the angle, proper and 
        !              improper dihedrals don't require that atoms are connected by bonds
        ! IF NBXMOD<0: use of only the bond connectivity data to construct the exclusion list
        !              and ignore CHARMM 

        ! NBXMod =     0        Preserve the existing exclusion lists
        ! NBXMod = +/- 1        Add nothing to the exclusion list

        ! NBXMod = +/- 2        - exclude 1-2  
        !                       - include 1-3,1-4,1-(>4) using first set of LJ parmeters
        if (nbxmod==2.and.abs(dist)==2) cycle

        ! NBXMod = +/- 3        - exclude 1-2,1-3  
        !                       - include 1-4, 1-(>4) using first set of LJ parmeters
        if (nbxmod==3.and.any(abs(dist)==(/2,3/)) ) cycle

        ! NBXMod = +/- 4        - exclude 1-2, 1-3,1-4  
        !                       - include 1-(>4) using first set of LJ parmeters
        if (nbxmod==4.and.any(abs(dist)==(/2,3,4/)) ) cycle

        ! NBXMod = +/- 5        - exclude 1-2, 1-3 
        !                       - include 1-4 with second set of LJ parameters if they are provided,
        !                         if not, than use first set of parameters
        !                       - include 1-(>4) using first set of LJ parmeters
        if (nbxmod==5.and.any(abs(dist)==(/2,3/)) ) cycle

        ! setting +/-1 factor for sign of change exclusion/inclusion    
        ! +1 for inclusions
        ! -1 for exclusions
        pmone=sign(1,dist)
        if (pmone==-1) spmone='-'
        if (pmone==+1) spmone='+'
        
        ! reseting counters, indices
        ixxc  =0
        psf_iatom=0
        iatom =0
        do 
            !---------------------------------------------------------
            ! add/remove changed between MRMD ATOM ad non-rx_atom1234
            ! 1. add/remove changed vdw and electrostatics 
            !    between (all non-rx_atom1234)-(MRMD ATOMS(=>within rx_atom1234))
            !    - for all NBXMOD value ad their distance larger than 1-4 in CHARMM and all PES
            !    - remove with CHARMM parameters for the MRMD atom
            !    - add    with PES parameters for the MRMD atom
            !
            !    - these are alway farthers from each other than 1-4 distance
            !      => second set of (special) vdw parameters are never used
            !---------------------------------------------------------
            if (abs(dist)==1.and.rx_natom>0) then

                !----------------------------
                ! ATOM 1: all non-rx_atom1234=> they are CHARMM atoms not in rx_atom1234
                !----------------------------

                ! increase CHARMM counter if iatom=0 (or whenever it reset to zero)
                if (iatom==0) psf_iatom=psf_iatom+1
                
                ! beyond max value=> all done => exit
                if (psf_iatom>natom) exit
                
                ! atom in rx_atom1234 list=> those have already handled=> skip
                if (rx_atom1234_inv(psf_iatom)>0) cycle

                ! shorter notation for CHARMM index
                II  = psf_iatom

                ! CHARMM parameters (not changed)
                !    - these are alway farthers from each other than 1-4 distance
                !      => second set of (special) vdw parameters are never used
                rx_rminp2(1) = VDWR(ITC(IAC(II)))
                rx_eps   (1) =  EFF(ITC(IAC(II)))
                rx_chrg  (1) =  CG(         II)                     

                !----------------------------
                ! ATOM 2: MRMD ATOMS
                !----------------------------
                ! increase counter => should never reach rx_natom+1, as it is reset
                iatom=iatom+1
                
                ! if rx_natom=0 exit=> tested earlier
                !if (iatom>rx_natom) exit
                
                ! CHARMM index of mrmd atom 
                JJ=rx_atom(iatom)
                                        
                ! atom index in CHARMM
                select case (dist)
                    case(-1)    ! remove interaction with CHARMM parameters
                        rx_rminp2(2) = VDWR(ITC(IAC(JJ)))
                        rx_eps   (2) =  EFF(ITC(IAC(JJ)))
                        rx_chrg  (2) =  CG(         JJ)                     
                    case( 1)    ! add interaction with PES parameters
                        rx_rminp2(2) = rx_vdw_rmin_half(isurf,iatom)
                        rx_eps   (2) = rx_vdw_eps      (isurf,iatom)
                        rx_chrg  (2) = rx_charges      (isurf,iatom)            
                end select
                
                ! resetting iatom counter when it reached rx_natom
                if (iatom==rx_natom) iatom=0
            endif
            
            !--------------------------------------------------------------
            ! 2. add/remove vdw and electrostatics between changed pairs of rx_atom1234 atoms
            !    - based on NBXMOD value
            !    - remove using arrays rx_exc_12, rx_exc_13, rx_exc_14, rx_exc_15 arrays
            !      using CHARMM parameters ! 
            !    -    add using arrays rx_inc_12, rx_inc_13, rx_inc_14, rx_inc_15 arrays
            !      using PES/CHARMM parameters !
            !--------------------------------------------------------------
            if (2<=abs(dist).and.abs(dist)<=5) then
                ! increase counter
                ixxc=ixxc+1
                
                ! beyond the last => exit loop 
                if (ixxc>nxxc) exit
                
                ! assign new parameters for both atoms
                do j=1,2
                    ! atom index in CHARMM
                    psf_iatom=xxc(ixxc,j)
                    
                    ! atom index in MRMD
                    iatom =rx_atom_inv(psf_iatom)

                    ! MRMD/non-MRMD atom
                    if (iatom>0) then
                        ! MRMD atom, get its CHARMM   parameters for exclusion
                        !                     MRMD parameters for inclusion  
                        select case (dist)
                            case(-5:-2)    ! remove interaction with CHARMM parameters
                                ! for NBXMOD=5 and 1-4 interaction=> special vdW
                                if (NBXMOD==5.and.dist==-4) then
                                    ! use special vdw parameters = second set of LJ parameters 
                                    rx_rminp2(j) = VDWR(ITC(IAC(psf_iatom))+MAXATC)
                                    rx_eps   (j) =  EFF(ITC(IAC(psf_iatom))+MAXATC)
                                else
                                    rx_rminp2(j) = VDWR(ITC(IAC(psf_iatom)))
                                    rx_eps   (j) =  EFF(ITC(IAC(psf_iatom)))
                                endif
                                rx_chrg  (j) =  CG(         psf_iatom)
                                
                            case( 2: 5)    ! add interaction with PES parameters 
                                if (NBXMOD==5.and.dist==4) then
                                    ! for NBXMOD=5 and 1-4 interaction
                                    ! use special vdw parameters = second set of LJ parameters 
                                    rx_rminp2(j) = rx_vdw_rmin_half2(isurf,iatom)
                                    rx_eps   (j) = rx_vdw_eps2      (isurf,iatom)
                                else
                                    rx_rminp2(j) = rx_vdw_rmin_half (isurf,iatom)
                                    rx_eps   (j) = rx_vdw_eps       (isurf,iatom)
                                endif
                                rx_chrg  (j) = rx_charges     (isurf,iatom)            
                        end select
                    else
                        ! this is not a MRMD atom => use CHARMM parameters for
                        ! both the removal and the adding

                        if (NBXMOD==5.and.abs(dist)==4) then
                            ! for NBXMOD=5 and 1-4 interaction
                            ! use special vdw parameters = second set of LJ parameters 
                            rx_rminp2(j) = VDWR(ITC(IAC(psf_iatom))+MAXATC)
                            rx_eps   (j) =  EFF(ITC(IAC(psf_iatom))+MAXATC)
                        else
                            rx_rminp2(j) = VDWR(ITC(IAC(psf_iatom)))
                            rx_eps   (j) =  EFF(ITC(IAC(psf_iatom)))
                        endif
                        rx_chrg  (j) =  CG(         psf_iatom)
                    endif
                enddo 
                
                ! CHARMM INDEX OF ATOMS
                II  = xxc(ixxc,1)
                JJ  = xxc(ixxc,2)
            endif
            
            !-----------------------------------------------
            ! ENERGY EXCLUSION/INCLUSION CALCULATION
            !-----------------------------------------------
                            
            ! DISTANCE^2 OF ATOMS
            S = (X(II)-X(JJ))**2+(Y(II)-Y(JJ))**2+(Z(II)-Z(JJ))**2                

            !-----------------------------------------------
            ! ELECTROSTATIC OF POINT CHARGES WITH SHIFTING 
            ! if beyond switchoff distance=> skip            
            !-----------------------------------------------
            if (QETERM(ELEC).and.s<C2OFNB.and.rx_chrg(1)/=0._chm_real &
                 .and.rx_chrg(2)/=0._chm_real) then
                !------------------------------------------------
                ! exclusion (dist<0), inclusion(dist>0)
                !------------------------------------------------
                if (rx_calcgrad) then
                    call MRMD_ELEC(natomx,X,Y,Z,ii,jj,rx_chrg(1),rx_chrg(2), &
                               dist,geompar,denergy,dfx(1:2),dfy(1:2),dfz(1:2))
                    dEdx((/ii,jj/))=dEdx((/ii,jj/))+pmone*dfx(1:2)
                    dEdy((/ii,jj/))=dEdy((/ii,jj/))+pmone*dfy(1:2)
                    dEdz((/ii,jj/))=dEdz((/ii,jj/))+pmone*dfz(1:2)
                else
                   call MRMD_ELEC(natomx,X,Y,Z,ii,jj,rx_chrg(1),rx_chrg(2),dist,geompar,denergy)
                endif

                if (rx_print_energy.and.PRNLEV>=8) then
                    write(outu,'(A10,x,I4,x,A10,x,G12.5,2(x,I5),2(x,A5),4(x,G11.4))') & 
                    'MRMD_SURF>', &
                    isurf,spmone//'nbon elec',pmone*denergy,ii,jj,rx_nbdist(0), &
                    rx_nbdist(abs(dist)),rx_chrg(1),rx_chrg(2),-9999._chm_real,geompar

                    if (PRNLEV>=10.and.rx_calcgrad) then
                        write(outu,'(4(A10,x))') &
                        'atom_i',spmone//'dV_dxi',spmone//'dV_dyi',spmone//'dV_dzi'
                        write(outu,'(I10,x,3(F10.4,x))') ii,pmone*dfx(1),pmone*dfy(1),pmone*dfz(1)
                        write(outu,'(I10,x,3(F10.4,x))') jj,pmone*dfx(2),pmone*dfy(2),pmone*dfz(2)
                    endif
                endif
                
                energies(pmone*1)=energies(pmone*1)+denergy
            endif

            !----------------------------------------
            ! LJ/GLJ energy with switching function
            !----------------------------------------
            if (QETERM(VDW).and.s<C2OFNB) then

                ! is it general vdW inclusion?
                ! yes <=> igvdw>0 
                ! no  <=> igvdw==0
                igvdw=0
                if (dist>=2) igvdw=xxc(ixxc,3)
                
                ! is it normal/special(atomindex>0) or general vdw (atomindex<0)?
                ! II, JJ are always positive
                if (igvdw==0) then
                !------------------------------------------------
                ! normal and special vdw
                ! exclusion (dist<0), inclusion(dist>0)
                !------------------------------------------------
                  if (rx_eps(1)/=0._chm_real.and.rx_eps(2)/=0._chm_real) then
                    if (rx_calcgrad) then
                        call MRMD_VDW(natomx,X,Y,Z,ii,jj,rx_rminp2(1),rx_eps(1),&
                        rx_rminp2(2),rx_eps(2),geompar,denergy,dfx(1:2),dfy(1:2),dfz(1:2))
                        dEdx((/ii,jj/))=dEdx((/ii,jj/))+pmone*dfx(1:2)
                        dEdy((/ii,jj/))=dEdy((/ii,jj/))+pmone*dfy(1:2)
                        dEdz((/ii,jj/))=dEdz((/ii,jj/))+pmone*dfz(1:2)
                    else
                        call MRMD_VDW(natomx,X,Y,Z,ii,jj,rx_rminp2(1),rx_eps(1), &
                                  rx_rminp2(2),rx_eps(2),geompar,denergy)
                    endif
                    ! add energy
                    energies(pmone*2)=energies(pmone*2)+denergy
                    
                    ! print energy
                    if (rx_print_energy.and.PRNLEV>=8) then
                        write(outu,'(A10,x,I4,x,A10,x,G12.5,2(x,I5),2(x,A5),4(x,G11.4))') & 
                        'MRMD_SURF>', &
                        isurf,spmone//'nbon vdw ',pmone*denergy,ii,jj,rx_nbdist(0), &
                        rx_nbdist(abs(dist)),sqrt(rx_eps(1)*rx_eps(2)),&
                        -9999._chm_real,sum(rx_rminp2(1:2)),geompar
                    endif                        
                  endif  
                else
                !-------------------------------------------------------------
                ! general vdw
                !------------------------------------------------------------- 
                  if (rx_gvdw_eps(isurf,igvdw)/=0._chm_real) then
                    if (rx_calcgrad) then
                        ! energy+forces
                        call MRMD_GVDW(natomx,X,Y,Z,ii,jj,rx_gvdw_eps(isurf,igvdw),&
                        rx_gvdw_rmin(isurf,igvdw),rx_gvdw_xatt(isurf,igvdw), &
                        rx_gvdw_xrep(isurf,igvdw),geompar,denergy,&
                        dfx(1:2),dfy(1:2),dfz(1:2))
                        ! always inclusion
                        dEdx((/ii,jj/))=dEdx((/ii,jj/))+dfx(1:2)
                        dEdy((/ii,jj/))=dEdy((/ii,jj/))+dfy(1:2)
                        dEdz((/ii,jj/))=dEdz((/ii,jj/))+dfz(1:2)                        
                    else
                        !only energy
                        call MRMD_GVDW(natomx,X,Y,Z,ii,jj,rx_gvdw_eps (isurf,igvdw),&
                        rx_gvdw_rmin(isurf,igvdw),rx_gvdw_xatt(isurf,igvdw), &
                        rx_gvdw_xrep(isurf,igvdw),geompar,denergy)
                    endif
                    
                    ! just inclusion
                    energies(3)=energies(3)+denergy

                    ! print energy
                    if (rx_print_energy.and.PRNLEV>=8) then
                        write(outu,'(A10,x,I4,x,A10,x,G12.5,2(x,I5),2(x,A5),&
                        x,G11.4,2(x,F5.2),2(x,G11.4))')                    & 
                            'MRMD_SURF>', &
                            isurf,'+nbon gvdw',denergy,ii,jj,rx_nbdist(0),rx_nbdist(abs(dist)),&
                            rx_gvdw_eps (isurf,igvdw), &
                            rx_gvdw_xatt(isurf,igvdw), &
                            rx_gvdw_xrep(isurf,igvdw), &
                            rx_gvdw_rmin(isurf,igvdw), &
                            geompar
                    endif
                  endif  
                endif

                if (rx_print_energy.and.PRNLEV>=10.and.rx_calcgrad) then
                    write(outu,'(4(A10,x))') &
                    'atom_i',spmone//'dV_dxi',spmone//'dV_dyi',spmone//'dV_dzi'
                    write(outu,'(I10,x,3(F10.4,x))') ii,pmone*dfx(1),pmone*dfy(1),pmone*dfz(1)
                    write(outu,'(I10,x,3(F10.4,x))') jj,pmone*dfx(2),pmone*dfy(2),pmone*dfz(2)
                endif
            endif
        enddo
    enddo

    !==============================================================================================
    ! BOND HARM POTENTIALS
    !==============================================================================================
    if (QETERM(BOND)) then
        !------------------------------------
        ! exlusion list of CHARMM harmonic bonds
        !------------------------------------
        do i=1,rx_exc_nbond

            ! CHARMM index of harmonic bond
            ibond=rx_exc_bond(i)

            ! its atoms
            II  =   IB(ibond)
            JJ  =   JB(ibond) 


            ! its parameters
            IX=ICB(ibond)
            FCON=CBC(IX)
            REQ =CBB(IX)  
            
            ! calculate energy and gradient
            ! gradient is summed and not stored separately 
            if (rx_calcgrad) then
                call MRMD_HARM(natomx,X,Y,Z,II,JJ,FCON,REQ,geompar,denergy,&
                           dfx(1:2),dfy(1:2),dfz(1:2))
                dEdx((/ii,jj/))=dEdx((/ii,jj/))-dfx(1:2)
                dEdy((/ii,jj/))=dEdy((/ii,jj/))-dfy(1:2)
                dEdz((/ii,jj/))=dEdz((/ii,jj/))-dfz(1:2)
            else
                call MRMD_HARM(natomx,X,Y,Z,II,JJ,FCON,REQ,geompar,denergy)
            endif

            if (rx_print_energy.and.PRNLEV>=7) then
                write(outu,'(A10,x,I4,x,A10,x,G12.5,4(x,I5),4(x,G11.4))') &
                'MRMD_SURF>', &
                isurf,'-bond harm',-denergy,ii,jj,-9999,-9999,FCON,-9999._chm_real,REQ,geompar
                if (PRNLEV>=10.and.rx_calcgrad) then
                     write(outu,'(4(A10,x))') 'atom_i','-dV_dxi','-dV_dyi','-dV_dzi'
                     write(outu,'(I10,x,3(F10.4,x))') ii,-dfx(1),-dfy(1),-dfz(1)
                     write(outu,'(I10,x,3(F10.4,x))') jj,-dfx(2),-dfy(2),-dfz(2)
                 endif
            endif
                            
            energies(-4)=energies(-4)+denergy
        enddo
        
        !---------------------------------------
        ! inclusion list of MRMD harmonic bonds
        !---------------------------------------
        
        do i=1,rx_inc_nharm(isurf)

            ! MRMD index of harmonic bond
            iharm=rx_inc_harm(isurf,i)

            ! its atoms
            II     = rx_harm_atoms(iharm,1)
            JJ     = rx_harm_atoms(iharm,2)

            ! its parameters
            FCON   = rx_harm_fc_half(isurf,iharm)
            REQ    = rx_harm_re     (isurf,iharm)

            ! its energy
            ! calculate energy and gradient
            ! gradient is incremental
            if (rx_calcgrad) then
                call MRMD_HARM(natomx,X,Y,Z,II,JJ,FCON,REQ,geompar,denergy,&
                            dfx(1:2),dfy(1:2),dfz(1:2))
                dEdx((/ii,jj/))=dEdx((/ii,jj/))+dfx(1:2)
                dEdy((/ii,jj/))=dEdy((/ii,jj/))+dfy(1:2)
                dEdz((/ii,jj/))=dEdz((/ii,jj/))+dfz(1:2)
            else
                call MRMD_HARM(natomx,X,Y,Z,II,JJ,FCON,REQ,geompar,denergy)
            endif
            
            if (rx_print_energy.and.PRNLEV>=7) then
                write(outu,'(A10,x,I4,x,A10,x,G12.5,4(x,I5),4(x,G11.4))') &
                'MRMD_SURF>', &
                isurf,'+bond harm',denergy,ii,jj,-9999,-9999,FCON,-9999._chm_real,REQ,geompar
                if (PRNLEV>=10.and.rx_calcgrad) then
                    write(outu,'(4(A10,x))') 'atom_i','+dV_dxi','+dV_dyi','+dV_dzi'
                    write(outu,'(I10,x,3(F10.4,x))') ii,dfx(1),dfy(1),dfz(1)
                    write(outu,'(I10,x,3(F10.4,x))') jj,dfx(2),dfy(2),dfz(2)
                endif   
            endif
                                     
            energies(4)=energies(4)+denergy
        enddo


        !----------------------------------
        ! exclusion list of CHARMM Morse bonds
        !----------------------------------
        ! they do not exists
    
        !------------------------------------
        ! inclusion list of MRMD Morse bonds
        !------------------------------------
        do i=1,rx_inc_nmors(isurf)
            ! MRMD index of morse bond
            imors=rx_inc_mors(isurf,i)

            ! its atoms
            II  = rx_mors_atoms(imors,1)
            JJ  = rx_mors_atoms(imors,2) 

            ! its parameters
            DE    = rx_mors_De  (isurf,imors)
            BETA  = rx_mors_beta(isurf,imors) 

            REQ   = rx_mors_re  (isurf,imors)
            
            ! calculate energy and gradient
            if (rx_calcgrad) then
              call MRMD_MORS(natomx,X,Y,Z,II,JJ,DE,BETA,REQ,geompar,denergy,&
                         dfx(1:2),dfy(1:2),dfz(1:2))
                dEdx((/ii,jj/))=dEdx((/ii,jj/))+dfx(1:2)
                dEdy((/ii,jj/))=dEdy((/ii,jj/))+dfy(1:2)
                dEdz((/ii,jj/))=dEdz((/ii,jj/))+dfz(1:2)
            else
                call MRMD_MORS(natomx,X,Y,Z,II,JJ,DE,BETA,REQ,geompar,denergy)
            endif

            if (rx_print_energy.and.PRNLEV>=7) then
                write(outu,'(A10,x,I4,x,A10,x,G12.5,4(x,I5),4(x,G11.4))') &
                'MRMD_SURF>', &
                isurf,'+bond mors',denergy,ii,jj,-9999,-9999,DE,BETA,REQ,geompar
                if (PRNLEV>=10.and.rx_calcgrad) then
                    write(outu,'(4(A10,x))') 'atom_i','+dV_dxi','+dV_dyi','+dV_dzi'
                    write(outu,'(I10,x,3(F10.4,x))') ii,dfx(1),dfy(1),dfz(1)
                    write(outu,'(I10,x,3(F10.4,x))') jj,dfx(2),dfy(2),dfz(2)
                endif  
            endif
                            
            energies(5)=energies(5)+denergy
        enddo
        
        !----------------------------------
        ! exclusion list of CHARMM RKHS bonds
        !----------------------------------
        ! they do not exist
    
        !------------------------------------
        ! inclusion list of MRMD RKHS bonds
        !------------------------------------
        do i=1,rx_inc_nrkhs(isurf)
            ! MRMD index of RKHS bond
            irkhs=rx_inc_rkhs(isurf,i)

            ! its atoms
            II  = rx_rkhs_atoms(irkhs,1)
            JJ  = rx_rkhs_atoms(irkhs,2) 
            
            ! calculate energy and gradient
            j = rx_rkhs_ngrid(irkhs,isurf) !ngrid, temporarily stored in j
            if (rx_calcgrad) then
              call MRMD_RKHS(natomx,X,Y,Z,II,JJ,j,rx_rkhs_grid(1:j,irkhs,isurf),&
                             rx_rkhs_coef(1:j,irkhs,isurf),rx_rkhs_asym(irkhs,isurf),&
                             geompar,denergy,dfx(1:2),dfy(1:2),dfz(1:2))
                dEdx((/ii,jj/))=dEdx((/ii,jj/))+dfx(1:2)
                dEdy((/ii,jj/))=dEdy((/ii,jj/))+dfy(1:2)
                dEdz((/ii,jj/))=dEdz((/ii,jj/))+dfz(1:2)
            else
              call MRMD_RKHS(natomx,X,Y,Z,II,JJ,j,rx_rkhs_grid(1:j,irkhs,isurf),&
                             rx_rkhs_coef(1:j,irkhs,isurf),rx_rkhs_asym(irkhs,isurf),&
                             geompar,denergy)
            endif

            if (rx_print_energy.and.PRNLEV>=7) then
                write(outu,'(A10,x,I4,x,A10,x,G12.5,4(x,I5),x,G11.4)') &
                'MRMD_SURF>', &
                isurf,'+bond rkhs',denergy,ii,jj,-9999,-9999,geompar
                if (PRNLEV>=10.and.rx_calcgrad) then
                    write(outu,'(4(A10,x))') 'atom_i','+dV_dxi','+dV_dyi','+dV_dzi'
                    write(outu,'(I10,x,3(F10.4,x))') ii,dfx(1),dfy(1),dfz(1)
                    write(outu,'(I10,x,3(F10.4,x))') jj,dfx(2),dfy(2),dfz(2)
                endif  
            endif
                            
            energies(10)=energies(10)+denergy
        enddo
        
    endif

    !==============================================================================================
    ! ANGLE POTENTIALS 
    !==============================================================================================
    if (QETERM(ANGLE)) then
    
        !-------------------------------
        ! exclusion list of CHARMM angles 
        !-------------------------------
        do i=1,rx_exc_nangl(isurf)
            ! CHARMM index of angle potential
            iangl=rx_exc_angl(isurf,i)
        
            ! its atoms
            II = IT(iangl)
            JJ = JT(iangl)
            KK = KT(iangl)

            ! its parameters
            IX=ICT(iangl)
            FCON = CTC(IX)
            TEQ  = CTB(IX)
            
            ! calculate energy and gradient
            if (rx_calcgrad) then
              call MRMD_ANGL(natomx,X,Y,Z,II,JJ,KK,FCON,TEQ,geompar,denergy,&
                         dfx(1:3),dfy(1:3),dfz(1:3))
                dEdx((/ii,jj,kk/))=dEdx((/ii,jj,kk/))-dfx(1:3)
                dEdy((/ii,jj,kk/))=dEdy((/ii,jj,kk/))-dfy(1:3)
                dEdz((/ii,jj,kk/))=dEdz((/ii,jj,kk/))-dfz(1:3)
            else
                call MRMD_ANGL(natomx,X,Y,Z,II,JJ,KK,FCON,TEQ,geompar,denergy)
            endif

            if (rx_print_energy.and.PRNLEV>=7) then
                write(outu,'(A10,x,I4,x,A10,x,G12.5,4(x,I5),4(x,G11.4))') &
                'MRMD_SURF>', &
                isurf,'-angl harm',-denergy,ii,jj,kk,-9999,FCON,-9999._chm_real,&
                TEQ*raddeg,geompar*raddeg
                if (PRNLEV>=10.and.rx_calcgrad) then
                    write(outu,'(4(A10,x))') 'atom_i','-dV_dxi','-dV_dyi','-dV_dzi'
                    write(outu,'(I10,x,3(F10.4,x))') ii,-dfx(1),-dfy(1),-dfz(1)
                    write(outu,'(I10,x,3(F10.4,x))') jj,-dfx(2),-dfy(2),-dfz(2)
                    write(outu,'(I10,x,3(F10.4,x))') kk,-dfx(3),-dfy(3),-dfz(3)
                endif  
            endif
            
            energies(-6)=energies(-6)+denergy
        enddo
        
        !-------------------------------
        ! inclusion list of MRMD angles 
        !-------------------------------
        do i=1,rx_inc_nangl(isurf)   
            ! MRMD index of angle potential
            iangl=rx_inc_angl(isurf,i)    

            ! its atoms
            II = rx_angl_atoms(iangl,1)
            JJ = rx_angl_atoms(iangl,2)
            KK = rx_angl_atoms(iangl,3)

            ! its parameters
            FCON = rx_angl_fc_half(isurf,iangl)
            TEQ  = rx_angl_phie   (isurf,iangl)

            ! calculate energy and gradient
            if (rx_calcgrad) then
              call MRMD_ANGL(natomx,X,Y,Z,II,JJ,KK,FCON,TEQ,geompar,denergy,&
                         dfx(1:3),dfy(1:3),dfz(1:3))
                dEdx((/ii,jj,kk/))=dEdx((/ii,jj,kk/))+dfx(1:3)
                dEdy((/ii,jj,kk/))=dEdy((/ii,jj,kk/))+dfy(1:3)
                dEdz((/ii,jj,kk/))=dEdz((/ii,jj,kk/))+dfz(1:3)
            else
                call MRMD_ANGL(natomx,X,Y,Z,II,JJ,KK,FCON,TEQ,geompar,denergy)
            endif

            if (rx_print_energy.and.PRNLEV>=7) then
                write(outu,'(A10,x,I4,x,A10,x,G12.5,4(x,I5),4(x,G11.4))') &
                'MRMD_SURF>', &
                isurf,'+angl harm',denergy,ii,jj,kk,-9999,FCON,-9999._chm_real,&
                TEQ*raddeg,geompar*raddeg
                if (PRNLEV>=10.and.rx_calcgrad) then
                    write(outu,'(4(A10,x))') 'atom_i','+dV_dxi','+dV_dyi','+dV_dzi'
                    write(outu,'(I10,x,3(F10.4,x))') ii,dfx(1),dfy(1),dfz(1)
                    write(outu,'(I10,x,3(F10.4,x))') jj,dfx(2),dfy(2),dfz(2)
                    write(outu,'(I10,x,3(F10.4,x))') kk,dfx(3),dfy(3),dfz(3)
                endif  
            endif
            energies(6)=energies(6)+denergy
        enddo
    endif

    !==============================================================================================
    ! UREY BRADLEY POTENTIALS  
    !==============================================================================================
    if (QETERM(UREYB)) then
        !--------------------------------
        ! exclusion list of Urey-Bradleys ! coupled with angle potential 
        !--------------------------------
        do i=1,rx_exc_nangl(isurf)  !coupled with angle potential!!!!
            ! CHARMM index of angle potential
            iangl=rx_exc_angl(isurf,i)
            
            ! Is there corresponding Urey-Bradley potential in the CHARMM
            ! index in MRMD
            j=rx_angl_inv(iangl) 
            if (j>0) then
                if (.not.rx_angl_exist(0,j)) cycle
            endif
            
            ! its 1st and 3rd atoms
            II = IT(iangl)
            KK = KT(iangl)

            ! UREY BRADLEY parameters
            IX   = ICT(iangl)
            FCON = CTUC(IX)
            REQ  = CTUB(IX)

            ! if not present or has zero force constant (half force constant)
            if (FCON<=0._chm_real) cycle 
            
            ! calculate energy and gradient
            if (rx_calcgrad) then
                call MRMD_HARM(natomx,X,Y,Z,II,KK,FCON,REQ,geompar,denergy,&
                           dfx(1:2),dfy(1:2),dfz(1:2))
                dEdx((/ii,kk/))=dEdx((/ii,kk/))-dfx(1:2)
                dEdy((/ii,kk/))=dEdy((/ii,kk/))-dfy(1:2)
                dEdz((/ii,kk/))=dEdz((/ii,kk/))-dfz(1:2)
            else
                call MRMD_HARM(natomx,X,Y,Z,II,KK,FCON,REQ,geompar,denergy)
            endif

            if (rx_print_energy.and.PRNLEV>=7) then
                write(outu,'(A10,x,I4,x,A10,x,G12.5,4(x,I5),4(x,G11.4))') &
                'MRMD_SURF>', &
                isurf,'-angl urey',-denergy,ii,jj,kk,-9999,FCON,-9999._chm_real,REQ,geompar
                if (PRNLEV>=10.and.rx_calcgrad) then
                    write(outu,'(4(A10,x))') 'atom_i','-dV_dxi','-dV_dyi','-dV_dzi'
                    write(outu,'(I10,x,3(F10.4,x))') ii,-dfx(1),-dfy(1),-dfz(1)
                    write(outu,'(I10,x,3(F10.4,x))') kk,-dfx(2),-dfy(2),-dfz(2)
                endif  
           endif
            
            energies(-7)=energies(-7)+denergy
        enddo
        !--------------------------------
        ! inclusion list of Urey-Bradleys  
        !--------------------------------
        do i=1,rx_inc_nangl(isurf) !!!coupled with angle potential  

            ! MRMD index of angle 
            iangl=rx_inc_angl(isurf,i)    

            ! is it present on surface isurf    
            if (.not.rx_urey_exist(isurf,iangl)) cycle

            ! its 1st and 3rd atoms
            II = rx_angl_atoms(iangl,1)
            KK = rx_angl_atoms(iangl,3)

            ! Urey-Bradley parameters
            FCON = rx_urey_fc_half(isurf,iangl)
            REQ  = rx_urey_re     (isurf,iangl)

            ! calculate energy and gradient
            if (rx_calcgrad) then
                call MRMD_HARM(natomx,X,Y,Z,II,KK,FCON,REQ,geompar,denergy, &
                           dfx(1:2),dfy(1:2),dfz(1:2))
                dEdx((/ii,kk/))=dEdx((/ii,kk/))+dfx(1:2)
                dEdy((/ii,kk/))=dEdy((/ii,kk/))+dfy(1:2)
                dEdz((/ii,kk/))=dEdz((/ii,kk/))+dfz(1:2)
            else
                call MRMD_HARM(natomx,X,Y,Z,II,KK,FCON,REQ,geompar,denergy)
            endif

            if (rx_print_energy.and.PRNLEV>=7) then
                write(outu,'(A10,x,I4,x,A10,x,G12.5,4(x,I5),4(x,G11.4))') &
                'MRMD_SURF>', &
                isurf,'+angl urey',denergy,ii,jj,kk,-9999,FCON,-9999._chm_real,REQ,geompar
                if (PRNLEV>=10.and.rx_calcgrad) then
                    write(outu,'(4(A10,x))') 'atom_i','+dV_dxi','+dV_dyi','+dV_dzi'
                    write(outu,'(I10,x,3(F10.4,x))') ii,dfx(1),dfy(1),dfz(1)
                    write(outu,'(I10,x,3(F10.4,x))') kk,dfx(2),dfy(2),dfz(2)
                endif  
            endif
            energies(7)=energies(7)+denergy
        enddo
    endif

    !==============================================================================================
    ! PROPER DIHEDRAL POTENTIALS
    !==============================================================================================
    if (QETERM(DIHE)) then
        !--------------------------------
        ! exclusion list of CHARMM dihedrals
        !--------------------------------
        do i=1,rx_exc_ndihe(isurf)
            ! MRMD index of dihedral potential
            idihe=rx_exc_dihe(isurf,i,1)
            
            ! its atoms
            II = IP(idihe)
            JJ = JP(idihe)
            KK = KP(idihe)
            LL = LP(idihe)

            ! its parameters   
            IX=ICP(idihe)+ rx_exc_dihe(isurf,i,2) ! index increment
            FCON=CPC(IX)
            MULT=abs(CPD(IX))
            PEQ =CPB(IX)
            PEQC=CPCOS(IX)
            PEQS=CPSIN(IX)

            ! calculate energy and gradient
            if (rx_calcgrad) then
                call MRMD_DIHE(natomx,X,Y,Z,II,JJ,KK,LL,FCON,MULT,PEQ,PEQC,PEQS,&
                           geompar,denergy,dfx(1:4),dfy(1:4),dfz(1:4))
                dEdx((/ii,jj,kk,ll/))=dEdx((/ii,jj,kk,ll/))-dfx(1:4)
                dEdy((/ii,jj,kk,ll/))=dEdy((/ii,jj,kk,ll/))-dfy(1:4)
                dEdz((/ii,jj,kk,ll/))=dEdz((/ii,jj,kk,ll/))-dfz(1:4)
            else
                call MRMD_DIHE(natomx,X,Y,Z,II,JJ,KK,LL,FCON,MULT,PEQ,PEQC,PEQS,&
                           geompar,denergy)
            endif

            if (rx_print_energy.and.PRNLEV>=7) then
                write(outu,'(A10,x,I4,x,A10,x,G12.5,4(x,I5),4(x,G11.4))') & 
                'MRMD_SURF>', &
                isurf,'-dihe four',-denergy,ii,jj,kk,ll,FCON,MULT,&
                PEQ*raddeg,geompar*raddeg
                if (PRNLEV>=10.and.rx_calcgrad) then
                    write(outu,'(4(A10,x))') 'atom_i','-dV_dxi','-dV_dyi','-dV_dzi'
                    write(outu,'(I10,x,3(F10.4,x))') ii,-dfx(1),-dfy(1),-dfz(1)
                    write(outu,'(I10,x,3(F10.4,x))') jj,-dfx(2),-dfy(2),-dfz(2)
                    write(outu,'(I10,x,3(F10.4,x))') kk,-dfx(3),-dfy(3),-dfz(3)
                    write(outu,'(I10,x,3(F10.4,x))') ll,-dfx(4),-dfy(4),-dfz(4)
                endif  
            endif
            
            energies(-8)=energies(-8)+denergy
        enddo

        !----------------------------------
        ! inclusion list of MRMD dihedrals
        !----------------------------------
        do i=1,rx_inc_ndihe(isurf)
            ! MRMD index of dihedral potential
            idihe=rx_inc_dihe(isurf,i)

            ! its atoms
            II = rx_dihe_atoms(idihe,1)

            JJ = rx_dihe_atoms(idihe,2)
            KK = rx_dihe_atoms(idihe,3)
            LL = rx_dihe_atoms(idihe,4)

            ! its parameters
            FCON=rx_dihe_k   (isurf,idihe)
            MULT=rx_dihe_mult(         idihe)
            PEQ =rx_dihe_phi0(isurf,idihe)
            PEQC=rx_dihe_cos0(isurf,idihe)
            PEQS=rx_dihe_sin0(isurf,idihe)

            ! calculate energy and gradient
            if (rx_calcgrad) then
               call MRMD_DIHE(natomx,X,Y,Z,II,JJ,KK,LL,FCON,MULT,PEQ,PEQC,PEQS,&
                           geompar,denergy,dfx(1:4),dfy(1:4),dfz(1:4))
                dEdx((/ii,jj,kk,ll/))=dEdx((/ii,jj,kk,ll/))+dfx(1:4)
                dEdy((/ii,jj,kk,ll/))=dEdy((/ii,jj,kk,ll/))+dfy(1:4)
                dEdz((/ii,jj,kk,ll/))=dEdz((/ii,jj,kk,ll/))+dfz(1:4)

            else
                call MRMD_DIHE(natomx,X,Y,Z,II,JJ,KK,LL,FCON,MULT,PEQ,PEQC,PEQS,&
                           geompar,denergy)
            endif

            if (rx_print_energy.and.PRNLEV>=7) then
                write(outu,'(A10,x,I4,x,A10,x,G12.5,4(x,I5),4(x,G11.4))') & 
                'MRMD_SURF>', &
                isurf,'+dihe four',denergy,ii,jj,kk,ll,FCON,MULT,&
                PEQ*raddeg,geompar*raddeg
                if (PRNLEV>=10.and.rx_calcgrad) then
                    write(outu,'(4(A10,x))') 'atom_i','+dV_dxi','+dV_dyi','+dV_dzi'
                    write(outu,'(I10,x,3(F10.4,x))') ii,dfx(1),dfy(1),dfz(1)
                    write(outu,'(I10,x,3(F10.4,x))') jj,dfx(2),dfy(2),dfz(2)
                    write(outu,'(I10,x,3(F10.4,x))') kk,dfx(3),dfy(3),dfz(3)
                    write(outu,'(I10,x,3(F10.4,x))') ll,dfx(4),dfy(4),dfz(4)
                endif  
            endif
            energies(8)=energies(8)+denergy
        enddo
    endif

    !==============================================================================================
    ! IMRPOPER DIHEDRAL POTENTIALS
    !==============================================================================================
    if (QETERM(IMDIHE)) then
        !--------------------------------
        ! exclusion list of CHARMM dihedrals
        !--------------------------------
        do i=1,rx_exc_nimpr(isurf)
            ! MRMD index of improper dihedral potential
            iimpr=rx_exc_impr(isurf,i)

            ! its atoms
            II = IM(iimpr)
            JJ = JM(iimpr)
            KK = KM(iimpr)
            LL = LM(iimpr)

            ! its parameters
            IX=ICI(iimpr)
            FCON=CIC(IX)
            MULT=CID(IX)
            PEQ =CIB(IX)
            PEQC=CICOS(IX)
            PEQS=CISIN(IX)

            ! calculate energy and gradient
            if (rx_calcgrad) then
                call MRMD_DIHE(natomx,X,Y,Z,II,JJ,KK,LL,FCON,MULT,PEQ,PEQC,PEQS,&
                           geompar,denergy,dfx(1:4),dfy(1:4),dfz(1:4))

                dEdx((/ii,jj,kk,ll/))=dEdx((/ii,jj,kk,ll/))-dfx(1:4)
                dEdy((/ii,jj,kk,ll/))=dEdy((/ii,jj,kk,ll/))-dfy(1:4)
                dEdz((/ii,jj,kk,ll/))=dEdz((/ii,jj,kk,ll/))-dfz(1:4)
            else
                call MRMD_DIHE(natomx,X,Y,Z,II,JJ,KK,LL,FCON,MULT,PEQ,PEQC,PEQS,&
                           geompar,denergy)
            endif

            if (rx_print_energy.and.PRNLEV>=7) then
                write(outu,'(A10,x,I4,x,A10,x,G12.5,4(x,I5),4(x,G11.4))') & 
                'MRMD_SURF>', &
                isurf,'-impr harm',-denergy,ii,jj,kk,ll,FCON,MULT,&
                PEQ*raddeg,geompar*raddeg
                if (PRNLEV>=10.and.rx_calcgrad) then
                    write(outu,'(4(A10,x))') 'atom_i','-dV_dxi','-dV_dyi','-dV_dzi'
                    write(outu,'(I10,x,3(F10.4,x))') ii,-dfx(1),-dfy(1),-dfz(1)
                    write(outu,'(I10,x,3(F10.4,x))') jj,-dfx(2),-dfy(2),-dfz(2)
                    write(outu,'(I10,x,3(F10.4,x))') kk,-dfx(3),-dfy(3),-dfz(3)
                    write(outu,'(I10,x,3(F10.4,x))') ll,-dfx(4),-dfy(4),-dfz(4)
                endif  
            endif
            
            energies(-9)=energies(-9)+denergy
        enddo

        !-------------------------------------------
        ! inclusion list of MRMD improper dihedrals
        !-------------------------------------------
        do i=1,rx_inc_nimpr(isurf)
            ! MRMD index of improper dihedral potential
            iimpr=rx_inc_impr(isurf,i)

            ! its atoms
            II = rx_impr_atoms(iimpr,1)
            JJ = rx_impr_atoms(iimpr,2)
            KK = rx_impr_atoms(iimpr,3)
            LL = rx_impr_atoms(iimpr,4)

            ! its parameters
            FCON=rx_impr_k   (isurf,iimpr)
            PEQ =rx_impr_phi0(isurf,iimpr)
            MULT=rx_impr_mult(         iimpr)
            PEQS=rx_impr_sin0(isurf,iimpr)
            PEQC=rx_impr_cos0(isurf,iimpr)

            ! calculate energy and gradient
            if (rx_calcgrad) then
               call MRMD_DIHE(natomx,X,Y,Z,II,JJ,KK,LL,FCON,MULT,PEQ,PEQC,PEQS,&
                           geompar,denergy,dfx(1:4),dfy(1:4),dfz(1:4))
                dEdx((/ii,jj,kk,ll/))=dEdx((/ii,jj,kk,ll/))+dfx(1:4)
                dEdy((/ii,jj,kk,ll/))=dEdy((/ii,jj,kk,ll/))+dfy(1:4)
                dEdz((/ii,jj,kk,ll/))=dEdz((/ii,jj,kk,ll/))+dfz(1:4)
            else
                call MRMD_DIHE(natomx,X,Y,Z,II,JJ,KK,LL,FCON,MULT,PEQ,PEQC,PEQS,&
                           geompar,denergy)
            endif

            if (rx_print_energy.and.PRNLEV>=7) then
                write(outu,'(A10,x,I4,x,A10,x,G12.5,4(x,I5),4(x,G11.4))') & 
                'MRMD_SURF>', &
                isurf,'+impr harm',denergy,ii,jj,kk,ll,FCON,MULT,&
                PEQ*raddeg,geompar*raddeg
                if (PRNLEV>=10.and.rx_calcgrad) then
                    write(outu,'(4(A10,x))') 'atom_i','+dV_dxi','+dV_dyi','+dV_dzi'
                    write(outu,'(I10,x,3(F10.4,x))') ii,dfx(1),dfy(1),dfz(1)
                    write(outu,'(I10,x,3(F10.4,x))') jj,dfx(2),dfy(2),dfz(2)
                    write(outu,'(I10,x,3(F10.4,x))') kk,dfx(3),dfy(3),dfz(3)
                    write(outu,'(I10,x,3(F10.4,x))') ll,dfx(4),dfy(4),dfz(4)
                endif  
            endif
            energies(9)=energies(9)+denergy
        enddo
    endif
!======================================================
! Energy calculation has finished
!======================================================

    !------------------------------------------------------
    ! energy=(CHARMM energy+)inclusions-exclusions 
    !------------------------------------------------------
    ener_exc= sum(energies(-rx_ninte:-1))
    ener_inc= sum(energies(  1:rx_ninte))
    energies(0)=-ener_exc+ener_inc+rx_levelshift(isurf)

    !------------------------------------------------------
    ! print energy corrections to CHARMM energy 
    !------------------------------------------------------
    if (rx_print_energy.and.PRNLEV>=6) then
        write(outu,'(''MRMD_SURF>'',80(''-''))')  
        write(outu,'(A,x,A,x,I2)') &
        'MRMD_SURF>','Energy corrections to CHARMM energy on surface:',isurf
        !  0: total - energies(1:isurf,-8:8)
        ! -1: harm removed from CHARMM,   +1: harm added on MRMD
        ! -2: mors removed from CHARMM,   +2: mors added on MRMD
        ! -3: angl removed from CHARMM,   +3: angl added on MRMD
        ! -4: dihe removed from CHARMM,   +4: dihe added on MRMD
        ! -5: impr removed from CHARMM,   +5: impr added on MRMD
        ! -6: elec removed from CHARMM,   +6: elec added on MRMD
        ! -7:  vdw removed from CHARMM,   +7:  vdw added on MRMD
        ! -8: gvdw removed from CHARMM,   +8: gvdw added on MRMD
        ! -9: urey removed from CHARMM,   +9: urey added on MRMD
        !-10: rkhs removed from CHARMM,  +10: rkhs added on MRMD
        write(outu,'(A,2(x,A17)   )') 'MRMD_SURF>            ','CHARMM-exclusion ',&
                                                               'MRMD-inclusion   '
        write(outu,'(A,2(x,G17.10))') 'MRMD_SURF> nbon elec :',-energies(-1) ,energies(1)
        write(outu,'(A,2(x,G17.10))') 'MRMD_SURF> nbon  vdw :',-energies(-2) ,energies(2)
        write(outu,'(A,2(x,G17.10))') 'MRMD_SURF> nbon gvdw :',-energies(-3) ,energies(3)
        write(outu,'(A,2(x,G17.10))') 'MRMD_SURF> bond harm :',-energies(-4) ,energies(4)
        write(outu,'(A,2(x,G17.10))') 'MRMD_SURF> bond mors :',-energies(-5) ,energies(5)
        write(outu,'(A,2(x,G17.10))') 'MRMD_SURF> angl harm :',-energies(-6) ,energies(6)
        write(outu,'(A,2(x,G17.10))') 'MRMD_SURF> angl urey :',-energies(-7) ,energies(7)
        write(outu,'(A,2(x,G17.10))') 'MRMD_SURF> dihe four :',-energies(-8) ,energies(8)
        write(outu,'(A,2(x,G17.10))') 'MRMD_SURF> impr harm :',-energies(-9) ,energies(9)
        write(outu,'(A,2(x,G17.10))') 'MRMD_SURF> bond rkhs :',-energies(-10),energies(10)
        write(outu,'(''MRMD_SURF>'',80(''-''))')  
    
        write(outu,'(A,2(x,G17.10))') 'MRMD_SURF>      sums :',-ener_exc,ener_inc
        write(outu,'(A,x,G17.10)')    'MRMD_SURF>   inc+exc :',-ener_exc+ener_inc
        write(outu,'(A,x,G17.10)')    'MRMD_SURF>     shift :',rx_levelshift(isurf)
        write(outu,'(A,x,G17.10)')    'MRMD_SURF>     total :',energies(0)
        if (rx_print_energy.and.PRNLEV>=9.and.rx_calcgrad) then
            write(OUTU,'(80(''-''))')
            write(outu,*) 'Correction to CHARMM gradient on PES:',isurf
            write(outu,'(4(A10,x))') 'atom','dV_dxi','dV_dyi','dV_dzi'        
            do i=1,natomx
                write(outu,'(I10,x,3(F10.4,x))') i,dEdx(i),dEdy(i),dEdz(i)
            enddo
        endif
    endif    
end subroutine MRMD_SURF

!===========================================================================
! Gauss*polynomial function and its derivative [GAPO]
!===========================================================================
  subroutine MRMD_GAPO(x,order,coeff,f,dfdx)
  implicit none
  !--------------------------------------------------------------------
  ! input

  integer       ,intent(in)  :: &
    order   ! polynomial order
  real(chm_real),intent(in)  :: &
    x,               & ! energy difference of the two surfaces             
    coeff(-2:order)    ! parameters of the GAPO function
                       ! -2      : center of Gaussian
                       ! -1      : decay coefficient
                       !  0:order: coefficients of the polynomial
  !---------------------------------------------------------------------------
  ! output
  real(chm_real),intent(out) :: f  ! value of GAPO function
  real(chm_real),intent(out),optional :: dfdx   ! derivative with respect to x
  !---------------------------------------------------------------------------
  ! local
  real(chm_real) y,z,poly,gaus,dpoly_dx,dgaus_dx
  integer i 

      ! calculate Gaussian and its derivative
      ! y=(x-b0)
      ! dy/dx=1
      y  = x-coeff(-2)
      ! z=y/b1
      z=y/coeff(-1) 
      ! exp(-z^2/2d0))
      gaus = exp(-0.5_chm_real*z*z)  

      ! calculate polynomial and its derivative
      ! f(y) =(((...+  a3)*y+  a2)*y+a1)*y+a0    
      poly     =       coeff(order)
      do i=order-1,0,-1
          poly    =  poly   *y +   coeff(i)  
      enddo
      
      ! function and its derivative
      f   = gaus*poly
       
      ! return here if derivative is not required
      if (.not.present(dfdx)) return

      ! calculate polynomial derivative
      ! f'(y)= ((...+3*a3)*y+2*a2)*y+a1  
      dpoly_dx = order*coeff(order)
      do i=order-1,1,-1
         dpoly_dx = dpoly_dx*y + i*coeff(i)
      enddo
      
      ! gaussian derivative
      ! dgaus/dz * dz/dx  = -z*gaus/b1
      dgaus_dx = -z*gaus/coeff(-1) 
        
      ! total derivative
      dfdx = dgaus_dx*poly+gaus*dpoly_dx 
  end subroutine MRMD_GAPO

!===========================================================================
! Electrostatic energy and gradient
!  - scaling wih 1/relative permittivity
!  - scaling with E14FAC for interactions between atoms in 1-4 position 
!  - shifting potential
!===========================================================================
  subroutine MRMD_ELEC(natomx,X,Y,Z,i,j,chgi,chgj,dist,r,energy,&
                       dEdx,dEdy,dEdz)
    use dimens_fcm
    use number
    use stream
    use psf
    use param
    use inbnd
    use consta

    implicit none

    !--------------------------------------------------------------------
    ! input
    integer,intent(in):: &
        natomx,i,j    ! total numbers of atoms, indices of the 2 atoms    
    real(chm_real),intent(in):: X(natomx),Y(natomx),Z(natomx) ! coordinates of all atoms
    real(chm_real),intent(in):: chgi,chgj ! charges of interacting atoms
    integer,intent(in):: dist         !  their relative positive in the bond network (1-n)
    !--------------------------------------------------------------------    
    ! output
    real(chm_real),intent(out):: r,energy ! distance, potential energy
    real(chm_real),optional,intent(out):: dEdx(2),dEdy(2),dEdz(2)  ! potential derivatives
    !--------------------------------------------------------------------

    ! local
    real(chm_real) rx,ry,rz,R2,R4,iR,iR2,df,dxi,dyi,dzi
    real(chm_real) CTOFNB1,CTOFNB2,CTOFNB4
    real(chm_real) keffqq ! Coulomb/rel_permittivity*charge1*charge2*{E14FAC(if 1-4position)}
    ! global:CTONNB, CTOFNB

    ! from module consta:
    ! CCELEC=332.0716D0 (values in AMBER,NAMD, etc compatibility modes are different)
    !IF AMBER
    !  real(chm_real), parameter :: CCELEC=332.0522173D0
    !ELIF DISCOVER
    !  real(chm_real),  parameter :: CCELEC=332.054D0
    !ELIF NAMD
    !  real(chm_real),  parameter :: CCELEC=332.0636D0
    !ELSE
    !  real(chm_real), parameter :: CCELEC=332.0716D0
    !ENDIF

    ! coordinate differences
    rx=X(i)-X(j)
    ry=Y(i)-Y(j)
    rz=Z(i)-Z(j)
    
    ! R^2=distance^2
    R2=rx*rx+ry*ry+rz*rz

    ! distance
    r=SQRT(R2)
    
    ! effective Coulomb constant*charge1*charge2
    keffqq=CCELEC/EPS*chgi*chgj
    
    ! further scaling if between atoms in 1-4 position
    if (abs(dist)==4) keffqq=E14FAC*keffqq

    ! early exiting if:
    ! - distance> beyond cutoff distance
    ! - if Coulomb constant is zero (should not be :)
    ! - if any of the charges are zero
    ! - if relative permitivity is +/-infinity
    ! - if charges are 1-4 position and E14FAC=0d0
    if (R>CTOFNB.or.keffqq==0._chm_real) then

        ! set energy to zero
        energy=0._chm_real
        
        ! return if only energy is requested
        if (.not.rx_calcgrad) return
        
        ! set gradient to zero
        dEdx   =0._chm_real
        dEdy   =0._chm_real
        dEdz   =0._chm_real
        return
    endif

    ! within cutoff distances
    ! R^4
    R4=R2*R2
    ! 1/R
    iR=1._chm_real/r
    ! 1/R^2
    iR2=iR*iR
    ! 1/Roff
    CTOFNB1=1._chm_real/CTOFNB
    ! 1/Roff^2
    CTOFNB2=CTOFNB1*CTOFNB1
    ! 1/Roff^4
    CTOFNB4=CTOFNB2*CTOFNB2

    ! Calculate electrostatic potential with shifting
    ! const*q1*q2/R*[1-2*(R/Roff)^2+(R/Roff)^4]

    ! shifting factor=
    ! = 0                     if r>roff 
    ! = (1-(r/roff)^2)^2      if r<=roff 
    ! after expansion: 1-2*(r/roff)^2+(r/roff)^4

    energy=keffqq*iR*(1._chm_real-2._chm_real*R2*CTOFNB2+R4*CTOFNB4)

    ! return if only energy is requested
    if (.not.rx_calcgrad) return

    ! Calculate gradient
    ! gradient=DV/DR~-const*q1*q2[1/R^2+2/Roff^2-3*R^2*/Roff^4]
    df=-keffqq*(iR2+2._chm_real*CTOFNB2-3._chm_real*R2*CTOFNB4)

    ! gradient/R
    df=df*iR

    ! gradient components
    dxi=df*rx
    dyi=df*ry
    dzi=df*rz

    ! gradient on atoms
    dEdx(1)=+dxi
    dEdy(1)=+dyi
    dEdz(1)=+dzi
    dEdx(2)=-dxi
    dEdy(2)=-dyi
    dEdz(2)=-dzi
  end subroutine MRMD_ELEC

!===========================================================================
!
!     VDW energy and gradient using potential switching
!
!===========================================================================
  subroutine MRMD_VDW(natomx,X,Y,Z,i,j,R1,E1,R2,E2,r,energy,dEdx,dEdy,dEdz)

    use number
    use stream
    use inbnd

    implicit none

    !--------------------------------------------------------------------------------------- 
    ! input
    integer,intent(in):: natomx,i,j ! total numbers of atoms, indices of the 2 atoms
    real(chm_real),intent(in):: X(natomx),Y(natomx),Z(natomx) ! coordinates of all atoms
    real(chm_real),intent(in):: R1,R2,E1,E2  ! rmin1/2,rmin2/2,eps1,eps2 !atomic LJ parameters    

    !--------------------------------------------------------------------------------------- 
    ! output
    real(chm_real),intent(out):: r,energy ! distance, potential energy
    real(chm_real),optional,intent(out) :: dEdx(2),dEdy(2),dEdz(2)  ! potential derivatives
    !--------------------------------------------------------------------------------------- 
    ! local variables
    real(chm_real) rx,ry,rz,S,RM2,SIG2,SIG6,SIG12,EN
    real(chm_real) EMIN,RUL12,RMIN2,DEN,df,dxi,dyi,dzi
    ! Variables for the switching function
    real(chm_real) RIJL,RIJU,C2ONNB,C2OFNB,RUL3,DFN,FUNCT

    ! global: CTOFNB, CTONNB

    ! coordinate differences
    rx=X(i)-X(j)
    ry=Y(i)-Y(j)
    rz=Z(i)-Z(j)

    ! distance^2
    S=rx*rx+ry*ry+rz*rz
    
    ! distance
    r=sqrt(S)

    ! distance> cutoff distance
    if (r>=CTOFNB.or.E1*E2==0._chm_real) then
        ! set energy to zero
        energy=0._chm_real
        
        ! return if only energy is requested
        if (.not.rx_calcgrad) return
        
        ! set gradient to zero
        dEdx   =0._chm_real
        dEdy   =0._chm_real
        dEdz   =0._chm_real
        return
    endif

    ! Ron^2
    C2ONNB=CTONNB*CTONNB

    ! Rmin^2
    RMIN2=(R1+R2)**2._chm_real

    ! Emin
    EMIN=SQRT(abs(E1*E2))

    ! 1/R^2 
    RM2=1._chm_real/S

    ! (Rmin/R)^2
    SIG2=RMIN2*RM2

    ! (Rmin/R)^6
    SIG6=SIG2*SIG2*SIG2

    ! (Rmin/R)^12
    SIG12=SIG6*SIG6

    ! energy
    ! V=Emin*((Rmin/R)^12-2*(Rmin/R)^6)
    energy=EMIN*(SIG12-SIG6-SIG6)

    ! save for later use for in switching function derivative
    EN=energy
        
    if (S>C2ONNB) then
        !---------------------------------------------------
        ! switching factor
        ! = 1                                                     if r<ron
        ! = (roff^2-r^2)^2*(roff^2+2r^2-3ron^2)/(roff^2-ron^2)^3  if ron<r<roff
        ! = 0                                                     if roff<r
        ! Roff^2>R^2>Ron^2 =>in switching region

        ! Ron^2 -R^2
        RIJL=C2ONNB-S
        ! Roff^2
        C2OFNB=CTOFNB*CTOFNB
        ! Roff^2-R^2
        RIJU=C2OFNB-S
        ! 1/(Roff^2-Ron^2)^3
        RUL3=1._chm_real/(C2OFNB-C2ONNB)**3._chm_real
        ! F= (Roff^2-R^2)^2*(Roff^2+2*R^2-3*Ron^2)/(Roff^2-Ron^2)^3
        FUNCT=RIJU*RIJU*(RIJU-3._chm_real*RIJL)*RUL3  
        ! switching correction
        energy=FUNCT*energy
    endif

    ! return if only energy is requested
    if (.not.rx_calcgrad) return

    ! dV/dR * 1/R without switching function derivation
    df=EMIN*RM2*12._chm_real*(SIG6-SIG12)

    ! dV/dR * 1/R
    ! Roff^2>R^2>Ron^2 =>in switching region
    ! note: dfn=dfswitch/dr * 1/R
    if (S>C2ONNB) then
        RUL12= 12._chm_real*RUL3      ! previously 12=LJ12, but this has nothing to do with LJ 12
        DFN  = RIJL*RIJU*RUL12  
        df   = DFN*EN+FUNCT*df
    endif

    ! Calculate gradient components x/R*dV/dR
    dxi=df*rx
    dyi=df*ry
    dzi=df*rz

    ! gradient on atoms
    dEdx(1)=+dxi
    dEdx(2)=-dxi

    dEdy(1)=+dyi
    dEdy(2)=-dyi

    dEdz(1)=+dzi
    dEdz(2)=-dzi        
  end subroutine MRMD_VDW
!===========================================================================
!     GENERAL-EXPONENT VDW energy and gradient using potential switching
!
!     Generalized a-b Lennard Jonnes potential by TNagy
!     V=eps*[(b/a)**(b/(b-a))] / (b/a-1) * [(sig/r)**b-(sig/r)**a]
!     ( for b=12,a=6=>V=4eps*[(sig/r)**b-(sig/r)**a] )
!     or 
!     V=eps*a/(b-a) * [(rmin/r)**b-b/a*(rmin/r)**a]
!     ( for b=12,a=6=>V=eps*[(rmin/r)**b-2*(rmin/r)**a] )
!     here rmin=sigma*(b/a)**(1/(b-a))
!
!     eps(ilon)=Emin= well depth   (positive value!)
!     sigma = separation where V=0
!     rmin  = separation at the bottom of the well
!     requirements: b>a>0 , epsilon>0
!
!     - analogous to 6-12 Lennard-Jones
!     - gives Lennard-Jones when b=12 and a=6
!
!===========================================================================
  subroutine MRMD_GVDW(NATOMX,X,Y,Z,I,J,Emin,Rmin,EXPATT,EXPREP,r,&
                    energy,dEdx,dEdy,dEdz)
    use number
    use stream
    use inbnd
    
    implicit none
    
    !--------------------------------------------------------------------------------------- 
    ! input
    INTEGER NATOMX,i,j ! total numbers of atoms, indices of the 2 atoms
    real(chm_real),intent(in):: X(NATOMX),Y(NATOMX),Z(NATOMX) ! coordinates of all atoms
    real(chm_real),intent(in):: Emin,Rmin,EXPATT,EXPREP  
    !--------------------------------------------------------------------------------------- 
    ! output
    real(chm_real),intent(out):: r,energy ! distance, potential energy
    real(chm_real),optional,intent(out):: dEdx(2),dEdy(2),dEdz(2) ! derivatives
    !--------------------------------------------------------------------------------------- 
    ! local variables
    real(chm_real) RX,RY,RZ,S,RM2,SIG,SIG6,SIG12,EN
    real(chm_real) rep_p_att,SIGREP,SIGATT
    real(chm_real) RUL12,RMIN2,DEN,DF,dxi,dyi,dzi
    
    ! Variables for the switching function
    real(chm_real) RIJL,RIJU,C2ONNB,C2OFNB,RUL3,DFN,FUNCT


        ! coordinate differences
        RX=X(I)-X(J)
        RY=Y(I)-Y(J)
        RZ=Z(I)-Z(J)
        
        ! distance^2
        S=RX*RX+RY*RY+RZ*RZ
        
        ! distance
        r=sqrt(S)

        ! distance> cutoff distance
        if (r>=CTOFNB) then
            ! set energy to zero
            energy=0._chm_real
            
            ! return if only energy is requested
            if (.not.rx_calcgrad) return
            
            ! set gradient to zero
            dEdx   =0._chm_real
            dEdy   =0._chm_real
            dEdz   =0._chm_real
            return
        endif

        ! Ron^2
        C2ONNB=CTONNB*CTONNB

        ! Rmin^2
        RMIN2=Rmin**2._chm_real

        ! 1/R^2
        RM2=1._chm_real/S

        ! Rmin/R
        SIG=sqrt(RMIN2*RM2)
        
        ! (Rmin/R)^rep
        SIGREP=SIG**EXPREP

        ! (Rmin/R)^att
        SIGATT=SIG**EXPATT
        
        ! EXPREP/EXPATT
        rep_p_att=EXPREP/EXPATT


        ! V=eps/(b/a-1) * [(rmin/r)**b-b/a*(rmin/r)**a]
        energy=abs(EMIN)/(rep_p_att-1._chm_real)*(SIGREP-rep_p_att*SIGATT)

        !save for later use for in switching function derivative
        EN=energy

        ! Roff^2>R^2>Ron^2 =>in switching region
        if (S>C2ONNB) then
            ! Ron^2 -R^2
            RIJL=C2ONNB-S
            ! Roff^2
            C2OFNB=CTOFNB*CTOFNB
            ! Roff^2-R^2
            RIJU=C2OFNB-S
            ! 1/(Roff^2-Ron^2)^3
            RUL3=1._chm_real/(C2OFNB-C2ONNB)**3._chm_real
            ! F= (Roff^2-R^2)^2*(Roff^2-R^2-3*(Ron^2-R^2))/(Roff^2-Ron^2)^3
            FUNCT=RIJU*RIJU*(RIJU-3._chm_real*RIJL)*RUL3
            ! switching correction
            energy=FUNCT*energy
        endif

        ! return if only energy is requested
        if (.not.rx_calcgrad) return

        ! dV/dr*1/r without switching function derivation
        ! dV/dr*1/r=eps/r^2*b/(b/a-1)*[(rmin/r)**a-(rmin/r)**b]
        DF=abs(EMIN)*RM2*EXPREP/(rep_p_att-1._chm_real)*(SIGATT-SIGREP)

        ! if Roff^2>R^2>Ron^2 =>in switching region
        ! adding correction due to switching to dV/dR*1/R 
        ! by also differentiating the switching function
        if (S>C2ONNB) then
            RUL12= 12._chm_real*RUL3    
            DFN  = RIJL*RIJU*RUL12 
            DF   = DFN*EN+FUNCT*DF
        endif

        ! Calculate gradient components
        dxi = DF*RX
        dyi = DF*RY
        dzi = DF*RZ

        ! Add gradient
        dEdx(1) = dxi 
        dEdx(2) =-dxi 

        dEdy(1) = dyi
        dEdy(2) =-dyi

        dEdz(1) = dzi
        dEdz(2) =-dzi        
 
  end subroutine MRMD_GVDW
!===========================================================================
! 
! HARMONIC BOND
!
!===========================================================================
  subroutine MRMD_HARM(natomx,X,Y,Z,I,J,fc_half,r_eq,r,energy,dEdx,dEdy,dEdz)

    ! Based on EBONDFS() from energy/enefscal.src
    ! Calculates bond energies and gradient
    use number
    use stream
    use dimens_fcm

    implicit none

    !--------------------------------------------------------------------------------------- 
    ! input
    integer,intent(in):: natomx,i,j ! total numbers of atoms, indices of the 2 atoms
    real(chm_real),intent(in):: X(natomx),Y(natomx),Z(natomx) ! coordinates of all atoms
    real(chm_real),intent(in):: fc_half,r_eq ! force constant, equilibrium distances
    !--------------------------------------------------------------------------------------- 
    !output
    real(chm_real),intent(out):: r,energy ! distance, potential energy
    real(chm_real),optional,intent(out):: dEdx(2),dEdy(2),dEdz(2) ! derivatives
    !--------------------------------------------------------------------------------------- 
    !local
    real(chm_real) rx,ry,rz,S,dr,df,dxi,dyi,dzi
    
        
        ! coordinate differences
        rx = X(i)-X(j)
        ry = Y(i)-Y(j)
        rz = Z(i)-Z(j)
        
        ! distance
        r = SQRT(rx*rx+ry*ry+rz*rz)

        ! deformation
        dr = r-r_eq
        
        !1/2*dV/dr =gradient/2
        df = fc_half*dr
        
        ! energy=fc/2*(r-req)^2
        energy = df*dr
        
        ! return if only energy is requested
        if (.not.rx_calcgrad) return

        ! gradient/distance 
        df  = 2._chm_real*df/r
        
        ! gradient components
        dxi = rx*df
        dyi = ry*df
        dzi = rz*df

        ! gradient on atoms 1-i, 2-j
        dEdx(1) = dxi 
        dEdx(2) =-dxi 

        dEdy(1) = dyi
        dEdy(2) =-dyi

        dEdz(1) = dzi
        dEdz(2) =-dzi

  end subroutine MRMD_HARM

!===========================================================================
!
! MORSE BOND
!
!===========================================================================
  subroutine MRMD_MORS(natomx,X,Y,Z,i,j,DE,BETA,r_eq,r,energy,dEdx,dEdy,dEdz)
    use dimens_fcm
    use number

    implicit none

    !--------------------------------------------------------------------------------------- 
    ! input
    integer,intent(in):: natomx, i, j ! total number of atoms, indices of the 2 atoms
    real(chm_real),intent(in):: X(natomx),Y(natomx),Z(natomx) ! coordinates of all atoms
    real(chm_real),intent(in):: DE, BETA, R_eq  ! Morse potential parameters
    !--------------------------------------------------------------------------------------- 
    ! ouput
    real(chm_real),intent(out):: r,energy ! distance, potential energy
    real(chm_real),optional,intent(out):: dEdx(2),dEdy(2),dEdz(2) ! derivatives
    !--------------------------------------------------------------------------------------- 
    ! local variables
    real(chm_real) rx,ry,rz,R2,dr,A,B,dxi,dyi,dzi,df

        ! coordinate differences
        rx = X(i)-X(j)
        ry = Y(i)-Y(j)
        rz = Z(i)-Z(j)
        
        ! distance
        r = SQRT(rx*rx+ry*ry+rz*rz)
        
        ! deformation
        dr = r-r_eq
        
        ! partial evaluations
        B = EXP(-BETA*dr)
        A = B*B
          
        ! energy  
        energy = DE*(1._chm_real-2._chm_real*B+A)
        
        ! return if only energy is requested
        if (.not.rx_calcgrad) return

        ! gradient (dV/dr)
        df=2._chm_real*BETA*DE*(B-A)
        
        ! gradient/distance
        df = df/r
        
        ! gradient components  2->1
        dxi = rx*df
        dyi = ry*df
        dzi = rz*df
        
        ! gradient on atoms
        dEdx(1)= dxi 
        dEdx(2)=-dxi

        dEdy(1)= dyi
        dEdy(2)=-dyi

        dEdz(1)= dzi
        dEdz(2)=-dzi
      
  end subroutine MRMD_MORS
  
!===========================================================================
!
! RKHS BOND
!
!===========================================================================
  subroutine MRMD_RKHS(natomx,X,Y,Z,i,j,ngrid,grid,coeff,asym,r,energy,dEdx,dEdy,dEdz)
    use dimens_fcm
    use number

    implicit none

    !--------------------------------------------------------------------------------------- 
    ! input
    integer,intent(in) :: natomx, ngrid, i, j ! total number of atoms, number of gridpoints, indices of the 2 atoms
    real(chm_real),intent(in) :: X(natomx),Y(natomx),Z(natomx) ! coordinates of all atoms
    real(chm_real),intent(in) :: grid(ngrid), coeff(ngrid), asym !gridpoints and rkhs coefficients + asymptote
    !--------------------------------------------------------------------------------------- 
    ! ouput
    real(chm_real),intent(out) :: r,energy ! distance, potential energy
    real(chm_real),optional,intent(out) :: dEdx(2),dEdy(2),dEdz(2) ! derivatives
    !--------------------------------------------------------------------------------------- 
    ! local variables
    real(chm_real) :: rx,ry,rz,dxi,dyi,dzi,df
    integer :: loop,n !loop variable, gridpoints

        ! coordinate differences
        rx = X(i)-X(j)
        ry = Y(i)-Y(j)
        rz = Z(i)-Z(j)
        
        ! distance
        r = SQRT(rx*rx+ry*ry+rz*rz)
        
        ! energy  
        energy = 0._chm_real
        do loop = 1,ngrid
            energy = energy + coeff(loop)*MRMD_RKHS_KERNEL(r,grid(loop))
        end do 
        energy = energy + asym
        
        ! return if only energy is requested
        if (.not.present(dEdx)) return
        
        ! gradient (dV/dr)
        df = 0._chm_real
        do loop = 1,ngrid
            df = df + coeff(loop)*MRMD_RKHS_DKERNEL(r,grid(loop))
        end do 
        
        ! gradient/distance
        df = df/r
        
        ! gradient components  2->1
        dxi = rx*df
        dyi = ry*df
        dzi = rz*df
        
        ! gradient on atoms
        dEdx(1)= dxi 
        dEdx(2)=-dxi

        dEdy(1)= dyi
        dEdy(2)=-dyi

        dEdz(1)= dzi
        dEdz(2)=-dzi
      
  end subroutine MRMD_RKHS
  
!===========================================================================
!     Calculates bond angles, angle energies and gradient
!     Equilibrium angle is passed in radians
!     Angle of current geometry is passed back in radians too 
!===========================================================================
  subroutine MRMD_ANGL(natomx,X,Y,Z,i,j,k,fc_half,TEQR,angle,energy,&
                       dEdx,dEdy,dEdz)

    use number
    use dimens_fcm
    use stream
    use consta
    use fast
    
    implicit none
    !--------------------------------------------------------------------------------------- 
    ! input
    integer,intent(in)::  natomx,i,j,k ! total number of atoms, indices of the 3 atoms
    real(chm_real),intent(in)::  X(natomx),Y(natomx),Z(natomx) ! coordinates of all atoms
    real(chm_real),intent(in)::  fc_half,TEQR ! force constant, equilibrium angle
    !--------------------------------------------------------------------------------------- 
    ! output
    real(chm_real),intent(out)::  angle,energy  ! angle, potential energy
    real(chm_real),optional,intent(out):: dEdx(3),dEdy(3),dEdz(3) ! derivatives
    !--------------------------------------------------------------------------------------- 
    ! local
    real(chm_real) dxi,dyi,dzi,DXJ,DYJ,DZJ,RI2,RJ2,RI,RJ
    real(chm_real) RIR,RJR,DXIR,DYIR,DZIR,DXJR,DYJR,DZJR,CST, &
                   AT,DA,df,E
    real(chm_real) ST2R,STR,DTXI,DTXJ,DTYI,DTYJ,DTZI,DTZJ
    real(chm_real) DFX,DFY,DFZ,DGX,DGY,DGZ,SMALLV


    SMALLV=RPRECI

    ! j->i bond vector, length^2,length,1/length
    dxi=X(i)-X(j)
    dyi=Y(i)-Y(j)
    dzi=Z(i)-Z(j)
    RI2=dxi*dxi+dyi*dyi+dzi*dzi
    RI=SQRT(RI2)
    RIR=1._chm_real/RI
    
    ! normalized vector i->j    
    DXIR=dxi*RIR
    DYIR=dyi*RIR
    DZIR=dzi*RIR
    
    ! j->k bond vector, length^2,length,1/length
    DXJ=X(k)-X(j)
    DYJ=Y(k)-Y(j)
    DZJ=Z(k)-Z(j)
    RJ2=DXJ*DXJ+DYJ*DYJ+DZJ*DZJ
    RJ=SQRT(RJ2)
    RJR=1._chm_real/RJ

    ! normalized vector j->k    
    DXJR=DXJ*RJR
    DYJR=DYJ*RJR
    DZJR=DZJ*RJR

    ! cosine of the bond angle
    CST=DXIR*DXJR+DYIR*DYJR+DZIR*DZJR

    ! arccos for large cos_angle
    if (ABS(CST)>=COSMAX) then
      ! setting CST to +/-1 
      if (ABS(CST)>1._chm_real) CST=SIGN(1._chm_real,CST)
      
      ! angle 
      AT=ACOS(CST)
      
      ! deformation in radian
      DA=AT-TEQR
      
      ! if the angle reaches linear and it is more than 0.1radian away from equilibrium
      if (ABS(DA)>0.1_chm_real.and.wrnlev>=6) then
        write(outu,'(A,/,A,3(x,I5))') &
        'MRMD_ANGL> WARNING: Angle is almost linear.', &
        ' Derivatives may be affected for atoms:',i,j,k
        
        write(outu,'(A10,x,6(5X,A,3F15.5,/))') &
        'MRMD_ANGL>', &
        'i ATOM:',X(i),Y(i),Z(i),   &
        'j ATOM:',X(j),Y(j),Z(j),   & 
        'k ATOM:',X(k),Y(k),Z(k),   &
        'DXIR  :',DXIR,DYIR,DZIR,   &
        'DXJR  :',DXJR,DYJR,DZJR,   &
        'CST   :',CST,AT*RADDEG,DA*RADDEG
      endif
    else
        ! angle
        AT=ACOS(CST)
        
        ! deformation in radian
        DA=AT-TEQR
    endif

    ! angle in radian [0,Pi)
    angle=at

    ! gradient/2
    df=fc_half*DA
    
    ! energy
    energy=df*DA

    ! return if only energy is requested
    if (.not.rx_calcgrad) return
    
    ! 2*gradient/2=gradient
    df=df+df

    ! ... Workaround for different cosine comparison in SLOW and
    ! ... FAST angle routines. Check which one are set by CHARMM
    ! ... If slow angle routine (EANGLE) is used
    if ((faster==-1).or.(lfast==-1)) then
      if (abs(cst)>=0.999) then
        st2r=1._chm_real/(1._chm_real-cst*cst+smallv)
        str=sqrt(st2r)
        if (teqr<pt001) then
          df=mintwo*fc_half*(1._chm_real+da*da/6._chm_real)
        else if (pi-teqr<pt001) then
          df=two*fc_half*(1._chm_real+da*da/6._chm_real)
        else
          df=-df*str
        endif
      else
        st2r=1._chm_real/(1._chm_real-cst*cst)
        str=sqrt(st2r)
        df=-df*str
      endif

    ! if fast angle routine (eanglfs) is used
    else
      if (abs(cst)>=0.999999) then
        st2r=1._chm_real/(1._chm_real-cst*cst+smallv)
        str=sqrt(st2r)
        if (teqr<pt001) then
          df=-2._chm_real*fc_half*(1._chm_real+da*da/6._chm_real)
        else if (pi-teqr<pt001) then
          df=2._chm_real*fc_half*(1._chm_real+da*da/6._chm_real)
        else
          df=-df*str
        endif
      else
        st2r=1._chm_real/(1._chm_real-cst*cst)
        str=sqrt(st2r)
        df=-df*str
      endif
    endif

    !  
    dtxi=rir*(dxjr-cst*dxir)
    dtxj=rjr*(dxir-cst*dxjr)

    dtyi=rir*(dyjr-cst*dyir)
    dtyj=rjr*(dyir-cst*dyjr)

    dtzi=rir*(dzjr-cst*dzir)
    dtzj=rjr*(dzir-cst*dzjr)

    ! x gradient components
    dfx=df*dtxi
    dgx=df*dtxj
    dEdx(1)=+dfx
    dEdx(2)=-dfx-dgx
    dEdx(3)=    +dgx

    ! y gradient components
    dfy=df*dtyi
    dgy=df*dtyj
    dEdy(1)=+dfy

    dEdy(2)=-dfy-dgy
    dEdy(3)=    +dgy

    ! z gradient components
    dfz=df*dtzi
    dgz=df*dtzj
    dEdz(1)=+dfz
    dEdz(2)=-dfz-dgz
    dEdz(3)=    +dgz


  end subroutine MRMD_ANGL

!===========================================================================
!
! This calculates dihedral angle,energy and gradient based on ephi in eintern.src
! input: peqr in radians: equilibrium
! output: phi in radians:  actual value
!
!===========================================================================
    subroutine MRMD_DIHE(natomx,X,Y,Z,i,j,k,l,FCONS,MULT,PEQR,PEQCOS,PEQSIN,&
                     phi,energy,dEdx,dEdy,dEdz)

    use dimens_fcm
    use number
    use econtmod
    use consta
    use stream

    implicit none

    !--------------------------------------------------------------------------------------- 
    ! input
    integer,intent(in)::  &
        natomx,i,j,k,l, &  ! number of atoms, indices of the 4 atoms
        MULT               ! multiplicity of dihedral potential
    real(chm_real),intent(in)::  X(natomx),Y(natomx),Z(natomx) ! coordinates of all atoms
    real(chm_real),intent(in)::  FCONS,PEQR,PEQCOS,PEQSIN   !force constant,phi0,cosphi0,sinphi0
    !--------------------------------------------------------------------------------------- 
    ! output
    real(chm_real),intent(out)::  phi,energy ! angle, potential energy
    real(chm_real),optional,intent(out) :: dEdx(4),dEdy(4),dEdz(4) ! dervatives
    !--------------------------------------------------------------------------------------- 
    ! local
    real(chm_real), PARAMETER :: RXMIN=0.005_chm_real,RXMIN2=0.000025_chm_real
    integer NWARN,IPER,NPER
    real(chm_real) CPBIC,E1,DF1,DDF1,FX,FY,FZ,GX,GY,GZ,HX,HY,HZ
    real(chm_real) AX,AY,AZ,BX,BY,BZ,RA2,RB2,RA2R,RB2R,RG2,RG, &
                   RGR,RGR2
    real(chm_real) RABR,CP,AP,SP,E,df,DDF,CA,SA,ARG,APR
    real(chm_real) GAA,GBB,FG,HG,FGA,HGB,FGRG2,HGRG2,DFRG3
    real(chm_real) DFX,DFY,DFZ,DHX,DHY,DHZ,DGX,DGY,DGZ
    real(chm_real) DTFX,DTFY,DTFZ,DTHX,DTHY,DTHZ,DTGX,DTGY,DTGZ

    ! I<-J =>F 
    FX=X(i)-X(j)
    FY=Y(i)-Y(j)
    FZ=Z(i)-Z(j)
    
    ! J<-K =>G
    GX=X(j)-X(k)
    GY=Y(j)-Y(k)
    GZ=Z(j)-Z(k)

    ! L<-K =>H
    HX=X(l)-X(k)
    HY=Y(l)-Y(k)
    HZ=Z(l)-Z(k)
    
    ! A=FxG cross product=> normal vector of IJK plane and its length^2
    AX=FY*GZ-FZ*GY
    AY=FZ*GX-FX*GZ
    AZ=FX*GY-FY*GX
    RA2=AX*AX+AY*AY+AZ*AZ
    
    ! B=HxG cross product=> normal vector of JKL plane and its length^2
    BX=HY*GZ-HZ*GY
    BY=HZ*GX-HX*GZ
    BZ=HX*GY-HY*GX
    RB2=BX*BX+BY*BY+BZ*BZ
    
    ! RG=|G|, RGR=1/|G|
    RG2=GX*GX+GY*GY+GZ*GZ
    RG=SQRT(RG2)

    ! if IJK liner or JKL linear or JK short=> generate warning
    ! => return with zero energy and gradient
    if (RA2<=RXMIN2.or.RB2<=RXMIN2.or.RG<=RXMIN) then
        if (WRNLEV>=6) then
            write(outu,'(A/A,4(x,I5))')                      &
            'MRMD_DIHE: WARNING: dihedral is almost linear.',  &
            ' derivatives may be affected for atoms:',i,j,k,l
        endif

        ! set angle to zero in degree
        PHI=0._chm_real

        ! set energy to zero
        energy=0._chm_real

        ! return if only energy is requested
        if (.not.rx_calcgrad) return

        ! set gradient to zero
        dEdx   =0._chm_real
        dEdy   =0._chm_real
        dEdz   =0._chm_real
        return
    endif

    ! 1/|G|
    RGR=1._chm_real/RG
    
    ! 1/|A|^2
    RA2R=1._chm_real/RA2

    ! 1/|B|^2
    RB2R=1._chm_real/RB2
    
    ! 1/|A|/|B|
    RABR=SQRT(RA2R*RB2R)
    
    ! CP=cos(phi)= AxB/|A||B|
    CP=(AX*BX+AY*BY+AZ*BZ)*RABR
    
    ! SP=sin(phi)
    !   G/|G|*sin(phi)=   BxA         /(|A||B|)     apply: *(G/|G|)
    ! =>    1*sin(phi)= G(BxA)        /(|A||B||G|)
    !                   A(GxB)        /(|A||B||G|)         !  
    !                   A(Gx(HxG))    /(|A||B||G|)         ! B=HxG     
    !                   A(((GG)H-GH)G)/(|A||B||G|)     ! A perpendicular to G
    !                   A|G||G|H      /(|A||B||G|)
    !                   |G|/(|A||B|)*A*H 
    SP=RG*RABR*(AX*HX+AY*HY+AZ*HZ)

    ! angle in degree (0,Pi]
    PHI=acos(CP)
    if (SP<0._chm_real) PHI=-PHI  ! (-Pi,Pi]  

    ! zero     or nonzero multiplicity
    ! improper or proper dihedral
    if (MULT/=0) then 
    ! MULT/=0 <=> proper dihedral 
        ! change multiplicity to positive (actually not allowed on input)
        IPER=abs(MULT)

        ! Calculation of cos(n*phi-phi0) and sin(n*phi-phi0).
        ! in an iterative manner
        E1 =1._chm_real
        DF1=0._chm_real
        do NPER=1,IPER
            ! cos(nper*phi)=cos((nper-1)*phi)*cos(phi)-sin((nper-1)*phi)*sin(phi)
            DDF1= E1*CP - DF1*SP

            ! sin(nper*phi)=cos((nper-1)*phi)*sin(phi)+sin((nper-1)*phi)*cos(phi)
            DF1 = E1*SP + DF1*CP

            ! save cos(nper*hi)
            E1=DDF1
        enddo
        
        ! cos(n*phi-phi0)=cos(n*phi)*cos(phi0)+sin(n*phi)*sin(phi0)
        E1 = E1*PEQCOS + DF1*PEQSIN
        
        ! sin(n*phi-phi0)=sin(n*phi)*cos(phi0)-cos(n*phi)*sin(phi0)
        DF1 =DF1*PEQCOS-DDF1*PEQSIN

        ! Fcons*(1+cos(n*phi-phi0))
        energy=FCONS*(1._chm_real+E1)
        
        ! first derivative 
        ! -FCONS*n*sin(n*phi-phi0)
        DF  = -FCONS*IPER*DF1
        
     else
     ! MULT=0 <=> improper dihedral =>quadratic potential and FCONS=half of the force constant

        ! cos(phi-phi0)=cos(phi)*cos(phi0)+sin(phi)*sin(phi0)
        CA = CP*PEQCOS + SP*PEQSIN
        
        ! sin(phi-phi0)=sin(phi)*cos(phi0)-cos(phi)*sin(phi0)
        SA = SP*PEQCOS - CP*PEQSIN

        ! accurate alpha=phi-phi0 calculation in [-Pi,Pi] range
        if (CA>PTONE ) then  ! PTONE=0.1_CHM_REAL
            ! if cos(alpha)>0.1 =>alpha=asin(sin(alpha))
            AP=ASIN(SA)
        else
            ! if cos(alpha)<=0.1 =>alpha=acos(max(-1,cos(alpha)))*sgn(sin(alpha))
            AP=SIGN(ACOS(MAX(CA,MINONE)),SA)
        endif

        ! 2*second derivative=force constant=k=2*k/2 (FCONS=half of the force constant)
        DDF=2._chm_real*FCONS

        ! first derivative = dV/dANGLE  = k*(phi-phi0)
        df=DDF*AP

        ! energy  = k/2*(phi-phi0)^2
        energy=0.5_chm_real*df*AP 
    endif
    
    ! return if only energy is requested
    if (.not.rx_calcgrad) return

    !------------------
    ! Derivatives
    !------------------
    
    ! DTFi=d(phi)/dFi 
        ! -|G|/|A|^2
        GAA=-RA2R*RG
    DTFX=GAA*AX
    DTFY=GAA*AY
    DTFZ=GAA*AZ
    
    ! DTGi=d(phi)/dGi
        ! F*G
        FG=FX*GX+FY*GY+FZ*GZ
        ! H*G
        HG=HX*GX+HY*GY+HZ*GZ
        ! F*G/|A|^2/|G|
        FGA=FG*RA2R*RGR
        ! H*G/|B|^2/|G|
        HGB=HG*RB2R*RGR
    DTGX=FGA*AX-HGB*BX
    DTGY=FGA*AY-HGB*BY
    DTGZ=FGA*AZ-HGB*BZ
    
    ! DTHi=d(phi)/dHi
        ! |G|/|B|^2
        GBB=RB2R*RG
    DTHX=GBB*BX
    DTHY=GBB*BY
    DTHZ=GBB*BZ

    ! dEdFi=dE/dphi*dphi/dFi
    DFX=df*DTFX
    DFY=df*DTFY
    DFZ=df*DTFZ
    
    ! dEdGi=dE/dphi*dphi/dGi
    DGX=df*DTGX
    DGY=df*DTGY
    DGZ=df*DTGZ

    ! dEdHi=dE/dphi*dphi/dHi
    DHX=df*DTHX
    DHY=df*DTHY
    DHZ=df*DTHZ

    ! Distribute over Ri.
    dEdx(1)=+DFX
    dEdx(2)=-DFX+DGX
    dEdx(3)=    -DGX-DHX
    dEdx(4)=        +DHX

    dEdy(1)=+DFY
    dEdy(2)=-DFY+DGY
    dEdy(3)=    -DGY-DHY
    dEdy(4)=        +DHY

    dEdz(1)=+DFZ
    dEdz(2)=-DFZ+DGZ
    dEdz(3)=    -DGZ-DHZ
    dEdz(4)=        +DHZ
  end subroutine MRMD_DIHE

!===========================================================================
!
! DUMP PDB FILE
!
!===========================================================================
  subroutine MRMD_PDB(crossgeom_unit,natomx,X,Y,Z,TIME,nsurf,factors)
    use parallel
    use dimens_fcm
    use number
    use stream
    use string
    use psf
    !use rtf
    use chutil,only:atomid

    implicit none  

    !--------------------------------------------------------------------------------------- 
    ! input
    integer,intent(in) ::   &
        crossgeom_unit,     &   ! unit for writing the geometries
        natomx,             &   ! number of atoms
        nsurf                   ! number of surfaces    
    real(chm_real),intent(in) :: &
        X(natomx),Y(natomx),Z(natomx),  &   ! coordinates of all atoms
        TIME,                           &   ! time
        factors(1:nsurf)                    ! switching weights
    !--------------------------------------------------------------------------------------- 
    !local
    character(len=4) HDR,SID,RID,REN,AC,ARID,ATYPEI
    integer IRES,IPT,i,l       

    ! write pdb only with the 0-th node
    if (mynod/=0) return 

    write(crossgeom_unit,'(A,F13.6,A)') 'REMARK CROSSING GEOMETRY AT TIME= ',TIME,'ps'
    do i=1,nsurf
        write(crossgeom_unit,'(A,I3,x,A,F10.5)') 'REMARK WEIGHT OF SURFACE',i,':',factors(i)
    enddo
    write(crossgeom_unit,'(A)') 'REMARK'

    do IRES=1,NRES
       do i=IBASE(IRES)+1,IBASE(IRES+1)
          call ATOMID(i,SID,RID,REN,AC)
          l=4
          call TRIME(RID,l)
          ARID='    '
          if (l==4.or.RID(l:l)>='A') then
             ARID(4-l+1:4)=RID(1:l)
          else
             ARID(4-l:3)=RID(1:l)
          endif
          ! shift atom names when they exceed 3 characters
          if (ATYPE(i)(4:4) == ' ') then
             ATYPEI=' '//ATYPE(i)(1:3)
          else
             ATYPEI=ATYPE(i)
          endif
          ! the SEGID is written to the last four characters of the line
          !brb..07-FEB-99 Change default occupancy from zero to one
          write(crossgeom_unit, &
               '(A,I5,1X,A4,1X,A4,2X,A4,3X,3F8.3,2F6.2,6X,A4)') &
               'ATOM  ',i,ATYPEI,REN,ARID,X(i),Y(i),Z(i),1.0,0.0 &
               ,SID
       enddo
    enddo
    ! write END statement for PDB file
    write(crossgeom_unit,'(A)') 'END'
  end subroutine MRMD_PDB

!===========================================================================
!
! Mathematical switching functions and their derivatives 
!
!===========================================================================
subroutine MRMD_SWITCH_FUNCT(x,switchf,f,dfdx)
#if KEY_PATHSCALE==1
    use consta, only: pi
#endif
    use stream
    implicit none
    ! input
    real(chm_real),intent(in)   ::x
    character(len=*),intent(in) ::switchf
    ! output
    real(chm_real),intent(out) :: f
    real(chm_real),intent(out),optional :: dfdx
    ! local
    real(chm_real),parameter :: one=1._chm_real
#if KEY_PATHSCALE==0
    real(chm_real),parameter :: pi=4._chm_real*ATAN(one)
#endif
    real(chm_real) ::xx,yy,sec,sechtan

       
        ! select switching function
        select case(switchf)
            case('JOHNSON7')
                ! Johnson
                if (x<=0._chm_real) then
                    f=0._chm_real
                    if (present(dfdx))  dfdx=0
                    return
                endif
                if (1._chm_real<=x) then
                    f=1._chm_real 
                    if (present(dfdx))  dfdx=0
                    return
                endif
                f=-x**4._chm_real*(((20._chm_real*x-70._chm_real)*x+84._chm_real)*x-35._chm_real)
                if (present(dfdx))  &
                dfdx=-140._chm_real*x**3._chm_real*(((x-3._chm_real)*x+3._chm_real)*x-1._chm_real)
            case('EXPDECAY')  ! it is rather a weighting functions than switching
                ! exponential exp(-(Vi-Vmin)/DV)   x=1-(Vi-Vmin)/DV
                ! set tiny weights to zero
                if (x<-33.5_chm_real) then   !x-1<log(1.e-15_chm_real)=-15*2.3=-34.5
                    f=0._chm_real
                    if (present(dfdx))  dfdx=0
                    return                
                endif
                f=exp(x-1._chm_real)
                if (present(dfdx))  dfdx=f
            case default
                ! self checking for further developments
                call wrndie(-3,'<mrmd.src> mrmd_switch_funct','Not possible switching function')
        end select
    end subroutine MRMD_SWITCH_FUNCT

!===========================================================================
!
! subroutine calculating surface switching factors and 
! their analytic derivatives with respect to surface energies
!
!===========================================================================
subroutine MRMD_SWITCH_WEIGHTS(nsurf,energies,rx_swch_dE,switchfunc,factors,dfdE)
implicit none
    ! input
    integer,intent(in)         :: nsurf
    real(chm_real),intent(in)  :: energies(nsurf)
    real(chm_real),intent(in)  :: rx_swch_dE
    character(len=*),intent(in):: switchfunc    

    ! output
    real(chm_real),intent(out) :: factors(nsurf)
    real(chm_real),intent(out),optional :: dfdE(nsurf,nsurf)

    ! local
    integer isurf1,isurf2,isurf3,minsurf
    real(chm_real) minenergy,x(nsurf)
    real(chm_real) f0(nsurf),sumfac,invsumfac,invsumfac2
    real(chm_real) df0dx,df0dE(nsurf,nsurf),dfdf0(nsurf,nsurf)

    ! minimal surface
    minsurf=1
    do isurf1=2,nsurf
        if (energies(isurf1)<energies(minsurf)) minsurf=isurf1
    enddo
    minenergy=energies(minsurf)

    ! Normalized difference from lowest=argument of the switching function
    ! it gives: 1 for the lowest surface    => (completely/partially) switched on
    ! it gives: <0 for Esurf>Emin+rx_swch_dE => switched of completely
    ! it gives: 0-1 for between             => partially switched on
    x=1_chm_real-(energies-minenergy)/rx_swch_dE        ! vector

    !dxi/dEj=(Kronecker(j,min)-Kronecker(j,i))/rx_swch_dE

    ! switching factors compared to lowest surface
                       !input          !input     !output
    if (present(dfdE)) then
        ! df0dx
        do isurf2=1,nsurf

            ! df0dx=df0(isurf2)/dx(isurf2)
            ! will be zero for all cases when f0(isurf2)==0
            ! raw weights
            call MRMD_SWITCH_FUNCT(x(isurf2),switchfunc,f0(isurf2),df0dx)    

            ! reset
            ! df0dE(isurf1,isurf2)
            ! df0(isurf2)/dE(isurf1)=df0(isurf2)/dx(isurf2)*dx(isurf2)/dE(isurf1)
            df0dE(:,isurf2)=0._chm_real
            
            ! if isurf2 is the lowest or higher than surfmin+rx_swch_dE
            ! that is completely switched on or completely switched off
            if (df0dx==0._chm_real) cycle  !(<==> f0(isurf2)=0._chm_real or ==1._chm_real)
            
            ! if between
            ! df0(isurf2)/dE(isurf1)=df0(isurf2)/dx(isurf2)*dx(isurf2)/dE(isurf1)
            do isurf1=1,nsurf
                ! if isurf1=isurf2 
                if (isurf1==isurf2)  df0dE(isurf1,isurf2)=df0dE(isurf1,isurf2)-df0dx/rx_swch_dE

                ! if isurf1=minsurf
                if (isurf1==minsurf) df0dE(isurf1,isurf2)=df0dE(isurf1,isurf2)+df0dx/rx_swch_dE
                ! all other cases are zero
            enddo
        enddo    
    else
        do isurf1=1,nsurf
            call MRMD_SWITCH_FUNCT(x(isurf1),switchfunc,f0(isurf1))    
        enddo
    endif
    
    ! sum of weights
    sumfac=sum(f0(1:nsurf))
    
    ! normalizing switching factors => now they add-up to one
    factors=f0/sumfac
    
    if (present(dfdE)) then
        ! dfdE(isurf1,isurf2)=df(isurf2)/dE(isurf1)=
        !
        !     =sum_isurf3(df(isurf2)/df0(isurf3)*df0(isurf3)/dE(isurf1))

        invsumfac=1._chm_real/sumfac
        invsumfac2=invsumfac*invsumfac
        dfdE =0._chm_real
        dfdf0=0._chm_real

        ! differentiating with respect to E(isurf1)
        ! df0dE(isurf1,isurf3)=df0(isurf3)/dE(isurf1))
        do isurf1=1,nsurf
            if (factors(isurf1)==0._chm_real) cycle

            ! differentiating f(isurf2)
            ! dfdf0(isurf3,isurf2)=df(isurf2)/df0(isurf3)
            do isurf2=1,nsurf
                if (factors(isurf2)==0._chm_real) cycle

                ! differentiating with respect to f0(isurf3)
                ! dfdf0(isurf3,isurf2)=df(isurf2)/df0(isurf3)
                do isurf3=1,nsurf
                    if (df0dE(isurf1,isurf3)==0._chm_real) cycle 
                    
                    !  dfdf0(isurf3,isurf2)=df(isurf2)/df0(isurf3)
                    if (isurf2==isurf3) then
                        ! with respect to same f0
                        dfdf0(isurf3,isurf2)=invsumfac-f0(isurf2)*invsumfac2
                    else
                        ! with respect to other f0
                        dfdf0(isurf3,isurf2)=         -f0(isurf2)*invsumfac2
                    endif
                    
                    ! summing for isurf3:
                    ! dfdE(isurf1,isurf2)=
                    !...+df(isurf2)/df0(isurf3)*df0(isurf3)/dE(isurf1)+...
                    dfdE(isurf1,isurf2)=dfdE(isurf1,isurf2)+ &
                                       dfdf0(isurf3,isurf2)*df0dE(isurf1,isurf3)
                enddo
            enddo
        enddo
    endif

end subroutine MRMD_SWITCH_WEIGHTS

!===========================================================================
!
! Setting string uppercase
!
!===========================================================================
function MRMD_UPPERCASE(string)
    implicit none
    ! input
    character(len=*), intent(in)   :: string
    ! output
    character (len=len(string)) :: MRMD_UPPERCASE
    ! local
    integer i,char_code,code_shift
        
        code_shift=ichar('A')-ichar('a')
        do i=1,len(string)
            char_code=ichar(string(i:i))
            if (ichar('a')<=char_code.and.char_code<=ichar('z')) char_code=char_code+code_shift
            MRMD_UPPERCASE(i:i)=char(char_code)
        enddo
end function MRMD_UPPERCASE

!===========================================================================
!
! kernel function for RKHS (distance like)
!
!===========================================================================
real(chm_real) function MRMD_RKHS_KERNEL(x,r)
    implicit none
    !input
    real(chm_real), intent(in)  :: x,r 

    if(r >= x) then !if r is larger
        MRMD_RKHS_KERNEL = (1._chm_real/(14._chm_real*r**7))* &
                           (1._chm_real-7*x/(9._chm_real*r))
    else !if x is larger
        MRMD_RKHS_KERNEL = (1._chm_real/(14._chm_real*x**7))* &
                           (1._chm_real-7*r/(9._chm_real*x))
    endif
    return
end function MRMD_RKHS_KERNEL

!===========================================================================
!
! kernel function derivative for RKHS (distance like)
!
!===========================================================================
real(chm_real) function MRMD_RKHS_DKERNEL(x,r)
    implicit none
    !input
    real(chm_real), intent(in)  :: x,r 

    if(r >= x) then !if r is larger
        MRMD_RKHS_DKERNEL = -1._chm_real/(18._chm_real*r**8)
    else !if x is larger
        MRMD_RKHS_DKERNEL = -(1._chm_real/18._chm_real)* &
                             (9._chm_real*x-8._chm_real*r)/(x**9)
    endif
    return
end function MRMD_RKHS_DKERNEL

!===========================================================================
!
! Subroutine to calculate the value of the asymptote using biscetion
!
!===========================================================================
subroutine MRMD_RKHS_CALC_ASYM(ngrid,grid,coef,energy,asym)
    implicit none
    !--------------------------------------------------------------------------------------- 
    ! input
    integer,intent(in) :: ngrid ! number of gridpoints
    real(chm_real),intent(in) :: grid(ngrid), coef(ngrid), energy(ngrid)
    !--------------------------------------------------------------------------------------- 
    ! ouput
    real(chm_real),intent(out) :: asym
    !--------------------------------------------------------------------------------------- 
    ! local variables
    integer, parameter :: maxiter = 100 ! this should be sufficient
    real(chm_real), parameter :: eps = 1e-12_chm_real ! closer than this is probably meaningless
    real(chm_real), dimension(2), parameter :: zerozero =  (/0._chm_real,0._chm_real/)! just for easy input
    integer :: i,bestind
    real(chm_real) :: a,b,c,fa,fb,fc,dum,dEdx(2),dEdy(2),dEdz(2)
    
    !search for the lowest energy grid point
    bestind = 1
    fa = energy(1)
    do i = 2,ngrid
        if(energy(i) <= fa) then
            bestind = i
            fa = energy(i)
        end if
    end do
    if(bestind == ngrid) then !this means the potential is dissociative
        asym = 0._chm_real
        return
    end if
    if(bestind == 1) & !this is literally impossible for a meaningful grid
        call wrndie(-3,'<mrmd.src> mrmd_rkhs_calc_asym', &
        'Erroneous grid data for RKHS, please check the input ')
        
    !initialize guesses
    a  = grid(bestind - 1)
    call MRMD_RKHS(2,(/0._chm_real,a/),zerozero,zerozero,1,2,ngrid,&
                    grid,coef,0._chm_real,a,dum,dEdx,dEdy,dEdz)
    fa = dEdx(2)
    b  = grid(bestind + 1)
    call MRMD_RKHS(2,(/0._chm_real,b/),zerozero,zerozero,1,2,ngrid,&
                    grid,coef,0._chm_real,b,dum,dEdx,dEdy,dEdz)
    fb = dEdx(2)
    
    do i = 1,maxiter ! 100 iterations should always be sufficients
        ! get new value of c and fc
        c = (a+b)/2._chm_real
        call MRMD_RKHS(2,(/0._chm_real,c/),zerozero,zerozero,1,2,ngrid,&
                       grid,coef,0._chm_real,c,dum,dEdx,dEdy,dEdz)
        fc = dEdx(2)
        !convergence check
        if((b-a)/2 < eps) then
            call MRMD_RKHS(2,(/0._chm_real,c/),zerozero,zerozero,1,2,ngrid,&
                           grid,coef,0._chm_real,c,fc,dEdx,dEdy,dEdz)
            asym = -fc
            return
        end if
        
        !if fa and fc have the same sign
        if(((fc < 0._chm_real).and.(fa < 0._chm_real)).or.&
           ((fc > 0._chm_real).and.(fa > 0._chm_real))) then
           a  = c
           fa = fc
        !if fb and fc have the same sign
        else
           b  = c
           fc = fc
        end if
        
        !check if a is still smaller than b, if not reverse their order
        if(b < a) then
            c  = a
            fc = fa
            a  = b
            fa = fb
            b  = c
            fb = fc
        end if
    
    end do
    c = (a+b)/2._chm_real
    call MRMD_RKHS(2,(/0._chm_real,c/),zerozero,zerozero,1,2,ngrid,&
                    grid,coef,0._chm_real,c,fc,dEdx,dEdy,dEdz)
    asym = -fc
    return
end subroutine MRMD_RKHS_CALC_ASYM

!===========================================================================
!
! Do a Cholesky decomposition for calculating the kernel coefficients
!
!===========================================================================
subroutine MRMD_CHOL_DECOMP(a,d)
    implicit none
    real(chm_real), dimension(:,:), intent(inout) :: a   ! matrix to decompose
    real(chm_real), dimension(:)  , intent(out)   :: d   ! vector to store diagonal entries
    real(chm_real)                                :: s   ! summation variable
    integer                                       :: i,n ! loop variable, matrix size
    n = size(d, DIM=1)
    if((size(a,DIM=1) /= size(a,DIM=2)).OR.(size(a,DIM=1) /= n)) &
        call wrndie(-3,'<mrmd.src> mrmd_chol_decomp',&
                       'Matrix is either non-square or not the same size as diagonal vector!')
    do i = 1,n
        s = a(i,i) - DOT_PRODUCT(a(i,1:i-1),a(i,1:i-1))
        if (s <= 0.d0) call wrndie(-3,'<mrmd.src> mrmd_chol_decomp',&
                                      'Matrix is not positive definite!')
        d(i) = SQRT(s)
        a(i+1:n,i)=(a(i,i+1:n) - MATMUL(a(i+1:n,1:i-1),a(i,1:i-1)))/d(i)
    end do
    return
end subroutine MRMD_CHOL_DECOMP

!===========================================================================
!
! Solves a*x=b using the decomposed matrix from mrmd_chol_decomp
!
!===========================================================================
SUBROUTINE MRMD_CHOL_SOLVE(a,d,b,x)
    IMPLICIT NONE
    real(chm_real), dimension(:,:), intent(in)    :: a   ! decomposed matrix
    real(chm_real), dimension(:)  , intent(in)    :: d,b ! diagonal vector, solution vector
    real(chm_real), dimension(:)  , intent(inout) :: x   ! result vector
    integer                                       :: i,n ! loop variable, matrix size
    n = size(d, DIM=1)
    if((size(a,DIM=1) /= size(a,DIM=2)).OR.(size(a,DIM=1) /= n).OR. &
       (size(b,DIM=1) /= N).OR.(size(x,DIM=1) /= N)) &
        call wrndie(-3,'<mrmd.src> mrmd_chol_solve',&
                       'Passed arguments have the wrong dimensions!')
    do i=1,n,1 !forward solution
        x(i) = (b(i) - dot_product(a(i,1:i-1),x(1:i-1)))/d(i)
    end do
    do i=n,1,-1 !backsubstitution
        x(i) = (x(i) - dot_product(a(i+1:n,i),x(i+1:n)))/d(i)
    end do
    return
END SUBROUTINE MRMD_CHOL_SOLVE

!===========================================================================
!
! allocates, reallocates, deallocates(newsize=0) real 1D array
!
!===========================================================================
  subroutine MRMD_ALLOC_REAL1(filen,procn,arran,array,l1,h1)
    implicit none
    character(len=*),intent(in) :: filen,procn,arran
    real(chm_real),allocatable,dimension(:),intent(inout) :: array
    integer,intent(in) :: l1,h1
    integer :: m1    
    if (allocated(array)) then
       m1 = size(array)
       call chmdealloc(filen,procn,arran,m1,crl=array)
    endif
    if (h1>=l1) &
    call chmalloc(filen,procn,arran,h1-l1+1,crl=array, &
    lbou=l1)
  end subroutine MRMD_ALLOC_REAL1
!===========================================================================
!
! allocates, reallocates, deallocates(newsize=0) real 2D array
!
!===========================================================================
  subroutine MRMD_ALLOC_REAL2(filen,procn,arran,array,l1,h1,l2,h2)
    use memory
    implicit none
    character(len=*),intent(in) :: filen,procn,arran
    real(chm_real),allocatable,dimension(:,:),intent(inout) :: array
    integer,intent(in) :: l1,h1,l2,h2
    integer :: m1,m2
    if (allocated(array)) then
       m1 = size(array,1)
       m2 = size(array,2)
       call chmdealloc(filen,procn,arran,m1,m2,crl=array)
    endif
    if (h1>=l1.and.h2>=l2) &
    call chmalloc(filen,procn,arran,h1-l1+1,h2-l2+1,crl=array, &
    lbou=l1,lbou2=l2)
  end subroutine MRMD_ALLOC_REAL2
!===========================================================================
!
! allocates, reallocates, deallocates(newsize=0) real 3D array
!
!===========================================================================
  subroutine MRMD_ALLOC_REAL3(filen,procn,arran,array,l1,h1,l2,h2,l3,h3)
    use memory
    implicit none
    character(len=*),intent(in) :: filen,procn,arran
    real(chm_real),allocatable,dimension(:,:,:),intent(inout) :: array
    integer,intent(in) :: l1,h1,l2,h2,l3,h3
    integer :: m1,m2,m3
    if (allocated(array)) then
       m1 = size(array,1)
       m2 = size(array,2)
       m3 = size(array,3)
       call chmdealloc(filen,procn,arran,m1,m2,m3,crl=array)
    endif
    if (h1>=l1.and.h2>=l2.and.h3>=l3) &
    call chmalloc(filen,procn,arran,h1-l1+1,h2-l2+1,h3-l3+1,crl=array, &
    lbou=l1,lbou2=l2,lbou3=l3)
  end subroutine MRMD_ALLOC_REAL3 
!===========================================================================
!
! allocates, reallocates, deallocates(newsize=0) integer 1D array
!
!===========================================================================
  subroutine MRMD_ALLOC_INTG1(filen,procn,arran,array,l1,h1)
    implicit none
    character(len=*),intent(in) :: filen,procn,arran
    integer,allocatable,dimension(:),intent(inout) :: array
    integer,intent(in) :: l1,h1
    integer :: m1    
    if (allocated(array)) then
       m1 = size(array)
       call chmdealloc(filen,procn,arran,m1,intg=array)
    endif
    if (h1>=l1) &
    call chmalloc(filen,procn,arran,h1-l1+1,intg=array, &
    lbou=l1)
  end subroutine MRMD_ALLOC_INTG1
!===========================================================================
!
! allocates, reallocates, deallocates(newsize=0) integer 2D array
!
!===========================================================================
  subroutine MRMD_ALLOC_INTG2(filen,procn,arran,array,l1,h1,l2,h2)
    use memory
    implicit none
    character(len=*),intent(in) :: filen,procn,arran
    integer,allocatable,dimension(:,:),intent(inout) :: array
    integer,intent(in) :: l1,h1,l2,h2
    integer :: m1,m2
    if (allocated(array)) then
       m1 = size(array,1)
       m2 = size(array,2)
       call chmdealloc(filen,procn,arran,m1,m2,intg=array)
    endif
    if (h1>=l1.and.h2>=l2) &
    call chmalloc(filen,procn,arran,h1-l1+1,h2-l2+1,intg=array, &
    lbou=l1,lbou2=l2)
  end subroutine MRMD_ALLOC_INTG2
!===========================================================================
!
! allocates, reallocates, deallocates(newsize=0) integer 3D array
!
!===========================================================================
  subroutine MRMD_ALLOC_INTG3(filen,procn,arran,array,l1,h1,l2,h2,l3,h3)
    implicit none
    character(len=*),intent(in) :: filen,procn,arran
    integer,allocatable,dimension(:,:,:),intent(inout) :: array
    integer,intent(in) :: l1,h1,l2,h2,l3,h3
    integer :: m1,m2,m3 
    if (allocated(array)) then
       m1 = size(array,1)
       m2 = size(array,2)
       m3 = size(array,3)
       call chmdealloc(filen,procn,arran,m1,m2,m3,intg=array)
    endif
    if (h1>=l1.and.h2>=l2.and.h3>=l3) &
    call chmalloc(filen,procn,arran,h1-l1+1,h2-l2+1,h3-l3+1,intg=array,&
    lbou=l1,lbou2=l2,lbou3=l3)
  end subroutine MRMD_ALLOC_INTG3
!===========================================================================
!
! allocates, reallocates, deallocates(newsize=0) logical 1D array
!
!===========================================================================
  subroutine MRMD_ALLOC_LOGI1(filen,procn,arran,array,l1,h1)
    implicit none
    character(len=*),intent(in) :: filen,procn,arran

    logical,allocatable,dimension(:),intent(inout) :: array
    integer,intent(in) :: l1,h1
    integer :: m1
    
    if (allocated(array)) then
       m1 = size(array)
       call chmdealloc(filen,procn,arran,m1,log=array)
    endif
    if (h1>=l1) &
    call chmalloc(filen,procn,arran,h1-l1+1,log=array,lbou=l1)
  end subroutine MRMD_ALLOC_LOGI1
!===========================================================================
!
! allocates, reallocates, deallocates(newsize=0) logical 2D array
!
!===========================================================================
  subroutine MRMD_ALLOC_LOGI2(filen,procn,arran,array,l1,h1,l2,h2)
    use memory
    implicit none
    character(len=*),intent(in) :: filen,procn,arran

    logical,allocatable,dimension(:,:),intent(inout) :: array
    integer,intent(in) :: l1,h1,l2,h2
    integer :: m1,m2
    
    if (allocated(array)) then
       m1 = size(array,1)
       m2 = size(array,2)
       call chmdealloc(filen,procn,arran,m1,m2,log=array)
    endif
    if (h1>=l1.and.h2>=l2) &
    call chmalloc(filen,procn,arran,h1-l1+1,h2-l2+1,log=array,lbou=l1,lbou2=l2)
  end subroutine MRMD_ALLOC_LOGI2
!===========================================================================
!
! allocates, reallocates, deallocates(newsize=0) logical char*16 1D array
!
!===========================================================================
  subroutine MRMD_ALLOC_CH16_1(filen,procn,arran,array,l1,h1)
    use memory
    implicit none
    character(len=*),intent(in) :: filen,procn,arran
    character(len=16),allocatable,dimension(:),intent(inout) :: array
    integer,intent(in) :: l1,h1
    integer :: m1
    
    if (allocated(array)) then
       m1 = size(array)
       call chmdealloc(filen,procn,arran,m1,ch16=array)
    endif
    if (h1>=l1) &
    call chmalloc(filen,procn,arran,h1-l1+1,ch16=array,lbou=l1)
  end subroutine MRMD_ALLOC_CH16_1

!===========================================================================
!
! Free up all possible memory
!
!===========================================================================
  subroutine MRMD_DEALLOC
    use stream
    implicit none
    character(len=9) ::   & ! 
        procn='MRMD_DEALLOC'         ! procedure name

    if (PRNLEV>=5) &
    write(outu,'(A)') 'MRMD_DEALLOC> Removing MRMD surface, deallocating arrays.'

    call MRMD_ALLOC(filen,procn,'rx_levelshift'    ,rx_levelshift    ,1,0)    
    call MRMD_ALLOC(filen,procn,'rx_gapo_surfs'    ,rx_gapo_surfs    ,1,0,1,0)    
    call MRMD_ALLOC(filen,procn,'rx_gapo_order'    ,rx_gapo_order    ,1,0)    
    call MRMD_ALLOC(filen,procn,'rx_gapo_coefs'    ,rx_gapo_coefs    ,1,0,1,0)    
    call MRMD_ALLOC(filen,procn,'rx_atom'          ,rx_atom          ,1,0)    
    call MRMD_ALLOC(filen,procn,'rx_atom_inv'      ,rx_atom_inv      ,1,0)    
    call MRMD_ALLOC(filen,procn,'rx_charges'       ,rx_charges       ,1,0,1,0)    
    call MRMD_ALLOC(filen,procn,'rx_vdw_eps'       ,rx_vdw_eps       ,1,0,1,0)    
    call MRMD_ALLOC(filen,procn,'rx_vdw_eps2'      ,rx_vdw_eps2      ,1,0,1,0)    
    call MRMD_ALLOC(filen,procn,'rx_vdw_rmin_half' ,rx_vdw_rmin_half ,1,0,1,0)    
    call MRMD_ALLOC(filen,procn,'rx_vdw_rmin_half2',rx_vdw_rmin_half2,1,0,1,0)    
    call MRMD_ALLOC(filen,procn,'rx_gvdw_exist'    ,rx_gvdw_exist    ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_gvdw_atoms'    ,rx_gvdw_atoms    ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_gvdw_eps'      ,rx_gvdw_eps      ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_gvdw_rmin'     ,rx_gvdw_rmin     ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_gvdw_xatt'     ,rx_gvdw_xatt     ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_gvdw_xrep'     ,rx_gvdw_xrep     ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_harm_exist'    ,rx_harm_exist    ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_harm_atoms'    ,rx_harm_atoms    ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_harm_fc_half'  ,rx_harm_fc_half  ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_harm_re'       ,rx_harm_re       ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_mors_exist'    ,rx_mors_exist    ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_mors_atoms'    ,rx_mors_atoms    ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_mors_De'       ,rx_mors_De       ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_mors_re'       ,rx_mors_re       ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_mors_beta'     ,rx_mors_beta     ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_rkhs_exist'    ,rx_rkhs_exist    ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_rkhs_atoms'    ,rx_rkhs_atoms    ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_rkhs_ngrid'    ,rx_rkhs_ngrid    ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_rkhs_grid'     ,rx_rkhs_grid     ,1,0,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_rkhs_coef'     ,rx_rkhs_coef     ,1,0,1,0,1,0)  
    call MRMD_ALLOC(filen,procn,'rx_rkhs_asym'     ,rx_rkhs_asym     ,1,0,1,0) 
    call MRMD_ALLOC(filen,procn,'rx_bond_exist'    ,rx_bond_exist    ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_bond'          ,rx_bond          ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_bond_inv'      ,rx_bond_inv      ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_bond_atoms'    ,rx_bond_atoms    ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_nharm'     ,rx_inc_nharm     ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_nmors'     ,rx_inc_nmors     ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_nmors'     ,rx_inc_nrkhs     ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_nbond'     ,rx_inc_nbond     ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_exc_bond'      ,rx_exc_bond      ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_harm'      ,rx_inc_harm      ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_mors'      ,rx_inc_mors      ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_rkhs'      ,rx_inc_rkhs      ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_bond'      ,rx_inc_bond      ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_exc_bond'      ,rx_exc_bond      ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_angl_exist'    ,rx_angl_exist    ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_angl_atoms'    ,rx_angl_atoms    ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_angl_fc_half'  ,rx_angl_fc_half  ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_angl_phie'     ,rx_angl_phie     ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_urey_exist'    ,rx_urey_exist    ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_urey_fc_half'  ,rx_urey_fc_half  ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_urey_re'       ,rx_urey_re       ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_angl'          ,rx_angl          ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_angl_inv'      ,rx_angl_inv      ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_exc_nangl'     ,rx_exc_nangl     ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_exc_angl'      ,rx_exc_angl      ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_nangl'     ,rx_inc_nangl     ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_angl'      ,rx_inc_angl      ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_dihe_exist'    ,rx_dihe_exist    ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_dihe_atoms'    ,rx_dihe_atoms    ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_dihe_mult'     ,rx_dihe_mult     ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_dihe_k'        ,rx_dihe_k        ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_dihe_phi0'     ,rx_dihe_phi0     ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_dihe_cos0'     ,rx_dihe_cos0     ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_dihe_sin0'     ,rx_dihe_sin0     ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_dihe_inv'      ,rx_dihe_inv      ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_exc_ndihe'     ,rx_exc_ndihe     ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_exc_dihe'      ,rx_exc_dihe      ,1,0,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_ndihe'     ,rx_inc_ndihe     ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_dihe'      ,rx_inc_dihe      ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_impr_exist'    ,rx_impr_exist    ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_impr_atoms'    ,rx_impr_atoms    ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_impr_mult'     ,rx_impr_mult     ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_impr_k'        ,rx_impr_k        ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_impr_phi0'     ,rx_impr_phi0     ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_impr_cos0'     ,rx_impr_cos0     ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_impr_sin0'     ,rx_impr_sin0     ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_impr_inv'      ,rx_impr_inv      ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_exc_nimpr'     ,rx_exc_nimpr     ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_exc_impr'      ,rx_exc_impr      ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_nimpr'     ,rx_inc_nimpr     ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_impr'      ,rx_inc_impr      ,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_exc_n12'       ,rx_exc_n12       ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_exc_n13'       ,rx_exc_n13       ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_exc_n14'       ,rx_exc_n14       ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_exc_n15'       ,rx_exc_n15       ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_n12'       ,rx_inc_n12       ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_n13'       ,rx_inc_n13       ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_n14'       ,rx_inc_n14       ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_n15'       ,rx_inc_n15       ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_atom1234'      ,rx_atom1234      ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_atom1234_inv'  ,rx_atom1234_inv  ,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_12'        ,rx_inc_12        ,1,0,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_exc_12'        ,rx_exc_12        ,1,0,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_13'        ,rx_inc_13        ,1,0,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_exc_13'        ,rx_exc_13        ,1,0,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_14'        ,rx_inc_14        ,1,0,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_exc_14'        ,rx_exc_14        ,1,0,1,0,1,0)
    call MRMD_ALLOC(filen,procn,'rx_inc_15'        ,rx_inc_15        ,1,0,1,0,1,0)

  end subroutine MRMD_DEALLOC
!===========================================================================
  
#endif /*(mrmd)*/

 end module MRMD_FCM
