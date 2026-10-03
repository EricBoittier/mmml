! qm and qm/mm info
module qm1_info
  use chm_kinds
  use dimens_fcm
  use iso_c_binding, only: c_double, c_int, c_char ! for interfacing with c code ! arat2025sep15

#if KEY_MNDO97==1
  implicit none
  !
  ! List of TYPE definition
  ! qm_control
  ! qm_main
  ! mm_main
  ! qm_scf_main
  ! qm_param
  ! qm_scf_indx   ! used mainly in qm1_scf_module.F90
  ! qm_scf_diis
  ! qm_fockmd_diis
  ! xlbomd_data   ! used in qmmm_interface.F90
  ! qm_gho_info   ! used mainly in mndgho_module.F90
  !
  integer, save :: num_qm_system= 1             ! default, only a single system, i.e., rs state
  integer, save :: irepl_high   =-1             ! replica no. for high-level qm/mm correction   

  TYPE, public :: qm_control
    logical          :: ifqnt=.false.           ! main logical flag (.true., if qm/mm is on.)
    logical          :: md_run=.false.          ! running molecular dynamics?

    ! DXL-BOMD (AMN Niklasson & Guishan Zheng) related
    logical          :: q_dxl_bomd=.false.      ! use DXL-BOMD (AMN Niklasson & Guishan Zheng)
    logical          :: q_do_dxl_scf=.false.    ! runtime control if scf finishes in N_scf_step (in scf_iter).
    integer          :: N_scf_step = 0          ! number of scf steps.

    ! Fock matrix dynamics (based on DIIS)
    logical          :: q_fockmd=.false.        ! use Fock matrix dynamics (ref: ).
    logical          :: q_do_fockmd_scf=.false. ! runtime control of the Fock matrix dynamics (in scf_iter).
    integer          :: imax_fdiss              ! maximum number of diis iteration.
    integer          :: i_fockmd_option=2       ! options for fock-md.
                                                ! 1: based on the extrapolation
                                                ! 2: based on the Verlet integration.

    !
    character(len=6) :: qm_model                ! mndo(1),am1(2),pm3(3),am1/d(4),mndo/d(5)
    integer          :: iqm_mode = 2            ! 1,2,3,4,5
    logical          :: q_am1_pm3=.false.       ! .false. (default), if mndo or mndod
                                                ! .true. if am1, pm3, or am1d
    logical          :: do_d_orbitals=.false.   ! have d-orbital specific terms.
 
    logical          :: qmqm_analyt=.false.     ! use qm-qm analytical derivative?
                                                ! .false. (default) for now!
    logical          :: q_diis=.false.          ! use DIIS.
    logical          :: qsrp_phot=.false.       ! use am1 or am1/d-PhoT qm model

    ! for qm/mm energy (in kcal/mol)
    real(chm_real)   :: E_total,E_scf,E_nuclear 
    real(chm_real)   :: E_ewald_corr

    ! for finite difference derivative.
    real(chm_real)   :: del =1.0D-06, &         ! displacement distance
                        rdel=5.0D+05

    ! for mapping between qm and mm part
    integer, allocatable :: qminb(:)            ! pointer for qm atoms in the charmm main array
    integer, allocatable :: mminb1(:)           ! pointer for mm atoms in the charmm main array
    integer, allocatable :: mminb2(:)           ! the reverse pointer of mminb1.
    real(chm_real),allocatable:: cgqmmm(:)      ! charge of mm atoms in the original CG array.

    ! for analysis
    logical          :: q_bond_order=.false.    ! bond order analysis.
    logical          :: q_m_charge  =.false.    ! Mulliken population analysis
    integer          :: ianal_unit  =6          ! output unit for analysis results.

    ! for mlayered qm/mm & high-level qm/mm calculations.
    integer,allocatable:: igmsel(:)             ! local copy of igmsel
    real(chm_real),allocatable:: AZNUC_local(:) ! local copy for aznuc to be used in ab initio qm/mm
    real(chm_real),allocatable:: cg_local(:)    ! local copy for CG array for high-level qm/mm.
    character(len=10),allocatable:: CAATOM_local(:) ! local copy for CAATOM

  END TYPE qm_control

  !
  TYPE, public :: qm_main
    integer          :: NUMAT=0                 ! number of qm atoms.
    logical          :: rij_qm_incore =.false.  ! use incore if true.

    real(chm_real)   :: qmcharge=0.0d0          ! total charge of qm region
    integer          :: i_qmcharge=0            ! integer version of qmcharge

    ! imult needs explanation, for both RHF and UHF
    ! 0; singlet, 2; doublet, 3; triplet
    integer          :: imult                   ! spin state of qm region.
    logical          :: uhf=.false.             ! default is RHF (so, uhf=.false.)
    !
    real(chm_real)   :: elec_eng, &             ! electronic energy (eV)
                        ener_atomic, &          ! sum of atomic energy (eV)
                        HofF_atomic, &          !        atomic heat of formation (eV)
                        enuclr_qmqm, &          ! core-core energy for qm-qm (eV)
                        enuclr_qmmm             ! core-core energy for qm-mm (eV)

    !
    integer,allocatable  :: nat(:)              ! atomic number of atom i
    integer,allocatable  :: nfirst(:), &        ! first basis orbital of atom i
                            nlast(:)            ! last  basis orbital of atom i.

    ! number of orbitals, electrons, occupation numbers
    integer          :: norbs                   ! number of orbitals
    integer,allocatable  :: num_orbs(:)         ! number of orbitals for atom i.
    
    integer          :: ijpair                  ! norbs*(norbs-1)/2+norbs

    integer          :: nel,    &               ! number of electrons
                        nalpha, &               !           alpha electrons
                        nbeta                   !           beta
    !  note: nbeta = number of doubly occupied molecular orbitals 
    !                if RHF and imult .ne. 1
    integer          :: numb, &                 ! number of highest occupied mol. orbital = no. of occup. orb.
                        nclo, &                 ! number of closed orbitals?
                        nmos                    !        of occupied mol. orbital
    !
    ! HALFE
    integer          :: iodd, &                 ! 0, for RHF
                        jodd                    ! 0, for RHF
    ! OCCFL
    integer          :: imocc, &                ! 1=abs(iuhf), if RHF, iuhf=-1 (default).
                        nocca, &                !
                        noccb                   !

  
    ! QM coordinates (3,numat)
    real(chm_real),allocatable :: qm_coord(:,:)
    real(chm_real),allocatable :: qm_grads(:,:) 
    real(chm_real),allocatable :: qm_grads2(:,:) 

    ! moved to qmmm_ewald type
    !real(chm_real),allocatable :: rijdata_qmqm(:,:)  ! incore data, rij, 1/rij.

    ! for the non-bond list generation
    !integer           :: nmaxgrp=201
    !integer           :: nqmgrp(200+1)           ! as nqmgrp(1) is the number of QM groups.
    integer             :: nmaxgrp =-9999
    integer,allocatable :: nqmgrp(:)

    ! d-orbital related info.
    ! DELEMT: updated in qm_info_setup routine
    integer      :: NELMD=0, &                  ! Number of atoms with D-orbitals.
                    NFOCK=5                     ! setp size in Subroutine FOCK (5 for sp, 10 for spd).

    ! occupation number related info.
  END TYPE qm_main

  ! 
  TYPE, public :: mm_main
    integer    :: natom                         ! the total number of atoms in the system.
    integer    :: natom_mm                      ! the total number of MM atoms in the system.
    integer    :: NUMATM                        ! number of MM atoms (within cutoff)
    integer    :: nlink=0                       ! number of qm-mm link atoms (H-link atoms) 
    logical    :: rij_mm_incore=.false.         ! use incore if true.

    ! for mapping between mndo97 routines and charmm
    integer,allocatable :: qm_mm_pair_list(:)
    ! mm coordinates
    real(chm_real),allocatable :: mm_coord(:,:)
    ! mm gradients
    real(chm_real),allocatable :: mm_grads(:,:)
    real(chm_real),allocatable :: mm_grads2(:,:)
    ! mm charges
    real(chm_real),allocatable :: mm_chrgs(:)
    !
    ! moved to the qmmm_ewald type
    !real(chm_real),allocatable :: rijdata_qmmm(:,:) ! incore data, rij, 1/rij.

    ! Cutoff based on group-based cutoff.
    ! 1. In the non-bond list (i.e., the default (default) group-based list), any MM group
    !    that is within the cutoff distance from any QM group is included.
    ! 2. When evaluating qm-mm interactions, for each QM group, any MM group in the list
    !    is evaluated for their interaction with each QM group if that MM group is within  
    !    the cutoff distance.
    ! 3. If the switch option is on, the interaction between the dist_on and dist_off 
    !    is switched off at the dist_off distance. 
    ! 4. When LQMEWD is true, the interaction energy between the i (QM) and j (MM) atom pair
    !    is 
    ! 
    !    E_ij (r_ij) = Sw(r_ij)*E_qm-mm-model(r_ij) + (1-Sw(r_ij))*E_qm-mm-long-distance-model(r_ij)
    !
    !    where E_qm-mm-model is the regular QM-MM interaction model, and 
    !          E_qm-mm-long-distance-model is the QM-Mulliken-charge and MM charge interaction,
    !          which is used in the QM/MM-Ewald and QM/MM-PME long interaction model.
    !
    integer :: inum_mm_grp               ! total number of mm groups within the cutoff dist.
                                         ! this is different from num_mm_group (nbndqm_ltm.src).
                                         ! in fact, this is the actual no. of mm groups in the cutoff.
    real(chm_real),allocatable :: r_num_atom_qm_grp(:)     ! 1/float(n_qm_atm_i-th qm group)
    real(chm_real),allocatable :: r_num_atom_mm_grp(:)     ! 1/float(n_mm_atm_j-th mm group)

    ! switching  function related.
    logical :: q_switch =.false.                       ! use switch function between on and off distances.
    integer,allocatable :: q_mmgrp_point_swt(:)        ! point which qm group is in the switch region of mm.
    integer,allocatable :: q_mmgrp_qmgrp_swt(:,:)      ! num_mm_group x num_qm_group pair
                                                       ! > 0  this qm-mm pair within the switch region. 
                                                       ! < 0  this qm-mm pair outside of the switch region.
    integer :: isize_swt_array
    real(chm_real),allocatable :: sw_val(:), &         ! swtching function value(rij)
                                  dxyz_sw(:,:),  &     ! gradient components for each group.
                                  dsw_val(:,:),dxyz_sw2(:,:)      ! dSw(rij)/drij*{xyz(1:3,qm)-xyz(1:3,mm)}
    integer,allocatable :: q_backmap_dxyz_sw(:)        ! pointer to which qm-mm group pair 
                                                       ! to add the gradient components.

    ! With this flag on, we replace the diagonal block of the qm/mm interaction is replaced by
    ! purely 1/r interaction, which is calculated in the ewald routines.
    logical :: q_diag_coulomb =.false.                 ! should be only used when qm/mm-ewald or qm/mm-pme is used.
    
    ! for qm/mm-ewald or pme version
    logical :: LQMEWD =.false.                         ! use qm/mm-Ewald.
    logical :: PMEwald=.false.                         ! use qm/mm-PME.
    integer :: EWMODE =1                               ! Ewald mode.
    integer :: NQMEWD =0                               !
    !
    real(chm_real),allocatable :: qm_charges(:)        ! qm mulliken charges.
  END TYPE mm_main

  !
  TYPE, public :: qm_scf_main
    ! array size 
    ! LM1    : numat
    ! LM2    : norbs
    ! LM3    : norbs
    ! LM4    : norbs*(norbs+1)/2
    ! LM6    : linear dimension of square fock matrix, 
    ! LM9    : LM6*LM6
    ! LWORK  : 8*NORBS
    ! LIWORK : norbs
    !
    integer :: dim_numat                          ! the size of the number of atoms (LM1)
    integer :: dim_norbs                          ! the size of the number of orbitals (LM2)
    integer :: dim_norbs2                         ! norbs*norbs (LM2*LM3)
    integer :: dim_linear_norbs                   ! linear dimension; LM4=norbs*(norbs+1)/2
    integer :: dim_linear_fock                    ! linear dimension of half-triangle 
                                                  !        of fock matrix calc.; LM6
    integer :: dim_linear_fock2                   ! square size of dim_linear_fock; LM9
                                                  ! neeed for two-electron interactions.
    integer :: dim_2C_2E_integral                 ! unique two-center two-electron integrals.
    integer :: dim_scratch                        ! the size of real scratch array   : LWORK=8*norbs
    integer :: dim_iscratch                       ! the size of integer scratch array: LIWORK=norbs
               
    ! SCRT
    real(chm_real) :: SCFCRT=1.0d-6               ! scf convergence criteria
    real(chm_real) :: PLCRT =1.0d-4               ! density conv. criteria

    ! maximum scf iteraction
    integer        :: KITSCF=200                  ! default value.

    ! INDEXT 
    ! size LMX, this is initialized in DYNSCF, construct_index
    ! LMX = 9*numat
    integer,allocatable :: INDX(:)                ! size LMX, this is initialized in DYNSCF
    !
    ! INDEXW: NW(i) is the first element point of the lower triangle of block diagonal
    !         term. *see routine define_pair_index (qm1_scf_module).
    ! LM1 = numat
    integer,allocatable :: NW(:)    ! size LM1

    ! needed arrays.
    ! H(LM4); W(LM9); FA(LM4)=FB; CA(LM2,LM3)=CB; PA(LM4)=PB; DA(LM4)=DB; EA(LM3)=EB
    ! Q(LM2*8); iwork(6*LMX)
    real(chm_real),allocatable:: H(:)               ! core-hamiltonian matrix.
    real(chm_real),allocatable:: W(:)               ! Two-electron integrals. (1D .vs. 2D?)
    real(chm_real),pointer    :: FA(:)=>null(), &   ! RHF of UHF-alpha Fock matrix.
                                 FB(:)=>null()      ! UHF-beta Fock matrix.
    real(chm_real),allocatable:: FAwork(:,:)        ! Fock matrix in square form, working array.
    real(chm_real),allocatable:: FBwork(:,:)        ! beta Fock matrix in square form
    real(chm_real),pointer    :: CA(:,:)=>null(), & ! RHF of UHF-alpha MO eigenvectors.
                                 CB(:,:)=>null()    ! UHF-beta MO eigenvectors.
    real(chm_real),pointer    :: PA(:)=>null(), &   ! RHF of UHF-alpha density matrix.
                                 PB(:)=>null()      ! UHF-beta density matrix.
    real(chm_real),allocatable:: PAwork(:,:)        ! Fock matrix in square form, working array.
    real(chm_real),allocatable:: PBwork(:,:)        ! beta Fock matrix in square form
    real(chm_real),pointer    :: DA(:)=>null(), &   ! RHF of UHF-alpha difference density matrix.
                                 DB(:)=>null()      ! UHF-beta difference density matrix.
    real(chm_real),pointer    :: EA(:)=>null(), &   ! RHF of UHF-alpha MO eigenvalues.
                                 EB(:)=>null()      ! UHF-beta MO eigenvalues.

    real(chm_real),allocatable:: Q(:)             ! Scaratch array for various purposes.
    integer,       allocatable:: iwork(:)         ! integer scratch array (see iter.)

    
    ! for precomputing some parameters:
    real(chm_real),allocatable:: H_1cent(:)      ! core-hamiltonina matrix for one-center terms.
    integer,       allocatable:: imap_h(:)       ! mapping H_1cent to H in one_center_h
    !integer               ::dim_imap_h          ! size of imap_h array
    
    ! for EXTRA1 & EXTRA2
    real(chm_real) :: RI(22),CORE_mat(10,2),WW(2025)  ! REPT(10,2)
    real(chm_real) :: SIJ(14),T(14),YY(675)
  END TYPE qm_scf_main

  !
  TYPE, public :: qm_param
    ! Here, we copy atomic parameters for each atoms. So, we only go over the no. of
    ! QM atoms, instead of referecing to the atomic number.
    !
    ! For definition of each variable, refers to qm1_param.F90.
    !
    ! The following arrays are all size of "numat", i.e., the total no. of qm atoms.
    logical,        allocatable :: q_atom_specific(:)  ! .true. if atom specific params are used
                                                       ! .false. (default), use the same param for all atoms
                                                       !                    with the same atom number
    integer,        allocatable :: ni_local(:),    &   ! qm_main_c%nat
                                   iorbs_local(:), &   ! qm_main_c%num_orbs = LORBS, so the same as LORBS below.
                                   ia_local(:),    &   ! qm_main_c%NFIRST
                                   ib_local(:),    &   ! qm_main_c%NLAST
                                   ip_local(:),    &   ! qm_scf_main_c%NW(i)
                                   is_local(:),    &   ! qm_scf_main_c%indx(ia)+ia
                                   iw_local(:),    &   ! qm_scf_main_c%indx(iorbs)+iorbs
                                   Jmax_local(:)       ! 
    integer,        allocatable :: int_ij(:),      &
                                   int_kl(:)

    integer,        allocatable :: IMPAR(:),IM1D(:),III(:),IIID(:),IOS(:),IOP(:),IOD(:), &
                                   LORBS(:)

    integer,        allocatable :: IF0SD(:),IG2SD(:), &
                                   MALPB(:)    ! IMP(:),   &

    real(chm_real), allocatable :: CORE(:),EHEAT(:),EISOL(:), &
                                   DELTA(:),OMEGA(:)

    real(chm_real), allocatable :: USS(:),UPP(:),ZS(:),ZP(:),ALP(:),BETAS(:),BETAP(:), &
                                   GSS(:),GPP(:),GSP(:),GP2(:),HSP(:),HPP(:),   &
                                   QQ(:),AM(:),AD(:),AQ(:),GNN(:)
    ! d-oribitals related params
    real(chm_real), allocatable :: UDD(:),ZD(:),BETAD(:),ZSN(:),ZPN(:),ZDN(:), &
                                   F0DD(:),F2DD(:),F4DD(:),F0SD(:),F0PD(:),F2PD(:),  &
                                   G2SD(:),G1PD(:),G3PD(:)
 
    ! 2d arrays
    real(chm_real), allocatable :: DD(:,:),     &  ! DD(6,0:numat)
                                   PO(:,:),     &  ! PO(9,0:numat) 
                                   ALPB(:,:),   &  ! ALPB(numat,numat), pair-wise parameters
                                   GUESS1(:,:), &  ! GUES(1(4,numat) 
                                   GUESS2(:,:), &  ! GUESS2(4,numat)
                                   GUESS3(:,:)     ! GUESS3(4,numat)

    real(chm_real), allocatable :: REPD(:,:),&   ! REPD(52,numat)
                                   w_save(:,:)

    ! for modified/local parameters
    real(chm_real), allocatable :: GSS_local(:), &  ! for RHF, these are modified
                                   GSP_local(:), &  ! see QMMM_module_prep.
                                   GPP_local(:), &
                                   GP2_local(:), &
                                   HSP_local(:), &
                                   HPP_local(:)

    ! for mm atoms
    real(chm_real)              :: PO_mm(9),DD_mm(6)
    
                                   
    !logical                     :: R2CENT(45,45)
    !
  END TYPE qm_param

  ! originally from qm1_scf_module.F90
  TYPE, private :: qm_scf_indx
     ! used is pair indexing
     ! see qm1_scf_module.F90, define_pair_indix, which is called from qmmm_load_parameters_setup_qm_info.
     ! FINDX1 & FINDX2
     ! IP(LMI)   indices of unique one-center AO pairs = I(I-1)/2+J
     ! IP1(LMI)  index of first AO in the one-center pair = I (coul)
     ! IP2(LMI)  index of 2nd   AO in the one-center pair = J (coul)
     ! JP1(LME)  index of first AO in the one-center pair = I (exch)
     ! JP2(LME)  index of 2nd   AO in the one-center pair = J (exch)
     ! JP3(LME)  coulomb pair index for given exchange pair index
     ! JX(LM1)   1st  exchange pair index for given atom
     ! JXLAST    last exchange pair index for last  atom
     ! LMI= 4*numat; LME= 81*numat
     integer,allocatable :: ip_local(:) ,ip1_local(:),ip2_local(:)
     logical,allocatable :: ip_check(:)
     !
     ! used in uhf case
     integer,allocatable :: jp1_local(:),jp2_local(:),jp3_local(:), &
                            jx_local(:)
     integer             :: jxlast_local
#if KEY_PARALLEL==1
     logical,allocatable :: q_mynod_fock(:)
     logical             :: q_mynod_fock_setup

     ! ewald_add_fock case
     logical,allocatable :: q_do_atom(:)
     logical             :: q_do_atom_setup
#endif

     ! cal_density_matrix case
     real(chm_real),allocatable :: pn_diag(:)
  END TYPE qm_scf_indx

  ! diis related part.
  TYPE, public :: qm_scf_diis
  ! for DIIS related.
    ! FDA(LM4,MXDIIS+1)=FDB; Ediis(LM4,MXDIIS+1); Bdiis(MX1P); Adiix(MX1P); Xdiix(MXDIIS+1);
    ! iwork_diis(6*LMX), LMX=9*LM1
    integer            :: mxdiis=100          ! maximum number of diis iterations allowed.
    integer            :: mx1   =101          ! = mxdiis+1
    integer            :: mx1p  =5151         ! =(mx1*(mx1+1))/2
    !
    real(chm_real),allocatable:: FDA(:,:), &  ! Fock matrices from Diis iterations, RHF or UHF-Alpha.
                                 FDB(:,:)     !                                   , UHF-beta
    real(chm_real),allocatable:: Ediis(:,:)   ! Error matrixces from Diis iterations.
    real(chm_real),allocatable:: Bdiis(:)     ! Coefficient matrix for Diis linear equations.
    real(chm_real),allocatable:: Adiis(:)     ! Coefficient matrix for Diis linear equations.
    real(chm_real),allocatable:: Xdiis(:)     ! RHS vector and solutions for Diis linear equations.
    ! integer scratch array
    integer,allocatable :: iwork_diis(:)      ! iwork in routine diis size: 12*numat

#if KEY_PARALLEL==1
    logical,allocatable :: q_ij_pair(:)             ! q_ij_pair array for square2
    logical             :: q_ij_pair_setup=.false.  ! 
    real(chm_real),allocatable :: bdiis_local(:)    ! bdiis_local
#endif
  END TYPE qm_scf_diis

  ! fock matrix dynamics (ref: )
  TYPE, public :: qm_fockmd_diis
     integer                :: mxfdiis        ! 5: n-2,n-1,n,n+1,n+2 or..
                                              ! 7: n-3,n-2,n-1,n,n+1,n+2,n+3
     real(chm_real),allocatable :: FA_sv(:)   ! Fock matrix from the previous md step.
     real(chm_real),allocatable :: FDA(:,:)   ! Fock matrices for Fock md iteration.
  END TYPE qm_fockmd_diis

  ! used in qmmm_interface.F90
  ! for DXL-BOMD with AMN Niklasson, JCP (2009) 130:214109 & Guishan Zheng, Harvard Univ.
  TYPE, private :: xlbomd_data
     integer                    :: Kth_sum_order=0 ! K value (for AMN Niklasson, JCP (2009) 130:214109).
     real(chm_real),allocatable :: cextr(:)        ! coefficient(0:K_sum_order)
     real(chm_real),allocatable :: pa_exp(:,:)     ! PA(K_sum_order,linear_norbs)
     real(chm_real),allocatable :: pa_aux(:), &    ! auxiliary PA for DXL-BOMD 
                                   pa_anew(:)      ! new auxiliary PA (i.e., pa_aux(i+1)
     real(chm_real)             :: coefk           ! kappa value (This is not used.)
  END TYPE xlbomd_data


  ! used in mndgho_module.F90
  TYPE, public :: qm_gho_info
     ! Store GHO related information 
     logical            :: q_gho=.FALSE.         ! Logical flag to use GHO atoms
     logical            :: uhfgho=.FALSE.        ! Logical flag to use UHF/GHO
     integer            :: mqm16=16
     integer            :: numat=0               ! number of qm atoms
     integer            :: nqmlnk=0              ! number of GHO atoms
     integer            :: norbsgho=0            ! number of AOS, same as qm2_struct%norbs
     integer            :: norbhb,naos,lin_naos,lin_norbhb  ! used in scf iteraction step.
     integer            :: norbao,nactatm,lin_norbao        ! used in FTOFHB
     integer, allocatable   :: IQLINK(:)         ! Pointer for GHO atom
     integer, allocatable   :: JQLINK(:,:)       ! Pointer for MM atoms connected to
                                                 ! GHO atom (3,nqmlnk)
     integer, allocatable   :: KQLINK(:)         ! Pointer for QM atom  connected to
                                                 ! GHO atom;  Sizes are (nqmlnk)
     real(chm_real),allocatable   :: QMATMQ(:)   ! Size  is  (nqmlnk) 
     real(chm_real),allocatable   :: BT(:)       ! Size  is  (nqmlnk*mqm16)    C' = BT C
     real(chm_real),allocatable   :: BTM(:)      ! Size  is  (nqmlnk*mqm16)    C  = BTM C'
     real(chm_real),allocatable   :: DBTMMM(:,:,:)  ! Size  is  (3,3,nqmlnk*mqm16)
     real(chm_real),allocatable   :: PHO(:)      ! density matrix for GHO, size is
                                                 ! (norbs*(norbs+1)/2)
     real(chm_real),allocatable   :: PBHO(:)     ! density matrix for GHO, size is
                                                 ! (norbs*(norbs+1)/2)
     real(chm_real),allocatable   :: FAOA(:)     ! density matrix for GHO, size is
                                                 ! (norbs*(norbs+1)/2)
     real(chm_real),allocatable   :: FAOB(:)     ! density matrix for GHO, 
                                                 ! Size is same as density matrix, 
                                                 ! which is (norbs*(norbs+1)/2)
     ! Local varibles only at qm2_scf and etc.
     real(chm_real), allocatable :: CAHB(:)      ! Size is norbs*norbs
     real(chm_real), allocatable :: CBHB(:)      !         norbs*norbs
     real(chm_real), allocatable :: DAHB(:)      !         norbs*(norbs*+1)/2
     real(chm_real), allocatable :: DBHB(:)      !         norbs*(norbs*+1)/2
     real(chm_real), allocatable :: FAHB(:)      !         norbs*(norbs*+1)/2
     real(chm_real), allocatable :: FBHB(:)      !         norbs*(norbs*+1)/2
     real(chm_real), allocatable :: PAHB(:)      !         norbs*(norbs*+1)/2
     real(chm_real), allocatable :: PBHB(:)      !         norbs*(norbs*+1)/2
     real(chm_real), allocatable :: PAOLD(:)     !         norbs*(norbs*+1)/2
     real(chm_real), allocatable :: PBOLD(:)     !         norbs*(norbs*+1)/2
     real(chm_real), allocatable :: FAHBwrk(:,:) !    norbs,norbs
     real(chm_real), allocatable :: FBHBwrk(:,:) !    norbs,norbs
  END TYPE qm_gho_info

  ! used in mndo97_mlay.F90
  ! mlayered qm/mm related variables
  TYPE,public :: mlay_array
    logical          :: qmlay_main=.false.      ! main flag for mlayered qm/mm method. default (.false.)
    logical          :: qmlay_high=.false.      ! use mlayered qm/mm method: .false. == low level qm/mm (default)
                                                !                            .true.  == high level qm/mm theory
    logical          :: qh_link =.false.        ! link used in the high level qm region? QQH should not set qh_link=.true.
    logical          :: qmlay_mts=.false.       ! use MTS mlayered qm/mm method (default =.false., i.e., do not use MTS)

    ! printing related
    logical          :: qmlay_print=.false.     ! print energy/gradients
    integer          :: imlay_print=6           ! unit to print energy/gradients.
                                                ! assume printing high-level region.
    real(chm_real)   :: e_pr_low,e_pr_high      ! low/high level energies
    real(chm_real),allocatable:: dx_mpr_low(:),dy_mpr_low(:),dz_mpr_low(:),   &  ! mm region
                                 dx_mpr_high(:),dy_mpr_high(:),dz_mpr_high(:)
    integer,allocatable::        mm_flag(:)     ! 0: mm atoms outside cutoff
                                                ! 1:          within  cutoff
                                                ! 5: qm atoms
                                                !

    integer          :: natom                   ! total no. of atoms (local copy)

    integer, allocatable :: igmsel(:)           ! igmsel local copy
                                                ! igmsel is allocated in mlay_memory_allocate

    integer          :: NHSTP = 1               ! The frequency to do the high-level qm/mm calc. during MD simulations.
                                                ! default == 1, do every MD step
    integer          :: NMDSTP= 0               ! MD step counter

    ! qh_link == .true. (QQH should not turn on qh_link=.true.)
    ! when qm-mm bonds are cut, where both qm and mm atoms are part of the main qm region
    ! the h-link atoms are introduced for the dangling bonds.
    ! num_h_link is the no. of such h-link atoms, the forces on which are projected.
    integer              :: num_h_link=0        ! no. of h-link atoms, excl
    integer, allocatable :: ihostguest(:,:)     ! host (qm) and guest (mm) atoms for cutting qm-mm bonds
                                                ! ihostguest(1,*): qm atom (link host)
                                                ! ihostguest(2,*): mm atom (link guest)
                                                ! ihostguest is allocated in copsel_dual
    real(chm_real),allocatable:: xyz_hlink(:,:) ! temporay copy of xyz_hlink_qm host and mm guest positions
    real(chm_real),allocatable:: r_h_ref(:)     ! reference distance for locating h-link atom
    logical, allocatable :: qcheck_tmp(:)       ! temporay logical array ...

    integer              :: lp_level=1          ! how to project forces on the h-link atoms.
                                                ! 1: default, forces along the H-QM atom is projected out.
                                                ! 2:          no force projection

    real(chm_real) :: lambda_scale=1.0d0, &     ! scaling factor for the mlayered qm/mm energy and gradients.
                      dlambda_scale=0.0d0       ! during mts md, change lambda by dlambda at each MD step, so
                                                ! to determine (non-equilibrium) work between lambda 0 and 1.
    real(chm_real) :: E_value                   ! Total energy, for mlayered qm/mm, it is a correction energy 
                                                !                                      (e.g., E_high - E_low).
    real(chm_real) :: E_value_save
    real(chm_real),allocatable:: dx_repl(:),dy_repl(:),dz_repl(:)                ! gradients 
    real(chm_real),allocatable:: dx_repl_save(:),dy_repl_save(:),dz_repl_save(:) ! gradients save for md
    logical,allocatable       :: q_mm_flag(:)   ! flag if atoms are included in the high-level qm/mm calc.

  END TYPE mlay_array

  ! Used in gukint/gukini.F90 qchem_mlayer subroutine for qchem checks - arat
  integer           :: QQBUILD=-1   ! arat-mtsmlp, -1: NOT SET, 1: OMP BUILD, 2: MPI BUILD
  integer           :: QQPARMODE=-1 ! arat-mtsmlp: -1: NOT SET, 0:EXECUTE WITH QC NTHREADS ENV VARIABLE, 1: EXECUTE BASED ON CHARMM MPI NPROCESS COUNT

  ! used in mndo97_mlp.F90
  ! MLP/delta-MLP QM/MM related
  TYPE,public :: qmmm_mlp_array
     logical          :: qmmm_mlp =.false.      ! main flag for MLP/delta-MLP qm/mm method. default (.false.)
     logical          :: qmhub_python  =.false.      ! use QMHub PYTHON FIle I/O interface
     logical          :: qmhub_dpmm =.false.      !     DPMM (LibTorch) interface
     logical          :: qmmm_mlp_only=.true.   ! only do MLP/delta-MLP not QMLAY_high (to overwrite QMLAY_high)
     integer          :: qmlp_mode=-1           ! MLP model (0: MLP; 1: delta-MLP)
     integer          :: nref_replica=1         ! reference qm region replica (default: the main one)

     real(chm_real) :: E_MLP                    ! MLP energy
     real(chm_real),allocatable:: dx_mlp(:), &  ! gradient
                                  dy_mlp(:), &
                                  dz_mlp(:)



     character(len=:), allocatable :: package_mlp, model_mlp, ctrl_mlp, filein_mlp, fileout_mlp ! arat2025sep15, ctrl is spec file for model
     character(kind=c_char), allocatable :: model_mlp_c(:),  ctrl_mlp_c(:) ! arat2025sep15

     integer(c_int) :: qmmax_mlp_c                                             ! arat2025sep15  
     integer(c_int) :: mmmax_mlp_c                                             ! arat2025sep15
     integer(c_int) :: ntypes_mlp_c                                             ! arat2025sep15    
     integer(c_int), allocatable :: types_mlp_c(:)                             ! arat2025sep15

     integer :: use_gpu_mlp, use_omp_mlp                                       ! arat2025sep15
     integer(c_int) :: use_gpu_mlp_c, use_omp_mlp_c                            ! arat2025sep15

     real(c_double),allocatable:: qmx_mlp_c(:), qmy_mlp_c(:),qmz_mlp_c(:)           ! arat2025sep15
     integer(c_int), allocatable :: qm_Z_mlp_c(:)                              ! arat2025sep15
     real(c_double),allocatable:: mmx_mlp_c(:), mmy_mlp_c(:),mmz_mlp_c(:), mmcg_mlp_c(:)           ! arat2025sep15

     real(c_double),allocatable:: qmdx_mlp_c(:), qmdy_mlp_c(:),qmdz_mlp_c(:)        ! arat2025sep15
     real(c_double),allocatable:: mmdx_mlp_c(:), mmdy_mlp_c(:),mmdz_mlp_c(:)         ! arat2025sep15

     integer(c_int)   :: qmmm_mlp_setup_err=0 ! error flag coming from C++

     ! relavant options

  END Type qmmm_mlp_array

  ! bonded term infor for qm region
  TYPE, public :: qm_bond_info
     logical :: qbond_qm = .false., &           ! flag if bond  is set
                qangle_qm= .false., &           !         angle
                qdihe_qm = .false., &           !         dihe
                qimph_qm = .false.              !         imph
     integer :: nbond_qm = 0, &                 ! no. of bonds  for the qm region
                nangle_qm= 0, &                 !        angles
                ndihe_qm = 0, &                 !        dihes
                nimph_qm = 0                    !        imphrs

     integer,allocatable:: i_mm_bond(:,:), &    ! atom indeces for the bonds  (2,nbond_qm) size
                           i_mm_angl(:,:), &    !                      angles (3,nangle_qm) size
                           i_mm_dihe(:,:), &    !                      dihes  (4,ndihe_qm) size
                           i_mm_imph(:,:)       !                      imphs  (4,nimph_qm) size
  END TYPE qm_bond_info

  ! assign 
  TYPE(qm_control)    ,target,allocatable,save :: qm_control_r(:)
  TYPE(qm_main)       ,target,allocatable,save :: qm_main_r(:)
  TYPE(mm_main)       ,target,allocatable,save :: mm_main_r(:)
  TYPE(qm_scf_main)   ,target,allocatable,save :: qm_scf_main_r(:)
  TYPE(qm_param)      ,target,allocatable,save :: qm_param_r(:)
  TYPE(qm_scf_indx)   ,target,allocatable,save :: qm_scf_indx_r(:)
  TYPE(qm_scf_diis)   ,target,allocatable,save :: qm_scf_diis_r(:)
  TYPE(qm_fockmd_diis),target,allocatable,save :: qm_fockmd_diis_r(:)
  TYPE(xlbomd_data)   ,target,allocatable,save :: dxlbomd_r(:)
  TYPE(qm_gho_info)   ,target,allocatable,save :: qm_gho_info_r(:)
  type(mlay_array)    ,target,allocatable,save :: mlay_r(:)
  type(qm_bond_info)  ,target,allocatable,save :: qm_bond_r(:)
  type(qmmm_mlp_array),save                    :: qmmm_mlp

  ! assign as pointers
  TYPE(qm_control)    ,pointer,save :: qm_control_c  =>null()
  TYPE(qm_main)       ,pointer,save :: qm_main_c     =>null()
  TYPE(mm_main)       ,pointer,save :: mm_main_c     =>null()
  TYPE(qm_scf_main)   ,pointer,save :: qm_scf_main_c =>null()
  TYPE(qm_param)      ,pointer,save :: qm_param_c    =>null()
  TYPE(qm_scf_indx)   ,pointer,save :: qm_scf_indx_c =>null()
  TYPE(qm_scf_diis)   ,pointer,save :: qm_scf_diis_c =>null()
  TYPE(qm_fockmd_diis),pointer,save :: qm_fockmd_diis_c =>null()
  TYPE(xlbomd_data)   ,pointer,save :: dxlbomd_c     =>null()
  TYPE(qm_gho_info)   ,pointer,save :: qm_gho_info_c =>null()
  type(mlay_array)    ,pointer,save :: mlay_c => null()
  type(qm_bond_info)  ,pointer,save :: qm_bond_c => null()

  ! 
  contains

     !==================================================================
     subroutine mndo97_memory_init(nrepl,qallocate)
     !
     ! allocate memories
     !
     use nbndqm_mod, only : map_grp_r,map_grp_c
     !!use qmmmewald_module, only : qmmm_ewald_r,qmmm_ewald_c  ! to avoid circular dependency 
     !!use H4_mndo, only : dist_r,dist_c,grad_r,grad_c         !
     implicit none
     integer :: nrepl
     logical :: qallocate

     ! first pointers, nullify
     if(associated(qm_control_c))     nullify(qm_control_c)
     if(associated(qm_main_c))        nullify(qm_main_c)
     if(associated(mm_main_c))        nullify(mm_main_c)
     if(associated(qm_scf_main_c))    nullify(qm_scf_main_c)
     if(associated(qm_param_c))       nullify(qm_param_c)
     if(associated(qm_scf_indx_c))    nullify(qm_scf_indx_c)
     if(associated(qm_scf_diis_c))    nullify(qm_scf_diis_c)
     if(associated(qm_fockmd_diis_c)) nullify(qm_fockmd_diis_c)
     if(associated(dxlbomd_c))        nullify(dxlbomd_c)
     if(associated(qm_gho_info_c))    nullify(qm_gho_info_c)
     if(associated(qm_bond_c))        nullify(qm_bond_c)
     !!if(associated(qmmm_ewald_c))     nullify(qmmm_ewald_c)
     if(associated(map_grp_c))        nullify(map_grp_c)
     !!if(associated(dist_c))           nullify(dist_c)
     !!if(associated(grad_c))           nullify(grad_c)

     ! deallocate memories
     if(allocated(qm_control_r))     deallocate(qm_control_r)
     if(allocated(qm_main_r))        deallocate(qm_main_r)
     if(allocated(mm_main_r))        deallocate(mm_main_r)
     if(allocated(qm_scf_main_r))    deallocate(qm_scf_main_r)
     if(allocated(qm_param_r))       deallocate(qm_param_r)
     if(allocated(qm_scf_indx_r))    deallocate(qm_scf_indx_r)
     if(allocated(qm_scf_diis_r))    deallocate(qm_scf_diis_r)
     if(allocated(qm_fockmd_diis_r)) deallocate(qm_fockmd_diis_r)
     if(allocated(dxlbomd_r))        deallocate(dxlbomd_r)
     if(allocated(qm_gho_info_r))    deallocate(qm_gho_info_r)
     if(allocated(qm_bond_r))        deallocate(qm_bond_r)
     !!if(allocated(qmmm_ewald_r))     deallocate(qmmm_ewald_r)
     if(allocated(map_grp_r))        deallocate(map_grp_r)
     !!if(allocated(dist_r))           deallocate(dist_r)
     !!if(allocated(grad_r))           deallocate(grad_r)

     ! allocagte memories
     allocate(qm_control_r(nrepl))
     allocate(qm_main_r(nrepl))
     allocate(mm_main_r(nrepl))
     allocate(qm_scf_main_r(nrepl))
     allocate(qm_param_r(nrepl))
     allocate(qm_scf_indx_r(nrepl))
     allocate(qm_scf_diis_r(nrepl))
     allocate(qm_fockmd_diis_r(nrepl))
     allocate(dxlbomd_r(nrepl))
     allocate(qm_gho_info_r(nrepl))
     allocate(qm_bond_r(nrepl))
     !!allocate(qmmm_ewald_r(nrepl))
     allocate(map_grp_r(nrepl))
     !!allocate(dist_r(nrepl))
     !!allocate(grad_r(nrepl))

     ! pointers, as a default, num_qm_system=1, i.e., a single system
     qm_control_c    =>qm_control_r(1)
     qm_main_c       =>qm_main_r(1)
     mm_main_c       =>mm_main_r(1)
     qm_scf_main_c   =>qm_scf_main_r(1)
     qm_param_c      =>qm_param_r(1)
     qm_scf_indx_c   =>qm_scf_indx_r(1)
     qm_scf_diis_c   =>qm_scf_diis_r(1)
     qm_fockmd_diis_c=>qm_fockmd_diis_r(1)
     dxlbomd_c       =>dxlbomd_r(1)
     qm_gho_info_c   =>qm_gho_info_r(1)
     qm_bond_c       =>qm_bond_r(1)
     !!qmmm_ewald_c    =>qmmm_ewald_r(1)
     map_grp_c       =>map_grp_r(1)
     !!dist_c          =>dist_r(1)
     !!grad_c          =>grad_r(1)

     return
     end subroutine mndo97_memory_init

     subroutine allocate_deallocate_qmmm(qm_control_l,qm_main_l,mm_main_l,qallocate)
     !
     ! allocate/deallocate arrays for map between qm and the charmm main array.
     ! if qallocate == .true. , allocate memory
     !                  false., deallocate memory

     implicit none
     TYPE(qm_control):: qm_control_l
     TYPE(qm_main)   :: qm_main_l
     TYPE(mm_main)   :: mm_main_l
     logical :: qallocate

     integer :: ier=0

     ! deallocate if arrays are allocated.
     if(allocated(qm_control_l%qminb) ) deallocate(qm_control_l%qminb,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm','qminb')
     if(allocated(qm_control_l%mminb1) ) deallocate(qm_control_l%mminb1,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm','mminb1')
     if(allocated(qm_control_l%mminb2) ) deallocate(qm_control_l%mminb2,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm','mminb2')
     if(allocated(qm_control_l%cgqmmm)) deallocate(qm_control_l%cgqmmm,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm','cgqmmm')
     if(allocated(qm_control_l%igmsel)) deallocate(qm_control_l%igmsel,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm','igmsel')

     ! now, allocate memory, only if qallocate==.true.
     if(qallocate) then
        allocate(qm_control_l%qminb(qm_main_l%numat),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm','qminb')
        allocate(qm_control_l%mminb1(mm_main_l%natom),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm','mminb1')
        allocate(qm_control_l%mminb2(mm_main_l%natom),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm','mminb2')
        allocate(qm_control_l%cgqmmm(mm_main_l%natom),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm','cgqmmm')
        allocate(qm_control_l%igmsel(mm_main_l%natom),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qmmm','igmsel')
     end if
     return
     end subroutine allocate_deallocate_qmmm

     !==================================================================
     subroutine allocate_deallocate_qm(qm_main_l,qallocate)
     !
     ! allocate/deallocate qm atoms related arrays.
     ! if qallocate == .true. , allocate memory
     !                  false., deallocate memory

     implicit none
     TYPE(qm_main) :: qm_main_l
     logical :: qallocate

     integer :: ier=0

     ! deallocate if arrays are allocated.
     if(allocated(qm_main_l%nat)     ) deallocate(qm_main_l%nat,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm','nat')
     if(allocated(qm_main_l%qm_coord)) deallocate(qm_main_l%qm_coord,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm','qm_coord')
     if(allocated(qm_main_l%qm_grads)) deallocate(qm_main_l%qm_grads,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm','qm_grads')
     if(allocated(qm_main_l%qm_grads2)) deallocate(qm_main_l%qm_grads2,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm','qm_grads2')

     !
     if(allocated(qm_main_l%num_orbs)) deallocate(qm_main_l%num_orbs,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm','num_orbs')
     if(allocated(qm_main_l%nfirst)  ) deallocate(qm_main_l%nfirst,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm','nfirst')
     if(allocated(qm_main_l%nlast)   ) deallocate(qm_main_l%nlast,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm','nlast')

     ! now, allocate memory, only if qallocate==.true.
     if(qallocate) then
        ! for qm atoms
        allocate(qm_main_l%nat(qm_main_l%numat),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm','nat')
        allocate(qm_main_l%qm_coord(3,qm_main_l%numat),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm','qm_coord')
        allocate(qm_main_l%qm_grads(3,qm_main_l%numat),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm','qm_grads')
        allocate(qm_main_l%qm_grads2(3,qm_main_l%numat),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm','qm_grads2')

        !
        allocate(qm_main_l%num_orbs(qm_main_l%numat),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm','num_orbs')
        allocate(qm_main_l%nfirst(qm_main_l%numat),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm','nfirst')
        allocate(qm_main_l%nlast(qm_main_l%numat),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm','nlast')
     end if
     return
     end subroutine allocate_deallocate_qm

     !==================================================================
     subroutine allocate_deallocate_mm(qm_main_l,mm_main_l,qallocate)
     !
     ! allocate/deallocate mm atoms related arrays.
     ! if qallocate == .true. , allocate memory
     !                  false., deallocate memory

     implicit none
     TYPE(qm_main) :: qm_main_l
     TYPE(mm_main) :: mm_main_l
     logical :: qallocate

     integer :: ier=0

     ! deallocate if arrays are allocated.
     if(allocated(mm_main_l%qm_mm_pair_list)) deallocate(mm_main_l%qm_mm_pair_list,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_mm','qm_mm_pair_list')
     if(allocated(mm_main_l%mm_coord))  deallocate(mm_main_l%mm_coord,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_mm','mm_coord')
     if(allocated(mm_main_l%mm_grads))  deallocate(mm_main_l%mm_grads,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_mm','mm_grads')
     if(allocated(mm_main_l%mm_grads2))  deallocate(mm_main_l%mm_grads2,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_mm','mm_grads2')
     if(allocated(mm_main_l%mm_chrgs))  deallocate(mm_main_l%mm_chrgs,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_mm','mm_chrgs')
     !if(allocated(mm_main_l%EMPOT)   )  deallocate(mm_main_l%EMPOT,stat=ier)
     !   if(ier.ne.0) call Aass(0,'allocate_deallocate_mm','EMPOT')
     !if(allocated(mm_main_l%ESLF)    )  deallocate(mm_main_l%ESLF,stat=ier)
     !   if(ier.ne.0) call Aass(0,'allocate_deallocate_mm','ESLF')
     if(allocated(mm_main_l%qm_charges)) deallocate(mm_main_l%qm_charges,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_mm','qm_charges')

     ! now, allocate memory, only if qallocate==.true.
     if(qallocate) then
        ! for mm atoms. 
        ! note: allocate by the size "natom," since then, it do not need any
        ! allocation/deallocation leter.
        allocate(mm_main_l%qm_mm_pair_list(mm_main_l%natom),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_mm','qm_mm_pair_list')
        allocate(mm_main_l%mm_coord(3,mm_main_l%natom),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_mm','mm_coord')
        allocate(mm_main_l%mm_grads(3,mm_main_l%natom),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_mm','mm_grads')
        allocate(mm_main_l%mm_grads2(3,mm_main_l%natom),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_mm','mm_grads2')
        allocate(mm_main_l%mm_chrgs(mm_main_l%natom),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_mm','mm_chrgs')

        allocate(mm_main_l%qm_charges(qm_main_l%numat),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_mm','qm_charges')
     end if
     return
     end subroutine allocate_deallocate_mm


     !==================================================================
     subroutine determine_qm_scf_arrray_size(qm_main_l,qm_scf_main_l,qallocate)
     !
     ! determine the array size to allocate for scf calculations.
     ! also allocate/deallocate indx array; if qallocate == .true. , allocate indx
     !                                                      .false., deallocate it.
     implicit none
     TYPE(qm_main)     :: qm_main_l
     TYPE(qm_scf_main) :: qm_scf_main_l
     logical :: qallocate

     integer :: i,j,iorbs,jorbs,iw,LM6,LMX,ijpair
     integer :: ier=0

     ! first, define array sizes
     qm_scf_main_l%dim_numat        = qm_main_l%numat
     LMX                            = 9*qm_scf_main_l%dim_numat

     ! indx is done here, since it is used also in the present routine.
     if(allocated(qm_scf_main_l%indx)) deallocate(qm_scf_main_l%indx,stat=ier)
        if(ier.ne.0) call Aass(0,'determine_qm_scf_arrray_size','indx')

     ! now, allocate memory, only if qallocate==.true.
     if(qallocate) then
        allocate(qm_scf_main_l%indx(LMX),stat=ier)
           if(ier.ne.0) call Aass(1,'determine_qm_scf_arrray_size','indx')

        ! do construct indx
        call construct_index(qm_scf_main_l%indx,LMX)
     else
        qm_scf_main_l%dim_numat        = 0
        qm_scf_main_l%dim_norbs        = 0
        return
     end if

     ! other array sizes.
     qm_scf_main_l%dim_norbs        = qm_main_l%norbs
     qm_scf_main_l%dim_norbs2       = qm_main_l%norbs*qm_main_l%norbs
     qm_scf_main_l%dim_linear_norbs = (qm_main_l%norbs*(qm_main_l%norbs+1))/2
     qm_scf_main_l%dim_scratch      = MAX(qm_main_l%norbs*qm_main_l%norbs,8*qm_main_l%norbs) 
     qm_scf_main_l%dim_iscratch     = qm_main_l%norbs

     ! find number of unique one-center AO pairs and two-center two-electron integrals
     ijpair=0
     do i=1,qm_main_l%numat
        iorbs=qm_main_l%num_orbs(i)
        ijpair=ijpair+qm_scf_main_l%indx(iorbs)+iorbs
     end do
     !
     qm_scf_main_l%dim_linear_fock  = ijpair
     qm_scf_main_l%dim_linear_fock2 = ijpair*ijpair  ! two

     !
     ijpair=0
     do i=2,qm_main_l%numat
        iorbs=qm_main_l%num_orbs(i)
        iw   =qm_scf_main_l%indx(iorbs)+iorbs
        do j=1,i-1
           jorbs=qm_main_l%num_orbs(j)
           ijpair=ijpair+iw*(qm_scf_main_l%indx(jorbs)+jorbs)
        end do
     end do
     qm_scf_main_l%dim_2C_2E_integral = ijpair
     ! 
     return

     contains

        subroutine construct_index(INDX,NORBS)
        !
        !  for the lower triangle matrix Z' of matrix Z(n,n),
        !  the last elements of Z' for (i-1,i-1) element of matrix Z
        !  is i*(i-1)/2.  Use this to point the starting point (or 
        !  end-point) of previous elements.
        !  
        implicit none
        integer :: indx(*),norbs
        integer :: i
        !
        do i=1,norbs
           indx(i)=i*(i-1)/2
        end do
        return
        end subroutine construct_index

     end subroutine determine_qm_scf_arrray_size


     !==================================================================
     subroutine allocate_deallocate_qm_scf(qm_main_l,qm_scf_main_l,qallocate)
     !
     ! allocate/deallocate scf related arrays.
     ! if qallocate == .true. , allocate memory
     !                  false., deallocate memory

     implicit none
     TYPE(qm_main)     :: qm_main_l
     TYPE(qm_scf_main) :: qm_scf_main_l
     logical :: qallocate

     integer :: LMI,LME,LMX,LMX6,dim_numat,dim_norbs,dim_norbs2,dim_linear_norbs, &
                dim_linear_fock,dim_linear_fock2,dim_scratch,dim_iscratch
     integer :: ier=0

     ! first, define array sizes (determined in determine_qm_scf_arrray_size).
     dim_numat       = qm_scf_main_l%dim_numat
     dim_norbs       = qm_scf_main_l%dim_norbs
     dim_norbs2      = qm_scf_main_l%dim_norbs2
     dim_linear_norbs= qm_scf_main_l%dim_linear_norbs
     dim_linear_fock = qm_scf_main_l%dim_linear_fock
     dim_linear_fock2= qm_scf_main_l%dim_linear_fock2
     dim_scratch     = qm_scf_main_l%dim_scratch
     dim_iscratch    = qm_scf_main_l%dim_iscratch
     LMI             = 45*qm_scf_main_l%dim_numat
     LME             = 81*qm_scf_main_l%dim_numat
     LMX             = 9*qm_scf_main_l%dim_numat
     LMX6            = 6*LMX

     ! deallocate if arrays are allocated.

     ! for integer arrays:
     if(allocated(qm_scf_main_l%NW))   deallocate(qm_scf_main_l%NW,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','NW')

     if(allocated(qm_scf_main_l%iwork)) deallocate(qm_scf_main_l%iwork,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','iwork')
     
     ! for real arrays:
     if(allocated(qm_scf_main_l%H))  deallocate(qm_scf_main_l%H,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','H')
     if(allocated(qm_scf_main_l%W))  deallocate(qm_scf_main_l%W,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','W')
     if(associated(qm_scf_main_l%FA)) deallocate(qm_scf_main_l%FA,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','FA')
     if(associated(qm_scf_main_l%FB)) deallocate(qm_scf_main_l%FB,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','FB')
     if(allocated(qm_scf_main_l%FAwork)) deallocate(qm_scf_main_l%FAwork,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','FAwork')
     if(allocated(qm_scf_main_l%FBwork)) deallocate(qm_scf_main_l%FBwork,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','FBwork')
     if(associated(qm_scf_main_l%CA)) deallocate(qm_scf_main_l%CA,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','CA')
     if(associated(qm_scf_main_l%CB)) deallocate(qm_scf_main_l%CB,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','CB')
     if(associated(qm_scf_main_l%PA)) deallocate(qm_scf_main_l%PA,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','PA')
     if(associated(qm_scf_main_l%PB)) deallocate(qm_scf_main_l%PB,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','PB')
     if(allocated(qm_scf_main_l%PAwork)) deallocate(qm_scf_main_l%PAwork,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','PAwork')
     if(allocated(qm_scf_main_l%PBwork)) deallocate(qm_scf_main_l%PBwork,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','PBwork')
     if(associated(qm_scf_main_l%DA)) deallocate(qm_scf_main_l%DA,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','DA')
     if(associated(qm_scf_main_l%DB)) deallocate(qm_scf_main_l%DB,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','DB')
     if(associated(qm_scf_main_l%EA)) deallocate(qm_scf_main_l%EA,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','EA')
     if(associated(qm_scf_main_l%EB)) deallocate(qm_scf_main_l%EB,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','EB')
     if(allocated(qm_scf_main_l%Q))  deallocate(qm_scf_main_l%Q,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','Q')

     ! for precomputation:
     if(allocated(qm_scf_main_l%H_1cent)) deallocate(qm_scf_main_l%H_1cent,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','H_1cent')
     if(allocated(qm_scf_main_l%imap_h)) deallocate(qm_scf_main_l%imap_h,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','imap_h')

     ! now, allocate memory, only if qallocate==.true.
     if(qallocate) then
        ! for integer arrays:
        allocate(qm_scf_main_l%NW(dim_numat),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','NW')

        allocate(qm_scf_main_l%iwork(LMX6),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','iwork')

        ! for real arrays:
        allocate(qm_scf_main_l%H(dim_linear_norbs),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','H')
        allocate(qm_scf_main_l%W(dim_linear_fock2),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','W')
        allocate(qm_scf_main_l%Q(dim_scratch),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','Q')

        ! for rhf or uhf-alpha 
        allocate(qm_scf_main_l%FA(dim_linear_norbs),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','FA')
        allocate(qm_scf_main_l%FAwork(dim_norbs,dim_norbs),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','FAwork')
        allocate(qm_scf_main_l%CA(dim_norbs,dim_norbs),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','CA')
        allocate(qm_scf_main_l%PA(dim_linear_norbs),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','PA')
        allocate(qm_scf_main_l%PAwork(dim_norbs,dim_norbs),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','PAwork')
        allocate(qm_scf_main_l%DA(dim_linear_norbs),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','DA')
        allocate(qm_scf_main_l%EA(dim_norbs),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','EA')

        ! for uhf-beta
        if(qm_main_l%uhf) then
           allocate(qm_scf_main_l%FB(dim_linear_norbs),stat=ier)
              if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','FB')
           allocate(qm_scf_main_l%FBwork(dim_norbs,dim_norbs),stat=ier)
              if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','FBwork')
           allocate(qm_scf_main_l%CB(dim_norbs,dim_norbs),stat=ier)
              if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','CB')
           allocate(qm_scf_main_l%PB(dim_linear_norbs),stat=ier)
              if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','PB')
           allocate(qm_scf_main_l%PBwork(dim_norbs,dim_norbs),stat=ier)
              if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','PBwork')
           allocate(qm_scf_main_l%DB(dim_linear_norbs),stat=ier)
              if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','DB')
           allocate(qm_scf_main_l%EB(dim_norbs),stat=ier)
              if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','EB')
        end if

        ! for precomputation
        allocate(qm_scf_main_l%H_1cent(dim_norbs),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','H_1cent')
        allocate(qm_scf_main_l%imap_h(dim_norbs),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_scf','imap_h')
     end if
     return
     end subroutine allocate_deallocate_qm_scf


     subroutine allocate_deallocate_qm_param(qm_param_l,numat,iqm_mode)
     !
     ! if allocated, deallocate all memories, and allocate new memories for qm parameters
     implicit none
     TYPE(qm_param) :: qm_param_l
     integer :: numat,iqm_mode
     integer :: ier=0

     ! deallocate memory
     if(allocated(qm_param_l%q_atom_specific)) deallocate(qm_param_l%q_atom_specific,stat=ier)

     if(allocated(qm_param_l%ni_local))    deallocate(qm_param_l%ni_local,stat=ier)
     if(allocated(qm_param_l%iorbs_local)) deallocate(qm_param_l%iorbs_local,stat=ier)
     if(allocated(qm_param_l%ia_local))    deallocate(qm_param_l%ia_local,stat=ier)
     if(allocated(qm_param_l%ib_local))    deallocate(qm_param_l%ib_local,stat=ier)
     if(allocated(qm_param_l%ip_local))    deallocate(qm_param_l%ip_local,stat=ier)
     if(allocated(qm_param_l%is_local))    deallocate(qm_param_l%is_local,stat=ier)
     if(allocated(qm_param_l%iw_local))    deallocate(qm_param_l%iw_local,stat=ier)
     if(allocated(qm_param_l%Jmax_local))  deallocate(qm_param_l%Jmax_local,stat=ier)
     if(allocated(qm_param_l%int_ij))      deallocate(qm_param_l%int_ij,stat=ier)
     if(allocated(qm_param_l%int_kl))      deallocate(qm_param_l%int_kl,stat=ier)
     if(allocated(qm_param_l%w_save))      deallocate(qm_param_l%w_save,stat=ier)

     if(allocated(qm_param_l%IMPAR))  deallocate(qm_param_l%IMPAR,stat=ier)
     if(allocated(qm_param_l%IM1D))   deallocate(qm_param_l%IM1D,stat=ier)
     if(allocated(qm_param_l%LORBS))  deallocate(qm_param_l%LORBS,stat=ier)
     if(allocated(qm_param_l%IOS))    deallocate(qm_param_l%IOS,stat=ier)
     if(allocated(qm_param_l%IOP))    deallocate(qm_param_l%IOP,stat=ier)
     if(allocated(qm_param_l%IOD))    deallocate(qm_param_l%IOD,stat=ier)
     if(allocated(qm_param_l%III))    deallocate(qm_param_l%III,stat=ier)
     if(allocated(qm_param_l%IIID))   deallocate(qm_param_l%IIID,stat=ier)
     if(allocated(qm_param_l%GNN))    deallocate(qm_param_l%GNN,stat=ier)
     if(allocated(qm_param_l%core))   deallocate(qm_param_l%core,stat=ier)
     if(allocated(qm_param_l%EHEAT))  deallocate(qm_param_l%EHEAT,stat=ier)
     if(allocated(qm_param_l%EISOL))  deallocate(qm_param_l%EISOL,stat=ier)
     if(allocated(qm_param_l%USS))    deallocate(qm_param_l%USS,stat=ier)
     if(allocated(qm_param_l%UPP))    deallocate(qm_param_l%UPP,stat=ier)
     if(allocated(qm_param_l%ZS))     deallocate(qm_param_l%ZS,stat=ier)
     if(allocated(qm_param_l%ZP))     deallocate(qm_param_l%ZP,stat=ier)
     if(allocated(qm_param_l%BETAS))  deallocate(qm_param_l%BETAS,stat=ier)
     if(allocated(qm_param_l%BETAP))  deallocate(qm_param_l%BETAP,stat=ier)
     if(allocated(qm_param_l%ALP))    deallocate(qm_param_l%ALP,stat=ier)
     if(allocated(qm_param_l%GSS))    deallocate(qm_param_l%GSS,stat=ier)
     if(allocated(qm_param_l%GSP))    deallocate(qm_param_l%GSP,stat=ier)
     if(allocated(qm_param_l%GPP))    deallocate(qm_param_l%GPP,stat=ier)
     if(allocated(qm_param_l%GP2))    deallocate(qm_param_l%GP2,stat=ier)
     if(allocated(qm_param_l%HSP))    deallocate(qm_param_l%HSP,stat=ier)
     if(allocated(qm_param_l%HPP))    deallocate(qm_param_l%HPP,stat=ier)
     if(allocated(qm_param_l%QQ))     deallocate(qm_param_l%QQ,stat=ier)
     if(allocated(qm_param_l%AM))     deallocate(qm_param_l%AM,stat=ier)
     if(allocated(qm_param_l%AD))     deallocate(qm_param_l%AD,stat=ier)
     if(allocated(qm_param_l%AQ))     deallocate(qm_param_l%AQ,stat=ier)
     if(allocated(qm_param_l%DD))     deallocate(qm_param_l%DD,stat=ier)
     if(allocated(qm_param_l%PO))     deallocate(qm_param_l%PO,stat=ier)
     if(allocated(qm_param_l%GUESS1)) deallocate(qm_param_l%GUESS1,stat=ier)
     if(allocated(qm_param_l%GUESS2)) deallocate(qm_param_l%GUESS2,stat=ier)
     if(allocated(qm_param_l%GUESS3)) deallocate(qm_param_l%GUESS3,stat=ier)
     if(allocated(qm_param_l%UDD))    deallocate(qm_param_l%UDD,stat=ier)
     if(allocated(qm_param_l%ZD))     deallocate(qm_param_l%ZD,stat=ier)
     if(allocated(qm_param_l%BETAD))  deallocate(qm_param_l%BETAD,stat=ier)
     if(allocated(qm_param_l%ZSN))    deallocate(qm_param_l%ZSN,stat=ier)
     if(allocated(qm_param_l%ZPN))    deallocate(qm_param_l%ZPN,stat=ier)
     if(allocated(qm_param_l%ZDN))    deallocate(qm_param_l%ZDN,stat=ier)
     if(allocated(qm_param_l%F0SD))   deallocate(qm_param_l%F0SD,stat=ier)
     if(allocated(qm_param_l%G2SD))   deallocate(qm_param_l%G2SD,stat=ier)
     if(allocated(qm_param_l%F0DD))   deallocate(qm_param_l%F0DD,stat=ier)
     if(allocated(qm_param_l%F2DD))   deallocate(qm_param_l%F2DD,stat=ier)
     if(allocated(qm_param_l%F4DD))   deallocate(qm_param_l%F4DD,stat=ier)
     if(allocated(qm_param_l%F0PD))   deallocate(qm_param_l%F0PD,stat=ier)
     if(allocated(qm_param_l%F2PD))   deallocate(qm_param_l%F2PD,stat=ier)
     if(allocated(qm_param_l%G1PD))   deallocate(qm_param_l%G1PD,stat=ier)
     if(allocated(qm_param_l%G3PD))   deallocate(qm_param_l%G3PD,stat=ier)
     if(allocated(qm_param_l%IF0SD))  deallocate(qm_param_l%IF0SD,stat=ier)
     if(allocated(qm_param_l%IG2SD))  deallocate(qm_param_l%IG2SD,stat=ier)
     if(allocated(qm_param_l%REPD))   deallocate(qm_param_l%REPD,stat=ier)
     if(allocated(qm_param_l%MALPB))  deallocate(qm_param_l%MALPB,stat=ier)
     if(allocated(qm_param_l%ALPB))   deallocate(qm_param_l%ALPB,stat=ier)
     if(allocated(qm_param_l%DELTA))  deallocate(qm_param_l%DELTA,stat=ier)
     if(allocated(qm_param_l%OMEGA))  deallocate(qm_param_l%OMEGA,stat=ier)

     if(allocated(qm_param_l%GSS_local))  deallocate(qm_param_l%GSS_local,stat=ier)
     if(allocated(qm_param_l%GSP_local))  deallocate(qm_param_l%GSP_local,stat=ier)
     if(allocated(qm_param_l%GPP_local))  deallocate(qm_param_l%GPP_local,stat=ier)
     if(allocated(qm_param_l%GP2_local))  deallocate(qm_param_l%GP2_local,stat=ier)
     if(allocated(qm_param_l%HSP_local))  deallocate(qm_param_l%HSP_local,stat=ier)
     if(allocated(qm_param_l%HPP_local))  deallocate(qm_param_l%HPP_local,stat=ier)


     ! allocate memory
     allocate(qm_param_l%q_atom_specific(numat),stat=ier)

     allocate(qm_param_l%ni_local(numat),stat=ier)
     allocate(qm_param_l%iorbs_local(numat),stat=ier)
     allocate(qm_param_l%ia_local(numat),stat=ier)
     allocate(qm_param_l%ib_local(numat),stat=ier)
     allocate(qm_param_l%ip_local(numat),stat=ier)
     allocate(qm_param_l%is_local(numat),stat=ier)
     allocate(qm_param_l%iw_local(numat),stat=ier)
     allocate(qm_param_l%Jmax_local(numat),stat=ier)
     allocate(qm_param_l%w_save(243,numat),stat=ier)
     allocate(qm_param_l%int_ij(243),stat=ier)  
     allocate(qm_param_l%int_kl(243),stat=ier)

     allocate(qm_param_l%IMPAR(numat),stat=ier)
     allocate(qm_param_l%IM1D(numat),stat=ier)
     allocate(qm_param_l%LORBS(numat),stat=ier)
     allocate(qm_param_l%IOS(numat),stat=ier)
     allocate(qm_param_l%IOP(numat),stat=ier)
     allocate(qm_param_l%IOD(numat),stat=ier)
     allocate(qm_param_l%III(numat),stat=ier)
     allocate(qm_param_l%IIID(numat),stat=ier)
     allocate(qm_param_l%GNN(numat),stat=ier)
     allocate(qm_param_l%core(numat),stat=ier)
     allocate(qm_param_l%EHEAT(numat),stat=ier)
     allocate(qm_param_l%EISOL(numat),stat=ier)
     allocate(qm_param_l%USS(numat),stat=ier)
     allocate(qm_param_l%UPP(numat),stat=ier)
     allocate(qm_param_l%ZS(numat),stat=ier)
     allocate(qm_param_l%ZP(numat),stat=ier)
     allocate(qm_param_l%BETAS(numat),stat=ier)
     allocate(qm_param_l%BETAP(numat),stat=ier)
     allocate(qm_param_l%ALP(numat),stat=ier)
     allocate(qm_param_l%GSS(numat),stat=ier)
     allocate(qm_param_l%GSP(numat),stat=ier)
     allocate(qm_param_l%GPP(numat),stat=ier)
     allocate(qm_param_l%GP2(numat),stat=ier)
     allocate(qm_param_l%HSP(numat),stat=ier)
     allocate(qm_param_l%HPP(numat),stat=ier)
     allocate(qm_param_l%QQ(numat),stat=ier)
     allocate(qm_param_l%AM(numat),stat=ier)
     allocate(qm_param_l%AD(numat),stat=ier)
     allocate(qm_param_l%AQ(numat),stat=ier)
     allocate(qm_param_l%DD(6,0:numat),stat=ier)
     allocate(qm_param_l%PO(9,0:numat),stat=ier)
     allocate(qm_param_l%DELTA(numat),stat=ier)
     allocate(qm_param_l%OMEGA(numat),stat=ier)

     ! for some local variables
     allocate(qm_param_l%GSS_local(numat),stat=ier)
     allocate(qm_param_l%GSP_local(numat),stat=ier)
     allocate(qm_param_l%GPP_local(numat),stat=ier)
     allocate(qm_param_l%GP2_local(numat),stat=ier)
     allocate(qm_param_l%HSP_local(numat),stat=ier)
     allocate(qm_param_l%HPP_local(numat),stat=ier)

     !if(.not. (iqm_mode.eq.1 .or. iqm_mode.eq.5)) then
        ! Gaussian core parameters
        allocate(qm_param_l%GUESS1(4,numat),stat=ier)
        allocate(qm_param_l%GUESS2(4,numat),stat=ier)
        allocate(qm_param_l%GUESS3(4,numat),stat=ier)
     !end if

     !if((iqm_mode.eq.4) .or. (iqm_mode.eq.5)) then
        ! d-orbital parameters
        allocate(qm_param_l%UDD(numat),stat=ier)
        allocate(qm_param_l%ZD(numat),stat=ier)
        allocate(qm_param_l%BETAD(numat),stat=ier)
        allocate(qm_param_l%ZSN(numat),stat=ier)
        allocate(qm_param_l%ZPN(numat),stat=ier)
        allocate(qm_param_l%ZDN(numat),stat=ier)
        allocate(qm_param_l%F0SD(numat),stat=ier)
        allocate(qm_param_l%G2SD(numat),stat=ier)
        allocate(qm_param_l%F0DD(numat),stat=ier)
        allocate(qm_param_l%F2DD(numat),stat=ier)
        allocate(qm_param_l%F4DD(numat),stat=ier)
        allocate(qm_param_l%F0PD(numat),stat=ier)
        allocate(qm_param_l%F2PD(numat),stat=ier)
        allocate(qm_param_l%G1PD(numat),stat=ier)
        allocate(qm_param_l%G3PD(numat),stat=ier)
        allocate(qm_param_l%IF0SD(numat),stat=ier)
        allocate(qm_param_l%IG2SD(numat),stat=ier)
        allocate(qm_param_l%REPD(52,numat),stat=ier)
     !end if

     !if(iqm_mode.eq.5) then
        ! mndo/d specific parameters
        allocate(qm_param_l%MALPB(numat),stat=ier)
        allocate(qm_param_l%ALPB(numat,numat),stat=ier)
     !end if

     return
     end subroutine allocate_deallocate_qm_param


     subroutine allocate_local_param_memories(qm_param_a,natm_types,iqm_mode,qalocation)
     use number, only: zero,one 
     use qm1_constant, only: minbig
     implicit none
     type(qm_param):: qm_param_a
     integer :: natm_types,iqm_mode
     logical :: qalocation

     integer :: ier=0

     ! allocate memories...
     if(qalocation) then
        ! allocate memories
        allocate(qm_param_a%q_atom_specific(natm_types),stat=ier)
        allocate(qm_param_a%ni_local(natm_types),stat=ier)
        allocate(qm_param_a%IMPAR(natm_types),stat=ier)
        allocate(qm_param_a%GNN(natm_types),stat=ier)
        allocate(qm_param_a%USS(natm_types),stat=ier)
        allocate(qm_param_a%UPP(natm_types),stat=ier)
        allocate(qm_param_a%ZS(natm_types),stat=ier)
        allocate(qm_param_a%ZP(natm_types),stat=ier)
        allocate(qm_param_a%BETAS(natm_types),stat=ier)
        allocate(qm_param_a%BETAP(natm_types),stat=ier)
        allocate(qm_param_a%ALP(natm_types),stat=ier)
        allocate(qm_param_a%GSS(natm_types),stat=ier)
        allocate(qm_param_a%GSP(natm_types),stat=ier)
        allocate(qm_param_a%GPP(natm_types),stat=ier)
        allocate(qm_param_a%GP2(natm_types),stat=ier)
        allocate(qm_param_a%HSP(natm_types),stat=ier)
        allocate(qm_param_a%PO(9,natm_types),stat=ier)
        !!allocate(qm_param_a%HPP(natm_types),stat=ier)

        ! Gaussian core parameters
        !if(.not. (iqm_mode == 1 .or. iqm_mode == 5)) then
           allocate(qm_param_a%GUESS1(4,natm_types),stat=ier)
           allocate(qm_param_a%GUESS2(4,natm_types),stat=ier)
           allocate(qm_param_a%GUESS3(4,natm_types),stat=ier)
        !end if
        ! d-orbital parameters
        !if((iqm_mode == 4) .or. (iqm_mode == 5)) then
           allocate(qm_param_a%UDD(natm_types),stat=ier)
           allocate(qm_param_a%ZD(natm_types),stat=ier)
           allocate(qm_param_a%BETAD(natm_types),stat=ier)
           allocate(qm_param_a%ZSN(natm_types),stat=ier)
           allocate(qm_param_a%ZPN(natm_types),stat=ier)
           allocate(qm_param_a%ZDN(natm_types),stat=ier)
        !end if

        ! initialize parameters 
        qm_param_a%q_atom_specific(1:natm_types) =.false.
        qm_param_a%ni_local(1:natm_types)= 0
        qm_param_a%IMPAR(1:natm_types)   = 0
        qm_param_a%GNN(1:natm_types)     = minbig
        qm_param_a%USS(1:natm_types)     = minbig
        qm_param_a%UPP(1:natm_types)     = minbig
        qm_param_a%ZS(1:natm_types)      = minbig
        qm_param_a%ZP(1:natm_types)      = minbig
        qm_param_a%BETAS(1:natm_types)   = minbig
        qm_param_a%BETAP(1:natm_types)   = minbig
        qm_param_a%ALP(1:natm_types)     = minbig
        qm_param_a%GSS(1:natm_types)     = minbig
        qm_param_a%GSP(1:natm_types)     = minbig
        qm_param_a%GPP(1:natm_types)     = minbig
        qm_param_a%GP2(1:natm_types)     = minbig
        qm_param_a%HSP(1:natm_types)     = minbig
        qm_param_a%PO(1:9,1:natm_types)  = minbig
        !if(.not. (iqm_mode == 1 .or. iqm_mode == 5)) then
           qm_param_a%GUESS1(1:4,1:natm_types)= minbig
           qm_param_a%GUESS2(1:4,1:natm_types)= minbig
           qm_param_a%GUESS3(1:4,1:natm_types)= minbig
        !end if
        !if((iqm_mode == 4) .or. (iqm_mode == 5)) then
           qm_param_a%UDD(1:natm_types)  = minbig
           qm_param_a%ZD(1:natm_types)   = minbig
           qm_param_a%BETAD(1:natm_types)= minbig
           qm_param_a%ZSN(1:natm_types)  = minbig
           qm_param_a%ZPN(1:natm_types)  = minbig
           qm_param_a%ZDN(1:natm_types)  = minbig
        !end if 
     else
        ! deallocate memories
        if(allocated(qm_param_a%q_atom_specific)) deallocate(qm_param_a%q_atom_specific,stat=ier)
        if(allocated(qm_param_a%ni_local))        deallocate(qm_param_a%ni_local,stat=ier)
        if(allocated(qm_param_a%IMPAR))           deallocate(qm_param_a%IMPAR,stat=ier)
        if(allocated(qm_param_a%GNN))             deallocate(qm_param_a%GNN,stat=ier)
        if(allocated(qm_param_a%USS))             deallocate(qm_param_a%USS,stat=ier)
        if(allocated(qm_param_a%UPP))             deallocate(qm_param_a%UPP,stat=ier)
        if(allocated(qm_param_a%ZS))              deallocate(qm_param_a%ZS,stat=ier)
        if(allocated(qm_param_a%ZP))              deallocate(qm_param_a%ZP,stat=ier)
        if(allocated(qm_param_a%BETAS))           deallocate(qm_param_a%BETAS,stat=ier)
        if(allocated(qm_param_a%BETAP))           deallocate(qm_param_a%BETAP,stat=ier)
        if(allocated(qm_param_a%ALP))             deallocate(qm_param_a%ALP,stat=ier)
        if(allocated(qm_param_a%GSS))             deallocate(qm_param_a%GSS,stat=ier)
        if(allocated(qm_param_a%GSP))             deallocate(qm_param_a%GSP,stat=ier)
        if(allocated(qm_param_a%GPP))             deallocate(qm_param_a%GPP,stat=ier)
        if(allocated(qm_param_a%GP2))             deallocate(qm_param_a%GP2,stat=ier)
        if(allocated(qm_param_a%HSP))             deallocate(qm_param_a%HSP,stat=ier)
        if(allocated(qm_param_a%PO))              deallocate(qm_param_a%PO, stat=ier)
        !!if(allocate(qm_param_a%HPP)) deallocate(qm_param_a%HPP,stat=ier)

        ! Gaussian core parameters
        !if(.not. (iqm_mode == 1 .or. iqm_mode == 5)) then
        if(allocated(qm_param_a%GUESS1)) deallocate(qm_param_a%GUESS1,stat=ier)
        if(allocated(qm_param_a%GUESS2)) deallocate(qm_param_a%GUESS2,stat=ier)
        if(allocated(qm_param_a%GUESS3)) deallocate(qm_param_a%GUESS3,stat=ier)
        !end if
        ! d-orbital parameters
        !if((iqm_mode == 4) .or. (iqm_mode == 5)) then
        if(allocated(qm_param_a%UDD))    deallocate(qm_param_a%UDD,stat=ier)
        if(allocated(qm_param_a%ZD))     deallocate(qm_param_a%ZD,stat=ier)
        if(allocated(qm_param_a%BETAD))  deallocate(qm_param_a%BETAD,stat=ier)
        if(allocated(qm_param_a%ZSN))    deallocate(qm_param_a%ZSN,stat=ier)
        if(allocated(qm_param_a%ZPN))    deallocate(qm_param_a%ZPN,stat=ier)
        if(allocated(qm_param_a%ZDN))    deallocate(qm_param_a%ZDN,stat=ier)
        !end if
     end if
     return
     end subroutine allocate_local_param_memories

     subroutine allocate_deallocate_qm_diis(qm_scf_main_l,qm_scf_diis_l,qm_fockmd_diis_l, &
                                            qdiis,q_fockmd,imax_fdiss,uhf,qallocate)
     !
     ! allocate/deallocate Diis related arrays.
     ! if qallocate == .true. , allocate memory
     !                  false., deallocate memory
#if KEY_PARALLEL==1
     use parallel
#endif

     implicit none
     TYPE(qm_scf_main)    :: qm_scf_main_l
     TYPE(qm_scf_diis)    :: qm_scf_diis_l
     TYPE(qm_fockmd_diis) :: qm_fockmd_diis_l
     logical :: qdiis,q_fockmd,uhf,qallocate
     integer :: imax_fdiss

     integer :: LMX,LMX6,dim_norbs,dim_linear_norbs,MXDIIS,MX1P
     integer :: ier=0
     integer :: mstart,mstop,msize,mxfdiis

     ! first, define array sizes (determined in determine_qm_scf_arrray_size)
     dim_norbs       = qm_scf_main_l%dim_norbs
     dim_linear_norbs= qm_scf_main_l%dim_linear_norbs
     LMX             = 9*qm_scf_main_l%dim_numat
     LMX6            = 6*LMX
     mxdiis          = qm_scf_diis_l%MXDIIS
     mx1p            = qm_scf_diis_l%MX1P

     ! deallocate if arrays are allocated.
     if(allocated(qm_scf_diis_l%FDA))   deallocate(qm_scf_diis_l%FDA,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_diis','FDA')
     if(allocated(qm_scf_diis_l%FDB))   deallocate(qm_scf_diis_l%FDB,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_diis','FDB')
     if(allocated(qm_scf_diis_l%Ediis)) deallocate(qm_scf_diis_l%Ediis,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_diis','Ediis')
     if(allocated(qm_scf_diis_l%Bdiis)) deallocate(qm_scf_diis_l%Bdiis,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_diis','Bdiis')
     if(allocated(qm_scf_diis_l%Adiis)) deallocate(qm_scf_diis_l%Adiis,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_diis','Adiix')
     if(allocated(qm_scf_diis_l%Xdiis)) deallocate(qm_scf_diis_l%Xdiis,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_diis','Xdiis')
     if(allocated(qm_scf_diis_l%iwork_diis)) deallocate(qm_scf_diis_l%iwork_diis,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_diis','iwork_diis')
#if KEY_PARALLEL==1
     if(allocated(qm_scf_diis_l%q_ij_pair))  deallocate(qm_scf_diis_l%q_ij_pair,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_diis','q_ij_pair')
     if(allocated(qm_scf_diis_l%bdiis_local)) deallocate(qm_scf_diis_l%bdiis_local,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_diis','bdiis_local')
#endif
     !
     if(allocated(qm_fockmd_diis_l%FA_sv)) deallocate(qm_fockmd_diis_l%FA_sv,stat=ier)
     if(allocated(qm_fockmd_diis_l%FDA))   deallocate(qm_fockmd_diis_l%FDA,stat=ier)

     ! now, allocate memory, only if qallocate==.true.
     if(qallocate.and.qdiis) then
        allocate(qm_scf_diis_l%FDA(dim_linear_norbs,mxdiis+1),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_diis','FDA')
        if(uhf) then
           allocate(qm_scf_diis_l%FDB(dim_linear_norbs,mxdiis+1),stat=ier)
              if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_diis','FDB')
        end if
        allocate(qm_scf_diis_l%Ediis(dim_linear_norbs,mxdiis+1),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_diis','Ediis')
        allocate(qm_scf_diis_l%Bdiis(mx1p),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_diis','Bdiis')
        allocate(qm_scf_diis_l%Adiis(mx1p),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_diis','Adiix')
        allocate(qm_scf_diis_l%Xdiis(mxdiis+1),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_diis','Xdiis')
        allocate(qm_scf_diis_l%iwork_diis(LMX6),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_diis','iwork_diis')

#if KEY_PARALLEL==1
        allocate(qm_scf_diis_l%bdiis_local(mxdiis),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_diis','bdiis_local')
        if(qm_gho_info_c%q_gho) then
           allocate(qm_scf_diis_l%q_ij_pair(qm_gho_info_c%norbhb),stat=ier)
        else
           allocate(qm_scf_diis_l%q_ij_pair(qm_main_c%norbs),stat=ier)
        end if
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_diis','q_ij_pair')
#endif
     end if

     ! now, allocate memory only if qallocate==.true.
     if(qallocate.and.q_fockmd) then
        qm_fockmd_diis_l%mxfdiis= imax_fdiss
        mxfdiis= imax_fdiss
        mstart = 1
        mstop  = dim_linear_norbs
#if KEY_PARALLEL==1
        if(numnod>1) then
           mstart = dim_linear_norbs*(mynod)/numnod + 1
           mstop  = dim_linear_norbs*(mynod+1)/numnod
        end if
#endif
        msize = (mstop-mstart)+1
        allocate(qm_fockmd_diis_l%FA_sv(msize),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_diis','FA_sv')
        allocate(qm_fockmd_diis_l%FDA(mxfdiis,msize),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_diis','FDA')
     end if

     return
     end subroutine allocate_deallocate_qm_diis


     subroutine allocate_pair_index(qm_scf_indx_l,dim_numat,norbs_qm,uhf)
     implicit none

     TYPE(qm_scf_indx) :: qm_scf_indx_l
     integer :: dim_numat,norbs_qm
     logical :: uhf
     integer :: LMI,LME,norbs,nn
     integer :: ier=0

     norbs     = 9*dim_numat
     LMI       = 45*dim_numat
     LME       = 81*dim_numat
     nn        = dim_numat*(dim_numat+1)/2

     ! deallocate memory.
     if(allocated(qm_scf_indx_l%ip_local))     deallocate(qm_scf_indx_l%ip_local,stat=ier)
     if(allocated(qm_scf_indx_l%ip1_local))    deallocate(qm_scf_indx_l%ip1_local,stat=ier)
     if(allocated(qm_scf_indx_l%ip2_local))    deallocate(qm_scf_indx_l%ip2_local,stat=ier)
     if(allocated(qm_scf_indx_l%jp1_local))    deallocate(qm_scf_indx_l%jp1_local,stat=ier)
     if(allocated(qm_scf_indx_l%jp2_local))    deallocate(qm_scf_indx_l%jp2_local,stat=ier)
     if(allocated(qm_scf_indx_l%jp3_local))    deallocate(qm_scf_indx_l%jp3_local,stat=ier)
     if(allocated(qm_scf_indx_l%jx_local))     deallocate(qm_scf_indx_l%jx_local,stat=ier)
     if(allocated(qm_scf_indx_l%ip_check))     deallocate(qm_scf_indx_l%ip_check,stat=ier)
#if KEY_PARALLEL==1
     qm_scf_indx_l%q_mynod_fock_setup =.false. ! flag for setup.
     if(allocated(qm_scf_indx_l%q_mynod_fock)) deallocate(qm_scf_indx_l%q_mynod_fock,stat=ier)

     qm_scf_indx_l%q_do_atom_setup =.false.    ! flag for setup.
     if(allocated(qm_scf_indx_l%q_do_atom))    deallocate(qm_scf_indx_l%q_do_atom,stat=ier)
#endif
     if(allocated(qm_scf_indx_l%pn_diag))      deallocate(qm_scf_indx_l%pn_diag,stat=ier)

     ! allocate memory.
     ! for integer arrays:
     allocate(qm_scf_indx_l%ip_local(LMI),stat=ier)
     allocate(qm_scf_indx_l%ip_check(LMI),stat=ier)
#if KEY_PARALLEL==1
     allocate(qm_scf_indx_l%q_mynod_fock(nn),stat=ier)
     allocate(qm_scf_indx_l%q_do_atom(dim_numat),stat=ier)
#endif
     !if(qm_gho_info_c%q_gho) then
     !   allocate(qm_scf_indx_l%pn_diag(qm_gho_info_c%norbhb),stat=ier)
     !else
        allocate(qm_scf_indx_l%pn_diag(norbs_qm),stat=ier)  ! norbs_qm=qm_main_l%norbs
     !end if

     if(uhf) then
        allocate(qm_scf_indx_l%ip1_local(LMI),stat=ier)
        allocate(qm_scf_indx_l%ip2_local(LMI),stat=ier)

        allocate(qm_scf_indx_l%jp1_local(LME),stat=ier)
        allocate(qm_scf_indx_l%jp2_local(LME),stat=ier)
        allocate(qm_scf_indx_l%jp3_local(LME),stat=ier)
        allocate(qm_scf_indx_l%jx_local(dim_numat),stat=ier)
     !else
     !   allocate(qm_scf_indx_l%ip1_local(1),stat=ier)
     !   allocate(qm_scf_indx_l%ip2_local(1),stat=ier)
     !
     !   allocate(qm_scf_indx_l%jp1_local(1),stat=ier)
     !   allocate(qm_scf_indx_l%jp2_local(1),stat=ier)
     !   allocate(qm_scf_indx_l%jp3_local(1),stat=ier)
     !   allocate(qm_scf_indx_l%jx_local(1),stat=ier)
     end if

     return
     end subroutine allocate_pair_index


     subroutine allocate_deallocate_gho(qm_scf_main_l,qm_gho_info_l, &
                                     qdiis,uhf,qallocate)
     !
     ! allocate/deallocate gho/scf related arrays.
     ! if qallocate == .true. , allocate memory
     !                  false., deallocate memory
     implicit none
     TYPE(qm_scf_main) :: qm_scf_main_l
     TYPE(qm_gho_info) :: qm_gho_info_l
     logical :: qdiis,uhf,qallocate

     integer :: dim_norbs,dim_norbs2,dim_linear_norbs
     integer :: ier=0

     ! first, define array sizes (determined in determine_qm_scf_arrray_size)
     dim_norbs       = qm_scf_main_l%dim_norbs
     dim_norbs2      = qm_scf_main_l%dim_norbs2
     dim_linear_norbs= qm_scf_main_l%dim_linear_norbs

     ! deallocate if arrays are allocated.
     ! for alpha orbitals.
     if(allocated(qm_gho_info_l%PHO))   deallocate(qm_gho_info_l%PHO,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','PHO')
     if(allocated(qm_gho_info_l%FAOA))  deallocate(qm_gho_info_l%FAOA,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','FAOA')
     if(allocated(qm_gho_info_l%CAHB))  deallocate(qm_gho_info_l%CAHB,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','CAHB')
     if(allocated(qm_gho_info_l%DAHB))  deallocate(qm_gho_info_l%DAHB,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','DAHB')
     if(allocated(qm_gho_info_l%FAHB))  deallocate(qm_gho_info_l%FAHB,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','FAHB')
     if(allocated(qm_gho_info_l%FAHBwrk))  deallocate(qm_gho_info_l%FAHBwrk,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','FAHBwrk')
     if(allocated(qm_gho_info_l%PAHB))  deallocate(qm_gho_info_l%PAHB,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','PAHB')
     if(allocated(qm_gho_info_l%PAOLD)) deallocate(qm_gho_info_l%PAOLD,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','PAOLD')

     ! for beta orbitals.
     if(allocated(qm_gho_info_l%PBHO))  deallocate(qm_gho_info_l%PBHO,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','PBHO')
     if(allocated(qm_gho_info_l%FAOB))  deallocate(qm_gho_info_l%FAOB,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','FAOB')
     if(allocated(qm_gho_info_l%CBHB))  deallocate(qm_gho_info_l%CBHB,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','CBHB')
     if(allocated(qm_gho_info_l%DBHB))  deallocate(qm_gho_info_l%DBHB,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','DBHB')
     if(allocated(qm_gho_info_l%FBHB))  deallocate(qm_gho_info_l%FBHB,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','FBHB')
     if(allocated(qm_gho_info_l%FBHBwrk))  deallocate(qm_gho_info_l%FBHBwrk,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','FBHBwrk')
     if(allocated(qm_gho_info_l%PBHB))  deallocate(qm_gho_info_l%PBHB,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','PBHB')
     if(allocated(qm_gho_info_l%PBOLD)) deallocate(qm_gho_info_l%PBOLD,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','PBOLD')

     ! now, allocate memory, only if qallocate==.true.
     if(qallocate) then
        ! for alpha orbitals.
        allocate(qm_gho_info_l%PHO(dim_linear_norbs),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_gho','PHO')
        allocate(qm_gho_info_l%FAOA(dim_linear_norbs),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_gho','FAOA')
        allocate(qm_gho_info_l%CAHB(dim_norbs2),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_gho','CAHB')
        allocate(qm_gho_info_l%DAHB(dim_linear_norbs),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_gho','DAHB')
        allocate(qm_gho_info_l%FAHB(dim_linear_norbs),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_gho','FAHB')
        allocate(qm_gho_info_l%FAHBwrk(dim_norbs,dim_norbs),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_gho','FAHBwrk')
        allocate(qm_gho_info_l%PAHB(dim_linear_norbs),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_gho','PAHB')
        allocate(qm_gho_info_l%PAOLD(dim_linear_norbs),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_gho','PAOLD')
        ! for beta orbitals.
        if(uhf) then
           allocate(qm_gho_info_l%PBHO(dim_linear_norbs),stat=ier)
              if(ier.ne.0) call Aass(1,'allocate_deallocate_gho','PBHO')
           allocate(qm_gho_info_l%FAOB(dim_linear_norbs),stat=ier)
              if(ier.ne.0) call Aass(1,'allocate_deallocate_gho','FAOB')
           allocate(qm_gho_info_l%CBHB(dim_norbs2),stat=ier)
              if(ier.ne.0) call Aass(1,'allocate_deallocate_gho','CBHB')
           allocate(qm_gho_info_l%DBHB(dim_linear_norbs),stat=ier)
              if(ier.ne.0) call Aass(1,'allocate_deallocate_gho','DBHB')
           allocate(qm_gho_info_l%FBHB(dim_linear_norbs),stat=ier)
              if(ier.ne.0) call Aass(1,'allocate_deallocate_gho','FBHB')
           allocate(qm_gho_info_l%FBHBwrk(dim_norbs,dim_norbs),stat=ier)
              if(ier.ne.0) call Aass(1,'allocate_deallocate_gho','FBHBwrk')
           allocate(qm_gho_info_l%PBHB(dim_linear_norbs),stat=ier)
              if(ier.ne.0) call Aass(1,'allocate_deallocate_gho','PBHB')
           allocate(qm_gho_info_l%PBOLD(dim_linear_norbs),stat=ier)
              if(ier.ne.0) call Aass(1,'allocate_deallocate_gho','PBOLD')
        end if
     end if
     return
     end subroutine allocate_deallocate_gho


     subroutine allocate_deallocate_gho_info(qm_gho_info_l,qallocate)
     !
     ! allocate/deallocate gho related arrays.
     ! if qallocate == .true. , allocate memory
     !                  false., deallocate memory
     implicit none
     TYPE(qm_gho_info) :: qm_gho_info_l
     logical :: qdiis,uhf,qallocate

     integer :: ngho,ngho2
     integer :: ier=0

     ! define array sizes
     ngho  = qm_gho_info_l%nqmlnk
     ngho2 = qm_gho_info_l%nqmlnk * qm_gho_info_l%mqm16

     ! deallocate if arrays are allocated.
     if(allocated(qm_gho_info_l%IQLINK)) deallocate(qm_gho_info_l%IQLINK,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho_info','IQLINK')
     if(allocated(qm_gho_info_l%JQLINK)) deallocate(qm_gho_info_l%JQLINK,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho_info','JQLINK')
     if(allocated(qm_gho_info_l%KQLINK)) deallocate(qm_gho_info_l%KQLINK,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho_info','KQLINK')
     if(allocated(qm_gho_info_l%QMATMQ)) deallocate(qm_gho_info_l%QMATMQ,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho_info','QMATMQ')
     if(allocated(qm_gho_info_l%BT))     deallocate(qm_gho_info_l%BT,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho_info','BT')
     if(allocated(qm_gho_info_l%BTM))    deallocate(qm_gho_info_l%BTM,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho_info','BTM')
     if(allocated(qm_gho_info_l%DBTMMM)) deallocate(qm_gho_info_l%DBTMMM,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_gho_info','DBTMMM')

     ! now, allocate memory, only if qallocate==.true.
     if(qallocate) then
        allocate(qm_gho_info_l%IQLINK(ngho),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_gho_info','IQLINK')
        allocate(qm_gho_info_l%JQLINK(3,ngho),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_gho_info','JQLINK')
        allocate(qm_gho_info_l%KQLINK(ngho),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_gho_info','KQLINK')
        allocate(qm_gho_info_l%QMATMQ(ngho),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_gho_info','QMATMQ')
        allocate(qm_gho_info_l%BT(ngho2),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_gho_info','BT')
        allocate(qm_gho_info_l%BTM(ngho2),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_gho_info','BTM')
        allocate(qm_gho_info_l%DBTMMM(3,3,ngho2),stat=ier)
           if(ier.ne.0) call Aass(1,'allocate_deallocate_gho_info','DBTMMM')
     end if
     return
     end subroutine allocate_deallocate_gho_info

     !==================================================================
     subroutine Aass( condition, routine_name, variable_name)
     ! Assertion failure reporter 

     use stream
#if KEY_PARALLEL==1
     use parallel
#endif

     implicit none

     integer      :: condition,mmy_node
     character(*) :: routine_name
     character(*) :: variable_name

#if KEY_PARALLEL==1
     mmy_node = MYNOD
#else
     mmy_node = 0
#endif

     if(condition.eq.1) then              ! allocation
        write(6,*)'Allocation failed in variable ',variable_name, &
                  ' of routine ', routine_name,' in node',mmy_node,'.'
     else if(condition.eq.0) then         ! deallocation
        write(6,*)'Deallocation failed in variable ',variable_name, &
                  ' of routine ', routine_name,' in node',mmy_node,'.'
     else
         write(6,*)'Wrong call of Aass routine from ', &
         routine_name,' in node',mmy_node,'.'
     end if

     call wrndie(-5,'<Aass>','Memory failure. CHARMM will stop.')

     return
     end subroutine Aass
     !--------------------------------------------------------------

#endif
end module qm1_info
! end
