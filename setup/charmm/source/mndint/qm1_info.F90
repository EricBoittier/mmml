! qm and qm/mm info
module qm1_info
  use chm_kinds
  use dimens_fcm

#if KEY_MNDO97==1
  implicit none
  !
  TYPE, public :: qm_control
    logical          :: ifqnt=.false.     ! main logical flag (.true., if qm/mm is on.)
    ! cpmd related.
    logical          :: cpmd=.false.          ! use cpmd for the density propagation.
    logical          :: md_run=.false.        ! running molecular dynamics?
    logical          :: qm_e_set=.false.      ! flag indicating scf energy was yet computed.
    logical          :: q_do_cpmd_pme=.false. ! flag to do pme only for cpmd. (on when md)
    logical          :: q_fast_pme=.false.    ! flag to 

    ! DXL-BOMD (AMN Niklasson & Guishan Zheng) related
    logical          :: q_dxl_bomd=.false.    ! use DXL-BOMD (AMN Niklasson & Guishan Zheng)
    logical          :: q_do_dxl_scf=.false.  ! runtime control if scf finishes in N_scf_step (in scf_iter).
    integer          :: N_scf_step = 0        ! number of scf steps.

    ! Fock matrix dynamics (based on DIIS)
    logical          :: q_fockmd=.false.        ! use Fock matrix dynamics (ref: ).
    logical          :: q_do_fockmd_scf=.false. ! runtime control of the Fock matrix dynamics (in scf_iter).
    integer          :: imax_fdiss              ! maximum number of diis iteration.
    integer          :: i_fockmd_option=2       ! options for fock-md.
                                                ! 1: based on the extrapolation
                                                ! 2: based on the Verlet integration.

    !
    character(len=6) :: qm_model          ! mndo(1),am1(2),pm3(3),am1/d(4),mndo/d(5)
    integer          :: iqm_mode = 2      ! 1,2,3,4,5
    logical          :: q_am1_pm3=.false. ! .false. (default), if mndo or mndod
                                          ! .true. if am1, pm3, or am1d
    logical          :: do_d_orbitals=.false.  ! have d-orbital specific terms.
 
    logical          :: qmqm_analyt=.false. ! use qm-qm analytical derivative?
                                            ! .false. (default) for now!
    logical          :: q_diis=.false.      ! use DIIS.

    ! for qm/mm energy (in kcal/mol)
    real(chm_real)   :: E_total,E_scf,E_nuclear 
    real(chm_real)   :: E_ewald_corr

    ! for finite difference derivative.
    real(chm_real)   :: del =1.0D-06, & ! displacement distance
                        rdel=5.0D+05

    ! for mapping between qm and mm part
    integer, pointer :: qminb(:) =>Null()    ! pointer for qm atoms in the charmm main array
    integer, pointer :: mminb1(:)=>Null()    ! pointer for mm atoms in the charmm main array
    integer, pointer :: mminb2(:)=>Null()    ! the reverse pointer of mminb1.
    real(chm_real),pointer:: cgqmmm(:)=>Null() ! charge of mm atoms in the original CG array.

    ! for analysis
    logical          :: q_bond_order=.false.   ! bond order analysis.
    logical          :: q_m_charge  =.false.   ! Mulliken population analysis
    integer          :: ianal_unit  =6         ! output unit for analysis results.

  END TYPE qm_control

  !
  TYPE, public :: qm_main
    integer          :: NUMAT=0          ! number of qm atoms.
    logical          :: rij_qm_incore =.false.  ! use incore if true.

    real(chm_real)   :: qmcharge=0.0d0   ! total charge of qm region
    integer          :: i_qmcharge=0     ! integer version of qmcharge

    ! imult needs explanation, for both RHF and UHF
    ! 0; singlet, 2; doublet, 3; triplet
    integer          :: imult            ! spin state of qm region.
    logical          :: uhf=.false.      ! default is RHF (so, uhf=.false.)
    !
    real(chm_real)   :: elec_eng, &      ! electronic energy (eV)
                        ener_atomic, &   ! sum of atomic energy (eV)
                        HofF_atomic, &   !        atomic heat of formation (eV)
                        enuclr_qmqm, &   ! core-core energy for qm-qm (eV)
                        enuclr_qmmm      ! core-core energy for qm-mm (eV)

    !
    integer,pointer  :: nat(:)=>Null()   ! atomic number of atom i
    integer,dimension(:),allocatable  :: nfirst, & ! first basis orbital of atom i
                        nlast     ! last  basis orbital of atom i.

    ! number of orbitals, electrons, occupation numbers
    integer          :: norbs            ! number of orbitals
    integer,dimension(:),allocatable  :: num_orbs ! number of orbitals for atom i.
    
    integer          :: ijpair           ! norbs*(norbs-1)/2+norbs

    integer          :: nel,    &        ! number of electrons
                        nalpha, &        !           alpha electrons
                        nbeta            !           beta
    !  note: nbeta = number of doubly occupied molecular orbitals 
    !                if RHF and imult .ne. 1
    integer          :: numb, &          ! number of highest occupied mol. orbital = no. of occup. orb.
                        nclo, &          ! number of closed orbitals?
                        nmos             !        of occupied mol. orbital
    !
    ! HALFE
    integer          :: iodd, &          ! 0, for RHF
                        jodd             ! 0, for RHF
    ! OCCFL
    integer          :: imocc, &         ! 1=abs(iuhf), if RHF, iuhf=-1 (default).
                        nocca, &         !
                        noccb            !

  
    ! QM coordinates (3,numat)
    real(chm_real),pointer :: qm_coord(:,:)=> NULL()
    real(chm_real),pointer :: qm_grads(:,:)=> Null() 
    real(chm_real),pointer :: qm_grads2(:,:)=> Null() 

    ! moved to qmmm_ewald type
    !real(chm_real),pointer :: rijdata_qmqm(:,:)=>Null()  ! incore data, rij, 1/rij.

    ! for the non-bond list generation
    !integer           :: nmaxgrp=201
    !integer           :: nqmgrp(200+1)  ! as nqmgrp(1) is the number of QM groups.
    integer           :: nmaxgrp
    integer,pointer   :: nqmgrp(:)=>Null()

    ! d-orbital related info.
    ! DELEMT
    integer      :: NELMD=0, &  ! Number of atoms with D-orbitals.
                    NFOCK=5     ! setp size in Subroutine FOCK (5 for sp, 10 for spd).

    ! occupation number related info.
    

  END TYPE qm_main

  ! 
  TYPE, public :: mm_main
    integer    :: natom                  ! the total number of atoms in the system.
    integer    :: natom_mm               ! the total number of MM atoms in the system.
    integer    :: NUMATM                 ! number of MM atoms (within cutoff)
    integer    :: nlink=0                ! number of qm-mm link atoms (H-link atoms) 
    logical    :: rij_mm_incore=.false.  ! use incore if true.
    logical    :: q_lookup=.false., &    ! use look-up table.
                  q_lookup_beta=.false., &
                  q_lookup_read=.false., &
                  q_lookup_setup=.false.
    !
    integer    :: i_lookup_unit

    ! for mapping between mndo97 routines and charmm
    integer,pointer :: qm_mm_pair_list(:)=>Null()  ! 
    ! mm coordinates
    real(chm_real),pointer :: mm_coord(:,:)=> NULL()
    ! mm gradients
    real(chm_real),pointer :: mm_grads(:,:)=> Null()
    real(chm_real),pointer :: mm_grads2(:,:)=> Null()
    ! mm charges
    real(chm_real),pointer :: mm_chrgs(:)  => Null()
    !
    ! moved to the qmmm_ewald type
    !real(chm_real),pointer :: rijdata_qmmm(:,:)=>NUll() ! incore data, rij, 1/rij.

    ! Cutoff based on group-based cutoff.
    ! 1. In the non-bond list, differently from that of the default (default) group-based list, 
    !    in which any MM group that is within the cutoff distance from any QM group
    !    is included, the list is splitted such that each QM group has different MM list, which
    !    is within the cutoff update distance. (see routine GUQME3_by_group in qmnbnd.src.)
    ! 2. When evaluating qm-mm interactions, for each QM group, any MM group in the list
    !    is exlcuded in the interaction evaluation if that MM group is away more than
    !    the cutoff distance.
    ! 3. If the following switch is on, the interaction between the dist_on and dist_off
    !    is switched off at the dist_off distance. (This is the default with q_cut_by_group=.true.
    ! 4. When LQMEWD is true, the interaction energy between the i (QM) and j (MM) atom pair is
    !
    !    E_ij (r_ij) = Sw(r_ij)*E_qm-mm-model(r_ij) + (1-Sw(r_ij))*E_qm-mm-long-distance-model(r_ij)
    !
    !    where E_qm-mm-model is the regular QM-MM interaction model, and
    !          E_qm-mm-long-distance-model is the QM-Mulliken-charge and MM charge interaction,
    !          which is used in the QM/MM-Ewald and QM/MM-PME long interaction model.
    !
    logical :: q_cut_by_group=.false.    ! default is to use group-based cutoff (internally.
                                         ! note that this is a duplicate copy of nbndqm_ltm.src.
    integer :: inum_mm_grp               ! total number of mm groups within the cutoff dist.
                                         ! this is different from num_mm_group (nbndqm_ltm.src).
                                         ! in fact, this is the actual no. of mm groups in the cutoff.
    real(chm_real) :: r_cut_group_dist_on, &
                      r_cut_group_dist_off
    real(chm_real),pointer :: r2_mmgrp_qmgrp_dist(:,:)=>Null() ! dist^2 between each qm-mm group pair.
    real(chm_real),pointer :: r_num_atom_qm_grp(:)=>Null()     ! 1/float(n_qm_atm_i-th qm group)
    real(chm_real),pointer :: r_num_atom_mm_grp(:)=>Null()     ! 1/float(n_mm_atm_j-th mm group)
    logical,pointer :: q_mmgrp_qmgrp_cut(:,:)=>Null()  ! num_mm_group x num_qm_group pair 
                                                       ! .true.  this qm-mm pair within the cutoff
                                                       ! .false. this qm-mm pair outside of the cutoff.
    ! switching  function related.
    logical :: q_switch =.false.                       ! use switch function between on and off distances.
    integer,pointer :: q_mmgrp_point_swt(:)  =>Null()  ! point which qm group is in the switch region of mm.
    integer,pointer :: q_mmgrp_qmgrp_swt(:,:)=>Null()  ! num_mm_group x num_qm_group pair
                                                       ! > 0  this qm-mm pair within the switch region. 
                                                       ! < 0  this qm-mm pair outside of the switch region.
    integer :: isize_swt_array
    real(chm_real),pointer :: sw_val(:)=>Null(), &      ! swtching function value(rij)
                              dxyz_sw(:,:)=>Null(),  &  ! gradient components for each group.
                              dsw_val(:,:)=>Null(),dxyz_sw2(:,:)=>Null()      ! dSw(rij)/drij*{xyz(1:3,qm)-xyz(1:3,mm)}
    integer,pointer :: q_backmap_dxyz_sw(:)=>Null()   ! pointer to which qm-mm group pair to add the gradient components.

    ! With this flag on, we replace the diagonal block of the qm/mm interaction is replaced by
    ! purely 1/r interaction, which is calculated in the ewald routines.
    logical :: q_diag_coulomb =.false.                  ! should be only used when qm/mm-ewald or qm/mm-pme is used.
    
    ! for qm/mm-ewald or pme version
    logical :: LQMEWD =.false.           ! use qm/mm-Ewald.
    logical :: PMEwald=.false.           ! use qm/mm-PME.
    integer :: EWMODE =1                 ! Ewald mode.
    integer :: NQMEWD =0                 !
    !
    real(chm_real),pointer :: qm_charges(:)=> Null() ! qm mulliken charges.

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
    integer :: dim_numat                 ! the size of the number of atoms (LM1)
    integer :: dim_norbs                 ! the size of the number of orbitals (LM2)
    integer :: dim_norbs2                ! norbs*norbs (LM2*LM3)
    integer :: dim_linear_norbs          ! linear dimension; LM4=norbs*(norbs+1)/2
    integer :: dim_linear_fock           ! linear dimension of half-triangle 
                                         !        of fock matrix calc.; LM6
    integer :: dim_linear_fock2          ! square size of dim_linear_fock; LM9
                                         ! neeed for two-electron interactions.
    integer :: dim_2C_2E_integral        ! unique two-center two-electron integrals.
    integer :: dim_scratch               ! the size of real scratch array   : LWORK=8*norbs
    integer :: dim_iscratch              ! the size of integer scratch array: LIWORK=norbs
               
    ! SCRT
    real(chm_real) :: SCFCRT=1.0d-6      ! scf convergence criteria
    real(chm_real) :: PLCRT =1.0d-4      ! density conv. criteria

    ! maximum scf iteraction
    integer        :: KITSCF=200         ! default value.

    ! INDEXT 
    ! size LMX, this is initialized in DYNSCF, construct_index
    ! LMX = 9*numat
    integer,dimension(:),allocatable :: INDX  ! size LMX, this is initialized in DYNSCF
    !
    ! INDEXW: NW(i) is the first element point of the lower triangle of block diagonal
    !         term. *see routine define_pair_index (qm1_scf_module).
    ! LM1 = numat
    integer,pointer :: NW(:)=>Null()    ! size LM1

    ! needed arrays.
    ! H(LM4); W(LM9); FA(LM4)=FB; CA(LM2,LM3)=CB; PA(LM4)=PB; DA(LM4)=DB; EA(LM3)=EB
    ! Q(LM2*8); iwork(6*LMX)
    real(chm_real),dimension(:),allocatable :: H       ! core-hamiltonian matrix.
    real(chm_real),dimension(:),allocatable :: W       ! Two-electron integrals. (1D .vs. 2D?)
    real(chm_real),pointer:: FA(:)=>Null(), &   ! RHF of UHF-alpha Fock matrix.
                             FB(:)=>Null()      ! UHF-beta Fock matrix.
    real(chm_real),pointer:: FAwork(:,:)=>Null()! Fock matrix in square form, working array.
    real(chm_real),pointer:: FBwork(:,:)=>Null()! beta Fock matrix in square form
    real(chm_real),pointer:: CA(:,:)=>Null(), & ! RHF of UHF-alpha MO eigenvectors.
                             CB(:,:)=>Null()    ! UHF-beta MO eigenvectors.
    real(chm_real),pointer:: PA(:)=>Null(), &   ! RHF of UHF-alpha density matrix.
                             PB(:)=>Null()      ! UHF-beta density matrix.
    real(chm_real),pointer:: PAwork(:,:)=>Null()! Fock matrix in square form, working array.
    real(chm_real),pointer:: PBwork(:,:)=>Null()! beta Fock matrix in square form
    real(chm_real),pointer:: DA(:)=>Null(), &   ! RHF of UHF-alpha difference density matrix.
                             DB(:)=>Null()      ! UHF-beta difference density matrix.
    real(chm_real),pointer:: EA(:)=>Null(), &   ! RHF of UHF-alpha MO eigenvalues.
                             EB(:)=>Null()      ! UHF-beta MO eigenvalues.

    real(chm_real),pointer:: Q(:)=>Null()       ! Scaratch array for various purposes.
    integer,       pointer:: iwork(:)=>Null()   ! integer scratch array (see iter.)

    
    ! for precomputing some parameters:
    real(chm_real),pointer:: H_1cent(:)=>Null() ! core-hamiltonina matrix for one-center terms.
    integer,       pointer:: imap_h(:) =>Null() ! mapping H_1cent to H in one_center_h
    !integer               ::dim_imap_h         ! size of imap_h array
    
    ! for EXTRA1 & EXTRA2
    real(chm_real) :: RI(22),CORE_mat(10,2),WW(2025)  ! REPT(10,2)
    real(chm_real) :: SIJ(14),T(14),YY(675)

  END TYPE qm_scf_main


  ! assign 
  TYPE(qm_control),  save :: qm_control_r
  TYPE(qm_main),     save :: qm_main_r
  TYPE(mm_main),     save :: mm_main_r
  TYPE(qm_scf_main), save :: qm_scf_main_r


  ! 
  contains

     !==================================================================
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

     ! deallocate if arrays are associated.
     if(associated(qm_control_l%qminb) ) deallocate(qm_control_l%qminb,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm','qminb')
     if(associated(qm_control_l%mminb1) ) deallocate(qm_control_l%mminb1,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm','mminb1')
     if(associated(qm_control_l%mminb2) ) deallocate(qm_control_l%mminb2,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm','mminb2')
     if(associated(qm_control_l%cgqmmm)) deallocate(qm_control_l%cgqmmm,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qmmm','cgqmmm')

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

     ! deallocate if arrays are associated.
     if(associated(qm_main_l%nat)     ) deallocate(qm_main_l%nat,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm','nat')
     if(associated(qm_main_l%qm_coord)) deallocate(qm_main_l%qm_coord,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm','qm_coord')
     if(associated(qm_main_l%qm_grads)) deallocate(qm_main_l%qm_grads,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm','qm_grads')
     if(associated(qm_main_l%qm_grads2)) deallocate(qm_main_l%qm_grads2,stat=ier)
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

     ! deallocate if arrays are associated.
     if(associated(mm_main_l%qm_mm_pair_list)) deallocate(mm_main_l%qm_mm_pair_list,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_mm','qm_mm_pair_list')
     if(associated(mm_main_l%mm_coord))  deallocate(mm_main_l%mm_coord,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_mm','mm_coord')
     if(associated(mm_main_l%mm_grads))  deallocate(mm_main_l%mm_grads,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_mm','mm_grads')
     if(associated(mm_main_l%mm_grads2))  deallocate(mm_main_l%mm_grads2,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_mm','mm_grads2')
     if(associated(mm_main_l%mm_chrgs))  deallocate(mm_main_l%mm_chrgs,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_mm','mm_chrgs')
     !if(associated(mm_main_l%EMPOT)   )  deallocate(mm_main_l%EMPOT,stat=ier)
     !   if(ier.ne.0) call Aass(0,'allocate_deallocate_mm','EMPOT')
     !if(associated(mm_main_l%ESLF)    )  deallocate(mm_main_l%ESLF,stat=ier)
     !   if(ier.ne.0) call Aass(0,'allocate_deallocate_mm','ESLF')
     if(associated(mm_main_l%qm_charges)) deallocate(mm_main_l%qm_charges,stat=ier)
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

     ! deallocate if arrays are associated.

     ! for integer arrays:
     if(associated(qm_scf_main_l%NW))   deallocate(qm_scf_main_l%NW,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','NW')

     if(associated(qm_scf_main_l%iwork)) deallocate(qm_scf_main_l%iwork,stat=ier)
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
     if(associated(qm_scf_main_l%FAwork)) deallocate(qm_scf_main_l%FAwork,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','FAwork')
     if(associated(qm_scf_main_l%FBwork)) deallocate(qm_scf_main_l%FBwork,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','FBwork')
     if(associated(qm_scf_main_l%CA)) deallocate(qm_scf_main_l%CA,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','CA')
     if(associated(qm_scf_main_l%CB)) deallocate(qm_scf_main_l%CB,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','CB')
     if(associated(qm_scf_main_l%PA)) deallocate(qm_scf_main_l%PA,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','PA')
     if(associated(qm_scf_main_l%PB)) deallocate(qm_scf_main_l%PB,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','PB')
     if(associated(qm_scf_main_l%PAwork)) deallocate(qm_scf_main_l%PAwork,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','PAwork')
     if(associated(qm_scf_main_l%PBwork)) deallocate(qm_scf_main_l%PBwork,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','PBwork')
     if(associated(qm_scf_main_l%DA)) deallocate(qm_scf_main_l%DA,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','DA')
     if(associated(qm_scf_main_l%DB)) deallocate(qm_scf_main_l%DB,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','DB')
     if(associated(qm_scf_main_l%EA)) deallocate(qm_scf_main_l%EA,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','EA')
     if(associated(qm_scf_main_l%EB)) deallocate(qm_scf_main_l%EB,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','EB')
     if(associated(qm_scf_main_l%Q))  deallocate(qm_scf_main_l%Q,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','Q')

     ! for precomputation:
     if(associated(qm_scf_main_l%H_1cent)) deallocate(qm_scf_main_l%H_1cent,stat=ier)
        if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_scf','H_1cent')
     if(associated(qm_scf_main_l%imap_h)) deallocate(qm_scf_main_l%imap_h,stat=ier)
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
