! qm and qm/mm setup and other interfaces.
module qmmm_interface
  use chm_kinds
  use dimens_fcm

#if KEY_MNDO97==1 /*mndo97*/
  contains

  !=====================================================================
  subroutine array_pointers(q_assign,irepl)
  !=====================================================================
  !
  ! assign pointers...
  !
  use qm1_info, only : qm_main_r,mm_main_r,qm_control_r,qm_scf_main_r,qm_param_r, &
                       qm_main_c,mm_main_c,qm_control_c,qm_scf_main_c,qm_param_c, &
                       qm_scf_diis_r,qm_scf_diis_c,          &
                       qm_fockmd_diis_r,qm_fockmd_diis_c, &
                       qm_scf_indx_r,qm_scf_indx_c,          &
                       qm_gho_info_r,qm_gho_info_c,          &
                       dxlbomd_r,dxlbomd_c,                  &
                       mlay_r,mlay_c,qm_bond_r,qm_bond_c
  use qmmmewald_module, only : qmmm_ewald_r,qmmm_ewald_c
  use nbndqm_mod, only : map_grp_r,map_grp_c
  use H4_mndo, only : dist_r,dist_c, &
                      grad_r,grad_c

  implicit none
  logical :: q_assign
  integer :: irepl

  ! nullify pointers
  if(associated(qm_main_c))        nullify(qm_main_c)
  if(associated(mm_main_c))        nullify(mm_main_c)
  if(associated(qm_control_c))     nullify(qm_control_c)
  if(associated(qm_scf_main_c))    nullify(qm_scf_main_c)
  if(associated(qm_param_c))       nullify(qm_param_c)
  if(associated(qm_gho_info_c))    nullify(qm_gho_info_c)
  if(associated(qmmm_ewald_c))     nullify(qmmm_ewald_c)
  if(associated(qm_scf_diis_c))    nullify(qm_scf_diis_c)
  if(associated(qm_fockmd_diis_c)) nullify(qm_fockmd_diis_c)
  if(associated(qm_scf_indx_c))    nullify(qm_scf_indx_c)
  if(associated(dist_c))           nullify(dist_c)
  if(associated(grad_c))           nullify(grad_c)

  if(associated(map_grp_c))        nullify(map_grp_c)

  ! qmmm_interface local
  if(associated(dxlbomd_c))        nullify(dxlbomd_c)

  ! mlayered qm/mm
  if(associated(mlay_c))           nullify(mlay_c)

  ! bond/angle/etc info for the qm region
  if(associated(qm_bond_c))        nullify(qm_bond_c)

!!  if(.not. q_assign) then
!!     ! default, pointers to rs state, to make the program to run.
!!     qm_main_c       => qm_main_r(1)
!!     mm_main_c       => mm_main_r(1)
!!     qm_control_c    => qm_control_r(1)
!!     qm_scf_main_c   => qm_scf_main_r(1)
!!     qm_param_c      => qm_param_r(1)
!!     qm_scf_diis_c   => qm_scf_diis_r(1)
!!     qm_fockmd_diis_c=> qm_fockmd_diis_r(1)
!!     qm_scf_indx_c   => qm_scf_indx_r(1)
!!     map_grp_c       => map_grp_r(1)
!!     qm_gho_info_c   => qm_gho_info_r(1)
!!     qmmm_ewald_c    => qmmm_ewald_r(1)
!!
!!     ! qm_control_c%q_dxl_bomd
!!     dxlbomd_c       => dxlbomd_r(1)
!!
!!     ! q_h4corr
!!     dist_c          => dist_r(1)
!!     grad_c          => grad_r(1)
!!
!!     return
!!  end if

  ! assign pointers: rs state (main): irepl==1,
  !                  ps state       : irepl==2
  qm_main_c       => qm_main_r(irepl)
  mm_main_c       => mm_main_r(irepl)
  qm_control_c    => qm_control_r(irepl)
  qm_scf_main_c   => qm_scf_main_r(irepl)
  qm_param_c      => qm_param_r(irepl)
  qm_scf_diis_c   => qm_scf_diis_r(irepl)
  qm_fockmd_diis_c=> qm_fockmd_diis_r(irepl)
  qm_scf_indx_c   => qm_scf_indx_r(irepl)
  map_grp_c       => map_grp_r(irepl)
  qm_gho_info_c   => qm_gho_info_r(irepl)
  qmmm_ewald_c    => qmmm_ewald_r(irepl)

  ! qm_control_c%q_dxl_bomd
  dxlbomd_c       => dxlbomd_r(irepl)

  ! q_h4corr
  dist_c          => dist_r(irepl)
  grad_c          => grad_r(irepl)

  ! mlayered qm/mm
  if(allocated(mlay_r)) mlay_c => mlay_r(irepl)

  ! bond/angle/etc info for the qm region
  if(allocated(qm_bond_r)) qm_bond_c => qm_bond_r(irepl)

  return
  end subroutine array_pointers


  !=====================================================================
  subroutine qmmm_init_set(nqmtheory,nqmcharge,nspin,numat,natgho,  &
                           natom,EWMODE,NQMEWD,                     &
                           qmcharge,scfconv,                        &
                           qlink,QMMM_NoDiis,LQMEWD,NOPMEwald,      &
                           q_bond_order,q_m_charge,iiunit,          &
                           q_dxl_bomd,K_order,N_scf_step,           &
                           q_fockmd,iopt_fdiss,imax_fdiss,          &
                           qmswtch,QNoMemIncore)
  !=====================================================================
  !
  ! setup qmmm options and the default values.
  !
  use number,only : one
  use gamess_fcm,only: igmsel
  use qm1_info, only : qm_control_c,qm_main_c,mm_main_c,qm_scf_main_c, &
                       dxlbomd_c,qm_gho_info_c, &
                       allocate_deallocate_qm,allocate_deallocate_mm, &
                       allocate_deallocate_qmmm

  implicit none
  !
  integer :: nqmtheory,nqmcharge,nspin,numat,natgho,natom,EWMODE,NQMEWD,K_order,N_scf_step,iiunit, &
             iopt_fdiss,imax_fdiss
  real(chm_real):: qmcharge,scfconv
  logical :: qlink,QMMM_NoDiis,LQMEWD,NOPMEwald,q_dxl_bomd,qmswtch, &
             QNoMemIncore,q_bond_order,q_m_charge,q_fockmd
  ! 
  ! local variables
  logical :: do_am1_pm3,do_d_orbitals

  ! for qm main control
  qm_control_c%ifqnt        =.true.
  qm_control_c%qmqm_analyt  =.false.  ! only use finite difference gradient.
  qm_control_c%q_diis       = .not.QMMM_NoDiis
  qm_control_c%md_run       =.false.  ! flag for using md.
  !
  ! QM method: MNDO/AM1/PM3/AM1d/MNDOd
  qm_control_c%iqm_mode     = nqmtheory
  !
  do_d_orbitals =.false.      ! default for AM1
  do_am1_pm3    =.true.       !
  if(nqmtheory.eq.1) then
     qm_control_c%qm_model(1:6) ='MNDO  '
     do_am1_pm3                 =.false.
  else if(nqmtheory.eq.2) then
     qm_control_c%qm_model(1:6) ='AM1   '
  else if(nqmtheory.eq.3) then
     qm_control_c%qm_model(1:6) ='PM3   '
  else if(nqmtheory.eq.4) then
     qm_control_c%qm_model(1:6) ='AM1/d '
     do_d_orbitals              =.true.
  else if(nqmtheory.eq.5) then
     qm_control_c%qm_model(1:6) ='MNDO/d'
     do_d_orbitals              =.true.
     do_am1_pm3                 =.false.
  end if
  !
  qm_control_c%q_am1_pm3    = do_am1_pm3     ! do Gaussian core terms for the AM1-type models 
  qm_control_c%do_d_orbitals= do_d_orbitals  ! have d-orbtials.

  ! analysis
  qm_control_c%ianal_unit   = iiunit         ! analysis output unit.
  qm_control_c%q_bond_order = q_bond_order   ! do bond order analysis
  qm_control_c%q_m_charge   = q_m_charge     ! do mulliken charge analysis.


  ! qm_main
  qm_main_c%numat     = numat
  qm_main_c%i_qmcharge= nqmcharge
  qm_main_c%qmcharge  = qmcharge
  qm_main_c%imult     = nspin       ! for singlet, "imult=0"
  qm_main_c%uhf       =.false.      ! for now, turn off UHF.
  if(QNoMemIncore) then
     qm_main_c%rij_qm_incore =.false.
     mm_main_c%rij_mm_incore =.false.
  else
     !for now, turn on this only if qm/mm-ewald is used.
     if (LQMEWD) then
        qm_main_c%rij_qm_incore =.true.   ! turn on the memory incore.
        mm_main_c%rij_mm_incore =.true.   ! turn on the memory incore.
     else
        qm_main_c%rij_qm_incore =.false.
        mm_main_c%rij_mm_incore =.false.
     end if
  end if

  ! scf convergence: see scf_iter routine, where SCFCRT and PLCRT are overwritten.
  qm_scf_main_c%SCFCRT= scfconv     ! 1.0d-6; default value.
  qm_scf_main_c%PLCRT = scfconv     ! 1.0d-4      ! default value. 
  !qm_scf_main_c%kitscf= 200        ! .

  ! For DXL-BOMD by AMN Niklasson, JCP (2009) 130:214109 and Guishan Zheng, Harvard Univ. 05/19/2010.
  qm_control_c%q_dxl_bomd=q_dxl_bomd  !
  qm_control_c%N_scf_step=N_scf_step  ! Number of scf cycle per each md step.
  if(qm_control_c%q_dxl_bomd) then
     dxlbomd_c%Kth_sum_order=K_order               ! the order to which the summation will be performed.
     if(qm_control_c%q_dxl_bomd) then
        if(allocated(dxlbomd_c%cextr)) deallocate(dxlbomd_c%cextr)
        allocate(dxlbomd_c%cextr(0:dxlbomd_c%Kth_sum_order))
        call initcoef_diss(dxlbomd_c%Kth_sum_order,dxlbomd_c%coefk,dxlbomd_c%cextr)
     end if
  end if

  ! Fock matrix dynamics (ref: )
  qm_control_c%q_fockmd        = q_fockmd
  qm_control_c%q_do_fockmd_scf =.false.        ! initialization.
  qm_control_c%i_fockmd_option = iopt_fdiss    ! Options for fock-MD (see qm_info.src and fock_diis routine.)
  qm_control_c%imax_fdiss      = imax_fdiss    ! maximum number of diis iterations.

  ! mm_main
  mm_main_c%natom   = natom         !
  mm_main_c%natom_mm= natom-numat   !
  mm_main_c%numatm  = natom-numat   ! for the maximum size of pointer arrays.
  mm_main_c%nlink   = 0             ! the number of h-link atom, 0 default.

  ! for qm/mm-ewald
  mm_main_c%LQMEWD = LQMEWD         ! use qm/mm-Ewald
  mm_main_c%PMEwald=.not.NOPMEwald  ! use qm/mm-PME version.
  mm_main_c%EWMODE = EWMODE  ! 1= within cutoff, interact with all components in Fock
                             !    outside of cutoff, only interact with the
                             !    diagonal components of Fock  
                             ! 2= (not used) all qm/mm interactions are evaluated based on
                             !    interactions with the diagonal Fock elements. So, in the
                             !    end, it is equivalent to Muliken charge - MM charge 1/r 
                             !    interaction.
                             ! 3= (hybrid of 1 and 2 options). I.e., for the off-diagonal elements
                             !    of the Fock matrix, interact with mm charges (within cutoff) by
                             !    the conventional qm/mm manner (i.e., regular qm/mm interaction),
                             !    and for the diagonal elements by 1/r with all mm charges.
  if(mm_main_c%EWMODE == 3) then
     mm_main_c%q_diag_coulomb =.true.
  else
     mm_main_c%q_diag_coulomb =.false.
  endif
  mm_main_c%NQMEWD = NQMEWD  ! 0= use Mulliken charge to represent charge of QM images.

  ! For cutoff options,
  ! by turning-on, the qm-mm interactions are evaluated based on the group-by-group pair distance.
  ! So, any group pair that is separated more than the cutoff distance is ignore,
  ! whereas in the default option, any atom pair, in which that MM atom in the group is within
  ! the cutoff distance from any QM group, is evaluated.
  !
  mm_main_c%q_switch          = qmswtch
 
  ! for GHO methods
  qm_gho_info_c%q_gho = qlink   !
  if(qm_gho_info_c%q_gho) then
     qm_gho_info_c%uhfgho= qm_main_c%uhf
     ! see subroutine GHOHYB
     !qm_gho_info_c%numat = qm_main_c%numat
     !qm_gho_info_c%nqmlnk= natgho
  end if

  ! now allocate memories for qm and mm atoms coordinates, gradients, charges, etc:
  call allocate_deallocate_qmmm(qm_control_c,qm_main_c,mm_main_c,.true.)
  call allocate_deallocate_qm(qm_main_c,.true.)
  call allocate_deallocate_mm(qm_main_c,mm_main_c,.true.)

  return
  end subroutine qmmm_init_set


  !=====================================================================
  subroutine qmmm_load_parameters_setup_qm_info(QSRP_PhoT)
  !=====================================================================
  !
  ! load parameters, setup qm info, and allocate scf/qm memories. 
  !
  use qm1_info, only : qm_control_c,qm_main_c,qm_scf_main_c, &
                       qm_param_c,                           &
                       determine_qm_scf_arrray_size,         &
                       allocate_deallocate_qm_scf,           &
                       allocate_deallocate_qm_diis,     &
                       qm_scf_diis_c,qm_fockmd_diis_c,  &
                       qm_gho_info_c, allocate_deallocate_gho
  use qm1_scf_module, only: define_pair_index
  use qm1_parameters, only: q_parm_loaded,                   &
                            initialize_elements_and_params
  use qm1_energy_module,only: QMMM_module_prep
  implicit none
  logical:: QSRP_PhoT,q_allocate

  ! now.................................................................
  ! 1) load qm and qmmm parameters and copied to qm_param_c for local usage.
  !if(.not.q_parm_loaded) then
     call initialize_elements_and_params(qm_control_c%iqm_mode,QSRP_PhoT)
  !end if

  ! 2) load qm parameters and qm info common to all methods:
  call qm_info_setup

  ! 3) determined array sizes to be allocated and allocate them
  call determine_qm_scf_arrray_size(qm_main_c,qm_scf_main_c,.true.)
  call allocate_deallocate_qm_scf(qm_main_c,qm_scf_main_c,.true.)
  ! moved below after gho setup
  !call allocate_deallocate_qm_diis(qm_scf_main_c,qm_scf_diis_c,qm_fockmd_diis_c, &
  !                                 qm_control_c%q_diis,qm_control_c%q_fockmd,    &
  !                                 qm_control_c%imax_fdiss,                      &
  !                                 qm_main_c%uhf,.true.)

  ! 3-1) define pair indexes: IP,IP1,IP2 for Coulomb part
  !                           JX,JP1,JP2,JP3 for Exchange part
  ! see qm1_scf_module
  call define_pair_index

  ! 4) for GHO
  if(qm_gho_info_c%q_gho) then
     call allocate_deallocate_gho(qm_scf_main_c,qm_gho_info_c, &
                                  qm_control_c%q_diis,         &
                                  qm_main_c%uhf,.true.)
     ! determine some gho-related variables
     qm_gho_info_c%norbhb    =qm_main_c%NORBS - 3*qm_gho_info_c%nqmlnk
     qm_gho_info_c%naos      =qm_gho_info_c%norbhb-qm_gho_info_c%nqmlnk
     qm_gho_info_c%lin_naos  =qm_scf_main_c%indx(qm_gho_info_c%naos)  +qm_gho_info_c%naos ! =(NAOS*(NAOS+1))/2
     qm_gho_info_c%lin_norbhb=qm_scf_main_c%indx(qm_gho_info_c%norbhb)+qm_gho_info_c%norbhb
 
     ! variables used in FTOFHB
     qm_gho_info_c%norbao    =qm_main_c%NORBS - 4*qm_gho_info_c%nqmlnk
     qm_gho_info_c%lin_norbao=qm_scf_main_c%indx(qm_gho_info_c%norbao)+qm_gho_info_c%norbao !=NORBAO*(NORBAO+1)/2
     qm_gho_info_c%nactatm   =qm_gho_info_c%numat-qm_gho_info_c%nqmlnk
  end if

  ! 3-2) allocation of the remaining arrays for qm_scf_diis.
  call allocate_deallocate_qm_diis(qm_scf_main_c,qm_scf_diis_c,qm_fockmd_diis_c, &
                                   qm_control_c%q_diis,qm_control_c%q_fockmd,    &
                                   qm_control_c%imax_fdiss,                      &
                                   qm_main_c%uhf,.true.)!
  

  ! 5) now prepare for qm1_energy_module setup
  ! the majority of the parameters are copied to local array "qm_param_c" in 
  ! the routine initialize_elements_and_params/load_param_to_local.
  ! In QMMM_module_prep, the remaining parameters/variables are copied to "qm_param_c"
  ! This separation is necessary, for the case where user provided parameters
  ! (for any specific atoms) are read-in.
  !
  ! Also, some parameters are modified for quick usage.
  call QMMM_module_prep

  ! 6) now precompute some things
  !    One-center part.
  call compute_one_center_h(qm_main_c%numat,qm_scf_main_c,qm_param_c)

  return
  end subroutine qmmm_load_parameters_setup_qm_info


  !=====================================================================
  subroutine parameter_update(irepl,iunit)
  !
  ! read parameters and reset for irepl system. 
  ! irepl  0 : all systems
  !       >0 : irepl system
  ! iunit    : units to read parameter files
  !
  use number
  use qm1_info
  use qm1_parameters
  use parallel
  use stream
  use qm1_constant, only: minbig
  
  implicit none
  integer :: irepl,iunit

  ! local variables
  integer :: i,j,k
  integer :: no_atom_types      ! no. of atom types in the parameter input file
  type(qm_param) :: qm_param_a
  integer :: ier=1

  ! string
  character(len=120):: line

#if KEY_PARALLEL==1
  if(mynod==0) then
#endif
     do
        read(iunit,'(A)') line
        if(line(1:1) == '!') cycle     ! skip comment lines.
        read(line,*) no_atom_types     ! read no. of atom types in the file
        exit
     end do
#if KEY_PARALLEL==1
  end if
  call PSND4(no_atom_types,1)          ! broadcase
#endif

  ! allocate in all nodes
  if(no_atom_types<=0) then
     if(prnlev>=2) write(outu,'(A)') 'Parameter update is ignored.'
     return
  end if
  call allocate_local_param_memories(qm_param_a,no_atom_types,qm_control_r(1)%iqm_mode,.true.)

  ! read parameters from iunit file
  call readpar(qm_param_a,iunit,no_atom_types,ier)
  if(ier /= 1) then
     ! deallocate memories and return
     if(prnlev>=2) write(outu,'(A)') 'Parameter update is ignored.'
     call allocate_local_param_memories(qm_param_a,no_atom_types,qm_control_r(1)%iqm_mode,.false.)
     return
  end if

  ! done reading all parameters and broadcast to all other nodes
  ! now update parameters.
  if(.not. (qm_control_r(1)%iqm_mode == 1 .or. qm_control_r(1)%iqm_mode == 5)) then
     ! for am1 and pm3, find the no. of gaussian core terms.
     do i=1,no_atom_types
        do j=4,1,-1
           if((qm_param_a%GUESS1(j,i) > minbig) .and. (qm_param_a%GUESS1(j,i) /= zero)) then
              ! first non-zero gaussian term, for counting of the total no. of gaussian terms
              qm_param_a%IMPAR(i) = j
              exit
           end if
        end do
     end do
  end if

  ! update parameters
  if(irepl==0) then
     ! do update parameters for all qm replicas
     do i=1,num_qm_system
        call do_parameter_update(i,qm_param_a,no_atom_types)
     end do
  else
     ! do update only a repl qm replica
     call do_parameter_update(irepl,qm_param_a,no_atom_types)
  end if

  ! now update qm_info_setup and one_center_h
  ! do the work of "qm_info_setup": compute sum of atomic energies and heats of formation.
  if(irepl==0) then
     ! for each qm replicas
     do i=1,num_qm_system
        ! update the EISOL value and convert into kcal/mol.
        qm_main_r(i)%ener_atomic = zero
        do j=1,qm_main_r(i)%numat
           qm_main_r(i)%ener_atomic=qm_main_r(i)%ener_atomic+qm_param_r(i)%EISOL(j)
        end do
        qm_main_r(i)%ener_atomic=EVCAL*qm_main_r(i)%ener_atomic

        ! now precompute some things
        ! One-center part.
        call compute_one_center_h(qm_main_r(i)%numat,qm_scf_main_r(i),qm_param_r(i))
     end do
  else
     ! update the EISOL value and convert into kcal/mol.
     qm_main_r(irepl)%ener_atomic = zero
     do i=1,qm_main_r(irepl)%numat
        qm_main_r(irepl)%ener_atomic=qm_main_r(irepl)%ener_atomic+qm_param_r(irepl)%EISOL(i)
     end do
     qm_main_r(irepl)%ener_atomic=EVCAL*qm_main_r(irepl)%ener_atomic

     ! now precompute some things
     ! One-center part.
     call compute_one_center_h(qm_main_r(irepl)%numat,qm_scf_main_r(irepl),qm_param_r(irepl))
  end if

  
  ! deallocate memories
  call allocate_local_param_memories(qm_param_a,no_atom_types,qm_control_r(1)%iqm_mode,.false.)

  return
  end subroutine parameter_update

  !=====================================================================
  subroutine find_unique_qm(ntype_local)
  !
  ! Find the unique number of qm atoms. (need for the Grimme dispersion correction).
  !
  ! THis routine is only used to map with SCC DFTB data structure.
  !
  !use mndo97, only   : nndim ! ,izp
  use qm1_info, only : qm_main_c

  implicit none
  integer :: ntype_local,i,j,ni,icnt
  integer :: nunique_qm
  logical, allocatable :: q_unique(:)

  !
  allocate(q_unique(qm_main_c%numat))
  ! find number of unique atoms.
  do i=2,qm_main_c%numat
     ni          = qm_main_c%nat(i)
     q_unique(i) =.true.
     do j=1,i-1
        if(ni == qm_main_c%nat(j)) then
           ! find qm atom with the same atom type.
           q_unique(i) =.false.
           exit
        end if
     end do
  end do
  !
  !total number of unique qm atoms.
  nunique_qm= 1
  !izp(1)    = 1
  do i=2,qm_main_c%numat
     if(q_unique(i)) nunique_qm = nunique_qm + 1
     !izp(i) = nunique_qm  ! mapping to SCC DFTB format.
  end do
  ntype_local = nunique_qm
  deallocate(q_unique)
  return
  end subroutine find_unique_qm


  !=====================================================================
  subroutine fill_mm_coords(natom,x,y,z,cg,mm_coord,mm_chrgs,qm_mm_pair_list,mminb1)
  !=====================================================================
  ! 
  ! fill mm_coord and mm_chrgs for qm/mm calculation, based on mminb info
  ! which is set at CH2MND routine.
  ! 
  use qm1_info, only : qm_main_c,mm_main_c

  implicit none
  integer :: natom
  real(chm_real):: x(natom),y(natom),z(natom),cg(natom)
  real(chm_real):: mm_coord(3,natom),mm_chrgs(natom)
  integer :: qm_mm_pair_list(*),mminb1(*)
  !
  integer :: n1,m,mmatm

  ! find the number of qm-mm pairs.
  n1=0
  do m=qm_main_c%numat+1,natom
     mmatm=mminb1(m)
     if(mmatm > 0) then
        n1=n1+1
        ! for mapping back to the original array (in gradient)
        qm_mm_pair_list(n1)=mmatm
        mm_coord(1,n1)     =x(mmatm)
        mm_coord(2,n1)     =y(mmatm)
        mm_coord(3,n1)     =z(mmatm)
        mm_chrgs(n1)       =cg(mmatm)
     end if
  end do
  mm_main_c%NUMATM = n1     ! total number of mm atoms included in qm-mm calculations
                            ! shouldn't it be the same as ncutoff?
  return
  end subroutine fill_mm_coords


  !=====================================================================
  subroutine fill_dist_qm_mm_array(numat,NUMATM,qm_coord,mm_coord,LQMEWD)
  !
  ! compute r_ij^2 and 1/r_ij to prepare for qm/mm calculations.
  ! For Ewald, also allocate additional memory for Error function values.
  !
  use number, only : one
  use qm1_info, only : qm_main_c,mm_main_c,Aass
  use qmmmewald_module, only : qmmm_ewald_c
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none
  integer :: numat,NUMATM
  real(chm_real):: qm_coord(3,numat),mm_coord(3,NUMATM)
  logical :: LQMEWD

  integer :: i,j,ii,jj,icnt_qm,icnt_mm,loop_count
  real(chm_real):: xyz_i(3), vec(3), Rij
  

  integer :: ier=0
  integer :: mmynod,nnumnod,mstart,mstop,isize

  ! only turn on this when using LQMEWD
  !if(.not.LQMEWD) then
  !  qm_main_c%rij_qm_incore =.false.
  !  mm_main_c%rij_mm_incore =.false.
  !end if

  !
#if KEY_PARALLEL==1
  mmynod  = mynod
  nnumnod = numnod

  mstart = numatm*mynod/numnod+1
  mstop  = numatm*(mynod+1)/numnod
#else
  mmynod  = 0
  nnumnod = 1

  mstart = 1
  mstop  = numatm
#endif

  ! first check, the size of arrays.
  ! for qm-qm pairs.
  icnt_qm   =0
#if KEY_PARALLEL==1
  loop_count=0 
#endif
  do i=2,numat
     do j=1,i-1
#if KEY_PARALLEL==1
        loop_count=loop_count+1
        if(mmynod .ne. mod(loop_count-1,nnumnod)) cycle
#endif
        !
        icnt_qm   =icnt_qm + 1   ! this is only needed counter.   
     end do
  end do

  ! for qm-mm pairs.
  isize   = mstop-mstart+1    ! numat - 1 + 1
  icnt_mm = numat*isize       ! total array size

  ! first check, if memory needs to be allocated.
  if(qm_main_c%rij_qm_incore) then
     ! check and deallocate
     if(allocated(qmmm_ewald_c%rijdata_qmqm)) then
        if(size(qmmm_ewald_c%rijdata_qmqm) < 2*icnt_qm) then
           deallocate(qmmm_ewald_c%rijdata_qmqm,stat=ier)
           if(ier.ne.0) call Aass(0,'fill_dist_qm_mm_array','rijdata_qmqm')
           if(allocated(qmmm_ewald_c%qmqmerfcx_data)) &
              deallocate(qmmm_ewald_c%qmqmerfcx_data,stat=ier)
           if(ier.ne.0) call Aass(0,'fill_dist_qm_mm_array','qmqmerfcx_data')
        end if
     end if

     ! now allocate needed memory.
     if(.not.allocated(qmmm_ewald_c%rijdata_qmqm)) &
        allocate(qmmm_ewald_c%rijdata_qmqm(2,icnt_qm),stat=ier)
     if(ier.ne.0) call Aass(1,'fill_dist_qm_mm_array','rijdata_qmqm')

     if(LQMEWD .and. .not. allocated(qmmm_ewald_c%qmqmerfcx_data)) &
        allocate(qmmm_ewald_c%qmqmerfcx_data(icnt_qm),stat=ier)
     if(ier.ne.0) call Aass(1,'fill_dist_qm_mm_array','qmqmerfcx_data')
  end if

  if(mm_main_c%rij_mm_incore) then
     ! check and deallocate
     if(allocated(qmmm_ewald_c%rijdata_qmmm)) then
        if(size(qmmm_ewald_c%rijdata_qmmm) < 2*icnt_mm) then
           deallocate(qmmm_ewald_c%rijdata_qmmm,stat=ier)
           if(ier.ne.0) call Aass(0,'fill_dist_qm_mm_array','rijdata_qmmm')
           if(allocated(qmmm_ewald_c%qmmmerfcx_data)) &
              deallocate(qmmm_ewald_c%qmmmerfcx_data,stat=ier)
           if(ier.ne.0) call Aass(0,'fill_dist_qm_mm_array','qmmmerfcx_data')
        end if
     end if

     ! now allocate needed memory. allocate memory a bit larger than needed to
     ! avoid allocate/deallocate every time this routine is called.
     if(.not. allocated(qmmm_ewald_c%rijdata_qmmm)) &
        allocate(qmmm_ewald_c%rijdata_qmmm(2,icnt_mm+100),stat=ier)
     if(ier.ne.0) call Aass(1,'fill_dist_qm_mm_array','rijdata_qmmm')

     if(LQMEWD .and. .not. allocated(qmmm_ewald_c%qmmmerfcx_data)) &
        allocate(qmmm_ewald_c%qmmmerfcx_data(icnt_mm+100),stat=ier)
     if(ier.ne.0) call Aass(1,'fill_dist_qm_mm_array','qmmmerfcx_data')
  end if

  ! now fill the memory.
  if(qm_main_c%rij_qm_incore) then
     icnt_qm   =0
#if KEY_PARALLEL==1
     loop_count=0
#endif
     do i=2,numat
        xyz_i(1:3) = qm_coord(1:3,i)
        do j=1,i-1
#if KEY_PARALLEL==1
           loop_count=loop_count+1
           if(mmynod .ne. mod(loop_count-1,nnumnod)) cycle
#endif
           !
           icnt_qm  = icnt_qm + 1   ! this is only needed counter.
           vec(1:3) = xyz_i(1:3)-qm_coord(1:3,j)
           Rij      = sqrt(vec(1)*vec(1)+vec(2)*vec(2)+vec(3)*vec(3))
           !
           qmmm_ewald_c%rijdata_qmqm(1,icnt_qm) = Rij      ! rij value
           qmmm_ewald_c%rijdata_qmqm(2,icnt_qm) = one/Rij  ! one/rij value.
        end do
     end do
  end if

  if(mm_main_c%rij_mm_incore) then
     icnt_mm = 0
     do i = 1, numat
        xyz_i(1:3) = qm_coord(1:3,i)
        do j = mstart,mstop     ! 1,numatm
           icnt_mm = icnt_mm + 1
           vec(1:3) = xyz_i(1:3)-mm_coord(1:3,j)
           Rij      = sqrt(vec(1)*vec(1)+vec(2)*vec(2)+vec(3)*vec(3))
           !
           qmmm_ewald_c%rijdata_qmmm(1,icnt_mm) = Rij      ! rij value
           qmmm_ewald_c%rijdata_qmmm(2,icnt_mm) = one/Rij  ! one/rij value
        end do
     end do
  end if

  return
  end subroutine fill_dist_qm_mm_array

  !=====================================================================
  subroutine qmmm_Ewald_init(natom,numat,erfmod,igmsel,            &
                             kmaxX,kmaxY,kmaxZ,KSQmax,kappa,       &
                             qcheck)
  !=====================================================================
  ! 
  ! initial setup for qm/mm-ewald calculations.
  ! 
  use qm1_info, only: qm_control_c,qm_main_c,mm_main_c,qm_scf_main_c
  use qmmmewald_module, only : qmmm_ewald_c,qm_ewald_setup,        &
                               allocate_deallocate_qmmm_ewald,     &
                               set_initialize_for_energy_gradient
  use number, only : zero
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none
  integer :: natom,numat,erfmod,kmaxX,kmaxY,kmaxZ,KSQmax
  integer :: igmsel(natom)
  real(chm_real):: kappa
  logical :: qcheck,q_ewald_call
  !
  integer :: i,nexcl
  integer :: ier=0
#if KEY_PARALLEL==1
  integer :: ntmp_iatotl(numnod)
#endif

  ! quick check.
  if(.not.mm_main_c%LQMEWD) return

  ! setup for qm/mm-ewald summation
  qmmm_ewald_c%natom  = Natom
  qmmm_ewald_c%kmaxqx = kmaxX
  qmmm_ewald_c%kmaxqy = kmaxY
  qmmm_ewald_c%kmaxqz = kmaxZ
  qmmm_ewald_c%ksqmaxq= KSQmax
  qmmm_ewald_c%Erfmod = erfmod
  qmmm_ewald_c%kappa  = kappa

  ! for parallel preparation
#if KEY_PARALLEL==1
  qmmm_ewald_c%iastrt = natom*mynod/numnod+1
  qmmm_ewald_c%iafinl = natom*(mynod+1)/numnod
  qmmm_ewald_c%iatotl = qmmm_ewald_c%iafinl - qmmm_ewald_c%iastrt + 1
  !
  ! check and iatotl is maximum over the entire node. so, memory allocation over the 
  ! node are same, but smaller than numnod=1.)
  !if(numnod > 1) then
  !   ntmp_iatotl = 0
  !   ntmp_iatotl(mynod+1)=qmmm_ewald_c%iatotl
  !   call igcomb(ntmp_iatotl,numnod)
  !   do i=1,numnod
  !      if(ntmp_iatotl(i) > qmmm_ewald_c%iatotl) qmmm_ewald_c%iatotl=ntmp_iatotl(i)
  !   end do
  !end if
#else
  qmmm_ewald_c%iastrt = 1          ! starting of do i=1,natom loop
  qmmm_ewald_c%iafinl = natom      ! ending of the loop
  qmmm_ewald_c%iatotl = natom      ! total term in do-loop
#endif


  ! check how many MM atoms are excluded from QM-MM non-bonded interactions.
  nexcl = 0
  do i=1,natom
     if(igmsel(i).eq.5) nexcl=nexcl+1
  end do
  qmmm_ewald_c%nexl_atm =nexcl

  ! now setup the rest.
  call qm_ewald_setup(qmmm_ewald_c%kmaxqx,qmmm_ewald_c%kmaxqy,qmmm_ewald_c%kmaxqz, &
                      qmmm_ewald_c%ksqmaxq,qmmm_ewald_c%totkq,qcheck)
  if(.not.Qcheck) return

  ! allocate memory
  ! provide
  call allocate_deallocate_qmmm_ewald(qm_main_c%numat,mm_main_c%natom,mm_main_c%PMEwald, &
                                      qmmm_ewald_c,.true.)

  ! now, fill exclusion list array.
  if(nexcl.gt.0) then
     nexcl = 0
     do i=1,natom
        if(igmsel(i).eq.5) then
           nexcl=nexcl+1
           qmmm_ewald_c%nexl_index(nexcl)=i
        end if
     end do
  end if

  ! do initialization
  !qmmm_ewald_c%scf_mchg   = zero
  !qmmm_ewald_c%scf_mchg_2 = zero
  qmmm_ewald_c%Kvec       = zero
  qmmm_ewald_c%structfac_mm=zero
  qmmm_ewald_c%empot      = zero
  qmmm_ewald_c%eslf       = zero
  qmmm_ewald_c%dexl_xyz   = zero
  !
  ! do some other initializations: Ktable,qmktable,d_ewald_mm
  !qmmm_ewald_c%Ktable     = zero
  !qmmm_ewald_c%qmktable   = zero
  !qmmm_ewald_c%d_ewald_mm = zero
  q_ewald_call =.true.
  call set_initialize_for_energy_gradient(q_ewald_call)

  return
  end subroutine qmmm_Ewald_init


  !=====================================================================
  subroutine qmmm_Ewald_setup_and_potential(volume,recip,x,y,z,cg,qcheck)
  !=====================================================================
  ! 
  ! do the qm/mm-ewald setup: 1) prepare K-vector and K-tables.
  !                           2) compute the ewald potential on qm atom sites.
  ! 
  use qm1_info, only : qm_control_c,qm_main_c,mm_main_c,qm_scf_main_c,qm_gho_info_c
  use qmmmewald_module,only: qm_ewald_calc_kvec,qm_ewald_calc_ktable,qm_ewald_mm_pot, &
                             qm_ewald_qm_pot,qm_ewald_mm_pot_exl,                     &
                             qmmm_ewald_c,get_exl_crd,set_initialize_for_energy_gradient
  !use qmmmpme_module,only : qm_pme_mm_pot
  !use qm1_scf_module,only : calc_mulliken,q_construct
  use number, only : zero,half
  use chm_kinds
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none

  ! passed in
  real(chm_real),intent(in) :: volume,recip(6)
  real(chm_real),intent(in) :: x(*),y(*),z(*)
  real(chm_real),intent(inout) :: cg(*)
  logical :: qcheck

  ! local variables
  integer ::itotal,itotkq,i,nexcl
  logical :: QNoPMEwald,q_ewald_call
  integer, save :: old_N = 0

  QNoPMEwald = .not. mm_main_c%PMEwald

! commented out the following, as it is not using QMPI (probably used for MPI-PI calc.
!!!!! very important.
!!! do special with QMPI: if you once turn on this, cannot turn off later!!!!
!!!#if KEY_PARALLEL==1
!!!  if (QMPI) then
!!!     if(qmmm_ewald_c%iatotl .ne. qmmm_ewald_c%natom) then
!!!        qmmm_ewald_c%iastrt = 1
!!!        qmmm_ewald_c%iafinl = qmmm_ewald_c%natom
!!!        qmmm_ewald_c%iatotl = qmmm_ewald_c%natom
!!!        ! now, memory allocation.
!!!        if(allocated(qmmm_ewald_c%Ktable)) deallocate(qmmm_ewald_c%Ktable)
!!!        if(QNoPMEwald) then               ! iatotl=natom
!!!           allocate(qmmm_ewald_c%Ktable(6,qmmm_ewald_c%iatotl,qmmm_ewald_c%totkq),stat=ier) 
!!!        else
!!!           allocate(qmmm_ewald_c%Ktable(6,1,1),stat=ier)       ! dummy allocation
!!!        end if
!!!        if(ier.ne.0) call Aass(1,'qmmm_Ewald_setup_and_potential','Ktable')
!!!     end if
!!!  end if
!!!#if KEY_PARALLEL==1

  ! initialization (the following will be done in the subroutine.)
  !qmmm_ewald_c%Ktable     = zero
  !qmmm_ewald_c%qmktable   = zero
  !if(QNoPMEwald) then
  !   qmmm_ewald_c%d_ewald_mm = zero
  !end if
  !qmmm_ewald_c%d_ewald_mm = zero
  if(QNoPMEwald) then
     q_ewald_call=.true.
  else
     q_ewald_call=.false.
  end if
  ! note: qmmm_ewald_c%d_ewald_mm is initialized in KSPACE_qmmm_prep in used of PMEwald.
  call set_initialize_for_energy_gradient(q_ewald_call)

  ! for ktable memory size
  if(QNoPMEwald) then
     itotal=qmmm_ewald_c%iatotl
     itotkq=qmmm_ewald_c%totkq
  else
     itotal=1
     itotkq=1
  end if

  ! copy volumne and reciprocal space lattice vector. They will be copied at
  ! every step as they change during dynamics.
  qmmm_ewald_c%volume     = volume
  qmmm_ewald_c%recip(1:6) = recip(1:6)

  ! for exclusion list: copy x,y,z,cg to qmmm_ewald_c%exl_xyz and qmmm_ewald_c%exl_chg.
  if(qmmm_ewald_c%nexl_atm > 0) call get_exl_crd(x,y,z,cg)

  ! 1) Kvector setup
  ! if recip and volume do not change, maybe skipped.  should check the possibility.
  ! in fact, it is done inside of the subroutine.
  call qm_ewald_calc_kvec(qmmm_ewald_c%kappa, qmmm_ewald_c%Volume, qmmm_ewald_c%Recip,  &
                          qmmm_ewald_c%totkq, qmmm_ewald_c%ksqmaxq,                     &
                          qmmm_ewald_c%kmaxqx,qmmm_ewald_c%kmaxqy, qmmm_ewald_c%kmaxqz, &
                          QNoPMEwald,qcheck)
  if(.not.qcheck) return  ! check: if wrong, CHARMM should stop.

  ! 2) Ktables setup
  call qm_ewald_calc_ktable(qmmm_ewald_c%natom, qm_main_c%numat,    qm_control_c%qminb, &
                            qmmm_ewald_c%iastrt,qmmm_ewald_c%iafinl,qmmm_ewald_c%iatotl,&
                            itotal,itotkq,                                              &
                            qmmm_ewald_c%totkq, qmmm_ewald_c%ksqmaxq,                   &
                            qmmm_ewald_c%kmaxqx,qmmm_ewald_c%kmaxqy,qmmm_ewald_c%kmaxqz,&
                            X,Y,Z,qmmm_ewald_c%recip,QNoPMEwald,qcheck)
  if(.not.qcheck) return  ! Check: if wrong, CHARMM should stop.

  ! 3) Ewald potential setup:
  !    compute the Ewald potential at the qm atom site from all MM atoms; the
  !    self-interaction/ real space contribution / reciprocal space contribution
  !    from "pure" mm atoms.
  !
  ! Note: in parallel run, empot is combined over node from the subroutine as
  !       explained below. so, each node has the same information in the end.
  call qm_ewald_mm_pot(qmmm_ewald_c%natom, qm_main_c%numat,    mm_main_c%numatm,    &
                       qmmm_ewald_c%iastrt,qmmm_ewald_c%iafinl,qmmm_ewald_c%iatotl, &
                       itotal,itotkq,                                               &
                       qmmm_ewald_c%totkq, qmmm_ewald_c%ksqmaxq,                    &
                       qmmm_ewald_c%kmaxqx,qmmm_ewald_c%kmaxqy,qmmm_ewald_c%kmaxqz, &
                       mm_main_c%mm_coord, mm_main_c%mm_chrgs, qm_main_c%qm_coord,  &
                       qmmm_ewald_c%empot,QNoPMEwald)
!  ! 3-1) the pme version
!  if(.not.QNoPMEwald) then
!     !! since before mndene call, the pme potential is evaluated. So, this routine now
!     !! will do only the summing up of empot values. Do it here explicitly.
!     !!call qm_pme_mm_pot(qmmm_ewald_c%natom,qm_main_c%numat,x,y,z,cg,mm_main_c%qm_charges, &
!     !!                   qmmm_ewald_c%recip,qmmm_ewald_c%volume,qmmm_ewald_c%empot,        &
!     !!                   qmmm_ewald_c%empot_pme,qmmm_ewald_c%empot_qm_pme,                 &
!     !!                   qmmm_ewald_c%kappa)
! see this summation below in the scf_energy routine.
!     qmmm_ewald_c%empot(1:qm_main_c%numat) = qmmm_ewald_c%empot(1:qm_main_c%numat) + &
!                                             qmmm_ewald_c%empot_pme(1:qm_main_c%numat)
!
!#if KEY_PARALLEL==1
!     if (numnod.gt.1) then
!        call GCOMB(qmmm_ewald_c%empot_pme,qm_main_c%numat)     ! qm-mm part
!        call GCOMB(qmmm_ewald_c%empot_qm_pme,qm_main_c%numat)  ! qm-qm part (energy should be 1/2.)
!     end if
!#endif
!  end if

  ! 4) for the qm-mm excluded list
  if(qmmm_ewald_c%nexl_atm > 0) &
     call qm_ewald_mm_pot_exl(qm_main_c%numat,qm_main_c%qm_coord,qmmm_ewald_c%empot) 
! see below scf_energy routine.
!#if KEY_PARALLEL==1
!  ! sum up values over all nodes.
!  if (numnod > 1) call GCOMB(qmmm_ewald_c%empot,qm_main_c%numat)
!#endif

  if(.not.Qcheck) return
  ! here, empot should be done! (also communicated between each node).

  ! 5) for self qm-qm (image) atoms
  call qm_ewald_qm_pot(qmmm_ewald_c%natom,  qm_main_c%numat,    qmmm_ewald_c%totkq,&
                       qmmm_ewald_c%ksqmaxq,qmmm_ewald_c%kmaxqx,                   &
                       qmmm_ewald_c%kmaxqy, qmmm_ewald_c%kmaxqz,                   &
                       qm_main_c%qm_coord,  qmmm_ewald_c%eslf)

  ! end of qm/mm-ewald potential part.
  !
  return
  end subroutine qmmm_Ewald_setup_and_potential


  !=====================================================================
  subroutine scf_energy(natom,xim,yim,zim,icall,qfirst)
  !=====================================================================
  ! 
  ! Compute energy of QM+QM/MM (electrostatic) part.
  !
  use qm1_info, only : qm_control_c,qm_main_c,mm_main_c,qm_scf_main_c,qm_param_c, &
                       qm_scf_diis_c,qm_fockmd_diis_c,dxlbomd_c,qm_gho_info_c,    &
                       allocate_deallocate_qm_diis
  use qm1_scf_module, only : scf_iter,bond_analysis
  use qmmmewald_module ,only : qmmm_ewald_c,qm_ewald_core,qm_pme_energy_corr
  use qm1_energy_module, only : hcorep,mmint,guessp,wstore,wstore_comm,iqm_mode,do_am1_pm3,do_d_orbitals
  use number, only : zero,half,two
  use qm1_constant,only: EVCAL,ccelec
  use contrl,only : ISTEPQM
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none
  integer :: natom,icall
  real(chm_real) :: xim(natom),yim(natom),zim(natom)
  logical :: qfirst
  
  ! local variables
  integer :: dim_iwork,dim_iwork6
  !logical,save :: qfirst=.true.  ! pass from the parent routine.

  integer :: i,k,ks,ii
  ! debug
  real(chm_real):: e_temp
  integer       :: msize,mstart,mstop
  integer, save :: ifockmd_counter=0,imd_counter=0
#if KEY_PARALLEL==1
  integer :: JPARPT_local(0:MAXNODE)
#endif
  ! 
  ! MNDO two-electron integrals, used as a local array
  real(chm_real),allocatable:: w_linear(:)

  !sizes
#if KEY_PARALLEL==1
  JPARPT_local(0) = 0
  do i=1,numnod
     JPARPT_local(i)= mstop*i/numnod  ! for linear vector
  end do
  mstart = JPARPT_local(mynod)+1
  mstop  = JPARPT_local(mynod+1)
  msize  = mstop - mstart + 1
#else
  mstart = 1
  mstop  = qm_scf_main_c%dim_linear_norbs
  msize  = mstop - mstart + 1
#endif

  ! initialization
  qm_main_c%enuclr_qmqm = zero
  qm_main_c%enuclr_qmmm = zero
  qm_scf_main_c%H(1:qm_scf_main_c%dim_linear_norbs) = zero
  qm_scf_main_c%W(1:qm_scf_main_c%dim_linear_fock2) = zero

  ! variables for qm1_energy_module
  iqm_mode     = qm_control_c%iqm_mode
  do_am1_pm3   = qm_control_c%q_am1_pm3
  do_d_orbitals= qm_control_c%do_d_orbitals

  ! integral calculations for scf.
  call hcorep(qm_scf_main_c%H,qm_scf_main_c%W,qm_scf_main_c%dim_linear_norbs,&
              qm_scf_main_c%dim_linear_fock,qm_scf_main_c%dim_linear_fock2,  &
              qm_main_c%enuclr_qmqm)

  ! for qm/mm
  if(mm_main_c%numatm.gt.0) call mmint(qm_scf_main_c%H, &
                                       qm_scf_main_c%dim_linear_norbs,  &
                                       qm_main_c%enuclr_qmmm)

  ! store MNDO integras in square form: 
  ! 1. complete MNDO integrals and transpose the W matrix.
  ! 2. this is doing the leftover from wstore subroutine in hcorep routine.
  ! 3. w_linear contains lower-trianglular part of the W matrix.
  if(allocated(w_linear)) deallocate(w_linear)
  allocate(w_linear(qm_scf_main_c%dim_linear_fock*(qm_scf_main_c%dim_linear_fock+1)/2))
  call wstore(qm_scf_main_c%W,w_linear,qm_scf_main_c%dim_linear_fock,0, &
              qm_main_c%numat,qm_main_c%uhf)

  ! do communication here.
#if KEY_PARALLEL==1
  if (numnod > 1) then
     ! H: 1-e matrix
     ! W: 2-e exchange/replusion integrals. (see below)
     call gcomb(qm_scf_main_c%H,qm_scf_main_c%dim_linear_norbs)
     call gcomb(w_linear,qm_scf_main_c%dim_linear_fock*(qm_scf_main_c%dim_linear_fock+1)/2)
     call gcomb(qm_main_c%enuclr_qmqm,1)

     !!!call gcomb(qm_scf_main_c%W,qm_scf_main_c%dim_linear_fock2) ! this is done within wstore
     !!!call gcomb(qm_main_c%enuclr_qmmm,1) : see below after adding qm_ewald_core.
  end if
#endif
  call wstore_comm(qm_scf_main_c%W,w_linear,qm_scf_main_c%dim_linear_fock)

  ! for guess density
  if(qfirst) then
     qfirst=.false.
     if(qm_main_c%uhf) then
        call guessp(qm_scf_main_c%PA,qm_scf_main_c%PB,qm_scf_main_c%dim_linear_norbs)
     else
        call guessp(qm_scf_main_c%PA,qm_scf_main_c%PA,qm_scf_main_c%dim_linear_norbs)
     end if
  end if

  ! PME related information (data communication): sum up values over all nodes.
  ! (see above qmmm_Ewald_setup_and_potential routine.)
  if(mm_main_c%PMEwald) qmmm_ewald_c%empot(1:qm_main_c%numat) = qmmm_ewald_c%empot(1:qm_main_c%numat) + &
                                             qmmm_ewald_c%empot_pme(1:qm_main_c%numat)
#if KEY_PARALLEL==1
  if (numnod > 1 .and. mm_main_c%LQMEWD) then
     if(mm_main_c%PMEwald) then
        call GCOMB(qmmm_ewald_c%empot_pme,qm_main_c%numat)     ! qm-mm part
        call GCOMB(qmmm_ewald_c%empot_qm_pme,qm_main_c%numat)  ! qm-qm part (energy should be 1/2.)
     end if
     call GCOMB(qmmm_ewald_c%empot,qm_main_c%numat)
     call gcombr(qmmm_ewald_c%eslf,qm_main_c%numat*qm_main_c%numat) ! (see qm_ewald_qm_pot routine).
  end if
#endif

  ! memory deallocation
  if(allocated(w_linear)) deallocate(w_linear)

  !
  if(qm_control_c%q_dxl_bomd .and. qm_control_c%md_run) then
     ! first memory.
     if(.not. allocated(dxlbomd_c%pa_exp)) then
        allocate(dxlbomd_c%pa_exp(dxlbomd_c%Kth_sum_order,msize))
        allocate(dxlbomd_c%pa_aux(msize))
        allocate(dxlbomd_c%pa_anew(msize))
        dxlbomd_c%pa_exp(1:dxlbomd_c%Kth_sum_order,1:msize) = zero
        imd_counter = 0   ! reset counter.
     end if
     imd_counter = imd_counter + 1

     ! now, propagate pa_aux and copy it to pa init.
     if(imd_counter > dxlbomd_c%Kth_sum_order) then
        qm_control_c%q_do_dxl_scf =.true.  ! do dxl-bomd scf cycle. (finish at N_scf_step.)
        ks                        = dxlbomd_c%Kth_sum_order
        ! determine pa_aux(i+1) = 2*pa_aux(i) - pa_aux(i-1) + kappa*(PA_old(i)-pa_aux(i)) + alpha*sum over k=0,K
        ! and copy to PA_initial = pa_aux(i+1)
        do i=1,msize
           dxlbomd_c%pa_anew(i) = dxlbomd_c%coefk*qm_scf_main_c%PA(mstart+i-1) + &
                                  dxlbomd_c%cextr(0)*dxlbomd_c%pa_aux(i)       + &
                                  dot_product(dxlbomd_c%cextr(1:ks),dxlbomd_c%pa_exp(1:ks,i))
           ! copy
           qm_scf_main_c%PA(mstart+i-1) = dxlbomd_c%pa_anew(i)
        end do
#if KEY_PARALLEL==1
        if(numnod>1) call VDGBRE(qm_scf_main_c%PA,JPARPT_local)
#endif
     else
        qm_control_c%q_do_dxl_scf =.false.  ! do full scf cycle.
     end if
  end if

  !
  if(qm_control_c%q_fockmd .and. qm_control_c%md_run) then
     if(.not.qm_control_c%q_do_fockmd_scf) ifockmd_counter = 0 ! initialization.
     ifockmd_counter = ifockmd_counter + 1
     qm_control_c%q_do_fockmd_scf=.true.
  else
     qm_control_c%q_do_fockmd_scf=.false.
  end if

  ! now, call scf_iteration
  icall = 0
  dim_iwork =9*qm_scf_main_c%dim_numat
  dim_iwork6=max(6*dim_iwork,qm_main_c%norbs)
  if(qm_main_c%uhf) then
     ! for UHF
     call scf_iter(qm_main_c%elec_eng,qm_scf_main_c%H,qm_scf_main_c%W, &
                   qm_scf_main_c%Q,                                    &
                   qm_scf_main_c%CA,qm_scf_main_c%DA,qm_scf_main_c%EA, &
                   qm_scf_main_c%FA,qm_scf_main_c%PA,                  &
                   qm_scf_main_c%CB,qm_scf_main_c%DB,qm_scf_main_c%EB, &
                   qm_scf_main_c%FB,qm_scf_main_c%PB,                  &
                   qm_scf_main_c%dim_numat,                            &
                   qm_scf_main_c%dim_norbs,qm_scf_main_c%dim_linear_norbs,  &
                   qm_scf_main_c%dim_linear_fock,                      &
                   qm_scf_main_c%dim_linear_fock2,qm_scf_main_c%dim_scratch,ifockmd_counter, & 
                   dim_iwork6,qm_scf_main_c%iwork,icall,qm_main_c%uhf)
  else
     ! for RHF
     call scf_iter(qm_main_c%elec_eng,qm_scf_main_c%H,qm_scf_main_c%W, &
                   qm_scf_main_c%Q,                                    &
                   qm_scf_main_c%CA,qm_scf_main_c%DA,qm_scf_main_c%EA, &
                   qm_scf_main_c%FA,qm_scf_main_c%PA,                  &
                   qm_scf_main_c%CA,qm_scf_main_c%DA,qm_scf_main_c%EA, &
                   qm_scf_main_c%FA,qm_scf_main_c%PA,                  &
                   qm_scf_main_c%dim_numat,                            &
                   qm_scf_main_c%dim_norbs,qm_scf_main_c%dim_linear_norbs,  &
                   qm_scf_main_c%dim_linear_fock,                      &
                   qm_scf_main_c%dim_linear_fock2,qm_scf_main_c%dim_scratch,ifockmd_counter, &
                   dim_iwork6,qm_scf_main_c%iwork,icall,qm_main_c%uhf)
  end if

  ! if scf failes, do 2nd attempt with
  ! (a) diis extrapolation on
  ! (b) new diagonal density guess
  ! (c) no other density extrapolation.
  if(icall == -1 .and. .not.qm_control_c%q_diis) then
     ! turn on diis
     qm_control_c%q_diis=.true.
     ! if so, allocate memory for diis?
     if(.not. allocated(qm_scf_diis_c%FDA)) &
        call allocate_deallocate_qm_diis(qm_scf_main_c,qm_scf_diis_c,qm_fockmd_diis_c, &
                                         qm_control_c%q_diis,qm_control_c%q_fockmd,    &
                                         qm_control_c%imax_fdiss,                      &
                                         qm_main_c%uhf,.true.)

     ! in previous scf_iter returned with icall=-1, so do not set icall = 0.
     if(qm_main_c%uhf) then
        ! for UHF
        call guessp(qm_scf_main_c%PA,qm_scf_main_c%PB,qm_scf_main_c%dim_linear_norbs)
        call scf_iter(qm_main_c%elec_eng,qm_scf_main_c%H,qm_scf_main_c%W, &
                      qm_scf_main_c%Q,                                    &
                      qm_scf_main_c%CA,qm_scf_main_c%DA,qm_scf_main_c%EA, &
                      qm_scf_main_c%FA,qm_scf_main_c%PA,                  &
                      qm_scf_main_c%CB,qm_scf_main_c%DB,qm_scf_main_c%EB, &
                      qm_scf_main_c%FB,qm_scf_main_c%PB,                  &
                      qm_scf_main_c%dim_numat,                            &
                      qm_scf_main_c%dim_norbs,qm_scf_main_c%dim_linear_norbs,  &
                      qm_scf_main_c%dim_linear_fock,                      &
                      qm_scf_main_c%dim_linear_fock2,qm_scf_main_c%dim_scratch,ifockmd_counter, &
                      dim_iwork6,qm_scf_main_c%iwork,icall,qm_main_c%uhf)
     else
        ! for RHF
        call guessp(qm_scf_main_c%PA,qm_scf_main_c%PA,qm_scf_main_c%dim_linear_norbs)
        call scf_iter(qm_main_c%elec_eng,qm_scf_main_c%H,qm_scf_main_c%W, &
                      qm_scf_main_c%Q,                                    &
                      qm_scf_main_c%CA,qm_scf_main_c%DA,qm_scf_main_c%EA, &
                      qm_scf_main_c%FA,qm_scf_main_c%PA,                  &
                      qm_scf_main_c%CA,qm_scf_main_c%DA,qm_scf_main_c%EA, &
                      qm_scf_main_c%FA,qm_scf_main_c%PA,                  &
                      qm_scf_main_c%dim_numat,                            &
                      qm_scf_main_c%dim_norbs,qm_scf_main_c%dim_linear_norbs,  &
                      qm_scf_main_c%dim_linear_fock,                      &
                      qm_scf_main_c%dim_linear_fock2,qm_scf_main_c%dim_scratch,ifockmd_counter, &
                      dim_iwork6,qm_scf_main_c%iwork,icall,qm_main_c%uhf)
     end if
  end if

  ! contributions from the Ewald sum-core interaction, only after scf converges.
  if(mm_main_c%LQMEWD) then
     qm_main_c%enuclr_qmmm = qm_main_c%enuclr_qmmm &
                            +qm_ewald_core(qm_main_c%numat,qm_param_c%core, &
                                           mm_main_c%qm_charges)
  end if

  !
#if KEY_PARALLEL==1
  if (numnod.gt.1) call gcomb(qm_main_c%enuclr_qmmm,1)
#endif

  ! for analysis
  if((qm_control_c%q_bond_order .or. qm_control_c%q_m_charge) .and. .not. qm_main_c%uhf) then
     call bond_analysis(qm_scf_main_c%dim_numat,qm_scf_main_c%PA,             &
                        qm_scf_main_c%dim_norbs,qm_scf_main_c%dim_linear_fock)
  end if

  ! for DXL-BOMD
  if(qm_control_c%q_dxl_bomd .and. qm_control_c%md_run) then
     if(imd_counter <= dxlbomd_c%Kth_sum_order) then
        ! k=0; pa_aux(i) = PA_scf(i), etc.
        if(imd_counter == 1) dxlbomd_c%pa_aux(1:msize) = qm_scf_main_c%PA(mstart:mstop) ! copy old pa.
        do ii=1,msize
           do k = imd_counter,2,-1
              dxlbomd_c%pa_exp(k,ii) = dxlbomd_c%pa_exp(k-1,ii)
           end do
           dxlbomd_c%pa_exp(1,ii) = dxlbomd_c%pa_aux(ii)
        end do
     else
        ! shift array by one column to update to the current auxiliary PA arrays.
        do i=1,msize
           do k = dxlbomd_c%Kth_sum_order,2,-1
              dxlbomd_c%pa_exp(k,i) = dxlbomd_c%pa_exp(k-1,i)
           end do
           dxlbomd_c%pa_exp(1,i) = dxlbomd_c%pa_aux(i)           ! copy old pa_aux.
           dxlbomd_c%pa_aux(i)   = dxlbomd_c%pa_anew(i)          ! copy current pa_aux to old pa_aux.
        end do
     end if
  end if

  ! save energy
  ! note: qm_main_c%ener_atomic is already in kcal/mol (qm_info_setup)
#if KEY_PARALLEL==1
  if(mynod == 0) then
#endif
     qm_control_c%E_scf    = EVCAL*qm_main_c%elec_eng
     qm_control_c%E_nuclear= EVCAL*(qm_main_c%enuclr_qmqm+qm_main_c%enuclr_qmmm)
     qm_control_c%E_total  = qm_main_c%HofF_atomic     &
                            +qm_control_c%E_scf        &
                            +qm_control_c%E_nuclear    &
                            -qm_main_c%ener_atomic
#if KEY_PARALLEL==1
  else
     qm_control_c%E_total  = zero
  end if
  ! for debug
  !if(mynod == 0) then
#endif
  !  write(6,*)'e_scf=',qm_main_c%elec_eng
  !  write(6,*)'E_nuclear=',qm_main_c%enuclr_qmqm+qm_main_c%enuclr_qmmm
  !  write(6,*)'HofF_atomic=',qm_main_c%HofF_atomic
  !  write(6,*)'ener_atomic=',qm_main_c%ener_atomic
  !  write(6,*)'e_total=',ISTEPQM,qm_control_c%E_total
#if KEY_PARALLEL==1
  !endif
#endif
  !
  return
  end subroutine scf_energy


  !=====================================================================
  subroutine scf_gradient(natom,xim,yim,zim,dx,dy,dz)
  !=====================================================================
  ! 
  ! Compute energy of QM+QM/MM (electrostatic) gradient part.
  ! gradients are computed as a finite difference of energy.
  !
  use qm1_info, only : qm_control_c,qm_main_c,mm_main_c,qm_scf_main_c,qm_param_c, &
                       qm_gho_info_r
  use qm1_gradient_module, only : qmqm_gradient,qmmm_gradient
  use mndgho_module, only : GHO_expansion
  use number, only : zero
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none
  integer :: natom,i,n
  real(chm_real):: xim(natom),yim(natom),zim(natom),dx(natom),dy(natom),dz(natom)
  real(chm_real):: PL,PM
  integer :: mstart,mend

  !
#if KEY_PARALLEL==1
  mstart = mm_main_c%numatm*mynod/numnod + 1
  mend   = mm_main_c%numatm*(mynod+1)/numnod
#else
  mstart = 1
  mend   = mm_main_c%numatm
#endif

  ! 
  ! initialization
  qm_main_c%qm_grads(1:3,1:qm_main_c%numat) = zero
  if(mm_main_c%numatm > 0) mm_main_c%mm_grads(1:3,mstart:mend) = zero
  
  ! for switching function
  if(mm_main_c%q_switch) mm_main_c%dxyz_sw(1:6,1:mm_main_c%isize_swt_array) =zero
  !
  ! for regular full scf calculations.
  if(qm_main_c%uhf) then
     ! for qm-qm gradient components
     call qmqm_gradient(qm_scf_main_c%PA,qm_scf_main_c%PB,qm_scf_main_c%dim_linear_norbs)  

     ! for qm-mm gradient components
     if(mm_main_c%numatm > 0) call qmmm_gradient(qm_scf_main_c%PA,qm_scf_main_c%PB,       &
                                                 natom,xim,yim,zim,                       &
                                                 qm_scf_main_c%dim_linear_norbs,          &
                                                 qm_main_c%numat,mm_main_c%numatm,        &
                                                 qm_scf_main_c%INDX,                      &
                                                 qm_main_c%qm_coord,mm_main_c%mm_coord,   &
                                                 mm_main_c%mm_chrgs,                      &
                                                 qm_scf_main_c%CORE_mat,qm_scf_main_c%WW, &
                                                 qm_scf_main_c%RI,qm_scf_main_c%YY,       &
                                                 qm_main_c%qm_grads,mm_main_c%mm_grads,   &
                                                 qm_control_c%q_am1_pm3,mstart,mend)
  else
     ! for qm-qm gradient components
     call qmqm_gradient(qm_scf_main_c%PA,qm_scf_main_c%PA,qm_scf_main_c%dim_linear_norbs)

     ! for qm-mm gradient components
     if(mm_main_c%numatm > 0) call qmmm_gradient(qm_scf_main_c%PA,qm_scf_main_c%PA,       &
                                                 natom,xim,yim,zim,                       &
                                                 qm_scf_main_c%dim_linear_norbs,          &
                                                 qm_main_c%numat,mm_main_c%numatm,        &
                                                 qm_scf_main_c%INDX,                      &
                                                 qm_main_c%qm_coord,mm_main_c%mm_coord,   &
                                                 mm_main_c%mm_chrgs,                      &
                                                 qm_scf_main_c%CORE_mat,qm_scf_main_c%WW, &
                                                 qm_scf_main_c%RI,qm_scf_main_c%YY,       &
                                                 qm_main_c%qm_grads,mm_main_c%mm_grads,   &
                                                 qm_control_c%q_am1_pm3,mstart,mend)
  end if
  ! done the gradient part from the qm-qm and qm-mm interactions.

  ! now, copying gradient into the charmm main dx/dy/dz arrays.

!!  ! this may not be needed as other options not used.
!!  if(mm_main_c%LQMEWD .and. mm_main_c%NQMEWD.ne.1) then
!!     !!! it's done: call gcomb(mm_main_c%qm_charges,qm_main_c%numat) 
!!     do i=1,qm_main_c%numat
!!        n=qm_control_c%qminb(i)
!!        qm_control_c%cgqmmm(n) = mm_main_c%qm_charges(i)
!!     end do
!!  end if

  ! it should apply even it is parallel
  call get_qmmm_gradient(natom,qm_main_c%numat,mm_main_c%numatm, &
                         qm_control_c%qminb,mm_main_c%qm_mm_pair_list, &
                         dx,dy,dz,qm_main_c%qm_grads,mm_main_c%mm_grads)
  !
  !do i=1,qm_main_c%numat
  !   n=qm_control_c%qminb(i)
  !   dx(n)=dx(n)+qm_main_c%qm_grads(1,i)
  !   dy(n)=dy(n)+qm_main_c%qm_grads(2,i)
  !   dz(n)=dz(n)+qm_main_c%qm_grads(3,i)
  !end do
  !! for MM atoms.
  !if(mm_main_c%numatm.gt.0) then
  !   do i=1,mm_main_c%NUMATM
  !      n=mm_main_c%qm_mm_pair_list(i)
  !      dx(n)=dx(n)+mm_main_c%mm_grads(1,i)
  !      dy(n)=dy(n)+mm_main_c%mm_grads(2,i)
  !      dz(n)=dz(n)+mm_main_c%mm_grads(3,i)
  !   end do
  !end if

  ! now for switching function contributions.
  if(mm_main_c%q_switch) call put_switching_gradient(natom,mm_main_c%NUMATM,dx,dy,dz)

  return

  contains
     subroutine get_qmmm_gradient(natom,numat,numatm,qminb,pair_list, &
                                  dx,dy,dz,qm_grads,mm_grads)
     !
     ! copy gradient info to the main gradient arrays.
     !
     use chm_kinds
     implicit none
     integer :: natom,numat,numatm
     integer :: qminb(*),pair_list(*)
     real(chm_real):: dx(natom),dy(natom),dz(natom), &
                      qm_grads(3,numat),mm_grads(3,natom)
     integer :: i,n

     do i=1,numat
        n=qminb(i)
        dx(n)=dx(n)+qm_grads(1,i)
        dy(n)=dy(n)+qm_grads(2,i)
        dz(n)=dz(n)+qm_grads(3,i)
     end do
     ! for MM atoms.
     do i=mstart,mend  ! 1,NUMATM
        n=pair_list(i)
        dx(n)=dx(n)+mm_grads(1,i)
        dy(n)=dy(n)+mm_grads(2,i)
        dz(n)=dz(n)+mm_grads(3,i)
     end do

     return
     end subroutine get_qmmm_gradient
     !
  end subroutine scf_gradient


  !=====================================================================
  subroutine qmmm_Ewald_gradient(x,y,z,dx,dy,dz,cg,virial,qcheck)
  !=====================================================================
  ! 
  ! do the qm/mm-ewald setup: 1) prepare K-vector and K-tables.
  !                           2) compute the ewald potential on qm atom sites.
  ! 
  use qm1_info, only: qm_control_c,qm_main_c,mm_main_c,qm_scf_main_c,qm_gho_info_c
  use qmmmewald_module,only: qm_ewald_real_space_gradient,qm_ewald_real_space_gradient_exl, &
                            qm_ewald_recip_space_gradient,   &
                            qmmm_ewald_c
  use qmmmpme_module,only: qm_pme_mm_grad
  use number, only : zero
  use chm_kinds
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none
  real(chm_real):: x(*),y(*),z(*),dx(*),dy(*),dz(*),cg(*),virial(9)
  logical :: qcheck

  ! logical variables
  integer :: i,m,nexcl,itotal,itotkq
  integer :: mstart,mstop
  logical :: QNoPMEwald

  QNoPMEwald = .not.mm_main_c%PMEwald

#if KEY_PARALLEL==1
  mstart = mm_main_c%NUMATM*mynod/numnod + 1
  mstop  = mm_main_c%NUMATM*(mynod+1)/numnod
#else
  mstart = 1
  mstop  = mm_main_c%NUMATM
#endif
  ! for ktable memory size
  if(QNoPMEwald) then
     itotal=qmmm_ewald_c%iatotl
     itotkq=qmmm_ewald_c%totkq
  else
     itotal=1
     itotkq=1
  end if

  ! initialization
  ! as qm_grads and mm_grads are already copied into the main dx/dy/dz arrays in
  ! scf_gradient routine, here we can use the array for the present purpose.
  qm_main_c%qm_grads2      = zero
  mm_main_c%mm_grads2      = zero
  !!qmmm_ewald_c%dexl_xyz = zero
  if(QNoPMEwald) qmmm_ewald_c%d_ewald_mm= zero

  ! for switching function
  if(mm_main_c%q_switch) mm_main_c%dxyz_sw2(1:6,1:mm_main_c%isize_swt_array) =zero

  ! Compute gradient contribution from the qm/mm-ewald correction.

  ! 1) Real space contribution:  since the real space contribution can be added
  !    into the dz,dy,dz array directly, the gradients are added in "qm_grads" and
  !    "mm_grads" and copied into the amin array below. however, the contribution
  !    from the qm-mm interactions exclued list is added into the "d_ewald_mm."
  call qm_ewald_real_space_gradient(qmmm_ewald_c%natom,qm_main_c%numat,mm_main_c%numatm, &
                                    mm_main_c%mm_coord,mm_main_c%mm_chrgs,               &
                                    qm_main_c%qm_coord,mm_main_c%qm_charges,             &
                                    qm_main_c%qm_grads2,mm_main_c%mm_grads2)

  if(qmmm_ewald_c%nexl_atm > 0) then
     qmmm_ewald_c%dexl_xyz = zero
     call qm_ewald_real_space_gradient_exl(qmmm_ewald_c%natom,qm_main_c%numat,           &
                                           qm_main_c%qm_coord,mm_main_c%qm_charges,      &
                                           qm_main_c%qm_grads2) 
  end if

  ! 2) Reciprocal space contribution:  due to the need of computing virial contribution 
  !    from the reciprocal space, the gradients are saved into "d_ewald_mm" and added
  !    into the main dx,dy,dz array later.
  call qm_ewald_recip_space_gradient(qmmm_ewald_c%natom,qm_main_c%numat,qm_control_c%qminb, &
                                     qmmm_ewald_c%iastrt,qmmm_ewald_c%iafinl,               &
                                     qmmm_ewald_c%iatotl,itotal,itotkq,                     &
                                     qmmm_ewald_c%totkq,qmmm_ewald_c%ksqmaxq,               &
                                     qmmm_ewald_c%kmaxqx,qmmm_ewald_c%kmaxqy,               &
                                     qmmm_ewald_c%kmaxqz,                                   &
                                     cg,mm_main_c%qm_charges,qmmm_ewald_c%recip,            &
                                     virial,QNoPMEwald)

  ! 2-1) Reciprocal spacecontribution from the PME version.
  if(.not.QNoPMEwald) then
     call qm_pme_mm_grad(qmmm_ewald_c%natom,qm_main_c%numat,qm_control_c%qminb, &
                         x,y,z,qmmm_ewald_c%d_ewald_mm,cg,mm_main_c%qm_charges, &
                         qmmm_ewald_c%recip,qmmm_ewald_c%volume,                &
                         qmmm_ewald_c%kappa,virial)
  end if


!  ! for d_ewald_mm/qm_grads/mm_grads/dexl_xyz.
!  ! combine gradients and virial.
!#if KEY_PARALLEL==1
!  ! combine gradients and virial. 
!  ! do I need to do this? have to check!
!  if(numnod > 1) then
!     call GCOMB(qm_main_c%qm_grads     ,3*qm_main_c%numat)
!     call GCOMB(mm_main_c%mm_grads     ,3*mm_main_c%numatm)
!     ! avoid communicate, but instead when do sum, sum over all atoms.
!     !!call GCOMB(qmmm_ewald_c%d_ewald_mm,3*qmmm_ewald_c%natom)
!     !!call GCOMB(virial,9)
!     !!if(mynod.ne.0) virial=zero
!  end if
!#endif

  ! Copying gradients into the CHARMM gradient dx/dy/dz arrays.
  ! 2) mm atoms within the cutoff
  !do i=mstart,mstop                    ! 1,mm_main_c%numatm
  !   m=mm_main_c%qm_mm_pair_list(i)
  !   dx(m)=dx(m)+mm_main_c%mm_grads(1,i)
  !   dy(m)=dy(m)+mm_main_c%mm_grads(2,i)
  !   dz(m)=dz(m)+mm_main_c%mm_grads(3,i)
  !end do
  call grad_copy_2_main(mstart,mstop,dx,dy,dz,mm_main_c%mm_grads2,mm_main_c%qm_mm_pair_list)

  ! 1) qm atoms: since qm_grads has not been broadcasted, so add over all qm atoms.
  !do i=1,qm_main_c%numat
  !   m=qm_control_c%qminb(i)
  !   dx(m)=dx(m)+qm_main_c%qm_grads(1,i)
  !   dy(m)=dy(m)+qm_main_c%qm_grads(2,i)
  !   dz(m)=dz(m)+qm_main_c%qm_grads(3,i)
  !end do
  call grad_copy_2_main(1,qm_main_c%numat,dx,dy,dz,qm_main_c%qm_grads2,qm_control_c%qminb)

#if KEY_PARALLEL==1
  if(mynod == 0) then
#endif
  ! 3) excluded MM atoms from the qm-mm non-bonded interactions. (igmsel(i)=5 case)
     if(qmmm_ewald_c%nexl_atm > 0) then
        !do i=1,qmmm_ewald_c%nexl_atm
        !   m=qmmm_ewald_c%nexl_index(i)
        !   dx(m)=dx(m)+qmmm_ewald_c%dexl_xyz(1,i)
        !   dy(m)=dy(m)+qmmm_ewald_c%dexl_xyz(2,i)
        !   dz(m)=dz(m)+qmmm_ewald_c%dexl_xyz(3,i)
        !end do
        call grad_copy_2_main(1,qmmm_ewald_c%nexl_atm,dx,dy,dz,qmmm_ewald_c%dexl_xyz, &
                              qmmm_ewald_c%nexl_index)
     end if
#if KEY_PARALLEL==1
  end if
#endif

  ! 4) reciprocal space contribution is in d_ewald_mm. (refer the GETGRDQ routine).

  ! now for switching function contributions.
  if(mm_main_c%q_switch) call put_switching_gradient2(qmmm_ewald_c%natom,mm_main_c%NUMATM,dx,dy,dz)

  return
  !
  contains
     subroutine grad_copy_2_main(ibegin,ifinal,dx,dy,dz,grads,index)
     !
     ! copy each gradient into the main gradient arrays.
     !
     use chm_kinds
     implicit none
     integer :: ibegin,ifinal,i,m,index(*)
     real(chm_real):: dx(*),dy(*),dz(*),grads(3,*) 

     do i=ibegin,ifinal
        m=index(i)
        dx(m)=dx(m)+grads(1,i)
        dy(m)=dy(m)+grads(2,i)
        dz(m)=dz(m)+grads(3,i)
     end do
     !
     return
     end subroutine grad_copy_2_main
     !
  end subroutine qmmm_Ewald_gradient


  !=====================================================================
  ! Private routines:
  !=====================================================================
  subroutine qm_info_setup
  !=====================================================================
  !
  ! new version of subroutine INPUT. So, majority will be set here.
  ! 
  use number, only : zero
  use qm1_info, only     : qm_main_c,mm_main_c,qm_param_c,qm_gho_info_c
  !!use qm1_parameters,only: CORE,EHEAT,EISOL,LORBS
  use qm1_constant, only : EVCAL

  implicit none
  ! local variables
  integer :: i,ib,ni,nodd,nel

  ! Note: 
  !       memories are allocated in qmmm_init_set 

  ! 1)
  ! determine the following vairables.
  !   nfirst(i) first basis orbital of atom i.
  !   nlast(i)  last  basis orbital of atom i.
  !   nel       number of electrons.
  !   nelmd     number of atoms with d-orbitals.
  !   nfock     step size in Fock (5 SP, 10 SPD).
  !
  ! CORE and LORBS are loaded in "initialize_elements_and_params"
  qm_main_c%nelmd  = 0
  qm_main_c%nfock  = 5
  nel              =-qm_main_c%i_qmcharge
  ib               = 0
  do i=1,qm_main_c%numat
     !!ni     = qm_main_c%nat(i)
     if(qm_param_c%LORBS(i)>= 9) qm_main_c%nelmd = qm_main_c%nelmd+1
     !
     qm_main_c%num_orbs(i)= qm_param_c%LORBS(i) ! number of orbitals of each atom.
     qm_main_c%nfirst(i)  = ib+1
     nel                  = nel+NINT(qm_param_c%CORE(i))
     ib                   = ib +qm_param_c%LORBS(i) 
     qm_main_c%nlast(i)   = ib
  end do
  if(qm_main_c%nelmd > 0) qm_main_c%nfock = 10

  ! if GHO, remove three auxiliary electrons from the QM-link atom when
  !         count the number of active electrons
  if(qm_gho_info_c%q_gho) nel = nel - 3*qm_gho_info_c%nqmlnk
  !
  qm_main_c%nel = nel

  ! 2)
  ! determine occupation number of mol. orbitals.
  !     nel      number of electrons       (NEL = NALPHA+NBETA).
  !     nalpha   number of alpha electrons 
  !     nbeta    number of beta  electrons 
  !     NBETA    equal to number of doubly occupied molecular orbitals
  !              in rhf calculations with imult.ne.1 (not singlet?)
  !     NUMB     number of highest occupied molecular orbital.
  nodd   = MAX(1,qm_main_c%imult)-1   ! 0, if imult=0 singlet (RHF).

  qm_main_c%norbs  = qm_main_c%nlast(qm_main_c%numat)
  qm_main_c%nbeta  =(qm_main_c%nel-nodd)/2
  qm_main_c%nalpha = qm_main_c%nbeta+nodd
  qm_main_c%numb   = qm_main_c%nalpha
  qm_main_c%nclo   = qm_main_c%nbeta

  qm_main_c%iodd   = 0  ! this is for the RHF singlet state
  qm_main_c%jodd   = 0  ! 
  ! for the RHF and not singlet state.
  if(qm_main_c%imult > 0 .and. .not.qm_main_c%uhf) then
     ! this is not used, it is for an excited singlet state with two singly
     ! occupied orbitals. 
     !if(qm_main_c%imult.eq.1) then
     ! this is for the singlet state of the open shell system (UHF) with two singly occupied orbitals.
     ! This scf solution usually corresponds to an excited single state.
     !   qm_main_c%numb = qm_main_c%nbeta+1
     !   qm_main_c%nclo = qm_main_c%nbeta-1
     !   qm_main_c%iodd = qm_main_c%nbeta
     !   qm_main_c%jodd = qm_main_c%numb
     !else if(qm_main_c%imult.eq.2) then
     if(qm_main_c%imult == 2) then           ! doublet
        qm_main_c%iodd = qm_main_c%numb
     else if(qm_main_c%imult == 3) then      ! triplet
        qm_main_c%iodd = qm_main_c%nbeta+1 
        qm_main_c%jodd = qm_main_c%nbeta+2
     end if
  end if
  qm_main_c%nmos   = qm_main_c%numb
  ! explicit definition of occupation numbers.
  qm_main_c%imocc  = 1  ! = ABS(IUHF=-1)
  qm_main_c%nocca  = 0  ! 0, if imocc .lt. 2
  qm_main_c%noccb  = 0  ! 0,

  ! if GHO
  if(qm_gho_info_c%q_gho) qm_gho_info_c%norbsgho = qm_main_c%norbs

  ! 4)
  ! compute sum of atomic energies and heats of formation.
  qm_main_c%ener_atomic = zero
  qm_main_c%HofF_atomic = zero
  do i=1,qm_main_c%numat
     !!ni     = qm_main_c%nat(i)
     qm_main_c%HofF_atomic=qm_main_c%HofF_atomic+qm_param_c%EHEAT(i)
     qm_main_c%ener_atomic=qm_main_c%ener_atomic+qm_param_c%EISOL(i)
  end do

  !
  ! convert EISOL into kcal/mol unit hear.
  qm_main_c%ener_atomic=EVCAL*qm_main_c%ener_atomic

  return
  end subroutine qm_info_setup


  !=====================================================================
  subroutine compute_one_center_h(numat,qm_scf_main_m,qm_param_m)
  !=====================================================================
  ! 
  ! precompute diagonal one-center terms.
  !
  use qm1_info ! , only : qm_main,qm_scf_main,qm_param
  use number        ,only : zero
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none
  type(qm_scf_main):: qm_scf_main_m
  type(qm_param)   :: qm_param_m
  integer :: numat,i,ni,ia,iorbs,j,icnt
  !!integer :: istart,istop
  
  !!istart = 1
  !!istop  = numat

  ! initialize.
  qm_scf_main_m%H_1cent(1:qm_scf_main_m%dim_norbs)=zero
  qm_scf_main_m%imap_h(1:qm_scf_main_m%dim_norbs) =0

  ! Diagonal one-center terms.
  ! this is done once at the beginning of QM setup.
  icnt=0
  do i=1,numat                          ! istart,istop
     ia     = qm_param_m%ia_local(i)    ! qm_main_c%nfirst(i)
     iorbs  = qm_param_m%iorbs_local(i) ! qm_main_c%num_orbs(i), NLAST(I)-IA+1
     icnt   = icnt+1
     qm_scf_main_m%imap_h(icnt) =qm_scf_main_m%INDX(ia)+ia
     qm_scf_main_m%H_1cent(icnt)=qm_param_m%USS(i)
     if(iorbs >= 9) then
        do j=ia+1,ia+3
           icnt   = icnt+1
           qm_scf_main_m%imap_h(icnt) =qm_scf_main_m%INDX(j)+j
           qm_scf_main_m%H_1cent(icnt)=qm_param_m%UPP(i)
        end do
        do j=ia+4,ia+8
           icnt   = icnt+1
           qm_scf_main_m%imap_h(icnt) =qm_scf_main_m%INDX(j)+j
           qm_scf_main_m%H_1cent(icnt)=qm_param_m%UDD(i)
        end do
     else if(iorbs >= 4) then
        do j=ia+1,ia+3
           icnt   = icnt+1
           qm_scf_main_m%imap_h(icnt) =qm_scf_main_m%INDX(j)+j
           qm_scf_main_m%H_1cent(icnt)=qm_param_m%UPP(i)
        end do
     end if
  end do
!#if KEY_PARALLEL==1
!  ! it is need to be broacasted.
!  if(numnod>1) then
!     call gcomb(qm_scf_main_m%H_1cent,qm_scf_main_m%dim_norbs)
!     call igcomb(qm_scf_main_m%imap_h,qm_scf_main_m%dim_norbs)
!  end if
!#endif
  return
  end subroutine compute_one_center_h

  !=====================================================================

  !=====================================================================
  subroutine put_switching_gradient(natom,numat,dx,dy,dz)
  !
  ! Copying the contribution of gradient component of swicthing function part
  ! into the main gradient arrays.
  ! 
  ! This is done separately here, because dS(rij)/d_x_i_k is also affects other
  ! atoms beloning to the same group.
  !
  use number,only : zero
  use psf,only : igpbs
  use qm1_info,only : qm_main_c,mm_main_c
  use nbndqm_mod, only: map_grp_c

  implicit none
  integer :: natom,numat
  real(chm_real):: dx(natom),dy(natom),dz(natom)

  integer :: i,j,k,is,ip,js,jp,irs,jrs,icnt,jqmcnt
  real(chm_real):: dxyz_qm(3,numat)

  if(.not.mm_main_c%q_switch) return

  do i=1,qm_main_c%nqmgrp(1)
     irs = qm_main_c%nqmgrp(i+1)  ! qm group
     is =  igpbs(irs) + 1
     ip =  igpbs(irs+1)
     jqmcnt = 0
     dxyz_qm(1:3,1:numat) = zero
     do j=1,mm_main_c%inum_mm_grp
        jrs = map_grp_c%map_mmgrp_to_group(j)
        js  = igpbs(jrs) + 1
        jp  = igpbs(jrs+1)
        if(mm_main_c%q_mmgrp_qmgrp_swt(j,i) > 0) then
           ! do this pair
           icnt = mm_main_c%q_mmgrp_qmgrp_swt(j,i)
           jqmcnt = jqmcnt + 1

           ! for qm atoms.
           do k=1,ip-is+1
              dxyz_qm(1:3,k) = dxyz_qm(1:3,k) + mm_main_c%dxyz_sw(1:3,icnt)
           end do

           ! for mm atoms.
           do k=js,jp
              dx(k) = dx(k) + mm_main_c%dxyz_sw(4,icnt) 
              dy(k) = dy(k) + mm_main_c%dxyz_sw(5,icnt) 
              dz(k) = dz(k) + mm_main_c%dxyz_sw(6,icnt) 
           end do
        end if
     end do

     ! for qm atoms
     if(jqmcnt > 0) then
        j = 1
        do k=is,ip
           dx(k) = dx(k) + dxyz_qm(1,j)
           dy(k) = dy(k) + dxyz_qm(2,j)
           dz(k) = dz(k) + dxyz_qm(3,j)
           j     = j + 1
        end do
     end if
  end do

  return
  end subroutine put_switching_gradient
  !=====================================================================

  subroutine put_switching_gradient2(natom,numat,dx,dy,dz)
  !
  ! Copying the contribution of gradient component of swicthing function part
  ! into the main gradient arrays.
  !
  ! This is done separately here, because dS(rij)/d_x_i_k is also affects other
  ! atoms beloning to the same group.
  !
  use number,only : zero
  use psf,only : igpbs
  use qm1_info,only : qm_main_c,mm_main_c
  use nbndqm_mod, only: map_grp_c

  implicit none
  integer :: natom,numat
  real(chm_real):: dx(natom),dy(natom),dz(natom)

  integer :: i,j,k,is,ip,js,jp,irs,jrs,icnt,jqmcnt
  real(chm_real):: dxyz_qm(3,numat)

  if(.not.mm_main_c%q_switch) return

  do i=1,qm_main_c%nqmgrp(1)
     irs = qm_main_c%nqmgrp(i+1)  ! qm group
     is =  igpbs(irs) + 1
     ip =  igpbs(irs+1)
     jqmcnt = 0
     dxyz_qm(1:3,1:numat) = zero
     do j=1,mm_main_c%inum_mm_grp
        jrs = map_grp_c%map_mmgrp_to_group(j)
        js  = igpbs(jrs) + 1
        jp  = igpbs(jrs+1)
        if(mm_main_c%q_mmgrp_qmgrp_swt(j,i) > 0) then
           ! do this pair
           icnt = mm_main_c%q_mmgrp_qmgrp_swt(j,i)
           jqmcnt = jqmcnt + 1

           ! for qm atoms.
           do k=1,ip-is+1
              dxyz_qm(1:3,k) = dxyz_qm(1:3,k) + mm_main_c%dxyz_sw2(1:3,icnt)
           end do

           ! for mm atoms.
           do k=js,jp
              dx(k) = dx(k) + mm_main_c%dxyz_sw2(4,icnt)
              dy(k) = dy(k) + mm_main_c%dxyz_sw2(5,icnt)
              dz(k) = dz(k) + mm_main_c%dxyz_sw2(6,icnt)
           end do
        end if
     end do

     ! for qm atoms
     if(jqmcnt > 0) then
        j = 1
        do k=is,ip
           dx(k) = dx(k) + dxyz_qm(1,j)
           dy(k) = dy(k) + dxyz_qm(2,j)
           dz(k) = dz(k) + dxyz_qm(3,j)
           j     = j + 1
        end do
     end if
  end do

  return
  end subroutine put_switching_gradient2
  !=====================================================================


  !-----------------------------------------------------------------------!
  !----------------- Extended Lagrangian with dissipation ----------------!
  ! AMN Niklasson, JCP (2009) 130:214109                                  !
  ! and Based on code by Guishan Zheng, Harvard Univ. 05/19/2010.         !
  !
  subroutine initcoef_diss(IOrder,coefk_local,cextr_local)
  !
  ! Iorder: 3  4  5  6  7  8  9
  !
  ! Alpha values are set here, but Kappa value is not used in the present implementation.
  ! Since in Niklasson's implementation, kappa is for the force constant of restraining force,
  ! which is not used in the present ADMP work as it is the Fock matrix component (dE/dP).
  !
  use chm_kinds
  use number

  implicit none
  integer :: IOrder
  real(chm_real) :: coefk_local,cextr_local(0:IOrder)
  real(chm_real) :: c(0:9,3:9)
  real(chm_real) :: kappa, alpha
  integer :: i

  !if(IOrder.gt.9) then
  !   wrndie(-1,'<initcoef_diss>','The allowed highest extrapolation order is 9.')
  !   return
  !endif

  !
  C(0:9,3:9)  = zero
  select case(IOrder)
     case (3)
        !    order 3
        kappa  =    1.692D0
        alpha  =  150.0D-3
        C(0,3) =   -2.0d0
        C(1,3) =    3.0d0
        C(2,3) =    0.0d0
        C(3,3) =   -1.0d0
     case (4)
        !    order 4
        kappa  =    1.75D0
        alpha  =   57.0D-3
        C(0,4) =   -3.0d0
        C(1,4) =    6.0d0
        C(2,4) =   -2.0d0
        C(3,4) =   -2.0d0
        C(4,4) =    1.0d0
     case (5)
        !    order 5
        kappa  =    1.82D0
        alpha  =   18.0D-3
        C(0,5) =   -6.0d0
        C(1,5) =   14.0d0
        C(2,5) =   -8.0d0
        C(3,5) =   -3.0d0
        C(4,5) =    4.0d0
        C(5,5) =   -1.0d0
     case (6)
        !    order 6
        kappa  =    1.84D0
        alpha  =    5.5D-3
        C(0,6) =  -14.0d0
        C(1,6) =   36.0d0
        C(2,6) =  -27.0d0
        C(3,6) =   -2.0d0
        C(4,6) =   12.0d0
        C(5,6) =   -6.0d0
        C(6,6) =    1.0d0
     case (7)
        !    order 7
        kappa  =    1.86D0
        alpha  =    1.6D-3
        C(0,7) =  -36.0d0
        C(1,7) =   99.0d0
        C(2,7) =  -88.0d0
        C(3,7) =   11.0d0
        C(4,7) =   32.0d0
        C(5,7) =  -25.0d0
        C(6,7) =    8.0d0
        C(7,7) =   -1.0d0
     case (8)
        !    order 8
        kappa  =    1.88D0
        alpha  =    0.44D-3
        C(0,8) =  -99.0d0
        C(1,8) =  286.0d0
        C(2,8) = -286.0d0
        C(3,8) =   78.0d0
        C(4,8) =   78.0d0
        C(5,8) =  -90.0d0
        C(6,8) =   42.0d0
        C(7,8) =  -10.0d0
        C(8,8) =    1.0d0
     case (9)
        !    order 9
        kappa  =    1.89D0
        alpha  =    0.12D-3
        C(0,9) = -286.0d0
        C(1,9) =  858.0d0
        C(2,9) = -936.0d0
        C(3,9) =  364.0d0
        C(4,9) =  168.0d0
        C(5,9) = -300.0d0
        C(6,9) =  184.0d0
        C(7,9) =  -63.0d0
        C(8,9) =   12.0d0
        C(9,9) =   -1.0d0
  end select

  ! BOMD case.

  ! Here, this is done even to take care of
  ! P_n+1 = 2P_n - P_n-1 + kappa*(D_n - P_n) + alpha*sum_k(0~K) c_k*P_n-k
  ! final coefficients
  do i = 0, IOrder
     cextr_local(i) = C(i,IOrder)*alpha
  end do
  coefk_local    = Kappa   ! this is for D_n and P_n
  cextr_local(0) = two - coefk_local + cextr_local(0)
  cextr_local(1) = cextr_local(1) - one

  return
  end subroutine initcoef_diss
  !-----------------------------------------------------------------------!
  
#endif /* (mndo97)*/
end module qmmm_interface
! end
