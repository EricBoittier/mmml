! qm and qm/mm setup and other interfaces.
module qmmm_interface
  use chm_kinds
  use dimens_fcm

#if KEY_MNDO97==1 /*mndo97*/

  ! counter for md & cpmd.
  integer, save :: ntime_step_md = 0           ! P. Ojeda Feb. 03 2015
  logical, save :: q_do_cpmd_noupdate =.false. ! for controling cpmd calls.
                                               ! (see scf_energy routine for usage.)

  logical, save :: q_do_bomd_mts   =.false.    ! logical flag for the preparation of cpmd. (see comments)
  logical, save :: q_do_mts_oldstep=.false.    ! (nstep-1) step for computing old density.

  ! for DXL-BOMD with AMN Niklasson, JCP (2009) 130:214109 & Guishan Zheng, Harvard Univ.
  integer, save :: Kth_sum_order=0             ! K value (for AMN Niklasson, JCP (2009) 130:214109).
  real(chm_real),pointer,save,private :: cextr(:)=>Null()     ! coefficient(0:K_sum_order)
  real(chm_real),pointer,save,private :: pa_exp(:,:)=>Null()  ! PA(K_sum_order,linear_norbs)
  real(chm_real),pointer,save,private :: pa_aux(:)=>Null(), & ! auxiliary PA for DXL-BOMD 
                                         pa_anew(:)=>Null()   ! new auxiliary PA (i.e., pa_aux(i+1)
  real(chm_real),save,private         :: coefk                ! kappa value (This is not used.)

  ! MNDO two-electron integrals
  real(chm_real),save,dimension(:),allocatable :: w_linear

  contains

  !=====================================================================
  subroutine qmmm_init_set(nqmtheory,nqmcharge,nspin,numat,natgho,  &
                           natom,EWMODE,NQMEWD,                     &
                           qmcharge,scfconv,                        &
                           qlink,QMMM_NoDiis,LQMEWD,NOPMEwald,      &
                           q_bond_order,q_m_charge,iiunit,          &
                           q_dxl_bomd,K_order,N_scf_step,           &
                           q_fockmd,iopt_fdiss,imax_fdiss,          &
                           q_cut_by_group,qmswtch,q_cpmd,QNoMemIncore)
  !=====================================================================
  !
  ! setup qmmm options and the default values.
  !
  use number,only : one
  use qm1_info, only : qm_control_r,qm_main_r,mm_main_r,qm_scf_main_r, &
                       allocate_deallocate_qm,allocate_deallocate_mm, &
                       allocate_deallocate_qmmm
  use qm1_lagdynamics_module,only : initcoef_diss
  use mndgho_module, only : qm_gho_info_r

  implicit none
  !
  integer :: nqmtheory,nqmcharge,nspin,numat,natgho,natom,EWMODE,NQMEWD,K_order,N_scf_step,iiunit, &
             iopt_fdiss,imax_fdiss
  real(chm_real):: qmcharge,scfconv
  logical :: qlink,QMMM_NoDiis,LQMEWD,NOPMEwald,q_dxl_bomd,q_cut_by_group,qmswtch, &
             q_cpmd,QNoMemIncore,q_bond_order,q_m_charge,q_fockmd
  ! 
  ! local variables
  logical :: do_am1_pm3,do_d_orbitals
  real(chm_real):: tmp_val

  ! for qm main control
  qm_control_r%ifqnt        =.true.
  qm_control_r%qmqm_analyt  =.false.  ! only use finite difference gradient.
  qm_control_r%q_diis       = .not.QMMM_NoDiis
  qm_control_r%cpmd         =q_cpmd   ! for cpmd.
  qm_control_r%md_run       =.false.  ! flag for using md.
  qm_control_r%qm_e_set     =.false.  ! .false. PA is not computed.
  qm_control_r%q_do_cpmd_pme=.false.  ! .false. PME-only is not used for qm/mm-pme.
                                      ! this it to use for cpmd control, as 
                                      ! the first md step should be scf. 
  ntime_step_md             = 0       ! local counter... of md step
  !
  ! QM method: MNDO/AM1/PM3/AM1d/MNDOd
  qm_control_r%iqm_mode     = nqmtheory
  !
  do_d_orbitals =.false.      ! default for AM1
  do_am1_pm3    =.true.       !
  if(nqmtheory.eq.1) then
     qm_control_r%qm_model(1:6) ='MNDO  '
     do_am1_pm3                 =.false.
  else if(nqmtheory.eq.2) then
     qm_control_r%qm_model(1:6) ='AM1   '
  else if(nqmtheory.eq.3) then
     qm_control_r%qm_model(1:6) ='PM3   '
  else if(nqmtheory.eq.4) then
     qm_control_r%qm_model(1:6) ='AM1/d '
     do_d_orbitals              =.true.
  else if(nqmtheory.eq.5) then
     qm_control_r%qm_model(1:6) ='MNDO/d'
     do_d_orbitals              =.true.
     do_am1_pm3                 =.false.
  end if
  !
  qm_control_r%q_am1_pm3    = do_am1_pm3     ! do Gaussian core terms for the AM1-type models 
  qm_control_r%do_d_orbitals= do_d_orbitals  ! have d-orbtials.

  ! analysis
  qm_control_r%ianal_unit   = iiunit         ! analysis output unit.
  qm_control_r%q_bond_order = q_bond_order   ! do bond order analysis
  qm_control_r%q_m_charge   = q_m_charge     ! do mulliken charge analysis.


  ! qm_main
  qm_main_r%numat     = numat
  qm_main_r%i_qmcharge= nqmcharge
  qm_main_r%qmcharge  = qmcharge
  qm_main_r%imult     = nspin       ! for singlet, "imult=0"
  qm_main_r%uhf       =.false.      ! for now, turn off UHF.
  if(QNoMemIncore) then
     qm_main_r%rij_qm_incore =.false.
     mm_main_r%rij_mm_incore =.false.
  else
     !for now, turn on this only if qm/mm-ewald is used.
     if (LQMEWD) then
        qm_main_r%rij_qm_incore =.true.   ! turn on the memory incore.
        mm_main_r%rij_mm_incore =.true.   ! turn on the memory incore.
     else
        qm_main_r%rij_qm_incore =.false.
        mm_main_r%rij_mm_incore =.false.
     end if
  end if

  ! scf convergence: see scf_iter routine, where SCFCRT and PLCRT are overwritten.
  qm_scf_main_r%SCFCRT= scfconv     ! 1.0d-6; default value.
  qm_scf_main_r%PLCRT = scfconv     ! 1.0d-4      ! default value. 
  !qm_scf_main_r%kitscf= 200        ! .

  ! For DXL-BOMD by AMN Niklasson, JCP (2009) 130:214109 and Guishan Zheng, Harvard Univ. 05/19/2010.
  qm_control_r%q_dxl_bomd=q_dxl_bomd  !
  qm_control_r%N_scf_step=N_scf_step  ! Number of scf cycle per each md step.
  Kth_sum_order=K_order               ! the order to which the summation will be performed.
  if(qm_control_r%q_dxl_bomd) then
     tmp_val = one
     if(associated(cextr)) deallocate(cextr)
     allocate(cextr(0:Kth_sum_order))
     call initcoef_diss(tmp_val,Kth_sum_order,coefk,cextr,.false.)
  end if

  ! Fock matrix dynamics (ref: )
  qm_control_r%q_fockmd        = q_fockmd
  qm_control_r%q_do_fockmd_scf =.false.        ! initialization.
  qm_control_r%i_fockmd_option = iopt_fdiss    ! Options for fock-MD (see qm_info.src and fock_diis routine.)
  qm_control_r%imax_fdiss      = imax_fdiss    ! maximum number of diis iterations.

  ! mm_main
  mm_main_r%natom   = natom         !
  mm_main_r%natom_mm= natom-numat   !
  mm_main_r%numatm  = natom-numat   ! for the maximum size of pointer arrays.
  mm_main_r%nlink   = 0             ! the number of h-link atom, 0 default.

  ! for qm/mm-ewald
  mm_main_r%LQMEWD = LQMEWD         ! use qm/mm-Ewald
  mm_main_r%PMEwald=.not.NOPMEwald  ! use qm/mm-PME version.
  mm_main_r%EWMODE = EWMODE  ! 1= within cutoff, interact with all components in Fock
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
  if(mm_main_r%EWMODE == 3) then
     mm_main_r%q_diag_coulomb =.true.
  else
     mm_main_r%q_diag_coulomb =.false.
  endif
  mm_main_r%NQMEWD = NQMEWD  ! 0= use Mulliken charge to represent charge of QM images.

  ! For cutoff options,
  ! by turning-on, the qm-mm interactions are evaluated based on the group-by-group pair distance.
  ! So, any group pair that is separated more than the cutoff distance is ignore,
  ! whereas in the default option, any atom pair, in which that MM atom in the group is within
  ! the cutoff distance from any QM group, is evaluated.
  !
  ! Also, use switching function between the r_cut_group_dist_on and r_cut_group_dist_off 
  ! distances (between the QM and MM groups).
  !
  mm_main_r%q_cut_by_group    = q_cut_by_group
  mm_main_r%q_switch          = qmswtch
 
  ! for GHO methods
  qm_gho_info_r%q_gho = qlink   !
  if(qm_gho_info_r%q_gho) then
     qm_gho_info_r%uhfgho= qm_main_r%uhf
     ! see subroutine GHOHYB
     !qm_gho_info_r%numat = qm_main_r%numat
     !qm_gho_info_r%nqmlnk= natgho
  end if

  ! now allocate memories for qm and mm atoms coordinates, gradients, charges, etc:
  call allocate_deallocate_qmmm(qm_control_r,qm_main_r,mm_main_r,.true.)
  call allocate_deallocate_qm(qm_main_r,.true.)
  call allocate_deallocate_mm(qm_main_r,mm_main_r,.true.)

  return
  end subroutine qmmm_init_set


  !=====================================================================
  subroutine qmmm_load_parameters_setup_qm_info(QSRP_PhoT)
  !=====================================================================
  !
  ! load parameters, setup qm info, and allocate scf/qm memories. 
  !
  use qm1_info, only : qm_control_r,qm_main_r,qm_scf_main_r, &
                       determine_qm_scf_arrray_size, &
                       allocate_deallocate_qm_scf
  use qm1_scf_module, only : qm_scf_diis_r,qm_fockmd_diis_r,allocate_deallocate_qm_diis
  use mndgho_module,only : qm_gho_info_r, allocate_deallocate_gho
  !
  use qm1_parameters, only : q_parm_loaded,  &
                             initialize_elements_and_params
  !
  use qm1_energy_module,only: QMMM_module_prep
  use qm1_scf_module, only : define_pair_index
  implicit none
  logical:: QSRP_PhoT

  ! now.................................................................
  ! 1) load qm and qmmm parameters 
  if(.not.q_parm_loaded)  &
     call initialize_elements_and_params(qm_control_r%iqm_mode,QSRP_PhoT)

  ! 2) load qm parameters and qm info common to all methods:
  call qm_info_setup

  ! 3) determined array sizes to be allocated and allocate them
  call determine_qm_scf_arrray_size(qm_main_r,qm_scf_main_r,.true.)
  call allocate_deallocate_qm_scf(qm_main_r,qm_scf_main_r,.true.)
  call allocate_deallocate_qm_diis(qm_scf_main_r,qm_scf_diis_r,qm_fockmd_diis_r, &
                                   qm_control_r%q_diis,qm_control_r%q_fockmd,    &
                                   qm_control_r%imax_fdiss,                      &
                                   qm_main_r%uhf,.true.)

  ! 3-1) define pair indexes: IP,IP1,IP2 for Coulomb part
  !                           JX,JP1,JP2,JP3 for Exchange part
  ! see qm1_scf_module
  call define_pair_index

  ! 4) for GHO
  if(qm_gho_info_r%q_gho) then
     call allocate_deallocate_gho(qm_scf_main_r,qm_gho_info_r, &
                                  qm_control_r%q_diis,         &
                                  qm_main_r%uhf,.true.)
     ! determine some gho-related variables
     qm_gho_info_r%norbhb    =qm_main_r%NORBS - 3*qm_gho_info_r%nqmlnk
     qm_gho_info_r%naos      =qm_gho_info_r%norbhb-qm_gho_info_r%nqmlnk
     qm_gho_info_r%lin_naos  =qm_scf_main_r%indx(qm_gho_info_r%naos)  +qm_gho_info_r%naos ! =(NAOS*(NAOS+1))/2
     qm_gho_info_r%lin_norbhb=qm_scf_main_r%indx(qm_gho_info_r%norbhb)+qm_gho_info_r%norbhb
 
     ! variables used in FTOFHB
     qm_gho_info_r%norbao    =qm_main_r%NORBS - 4*qm_gho_info_r%nqmlnk
     qm_gho_info_r%lin_norbao=qm_scf_main_r%indx(qm_gho_info_r%norbao)+qm_gho_info_r%norbao !=NORBAO*(NORBAO+1)/2
     qm_gho_info_r%nactatm   =qm_gho_info_r%numat-qm_gho_info_r%nqmlnk
  end if


  ! 5) now precompute some things
  ! 5-1) One-center part.
  call compute_one_center_h

  ! 6) now prepare for qm1_energy_module setup
  call QMMM_module_prep

  return
  end subroutine qmmm_load_parameters_setup_qm_info


  !=====================================================================
  subroutine qm_cpmd_init(q_cpmd,COMLYN,COMLEN)
  !=====================================================================
  !
  ! input usage: CPMD -
  !              PAMAss [real]  INCKineticEnergy
  !              CONStraint  CONType [int] - ! apply density constraint
  !              NOSE  MAS1 [real]  MAS2 [real]  MAS3 [real] -
  !              TEMPerature [real]        -
  !              Timestep    [real] NCYCle [int] -
  !              DISSipation KORDer [int] ASCAle [real] -  ! the dissipation and its order 
  !              -                                         ! (see AMN Niklasson, JCP (2009) 130:214109)
  !              MSTEP [int]  ! mstep where BOMD density gradient is evaluated.
  !
  ! for Curvy-steps ELMD.
  !              CPMD -
  !              PAMAss [real]  INCKineticEnergy -
  !              CURVyMD -  ! turn on
  !              TEMPerature [real] LANGevin FBETa [real]  - 
  !              Timestep    [real] NCYCle [int] -
  !              MSTEP [int]  ! mstep where BOMD density gradient is evaluated. 
  !
  use chm_kinds
  use dimens_fcm
  use number
  use consta, only : TIMFAC
  use string
  use stream
  use qm1_info, only : qm_scf_main_r,qm_main_r
  use qm1_lagdynamics_module,only : qm_cpmd_main_r,allocate_deallocate_qm_cpmd,initcoef_diss

  implicit none
  logical         :: q_cpmd
  CHARACTER(len=*):: COMLYN
  INTEGER         :: COMLEN
  !!integer         :: K_order
  !!real(chm_real)  :: A_scale

  ! default:
  qm_cpmd_main_r%q_admp_md =.true.  ! use ADMP
  qm_cpmd_main_r%q_curvy_md=.false. ! use Curvy steps MD
  qm_cpmd_main_r%q_cnst    =.false.
  qm_cpmd_main_r%q_nose    =.false.
  qm_cpmd_main_r%q_langevin=.false.
  !!qm_cpmd_main_r%q_exl_diss=.false.

  if(.not.q_cpmd) return

  if(PRNLEV.ge.2) write(outu,20) 'CPMD run info:'

  ! default is ADMP.
  qm_cpmd_main_r%q_admp_md =.true.

  qm_cpmd_main_r%Mass_PA=GTRMF(COMLYN,COMLEN,'PAMA',one)
  if(PRNLEV.ge.2) write(outu,22) 'PA density mass:',qm_cpmd_main_r%Mass_PA

  qm_cpmd_main_r%q_curvy_md=(INDXA(COMLYN,COMLEN,'CURV').GT.0)
  if(qm_cpmd_main_r%q_curvy_md) then
     qm_cpmd_main_r%q_admp_md =.false. ! to turn-off ADMP.

     if(prnlev.ge.2) write(outu,20) 'Curvy-steps ELMD will be used for P propagation.'
  end if
  !
  if(qm_cpmd_main_r%q_admp_md) then
     qm_cpmd_main_r%q_cnst     =(INDXA(COMLYN,COMLEN,'CONS').GT.0)
     qm_cpmd_main_r%i_cnst_type= GTRMI(COMLYN,COMLEN,'CONT',1)  ! constraint type (see qm1_cpmd.src)
     if(qm_cpmd_main_r%q_cnst) then
        if(prnlev.ge.2) &
           write(outu,20) 'Constraint will be applied to keep No. of Elect. and Idempotency.'
        if(qm_cpmd_main_r%i_cnst_type<1 .or. qm_cpmd_main_r%i_cnst_type>3) then
           if(prnlev.ge.2) &
           write(outu,20) 'Unknown P constraned minimization option. Use default.'
           qm_cpmd_main_r%i_cnst_type = 1
        end if
        if(prnlev.ge.2) then
           if(qm_cpmd_main_r%i_cnst_type == 1) then
              write(outu,20) &
              'P <- P + P_old*T*P_old + Q_old*T*Q_old, where Q=I-P (Schlegel et al.).'
           else if(qm_cpmd_main_r%i_cnst_type == 2) then
              write(outu,20) &
              'P <- P + (P-Q)*PQ, where Q=I-P.'
           else if(qm_cpmd_main_r%i_cnst_type == 3) then
              write(outu,20) &
              'P <- P + (P-Q)*PQ + 3(P-Q)*PQ^2, where Q=I-P.'
           end if
        end if
     end if
  end if
  !
  ! If E_scf include electronic kinetic energy.
  qm_cpmd_main_r%q_include_kinetic_E = (INDXA(COMLYN,COMLEN,'INCK').GT.0)
  if(qm_cpmd_main_r%q_include_kinetic_E .and. prnlev.ge.2) then
     write(outu,20) 'Elec. K.E. is included in the QUANTM energy.'
  end if
  !
  ! thermostat info.
  qm_cpmd_main_r%Temperature=GTRMF(COMLYN,COMLEN,'TEMP',ZERO)
  qm_cpmd_main_r%Delta      =GTRMF(COMLYN,COMLEN,'TIME',ZERO)/TIMFAC ! conversion done here.
  qm_cpmd_main_r%num_iter   =GTRMI(COMLYN,COMLEN,'NCYC',1)      ! number of iteration between
                                                                ! each time step.
  if(qm_cpmd_main_r%num_iter>1 .and. prnlev.ge.2) then
     write(outu,26) 'Electron density is propagated over ',qm_cpmd_main_r%num_iter, &
                    ' cycle with time step ',qm_cpmd_main_r%Delta*TIMFAC,'.'
  end if

  if(qm_cpmd_main_r%q_admp_md) then
     ! use Nose thermostat.
     qm_cpmd_main_r%q_nose = (INDXA(COMLYN,COMLEN,'NOSE').GT.0)
     if(qm_cpmd_main_r%q_nose) then
        qm_cpmd_main_r%Nose_mass(1)=GTRMF(COMLYN,COMLEN,'MAS1',1000.0d0)
        qm_cpmd_main_r%Nose_mass(2)=GTRMF(COMLYN,COMLEN,'MAS2',50.0d0)
        qm_cpmd_main_r%Nose_mass(3)=GTRMF(COMLYN,COMLEN,'MAS3',30.0d0)
        if(PRNLEV.ge.2) then
           write(outu,22) 'Thermostat temperature:',qm_cpmd_main_r%Temperature
           write(outu,24)                   &
             'Use three-mass Nose chain for thermostat with mass: ', &
              qm_cpmd_main_r%Nose_mass(1),  &
              qm_cpmd_main_r%Nose_mass(2),  &
              qm_cpmd_main_r%Nose_mass(3),'.'
        end if

        ! initialize.
        qm_cpmd_main_r%NOSE(1:3)    = one
        qm_cpmd_main_r%NOSE_Old(1:3)= one
        qm_cpmd_main_r%NOSEV(1:3)   = zero  !
     end if

     !!! Use extended Lagrangian with dissipation (AMN Niklasson, JCP (2009) 130:214109).
     !!qm_cpmd_main_r%q_exl_diss = (INDXA(COMLYN,COMLEN,'DISS').GT.0)
     !!K_order = GTRMI(COMLYN,COMLEN,'KORD',0)
     !!A_scale = GTRMF(COMLYN,COMLEN,'ASCA',zero)
     !!if(K_order < 3 .or. K_order > 9) qm_cpmd_main_r%q_exl_diss =.false.
     !!if(qm_cpmd_main_r%q_exl_diss) then
     !!   if(prnlev >= 2) write(outu,21) &
     !!      'Extended Lagrangian with dissipation with K sum over:',K_order
     !!
     !!   ! initialize the coefficients, kappa, and alpha values.
     !!   ! dummy allocation just to call initcoef_diss
     !!   if(associated(cextr)) deallocate(cextr)
     !!   allocate(cextr(0:K_order))
     !!   call initcoef_diss(A_scale,K_order,coefk,cextr,.true.)
     !!   !
     !!   deallocate(cextr)
     !!end if
  end if

  if(qm_cpmd_main_r%q_curvy_md) then
     qm_cpmd_main_r%q_langevin= (INDXA(COMLYN,COMLEN,'LANG').gt.0)
     if(qm_cpmd_main_r%q_langevin) then
        qm_cpmd_main_r%fbeta_delta=GTRMF(COMLYN,COMLEN,'FBET',zero)
        if(PRNLEV.ge.2) then
           write(outu,22) 'Thermostat temperature:',qm_cpmd_main_r%Temperature
           write(outu,22)                   &
           'Use Langevin thermostat with friction:',qm_cpmd_main_r%fbeta_delta
        end if
     end if
  end if

  ! MTS-LN (Curvy) or MTS (ADMP) approaches.
  qm_cpmd_main_r%num_miter = GTRMI(COMLYN,COMLEN,'MSTE',1)
  if(qm_cpmd_main_r%num_miter > 1 .and. PRNLEV >= 2) then
     if(qm_cpmd_main_r%q_admp_md) then
        write(outu,21) 'ADMP-MTS approach will be used. BOMD cal. every', &
                       qm_cpmd_main_r%num_miter,' MD step.'
     else 
        write(outu,21) 'CURV-MTS-LN approach will be used. BOMD cal. every', &
                       qm_cpmd_main_r%num_miter,' MD step.'
     end if
  end if

  ! now allocate memory
  call allocate_deallocate_qm_cpmd(qm_scf_main_r,qm_cpmd_main_r,q_cpmd,qm_main_r%uhf,.true.)

  ! initialize density velocities.
  qm_cpmd_main_r%vPA = zero

  if(prnlev.ge.2) write(outu,'(A)') ' '

20 format('CPMD_INIT> ',A)
21 format('CPMD_INIT> ',A,I5,A)
22 format('CPMD_INIT> ',A,F12.5)
24 format('CPMD_INIT> ',A,3F10.2,A)
26 format('CPMD_INIT> ',A,I3,A,F8.6,A)

  return
  end subroutine qm_cpmd_init

  subroutine qm_lookup_init(dr,r_min,r_max,dr_b,r_min_b,r_max_b,iunit,q_first)
  !
  ! Prepare look-up tables (write into file).
  !
  use qm1_info,only : mm_main_r
  use qm1_energy_module,only: mmint_prep_core,betaij_prep_lookup
  !
  implicit none
  real(chm_real):: dr,r_min,r_max,dr_b,r_min_b,r_max_b
  integer       :: iunit
  logical       :: q_first

  !write(6,*) 'dr:',dr
  !write(6,*) 'r_min:',r_min,' r_max:',r_max
  if(q_first) then
     ! do the look-up table setup and save into file.
     call mmint_prep_core(dr,r_min,r_max,iunit)
     !
     if(mm_main_r%q_lookup_beta) call betaij_prep_lookup(dr_b,r_min_b,r_max_b)
  else
     ! do read the look-up table info.

  end if
  return
  end subroutine qm_lookup_init


  !=====================================================================
  subroutine fill_mm_coords(natom,x,y,z,cg,mm_coord,mm_chrgs,qm_mm_pair_list,mminb1)
  !=====================================================================
  ! 
  ! fill mm_coord and mm_chrgs for qm/mm calculation, based on mminb info
  ! which is set at CH2MND routine.
  ! 
  use qm1_info, only : qm_main_r,mm_main_r
  use nbndqm_mod, only: map_mmatom_to_group

  implicit none
  integer :: natom
  real(chm_real):: x(natom),y(natom),z(natom),cg(natom)
  real(chm_real):: mm_coord(3,natom),mm_chrgs(natom)
  integer :: qm_mm_pair_list(*),mminb1(*)
  !
  integer :: n1,m,mmatm

  ! find the number of qm-mm pairs.
  n1=0
  do m=qm_main_r%numat+1,natom
     mmatm=mminb1(m)
     if(mmatm.gt.0) then
        n1=n1+1
        ! for mapping back to the original array (in gradient)
        qm_mm_pair_list(n1)=mmatm
        mm_coord(1,n1)     =x(mmatm)
        mm_coord(2,n1)     =y(mmatm)
        mm_coord(3,n1)     =z(mmatm)
        mm_chrgs(n1)       =CG(mmatm)
     end if
  end do
  mm_main_r%NUMATM = n1     ! total number of mm atoms included in qm-mm calculations
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
  use qm1_info, only : qm_main_r,mm_main_r,Aass
  use qmmmewald_module, only : qmmm_ewald_r
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
#if KEY_PARALLEL==1
  integer :: ISTRT_CHECK       ! external function
#endif

  ! only turn on this when using LQMEWD
  !if(.not.LQMEWD) then
  !  qm_main_r%rij_qm_incore =.false.
  !  mm_main_r%rij_mm_incore =.false.
  !end if

  !
#if KEY_PARALLEL==1
  mmynod  = mynod
  nnumnod = numnod
#else
  mmynod  = 0
  nnumnod = 1
#endif
  !
  mstart = 1
  mstop  = numatm
#if KEY_PARALLEL==1
  if(nnumnod>1) mstart = ISTRT_CHECK(mstop,numatm)
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
  if(qm_main_r%rij_qm_incore) then
     ! check and deallocate
     if(associated(qmmm_ewald_r%rijdata_qmqm)) then
        if(size(qmmm_ewald_r%rijdata_qmqm) < 2*icnt_qm) then
           deallocate(qmmm_ewald_r%rijdata_qmqm,stat=ier)
           if(ier.ne.0) call Aass(0,'fill_dist_qm_mm_array','rijdata_qmqm')
           if(associated(qmmm_ewald_r%qmqmerfcx_data)) &
              deallocate(qmmm_ewald_r%qmqmerfcx_data,stat=ier)
           if(ier.ne.0) call Aass(0,'fill_dist_qm_mm_array','qmqmerfcx_data')
        end if
     end if

     ! now allocate needed memory.
     if(.not.associated(qmmm_ewald_r%rijdata_qmqm)) &
        allocate(qmmm_ewald_r%rijdata_qmqm(2,icnt_qm),stat=ier)
     if(ier.ne.0) call Aass(1,'fill_dist_qm_mm_array','rijdata_qmqm')

     if(LQMEWD .and. .not.associated(qmmm_ewald_r%qmqmerfcx_data)) &
        allocate(qmmm_ewald_r%qmqmerfcx_data(icnt_qm),stat=ier)
     if(ier.ne.0) call Aass(1,'fill_dist_qm_mm_array','qmqmerfcx_data')
  end if

  if(mm_main_r%rij_mm_incore) then
     ! check and deallocate
     if(associated(qmmm_ewald_r%rijdata_qmmm)) then
        if(size(qmmm_ewald_r%rijdata_qmmm) < 2*icnt_mm) then
           deallocate(qmmm_ewald_r%rijdata_qmmm,stat=ier)
           if(ier.ne.0) call Aass(0,'fill_dist_qm_mm_array','rijdata_qmmm')
           if(associated(qmmm_ewald_r%qmmmerfcx_data)) &
              deallocate(qmmm_ewald_r%qmmmerfcx_data,stat=ier)
           if(ier.ne.0) call Aass(0,'fill_dist_qm_mm_array','qmmmerfcx_data')
        end if
     end if

     ! now allocate needed memory. allocate memory a bit larger than needed to
     ! avoid allocate/deallocate every time this routine is called.
     if(.not. associated(qmmm_ewald_r%rijdata_qmmm)) &
        allocate(qmmm_ewald_r%rijdata_qmmm(2,icnt_mm+100),stat=ier)
     if(ier.ne.0) call Aass(1,'fill_dist_qm_mm_array','rijdata_qmmm')

     if(LQMEWD .and. .not.associated(qmmm_ewald_r%qmmmerfcx_data)) &
        allocate(qmmm_ewald_r%qmmmerfcx_data(icnt_mm+100),stat=ier)
     if(ier.ne.0) call Aass(1,'fill_dist_qm_mm_array','qmmmerfcx_data')
  end if

  ! now fill the memory.
  if(qm_main_r%rij_qm_incore) then
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
           qmmm_ewald_r%rijdata_qmqm(1,icnt_qm) = Rij      ! rij value
           qmmm_ewald_r%rijdata_qmqm(2,icnt_qm) = one/Rij  ! one/rij value.
        end do
     end do
  end if

  if(mm_main_r%rij_mm_incore) then
     icnt_mm = 0
     do i = 1, numat
        xyz_i(1:3) = qm_coord(1:3,i)
        do j = mstart,mstop     ! 1,numatm
           icnt_mm = icnt_mm + 1
           vec(1:3) = xyz_i(1:3)-mm_coord(1:3,j)
           Rij      = sqrt(vec(1)*vec(1)+vec(2)*vec(2)+vec(3)*vec(3))
           !
           qmmm_ewald_r%rijdata_qmmm(1,icnt_mm) = Rij      ! rij value
           qmmm_ewald_r%rijdata_qmmm(2,icnt_mm) = one/Rij  ! one/rij value
        end do
     end do
  end if

  return
  end subroutine fill_dist_qm_mm_array

  !=====================================================================
  subroutine qmmm_Ewald_init(natom,numat,erfmod,igmsel,            &
                             kmaxX,kmaxY,kmaxZ,KSQmax,kappa,       &
                             q_fast_pme,qcheck)
  !=====================================================================
  ! 
  ! initial setup for qm/mm-ewald calculations.
  ! 
  use qm1_info, only: qm_control_r,qm_main_r,mm_main_r,qm_scf_main_r
  use qmmmewald_module, only : qmmm_ewald_r,qm_ewald_setup,            &
                              allocate_deallocate_qmmm_ewald,         &
                              set_initialize_for_energy_gradient
  use number, only : zero
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none
  integer :: natom,numat,erfmod,kmaxX,kmaxY,kmaxZ,KSQmax
  integer :: igmsel(natom)
  real(chm_real):: kappa
  logical :: q_fast_pme,qcheck,q_ewald_call
  !
  integer :: i,nexcl
  integer :: ier=0
#if KEY_PARALLEL==1
  integer :: ntmp_iatotl(numnod)
  integer :: ISTRT_CHECK                 ! for external function
#endif

  ! quick check.
  if(.not.mm_main_r%LQMEWD) return

  ! setup for qm/mm-ewald summation
  qmmm_ewald_r%natom  = Natom
  qmmm_ewald_r%kmaxqx = kmaxX
  qmmm_ewald_r%kmaxqy = kmaxY
  qmmm_ewald_r%kmaxqz = kmaxZ
  qmmm_ewald_r%ksqmaxq= KSQmax
  qmmm_ewald_r%Erfmod = erfmod
  qmmm_ewald_r%kappa  = kappa
  qm_control_r%q_fast_pme = q_fast_pme  ! for fast routines when CPMD, MD, and PME.

  ! for parallel preparation
  qmmm_ewald_r%iastrt = 1          ! starting of do i=1,natom loop
  qmmm_ewald_r%iafinl = natom      ! ending of the loop
  qmmm_ewald_r%iatotl = natom      ! total term in do-loop

  !
#if KEY_PARALLEL==1
  if(numnod>1) qmmm_ewald_r%iastrt=ISTRT_CHECK(qmmm_ewald_r%iafinl,natom)
  !
  qmmm_ewald_r%iatotl = qmmm_ewald_r%iafinl - qmmm_ewald_r%iastrt + 1
  !
  ! check and iatotl is maximum over the entire node. so, memory allocation over the 
  ! node are same, but smaller than numnod=1.)
  if(numnod.gt.1) then
     ntmp_iatotl = 0
     ntmp_iatotl(mynod+1)=qmmm_ewald_r%iatotl
     call igcomb(ntmp_iatotl,numnod)
     do i=1,numnod
        if(ntmp_iatotl(i).gt.qmmm_ewald_r%iatotl) qmmm_ewald_r%iatotl=ntmp_iatotl(i)
     end do
  end if
#endif


  ! check how many MM atoms are excluded from QM-MM non-bonded interactions.
  nexcl = 0
  do i=1,natom
     if(igmsel(i).eq.5) nexcl=nexcl+1
  end do
  qmmm_ewald_r%nexl_atm =nexcl

  ! now setup the rest.
  call qm_ewald_setup(qmmm_ewald_r%kmaxqx,qmmm_ewald_r%kmaxqy,qmmm_ewald_r%kmaxqz, &
                      qmmm_ewald_r%ksqmaxq,qmmm_ewald_r%totkq,qcheck)
  if(.not.Qcheck) return

  ! allocate memory
  ! provide
  call allocate_deallocate_qmmm_ewald(qm_main_r%numat,mm_main_r%natom,mm_main_r%PMEwald, &
                                      qmmm_ewald_r,.true.)

  ! now, fill exclusion list array.
  if(nexcl.gt.0) then
     nexcl = 0
     do i=1,natom
        if(igmsel(i).eq.5) then
           nexcl=nexcl+1
           qmmm_ewald_r%nexl_index(nexcl)=i
        end if
     end do
  end if

  ! do initialization
  !qmmm_ewald_r%scf_mchg   = zero
  !qmmm_ewald_r%scf_mchg_2 = zero
  qmmm_ewald_r%Kvec       = zero
  qmmm_ewald_r%structfac_mm=zero
  qmmm_ewald_r%empot      = zero
  qmmm_ewald_r%eslf       = zero
  qmmm_ewald_r%dexl_xyz   = zero
  !
  ! do some other initializations: Ktable,qmktable,d_ewald_mm
  !qmmm_ewald_r%Ktable     = zero
  !qmmm_ewald_r%qmktable   = zero
  !qmmm_ewald_r%d_ewald_mm = zero
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
  use qm1_info, only : qm_control_r,qm_main_r,mm_main_r,qm_scf_main_r
  use mndgho_module,only  : qm_gho_info_r
  use qmmmewald_module,only: qm_ewald_calc_kvec,qm_ewald_calc_ktable,qm_ewald_mm_pot, &
                            qm_ewald_qm_pot,qm_ewald_mm_pot_exl,                     &
                            qmmm_ewald_r,get_exl_crd,set_initialize_for_energy_gradient
  !use qmmmpme_module,only : qm_pme_mm_pot
  use qm1_scf_module,only : calc_mulliken,q_construct
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

  QNoPMEwald = .not.mm_main_r%PMEwald

  ! control for CPMD and PME case (Fast PME case?)
  ! Use PME only (for QM-QM pairs as well) when using CPMD.
  ! This is only on when using CPMD and MD, after 1st-MD step, in which the energy
  ! is determined using SCF procedure. So, the CPMD is only fully used after the
  ! first MD step; i.e., from the 2nd MD step.
  !
  ! Note: This should only be used without multiple timestepping for CPMD side. Otherwise,
  !       this will give rise a wrong PME contribution.
  !
  if(qm_control_r%cpmd .and. qm_control_r%md_run .and. qm_control_r%q_fast_pme) then
     if(qm_control_r%qm_e_set) qm_control_r%q_do_cpmd_pme =.true.
  end if

! commented out the following, as it is not using QMPI (probably used for MPI-PI calc.
!!!!! very important.
!!! do special with QMPI: if you once turn on this, cannot turn off later!!!!
!!!#if KEY_PARALLEL==1
!!!  if (QMPI) then
!!!     if(qmmm_ewald_r%iatotl .ne. qmmm_ewald_r%natom) then
!!!        qmmm_ewald_r%iastrt = 1
!!!        qmmm_ewald_r%iafinl = qmmm_ewald_r%natom
!!!        qmmm_ewald_r%iatotl = qmmm_ewald_r%natom
!!!        ! now, memory allocation.
!!!        if(associated(qmmm_ewald_r%Ktable)) deallocate(qmmm_ewald_r%Ktable)
!!!        if(QNoPMEwald) then               ! iatotl=natom
!!!           Allocate(qmmm_ewald_r%Ktable(6,qmmm_ewald_r%iatotl,qmmm_ewald_r%totkq),stat=ier) 
!!!        else
!!!           Allocate(qmmm_ewald_r%Ktable(6,1,1),stat=ier)       ! dummy allocation
!!!        end if
!!!        if(ier.ne.0) call Aass(1,'qmmm_Ewald_setup_and_potential','Ktable')
!!!     end if
!!!  end if
!!!#if KEY_PARALLEL==1

  ! initialization (the following will be done in the subroutine.)
  !qmmm_ewald_r%Ktable     = zero
  !qmmm_ewald_r%qmktable   = zero
  !if(QNoPMEwald) then
  !   qmmm_ewald_r%d_ewald_mm = zero
  !end if
  !qmmm_ewald_r%d_ewald_mm = zero
  if(QNoPMEwald) then
     q_ewald_call=.true.
  else
     q_ewald_call=.false.
  end if
  ! note: qmmm_ewald_r%d_ewald_mm is initialized in KSPACE_qmmm_prep in used of PMEwald.
  call set_initialize_for_energy_gradient(q_ewald_call)

  ! for ktable memory size
  if(QNoPMEwald) then
     itotal=qmmm_ewald_r%iatotl
     itotkq=qmmm_ewald_r%totkq
  else
     itotal=1
     itotkq=1
  end if

  ! copy volumne and reciprocal space lattice vector. They will be copied at
  ! every step as they change during dynamics.
  qmmm_ewald_r%volume     = volume
  qmmm_ewald_r%recip(1:6) = recip(1:6)

  ! for exclusion list: copy x,y,z,cg to qmmm_ewald_r%exl_xyz and qmmm_ewald_r%exl_chg.
  if(qmmm_ewald_r%nexl_atm .gt. 0) call get_exl_crd(x,y,z,cg)

  if(.not.( qm_control_r%q_do_cpmd_pme )) then
     ! 1) Kvector setup
     ! if recip and volume do not change, maybe skipped.  should check the possibility.
     ! in fact, it is done inside of the subroutine.
     call qm_ewald_calc_kvec(qmmm_ewald_r%kappa,qmmm_ewald_r%Volume,qmmm_ewald_r%Recip,  &
                             qmmm_ewald_r%totkq,qmmm_ewald_r%ksqmaxq,                    &
                             qmmm_ewald_r%kmaxqx,qmmm_ewald_r%kmaxqy,qmmm_ewald_r%kmaxqz,&
                             QNoPMEwald,qcheck)
     if(.not.qcheck) return  ! check: if wrong, CHARMM should stop.

     ! 2) Ktables setup
     call qm_ewald_calc_ktable(qmmm_ewald_r%natom,qm_main_r%numat,qm_control_r%qminb,      &
                               qmmm_ewald_r%iastrt,qmmm_ewald_r%iafinl,qmmm_ewald_r%iatotl,&
                               itotal,itotkq,                                              &
                               qmmm_ewald_r%totkq,qmmm_ewald_r%ksqmaxq,                    &
                               qmmm_ewald_r%kmaxqx,qmmm_ewald_r%kmaxqy,qmmm_ewald_r%kmaxqz,&
                               X,Y,Z,qmmm_ewald_r%recip,QNoPMEwald,qcheck)
     if(.not.qcheck) return  ! Check: if wrong, CHARMM should stop.
  end if

  ! 3) Ewald potential setup:
  !    compute the Ewald potential at the qm atom site from all MM atoms; the
  !    self-interaction/ real space contribution / reciprocal space contribution
  !    from "pure" mm atoms.
  !
  ! Note: in parallel run, empot is combined over node from the subroutine as
  !       explained below. so, each node has the same information in the end.
  call qm_ewald_mm_pot(qmmm_ewald_r%natom,qm_main_r%numat,mm_main_r%numatm,         &
                       qmmm_ewald_r%iastrt,qmmm_ewald_r%iafinl,qmmm_ewald_r%iatotl, &
                       itotal,itotkq,                                               &
                       qmmm_ewald_r%totkq,qmmm_ewald_r%ksqmaxq,                     &
                       qmmm_ewald_r%kmaxqx,qmmm_ewald_r%kmaxqy,qmmm_ewald_r%kmaxqz, &
                       mm_main_r%mm_coord,mm_main_r%mm_chrgs,qm_main_r%qm_coord,    &
                       qmmm_ewald_r%empot,QNoPMEwald)
  ! 3-1) the pme version
  if(.not.QNoPMEwald) then
     if(qm_control_r%q_do_cpmd_pme) then
        ! Q = PA + PB (UHF) or 2*PA (constructed in fockx).
        if(qm_main_r%uhf) then
           ! for UHF
           call q_construct(qm_scf_main_r%PA,qm_scf_main_r%PB,qm_scf_main_r%Q, &
                            qm_scf_main_r%dim_linear_norbs,qm_scf_main_r%dim_linear_fock, &
                            qm_main_r%uhf)
        else
           ! for RHF
           call q_construct(qm_scf_main_r%PA,qm_scf_main_r%PA,qm_scf_main_r%Q, &
                            qm_scf_main_r%dim_linear_norbs,qm_scf_main_r%dim_linear_fock, &
                            qm_main_r%uhf)
        end if
        call calc_mulliken(qm_scf_main_r%dim_numat,qm_main_r%nat,qm_main_r%nfirst, &
                           qm_main_r%num_orbs,                         &
                           qm_scf_main_r%Q,mm_main_r%qm_charges,       &
#if KEY_PARALLEL==1
                           numnod,                                     & 
#endif
                           qm_scf_main_r%dim_linear_fock)
     end if
     !! since before mndene call, the pme potential is evaluated. So, this routine now
     !! will do only the summing up of empot values. Do it here explicitly.
     !!call qm_pme_mm_pot(qmmm_ewald_r%natom,qm_main_r%numat,x,y,z,cg,mm_main_r%qm_charges, &
     !!                   qmmm_ewald_r%recip,qmmm_ewald_r%volume,qmmm_ewald_r%empot,        &
     !!                   qmmm_ewald_r%empot_pme,qmmm_ewald_r%empot_qm_pme,                 &
     !!                   qmmm_ewald_r%kappa)
! see this summation below in the scf_energy routine.
!     qmmm_ewald_r%empot(1:qm_main_r%numat) = qmmm_ewald_r%empot(1:qm_main_r%numat) + &
!                                             qmmm_ewald_r%empot_pme(1:qm_main_r%numat)
!
!#if KEY_PARALLEL==1
!     if (numnod.gt.1) then
!        call GCOMB(qmmm_ewald_r%empot_pme,qm_main_r%numat)     ! qm-mm part
!        call GCOMB(qmmm_ewald_r%empot_qm_pme,qm_main_r%numat)  ! qm-qm part (energy should be 1/2.)
!     end if
!#endif
  end if

  ! 4) for the qm-mm excluded list
  if(qmmm_ewald_r%nexl_atm.gt.0) &
     call qm_ewald_mm_pot_exl(qm_main_r%numat,qm_main_r%qm_coord,qmmm_ewald_r%empot) 
! see below scf_energy routine.
!#if KEY_PARALLEL==1
!  ! sum up values over all nodes.
!  if (numnod > 1) call GCOMB(qmmm_ewald_r%empot,qm_main_r%numat)
!#endif

  if(.not.Qcheck) return
  ! here, empot should be done! (also communicated between each node).

  ! 5) for self qm-qm (image) atoms
  call qm_ewald_qm_pot(qmmm_ewald_r%natom,qm_main_r%numat,qmmm_ewald_r%totkq,&
                       qmmm_ewald_r%ksqmaxq,qmmm_ewald_r%kmaxqx,             &
                       qmmm_ewald_r%kmaxqy,qmmm_ewald_r%kmaxqz,              &
                       qm_main_r%qm_coord,qmmm_ewald_r%eslf)

  ! end of qm/mm-ewald potential part.
  !
  return
  end subroutine qmmm_Ewald_setup_and_potential


  !=====================================================================
  subroutine scf_energy(natom,xim,yim,zim,icall)
  !=====================================================================
  ! 
  ! Compute energy of QM+QM/MM (electrostatic) part.
  !
  use qm1_info, only : qm_control_r,qm_main_r,mm_main_r,qm_scf_main_r
  use mndgho_module, only : qm_gho_info_r
  use qm1_scf_module, only : scf_iter,qm_scf_diis_r,qm_fockmd_diis_r,allocate_deallocate_qm_diis,bond_analysis
  use qmmmewald_module ,only : qmmm_ewald_r,qm_ewald_core,qm_pme_energy_corr
  use qm1_energy_module, only : guessp,wstore_comm  ! hcorep,mmint
  use qm1_lagdynamics_module, only: qm_cpmd_main_r,cpmd_energy,cpmd_energy_only
  use number, only : zero,half,two
  use qm1_constant,only: EVCAL,ccelec
  use contrl,only : ISTEPQM
#if KEY_PARALLEL==1
  use parallel
#endif

  integer :: natom,icall
  real(chm_real) :: xim(natom),yim(natom),zim(natom) 
  
  ! local variables
  integer :: dim_iwork,dim_iwork6
  logical,save :: qfirst=.true.
  logical :: q_do_cpmd,q_first_bomd_mts

  integer :: i,k,ks
  ! debug
  real(chm_real) :: e_temp
  integer, save :: size_n=0
  integer, save :: msize,mstart,mstop
  integer, save :: ifockmd_counter=0,imd_counter=0
#if KEY_PARALLEL==1
  integer, save  :: JPARPT_local(0:MAXNODE)
#endif

  if(size_n /= qm_scf_main_r%dim_linear_norbs) then
     mstart = 1
     mstop  = qm_scf_main_r%dim_linear_norbs
#if KEY_PARALLEL==1
     JPARPT_local(0) = 0
     do i=1,numnod
        JPARPT_local(i)= mstop*i/numnod  ! for linear vector
     end do
     mstart = JPARPT_local(mynod)+1
     mstop  = JPARPT_local(mynod+1)
#endif
     msize  = mstop - mstart + 1
  end if

!  ! initialization
!  qm_main_r%enuclr_qmqm = zero
!  qm_main_r%enuclr_qmmm = zero
!
!  ! integral calculations for scf.
!  call hcorep(qm_scf_main_r%H,qm_scf_main_r%W,qm_scf_main_r%dim_linear_norbs,&
!              qm_scf_main_r%dim_linear_fock,qm_scf_main_r%dim_linear_fock2,  &
!              qm_main_r%enuclr_qmqm)
!
!  ! for qm/mm
!  if(mm_main_r%numatm.gt.0) call mmint(qm_scf_main_r%H, &
!                                       qm_scf_main_r%dim_linear_norbs,  &
!                                       qm_main_r%enuclr_qmmm)
!
!  ! store MNDO integras in square form:
!  ! 1. complete MNDO integrals and transpose the W matrix.
!  ! 2. this is doing the leftover from wstore subroutine in hcorep routine.
!  ! 3. w_linear contains lower-trianglular part of the W matrix.
!  if(associated(w_linear)) deallocate(w_linear)
!  allocate(w_linear(qm_scf_main_r%dim_linear_fock*(qm_scf_main_r%dim_linear_fock+1)/2)) 
!  call wstore(qm_scf_main_r%W,w_linear,qm_scf_main_r%dim_linear_fock,0, &
!              qm_main_r%numat,qm_main_r%uhf)
  ! only for communication.
#if KEY_PARALLEL==1
  if (numnod > 1) then
     call gcomb(w_linear,qm_scf_main_r%dim_linear_fock*(qm_scf_main_r%dim_linear_fock+1)/2)
  end if
#endif
  !====================START OPENMP PARALLEL========================!
!$omp parallel NUM_THREADS(2)
!$omp sections
!$omp section
  call wstore_comm(qm_scf_main_r%W,w_linear,qm_scf_main_r%dim_linear_fock)
  !------------------------END SECTION 1----------------------------!

!$omp section
!
! do communication here.
#if KEY_PARALLEL==1
  if (numnod.gt.1) then
     ! H: 1-e matrix
     ! W: 2-e exchange/replusion integrals. (see below)
     call gcomb(qm_scf_main_r%H,qm_scf_main_r%dim_linear_norbs)
     call gcomb(qm_main_r%enuclr_qmqm,1)
     !!!call gcomb(qm_scf_main_r%W,qm_scf_main_r%dim_linear_fock2) ! this is done within wstore
     !!!call gcomb(qm_main_r%enuclr_qmmm,1) : see below after adding qm_ewald_core.
  end if
#endif
!
!
!  ! for guess density
!  if(qfirst) then
!     qfirst=.false.
!     if(qm_main_r%uhf) then
!        call guessp(qm_scf_main_r%PA,qm_scf_main_r%PB,qm_scf_main_r%dim_linear_norbs)
!     else
!        call guessp(qm_scf_main_r%PA,qm_scf_main_r%PA,qm_scf_main_r%dim_linear_norbs)
!     end if
!  end if

  ! PME related information (data communication): sum up values over all nodes.
  ! (see above qmmm_Ewald_setup_and_potential routine.)
  if(mm_main_r%PMEwald) qmmm_ewald_r%empot(1:qm_main_r%numat) = qmmm_ewald_r%empot(1:qm_main_r%numat) + &
                                             qmmm_ewald_r%empot_pme(1:qm_main_r%numat)
#if KEY_PARALLEL==1
  if (numnod.gt.1 .and. mm_main_r%LQMEWD) then
     if(mm_main_r%PMEwald) then
        call GCOMB(qmmm_ewald_r%empot_pme,qm_main_r%numat)     ! qm-mm part
        call GCOMB(qmmm_ewald_r%empot_qm_pme,qm_main_r%numat)  ! qm-qm part (energy should be 1/2.)
     end if
     call GCOMB(qmmm_ewald_r%empot,qm_main_r%numat)
     call gcombr(qmmm_ewald_r%eslf,qm_main_r%numat*qm_main_r%numat) ! (see qm_ewald_qm_pot routine).
  end if
#endif
  !-------------------------END SECTION 2---------------------------!
!$omp end sections
!$omp end parallel
  !======================END OPENMP PARALLEL========================!

  ! memory
  if(allocated(w_linear)) deallocate(w_linear)

  ! for controlling CPMD:
  ! in the very first md calc, the PA should be computed from scf step.
  q_do_bomd_mts     = .false.
  q_do_mts_oldstep  = .false.
  q_do_cpmd_noupdate= .false.
  if(qm_control_r%cpmd .and. qm_control_r%md_run) then
     ntime_step_md=ntime_step_md + 1  ! md step counter.

     ! some implementation note:
     ! it seems in verlet/leap-from verlet (dynamc.src), when the dynamics print energy/average 
     ! properties, it appears that the energy (after printing averages) is the same before 
     ! the average printing. On the other hands, in velocity verlet (dynamvv.src), they are 
     ! different. Why? Need to track it down.
     !
     ! For now, I am only work with dynamc.src, and turn off cpmd on dynamvv.src.
     !
     if(ISTEPQM /= ntime_step_md) then
        ! outside of the md routine: density should not be propagated.
        !q_do_cpmd_noupdate = .true.  ! only energy is computed.
        !ntime_step_md = ISTEPQM
     else
        ! within md routine: density will be propagated as normal.
        q_do_cpmd_noupdate = .false. ! full energy/density propagation.
     end if

     ! here, we use two step preparation for cpmd run. 
     ! In the first step, full scf is done to determine PA_old.
     ! In the 2nd   step, full scf is done to determine PA followed by one dt step propagation
     !                    of density for the next MD step, in which cpmd is fully on.
     ! The same principle is applied to MTS case (1st and 2nd MD steps).
     if(ntime_step_md == 1) then
        ! MD start: full scf calc, propagation of coords, and copy for old density.
        q_do_mts_oldstep = .true.
        q_do_cpmd        = .false.
     else if(ntime_step_md == 2) then
        ! Second MD step: full scf calc., propagation of coords, and preparation of density.
        q_do_bomd_mts    = .true.
        q_do_cpmd        = .false.
     else
        ! now, everything is ready for the "full" cpmd run.
        q_do_cpmd =.true.
        if(qm_cpmd_main_r%num_miter>1) then
           ! using MTS.
           if(mod(ntime_step_md-1,qm_cpmd_main_r%num_miter) == 0) then
              ! full scf calc, propagation of coords, and copy for old density.
              q_do_mts_oldstep = .true.
              q_do_bomd_mts    = .false.
              q_do_cpmd        = .false.
           !else if(mod(ntime_step_md,qm_cpmd_main_r%num_miter) == 0) then
           else if(mod(ntime_step_md-1,qm_cpmd_main_r%num_miter) == 1) then
              ! Second MD step: full scf calc., propagation of coords, and preparation of density.
              q_do_mts_oldstep = .false.
              q_do_bomd_mts    = .true.
              q_do_cpmd        = .false.
           else
              q_do_mts_oldstep = .false.
              q_do_bomd_mts    = .false.
              q_do_cpmd        = .true.
           end if
        end if
     end if
  else
     q_do_cpmd =.false.    ! only use scf calc.

     !
     if(qm_control_r%q_dxl_bomd .and. qm_control_r%md_run) then
        ! first memory.
        if(.not. associated(pa_exp)) then
           allocate(pa_exp(Kth_sum_order,msize))
           allocate(pa_aux(msize))
           allocate(pa_anew(msize))
           pa_exp(1:Kth_sum_order,1:msize) = zero
           imd_counter = 0   ! reset counter.
        end if
        imd_counter = imd_counter + 1

        ! now, propagate pa_aux and copy it to pa init.
        if(imd_counter > Kth_sum_order) then
           qm_control_r%q_do_dxl_scf =.true.  ! do dxl-bomd scf cycle. (finish at N_scf_step.)
           ks = Kth_sum_order
           ! determine pa_aux(i+1) = 2*pa_aux(i) - pa_aux(i-1) + kappa*(PA_old(i)-pa_aux(i)) + alpha*sum over k=0,K
           ! and copy to PA_initial = pa_aux(i+1)
!$omp parallel do NUM_THREADS(2)
           do i=1,msize
              pa_anew(i) = coefk*qm_scf_main_r%PA(mstart+i-1) + cextr(0)*pa_aux(i) + &
                           dot_product(cextr(1:ks),pa_exp(1:ks,i))
              ! copy
              qm_scf_main_r%PA(mstart+i-1) = pa_anew(i)
           end do
!$omp end parallel do
#if KEY_PARALLEL==1
           if(numnod>1) call VDGBRE(qm_scf_main_r%PA,JPARPT_local)
#endif
        else
           qm_control_r%q_do_dxl_scf =.false.  ! do full scf cycle.
        end if
     end if

     !
     if(qm_control_r%q_fockmd .and. qm_control_r%md_run) then
        if(.not.qm_control_r%q_do_fockmd_scf) ifockmd_counter = 0 ! initialization.
        ifockmd_counter = ifockmd_counter + 1
        qm_control_r%q_do_fockmd_scf=.true.
     else
        qm_control_r%q_do_fockmd_scf=.false.
     end if
  end if

  ! now, call scf_iteration
  icall = 0
  dim_iwork =9*qm_scf_main_r%dim_numat
  dim_iwork6=max(6*dim_iwork,qm_main_r%norbs)
  !if(mynod==0) write(6,*)ISTEPQM,q_do_cpmd,q_do_cpmd_noupdate,q_do_bomd_mts,q_do_mts_oldstep
  if(q_do_cpmd) then
     if(q_do_cpmd_noupdate) then
        ! This is the case, during MD, the energy is computed (based on the same coordinates).
        ! This occurs because at some time, it is existed from the MD do loop and reports 
        ! some statistical information (average, fluctuation, etc) and goes back to the main
        ! md do-loop. While there is one extra energy call before based on the present coords
        ! without them being updated. So, we do not propagate the density in this energy call.
        !
        ! Note that, in this case, 
        !
        ! for RHF
        call cpmd_energy_only(qm_main_r%elec_eng,qm_scf_main_r%H,qm_scf_main_r%W,   &
                      qm_scf_main_r%Q,                                              &
                      qm_scf_main_r%FA,qm_scf_main_r%PA,qm_scf_main_r%PA, & ! qm_cpmd_main_r%PA_scf,qm_cpmd_main_r%PA_scf, & 
                      qm_scf_main_r%dim_numat,                                      &
                      qm_scf_main_r%dim_norbs,qm_scf_main_r%dim_linear_norbs,       &
                      qm_scf_main_r%dim_linear_fock,                                &
                      qm_scf_main_r%dim_linear_fock2,qm_scf_main_r%dim_scratch,     &
                      qm_main_r%uhf)
        if(qm_gho_info_r%q_gho) then
           ! for GHO method: since CA is not explicitly used within the scf cycle. So,
           ! copy transformed orbitals to AO basis here, used by Mulliken analysis.
           qm_gho_info_r%PHO(1:qm_scf_main_r%dim_linear_norbs) = &
                           two*qm_cpmd_main_r%PA_old(1:qm_scf_main_r%dim_linear_norbs)
        end if
     else
        ! when using cpmd run. (also, not first md step.)
        call cpmd_energy(qm_main_r%elec_eng,qm_scf_main_r%H,qm_scf_main_r%W, &
                         qm_scf_main_r%Q,                                    &
                         qm_scf_main_r%FA,qm_scf_main_r%PA,qm_scf_main_r%PA, &
                         qm_scf_main_r%dim_numat,                            &
                         qm_scf_main_r%dim_norbs,qm_scf_main_r%dim_linear_norbs,  &
                         qm_scf_main_r%dim_linear_fock,                      &
                         qm_scf_main_r%dim_linear_fock2,qm_scf_main_r%dim_scratch, &
                         qm_main_r%uhf,.false.)
     end if
  else
     ! regular SCF-based energy calculation.
     if(qm_main_r%uhf) then
        ! for UHF
        call scf_iter(qm_main_r%elec_eng,qm_scf_main_r%H,qm_scf_main_r%W, &
                      qm_scf_main_r%Q,                                    &
                      qm_scf_main_r%CA,qm_scf_main_r%DA,qm_scf_main_r%EA, &
                      qm_scf_main_r%FA,qm_scf_main_r%PA,                  &
                      qm_scf_main_r%CB,qm_scf_main_r%DB,qm_scf_main_r%EB, &
                      qm_scf_main_r%FB,qm_scf_main_r%PB,                  &
                      qm_scf_main_r%dim_numat,                            &
                      qm_scf_main_r%dim_norbs,qm_scf_main_r%dim_linear_norbs,  &
                      qm_scf_main_r%dim_linear_fock,                      &
                      qm_scf_main_r%dim_linear_fock2,qm_scf_main_r%dim_scratch,ifockmd_counter, & 
                      dim_iwork6,qm_scf_main_r%iwork,icall,qm_main_r%uhf,q_do_cpmd) ! q_do_bomd_mts)
     else
        ! for RHF
        call scf_iter(qm_main_r%elec_eng,qm_scf_main_r%H,qm_scf_main_r%W, &
                      qm_scf_main_r%Q,                                    &
                      qm_scf_main_r%CA,qm_scf_main_r%DA,qm_scf_main_r%EA, &
                      qm_scf_main_r%FA,qm_scf_main_r%PA,                  &
                      qm_scf_main_r%CA,qm_scf_main_r%DA,qm_scf_main_r%EA, &
                      qm_scf_main_r%FA,qm_scf_main_r%PA,                  &
                      qm_scf_main_r%dim_numat,                            &
                      qm_scf_main_r%dim_norbs,qm_scf_main_r%dim_linear_norbs,  &
                      qm_scf_main_r%dim_linear_fock,                      &
                      qm_scf_main_r%dim_linear_fock2,qm_scf_main_r%dim_scratch,ifockmd_counter, &
                      dim_iwork6,qm_scf_main_r%iwork,icall,qm_main_r%uhf,q_do_cpmd) ! q_do_bomd_mts)
     end if

     ! if scf failes, do 2nd attempt with
     ! (a) diis extrapolation on
     ! (b) new diagonal density guess
     ! (c) no other density extrapolation.
     if(icall.eq.-1 .and. .not.qm_control_r%q_diis) then
        ! turn on diis
        qm_control_r%q_diis=.true.
        ! if so, allocate memory for diis?
        if(.not.associated(qm_scf_diis_r%FDA)) &
           call allocate_deallocate_qm_diis(qm_scf_main_r,qm_scf_diis_r,qm_fockmd_diis_r, &
                                            qm_control_r%q_diis,qm_control_r%q_fockmd,    &
                                            qm_control_r%imax_fdiss,                      &
                                            qm_main_r%uhf,.true.)

        ! in previous scf_iter returned with icall=-1, so do not set icall = 0.
        if(qm_main_r%uhf) then
           ! for UHF
           call guessp(qm_scf_main_r%PA,qm_scf_main_r%PB,qm_scf_main_r%dim_linear_norbs)
           call scf_iter(qm_main_r%elec_eng,qm_scf_main_r%H,qm_scf_main_r%W, &
                         qm_scf_main_r%Q,                                    &
                         qm_scf_main_r%CA,qm_scf_main_r%DA,qm_scf_main_r%EA, &
                         qm_scf_main_r%FA,qm_scf_main_r%PA,                  &
                         qm_scf_main_r%CB,qm_scf_main_r%DB,qm_scf_main_r%EB, &
                         qm_scf_main_r%FB,qm_scf_main_r%PB,                  &
                         qm_scf_main_r%dim_numat,                            &
                         qm_scf_main_r%dim_norbs,qm_scf_main_r%dim_linear_norbs,  &
                         qm_scf_main_r%dim_linear_fock,                      &
                         qm_scf_main_r%dim_linear_fock2,qm_scf_main_r%dim_scratch,ifockmd_counter, &
                         dim_iwork6,qm_scf_main_r%iwork,icall,qm_main_r%uhf,q_do_cpmd) ! q_do_bomd_mts)
        else
           ! for RHF
           call guessp(qm_scf_main_r%PA,qm_scf_main_r%PA,qm_scf_main_r%dim_linear_norbs)
           call scf_iter(qm_main_r%elec_eng,qm_scf_main_r%H,qm_scf_main_r%W, &
                         qm_scf_main_r%Q,                                    &
                         qm_scf_main_r%CA,qm_scf_main_r%DA,qm_scf_main_r%EA, &
                         qm_scf_main_r%FA,qm_scf_main_r%PA,                  &
                         qm_scf_main_r%CA,qm_scf_main_r%DA,qm_scf_main_r%EA, &
                         qm_scf_main_r%FA,qm_scf_main_r%PA,                  &
                         qm_scf_main_r%dim_numat,                            &
                         qm_scf_main_r%dim_norbs,qm_scf_main_r%dim_linear_norbs,  &
                         qm_scf_main_r%dim_linear_fock,                      &
                         qm_scf_main_r%dim_linear_fock2,qm_scf_main_r%dim_scratch,ifockmd_counter, &
                         dim_iwork6,qm_scf_main_r%iwork,icall,qm_main_r%uhf,q_do_cpmd) ! q_do_bomd_mts)
        end if
     end if

     ! if it is cpmd and md-run
     ! 1) the PA was computed from the above scf procedure.
     ! 2) we should call cpmd_energy to prepare the cpmd run.
     if(q_do_mts_oldstep) then
        ! for the first step: we only need PA_old.
        if(qm_gho_info_r%q_gho) then
           qm_cpmd_main_r%PA_old(1:qm_gho_info_r%lin_norbhb)       = &
                          qm_gho_info_r%PAHB(1:qm_gho_info_r%lin_norbhb)
           if(qm_cpmd_main_r%q_curvy_md) qm_cpmd_main_r%vPA(1:qm_gho_info_r%lin_norbhb) = zero
        else
           qm_cpmd_main_r%PA_old(1:qm_scf_main_r%dim_linear_norbs) = &
                          qm_scf_main_r%PA(1:qm_scf_main_r%dim_linear_norbs)
           if(qm_cpmd_main_r%q_curvy_md) qm_cpmd_main_r%vPA(1:qm_scf_main_r%dim_linear_norbs) = zero
        end if
     else if(q_do_bomd_mts) then
        ! here, energy and fock matrix (gradient) are ready, and density propagation is done
        ! after the evalulation of gradient (see scf_gradient).
        call cpmd_energy(qm_main_r%elec_eng,qm_scf_main_r%H,qm_scf_main_r%W, &
                         qm_scf_main_r%Q,                                    &
                         qm_scf_main_r%FA,qm_scf_main_r%PA,qm_scf_main_r%PA, &
                         qm_scf_main_r%dim_numat,                            &
                         qm_scf_main_r%dim_norbs,qm_scf_main_r%dim_linear_norbs,  &
                         qm_scf_main_r%dim_linear_fock,                      &
                         qm_scf_main_r%dim_linear_fock2,qm_scf_main_r%dim_scratch, &
                         qm_main_r%uhf,.true.)

        qm_control_r%qm_e_set=.true. ! from next md step, it is set to use cpmd.
     end if
  end if

  ! contributions from the Ewald sum-core interaction, only after scf converges.
  if(mm_main_r%LQMEWD) then
     qm_main_r%enuclr_qmmm = qm_main_r%enuclr_qmmm &
                            +qm_ewald_core(qm_main_r%numat,qm_main_r%nat, &
                                           mm_main_r%qm_charges)
     
     ! complete the energy term
     if(qm_control_r%q_do_cpmd_pme) then
        qm_control_r%E_ewald_corr =(qm_pme_energy_corr(qm_main_r%numat,mm_main_r%qm_charges)) &
                                   *ccelec  ! =EVCAL*EV*A0
#if KEY_PARALLEL==1
        call gcomb(qm_control_r%E_ewald_corr,1)
        if(mynod /= 0) qm_control_r%E_ewald_corr = zero 
        !if(mynod == 0) write(6,'(A,F12.5)') 'E_corr:',qm_control_r%E_ewald_corr
#endif
     end if
  end if

  !
#if KEY_PARALLEL==1
  if (numnod.gt.1) call gcomb(qm_main_r%enuclr_qmmm,1)
#endif

  ! for analysis
  if((qm_control_r%q_bond_order .or. qm_control_r%q_m_charge) .and. .not. qm_main_r%uhf) then
     call bond_analysis(qm_scf_main_r%dim_numat,qm_scf_main_r%PA,             &
                        qm_scf_main_r%dim_norbs,qm_scf_main_r%dim_linear_fock)
  end if

  ! for DXL-BOMD
  if(qm_control_r%q_dxl_bomd .and. qm_control_r%md_run) then
     if(imd_counter <= Kth_sum_order) then
        ! k=0; pa_aux(i) = PA_scf(i), etc.
        if(imd_counter == 1) pa_aux(1:msize) = qm_scf_main_r%PA(mstart:mstop) ! copy old pa.
!$omp parallel do private(ii,k) NUM_THREADS(2)
        do ii=1,msize
           do k = imd_counter,2,-1
              pa_exp(k,ii) = pa_exp(k-1,ii)
           end do
           pa_exp(1,ii) = pa_aux(ii)
        end do
!$omp end parallel do
     else
        ! shift array by one column to update to the current auxiliary PA arrays.
!$omp parallel do private(i,k) NUM_THREADS(2)
        do i=1,msize
           do k = Kth_sum_order,2,-1
              pa_exp(k,i) = pa_exp(k-1,i)
           end do
           pa_exp(1,i) = pa_aux(i)           ! copy old pa_aux.
           pa_aux(i)   = pa_anew(i)          ! copy current pa_aux to old pa_aux.
        end do
!$omp end parallel do
     end if
  end if

  ! save energy
  ! note: qm_main_r%ener_atomic is already in kcal/mol (qm_info_setup)
#if KEY_PARALLEL==1
  if(mynod.eq.0) then
#endif
     qm_control_r%E_scf    = EVCAL*qm_main_r%elec_eng
     qm_control_r%E_nuclear= EVCAL*(qm_main_r%enuclr_qmqm+qm_main_r%enuclr_qmmm)
     qm_control_r%E_total  = qm_main_r%HofF_atomic     &
                            +qm_control_r%E_scf        &
                            +qm_control_r%E_nuclear    &
                            -qm_main_r%ener_atomic
#if KEY_PARALLEL==1
  else
     qm_control_r%E_total  = zero
  end if
  ! for debug
  !if(mynod.eq.0) then
#endif
  !  write(6,*)'e_scf=',qm_main_r%elec_eng
  !  write(6,*)'E_nuclear=',qm_main_r%enuclr_qmqm+qm_main_r%enuclr_qmmm
  !  write(6,*)'HofF_atomic=',qm_main_r%HofF_atomic
  !  write(6,*)'ener_atomic=',qm_main_r%ener_atomic
  !  write(6,*)'e_total=',ISTEPQM,qm_control_r%E_total
#if KEY_PARALLEL==1
  !endif
#endif
  !
  return
  end subroutine scf_energy


  !=====================================================================
  subroutine scf_energy_prep(natom,xim,yim,zim)
  !=====================================================================
  ! 
  ! Prepare for the routine to compute energy of QM+QM/MM (electrostatic) part.
  !
  use qm1_info, only : qm_main_r,mm_main_r,qm_scf_main_r
  use qm1_energy_module, only : hcorep,mmint,guessp,wstore
  use number, only : zero
#if KEY_PARALLEL==1
  use parallel
#endif

  integer :: natom
  real(chm_real) :: xim(natom),yim(natom),zim(natom) 
  
  ! local variables
  logical,save :: qfirst=.true.

  ! initialization
  qm_main_r%enuclr_qmqm = zero
  qm_main_r%enuclr_qmmm = zero

  ! integral calculations for scf.
  call hcorep(qm_scf_main_r%H,qm_scf_main_r%W,qm_scf_main_r%dim_linear_norbs,&
              qm_scf_main_r%dim_linear_fock,qm_scf_main_r%dim_linear_fock2,  &
              qm_main_r%enuclr_qmqm)

  ! for qm/mm
  if(mm_main_r%numatm.gt.0) call mmint(qm_scf_main_r%H, &
                                       qm_scf_main_r%dim_linear_norbs,  &
                                       qm_main_r%enuclr_qmmm)

! communication will be done in scf_energy routine.
!#if KEY_PARALLEL==1
!  if (numnod.gt.1) then
!     ! H: 1-e matrix
!     ! W: 2-e exchange/replusion integrals. (see below)
!     call gcomb(qm_scf_main_r%H,qm_scf_main_r%dim_linear_norbs)
!     call gcomb(qm_main_r%enuclr_qmqm,1)
!     !!!call gcomb(qm_scf_main_r%W,qm_scf_main_r%dim_linear_fock2) ! this is done within wstore
!     !!!call gcomb(qm_main_r%enuclr_qmmm,1) : see below after adding qm_ewald_core.
!  end if
!#endif
!
  ! store MNDO integras in square form: 
  ! 1. complete MNDO integrals and transpose the W matrix.
  ! 2. this is doing the leftover from wstore subroutine in hcorep routine.
  ! 3. w_linear contains lower-trianglular part of the W matrix.
  if(allocated(w_linear)) deallocate(w_linear)
  allocate(w_linear(qm_scf_main_r%dim_linear_fock*(qm_scf_main_r%dim_linear_fock+1)/2))
  call wstore(qm_scf_main_r%W,w_linear,qm_scf_main_r%dim_linear_fock,0, &
              qm_main_r%numat,qm_main_r%uhf)

  ! for guess density
  if(qfirst) then
     qfirst=.false.
     if(qm_main_r%uhf) then
        call guessp(qm_scf_main_r%PA,qm_scf_main_r%PB,qm_scf_main_r%dim_linear_norbs)
     else
        call guessp(qm_scf_main_r%PA,qm_scf_main_r%PA,qm_scf_main_r%dim_linear_norbs)
     end if
  end if
  !
  return
  end subroutine scf_energy_prep


  !=====================================================================
  subroutine scf_gradient(natom,xim,yim,zim,dx,dy,dz)
  !=====================================================================
  ! 
  ! Compute energy of QM+QM/MM (electrostatic) gradient part.
  ! gradients are computed as a finite difference of energy.
  !
  use qm1_info, only : qm_control_r,qm_main_r,mm_main_r,qm_scf_main_r
  use qm1_gradient_module, only : qmqm_gradient,qmmm_gradient
  use mndgho_module, only : qm_gho_info_r,GHO_expansion,GHO_expansion_cpmd
  use number, only : zero
  use qm1_lagdynamics_module, only: qm_cpmd_main_r,update_PA,density_const,update_PA_old
#if KEY_PARALLEL==1
  use parallel
#endif

  integer :: natom,i,n
  real(chm_real):: xim(natom),yim(natom),zim(natom),dx(natom),dy(natom),dz(natom)
  real(chm_real):: PL,PM
  logical :: q_kinetic_print,q_do_update,q_do_cpmd,q_do_cpmd_first
  integer :: mstart,mend
#if KEY_PARALLEL==1
  integer :: ISTRT_CHECK            ! external function
#endif

  !
  mstart = 1
  mend   = mm_main_r%numatm
#if KEY_PARALLEL==1
  if(numnod>1) mstart = ISTRT_CHECK(mend,mm_main_r%numatm)
#endif

  ! q_do_update ==.false.; only do energy calc. without density being updated.
  q_do_update =.true.
  if(qm_control_r%cpmd .and. qm_control_r%md_run .and. q_do_cpmd_noupdate) q_do_update =.false.

  ! control of cpmd call, in particular, for starting and mts update.
  q_do_cpmd       =.false.
  q_do_cpmd_first =.false.
  if(qm_control_r%cpmd .and. qm_control_r%md_run) then
     q_do_cpmd =.true.
     if(q_do_mts_oldstep) q_do_cpmd       =.false.
     if(q_do_bomd_mts)    q_do_cpmd_first =.true.  ! this is for special update at the begining.
  end if

  ! 
  ! initialization
  qm_main_r%qm_grads(1:3,1:qm_main_r%numat) = zero
  if(mm_main_r%numatm.gt.0) mm_main_r%mm_grads(1:3,mstart:mend) = zero
  
  ! for switching function
  if(mm_main_r%q_switch) mm_main_r%dxyz_sw(1:6,1:mm_main_r%isize_swt_array) =zero
  !
  if(qm_main_r%uhf) then
     if(q_do_update) then
        ! this is for the regular full scf/cpmd calculations. 

        ! for qm-qm gradient components
        call qmqm_gradient(qm_scf_main_r%PA,qm_scf_main_r%PB,qm_scf_main_r%dim_linear_norbs)  

        ! for qm-mm gradient components
        if(mm_main_r%numatm.gt.0) call qmmm_gradient(qm_scf_main_r%PA,qm_scf_main_r%PB, &
                                                     natom,xim,yim,zim,                 &
                                                     qm_scf_main_r%dim_linear_norbs,    &
                                                     qm_main_r%numat,mm_main_r%numatm,  &
                                                     qm_main_r%NAT,qm_main_r%NFIRST,qm_main_r%NLAST, &
                                                     qm_main_r%num_orbs,qm_scf_main_r%INDX, &
                                                     qm_main_r%qm_coord,mm_main_r%mm_coord, &
                                                     mm_main_r%mm_chrgs,                    &
                                                     qm_scf_main_r%CORE_mat,qm_scf_main_r%WW, &
                                                     qm_scf_main_r%RI,qm_scf_main_r%YY,     &
                                                     qm_main_r%qm_grads,mm_main_r%mm_grads, &
                                                     qm_control_r%q_am1_pm3,mstart,mend)
     else
        ! this is for the case, where the energy is called during MD, but without its
        ! coordinates being updated. Therefore, the energy call is basically duplicated
        ! in which the second energy call requires it being only need for energy without
        ! density being updated.

        ! for qm-qm gradient components
        call qmqm_gradient(qm_cpmd_main_r%PA_scf,qm_cpmd_main_r%PB_scf,qm_scf_main_r%dim_linear_norbs)

        ! for qm-mm gradient components
        if(mm_main_r%numatm.gt.0) call qmmm_gradient(qm_cpmd_main_r%PA_scf,qm_cpmd_main_r%PB_scf, &
                                                     natom,xim,yim,zim,                 &
                                                     qm_scf_main_r%dim_linear_norbs,    &
                                                     qm_main_r%numat,mm_main_r%numatm,  &
                                                     qm_main_r%NAT,qm_main_r%NFIRST,qm_main_r%NLAST, &
                                                     qm_main_r%num_orbs,qm_scf_main_r%INDX, &
                                                     qm_main_r%qm_coord,mm_main_r%mm_coord, &
                                                     mm_main_r%mm_chrgs,                    &
                                                     qm_scf_main_r%CORE_mat,qm_scf_main_r%WW, &
                                                     qm_scf_main_r%RI,qm_scf_main_r%YY,     &
                                                     qm_main_r%qm_grads,mm_main_r%mm_grads, &
                                                     qm_control_r%q_am1_pm3,mstart,mend) 
     end if
  else
     if(q_do_update) then
        ! this is for the regular full scf/cpmd calculations.

        ! for qm-qm gradient components
        call qmqm_gradient(qm_scf_main_r%PA,qm_scf_main_r%PA,qm_scf_main_r%dim_linear_norbs)

        ! for qm-mm gradient components
        if(mm_main_r%numatm.gt.0) call qmmm_gradient(qm_scf_main_r%PA,qm_scf_main_r%PA, &
                                                     natom,xim,yim,zim,                 &
                                                     qm_scf_main_r%dim_linear_norbs,    &
                                                     qm_main_r%numat,mm_main_r%numatm,  &
                                                     qm_main_r%NAT,qm_main_r%NFIRST,qm_main_r%NLAST, &
                                                     qm_main_r%num_orbs,qm_scf_main_r%INDX, &
                                                     qm_main_r%qm_coord,mm_main_r%mm_coord, &
                                                     mm_main_r%mm_chrgs,                    &
                                                     qm_scf_main_r%CORE_mat,qm_scf_main_r%WW, &
                                                     qm_scf_main_r%RI,qm_scf_main_r%YY,     &
                                                     qm_main_r%qm_grads,mm_main_r%mm_grads, &
                                                     qm_control_r%q_am1_pm3,mstart,mend)
     else
        ! this is for the case, where the energy is called during MD, but without its
        ! coordinates being updated. Therefore, the energy call is basically duplicated
        ! in which the second energy call requires it being only need for energy without
        ! density being updated.

        ! for qm-qm gradient components
        call qmqm_gradient(qm_cpmd_main_r%PA_scf,qm_cpmd_main_r%PA_scf,qm_scf_main_r%dim_linear_norbs)

        ! for qm-mm gradient components
        if(mm_main_r%numatm.gt.0) call qmmm_gradient(qm_cpmd_main_r%PA_scf,qm_cpmd_main_r%PA_scf, &
                                                     natom,xim,yim,zim,                 &
                                                     qm_scf_main_r%dim_linear_norbs,    &
                                                     qm_main_r%numat,mm_main_r%numatm,  &
                                                     qm_main_r%NAT,qm_main_r%NFIRST,qm_main_r%NLAST, &
                                                     qm_main_r%num_orbs,qm_scf_main_r%INDX, &
                                                     qm_main_r%qm_coord,mm_main_r%mm_coord, &
                                                     mm_main_r%mm_chrgs,                    &
                                                     qm_scf_main_r%CORE_mat,qm_scf_main_r%WW, &
                                                     qm_scf_main_r%RI,qm_scf_main_r%YY,     &
                                                     qm_main_r%qm_grads,mm_main_r%mm_grads, &
                                                     qm_control_r%q_am1_pm3,mstart,mend)
     end if
  end if
  ! done the gradient part from the qm-qm and qm-mm interactions.

  ! special for cpmd: we have to update PA for next md step.
  if(q_do_cpmd) then  ! (qm_control_r%cpmd .and. qm_control_r%md_run)
     q_kinetic_print = qm_cpmd_main_r%q_include_kinetic_E
     if(qm_gho_info_r%q_gho) then
        if(q_do_update) then
           if(q_do_cpmd_first) then
              ! updating PA_old from PA_old(t-Dt) -> PA_old(t-dt), approximately.
              call update_PA_old(qm_gho_info_r%PAHB,qm_cpmd_main_r%PA_new,qm_cpmd_main_r%PA_old, &
                                 qm_cpmd_main_r%dPA,qm_cpmd_main_r%vPA,qm_cpmd_main_r%Mass_PA,   &
                                 qm_cpmd_main_r%Delta,                                           &
                                 qm_gho_info_r%lin_norbhb,qm_gho_info_r%norbhb)
              if(qm_cpmd_main_r%q_cnst) then
                 ! for this, PA_old (P_i-1), PAHB (P_i) to back determine P_i-1.
                 call density_const(qm_cpmd_main_r%PA_old,qm_gho_info_r%PAHB,    &
                                    qm_gho_info_r%norbhb,qm_gho_info_r%lin_norbhb)
              end if
           end if
           call update_PA(qm_gho_info_r%PAHB,qm_cpmd_main_r%PA_new,qm_cpmd_main_r%PA_old, &
                          qm_cpmd_main_r%dPA,qm_cpmd_main_r%vPA,qm_cpmd_main_r%Mass_PA,   &
                          qm_cpmd_main_r%NOSE_Mass,qm_cpmd_main_r%NOSEV,                  &
                          qm_cpmd_main_r%NOSE,qm_cpmd_main_r%NOSE_Old,                    &
                          qm_cpmd_main_r%Delta,qm_cpmd_main_r%Temperature,                &
                          qm_gho_info_r%lin_norbhb,qm_gho_info_r%norbhb,q_kinetic_print)
           ! apply constraints
           if(qm_cpmd_main_r%q_cnst) then
              call density_const(qm_gho_info_r%PAHB,qm_cpmd_main_r%PA_old,    &
                                 qm_gho_info_r%norbhb,qm_gho_info_r%lin_norbhb)
           end if
        end if

        ! for gho expansion PAHB -> PA.
        call GHO_expansion_cpmd(qm_gho_info_r%norbhb,qm_gho_info_r%naos,                 &
                                qm_gho_info_r%lin_naos,qm_gho_info_r%nqmlnk,             &
                                qm_gho_info_r%lin_norbhb,qm_scf_main_r%dim_norbs,        &
                                qm_scf_main_r%dim_linear_norbs,qm_gho_info_r%mqm16,      &
                                qm_scf_main_r%PA,qm_scf_main_r%PA,                       &
                                qm_gho_info_r%PAHB,qm_gho_info_r%PAHB,                   &
                                qm_gho_info_r%PAOLD,qm_gho_info_r%PAOLD,                 &
                                qm_gho_info_r%QMATMQ,qm_gho_info_r%BT,qm_gho_info_r%BTM, &
                                qm_scf_main_r%indx,qm_main_r%uhf)
        qm_gho_info_r%PAHB(1:qm_gho_info_r%lin_norbhb)=qm_gho_info_r%PAOLD(1:qm_gho_info_r%lin_norbhb)

        ! for gho expansion PA_old -> PA_scf 
        call GHO_expansion_cpmd(qm_gho_info_r%norbhb,qm_gho_info_r%naos,                 &
                                qm_gho_info_r%lin_naos,qm_gho_info_r%nqmlnk,             &
                                qm_gho_info_r%lin_norbhb,qm_scf_main_r%dim_norbs,        &
                                qm_scf_main_r%dim_linear_norbs,qm_gho_info_r%mqm16,      &
                                qm_cpmd_main_r%PA_scf,qm_cpmd_main_r%PA_scf,             &
                                qm_cpmd_main_r%PA_old,qm_cpmd_main_r%PA_old,             &
                                qm_gho_info_r%PAOLD,qm_gho_info_r%PAOLD,                 &
                                qm_gho_info_r%QMATMQ,qm_gho_info_r%BT,qm_gho_info_r%BTM, &
                                qm_scf_main_r%indx,qm_main_r%uhf)
        qm_cpmd_main_r%PA_old(1:qm_gho_info_r%lin_norbhb)=qm_gho_info_r%PAOLD(1:qm_gho_info_r%lin_norbhb)
     else
        if(q_do_update) then
           if(q_do_cpmd_first) then
              ! updating PA_old from PA_old(t-Dt) -> PA_old(t-dt), approximately.
              call update_PA_old(qm_scf_main_r%PA,qm_cpmd_main_r%PA_new,qm_cpmd_main_r%PA_old,   &
                                 qm_cpmd_main_r%dPA,qm_cpmd_main_r%vPA,qm_cpmd_main_r%Mass_PA,   &
                                 qm_cpmd_main_r%Delta,                                           &
                                 qm_scf_main_r%dim_linear_norbs,qm_scf_main_r%dim_norbs)
              if(qm_cpmd_main_r%q_cnst) then
                 ! for this, PA_old (P_i-1), PA (P_i) to back determine P_i-1.
                 call density_const(qm_cpmd_main_r%PA_old,qm_scf_main_r%PA,           &
                                    qm_scf_main_r%dim_norbs,qm_scf_main_r%dim_linear_norbs)
              end if
           end if
           call update_PA(qm_scf_main_r%PA,qm_cpmd_main_r%PA_new,qm_cpmd_main_r%PA_old,   &
                          qm_cpmd_main_r%dPA,qm_cpmd_main_r%vPA,qm_cpmd_main_r%Mass_PA,   &
                          qm_cpmd_main_r%NOSE_Mass,qm_cpmd_main_r%NOSEV,                  &
                          qm_cpmd_main_r%NOSE,qm_cpmd_main_r%NOSE_Old,                    &
                          qm_cpmd_main_r%Delta,qm_cpmd_main_r%Temperature,                &
                          qm_scf_main_r%dim_linear_norbs,qm_scf_main_r%dim_norbs,q_kinetic_print)

           ! apply constraints
           if(qm_cpmd_main_r%q_cnst) then
              call density_const(qm_scf_main_r%PA,qm_cpmd_main_r%PA_old,                  &
                                 qm_scf_main_r%dim_norbs,qm_scf_main_r%dim_linear_norbs)
           end if
        end if
     end if
  end if

  ! now, copying gradient into the charmm main dx/dy/dz arrays.

!!  ! this may not be needed as other options not used.
!!  if(mm_main_r%LQMEWD .and. mm_main_r%NQMEWD.ne.1) then
!!     !!! it's done: call gcomb(mm_main_r%qm_charges,qm_main_r%numat) 
!!     do i=1,qm_main_r%numat
!!        n=qm_control_r%qminb(i)
!!        qm_control_r%cgqmmm(n) = mm_main_r%qm_charges(i)
!!     end do
!!  end if

  ! it should apply even it is parallel
  call get_qmmm_gradient(natom,qm_main_r%numat,mm_main_r%numatm, &
                         qm_control_r%qminb,mm_main_r%qm_mm_pair_list, &
                         dx,dy,dz,qm_main_r%qm_grads,mm_main_r%mm_grads)
  !
  !do i=1,qm_main_r%numat
  !   n=qm_control_r%qminb(i)
  !   dx(n)=dx(n)+qm_main_r%qm_grads(1,i)
  !   dy(n)=dy(n)+qm_main_r%qm_grads(2,i)
  !   dz(n)=dz(n)+qm_main_r%qm_grads(3,i)
  !end do
  !! for MM atoms.
  !if(mm_main_r%numatm.gt.0) then
  !   do i=1,mm_main_r%NUMATM
  !      n=mm_main_r%qm_mm_pair_list(i)
  !      dx(n)=dx(n)+mm_main_r%mm_grads(1,i)
  !      dy(n)=dy(n)+mm_main_r%mm_grads(2,i)
  !      dz(n)=dz(n)+mm_main_r%mm_grads(3,i)
  !   end do
  !end if

  ! now for switching function contributions.
  if(mm_main_r%q_switch) call put_switching_gradient(natom,mm_main_r%NUMATM,dx,dy,dz)

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
  use qm1_info, only: qm_control_r,qm_main_r,mm_main_r,qm_scf_main_r
  use mndgho_module, only : qm_gho_info_r
  use qmmmewald_module,only: qm_ewald_real_space_gradient,qm_ewald_real_space_gradient_exl, &
                            qm_ewald_recip_space_gradient,   &
                            qmmm_ewald_r
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
  integer    :: i,m,nexcl,itotal,itotkq
  integer    :: mstart,mstop
#if KEY_PARALLEL==1
  integer    :: ISTRT_CHECK               !external function
#endif
  logical :: QNoPMEwald

  QNoPMEwald = .not.mm_main_r%PMEwald

  mstart = 1
  mstop  = mm_main_r%NUMATM
#if KEY_PARALLEL==1
  if(numnod>1) mstart = ISTRT_CHECK(mstop,mm_main_r%NUMATM)
#endif
  ! for ktable memory size
  if(QNoPMEwald) then
     itotal=qmmm_ewald_r%iatotl
     itotkq=qmmm_ewald_r%totkq
  else
     itotal=1
     itotkq=1
  end if

  ! initialization
  ! as qm_grads and mm_grads are already copied into the main dx/dy/dz arrays in
  ! scf_gradient routine, here we can use the array for the present purpose.
  qm_main_r%qm_grads2      = zero
  mm_main_r%mm_grads2      = zero
  !!qmmm_ewald_r%dexl_xyz = zero
  if(QNoPMEwald) qmmm_ewald_r%d_ewald_mm= zero

  ! for switching function
  if(mm_main_r%q_switch) mm_main_r%dxyz_sw2(1:6,1:mm_main_r%isize_swt_array) =zero

  ! Compute gradient contribution from the qm/mm-ewald correction.

  ! 1) Real space contribution:  since the real space contribution can be added
  !    into the dz,dy,dz array directly, the gradients are added in "qm_grads" and
  !    "mm_grads" and copied into the amin array below. however, the contribution
  !    from the qm-mm interactions exclued list is added into the "d_ewald_mm."
  call qm_ewald_real_space_gradient(qmmm_ewald_r%natom,qm_main_r%numat,mm_main_r%numatm, &
                                    mm_main_r%mm_coord,mm_main_r%mm_chrgs,               &
                                    qm_main_r%qm_coord,mm_main_r%qm_charges,             &
                                    qm_main_r%qm_grads2,mm_main_r%mm_grads2)

  if(qmmm_ewald_r%nexl_atm > 0) then
     qmmm_ewald_r%dexl_xyz = zero
     call qm_ewald_real_space_gradient_exl(qmmm_ewald_r%natom,qm_main_r%numat,           &
                                           qm_main_r%qm_coord,mm_main_r%qm_charges,      &
                                           qm_main_r%qm_grads2) 
  end if

  ! 2) Reciprocal space contribution:  due to the need of computing virial contribution 
  !    from the reciprocal space, the gradients are saved into "d_ewald_mm" and added
  !    into the main dx,dy,dz array later.
  if(.not. qm_control_r%q_do_cpmd_pme) then
     call qm_ewald_recip_space_gradient(qmmm_ewald_r%natom,qm_main_r%numat,qm_control_r%qminb, &
                                        qmmm_ewald_r%iastrt,qmmm_ewald_r%iafinl,               &
                                        qmmm_ewald_r%iatotl,itotal,itotkq,                     &
                                        qmmm_ewald_r%totkq,qmmm_ewald_r%ksqmaxq,               &
                                        qmmm_ewald_r%kmaxqx,qmmm_ewald_r%kmaxqy,               &
                                        qmmm_ewald_r%kmaxqz,                                   &
                                        cg,mm_main_r%qm_charges,qmmm_ewald_r%recip,            &
                                        virial,QNoPMEwald)
  end if

  ! 2-1) Reciprocal spacecontribution from the PME version.
  if(.not.QNoPMEwald) then
     call qm_pme_mm_grad(qmmm_ewald_r%natom,qm_main_r%numat,qm_control_r%qminb, &
                         x,y,z,qmmm_ewald_r%d_ewald_mm,cg,mm_main_r%qm_charges, &
                         qmmm_ewald_r%recip,qmmm_ewald_r%volume,                &
                         qmmm_ewald_r%kappa,virial)
  end if


!  ! for d_ewald_mm/qm_grads/mm_grads/dexl_xyz.
!  ! combine gradients and virial.
!#if KEY_PARALLEL==1
!  ! combine gradients and virial. 
!  ! do I need to do this? have to check!
!  if(numnod > 1) then
!     call GCOMB(qm_main_r%qm_grads     ,3*qm_main_r%numat)
!     call GCOMB(mm_main_r%mm_grads     ,3*mm_main_r%numatm)
!     ! avoid communicate, but instead when do sum, sum over all atoms.
!     !!call GCOMB(qmmm_ewald_r%d_ewald_mm,3*qmmm_ewald_r%natom)
!     !!call GCOMB(virial,9)
!     !!if(mynod.ne.0) virial=zero
!  end if
!#endif

  ! Copying gradients into the CHARMM gradient dx/dy/dz arrays.
  ! 2) mm atoms within the cutoff
  !do i=mstart,mstop                    ! 1,mm_main_r%numatm
  !   m=mm_main_r%qm_mm_pair_list(i)
  !   dx(m)=dx(m)+mm_main_r%mm_grads(1,i)
  !   dy(m)=dy(m)+mm_main_r%mm_grads(2,i)
  !   dz(m)=dz(m)+mm_main_r%mm_grads(3,i)
  !end do
  call grad_copy_2_main(mstart,mstop,dx,dy,dz,mm_main_r%mm_grads2,mm_main_r%qm_mm_pair_list)

  ! 1) qm atoms: since qm_grads has not been broadcasted, so add over all qm atoms.
  !do i=1,qm_main_r%numat
  !   m=qm_control_r%qminb(i)
  !   dx(m)=dx(m)+qm_main_r%qm_grads(1,i)
  !   dy(m)=dy(m)+qm_main_r%qm_grads(2,i)
  !   dz(m)=dz(m)+qm_main_r%qm_grads(3,i)
  !end do
  call grad_copy_2_main(1,qm_main_r%numat,dx,dy,dz,qm_main_r%qm_grads2,qm_control_r%qminb)

#if KEY_PARALLEL==1
  if(mynod == 0) then
#endif
  ! 3) excluded MM atoms from the qm-mm non-bonded interactions. (igmsel(i)=5 case)
     if(qmmm_ewald_r%nexl_atm > 0) then
        !do i=1,qmmm_ewald_r%nexl_atm
        !   m=qmmm_ewald_r%nexl_index(i)
        !   dx(m)=dx(m)+qmmm_ewald_r%dexl_xyz(1,i)
        !   dy(m)=dy(m)+qmmm_ewald_r%dexl_xyz(2,i)
        !   dz(m)=dz(m)+qmmm_ewald_r%dexl_xyz(3,i)
        !end do
        call grad_copy_2_main(1,qmmm_ewald_r%nexl_atm,dx,dy,dz,qmmm_ewald_r%dexl_xyz, &
                              qmmm_ewald_r%nexl_index)
     end if
#if KEY_PARALLEL==1
  end if
#endif

  ! 4) reciprocal space contribution is in d_ewald_mm. (refer the GETGRDQ routine).

  ! now for switching function contributions.
  if(mm_main_r%q_switch) call put_switching_gradient2(qmmm_ewald_r%natom,mm_main_r%NUMATM,dx,dy,dz)

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
  use qm1_info, only : qm_main_r,mm_main_r
  use mndgho_module, only : qm_gho_info_r
  use qm1_parameters, only : CORE,EHEAT,EISOL,LORBS
  use qm1_constant, only   : EVCAL

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
  qm_main_r%nelmd  = 0
  qm_main_r%nfock  = 5
  nel    = -qm_main_r%i_qmcharge
  ib     = 0
  do i=1,qm_main_r%numat
     ni     = qm_main_r%nat(i)
     if(LORBS(ni).ge.9) qm_main_r%nelmd = qm_main_r%nelmd+1
     !
     qm_main_r%num_orbs(i)= LORBS(ni) ! number of orbitals of each atom.
     qm_main_r%nfirst(i) = ib+1
     nel    = nel+NINT(CORE(ni))
     ib     = ib +LORBS(ni) 
     qm_main_r%nlast(i) = ib
  end do
  if(qm_main_r%nelmd.gt.0) qm_main_r%nfock = 10

  ! if GHO, remove three auxiliary electrons from the QM-link atom when
  !         count the number of active electrons
  if(qm_gho_info_r%q_gho) nel = nel - 3*qm_gho_info_r%nqmlnk
  !
  qm_main_r%nel = nel

  ! 2)
  ! determine occupation number of mol. orbitals.
  !     nel      number of electrons       (NEL = NALPHA+NBETA).
  !     nalpha   number of alpha electrons 
  !     nbeta    number of beta  electrons 
  !     NBETA    equal to number of doubly occupied molecular orbitals
  !              in rhf calculations with imult.ne.1 (not singlet?)
  !     NUMB     number of highest occupied molecular orbital.
  nodd   = MAX(1,qm_main_r%imult)-1   ! 0, if imult=0 singlet (RHF).

  qm_main_r%norbs  = qm_main_r%nlast(qm_main_r%numat)
  qm_main_r%nbeta  =(qm_main_r%nel-nodd)/2
  qm_main_r%nalpha = qm_main_r%nbeta+nodd
  qm_main_r%numb   = qm_main_r%nalpha
  qm_main_r%nclo   = qm_main_r%nbeta

  qm_main_r%iodd   = 0  ! this is for the RHF singlet state
  qm_main_r%jodd   = 0  ! 
  ! for the RHF and not singlet state.
  if(qm_main_r%imult.gt.0 .and. .not.qm_main_r%uhf) then
     ! this is not used, it is for an excited singlet state with two singly
     ! occupied orbitals. 
     !if(qm_main_r%imult.eq.1) then
     ! this is for the singlet state of the open shell system (UHF) with two singly occupied orbitals.
     ! This scf solution usually corresponds to an excited single state.
     !   qm_main_r%numb = qm_main_r%nbeta+1
     !   qm_main_r%nclo = qm_main_r%nbeta-1
     !   qm_main_r%iodd = qm_main_r%nbeta
     !   qm_main_r%jodd = qm_main_r%numb
     !else if(qm_main_r%imult.eq.2) then
     if(qm_main_r%imult.eq.2) then           ! doublet
        qm_main_r%iodd = qm_main_r%numb
     else if(qm_main_r%imult.eq.3) then      ! triplet
        qm_main_r%iodd = qm_main_r%nbeta+1 
        qm_main_r%jodd = qm_main_r%nbeta+2
     end if
  end if
  qm_main_r%nmos   = qm_main_r%numb
  ! explicit definition of occupation numbers.
  qm_main_r%imocc  = 1  ! = ABS(IUHF=-1)
  qm_main_r%nocca  = 0  ! 0, if imocc .lt. 2
  qm_main_r%noccb  = 0  ! 0,

  ! if GHO
  if(qm_gho_info_r%q_gho) qm_gho_info_r%norbsgho = qm_main_r%NORBS

  ! 4)
  ! compute sum of atomic energies and heats of formation.
  qm_main_r%ener_atomic = zero
  qm_main_r%HofF_atomic = zero
  do i=1,qm_main_r%numat
     ni     = qm_main_r%nat(i)
     qm_main_r%HofF_atomic=qm_main_r%HofF_atomic+EHEAT(ni)
     qm_main_r%ener_atomic=qm_main_r%ener_atomic+EISOL(ni)
  end do
  !
  ! convert EISOL into kcal/mol unit hear.
  qm_main_r%ener_atomic=EVCAL*qm_main_r%ener_atomic

  return
  end subroutine qm_info_setup


  !=====================================================================
  subroutine compute_one_center_h
  !=====================================================================
  ! 
  ! precompute diagonal one-center terms.
  !
  use qm1_info, only : qm_main_r,qm_scf_main_r
  use qm1_parameters,only : USS,UPP,UDD
  use number        ,only : zero
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none
  integer :: i,ni,ia,iorbs,j,icnt
  integer :: istart,iend
!#if KEY_PARALLEL==1
!  integer :: ISTRT_CHECK        ! external function
!#endif
  
  istart = 1
  iend   = qm_main_r%numat
!#if KEY_PARALLEL==1
!  if(numnod > 1) istart = ISTRT_CHECK(iend,qm_main_r%numat)
!#endif

  ! initialize.
  qm_scf_main_r%H_1cent(1:qm_scf_main_r%dim_norbs)=zero
  qm_scf_main_r%imap_h(1:qm_scf_main_r%dim_norbs) =0

  ! Diagonal one-center terms.
  ! this is done once at the beginning of QM setup.
  icnt=0
  do i=istart,iend           ! 1,qm_main_r%numat
     ni     = qm_main_r%NAT(i)
     ia     = qm_main_r%NFIRST(i)
     iorbs  = qm_main_r%num_orbs(i)  ! = NLAST(I)-IA+1
     icnt   = icnt+1
     qm_scf_main_r%imap_h(icnt) =qm_scf_main_r%INDX(ia)+ia
     qm_scf_main_r%H_1cent(icnt)=USS(ni)
     if(iorbs.ge.9) then
        do j=ia+1,ia+3
           icnt   = icnt+1
           qm_scf_main_r%imap_h(icnt) =qm_scf_main_r%INDX(j)+j
           qm_scf_main_r%H_1cent(icnt)=UPP(ni)
        end do
        do j=ia+4,ia+8
           icnt   = icnt+1
           qm_scf_main_r%imap_h(icnt) =qm_scf_main_r%INDX(j)+j
           qm_scf_main_r%H_1cent(icnt)=UDD(ni)
        end do
     else if(iorbs.ge.4) then
        do j=ia+1,ia+3
           icnt   = icnt+1
           qm_scf_main_r%imap_h(icnt) =qm_scf_main_r%INDX(j)+j
           qm_scf_main_r%H_1cent(icnt)=UPP(ni)
        end do
     end if
  end do
!#if KEY_PARALLEL==1
!  ! it is need to be broacasted.
!  if(numnod>1) then
!     call gcomb(qm_scf_main_r%H_1cent,qm_scf_main_r%dim_norbs)
!     call igcomb(qm_scf_main_r%imap_h,qm_scf_main_r%dim_norbs)
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
  use qm1_info,only : qm_main_r,mm_main_r
  use nbndqm_mod, only: map_mmgrp_to_group

  implicit none
  integer :: natom,numat
  real(chm_real):: dx(natom),dy(natom),dz(natom)

  integer :: i,j,k,is,ip,js,jp,irs,jrs,icnt,jqmcnt
  real(chm_real):: dxyz_qm(3,numat)

  if(.not.mm_main_r%q_switch) return

  do i=1,qm_main_r%nqmgrp(1)
     irs = qm_main_r%nqmgrp(i+1)  ! qm group
     is =  igpbs(irs) + 1
     ip =  igpbs(irs+1)
     jqmcnt = 0
     dxyz_qm(1:3,1:numat) = zero
     do j=1,mm_main_r%inum_mm_grp
        jrs = map_mmgrp_to_group(j)
        js  = igpbs(jrs) + 1
        jp  = igpbs(jrs+1)
        if(mm_main_r%q_mmgrp_qmgrp_swt(j,i) > 0) then
           ! do this pair
           icnt = mm_main_r%q_mmgrp_qmgrp_swt(j,i)
           jqmcnt = jqmcnt + 1

           ! for qm atoms.
           do k=1,ip-is+1
              dxyz_qm(1:3,k) = dxyz_qm(1:3,k) + mm_main_r%dxyz_sw(1:3,icnt)
           end do

           ! for mm atoms.
           do k=js,jp
              dx(k) = dx(k) + mm_main_r%dxyz_sw(4,icnt) 
              dy(k) = dy(k) + mm_main_r%dxyz_sw(5,icnt) 
              dz(k) = dz(k) + mm_main_r%dxyz_sw(6,icnt) 
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
  use qm1_info,only : qm_main_r,mm_main_r
  use nbndqm_mod, only: map_mmgrp_to_group

  implicit none
  integer :: natom,numat
  real(chm_real):: dx(natom),dy(natom),dz(natom)

  integer :: i,j,k,is,ip,js,jp,irs,jrs,icnt,jqmcnt
  real(chm_real):: dxyz_qm(3,numat)

  if(.not.mm_main_r%q_switch) return

  do i=1,qm_main_r%nqmgrp(1)
     irs = qm_main_r%nqmgrp(i+1)  ! qm group
     is =  igpbs(irs) + 1
     ip =  igpbs(irs+1)
     jqmcnt = 0
     dxyz_qm(1:3,1:numat) = zero
     do j=1,mm_main_r%inum_mm_grp
        jrs = map_mmgrp_to_group(j)
        js  = igpbs(jrs) + 1
        jp  = igpbs(jrs+1)
        if(mm_main_r%q_mmgrp_qmgrp_swt(j,i) > 0) then
           ! do this pair
           icnt = mm_main_r%q_mmgrp_qmgrp_swt(j,i)
           jqmcnt = jqmcnt + 1

           ! for qm atoms.
           do k=1,ip-is+1
              dxyz_qm(1:3,k) = dxyz_qm(1:3,k) + mm_main_r%dxyz_sw2(1:3,icnt)
           end do

           ! for mm atoms.
           do k=js,jp
              dx(k) = dx(k) + mm_main_r%dxyz_sw2(4,icnt)
              dy(k) = dy(k) + mm_main_r%dxyz_sw2(5,icnt)
              dz(k) = dz(k) + mm_main_r%dxyz_sw2(6,icnt)
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
#endif /* (mndo97)*/
end module qmmm_interface
! end
