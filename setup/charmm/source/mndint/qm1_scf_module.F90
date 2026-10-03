module qm1_scf_module
  use chm_kinds
  use number
  use qm1_constant

  ! used is pair indexing. 
  ! see below define_pair_indix, which is called from qmmm_load_parameters_setup_qm_info.
  ! and memory/arrays are defined in qm1_info.F90
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

  ! for qm/mm-ewald related.
  real(chm_real),allocatable,save :: empot_local(:),empot_all(:)

  contains
  !
#if KEY_MNDO97==1
  subroutine scf_iter(E_scf,H,W,Q,                   &
                      CA,DA,EA,FA,PA,CB,DB,EB,FB,PB, &
                      numat,                         &
                      dim_norbs,dim_linear_norbs,    &
                      dim_linear_fock,dim_linear_fock2, &
                      dim_scratch,ifockmd_counter,dim_iwork,         &
                      iwork,icall,UHF)
  !
  ! Scf iteration
  ! E_scf  : scf electonic energy
  ! H      : core hamiltonian matrix
  ! W      : two-electron integrals
  ! Q      : scratch array
  ! CA     : rhf or uhf-alpha MO eigenvectors. 
  ! DA     : rhf or uhf-alpha difference density matrix.
  ! EA     : rhf or uhf-alpha MO eigenvalues.
  ! FA     : rhf or uhf-alpha Fock matrix.
  ! PA     : rhf or uhf-alpha density matrix.
  ! CB     : uhf-beta MO eigenvectors.
  ! DB     : uhf-beta difference density matrix.
  ! EB     : uhf-beta MO eigenvalues.
  ! FB     : uhf-beta Fock matrix.
  ! PB     : uhf-beta density matrix.
  ! iwork  ! integer scratch array.
  ! icall  : error flag.
  ! uhf    : UHF flag
  !
  ! other matrices from Diis (qm_gho_info_c):
  ! FDA    : rhf or uhf-alpha Fock matrices from diis iterations.
  ! FDB    : uhf-beta Fock matrices from diis iterations.
  ! Ediis  : error matrices from different diis iterations.
  ! Bdiis  : coefficient matrix for diis linear equations.
  ! Adiis  : coefficient matrix for diis linear equations.
  ! Xdiis  : RHS vector and solutions of diis linear equations.
  !
  use qm1_info, only : qm_control_c,qm_main_c,mm_main_c,qm_scf_main_c,qm_param_c, &
                       qm_scf_indx_c,qm_scf_diis_c,qm_fockmd_diis_c, &
                       qm_gho_info_c
  use qmmmewald_module,only : qm_ewald_prepare_fock  ! qm_ewald_add_fock,qm_ewald_correct_ee
  use mndgho_module,only : GHO_expansion,FTOFHB,CTRASF
  use qm1_diagonalization
#if KEY_PARALLEL==1
  use parallel
!#if KEY_MPI==1  /*MPI run*/
!  use mpi_f08
!#endif /*MPI run*/
#endif
  use stream, only : prnlev

  implicit none
  !
  integer :: numat,dim_norbs,dim_linear_norbs,dim_linear_fock,dim_linear_fock2,dim_scratch,dim_iwork
  integer :: iwork(dim_iwork),icall,ifockmd_counter
  real(chm_real):: E_scf
  real(chm_real):: H(dim_linear_norbs),W(dim_linear_fock2),Q(dim_scratch), &
                   CA(dim_norbs,dim_norbs),CB(dim_norbs,dim_norbs),        &
                   DA(dim_linear_norbs),DB(dim_linear_norbs),              &
                   EA(dim_norbs),EB(dim_norbs),                            &
                   FA(dim_linear_norbs),FB(dim_linear_norbs),              &
                   PA(dim_linear_norbs),PB(dim_linear_norbs)
  logical :: UHF

  ! local variables
  real(chm_real),parameter :: TRANS=0.10D0,     &
                              r_three=one/three 
  character(LEN=4) :: MEXT,MFAST
  character(LEN=4) :: MXS(qm_scf_main_c%KITSCF),MFS(qm_scf_main_c%KITSCF)
  real(chm_real)   :: EDS(qm_scf_main_c%KITSCF),EES(qm_scf_main_c%KITSCF), &
                      ERS(qm_scf_main_c%KITSCF),PLS(qm_scf_main_c%KITSCF)
  !
  integer :: i,j,ii,jj,KEXT,NDIIS,NITER,nstart,nstep,info
  real(chm_real):: EEP,PL,PM,EF,EH,EFA,EFB,EHA,EHB,E_error,EDmax,EFA_L,EFB_L
  logical :: DODIIS,FASTDG
  logical :: scf_succeed,fockmd_on
  integer :: nnumnod,mmynod
#if KEY_PARALLEL==1
  integer :: JPARPT_local(0:numnod),KPARPT_local(0:numnod),JPARPT_fock(0:numnod), &
             JPARPT_diis(0:numnod)
#endif
  integer :: mstart,mstop,msize,fstart,fstop,kstart,kstop,kmstart,kmstop,kkmax, &
             mkstart,mkstop,mstart_diis,mstop_diis,mstart_a_dens,mstop_a_dens,  &
             mstart_b_dens,mstop_b_dens
  real(chm_real):: t1,t2

  !!! for qm/mm-ewald related.
  !!real(chm_real),allocatable :: empot_local(:),empot_all(:)

  ! for parallelization, define variables for arrays.
#if KEY_PARALLEL==1
  mmynod  = mynod
  nnumnod = numnod
#else
  nnumnod= 1
  mmynod = 0
#endif
  call setup_array_index(qm_gho_info_c%q_gho) 

  ! specific for qm/mm-Ewald part. allocate local memory
  if(mm_main_c%LQMEWD) then
     allocate(empot_all(numat))
     allocate(empot_local(numat))
  end if

  ! arrays and memories for diagonalization routines
  if(qm_gho_info_c%q_gho) then
     if(UHF) then                  !norbs(==n),nvect,nv,nocc,nocc(beta)
        call setup_diag_array_info(qm_gho_info_c%norbhb,qm_gho_info_c%norbhb,qm_gho_info_c%norbhb, &
                                   qm_main_c%numb,qm_main_c%nbeta,uhf)
     else
        call setup_diag_array_info(qm_gho_info_c%norbhb,qm_gho_info_c%norbhb,qm_gho_info_c%norbhb, &
                                   qm_main_c%numb,qm_main_c%numb,uhf)
     end if
  else
     if(UHF) then
        call setup_diag_array_info(qm_main_c%norbs,qm_main_c%norbs,dim_norbs, &
                                   qm_main_c%numb,qm_main_c%nbeta,uhf)
     else
        call setup_diag_array_info(qm_main_c%norbs,qm_main_c%norbs,dim_norbs, &
                                   qm_main_c%numb,qm_main_c%numb,uhf)
     end if
  end if

  ! do some initialization
  niter = 0
  ndiis = 0
  kext  = 0
  info  = 0
  EEP   = zero
  PL    = zero
  FASTDG=.false. ! .true.  ! at the beginning, but will be determined again below.  
  if(qm_control_c%q_diis) then
     nstart = -1  ! if diis is on, do not use damping/extrapolatin 
  else            ! in cal_density_matrix
     nstart = 4
  end if
  nstep  = 4      ! use extrapolation, it should be set negative if damping.

  ! overwrite here.
  qm_scf_main_c%SCFCRT=1.000000000000000d-006
  qm_scf_main_c%PLCRT =1.000000000000000d-006 

  ! for Fock matrix dynamics (ref: ).
  fockmd_on=.false.
  if(qm_control_c%q_fockmd .and. qm_control_c%q_do_fockmd_scf) then
     call fock_diis(qm_fockmd_diis_c%FA_sv,qm_fockmd_diis_c%FDA,     &
                    dim_linear_norbs,qm_fockmd_diis_c%mxfdiis,msize, &
                    qm_control_c%i_fockmd_option,ifockmd_counter,fockmd_on, &
                    mstart,mstop,mmynod,nnumnod)
     if(fockmd_on) then
        ! run a single diagonalization to determine the updated density.
        FA(mstart:mstop) = qm_fockmd_diis_c%FA_sv(1:msize)
#if KEY_PARALLEL==1
        if(nnumnod>1) call VDGBRE(FA,JPARPT_local)
#endif
        if (qm_gho_info_c%q_gho) then
           ! transform f into hb for QM link atom
           call FTOFHB(FA,qm_gho_info_c%FAHB,qm_gho_info_c%BT,    &
                       qm_gho_info_c%numat,qm_gho_info_c%nqmlnk,  &
                       dim_norbs,qm_gho_info_c%norbao,            &
                       qm_gho_info_c%lin_norbao,qm_gho_info_c%nactatm, &
                       qm_main_c%NFIRST,qm_main_c%NLAST,          &
                       qm_scf_main_c%indx)
           call square(qm_gho_info_c%FAHB,qm_gho_info_c%FAHBwrk, &
                       qm_gho_info_c%norbhb,qm_gho_info_c%norbhb, &
                       qm_gho_info_c%lin_norbhb,.false.)
           call evvrsp(qm_gho_info_c%norbhb,qm_gho_info_c%norbhb, &
                       qm_gho_info_c%norbhb,                      &
                       qm_gho_info_c%FAHBwrk,Q,IWORK,EA,qm_gho_info_c%CAHB,INFO,.false.)
           call cal_density_matrix(qm_gho_info_c%CAHB,qm_gho_info_c%DAHB,     &
                                   qm_gho_info_c%PAHB,qm_gho_info_c%FAHBwrk,PL,  &
                                   qm_gho_info_c%norbhb,                      &
                                   qm_gho_info_c%lin_norbhb,                  &
                                   qm_gho_info_c%norbhb,qm_main_c%numb,       &
                                   qm_main_c%iodd,qm_main_c%jodd,niter,kext,nstart,nstep, &
                                   mstart_a_dens,mstop_a_dens)

           ! do the GHO-expasion.
           call GHO_expansion(qm_gho_info_c%norbhb,qm_gho_info_c%naos,     &
                              qm_gho_info_c%lin_naos,qm_gho_info_c%nqmlnk, &
                              qm_gho_info_c%lin_norbhb,dim_norbs,          &
                              dim_linear_norbs,qm_gho_info_c%mqm16,        &
                              PL,PM,PA,PA,                                 &
                              qm_gho_info_c%PAHB,qm_gho_info_c%PAHB,       &
                              qm_gho_info_c%PAOLD,qm_gho_info_c%PAOLD,     &
                              qm_gho_info_c%QMATMQ,qm_gho_info_c%BT,qm_gho_info_c%BTM, &
                              qm_scf_main_c%indx,UHF)
        else
           call square(FA,qm_scf_main_c%FAwork,qm_main_c%norbs,dim_norbs,dim_linear_norbs,.false.)
           call evvrsp(qm_main_c%norbs,qm_main_c%norbs,dim_norbs,           &
                       qm_scf_main_c%FAwork,Q,IWORK,EA,CA,INFO,.false.)
           call cal_density_matrix(CA,DA,PA,qm_scf_main_c%FAwork,PL,     &
                                   dim_norbs,dim_linear_norbs,           &
                                   qm_main_c%norbs,qm_main_c%numb,       &
                                   qm_main_c%iodd,qm_main_c%jodd,niter,kext,nstart,nstep,&
                                   mstart_a_dens,mstop_a_dens)
        end if
        !! lower the scf criteria.
        !qm_scf_main_c%SCFCRT=1.000000000000000d-006
        !qm_scf_main_c%PLCRT =1.000000000000000d-006
     end if
  end if

  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !
  ! START OF SCF LOOP.
  !
  ! niter : current scf step
  ! kext  : 0 (at the beginning) see init_iter
  !        -1 (if scf converged) see below
  !         1 (do extrapolation) see cal_density_matrix
  !         2 (do damping)       see cal_density_matrix
  scf_succeed=.true.

  Scfloop: Do                     ! main loop
     ! construct F-matrix for RHF or Falpha-matrix for UHF
     !FA(1:dim_linear_norbs)  = H(1:dim_linear_norbs)
     !
     ! now do the copy fa=h
     FA(mstart:mstop) = H(mstart:mstop)  ! FA(1:dim_linear_norbs) = H(1:dim_linear_norbs)

     ! coulumb and exchange contribution.
     ! in subroutine fockx, LM6 = dim_linear_fock
     !                      LM4 = dim_linear_norbs
     call fockx(FA,PA,PB,Q,W,dim_linear_norbs,dim_linear_fock,UHF,         &
                numat,qm_main_c%nfirst,qm_main_c%nlast,qm_main_c%num_orbs, &
                qm_scf_main_c%NW,qm_scf_main_c%INDX,                       &
                mstart,mstop,fstart,fstop,                                 &
#if KEY_PARALLEL==1
                JPARPT_fock(0:numnod),                                     &
#endif
                qm_scf_indx_c%ip_local,qm_scf_indx_c%ip_check,             &
                [0],[0],[0],[0],[0],[0])
                !qm_scf_indx_c%ip1_local,qm_scf_indx_c%ip2_local,           & ! These are used in UHF. Thus, the memory
                !qm_scf_indx_c%jp1_local,qm_scf_indx_c%jp2_local,           & ! is defined with size 1.
                !qm_scf_indx_c%jp3_local,qm_scf_indx_c%jx_local)              !
     !exit Scfloop

     ! QM/MM-Ewald
     if(mm_main_c%LQMEWD) then
        ! Q = PA + PB (UHF) or 2*PA (constructed in fockx). 
        call calc_mulliken(numat,qm_main_c%num_orbs,                       &
                           qm_param_c%core,Q,mm_main_c%qm_charges,         &
                           mkstart,mkstop,                                 &
#if KEY_PARALLEL==1
                           KPARPT_local(0:numnod),                         & 
#endif
                           qm_scf_main_c%INDX)
        ! compute Ewald correction potential on qm atom site and mofidy Fock matrix. 
        ! Since Eslf(nquant,nquant) matrix is used, it is only need to do matrix
        ! multiplication to get the correction terms from QM images to be SCF iterated.
        call qm_ewald_prepare_fock(numat,empot_all,empot_local,            &
                                   mkstart,mkstop,                         &
#if KEY_PARALLEL==1
                                   KPARPT_local(0:numnod),                 &
#endif
                                   mm_main_c%qm_charges)

        ! now correct the fock matrix.
        call qm_ewald_add_fock(numat,qm_main_c%nfirst,qm_main_c%num_orbs,  &
                               qm_scf_main_c%indx,FA,mm_main_c%qm_charges, &
                               dim_linear_norbs) 
     end if
#if KEY_PARALLEL==1
     if(nnumnod>1) call VDGBRE(FA,JPARPT_local)
#endif
     !!if(mynod==0) write(6,'(A17,F12.5)')'Time (Fock     )=',(t2-t1)*1000.0d0

     ! construct the Fbeta-matrix for UHF.
     if(UHF) then
        !FB(1:dim_linear_norbs)  = H(1:dim_linear_norbs)
        ! now do the copy fb=h
        FB(mstart:mstop) = H(mstart:mstop)  ! FB(1:dim_linear_norbs) = H(1:dim_linear_norbs)

        call fockx(FB,PB,PA,Q,W,dim_linear_norbs,dim_linear_fock,UHF,         &
                   numat,qm_main_c%nfirst,qm_main_c%nlast,qm_main_c%num_orbs, &
                   qm_scf_main_c%NW,qm_scf_main_c%INDX,                       &
                   mstart,mstop,fstart,fstop,                                 &
#if KEY_PARALLEL==1
                   JPARPT_fock(0:numnod),                                     &
#endif
                   qm_scf_indx_c%ip_local,qm_scf_indx_c%ip_check,             &
                   qm_scf_indx_c%ip1_local,qm_scf_indx_c%ip2_local,           & ! These are used in UHF.
                   qm_scf_indx_c%jp1_local,qm_scf_indx_c%jp2_local,           & !
                   qm_scf_indx_c%jp3_local,qm_scf_indx_c%jx_local)              !

        ! QM/MM-Ewald
        if(mm_main_c%LQMEWD) then
           ! As the following done above.. no need to do here.
           !call calc_mulliken(numat,qm_main_c%num_orbs,                      &
           !                   qm_param_c%core,Q,mm_main_c%qm_charges,        &
           !                   mkstart,mkstop,                                &
           !#if KEY_PARALLEL==1
           !                   KPARPT_local(0:numnod),                        &
           !#endif
           !                   qm_scf_main_c%INDX)
           !
           ! As the following done above.. no need to do here.
           ! compute Ewald correction potential on qm atom site and mofidy Fock matrix. 
           ! Since Eslf(nquant,nquant) matrix is used, it is only need to do matrix
           ! multiplication to get the correction terms from QM images to be SCF iterated.
           !call qm_ewald_prepare_fock(numat,empot_all,empot_local,           &
           !                           mkstart,mkstop,                        &
           !#if KEY_PARALLEL==1
           !                           KPARPT_local(0:numnod),                &
           !#endif
           !                           mm_main_c%qm_charges)

           ! now correct the fock matrix.
           call qm_ewald_add_fock(numat,qm_main_c%nfirst,qm_main_c%num_orbs,  &
                                  qm_scf_main_c%indx,FB,mm_main_c%qm_charges, &
                                  dim_linear_norbs) 
        end if
#if KEY_PARALLEL==1
        if(nnumnod>1) call VDGBRE(FB,JPARPT_local)
#endif
     end if         ! (UHF)

     ! Energy calculation.
     ! Note that there were two escf call in the original code, one with F=H copy (core-Hamiltonian),
     ! and the other one with full F. Here, in this implementation, the two separate calls are
     ! merged to a single energy call.
     ! This is done here, becaue the energy calls are done with F and H (and not with FAHB).
     EFA = escf(dim_norbs,PA,FA,H,kmstart,kmstop,kstart,kstop,dim_linear_norbs)
     if(UHF) then
        EFB = escf(dim_norbs,PB,FB,H,kmstart,kmstop,kstart,kstop,dim_linear_norbs)
     else
        EFB = EFA    ! computed for complete Fock-matrix (here) and H-core matrix.
     end if

     ! In case of using QM/MM-Ewald, MM atom contributes in full.
     EFA_L = zero
     EFB_L = zero
     if(mm_main_c%LQMEWD) then
        ! empot_local is already copied above, qm_ewald_prepare_fock.
        EFA_L = qm_ewald_correct_ee(numat,qm_main_c%nfirst,qm_main_c%num_orbs, &
                                    mkstart,mkstop,qm_scf_main_c%indx,PA)
        if(UHF) then
           EFB_L = qm_ewald_correct_ee(numat,qm_main_c%nfirst,qm_main_c%num_orbs, &
                                       mkstart,mkstop,qm_scf_main_c%indx,PB)
        else
           EFB_L = EFA_L
        end if
     end if

     ! For GHO.
     if (qm_gho_info_c%q_gho) then
        ! only need to do before exit: converged or scf iteraction exceeded.
        if(kext == -1 .or. niter >= qm_scf_main_c%KITSCF) then
          ! store Fock matrix in AO basis for derivative
          qm_gho_info_c%FAOA(1:dim_linear_norbs)=FA(1:dim_linear_norbs)
        end if

        ! transform f into hb for QM link atom
        call FTOFHB(FA,qm_gho_info_c%FAHB,qm_gho_info_c%BT,    &
                    qm_gho_info_c%numat,qm_gho_info_c%nqmlnk,  &
                    dim_norbs,qm_gho_info_c%norbao,            &
                    qm_gho_info_c%lin_norbao,qm_gho_info_c%nactatm, &
                    qm_main_c%NFIRST,qm_main_c%NLAST,          &
                    qm_scf_main_c%indx)

        if(UHF) then
           ! only need to do before exit: converged or scf iteraction exceeded.
           ! either converged or scf interation exceeded.
           if(kext == -1 .or. niter >= qm_scf_main_c%KITSCF) then
              qm_gho_info_c%FAOB(1:dim_linear_norbs)=FB(1:dim_linear_norbs)
           end if

           ! for GHO, transform beta Fock matrix to hybrid basis
           call FTOFHB(FB,qm_gho_info_c%FBHB,qm_gho_info_c%BT,    &
                       qm_gho_info_c%numat,qm_gho_info_c%nqmlnk,  &
                       dim_norbs,qm_gho_info_c%norbao,            &
                       qm_gho_info_c%lin_norbao,qm_gho_info_c%nactatm, &
                       qm_main_c%nfirst,qm_main_c%nlast,          &
                       qm_scf_main_c%indx)
        end if
     end if

     ! apply Diis convergence acceleration.
     dodiis =(qm_control_c%q_diis .and. niter >= 1 .and. kext /= -1)
     if(dodiis) then ! if(dodiis .and. .not.(qm_control_c%q_dxl_bomd .and. qm_control_c%q_do_dxl_scf)) then
       if (qm_gho_info_c%q_gho) then
         ! GHO-DIIS extrapolation
         if(UHF) then
            call diis(qm_gho_info_c%FAHB,qm_gho_info_c%FBHB,             &
                      qm_gho_info_c%PAOLD,qm_gho_info_c%PBOLD,           &
                      qm_scf_main_c%FAwork,qm_scf_main_c%FBwork,         &
                      qm_scf_main_c%PAwork,qm_scf_main_c%PBwork,         &
                      qm_scf_diis_c%FDA,qm_scf_diis_c%FDB,               &
                      qm_scf_diis_c%Ediis,                               &
                      qm_scf_diis_c%Adiis,qm_scf_diis_c%Bdiis,           &
                      qm_scf_diis_c%Xdiis,                               &
                      qm_gho_info_c%norbhb,qm_gho_info_c%lin_norbhb,     &
                      qm_gho_info_c%norbhb,NDIIS,qm_scf_diis_c%mxdiis,   &
#if KEY_PARALLEL==1
                      JPARPT_diis(0:numnod),                             &
#endif
                      mstart_diis,mstop_diis,                            &
                      qm_scf_diis_c%iwork_diis,EDMAX,.true.,UHF)
         else
            call diis(qm_gho_info_c%FAHB,qm_gho_info_c%FAHB,             &
                      qm_gho_info_c%PAOLD,qm_gho_info_c%PAOLD,           &
                      qm_scf_main_c%FAwork,qm_scf_main_c%FAwork,         &
                      qm_scf_main_c%PAwork,qm_scf_main_c%PAwork,         &
                      qm_scf_diis_c%FDA,qm_scf_diis_c%FDA,               &
                      qm_scf_diis_c%Ediis,                               &
                      qm_scf_diis_c%Adiis,qm_scf_diis_c%Bdiis,           &
                      qm_scf_diis_c%Xdiis,                               &
                      qm_gho_info_c%norbhb,qm_gho_info_c%lin_norbhb,     &
                      qm_gho_info_c%norbhb,NDIIS,qm_scf_diis_c%mxdiis,   &
#if KEY_PARALLEL==1
                      JPARPT_diis(0:numnod),                             &
#endif
                      mstart_diis,mstop_diis,                            &
                      qm_scf_diis_c%iwork_diis,EDMAX,.true.,UHF)
         end if
       else
         ! normal diis extrapolation.
         if(UHF) then
            call diis(FA,FB,PA,PB,                                       &
                      qm_scf_main_c%FAwork,qm_scf_main_c%FBwork,         &
                      qm_scf_main_c%PAwork,qm_scf_main_c%PBwork,         &
                      qm_scf_diis_c%FDA,qm_scf_diis_c%FDB,               &
                      qm_scf_diis_c%Ediis,                               &
                      qm_scf_diis_c%Adiis,qm_scf_diis_c%Bdiis,           &
                      qm_scf_diis_c%Xdiis,                               &
                      dim_norbs,dim_linear_norbs,                        &
                      qm_main_c%NORBS,NDIIS,qm_scf_diis_c%mxdiis,        &
#if KEY_PARALLEL==1
                      JPARPT_diis(0:numnod),                             &
#endif
                      mstart_diis,mstop_diis,                            &
                      qm_scf_diis_c%iwork_diis,EDMAX,.true.,UHF)
         else
            call diis(FA,FB,PA,PB,                                       &
                      qm_scf_main_c%FAwork,qm_scf_main_c%FAwork,         &
                      qm_scf_main_c%PAwork,qm_scf_main_c%PAwork,         &   
                      qm_scf_diis_c%FDA,qm_scf_diis_c%FDA,               &
                      qm_scf_diis_c%Ediis,                               &
                      qm_scf_diis_c%Adiis,qm_scf_diis_c%Bdiis,           &
                      qm_scf_diis_c%Xdiis,                               &
                      dim_norbs,dim_linear_norbs,                        &
                      qm_main_c%NORBS,NDIIS,qm_scf_diis_c%mxdiis,        &
#if KEY_PARALLEL==1
                      JPARPT_diis(0:numnod),                             &
#endif
                      mstart_diis,mstop_diis,                            &
                      qm_scf_diis_c%iwork_diis,EDMAX,.true.,UHF)
         end if
       end if
     end if
     !!if(mynod==0) write(6,'(A17,F12.5)')'Time (DIIS     )=',(t2-t1)*1000.0d0


     ! Summing up energy terms. (See above the energy (ESCF) calls.)
     E_scf  = EFA+EFA_L+EFB+EFB_L
     ! only broadcast here.
#if KEY_PARALLEL==1
     if(nnumnod>1) call gcomb(E_scf,1)
#endif

     E_ERROR= E_scf-EEP  ! current E - past E
     EEP    = E_scf
     !!if(mynod==0) write(6,*) niter,E_scf

     ! Check energy related information.
     ! save scf information
     if(niter > 0 .and. niter <= qm_scf_main_c%KITSCF) then
        if(dodiis) then
           MEXT  = 'DIIS'
        else if(kext >= 2) then
           MEXT  = 'DAMP'
        else if(kext == 1) then
           MEXT  = 'YES '
        else
           MEXT  = 'NO  '
        end if
        if(fastdg) then
           MFAST = 'YES '
        else
           MFAST = 'NO  '
        end if
        EES(niter) = E_scf
        ERS(niter) = E_error
        PLS(niter) = PL
        MXS(niter) = MEXT
        MFS(niter) = MFAST
        EDS(niter) = EDMAX
     end if

     ! scf convergence test and set flag for fast diagonalization.
     ! if kext=-1, meaning the scf convergence has been achieved.
     if(niter > 0) then
        if(kext == -1) exit Scfloop  ! exit main iteration loop
        if(abs(E_error) < qm_scf_main_c%SCFCRT .and. PL < qm_scf_main_c%PLCRT .and. kext /= 1) then
           !ITSAVE= niter+1  ! itsave may not be used in the current
                             ! implementation.
           kext   =-1        ! converged, and exit at next step.
        else
           ! so, meaning, not yet scf converged.
           if(niter > qm_scf_main_c%KITSCF) then
              ! Scf failed by exceeding scf cycle; Exit the main loop.
              scf_succeed=.false.
              exit Scfloop
           end if

           ! for DXL-BOMD
           if(qm_control_c%q_dxl_bomd .and. qm_control_c%q_do_dxl_scf .and. qm_control_c%N_scf_step == niter) then
              kext   =-1        ! converged, and exit at next step.
           end if
           ! FASTDG on, if PL < TRANS (=0.10d0) .and. niter < kitscf-50.
           if(fockmd_on) then
              FASTDG =(PL.lt.TRANS .and. (niter.ge.1))
           else
              FASTDG =(PL.lt.TRANS .and. (niter.gt.1))
           end if
           if(niter > qm_scf_main_c%KITSCF-50) FASTDG =.false. ! turn off is niter approaches kitscf 
        end if
     end if

     ! for now, consider it is not parallelized.
     ! diagonalize the F-matrix.
     ! info: error code.
     if(FASTDG) then
        if(qm_gho_info_c%q_gho) then
           call square(qm_gho_info_c%FAHB,qm_gho_info_c%FAHBwrk,     &
                       qm_gho_info_c%norbhb,qm_gho_info_c%norbhb, &
                       qm_gho_info_c%lin_norbhb,.false.) !.true.)
           call fast_diag(qm_gho_info_c%FAHBwrk,qm_gho_info_c%CAHB,EA,Q, &   ! qm_gho_info_c%FAHBwrk,
                          qm_gho_info_c%norbhb,qm_gho_info_c%norbhb,qm_main_c%numb, &
#if KEY_PARALLEL==1
                          KPARPT_fast_a(0:numnod),                                  &
#endif
                          fmo_local_a,nv1d_a,nv2d_a,mstart_fast,mstop_fast,         &
                          kstart_fast_a,kstop_fast_a,fstart_fast_a,fstop_fast_a)
        else
           call square(FA,qm_scf_main_c%FAwork,qm_main_c%norbs,dim_norbs,dim_linear_norbs,.false.) !.true.)
           call fast_diag(qm_scf_main_c%FAwork,CA,EA,Q,dim_norbs, &          ! qm_scf_main_c%FAwork,
                          qm_main_c%norbs,qm_main_c%numb,                       &
#if KEY_PARALLEL==1
                          KPARPT_fast_a(0:numnod),                              &
#endif
                          fmo_local_a,nv1d_a,nv2d_a,mstart_fast,mstop_fast,     &
                          kstart_fast_a,kstop_fast_a,fstart_fast_a,fstop_fast_a)
        end if
        if(UHF) then
           if(qm_gho_info_c%q_gho) then
              call square(qm_gho_info_c%FBHB,qm_gho_info_c%FBHBwrk,     &
                          qm_gho_info_c%norbhb,qm_gho_info_c%norbhb, &
                          qm_gho_info_c%lin_norbhb,.false.) !.true.)
              call fast_diag(qm_gho_info_c%FBHBwrk,qm_gho_info_c%CBHB,EB,Q, &  ! qm_gho_info_c%FBHBwrk,
                             qm_gho_info_c%norbhb,qm_gho_info_c%norbhb,qm_main_c%nbeta, &
#if KEY_PARALLEL==1
                             KPARPT_fast_b(0:numnod),                                   &
#endif
                             fmo_local_b,nv1d_b,nv2d_b,mstart_fast,mstop_fast,          &
                             kstart_fast_b,kstop_fast_b,fstart_fast_b,fstop_fast_b)
           else
              call square(FB,qm_scf_main_c%FBwork,qm_main_c%norbs,dim_norbs,dim_linear_norbs,.false.) !.true.)
              call fast_diag(qm_scf_main_c%FBwork,CB,EB,Q,dim_norbs, &         ! qm_scf_main_c%FBwork,
                             qm_main_c%norbs,qm_main_c%nbeta,                      &
#if KEY_PARALLEL==1
                             KPARPT_fast_b(0:numnod),                              &
#endif
                             fmo_local_b,nv1d_b,nv2d_b,mstart_fast,mstop_fast,     &
                             kstart_fast_b,kstop_fast_b,fstart_fast_b,fstop_fast_b)
           end if
        end if
        !!if(mynod==0) write(6,'(A17,F12.5)')'Time (FAST-DIAG)=',(t2-t1)*1000.0d0
     else
        ! When do the full diagonalization, instead of call diagon, here we 
        ! call explicitly square and evvrsp, which do the job of diagon when
        ! IDIAG=0 (default diagonalizer).
        ! refer DIAGON.f and full_diagonalization.f
        !
        if(qm_gho_info_c%q_gho) then
           call square(qm_gho_info_c%FAHB,qm_gho_info_c%FAHBwrk, &
                       qm_gho_info_c%norbhb,qm_gho_info_c%norbhb, &
                       qm_gho_info_c%lin_norbhb,.false.)
           call evvrsp(qm_gho_info_c%norbhb,qm_gho_info_c%norbhb, &
                       qm_gho_info_c%norbhb,                      &
                       qm_gho_info_c%FAHBwrk,Q,IWORK,EA,qm_gho_info_c%CAHB,INFO,.false.)
        else
           call square(FA,qm_scf_main_c%FAwork,qm_main_c%norbs,dim_norbs,dim_linear_norbs,.false.)
           call evvrsp(qm_main_c%norbs,qm_main_c%norbs,dim_norbs,           &
                       qm_scf_main_c%FAwork,Q,IWORK,EA,CA,INFO,.false.)
        end if
        !!if(mynod==0) write(6,'(A17,F12.5)')'Time (FULL-DIAG)=',(t2-t1)*1000.0d0
        ! Error section:
        if(info /= 0 .and. prnlev >= 2) write(6,400) info
        !
        if(UHF) then
           if(qm_gho_info_c%q_gho) then
              call square(qm_gho_info_c%FBHB,qm_gho_info_c%FBHBwrk, &
                          qm_gho_info_c%norbhb,qm_gho_info_c%norbhb, &
                          qm_gho_info_c%lin_norbhb,.false.)
              call evvrsp(qm_gho_info_c%norbhb,qm_gho_info_c%norbhb, &
                          qm_gho_info_c%norbhb,                      &
                          qm_gho_info_c%FBHBwrk,Q,IWORK,EB,qm_gho_info_c%CBHB,INFO,.false.)
           else
              call square(FB,qm_scf_main_c%FBwork,qm_main_c%norbs,dim_norbs,dim_linear_norbs,.false.)
              call evvrsp(qm_main_c%norbs,qm_main_c%norbs,dim_norbs,           &
                          qm_scf_main_c%FBwork,Q,IWORK,EB,CB,INFO,.false.)
           end if
           ! Error section:
           if(info /= 0 .and. prnlev >= 2) write(6,405) info
        end if
        ! error
        if(info /= 0) then
           if(ndiis > 0) ndiis=ndiis-1
           if(prnlev >= 2) write(6,410)
           Cycle Scfloop     ! main iteration do loop
        end if
     end if
     niter  = niter+1

     ! COMPUTE THE DENSITY MATRIX AND EXTRAPOLATE, IF POSSIBLE.
     ! for GHO:
     if (qm_gho_info_c%q_gho) then
        if(UHF) then
           call cal_density_matrix(qm_gho_info_c%CAHB,qm_gho_info_c%DAHB,     &
                                   qm_gho_info_c%PAHB,qm_gho_info_c%FAHBwrk,PL,  &
                                   qm_gho_info_c%norbhb,                      &
                                   qm_gho_info_c%lin_norbhb,                  &
                                   qm_gho_info_c%norbhb,qm_main_c%NALPHA,     &
                                   0,0, niter,kext,nstart,nstep,              &
                                   mstart_a_dens,mstop_a_dens)
           call cal_density_matrix(qm_gho_info_c%CBHB,qm_gho_info_c%DBHB,     &
                                   qm_gho_info_c%PBHB,qm_gho_info_c%FBHBwrk,PM,  &
                                   qm_gho_info_c%norbhb,                      &
                                   qm_gho_info_c%lin_norbhb,                  &
                                   qm_gho_info_c%norbhb,qm_main_c%NBETA ,     &
                                   0,0,niter,kext,nstart,nstep,               &
                                   mstart_b_dens,mstop_b_dens)
           if(PM > PL) PL=PM
 
           ! do the GHO expansion
           call GHO_expansion(qm_gho_info_c%norbhb,qm_gho_info_c%naos,     &
                              qm_gho_info_c%lin_naos,qm_gho_info_c%nqmlnk, &
                              qm_gho_info_c%lin_norbhb,dim_norbs,          &
                              dim_linear_norbs,qm_gho_info_c%mqm16,        &
                              PL,PM,PA,PB,                                 &
                              qm_gho_info_c%PAHB,qm_gho_info_c%PBHB,       &
                              qm_gho_info_c%PAOLD,qm_gho_info_c%PBOLD,     &
                              qm_gho_info_c%QMATMQ,qm_gho_info_c%BT,qm_gho_info_c%BTM, &
                              qm_scf_main_c%indx,UHF)
        else
           call cal_density_matrix(qm_gho_info_c%CAHB,qm_gho_info_c%DAHB,     &
                                   qm_gho_info_c%PAHB,qm_gho_info_c%FAHBwrk,PL,  &
                                   qm_gho_info_c%norbhb,                      &
                                   qm_gho_info_c%lin_norbhb,                  &
                                   qm_gho_info_c%norbhb,qm_main_c%numb,       &
                                   qm_main_c%iodd,qm_main_c%jodd,niter,kext,nstart,nstep, &
                                   mstart_a_dens,mstop_a_dens)

           ! do the GHO-expasion.
           call GHO_expansion(qm_gho_info_c%norbhb,qm_gho_info_c%naos,     &
                              qm_gho_info_c%lin_naos,qm_gho_info_c%nqmlnk, &
                              qm_gho_info_c%lin_norbhb,dim_norbs,          &
                              dim_linear_norbs,qm_gho_info_c%mqm16,        &
                              PL,PM,PA,PA,                                 &
                              qm_gho_info_c%PAHB,qm_gho_info_c%PAHB,       &
                              qm_gho_info_c%PAOLD,qm_gho_info_c%PAOLD,     &
                              qm_gho_info_c%QMATMQ,qm_gho_info_c%BT,qm_gho_info_c%BTM, &
                              qm_scf_main_c%indx,UHF)
        end if
     else          ! q_gho
        if(UHF) then
           call cal_density_matrix(CA,DA,PA,qm_scf_main_c%FAwork,PL,     &
                                   dim_norbs,dim_linear_norbs,           &
                                   qm_main_c%norbs,qm_main_c%nalpha,     &
                                   0,0,niter,kext,nstart,nstep,          &
                                   mstart_a_dens,mstop_a_dens)
           call cal_density_matrix(CB,DB,PB,qm_scf_main_c%FAwork,PM,     &
                                   dim_norbs,dim_linear_norbs,           &
                                   qm_main_c%norbs,qm_main_c%nbeta ,     &
                                   0,0,niter,kext,nstart,nstep,          &
                                   mstart_b_dens,mstop_b_dens)
           if(PM > PL) PL=PM
        else
           call cal_density_matrix(CA,DA,PA,qm_scf_main_c%FAwork,PL,     &
                                   dim_norbs,dim_linear_norbs,           &
                                   qm_main_c%norbs,qm_main_c%numb,       &
                                   qm_main_c%iodd,qm_main_c%jodd,niter,kext,nstart,nstep, &
                                   mstart_a_dens,mstop_a_dens)
        end if
     end if        ! q_gho
     !!!if(mynod==0) write(6,'(A17,F12.5)')'Time (DENSITY  )=',(t2-t1)*1000.0d0
  End do Scfloop   ! main iteration loop
  ! END OF SCF LOOP.
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  ! for Fock matrix dynamics (ref: ).
  if(qm_control_c%q_fockmd .and. qm_control_c%md_run) then
     ! save it for the next md step.
     qm_fockmd_diis_c%FA_sv(1:msize) = FA(mstart:mstop)
  end if
  !if(mynod==0) write(6,*) 'SCF Final: ',niter,E_scf

  ! for GHO method: since CA is not explicitly used within the scf cycle. So,
  ! transform orbitals to AO basis here, used by Mulliken analysis.
  if(qm_gho_info_c%q_gho) then
     ! this CTRASF appears not to be necessary.
     call CTRASF(qm_main_c%norbs,qm_gho_info_c%nqmlnk,qm_gho_info_c%mqm16, &
                 qm_gho_info_c%norbhb,qm_gho_info_c%naos,                  &
                 qm_gho_info_c%BT,qm_gho_info_c%CAHB,CA)
     if(UHF) then
        ! for UHF:
        call CTRASF(qm_main_c%norbs,qm_gho_info_c%nqmlnk,qm_gho_info_c%mqm16, &
                    qm_gho_info_c%norbhb,qm_gho_info_c%naos,                  &
                    qm_gho_info_c%BT,qm_gho_info_c%CBHB,CB)
        ! store the density for gho-related derivative calculation.
        do i=1,dim_linear_norbs
           qm_gho_info_c%PHO(i) = qm_gho_info_c%PAHB(i)
           qm_gho_info_c%PBHO(i)= qm_gho_info_c%PBHB(i)
        end do
     else
        ! for RHF: it only needs to be done here, not in the GHO_expansion.
        ! store the density for gho-related derivative calculation (density in HB N+4 basis).
        qm_gho_info_c%PHO(1:dim_linear_norbs)=qm_gho_info_c%PAHB(1:dim_linear_norbs)*two
     end if
  end if

  ! Scf convergence has been achieved
  if(scf_succeed) then
     ! if successful at 2nd attempt for scf using Diis.
     if(icall == -1 .and. qm_control_c%q_diis) then 
        if(prnlev >= 2) write(6,580)
        icall = 0
     end if
  else       ! scf, failed?
     ! no scf convergence.
     if(prnlev >= 2) then
        write(6,560) 'SCF_ITER routine'
        write(6,520)
        if(qm_control_c%q_diis) then
           write(6,505)
           do i=1,MIN(niter,qm_scf_main_c%KITSCF)
              write(6,510) i,EES(i),ERS(i),PLS(i),MXS(i),MFS(i),EDS(i)
           end do
           write(6,530) NITER,E_scf,E_error,PL,EDMAX
        else
           write(6,500)
           do i=1,MIN(niter,qm_scf_main_c%KITSCF)
              write(6,510) i,EES(i),ERS(i),PLS(i),MXS(i),MFS(i)
           end do
           write(6,530) NITER,E_scf,E_error,PL
        end if
        write(6,540) qm_scf_main_c%SCFCRT,qm_scf_main_c%PLCRT
     end if
     icall  = -1  !  indicates failure to reach SCF convergence.
  end if

  ! specific for qm/mm-Ewald part. deallocate local memory
  if(mm_main_c%LQMEWD) then
     if(allocated(empot_all))   deallocate(empot_all)
     if(allocated(empot_local)) deallocate(empot_local)
  end if

  400 FORMAT(1X,'Full diagonalization: Failed in alpha with error code ',I6,'.')
  405 FORMAT(1X,'Full diagonalization: Failed in beta  with error code ',I6,'.')
  410 FORMAT(1X,'Full diagonalization: Back to the calling routine and try recover.')
  500 FORMAT(///5X,'INFORMATION ON SCF ITERATIONS.',                        &
             // 5X,'NITER',12X,'ENERGY',14X,'DELTAE',14X,'DELTAP',          &
               11X,'EXTRAP',7X,'FAST'/)
  505 FORMAT(///5X,'INFORMATION ON SCF ITERATIONS.',                        &
             // 5X,'NITER',12X,'ENERGY',14X,'DELTAE',14X,'DELTAP',          &
               11X,'EXTRAP',7X,'FAST',11X,'DIIS ERROR'/)
  510 FORMAT(   5X,I3,4X,3F20.10,9X,A4,8X,A4,F20.10)
  520 FORMAT(// 5X,'UNABLE TO ACHIEVE SCF CONVERGENCE'/)
  530 FORMAT(/  5X,I3,4X,3F20.10,25X,F20.10)
  540 FORMAT(/  5X,'CONVERGENCE CRITERIA',7X,2F20.10//)
  560 FORMAT(///5X,A)
  580 FORMAT(/  5X,'SCF CONVERGENCE HAS BEEN ACHIEVED USING DIIS.')


  return

  contains
     subroutine setup_array_index(gho_use)
        !
        ! index/variables for parallelizations
        ! 
        implicit none
        logical :: gho_use

        ! Prepare array for vector allgather calls using VDGBRE
        ! mapping for each node (Hard weird).
#if KEY_PARALLEL==1
        JPARPT_local(0)=0
        KPARPT_local(0)=0
        do i=1,numnod
           JPARPT_local(i)= dim_linear_norbs*i/numnod ! for linear vector
           KPARPT_local(i)= numat*i/numnod
        end do
        mstart = JPARPT_local(mynod)+1
        mstop  = JPARPT_local(mynod+1)                ! for dim_linear_norbs (also LM4 in fockx)
        msize  = (mstop-mstart)+1

        mkstart= KPARPT_local(mynod)+1                ! for numat
        mkstop = KPARPT_local(mynod+1) 

        JPARPT_fock(0)=0
        do i=1,numnod
           JPARPT_fock(i)= dim_linear_fock*i/numnod
        end do
        fstart = JPARPT_fock(mynod)+1                 !
        fstop  = JPARPT_fock(mynod+1)                 ! for fockx call (LM6)

        ! used in function escf
        kkmax  = (dim_norbs*(dim_norbs+1))/2          ! indx(dim_norbs)+dim_norbs
        kstart = kkmax*mynod/numnod + 1
        kstop  = kkmax*(mynod+1)/numnod

        kmstart= dim_norbs*mynod/numnod + 1
        kmstop = dim_norbs*(mynod+1)/numnod

        ! used in diis
        if(gho_use) then
           JPARPT_diis(0)=0
           do i=1,numnod
              JPARPT_diis(i) = qm_gho_info_c%lin_norbhb*i/numnod
           end do
        else
           JPARPT_diis(0)=0
           do i=1,numnod
              JPARPT_diis(i) = dim_linear_norbs*i/numnod
           end do
        end if
        mstart_diis = JPARPT_diis(mynod)+1
        mstop_diis  = JPARPT_diis(mynod+1)

        ! used in cal_density_matrix
        if(UHF) then
           mstart_a_dens = qm_main_c%nalpha*mynod/numnod + 1
           mstop_a_dens  = qm_main_c%nalpha*(mynod+1)/numnod

           mstart_b_dens = qm_main_c%nbeta*mynod/numnod + 1
           mstop_b_dens  = qm_main_c%nbeta*(mynod+1)/numnod
        else
           mstart_a_dens = qm_main_c%numb*mynod/numnod + 1
           mstop_a_dens  = qm_main_c%numb*(mynod+1)/numnod
        end if
#else
        mstart = 1
        mstop  = dim_linear_norbs
        msize  = (mstop-mstart)+1

        mkstart= 1
        mkstop = numat

        fstart = 1
        fstop  = dim_linear_fock

        ! used in function escf
        kstart = 1
        kstop  = (dim_norbs*(dim_norbs+1))/2

        kmstart= 1
        kmstop = dim_norbs

        ! used in diis
        if(gho_use) then
           mstart_diis = 1
           mstop_diis  = qm_gho_info_c%lin_norbhb
        else
           mstart_diis = 1
           mstop_diis  = dim_linear_norbs
        end if

        ! used in cal_density_matrix
        if(UHF) then
           mstart_a_dens = 1
           mstop_a_dens  = qm_main_c%nalpha

           mstart_b_dens = 1
           mstop_b_dens  = qm_main_c%nbeta
        else
           mstart_a_dens = 1
           mstop_a_dens  = qm_main_c%numb
        end if
#endif
        return
     end subroutine setup_array_index
  end subroutine scf_iter


  subroutine calc_mulliken(numat,num_orbs,core,Q,mul_chg,             &
                           mstart,mstop,                              &
#if KEY_PARALLEL==1
                           KPARPT_local,                              &
#endif
                           indx_local)
  !
  ! Calculates the Mulliken charges on QM atoms from QM/MM calculations 
  !
  ! Variables:
  !   i       : a number from 1 to numat
  !   mul_chg : Mulliken charges returned, electron units
  !
  use chm_kinds
  use number,only : zero
  use parallel, only : mynod,numnod

  implicit none
  !
  integer, intent(in) :: numat
  integer             :: mstart,mstop
#if KEY_PARALLEL==1
  integer, intent(in) :: KPARPT_local(0:numnod)
#endif
  integer, intent(in) :: num_orbs(numat)
  real(chm_real),intent(in) :: core(*)
  real(chm_real),intent(in) :: Q(*)
  real(chm_real),intent(out):: mul_chg(numat)
  integer, intent(in) :: indx_local(*)

  ! local variables
  integer :: i,j,KL,KL2,iorbs
  real(chm_real) :: ch_local
!!  integer, save :: mstart,mstop
!!
!!  !
!!#if KEY_PARALLEL==1
!!  mstart = KPARPT_local(mynod)+1
!!  mstop  = KPARPT_local(mynod+1)
!!#else
!!  mstart= 1
!!  mstop = numat
!!#endif

  KL = 1
#if KEY_PARALLEL==1
  do i=1,mstart-1
     iorbs= num_orbs(i)
     KL   = KL + indx_local(iorbs) + iorbs
  end do
#endif
  do i=mstart,mstop
     iorbs    = num_orbs(i)
     !KL2      = KL
     ch_local = Q(KL)  ! Q(KL2); KL2 = KL
     if(iorbs >= 9) then
        do j=1,8
           !!KL2     = KL + indx_local(j+1) + j
           ch_local= ch_local + Q(KL+indx_local(j+1)+j) ! Q(KL2)
        end do
     else if(iorbs >= 4) then
        do j=1,3
           !KL2     = KL + indx_local(j+1) + j
           ch_local= ch_local + Q(KL+indx_local(j+1)+j) ! Q(KL2)
        end do
     end if
     ! core is taken cared together (see also fockx routine). 
     mul_chg(i) = core(i) - ch_local
     KL         = KL + indx_local(iorbs) + iorbs
  end do
#if KEY_PARALLEL==1
  if(numnod>1) call VDGBRE(mul_chg,KPARPT_local)
#endif

  return
  end subroutine calc_mulliken


  real(chm_real) function escf(N,P,F,H,mstart,mstop,kstart,kstop,dim_linear_norbs)
  !
  ! Compute contributions to the electronic energy.
  !
  ! NOTATION. I=INPUT,O=OUTPUT.
  ! N         NUMBER OF BASIS FUNCTIONS (I).
  ! P(LM4)    DENSITY MATRIX (I).
  ! F(LM4)    FOCK MATRIX (I) OR CORE HAMILTONIAN (I).
  !
  use chm_kinds
  use number, only : zero,half
#if KEY_PARALLEL==1
  use parallel
#endif
  !
  implicit none

  integer :: N, dim_linear_norbs
  real(chm_real):: P(dim_linear_norbs),F(dim_linear_norbs),H(dim_linear_norbs)
  integer :: mstart,mstop,kstart,kstop

  real(chm_real):: ddot1d_mn ! external function

  ! local variables
  integer :: KMAX,I,J,IORBS,jmax
  real(chm_real):: EE,EEDIAG
  !
!!  integer,save :: old_N = 0
!!  integer,save :: nnumnod
!!
!! the followings are defined in subroutine scf_iter
!!  kmax = (n*(n+1))/2 !  indx_local(n)+n  ! (N*(N+1))/2
!!#if KEY_PARALLEL==1
!!  !nnumnod = numnod
!!  ! mstart= n*mynod/numnod+1
!!  ! mstop = n*(munod+1)/numnod
!!  ! kstart= kmax*mynod/numnod+1
!!  ! kstop = kmax*(munod+1)/numnod
!!#else
!!  !nnumnod = 1
!!  ! mstart= 1
!!  ! mstop = n
!!  ! kstart= 1
!!  ! kstop = kmax
!!#endif

  !EE = DOT_PRODUCT(P(kstart:kstop),F(kstart:kstop)) + DOT_PRODUCT(P(kstart:kstop),H(kstart:kstop))
  EE   = ddot1d_mn(kstop-kstart+1,P(kstart:kstop),1,F(kstart:kstop),H(kstart:kstop),1,.true.)

  ! for diagonal term:
  EEdiag = zero
  j=(mstart*(mstart-1))/2  ! counter update
  do i=mstart,mstop   ! 1,n
     j      = j+i     ! = qm_scf_main_c%INDX(i)+i = (i*(i+1))/2
     EEdiag = EEdiag + P(j)*(F(j)+H(j))
  end do

  ! this will not be broadcasted here... see above.
  ESCF = EE - half*EEdiag
  return
  end function escf


  ! for qm/mm-Ewald part.
  SUBROUTINE qm_ewald_add_fock(numat,nfirst,num_orbs,indx,fock_matrix,qm_charges,dim_linear_norbs)
  !
  ! 1) empot_all: contains Eslf + Empot at QM atom.
  ! 2) Modify Fock matrix in diagonal elements
  !
  use chm_kinds
  use qm1_constant
  use qm1_info, only : qm_scf_indx_c
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none

  ! Passed in
  integer, intent(in)    :: numat,dim_linear_norbs
  integer, intent(in)    :: nfirst(numat),num_orbs(numat),indx(*)
  real(chm_real),intent(in)   :: qm_charges(numat)
  real(chm_real),intent(inout):: fock_matrix(*)

  ! Local variables
  integer        :: i,j,ia,LL,iorbs
  real(chm_real) :: ewdpot
  real(chm_real),parameter :: ev_a0 = EV*A0 ! convert (electrons/angstrom) to (eV/Bohr)
  integer :: nnumnod,istart

  ! parallelization is synchronized with calc_mulliken routine
  !                    to avoid broad casting.
#if KEY_PARALLEL==1
  nnumnod = numnod
  istart  = mynod+1

  if(.not.qm_scf_indx_c%q_do_atom_setup) call fill_q_do_atom(qm_scf_indx_c%q_do_atom_setup)
#else
  nnumnod = 1
  istart  = 1
#endif

  ! add Empot+Eslf contribution to the diagonal elements of the fock matrix
  do i = istart, numat
#if KEY_PARALLEL==1
     if(qm_scf_indx_c%q_do_atom(i)) then
#endif
        ia = nfirst(i)
        LL = indx(ia)+ia

        ewdpot = empot_all(i)*ev_a0   ! conversion here.
        iorbs  = num_orbs(i)
        fock_matrix(LL)= fock_matrix(LL)-ewdpot
        if(iorbs >= 9) then
           do j=1,8
              LL             = INDX(j+ia)+j+ia
              fock_matrix(LL)= fock_matrix(LL)-ewdpot
           end do
        else if(iorbs >= 4) then
           do j=1,3
              LL             = INDX(j+ia)+j+ia
              fock_matrix(LL)= fock_matrix(LL)-ewdpot
           end do
        end if
#if KEY_PARALLEL==1
     end if
#endif
  end do

  return

#if KEY_PARALLEL==1
  !
  contains
     subroutine fill_q_do_atom(q_ewd)
     !
     ! loop over to fill q_do_atom. to be synchronized with subroutine fockx.
     !
     implicit none
     logical :: q_ewd
     integer :: fstart,fstop

     if(q_ewd) return    ! if .true., it is setup.

     fstart = dim_linear_norbs*(mynod)  /numnod + 1
     fstop  = dim_linear_norbs*(mynod+1)/numnod

     loopII: do i = istart, numat
        qm_scf_indx_c%q_do_atom(i) =.false.
        ia = nfirst(i)
        LL = indx(ia)+ia
        if(LL>=fstart .and. LL<=fstop) then
           qm_scf_indx_c%q_do_atom(i) =.true.
           cycle loopII
        end if
        iorbs  = num_orbs(i)
        if(iorbs >= 9) then
           do j=1,8
              LL             = INDX(j+ia)+j+ia
              if(LL>=fstart .and. LL<=fstop) then
                 qm_scf_indx_c%q_do_atom(i) =.true.
                 cycle loopII
              end if
           end do
        else if(iorbs >= 4) then
           do j=1,3
              LL             = INDX(j+ia)+j+ia
              if(LL>=fstart .and. LL<=fstop) then
                 qm_scf_indx_c%q_do_atom(i) =.true.
                 cycle loopII
              end if
           end do
        end if
     end do loopII
     q_ewd =.true. 
     return
     end subroutine fill_q_do_atom
#endif
  END SUBROUTINE qm_ewald_add_fock


  real(chm_real) function qm_ewald_correct_ee(numat,nfirst,num_orbs,mstart,mstop,indx,p)
  !
  ! Up to this poiint, the energy for the Ewald sum only included half of the term
  ! from MM atoms, and half from the QM image atoms. The QM atoms should contribute
  ! only half, but the MM atoms should contribute in full. This routine compute the
  ! rest of the energy for this.
  !
  ! This routine should be called after a call to escf.
  !
  use chm_kinds
  use qm1_constant
#if KEY_PARALLEL==1
  use parallel
#endif
  !
  implicit none

  ! Passed in
  integer, intent(in) :: numat
  integer, intent(in) :: nfirst(numat),num_orbs(numat),indx(*)
  integer             :: mstart,mstop
  real(chm_real),intent(in) :: P(*)

  ! Local variables
  integer        :: i,i1,ia,ib,i2,iorbs,j,LL
  real(chm_real) :: etemp
  real(chm_real),parameter :: ev_a0 = EV*A0 ! convert (electrons/angstrom) to (eV/Bohr)

!!  !
!!  integer,save :: mstart,mstop
!!
!!  ! for parallelization
!!  mstart=1
!!  mstop =numat
!!#if KEY_PARALLEL==1
!!  if(numnod>1) mstart = ISTRT_CHECK(mstop,numat)
!!#endif

  etemp    = zero
  do i = mstart,mstop                     ! 1, numat
     ia   = nfirst(i)
     LL   = indx(ia)+ia
     iorbs= num_orbs(i)
     etemp= etemp-empot_local(i)*P(LL)
     if(iorbs >= 9) then
        do j=1,8
           LL    = indx(j+ia)+j+ia
           etemp = etemp -empot_local(i)*P(LL)
        end do
     else if(iorbs >= 4) then
        do j=1,3
           LL    = indx(j+ia)+j+ia
           etemp = etemp -empot_local(i)*P(LL)
        end do
     end if
  end do
  qm_ewald_correct_ee = half*etemp*ev_a0

  return
  END function qm_ewald_correct_ee


  subroutine cal_density_matrix(C,D,P,PN,PL,                         &
                                dim_norbs,dim_linear_norbs,          &
                                N,NOCC,                              &
                                IODD,JODD,NITER,KEXT,NSTART,NSTEP,   &
                                mstart,mstop)
  !
  ! CALCULATION OF DENSITY MATRIX.
  ! OPTIONALLY COMBINED WITH EXTRAPOLATION OR DAMPING.
  !
  ! NOTATION. I=INPUT, O=OUTPUT.
  ! C(dim_norbs,dim_norbs)  EIGENVECTORS (I).
  ! D(dim_linear_norbs)     DIFFERENCE DENSITY MATRIX (I,O).
  ! P(dim_linear_norbs)     DENSITY MATRIX (I,O).
  ! PN(dim_linear_norbs)    SCRATCH ARRAY.
  ! PL        MAXIMUM CHANGE IN DIAGONAL ELEMENTS (O).
  ! N         NUMBER OF BASIS FUNCTIONS (I).
  ! NOCC      NUMBER OF OCCUPIED ORBITALS (I).
  ! IODD      INDEX OF FIRST  SINGLY OCCUPIED RHF-MO (I).; likely 0 for RHF
  ! JODD      INDEX OF SECOND SINGLY OCCUPIED RHF-MO (I).; likely 0 for RHF
  ! NITER     NUMBER OF SCF ITERATION (I).
  ! KEXT      TYPE OF SCF ITERATION (I,O).
  !           =-1 CONVERGED DENSITY, NO MODIFICATION ALLOWED (I).
  !           = 0 STANDARD CASE, NO EXTRAPOLATION OR DAMPING (O).
  !           = 1 EXTRAPOLATION FOR DENSITY DONE (O).
  !           = 2 DAMPING FOR DENSITY DONE (O).
  ! ISPIN     TYPE OF DENSITY MATRIX (I).
  !           = 1 RHF OR UHF-ALPHA DENSITY MATRIX (I).
  !           = 2 UHF-BETA DENSITY MATRIX (I).
  ! NSTART    do damping/extrapolation?
  !           =-1, do not do damping/extrap., if DIIS
  !           = 4, do damping/extrap. from 4th scf cycl, if no DIIS
  ! NSTEP     step 
  !           = 4
  !
  ! COMMENTS ON OTHER AVAILABLE INPUT OPTIONS.
  ! NSTEP .GT.0  -  ATTEMPT EXTRAPOLATION, IF CURRENTLY POSSIBLE.
  ! NSTEP .LT.0  -  ATTEMPT DAMPING, IF CURRENTLY POSSIBLE.
  ! NSTART.LT.0  -  NO EXTRAPOLATION OR DAMPING, PL STILL EVALUATED.
  ! NITER .LT.0  -  NO EXTRAPOLATION OR DAMPING, PL NOT EVALUATED.
  !
  use qm1_info, only : qm_scf_indx_c
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none
  !
  integer:: dim_norbs,dim_linear_norbs
  integer:: N,NOCC,IODD,JODD,NITER,KEXT,NSTART,NSTEP
  real(chm_real):: C(dim_norbs,dim_norbs),D(dim_linear_norbs), &
                   P(dim_linear_norbs),PL, &
                   PN(dim_linear_norbs) ! PN(dim_norbs*dim_norbs)
  integer:: mstart,mstop

  ! local variables
  integer :: I,J,K,II,IJ
  real(chm_real):: YLAMB,DEN1,DEN2,DKPI,DKI,A,FAC,c_tmp
  real(chm_real), save :: YL
  real(chm_real) :: YCRIT=0.06D0
  !

  !!  ! for parallelization
  !!#if KEY_PARALLEL==1
  !!  mstart  = nocc*mynod/numnod + 1
  !!  mstop   = nocc*(mynod+1)/numnod
  !!#else
  !!  mstart  = 1
  !!  mstop   = nocc
  !!#endif

  ! so, if diis is on, 
  if(NSTART < 0) then  ! meaning, Diis is on.
     ! simplified code without extrapolation or damping.
     !
     ! calculate density matrix by matrix multiplication.
     ! likely, either rhf, or uhf
     ! imocc=1 for default
     !call zzero(N,N,PN,dim_norbs)
     !call dgemm_mn('N','T',N,N,NOCC,one,C,dim_norbs,C,dim_norbs,zero,PN,dim_norbs)
     !call linear (PN,PN,N,dim_norbs,dim_linear_norbs)
     !
     ! only do a lower diagonal, as C:=A*B'
     ii = 0
     do i=1,n
        ii         = ii + i
        qm_scf_indx_c%pn_diag(i) = p(ii)
     end do

     p(1:dim_linear_norbs) = zero
     do k=mstart,mstop              ! 1,nocc
        ij = 0
        do i=1,n
           c_tmp       =c(i,k)
           p(ij+1:ij+i)=p(ij+1:ij+i)+c_tmp*c(1:i,k)
           ij          =ij+i
        end do
     end do

     ! as cal_density_matrix is called mostly with iodd=0, jodd=0. So, do not worry much now.
     ! this part is of concern for UHF or multiplicity larger than 1.
     if(iodd > 0) then
        if(iodd >= mstart .and. iodd <=mstop) then
           ij = 0
           do i=1,n
              c_tmp       =PT5*C(i,iodd)
              P(ij+1:ij+i)=P(ij+1:ij+i)-c_tmp*C(1:i,iodd)
              ij          =ij+i
           end do
        end if
        if(jodd > 0) then
           if(jodd >= mstart .and. jodd <=mstop) then
             ij = 0
             do i=1,n
                c_tmp       =PT5*C(i,jodd)
                P(ij+1:ij+i)=P(ij+1:ij+i)-c_tmp*C(1:i,jodd)
                ij          =ij+i
             end do
           end if
        end if
     end if

#if KEY_PARALLEL==1
     if(numnod>1) call gcomb(p(1:dim_linear_norbs),dim_linear_norbs)
#endif

     ! calculate maximum change in diagonal matrix elements.
     if(niter >= 0 .and. nstart < 0) then
        PL  = zero
        ii  = 0
        do i=1,n
           ii = ii + i  ! =qm_scf_main_c%INDX(i)+i  ! I*(I+1)/2
           !if(ABS(P(ii)-qm_scf_indx_c%pn_diag(i)) > PL) PL=ABS(P(ii)-qm_scf_indx_c%pn_diag(i))
           PL = max(ABS(P(ii)-qm_scf_indx_c%pn_diag(i)),PL)
        end do
     end if

  else
     ! full calculation allowing for extrapolation or damping. the new density
     ! matrix P(i) and the difference density matrix D(i) are determined along
     ! with some auxiliary variables.
     !
     ! initialization:
     if(niter == 1) then
        YL  = zero
        d(1:dim_linear_norbs)=zero
     end if
     YLAMB  = zero
     DEN1   = zero
     DEN2   = zero
     PL     = zero
     ! calculate density matrix by matrix multiplication.
     !call zzero(N,N,PN,dim_norbs)
     !call dgemm_mn('N','T',N,N,NOCC,one,C,dim_norbs,C,dim_norbs,zero,PN,dim_norbs)
     !call linear(PN,PN,N,dim_norbs,dim_linear_norbs)
     !
     ! only do a lower diagonal part, as C:=A*B'
     pn(1:dim_linear_norbs) = zero
     do k=mstart,mstop              ! 1,nocc
        ij = 0
        do i=1,n
           c_tmp        =c(i,k)
           pn(ij+1:ij+i)=pn(ij+1:ij+i)+c_tmp*c(1:i,k)
           ij           =ij+i
        end do
     end do

     ! iodd=0; jodd=0 for default
     if(iodd > 0) then
        if(iodd >= mstart .and. iodd <=mstop) then
           ij = 0
           do i=1,n
              c_tmp        =PT5*C(i,iodd)
              PN(ij+1:ij+i)=PN(ij+1:ij+i)-c_tmp*C(1:i,iodd)
              ij           =ij+i
           end do
        end if
        if(jodd > 0) then
          if(jodd >= mstart .and. jodd <=mstop) then
             ij = 0
             do i=1,n
                c_tmp        =PT5*C(i,jodd)
                PN(ij+1:ij+i)=PN(ij+1:ij+i)-c_tmp*C(1:i,jodd)
                ij           =ij+i
             end do
          end if
        end if
     end if
#if KEY_PARALLEL==1
     if(numnod>1) call gcomb(pn(1:dim_linear_norbs),dim_linear_norbs)
#endif

     ! downgraded (not parallized).
     do ij=1,dim_linear_norbs
        DKPI  = PN(ij)-P(ij)
        DKI   = D(ij)
        DEN1  = DEN1 +DKI *DKI
        DEN2  = DEN2 +DKPI*DKPI
        YLAMB = YLAMB+DKI*DKPI
        D(ij) = DKPI
        P(ij) = PN(ij)
     end do

     ! calculate maximum change in diagonal matrix elements.
     ii = 0
     do i=1,n
        ii    = ii + i  ! =qm_scf_main_c%INDX(i)+i  ! I*(I+1)/2
        PL    = MAX(abs(D(ii)),PL)
     end do

     ! check for exrapolation or damping.
     ! P(i)  holds the new density matrix.
     ! D(i)  holds the difference between the new and old density matrix.
     !
     ! return if Kext is negative (see argument list).
     if(kext < 0) return   ! meaning, it is converged density.

     kext = 0
     if(nstep > 0) then
        ! for extrapolation
        if(niter >= nstart) then
           ii = niter-nstart
           ii = ii-(ii/nstep)*nstep
           if(ii == 0) kext=1
        end if
        if((niter < 2) .or. (den1 == zero) .or. (den2 == zero)) then
           kext = 0
        else
           YLAMB = YLAMB/DEN1
           if(ABS(YLAMB) >= one)     YLAMB=YLAMB*DEN1/DEN2
           if(ABS(YLAMB-YL) > YCRIT) kext=0
           if(YLAMB == one)          kext=0
           YL    = YLAMB
        end if
        if(kext == 1) FAC = one/(one-YLAMB)-one
     else if(nstep < 0) then
        ! for damping
        if(niter >= nstart .and. niter > 1) then
           kext = 2
           fac  = DBLE(MIN(-nstep,9))*PT1   ! /10.0D0
        end if
     end if
     ! update of density matrix 
     if(kext>0) P(1:dim_linear_norbs)=P(1:dim_linear_norbs)+fac*D(1:dim_linear_norbs)
  end if      ! (NSTART.LT.0)

  return
  end subroutine cal_density_matrix


  subroutine bond_analysis(numat,PA,dim_norbs,dim_linear)
  !
  ! Determine bond order indices and valencies based on
  ! D.R.ARMSTRONG, P.G.PERKINS, AND J.J.P.STEWART, J.CHEM.SOC.DALTON 838 (1973).
  !
  ! For now, only assume RHO.
  !
  use qm1_info, only : qm_control_c,qm_main_c,qm_param_c
  !!use qm1_parameters,only : CORE
  use stream,only : prnlev

  implicit none
  integer :: numat,dim_norbs,dim_linear
  real(chm_real):: PA(dim_linear)         ! P_alpha density linear matrix

  integer :: i,j,ia,ib,ja,jb,ii,ij,jj,n_size,iunit
  real(chm_real):: x_tmp,y_tmp
  real(chm_real),allocatable:: qm_charge(:), &
                               qmqm_bond(:), &
                               PA_sq(:,:)

  n_size = numat*(numat+1)/2  ! size or array.
  allocate(qm_charge(numat))
  allocate(qmqm_bond(n_size))
  allocate(pa_sq(dim_norbs,dim_norbs))

  ! make square matrix.
  ij = 0
  do i=1,dim_norbs
     do j=1,i 
        ij = ij + 1
        pa_sq(j,i) = pa(ij)
        pa_sq(i,j) = pa(ij)
     end do
  end do

  ! now
  ij = 0
  do i=1,numat
     ia = qm_main_c%nfirst(i)
     ib = qm_main_c%nlast(i)
     do j=1,i
        ij = ij + 1
        ja = qm_main_c%nfirst(j)
        jb = qm_main_c%nlast(j)
        x_tmp = zero
        do ii = ia,ib
           do jj = ja,jb
              x_tmp = x_tmp + pa_sq(ii,jj)*pa_sq(ii,jj)
           end do
        end do
        qmqm_bond(ij) = four*x_tmp
     end do
     x_tmp = -qmqm_bond(ij)
     y_tmp = zero
     do ii=ia,ib
        x_tmp = x_tmp + four*pa_sq(ii,ii)
        y_tmp = y_tmp + pa_sq(ii,ii)
     end do
     qmqm_bond(ij) = x_tmp
     qm_charge(i)  =-y_tmp*two+qm_param_c%CORE(i)
  end do

  ! now printing..
  iunit = qm_control_c%ianal_unit
  if(prnlev >= 2) then
     ! bond order and mulliken charges
     write(iunit,120)
     do i=1,numat
        ia = i*(i-1)/2 + 1
        ib = ia + i - 1
        ii = ib - ia + 1
        if(ii<=25) then
           write(iunit,250) qm_control_c%qminb(i),qm_charge(i),(qmqm_bond(ij),ij=ia,ib)
        else 
           jj = ia + 25 - 1
           write(iunit,250) qm_control_c%qminb(i),qm_charge(i),(qmqm_bond(ij),ij=ia,jj)
           do 
              ia = jj + 1
              ii = ib - ia + 1
              if(ii<=25) then
                 write(iunit,260) (qmqm_bond(ij),ij=ia,ib)
                 exit
              else
                 jj = ia + 25 -1
                 write(iunit,260) (qmqm_bond(ij),ij=ia,jj)
              end if
           end do 
        end if
     end do
  end if
  100 format(/  5X,'Mulliken Charge       ')
  110 format(/  5X,'Bond order matrix     ')
  120 format(/  5X,'Mulliken Charge and Bond order matrix')
  130 format(   1X,I7,F10.4)
  150 format(   1X,I7,1X,25F9.4)
  160 format(   9X,25F9.4)
  250 format(   1X,I7,1X,F10.4,1X,25F9.4)
  260 format(   20X,25F9.4)

  ! free memory.
  if(allocated(qm_charge)) deallocate(qm_charge)
  if(allocated(qmqm_bond)) deallocate(qmqm_bond)
  if(allocated(pa_sq))     deallocate(pa_sq)

  return
  end subroutine bond_analysis


  subroutine fockx(F,PA,PB,Q,W,LM4,LM6,UHF,numat,nfirst,nlast,num_orbs,NW,indx_local,   &
                   mstart,mstop,fstart,fstop,                                           &
#if KEY_PARALLEL==1
                   JPARPT_fock,                                                         &
#endif
                   ip_local,ip_check,                                                   &
                   ip1_local,ip2_local,jp1_local,jp2_local,jp3_local,jx_local)
  !
  ! TWO-ELECTRON CONTRIBUTIONS TO MNDO-TYPE FOCK MATRIX.
  ! SCALAR CODE FOR TWO-CENTER EXCHANGE CONTRIBUTIONS.
  !
  ! NOTATION. I=INPUT, O=OUTPUT, S=SCRATCH.
  ! F(LM4)    FOCK MATRIX (I,O).
  ! PA(LM4)   RHF OR UHF-ALPHA DENSITY MATRIX (I).
  ! PB(LM4)   UHF-BETA DENSITY MATRIX (I).
  ! Q(LM6)    SCRATCH ARRAY FOR ONE-CENTER PAIR TERMS (S).
  ! W(LM6,*)  TWO-ELECTRON INTEGRALS (I).
  !
  use qm1_info, only : qm_scf_indx_c
#if KEY_PARALLEL==1
  use parallel 
#endif

  implicit none

  integer :: LM4,LM6,numat
  real(chm_real):: F(LM4),PA(LM4),PB(LM4),Q(LM6),W(LM6,LM6)
  logical :: UHF,ip_check(*)
  integer :: nfirst(numat),nlast(numat),num_orbs(numat),NW(numat)
  integer :: indx_local(*),ip_local(*),ip1_local(*),ip2_local(*), &
             jp1_local(*),jp2_local(*),jp3_local(*),jx_local(*)
  integer :: mstart,mstop,fstart,fstop
#if KEY_PARALLEL==1
  integer :: JPARPT_fock(0:numnod)
#endif

  ! local variables
  integer :: i,j,K,L,N
  integer :: IA,IB,IC,II,IJ,IK,IL,IS,IW,IX,IY,IZ,I2,JA,JB,JJ,JK,JL,JW,J2
  integer :: KA,KB,KL,KS,KX,KY,KZ,LL
  integer :: IJK,IJS,IJW,IXS,IYS,IZS,IYX,IZX,IZY,KLS,KLW,KL2
  integer :: IJMIN,KLMIN,KLMAX,KSTART
  integer :: IORBS,IORBS_tmp,JORBS,NPASS
  real(chm_real):: A,SUM,sum2,temp,temp2,temp_sum,WIJKL,PA_tmp(4),F_tmp(4)
  real(chm_real),parameter ::ev_a0=EV*A0

  integer,parameter :: IWW(4,4)=reshape( (/1,2,4,7, 2,3,5,8, &
                                  4,5,6,9, 7,8,9,10/),(/4,4/))
  integer       :: mmynod,nnumnod,iicnt
  real(chm_real):: ddot_mn ! external function

  ! for parallelization
#if KEY_PARALLEL==1
  mmynod = mynod
  nnumnod= numnod
#else
  nnumnod= 1
  mmynod = 0
#endif


  ! Coulomb contributions:
  ! one-center exchange contributions are implicitly included for RHF.
  !
  ! Note: when use parallel, Q contains only terms between fstart and fstop. 
  !       Then, it is broadcasted. (Should be synchronized with calc_mulliken routine.)
  !
  ! LM6: dim_linear_fock: one center AO pairs (see determine_qm_scf_arrray_size)
  !      so this is much smaller than fock size.
  if(UHF) then
     do KL=fstart,fstop  ! 1,LM6
        ! Q(KL)  = (PA(ip_local(KL))+PB(ip_local(KL)))
        ! if(ip1_local(KL).ne.ip2_local(KL)) Q(KL)=Q(KL)*TWO
        if(ip_check(kl)) then
           Q(KL)= two*(PA(ip_local(KL))+PB(ip_local(KL)))
        else
           Q(KL)= (PA(ip_local(KL))+PB(ip_local(KL)))
        end if
     end do
  else
     do KL=fstart,fstop  ! 1,LM6
        ! Q(KL)  = two*PA(ip_local(KL))
        ! if(ip1_local(KL).ne.ip2_local(KL)) Q(KL)=Q(KL)*TWO
        ! see below in define_pair_index for ip_check defintion.
        if(ip_check(kl)) then
           Q(KL)= two*(PA(ip_local(KL))+PA(ip_local(KL)))
        else
           Q(KL)= (PA(ip_local(KL))+PA(ip_local(KL)))
        end if
     end do
  end if
#if KEY_PARALLEL==1
  if(nnumnod>1) call VDGBRE(Q,JPARPT_fock)

  ! fill the q_mynod_fock array.
  call fill_q_mynod_fock(qm_scf_indx_c%q_mynod_fock_setup) 
#endif

  ! for serial
  !F(ip_local(1:lm6))=F(ip_local(1:lm6))+MATMUL(Q(1:LM6),W(1:LM6,1:LM6))
  !
  ! now with parallel
  !do ij=1,LM6
  !   F(ip_local(ij))=F(ip_local(ij)) + ddot_mn(fstop-fstart+1,Q(fstart:fstop),1,W(fstart:fstop,ij),1)
  !end do
  do ij=1,LM6
     ik = ip_local(ij)
     if(ik>=mstart .and. ik<=mstop) then
        F(ik) = F(ik) + ddot_mn(LM6,Q(1:LM6),1,W(1:LM6,ij),1)
     end if
  end do

  ! two-center exchange contibutions: offdiagonal two-center terms (ij,kl).
  loopII: do ii=2,numat  ! 1,NUMAT
     iicnt  = ((ii-1)*(ii-2))/2
     ia     = NFIRST(ii)
     ib     = NLAST(ii)
     ic     = indx_local(ia)
     iorbs  = num_orbs(ii)
     iw     = indx_local(iorbs)+iorbs
     ij     = NW(ii)-1
     loopJJ: do jj=1,ii-1
        ja     = NFIRST(jj)
        jb     = NLAST(jj)
        jorbs  = num_orbs(jj)
        jw     = indx_local(jorbs)+jorbs
        KL     = NW(jj)-1
#if KEY_PARALLEL==1
        iicnt  = iicnt + 1
        if(qm_scf_indx_c%q_mynod_fock(iicnt)) cycle loopJJ
#endif
        if(iw==1 .and. jw==1) then
           is   = ic+ja
           F(is)= F(is)-PA(is)*W(ij+1,KL+1)
        else if(iw==1 .and. jw==10) then
           is  = ic+ja
           ix  = is+1
           iy  = is+2
           iz  = is+3
           ijs = ij+1
           PA_tmp(1)= PA(is)
           PA_tmp(2)= PA(ix)
           PA_tmp(3)= PA(iy)
           PA_tmp(4)= PA(iz)
           F(is) = F(is) - (PA_tmp(1)*W(kl+1,ijs)+PA_tmp(2)*W(kl+2,ijs)+PA_tmp(3)*W(kl+4,ijs)+PA_tmp(4)*W(kl+ 7,ijs))
           F(ix) = F(ix) - (PA_tmp(1)*W(kl+2,ijs)+PA_tmp(2)*W(kl+3,ijs)+PA_tmp(3)*W(kl+5,ijs)+PA_tmp(4)*W(kl+ 8,ijs))
           F(iy) = F(iy) - (PA_tmp(1)*W(kl+4,ijs)+PA_tmp(2)*W(kl+5,ijs)+PA_tmp(3)*W(kl+6,ijs)+PA_tmp(4)*W(kl+ 9,ijs))
           F(iz) = F(iz) - (PA_tmp(1)*W(kl+7,ijs)+PA_tmp(2)*W(kl+8,ijs)+PA_tmp(3)*W(kl+9,ijs)+PA_tmp(4)*W(kl+10,ijs))
        else if(iw==10 .and. jw==1) then
           is  = ic+ja
           ix  = indx_local(ia+1)+ja
           iy  = indx_local(ia+2)+ja
           iz  = indx_local(ia+3)+ja
           kls = KL+1
           PA_tmp(1)= PA(is)
           PA_tmp(2)= PA(ix)
           PA_tmp(3)= PA(iy)
           PA_tmp(4)= PA(iz)
           F(is) = F(is) - (PA_tmp(1)*W(ij+1,kls)+PA_tmp(2)*W(ij+2,kls)+PA_tmp(3)*W(ij+4,kls)+PA_tmp(4)*W(ij+ 7,kls))
           F(ix) = F(ix) - (PA_tmp(1)*W(ij+2,kls)+PA_tmp(2)*W(ij+3,kls)+PA_tmp(3)*W(ij+5,kls)+PA_tmp(4)*W(ij+ 8,kls))
           F(iy) = F(iy) - (PA_tmp(1)*W(ij+4,kls)+PA_tmp(2)*W(ij+5,kls)+PA_tmp(3)*W(ij+6,kls)+PA_tmp(4)*W(ij+ 9,kls))
           F(iz) = F(iz) - (PA_tmp(1)*W(ij+7,kls)+PA_tmp(2)*W(ij+8,kls)+PA_tmp(3)*W(ij+9,kls)+PA_tmp(4)*W(ij+10,kls))
        else if(iw==10 .and. jw==10) then
           do i=1,4
              is  = indx_local(ia+i-1)+ja
              ix  = is+1
              iy  = is+2
              iz  = is+3
              F_tmp(1:4)=zero
              do k=1,4
                 ks  = indx_local(ia+k-1)+ja
                 kx  = ks+1
                 ky  = ks+2
                 kz  = ks+3
                 ijk = ij+IWW(k,i)  ! IWW(i,k)
                 PA_tmp(1)=PA(ks)
                 PA_tmp(2)=PA(kx)
                 PA_tmp(3)=PA(ky)
                 PA_tmp(4)=PA(kz)
                 F_tmp(1) =F_tmp(1)+PA_tmp(1)*W(kl+1,ijk)+PA_tmp(2)*W(kl+2,ijk)+PA_tmp(3)*W(kl+4,ijk)+PA_tmp(4)*W(kl+ 7,ijk)
                 F_tmp(2) =F_tmp(2)+PA_tmp(1)*W(kl+2,ijk)+PA_tmp(2)*W(kl+3,ijk)+PA_tmp(3)*W(kl+5,ijk)+PA_tmp(4)*W(kl+ 8,ijk)
                 F_tmp(3) =F_tmp(3)+PA_tmp(1)*W(kl+4,ijk)+PA_tmp(2)*W(kl+5,ijk)+PA_tmp(3)*W(kl+6,ijk)+PA_tmp(4)*W(kl+ 9,ijk)
                 F_tmp(4) =F_tmp(4)+PA_tmp(1)*W(kl+7,ijk)+PA_tmp(2)*W(kl+8,ijk)+PA_tmp(3)*W(kl+9,ijk)+PA_tmp(4)*W(kl+10,ijk)
              end do
              F(is) = F(is) - F_tmp(1)
              F(ix) = F(ix) - F_tmp(2)
              F(iy) = F(iy) - F_tmp(3)
              F(iz) = F(iz) - F_tmp(4)
           end do
        else
           ! General code   - also valid for D-orbitals.
           ! contributions from (ii,kl)
           loopI1: do i=ia,ib
              ka  = indx_local(i)
              ijw = ij+indx_local(i-ia+2)
              klw = kl
              do k=ja,jb
                 ik   = ka+k
                 sum  = zero
                 temp = PA(ik)
                 do l=ja,k-1
                    il    = ka+l
                    klw   = klw+1
                    sum   = sum   + W(klw,ijw)*PA(il)
                    F(il) = F(il) - W(klw,ijw)*temp   ! W(klw,ijw)*PA(ik)
                 end do
                 ! for L=K
                 il    = ka+k
                 klw   = klw+1
                 F(ik) = F(ik) - sum - W(klw,ijw)*PA(il)  ! -sum of A*PA(il)
              end do
           end do loopI1
           ! contribution from (ij,kl) with i.ne.j
           loopI2: do i=ia+1,ib
              ka = indx_local(i)
              do j=ia,i-1
                 kb  = indx_local(j)
                 ijw = ij+indx_local(i-ia+1)+j-ia+1
                 klw = kl
                 do k=ja,jb
                    ik    = ka+k
                    jk    = kb+k
                    sum   = zero
                    sum2  = zero
                    temp  = PA(ik)
                    temp2 = PA(jk)
                    do l=ja,k-1
                       il   = ka+l
                       jl   = kb+l
                       klw  = klw+1
                       sum  = sum + W(klw,ijw)*PA(jl)
                       sum2 = sum2+ W(klw,ijw)*PA(il)
                       ! for K.NE.L
                       F(il)= F(il) - W(klw,ijw)*temp2 ! A*PA(jk)
                       F(jl)= F(jl) - W(klw,ijw)*temp  ! A*PA(ik)
                    end do
                    ! for L=K
                    il     = ka+k
                    jl     = kb+k
                    klw    = klw+1
                    F(ik)  = F(ik) - sum - W(klw,ijw)*PA(jl)  ! - sum of A*PA(jl)
                    F(jk)  = F(jk) - sum2- W(klw,ijw)*PA(il)  ! - sum of A*PA(il)
                 end do
              end do
           end do loopI2
        end if
     end do loopJJ
  end do loopII

  ! one-center exchange contributions for UHF.
  ! offdiagonal one-center terms (ii,kk) for SP-basis.
  if(UHF) then
     do i=mmynod+1,NUMAT, nnumnod            ! if not parallel, mmynod=0,nnumnod=1
        ia     = NFIRST(I)
        iorbs  = num_orbs(i)
        if(iorbs==4) then
           ij     = NW(i)-1
           ixs    = indx_local(ia+1)+ia
           iys    = indx_local(ia+2)+ia
           izs    = indx_local(ia+3)+ia
           iyx    = iys+1
           izx    = izs+1
           izy    = izs+2
           F(ixs) = F(ixs)-PA(ixs)*W(ij+3,ij+1)
           F(iys) = F(iys)-PA(iys)*W(ij+6,ij+1)
           F(izs) = F(izs)-PA(izs)*W(ij+10,ij+1)
           F(iyx) = F(iyx)-PA(iyx)*W(ij+6,ij+3)
           F(izx) = F(izx)-PA(izx)*W(ij+10,ij+3)
           F(izy) = F(izy)-PA(izy)*W(ij+10,ij+6)
        end if
     end do
     ! diagonal one-center terms (ij,ij), general code.
     do ij=fstart,fstop  ! 1,LM6
        i2    = ip_local(ij)
        i     = ip1_local(ij)
        j     = ip2_local(ij)
        temp  = W(ij,ij)

        F(i2) = F(i2) - PA(i2)*temp
        if(i /= j) then
           ii    = indx_local(i)+i
           jj    = indx_local(j)+j
           F(ii) = F(ii)-PA(jj)*temp
           F(jj) = F(jj)-PA(ii)*temp
        end if
     end do
     ! All one-center terms are now included for an SP-basis.
     ! Any offidiagonal terms for an SPD-basis. 
#if KEY_PARALLEL==1
     iicnt = 0
#endif
     loopN: do n=1,NUMAT
        ia     = NFIRST(n)
        iorbs  = num_orbs(n)
        if(iorbs > 4) then
           klmin  = jx_local(n)
           klmax  = klmin+iorbs*iorbs-1
           kstart = jp1_local(klmin)
           loopKL: do kl=klmin,klmax
#if KEY_PARALLEL==1
              iicnt  = iicnt + 1
              if(mmynod /= mod(iicnt-1,nnumnod)) cycle loopKL
#endif
              k      = jp1_local(kl)
              l      = jp2_local(kl)
              ! only elements F(ik) wiht i.ge.k are needed. therefore, 
              ! the loop over ij can start at ijmin.ge.klmin.
              ijmin  = klmin+iorbs*(k-kstart)
              j2     = jp3_local(kl)
              loopIJ: do ij=ijmin,klmax
                 ! diagonal terms have been included above also for an SPD-basis.
                 if(jp3_local(ij) /= j2) then
                    wijkl  = W(jp3_local(ij),j2)
                    if(wijkl /= zero) then
                       i   = jp1_local(ij)
                       j   = jp2_local(ij)
                       ik  = indx_local(i)+k
                       jl  = indx_local(MAX(j,l))+MIN(j,l)
                       F(ik) = F(ik)-PA(jl)*WIJKL
                    end if
                 end if
              end do loopIJ
           end do loopKL
        end if
     end do loopN
  end if  ! UHF

  return

#if KEY_PARALLEL==1
  ! 
  contains
     subroutine fill_q_mynod_fock(q_fock)
     !
     ! loop over to fill q_mynod_fock
     !
     implicit none
     logical :: q_fock

     if(q_fock) return

     ! two-center exchange contibutions: offdiagonal two-center terms (ij,kl).
     ! this routine loops over and check if elements belongs mynod (mstart <= ii <=mstop)
     iicnt  = 0
     loopII: do ii=1,NUMAT
        ia     = NFIRST(ii)
        ib     = NLAST(ii)
        ic     = indx_local(ia)
        iorbs  = num_orbs(ii)
        iw     = indx_local(iorbs)+iorbs
        loopJJ: do jj=1,ii-1
           ja     = NFIRST(jj)
           jb     = NLAST(jj)
           jorbs  = num_orbs(jj)
           jw     = indx_local(jorbs)+jorbs
           iicnt  = iicnt + 1
           qm_scf_indx_c%q_mynod_fock(iicnt) =.true.   ! .true. : skip iicnt
                                                       ! .false.: do calculate iicnt
           if(iw == 1 .and. jw == 1) then
              is   = ic+ja
              if(is>=mstart .and. is<=mstop) qm_scf_indx_c%q_mynod_fock(iicnt) =.false.
           else if(iw == 1 .and. jw == 10) then
              is  = ic+ja
              ix  = is+1
              iy  = is+2
              iz  = is+3
              if( (is>=mstart .and. is<=mstop) .or. (ix>=mstart .and. ix<=mstop) .or. &
                  (iy>=mstart .and. iy<=mstop) .or. (iz>=mstart .and. iz<=mstop)) then
                 qm_scf_indx_c%q_mynod_fock(iicnt) =.false.
              end if
           else if(iw == 10 .and. jw == 1) then
              is  = ic+ja
              ix  = indx_local(ia+1)+ja
              iy  = indx_local(ia+2)+ja
              iz  = indx_local(ia+3)+ja
              if( (is>=mstart .and. is<=mstop) .or. (ix>=mstart .and. ix<=mstop) .or. &
                  (iy>=mstart .and. iy<=mstop) .or. (iz>=mstart .and. iz<=mstop)) then
                 qm_scf_indx_c%q_mynod_fock(iicnt) =.false.
              end if
           else if(iw == 10 .and. jw == 10) then
              do i=1,4
                 is  = indx_local(ia+i-1)+ja
                 ix  = is+1
                 iy  = is+2
                 iz  = is+3
                 if( (is>=mstart .and. is<=mstop) .or. (ix>=mstart .and. ix<=mstop) .or. &
                     (iy>=mstart .and. iy<=mstop) .or. (iz>=mstart .and. iz<=mstop)) then
                    qm_scf_indx_c%q_mynod_fock(iicnt) =.false.
                    cycle loopJJ
                 end if
              end do
           else
              ! General code   - also valid for D-orbitals.
              ! contributions from (ii,kl)
              loopI1: do i=ia,ib
                 ka  = indx_local(i)
                 do k=ja,jb
                    ik   = ka+k
                    if(ik>=mstart .and. ik<=mstop) then
                       qm_scf_indx_c%q_mynod_fock(iicnt) =.false.
                       cycle loopJJ
                    end if
                    do l=ja,k-1
                       il    = ka+l
                       if(il>=mstart .and. il<=mstop) then
                          qm_scf_indx_c%q_mynod_fock(iicnt) =.false.
                          cycle loopJJ
                       end if
                    end do
                 end do
              end do loopI1
   
              ! contribution from (ij,kl) with i.ne.j
              loopI2: do i=ia+1,ib
                 ka = indx_local(i)
                 do j=ia,i-1
                    kb  = indx_local(j)
                    do k=ja,jb
                       ik    = ka+k
                       jk    = kb+k
                       if((ik>=mstart .and. ik<=mstop) .or. (jk>=mstart .and. jk<=mstop)) then
                          qm_scf_indx_c%q_mynod_fock(iicnt) =.false.
                          cycle loopJJ
                       end if
                       do l=ja,k-1
                          il   = ka+l
                          jl   = kb+l
                          if((il>=mstart .and. il<=mstop) .or. (jl>=mstart .and. jl<=mstop)) then
                             qm_scf_indx_c%q_mynod_fock(iicnt) =.false.
                             cycle loopJJ
                          end if
                       end do
                    end do
                 end do
              end do loopI2
           end if
        end do loopJJ
     end do loopII

     q_fock =.true.   ! qm_scf_indx_c%q_mynod_fock is setup.
     return
     end subroutine fill_q_mynod_fock
#endif
  end subroutine fockx


  subroutine q_construct(PA,PB,Q,ip_local,ip_check,LM4,LM6,UHF)
  !
  ! Construct Q= PA+PB  (see fockx subroutine)
  !
  ! PA(LM4)   RHF OR UHF-ALPHA DENSITY MATRIX.
  ! PB(LM4)   UHF-BETA DENSITY MATRIX.
  ! Q(LM6)    SCRATCH ARRAY FOR ONE-CENTER PAIR TERMS.
  !
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none

  integer :: LM4,LM6,numat
  real(chm_real):: PA(LM4),PB(LM4),Q(LM6)
  integer :: ip_local(*)
  logical :: UHF,ip_check(*)

  ! local variables
  integer :: KL,i
  integer :: mstart,mstop
#if KEY_PARALLEL==1
  integer :: JPARPT_qstrt(0:numnod)
#endif
  real(chm_real),parameter ::ev_a0=EV*A0

  ! for parallelization
#if KEY_PARALLEL==1
  JPARPT_qstrt(0)=0
  do i=1,numnod
     JPARPT_qstrt(i)= LM6*i/numnod ! for linear vector
  end do
  mstart = JPARPT_qstrt(mynod)+1
  mstop  = JPARPT_qstrt(mynod+1)
#else
  mstart = 1
  mstop  = LM6
#endif

  ! Note: when use parallel, Q contains only terms between mstart and mstop. 
  !       So care should be given in particular at calc_mulliken subroutine.
#if KEY_PARALLEL==1
  if(numnod>1) Q(1:LM6) = zero
#endif
  if(UHF) then
     do KL=mstart,mstop  ! 1,LM6
        if(ip_check(kl)) then
           Q(KL)= two*(PA(ip_local(KL))+PB(ip_local(KL)))
        else
           Q(KL)= (PA(ip_local(KL))+PB(ip_local(KL)))
        end if
     end do
  else
     do KL=mstart,mstop  ! 1,LM6
        if(ip_check(kl)) then
           Q(KL)= two*(PA(ip_local(KL))+PA(ip_local(KL)))
        else
           Q(KL)= (PA(ip_local(KL))+PA(ip_local(KL)))
        end if
     end do
  end if
#if KEY_PARALLEL==1
  if(numnod>1) call VDGBRE(Q,JPARPT_qstrt)
#endif
  return
  end subroutine q_construct


  subroutine diis(FA,FB,PA,PB,FAwork,FBwork,PAwork,PBwork,            &
                  FDA,FDB,Ediis,Adiis,Bdiis,Xdiis,                    &
                  dim_norbs,dim_linear_norbs,N,NDIIS,mxdiis,          &
#if KEY_PARALLEL==1
                  JPARPT_diis,                                        &
#endif
                  mstart,mstop,                                       &
                  iwork_diis,EDMAX,qprint,UHF)
  !
  ! Diis convergence acceleration.
  ! It has possible memory leak as, for example, FA and PB has defined not in a 
  ! square matrix form.
  ! 
  ! REFERENCES.
  !     1) P. PULAY, CHEM.PHYS.LETT. 73, 393-398 (1980).
  !     2) P. PULAY, J.COMPUT.CHEM. 3, 556-560 (1982).
  !     3) T.P. HAMILTON AND P. PULAY, J.CHEM.PHYS. 84, 5728-5734 (1986).
  ! 
  ! NOTATION : I=INPUT, O=OUTPUT, S=SCRATCH.
  ! FA(dim_linear_norbs)    CURRENT FOCK MATRIX (I). RHF OR ALPHA.
  !                         UPDATED FOCK MATRIX (O). RHF OR ALPHA.
  ! FB(dim_linear_norbs)    CURRENT FOCK MATRIX (I). BETA (UHF).
  !                         UPDATED FOCK MATRIX (O). BETA (UHF).
  ! PA(dim_linear_norbs)    CURRENT DENSITY MATRIX (I). RHF OR ALPHA.
  ! PB(dim_linear_norbs)    CURRENT DENSITY MATRIX (I). BETA (UHF).
  ! FDA(dim_linear_norbs,*) FOCK MATRICES FROM DIIS ITERATIONS (I,O). RHF OR ALPHA.
  ! FDB(dim_linear_norbs,*) FOCK MATRICES FROM DIIS ITERATIONS (I,O). BETA (UHF).
  ! ED(dim_linear_norbs,*)  ERROR MATRICES FROM DIFFERENT DIIS ITERATIONS (I,O).
  ! BD(MX1P)                COEFFICIENT MATRIX FOR LINEAR EQUATIONS (I,O).
  ! AD(MX1P)                COEFFICIENT MATRIX FOR LINEAR EQUATIONS (S).
  ! X(MX+1)                 RHS VECTOR FOR LINEAR EQUATIONS (S) OVERWRITTEN BY
  !                         SOLUTIONS  OF  LINEAR EQUATIONS (S).
  ! dim_norbs               LEADING DIMENSION OF CORRESPONDING SQUARE ARRAYS (I).
  ! dim_linear_norbs        LEADING DIMENSION OF F,P,FD,ED: dim_linear_norbs=N*(N+1)/2 (I).
  ! N          NUMBER OF BASIS FUNCTIONS (I).
  ! NDIIS      OVERALL COUNTER FOR DIIS ITERATIONS (I,O).
  ! NPRINT     PRINTING FLAG (I).
  ! EDMAX      MAXIMUM ELEMENT OF ERROR MATRIX (O), ABSOLUTE VALUE.
  !
  ! MATRICES ARE STORED IN PACKED LINEAR FORM.
  ! AD(MX1P) IS DESTROYED DURING THE SOLUTION OF THE LINEAR EQUATIONS.
  ! BD(MX1P) IS KEPT AND UPDATED IN EACH DIIS ITERATION.
  !
  ! ALL MATRICES ARE HELD IN MEMORY WHICH ALLOWS US TO IMPLEMENT A
  ! SIMPLE OVERFLOW MECHANISM IN THE CASE OF NDIIS.GT.MXDIIS:
  ! THERE IS A SEPARATE COUNTER KDIIS=MOD(NDIIS-1,MXDIIS)+1
  ! WHICH REMAINS BETWEEN 1 AND MXDIIS. IN THE CASE OF OVERFLOW,
  ! THE RELEVANT MATRICES ARE UPDATED WITH REGARD TO THE CURRENT
  ! INDEX KDIIS, AND THE DIIS EXTRAPOLATION IS DONE USING THE
  ! LATEST MXDIIS ITERATIONS.
  !
  use qm1_info, only : qm_scf_diis_c
#if KEY_PARALLEL==1
  use parallel
#if KEY_MPI==1  /*MPI run*/
  use mpi_f08
#endif /*MPI run*/
#endif

  implicit none
  !
  integer       :: dim_norbs,dim_linear_norbs,N,NDIIS,mxdiis
  real(chm_real):: FA(dim_linear_norbs),PA(dim_linear_norbs), &
                   FB(dim_linear_norbs),PB(dim_linear_norbs)
  real(chm_real):: FAwork(dim_norbs,dim_norbs),PAwork(dim_norbs,dim_norbs), &
                   FBwork(dim_norbs,dim_norbs),PBwork(dim_norbs,dim_norbs)
  real(chm_real):: FDA(dim_linear_norbs,*),FDB(dim_linear_norbs,*), &
                   Ediis(dim_linear_norbs,*),Adiis(*),Bdiis(*),Xdiis(*)
  real(chm_real):: EDMAX
  integer       :: iwork_diis(*) 
  logical       :: qprint,UHF
#if KEY_PARALLEL==1
  integer       :: JPARPT_diis(0:numnod)
#endif
  integer       :: mstart,mstop

  ! local variables
  integer       :: i,j,k,M,ij,KD,NEW,MX1,KX1P,KDIIS,IEDMAX,info,iidim
  real(chm_real):: CX,cx1,aa,bb
  real(chm_real):: Ediis_kd(dim_linear_norbs)
  real(chm_real),parameter :: SCALE=1.02D0
  integer       :: idamax_mn           ! external function
  real(chm_real):: ddot_mn,ddot2d_mn   ! external function

#if KEY_PARALLEL==1
  real(chm_real):: edmax_local
#if KEY_MPI==1
  integer*4 :: IERR
#endif
#endif
  integer :: nnumnod,mmynod
  real(chm_real):: s_aux

  ! for parallelization
#if KEY_PARALLEL==1
  nnumnod = numnod
  mmynod  = mynod
#else
  nnumnod = 1
  mmynod  = 0
#endif
  iidim  = mstop - mstart + 1  ! dimension

#if KEY_PARALLEL==1
  if(.not. qm_scf_diis_c%q_ij_pair_setup) then
     ij=0
     qm_scf_diis_c%q_ij_pair(1:n)=.false.
     do i=1,n
        do j=1,i
           ij = ij + 1
           if(ij >= mstart .and. ij <=mstop) then
              qm_scf_diis_c%q_ij_pair(i) =.true.
              qm_scf_diis_c%q_ij_pair(j) =.true.
           end if
        end do
     end do
     qm_scf_diis_c%q_ij_pair_setup =.true.
  end if
#endif

  ! initialization.
  ndiis  = ndiis+1
  kdiis  = MOD(ndiis-1,mxdiis) + 1

  ! Compute the error matrix ED = F*P - P*F, using linearly packed matrices.
  ! in principle, Ework(1:n,1:n)= MATMUL(FAwork(1:n,1:n),PAwork(1:n,1:n)) &
  !                              -MATMUL(PAwork(1:n,1:n),FAwork(1:n,1:n))
  ! after each matrix has been squared form.
  !
  call square2(FA,FAwork,PA,PAwork,N,dim_norbs,dim_linear_norbs &
#if KEY_PARALLEL==1
              ,qm_scf_diis_c%q_ij_pair  &
#endif
               )
  !
  ! using a lower diagonal part, C:=A*B-B*A, where FA and PA are symmetric matrices.
  !Ediis_kd(mstart:mstop) = zero
  ij=0
  do i=1,n
     do j=1,i
        ij = ij+1
        if(ij >= mstart .and. ij <=mstop) then
           !Ediis_kd(ij)=Ediis_kd(ij)+ DOT_PRODUCT(FAwork(1:n,j),PAwork(1:n,i)) &
           !                         - DOT_PRODUCT(PAwork(1:n,j),FAwork(1:n,i))
           !Ediis_kd(ij)= ddot_mn(n,FAwork(1:n,j),1,PAwork(1:n,i),1) &
           !             -ddot_mn(n,PAwork(1:n,j),1,FAwork(1:n,i),1)
           ! .false. do minus.
           Ediis_kd(ij)= ddot2d_mn(n,FAwork(1:n,j),PAwork(1:n,j),1,  &
                                     PAwork(1:n,i),FAwork(1:n,i),1,.false.) 
        end if
     end do
  end do

  if(UHF) then
     call square2(FB,FBwork,PB,PBwork,N,dim_norbs,dim_linear_norbs &
#if KEY_PARALLEL==1
                 ,qm_scf_diis_c%q_ij_pair  &
#endif
                  )
     ! using a lower diagonal part, C:=A*B-B*A, where FB and PB are symmetric matrices.
     ij=0
     do i=1,n
        do j=1,i
           ij = ij+1
           if(ij >= mstart .and. ij <=mstop) then
              !Ediis_kd(ij)=Ediis_kd(ij)+ ddot_mn(n,FBwork(1:n,j),1,PBwork(1:n,i),1) &
              !                         - ddot_mn(n,PBwork(1:n,j),1,FBwork(1:n,i),1)
              ! .false. do minus.
              Ediis_kd(ij)=Ediis_kd(ij)+ddot2d_mn(n,FBwork(1:n,j),PBwork(1:n,j),1, &
                                                    PBwork(1:n,i),FBwork(1:n,i),1,.false.)
           end if
        end do
     end do
  end if

!!#if KEY_PARALLEL==1
!!  ! broadcast.
!!  if(nnumnod>1) call VDGBRE(Ediis_kd(1:dim_linear_norbs),JPARPT_diis)
!!#endif


  ! This is only needed for printing purpose... so, think about it how to avoid.
  ! find the largest element of the error matrix (in absolute value).
  iedmax = idamax_mn(mstop-mstart+1,Ediis_kd(mstart:mstop),1) ! dblas function
#if KEY_PARALLEL==1 /*MPI parallel*/
#if KEY_MPI==1 /*MPI run*/
  edmax_local = abs(Ediis_kd(iedmax))
  ! only master node needs this information.
  call MPI_REDUCE(edmax_local,edmax,1,MPI_DOUBLE_PRECISION,MPI_MAX,0,COMM_CHARMM,ierr)
#endif         /*MPI run*/
#else          /*MPI parallel*/
  edmax  = abs(Ediis_kd(iedmax))
#endif         /*MPI parallel*/
  !

  ! STORE THE CURRENT FOCK MATRIX. : FDA is only used in diis routine.
  FDA(mstart:mstop,kdiis) = FA(mstart:mstop)
  if(UHF) FDB(mstart:mstop,kdiis) = FB(mstart:mstop)

  ! Update the coefficient matrix (BD) of the linear equations:
  !   For the defintion of BD< see eq. (1) in reference 2. It is obvious from this
  !   defintion that only the new elements of BD (referring to kdiis) need to be
  !   evaluated.  The nontrivial elements BD(kdiis+1,ldiis+1) are obtained from
  !   the scalar product of the full symmetric error matrices ED for iterations
  !   kdiis and ldiis, respectively.  ED is antisymmetric and zero on the
  !   diagonal, hene this scalar product can be computed as twice the dot-product
  !   of linearly packted matrices (ED). Standard case, DIIS iteractions 1...MXDIIS.
  if(kdiis == ndiis) then
     mx1 = kdiis+1
  else
     mx1 = mxdiis+1  ! case of overflow, ndiis > mxdiis
  end if
  kd        = (kdiis+1)*kdiis/2 + 1  ! =indx_local(kdiis+1)+1
  Bdiis(1)  = zero
  Bdiis(kd) =-one
  Ediis(mstart:mstop,kdiis)=Ediis_kd(mstart:mstop)
#if KEY_PARALLEL==1
  if(nnumnod>1) then
     do i=1,mx1-1
        qm_scf_diis_c%bdiis_local(i)=-two*ddot_mn(mstop-mstart+1,Ediis(mstart:mstop,i),1, &
                                                  Ediis_kd(mstart:mstop),1)
     end do
     call gcomb(qm_scf_diis_c%bdiis_local(1:mx1-1),mx1-1) 
     do i=1,mx1-1
        if(kdiis >= i) then
           new = kd+i
        else
           new = (i+1)*i/2 + 1 + kdiis ! =indx_local(I+1)+1+kdiis
        end if
        Bdiis(new)=qm_scf_diis_c%bdiis_local(i)
     end do
  else
#endif
     do i=1,mx1-1
        if(kdiis >= i) then
           new = kd+i
        else
           new = (i+1)*i/2 + 1 + kdiis ! =indx_local(I+1)+1+kdiis
        end if
        Bdiis(new)=-two*ddot_mn(dim_linear_norbs,Ediis(1:dim_linear_norbs,i),1,  &
                                Ediis_kd(1:dim_linear_norbs),1)
     end do
#if KEY_PARALLEL==1
  end if
#endif
  !if(jdiis.lt.2) &
  Bdiis(kd+kdiis)=Bdiis(kd+kdiis)*SCALE

  ! now, do the diis-extrapolation
  if(ndiis /= 1) then
     ! copy coefficients from BD(mx1p) to AD(mx1p) & scratch array will be
     ! overwritten by dspsv_mn.
     kx1p         = mx1*(mx1-1)/2+mx1 ! =indx_local(mx1)+mx1
     Adiis(1:kx1p)= Bdiis(1:kx1p)

     ! define rhs vector of linear equations.
     Xdiis(1)    =-one
     Xdiis(2:mx1)= zero

     ! solve the system of linear equations.
     call dspsv_mn('U',mx1,1,Adiis,iwork_diis,Xdiis,mx1,info)
     if(info /= 0) then
        if(qprint) write(6,500) info
        ndiis  = 0
        return
     end if
     ! diis extrapolation for the fock matrix.
     if(UHF) then
        do i=mstart,mstop  ! 1,dim_linear_norbs
           FA(i)=zero
           FB(i)=zero
        end do
        do m=2,mx1
           cx     = Xdiis(m)
           !call daxpy_mn(iidim,cx,FDA(mstart:mstop,m-1),1,FA(mstart:mstop),1)
           !call daxpy_mn(iidim,cx,FDB(mstart:mstop,m-1),1,FB(mstart:mstop),1)
           do j=mstart,mstop
              FA(j) = FA(j) + cx*FDA(j,m-1)
              FB(j) = FB(j) + cx*FDB(j,m-1)
           end do
        end do
#if KEY_PARALLEL==1
        if(nnumnod>1) then
           call VDGBRE(FA,JPARPT_diis)
           call VDGBRE(FB,JPARPT_diis)
        end if
#endif
     else
        !FA(mstart:mstop) =zero
        !do m=2,mx1
        !   cx     = Xdiis(m)
        !   call daxpy_mn(iidim,cx,FDA(mstart:mstop,m-1),1,FA(mstart:mstop),1)
        !end do
        !
        ! equivalent to above.
        cx = Xdiis(2)
        FA(mstart:mstop) = cx*FDA(mstart:mstop,1)
        if(mx1>=3) then
           do m=3,mx1
              cx     = Xdiis(m)
              FA(mstart:mstop) = FA(mstart:mstop) + cx*FDA(mstart:mstop,m-1)
           end do
        end if
#if KEY_PARALLEL==1
        ! broadcast here, as FA was a complete matrix before this routine.
        if(nnumnod>1) call VDGBRE(FA,JPARPT_diis)
#endif
     end if
  end if

  return

!  contains
!
!     subroutine cal_commutator(A,B,C,dim_linear_norbs,N,MODE)
!     !
!     ! Evaluate the commutator  C = A*B - B*A  for symmetric matrices.
!     !
!     ! (origially, the routine is DSPMM.)
!     !
!     ! NOTATION. I=INPUT, O=OUTPUT.
!     ! A(dim_linear_norbs)    real symmetric matrix packed linearly (I).
!     ! B(dim_linear_norbs)    real symmetric matrix packed linearly (I).
!     ! C(dim_linear_norbs)    real symmetric matrix packed linearly (I,O).
!     ! dim_linear_norbs       dimension of A,B,C (I).
!     ! N         order of A,B,C (I).
!     ! IND(N)    index array (I): ind(K)=(K*(K-1))/2.
!     ! MODE      initialization of array C (I).
!     !           = 0 initialize to zero.
!     !
!
!     !use number, only : zero
!
!     implicit none
!     integer :: dim_linear_norbs,n,mode
!     real(chm_real):: a(lm4),b(lm4),c(lm4)
!
!     integer :: i,j,k,ik,ij,jk
!     real(chm_real):: aik,bik
!
!     ! initialization
!     if(mode.eq.0) c(1:dim_linear_norbs)=zero
!
!     ! triple loop over all indices.
!     do k=1,n
!        do i=2,n
!           ik     = indx_local(MAX(i,k))+MIN(i,k)
!           aik    = A(ik)
!           bik    = B(ik)
!           do j=1,i-1
!              ij     = indx_local(i)+j
!              jk     = indx_local(MAX(j,k))+MIN(j,k)
!              c(ij)  = c(ij) + aik*b(jk) - bik*a(jk)
!           end do
!        end do
!     end do
!     return
!     end subroutine cal_commutator

500 FORMAT( 1X,'ERROR CODE INFO =',I4,' FROM DSPSV_MN IN DIIS SECTION.',  &
           /1X,'DIIS PROCEDURE IS RESTARTED.')
  end subroutine diis


  subroutine fock_diis(FA,FDA,dim_linear_norbs,mxfdiis_local,msize_local, &
                       ifock_option,ifockmd_counter,fockmd_on,            &
                       mstart,mstop,mmynod,nnumnod)
  !
  ! Fock matrix extrapolation, based on the Taylor expansion.
  !
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none
  !
  integer :: dim_linear_norbs,mxfdiis_local,msize_local,ifockmd_counter,ifock_option
  real(chm_real):: FA(msize_local),FDA(mxfdiis_local,msize_local)
  logical       :: fockmd_on
  integer :: mstart,mstop,mmynod,nnumnod    ! for parallelization

  ! local variables
  integer :: i,j
  real(chm_real):: fval(mxfdiis_local),alpha_1,alpha_2,alpha_3,alpha_4,       &
                   alpha_5,alpha_6,De1p,De1m,De2p,De2m,De3p,De3m,             &
                   f_n2p,f_n2m,f_n3p,f_n3m,f_n4p,df_n2p,df_n2m,df_n3p,df_n3m, &
                   dt2an,dt2an3p
  real(chm_real),parameter :: a_3  = 27.0d0/16.0d0, &  ! 3^3/2^4
                              a_4  = 81.0d0/32.0d0, &  ! 3^4/2^5
                              r_12 =  1.0d0/12.0d0, &  ! 1/12
                              r_24 =  1.0d0/24.0d0, &  ! 1/24
                              r_180=  1.0d0/180.0d0,&  ! 1/180
                              r_a5 = 1024.0d0/(2.0d0*(3.0d0**5)), & !
                              r_a6 = 4096.0d0/(2.0d0*(3.0d0**6))
  real(chm_real),parameter :: aa5=(59.0d0/12.0d0),  &  !  59/12
                              aa4=(-29.0d0/3.0d0),  &  ! -29/3
                              aa3=(19.0d0/2.0d0),   &  !  19/2
                              aa2=(-14.0d0/3.0d0),  &  ! -14/3
                              aa1=(11.0d0/12.0d0)      !  11/12
  real(chm_real),parameter :: bb1=(10.0d0/9.0d0), &
                              bb2=(5.0d0/3.0d0),  &
                              r_120 =(1.0d0/120.0d0), &      ! 1/120
                              r_3120=(1.0d0/(3.0d0*120.0d0)) ! 1/(3*120)
                       

  ! do the fock extrapolation.
  if(ifockmd_counter >= (mxfdiis_local+1)) then
     if(mxfdiis_local==5) then
        if(ifock_option == 1) then
           ! Based on the extrapolation plus cubic fit.
           do i=1,msize_local ! mstart,mstop
              ! copy data points first.
              do j=1,mxfdiis_local-1
                 FDA(j,i) = FDA(j+1,i)
              end do
              FDA(mxfdiis_local,i)    = FA(i)        ! copy a new value.

              !
              ! fval(1)=F_n-2, fval(2)=F_n-1, fval(3)=F_n, fval(4)=F_n+1, fval(5)=F_n+2
              do j=1,mxfdiis_local
                 fval(j) = FDA(j,i)
              end do
              ! a_1 = -(1/12)*(F_n(5)- 8*F_n(4)          + 8*F_n(2)-F_n(1))
              ! a_2 = -(1/24)*(F_n(5)-16*F_n(4)+30*F_n(3)-16*F_n(2)+F_n(1))
              ! a_3 =  (1/12)*(F_n(5)- 2*F_n(4)          + 2*F_n(2)-F_n(1))
              ! a_4 =  (1/24)*(F_n(5)- 4*F_n(4)+ 6*F_n(3)- 4*F_n(2)+F_n(1))
              alpha_1 =-r_12*(fval(5)- 8.0d0*fval(4)               + 8.0d0*fval(2)-fval(1))
              alpha_2 =-r_24*(fval(5)-16.0d0*fval(4)+30.0d0*fval(3)-16.0d0*fval(2)+fval(1))
              alpha_3 = r_12*(fval(5)- 2.0d0*fval(4)               + 2.0d0*fval(2)-fval(1))
              alpha_4 = r_24*(fval(5)- 4.0d0*fval(4)+ 6.0d0*fval(3)- 4.0d0*fval(2)+fval(1))

              !
              FA(i) = fval(3)+3.0d0*alpha_1+ 9.0d0*alpha_2+27.0d0*alpha_3+ 81.0d0*alpha_4
           end do
        else
           !
           ! do the Verlet integration, F(n+1) = 2F(n)-F(n-1) + dt^2 * a(F(n)),
           ! in which a(F(n)) was determined by the Taylor expansion of F(n+i) around F(n),
           ! followed by the second time derivatizations, which leads to the acceleration at time n.
           !
           ! Note that the data points are F(n),F(n-1),F(n-2),F(n-3),F(n-4), and
           ! we use the extrapolation and 4-th order Taylor expansion.
           !
           do i=1,msize_local ! mstart,mstop
              ! copy data points first.
              do j=1,mxfdiis_local-1
                 FDA(j,i) = FDA(j+1,i)
              end do
              FDA(mxfdiis_local,i)    = FA(i)        ! copy a new value.

              ! here, dt^2(d^2F_n+i/dt^2) = (dt^2)*[F_n(2)+ i*dt*F_n(3) + (i^2)*(dt^2)*F_n(4)]
              !                           = (dt^2)*F_n(2) + i*(dt^3)*F_n(3) + (i^2)*(dt^4)*F_n(4)
              ! at i=0 (F_n)
              ! dt^2(d^2F_n+i/dt^2) = (dt^2)*F_n(2)
              !
              ! where
              ! (dt^2)*F_n(2) = (1/12)*(11*F_n-4 -56*F_n-3 +114*F_n-2 -104*F_n-1 + 35*F_n)
              !               = (dt^2)*a(F(n))
              ! (dt)^2*a(F(n))
              dt2an = r_12*(11.0d0*FDA(1,i)-56.0d0*FDA(2,i)+114.0d0*FDA(3,i)-104.0d0*FDA(4,i)+35.0d0*FDA(5,i))

              ! Then, F_n+1 = 2*F_n - F_n-1 + dt^2*a(F(n))
              ! 1: F_n-4, 2: F_n-3, 3: F_n-2, 4: F_n-1, 5: F_n 
              FA(i) = 2.0d0*FDA(5,i)-FDA(4,i)+dt2an
           end do
        end if
     else if(mxfdiis_local==7) then
        if(ifock_option == 1) then
           do i=1,msize_local ! mstart,mstop
              ! copy data points first.
              do j=1,mxfdiis_local-1
                 FDA(j,i) = FDA(j+1,i)
              end do
              FDA(mxfdiis_local,i)    = FA(i)        ! copy a new value.

              !
              ! fval(1)=F_n-3, fval(2)=F_n-2, fval(3)=F_n-1, fval(4)=F_n,
              ! fval(5)=F_n+1, fval(6)=F_n+2, fval(7)=F_n+3
              do j=1,mxfdiis_local
                 fval(j) = FDA(j,i)
              end do
              ! De1p=(1/2)*(F_n+1-2*F_n+F_n-1), De1m=(1/2)*(F_n+1-F_n-1)
              ! De2p=(1/2)*(F_n+2-2*F_n+F_n-2), De2m=(1/2)*(F_n+2-F_n-2)
              ! De3p=(1/2)*(F_n+3-2*F_n+F_n-3), De3m=(1/2)*(F_n+3-F_n-3)
              De1p=0.5d0*(fval(5)-2.0d0*fval(4)+fval(3))
              De1m=0.5d0*(fval(5)              -fval(3))
              De2p=0.5d0*(fval(6)-2.0d0*fval(4)+fval(2))
              De2m=0.5d0*(fval(6)              -fval(2))
              De3p=0.5d0*(fval(7)-2.0d0*fval(4)+fval(1))
              De3m=0.5d0*(fval(7)              -fval(1))

              ! a_1 = (1/1!)*(dt^1)*Fn^(1) =(1/  120)[ 180*De1m - 36*De2m + 4*De3m]
              ! a_2 = (1/2!)*(dt^2)*Fn^(2) =(1/3*120)[ 540*De1p - 54*De2p + 4*De3p]
              ! a_3 = (1/3!)*(dt^3)*Fn^(3) =(1/  120)[ -65*De1m + 40*De2m - 5*De3m]
              ! a_4 = (1/4!)*(dt^4)*Fn^(4) =(1/3*120)[-195*De1p + 60*De2p - 5*De3p]
              ! a_5 = (1/5!)*(dt^5)*Fn^(5) =(1/  120)[   5*De1m -  4*De2m +   De3m]
              ! a_6 = (1/6!)*(dt^6)*Fn^(6) =(1/3*120)[  15*De1p -  6*De2p +   De3p]
              alpha_1 = r_120 *( 180.0d0*De1m - 36.0d0*De2m + 4.0d0*De3m)
              alpha_2 = r_3120*( 540.0d0*De1p - 54.0d0*De2p + 4.0d0*De3p)
              alpha_3 = r_120 *( -65.0d0*De1m + 40.0d0*De2m - 5.0d0*De3m)
              alpha_4 = r_3120*(-195.0d0*De1p + 60.0d0*De2p - 5.0d0*De3p)
              alpha_5 = r_120 *(   5.0d0*De1m -  4.0d0*De2m +       De3m)
              alpha_6 = r_3120*(  15.0d0*De1p -  6.0d0*De2p +       De3p)

              ! F_n+4
              FA(i) = fval(4)+4.0d0*alpha_1+16.0d0*alpha_2+64.0d0*alpha_3+256.0d0*alpha_4+ &
                              1024.0d0*alpha_5+4096.0d0*alpha_6
           end do
        else
           ! Based on the extrapolation and 6-th order Taylor expansion.
           do i=1,msize_local ! mstart,mstop
              ! copy data points first.
              do j=1,mxfdiis_local-1
                 FDA(j,i) = FDA(j+1,i)
              end do
              FDA(mxfdiis_local,i)    = FA(i)        ! copy a new value.
              !
              ! here, dt^2(d^2F_n+i/dt^2) = (dt^2)*[F_n(2)+ i*dt*F_n(3) + (i^2)*(dt^2)*F_n(4) +
              !                                     (i^3)*(dt^3)*F_n(5) + (i^4)*(dt^4)*F_n(6)]
              ! at i=0 (F_n)
              ! dt^2(d^2F_n+i/dt^2) = (dt^2)*F_n(2)
              !
              ! where
              ! (dt^2)*F_n(2) = (1/180)*[137*F_n-6 -972*F_n-5 +2970*F_n-4 -5080*F_n-3 +5265*F_n-2 -3132*F_n-1 +812*F_n]
              !               = (dt^2)*a(F(n))
              ! (dt)^2*a(F(n))
              ! and (1) F_n-6; (2) F_n-5; (3) F_n-4; (4) F_n-3; (5) F_n-2; (6) F_n-1; (7) F_n orders.
              dt2an = r_180*( 137.0d0*FDA(1,i)- 972.0d0*FDA(2,i)+2970.0d0*FDA(3,i)-5080.0d0*FDA(4,i)+ &
                             5265.0d0*FDA(5,i)-3132.0d0*FDA(6,i)+ 812.0d0*FDA(7,i))

              ! F(n+1) = 2*F(n) - F(n-1) + (dt)^2*a(n)
              FA(i) = 2.0d0*FDA(7,i)-FDA(6,i)+dt2an
           end do
        end if
     else
          call wrndie(-1,'<FOCK DIIS>','Wrong extrapolation order.')
     end if
     fockmd_on=.true.
  else
     ! copy...
     ! n-2,n-1,n,n+1,n+2 data points.
     do i=1,msize_local  ! mstart,mstop
        do j=1,mxfdiis_local-1
           FDA(j,i) = FDA(j+1,i)
        end do
        FDA(mxfdiis_local,i)    = FA(i)        ! copy a new value.
     end do
  end if
  !
  return
  end subroutine fock_diis


  subroutine define_pair_index
  !=====================================================================
  !
  ! Definition of pair indices  (refer DYNINT subroutine)
  !
  ! notation:
  ! IP(LMI)   indices of unique one-center AO pairs = I(I-1)/2+J
  ! IP1(LMI)  index of first AO in the one-center pair = I (coul)
  ! IP2(LMI)  index of 2nd   AO in the one-center pair = J (coul)
  ! JP1(LME)  index of first AO in the one-center pair = I (exch)
  ! JP2(LME)  index of 2nd   AO in the one-center pair = J (exch)
  ! JP3(LME)  coulomb pair index for given exchange pair index
  ! JX(LM1)   1st  exchange pair index for given atom
  ! JXLAST    last exchange pair index for last  atom
  ! NW(LM1)   1st  coulumb  pair index for given atom
  !

  use qm1_info, only : qm_main_c, qm_scf_main_c, qm_scf_indx_c, &
                       allocate_pair_index !, qm_gho_info_c

  implicit none
  ! local variables
  integer :: i,j,ii,k,ia,ib,id,nwii,i4,j4

  ! set memory allocation and check.
  call allocate_pair_index(qm_scf_indx_c,qm_scf_main_c%dim_numat,qm_main_c%norbs,qm_main_c%uhf)

  ! define pair indices and pair factors
  if(.not. qm_main_c%uhf) then
     ! Coulomb part.
     k      = 0
     do ii=1,qm_main_c%NUMAT
        qm_scf_main_c%NW(ii) = k+1    ! lower triangle of a given block.
        ia     = qm_main_c%NFIRST(ii) ! at
        ib     = qm_main_c%NLAST(ii)
        do i=ia,ib
           id     = qm_scf_main_c%INDX(i)
           do j=ia,i
              k      = k+1
              qm_scf_indx_c%ip_local(k)  = id+j ! location in the linear Fock matrix.
              if(i == j) then
                 qm_scf_indx_c%ip_check(k)=.false.
              else
                 qm_scf_indx_c%ip_check(k)=.true.  ! used in fockx
              end if
           end do
        end do
     end do

  else
     ! UHF case.

     ! Coulomb part.
     k      = 0
     do ii=1,qm_main_c%NUMAT
        qm_scf_main_c%NW(ii) = k+1    ! lower triangle of a given block.
        ia     = qm_main_c%NFIRST(ii) ! at
        ib     = qm_main_c%NLAST(ii)
        do i=ia,ib
           id     = qm_scf_main_c%INDX(i)
           do j=ia,i
              k      = k+1
              qm_scf_indx_c%ip_local(k)  = id+j ! location in the linear Fock matrix.
              qm_scf_indx_c%ip1_local(k) = i    ! i,j mapping in the Fock matrix.
              qm_scf_indx_c%ip2_local(k) = j    ! (in the form of the square matrix)
              if(i == j) then
                 qm_scf_indx_c%ip_check(k)=.false.
              else
                 qm_scf_indx_c%ip_check(k)=.true.  ! used in fockx
              end if
           end do
        end do
     end do

     ! Exchange part. 
     k    = 0
     do ii=1,qm_main_c%NUMAT
        qm_scf_indx_c%jx_local(ii) = k+1       ! square matrix of a given block.
        nwii   = qm_scf_main_c%NW(ii)-1  ! 
        ia     = qm_main_c%NFIRST(ii)
        ib     = qm_main_c%NLAST(ii)
        do i=ia,ib
           i4        = i-ia+1
           do j=ia,i
              k      = k+1
              j4     = j-ia+1
              qm_scf_indx_c%jp1_local(k) = i  ! mapping for lower triangle.
              qm_scf_indx_c%jp2_local(k) = j
              qm_scf_indx_c%jp3_local(k) = nwii+qm_scf_main_c%INDX(i4)+j4   ! I.GE.J
           end do
           do j=i+1,ib
              k      = k+1
              j4     = j-ia+1
              qm_scf_indx_c%jp1_local(k) = i  ! mapping for upper triangle.
              qm_scf_indx_c%jp2_local(k) = j
              qm_scf_indx_c%jp3_local(k) = nwii+qm_scf_main_c%INDX(j4)+i4  ! I.LT.J
           end do
        end do
     end do
  end if
  qm_scf_indx_c%jxlast_local = k

  return
  end subroutine define_pair_index

  !
#endif
end module qm1_scf_module

