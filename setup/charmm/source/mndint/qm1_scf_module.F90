module qm1_scf_module
  use chm_kinds
  use number
  use qm1_constant

  ! used is pair indexing. (see below define_pair_indix, which is called from qmmm_load_parameters_setup_qm_info.)
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
  integer,pointer,save :: IP_local(:) =>Null(),IP1_local(:)=>Null(),IP2_local(:)=>Null()
  integer,pointer,save :: indx_local(:)=>Null()  ! local copy of indx array
  logical,pointer,save :: ip_check(:)=>Null()
  !
  ! used in uhf case
  integer,pointer,save :: JP1_local(:)=>Null(),JP2_local(:)=>Null(),JP3_local(:)=>Null(), &
                          JX_local(:) =>Null()
  integer,save         :: JXLAST_local

  ! for qm/mm-ewald related.
  real(chm_real),pointer,save :: empot_local(:)=>Null(),empot_all(:)=>Null()

  ! diis related part.
  TYPE, public :: qm_scf_diis
  ! for DIIS related.
    ! FDA(LM4,MXDIIS+1)=FDB; Ediis(LM4,MXDIIS+1); Bdiis(MX1P); Adiix(MX1P); Xdiix(MXDIIS+1);
    ! iwork_diis(6*LMX), LMX=9*LM1
    integer            :: mxdiis=100            ! maximum number of diis iterations allowed.
    integer            :: mx1   =101  ! = mxdiis+1
    integer            :: mx1p  =5151 ! =(mx1*(mx1+1))/2
    !
    real(chm_real),pointer:: FDA(:,:)=>Null(), &! Fock matrices from Diis iterations, RHF or UHF-Alpha.
                             FDB(:,:)=>Null()   !                                   , UHF-beta
    real(chm_real),pointer:: Ediis(:,:)=>Null() ! Error matrixces from Diis iterations.
    real(chm_real),pointer:: Bdiis(:)=>Null()   ! Coefficient matrix for Diis linear equations.
    real(chm_real),pointer:: Adiis(:)=>Null()   ! Coefficient matrix for Diis linear equations.
    real(chm_real),pointer:: Xdiis(:)=>Null()   ! RHS vector and solutions for Diis linear equations.
    ! integer scratch array
    integer,pointer :: iwork_diis(:)=>Null()    ! iwork in routine diis size: 12*numat
  END TYPE qm_scf_diis

  ! fock matrix dynamics (ref: )
  TYPE, public :: qm_fockmd_diis
     integer                :: mxfdiis            ! 5: n-2,n-1,n,n+1,n+2 or..
                                                  ! 7: n-3,n-2,n-1,n,n+1,n+2,n+3
     real(chm_real),pointer :: FA_sv(:)=>Null()   ! Fock matrix from the previous md step.
     real(chm_real),pointer :: FDA(:,:)=>Null()   ! Fock matrices for Fock md iteration.
  END TYPE qm_fockmd_diis

  TYPE(qm_scf_diis), save :: qm_scf_diis_r
  TYPE(qm_fockmd_diis),save :: qm_fockmd_diis_r

  contains
  !
#if KEY_MNDO97==1
  !
  subroutine allocate_deallocate_qm_diis(qm_scf_main_l,qm_scf_diis_l,qm_fockmd_diis_l, &
                                         qdiis,q_fockmd,imax_fdiss,uhf,qallocate)
  !
  ! allocate/deallocate Diis related arrays.
  ! if qallocate == .true. , allocate memory
  !                  false., deallocate memory

  use qm1_info,only : qm_scf_main,Aass
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

  ! deallocate if arrays are associated.
  if(associated(qm_scf_diis_l%FDA))   deallocate(qm_scf_diis_l%FDA,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_diis','FDA')
  if(associated(qm_scf_diis_l%FDB))   deallocate(qm_scf_diis_l%FDB,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_diis','FDB')
  if(associated(qm_scf_diis_l%Ediis)) deallocate(qm_scf_diis_l%Ediis,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_diis','Ediis')
  if(associated(qm_scf_diis_l%Bdiis)) deallocate(qm_scf_diis_l%Bdiis,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_diis','Bdiis')
  if(associated(qm_scf_diis_l%Adiis)) deallocate(qm_scf_diis_l%Adiis,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_diis','Adiix')
  if(associated(qm_scf_diis_l%Xdiis)) deallocate(qm_scf_diis_l%Xdiis,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_diis','Xdiis')
  if(associated(qm_scf_diis_l%iwork_diis)) deallocate(qm_scf_diis_l%iwork_diis,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_diis','iwork_diis')

  !
  if(associated(qm_fockmd_diis_l%FA_sv)) deallocate(qm_fockmd_diis_l%FA_sv,stat=ier)
  if(associated(qm_fockmd_diis_l%FDA))   deallocate(qm_fockmd_diis_l%FDA,stat=ier)

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


  subroutine scf_iter(E_scf,H,W,Q,                   &
                      CA,DA,EA,FA,PA,CB,DB,EB,FB,PB, &
                      numat,                         &
                      dim_norbs,dim_linear_norbs,    &
                      dim_linear_fock,dim_linear_fock2, &
                      dim_scratch,ifockmd_counter,dim_iwork,         &
                      iwork,icall,UHF,q_cpmd_bomd)
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
  ! other matrices from Diis (qm_gho_info_r):
  ! FDA    : rhf or uhf-alpha Fock matrices from diis iterations.
  ! FDB    : uhf-beta Fock matrices from diis iterations.
  ! Ediis  : error matrices from different diis iterations.
  ! Bdiis  : coefficient matrix for diis linear equations.
  ! Adiis  : coefficient matrix for diis linear equations.
  ! Xdiis  : RHS vector and solutions of diis linear equations.
  !
  !use chm_kinds
  !use number
  !use qm1_constant
  use qm1_info, only : qm_control_r,qm_main_r,mm_main_r,qm_scf_main_r
  use qm1_parameters, only : CORE
  use qmmmewald_module,only : qm_ewald_prepare_fock  ! qm_ewald_add_fock,qm_ewald_correct_ee
  use mndgho_module,only : qm_gho_info_r,GHO_expansion,FTOFHB,CTRASF
  use qm1_diagonalization, only : fast_diag,evvrsp
#if KEY_PARALLEL==1
  use parallel
!#if KEY_MPI==1  /*MPI run*/
!  use mpi
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
  logical :: UHF,q_cpmd_bomd

  ! local variables
  real(chm_real),parameter :: TRANS=0.10D0,     &
                              r_three=one/three 
  character(LEN=4) :: MEXT,MFAST
  character(LEN=4) :: MXS(qm_scf_main_r%KITSCF),MFS(qm_scf_main_r%KITSCF)
  real(chm_real)   :: EDS(qm_scf_main_r%KITSCF),EES(qm_scf_main_r%KITSCF), &
                      ERS(qm_scf_main_r%KITSCF),PLS(qm_scf_main_r%KITSCF)
  !
  integer :: i,j,ii,jj,KEXT,NDIIS,NITER,nstart,nstep,info
  real(chm_real):: EEP,PL,PM,EF,EH,EFA,EFB,EHA,EHB,E_error,EDmax,EFA_L,EFB_L
  logical :: DODIIS,FASTDG
  logical :: scf_succeed,fockmd_on
  integer :: nnumnod,mmynod
#if KEY_PARALLEL==1
  !integer :: ISTRT_CHECK       ! external function
  integer,save  :: JPARPT_local(0:MAXNODE),KPARPT_local(0:MAXNODE)
!#if KEY_MPI==1
!  integer*4 :: IERR
!  real(chm_real):: Escf_local
!#endif
#endif
  integer, save :: old_N = 0, old_nqm=0
  integer, save :: mstart,mstop,msize  !,iqmst,iqmend
  real(chm_real):: t1,t2

  !!!real(chm_real),pointer,save :: empot_local(:)=>Null(),empot_all(:)=>Null()
  !

  ! for parallelization
#if KEY_PARALLEL==1
  mmynod  = mynod
  nnumnod = numnod
#else
  nnumnod = 1
  mmynod  = 0
#endif
  if(old_N .ne. dim_linear_norbs) then
     old_N  = dim_linear_norbs
     mstart = 1
     mstop  = dim_linear_norbs
     !iqmst  = 1
     !iqmend = numat
#if KEY_PARALLEL==1
     if(numnod>1) then
        !mstart = ISTRT_CHECK(mstop,dim_linear_norbs)
        JPARPT_local(0)=0
        KPARPT_local(0)=0
        do i=1,numnod
           JPARPT_local(i)= dim_linear_norbs*i/numnod ! for linear vector
           KPARPT_local(i)= numat*i/numnod
        end do
        mstart = JPARPT_local(mynod)+1
        mstop  = JPARPT_local(mynod+1)

        !iqmst  = ISTRT_CHECK(iqmend,numat)
        !iqmst  = KPARPT_local(mynod)+1
        !iqmend = KPARPT_local(mynod+1)
     end if
#endif
     msize = (mstop-mstart)+1
  end if

  ! specific for qm/mm-Ewald part.
  if(old_nqm.ne.numat) then
     old_nqm = numat
     if(mm_main_r%LQMEWD) then
        if(associated(empot_all))   deallocate(empot_all)
        if(associated(empot_local)) deallocate(empot_local)
        allocate(empot_all(numat))
        allocate(empot_local(numat))
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
  if(qm_control_r%q_diis) then
     nstart = -1  ! if diis is on, do not use damping/extrapolatin 
  else            ! in cal_density_matrix
     nstart = 4
  end if
  nstep  = 4      ! use extrapolation, it should be set negative if damping.

  ! overwrite here.
  !qm_scf_main_r%SCFCRT=1.000000000000000d-006
  !qm_scf_main_r%PLCRT =1.000000000000000d-006
  !
  !if(mm_main_r%q_cut_by_group .or. mm_main_r%q_diag_coulomb) then
  !   qm_scf_main_r%SCFCRT=1.000000000000000d-006
  !   qm_scf_main_r%PLCRT =1.000000000000000d-006
  !else
     qm_scf_main_r%SCFCRT=1.000000000000000d-006
     qm_scf_main_r%PLCRT =1.000000000000000d-006 
  !end if

  ! for Fock matrix dynamics (ref: ).
  fockmd_on=.false.
  if(qm_control_r%q_fockmd .and. qm_control_r%q_do_fockmd_scf) then
     call fock_diis(qm_fockmd_diis_r%FA_sv,qm_fockmd_diis_r%FDA,     &
                    dim_linear_norbs,qm_fockmd_diis_r%mxfdiis,msize, &
                    qm_control_r%i_fockmd_option,ifockmd_counter,fockmd_on)
     if(fockmd_on) then
        ! run a single diagonalization to determine the updated density.
        FA(mstart:mstop) = qm_fockmd_diis_r%FA_sv(1:msize)
#if KEY_PARALLEL==1
        if(nnumnod>1) call VDGBRE(FA,JPARPT_local)
#endif
        if (qm_gho_info_r%q_gho) then
           ! transform f into hb for QM link atom
           call FTOFHB(FA,qm_gho_info_r%FAHB,qm_gho_info_r%BT,    &
                       qm_gho_info_r%numat,qm_gho_info_r%nqmlnk,  &
                       dim_norbs,qm_gho_info_r%norbao,            &
                       qm_gho_info_r%lin_norbao,qm_gho_info_r%nactatm, &
                       qm_main_r%NFIRST,qm_main_r%NLAST,          &
                       qm_scf_main_r%indx)
           call square(qm_gho_info_r%FAHB,qm_gho_info_r%FAHBwrk, &
                       qm_gho_info_r%norbhb,qm_gho_info_r%norbhb, &
                       qm_gho_info_r%lin_norbhb,.false.)
           call evvrsp(qm_gho_info_r%norbhb,qm_gho_info_r%norbhb, &
                       qm_gho_info_r%norbhb,                      &
                       qm_gho_info_r%FAHBwrk,Q,IWORK,EA,qm_gho_info_r%CAHB,INFO,.false.)
           call cal_density_matrix(qm_gho_info_r%CAHB,qm_gho_info_r%DAHB,     &
                                   qm_gho_info_r%PAHB,qm_gho_info_r%FAHBwrk,PL,  &
                                   qm_gho_info_r%norbhb,                      &
                                   qm_gho_info_r%lin_norbhb,                  &
                                   qm_gho_info_r%norbhb,qm_main_r%numb,       &
                                   qm_main_r%iodd,qm_main_r%jodd,niter,kext,nstart,nstep)

           ! do the GHO-expasion.
           call GHO_expansion(qm_gho_info_r%norbhb,qm_gho_info_r%naos,     &
                              qm_gho_info_r%lin_naos,qm_gho_info_r%nqmlnk, &
                              qm_gho_info_r%lin_norbhb,dim_norbs,          &
                              dim_linear_norbs,qm_gho_info_r%mqm16,        &
                              PL,PM,PA,PA,                                 &
                              qm_gho_info_r%PAHB,qm_gho_info_r%PAHB,       &
                              qm_gho_info_r%PAOLD,qm_gho_info_r%PAOLD,     &
                              qm_gho_info_r%QMATMQ,qm_gho_info_r%BT,qm_gho_info_r%BTM, &
                              qm_scf_main_r%indx,UHF)
        else
           call square(FA,qm_scf_main_r%FAwork,qm_main_r%norbs,dim_norbs,dim_linear_norbs,.false.)
           call evvrsp(qm_main_r%norbs,qm_main_r%norbs,dim_norbs,           &
                       qm_scf_main_r%FAwork,Q,IWORK,EA,CA,INFO,.false.)
           call cal_density_matrix(CA,DA,PA,qm_scf_main_r%FAwork,PL,     &
                                   dim_norbs,dim_linear_norbs,           &
                                   qm_main_r%norbs,qm_main_r%numb,       &
                                   qm_main_r%iodd,qm_main_r%jodd,niter,kext,nstart,nstep)
        end if

        ! now, lower the scf criteria.
        qm_scf_main_r%SCFCRT=1.000000000000000d-006
        qm_scf_main_r%PLCRT =1.000000000000000d-006
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
     call fockx(FA,PA,PB,Q,W,dim_linear_norbs,dim_linear_fock,UHF,     &
                numat,qm_main_r%nfirst,qm_main_r%nlast,qm_main_r%num_orbs, &
                qm_scf_main_r%NW)
     !exit Scfloop

     ! QM/MM-Ewald
     if(mm_main_r%LQMEWD) then
        ! Q = PA + PB (UHF) or 2*PA (constructed in fockx). 
        call calc_mulliken(numat,qm_main_r%nat,qm_main_r%nfirst,qm_main_r%num_orbs, &
                           Q,mm_main_r%qm_charges,                                  &
#if KEY_PARALLEL==1
                           nnumnod,                       & 
#endif
                           dim_linear_fock)
        ! compute Ewald correction potential on qm atom site and mofidy Fock matrix. 
        ! Since Eslf(nquant,nquant) matrix is used, it is only need to do matrix
        ! multiplication to get the correction terms from QM images to be SCF iterated.
        call qm_ewald_prepare_fock(numat,empot_all,empot_local,mm_main_r%qm_charges)

        ! now correct the fock matrix.
        call qm_ewald_add_fock(numat,qm_main_r%nfirst,qm_main_r%num_orbs,  &
                               qm_scf_main_r%indx,FA,mm_main_r%qm_charges, &
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

        call fockx(FB,PB,PA,Q,W,dim_linear_norbs,dim_linear_fock,UHF,     &
                   numat,qm_main_r%nfirst,qm_main_r%nlast,qm_main_r%num_orbs, &
                   qm_scf_main_r%NW)

        ! QM/MM-Ewald
        if(mm_main_r%LQMEWD) then
           ! As the following done above.. no need to do here.
           !call calc_mulliken(numat,qm_main_r%nat,qm_main_r%nfirst,qm_main_r%num_orbs, &
           !                   Q,mm_main_r%qm_charges,                                  &
           !#if KEY_PARALLEL==1
           !                   nnumnod,                       &
           !#endif
           !                   dim_linear_fock)
           !
           ! As the following done above.. no need to do here.
           ! compute Ewald correction potential on qm atom site and mofidy Fock matrix. 
           ! Since Eslf(nquant,nquant) matrix is used, it is only need to do matrix
           ! multiplication to get the correction terms from QM images to be SCF iterated.
           !call qm_ewald_prepare_fock(numat,empot_all,empot_local,mm_main_r%qm_charges)

           ! now correct the fock matrix.
           call qm_ewald_add_fock(numat,qm_main_r%nfirst,qm_main_r%num_orbs,  &
                                  qm_scf_main_r%indx,FB,mm_main_r%qm_charges, &
                                  dim_linear_norbs) 
        end if
#if KEY_PARALLEL==1
        if(nnumnod>1) call VDGBRE(FB,JPARPT_local)
#endif
     end if         ! (UHF)

     !====================START OPENMP PARALLEL========================!
!$omp parallel NUM_THREADS(2)
!$omp sections
!$omp section
     ! Energy calculation.
     ! Note that there were two escf call in the original code, one with F=H copy (core-Hamiltonian),
     ! and the other one with full F. Here, in this implementation, the two separate calls are
     ! merged to a single energy call.
     ! This is done here, becaue the energy calls are done with F and H (and not with FAHB).
     EFA = escf(dim_norbs,PA,FA,H,dim_linear_norbs)
     if(UHF) then
        EFB = escf(dim_norbs,PB,FB,H,dim_linear_norbs)
     else
        EFB = EFA    ! computed for complete Fock-matrix (here) and H-core matrix.
     end if
     !------------------------END SECTION 1----------------------------!

!$omp section
     ! In case of using QM/MM-Ewald, MM atom contributes in full.
     EFA_L = zero
     EFB_L = zero
     if(mm_main_r%LQMEWD) then
        ! empot_local is already copied above, qm_ewald_prepare_fock.
        EFA_L = qm_ewald_correct_ee(numat,qm_main_r%nfirst,qm_main_r%num_orbs, &
                                      qm_scf_main_r%indx,PA)
        if(UHF) then
           EFB_L = qm_ewald_correct_ee(numat,qm_main_r%nfirst,qm_main_r%num_orbs, &
                                      qm_scf_main_r%indx,PB)
        else
           EFB_L = EFA_L
        end if
     end if


     ! For GHO.
     if (qm_gho_info_r%q_gho) then
        ! only need to do before exit: converged or scf iteraction exceeded.
        if(kext.eq.-1 .or. niter.ge.qm_scf_main_r%KITSCF) then
          ! store Fock matrix in AO basis for derivative
          qm_gho_info_r%FAOA(1:dim_linear_norbs)=FA(1:dim_linear_norbs)
        end if

        ! transform f into hb for QM link atom
        call FTOFHB(FA,qm_gho_info_r%FAHB,qm_gho_info_r%BT,    &
                    qm_gho_info_r%numat,qm_gho_info_r%nqmlnk,  &
                    dim_norbs,qm_gho_info_r%norbao,            &
                    qm_gho_info_r%lin_norbao,qm_gho_info_r%nactatm, &
                    qm_main_r%NFIRST,qm_main_r%NLAST,          &
                    qm_scf_main_r%indx)

        if(UHF) then
           ! only need to do before exit: converged or scf iteraction exceeded.
           ! either converged or scf interation exceeded.
           if(kext.eq.-1 .or. niter.ge.qm_scf_main_r%KITSCF) then
              qm_gho_info_r%FAOB(1:dim_linear_norbs)=FB(1:dim_linear_norbs)
           end if

           ! for GHO, transform beta Fock matrix to hybrid basis
           call FTOFHB(FB,qm_gho_info_r%FBHB,qm_gho_info_r%BT,    &
                       qm_gho_info_r%numat,qm_gho_info_r%nqmlnk,  &
                       dim_norbs,qm_gho_info_r%norbao,            &
                       qm_gho_info_r%lin_norbao,qm_gho_info_r%nactatm, &
                       qm_main_r%nfirst,qm_main_r%nlast,          &
                       qm_scf_main_r%indx)
        end if
     end if
     !-------------------------END SECTION 2---------------------------!
!$omp end sections
!$omp end parallel
     !======================END OPENMP PARALLEL========================!

     ! apply Diis convergence acceleration.
     dodiis =(qm_control_r%q_diis .and. niter.ge.1 .and. kext.ne.-1)
     if(dodiis) then ! if(dodiis .and. .not.(qm_control_r%q_dxl_bomd .and. qm_control_r%q_do_dxl_scf)) then
       if (qm_gho_info_r%q_gho) then
         ! GHO-DIIS extrapolation
         if(UHF) then
            call diis(qm_gho_info_r%FAHB,qm_gho_info_r%FBHB,             &
                      qm_gho_info_r%PAOLD,qm_gho_info_r%PBOLD,           &
                      qm_scf_main_r%FAwork,qm_scf_main_r%FBwork,         &
                      qm_scf_main_r%PAwork,qm_scf_main_r%PBwork,         &
                      qm_scf_diis_r%FDA,qm_scf_diis_r%FDB,               &
                      qm_scf_diis_r%Ediis,                               &
                      qm_scf_diis_r%Adiis,qm_scf_diis_r%Bdiis,           &
                      qm_scf_diis_r%Xdiis,                               &
                      qm_gho_info_r%norbhb,qm_gho_info_r%lin_norbhb,     &
                      qm_gho_info_r%norbhb,NDIIS,qm_scf_diis_r%mxdiis,   &
                      qm_scf_diis_r%iwork_diis,EDMAX,.true.,UHF)
         else
            call diis(qm_gho_info_r%FAHB,qm_gho_info_r%FAHB,             &
                      qm_gho_info_r%PAOLD,qm_gho_info_r%PAOLD,           &
                      qm_scf_main_r%FAwork,qm_scf_main_r%FAwork,         &
                      qm_scf_main_r%PAwork,qm_scf_main_r%PAwork,         &
                      qm_scf_diis_r%FDA,qm_scf_diis_r%FDA,               &
                      qm_scf_diis_r%Ediis,                               &
                      qm_scf_diis_r%Adiis,qm_scf_diis_r%Bdiis,           &
                      qm_scf_diis_r%Xdiis,                               &
                      qm_gho_info_r%norbhb,qm_gho_info_r%lin_norbhb,     &
                      qm_gho_info_r%norbhb,NDIIS,qm_scf_diis_r%mxdiis,   &
                      qm_scf_diis_r%iwork_diis,EDMAX,.true.,UHF)
         end if
       else
         ! normal diis extrapolation.
         if(UHF) then
            call diis(FA,FB,PA,PB,                               &
                      qm_scf_main_r%FAwork,qm_scf_main_r%FBwork, &
                      qm_scf_main_r%PAwork,qm_scf_main_r%PBwork, &
                      qm_scf_diis_r%FDA,qm_scf_diis_r%FDB,       &
                      qm_scf_diis_r%Ediis,                       &
                      qm_scf_diis_r%Adiis,qm_scf_diis_r%Bdiis,   &
                      qm_scf_diis_r%Xdiis,                       &
                      dim_norbs,dim_linear_norbs,                &
                      qm_main_r%NORBS,NDIIS,qm_scf_diis_r%mxdiis,   &
                      qm_scf_diis_r%iwork_diis,EDMAX,.true.,UHF)
         else
            call diis(FA,FB,PA,PB,                               &
                      qm_scf_main_r%FAwork,qm_scf_main_r%FAwork, &
                      qm_scf_main_r%PAwork,qm_scf_main_r%PAwork, &   
                      qm_scf_diis_r%FDA,qm_scf_diis_r%FDA,       &
                      qm_scf_diis_r%Ediis,                       &
                      qm_scf_diis_r%Adiis,qm_scf_diis_r%Bdiis,   &
                      qm_scf_diis_r%Xdiis,                       &
                      dim_norbs,dim_linear_norbs,                &
                      qm_main_r%NORBS,NDIIS,qm_scf_diis_r%mxdiis,   &
                      qm_scf_diis_r%iwork_diis,EDMAX,.true.,UHF)
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
     !if(mynod==0) write(6,*) niter,E_scf

     ! Check energy related information.
     ! save scf information
     if(niter.gt.0 .and. niter.le.qm_scf_main_r%KITSCF) then
        if(dodiis) then
           MEXT  = 'DIIS'
        else if(kext.ge.2) then
           MEXT  = 'DAMP'
        else if(kext.eq.1) then
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
     if(niter.gt.0) then
        if(kext.eq.-1) exit Scfloop  ! exit main iteration loop
        if(abs(E_error).lt.qm_scf_main_r%SCFCRT .and. PL.lt.qm_scf_main_r%PLCRT .and. kext.ne.1) then
           !ITSAVE= niter+1  ! itsave may not be used in the current
                             ! implementation.
           kext   =-1        ! converged, and exit at next step.
        else
           ! so, meaning, not yet scf converged.
           if(niter.gt.qm_scf_main_r%KITSCF) then
              ! Scf failed by exceeding scf cycle; Exit the main loop.
              scf_succeed=.false.
              exit Scfloop
           end if

           ! for DXL-BOMD
           if(qm_control_r%q_dxl_bomd .and. qm_control_r%q_do_dxl_scf .and. qm_control_r%N_scf_step == niter) then
              kext   =-1        ! converged, and exit at next step.
           end if
           ! FASTDG on, if PL < TRANS (=0.10d0) .and. niter < kitscf-50.
           if(fockmd_on) then
              FASTDG =(PL.lt.TRANS .and. (niter.ge.1))
           else
              FASTDG =(PL.lt.TRANS .and. (niter.gt.1))
           end if
           if(niter.gt.qm_scf_main_r%KITSCF-50) FASTDG =.false. ! turn off is niter approaches kitscf 
        end if
     end if

     ! for now, consider it is not parallelized.
     ! diagonalize the F-matrix.
     ! info: error code.
     if(FASTDG) then
        if(qm_gho_info_r%q_gho) then
           call square(qm_gho_info_r%FAHB,qm_gho_info_r%FAHBwrk,     &
                       qm_gho_info_r%norbhb,qm_gho_info_r%norbhb, &
                       qm_gho_info_r%lin_norbhb,.false.) !.true.)
           call fast_diag(qm_gho_info_r%FAHBwrk,qm_gho_info_r%CAHB,EA,Q, &   ! qm_gho_info_r%FAHBwrk,
                          qm_gho_info_r%norbhb,qm_gho_info_r%norbhb,qm_main_r%numb)
        else
           call square(FA,qm_scf_main_r%FAwork,qm_main_r%norbs,dim_norbs,dim_linear_norbs,.false.) !.true.)
           call fast_diag(qm_scf_main_r%FAwork,CA,EA,Q,dim_norbs, &          ! qm_scf_main_r%FAwork,
                          qm_main_r%norbs,qm_main_r%numb)
        end if
        if(UHF) then
           if(qm_gho_info_r%q_gho) then
              call square(qm_gho_info_r%FBHB,qm_gho_info_r%FBHBwrk,     &
                          qm_gho_info_r%norbhb,qm_gho_info_r%norbhb, &
                          qm_gho_info_r%lin_norbhb,.false.) !.true.)
              call fast_diag(qm_gho_info_r%FBHBwrk,qm_gho_info_r%CBHB,EB,Q, &  ! qm_gho_info_r%FBHBwrk,
                             qm_gho_info_r%norbhb,qm_gho_info_r%norbhb,qm_main_r%nbeta)
           else
              call square(FB,qm_scf_main_r%FBwork,qm_main_r%norbs,dim_norbs,dim_linear_norbs,.false.) !.true.)
              call fast_diag(qm_scf_main_r%FBwork,CB,EB,Q,dim_norbs, &         ! qm_scf_main_r%FBwork,
                             qm_main_r%norbs,qm_main_r%nbeta)
           end if
        end if
        !!if(mynod==0) write(6,'(A17,F12.5)')'Time (FAST-DIAG)=',(t2-t1)*1000.0d0
     else
        ! When do the full diagonalization, instead of call diagon, here we 
        ! call explicitly square and evvrsp, which do the job of diagon when
        ! IDIAG=0 (default diagonalizer).
        ! refer DIAGON.f and full_diagonalization.f
        !
        if(qm_gho_info_r%q_gho) then
           call square(qm_gho_info_r%FAHB,qm_gho_info_r%FAHBwrk, &
                       qm_gho_info_r%norbhb,qm_gho_info_r%norbhb, &
                       qm_gho_info_r%lin_norbhb,.false.)
           call evvrsp(qm_gho_info_r%norbhb,qm_gho_info_r%norbhb, &
                       qm_gho_info_r%norbhb,                      &
                       qm_gho_info_r%FAHBwrk,Q,IWORK,EA,qm_gho_info_r%CAHB,INFO,.false.)
        else
           call square(FA,qm_scf_main_r%FAwork,qm_main_r%norbs,dim_norbs,dim_linear_norbs,.false.)
           call evvrsp(qm_main_r%norbs,qm_main_r%norbs,dim_norbs,           &
                       qm_scf_main_r%FAwork,Q,IWORK,EA,CA,INFO,.false.)
        end if
        !!if(mynod==0) write(6,'(A17,F12.5)')'Time (FULL-DIAG)=',(t2-t1)*1000.0d0
        ! Error section:
        if(info.ne.0 .and. prnlev.ge.2) write(6,400) info
        !
        if(UHF) then
           if(qm_gho_info_r%q_gho) then
              call square(qm_gho_info_r%FBHB,qm_gho_info_r%FBHBwrk, &
                          qm_gho_info_r%norbhb,qm_gho_info_r%norbhb, &
                          qm_gho_info_r%lin_norbhb,.false.)
              call evvrsp(qm_gho_info_r%norbhb,qm_gho_info_r%norbhb, &
                          qm_gho_info_r%norbhb,                      &
                          qm_gho_info_r%FBHBwrk,Q,IWORK,EB,qm_gho_info_r%CBHB,INFO,.false.)
           else
              call square(FB,qm_scf_main_r%FBwork,qm_main_r%norbs,dim_norbs,dim_linear_norbs,.false.)
              call evvrsp(qm_main_r%norbs,qm_main_r%norbs,dim_norbs,           &
                          qm_scf_main_r%FBwork,Q,IWORK,EB,CB,INFO,.false.)
           end if
           ! Error section:
           if(info.ne.0 .and. prnlev.ge.2) write(6,405) info
        end if
        ! error
        if(info.ne.0) then
           if(ndiis.gt.0) ndiis=ndiis-1
           if(prnlev.ge.2) write(6,410)
           Cycle Scfloop     ! main iteration do loop
        end if
     end if
     niter  = niter+1

     ! COMPUTE THE DENSITY MATRIX AND EXTRAPOLATE, IF POSSIBLE.
     ! for GHO:
     if (qm_gho_info_r%q_gho) then
        if(UHF) then
           call cal_density_matrix(qm_gho_info_r%CAHB,qm_gho_info_r%DAHB,     &
                                   qm_gho_info_r%PAHB,qm_gho_info_r%FAHBwrk,PL,  &
                                   qm_gho_info_r%norbhb,                      &
                                   qm_gho_info_r%lin_norbhb,                  &
                                   qm_gho_info_r%norbhb,qm_main_r%NALPHA,     &
                                   0,0, niter,kext,nstart,nstep)
           call cal_density_matrix(qm_gho_info_r%CBHB,qm_gho_info_r%DBHB,     &
                                   qm_gho_info_r%PBHB,qm_gho_info_r%FBHBwrk,PM,  &
                                   qm_gho_info_r%norbhb,                      &
                                   qm_gho_info_r%lin_norbhb,                  &
                                   qm_gho_info_r%norbhb,qm_main_r%NBETA ,     &
                                   0,0,niter,kext,nstart,nstep)
           if(PM.GT.PL) PL=PM
 
           ! do the GHO expansion
           call GHO_expansion(qm_gho_info_r%norbhb,qm_gho_info_r%naos,     &
                              qm_gho_info_r%lin_naos,qm_gho_info_r%nqmlnk, &
                              qm_gho_info_r%lin_norbhb,dim_norbs,          &
                              dim_linear_norbs,qm_gho_info_r%mqm16,        &
                              PL,PM,PA,PB,                                 &
                              qm_gho_info_r%PAHB,qm_gho_info_r%PBHB,       &
                              qm_gho_info_r%PAOLD,qm_gho_info_r%PBOLD,     &
                              qm_gho_info_r%QMATMQ,qm_gho_info_r%BT,qm_gho_info_r%BTM, &
                              qm_scf_main_r%indx,UHF)
        else
           call cal_density_matrix(qm_gho_info_r%CAHB,qm_gho_info_r%DAHB,     &
                                   qm_gho_info_r%PAHB,qm_gho_info_r%FAHBwrk,PL,  &
                                   qm_gho_info_r%norbhb,                      &
                                   qm_gho_info_r%lin_norbhb,                  &
                                   qm_gho_info_r%norbhb,qm_main_r%numb,       &
                                   qm_main_r%iodd,qm_main_r%jodd,niter,kext,nstart,nstep)

           ! do the GHO-expasion.
           call GHO_expansion(qm_gho_info_r%norbhb,qm_gho_info_r%naos,     &
                              qm_gho_info_r%lin_naos,qm_gho_info_r%nqmlnk, &
                              qm_gho_info_r%lin_norbhb,dim_norbs,          &
                              dim_linear_norbs,qm_gho_info_r%mqm16,        &
                              PL,PM,PA,PA,                                 &
                              qm_gho_info_r%PAHB,qm_gho_info_r%PAHB,       &
                              qm_gho_info_r%PAOLD,qm_gho_info_r%PAOLD,     &
                              qm_gho_info_r%QMATMQ,qm_gho_info_r%BT,qm_gho_info_r%BTM, &
                              qm_scf_main_r%indx,UHF)
        end if
     else          ! q_gho
        if(UHF) then
           call cal_density_matrix(CA,DA,PA,qm_scf_main_r%FAwork,PL,     &
                                   dim_norbs,dim_linear_norbs,           &
                                   qm_main_r%norbs,qm_main_r%nalpha,     &
                                   0,0,niter,kext,nstart,nstep)
           call cal_density_matrix(CB,DB,PB,qm_scf_main_r%FAwork,PM,     &
                                   dim_norbs,dim_linear_norbs,           &
                                   qm_main_r%norbs,qm_main_r%nbeta ,     &
                                   0,0,niter,kext,nstart,nstep)
           if(PM.GT.PL) PL=PM
        else
           call cal_density_matrix(CA,DA,PA,qm_scf_main_r%FAwork,PL,     &
                                   dim_norbs,dim_linear_norbs,           &
                                   qm_main_r%norbs,qm_main_r%numb,       &
                                   qm_main_r%iodd,qm_main_r%jodd,niter,kext,nstart,nstep)
        end if
     end if        ! q_gho
     !!if(mynod==0) write(6,'(A17,F12.5)')'Time (DENSITY  )=',(t2-t1)*1000.0d0
  End do Scfloop   ! main iteration loop
  ! END OF SCF LOOP.
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  ! for Fock matrix dynamics (ref: ).
  if(qm_control_r%q_fockmd .and. qm_control_r%md_run) then
     ! save it for the next md step.
     qm_fockmd_diis_r%FA_sv(1:msize) = FA(mstart:mstop)
  end if
  !if(mynod==0) write(6,*) 'SCF Final: ',niter,E_scf

  ! Specific line for CPMD-CURVy-MTS-LN.
  ! q_cpmd_bomd =.true.: scf_iter is called to determined density only.
  !                      in this case, after reaching the scf, the density
  !                      does not have to be copied to old one...
  !                      in particular, GHO case..
  if(.not.q_cpmd_bomd) then
     ! for GHO method: since CA is not explicitly used within the scf cycle. So,
     ! transform orbitals to AO basis here, used by Mulliken analysis.
     if(qm_gho_info_r%q_gho) then
        ! this CTRASF appears not to be necessary.
        call CTRASF(qm_main_r%norbs,qm_gho_info_r%nqmlnk,qm_gho_info_r%mqm16, &
                    qm_gho_info_r%norbhb,qm_gho_info_r%naos,                  &
                    qm_gho_info_r%BT,qm_gho_info_r%CAHB,CA)
        if(UHF) then
           ! for UHF:
           call CTRASF(qm_main_r%norbs,qm_gho_info_r%nqmlnk,qm_gho_info_r%mqm16, &
                       qm_gho_info_r%norbhb,qm_gho_info_r%naos,                  &
                       qm_gho_info_r%BT,qm_gho_info_r%CBHB,CB)
           ! store the density for gho-related derivative calculation.
           do i=1,dim_linear_norbs
              qm_gho_info_r%PHO(i) = qm_gho_info_r%PAHB(i)
              qm_gho_info_r%PBHO(i)= qm_gho_info_r%PBHB(i)
           end do
        else
           ! for RHF: it only needs to be done here, not in the GHO_expansion.
           ! store the density for gho-related derivative calculation (density in HB N+4 basis).
           qm_gho_info_r%PHO(1:dim_linear_norbs)=qm_gho_info_r%PAHB(1:dim_linear_norbs)*two
        end if
     end if

     ! special for cpmd preparation.
     if(qm_control_r%cpmd .and. qm_control_r%md_run) then
        ! PAHB is broken after GHO_expansion. So, we should reconstruct it.
        if(qm_gho_info_r%q_gho) then
           ii = qm_gho_info_r%lin_norbhb
           qm_gho_info_r%PAHB(1:ii)=qm_gho_info_r%PAOLD(1:ii)
           if(UHF) qm_gho_info_r%PBHB(1:ii)=qm_gho_info_r%PBOLD(1:ii)
        end if
     end if
  end if
  ! Scf convergence has been achieved
  if(scf_succeed) then
     ! if successful at 2nd attempt for scf using Diis.
     if(icall.eq.-1 .and. qm_control_r%q_diis) then 
        if(prnlev.ge.2) write(6,580)
        icall = 0
     end if
  else       ! scf, failed?
     ! no scf convergence.
     if(prnlev.ge.2) then
        write(6,560) 'SCF_ITER routine'
        write(6,520)
        if(qm_control_r%q_diis) then
           write(6,505)
           do i=1,MIN(niter,qm_scf_main_r%KITSCF)
              write(6,510) i,EES(i),ERS(i),PLS(i),MXS(i),MFS(i),EDS(i)
           end do
           write(6,530) NITER,E_scf,E_error,PL,EDMAX
        else
           write(6,500)
           do i=1,MIN(niter,qm_scf_main_r%KITSCF)
              write(6,510) i,EES(i),ERS(i),PLS(i),MXS(i),MFS(i)
           end do
           write(6,530) NITER,E_scf,E_error,PL
        end if
        write(6,540) qm_scf_main_r%SCFCRT,qm_scf_main_r%PLCRT
     end if
     icall  = -1  !  indicates failure to reach SCF convergence.
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

  ! contains
  !====================================================================!
  !=====================beginning of contains =========================!
  !
  !========================= end of contains ==========================!
  !====================================================================!

  end subroutine scf_iter


  subroutine calc_mulliken(numat,nat,nfirst,num_orbs,Q,mul_chg,  &
#if KEY_PARALLEL==1
                           nnumnod,    &
#endif
                           dim_linear_fock)
  !
  ! Calculates the Mulliken charges on QM atoms from QM/MM calculations 
  !
  ! Variables:
  !   i       : a number from 1 to numat
  !   mul_chg : Mulliken charges returned, electron units
  !
  use chm_kinds
  use number,only : zero
  use qm1_parameters,only : CORE
  use parallel, only : mynod,MAXNODE

  implicit none
  !
  integer, intent(in) :: numat
#if KEY_PARALLEL==1
  integer, intent(in) :: nnumnod
#endif
  integer, intent(in) :: nat(numat),nfirst(numat),num_orbs(numat)
  integer, intent(in) :: dim_linear_fock
  real(chm_real),intent(in) :: Q(*)
  real(chm_real),intent(out):: mul_chg(numat)

  ! local variables
  integer :: i,j,ia,KL,KL2,iorbs
  real(chm_real) :: ch_local
  integer, save :: old_N = 0
  integer, save :: mstart,mstop
#if KEY_PARALLEL==1
  integer,save  :: JPARPT_local(0:MAXNODE)
#endif

  if(old_N .ne. numat) then
     old_N = numat
     mstart= 1
     mstop = numat
     !
#if KEY_PARALLEL==1
     JPARPT_local(0)=0
     do i=1,nnumnod
        JPARPT_local(i)= numat*i/nnumnod ! for linear vector
     end do
     mstart = JPARPT_local(mynod)+1
     mstop  = JPARPT_local(mynod+1)
#endif
  end if

  KL = 1
#if KEY_PARALLEL==1
  do i=1,mstart-1
     iorbs= num_orbs(i)
     KL   = KL + indx_local(iorbs) + iorbs
  end do
#endif
  do i=mstart,mstop
     iorbs= num_orbs(i)
     ia  = NFIRST(i)
     KL2 = KL
     ch_local = Q(KL2)
     if(iorbs.ge.9) then
        do j=1,8
           KL2    = KL + indx_local(j+1) + j
           ch_local  = ch_local + Q(KL2)
        end do
     else if(iorbs.ge.4) then
        do j=1,3
           KL2    = KL + indx_local(j+1) + j
           ch_local  = ch_local + Q(KL2)
        end do
     end if
     ! core is taken cared together (see also fockx routine). 
     mul_chg(i) = CORE(nat(i)) - ch_local
     KL         = KL + indx_local(iorbs) + iorbs
  end do
#if KEY_PARALLEL==1
  if(nnumnod>1) call VDGBRE(mul_chg,JPARPT_local)
#endif

  return
  end subroutine calc_mulliken


  real(chm_real) function escf(N,P,F,H,dim_linear_norbs)
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
#if KEY_PARALLEL==1
  integer       :: ISTRT_CHECK    ! external function
#endif
  real(chm_real):: ddot1d_mn ! external function

  ! local variables
  integer :: KMAX,I,J,IORBS,jmax
  real(chm_real):: EE,EEDIAG
  !
  integer,save :: old_N = 0
  integer,save :: kstart,kstop,mstart,mstop,nnumnod

  !
  kmax = indx_local(n)+n  ! (N*(N+1))/2
#if KEY_PARALLEL==1
  nnumnod = numnod
#else
  nnumnod = 1
#endif

  if(old_N .ne. dim_linear_norbs) then
     old_N = dim_linear_norbs
     kstart= 1
     kstop = kmax
     mstart= 1
     mstop = n
#if KEY_PARALLEL==1
     if(nnumnod>1) then    ! nnumnod defined above
        kstart = ISTRT_CHECK(kstop,kmax)
        mstart = ISTRT_CHECK(mstop,n)
     end if
#endif
  end if

  !EE = DOT_PRODUCT(P(kstart:kstop),F(kstart:kstop)) + DOT_PRODUCT(P(kstart:kstop),H(kstart:kstop))
  EE   = ddot1d_mn(kstop-kstart+1,P(kstart:kstop),1,F(kstart:kstop),H(kstart:kstop),1,.true.)

  ! for diagonal term:
  EEdiag = zero
#if KEY_PARALLEL==1
  j = (mstart*(mstart-1))/2
#else
  j = 0
#endif
  do i=mstart,mstop   ! 1,n
     j      = j+i   !  = indx_local(i)+i = (i*(i+1))/2
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
#if KEY_PARALLEL==1
  integer,save         :: old_N = 0
  logical,save,pointer :: q_do_atom(:)=>Null()
  logical,save  :: q_first_ewd=.true.
#endif

  ! parallelization is synchronized with calc_mulliken routine
  !                    to avoid broad casting.
#if KEY_PARALLEL==1
  nnumnod = numnod
  istart  = mynod+1
#else
  nnumnod = 1
  istart  = 1
#endif

#if KEY_PARALLEL==1
  if(old_N .ne. numat) then
     old_N = numat
     if(associated(q_do_atom)) deallocate(q_do_atom)
     allocate(q_do_atom(numat))

     q_first_ewd=.true.
     call fill_q_do_atom(q_first_ewd)
  end if
#endif
  

  ! add Empot+Eslf contribution to the diagonal elements of the fock matrix
  do i = istart, numat
#if KEY_PARALLEL==1
     if(q_do_atom(i)) then
#endif
        ia = nfirst(i)
        LL = indx(ia)+ia

        ewdpot = empot_all(i)*ev_a0   ! conversion here.
        iorbs  = num_orbs(i)
        fock_matrix(LL)= fock_matrix(LL)-ewdpot
        if(iorbs.ge.9) then
           do j=1,8
              LL             = INDX(j+ia)+j+ia
              fock_matrix(LL)= fock_matrix(LL)-ewdpot
           end do
        else if(iorbs.ge.4) then
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

     if(.not.q_ewd) return

     fstart = dim_linear_norbs*(mynod)  /numnod + 1
     fstop  = dim_linear_norbs*(mynod+1)/numnod

     loopII: do i = istart, numat
        q_do_atom(i) =.false.
        ia = nfirst(i)
        LL = indx(ia)+ia
        if(LL>=fstart .and. LL<=fstop) then
           q_do_atom(i) =.true.
           cycle loopII
        end if
        iorbs  = num_orbs(i)
        if(iorbs.ge.9) then
           do j=1,8
              LL             = INDX(j+ia)+j+ia
              if(LL>=fstart .and. LL<=fstop) then
                 q_do_atom(i) =.true.
                cycle loopII
              end if
           end do
        else if(iorbs.ge.4) then
           do j=1,3
              LL             = INDX(j+ia)+j+ia
              if(LL>=fstart .and. LL<=fstop) then
                 q_do_atom(i) =.true.
                 cycle loopII
              end if
           end do
        end if
     end do loopII
     q_ewd =.false.
     return
     end subroutine fill_q_do_atom
#endif
  END SUBROUTINE qm_ewald_add_fock


  real(chm_real) function qm_ewald_correct_ee(numat,nfirst,num_orbs,indx,p)
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
  real(chm_real),intent(in) :: P(*)
  !!!real(chm_real),intent(in) :: empot_local(numat)

  ! Local variables
  integer        :: i,i1,ia,ib,i2,iorbs,j,LL
  real(chm_real) :: etemp
  real(chm_real),parameter :: ev_a0 = EV*A0 ! convert (electrons/angstrom) to (eV/Bohr)

#if KEY_PARALLEL==1
  integer :: ISTRT_CHECK           ! for external function
#endif
  !
  integer,save :: old_N = 0
  integer,save :: mstart,mstop

  ! for parallelization
  if(old_N .ne. numat) then
     old_N =numat
     mstart=1
     mstop =numat
#if KEY_PARALLEL==1
     if(numnod>1) mstart = ISTRT_CHECK(mstop,numat)
  !else
  !   if(QMPI) then
  !      mstart=1
  !      mstop =numat
  !   end if
#endif
  end if

  etemp    = zero
  do i = mstart,mstop                     ! 1, numat
     ia   = nfirst(i)
     LL   = indx(ia)+ia
     iorbs= num_orbs(i)
     etemp= etemp-empot_local(i)*P(LL)
     if(iorbs.ge.9) then
        do j=1,8
           LL    = indx(j+ia)+j+ia
           etemp = etemp -empot_local(i)*P(LL)
        end do
     else if(iorbs.ge.4) then
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
                                IODD,JODD,NITER,KEXT,NSTART,NSTEP)
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
  !use chm_kinds 
  !use qm1_info, only : qm_main_r,qm_scf_main_r
  !use number
  !use qm1_constant
#if KEY_PARALLEL==1
  use parallel
!#if KEY_MNDOOPENMP==1  /*OpenMP specific*/
!  use omp_lib
!#endif                 /*OpenMP specific*/
#endif

  implicit none
  !
  integer:: dim_norbs,dim_linear_norbs
  integer:: N,NOCC,IODD,JODD,NITER,KEXT,NSTART,NSTEP
  real(chm_real):: C(dim_norbs,dim_norbs),D(dim_linear_norbs), &
                   P(dim_linear_norbs),PL, &
                   PN(dim_linear_norbs) ! PN(dim_norbs*dim_norbs)

  ! local variables
  integer :: I,J,K,II,IJ
  real(chm_real):: YLAMB,DEN1,DEN2,DKPI,DKI,A,FAC,c_tmp
  real(chm_real), save :: YL
  real(chm_real) :: YCRIT=0.06D0
  !
  integer :: nnumnod,mmynod,id,nb2
  integer, save :: old_N = 0
  integer, save :: mstart,mstop,istart,istop
  real(chm_real),save,allocatable,dimension(:) :: pn_diag
#if KEY_PARALLEL==1
  integer :: ISTRT_CHECK       ! external function
  integer,save  :: JPARPT_local(0:MAXNODE)
#endif

  ! for parallelization
#if KEY_PARALLEL==1
  mmynod  = mynod
  nnumnod = numnod
#else
  nnumnod = 1
#endif
  if(old_N .ne. dim_linear_norbs) then
     old_N  = dim_linear_norbs
     mstart = 1
     mstop  = nocc
#if KEY_PARALLEL==1
     !if(nnumnod>1) mstart = ISTRT_CHECK(mstop,nocc)
     JPARPT_local(0)=0
     do i=1,nnumnod
        JPARPT_local(i)= nocc*i/nnumnod ! for linear vector
     end do
     mstart = JPARPT_local(mynod)+1
     mstop  = JPARPT_local(mynod+1)
#endif
  end if


  ! so, if diis is on, 
  if(NSTART.LT.0) then  ! meaning, Diis is on.
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
     if(allocated(pn_diag) .and. (size(pn_diag) < n)) deallocate(pn_diag)
     if(.not.allocated(pn_diag)) allocate(pn_diag(n))
     ii = 0
     do i=1,n
        ii = ii + i
        pn_diag(i) = p(ii)
     end do

#if KEY_MNDOOPENMP==1  /*OpenMP specific*/
     nb2 = (mstop-mstart+1)/2 + mstart
!$omp parallel private(k,ij,i,c_tmp,id) NUM_THREADS(2)
!$omp do
     do i=1,dim_linear_norbs
        p(i) = zero
        pn(i)= zero
     end do
!$omp end do
!$omp barrier
     id = OMP_get_thread_num()
     if(id == 0) then
        do k=mstart,nb2 ! mstart,mstop              ! 1,nocc
           ij=0
           do i=1,n
              c_tmp=c(i,k)
              p(ij+1:ij+i)=p(ij+1:ij+i)+c_tmp*c(1:i,k)
              ij=ij+i
           end do
        end do
     else
        do k=nb2+1,mstop ! mstart,mstop              ! 1,nocc
           ij=0
           do i=1,n
              c_tmp=c(i,k)
              pn(ij+1:ij+i)=pn(ij+1:ij+i)+c_tmp*c(1:i,k)
              ij=ij+i
           end do
        end do
     end if
!$omp barrier
!$omp do
     do i=1,dim_linear_norbs
        p(i) = p(i) + pn(i)
     end do
!$omp end do
!$omp end parallel

#else  /*OpenMP specific*/
     p(1:dim_linear_norbs) = zero
     do k=mstart,mstop              ! 1,nocc
        ij=0
        do i=1,n
           c_tmp=c(i,k)
           p(ij+1:ij+i)=p(ij+1:ij+i)+c_tmp*c(1:i,k)
           ij=ij+i
        end do
     end do
#endif /*OpenMP specific*/

     ! as cal_density_matrix is called mostly with iodd=0, jodd=0. So, do not worry much now.
     ! this part is of concern for UHF or multiplicity larger than 1.
     if(iodd.gt.0) then
        if(iodd >= mstart .and. iodd <=mstop) then
           ij  = 0
           do i=1,n
              c_tmp=PT5*C(i,iodd)
              P(ij+1:ij+i)=P(ij+1:ij+i)-c_tmp*C(1:i,iodd)
              ij=ij+i
           end do
        end if
        if(jodd.gt.0) then
           if(jodd >= mstart .and. jodd <=mstop) then
             ij  = 0
             do i=1,n
                c_tmp=PT5*C(i,jodd)
                P(ij+1:ij+i)=P(ij+1:ij+i)-c_tmp*C(1:i,jodd)
                ij=ij+i
             end do
           end if
        end if
     end if

#if KEY_PARALLEL==1
     if(nnumnod>1) call gcomb(p(1:dim_linear_norbs),dim_linear_norbs)
#endif

     ! calculate maximum change in diagonal matrix elements.
     if(niter.ge.0 .and. nstart.lt.0) then
        PL  = zero
        ii  = 0
        do i=1,n
           ii  = ii + i  ! =indx_local(i)+i  ! I*(I+1)/2
           !if(ABS(P(ii)-pn_diag(i)).gt.PL) PL=ABS(P(ii)-pn_diag(i))
           PL = max(ABS(P(ii)-pn_diag(i)),PL)
        end do
     end if

  else
     ! full calculation allowing for extrapolation or damping. the new density
     ! matrix P(i) and the difference density matrix D(i) are determined along
     ! with some auxiliary variables.
     !
     ! initialization:
     if(niter.eq.1) then
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
        ij=0
        do i=1,n
           c_tmp=c(i,k)
           pn(ij+1:ij+i)=pn(ij+1:ij+i)+c_tmp*c(1:i,k)
           ij=ij+i
        end do
     end do

     ! iodd=0; jodd=0 for default
     if(iodd.gt.0) then
        if(iodd >= mstart .and. iodd <=mstop) then
           ij  = 0
           do i=1,n
              c_tmp=PT5*C(i,iodd)
              PN(ij+1:ij+i)=PN(ij+1:ij+i)-c_tmp*C(1:i,iodd)
              ij=ij+i
           end do
        end if
        if(jodd.gt.0) then
          if(jodd >= mstart .and. jodd <=mstop) then
             ij  = 0
             do i=1,n
                c_tmp=PT5*C(i,jodd)
                PN(ij+1:ij+i)=PN(ij+1:ij+i)-c_tmp*C(1:i,jodd)
                ij=ij+i
             end do
          end if
        end if
     end if
#if KEY_PARALLEL==1
     if(nnumnod>1) call gcomb(pn(1:dim_linear_norbs),dim_linear_norbs)
#endif

     ! downgraded (not parallized).
     do ij=1,dim_linear_norbs
        DKPI   = PN(ij)-P(ij)
        DKI    = D(ij)
        DEN1   = DEN1 +DKI *DKI
        DEN2   = DEN2 +DKPI*DKPI
        YLAMB  = YLAMB+DKI*DKPI
        D(ij)  = DKPI
        P(ij)  = PN(ij)
     end do

     ! calculate maximum change in diagonal matrix elements.
     ii = 0
     do i=1,n
        ii     = ii + i  ! =indx_local(i)+i  ! I*(I+1)/2
        PL     = MAX(abs(D(ii)),PL)
     end do

     ! check for exrapolation or damping.
     ! P(i)  holds the new density matrix.
     ! D(i)  holds the difference between the new and old density matrix.
     !
     ! return if Kext is negative (see argument list).
     if(kext.lt.0) return   ! meaning, it is converged density.

     kext   = 0
     if(nstep.gt.0) then
        ! for extrapolation
        if(niter.ge.nstart) then
           ii  = niter-nstart
           ii  = ii-(ii/nstep)*nstep
           if(ii.eq.0) kext=1
        end if
        if(niter.lt.2 .or. den1.eq.zero .or. den2.eq.zero) then
           kext = 0
        else
           YLAMB = YLAMB/DEN1
           if(ABS(YLAMB).ge.one) YLAMB=YLAMB*DEN1/DEN2
           if(ABS(YLAMB-YL).gt.YCRIT) kext=0
           if(YLAMB.eq.one) kext=0
           YL = YLAMB
        end if
        if(kext.eq.1) FAC = one/(one-YLAMB)-one
     else if(nstep.lt.0) then
        ! for damping
        if(niter.ge.nstart .and. niter.gt.1) then
           kext = 2
           fac  = DBLE(MIN(-nstep,9))*PT1   ! /10.0D0
        end if
     end if
     ! update of density matrix 
     if(kext.gt.0) P(1:dim_linear_norbs)=P(1:dim_linear_norbs)+fac*D(1:dim_linear_norbs)
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
  use qm1_info, only : qm_control_r,qm_main_r
  use qm1_parameters,only : CORE
  use stream,only : prnlev

  implicit none
  integer :: numat,dim_norbs,dim_linear
  real(chm_real):: PA(dim_linear)         ! P_alpha density linear matrix

  integer :: i,j,ia,ib,ja,jb,ii,ij,jj,n_size,iunit
  real(chm_real):: x_tmp,y_tmp
  real(chm_real),pointer:: qm_charge(:)=>NUll(), &
                           qmqm_bond(:)=>Null(), &
                           PA_sq(:,:)=>Null()

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
     ia = qm_main_r%nfirst(i)
     ib = qm_main_r%nlast(i)
     do j=1,i
        ij = ij + 1
        ja = qm_main_r%nfirst(j)
        jb = qm_main_r%nlast(j)
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
     qm_charge(i)  =-y_tmp*two+CORE(qm_main_r%nat(i))
  end do

  ! now printing..
  iunit = qm_control_r%ianal_unit
  if(prnlev >= 2) then
     ! bond order and mulliken charges
     write(iunit,120)
     do i=1,numat
        ia = i*(i-1)/2 + 1
        ib = ia + i - 1
        ii = ib - ia + 1
        if(ii<=25) then
           write(iunit,250) qm_control_r%qminb(i),qm_charge(i),(qmqm_bond(ij),ij=ia,ib)
        else 
           jj = ia + 25 - 1
           write(iunit,250) qm_control_r%qminb(i),qm_charge(i),(qmqm_bond(ij),ij=ia,jj)
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
  if(associated(qm_charge)) deallocate(qm_charge)
  if(associated(qmqm_bond)) deallocate(qmqm_bond)
  if(associated(pa_sq))     deallocate(pa_sq)

  return
  end subroutine bond_analysis


  subroutine fockx(F,PA,PB,Q,W,LM4,LM6,UHF,numat,nfirst,nlast,num_orbs,NW)
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
  !use chm_kinds
  !use number
  !use qm1_constant
  !use qm1_info, only : qm_main_r 
#if KEY_PARALLEL==1
  use parallel 
!#if KEY_MNDOOPENMP==1  /*OpenMP specific*/
!  use omp_lib
!#endif                 /*OpenMP specific*/
#endif

  implicit none

  integer :: LM4,LM6,numat
  real(chm_real):: F(LM4),PA(LM4),PB(LM4),Q(LM6),W(LM6,LM6)
  logical :: UHF
  integer :: nfirst(numat),nlast(numat),num_orbs(numat),NW(numat)

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
  integer, save :: old_N = 0
  integer, save :: mstart,mstop,fstart,fstop
#if KEY_PARALLEL==1
  !integer       :: ISTRT_CHECK           ! external function
  integer,save  :: JPARPT_local(0:MAXNODE),KPARPT_local(0:MAXNODE)
  logical,save,pointer :: q_mynod_fock(:)=>Null()
  logical,save  :: q_first_fock=.true.
#endif
  real(chm_real):: ddot_mn ! external function
  integer :: tid,nb2,fst1,fst2,fsta,ist,ift,iskip

  ! for parallelization
#if KEY_PARALLEL==1
  mmynod = mynod
  nnumnod= numnod
#else
  nnumnod= 1
  mmynod = 0
#endif
  if(old_N .ne. LM6) then
     old_N  = LM6
     mstart = 1
     mstop  = LM6

     fstart = 1
     fstop  = LM4
#if KEY_PARALLEL==1
     JPARPT_local(0)=0
     do i=1,nnumnod
        JPARPT_local(i)= LM6*i/nnumnod ! for linear vector
     end do
     mstart = JPARPT_local(mynod)+1
     mstop  = JPARPT_local(mynod+1)

     KPARPT_local(0)=0
     do i=1,nnumnod
        KPARPT_local(i)= LM4*i/nnumnod ! for linear vector
     end do
     fstart = KPARPT_local(mynod)+1
     fstop  = KPARPT_local(mynod+1)

     if(associated(q_mynod_fock)) deallocate(q_mynod_fock)
     allocate(q_mynod_fock(numat*(numat+1)/2))
#endif
  end if


  ! Coulomb contributions:
  ! one-center exchange contributions are implicitly included for RHF.
  !
  ! Note: when use parallel, Q contains only terms between mstart and mstop. 
  !       Then, it is broadcasted. (Should be synchronized with calc_mulliken routine.)
  !
  ! LM6: dim_linear_fock: one center AO pairs (see determine_qm_scf_arrray_size)
  !      so this is much smaller than fock size.
  if(UHF) then
     do KL=mstart,mstop  ! 1,LM6
        ! Q(KL)  = (PA(IP_local(KL))+PB(IP_local(KL)))
        ! if(IP1_local(KL).ne.IP2_local(KL)) Q(KL)=Q(KL)*TWO
        if(ip_check(kl)) then
           Q(KL)= two*(PA(IP_local(KL))+PB(IP_local(KL)))
        else
           Q(KL)= (PA(IP_local(KL))+PB(IP_local(KL)))
        end if
     end do
  else
     do KL=mstart,mstop  ! 1,LM6
        ! Q(KL)  = two*PA(IP_local(KL))
        ! if(IP1_local(KL).ne.IP2_local(KL)) Q(KL)=Q(KL)*TWO
        ! see below in define_pair_index for ip_check defintion.
        if(ip_check(kl)) then
           Q(KL)= two*(PA(IP_local(KL))+PA(IP_local(KL)))
        else
           Q(KL)= (PA(IP_local(KL))+PA(IP_local(KL)))
        end if
     end do
  end if
#if KEY_PARALLEL==1
  if(nnumnod>1) call VDGBRE(Q,JPARPT_local)
#endif
  !
#if KEY_PARALLEL==1
  ! fill the q_mynod_fock array.
  call fill_q_mynod_fock(q_first_fock)
#endif

  ! for serial
  !F(ip_local(1:lm6))=F(ip_local(1:lm6))+MATMUL(Q(1:LM6),W(1:LM6,1:LM6))
  !
  ! now with parallel
#if KEY_MNDOOPENMP==1  /*OpenMP specific*/
  nb2 = ((fstop-fstart)+1)/2
  fst1= fstart+nb2
  fst2= fst1+1
  fsta= numat/2 - 1
!$omp parallel private(ij,ik,tid) NUM_THREADS(2)
  tid = OMP_get_thread_num()
  if(tid==0) then
     do ij=1,LM6
        ik = ip_local(ij)
        !if(ik>=fstart .and. ik<=fstop) then
        if(ik>=fstart .and. ik<=fst1) then
!           F(ik) = F(ik) + ddot_mn(LM6,Q(1:LM6),1,W(1:LM6,ij),1)
           F(ik) = F(ik) + DOT_PRODUCT(Q(1:LM6),W(1:LM6,ij))
        end if
     end do
  else
     do ij=1,LM6
        ik = ip_local(ij)
        !if(ik>=fstart .and. ik<=fstop) then
        if(ik>=fst2 .and. ik<=fstop) then
           !F(ik) = F(ik) + ddot_mn(LM6,Q(1:LM6),1,W(1:LM6,ij),1)
           F(ik) = F(ik) + DOT_PRODUCT(Q(1:LM6),W(1:LM6,ij))
        end if
     end do
  end if
!$omp end parallel
#else  /*OpenMP specific*/
  !do ij=1,LM6
  !   F(ip_local(ij))=F(ip_local(ij)) + ddot_mn(mstop-mstart+1,Q(mstart:mstop),1,W(mstart:mstop,ij),1)
  !end do
  do ij=1,LM6
     ik = ip_local(ij)
     if(ik>=fstart .and. ik<=fstop) then
        F(ik) = F(ik) + ddot_mn(LM6,Q(1:LM6),1,W(1:LM6,ij),1)
     end if
  end do
#endif /*OpenMP specific*/

  ! two-center exchange contibutions: offdiagonal two-center terms (ij,kl).
#if KEY_MNDOOPENMP==1  /*OpenMP specific*/
!$omp parallel NUM_THREADS(2) &
!$omp & private(iicnt,tid,i,k,ii,jj,ia,ib,ic,iorbs,iw,ij,ja,jb,jorbs,jw,KL,is,ix,iy,iz,ijs,kls)  &
!$omp & private(PA_tmp,F_tmp,ks,kx,ky,kz,ijk,ka,ijw,klw,ik,sum,sum2,temp,temp2,iL,jL,L,jk,ist,iskip)
  tid = OMP_get_thread_num()
  if(tid==0) then
     ist   = 2
     iskip = 2
  else
     ist   = 3
     iskip = 2
  end if
#else  /*OpenMP specific*/
  ist   = 2
  iskip = 1
#endif /*OpenMP specific*/
  !
  loopII: do ii=ist,numat,iskip  ! 1,NUMAT
     iicnt  = ((ii-1)*(ii-2))/2
     ia     = NFIRST(ii)
     ib     = NLAST(ii)
     ic     = INDX_local(ia)
     iorbs  = num_orbs(ii)
     iw     = INDX_local(iorbs)+iorbs
     ij     = NW(ii)-1
     loopJJ: do jj=1,ii-1
        ja     = NFIRST(jj)
        jb     = NLAST(jj)
        jorbs  = num_orbs(jj)
        jw     = INDX_local(jorbs)+jorbs
        KL     = NW(jj)-1
#if KEY_PARALLEL==1
        iicnt  = iicnt + 1
        if(.not.q_mynod_fock(iicnt)) cycle loopJJ
#endif
        if(iw.eq.1 .and. jw.eq.1) then
           is   = ic+ja
           F(is)= F(is)-PA(is)*W(ij+1,KL+1)
        else if(iw.eq.1 .and. jw.eq.10) then
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
        else if(iw.eq.10 .and. jw.eq.1) then
           is  = ic+ja
           ix  = INDX_local(ia+1)+ja
           iy  = INDX_local(ia+2)+ja
           iz  = INDX_local(ia+3)+ja
           kls = KL+1
           PA_tmp(1)= PA(is)
           PA_tmp(2)= PA(ix)
           PA_tmp(3)= PA(iy)
           PA_tmp(4)= PA(iz)
           F(is) = F(is) - (PA_tmp(1)*W(ij+1,kls)+PA_tmp(2)*W(ij+2,kls)+PA_tmp(3)*W(ij+4,kls)+PA_tmp(4)*W(ij+ 7,kls))
           F(ix) = F(ix) - (PA_tmp(1)*W(ij+2,kls)+PA_tmp(2)*W(ij+3,kls)+PA_tmp(3)*W(ij+5,kls)+PA_tmp(4)*W(ij+ 8,kls))
           F(iy) = F(iy) - (PA_tmp(1)*W(ij+4,kls)+PA_tmp(2)*W(ij+5,kls)+PA_tmp(3)*W(ij+6,kls)+PA_tmp(4)*W(ij+ 9,kls))
           F(iz) = F(iz) - (PA_tmp(1)*W(ij+7,kls)+PA_tmp(2)*W(ij+8,kls)+PA_tmp(3)*W(ij+9,kls)+PA_tmp(4)*W(ij+10,kls))
        else if(iw.eq.10 .and. jw.eq.10) then
           do i=1,4
              is  = INDX_local(ia+i-1)+ja
              ix  = is+1
              iy  = is+2
              iz  = is+3
              F_tmp(1:4)=zero
              do k=1,4
                 ks  = INDX_local(ia+k-1)+ja
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
              ka  = INDX_local(i)
              ijw = ij+INDX_local(i-ia+2)
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
              ka = INDX_local(i)
              do j=ia,i-1
                 kb  = INDX_local(j)
                 ijw = ij+INDX_local(i-ia+1)+j-ia+1
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
#if KEY_MNDOOPENMP==1  /*OpenMP specific*/
!$omp end parallel
#endif /*OpenMP specific*/

  ! one-center exchange contributions for UHF.
  ! offdiagonal one-center terms (ii,kk) for SP-basis.
  if(UHF) then
     do i=mmynod+1,NUMAT, nnumnod            ! if not parallel, mmynod=0,nnumnod=1
        ia     = NFIRST(I)
        iorbs  = num_orbs(i)
        if(iorbs.eq.4) then
           ij     = NW(i)-1
           ixs    = INDX_local(ia+1)+ia
           iys    = INDX_local(ia+2)+ia
           izs    = INDX_local(ia+3)+ia
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
     do ij=mstart,mstop  ! 1,LM6
        i2    =IP_local(ij)
        i     = IP1_local(ij)
        j     = IP2_local(ij)
        temp  = W(ij,ij)

        F(i2) = F(i2) - PA(i2)*temp
        if(i.ne.j) then
           ii    = INDX_local(i)+i
           jj    = INDX_local(j)+j
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
        if(iorbs.gt.4) then
           klmin  = JX_local(n)
           klmax  = klmin+iorbs*iorbs-1
           kstart = JP1_local(klmin)
           loopKL: do kl=klmin,klmax
#if KEY_PARALLEL==1
              iicnt  = iicnt + 1
              if(mmynod .ne. mod(iicnt-1,nnumnod)) cycle loopKL
#endif
              !
              k      = JP1_local(kl)
              l      = JP2_local(kl)
              ! only elements F(ik) wiht i.ge.k are needed. therefore, 
              ! the loop over ij can start at ijmin.ge.klmin.
              ijmin  = klmin+iorbs*(k-kstart)
              j2     = JP3_local(kl)
              loopIJ: do ij=ijmin,klmax
                 ! diagonal terms have been included above also for an SPD-basis.
                 if(JP3_local(ij) .ne. j2) then
                    wijkl  = W(JP3_local(ij),j2)
                    if(wijkl.ne.zero) then
                       i   = JP1_local(ij)
                       j   = JP2_local(ij)
                       ik  = INDX_local(i)+k
                       jl  = INDX_local(MAX(j,l))+MIN(j,l)
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

     if(.not.q_fock) return

     ! two-center exchange contibutions: offdiagonal two-center terms (ij,kl).
     ! this routine loops over and check if elements belongs mynod (fstart <= ii <=fstop)
     iicnt  = 0
     loopII: do ii=1,NUMAT
        ia     = NFIRST(ii)
        ib     = NLAST(ii)
        ic     = INDX_local(ia)
        iorbs  = num_orbs(ii)
        iw     = INDX_local(iorbs)+iorbs
        loopJJ: do jj=1,ii-1
           ja     = NFIRST(jj)
           jb     = NLAST(jj)
           jorbs  = num_orbs(jj)
           jw     = INDX_local(jorbs)+jorbs
           iicnt  = iicnt + 1
           q_mynod_fock(iicnt) =.false.
           if(iw.eq.1 .and. jw.eq.1) then
              is   = ic+ja
              if(is>=fstart .and. is<=fstop) q_mynod_fock(iicnt) =.true.
           else if(iw.eq.1 .and. jw.eq.10) then
              is  = ic+ja
              ix  = is+1
              iy  = is+2
              iz  = is+3
              if( (is>=fstart .and. is<=fstop) .or. (ix>=fstart .and. ix<=fstop) .or. &
                  (iy>=fstart .and. iy<=fstop) .or. (iz>=fstart .and. iz<=fstop)) then
                 q_mynod_fock(iicnt) =.true.
              end if
           else if(iw.eq.10 .and. jw.eq.1) then
              is  = ic+ja
              ix  = INDX_local(ia+1)+ja
              iy  = INDX_local(ia+2)+ja
              iz  = INDX_local(ia+3)+ja
              if( (is>=fstart .and. is<=fstop) .or. (ix>=fstart .and. ix<=fstop) .or. &
                  (iy>=fstart .and. iy<=fstop) .or. (iz>=fstart .and. iz<=fstop)) then
                 q_mynod_fock(iicnt) =.true.
              end if
           else if(iw.eq.10 .and. jw.eq.10) then
              do i=1,4
                 is  = INDX_local(ia+i-1)+ja
                 ix  = is+1
                 iy  = is+2
                 iz  = is+3
                 if( (is>=fstart .and. is<=fstop) .or. (ix>=fstart .and. ix<=fstop) .or. &
                     (iy>=fstart .and. iy<=fstop) .or. (iz>=fstart .and. iz<=fstop)) then
                    q_mynod_fock(iicnt) =.true.
                    cycle loopJJ
                 end if
              end do
           else
              ! General code   - also valid for D-orbitals.
              ! contributions from (ii,kl)
              loopI1: do i=ia,ib
                 ka  = INDX_local(i)
                 do k=ja,jb
                    ik   = ka+k
                    if(ik>=fstart .and. ik<=fstop) then
                       q_mynod_fock(iicnt) =.true.
                       cycle loopJJ
                    end if
                    do l=ja,k-1
                       il    = ka+l
                       if(il>=fstart .and. il<=fstop) then
                          q_mynod_fock(iicnt) =.true.
                          cycle loopJJ
                       end if
                    end do
                 end do
              end do loopI1
   
              ! contribution from (ij,kl) with i.ne.j
              loopI2: do i=ia+1,ib
                 ka = INDX_local(i)
                 do j=ia,i-1
                    kb  = INDX_local(j)
                    do k=ja,jb
                       ik    = ka+k
                       jk    = kb+k
                       if((ik>=fstart .and. ik<=fstop) .or. (jk>=fstart .and. jk<=fstop)) then
                          q_mynod_fock(iicnt) =.true.
                          cycle loopJJ
                       end if
                       do l=ja,k-1
                          il   = ka+l
                          jl   = kb+l
                          if((il>=fstart .and. il<=fstop) .or. (jl>=fstart .and. jl<=fstop)) then
                             q_mynod_fock(iicnt) =.true.
                             cycle loopJJ
                          end if
                       end do
                    end do
                 end do
              end do loopI2
           end if
        end do loopJJ
     end do loopII

     q_fock =.false.
     return
     end subroutine fill_q_mynod_fock
#endif
  end subroutine fockx


  subroutine q_construct(PA,PB,Q,LM4,LM6,UHF)
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
  logical :: UHF

  ! local variables
  integer :: KL,i
  integer, save :: old_N = 0
  integer, save :: mstart,mstop
#if KEY_PARALLEL==1
  !integer       :: ISTRT_CHECK           ! external function
  integer,save  :: JPARPT_local(0:MAXNODE)
#endif
  real(chm_real),parameter ::ev_a0=EV*A0

  ! for parallelization
  if(old_N .ne. LM6) then
     old_N  = LM6
#if KEY_PARALLEL==1
     JPARPT_local(0)=0
     do i=1,numnod
        JPARPT_local(i)= LM6*i/numnod ! for linear vector
     end do
     mstart = JPARPT_local(mynod)+1
     mstop  = JPARPT_local(mynod+1)
#else
     mstart = 1
     mstop  = LM6
#endif
  end if

  ! Note: when use parallel, Q contains only terms between mstart and mstop. 
  !       So care should be given in particular at calc_mulliken subroutine.
#if KEY_PARALLEL==1
  if(numnod>1) Q(1:LM6) = zero
#endif
  if(UHF) then
     do KL=mstart,mstop  ! 1,LM6
        if(ip_check(kl)) then
           Q(KL)= two*(PA(IP_local(KL))+PB(IP_local(KL)))
        else
           Q(KL)= (PA(IP_local(KL))+PB(IP_local(KL)))
        end if
     end do
  else
     do KL=mstart,mstop  ! 1,LM6
        if(ip_check(kl)) then
           Q(KL)= two*(PA(IP_local(KL))+PA(IP_local(KL)))
        else
           Q(KL)= (PA(IP_local(KL))+PA(IP_local(KL)))
        end if
     end do
  end if
#if KEY_PARALLEL==1
  if(numnod>1) call VDGBRE(Q,JPARPT_local)
#endif
  return
  end subroutine q_construct


  subroutine diis(FA,FB,PA,PB,FAwork,FBwork,PAwork,PBwork,            &
                  FDA,FDB,Ediis,Adiis,Bdiis,Xdiis,                    &
                  dim_norbs,dim_linear_norbs,N,NDIIS,mxdiis,          &
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
  !use chm_kinds
  !use qm1_info, only : qm_scf_main_r
  !use number, only : zero,one,two
#if KEY_PARALLEL==1
  use parallel
!#if KEY_MNDOOPENMP==1  /*OpenMP specific*/
!  use omp_lib
!#endif                 /*OpenMP specific*/
#if KEY_MPI==1  /*MPI run*/
  use mpi
#endif /*MPI run*/
#endif

  implicit none
  !
  integer :: dim_norbs,dim_linear_norbs,N,NDIIS,mxdiis
  real(chm_real):: FA(dim_linear_norbs),PA(dim_linear_norbs), &
                   FB(dim_linear_norbs),PB(dim_linear_norbs)
  real(chm_real):: FAwork(dim_norbs,dim_norbs),PAwork(dim_norbs,dim_norbs), &
                   FBwork(dim_norbs,dim_norbs),PBwork(dim_norbs,dim_norbs)
  real(chm_real):: FDA(dim_linear_norbs,*),FDB(dim_linear_norbs,*), &
                   Ediis(dim_linear_norbs,*),Adiis(*),Bdiis(*),Xdiis(*)
  real(chm_real):: EDMAX
  integer       :: iwork_diis(*) 
  logical       :: qprint,UHF

  ! local variables
  integer :: i,j,k,M,ij,KD,NEW,MX1,KX1P,KDIIS,IEDMAX,info
  real(chm_real):: CX,cx1,aa,bb
  real(chm_real):: Ediis_kd(dim_linear_norbs)
  real(chm_real),parameter :: SCALE=1.02D0
  integer       :: idamax_mn           ! external function
  real(chm_real):: ddot_mn,ddot2d_mn   ! external function
  real(chm_real),save,pointer :: bdiis_local(:)=>Null()

  integer, save :: old_N = 0
  integer, save :: mstart,mstop,iidim,iistart,iiend
#if KEY_PARALLEL==1
  integer :: ISTRT_CHECK       ! external function
  integer,save  :: JPARPT_local(0:MAXNODE),KPARPT_local(0:MAXNODE)
  real(chm_real):: edmax_local
  logical,       save,pointer :: q_ij_pair(:)=>Null()
#if KEY_MPI==1
  integer*4 :: IERR
#endif
#endif
  integer :: nnumnod,mmynod
  real(chm_real):: s_aux
  integer :: tid,nb2,mst1,mst2

  ! for parallelization
#if KEY_PARALLEL==1
  nnumnod = numnod
  mmynod  = mynod
#else
  nnumnod = 1
  mmynod  = 0
#endif
  if(old_N .ne. dim_linear_norbs) then
     old_N  = dim_linear_norbs
     mstart = 1
     mstop  = dim_linear_norbs
     iistart= 1
     iiend  = n
#if KEY_PARALLEL==1
     if(nnumnod>1) then
        !mstart = ISTRT_CHECK(mstop,dim_linear_norbs)
        iistart= ISTRT_CHECK(iiend,n)

        ! Prepare array for vector allgather calls using VDGBRE
        ! mapping for each node (Hard weird).
        JPARPT_local(0)=0
        do i=1,nnumnod
           JPARPT_local(i)= dim_linear_norbs*i/nnumnod ! for linear vector
        end do
        mstart = JPARPT_local(mynod)+1
        mstop  = JPARPT_local(mynod+1)
     end if
     if(associated(bdiis_local)) deallocate(bdiis_local)
     allocate(bdiis_local(mxdiis))

     ! for logical array.
     if(associated(q_ij_pair)) deallocate(q_ij_pair)
     allocate(q_ij_pair(n))
     ij=0
     q_ij_pair(1:n)=.false.
     do i=1,n
        do j=1,i
           ij = ij + 1
           if(ij >= mstart .and. ij <=mstop) then
              q_ij_pair(i) =.true.
              q_ij_pair(j) =.true.
           end if
        end do
     end do
#endif
     iidim  = mstop - mstart + 1  ! dimension
  end if

  ! initialization.
  ndiis  = ndiis+1
  kdiis  = MOD(ndiis-1,mxdiis) + 1

  ! Compute the error matrix ED = F*P - P*F, using linearly packed matrices.
  ! in principle, Ework(1:n,1:n)= MATMUL(FAwork(1:n,1:n),PAwork(1:n,1:n)) -MATMUL(PAwork(1:n,1:n),FAwork(1:n,1:n))
  ! after each matrix has been squared form.
  !
  call square2(FA,FAwork,PA,PAwork,N,dim_norbs,dim_linear_norbs &
#if KEY_PARALLEL==1
              ,q_ij_pair  &
#endif
               )
  !
  ! using a lower diagonal part, C:=A*B-B*A, where FA and PA are symmetric matrices.
  !Ediis_kd(mstart:mstop) = zero
#if KEY_MNDOOPENMP==1  /*OpenMP specific*/
  nb2 = ((mstop-mstart)+1)/2
  mst1= mstart + nb2
  mst2= mst1   + 1
!$omp parallel private(i,j,ij,tid) NUM_THREADS(2)
  tid = OMP_get_thread_num()
  if(tid==0) then 
     do i=1,n
        do j=1,i
           ij = i*(i-1)/2 + j
           if(ij >= mstart .and. ij <= mst1) then
              ! .false. do minus.
!              Ediis_kd(ij)= ddot2d_mn(n,FAwork(1:n,j),PAwork(1:n,j),1,PAwork(1:n,i),FAwork(1:n,i),1,.false.)
               Ediis_kd(ij)=DOT_PRODUCT(FAwork(1:n,j),PAwork(1:n,i)) - DOT_PRODUCT(PAwork(1:n,j),FAwork(1:n,i))
           end if
        end do
     end do
  else
     do i=1,n
        do j=1,i
           ij = i*(i-1)/2 + j
           if(ij >= mst2 .and. ij <= mstop) then
              ! .false. do minus.
!              Ediis_kd(ij)= ddot2d_mn(n,FAwork(1:n,j),PAwork(1:n,j),1,PAwork(1:n,i),FAwork(1:n,i),1,.false.)
               Ediis_kd(ij)=DOT_PRODUCT(FAwork(1:n,j),PAwork(1:n,i)) - DOT_PRODUCT(PAwork(1:n,j),FAwork(1:n,i))
           end if
        end do
     end do
  end if
!$omp end parallel
#else  /*OpenMP specific*/
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
           Ediis_kd(ij)= ddot2d_mn(n,FAwork(1:n,j),PAwork(1:n,j),1,PAwork(1:n,i),FAwork(1:n,i),1,.false.) 
        end if
     end do
  end do
#endif /*OpenMP specific*/

  if(UHF) then
     call square2(FB,FBwork,PB,PBwork,N,dim_norbs,dim_linear_norbs &
#if KEY_PARALLEL==1
                 ,q_ij_pair  &
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
              Ediis_kd(ij)=Ediis_kd(ij)+ddot2d_mn(n,FBwork(1:n,j),PBwork(1:n,j),1,PBwork(1:n,i),FBwork(1:n,i),1,.false.)
           end if
        end do
     end do
  end if

!!#if KEY_PARALLEL==1
!!  ! broadcast.
!!  if(nnumnod>1) call VDGBRE(Ediis_kd(1:dim_linear_norbs),JPARPT_local)
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
  if(kdiis.eq.ndiis) then
     mx1 = kdiis+1
  else
     mx1 = mxdiis+1  ! case of overflow, ndiis > mxdiis
  end if
  kd        = (kdiis+1)*kdiis/2 + 1  ! =INDX_local(kdiis+1)+1
  Bdiis(1)  = zero
  Bdiis(kd) =-one
  Ediis(mstart:mstop,kdiis)=Ediis_kd(mstart:mstop)
#if KEY_PARALLEL==1
  if(nnumnod>1) then
!$omp parallel do private(i) NUM_THREADS(2)
     do i=1,mx1-1
        bdiis_local(i)=-two*ddot_mn(mstop-mstart+1,Ediis(mstart:mstop,i),1,Ediis_kd(mstart:mstop),1)
     end do
!omp end parallel do
     call gcomb(bdiis_local(1:mx1-1),mx1-1) 
     do i=1,mx1-1
        if(kdiis.ge.i) then
           new = kd+i
        else
           new = (i+1)*i/2 + 1 + kdiis ! =INDX_local(I+1)+1+kdiis
        end if
        Bdiis(new)=bdiis_local(i)
     end do
  else
#endif
     do i=1,mx1-1
        if(kdiis.ge.i) then
           new = kd+i
        else
           new = (i+1)*i/2 + 1 + kdiis ! =INDX_local(I+1)+1+kdiis
        end if
        Bdiis(new)=-two*ddot_mn(dim_linear_norbs,Ediis(1:dim_linear_norbs,i),1,Ediis_kd(1:dim_linear_norbs),1)
     end do
#if KEY_PARALLEL==1
  end if
#endif
  !if(jdiis.lt.2) &
  Bdiis(kd+kdiis)=Bdiis(kd+kdiis)*SCALE

  ! now, do the diis-extrapolation
  if(ndiis.ne.1) then
     ! copy coefficients from BD(mx1p) to AD(mx1p) & scratch array will be
     ! overwritten by dspsv_mn.
     kx1p         = mx1*(mx1-1)/2+mx1 ! =INDX_local(mx1)+mx1
     Adiis(1:kx1p)= Bdiis(1:kx1p)

     ! define rhs vector of linear equations.
     Xdiis(1)    =-one
     Xdiis(2:mx1)= zero

     ! solve the system of linear equations.
     call dspsv_mn('U',mx1,1,Adiis,iwork_diis,Xdiis,mx1,info)
     if(info.ne.0) then
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
           call VDGBRE(FA,JPARPT_local)
           call VDGBRE(FB,JPARPT_local)
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
#if KEY_MNDOOPENMP==1  /*OpenMP specific*/
        nb2 = ((mstop-mstart)+1)/2 + mstart
        cx1 = Xdiis(2)
!$omp parallel private(m,cx,tid) NUM_THREADS(2)
        tid = OMP_get_thread_num()
        if(tid==0) then
           FA(mstart:nb2) = cx1*FDA(mstart:nb2,1)
           if(mx1>=3) then
              do m=3,mx1
                 cx     = Xdiis(m)
                 FA(mstart:nb2) = FA(mstart:nb2) + cx*FDA(mstart:nb2,m-1)
              end do
           end if
        else
           FA(nb2+1:mstop) = cx1*FDA(nb2+1:mstop,1)
           if(mx1>=3) then
              do m=3,mx1
                 cx     = Xdiis(m)
                 FA(nb2+1:mstop) = FA(nb2+1:mstop) + cx*FDA(nb2+1:mstop,m-1)
              end do
           end if
        end if
!$omp end parallel
#else  /*OpenMP specific*/
        !
        cx = Xdiis(2)
        FA(mstart:mstop) = cx*FDA(mstart:mstop,1)
        if(mx1>=3) then
           do m=3,mx1
              cx     = Xdiis(m)
              FA(mstart:mstop) = FA(mstart:mstop) + cx*FDA(mstart:mstop,m-1)
           end do
        end if
#endif /*OpenMP specific*/
#if KEY_PARALLEL==1
        ! broadcast here, as FA was a complete matrix before this routine.
        if(nnumnod>1) call VDGBRE(FA,JPARPT_local)
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
                       ifock_option,ifockmd_counter,fockmd_on)
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
                       

  integer, save :: old_N = 0
  integer, save :: mstart,mstop
  integer :: nnumnod,mmynod

  ! for parallelization
#if KEY_PARALLEL==1
  nnumnod = numnod
  mmynod  = mynod
#else
  nnumnod = 1
  mmynod  = 0
#endif
  if(old_N .ne. dim_linear_norbs) then
     old_N  = dim_linear_norbs
     mstart = 1
     mstop  = dim_linear_norbs
#if KEY_PARALLEL==1
     if(nnumnod>1) then
        mstart = dim_linear_norbs*(mynod)/nnumnod + 1
        mstop  = dim_linear_norbs*(mynod+1)/nnumnod
     end if
#endif
  end if

  ! now, do the fock extrapolation.
  if(ifockmd_counter >= (mxfdiis_local+1)) then
     if(mxfdiis_local==5) then
        if(ifock_option == 1) then
!$omp parallel do private(i,j,fval,alpha_1,alpha_2,f_n2p,f_n2m,f_n3p,df_n2p,df_n2m) NUM_THREADS(2)
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
!$omp end parallel do

        else
!$omp parallel do private(i,j,fval,alpha_1,alpha_2,f_n2p,f_n2m,f_n3p,df_n2p,df_n2m) NUM_THREADS(2)
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
!$omp end parallel do
        end if
     else if(mxfdiis_local==7) then
        if(ifock_option == 1) then
!$omp parallel do private(i,j,fval,alpha_1,alpha_2,alpha_3,alpha_4,f_n3p,f_n3m,f_n4p,df_n3p,df_n3m) NUM_THREADS(2)
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
!$omp end parallel do

        else
!$omp parallel do private(i,j,fval,alpha_1,alpha_2,alpha_3,alpha_4,f_n3p,f_n3m,df_n3p,df_n3m,dt2an3p) NUM_THREADS(2)
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
!$omp end parallel do
        end if
     else
          call wrndie(-1,'<FOCK DIIS>','Wrong extrapolation order.')
     end if
     fockmd_on=.true.
  else
     ! only copy...
     ! n-2,n-1,n,n+1,n+2 data points.
!$omp parallel do private(i,j) NUM_THREADS(2)
     do i=1,msize_local  ! mstart,mstop
        do j=1,mxfdiis_local-1
           FDA(j,i) = FDA(j+1,i)
        end do
        FDA(mxfdiis_local,i)    = FA(i)        ! copy a new value.
     end do
!$omp end parallel do
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

  use qm1_info, only : qm_main_r, qm_scf_main_r

  ! local variables
  integer :: i,j,ii,k,ia,ib,id,nwii

  ! set memory allocation and check.
  call allocate_pair_index(qm_scf_main_r%dim_numat,qm_main_r%uhf)

  ! copy local copy.
  indx_local(1:qm_main_r%norbs)=qm_scf_main_r%INDX(1:qm_main_r%norbs)

  ! define pair indices and pair factors
  if(.not. qm_main_r%uhf) then
     ! Coulomb part.
     k      = 0
     do ii=1,qm_main_r%NUMAT
        qm_scf_main_r%NW(ii) = k+1    ! lower triangle of a given block.
        ia     = qm_main_r%NFIRST(ii) ! at
        ib     = qm_main_r%NLAST(ii)
        do i=ia,ib
           id     = indx_local(i) ! qm_scf_main_r%INDX(i)
           do j=ia,i
              k      = k+1
              IP_local(k)  = id+j ! location in the linear Fock matrix.
              if(i.eq.j) then
                 ip_check(k)=.false.
              else
                 ip_check(k)=.true.  ! used in fockx
              end if
           end do
        end do
     end do

  else
     ! UHF case.

     ! Coulomb part.
     k      = 0
     do ii=1,qm_main_r%NUMAT
        qm_scf_main_r%NW(ii) = k+1    ! lower triangle of a given block.
        ia     = qm_main_r%NFIRST(ii) ! at
        ib     = qm_main_r%NLAST(ii)
        do i=ia,ib
           id     = indx_local(i) ! qm_scf_main_r%INDX(i)
           do j=ia,i
              k      = k+1
              IP_local(k)  = id+j ! location in the linear Fock matrix.
              IP1_local(k) = i    ! i,j mapping in the Fock matrix.
              IP2_local(k) = j    ! (in the form of the square matrix)
              if(i.eq.j) then
                 ip_check(k)=.false.
              else
                 ip_check(k)=.true.  ! used in fockx
              end if
           end do
        end do
     end do

     ! Exchange part. 
     k    = 0
     do ii=1,qm_main_r%NUMAT
        JX_local(ii) = k+1       ! square matrix of a given block.
        nwii   = qm_scf_main_r%NW(ii)-1  ! 
        ia     = qm_main_r%NFIRST(ii)
        ib     = qm_main_r%NLAST(ii)
        do i=ia,ib
           i4     = i-ia+1
           do j=ia,i
              k      = k+1
              j4     = j-ia+1
              JP1_local(k) = i  ! mapping for lower triangle.
              JP2_local(k) = j
              !JP3_local(k) = nwii+qm_scf_main_r%INDX(i4)+j4   ! I.GE.J
              JP3_local(k) = nwii+indx_local(i4)+j4
           end do
           do j=i+1,ib
              k      = k+1
              j4     = j-ia+1
              JP1_local(k) = i  ! mapping for upper triangle.
              JP2_local(k) = j
              !JP3_local(k) = nwii+qm_scf_main_r%INDX(j4)+i4  ! I.LT.J
              JP3_local(k) = nwii+nwii+indx_local(j4)+i4
           end do
        end do
     end do
  end if
  JXLAST_local = k

  return
  !
  contains
     subroutine allocate_pair_index(dim_numat,uhf)
     !
     !
     !
     implicit none

     integer :: LMI,LME,norbs,dim_numat
     integer,save :: numat_old=-1
     integer :: ier=0
     logical :: uhf

     ! check quick return, if arrays are allocated previously.
     if(numat_old.eq.dim_numat) return

     numat_old = dim_numat
     norbs     = 9*dim_numat
     LMI       = 45*dim_numat
     LME       = 81*dim_numat

     ! deallocate memory.
     if(associated(IP_local))   deallocate(IP_local,stat=ier)
     if(associated(IP1_local))  deallocate(IP1_local,stat=ier)
     if(associated(IP2_local))  deallocate(IP2_local,stat=ier)
     if(associated(JP1_local))  deallocate(JP1_local,stat=ier)
     if(associated(JP2_local))  deallocate(JP2_local,stat=ier)
     if(associated(JP3_local))  deallocate(JP3_local,stat=ier)
     if(associated(JX_local))   deallocate(JX_local,stat=ier)
     if(associated(indx_local)) deallocate(indx_local,stat=ier)
     if(associated(ip_check))   deallocate(ip_check,stat=ier)

     ! allocate memory.
     ! for integer arrays:
     allocate(IP_local(LMI),stat=ier)
     allocate(indx_local(norbs),stat=ier)
     allocate(ip_check(LMI),stat=ier)
     if(uhf) then
        allocate(IP1_local(LMI),stat=ier)
        allocate(IP2_local(LMI),stat=ier)

        allocate(JP1_local(LME),stat=ier)
        allocate(JP2_local(LME),stat=ier)
        allocate(JP3_local(LME),stat=ier)
        allocate(JX_local(dim_numat),stat=ier)
     end if

     return
     end subroutine allocate_pair_index
     !
  end subroutine define_pair_index

  !
#endif
end module qm1_scf_module

