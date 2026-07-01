module qm1_lagdynamics_module
  use chm_kinds
  use number
  use qm1_constant

  implicit none
  ! cpmd related date.
  TYPE, public :: qm_cpmd_main
    real(chm_real),pointer:: PA_old(:)=>Null(), &   ! old RHF of UHF-alpha density matrix.
                             PB_old(:)=>Null(), &   ! 
                             PA_new(:)=>Null(), &   ! new RHF of UHF-alpha density matrix.
                             PB_new(:)=>Null(), &   ! 
                             PA_scf(:)=>Null(), &   ! BOMD scf density (for MTS-LN or MTS-ADMP).
                             PB_scf(:)=>Null(), &
                             dPA(:)=>Null(),    &   ! gradients of density matrix.
                             vPA(:)=>Null()         ! velocity of density matrix.
   
    ! for multiple time step md.
    real(chm_real),pointer:: FA_old(:)=>Null()      ! F based on old coordinates.
    real(chm_real),pointer:: H_save(:)=>Null()      ! h_core matrix from previous md step
    real(chm_real),pointer:: W_save(:)=>Null()      ! W matrix from previous md step.

    real(chm_real),pointer:: eslf_save(:,:)=>Null() ! eslf save for qm/mm-ewald
    real(chm_real),pointer:: empot_save(:)=>Null()  ! empot save

    real(chm_real)        :: Mass_PA=one            ! the mass of PA particule.

    ! for ADMP: adiabatic density matrix propagation (Schlegel et al.)
    logical               :: q_admp_md =.true.      ! default to use ADMP.

    ! for Curvy-steps ELMD.
    logical               :: q_curvy_md=.false.     ! use Curvy-steps ELMD (Herbert and MHG 2004).

    ! other info for CPMD
    logical               :: q_cnst=.false.         ! use constraints.
    integer               :: i_cnst_type = 1        ! 1: default: Schlegel et al.
                                                    ! 2: P = P + (P-Q)*PQ = 3P**2 - 2P**3 
                                                    ! 3: P = P + (P-Q)*PQ + 3(P-Q)*PQ^2 ! + 10*(P-Q)*PQ^3

    ! CPMD thermostat (Nose-Hoover chain or langevin).
    logical               :: q_nose=.false.         ! use nose thermostat.
    logical               :: q_langevin=.false.     ! use langevin dynamics.
!!    logical               :: q_exl_diss=.false.     ! use extended Lagrangian with dissipation.
    real(chm_real)        :: Temperature= zero      ! thermostat temperature.
    real(chm_real)        :: Delta = zero           ! time step
    integer               :: num_iter=1             ! number of iteration between each time step.
    integer               :: num_miter=1            ! Mstep for BOMD scf evaluation (for MTS-LN approach).
!!    integer               :: K_sum_order=0          ! K value (for AMN Niklasson, JCP (2009) 130:214109).
!!    real(chm_real)        :: Alpha_scale=one        ! alpha value scale.
    !
    real(chm_real)        :: NOSE_Mass(3),NOSEV(3),NOSE(3),NOSE_Old(3)
    !
    ! as all PA is considered to be the same, so no need to have fbeta and gamma the size of matrix.
    real(chm_real)        :: fbeta_delta            ! fbeta values for the langevan particles.
    real(chm_real)        :: gamma_delta(4)         ! gamma array for langevin particles.

    ! The stability checking purpose.
    logical               :: q_include_kinetic_E=.false. ! if .true., elec. KE is included in the E_scf.
    !
  END TYPE qm_cpmd_main

  !
  TYPE(qm_cpmd_main), save :: qm_cpmd_main_r


  ! local memory
  integer,save,private                :: size_norbs  = 0
  integer,save,private                :: ij_pair_cnt = 0     ! indexing of do i and do j
  integer,       pointer,save,private :: i_indx(:)=>Null()   ! for upper/lower triangle 
  integer,       pointer,save,private :: j_indx(:)=>Null()   ! of a given matrix. (see below.)
  real(chm_real),pointer,save,private :: PAwork(:,:)=>Null()
  real(chm_real),pointer,save,private :: PminQ(:,:) =>Null()
  real(chm_real),pointer,save,private :: PQwork(:,:)=>Null()
  real(chm_real),pointer,save,private :: PA_old_work(:,:)=>Null()
  real(chm_real),pointer,save,private :: PA2(:,:)=>Null()
  real(chm_real),pointer,save,private :: PA3(:,:)=>Null()
  real(chm_real),pointer,save,private :: T(:,:)=>Null()
  real(chm_real),pointer,save,private :: W2(:,:)=>Null()
  real(chm_real),pointer,save,private :: PA_local(:)=>Null()
  real(chm_real),pointer,save,private :: PA_linear(:)=>Null()
  real(chm_real),pointer,save,private :: PA_linear2(:)=>Null()
  real(chm_real),pointer,save,private :: PA_square(:)=>Null()
  real(chm_real),pointer,save,private :: FAHB_old(:)=>Null()

  ! for curvy-steps ELMD
  !real(chm_real),pointer,save,private :: PAwork(:,:)=>Null() 
  !real(chm_real),pointer,save,private :: PQwork(:,:)=>Null()
  !real(chm_real),pointer,save,private :: PA_old_work(:,:)=>Null()
  !real(chm_real),pointer,save,private :: PA2(:,:)=>Null()
  !real(chm_real),pointer,save,private :: PA_local(:)=>Null()
#if KEY_PARALLEL==1
  real(chm_real),pointer,save,private :: PA_check(:)=>Null()  ! for checking PA
#endif
  real(chm_real),pointer,save,private :: W_new(:,:)=>Null()   ! square form of F, and scracth
  real(chm_real),pointer,save,private :: D_new(:,:)=>Null()   ! Delta in square form
  real(chm_real),pointer,save,private :: dF_linear(:)=>Null() ! FP-PF
  real(chm_real),pointer,save,private :: vPA_new(:)=>Null() 

  ! for random numbers
  real(chm_real),pointer,save,private :: FRAND(:)=>Null()     ! for random numbers.

  !!! for Extended Lagrangian with dissipation (AMN Niklasson, JCP (2009) 130:214109).
  !!real(chm_real),pointer,save,private :: cextr(:)=>Null()     ! coefficient(0:K_sum_order)
  !!real(chm_real),pointer,save,private :: pa_exp(:,:)=>Null()  ! PA(K_sum_order,linear_norbs)
  !!real(chm_real),save,private         :: coefk                ! kappa value (This is not used.)

  contains
  !

#if KEY_MNDO97==1 /*mndo97*/
  !
  subroutine allocate_deallocate_qm_cpmd(qm_scf_main_l,qm_cpmd_main_l,q_cpmd,q_uhf,qallocate)
  !
  ! allocate/deallocate CPMD related arrays.
  ! if qallocate == .true. , allocate memory
  !                  false., deallocate memory
  !
  use qm1_info,only : qm_scf_main,Aass

  implicit none
  TYPE(qm_scf_main) :: qm_scf_main_l
  TYPE(qm_cpmd_main):: qm_cpmd_main_l
  logical :: q_cpmd,q_uhf,qallocate

  integer :: dim_norbs,dim_linear_norbs,dim_linear_fock2,dim_numat
  integer :: ier=0

  ! first, define array sizes (determined in determine_qm_scf_arrray_size)
  dim_numat       = qm_scf_main_l%dim_numat
  dim_norbs       = qm_scf_main_l%dim_norbs
  dim_linear_norbs= qm_scf_main_l%dim_linear_norbs
  dim_linear_fock2= qm_scf_main_l%dim_linear_fock2

  ! deallocate if arrays are associated.
  if(associated(qm_cpmd_main_l%PA_old)) deallocate(qm_cpmd_main_l%PA_old,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_cpmd','PA_old')
  if(associated(qm_cpmd_main_l%PB_old)) deallocate(qm_cpmd_main_l%PB_old,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_cpmd','PB_old')
  if(associated(qm_cpmd_main_l%PA_new)) deallocate(qm_cpmd_main_l%PA_new,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_cpmd','PA_new')
  if(associated(qm_cpmd_main_l%PB_new)) deallocate(qm_cpmd_main_l%PB_new,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_cpmd','PB_new')
  if(associated(qm_cpmd_main_l%PA_scf)) deallocate(qm_cpmd_main_l%PA_scf,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_cpmd','PA_scf')
  if(associated(qm_cpmd_main_l%PB_scf)) deallocate(qm_cpmd_main_l%PB_scf,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_cpmd','PB_scf')
  if(associated(qm_cpmd_main_l%dPA)) deallocate(qm_cpmd_main_l%dPA,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_cpmd','dPA')
  if(associated(qm_cpmd_main_l%vPA)) deallocate(qm_cpmd_main_l%vPA,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_cpmd','vPA')

  !
  if(associated(qm_cpmd_main_l%FA_old)) deallocate(qm_cpmd_main_l%FA_old,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_cpmd','FA_old')
  if(associated(qm_cpmd_main_l%H_save)) deallocate(qm_cpmd_main_l%H_save,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_cpmd','H_save')
  if(associated(qm_cpmd_main_l%W_save)) deallocate(qm_cpmd_main_l%W_save,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_cpmd','W_save')
  if(associated(qm_cpmd_main_l%eslf_save)) deallocate(qm_cpmd_main_l%eslf_save,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_cpmd','eslf_save')
  if(associated(qm_cpmd_main_l%empot_save)) deallocate(qm_cpmd_main_l%empot_save,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_qm_cpmd','empot_save') 

  ! now, allocate memory, only if qallocate==.true.
  if(qallocate.and.q_cpmd) then
     allocate(qm_cpmd_main_l%PA_old(dim_linear_norbs),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_cpmd','PA_old')
     allocate(qm_cpmd_main_l%PA_new(dim_linear_norbs),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_cpmd','PA_new')
     allocate(qm_cpmd_main_l%dPA(dim_linear_norbs),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_cpmd','dPA')
     allocate(qm_cpmd_main_l%vPA(dim_linear_norbs),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_cpmd','vPA')

     !
     allocate(qm_cpmd_main_l%FA_old(dim_linear_norbs),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_cpmd','FA_old')
     allocate(qm_cpmd_main_l%H_save(dim_linear_norbs),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_cpmd','H_save')
     allocate(qm_cpmd_main_l%W_save(dim_linear_fock2),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_cpmd','W_save')
     allocate(qm_cpmd_main_l%eslf_save(dim_numat,dim_numat),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_cpmd','eslf_save')
     allocate(qm_cpmd_main_l%empot_save(dim_numat),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_cpmd','empot_save')

     ! MTS-LN (Curvy only) & MTS (ADMP only) or for extra energy call.
     allocate(qm_cpmd_main_l%PA_scf(dim_linear_norbs),stat=ier)
     if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_cpmd','PA_scf')

     ! for uhf case.
     if(q_uhf) then
        allocate(qm_cpmd_main_l%PB_old(dim_linear_norbs),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_cpmd','PB_old')
        allocate(qm_cpmd_main_l%PB_new(dim_linear_norbs),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_cpmd','PB_new')
        allocate(qm_cpmd_main_l%PB_scf(dim_linear_norbs),stat=ier)
        if(ier.ne.0) call Aass(1,'allocate_deallocate_qm_cpmd','PB_scf')
     end if
  end if

  return
  end subroutine allocate_deallocate_qm_cpmd

 
  subroutine cpmd_energy(E_scf,H,W,Q,FA,PA,PB,               &
                         numat,                              &
                         dim_norbs,dim_linear_norbs,         &
                         dim_linear_fock,dim_linear_fock2,   &
                         dim_scratch,UHF,q_first_call)
  !
  ! energy dynamics loop:  assume only RHF case. (no UHF)
  !
  ! Compute energy and density gradient componet and restraints.
  !
  !
  !
  ! E_scf  : scf electonic energy
  ! H      : core hamiltonian matrix
  ! W      : two-electron integrals
  ! Q      : scratch array
  ! FA     : rhf or uhf-alpha Fock matrix.
  ! PA     : rhf or uhf-alpha density matrix.
  ! PB     : uhf-beta density matrix.
  ! uhf    : UHF flag
  !
  use qm1_info, only       : qm_control_r,qm_main_r,mm_main_r,qm_scf_main_r
  use qm1_parameters, only : CORE
  use qmmmewald_module,only : qmmm_ewald_r,qm_ewald_prepare_fock
  use mndgho_module,only   : qm_gho_info_r,GHO_expansion_cpmd,FTOFHB_cpmd,CTRASF ! GHO_expansion,FTOFHB
  use qm1_scf_module, only : fockx,calc_mulliken,qm_ewald_add_fock,  &
                             escf ,qm_ewald_correct_ee,             &
                             empot_local,empot_all
  use qm1_constant,only: EVCAL
#if KEY_PARALLEL==1
  use parallel
#endif
  use stream, only : prnlev

  implicit none
  !
  integer :: numat,dim_norbs,dim_linear_norbs,dim_linear_fock,dim_linear_fock2,dim_scratch
  real(chm_real):: E_scf
  real(chm_real):: H(dim_linear_norbs),W(dim_linear_fock2),Q(dim_scratch), &
                   FA(dim_linear_norbs),                                   &
                   PA(dim_linear_norbs),PB(dim_linear_norbs)
  logical :: UHF,q_first_call,q_kinetic_print

  !
  integer :: i,j,ii,jj,nstart,nstep,info,ncycle
  real(chm_real):: EF,EH,EFA,EFB,EHA,EHB,E_penalty,PL,PM
  real(chm_real):: ewdpot
  real(chm_real):: E_scf_new,fa_scale_1,fa_scale_2,r_factor
  integer :: nnumnod,mmynod
#if KEY_PARALLEL==1
  !integer :: ISTRT_CHECK       ! external function
  integer,save  :: JPARPT_local(0:MAXNODE),KPARPT_local(0:MAXNODE)
#endif
  real(chm_real):: ddot_mn ! external function
  integer, save :: old_N = 0, old_nqm=0
  integer, save :: mstart,mstop,istart,jstt,jend,istt,iend,iqmst,iqmend
  !

  if(UHF) call wrndie(-5,'<CPMD_DYNAMICS>','CPMD does not support UHF.')

  ! for parallelization
#if KEY_PARALLEL==1
  mmynod  = mynod
  nnumnod = numnod
  istart  = mynod+1
#else
  nnumnod = 1
  mmynod  = 0
  istart  = 1
#endif
  if(old_N .ne. dim_linear_norbs) then
     old_N  = dim_linear_norbs
     mstart = 1
     mstop  = dim_linear_norbs
     iqmst  = 1
     iqmend = numat
     if(qm_gho_info_r%q_gho) then
        istt = 1
        iend = qm_gho_info_r%lin_norbhb
     else
        istt = 1
        iend = dim_linear_norbs
     end if
#if KEY_PARALLEL==1
     if(numnod>1) then
        !mstart = ISTRT_CHECK(mstop,dim_linear_norbs)
        JPARPT_local(0)=0
        KPARPT_local(0)=0
        do i=1,numnod
           JPARPT_local(i)= dim_linear_norbs*i/numnod ! for linear vector
           KPARPT_local(i)= numat**i/numnod
        end do
        mstart = JPARPT_local(mynod)+1
        mstop  = JPARPT_local(mynod+1)

        !iqmst  = ISTRT_CHECK(iqmend,numat)
        iqmst  = KPARPT_local(mynod)+1
        iqmend = KPARPT_local(mynod+1)
     end if
     if(qm_gho_info_r%q_gho) then
        jstt = qm_gho_info_r%lin_norbhb*(mmynod)/nnumnod + 1
        jend = qm_gho_info_r%lin_norbhb*(mmynod+1)/nnumnod
     else
        jstt = dim_linear_norbs*(mmynod)/nnumnod + 1
        jend = dim_linear_norbs*(mmynod+1)/nnumnod
     end if
#else
     if(qm_gho_info_r%q_gho) then
        jstt = 1
        jend = qm_gho_info_r%lin_norbhb
     else
        jstt = 1
        jend = dim_linear_norbs
     end if
#endif
     if(associated(FAHB_old)) deallocate(FAHB_old)
  end if

  ! specific for gho.
  if(qm_gho_info_r%q_gho) then
     if(.not.associated(FAHB_old)) allocate(FAHB_old(dim_linear_norbs))
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

  ! if q_first_call=.false., it is a call to prepare cpmd, 
  !                          so no need to do the following loop.
  if(q_first_call) then
     ncycle   = 1
     r_factor = one
     !
     if(qm_cpmd_main_r%q_langevin) then
        if (qm_gho_info_r%q_gho) then
           call LNGFIL_PA(qm_cpmd_main_r%Delta,qm_gho_info_r%lin_norbhb)
        else
           call LNGFIL_PA(qm_cpmd_main_r%Delta,dim_linear_norbs)
        end if
     end if
  else
     ncycle   = qm_cpmd_main_r%num_iter
     r_factor = one/float(ncycle)
  end if

  ! Things to do:
  ! 1) I don't need to compute energy except ii==ncycle.
  ! 2) GHO implementation.

  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ! START OF inner DYNAMICS LOOP for density propagation.
  loopii: do ii=1,ncycle
     !------------------------------------------------------------------
     ! Start of energy and fock matrix contruction.
     ! 1: based on old qm and mm coordinates, which was saved from previous step.
     ! 2: based on new (current) qm and mm coordinates.
     !------------------------------------------------------------------
     ! 1: with old qm/mm coordinates.
     if(ii < ncycle) then
        ! construct F-matrix for RHF or Falpha-matrix for UHF
#if KEY_PARALLEL==1
        if(nnumnod>1) then
           !qm_cpmd_main_r%FA_old(1:mstart-1)               = zero
           qm_cpmd_main_r%FA_old(mstart:mstop)             = qm_cpmd_main_r%H_save(mstart:mstop)
           !qm_cpmd_main_r%FA_old(mstop+1:dim_linear_norbs) = zero
        else
#endif
           qm_cpmd_main_r%FA_old(1:dim_linear_norbs)  = qm_cpmd_main_r%H_save(1:dim_linear_norbs)
#if KEY_PARALLEL==1
        end if
#endif

        ! coulumb and exchange contribution.
        call fockx(qm_cpmd_main_r%FA_old,PA,PB,Q,qm_cpmd_main_r%W_save,       &
                   dim_linear_norbs,dim_linear_fock,UHF,                      &
                   numat,qm_main_r%nfirst,qm_main_r%nlast,qm_main_r%num_orbs, &
                   qm_scf_main_r%NW)

        ! QM/MM-Ewald
        !
        ! Note:
        ! MTS (multiple time scale) approach is not correct when 
        !              qm_control_r%q_do_cpmd_pme =.true.
        ! as empot_save is saved (for PME part) including the qm components. So,
        ! unless the qm charge at that step is corrected for that, it is not correct.
        !
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
           !
           ! do the job of the qm_ewald_prepare_fock routine.
           if(qm_control_r%q_do_cpmd_pme) then
              do i = iqmst,iqmend ! istart, numat, nnumnod
                 !ewdpot = DOT_PRODUCT(qm_cpmd_main_r%eslf_save(1:numat,i),mm_main_r%qm_charges(1:numat))
                 ewdpot = ddot_mn(numat,qm_cpmd_main_r%eslf_save(1:numat,i),1,mm_main_r%qm_charges(1:numat),1)
                 empot_all(i) = (qm_cpmd_main_r%empot_save(i) + qmmm_ewald_r%empot_qm_pme(i) + ewdpot) 
              end do
           else
              do i = iqmst,iqmend ! istart, numat, nnumnod
                 !ewdpot = DOT_PRODUCT(qm_cpmd_main_r%eslf_save(1:numat,i),mm_main_r%qm_charges(1:numat))
                 ewdpot = ddot_mn(numat,qm_cpmd_main_r%eslf_save(1:numat,i),1,mm_main_r%qm_charges(1:numat),1)
                 empot_all(i) = (qm_cpmd_main_r%empot_save(i) + ewdpot) !!!!* ev_a0
              end do
           end if
           ! copy empot to empot_local
           empot_local(1:numat)=qm_cpmd_main_r%empot_save(1:numat)

#if KEY_PARALLEL==1
           if(nnumnod>1) call VDGBRE(empot_all,KPARPT_local)
#endif

           ! now correct the fock matrix.
           call qm_ewald_add_fock(numat,qm_main_r%nfirst,qm_main_r%num_orbs,  &
                                  qm_scf_main_r%indx,qm_cpmd_main_r%FA_old,   &
                                  mm_main_r%qm_charges,dim_linear_norbs) 
        end if
        !
#if KEY_PARALLEL==1
        !if(nnumnod>1) call gcomb(qm_cpmd_main_r%FA_old,dim_linear_norbs)
        if(nnumnod>1) call VDGBRE(qm_cpmd_main_r%FA_old,JPARPT_local)
#endif

        if (qm_gho_info_r%q_gho) then
           !!! No need: store Fock matrix in AO basis for derivative
           !!!qm_gho_info_r%FAOA(1:dim_linear_norbs)=qm_cpmd_main_r%FA_old(1:dim_linear_norbs)

           ! transform f into hb for QM link atom
           call FTOFHB_cpmd(qm_cpmd_main_r%FA_old,FAHB_old,qm_gho_info_r%BT,    &
                            qm_gho_info_r%numat,qm_gho_info_r%nqmlnk,  &
                            dim_norbs,qm_gho_info_r%norbao,            &
                            qm_gho_info_r%lin_norbao,qm_gho_info_r%nactatm, &
                            qm_main_r%NFIRST,qm_main_r%NLAST,          &
                            qm_scf_main_r%indx)
           !call FTOFHB(qm_cpmd_main_r%FA_old,FAHB_old,qm_gho_info_r%BT,    &
           !            qm_gho_info_r%numat,qm_gho_info_r%nqmlnk,  &
           !            dim_norbs,qm_gho_info_r%norbao,            &
           !            qm_gho_info_r%lin_norbao,qm_gho_info_r%nactatm, &
           !            qm_main_r%NFIRST,qm_main_r%NLAST,          &
           !            qm_scf_main_r%indx)
        end if
     end if
     !------------------------------------------------------------------
     !------------------------------------------------------------------
     ! 2: with new qm/mm coordinates.
     ! construct F-matrix for RHF or Falpha-matrix for UHF
#if KEY_PARALLEL==1
     if(nnumnod>1) then
        !FA(1:mstart-1)               = zero
        FA(mstart:mstop)             = H(mstart:mstop)
        !FA(mstop+1:dim_linear_norbs) = zero
     else
#endif
        FA(1:dim_linear_norbs)  = H(1:dim_linear_norbs)
#if KEY_PARALLEL==1
     end if
#endif

     ! coulumb and exchange contribution.
     call fockx(FA,PA,PB,Q,W,dim_linear_norbs,dim_linear_fock,UHF,     &
                numat,qm_main_r%nfirst,qm_main_r%nlast,qm_main_r%num_orbs, &
                qm_scf_main_r%NW)

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
     !
#if KEY_PARALLEL==1
     !if(nnumnod>1) call gcomb(FA,dim_linear_norbs)
     if(nnumnod>1) call VDGBRE(FA,JPARPT_local)
#endif

     if (qm_gho_info_r%q_gho) then
        ! store Fock matrix in AO basis for derivative
        if(ii.eq.ncycle) qm_gho_info_r%FAOA(1:dim_linear_norbs)=FA(1:dim_linear_norbs)

        ! transform f into hb for QM link atom
        call FTOFHB_cpmd(FA,qm_gho_info_r%FAHB,qm_gho_info_r%BT,    &
                         qm_gho_info_r%numat,qm_gho_info_r%nqmlnk,  &
                         dim_norbs,qm_gho_info_r%norbao,            &
                         qm_gho_info_r%lin_norbao,qm_gho_info_r%nactatm, &
                         qm_main_r%NFIRST,qm_main_r%NLAST,          &
                         qm_scf_main_r%indx)
        !call FTOFHB(FA,qm_gho_info_r%FAHB,qm_gho_info_r%BT,    &
        !            qm_gho_info_r%numat,qm_gho_info_r%nqmlnk,  &
        !            dim_norbs,qm_gho_info_r%norbao,            &
        !            qm_gho_info_r%lin_norbao,qm_gho_info_r%nactatm, &
        !            qm_main_r%NFIRST,qm_main_r%NLAST,          &
        !            qm_scf_main_r%indx)
     end if

     if(ii.eq.ncycle) then
        ! Calculate the electronic energy.
        !
        ! Note: There is another call of ESCF above, only with F=H copy (core-hamiltonian).
        !       So, there is a room to remove redundant calculations.
        EFA = escf(dim_norbs,PA,FA,H,dim_linear_norbs)
        EFB = EFA    ! computed for complete Fock-matrix (here) and H-core matrix.

        ! In case of using QM/MM-Ewald, MM atom contributes in full.
        if(mm_main_r%LQMEWD) then
           ! empot_local is already copied above, qm_ewald_prepare_fock.
           EFA = EFA+qm_ewald_correct_ee(numat,qm_main_r%nfirst,qm_main_r%num_orbs, &
                                         qm_scf_main_r%indx,PA)
           EFB = EFA
        end if

        !EF        = EFA+EFB
        E_scf_new = EFA+EFB
     end if
     !------------------------------------------------------------------
     ! End of energy and fock matrix contruction.
     !------------------------------------------------------------------

     !------------------------------------------------------------------
     ! Now the CPMD part.
     ! 1. density gradient is just Fock matrix element.
     ! 2. density is propagated.
     ! 3. apply constraints to satisfy the total number of electrons and
     !    the idempotency.
     !------------------------------------------------------------------
     ! for the gradient of density matrix.
     ! dE/dP_ij = F_ij
     if(ii.eq.ncycle) then
        if(qm_gho_info_r%q_gho) then
           qm_cpmd_main_r%dPA(istt:iend)=qm_gho_info_r%FAHB(istt:iend)*EVCAL
        else
           qm_cpmd_main_r%dPA(istt:iend)=FA(istt:iend)*EVCAL
        end if
     else 
        ! linealy mix FA_old with FA
        fa_scale_1 = float(ii)*r_factor          ! scaling factor, for new FA
        fa_scale_2 = one - fa_scale_1            ! scaling factor, for old FA_old

        ! unit conversion
        fa_scale_1 = EVCAL*fa_scale_1
        fa_scale_2 = EVCAL*fa_scale_2

        if(qm_gho_info_r%q_gho) then
           qm_cpmd_main_r%dPA(istt:iend)= qm_gho_info_r%FAHB(istt:iend)*fa_scale_1 &
                                         +FAHB_old(istt:iend)*fa_scale_2
        else
           qm_cpmd_main_r%dPA(istt:iend)= FA(istt:iend)*fa_scale_1 &
                                         +qm_cpmd_main_r%FA_old(istt:iend)*fa_scale_2
        end if
     end if

     ! now update PA for the time step t+dt, then apply the PA constraints.
     ! for last update will be done in scf_gradient and PAwork will be updated.
     if(ii < ncycle) then
        q_kinetic_print =.false.
        if(qm_gho_info_r%q_gho) then
           call update_PA(qm_gho_info_r%PAHB,qm_cpmd_main_r%PA_new,         &
                          qm_cpmd_main_r%PA_old,                            &
                          qm_cpmd_main_r%dPA,qm_cpmd_main_r%vPA,            &
                          qm_cpmd_main_r%Mass_PA,                           &
                          qm_cpmd_main_r%NOSE_Mass,qm_cpmd_main_r%NOSEV,    &
                          qm_cpmd_main_r%NOSE,qm_cpmd_main_r%NOSE_Old,      &
                          qm_cpmd_main_r%Delta,qm_cpmd_main_r%Temperature,  &
                          qm_gho_info_r%lin_norbhb,qm_gho_info_r%norbhb,q_kinetic_print)
           ! apply constraints
           if(qm_cpmd_main_r%q_cnst) then
              call density_const(qm_gho_info_r%PAHB,qm_cpmd_main_r%PA_old,          &
                                 qm_gho_info_r%norbhb,qm_gho_info_r%lin_norbhb)
           end if

           ! for gho expansion PAHB -> PA.
           ! only works with RHF.
           call GHO_expansion_cpmd(qm_gho_info_r%norbhb,qm_gho_info_r%naos,       &
                                   qm_gho_info_r%lin_naos,qm_gho_info_r%nqmlnk,   &
                                   qm_gho_info_r%lin_norbhb,dim_norbs,            &
                                   dim_linear_norbs,qm_gho_info_r%mqm16,          &
                                   PA,PA,                                         &
                                   qm_gho_info_r%PAHB,qm_gho_info_r%PAHB,         &
                                   qm_gho_info_r%PAOLD,qm_gho_info_r%PAOLD,       &
                                   qm_gho_info_r%QMATMQ,qm_gho_info_r%BT,qm_gho_info_r%BTM, &
                                   qm_scf_main_r%indx,UHF)
           !call GHO_expansion(qm_gho_info_r%norbhb,qm_gho_info_r%naos,       &
           !                   qm_gho_info_r%lin_naos,qm_gho_info_r%nqmlnk,   &
           !                   qm_gho_info_r%lin_norbhb,dim_norbs,            &
           !                   dim_linear_norbs,qm_gho_info_r%mqm16,          &
           !                   PL,PM,PA,PA,                                   &
           !                   qm_gho_info_r%PAHB,qm_gho_info_r%PAHB,         &
           !                   qm_gho_info_r%PAOLD,qm_gho_info_r%PAOLD,       &
           !                   qm_gho_info_r%QMATMQ,qm_gho_info_r%BT,qm_gho_info_r%BTM, &
           !                   qm_scf_main_r%indx,UHF)
           qm_gho_info_r%PAHB(1:qm_gho_info_r%lin_norbhb)=qm_gho_info_r%PAOLD(1:qm_gho_info_r%lin_norbhb)
        else
           call update_PA(PA,qm_cpmd_main_r%PA_new,qm_cpmd_main_r%PA_old,   &
                          qm_cpmd_main_r%dPA,qm_cpmd_main_r%vPA,            &
                          qm_cpmd_main_r%Mass_PA,                           &
                          qm_cpmd_main_r%NOSE_Mass,qm_cpmd_main_r%NOSEV,    &
                          qm_cpmd_main_r%NOSE,qm_cpmd_main_r%NOSE_Old,      &
                          qm_cpmd_main_r%Delta,qm_cpmd_main_r%Temperature,  &
                          dim_linear_norbs,dim_norbs,q_kinetic_print)

           ! apply constraints
           if(qm_cpmd_main_r%q_cnst) then
              call density_const(PA,qm_cpmd_main_r%PA_old,dim_norbs,dim_linear_norbs)
           end if
        end if
     end if
     ! End of CPMD part.
     !------------------------------------------------------------------
  end do loopii 
  ! END OF the inner dynamics loop.
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  ! as in the end of the inner loop, E_scf_new should be the final E_scf.
  E_scf  = E_scf_new
#if KEY_PARALLEL==1
  if(nnumnod>1) call gcomb(E_scf,1)
#endif

  !
  ! Preparation for next MD step: save H and W for the current coordinates,
  !                               which become old coords in the next md step.
  ! save H_save and W_save to be used in the next md step.
  qm_cpmd_main_r%H_save(1:dim_linear_norbs) = H(1:dim_linear_norbs)
  qm_cpmd_main_r%W_save(1:dim_linear_fock2) = W(1:dim_linear_fock2)
  !
  ! for qm/mm-ewald to be saved for next md step.
  if(mm_main_r%LQMEWD) then
     qm_cpmd_main_r%eslf_save(1:numat,1:numat) = qmmm_ewald_r%eslf(1:numat,1:numat)
     qm_cpmd_main_r%empot_save(1:numat)        = qmmm_ewald_r%empot(1:numat)
  end if

  !
  ! for GHO method: since CA is not explicitly used within the scf cycle. So,
  ! transform orbitals to AO basis here, used by Mulliken analysis.
  if(qm_gho_info_r%q_gho) then
     ! for RHF: it only needs to be done here, not in the GHO_expansion.
     ! store the density for gho-related derivative calculation (density in HB N+4 basis).
     qm_gho_info_r%PHO(1:dim_linear_norbs)=two*qm_gho_info_r%PAHB(1:dim_linear_norbs)
  end if

  return
  end subroutine cpmd_energy


  subroutine cpmd_energy_only(E_scf,H,W,Q,FA,PA,PB,               &
                              numat,                              &
                              dim_norbs,dim_linear_norbs,         &
                              dim_linear_fock,dim_linear_fock2,   &
                              dim_scratch,UHF)
  !
  ! energy dynamics loop:  assume only RHF case. (no UHF)
  !
  ! Compute energy only. This routine is called when the energy only has to be
  ! reported (for printing purpose) during MD without updating density.
  ! 
  ! So, there will be no inner cycle, no-density update, etc.
  !
  ! E_scf  : scf electonic energy
  ! H      : core hamiltonian matrix
  ! W      : two-electron integrals
  ! Q      : scratch array
  ! FA     : rhf or uhf-alpha Fock matrix.
  ! PA     : rhf or uhf-alpha density matrix. <- should be old density, as PA is new
  ! PB     : uhf-beta density matrix.         <- should be old density.
  ! uhf    : UHF flag
  !
  use qm1_info, only       : qm_control_r,qm_main_r,mm_main_r,qm_scf_main_r
  use qm1_parameters, only : CORE
  use qmmmewald_module,only : qmmm_ewald_r,qm_ewald_prepare_fock
  use mndgho_module,only   : qm_gho_info_r,GHO_expansion_cpmd,FTOFHB_cpmd,CTRASF ! GHO_expansion,FTOFHB
  use qm1_scf_module, only : fockx,calc_mulliken,qm_ewald_add_fock,  &
                             escf,qm_ewald_correct_ee,              &
                             empot_local,empot_all
  use qm1_constant,only: EVCAL
#if KEY_PARALLEL==1
  use parallel
#endif
  use stream, only : prnlev

  implicit none
  !
  integer :: numat,dim_norbs,dim_linear_norbs,dim_linear_fock,dim_linear_fock2,dim_scratch
  real(chm_real):: E_scf
  real(chm_real):: H(dim_linear_norbs),W(dim_linear_fock2),Q(dim_scratch), &
                   FA(dim_linear_norbs),                                   &
                   PA(dim_linear_norbs),PB(dim_linear_norbs)
  logical :: UHF

  !
  integer :: i,j,ii,jj,nstart,nstep,info,ncycle
  real(chm_real):: EF,EH,EFA,EFB,EHA,EHB,E_penalty,PL,PM
  real(chm_real):: ewdpot
  real(chm_real):: E_scf_new
  integer :: nnumnod,mmynod
#if KEY_PARALLEL==1
  !integer :: ISTRT_CHECK       ! external function
  integer,save  :: JPARPT_local(0:MAXNODE)
#endif
  integer, save :: old_N = 0, old_nqm=0
  integer, save :: mstart,mstop,istart,jstt,jend,istt,iend
  !

  if(UHF) call wrndie(-5,'<CPMD_DYNAMICS>','CPMD does not support UHF.')

  ! for parallelization
#if KEY_PARALLEL==1
  mmynod  = mynod
  nnumnod = numnod
  istart  = mynod+1
#else
  nnumnod = 1
  mmynod  = 0
  istart  = 1
#endif
  if(old_N .ne. dim_linear_norbs) then
     old_N  = dim_linear_norbs
     mstart = 1
     mstop  = dim_linear_norbs
     if(qm_gho_info_r%q_gho) then
        istt = 1
        iend = qm_gho_info_r%lin_norbhb
     else
        istt = 1
        iend = dim_linear_norbs
     end if
#if KEY_PARALLEL==1
     if(numnod>1) then
        !mstart = ISTRT_CHECK(mstop,dim_linear_norbs)
        JPARPT_local(0)=0
        do i=1,numnod
           JPARPT_local(i)= dim_linear_norbs*i/numnod ! for linear vector
        end do
        mstart = JPARPT_local(mynod)+1
        mstop  = JPARPT_local(mynod+1)
     end if
     if(qm_gho_info_r%q_gho) then
        jstt = qm_gho_info_r%lin_norbhb*(mmynod)/nnumnod + 1
        jend = qm_gho_info_r%lin_norbhb*(mmynod+1)/nnumnod
     else
        jstt = dim_linear_norbs*(mmynod)/nnumnod + 1
        jend = dim_linear_norbs*(mmynod+1)/nnumnod
     end if
#else
     if(qm_gho_info_r%q_gho) then
        jstt = 1
        jend = qm_gho_info_r%lin_norbhb
     else
        jstt = 1
        jend = dim_linear_norbs
     end if
#endif
     !if(associated(FAHB_old)) deallocate(FAHB_old)
  end if

  !! specific for gho.
  !if(qm_gho_info_r%q_gho) then
  !   if(.not.associated(FAHB_old)) allocate(FAHB_old(dim_linear_norbs))
  !end if
  !
  ! specific for qm/mm-Ewald part.
  !if(old_nqm.ne.numat) then
  !   old_nqm = numat
  !   if(mm_main_r%LQMEWD) then
  !      if(associated(empot_all))   deallocate(empot_all)
  !      if(associated(empot_local)) deallocate(empot_local)
  !      allocate(empot_all(numat))
  !      allocate(empot_local(numat))
  !   end if
  !end if

  ! Things what is doing here.
  ! 1) Only energy is evaluated based on the present density.

  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !------------------------------------------------------------------
  ! Start of energy and fock matrix contruction.
  ! 1: based on new (current) qm and mm coordinates.
  ! construct F-matrix for RHF or Falpha-matrix for UHF
#if KEY_PARALLEL==1
  if(nnumnod>1) then
     !FA(1:mstart-1)               = zero
     FA(mstart:mstop)             = H(mstart:mstop)
     !FA(mstop+1:dim_linear_norbs) = zero
  else
#endif
     FA(1:dim_linear_norbs)  = H(1:dim_linear_norbs)
#if KEY_PARALLEL==1
  end if
#endif

  ! coulumb and exchange contribution.
  call fockx(FA,PA,PB,Q,W,dim_linear_norbs,dim_linear_fock,UHF,     &
             numat,qm_main_r%nfirst,qm_main_r%nlast,qm_main_r%num_orbs, &
             qm_scf_main_r%NW)

  ! QM/MM-Ewald
  !
  ! Note:
  ! MTS (multiple time scale) approach is not correct when
  !              qm_control_r%q_do_cpmd_pme =.true.
  ! as empot_save is saved (for PME part) including the qm components. So,
  ! unless the qm charge at that step is corrected for that, it is not correct.
  !
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
  !
#if KEY_PARALLEL==1
  !if(nnumnod>1) call gcomb(FA,dim_linear_norbs)
  if(nnumnod>1) call VDGBRE(FA,JPARPT_local)
#endif

  if (qm_gho_info_r%q_gho) then
     ! store Fock matrix in AO basis for derivative
     qm_gho_info_r%FAOA(1:dim_linear_norbs)=FA(1:dim_linear_norbs)

     ! transform f into hb for QM link atom
     call FTOFHB_cpmd(FA,qm_gho_info_r%FAHB,qm_gho_info_r%BT,    &
                      qm_gho_info_r%numat,qm_gho_info_r%nqmlnk,  &
                      dim_norbs,qm_gho_info_r%norbao,            &
                      qm_gho_info_r%lin_norbao,qm_gho_info_r%nactatm, &
                      qm_main_r%NFIRST,qm_main_r%NLAST,          &
                      qm_scf_main_r%indx)
  end if

  ! Calculate the electronic energy.
  !
  ! Note: There is another call of ESCF above, only with F=H copy (core-hamiltonian).
  !       So, there is a room to remove redundant calculations.
  EFA = escf(dim_norbs,PA,FA,H,dim_linear_norbs)
  EFB = EFA    ! computed for complete Fock-matrix (here) and H-core matrix.

  ! In case of using QM/MM-Ewald, MM atom contributes in full.
  if(mm_main_r%LQMEWD) then
     ! empot_local is already copied above, qm_ewald_prepare_fock.
     EFA = EFA+qm_ewald_correct_ee(numat,qm_main_r%nfirst,qm_main_r%num_orbs, &
                                   qm_scf_main_r%indx,PA)
     EFB = EFA
  end if

  !EF        = EFA+EFB
  E_scf_new = EFA+EFB
  !------------------------------------------------------------------
  ! End of energy and fock matrix contruction.
  !------------------------------------------------------------------

  !------------------------------------------------------------------
  ! Now the CPMD part.
  ! 1. density gradient is just Fock matrix element.
  !------------------------------------------------------------------
  ! for the gradient of density matrix.
  ! dE/dP_ij = F_ij
  if(qm_gho_info_r%q_gho) then
     qm_cpmd_main_r%dPA(istt:iend)=qm_gho_info_r%FAHB(istt:iend)*EVCAL
  else
     qm_cpmd_main_r%dPA(istt:iend)=FA(istt:iend)*EVCAL
  end if
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  ! as in the end of the inner loop, E_scf_new should be the final E_scf.
  E_scf  = E_scf_new
#if KEY_PARALLEL==1
  if(nnumnod>1) call gcomb(E_scf,1)
#endif

  !
  ! for GHO method: since CA is not explicitly used within the scf cycle. So,
  ! transform orbitals to AO basis here, used by Mulliken analysis.
  !if(qm_gho_info_r%q_gho) then
  !   ! for RHF: it only needs to be done here, not in the GHO_expansion.
  !   ! store the density for gho-related derivative calculation (density in HB N+4 basis).
  !   qm_gho_info_r%PHO(1:dim_linear_norbs)=two*qm_gho_info_r%PAHB(1:dim_linear_norbs)
  !end if

  return
  end subroutine cpmd_energy_only


  subroutine density_const(PA,PA_old,dim_norbs,dim_linear_norbs)
  !
  ! apply constraints to the density
  ! 1) total number of electrons
  ! 2) idempotency
  !
  ! PA        : input density matrix (contains lower triangle).
  !
  use number, only : zero,one,two,three
#if KEY_PARALLEL==1
  use parallel
#endif
  use stream, only : prnlev
  use qm1_info,only: qm_main_r

  implicit none
  integer :: dim_norbs,dim_linear_norbs
  real(chm_real):: PA(dim_linear_norbs),PA_old(dim_linear_norbs)

  real(chm_real),parameter :: ThresConv=1.0D-12,ThresElec =1.0D-12, &
                              ThresLow =1.0D-09,ThresUpper=1.0D-08
  integer :: i,j,ii,ij,jj,kk,n,icnt,jcnt,i_cnst_type_local
  real(chm_real):: sum_p,sum_e,rdot_tmp
  integer, save :: nnumnod,nmynod
  integer, save :: mstart,mstop,jstart,jstop,isize,totalsize
#if KEY_PARALLEL==1
  integer       :: ISTRT_CHECK       ! external function
  integer,save  :: JPARPT_local(0:MAXNODE),KPARPT_local(0:MAXNODE) 
#endif
  real(chm_real):: ddot_mn ! external function

  !
  if(.not.qm_cpmd_main_r%q_admp_md) return

  ! memory allocation
  if(size_norbs .ne. dim_norbs) then
     size_norbs = dim_norbs
     if(associated(PAwork))      deallocate(PAwork)
     if(associated(PA_old_work)) deallocate(PA_old_work)
     if(associated(PminQ))       deallocate(PminQ)
     if(associated(PQwork))      deallocate(PQwork)
     if(associated(PA2))         deallocate(PA2)
     if(associated(PA3))         deallocate(PA3)
     if(associated(T))           deallocate(T)
     if(associated(PA_local))    deallocate(PA_local)
     if(associated(PA_linear))   deallocate(PA_linear)
     if(associated(PA_square))   deallocate(PA_square)

     ! total count
     icnt=0
     do i=1,dim_norbs
        icnt = icnt+i
     end do
     mstart    = 1
     mstop     = icnt
     jstart    = 1
     jstop     = dim_norbs*dim_norbs
     totalsize = icnt
     jcnt      = dim_norbs*dim_norbs
#if KEY_PARALLEL==1
     nnumnod   = numnod
     nmynod    = mynod
     if(nnumnod > totalsize) call wrndie(-5,'<density_const>', &
     'Number of node > total array size. Decrease of No. cpus for efficiecy.')

     ! Prepare array for vector allgather calls using VDGBRE
     ! mapping for each node (Hard weird).
     JPARPT_local(0)=0
     KPARPT_local(0)=0
     do i=1,nnumnod
        JPARPT_local(i)= totalsize*i/nnumnod   ! for linear vector.
        KPARPT_local(i)= jcnt*i/nnumnod        ! for square vector
     end do
     mstart = JPARPT_local(mynod)+1
     mstop  = JPARPT_local(mynod+1)
     jstart = KPARPT_local(mynod)+1
     jstop  = KPARPT_local(mynod+1)
#else
     nmynod = 0
     nnumnod= 1
#endif
     isize  = mstop-mstart+1   ! size of array for the current node.
  end if

  ! common
  if(.not.associated(PAwork))      allocate(PAwork(dim_norbs,dim_norbs))
  if(.not.associated(PA2))         allocate(PA2(dim_norbs,dim_norbs))
  if(.not.associated(T))           allocate(T(dim_norbs,dim_norbs))
  if(.not.associated(PA_local))    allocate(PA_local(dim_norbs))
  if(.not.associated(PA_linear))   allocate(PA_linear(dim_linear_norbs))
  if(.not.associated(PA_square))   allocate(PA_square(dim_norbs*dim_norbs))
  if(.not.associated(PA3))         allocate(PA3(dim_norbs,dim_norbs))
  if(.not.associated(PminQ )) allocate(PminQ(dim_norbs,dim_norbs))  ! for P-Q = 2P-I
  if(.not.associated(PQwork)) allocate(PQwork(dim_norbs,dim_norbs)) ! for PQ  = P-P^2

  ! make the matrix in square form.
  call make_square(PA,PAwork,dim_norbs,dim_norbs,dim_linear_norbs)
  !
  ! special for PA_old_work
  if(.not.associated(PA_old_work)) then
    allocate(PA_old_work(dim_norbs,dim_norbs))
    call make_square(PA_old,PA_old_work,dim_norbs,dim_norbs,dim_linear_norbs)
  !else
  !  ! as PA_old_work is already for PA_old (see update_PA).
  !  call make_square(PA_old,PA_old_work,dim_norbs,dim_norbs,dim_linear_norbs)
  end if

! original form based on matrix multiplications.
!     ! P**2  and P**3
!     PA2(1:n,1:n)=MATMUL(PAwork(1:n,1:n),PAwork(1:n,1:n))
!     PA3(1:n,1:n)=MATMUL(PA2(1:n,1:n)   ,PAwork(1:n,1:n))
!     ! T= 3P**2 - 2P**3 - P
!     T(1:n,1:n)  =three*PA2(1:n,1:n) - two*PA3(1:n,1:n) - PAwork(1:n,1:n)
!
!     ! P_i * T * P_i
!     PA2(1:n,1:n)=MATMUL(T(1:n,1:n),PA_old_work(1:n,1:n))
!     PA2(1:n,1:n)=MATMUL(PA_old_work(1:n,1:n),PA2(1:n,1:n))
!     
!     ! Q_i * T * Q_i
!     PA3(1:n,1:n)=MATMUL(T(1:n,1:n),Q(1:n,1:n))
!     PA3(1:n,1:n)=MATMUL(Q(1:n,1:n),PA3(1:n,1:n))
!     
!     ! P_i+1 <- P_i+1 + P_i*T*P_i + Q_i*T*Q_i
!     PAwork(1:n,1:n) = PAwork(1:n,1:n) + PA2(1:n,1:n) + PA3(1:n,1:n)


  sum_p = one
  jj=0
  n=dim_norbs  ! dimension
  do
     ! compute PQ = P-P^2
     do ii=mstart,mstop
        !PA_linear(ii) = PAwork(j_indx(ii),i_indx(ii)) &
        !               -DOT_PRODUCT(PAwork(1:n,i_indx(ii)),PAwork(1:n,j_indx(ii)))
        PA_linear(ii) = PAwork(j_indx(ii),i_indx(ii)) &
                       -ddot_mn(n,PAwork(1:n,i_indx(ii)),1,PAwork(1:n,j_indx(ii)),1)
     end do
#if KEY_PARALLEL==1
     if(nnumnod>1) call VDGBRE(PA_linear,JPARPT_local)
#endif
     call make_square(PA_linear,PQwork,dim_norbs,dim_norbs,dim_linear_norbs)

     ! check for convergence: SQRT{Tr [ (P*P - P)^2 ]}/N < ThresConv.
     sum_p =zero
     do i=nmynod+1,n,nnumnod   ! 1,n
        sum_p = sum_p + ddot_mn(n,PQwork(1:n,i),1,PQwork(1:n,i),1)  ! DOT_PRODUCT(PQwork(1:n,i),PQwork(1:n,i))
     end do
#if KEY_PARALLEL==1
     if(nnumnod>1) call gcomb(sum_p,1) 
#endif
     sum_p = sqrt(sum_p)/float(n)
     !if(prnlev.ge.2) write(6,*) 'idemp:',jj,sum_p
     if(sum_p.le.ThresConv) then
        ! check the total number of electrons is conserved.
        sum_e = zero
        do i=1,n
           sum_e = sum_e + PAwork(i,i)
        end do
        !if(prnlev.ge.2) write(6,*)'density:',jj,sum_e,qm_main_r%nalpha
        if(abs(sum_e-float(qm_main_r%nalpha)) .le. ThresElec) exit
     end if

     ! PQ is computed above. now, compute P-Q = 2P - I
     do i=1,n
        PminQ(1:n,i) = two*PAwork(1:n,i)
        PminQ(i,i)   = PminQ(i,i) - one
     end do

     ! T = 3P**2 - 2P**3 - P = (P-Q)*PQ; by looping over the lower triangle.
     do ii=mstart,mstop
        !PA_linear(ii) = DOT_PRODUCT(PminQ(1:n,j_indx(ii)),PQwork(1:n,i_indx(ii)))
        PA_linear(ii) = ddot_mn(n,PminQ(1:n,j_indx(ii)),1,PQwork(1:n,i_indx(ii)),1)
     end do
#if KEY_PARALLEL==1
     if(nnumnod>1) call VDGBRE(PA_linear,JPARPT_local)
#endif
     call make_square(PA_linear,T,dim_norbs,dim_norbs,dim_linear_norbs)

     !
     ! first step: Use Schlegel et al. approach.
     i_cnst_type_local = qm_cpmd_main_r%i_cnst_type
     !if(jj.eq.0 .or. sum_p.ge.ThresUpper) i_cnst_type_local = 1

     ! now, update P_i+1: three different options.
     if(i_cnst_type_local .eq. 1) then
        ! P_i+1 <- P_i+1 + P_i*T*P_i + Q_i*T*Q_i as
        ! P*T*P+Q*T*Q is transformed into : T - (T*P_i + P_i*T - 2*P_i*T*P_i).

        ! note that T*P_i and T*Q_i are not symmetric matrices. 
        !
        ! T*P_i + P_i*T : j,i element = T(:,j).P(:,i)+P(:,j).T(:,i)
        ! use 
        ! PA2 = two* (P_i * T * P_i)
        ! PA3 = T - (T*P_i + P_i*T)
        icnt = 0
        jcnt = 0
        do i=1,n
           do j=1,i
              icnt = icnt + 1
              jcnt = jcnt + 1
              if((icnt.ge.mstart .and. icnt.le.mstop) .or. (jcnt.ge.jstart .and. jcnt.le.jstop)) then
                 PA2(j,i) = ddot_mn(n,T(1:n,j),1,PA_old_work(1:n,i),1)  ! DOT_PRODUCT(T(1:n,j),PA_old_work(1:n,i))
                 if(icnt.ge.mstart .and. icnt.le.mstop) then
                    !PA3(j,i) = T(j,i)-(PA2(j,i)+DOT_PRODUCT(PA_old_work(1:n,j),T(1:n,i)))
                    PA3(j,i) = T(j,i)-(PA2(j,i)+ddot_mn(n,PA_old_work(1:n,j),1,T(1:n,i),1))
                 end if
                 if(jcnt.ge.jstart .and. jcnt.le.jstop) then
                    PA_square(jcnt) = PA2(j,i) ! DOT_PRODUCT(T(1:n,j),PA_local(1:n))
                 end if
              end if
           end do
           ! to complete the square matrix.
           do j=i+1,n
              jcnt = jcnt + 1
              if(jcnt.ge.jstart .and. jcnt.le.jstop) then
                 PA_square(jcnt) = ddot_mn(n,T(1:n,j),1,PA_old_work(1:n,i),1)  ! DOT_PRODUCT(T(1:n,j),PA_old_work(1:n,i))
              end if
           end do
        end do
#if KEY_PARALLEL==1
        if(nnumnod>1) call VDGBRE(PA_square,KPARPT_local)
#endif

        ! now, complete the two* (P_i * (T * P_i)) and 
        ! P_i+1 <- P_i+1 + P_i*T*P_i + Q_i*T*Q_i
        ! P*T*P+Q*T*Q is transformed into : T - (T*P_i + P_i*T - 2*P_i*T*P_i).
        ! So, the above matrix multiplications are also changed accordingly.
        ! PAwork(1:n,1:n) = PAwork(1:n,1:n) + PA2(1:n,1:n) + PA3(1:n,1:n)
        !
        ! only loop over the lower triangle.
        do ii=mstart,mstop
           i = i_indx(ii)
           j = j_indx(ii)
           ij= n*(i-1)
           !PA_linear(ii)=PAwork(j,i)+two*DOT_PRODUCT(PA_square(ij+1:ij+n),PA_old_work(1:n,j)) &
           !                         +PA3(j,i)
           PA_linear(ii)=PAwork(j,i)+two*ddot_mn(n,PA_square(ij+1:ij+n),1,PA_old_work(1:n,j),1) &
                                    +PA3(j,i)
        end do
#if KEY_PARALLEL==1
        if(nnumnod>1) call VDGBRE(PA_linear,JPARPT_local)
#endif
        call make_square(PA_linear,PAwork,dim_norbs,dim_norbs,dim_linear_norbs)

     else if(i_cnst_type_local == 2) then
        ! This is for low-level convergence:
        ! Here, we accelerate it approximately: P_i*T*P_i + Q_i*T*Q_i ~ T
        ! as T*P_i + P_i*T - 2*P_i*T*P_i ~ 0
        ! P_i+1 <- P_i+1 + T, where T=(P-Q)*PQ
        PAwork(1:n,1:n) = PAwork(1:n,1:n) + T(1:n,1:n)

     else if(i_cnst_type_local == 3) then
        ! (P-Q)*PQ^2 - (P-Q)*PQ*PQ: loop over the lower triangle.
        ! and update P_i+1 <- P_i+1 + (P-Q)*PQ + 3*(P-Q)*PQ^2
        do ii=mstart,mstop
           i = i_indx(ii)
           j = j_indx(ii)
           PA_linear(ii)= PAwork(j,i) + T(i,j) + three*ddot_mn(n,T(1:n,j),1,PQwork(1:n,i),1)   ! DOT_PRODUCT(T(1:n,j),PQwork(1:n,i))
        end do
#if KEY_PARALLEL==1
        if(nnumnod>1) call VDGBRE(PA_linear,JPARPT_local)
#endif
        call make_square(PA_linear,PAwork,dim_norbs,dim_norbs,dim_linear_norbs)
     end if

     ! update counter.
     jj    = jj+1
  end do

  ! copy PAwork to PA:
  ij    = 0
  do i=1,n
     PA(ij+1:ij+i) = PAwork(1:i,i)
     ij     = ij + i
  end do

  return
  end subroutine density_const


  subroutine update_PA(PA,PA_new,PA_old,F_PA,V_PA,Mass_PA,  &
                       NOSE_Mass,NOSEV,NOSE,NOSE_Old,       &
                       Delta,Temperature,                   &
                       dim_linear,dim_norbs,q_print)
  !
  ! update PA density 
  ! if ADMP: (Verlet algorithm)
  ! 1) if nose is on: the Nose-Hoover chain thermostat.
  ! 2) if not.      : no thermostat with the Verlet.
  !
  ! if Curvy-steps ELMD.
  ! 1) use velocity verlet algorithm without thermostat.
  !
  use chm_kinds
  use number
  use consta, only : KBOLTZ,TIMFAC
#if KEY_PARALLEL==1
  use parallel
#endif
  use stream, only : prnlev
  use qm1_info, only : qm_control_r,qm_main_r

  implicit none
  integer        :: dim_linear,dim_norbs
  real(chm_real) :: PA(*),PA_new(*),PA_old(*),F_PA(*),V_PA(*),Mass_PA
  real(chm_real) :: NOSE_Mass(3),NOSEV(3),NOSE(3),NOSE_Old(3),Delta,  &
                    Temperature
  logical        :: q_print
  ! local 
  integer,parameter :: MaxIter=1000,max_cycle=20,max_inner_cycle=5 ! 10
  real(chm_real),parameter :: ThresConv=1.0D-12,ThresElec =1.0D-12, &
                              ThresDens=1.0D-15 ! 1.0D-30
  real(chm_real),save :: r_fact(max_cycle)
  integer :: i,j,k,ii,jj,icnt,jcnt,ks,n,linear_nsize
  real(chm_real) :: NOSE_New(3),KIN_Old,X1,X2,Kb_T,Kinetic_E,     &
                    rtmp_n,rtmp_p,r_check,delta2,r_2delta,r_mass, &
                    dt_m,sum_p,sum_e,delta_local
  real(chm_real) :: e_conv,t_conv,m_conv,mass_local
  integer, save  :: nnumnod,nmynod
  integer, save  :: size_n=0, imd_counter=0
  integer, save  :: mstart,mstop,msize,jstart,jstop
#if KEY_PARALLEL==1
  integer, save  :: JPARPT_local(0:MAXNODE),KPARPT_local(0:MAXNODE)
#endif
  real(chm_real):: ddot_mn,ddot2d_mn ! external function
  real(chm_real),save :: r_dim_linear,r_norbs,r_elec
  !!logical :: q_do_diss

  if(size_n .ne. dim_linear) then
     size_n = dim_linear
     mstart = 1
     mstop  = dim_linear
     msize  = dim_linear

     jstart = 1
     jstop  = dim_norbs*dim_norbs
     icnt   = dim_norbs*dim_norbs
#if KEY_PARALLEL==1
     nnumnod = numnod
     JPARPT_local(0) = 0
     do i=1,nnumnod
        JPARPT_local(i)= dim_linear*i/nnumnod  ! for linear vector
     end do
     mstart = JPARPT_local(mynod)+1
     mstop  = JPARPT_local(mynod+1)
     msize  = mstop - mstart + 1

     KPARPT_local(0)=0
     do i=1,nnumnod
        KPARPT_local(i)= icnt*i/nnumnod        ! for square vector
     end do
     jstart = KPARPT_local(mynod)+1
     jstop  = KPARPT_local(mynod+1)
#endif

     ! prepare i_indx and j_indx, which contain the i and j indices for do looping
     ! over lower or upper triangle in matrix handling. This is done here as it
     ! is used many time during the CPMD calculations.
     ij_pair_cnt = dim_linear ! mstop - mstart + 1
     if(associated(i_indx)) deallocate(i_indx)
     if(associated(j_indx)) deallocate(j_indx)
     if(.not.associated(i_indx)) allocate(i_indx(ij_pair_cnt))
     if(.not.associated(j_indx)) allocate(j_indx(ij_pair_cnt))
     icnt = 0
     ii   = 0
     do i=1,dim_norbs
        do j=1,i
           icnt=icnt+1
           ii = ii + 1
           if(icnt.ge.mstart .and. icnt.le.mstop) then 
              i_indx(ii) = i
              j_indx(ii) = j
           end if
        end do
     end do

     if(qm_cpmd_main_r%q_admp_md) then
        if(associated(PA_old_work)) deallocate(PA_old_work)
        if(associated(W_new))       deallocate(W_new)
        if(associated(W2))          deallocate(W2)
        !if(associated(PA_linear))   deallocate(PA_linear)
        !if(associated(T))           deallocate(T)
        !if(associated(PA_linear2))  deallocate(PA_linear2)

        !!! Extended Lagrangian MD with dissipation
        !!! ref: AMN Niklasson, JCP (2009) 130:214109.
        !!if(qm_cpmd_main_r%q_exl_diss) then
        !!   if(associated(pa_exp)) deallocate(pa_exp)
        !!end if

     else if(qm_cpmd_main_r%q_curvy_md) then
        size_norbs = dim_norbs
        if(associated(PAwork))      deallocate(PAwork)
        if(associated(PQwork))      deallocate(PQwork)
        if(associated(PA_old_work)) deallocate(PA_old_work)
        if(associated(PA2))         deallocate(PA2)
        if(associated(PA_local))    deallocate(PA_local)
        if(associated(W_new))       deallocate(W_new)
        if(associated(D_new))       deallocate(D_new)
        if(associated(dF_linear))   deallocate(dF_linear)
        if(associated(vPA_new))     deallocate(vPA_new)
#if KEY_PARALLEL==1
        if(associated(PA_check))    deallocate(PA_check)
#endif

        r_fact(1) = one
        sum_e= one
        do i=2,max_cycle
           sum_e= sum_e + one
           r_fact(i) =r_fact(i-1)*sum_e
        end do
        ! inverse of factorial
        r_fact(1:max_cycle) = one/r_fact(1:max_cycle)

        r_dim_linear = one/float(dim_linear)
        r_norbs      = one/float(dim_norbs)
        r_elec       = float(qm_main_r%nalpha)
     end if
  end if

  ! for langevin dynamics: prepare for random numbers
  if(qm_cpmd_main_r%q_langevin) then
     linear_nsize = mstop - mstart + 1
     call DLNGV_PA(linear_nsize)
  end if

  if(qm_cpmd_main_r%q_admp_md) then
     ! allocate memory
     n = dim_norbs
     if(.not.associated(PA_old_work)) allocate(PA_old_work(dim_norbs,dim_norbs))
     if(.not.associated(W_new))       allocate(W_new(dim_norbs,dim_norbs))
     if(.not.associated(W2))          allocate(W2(dim_norbs,dim_norbs))
     !!q_do_diss =.false.
     !!if(qm_cpmd_main_r%q_exl_diss) then
     !!   ! Extended Lagrangian MD with dissipation
     !!   ! ref: AMN Niklasson, JCP (2009) 130:214109.
     !!   if(.not.associated(pa_exp)) then
     !!      allocate(pa_exp(qm_cpmd_main_r%K_sum_order,msize))
     !!      !
     !!      pa_exp(1:qm_cpmd_main_r%K_sum_order,1:msize) = zero
     !!      imd_counter = 0   ! reset counter.
     !!   end if
     !!   imd_counter = imd_counter + 1
     !!
     !!   ! k=0; PA current, k=1, PA_old, etc.
     !!   if(imd_counter <= qm_cpmd_main_r%K_sum_order) then
     !!      do ii=1,msize
     !!         do k = imd_counter,2,-1
     !!            pa_exp(k,ii) = pa_exp(k-1,ii)
     !!         end do
     !!      end do
     !!      pa_exp(1,1:msize) = PA_old(mstart:mstop)
     !!   else
     !!      q_do_diss =.true.
     !!      !
     !!      ! shift array by one column to update to the current PA arrays.
     !!      do ii=1,msize
     !!         do k = qm_cpmd_main_r%K_sum_order,2,-1
     !!            pa_exp(k,ii) = pa_exp(k-1,ii)
     !!         end do
     !!      end do
     !!      pa_exp(1,1:msize) = PA_old(mstart:mstop)
     !!   end if
     !!end if

     ! Note:
     ! although pure dE/dP_ij = F_ji, the idempotency constraint imposes new terms,
     ! which corresponds to Eq.5 of Schlegel et al. JCP 114, 9758.
     ! This new terms make the dE/dP = F*P + P*F - two*P*F*P
     ! after applying the P2=P condition to Eq. 5.
     !
     ! Now, if we use dE/dP = F, the constraint term leaves P*F*P + Q*F*Q to be added
     ! in the Lagrangean multipliers, which equals to dE/dP = F*P + P*F - two*P*F*P
     ! in the end. So, I am using dE/dP = F*P + P*F - two*P*F*P here.
     ! (dE/dP = F - (P*F*P + Q*F*Q))

     ! FP and PF, then PFP, PF = (FP)^T; W2=FP
     ! F_PA already contains all info.
     !#if KEY_PARALLEL==1
     !if(nnumnod>1) call VDGBRE(F_PA,JPARPT_local)
     !#endif
     call make_square2(F_PA,W_new,PA,PA_old_work,dim_norbs,dim_norbs,dim_linear)
     !call make_square(F_PA,W_new,    dim_norbs,dim_norbs,dim_linear) ! W_new <-F
     !call make_square(PA,PA_old_work,dim_norbs,dim_norbs,dim_linear) ! PA_old_work <-P
     jcnt = 0
     do i=1,n
        do j=1,n
           jcnt = jcnt+1
           if(jcnt.ge.jstart .and. jcnt.le.jstop) then
              W2(j,i) = ddot_mn(n,W_new(1:n,j),1,PA_old_work(1:n,i),1)  ! DOT_PRODUCT(W_new(1:n,j),PA_old_work(1:n,i))
           end if
        end do
     end do

#if KEY_PARALLEL==1
     if(nnumnod>1) call VDGBRE(W2,KPARPT_local)
#endif
     ! now, F_PA_ji = W2(j,i) + W2(i,j) - two*(PFP = P*W2)(j,i)
     do ii=mstart,mstop
        i = i_indx(ii)
        j = j_indx(ii)
        F_PA(ii) = W2(j,i)+W2(i,j)-two*ddot_mn(n,PA_old_work(1:n,j),1,W2(1:n,i),1)  ! DOT_PRODUCT(PA_old_work(1:n,j),W2(1:n,i))
     end do

     !
     Kb_T    = KBOLTZ * Temperature
     delta2  = DELTA**2
     r_2delta= one/(two*DELTA)
     r_mass  = one/Mass_PA

!     ! kinetic energy
!     Kinetic_E = DOT_PRODUCT(V_PA(mstart:mstop),V_PA(mstart:mstop))
!     Kinetic_E = Half*Mass_PA*Kinetic_E
!#if KEY_PARALLEL==1
!     if(nnumnod>1) call gcomb(Kinetic_E,1)
!#endif

     if(qm_cpmd_main_r%q_nose) then
        ! kinetic energy
        Kinetic_E = DOT_PRODUCT(V_PA(mstart:mstop),V_PA(mstart:mstop))
        Kinetic_E = Half*Mass_PA*Kinetic_E
#if KEY_PARALLEL==1
        if(nnumnod>1) call gcomb(Kinetic_E,1)
#endif

        do i=1,MaxIter
           ! Nose chain
           X1          = (NOSE_Mass(2)*NOSEV(2)**2 - Kb_T)/NOSE_Mass(3)
           NOSE_New(3) = (two*NOSE(3)-NOSE_Old(3)+delta2*X1)
           X2          = NOSE_New(3)-NOSE_Old(3)

           X1          = (NOSE_Mass(1)*NOSEV(1)**2 - Kb_T)/NOSE_Mass(2)
           NOSE_New(2) = (two*NOSE(2)-NOSE_Old(2)*(one-PT25*X2)+delta2*X1)/(one+PT25*X2)
           X2          = NOSE_New(2)-NOSE_Old(2)
           NOSEV(2)    = X2*r_2delta  ! /(two*DELTA)

           X1          = (two*Kinetic_E - Kb_T*real(dim_linear))/NOSE_Mass(1)
           NOSE_New(1) = (two*NOSE(1)-NOSE_Old(1)*(one-PT25*X2)+delta2*X1)/(one+PT25*X2)
           X2          = NOSE_New(1)-NOSE_Old(1)
           NOSEV(1)    = X2*r_2delta  ! /(two*DELTA)

           KIN_Old = Kinetic_E

           ! update PA_value
           rtmp_n    = (ONE-PT25*X2)
           rtmp_p    = one/(ONE+PT25*X2)
           Kinetic_E = zero
           do j=mstart,mstop  ! 1,dim_linear
              PA_new(j) =(two*PA(j) - PA_old(j)*rtmp_n - delta2*F_PA(j)*r_mass)*rtmp_p   ! F_PA was gradient
              V_PA(j)   =(PA_new(j) - PA_old(j))*r_2delta
              Kinetic_E = Kinetic_E + V_PA(j)*V_PA(j)
           end do
           Kinetic_E = Half*Mass_PA*Kinetic_E
#if KEY_PARALLEL==1
           if(nnumnod>1) call gcomb(Kinetic_E,1)
#endif
           if(ABS(KIN_Old-Kinetic_E) .lt. ThresConv) exit
        end do

        do i=1,3
           NOSE_Old(i)=Nose(i)
           Nose(i)    =NOSE_New(i)
        end do
     else

        ! as F_PA is dE/dP_ji computed above; where F_PA was gradient.
        !!if(q_do_diss) then
        !!   ii = 0
        !!   ks = qm_cpmd_main_r%K_sum_order
        !!   do j=mstart,mstop  ! 1,dim_linear
        !!      ii = ii + 1   ! counter
        !!      PA_new(j) = cextr(0)*PA(j) + dot_product(cextr(1:ks),pa_exp(1:ks,ii)) - delta2*F_PA(j)*r_mass
        !!      V_PA(j)   =(PA_new(j) - PA_old(j))*r_2delta
        !!   end do
        !!else
           do j=mstart,mstop  ! 1,dim_linear
              PA_new(j) =(two*PA(j) - PA_old(j) - delta2*F_PA(j)*r_mass)  ! F_PA was gradient
              V_PA(j)   =(PA_new(j) - PA_old(j))*r_2delta
           end do
        !!end if
     end if

     ! kinetic energy printing
     if(q_print) then
        Kinetic_E = DOT_PRODUCT(V_PA(mstart:mstop),V_PA(mstart:mstop))
        Kinetic_E = Half*Mass_PA*Kinetic_E
#if KEY_PARALLEL==1
        if(nnumnod>1) call gcomb(Kinetic_E,1)
        if(mynod.eq.0)  & 
#endif
        qm_control_r%E_total = qm_control_r%E_total + Kinetic_E

        if(prnlev.ge.2) &
        write(6,*)'Kinetic E:',Kinetic_E,' Temperature:',two*Kinetic_E/(float(dim_linear)*KBOLTZ)
     end if

     ! returned values
     PA_old(1:dim_linear) = PA(1:dim_linear)  ! to avoid communication.
     PA(mstart:mstop)     = PA_new(mstart:mstop)
#if KEY_PARALLEL==1
     if(nnumnod>1) call VDGBRE(PA,JPARPT_local)
#endif

     ! also note that PA_old_work is a square form of PA (now is PA_old). So, in the
     ! density_const, we can avoid one additional call of make_square.

  else if(qm_cpmd_main_r%q_curvy_md) then
     ! memory
     if(.not.associated(PAwork))      allocate(PAwork(dim_norbs,dim_norbs))
     if(.not.associated(PQwork))      allocate(PQwork(dim_norbs,dim_norbs))
     if(.not.associated(PA_old_work)) allocate(PA_old_work(dim_norbs,dim_norbs))
     if(.not.associated(PA2))         allocate(PA2(dim_norbs,dim_norbs))
     if(.not.associated(PA_local))    allocate(PA_local(dim_norbs))
     if(.not.associated(W_new))       allocate(W_new(dim_norbs,dim_norbs))
     if(.not.associated(D_new))       allocate(D_new(dim_norbs,dim_norbs))
     if(.not.associated(dF_linear))   allocate(dF_linear(dim_linear))
     if(.not.associated(vPA_new))     allocate(vPA_new(dim_linear))
#if KEY_PARALLEL==1
     if(.not.associated(PA_check))    allocate(PA_check(nnumnod))
#endif

     !
     ! sequence of events (based on leapfrog verlet algorithm):
     ! 1. at t, compute G_cur(t).
     ! 2. v_new(t+dt/2)   = v_old(t-dt/2) - G_cur*dt/m
     ! 3. delta_new(t+dt) = dt*v_new(t+dt/2)
     ! 4. if needed, v_cur(t) = half*(v_old(t-dt/2)+v_new(t+dt/2))
     ! 5. update P
     !    P_new = exp(delta_new)*P_old*exp(-delta_new)
     ! ... now, t < t+dt
     ! 6. v_old <- v_new
     !
     ! where G(t) = dE/ddelta = FP - PF, for i>j as G_ii=zero. And, m: mass.
     !
     ! in the update
     ! P(t+dt) = exp(delta(t+dt)*P(t)*exp(-delta(t+dt))
     !
     !           using Baker-Hausdorff expansion
     !         = P(t)+ 1/1![P(t),-delta(t+dt)] + 1/2![ [P(t),-delta(t+dt)],-delta(t+dt) ]
     !               + ...
     ! where
     ! [P(t),-delta(t+dt)] =P*(-delta) - (-delta)*P = P*(-delta) + (P*(-delta))^T
     !
     ! 1. compute gradient component: G = F*P - P*F

     n          = dim_norbs
     delta_local= DELTA
     dt_m       = delta_local/Mass_PA  ! (dt/m)

#if KEY_PARALLEL==1
     if(nnumnod>1) call VDGBRE(F_PA,JPARPT_local)
#endif
     call make_square2(F_PA,W_new,PA,PA_old_work,dim_norbs,dim_norbs,dim_linear)
     !call make_square(F_PA,W_new,    dim_norbs,dim_norbs,dim_linear)
     !call make_square(PA,PA_old_work,dim_norbs,dim_norbs,dim_linear)
     do ii=mstart,mstop
        i = i_indx(ii)
        j = j_indx(ii)
        if(i.ne.j) then
           !dF_linear(ii)= DOT_PRODUCT(W_new(1:n,j),PA_old_work(1:n,i)) - &
           !               DOT_PRODUCT(PA_old_work(1:n,j),W_new(1:n,i))
           !dF_linear(ii)= ddot_mn(n,W_new(1:n,j),1,PA_old_work(1:n,i),1) - &
           !               ddot_mn(n,PA_old_work(1:n,j),1,W_new(1:n,i),1)
           dF_linear(ii)= ddot2d_mn(n,W_new(1:n,j),PA_old_work(1:n,j),1,PA_old_work(1:n,i),W_new(1:n,i),1,.false.)
        else
           dF_linear(ii)= zero
        end if
     end do

     ! 2. vPA_new = V_PA_old - G*dt/m: V from t-dt/2 to t + dt/2
     if(qm_cpmd_main_r%q_langevin) then
        jcnt = 0
        do ii=mstart,mstop
           jcnt = jcnt + 1
           i    = i_indx(ii)
           j    = j_indx(ii)
           if(j.ne.i) then
              ! this is original with white noise.
              !vPA_new(ii) = qm_cpmd_main_r%gamma_delta(3)*V_PA(ii) &
              !             -qm_cpmd_main_r%gamma_delta(2)*( dF_linear(ii) &
              !                                             -FRAND(jcnt) )
              ! just put no random force (no white noise, which is heating up)..
              vPA_new(ii) = qm_cpmd_main_r%gamma_delta(3)*V_PA(ii) &
                           -qm_cpmd_main_r%gamma_delta(2)*dF_linear(ii) 
           else
              vPA_new(ii) = V_PA(ii) ! -dF_linear(ii)*dt_m
           end if
        end do
     else
        vPA_new(mstart:mstop)=V_PA(mstart:mstop)-dF_linear(mstart:mstop)*dt_m
     end if

     ! 3. delta = dt*vPA_new
     ! as delta(t)=zero always. (see Herbert & MHG, JCP 2004, 121, 11542.)
     PA_new(mstart:mstop) = delta_local*vPA_new(mstart:mstop)
#if KEY_PARALLEL==1
     if(nnumnod>1) call VDGBRE(PA_new,JPARPT_local)
#endif

     ! make it square, but D = -D^T. 
     ! Note: Since below we have to take the negative of D, we take a negative here.
     icnt=0
     do i=1,n
        do j=1,i
           icnt       = icnt+1
           D_new(j,i) = -PA_new(icnt)
           D_new(i,j) =  PA_new(icnt)  ! -D_new(j,i)

           ! W_new = D_new^T (transpose of D_new).
           W_new(j,i) =  PA_new(icnt)  ! =D_new(i,j)
           W_new(i,j) = -PA_new(icnt)  ! =D_new(j,i)
        end do
     end do
     !! W_new = D_new^T (transpose of D_new).
     !W_new(1:n,1:n) = TRANSPOSE(D_new(1:n,1:n))

     ! 4. kinetic energy printing: unit should be changed.
     if(q_print) then
        ! V_PA is V_cur(t)
        if(qm_cpmd_main_r%q_langevin) then
           V_PA(mstart:mstop) = qm_cpmd_main_r%gamma_delta(4)*(V_PA(mstart:mstop)+vPA_new(mstart:mstop))
        else
           V_PA(mstart:mstop) = half*(V_PA(mstart:mstop)+vPA_new(mstart:mstop))
        end if
     
        Kinetic_E = DOT_PRODUCT(V_PA(mstart:mstop),V_PA(mstart:mstop))
        Kinetic_E = Half*Mass_PA*Kinetic_E
#if KEY_PARALLEL==1
        if(nnumnod>1) call gcomb(Kinetic_E,1)
        if(mynod.eq.0)  & 
#endif
        qm_control_r%E_total = qm_control_r%E_total + Kinetic_E
#if KEY_PARALLEL==1
        if(prnlev.ge.2) &
#endif
        write(6,*)'Kinetic E:',Kinetic_E, &
                  ' Temperature:',two*Kinetic_E/(float(dim_linear)*KBOLTZ)
     end if

     ! 6. V_PA <- vPA_new
     V_PA(mstart:mstop) = vPA_new(mstart:mstop)

     ! 5. update P
     ! for ii=1
     !D_new = -D_new  ! <- this is taken cared above.
     !PQwork(1:n,1:n) = MATMUL(PA_old_work,D_new) - MATMUL(D_new,PA_old_work)
     !PAwork(1:n,1:n) = PA_old_work(1:n,1:n) + PQwork(1:n,1:n)
     !PA_old_work(1:n,1:n) = PQwork(1:n,1:n)
     !
     do ii=mstart,mstop
        i = i_indx(ii)
        j = j_indx(ii)
        !dF_linear(ii) = DOT_PRODUCT(PA_old_work(1:n,j),D_new(1:n,i)) &
        !               -DOT_PRODUCT(W_new(1:n,j),PA_old_work(1:n,i))
        !dF_linear(ii) = ddot_mn(n,PA_old_work(1:n,j),1,D_new(1:n,i),1) &
        !               -ddot_mn(n,W_new(1:n,j),1,PA_old_work(1:n,i),1)
        dF_linear(ii) = ddot2d_mn(n,PA_old_work(1:n,j),W_new(1:n,j),1,D_new(1:n,i),PA_old_work(1:n,i),1,.false.)
        PAwork(j,i) = dF_linear(ii)
     end do
#if KEY_PARALLEL==1
     if(nnumnod>1) call VDGBRE(dF_linear,JPARPT_local)
#endif
     call make_square(dF_linear,PQwork,dim_norbs,dim_norbs,dim_linear)

     do ii=2,max_cycle
        !PQwork(1:n,1:n) = MATMUL(PQwork,D_new) - MATMUL(D_new,PQwork)
        !PAwork(1:n,1:n) = PAwork(1:n,1:n) + r_fact(ii)*PQwork(1:n,1:n)
        r_check =zero
        !sum_e   = zero
        do jj=mstart,mstop
           i = i_indx(jj)
           j = j_indx(jj)
           !dF_linear(jj) = DOT_PRODUCT(PQwork(1:n,j),D_new(1:n,i)) &
           !               -DOT_PRODUCT(W_new(1:n,j),PQwork(1:n,i))
           !dF_linear(jj) = ddot_mn(n,PQwork(1:n,j),1,D_new(1:n,i),1) &
           !               -ddot_mn(n,W_new(1:n,j),1,PQwork(1:n,i),1)
           dF_linear(ii) = ddot2d_mn(n,PQwork(1:n,j),W_new(1:n,j),1,D_new(1:n,i),PQwork(1:n,i),1,.false.)
           rtmp_p      = r_fact(ii)*dF_linear(jj)
           PAwork(j,i) = PAwork(j,i) + rtmp_p
           if(abs(rtmp_p)>r_check) r_check = abs(rtmp_p)  ! find maximum
           !if(i.eq.j) sum_e = sum_e + PAwork(i,i)
        end do
        !
        !------------------CHECK--------------------!
        if(ii > max_inner_cycle) then
#if KEY_PARALLEL==1
           !find maximum over entire nodes.
           if(nnumnod>1) then
              PA_check(1:nnumnod) = zero
              PA_check(nmynod+1)  = r_check
              call gcomb(PA_check,nnumnod)
              r_check = PA_check(1)
              do i=2,nnumnod
                 if(PA_check(i) > r_check) r_check = PA_check(i)
              end do
           end if
#endif
           if(r_check .le. ThresDens) exit
           ! this is not needed as it should preserve the idempotency.
           !if(r_check .le. ThresDens) then
           !#if KEY_PARALLEL==1
           !   if(nnumnod>1) call gcomb(sum_e,1)
           !#endif
           !   if(abs(sum_e-float(qm_main_r%nalpha)) .le. ThresElec) exit
           !end if
        end if
        !------------------CHECK--------------------!
        !
#if KEY_PARALLEL==1
        if(nnumnod>1) call VDGBRE(dF_linear,JPARPT_local)
#endif
        call make_square(dF_linear,PQwork,dim_norbs,dim_norbs,dim_linear)
     end do

     ! returned values: PA_new = PA_old + PAwork
     do ii=mstart,mstop
        PA(ii) = PA(ii) + PAwork(j_indx(ii),i_indx(ii))
     end do
     !icnt = 0
     !do i=1,n
     !   do j=1,i
     !      icnt = icnt + 1
     !      if(icnt.ge.mstart .and. icnt.le.mstop) PA(icnt) = PA(icnt) + PAwork(j,i)
     !   end do
     !end do
#if KEY_PARALLEL==1
     if(nnumnod>1) call VDGBRE(PA,JPARPT_local)
#endif

     if(.not.q_print) return

     !------------------CHECK--------------------!
     call make_square(PA,PAwork,dim_norbs,dim_norbs,dim_linear)
     ! check if it is converged: SQRT{Tr [ (P*P - P)^2 ]}/N < ThresConv.
     do ii=mstart,mstop
        i = i_indx(ii)
        j = j_indx(ii)
        dF_linear(ii)=PAwork(j,i)-ddot_mn(n,PAwork(1:n,i),1,PAwork(1:n,j),1)  ! DOT_PRODUCT(PAwork(1:n,i),PAwork(1:n,j))
     end do
#if KEY_PARALLEL==1
     if(nnumnod>1) call VDGBRE(dF_linear,JPARPT_local)
#endif
     call make_square(dF_linear,PA2,dim_norbs,dim_norbs,dim_linear)
     sum_p =zero
     do i=nmynod+1,n,nnumnod   ! 1,n
        sum_p = sum_p + ddot_mn(n,PA2(1:n,i),1,PA2(1:n,i),1)  ! DOT_PRODUCT(PA2(1:n,i),PA2(1:n,i))
     end do
#if KEY_PARALLEL==1
     if(nnumnod>1) call gcomb(sum_p,1)
#endif
     sum_p = sqrt(sum_p)/float(n)
     if(prnlev.ge.2.and.abs(sum_p)>ThresConv) write(6,*) 'idempotency:',sum_p
     !------------------CHECK--------------------!
  end if
  return
  end subroutine update_PA


  subroutine update_PA_old(PA,PA_new,PA_old,F_PA,V_PA,Mass_PA,Delta,dim_linear,dim_norbs)
  !
  ! MTS-LN-Curvy or MTS-ADMP
  ! update based on curvy slow-varying density force ( (FP-PF)scf(t) )
  !                 ADMP  slow-varying density force ( (F)_scf(t) )
  !
  ! On return,
  ! 1. ADMP case: P_old_scf(t-dt) = (N-1)/N*P_scf(t) +1/N*P_old_scf(t-Dt) - (1-N)*dt^2/2m*Fock(t)
  !                                + lagrangian component (done by calling density_const routine.)
  ! where Dt = N*dt
  !
  use chm_kinds
  use number
  use consta, only : KBOLTZ,TIMFAC
#if KEY_PARALLEL==1
  use parallel
#endif
  use stream, only : prnlev
  use qm1_info, only : qm_control_r,qm_main_r

  implicit none
  integer        :: dim_linear,dim_norbs
  real(chm_real) :: PA(*),PA_new(*),PA_old(*),F_PA(*),V_PA(*),Mass_PA,Delta

  !
  integer,parameter :: MaxIter=1000,max_cycle=20,max_inner_cycle=5 ! 10
  real(chm_real),parameter :: ThresConv=1.0D-12,ThresElec =1.0D-12, &
                              ThresDens=1.0D-15 ! 1.0D-30
  real(chm_real),save :: r_fact(max_cycle)
  integer :: i,j,k,ii,jj,icnt,jcnt,ks,n,linear_nsize
  real(chm_real) :: KIN_Old,X1,X2,Kb_T,Kinetic_E,     &
                    rtmp_n,rtmp_p,r_check,delta2,r_2delta,r_mass, &
                    dt_m,sum_p,sum_e,delta_local
  real(chm_real) :: e_conv,t_conv,m_conv,mass_local
  integer, save  :: nnumnod,nmynod
  integer, save  :: size_n=0, imd_counter=0
  integer, save  :: mstart,mstop,msize,jstart,jstop
#if KEY_PARALLEL==1
  integer, save  :: JPARPT_local(0:MAXNODE),KPARPT_local(0:MAXNODE)
#endif
  real(chm_real):: ddot_mn,ddot2d_mn ! external function
  real(chm_real),save :: r_dim_linear,r_norbs,r_elec,r_factor_1,r_factor_2,r_factor_3


  ! for now, only works for ADMP.
  if(qm_cpmd_main_r%q_curvy_md)then
     call wrndie(-5,'<update_PA_old>','Updating PA_old is not implemented for Curvy-MD.')
     return
  end if

  ! for now, this is a copy of update_PA. So, memory will be deallocated/allocated again in update_PA.
  if(size_n .ne. dim_linear) then
     size_n = dim_linear
     mstart = 1
     mstop  = dim_linear
     msize  = dim_linear

     jstart = 1
     jstop  = dim_norbs*dim_norbs
     icnt   = dim_norbs*dim_norbs
#if KEY_PARALLEL==1
     nnumnod = numnod
     JPARPT_local(0) = 0
     do i=1,nnumnod
        JPARPT_local(i)= dim_linear*i/nnumnod  ! for linear vector
     end do
     mstart = JPARPT_local(mynod)+1
     mstop  = JPARPT_local(mynod+1)
     msize  = mstop - mstart + 1

     KPARPT_local(0)=0
     do i=1,nnumnod
        KPARPT_local(i)= icnt*i/nnumnod        ! for square vector
     end do
     jstart = KPARPT_local(mynod)+1
     jstop  = KPARPT_local(mynod+1)
#endif

     ! prepare i_indx and j_indx, which contain the i and j indices for do looping
     ! over lower or upper triangle in matrix handling. This is done here as it
     ! is used many time during the CPMD calculations.
     ij_pair_cnt = dim_linear ! mstop - mstart + 1
     if(associated(i_indx)) deallocate(i_indx)
     if(associated(j_indx)) deallocate(j_indx)
     if(.not.associated(i_indx)) allocate(i_indx(ij_pair_cnt))
     if(.not.associated(j_indx)) allocate(j_indx(ij_pair_cnt))
     icnt = 0
     ii   = 0
     do i=1,dim_norbs
        do j=1,i
           icnt=icnt+1
           ii = ii + 1
           if(icnt.ge.mstart .and. icnt.le.mstop) then 
              i_indx(ii) = i
              j_indx(ii) = j
           end if
        end do
     end do

     if(qm_cpmd_main_r%q_admp_md) then
        if(associated(PA_old_work)) deallocate(PA_old_work)
        if(associated(W_new))       deallocate(W_new)
        if(associated(W2))          deallocate(W2)
        if(associated(dF_linear))   deallocate(dF_linear)

     else if(qm_cpmd_main_r%q_curvy_md) then
        size_norbs = dim_norbs
        if(associated(PAwork))      deallocate(PAwork)
        if(associated(PQwork))      deallocate(PQwork)
        if(associated(PA_old_work)) deallocate(PA_old_work)
        if(associated(PA2))         deallocate(PA2)
        if(associated(PA_local))    deallocate(PA_local)
        if(associated(W_new))       deallocate(W_new)
        if(associated(D_new))       deallocate(D_new)
        if(associated(dF_linear))   deallocate(dF_linear)
        if(associated(vPA_new))     deallocate(vPA_new)
#if KEY_PARALLEL==1
        if(associated(PA_check))    deallocate(PA_check)
#endif

        r_fact(1) = one
        sum_e= one
        do i=2,max_cycle
           sum_e= sum_e + one
           r_fact(i) =r_fact(i-1)*sum_e
        end do
        ! inverse of factorial
        r_fact(1:max_cycle) = one/r_fact(1:max_cycle)

        r_dim_linear = one/float(dim_linear)
        r_norbs      = one/float(dim_norbs)
        r_elec       = float(qm_main_r%nalpha)
     end if
  end if

  if(qm_cpmd_main_r%q_admp_md) then
     ! allocate memory
     n = dim_norbs
     if(.not.associated(PA_old_work)) allocate(PA_old_work(dim_norbs,dim_norbs))
     if(.not.associated(W_new))       allocate(W_new(dim_norbs,dim_norbs))
     if(.not.associated(W2))          allocate(W2(dim_norbs,dim_norbs))
     if(.not.associated(dF_linear))   allocate(dF_linear(dim_linear))

     r_factor_1 = float(qm_cpmd_main_r%num_iter-1)/float(qm_cpmd_main_r%num_iter)
     r_factor_2 = one/float(qm_cpmd_main_r%num_iter)
     r_factor_3 = half*float(qm_cpmd_main_r%num_iter-1)*(DELTA**2)/Mass_PA

     ! Note:
     ! although pure dE/dP_ij = F_ji, the idempotency constraint imposes new terms,
     ! which corresponds to Eq.5 of Schlegel et al. JCP 114, 9758.
     ! This new terms make the dE/dP = F*P + P*F - two*P*F*P
     ! after applying the P2=P condition to Eq. 5.
     !
     ! Now, if we use dE/dP = F, the constraint term leaves P*F*P + Q*F*Q to be added
     ! in the Lagrangean multipliers, which equals to dE/dP = F*P + P*F - two*P*F*P
     ! in the end. So, I am using dE/dP = F*P + P*F - two*P*F*P here.
     ! (dE/dP = F - (P*F*P + Q*F*Q))

     ! FP and PF, then PFP, PF = (FP)^T; W2=FP, using PA(t).
     ! F_PA already contains all info.
     call make_square2(F_PA,W_new,PA,PA_old_work,dim_norbs,dim_norbs,dim_linear)
     jcnt = 0
     do i=1,n
        do j=1,n
           jcnt = jcnt+1
           if(jcnt.ge.jstart .and. jcnt.le.jstop) then
              W2(j,i) = ddot_mn(n,W_new(1:n,j),1,PA_old_work(1:n,i),1)  ! DOT_PRODUCT(W_new(1:n,j),PA_old_work(1:n,i))
           end if
        end do
     end do
#if KEY_PARALLEL==1
     if(nnumnod>1) call VDGBRE(W2,KPARPT_local)
#endif
     ! now, F_PA_ji = W2(j,i) + W2(i,j) - two*(PFP = P*W2)(j,i)
     do ii=mstart,mstop
        i = i_indx(ii)
        j = j_indx(ii)
        dF_linear(ii) = W2(j,i)+W2(i,j)-two*ddot_mn(n,PA_old_work(1:n,j),1,W2(1:n,i),1) ! DOT_PRODUCT(PA_old_work(1:n,j),W2(1:n,i))
     end do

     ! as F_PA is dE/dP_ji computed above; where F_PA was gradient.
     do j=mstart,mstop  ! 1,dim_linear
        PA_old(j) = r_factor_1*PA(j) + r_factor_2*PA_old(j) - r_factor_3*dF_linear(j)
     end do
#if KEY_PARALLEL==1
     if(nnumnod>1) call VDGBRE(PA_old,JPARPT_local)
#endif
     !
     ! also note that PA_old_work is a square form of PA. So, in the
     ! density_const, we can avoid one additional call of make_square.

     ! to be used in later upate_PA: F_PA is not changed.
     !if(associated(dF_linear))   deallocate(dF_linear)

  else if(qm_cpmd_main_r%q_curvy_md) then
     ! memory
     if(.not.associated(PAwork))      allocate(PAwork(dim_norbs,dim_norbs))
     if(.not.associated(PQwork))      allocate(PQwork(dim_norbs,dim_norbs))
     if(.not.associated(PA_old_work)) allocate(PA_old_work(dim_norbs,dim_norbs))
     if(.not.associated(PA2))         allocate(PA2(dim_norbs,dim_norbs))
     if(.not.associated(PA_local))    allocate(PA_local(dim_norbs))
     if(.not.associated(W_new))       allocate(W_new(dim_norbs,dim_norbs))
     if(.not.associated(D_new))       allocate(D_new(dim_norbs,dim_norbs))
     if(.not.associated(dF_linear))   allocate(dF_linear(dim_linear))
     if(.not.associated(vPA_new))     allocate(vPA_new(dim_linear))
#if KEY_PARALLEL==1
     if(.not.associated(PA_check))    allocate(PA_check(nnumnod))
#endif

     !
     ! sequence of events (based on leapfrog verlet algorithm):
     ! 1. at t, compute G_cur(t).
     ! 2. v_new(t+dt/2)   = v_old(t-dt/2) - G_cur*dt/m
     ! 3. delta_new(t+dt) = dt*v_new(t+dt/2)
     ! 4. if needed, v_cur(t) = half*(v_old(t-dt/2)+v_new(t+dt/2))
     ! 5. update P
     !    P_new = exp(delta_new)*P_old*exp(-delta_new)
     ! ... now, t < t+dt
     ! 6. v_old <- v_new
     !
     ! where G(t) = dE/ddelta = FP - PF, for i>j as G_ii=zero. And, m: mass.
     !
     ! in the update
     ! P(t+dt) = exp(delta(t+dt)*P(t)*exp(-delta(t+dt))
     !
     !           using Baker-Hausdorff expansion
     !         = P(t)+ 1/1![P(t),-delta(t+dt)] + 1/2![ [P(t),-delta(t+dt)],-delta(t+dt) ]
     !               + ...
     ! where
     ! [P(t),-delta(t+dt)] =P*(-delta) - (-delta)*P = P*(-delta) + (P*(-delta))^T
     !
     ! 1. compute gradient component: G = F*P - P*F

     n          = dim_norbs
     delta_local= DELTA
     dt_m       = delta_local/Mass_PA  ! (dt/m)

#if KEY_PARALLEL==1
     if(nnumnod>1) call VDGBRE(F_PA,JPARPT_local)
#endif
     call make_square2(F_PA,W_new,PA,PA_old_work,dim_norbs,dim_norbs,dim_linear)
     do ii=mstart,mstop
        i = i_indx(ii)
        j = j_indx(ii)
        if(i.ne.j) then
           !dF_linear(ii)= DOT_PRODUCT(W_new(1:n,j),PA_old_work(1:n,i)) - &
           !               DOT_PRODUCT(PA_old_work(1:n,j),W_new(1:n,i))
           !dF_linear(ii)= ddot_mn(n,W_new(1:n,j),1,PA_old_work(1:n,i),1) - &
           !               ddot_mn(n,PA_old_work(1:n,j),1,W_new(1:n,i),1)
           dF_linear(ii)= ddot2d_mn(n,W_new(1:n,j),PA_old_work(1:n,j),1,PA_old_work(1:n,i),W_new(1:n,i),1,.false.)
        else
           dF_linear(ii)= zero
        end if
     end do

     ! 2. vPA_new = V_PA_old - G*dt/m: V from t-dt/2 to t + dt/2
     vPA_new(mstart:mstop)=V_PA(mstart:mstop)-dF_linear(mstart:mstop)*dt_m

     ! 3. delta = dt*vPA_new
     ! as delta(t)=zero always. (see Herbert & MHG, JCP 2004, 121, 11542.)
     PA_new(mstart:mstop) = delta_local*vPA_new(mstart:mstop)
#if KEY_PARALLEL==1
     if(nnumnod>1) call VDGBRE(PA_new,JPARPT_local)
#endif

     ! make it square, but D = -D^T. 
     ! Note: Since below we have to take the negative of D, we take a negative here.
     icnt=0
     do i=1,n
        do j=1,i
           icnt       = icnt+1
           D_new(j,i) = -PA_new(icnt)
           D_new(i,j) =  PA_new(icnt)  ! -D_new(j,i)

           ! W_new = D_new^T (transpose of D_new).
           W_new(j,i) =  PA_new(icnt)  ! =D_new(i,j)
           W_new(i,j) = -PA_new(icnt)  ! =D_new(j,i)
        end do
     end do
     !! W_new = D_new^T (transpose of D_new).
     !W_new(1:n,1:n) = TRANSPOSE(D_new(1:n,1:n))

     ! 6. V_PA <- vPA_new
     V_PA(mstart:mstop) = vPA_new(mstart:mstop)

     ! 5. update P
     ! for ii=1
     !D_new = -D_new  ! <- this is taken cared above.
     !PQwork(1:n,1:n) = MATMUL(PA_old_work,D_new) - MATMUL(D_new,PA_old_work)
     !PAwork(1:n,1:n) = PA_old_work(1:n,1:n) + PQwork(1:n,1:n)
     !PA_old_work(1:n,1:n) = PQwork(1:n,1:n)
     !
     do ii=mstart,mstop
        i = i_indx(ii)
        j = j_indx(ii)
        !dF_linear(ii) = DOT_PRODUCT(PA_old_work(1:n,j),D_new(1:n,i)) &
        !               -DOT_PRODUCT(W_new(1:n,j),PA_old_work(1:n,i))
        !dF_linear(ii) = ddot_mn(n,PA_old_work(1:n,j),1,D_new(1:n,i),1) &
        !               -ddot_mn(n,W_new(1:n,j),1,PA_old_work(1:n,i),1)
        dF_linear(ii) = ddot2d_mn(n,PA_old_work(1:n,j),W_new(1:n,j),1,D_new(1:n,i),PA_old_work(1:n,i),1,.false.)
        PAwork(j,i) = dF_linear(ii)
     end do
#if KEY_PARALLEL==1
     if(nnumnod>1) call VDGBRE(dF_linear,JPARPT_local)
#endif
     call make_square(dF_linear,PQwork,dim_norbs,dim_norbs,dim_linear)

     do ii=2,max_cycle
        !PQwork(1:n,1:n) = MATMUL(PQwork,D_new) - MATMUL(D_new,PQwork)
        !PAwork(1:n,1:n) = PAwork(1:n,1:n) + r_fact(ii)*PQwork(1:n,1:n)
        r_check =zero
        !sum_e   = zero
        do jj=mstart,mstop
           i = i_indx(jj)
           j = j_indx(jj)
           !dF_linear(jj) = DOT_PRODUCT(PQwork(1:n,j),D_new(1:n,i)) &
           !               -DOT_PRODUCT(W_new(1:n,j),PQwork(1:n,i))
           !dF_linear(jj) = ddot_mn(n,PQwork(1:n,j),1,D_new(1:n,i),1) &
           !               -ddot_mn(n,W_new(1:n,j),1,PQwork(1:n,i),1)
           dF_linear(jj) = ddot2d_mn(n,PQwork(1:n,j),W_new(1:n,j),1,D_new(1:n,i),PQwork(1:n,i),1,.false.)
           rtmp_p      = r_fact(ii)*dF_linear(jj)
           PAwork(j,i) = PAwork(j,i) + rtmp_p
           if(abs(rtmp_p)>r_check) r_check = abs(rtmp_p)  ! find maximum
        end do
        !
        !------------------CHECK--------------------!
        if(ii > max_inner_cycle) then
#if KEY_PARALLEL==1
           !find maximum over entire nodes.
           if(nnumnod>1) then
              PA_check(1:nnumnod) = zero
              PA_check(nmynod+1)  = r_check
              call gcomb(PA_check,nnumnod)
              r_check = PA_check(1)
              do i=2,nnumnod
                 if(PA_check(i) > r_check) r_check = PA_check(i)
              end do
           end if
#endif
           if(r_check .le. ThresDens) exit
        end if
        !------------------CHECK--------------------!
        !
#if KEY_PARALLEL==1
        if(nnumnod>1) call VDGBRE(dF_linear,JPARPT_local)
#endif
        call make_square(dF_linear,PQwork,dim_norbs,dim_norbs,dim_linear)
     end do

     ! returned values: PA_new = PA_old + PAwork
     do ii=mstart,mstop
        PA(ii) = PA(ii) + PAwork(j_indx(ii),i_indx(ii))
     end do
#if KEY_PARALLEL==1
     if(nnumnod>1) call VDGBRE(PA,JPARPT_local)
#endif
  end if
  return
  end subroutine update_PA_old


  subroutine make_square(A,B,N,n1,n2)
  ! 
  ! STORE LOWER TRIANGLE MATRIX A(LM4) AS SYMMETRIC SQUARE MATRIX
  ! B(LM2,LM2). A AND B MAY OCCUPY THE SAME CORE SPACE.
  ! N ROWS AND COLUMNS ARE COPIED.
  ! 
  use chm_kinds

  implicit none
  integer :: N,n2,n1
  real(chm_real):: A(n2),B(n1,n1)

  ! local variables
  integer :: icnt,I,J,jj

  icnt=0
  do i=1,n
     do j=1,i
        icnt   = icnt+1
        b(j,i) = A(icnt)
        b(i,j) = A(icnt)
     end do
  end do
  !do i=2,n
  !   do j=1,i-1
  !      b(i,j)=b(j,i)
  !   end do
  !end do
  return
  end subroutine make_square

  subroutine make_square2(A,B,C,D,N,n1,n2)
  !
  ! STORE LOWER TRIANGLE MATRIX A(LM4) AS SYMMETRIC SQUARE MATRIX
  ! B(LM2,LM2). A AND B MAY OCCUPY THE SAME CORE SPACE.
  ! N ROWS AND COLUMNS ARE COPIED.
  !
  use chm_kinds

  implicit none
  integer :: N,n2,n1
  real(chm_real):: A(n2),B(n1,n1),C(n2),D(n1,n1)

  ! local variables
  integer :: icnt,I,J,jj

  icnt=0
  do i=1,n
     do j=1,i
        icnt   = icnt+1
        b(j,i) = A(icnt)
        b(i,j) = A(icnt)
        d(j,i) = C(icnt)
        d(i,j) = C(icnt)
     end do
  end do
  !do i=2,n
  !   do j=1,i-1
  !      b(i,j)=b(j,i)
  !      d(i,j)=d(j,i)
  !   end do
  !end do
  return
  end subroutine make_square2


  !-----------------------------------------------------------------------!
  !----------------- Langevin Dynamics Part (Thermostat) -----------------!
  subroutine LNGFIL_PA(DELTA,n_size)
  !
  ! This routine fills the GAMMALD arrays for Langevin dynamics.
  !
  ! This routine needs to be called only once!
  !
  ! gamma_delta(i,1) - RDF (std.dev. of random force)
  ! gamma_delta(i,2) - BETA  ( dx scale factor)
  ! gamma_delta(i,3) - ALPHA ( x-xold scale factor)
  ! gamma_delta(i,4) - Velocity compute scale factor
  !
  ! Charles Brooks III and Axel Brunger, 3-JULY-1983
  ! Modified for a more accurate integration
  ! Bernard R. Brooks  December,1987
  !
  ! input/output
  use chm_kinds
  use number
  !
  use euler
  use consta, only : KBOLTZ,TIMFAC
  use stream
  implicit none

  integer        :: n_size  ! size of matrix vector. for numnod=1, n_size=dim_linear
  real(chm_real) :: DELTA

  ! local
  integer:: i,ii
  real(chm_real) :: RFD,GAM,Kb_T,r_delta,pmass

  ! check
  if(.not.qm_cpmd_main_r%q_langevin) return

  Kb_T   =Kboltz*qm_cpmd_main_r%Temperature
  r_delta=one/DELTA
  pmass  =qm_cpmd_main_r%Mass_PA

  if(abs(qm_cpmd_main_r%fbeta_delta) > rsmall) then
     gam      = TIMFAC*qm_cpmd_main_r%fbeta_delta*Delta
     rfd      = SQRT(two*pmass*gam*Kb_T) * r_delta
     !
     ! gamma_delta(2) is multiplied by DELTA for next calculation
     ! gamma_delta(4) is divided by DELTA for net calculation.
     qm_cpmd_main_r%gamma_delta(1)= rfd
     qm_cpmd_main_r%gamma_delta(2)= DELTA/( (one+gam*half)*pmass)
     qm_cpmd_main_r%gamma_delta(3)= (one-gam*half)/(one+gam*half)
     qm_cpmd_main_r%gamma_delta(4)= half*SQRT(one+gam*half)
  else
     qm_cpmd_main_r%gamma_delta(1)= zero
     qm_cpmd_main_r%gamma_delta(2)= DELTA/pmass
     qm_cpmd_main_r%gamma_delta(3)= one
     qm_cpmd_main_r%gamma_delta(4)= half
  end if

  if(PRNLEV.GE.2) then
    WRITE(OUTU,9000) qm_cpmd_main_r%Temperature, DELTA, n_size
9000    FORMAT(' LNGFIL_PA: TBATH = ',F12.6,'  DELTA =',F12.6,/, &
               ' LNGFIL_PA: Langevin dynamics setup for ',I7,    &
               ' density variables',/)
  end if
  return
  end subroutine LNGFIL_PA


  subroutine DLNGV_PA(n_size)
  !
  ! This function routine generates 3*PA_ij Gaussian
  ! random deviates of 0.0 mean and standard deviation RF.
  ! The algorithm from Box and Muller.
  !
  ! Bernard R. Brooks   January, 1988
  !
  use chm_kinds
  use clcg_mod,only:random
  use number
  use consta, only : PI
  use energym
  use fourdm
  use rndnum
  use parallel
  use reawri

  implicit none

  integer :: n_size

  ! local
  integer, save :: old_n_size=0
  integer :: ig,i,k
  real(chm_real):: A,B
  real(chm_real),parameter :: PIS = PI

  if(n_size .ne. old_n_size) then
     old_n_size = n_size
     if(associated(FRAND)) deallocate(FRAND)
  end if
  if(.not.associated(FRAND)) allocate(FRAND(n_size))

  if (.not.qoldrng) then     !yw 05-Aug-2008
     ig=1
#if KEY_PARALLEL==1
     ig=mynodp
#endif

     K=0
     do i=1,n_size
        if(k.eq.0) then
           A = qm_cpmd_main_r%gamma_delta(1)*SQRT(mintwo*LOG(RANDOM(IG)))
           B = two*PIS*RANDOM(IG)
           K = 1
           FRAND(i)=A*COS(B)
        else
           K = 0
           FRAND(i)=A*SIN(B)
        end if
     end do
  else
     K=0
     do i=1,n_size
        if(k.eq.0) then
           A = qm_cpmd_main_r%gamma_delta(1)*SQRT(mintwo*LOG(random(iseed))) ! RANDOM(IG)))
           B = two*PIS*random(iseed)  ! RANDOM(IG)
           K = 1
           FRAND(i)=A*COS(B)
        else
           K = 0
           FRAND(i)=A*SIN(B)
        end if
     end do
  end if

  return
  end subroutine DLNGV_PA
  !-----------------------------------------------------------------------!

  !-----------------------------------------------------------------------!
  !----------------- Extended Lagrangian with dissipation ----------------!
  ! AMN Niklasson, JCP (2009) 130:214109                                  !
  ! and Based on code by Guishan Zheng, Harvard Univ. 05/19/2010.         !
  !
  subroutine initcoef_diss(A_scale,IOrder,coefk_local,cextr_local,q_cpmd)
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
  real(chm_real) :: A_scale, kappa, alpha
  logical        :: q_cpmd
  integer :: i

  !if(IOrder.gt.9) then
  !   wrndie(-1,'<initcoef_diss>','The allowed highest extrapolation order is 9.')
  !   return
  !endif

  ! memory and initialization setup.
  !!if(q_cpmd) then
  !!   ! CPMD case.
  !!   qm_cpmd_main_r%K_sum_order = IOrder
  !!   qm_cpmd_main_r%Alpha_scale = A_scale
  !!   if(associated(cextr)) deallocate(cextr)
  !!   allocate(cextr(0:qm_cpmd_main_r%K_sum_order))
  !!else
  !!   ! BOMD case.
  !!   ! cextr_local is cextr ... here
  !!end if

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

  !!if(q_cpmd) then
  !!   ! CPMD case.
  !! 
  !!   ! Here, this is done even to take care of 
  !!   ! P_n+1 = 2P_n - P_n-1 + alpha*sum_k(0~K) c_k*P_n-k
  !!   !    final coefficients
  !!   do i = 0, IOrder
  !!      cextr(i) = C(i,IOrder)*alpha*A_scale
  !!   end do
  !!   coefk = Kappa  ! this is not used.
  !!   !cextr(0) = two - coefk + cextr(0)
  !!   cextr(0) = two + cextr(0)  ! as Kappa is not used (see above the note).
  !!   cextr(1) = cextr(1) - one
  !!else
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
  !!end if

  return
  end subroutine initcoef_diss

  !-----------------------------------------------------------------------!
  !
#endif /*mndo97*/
end module qm1_lagdynamics_module
