module qm1_energy_module
  use chm_kinds
  use number
  use qm1_constant

  integer,save :: numat_local=-1, iqm_mode
  logical,save :: do_am1_pm3=.false.
  logical,save :: do_d_orbitals=.false.
  integer,save,pointer :: ni_local(:)=>Null(),    &
                          iorbs_local(:)=>NUll(), &
                          ia_local(:)=>NUll(),    &
                          ib_local(:)=>NUll(),    &
                          ip_local(:)=>Null(),    &
                          is_local(:)=>NUll(),    &
                          iw_local(:)=>NUll(),    &
                          Jmax_local(:)=>NUll()
  real(chm_real),save,pointer :: ZS_local(:)=>Null(),    &
                                 ZP_local(:)=>Null(),    &
                                 ZD_local(:)=>Null()
  real(chm_real),save,pointer :: BETAS_local(:)=>Null(), &
                                 BETAP_local(:)=>Null(), &
                                 BETAD_local(:)=>Null()
  real(chm_real),save,pointer :: OMEGA_local(:)=>Null(), &
                                 DELTA_local(:)=>Null(), &
                                 CORE_local(:)=>Null()
  real(chm_real),save,pointer :: GSS_local(:)=>Null(),   &
                                 GSP_local(:)=>Null(),   &
                                 GPP_local(:)=>Null(),   &
                                 GP2_local(:)=>Null(),   &
                                 HSP_local(:)=>Null(),   &
                                 HPP_local(:)=>Null()
  real(chm_real),save,pointer :: w_save(:,:)=>Null()
  integer,save,pointer        :: int_ij(:)=>Null(),      &
                                 int_kl(:)=>Null()
  ! for core-core interactiosn
  real(chm_real),save         :: PO_1_mm,PO_2_mm,PO_3_mm,PO_7_mm,PO_9_mm,DD_2_mm,DD_3_mm
  real(chm_real),save,pointer :: ALP_local(:)=>Null(),   &
                                 PO_1(:)=>Null(),        &  ! used in hhpair & other routines.
                                 PO_2(:)=>Null(),        &
                                 PO_3(:)=>Null(),        &
                                 PO_7(:)=>Null(),        &
                                 PO_9(:)=>Null(),        &
                                 DD_2(:)=>Null(),        &
                                 DD_3(:)=>Null()
  integer,save,pointer        :: MALPB_local(:) =>Null()    ! mndo/d specific
  real(chm_real),save,pointer :: ALPB_local(:,:)=>Null()    ! mndo/d specific
  ! for gaussian core-core terms
  integer,save,pointer        :: IMPAR_local(:)=>Null()
  real(chm_real),save,pointer :: GUESS1_local(:,:)=>Null(), &
                                 GUESS2_local(:,:)=>Null(), &
                                 GUESS3_local(:,:)=>Null(), &
                                 GNN_local(:)=>Null()

  !
  ! for loop-up table type calculations.
  ! mmint core-core interaction part.
  logical, save, pointer :: q_unique_atom(:)=>Null()
  integer, save, pointer :: i_index_look_up(:)=>Null()
  real(chm_real),save,pointer :: r_core_val(:)=>Null(), &
                                 coef_core_val(:,:,:,:)=>Null(), &
                                 core_val_shift(:,:)=>Null(), &
                                 core_cut_val(:)=>Null()
  integer, save :: nunique_qm,i_npnt
  real(chm_real),save :: dr_width,r_dr_width,rval_min,rval_max

  ! betaij part.
  logical, save, pointer :: q_unique_pair(:)=>Null()    ! size n*(n-1)
  integer, save, pointer :: ij_pair_look_up(:)=>Null()  ! size n*(n-1), point which array should look.
  integer, save :: iunique_cnt_beta,num_beta_pnt
  real(chm_real),pointer :: r_beta_val(:)=>Null()       ! array of npoint
  real(chm_real),save :: dr_width_beta,r_dr_width_beta,rval_min_beta,rval_max_beta

  type,public :: look_up_beta
     real(chm_real),pointer :: coef_val(:,:,:)=>Null() ! 4 x 14 x npoint
     real(chm_real),pointer :: val_shift(:)=>Null()    ! array of 14
    !! real(chm_real) :: dr_width,r_dr_width,rval_min,rval_max
  end type look_up_beta
  type(look_up_beta), save, pointer :: look_up_beta_r(:)=>Null()

  contains

#if KEY_MNDO97==1 /*mndo97*/
  subroutine QMMM_module_prep
  !
  ! Prepare locally used variables.
  !
  use qm1_info, only : qm_control_r,qm_main_r,qm_scf_main_r
  use qm1_parameters, only : OMEGA,DELTA,CORE,BETAS,BETAP,BETAD,ALP,PO,DD, &
                             ZS,ZP,ZD,GSS,GPP,GSP,GP2,HSP,HPP,REPD, &
                             INTIJ,INTKL,INTREP,INTRF1,INTRF2, &
                             MALPB,ALPB,IMPAR,GUESS1,GUESS2,GUESS3,GNN
  implicit none

  integer :: i,j,ii,ni,nj,ia,iorbs,int1,int2
  real(chm_real):: rf

  ! check if memory has to be allocated.
  if(numat_local.ne.qm_main_r%numat) then
     numat_local=qm_main_r%numat

     if(associated(ni_local)) then
        deallocate(ni_local)
        deallocate(iorbs_local)
        deallocate(ia_local)
        deallocate(ib_local)
        deallocate(ip_local)
        deallocate(is_local)
        deallocate(iw_local)
        deallocate(Jmax_local)
        deallocate(BETAS_local)
        deallocate(BETAP_local)
        deallocate(OMEGA_local)
        deallocate(DELTA_local)
        deallocate(CORE_local)
        deallocate(GSS_local)
        deallocate(GSP_local)
        deallocate(GPP_local)
        deallocate(GP2_local)
        deallocate(HSP_local)
        deallocate(HPP_local)
        deallocate(w_save)
        deallocate(ALP_local)
        deallocate(ZS_local)
        deallocate(ZP_local)
        deallocate(PO_1)
        deallocate(PO_2)
        deallocate(PO_3)
        deallocate(PO_7)
        deallocate(PO_9)
        deallocate(DD_2)
        deallocate(DD_3)
     end if
     if(associated(BETAD_local)) deallocate(BETAD_local)
     if(associated(ZD_local))    deallocate(ZD_local)
     if(associated(w_save)) then
        deallocate(w_save)
        deallocate(int_ij)
        deallocate(int_kl)
     end if
     if(associated(MALPB_local)) deallocate(MALPB_local)
     if(associated(ALPB_local))  deallocate(ALPB_local)
     if(associated(IMPAR_local)) then
        deallocate(IMPAR_local)
        deallocate(GUESS1_local)
        deallocate(GUESS2_local)
        deallocate(GUESS3_local)
        deallocate(GNN_local)
     end if

     ! now allocate memory
     allocate(ni_local(qm_main_r%numat))
     allocate(iorbs_local(qm_main_r%numat))
     allocate(ia_local(qm_main_r%numat))
     allocate(ib_local(qm_main_r%numat))
     allocate(ip_local(qm_main_r%numat))
     allocate(is_local(qm_main_r%numat))
     allocate(iw_local(qm_main_r%numat))
     allocate(Jmax_local(qm_main_r%numat))
     allocate(BETAS_local(qm_main_r%numat))
     allocate(BETAP_local(qm_main_r%numat))
     allocate(OMEGA_local(qm_main_r%numat))
     allocate(DELTA_local(qm_main_r%numat))
     allocate(CORE_local(qm_main_r%numat))
     allocate(GSS_local(qm_main_r%numat))
     allocate(GSP_local(qm_main_r%numat))
     allocate(GPP_local(qm_main_r%numat))
     allocate(GP2_local(qm_main_r%numat))
     allocate(HSP_local(qm_main_r%numat))
     allocate(HPP_local(qm_main_r%numat))
     if(qm_control_r%do_d_orbitals) then
        allocate(w_save(243,qm_main_r%numat))
        allocate(int_ij(243))
        allocate(int_kl(243))
     end if
     allocate(ALP_local(qm_main_r%numat))
     allocate(ZS_local(qm_main_r%numat))
     allocate(ZP_local(qm_main_r%numat))
     allocate(PO_1(qm_main_r%numat))
     allocate(PO_2(qm_main_r%numat))
     allocate(PO_3(qm_main_r%numat))
     allocate(PO_7(qm_main_r%numat))
     allocate(PO_9(qm_main_r%numat))
     allocate(DD_2(qm_main_r%numat))
     allocate(DD_3(qm_main_r%numat))

     allocate(BETAD_local(qm_main_r%numat))
     allocate(ZD_local(qm_main_r%numat))

     if(qm_control_r%iqm_mode.eq.5) then
        allocate(MALPB_local(qm_main_r%numat))
        allocate(ALPB_local(qm_main_r%numat,qm_main_r%numat))
     end if
     if(qm_control_r%q_am1_pm3) then
        allocate(IMPAR_local(qm_main_r%numat))
        allocate(GUESS1_local(4,qm_main_r%numat))
        allocate(GUESS2_local(4,qm_main_r%numat))
        allocate(GUESS3_local(4,qm_main_r%numat))
        allocate(GNN_local(qm_main_r%numat))
     end if
  else
     if(.not.associated(ni_local)) call wrndie(-5,'<QMMM_module_prep>', &
                         ' QMMM_module_prep memories are not allocated.')
  end if

  !
  do_am1_pm3   =qm_control_r%q_am1_pm3
  do_d_orbitals=qm_control_r%do_d_orbitals
  iqm_mode     =qm_control_r%iqm_mode

  ! now start seting up.
  do i=1,qm_main_r%numat
     ni             = qm_main_r%nat(i)
     ia             = qm_main_r%NFIRST(i)
     iorbs          = qm_main_r%num_orbs(i)
     ni_local(i)    = ni
     iorbs_local(i) = iorbs
     ia_local(i)    = ia
     ib_local(i)    = qm_main_r%NLAST(i)
     ip_local(i)    = qm_scf_main_r%NW(i)
     is_local(i)    = qm_scf_main_r%indx(ia)+ia
     iw_local(i)    = qm_scf_main_r%indx(iorbs)+iorbs
     if(iorbs.eq.9) then
        Jmax_local(i)=10
     else
        Jmax_local(i)=4
     end if

     OMEGA_local(i)= OMEGA(ni)
     DELTA_local(i)= DELTA(ni)
     CORE_local(i) = CORE(ni)
     if(qm_main_r%uhf) then
        GSS_local(i)  = GSS(ni)
        GSP_local(i)  = GSP(ni)
        GPP_local(i)  = GPP(ni)
        GP2_local(i)  = GP2(ni)
        HSP_local(i)  = HSP(ni)
        HPP_local(i)  = HPP(ni)
     else
        GSS_local(i)  = GSS(ni)*PT5
        GSP_local(i)  = GSP(ni)-HSP(ni)*PT5
        GPP_local(i)  = GPP(ni)*PT5
        GP2_local(i)  = GP2(ni)-HPP(ni)*PT5
        HSP_local(i)  = HSP(ni)*PT75-GSP(ni)*PT25
        HPP_local(i)  = HPP(ni)*PT75-GP2(ni)*PT25
     end if

     ! for resonance integrals.
     BETAS_local(i)= BETAS(ni)
     BETAP_local(i)= BETAP(ni)
     ZS_local(i)   = ZS(ni)
     ZP_local(i)   = ZP(ni)
     if(iorbs.eq.9) then
        BETAD_local(i)=BETAD(ni)
        ZD_local(i)   =ZD(ni)
        if(qm_main_r%uhf) then
           do j=1,243
              w_save(j,i) = REPD(INTREP(j),ni)
           end do
        else
           do j=1,243
              int1 = INTRF1(j)
              int2 = INTRF2(j)
              rf   = REPD(INTREP(j),ni)
              if(int1.gt.0) rf = rf-PT25*REPD(int1,ni)
              if(int2.gt.0) rf = rf-PT25*REPD(int2,ni)
              w_save(j,i) = rf
           end do
        end if
     end if

     ! for core-core interactions & and repp.
     ALP_local(i) = ALP(ni)
     PO_1(i)      = PO(1,ni)
     PO_2(i)      = PO(2,ni)
     PO_3(i)      = PO(3,ni)
     PO_7(i)      = PO(7,ni)
     PO_9(i)      = PO(9,ni)

     DD_2(i)      = DD(2,ni)
     DD_3(i)      = DD(3,ni)
  end do

  if(do_d_orbitals) then
     do i=1,243
        int_ij(i)=INTIJ(i)
        int_kl(i)=INTKL(i)
     end do
  end if

  ! for mm atoms
  PO_1_mm = PO(1,0)
  PO_2_mm = PO(2,0)
  PO_3_mm = PO(3,0)
  PO_7_mm = PO(7,0)
  PO_9_mm = PO(9,0)
  DD_2_mm = DD(2,0)
  DD_3_mm = DD(3,0)

  ! mndo/d specific
  if(qm_control_r%iqm_mode.eq.5) then
     do i=1,qm_main_r%numat
        ni            = ni_local(i)
        MALPB_local(i)= MALPB(ni)
        do j=1,qm_main_r%numat
           nj=ni_local(j)
           ALPB_local(j,i)=ALPB(nj,ni)
        end do
     end do
  end if

  ! am1/pm3/am1/d specific.
  if(qm_control_r%q_am1_pm3) then
     ! allocate memory..
     do i=1,numat_local
        ni             = ni_local(i)
        ii             = IMPAR(ni)
        IMPAR_local(i) = ii
        GUESS1_local(1:ii,i)=GUESS1(1:ii,ni)
        GUESS2_local(1:ii,i)=GUESS2(1:ii,ni)
        GUESS3_local(1:ii,i)=GUESS3(1:ii,ni)
        GNN_local(i)=GNN(ni)
     end do
  end if

  return
  end subroutine QMMM_module_prep

  subroutine betaij(I,J,NI,NJ,iorbs,jorbs,R,T)
  !
  ! RESONANCE INTEGRALS IN LOCAL COORDINATES.
  !
  ! NOTATION. I=INPUT,O=OUTPUT.
  ! NI,NJ     ATOMIC NUMBERS (I).
  ! R         INTERNUCLEAR DISTANCE, IN ATOMIC UNITS (I).
  ! T(14)     LOCAL RESONANCE INTEGRALS, IN EV (O).
  !
  !use chm_kinds      
  !use qm1_parameters, only : BETAS,BETAP,BETAD
  !use qm1_constant, only : PT5

  implicit none

  integer :: I,J,NI,NJ,iorbs,jorbs
  real(chm_real):: R,T(14)

  ! local variables:
  real(chm_real):: TT

  ! compute overlap integrals.
  call overlp(i,j,ni,nj,iorbs,jorbs,R,T)

  ! initialization.
  !iorbs  = LORBS(ni)
  !jorbs  = LORBS(nj)

  ! MNDO-type resonance integrals. 
  ! cases : 1. 1-1 (norbs=1, norbs=1).
  !         2. 1-4 (norbs=1, nobrs=4), 
  !         3. 4-4 (norbs=4, norbs-4).
  !         4. 1-9 (norbs=1, norbs=9).
  !         5. 4-9 (norbs=4, norbs=9).
  !         6. 9-9 (norbs=9, norbs=9).
  select case(iorbs+jorbs)
     case ( 2)                ! case 1
        T(1) = PT5*(BETAS_local(I)+BETAS_local(J))*T(1)
     case ( 5)                ! case 2
        T(1) = PT5*(BETAS_local(I)+BETAS_local(J))*T(1)
        T(2) = PT5*(BETAS_local(I)+BETAP_local(J))*T(2)
        T(3) = PT5*(BETAP_local(I)+BETAS_local(J))*T(3)
     case ( 8)                ! case 3
        T(1) = PT5*(BETAS_local(I)+BETAS_local(J))*T(1)
        T(2) = PT5*(BETAS_local(I)+BETAP_local(J))*T(2)
        T(3) = PT5*(BETAP_local(I)+BETAS_local(J))*T(3)
        TT   = PT5*(BETAP_local(I)+BETAP_local(J))
        T(4:5) = TT*T(4:5)
     case (10)                ! case 4
        T(1) = PT5*(BETAS_local(I)+BETAS_local(J))*T(1)
        T(2) = PT5*(BETAS_local(I)+BETAP_local(J))*T(2)
        T(3) = PT5*(BETAP_local(I)+BETAS_local(J))*T(3)

        T(6) = PT5*(BETAD_local(I)+BETAS_local(J))*T(6)
        T(7) = PT5*(BETAS_local(I)+BETAD_local(J))*T(7)
     case (13)                ! case 5
        T(1) = PT5*(BETAS_local(I)+BETAS_local(J))*T(1)
        T(2) = PT5*(BETAS_local(I)+BETAP_local(J))*T(2)
        T(3) = PT5*(BETAP_local(I)+BETAS_local(J))*T(3)
        TT   = PT5*(BETAP_local(I)+BETAP_local(J))
        T(4:5) = TT*T(4:5)
        T(6)  = PT5*(BETAD_local(I)+BETAS_local(J))*T(6)
        T(7)  = PT5*(BETAS_local(I)+BETAD_local(J))*T(7)
        T(8)  = PT5*(BETAD_local(I)+BETAP_local(J))*T(8)
        T(9)  = PT5*(BETAP_local(I)+BETAD_local(J))*T(9)
        T(10) = PT5*(BETAD_local(I)+BETAP_local(J))*T(10)
        T(11) = PT5*(BETAP_local(I)+BETAD_local(J))*T(11)
     case (18)                ! case 6
        T(1) = PT5*(BETAS_local(I)+BETAS_local(J))*T(1)
        T(2) = PT5*(BETAS_local(I)+BETAP_local(J))*T(2)
        T(3) = PT5*(BETAP_local(I)+BETAS_local(J))*T(3)
        TT   = PT5*(BETAP_local(I)+BETAP_local(J))
        T(4:5) = TT*T(4:5)
        T(6)  = PT5*(BETAD_local(I)+BETAS_local(J))*T(6)
        T(7)  = PT5*(BETAS_local(I)+BETAD_local(J))*T(7)
        T(8)  = PT5*(BETAD_local(I)+BETAP_local(J))*T(8)
        T(9)  = PT5*(BETAP_local(I)+BETAD_local(J))*T(9)
        T(10) = PT5*(BETAD_local(I)+BETAP_local(J))*T(10)
        T(11) = PT5*(BETAP_local(I)+BETAD_local(J))*T(11)
        TT    = PT5*(BETAD_local(I)+BETAD_local(J))
        T(12:14) = TT*T(12:14)
  end select
  return
  end subroutine betaij


  subroutine core_repul(I,J,NI,NJ,R,WIJ,ENUCLR)
  ! 
  ! CORE-CORE REPULSION FUNCTION IN MNDO-TYPE METHODS.
  ! CONTRIBUTION ENUCLR FROM A GIVEN ATOM PAIR I-J.
  ! 
  ! NOTATION. I=INPUT, O=OUTPUT.
  ! I,J       ATOM PAIR I-J (I).
  ! NI,NJ     CORRESPONDING ATOMIC NUMBERS (I).
  ! R         DISTANCE IN ATOMIC UNITS (I).
  ! WIJ       TWO-CENTER INTEGRAL (SS,SS) FOR IOP.GT.-10 (O).
  !
  !use chm_kinds
  !use number
  !use qm1_constant
  !use qm1_parameters, only : ALP,DD,PO,CORE, &
  !                           ALPB, MALPB      ! mndo/d only
  !use qm1_info, only : qm_control_r

  implicit none
  !
  integer       :: I,J,NI,NJ
  real(chm_real):: R,WIJ,ENUCLR

  ! local variables
  integer :: k,MALPNI,MALPNJ
  real(chm_real):: ALPNI,ALPNJ,RIJ,ENI,ENJ,SCALE,ENUC,ENUC1 

  ! set:
  ALPNI  = ALP_local(I)
  ALPNJ  = ALP_local(J)
  RIJ    = R*A0   ! CONVERT TO ANGSTROM.

  ! bond parameters in mndo/d (if defined).
  ! ALPB(nj,ni) is used as alpha parameters for the element with the
  !             atomic number NI in the case of the pair NJ-NI,
  !            -if there are bond parameters for NI (MALPB(ni).gt.0),
  !            -if there may be a bond parameter for NJ-NI (NJ.le.MALPB(ni)).
  !            -if ALPB(nj,ni) is nonzero and positive.
  ! The standard ALPHA parameter ALP(ni) is used if any one of there
  ! conditions is not satisfied.
  if(iqm_mode.eq.5) then  ! mndo/d
     MALPni = MALPB_local(i)
     MALPnj = MALPB_local(j)
     if(MALPni.gt.0 .and. nj.le.MALPni) then
        if(ALPB_local(j,i).gt.ZERO) ALPNI = ALPB_local(j,i)
     end if
     if(MALPnj.gt.0 .and. ni.le.MALPnj) then
        if(ALPB_local(i,j).gt.ZERO) ALPNJ = ALPB_local(i,j)
     end if
  end if
  ! calculate scale factor including expeonetial terms.
  ENI    = EXP(-ALPNI*RIJ)
  if(NI.EQ.NJ) then
     ENJ = ENI
  else
     ENJ = EXP(-ALPNJ*RIJ)
  end if
  SCALE  = ONE+ENI+ENJ

  ! H-O or H-N pairs
  if(ni.eq.1 .and. (nj.eq.7 .or. nj.eq.8)) then
     scale=scale+(RIJ-one)*ENJ
  else if(nj.eq.1 .and. (ni.eq.7 .or. ni.eq.8)) then
     scale=scale+(RIJ-one)*ENI
  end if

  ! calculate basic replusive term.
  if (do_d_orbitals) then
     ! since RIJ=R*A0, RIJ/A0=R
     ! WIJ = EV/SQRT(RIJ*RIJ/(A0*A0)+(PO_9(i)+PO_9(j))**2)
     WIJ = EV/SQRT(R*R+(PO_9(i)+PO_9(j))**2)
  end if 
  ENUC   = CORE_local(i)*CORE_local(j)*WIJ

  ! core-core replusion terms for AM1/PM3/AM1/d
  ENUC1  = ZERO
  if(do_am1_pm3) call repam1_qmqm(i,j,NI,NJ,RIJ,ENUC1)

  ! Add AM1 core-core repulsion TERMS.
  ENUCLR = ENUC*SCALE+ENUC1

  return
  end subroutine core_repul


  subroutine guessp(PA,PB,dim_linear_norbs)
  !
  ! Definition of initial density matrix.
  !
  ! NOTATION. I=INPUT, O=OUTPUT, S=SCRATCH.
  ! PA(dim_linear_norbs)   RHF OR UHF-ALPHA DENSITY MATRIX (O).
  ! PB(dim_linear_norbs)   UHF-BETA DENSITY MATRIX (O).
  ! 
  !use chm_kinds
  use qm1_info, only : qm_main_r,qm_scf_main_r
  use qm1_parameters, only : III,IIID ! ,CORE
  !use number
  !use qm1_constant

  implicit none
  !
  integer :: dim_linear_norbs
  real(chm_real):: PA(dim_linear_norbs),PB(dim_linear_norbs)

  ! local variables
  integer :: i,j,k,kk,ia,is
  integer :: NSPORB,NI,IORBS
  real(chm_real):: YY,W,TEMP,DA,DB,DC,FA,FB,charge
  real(chm_real),parameter :: r_12=1.0d0/twelve


  ! options.
  charge = qm_main_r%qmcharge

  ! diagonal trial density matrix.
  K      = 0
  PA(1:dim_linear_norbs)=zero

  ! NSPORB : number of atomic orbitals initially populated.
  !          D-orbitals of main-group elements are not populated.
  !          P-orbitals of transition elementes are not populated.
  NSPORB = qm_main_r%norbs
  do i=1,qm_main_r%numat
     ni     = ni_local(i)    ! qm_main_r%nat(i)
     iorbs  = iorbs_local(i) ! qm_main_r%num_orbs(i)
     if(iorbs.eq.9) then
        if(III(ni).le.IIID(ni)) NSPORB=NSPORB-5
        if(III(ni).gt.IIID(ni)) NSPORB=NSPORB-3
     end if
  end do
  YY     = charge/real(NSPORB)
  ! now, loop over all atoms.
  do i=1,qm_main_r%numat
     ia     = ia_local(i)    ! qm_main_r%nfirst(I)
     iorbs  = iorbs_local(i) ! qm_main_r%num_orbs(i)
     is     = is_local(i)    ! qm_scf_main_r%indx(ia)+ia ! =ia*(ia+1)/2
     ni     = ni_local(i)    ! qm_main_r%nat(I)
     if(iorbs.eq.1) then
        ! atoms with an S-basis
        PA(is) = (CORE_local(i)-YY)*PT5
     else if(iorbs.eq.4) then
       ! atoms with an SP-basis.
        W   = (CORE_local(i)*PT25-YY)*PT5
        PA(is)        = W
        PA(is+ia+1)   = W
        PA(is+2*ia+3) = W
        PA(is+3*ia+6) = W
     else if(iorbs.eq.9 .and. (III(ni).le.IIID(ni))) then
        ! main-group elements with an SPD-basis.
        W   = (CORE_local(i)*PT25-YY)*PT5
        PA(is)        = W
        PA(is+ia+1)   = W
        PA(is+2*ia+3) = W
        PA(is+3*ia+6) = W
     else if(iorbs.eq.9 .and. (III(ni).gt.IIID(ni))) then
        ! transition-metal elements with an SPD-basis.
        ! up to 10 electrons, put into S and D orbitals.
        ! more than 10 electrons, fill D orbitials and put the rest in S orbitail.
        temp = CORE_local(i)-YY*six
        if(temp.lt.ten) then
           W   = temp*r_12      ! /twelve
           PA(is)         = W
           PA(is+4*IA+10) = W
           PA(is+5*IA+15) = W
           PA(is+6*IA+21) = W
           PA(is+7*IA+28) = W
           PA(is+8*IA+36) = W
        else
           W   = (temp-ten)*PT5
           PA(is)         = W
           PA(is+4*ia+10) = one
           PA(is+5*ia+15) = one
           PA(is+6*ia+21) = one
           PA(is+7*ia+28) = one
           PA(is+8*ia+36) = one
        end if
     end if
  end do

  ! Use diagonal trial density matrix for UHF with Ktrial=0.
  ! perturb initial densiity matrices for UHF singlets.
  if(qm_main_r%UHF) then
     da  = real(qm_main_r%nalpha)
     db  = real(qm_main_r%nbeta)
     temp= two/(da+db)
     fa  = da*temp
     fb  = db*temp
     do i=1,dim_linear_norbs
        PB(i) = PA(i)*fb
        PA(i) = PA(i)*fa
     end do
     if(qm_main_r%nalpha.eq.qm_main_r%nbeta) then
        da = 0.98D0
        db = two-da
        k  = 0
        do j=1,qm_main_r%norbs
           k  = k+j
           dc = da
           da = db
           db = dc
           PA(k) = PA(k)*da
           PB(k) = PB(k)*db
        end do
     end if
  end if

  return
  end subroutine guessp


  subroutine hcorep(H,W,linear_norbs,linear_fock,linear_fock2,ENUCLR)
  !
  ! Full core hamiltonian for MNDO and related methods. And, note that
  ! the one-center part of H(linear_norbs) and Enuclr are guarded.
  !
  ! NOTATION. I=INPUT, O=OUTPUT.
  ! H(linear_norbs)    core hamiltonian matrix (O).
  ! W(linear_fock2)    two-electron integrals (O).
  !
  !use chm_kinds
  use qm1_info, only : qm_control_r,qm_main_r,qm_scf_main_r,mm_main_r
  !use number  , only : zero
  use qm1_mndod, only : reppd_qmqm,rotd
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none
  !
  integer :: linear_norbs,linear_fock,linear_fock2
  real(chm_real):: H(linear_norbs),W(linear_fock2),ENUCLR

  ! local variables
  integer :: i,j,k,ii,jj,ij,ia,ja,is,js,ip,jp,iw,jw,ijp,kr,ni,nj,ll,iorbs,jorbs
  integer :: iicnt,ipnt
  real(chm_real):: En,Hij,Wij,R,delr
  logical :: qi_h,qj_h
  integer :: istart_norbs,iend_norbs
#if KEY_PARALLEL==1
  integer :: ISTRT_CHECK            ! external function
  integer :: mmynod, nnumnod,icnt
#endif

  istart_norbs  = 1
  iend_norbs    = qm_main_r%norbs
#if KEY_PARALLEL==1
  mmynod  = mynod
  nnumnod = numnod
  if(nnumnod>1) istart_norbs = ISTRT_CHECK(iend_norbs,qm_main_r%norbs)
#endif

  ! initialize some variables.
  kr     = 0
  Enuclr = zero
  h(1:linear_norbs)=zero
  w(1:linear_fock2)=zero

  ! Diagonal one-center terms: replace "call one_center_h(H)"
  !          as this was precomputed at the beginning of QM setup.
  !          see subroutine compute_one_center_h.

  ! note: this should be carefully handled in using parallel as H_1cent is filled for each atom 
  !       at the subroutine compute_one_center_h. So, either only fill for part of atoms based on
  !       parallel partitioning or for 1:qm_main_r%norbs should be changed or do only for the
  !       master (or only one) node.
  !#if KEY_PARALLEL==1
  !!if(mmynod.eq.0) then
  !#endif
  !!   h(qm_scf_main_r%imap_h(1:qm_main_r%norbs))=qm_scf_main_r%H_1cent(1:qm_main_r%norbs)
  !!#if KEY_PARALLEL==1
  !!end if
  !!#endif
  !! taken cared for parallelization.
  h(qm_scf_main_r%imap_h(istart_norbs:iend_norbs))=qm_scf_main_r%H_1cent(istart_norbs:iend_norbs)

  ! loop over atom pairs for offdiagonal two-center terms.
  ! atoms i and j are identified at the beginning of the loop.
#if KEY_PARALLEL==1
  icnt=0
#endif
  iicnt=0
  if(mm_main_r%q_lookup_beta) r_dr_width_beta = one/dr_width_beta

  loopii: do i=2,qm_main_r%numat
     ni     = ni_local(i)     ! qm_main_r%nat(i)
     ia     = ia_local(i)     ! qm_main_r%nfirst(i)
     is     = is_local(i)     ! qm_scf_main_r%indx(ia)+ia
     iorbs  = iorbs_local(i)  ! qm_main_r%num_orbs(i)
     iw     = iw_local(i)     ! qm_scf_main_r%indx(iorbs)+iorbs
     ip     = ip_local(i)     ! qm_scf_main_r%NW(i)

     qi_h   = ni.eq.1  ! h-atom?
     loopjj: do j=1,i-1
        if(ni > 1 .or. ni_local(j) > 1) iicnt = iicnt + 1
#if KEY_PARALLEL==1
        icnt = icnt + 1
        if(mmynod .ne. mod(icnt-1,nnumnod)) cycle loopjj
#endif
        nj     = ni_local(j)     ! qm_main_r%nat(J)
        ja     = ia_local(j)     ! qm_main_r%nfirst(j)
        js     = is_local(j)     ! qm_scf_main_r%indx(ja)+ja
        jorbs  = iorbs_local(j)  ! qm_main_r%num_orbs(j)
        jw     = iw_local(j)     ! qm_scf_main_r%indx(jorbs)+jorbs
        jp     = ip_local(j)     ! qm_scf_main_r%NW(j)

        qj_h   = nj.eq.1 ! h-atom?
        ! for H-H pair.
        if(qi_h .and. qj_h) then
           call hhpair (j,i,j,i,qm_main_r%qm_coord,Hij,Wij,En)
           H(qm_scf_main_r%indx(ia)+ja) = Hij
           H(is)  = H(is)-Wij
           H(js)  = H(js)-Wij
           Enuclr = Enuclr+En
           ! IJp points the current point in the square matrix of
           ! W(linear_fock,linear_fock)
           !            but in the lower triangle part.
           IJp    = (jp-1)*linear_fock + ip  ! <= linear_fock?
           W(IJp) = Wij
        else 
           ! distance R (au) and rotation matrix.
           call rotmat(J,I,JORBS,IORBS,qm_main_r%NUMAT,qm_main_r%qm_coord,R,qm_scf_main_r%YY)
           ! two-electron integrals in local coordinates & compute and store
           !              the semiempirical integrals.
           call repp_qmqm(i,j,NI,NJ,R,qm_scf_main_r%RI,qm_scf_main_r%CORE_mat)
           if(iorbs.ge.9 .or. jorbs.ge.9)  &
              call reppd_qmqm(NI,NJ,R,qm_scf_main_r%RI,qm_scf_main_r%CORE_mat, &
                              qm_scf_main_r%WW,IW,JW)
           ! transform two-electron integrals to molecular coordinates.
           if(iorbs.le.4 .and. jorbs.le.4) then
              call rotate(IW,JW,IP,JP,KR,qm_scf_main_r%RI,qm_scf_main_r%YY, &
                          W,W,linear_fock,linear_fock2,0)
           else
              call rotd(qm_scf_main_r%WW,qm_scf_main_r%YY,IW,JW)
              call w2mat(ip,jp,qm_scf_main_r%WW,W,linear_fock,iw,jw)
           end if

           ! resonance integrals.
           if(mm_main_r%q_lookup_beta) then
              if(r >= rval_min_beta .and. r <= rval_max_beta) then
                 ipnt = ij_pair_look_up(iicnt)  ! the array counter.
                 select case(iorbs+jorbs)
                    case (2)
                       ii=1
                    case (5)
                       ii=3
                    case (8)
                       ii=5
                    case (10)
                       ii=7
                    case (13)
                       ii=11
                    case (18)
                       ii=15
                 end select
                 jj  = int((r - rval_min_beta)*r_dr_width_beta) + 1
                 delr= r - r_beta_val(jj)
                 do k=1,ii
                    qm_scf_main_r%T(k)=((look_up_beta_r(ipnt)%coef_val(1,k,jj) *delr + &
                                         look_up_beta_r(ipnt)%coef_val(2,k,jj))*delr + &
                                         look_up_beta_r(ipnt)%coef_val(3,k,jj))*delr + &
                                         look_up_beta_r(ipnt)%coef_val(4,k,jj)       + &
                                         look_up_beta_r(ipnt)%val_shift(k)
                 end do
              else
                  call betaij(i,j,NI,NJ,iorbs,jorbs,R,qm_scf_main_r%T)
              end if
           else
              call betaij(i,j,NI,NJ,iorbs,jorbs,R,qm_scf_main_r%T)
           end if
           call rotbet(IA,JA,IORBS,JORBS,qm_scf_main_r%T,qm_scf_main_r%YY,H,linear_norbs, &
                       qm_scf_main_r%indx)

           ! core-electron attractions.
           call rotcora_qmqm(IA,JA,IORBS,JORBS,IS,JS,qm_scf_main_r%CORE_mat, &
                             qm_scf_main_r%YY,H,linear_norbs)

           ! core-core repulsions.
           Wij = qm_scf_main_r%RI(1)
           call core_repul(i,j,ni,nj,R,Wij,En)
           Enuclr = Enuclr+En                    ! combined Enuclr in the parent routine.
        end if
     end do loopjj
  end do loopii

  !!!! store MNDO integrals in square form: 
  ! The following call is done in the parent subroutine after gcomb, as
  ! this should be done the same change for each node.
  ! Complete defintion of square matrix of MNDO two-electron integrals
  ! the MNDO integrals in square form is stored in the parent subroutine after gcomb!
  !call wstore(W,linear_fock,0,qm_main_r%numat,qm_main_r%uhf)

  !
  return

  contains

     subroutine w2mat(IP,JP,WW,W,LM6,LIMIJ,LIMKL)
     !
     ! store two-center two-electron integrals in a square matrix.
     !
     !use chm_kinds

     implicit none
     integer       :: IP,JP,LM6,LIMIJ,LIMKL
     real(chm_real):: W(LM6,LM6),WW(LIMKL,LIMIJ)

     ! local variable
     integer :: KL,ipa,jpa,ij

     ipa    = ip-1
     jpa    = jp-1
     do kl=1,LIMKL
        do ij=1,LIMIJ
           W(IPA+IJ,JPA+KL) = WW(KL,IJ)
        end do
        !W(ipa+1:ipa+LIMIJ,jpa+kl) = WW(kl,1:LIMIJ)
     end do
     return
     end subroutine w2mat

!
! Done in subroutine compute_one_center_h
!
!     subroutine one_center_h(H)
!     !
!     ! fill diagonal one-center terms.
!     !
!     !use chm_kinds
!     use qm1_info, only : qm_main_r,qm_scf_main_r
!     use qm1_parameters,only : USS,UPP,UDD
!
!     real(chm_real):: H(*)
!
!     ! local
!     integer :: i,ni,ia,ll,iorbs,j
!
!     ! DIAGONAL ONE-CENTER TERMS.
!     ! this can be done once at the beginning of QM setup.
!     ! work on this later to make it go over the loop once.
!     do i=1,qm_main_r%numat
!        ni     = ni_local(i)    ! qm_main_r%NAT(i)
!        ia     = ia_local(i)    ! qm_main_r%NFIRST(i)
!        iorbs  = iorbs_local(i) ! qm_main_r%num_orbs(i)  ! = NLAST(I)-IA+1
!        H(qm_scf_main_r%INDX(ia)+ia) = USS(ni)
!        if(iorbs.ge.9) then
!           do j=ia+1,ia+3
!              H(qm_scf_main_r%INDX(j)+j)  = UPP(ni)
!           end do
!           do j=ia+4,ia+8
!              H(qm_scf_main_r%INDX(j)+j)  = UDD(ni)
!           end do
!        else if(iorbs.ge.4) then
!           do j=ia+1,ia+3
!              H(qm_scf_main_r%INDX(j)+j)  = UPP(ni)
!           end do
!        end if
!     end do
!     return
!     end subroutine one_center_h
  !=====================================================================
  end subroutine hcorep


  subroutine hhpair(jqm,iqm,j,i,coord,hij,wij,enuclr)
  !
  ! integrals for a hydrogen-hydrogen pair.
  !
  ! Notation:  (I): input; (O): output
  ! j,i     Atom pair i-j (I).
  ! coord   Cartesian coordinates in Angstrom (I).
  ! HIJ     Two-center resonance integral (O).
  ! WIJ     Two-center two-electron integrals (SS,SS) (O).
  ! Enuclr  Contribution to core-core repulsion (O).
  !use chm_kinds
  !use number
  !use qm1_constant
  !use qm1_info, only : qm_control_r
  !use qm1_parameters, only : ZS,PO,BETAS,ALP
  !
  implicit none

  integer :: jqm,iqm,j,i
  real(chm_real):: COORD(3,*),HIJ,WIJ,ENUCLR

  ! local variables
  real(chm_real):: R,ZR,RIJ,ENI,SCALE,ZI
  real(chm_real),parameter :: r_three=one/three, &
                              r_A0   =one/A0

  ! distance R (AU), and resonance integral Hij.
  r = SQRT((coord(1,j)-coord(1,i))**2+(coord(2,j)-coord(2,i))**2+(coord(3,j)-coord(3,i))**2)*r_A0
  zr= zs_local(iqm)*r  ! zs(1)*r
  ! resonance integral Hij.
  if(zr.lt.bigexp) then
     Hij = BETAS_local(iqm)*EXP(-zr)*(one+zr+zr*zr*r_three)
  else
     Hij = zero
  end if
  ! two-electron integral Wij
  Wij    = EV/SQRT(r*r+FOUR*PO_1(iqm)**2)
  ! core-core replusion Enuclr.
  Rij    = r*A0
  Enuclr = Wij*(one+two*(EXP(-ALP_local(iqm)*Rij)))
  ! am1-type core-core terms
  if(do_am1_pm3) call repam1_hh_pair(iqm,1,RIJ,ENUCLR)

  return
  end subroutine hhpair

          
  subroutine mmint(H,dim_linear_norbs,ENUCLR)
  !
  ! Electrostatic contributions from mm point charges to the Core hamiltonian
  ! and the core-core repulsions.
  !
  !use chm_kinds
  use qm1_info, only : qm_control_r,qm_main_r,mm_main_r,qm_scf_main_r
  use nbndqm_mod, only: map_qmatom_to_group,map_mmatom_to_group
  !use qm1_parameters, only : CORE,OMEGA,DELTA
  use qm1_mndod, only : reppd_qmmm
#if KEY_PARALLEL==1
  use parallel
#endif
  !
  implicit none
  !
  integer :: dim_linear_norbs,m
  real(chm_real):: H(dim_linear_norbs),ENUCLR,enucqm

  integer :: i,ij,jj,irs_qm,irs_mm,numqm
  real(chm_real):: XCOORD(3,2),PTCHG,PTCHG_SIGN,R,scale,enuc,r_sq
  real(chm_real):: RI_local(22),CORE_mat(10,2)
  !
  integer :: mstart,mstop
#if KEY_PARALLEL==1
  integer :: ISTRT_CHECK            ! for external function
#endif

  ! for look-up
  real(chm_real):: rr_val,rr_min,rr_max,delr,scale2,enuc2,sw_scale
  integer       :: i_do_switching

  mstart  = 1
  mstop   = mm_main_r%numatm
  numqm = qm_main_r%numat
#if KEY_PARALLEL==1
  if(numnod>1) mstart = ISTRT_CHECK(mstop,mm_main_r%numatm)
#endif

  ! 
  ! MM point charges (M.J.Field et al. J.Comput.Chem. 11, 700 (1990))
  do m=mstart,mstop               ! 1,mm_main_r%numatm
     !if(mm_main_r%mm_chrgs(m).ne.zero) then
     ! in qmint, it loops over qm atoms.
     ! call qmint(H,dim_linear_norbs,enucqm,mm_main_r%mm_coord(1:3,m), &
     !            mm_main_r%mm_chrgs(m),                               &
     !            qm_main_r%numat,qm_main_r%NAT,qm_main_r%NFIRST,      &
     !            qm_main_r%num_orbs,qm_scf_main_r%indx,               &
     !            qm_main_r%qm_coord,qm_scf_main_r%CORE_mat,           &
     !            qm_scf_main_r%WW,qm_scf_main_r%RI,qm_scf_main_r%YY,  &
     !            qm_control_r%q_am1_pm3)

     if(mm_main_r%q_cut_by_group .or. mm_main_r%q_switch) irs_mm = map_mmatom_to_group(m) 

     XCOORD(1:3,1) = mm_main_r%mm_coord(1:3,m)
     PTCHG         = mm_main_r%mm_chrgs(m)
     if(PTCHG.ge.zero) then
        PTCHG_SIGN = one
     else
        PTCHG_SIGN =-one
     end if
     ! loop over qm atoms for each mm atom.
     ENUCQM = zero
     if(mm_main_r%q_lookup) then
        r_dr_width = one/dr_width
        do i=1,numqm
           XCOORD(1:3,2) = qm_main_r%qm_coord(1:3,i)

           ! group-by-group-based cutoff case, skip the pair if its distance is longer than cutoff.
           ! otherwise (default group-based case), include all mm atoms (default).
           if(mm_main_r%q_cut_by_group) then
              irs_qm = map_qmatom_to_group(i)
              if(.not.mm_main_r%q_mmgrp_qmgrp_cut(irs_mm,irs_qm)) cycle
           else if(mm_main_r%q_switch) then
              irs_qm = map_qmatom_to_group(i)
           end if

           ! distance R (au) and rotation matrix.
           call rotmat_qmmm(1,2,1,iorbs_local(i),2,XCOORD,R,qm_scf_main_r%YY)
           ! local charge-electron attraction integrals.
           call repp_qmmm(ni_local(i),0,i,R,RI_local,CORE_mat)
           if(iorbs_local(i).ge.9) call reppd_qmmm(ni_local(i),0,R,RI_local,CORE_mat, &
                                                   qm_scf_main_r%WW,iw_local(i),1)

           ! multiplication by point charge.
           sw_scale = one
           if(mm_main_r%q_switch) then
              i_do_switching = mm_main_r%q_mmgrp_qmgrp_swt(irs_mm,irs_qm)
              ! apply switching function
              if(i_do_switching > 0) sw_scale = mm_main_r%sw_val(i_do_switching)
           end if
           CORE_mat(1:Jmax_local(i),1) = CORE_mat(1:Jmax_local(i),1)*PTCHG*sw_scale

           ! contributions to the core hamiltonian.
           call rotcora_qmmm(ia_local(i),0,iorbs_local(i),0,is_local(i),0,CORE_mat, &
                             qm_scf_main_r%YY,H,dim_linear_norbs,mm_main_r%q_diag_coulomb)

           ! contributions to the core-core repulsions
           if(.not.mm_main_r%q_diag_coulomb) then
              ij     = i_index_look_up(i)  ! matching atom id.
              R      = R*A0
              rr_val = r 
              if(rr_val >= rval_min .and. rr_val <= rval_max) then
                 jj = int( (rr_val-rval_min)*r_dr_width) + 1
                 delr= rr_val - r_core_val(jj)
                 scale=((coef_core_val(1,jj,1,ij)*delr +coef_core_val(2,jj,1,ij))*delr + &
                         coef_core_val(3,jj,1,ij))*delr+coef_core_val(4,jj,1,ij) + &
                         core_val_shift(1,ij)
                 enuc = CORE_local(i)*RI_local(1)*(one+PTCHG_SIGN*scale)
                 if(do_am1_pm3 .and. rr_val <= core_cut_val(ij)) then
                    enuc = enuc + ((coef_core_val(1,jj,2,ij)*delr +coef_core_val(2,jj,2,ij))*delr + &
                                    coef_core_val(3,jj,2,ij))*delr+coef_core_val(4,jj,2,ij) + &
                                    core_val_shift(2,ij)
                 end if
              else
                 scale= EXP(-OMEGA_local(i)*(R-DELTA_local(i)))+EXP(-five*R)
                 enuc = CORE_local(i)*RI_local(1)*(one+PTCHG_SIGN*scale)
                 if(do_am1_pm3) call repam1_qmmm(i,ni_local(i),0,R,enuc)
              end if

              ENUCQM = ENUCQM+enuc*sw_scale
           end if
        end do
     else
        do i=1,numqm
           XCOORD(1:3,2) = qm_main_r%qm_coord(1:3,i)

           ! group-by-group-based cutoff case, skip the pair if its distance is longer than cutoff.
           ! otherwise (default group-based case), include all mm atoms (default).
           if(mm_main_r%q_cut_by_group) then
              irs_qm = map_qmatom_to_group(i)
              if(.not.mm_main_r%q_mmgrp_qmgrp_cut(irs_mm,irs_qm)) cycle
           else if(mm_main_r%q_switch) then
              irs_qm = map_qmatom_to_group(i)
           end if

           ! distance R (au) and rotation matrix.
           call rotmat_qmmm(1,2,1,iorbs_local(i),2,XCOORD,R,qm_scf_main_r%YY)
           ! local charge-electron attraction integrals.
           call repp_qmmm(ni_local(i),0,i,R,RI_local,CORE_mat)
           if(iorbs_local(i).ge.9) call reppd_qmmm(ni_local(i),0,R,RI_local,CORE_mat, &
                                                   qm_scf_main_r%WW,iw_local(i),1)

           ! multiplication by point charge.
           sw_scale = one
           if(mm_main_r%q_switch) then
              i_do_switching = mm_main_r%q_mmgrp_qmgrp_swt(irs_mm,irs_qm)
              ! apply switching function
              if(i_do_switching > 0) sw_scale = mm_main_r%sw_val(i_do_switching)
           end if
           CORE_mat(1:Jmax_local(i),1) = CORE_mat(1:Jmax_local(i),1)*PTCHG*sw_scale

           ! contributions to the core hamiltonian.
           call rotcora_qmmm(ia_local(i),0,iorbs_local(i),0,is_local(i),0,CORE_mat, &
                             qm_scf_main_r%YY,H,dim_linear_norbs,mm_main_r%q_diag_coulomb)
   
           ! contributions to the core-core repulsions.
           if(.not.mm_main_r%q_diag_coulomb) then
              R = R*A0
              scale= EXP(-OMEGA_local(i)*(R-DELTA_local(i)))+EXP(-five*R)
              enuc = CORE_local(i)*RI_local(1)*(one+PTCHG_SIGN*scale)
              if(do_am1_pm3) call repam1_qmmm(i,ni_local(i),0,R,enuc)

              ENUCQM = ENUCQM+enuc*sw_scale
           end if
        end do
     end if

     if(.not.mm_main_r%q_diag_coulomb) then
        ENUCLR = ENUCLR+enucqm*PTCHG  ! multiply by the MM point charge.
     end if
     !end if
  end do

  return

!  contains
!     !==================================================================
!     subroutine qmint(H,lin_dim,ENUCQM,mm_crd,PTCHG, &
!                      numat,nat,nfirst,num_orbs,indx,&
!                      qm_coord,CORE_mat,WW,RI,YY,    &
!                      q_am1_pm3)
!     !
!     ! Electrostatic interactions of electrons and cores (qm part) with one MM
!     ! point charge (mm part).
!     ! Integral evaluation reference:
!     ! D. Bakowies and W. Thiel, J. Comput. Chem. 17, 87 (1996).
!     !
!     ! The parameter values for DELTA and OMEGA are defined in subroutine 
!     ! fill_qmmm_parameters. 
!     !
!     ! In this option, the MM charge is treated as a purely 
!     ! classical monopole as the Field-Bash-Karplus approach.
!     ! - the one-electron integrals are evaluated as before, multiplied by
!     !   PTCHG, and then added to the core hamiltonian, which thn incldes the
!     !   interaction with a MM charge (MM).  the procedure is used with the
!     !   standard core hamiltonian as input to compute the QM wavefunction in the
!     !   presence of mm charges.  the corresponding sum over P(mu,nu)*H(mu,nu)
!     !   will incorporate the attractive electrostatic QM/MM interactions.
!     ! - the ENUCQM is multipled by PTCHG to obtain the repulsive electrostatic
!     !   QM/MM interaction.
!     !
!     ! NOTATION.    I=INPUT, O=OUTPUT.
!     ! H(lin_dim)   Core hamiltonian matrix (I,O).
!     ! ENUCQM       charge-core repulsion terms summed over all cores (O).
!     ! mm_crd(3)    cartesian coords. of MM point charge in Angstrom (I).
!     ! PTCHG        value of the point charge in atomic units (I).
!     !
!     !use chm_kinds
!     use qm1_info, only : qm_control_r
!     use qm1_mndod, only : reppd_qmmm
!     use qm1_parameters, only : CORE,OMEGA,DELTA   
!     !use number, only : zero,one
!     !use qm1_constant
!     !
!     implicit none
!     integer :: lin_dim
!     real(chm_real):: H(lin_dim),mm_crd(3),ENUCQM,PTCHG
!     integer :: numat,nat(*),nfirst(*),num_orbs(*),indx(*)
!     real(chm_real):: qm_coord(3,numat),CORE_mat(10,2),WW(2025),RI(22),YY(675)
!     logical       :: q_am1_pm3
!     ! 
!     integer :: i,NI,IW,IORBS,IA,IS,JMAX
!     real(chm_real):: XCOORD(3,2),enuc,PTCHG_SIGN,R,scale
!
!     ! specify mm point charge.
!     XCOORD(1:3,1) = mm_crd(1:3)
!     if(PTCHG.ge.zero) then
!        PTCHG_SIGN = one
!     else
!        PTCHG_SIGN =-one
!     end if
!
!     ! loop over qm atoms for each mm atom.
!     ENUCQM = zero
!     do i=1,numat
!        !!!if(qm_main_r%hlink(i).gt.0) cycle; assume even h-link fully interacts with all MM atoms.
!        ! local variables.
!        ni    = NAT(i)
!        iorbs = num_orbs(i)  ! NLAST(i)-NFIRST(i)+1
!        iw    = indx(iorbs)+iorbs
!        ia    = NFIRST(i)
!        is    = indx(ia)+ia
!        if(iorbs.eq.9) then
!           Jmax=10
!        else
!           Jmax=4
!        end if
!        XCOORD(1:3,2) = qm_coord(1:3,i)
!
!        ! distance R (au) and rotation matrix.
!        call rotmat(1,2,1,iorbs,2,XCOORD,R,YY)
!        ! local charge-electron attraction integrals.
!        call repp_qmmm(ni,0,i,R,RI,CORE_mat)
!        if(iorbs.ge.9) call reppd_qmmm(ni,0,R,RI,CORE_mat,WW,iw,1)
!
!        ! multiplication by point charge.
!        CORE_mat(1:Jmax,1) = CORE_mat(1:Jmax,1)*PTCHG
!
!        ! contributions to the core hamiltonian.
!        call rotcora_qmmm(ia,0,iorbs,0,is,0,CORE_mat,YY,H,lin_dim,mm_main_r%q_diag_coulomb)
!
!        ! contributions to the core-core repulsions.
!        scale= EXP(-OMEGA(ni)*(R*A0-DELTA(ni)))+EXP(-five*R*A0)
!        enuc = CORE(ni)*RI(1)*(one+PTCHG_SIGN*scale)
!        if(q_am1_pm3) call repam1_qmmm(i,ni,0,R*A0,enuc)
!
!        ENUCQM = ENUCQM+enuc
!     end do
!     ! multiply by the MM point charge.
!     ENUCQM = ENUCQM*PTCHG
!
!     return
!     end subroutine qmint
!     !==================================================================
  end subroutine mmint

  subroutine find_unique_qm(ntype)
  !
  ! Find the unique number of qm atoms. (need for the Grimme dispersion correction).
  !
  ! THis routine is only used to map with SCC DFTB data structure.
  !
  use mndo97, only   : nndim,izp
  use qm1_info, only : qm_main_r

  implicit none
  integer :: ntype,i,j,ni,icnt
  logical, pointer :: q_unique(:)=>Null()

  !
  allocate(q_unique(qm_main_r%numat))
  ! find number of unique atoms.
  izp(1) = 1
  do i=2,qm_main_r%numat
     ni          = qm_main_r%nat(i)
     q_unique(i) =.true.
     do j=1,i-1
        if(ni == qm_main_r%nat(j)) then
           ! find qm atom with the same atom type.
           q_unique(i) =.false.
           exit
        end if
     end do
  end do
  !
  !total number of unique qm atoms.
  nunique_qm= 1
  izp(1)    = 1
  do i=2,qm_main_r%numat
     if(q_unique(i)) nunique_qm = nunique_qm + 1
     izp(i) = nunique_qm  ! mapping to SCC DFTB format.
  end do
  ntype = nunique_qm
  deallocate(q_unique)
  return
  end subroutine find_unique_qm

  subroutine mmint_prep_core(dr,r_min,r_max,iunit)
  !
  ! Core-core potential  contributions from mm point charges to the Core hamiltonian.
  ! Saved information for the coefficients and r_values.
  !
  use qm1_info, only : qm_main_r
#if KEY_PARALLEL==1
  use parallel 
#endif
  !
  implicit none
  real(chm_real):: dr,r_min,r_max
  integer :: iunit
  !
  integer :: i,j,ii,ij,ni
  real(chm_real):: rr
  real(chm_real):: R,scale,enuc,enuc_1,enuc_2
  real(chm_real),pointer :: core_val(:,:)=>Null()
  logical :: q_core_zero_check


  ! contributions to the core-core repulsions.
  if(associated(q_unique_atom)) deallocate(q_unique_atom)
  if(.not.associated(q_unique_atom)) allocate(q_unique_atom(qm_main_r%numat))

  if(associated(i_index_look_up)) deallocate(i_index_look_up)
  if(.not.associated(i_index_look_up)) allocate(i_index_look_up(qm_main_r%numat))
  ! find number of unique atoms.
  q_unique_atom(1:qm_main_r%numat)=.true.
  i_index_look_up(1) = 1
  do i=2,qm_main_r%numat
     ni=qm_main_r%nat(i)
     i_index_look_up(i) = i
     do j=1,i-1
        if(ni == qm_main_r%nat(j)) then
           ! find qm atom with the same atom type.
           q_unique_atom(i)   =.false.
           i_index_look_up(i) = j    ! so, later it looks for this value.
           exit
        end if
     end do
  end do
  ! total number of unique qm atoms.
  nunique_qm=0
  do i=1,qm_main_r%numat
     if(q_unique_atom(i)) then
        nunique_qm = nunique_qm + 1
        i_index_look_up(i) = nunique_qm  ! point the array position.
     end if
  end do

  ! re-set i_index_look_up: i_index_look_up(i) should point the array position in coef values.
  do i = 1,qm_main_r%numat
     if(.not. q_unique_atom(i)) then
        j = i_index_look_up(i)
        i_index_look_up(i) = i_index_look_up(j)  
     end if
  end do

  ! now, count the number of arrays.
  i_npnt = 0
  rr=r_min
  do
    i_npnt = i_npnt + 1
    if(rr > r_max) exit
    rr = rr + dr
  end do

  ! allocate memory
  if(associated(r_core_val))     deallocate(r_core_val)
  if(associated(coef_core_val))  deallocate(coef_core_val)
  if(associated(core_val_shift)) deallocate(core_val_shift)
  if(associated(core_cut_val))   deallocate(core_cut_val)
  allocate(r_core_val(i_npnt),coef_core_val(4,i_npnt,2,nunique_qm),core_val_shift(2,nunique_qm))
  allocate(core_cut_val(nunique_qm))

  ! local memory
  if(associated(core_val))      deallocate(core_val)
  allocate(core_val(i_npnt,2))

  core_cut_val(1:nunique_qm)=r_max
  rr=r_min
  ii=0
  do i=1,qm_main_r%numat
     if(q_unique_atom(i)) then
        ii=ii+1
        rr=r_min
        ij=0
!#if KEY_PARALLEL==1
!        if(mynod == 0) then
!#endif
!           write(iunit,*) i_npnt,i,qm_main_r%nat(i)
!#if KEY_PARALLEL==1
!        end if
!#endif
        q_core_zero_check =.true.
        do
           r = rr
           ij= ij+ 1
           scale= EXP(-OMEGA_local(i)*(R-DELTA_local(i)))+EXP(-five*R)
           enuc  = zero
           if(do_am1_pm3) then
              call repam1_qmmm(i,ni_local(i),0,R,enuc)
              if(enuc .eq. zero .and. q_core_zero_check) then
                 core_cut_val(ii) = r
                 q_core_zero_check= .false.
              end if
           end if
           r_core_val(ij) = rr
           core_val(ij,1) = scale          ! scale values
           core_val(ij,2) = enuc           ! negative mm charge
           if(rr > r_max) exit
           rr = rr + dr
        end do

        ! for scale values for mm particles. (see qm1_util_module.src)
        call set_spline_lookup(i_npnt,dr,r_core_val,core_val(1:i_npnt,1), &
                        coef_core_val(1:4,1:i_npnt,1,ii),        &
                        core_val_shift(1,ii))
!#if KEY_PARALLEL==1
!        if(mynod == 0) then
!#endif
!           write(iunit,*) 'for scale values'
!           write(iunit,*) dr_width,core_val_shift(1,ii)
!           do j=1,i_npnt
!              write(iunit,*) r_core_val(j),(coef_core_val(ij,j,1,ii),ij=1,4)
!           end do
!#if KEY_PARALLEL==1
!        end if
!#endif
        !
        ! for Gaussican core-core values. (see qm1_util_module.src)
        call set_spline_lookup(i_npnt,dr,r_core_val,core_val(1:i_npnt,2), &
                        coef_core_val(1:4,1:i_npnt,2,ii),        &
                        core_val_shift(2,ii))
!#if KEY_PARALLEL==1
!        if(mynod == 0) then
!#endif
!           write(iunit,*) 'for Gaussian core values.'
!           write(iunit,*) dr_width,core_val_shift(2,ii)
!           do j=1,i_npnt
!              write(iunit,*) r_core_val(j),(coef_core_val(ij,j,2,ii),ij=1,4)
!           end do
!#if KEY_PARALLEL==1
!        end if
!#endif
     end if
  end do
  ! re-set r_min and r_max values to avoid large error near the ends.
  !r_min = r_core_val(1)
  r_max = r_core_val(i_npnt-2)
  do i=1,nunique_qm
     core_cut_val(i) = core_cut_val(i) + one
     if(core_cut_val(i) > r_max) then
        core_cut_val(i) = r_max
     end if
  end do
  deallocate(core_val)
  return
  end subroutine mmint_prep_core


  subroutine betaij_prep_lookup(dr,r_min,r_max)
  !
  ! Prepare look-up table for beta_ij values.
  !
  use qm1_info, only :qm_main_r,mm_main_r
  implicit none

  real(chm_real) :: dr,r_min,r_max

  integer, pointer :: i_index_look_up_pair(:,:)=>Null() ! size 2 * n*(n-1)
  real(chm_real),pointer :: beta_val(:,:)=>Null(), &
                            coef_val(:,:)=>Null()
  integer :: i,j,nt,nij,ij,ni,nj,ki,kj,ii,jj,iunique_cnt
  integer :: iorbs,jorbs
  integer :: ni_2,nj_2
  logical :: do_match
  real(chm_real) :: rr,tt(14)
  
  if(associated(q_unique_pair))        deallocate(q_unique_pair)
  if(associated(ij_pair_look_up))      deallocate(ij_pair_look_up)
  if(associated(i_index_look_up_pair)) deallocate(i_index_look_up_pair)

  ! count the total number of pairs to be considered.
  nt  = qm_main_r%numat
  nij = 0
  do i=2,qm_main_r%numat
     ni     = ni_local(i)
     do j=1,i-1
        nj     = ni_local(j)
        if(ni > 1 .or. nj > 1) then
           nij= nij + 1
        end if
     end do
  end do
  nij = nt*(nt-1)
  allocate(q_unique_pair(nij))
  allocate(ij_pair_look_up(nij))
  allocate(i_index_look_up_pair(2,nij))  ! (1,ij) : for i-th atom; (2,ij) for j-th atom.

  q_unique_pair(1:nij) = .true.
  ij = 0
  iunique_cnt = 0
  do i=2,nt
     ni     = ni_local(i)
     do j=1,i-1
        nj     = ni_local(j)
        if(ni > 1 .or. nj > 1) then
           ij = ij + 1
           i_index_look_up_pair(1,ij) = ni
           i_index_look_up_pair(2,ij) = nj

           do_match =.false.
           do ki=1,ij-1
              if(q_unique_pair(ki)) then
                 ni_2 = i_index_look_up_pair(1,ki)
                 nj_2 = i_index_look_up_pair(2,ki)
                 if(ni==ni_2 .and. nj==nj_2) then
                    do_match =.true.
                    exit
                 end if
              end if
           end do
           if(do_match) then
              q_unique_pair(ij)   =.false. ! there is an existing pair.
              ij_pair_look_up(ij) = ij_pair_look_up(ki)
           else
              iunique_cnt=iunique_cnt+1
              q_unique_pair(ij)   =.true.
              ij_pair_look_up(ij) = iunique_cnt
           end if
        end if
     end do
  end do
  deallocate(i_index_look_up_pair)

  ! find num_beta_pnt
  nij = 0
  rr  =r_min
  do
    nij  = nij + 1
    if(rr > r_max) exit
    rr = rr + dr
  end do
  num_beta_pnt = nij

  !
  iunique_cnt_beta = iunique_cnt
  if(associated(look_up_beta_r)) deallocate(look_up_beta_r)
  if(associated(r_beta_val))     deallocate(r_beta_val)
  allocate(r_beta_val(num_beta_pnt))
  allocate(look_up_beta_r(iunique_cnt))

  if(associated(beta_val)) deallocate(beta_val)
  if(associated(coef_val)) deallocate(coef_val)
  allocate(beta_val(num_beta_pnt,14))
  allocate(coef_val(4,num_beta_pnt))

  nij = 0
  rr  = r_min
  do
     nij = nij + 1
     r_beta_val(nij) = rr
     if(rr > r_max) exit
     rr = rr + dr
  end do

  ij = 0
  iunique_cnt = 0
  do i=2,nt
     ni     = ni_local(i)
     iorbs  = iorbs_local(i)
     do j=1,i-1
        nj     = ni_local(j)
        jorbs  = iorbs_local(j)
        if(ni > 1 .or. nj > 1) then
           ij = ij + 1
           if(q_unique_pair(ij)) then
              ! do this pair.
              iunique_cnt=iunique_cnt+1
              select case(iorbs+jorbs)
                 case (2)
                    ii=1
                 case (5)
                    ii=3
                 case (8)
                    ii=5
                 case (10)
                    ii=7
                 case (13)
                    ii=11
                 case (18)
                    ii=15
              end select

              ! memory
              if(associated(look_up_beta_r(iunique_cnt)%coef_val))  deallocate(look_up_beta_r(iunique_cnt)%coef_val)
              if(associated(look_up_beta_r(iunique_cnt)%val_shift)) deallocate(look_up_beta_r(iunique_cnt)%val_shift)
              allocate(look_up_beta_r(iunique_cnt)%coef_val(4,ii,num_beta_pnt))
              allocate(look_up_beta_r(iunique_cnt)%val_shift(ii))
              !
              do ki = 1,num_beta_pnt
                 call betaij(i,j,ni,nj,iorbs,jorbs,r_beta_val(ki),tt)
                 beta_val(ki,1:ii) = tt(1:ii)
              end do

              ! 
              do kj = 1, ii
                 call set_spline_lookup(num_beta_pnt,dr,r_beta_val,beta_val(1:num_beta_pnt,kj), &
                                        coef_val,look_up_beta_r(iunique_cnt)%val_shift(kj))
                 ! copy
                 do ki = 1, num_beta_pnt
                    look_up_beta_r(iunique_cnt)%coef_val(1:4,kj,ki)=coef_val(1:4,ki)
                 end do
              end do
           end if
        end if
     end do
  end do
  
  ! re-set r_min and r_max values to avoid large error near the ends.
  r_max = r_beta_val(num_beta_pnt-2) !

  deallocate(coef_val)
  deallocate(beta_val)
  return
  end subroutine betaij_prep_lookup


  subroutine overlp (iqm,jqm,NI,NJ,iorbs,jorbs,rij,Z)
  !
  ! overlap integrals
  !
  ! Notation: (I) input; (O) output.
  ! Ni, Nj    atomic numbers of atoms i,j (I)
  ! Rij       internuclear distance in atomic units (I)
  ! Z(i)      overlap integrals (O).
  ! 
  !use chm_kinds
  !use number
  !use qm1_constant
  use qm1_parameters,only: III,IIID ! ,ZS,ZP,ZD

  implicit none
  !
  integer :: iqm,jqm,Ni,Nj,iorbs,jorbs
  real(chm_real):: Rij,Z(14)

  ! local variables
  logical :: DIFF
  integer :: i,j,ii,ij,jj,k
  integer :: N1,N2,N1P,N2P,NT
  integer :: ISP,IPS,IOR,JOR,N1D,N2D
  real(chm_real):: ZSI,ZSJ,ZPI,ZPJ,ZDI,ZDJ,ZIMIN,ZJMIN
  real(chm_real):: FAC,SA,SB,PA,PB,W,D,E,rij2,rij3,rtmp2,rtmp3,rtmp5
  real(chm_real):: A(15),B(15)
  real(chm_real),parameter :: r_3   = one/three,    &
                              r_8   = 0.125D0,      &
                              r_16  = 0.0625D0,     &
                              r_32  = 0.03125D0,    &
                              r_7PT5= one/7.5d0,    &
                              r_480 = one/480.0D0

  ! initialize the overlap array
  Z(1:14)=zero
  !
  zsi    = ZS_local(iqm) ! ZS(ni)
  zpi    = ZP_local(iqm) ! ZP(ni)
  zsj    = ZS_local(jqm) ! ZS(nj)
  zpj    = ZP_local(jqm) ! ZP(nj)
  rij2   = rij*rij
  rij3   = rij2*rij

  ! check for immediate return (overlaps below threshold).
  ZIMIN  = MIN(ZSI,ZPI)
  ZJMIN  = MIN(ZSJ,ZPJ)
  if(iorbs.ge.9) ZIMIN = MIN(ZIMIN,ZD_local(iqm))  ! ZD(NI)
  if(jorbs.ge.9) ZJMIN = MIN(ZJMIN,ZD_local(jqm))  ! ZD(NI)
  !if((PT5*(ZIMIN+ZJMIN)*rij).gt.BIGEXP) return

  ! n1, n2 : main quantum numbers for S-orbitals.
  ! n1p,n2p: main quantum numbers for P-orbitals.
  diff   = zsi.ne.zpi .or. zsj.ne.zpj
  n1     = III(ni) 
  n2     = III(nj)
  n1p    = MAX(n1,2) ! MAX(III(ni),2)
  n2p    = MAX(n2,2) ! MAX(III(nj),2)
  nt     = n1+n2

  ! depending on main quantum numbers:
  if(n1.le.3 .and. n2.le.3) then   ! low quantum number atoms.
    if(n1.le.n2) then
       ii  = n2*(n2-1)/2+n1
       isp = 2
       ips = 3
       ior = iorbs
       jor = jorbs
       fac =-one
       sa  = zsi
       pa  = zpi
       sb  = zsj
       pb  = zpj
    else
       ii  = n1*(n1-1)/2+n2
       isp = 3
       ips = 2
       ior = jorbs
       jor = iorbs
       fac = one
       sa  = zsj
       pa  = zpj
       sb  = zsi
       pb  = zpi
    end if

    ! THE ORDERING OF THE ELEMENTS WITHIN Z IS
    ! Z(1)   = S(I)       / S(J)
    ! Z(2)   = S(I)       / P-SIGMA(J)
    ! Z(3)   = P-SIGMA(I) / S(J)
    ! Z(4)   = P-SIGMA(I) / P-SIGMA(J)
    ! Z(5)   = P-PI(I)    / P-PI(J)
    ! Z(6)   = D-SIGMA(J) / S(I)
    ! Z(7)   = S(J)       / D-SIGMA(I)
    ! Z(8)   = D-SIGMA(J) / P-SIGMA(I)
    ! Z(9)   = P-SIGMA(J) / D-SIGMA(I)
    ! Z(10)  = D-PI(J)    / P-PI(I)
    ! Z(11)  = P-PI(J)    / D-PI(I)
    ! Z(12)  = D-SIGMA(J) / D-SIGMA(I)
    ! Z(13)  = D-PI(J)    / D-PI(I)
    ! Z(14)  = D-DELTA(J) / D-DELTA(I)
    select case(ii)   ! since 1<=n1<=3, 1<=n2<=3, 1<=ii<=6.
      case (1)   ! 1st row - 1st row overlaps
        call set(nt,zsi,zsj,A,B,rij)
        if(ni.eq.nj) then
           W   = (zsi*rij)**3
        else
           W   = SQRT((zsi*zsj*rij2)**3)         ! rij*rij
        end if
        Z(1)   = PT25*W*(A(3)*B(1)-B(3)*A(1))
        !if(iorbs.eq.1 .and. jorbs.eq.1) return
        !return
      case (2)   ! 1st row - 2nd row overlaps
        call set(nt,sa,sb,A,B,rij)
        rtmp3  = sa**3
        W      = SQRT((rtmp3)*(sb**5))*(rij2**2)*r_8   ! rij**4
        Z(1)   = W*RT3*(A(4)*B(1)+B(4)*A(1)-A(3)*B(2)-B(3)*A(2))
        if(diff) then
           call set(nt,sa,pb,A,B,rij)
           W   = SQRT((rtmp3)*(pb**5))*(rij2**2)*r_8   ! rij**4
        end if
        Z(isp) = W*fac*(A(3)*B(1)-B(3)*A(1)-A(4)*B(2)+B(4)*A(2))
        !if (jor.lt.9) return
      case (3)   ! 2nd row - 2nd row overlaps
        call set(nt,zsi,zsj,A,B,rij)
        if(ni.eq.nj) then
           W   = r_16*(zsi*rij)**5
        else
           W   = r_16*SQRT((zsi*zsj*rij2)**5)   ! rij*rij
        end if
        Z(1)   = W*(A(5)*B(1)+B(5)*A(1)-two*A(3)*B(3))*r_3
        if(diff) then
           call set(nt,zsi,zpj,A,B,rij)
           W   = r_16*SQRT((zsi*zpj*rij2)**5)   ! rij*rij
        end if
        D      = A(4)*(B(1)-B(3))-A(2)*(B(3)-B(5))
        E      = B(4)*(A(1)-A(3))-B(2)*(A(3)-A(5))
        Z(2)   =-W*RT3*(D-E)
        if(diff) then
           call set(nt,zpi,zsj,A,B,rij)
           W   = r_16*SQRT((zpi*zsj*rij2)**5)
           D   = A(4)*(B(1)-B(3))-A(2)*(B(3)-B(5))
           E   = B(4)*(A(1)-A(3))-B(2)*(A(3)-A(5))
        end if
        Z(3)   = W*RT3*(D+E)
        if(diff) then
           call set(nt,zpi,zpj,A,B,rij)
           if(ni.eq.nj) then
             W = r_16*(zpi*rij)**5
           else
             W = r_16*SQRT((zpi*zpj*rij2)**5)
           end if
        end if
        Z(4)   = W*(B(3)*(A(5)+A(1))-A(3)*(B(5)+B(1)))
        Z(5)   = PT5*W*(A(5)*(B(1)-B(3))-B(5)*(A(1)-A(3))-A(3)*B(1)+B(3)*A(1))
        !if(iorbs.le.4 .and. jorbs.le.4) return
      case (4)   ! 1st row - 3rd row overlaps
        call set(nt,sa,sb,A,B,rij)
        rtmp3= sa**3
        W   = SQRT((rtmp3)*(sb**7)*r_7PT5)*(rij2*rij3)*r_16   ! rij**5
        Z(1)= W*RT3*(A(5)*B(1)-B(5)*A(1)-two*(A(4)*B(2)-B(4)*A(2)))
        if(diff) then
           call set(nt,sa,pb,A,B,rij)
           W= SQRT((rtmp3)*(pb**7)*r_7PT5)*(rij2*rij3)*r_16   ! rij**5
        end if
        Z(isp) = W*fac*(A(4)*(B(1)+B(3))+B(4)*(A(1)+A(3))   &
                       -B(2)*(A(3)+A(5))-A(2)*(B(3)+B(5)))
        !if (jor.le.4) return
      case (5)   ! 2nd row - 3rd row overlaps
        call set(nt,sa,sb,A,B,rij)
        rtmp5 = sa**5
        W      = SQRT((rtmp5)*(sb**7)*r_7PT5)*(rij2**3)*r_32    ! rij**6
        Z(1)   = W*(A(6)*B(1)-A(5)*B(2)-two*(A(4)*B(3)-A(3)*B(4))  &
                   +A(2)*B(5)-A(1)*B(6))*r_3
        if(diff) then
           call set(nt,sa,pb,A,B,rij)
           W   = SQRT((rtmp5)*(pb**7)*r_7PT5)*(rij2**3)*r_32
        end if
        Z(isp) = W*RT3*fac*(-A(6)*B(2)+A(5)*B(1)         &
                            -two*(A(3)*B(3)-A(4)*B(4))   &
                            -A(2)*B(6)+A(1)*B(5))
        if(diff) then
           rtmp5= pa**5
           call set(nt,pa,sb,A,B,rij)
           W   = SQRT((rtmp5)*(sb**7)*r_7PT5)*(rij3**2)*r_32  ! rij**6
        end if
        Z(ips)=W*RT3*fac*(A(5)*(two*B(3)-B(1))-B(5)*(two*A(3)-A(1))   &
                         +A(2)*(B(6)-two*B(4))-B(2)*(A(6)-two*A(4)))
        if(diff) then
           call set(nt,pa,pb,A,B,rij)
           W   = SQRT((rtmp5)*(pb**7)*r_7PT5)*(rij3**2)*r_32
        end if
        Z(4)= W*(-B(4)*(A(1)+A(5))-A(4)*(B(1)+B(5)) +B(3)*(A(2)+A(6))+A(3)*(B(2)+B(6)))
        Z(5)= PT5*W*( A(6)*(B(1)-B(3))+B(6)*(A(1)-A(3))  &
                     -A(5)*(B(2)-B(4))-B(5)*(A(2)-A(4))  &
                     -A(4)*B(1)-B(4)*A(1)+A(3)*B(2)+B(3)*A(2) )
        !if(iorbs.le.4 .and. jorbs.le.4) return
      case (6)  ! 3rd row - 3rd row overlaps
        call set(nt,zsi,zsj,A,B,rij)
        if(ni.eq.nj) then
           W   = ((zsi*rij)**7)*r_480
        else
           W   = SQRT((zsi*zsj*rij2)**7)*r_480
        end if
        Z(1)=W*(A(7)*B(1)-three*(A(5)*B(3)-A(3)*B(5))-A(1)*B(7))*r_3
        if(diff) then
           call set(nt,zsi,zpj,A,B,rij)
           W = SQRT((zsi*zpj*rij2)**7)*r_480
        end if
        D   = A(6)*(B(1)-B(3))-two*A(4)*(B(3)-B(5))+A(2)*(B(5)-B(7))
        E   = B(6)*(A(1)-A(3))-two*B(4)*(A(3)-A(5))+B(2)*(A(5)-A(7))
        Z(2)=-W*RT3*(D+E)
        if(diff) then
           call set(nt,zpi,zsj,A,B,rij)
           W=SQRT((zpi*zsj*rij2)**7)*r_480
           D=A(6)*(B(1)-B(3))-two*A(4)*(B(3)-B(5))+A(2)*(B(5)-B(7))
           E=B(6)*(A(1)-A(3))-two*B(4)*(A(3)-A(5))+B(2)*(A(5)-A(7))
        end if
        Z(3)   = W*RT3*(D-E)
        if(diff) then
           call set(nt,zpi,zpj,A,B,rij)
           if(ni.eq.nj) then
             W = ((zpi*rij)**7)*r_480
           else
             W = SQRT((zpi*zpj*rij2)**7)*r_480
           end if
        end if
        Z(4)   = W*(A(3)*(B(7)+two*B(3))-A(5)*(B(1)+two*B(5))-B(5)*A(1)+A(7)*B(3))
        Z(5)   = PT5*W*(A(7)*(B(1)-B(3))     +B(7)*(A(1)-A(3))        &
                       +A(5)*(B(5)-B(3)-B(1))+B(5)*(A(5)-A(3)-A(1))   &
                       +two*A(3)*B(3))
        !if(iorbs.le.4 .and. jorbs.le.4) return
    end select
  !
  ! overlaps involving higher rows.
  else if(n1.gt.3 .or. n2.gt.3) then
     call set(n1 +n2 ,zsi,zsj,A,B,rij)
     Z(1)   = ss(n1 ,0,0,n2 ,0,zsi*rij,zsj*rij,A,B)
     if(jorbs.ge.4) then
        if(diff) call set(n1 +n2p,zsi,zpj,A,B,rij)
        Z(2) = ss(n1 ,0,0,n2p,1,zsi*rij,zpj*rij,A,B)
     end if
     if(iorbs.ge.4) then
        if(diff) call set(n1p+n2 ,zpi,zsj,A,B,rij)
        Z(3) = ss(n1p,1,0,n2 ,0,zpi*rij,zsj*rij,A,B)
     end if
     if(iorbs.ge.4 .and. jorbs.ge.4) then
        if(diff) call set(n1p+n2p,zpi,zpj,A,B,rij)
        Z(4) = ss(n1p,1,0,n2p,1,zpi*rij,zpj*rij,A,B)
        Z(5) = ss(n1p,1,1,n2p,1,zpi*rij,zpj*rij,A,B)
     end if
     !if(iorbs.le.4 .and. jorbs.le.4) return
  end if
  ! returns, if not having d-orbitals.
  if(iorbs.le.4 .and. jorbs.le.4) return

  ! overlaps involving D-orbitals.
  if(iorbs.ge.9 .or. jorbs.ge.9) then
     zdi    = ZD_local(iqm) ! ZD(ni)
     zdj    = ZD_local(jqm) ! ZD(nj)
     n1d    = IIID(ni) 
     n2d    = IIID(nj) 
     if(iorbs.ge.9 .and. jorbs.le.4) then
        call set(n1d+n2 ,zdi,zsj,A,B,rij)
        Z(6)  = ss(n1d,2,0,n2 ,0,zdi*rij,zsj*rij,A,B)
        if(jorbs.eq.4) then
           call set(n1d+n2p,zdi,zpj,A,B,rij)
           Z(8)  = ss(n1d,2,0,n2p,1,zdi*rij,zpj*rij,A,B)
           Z(10) = ss(n1d,2,1,n2p,1,zdi*rij,zpj*rij,A,B)*three
        end if
     else if(iorbs.le.4 .and. jorbs.ge.9) then
        call set(n1 +n2d,zsi,zdj,A,B,rij)
        Z(7)  = ss(n1 ,0,0,n2d,2,zsi*rij,zdj*rij,A,B)
        if(iorbs.eq.4) then
           call set(n1p+n2d,zpi,zdj,A,B,rij)
           Z(9)  = ss(n1p,1,0,n2d,2,zpi*rij,zdj*rij,A,B)
           Z(11) = ss(n1p,1,1,n2d,2,zpi*rij,zdj*rij,A,B)*three
        end if
     else if(iorbs.ge.9 .and. jorbs.ge.9) then
        call set(n1d+n2 ,zdi,zsj,A,B,rij)
        Z(6)  = ss(n1d,2,0,n2 ,0,zdi*rij,zsj*rij,A,B)
        call set(n1d+n2p,zdi,zpj,A,B,rij)
        Z(8)  = ss(n1d,2,0,n2p,1,zdi*rij,zpj*rij,A,B)
        Z(10) = ss(n1d,2,1,n2p,1,zdi*rij,zpj*rij,A,B)*three
        call set(n1 +n2d,zsi,zdj,A,B,rij)
        Z(7)  = ss(n1 ,0,0,n2d,2,zsi*rij,zdj*rij,A,B)
        call set(n1p+n2d,zpi,zdj,A,B,rij)
        Z(9)  = ss(n1p,1,0,n2d,2,zpi*rij,zdj*rij,A,B)
        Z(11) = ss(n1p,1,1,n2d,2,zpi*rij,zdj*rij,A,B)*three
        call set(n1d+n2d,zdi,zdj,A,B,rij)
        Z(12) = ss(n1d,2,0,n2d,2,zdi*rij,zdj*rij,A,B)
        Z(13) = ss(n1d,2,1,n2d,2,zdi*rij,zdj*rij,A,B)*nine
        Z(14) = ss(n1d,2,2,n2d,2,zdi*rij,zdj*rij,A,B)
     end if
  end if

  return

  !====================================================================!
  contains
      subroutine set(N,SA,SB,A,B,RAB)
      !
      ! calculation of auxiliary integrals for STO overlaps.
      !
      ! on output:
      ! A and B are filled.
      !
      !use chm_kinds
      !use number
      !use qm1_constant

      implicit none

      integer :: N
      real(chm_real):: SA,SB,A(15),B(15),RAB

      ! local variables
      integer :: i,m,last,MA
      real(chm_real):: alpha,beta,C,Y,R_ALPHA,ABSX,EXPX,EXPMX,RX
      real(chm_real):: BETPOW(17)
      !
      real(chm_real),parameter :: CUTOFF=1.0D-06
      ! B0(i) contains the B-integrals for zero argument.
      real(chm_real),parameter:: B0(15)=(/             2.0D0,0.0D0, &
               0.666666666666667D0,0.0D0,              0.4D0,0.0D0, &
               0.285714285714286D0,0.0D0,0.222222222222222D0,0.0D0, &
               0.181818181818182D0,0.0D0,0.153846153846154D0,0.0D0, &
               0.133333333333333D0/)
      ! FC(i) contains the factorials of i-1.
      real(chm_real),parameter:: FC(17)=(/                          & 
                1.0D0,       1.0D0,       2.0D0,        6.0D0,        24.0D0, &
              120.0D0,     720.0D0,    5040.0D0,    40320.0D0,    362880.0D0, &
          3628800.0D0,39916800.0D0,4.790016D+08,6.2270208D+09,8.71782912D+10, &
      1.307674368D+12,2.092278989D+13/)

      ! initializeation
      ALPHA  = PT5*RAB*(SA+SB)
      BETA   = PT5*RAB*(SA-SB)

      ! axsiliary A-integrals for calculation of overlaps.
      C      = EXP(-ALPHA)
      R_ALPHA= ONE/ALPHA
      A(1)   = C*R_ALPHA
      do i=1,n
         A(i+1) = (A(i)*float(i)+C)*R_ALPHA
      end do

      ! auxiliary B-integrals for calculation of overlaps.
      ! The code is valid only for N.le.14, i.e. for overlaps involving
      ! orbtials with main quantum numbers up to 7.
      !
      ABSX   = abs(BETA)
      if(ABSX.lt.CUTOFF) then
         ! zero argument
         B(1:n+1)=B0(1:n+1)
      else
         ! large argument
         if((ABSX.gt.PT5 .and. n.le.5) .or. (ABSX.gt.one .and. n.le.7) .or. &
            (ABSX.gt.two .and. n.le.10).or. (ABSX.gt.three)) then
            EXPX   = EXP(BETA)
            EXPMX  = one/EXPX
            RX     = one/BETA
            B(1)   = (EXPX-EXPMX)*RX
            do i=1,n
               EXPX  = -EXPX
               B(i+1)= (float(i)*B(i)+EXPX-EXPMX)*RX
            end do
         else
            ! small argument
            if(ABSX.le.PT5) then
               last = 6
            else if(ABSX.le.one) then
               last = 7
            else if(ABSX.le.two) then
               last = 12
            else
               last = 15
            end if
            BETPOW(1) = one
            do m=1,last
               BETPOW(m+1) = -BETA*BETPOW(m)
            end do
            do i=1,n+1
               y      = zero
               ma     = 1-MOD(i,2)
               do m=ma,last,2
                  y   = y+BETPOW(m+1)/(FC(m+1)*float(m+i))
               end do
               B(i)   = y*two
            end do
         end if
      end if
      return
      end subroutine set

      real(chm_real) function ss(NA,LA,MM,NB,LB,ALPHA,BETA,A,B)
      !
      ! compute overlap integrals between Slater-type orbitals.
      !
      ! Quantum numbers: (NA,LA,MM) and (NB,LB,MM),
      !                  where Na and Nb must be positive and less than or equal to 7.
      !                  Further restrictions are LA.le.Na, LB.le.Nb,
      !                                           MM.le.LA, and MM.le.LB.
      !
      !use chm_kinds
      !use number
      !use qm1_constant

      implicit none

      integer :: NA,LA,MM,NB,LB
      real(chm_real):: A(15),B(15),ALPHA,BETA

      ! local variables
      integer :: i,j,k,L,N,M
      integer :: IC,ID,IE,IJ,IU,IV,JC,JD,JE,KA,KB,KC,KD,KE,KF
      integer :: IBA,IBB,IBC,IBD,IBE,IBF,IU1,IV1,IUC,IVC,IFF,JFF,LAM,LBM,NAB
      integer :: IADA,IADB,IADM,IADU,IADV,NAMU,NBMV
      integer :: IADNA,IADNB,iexpn,jexpn,il,jl,ik,jk
      real(chm_real):: X,CU,SUM,reduced_factor

      ! addresses for index pairs(00,10,20,30,40,50,60,70).
      integer,parameter :: IAD(8)=(/1,2,4,7,11,16,22,29/)
      ! binomial coefficients(00,10,11,20,...,77).
      integer,parameter :: IBINOM(36)=(/         &
                   1, 1, 1, 1, 2, 1, 1, 3, 3, 1, &
                   1, 4, 6, 4, 1, 1, 5,10,10, 5, &
                   1, 1, 6,15,20,15, 6, 1, 1, 7, &
                  21,35,35,21, 7, 1/)
      ! C-coefficients for associate legendre polynomials.
      real(chm_real),parameter :: C(21,3)=reshape( (/     &
               8.0D0,  8.0D0,  4.0D0, -4.0D0,  4.0D0,  &
               4.0D0,-12.0D0, -6.0D0, 20.0D0,  5.0D0,  &
               3.0D0,-30.0D0,-10.0D0, 35.0D0,  7.0D0,  &
              15.0D0,  7.5D0,-70.0D0,-17.5D0, 63.0D0,  &
              10.5D0,                                  &
               0.0D0,  0.0d0,  0.0d0, 12.0D0,  0.0D0,  &
               0.0d0, 20.0D0, 30.0D0,  0.0D0,  0.0d0,  &
             -30.0D0, 70.0D0, 70.0D0,  0.0D0,  0.0d0,  &
             -70.0D0,-105.D0,210.0D0,157.5D0,  0.0D0,  &
               0.0d0,                                  &
      (0.0d0,i=1,10), 35.0D0,(0.0d0,i=1,4),63.0D0,157.5D0,(0.0d0,i=1,4)/),(/21,3/))
      ! factorials FC(i) of (i-1).
      real(chm_real),parameter :: FC(15)=(/            &
                   1.0D0,        1.0D0,         2.0D0,       6.0D0, &
                  24.0D0,      120.0D0,       720.0D0,    5040.0D0, &
               40320.0D0,   362880.0D0,   3628800.0D0,39916800.0D0, &
            4.790016D+08,6.2270208D+09,8.71782912D+10/)
      real(chm_real),parameter :: r_eight=one/eight

      ! info:
      ! NA, NB : main quantum number
      ! LA, LB : orbital quantum number (l=0 for S; l=1 for P; l=2 for D)
      ! MM     : magnetic quantum number

      ! initialization.
      M      = ABS(MM)
      NAB    = NA+NB+1
      X      = zero
      iexpn  =2*NA+1
      jexpn  =2*NB+1
      reduced_factor=SQRT( ((ALPHA**iexpn)*(BETA**jexpn)) &
                          /( FC(iexpn)    * FC(jexpn)   ) )

      if(LA.eq.0 .and. LB .eq.0) then
      ! overlap integrals involving S-functions.
         iada   = IAD(NA+1)
         iadb   = IAD(NB+1)
         do i=0,na
            iba    = IBINOM(iada+i)
            do j=0,nb
               ibb    = iba*IBINOM(iadb+j)
               if(MOD(j,2).eq.1) ibb=-ibb
               ij     = i+j
               X      = X+float(ibb)*A(nab-ij)*B(ij+1)
            end do
         end do
         SS=X*PT5*reduced_factor

      else if (LA.le.1 .and. LB.le.1) then 
      ! overlap integrals involving P-functions.
         if(M.le.0) then
            ! special case M=0, S-P(SIGMA), P(SIGMA)-S, P(SIGMA)-P(SIGMA).
            iu     = MOD(LA,2)
            iv     = MOD(LB,2)
            namu   = NA-iu
            nbmv   = NB-iv
            iadna  = iad(namu+1)
            iadnb  = iad(nbmv+1)
            do kc=0,iu
               ic     = NAB-iu-iv+kc
               jc     = 1+kc
               do kd=0,iv
                  id     = ic+kd
                  jd     = jc+kd
                  do ke=0,namu
                     ibe    = IBINOM(iadna+ke)
                     ie     = id-ke
                     je     = jd+ke
                     do kf=0,nbmv
                        ibf    = ibe*IBINOM(iadnb+kf)
                        if(MOD(kd+kf,2).eq.1) ibf=-ibf
                        X      = X+float(ibf)*A(ie-kf)*B(je+kf)
                     end do
                  end do
               end do
            end do
            ! overlap integral from reduced overlap integral.
            SS= X*SQRT(float((2*LA+1)*(2*LB+1))*PT25)*reduced_factor
            if(MOD(lb,2).eq.1) SS=-SS
         else
            ! special case LA=LB=M=1, P(PI)-P(PI).
            iadna  = iad(NA)
            iadnb  = iad(NB)
            do ke=0,NA-1
               ibe    = IBINOM(iadna+ke)
               ie     = NAB-ke
               je     = ke+1
               do kf=0,NB-1
                  ibf = ibe*IBINOM(iadnb+kf)
                  if(MOD(kf,2).eq.1) ibf=-ibf
                  i=ie-kf
                  j=je+kf
                         !=(A(i)*B(j)-A(i)*B(j+2)-A(i-2)*B(j)+A(i-2)*B(j+2))
                  X=X+float(ibf)*((A(i)-A(i-2))*(B(j)-B(j+2)))
               end do
            end do
            ! overlap integral from reduced overlap integral.
            SS = X*PT75*reduced_factor
            if(MOD(LB+MM,2).eq.1) SS=-SS
         end if

      else
      ! general case that LA .gt. 1 or LB .gt. 1, M.ge.0.
      ! overlal integrals involving non-S functions.
         lam    = LA-M
         lbm    = LB-M
         iada   = iad(LA+1)+M
         iadb   = iad(LB+1)+M
         iadm   = iad(M+1)
         iu1    = MOD(lam,2)
         iv1    = MOD(lbm,2)
         iuc    = 0
         do iu=iu1,lam,2
            iuc    = iuc+1
            CU     = C(iada,iuc)
            namu   = NA-M-iu
            iadna  = iad(namu+1)
            iadu   = iad(iu+1)
            ivc    = 0
            do iv=iv1,lbm,2
               ivc    = ivc+1
               nbmv   = NB-M-iv
               iadnb  = iad(nbmv+1)
               iadv   = iad(iv+1)
               SUM    = zero ! 0.0D0
               do kc=0,iu
                  ibc    = IBINOM(iadu+kc)
                  ic     = NAB-iu-iv+kc
                  jc     = 1+kc
                  do kd=0,iv
                     ibd    = ibc*IBINOM(iadv+kd)
                     id     = ic+kd
                     jd     = jc+kd
                     do ke=0,namu
                        ibe    = ibd*IBINOM(iadna+ke)
                        ie     = id-ke
                        je     = jd+ke
                        do kf=0,nbmv
                           ibf    = ibe*IBINOM(iadnb+kf)
                           iff    = ie-kf
                           jff    = je+kf
                           do ka=0,M
                              iba    = ibf*IBINOM(iadm+ka)
                              i      = iff-2*ka
                              do kb=0,M
                                 ibb    = iba*IBINOM(iadm+kb)
                                 if(MOD(ka+kb+kd+kf,2).eq.1) ibb=-ibb
                                 j      = jff+2*kb
                                 SUM    = SUM+float(ibb)*A(i)*B(j)
                              end do
                           end do
                        end do
                     end do
                  end do
               end do
               X      = X+SUM*CU*C(iadb,ivc)
            end do
         end do
         ! overlap integral from reduced overlap integral.
         !SS     = X*((FC(M+2)/8.0D0)**2)* SQRT( (2*LA+1)*FC(LA-M+1)*    &
         !         (2*LB+1)*FC(LB-M+1)/(4.0D0*FC(LA+M+1)*FC(LB+M+1)))  &
         !        *reduced_factor
         il=2*LA+1
         jl=2*LB+1
         ik=LA+1
         jk=LB+1
         SS =X*((FC(M+2)*r_eight)**2)                                               &
              *SQRT(float(il)*FC(ik-M)*float(jl)*FC(jk-M)/(four*FC(ik+M)*FC(jk+M))) &
              *reduced_factor
         if(MOD(LB+MM,2).eq.1) SS=-SS
      end if
      return
      end function ss
  !==end of contains===================================================!
  !====================================================================!

  end subroutine overlp


  subroutine repam1_qmqm(iqm,jqm,NI,NJ,R,ENUCLR)
  !
  ! Core repulsion function for AM1 and PM3.
  !
  ! The contributions from the Gaussian terms are added.
  ! 1. H-H pair is separated below as repam1_hh_pair.
  ! 2. QM-MM pair is separated below as repam1_qmmm.
  !
  !use chm_kinds
  !use number
  !use qm1_constant
  use qm1_parameters, only :  BORON1,BORON2,BORON3 ! CORE,GNN,GUESS1,GUESS2,GUESS3,IMPAR

  implicit none

  integer :: iqm,jqm,ni,nj
  real(chm_real):: r,ENUCLR

  real(chm_real),parameter :: CUTOFF=25.0D0

  ! local variables
  integer :: nk,nl,imx,ig,i
  real(chm_real):: ADD,XX,GNIJ

  ADD    = ZERO
  if((ni.eq.5.or.nj.eq.5).and.(iqm_mode.eq.2.or.iqm_mode.eq.4)) then
     ! special section for AM1 and AM1/d: atom pairs involing Boron.
     NK  = NI+NJ-5
     if(NK.eq.1) then       ! B-H pair
        NL=2
     else if(NK.eq.6) then  ! B-C pair
        NL=3
     else if(NK.eq.9.or.NK.eq.17.or.NK.eq.35.or.NK.eq.53) then
        NL=4                ! B-F/Cl/Br/I pairs
     else 
        NL=1                ! all others
     end if
     if(ni.eq.5) then
        ig=iqm
     else if(nj.eq.5) then
        ig=jqm
     end if
     IMPAR_local(ig)=3
     do i=1,3
        GUESS1_local(I,ig)=BORON1(I,NL)
        GUESS2_local(I,ig)=BORON2(I,NL)
        GUESS3_local(I,ig)=BORON3(I,NL)
     end do
  end if
  ! GENERAL SECTION: since it will be only called for QM-QM pair (also not h-h pair).
  ! NI .gt. 0: NI is qm atom, the same for NJ
  do ig=1,IMPAR_local(iqm)
     XX  = GUESS2_local(ig,iqm)*(R-GUESS3_local(ig,iqm))**2 
     if(XX.LT.CUTOFF) ADD = ADD+GUESS1_local(ig,iqm)*EXP(-XX)
  end do
  !end if
  if(NI.eq.NJ) then
     ADD = ADD+ADD
  else ! if(NJ.gt.0) then
     do ig=1,IMPAR_local(jqm)
        XX  = GUESS2_local(ig,jqm)*(R-GUESS3_local(ig,jqm))**2
        if(XX.LT.CUTOFF) ADD = ADD+GUESS1_local(ig,jqm)*EXP(-XX)
     end do
  end if
  ! Small modification for Gaussian Core-Core repulsion scaling.
  !if (NI.gt.0) then
  !   GNIJ= GNN_local(iqm)*CORE_local(iqm)
  !else
  !   GNIJ = one
  !end if
  !if (NJ.gt.0) GNIJ= GNIJ*GNN_local(jqm)*CORE_local(jqm)
  GNIJ= GNN_local(iqm)*GNN_local(jqm)*CORE_local(iqm)*CORE_local(jqm)

  ENUCLR = ENUCLR+GNIJ*ADD/R

  return
  end subroutine repam1_qmqm
  !=====================================================================


  subroutine repam1_hh_pair(iqm,NI,R,ENUCLR)
  !
  ! Core repulsion function for AM1 and PM3 for H-H pair.
  !
  !use chm_kinds
  use qm1_parameters, only : CORE,GNN,GUESS1,GUESS2,GUESS3,IMPAR
  !use number,only : zero

  implicit none

  integer :: iqm,ni
  real(chm_real):: r,ENUCLR

  real(chm_real),parameter :: CUTOFF=25.0D0

  ! local variables
  integer :: nk,nl,imx,ig
  real(chm_real):: ADD,XX,GNIJ

  ADD    = ZERO
  ! since NI=NJ=1.
  do ig=1,IMPAR_local(iqm)
     XX  = GUESS2_local(ig,iqm)*(r-GUESS3_local(ig,iqm))**2
     if(XX.LT.CUTOFF) ADD = ADD+GUESS1_local(ig,iqm)*EXP(-XX)
  end do
  ADD = ADD+ADD
  ! Gaussian Core-Core repulsion scaling.
  GNIJ= GNN_local(iqm)*GNN_local(iqm)*CORE_local(iqm)*CORE_local(iqm)
  !
  ENUCLR = ENUCLR+GNIJ*ADD/R

  return
  end subroutine repam1_hh_pair
  !=====================================================================


  subroutine repam1_qmmm(iqm,NI,NJ,R,ENUCLR)
  !
  ! Duplication of repam1 for QM-MM pair, where NJ=0 corresponds to MM atom.
  !
  !use chm_kinds
  !use number
  !use qm1_constant
  use qm1_parameters, only : BORON1,BORON2,BORON3 ! CORE,GNN,GUESS1,GUESS2,GUESS3,IMPAR

  implicit none

  integer :: ni,nj,iqm
  real(chm_real):: r,ENUCLR

  real(chm_real),parameter :: CUTOFF=25.0D0

  ! local variables
  integer :: nk,nl,imx,ig,i
  real(chm_real):: ADD,XX,GNIJ

  ADD    = ZERO
  ! nj=0
  if((ni.eq.5).and.(iqm_mode.eq.2.or.iqm_mode.eq.4)) then
     ! special section for AM1 and AM1/d. atom pairs involing Boron.
     !NK  = NI+NJ-5   ! since Nj=0 and Ni=5, NK=0
     NL=1                ! all others
     IMPAR_local(iqm)=3
     do i=1,3
        GUESS1_local(i,iqm)=BORON1(i,NL)
        GUESS2_local(i,iqm)=BORON2(i,NL)
        GUESS3_local(i,iqm)=BORON3(i,NL)
     end do
  end if
  ! QM-MM specific...
  ! Since ni is qm atom, Ni>0, and since nj is mm atom, Nj=0
  !if(NI.gt.0) then
  do ig=1,IMPAR_local(iqm)
     XX  = GUESS2_local(ig,iqm)*(R-GUESS3_local(ig,iqm))**2 
     if(XX.LT.CUTOFF) ADD = ADD+GUESS1_local(ig,iqm)*EXP(-XX)
  end do
  !end if
  !if(NI.eq.NJ) then
  !   ADD = ADD+ADD
  !else if(NJ.gt.0) then
  !   do ig=1,IMPAR(nj)
  !      XX  = GUESS2(ig,nj)*(R-GUESS3(ig,nj))**2
  !      if(XX.LT.CUTOFF) ADD = ADD+GUESS1(ig,nj)*EXP(-XX)
  !   end do
  !end if
  ! Small modification for Gaussian Core-Core repulsion scaling.
  !if (NI.gt.0) then
  GNIJ= GNN_local(iqm)*CORE_local(iqm)
  !else
  !   GNIJ = one
  !end if
  !if (NJ.gt.0) GNIJ= GNIJ*GNN(NJ)*CORE(NJ)

  ENUCLR = ENUCLR+GNIJ*ADD/R

  return
  end subroutine repam1_qmmm
  !=====================================================================


  subroutine repp_qmqm(iqm,jqm,NI,NJ,R,A,CORE_mat)
  !
  ! calculation of the two-center two-electron integrals and the core-electron
  ! attraction integrals in local coordinates.
  !
  ! NOTATION. I=INPUT, O=OUTPUT.
  ! ni         atomic number of atom i (I).
  ! nJ         atomic number of atom j (I).
  ! r          interatomic distance, in atomic units (au) (I).
  ! a(22)      local two-center two-eleectron integrals, in eV (O).
  ! core_mat() local two-center core-electron integrals, in eV (O).
  !
  ! point-charge multipoles from TCA 1977 are employed (SP basis).
  ! by default, penetration integrals are neglected.
  ! however, if the additive term PO(9,N) for the core differs
  ! from the additive term PO(1,N) for SS, the core-electron
  ! attraction integrals are evaluated explicitly using PO(9,N).
  ! in this case, PO(1,N) and PO(7,N) are the additive terms
  ! for the monopoles of SS and PP, respectively.
  !
  !use chm_kinds
  !use number
  !use qm1_constant
  use qm1_parameters, only : LORBS ! ,CORE,DD,PO
  use qm1_info,only : qm_control_r
            
  implicit none

  integer :: iqm,jqm,NI,NJ
  real(chm_real):: R,A(22),CORE_mat(10,2)

  ! local variables:
  integer :: iorbs,jorbs,i
  real(chm_real):: CORENI,CORENJ,COREV,R2,EE,DA,DB,QA,QB,             &
                   ACI,ACJ,ADD,ADE,ADI,ADJ,ADQ,AED,AEE,AEI,AEJ,AEQ,   &
                   AQD,AQE,AQI,AQJ,AQQ,DZE,EDZ,                       &
                   DXDX,DZDZ,EQXX,EQZZ,QXXE,QZZE,                     &
                   DMADD,DPADD,QMADD,QPADD,                           &
                   DXQXZ,QXZDX,DZQXX,DZQZZ,QXXDZ,QZZDZ,               &
                   QXXQXX,QXXQYY,QXXQZZ,QZZQXX,QZZQZZ,QXZQXZ,         &
                   RMDA2,RMDB2,RPDA2,RPDB2,RMQA2,RPQA2,               &
                   RM2QA2,RM2QB2,RP2QA2,RP2QB2,                       &
                   TWOQA,TWOQB,TWODA,TWOQA2,TWOQB2,TWOQAQ,TWOQBQ,     &
                   X89,X1010,X1011,X1213,X1111,X2021,X2324

  real(chm_real):: X(69)
  real(chm_real),parameter :: small=1.0D-06
  real(chm_real),parameter :: PXX(33)=(/                              &
                     1.0D0   ,-0.5D0   ,-0.5D0   , 0.5D0   , 0.25D0  ,&
                     0.25D0  , 0.5D0   , 0.25D0  ,-0.25D0  , 0.25D0  ,&
                    -0.25D0  ,-0.125D0 ,-0.125D0 , 0.50D0  , 0.125D0 ,&
                     0.125D0 ,-0.5D0   ,-0.25D0  ,-0.25D0  ,-0.25D0  ,&
                     0.25D0  ,-0.125D0 , 0.125D0 ,-0.125D0 , 0.125D0 ,&
                     0.125D0 , 0.25D0  , 0.0625D0, 0.0625D0,-0.25D0  ,&
                     0.25D0  , 0.25D0  ,-0.25D0  /)
  real(chm_real),parameter :: PXY(69)=(/                              &
                     1.0D0   ,-0.5D0   ,-0.5D0   , 0.5D0   , 0.25D0  ,&
                     0.25D0  , 0.5D0   , 0.25D0  ,-0.25D0  , 0.25D0  ,&
                    -0.25D0  ,-0.125D0 ,-0.125D0 ,-0.50D0  ,-0.50D0  ,&
                     0.5D0   , 0.25D0  , 0.25D0  , 0.5D0   , 0.25D0  ,&
                    -0.25D0  ,-0.25D0  ,-0.125D0 ,-0.125D0 , 0.5D0   ,&
                    -0.5D0   , 0.25D0  , 0.25D0  ,-0.25D0  ,-0.25D0  ,&
                    -0.25D0  , 0.25D0  ,-0.25D0  , 0.25D0  ,-0.125D0 ,&
                     0.125D0 ,-0.125D0 , 0.125D0 ,-0.125D0 , 0.125D0 ,&
                    -0.125D0 , 0.125D0 , 0.125D0 , 0.125D0 , 0.25D0  ,&
                     0.125D0 , 0.125D0 , 0.125D0 , 0.125D0 , 0.0625D0,&
                     0.0625D0, 0.0625D0, 0.0625D0,-0.25D0  , 0.25D0  ,&
                     0.25D0  ,-0.25D0  ,-0.25D0  , 0.25D0  , 0.25D0  ,&
                    -0.25D0  , 0.125D0 ,-0.125D0 ,-0.125D0 , 0.125D0 ,&
                    -0.125D0 , 0.125D0 , 0.125D0 ,-0.125D0 /)


  ! initializations for point MM charge vs. QM charge.
  iorbs  = LORBS(ni)
  jorbs  = LORBS(nj)
  coreni = CORE_local(iqm)  !CORE(ni)
  corenj = CORE_local(jqm)  !CORE(nj)
  !
  R2        = R*R
  AEE       =(PO_1(iqm)+PO_1(jqm))**2  ! (PO(1,ni)+PO(1,nj))**2
  ! H - H pair
  if(iorbs.le.1 .and. jorbs.le.1) then
     EE     = one/SQRT(R2+AEE)
     A(1)   = EE*eV
     CORE_mat(1,1) = -CORENJ*A(1)
     CORE_mat(1,2) = -CORENI*A(1)
  ! Heavy atom - H pair
  else if(iorbs.ge.4 .and. jorbs.le.1) then
     DA     = DD_2(iqm) ! DD(2,ni)
     QA     = DD_3(iqm) ! DD(3,ni)
     TWOQA  = QA+QA
     ADE    = (PO_2(iqm)+PO_1(jqm))**2  ! (PO(2,ni)+PO(1,nj))**2
     AQE    = (PO_3(iqm)+PO_1(jqm))**2  ! (PO(3,ni)+PO(1,nj))**2
     X(1)   = (R2+AEE)
     X(2)   = (R2+AQE)
     X(3)   = ((R+DA)**2+ADE)
     X(4)   = ((R-DA)**2+ADE)
     X(5)   = ((R-TWOQA)**2+AQE)
     X(6)   = ((R+TWOQA)**2+AQE)
     X(7)   = (R2+TWOQA*TWOQA+AQE)
     X(1:7) = PXY(1:7)/SQRT(X(1:7))
     A(1)   =  X(1)*eV
     A(2)   = (X(3)+X(4))*eV
     A(3)   = (X(1)+X(2)+X(5)+X(6))*eV
     A(4)   = (X(1)+X(2)+X(7))*eV
     CORE_mat(1:4,1)= -CORENJ * A(1:4)
     CORE_mat(1,2)  = -CORENI * A(1)
  ! H - Heavy atom pair
  else if(iorbs.le.1 .and. jorbs.ge.4) then
     DB     = DD_2(jqm) ! DD(2,nj)
     QB     = DD_3(jqm) ! DD(3,nj)
     TWOQB  = QB+QB
     AED    = (PO_1(iqm)+PO_2(jqm))**2  ! (PO(1,ni)+PO(2,nj))**2
     AEQ    = (PO_1(iqm)+PO_3(jqm))**2  ! (PO(1,ni)+PO(3,nj))**2
     X(1)   = (R2+AEE)
     X(2)   = (R2+AEQ)
     X(3)   = ((R-DB)**2+AED)
     X(4)   = ((R+DB)**2+AED)
     X(5)   = ((R-TWOQB)**2+AEQ)
     X(6)   = ((R+TWOQB)**2+AEQ)
     X(7)   = (R2+TWOQB*TWOQB+AEQ)
     X(1:7) = PXY(1:7)/SQRT(X(1:7))
     A(1)   =  X(1)*eV
     A(5)   = (X(3)+X(4))*eV
     A(11)  = (X(1)+X(2)+X(5)+X(6))*eV
     A(12)  = (X(1)+X(2)+X(7))*eV
     CORE_mat(1,1) = -CORENJ * A(1)
     CORE_mat(1,2) = -CORENI * A(1)
     CORE_mat(2,2) = -CORENI * A(5)
     CORE_mat(3,2) = -CORENI * A(11)
     CORE_mat(4,2) = -CORENI * A(12)
  ! Heavy atom - Heavy atom pair
  else
     DA     = DD_2(iqm) ! DD(2,ni)
     QA     = DD_3(iqm) ! DD(3,ni)
     TWOQA  = QA+QA
     TWOQA2 = TWOQA*TWOQA
     RPDA2  = (R+DA)**2
     RMDA2  = (R-DA)**2
     RP2QA2 = (R+TWOQA)**2
     RM2QA2 = (R-TWOQA)**2
     ADE    = (PO_2(iqm)+PO_1(jqm))**2  ! (PO(2,ni)+PO(1,nj))**2
     AQE    = (PO_3(iqm)+PO_1(jqm))**2  ! (PO(3,ni)+PO(1,nj))**2
     ADD    = (PO_2(iqm)+PO_2(jqm))**2  ! (PO(2,ni)+PO(2,nj))**2
     ADQ    = (PO_2(iqm)+PO_3(jqm))**2  ! (PO(2,ni)+PO(3,nj))**2
     AQQ    = (PO_3(iqm)+PO_3(jqm))**2  ! (PO(3,ni)+PO(3,nj))**2
     TWOQAQ = TWOQA2+AQQ
     X(1)   = R2+AEE
     X(2)   = R2+AQE
     X(3)   = RPDA2+ADE
     X(4)   = RMDA2+ADE
     X(5)   = RM2QA2+AQE
     X(6)   = RP2QA2+AQE
     X(7)   = R2+TWOQA2+AQE
     X(8)   = RPDA2+ADQ
     X(9)   = RMDA2+ADQ
     X(10)  = R2+AQQ
     X(11)  = R2+TWOQAQ
     X(12)  = RP2QA2+AQQ
     X(13)  = RM2QA2+AQQ

     if(ni.ne.nj) then
        DB     = DD_2(jqm) ! DD(2,nj)
        QB     = DD_3(jqm) ! DD(3,nj)
        TWOQB  = QB+QB
        TWOQB2 = TWOQB*TWOQB
        TWOQBQ = TWOQB2+AQQ
        RPDB2  = (R+DB)**2
        RMDB2  = (R-DB)**2
        RP2QB2 = (R+TWOQB)**2
        RM2QB2 = (R-TWOQB)**2
        AED    = (PO_1(iqm)+PO_2(jqm))**2  ! (PO(1,ni)+PO(2,nj))**2
        AEQ    = (PO_1(iqm)+PO_3(jqm))**2  ! (PO(1,ni)+PO(3,nj))**2
        AQD    = (PO_3(iqm)+PO_2(jqm))**2  ! (PO(3,ni)+PO(2,nj))**2
        X(14)  = R2+AEQ
        X(15)  = RMDB2+AED
        X(16)  = RPDB2+AED
        X(17)  = RM2QB2+AEQ
        X(18)  = RP2QB2+AEQ
        X(19)  = R2+TWOQB2+AEQ
        X(20)  = RMDB2+AQD
        X(21)  = RPDB2+AQD
        X(22)  = R2+TWOQBQ
        X(23)  = RP2QB2+AQQ
        X(24)  = RM2QB2+AQQ
        X(25)  = R2+(DA-DB)**2+ADD
        X(26)  = R2+(DA+DB)**2+ADD
        X(27)  = (R+DA-DB)**2+ADD
        X(28)  = (R-DA+DB)**2+ADD
        X(29)  = (R-DA-DB)**2+ADD
        X(30)  = (R+DA+DB)**2+ADD
        X(31)  = RPDA2+TWOQB2+ADQ
        X(32)  = RMDA2+TWOQB2+ADQ
        X(33)  = RMDB2+TWOQA2+AQD
        X(34)  = RPDB2+TWOQA2+AQD
        X(35)  = (R+DA-TWOQB)**2+ADQ
        X(36)  = (R-DA-TWOQB)**2+ADQ
        X(37)  = (R+DA+TWOQB)**2+ADQ
        X(38)  = (R-DA+TWOQB)**2+ADQ
        X(39)  = (R+TWOQA-DB)**2+AQD
        X(40)  = (R+TWOQA+DB)**2+AQD
        X(41)  = (R-TWOQA-DB)**2+AQD
        X(42)  = (R-TWOQA+DB)**2+AQD
        X(43)  = R2+FOUR*(QA-QB)**2+AQQ
        X(44)  = R2+FOUR*(QA+QB)**2+AQQ
        X(45)  = R2+TWOQA2+TWOQBQ
        X(46)  = RM2QB2+TWOQAQ
        X(47)  = RP2QB2+TWOQAQ
        X(48)  = RP2QA2+TWOQBQ
        X(49)  = RM2QA2+TWOQBQ
        X(50)  = (R+TWOQA-TWOQB)**2+AQQ
        X(51)  = (R+TWOQA+TWOQB)**2+AQQ
        X(52)  = (R-TWOQA-TWOQB)**2+AQQ
        X(53)  = (R-TWOQA+TWOQB)**2+AQQ
        X(54)  = (R-QB)**2+(DA-QB)**2+ADQ
        X(55)  = (R+QB)**2+(DA-QB)**2+ADQ
        X(56)  = (R-QB)**2+(DA+QB)**2+ADQ
        X(57)  = (R+QB)**2+(DA+QB)**2+ADQ
        X(58)  = (R+QA)**2+(QA-DB)**2+AQD
        X(59)  = (R-QA)**2+(QA-DB)**2+AQD
        X(60)  = (R+QA)**2+(QA+DB)**2+AQD
        X(61)  = (R-QA)**2+(QA+DB)**2+AQD
        QMADD  = (QA-QB)**2+AQQ
        QPADD  = (QA+QB)**2+AQQ
        X(62)  = (R+QA-QB)**2+QMADD
        X(63)  = (R+QA+QB)**2+QMADD
        X(64)  = (R-QA-QB)**2+QMADD
        X(65)  = (R-QA+QB)**2+QMADD
        X(66)  = (R+QA-QB)**2+QPADD
        X(67)  = (R+QA+QB)**2+QPADD
        X(68)  = (R-QA-QB)**2+QPADD
        X(69)  = (R-QA+QB)**2+QPADD
        X(1:69)= PXY(1:69)/SQRT(X(1:69))
        EE     = X(1)
        DZE    = X(3) +X(4)
        QZZE   = X(2) +X(5) +X(6)
        QXXE   = X(2) +X(7)
        EDZ    = X(15)+X(16)
        EQZZ   = X(14)+X(17)+X(18)
        EQXX   = X(14)+X(19)
        DXDX   = X(25)+X(26)
        DZDZ   = X(27)+X(28)+X(29)+X(30)
        X89    = X(8) +X(9)
        X2021  = X(20)+X(21)
        DZQXX  = X89  +X(31)+X(32)
        QXXDZ  = X2021+X(33)+X(34)
        DZQZZ  = X89  +X(35)+X(36)+X(37)+X(38)
        QZZDZ  = X2021+X(39)+X(40)+X(41)+X(42)
        X1011  = X(10)+X(11)
        X1213  = X(12)+X(13)
        X2324  = X(23)+X(24)
        QXXQXX = X1011+X(22)+X(43)+X(44)
        QXXQYY = X1011+X(22)+X(45)
        QXXQZZ = X1011+X2324+X(46)+X(47)
        QZZQXX = X(10)+X1213+X(22)+X(48)+X(49)
        QZZQZZ = X(10)+X1213+X2324+X(50)+X(51)+X(52)+X(53)
        DXQXZ  = X(54)+X(55)+X(56)+X(57)
        QXZDX  = X(58)+X(59)+X(60)+X(61)
        QXZQXZ = ZERO
        do I=62,69
           QXZQXZ = QXZQXZ+X(I)
        end do
     else   ! meaning ni.eq.nj
        TWODA  = DA+DA
        X(14)  = R2+ADD
        X(15)  = RP2QA2+TWOQAQ
        X(16)  = RM2QA2+TWOQAQ
        X(17)  = R2+TWODA**2+ADD
        X(18)  = (R-TWODA)**2+ADD
        X(19)  = (R+TWODA)**2+ADD
        X(20)  = RPDA2+TWOQA2+ADQ
        X(21)  = RMDA2+TWOQA2+ADQ
        X(22)  = (R+DA-TWOQA)**2+ADQ
        X(23)  = (R-DA-TWOQA)**2+ADQ
        X(24)  = (R+DA+TWOQA)**2+ADQ
        X(25)  = (R-DA+TWOQA)**2+ADQ
        X(26)  = R2+FOUR*TWOQA2+AQQ
        X(27)  = R2+TWOQA2+TWOQAQ
        X(28)  = (R+TWOQA+TWOQA)**2+AQQ
        X(29)  = (R-TWOQA-TWOQA)**2+AQQ
        RMQA2  = (R-QA)**2
        RPQA2  = (R+QA)**2
        DMADD  = (DA-QA)**2+ADQ
        DPADD  = (DA+QA)**2+ADQ
        X(30)  = RMQA2+DMADD
        X(31)  = RPQA2+DMADD
        X(32)  = RMQA2+DPADD
        X(33)  = RPQA2+DPADD
        X(1:33)= PXX(1:33)/SQRT(X(1:33))
        EE     = X(1)
        DZE    = X(3) +X(4)
        QZZE   = X(2) +X(5) +X(6)
        QXXE   = X(2) +X(7)
        EDZ    =-DZE
        EQZZ   = QZZE
        EQXX   = QXXE
        DXDX   = X(14)+X(17)
        DZDZ   = X(14)+X(18)+X(19)
        X89    = X(8) +X(9)
        DZQXX  = X89  +X(20)+X(21)
        QXXDZ  =-DZQXX
        DZQZZ  = X89  +X(22)+X(23)+X(24)+X(25)
        QZZDZ  =-DZQZZ
        X1010  = X(10)+X(10)*PT5
        X1111  = X(11)+X(11)
        X1213  = X(12)+X(13)
        QXXQXX = X1010+X1111+X(26)
        QXXQYY = X1111+X(10)+X(27)
        QXXQZZ = X(10)+X(11)+X1213+X(15)+X(16)
        QZZQXX = QXXQZZ
        QZZQZZ = X1010+X1213+X1213+X(28)+X(29)
        DXQXZ  = X(30)+X(31)+X(32)+X(33)
        QXZDX  =-DXQXZ
        QXZQXZ = QXXQZZ
     end if
     A(1)  = EE
     A(2)  = DZE
     A(3)  = EE + QZZE
     A(4)  = EE + QXXE
     A(5)  = EDZ
     A(6)  = DZDZ
     A(7)  = DXDX
     A(8)  = EDZ + QZZDZ
     A(9)  = EDZ + QXXDZ
     A(10) = QXZDX
     A(11) = EE  + EQZZ
     A(12) = EE  + EQXX
     A(13) = DZE + DZQZZ
     A(14) = DZE + DZQXX
     A(15) = DXQXZ
     A(16) = EE + EQZZ + QZZE + QZZQZZ
     A(17) = EE + EQZZ + QXXE + QXXQZZ
     A(18) = EE + EQXX + QZZE + QZZQXX
     A(19) = EE + EQXX + QXXE + QXXQXX
     A(20) = QXZQXZ
     A(21) = EE + EQXX + QXXE + QXXQYY
     A(22) = PT5*(A(19)-A(21))
     A(1:22)    = A(1:22)*eV
     CORE_mat(1:4,1)= -CORENJ * A(1:4)
     CORE_mat(1,2)  = -CORENI * A(1)
     CORE_mat(2,2)  = -CORENI * A(5)
     CORE_mat(3,2)  = -CORENI * A(11)
     CORE_mat(4,2)  = -CORENI * A(12)
  end if
  ! 
  ! for d-orbitals:
  ! Calculate the nuclear attraction integrals in local coordinates with a separate additive term 
  ! for the Core-mat (SP basis). Omit the calculation for identical additive terms (SS=Core_mat). 
  ! This option is only valid for SP-type integrals in MNDO/d and AM1/d methods.
  if(do_d_orbitals) then
     ACI    = PO_9(iqm) ! PO(9,ni)
     ACJ    = PO_9(jqm) ! PO(9,nj)
     ! electrons at atom A (ni) and Core of atom B (nj).
     if(abs(ACJ-PO_1(jqm)).gt.small) then
        CORE_mat(1,1) = -CORENJ*eV/SQRT(R2+(PO_1(iqm)+ACJ)**2)
        if(iorbs.ge.4) then
           DA     = DD_2(iqm)
           QA     = DD_3(iqm)
           AEJ    = (PO_7(iqm)+ACJ)**2
           ADJ    = (PO_2(iqm)+ACJ)**2
           AQJ    = (PO_3(iqm)+ACJ)**2
           TWOQA  = QA+QA
           X(1)   = (R2+AEJ)
           X(2)   = (R2+AQJ)
           X(3)   = ((R+DA)**2+ADJ)
           X(4)   = ((R-DA)**2+ADJ)
           X(5)   = ((R-TWOQA)**2+AQJ)
           X(6)   = ((R+TWOQA)**2+AQJ)
           X(7)   = (R2+TWOQA*TWOQA+AQJ)
           X(1:7) = PXY(1:7)/SQRT(X(1:7))
           COREV  = CORENJ*eV
           CORE_mat(2,1) = -COREV * (X(3)+X(4))
           CORE_mat(3,1) = -COREV * (X(1)+X(2)+X(5)+X(6))
           CORE_mat(4,1) = -COREV * (X(1)+X(2)+X(7))
        end if
     end if
     ! electrons at atom B (nj) and core of atom A (ni).
     if(abs(ACI-PO_1(iqm)).gt.small) then
        CORE_mat(1,2) = -CORENI*eV/SQRT(R2+(PO_1(jqm)+ACI)**2)
        if(jorbs.ge.4) then
           DB     = DD_2(jqm)
           QB     = DD_3(jqm)
           AEI    = (PO_7(jqm)+ACI)**2
           ADI    = (PO_2(jqm)+ACI)**2
           AQI    = (PO_3(jqm)+ACI)**2
           TWOQB  = QB+QB
           X(1)   = (R2+AEI)
           X(2)   = (R2+AQI)
           X(3)   = ((R+DB)**2+ADI)
           X(4)   = ((R-DB)**2+ADI)
           X(5)   = ((R-TWOQB)**2+AQI)
           X(6)   = ((R+TWOQB)**2+AQI)
           X(7)   = (R2+TWOQB*TWOQB+AQI)
           X(1:7) = PXY(1:7)/SQRT(X(1:7))
           COREV  = CORENI*eV
           CORE_mat(2,2) =  COREV * (X(3)+X(4))
           CORE_mat(3,2) = -COREV * (X(1)+X(2)+X(5)+X(6))
           CORE_mat(4,2) = -COREV * (X(1)+X(2)+X(7))
        end if
     end if
  end if
  return
  end subroutine repp_qmqm


  subroutine repp_qmmm(NI,NJ,iqm,R,A,CORE_mat)
  !
  ! for MM charge: NJ=0
  !
  ! calculation of the two-center two-electron integrals and the core-electron
  ! attraction integrals in local coordinates.
  !
  ! NOTATION. I=INPUT, O=OUTPUT.
  ! ni         atomic number of atom i (I).
  ! nJ         atomic number of atom j (I).
  ! r          interatomic distance, in atomic units (au) (I).
  ! a(22)      local two-center two-eleectron integrals, in eV (O).
  ! core_mat() local two-center core-electron integrals, in eV (O).
  !
  ! point-charge multipoles from TCA 1977 are employed (SP basis).
  ! by default, penetration integrals are neglected.
  ! however, if the additive term PO(9,N) for the core differs
  ! from the additive term PO(1,N) for SS, the core-electron
  ! attraction integrals are evaluated explicitly using PO(9,N).
  ! in this case, PO(1,N) and PO(7,N) are the additive terms
  ! for the monopoles of SS and PP, respectively.
  !
  ! SPECIAL CONVENTION: NI=0 OR NJ=0 DENOTES AN EXTERNAL POINT
  ! CHARGE WITHOUT BASIS ORBITALS. THE CHARGE IS 1 ATOMIC UNIT.
  ! THE VALUES OF DD(I,0) AND PO(I,0) ARE DEFINED TO BE ZERO.
  !
  !use chm_kinds
  !use number
  !use qm1_constant
  use qm1_parameters, only : LORBS ! ,CORE,DD,PO
  use qm1_info, only : qm_control_r
    
  implicit none

  integer :: NI,NJ,iqm
  real(chm_real):: R,A(22),CORE_mat(10,2)

  ! local variables:
  integer :: iorbs,jorbs,i
  real(chm_real):: CORENI,CORENJ,COREV,R2,EE,DA,DB,QA,QB,             &
                   ACI,ACJ,ADE,ADI,ADJ,AEE,AEI,AEJ,                   &
                   AQE,AQI,AQJ,TWOQA,TWOQB

  real(chm_real):: X(69)
  real(chm_real),parameter :: small=1.0D-06
  real(chm_real),parameter :: PXY(69)=(/                              &
                     1.0D0   ,-0.5D0   ,-0.5D0   , 0.5D0   , 0.25D0  ,&
                     0.25D0  , 0.5D0   , 0.25D0  ,-0.25D0  , 0.25D0  ,&
                    -0.25D0  ,-0.125D0 ,-0.125D0 ,-0.50D0  ,-0.50D0  ,&
                     0.5D0   , 0.25D0  , 0.25D0  , 0.5D0   , 0.25D0  ,&
                    -0.25D0  ,-0.25D0  ,-0.125D0 ,-0.125D0 , 0.5D0   ,&
                    -0.5D0   , 0.25D0  , 0.25D0  ,-0.25D0  ,-0.25D0  ,&
                    -0.25D0  , 0.25D0  ,-0.25D0  , 0.25D0  ,-0.125D0 ,&
                     0.125D0 ,-0.125D0 , 0.125D0 ,-0.125D0 , 0.125D0 ,&
                    -0.125D0 , 0.125D0 , 0.125D0 , 0.125D0 , 0.25D0  ,&
                     0.125D0 , 0.125D0 , 0.125D0 , 0.125D0 , 0.0625D0,&
                     0.0625D0, 0.0625D0, 0.0625D0,-0.25D0  , 0.25D0  ,&
                     0.25D0  ,-0.25D0  ,-0.25D0  , 0.25D0  , 0.25D0  ,&
                    -0.25D0  , 0.125D0 ,-0.125D0 ,-0.125D0 , 0.125D0 ,&
                    -0.125D0 , 0.125D0 , 0.125D0 ,-0.125D0 /)


  ! initializations for point MM charge vs. QM charge.
  iorbs  = LORBS(ni)  ! qm atom
  !coreni = CORE_local(iqm)
  !
  jorbs  = 1          ! mm atom
  corenj = one
  !
  R2        = R*R
  AEE       =(PO_1(iqm)+PO_1_mm)**2 ! (PO(1,ni)+PO(1,nj))**2
  ! H - H pair
  if(iorbs.le.1) then  ! (iorbs.le.1 .and. jorbs.le.1)
     EE     = one/SQRT(R2+AEE)
     A(1)   = EE*eV
     CORE_mat(1,1) = -CORENJ*A(1)
  ! Heavy atom - H pair
  else if(iorbs.ge.4) then ! (iorbs.ge.4 .and. jorbs.le.1)
     DA     = DD_2(iqm)  ! DD(2,ni)
     QA     = DD_3(iqm)  ! DD(3,ni)
     TWOQA  = QA+QA
     ADE    = (PO_2(iqm)+PO_1_mm)**2  ! (PO(2,ni)+PO(1,nj))**2
     AQE    = (PO_3(iqm)+PO_1_mm)**2  ! (PO(3,ni)+PO(1,nj))**2
     X(1)   = (R2+AEE)
     X(2)   = (R2+AQE)
     X(3)   = ((R+DA)**2+ADE)
     X(4)   = ((R-DA)**2+ADE)
     X(5)   = ((R-TWOQA)**2+AQE)
     X(6)   = ((R+TWOQA)**2+AQE)
     X(7)   = (R2+TWOQA*TWOQA+AQE)
     X(1:7) = PXY(1:7)/SQRT(X(1:7))
     A(1)   =  X(1)*eV
     A(2)   = (X(3)+X(4))*eV
     A(3)   = (X(1)+X(2)+X(5)+X(6))*eV
     A(4)   = (X(1)+X(2)+X(7))*eV
     CORE_mat(1:4,1)= -CORENJ * A(1:4)
  end if
  return
  end subroutine repp_qmmm


  subroutine rotate(IW,JW,IP,JP,KR,RI,YY,W,WW,LM6,LM9,IMODE)
  ! hcorep   rotate(IW,JW,IP,JP,KR,RI,YY,W,W ,LM6  ,LM9,0)     ; where W(LM6,LM6; LM9)
  ! dhcore   rotate(iw,jw,ip,jp,kr,RI,YY,W,W ,iw+jw,LM9,1)     ; where W(45 ,45 ; lmw=2025)
  !
  ! two-electron repulsion integrals: transformation from local to mol. coords.
  !
  ! Storage of the transofrmed integrals for IMODE
  ! IMODE : 0, square array WW(LM6,LM6); calls from HCOREP.
  !       : 1, linear array W(LM9)     ; calls from DHCORE.
  ! A given call either refers to W(LM9) or WW(LM6,LM6).
  !
  ! INPUT
  ! IW,JW    number of one-center pairs at atoms I and J.
  ! IP,JP    address of (SS,SS) in WW(LM6,LM6).
  ! KR+1     address of (SS,SS) in W(LM9); only used in calls from dhcore.
  ! RI       local two-electron integrals.
  ! YY       precombined rotation matrix elements.
  !
  !use chm_kinds
  !use number
  !use qm1_constant

  implicit none

  integer:: iw,jw,ip,jp,kr,LM6,LM9,IMODE
  real(chm_real):: RI(22),YY(15,45),W(LM9),WW(LM6,LM6)

  ! local variables
  integer:: i,j,ij,k
  real(chm_real):: rsum(6),yy_4(3),yy_5(6),yy_6(6),yy_7(3),yy_8(6),yy_9(6),yy_10(6)
  real(chm_real):: T(9,6),SSPB(9),rPASS(9),PSPS(6),PSPP(6,3),PPPS(3,6),PPPP(6,6)
  integer,parameter:: IPP(3)=(/ 1, 3, 6/)
  integer,parameter:: JPP(3)=(/10,30,60/)
  integer,parameter:: ISS(6)=(/ 2, 4, 5, 7, 8, 9/)
  integer,parameter:: JSS(6)=(/20,40,50,70,80,90/)


  ! transform the integrals
  !if(iw.eq.1 .and. jw.eq.1) then
  !   continue
  !else
  if (iw.gt.1 .or. jw.gt.1) then
     rsum(1:6)=yy(1:6,6)+yy(1:6,10)

     ! integral types (SS,PS) and (SS,PP).
     if(jw.gt.1) then
        SSPB(IPP(1:3))= RI(5)*YY(1:3,2)  ! ipp=1,3,6
        SSPB(ISS(1:6))= RI(11)*YY(1:6,3)+RI(12)*rsum(1:6)
     end if

     ! integral types (PS,SS) and (PP,SS).
     if(iw.gt.1) then
        rPASS(IPP(1:3))= RI(2)*YY(1:3,2)  ! ipp=1,3,6
        rPASS(ISS(1:6))= RI(3)*YY(1:6,3)+RI(4)*rsum(1:6)
     end if

     if(iw.gt.1 .and. jw.gt.1) then
        ! integral type (PS,PS) and auxiliary terms for (PS,PP).
        do i=1,6
           PSPS(i)= RI( 6)*YY(i,3)+RI( 7)*rsum(i)
           T(1,i) = RI(13)*YY(i,3)+RI(14)*rsum(i)
           T(2,i) = RI(15)*YY(i,8)
           T(3,i) = RI(15)*YY(i,5)
        end do
        ! intergal type (PS,PP).
        !yy_4(1:3)=YY(1:3,4); yy_7(1:3)=YY(1:3,7)
        !yy_5(1:6)=YY(1:6,5); yy_6(1:6)=YY(1:6,6); yy_8(1:6)=YY(1:6,8)
        !yy_9(1:6)=YY(1:6,9); yy_10(1:6)=YY(1:6,10)

        yy_4(1:3) =YY(1:3,4)
        yy_5(1:6) =YY(1:6,5)
        yy_6(1:6) =YY(1:6,6)
        yy_7(1:3) =YY(1:3,7)
        yy_8(1:6) =YY(1:6,8)
        yy_9(1:6) =YY(1:6,9)
        yy_10(1:6)=YY(1:6,10)
        do i=1,6
           !PSPP(i,1:3)=YY(1:3,2)*T(1,i) +YY(1:3,7)*T(2,i) +YY(1:3,4)*T(3,i)
           PSPP(i,1:3)=YY(1:3,2)*T(1,i) +yy_7(1:3)*T(2,i) +yy_4(1:3)*T(3,i)
        end do
        ! auxiliary terms for (PP,PS) and (PP,PP).
        do I=1,6
           !T(1,I) = RI( 8)*YY(I,3) +RI( 9)*rsum(I)
           !T(2,I) = RI(10)*YY(I,8)
           !T(3,I) = RI(10)*YY(I,5)
           !T(4,I) = RI(16)*YY(I,3) +RI(17)*rsum(I)
           !T(5,I) = RI(18)*YY(I,3) +RI(19)*YY(I,10) +RI(21)*YY(I,6)
           !T(6,I) = RI(18)*YY(I,3) +RI(19)*YY(I,6)  +RI(21)*YY(I,10)
           !T(7,I) = RI(20)*YY(I,8)
           !T(8,I) = RI(20)*YY(I,5)
           !T(9,I) = RI(22)*YY(I,9)

           T(1,I) = RI( 8)*YY(I,3) +RI( 9)*rsum(I)
           T(2,I) = RI(10)*yy_8(i)
           T(3,I) = RI(10)*yy_5(i)
           T(4,I) = RI(16)*YY(I,3) +RI(17)*rsum(I)
           T(5,I) = RI(18)*YY(I,3) +RI(19)*yy_10(i) +RI(21)*yy_6(i)
           T(6,I) = RI(18)*YY(I,3) +RI(19)*yy_6(i)  +RI(21)*yy_10(i)
           T(7,I) = RI(20)*yy_8(i)
           T(8,I) = RI(20)*yy_5(i)
           T(9,I) = RI(22)*yy_9(i)
        end do
        ! integral types (PP,PS) and (PP,PP).
        do I=1,6
           !PPPS(1:3,I)= YY(1:3,2)*T(1,I) +YY(1:3,7) *T(2,I) +YY(1:3,4)*T(3,I)
           !PPPP(1:6,I)= YY(1:6,3)*T(4,I) +YY(1:6,10)*T(5,I) +YY(1:6,6)*T(6,I)  &
           !            +YY(1:6,8)*T(7,I) +YY(1:6,5) *T(8,I) +YY(1:6,9)*T(9,I)
           PPPS(1:3,I)= YY(1:3,2)*T(1,I) +yy_7(1:3) *T(2,I) +yy_4(1:3)*T(3,I)
           PPPP(1:6,I)= YY(1:6,3)*T(4,I) +yy_10(1:6)*T(5,I) +yy_6(1:6)*T(6,I)  &
                       +yy_8(1:6)*T(7,I) +yy_5(1:6) *T(8,I) +yy_9(1:6)*T(9,I)
        end do
     end if
  end if

  ! store integrals.
  if(imode.eq.1) then   ! using linear array W.
     k    = kr+1
     w(k) = RI(1)
     ! integral types (SS,PS) and (SS,PP).
     if(jw.gt.1) w(k+1:k+9) = SSPB(1:9)
     ! integral types (PS,SS) and (PP,SS).
     if(iw.gt.1 .and. jw.eq.1) w(k+1:k+9) = rPASS(1:9)
     if(iw.gt.1 .and. jw.gt.1) then
        ! integral types (PS,SS) and (PP,SS).
        do i=1,9
           w(k+i*10) = rPASS(i)
        end do
        ! integral type (PS,PS).
        w(k+11) = PSPS(1)
        w(k+13) = PSPS(2)
        w(k+16) = PSPS(4)
        w(k+31) = PSPS(2)
        w(k+33) = PSPS(3)
        w(k+36) = PSPS(5)
        w(k+61) = PSPS(4)
        w(k+63) = PSPS(5)
        w(k+66) = PSPS(6)
        ! integral type (PS,PP).
        do i=1,3
           ij  = k+JPP(i)
           w(ij+ISS(1:6)) = PSPP(1:6,i)
        end do
        ! integral types (PP,PS) and (PP,PP).
        do i=1,6
           ij  = k+JSS(i)
           w(ij+IPP(1:3)) = PPPS(1:3,i)
           w(ij+ISS(1:6)) = PPPP(1:6,i)
        end do
     end if
  else  ! imode == 0, using square array WW.
     ww(ip,jp) = RI(1)
     ! integral type (SS,PS) and (SS,PP).
     if(jw.gt.1) ww(ip,jp+1:jp+9) = SSPB(1:9)
     ! integral type (PS,SS) and (PP,SS).
     if(iw.gt.1) ww(ip+1:ip+9,jp) = rPASS(1:9)
     if(iw.gt.1 .and. jw.gt.1) then
        ! integral type (PS,PS).
        ww(ip+1,jp+1) = PSPS(1)
        ww(ip+3,jp+1) = PSPS(2)
        ww(ip+6,jp+1) = PSPS(4)
        ww(ip+1,jp+3) = PSPS(2)
        ww(ip+3,jp+3) = PSPS(3)
        ww(ip+6,jp+3) = PSPS(5)
        ww(ip+1,jp+6) = PSPS(4)
        ww(ip+3,jp+6) = PSPS(5)
        ww(ip+6,jp+6) = PSPS(6)
        ! integral type (PS,PP).
        do i=1,3
           ij  = ip+IPP(i)
           ww(ij,jp+ISS(1:6)) = PSPP(1:6,i)
        end do
        ! integral types (PP,PS) and (PP,PP).
        do i=1,6
           ij  = ip+ISS(i)
           ww(ij,jp+IPP(1:3)) = PPPS(1:3,i)
           ww(ij,jp+ISS(1:6)) = PPPP(1:6,i)
        end do
     end if
  end if
  return
  end subroutine rotate


  subroutine rotbet(IA,JA,IORBS,JORBS,T,YY,H,LM4,indx)
  !
  ! THIS ROUTINE TRANSFORMS TWO-CENTER ONE-ELECTRON INTEGRALS FROM
  ! LOCAL TO MOLECULAR COORDINATES, AND INCLUDES THEM IN H(LM4).
  ! USEFUL FOR RESONANCE INTEGRALS AND FOR OVERLAP INTEGRALS.
  !
  ! NOTATION. I=INPUT, O=OUTPUT.
  ! IA        INDEX OF FIRST ORBITAL AT ATOM I (I).
  ! JA        INDEX OF FIRST ORBITAL AT ATOM J (I).
  ! IORBS     NUMBER OF ORBITALS AT ATOM I (I).
  ! JORBS     NUMBER OF ORBITALS AT ATOM J (I).
  ! T(14)     LOCAL TWO-CENTER ONE-ELECTRON INTEGRALS (I).
  ! YY()      PRECOMPUTED COMBINATION OF ROTATION MATRIX ELEMENTS (I).
  ! H(LM4)    ONE-ELECTRON MATRIX IN MOLECULAR COORDINATES (O).
  !
  !use chm_kinds
  !use qm1_info, only : qm_scf_main_r

  implicit none

  integer :: ia,ja,iorbs,jorbs,LM4,indx(*)
  real(chm_real):: H(LM4),T(14),YY(15,45)

  ! local variables
  integer :: i,j,m,ii,ij,is,ix,iy,iz
  real(chm_real):: HDD(15),T45

  ! section for an SP-basis.
  ! S(I)-S(J)
  is     = indx(ia)+ja
  H(is)  = T(1)
  if(iorbs.eq.1 .and. jorbs.eq.1) return

  ! S(I)-P(J)
  if(jorbs.ge.4) H(is+1:is+3) = T(2)*YY(1:3,2)

  ! P(I)-S(J)
  if(iorbs.ge.4) then
     ix      = indx(ia+1)+ja
     iy      = indx(ia+2)+ja
     iz      = indx(ia+3)+ja
     H(ix)   = T(3)*YY(1,2)
     H(iy)   = T(3)*YY(2,2)
     H(iz)   = T(3)*YY(3,2)
     ! P(I)-P(J).
     if(jorbs.ge.4) then
        T45     = T(4)-T(5)
        H(ix+1) = YY(1,3)*T45+T(5)
        H(ix+2) = YY(2,3)*T45
        H(ix+3) = YY(4,3)*T45
        !
        H(iy+1) = H(ix+2)
        H(iy+2) = YY(3,3)*T45+T(5)
        H(iy+3) = YY(5,3)*T45
        !
        H(iz+1) = H(ix+3)
        H(iz+2) = H(iy+3)
        H(iz+3) = YY(6,3)*T45+T(5)
     end if
  end if

  ! section involving D-orbitals.
  ! D(I)-S(J)
  if(iorbs.ge.9) then
     do i=1,5
        H(indx(ia+3+i)+ja) = T(6)*YY(i,11)
     end do
     ! D(I)-P(J)
     if(jorbs.ge.4) then
        ij     = 0
        do i=1,5
           M      = indx(ia+3+i)+ja
           do j=1,3
              ij     = ij+1
              H(M+j) = T(8)*YY(ij,12)+T(10)*(YY(ij,18)+YY(ij,25))
           end do
        end do
     end if
  end if
  ! S(I)-D(J)
  if(jorbs.ge.9) then
     M      = indx(ia)+ja+3
     H(M+1:M+5) = T(7)*YY(1:5,11)
     ! P(I)-D(J)
     if(iorbs.ge.4) then
        do i=1,3
           M      = indx(ia+i)+ja+3
           do j=1,5
              ij     = 3*(j-1)+i
              H(M+j) = T(9)*YY(ij,12)+T(11)*(YY(ij,18)+YY(ij,25))
           end do
        end do
        ! D(I)-D(J)
        if(iorbs.ge.9) then
           HDD(1:15) = T(12)* YY(1:15,15)              &
                      +T(13)*(YY(1:15,21)+YY(1:15,28)) &
                      +T(14)*(YY(1:15,36)+YY(1:15,45))
           do i=1,5
              M      = indx(ia+3+i)+ja+3
              ii     = indx(i)
              H(M+1:M+i)=HDD(ii+1:ii+i)
              do j=i+1,5
                 H(M+j) = HDD(indx(j)+i)
              end do
           end do
        end if
     end if
  end if

  return
  end subroutine rotbet


  subroutine rotcora_qmqm(IA,JA,IORBS,JORBS,IP,JP,CORE_mat,YY,H,LMH)
  !
  ! this routine transforms the core electron attraction integrals from local to
  ! mol. coords., and includes them in the core hamiltonian.
  !
  ! NOTATION. I=INPUT, O=OUTPUT, S=SCRATCH.
  ! ia        index of first basis orbital at atom i (I).
  ! ja        index of first basis orbital at atom j (I).
  ! iorbs     number of basis orbitals at atom i (I).
  ! jorbs     number of basis orbitals at atom j (I).
  ! ip        index for (S,S) of atom i in linear array H (I,S).
  ! jp        index for (S,S) of atom j in linear array H (I,S).
  ! core()    local core electron attraction integrals (I).
  ! YY()      precombined elements of rotation matrix (I).
  ! H(LMH)    core hamiltonian matrix (O).
  !
  ! depedning on the input data in the argument list, the integrals are included either in  
  ! the full core hamiltonian H(LM4) or in the one-center part H(LM6) of the core hamiltonian.
  !
  ! argument       first case              second case
  ! ia             nfirst(i)               1
  ! ja             nfirst(j)               1
  ! ip             indx(ia)+ia             NW(i)
  ! jp             indx(ja)+ja             NW(j)
  ! LMH            LM4                     LM6
  !
  !use chm_kinds

  implicit none

  integer :: IA,JA,IORBS,JORBS,IP,JP,LMH
  real(chm_real):: CORE_mat(10,2),YY(15,45),H(LMH)

  ! local variables
  integer :: i,j,k,kk,is,L,ix,iy,iz,idp,idd,id
  real(chm_real):: HPP(6),HDP(15),HDD(15),YY_1(15),YY_2(15),YY_3(15)

  do kk=1,2
     if(kk.eq.1) then
        is  = ip
        k   = ia-1
        L   = iorbs
     else
        is  = jp
        k   = ja-1
        L   = jorbs
     end if

     ! S-S
     H(is)  = H(is)+CORE_mat(1,kk)
     if(L.ge.4) then
        ! intermediate results for P-P
        HPP(1:6)=CORE_mat(3,kk)*YY(1:6,3)+CORE_mat(4,kk)*(YY(1:6,6)+YY(1:6,10))
        ! P-S
        ix     = is+1+k
        iy     = ix+2+k
        iz     = iy+3+k
        H(ix)  = H(ix)+CORE_mat(2,kk)*YY(1,2)
        H(iy)  = H(iy)+CORE_mat(2,kk)*YY(2,2)
        H(iz)  = H(iz)+CORE_mat(2,kk)*YY(3,2)
        ! P-P
        H(ix+1)     = H(ix+1)     +HPP(1)
        H(iy+1:iy+2)= H(iy+1:iy+2)+HPP(2:3)
        H(iz+1:iz+3)= H(iz+1:iz+3)+HPP(4:6)

        if(L.ge.9) then
           ! intermediate results for D-P and D-D
           !YY_1(1:15)=CORE_mat( 8,kk)*(YY(1:15,18)+YY(1:15,25))
           !YY_2(1:15)=CORE_mat( 9,kk)*(YY(1:15,21)+YY(1:15,28))
           !YY_3(1:15)=CORE_mat(10,kk)*(YY(1:15,36)+YY(1:15,45))
           !do i=1,15
           !   !HDP(i)=CORE_mat(6,kk)*YY(i,12)+CORE_mat( 8,kk)*(YY(i,18)+YY(i,25))
           !   !HDD(i)=CORE_mat(7,kk)*YY(i,15)+CORE_mat( 9,kk)*(YY(i,21)+YY(i,28)) &
           !   !                              +CORE_mat(10,kk)*(YY(i,36)+YY(i,45))
           !   HDP(i)=CORE_mat(6,kk)*YY(i,12)+YY_1(i)
           !   HDD(i)=CORE_mat(7,kk)*YY(i,15)+YY_2(i)+YY_3(i)
           !end do
           do i=1,15
              HDP(i)=CORE_mat(6,kk)*YY(i,12)+CORE_mat( 8,kk)*(YY(i,18)+YY(i,25))
              HDD(i)=CORE_mat(7,kk)*YY(i,15)+CORE_mat( 9,kk)*(YY(i,21)+YY(i,28)) &
                                            +CORE_mat(10,kk)*(YY(i,36)+YY(i,45))
           end do
           ! D-S
           idp    = 0
           idd    = 0
           id     = iz+3+k
           do i=5,9
              H(id+1)     = H(id+1)+CORE_mat(5,kk)*YY(i-4,11)
              ! D-P
              H(id+2:id+4)= H(id+2:id+4)+HDP(idp+1:idp+3)
              ! D-D
              H(id+5:id+i)= H(id+5:id+i)+HDD(idd+1:idd+(i-5)+1)

              idp= idp+3
              idd= idd+(i-5)+1
              id = id+i+k
           end do
        end if
     end if
  end do

  return
  end subroutine rotcora_qmqm


  subroutine rotcora_qmmm(IA,JA,IORBS,JORBS,IP,JP,CORE_mat,YY,H,LMH,q_ignore_diag)
  !
  ! this routine is a copy of rotcora. See subroutine rotcora.
  !
  ! for MM: (called with ja=0;jorbs=0;jp=0)
  ! for iorbs.le.0 or jorbs.le.0, there are no basis functions at atoms i or j,
  ! respectively, and hence no contributions to the core hamiltonian.
  !
  ! NOTATION. I=INPUT, O=OUTPUT, S=SCRATCH.
  ! ia        index of first basis orbital at atom i (I).
  ! ja        index of first basis orbital at atom j (I).
  ! iorbs     number of basis orbitals at atom i (I).
  ! jorbs     number of basis orbitals at atom j (I).
  ! ip        index for (S,S) of atom i in linear array H (I,S).
  ! jp        index for (S,S) of atom j in linear array H (I,S).
  ! core()    local core electron attraction integrals (I).
  ! YY()      precombined elements of rotation matrix (I).
  ! H(LMH)    core hamiltonian matrix (O).
  !
  ! q_ignore_diag ignore interaction with the diagonal elements. (i.e., set to zero).
  !
  !use chm_kinds

  implicit none

  integer :: IA,JA,IORBS,JORBS,IP,JP,LMH
  real(chm_real):: CORE_mat(10,2),YY(15,45),H(LMH)
  logical :: q_ignore_diag

  ! local variables
  integer :: i,j,k,kk,is,L,ix,iy,iz,idp,idd,id
  real(chm_real):: HPP(6),HDP(15),HDD(15),YY_1(15),YY_2(15),YY_3(15)

  ! no need for loop, only do kk=1
  kk = 1
  is = ip
  k  = ia-1

  if(q_ignore_diag) then
     ! S-S
     !H(is)  = H(is)+CORE_mat(1,1)
     if(iorbs >= 4) then
        ! intermediate results for P-P
        HPP(1:6)=CORE_mat(3,1)*YY(1:6,3)+CORE_mat(4,1)*(YY(1:6,6)+YY(1:6,10))
        ! P-S/P-P
        ix     = is+1+k
        iy     = ix+2+k
        iz     = iy+3+k
        H(ix)       = H(ix)       +CORE_mat(2,1)*YY(1,2)
        !H(ix+1)     = H(ix+1)     +HPP(1)
        H(iy)       = H(iy)       +CORE_mat(2,1)*YY(2,2)
        H(iy+1)     = H(iy+1)+HPP(2)
        !H(iy+2)     = H(iy+2)+HPP(3)
        H(iz)       = H(iz)       +CORE_mat(2,1)*YY(3,2)
        H(iz+1:iz+2)= H(iz+1:iz+2)+HPP(4:5)
        !H(iz+3)     = H(iz+3)+HPP(6)

        if(iorbs >= 9) then
           ! intermediate results for D-P and D-D
           do i=1,15
              HDP(i)=CORE_mat(6,1)*YY(i,12)+CORE_mat( 8,1)*(YY(i,18)+YY(i,25))
              HDD(i)=CORE_mat(7,1)*YY(i,15)+CORE_mat( 9,1)*(YY(i,21)+YY(i,28)) &
                                           +CORE_mat(10,1)*(YY(i,36)+YY(i,45))
           end do
           !
           idp    = 0
           idd    = 0
           id     = iz+3+k
           do i=5,9
              ! D-S
              H(id+1)     = H(id+1)+CORE_mat(5,1)*YY(i-4,11)
              ! D-P
              H(id+2:id+4)= H(id+2:id+4)+HDP(idp+1:idp+3)
              ! D-D
              !H(id+5:id+i  )= H(id+5:id+i  )+HDD(idd+1:idd+(i-5)+1)
              H(id+5:id+i-1)= H(id+5:id+i-1)+HDD(idd+1:idd+(i-4)+1)
              !H(id+i)       = H(id+i)+HDD(idd+(i-5)+1)

              idp= idp+3
              idd= idd+(i-5)+1
              id = id+i+k
           end do
        end if
     end if
  else
     ! S-S
     H(is)  = H(is)+CORE_mat(1,1)
     if(iorbs >= 4) then
        ! intermediate results for P-P
        HPP(1:6)=CORE_mat(3,1)*YY(1:6,3)+CORE_mat(4,1)*(YY(1:6,6)+YY(1:6,10))
        ! P-S/P-P
        ix     = is+1+k
        iy     = ix+2+k
        iz     = iy+3+k
        H(ix)       = H(ix)       +CORE_mat(2,1)*YY(1,2)
        H(ix+1)     = H(ix+1)     +HPP(1)
        H(iy)       = H(iy)       +CORE_mat(2,1)*YY(2,2)
        H(iy+1:iy+2)= H(iy+1:iy+2)+HPP(2:3)
        H(iz)       = H(iz)       +CORE_mat(2,1)*YY(3,2)
        H(iz+1:iz+3)= H(iz+1:iz+3)+HPP(4:6)

        if(iorbs >= 9) then
           ! intermediate results for D-P and D-D
           do i=1,15
              HDP(i)=CORE_mat(6,1)*YY(i,12)+CORE_mat( 8,1)*(YY(i,18)+YY(i,25))
              HDD(i)=CORE_mat(7,1)*YY(i,15)+CORE_mat( 9,1)*(YY(i,21)+YY(i,28)) &
                                           +CORE_mat(10,1)*(YY(i,36)+YY(i,45))
           end do
           !
           idp    = 0
           idd    = 0
           id     = iz+3+k
           do i=5,9
              ! D-S
              H(id+1)     = H(id+1)+CORE_mat(5,1)*YY(i-4,11)
              ! D-P
              H(id+2:id+4)= H(id+2:id+4)+HDP(idp+1:idp+3)
              ! D-D
              H(id+5:id+i)= H(id+5:id+i)+HDD(idd+1:idd+(i-5)+1)

              idp= idp+3
              idd= idd+(i-5)+1
              id = id+i+k
           end do
        end if
     end if
  end if
  return
  end subroutine rotcora_qmmm


  subroutine rotmat(J,I,JORBS,IORBS,Numatom,coord,R,YY)
  !
  ! rotation matrix for a given atom pair i-j (i.gt.j).
  !
  ! NOTATION. I=INPUT, O=OUTPUT.
  ! J,I       numbers of atoms in pair i-j (I).
  ! JORBS     number of basis functions at atom j (I).
  ! IORBS     number of basis functions at atom i (I).
  ! Numatom   number of atoms in array coord (I).
  ! COORD()   cartesian coordinates, in Angstrom (I).
  ! R         interatomic distance, in atomic units (O).
  ! YY()      precombined elements of the rotation matrix (O).
  !
  !use chm_kinds
  !use number
  !use qm1_constant

  implicit none

  integer :: j,i,jorbs,iorbs,numatom
  real(chm_real):: coord(3,numatom),YY(15,45),R

  ! local variables
  integer :: k,L,KL
  real(chm_real):: X(3),B,SQB,CA,CB,SA,SB,C2A,C2B,S2A,S2B
  real(chm_real):: p_tmp1(3),p_tmp2(3),d_tmp1(5),d_tmp2(5),  &
                   P_trans(3,3),D_trans(5,5)  ! ,P(3,3),D(5,5) not used since we use transpose of them
  ! i*(i-1)/2
  integer,parameter :: indx(9)=(/0,1,3,6,10,15,21,28,36/)
  !
  real(chm_real),parameter :: small=1.0D-07
  real(chm_real),parameter :: r_A0=one/A0


  ! calculate geometric data and interatomic distance.
  ! ca  = COS(phi)    , sa  = SIN(phi)
  ! cb  = COS(theta)  , sb  = SIN(theta)
  ! c2a = COS(2*phi)  , s2a = SIN(2*phi)
  ! c2b = COS(2*theta), s2b = SIN(2*phi) ; not 2*theta?

  ! when precombines the rotation matrix elements below.
  ! 
  ! the first index of YY(ij,kl) is consecutive. elements are dfined as
  ! 1 for KL=SS;  3 for KL=PS;  6 for KL=PP;  5 for KL=DS;  15 for KL=DP and
  ! KL=DD.
  ! 
  ! the second index of YY(ij,kl) is a standard pair index:
  ! KL=(K*(K-1))/2+L, order of K and L as in integral evaluation.

  x(1:3) = coord(1:3,j)-coord(1:3,i)

  b      = x(1)*x(1)+x(2)*x(2)
  r      = SQRT(b+x(3)*x(3))
  sqb    = SQRT(b)  ! =sqrt(x(1)**2+x(2)**2), if atoms are z-axix, it will be zero
  sb     = sqb/r    ! normalized sqb by r.
  ! check for special case (both atoms on z axis).
  if(sb.gt.small) then  ! it atoms are on z-axis.
     ca  = x(1)/sqb
     sa  = x(2)/sqb
     cb  = x(3)/r
  else
     SA  = zero
     SB  = zero
     if(x(3).lt.zero) then
        CA  =-one
        CB  =-one
     else if(x(3).gt.zero) then
        CA  = one
        CB  = one
     else
        CA  = zero
        CB  = zero
     end if
  end if
  R      = r*r_A0  ! /A0; convert distance to atomic unit.

  ! Rotation matrix elements
  !P(1,1) = CA*SB
  !P(2,1) = CA*CB
  !P(3,1) =-SA
  !P(1,2) = SA*SB
  !P(2,2) = SA*CB
  !P(3,2) = CA
  !P(1,3) = CB
  !P(2,3) =-SB
  !P(3,3) = ZERO

  ! below, we only use the transpose of P
  P_trans(1,1)= CA*SB
  P_trans(2,1)= SA*SB
  P_trans(3,1)= CB

  P_trans(1,2)= CA*CB
  P_trans(2,2)= SA*CB
  P_trans(3,2)=-SB

  P_trans(1,3)=-SA
  P_trans(2,3)= CA
  P_trans(3,3)= zero

  ! precombine rotation matrix elements.
  ! S-S
  YY(1,1)   = one
  if(iorbs >= 4 .or. jorbs >= 4) then
     ! P-S
     do k=1,3
        KL        = indx(K+1)+1
        YY(1:3,KL)= P_trans(1:3,k)  ! =P(K,1:3)
     end do
     ! P-P
     do k=1,3
        KL         = indx(k+1)+k+1
        p_tmp1(1:3)= P_trans(1:3,k)        ! =P(K,1:3)
        YY(1,KL)   = p_tmp1(1)*p_tmp1(1)   ! =P(K,1)*P(K,1)
        YY(2:3,KL) = p_tmp1(1:2)*p_tmp1(2) ! =P(K,1:2)*P(K,2)
        YY(4:6,KL) = p_tmp1(1:3)*p_tmp1(3) ! =P(K,1:3)*P(K,3)
     end do
     do k=2,3
        p_tmp1(1:3)=P_trans(1:3,k)      ! =P(K,1:3)
        do L=1,k-1
           KL         =indx(k+1)+L+1
           p_tmp2(1:3)=P_trans(1:3,L)      ! =P(L,1:3)
           YY(1,KL)   =p_tmp1(1)*p_tmp2(1)*two
           YY(2:3,KL) =p_tmp1(1:2)*p_tmp2(2)+p_tmp1(2)*p_tmp2(1:2)
           YY(4:6,KL) =p_tmp1(1:3)*p_tmp2(3)+p_tmp1(3)*p_tmp2(1:3)
        end do
     end do
     if(iorbs.ge.9 .or. jorbs.ge.9) then
        C2A    = two*CA*CA-one  
        C2B    = two*CB*CB-one  
        S2A    = two*SA*CA
        S2B    = two*SB*CB
        !
        !D(1,1) = PT5SQ3*C2A*SB*SB
        !D(2,1) = PT5*C2A*S2B
        !D(3,1) =-S2A*SB
        !D(4,1) = C2A*(CB*CB+PT5*SB*SB)
        !D(5,1) =-S2A*CB
        !
        !D(1,2) = PT5SQ3*CA*S2B
        !D(2,2) = CA*C2B
        !D(3,2) =-SA*CB
        !D(4,2) =-PT5*CA*S2B
        !D(5,2) = SA*SB
        !
        !D(1,3) = CB*CB-PT5*SB*SB
        !D(2,3) =-PT5SQ3*S2B
        !D(3,3) = ZERO
        !D(4,3) = PT5SQ3*SB*SB
        !D(5,3) = ZERO
        !
        !D(1,4) = PT5SQ3*SA*S2B
        !D(2,4) = SA*C2B
        !D(3,4) = CA*CB
        !D(4,4) =-PT5*SA*S2B
        !D(5,4) =-CA*SB
        !
        !D(1,5) = PT5SQ3*S2A*SB*SB
        !D(2,5) = PT5*S2A*S2B
        !D(3,5) = C2A*SB
        !D(4,5) = S2A*(CB*CB+PT5*SB*SB)
        !D(5,5) = C2A*CB

        ! below, we only use the transpose of D.
        D_trans(1,1)= pt5sq3*C2A*SB*SB
        D_trans(2,1)= pt5sq3*CA*S2B
        D_trans(3,1)= CB*CB-pt5*SB*SB
        D_trans(4,1)= pt5sq3*SA*S2B
        D_trans(5,1)= pt5sq3*S2A*SB*SB

        D_trans(1,2)= pt5*C2A*S2B
        D_trans(2,2)= CA*C2B
        D_trans(3,2)=-pt5SQ3*S2B
        D_trans(4,2)= SA*C2B
        D_trans(5,2)= pt5*S2A*S2B

        D_trans(1,3)=-S2A*SB
        D_trans(2,3)=-SA*CB
        D_trans(3,3)= zero
        D_trans(4,3)= CA*CB
        D_trans(5,3)= C2A*SB

        D_trans(1,4)= C2A*(CB*CB+pt5*SB*SB)
        D_trans(2,4)=-pt5*CA*S2B
        D_trans(3,4)= pt5sq3*SB*SB
        D_trans(4,4)=-pt5*SA*S2B
        D_trans(5,4)= S2A*(CB*CB+pt5*SB*SB)

        D_trans(1,5)=-S2A*CB
        D_trans(2,5)= SA*SB
        D_trans(3,5)= zero
        D_trans(4,5)=-CA*SB
        D_trans(5,5)= C2A*CB

        ! precombine rotation matrix elements.
        ! D-S
        do k=1,5
           KL        = indx(k+4)+1
           YY(1:5,KL)= D_trans(1:5,k)
        end do
        ! D-P
        do k=1,5
           d_tmp1(1:5) = D_trans(1:5,k)  ! = D(K,1:5)
           do L=1,3
              KL          = indx(k+4)+L+1
              p_tmp2(1:3) = P_trans(1:3,L) ! = P(L,1:3)
              YY(1:3,KL)  = d_tmp1(1)*p_tmp2(1:3)
              YY(4:6,KL)  = d_tmp1(2)*p_tmp2(1:3)
              YY(7:9,KL)  = d_tmp1(3)*p_tmp2(1:3)
              YY(10:12,KL)= d_tmp1(4)*p_tmp2(1:3)
              YY(13:15,KL)= d_tmp1(5)*p_tmp2(1:3)
           end do
        end do
        ! D-D
        do k=1,5
           KL        = indx(k+4)+k+4
           d_tmp1(1:5) = D_trans(1:5,k)  ! = D(K,1:5)
           YY(1,KL)    = d_tmp1(1)  *d_tmp1(1)
           YY(2:3,KL)  = d_tmp1(1:2)*d_tmp1(2)
           YY(4:6,KL)  = d_tmp1(1:3)*d_tmp1(3)
           YY(7:10,KL) = d_tmp1(1:4)*d_tmp1(4)
           YY(11:15,KL)= d_tmp1(1:5)*d_tmp1(5)
        end do
        do k=2,5
           d_tmp1(1:5) = D_trans(1:5,k)    ! = D(K,1:5)
           do L=1,k-1
              KL          = indx(k+4)+L+4
              d_tmp2(1:5) = D_trans(1:5,L)  ! = D(L,1:5)
              YY(1,KL)    = d_tmp1(1)  *d_tmp2(1)*two
              YY(2:3,KL)  = d_tmp1(1:2)*d_tmp2(2)+d_tmp1(2)*d_tmp2(1:2)
              YY(4:6,KL)  = d_tmp1(1:3)*d_tmp2(3)+d_tmp1(3)*d_tmp2(1:3)
              YY(7:10,KL) = d_tmp1(1:4)*d_tmp2(4)+d_tmp1(4)*d_tmp2(1:4)
              YY(11:15,KL)= d_tmp1(1:5)*d_tmp2(5)+d_tmp1(5)*d_tmp2(1:5)
           end do
        end do
     end if
  end if

  return
  end subroutine rotmat

  subroutine rotmat_qmmm(J,I,JORBS,IORBS,Numatom,coord,R,YY)
  !
  ! rotation matrix for a given atom pair i-j (i.gt.j); j: mm and i: qm.
  !
  ! NOTATION. I=INPUT, O=OUTPUT.
  ! J,I       numbers of atoms in pair i-j (I).
  ! JORBS     number of basis functions at atom j (I).
  ! IORBS     number of basis functions at atom i (I).
  ! Numatom   number of atoms in array coord (I).
  ! COORD()   cartesian coordinates, in Angstrom (I).
  ! R         interatomic distance, in atomic units (O).
  ! YY()      precombined elements of the rotation matrix (O).
  !
  !use qm1_info, only : mm_main_r
  implicit none

  integer :: j,i,jorbs,iorbs,numatom
  real(chm_real):: coord(3,numatom),YY(15,45),R

  ! local variables
  integer :: k,L,KL
  real(chm_real):: X(3),B,SQB,CA,CB,SA,SB,C2A,C2B,S2A,S2B,SB2,CB2
  real(chm_real):: p_tmp1(3),p_tmp2(3),d_tmp1(5),d_tmp2(5),  &
                   P_trans(3,3),D_trans(5,5)  ! ,P(3,3),D(5,5) not used since we use transpose of them
  ! i*(i-1)/2
  integer,parameter :: indx(9)=(/0,1,3,6,10,15,21,28,36/)
  !
  real(chm_real),parameter :: small=1.0D-07
  real(chm_real),parameter :: r_A0=one/A0


  ! calculate geometric data and interatomic distance.
  ! ca  = COS(phi)    , sa  = SIN(phi)
  ! cb  = COS(theta)  , sb  = SIN(theta)
  ! c2a = COS(2*phi)  , s2a = SIN(2*phi)
  ! c2b = COS(2*theta), s2b = SIN(2*phi) ; not 2*theta?

  ! when precombines the rotation matrix elements below.
  !
  ! the first index of YY(ij,kl) is consecutive. elements are dfined as
  ! 1 for KL=SS;  3 for KL=PS;  6 for KL=PP;  5 for KL=DS;  15 for KL=DP and
  ! KL=DD.
  !
  ! the second index of YY(ij,kl) is a standard pair index:
  ! KL=(K*(K-1))/2+L, order of K and L as in integral evaluation.

  x(1:3) = coord(1:3,j)-coord(1:3,i)

  b      = x(1)*x(1)+x(2)*x(2)
  r      = SQRT(b+x(3)*x(3))
  sqb    = SQRT(b)  ! =sqrt(x(1)**2+x(2)**2), if atoms are z-axix, it will be zero
  sb     = sqb/r    ! normalized sqb by r.
  ! check for special case (both atoms on z axis).
  if(sb.gt.small) then  ! it atoms are on z-axis.
     ca  = x(1)/sqb
     sa  = x(2)/sqb
     cb  = x(3)/r
  else
     SA  = zero
     SB  = zero
     if(x(3).lt.zero) then
        CA  =-one
        CB  =-one
     else if(x(3).gt.zero) then
        CA  = one
        CB  = one
     else
        CA  = zero
        CB  = zero
     end if
  end if
  R      = r*r_A0  ! /A0; convert distance to atomic unit.

  ! precombine rotation matrix elements: jorbs == 1
  ! for S-S pair..
  YY(1,1)   = one
  if(iorbs >= 4) then
     ! Rotation matrix elements
     ! below, we only use the transpose of P
     P_trans(1,1)= CA*SB
     P_trans(2,1)= SA*SB
     P_trans(3,1)= CB

     P_trans(1,2)= CA*CB
     P_trans(2,2)= SA*CB
     P_trans(3,2)=-SB

     P_trans(1,3)=-SA
     P_trans(2,3)= CA
     P_trans(3,3)= zero

     ! S-S pair
     !YY(1,1)   = one

     ! P-S pair
     YY(1:3,2)= P_trans(1:3,1)  ! kl=indx(k+1)+1, k=1

     ! P-P pair
     do k=1,3
        KL         = indx(k+1)+k+1
        YY(1,KL)   = P_trans(1,k)  *P_trans(1,k)
        YY(2:3,KL) = P_trans(1:2,k)*P_trans(2,k)
        YY(4:6,KL) = P_trans(1:3,k)*P_trans(3,k)
     end do

     if(iorbs >= 9) then
        C2A    = two*CA*CA-one
        C2B    = two*CB*CB-one
        S2A    = two*SA*CA
        S2B    = two*SB*CB

        CB2    = CB*CB
        SB2    = SB*SB

        ! below, we only use the transpose of D.
        D_trans(1,1)= pt5sq3*C2A*SB2
        D_trans(2,1)= pt5sq3*CA*S2B
        D_trans(3,1)= CB2-pt5*SB2
        D_trans(4,1)= pt5sq3*SA*S2B
        D_trans(5,1)= pt5sq3*S2A*SB2

        D_trans(1,2)= pt5*C2A*S2B
        D_trans(2,2)= CA*C2B
        D_trans(3,2)=-pt5SQ3*S2B
        D_trans(4,2)= SA*C2B
        D_trans(5,2)= pt5*S2A*S2B

        D_trans(1,3)=-S2A*SB
        D_trans(2,3)=-SA*CB
        D_trans(3,3)= zero
        D_trans(4,3)= CA*CB
        D_trans(5,3)= C2A*SB

        D_trans(1,4)= C2A*(CB2+pt5*SB2)
        D_trans(2,4)=-pt5*CA*S2B
        D_trans(3,4)= pt5sq3*SB2
        D_trans(4,4)=-pt5*SA*S2B
        D_trans(5,4)= S2A*(CB2+pt5*SB2)

        D_trans(1,5)=-S2A*CB
        D_trans(2,5)= SA*SB
        D_trans(3,5)= zero
        D_trans(4,5)=-CA*SB
        D_trans(5,5)= C2A*CB

        ! D-S
        YY(1:5,11)= D_trans(1:5,1)  ! k=1; kl=11 (=indx(k+4)+1
        ! D-P
        do k=1,3
           KL          = indx(k+4)+k+1
           YY(1:3,KL)  = D_trans(1,k)*P_trans(1:3,k)
           YY(4:6,KL)  = D_trans(2,k)*P_trans(1:3,k)
           YY(7:9,KL)  = D_trans(3,k)*P_trans(1:3,k)
           YY(10:12,KL)= D_trans(4,k)*P_trans(1:3,k)
           YY(13:15,KL)= D_trans(5,k)*P_trans(1:3,k)
        end do
        ! D-D
        do k=1,5
           KL        = indx(k+4)+k+4
           d_tmp1(1:5) = D_trans(1:5,k)  ! = D(K,1:5)
           YY(1,KL)    = d_tmp1(1)  *D_trans(1,k)
           YY(2:3,KL)  = d_tmp1(1:2)*D_trans(2,k)
           YY(4:6,KL)  = d_tmp1(1:3)*D_trans(3,k)
           YY(7:10,KL) = d_tmp1(1:4)*D_trans(4,k)
           YY(11:15,KL)= d_tmp1(1:5)*D_trans(5,k)
        end do
     end if
  end if

  return
  end subroutine rotmat_qmmm

!  real(chm_real) function spcw(C1,C2,C3,C4,CKL,W,LM2,LM6)
!  !
!  ! SPCW calculates the repulsion between electron 1 in molecular orbitals C1,C2
!  ! and electron 2 in C3,C4 for the valuence shell. (Special MNDO version with 
!  ! integrals W(ij,kl) in square array.)
!  !
!  !use chm_kinds
!  !use number, only : zero
!  use qm1_info, only : qm_scf_main_r
!
!  implicit none
!
!  integer :: LM2,LM6
!  real(chm_real):: C1(LM2),C2(LM2),C3(LM2),C4(LM2),CKL(LM6),W(LM6,LM6)
!
!  ! local variables
!  integer :: KL,K,L,IJ,I,J
!  real(chm_real):: WIJ,CIJ
!
!  SPCW   = zero
!  do kl=1,LM6
!     k      = qm_scf_main_r%IP1(kl)
!     l      = qm_scf_main_r%IP2(kl)
!     ckl(kl)= C3(k)*C4(l)
!     if(k.ne.l) ckl(kl)=ckl(kl)+C3(l)*C4(k)
!  end do
!  do ij=1,LM6
!     wij=DOT_PRODUCT(ckl(1:LM6),w(1:LM6,ij))
!     I      = qm_scf_main_r%IP1(ij)
!     J      = qm_scf_main_r%IP2(ij)
!     cij    = C1(i)*C2(j)
!     if(i.ne.j) cij=cij+C1(j)*C2(i)
!     SPCW   = SPCW+cij*wij
!  end do
!  return
!  end function spcw


  subroutine wstore(W,w_linear,linear_fock,MODE,numat,uhf)
  !
  ! complete defintion of square matrix of MNDO two-electron integrals by
  ! including the one-center terms and the terms with transposed indices.
  !
  ! MODE= 0   include rhf one-center integrals and transpose.
  ! MODE= 1   include raw one-center integrals and transpose (UHF).
  !
  !use chm_kinds
  !use number
  !use qm1_constant 
  !use qm1_info, only : qm_main_r,qm_scf_main_r
  !use qm1_parameters, only : GSS,GPP,GSP,GP2,HSP,HPP,REPD, &
  !                           INTIJ,INTKL,INTREP,INTRF1,INTRF2
  !
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none
 
  integer       :: linear_fock,MODE,numat
  real(chm_real):: W(linear_fock,linear_fock),w_linear(*)
  logical       :: uhf

  ! local varibale
  integer :: MODW,I,J,II,IJ,ij0,IW,IP,IPM,IPX,IPY,IPZ,NI,IORBS,KL,INTi,INT1,INT2
  real(chm_real):: GSPNI,GPPNI,GP2NI,HSPNI,HPPNI,RF

  integer       :: mmynod,nnumnod,iicnt,istart

  ! for parallelization: should do the same work for each processor.
#if KEY_PARALLEL==1
  istart = mynod+1
  mmynod = mynod
  nnumnod= numnod
#else
  istart = 1
  nnumnod= 1
#endif

! This is unnecessary.
!  ! initialize one-center integrals (upper triangle) to zero.
!  !modw   = mode
!  !if(mode.eq.0 .and. uhf) modw=1
!  do ii=istart,numat,nnumnod
!     iorbs  = iorbs_local(ii) ! num_orbs(ii)  ! NLAST(II)-NFIRST(II)+1
!     if(iorbs.ge.4) then
!        iw  = iw_local(ii)    !indx(iorbs)+iorbs ! IORBS*(IORBS+1)/2
!        ip  = ip_local(ii)    !nw(ii) 
!        ipm = ip+iw-1
!        do j=ip,ipm-1
!           w(j+1:ipm,j) = zero
!        end do
!     end if
!  end do

  ! include non-zero one-center terms.
  if(uhf) then
     do ii=istart,numat,nnumnod
        ip  = ip_local(ii) ! nw(ii)
        ni  = ni_local(ii) ! nat(ii)
        w(ip,ip) = GSS_local(ii) ! GSS(ni)
        iorbs    = iorbs_local(ii) ! num_orbs(ii)  ! NLAST(II)-NFIRST(II)+1
        if(iorbs.ge.4) then
           ipx = ip+2
           ipy = ip+5
           ipz = ip+9
           GSPNI = GSP_local(ii) 
           GPPNI = GPP_local(ii)
           GP2NI = GP2_local(ii)
           HSPNI = HSP_local(ii)
           HPPNI = HPP_local(ii)
           w(ipx ,ip  ) = GSPNI ! GSP(ni)
           w(ipy ,ip  ) = GSPNI ! GSP(ni)
           w(ipz ,ip  ) = GSPNI ! GSP(ni)
           !w(ip  ,ipx ) = GSPNI ! GSP(ni)
           !w(ip  ,ipy ) = GSPNI ! GSP(ni)
           !w(ip  ,ipz ) = GSPNI ! GSP(ni)
           w(ipx ,ipx ) = GPPNI ! GPP(ni)
           w(ipy ,ipy ) = GPPNI ! GPP(ni)
           w(ipz ,ipz ) = GPPNI ! GPP(ni)
           w(ipy ,ipx ) = GP2NI ! GP2(ni)
           w(ipz ,ipx ) = GP2NI ! GP2(ni)
           w(ipz ,ipy ) = GP2NI ! GP2(ni)
           !w(ipx ,ipy ) = GP2NI ! GP2(ni)
           !w(ipx ,ipz ) = GP2NI ! GP2(ni)
           !w(ipy ,ipz ) = GP2NI ! GP2(ni)
           w(ip+1,ip+1) = HSPNI ! HSP(ni)
           w(ip+3,ip+3) = HSPNI ! HSP(ni)
           w(ip+6,ip+6) = HSPNI ! HSP(ni)
           w(ip+4,ip+4) = HPPNI ! HPP(ni)
           w(ip+7,ip+7) = HPPNI ! HPP(ni)
           w(ip+8,ip+8) = HPPNI ! HPP(ni)
           if(iorbs.ge.9) then
              ij0    = ip-1
              do i=1,243
                 !w(intij(i)+ij0,intkl(i)+ij0) = REPD(INTREP(i),ni)
                 w(int_ij(i)+ij0,int_kl(i)+ij0) = w_save(i,ii)
              end do
           end if
        end if
     end do
  else
     do ii=istart,numat,nnumnod
        ip  = ip_local(ii) ! NW(ii)
        ! ni  = ni_local(ii) ! nat(ii)
        W(ip,ip) = GSS_local(ii) ! GSS_local(ii)*PT5  ! GSS(ni)*PT5
        iorbs    = iorbs_local(ii) ! num_orbs(ii)  ! NLAST(II)-NFIRST(II)+1
        if(iorbs.ge.4) then
           ipx = ip+2
           ipy = ip+5
           ipz = ip+9
           !GSPNI = GSP(ni)-HSP(ni)*PT5
           !GPPNI = GPP(ni)*PT5
           !GP2NI = GP2(ni)-HPP(ni)*PT5
           !HSPNI = HSP(ni)*PT75-GSP(ni)*PT25
           !HPPNI = HPP(ni)*PT75-GP2(ni)*PT25

           ! already computed the following multiplications (see QMMM_module_prep)
           GSPNI = GSP_local(ii) ! GSP_local(ii)-HSP_local(ii)*PT5
           GPPNI = GPP_local(ii) ! GPP_local(ii)*PT5
           GP2NI = GP2_local(ii) ! GP2_local(ii)-HPP_local(ii)*PT5
           HSPNI = HSP_local(ii) ! HSP_local(ii)*PT75-GSP_local(ii)*PT25
           HPPNI = HPP_local(ii) ! HPP_local(ii)*PT75-GP2_local(ii)*PT25
           w(ipx ,ip  ) = GSPNI
           w(ipy ,ip  ) = GSPNI
           w(ipz ,ip  ) = GSPNI
           !w(ip  ,ipx ) = GSPNI
           !w(ip  ,ipy ) = GSPNI
           !w(ip  ,ipz ) = GSPNI
           w(ipx ,ipx ) = GPPNI
           w(ipy ,ipy ) = GPPNI
           w(ipz ,ipz ) = GPPNI
           w(ipy ,ipx ) = GP2NI
           w(ipz ,ipx ) = GP2NI
           w(ipz ,ipy ) = GP2NI
           !w(ipx ,ipy ) = GP2NI
           !w(ipx ,ipz ) = GP2NI
           !w(ipy ,ipz ) = GP2NI
           w(ip+1,ip+1) = HSPNI
           w(ip+3,ip+3) = HSPNI
           w(ip+4,ip+4) = HPPNI  ! note this is right.
           w(ip+6,ip+6) = HSPNI
           w(ip+7,ip+7) = HPPNI
           w(ip+8,ip+8) = HPPNI
           if(iorbs.ge.9) then
              ij0    = ip-1
              do i=1,243
                 !int1   = INTRF1(i)
                 !int2   = INTRF2(i)
                 !rf     = REPD(INTREP(i),ni)
                 !if(int1.gt.0) rf = rf-PT25*REPD(int1,ni)
                 !if(int2.gt.0) rf = rf-PT25*REPD(int2,ni)
                 !w(intij(i)+ij0,intkl(i)+ij0) = rf

                 !int1 = int_ij(i)+ij0
                 !int2 = int_kl(i)+ij0
                 w(int_ij(i)+ij0,int_kl(i)+ij0) = w_save(i,ii)
              end do
           end if
        end if
     end do
  end if
  !
#if KEY_PARALLEL==1
  if(nnumnod>1) then
     ! w contains only lower-triangular info.
     ij = 0
     do i=1,linear_fock
        !do j=i,linear_fock
        !   ij = ij + 1
        !   w_linear(ij) = w(j,i)
        !end do
        w_linear(ij+1:ij+linear_fock-i+1) = w(i:linear_fock,i)
        ij = ij + linear_fock-i+1
     end do
!     call gcomb(w_linear,linear_fock*(linear_fock+1)/2)
!     ij = 0
!     do i=1,linear_fock
!        !do j=i,linear_fock
!        !   ij = ij + 1
!        !   w(j,i) = w_linear(ij)
!        !end do
!        w(i:linear_fock,i)=w_linear(ij+1:ij+linear_fock-i+1)
!        ij = ij + linear_fock-i+1
!     end do
!     do i=2,linear_fock
!        do j=1,i-1
!           w(j,i) = w(i,j)
!        end do
!     end do
!  else
#endif
!     ! Include terms with transposed indices. 
!     !   The following call is equivalent to the following do loops.
!     !call square_transpose(W,linear_fock)
!     !
!     loopii: do i=2,linear_fock
!        loopjj: do j=1,i-1
!           w(j,i)=w(i,j)
!        end do loopjj
!     end do loopii
#if KEY_PARALLEL==1
  end if
#endif
  !
  return
  end subroutine wstore

  subroutine wstore_comm(W,w_linear,linear_fock)
  !
  ! complete defintion of square matrix of MNDO two-electron integrals by
  ! including the one-center terms and the terms with transposed indices.
  !
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none
 
  integer       :: linear_fock
  real(chm_real):: W(linear_fock,linear_fock),w_linear(*)

  ! local varibale
  integer :: I,J,II,IJ

  integer       :: nnumnod

  ! for parallelization: should do the same work for each processor.
#if KEY_PARALLEL==1
  nnumnod= numnod
#else
  nnumnod= 1
#endif
  !
#if KEY_PARALLEL==1
  if(nnumnod>1) then
     ! w contains only lower-triangular info.
     ij = 0
     do i=1,linear_fock
        w(i:linear_fock,i)=w_linear(ij+1:ij+linear_fock-i+1)
        ij = ij + linear_fock-i+1
     end do
     do i=2,linear_fock
        do j=1,i-1
           w(j,i) = w(i,j)
        end do
     end do
  else
#endif
     ! Include terms with transposed indices. 
     !   The following call is equivalent to the following do loops.
     !call square_transpose(W,linear_fock)
     !
     loopii: do i=2,linear_fock
        loopjj: do j=1,i-1
           w(j,i)=w(i,j)
        end do loopjj
     end do loopii
#if KEY_PARALLEL==1
  end if
#endif
  !
  return
  end subroutine wstore_comm
  
  !
#endif /*mndo97*/
  !
end module qm1_energy_module
