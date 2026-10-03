module qm1_energy_module
  use chm_kinds
  use number
  use qm1_constant

  integer,save :: iqm_mode                    ! this variable is set at routine scf_energy
  logical,save :: do_am1_pm3                  !
  logical,save :: do_d_orbitals               !

!!  ! other variables
!!  integer, save :: nunique_qm

  contains

#if KEY_MNDO97==1 /*mndo97*/
  subroutine QMMM_module_prep
  !
  ! Prepare locally used variables.
  !
  use qm1_info, only : qm_main_c,qm_scf_main_c,qm_param_c,qm_control_c
  implicit none

  integer :: i,ia,iorbs

  ! now start seting up.
  iqm_mode     = qm_control_c%iqm_mode        ! local variables
  do_am1_pm3   = qm_control_c%q_am1_pm3       !
  do_d_orbitals= qm_control_c%do_d_orbitals   !

  do i=1,qm_main_c%numat
     ia             = qm_main_c%NFIRST(i)
     iorbs          = qm_main_c%num_orbs(i)
     qm_param_c%ni_local(i)    = qm_main_c%nat(i)               ! atomic number
     qm_param_c%iorbs_local(i) = qm_main_c%num_orbs(i)          ! norbs or num_orbs?
     qm_param_c%ia_local(i)    = qm_main_c%NFIRST(i)            ! nfirst
     qm_param_c%ib_local(i)    = qm_main_c%NLAST(i)             ! nlast
     qm_param_c%ip_local(i)    = qm_scf_main_c%NW(i)            ! nw
     qm_param_c%is_local(i)    = qm_scf_main_c%indx(ia)+ia
     qm_param_c%iw_local(i)    = qm_scf_main_c%indx(iorbs)+iorbs
     if(iorbs == 9) then
        qm_param_c%Jmax_local(i)=10
     else
        qm_param_c%Jmax_local(i)=4
     end if
  end do

  return
  end subroutine QMMM_module_prep

  subroutine betaij(NI,NJ,iorbs,jorbs,R,T,               &
                    zsi,zpi,zdi,betas_i,betap_i,betad_i, &
                    zsj,zpj,zdj,betas_j,betap_j,betad_j)
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

  integer :: NI,NJ,iorbs,jorbs
  real(chm_real):: R,T(14)
  real(chm_real):: zsi,zpi,zdi,betas_i,betap_i,betad_i,  &
                   zsj,zpj,zdj,betas_j,betap_j,betad_j

  ! local variables:
  real(chm_real):: TT

  ! compute overlap integrals.
  call overlp(ni,nj,iorbs,jorbs,R,T,zsi,zpi,zdi,zsj,zpj,zdj)

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
        T(1)  = PT5*(betas_i + betas_j)*T(1)
     case ( 5)                ! case 2
        T(1)  = PT5*(betas_i + betas_j)*T(1)
        T(2)  = PT5*(betas_i + betap_j)*T(2)
        T(3)  = PT5*(betap_i + betas_j)*T(3)
     case ( 8)                ! case 3
        T(1)  = PT5*(betas_i + betas_j)*T(1)
        T(2)  = PT5*(betas_i + betap_j)*T(2)
        T(3)  = PT5*(betap_i + betas_j)*T(3)
        TT    = PT5*(betap_i + betap_j)
        T(4:5) = TT*T(4:5)
     case (10)                ! case 4
        T(1)  = PT5*(betas_i + betas_j)*T(1)
        T(2)  = PT5*(betas_i + betap_j)*T(2)
        T(3)  = PT5*(betap_i + betas_j)*T(3)

        T(6)  = PT5*(betad_i + betas_j)*T(6)
        T(7)  = PT5*(betas_i + betad_j)*T(7)
     case (13)                ! case 5
        T(1)  = PT5*(betas_i + betas_j)*T(1)
        T(2)  = PT5*(betas_i + betap_j)*T(2)
        T(3)  = PT5*(betap_i + betas_j)*T(3)
        TT    = PT5*(betap_i + betap_j)
        T(4:5) = TT*T(4:5)
        T(6)  = PT5*(betad_i + betas_j)*T(6)
        T(7)  = PT5*(betas_i + betad_j)*T(7)
        T(8)  = PT5*(betad_i + betap_j)*T(8)
        T(9)  = PT5*(betap_i + betad_j)*T(9)
        T(10) = PT5*(betad_i + betap_j)*T(10)
        T(11) = PT5*(betap_i + betad_j)*T(11)
     case (18)                ! case 6
        T(1)  = PT5*(betas_i + betas_j)*T(1)
        T(2)  = PT5*(betas_i + betap_j)*T(2)
        T(3)  = PT5*(betap_i + betas_j)*T(3)
        TT    = PT5*(betap_i + betap_j)
        T(4:5) = TT*T(4:5)
        T(6)  = PT5*(betad_i + betas_j)*T(6)
        T(7)  = PT5*(betas_i + betad_j)*T(7)
        T(8)  = PT5*(betad_i + betap_j)*T(8)
        T(9)  = PT5*(betap_i + betad_j)*T(9)
        T(10) = PT5*(betad_i + betap_j)*T(10)
        T(11) = PT5*(betap_i + betad_j)*T(11)
        TT    = PT5*(betad_i + betad_j)
        T(12:14) = TT*T(12:14)
  end select
  return
  end subroutine betaij


  subroutine core_repul(I,J,NI,NJ,R,WIJ,ENUCLR,q_specific_pair)
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
  !use qm1_parameters, only : ALP,DD,PO,CORE, &
  !                           ALPB, MALPB      ! mndo/d only
  use qm1_info, only : qm_param_c

  implicit none
  !
  integer       :: I,J,NI,NJ
  real(chm_real):: R,WIJ,ENUCLR
  logical       :: q_specific_pair

  ! local variables
  integer :: k,MALPNI,MALPNJ
  real(chm_real):: ALPNI,ALPNJ,RIJ,ENI,ENJ,scale_val,ENUC,ENUC1 

  ! set:
  ALPNI  = qm_param_c%ALP(i)  ! ALP(ni)
  ALPNJ  = qm_param_c%ALP(j)  ! ALP(nj)
  RIJ    = R*A0   ! CONVERT TO ANGSTROM.

  ! bond parameters in mndo/d (if defined).
  ! ALPB(nj,ni) is used as alpha parameters for the element with the
  !             atomic number NI in the case of the pair NJ-NI,
  !            -if there are bond parameters for NI (MALPB(ni).gt.0),
  !            -if there may be a bond parameter for NJ-NI (NJ.le.MALPB(ni)).
  !            -if ALPB(nj,ni) is nonzero and positive.
  ! The standard ALPHA parameter ALP(ni) is used if any one of there
  ! conditions is not satisfied.
  if(iqm_mode == 5) then  ! mndo/d
     MALPni = qm_param_c%MALPB(i)   ! MALPB(ni)
     MALPnj = qm_param_c%MALPB(j)   ! MALPB(nj)
     if(MALPni > 0 .and. nj <= MALPni) then
        if(qm_param_c%ALPB(j,i) > ZERO) ALPNI = qm_param_c%ALPB(j,i)  ! ALPB(nj,ni)
     end if
     if(MALPnj > 0 .and. ni <= MALPnj) then
        if(qm_param_c%ALPB(i,j) > ZERO) ALPNJ = qm_param_c%ALPB(i,j)  ! ALPB(ni,nj)
     end if
  end if
  ! calculate scale factor including expeonetial terms.
  ENI    = EXP(-ALPNI*RIJ)
  if(ALPNI == ALPNJ) then  ! (NI == NJ)
     ENJ = ENI
  else
     ENJ = EXP(-ALPNJ*RIJ)
  end if
  scale_val  = ONE+ENI+ENJ

  ! H-O or H-N pairs
  if(ni == 1 .and. (nj == 7 .or. nj == 8)) then
     scale_val = scale_val + (RIJ-one)*ENJ
  else if(nj == 1 .and. (ni == 7 .or. ni == 8)) then
     scale_val = scale_val + (RIJ-one)*ENI
  end if

  ! calculate basic replusive term.
  if (do_d_orbitals) then
     ! since RIJ=R*A0, RIJ/A0=R
     ! WIJ = EV/SQRT(RIJ*RIJ/(A0*A0)+(PO_9(i)+PO_9(j))**2)
     WIJ = EV/SQRT(R*R+(qm_param_c%po(9,i)+qm_param_c%po(9,j))**2)
  end if 
  ENUC   = qm_param_c%core(i)*qm_param_c%core(j)*WIJ  ! CORE(ni)*CORE(nj)*WIJ

  ! core-core replusion terms for AM1/PM3/AM1/d
  ENUC1  = ZERO
  if(do_am1_pm3) call repam1_qmqm(i,j,NI,NJ,RIJ,ENUC1,q_specific_pair)

  ! Add AM1 core-core repulsion TERMS.
  ENUCLR = ENUC*scale_val + ENUC1

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
  use qm1_info, only : qm_main_c,qm_param_c  ! ,qm_scf_main_c
  !use qm1_parameters, only : III,IIID       ! ,CORE
  !use number
  !use qm1_constant

  implicit none
  !
  integer :: dim_linear_norbs
  real(chm_real):: PA(dim_linear_norbs),PB(dim_linear_norbs)

  ! local variables
  integer :: i,j,k,kk,ia,is
  integer :: NSPORB,NI,IORBS
  real(chm_real):: YY,W,TEMP,DA,DB,DC,FA,FB,charge,core
  real(chm_real),parameter :: r_12=1.0d0/twelve


  ! options.
  charge = qm_main_c%qmcharge

  ! diagonal trial density matrix.
  K      = 0
  PA(1:dim_linear_norbs)=zero

  ! NSPORB : number of atomic orbitals initially populated.
  !          D-orbitals of main-group elements are not populated.
  !          P-orbitals of transition elementes are not populated.
  NSPORB = qm_main_c%norbs
  do i=1,qm_main_c%numat
     !ni     = qm_param_c%ni_local(i)    ! qm_main_c%nat(i)
     !iorbs  = qm_param_c%iorbs_local(i) ! qm_main_c%num_orbs(i)
     !if(iorbs == 9) then
     !   if(qm_param_c%III(i) <= qm_param_c%IIID(i)) NSPORB=NSPORB-5  ! III(ni) <= IIID(ni)
     !   if(qm_param_c%III(i) >  qm_param_c%IIID(i)) NSPORB=NSPORB-3  ! III(ni) >  IIID(ni)
     !end if
     if(qm_param_c%iorbs_local(i) == 9) then  ! iorbs == 9
        if(qm_param_c%III(i) <= qm_param_c%IIID(i)) then
           NSPORB=NSPORB-5
        else
           NSPORB=NSPORB-3
        end if
     end if
  end do
  YY     = charge/real(NSPORB)
  ! now, loop over all atoms.
  do i=1,qm_main_c%numat
     ia     = qm_param_c%ia_local(i)    ! qm_main_c%nfirst(I)
     iorbs  = qm_param_c%iorbs_local(i) ! qm_main_c%num_orbs(i)
     is     = qm_param_c%is_local(i)    ! qm_scf_main_c%indx(ia)+ia ! =ia*(ia+1)/2
     !!ni     = qm_param_c%ni_local(i)    ! qm_main_c%nat(I)
     core   = qm_param_c%core(i)        ! CORE(qm_main_c%nat(i))
     if(iorbs == 1) then
        ! atoms with an S-basis
        PA(is) = (core-YY)*PT5
     else if(iorbs == 4) then
       ! atoms with an SP-basis.
        W             =(core*PT25-YY)*PT5
        PA(is)        = W
        PA(is+ia+1)   = W
        PA(is+2*ia+3) = W
        PA(is+3*ia+6) = W
     else if((iorbs == 9) .and. (qm_param_c%III(i) <= qm_param_c%IIID(i))) then ! III(ni) <= IIID(ni)
        ! main-group elements with an SPD-basis.
        W             =(core*PT25-YY)*PT5
        PA(is)        = W
        PA(is+ia+1)   = W
        PA(is+2*ia+3) = W
        PA(is+3*ia+6) = W
     else if((iorbs == 9) .and. (qm_param_c%III(i) >  qm_param_c%IIID(i))) then ! (III(ni) >  IIID(ni)
        ! transition-metal elements with an SPD-basis.
        ! up to 10 electrons, put into S and D orbitals.
        ! more than 10 electrons, fill D orbitials and put the rest in S orbitail.
        temp = core-YY*six
        if(temp < ten) then
           W              = temp*r_12      ! /twelve
           PA(is)         = W
           PA(is+4*IA+10) = W
           PA(is+5*IA+15) = W
           PA(is+6*IA+21) = W
           PA(is+7*IA+28) = W
           PA(is+8*IA+36) = W
        else
           W              =(temp-ten)*PT5
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
  if(qm_main_c%UHF) then
     da  = real(qm_main_c%nalpha)
     db  = real(qm_main_c%nbeta)
     temp= two/(da+db)
     fa  = da*temp
     fb  = db*temp
     do i=1,dim_linear_norbs
        PB(i) = PA(i)*fb
        PA(i) = PA(i)*fa
     end do
     if(qm_main_c%nalpha == qm_main_c%nbeta) then
        da = 0.98D0
        db = two-da
        k  = 0
        do j=1,qm_main_c%norbs
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
  use qm1_info, only : qm_control_c,qm_main_c,qm_scf_main_c,mm_main_c,qm_param_c
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
  logical :: qi_h,qj_h, q_specific_pair,qlocal
  integer :: istart_norbs,iend_norbs
#if KEY_PARALLEL==1
  !!integer :: ISTRT_CHECK            ! external function
  integer :: mmynod, nnumnod,icnt
#endif

#if KEY_PARALLEL==1
  mmynod  = mynod
  nnumnod = numnod
  !if(nnumnod>1) istart_norbs = ISTRT_CHECK(iend_norbs,qm_main_c%norbs)

  istart_norbs = qm_main_c%norbs*mynod/numnod+1
  iend_norbs   = qm_main_c%norbs*(mynod+1)/numnod
#else
  istart_norbs  = 1
  iend_norbs    = qm_main_c%norbs
#endif

  ! initialize some variables.
  kr     = 0
  Enuclr = zero
  !h(1:linear_norbs)=zero    ! initialized in scf_energy
  !w(1:linear_fock2)=zero    !

  ! Diagonal one-center terms: replace "call one_center_h(H)"
  !          as this was precomputed at the beginning of QM setup.
  !          see subroutine compute_one_center_h.

  ! note: this should be carefully handled in using parallel as H_1cent is filled for each atom 
  !       at the subroutine compute_one_center_h. So, either only fill for part of atoms based on
  !       parallel partitioning or for 1:qm_main_c%norbs should be changed or do only for the
  !       master (or only one) node.
  !#if KEY_PARALLEL==1
  !!if(mmynod.eq.0) then
  !#endif
  !!   h(qm_scf_main_c%imap_h(1:qm_main_c%norbs))=qm_scf_main_c%H_1cent(1:qm_main_c%norbs)
  !!#if KEY_PARALLEL==1
  !!end if
  !!#endif
  !! taken care for parallelization.
  h(qm_scf_main_c%imap_h(istart_norbs:iend_norbs))=qm_scf_main_c%H_1cent(istart_norbs:iend_norbs)

  ! loop over atom pairs for offdiagonal two-center terms.
  ! atoms i and j are identified at the beginning of the loop.
#if KEY_PARALLEL==1
  icnt=0
#endif
  iicnt=0
  loopii: do i=2,qm_main_c%numat
     ni     = qm_param_c%ni_local(i)     ! qm_main_c%nat(i)
     ia     = qm_param_c%ia_local(i)     ! qm_main_c%nfirst(i)
     is     = qm_param_c%is_local(i)     ! qm_scf_main_c%indx(ia)+ia
     iorbs  = qm_param_c%iorbs_local(i)  ! qm_main_c%num_orbs(i)
     iw     = qm_param_c%iw_local(i)     ! qm_scf_main_c%indx(iorbs)+iorbs
     ip     = qm_param_c%ip_local(i)     ! qm_scf_main_c%NW(i)
     qlocal = .not.qm_param_c%q_atom_specific(i)

     qi_h   = ni.eq.1  ! h-atom?
     loopjj: do j=1,i-1
        if(ni > 1 .or. qm_param_c%ni_local(j) > 1) iicnt = iicnt + 1
#if KEY_PARALLEL==1
        icnt = icnt + 1
        if(mmynod .ne. mod(icnt-1,nnumnod)) cycle loopjj
#endif
        nj     = qm_param_c%ni_local(j)     ! qm_main_c%nat(J)
        ja     = qm_param_c%ia_local(j)     ! qm_main_c%nfirst(j)
        js     = qm_param_c%is_local(j)     ! qm_scf_main_c%indx(ja)+ja
        jorbs  = qm_param_c%iorbs_local(j)  ! qm_main_c%num_orbs(j)
        jw     = qm_param_c%iw_local(j)     ! qm_scf_main_c%indx(jorbs)+jorbs
        jp     = qm_param_c%ip_local(j)     ! qm_scf_main_c%NW(j)

        if((ni==nj) .and. (.not.qm_param_c%q_atom_specific(j) .and. qlocal)) then
           q_specific_pair =.false.
        else
           q_specific_pair =.true.  ! default, such that pair terms.
        end if

        qj_h   = nj.eq.1 ! h-atom?
        ! for H-H pair.
        if(qi_h .and. qj_h) then
           call hhpair(j,i,j,i,qm_main_c%qm_coord,Hij,Wij,En,q_specific_pair)
           H(qm_scf_main_c%indx(ia)+ja) = Hij
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
           call rotmat(J,I,JORBS,IORBS,qm_main_c%NUMAT,qm_main_c%qm_coord,R,qm_scf_main_c%YY)
           ! two-electron integrals in local coordinates & compute and store
           !              the semiempirical integrals.
           call repp_qmqm(i,j,NI,NJ,R,qm_scf_main_c%RI,qm_scf_main_c%CORE_mat,     &
                          qm_param_c%LORBS(i),qm_param_c%LORBS(j),                 &
                          qm_param_c%core(i), qm_param_c%core(j),                  &
                          qm_param_c%po(1:9,i),qm_param_c%po(1:9,j),               &
                          qm_param_c%dd(1:6,i),qm_param_c%dd(1:6,j),               &
                          q_specific_pair)
           if(iorbs>=9 .or. jorbs>=9)  &
              call reppd_qmqm(i,j,NI,NJ,R,qm_scf_main_c%RI,qm_scf_main_c%CORE_mat, &
                              qm_scf_main_c%WW,IW,JW, &
                              qm_param_c%po(1:9,i),qm_param_c%po(1:9,j), &
                              qm_param_c%dd(1:6,i),qm_param_c%dd(1:6,j), &
                              qm_param_c%core)
           ! transform two-electron integrals to molecular coordinates.
           if(iorbs<=4 .and. jorbs<=4) then
              call rotate(IW,JW,IP,JP,KR,qm_scf_main_c%RI,qm_scf_main_c%YY, &
                          W,W,linear_fock,linear_fock2,0)
           else
              call rotd(qm_scf_main_c%WW,qm_scf_main_c%YY,IW,JW)
              call w2mat(ip,jp,qm_scf_main_c%WW,W,linear_fock,iw,jw)
           end if

           ! resonance integrals.
           call betaij(ni,nj,iorbs,jorbs,R,qm_scf_main_c%T,                               &
                       qm_param_c%zs(i),qm_param_c%zp(i),qm_param_c%zd(i),                &
                       qm_param_c%betas(i),qm_param_c%betap(i),qm_param_c%betad(i),       &
                       qm_param_c%zs(j),qm_param_c%zp(j),qm_param_c%zd(j),                &
                       qm_param_c%betas(j),qm_param_c%betap(j),qm_param_c%betad(j))
           call rotbet(IA,JA,IORBS,JORBS,qm_scf_main_c%T,qm_scf_main_c%YY,H,linear_norbs, &
                       qm_scf_main_c%indx)

           ! core-electron attractions.
           call rotcora_qmqm(IA,JA,IORBS,JORBS,IS,JS,qm_scf_main_c%CORE_mat, &
                             qm_scf_main_c%YY,H,linear_norbs)

           ! core-core repulsions.
           Wij = qm_scf_main_c%RI(1)
           call core_repul(i,j,ni,nj,R,Wij,En,q_specific_pair)
           Enuclr = Enuclr+En                    ! combined Enuclr in the parent routine.
        end if
     end do loopjj
  end do loopii

  !!!! store MNDO integrals in square form: 
  ! The following call is done in the parent subroutine after gcomb, as
  ! this should be done the same change for each node.
  ! Complete defintion of square matrix of MNDO two-electron integrals
  ! the MNDO integrals in square form is stored in the parent subroutine after gcomb!
  !call wstore(W,linear_fock,0,qm_main_c%numat,qm_main_c%uhf)

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
!     use qm1_info, only : qm_main_c,qm_scf_main_c,qm_param_c
!     !!use qm1_parameters,only : USS,UPP,UDD
!
!     real(chm_real):: H(*)
!
!     ! local
!     integer :: i,ni,ia,ll,iorbs,j
!
!     ! DIAGONAL ONE-CENTER TERMS.
!     ! this can be done once at the beginning of QM setup.
!     ! work on this later to make it go over the loop once.
!     do i=1,qm_main_c%numat
!        ni     = qm_param_c%ni_local(i)    ! qm_main_c%NAT(i)
!        ia     = qm_param_c%ia_local(i)    ! qm_main_c%NFIRST(i)
!        iorbs  = qm_param_c%iorbs_local(i) ! qm_main_c%num_orbs(i)  ! = NLAST(I)-IA+1
!        H(qm_scf_main_c%INDX(ia)+ia) = qm_param_c%USS(i)
!        if(iorbs >= 9) then
!           do j=ia+1,ia+3
!              H(qm_scf_main_c%INDX(j)+j)  = qm_param_c%UPP(i)
!           end do
!           do j=ia+4,ia+8
!              H(qm_scf_main_c%INDX(j)+j)  = qm_param_c%UDD(i)
!           end do
!        else if(iorbs >= 4) then
!           do j=ia+1,ia+3
!              H(qm_scf_main_c%INDX(j)+j)  = qm_param_c%UPP(i)
!           end do
!        end if
!     end do
!     return
!     end subroutine one_center_h
  !=====================================================================
  end subroutine hcorep


  subroutine hhpair(jqm,iqm,j,i,coord,hij,wij,enuclr,q_specific_pair)
  !
  ! integrals for a hydrogen-hydrogen pair.
  !
  ! Notation:  (I): input; (O): output
  ! j,i     Atom pair i-j (I).
  ! coord   Cartesian coordinates in Angstrom (I).
  ! HIJ     Two-center resonance integral (O).
  ! WIJ     Two-center two-electron integrals (SS,SS) (O).
  ! Enuclr  Contribution to core-core repulsion (O).
  use qm1_info, only : qm_param_c
  !use qm1_parameters, only : ZS,PO,BETAS,ALP
  !
  implicit none

  integer :: jqm,iqm,j,i
  logical :: q_specific_pair
  real(chm_real):: COORD(3,*),HIJ,WIJ,ENUCLR

  ! local variables
  real(chm_real):: R,ZR,RIJ,ENI,SCALE,ZI
  real(chm_real),parameter :: r_three=one/three, &
                              r_A0   =one/A0

  ! the following (equations) needs to be checked.
  ! distance R (AU), and resonance integral Hij.
  ! 1s-1s overlap integral
  ! S(r_ij) = exp(-zr)*(1+zr+zr^2/3)
  r = SQRT((coord(1,j)-coord(1,i))**2+(coord(2,j)-coord(2,i))**2+(coord(3,j)-coord(3,i))**2)*r_A0
  if(q_specific_pair) then
     ! if two H atoms using different parameters
     zr =PT5*(qm_param_c%ZS(iqm)+qm_param_c%ZS(jqm))*r  ! zs(1)*r
     ! resonance integral Hij.
     if(zr < bigexp) then
        Hij = PT5*(qm_param_c%BETAS(iqm)+qm_param_c%BETAS(jqm))*EXP(-zr)*(one+zr+zr*zr*r_three)
     else
        Hij = zero
     end if
     ! two-electron integral Wij
     Wij    = EV/SQRT(r*r + (qm_param_c%PO(1,iqm)+qm_param_c%PO(1,jqm))**2)
     ! core-core replusion Enuclr.
     Rij    = r*A0
     Enuclr = Wij*(one + EXP(-qm_param_c%ALP(iqm)*Rij) + EXP(-qm_param_c%ALP(jqm)*Rij))
  else
     ! two h atoms with the same parameters 
     zr= qm_param_c%ZS(iqm)*r  ! zs(1)*r
     ! resonance integral Hij.
     if(zr < bigexp) then
        Hij = qm_param_c%BETAS(iqm)*EXP(-zr)*(one+zr+zr*zr*r_three)
     else
        Hij = zero
     end if
     ! two-electron integral Wij
     Wij    = EV/SQRT(r*r+FOUR*qm_param_c%PO(1,iqm)**2)
     ! core-core replusion Enuclr.
     Rij    = r*A0
     Enuclr = Wij*(one+two*(EXP(-qm_param_c%ALP(iqm)*Rij)))
  end if
  ! am1-type core-core terms
  if(do_am1_pm3) call repam1_hh_pair(iqm,jqm,1,RIJ,ENUCLR,q_specific_pair)

  return
  end subroutine hhpair

          
  subroutine mmint(H,dim_linear_norbs,ENUCLR)
  !
  ! Electrostatic contributions from mm point charges to the Core hamiltonian
  ! and the core-core repulsions.
  !
  !use chm_kinds
  use qm1_info, only : qm_control_c,qm_main_c,mm_main_c,qm_scf_main_c,qm_param_c
  use nbndqm_mod, only: map_grp_c
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

  integer :: i,ij,jj,irs_qm,irs_mm,numqm,ni,iorbs,iw,ia,is,lorbs
  real(chm_real):: XCOORD(3,2),PTCHG,PTCHG_SIGN,R,scale_val,enuc,r_sq,sw_scale
  real(chm_real):: RI_local(22),CORE_mat(10,2),po_qm(9),po_mm(9),dd_qm(6),dd_mm(6)
  integer       :: i_do_switching
  !
  integer :: mstart,mstop

  numqm   = qm_main_c%numat
#if KEY_PARALLEL==1
  mstart  = mm_main_c%numatm*mynod/numnod+1
  mstop   = mm_main_c%numatm*(mynod+1)/numnod
#else
  mstart  = 1
  mstop   = mm_main_c%numatm
#endif

  !
  po_mm(1:9) = qm_param_c%PO_mm(1:9)
  dd_mm(1:6) = qm_param_c%DD_mm(1:6)

  ! 
  ! MM point charges (M.J.Field et al. J.Comput.Chem. 11, 700 (1990))
  do m=mstart,mstop               ! 1,mm_main_c%numatm
     !if(mm_main_c%mm_chrgs(m).ne.zero) then

     if(mm_main_c%q_switch) irs_mm = map_grp_c%map_mmatom_to_group(m) 

     XCOORD(1:3,1) = mm_main_c%mm_coord(1:3,m)
     PTCHG         = mm_main_c%mm_chrgs(m)
     if(PTCHG >= zero) then
        PTCHG_SIGN = one
     else
        PTCHG_SIGN =-one
     end if
     ! loop over qm atoms for each mm atom.
     ENUCQM = zero
     do i=1,numqm
        XCOORD(1:3,2) = qm_main_c%qm_coord(1:3,i)
        iorbs         = qm_param_c%iorbs_local(i)
        ni            = qm_param_c%ni_local(i)
        ia            = qm_param_c%ia_local(i)
        is            = qm_param_c%is_local(i)
        iw            = qm_param_c%iw_local(i)
        !!lorbs         = qm_param_c%LORBS(i)     ! probably the same as iorbs
        po_qm(1:9)    = qm_param_c%po(1:9,i)
        dd_qm(1:6)    = qm_param_c%dd(1:6,i)

        ! default group-based case, include all mm atoms (default).
        if(mm_main_c%q_switch) irs_qm = map_grp_c%map_qmatom_to_group(i)

        ! distance R (au) and rotation matrix.
        call rotmat_qmmm(1,2,1,iorbs,2,XCOORD,R,qm_scf_main_c%YY)
        ! local charge-electron attraction integrals.
        call repp_qmmm(ni,0,i,iorbs,R,RI_local,CORE_mat,po_qm,po_mm,dd_qm,dd_mm)
        if(iorbs >= 9) call reppd_qmmm(ni,0,R,RI_local,CORE_mat, &
                                       qm_scf_main_c%WW,iw,1,po_qm,po_mm,dd_qm,dd_mm)

        ! multiplication by point charge.
        sw_scale = one
        if(mm_main_c%q_switch) then
           i_do_switching = mm_main_c%q_mmgrp_qmgrp_swt(irs_mm,irs_qm)
           ! apply switching function
           if(i_do_switching > 0) sw_scale = mm_main_c%sw_val(i_do_switching)
        end if
        CORE_mat(1:qm_param_c%Jmax_local(i),1) = CORE_mat(1:qm_param_c%Jmax_local(i),1)*PTCHG*sw_scale

        ! contributions to the core hamiltonian.
        call rotcora_qmmm(ia,0,iorbs,0,is,0,CORE_mat, &
                          qm_scf_main_c%YY,H,dim_linear_norbs,mm_main_c%q_diag_coulomb)
   
        ! contributions to the core-core repulsions.
        if(.not.mm_main_c%q_diag_coulomb) then
           R = R*A0
           scale_val= EXP(-qm_param_c%OMEGA(i)*(R-qm_param_c%DELTA(i)))+EXP(-five*R)
           enuc     = qm_param_c%core(i)*RI_local(1)*(one+PTCHG_SIGN*scale_val)
           if(do_am1_pm3) call repam1_qmmm(i,ni,0,R,enuc)

           ENUCQM = ENUCQM+enuc*sw_scale
        end if
     end do

     if(.not.mm_main_c%q_diag_coulomb) then
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
!     use qm1_info, only : qm_control_c
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
!        !!!if(qm_main_c%hlink(i).gt.0) cycle; assume even h-link fully interacts with all MM atoms.
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
!        call repp_qmmm(ni,0,i,lorbs,R,RI,CORE_mat,po_qm,po_mm,dd_qm,dd_mm)
!        if(iorbs.ge.9) call reppd_qmmm(ni,0,R,RI,CORE_mat,WW,iw,1)
!
!        ! multiplication by point charge.
!        CORE_mat(1:Jmax,1) = CORE_mat(1:Jmax,1)*PTCHG
!
!        ! contributions to the core hamiltonian.
!        call rotcora_qmmm(ia,0,iorbs,0,is,0,CORE_mat,YY,H,lin_dim,mm_main_c%q_diag_coulomb)
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

!!  ! moved to qmmm_interface.F90
!!  subroutine find_unique_qm(ntype_local)
!!  !
!!  ! Find the unique number of qm atoms. (need for the Grimme dispersion correction).
!!  !
!!  ! THis routine is only used to map with SCC DFTB data structure.
!!  !
!!  !use mndo97, only   : nndim ! ,izp
!!  use qm1_info, only : qm_main_c
!!
!!  implicit none
!!  integer :: ntype_local,i,j,ni,icnt
!!  logical, allocatable :: q_unique(:)
!!
!!  !
!!  allocate(q_unique(qm_main_c%numat))
!!  ! find number of unique atoms.
!!  do i=2,qm_main_c%numat
!!     ni          = qm_main_c%nat(i)
!!     q_unique(i) =.true.
!!     do j=1,i-1
!!        if(ni == qm_main_c%nat(j)) then
!!           ! find qm atom with the same atom type.
!!           q_unique(i) =.false.
!!           exit
!!        end if
!!     end do
!!  end do
!!  !
!!  !total number of unique qm atoms.
!!  nunique_qm= 1
!!  !izp(1)    = 1
!!  do i=2,qm_main_c%numat
!!     if(q_unique(i)) nunique_qm = nunique_qm + 1
!!     !izp(i) = nunique_qm  ! mapping to SCC DFTB format.
!!  end do
!!  ntype_local = nunique_qm
!!  deallocate(q_unique)
!!  return
!!  end subroutine find_unique_qm


  subroutine overlp (NI,NJ,iorbs,jorbs,rij,Z,zsi,zpi,zdi,zsj,zpj,zdj)
  !
  ! overlap integrals
  !
  ! Notation: (I) input; (O) output.
  ! Ni, Nj    atomic numbers of atoms i,j (I)
  ! Rij       internuclear distance in atomic units (I)
  ! Z(i)      overlap integrals (O).
  ! 
  use qm1_parameters,only: III,IIID ! ,ZS,ZP,ZD

  implicit none
  !
  integer :: NI,NJ,iorbs,jorbs
  real(chm_real):: Rij,Z(14),zsi,zpi,zdi,zsj,zpj,zdj

  ! local variables
  logical :: DIFF
  integer :: i,j,ii,ij,jj,k
  integer :: N1,N2,N1P,N2P,NT
  integer :: ISP,IPS,IOR,JOR,N1D,N2D
  real(chm_real):: ZIMIN,ZJMIN
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
  !!zsi    = ZS(ni)
  !!zpi    = ZP(ni)
  !!zsj    = ZS(nj)
  !!zpj    = ZP(nj)
  !!zdi    = ZD(ni)
  !!zdj    = ZD(nj)
  rij2   = rij*rij
  rij3   = rij2*rij

  ! check for immediate return (overlaps below threshold).
  ZIMIN  = MIN(ZSI,ZPI)
  ZJMIN  = MIN(ZSJ,ZPJ)
  if(iorbs >= 9) ZIMIN = MIN(ZIMIN,zdi)
  if(jorbs >= 9) ZJMIN = MIN(ZJMIN,zdj)
  !if((PT5*(ZIMIN+ZJMIN)*rij).gt.BIGEXP) return

  ! n1, n2 : main quantum numbers for S-orbitals.
  ! n1p,n2p: main quantum numbers for P-orbitals.
  diff   = zsi /= zpi .or. zsj /= zpj
  n1     = III(ni) 
  n2     = III(nj)
  n1p    = MAX(n1,2) ! MAX(III(ni),2)
  n2p    = MAX(n2,2) ! MAX(III(nj),2)
  nt     = n1+n2

  ! depending on main quantum numbers:
  if(n1 <= 3 .and. n2 <= 3) then   ! low quantum number atoms.
    if(n1 <= n2) then
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
        !if(ni == nj) then
        !   W   = (zsi*rij)**3
        !else
           W   = SQRT((zsi*zsj*rij2)**3)         ! rij*rij
        !end if
        Z(1)   = PT25*W*(A(3)*B(1)-B(3)*A(1))
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
      case (3)   ! 2nd row - 2nd row overlaps
        call set(nt,zsi,zsj,A,B,rij)
        !if(ni == nj) then
        !   W   = r_16*(zsi*rij)**5
        !else
           W   = r_16*SQRT((zsi*zsj*rij2)**5)   ! rij*rij
        !end if
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
           !if(ni == nj) then
           !  W = r_16*(zpi*rij)**5
           !else
             W = r_16*SQRT((zpi*zpj*rij2)**5)
           !end if
        end if
        Z(4)   = W*(B(3)*(A(5)+A(1))-A(3)*(B(5)+B(1)))
        Z(5)   = PT5*W*(A(5)*(B(1)-B(3))-B(5)*(A(1)-A(3))-A(3)*B(1)+B(3)*A(1))
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
      case (6)  ! 3rd row - 3rd row overlaps
        call set(nt,zsi,zsj,A,B,rij)
        !if(ni == nj) then
        !   W   = ((zsi*rij)**7)*r_480
        !else
           W   = SQRT((zsi*zsj*rij2)**7)*r_480
        !end if
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
           !if(ni == nj) then
           !  W = ((zpi*rij)**7)*r_480
           !else
             W = SQRT((zpi*zpj*rij2)**7)*r_480
           !end if
        end if
        Z(4)   = W*(A(3)*(B(7)+two*B(3))-A(5)*(B(1)+two*B(5))-B(5)*A(1)+A(7)*B(3))
        Z(5)   = PT5*W*(A(7)*(B(1)-B(3))     +B(7)*(A(1)-A(3))        &
                       +A(5)*(B(5)-B(3)-B(1))+B(5)*(A(5)-A(3)-A(1))   &
                       +two*A(3)*B(3))
    end select
  !
  ! overlaps involving higher rows.
  else if(n1 > 3 .or. n2 > 3) then
     call set(n1 +n2 ,zsi,zsj,A,B,rij)
     Z(1)   = ss(n1 ,0,0,n2 ,0,zsi*rij,zsj*rij,A,B)
     if(jorbs >= 4) then
        if(diff) call set(n1 +n2p,zsi,zpj,A,B,rij)
        Z(2) = ss(n1 ,0,0,n2p,1,zsi*rij,zpj*rij,A,B)
     end if
     if(iorbs >= 4) then
        if(diff) call set(n1p+n2 ,zpi,zsj,A,B,rij)
        Z(3) = ss(n1p,1,0,n2 ,0,zpi*rij,zsj*rij,A,B)
     end if
     if(iorbs >= 4 .and. jorbs >= 4) then
        if(diff) call set(n1p+n2p,zpi,zpj,A,B,rij)
        Z(4) = ss(n1p,1,0,n2p,1,zpi*rij,zpj*rij,A,B)
        Z(5) = ss(n1p,1,1,n2p,1,zpi*rij,zpj*rij,A,B)
     end if
  end if
  ! returns, if not having d-orbitals.
  !if(iorbs <= 4 .and. jorbs <= 4) return

  ! overlaps involving D-orbitals.
  if(iorbs >= 9 .or. jorbs >= 9) then
     !!zdi    = ZD(ni)
     !!zdj    = ZD(nj)
     n1d    = IIID(ni) 
     n2d    = IIID(nj) 
     if(iorbs >= 9 .and. jorbs <= 4) then
        call set(n1d+n2 ,zdi,zsj,A,B,rij)
        Z(6)  = ss(n1d,2,0,n2 ,0,zdi*rij,zsj*rij,A,B)
        if(jorbs == 4) then
           call set(n1d+n2p,zdi,zpj,A,B,rij)
           Z(8)  = ss(n1d,2,0,n2p,1,zdi*rij,zpj*rij,A,B)
           Z(10) = ss(n1d,2,1,n2p,1,zdi*rij,zpj*rij,A,B)*three
        end if
     else if(iorbs <= 4 .and. jorbs >= 9) then
        call set(n1 +n2d,zsi,zdj,A,B,rij)
        Z(7)  = ss(n1 ,0,0,n2d,2,zsi*rij,zdj*rij,A,B)
        if(iorbs == 4) then
           call set(n1p+n2d,zpi,zdj,A,B,rij)
           Z(9)  = ss(n1p,1,0,n2d,2,zpi*rij,zdj*rij,A,B)
           Z(11) = ss(n1p,1,1,n2d,2,zpi*rij,zdj*rij,A,B)*three
        end if
     else if(iorbs >= 9 .and. jorbs >= 9) then
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
      if(ABSX < CUTOFF) then
         ! zero argument
         B(1:n+1)=B0(1:n+1)
      else
         ! large argument
         if((ABSX > PT5 .and. n <= 5 ).or. (ABSX > one .and. n <= 7) .or. &
            (ABSX > two .and. n <= 10).or. (ABSX > three)) then
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
            if(ABSX <= PT5) then
               last = 6
            else if(ABSX <= one) then
               last = 7
            else if(ABSX <= two) then
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
      reduced_factor=SQRT(((ALPHA**iexpn)*(BETA**jexpn))/(FC(iexpn)*FC(jexpn)))

      if(LA == 0 .and. LB == 0) then
         ! overlap integrals involving S-functions.
         iada   = IAD(NA+1)
         iadb   = IAD(NB+1)
         do i=0,na
            iba    = IBINOM(iada+i)
            do j=0,nb
               ibb    = iba*IBINOM(iadb+j)
               if(MOD(j,2) == 1) ibb=-ibb
               ij     = i+j
               X      = X+float(ibb)*A(nab-ij)*B(ij+1)
            end do
         end do
         SS=X*PT5*reduced_factor

      else if (LA == 1 .and. LB == 1) then 
         ! overlap integrals involving P-functions.
         if(M <= 0) then
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
                        if(MOD(kd+kf,2) == 1) ibf=-ibf
                        X      = X+float(ibf)*A(ie-kf)*B(je+kf)
                     end do
                  end do
               end do
            end do
            ! overlap integral from reduced overlap integral.
            SS= X*SQRT(float((2*LA+1)*(2*LB+1))*PT25)*reduced_factor
            if(MOD(lb,2) == 1) SS=-SS
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
                  if(MOD(kf,2) == 1) ibf=-ibf
                  i=ie-kf
                  j=je+kf
                         !=(A(i)*B(j)-A(i)*B(j+2)-A(i-2)*B(j)+A(i-2)*B(j+2))
                  X=X+float(ibf)*((A(i)-A(i-2))*(B(j)-B(j+2)))
               end do
            end do
            ! overlap integral from reduced overlap integral.
            SS = X*PT75*reduced_factor
            if(MOD(LB+MM,2) == 1) SS=-SS
         end if

      else
         ! general case that LA > 1 or LB > 1, M >= 0.
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
                                 if(MOD(ka+kb+kd+kf,2) == 1) ibb=-ibb
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
         if(MOD(LB+MM,2) == 1) SS=-SS
      end if
      return
      end function ss
  !==end of contains===================================================!
  !====================================================================!

  end subroutine overlp


  subroutine repam1_qmqm(iqm,jqm,NI,NJ,R,ENUCLR,q_specific_pair)
  !
  ! Core repulsion function for AM1 and PM3.
  !
  ! The contributions from the Gaussian terms are added.
  ! 1. H-H pair is separated below as repam1_hh_pair.
  ! 2. QM-MM pair is separated below as repam1_qmmm.
  !
  use qm1_info, only: qm_param_c
  use qm1_parameters, only :  BORON1,BORON2,BORON3 ! CORE,GNN,GUESS1,GUESS2,GUESS3,IMPAR

  implicit none

  integer :: iqm,jqm,ni,nj
  real(chm_real):: r,ENUCLR
  logical :: q_specific_pair

  real(chm_real),parameter :: CUTOFF=25.0D0

  ! local variables
  integer :: nk,nl,imx,i,j
  real(chm_real):: ADD,XX,GNIJ

  ADD    = ZERO
  if(ni==5.or.nj==5) then  ! special section for AM1 and AM1/d
                           ! for atom pairs with Boron.
     if(iqm_mode==2.or.iqm_mode==4) then
        NK  = NI+NJ-5
        if(NK == 1) then       ! B-H pair
           NL=2
        else if(NK == 6) then  ! B-C pair
           NL=3
        else if(NK == 9.or.NK == 17.or.NK == 35.or.NK == 53) then
           NL=4                ! B-F/Cl/Br/I pairs
        else 
           NL=1                ! all others
        end if
        if(ni == 5) then
           qm_param_c%impar(iqm) = 3
           do i=1,3
              qm_param_c%GUESS1(i,iqm)=BORON1(i,NL)
              qm_param_c%GUESS2(i,iqm)=BORON2(i,NL)
              qm_param_c%GUESS3(i,iqm)=BORON3(i,NL)
           end do
        else if(nj == 5) then
           qm_param_c%impar(jqm) = 3
           do i=1,3
              qm_param_c%GUESS1(i,jqm)=BORON1(i,NL)
              qm_param_c%GUESS1(i,jqm)=BORON2(i,NL)
              qm_param_c%GUESS1(i,jqm)=BORON3(i,NL)
           end do
        end if
     end if
  end if
  ! GENERAL SECTION: since it will be only called for QM-QM pair (also not h-h pair).
  ! NI .gt. 0: NI is qm atom, the same for NJ
  do i=1,qm_param_c%impar(iqm)
     XX  = qm_param_c%GUESS2(i,iqm)*(r-qm_param_c%GUESS3(i,iqm))**2 
     if(xx<cutoff) ADD = ADD + qm_param_c%GUESS1(i,iqm)*EXP(-XX)
  end do

  if(q_specific_pair) then
     ! two atoms with different set of parameters
     do j=1,qm_param_c%impar(jqm)
        XX  = qm_param_c%GUESS2(j,jqm)*(r-qm_param_c%GUESS3(j,jqm))**2
        if(xx<cutoff) ADD = ADD + qm_param_c%GUESS1(j,jqm)*EXP(-XX)
     end do

     ! Gaussian Core-Core repulsion scaling.
     GNIJ= qm_param_c%GNN(iqm)*qm_param_c%GNN(jqm)*qm_param_c%core(iqm)*qm_param_c%core(jqm)
  else
     ! two atoms with the same set of parameters
     ADD = ADD+ADD

     ! Gaussian Core-Core repulsion scaling.
     GNIJ= qm_param_c%GNN(iqm)*qm_param_c%GNN(iqm)*qm_param_c%core(iqm)*qm_param_c%core(iqm)
  end if
  !
  ENUCLR = ENUCLR+GNIJ*ADD/r

  return
  end subroutine repam1_qmqm
  !=====================================================================


  subroutine repam1_hh_pair(iqm,jqm,NI,R,ENUCLR,q_specific_pair)
  !
  ! Core repulsion function for AM1 and PM3 for H-H pair.
  !
  !use chm_kinds
  use qm1_info, only: qm_param_c
  !!use qm1_parameters, only : CORE,GNN,GUESS1,GUESS2,GUESS3,IMPAR
  !use number,only : zero

  implicit none

  integer :: iqm,jqm,ni
  real(chm_real):: r,ENUCLR
  logical :: q_specific_pair

  real(chm_real),parameter :: CUTOFF=25.0D0

  ! local variables
  integer :: nk,nl,imx,i,j
  real(chm_real):: ADD,XX,GNIJ

  ADD    = ZERO
  ! since NI=NJ=1.
  do i=1,qm_param_c%IMPAR(iqm)
     XX  = qm_param_c%GUESS2(i,iqm)*(r-qm_param_c%GUESS3(i,iqm))**2
     ADD = ADD + qm_param_c%GUESS1(i,iqm)*EXP(-XX)
  end do
  if(q_specific_pair) then
     ! two H atoms with different set of parameters
     do j=1,qm_param_c%IMPAR(jqm)
        XX  = qm_param_c%GUESS2(j,jqm)*(r-qm_param_c%GUESS3(j,jqm))**2
        ADD = ADD + qm_param_c%GUESS1(j,jqm)*EXP(-XX)
     end do

     ! Gaussian Core-Core repulsion scaling.
     GNIJ= qm_param_c%GNN(iqm)*qm_param_c%GNN(jqm)*qm_param_c%core(iqm)*qm_param_c%core(jqm)
  else
     ! two H atoms with the same set of parameters
     ADD = ADD+ADD

     ! Gaussian Core-Core repulsion scaling.
     GNIJ= qm_param_c%GNN(iqm)*qm_param_c%GNN(iqm)*qm_param_c%core(iqm)*qm_param_c%core(iqm)
  end if
  !
  ENUCLR = ENUCLR+GNIJ*ADD/r

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
  use qm1_info, only: qm_param_c
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
  if((ni == 5).and.(iqm_mode == 2 .or. iqm_mode == 4)) then
     ! special section for AM1 and AM1/d. atom pairs involing Boron.
     NL=1                ! all others
     qm_param_c%impar(iqm) = 3
     do i=1,3
        qm_param_c%GUESS1(i,iqm)=BORON1(i,NL)
        qm_param_c%GUESS2(i,iqm)=BORON2(i,NL)
        qm_param_c%GUESS3(i,iqm)=BORON3(i,NL)
     end do
  end if
  ! QM-MM specific...
  ! Since ni is qm atom, Ni>0, and since nj is mm atom, Nj=0
  do i=1,qm_param_c%impar(iqm)
     XX  = qm_param_c%GUESS2(i,iqm)*(R-qm_param_c%GUESS3(i,iqm))**2 
     if(XX<cutoff) ADD = ADD + qm_param_c%GUESS1(i,iqm)*EXP(-XX)
  end do
  ! Gaussian Core-Core repulsion scaling.
  GNIJ= qm_param_c%GNN(iqm)*qm_param_c%core(iqm)

  ENUCLR = ENUCLR+GNIJ*ADD/R

  return
  end subroutine repam1_qmmm
  !=====================================================================


  subroutine repp_qmqm(iqm,jqm,NI,NJ,R,A,CORE_mat,                    &
                       iorbs,jorbs,coreni,corenj,po_i,po_j,dd_i,dd_j, &
                       q_specific_pair)
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
  !!use qm1_info,only : qm_control_c
            
  implicit none

  integer :: iqm,jqm,NI,NJ,iorbs,jorbs
  real(chm_real):: R,A(22),CORE_mat(10,2)
  real(chm_real):: coreni,corenj,PO_i(9),PO_j(9),DD_i(3),DD_j(3)
  logical :: q_specific_pair

  ! local variables:
  integer :: i
  real(chm_real):: COREV,R2,EE,DA,DB,QA,QB,                           &
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
  !iorbs   = qm_param_c%LORBS(iqm) ! LORBS(ni)
  !jorbs   = qm_param_c%LORBS(jqm) ! LORBS(nj)
  !coreni  = qm_param_c%core(iqm)  ! CORE_local(iqm)  !CORE(ni)
  !corenj  = qm_param_c%core(jqm)  ! CORE_local(jqm)  !CORE(nj)
  !
  R2        = R*R
  AEE       =(PO_i(1) + PO_j(1))**2  ! (PO(1,ni)+PO(1,nj))**2
  ! H - H pair
  if(iorbs <= 1 .and. jorbs <= 1) then
     EE     = one/SQRT(R2+AEE)
     A(1)   = EE*eV
     CORE_mat(1,1) = -CORENJ*A(1)
     CORE_mat(1,2) = -CORENI*A(1)
  ! Heavy atom - H pair
  else if(iorbs >= 4 .and. jorbs <= 1) then
     DA     = DD_i(2)               ! DD(2,ni)
     QA     = DD_i(3)               ! DD(3,ni)
     TWOQA  = QA+QA
     ADE    = (PO_i(2)+PO_j(1))**2  ! (PO(2,ni)+PO(1,nj))**2
     AQE    = (PO_i(3)+PO_j(1))**2  ! (PO(3,ni)+PO(1,nj))**2
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
  else if(iorbs <= 1 .and. jorbs >= 4) then
     DB     = DD_j(2)               ! DD(2,nj)
     QB     = DD_j(3)               ! DD(3,nj)
     TWOQB  = QB+QB
     AED    = (PO_i(1)+PO_j(2))**2  ! (PO(1,ni)+PO(2,nj))**2
     AEQ    = (PO_i(1)+PO_j(3))**2  ! (PO(1,ni)+PO(3,nj))**2
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
     DA     = DD_i(2)               ! DD(2,ni)
     QA     = DD_i(3)               ! DD(3,ni)
     TWOQA  = QA+QA
     TWOQA2 = TWOQA*TWOQA
     RPDA2  = (R+DA)**2
     RMDA2  = (R-DA)**2
     RP2QA2 = (R+TWOQA)**2
     RM2QA2 = (R-TWOQA)**2
     ADE    = (PO_i(2)+PO_j(1))**2  ! (PO(2,ni)+PO(1,nj))**2
     AQE    = (PO_i(3)+PO_j(1))**2  ! (PO(3,ni)+PO(1,nj))**2
     ADD    = (PO_i(2)+PO_j(2))**2  ! (PO(2,ni)+PO(2,nj))**2
     ADQ    = (PO_i(2)+PO_j(3))**2  ! (PO(2,ni)+PO(3,nj))**2
     AQQ    = (PO_i(3)+PO_j(3))**2  ! (PO(3,ni)+PO(3,nj))**2
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

     if(ni == nj .and. q_specific_pair) then ! the same atom pairs with the same parameters
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
     else                                    ! different atom pairs
        DB     = DD_j(2)                ! DD(2,nj)
        QB     = DD_j(3)                ! DD(3,nj)
        TWOQB  = QB+QB
        TWOQB2 = TWOQB*TWOQB
        TWOQBQ = TWOQB2+AQQ
        RPDB2  = (R+DB)**2
        RMDB2  = (R-DB)**2
        RP2QB2 = (R+TWOQB)**2
        RM2QB2 = (R-TWOQB)**2
        AED    = (PO_i(1)+PO_j(2))**2  ! (PO(1,ni)+PO(2,nj))**2
        AEQ    = (PO_i(1)+PO_j(3))**2  ! (PO(1,ni)+PO(3,nj))**2
        AQD    = (PO_i(3)+PO_j(2))**2  ! (PO(3,ni)+PO(2,nj))**2
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
     ACI    = PO_i(9) ! PO(9,ni)
     ACJ    = PO_j(9) ! PO(9,nj)
     ! electrons at atom A (ni) and Core of atom B (nj).
     if(abs(ACJ-PO_j(1)) > small) then
        CORE_mat(1,1) = -CORENJ*eV/SQRT(R2+(PO_i(1)+ACJ)**2)
        if(iorbs >= 4) then
           DA     = DD_i(2)
           QA     = DD_i(3)
           AEJ    = (PO_i(7)+ACJ)**2
           ADJ    = (PO_i(2)+ACJ)**2
           AQJ    = (PO_i(3)+ACJ)**2
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
     if(abs(ACI-PO_i(1)) > small) then
        CORE_mat(1,2) = -CORENI*eV/SQRT(R2+(PO_j(1)+ACI)**2)
        if(jorbs >= 4) then
           DB     = DD_j(2)
           QB     = DD_j(3)
           AEI    = (PO_j(7)+ACI)**2
           ADI    = (PO_j(2)+ACI)**2
           AQI    = (PO_j(3)+ACI)**2
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


  subroutine repp_qmmm(NI,NJ,iqm,lorbs,R,A,CORE_mat,PO_qm,PO_mm,DD_qm,DD_mm)
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
  !!use qm1_parameters, only : LORBS ! ,CORE,DD,PO
  use qm1_info, only : qm_control_c
    
  implicit none

  integer :: NI,NJ,iqm,lorbs
  real(chm_real):: R,A(22),CORE_mat(10,2),PO_qm(9),PO_mm(9),DD_qm(6),DD_mm(6)

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
  iorbs  = lorbs      ! LORBS(ni)  ! qm atom
  jorbs  = 1          ! mm atom
  corenj = one
  !
  R2        = R*R
  AEE       =(PO_qm(1)+PO_mm(1))**2 ! (PO(1,ni)+PO(1,nj))**2
  ! H - H pair
  if(iorbs <= 1) then  ! (iorbs.le.1 .and. jorbs.le.1)
     EE     = one/SQRT(R2+AEE)
     A(1)   = EE*eV
     CORE_mat(1,1) = -CORENJ*A(1)
  ! Heavy atom - H pair
  else if(iorbs >= 4) then ! (iorbs.ge.4 .and. jorbs.le.1)
     DA     = DD_qm(2)  ! DD(2,ni)
     QA     = DD_qm(3)  ! DD(3,ni)
     TWOQA  = QA+QA
     ADE    = (PO_qm(2)+PO_mm(1))**2  ! (PO(2,ni)+PO(1,nj))**2
     AQE    = (PO_qm(3)+PO_mm(1))**2  ! (PO(3,ni)+PO(1,nj))**2
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
  if (iw > 1 .or. jw > 1) then
     rsum(1:6)=yy(1:6,6)+yy(1:6,10)

     ! integral types (SS,PS) and (SS,PP).
     if(jw > 1) then
        SSPB(IPP(1:3))= RI(5)*YY(1:3,2)  ! ipp=1,3,6
        SSPB(ISS(1:6))= RI(11)*YY(1:6,3)+RI(12)*rsum(1:6)
     end if

     ! integral types (PS,SS) and (PP,SS).
     if(iw > 1) then
        rPASS(IPP(1:3))= RI(2)*YY(1:3,2)  ! ipp=1,3,6
        rPASS(ISS(1:6))= RI(3)*YY(1:6,3)+RI(4)*rsum(1:6)
     end if

     if(iw > 1 .and. jw > 1) then
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
  if(imode == 1) then   ! using linear array W. (call from dhcore)
     k    = kr+1
     w(k) = RI(1)
     ! integral types (SS,PS) and (SS,PP).
     if(jw > 1) w(k+1:k+9) = SSPB(1:9)
     ! integral types (PS,SS) and (PP,SS).
     if(iw > 1 .and. jw == 1) w(k+1:k+9) = rPASS(1:9)
     if(iw > 1 .and. jw > 1) then
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
  else  ! imode == 0, using square array WW. (call from hcorep)
     ww(ip,jp) = RI(1)
     ! integral type (SS,PS) and (SS,PP).
     if(jw > 1) ww(ip,jp+1:jp+9) = SSPB(1:9)
     ! integral type (PS,SS) and (PP,SS).
     if(iw > 1) ww(ip+1:ip+9,jp) = rPASS(1:9)
     if(iw > 1 .and. jw > 1) then
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
  if(iorbs == 1 .and. jorbs == 1) return

  ! S(I)-P(J)
  if(jorbs >= 4) H(is+1:is+3) = T(2)*YY(1:3,2)

  ! P(I)-S(J)
  if(iorbs >= 4) then
     ix      = indx(ia+1)+ja
     iy      = indx(ia+2)+ja
     iz      = indx(ia+3)+ja
     H(ix)   = T(3)*YY(1,2)
     H(iy)   = T(3)*YY(2,2)
     H(iz)   = T(3)*YY(3,2)
     ! P(I)-P(J).
     if(jorbs >= 4) then
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
  if(iorbs >= 9) then
     do i=1,5
        H(indx(ia+3+i)+ja) = T(6)*YY(i,11)
     end do
     ! D(I)-P(J)
     if(jorbs >= 4) then
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
  if(jorbs >= 9) then
     M          = indx(ia)+ja+3
     H(M+1:M+5) = T(7)*YY(1:5,11)
     ! P(I)-D(J)
     if(iorbs >= 4) then
        do i=1,3
           M      = indx(ia+i)+ja+3
           do j=1,5
              ij     = 3*(j-1)+i
              H(M+j) = T(9)*YY(ij,12)+T(11)*(YY(ij,18)+YY(ij,25))
           end do
        end do
        ! D(I)-D(J)
        if(iorbs >= 9) then
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
  implicit none

  integer :: IA,JA,IORBS,JORBS,IP,JP,LMH
  real(chm_real):: CORE_mat(10,2),YY(15,45),H(LMH)

  ! local variables
  integer :: i,j,k,kk,is,L,ix,iy,iz,idp,idd,id
  real(chm_real):: HPP(6),HDP(15),HDD(15),YY_1(15),YY_2(15),YY_3(15)

  do kk=1,2
     if(kk == 1) then
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
     if(L >= 4) then
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

        if(L >= 9) then
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
  if(sb > small) then  ! it atoms are on z-axis.
     ca  = x(1)/sqb
     sa  = x(2)/sqb
     cb  = x(3)/r
  else
     SA  = zero
     SB  = zero
     if(x(3) < zero) then
        CA  =-one
        CB  =-one
     else if(x(3) > zero) then
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
     if(iorbs >= 9 .or. jorbs >= 9) then
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
  !use qm1_info, only : mm_main_c
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

  x(1:3)= coord(1:3,j)-coord(1:3,i)

  b     = x(1)*x(1)+x(2)*x(2)
  r     = SQRT(b+x(3)*x(3))
  sqb   = SQRT(b)  ! =sqrt(x(1)**2+x(2)**2), if atoms are z-axix, it will be zero
  sb    = sqb/r    ! normalized sqb by r.
  ! check for special case (both atoms on z axis).
  if(sb > small) then  ! it atoms are on z-axis.
     ca = x(1)/sqb
     sa = x(2)/sqb
     cb = x(3)/r
  else
     SA = zero
     SB = zero
     if(x(3) < zero) then
        CA =-one
        CB =-one
     else if(x(3) > zero) then
        CA = one
        CB = one
     else
        CA = zero
        CB = zero
     end if
  end if
  R     = r*r_A0  ! /A0; convert distance to atomic unit.

  ! precombine rotation matrix elements: jorbs == 1
  ! for S-S pair..
  YY(1,1) = one
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
!  use qm1_info, only : qm_scf_main_c
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
!     k      = qm_scf_main_c%IP1(kl)
!     l      = qm_scf_main_c%IP2(kl)
!     ckl(kl)= C3(k)*C4(l)
!     if(k.ne.l) ckl(kl)=ckl(kl)+C3(l)*C4(k)
!  end do
!  do ij=1,LM6
!     wij=DOT_PRODUCT(ckl(1:LM6),w(1:LM6,ij))
!     I      = qm_scf_main_c%IP1(ij)
!     J      = qm_scf_main_c%IP2(ij)
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
  use qm1_info, only : qm_param_c  ! ,qm_main_c,qm_scf_main_c
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
        ip  = qm_param_c%ip_local(ii)          ! nw(ii)
        !ni  = qm_param_c%ni_local(ii)         ! nat(ii)
        w(ip,ip) = qm_param_c%GSS_local(ii)    ! GSS(ni)
        !iorbs    = qm_param_c%iorbs_local(ii) ! num_orbs(ii)  ! NLAST(II)-NFIRST(II)+1
        if(qm_param_c%iorbs_local(ii) >= 4) then  ! iorbs >= 4
           ipx = ip+2
           ipy = ip+5
           ipz = ip+9
           !GSPNI = qm_param_c%GSP_local(ii) 
           !GPPNI = qm_param_c%GPP_local(ii)
           !GP2NI = qm_param_c%GP2_local(ii)
           !HSPNI = qm_param_c%HSP_local(ii)
           !HPPNI = qm_param_c%HPP_local(ii)

           w(ipx ,ip  ) = qm_param_c%GSP_local(ii)  ! GSPNI ! GSP(ni)
           w(ipy ,ip  ) = qm_param_c%GSP_local(ii)  ! GSPNI ! GSP(ni)
           w(ipz ,ip  ) = qm_param_c%GSP_local(ii)  ! GSPNI ! GSP(ni)
           !w(ip  ,ipx ) = GSPNI ! GSP(ni)
           !w(ip  ,ipy ) = GSPNI ! GSP(ni)
           !w(ip  ,ipz ) = GSPNI ! GSP(ni)
           w(ipx ,ipx ) = qm_param_c%GPP_local(ii)  ! GPPNI ! GPP(ni)
           w(ipy ,ipy ) = qm_param_c%GPP_local(ii)  ! GPPNI ! GPP(ni)
           w(ipz ,ipz ) = qm_param_c%GPP_local(ii)  ! GPPNI ! GPP(ni)
           w(ipy ,ipx ) = qm_param_c%GP2_local(ii)  ! GP2NI ! GP2(ni)
           w(ipz ,ipx ) = qm_param_c%GP2_local(ii)  ! GP2NI ! GP2(ni)
           w(ipz ,ipy ) = qm_param_c%GP2_local(ii)  ! GP2NI ! GP2(ni)
           !w(ipx ,ipy ) = GP2NI ! GP2(ni)
           !w(ipx ,ipz ) = GP2NI ! GP2(ni)
           !w(ipy ,ipz ) = GP2NI ! GP2(ni)
           w(ip+1,ip+1) = qm_param_c%HSP_local(ii)  ! HSPNI ! HSP(ni)
           w(ip+3,ip+3) = qm_param_c%HSP_local(ii)  ! HSPNI ! HSP(ni)
           w(ip+4,ip+4) = qm_param_c%HPP_local(ii)  ! HPPNI ! HPP(ni)
           w(ip+6,ip+6) = qm_param_c%HSP_local(ii)  ! HSPNI ! HSP(ni)
           w(ip+7,ip+7) = qm_param_c%HPP_local(ii)  ! HPPNI ! HPP(ni)
           w(ip+8,ip+8) = qm_param_c%HPP_local(ii)  ! HPPNI ! HPP(ni)
           if(qm_param_c%iorbs_local(ii) >= 9) then ! iorbs >= 9
              ij0    = ip-1
              do i=1,243
                 !w(intij(i)+ij0,intkl(i)+ij0) = REPD(INTREP(i),ni)
                 w(qm_param_c%int_ij(i)+ij0,qm_param_c%int_kl(i)+ij0) = qm_param_c%w_save(i,ii)
              end do
           end if
        end if
     end do
  else
     do ii=istart,numat,nnumnod
        ip  = qm_param_c%ip_local(ii) ! NW(ii)
        ! ni  = qm_param_c%ni_local(ii) ! nat(ii)
        w(ip,ip) = qm_param_c%GSS_local(ii) ! GSS_local(ii)*PT5  ! GSS(ni)*PT5
        !iorbs    = qm_param_c%iorbs_local(ii) ! num_orbs(ii)  ! NLAST(II)-NFIRST(II)+1
        if(qm_param_c%iorbs_local(ii) >= 4) then  ! iorbs >= 4
           ipx = ip+2
           ipy = ip+5
           ipz = ip+9
           !GSPNI = GSP(ni)-HSP(ni)*PT5
           !GPPNI = GPP(ni)*PT5
           !GP2NI = GP2(ni)-HPP(ni)*PT5
           !HSPNI = HSP(ni)*PT75-GSP(ni)*PT25
           !HPPNI = HPP(ni)*PT75-GP2(ni)*PT25

           ! already computed the following multiplications (see QMMM_module_prep)
           !!GSPNI = qm_param_c%GSP_local(ii) ! GSP_local(ii)-HSP_local(ii)*PT5
           !!GPPNI = qm_param_c%GPP_local(ii) ! GPP_local(ii)*PT5
           !!GP2NI = qm_param_c%GP2_local(ii) ! GP2_local(ii)-HPP_local(ii)*PT5
           !!HSPNI = qm_param_c%HSP_local(ii) ! HSP_local(ii)*PT75-GSP_local(ii)*PT25
           !!HPPNI = qm_param_c%HPP_local(ii) ! HPP_local(ii)*PT75-GP2_local(ii)*PT25
           w(ipx ,ip  ) = qm_param_c%GSP_local(ii)  ! GSPNI
           w(ipy ,ip  ) = qm_param_c%GSP_local(ii)  ! GSPNI
           w(ipz ,ip  ) = qm_param_c%GSP_local(ii)  ! GSPNI
           !w(ip  ,ipx ) = GSPNI
           !w(ip  ,ipy ) = GSPNI
           !w(ip  ,ipz ) = GSPNI
           w(ipx ,ipx ) = qm_param_c%GPP_local(ii)  ! GPPNI
           w(ipy ,ipy ) = qm_param_c%GPP_local(ii)  ! GPPNI
           w(ipz ,ipz ) = qm_param_c%GPP_local(ii)  ! GPPNI
           w(ipy ,ipx ) = qm_param_c%GP2_local(ii)  ! GP2NI
           w(ipz ,ipx ) = qm_param_c%GP2_local(ii)  ! GP2NI
           w(ipz ,ipy ) = qm_param_c%GP2_local(ii)  ! GP2NI
           !w(ipx ,ipy ) = GP2NI
           !w(ipx ,ipz ) = GP2NI
           !w(ipy ,ipz ) = GP2NI
           w(ip+1,ip+1) = qm_param_c%HSP_local(ii)  ! HSPNI
           w(ip+3,ip+3) = qm_param_c%HSP_local(ii)  ! HSPNI
           w(ip+4,ip+4) = qm_param_c%HPP_local(ii)  ! HPPNI
           w(ip+6,ip+6) = qm_param_c%HSP_local(ii)  ! HSPNI
           w(ip+7,ip+7) = qm_param_c%HPP_local(ii)  ! HPPNI
           w(ip+8,ip+8) = qm_param_c%HPP_local(ii)  ! HPPNI
           if(qm_param_c%iorbs_local(ii) == 9) then ! iorbs >= 9
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
                 w(qm_param_c%int_ij(i)+ij0,qm_param_c%int_kl(i)+ij0) = qm_param_c%w_save(i,ii)
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
