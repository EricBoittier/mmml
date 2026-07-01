module mndgho_module
  use chm_kinds
  use number
  implicit none

  !
  TYPE, public :: qm_gho_info
  ! Store GHO related information 
  logical            :: q_gho=.FALSE.  ! Logical flag to use GHO atoms
  logical            :: uhfgho=.FALSE. ! Logical flag to use UHF/GHO
  integer            :: mqm16=16
  integer            :: numat=0        ! number of qm atoms
  integer            :: nqmlnk=0       ! number of GHO atoms
  integer            :: norbsgho=0     ! number of AOS, same as qm2_struct%norbs
  integer            :: norbhb,naos,lin_naos,lin_norbhb  ! used in scf iteraction step.
  integer            :: norbao,nactatm,lin_norbao        ! used in FTOFHB
  integer, POINTER   :: IQLINK(:)   => NULL() ! Pointer for GHO atom
  integer, POINTER   :: JQLINK(:,:) => NULL() ! Pointer for MM atoms connected to
                                              ! GHO atom (3,nqmlnk)
  integer, POINTER   :: KQLINK(:)   => NULL() ! Pointer for QM atom  connected to
                                              ! GHO atom;  Sizes are (nqmlnk)
  real(chm_real),POINTER   :: QMATMQ(:)=> NULL() ! Size  is  (nqmlnk) 
  real(chm_real),POINTER   :: BT(:)    => NULL() ! Size  is  (nqmlnk*mqm16)    C' = BT C
  real(chm_real),POINTER   :: BTM(:)   => NULL() ! Size  is  (nqmlnk*mqm16)    C  = BTM C'
  real(chm_real),POINTER   :: DBTMMM(:,:,:) => NULL() ! Size  is  (3,3,nqmlnk*mqm16)
  real(chm_real),POINTER   :: PHO(:)   => NULL() ! density matrix for GHO, size is
                                                 ! (norbs*(norbs+1)/2)
  real(chm_real),POINTER   :: PBHO(:)  => NULL() ! density matrix for GHO, size is
                                                 ! (norbs*(norbs+1)/2)
  real(chm_real),POINTER   :: FAOA(:)  => NULL() ! density matrix for GHO, size is
                                                 ! (norbs*(norbs+1)/2)
  real(chm_real),POINTER   :: FAOB(:)  => NULL() ! density matrix for GHO, 
                                                 ! Size is same as density matrix, 
                                                 ! which is (norbs*(norbs+1)/2)
  ! Local varibles only at qm2_scf and etc.
  real(chm_real), POINTER :: CAHB(:) =>NULL() ! Size is norbs*norbs
  real(chm_real), POINTER :: CBHB(:) =>NULL() !         norbs*norbs
  real(chm_real), POINTER :: DAHB(:) =>NULL() !         norbs*(norbs*+1)/2
  real(chm_real), POINTER :: DBHB(:) =>NULL() !         norbs*(norbs*+1)/2
  real(chm_real), POINTER :: FAHB(:) =>NULL() !         norbs*(norbs*+1)/2
  real(chm_real), POINTER :: FBHB(:) =>NULL() !         norbs*(norbs*+1)/2
  real(chm_real), POINTER :: PAHB(:) =>NULL() !         norbs*(norbs*+1)/2
  real(chm_real), POINTER :: PBHB(:) =>NULL() !         norbs*(norbs*+1)/2
  real(chm_real), POINTER :: PAOLD(:)=>NULL() !         norbs*(norbs*+1)/2
  real(chm_real), POINTER :: PBOLD(:)=>NULL() !         norbs*(norbs*+1)/2
  real(chm_real), POINTER :: FAHBwrk(:,:) =>NULL() !    norbs,norbs
  real(chm_real), POINTER :: FBHBwrk(:,:) =>NULL() !    norbs,norbs
     
  END TYPE qm_gho_info

  ! assign
  TYPE(qm_gho_info), save :: qm_gho_info_r


  contains

#if KEY_MNDO97==1 /*mndo97*/
  subroutine allocate_deallocate_gho(qm_scf_main_l,qm_gho_info_l, &
                                     qdiis,uhf,qallocate)
  !
  ! allocate/deallocate gho/scf related arrays.
  ! if qallocate == .true. , allocate memory
  !                  false., deallocate memory
  use qm1_info,only : qm_scf_main,Aass

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

  ! deallocate if arrays are associated.
  ! for alpha orbitals.
  if(associated(qm_gho_info_l%PHO))   deallocate(qm_gho_info_l%PHO,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','PHO')
  if(associated(qm_gho_info_l%FAOA))  deallocate(qm_gho_info_l%FAOA,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','FAOA')
  if(associated(qm_gho_info_l%CAHB))  deallocate(qm_gho_info_l%CAHB,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','CAHB')
  if(associated(qm_gho_info_l%DAHB))  deallocate(qm_gho_info_l%DAHB,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','DAHB')
  if(associated(qm_gho_info_l%FAHB))  deallocate(qm_gho_info_l%FAHB,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','FAHB')
  if(associated(qm_gho_info_l%FAHBwrk))  deallocate(qm_gho_info_l%FAHBwrk,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','FAHBwrk')
  if(associated(qm_gho_info_l%PAHB))  deallocate(qm_gho_info_l%PAHB,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','PAHB')
  if(associated(qm_gho_info_l%PAOLD)) deallocate(qm_gho_info_l%PAOLD,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','PAOLD')

  ! for beta orbitals.
  if(associated(qm_gho_info_l%PBHO))  deallocate(qm_gho_info_l%PBHO,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','PBHO')
  if(associated(qm_gho_info_l%FAOB))  deallocate(qm_gho_info_l%FAOB,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','FAOB')
  if(associated(qm_gho_info_l%CBHB))  deallocate(qm_gho_info_l%CBHB,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','CBHB')
  if(associated(qm_gho_info_l%DBHB))  deallocate(qm_gho_info_l%DBHB,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','DBHB')
  if(associated(qm_gho_info_l%FBHB))  deallocate(qm_gho_info_l%FBHB,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','FBHB')
  if(associated(qm_gho_info_l%FBHBwrk))  deallocate(qm_gho_info_l%FBHBwrk,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','FBHBwrk')
  if(associated(qm_gho_info_l%PBHB))  deallocate(qm_gho_info_l%PBHB,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho','PBHB')
  if(associated(qm_gho_info_l%PBOLD)) deallocate(qm_gho_info_l%PBOLD,stat=ier)
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
  use qm1_info,only : Aass

  implicit none
  TYPE(qm_gho_info) :: qm_gho_info_l
  logical :: qdiis,uhf,qallocate

  integer :: ngho,ngho2
  integer :: ier=0

  ! define array sizes
  ngho  = qm_gho_info_l%nqmlnk
  ngho2 = qm_gho_info_l%nqmlnk * qm_gho_info_l%mqm16

  ! deallocate if arrays are associated.
  if(associated(qm_gho_info_l%IQLINK)) deallocate(qm_gho_info_l%IQLINK,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho_info','IQLINK')
  if(associated(qm_gho_info_l%JQLINK)) deallocate(qm_gho_info_l%JQLINK,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho_info','JQLINK')
  if(associated(qm_gho_info_l%KQLINK)) deallocate(qm_gho_info_l%KQLINK,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho_info','KQLINK')
  if(associated(qm_gho_info_l%QMATMQ)) deallocate(qm_gho_info_l%QMATMQ,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho_info','QMATMQ')
  if(associated(qm_gho_info_l%BT))     deallocate(qm_gho_info_l%BT,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho_info','BT')
  if(associated(qm_gho_info_l%BTM))    deallocate(qm_gho_info_l%BTM,stat=ier)
     if(ier.ne.0) call Aass(0,'allocate_deallocate_gho_info','BTM')
  if(associated(qm_gho_info_l%DBTMMM)) deallocate(qm_gho_info_l%DBTMMM,stat=ier)
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


  SUBROUTINE GHOHYB(natom,ISL,LSL,NBOND,IB,JB,cggho,X,Y,Z,CG,qfail,qgho)
  !
  ! prepare for GHO setup, initial setup
  !
  use stream

  implicit none

  ! passed in
  integer, intent(in)    :: natom,ISL(*),LSL(*),NBOND,IB(*),JB(*)
  real(chm_real),intent(in)   :: X(*),Y(*),Z(*)
  real(chm_real),intent(inout):: CG(*)
  real(chm_real),intent(inout):: cggho        ! charge on GHO atoms to be used in QM/MM-Ewald sum
  logical, intent(inout) :: qfail,qgho
  !
  ! local variables
  integer :: ier=0
  integer :: i,j,ii,jj,ibnd,itst,ngho,ngho2

  !
  ! put qm_gho_info_r%numat to common block to used by GHO routine later
  qm_gho_info_r%nqmlnk = 0
  qm_gho_info_r%numat  = 0

  ! loop over Natom to check
  do i=1,natom
     if((ISL(i).eq.1) .and. (LSL(i).ne.1)) then           ! for pure QM atom
        qm_gho_info_r%numat  = qm_gho_info_r%numat + 1
     else if((ISL(i).eq.1) .and. (LSL(i).eq.1)) then      ! for GHO atom 
        qm_gho_info_r%nqmlnk = qm_gho_info_r%nqmlnk + 1
        qm_gho_info_r%numat  = qm_gho_info_r%numat + 1
     end if
  end do

  if(qm_gho_info_r%nqmlnk.gt.0) then
     ! allocate some relevant arrays.
     call allocate_deallocate_gho_info(qm_gho_info_r,.true.)

     ! loop over again and fill qm_gho_info_r%IQLINK array
     ii=0
     do i = 1, natom
        if( (ISL(i).eq.1) .and. (LSL(i).eq.1) ) then
           ii=ii+1
           qm_gho_info_r%IQLINK(ii) = i 
        end if
     end do

     ! check the connectivity of GHO boundary atoms to MM and QM fragment
     do i=1,qm_gho_info_r%nqmlnk
        ibnd = 0
        itst = 0
        do j = 1,NBOND
           ii=IB(J)
           jj=JB(J)
           if(II .eq. qm_gho_info_r%IQLINK(i)) then              ! II is gho atom
              if(ISL(jj) .gt. 0) then                       ! jj is qm atom
                 itst = itst + 1
                 if(itst.gt.1) then
                    if(prnlev.ge.2) write(6,*) 'GHOHYB> Too many QM atoms connected to the GHO boundary atom'
                    qfail=.false.
                    return
                 end if
                 qm_gho_info_r%KQLINK(i) = jj
              else                                          ! jj is mm atom
                 ibnd = ibnd + 1
                 if(ibnd.gt.3) then
                    if(prnlev.ge.2) write(6,*) 'GHOHYB> Too many MM bonds connecting the GHO boundary atom'
                    qfail=.false.
                    return
                 end if
                 qm_gho_info_r%JQLINK(ibnd,i) = jj
              end if
           else if(JJ .eq. qm_gho_info_r%IQLINK(i)) then         ! JJ is gho atom
              if(ISL(ii) .gt. 0) then                       ! ii is qm atom
                 itst = itst + 1
                 if(itst.gt.1) then
                    if(prnlev.ge.2) write(6,*) 'GHOHYB> Too many QM atoms connected to the GHO boundary atom'
                    qfail=.false.
                    return
                 end if
                 qm_gho_info_r%KQLINK(i) = ii
              else                                          ! ii is mm atom
                 ibnd = ibnd + 1
                 if(ibnd.gt.3) then
                    if(prnlev.ge.2) write(6,*) 'GHOHYB> Too many MM bonds connecting the GHO boundary atom'
                    qfail=.false.
                    return
                 end if
                 qm_gho_info_r%JQLINK(ibnd,i) = ii
              end if
           end if
        end do
     end do

     ! determine core potentials, record QM-link atom charges for
     ! auxiliary density, and then zero MM charge on QM-link atom
     cggho = zero
     do i=1,qm_gho_info_r%nqmlnk
        qm_gho_info_r%QMATMQ(i)     = cg(qm_gho_info_r%IQLINK(i))
        cg(qm_gho_info_r%IQLINK(i)) = zero
        cggho                       = cggho+qm_gho_info_r%QMATMQ(i)
     end do

     ! Define hybrid orbital transformation matrix
     call HBDEF(X,Y,Z,qm_gho_info_r%BT,qm_gho_info_r%BTM,qm_gho_info_r%DBTMMM, &
                qm_gho_info_r%nqmlnk,qm_gho_info_r%mqm16,                      &
                qm_gho_info_r%IQLINK,qm_gho_info_r%JQLINK,qm_gho_info_r%KQLINK,&
                qfail)

  else
     ! no atoms selected...
     if(prnlev.ge.2) write(6,*) 'GHOHYB> No GHO atoms selected, GHO will not be used.'
     qgho =.false.
  end if

  return
  END SUBROUTINE GHOHYB


  SUBROUTINE HBDEF(X,Y,Z,BT,BTM,DBTMMM,NATVB,mqm16,IATVB,JATVB,KATVB,QFail)
  ! 
  ! Define transformation matrix
  !
  use stream

  implicit none

  ! Passed in
  real(chm_real), intent(in)   :: X(*),Y(*),Z(*)
  real(chm_real), intent(out)  :: BT(*),BTM(*),DBTMMM(3,3,*)
  integer,intent(in)     :: IATVB(*),JATVB(3,*),KATVB(*)
  integer,intent(in)     :: NATVB,mqm16
  logical,intent(inout)  :: QFail

  !   Local variables
  integer                :: i,ii,jq,j1,j2,j3,ni
  real(chm_real)         :: A(3),B(3),C(3),AB(3),AC(3),P(3),T(3),xyz_mm(3)

  ni = 0
  do i = 1,NATVB 
     ! each gho atom is connected to 1 qm and 3 mm atoms.
     ii       =IATVB(i)             ! gho atom
     xyz_mm(1)=x(ii)
     xyz_mm(2)=y(ii)
     xyz_mm(3)=z(ii)

     j1   = JATVB(1,i)              ! MM atoms
     a(1) = x(j1)-xyz_mm(1)
     a(2) = y(j1)-xyz_mm(2)
     a(3) = z(j1)-xyz_mm(3)

     j2   = JATVB(2,i)
     b(1) = x(j2)-xyz_mm(1)
     b(2) = y(j2)-xyz_mm(2)
     b(3) = z(j2)-xyz_mm(3)

     j3   = JATVB(3,i)
     c(1) = x(j3)-xyz_mm(1)
     c(2) = y(j3)-xyz_mm(2)
     c(3) = z(j3)-xyz_mm(3)

     jq   = KATVB(i)                ! QM atom
     t(1) = x(jq)-xyz_mm(1)
     t(2) = y(jq)-xyz_mm(2)
     t(3) = z(jq)-xyz_mm(3)

     p(1:3) = zero
     call HBDRIV(a,b,c,t,p,BT(ni+1:ni+mqm16),BTM(ni+1:ni+mqm16),DBTMMM(1:3,1:3,ni+1:ni+mqm16))

     if(p(1).ne.zero) then
        if(prnlev.ge.2) write(6,*) 'HBDEF> HYBRID ORBITAL ILLDEFINED.'
        qfail=.FALSE.
        return
     end if

     ni = ni+mqm16
  end do

  return

  contains
     !==================================================================
     subroutine HBDRIV(A,B,C,D,O,BT,BTM,DBTMMM)
     !
     !
     !
     implicit none

     ! Passed in
     real(chm_real), intent(in)   :: A(3),B(3),C(3),D(3)
     real(chm_real), intent(inout):: O(3)
     real(chm_real), intent(out)  :: BT(4,4),BTM(4,4),DBTMMM(3,3,16)

     ! Local variables
     integer         :: I,J,K,N,IJ
     real(chm_real)  :: AA(3),BB(3),CC(3),U(3),V(3),T(4,4), &
                        X(3),Y(3),Z(3),TETH(4,4),           &
                        xx(3),ab(3),bc(3),ca(3)
     real(chm_real)  :: dthma(3,4,4),dthmb(3,4,4),dthmc(3,4,4), &
                        DBTMM(4,4),DBTMMB(4,4),DBTMMC(4,4)
     real(chm_real)  :: dxa(3,3),dxb(3,3),dxc(3,3),dya(3,3),dyb(3,3),dyc(3,3),dza(3,3), &
                        dzb(3,3),dzc(3,3),                                              & 
                        daa(3,3),dd1a(3),dd1b(3),dd1c(3),                               & 
                        drxa(3),drxb(3),drxc(3),  drza(3),drzb(3),drzc(3),              &
                        dcsa(3),dcsb(3),dcsc(3),  dcpa(3),dcpb(3),dcpc(3)
     real(chm_real)  :: GRADA(4,4,3),GRADB(4,4,3),GRADC(4,4,3)
     real(chm_real)  :: DD,RA,RB,RC,CS,PFAC,RX,RY,RZ,RA2,RB2,RC2,CS2, &
                        THRFAC,RBA,RAC,RBC,D0,rtemp(3),tmp_sum

     integer, parameter :: IR(3,3)=reshape( (/ 0,3,2, 3,0,1, 2,1,0 /),(/3,3/))
     real(chm_real),parameter :: AF(3,3)=reshape( (/ 0.0D0, 1.0D0,-1.0D0,   &
                                                    -1.0D0, 0.0D0, 1.0D0,   &
                                                     1.0D0,-1.0D0, 0.0D0 /),(/3,3/))
     real(chm_real),parameter :: r_three=one/three

     !
     ! data teth/0.50,0.866,0.0,0.0,0.5,-0.2887,0.8165,0.0,  &
     !           0.5,-0.2887,-0.4082,0.7071, 0.5,-0.2887,-0.4082,-0.7071/
     !

     ra2 = a(1)**2+a(2)**2+a(3)**2
     rb2 = b(1)**2+b(2)**2+b(3)**2
     rc2 = c(1)**2+c(2)**2+c(3)**2
     ra = sqrt(ra2)
     rb = sqrt(rb2)
     rc = sqrt(rc2)
  
     ! do some initialization
     grada=zero
     gradb=zero
     gradc=zero

     rtemp(1)=one/ra
     rtemp(2)=one/rb
     rtemp(3)=one/rc

     aa(1:3) = a(1:3)*rtemp(1)
     bb(1:3) = b(1:3)*rtemp(2)
     cc(1:3) = c(1:3)*rtemp(3)
     u(1:3)  = bb(1:3)-aa(1:3)
     v(1:3)  = cc(1:3)-aa(1:3)

     call acb(u,v,x,rx)
 
     d0 = (aa(1)*x(1)+aa(2)*x(2)+aa(3)*x(3))/rx
     dd  = abs(d0)
     pfac = one
     if(d0 .GT. zero) pfac = -one

     ! tetrahedarl hybrid orbitals:
     cs = sqrt(dd / (one+dd) )
     cs2 = sqrt( (one - cs**2) * r_three )

     do i = 1,4
        teth(1,i) = cs2
        teth(2:4,i)=zero
     end do

     teth(1,1) =  cs

     teth(2,1) =  sqrt(one-teth(1,1)**2)

     teth(2,2) = -teth(1,1)*teth(1,2)/teth(2,1)
     teth(3,2) =  sqrt(one-teth(1,2)**2-teth(2,2)**2)

     teth(2,3) = -teth(1,1)*teth(1,3)/teth(2,1)
     teth(3,3) =-(teth(1,2)*teth(1,3)+teth(2,2)*teth(2,3))/teth(3,2)
     teth(4,3) =  sqrt(one-teth(1,3)**2-teth(2,3)**2-teth(3,3)**2)

     teth(2,4) = -teth(1,1)*teth(1,4)/teth(2,1)
     teth(3,4) =-(teth(1,2)*teth(1,4)+teth(2,2)*teth(2,4))/teth(3,2)
     teth(4,4) =-(teth(1,3)*teth(1,4)+teth(2,3)*teth(2,4)+teth(3,3)*teth(3,4))/teth(4,3)

     x(1:3) = pfac*x(1:3)
     call acb(x,aa,z,rz)
     call acb(z,x,y,ry)

     do i=1,4
        t(1,i) = zero
        t(i,1) = zero
     end do
     t(1,1) = one

     rtemp(1)=one/rx
     rtemp(2)=one/ry
     rtemp(3)=one/rz

     t(2:4,2) = x(1:3)*rtemp(1)
     t(2:4,3) = y(1:3)*rtemp(2)
     t(2:4,4) = z(1:3)*rtemp(3)

     call acb(bb,cc,bc,rbc)
     call acb(cc,aa,ca,rac)
     call acb(aa,bb,ab,rba)

     ! dai
     rtemp(1)=one/ra
     rtemp(2)=one/rb
     rtemp(3)=one/rc
     do i = 1,3
        ! dxj
        do j = 1,3
           ! dxj/dai
           dxa(j,i) = -aa(i)*(ab(j)+ca(j))*rtemp(1)
           dxb(j,i) = -bb(i)*(bc(j)+ab(j))*rtemp(2)
           dxc(j,i) = -cc(i)*(ca(j)+bc(j))*rtemp(3)
           if(j.ne.i) then
              dxa(j,i) = dxa(j,i)+af(j,i)*( cc(ir(j,i))-bb(ir(j,i)) )*rtemp(1)
              dxb(j,i) = dxb(j,i)+af(j,i)*( aa(ir(j,i))-cc(ir(j,i)) )*rtemp(2)
              dxc(j,i) = dxc(j,i)+af(j,i)*( bb(ir(j,i))-aa(ir(j,i)) )*rtemp(3)
           end if
        end do
     end do
     if(pfac.eq.one) then
        dxa=-dxa
        dxb=-dxb
        dxc=-dxc
     end if

     ! (Rx^2)'
     do i=1,3
        drxa(i) = two*(x(1)*dxa(1,i)+x(2)*dxa(2,i)+x(3)*dxa(3,i))
        drxb(i) = two*(x(1)*dxb(1,i)+x(2)*dxb(2,i)+x(3)*dxb(3,i))
        drxc(i) = two*(x(1)*dxc(1,i)+x(2)*dxc(2,i)+x(3)*dxc(3,i))
     end do

     ! dxj/dmi
     rtemp(1)=one/rx
     rtemp(2)=one/(rx**3)
     do i = 1,3
        grada(2:4,2,i) = dxa(1:3,i)*rtemp(1) - half*x(1:3)*drxa(i)*rtemp(2)
        gradb(2:4,2,i) = dxb(1:3,i)*rtemp(1) - half*x(1:3)*drxb(i)*rtemp(2)
        gradc(2:4,2,i) = dxc(1:3,i)*rtemp(1) - half*x(1:3)*drxc(i)*rtemp(2)
     end do
              
     ! daaj/dai
     rtemp(1)=one/ra
     do i = 1,3
        daa(1:3,i) = -aa(1:3)*aa(i)*rtemp(1)
        daa(i,i)   =  daa(i,i)+rtemp(1)
     end do

     do i=1,3
        dza(1,i) = dxa(2,i)*aa(3)+x(2)*daa(3,i)-daa(2,i)*x(3)-aa(2)*dxa(3,i)
        dza(2,i) = dxa(3,i)*aa(1)+x(3)*daa(1,i)-daa(3,i)*x(1)-aa(3)*dxa(1,i)
        dza(3,i) = dxa(1,i)*aa(2)+x(1)*daa(2,i)-daa(1,i)*x(2)-aa(1)*dxa(2,i)
 
        dzb(1,i) = dxb(2,i)*aa(3)-aa(2)*dxb(3,i)
        dzb(2,i) = dxb(3,i)*aa(1)-aa(3)*dxb(1,i)
        dzb(3,i) = dxb(1,i)*aa(2)-aa(1)*dxb(2,i)

        dzc(1,i) = dxc(2,i)*aa(3)-aa(2)*dxc(3,i)
        dzc(2,i) = dxc(3,i)*aa(1)-aa(3)*dxc(1,i)
        dzc(3,i) = dxc(1,i)*aa(2)-aa(1)*dxc(2,i)
     end do

     do i=1,3
        ! (Rz^2)'
        drza(i) = two*(z(1)*dza(1,i)+z(2)*dza(2,i)+z(3)*dza(3,i))
        drzb(i) = two*(z(1)*dzb(1,i)+z(2)*dzb(2,i)+z(3)*dzb(3,i))
        drzc(i) = two*(z(1)*dzc(1,i)+z(2)*dzc(2,i)+z(3)*dzc(3,i))
     end do
     
     ! dzj/dmi
     rtemp(1)=one/rz
     rtemp(2)=one/(rz**3)
     do i = 1,3
        grada(2:4,4,i) = dza(1:3,i)*rtemp(1) - half*z(1:3)*drza(i)*rtemp(2)
        gradb(2:4,4,i) = dzb(1:3,i)*rtemp(1) - half*z(1:3)*drzb(i)*rtemp(2)
        gradc(2:4,4,i) = dzc(1:3,i)*rtemp(1) - half*z(1:3)*drzc(i)*rtemp(2)
     end do
 
     do i=1,3
        dya(1,i)= dza(2,i)*x(3)+z(2)*dxa(3,i) - dxa(2,i)*z(3)-x(2)*dza(3,i)
        dya(2,i)= dza(3,i)*x(1)+z(3)*dxa(1,i) - dxa(3,i)*z(1)-x(3)*dza(1,i)
        dya(3,i)= dza(1,i)*x(2)+z(1)*dxa(2,i) - dxa(1,i)*z(2)-x(1)*dza(2,i)

        dyb(1,i)= dzb(2,i)*x(3)+z(2)*dxb(3,i) - dxb(2,i)*z(3)-x(2)*dzb(3,i)
        dyb(2,i)= dzb(3,i)*x(1)+z(3)*dxb(1,i) - dxb(3,i)*z(1)-x(3)*dzb(1,i)
        dyb(3,i)= dzb(1,i)*x(2)+z(1)*dxb(2,i) - dxb(1,i)*z(2)-x(1)*dzb(2,i)

        dyc(1,i)= dzc(2,i)*x(3)+z(2)*dxc(3,i) - dxc(2,i)*z(3)-x(2)*dzc(3,i)
        dyc(2,i)= dzc(3,i)*x(1)+z(3)*dxc(1,i) - dxc(3,i)*z(1)-x(3)*dzc(1,i)
        dyc(3,i)= dzc(1,i)*x(2)+z(1)*dxc(2,i) - dxc(1,i)*z(2)-x(1)*dzc(2,i)
     end do

     ! dyj/dmi
     rtemp(1)=one/ry
     rtemp(2)=one/(rz**2)
     rtemp(3)=one/(rx**2)
     do i = 1,3
        grada(2:4,3,i)=( dya(1:3,i)-half*y(1:3)*(drza(i)*rtemp(2)+drxa(i)*rtemp(3)) )*rtemp(1)
        gradb(2:4,3,i)=( dyb(1:3,i)-half*y(1:3)*(drzb(i)*rtemp(2)+drxb(i)*rtemp(3)) )*rtemp(1)
        gradc(2:4,3,i)=( dyc(1:3,i)-half*y(1:3)*(drzc(i)*rtemp(2)+drxc(i)*rtemp(3)) )*rtemp(1)
     end do

     ! d d/dmi
     ! initialization
     dthma = zero
     dthmb = zero
     dthmc = zero

     ! MJF . signs have been changed here!
     rtemp(1)=one/rx
     rtemp(2)=one/(rx**3)
     tmp_sum = x(1)*aa(1)+x(2)*aa(2)+x(3)*aa(3)
     do i=1,3
        dd1a(i) = ( dxa(1,i)*aa(1)+dxa(2,i)*aa(2)+dxa(3,i)*aa(3)            &
                   -daa(1,i)*x(1) -daa(2,i)*x(2) -daa(3,i)*x(3)  )*rtemp(1) &
                 - half*tmp_sum*drxa(i)*rtemp(2)
        dd1b(i) = ( dxb(1,i)*aa(1)+dxb(2,i)*aa(2)+dxb(3,i)*aa(3) )*rtemp(1) &
                 - half*tmp_sum*drxb(i)*rtemp(2)
        dd1c(i) = ( dxc(1,i)*aa(1)+dxc(2,i)*aa(2)+dxc(3,i)*aa(3) )*rtemp(1) &
                 - half*tmp_sum*drxc(i)*rtemp(2)
     end do

     rtemp(1) =(one/teth(1,1)-teth(1,1))*(teth(2,1)**2)
     dcsa(1:3)=-half*rtemp(1)*dd1a(1:3)
     dcsb(1:3)=-half*rtemp(1)*dd1b(1:3)
     dcsc(1:3)=-half*rtemp(1)*dd1c(1:3)
     dcpa(1:3)= half*dd1a(1:3)*teth(2,1)**3
     dcpb(1:3)= half*dd1b(1:3)*teth(2,1)**3
     dcpc(1:3)= half*dd1c(1:3)*teth(2,1)**3

     !
     thrfac = one/sqrt(three)

     dthma(1:3,1,1) = dcsa(1:3)
     dthma(1:3,2,1) = dcpa(1:3)
     dthma(1:3,1,2) = dcpa(1:3)*thrfac
     dthma(1:3,1,3) = dcpa(1:3)*thrfac
     dthma(1:3,1,4) = dcpa(1:3)*thrfac
     dthma(1:3,2,2) =-dcsa(1:3)*thrfac
     dthma(1:3,2,3) =-dcsa(1:3)*thrfac
     dthma(1:3,2,4) =-dcsa(1:3)*thrfac

     dthmb(1:3,1,1) = dcsb(1:3)
     dthmb(1:3,2,1) = dcpb(1:3)
     dthmb(1:3,1,2) = dcpb(1:3)*thrfac
     dthmb(1:3,1,3) = dcpb(1:3)*thrfac
     dthmb(1:3,1,4) = dcpb(1:3)*thrfac
     dthmb(1:3,2,2) =-dcsb(1:3)*thrfac
     dthmb(1:3,2,3) =-dcsb(1:3)*thrfac
     dthmb(1:3,2,4) =-dcsb(1:3)*thrfac

     dthmc(1:3,1,1) = dcsc(1:3)
     dthmc(1:3,2,1) = dcpc(1:3)
     dthmc(1:3,1,2) = dcpc(1:3)*thrfac
     dthmc(1:3,1,3) = dcpc(1:3)*thrfac
     dthmc(1:3,1,4) = dcpc(1:3)*thrfac
     dthmc(1:3,2,2) =-dcsc(1:3)*thrfac
     dthmc(1:3,2,3) =-dcsc(1:3)*thrfac
     dthmc(1:3,2,4) =-dcsc(1:3)*thrfac

     ! collect
     do j=1,4
        do i=1,4
           bt(i,j) = zero
           do k=1,4
              bt(i,j) = bt(i,j)+t(i,k)*teth(k,j)
           end do
           btm(j,i)=bt(i,j)
        end do
     end do

     ! derivatives
     do n=1,3
        do j=1,4
           do i=1,4
              dbtmm(i,j) = zero
              dbtmmb(i,j)= zero
              dbtmmc(i,j)= zero
              ! MJF . Lines have been uncommented!
              do k = 1,4
                 dbtmm(i,j)  = dbtmm(i,j)  + grada(i,k,n)*teth(k,j) + t(i,k)*dthma(n,k,j)
                 dbtmmb(i,j) = dbtmmb(i,j) + gradb(i,k,n)*teth(k,j) + t(i,k)*dthmb(n,k,j)
                 dbtmmc(i,j) = dbtmmc(i,j) + gradc(i,k,n)*teth(k,j) + t(i,k)*dthmc(n,k,j)
              end do
           end do
        end do
 
        !
        ij=0
        do i = 1,4
           do j = 1,4
              ij=ij+1
              dbtmmm(n,1,ij) = dbtmm(i,j)
              dbtmmm(n,2,ij) = dbtmmb(i,j)
              dbtmmm(n,3,ij) = dbtmmc(i,j)
           end do
        end do
     end do
     return
     end subroutine HBDRIV
     !==================================================================

     subroutine acb(a,b,c,S)
     implicit none
     real(chm_real), intent(in)  :: a(3),b(3)
     real(chm_real), intent(out) :: c(3),S

     c(1) = a(2)*b(3)-b(2)*a(3)
     c(2) = b(1)*a(3)-a(1)*b(3)
     c(3) = a(1)*b(2)-b(1)*a(2)
     S    = SQRT(c(1)*c(1)+c(2)*c(2)+c(3)*c(3))

     return
     end subroutine acb
     !==================================================================

  END SUBROUTINE HBDEF


  SUBROUTINE FTOFHB(F,FHB,BT,numat,nqmlnk,norbs, &
                    norbao,lin_norbao,nactatm,nfirst,nlast, &
                    indx)
  !
  ! Tansform a full Fock matrix in AO basis into active HO basis
  ! 
  ! On input
  !    F       : Fock matrix in AO, lower triangle
  !    BT      : Transformation matrix for each GHO boundary atom, 
  !              (4x4,Nqmlnk)
  !    numat   : Number of QM atoms
  !    nqmlnk  : Number of GHO boundary atoms
  !    norbs   : Number of AOs
  !    nfirst  : location of the start of ith orbital
  !    nlast   : location of the last of ith orbital
  !
  ! On output
  !    FHB     : Fock matrix in HO, include only active orbitals
  !
  use chm_kinds

  implicit none

  ! Passed in
  integer, intent(in)    :: numat,nqmlnk,norbs,norbao,lin_norbao,nactatm
  integer, intent(in)    :: nfirst(*),nlast(*),indx(*)
  real(chm_real),intent(in)    :: F(*),Bt(16,*)
  real(chm_real),intent(inout) :: Fhb(*)

  ! Local variables
  integer                :: i,j,k,ii,jj,ij,ia,ib,ja,jb,I1
  integer                :: L,IAMONE,INDF
  real(chm_real)         :: FTMP(10),FTMP2(10)

  ! Core part not affected
  FHB(1:lin_norbao) = F(1:lin_norbao)

  ! Loop over GHO boundary atoms for orbitals to be transformed
  do i=1,nqmlnk
     i1     = indx(norbao+i)
     ii     = nactatm+i
     ia     = nfirst(ii)
     ib     = nlast(ii)
     IAMONE = ia-1
     ij     = i1

     do j=1,norbao                           ! F(mu,l), AO-HO block
        ij = ij + 1                          ! Only one active HO 
                                             ! per QM-boundary atom
        FHB(ij) = zero
        do k=ia,ib
           FHB(ij) = FHB(ij)+BT(k-ia+1,i)*F(j+indx(k))
        end do
     end do

     do j=1,i-1                         ! F(l,l'), HO-other HO block
        ja      = nfirst(nactatm+j)
        ij      = ij +1
        FHB(ij) = zero
        do L=1,4
           ftmp(L)=zero
           indf   =ja-1+indx(ia+L-1)
           do k=1,4
              ftmp(L) = ftmp(L)+BT(k,j)*F(indf+K)
           end do

           FHB(ij) = FHB(ij) + BT(L,i)*ftmp(L)
        end do
     end do

     L = 0                              ! F(l,l), HO-HO corner block
     do j=ia,ib
        ja = indx(j)
        do k = ia, j
           L = L+1
           ftmp(L)  = F(k+ja)
           ftmp2(L) = zero
        end do
     end do

     FHB(ij+1) = VBFTN(ftmp,ftmp2,BT(1:16,I),4)
  end do
 
  return
  END SUBROUTINE FTOFHB


  SUBROUTINE FTOFHB_cpmd(F,FHB,BT,numat,nqmlnk,norbs, &
                         norbao,lin_norbao,nactatm,nfirst,nlast, &
                         indx)
  !
  ! Tansform a full Fock matrix in AO basis into active HO basis
  !
  ! On input
  !    F       : Fock matrix in AO, lower triangle
  !    BT      : Transformation matrix for each GHO boundary atom,
  !              (4x4,Nqmlnk)
  !    numat   : Number of QM atoms
  !    nqmlnk  : Number of GHO boundary atoms
  !    norbs   : Number of AOs
  !    nfirst  : location of the start of ith orbital
  !    nlast   : location of the last of ith orbital
  !
  ! On output
  !    FHB     : Fock matrix in HO, include only active orbitals
  !
  use chm_kinds
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none

  ! Passed in
  integer, intent(in)    :: numat,nqmlnk,norbs,norbao,lin_norbao,nactatm
  integer, intent(in)    :: nfirst(*),nlast(*),indx(*)
  real(chm_real),intent(in)    :: F(*),Bt(16,*)
  real(chm_real),intent(inout) :: Fhb(*)

  ! Local variables
  integer                :: i,j,k,ii,jj,ij,ia,ib,ja,jb,I1
  integer                :: L,IAMONE,INDF
  real(chm_real)         :: FTMP(10),FTMP2(10)
#if KEY_PARALLEL==1
  integer, save :: nnumnod
  logical, save :: q_first=.true.
  logical, pointer, save :: q_do_this(:)=>Null()

  if(q_first) then
     nnumnod = numnod
     if(associated(q_do_this))      deallocate(q_do_this)
     if(.not.associated(q_do_this)) allocate(q_do_this(nqmlnk))

     q_do_this(1:nqmlnk) = .false.
     do i=mynod+1,nqmlnk,nnumnod
        q_do_this(i) =.true.
     end do
     q_first =.false.
  end if
#endif

  ! Core part not affected
  FHB(1:lin_norbao) = F(1:lin_norbao)

  ! Loop over GHO boundary atoms for orbitals to be transformed
  do i=1,nqmlnk
     i1     = indx(norbao+i)
     ii     = nactatm+i
     ia     = nfirst(ii)
     ib     = nlast(ii)
     IAMONE = ia-1
     ij     = i1
#if KEY_PARALLEL==1
     if(.not.q_do_this(i)) then
        FHB(ij+1:ij+norbao) = zero
        ij = ij + norbao
        do j=1,i-1
           ij      = ij +1
           FHB(ij) = zero
        end do
        FHB(ij+1) = zero
     else
#endif
        do j=1,norbao                           ! F(mu,l), AO-HO block
           ij = ij + 1                          ! Only one active HO
                                                ! per QM-boundary atom
           FHB(ij) = zero
           do k=ia,ib
              FHB(ij) = FHB(ij)+BT(k-ia+1,i)*F(j+indx(k))
           end do
        end do

        do j=1,i-1                         ! F(l,l'), HO-other HO block
           ja      = nfirst(nactatm+j)
           ij      = ij +1
           FHB(ij) = zero
           do L=1,4
              ftmp(L)=zero
              indf   =ja-1+indx(ia+L-1)
              do k=1,4
                 ftmp(L) = ftmp(L)+BT(k,j)*F(indf+K)
              end do

              FHB(ij) = FHB(ij) + BT(L,i)*ftmp(L)
           end do
        end do

        L = 0                              ! F(l,l), HO-HO corner block
        do j=ia,ib
           ja = indx(j)
           do k = ia, j
              L = L+1
              ftmp(L)  = F(k+ja)
              ftmp2(L) = zero
           end do
        end do

        FHB(ij+1) = VBFTN(ftmp,ftmp2,BT(1:16,I),4)
#if KEY_PARALLEL==1
     end if
#endif
  end do
#if KEY_PARALLEL==1
  if(nnumnod > 1) then
     ii = indx(norbao+1)+1
     jj = ij+1
     call gcomb(FHB(ii:jj),jj-ii+1)
  end if
#endif

  return
  END SUBROUTINE FTOFHB_cpmd


  real(chm_real) FUNCTION vbftn(F1,F1vb,X,ndim)
  !
  ! on return, F1vb(1) is returned as vbftn
  ! has to check, since we only need F1vb(1).
  !
  use chm_kinds
  implicit none

  integer, intent(in)    :: ndim
  real(chm_real),intent(in)    :: F1(*), X(ndim,ndim)
  real(chm_real),intent(inout) :: F1vb(*)

  ! Local variables
  integer        :: i,j,k,L,L1,L2
  real(chm_real) :: fac,x_i(ndim),x_j(ndim)

  L1=0
  loopII: do i=1,ndim
     x_i(1:ndim)=x(1:ndim,i)     ! below x(k,i)
     do j=1,i
        L1      = L1+1
        if(L1 > 1) exit loopII
        L2      = 0
        F1vb(L1)= zero
        x_j(1:ndim)=x(1:ndim,j)  ! below x(k,j)
        do k=1,ndim
           do L=1,k-1
              L2      = L2+1
              fac     = x_i(k)*x_j(L)+x_j(k)*x_i(L)  ! =x(k,i)*x(L,j)+x(k,j)*x(L,i)
              F1vb(L1)= F1vb(L1) + fac*F1(L2)
           end do
           L2      = L2+1
           fac     = x_i(k)*x_j(k)  ! =x(k,i)*X(k,j)
           F1vb(L1)= F1vb(L1) + fac*F1(L2)
        end do
     end do
  end do loopII
  vbftn = F1vb(1)
  return
  END FUNCTION vbftn


  SUBROUTINE GHO_expansion(norbhb,naos,linao,nqmlnk,lin_norbhb,dim_norbs, &
                           dim_linear_norbs,mqm16,PL,PM,PA,PB,            &
                           PAHB,PBHB,PAOLD,PBOLD,                         &
                           QMATMQ,BT,BTM,indx,UHF)
  !
  use chm_kinds
  use number, only : zero,one,two,three
  use qm1_constant, only : PT5

  implicit none
  !
  integer :: norbhb,naos,linao,nqmlnk,lin_norbhb,dim_norbs,dim_linear_norbs,mqm16
  integer :: indx(*)
  real(chm_real):: PL,PM
  real(chm_real):: PA(dim_linear_norbs),PB(dim_linear_norbs),         &
                   PAHB(dim_linear_norbs),PBHB(dim_linear_norbs),     &
                   PAOLD(dim_linear_norbs),PBOLD(dim_linear_norbs),   &
                   QMATMQ(nqmlnk),BT(nqmlnk*mqm16),BTM(nqmlnk*mqm16)
  logical :: UHF
  !
  ! local variables
  integer :: i,j,ii,jj,ij,K,L,I1,J1,KK1,KK2,KK3
  integer :: IQATM,IORB1B
  real(chm_real):: XBT1
  real(chm_real),parameter :: r_three=one/three


  ! done in cal_density_matrix routine.
  ! compute the maximum diagnal density change explicitly for GHO, 
  ! since the density difference calculated above is based on densities
  ! in different basis (does not affect the energy converge. 
  ! Here the bug fix make the density looks better), 0319PJ05
  !if (niter .gt. 1) then  ! niter is already greater than 1.
  PL = zero
  PM = zero
  if (UHF) then
     do i = 1, NORBHB
        j = indx(i)+i  ! = i*(i+1)/2
        PL = MAX(ABS(PAHB(j)-PAOLD(j)), PL)
        PM = MAX(ABS(PBHB(j)-PBOLD(j)), PM)
     end do
     PL = MAX(PL, PM)
  else
     do i = 1, NORBHB
        j = indx(i)+i  ! = i*(i+1)/2
        PL = MAX(ABS(PAHB(j)-PAOLD(j)), PL)
     end do
  end if
  !end if
  if(UHF) then
     do i = 1, lin_norbhb
        PAOLD(i) = PAHB(i)
        PBOLD(i) = PBHB(i)
     end do
  else
     PAOLD(1:lin_norbhb)=PAHB(1:lin_norbhb)  ! as PBHB is the same as PAHB.
  end if

  ! this call is moved to the end of the scf cycle, as
  ! CA is not explicitly used within scf cycle for gho.
  !call CTRASF(norbs,nqmlnk,16,norbhb,naos,BT,CAHB,CA)
  !if(UHF) call CTRASF(norbs,nqmlnk,16,norbhb,naos,BT,CBHB,CB)

  ! GHO expansion I: RHF total or UHF alpha density 
  ! Relocate the positions of active hybrid orbitals if
  ! there are more than one QM-boundary atom.
  do i = NORBHB,NAOS+2,-1
     iqatm = i-NAOS
     ii = indx(i)   ! i*(i-1)/2
     iorb1b = NAOS+4*(IQATM-1)
     jj = indx(iorb1b)+iorb1b ! IORB1B*(IORB1B+1)/2
     do j = 1,NAOS
        ii = ii+1
        jj = jj+1
        PAHB(jj) = PAHB(ii)
        PAHB(ii) = zero
     end do
     ! HB-HB blocks
     do j = 1,iqatm
        ii = ii+1
        jj = jj+1
        PAHB(jj) = PAHB(ii)
        PAHB(ii) = zero
        if(j.ne.iqatm) then
           PAHB(jj+1:jj+3) = zero
           jj = jj+3
        end if
     end do
     ! The rest three auxiliary orbitals
     do j = 2,4
        PAHB(jj+1:jj+IORB1B+j)=zero  ! CALL VZERO(PAHB(jj+1),IORB1B+j)
        jj       = jj+IORB1B+j
        PAHB(jj) =(one-QMATMQ(iqatm)*r_three)*PT5   ! /three & /two
     end do
  end do

  ! Auxiliary density for the first QM-boundary atom
  do i = NAOS+2,NAOS+4    ! NFIRST(IQLINK(1))+1,NLAST(IQLINK(1))
     jj             = indx(i)  ! i*(i-1)/2
     PAHB(jj+1:jj+i)= zero                   ! CALL VZERO(PAHB(jj+1),i)
     PAHB(jj+i)     =(one-QMATMQ(1)*r_three)*PT5
  end do
  !
  ! AO blocks...Not affected by orbital transformation
  PA(1:LINAO) = PAHB(1:LINAO)

  ij = LINAO
  ! Loop over QM-boundary atoms
  do k = 1,NQMLNK
     j1     = 16*(k-1)
     i1     = NAOS+4*(k-1)
     iorb1b = indx(i1)+i1   ! i1*(i1+1)/2
     ! FOR EACH BOUNDARY ATOM, THERE ARE FOUR AOs.
     ! BOUNDARY ATOMS MUST BE NON-HYDROGEN ATOMS...
     l = 0
     do i = 1,4
        ! SINCE ONLY ONE HYBRID-ORBITAL DENSITY ON THE BOUNDARY ATOM IS NON-ZERO
        ! SUM OVER ORBITALS IS NOT NEEDED
        XBT1            = BT(j1+i)                   ! BTM(J1+4*(I-1)+1)
        PA(ij+1:ij+NAOS)= PAHB(iorb1b+1:iorb1b+NAOS)*XBT1
        ij = ij+NAOS
        !   BOUNDARY ATOM-OTHER BOUNDARY ATOM BLOCKS
        do l = 1,k-1
           kk1 = 16*(l-1)
           kk3 = IORB1B+NAOS+4*(l-1)+1
           PA(ij+1:ij+4)=PAHB(kk3)*XBT1*BT(kk1+1:kk1+4)  ! BTM(KK1+4*(J-1)+1)
           ij  = ij+4
        end do
        !   BOUNDARY ATOM SELF BLOCK
        kk1 = 4*(i-1)+j1
        do j = 1,i
           ij     = ij+1
           kk2    = 4*(j-1)+j1
           PA(ij) = zero
           do L = 1,4
              kk3    = indx(i1+L)+i1+L  ! (i1+L)*(i1+L+1)/2
              PA(ij) = PA(ij)+PAHB(kk3)*BTM(kk1+L)*BTM(kk2+L)
           end do
        end do
     end do
  end do

  ! this is moved to the end of the scf cycle.
  ! store the for derivative calculation (density in HB N+4 basis)
  !if (UHF) then
  !   PHO(1:dim_linear_norbs)=PAHB(1:dim_linear_norbs)
  !else
  !   PHO(1:dim_linear_norbs)=PAHB(1:dim_linear_norbs)*two
  !end if

  ! GHO expansion II: UHF beta density
  if (UHF) then
     !   Relocate the positions of active hybrid orbitals if
     !   there are more than one QM-boundary atom.
     do i = NORBHB,NAOS+2,-1
        iqatm  = i-NAOS
        ii     = indx(i)  ! i*(i-1)/2
        iorb1b = NAOS+4*(iqatm-1)
        jj = indx(iorb1b)+iorb1b  ! IORB1B*(IORB1B+1)/2
        do j = 1,NAOS
           ii       = ii+1
           jj       = jj+1
           PBHB(jj) = PBHB(ii)
           PBHB(ii) = zero
        end do
        !   HB-HB blocks
        do j = 1,iqatm
           ii       = ii+1
           jj       = jj+1
           PBHB(jj) = PBHB(ii)
           PBHB(ii) = zero
           if(j.ne.iqatm) then
              PBHB(jj+1:jj+3) = zero
              jj              = jj+3
           end if
        end do
        ! The rest three auxiliary orbitals
        do j = 2,4
           PBHB(jj+1:jj+IORB1B+j)= zero  ! CALL VZERO(PBHB(jj+1),IORB1B+j)
           jj       = jj+IORB1B+j
           PBHB(jj) =(one-QMATMQ(iqatm)*r_three)*PT5
        end do
     end do
     !  Auxiliary density for the first QM-boundary atom
     do i = NAOS+2,NAOS+4    ! NFIRST(IQLINK(1))+1,NLAST(IQLINK(1))
        jj             = indx(i)  ! i*(i-1)/2
        PBHB(jj+1:jj+i)= zero                   ! CALL VZERO(PBHB(jj+1),i)
        PBHB(jj+i)     =(one-QMATMQ(1)*r_three)*PT5
     end do
     !
     ! AO blocks...Not affected by orbital transformation
     PB(1:LINAO) = PBHB(1:LINAO)

     ij = LINAO
     !  Loop over QM-boundary atoms
     do k = 1,NQMLNK
        j1     = 16*(k-1)
        i1     = NAOS+4*(k-1)
        iorb1b = indx(i1)+i1  ! i1*(i1+1)/2
        ! FOR EACH BOUNDARY ATOM, THERE ARE FOUR AOs.
        ! BOUNDARY ATOMS MUST BE NON-HYDROGEN ATOMS...
        l = 0
        do i = 1,4
           ! SINCE ONLY ONE HYBRID-ORBITAL DENSITY ON THE BOUNDARY ATOM IS NON-ZERO
           ! SUM OVER ORBITALS IS NOT NEEDED
           XBT1            = BT(j1+i)                   ! BTM(J1+4*(I-1)+1)
           PB(ij+1:ij+NAOS)= PBHB(iorb1b+1:iorb1b+NAOS)*XBT1
           ij = ij+NAOS
           !   BOUNDARY ATOM-OTHER BOUNDARY ATOM BLOCKS
           do l = 1,k-1
              kk1 = 16*(l-1)
              kk3 = IORB1B+NAOS+4*(l-1)+1
              PB(ij+1:ij+4)=PBHB(kk3)*XBT1*BT(kk1+1:kk1+4)  ! BTM(KK1+4*(J-1)+1)
              ij=ij+4
           end do
           !   BOUNDARY ATOM SELF BLOCK
           kk1 = 4*(i-1)+j1
           do j = 1,i
              ij  = ij+1
              kk2 = 4*(j-1)+j1
              PB(ij) = zero
              do l = 1,4
                 kk3    = indx(i1+l)+i1+l  ! (i1+l)*(i1+l+1)/2
                 PB(ij) = PB(ij)+PBHB(kk3)*BTM(kk1+l)*BTM(kk2+l)
              end do
           end do
        end do
     end do
     ! this is moved to the end of the scf cycle.
     ! store the for derivative calculation (density in HB N+4 basis)
     !PBHO(1:dim_linear_norbs)=PBHB(1:dim_linear_norbs)
  end if       ! UHF

  return
  END SUBROUTINE GHO_expansion


  SUBROUTINE GHO_expansion_cpmd(norbhb,naos,linao,nqmlnk,lin_norbhb,dim_norbs, &
                                dim_linear_norbs,mqm16,PA,PB,                  &
                                PAHB,PBHB,PAOLD,PBOLD,                         &
                                QMATMQ,BT,BTM,indx,UHF)
  !
  use chm_kinds
  use number, only : zero,one,two,three
  use qm1_constant, only : PT5
#if KEY_PARALLEL==1
  use parallel
#endif

  implicit none
  !
  integer :: norbhb,naos,linao,nqmlnk,lin_norbhb,dim_norbs,dim_linear_norbs,mqm16
  integer :: indx(*)
  real(chm_real):: PA(dim_linear_norbs),PB(dim_linear_norbs),         &
                   PAHB(dim_linear_norbs),PBHB(dim_linear_norbs),     &
                   PAOLD(dim_linear_norbs),PBOLD(dim_linear_norbs),   &
                   QMATMQ(nqmlnk),BT(nqmlnk*mqm16),BTM(nqmlnk*mqm16)
  logical :: UHF
  !
  ! local variables
  integer :: i,j,ii,jj,ij,K,L,I1,J1,KK1,KK2,KK3
  integer :: IQATM,IORB1B
  real(chm_real):: XBT1
  real(chm_real),parameter :: r_three=one/three

#if KEY_PARALLEL==1
  logical, save :: q_first=.true.
  logical, pointer, save :: q_do_this(:)=>Null()

  if(q_first) then
     if(associated(q_do_this))      deallocate(q_do_this)
     if(.not.associated(q_do_this)) allocate(q_do_this(NQMLNK))

     q_do_this(1:NQMLNK) = .false.
     do i=mynod+1,NQMLNK,numnod
        q_do_this(i) =.true.
     end do
     q_first =.false.
  end if
#endif

  ! if UHF, GHO_expansion should be used.

  ! This is a copy of subroutine GHO_expansion for cpmd run. So, refer that routine for details.
  PAOLD(1:lin_norbhb)=PAHB(1:lin_norbhb)  ! as PBHB is the same as PAHB.

  ! GHO expansion I: RHF total or UHF alpha density 
  ! Relocate the positions of active hybrid orbitals if
  ! there are more than one QM-boundary atom.
  loopii: do i = NORBHB,NAOS+2,-1
     iqatm = i-NAOS
#if KEY_PARALLEL==1
     if(.not.q_do_this(iqatm)) cycle loopii
#endif
     ii = indx(i)   ! i*(i-1)/2
     iorb1b = NAOS+4*(IQATM-1)
     jj = indx(iorb1b)+iorb1b ! IORB1B*(IORB1B+1)/2
     do j = 1,NAOS
        ii = ii+1
        jj = jj+1
        PAHB(jj) = PAHB(ii)
        PAHB(ii) = zero
     end do
     ! HB-HB blocks
     do j = 1,iqatm
        ii = ii+1
        jj = jj+1
        PAHB(jj) = PAHB(ii)
        PAHB(ii) = zero
        if(j.ne.iqatm) then
           PAHB(jj+1:jj+3) = zero
           jj = jj+3
        end if
     end do
     ! The rest three auxiliary orbitals
     do j = 2,4
        PAHB(jj+1:jj+IORB1B+j)=zero  ! CALL VZERO(PAHB(jj+1),IORB1B+j)
        jj       = jj+IORB1B+j
        PAHB(jj) =(one-QMATMQ(iqatm)*r_three)*PT5   ! /three & /two
     end do
  end do loopii

  ! Auxiliary density for the first QM-boundary atom
#if KEY_PARALLEL==1
  if(q_do_this(1)) then
#endif
     do i = NAOS+2,NAOS+4    ! NFIRST(IQLINK(1))+1,NLAST(IQLINK(1))
        jj             = indx(i)  ! i*(i-1)/2
        PAHB(jj+1:jj+i)= zero                   ! CALL VZERO(PAHB(jj+1),i)
        PAHB(jj+i)     =(one-QMATMQ(1)*r_three)*PT5
     end do
#if KEY_PARALLEL==1
  end if
#endif
  !
  ! AO blocks...Not affected by orbital transformation
  PA(1:LINAO) = PAHB(1:LINAO)
#if KEY_PARALLEL==1
  PA(LINAO+1:dim_linear_norbs) = zero
#endif

  ij = LINAO
  ! Loop over QM-boundary atoms
  loopkk: do k = 1,NQMLNK
#if KEY_PARALLEL==1
     if(.not.q_do_this(k)) then
        do i=1,4
           ij = ij+NAOS + 4*(k-1) + i
        end do
     else
#endif
        j1     = 16*(k-1)
        i1     = NAOS+4*(k-1)
        iorb1b = indx(i1)+i1   ! i1*(i1+1)/2
        ! FOR EACH BOUNDARY ATOM, THERE ARE FOUR AOs.
        ! BOUNDARY ATOMS MUST BE NON-HYDROGEN ATOMS...
        do i = 1,4
           ! SINCE ONLY ONE HYBRID-ORBITAL DENSITY ON THE BOUNDARY ATOM IS NON-ZERO
           ! SUM OVER ORBITALS IS NOT NEEDED
           XBT1            = BT(j1+i)                   ! BTM(J1+4*(I-1)+1)
           PA(ij+1:ij+NAOS)= PAHB(iorb1b+1:iorb1b+NAOS)*XBT1
           ij = ij+NAOS
           !   BOUNDARY ATOM-OTHER BOUNDARY ATOM BLOCKS
           do l = 1,k-1
              kk1 = 16*(l-1)
              kk3 = IORB1B+NAOS+4*(l-1)+1
              PA(ij+1:ij+4)=PAHB(kk3)*XBT1*BT(kk1+1:kk1+4)  ! BTM(KK1+4*(J-1)+1)
              ij  = ij+4
           end do
           !   BOUNDARY ATOM SELF BLOCK
           kk1 = 4*(i-1)+j1
           do j = 1,i
              ij     = ij+1
              kk2    = 4*(j-1)+j1
              PA(ij) = zero
              do L = 1,4
                 kk3    = indx(i1+L)+i1+L  ! (i1+L)*(i1+L+1)/2
                 PA(ij) = PA(ij)+PAHB(kk3)*BTM(kk1+L)*BTM(kk2+L)
              end do
           end do
        end do
#if KEY_PARALLEL==1
     end if
#endif
  end do loopkk
#if KEY_PARALLEL==1
  if(numnod>1) call gcomb(PA(LINAO+1:dim_linear_norbs),dim_linear_norbs-LINAO)
#endif
  return
  END SUBROUTINE GHO_expansion_cpmd


  SUBROUTINE CTRASF(norbs,nqmlnk,mqm16,norbhb,naos,BT,CHB,C)
  !
  implicit none

  ! Passed in
  integer, intent(in)    :: norbs,nqmlnk,mqm16,norbhb,naos
  real(chm_real),  intent(in)    :: BT(mqm16*nqmlnk), CHB(norbs*norbs)
  real(chm_real),  intent(inout) :: C(norbs*norbs)

  ! Local variables
  integer                :: i,j,i1,j1,ij,mqm15,ii,jj

  !
  mqm15  = mqm16-1

  do i=1,norbhb
     i1 = norbs*(i-1)
     j1 = norbhb*(i-1)

     ! Upto normal naos
     C(i1+1:i1+naos) = CHB(i1+1:i1+naos)

     i1 = i1+naos
     j1 = j1+naos

     do j=naos+1,norbhb
        ij = mqm16*(j-naos) - mqm15
        C(i1+1:i1+4)=CHB(j1+1:j1+4)*BT(ij)

        i1 = i1+4
        j1 = j1+4
     end do
  end do

  ! Append and transform auxiliary hybrid orbitals
  do i=1,nqmlnk
     ii=naos+(i-1)*4
     jj=naos+i*4
     do j=1,3
        ij = norbs*(norbhb+i*j-1)
        ij = mqm16*(i-1) + 4*j
        do j1 = 1,norbs
           i1 = i1+1
           if( j1.gt.ii .and. j1.le.jj ) then
              ij    = ij+1
              C(i1) = BT(ij)
           else
              C(i1) = zero
           end if
        end do
     end do
  end do

  return
  END SUBROUTINE CTRASF


  SUBROUTINE DQLINK(dx,dy,dz,X,Y,Z,CG,Fock,PHO,eclass,   &
                    nqmlnk,norbao,lin_norbao,mqm16,      &
                    indx,IQLINK,JQLINK,BT,BTM,DBTMMM,UHF)
  !
  ! Computes the derivatives of the density matrix as a result of 
  ! the transformation from hybrid orbitals to atomic orbitals
  ! basis. Updates nucleus force arrays, and computes a classical
  ! energy contribution from MM atoms directly connected to the
  ! QM boundary atom.
  !
  ! Note:
  ! In the 1st call from MNDENE, the PHO is PHO. However, in the 2nd call in case
  ! of UHF, PHO is PBHO. So, it can avoid using if(UHF) in the middle of the loop.
  !
  ! References:
  !   J. Gao, P. Amara, C. Alhambra, M. J. Field J. Phys. Chem. A, 
  !   102, 4714 (1998).
  !
  use qm1_constant
#if KEY_PARALLEL==1
  use parallel 
#endif

  implicit none

  ! Passed in
  real(chm_real),intent(inout) :: dx(*),dy(*),dz(*)
  real(chm_real),intent(in)    :: X(*),Y(*),Z(*),CG(*),Fock(*),PHO(*)
  real(chm_real),intent(inout) :: Eclass
  integer,intent(in)           :: nqmlnk,norbao,lin_norbao,mqm16
  integer,intent(in)           :: IQLINK(nqmlnk),JQLINK(3,nqmlnk),indx(*)
  real(chm_real),intent(in)    :: BT(nqmlnk*mqm16),BTM(nqmlnk*mqm16), &
                                  DBTMMM(3,3,nqmlnk*mqm16)
  logical, intent(in)          :: UHF

  ! Local variables
  integer :: i,j,k,L,m,n,ii,jj,LL,mm,i1,L1,m1,ij
  integer :: kk1,kk2,kk3,ii2,jj2,n16,n16i1,n16j
  integer :: Linao
  integer :: i1diag(4)

  real(chm_real) :: xdta(4,4),ydta(4,4),zdta(4,4), &
                    xdtb(4,4),ydtb(4,4),zdtb(4,4), &
                    xdtc(4,4),ydtc(4,4),zdtc(4,4), xyz(3),dxyz(3)
  real(chm_real) :: pf, pf1, pf2, xfac,rr1,rr2,fact1
  real(chm_real) :: tinv(4,4),dmmxyz(3,3),dmmxyz1(3,3),dL2xyz(3,3),dLtmp(3),delxyz(3)
  !
  integer :: nnumnod,mmynod

  ! for parallelization
#if KEY_PARALLEL==1
  mmynod  = mynod
  nnumnod = numnod
#else
  nnumnod = 1
  mmynod  = 0
#endif

  ! get the number of orbitals
  ij    = lin_norbao
  n16   = 0

  !*** main loop ***
  ! loop over GHO boundary atoms
  do i=1,nqmlnk   ! mmynod+1,nqmlnk,nnumnod     ! 1,nqmlnk as if not parallel, mmynod=0,nnumnod=1
#if KEY_PARALLEL==1
     if(mmynod .ne. mod(i-1,nnumnod)) then
        ! index counter
        do j=1,4
           ! AO-HB blocks
           ij =  ij + norbao

           ! HB-other HB blocks
           do i1 = 1, i-1
              ij = ij + 4
           end do

           ! HB-HB blocks
           ij = ij + j
        end do
        n16 = n16 + mqm16
     else
#endif
        L1    = 4*(i-1)
        ii    = IQLINK(i)
        k     = norbao + 4*(i-1) + 1   ! nfirst
        L     = k + 3                  ! nlast
        Linao = indx(k)

        ! locate position of diagonal elements
        do j=k,L
           i1diag(j-k+1) = indx(j)+j  ! j*(j+1)/2
        end do

        ! initialize temporary variables
        dmmxyz = zero
        n16j   = n16
        do j=1,4
           do k=1,4
              n16j        = n16j+1
              tinv(k,j) = BTM(n16j)
              xdta(k,j) = DBTMMM(1,1,n16j)
              ydta(k,j) = DBTMMM(2,1,n16j)
              zdta(k,j) = DBTMMM(3,1,n16j)
              xdtb(k,j) = DBTMMM(1,2,n16j)
              ydtb(k,j) = DBTMMM(2,2,n16j)
              zdtb(k,j) = DBTMMM(3,2,n16j)
              xdtc(k,j) = DBTMMM(1,3,n16j)
              ydtc(k,j) = DBTMMM(2,3,n16j)
              zdtc(k,j) = DBTMMM(3,3,n16j)
           end do
        end do

        ! AO-HB blocks
        do j=1,4
           do k=1,norbao
              ij = ij + 1
              pf = two * PHO(Linao+k) * Fock(ij)

              dmmxyz(1,1) = dmmxyz(1,1) + xdta(1,j)*pf         ! x axis
              dmmxyz(2,1) = dmmxyz(2,1) + xdtb(1,j)*pf
              dmmxyz(3,1) = dmmxyz(3,1) + xdtc(1,j)*pf

              dmmxyz(1,2) = dmmxyz(1,2) + ydta(1,j)*pf         ! y axis
              dmmxyz(2,2) = dmmxyz(2,2) + ydtb(1,j)*pf
              dmmxyz(3,2) = dmmxyz(3,2) + ydtc(1,j)*pf

              dmmxyz(1,3) = dmmxyz(1,3) + zdta(1,j)*pf         ! zaxis
              dmmxyz(2,3) = dmmxyz(2,3) + zdtb(1,j)*pf
              dmmxyz(3,3) = dmmxyz(3,3) + zdtc(1,j)*pf
           end do

           ! HB-other HB blocks
           m1 = Linao + norbao
           do i1 = 1, i-1
              n16i1  = mqm16 * (i1-1)
              dL2xyz = zero

              kk3 = m1 + 4*(i1-1) + 1
              do L1 = 1,4
                 ij = ij +1
                 pf = two * PHO(kk3) * Fock(ij)

                 pf1 = BT(n16+j)*pf
                 pf2 = BT(n16i1+L1)*pf

                 dmmxyz(1,1) = dmmxyz(1,1) + xdta(1,j)*pf2     ! x axis
                 dmmxyz(2,1) = dmmxyz(2,1) + xdtb(1,j)*pf2
                 dmmxyz(3,1) = dmmxyz(3,1) + xdtc(1,j)*pf2

                 dmmxyz(1,2) = dmmxyz(1,2) + ydta(1,j)*pf2     ! y axis
                 dmmxyz(2,2) = dmmxyz(2,2) + ydtb(1,j)*pf2
                 dmmxyz(3,2) = dmmxyz(3,2) + ydtc(1,j)*pf2

                 dmmxyz(1,3) = dmmxyz(1,3) + zdta(1,j)*pf2     ! z axis
                 dmmxyz(2,3) = dmmxyz(2,3) + zdtb(1,j)*pf2
                 dmmxyz(3,3) = dmmxyz(3,3) + zdtc(1,j)*pf2

                 kk1 = n16i1 + 4*(L1-1) + 1
                 ! x,y,z axis for a
                 dL2xyz(1:3,1) = dL2xyz(1:3,1) + pf1*DBTMMM(1:3,1,kk1)  
                 ! x,y,z axis for b
                 dL2xyz(1:3,2) = dL2xyz(1:3,2) + pf1*DBTMMM(1:3,2,kk1) 
                 ! x,y,z axis for c
                 dL2xyz(1:3,3) = dL2xyz(1:3,3) + pf1*DBTMMM(1:3,3,kk1)
              end do

              ii2   = IQLINK(i1)
              jj    = JQLINK(1,i1)
              LL    = JQLINK(2,i1)
              mm    = JQLINK(3,i1)

              dx(jj) = dx(jj) - dL2xyz(1,1)*EVCAL
              dy(jj) = dy(jj) - dL2xyz(2,1)*EVCAL
              dz(jj) = dz(jj) - dL2xyz(3,1)*EVCAL
              dLtmp(1:3) = dL2xyz(1:3,1)

              dx(LL) = dx(LL) - dL2xyz(1,2)*EVCAL
              dy(LL) = dy(LL) - dL2xyz(2,2)*EVCAL
              dz(LL) = dz(LL) - dL2xyz(3,2)*EVCAL
              dLtmp(1:3) = dLtmp(1:3) + dL2xyz(1:3,2)

              dx(mm) = dx(mm) - dL2xyz(1,3)*EVCAL
              dy(mm) = dy(mm) - dL2xyz(2,3)*EVCAL
              dz(mm) = dz(mm) - dL2xyz(3,3)*EVCAL
              dLtmp(1:3) = dLtmp(1:3) + dL2xyz(1:3,3)

              dLtmp  = dLtmp*EVCAL
              dx(ii2)= dx(ii2)+dLtmp(1)
              dy(ii2)= dy(ii2)+dLtmp(2)
              dz(ii2)= dz(ii2)+dLtmp(3)
           end do

           ! HB-HB blocks
           do k=1,j
              ij   = ij + 1
              if(k.eq.j) then
                 xfac = one
              else
                 xfac = two
              end if
              do L =1,4
                 pf = xfac * PHO(i1diag(L)) * Fock(ij)

                 dmmxyz(1,1)=dmmxyz(1,1)+pf*(xdta(L,j)*tinv(L,k)+tinv(L,j)*xdta(L,k))
                 dmmxyz(2,1)=dmmxyz(2,1)+pf*(xdtb(L,j)*tinv(L,k)+tinv(L,j)*xdtb(L,k))
                 dmmxyz(3,1)=dmmxyz(3,1)+pf*(xdtc(L,j)*tinv(L,k)+tinv(L,j)*xdtc(L,k))

                 dmmxyz(1,2)=dmmxyz(1,2)+pf*(ydta(L,j)*tinv(L,k)+tinv(L,j)*ydta(L,k))
                 dmmxyz(2,2)=dmmxyz(2,2)+pf*(ydtb(L,j)*tinv(L,k)+tinv(L,j)*ydtb(L,k))
                 dmmxyz(3,2)=dmmxyz(3,2)+pf*(ydtc(L,j)*tinv(L,k)+tinv(L,j)*ydtc(L,k))

                 dmmxyz(1,3)=dmmxyz(1,3)+pf*(zdta(L,j)*tinv(L,k)+tinv(L,j)*zdta(L,k))
                 dmmxyz(2,3)=dmmxyz(2,3)+pf*(zdtb(L,j)*tinv(L,k)+tinv(L,j)*zdtb(L,k))
                 dmmxyz(3,3)=dmmxyz(3,3)+pf*(zdtc(L,j)*tinv(L,k)+tinv(L,j)*zdtc(L,k))
              end do
           end do
        end do

        ! conversion factor
        dmmxyz1 = dmmxyz * EVCAL

        ! add gradient
        jj    = JQLINK(1,i)
        LL    = JQLINK(2,i)
        mm    = JQLINK(3,i)

        dx(jj)   = dx(jj) - dmmxyz1(1,1)
        dx(LL)   = dx(LL) - dmmxyz1(2,1)
        dx(mm)   = dx(mm) - dmmxyz1(3,1)
        dLtmp(1) = dmmxyz1(1,1)+dmmxyz1(2,1)+dmmxyz1(3,1)

        dy(jj)   = dy(jj) - dmmxyz1(1,2)
        dy(LL)   = dy(LL) - dmmxyz1(2,2)
        dy(mm)   = dy(mm) - dmmxyz1(3,2)
        dLtmp(2) = dmmxyz1(1,2)+dmmxyz1(2,2)+dmmxyz1(3,2)

        dz(jj)   = dz(jj) - dmmxyz1(1,3)
        dz(LL)   = dz(LL) - dmmxyz1(2,3)
        dz(mm)   = dz(mm) - dmmxyz1(3,3)
        dLtmp(3) = dmmxyz1(1,3)+dmmxyz1(2,3)+dmmxyz1(3,3)

        dx(ii)   = dx(ii) + dLtmp(1)
        dy(ii)   = dy(ii) + dLtmp(2)
        dz(ii)   = dz(ii) + dLtmp(3)

        n16 = n16 + mqm16
        !
        ! end of electron interactions


        ! 
        ! include nuclear interactions (ECLASS) between MM-boundary atoms
        !
        ! avoid double counting in Beta correction for UHF case
        if(.not.UHF) then
           do m = 1,2
              jj    = JQLINK(m,i)
              xyz(1)= x(jj)
              xyz(2)= y(jj)
              xyz(3)= z(jj)
              dxyz(1:3)=zero
              do n = m+1,3
                 jj2 = JQLINK(n,i)
                 delxyz(1) = x(jj2) - xyz(1)
                 delxyz(2) = y(jj2) - xyz(2)
                 delxyz(3) = z(jj2) - xyz(3)
                 rr2       = delxyz(1)**2+delxyz(2)**2+delxyz(3)**2
                 rr1       = SQRT(RR2)
                 !
                 fact1     = CCELEC*CG(jj)*CG(jj2) / rr1
                 ECLASS    = ECLASS + fact1
                 !
                 fact1      = fact1/rr2
                 dLtmp(1:3) = delxyz(1:3)*fact1
                 ! for jj
                 dxyz(1:3)  = dxyz(1:3)+dLtmp(1:3)
                 ! for jj2
                 dx(jj2)    = dx(jj2)- dLtmp(1)
                 dy(jj2)    = dy(jj2)- dLtmp(2)
                 dz(jj2)    = dz(jj2)- dLtmp(3)
              end do
              ! for jj
              dx(jj) = dx(jj)+dxyz(1)
              dy(jj) = dy(jj)+dxyz(2)
              dz(jj) = dz(jj)+dxyz(3)
           end do
        end if
#if KEY_PARALLEL==1
      end if
#endif
  end do
  !*** end main loop ***

  return
  END SUBROUTINE DQLINK


!  SUBROUTINE Get_Charge_GHO(numat_1,qminb1_dual,cginb,Qdual_check)
!  !
!  ! copy charge on gho atoms to cginb array.
!  !
!  use qmmm_module, only : qm2_ghos,qm2_ghos_r
!
!  implicit none
!
!  ! Passed in
!  integer, intent(in)    :: numat_1,qminb1_dual(*)
!  real(chm_real),intent(inout) :: cginb(*)
!  logical, intent(in)    :: Qdual_check(2)
!
!  ! Local variables
!  integer                :: i,j,ii,jj
!
!  ! for dual quantum region
!  if(Qdual_check(1)) then
!     Do i=1,qm2_ghos_r%nqmlnk
!        ii = qm2_ghos_r%IQLINK(i)
!        do j=1,numat_1
!           jj=iabs(qminb1_dual(j))
!           if(ii.eq.jj) cginb(j) = qm2_ghos_r%QMATMQ(i)
!        end do
!     End do
!  else 
!     Do i=1,qm2_ghos%nqmlnk
!        ii = qm2_ghos%IQLINK(i)
!        do j=1,numat_1
!           jj=iabs(qminb1_dual(j))
!           if(ii.eq.jj) cginb(j) = qm2_ghos%QMATMQ(i)
!        end do
!     End do
!  end if
!
!  return
!  END SUBROUTINE Get_Charge_GHO
!
!  SUBROUTINE Get_CG_GHO(CG,Qdual_check)
!  !
!  ! copy charge on gho atoms to CG array.
!  !
!  use qmmm_module, only : qm2_ghos,qm2_ghos_r,qm2_ghos_p
!
!  implicit none
!
!  ! Passed in
!  real(chm_real),intent(inout) :: CG(*)
!  logical, intent(in)    :: Qdual_check(2)
!
!  ! Local variables
!  integer                :: i,j,ii,jj
!
!  ! for dual quantum region
!  if(Qdual_check(1)) then
!     if(Qdual_check(2)) then             ! get 1st gho charge for 
!        Do i=1,qm2_ghos_r%nqmlnk         ! 2nd qm
!           ii     = qm2_ghos_r%IQLINK(i)
!           CG(ii) = qm2_ghos_r%QMATMQ(i)
!        End do
!     else
!        Do i=1,qm2_ghos_p%nqmlnk         ! get 2nd gho charge for
!           ii     = qm2_ghos_p%IQLINK(i) ! 1st qm
!           CG(ii) = qm2_ghos_p%QMATMQ(i)
!        End do
!     end if
!  else
!     Do i=1,qm2_ghos%nqmlnk
!        ii     = qm2_ghos%IQLINK(i)
!        CG(ii) = qm2_ghos%QMATMQ(i)
!     End do
!  end if
!
!  return
!  END SUBROUTINE Get_CG_GHO

#endif /*mndo97*/

end module mndgho_module
