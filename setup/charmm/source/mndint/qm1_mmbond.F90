! mm bonded energies for qm region.
module mmbonded_mod
  use chm_kinds
  use number


  ! common arrays.
  logical, save :: q_mts_mmbond=.false. ! to be used for MTS MNDO97-QM/MM calc.
                                        ! where mm-bonds/angles are solved at inner time step,
                                        ! where se-qm/mm is solved at outer time step.

  logical, save :: q_mmbond    =.false. ! to be used for correcting bond freq. for
                                        ! MTS AI-QM/MM calc. where mm-bonds are used for correcting
                                        ! se-qm/mm bond terms....

!!!#if KEY_MTS==1    /*mts*/
  integer, save :: n_replica=1 ! default, no. of QM mm region (for EVB)
                               ! e.g., rs, ps, int, etc
  logical, save :: do_mm_bond=.false., do_mm_angl=.false., &
                   do_mm_dihe=.false., do_mm_impr=.false., &
                   do_morse_mbond=.false.
  logical, save :: q_print_svb
  real(chm_real),allocatable,dimension(:),  save:: EVALN_all,H_mat_linear
  real(chm_real),allocatable,dimension(:,:),save:: H_mat,C_mat,H_coupling ! ,H_coupling_pot
  real(chm_real),allocatable,dimension(:,:),save:: dx_l,dy_l,dz_l
  real(chm_real),allocatable,dimension(:),  save:: s1,s2,s3,s4,s5,s6,s7,evals

  !
  integer,save :: n_mm_bond_atoms=0      ! number of atoms involved.
  integer,allocatable,dimension(:),save:: i_map_atoms
  !
  TYPE, public :: mts_mmbond
     integer :: n_mm_bond=0,n_mm_angl=0,n_mm_dihe=0,n_mm_imph=0, &
                n_mmbond =0
     integer,allocatable,dimension(:,:):: i_mm_bond, &
                                          i_mm_angl, &
                                          i_mm_dihe, &
                                          i_mm_imph, &
                                          i_morse_bond
     integer,allocatable,dimension(:):: icb_bond, &
                                        ict_angl, &
                                        icp_dihe,IDIH_dihe, &
                                        ici_imph,IIMP_imph
     real(chm_real),allocatable,dimension(:,:):: RBND_mm, &
                                                 RANG_mm, &
                                                 RANG_UB_mm, &
                                                 RDIH_mm, &
                                                 RIMP_mm
     real(chm_real),allocatable,dimension(:):: De_bond,Alp_bond,Rij_ref_bond

     !!! to save energy and gradients to be used when necessary
     !!real(chm_real) :: E_mmbond
     !!real(chm_real),allocatable,dimension(:):: dx_bond,dy_bond,dz_bond
  END TYPE mts_mmbond

  ! assign
  TYPE(mts_mmbond),allocatable,save:: mmbond_r(:)

!!!#endif /*mts*/

  contains
  !=====================================================================
#if KEY_MNDO97==1 /*mndo97*/
!!!#if KEY_MTS==1    /*mts*/
  SUBROUTINE Setup_mmbond(COMLYN,COMLEN,qmm_bond,qmm_angle,qmm_dihe,qmm_imph,qmm_morse,qmm_print, &
                          qmm_bond_corr, &
                          n_mm_replica,q_mm_write,q_mm_read,iunit_bond_write,force_scale)
  !
  ! check pairs and save paired atoms and parameters.
  ! For now, BOND, ANGL, DIHE and IMDI are supported.
  !
  ! And setup data array for bond, angle, dihe, improper energy calculations.
  !
  use exfunc
  use dimens_fcm
  !use string
  use stream
  use psf
  use gamess_fcm, only: igmsel
  use mndgho, only : QLINK
  use code
  use param
  use parallel

  implicit none
  CHARACTER(len=*):: COMLYN
  INTEGER:: COMLEN
  logical:: qmm_bond,qmm_angle,qmm_dihe,qmm_imph,qmm_morse,q_mm_write,q_mm_read,qmm_print, &
            qmm_bond_corr
  integer:: n_mm_replica,iunit_bond_write
  real(chm_real):: force_scale

  !
  character(len=4) :: wrd
  integer:: iterm,i,ii,jj,kk,ll,m,ic,icnt_mm
  integer:: ier=0
  integer,allocatable,dimension(:):: mm_bond_local,mm_angl_local, &
                                     mm_dihe_local,mm_imph_local
  logical:: q_do_this,q_check,qcheck
  logical,allocatable,dimension(:):: q_mm_atoms

  !
  if(n_mm_replica>=2) n_replica = n_mm_replica   ! no. of mm bond replicas.
                                                 ! n_replica > 1, EVB potential approach will be
                                                 !                used. In this case, mm bonds info
                                                 !                must be read in...
  !
  ! memory allocation
  if(allocated(mmbond_r))    deallocate(mmbond_r)
  if(allocated(dx_l))        deallocate(dx_l)
  if(allocated(dy_l))        deallocate(dy_l)
  if(allocated(dz_l))        deallocate(dz_l)
  if(allocated(i_map_atoms)) deallocate(i_map_atoms)
  allocate(mmbond_r(n_replica))
  allocate(dx_l(natom,n_replica+1))
  allocate(dy_l(natom,n_replica+1))
  allocate(dz_l(natom,n_replica+1))
  allocate(i_map_atoms(natom))

  if(n_replica>=2) then
     if(allocated(EVALN_all))      deallocate(EVALN_all)
     if(allocated(H_mat_linear))   deallocate(H_mat_linear)   ! linear H matrix
     if(allocated(H_mat))          deallocate(H_mat)
     if(allocated(C_mat))          deallocate(C_mat)          ! eigenvector
     if(allocated(H_coupling))     deallocate(H_coupling)     ! constant H coupling param
     !if(allocated(H_coupling_pot)) deallocate(H_coupling_pot) ! dist-dependent H coupling potential
     if(allocated(evals))          deallocate(evals)          ! eigenvalues
     allocate(EVALN_all(n_replica))
     allocate(H_mat_linear(n_replica*(n_replica+1)/2))
     allocate(H_mat(n_replica,n_replica))
     allocate(C_mat(n_replica,n_replica))
     allocate(H_coupling(n_replica,n_replica))
     !allocate(H_coupling_pot(n_replica,n_replica))
     allocate(evals(n_replica))

     ! scratch arrays for matrix diagonalization
     if(allocated(s1)) deallocate(s1)
     if(allocated(s2)) deallocate(s2)
     if(allocated(s3)) deallocate(s3)
     if(allocated(s4)) deallocate(s4)
     if(allocated(s5)) deallocate(s5)
     if(allocated(s6)) deallocate(s6)
     if(allocated(s7)) deallocate(s7)
     allocate(s1(n_replica))
     allocate(s2(n_replica))
     allocate(s3(n_replica))
     allocate(s4(n_replica))
     allocate(s5(n_replica))
     allocate(s6(n_replica))
     allocate(s7(n_replica))
  end if

  !
  q_print_svb     = qmm_print
  do_mm_bond      =.false.
  do_mm_angl      =.false.
  do_mm_dihe      =.false.
  do_mm_impr      =.false.
  do_morse_mbond  =.false.
  n_mm_bond_atoms = 0
  do i=1,n_replica
     mmbond_r(i)%n_mm_bond = 0
     mmbond_r(i)%n_mm_angl = 0
     mmbond_r(i)%n_mm_dihe = 0
     mmbond_r(i)%n_mm_imph = 0
     mmbond_r(i)%n_mmbond  = 0
  end do

!!  ! allocate memory arrays for MM-BOND-correction scheme, which is used in 
!!  ! mm-level bond freq. corrections for MTS AI-QM/MM calc.
!!  if(qmm_bond_corr) then
!!     ! only keep bond terms..
!!     ! 
!!     do i=1,n_replica
!!        if(allocated(mmbond_r(i)%dx_bond)) deallocate(mmbond_r(i)%dx_bond)
!!        if(allocated(mmbond_r(i)%dy_bond)) deallocate(mmbond_r(i)%dy_bond)
!!        if(allocated(mmbond_r(i)%dz_bond)) deallocate(mmbond_r(i)%dz_bond)
!!
!!        allocate(mmbond_r(i)%dx_bond(natom))
!!        allocate(mmbond_r(i)%dy_bond(natom))
!!        allocate(mmbond_r(i)%dz_bond(natom))
!!
!!        ! initialize
!!        mmbond_r(i)%E_mmbond = zero
!!        do j=1,natom
!!           mmbond_r(i)%dx_bond(j) =zero
!!           mmbond_r(i)%dy_bond(j) =zero
!!           mmbond_r(i)%dz_bond(j) =zero
!!        end do
!!     end do
!!  end if

  ! local memory, for later filling in i_map_atoms.
  if(allocated(q_mm_atoms)) deallocate(q_mm_atoms)
  allocate(q_mm_atoms(natom))
  icnt_mm = 0
  q_mm_atoms(1:natom) = .false.

  ! read previous setup or new setup
  if(q_mm_read) then
     q_check =.true.  ! default
     do_mm_bond     = qmm_bond
     do_mm_angl     = qmm_angle
     do_morse_mbond = qmm_morse
     do i=1,n_replica
        qcheck=.true.
        call READ_mm_bond_info(iunit_bond_write+i-1,i,natom,q_mm_atoms,qcheck)
        if(.not. qcheck) then
           if(prnlev>=2) write(outu,22) 'Error in file:',i
           q_check=.false.
        end if
     end do

     if(q_check) then
        ! memory and others are set correctly for n_replica mm setups (e.g., rs, ps, etc)
        q_mts_mmbond               =.true.
        if(qmm_bond_corr) q_mmbond =.true.

        ! setup coupling parameters/do the remainder on the routine Setup_mmbond_coupling
        if(allocated(H_coupling)) H_coupling(1:n_replica,1:n_replica) = zero
     else
        q_mts_mmbond               =.false.
        !q_mmbond                   =.false.
     end if
  else
     !
     ! below, n_replica must be 1.

     ! bond pairs
     if(qmm_bond) then
        iterm = 0
        allocate(mm_bond_local(nbond))
        do i=1,nbond
           ii = ib(i)
           jj = jb(i)
           q_do_this =.false.
           ! only between QM pairs.
           ! for gho-atom --- mm pairs should be computed at the mm level. (Check to make sure.)
           if(abs(igmsel(ii)) == 2) q_do_this =.true.                       ! h-atom case
           if(abs(igmsel(jj)) == 2) q_do_this =.true.                       !
           if(((abs(igmsel(ii)) == 1) .or. (abs(igmsel(ii))==2)) .and.  &
              ((abs(igmsel(jj)) == 1) .or. (abs(igmsel(jj))==2)) ) q_do_this =.true.
           if(q_do_this) then
              ic = ICB(i)
              if(cbc(ic) == zero) cycle
              iterm                = iterm + 1
              mm_bond_local(iterm) = i
           end if
        end do
        mmbond_r(1)%n_mm_bond = iterm  ! total number of mm bond terms for qm region
        if(mmbond_r(1)%n_mm_bond>0) then
           do_mm_bond=.true.
           if(allocated(mmbond_r(1)%i_mm_bond)) deallocate(mmbond_r(1)%i_mm_bond,stat=ier)
           if(allocated(mmbond_r(1)%icb_bond))  deallocate(mmbond_r(1)%icb_bond,stat=ier)
           if(allocated(mmbond_r(1)%RBND_mm))   deallocate(mmbond_r(1)%RBND_mm,stat=ier)
           allocate(mmbond_r(1)%i_mm_bond(2,mmbond_r(1)%n_mm_bond),stat=ier)
           allocate(mmbond_r(1)%icb_bond(mmbond_r(1)%n_mm_bond),stat=ier)
           allocate(mmbond_r(1)%RBND_mm(2,mmbond_r(1)%n_mm_bond),stat=ier)
           do m=1,mmbond_r(1)%n_mm_bond
              i = mm_bond_local(m)
              ic = ICB(i)
              mmbond_r(1)%i_mm_bond(1,m) = ib(i)
              mmbond_r(1)%i_mm_bond(2,m) = jb(i)
              mmbond_r(1)%icb_bond(m)    = ic
              mmbond_r(1)%RBND_mm(1,m)   = cbb(ic)
              mmbond_r(1)%RBND_mm(2,m)   = cbc(ic)*force_scale

              !
              q_mm_atoms(ib(i)) = .true.   ! i.e., atoms included in the calculation
              q_mm_atoms(jb(i)) = .true.
           end do
           if(prnlev>=2) write(outu,22) 'Total number of bonds:',mmbond_r(1)%n_mm_bond
22         format('Setup_mmbond> ',A,I7)
        else
           do_mm_bond=.false.
           call wrndie(-1,'<Setup_mmbond>','No MM bonds selected in the qm region.')
        end if
        deallocate(mm_bond_local)
     end if
     !
     ! angle pairs
     ! for now, we are not including UREY-b term
     if(qmm_angle) then
        iterm = 0
        allocate(mm_angl_local(ntheta))
        do i=1,ntheta
           ii = it(i)
           jj = jt(i)
           kk = kt(i)
           q_do_this =.false.
           !
           ! Should include between all qm atoms and 
           ! For h-link atom, h-atom---qm-atom---qm-atom or vice versa.
           if( abs(igmsel(jj)) == 2 ) q_do_this =.true.                                     ! h-atom case
           if((abs(igmsel(ii)) == 2) .and. ((igmsel(jj) == 0) .or.(igmsel(jj) == 5))) q_do_this =.true.
           if((abs(igmsel(kk)) == 2) .and. ((igmsel(jj) == 0) .or.(igmsel(jj) == 5))) q_do_this =.true.
           if ((abs(igmsel(ii))==1 .or. abs(igmsel(ii))==2) .and. &
               (abs(igmsel(jj))==1 .or. abs(igmsel(jj))==2) .and. &
               (abs(igmsel(kk))==1 .or. abs(igmsel(kk))==2)) q_do_this =.true.
           if(q_do_this) then
              ! all qm atoms.
              iterm                = iterm + 1
              mm_angl_local(iterm) = i
           end if
        end do
        mmbond_r(1)%n_mm_angl = iterm ! total number of mm angle terms for qm region.
        if(mmbond_r(1)%n_mm_angl>0) then
           do_mm_angl=.true.
           if(allocated(mmbond_r(1)%i_mm_angl))  deallocate(mmbond_r(1)%i_mm_angl,stat=ier)
           if(allocated(mmbond_r(1)%ict_angl))   deallocate(mmbond_r(1)%ict_angl,stat=ier)
           if(allocated(mmbond_r(1)%RANG_mm))    deallocate(mmbond_r(1)%RANG_mm,stat=ier)
           if(allocated(mmbond_r(1)%RANG_UB_mm)) deallocate(mmbond_r(1)%RANG_UB_mm,stat=ier)
           allocate(mmbond_r(1)%i_mm_angl(3,mmbond_r(1)%n_mm_angl),stat=ier)
           allocate(mmbond_r(1)%ict_angl(mmbond_r(1)%n_mm_angl),stat=ier)
           allocate(mmbond_r(1)%RANG_mm(2,mmbond_r(1)%n_mm_angl),stat=ier)
           allocate(mmbond_r(1)%RANG_UB_mm(2,mmbond_r(1)%n_mm_angl),stat=ier)
           do m=1,mmbond_r(1)%n_mm_angl
              i = mm_angl_local(m)
              ic= ICT(i)
              mmbond_r(1)%i_mm_angl(1,m) = it(i)
              mmbond_r(1)%i_mm_angl(2,m) = jt(i)
              mmbond_r(1)%i_mm_angl(3,m) = kt(i)
              mmbond_r(1)%ict_angl(m)    = ic
              mmbond_r(1)%RANG_mm(1,m)   = ctb(ic)
              mmbond_r(1)%RANG_mm(2,m)   = ctc(ic)*force_scale
              if(ctuc(ic) == zero) then
                 mmbond_r(1)%RANG_UB_mm(1:2,m)= zero
              else
                 mmbond_r(1)%RANG_UB_mm(1,m)= ctub(ic)              ! u-b equilibrium distance
                 mmbond_r(1)%RANG_UB_mm(2,m)= ctuc(ic)*force_scale  ! u-b force constant
              end if

              !
              q_mm_atoms(it(i)) = .true.   ! i.e., atoms included in the calculation
              q_mm_atoms(jt(i)) = .true.
              q_mm_atoms(kt(i)) = .true.
           end do
           if(prnlev>=2) write(outu,22) 'Total number of angles:',mmbond_r(1)%n_mm_angl
        else
           do_mm_angl=.false.
           call wrndie(-1,'<Setup_mmbond>','No MM angles selected in the qm region.')
        end if
        deallocate(mm_angl_local)
     end if
     !
     ! dihedral pairs
     if(qmm_dihe) then
        iterm = 0
        allocate(mm_dihe_local(nphi))
        do i=1,nphi
           ii = ip(i)
           jj = jp(i)
           kk = kp(i)
           ll = lp(i)
           q_do_this =.false.
           if(abs(igmsel(ii)) == 2) q_do_this =.true.  ! h-atom case
           if(abs(igmsel(jj)) == 2) q_do_this =.true.
           if(abs(igmsel(kk)) == 2) q_do_this =.true.
           if(abs(igmsel(ll)) == 2) q_do_this =.true.
           if(QLINK) then
              ! for GHO atoms: Q-Q-Q-Q only
              if((abs(igmsel(ii))==1 .or. abs(igmsel(ii))==2) .and. &
                 (abs(igmsel(jj))==1 .or. abs(igmsel(jj))==2) .and. &
                 (abs(igmsel(kk))==1 .or. abs(igmsel(kk))==2) .and. &
                 (abs(igmsel(ll))==1 .or. abs(igmsel(ll))==2)) q_do_this =.true.
           else
              ! for link atom: X-Q-Q-X, in which jj and kk should not be h-link atom. 
              !if(abs(igmsel(jj))==1 .and. abs(igmsel(kk))==1) q_do_this =.true.
              if(((abs(igmsel(jj)) == 1).or.(abs(igmsel(jj)) == 2)) .and. &
                 ((abs(igmsel(kk)) == 1).or.(abs(igmsel(kk)) == 2))) q_do_this =.true.
           end if
           if(q_do_this) then
              iterm                = iterm + 1
              mm_dihe_local(iterm) = i
           end if
        end do
        mmbond_r(1)%n_mm_dihe = iterm ! total number of mm dihedral terms for qm region.
        if(mmbond_r(1)%n_mm_dihe>0) then
           do_mm_dihe=.true.
           if(allocated(mmbond_r(1)%i_mm_dihe)) deallocate(mmbond_r(1)%i_mm_dihe,stat=ier)
           if(allocated(mmbond_r(1)%icp_dihe))  deallocate(mmbond_r(1)%icp_dihe,stat=ier)
           if(allocated(mmbond_r(1)%IDIH_dihe)) deallocate(mmbond_r(1)%IDIH_dihe,stat=ier)
           if(allocated(mmbond_r(1)%RDIH_mm)) deallocate(mmbond_r(1)%RDIH_mm,stat=ier)
           allocate(mmbond_r(1)%i_mm_dihe(4,mmbond_r(1)%n_mm_dihe),stat=ier)
           allocate(mmbond_r(1)%icp_dihe(mmbond_r(1)%n_mm_dihe),stat=ier)
           allocate(mmbond_r(1)%IDIH_dihe(mmbond_r(1)%n_mm_dihe),stat=ier)
           allocate(mmbond_r(1)%RDIH_mm(3,mmbond_r(1)%n_mm_dihe),stat=ier)
           do m=1,mmbond_r(1)%n_mm_dihe
              i = mm_dihe_local(m)
              ic= ICP(i)
              mmbond_r(1)%i_mm_dihe(1,m) = ip(i)
              mmbond_r(1)%i_mm_dihe(2,m) = jp(i)
              mmbond_r(1)%i_mm_dihe(3,m) = kp(i)
              mmbond_r(1)%i_mm_dihe(4,m) = lp(i)
              mmbond_r(1)%icp_dihe(m)    = ic
              mmbond_r(1)%IDIH_dihe(m)   = CPD(ic)
              mmbond_r(1)%RDIH_mm(1,m) = CPCOS(ic)
              mmbond_r(1)%RDIH_mm(2,m) = CPSIN(ic)
              mmbond_r(1)%RDIH_mm(3,m) = CPC(ic)*force_scale

              !
              q_mm_atoms(ip(i)) = .true.   ! i.e., atoms included in the calculation
              q_mm_atoms(jp(i)) = .true.
              q_mm_atoms(kp(i)) = .true.
              q_mm_atoms(lp(i)) = .true.
           end do
           if(prnlev>=2) write(outu,22) 'Total number of dihedrals:',mmbond_r(1)%n_mm_dihe
        else
           do_mm_dihe=.false.
           call wrndie(-1,'<Setup_mmbond>','No MM dihedral angles selected in the qm region.')
        end if
        deallocate(mm_dihe_local)
     end if
     !
     ! improper dihedral pairs
     if(qmm_imph) then
        iterm = 0
        allocate(mm_imph_local(nimphi))
        do i=1,nimphi
           ii = im(i)
           jj = jm(i)
           kk = km(i)
           ll = lm(i)
           q_do_this =.false.
           if(abs(igmsel(ii)) == 2) q_do_this =.true.  ! h-atom case
           if(abs(igmsel(jj)) == 2) q_do_this =.true.
           if(abs(igmsel(kk)) == 2) q_do_this =.true.
           if(abs(igmsel(ll)) == 2) q_do_this =.true.
           if(QLINK) then
              ! for GHO atoms: Q-Q-Q-Q only
              if((abs(igmsel(ii))==1 .or. abs(igmsel(ii))==2) .and. &
                 (abs(igmsel(jj))==1 .or. abs(igmsel(jj))==2) .and. &
                 (abs(igmsel(kk))==1 .or. abs(igmsel(kk))==2) .and. &
                 (abs(igmsel(ll))==1 .or. abs(igmsel(ll))==2)) q_do_this =.true.
           else
              ! for link atom: X-Q-Q-X, in which ii and ll should not be h-link atom.
              !if(abs(igmsel(ii))==1 .and. abs(igmsel(ll))==1) q_do_this =.true.
              if(((abs(igmsel(ii)) == 1).or.(abs(igmsel(ii)) == 2)) .and. &
                 ((abs(igmsel(ll)) == 1).or.(abs(igmsel(ll)) == 2))) q_do_this =.true.
           end if
           if(q_do_this) then
              iterm                = iterm + 1
              mm_imph_local(iterm) = i
           end if
        end do
        mmbond_r(1)%n_mm_imph = iterm ! total number of mm improper terms for qm region.
        if(mmbond_r(1)%n_mm_imph>0) then
           do_mm_impr=.true.
           if(allocated(mmbond_r(1)%i_mm_imph)) deallocate(mmbond_r(1)%i_mm_imph,stat=ier)
           if(allocated(mmbond_r(1)%ici_imph))  deallocate(mmbond_r(1)%ici_imph,stat=ier)
           if(allocated(mmbond_r(1)%IIMP_imph)) deallocate(mmbond_r(1)%IIMP_imph,stat=ier)
           if(allocated(mmbond_r(1)%RIMP_mm)) deallocate(mmbond_r(1)%RIMP_mm,stat=ier)
           allocate(mmbond_r(1)%i_mm_imph(4,mmbond_r(1)%n_mm_imph),stat=ier)
           allocate(mmbond_r(1)%ici_imph(mmbond_r(1)%n_mm_imph),stat=ier)
           allocate(mmbond_r(1)%IIMP_imph(mmbond_r(1)%n_mm_imph),stat=ier)
           allocate(mmbond_r(1)%RIMP_mm(4,mmbond_r(1)%n_mm_imph),stat=ier)
           do m=1,mmbond_r(1)%n_mm_imph
              i = mm_imph_local(m)
              ic= ICI(i)
              mmbond_r(1)%i_mm_imph(1,m) = im(i)
              mmbond_r(1)%i_mm_imph(2,m) = jm(i)
              mmbond_r(1)%i_mm_imph(3,m) = km(i)
              mmbond_r(1)%i_mm_imph(4,m) = lm(i)
              mmbond_r(1)%ici_imph(m)    = ic
              mmbond_r(1)%IIMP_imph(m)   = CID(ic)
              mmbond_r(1)%RIMP_mm(1,m) = CICOS(ic)
              mmbond_r(1)%RIMP_mm(2,m) = CISIN(ic)
              mmbond_r(1)%RIMP_mm(3,m) = CIB(ic)
              mmbond_r(1)%RIMP_mm(4,m) = CIC(ic)*force_scale

              !
              q_mm_atoms(im(i)) = .true.
              q_mm_atoms(jm(i)) = .true.
              q_mm_atoms(km(i)) = .true.
              q_mm_atoms(lm(i)) = .true.
           end do
           if(prnlev>=2) write(outu,22) 'Total number of improper dihedrals:',mmbond_r(1)%n_mm_imph
        else
           do_mm_impr=.false.
           call wrndie(-1,'<Setup_mmbond>','No MM improper angles selected in the qm region.')
        end if
        deallocate(mm_imph_local)
     end if
     !
     if(qmm_morse) call Setup_mmbond_morse(COMLYN,COMLEN,natom,q_mm_atoms,force_scale)
     !
     ! finally, the main flag
     if(do_mm_bond.or.do_mm_angl.or.do_mm_dihe.or.do_mm_impr.or.do_morse_mbond) then
        q_mts_mmbond               =.true.
        if(qmm_bond_corr) q_mmbond =.true.
     else
        q_mts_mmbond               =.false.
        q_mmbond                   =.false.
     end if
     !
     ! write mm bond/angle information into iunit_bond_write unit.
     ! this file can be read in...
     if(q_mm_write) call WRITE_mm_bond_info(iunit_bond_write,1)
  end if
  !
  ! now, fill in i_map_atoms array based on q_mm_atoms, for atoms included in the
  ! mm bonds/angles calculations.
  ii = 0
  do i=1,natom
     if(q_mm_atoms(i)) then
        ii = ii + 1
        i_map_atoms(ii) = i
     end if
  end do
  n_mm_bond_atoms = ii    ! no. of atoms
  !
  ! local memory
  if(allocated(q_mm_atoms)) deallocate(q_mm_atoms)
  ! 
  return
  END SUBROUTINE Setup_mmbond


  SUBROUTINE Setup_mmbond_morse(COMLYN,COMLEN,natomx,q_mm_atoms,force_scale)
  !
  ! Setup bond pairs for Morse-type energy form
  ! for the simulation of chemical reactions.
  !
  ! Only works for bond terms.
  !
  ! Usage:
  ! MORSe MBONd [int] 2x[int] 2x[real] ...  ! 2x[int] repeats for N_mbond times.
  !
  ! MBONd [int] : There are int-number (N_mbond) of Morse-bonds, each of which is described
  !               by the Morse potential.
  !       2x[int]: The first atom should be the middle atom.
  !                The second atom should be the atom attached to the first atom in psf.
  !       2x[real]: The first real number is De value (for both bonds).
  !                 The second real number is alpha value (for both bonds).
  !                 If negative, it will be SQRT(k_bond/De) value.
  !
  ! For bond,  U = De*[1- EXP{-alpha*|r_ij - r_ref|}]^2
  !
  use exfunc
  use dimens_fcm
  use string
  use stream
  use psf
  use code
  use param

  implicit none
  CHARACTER(len=*):: COMLYN
  INTEGER ::  COMLEN
  integer :: natomx
  logical :: q_mm_atoms(natomx)
  real(chm_real):: force_scale

  integer:: iterm,i,ii,jj,kk,i1,j1,k1,m,icnt
  integer:: ier=0
  logical:: q_test

  !
  if(n_replica>1) then
     call wrndie(-5,'<Setup_mmbond_morse>','MORSe option not supported if NUMQ > 1. Ignored.')
     return
  end if

  mmbond_r(1)%n_mmbond = 0
  do_morse_mbond       =.false.
  if(allocated(mmbond_r(1)%i_morse_bond)) deallocate(mmbond_r(1)%i_morse_bond,stat=ier)
  if(allocated(mmbond_r(1)%De_bond))      deallocate(mmbond_r(1)%De_bond,stat=ier)
  if(allocated(mmbond_r(1)%Alp_bond))     deallocate(mmbond_r(1)%Alp_bond,stat=ier)
  if(allocated(mmbond_r(1)%Rij_ref_bond)) deallocate(mmbond_r(1)%Rij_ref_bond,stat=ier)

  ! bond pairs.
  if(COMLEN<=0) return

  !
  mmbond_r(1)%n_mmbond=gtrmi(COMLYN,COMLEN,'MBON',-1)
  if(do_mm_bond .and. mmbond_r(1)%n_mm_bond>0 .and. mmbond_r(1)%n_mmbond>0) then
     allocate(mmbond_r(1)%i_morse_bond(2,mmbond_r(1)%n_mmbond),stat=ier)
     allocate(mmbond_r(1)%De_bond(mmbond_r(1)%n_mmbond),stat=ier)
     allocate(mmbond_r(1)%Alp_bond(mmbond_r(1)%n_mmbond),stat=ier)
     allocate(mmbond_r(1)%Rij_ref_bond(mmbond_r(1)%n_mmbond),stat=ier)
     do i=1,mmbond_r(1)%n_mmbond
        ! bond: i-----j
        mmbond_r(1)%i_morse_bond(1,i)=NEXTI(COMLYN,COMLEN)  ! the center  atom (i)
        mmbond_r(1)%i_morse_bond(2,i)=NEXTI(COMLYN,COMLEN)  ! the leaving atom (j)
        mmbond_r(1)%De_bond(i)       =NEXTF(COMLYN,COMLEN)
        mmbond_r(1)%De_bond(i)       =mmbond_r(1)%De_bond(i)*force_scale
        mmbond_r(1)%Alp_bond(i)      =NEXTF(COMLYN,COMLEN)
        q_mm_atoms(mmbond_r(1)%i_morse_bond(1,i)) =.true.
        q_mm_atoms(mmbond_r(1)%i_morse_bond(2,i)) =.true.
     end do

     ! now check bond list..
     icnt=0
     do m=1,mmbond_r(1)%n_mm_bond
        ii = mmbond_r(1)%i_mm_bond(1,m)
        jj = mmbond_r(1)%i_mm_bond(2,m)
        ! go over n_mmbond list to find.
        q_test=.true.
        do i= 1,mmbond_r(1)%n_mmbond
           if((ii==mmbond_r(1)%i_morse_bond(1,i) .and. jj==mmbond_r(1)%i_morse_bond(2,i)) .or. &
              (jj==mmbond_r(1)%i_morse_bond(1,i) .and. ii==mmbond_r(1)%i_morse_bond(2,i))) then
              if(prnlev>=2) then
                write(outu,21) 'The following pair is treated by the Morse-type Bond.',ii,jj
              end if
              mmbond_r(1)%icb_bond(m)     =-ABS(mmbond_r(1)%icb_bond(m)) ! flag to skip in the regular bond list.
              !
              if(mmbond_r(1)%Alp_bond(i) < zero) then
                 mmbond_r(1)%Alp_bond(i)=SQRT(mmbond_r(1)%RBND_mm(2,m)/mmbond_r(1)%De_bond(i))
              end if
              !
              mmbond_r(1)%Rij_ref_bond(i) = mmbond_r(1)%RBND_mm(1,m)
              !
              icnt=icnt+1
              exit
           end if
        end do
     end do
     if(icnt == mmbond_r(1)%n_mmbond) then
        do_morse_mbond=.true.
        if(prnlev>=2) write(outu,20) 'Total number of Morse-type bond:',mmbond_r(1)%n_mmbond
     else
        call wrndie(-5,'<Setup_mmbond_morse>','Input error for MBONd selection.')
     end if
  end if
20 format('Setup_mmbond_morse> ',A,I7)
21 format('Setup_mmbond_morse> ',A,2I7)
  !
  return
  END SUBROUTINE Setup_mmbond_morse


  SUBROUTINE Setup_mmbond_coupling(COMLYN,COMLEN,q_coupling)
  !-----------------------------------------------------------------------
  ! Read coupling parameter values... 
  ! This is separated from the above Setup_mmbond routine as 
  ! reading coupling/constant values need to be handled at the
  ! end of the setup.
  use exfunc
  use dimens_fcm
  use string
  use stream

  implicit none
  CHARACTER(len=*):: COMLYN
  INTEGER ::  COMLEN
  integer :: i,ii,jj,n_coupling
  logical :: q_coupling

  ! setup coupling parameters
  H_coupling(1:n_replica,1:n_replica) = zero
  if(q_coupling) then
     n_coupling = GTRMI(COMLYN,COMLEN,'CONS',0)
     do i=1,n_coupling
        ii                = NEXTI(COMLYN,COMLEN)
        jj                = NEXTI(COMLYN,COMLEN)
        H_coupling(ii,jj) = NEXTF(COMLYN,COMLEN)
        H_coupling(jj,ii) = H_coupling(ii,jj)
    end do
    !! debug
    !if(prnlev>=2) then
    !   do i=1,n_replica
    !      write(outu,*) i,(H_coupling(jj,i),jj=1,n_replica)
    !   end do
    !end if
  end if

  return
  END SUBROUTINE Setup_mmbond_coupling

  SUBROUTINE Energy_mmbond(natom,EVALN,Prefct,X,Y,Z,DX,DY,DZ)
  !-----------------------------------------------------------------------
  ! Call each routine to compute internal valence energy terms
  ! connecting qm atoms.
  !
  ! Prefct: pre-factor, either 1.0 or -1.0 (depending on the step).
  !
  use parallel
  use stream
  implicit none
  integer       :: natom
  real(chm_real):: EVALN,Prefct,X(natom),Y(natom),Z(natom),DX(natom),DY(natom),DZ(natom)

  real(chm_real) :: EBND,EANG,EDIH,EIMP,EBND_morse,EVALN_local,EE_ll(5)
  real(chm_real) :: r_tmp,eval_low,Prefct_local
  integer :: i,j,icnt,nnumnod,mmynod,ii,jj

  ! all routines assume Scalar Fast routine version
  ! also, Urey-Bradley terms will be ignored...Unluckily.. : (
  EBND      =ZERO
  EANG      =ZERO
  EDIH      =ZERO
  EIMP      =ZERO
  EBND_morse=ZERO

#if KEY_PARALLEL==1
  mmynod  = mynod
  nnumnod = numnod
#else
  nnumnod = 1
  mmynod  = 0
#endif

  do i=1,n_replica
     do j=1,n_mm_bond_atoms  ! 1,natom
        jj = i_map_atoms(j)
        dx_l(jj,i) = zero
        dy_l(jj,i) = zero
        dz_l(jj,i) = zero
     end do

     !
     EBND      =ZERO
     EANG      =ZERO
     EDIH      =ZERO
     EIMP      =ZERO
     EBND_morse=ZERO
     Prefct_local = one

     ! bond energy
     if(do_mm_bond) call EBONDFS_mm(natom,i,EBND,mmynod,nnumnod, &
                                    X,Y,Z,DX_l(1:natom,i),DY_l(1:natom,i),DZ_l(1:natom,i))

     ! angle energy
     if(do_mm_angl) call EANGLFS_mm(natom,i,EANG,mmynod,nnumnod, &
                                    X,Y,Z,DX_l(1:natom,i),DY_l(1:natom,i),DZ_l(1:natom,i))

     ! dihedral energy
     if(do_mm_dihe) call EPHIFS_mm(natom,i,EDIH,mmynod,nnumnod, &
                                   X,Y,Z,DX_l(1:natom,i),DY_l(1:natom,i),DZ_l(1:natom,i))

     ! improper dihedral energy
     if(do_mm_impr) call EIPHIFS_mm(natom,i,EIMP,mmynod,nnumnod, &
                                    X,Y,Z,DX_l(1:natom,i),DY_l(1:natom,i),DZ_l(1:natom,i))

     ! morse mbond energy
     if(do_morse_mbond) call EMBOND_morse(natom,i,EBND_morse,mmynod,nnumnod, &
                                          X,Y,Z,DX_l(1:natom,i),DY_l(1:natom,i),DZ_l(1:natom,i))

     ! This value will be added to the qmmm energy.
     EVALN_local = EBND + EANG + EDIH + EIMP + EBND_morse
     !if(allocated(EVALN_all)) EVALN_all(i)= EVALN_local
     if(n_replica>=2) then
        EVALN_all(i)= EVALN_local
        !! debug
        !if(Prefct<zero .and. prnlev>=2) then
        !   write(6,'(A,6F12.5)') 'Energy:',EBND,EANG,EDIH,EIMP,EBND_morse,EVALN_all(i)
        !end if
     else
        !if(Prefct<zero .and. prnlev>=2) then
        !   write(6,'(A,6F12.5)') 'Energy:',EBND,EANG,EDIH,EIMP,EBND_morse,EVALN_local
        !end if
     end if
  end do

  !
  if(n_replica == 1) then
#if KEY_PARALLEL==1
     call gcomb(EVALN_local,n_replica)
     if(MYNOD == 0) then
#endif
        EVALN = EVALN + Prefct*EVALN_local
#if KEY_PARALLEL==1
     end if
#endif
     if(Prefct<zero .and. q_print_svb) then
        EE_ll(1) = EBND
        EE_ll(2) = EANG
        EE_ll(3) = EDIH
        EE_ll(4) = EIMP
        EE_ll(5) = EBND_morse
#if KEY_PARALLEL==1
        call gcomb(EE_ll,5)
#endif
        !write(outu,110) 'E_MBOND:',EVALN_local,' Other E values:',EBND,EANG,EDIH,EIMP,EBND_morse
        if(prnlev>=2) write(outu,110) 'E_MBOND:',EVALN_local,' Other E values:',(EE_ll(i),i=1,5)
     end if

     ! gradients
     do i=1,n_mm_bond_atoms  ! 1,natom
        jj     = i_map_atoms(i)
        dx(jj) = dx(jj) + Prefct*dx_l(jj,1)
        dy(jj) = dy(jj) + Prefct*dy_l(jj,1)
        dz(jj) = dz(jj) + Prefct*dz_l(jj,1)
     end do

  else if(n_replica>=2) then
     ! now do, evb evaluations.
     ! 1) H_matrix is setup for EVB potentials (diagonal and off-diagonal), including
     !             coupling/shift parameters
     ! 2) Diagonalization of H_matrix, smallest eivenvalue (the potential energy)
     !             and eigenvector are determined.
     ! 3) Return the energy and gradients following the Hellman-Feynmann theorem.

     ! Step 1)
     ! H_matrix, currently only supporting a constant coupling term.
#if KEY_PARALLEL==1
     call gcomb(EVALN_all,n_replica)
#endif
     icnt = 0
     do i=1,n_replica
        icnt = icnt + 1
        ! diagonal element
        H_mat(i,i)         = EVALN_all(i) + H_coupling(i,i)
        H_mat_linear(icnt) = EVALN_all(i) + H_coupling(i,i)

        ! off-diagonal elements
        do j=i+1,n_replica
           icnt       = icnt + 1
           H_mat(j,i) = H_coupling(j,i)
           H_mat(i,j) = H_coupling(j,i)
           H_mat_linear(icnt) = H_coupling(j,i)
        end do
     end do
!#if KEY_PARALLEL==1
!     if(mynod==0) then
!        do i=1,n_replica
!           write(6,'(i3,2(2X,F12.5))') i,H_mat(1,i),H_mat(2,i)
!        end do
!     end if
!#endif

     ! Step 2)
     ! diagonalize matrix
     ! s1-7 : scratach arrays for matrix diagonalization
     ! evals: eigenvalue array 
     call diagq(n_replica,n_replica,H_mat_linear,C_mat,s1,s2,s3,s4,evals,s5,s6,s7,0)

     ! find the lowest eigenvalue: EVB potential energy
     EVALN_local= evals(1)
     ii         = 1
     do i = 2, n_replica
        if(evals(i) < EVALN_local) then
           EVALN_local= evals(i)
           ii         = i
        end if
     end do

     ! eigenvector corresponding to the lowest eigenvalue
#if KEY_PARALLEL==1
     if(mynod==0) then
#endif
        s1(1:n_replica) = C_mat(1:n_replica,ii)

        ! 3) Step 3)
        ! energy
        EVALN = EVALN + Prefct*EVALN_local
        if(mmynod==0 .and. Prefct<zero .and. q_print_svb) then
           write(outu,110) 'E_EVB(MM):',EVALN_local,'    E_ii:',(EVALN_all(i),i=1,n_replica)
        end if
        !if(mmynod==0) write(6,*) Prefct,EVALN_local,(EVALN_all(i),i=1,n_replica)
#if KEY_PARALLEL==1
     end if
     ! broadcast
     call psnd8m(s1(1:n_replica),n_replica)
#endif

     ! gradients
     ! based on evb_ltm.F90; subroutine evb; Hellman-Feynmann approach
     !
     ! If M is H_ma, its eigenvector matrix is u, and its eigenvalue
     ! matrix is D (i.e., D=(U^T)*M*U), then dD/dp=(U^T)*(dM/dp)*(U), 
     ! where M depends on some parameter p. in this case, p represents the 
     ! cartesian directions - x,y,z, so that we have dM/dx, dM/dy, and dM/dz.
     !
     ! For a two state system, this method gives results identical
     ! to those obtained with an analytic formula, but is extendable to 
     ! multi-state systems where analytic forms are not available.
     ! NB:
     ! Analytic 2-state gradients (e.g., dx) would be calculated as follows:
     ! dx(i)=0.5*(dx_l(i,1)+dx_l(i,2) -tmpfac*(dx_l(i,1)-dx_l(i,2)))
     ! where tmpfac=(v1-v2)/sqrt((v1-v2)**2+4*ensebeta**2)
     ! J.N.Harvey version of gradients routine

     ! diagonal part of gradients
     do j=1,n_replica
        r_tmp = Prefct*s1(j)*s1(j)  ! squaring it here.
        do i=1,n_mm_bond_atoms  ! 1,natom
           ii     = i_map_atoms(i)
           dx(ii) = dx(ii) + r_tmp*dx_l(ii,j)  ! s1(j)**2*dx_l(ii,j)
           dy(ii) = dy(ii) + r_tmp*dy_l(ii,j)  ! s1(j)**2*dy_l(ii,j)
           dz(ii) = dz(ii) + r_tmp*dz_l(ii,j)  ! s1(j)**2*dz_l(ii,j)
        end do
     end do

     ! coupling term, but constant terms do not contribute to the off-diagonal gradients.
     ! ... do nothing ...
  end if
110 format(A,1X,F12.5,A,10F12.5)
  return
  END SUBROUTINE Energy_mmbond


  SUBROUTINE EBONDFS_mm(natom,irepl,EB,mmynod,nnumnod,X,Y,Z,DX,DY,DZ)
  !-----------------------------------------------------------------------
  !     calculates bond energies and forces as fast as possible
  !     Fast SCALAR version
  !     14-JUL-1983, Bernard R. Brooks
  !     31-Aug-1991, Youngdo Won
  !-----------------------------------------------------------------------
  use dimens_fcm
  use exfunc
  use stream
  !use psf
  !use code
  !
  implicit none

  integer        :: natom,irepl
  real(chm_real) :: EB
  real(chm_real) :: X(natom),Y(natom),Z(natom), DX(natom),DY(natom),DZ(natom)
  integer :: nnumnod,mmynod
  !
  real(chm_real) :: RXYZ(3),S,R,DB,DF,DXYZ(3),RCBC,RCBB,exv
  INTEGER :: MM,I,J,IC,II
  integer :: ist
  !
  !
  if(mmbond_r(irepl)%n_mm_bond<=0) return
  exv= zero
  ist=mmynod+1
  do ii=ist,mmbond_r(irepl)%n_mm_bond,nnumnod  ! 1,n_mm_bond
     i=mmbond_r(irepl)%i_mm_bond(1,ii)
     j=mmbond_r(irepl)%i_mm_bond(2,ii)
     if (i > 0 .and. mmbond_r(irepl)%icb_bond(ii) > 0) then
       RCBB=mmbond_r(irepl)%RBND_mm(1,II)
       RCBC=mmbond_r(irepl)%RBND_mm(2,II)

       !if(RCBC /= ZERO) then
          RXYZ(1)=X(I)-X(J)
          RXYZ(2)=Y(I)-Y(J)
          RXYZ(3)=Z(I)-Z(J)
          S=SQRT(RXYZ(1)*RXYZ(1)+RXYZ(2)*RXYZ(2)+RXYZ(3)*RXYZ(3))

          if(S > ZERO) then
             R        =TWO/S
             DB       =S-RCBB
             DF       =RCBC*DB
             exv      =exv+DF*DB
             !
             DF       =DF*R
             DXYZ(1:3)=RXYZ(1:3)*DF
             DX(I)    =DX(I)+DXYZ(1)
             DY(I)    =DY(I)+DXYZ(2)
             DZ(I)    =DZ(I)+DXYZ(3)
             DX(J)    =DX(J)-DXYZ(1)
             DY(J)    =DY(J)-DXYZ(2)
             DZ(J)    =DZ(J)-DXYZ(3)
           end if
       !end if
     end if
  end do
  !
  EB = exv
  return
  END SUBROUTINE EBONDFS_mm

  SUBROUTINE EANGLFS_mm(natom,irepl,ET,mmynod,nnumnod,X,Y,Z,DX,DY,DZ)
  !-----------------------------------------------------------------------
  !     calculates bond angles and bond angle energies as fast as possible
  !     Fast SCALAR version
  !     14-JUL-1983, Bernard R. Brooks
  !     31-Aug-1991, Youngdo Won
  !-----------------------------------------------------------------------
  use dimens_fcm
  use exfunc
  !
  use stream
  use consta
  !use psf
  !use code
  !
  implicit none

  integer        :: natom,irepl
  real(chm_real) :: ET
  real(chm_real) :: X(natom),Y(natom),Z(natom), DX(natom),DY(natom),DZ(natom)
  integer :: mmynod,nnumnod
  !
  real(chm_real) :: DRI(3),DRJ(3),DRIR(3),DRJR(3),DTRI(3),DTRJ(3)
  real(chm_real) :: RI,RJ,RIR,RJR,SMALLV,ST2R,STR
  real(chm_real) :: CST,AT,DA,DF,RCTB,RCTC,exv
  INTEGER :: NWARN,ITH,I,J,K,IC,II
  integer :: ist

  exv=zero
  SMALLV=RPRECI
  if(mmbond_r(irepl)%n_mm_angl<=0) return
  NWARN=0
  ist=mmynod+1
  do ii=ist,mmbond_r(irepl)%n_mm_angl,nnumnod  ! 1,n_mm_angl
     i=mmbond_r(irepl)%i_mm_angl(1,ii)
     j=mmbond_r(irepl)%i_mm_angl(2,ii)
     k=mmbond_r(irepl)%i_mm_angl(3,ii)
     if(i > 0 .and. mmbond_r(irepl)%ict_angl(ii) > 0) then
        ! for angle term
        RCTB=mmbond_r(irepl)%RANG_mm(1,II)
        RCTC=mmbond_r(irepl)%RANG_mm(2,II)
        if(RCTC /= zero) then
           DRI(1)=X(I)-X(J)
           DRI(2)=Y(I)-Y(J)
           DRI(3)=Z(I)-Z(J)

           DRJ(1)=X(K)-X(J)
           DRJ(2)=Y(K)-Y(J)
           DRJ(3)=Z(K)-Z(J)

           RI=SQRT(DRI(1)*DRI(1)+DRI(2)*DRI(2)+DRI(3)*DRI(3))
           RJ=SQRT(DRJ(1)*DRJ(1)+DRJ(2)*DRJ(2)+DRJ(3)*DRJ(3))

           if(RI > ZERO .and. RJ > ZERO) then
              RIR=ONE/RI
              RJR=ONE/RJ

              DRIR(1:3)=DRI(1:3)*RIR
              DRJR(1:3)=DRJ(1:3)*RJR
              CST=DRIR(1)*DRJR(1)+DRIR(2)*DRJR(2)+DRIR(3)*DRJR(3)

              if (ABS(CST) >= 0.999999) then
                 if(abs(cst) > one) CST=SIGN(one,CST)
                 NWARN=NWARN+1
                 if ((NWARN <= 5 .and. WRNLEV >= 5) .or. WRNLEV >= 6) WRITE(OUTU,445) ii,I,J,K
              end if
 445          FORMAT(' EANGLFS_mm> Warning: Angle',I5,' is almost linear.', &
                     /' Derivatives may be affected for atoms:',3I5)

              ! orignal?
              AT=ACOS(CST)
              DA=AT-RCTB
              DF=RCTC*DA

              exv= exv+DF*DA
              if(abs(cst) >= 0.999999) then
                 st2r = one/(one-cst*cst+SMALLV)
                 str  = sqrt(st2r)
                 df   =-TWO*DF*str
              else
                 st2r = one/(one-cst*cst)
                 str  = sqrt(st2r)
                 df   =-TWO*DF*str
              end if

              DTRI(1:3)=RIR*(DRJR(1:3)-CST*DRIR(1:3))
              DTRJ(1:3)=RJR*(DRIR(1:3)-CST*DRJR(1:3))

              DRI(1:3)=DF*DTRI(1:3)
              DRJ(1:3)=DF*DTRJ(1:3)

              DX(I) =DX(I)+DRI(1)
              DX(K) =DX(K)+DRJ(1)
              DX(J) =DX(J)-DRI(1)-DRJ(1)

              DY(I) =DY(I)+DRI(2)
              DY(K) =DY(K)+DRJ(2)
              DY(J) =DY(J)-DRI(2)-DRJ(2)

              DZ(I) =DZ(I)+DRI(3)
              DZ(K) =DZ(K)+DRJ(3)
              DZ(J) =DZ(J)-DRI(3)-DRJ(3)
           end if
        end if   ! (RCTC /= zero)

        ! for U-B  term
        RCTB=mmbond_r(irepl)%RANG_UB_mm(1,II)
        RCTC=mmbond_r(irepl)%RANG_UB_mm(2,II)
        if(RCTC /= zero) then
           DRI(1)=X(I)-X(K)
           DRI(2)=Y(I)-Y(K)
           DRI(3)=Z(I)-Z(K)
           RI=SQRT(DRI(1)*DRI(1)+DRI(2)*DRI(2)+DRI(3)*DRI(3))

           if(RI > ZERO) then
             RJR    =TWO/RI
             da     =RI - RCTB
             df     =RCTC*da
             exv    =exv+df*da
             !
             df       =df*RJR
             DTRI(1:3)=DRI(1:3)*DF
             DX(I)    =DX(I)+DTRI(1)
             DY(I)    =DY(I)+DTRI(2)
             DZ(I)    =DZ(I)+DTRI(3)
             DX(K)    =DX(K)-DTRI(1)
             DY(K)    =DY(K)-DTRI(2)
             DZ(K)    =DZ(K)-DTRI(3)
           end if
        end if  ! (RCTC /= zero) 
     end if
  end do
  !
  ET = exv
  return
  END SUBROUTINE EANGLFS_mm

  SUBROUTINE EPHIFS_mm(natom,irepl,EP,mmynod,nnumnod,X,Y,Z,DX,DY,DZ)
  !-----------------------------------------------------------------------
  !     Fast SCALAR version dihedral energy and force routine.
  !     Dihedral energy terms are expressed as a function of PHI.
  !     This avoids all problems as dihedrals become planar.
  !     The functional form is:
  !
  !     E = K*(1+COS(n*Phi-Phi0))
  !     Where:
  !     n IS A POSITIVE INTEGER COS(n(Phi-Phi0) AND SIN(n(Phi-Phi0)
  !            ARE CALCULATED BY RECURENCE TO AVOID LIMITATION ON n
  !     K IS THE FORCE CONSTANT IN kcal/mol/rad
  !     Phi0/n IS A MAXIMUM IN ENERGY.
  !
  !     The parameter of the routine is:
  !     EP: Diheral Energy returned
  !     Data come form param.f90
  !
  !     31-Aug-1991, Youngdo Won
  !
  !     New formulation by
  !           Arnaud Blondel    1994
  !
  !-----------------------------------------------------------------------
  use dimens_fcm
  use exfunc
  !
  use stream
  use consta
  !use code
  !use psf

  implicit none

  integer        :: natom,irepl
  real(chm_real) :: EP
  real(chm_real) :: X(natom),Y(natom),Z(natom), DX(natom),DY(natom),DZ(natom)
  integer        :: mmynod,nnumnod

  real(chm_real) :: FR(3),GR(3),HR(3),AR(3),BR(3),DFR(3),DGR(3),DHR(3)
  real(chm_real) :: RA2,RB2,RG,RGR,RA2R,RB2R,RABR,CT,ST
  real(chm_real) :: GAA,GBB,FG,HG,FGA,HGB
  real(chm_real) :: E,DF,E1,DF1,DDF1,ARG
  real(chm_real) :: RCPCOS,RCPSIN,RCPC,exv
  INTEGER :: NWARN,IPHI,I,J,K,L,IC,IPER,NPER,II,ICPD,ist
  LOGICAL :: LREP
  !
  real(chm_real),parameter :: RXMIN=0.005D0,RXMIN2=0.000025D0

  ! Initialize the torsion energy to zero
  exv= zero
  if(mmbond_r(irepl)%n_mm_dihe <=0) return
  NWARN=0
  ist=mmynod+1
  do ii=ist,mmbond_r(irepl)%n_mm_dihe,nnumnod  ! 1,n_mm_dihe
     i   = mmbond_r(irepl)%i_mm_dihe(1,ii)
     j   = mmbond_r(irepl)%i_mm_dihe(2,ii)
     k   = mmbond_r(irepl)%i_mm_dihe(3,ii)
     l   = mmbond_r(irepl)%i_mm_dihe(4,ii)

     icpd  =mmbond_r(irepl)%IDIH_dihe(ii)
     RCPCOS=mmbond_r(irepl)%RDIH_mm(1,ii)
     RCPSIN=mmbond_r(irepl)%RDIH_mm(2,ii)
     RCPC  =mmbond_r(irepl)%RDIH_mm(3,ii)

     ic = 1
     if(icpd /= 0) then
        ! F=Ri-Rj, G=Rj-Rk, H-Rl-Rk.
        FR(1)=X(I)-X(J)
        FR(2)=Y(I)-Y(J)
        FR(3)=Z(I)-Z(J)
        GR(1)=X(J)-X(K)
        GR(2)=Y(J)-Y(K)
        GR(3)=Z(J)-Z(K)
        HR(1)=X(L)-X(K)
        HR(2)=Y(L)-Y(K)
        HR(3)=Z(L)-Z(K)

        ! A=F^G, B=H^G.
        AR(1)=FR(2)*GR(3)-FR(3)*GR(2)
        AR(2)=FR(3)*GR(1)-FR(1)*GR(3)
        AR(3)=FR(1)*GR(2)-FR(2)*GR(1)
        BR(1)=HR(2)*GR(3)-HR(3)*GR(2)
        BR(2)=HR(3)*GR(1)-HR(1)*GR(3)
        BR(3)=HR(1)*GR(2)-HR(2)*GR(1)

        RA2=AR(1)*AR(1)+AR(2)*AR(2)+AR(3)*AR(3)
        RB2=BR(1)*BR(1)+BR(2)*BR(2)+BR(3)*BR(3)
        RG =SQRT(GR(1)*GR(1)+GR(2)*GR(2)+GR(3)*GR(3))

        ! Warnings have been simplified.
        if((RA2 <= RXMIN2).or.(RB2 <= RXMIN2).or.(RG <= RXMIN)) then
           NWARN=NWARN+1
           if((NWARN <= 5 .and. WRNLEV >= 5) .or. WRNLEV >= 6) WRITE(OUTU,20) ii,I,J,K,L
   20      FORMAT(' EPHIFS_mm: WARNING.  dihedral',I5,' is almost linear.'/ &
                  ' derivatives may be affected for atoms:',4I5)
        else
           !
           RGR =ONE/RG
           RA2R=ONE/RA2
           RB2R=ONE/RB2
           RABR=SQRT(RA2R*RB2R)

           ! CT=cos(phi)
           CT=(AR(1)*BR(1)+AR(2)*BR(2)+AR(3)*BR(3))*RABR

           ! ST=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
           ! which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
           ST=RG*RABR*(AR(1)*HR(1)+AR(2)*HR(2)+AR(3)*HR(3))

           ! Energy and derivative contributions.
           E =ZERO
           DF=ZERO

 30        CONTINUE
           IPER=ICPD
           if(IPER >= 0) then
              LREP=.FALSE.
           else
              LREP=.TRUE.
              IPER=-IPER
           end if
           E1  =ONE
           DF1 =ZERO
           DDF1=ZERO

           !calculation of cos(n*phi-phi0) and sin(n*phi-phi0).
           do NPER=1,IPER
              DDF1=E1*CT-DF1*ST
              DF1 =E1*ST+DF1*CT
              E1  =DDF1
           end do
           E1 = E1*RCPCOS +DF1*RCPSIN
           DF1= DF1*RCPCOS-DDF1*RCPSIN
           DF1=-IPER*DF1
           E1 = ONE+E1

           !brb...03-Jul-2004 Zero-period dihedral bugfix
           if(IPER == 0) E1=ONE

           ARG=RCPC
           E  =E+ARG*E1
           DF =DF+ARG*DF1

           if(LREP) then
              IC=IC+1
              GOTO 30
           end if

           ! Cumulate the energy
           exv= exv+E

           !     Compute derivatives wrt catesian coordinates.
           !
           ! GAA=dE/dphi.|G|/A^2, GBB=dE/dphi.|G|/B^2, FG=F.G, HG=H.G
           ! FGA=dE/dphi*F.G/(|G|A^2), HGB=dE/dphi*H.G/(|G|B^2)
           FG  = FR(1)*GR(1)+FR(2)*GR(2)+FR(3)*GR(3)
           HG  = HR(1)*GR(1)+HR(2)*GR(2)+HR(3)*GR(3)

           RA2R= DF*RA2R
           RB2R= DF*RB2R
           FGA = FG*RA2R*RGR
           HGB = HG*RB2R*RGR
           GAA = RA2R*RG
           GBB = RB2R*RG

           ! DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.
           DFR(1)=-GAA*AR(1)
           DFR(2)=-GAA*AR(2)
           DFR(3)=-GAA*AR(3)
           DGR(1)= FGA*AR(1) - HGB*BR(1)
           DGR(2)= FGA*AR(2) - HGB*BR(2)
           DGR(3)= FGA*AR(3) - HGB*BR(3)
           DHR(1)= GBB*BR(1)
           DHR(2)= GBB*BR(2)
           DHR(3)= GBB*BR(3)

           ! Distribute over Ri.
           DX(I)=DX(I)+DFR(1)
           DY(I)=DY(I)+DFR(2)
           DZ(I)=DZ(I)+DFR(3)
           DX(J)=DX(J)-DFR(1)+DGR(1)
           DY(J)=DY(J)-DFR(2)+DGR(2)
           DZ(J)=DZ(J)-DFR(3)+DGR(3)
           DX(K)=DX(K)-DHR(1)-DGR(1)
           DY(K)=DY(K)-DHR(2)-DGR(2)
           DZ(K)=DZ(K)-DHR(3)-DGR(3)
           DX(L)=DX(L)+DHR(1)
           DY(L)=DY(L)+DHR(2)
           DZ(L)=DZ(L)+DHR(3)
        end if
     end if
  end do
  !
  EP = exv
  return
  END SUBROUTINE EPHIFS_mm

  SUBROUTINE EIPHIFS_mm(natom,irepl,EIP,mmynod,nnumnod,X,Y,Z,DX,DY,DZ)
  !-----------------------------------------------------------------------
  !     Fast SCALAR version improper torsion energy and force routine.
  !
  !     For improper dihedrals, the energy term is given by:
  !     E = K*(Phi-phi0)**2
  !     WHERE
  !     K IS THE FORCE CONSTANT IN kcal/mol/rad
  !     Phi0 IS THE MINIMUM IN ENERGY.
  !
  !     The parameter of the routine is:
  !     EPI: Diheral Energy returned
  !     Data come form param.f90
  !
  !     31-Aug-1991, Youngdo Won
  !
  !     New formulation by
  !           Arnaud Blondel    1994
  !
  !-----------------------------------------------------------------------
  use dimens_fcm
  use exfunc
  !
  use stream
  use consta
  !use psf
  !use code
  !
  implicit none

  integer        :: natom,irepl
  real(chm_real) :: EIP
  real(chm_real) :: X(natom),Y(natom),Z(natom), DX(natom),DY(natom),DZ(natom)
  integer        :: mmynod,nnumnod

  real(chm_real) :: FR(3),GR(3),HR(3),AR(3),BR(3),DFR(3),DGR(3),DHR(3)
  real(chm_real) :: RA2,RB2,RG,RGR,RA2R,RB2R,RABR,CT,AP,ST,CA,SA
  real(chm_real) :: GAA,GBB,FG,HG,FGA,HGB
  real(chm_real) :: E,DF,RCICOS,RCISIN,RCIB,RCIC,exv
  INTEGER :: NWARN,NWARNX,IPHI,I,J,K,L,IC,II,ICID,ist

  real(chm_real),parameter :: RXMIN=0.005D0,RXMIN2=0.000025D0

  exv=zero
  if(mmbond_r(irepl)%n_mm_imph<=0) return
  NWARN=0
  NWARNX=0
  ist = mmynod+1
  do ii=ist,mmbond_r(irepl)%n_mm_imph,nnumnod  ! 1,n_mm_imph
     i =mmbond_r(irepl)%i_mm_imph(1,ii)
     j =mmbond_r(irepl)%i_mm_imph(2,ii)
     k =mmbond_r(irepl)%i_mm_imph(3,ii)
     l =mmbond_r(irepl)%i_mm_imph(4,ii)

     ICID  =mmbond_r(irepl)%IIMP_imph(ii)
     RCICOS=mmbond_r(irepl)%RIMP_mm(1,ii)
     RCISIN=mmbond_r(irepl)%RIMP_mm(2,ii)
     RCIB  =mmbond_r(irepl)%RIMP_mm(3,ii)
     RCIC  =mmbond_r(irepl)%RIMP_mm(4,ii)

#if KEY_OPLS==0
     if (ICID /= 0) then
        CALL WRNDIE(-3,'<EIPHIFS_mm>','Bad periodicity value for improper dihedral angles.')
     else
#endif
        ! F=Ri-Rj, G=Rj-Rk, H-Rl-Rk.
        FR(1)=X(I)-X(J)
        FR(2)=Y(I)-Y(J)
        FR(3)=Z(I)-Z(J)
        GR(1)=X(J)-X(K)
        GR(2)=Y(J)-Y(K)
        GR(3)=Z(J)-Z(K)
        HR(1)=X(L)-X(K)
        HR(2)=Y(L)-Y(K)
        HR(3)=Z(L)-Z(K)

        ! A=F^G, B=H^G.
        AR(1)=FR(2)*GR(3)-FR(3)*GR(2)
        AR(2)=FR(3)*GR(1)-FR(1)*GR(3)
        AR(3)=FR(1)*GR(2)-FR(2)*GR(1)
        BR(1)=HR(2)*GR(3)-HR(3)*GR(2)
        BR(2)=HR(3)*GR(1)-HR(1)*GR(3)
        BR(3)=HR(1)*GR(2)-HR(2)*GR(1)

        RA2=AR(1)*AR(1)+AR(2)*AR(2)+AR(3)*AR(3)
        RB2=BR(1)*BR(1)+BR(2)*BR(2)+BR(3)*BR(3)
        RG =SQRT(GR(1)*GR(1)+GR(2)*GR(2)+GR(3)*GR(3))
        ! Warnings have been simplified.
        if((RA2 <= RXMIN2).or.(RB2 <= RXMIN2).or.(RG <= RXMIN)) then
           NWARN=NWARN+1
           if((NWARN <= 5 .and. WRNLEV >= 5) .or. WRNLEV >= 6) WRITE(OUTU,20) ii,I,J,K,L
 20        FORMAT(' EIPHIFS_mm: WARNING.  dihedral',I5,' is almost linear.'/ &
                  ' derivatives may be affected for atoms:',4I5)
        else
           RGR = ONE/RG
           RA2R= ONE/RA2
           RB2R= ONE/RB2
           RABR= SQRT(RA2R*RB2R)
           ! CT=cos(phi)
           CT=(AR(1)*BR(1)+AR(2)*BR(2)+AR(3)*BR(3))*RABR

           ! ST=sin(phi), Note that sin(phi).G/|G|=B^A/(|A|.|B|)
           ! which can be simplify to sin(phi)=|G|H.A/(|A|.|B|)
           ST=RG*RABR*(AR(1)*HR(1)+AR(2)*HR(2)+AR(3)*HR(3))

           !calculate of cos(phi-phi0),sin(phi-phi0) and (Phi-Phi0).
           CA=CT*RCICOS+ST*RCISIN
           SA=ST*RCICOS-CT*RCISIN
           if (CA > PTONE ) then
              AP=ASIN(SA)
           else
              AP=SIGN(ACOS(MAX(CA,MINONE)),SA)

              ! Warning is now triggered at deltaphi=84.26...deg (used to be 90).
              NWARNX=NWARNX+1
              if((NWARNX <= 5 .and. WRNLEV >= 5) .or. WRNLEV >= 6) then
                 WRITE(OUTU,80) ii,AP*RADDEG,RCIB*RADDEG,I,J,K,L
 80              FORMAT(' EIPHIFS_mm: WARNING. bent improper torsion angle', &
                     ' is '//'far ','from minimum for;'/3X,' IPHI=',I5, &
                     '  with deltaPHI=',F9.4,' MIN=',F9.4,' ATOMS:',4I5)
              end if
           end if

           DF=RCIC*AP
           E=DF*AP
           DF=TWO*DF

           exv = exv+E

           ! Compute derivatives wrt catesian coordinates.
           !
           ! GAA=dE/dphi.|G|/A^2, GBB=dE/dphi.|G|/B^2, FG=F.G, HG=H.G
           ! FGA=dE/dphi*F.G/(|G|A^2), HGB=dE/dphi*H.G/(|G|B^2)
           FG=FR(1)*GR(1)+FR(2)*GR(2)+FR(3)*GR(3)
           HG=HR(1)*GR(1)+HR(2)*GR(2)+HR(3)*GR(3)
           RA2R=DF*RA2R
           RB2R=DF*RB2R
           FGA=FG*RA2R*RGR
           HGB=HG*RB2R*RGR
           GAA=RA2R*RG
           GBB=RB2R*RG

           ! DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.
           DFR(1)=-GAA*AR(1)
           DFR(2)=-GAA*AR(2)
           DFR(3)=-GAA*AR(3)
           DGR(1)= FGA*AR(1) - HGB*BR(1)
           DGR(2)= FGA*AR(2) - HGB*BR(2)
           DGR(3)= FGA*AR(3) - HGB*BR(3)
           DHR(1)= GBB*BR(1)
           DHR(2)= GBB*BR(2)
           DHR(3)= GBB*BR(3)

           ! Distribute over Ri.
           DX(I)=DX(I)+DFR(1)
           DY(I)=DY(I)+DFR(2)
           DZ(I)=DZ(I)+DFR(3)
           DX(J)=DX(J)-DFR(1)+DGR(1)
           DY(J)=DY(J)-DFR(2)+DGR(2)
           DZ(J)=DZ(J)-DFR(3)+DGR(3)
           DX(K)=DX(K)-DHR(1)-DGR(1)
           DY(K)=DY(K)-DHR(2)-DGR(2)
           DZ(K)=DZ(K)-DHR(3)-DGR(3)
           DX(L)=DX(L)+DHR(1)
           DY(L)=DY(L)+DHR(2)
           DZ(L)=DZ(L)+DHR(3)
        end if
#if KEY_OPLS==0
     end if
#endif 
  end do
  !
  EIP = exv
  return
  END SUBROUTINE EIPHIFS_mm

  SUBROUTINE EMBOND_morse(natom,irepl,EB,mmynod,nnumnod,X,Y,Z,DX,DY,DZ)
  !-----------------------------------------------------------------------
  !     calculates bond energies and forces based on the Morse potential.
  !
  !     For each pair, use Morse potential:
  !     U_ij (r_ij) = De*(1 - EXP(-alpha*(r_ij - r_ij_ref)))**2
  !-----------------------------------------------------------------------
  use dimens_fcm
  use exfunc
  use stream
  !use psf
  !use code
  !
  implicit none

  integer        :: natom,irepl
  real(chm_real) :: EB
  real(chm_real) :: X(natom),Y(natom),Z(natom), DX(natom),DY(natom),DZ(natom)
  integer :: nnumnod,mmynod

  real(chm_real) :: exv,De_val,alpha_val,ref_1,DXYZ(3)
  real(chm_real) :: RXYZ1(3),DXYZ1(3),Rij_1,dR_1,rexp1,E_1,dE1_dr,dF_1
  INTEGER :: ii,i1,j1,j2
  integer :: ist
  !
  !
  if(mmbond_r(irepl)%n_mmbond<=0) return
  exv= zero
  ist= mmynod+1
  do ii=ist,mmbond_r(irepl)%n_mmbond,nnumnod  ! 1,n_mm_bond
     i1=mmbond_r(irepl)%i_morse_bond(1,ii)
     j1=mmbond_r(irepl)%i_morse_bond(2,ii)
     De_val   = mmbond_r(irepl)%De_bond(ii)
     alpha_val= mmbond_r(irepl)%Alp_bond(ii)
     ref_1    = mmbond_r(irepl)%Rij_ref_bond(ii)

     ! energy of the bond
     RXYZ1(1)=X(I1)-X(J1)
     RXYZ1(2)=Y(I1)-Y(J1)
     RXYZ1(3)=Z(I1)-Z(J1)
     Rij_1   =SQRT(RXYZ1(1)*RXYZ1(1)+RXYZ1(2)*RXYZ1(2)+RXYZ1(3)*RXYZ1(3))
     dR_1    =Rij_1 - ref_1
     rexp1   =EXP(-alpha_val*dR_1)
     E_1     =De_val*(one-rexp1)*(one-rexp1)
     !
     ! gradient factor
     dE1_dr= two*De_val*(one-rexp1)*rexp1*alpha_val

     ! energy
     exv  = exv + E_1

     ! Gradient (i1 --- j1)
     DF_1    =dE1_dr/Rij_1
     DXYZ1(1)=RXYZ1(1)*DF_1
     DXYZ1(2)=RXYZ1(2)*DF_1
     DXYZ1(3)=RXYZ1(3)*DF_1
     !
     DX(I1)  =DX(I1)+DXYZ1(1)
     DY(I1)  =DY(I1)+DXYZ1(2)
     DZ(I1)  =DZ(I1)+DXYZ1(3)
     DX(J1)  =DX(J1)-DXYZ1(1)
     DY(J1)  =DY(J1)-DXYZ1(2)
     DZ(J1)  =DZ(J1)-DXYZ1(3)
  end do
  !
  EB = exv
  return
  END SUBROUTINE EMBOND_morse

  SUBROUTINE WRITE_mm_bond_info(iiunit,irepl)
  !
  ! Write mm bond/angle data into the specified file.
  ! The file can be read in.
  !
  use parallel
  implicit none

  integer :: iiunit,irepl
  integer :: ii,jj

#if KEY_PARALLEL==1
  if(mynod==0) then
#endif
     if(do_mm_bond) then
        write(iiunit,'(A4,I7)') 'BOND',mmbond_r(irepl)%n_mm_bond
        do ii=1,mmbond_r(irepl)%n_mm_bond
           write(iiunit,110) mmbond_r(irepl)%icb_bond(ii), &
                             mmbond_r(irepl)%i_mm_bond(1,ii),mmbond_r(irepl)%i_mm_bond(2,ii), &
                             mmbond_r(irepl)%RBND_mm(1,ii),mmbond_r(irepl)%RBND_mm(2,ii)
        end do
     end if
     if(do_mm_angl) then
        write(iiunit,'(A4,I7)') 'ANGL',mmbond_r(irepl)%n_mm_angl
        do ii=1,mmbond_r(irepl)%n_mm_angl
           write(iiunit,120) mmbond_r(irepl)%ict_angl(ii), &
                             mmbond_r(irepl)%i_mm_angl(1,ii),mmbond_r(irepl)%i_mm_angl(2,ii),mmbond_r(irepl)%i_mm_angl(3,ii), &
                             mmbond_r(irepl)%RANG_mm(1,II),mmbond_r(irepl)%RANG_mm(2,II), &
                             mmbond_r(irepl)%RANG_UB_mm(1,II),mmbond_r(irepl)%RANG_UB_mm(2,II)
        end do
     end if
     if(do_morse_mbond) then
        write(iiunit,'(A4,I7)') 'MBON',mmbond_r(irepl)%n_mmbond
        do ii=1,mmbond_r(irepl)%n_mmbond
           write(iiunit,130) mmbond_r(irepl)%i_morse_bond(1,ii),mmbond_r(irepl)%i_morse_bond(2,ii), &
                             mmbond_r(irepl)%De_bond(ii),mmbond_r(irepl)%Alp_bond(ii), &
                             mmbond_r(irepl)%Rij_ref_bond(ii)
        end do
     end if
#if KEY_PARALLEL==1
  end if
#endif
110 format(I4,2I10,2(1X,F15.10))
120 format(I4,3I10,4(1X,F15.10))
130 format(2I10,3(1X,F15.10))
  return
  END SUBROUTINE WRITE_mm_bond_info

  SUBROUTINE READ_mm_bond_info(iiunit,irepl,natom,q_mm_atoms,qcheck)
  !
  ! Read mm bond/angle data into the specified file.
  !
  use parallel
  use stream, only: outu,prnlev
  implicit none

  integer :: iiunit,irepl,natom
  logical :: qcheck,q_mm_atoms(natom)
  integer :: ii,jj,iline,itmp,jtmp
  integer :: ier=0
  character(LEN=4)  :: cbond

  !qcheck=.false.

  ! read is only done for bond and angle terms...
  if(do_mm_bond) then
#if KEY_PARALLEL==1
     if(mynod==0) then
#endif
        read(iiunit,'(A4,I7)',end=100) cbond(1:4),iline
#if KEY_PARALLEL==1
     end if
     call psnd4(iline,1)
#endif
     mmbond_r(irepl)%n_mm_bond = iline

     ! allocate memory
     if(allocated(mmbond_r(irepl)%i_mm_bond)) deallocate(mmbond_r(irepl)%i_mm_bond,stat=ier)
     if(allocated(mmbond_r(irepl)%icb_bond))  deallocate(mmbond_r(irepl)%icb_bond,stat=ier)
     if(allocated(mmbond_r(irepl)%RBND_mm))   deallocate(mmbond_r(irepl)%RBND_mm,stat=ier)
     allocate(mmbond_r(irepl)%i_mm_bond(2,mmbond_r(irepl)%n_mm_bond),stat=ier)
     allocate(mmbond_r(irepl)%icb_bond(mmbond_r(irepl)%n_mm_bond),stat=ier)
     allocate(mmbond_r(irepl)%RBND_mm(2,mmbond_r(irepl)%n_mm_bond),stat=ier)

#if KEY_PARALLEL==1
     if(mynod==0) then
#endif
        do ii=1,mmbond_r(irepl)%n_mm_bond
           read(iiunit,110,end=100) mmbond_r(irepl)%icb_bond(ii), &
                                    mmbond_r(irepl)%i_mm_bond(1,ii),mmbond_r(irepl)%i_mm_bond(2,ii), &
                                    mmbond_r(irepl)%RBND_mm(1,ii),mmbond_r(irepl)%RBND_mm(2,ii)
        end do
#if KEY_PARALLEL==1
     end if
     itmp= 2*mmbond_r(irepl)%n_mm_bond
     call psnd4m(mmbond_r(irepl)%icb_bond,mmbond_r(irepl)%n_mm_bond)
     call psnd4m(mmbond_r(irepl)%i_mm_bond,itmp)
     call psnd8m(mmbond_r(irepl)%RBND_mm,itmp)
#endif

     ! atoms included in the calculation
     do ii=1,mmbond_r(irepl)%n_mm_bond
        q_mm_atoms(mmbond_r(irepl)%i_mm_bond(1,ii)) =.true.
        q_mm_atoms(mmbond_r(irepl)%i_mm_bond(2,ii)) =.true.
     end do
     if(prnlev>=2) write(outu,130) 'Total number of bonds:',mmbond_r(irepl)%n_mm_bond,irepl
  end if  ! do_mm_bond

  !
  if(do_mm_angl) then
#if KEY_PARALLEL==1
     if(mynod==0) then
#endif
        read(iiunit,'(A4,I7)',end=100) cbond(1:4),iline
#if KEY_PARALLEL==1
     end if
     call psnd4(iline,1)
#endif
     mmbond_r(irepl)%n_mm_angl = iline

     ! allocate memory
     if(allocated(mmbond_r(irepl)%i_mm_angl))  deallocate(mmbond_r(irepl)%i_mm_angl,stat=ier)
     if(allocated(mmbond_r(irepl)%ict_angl))   deallocate(mmbond_r(irepl)%ict_angl,stat=ier)
     if(allocated(mmbond_r(irepl)%RANG_mm))    deallocate(mmbond_r(irepl)%RANG_mm,stat=ier)
     if(allocated(mmbond_r(irepl)%RANG_UB_mm)) deallocate(mmbond_r(irepl)%RANG_UB_mm,stat=ier)
     allocate(mmbond_r(irepl)%i_mm_angl(3,mmbond_r(irepl)%n_mm_angl),stat=ier)
     allocate(mmbond_r(irepl)%ict_angl(mmbond_r(irepl)%n_mm_angl),stat=ier)
     allocate(mmbond_r(irepl)%RANG_mm(2,mmbond_r(irepl)%n_mm_angl),stat=ier)
     allocate(mmbond_r(irepl)%RANG_UB_mm(2,mmbond_r(irepl)%n_mm_angl),stat=ier)

#if KEY_PARALLEL==1
     if(mynod==0) then
#endif
        do ii=1,mmbond_r(irepl)%n_mm_angl
            read(iiunit,120,end=100) mmbond_r(irepl)%ict_angl(ii), &
                                     mmbond_r(irepl)%i_mm_angl(1,ii),mmbond_r(irepl)%i_mm_angl(2,ii),mmbond_r(irepl)%i_mm_angl(3,ii), &
                                     mmbond_r(irepl)%RANG_mm(1,II),mmbond_r(irepl)%RANG_mm(2,II), &
                                     mmbond_r(irepl)%RANG_UB_mm(1,II),mmbond_r(irepl)%RANG_UB_mm(2,II)
        end do
#if KEY_PARALLEL==1
     end if
     itmp = 2*mmbond_r(irepl)%n_mm_angl
     jtmp = 3*mmbond_r(irepl)%n_mm_angl
     call psnd4m(mmbond_r(irepl)%ict_angl,mmbond_r(irepl)%n_mm_angl)
     call psnd4m(mmbond_r(irepl)%i_mm_angl,jtmp)
     call psnd8m(mmbond_r(irepl)%RANG_mm,itmp)
     call psnd8m(mmbond_r(irepl)%RANG_UB_mm,itmp)
#endif

     ! atoms included in the calculation
     do ii=1,mmbond_r(irepl)%n_mm_angl
        q_mm_atoms(mmbond_r(irepl)%i_mm_angl(1,ii)) =.true.
        q_mm_atoms(mmbond_r(irepl)%i_mm_angl(2,ii)) =.true.
        q_mm_atoms(mmbond_r(irepl)%i_mm_angl(3,ii)) =.true.
     end do
     if(prnlev>=2) write(outu,130) 'Total number of angles:',mmbond_r(irepl)%n_mm_angl,irepl
  end if  ! do_mm_angl

  !
  if(do_morse_mbond) then
#if KEY_PARALLEL==1
     if(mynod==0) then
#endif
        read(iiunit,'(A4,I7)',end=100) cbond(1:4),iline
#if KEY_PARALLEL==1
     end if
     call psnd4(iline,1)
#endif
     mmbond_r(irepl)%n_mmbond = iline

     if(mmbond_r(irepl)%n_mmbond<=0) then
        ! no Morse potential bonds
        do_morse_mbond =.false.
        call wrndie(-5,'<READ_mm_bond_info>','Read file error for MBONd selection.')
        return
     end if

     ! allocate memory
     if(allocated(mmbond_r(irepl)%i_morse_bond)) deallocate(mmbond_r(irepl)%i_morse_bond,stat=ier)
     if(allocated(mmbond_r(irepl)%De_bond))      deallocate(mmbond_r(irepl)%De_bond,stat=ier)
     if(allocated(mmbond_r(irepl)%Alp_bond))     deallocate(mmbond_r(irepl)%Alp_bond,stat=ier)
     if(allocated(mmbond_r(irepl)%Rij_ref_bond)) deallocate(mmbond_r(irepl)%Rij_ref_bond,stat=ier)
     allocate(mmbond_r(irepl)%i_morse_bond(2,mmbond_r(irepl)%n_mmbond),stat=ier)
     allocate(mmbond_r(irepl)%De_bond(mmbond_r(irepl)%n_mmbond),stat=ier)
     allocate(mmbond_r(irepl)%Alp_bond(mmbond_r(irepl)%n_mmbond),stat=ier)
     allocate(mmbond_r(irepl)%Rij_ref_bond(mmbond_r(irepl)%n_mmbond),stat=ier)

#if KEY_PARALLEL==1
     if(mynod==0) then
#endif
        do ii=1,mmbond_r(irepl)%n_mmbond
           read(iiunit,115,end=100) mmbond_r(irepl)%i_morse_bond(1,ii),mmbond_r(irepl)%i_morse_bond(2,ii), &
                                    mmbond_r(irepl)%De_bond(ii),mmbond_r(irepl)%Alp_bond(ii), &
                                    mmbond_r(irepl)%Rij_ref_bond(ii)
        end do
#if KEY_PARALLEL==1
     end if
     itmp= 2*mmbond_r(irepl)%n_mmbond
     call psnd4m(mmbond_r(irepl)%i_morse_bond,itmp)
     call psnd8m(mmbond_r(irepl)%De_bond,mmbond_r(irepl)%n_mmbond)
     call psnd8m(mmbond_r(irepl)%Alp_bond,mmbond_r(irepl)%n_mmbond)
     call psnd8m(mmbond_r(irepl)%Rij_ref_bond,mmbond_r(irepl)%n_mmbond)
#endif

     ! atoms included in the calculation
     do ii=1,mmbond_r(irepl)%n_mmbond
        q_mm_atoms(mmbond_r(irepl)%i_morse_bond(1,ii)) =.true.
        q_mm_atoms(mmbond_r(irepl)%i_morse_bond(2,ii)) =.true.
     end do
     if(prnlev>=2) write(outu,130) 'Total number of Morse bonds:',mmbond_r(irepl)%n_mmbond
  end if  ! do_morse_mbond
  ! normal exit
  !qcheck =.true.
110 format(I4,2I10,2(1X,F15.10))
115 format(2I10,3(1X,F15.10))
120 format(I4,3I10,4(1X,F15.10))
130 format('READ_mm_bond_info>',A,I7,I3)
  return

100 continue
  ! error in the file
  call wrndie(-5,'<READ_mm_bond_info>','READ file error.')
  qcheck =.false.
  return
  END SUBROUTINE READ_mm_bond_info
  !=====================================================================
!!!#endif /*mts*/
#endif /*mndo97*/
end module mmbonded_mod
! end
