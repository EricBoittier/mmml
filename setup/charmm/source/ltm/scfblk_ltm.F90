module scfblk
  use chm_kinds
  use dimens_fcm
  !CHARMM Element source/fcm/scfblk.fcm 1.1
#if KEY_QUANTUM==1 /*scfblkfcm*/
  !
  !     Here the options and variables for the SCF/CI calculation are kept.
  !
  integer, parameter :: LENSBI = 2, LENSBL = 7, LENSBR = 5
  !
  logical, save :: ALLCON, CAMKIN, NEWDG, OKNEWD, OKPULY, QFIRST, UHF  
  integer, save :: IFILL, ITRMAX                                       
  real(chm_real), save :: BSHIFT, PL, PLB, PLTEST, SCFCRT              
  !
  logical, save :: CI                                                  
  integer, save :: LROOT, NMOS, NCIS                                   
  !
  !     Heap pointers to the arrays needed by the MOPAC SCF and CI
  !     programs are stored here.
  !
  integer, parameter :: LENDEN = 11
  !
  ! for matrix information...
  real(chm_real),dimension(:),allocatable,save :: H_matrix                         
  real(chm_real),dimension(:),allocatable,save :: CALPHA,CBETA,EIGSA,EIGSB,FA,FB,& 
       PDENS,PDENSA,PDENSB,PAOLD,PBOLD  
  real(chm_real),dimension(:),allocatable,save :: &
       H1PERT,H2PERT,H0GAS,PGAS,PENZYM  ! moved from quantm_ltm.src

  ! for scf iteration for qm-qm and qm-mm pairs.
  integer, dimension(4),save :: nlen_check=-1     ! check for array setup or check
  integer,              save :: nlen_jnbl =-1     !
  integer, allocatable,dimension(:),save :: jnbl_qmmm
  real(chm_real),allocatable,dimension(:),save :: dxe1bm
  real(chm_real),allocatable,dimension(:),save :: dye1bm
  real(chm_real),allocatable,dimension(:),save :: dze1bm
  real(chm_real),allocatable,dimension(:),save :: dxe1bq
  real(chm_real),allocatable,dimension(:),save :: dye1bq
  real(chm_real),allocatable,dimension(:),save :: dze1bq
  real(chm_real),allocatable,dimension(:),save :: e1bdxs
  real(chm_real),allocatable,dimension(:),save :: e1bdys
  real(chm_real),allocatable,dimension(:),save :: e1bdzs
  real(chm_real),allocatable,dimension(:),save :: dxe1xm
  real(chm_real),allocatable,dimension(:),save :: dye1xm
  real(chm_real),allocatable,dimension(:),save :: dze1xm
  real(chm_real),allocatable,dimension(:),save :: dxe1xq
  real(chm_real),allocatable,dimension(:),save :: dye1xq
  real(chm_real),allocatable,dimension(:),save :: dze1xq
  real(chm_real),allocatable,dimension(:),save :: e1xdxs
  real(chm_real),allocatable,dimension(:),save :: e1xdys
  real(chm_real),allocatable,dimension(:),save :: e1xdzs

  ! for working arrays for scf iteraction.
  integer, save :: LINEAR_old=-1, &
       LPULAY_old=-1, &
       LPULY2_old=-1, &
       NORBS2_old=-1
  real(chm_real),allocatable,dimension(:),save :: AR1_keep,AR2_keep,AR3_keep,AR4_keep,   &
       BR1_keep,BR2_keep,BR3_keep,BR4_keep,   &
       POLD1_keep,POLD2_keep,POLD3_keep,      &
       PBOLD1_keep,PBOLD2_keep,PBOLD3_keep,   &
       WORK_keep,FHB_keep,CHB_keep,PAHB_keep
  ! For UHF-GHO and density damping ... PJ 12/2002
  real(chm_real),allocatable,dimension(:),save :: FBHB_keep,CBHB_keep,PBHB_keep,         & 
       PAPRE_keep,PBPRE_keep

  !*************************************************************************************
contains                  !CONTAINS
  !*************************************************************************************

  !-----------------------------------------------------------------------
  subroutine setup_qm_arrays(LINEAR,LENMAT,LENORB,LENPAB,QMPERT_local,QDECOM_local)
    use chm_kinds
    use memory
    implicit none
    integer :: LINEAR,LENMAT,LENORB,LENPAB
    logical :: QMPERT_local,QDECOM_local

    !     deallocate first
    call reset_qm_arrays(LINEAR,LENMAT,LENORB,LENPAB)

    !     now allocate arrays
    call chmalloc('scfblk_ltm.src','setup_qm_arrays','H_matrix',LINEAR,crl=H_matrix)

    call chmalloc('scfblk_ltm.src','setup_qm_arrays','CALPHA',LENMAT,crl=CALPHA)
    call chmalloc('scfblk_ltm.src','setup_qm_arrays','CBETA',LENMAT,crl=CBETA)
    call chmalloc('scfblk_ltm.src','setup_qm_arrays','EIGSA',LENORB,crl=EIGSA)
    call chmalloc('scfblk_ltm.src','setup_qm_arrays','EIGSB',LENORB,crl=EIGSB)

    call chmalloc('scfblk_ltm.src','setup_qm_arrays','FA',LENMAT,crl=FA)
    call chmalloc('scfblk_ltm.src','setup_qm_arrays','FB',LENMAT,crl=FB)

    call chmalloc('scfblk_ltm.src','setup_qm_arrays','PDENS',LENPAB,crl=PDENS)
    call chmalloc('scfblk_ltm.src','setup_qm_arrays','PDENSA',LENPAB,crl=PDENSA)
    call chmalloc('scfblk_ltm.src','setup_qm_arrays','PDENSB',LENPAB,crl=PDENSB)
    call chmalloc('scfblk_ltm.src','setup_qm_arrays','PAOLD',LENPAB,crl=PAOLD)
    call chmalloc('scfblk_ltm.src','setup_qm_arrays','PBOLD',LENPAB,crl=PBOLD)

    if(QMPERT_local) then
       call chmalloc('scfblk_ltm.src','setup_qm_arrays','H1PERT',LENPAB,crl=H1PERT)
       call chmalloc('scfblk_ltm.src','setup_qm_arrays','H2PERT',LENPAB,crl=H2PERT)
       call chmalloc('scfblk_ltm.src','setup_qm_arrays','H0GAS',LENPAB,crl=H0GAS)
    end if
    if(QDECOM_local) then
       call chmalloc('scfblk_ltm.src','setup_qm_arrays','PGAS',LENPAB,crl=PGAS)
       call chmalloc('scfblk_ltm.src','setup_qm_arrays','PENZYM',LENPAB,crl=PENZYM)
       call chmalloc('scfblk_ltm.src','setup_qm_arrays','H1PERT',LENPAB,crl=H1PERT)
       call chmalloc('scfblk_ltm.src','setup_qm_arrays','H0GAS',LENPAB,crl=H0GAS)
    end if

    return
  END subroutine setup_qm_arrays

  subroutine setup_scf_arrays(nlen,kk,INDX1E,NMAX,q_exclusion)
    use chm_kinds
    use memory
    implicit none
    integer :: nlen,kk
    integer,optional :: INDX1E,NMAX 
    logical :: q_exclusion
    integer :: ii,jj,ij

    ii=0
    jj=0

    !     now allocate arrays.
    if(.not.q_exclusion ) then
       !        check to see whether to do this.
       if(nlen_check(1).ne.nlen .or. nlen_check(2).ne.INDX1E &
            .or. nlen_check(3).ne.NMAX) then
          if(present(INDX1E)) ii=INDX1E
          if(present(NMAX))   jj=NMAX

          !           deallocation first
          call reset_scf_arrays(nlen,kk,ii,jj,q_exclusion,.false.)

          call chmalloc('scfblk_ltm.src','setup_scf_arrays','dxe1bm',nlen,crl=dxe1bm)
          call chmalloc('scfblk_ltm.src','setup_scf_arrays','dye1bm',nlen,crl=dye1bm)
          call chmalloc('scfblk_ltm.src','setup_scf_arrays','dze1bm',nlen,crl=dze1bm)
          !JG Only need to claim needed memory
          call chmalloc('scfblk_ltm.src','setup_scf_arrays','dxe1bq',INDX1E,crl=dxe1bq)
          call chmalloc('scfblk_ltm.src','setup_scf_arrays','dye1bq',INDX1E,crl=dye1bq)
          call chmalloc('scfblk_ltm.src','setup_scf_arrays','dze1bq',INDX1E,crl=dze1bq)
          call chmalloc('scfblk_ltm.src','setup_scf_arrays','e1bdxs',INDX1E*NMAX,crl=e1bdxs)
          call chmalloc('scfblk_ltm.src','setup_scf_arrays','e1bdys',INDX1E*NMAX,crl=e1bdys)
          call chmalloc('scfblk_ltm.src','setup_scf_arrays','e1bdzs',INDX1E*NMAX,crl=e1bdzs)

          nlen_check(1)=nlen
          nlen_check(2)=INDX1E
          nlen_check(3)=NMAX
       end if
    else
       if(nlen_check(4).ne.nlen) then             ! has to setup these arrays. 
          !           deallocation first
          call reset_scf_arrays(nlen,kk,ii,jj,q_exclusion,.false.)

          call chmalloc('scfblk_ltm.src','setup_scf_arrays','dxe1xm',nlen,crl=dxe1xm)
          call chmalloc('scfblk_ltm.src','setup_scf_arrays','dye1xm',nlen,crl=dye1xm)
          call chmalloc('scfblk_ltm.src','setup_scf_arrays','dze1xm',nlen,crl=dze1xm)
          call chmalloc('scfblk_ltm.src','setup_scf_arrays','dxe1xq',nlen,crl=dxe1xq)
          call chmalloc('scfblk_ltm.src','setup_scf_arrays','dye1xq',nlen,crl=dye1xq)
          call chmalloc('scfblk_ltm.src','setup_scf_arrays','dze1xq',nlen,crl=dze1xq)
          call chmalloc('scfblk_ltm.src','setup_scf_arrays','e1xdxs',nlen,crl=e1xdxs)
          call chmalloc('scfblk_ltm.src','setup_scf_arrays','e1xdys',nlen,crl=e1xdys)
          call chmalloc('scfblk_ltm.src','setup_scf_arrays','e1xdzs',nlen,crl=e1xdzs)

          nlen_check(4)=nlen
       end if
    end if

    if(nlen_jnbl.ne.kk) then
       if(allocated(jnbl_qmmm)) then
          ij=size(jnbl_qmmm)
          call reset_scf_arrays(nlen,ij,ii,jj,q_exclusion,.true.)
       end if
       call chmalloc('scfblk_ltm.src','setup_scf_arrays','jnbl_qmmm',kk,intg=jnbl_qmmm)
       nlen_jnbl = kk
    end if

    return
  End subroutine setup_scf_arrays

  subroutine reset_qm_arrays(LINEAR,LENMAT,LENORB,LENPAB)
    use chm_kinds
    use memory

    implicit none
    integer :: LINEAR,LENMAT,LENORB,LENPAB

    if(allocated(H_matrix)) &
         call chmdealloc('scfblk_ltm.src','setup_qm_arrays','H_matrix',LINEAR,crl=H_matrix)

    if(allocated(CALPHA)) &
         call chmdealloc('scfblk_ltm.src','reset_qm_arrays','CALPHA',LENMAT,crl=CALPHA)
    if(allocated(CBETA)) &
         call chmdealloc('scfblk_ltm.src','reset_qm_arrays','CBETA',LENMAT,crl=CBETA)
    if(allocated(EIGSA)) &
         call chmdealloc('scfblk_ltm.src','reset_qm_arrays','EIGSA',LENORB,crl=EIGSA)
    if(allocated(EIGSB)) &
         call chmdealloc('scfblk_ltm.src','reset_qm_arrays','EIGSB',LENORB,crl=EIGSB)

    if(allocated(FA)) call chmdealloc('scfblk_ltm.src','reset_qm_arrays','FA',LENMAT,crl=FA)
    if(allocated(FB)) call chmdealloc('scfblk_ltm.src','reset_qm_arrays','FB',LENMAT,crl=FB)

    if(allocated(PDENS)) &
         call chmdealloc('scfblk_ltm.src','reset_qm_arrays','PDENS',LENPAB,crl=PDENS)
    if(allocated(PDENSA)) &
         call chmdealloc('scfblk_ltm.src','reset_qm_arrays','PDENSA',LENPAB,crl=PDENSA)
    if(allocated(PDENSB)) &
         call chmdealloc('scfblk_ltm.src','reset_qm_arrays','PDENSB',LENPAB,crl=PDENSB)
    if(allocated(PAOLD)) &
         call chmdealloc('scfblk_ltm.src','reset_qm_arrays','PAOLD',LENPAB,crl=PAOLD)
    if(allocated(PBOLD)) &
         call chmdealloc('scfblk_ltm.src','reset_qm_arrays','PBOLD',LENPAB,crl=PBOLD)

    if(allocated(H1PERT)) &
         call chmdealloc('scfblk_ltm.src','reset_qm_arrays','H1PERT',LENPAB,crl=H1PERT)
    if(allocated(H2PERT)) &
         call chmdealloc('scfblk_ltm.src','reset_qm_arrays','H2PERT',LENPAB,crl=H2PERT)
    if(allocated(H0GAS)) &
         call chmdealloc('scfblk_ltm.src','reset_qm_arrays','H0GAS',LENPAB,crl=H0GAS)
    if(allocated(PGAS)) &
         call chmdealloc('scfblk_ltm.src','reset_qm_arrays','PGAS',LENPAB,crl=PGAS)
    if(allocated(PENZYM)) &
         call chmdealloc('scfblk_ltm.src','reset_qm_arrays','PENZYM',LENPAB,crl=PENZYM)

    return
  END subroutine reset_qm_arrays

  subroutine reset_scf_arrays(nlen,kk,INDX1E,NMAX,q_exclusion,q_jnbl)
    use chm_kinds
    use memory
    implicit none
    integer :: nlen,kk,INDX1E,NMAX
    logical :: q_exclusion,q_jnbl

    !     deallocate arrays.
    if(q_jnbl) then
       if(allocated(jnbl_qmmm)) &
            call chmdealloc('scfblk_ltm.src','reset_scf_arrays', &
            'jnbl_qmmm',kk,intg=jnbl_qmmm)
       return
    end if
    if(.not.q_exclusion ) then
       if(allocated(dxe1bm)) &
            call chmdealloc('scfblk_ltm.src','reset_scf_arrays','dxe1bm',nlen,crl=dxe1bm)
       if(allocated(dye1bm)) &
            call chmdealloc('scfblk_ltm.src','reset_scf_arrays','dye1bm',nlen,crl=dye1bm)
       if(allocated(dze1bm)) &
            call chmdealloc('scfblk_ltm.src','reset_scf_arrays','dze1bm',nlen,crl=dze1bm)
       !JG Only need to claim needed memory
       if(allocated(dxe1bq)) &
            call chmdealloc('scfblk_ltm.src','reset_scf_arrays','dxe1bq',INDX1E,crl=dxe1bq)
       if(allocated(dye1bq)) &
            call chmdealloc('scfblk_ltm.src','reset_scf_arrays','dye1bq',INDX1E,crl=dye1bq)
       if(allocated(dze1bq)) &
            call chmdealloc('scfblk_ltm.src','reset_scf_arrays','dze1bq',INDX1E,crl=dze1bq)
       if(allocated(e1bdxs)) &
            call chmdealloc('scfblk_ltm.src','reset_scf_arrays', &
            'e1bdxs',INDX1E*NMAX,crl=e1bdxs)
       if(allocated(e1bdys)) &
            call chmdealloc('scfblk_ltm.src','reset_scf_arrays', &
            'e1bdys',INDX1E*NMAX,crl=e1bdys)
       if(allocated(e1bdzs)) &
            call chmdealloc('scfblk_ltm.src','reset_scf_arrays', &
            'e1bdzs',INDX1E*NMAX,crl=e1bdzs)

    else
       if(allocated(dxe1xm)) &
            call chmdealloc('scfblk_ltm.src','reset_scf_arrays','dxe1xm',nlen,crl=dxe1xm)
       if(allocated(dye1xm)) &
            call chmdealloc('scfblk_ltm.src','reset_scf_arrays','dye1xm',nlen,crl=dye1xm)
       if(allocated(dze1xm)) &
            call chmdealloc('scfblk_ltm.src','reset_scf_arrays','dze1xm',nlen,crl=dze1xm)
       if(allocated(dxe1xq)) &
            call chmdealloc('scfblk_ltm.src','reset_scf_arrays','dxe1xq',nlen,crl=dxe1xq)
       if(allocated(dye1xq)) &
            call chmdealloc('scfblk_ltm.src','reset_scf_arrays','dye1xq',nlen,crl=dye1xq)
       if(allocated(dze1xq)) &
            call chmdealloc('scfblk_ltm.src','reset_scf_arrays','dze1xq',nlen,crl=dze1xq)
       if(allocated(e1xdxs)) &
            call chmdealloc('scfblk_ltm.src','reset_scf_arrays','e1xdxs',nlen,crl=e1xdxs)
       if(allocated(e1xdys)) &
            call chmdealloc('scfblk_ltm.src','reset_scf_arrays','e1xdys',nlen,crl=e1xdys)
       if(allocated(e1xdzs)) &
            call chmdealloc('scfblk_ltm.src','reset_scf_arrays','e1xdzs',nlen,crl=e1xdzs)
    end if

    return
  End subroutine reset_scf_arrays

  !
#endif /* (scfblkfcm)*/
  !
end module scfblk

