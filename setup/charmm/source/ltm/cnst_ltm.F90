module cnst_fcm
  use chm_kinds
  use dimens_fcm
  implicit none
  character(len=*),private,parameter :: file_name="cnst_ltm.src"
  !
  !     The Harmonic Restraints
  !
  !     Purpose:
  !
  !     To store harmonic restraints on atoms and torsions angles
  !     arbitrarily formed on four atoms.
  !
  !     Documentation information file: doc/cons.doc
  !
  !     CHARMM commands:  CONS { HARMonic ... }
  !                            { DIHEdral ... }
  !                            { CLDH         }
  !                            { IC       ... }
  !                            { DROPlet  ... }
  !                            { HMCM     ... }
  !                            { PATH     ... }
  !
  !     Note: Data for restraint/constraint commands; CONS FIX, CONS RMSD,
  !           CONS CLEAR, SHAKE, NOE, RESD, PULL, RMSD, and DMCO are not
  !           contained in this fcm file.
  !
  !     Variable   Purpose
  !
  !
  !
  !  DIHEDRAL RESTRAINTS
  !     NCSPHI     Number of dihedral restraints
  !     CCSB       Zero energy dihedral angle
  !     CCSCOS     Cos(CCSB) for CCSD=0, Cos(CCSD*CCSB+PI) for CCSD.ne.0
  !     CCSSIN     Sin(CCSB) for CCSD=0, Sin(CCSD*CCSB+PI) for CCSD.ne.0
  !     CCSC       Force constant for dihedral
  !     CCSD       Periodicity for restrained dihedral
  !     CCSW       Half width of flat bottom improper dihedral.
  !     ICCS       Fake index into CCSB and CCSC for energy routines.
  !                ICCS(I)=I
  !     ICS        First atom of dihedral restraint
  !     JCS        Second atom of dihedral restraint
  !     KCS        Third atom of dihedral restraint
  !     LCS        Fourth atom of dihedral restraint
  !
  !  HARMONIC RESTRAINTS
  !     QCNSTR        Flag meaning some atom has a non-zero restraint
  !     KCNSTR(NATOM) Force constant of atomic restraint
  !     KCEXPN(nset)  Exponent of the restraint for each set (default 2)
  !     REFX(NATOM)   X component of reference coordinates
  !     REFY(NATOM)   Y component of reference coordinates
  !     REFZ(NATOM    Z component of reference coordinates
  !     XHSCALE(nset) X component global scale factor (default 1.0)
  !     YHSCALE(nset) Y component global scale factor (default 1.0)
  !     ZHSCALE(nset) Z component global scale factor (default 1.0)
  !     NUMHSETS      The total number of active restraint sets
  !     NUMPCA        The total number of PCA-based restraints
  !     TYPHSET(nset)     The type of each restraint set
  !                   0 - direct with reference (no best fit)
  !                   1 - best fit rotation and translation with reference
  !                  >1 - best fit with alternate set (other than 1)
  !     IHSET(NATOM)  The set number for each atom
  !     QHNORT(nset)  Flag to suppress bestfit rotation
  !     QHNOTR(nset)  Flag to suppress bestfit translation
  !
  ! Future: delete fcm data
  !      IHSET(MAXA),KCNSTR(MAXA), REFX(MAXA), REFY(MAXA), REFZ(MAXA)
  ! Future: Add new fcm pointer data
  !      NHSET(MAXHSET)   -->  NPAIR            -->  NPAIR
  !      IHSET(MAXHSET)   -->    ATOMPR(2,NPAIR)
  !      KCNSTR(MAXHSET)  -->    BMASS(NPAIR)
  !      REFXYZ(MAXHSET)  -->    VB(3,NPAIR)
  !
  !
  !  INTERNAL COORDINATE RESTRAINTS
  !     LCIC       Internal coordinate restraint are used (if true)
  !     CCBIC      Force constant used for IC bonds
  !     CCTIC      Force constant used for IC angles
  !     CCPIC      Force constant used for IC dihedrals
  !     CCIIC      Force constant used for IC dihedrals
  !                 (see IC tables for more info)
  !     KBEXPN     Exponent for BOND potential (default 2)
  !     LUPPER     Flag .TRUE. if BOND restraints are to be applied
  !                 only when r>r0 (ie r0 is upper limit, eg for NOEs).
  !
  !  QUARTIC DROPLET RESTRAINTS (CDRO)
  !     QQCNST     Flag indicating that droplet potential is in use
  !     KQCNST     Force constant for the droplet potential
  !     KQEXPN     Exponent for the droplet potential (default 4)
  !     LQMASS     Logical flag for mass weighting of the droplet energy
  !
  !     FBETA      Langevin dynamics friction coefficient
  !


  integer :: ncsphi, kqexpn, kbexpn, numhsets, numpca
  integer,allocatable,dimension(:) :: ihset
  integer, allocatable,dimension(:) :: ics, jcs, kcs, lcs,iccs, ccsd, kcexpn, typhset

  real(chm_real) :: ccbic, cctic, ccpic, cciic, kqcnst
  real(chm_real),allocatable,dimension(:) :: kcnstr, refx, refy, refz,fbeta
  real(chm_real),allocatable,dimension(:,:) :: pcax, pcay, pcaz
  real(chm_real), allocatable,dimension(:) :: ccsc,ccsb,ccsw,ccscos,ccssin, &
       xhscale,yhscale, zhscale

  logical :: qcnstr, lcic, qqcnst, lqmass, lupper, &
       cnst_is_allocated = .false.
  logical, allocatable,dimension(:) :: qhnort, qhnotr

  !
  !     MAXNPATHS     THE MAXIMUM NUMBER OF PATHS
  !     MAXNATMS      THE MAXIMUM NUMBER OF ATOMS USED TO DEFINE A PATH
  !     MAXNCOM       THE MAXIMUM NUMBER OF COM POINTS PROJECTED ON A GIVEN PATH
  !     MAXCOMATMS    THE MAXIMUM NUMBER OF ATOMS MAKING UP ONE COM POINT
  !
  !     PATHNCOUNT    Keeps track of the number of spline interpolated paths
  !     PATHN         Keeps track of the current path in use
  !     PATHNATMS     Keeps track of the number of atoms in each path
  !     PATHNCOM      Keeps track of the number of COM in each path
  !     PATHNCOMATMS  Keeps track of the number atoms for each COM
  !     PATHINDX      Path index (1-dimensional) - contains COM atoms and path
  !                   number info using PATHCOMI below as an index
  !     PATHCOMLYN    Counts the number of lines for PATHINDX
  !     PATHCOMI      COM index - a 2-dimensional array that keeps track of
  !                   line location for PATHINDX
  !     PATHCOMILEN   COM length index - a 2-dimensional array that keeps track
  !                   of the number of lines to read after PATHCOMI returns the
  !                   line location information in PATHINDX
  !     PATHIOSTAT    Tracks IOSTAT for UNIT 77 when it is opened
  !     PATHFILEX     COORDINATES TO DEFINE SPLINE POINTS FROM FILE
  !     PATHFILEY         File format should adhere to:
  !     PATHFILEZ         3F8.3
  !     PATHA         Coefficients for cubic spline interpolation
  !     PATHB             At^3 + Bt^2 + Ct +D
  !     PATHC             1st dimension has 1=X, 2=Y, 3=Z
  !     PATHD             2nd dimension is the spline piece (starting with 1)
  !                       3rd dimension is the associated path
  !     PATHCOMX      Temporary real values for calculating COM for a given set
  !     PATHCOMY      of atoms
  !     PATHCOMZ
  !     PATHTINI      Initial reference value for T for a given COM
  !     PATHTPRIME    Initial reference value for T for a given COM
  !                   after the first window
  !     PATHTZERO     T-zero is the location of where the projection of the
  !                   COM should move towards
  !     PATHK         The force constants (A 2-dimensional array)
  !     PATHTOL       Tolerance level allowing the projection value, TVAL,
  !                   to be outside of zero and one.
  !     PATHPDX       Partial derivative part of force wrt X
  !     PATHPDY       Partial derivative part of force wrt Y
  !     PATHPDZ       Partial derivative part of force wrt Z
  !     PATHROOTTYPE  Used to keep track of the type of cubic root calculated
  !     CPATHE        Energy returned to CHARMM for CPATH
  !     PROJUNIT      Unit where COM projections are to be written for each
  !                   trajectory step
  !
#if KEY_CPATH==1
  integer,parameter :: MAXNPATHS=10
  integer,parameter :: MAXNATMS=1000
  integer,parameter :: MAXNCOM=2000
  integer,parameter :: MAXCOMATMS=500

  integer :: PATHNCOUNT,PATHN,PATHROOTTYPE, &
       PATHNCOMATMS,PATHCOMLYN, &
       PATHIOSTAT,PROJUNIT
  !integer :: PATHNATMS(MAXNPATHS),PATHNCOM(MAXNPATHS),PATHCOMI(MAXNPATHS,MAXNCOM), &
  !     PATHCOMILEN(MAXNPATHS,MAXNCOM),PATHINDX(MAXNCOM*MAXCOMATMS)

  integer, allocatable, dimension(:) :: PATHNATMS,PATHNCOM, PATHINDX
  integer, allocatable, dimension(:,:) :: PATHCOMI, PATHCOMILEN

  !real(chm_real) :: PATHA(3,MAXNATMS,MAXNPATHS), PATHB(3,MAXNATMS,MAXNPATHS),PATHC(3,MAXNATMS,MAXNPATHS), &
  !     PATHD(3,MAXNATMS,MAXNPATHS),PATHTINI(MAXNPATHS,MAXNCOM),PATHTPRIME(MAXNPATHS,MAXNCOM), &
  !     PATHTZERO(MAXNPATHS,MAXNCOM),PATHK(MAXNPATHS,MAXNCOM)

  real(chm_real), allocatable, dimension(:,:,:) :: PATHA, PATHB,PATHC,PATHD

  real(chm_real), allocatable, dimension(:,:) :: PATHTINI,PATHTPRIME,PATHTZERO,PATHK

  real(chm_real) :: PATHCOMX,PATHCOMY,PATHCOMZ,PATHTOL, &
       PATHPDX,PATHPDY,PATHPDZ,CPATHE, &
       PATHFILEX,PATHFILEY,PATHFILEZ
#endif


contains

  subroutine cnst_init()
    !=======================================================================
    ! CNST.FCM
    !     harmonic constraints
    qcnstr=.false.
    cnst_is_allocated = .false.
    numhsets=0
    numpca = 0
    !     internal coordinate constraints
    lcic=.false.
    ccbic=0.0
    cctic=0.0
    ccpic=0.0
    cciic=0.0
    !     dihedral constraints
    ncsphi=0
    !     droplet constraints
    qqcnst=.false.
    lqmass=.false.
    kqcnst=0.0
    kqexpn=4
    return
  end subroutine cnst_init

#if KEY_CPATH==1
  subroutine cpath_init()

    call cpath_alloc
    pathnatms(1:maxnpaths)=0
    pathncom (1:maxnpaths)=0
    pathn=0
    pathncomatms=0
    pathcomlyn=0
    pathtol=1e-05
    cpathe=0
    projunit=-1
    return
  end subroutine cpath_init

  subroutine cpath_alloc()
    use memory
    implicit none
    character(len=*), parameter :: routine_name="cpath_alloc"

    call chmalloc(file_name,routine_name,'PATHNATMS',MAXNPATHS,intg=PATHNATMS)
    call chmalloc(file_name,routine_name,'PATHNCOM',MAXNPATHS,intg=PATHNCOM)
    call chmalloc(file_name,routine_name,'PATHCOMI',MAXNPATHS,MAXNCOM,intg=PATHCOMI)
    call chmalloc(file_name,routine_name,'PATHCOMILEN',MAXNPATHS,MAXNCOM,intg=PATHCOMILEN)
    call chmalloc(file_name,routine_name,'PATHINDX',MAXNCOM*MAXCOMATMS,intg=PATHINDX)

    call chmalloc(file_name,routine_name,'PATHA',3,MAXNATMS,MAXNPATHS,crl=PATHA)
    call chmalloc(file_name,routine_name,' PATHB',3,MAXNATMS,MAXNPATHS,crl= PATHB)
    call chmalloc(file_name,routine_name,'PATHC',3,MAXNATMS,MAXNPATHS,crl=PATHC)
    call chmalloc(file_name,routine_name,'PATHD',3,MAXNATMS,MAXNPATHS,crl=PATHD)
    call chmalloc(file_name,routine_name,'PATHTINI',MAXNPATHS,MAXNCOM,crl=PATHTINI)
    call chmalloc(file_name,routine_name,'PATHTPRIME',MAXNPATHS,MAXNCOM,crl=PATHTPRIME)
    call chmalloc(file_name,routine_name,'PATHTZERO',MAXNPATHS,MAXNCOM,crl=PATHTZERO)
    call chmalloc(file_name,routine_name,'PATHK',MAXNPATHS,MAXNCOM,crl=PATHK)

  end subroutine cpath_alloc

  subroutine cpath_dealloc()
    use memory
    implicit none
    character(len=*), parameter :: routine_name="cpath_dealloc"

    call chmdealloc(file_name,routine_name,'PATHNATMS',MAXNPATHS,intg=PATHNATMS)
    call chmdealloc(file_name,routine_name,'PATHNCOM',MAXNPATHS,intg=PATHNCOM)
    call chmdealloc(file_name,routine_name,'PATHCOMI',MAXNPATHS,MAXNCOM,intg=PATHCOMI)
    call chmdealloc(file_name,routine_name,'PATHCOMILEN',MAXNPATHS,MAXNCOM,intg=PATHCOMILEN)
    call chmdealloc(file_name,routine_name,'PATHINDX',MAXNCOM*MAXCOMATMS,intg=PATHINDX)

    call chmdealloc(file_name,routine_name,'PATHA',3,MAXNATMS,MAXNPATHS,crl=PATHA)
    call chmdealloc(file_name,routine_name,' PATHB',3,MAXNATMS,MAXNPATHS,crl= PATHB)
    call chmdealloc(file_name,routine_name,'PATHC',3,MAXNATMS,MAXNPATHS,crl=PATHC)
    call chmdealloc(file_name,routine_name,'PATHD',3,MAXNATMS,MAXNPATHS,crl=PATHD)
    call chmdealloc(file_name,routine_name,'PATHTINI',MAXNPATHS,MAXNCOM,crl=PATHTINI)
    call chmdealloc(file_name,routine_name,'PATHTPRIME',MAXNPATHS,MAXNCOM,crl=PATHTPRIME)
    call chmdealloc(file_name,routine_name,'PATHTZERO',MAXNPATHS,MAXNCOM,crl=PATHTZERO)
    call chmdealloc(file_name,routine_name,'PATHK',MAXNPATHS,MAXNCOM,crl=PATHK)

  end subroutine cpath_dealloc
#endif

  subroutine allocate_cnst(natom)
    use number, only: zero, anum
    use memory, only: chmalloc, chmdealloc, chmrealloc

    implicit none

    ! args
    integer, intent(in) :: natom

    ! locals
    character(len=*), parameter :: routine_name = 'allocate_cnst'
    integer :: old_size, new_start

    if (allocated(ihset)) then
       old_size = size(ihset)
       if (natom /= old_size) then
          if (qcnstr) then
             CALL WRNDIE(0,'<ALLOCATE_CNST>', &
                  'NATOM changed while CONS HARM is on.'//CHAR(10)&
                  //'      It is safer to turn off CONS HARM, change NATOM,'//CHAR(10)&
                  //'      then turn on CONS HARM again.')
          endif
          call chmrealloc(file_name, routine_name, 'ihset',  natom, intg=ihset)
          call chmrealloc(file_name, routine_name, 'KCNSTR', natom, crl=KCNSTR)
          call chmrealloc(file_name, routine_name, 'REFX  ', natom, crl=REFX  )
          call chmrealloc(file_name, routine_name, 'REFY  ', natom, crl=REFY  )
          call chmrealloc(file_name, routine_name, 'REFZ  ', natom, crl=REFZ  )
          call chmrealloc(file_name, routine_name, 'FBETA ', natom, crl=FBETA )

          !! initialize new elements in the expanded arrays
          !if (old_size .lt. natom) then
          !   new_start = old_size + 1
          !   ihset(new_start : natom) = 0
          !   fbeta(new_start : natom) = zero
          !   kcnstr(new_start : natom) = zero
          !   ihset(new_start : natom) = 0
          !   refx(new_start : natom) = anum
          !   refy(new_start : natom) = anum
          !   refz(new_start : natom) = anum
          !end if
       endif
    else
       call chmalloc(file_name, routine_name, 'ihset',  natom, intg=ihset)
       call chmalloc(file_name, routine_name, 'KCNSTR', natom, crl=KCNSTR)
       call chmalloc(file_name, routine_name, 'REFX  ', natom, crl=REFX  )
       call chmalloc(file_name, routine_name, 'REFY  ', natom, crl=REFY  )
       call chmalloc(file_name, routine_name, 'REFZ  ', natom, crl=REFZ  )
       call chmalloc(file_name, routine_name, 'FBETA ', natom, crl=FBETA )
       fbeta = zero
       kcnstr = zero
       ihset = 0
       refx = anum
       refy = anum
       refz = anum
    end if
    cnst_is_allocated = .true.
  end subroutine allocate_cnst

  subroutine deallocate_cnst(natom)
    use memory
    integer,intent(in) :: natom
    integer :: nsize
    character(len=*),parameter :: routine_name="allocate_cnst"

    if (allocated(ihset))then
       !clb addedd as protection against calling cons clear commands after changing natom
       nsize = size(ihset)
       call chmdealloc(file_name,routine_name,'ihset', nsize,intg=ihset)
       call chmdealloc(file_name,routine_name,'KCNSTR',nsize,crl=KCNSTR)
       call chmdealloc(file_name,routine_name,'REFX  ',nsize,crl=REFX  )
       call chmdealloc(file_name,routine_name,'REFY  ',nsize,crl=REFY  )
       call chmdealloc(file_name,routine_name,'REFZ  ',nsize,crl=REFZ  )
       call chmdealloc(file_name,routine_name,'FBETA ',nsize,crl=FBETA )
    endif
    cnst_is_allocated = .false.
  end subroutine deallocate_cnst
end module cnst_fcm
