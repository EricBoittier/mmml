module shake
  use chm_kinds
  use dimens_fcm

  implicit none
  character(len=*),private,parameter :: file_name   ="shake_ltm.src"
!
!     SHAKE information
!
!     Purpose: Stores constraint lengths and flags needed by the SHAKE
!     algorithm for constraining bond lengths and angles during a
!     molecular dynamics simulation.
!
!     Notes: The index in the table below gives the domain of an array.
!     An index of "Shake" means a constraint as enforced by SHAKE.
!
!     Variable   Index   Purpose
!
!     MAXSHK             Maximum number of shake constraints
!     NCONST             Number of shake constraints in use
!     NCONST_TOT         Number of shake constraints in use for FSTSHK
!     MXITER             Maximum number of shake iterations before death
!     IDGFEX             Excess number of SHAKE restraints (estimate)
!     SHKTOL             Maximum allowed relative variation in a
!                        constrained length
!     SHKSCA             Convergence scale factor for SHAKEA
!     CONSTR     Shake   The distance between the two atoms involved
!                        in a constraint
!     IDGF2      Atom    The number of degrees of freedom for an atom
!                        times 2
!     SHKAPR     Shake   Holds pairs of atoms involved in a constraint
!
!     QSHAKE             A flag to indicate whether SHAKE is being used.
!     QFSHAKE            A flag to indicate use of FAST version.
!     QHOLO              A flag (set in UPDATE) to indicate that some
!                        holonomic constraints are being used.
!                        These currently include: 
!
!  SHAKE bond distance constraints   (QSHAKE) or (NCONST.GT.0)
#if KEY_TSM==1
!  Internal Coordinate constraints   (IICF.GT.0)   /* (icfix.fcm) */
#endif
#if KEY_NOST2==0
!  ST2 water model 5 center geometry (NST2.GT.0)   /* (psf.fcm) */
#endif
#if KEY_LONEPAIR==1
!  General Lone-pair constraints     (NUMLP.GT.0)  /* (lonepr.fcm) */
#endif
!
!
      INTEGER,allocatable,dimension(:) :: IDGF2
      integer,allocatable,dimension(:,:) :: SHKAPR
      INTEGER NCONST, MXITER,IDGFEX, NCONST_TOT, nconst_pll
!
      LOGICAL QSHAKE,QHOLO,QFSHAKE
!
      real(chm_real),allocatable,dimension(:) :: CONSTR
      real(chm_real) :: SHKTOL, SHKSCA

contains

  subroutine shake_iniall
    qshake=.false.
    qfshake=.false.
    qholo =.false.
    nconst=0
    idgfex=0
    mxiter=500
#if KEY_SINGLE==1
    shktol=1.0e-5
#else /**/
    shktol=1.0e-10
#endif 
    return
  end subroutine shake_iniall

  subroutine allocate_shake(natom)
    use memory
    use stream,only:outu
    integer,intent(in) :: natom
    character(len=*),parameter :: routine_name="allocate_shake"
    if(.not. allocated(constr)) then
       call chmalloc(file_name,routine_name,'constr',maxshk,crl=constr)
       call chmalloc(file_name,routine_name,'idgf2 ',natom,intg=idgf2)
       call chmalloc(file_name,routine_name,'shkapr',2,maxshk,intg=shkapr)
    else
       if( size(idgf2) .lt. natom) then
!          write(outu,*)"ALLOCATE_SHAKE>>> reallocating for new natom: ", &
!               natom,"  was ",size(idgf2)
          call chmdealloc(file_name,routine_name,'idgf2 ',size(idgf2),intg=idgf2)
          call chmalloc(file_name,routine_name,'idgf2 ',natom,intg=idgf2)
       endif
    endif
    return
  end subroutine allocate_shake

  subroutine deallocate_shake(natom)
    use memory
    integer,intent(in) :: natom
    character(len=*),parameter :: routine_name="deallocate_shake"

    if(allocated(constr)) then
       call chmdealloc(file_name,routine_name,'constr',maxshk,crl=constr)
       call chmdealloc(file_name,routine_name,'idgf2 ',natom,intg=idgf2)
       call chmdealloc(file_name,routine_name,'shkapr',2,maxshk,intg=shkapr)
    endif
    return
  end subroutine deallocate_shake

end module shake

