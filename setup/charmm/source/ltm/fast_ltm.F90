module fast
  use chm_kinds
  use dimens_fcm
  use param_store, only: set_param

  implicit none

  character(len=*),private,parameter :: file_name   ="fast_ltm.src"
!
!     FAST COMMON FILE
!     contains arrays necessary for the vectorization of the nonbond
!     code and fast energy routines
!
!     FASTER  User requested FAST level
!             -1 use only the standard "SLOW" routines
!              0 use FAST(est) routines if possible, no warnings if not
!              1 use generic FAST routines, warning messages if not possible 
!              2 use expanded FAST routines, warning messages if not possible 
!     LFAST   Local FASTER value to control energy routines
!              LFAST is usually set to FASTER, but can be less if fast
!              routines are not available for current command or options.
!
!     NITCC        - Number of different vdw types in use (based on PSF)
!     ICUSED(NATC) - Pointer from atom type (IAC) to compressed vdw type (IACNB).
!     ICCOUNT(NATC)- Count of number of atoms of each type (IAC)
!     LOWTP(NATC)  - lower triangle pointer: n*(n-3)/2
!     IACNB(NATOM) - compressed vdw type for each atoms
!
!     IGCNB(NITCC) - pointer from vdw type to last of RTF MASS list
!
! (Dimension values defined in dimens.fcm)
!     MAXATC  - Maximum number of atom types in the top/par files
!
      INTEGER,allocatable,dimension(:) :: IACNB 
      integer :: NITCC, ICUSED(MAXATC), FASTER, LFAST
      INTEGER ICCOUNT(MAXATC), LOWTP(MAXATC), IGCNB(MAXATC), NITCC2
#if KEY_PERT==1
#if KEY_LRVDW==1
      INTEGER ICCOUNTM(MAXATC),ICCOUNTL(MAXATC) 
#endif 
#endif 

#if KEY_LRVDW==1 /*lrvdw*/
      real(chm_real)  LRCa,LRCb,LRC,LRC2
#if KEY_PERT==1
      real(chm_real)  LRCAP,LRCBP
      real(chm_real) lrvdw_const_ms_l,lrvdw_const_ms_m  ! using Shirts's algorithm
#endif 
      real(chm_real) lrvdw_const_ms  ! using Shirts's algorithm
      real(chm_real) lrvdw_const
      PARAMETER(lrvdw_const=3.141592653589793D0*2.0D0/3.d0)
#endif /*  (lrvdw)*/
!
contains
  subroutine faster_init()
    faster=0
    lfast=0
    call set_param('FASTER',faster)
    return
  end subroutine faster_init

  subroutine allocate_fast_ltm()
    use memory
    character(len=*),parameter :: routine_name="allocate_fast_ltm"
    call chmalloc(file_name,routine_name,'iacnb',maxaim,intg=iacnb)
    return
  end subroutine allocate_fast_ltm


end module fast

