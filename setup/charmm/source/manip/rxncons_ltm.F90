module rxncons
  use chm_kinds
  use dimens_fcm

  implicit none
  character(len=*),private,parameter :: file_name   ="rxncons_ltm.src"
!
#if KEY_RXNCONS==1 /*rxncons_fcm*/
!     The reaction coordinate constraint data
!
!     Purpose:
!
!     To store the lists needed to define and effect the handling of
!     reaction coordinate constraint in CHARMM.
!
!     Variable  Index    Purpose
!
!     LRXCNS             Is reaction coordinate constraint present?
!     IRXCNS             Identity of the constraint type
!                 0      no reaction coord cons
!                 1      d1-d2, difference in the length of two bonds
!                 2      d1, the distance of a bond
!                 3      sapling the middle plane of two reference structures
!                 4      constraint the rms best distances between the replicas
!                        of a path
!     NRXATM   integer   number of atoms defining the rxn coordinate
!     IRXATM   integer   index of the atoms that are involved in defining
!                        the reaction coordinate
!     LCALRX   logical   ????
!     IUNLAG   integer   the unit of text file for printing the lagrange
!                        multipliers
!     NRXNCS   integer   number of constraints
!                        !!! only one is allowed at this point !!!
!     LAGRANM  real      lagrange multipliers
!     LAGRGM   integer   pointer for lagrange multipliers if needed.
!     PLAGRANM logical   print lagrange multipliers?
!     LCLEAN   logical   clean up the data structure
!     LRXPRN   logical   print reaction coordinate stuffs?
!     MAXITR   integer   maximium number of iterations
!     XCORD    real      x coordinate of the atoms that define the reaction
!                        coordinate
!     YCORD    real      y coordinate of the atoms that define the reaction
!                        coordinate
!     ZCORD    real      z coordinate of the atoms that define the reaction
!                        coordinate
!     XREFC    real      reference x coordinate of the atoms that define the
!                        reaction coordinate
!     YREFC    real      reference y coordinate of the atoms that define the
!                        reaction coordinate
!     ZREFC    real      reference z coordinate of the atoms that define the
!                        reaction coordinate
!     VRXNCS   real      value of the constrained degree of freedom
!
!     RCNSDL   real      DL along the constraint
!
!     RCNSDR   real      DR away from the  reference
!
! integers
      integer,allocatable,dimension(:) :: IRXATM
      INTEGER IRXCNS,NRXATM
      integer,allocatable,dimension(:) :: LPRID
      INTEGER IUNLAG,NRXNCS,LAGRGM,MAXITR,NSAVLG
! logicals
      LOGICAL LRXCNS,LCALRX,PLAGRANM,LCLEAN,LRXPRN
! reals
      real(chm_real),allocatable,dimension(:) :: XCORD, YCORD, ZCORD
      real(chm_real),allocatable,dimension(:) :: XREFC, YREFC, ZREFC
      REAL(chm_real) VRXNCS,LAGRANM,FRCONS,DTSQ,ZFAC,GFAC
      REAL(chm_real) RCNSDL,RCNSDS,RCNSDSM,RCNSDC,RCNSDCM
      REAL(chm_real) RCNSDR,RCNSDRC,RCNSAMF
!

contains
  subroutine rxncons_iniall
    IRXCNS=0
    LRXCNS=.FALSE.
    PLAGRANM=.FALSE.
    return
  end subroutine rxncons_iniall

#endif /* (rxncons_fcm)*/

end module rxncons

