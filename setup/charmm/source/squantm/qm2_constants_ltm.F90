module qm2_constants
  use chm_kinds
  use dimens_fcm
  use number

#if KEY_SQUANTM==1 || KEY_G09==1 || KEY_QTURBO==1 /*sqnt*/
!---------------------- constants.h --------------------
! constants for numerical gradients
      real(chm_real8), parameter :: CHNGE = 1.D-4
      real(chm_real8), parameter :: HALFCHNGE = 5.D-5
      real(chm_real8), parameter :: oneCHNGE = 1.0D4
!       oneCHNGE = one/CHNGE

!-------------------------------------------------------
! Physical Constants

! This is based on old values of the electric constants.
      real(chm_real8),parameter :: AMBER_ELECTROSTATIC=18.2223d0

! 1998 value of the Bohr radius, physics.nist.gov/constants.
! a0 is the standard abbreviation, but perhaps that is too cryptic here.
! The previous value was 0.5291771 in Amber 7.
      real(chm_real8),parameter :: BOHR_RADIUS = 0.5291772083d0


!-------------------------------------------------------
! Numeric Constants

      real(chm_real8),parameter :: PIi = 3.1415926535897932384626433832795d0

      real(chm_real8),parameter :: PI2   = PIi*PIi
      real(chm_real8),parameter :: HALFPI= 0.5d0 * PIi
      real(chm_real8),parameter :: TWO_PI= 2.0d0 * PIi
      real(chm_real8),parameter :: FOURPI= 4.0d0 * PIi
      real(chm_real8),parameter :: INVPI = 1.0d0 / PIi

      real(chm_real8),parameter :: SQRT2=1.4142135623730950488016887242097d0


!-------------------------------------------------------
! Unusual Constants
! first 5 digit palindromic prime
      integer,parameter ::RETIRED_INPUT_OPTION= -10301


!-------------------------------------------------------
! Generic Floating Point Constants

      real(chm_real8),parameter :: TEN_TO_MINUS2 = 1.0d-2
      real(chm_real8),parameter :: TEN_TO_MINUS3 = 1.0d-3
      real(chm_real8),parameter :: TEN_TO_MINUS4 = 1.0d-4
      real(chm_real8),parameter :: TEN_TO_MINUS5 = 1.0d-5
      real(chm_real8),parameter :: TEN_TO_MINUS6 = 1.0d-6
      real(chm_real8),parameter :: TEN_TO_MINUS10= 1.0d-10
      real(chm_real8),parameter :: TEN_TO_PLUS3  = 1.0d+3
      real(chm_real8),parameter :: TENTOPLUS10   = 1.0d+10

!      real(chm_real8),parameter :: zero      =  0.0d0
!      real(chm_real8),parameter :: one       =  1.0d0
!      real(chm_real8),parameter :: two       =  2.0d0
!      real(chm_real8),parameter :: three     =  3.0d0
!      real(chm_real8),parameter :: four      =  4.0d0
!      real(chm_real8),parameter :: five      =  5.0d0
!      real(chm_real8),parameter :: six       =  6.0d0
!      real(chm_real8),parameter :: seven     =  7.0d0
!      real(chm_real8),parameter :: eight     =  8.0d0
!      real(chm_real8),parameter :: nine      =  9.0d0
!      real(chm_real8),parameter :: ten       = 10.0d0
!      real(chm_real8),parameter :: eleven    = 11.0d0
!      real(chm_real8),parameter :: twelve    = 12.0d0
      real(chm_real8),parameter :: sixteen   = 16.0d0
      real(chm_real8),parameter :: thirtytwo = 32.0d0

!      real(chm_real8),parameter :: half         = 0.5D0       ! one/two
!      real(chm_real8),parameter :: third        = one/three
!      real(chm_real8),parameter :: fourth       = 0.25D0      ! one/four
      real(chm_real8),parameter :: fifth        = 0.2D0       ! one/five
!      real(chm_real8),parameter :: sixth        = one/six
      real(chm_real8),parameter :: seventh      = one/seven
!      real(chm_real8),parameter :: eighth       = one/eight
      real(chm_real8),parameter :: ninth        = one/nine
      real(chm_real8),parameter :: tenth        = 0.1D0       ! one/ten
      real(chm_real8),parameter :: eleventh     = one/eleven
      real(chm_real8),parameter :: twelfth      = one/twelve
      real(chm_real8),parameter :: sixteenth    = one/sixteen
      real(chm_real8),parameter :: thirtysecond = one/thirtytwo
!------------------ End constants.h --------------------
#endif /* (sqnt)*/
end module qm2_constants

