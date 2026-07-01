module qm1_constant
   use chm_kinds
   use number
   use consta
#if KEY_MNDO97==1
! constant tiles
      ! Note: in charmm (consta_ltm.src): 
      ! TOKCAL = 627.5095D0, slightly different from EV*EVCAL=627.48981
      ! BOHRR = 0.529177249D0, slightly different from A0=0.529167D0
      real(chm_real), parameter :: A0=BOHRR  ! 0.529167D0  
      real(chm_real), parameter :: EV=27.21D0
      real(chm_real), parameter :: EVCAL=23.061D0 
      ! real(chm_real), parameter :: PI=3.141592653589793D0
      real(chm_real), parameter :: PI2=PI*PI
      real(chm_real), parameter :: INVPI = 1.0d0/PI
      real(chm_real), parameter :: HALFPI= 0.5d0*PI
      real(chm_real), parameter :: FOURPI= 4.0d0*PI
      real(chm_real), parameter :: BIGEXP=50.0D0
      real(chm_real), parameter :: RT3=0.577350269189626D0   ! RT3=1.0D0/SQRT(3.0D0)

      ! real(chm_real), parameter :: hundrd=100.0d0
      real(chm_real), parameter :: PT1=0.1d0
      real(chm_real), parameter :: PT2=0.2d0
      real(chm_real), parameter :: PT5=0.5d0
      real(chm_real), parameter :: PT9=0.9d0
      real(chm_real), parameter :: PT5SQ3=0.8660254037841D0  ! =SQRT(three)/two

      integer,parameter   :: IZERO=0

! may not be used
!      convert rad to deg; 180/Pi
!      real(chm_real), parameter :: AFACT=57.29577951308232D0
!      real(chm_real), parameter :: W1=14.399D0
!      real(chm_real), parameter :: W2=7.1995D0
!      real(chm_real), parameter :: DIPFAC=5.0832D0

! defined in number module
!      real(chm_real), parameter :: zero=0.0d0
!      real(chm_real), parameter :: one=1.0d0
!      real(chm_real), parameter :: two=2.0d0
!      real(chm_real), parameter :: three=3.0d0
!      real(chm_real), parameter :: four=4.0d0
!      real(chm_real), parameter :: five=5.0d0
!      real(chm_real), parameter :: six=6.0d0
!      real(chm_real), parameter :: seven=7.0d0
!      real(chm_real), parameter :: eight=8.0d0
!      real(chm_real), parameter :: nine=9.0d0
!      real(chm_real), parameter :: ten=10.0d0
!      real(chm_real), parameter :: eleven=11.0d0
!      real(chm_real), parameter :: twelve=12.0d0
!      real(chm_real), parameter :: twenty=20.0d0
!      real(chm_real), parameter :: thirty=30.0d0
!      real(chm_real), parameter :: PT01=0.01d0
!      real(chm_real), parameter :: PT25=0.25d0
!      real(chm_real), parameter :: PT75=0.75d0

! end constant files.
#endif
end module qm1_constant
