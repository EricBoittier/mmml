module number
  use chm_kinds
!
! This file contains floating point numbers.
!
! positive numbers
      real(chm_real),parameter :: &
                 ZERO   =  0.0_CHM_REAL, ONE    =  1.0_CHM_REAL, &
                 TWO    =  2.0_CHM_REAL, THREE  =  3.0_CHM_REAL, &
                 FOUR   =  4.0_CHM_REAL, FIVE   =  5.0_CHM_REAL, &
                 SIX    =  6.0_CHM_REAL, SEVEN  =  7.0_CHM_REAL, &
                 EIGHT  =  8.0_CHM_REAL, NINE   =  9.0_CHM_REAL, &
                 TEN    = 10.0_CHM_REAL, ELEVEN = 11.0_CHM_REAL, &
                 TWELVE = 12.0_CHM_REAL, THIRTN = 13.0_CHM_REAL, &
                 FIFTN  = 15.0_CHM_REAL, NINETN = 19.0_CHM_REAL, &
                 TWENTY = 20.0_CHM_REAL, THIRTY = 30.0_CHM_REAL

      real(chm_real),parameter :: &
                 FIFTY  =   50.0_CHM_REAL, SIXTY  =  60.0_CHM_REAL, &
                 SVNTY2 =   72.0_CHM_REAL, EIGHTY =  80.0_CHM_REAL, &
                 NINETY =   90.0_CHM_REAL, HUNDRD = 100.0_CHM_REAL, &
                 ONE2TY =  120.0_CHM_REAL, ONE8TY = 180.0_CHM_REAL, &
                 THRHUN =  300.0_CHM_REAL, THR6TY = 360.0_CHM_REAL, &
                 NINE99 =  999.0_CHM_REAL, FIFHUN =1500.0_CHM_REAL, &
                 THOSND = 1000.0_CHM_REAL, FTHSND = 5000.0_CHM_REAL, &
                 MEGA      = 1.0E6_CHM_REAL
! negative numbers
      real(chm_real),parameter :: &
           MINONE = -1.0_CHM_REAL,  MINTWO = -2.0_CHM_REAL,  MINSIX = -6.0_CHM_REAL
!
! common fractions
      real(chm_real),parameter :: &
                 TENM20 = 1.0E-20_CHM_REAL,  TENM14 = 1.0E-14_CHM_REAL, &
                 TENM10 = 1.0E-10_CHM_REAL,  TENM8  = 1.0E-8_CHM_REAL,  &
                 TENM6  = 1.0E-6_CHM_REAL,   TENM5  = 1.0E-5_CHM_REAL,  &
                 TENM4  = 1.0E-4_CHM_REAL,   TENM3  = 1.0E-3_CHM_REAL,  &
                 TENM2  = 1.0E-2_CHM_REAL,   TENM1  = 1.0E-1_CHM_REAL,  &
                 PT0001 = 1.0E-4_CHM_REAL,   PT0005 = 5.0E-4_CHM_REAL, &
                 PT001  = 1.0E-3_CHM_REAL,   PT005  = 5.0E-3_CHM_REAL, &
                 PT01   = 0.01_CHM_REAL,     PT02   = 0.02_CHM_REAL,   &
                 PT05   = 0.05_CHM_REAL,     PTONE  = 0.1_CHM_REAL, &
                 PT125  = 0.125_CHM_REAL,    SIXTH  = ONE/SIX, &
                                             FOURTH = 0.25_chm_real, &
                                             EIGHTH = 0.125_chm_real, &
                 PT25   = 0.25_CHM_REAL,     THIRD  = ONE/THREE, &
                 PTFOUR = 0.4_CHM_REAL,      HALF   = 0.5_CHM_REAL, &
                 PTSIX  = 0.6_CHM_REAL,      PT75   = 0.75_CHM_REAL, &
                 PT9999 = 0.9999_CHM_REAL,   ONEPT5 = 1.5_CHM_REAL,  &
                 TWOPT4 = 2.4_CHM_REAL
!
! others
      real(chm_real),parameter :: ANUM=9999.0_CHM_REAL
      real(chm_real),parameter :: FMARK=-999.0_CHM_REAL
#if KEY_SINGLE==1
      real(chm_real),parameter :: RSMALL=1.0E-4_CHM_REAL
#else /**/
      real(chm_real),parameter :: RSMALL=1.0E-10_CHM_REAL
#endif 
      real(chm_real),parameter :: RBIG=1.0E20_CHM_REAL
#if KEY_MSMMPT==1
      real(chm_real),save :: PBCXSIZE=-999.0_CHM_REAL, &
      PBCYSIZE=-999.0_CHM_REAL,PBCZSIZE=-999.0_CHM_REAL
      INTEGER,SAVE :: MMPTNATOMX
#endif
                                                        
!
! Machine constants (these are very machine dependent).
!
! RPRECI should be the smallest number you can add to 1.0 and get a number
! that is different than 1.0.  Actually: the following code must pass for
! for all real numbers A (where no overflow or underflow conditions exist).
! 
!         B = A * RPRECI
!         C = A + B
!         IF(C.EQ.A) STOP 'precision variable is too small'
! 
! The RBIGST value should be the smaller of:
! 
!         - The largest real value               
!         - The reciprocal of the smallest real value 
!
! If there is doubt, be conservative.
!
!     DIAGQ doesn't always terminate with these - mkg
!     real(chm_real), parameter :: RPRECI = epsilon(ONE)
!     real(chm_real), parameter :: RBIGST = ONE / tiny(ONE)
#if KEY_SINGLE==1
      real(chm_real), parameter :: RPRECI = 1.1921E-7
      real(chm_real), parameter :: RBIGST = 8.5070E+37
#else /**/
      real(chm_real), parameter :: RPRECI = 2.22045D-16
      real(chm_real), parameter :: RBIGST = 4.49423D+307
#endif 
!
end module number

