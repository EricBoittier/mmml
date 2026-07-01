module qm2_conversions
  use chm_kinds
  use dimens_fcm
  use number
  use consta
#if KEY_SQUANTM==1 || KEY_QTURBO==1 || KEY_G09==1
  use qm2_constants, only : PIi  
#endif

#if KEY_SQUANTM==1 || KEY_QTURBO==1 || KEY_G09==1
  ! 1) contains common conversion factors
  !--------- CONSTANTS AND CONVERSION FACTORS ---------------
  ! Conversion from Bohr to AU
  ! Bohrs * this = angstroms - Same constants as used in dynamo v2.
  !                0.529177249D0 : dynamo v2.
  !                0.52917706D0  : Gaussian 98
  !                0.529177D0    : Mopac6 hcore.f
  real(chm_real8), parameter :: BOHRS_TO_A = 0.529177249D0


  ! 1.0d0 / BOHRS_TO_A
  ! 1.88976D0      : used in Mopac6 gover.f
  real(chm_real8), parameter :: A_TO_BOHRS   = one/BOHRS_TO_A

  ! 3.5711928576D0 : used in Mopac6 gover.f
  real(chm_real8), parameter :: A2_TO_BOHRS2 = A_TO_BOHRS*A_TO_BOHRS
  real(chm_real8), parameter :: A3_TO_BOHRS3 = A2_TO_BOHRS2*A_TO_BOHRS
  real(chm_real8), parameter :: A4_TO_BOHRS4 = A2_TO_BOHRS2*A2_TO_BOHRS2

  ! Conversion from AU to EV
  ! AU to EV
  ! - dynamo v2 and Gaussian 98
  ! - Mopac6: 27.21D0 in calpar.f, delri.f and repp.f
  !           27.2107 in ffhpol.f
  !           27.211  in manual
  ! - more precise would be: 1 a.u. 27.211396 eV

  real(chm_real8), parameter :: HALF_AU_TO_EV   = AU_TO_EV * half
  real(chm_real8), parameter :: FOURTH_AU_TO_EV = AU_TO_EV * fourth
  real(chm_real8), parameter :: EIGHTH_AU_TO_EV = AU_TO_EV * eighth
  real(chm_real8), parameter :: SXNTH_AU_TO_EV  = EIGHTH_AU_TO_EV*half

  ! Converstion from EV to Kcal/mol
  ! Conversion from EV to KCAL/MOL : 23.060362D0
  !                                  23.061d0 : Dynamo
  !                                  23.061   : Mopac6

  real(chm_real8), parameter :: KCAL_TO_EV = one / EV_TO_KCAL

  ! Conversion from AU to Kcal/mol
  real(chm_real8), parameter :: AU_TO_KCAL           =AU_TO_EV*EV_TO_KCAL
  real(chm_real8), parameter :: A2_TO_BOHRS2xAU_TO_EV=A2_TO_BOHRS2*AU_TO_EV

  !
  !***Note***
  ! Added by namkh, since it is used in routine DQLINK4 for GHO correction
  ! CCELEC constants in CHARMM
  ! CCELEC is 1/ (4 pi eps ) in AKMA units, conversion from SI
  ! units: CCELEC = e*e*Na / (4*pi*eps*1Kcal*1A)
  !
  ! for details. refer consta.fcm
  ! this old CHARMM value is kept for compatibility reasons
  ! use compatibility with other electrostatics
  !     332.0522173D0 : AMBER
  !     332.054D0     : DISCOVER
  !     332.0636D0    : old value of dubious origin
  !     331.843D0     : value from 1986-1987 CRC Handbook
  !                     of Chemistry and Physics

  !---mfc--- get CCELEC from const_ltm.src (  use consta)
  !---mfc--- real(chm_real8), parameter :: CCELEC = 332.0716D0


  ! It should be moved some other places..
  ! Exponential decay factor for the MM atom in the core-core interaction
  real(chm_real8), parameter :: ALPH_MM = 5.0d0

  ! Tolerance at which to define a vector is along the axis.
  real(chm_real8), parameter :: AXIS_TOL = 1.0d-8

  ! Distance^2 in bohrs at which to assume Gaussian overlap is zero.
  real(chm_real8), parameter :: OVERLAP_CUTOFF = 100.0d0*A2_TO_BOHRS2

  ! Value of x at which to assume Exp(-x) = zero.
  real(chm_real8), parameter :: EXPONENTIAL_CUTOFF = 30.0d0


  ! 2) Common constants and definitions
  ! ---- Common constants and definitions ----

  ! Degrees in a radian
  real(chm_real8), parameter  :: DEGS_PER_RADIAN = 180.0d0 / PIi

  ! Locks maximum valence orbitals at 4 = S,P
  ! Not yet with D or F orbitals

  INTEGER, PARAMETER :: MAX_VALENCE_ORBITALS =4
  INTEGER, PARAMETER :: MAX_VALENCE_DIMENSION= &
       MAX_VALENCE_ORBITALS*(MAX_VALENCE_ORBITALS+1)/2

  integer, parameter :: PM3 = 1
  integer, parameter :: AM1 = 2
  integer, parameter :: MNDO = 3
  integer, parameter :: PDDGPM3 = 4
  integer, parameter :: PDDGMNDO = 5
  integer, parameter :: PM3CARB1 = 6
  real(chm_real8), parameter :: ONE_SQRT2 = 0.7071067811865475244D0
  !                                                 ! 1.0D0/sqrt(2.0D0)
  !------------------End of Conversion Factors---------------------------!
#endif 
end module qm2_conversions

