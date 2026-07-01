module qm2_parameters
  use chm_kinds
  use dimens_fcm
#if KEY_SQUANTM==1 || KEY_QTURBO==1 || KEY_G09==1
  !***********************************************************
  ! Written by: Ross Walker (TSRI 2005)
  !
  ! File contains the semi-empirical parameters for use in amber qmmm. 
  ! These are the same as the mopac 7 parameters.
  !
  ! This file is not intended to be used directly but instead, for speed, 
  ! is designed to be loaded into a parameter array for each atom.
  !***********************************************************
  ! Total number of elements
  INTEGER, PARAMETER :: nelements = 86

  ! check purpose
  logical, save :: q_parm_loaded=.false.        ! to set .true. after call qm2_initialize_elements

  !Total number of elements
  real(chm_real8),DIMENSION(1:nelements),save :: heat_of_form
  real(chm_real8),DIMENSION(1:nelements),save :: elec_eng_mndo, elec_eng_am1,elec_eng_pm3, &
                                                 elec_eng_pddgpm3,elec_eng_pddgmndo, &
                                                 elec_eng_PM3CARB1
  real(chm_real8),DIMENSION(1:nelements),save :: s_orb_exp_mndo, s_orb_exp_am1,   &
                                                 s_orb_exp_pm3, s_orb_exp_pddgpm3,   &
                                                 s_orb_exp_pddgmndo
  real(chm_real8),DIMENSION(1:nelements),save :: p_orb_exp_mndo, p_orb_exp_am1,   &
                                                 p_orb_exp_pm3, p_orb_exp_pddgpm3,   &
                                                 p_orb_exp_pddgmndo
  real(chm_real8),DIMENSION(1:nelements),save :: betas_mndo, betas_am1, betas_pm3,   &
                                                 betas_pddgpm3, betas_pddgmndo,   &
                                                 betas_pm3carb1
  real(chm_real8),DIMENSION(1:nelements),save :: betap_mndo, betap_am1, betap_pm3,   &
                                                 betap_pddgpm3, betap_pddgmndo,   &
                                                 betap_pm3carb1
  real(chm_real8),DIMENSION(1:4,1:nelements),save :: FN1_am1, FN2_am1, FN3_am1, &
                                                     FN1_pm3, FN2_pm3, FN3_pm3, &
                                                     FN1_pddgpm3, FN2_pddgpm3,FN3_pddgpm3
  real(chm_real8),DIMENSION(1:nelements),save :: GSS_mndo,GSP_mndo,GPP_mndo,GP2_mndo, HSP_mndo
  real(chm_real8),DIMENSION(1:nelements),save :: GSS_am1, GSP_am1, GPP_am1, GP2_am1,  HSP_am1
  real(chm_real8),DIMENSION(1:nelements),save :: GSS_pm3, GSP_pm3, GPP_pm3, GP2_pm3,  HSP_pm3
  real(chm_real8),DIMENSION(1:nelements),save :: GSS_pddgpm3, GSP_pddgpm3, GPP_pddgpm3, GP2_pddgpm3, HSP_pddgpm3
  real(chm_real8),DIMENSION(1:nelements),save :: GSS_pddgmndo, GSP_pddgmndo, GPP_pddgmndo, GP2_pddgmndo, HSP_pddgmndo
  real(chm_real8),DIMENSION(1:nelements),save :: DD_am1, DD_mndo, DD_pm3, DD_pddgpm3, DD_pddgmndo
  real(chm_real8),DIMENSION(1:nelements),save :: QQ_am1, QQ_mndo, QQ_pm3, QQ_pddgpm3, QQ_pddgmndo
  real(chm_real8),DIMENSION(1:nelements),save :: AM_am1, AM_mndo, AM_pm3, AM_pddgpm3, AM_pddgmndo
  real(chm_real8),DIMENSION(1:nelements),save :: AD_am1, AD_mndo, AD_pm3, AD_pddgpm3, AD_pddgmndo
  real(chm_real8),DIMENSION(1:nelements),save :: AQ_am1, AQ_mndo, AQ_pm3, AQ_pddgpm3, AQ_pddgmndo
  real(chm_real8),DIMENSION(1:nelements),save :: alp_mndo, alp_am1, alp_pm3, alp_pddgpm3, alp_pddgmndo, alp_pm3carb1
  real(chm_real8),DIMENSION(1:nelements),save :: USS_mndo, USS_am1, USS_pm3, USS_pddgpm3, USS_pddgmndo, USS_pm3carb1
  real(chm_real8),DIMENSION(1:nelements),save :: UPP_mndo, UPP_am1, UPP_pm3, UPP_pddgpm3, UPP_pddgmndo, UPP_pm3carb1
  real(chm_real8),DIMENSION(1:nelements),save :: PDDGC1_pm3, PDDGC2_pm3, PDDGE1_pm3, PDDGE2_pm3
  real(chm_real8),DIMENSION(1:nelements),save :: PDDGC1_mndo, PDDGC2_mndo, PDDGE1_mndo, PDDGE2_mndo

  integer :: atomic_number

  integer, DIMENSION(1:nelements),save :: core_chg
  integer, DIMENSION(1:nelements),save :: natomic_orbs
  integer, DIMENSION(1:nelements),save :: NUM_FN_am1, NUM_FN_pm3, NUM_FN_pddgpm3    ! Number of FN arrays that are not zer

  logical, dimension(nelements),save :: element_supported_mndo,   &
                                        element_supported_am1,   &
                                        element_supported_pm3,   &
                                        element_supported_pddgpm3,   &
                                        element_supported_pddgmndo

contains

  subroutine qm2_initialize_elements
    ! SECTION 1: Parameters common to all semi-empirical methods
    !-----------------------------------------------------------
    ! core_chg = core charges of elements
    ! heat_of_form = heat of formation

    ! natomic_orbs = number of atomic orbitals
    !***********************************************************************
    !*                      VALENCE SHELLS ARE DEFINED AS                  *
    !*  PQN   VALENCE SHELLS                                               *
    !*                 P-GROUP              F-GROUP    TRANSITION METALS   *
    !*   1       1S                                                        *
    !*   2       2S 2P                                                     *
    !*   3       3S 3P  OR  3S 3P 3D                                       *
    !*   4       4S 4P                                    4S 4P 3D         *
    !*   5       5S 5P                                    5S 5P 4D         *
    !*   6       6S 6P                       6S 4F        6S 6P 5D         *
    !*   7  NOT ASSIGNED YET  ****DO  NOT  USE****                         *
    !***********************************************************************

    !Enthalpies of formation of gaseous atoms are take from 'Annual reports,
    !1974, 71B, P117'
    !NOTE: for natomic_orbs only values of 1 and 4 are currently supported.

    if(.not. q_parm_loaded) then
       q_parm_loaded=.true.

       core_chg(  1)= 1; natomic_orbs(  1)=1; heat_of_form(  1)= 52.102D0 !H
       core_chg(  2)= 0; natomic_orbs(  2)=1; heat_of_form(  2)=  0.000D0 !He

       core_chg(  3)= 1; natomic_orbs(  3)=4; heat_of_form(  3)= 38.410D0 !Li
       core_chg(  4)= 2; natomic_orbs(  4)=4; heat_of_form(  4)= 76.960D0 !Be
       core_chg(  5)= 3; natomic_orbs(  5)=4; heat_of_form(  5)=135.700D0 !B
       core_chg(  6)= 4; natomic_orbs(  6)=4; heat_of_form(  6)=170.890D0 !C
       core_chg(  7)= 5; natomic_orbs(  7)=4; heat_of_form(  7)=113.000D0 !N
       core_chg(  8)= 6; natomic_orbs(  8)=4; heat_of_form(  8)= 59.559D0 !O
       core_chg(  9)= 7; natomic_orbs(  9)=4; heat_of_form(  9)= 18.890D0 !F
       core_chg( 10)= 0; natomic_orbs( 10)=4; heat_of_form( 10)=  0.000D0 !Ne

       core_chg( 11)= 1; natomic_orbs( 11)=0; heat_of_form( 11)= 25.850D0 !Na
       core_chg( 12)= 2; natomic_orbs( 12)=4; heat_of_form( 12)= 35.000D0 !Mg
       core_chg( 13)= 3; natomic_orbs( 13)=4; heat_of_form( 13)= 79.490D0 !Al
       core_chg( 14)= 4; natomic_orbs( 14)=4; heat_of_form( 14)=108.390D0 !Si
       core_chg( 15)= 5; natomic_orbs( 15)=4; heat_of_form( 15)= 75.570D0 !P
       core_chg( 16)= 6; natomic_orbs( 16)=4; heat_of_form( 16)= 66.400D0 !S
       core_chg( 17)= 7; natomic_orbs( 17)=4; heat_of_form( 17)= 28.990D0 !Cl
       core_chg( 18)= 0; natomic_orbs( 18)=4; heat_of_form( 18)=  0.000D0 !Ar

       core_chg( 19)= 1; natomic_orbs( 19)=0; heat_of_form( 19)= 21.420D0 !K
       core_chg( 20)= 2; natomic_orbs( 20)=4; heat_of_form( 20)= 42.600D0 !Ca
       core_chg( 21)= 3; natomic_orbs( 21)=9; heat_of_form( 21)= 90.300D0 !Sc
       core_chg( 22)= 4; natomic_orbs( 22)=9; heat_of_form( 22)=112.300D0 !Ti
       core_chg( 23)= 5; natomic_orbs( 23)=9; heat_of_form( 23)=122.900D0 !V
       core_chg( 24)= 6; natomic_orbs( 24)=9; heat_of_form( 24)= 95.000D0 !Cr
       core_chg( 25)= 7; natomic_orbs( 25)=9; heat_of_form( 25)= 67.700D0 !Mn
       core_chg( 26)= 8; natomic_orbs( 26)=9; heat_of_form( 26)= 99.300D0 !Fe
       core_chg( 27)= 9; natomic_orbs( 27)=9; heat_of_form( 27)=102.400D0 !Co
       core_chg( 28)=10; natomic_orbs( 28)=9; heat_of_form( 28)=102.800D0 !Ni
       core_chg( 29)=11; natomic_orbs( 29)=9; heat_of_form( 29)= 80.700D0 !Cu
       core_chg( 30)= 2; natomic_orbs( 30)=4; heat_of_form( 30)= 31.170D0 !Zn
       core_chg( 31)= 3; natomic_orbs( 31)=4; heat_of_form( 31)= 65.400D0 !Ga
       core_chg( 32)= 4; natomic_orbs( 32)=4; heat_of_form( 32)= 89.500D0 !Ge
       core_chg( 33)= 5; natomic_orbs( 33)=4; heat_of_form( 33)= 72.300D0 !As
       core_chg( 34)= 6; natomic_orbs( 34)=4; heat_of_form( 34)= 54.300D0 !Se
       core_chg( 35)= 7; natomic_orbs( 35)=4; heat_of_form( 35)= 26.740D0 !Br
       core_chg( 36)= 0; natomic_orbs( 36)=4; heat_of_form( 36)=  0.000D0 !Kr

       core_chg( 37)= 1; natomic_orbs( 37)=4; heat_of_form( 37)= 19.600D0 !Rb
       core_chg( 38)= 2; natomic_orbs( 38)=4; heat_of_form( 38)= 39.100D0 !Sr
       core_chg( 39)= 3; natomic_orbs( 39)=9; heat_of_form( 39)=101.500D0 !Y
       core_chg( 40)= 4; natomic_orbs( 40)=9; heat_of_form( 40)=145.500D0 !Zr
       core_chg( 41)= 5; natomic_orbs( 41)=9; heat_of_form( 41)=172.400D0 !Nb
       core_chg( 42)= 6; natomic_orbs( 42)=9; heat_of_form( 42)=157.300D0 !Mo
       core_chg( 43)= 7; natomic_orbs( 43)=9; heat_of_form( 43)=  0.000D0 !Tc
       core_chg( 44)= 8; natomic_orbs( 44)=9; heat_of_form( 44)=155.500D0 !Ru
       core_chg( 45)= 9; natomic_orbs( 45)=9; heat_of_form( 45)=133.000D0 !Rh
       core_chg( 46)=10; natomic_orbs( 46)=9; heat_of_form( 46)= 90.000D0 !Pd
       core_chg( 47)=11; natomic_orbs( 47)=9; heat_of_form( 47)= 68.100D0 !Ag
       core_chg( 48)= 2; natomic_orbs( 48)=4; heat_of_form( 48)= 26.720D0 !Cd
       core_chg( 49)= 3; natomic_orbs( 49)=4; heat_of_form( 49)= 58.000D0 !In
       core_chg( 50)= 4; natomic_orbs( 50)=4; heat_of_form( 50)= 72.200D0 !Sn
       core_chg( 51)= 5; natomic_orbs( 51)=4; heat_of_form( 51)= 63.200D0 !Sb
       core_chg( 52)= 6; natomic_orbs( 52)=4; heat_of_form( 52)= 47.000D0 !Te
       core_chg( 53)= 7; natomic_orbs( 53)=4; heat_of_form( 53)= 25.517D0 !I
       core_chg( 54)= 0; natomic_orbs( 54)=4; heat_of_form( 54)=  0.000D0 !Xe

       core_chg( 55)= 1; natomic_orbs( 55)=0; heat_of_form( 55)= 18.700D0 !Cs
       core_chg( 56)= 2; natomic_orbs( 56)=0; heat_of_form( 56)= 42.500D0 !Ba
       core_chg( 57)= 3; natomic_orbs( 57)=8; heat_of_form( 57)=  0.000D0 !La
       core_chg( 58)= 4; natomic_orbs( 58)=8; heat_of_form( 58)=101.300D0 !Ce
       core_chg( 59)= 5; natomic_orbs( 59)=8; heat_of_form( 59)=  0.000D0 !Pr
       core_chg( 60)= 6; natomic_orbs( 60)=8; heat_of_form( 60)=  0.000D0 !Nd
       core_chg( 61)= 7; natomic_orbs( 61)=8; heat_of_form( 61)=  0.000D0 !Pm
       core_chg( 62)= 8; natomic_orbs( 62)=8; heat_of_form( 62)= 49.400D0 !Sm
       core_chg( 63)= 9; natomic_orbs( 63)=8; heat_of_form( 63)=  0.000D0 !Eu
       core_chg( 64)=10; natomic_orbs( 64)=8; heat_of_form( 64)=  0.000D0 !Gd
       core_chg( 65)=11; natomic_orbs( 65)=8; heat_of_form( 65)=  0.000D0 !Tb
       core_chg( 66)=12; natomic_orbs( 66)=8; heat_of_form( 66)=  0.000D0 !Dy
       core_chg( 67)=13; natomic_orbs( 67)=8; heat_of_form( 67)=  0.000D0 !Ho
       core_chg( 68)=14; natomic_orbs( 68)=8; heat_of_form( 68)= 75.800D0 !Er
       core_chg( 69)=15; natomic_orbs( 69)=8; heat_of_form( 69)=  0.000D0 !Tm
       core_chg( 70)=16; natomic_orbs( 70)=8; heat_of_form( 70)= 36.350D0 !Yb
       core_chg( 71)= 3; natomic_orbs( 71)=9; heat_of_form( 71)=  0.000D0 !Lu
       core_chg( 72)= 4; natomic_orbs( 72)=9; heat_of_form( 72)=148.000D0 !Hf
       core_chg( 73)= 5; natomic_orbs( 73)=9; heat_of_form( 73)=186.900D0 !Ta
       core_chg( 74)= 6; natomic_orbs( 74)=9; heat_of_form( 74)=203.100D0 !W
       core_chg( 75)= 7; natomic_orbs( 75)=9; heat_of_form( 75)=185.000D0 !Re
       core_chg( 76)= 8; natomic_orbs( 76)=9; heat_of_form( 76)=188.000D0 !Os
       core_chg( 77)= 9; natomic_orbs( 77)=9; heat_of_form( 77)=160.000D0 !Ir
       core_chg( 78)=10; natomic_orbs( 78)=9; heat_of_form( 78)=135.200D0 !Pt
       core_chg( 79)=11; natomic_orbs( 79)=9; heat_of_form( 79)= 88.000D0 !Au
       core_chg( 80)= 2; natomic_orbs( 80)=4; heat_of_form( 80)= 14.690D0 !Hg
       core_chg( 81)= 3; natomic_orbs( 81)=4; heat_of_form( 81)= 43.550D0 !Tl
       core_chg( 82)= 4; natomic_orbs( 82)=4; heat_of_form( 82)= 46.620D0 !Pb
       core_chg( 83)= 5; natomic_orbs( 83)=4; heat_of_form( 83)= 50.100D0 !Bi
       core_chg( 84)= 6; natomic_orbs( 84)=4; heat_of_form( 84)=  0.000D0 !Po

!!!***Note***
!!! GHO uses atom number 85 (GHO-Carbon atom)
       core_chg( 85)= 4; natomic_orbs( 85)=4; heat_of_form( 85)=170.890D0 
!!!     core_chg( 85)= 7; natomic_orbs( 85)= 4; heat_of_form( 85)=  0.000D0 !At
!!!***

       core_chg( 86)= 0; natomic_orbs( 86)=4; heat_of_form( 86)=  0.000D0 !Rn

    !-----------------------------------------------------------
    !END OF SECTION 1 - COMMON PARAMETERS
    !-----------------------------------------------------------


    !-----------------------------------------------------------
    !SECTION 2 - MNDO, AM1, PM3, PDDG/PM3 and PDDG/MNDO PARAMS BY ELEMENT
    !-----------------------------------------------------------
    !This section contains the MNDO, AM1, PM3 and PDDG parameter sets
    !for each element that is supported.
    !Set element supported array to false and then set each entry true
    !as and when we add the parameters.

       element_supported_mndo(1:nelements) = .false.
       element_supported_am1(1:nelements) = .false.
       element_supported_pm3(1:nelements) = .false.
       element_supported_pddgpm3(1:nelements) = .false.
       element_supported_pddgmndo(1:nelements) = .false.

    !Parameter meanings
    !   elec_eng - Calculated electronic energy for each element
    !   s_orb_exp, p_orb_exp - The Slater exponents of the basis functions
    !   betas, betap - two centre, one electron core integral parameters.
    !   FN1,FN2,FN3 - PM3 / AM1 specific parameters for the core-core interactions
    !    GSS ::= (SS,SS)                                                           
    !    GPP ::= (PP,PP)                                                           
    !    GSP ::= (SS,PP)                                                           
    !    GP2 ::= (PP,P*P*)                                                         
    !    HSP ::= (SP,SP) 
    !   GSS, GSP, GPP, GP2, HSP - Coulomb and exchange one centre-two electron integral params.
    !   DD, QQ, AD, AM, AQ - parameters for multipole expansion of the two centre, 
    !                        two electron integrals.
    !   ALP - Exponents for the core-core repulsion terms
    !   USS, UPP - electron kinetic energy integral parameters.

    ! Initialise the arrays
       elec_eng_mndo(1:nelements) = 0.0d0
       elec_eng_am1(1:nelements) = 0.0d0
       elec_eng_pm3(1:nelements) = 0.0d0
       elec_eng_pddgmndo(1:nelements) = 0.0d0
       elec_eng_pddgpm3(1:nelements) = 0.0d0
       elec_eng_PM3CARB1(1:nelements) = 0.0d0
       s_orb_exp_mndo(1:nelements) = 0.0d0
       s_orb_exp_am1(1:nelements) = 0.0d0
       s_orb_exp_pm3(1:nelements) = 0.0d0
       s_orb_exp_pddgmndo(1:nelements) = 0.0d0
       s_orb_exp_pddgpm3(1:nelements) = 0.0d0
       p_orb_exp_mndo(1:nelements) = 0.0d0
       p_orb_exp_am1(1:nelements) = 0.0d0
       p_orb_exp_pm3(1:nelements) = 0.0d0
       p_orb_exp_pddgmndo(1:nelements) = 0.0d0
       p_orb_exp_pddgpm3(1:nelements) = 0.0d0
       betas_mndo(1:nelements) = 0.0d0
       betas_am1(1:nelements) = 0.0d0
       betas_pm3(1:nelements) = 0.0d0
       betas_pddgmndo(1:nelements) = 0.0d0
       betas_pddgpm3(1:nelements) = 0.0d0
       betas_pm3carb1(1:nelements) = 0.0d0
       betap_mndo(1:nelements) = 0.0d0
       betap_am1(1:nelements) = 0.0d0
       betap_pm3(1:nelements) = 0.0d0
       betap_pddgmndo(1:nelements) = 0.0d0
       betap_pddgmndo(1:nelements) = 0.0d0
       betap_pm3carb1(1:nelements) = 0.0d0
       FN1_am1 = 0.0d0
       FN2_am1 = 0.0d0
       FN3_am1 = 0.0d0
       NUM_FN_am1 = 0
       FN1_pm3 = 0.0d0
       FN2_pm3 = 0.0d0
       FN3_pm3 = 0.0d0
       NUM_FN_pm3 = 0
       FN1_pddgpm3 = 0.0d0
       FN2_pddgpm3 = 0.0d0
       FN3_pddgpm3 = 0.0d0
       NUM_FN_pddgpm3 = 0
       GSS_mndo = 0.0d0; GSP_mndo = 0.0d0; GPP_mndo = 0.0d0 
       GP2_mndo = 0.0d0; HSP_mndo = 0.0d0

       GSS_am1 = 0.0d0; GSP_am1 = 0.0d0; GPP_am1 = 0.0d0 
       GP2_am1 = 0.0d0; HSP_am1 = 0.0d0

       GSS_pm3 = 0.0d0; GSP_pm3 = 0.0d0; GPP_pm3 = 0.0d0
       GP2_pm3 = 0.0d0; HSP_pm3 = 0.0d0

       GSS_pddgmndo = 0.0d0; GSP_pddgmndo = 0.0d0; GPP_pddgmndo = 0.0d0
       GP2_pddgmndo = 0.0d0; HSP_pddgmndo = 0.0d0

       GSS_pddgpm3 = 0.0d0; GSP_pddgpm3 = 0.0d0; GPP_pddgpm3 = 0.0d0
       GP2_pddgpm3 = 0.0d0; HSP_pddgpm3 = 0.0d0

       DD_mndo = 0.0d0; QQ_mndo = 0.0d0; AD_mndo = 0.0d0
       AM_mndo = 0.0d0; AQ_mndo = 0.0d0

       DD_am1 = 0.0d0; QQ_am1 = 0.0d0; AD_am1 = 0.0d0
       AM_am1 = 0.0d0; AQ_am1 = 0.0d0

       DD_pm3 = 0.0d0; QQ_pm3 = 0.0d0; AD_pm3 = 0.0d0
       AM_pm3 = 0.0d0; AQ_pm3 = 0.0d0

       DD_pddgmndo = 0.0d0; QQ_pddgmndo = 0.0d0; AD_pddgmndo = 0.0d0
       AM_pddgmndo = 0.0d0; AQ_pddgmndo = 0.0d0

       DD_pddgpm3 = 0.0d0; QQ_pddgpm3 = 0.0d0; AD_pddgpm3 = 0.0d0
       AM_pddgpm3 = 0.0d0; AQ_pddgpm3 = 0.0d0

       alp_mndo = 0.0d0; alp_am1=0.0d0; alp_pm3 = 0.0d0
       alp_pddgmndo = 0.0d0; alp_pddgpm3 = 0.0d0; alp_pm3carb1 = 0.0d0

       uss_mndo = 0.0d0; uss_am1=0.0d0; uss_pm3=0.0d0
       uss_pddgmndo = 0.0d0; uss_pddgpm3 = 0.0d0; uss_pm3carb1 = 0.0d0

       upp_mndo = 0.0d0; upp_am1=0.0d0; upp_pm3=0.0d0
       upp_pddgmndo = 0.0d0; upp_pddgpm3 = 0.0d0; upp_pm3carb1 = 0.0d0

       PDDGC1_pm3 = 0.0d0; PDDGC2_pm3 = 0.0d0; PDDGE1_pm3 = 0.0d0
       PDDGE2_pm3 = 0.0d0

       PDDGC1_mndo = 0.0d0; PDDGC2_mndo = 0.0d0; PDDGE1_mndo = 0.0d0
       PDDGE2_mndo = 0.0d0

    !-------------------
    !HYDROGEN
    !-------------------
    atomic_number = 1
    !MNDO
    ! Reference: M.J.S. DEWAR, W. THIEL, J. AM. CHEM. SOC., 
    !            99, 4899, (1977)
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -11.9062760D0
    s_orb_exp_mndo(atomic_number) = 1.3319670D0
    p_orb_exp_mndo(atomic_number) = 0.0d0
    betas_mndo(atomic_number) = -6.9890640D0
    betap_mndo(atomic_number) = 0.0d0
    GSS_mndo(atomic_number) = 12.848D00
    DD_mndo(atomic_number) = 0.0d0
    QQ_mndo(atomic_number) = 0.0d0
    AM_mndo(atomic_number) = 0.4721793D0
    AD_mndo(atomic_number) = 0.4721793D0
    AQ_mndo(atomic_number) = 0.4721793D0
    alp_mndo(atomic_number) = 2.5441341D0
    USS_mndo(atomic_number) = -11.9062760D0
    UPP_mndo(atomic_number) = 0.0d0

    !AM1
    ! Reference: M.J.S. DEWAR ET AL, J. AM. CHEM. SOC. 
    !            107 3902-3909 (1985)
    element_supported_am1(atomic_number) = .true.
    elec_eng_am1(atomic_number) = -11.3964270D0
    s_orb_exp_am1(atomic_number) = 1.1880780D0
    p_orb_exp_am1(atomic_number) = 0.0d0
    betas_am1(atomic_number) = -6.1737870D0
    betap_am1(atomic_number) = 0.0d0
    FN1_am1(1,atomic_number) = 0.1227960D0
    FN2_am1(1,atomic_number) = 5.0000000D0
    FN3_am1(1,atomic_number) = 1.2000000D0
    FN1_am1(2,atomic_number) = 0.0050900D0
    FN2_am1(2,atomic_number) = 5.0000000D0
    FN3_am1(2,atomic_number) = 1.8000000D0
    FN1_am1(3,atomic_number) =-0.0183360D0
    FN2_am1(3,atomic_number) = 2.0000000D0
    FN3_am1(3,atomic_number) = 2.1000000D0
    FN1_am1(4,atomic_number) = 0.0d0 
    FN2_am1(4,atomic_number) = 0.0d0
    FN3_am1(4,atomic_number) = 0.0d0
    NUM_FN_am1(atomic_number) = 3
    GSS_am1(atomic_number) = 12.8480000D0
    DD_am1(atomic_number) = 0.0d0
    QQ_am1(atomic_number) = 0.0d0
    AM_am1(atomic_number) = 0.4721793D0
    AD_am1(atomic_number) = 0.4721793D0
    AQ_am1(atomic_number) = 0.4721793D0
    alp_am1(atomic_number) = 2.8823240D0
    USS_am1(atomic_number) = -11.3964270D0
    UPP_am1(atomic_number) = 0.0d0

    !PM3    
    ! Reference J. J. P. STEWART, J. COMP. CHEM. 10, 209 (1989).
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -13.0733210D0
    s_orb_exp_pm3(atomic_number) = 0.9678070D0
    p_orb_exp_pm3(atomic_number) = 0.0d0
    betas_pm3(atomic_number) = -5.6265120D0
    betap_pm3(atomic_number) = 0.0d0
    FN1_pm3(1,atomic_number) = 1.1287500D0
    FN2_pm3(1,atomic_number) = 5.0962820D0
    FN3_pm3(1,atomic_number) = 1.5374650D0 
    FN1_pm3(2,atomic_number) =-1.0603290D0
    FN2_pm3(2,atomic_number) = 6.0037880D0 
    FN3_pm3(2,atomic_number) = 1.5701890D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 14.7942080D0
    DD_pm3(atomic_number) = 0.0d0
    QQ_pm3(atomic_number) = 0.0d0
    AM_pm3(atomic_number) = 0.5437048D0
    AD_pm3(atomic_number) = 0.5437048D0
    AQ_pm3(atomic_number) = 0.5437048D0
    alp_pm3(atomic_number) = 3.3563860D0
    USS_pm3(atomic_number) = -13.0733210D0
    UPP_pm3(atomic_number) = 0.0d0

    !PDDG/PM3
    ! Reference Repasky, Chandrasekhar and Jorgensen, 2002, J Comp Chem, 
    !           23: 1601-1622
    element_supported_pddgpm3(atomic_number) = .true.
    elec_eng_pddgpm3(atomic_number) = -13.120566198192D0
    s_orb_exp_pddgpm3(atomic_number) = 0.97278550084430D0
    p_orb_exp_pddgpm3(atomic_number) = 0.0d0
    betas_pddgpm3(atomic_number) = -6.1526542062173D0
    betap_pddgpm3(atomic_number) = 0.0d0
    FN1_pddgpm3(1,atomic_number) = 1.12224395962630D0
    FN2_pddgpm3(1,atomic_number) = 4.70779030777590D0
    FN3_pddgpm3(1,atomic_number) = 1.54709920873910D0
    FN1_pddgpm3(2,atomic_number) =-1.0697373657305D0
    FN2_pddgpm3(2,atomic_number) = 5.85799464741120D0
    FN3_pddgpm3(2,atomic_number) = 1.56789274832050D0
    FN1_pddgpm3(3,atomic_number) = 0.0d0
    FN2_pddgpm3(3,atomic_number) = 0.0d0
    FN3_pddgpm3(3,atomic_number) = 0.0d0
    FN1_pddgpm3(4,atomic_number) = 0.0d0
    FN2_pddgpm3(4,atomic_number) = 0.0d0
    FN3_pddgpm3(4,atomic_number) = 0.0d0
    NUM_FN_pddgpm3(atomic_number) = 2
    GSS_pddgpm3(atomic_number) = 14.7942080D0
    DD_pddgpm3(atomic_number) = 0.0d0
    QQ_pddgpm3(atomic_number) = 0.0d0
    AM_pddgpm3(atomic_number) = 0.54370481704970D0
    AD_pddgpm3(atomic_number) = 0.54370481704970D0
    AQ_pddgpm3(atomic_number) = 0.54370481704970D0
    alp_pddgpm3(atomic_number) = 3.38168610300700D0
    USS_pddgpm3(atomic_number) = -12.893272003385D0
    UPP_pddgpm3(atomic_number) = 0.0d0
    PDDGC1_pm3(atomic_number) = 0.05719290135800D0
    PDDGC2_pm3(atomic_number) = -0.0348228612590D0
    PDDGE1_pm3(atomic_number) = 0.66339504047230D0
    PDDGE2_pm3(atomic_number) = 1.08190071942210D0

    !PDDG/MNDO
    ! Reference: Repasky, Chandrasekhar and Jorgensen, 2002, J Comp Chem,
    !            23: 1601-1622
    element_supported_pddgmndo(atomic_number) = .true.
    elec_eng_pddgmndo(atomic_number) = -12.015955786557D0
    s_orb_exp_pddgmndo(atomic_number) = 1.32243115467370D0
    p_orb_exp_pddgmndo(atomic_number) = 0.0d0
    betas_pddgmndo(atomic_number) = -7.4935039195719D0
    betap_pddgmndo(atomic_number) = 0.0d0
    GSS_pddgmndo(atomic_number) = 12.848D00
    DD_pddgmndo(atomic_number) = 0.0d0
    QQ_pddgmndo(atomic_number) = 0.0d0
    AM_pddgmndo(atomic_number) = 0.47217934812430D0
    AD_pddgmndo(atomic_number) = 0.47217934812430D0
    AQ_pddgmndo(atomic_number) = 0.47217934812430D0
    alp_pddgmndo(atomic_number) = 2.49181323064320D0
    USS_pddgmndo(atomic_number) = -11.724114276410D0
    UPP_pddgmndo(atomic_number) = 0.0d0
    PDDGC1_mndo(atomic_number) = -0.1088607444359D0
    PDDGC2_mndo(atomic_number) = -0.0247060666203D0
    PDDGE1_mndo(atomic_number) = 0.46072116172000D0
    PDDGE2_mndo(atomic_number) = 1.29873123436820D0

    !PM3CARB-1
    !Reference: McNamara, J.P., Muslim, A.M., Abdel-Aal, H., Wang, H., 
    !           Mohr, M., Hillier, I.H., Bryce, R.A.,
    !           2004, Chem Phys Lett, 394, 429-436
    !Note: PM3CARB-1 is not a complete parameter set, it is simply PM3 
    !      with a few parameters changed for Oxygen and Hydrogen.
    !      So when loading these parameters into the correct structure 
    !      the code needs to load pm3 parameters except for those
    !      that are different for PM3CARB-1
    elec_eng_PM3CARB1(atomic_number) = -13.514849D0
    USS_PM3CARB1(atomic_number) = -13.514849D0
    UPP_PM3CARB1(atomic_number) = 0.0d0
    betas_PM3CARB1(atomic_number) = -4.011786D0
    betap_PM3CARB1(atomic_number) = 0.0d0
    alp_PM3CARB1(atomic_number) = 2.753199D0

    !-------------------
    !END HYDROGEN
    !-------------------

    !-------------------
    !LITHIUM
    !-------------------
    atomic_number = 3
    !MNDO
    ! Reference:   TAKEN FROM MNDOC BY W.THIEL,QCPE NO.438, V. 2, P.63,
    !              (1982).
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -5.1280000D0
    s_orb_exp_mndo(atomic_number) = 0.7023800D0
    p_orb_exp_mndo(atomic_number) = 0.7023800D0
    betas_mndo(atomic_number) = -1.3500400D0
    betap_mndo(atomic_number) = -1.3500400D0
    GSS_mndo(atomic_number) = 7.3000000D0
    GSP_mndo(atomic_number) = 5.4200000D0
    GPP_mndo(atomic_number) = 5.0000000D0
    GP2_mndo(atomic_number) = 4.5200000D0
    HSP_mndo(atomic_number) = 0.8300000D0
    DD_mndo(atomic_number) = 2.0549783D0
    QQ_mndo(atomic_number) = 1.7437069D0
    AM_mndo(atomic_number) = 0.2682837D0
    AD_mndo(atomic_number) = 0.2269793D0
    AQ_mndo(atomic_number) = 0.2614581D0
    alp_mndo(atomic_number) = 1.2501400D0
    USS_mndo(atomic_number) = -5.1280000D0
    UPP_mndo(atomic_number) = -2.7212000D0

    !AM1
    ! Reference: As MNDO...
    element_supported_am1(atomic_number) = .true.
    elec_eng_am1(atomic_number) = -5.1280000D0
    s_orb_exp_mndo(atomic_number) = 0.7023800D0
    p_orb_exp_mndo(atomic_number) = 0.7023800D0
    betas_am1(atomic_number) = -1.3500400D0
    betap_am1(atomic_number) = -1.3500400D0
    FN1_am1(1,atomic_number) = 0.0d0
    FN2_am1(1,atomic_number) = 0.0d0
    FN3_am1(1,atomic_number) = 0.0d0
    FN1_am1(2,atomic_number) = 0.0d0
    FN2_am1(2,atomic_number) = 0.0d0
    FN3_am1(2,atomic_number) = 0.0d0
    FN1_am1(3,atomic_number) = 0.0d0
    FN2_am1(3,atomic_number) = 0.0d0
    FN3_am1(3,atomic_number) = 0.0d0
    FN1_am1(4,atomic_number) = 0.0d0
    FN2_am1(4,atomic_number) = 0.0d0
    FN3_am1(4,atomic_number) = 0.0d0
    NUM_FN_am1(atomic_number) = 0
    GSS_am1(atomic_number) = 7.3000000D0
    GSP_am1(atomic_number) = 5.4200000D0
    GPP_am1(atomic_number) = 5.0000000D0
    GP2_am1(atomic_number) = 4.5200000D0
    HSP_am1(atomic_number) = 0.8300000D0
    DD_am1(atomic_number) = 2.0549783D0
    QQ_am1(atomic_number) = 1.7437069D0
    AM_am1(atomic_number) = 0.2682837D0
    AD_am1(atomic_number) = 0.2269793D0
    AQ_am1(atomic_number) = 0.2614581D0
    alp_am1(atomic_number) = 1.2501400D0
    USS_am1(atomic_number) = -5.1280000D0
    UPP_am1(atomic_number) = -2.7212000D0

    !PM3
    ! Reference NONE
    element_supported_pm3(atomic_number) = .false.

    !PDDG/PM3
    ! Reference NONE
    element_supported_pddgpm3(atomic_number) = .false.

    !-------------------
    !END LITHIUM
    !-------------------

    !-------------------
    !BERYLLIUM
    !-------------------
    atomic_number = 4
    !MNDO
    ! Reference: M.J.S. DEWAR, H.S. RZEPA, J. AM. CHEM. SOC., 100, 777, 
    !            (1978)
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -24.2047560D0
    s_orb_exp_mndo(atomic_number) = 1.0042100D0
    p_orb_exp_mndo(atomic_number) = 1.0042100D0
    betas_mndo(atomic_number) = -4.0170960D0
    betap_mndo(atomic_number) = -4.0170960D0
    GSS_mndo(atomic_number) = 9.00D00
    GSP_mndo(atomic_number) = 7.43D00
    GPP_mndo(atomic_number) = 6.97D00
    GP2_mndo(atomic_number) = 6.22D00
    HSP_mndo(atomic_number) = 1.28D00
    DD_mndo(atomic_number) = 1.4373245D0
    QQ_mndo(atomic_number) = 1.2196103D0
    AM_mndo(atomic_number) = 0.3307607D0
    AD_mndo(atomic_number) = 0.3356142D0
    AQ_mndo(atomic_number) = 0.3846373D0
    alp_mndo(atomic_number) = 1.6694340D0
    USS_mndo(atomic_number) = -16.6023780D0
    UPP_mndo(atomic_number) = -10.7037710D0

    !AM1
    ! Reference: As MNDO...
    element_supported_am1(atomic_number) = .true.
    elec_eng_am1(atomic_number) = -24.2047560D0
    s_orb_exp_am1(atomic_number) = 1.0042100D0
    p_orb_exp_am1(atomic_number) = 1.0042100D0
    betas_am1(atomic_number) = -4.0170960D0
    betap_am1(atomic_number) = -4.0170960D0
    FN1_am1(1,atomic_number) = 0.0d0
    FN2_am1(1,atomic_number) = 0.0d0
    FN3_am1(1,atomic_number) = 0.0d0
    FN1_am1(2,atomic_number) = 0.0d0
    FN2_am1(2,atomic_number) = 0.0d0
    FN3_am1(2,atomic_number) = 0.0d0
    FN1_am1(3,atomic_number) = 0.0d0
    FN2_am1(3,atomic_number) = 0.0d0
    FN3_am1(3,atomic_number) = 0.0d0
    FN1_am1(4,atomic_number) = 0.0d0
    FN2_am1(4,atomic_number) = 0.0d0
    FN3_am1(4,atomic_number) = 0.0d0
    NUM_FN_am1(atomic_number) = 0
    GSS_am1(atomic_number) = 9.0000000D0
    GSP_am1(atomic_number) = 7.4300000D0
    GPP_am1(atomic_number) = 6.9700000D0
    GP2_am1(atomic_number) = 6.2200000D0
    HSP_am1(atomic_number) = 1.2800000D0
    DD_am1(atomic_number) = 1.4373245D0
    QQ_am1(atomic_number) = 1.2196103D0
    AM_am1(atomic_number) = 0.3307607D0
    AD_am1(atomic_number) = 0.3356142D0
    AQ_am1(atomic_number) = 0.3846373D0
    alp_am1(atomic_number) = 1.6694340D0
    USS_am1(atomic_number) = -16.6023780D0
    UPP_am1(atomic_number) = -10.7037710D0

    !PM3
    ! Reference:  J. J. P. STEWART, J. COMP. CHEM.(ACCEPTED)
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -25.5166530D0
    s_orb_exp_pm3(atomic_number) = 0.8774390D0
    p_orb_exp_pm3(atomic_number) = 1.5087550D0
    betas_pm3(atomic_number) = -3.9620530D0
    betap_pm3(atomic_number) = -2.7806840D0
    FN1_pm3(1,atomic_number) = 1.6315720D0 
    FN2_pm3(1,atomic_number) = 2.6729620D0
    FN3_pm3(1,atomic_number) = 1.7916860D0
    FN1_pm3(2,atomic_number) =-2.1109590D0
    FN2_pm3(2,atomic_number) = 1.9685940D0
    FN3_pm3(2,atomic_number) = 1.7558710D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 9.0128510D0
    GSP_pm3(atomic_number) = 6.5761990D0
    GPP_pm3(atomic_number) = 6.0571820D0
    GP2_pm3(atomic_number) = 9.0052190D0
    HSP_pm3(atomic_number) = 0.5446790D0
    DD_pm3(atomic_number) = 1.0090531D0
    QQ_pm3(atomic_number) = 0.8117586D0
    AM_pm3(atomic_number) = 0.3312330D0
    AD_pm3(atomic_number) = 0.2908996D0
    AQ_pm3(atomic_number) = 0.3530008D0
    alp_pm3(atomic_number) = 1.5935360D0
    USS_pm3(atomic_number) = -17.2647520D0
    UPP_pm3(atomic_number) = -11.3042430D0

    !PDDG/PM3
    ! Reference NONE
    element_supported_pddgpm3(atomic_number) = .false.

    !-------------------
    !END BERYLLIUM
    !-------------------

    !-------------------
    !BORON
    !-------------------
    atomic_number = 5
    !MNDO
    ! Reference: M.J.S. DEWAR, M.L. MCKEE, J. AM. CHEM. SOC., 99, 5231, 
    !            (1977)
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -64.3159500D0
    s_orb_exp_mndo(atomic_number) = 1.5068010D0
    p_orb_exp_mndo(atomic_number) = 1.5068010D0
    betas_mndo(atomic_number) = -8.2520540D0
    betap_mndo(atomic_number) = -8.2520540D0
    GSS_mndo(atomic_number) = 10.59D00
    GSP_mndo(atomic_number) = 9.56D00
    GPP_mndo(atomic_number) = 8.86D00
    GP2_mndo(atomic_number) = 7.86D00
    HSP_mndo(atomic_number) = 1.81D00
    DD_mndo(atomic_number) = 0.9579073D0
    QQ_mndo(atomic_number) = 0.8128113D0
    AM_mndo(atomic_number) = 0.3891951D0
    AD_mndo(atomic_number) = 0.4904730D0
    AQ_mndo(atomic_number) = 0.5556979D0
    alp_mndo(atomic_number) = 2.1349930D0
    USS_mndo(atomic_number) = -34.5471300D0
    UPP_mndo(atomic_number) = -23.1216900D0

    !AM1
    ! Reference: None
    element_supported_am1(atomic_number) = .false.

    !PM3
    ! Reference: NONE
    element_supported_pm3(atomic_number) = .false.

    !PDDG/PM3
    ! Reference NONE
    element_supported_pddgpm3(atomic_number) = .false.

    !-------------------
    !END BORON
    !-------------------

    !-------------------
    !CARBON
    !-------------------
    atomic_number = 6
    !MNDO
    ! Reference: M.J.S. DEWAR, W. THIEL, J. AM. CHEM. SOC., 99, 4899, 
    !            (1977)
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -120.5006060D0
    s_orb_exp_mndo(atomic_number) = 1.7875370D0
    p_orb_exp_mndo(atomic_number) = 1.7875370D0
    betas_mndo(atomic_number) = -18.9850440D0
    betap_mndo(atomic_number) = -7.9341220D0
    GSS_mndo(atomic_number) = 12.23D00
    GSP_mndo(atomic_number) = 11.47D00
    GPP_mndo(atomic_number) = 11.08D00
    GP2_mndo(atomic_number) = 9.84D00
    HSP_mndo(atomic_number) = 2.43D00
    DD_mndo(atomic_number) = 0.8074662D0
    QQ_mndo(atomic_number) = 0.6851578D0
    AM_mndo(atomic_number) = 0.4494671D0
    AD_mndo(atomic_number) = 0.6149474D0
    AQ_mndo(atomic_number) = 0.6685897D0
    alp_mndo(atomic_number) = 2.5463800D0
    USS_mndo(atomic_number) = -52.2797450D0
    UPP_mndo(atomic_number) = -39.2055580D0

    !AM1
    ! Reference:  M.J.S. DEWAR ET AL, J. AM. CHEM. SOC. 107 3902-3909 
    !             (1985)
    element_supported_am1(atomic_number) = .true.
    elec_eng_am1(atomic_number) = -120.8157940D0
    s_orb_exp_am1(atomic_number) = 1.8086650D0
    p_orb_exp_am1(atomic_number) = 1.6851160D0
    betas_am1(atomic_number) = -15.7157830D0
    betap_am1(atomic_number) = -7.7192830D0
    FN1_am1(1,atomic_number) = 0.0113550D0
    FN2_am1(1,atomic_number) = 5.0000000D0
    FN3_am1(1,atomic_number) = 1.6000000D0
    FN1_am1(2,atomic_number) = 0.0459240D0
    FN2_am1(2,atomic_number) = 5.0000000D0
    FN3_am1(2,atomic_number) = 1.8500000D0
    FN1_am1(3,atomic_number) =-0.0200610D0
    FN2_am1(3,atomic_number) = 5.0000000D0
    FN3_am1(3,atomic_number) = 2.0500000D0
    FN1_am1(4,atomic_number) =-0.0012600D0
    FN2_am1(4,atomic_number) = 5.0000000D0
    FN3_am1(4,atomic_number) = 2.6500000D0
    NUM_FN_am1(atomic_number) = 4
    GSS_am1(atomic_number) = 12.2300000D0
    GSP_am1(atomic_number) = 11.4700000D0
    GPP_am1(atomic_number) = 11.0800000D0
    GP2_am1(atomic_number) = 9.8400000D0
    HSP_am1(atomic_number) = 2.4300000D0
    DD_am1(atomic_number) = 0.8236736D0
    QQ_am1(atomic_number) = 0.7268015D0
    AM_am1(atomic_number) = 0.4494671D0
    AD_am1(atomic_number) = 0.6082946D0
    AQ_am1(atomic_number) = 0.6423492D0
    alp_am1(atomic_number) = 2.6482740D0
    USS_am1(atomic_number) = -52.0286580D0
    UPP_am1(atomic_number) = -39.6142390D0

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM.10, 209 (1989).
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -111.2299170D0 
    s_orb_exp_pm3(atomic_number) = 1.5650850D0
    p_orb_exp_pm3(atomic_number) = 1.8423450D0
    betas_pm3(atomic_number) = -11.9100150D0
    betap_pm3(atomic_number) = -9.8027550D0
    FN1_pm3(1,atomic_number) = 0.0501070D0
    FN2_pm3(1,atomic_number) = 6.0031650D0
    FN3_pm3(1,atomic_number) = 1.6422140D0
    FN1_pm3(2,atomic_number) = 0.0507330D0
    FN2_pm3(2,atomic_number) = 6.0029790D0
    FN3_pm3(2,atomic_number) = 0.8924880D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 11.2007080D0
    GSP_pm3(atomic_number) = 10.2650270D0
    GPP_pm3(atomic_number) = 10.7962920D0
    GP2_pm3(atomic_number) = 9.0425660D0
    HSP_pm3(atomic_number) = 2.2909800D0
    DD_pm3(atomic_number) = 0.8332396D0
    QQ_pm3(atomic_number) = 0.6647750D0
    AM_pm3(atomic_number) = 0.4116394D0
    AD_pm3(atomic_number) = 0.5885862D0
    AQ_pm3(atomic_number) = 0.7647667D0
    alp_pm3(atomic_number) = 2.7078070D0
    USS_pm3(atomic_number) = -47.2703200D0
    UPP_pm3(atomic_number) = -36.2669180D0

    !PDDG/PM3
    ! Reference Repasky, Chandrasekhar and Jorgensen, 2002, J Comp Chem, 
    !           23: 1601-1622
    element_supported_pddgpm3(atomic_number) = .true.
    elec_eng_pddgpm3(atomic_number) = -113.42824208974D0
    s_orb_exp_pddgpm3(atomic_number) = 1.56786358751710D0
    p_orb_exp_pddgpm3(atomic_number) = 1.84665852120070D0
    betas_pddgpm3(atomic_number) = -11.952818190434D0
    betap_pddgpm3(atomic_number) = -9.9224112120852D0
    FN1_pddgpm3(1,atomic_number) = 0.04890550330860D0
    FN2_pddgpm3(1,atomic_number) = 5.76533980799120D0
    FN3_pddgpm3(1,atomic_number) = 1.68223169651660D0
    FN1_pddgpm3(2,atomic_number) = 0.04769663311610D0
    FN2_pddgpm3(2,atomic_number) = 5.97372073873460D0
    FN3_pddgpm3(2,atomic_number) = 0.89440631619350D0
    FN1_pddgpm3(3,atomic_number) = 0.0d0
    FN2_pddgpm3(3,atomic_number) = 0.0d0
    FN3_pddgpm3(3,atomic_number) = 0.0d0
    FN1_pddgpm3(4,atomic_number) = 0.0d0
    FN2_pddgpm3(4,atomic_number) = 0.0d0
    FN3_pddgpm3(4,atomic_number) = 0.0d0
    NUM_FN_pddgpm3(atomic_number) = 2
    GSS_pddgpm3(atomic_number) = 11.2007080D0
    GSP_pddgpm3(atomic_number) = 10.2650270D0
    GPP_pddgpm3(atomic_number) = 10.7962920D0
    GP2_pddgpm3(atomic_number) = 9.0425660D0
    HSP_pddgpm3(atomic_number) = 2.2909800D0
    DD_pddgpm3(atomic_number) = 0.83141317761820D0
    QQ_pddgpm3(atomic_number) = 0.66322216984400D0
    AM_pddgpm3(atomic_number) = 0.41163939928160D0
    AD_pddgpm3(atomic_number) = 0.58929751233060D0
    AQ_pddgpm3(atomic_number) = 0.76594914927100D0
    alp_pddgpm3(atomic_number) = 2.72577212540530D0
    USS_pddgpm3(atomic_number) = -48.241240946951D0
    UPP_pddgpm3(atomic_number) = -36.461255999939D0
    PDDGC1_pm3(atomic_number) = -0.0007433618099D0
    PDDGC2_pm3(atomic_number) = 0.00098516072940D0
    PDDGE1_pm3(atomic_number) = 0.83691519687330D0
    PDDGE2_pm3(atomic_number) = 1.58523608520060D0

    !PDDG/MNDO
    ! Reference: Repasky, Chandrasekhar and Jorgensen, 2002, J Comp Chem,
    !            23: 1601-1622
    element_supported_pddgmndo(atomic_number) = .true.
    elec_eng_pddgmndo(atomic_number) = -123.86441152368D0
    s_orb_exp_pddgmndo(atomic_number) = 1.80981702301050D0
    p_orb_exp_pddgmndo(atomic_number) = 1.82500792388930D0
    betas_pddgmndo(atomic_number) = -18.841334137411D0
    betap_pddgmndo(atomic_number) = -7.9222341225346D0
    GSS_pddgmndo(atomic_number) = 12.23D00
    GSP_pddgmndo(atomic_number) = 11.47D00
    GPP_pddgmndo(atomic_number) = 11.08D00
    GP2_pddgmndo(atomic_number) = 9.84D00 
    HSP_pddgmndo(atomic_number) = 2.43D00
    DD_pddgmndo(atomic_number) = 0.79415790778110D0
    QQ_pddgmndo(atomic_number) = 0.67109016643690D0
    AM_pddgmndo(atomic_number) = 0.44946710986610D0
    AD_pddgmndo(atomic_number) = 0.62058137837870D0
    AQ_pddgmndo(atomic_number) = 0.67810055293470D0
    alp_pddgmndo(atomic_number) = 2.55552238806810D0
    USS_pddgmndo(atomic_number) = -53.837582488984D0
    UPP_pddgmndo(atomic_number) = -39.936408766823D0
    PDDGC1_mndo(atomic_number) = -0.0068893327627D0
    PDDGC2_mndo(atomic_number) = -0.0277514418977D0
    PDDGE1_mndo(atomic_number) = 1.19245557326430D0
    PDDGE2_mndo(atomic_number) = 1.32952163414800D0

    !-------------------
    !END CARBON
    !-------------------

    !-------------------
    !NITROGEN
    !-------------------
    atomic_number = 7
    !MNDO
    ! Reference: M.J.S. DEWAR, W. THIEL, J. AM. CHEM. SOC., 99, 4899, 
    !            (1977)
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -202.5662010D0 ! Dynamo has -202.5812010 Here
    s_orb_exp_mndo(atomic_number) = 2.2556140D0
    p_orb_exp_mndo(atomic_number) = 2.2556140D0
    betas_mndo(atomic_number) = -20.4957580D0
    betap_mndo(atomic_number) = -20.4957580D0
    GSS_mndo(atomic_number) = 13.59D00
    GSP_mndo(atomic_number) = 12.66D00 
    GPP_mndo(atomic_number) = 12.98D00
    GP2_mndo(atomic_number) = 11.59D00
    HSP_mndo(atomic_number) = 3.14D00
    DD_mndo(atomic_number) = 0.6399037D0
    QQ_mndo(atomic_number) = 0.5429763D0
    AM_mndo(atomic_number) = 0.4994487D0
    AD_mndo(atomic_number) = 0.7843643D0
    AQ_mndo(atomic_number) = 0.81264450D0 ! Dynamo has 0.8144720 here
    alp_mndo(atomic_number) = 2.8613420D0
    USS_mndo(atomic_number) = -71.9321220D0
    UPP_mndo(atomic_number) = -57.1723190D0

    !AM1
    ! Reference:  M.J.S. DEWAR ET AL, J. AM. CHEM. SOC. 107 3902-3909
    !             (1985)
    element_supported_am1(atomic_number) = .true.
    elec_eng_am1(atomic_number) = -202.4077430D0
    s_orb_exp_am1(atomic_number) = 2.3154100D0
    p_orb_exp_am1(atomic_number) = 2.1579400D0
    betas_am1(atomic_number) = -20.2991100D0
    betap_am1(atomic_number) = -18.2386660D0
    FN1_am1(1,atomic_number) = 0.0252510D0
    FN2_am1(1,atomic_number) = 5.0000000D0
    FN3_am1(1,atomic_number) = 1.5000000D0
    FN1_am1(2,atomic_number) = 0.0289530D0
    FN2_am1(2,atomic_number) = 5.0000000D0
    FN3_am1(2,atomic_number) = 2.1000000D0
    FN1_am1(3,atomic_number) =-0.0058060D0
    FN2_am1(3,atomic_number) = 2.0000000D0
    FN3_am1(3,atomic_number) = 2.4000000D0
    FN1_am1(4,atomic_number) = 0.0000000D0
    FN2_am1(4,atomic_number) = 0.0000000D0
    FN3_am1(4,atomic_number) = 0.0000000D0
    NUM_FN_am1(atomic_number) = 3
    GSS_am1(atomic_number) = 13.5900000D0
    GSP_am1(atomic_number) = 12.6600000D0
    GPP_am1(atomic_number) = 12.9800000D0
    GP2_am1(atomic_number) = 11.5900000D0
    HSP_am1(atomic_number) = 3.1400000D0
    DD_am1(atomic_number) = 0.6433247D0
    QQ_am1(atomic_number) = 0.5675528D0
    AM_am1(atomic_number) = 0.4994487D0
    AD_am1(atomic_number) = 0.7820840D0
    AQ_am1(atomic_number) = 0.7883498D0
    alp_am1(atomic_number) = 2.9472860D0
    USS_am1(atomic_number) = -71.8600000D0
    UPP_am1(atomic_number) = -57.1675810D0

    !PM3
    ! Reference:  J. J. P. STEWART, J. COMP. CHEM.10, 209 (1989).
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -157.6137755D0
    s_orb_exp_pm3(atomic_number) = 2.0280940D0
    p_orb_exp_pm3(atomic_number) = 2.3137280D0
    betas_pm3(atomic_number) = -14.0625210D0
    betap_pm3(atomic_number) = -20.0438480D0
    FN1_pm3(1,atomic_number) = 1.5016740D0
    FN2_pm3(1,atomic_number) = 5.9011480D0
    FN3_pm3(1,atomic_number) = 1.7107400D0
    FN1_pm3(2,atomic_number) = -1.5057720D0
    FN2_pm3(2,atomic_number) = 6.0046580D0
    FN3_pm3(2,atomic_number) = 1.7161490D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 11.9047870D0
    GSP_pm3(atomic_number) = 7.3485650D0
    GPP_pm3(atomic_number) = 11.7546720D0
    GP2_pm3(atomic_number) = 10.8072770D0
    HSP_pm3(atomic_number) = 1.1367130D0
    DD_pm3(atomic_number) = 0.6577006D0
    QQ_pm3(atomic_number) = 0.5293383D0
    AM_pm3(atomic_number) = 0.4375151D0
    AD_pm3(atomic_number) = 0.5030995D0
    AQ_pm3(atomic_number) = 0.7364933D0
    alp_pm3(atomic_number) = 2.8305450D0
    USS_pm3(atomic_number) = -49.3356720D0
    UPP_pm3(atomic_number) = -47.5097360D0

    !PDDG/PM3
    ! Reference Repasky, Chandrasekhar and Jorgensen, 2002, J Comp Chem,
    !           23: 1601-1622
    element_supported_pddgpm3(atomic_number) = .true.
    elec_eng_pddgpm3(atomic_number) = -158.41620481951D0
    s_orb_exp_pddgpm3(atomic_number) = 2.03580684361910D0
    p_orb_exp_pddgpm3(atomic_number) = 2.32432725808280D0
    betas_pddgpm3(atomic_number) = -14.117229602371D0
    betap_pddgpm3(atomic_number) = -19.938508878969D0
    FN1_pddgpm3(1,atomic_number) = 1.51332030575080D0
    FN2_pddgpm3(1,atomic_number) = 5.90439402634500D0
    FN3_pddgpm3(1,atomic_number) = 1.72837621719040D0
    FN1_pddgpm3(2,atomic_number) = -1.5118916914302D0
    FN2_pddgpm3(2,atomic_number) = 6.03001440913320D0
    FN3_pddgpm3(2,atomic_number) = 1.73410826456840D0
    FN1_pddgpm3(3,atomic_number) = 0.0d0
    FN2_pddgpm3(3,atomic_number) = 0.0d0
    FN3_pddgpm3(3,atomic_number) = 0.0d0
    FN1_pddgpm3(4,atomic_number) = 0.0d0
    FN2_pddgpm3(4,atomic_number) = 0.0d0
    FN3_pddgpm3(4,atomic_number) = 0.0d0
    NUM_FN_pddgpm3(atomic_number) = 2
    GSS_pddgpm3(atomic_number) = 11.9047870D0
    GSP_pddgpm3(atomic_number) = 7.3485650D0
    GPP_pddgpm3(atomic_number) = 11.7546720D0
    GP2_pddgpm3(atomic_number) = 10.8072770D0
    HSP_pddgpm3(atomic_number) = 1.1367130D0
    DD_pddgpm3(atomic_number) = 0.65485453533410D0
    QQ_pddgpm3(atomic_number) = 0.52692445400390D0
    AM_pddgpm3(atomic_number) = 0.43751514361910D0
    AD_pddgpm3(atomic_number) = 0.50442060161600D0
    AQ_pddgpm3(atomic_number) = 0.73887610477190D0
    alp_pddgpm3(atomic_number) = 2.84912399973850D0
    USS_pddgpm3(atomic_number) = -49.454546358059D0
    UPP_pddgpm3(atomic_number) = -47.757406358412D0
    PDDGC1_pm3(atomic_number) = -0.0031600751673D0
    PDDGC2_pm3(atomic_number) = 0.01250092178130D0
    PDDGE1_pm3(atomic_number) = 1.00417177651930D0
    PDDGE2_pm3(atomic_number) = 1.51633618021020D0

    !PDDG/MNDO
    ! Reference: Repasky, Chandrasekhar and Jorgensen, 2002, J Comp Chem,
    !            23: 1601-1622
    element_supported_pddgmndo(atomic_number) = .true.
    elec_eng_pddgmndo(atomic_number) = -206.46662581723320D0
    s_orb_exp_pddgmndo(atomic_number) = 2.23142379586030D0
    p_orb_exp_pddgmndo(atomic_number) = 2.25345995688440D0
    betas_pddgmndo(atomic_number) = -20.37577411084280D0
    betap_pddgmndo(atomic_number) = -21.08537341050740D0
    GSS_pddgmndo(atomic_number) = 13.59D00
    GSP_pddgmndo(atomic_number) = 12.66D00
    GPP_pddgmndo(atomic_number) = 12.98D00
    GP2_pddgmndo(atomic_number) = 11.59D00
    HSP_pddgmndo(atomic_number) = 3.14D00
    DD_pddgmndo(atomic_number) = 0.64362354949890D0
    QQ_pddgmndo(atomic_number) = 0.54349528938820D0
    AM_pddgmndo(atomic_number) = 0.49944873451190D0
    AD_pddgmndo(atomic_number) = 0.78188576956680D0
    AQ_pddgmndo(atomic_number) = 0.81211139492050D0
    alp_pddgmndo(atomic_number) = 2.84367788492060D0
    USS_pddgmndo(atomic_number) = -71.87189435530930D0
    UPP_pddgmndo(atomic_number) = -58.21661676886340D0
    PDDGC1_mndo(atomic_number) = 0.03502693409010D0
    PDDGC2_mndo(atomic_number) = -0.00172140634590D0
    PDDGE1_mndo(atomic_number) = 1.01162987147870D0
    PDDGE2_mndo(atomic_number) = 2.27842256966070D0

    !-------------------
    !END NITROGEN
    !-------------------

    !-------------------
    !OXYGEN
    !-------------------
    atomic_number = 8
    !MNDO
    ! Reference: M.J.S. DEWAR, W. THIEL, J. AM. CHEM. SOC., 99, 4899, 
    !            (1977)
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -317.8685060D0
    s_orb_exp_mndo(atomic_number) = 2.6999050D0
    p_orb_exp_mndo(atomic_number) = 2.6999050D0
    betas_mndo(atomic_number) = -32.6880820D0
    betap_mndo(atomic_number) = -32.6880820D0
    GSS_mndo(atomic_number) = 15.42D00
    GSP_mndo(atomic_number) = 14.48D00 
    GPP_mndo(atomic_number) = 14.52D00
    GP2_mndo(atomic_number) = 12.98D00 
    HSP_mndo(atomic_number) = 3.94D00
    DD_mndo(atomic_number) = 0.5346024D0
    QQ_mndo(atomic_number) = 0.4536252D0
    AM_mndo(atomic_number) = 0.5667034D0
    AD_mndo(atomic_number) = 0.9592562D0
    AQ_mndo(atomic_number) = 0.9495934D0
    alp_mndo(atomic_number) = 3.1606040D0
    USS_mndo(atomic_number) = -99.6443090D0
    UPP_mndo(atomic_number) = -77.7974720D0

    !AM1
    ! Reference: M.J.S. DEWAR ET AL, J. AM. CHEM. SOC. 107 3902-3909
    !            (1985)
    element_supported_am1(atomic_number) = .true.
    elec_eng_am1(atomic_number) = -316.0995200D0
    s_orb_exp_am1(atomic_number) = 3.1080320D0
    p_orb_exp_am1(atomic_number) = 2.5240390D0
    betas_am1(atomic_number) = -29.2727730D0
    betap_am1(atomic_number) = -29.2727730D0
    FN1_am1(1,atomic_number) = 0.2809620D0
    FN2_am1(1,atomic_number) = 5.0000000D0
    FN3_am1(1,atomic_number) = 0.8479180D0
    FN1_am1(2,atomic_number) = 0.0814300D0
    FN2_am1(2,atomic_number) = 7.0000000D0
    FN3_am1(2,atomic_number) = 1.4450710D0
    FN1_am1(3,atomic_number) = 0.0000000D0
    FN2_am1(3,atomic_number) = 0.0000000D0
    FN3_am1(3,atomic_number) = 0.0000000D0
    FN1_am1(4,atomic_number) = 0.0000000D0
    FN2_am1(4,atomic_number) = 0.0000000D0
    FN3_am1(4,atomic_number) = 0.0000000D0
    NUM_FN_am1(atomic_number) = 2
    GSS_am1(atomic_number) = 15.4200000D0
    GSP_am1(atomic_number) = 14.4800000D0
    GPP_am1(atomic_number) = 14.5200000D0
    GP2_am1(atomic_number) = 12.9800000D0
    HSP_am1(atomic_number) = 3.9400000D0
    DD_am1(atomic_number) = 0.4988896D0
    QQ_am1(atomic_number) = 0.4852322D0
    AM_am1(atomic_number) = 0.5667034D0
    AD_am1(atomic_number) = 0.9961066D0
    AQ_am1(atomic_number) = 0.9065223D0
    alp_am1(atomic_number) = 4.4553710D0
    USS_am1(atomic_number) = -97.8300000D0
    UPP_am1(atomic_number) = -78.2623800D0

    !PM3
    ! Reference:  J. J. P. STEWART, J. COMP. CHEM. 10, 209 (1989).
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -289.3422065D0
    s_orb_exp_pm3(atomic_number) = 3.7965440D0
    p_orb_exp_pm3(atomic_number) = 2.3894020D0
    betas_pm3(atomic_number) = -45.2026510D0
    betap_pm3(atomic_number) = -24.7525150D0
    FN1_pm3(1,atomic_number) = -1.1311280D0
    FN2_pm3(1,atomic_number) = 6.0024770D0
    FN3_pm3(1,atomic_number) = 1.6073110D0 
    FN1_pm3(2,atomic_number) = 1.1378910D0
    FN2_pm3(2,atomic_number) = 5.9505120D0
    FN3_pm3(2,atomic_number) = 1.5983950D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 15.7557600D0
    GSP_pm3(atomic_number) = 10.6211600D0
    GPP_pm3(atomic_number) = 13.6540160D0
    GP2_pm3(atomic_number) = 12.4060950D0
    HSP_pm3(atomic_number) = 0.5938830D0
    DD_pm3(atomic_number) = 0.4086173D0
    QQ_pm3(atomic_number) = 0.5125738D0
    AM_pm3(atomic_number) = 0.5790430D0
    AD_pm3(atomic_number) = 0.5299630D0
    AQ_pm3(atomic_number) = 0.8179630D0
    alp_pm3(atomic_number) = 3.2171020D0
    USS_pm3(atomic_number) = -86.9930020D0
    UPP_pm3(atomic_number) = -71.8795800D0

    !PDDG/PM3
    ! Reference Repasky, Chandrasekhar and Jorgensen, 2002, J Comp Chem,
    !           23: 1601-1622
    element_supported_pddgpm3(atomic_number) = .true.
    elec_eng_pddgpm3(atomic_number) = -292.18876564023D0
    s_orb_exp_pddgpm3(atomic_number) = 3.81456531095080D0
    p_orb_exp_pddgpm3(atomic_number) = 2.31801122165690D0
    betas_pddgpm3(atomic_number) = -44.874553472211D0
    betap_pddgpm3(atomic_number) = -24.601939339720D0
    FN1_pddgpm3(1,atomic_number) = -1.1384554300359D0
    FN2_pddgpm3(1,atomic_number) = 6.00004254473730D0
    FN3_pddgpm3(1,atomic_number) = 1.62236167639400D0
    FN1_pddgpm3(2,atomic_number) = 1.14600702743950D0
    FN2_pddgpm3(2,atomic_number) = 5.96349383486760D0
    FN3_pddgpm3(2,atomic_number) = 1.61478803799000D0
    FN1_pddgpm3(3,atomic_number) = 0.0d0
    FN2_pddgpm3(3,atomic_number) = 0.0d0
    FN3_pddgpm3(3,atomic_number) = 0.0d0
    FN1_pddgpm3(4,atomic_number) = 0.0d0
    FN2_pddgpm3(4,atomic_number) = 0.0d0
    FN3_pddgpm3(4,atomic_number) = 0.0d0
    NUM_FN_pddgpm3(atomic_number) = 2
    GSS_pddgpm3(atomic_number) = 15.7557600D0
    GSP_pddgpm3(atomic_number) = 10.6211600D0
    GPP_pddgpm3(atomic_number) = 13.6540160D0
    GP2_pddgpm3(atomic_number) = 12.4060950D0
    HSP_pddgpm3(atomic_number) = 0.5938830D0
    DD_pddgpm3(atomic_number) = 0.40374106034030D0
    QQ_pddgpm3(atomic_number) = 0.52836019944550D0
    AM_pddgpm3(atomic_number) = 0.57904300171250D0
    AD_pddgpm3(atomic_number) = 0.53403626392780D0
    AQ_pddgpm3(atomic_number) = 0.80090914697820D0
    alp_pddgpm3(atomic_number) = 3.22530882036500D0
    USS_pddgpm3(atomic_number) = -87.412505208248D0
    UPP_pddgpm3(atomic_number) = -72.183069806393D0
    PDDGC1_pm3(atomic_number) = -0.00099962677420D0
    PDDGC2_pm3(atomic_number) = -0.00152161350520D0
    PDDGE1_pm3(atomic_number) = 1.36068502987020D0
    PDDGE2_pm3(atomic_number) = 1.36640659538530D0

    !PDDG/MNDO
    ! Reference: Repasky, Chandrasekhar and Jorgensen, 2002, J Comp Chem,
    !            23: 1601-1622
    element_supported_pddgmndo(atomic_number) = .true.
    elec_eng_pddgmndo(atomic_number) = -310.87974465179D0
    s_orb_exp_pddgmndo(atomic_number) = 2.56917199926090D0
    p_orb_exp_pddgmndo(atomic_number) = 2.69715151721790D0
    betas_pddgmndo(atomic_number) = -33.606335624658D0
    betap_pddgmndo(atomic_number) = -27.984442042827D0
    GSS_pddgmndo(atomic_number) = 15.42D00
    GSP_pddgmndo(atomic_number) = 14.48D00
    GPP_pddgmndo(atomic_number) = 14.52D00
    GP2_pddgmndo(atomic_number) = 12.98D00
    HSP_pddgmndo(atomic_number) = 3.94D00
    DD_pddgmndo(atomic_number) = 0.54734406013390D0
    QQ_pddgmndo(atomic_number) = 0.45408827185760D0
    AM_pddgmndo(atomic_number) = 0.56670342061620D0
    AD_pddgmndo(atomic_number) = 0.94709991169520D0
    AQ_pddgmndo(atomic_number) = 0.94892420489320D0
    alp_pddgmndo(atomic_number) = 3.23884200872830D0
    USS_pddgmndo(atomic_number) = -97.884970179897D0
    UPP_pddgmndo(atomic_number) = -77.342673804072D0
    PDDGC1_mndo(atomic_number) = 0.08634413812890D0
    PDDGC2_mndo(atomic_number) = 0.03040342779910D0
    PDDGE1_mndo(atomic_number) = 0.72540784783600D0
    PDDGE2_mndo(atomic_number) = 0.70972848794410D0

    !PM3CARB-1
    !Reference: McNamara, J.P., Muslim, A.M., Abdel-Aal, H., Wang, H., 
    !           Mohr, M., Hillier, I.H., Bryce, R.A.,
    !           2004, Chem Phys Lett, 394, 429-436
    !Note: PM3CARB-1 is not a complete parameter set, it is simply PM3 
    !      with a few parameters changed for Oxygen and Hydrogen.
    !      So when loading these parameters into the correct structure 
    !      the code needs to load pm3 parameters except for those that
    !      are different for PM3CARB-1
    elec_eng_PM3CARB1(atomic_number) = -317.442829D0
    USS_PM3CARB1(atomic_number) = -90.938073D0
    UPP_PM3CARB1(atomic_number) = -76.932200D0
    betas_PM3CARB1(atomic_number) = -44.449581D0
    betap_PM3CARB1(atomic_number) = -35.343869D0
    alp_PM3CARB1(atomic_number) = 3.031867D0

    !-------------------
    !END OXYGEN
    !-------------------

    !-------------------
    !FLUORINE
    !-------------------
    atomic_number = 9
    !MNDO
    ! Reference: M.J.S. DEWAR, H.S. RZEPA, J. AM. CHEM. SOC., 100, 777,
    !            (1978)
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -476.6837810D0
    s_orb_exp_mndo(atomic_number) = 2.8484870D0
    p_orb_exp_mndo(atomic_number) = 2.8484870D0
    betas_mndo(atomic_number) = -48.2904660D0
    betap_mndo(atomic_number) = -36.5085400D0
    GSS_mndo(atomic_number) = 16.92D00
    GSP_mndo(atomic_number) = 17.25D00
    GPP_mndo(atomic_number) = 16.71D00
    GP2_mndo(atomic_number) = 14.91D00
    HSP_mndo(atomic_number) = 4.83D00
    DD_mndo(atomic_number) = 0.5067166D0
    QQ_mndo(atomic_number) = 0.4299633D0
    AM_mndo(atomic_number) = 0.6218302D0
    AD_mndo(atomic_number) = 1.0850301D0
    AQ_mndo(atomic_number) = 1.0343643D0
    alp_mndo(atomic_number) = 3.4196606D0
    USS_mndo(atomic_number) = -131.0715480D0
    UPP_mndo(atomic_number) = -105.7821370D0

    !AM1
    ! Reference: M.J.S. DEWAR AND E. G. ZOEBISCH, THEOCHEM, 180, 1 (1988).
    element_supported_am1(atomic_number) = .true.
    elec_eng_am1(atomic_number) = -482.2905830D0
    s_orb_exp_am1(atomic_number) = 3.7700820D0
    p_orb_exp_am1(atomic_number) = 2.4946700D0
    betas_am1(atomic_number) = -69.5902770D0
    betap_am1(atomic_number) = -27.9223600D0
    FN1_am1(1,atomic_number) = 0.2420790D0
    FN2_am1(1,atomic_number) = 4.8000000D0
    FN3_am1(1,atomic_number) = 0.9300000D0
    FN1_am1(2,atomic_number) = 0.0036070D0
    FN2_am1(2,atomic_number) = 4.6000000D0
    FN3_am1(2,atomic_number) = 1.6600000D0
    FN1_am1(3,atomic_number) = 0.0000000D0
    FN2_am1(3,atomic_number) = 0.0000000D0
    FN3_am1(3,atomic_number) = 0.0000000D0
    FN1_am1(4,atomic_number) = 0.0000000D0
    FN2_am1(4,atomic_number) = 0.0000000D0
    FN3_am1(4,atomic_number) = 0.0000000D0
    NUM_FN_am1(atomic_number) = 2
    GSS_am1(atomic_number) = 16.9200000D0
    GSP_am1(atomic_number) = 17.2500000D0
    GPP_am1(atomic_number) = 16.7100000D0
    GP2_am1(atomic_number) = 14.9100000D0
    HSP_am1(atomic_number) = 4.8300000D0
    DD_am1(atomic_number) = 0.4145203D0
    QQ_am1(atomic_number) = 0.4909446D0
    AM_am1(atomic_number) = 0.6218302D0
    AD_am1(atomic_number) = 1.2088792D0
    AQ_am1(atomic_number) = 0.9449355D0
    alp_am1(atomic_number) = 5.5178000D0
    USS_am1(atomic_number) = -136.1055790D0
    UPP_am1(atomic_number) = -104.8898850D0

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM. 10, 209 (1989).
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -437.5171690D0
    s_orb_exp_pm3(atomic_number) = 4.7085550D0
    p_orb_exp_pm3(atomic_number) = 2.4911780D0
    betas_pm3(atomic_number) = -48.4059390D0
    betap_pm3(atomic_number) = -27.7446600D0
    FN1_pm3(1,atomic_number) = -0.0121660D0
    FN2_pm3(1,atomic_number) = 6.0235740D0
    FN3_pm3(1,atomic_number) = 1.8568590D0
    FN1_pm3(2,atomic_number) = -0.0028520D0
    FN2_pm3(2,atomic_number) = 6.0037170D0
    FN3_pm3(2,atomic_number) = 2.6361580D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 10.4966670D0
    GSP_pm3(atomic_number) = 16.0736890D0
    GPP_pm3(atomic_number) = 14.8172560D0
    GP2_pm3(atomic_number) = 14.4183930D0
    HSP_pm3(atomic_number) = 0.7277630D0
    DD_pm3(atomic_number) = 0.3125302D0
    QQ_pm3(atomic_number) = 0.4916328D0
    AM_pm3(atomic_number) = 0.3857650D0
    AD_pm3(atomic_number) = 0.6768503D0
    AQ_pm3(atomic_number) = 0.6120047D0
    alp_pm3(atomic_number) = 3.3589210D0
    USS_pm3(atomic_number) = -110.4353030D0
    UPP_pm3(atomic_number) = -105.6850470D0

    !PDDG/PM3
    ! Reference Tubert-Brohman, Werneck, Repasky and Jorgensen, 2003, 
    !           J Comp Chem, 25: 138-150
    element_supported_pddgpm3(atomic_number) = .true.
    elec_eng_pddgpm3(atomic_number) = -442.457133D0
    s_orb_exp_pddgpm3(atomic_number) = 5.538033D0
    p_orb_exp_pddgpm3(atomic_number) = 2.538066D0
    betas_pddgpm3(atomic_number) = -50.937301D0
    betap_pddgpm3(atomic_number) = -31.636976D0
    FN1_pddgpm3(1,atomic_number) = -0.008079D0
    FN2_pddgpm3(1,atomic_number) = 5.938969D0
    FN3_pddgpm3(1,atomic_number) = 1.863949D0
    FN1_pddgpm3(2,atomic_number) = -0.002659D0
    FN2_pddgpm3(2,atomic_number) = 5.925105D0
    FN3_pddgpm3(2,atomic_number) = 2.388864D0
    FN1_pddgpm3(3,atomic_number) = 0.0d0
    FN2_pddgpm3(3,atomic_number) = 0.0d0
    FN3_pddgpm3(3,atomic_number) = 0.0d0
    FN1_pddgpm3(4,atomic_number) = 0.0d0
    FN2_pddgpm3(4,atomic_number) = 0.0d0
    FN3_pddgpm3(4,atomic_number) = 0.0d0
    NUM_FN_pddgpm3(atomic_number) = 2
    GSS_pddgpm3(atomic_number) = 10.4966670D0
    GSP_pddgpm3(atomic_number) = 16.0736890D0
    GPP_pddgpm3(atomic_number) = 14.8172560D0
    GP2_pddgpm3(atomic_number) = 14.4183930D0
    HSP_pddgpm3(atomic_number) = 0.7277630D0
    DD_pddgpm3(atomic_number) = 0.246601D0
    QQ_pddgpm3(atomic_number) = 0.482551D0
    AM_pddgpm3(atomic_number) = 0.3857649642D0
    AD_pddgpm3(atomic_number) = 0.7878569189D0
    AQ_pddgpm3(atomic_number) = 0.620499825D0
    alp_pddgpm3(atomic_number) = 3.200571D0
    USS_pddgpm3(atomic_number) = -111.400432D0
    UPP_pddgpm3(atomic_number) = -106.395264D0
    PDDGC1_pm3(atomic_number) = -0.012866D0
    PDDGC2_pm3(atomic_number) = 0.007315D0
    PDDGE1_pm3(atomic_number) = 1.305681D0
    PDDGE2_pm3(atomic_number) = 1.842572D0

    !PDDG/MNDO
    ! Reference: Repasky, Chandrasekhar and Jorgensen, 2002, 
    !            J Comp Chem, 23: 1601-1622
    element_supported_pddgmndo(atomic_number) = .true.
    elec_eng_pddgmndo(atomic_number) = -488.703243D0
    s_orb_exp_pddgmndo(atomic_number) = 4.328519D0
    p_orb_exp_pddgmndo(atomic_number) = 2.905042D0
    betas_pddgmndo(atomic_number) = -67.827612D0
    betap_pddgmndo(atomic_number) = -40.924818D0
    GSS_pddgmndo(atomic_number) = 16.92D00
    GSP_pddgmndo(atomic_number) = 17.25D00
    GPP_pddgmndo(atomic_number) = 16.71D00
    GP2_pddgmndo(atomic_number) = 14.91D00
    HSP_pddgmndo(atomic_number) = 4.83D00
    DD_pddgmndo(atomic_number) = 0.361556D0
    QQ_pddgmndo(atomic_number) = 0.421593D0
    AM_pddgmndo(atomic_number) = 0.6218302205D0
    AD_pddgmndo(atomic_number) = 1.303600806D0
    AQ_pddgmndo(atomic_number) = 1.048409249D0
    alp_pddgmndo(atomic_number) = 3.322382D0
    USS_pddgmndo(atomic_number) = -134.220379D0
    UPP_pddgmndo(atomic_number) = -107.155961D0
    PDDGC1_mndo(atomic_number) = -0.011579D0
    PDDGC2_mndo(atomic_number) = -0.012943D0
    PDDGE1_mndo(atomic_number) = 0.834606D0
    PDDGE2_mndo(atomic_number) = 1.875603D0

    !-------------------
    !END FLUORINE
    !-------------------

    !-------------------
    !MAGNESIUM
    !-------------------
    atomic_number = 12
    !MNDO
    ! Reference: NONE
    element_supported_mndo(atomic_number) = .false.

    !AM1
    ! Reference: M.C. HUTTER, J.R. REIMERS, N.S. HUSH,
    !            J.PHYS.CHEM. (1998)
    element_supported_am1(atomic_number) = .true.
    elec_eng_am1(atomic_number) = -22.43786349D0
    s_orb_exp_am1(atomic_number) = 1.22339270D0
    p_orb_exp_am1(atomic_number) = 1.02030798D0
    betas_am1(atomic_number) = -1.25974355D0
    betap_am1(atomic_number) = -0.77836604D0
    FN1_am1(1,atomic_number) = 2.55017735D0
    FN2_am1(1,atomic_number) = 4.29397225D0
    FN3_am1(1,atomic_number) = 0.79989601D0
    FN1_am1(2,atomic_number) = -0.00565806D0
    FN2_am1(2,atomic_number) = 2.96053910D0
    FN3_am1(2,atomic_number) = 1.47499983D0
    FN1_am1(3,atomic_number) = -0.00610286D0
    FN2_am1(3,atomic_number) = 2.61416919D0
    FN3_am1(3,atomic_number) = 2.42604040D0
    FN1_am1(4,atomic_number) = 0.0000000D0
    FN2_am1(4,atomic_number) = 0.0000000D0
    FN3_am1(4,atomic_number) = 0.0000000D0
    NUM_FN_am1(atomic_number) = 3
    GSS_am1(atomic_number) = 7.50132277D0
    GSP_am1(atomic_number) = 6.34591536D0
    GPP_am1(atomic_number) = 4.77534467D0
    GP2_am1(atomic_number) = 4.34017279D0
    HSP_am1(atomic_number) = 0.48930466D0
    DD_am1(atomic_number) = 1.75012115D0
    QQ_am1(atomic_number) = 1.64001467D0
    AM_am1(atomic_number) = 0.27568257D0
    AD_am1(atomic_number) = 0.19957236D0
    AQ_am1(atomic_number) = 0.26440152D0
    alp_am1(atomic_number) =  1.67049799D0
    USS_am1(atomic_number) = -14.96959313D0 
    UPP_am1(atomic_number) = -11.56229248D0

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM.(ACCEPTED)
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -22.5530760D0
    s_orb_exp_pm3(atomic_number) = 0.6985520D0
    p_orb_exp_pm3(atomic_number) = 1.4834530D0
    betas_pm3(atomic_number) = -2.0716910D0
    betap_pm3(atomic_number) = -0.5695810D0
    FN1_pm3(1,atomic_number) = 2.1170500D0
    FN2_pm3(1,atomic_number) = 6.0094770D0
    FN3_pm3(1,atomic_number) = 2.0844060D0
    FN1_pm3(2,atomic_number) = -2.5477670D0
    FN2_pm3(2,atomic_number) = 4.3953700D0
    FN3_pm3(2,atomic_number) = 2.0636740D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 6.6943000D0
    GSP_pm3(atomic_number) = 6.7939950D0
    GPP_pm3(atomic_number) = 6.9104460D0
    GP2_pm3(atomic_number) = 7.0908230D0
    HSP_pm3(atomic_number) = 0.5433000D0
    DD_pm3(atomic_number) = 1.1403950D0
    QQ_pm3(atomic_number) = 1.1279899D0
    AM_pm3(atomic_number) = 0.2460235D0
    AD_pm3(atomic_number) = 0.2695751D0
    AQ_pm3(atomic_number) = 0.2767522D0
    alp_pm3(atomic_number) = 1.3291470D0
    USS_pm3(atomic_number) = -14.6236880D0
    UPP_pm3(atomic_number) = -14.1734600D0

    !-------------------
    !END MAGNESIUM
    !-------------------

    !-------------------
    !ALUMINIUM
    !-------------------
    atomic_number = 13
    !MNDO
    ! Reference: L.P. DAVIS, ET.AL.  J. COMP. CHEM., 2, 433, (1981)
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -44.4840720D0
    s_orb_exp_mndo(atomic_number) = 1.4441610D0
    p_orb_exp_mndo(atomic_number) = 1.4441610D0
    betas_mndo(atomic_number) = -2.6702840D0
    betap_mndo(atomic_number) = -2.6702840D0
    GSS_mndo(atomic_number) = 8.09D00
    GSP_mndo(atomic_number) = 6.63D00
    GPP_mndo(atomic_number) = 5.98D00
    GP2_mndo(atomic_number) = 5.40D00
    HSP_mndo(atomic_number) = 0.70D00
    DD_mndo(atomic_number) = 1.3992387D0
    QQ_mndo(atomic_number) = 1.1586797D0
    AM_mndo(atomic_number) = 0.2973172D0
    AD_mndo(atomic_number) = 0.2635574D0
    AQ_mndo(atomic_number) = 0.3673560D0
    alp_mndo(atomic_number) = 1.8688394D0
    USS_mndo(atomic_number) = -23.8070970D0
    UPP_mndo(atomic_number) = -17.5198780D0

    !AM1
    ! Reference: M. J. S. Dewar, A. J. Holder, Organometallics, 
    !            9, 508-511 (1990).
    element_supported_am1(atomic_number) = .true.
    elec_eng_am1(atomic_number) = -46.4208150D0
    s_orb_exp_am1(atomic_number) = 1.5165930D0
    p_orb_exp_am1(atomic_number) = 1.3063470D0
    betas_am1(atomic_number) = -3.8668220D0
    betap_am1(atomic_number) = -2.3171460D0
    FN1_am1(1,atomic_number) = 0.0900000D0
    FN2_am1(1,atomic_number) = 12.3924430D0
    FN3_am1(1,atomic_number) = 2.0503940D0
    FN1_am1(2,atomic_number) = 0.0000000D0
    FN2_am1(2,atomic_number) = 0.0000000D0
    FN3_am1(2,atomic_number) = 0.0000000D0
    FN1_am1(3,atomic_number) = 0.0000000D0
    FN2_am1(3,atomic_number) = 0.0000000D0
    FN3_am1(3,atomic_number) = 0.0000000D0
    FN1_am1(4,atomic_number) = 0.0000000D0
    FN2_am1(4,atomic_number) = 0.0000000D0
    FN3_am1(4,atomic_number) = 0.0000000D0
    NUM_FN_am1(atomic_number) = 1
    GSS_am1(atomic_number) = 8.0900000D0
    GSP_am1(atomic_number) = 6.6300000D0
    GPP_am1(atomic_number) = 5.9800000D0
    GP2_am1(atomic_number) = 5.4000000D0
    HSP_am1(atomic_number) = 0.7000000D0
    DD_am1(atomic_number) = 1.4040443D0
    QQ_am1(atomic_number) = 1.2809154D0
    AM_am1(atomic_number) = 0.2973172D0
    AD_am1(atomic_number) = 0.2630229D0
    AQ_am1(atomic_number) = 0.3427832D0
    alp_am1(atomic_number) = 1.9765860D0
    USS_am1(atomic_number) = -24.3535850D0
    UPP_am1(atomic_number) = -18.3636450D0

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM. 10, 209 (1989).
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -46.8647630D0
    s_orb_exp_pm3(atomic_number) = 1.7028880D0
    p_orb_exp_pm3(atomic_number) = 1.0736290D0
    betas_pm3(atomic_number) = -0.5943010D0
    betap_pm3(atomic_number) = -0.9565500D0
    FN1_pm3(1,atomic_number) = -0.4730900D0
    FN2_pm3(1,atomic_number) = 1.9158250D0
    FN3_pm3(1,atomic_number) = 1.4517280D0
    FN1_pm3(2,atomic_number) = -0.1540510D0
    FN2_pm3(2,atomic_number) = 6.0050860D0
    FN3_pm3(2,atomic_number) = 2.5199970D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 5.7767370D0
    GSP_pm3(atomic_number) = 11.6598560D0
    GPP_pm3(atomic_number) = 6.3477900D0
    GP2_pm3(atomic_number) = 6.1210770D0
    HSP_pm3(atomic_number) = 4.0062450D0
    DD_pm3(atomic_number) = 1.2102799D0
    QQ_pm3(atomic_number) = 1.5585645D0
    AM_pm3(atomic_number) = 0.2123020D0
    AD_pm3(atomic_number) = 0.6418584D0
    AQ_pm3(atomic_number) = 0.2262838D0
    alp_pm3(atomic_number) = 1.5217030D0
    USS_pm3(atomic_number) = -24.8454040D0
    UPP_pm3(atomic_number) = -22.2641590D0

    !-------------------
    !END ALUMINIUM
    !-------------------

    !-------------------
    !SILICON
    !-------------------
    atomic_number = 14
    !MNDO
    ! Reference: M.J.S.DEWAR, ET. AL. ORGANOMETALLICS  5, 375 (1986)
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -82.8394220D0
    s_orb_exp_mndo(atomic_number) = 1.3159860D0
    p_orb_exp_mndo(atomic_number) = 1.7099430D0
    betas_mndo(atomic_number) = -9.0868040D0
    betap_mndo(atomic_number) = -1.0758270D0
    GSS_mndo(atomic_number) = 9.82D00
    GSP_mndo(atomic_number) = 8.36D00
    GPP_mndo(atomic_number) = 7.31D00
    GP2_mndo(atomic_number) = 6.54D00
    HSP_mndo(atomic_number) = 1.32D00
    DD_mndo(atomic_number) = 1.2580349D0
    QQ_mndo(atomic_number) = 0.9785824D0
    AM_mndo(atomic_number) = 0.3608967D0
    AD_mndo(atomic_number) = 0.3664244D0
    AQ_mndo(atomic_number) = 0.4506740D0
    alp_mndo(atomic_number) = 2.2053160D0
    USS_mndo(atomic_number) = -37.0375330D0
    UPP_mndo(atomic_number) = -27.7696780D0

    !AM1
    ! Reference: M.J.S.DEWAR, C. JIE, ORGANOMETALLICS, 6, 1486-1490 
    !            (1987).
    element_supported_am1(atomic_number) = .true.
    elec_eng_am1(atomic_number) = -79.0017420D0
    s_orb_exp_am1(atomic_number) = 1.830697D0
    p_orb_exp_am1(atomic_number) = 1.2849530D0
    betas_am1(atomic_number) = -3.784852D0
    betap_am1(atomic_number) = -1.968123D0
    FN1_am1(1,atomic_number) = 0.25D0
    FN2_am1(1,atomic_number) = 9.000D0
    FN3_am1(1,atomic_number) = 0.911453D0
    FN1_am1(2,atomic_number) = 0.061513D0
    FN2_am1(2,atomic_number) = 5.00D0
    FN3_am1(2,atomic_number) = 1.995569D0
    FN1_am1(3,atomic_number) = 0.0207890D0
    FN2_am1(3,atomic_number) = 5.00D0
    FN3_am1(3,atomic_number) = 2.990610D0
    FN1_am1(4,atomic_number) = 0.0000000D0
    FN2_am1(4,atomic_number) = 0.0000000D0
    FN3_am1(4,atomic_number) = 0.0000000D0
    NUM_FN_am1(atomic_number) = 3
    GSS_am1(atomic_number) = 9.8200000D0
    GSP_am1(atomic_number) = 8.3600000D0
    GPP_am1(atomic_number) = 7.3100000D0
    GP2_am1(atomic_number) = 6.5400000D0
    HSP_am1(atomic_number) = 1.3200000D0
    DD_am1(atomic_number) = 1.1631107D0
    QQ_am1(atomic_number) = 1.3022422D0
    AM_am1(atomic_number) = 0.3608967D0
    AD_am1(atomic_number) = 0.3829813D0
    AQ_am1(atomic_number) = 0.3712106D0
    alp_am1(atomic_number) = 2.257816D0
    USS_am1(atomic_number) = -33.9536220D0
    UPP_am1(atomic_number) = -28.9347490D0

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM. 10, 209 (1989).
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -67.7882140D0
    s_orb_exp_pm3(atomic_number) = 1.6350750D0
    p_orb_exp_pm3(atomic_number) = 1.3130880D0
    betas_pm3(atomic_number) = -2.8621450D0
    betap_pm3(atomic_number) = -3.9331480D0
    FN1_pm3(1,atomic_number) = -0.3906000D0
    FN2_pm3(1,atomic_number) = 6.0000540D0
    FN3_pm3(1,atomic_number) = 0.6322620D0
    FN1_pm3(2,atomic_number) = 0.0572590D0
    FN2_pm3(2,atomic_number) = 6.0071830D0
    FN3_pm3(2,atomic_number) = 2.0199870D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 5.0471960D0
    GSP_pm3(atomic_number) = 5.9490570D0
    GPP_pm3(atomic_number) = 6.7593670D0
    GP2_pm3(atomic_number) = 5.1612970D0
    HSP_pm3(atomic_number) = 0.9198320D0
    DD_pm3(atomic_number) = 1.3144550D0
    QQ_pm3(atomic_number) = 1.2743396D0
    AM_pm3(atomic_number) = 0.1854905D0
    AD_pm3(atomic_number) = 0.3060715D0
    AQ_pm3(atomic_number) = 0.4877432D0
    alp_pm3(atomic_number) = 2.1358090D0
    USS_pm3(atomic_number) = -26.7634830D0
    UPP_pm3(atomic_number) = -22.8136350D0

    !-------------------
    !END SILICON
    !-------------------

    !-------------------
    !PHOSPHORUS
    !-------------------
    atomic_number = 15
    !MNDO
    ! Reference: M.J.S.DEWAR, M.L.MCKEE, H.S.RZEPA,J. AM. CHEM. SOC.,
    !            100 3607 1978
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -152.9599600D0
    s_orb_exp_mndo(atomic_number) = 2.1087200D0
    p_orb_exp_mndo(atomic_number) = 1.7858100D0
    betas_mndo(atomic_number) = -6.7916000D0
    betap_mndo(atomic_number) = -6.7916000D0
    GSS_mndo(atomic_number) = 11.56D00
    GSP_mndo(atomic_number) = 10.08D00
    GPP_mndo(atomic_number) = 8.64D00
    GP2_mndo(atomic_number) = 7.68D00
    HSP_mndo(atomic_number) = 1.92D00
    DD_mndo(atomic_number) = 1.0129699D0
    QQ_mndo(atomic_number) = 0.9370090D0
    AM_mndo(atomic_number) = 0.4248438D0
    AD_mndo(atomic_number) = 0.4882420D0
    AQ_mndo(atomic_number) = 0.4979406D0
    alp_mndo(atomic_number) = 2.4152800D0
    USS_mndo(atomic_number) = -56.1433600D0
    UPP_mndo(atomic_number) = -42.8510800D0

    !AM1
    ! Reference: M.J.S.DEWAR, JIE, C, THEOCHEM, 187,1 (1989)
    element_supported_am1(atomic_number) = .true.
    elec_eng_am1(atomic_number) = -124.4368355D0
    s_orb_exp_am1(atomic_number) = 1.9812800D0
    p_orb_exp_am1(atomic_number) = 1.8751500D0
    betas_am1(atomic_number) = -6.3537640D0
    betap_am1(atomic_number) = -6.5907090D0
    FN1_am1(1,atomic_number) = -0.0318270D0
    FN2_am1(1,atomic_number) = 6.0000000D0
    FN3_am1(1,atomic_number) = 1.4743230D0
    FN1_am1(2,atomic_number) = 0.0184700D0
    FN2_am1(2,atomic_number) = 7.0000000D0
    FN3_am1(2,atomic_number) = 1.7793540D0
    FN1_am1(3,atomic_number) = 0.0332900D0
    FN2_am1(3,atomic_number) = 9.0000000D0
    FN3_am1(3,atomic_number) = 3.0065760D0
    FN1_am1(4,atomic_number) = 0.0000000D0
    FN2_am1(4,atomic_number) = 0.0000000D0
    FN3_am1(4,atomic_number) = 0.0000000D0
    NUM_FN_am1(atomic_number) = 3
    GSS_am1(atomic_number) = 11.5600050D0
    GSP_am1(atomic_number) = 5.2374490D0
    GPP_am1(atomic_number) = 7.8775890D0
    GP2_am1(atomic_number) = 7.3076480D0
    HSP_am1(atomic_number) = 0.7792380D0
    DD_am1(atomic_number) = 1.0452022D0
    QQ_am1(atomic_number) = 0.8923660D0
    AM_am1(atomic_number) = 0.4248440D0
    AD_am1(atomic_number) = 0.3275319D0
    AQ_am1(atomic_number) = 0.4386854D0
    alp_am1(atomic_number) = 2.4553220D0
    USS_am1(atomic_number) = -42.0298630D0
    UPP_am1(atomic_number) = -34.0307090D0

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM. 10, 209 (1989).
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -117.9591740D0
    s_orb_exp_pm3(atomic_number) = 2.0175630D0
    p_orb_exp_pm3(atomic_number) = 1.5047320D0
    betas_pm3(atomic_number) = -12.6158790D0
    betap_pm3(atomic_number) = -4.1600400D0
    FN1_pm3(1,atomic_number) = -0.6114210D0
    FN2_pm3(1,atomic_number) = 1.9972720D0
    FN3_pm3(1,atomic_number) = 0.7946240D0
    FN1_pm3(2,atomic_number) = -0.0939350D0
    FN2_pm3(2,atomic_number) = 1.9983600D0
    FN3_pm3(2,atomic_number) = 1.9106770D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 7.8016150D0
    GSP_pm3(atomic_number) = 5.1869490D0
    GPP_pm3(atomic_number) = 6.6184780D0
    GP2_pm3(atomic_number) = 6.0620020D0
    HSP_pm3(atomic_number) = 1.5428090D0
    DD_pm3(atomic_number) = 1.0644947D0
    QQ_pm3(atomic_number) = 1.1120386D0
    AM_pm3(atomic_number) = 0.2867187D0
    AD_pm3(atomic_number) = 0.4309446D0
    AQ_pm3(atomic_number) = 0.3732517D0
    alp_pm3(atomic_number) = 1.9405340D0
    USS_pm3(atomic_number) = -40.4130960D0
    UPP_pm3(atomic_number) = -29.5930520D0

    !-------------------
    !END PHOSPHORUS
    !-------------------

    !-------------------
    !SULPHUR
    !-------------------
    atomic_number = 16
    !MNDO
    ! Reference: M.J.S.DEWAR, C.H. REYNOLDS, J. COMP. CHEM. 7, 140-143
    !            (1986)
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -226.0123900D0
    s_orb_exp_mndo(atomic_number) = 2.3129620D0
    p_orb_exp_mndo(atomic_number) = 2.0091460D0
    betas_mndo(atomic_number) = -10.7616700D0
    betap_mndo(atomic_number) = -10.1084330D0
    GSS_mndo(atomic_number) = 12.88D00
    GSP_mndo(atomic_number) = 11.26D00
    GPP_mndo(atomic_number) = 9.90D00
    GP2_mndo(atomic_number) = 8.83D00
    HSP_mndo(atomic_number) = 2.26D00
    DD_mndo(atomic_number) = 0.9189935D0
    QQ_mndo(atomic_number) = 0.8328514D0
    AM_mndo(atomic_number) = 0.4733554D0
    AD_mndo(atomic_number) = 0.5544502D0
    AQ_mndo(atomic_number) = 0.5585244D0
    alp_mndo(atomic_number) = 2.4780260D0
    USS_mndo(atomic_number) = -72.2422810D0
    UPP_mndo(atomic_number) = -56.9732070D0

    !AM1
    ! Reference: M.J.S.DEWAR, Y-C YUAN, THEOCHEM, IN PRESS 
    element_supported_am1(atomic_number) = .true.
    elec_eng_am1(atomic_number) = -191.7321930D0
    s_orb_exp_am1(atomic_number) = 2.3665150D0
    p_orb_exp_am1(atomic_number) = 1.6672630D0
    betas_am1(atomic_number) = -3.9205660D0
    betap_am1(atomic_number) = -7.9052780D0
    FN1_am1(1,atomic_number) = -0.5091950D0
    FN2_am1(1,atomic_number) = 4.5936910D0
    FN3_am1(1,atomic_number) = 0.7706650D0
    FN1_am1(2,atomic_number) = -0.0118630D0
    FN2_am1(2,atomic_number) = 5.8657310D0
    FN3_am1(2,atomic_number) = 1.5033130D0
    FN1_am1(3,atomic_number) = 0.0123340D0
    FN2_am1(3,atomic_number) = 13.5573360D0
    FN3_am1(3,atomic_number) = 2.0091730D0
    FN1_am1(4,atomic_number) = 0.0000000D0
    FN2_am1(4,atomic_number) = 0.0000000D0
    FN3_am1(4,atomic_number) = 0.0000000D0
    NUM_FN_am1(atomic_number) = 3
    GSS_am1(atomic_number) = 11.7863290D0
    GSP_am1(atomic_number) = 8.6631270D0
    GPP_am1(atomic_number) = 10.0393080D0
    GP2_am1(atomic_number) = 7.7816880D0
    HSP_am1(atomic_number) = 2.5321370D0
    DD_am1(atomic_number) = 0.9004265D0
    QQ_am1(atomic_number) = 1.0036329D0
    AM_am1(atomic_number) = 0.4331617D0
    AD_am1(atomic_number) = 0.5907115D0
    AQ_am1(atomic_number) = 0.6454943D0
    alp_am1(atomic_number) = 2.4616480D0
    USS_am1(atomic_number) = -56.6940560D0
    UPP_am1(atomic_number) = -48.7170490D0

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM.10, 209 (1989).
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) =  -183.4537395D0
    s_orb_exp_pm3(atomic_number) = 1.8911850D0
    p_orb_exp_pm3(atomic_number) = 1.6589720D0
    betas_pm3(atomic_number) = -8.8274650D0
    betap_pm3(atomic_number) = -8.0914150D0
    FN1_pm3(1,atomic_number) = -0.3991910D0
    FN2_pm3(1,atomic_number) = 6.0006690D0
    FN3_pm3(1,atomic_number) = 0.9621230D0
    FN1_pm3(2,atomic_number) = -0.0548990D0
    FN2_pm3(2,atomic_number) = 6.0018450D0
    FN3_pm3(2,atomic_number) = 1.5799440D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 8.9646670D0
    GSP_pm3(atomic_number) = 6.7859360D0
    GPP_pm3(atomic_number) = 9.9681640D0
    GP2_pm3(atomic_number) = 7.9702470D0
    HSP_pm3(atomic_number) = 4.0418360D0
    DD_pm3(atomic_number) = 1.1214313D0
    QQ_pm3(atomic_number) = 1.0086488D0
    AM_pm3(atomic_number) = 0.3294622D0
    AD_pm3(atomic_number) = 0.6679118D0
    AQ_pm3(atomic_number) = 0.6137472D0
    alp_pm3(atomic_number) = 2.2697060D0
    USS_pm3(atomic_number) = -49.8953710D0
    UPP_pm3(atomic_number) = -44.3925830D0

    !PDDG/PM3
    ! Reference Repasky, Chandrasekhar and Jorgensen, 2002, 
    !           J Comp Chem, 23: 1601-1622
    element_supported_pddgpm3(atomic_number) = .true.
    elec_eng_pddgpm3(atomic_number) = -166.3365540000D0
    s_orb_exp_pddgpm3(atomic_number) = 1.0120020000D0
    p_orb_exp_pddgpm3(atomic_number) = 1.8769990000D0
    betas_pddgpm3(atomic_number) = -2.9539120000D0
    betap_pddgpm3(atomic_number) = -8.5077790000D0
    FN1_pddgpm3(1,atomic_number) = -0.330692000D0
    FN2_pddgpm3(1,atomic_number) = 6.000000D0
    FN3_pddgpm3(1,atomic_number) = 0.823837000D0
    FN1_pddgpm3(2,atomic_number) = 0.024171000D0
    FN2_pddgpm3(2,atomic_number) = 6.000000D0
    FN3_pddgpm3(2,atomic_number) = 2.017756000D0
    FN1_pddgpm3(3,atomic_number) = 0.0d0
    FN2_pddgpm3(3,atomic_number) = 0.0d0
    FN3_pddgpm3(3,atomic_number) = 0.0d0
    FN1_pddgpm3(4,atomic_number) = 0.0d0
    FN2_pddgpm3(4,atomic_number) = 0.0d0
    FN3_pddgpm3(4,atomic_number) = 0.0d0
    NUM_FN_pddgpm3(atomic_number) = 2
    GSS_pddgpm3(atomic_number) = 8.9646670D0
    GSP_pddgpm3(atomic_number) = 6.7859360D0
    GPP_pddgpm3(atomic_number) = 9.9681640D0
    GP2_pddgpm3(atomic_number) = 7.9702470D0
    HSP_pddgpm3(atomic_number) = 4.0418360D0
    DD_pddgpm3(atomic_number) = 1.0069890D0
    QQ_pddgpm3(atomic_number) = 0.8914870D0
    AM_pddgpm3(atomic_number) = 0.329462221D0
    AD_pddgpm3(atomic_number) = 0.702570847D0
    AQ_pddgpm3(atomic_number) = 0.662834599D0
    alp_pddgpm3(atomic_number) = 2.5397510000D0
    USS_pddgpm3(atomic_number) = -43.9063660000D0
    UPP_pddgpm3(atomic_number) = -43.4613480000D0
    PDDGC1_pm3(atomic_number) = 0.120430D0
    PDDGC2_pm3(atomic_number) = 0.672870000D0
    PDDGE1_pm3(atomic_number) = -0.002663D0
    PDDGE2_pm3(atomic_number) = 2.032340000D0

    !-------------------
    !END SULPHUR
    !-------------------

    !-------------------
    !CHLORINE
    !-------------------
    atomic_number = 17
    !MNDO
    ! Reference:  M.J.S.DEWAR, H.S.RZEPA, J. COMP. CHEM., 4, 158, (1983)
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -353.1176670D0
    s_orb_exp_mndo(atomic_number) = 3.7846450D0
    p_orb_exp_mndo(atomic_number) = 2.0362630D0
    betas_mndo(atomic_number) = -14.2623200D0
    betap_mndo(atomic_number) = -14.2623200D0
    GSS_mndo(atomic_number) = 15.03D00
    GSP_mndo(atomic_number) = 13.16D00
    GPP_mndo(atomic_number) = 11.30D00
    GP2_mndo(atomic_number) = 9.97D00
    HSP_mndo(atomic_number) = 2.42D00
    DD_mndo(atomic_number) = 0.4986870D0
    QQ_mndo(atomic_number) = 0.8217603D0
    AM_mndo(atomic_number) = 0.5523705D0
    AD_mndo(atomic_number) = 0.8061220D0
    AQ_mndo(atomic_number) = 0.6053435D0
    alp_mndo(atomic_number) = 2.5422010D0
    USS_mndo(atomic_number) = -100.2271660D0
    UPP_mndo(atomic_number) = -77.3786670D0

    !AM1
    ! Reference: M.J.S. DEWAR AND E. G. ZOEBISCH, THEOCHEM, 180, 1 (1988).
    element_supported_am1(atomic_number) = .true.
    elec_eng_am1(atomic_number) = -372.1984310D0
    s_orb_exp_am1(atomic_number) = 3.6313760D0
    p_orb_exp_am1(atomic_number) = 2.0767990D0
    betas_am1(atomic_number) = -24.5946700D0
    betap_am1(atomic_number) = -14.6372160D0
    FN1_am1(1,atomic_number) = 0.0942430D0
    FN2_am1(1,atomic_number) = 4.0000000D0
    FN3_am1(1,atomic_number) = 1.3000000D0
    FN1_am1(2,atomic_number) = 0.0271680D0
    FN2_am1(2,atomic_number) = 4.0000000D0
    FN3_am1(2,atomic_number) = 2.1000000D0
    FN1_am1(3,atomic_number) = 0.0000000D0
    FN2_am1(3,atomic_number) = 0.0000000D0
    FN3_am1(3,atomic_number) = 0.0000000D0
    FN1_am1(4,atomic_number) = 0.0000000D0
    FN2_am1(4,atomic_number) = 0.0000000D0
    FN3_am1(4,atomic_number) = 0.0000000D0
    NUM_FN_am1(atomic_number) = 2
    GSS_am1(atomic_number) = 15.0300000D0
    GSP_am1(atomic_number) = 13.1600000D0
    GPP_am1(atomic_number) = 11.3000000D0
    GP2_am1(atomic_number) = 9.9700000D0
    HSP_am1(atomic_number) = 2.4200000D0
    DD_am1(atomic_number) = 0.5406286D0
    QQ_am1(atomic_number) = 0.8057208D0
    AM_am1(atomic_number) = 0.5523705D0
    AD_am1(atomic_number) = 0.7693200D0
    AQ_am1(atomic_number) = 0.6133369D0
    alp_am1(atomic_number) = 2.9193680D0
    USS_am1(atomic_number) = -111.6139480D0
    UPP_am1(atomic_number) = -76.6401070D0

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM.10, 209 (1989).
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -315.1949480D0
    s_orb_exp_pm3(atomic_number) = 2.2462100D0
    p_orb_exp_pm3(atomic_number) = 2.1510100D0
    betas_pm3(atomic_number) = -27.5285600D0
    betap_pm3(atomic_number) = -11.5939220D0
    FN1_pm3(1,atomic_number) = -0.1715910D0
    FN2_pm3(1,atomic_number) = 6.0008020D0
    FN3_pm3(1,atomic_number) = 1.0875020D0
    FN1_pm3(2,atomic_number) = -0.0134580D0
    FN2_pm3(2,atomic_number) = 1.9666180D0
    FN3_pm3(2,atomic_number) = 2.2928910D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 16.0136010D0
    GSP_pm3(atomic_number) = 8.0481150D0
    GPP_pm3(atomic_number) = 7.5222150D0
    GP2_pm3(atomic_number) = 7.5041540D0
    HSP_pm3(atomic_number) = 3.4811530D0
    DD_pm3(atomic_number) = 0.9175856D0
    QQ_pm3(atomic_number) = 0.7779230D0
    AM_pm3(atomic_number) = 0.5885190D0
    AD_pm3(atomic_number) = 0.6814522D0
    AQ_pm3(atomic_number) = 0.3643694D0
    alp_pm3(atomic_number) = 2.5172960D0
    USS_pm3(atomic_number) = -100.6267470D0
    UPP_pm3(atomic_number) = -53.6143960D0

    !PDDG/PM3
    ! Reference Tubert-Brohman, Werneck, Repasky and Jorgensen, 2003, 
    !           J Comp Chem, 25: 138-150
    element_supported_pddgpm3(atomic_number) = .true.
    elec_eng_pddgpm3(atomic_number) = -305.715201D0
    s_orb_exp_pddgpm3(atomic_number) = 2.548268D0
    p_orb_exp_pddgpm3(atomic_number) = 2.284624D0
    betas_pddgpm3(atomic_number) = -26.913129D0
    betap_pddgpm3(atomic_number) = -14.991178D0
    FN1_pddgpm3(1,atomic_number) = -0.112222D0
    FN2_pddgpm3(1,atomic_number) = 5.963719D0
    FN3_pddgpm3(1,atomic_number) = 1.027719D0
    FN1_pddgpm3(2,atomic_number) = -0.013061D0
    FN2_pddgpm3(2,atomic_number) = 1.999556D0
    FN3_pddgpm3(2,atomic_number) = 2.286377D0
    FN1_pddgpm3(3,atomic_number) = 0.0d0
    FN2_pddgpm3(3,atomic_number) = 0.0d0
    FN3_pddgpm3(3,atomic_number) = 0.0d0
    FN1_pddgpm3(4,atomic_number) = 0.0d0
    FN2_pddgpm3(4,atomic_number) = 0.0d0
    FN3_pddgpm3(4,atomic_number) = 0.0d0
    NUM_FN_pddgpm3(atomic_number) = 2
    GSS_pddgpm3(atomic_number) = 16.0136010D0
    GSP_pddgpm3(atomic_number) = 8.0481150D0
    GPP_pddgpm3(atomic_number) = 7.5222150D0
    GP2_pddgpm3(atomic_number) = 7.5041540D0
    HSP_pddgpm3(atomic_number) = 3.4811530D0
    DD_pddgpm3(atomic_number) = 0.827561D0 
    QQ_pddgpm3(atomic_number) = 0.732427D0
    AM_pddgpm3(atomic_number) = 0.5885191681D0
    AD_pddgpm3(atomic_number) = 0.7182215685D0
    AQ_pddgpm3(atomic_number) = 0.2174760254D0
    alp_pddgpm3(atomic_number) = 2.497617D0
    USS_pddgpm3(atomic_number) = -95.094434D0
    UPP_pddgpm3(atomic_number) = -53.921651D0
    PDDGC1_pm3(atomic_number) = -0.016552D0
    PDDGC2_pm3(atomic_number) = -0.016646D0
    PDDGE1_pm3(atomic_number) = 1.727690D0
    PDDGE2_pm3(atomic_number) = 1.784655D0

    !PDDG/MNDO
    ! Reference: Repasky, Chandrasekhar and Jorgensen, 2002, 
    !            J Comp Chem, 23: 1601-1622
    element_supported_pddgmndo(atomic_number) = .true.
    elec_eng_pddgmndo(atomic_number) = -378.909727D0
    s_orb_exp_pddgmndo(atomic_number) = 4.212404D0
    p_orb_exp_pddgmndo(atomic_number) = 2.037647D0
    betas_pddgmndo(atomic_number) = -15.663317D0
    betap_pddgmndo(atomic_number) = -15.399331D0
    GSS_pddgmndo(atomic_number) = 15.03D0
    GSP_pddgmndo(atomic_number) = 13.16D0
    GPP_pddgmndo(atomic_number) = 11.30D0
    GP2_pddgmndo(atomic_number) = 9.97D0
    HSP_pddgmndo(atomic_number) = 2.42D0
    DD_pddgmndo(atomic_number) = 0.411609D0
    QQ_pddgmndo(atomic_number) = 0.821202D0
    AM_pddgmndo(atomic_number) = 0.5523702206D0
    AD_pddgmndo(atomic_number) = 0.9021232373D0
    AQ_pddgmndo(atomic_number) = 0.6056172208D0
    alp_pddgmndo(atomic_number) = 2.602846D0
    USS_pddgmndo(atomic_number) = -111.133653D0
    UPP_pddgmndo(atomic_number) = -78.062493D0
    PDDGC1_mndo(atomic_number) = -0.017119D0
    PDDGC2_mndo(atomic_number) = 0.005497D0
    PDDGE1_mndo(atomic_number) = 1.466335D0
    PDDGE2_mndo(atomic_number) = 2.236842D0

    !-------------------
    !END CHLORINE
    !-------------------

    !-------------------
    !ZINC
    !-------------------
    atomic_number = 30
    !MNDO
    ! Reference: M.J.S. DEWAR, K.M. MERZ, ORGANOMETALLICS, 5, 1494-1496
    !            (1986)
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -29.8794320D0
    s_orb_exp_mndo(atomic_number) = 2.0473590D0
    p_orb_exp_mndo(atomic_number) = 1.4609460D0
    betas_mndo(atomic_number) = -1.0000000D0
    betap_mndo(atomic_number) = -2.0000000D0
    GSS_mndo(atomic_number) = 11.8000000D0
    GSP_mndo(atomic_number) = 11.1820180D0
    GPP_mndo(atomic_number) = 13.3000000D0
    GP2_mndo(atomic_number) = 12.9305200D0
    HSP_mndo(atomic_number) = 0.4846060D0
    DD_mndo(atomic_number) = 1.3037826D0
    QQ_mndo(atomic_number) = 1.4520183D0
    AM_mndo(atomic_number) = 0.4336641D0
    AD_mndo(atomic_number) = 0.2375912D0
    AQ_mndo(atomic_number) = 0.2738858D0
    alp_mndo(atomic_number) = 1.5064570D0
    USS_mndo(atomic_number) = -20.8397160D0
    UPP_mndo(atomic_number) = -19.6252240D0

    !AM1
    ! Reference: M.J.S. DEWAR, K.M. MERZ, ORGANOMETALLICS, 7, 522-524
    !            (1988)
    element_supported_am1(atomic_number) = .true.
    elec_eng_am1(atomic_number) = -30.2800160D0
    s_orb_exp_am1(atomic_number) = 1.9542990D0
    p_orb_exp_am1(atomic_number) = 1.3723650D0
    betas_am1(atomic_number) = -1.9974290D0
    betap_am1(atomic_number) = -4.7581190D0
    FN1_am1(1,atomic_number) = 0.0000000D0
    FN2_am1(1,atomic_number) = 0.0000000D0
    FN3_am1(1,atomic_number) = 0.0000000D0
    FN1_am1(2,atomic_number) = 0.0000000D0
    FN2_am1(2,atomic_number) = 0.0000000D0
    FN3_am1(2,atomic_number) = 0.0000000D0
    FN1_am1(3,atomic_number) = 0.0000000D0
    FN2_am1(3,atomic_number) = 0.0000000D0
    FN3_am1(3,atomic_number) = 0.0000000D0
    FN1_am1(4,atomic_number) = 0.0000000D0
    FN2_am1(4,atomic_number) = 0.0000000D0
    FN3_am1(4,atomic_number) = 0.0000000D0
    NUM_FN_am1(atomic_number) = 0
    GSS_am1(atomic_number) = 11.8000000D0
    GSP_am1(atomic_number) = 11.1820180D0
    GPP_am1(atomic_number) = 13.3000000D0
    GP2_am1(atomic_number) = 12.9305200D0
    HSP_am1(atomic_number) = 0.4846060D0
    DD_am1(atomic_number) = 1.3581113D0
    QQ_am1(atomic_number) = 1.5457406D0
    AM_am1(atomic_number) = 0.4336641D0
    AD_am1(atomic_number) = 0.2317423D0
    AQ_am1(atomic_number) = 0.2621165D0
    alp_am1(atomic_number) = 1.4845630D0
    USS_am1(atomic_number) = -21.0400080D0
    UPP_am1(atomic_number) = -17.6555740D0

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM. (ACCEPTED)
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -27.3872000D0
    s_orb_exp_pm3(atomic_number) = 1.8199890D0
    p_orb_exp_pm3(atomic_number) = 1.5069220D0
    betas_pm3(atomic_number) = -0.7155780D0
    betap_pm3(atomic_number) = -6.3518640D0
    FN1_pm3(1,atomic_number) = -0.1112340D0
    FN2_pm3(1,atomic_number) = 6.0014780D0
    FN3_pm3(1,atomic_number) = 1.5160320D0
    FN1_pm3(2,atomic_number) = -0.1323700D0
    FN2_pm3(2,atomic_number) = 1.9958390D0
    FN3_pm3(2,atomic_number) = 2.5196420D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 9.6771960D0
    GSP_pm3(atomic_number) = 7.7362040D0
    GPP_pm3(atomic_number) = 4.9801740D0
    GP2_pm3(atomic_number) = 4.6696560D0
    HSP_pm3(atomic_number) = 0.6004130D0
    DD_pm3(atomic_number) = 1.5005758D0
    QQ_pm3(atomic_number) = 1.4077174D0
    AM_pm3(atomic_number) = 0.3556485D0
    AD_pm3(atomic_number) = 0.2375689D0
    AQ_pm3(atomic_number) = 0.2661069D0
    alp_pm3(atomic_number) = 1.3501260D0
    USS_pm3(atomic_number) = -18.5321980D0
    UPP_pm3(atomic_number) = -11.0474090D0

    !-------------------
    !END ZINC
    !-------------------

    !-------------------
    !GALLIUM
    !-------------------
    atomic_number = 31
    !MNDO
    ! Reference: None
    element_supported_mndo(atomic_number) = .false.

    !AM1
    ! Reference: None
    element_supported_am1(atomic_number) = .false.

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM.(ACCEPTED)
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -57.3280250D0
    s_orb_exp_pm3(atomic_number) = 1.8470400D0
    p_orb_exp_pm3(atomic_number) = 0.8394110D0
    betas_pm3(atomic_number) = -4.9456180D0
    betap_pm3(atomic_number) = -0.4070530D0
    FN1_pm3(1,atomic_number) = -0.5601790D0
    FN2_pm3(1,atomic_number) = 5.6232730D0
    FN3_pm3(1,atomic_number) = 1.5317800D0
    FN1_pm3(2,atomic_number) = -0.2727310D0
    FN2_pm3(2,atomic_number) = 1.9918430D0
    FN3_pm3(2,atomic_number) = 2.1838640D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 8.4585540D0
    GSP_pm3(atomic_number) = 8.9256190D0
    GPP_pm3(atomic_number) = 5.0868550D0
    GP2_pm3(atomic_number) = 4.9830450D0
    HSP_pm3(atomic_number) = 2.0512600D0
    DD_pm3(atomic_number) = 0.9776692D0
    QQ_pm3(atomic_number) = 2.5271534D0
    AM_pm3(atomic_number) = 0.3108620D0
    AD_pm3(atomic_number) = 0.5129360D0
    AQ_pm3(atomic_number) = 0.1546208D0
    alp_pm3(atomic_number) = 1.6051150D0
    USS_pm3(atomic_number) = -29.8555930D0
    UPP_pm3(atomic_number) = -21.8753710D0

    !-------------------
    !END GALLIUM
    !-------------------

    !-------------------
    !GERMANIUM
    !-------------------
    atomic_number = 32
    !MNDO
    ! Reference: M.J.S.DEWAR, G.L.GRADY, E.F.HEALY,ORGANOMETALLICS 
    !            6 186-189, (1987)
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -76.2489440D0
    s_orb_exp_mndo(atomic_number) = 1.2931800D0
    p_orb_exp_mndo(atomic_number) = 2.0205640D0
    betas_mndo(atomic_number) = -4.5164790D0
    betap_mndo(atomic_number) = -1.7555170D0
    GSS_mndo(atomic_number) = 9.8000000D0
    GSP_mndo(atomic_number) = 8.3000000D0
    GPP_mndo(atomic_number) = 7.3000000D0
    GP2_mndo(atomic_number) = 6.5000000D0
    HSP_mndo(atomic_number) = 1.3000000D0
    DD_mndo(atomic_number) = 1.2556091D0
    QQ_mndo(atomic_number) = 1.0498655D0
    AM_mndo(atomic_number) = 0.3601617D0
    AD_mndo(atomic_number) = 0.3643722D0
    AQ_mndo(atomic_number) = 0.4347337D0
    alp_mndo(atomic_number) = 1.9784980D0
    USS_mndo(atomic_number) = -33.9493670D0
    UPP_mndo(atomic_number) = -27.4251050D0

    !AM1
    ! Reference: M.J.S.Dewar and C.Jie, Organometallics, 8, 1544, (1989)
    element_supported_am1(atomic_number) = .true.
    elec_eng_am1(atomic_number) = -78.7084810D0
    s_orb_exp_am1(atomic_number) = 1.2196310D0
    p_orb_exp_am1(atomic_number) = 1.9827940D0
    betas_am1(atomic_number) = -4.3566070D0
    betap_am1(atomic_number) = -0.9910910D0
    FN1_am1(1,atomic_number) = 0.0000000D0
    FN2_am1(1,atomic_number) = 0.0000000D0
    FN3_am1(1,atomic_number) = 0.0000000D0
    FN1_am1(2,atomic_number) = 0.0000000D0
    FN2_am1(2,atomic_number) = 0.0000000D0
    FN3_am1(2,atomic_number) = 0.0000000D0
    FN1_am1(3,atomic_number) = 0.0000000D0
    FN2_am1(3,atomic_number) = 0.0000000D0
    FN3_am1(3,atomic_number) = 0.0000000D0
    FN1_am1(4,atomic_number) = 0.0000000D0
    FN2_am1(4,atomic_number) = 0.0000000D0
    FN3_am1(4,atomic_number) = 0.0000000D0
    NUM_FN_am1(atomic_number) = 0
    GSS_am1(atomic_number) = 10.1686050D0
    GSP_am1(atomic_number) = 8.1444730D0
    GPP_am1(atomic_number) = 6.6719020D0
    GP2_am1(atomic_number) = 6.2697060D0
    HSP_am1(atomic_number) = 0.9370930D0
    DD_am1(atomic_number) = 1.2472095D0
    QQ_am1(atomic_number) = 1.0698642D0
    AM_am1(atomic_number) = 0.3737084D0
    AD_am1(atomic_number) = 0.3180309D0
    AQ_am1(atomic_number) = 0.3485612D0
    alp_am1(atomic_number) = 2.1364050D0
    USS_am1(atomic_number) = -34.1838890D0
    UPP_am1(atomic_number) = -28.6408110D0

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM.(ACCEPTED)
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -84.0156006D0
    s_orb_exp_pm3(atomic_number) = 2.2373526D0
    p_orb_exp_pm3(atomic_number) = 1.5924319D0
    betas_pm3(atomic_number) = -5.3250024D0
    betap_pm3(atomic_number) = -2.2501567D0
    FN1_pm3(1,atomic_number) = 0.9631726D0
    FN2_pm3(1,atomic_number) = 6.0120134D0
    FN3_pm3(1,atomic_number) = 2.1633655D0
    FN1_pm3(2,atomic_number) =-0.9593891D0
    FN2_pm3(2,atomic_number) = 5.7491802D0
    FN3_pm3(2,atomic_number) = 2.1693724D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 5.3769635D0
    GSP_pm3(atomic_number) = 10.2095293D0
    GPP_pm3(atomic_number) = 7.6718647D0
    GP2_pm3(atomic_number) = 6.9242663D0
    HSP_pm3(atomic_number) = 1.3370204D0
    DD_pm3(atomic_number) = 1.1920304D0
    QQ_pm3(atomic_number) = 1.3321263D0
    AM_pm3(atomic_number) = 0.1976098D0
    AD_pm3(atomic_number) = 0.3798182D0
    AQ_pm3(atomic_number) = 0.3620669D0
    alp_pm3(atomic_number) = 1.9723370D0
    USS_pm3(atomic_number) = -35.4671955D0
    UPP_pm3(atomic_number) = -31.5863583D0

    !-------------------
    !END GERMANIUM
    !-------------------

    !-------------------
    !ARSENIC
    !-------------------
    atomic_number = 33
    !MNDO
    ! Reference: None
    element_supported_mndo(atomic_number) = .false.

    !AM1
    ! Reference: None
    element_supported_am1(atomic_number) = .false.

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM.(ACCEPTED)
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -122.6326140D0
    s_orb_exp_pm3(atomic_number) = 2.6361770D0
    p_orb_exp_pm3(atomic_number) = 1.7038890D0
    betas_pm3(atomic_number) = -8.2321650D0
    betap_pm3(atomic_number) = -5.0173860D0
    FN1_pm3(1,atomic_number) =-0.4600950D0 
    FN2_pm3(1,atomic_number) = 1.9831150D0
    FN3_pm3(1,atomic_number) = 1.0867930D0
    FN1_pm3(2,atomic_number) =-0.0889960D0
    FN2_pm3(2,atomic_number) = 1.9929440D0
    FN3_pm3(2,atomic_number) = 2.1400580D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 8.7890010D0
    GSP_pm3(atomic_number) = 5.3979830D0
    GPP_pm3(atomic_number) = 8.2872500D0
    GP2_pm3(atomic_number) = 8.2103460D0
    HSP_pm3(atomic_number) = 1.9510340D0
    DD_pm3(atomic_number) = 0.9679655D0
    QQ_pm3(atomic_number) = 1.2449874D0
    AM_pm3(atomic_number) = 0.3230063D0
    AD_pm3(atomic_number) = 0.5042239D0
    AQ_pm3(atomic_number) = 0.2574219D0
    alp_pm3(atomic_number) = 1.7944770D0
    USS_pm3(atomic_number) = -38.5074240D0
    UPP_pm3(atomic_number) = -35.1524150D0

    !-------------------
    !END ARSENIC
    !-------------------

    !-------------------
    !SELENIUM
    !-------------------
    atomic_number = 34
    !MNDO
    ! Reference: None
    element_supported_mndo(atomic_number) = .false.

    !AM1
    ! Reference: None
    element_supported_am1(atomic_number) = .false.

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM.(ACCEPTED)
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -192.7748115D0
    s_orb_exp_pm3(atomic_number) = 2.8280510D0
    p_orb_exp_pm3(atomic_number) = 1.7325360D0
    betas_pm3(atomic_number) = -6.1578220D0
    betap_pm3(atomic_number) = -5.4930390D0
    FN1_pm3(1,atomic_number) = 0.0478730D0
    FN2_pm3(1,atomic_number) = 6.0074000D0
    FN3_pm3(1,atomic_number) = 2.0817170D0
    FN1_pm3(2,atomic_number) = 0.1147200D0
    FN2_pm3(2,atomic_number) = 6.0086720D0
    FN3_pm3(2,atomic_number) = 1.5164230D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 7.4325910D0
    GSP_pm3(atomic_number) = 10.0604610D0
    GPP_pm3(atomic_number) = 9.5683260D0
    GP2_pm3(atomic_number) = 7.7242890D0
    HSP_pm3(atomic_number) = 4.0165580D0
    DD_pm3(atomic_number) = 0.8719813D0
    QQ_pm3(atomic_number) = 1.2244019D0
    AM_pm3(atomic_number) = 0.2731566D0
    AD_pm3(atomic_number) = 0.7509697D0
    AQ_pm3(atomic_number) = 0.5283737D0
    alp_pm3(atomic_number) = 3.0439570D0
    USS_pm3(atomic_number) = -55.3781350D0
    UPP_pm3(atomic_number) = -49.8230760D0

    !-------------------
    !END SELENIUM
    !-------------------

    !-------------------
    !BROMINE
    !-------------------
    atomic_number = 35
    !MNDO
    ! Reference: M.J.S.DEWAR, E.F. HEALY, J. COMP. CHEM., 4, 542, (1983)
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -346.6812500D0
    s_orb_exp_mndo(atomic_number) = 3.8543019D0
    p_orb_exp_mndo(atomic_number) = 2.1992091D0
    betas_mndo(atomic_number) = -8.9171070D0
    betap_mndo(atomic_number) = -9.9437400D0
    GSS_mndo(atomic_number) = 15.03643948D0
    GSP_mndo(atomic_number) = 13.03468242D0
    GPP_mndo(atomic_number) = 11.27632539D0
    GP2_mndo(atomic_number) = 9.85442552D0
    HSP_mndo(atomic_number) = 2.45586832D0
    DD_mndo(atomic_number) = 0.6051074D0
    QQ_mndo(atomic_number) = 0.9645873D0
    AM_mndo(atomic_number) = 0.5526068D0
    AD_mndo(atomic_number) = 0.7258330D0
    AQ_mndo(atomic_number) = 0.5574589D0
    alp_mndo(atomic_number) = 2.4457051D0
    USS_mndo(atomic_number) = -99.9864405D0
    UPP_mndo(atomic_number) = -75.6713075D0

    !AM1
    ! Reference:  Br: (AM1): M.J.S. DEWAR AND E. G. ZOEBISCH, THEOCHEM, 
    !             180, 1 (1988).
    element_supported_am1(atomic_number) = .true.
    elec_eng_am1(atomic_number) = -352.3142087D0
    s_orb_exp_am1(atomic_number) = 3.0641330D0
    p_orb_exp_am1(atomic_number) = 2.0383330D0
    betas_am1(atomic_number) = -19.3998800D0
    betap_am1(atomic_number) = -8.9571950D0
    FN1_am1(1,atomic_number) = 0.0666850D0
    FN2_am1(1,atomic_number) = 4.0000000D0
    FN3_am1(1,atomic_number) = 1.5000000D0
    FN1_am1(2,atomic_number) = 0.0255680D0
    FN2_am1(2,atomic_number) = 4.0000000D0
    FN3_am1(2,atomic_number) = 2.3000000D0
    FN1_am1(3,atomic_number) = 0.0000000D0
    FN2_am1(3,atomic_number) = 0.0000000D0
    FN3_am1(3,atomic_number) = 0.0000000D0
    FN1_am1(4,atomic_number) = 0.0000000D0
    FN2_am1(4,atomic_number) = 0.0000000D0
    FN3_am1(4,atomic_number) = 0.0000000D0
    NUM_FN_am1(atomic_number) = 2
    GSS_am1(atomic_number) = 15.0364395D0
    GSP_am1(atomic_number) = 13.0346824D0
    GPP_am1(atomic_number) = 11.2763254D0
    GP2_am1(atomic_number) = 9.8544255D0
    HSP_am1(atomic_number) = 2.4558683D0
    DD_am1(atomic_number) = 0.8458104D0
    QQ_am1(atomic_number) = 1.0407133D0
    AM_am1(atomic_number) = 0.5526071D0
    AD_am1(atomic_number) = 0.6024598D0
    AQ_am1(atomic_number) = 0.5307555D0
    alp_am1(atomic_number) = 2.5765460D0
    USS_am1(atomic_number) = -104.6560630D0
    UPP_am1(atomic_number) = -74.9300520D0

    !PM3
    ! Reference:  J. J. P. STEWART, J. COMP. CHEM.10, 209 (1989).
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -352.5398970D0
    s_orb_exp_pm3(atomic_number) = 5.3484570D0
    p_orb_exp_pm3(atomic_number) = 2.1275900D0
    betas_pm3(atomic_number) = -31.1713420D0
    betap_pm3(atomic_number) = -6.8140130D0
    FN1_pm3(1,atomic_number) = 0.9604580D0 
    FN2_pm3(1,atomic_number) = 5.9765080D0
    FN3_pm3(1,atomic_number) = 2.3216540D0
    FN1_pm3(2,atomic_number) =-0.9549160D0
    FN2_pm3(2,atomic_number) = 5.9447030D0
    FN3_pm3(2,atomic_number) = 2.3281420D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 15.9434250D0
    GSP_pm3(atomic_number) = 16.0616800D0
    GPP_pm3(atomic_number) = 8.2827630D0
    GP2_pm3(atomic_number) = 7.8168490D0
    HSP_pm3(atomic_number) = 0.5788690D0
    DD_pm3(atomic_number) = 0.2759025D0
    QQ_pm3(atomic_number) = 0.9970532D0
    AM_pm3(atomic_number) = 0.5859399D0
    AD_pm3(atomic_number) = 0.6755383D0
    AQ_pm3(atomic_number) = 0.3823719D0
    alp_pm3(atomic_number) = 2.5118420D0
    USS_pm3(atomic_number) = -116.6193110D0
    UPP_pm3(atomic_number) = -74.2271290D0

    !PDDG/PM3
    ! Reference Tubert-Brohman, Werneck, Repasky and Jorgensen, 2003, 
    !           J Comp Chem, 25: 138-150
    element_supported_pddgpm3(atomic_number) = .true.
    elec_eng_pddgpm3(atomic_number) = -351.013887D0
    s_orb_exp_pddgpm3(atomic_number) = 4.345079D0
    p_orb_exp_pddgpm3(atomic_number) = 2.190961D0
    betas_pddgpm3(atomic_number) = -21.538044D0
    betap_pddgpm3(atomic_number) = -8.524764D0
    FN1_pddgpm3(1,atomic_number) = 0.961362D0
    FN2_pddgpm3(1,atomic_number) = 6.013600D0
    FN3_pddgpm3(1,atomic_number) = 2.340445D0
    FN1_pddgpm3(2,atomic_number) = -0.948834D0
    FN2_pddgpm3(2,atomic_number) = 5.976329D0
    FN3_pddgpm3(2,atomic_number) = 2.348745D0
    FN1_pddgpm3(3,atomic_number) = 0.0d0
    FN2_pddgpm3(3,atomic_number) = 0.0d0
    FN3_pddgpm3(3,atomic_number) = 0.0d0
    FN1_pddgpm3(4,atomic_number) = 0.0d0
    FN2_pddgpm3(4,atomic_number) = 0.0d0
    FN3_pddgpm3(4,atomic_number) = 0.0d0
    NUM_FN_pddgpm3(atomic_number) = 2
    GSS_pddgpm3(atomic_number) = 15.9434250D0
    GSP_pddgpm3(atomic_number) = 16.0616800D0
    GPP_pddgpm3(atomic_number) = 8.2827630D0
    GP2_pddgpm3(atomic_number) = 7.8168490D0
    HSP_pddgpm3(atomic_number) = 0.5788690D0
    DD_pddgpm3(atomic_number) = 0.473860D0
    QQ_pddgpm3(atomic_number) = 0.968214D0
    AM_pddgpm3(atomic_number) = 0.5859377289D0
    AD_pddgpm3(atomic_number) = 0.4778150474D0
    AQ_pddgpm3(atomic_number) = 0.3904288705D0
    alp_pddgpm3(atomic_number) = 2.424673D0
    USS_pddgpm3(atomic_number) = -115.841963D0
    UPP_pddgpm3(atomic_number) = -74.205146D0
    PDDGC1_pm3(atomic_number) = -0.013772D0
    PDDGC2_pm3(atomic_number) = 0.008849D0
    PDDGE1_pm3(atomic_number) = 1.852030D0
    PDDGE2_pm3(atomic_number) = 2.338958D0

    !PDDG/MNDO
    ! Reference: Repasky, Chandrasekhar and Jorgensen, 2002, 
    !            J Comp Chem, 23: 1601-1622
    element_supported_pddgmndo(atomic_number) = .true.
    elec_eng_pddgmndo(atomic_number) = -349.564096D0
    s_orb_exp_pddgmndo(atomic_number) = 3.999975D0
    p_orb_exp_pddgmndo(atomic_number) = 2.245040D0
    betas_pddgmndo(atomic_number) = -7.054170D0
    betap_pddgmndo(atomic_number) = -10.221030D0
    GSS_pddgmndo(atomic_number) = 15.03643948D0
    GSP_pddgmndo(atomic_number) = 13.03468242D0
    GPP_pddgmndo(atomic_number) = 11.27632539D0
    GP2_pddgmndo(atomic_number) = 9.85442552D0
    HSP_pddgmndo(atomic_number) = 2.45586832D0
    DD_pddgmndo(atomic_number) = 0.574623D0
    QQ_pddgmndo(atomic_number) = 0.944892D0
    AM_pddgmndo(atomic_number) = 0.5526070897D0
    AD_pddgmndo(atomic_number) = 0.7475327681D0
    AQ_pddgmndo(atomic_number) = 0.5649857963D0
    alp_pddgmndo(atomic_number) = 2.414265D0
    USS_pddgmndo(atomic_number) = -100.637007D0
    UPP_pddgmndo(atomic_number) = -76.015735D0
    PDDGC1_mndo(atomic_number) = -0.017133D0
    PDDGC2_mndo(atomic_number) = -0.016964D0
    PDDGE1_mndo(atomic_number) = 2.201539D0
    PDDGE2_mndo(atomic_number) = 2.255764D0

    !-------------------
    !END BROMINE
    !-------------------

    !-------------------
    !CADMIUM
    !-------------------
    atomic_number = 48
    !MNDO
    ! Reference: None
    element_supported_mndo(atomic_number) = .false.

    !AM1
    ! Reference: None
    element_supported_am1(atomic_number) = .false.

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM.(ACCEPTED)
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -22.4502080D0
    s_orb_exp_pm3(atomic_number) = 1.6793510D0
    p_orb_exp_pm3(atomic_number) = 2.0664120D0
    betas_pm3(atomic_number) = -8.5819440D0
    betap_pm3(atomic_number) = -0.6010340D0
    FN1_pm3(1,atomic_number) = 0.0d0
    FN2_pm3(1,atomic_number) = 0.0d0
    FN3_pm3(1,atomic_number) = 0.0d0
    FN1_pm3(2,atomic_number) = 0.0d0
    FN2_pm3(2,atomic_number) = 0.0d0
    FN3_pm3(2,atomic_number) = 0.0d0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 0
    GSS_pm3(atomic_number) = 9.2069600D0
    GSP_pm3(atomic_number) = 8.2315390D0
    GPP_pm3(atomic_number) = 4.9481040D0
    GP2_pm3(atomic_number) = 4.6696560D0
    HSP_pm3(atomic_number) = 1.6562340D0
    DD_pm3(atomic_number) = 1.5982681D0
    QQ_pm3(atomic_number) = 1.2432402D0
    AM_pm3(atomic_number) = 0.3383668D0
    AD_pm3(atomic_number) = 0.3570290D0
    AQ_pm3(atomic_number) = 0.2820582D0
    alp_pm3(atomic_number) = 1.5253820D0
    USS_pm3(atomic_number) = -15.8285840D0
    UPP_pm3(atomic_number) = 8.7497950D0

    !-------------------
    !END CADMIUM
    !-------------------

    !-------------------
    !INDIUM
    !-------------------
    atomic_number = 49
    !MNDO
    ! Reference: None
    element_supported_mndo(atomic_number) = .false.

    !AM1
    ! Reference: None
    element_supported_am1(atomic_number) = .false.

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM.(ACCEPTED)
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -51.9750470D0
    s_orb_exp_pm3(atomic_number) = 2.0161160D0
    p_orb_exp_pm3(atomic_number) = 1.4453500D0
    betas_pm3(atomic_number) = -2.9933190D0
    betap_pm3(atomic_number) = -1.8289080D0
    FN1_pm3(1,atomic_number) = -0.3431380D0
    FN2_pm3(1,atomic_number) = 1.9940340D0
    FN3_pm3(1,atomic_number) = 1.6255160D0
    FN1_pm3(2,atomic_number) = -0.1095320D0
    FN2_pm3(2,atomic_number) = 5.6832170D0
    FN3_pm3(2,atomic_number) = 2.8670090D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 6.5549000D0
    GSP_pm3(atomic_number) = 8.2298730D0
    GPP_pm3(atomic_number) = 6.2992690D0
    GP2_pm3(atomic_number) = 4.9842110D0
    HSP_pm3(atomic_number) = 2.6314610D0
    DD_pm3(atomic_number) = 1.5766241D0
    QQ_pm3(atomic_number) = 1.7774563D0
    AM_pm3(atomic_number) = 0.2409004D0
    AD_pm3(atomic_number) = 0.4532655D0
    AQ_pm3(atomic_number) = 0.3689812D0
    alp_pm3(atomic_number) = 1.4183850D0
    USS_pm3(atomic_number) = -26.1762050D0
    UPP_pm3(atomic_number) = -20.0058220D0

    !-------------------
    !END INDIUM
    !-------------------

    !-------------------
    !TIN
    !-------------------
    atomic_number = 50
    !MNDO
    ! Reference: M.J.S.DEWAR,G.L.GRADY,J.J.P.STEWART, J.AM.CHEM.SOC.,
    !            106 6771 (1984)
    element_supported_mndo(atomic_number) = .false.
    elec_eng_mndo(atomic_number) = -92.3241020D0
    s_orb_exp_mndo(atomic_number) = 2.0803800D0
    p_orb_exp_mndo(atomic_number) = 1.9371060D0
    betas_mndo(atomic_number) = -3.2351470D0
    betap_mndo(atomic_number) = -4.2904160D0
    GSS_mndo(atomic_number) = 9.8000000D0
    GSP_mndo(atomic_number) = 8.3000000D0
    GPP_mndo(atomic_number) = 7.3000000D0
    GP2_mndo(atomic_number) = 6.5000000D0
    HSP_mndo(atomic_number) = 1.3000000D0
    DD_mndo(atomic_number) = 1.5697766D0
    QQ_mndo(atomic_number) = 1.3262292D0
    AM_mndo(atomic_number) = 0.3601617D0
    AD_mndo(atomic_number) = 0.3219998D0
    AQ_mndo(atomic_number) = 0.3713827D0
    alp_mndo(atomic_number) = 1.8008140D0
    USS_mndo(atomic_number) = -40.8518020D0
    UPP_mndo(atomic_number) = -28.5602490D0

    !AM1
    ! Reference: None
    element_supported_am1(atomic_number) = .false.

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM.(ACCEPTED)
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -78.8877790D0
    s_orb_exp_pm3(atomic_number) = 2.3733280D0
    p_orb_exp_pm3(atomic_number) = 1.6382330D0
    betas_pm3(atomic_number) = -2.7858020D0
    betap_pm3(atomic_number) = -2.0059990D0
    FN1_pm3(1,atomic_number) =-0.1503530D0
    FN2_pm3(1,atomic_number) = 6.0056940D0
    FN3_pm3(1,atomic_number) = 1.7046420D0
    FN1_pm3(2,atomic_number) =-0.0444170D0
    FN2_pm3(2,atomic_number) = 2.2573810D0
    FN3_pm3(2,atomic_number) = 2.4698690D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 10.1900330D0
    GSP_pm3(atomic_number) = 7.2353270D0
    GPP_pm3(atomic_number) = 5.6738100D0
    GP2_pm3(atomic_number) = 5.1822140D0
    HSP_pm3(atomic_number) = 1.0331570D0
    DD_pm3(atomic_number) = 1.3120038D0
    QQ_pm3(atomic_number) = 1.5681814D0
    AM_pm3(atomic_number) = 0.3744959D0
    AD_pm3(atomic_number) = 0.3218163D0
    AQ_pm3(atomic_number) = 0.2832529D0
    alp_pm3(atomic_number) = 1.6996500D0
    USS_pm3(atomic_number) = -34.5501920D0
    UPP_pm3(atomic_number) = -25.8944190D0

    !-------------------
    !END TIN
    !-------------------

    !-------------------
    !ANTIMONY
    !-------------------
    atomic_number = 51
    !MNDO
    ! Reference: None
    element_supported_mndo(atomic_number) = .false.

    !AM1
    ! Reference: None
    element_supported_am1(atomic_number) = .false.

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM.(ACCEPTED)
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -148.9382890D0
    s_orb_exp_pm3(atomic_number) = 2.3430390D0
    p_orb_exp_pm3(atomic_number) = 1.8999920D0
    betas_pm3(atomic_number) = -14.7942170D0
    betap_pm3(atomic_number) = -2.8179480D0
    FN1_pm3(1,atomic_number) = 3.0020280D0
    FN2_pm3(1,atomic_number) = 6.0053420D0
    FN3_pm3(1,atomic_number) = 0.8530600D0
    FN1_pm3(2,atomic_number) =-0.0188920D0
    FN2_pm3(2,atomic_number) = 6.0114780D0
    FN3_pm3(2,atomic_number) = 2.7933110D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 9.2382770D0
    GSP_pm3(atomic_number) = 5.2776800D0
    GPP_pm3(atomic_number) = 6.3500000D0
    GP2_pm3(atomic_number) = 6.2500000D0
    HSP_pm3(atomic_number) = 2.4244640D0
    DD_pm3(atomic_number) = 1.4091903D0
    QQ_pm3(atomic_number) = 1.3521354D0
    AM_pm3(atomic_number) = 0.3395177D0
    AD_pm3(atomic_number) = 0.4589010D0
    AQ_pm3(atomic_number) = 0.2423472D0
    alp_pm3(atomic_number) = 2.0343010D0
    USS_pm3(atomic_number) = -56.4321960D0
    UPP_pm3(atomic_number) = -29.4349540D0

    !-------------------
    !END ANTIMONY
    !-------------------

    !-------------------
    !TELLURIUM
    !-------------------
    atomic_number = 52
    !MNDO
    ! Reference: None
    element_supported_mndo(atomic_number) = .false.

    !AM1
    ! Reference: None
    element_supported_am1(atomic_number) = .false.

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM.(ACCEPTED)
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -168.0945925D0
    s_orb_exp_pm3(atomic_number) = 4.1654920D0
    p_orb_exp_pm3(atomic_number) = 1.6475550D0
    betas_pm3(atomic_number) = -2.6651460D0
    betap_pm3(atomic_number) = -3.8954300D0
    FN1_pm3(1,atomic_number) = 0.0333910D0
    FN2_pm3(1,atomic_number) = 5.9563790D0
    FN3_pm3(1,atomic_number) = 2.2775750D0
    FN1_pm3(2,atomic_number) =-1.9218670D0
    FN2_pm3(2,atomic_number) = 4.9732190D0
    FN3_pm3(2,atomic_number) = 0.5242430D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 10.2550730D0
    GSP_pm3(atomic_number) = 8.1691450D0
    GPP_pm3(atomic_number) = 7.7775920D0
    GP2_pm3(atomic_number) = 7.7551210D0
    HSP_pm3(atomic_number) = 3.7724620D0
    DD_pm3(atomic_number) = 0.3484177D0
    QQ_pm3(atomic_number) = 1.5593085D0
    AM_pm3(atomic_number) = 0.3768862D0
    AD_pm3(atomic_number) = 1.1960743D0
    AQ_pm3(atomic_number) = 0.2184786D0
    alp_pm3(atomic_number) = 2.4850190D0
    USS_pm3(atomic_number) = -44.9380360D0
    UPP_pm3(atomic_number) = -46.3140990D0

    !-------------------
    !END TELLURIUM
    !-------------------

    !-------------------
    !IODINE
    !-------------------
    atomic_number = 53
    !MNDO
    ! Reference: M.J.S.DEWAR, E.F. HEALY, J.J.P. STEWART, J.COMP.CHEM., 
    !            5,358,(1984)
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -340.5983600D0
    s_orb_exp_mndo(atomic_number) = 2.2729610D0
    p_orb_exp_mndo(atomic_number) = 2.1694980D0
    betas_mndo(atomic_number) = -7.4144510D0
    betap_mndo(atomic_number) = -6.1967810D0
    GSS_mndo(atomic_number) = 15.04044855D0
    GSP_mndo(atomic_number) = 13.05655798D0
    GPP_mndo(atomic_number) = 11.14778369D0
    GP2_mndo(atomic_number) = 9.91409071D0
    HSP_mndo(atomic_number) = 2.45638202D0
    DD_mndo(atomic_number) = 1.4253233D0
    QQ_mndo(atomic_number) = 1.1841707D0
    AM_mndo(atomic_number) = 0.5527541D0
    AD_mndo(atomic_number) = 0.4593451D0
    AQ_mndo(atomic_number) = 0.4585376D0
    alp_mndo(atomic_number) = 2.2073200D0
    USS_mndo(atomic_number) = -100.0030538D0
    UPP_mndo(atomic_number) = -74.6114692D0

    !AM1
    ! Reference: M.J.S. DEWAR AND E. G. ZOEBISCH, THEOCHEM, 180, 1 
    !            (1988).
    element_supported_am1(atomic_number) = .true.
    elec_eng_am1(atomic_number) = -346.8642857D0
    s_orb_exp_am1(atomic_number) = 2.1028580D0
    p_orb_exp_am1(atomic_number) = 2.1611530D0
    betas_am1(atomic_number) = -8.4433270D0
    betap_am1(atomic_number) = -6.3234050D0
    FN1_am1(1,atomic_number) = 0.0043610D0
    FN2_am1(1,atomic_number) = 2.3000000D0
    FN3_am1(1,atomic_number) = 1.8000000D0
    FN1_am1(2,atomic_number) = 0.0157060D0
    FN2_am1(2,atomic_number) = 3.0000000D0
    FN3_am1(2,atomic_number) = 2.2400000D0
    FN1_am1(3,atomic_number) = 0.0000000D0
    FN2_am1(3,atomic_number) = 0.0000000D0
    FN3_am1(3,atomic_number) = 0.0000000D0
    FN1_am1(4,atomic_number) = 0.0000000D0
    FN2_am1(4,atomic_number) = 0.0000000D0
    FN3_am1(4,atomic_number) = 0.0000000D0
    NUM_FN_am1(atomic_number) = 2
    GSS_am1(atomic_number) = 15.0404486D0
    GSP_am1(atomic_number) = 13.0565580D0
    GPP_am1(atomic_number) = 11.1477837D0
    GP2_am1(atomic_number) = 9.9140907D0
    HSP_am1(atomic_number) = 2.4563820D0
    DD_am1(atomic_number) = 1.4878778D0
    QQ_am1(atomic_number) = 1.1887388D0
    AM_am1(atomic_number) = 0.5527544D0
    AD_am1(atomic_number) = 0.4497523D0
    AQ_am1(atomic_number) = 0.4631775D0
    alp_am1(atomic_number) = 2.2994240D0
    USS_am1(atomic_number) = -103.5896630D0
    UPP_am1(atomic_number) = -74.4299970D0

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM.10, 209 (1989).
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -288.3160860D0
    s_orb_exp_pm3(atomic_number) = 7.0010130D0
    p_orb_exp_pm3(atomic_number) = 2.4543540D0
    betas_pm3(atomic_number) = -14.4942340D0
    betap_pm3(atomic_number) = -5.8947030D0
    FN1_pm3(1,atomic_number) =-0.1314810D0
    FN2_pm3(1,atomic_number) = 5.2064170D0
    FN3_pm3(1,atomic_number) = 1.7488240D0
    FN1_pm3(2,atomic_number) =-0.0368970D0
    FN2_pm3(2,atomic_number) = 6.0101170D0
    FN3_pm3(2,atomic_number) = 2.7103730D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 13.6319430D0
    GSP_pm3(atomic_number) = 14.9904060D0
    GPP_pm3(atomic_number) = 7.2883300D0
    GP2_pm3(atomic_number) = 5.9664070D0
    HSP_pm3(atomic_number) = 2.6300350D0
    DD_pm3(atomic_number) = 0.1581469D0
    QQ_pm3(atomic_number) = 1.0467302D0
    AM_pm3(atomic_number) = 0.5009902D0
    AD_pm3(atomic_number) = 1.6699104D0
    AQ_pm3(atomic_number) = 0.5153082D0
    alp_pm3(atomic_number) = 1.9901850D0
    USS_pm3(atomic_number) = -96.4540370D0
    UPP_pm3(atomic_number) = -61.0915820D0

    !PDDG/PM3
    ! Reference Tubert-Brohman, Werneck, Repasky and Jorgensen, 2003,
    !           J Comp Chem, 25: 138-150
    element_supported_pddgpm3(atomic_number) = .true.
    elec_eng_pddgpm3(atomic_number) = -291.537869D0
    s_orb_exp_pddgpm3(atomic_number) = 5.062801D0
    p_orb_exp_pddgpm3(atomic_number) = 2.417757D0
    betas_pddgpm3(atomic_number) = -16.592621D0
    betap_pddgpm3(atomic_number) = -6.599816D0
    FN1_pddgpm3(1,atomic_number) = -0.136003D0
    FN2_pddgpm3(1,atomic_number) = 3.852912D0
    FN3_pddgpm3(1,atomic_number) = 1.697455D0
    FN1_pddgpm3(2,atomic_number) = -0.037287D0
    FN2_pddgpm3(2,atomic_number) = 5.229264D0
    FN3_pddgpm3(2,atomic_number) = 2.768669D0
    FN1_pddgpm3(3,atomic_number) = 0.0d0
    FN2_pddgpm3(3,atomic_number) = 0.0d0
    FN3_pddgpm3(3,atomic_number) = 0.0d0
    FN1_pddgpm3(4,atomic_number) = 0.0d0
    FN2_pddgpm3(4,atomic_number) = 0.0d0
    FN3_pddgpm3(4,atomic_number) = 0.0d0
    NUM_FN_pddgpm3(atomic_number) = 2
    GSS_pddgpm3(atomic_number) = 13.6319430D0
    GSP_pddgpm3(atomic_number) = 14.9904060D0
    GPP_pddgpm3(atomic_number) = 7.2883300D0
    GP2_pddgpm3(atomic_number) = 5.9664070D0
    HSP_pddgpm3(atomic_number) = 2.6300350D0
    DD_pddgpm3(atomic_number) = 0.407261D0
    QQ_pddgpm3(atomic_number) = 1.062574D0
    AM_pddgpm3(atomic_number) = 0.5009899562D0
    AD_pddgpm3(atomic_number) = 0.9393375791D0
    AQ_pddgpm3(atomic_number) = 0.5103170804D0
    alp_pddgpm3(atomic_number) = 1.978170D0
    USS_pddgpm3(atomic_number) = -97.664174D0
    UPP_pddgpm3(atomic_number) = -61.167137D0
    PDDGC1_pm3(atomic_number) = 0.012901D0
    PDDGC2_pm3(atomic_number) = -0.012825D0
    PDDGE1_pm3(atomic_number) = 1.994299D0
    PDDGE2_pm3(atomic_number) = 2.263417D0

    !PDDG/MNDO
    ! Reference: Repasky, Chandrasekhar and Jorgensen, 2002,
    !            J Comp Chem, 23: 1601-1622
    element_supported_pddgmndo(atomic_number) = .true.
    elec_eng_pddgmndo(atomic_number) = -356.076398D0
    s_orb_exp_pddgmndo(atomic_number) = 2.718404D0
    p_orb_exp_pddgmndo(atomic_number) = 2.461813D0
    betas_pddgmndo(atomic_number) = -6.698375D0
    betap_pddgmndo(atomic_number) = -5.693814D0
    GSS_pddgmndo(atomic_number) = 15.04044855D0
    GSP_pddgmndo(atomic_number) = 13.05655798D0
    GPP_pddgmndo(atomic_number) = 11.14778369D0
    GP2_pddgmndo(atomic_number) = 9.91409071D0
    HSP_pddgmndo(atomic_number) = 2.45638202D0
    DD_pddgmndo(atomic_number) = 1.209529D0
    QQ_pddgmndo(atomic_number) = 1.043559D0
    AM_pddgmndo(atomic_number) = 0.5527537084D0
    AD_pddgmndo(atomic_number) = 0.4988481596D0
    AQ_pddgmndo(atomic_number) = 0.50403175D0
    alp_pddgmndo(atomic_number) = 2.242446D0
    USS_pddgmndo(atomic_number) = -106.588422D0
    UPP_pddgmndo(atomic_number) = -75.282605D0
    PDDGC1_mndo(atomic_number) = 0.009616D0
    PDDGC2_mndo(atomic_number) = -0.007505D0
    PDDGE1_mndo(atomic_number) = 2.572332D0
    PDDGE2_mndo(atomic_number) = 2.936456D0

    !-------------------
    !END IODINE
    !-------------------

    !-------------------
    !MERCURY
    !-------------------
    atomic_number = 80
    !MNDO
    ! Reference: M.J.S.DEWAR,  ET. AL. ORGANOMETALLICS 4, 1964, (1985)
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -28.8191480D0
    s_orb_exp_mndo(atomic_number) = 2.2181840D0
    p_orb_exp_mndo(atomic_number) = 2.0650380D0
    betas_mndo(atomic_number) = -0.4045250D0
    betap_mndo(atomic_number) = -6.2066830D0
    GSS_mndo(atomic_number) = 10.8000000D0
    GSP_mndo(atomic_number) = 9.3000000D0
    GPP_mndo(atomic_number) = 14.3000000D0
    GP2_mndo(atomic_number) = 13.5000000D0
    HSP_mndo(atomic_number) = 1.3000000D0
    DD_mndo(atomic_number) = 1.7378048D0
    QQ_mndo(atomic_number) = 1.4608064D0
    AM_mndo(atomic_number) = 0.3969129D0
    AD_mndo(atomic_number) = 0.3047694D0
    AQ_mndo(atomic_number) = 0.3483102D0
    alp_mndo(atomic_number) = 1.3356410D0
    USS_mndo(atomic_number) = -19.8095740D0
    UPP_mndo(atomic_number) = -13.1025300D0

    !AM1
    ! Reference: M.J.S.Dewar and C.Jie, Organometallics 8, 1547, (1989)
    element_supported_am1(atomic_number) = .true.
    elec_eng_am1(atomic_number) = -29.0831560D0
    s_orb_exp_am1(atomic_number) = 2.0364130D0
    p_orb_exp_am1(atomic_number) = 1.9557660D0
    betas_am1(atomic_number) = -0.9086570D0
    betap_am1(atomic_number) = -4.9093840D0
    FN1_am1(1,atomic_number) = 0.0000000D0
    FN2_am1(1,atomic_number) = 0.0000000D0
    FN3_am1(1,atomic_number) = 0.0000000D0
    FN1_am1(2,atomic_number) = 0.0000000D0
    FN2_am1(2,atomic_number) = 0.0000000D0
    FN3_am1(2,atomic_number) = 0.0000000D0
    FN1_am1(3,atomic_number) = 0.0000000D0
    FN2_am1(3,atomic_number) = 0.0000000D0
    FN3_am1(3,atomic_number) = 0.0000000D0
    FN1_am1(4,atomic_number) = 0.0000000D0
    FN2_am1(4,atomic_number) = 0.0000000D0
    FN3_am1(4,atomic_number) = 0.0000000D0
    NUM_FN_am1(atomic_number) = 0
    GSS_am1(atomic_number) = 10.8000000D0
    GSP_am1(atomic_number) = 9.3000000D0
    GPP_am1(atomic_number) = 14.3000000D0
    GP2_am1(atomic_number) = 13.5000000D0
    HSP_am1(atomic_number) = 1.3000000D0
    DD_am1(atomic_number) = 1.8750829D0
    QQ_am1(atomic_number) = 1.5424241D0
    AM_am1(atomic_number) = 0.3969129D0
    AD_am1(atomic_number) = 0.2926605D0
    AQ_am1(atomic_number) = 0.3360599D0
    alp_am1(atomic_number) = 1.4847340D0
    USS_am1(atomic_number) = -19.9415780D0
    UPP_am1(atomic_number) = -11.1108700D0

    !PM3
    ! Reference:  J. J. P. STEWART, J. COMP. CHEM.(ACCEPTED)
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -28.8997380D0
    s_orb_exp_pm3(atomic_number) = 1.4768850D0
    p_orb_exp_pm3(atomic_number) = 2.4799510D0
    betas_pm3(atomic_number) = -3.1013650D0
    betap_pm3(atomic_number) = -3.4640310D0
    FN1_pm3(1,atomic_number) = 1.0827200D0
    FN2_pm3(1,atomic_number) = 6.4965980D0
    FN3_pm3(1,atomic_number) = 1.1951460D0
    FN1_pm3(2,atomic_number) =-0.0965530D0
    FN2_pm3(2,atomic_number) = 3.9262810D0
    FN3_pm3(2,atomic_number) = 2.6271600D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 6.6247200D0
    GSP_pm3(atomic_number) = 10.6392970D0
    GPP_pm3(atomic_number) = 14.7092830D0
    GP2_pm3(atomic_number) = 16.0007400D0
    HSP_pm3(atomic_number) = 2.0363110D0
    DD_pm3(atomic_number) = 1.2317811D0
    QQ_pm3(atomic_number) = 1.2164033D0
    AM_pm3(atomic_number) = 0.2434664D0
    AD_pm3(atomic_number) = 0.4515472D0
    AQ_pm3(atomic_number) = 0.2618394D0
    alp_pm3(atomic_number) = 1.5293770D0
    USS_pm3(atomic_number) = -17.7622290D0
    UPP_pm3(atomic_number) = -18.3307510D0

    !-------------------
    !END MERCURY
    !-------------------

    !-------------------
    !THALLIUM
    !-------------------
    atomic_number = 81
    !MNDO
    ! Reference: None
    element_supported_mndo(atomic_number) = .false.

    !AM1
    ! Reference: None
    element_supported_am1(atomic_number) = .false.

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM. (ACCEPTED) 
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -56.6492050D0
    s_orb_exp_pm3(atomic_number) = 6.8679210D0
    p_orb_exp_pm3(atomic_number) = 1.9694450D0
    betas_pm3(atomic_number) = -1.0844950D0
    betap_pm3(atomic_number) = -7.9467990D0
    FN1_pm3(1,atomic_number) =-1.3613990D0
    FN2_pm3(1,atomic_number) = 3.5572260D0 
    FN3_pm3(1,atomic_number) = 1.0928020D0
    FN1_pm3(2,atomic_number) =-0.0454010D0
    FN2_pm3(2,atomic_number) = 2.3069950D0
    FN3_pm3(2,atomic_number) = 2.9650290D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 10.4604120D0
    GSP_pm3(atomic_number) = 11.2238830D0
    GPP_pm3(atomic_number) = 4.9927850D0
    GP2_pm3(atomic_number) = 8.9627270D0
    HSP_pm3(atomic_number) = 2.5304060D0
    DD_pm3(atomic_number) = 0.0781362D0
    QQ_pm3(atomic_number) = 1.5317110D0
    AM_pm3(atomic_number) = 0.3844326D0
    AD_pm3(atomic_number) = 2.5741815D0
    AQ_pm3(atomic_number) = 0.2213264D0
    alp_pm3(atomic_number) = 1.3409510D0
    USS_pm3(atomic_number) = -30.0531700D0
    UPP_pm3(atomic_number) = -26.9206370D0

    !-------------------
    !END THALLIUM
    !-------------------

    !-------------------
    !LEAD
    !-------------------
    atomic_number = 82
    !MNDO
    ! Reference: M.J.S.DEWAR, ET.AL ORGANOMETALLICS 4 1973-1980 (1985)
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -105.8345040D0
    s_orb_exp_mndo(atomic_number) = 2.4982860D0
    p_orb_exp_mndo(atomic_number) = 2.0820710D0
    betas_mndo(atomic_number) = -8.0423870D0
    betap_mndo(atomic_number) = -3.0000000D0
    GSS_mndo(atomic_number) = 9.8000000D0
    GSP_mndo(atomic_number) = 8.3000000D0
    GPP_mndo(atomic_number) = 7.3000000D0
    GP2_mndo(atomic_number) = 6.5000000D0
    HSP_mndo(atomic_number) = 1.3000000D0
    DD_mndo(atomic_number) = 1.5526624D0
    QQ_mndo(atomic_number) = 1.4488558D0
    AM_mndo(atomic_number) = 0.3601617D0
    AD_mndo(atomic_number) = 0.3239309D0
    AQ_mndo(atomic_number) = 0.3502057D0
    alp_mndo(atomic_number) = 1.7283330D0
    USS_mndo(atomic_number) = -47.3196920D0
    UPP_mndo(atomic_number) = -28.8475600D0

    !AM1
    ! Reference: None
    element_supported_am1(atomic_number) = .false.

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM. (ACCEPTED)
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -73.4660775D0
    s_orb_exp_pm3(atomic_number) = 3.1412890D0
    p_orb_exp_pm3(atomic_number) = 1.8924180D0
    betas_pm3(atomic_number) = -6.1260240D0
    betap_pm3(atomic_number) = -1.3954300D0
    FN1_pm3(1,atomic_number) =-0.1225760D0
    FN2_pm3(1,atomic_number) = 6.0030620D0
    FN3_pm3(1,atomic_number) = 1.9015970D0
    FN1_pm3(2,atomic_number) =-0.0566480D0
    FN2_pm3(2,atomic_number) = 4.7437050D0
    FN3_pm3(2,atomic_number) = 2.8618790D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 7.0119920D0
    GSP_pm3(atomic_number) = 6.7937820D0
    GPP_pm3(atomic_number) = 5.1837800D0
    GP2_pm3(atomic_number) = 5.0456510D0
    HSP_pm3(atomic_number) = 1.5663020D0
    DD_pm3(atomic_number) = 0.9866290D0
    QQ_pm3(atomic_number) = 1.5940562D0
    AM_pm3(atomic_number) = 0.2576991D0
    AD_pm3(atomic_number) = 0.4527678D0
    AQ_pm3(atomic_number) = 0.2150175D0
    alp_pm3(atomic_number) = 1.6200450D0
    USS_pm3(atomic_number) = -30.3227560D0
    UPP_pm3(atomic_number) = -24.4258340D0

    !-------------------
    !END LEAD
    !-------------------

    !-------------------
    !BISMUTH
    !-------------------
    atomic_number = 83
    !MNDO
    ! Reference: None
    element_supported_mndo(atomic_number) = .false.

    !AM1
    ! Reference: None
    element_supported_am1(atomic_number) = .false.

    !PM3
    ! Reference: J. J. P. STEWART, J. COMP. CHEM. (ACCEPTED)
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -109.2774910D0
    s_orb_exp_pm3(atomic_number) = 4.9164510D0
    p_orb_exp_pm3(atomic_number) = 1.9349350D0
    betas_pm3(atomic_number) = -5.6072830D0
    betap_pm3(atomic_number) = -5.8001520D0
    FN1_pm3(1,atomic_number) = 2.5816930D0
    FN2_pm3(1,atomic_number) = 5.0940220D0
    FN3_pm3(1,atomic_number) = 0.4997870D0
    FN1_pm3(2,atomic_number) = 0.0603200D0
    FN2_pm3(2,atomic_number) = 6.0015380D0
    FN3_pm3(2,atomic_number) = 2.4279700D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0 
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 4.9894800D0
    GSP_pm3(atomic_number) = 6.1033080D0
    GPP_pm3(atomic_number) = 8.6960070D0
    GP2_pm3(atomic_number) = 8.3354470D0
    HSP_pm3(atomic_number) = 0.5991220D0
    DD_pm3(atomic_number) = 0.2798609D0
    QQ_pm3(atomic_number) = 1.5590294D0
    AM_pm3(atomic_number) = 0.1833693D0
    AD_pm3(atomic_number) = 0.6776013D0
    AQ_pm3(atomic_number) = 0.2586520D0
    alp_pm3(atomic_number) = 1.8574310D0
    USS_pm3(atomic_number) = -33.4959380D0
    UPP_pm3(atomic_number) = -35.5210260D0

    !-------------------
    !END BISMUTH
    !-------------------

    !--------------------------------------------------------------------
    ! Special section for GHO atom: Atom number 85 (Based on Carbon atom)
    !--------------------------------------------------------------------
    atomic_number = 85
    !MNDO
    ! GHO parameters based on AM1 GHO parameters
    ! Referece: Gao, J., Amara, P., Alhambra, C., Field, MJ.,
    !           J. Phys. Chem. A. 102, 4714 (1998)
    element_supported_mndo(atomic_number) = .true.
    elec_eng_mndo(atomic_number) = -120.5006060D0
    s_orb_exp_mndo(atomic_number) = 1.7875370D0
    p_orb_exp_mndo(atomic_number) = 1.7875370D0
    betas_mndo(atomic_number) =  -5.5005241D0
    betap_mndo(atomic_number) = -14.6666377D0
    GSS_mndo(atomic_number) = 12.2300000D00
    GSP_mndo(atomic_number) = 11.4700000D00
    GPP_mndo(atomic_number) = 11.0800000D00
    GP2_mndo(atomic_number) = 9.8400000D00
    HSP_mndo(atomic_number) = 2.4300000D00
    DD_mndo(atomic_number) = 0.8074662D0
    QQ_mndo(atomic_number) = 0.6851578D0
    AM_mndo(atomic_number) = 0.4494671D0
    AD_mndo(atomic_number) = 0.6149474D0
    AQ_mndo(atomic_number) = 0.6685897D0
    alp_mndo(atomic_number) = 2.5463800D0
    USS_mndo(atomic_number) = -52.0286580D0
    UPP_mndo(atomic_number) = -38.7031115D0

    !AM1
    ! Reference:  Gao, J., Amara, P., Alhambra, C., Field, MJ., 
    !             J. Phys. Chem. A. 102, 4714 (1998)
    element_supported_am1(atomic_number) = .true.
    elec_eng_am1(atomic_number) = -120.8157940D0
    s_orb_exp_am1(atomic_number) = 1.8086650D0
    p_orb_exp_am1(atomic_number) = 1.6851160D0
    betas_am1(atomic_number) =  -5.5005241D0
    betap_am1(atomic_number) = -14.6666377D0
    FN1_am1(1,atomic_number) = 0.0113550D0
    FN2_am1(1,atomic_number) = 5.0000000D0
    FN3_am1(1,atomic_number) = 1.6000000D0
    FN1_am1(2,atomic_number) = 0.0459240D0
    FN2_am1(2,atomic_number) = 5.0000000D0
    FN3_am1(2,atomic_number) = 1.8500000D0
    FN1_am1(3,atomic_number) =-0.0200610D0
    FN2_am1(3,atomic_number) = 5.0000000D0
    FN3_am1(3,atomic_number) = 2.0500000D0
    FN1_am1(4,atomic_number) =-0.0012600D0
    FN2_am1(4,atomic_number) = 5.0000000D0
    FN3_am1(4,atomic_number) = 2.6500000D0
    NUM_FN_am1(atomic_number) = 4
    GSS_am1(atomic_number) = 12.2300000D0
    GSP_am1(atomic_number) = 11.4700000D0
    GPP_am1(atomic_number) = 11.0800000D0
    GP2_am1(atomic_number) = 9.8400000D0
    HSP_am1(atomic_number) = 2.4300000D0
    DD_am1(atomic_number) = 0.8236736D0
    QQ_am1(atomic_number) = 0.7268015D0
    AM_am1(atomic_number) = 0.4494671D0
    AD_am1(atomic_number) = 0.6082946D0
    AQ_am1(atomic_number) = 0.6423492D0
    alp_am1(atomic_number) = 2.6482740D0
    USS_am1(atomic_number) = -52.0286580D0
    UPP_am1(atomic_number) = -38.7031115D0

    !PM3
    ! Reference: Garcia-Viloca, M., Gao, J., Theor. Chem. Acc. 
    !            111, 280-286 (2004)
    element_supported_pm3(atomic_number) = .true.
    elec_eng_pm3(atomic_number) = -111.2299170D0
    s_orb_exp_pm3(atomic_number) = 1.5650850D0
    p_orb_exp_pm3(atomic_number) = 1.8423450D0
    betas_pm3(atomic_number) =  -2.3820030D0
    betap_pm3(atomic_number) = -14.7041325D0
    FN1_pm3(1,atomic_number) = 0.0501070D0
    FN2_pm3(1,atomic_number) = 6.0031650D0
    FN3_pm3(1,atomic_number) = 1.6422140D0
    FN1_pm3(2,atomic_number) = 0.0507330D0
    FN2_pm3(2,atomic_number) = 6.0029790D0
    FN3_pm3(2,atomic_number) = 0.8924880D0
    FN1_pm3(3,atomic_number) = 0.0d0
    FN2_pm3(3,atomic_number) = 0.0d0
    FN3_pm3(3,atomic_number) = 0.0d0
    FN1_pm3(4,atomic_number) = 0.0d0
    FN2_pm3(4,atomic_number) = 0.0d0
    FN3_pm3(4,atomic_number) = 0.0d0
    NUM_FN_pm3(atomic_number) = 2
    GSS_pm3(atomic_number) = 11.2007080D0
    GSP_pm3(atomic_number) = 10.2650270D0
    GPP_pm3(atomic_number) = 10.7962920D0
    GP2_pm3(atomic_number) = 9.0425660D0
    HSP_pm3(atomic_number) = 2.2909800D0
    DD_pm3(atomic_number) = 0.8332396D0
    QQ_pm3(atomic_number) = 0.6647750D0
    AM_pm3(atomic_number) = 0.4116394D0
    AD_pm3(atomic_number) = 0.5885862D0
    AQ_pm3(atomic_number) = 0.7647667D0
    alp_pm3(atomic_number) = 2.7078070D0
    USS_pm3(atomic_number) = -47.2703200D0
    UPP_pm3(atomic_number) = -35.2877112D0

    ! note: it has never been tested, so you should test the parameters by yourself.
    ! temporary working parameters for PDDG/PM3-GHO: just copied from C/GHO, PM3, PDDG/PM3 paramters.
    ! core-core interaction: from PDDG-PM3 Carbon 
    ! electronic parameters: from PM3-GHO parameters
    element_supported_pddgpm3(atomic_number) = .true.
    elec_eng_pddgpm3(atomic_number) = -113.42824208974D0    ! pddg/pm3
    s_orb_exp_pddgpm3(atomic_number) = 1.56786358751710D0   ! pddg/pm3
    p_orb_exp_pddgpm3(atomic_number) = 1.84665852120070D0   ! pddg/pm3
    betas_pddgpm3(atomic_number) = -2.3820030D0             ! pm3-gho
    betap_pddgpm3(atomic_number) = -14.7041325D0            ! pm3-gho
    FN1_pddgpm3(1,atomic_number) = 0.04890550330860D0       ! pddg/pm3
    FN2_pddgpm3(1,atomic_number) = 5.76533980799120D0       ! pddg/pm3
    FN3_pddgpm3(1,atomic_number) = 1.68223169651660D0       ! pddg/pm3
    FN1_pddgpm3(2,atomic_number) = 0.04769663311610D0       ! pddg/pm3
    FN2_pddgpm3(2,atomic_number) = 5.97372073873460D0       ! pddg/pm3
    FN3_pddgpm3(2,atomic_number) = 0.89440631619350D0       ! pddg/pm3
    FN1_pddgpm3(3,atomic_number) = 0.0d0                    ! common
    FN2_pddgpm3(3,atomic_number) = 0.0d0                    ! common
    FN3_pddgpm3(3,atomic_number) = 0.0d0                    ! common
    FN1_pddgpm3(4,atomic_number) = 0.0d0                    ! common
    FN2_pddgpm3(4,atomic_number) = 0.0d0                    ! common
    FN3_pddgpm3(4,atomic_number) = 0.0d0                    ! common
    NUM_FN_pddgpm3(atomic_number) = 2                       ! common
    GSS_pddgpm3(atomic_number) = 11.2007080D0               ! common
    GSP_pddgpm3(atomic_number) = 10.2650270D0               ! common
    GPP_pddgpm3(atomic_number) = 10.7962920D0               ! common
    GP2_pddgpm3(atomic_number) = 9.0425660D0                ! common
    HSP_pddgpm3(atomic_number) = 2.2909800D0                ! common
    DD_pddgpm3(atomic_number) = 0.83141317761820D0          ! pddg/pm3
    QQ_pddgpm3(atomic_number) = 0.66322216984400D0          ! pddg/pm3
    AM_pddgpm3(atomic_number) = 0.41163939928160D0          ! pddg/pm3
    AD_pddgpm3(atomic_number) = 0.58929751233060D0          ! pddg/pm3
    AQ_pddgpm3(atomic_number) = 0.76594914927100D0          ! pddg/pm3
    alp_pddgpm3(atomic_number) = 2.72577212540530D0         ! pddg/pm3
    USS_pddgpm3(atomic_number) = -48.241240946951D0         ! pddg/pm3
    UPP_pddgpm3(atomic_number) = -36.461255999939D0         ! pddg/pm3
    PDDGC1_pm3(atomic_number) = -0.0007433618099D0          ! pddg/pm3  
    PDDGC2_pm3(atomic_number) = 0.00098516072940D0          ! pddg/pm3
    PDDGE1_pm3(atomic_number) = 0.83691519687330D0          ! pddg/pm3 
    PDDGE2_pm3(atomic_number) = 1.58523608520060D0          ! pddg/pm3

    !-------------------
    !END of GHO atom
    !-------------------
    !------------------------------------------------------------
    !END OF SECTION 2 - MNDO, AM1, PM3 and PDDG PARAMS BY ELEMENT
    !------------------------------------------------------------

    end if       ! .not. q_parm_loaded
    return
  end subroutine qm2_initialize_elements
#endif 
end module qm2_parameters

