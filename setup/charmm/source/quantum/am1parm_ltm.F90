module am1parm

  use chm_kinds
  use dimens_fcm
  implicit none
  integer :: dummy_am1_var
#if KEY_QUANTUM==1 /*quantum*/

      CHARACTER(len=2),dimension(0:100),save :: ELEMNT                       ! COMMON /MLEMTS/
      INTEGER,dimension(0:100),save :: NATORB                                ! COMMON /MATORB/
      real(chm_real),dimension(0:100) :: AMS                                 ! COMMON /MSTOPE/
      real(chm_real),dimension(0:100) :: CORE                                ! COMMON /MCORE /
      real(chm_real),dimension(0:100) :: EHEAT                               ! COMMON /EHEAT /



!     Atom parameter availability arrays for the MNDO, AM1 and PM3
!     semi-empirical parametrisations.

      LOGICAL,dimension(0:100),save :: AM1PAR, MNDPAR, PM3PAR                ! COMMON /MOPARM/
      INTEGER,parameter :: NGAUSS=6

!     AM1 common blocks.
!     Parameter common blocks. Used for AM1 parameters before set up
!     and for MNDO, AM1 or PM3 parameters after depending upon the options
!     selected.

      real(chm_real),save ::ALFA(0:100)                                      ! COMMON /ALPHAM/
      real(chm_real),save ::EISOL(0:100)                                     ! COMMON /ATOMIM/
      real(chm_real),save ::BETAS(0:100), BETAP(0:100), BETAD(0:100)         ! COMMON /BETASM/
      real(chm_real),save ::ZS(0:100), ZP(0:100), ZD(0:100)                  ! COMMON /MXPONT/
      real(chm_real),save ::FN1(0:100,10), FN2(0:100,10), FN3(0:100,10)      ! COMMON /IDEASM/
      real(chm_real),save ::DD(0:100), QQ(0:100), &                          ! COMMON /NULTIP/
                            bdd(0:100,3)
!                           bdd(:,1:3) = AM(0:100), AD(0:100),AQ(0:100)
      real(chm_real),save ::USS(0:100), UPP(0:100), UDD(0:100)               ! COMMON /ONELEM/
      real(chm_real),save ::GSS(0:100), GSP(0:100), GPP(0:100), GP2(0:100),& ! COMMON /MWOELE/
                            HSP(0:100), GSD(0:100), GPD(0:100), GDD(0:100)
      real(chm_real),save ::VS(0:100), VP(0:100), VD(0:100)                  ! COMMON /VSIPSM/

!     PM3 common blocks.
!     Common blocks holding data for the PM3 parametrisation. They
!     are only used in setting up the calculation.

      real(chm_real),save :: ALPP(0:100), EISOLP(0:100), BETASP(0:100), &    ! COMMON /MPM3/
                             BETAPP(0:100), BETADP(0:100), &
                             ZSP(0:100), ZPP(0:100), ZDP(0:100), &
                             FN1P(0:100,10),FN2P(0:100,10), FN3P(0:100,10), &
                             DDP(0:100), QQP(0:100), &
                             AMP(0:100), ADP(0:100), AQP(0:100), &
                             USSP(0:100), UPPP(0:100), UDDP(0:100), &
                             GSSP(0:100), GSPP(0:100), GPPP(0:100), &
                             GP2P(0:100), HSPP(0:100), GSDP(0:100), &
                             GPDP(0:100), GDDP(0:100), &
                             VSP(0:100), VPP(0:100), VDP(0:100)

!     MNDO common blocks.
!     Common blocks holding data for the MNDO parametrisation. They
!     are only used in setting up the calculation.

      real(chm_real),save :: ALPM(0:100), EISOLM(0:100), &                   ! COMMON /MNDOM/
                             BETASM(0:100), BETAPM(0:100), BETADM(0:100), &
                             ZSM(0:100), ZPM(0:100), ZDM(0:100), &
                             DDM(0:100), QQM(0:100), AMM(0:100), ADM(0:100), &
                             AQM(0:100), USSM(0:100), UPPM(0:100), UDDM(0:100), &
                             GSSM(0:100), GSPM(0:100), GPPM(0:100), GP2M(0:100), &
                             HSPM(0:100)
!
! MOVED FROM BLOCK DATA .... quantum/qmdata.src
! later this will move back as a real module!
!
      DATA ELEMNT /'QQ', &
                   ' H','HE','LI','BE',' B',' C',' N',' O',' F','NE', &
                   'NA','MG','AL','SI',' P',' S','CL','AR',' K','CA', &
                   'SC','TI',' V','CR','MN','FE','CO','NI','CU','ZN', &
                   'GA','GE','AS','SE','BR','KR','RB','SR',' Y','ZR', &
                   'NB','MO','TC','RU','RH','PD','AG','CD','IN','SN', &
                   'SB','TE',' I','XE','CS','BA','LA','CE','PR','ND', &
                   'PM','SM','EU','GD','TB','DY','HO','ER','TM','YB', &
                   'LU','HF','TA',' W','RE','OS','IR','PT','AU','HG', &
                   'TL','PB','BI','PO','AT','RN','FR','RA','AC','TH', &
                   'PA',' U','NP','PU','AM','CM','BK','CF','ES','FM'/
!
!     The number of orbitals on each atom.
!
      DATA NATORB / 1, &
                    1, 1, 4, 4, 4, 4, 4, 4, 4, 4, &
                    0, 4, 4, 4, 4, 4, 4, 4, 0, 4, &
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 4, &
                    4, 4, 4, 4, 4, 4, 4, 4, 9, 9, &
                    9, 9, 9, 9, 9, 9, 9, 4, 4, 4, &
                    4, 4, 4, 4, 2, 2, 8, 8, 8, 8, &
                    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, &
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 4, &
                    4, 4, 4, 4, 4, 4, 0, 0, 0, 0, &
                    4, 4, 4, 4, 4, 4, 0, 0, 0, 0/

!
!     Atomic masses of the most common isotopes for each atom.
!
      DATA  AMS / 1.00790D0, &
         1.00790D0,   4.00260D0,   6.94000D0,   9.01218D0,  10.81000D0, &
        12.01100D0,  14.00670D0,  15.99940D0,  18.99840D0,  20.17900D0, &
        22.98977D0,  24.30500D0,  26.98154D0,  28.08550D0,  30.97376D0, &
        32.06000D0,  35.45300D0,  39.94800D0,  39.09830D0,  40.08000D0, &
        44.95590D0,  47.90000D0,  50.94150D0,  51.99600D0,  54.93800D0, &
        55.84700D0,  58.93320D0,  58.71000D0,  63.54600D0,  65.38000D0, &
        69.73500D0,  72.59000D0,  74.92160D0,  78.96000D0,  79.90400D0, &
        83.80000D0,  85.46780D0,  87.62000D0,  88.90590D0,  91.22000D0, &
        92.90640D0,  95.94000D0,  98.90620D0, 101.0700D0,  102.9055D0, &
       106.4000D0,  107.8680D0,  112.4100D0,  114.8200D0,  118.6900D0, &
       121.7500D0,  127.6000D0,  126.9045D0,  131.3000D0,  132.9054D0, &
       137.3300D0,    0.0000D0,    0.0000D0,    0.0000D0,    0.0000D0, &
         0.0000D0,    0.0000D0,    0.0000D0,    0.0000D0,    0.0000D0, &
         0.0000D0,    0.0000D0,    0.0000D0,    0.0000D0,    0.0000D0, &
         0.0000D0,  178.4900D0,  180.9479D0,  183.8500D0,  186.2070D0, &
       190.2000D0,  192.2200D0,  195.0900D0,  196.9665D0,  200.5900D0, &
       204.3700D0,  207.2000D0, 208.9804D0,     0.0000D0,    0.0000D0, &
       15*0.0000D0/
!
!     Charges on the atom cores.
!
      DATA CORE / 1.D0, &
         1.D0, 0.D0, 1.D0, 2.D0, 3.D0, 4.D0, 5.D0, 6.D0, 7.D0, 0.D0, &
         1.D0, 2.D0, 3.D0, 4.D0, 5.D0, 6.D0, 7.D0, 0.D0, 1.D0, 2.D0, &
         3.D0, 4.D0, 5.D0, 6.D0, 7.D0, 8.D0, 9.D0,10.D0,11.D0, 2.D0, &
         3.D0, 4.D0, 5.D0, 6.D0, 7.D0, 0.D0, 1.D0, 2.D0, 3.D0, 4.D0, &
         5.D0, 6.D0, 7.D0, 8.D0, 9.D0,10.D0,11.D0, 2.D0, 3.D0, 4.D0, &
         5.D0, 6.D0, 7.D0, 0.D0, 1.D0, 2.D0, 3.D0, 4.D0, 5.D0, 6.D0, &
         7.D0, 8.D0, 9.D0,10.D0,11.D0,12.D0,13.D0,14.D0,15.D0,16.D0, &
         3.D0, 4.D0, 5.D0, 6.D0, 7.D0, 8.D0, 9.D0,10.D0,11.D0, 2.D0, &
         3.D0, 4.D0, 5.D0, 6.D0, 7.D0, 0.D0, 0.D0, 0.D0, 0.D0, 0.D0, &
         0.D0, 0.D0, 0.D0, 0.D0, 0.D0, 0.D0, 0.D0, 0.D0, 0.D0, 0.D0/
!
!     Enthalpies of formation of gaseous atoms from the CRC Handbook
!     1981-1982 and Annual Reports 71B, p119 (1974).
!
       DATA EHEAT  /  52.102D0, &
            52.102D0,   0.000D0,  38.410D0,  76.960D0, 135.700D0, &
           170.890D0, 113.000D0,  59.559D0,  18.890D0,   0.000D0, &
             0.000D0,  35.000D0,  79.490D0, 108.390D0,  75.570D0, &
            66.400D0,  28.990D0,   0.000D0,   0.000D0,  42.600D0, &
            90.300D0, 112.300D0, 122.900D0,  95.000D0,  67.700D0, &
            99.300D0, 102.400D0, 102.800D0,  80.700D0,  31.170D0, &
            65.400D0,  89.500D0,  72.300D0,  54.300D0,  26.740D0, &
             0.000D0,  19.600D0,  39.100D0, 101.500D0, 145.500D0, &
           172.400D0, 157.300D0,   0.000D0, 155.500D0, 133.000D0, &
            90.000D0,  68.100D0,  26.720D0,  58.000D0,  72.200D0, &
            63.200D0,  47.000D0,  25.517D0,   0.000D0,  18.700D0, &
            42.500D0,   0.000D0, 101.300D0,   0.000D0,   0.000D0, &
             0.000D0,  49.400D0,   0.000D0,   0.000D0,   0.000D0, &
             0.000D0,   0.000D0,  75.800D0,   0.000D0,  36.350D0, &
             0.000D0, 148.000D0, 186.900D0, 203.100D0, 185.000D0, &
           188.000D0, 160.000D0, 135.200D0,  88.000D0,  14.690D0, &
            43.550D0,  46.620D0,  50.100D0,   0.000D0,   0.000D0, &
         15* 0.000D0/
!
!     AM1 parameter availability array.
!
      DATA AM1PAR / .TRUE., &
                    .TRUE., .FALSE.,  .TRUE.,  .TRUE.,  .TRUE., &
                    .TRUE.,  .TRUE.,  .TRUE.,  .TRUE., .FALSE., &
                   .FALSE., .FALSE.,  .TRUE.,  .TRUE.,  .TRUE., &
                    .TRUE.,  .TRUE., .FALSE., .FALSE., .FALSE., &
                   .FALSE., .FALSE., .FALSE.,  .TRUE., .FALSE., &
                   .FALSE., .FALSE., .FALSE., .FALSE., .FALSE., &
                   .FALSE., .FALSE., .FALSE., .FALSE.,  .TRUE., &
              15 * .FALSE., &
                   .FALSE., .FALSE.,  .TRUE., .FALSE., .FALSE., &
              45 * .FALSE./
!
!     PM3 parameter availability array.
!     new parameters from mopac97 block.f doesn't contain
!     2 electron integral parameters, so Li, Ca (from Wavefunction), Ti
!
      DATA PM3PAR / .TRUE., &
                    .TRUE., .FALSE.,  .TRUE.,  .TRUE., .FALSE., &
                    .TRUE.,  .TRUE.,  .TRUE.,  .TRUE., .FALSE., &
                   .FALSE.,  .TRUE.,  .TRUE.,  .TRUE.,  .TRUE., &
                    .TRUE.,  .TRUE., .FALSE., .FALSE.,  .TRUE., &
                   .FALSE.,  .TRUE., .FALSE., .FALSE., .FALSE., &
                   .FALSE., .FALSE., .FALSE., .FALSE.,  .TRUE., &
                    .TRUE.,  .TRUE.,  .TRUE.,  .TRUE.,  .TRUE., &
              10 * .FALSE., &
                   .FALSE., .FALSE.,  .TRUE.,  .TRUE.,  .TRUE., &
                    .TRUE.,  .TRUE.,  .TRUE., .FALSE., .FALSE., &
              20 * .FALSE., &
                   .FALSE., .FALSE., .FALSE., .FALSE.,  .TRUE., &
                    .TRUE.,  .TRUE.,  .TRUE., .FALSE., .FALSE., &
              15 * .FALSE./
!
!     MNDO parameter availability array.
!
      DATA MNDPAR / .TRUE., &
                    .TRUE., .FALSE.,  .TRUE.,  .TRUE.,  .TRUE., &
                    .TRUE.,  .TRUE.,  .TRUE.,  .TRUE., .FALSE., &
                    .TRUE., .FALSE.,  .TRUE.,  .TRUE.,  .TRUE., &
                    .TRUE.,  .TRUE., .FALSE.,  .TRUE., .FALSE., &
                   .FALSE., .FALSE., .FALSE.,  .TRUE., .FALSE., &
                   .FALSE., .FALSE., .FALSE., .FALSE., .FALSE., &
                   .FALSE.,  .TRUE., .FALSE., .FALSE.,  .TRUE., &
              10 * .FALSE., &
                   .FALSE., .FALSE., .FALSE., .FALSE.,  .TRUE., &
                   .FALSE., .FALSE.,  .TRUE., .FALSE., .FALSE., &
              20 * .FALSE., &
                   .FALSE., .FALSE., .FALSE., .FALSE.,  .TRUE., &
                   .FALSE.,  .TRUE., .FALSE., .FALSE., .FALSE., &
              15 * .FALSE./
!=========================================================================
!     Capped bond parameters (hydrogen-like with zero charge).
!=========================================================================
!      DATA ALFA(0)  /  2.5441341D0/, ALPM(0)   /  2.5441341D0/
!      DATA EISOL(0) /  4.0000000D0/, EISOLM(0) /  4.0000000D0/
!      DATA BETAS(0) / -9.9999990D6/, BETASM(0) / -9.9999990D6/
!      DATA ZS(0)    /  4.0000000D0/, ZSM(0)    /  4.0000000D0/
!      DATA ZP(0)    /  0.3000000D0/, ZPM(0)    /  0.3000000D0/
!      DATA DD(0)    /  0.0684105D0/, DDM(0)    /  0.0684105D0/
!      DATA QQ(0)    /  1.0540926D0/, QQM(0)    /  1.0540926D0/
!      DATA AM(0)    /  0.4721793D0/, AMM(0)    /  0.4721793D0/
!      DATA BDD(0,2)    /  0.9262742D0/, ADM(0)    /  0.9262742D0/
!      DATA BDD(0,3)    /  0.2909059D0/, AQM(0)    /  0.2909059D0/
!      DATA USS(0)   /-11.9062760D0/, USSM(0)   /-11.9062760D0/
!      DATA GSS(0)   / 12.8480000D0/, GSSM(0)   / 12.8480000D0/
!      DATA HSP(0)   /  0.1000000D0/, HSPM(0)   /  0.1000000D0/
!=========================================================================
!     Capped bond parameters (temporarily using hydrogen parameters).
!=========================================================================
      DATA ALFA(0)  /  2.8823240D0/, ALPM(0)   /  2.5441341D0/
      DATA EISOL(0) /-11.3964270D0/, EISOLM(0) /-11.9062760D0/
      DATA BETAS(0) / -6.1737870D0/, BETASM(0) / -6.9890640D0/
      DATA ZS(0)    /  1.1880780D0/, ZSM(0)    /  1.3319670D0/
      DATA BDD(0,1)    /  0.4721793D0/, AMM(0)    /  0.4721793D0/
      DATA BDD(0,2)    /  0.4721793D0/, ADM(0)    /  0.4721793D0/
      DATA BDD(0,3)    /  0.4721793D0/, AQM(0)    /  0.4721793D0/
      DATA USS(0)   /-11.3964270D0/, USSM(0)   /-11.9062760D0/
      DATA GSS(0)   / 12.8480000D0/, GSSM(0)   / 12.8480000D0/
      DATA VS(0)    /-13.6050000D0/
      DATA VP(0)    /  0.0000000D0/
!
      DATA FN1(0,1) /  0.1227960D0/
      DATA FN2(0,1) /  5.0000000D0/
      DATA FN3(0,1) /  1.2000000D0/
      DATA FN1(0,2) /  0.0050900D0/
      DATA FN2(0,2) /  5.0000000D0/
      DATA FN3(0,2) /  1.8000000D0/
      DATA FN1(0,3) / -0.0183360D0/
      DATA FN2(0,3) /  2.0000000D0/
      DATA FN3(0,3) /  2.1000000D0/
!      DATA REFPM3 ( 1)/'  H: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1 10, 209 (1989).                '/
      DATA USSP   (  0)/     -13.0733210D0/
      DATA BETASP (  0)/      -5.6265120D0/
      DATA ZSP    (  0)/       0.9678070D0/
      DATA ALPP   (  0)/       3.3563860D0/
      DATA EISOLP (  0)/     -13.0733210D0/
      DATA GSSP   (  0)/      14.7942080D0/
      DATA AMP    (  0)/       0.5437048D0/
      DATA ADP    (  0)/       0.5437048D0/
      DATA AQP    (  0)/       0.5437048D0/
      DATA FN1P   (  0,1)/       1.1287500D0/
      DATA FN2P   (  0,1)/       5.0962820D0/
      DATA FN3P   (  0,1)/       1.5374650D0/
      DATA FN1P   (  0,2)/      -1.0603290D0/
      DATA FN2P   (  0,2)/       6.0037880D0/
      DATA FN3P   (  0,2)/       1.5701890D0/
      DATA VSP(0) /  -13.605  /
      DATA VPP(0)  /  0.0D00  /
!=========================================================================
!     Hydrogen parameters.
!=========================================================================
      DATA ALFA(1)  /  2.8823240D0/, ALPM(1)   /  2.5441341D0/
      DATA EISOL(1) /-11.3964270D0/, EISOLM(1) /-11.9062760D0/
      DATA BETAS(1) / -6.1737870D0/, BETASM(1) / -6.9890640D0/
      DATA ZS(1)    /  1.1880780D0/, ZSM(1)    /  1.3319670D0/
      DATA BDD(1,1)    /  0.4721793D0/, AMM(1)    /  0.4721793D0/
      DATA BDD(1,2)    /  0.4721793D0/, ADM(1)    /  0.4721793D0/
      DATA BDD(1,3)    /  0.4721793D0/, AQM(1)    /  0.4721793D0/
      DATA USS(1)   /-11.3964270D0/, USSM(1)   /-11.9062760D0/
      DATA GSS(1)   / 12.8480000D0/, GSSM(1)   / 12.8480000D0/
      DATA VS(1)    /-13.6050000D0/
      DATA VP(1)    /  0.0000000D0/
!
      DATA FN1(1,1) /  0.1227960D0/
      DATA FN2(1,1) /  5.0000000D0/
      DATA FN3(1,1) /  1.2000000D0/
      DATA FN1(1,2) /  0.0050900D0/
      DATA FN2(1,2) /  5.0000000D0/
      DATA FN3(1,2) /  1.8000000D0/
      DATA FN1(1,3) / -0.0183360D0/
      DATA FN2(1,3) /  2.0000000D0/
      DATA FN3(1,3) /  2.1000000D0/
!      DATA REFPM3 ( 1)/'  H: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1 10, 209 (1989).                '/
      DATA USSP   (  1)/     -13.0733210D0/
      DATA BETASP (  1)/      -5.6265120D0/
      DATA ZSP    (  1)/       0.9678070D0/
      DATA ALPP   (  1)/       3.3563860D0/
      DATA EISOLP (  1)/     -13.0733210D0/
      DATA GSSP   (  1)/      14.7942080D0/
      DATA AMP    (  1)/       0.5437048D0/
      DATA ADP    (  1)/       0.5437048D0/
      DATA AQP    (  1)/       0.5437048D0/
      DATA FN1P   (  1,1)/       1.1287500D0/
      DATA FN2P   (  1,1)/       5.0962820D0/
      DATA FN3P   (  1,1)/       1.5374650D0/
      DATA FN1P   (  1,2)/      -1.0603290D0/
      DATA FN2P   (  1,2)/       6.0037880D0/
      DATA FN3P   (  1,2)/       1.5701890D0/
      DATA VSP(1) /  -13.605  /
      DATA VPP(1)  /  0.0D00  /
!=========================================================================
!     Lithium parameters.
!=========================================================================
      DATA ALFA(3)  /  1.2501400D0/, ALPM(3)   /  1.2501400D0/
      DATA EISOL(3) / -5.1280000D0/, EISOLM(3) / -5.1280000D0/
      DATA BETAS(3) / -1.3500400D0/, BETASM(3) / -1.3500400D0/
      DATA BETAP(3) / -1.3500400D0/, BETAPM(3) / -1.3500400D0/
      DATA ZS(3)    /  0.7023800D0/, ZSM(3)    /  0.7023800D0/
      DATA ZP(3)    /  0.7023800D0/, ZPM(3)    /  0.7023800D0/
      DATA DD(3)    /  2.0549783D0/, DDM(3)    /  2.0549783D0/
      DATA QQ(3)    /  1.7437069D0/, QQM(3)    /  1.7437069D0/
      DATA BDD(3,1)    /  0.2682837D0/, AMM(3)    /  0.2682837D0/
      DATA BDD(3,2)    /  0.2269793D0/, ADM(3)    /  0.2269793D0/
      DATA BDD(3,3)    /  0.2614581D0/, AQM(3)    /  0.2614581D0/
      DATA USS(3)   / -5.1280000D0/, USSM(3)   / -5.1280000D0/
      DATA UPP(3)   / -2.7212000D0/, UPPM(3)   / -2.7212000D0/
      DATA GSS(3)   /  7.3000000D0/, GSSM(3)   /  7.3000000D0/
      DATA GSP(3)   /  5.4200000D0/, GSPM(3)   /  5.4200000D0/
      DATA GPP(3)   /  5.0000000D0/, GPPM(3)   /  5.0000000D0/
      DATA GP2(3)   /  4.5200000D0/, GP2M(3)   /  4.5200000D0/
      DATA HSP(3)   /  0.8300000D0/, HSPM(3)   /  0.8300000D0/
!      DATA REFPM3( 3)/ ' Li: (PM3): E. ANDERS, R. KOCH, P. FREUNSCHT,
!     1 J. COMP. CHEM 14 1301-1312 1993'/
      DATA USSP   (  3)/      -5.3000000D0/
      DATA UPPP   (  3)/      -3.4000000D0/
      DATA BETASP (  3)/      -0.5500000D0/
      DATA BETAPP (  3)/      -1.5000000D0/
      DATA ZSP    (  3)/       0.6500000D0/
      DATA ZPP    (  3)/       0.7500000D0/
      DATA ALPP   (  3)/       1.2550000D0/
      DATA GSSP   (  3)/       4.5000000D0/
      DATA GSPP   (  3)/       3.0000000D0/
      DATA GPPP   (  3)/       5.2500000D0/
      DATA GP2P   (  3)/       4.5000000D0/
      DATA HSPP   (  3)/       0.1500000D0/
      DATA FN1P   (  3,1)/      -0.4500000D0/
      DATA FN2P   (  3,1)/       5.0000000D0/
      DATA FN3P   (  3,1)/       1.0000000D0/
      DATA FN1P   (  3,2)/       0.8000000D0/
      DATA FN2P   (  3,2)/       6.5000000D0/
      DATA FN3P   (  3,2)/       1.0000000D0/
!=========================================================================
!     Beryllium parameters.
!=========================================================================
      DATA ALFA(4)  /  1.6694340D0/, ALPM(4)   /  1.6694340D0/
      DATA EISOL(4) /-24.2047560D0/, EISOLM(4) /-24.2047560D0/
      DATA BETAS(4) / -4.0170960D0/, BETASM(4) / -4.0170960D0/
      DATA BETAP(4) / -4.0170960D0/, BETAPM(4) / -4.0170960D0/
      DATA ZS(4)    /  1.0042100D0/, ZSM(4)    /  1.0042100D0/
      DATA ZP(4)    /  1.0042100D0/, ZPM(4)    /  1.0042100D0/
      DATA DD(4)    /  1.4373245D0/, DDM(4)    /  1.4373245D0/
      DATA QQ(4)    /  1.2196103D0/, QQM(4)    /  1.2196103D0/
      DATA BDD(4,1)    /  0.3307607D0/, AMM(4)    /  0.3307607D0/
      DATA BDD(4,2)    /  0.3356142D0/, ADM(4)    /  0.3356142D0/
      DATA BDD(4,3)    /  0.3846373D0/, AQM(4)    /  0.3846373D0/
      DATA USS(4)   /-16.6023780D0/, USSM(4)   /-16.6023780D0/
      DATA UPP(4)   /-10.7037710D0/, UPPM(4)   /-10.7037710D0/
      DATA GSS(4)   /  9.0000000D0/, GSSM(4)   /  9.0000000D0/
      DATA GSP(4)   /  7.4300000D0/, GSPM(4)   /  7.4300000D0/
      DATA GPP(4)   /  6.9700000D0/, GPPM(4)   /  6.9700000D0/
      DATA GP2(4)   /  6.2200000D0/, GP2M(4)   /  6.2200000D0/
      DATA HSP(4)   /  1.2800000D0/, HSPM(4)   /  1.2800000D0/
!      DATA REFPM3( 4)/ ' Be: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1(ACCEPTED)                      '/
      DATA USSP   (  4)/     -17.2647520D0/
      DATA UPPP   (  4)/     -11.3042430D0/
      DATA BETASP (  4)/      -3.9620530D0/
      DATA BETAPP (  4)/      -2.7806840D0/
      DATA ZSP    (  4)/       0.8774390D0/
      DATA ZPP    (  4)/       1.5087550D0/
      DATA ALPP   (  4)/       1.5935360D0/
      DATA EISOLP (  4)/     -25.5166530D0/
      DATA GSSP   (  4)/       9.0128510D0/
      DATA GSPP   (  4)/       6.5761990D0/
      DATA GPPP   (  4)/       6.0571820D0/
      DATA GP2P   (  4)/       9.0052190D0/
      DATA HSPP   (  4)/       0.5446790D0/
      DATA DDP    (  4)/       1.0090531D0/
      DATA QQP    (  4)/       0.8117586D0/
      DATA AMP    (  4)/       0.3312330D0/
      DATA ADP    (  4)/       0.2908996D0/
      DATA AQP    (  4)/       0.3530008D0/
      DATA FN1P   (  4,1)/       1.6315720D0/
      DATA FN2P   (  4,1)/       2.6729620D0/
      DATA FN3P   (  4,1)/       1.7916860D0/
      DATA FN1P   (  4,2)/      -2.1109590D0/
      DATA FN2P   (  4,2)/       1.9685940D0/
      DATA FN3P   (  4,2)/       1.7558710D0/
!=========================================================================
!     Boron parameters.
!=========================================================================
      DATA ALFA(5)  /  2.1349930D0/, ALPM(5)   /  2.1349930D0/
      DATA EISOL(5) /-64.3159500D0/, EISOLM(5) /-64.3159500D0/
      DATA BETAS(5) / -8.2520540D0/, BETASM(5) / -8.2520540D0/
      DATA BETAP(5) / -8.2520540D0/, BETAPM(5) / -8.2520540D0/
      DATA ZS(5)    /  1.5068010D0/, ZSM(5)    /  1.5068010D0/
      DATA ZP(5)    /  1.5068010D0/, ZPM(5)    /  1.5068010D0/
      DATA DD(5)    /  0.9579073D0/, DDM(5)    /  0.9579073D0/
      DATA QQ(5)    /  0.8128113D0/, QQM(5)    /  0.8128113D0/
      DATA BDD(5,1)    /  0.3891951D0/, AMM(5)    /  0.3891951D0/
      DATA BDD(5,2)    /  0.4904730D0/, ADM(5)    /  0.4904730D0/
      DATA BDD(5,3)    /  0.5556979D0/, AQM(5)    /  0.5556979D0/
      DATA USS(5)   /-34.5471300D0/, USSM(5)   /-34.5471300D0/
      DATA UPP(5)   /-23.1216900D0/, UPPM(5)   /-23.1216900D0/
      DATA GSS(5)   / 10.5900000D0/, GSSM(5)   / 10.5900000D0/
      DATA GSP(5)   /  9.5600000D0/, GSPM(5)   /  9.5600000D0/
      DATA GPP(5)   /  8.8600000D0/, GPPM(5)   /  8.8600000D0/
      DATA GP2(5)   /  7.8600000D0/, GP2M(5)   /  7.8600000D0/
      DATA HSP(5)   /  1.8100000D0/, HSPM(5)   /  1.8100000D0/
      DATA VS(5)    /-15.1600000D0/
      DATA VP(5)    / -8.5200000D0/
!=========================================================================
!     Carbon parameters.
!=========================================================================
      DATA ALFA(6)  /   2.6482740D0/, ALPM(6)   /   2.5463800D0/
      DATA EISOL(6) /-120.8157940D0/, EISOLM(6) /-120.5006060D0/
      DATA BETAS(6) / -15.7157830D0/, BETASM(6) / -18.9850440D0/
      DATA BETAP(6) /  -7.7192830D0/, BETAPM(6) /  -7.9341220D0/
      DATA ZS(6)    /   1.8086650D0/, ZSM(6)    /   1.7875370D0/
      DATA ZP(6)    /   1.6851160D0/, ZPM(6)    /   1.7875370D0/
      DATA DD(6)    /   0.8236736D0/, DDM(6)    /   0.8074662D0/
      DATA QQ(6)    /   0.7268015D0/, QQM(6)    /   0.6851578D0/
      DATA BDD(6,1)    /   0.4494671D0/, AMM(6)    /   0.4494671D0/
      DATA BDD(6,2)    /   0.6082946D0/, ADM(6)    /   0.6149474D0/
      DATA BDD(6,3)    /   0.6423492D0/, AQM(6)    /   0.6685897D0/
      DATA USS(6)   / -52.0286580D0/, USSM(6)   / -52.2797450D0/
      DATA UPP(6)   / -39.6142390D0/, UPPM(6)   / -39.2055580D0/
      DATA GSS(6)   /  12.2300000D0/, GSSM(6)   /  12.2300000D0/
      DATA GSP(6)   /  11.4700000D0/, GSPM(6)   /  11.4700000D0/
      DATA GPP(6)   /  11.0800000D0/, GPPM(6)   /  11.0800000D0/
      DATA GP2(6)   /   9.8400000D0/, GP2M(6)   /   9.8400000D0/
      DATA HSP(6)   /   2.4300000D0/, HSPM(6)   /   2.4300000D0/
      DATA VS(6)    / -21.3400000D0/
      DATA VP(6)    / -11.5400000D0/
!
      DATA FN1(6,1) /  0.0113550D0/
      DATA FN2(6,1) /  5.0000000D0/
      DATA FN3(6,1) /  1.6000000D0/
      DATA FN1(6,2) /  0.0459240D0/
      DATA FN2(6,2) /  5.0000000D0/
      DATA FN3(6,2) /  1.8500000D0/
      DATA FN1(6,3) / -0.0200610D0/
      DATA FN2(6,3) /  5.0000000D0/
      DATA FN3(6,3) /  2.0500000D0/
      DATA FN1(6,4) / -0.0012600D0/
      DATA FN2(6,4) /  5.0000000D0/
      DATA FN3(6,4) /  2.6500000D0/
!      DATA REFPM3 ( 6)/'  C: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1 10, 209 (1989).                '/
      DATA USSP   (  6)/     -47.2703200D0/
      DATA UPPP   (  6)/     -36.2669180D0/
      DATA BETASP (  6)/     -11.9100150D0/
      DATA BETAPP (  6)/      -9.8027550D0/
      DATA ZSP    (  6)/       1.5650850D0/
      DATA ZPP    (  6)/       1.8423450D0/
      DATA ALPP   (  6)/       2.7078070D0/
      DATA EISOLP (  6)/    -111.2299170D0/
      DATA GSSP   (  6)/      11.2007080D0/
      DATA GSPP   (  6)/      10.2650270D0/
      DATA GPPP   (  6)/      10.7962920D0/
      DATA GP2P   (  6)/       9.0425660D0/
      DATA HSPP   (  6)/       2.2909800D0/
      DATA DDP    (  6)/       0.8332396D0/
      DATA QQP    (  6)/       0.6647750D0/
      DATA AMP    (  6)/       0.4116394D0/
      DATA ADP    (  6)/       0.5885862D0/
      DATA AQP    (  6)/       0.7647667D0/
      DATA FN1P   (  6,1)/       0.0501070D0/
      DATA FN2P   (  6,1)/       6.0031650D0/
      DATA FN3P   (  6,1)/       1.6422140D0/
      DATA FN1P   (  6,2)/       0.0507330D0/
      DATA FN2P   (  6,2)/       6.0029790D0/
      DATA FN3P   (  6,2)/       0.8924880D0/

      DATA VSP(6)/-21.34D00/
      DATA VPP(6)/-11.54D00/
!=========================================================================
!     Nitrogen parameters.
!=========================================================================
      DATA ALFA(7)  /   2.9472860D0/, ALPM(7)   /   2.8613420D0/
      DATA EISOL(7) /-202.4077430D0/, EISOLM(7) /-202.5812010D0/
      DATA BETAS(7) / -20.2991100D0/, BETASM(7) / -20.4957580D0/
      DATA BETAP(7) / -18.2386660D0/, BETAPM(7) / -20.4957580D0/
      DATA ZS(7)    /   2.3154100D0/, ZSM(7)    /   2.2556140D0/
      DATA ZP(7)    /   2.1579400D0/, ZPM(7)    /   2.2556140D0/
      DATA DD(7)    /   0.6433247D0/, DDM(7)    /   0.6399037D0/
      DATA QQ(7)    /   0.5675528D0/, QQM(7)    /   0.5429763D0/
      DATA BDD(7,1)    /   0.4994487D0/, AMM(7)    /   0.4994487D0/
      DATA BDD(7,2)    /   0.7820840D0/, ADM(7)    /   0.7843643D0/
      DATA BDD(7,3)    /   0.7883498D0/, AQM(7)    /   0.8144720D0/
      DATA USS(7)   / -71.8600000D0/, USSM(7)   / -71.9321220D0/
      DATA UPP(7)   / -57.1675810D0/, UPPM(7)   / -57.1723190D0/
      DATA GSS(7)   /  13.5900000D0/, GSSM(7)   /  13.5900000D0/
      DATA GSP(7)   /  12.6600000D0/, GSPM(7)   /  12.6600000D0/
      DATA GPP(7)   /  12.9800000D0/, GPPM(7)   /  12.9800000D0/
      DATA GP2(7)   /  11.5900000D0/, GP2M(7)   /  11.5900000D0/
      DATA HSP(7)   /   3.1400000D0/, HSPM(7)   /   3.1400000D0/
      DATA VS(7)    / -27.5100000D0/
      DATA VP(7)    / -14.3400000D0/
!
      DATA FN1(7,1) /   0.0252510D0/
      DATA FN2(7,1) /   5.00000D0/
      DATA FN3(7,1) /   1.5000000D0/
      DATA FN1(7,2) /   0.0289530D0/
      DATA FN2(7,2) /   5.0000000D0/
      DATA FN3(7,2) /   2.1000000D0/
      DATA FN1(7,3) /  -0.0058060D0/
      DATA FN2(7,3) /   2.0000000D0/
      DATA FN3(7,3) /   2.4000000D0/
!      DATA REFPM3 ( 7)/'  N: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1 10, 209 (1989).                '/
      DATA USSP   (  7)/     -49.3356720D0/
      DATA UPPP   (  7)/     -47.5097360D0/
      DATA BETASP (  7)/     -14.0625210D0/
      DATA BETAPP (  7)/     -20.0438480D0/
      DATA ZSP    (  7)/       2.0280940D0/
      DATA ZPP    (  7)/       2.3137280D0/
      DATA ALPP   (  7)/       2.8305450D0/
      DATA EISOLP (  7)/    -157.6137755D0/
      DATA GSSP   (  7)/      11.9047870D0/
      DATA GSPP   (  7)/       7.3485650D0/
      DATA GPPP   (  7)/      11.7546720D0/
      DATA GP2P   (  7)/      10.8072770D0/
      DATA HSPP   (  7)/       1.1367130D0/
      DATA DDP    (  7)/       0.6577006D0/
      DATA QQP    (  7)/       0.5293383D0/
      DATA AMP    (  7)/       0.4375151D0/
      DATA ADP    (  7)/       0.5030995D0/
      DATA AQP    (  7)/       0.7364933D0/
      DATA FN1P   (  7,1)/       1.5016740D0/
      DATA FN2P   (  7,1)/       5.9011480D0/
      DATA FN3P   (  7,1)/       1.7107400D0/
      DATA FN1P   (  7,2)/      -1.5057720D0/
      DATA FN2P   (  7,2)/       6.0046580D0/
      DATA FN3P   (  7,2)/       1.7161490D0/
      DATA VSP(7)/-27.51D00/
      DATA VPP(7)/-14.34D00/
!=========================================================================
!     Oxygen parameters.
!=========================================================================
      DATA ALFA(8)  /   4.4553710D0/, ALPM(8)   /   3.1606040D0/
      DATA EISOL(8) /-316.0995200D0/, EISOLM(8) /-317.8685060D0/
      DATA BETAS(8) / -29.2727730D0/, BETASM(8) / -32.6880820D0/
      DATA BETAP(8) / -29.2727730D0/, BETAPM(8) / -32.6880820D0/
      DATA ZS(8)    /   3.1080320D0/, ZSM(8)    /   2.6999050D0/
      DATA ZP(8)    /   2.5240390D0/, ZPM(8)    /   2.6999050D0/
      DATA DD(8)    /   0.4988896D0/, DDM(8)    /   0.5346024D0/
      DATA QQ(8)    /   0.4852322D0/, QQM(8)    /   0.4536252D0/
      DATA BDD(8,1)    /   0.5667034D0/, AMM(8)    /   0.5667034D0/
      DATA BDD(8,2)    /   0.9961066D0/, ADM(8)    /   0.9592562D0/
      DATA BDD(8,3)    /   0.9065223D0/, AQM(8)    /   0.9495934D0/
      DATA USS(8)   / -97.8300000D0/, USSM(8)   / -99.6443090D0/
      DATA UPP(8)   / -78.2623800D0/, UPPM(8)   / -77.7974720D0/
      DATA GSS(8)   /  15.4200000D0/, GSSM(8)   /  15.4200000D0/
      DATA GSP(8)   /  14.4800000D0/, GSPM(8)   /  14.4800000D0/
      DATA GPP(8)   /  14.5200000D0/, GPPM(8)   /  14.5200000D0/
      DATA GP2(8)   /  12.9800000D0/, GP2M(8)   /  12.9800000D0/
      DATA HSP(8)   /   3.9400000D0/, HSPM(8)   /   3.9400000D0/
      DATA VS(8)    / -35.3000000D0/
      DATA VP(8)    / -17.9100000D0/
!
      DATA FN1(8,1) /   0.2809620D0/
      DATA FN2(8,1) /   5.0000000D0/
      DATA FN3(8,1) /   0.8479180D0/
      DATA FN1(8,2) /   0.0814300D0/
      DATA FN2(8,2) /   7.0000000D0/
      DATA FN3(8,2) /   1.4450710D0/
!      DATA REFPM3 ( 8)/'  O: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1 10, 209 (1989).                '/
      DATA USSP   (  8)/     -86.9930020D0/
      DATA UPPP   (  8)/     -71.8795800D0/
      DATA BETASP (  8)/     -45.2026510D0/
      DATA BETAPP (  8)/     -24.7525150D0/
      DATA ZSP    (  8)/       3.7965440D0/
      DATA ZPP    (  8)/       2.3894020D0/
      DATA ALPP   (  8)/       3.2171020D0/
      DATA EISOLP (  8)/    -289.3422065D0/
      DATA GSSP   (  8)/      15.7557600D0/
      DATA GSPP   (  8)/      10.6211600D0/
      DATA GPPP   (  8)/      13.6540160D0/
      DATA GP2P   (  8)/      12.4060950D0/
      DATA HSPP   (  8)/       0.5938830D0/
      DATA DDP    (  8)/       0.4086173D0/
      DATA QQP    (  8)/       0.5125738D0/
      DATA AMP    (  8)/       0.5790430D0/
      DATA ADP    (  8)/       0.5299630D0/
      DATA AQP    (  8)/       0.8179630D0/
      DATA FN1P   (  8,1)/      -1.1311280D0/
      DATA FN2P   (  8,1)/       6.0024770D0/
      DATA FN3P   (  8,1)/       1.6073110D0/
      DATA FN1P   (  8,2)/       1.1378910D0/
      DATA FN2P   (  8,2)/       5.9505120D0/
      DATA FN3P   (  8,2)/       1.5983950D0/
      DATA VSP(8)/-35.30D00/
      DATA VPP(8)/-17.91D00/
!=========================================================================
!     Fluorine parameters.
!=========================================================================
      DATA ALFA(9)  /   5.5178000D0/, ALPM(9)   /   3.4196606D0/
      DATA EISOL(9) /-482.2905830D0/, EISOLM(9) /-476.6837810D0/
      DATA BETAS(9) / -69.5902770D0/, BETASM(9) / -48.2904660D0/
      DATA BETAP(9) / -27.9223600D0/, BETAPM(9) / -36.5085400D0/
      DATA ZS(9)    /   3.7700820D0/, ZSM(9)    /   2.8484870D0/
      DATA ZP(9)    /   2.4946700D0/, ZPM(9)    /   2.8484870D0/
      DATA DD(9)    /   0.4145203D0/, DDM(9)    /   0.5067166D0/
      DATA QQ(9)    /   0.4909446D0/, QQM(9)    /   0.4299633D0/
      DATA BDD(9,1)    /   0.6218302D0/, AMM(9)    /   0.6218302D0/
      DATA BDD(9,2)    /   1.2088792D0/, ADM(9)    /   1.0850301D0/
      DATA BDD(9,3)    /   0.9449355D0/, AQM(9)    /   1.0343643D0/
      DATA USS(9)   /-136.1055790D0/, USSM(9)   /-131.0715480D0/
      DATA UPP(9)   /-104.8898850D0/, UPPM(9)   /-105.7821370D0/
      DATA GSS(9)   /  16.9200000D0/, GSSM(9)   /  16.9200000D0/
      DATA GSP(9)   /  17.2500000D0/, GSPM(9)   /  17.2500000D0/
      DATA GPP(9)   /  16.7100000D0/, GPPM(9)   /  16.7100000D0/
      DATA GP2(9)   /  14.9100000D0/, GP2M(9)   /  14.9100000D0/
      DATA HSP(9)   /   4.8300000D0/, HSPM(9)   /   4.8300000D0/
      DATA VS(9)    / -43.7000000D0/
      DATA VP(9)    / -20.8900000D0/
!
      DATA FN1(9,1) /   0.2420790D0/
      DATA FN2(9,1) /   4.8000000D0/
      DATA FN3(9,1) /   0.9300000D0/
      DATA FN1(9,2) /   0.0036070D0/
      DATA FN2(9,2) /   4.6000000D0/
      DATA FN3(9,2) /   1.6600000D0/
!      DATA REFPM3 ( 9)/'  F: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1 10, 209 (1989).                '/
      DATA USSP   (  9)/    -110.4353030D0/
      DATA UPPP   (  9)/    -105.6850470D0/
      DATA BETASP (  9)/     -48.4059390D0/
      DATA BETAPP (  9)/     -27.7446600D0/
      DATA ZSP    (  9)/       4.7085550D0/
      DATA ZPP    (  9)/       2.4911780D0/
      DATA ALPP   (  9)/       3.3589210D0/
      DATA EISOLP (  9)/    -437.5171690D0/
      DATA GSSP   (  9)/      10.4966670D0/
      DATA GSPP   (  9)/      16.0736890D0/
      DATA GPPP   (  9)/      14.8172560D0/
      DATA GP2P   (  9)/      14.4183930D0/
      DATA HSPP   (  9)/       0.7277630D0/
      DATA DDP    (  9)/       0.3125302D0/
      DATA QQP    (  9)/       0.4916328D0/
      DATA AMP    (  9)/       0.3857650D0/
      DATA ADP    (  9)/       0.6768503D0/
      DATA AQP    (  9)/       0.6120047D0/
      DATA FN1P   (  9,1)/      -0.0121660D0/
      DATA FN2P   (  9,1)/       6.0235740D0/
      DATA FN3P   (  9,1)/       1.8568590D0/
      DATA FN1P   (  9,2)/      -0.0028520D0/
      DATA FN2P   (  9,2)/       6.0037170D0/
      DATA FN3P   (  9,2)/       2.6361580D0/
      DATA VSP(9)/-43.70D00/
      DATA VPP(9)/-20.89D00/
!=========================================================================
!     Sodium parameters.
!=========================================================================
      DATA ALFA(11)  / 1.32D0/, ALPM(11)   / 1.32D0/
      DATA EISOL(11) / 0.00D0/, EISOLM(11) / 0.00D0/
      DATA BDD(11,1)    / 0.50D0/, AMM(11)    / 0.50D0/
      DATA VS(11)    /10.00D0/
!=========================================================================
!     Magnesium parameters.
!=========================================================================
!     DATA REFPM3(12)/ ' Mg: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1(ACCEPTED)                      '/
      DATA USSP   ( 12)/     -14.6236880D0/
      DATA UPPP   ( 12)/     -14.1734600D0/
      DATA BETASP ( 12)/      -2.0716910D0/
      DATA BETAPP ( 12)/      -0.5695810D0/
      DATA ZSP    ( 12)/       0.6985520D0/
      DATA ZPP    ( 12)/       1.4834530D0/
      DATA ALPP   ( 12)/       1.3291470D0/
      DATA EISOLP ( 12)/     -22.5530760D0/
      DATA GSSP   ( 12)/       6.6943000D0/
      DATA GSPP   ( 12)/       6.7939950D0/
      DATA GPPP   ( 12)/       6.9104460D0/
      DATA GP2P   ( 12)/       7.0908230D0/
      DATA HSPP   ( 12)/       0.5433000D0/
      DATA DDP    ( 12)/       1.1403950D0/
      DATA QQP    ( 12)/       1.1279899D0/
      DATA AMP    ( 12)/       0.2460235D0/
      DATA ADP    ( 12)/       0.2695751D0/
      DATA AQP    ( 12)/       0.2767522D0/
      DATA FN1P   ( 12,1)/       2.1170500D0/
      DATA FN2P   ( 12,1)/       6.0094770D0/
      DATA FN3P   ( 12,1)/       2.0844060D0/
      DATA FN1P   ( 12,2)/      -2.5477670D0/
      DATA FN2P   ( 12,2)/       4.3953700D0/
      DATA FN3P   ( 12,2)/       2.0636740D0/
!=========================================================================
!     Aluminium parameters.
!=========================================================================
      DATA ALFA(13)  /  1.8688394D0/, ALPM(13)   /  1.8688394D0/
      DATA EISOL(13) /-44.4840720D0/, EISOLM(13) /-44.4840711D0/
      DATA BETAS(13) / -2.6702840D0/, BETASM(13) / -2.6702840D0/
      DATA BETAP(13) / -2.6702840D0/, BETAPM(13) / -2.6702840D0/
      DATA ZS(13)    /  1.4441610D0/, ZSM(13)    /  1.4441610D0/
      DATA ZP(13)    /  1.4441610D0/, ZPM(13)    /  1.4441610D0/
      DATA ZD(13)    /  1.0000000D0/, ZDM(13)    /  1.0000000D0/
      DATA DD(13)    /  1.3992387D0/, DDM(13)    /  1.3992387D0/
      DATA QQ(13)    /  1.1586797D0/, QQM(13)    /  1.1586797D0/
      DATA BDD(13,1)    /  0.2973172D0/, AMM(13)    /  0.2973172D0/
      DATA BDD(13,2)    /  0.2635574D0/, ADM(13)    /  0.2635574D0/
      DATA BDD(13,3)    /  0.3673560D0/, AQM(13)    /  0.3673560D0/
      DATA USS(13)   /-23.8070970D0/, USSM(13)   /-23.8070970D0/
      DATA UPP(13)   /-17.5198780D0/, UPPM(13)   /-17.5198780D0/
      DATA GSS(13)   /  8.0900000D0/, GSSM(13)   /  8.0900000D0/
      DATA GSP(13)   /  6.6300000D0/, GSPM(13)   /  6.6300000D0/
      DATA GPP(13)   /  5.9800000D0/, GPPM(13)   /  5.9800000D0/
      DATA GP2(13)   /  5.4000000D0/, GP2M(13)   /  5.4000000D0/
      DATA HSP(13)   /  0.7000000D0/, HSPM(13)   /  0.7000000D0/
!      DATA REFPM3 (13)/' Al: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1 10, 209 (1989).                '/
      DATA USSP   ( 13)/     -24.8454040D0/
      DATA UPPP   ( 13)/     -22.2641590D0/
      DATA BETASP ( 13)/      -0.5943010D0/
      DATA BETAPP ( 13)/      -0.9565500D0/
      DATA ZSP    ( 13)/       1.7028880D0/
      DATA ZPP    ( 13)/       1.0736290D0/
      DATA ZDP    ( 13)/       1.0000000D0/
      DATA ALPP   ( 13)/       1.5217030D0/
      DATA EISOLP ( 13)/     -46.8647630D0/
      DATA GSSP   ( 13)/       5.7767370D0/
      DATA GSPP   ( 13)/      11.6598560D0/
      DATA GPPP   ( 13)/       6.3477900D0/
      DATA GP2P   ( 13)/       6.1210770D0/
      DATA HSPP   ( 13)/       4.0062450D0/
      DATA DDP    ( 13)/       1.2102799D0/
      DATA QQP    ( 13)/       1.5585645D0/
      DATA AMP    ( 13)/       0.2123020D0/
      DATA ADP    ( 13)/       0.6418584D0/
      DATA AQP    ( 13)/       0.2262838D0/
      DATA FN1P   ( 13,1)/      -0.4730900D0/
      DATA FN2P   ( 13,1)/       1.9158250D0/
      DATA FN3P   ( 13,1)/       1.4517280D0/
      DATA FN1P   ( 13,2)/      -0.1540510D0/
      DATA FN2P   ( 13,2)/       6.0050860D0/
      DATA FN3P   ( 13,2)/       2.5199970D0/
!=========================================================================
!     Silicon parameters.
!=========================================================================
      DATA ALFA(14)  /  2.1961078D0/, ALPM(14)   /  2.2053160D0/
      DATA EISOL(14) /-90.5399580D0/, EISOLM(14) /-82.8394220D0/
      DATA BETAS(14) / -4.2562180D0/, BETASM(14) / -9.0868040D0/
      DATA BETAP(14) / -4.2562180D0/, BETAPM(14) / -1.0758270D0/
      DATA ZS(14)    /  1.4353060D0/, ZSM(14)    /  1.3159860D0/
      DATA ZP(14)    /  1.4353060D0/, ZPM(14)    /  1.7099430D0/
      DATA ZD(14)    /  1.0000000D0/, ZDM(14)    /  1.0000000D0/
      DATA DD(14)    /  1.4078712D0/, DDM(14)    /  1.2580349D0/
      DATA QQ(14)    /  1.1658281D0/, QQM(14)    /  0.9785824D0/
      DATA BDD(14,1)    /  0.3608967D0/, AMM(14)    /  0.3608967D0/
      DATA BDD(14,2)    /  0.3441817D0/, ADM(14)    /  0.3664244D0/
      DATA BDD(14,3)    /  0.3999442D0/, AQM(14)    /  0.4506740D0/
      DATA USS(14)   /-40.5682920D0/, USSM(14)   /-37.0375330D0/
      DATA UPP(14)   /-28.0891870D0/, UPPM(14)   /-27.7696780D0/
      DATA GSS(14)   /  9.8200000D0/, GSSM(14)   /  9.8200000D0/
      DATA GSP(14)   /  8.3600000D0/, GSPM(14)   /  8.3600000D0/
      DATA GPP(14)   /  7.3100000D0/, GPPM(14)   /  7.3100000D0/
      DATA GP2(14)   /  6.5400000D0/, GP2M(14)   /  6.5400000D0/
      DATA HSP(14)   /  1.3200000D0/, HSPM(14)   /  1.3200000D0/
      DATA VS(14)    /-17.8200000D0/
      DATA VP(14)    / -8.5100000D0/
!      DATA REFPM3 (14)/' Si: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1 10, 209 (1989).                '/
      DATA USSP   ( 14)/     -26.7634830D0/
      DATA UPPP   ( 14)/     -22.8136350D0/
      DATA BETASP ( 14)/      -2.8621450D0/
      DATA BETAPP ( 14)/      -3.9331480D0/
      DATA ZSP    ( 14)/       1.6350750D0/
      DATA ZPP    ( 14)/       1.3130880D0/
      DATA ZDP    ( 14)/       1.0000000D0/
      DATA ALPP   ( 14)/       2.1358090D0/
      DATA EISOLP ( 14)/     -67.7882140D0/
      DATA GSSP   ( 14)/       5.0471960D0/
      DATA GSPP   ( 14)/       5.9490570D0/
      DATA GPPP   ( 14)/       6.7593670D0/
      DATA GP2P   ( 14)/       5.1612970D0/
      DATA HSPP   ( 14)/       0.9198320D0/
      DATA DDP    ( 14)/       1.3144550D0/
      DATA QQP    ( 14)/       1.2743396D0/
      DATA AMP    ( 14)/       0.1854905D0/
      DATA ADP    ( 14)/       0.3060715D0/
      DATA AQP    ( 14)/       0.4877432D0/
      DATA FN1P   ( 14,1)/      -0.3906000D0/
      DATA FN2P   ( 14,1)/       6.0000540D0/
      DATA FN3P   ( 14,1)/       0.6322620D0/
      DATA FN1P   ( 14,2)/       0.0572590D0/
      DATA FN2P   ( 14,2)/       6.0071830D0/
      DATA FN3P   ( 14,2)/       2.0199870D0/
      DATA VSP(14)/-17.82D00/
      DATA VPP(14)/-8.51D00/
!=========================================================================
!     Phosphorus parameters.
!=========================================================================
      DATA ALFA(15)  /   2.4553220D0/, ALPM(15)   /   2.4152800D0/
      DATA EISOL(15) /-124.4368355D0/, EISOLM(15) /-152.9599600D0/
      DATA BETAS(15) /  -6.3537640D0/, BETASM(15) /  -6.7916000D0/
      DATA BETAP(15) /  -6.5907090D0/, BETAPM(15) /  -6.7916000D0/
      DATA ZS(15)    /   1.9812800D0/, ZSM(15)    /   2.1087200D0/
      DATA ZP(15)    /   1.8751500D0/, ZPM(15)    /   1.7858100D0/
      DATA ZD(15)    /   1.0000000D0/, ZDM(15)    /   1.0000000D0/
      DATA DD(15)    /   1.0452022D0/, DDM(15)    /   1.0129699D0/
      DATA QQ(15)    /   0.8923660D0/, QQM(15)    /   0.9370090D0/
      DATA BDD(15,1)    /   0.4248440D0/, AMM(15)    /   0.4248438D0/
      DATA BDD(15,2)    /   0.3275319D0/, ADM(15)    /   0.4882420D0/
      DATA BDD(15,3)    /   0.4386854D0/, AQM(15)    /   0.4979406D0/
      DATA USS(15)   / -42.0298630D0/, USSM(15)   / -56.1433600D0/
      DATA UPP(15)   / -34.0307090D0/, UPPM(15)   / -42.8510800D0/
      DATA GSS(15)   /  11.5600050D0/, GSSM(15)   /  11.5600000D0/
      DATA GSP(15)   /   5.2374490D0/, GSPM(15)   /  10.0800000D0/
      DATA GPP(15)   /   7.8775890D0/, GPPM(15)   /   8.6400000D0/
      DATA GP2(15)   /   7.3076480D0/, GP2M(15)   /   7.6800000D0/
      DATA HSP(15)   /   0.7792380D0/, HSPM(15)   /   1.9200000D0/
      DATA VS(15)    / -21.1000000D0/
      DATA VP(15)    / -10.2900000D0/

      DATA FN1( 15,1)/      -0.0318270D0/
      DATA FN2( 15,1)/       6.0000000D0/
      DATA FN3( 15,1)/       1.4743230D0/
      DATA FN1( 15,2)/       0.0184700D0/
      DATA FN2( 15,2)/       7.0000000D0/
      DATA FN3( 15,2)/       1.7793540D0/
      DATA FN1( 15,3)/       0.0332900D0/
      DATA FN2( 15,3)/       9.0000000D0/
      DATA FN3( 15,3)/       3.0065760D0/                               
!      DATA REFPM3 (15)/'  P: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1 10, 209 (1989).                '/
      DATA USSP   ( 15)/     -40.4130960D0/
      DATA UPPP   ( 15)/     -29.5930520D0/
      DATA BETASP ( 15)/     -12.6158790D0/
      DATA BETAPP ( 15)/      -4.1600400D0/
      DATA ZSP    ( 15)/       2.0175630D0/
      DATA ZPP    ( 15)/       1.5047320D0/
      DATA ZDP    ( 15)/       1.0000000D0/
      DATA ALPP   ( 15)/       1.9405340D0/
      DATA EISOLP ( 15)/    -117.9591740D0/
      DATA GSSP   ( 15)/       7.8016150D0/
      DATA GSPP   ( 15)/       5.1869490D0/
      DATA GPPP   ( 15)/       6.6184780D0/
      DATA GP2P   ( 15)/       6.0620020D0/
      DATA HSPP   ( 15)/       1.5428090D0/
      DATA DDP    ( 15)/       1.0644947D0/
      DATA QQP    ( 15)/       1.1120386D0/
      DATA AMP    ( 15)/       0.2867187D0/
      DATA ADP    ( 15)/       0.4309446D0/
      DATA AQP    ( 15)/       0.3732517D0/
      DATA FN1P   ( 15,1)/      -0.6114210D0/
      DATA FN2P   ( 15,1)/       1.9972720D0/
      DATA FN3P   ( 15,1)/       0.7946240D0/
      DATA FN1P   ( 15,2)/      -0.0939350D0/
      DATA FN2P   ( 15,2)/       1.9983600D0/
      DATA FN3P   ( 15,2)/       1.9106770D0/
      DATA VSP(15)/-21.10D00/
      DATA VPP(15)/-10.29D00/
!=========================================================================
!     Sulphur parameters.
!=========================================================================
! INCORPORATED FROM MOPAC-version 6.0
      DATA ALFA(16)  /   2.4616480D0/, ALPM(16)   /   2.4780260D0/
      DATA EISOL(16) /-191.7321930D0/, EISOLM(16) /-226.0123900D0/
      DATA BETAS(16) /  -3.9205660D0/, BETASM(16) / -10.7616700D0/
      DATA BETAP(16) /  -7.9052780D0/, BETAPM(16) / -10.1084330D0/
      DATA ZS(16)    /   2.3665150D0/, ZSM(16)    /   2.3129620D0/
      DATA ZP(16)    /   1.6672630D0/, ZPM(16)    /   2.0091460D0/
      DATA ZD(16)    /   1.0000000D0/, ZDM(16)    /   1.0000000D0/
      DATA DD(16)    /   0.9004265D0/, DDM(16)    /   0.9189935D0/
      DATA QQ(16)    /   1.0036329D0/, QQM(16)    /   0.8328514D0/
      DATA BDD(16,1)    /   0.4331617D0/, AMM(16)    /   0.4733554D0/
      DATA BDD(16,2)    /   0.5907115D0/, ADM(16)    /   0.5544502D0/
      DATA BDD(16,3)    /   0.6454943D0/, AQM(16)    /   0.5585244D0/
      DATA USS(16)   / -56.6940560D0/, USSM(16)   / -72.2422810D0/
      DATA UPP(16)   / -48.7170490D0/, UPPM(16)   / -56.9732070D0/
      DATA GSS(16)   /  11.7863290D0/, GSSM(16)   /  12.8800000D0/
      DATA GSP(16)   /   8.6631270D0/, GSPM(16)   /  11.2600000D0/
      DATA GPP(16)   /  10.0393080D0/, GPPM(16)   /   9.9000000D0/
      DATA GP2(16)   /   7.7816880D0/, GP2M(16)   /   8.8300000D0/
      DATA HSP(16)   /   2.5321370D0/, HSPM(16)   /   2.2600000D0/
      DATA VS(16)    / -23.8400000D0/
      DATA VP(16)    / -12.4100000D0/
      DATA FN1( 16,1)/  -0.5091950D0/
      DATA FN2( 16,1)/   4.5936910D0/
      DATA FN3( 16,1)/   0.7706650D0/
      DATA FN1( 16,2)/  -0.0118630D0/
      DATA FN2( 16,2)/   5.8657310D0/
      DATA FN3( 16,2)/   1.5033130D0/
      DATA FN1( 16,3)/   0.0123340D0/
      DATA FN2( 16,3)/  13.5573360D0/
      DATA FN3( 16,3)/   2.0091730D0/
!
!
!      DATA ALFA(16)  /   2.4916445D0/, ALPM(16)   /   2.4780260D0/
!      DATA EISOL(16) /-235.4413560D0/, EISOLM(16) /-226.0123900D0/
!      DATA BETAS(16) / -11.1422310D0/, BETASM(16) / -10.7616700D0/
!      DATA BETAP(16) / -11.1422310D0/, BETAPM(16) / -10.1084330D0/
!      DATA ZS(16)    /   2.6135910D0/, ZSM(16)    /   2.3129620D0/
!      DATA ZP(16)    /   2.0343930D0/, ZPM(16)    /   2.0091460D0/
!      DATA ZD(16)    /   1.0000000D0/, ZDM(16)    /   1.0000000D0/
!      DATA DD(16)    /   0.8231596D0/, DDM(16)    /   0.9189935D0/
!      DATA QQ(16)    /   0.8225156D0/, QQM(16)    /   0.8328514D0/
!      DATA BDD(16,1)    /   0.4733554D0/, AMM(16)    /   0.4733554D0/
!      DATA BDD(16,2)    /   0.5889395D0/, ADM(16)    /   0.5544502D0/
!      DATA BDD(16,3)    /   0.5632724D0/, AQM(16)    /   0.5585244D0/
!      DATA USS(16)   / -75.2391520D0/, USSM(16)   / -72.2422810D0/
!      DATA UPP(16)   / -57.8320130D0/, UPPM(16)   / -56.9732070D0/
!      DATA GSS(16)   /  12.8800000D0/, GSSM(16)   /  12.8800000D0/
!      DATA GSP(16)   /  11.2600000D0/, GSPM(16)   /  11.2600000D0/
!      DATA GPP(16)   /   9.9000000D0/, GPPM(16)   /   9.9000000D0/
!      DATA GP2(16)   /   8.8300000D0/, GP2M(16)   /   8.8300000D0/
!      DATA HSP(16)   /   2.2600000D0/, HSPM(16)   /   2.2600000D0/
!      DATA VS(16)    / -23.8400000D0/
!      DATA VP(16)    / -12.4100000D0/
!
!      DATA REFPM3 (16)/'  S: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1 10, 209 (1989).                '/
      DATA USSP   ( 16)/     -49.8953710D0/
      DATA UPPP   ( 16)/     -44.3925830D0/
      DATA BETASP ( 16)/      -8.8274650D0/
      DATA BETAPP ( 16)/      -8.0914150D0/
      DATA ZSP    ( 16)/       1.8911850D0/
      DATA ZPP    ( 16)/       1.6589720D0/
      DATA ZDP    ( 16)/       1.0000000D0/
      DATA ALPP   ( 16)/       2.2697060D0/
      DATA EISOLP ( 16)/    -183.4537395D0/
      DATA GSSP   ( 16)/       8.9646670D0/
      DATA GSPP   ( 16)/       6.7859360D0/
      DATA GPPP   ( 16)/       9.9681640D0/
      DATA GP2P   ( 16)/       7.9702470D0/
      DATA HSPP   ( 16)/       4.0418360D0/
      DATA DDP    ( 16)/       1.1214313D0/
      DATA QQP    ( 16)/       1.0086488D0/
      DATA AMP    ( 16)/       0.3294622D0/
      DATA ADP    ( 16)/       0.6679118D0/
      DATA AQP    ( 16)/       0.6137472D0/
      DATA FN1P   ( 16,1)/      -0.3991910D0/
      DATA FN2P   ( 16,1)/       6.0006690D0/
      DATA FN3P   ( 16,1)/       0.9621230D0/
      DATA FN1P   ( 16,2)/      -0.0548990D0/
      DATA FN2P   ( 16,2)/       6.0018450D0/
      DATA FN3P   ( 16,2)/       1.5799440D0/
      DATA VSP(16)/-23.84D00/
      DATA VPP(16)/-12.41D00/
!=========================================================================
!     Chlorine parameters.
!=========================================================================
      DATA ALFA(17)  /   2.9193680D0/, ALPM(17)   /   2.5422010D0/
      DATA EISOL(17) /-372.1984310D0/, EISOLM(17) /-353.1176670D0/
      DATA BETAS(17) / -24.5946700D0/, BETASM(17) / -14.2623200D0/
      DATA BETAP(17) / -14.6372160D0/, BETAPM(17) / -14.2623200D0/
      DATA ZS(17)    /   3.6313760D0/, ZSM(17)    /   3.7846450D0/
      DATA ZP(17)    /   2.0767990D0/, ZPM(17)    /   2.0362630D0/
      DATA ZD(17)    /   1.0000000D0/, ZDM(17)    /   1.0000000D0/
      DATA DD(17)    /   0.5406286D0/, DDM(17)    /   0.4986870D0/
      DATA QQ(17)    /   0.8057208D0/, QQM(17)    /   0.8217603D0/
      DATA BDD(17,1)    /   0.5523705D0/, AMM(17)    /   0.5523705D0/
      DATA BDD(17,2)    /   0.7693200D0/, ADM(17)    /   0.8061220D0/
      DATA BDD(17,3)    /   0.6133369D0/, AQM(17)    /   0.6053435D0/
      DATA USS(17)   /-111.6139480D0/, USSM(17)   /-100.2271660D0/
      DATA UPP(17)   / -76.6401070D0/, UPPM(17)   / -77.3786670D0/
      DATA GSS(17)   /  15.0300000D0/, GSSM(17)   /  15.0300000D0/
      DATA GSP(17)   /  13.1600000D0/, GSPM(17)   /  13.1600000D0/
      DATA GPP(17)   /  11.3000000D0/, GPPM(17)   /  11.3000000D0/
      DATA GP2(17)   /   9.9700000D0/, GP2M(17)   /   9.9700000D0/
      DATA HSP(17)   /   2.4200000D0/, HSPM(17)   /   2.4200000D0/
      DATA VS(17)    / -25.2600000D0/
      DATA VP(17)    / -15.0900000D0/
!
      DATA FN1(17,1) /   0.0942430D0/
      DATA FN2(17,1) /   4.0000000D0/
      DATA FN3(17,1) /   1.3000000D0/
      DATA FN1(17,2) /   0.0271680D0/
      DATA FN2(17,2) /   4.0000000D0/
      DATA FN3(17,2) /   2.1000000D0/
!      DATA REFPM3 (17)/' Cl: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1 10, 209 (1989).                '/
      DATA USSP   ( 17)/    -100.6267470D0/
      DATA UPPP   ( 17)/     -53.6143960D0/
      DATA BETASP ( 17)/     -27.5285600D0/
      DATA BETAPP ( 17)/     -11.5939220D0/
      DATA ZSP    ( 17)/       2.2462100D0/
      DATA ZPP    ( 17)/       2.1510100D0/
      DATA ZDP    ( 17)/       1.0000000D0/
      DATA ALPP   ( 17)/       2.5172960D0/
      DATA EISOLP ( 17)/    -315.1949480D0/
      DATA GSSP   ( 17)/      16.0136010D0/
      DATA GSPP   ( 17)/       8.0481150D0/
      DATA GPPP   ( 17)/       7.5222150D0/
      DATA GP2P   ( 17)/       7.5041540D0/
      DATA HSPP   ( 17)/       3.4811530D0/
      DATA DDP    ( 17)/       0.9175856D0/
      DATA QQP    ( 17)/       0.7779230D0/
      DATA AMP    ( 17)/       0.5885190D0/
      DATA ADP    ( 17)/       0.6814522D0/
      DATA AQP    ( 17)/       0.3643694D0/
      DATA FN1P   ( 17,1)/      -0.1715910D0/
      DATA FN2P   ( 17,1)/       6.0008020D0/
      DATA FN3P   ( 17,1)/       1.0875020D0/
      DATA FN1P   ( 17,2)/      -0.0134580D0/
      DATA FN2P   ( 17,2)/       1.9666180D0/
      DATA FN3P   ( 17,2)/       2.2928910D0/
      DATA VSP(17)/-25.26D00/
      DATA VPP(17)/-15.09D00/
!=========================================================================
!     Potassium parameters.
!=========================================================================
      DATA ALFA(19)  / 1.16D0/, ALPM(19)   / 1.16D0/
      DATA EISOL(19) / 0.00D0/, EISOLM(19) / 0.00D0/
      DATA BDD(19,1)    / 0.50D0/, AMM(19)    / 0.50D0/
      DATA VS(19)    /10.00D0/
!=========================================================================
!     Calcium parameters.
!=========================================================================
!      DATA REFPM3(20)/ ' Ca: (PM3): Wavefunction Incorporated'/
      DATA USSP   ( 20)/     -11.35010387D0/
      DATA UPPP   ( 20)/     -10.34987587D0/
      DATA BETASP ( 20)/      -10.45737746D0/
      DATA BETAPP ( 20)/      -5.10954286D0/
      DATA ZSP    ( 20)/       0.69567815D0/
      DATA ZPP    ( 20)/       1.05125946D0/
      DATA ALPP   ( 20)/       1.90107319D0/
      DATA GSSP   ( 20)/       5.76D0/
      DATA GSPP   ( 20)/       5.04D0/
      DATA GPPP   ( 20)/       4.32D0/
      DATA GP2P   ( 20)/       4.00D0/
      DATA HSPP   ( 20)/       0.52D0/
      DATA FN1P   ( 20,1)/       0.52766269D0/
      DATA FN2P   ( 20,1)/       10.0D0/
      DATA FN3P   ( 20,1)/       0.51696708D0/
      DATA FN1P   ( 20,2)/      -0.00139269D0/
      DATA FN2P   ( 20,2)/       6.0D0/
      DATA FN3P   ( 20,2)/       2.56686118D0/
!=========================================================================
!     Titanium parameters
!=========================================================================
!        DATA REFPM3 (22)/' Ti: For Russel-Saunders work only
!     1                                '/
      DATA USSP    (22)/      10.0000000D0/
      DATA UPPP    (22)/      10.0000000D0/
      DATA UDDP    (22)/     -30.0000000D0/
      DATA BETASP  (22)/      -0.1000000D0/
      DATA BETAPP  (22)/      -0.1000000D0/
      DATA ZSP     (22)/       1.5000000D0/
      DATA ZPP     (22)/       1.5000000D0/
      DATA ZDP     (22)/       2.8845490D0/
      DATA ALPP    (22)/       3.0683070D0/
      DATA GSSP    (22)/       6.0000000D0/
      DATA GSPP    (22)/       4.1500000D0/
      DATA GPPP    (22)/       5.0000000D0/
      DATA GP2P    (22)/       3.5000000D0/
      DATA HSPP    (22)/       1.0000000D0/
!=========================================================================
!     Chromium parameters.
!=========================================================================
      DATA ALPM(24)   /   3.0683070D0/
      DATA EISOLM(24) /-134.8187920D0/
      DATA BETASM(24) /  -0.1000000D0/
      DATA BETAPM(24) /  -0.1000000D0/
      DATA BETADM(24) /  -8.7766360D0/
      DATA ZSM(24)    /   1.5000000D0/
      DATA ZPM(24)    /   1.5000000D0/
      DATA ZDM(24)    /   2.8845490D0/
      DATA DDM(24)    /   1.7320508D0/
      DATA QQM(24)    /   1.4142136D0/
      DATA AMM(24)    /   0.2205072D0/
      DATA ADM(24)    /   0.2711332D0/
      DATA AQM(24)    /   0.4464656D0/
      DATA USSM(24)   / -17.5170270D0/
      DATA UPPM(24)   / -12.5337290D0/
      DATA UDDM(24)   / -44.1249280D0/
      DATA GSSM(24)   /   6.0000000D0/
      DATA GSPM(24)   /   4.1500000D0/
      DATA GPPM(24)   /   5.0000000D0/
      DATA GP2M(24)   /   3.5000000D0/
      DATA HSPM(24)   /   1.0000000D0/
      DATA GSD(24)    /   2.8746410D0/
      DATA GPD(24)    /   3.0000000D0/
      DATA GDD(24)    /   8.8949670D0/
!=========================================================================
!     Zinc parameters
!=========================================================================
!                    DATA FOR ELEMENT 30        ZINC
!      DATA REFPM3(30)/ ' Zn: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1(ACCEPTED)                      '/
      DATA USSP   ( 30)/     -18.5321980D0/
      DATA UPPP   ( 30)/     -11.0474090D0/
      DATA BETASP ( 30)/      -0.7155780D0/
      DATA BETAPP ( 30)/      -6.3518640D0/
      DATA ZSP    ( 30)/       1.8199890D0/
      DATA ZPP    ( 30)/       1.5069220D0/
      DATA ZDP    ( 30)/       1.0000000D0/
      DATA ALPP   ( 30)/       1.3501260D0/
      DATA EISOLP ( 30)/     -27.3872000D0/
      DATA GSSP   ( 30)/       9.6771960D0/
      DATA GSPP   ( 30)/       7.7362040D0/
      DATA GPPP   ( 30)/       4.9801740D0/
      DATA GP2P   ( 30)/       4.6696560D0/
      DATA HSPP   ( 30)/       0.6004130D0/
      DATA DDP    ( 30)/       1.5005758D0/
      DATA QQP    ( 30)/       1.4077174D0/
      DATA AMP    ( 30)/       0.3556485D0/
      DATA ADP    ( 30)/       0.2375689D0/
      DATA AQP    ( 30)/       0.2661069D0/
      DATA FN1P   ( 30,1)/      -0.1112340D0/
      DATA FN2P   ( 30,1)/       6.0014780D0/
      DATA FN3P   ( 30,1)/       1.5160320D0/
      DATA FN1P   ( 30,2)/      -0.1323700D0/
      DATA FN2P   ( 30,2)/       1.9958390D0/
      DATA FN3P   ( 30,2)/       2.5196420D0/
!=========================================================================
!     Gallium parameters
!=========================================================================
!                    DATA FOR ELEMENT 31        GALLIUM
!      DATA REFPM3(31)/ ' Ga: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1(ACCEPTED)                      '/
      DATA USSP   ( 31)/     -29.8555930D0/
      DATA UPPP   ( 31)/     -21.8753710D0/
      DATA BETASP ( 31)/      -4.9456180D0/
      DATA BETAPP ( 31)/      -0.4070530D0/
      DATA ZSP    ( 31)/       1.8470400D0/
      DATA ZPP    ( 31)/       0.8394110D0/
      DATA ALPP   ( 31)/       1.6051150D0/
      DATA EISOLP ( 31)/     -57.3280250D0/
      DATA GSSP   ( 31)/       8.4585540D0/
      DATA GSPP   ( 31)/       8.9256190D0/
      DATA GPPP   ( 31)/       5.0868550D0/
      DATA GP2P   ( 31)/       4.9830450D0/
      DATA HSPP   ( 31)/       2.0512600D0/
      DATA DDP    ( 31)/       0.9776692D0/
      DATA QQP    ( 31)/       2.5271534D0/
      DATA AMP    ( 31)/       0.3108620D0/
      DATA ADP    ( 31)/       0.5129360D0/
      DATA AQP    ( 31)/       0.1546208D0/
      DATA FN1P   ( 31,1)/      -0.5601790D0/
      DATA FN2P   ( 31,1)/       5.6232730D0/
      DATA FN3P   ( 31,1)/       1.5317800D0/
      DATA FN1P   ( 31,2)/      -0.2727310D0/
      DATA FN2P   ( 31,2)/       1.9918430D0/
      DATA FN3P   ( 31,2)/       2.1838640D0/
!=========================================================================
!     Germanium parameters.
!=========================================================================
      DATA ALPM(32)   /  1.9784980D0/
      DATA EISOLM(32) /-76.2489440D0/
      DATA BETASM(32) / -4.5164790D0/
      DATA BETAPM(32) / -1.7555170D0/
      DATA ZSM(32)    /  1.2931800D0/
      DATA ZPM(32)    /  2.0205640D0/
      DATA DDM(32)    /  1.2556091D0/
      DATA QQM(32)    /  1.0498655D0/
      DATA AMM(32)    /  0.3601617D0/
      DATA ADM(32)    /  0.3643722D0/
      DATA AQM(32)    /  0.4347337D0/
      DATA USSM(32)   /-33.9493670D0/
      DATA UPPM(32)   /-27.4251050D0/
      DATA GSSM(32)   /  9.8000000D0/
      DATA GSPM(32)   /  8.3000000D0/
      DATA GPPM(32)   /  7.3000000D0/
      DATA GP2M(32)   /  6.5000000D0/
      DATA HSPM(32)   /  1.3000000D0/
!      DATA REFPM3(32)/ ' Ge: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1(ACCEPTED)                      '/
      DATA USSP   ( 32)/     -35.4671955D0/
      DATA UPPP   ( 32)/     -31.5863583D0/
      DATA BETASP ( 32)/      -5.3250024D0/
      DATA BETAPP ( 32)/      -2.2501567D0/
      DATA ZSP    ( 32)/       2.2373526D0/
      DATA ZPP    ( 32)/       1.5924319D0/
      DATA ALPP   ( 32)/       1.9723370D0/
      DATA EISOLP ( 32)/     -84.0156006D0/
      DATA GSSP   ( 32)/       5.3769635D0/
      DATA GSPP   ( 32)/      10.2095293D0/
      DATA GPPP   ( 32)/       7.6718647D0/
      DATA GP2P   ( 32)/       6.9242663D0/
      DATA HSPP   ( 32)/       1.3370204D0/
      DATA DDP    ( 32)/       1.1920304D0/
      DATA QQP    ( 32)/       1.3321263D0/
      DATA AMP    ( 32)/       0.1976098D0/
      DATA ADP    ( 32)/       0.3798182D0/
      DATA AQP    ( 32)/       0.3620669D0/
      DATA FN1P   ( 32,1)/       0.9631726D0/
      DATA FN2P   ( 32,1)/       6.0120134D0/
      DATA FN3P   ( 32,1)/       2.1633655D0/
      DATA FN1P   ( 32,2)/      -0.9593891D0/
      DATA FN2P   ( 32,2)/       5.7491802D0/
      DATA FN3P   ( 32,2)/       2.1693724D0/
!=========================================================================
!     Arsenic parameters
!=========================================================================
!                    DATA FOR ELEMENT 33        ARSENIC
!      DATA REFPM3(33)/ ' As: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1(ACCEPTED)                      '/
      DATA USSP   ( 33)/     -38.5074240D0/
      DATA UPPP   ( 33)/     -35.1524150D0/
      DATA BETASP ( 33)/      -8.2321650D0/
      DATA BETAPP ( 33)/      -5.0173860D0/
      DATA ZSP    ( 33)/       2.6361770D0/
      DATA ZPP    ( 33)/       1.7038890D0/
      DATA ALPP   ( 33)/       1.7944770D0/
      DATA EISOLP ( 33)/    -122.6326140D0/
      DATA GSSP   ( 33)/       8.7890010D0/
      DATA GSPP   ( 33)/       5.3979830D0/
      DATA GPPP   ( 33)/       8.2872500D0/
      DATA GP2P   ( 33)/       8.2103460D0/
      DATA HSPP   ( 33)/       1.9510340D0/
      DATA DDP    ( 33)/       0.9679655D0/
      DATA QQP    ( 33)/       1.2449874D0/
      DATA AMP    ( 33)/       0.3230063D0/
      DATA ADP    ( 33)/       0.5042239D0/
      DATA AQP    ( 33)/       0.2574219D0/
      DATA FN1P   ( 33,1)/      -0.4600950D0/
      DATA FN2P   ( 33,1)/       1.9831150D0/
      DATA FN3P   ( 33,1)/       1.0867930D0/
      DATA FN1P   ( 33,2)/      -0.0889960D0/
      DATA FN2P   ( 33,2)/       1.9929440D0/
      DATA FN3P   ( 33,2)/       2.1400580D0/
!=========================================================================
!     Selenium parameters
!=========================================================================
!                    DATA FOR ELEMENT 34        SELENIUM
!      DATA REFPM3(34)/ ' Se: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1(ACCEPTED)                      '/
      DATA USSP   ( 34)/     -55.3781350D0/
      DATA UPPP   ( 34)/     -49.8230760D0/
      DATA BETASP ( 34)/      -6.1578220D0/
      DATA BETAPP ( 34)/      -5.4930390D0/
      DATA ZSP    ( 34)/       2.8280510D0/
      DATA ZPP    ( 34)/       1.7325360D0/
      DATA ALPP   ( 34)/       3.0439570D0/
      DATA EISOLP ( 34)/    -192.7748115D0/
      DATA GSSP   ( 34)/       7.4325910D0/
      DATA GSPP   ( 34)/      10.0604610D0/
      DATA GPPP   ( 34)/       9.5683260D0/
      DATA GP2P   ( 34)/       7.7242890D0/
      DATA HSPP   ( 34)/       4.0165580D0/
      DATA DDP    ( 34)/       0.8719813D0/
      DATA QQP    ( 34)/       1.2244019D0/
      DATA AMP    ( 34)/       0.2731566D0/
      DATA ADP    ( 34)/       0.7509697D0/
      DATA AQP    ( 34)/       0.5283737D0/
      DATA FN1P   ( 34,1)/       0.0478730D0/
      DATA FN2P   ( 34,1)/       6.0074000D0/
      DATA FN3P   ( 34,1)/       2.0817170D0/
      DATA FN1P   ( 34,2)/       0.1147200D0/
      DATA FN2P   ( 34,2)/       6.0086720D0/
      DATA FN3P   ( 34,2)/       1.5164230D0/
!=========================================================================
!     Bromine parameters.
!=========================================================================
      DATA ALFA(35)  /   2.5765460D0/, ALPM(35)   /   2.44570510D0/
      DATA EISOL(35) /-352.3142087D0/, EISOLM(35) /-346.68125000D0/
      DATA BETAS(35) / -19.3998800D0/, BETASM(35) /  -8.91710700D0/
      DATA BETAP(35) /  -8.9571950D0/, BETAPM(35) /  -9.94374000D0/
      DATA ZS(35)    /   3.0641330D0/, ZSM(35)    /   3.85430190D0/
      DATA ZP(35)    /   2.0383330D0/, ZPM(35)    /   2.19920910D0/
      DATA ZD(35)    /   1.0000000D0/, ZDM(35)    /   1.00000000D0/
      DATA DD(35)    /   0.8458104D0/, DDM(35)    /   0.60510740D0/
      DATA QQ(35)    /   1.0407133D0/, QQM(35)    /   0.96458730D0/
      DATA BDD(35,1)    /   0.5526071D0/, AMM(35)    /   0.55260680D0/
      DATA BDD(35,2)    /   0.6024598D0/, ADM(35)    /   0.72583300D0/
      DATA BDD(35,3)    /   0.5307555D0/, AQM(35)    /   0.55745890D0/
      DATA USS(35)   /-104.6560630D0/, USSM(35)   / -99.98644050D0/
      DATA UPP(35)   / -74.9300520D0/, UPPM(35)   / -75.67130750D0/
      DATA GSS(35)   /  15.0364395D0/, GSSM(35)   /  15.03643948D0/
      DATA GSP(35)   /  13.0346824D0/, GSPM(35)   /  13.03468242D0/
      DATA GPP(35)   /  11.2763254D0/, GPPM(35)   /  11.27632539D0/
      DATA GP2(35)   /   9.8544255D0/, GP2M(35)   /   9.85442552D0/
      DATA HSP(35)   /   2.4558683D0/, HSPM(35)   /   2.45586832D0/
!
      DATA FN1(35,1) /   0.0666850D0/
      DATA FN2(35,1) /   4.0000000D0/
      DATA FN3(35,1) /   1.5000000D0/
      DATA FN1(35,2) /   0.0255680D0/
      DATA FN2(35,2) /   4.0000000D0/
      DATA FN3(35,2) /   2.3000000D0/
!      DATA REFPM3 (35)/' Br: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1 10, 209 (1989).                '/
      DATA USSP   ( 35)/    -116.6193110D0/
      DATA UPPP   ( 35)/     -74.2271290D0/
      DATA BETASP ( 35)/     -31.1713420D0/
      DATA BETAPP ( 35)/      -6.8140130D0/
      DATA ZSP    ( 35)/       5.3484570D0/
      DATA ZPP    ( 35)/       2.1275900D0/
      DATA ZDP    ( 35)/       1.0000000D0/
      DATA ALPP   ( 35)/       2.5118420D0/
      DATA EISOLP ( 35)/    -352.5398970D0/
      DATA GSSP   ( 35)/      15.9434250D0/
      DATA GSPP   ( 35)/      16.0616800D0/
      DATA GPPP   ( 35)/       8.2827630D0/
      DATA GP2P   ( 35)/       7.8168490D0/
      DATA HSPP   ( 35)/       0.5788690D0/
      DATA DDP    ( 35)/       0.2759025D0/
      DATA QQP    ( 35)/       0.9970532D0/
      DATA AMP    ( 35)/       0.5859399D0/
      DATA ADP    ( 35)/       0.6755383D0/
      DATA AQP    ( 35)/       0.3823719D0/
      DATA FN1P   ( 35,1)/       0.9604580D0/
      DATA FN2P   ( 35,1)/       5.9765080D0/
      DATA FN3P   ( 35,1)/       2.3216540D0/
      DATA FN1P   ( 35,2)/      -0.9549160D0/
      DATA FN2P   ( 35,2)/       5.9447030D0/
      DATA FN3P   ( 35,2)/       2.3281420D0/
!=========================================================================
!     Cadmium parameters
!=========================================================================
!                    DATA FOR ELEMENT 48        CADPMIUM
!      DATA REFPM3(48)/ ' Cd: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1(ACCEPTED)                      '/
      DATA USSP   ( 48)/     -15.8285840D0/
      DATA UPPP   ( 48)/       8.7497950D0/
      DATA BETASP ( 48)/      -8.5819440D0/
      DATA BETAPP ( 48)/      -0.6010340D0/
      DATA ZSP    ( 48)/       1.6793510D0/
      DATA ZPP    ( 48)/       2.0664120D0/
      DATA ALPP   ( 48)/       1.5253820D0/
      DATA EISOLP ( 48)/     -22.4502080D0/
      DATA GSSP   ( 48)/       9.2069600D0/
      DATA GSPP   ( 48)/       8.2315390D0/
      DATA GPPP   ( 48)/       4.9481040D0/
      DATA GP2P   ( 48)/       4.6696560D0/
      DATA HSPP   ( 48)/       1.6562340D0/
      DATA DDP    ( 48)/       1.5982681D0/
      DATA QQP    ( 48)/       1.2432402D0/
      DATA AMP    ( 48)/       0.3383668D0/
      DATA ADP    ( 48)/       0.3570290D0/
      DATA AQP    ( 48)/       0.2820582D0/
!=========================================================================
!     Indium parameters
!=========================================================================
!                    DATA FOR ELEMENT 49        INDIUM
!      DATA REFPM3(49)/ ' In: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1(ACCEPTED)                      '/
      DATA USSP   ( 49)/     -26.1762050D0/
      DATA UPPP   ( 49)/     -20.0058220D0/
      DATA BETASP ( 49)/      -2.9933190D0/
      DATA BETAPP ( 49)/      -1.8289080D0/
      DATA ZSP    ( 49)/       2.0161160D0/
      DATA ZPP    ( 49)/       1.4453500D0/
      DATA ALPP   ( 49)/       1.4183850D0/
      DATA EISOLP ( 49)/     -51.9750470D0/
      DATA GSSP   ( 49)/       6.5549000D0/
      DATA GSPP   ( 49)/       8.2298730D0/
      DATA GPPP   ( 49)/       6.2992690D0/
      DATA GP2P   ( 49)/       4.9842110D0/
      DATA HSPP   ( 49)/       2.6314610D0/
      DATA DDP    ( 49)/       1.5766241D0/
      DATA QQP    ( 49)/       1.7774563D0/
      DATA AMP    ( 49)/       0.2409004D0/
      DATA ADP    ( 49)/       0.4532655D0/
      DATA AQP    ( 49)/       0.3689812D0/
      DATA FN1P   ( 49,1)/      -0.3431380D0/
      DATA FN2P   ( 49,1)/       1.9940340D0/
      DATA FN3P   ( 49,1)/       1.6255160D0/
      DATA FN1P   ( 49,2)/      -0.1095320D0/
      DATA FN2P   ( 49,2)/       5.6832170D0/
      DATA FN3P   ( 49,2)/       2.8670090D0/
!=========================================================================
!     Tin parameters.
!=========================================================================
      DATA ALPM(50)   /  1.8008140D0/
      DATA EISOLM(50) /-92.3241020D0/
      DATA BETASM(50) / -3.2351470D0/
      DATA BETAPM(50) / -4.2904160D0/
      DATA ZSM(50)    /  2.0803800D0/
      DATA ZPM(50)    /  1.9371060D0/
      DATA DDM(50)    /  1.5697766D0/
      DATA QQM(50)    /  1.3262292D0/
      DATA AMM(50)    /  0.3601617D0/
      DATA ADM(50)    /  0.3219998D0/
      DATA AQM(50)    /  0.3713827D0/
      DATA USSM(50)   /-40.8518020D0/
      DATA UPPM(50)   /-28.5602490D0/
      DATA GSSM(50)   /  9.8000000D0/
      DATA GSPM(50)   /  8.3000000D0/
      DATA GPPM(50)   /  7.3000000D0/
      DATA GP2M(50)   /  6.5000000D0/
      DATA HSPM(50)   /  1.3000000D0/
!      DATA REFPM3(50)/ ' Sn: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1(ACCEPTED)                      '/
      DATA USSP   ( 50)/     -34.5501920D0/
      DATA UPPP   ( 50)/     -25.8944190D0/
      DATA BETASP ( 50)/      -2.7858020D0/
      DATA BETAPP ( 50)/      -2.0059990D0/
      DATA ZSP    ( 50)/       2.3733280D0/
      DATA ZPP    ( 50)/       1.6382330D0/
      DATA ALPP   ( 50)/       1.6996500D0/
      DATA EISOLP ( 50)/     -78.8877790D0/
      DATA GSSP   ( 50)/      10.1900330D0/
      DATA GSPP   ( 50)/       7.2353270D0/
      DATA GPPP   ( 50)/       5.6738100D0/
      DATA GP2P   ( 50)/       5.1822140D0/
      DATA HSPP   ( 50)/       1.0331570D0/
      DATA DDP    ( 50)/       1.3120038D0/
      DATA QQP    ( 50)/       1.5681814D0/
      DATA AMP    ( 50)/       0.3744959D0/
      DATA ADP    ( 50)/       0.3218163D0/
      DATA AQP    ( 50)/       0.2832529D0/
      DATA FN1P   ( 50,1)/      -0.1503530D0/
      DATA FN2P   ( 50,1)/       6.0056940D0/
      DATA FN3P   ( 50,1)/       1.7046420D0/
      DATA FN1P   ( 50,2)/      -0.0444170D0/
      DATA FN2P   ( 50,2)/       2.2573810D0/
      DATA FN3P   ( 50,2)/       2.4698690D0/
!=========================================================================
!     Antimony parameters
!=========================================================================
!                    DATA FOR ELEMENT 51        ANTIMONY
!      DATA REFPM3(51)/ ' Sb: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1(ACCEPTED)                      '/
      DATA USSP   ( 51)/     -56.4321960D0/
      DATA UPPP   ( 51)/     -29.4349540D0/
      DATA BETASP ( 51)/     -14.7942170D0/
      DATA BETAPP ( 51)/      -2.8179480D0/
      DATA ZSP    ( 51)/       2.3430390D0/
      DATA ZPP    ( 51)/       1.8999920D0/
      DATA ALPP   ( 51)/       2.0343010D0/
      DATA EISOLP ( 51)/    -148.9382890D0/
      DATA GSSP   ( 51)/       9.2382770D0/
      DATA GSPP   ( 51)/       5.2776800D0/
      DATA GPPP   ( 51)/       6.3500000D0/
      DATA GP2P   ( 51)/       6.2500000D0/
      DATA HSPP   ( 51)/       2.4244640D0/
      DATA DDP    ( 51)/       1.4091903D0/
      DATA QQP    ( 51)/       1.3521354D0/
      DATA AMP    ( 51)/       0.3395177D0/
      DATA ADP    ( 51)/       0.4589010D0/
      DATA AQP    ( 51)/       0.2423472D0/
      DATA FN1P   ( 51,1)/       3.0020280D0/
      DATA FN2P   ( 51,1)/       6.0053420D0/
      DATA FN3P   ( 51,1)/       0.8530600D0/
      DATA FN1P   ( 51,2)/      -0.0188920D0/
      DATA FN2P   ( 51,2)/       6.0114780D0/
      DATA FN3P   ( 51,2)/       2.7933110D0/
!=========================================================================
!     Tellurium parameters
!=========================================================================
!                    DATA FOR ELEMENT 52        TELLURIUM
!      DATA REFPM3(52)/ ' Te: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1(ACCEPTED)                      '/
      DATA USSP   ( 52)/     -44.9380360D0/
      DATA UPPP   ( 52)/     -46.3140990D0/
      DATA BETASP ( 52)/      -2.6651460D0/
      DATA BETAPP ( 52)/      -3.8954300D0/
      DATA ZSP    ( 52)/       4.1654920D0/
      DATA ZPP    ( 52)/       1.6475550D0/
      DATA ALPP   ( 52)/       2.4850190D0/
      DATA EISOLP ( 52)/    -168.0945925D0/
      DATA GSSP   ( 52)/      10.2550730D0/
      DATA GSPP   ( 52)/       8.1691450D0/
      DATA GPPP   ( 52)/       7.7775920D0/
      DATA GP2P   ( 52)/       7.7551210D0/
      DATA HSPP   ( 52)/       3.7724620D0/
      DATA DDP    ( 52)/       0.3484177D0/
      DATA QQP    ( 52)/       1.5593085D0/
      DATA AMP    ( 52)/       0.3768862D0/
      DATA ADP    ( 52)/       1.1960743D0/
      DATA AQP    ( 52)/       0.2184786D0/
      DATA FN1P   ( 52,1)/       0.0333910D0/
      DATA FN2P   ( 52,1)/       5.9563790D0/
      DATA FN3P   ( 52,1)/       2.2775750D0/
      DATA FN1P   ( 52,2)/      -1.9218670D0/
      DATA FN2P   ( 52,2)/       4.9732190D0/
      DATA FN3P   ( 52,2)/       0.5242430D0/
!=========================================================================
!     Iodine parameters.
!=========================================================================
      DATA USS(53)   /-103.5896630D0/, USSM(53)   /-100.00305380D0/
      DATA UPP(53)   / -74.4299970D0/, UPPM(53)   / -74.61146920D0/
      DATA BETAS(53) /  -8.4433270D0/, BETASM(53) /  -7.41445100D0/
      DATA BETAP(53) /  -6.3234050D0/, BETAPM(53) /  -6.19678100D0/
      DATA ZS(53)    /   2.1028580D0/, ZSM(53)    /   2.27296100D0/
      DATA ZP(53)    /   2.1611530D0/, ZPM(53)    /   2.16949800D0/
      DATA ZD(53)    /   1.0000000D0/, ZDM(53)    /   1.00000000D0/
      DATA ALFA(53)  /   2.2994240D0/, ALPM(53)   /   2.20732000D0/
      DATA EISOL(53) /-346.8642857D0/, EISOLM(53) /-340.59836000D0/
      DATA GSS(53)   /  15.0404486D0/, GSSM(53)   /  15.04044855D0/
      DATA GSP(53)   /  13.0565580D0/, GSPM(53)   /  13.05655798D0/
      DATA GPP(53)   /  11.1477837D0/, GPPM(53)   /  11.14778369D0/
      DATA GP2(53)   /   9.9140907D0/, GP2M(53)   /   9.91409071D0/
      DATA HSP(53)   /   2.4563820D0/, HSPM(53)   /   2.45638202D0/
      DATA DD(53)    /   1.4878778D0/, DDM(53)    /   1.42532330D0/
      DATA QQ(53)    /   1.1887388D0/, QQM(53)    /   1.18417070D0/
      DATA BDD(53,1)    /   0.5527544D0/, AMM(53)    /   0.55275410D0/
      DATA BDD(53,2)    /   0.4497523D0/, ADM(53)    /   0.45934510D0/
      DATA BDD(53,3)    /   0.4631775D0/, AQM(53)    /   0.45853760D0/
!
      DATA FN1(53,1) /   0.0043610D0/
      DATA FN2(53,1) /   2.3000000D0/
      DATA FN3(53,1) /   1.8000000D0/
      DATA FN1(53,2) /   0.0157060D0/
      DATA FN2(53,2) /   3.0000000D0/
      DATA FN3(53,2) /   2.2400000D0/
!      DATA REFPM3 (53)/'  I: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1 10, 209 (1989).                '/
      DATA USSP   ( 53)/     -96.4540370D0/
      DATA UPPP   ( 53)/     -61.0915820D0/
      DATA BETASP ( 53)/     -14.4942340D0/
      DATA BETAPP ( 53)/      -5.8947030D0/
      DATA ZSP    ( 53)/       7.0010130D0/
      DATA ZPP    ( 53)/       2.4543540D0/
      DATA ZDP    ( 53)/       1.0000000D0/
      DATA ALPP   ( 53)/       1.9901850D0/
      DATA EISOLP ( 53)/    -288.3160860D0/
      DATA GSSP   ( 53)/      13.6319430D0/
      DATA GSPP   ( 53)/      14.9904060D0/
      DATA GPPP   ( 53)/       7.2883300D0/
      DATA GP2P   ( 53)/       5.9664070D0/
      DATA HSPP   ( 53)/       2.6300350D0/
      DATA DDP    ( 53)/       0.1581469D0/
      DATA QQP    ( 53)/       1.0467302D0/
      DATA AMP    ( 53)/       0.5009902D0/
      DATA ADP    ( 53)/       1.6699104D0/
      DATA AQP    ( 53)/       0.5153082D0/
      DATA FN1P   ( 53,1)/      -0.1314810D0/
      DATA FN2P   ( 53,1)/       5.2064170D0/
      DATA FN3P   ( 53,1)/       1.7488240D0/
      DATA FN1P   ( 53,2)/      -0.0368970D0/
      DATA FN2P   ( 53,2)/       6.0101170D0/
      DATA FN3P   ( 53,2)/       2.7103730D0/
!=========================================================================
!     Mercury parameters.
!=========================================================================
      DATA ALPM(80)   /  1.3356410D0/
      DATA EISOLM(80) /-28.8191480D0/
      DATA BETASM(80) / -0.4045250D0/
      DATA BETAPM(80) / -6.2066830D0/
      DATA ZSM(80)    /  2.2181840D0/
      DATA ZPM(80)    /  2.0650380D0/
      DATA DDM(80)    /  1.7378048D0/
      DATA QQM(80)    /  1.4608064D0/
      DATA AMM(80)    /  0.3969129D0/
      DATA ADM(80)    /  0.3047694D0/
      DATA AQM(80)    /  0.3483102D0/
      DATA USSM(80)   /-19.8095740D0/
      DATA UPPM(80)   /-13.1025300D0/
      DATA GSSM(80)   / 10.8000000D0/
      DATA GSPM(80)   /  9.3000000D0/
      DATA GPPM(80)   / 14.3000000D0/
      DATA GP2M(80)   / 13.5000000D0/
      DATA HSPM(80)   /  1.3000000D0/
!      DATA REFPM3(80)/ ' Hg: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1(ACCEPTED)                      '/
      DATA USSP   ( 80)/     -17.7622290D0/
      DATA UPPP   ( 80)/     -18.3307510D0/
      DATA BETASP ( 80)/      -3.1013650D0/
      DATA BETAPP ( 80)/      -3.4640310D0/
      DATA ZSP    ( 80)/       1.4768850D0/
      DATA ZPP    ( 80)/       2.4799510D0/
      DATA ALPP   ( 80)/       1.5293770D0/
      DATA EISOLP ( 80)/     -28.8997380D0/
      DATA GSSP   ( 80)/       6.6247200D0/
      DATA GSPP   ( 80)/      10.6392970D0/
      DATA GPPP   ( 80)/      14.7092830D0/
      DATA GP2P   ( 80)/      16.0007400D0/
      DATA HSPP   ( 80)/       2.0363110D0/
      DATA DDP    ( 80)/       1.2317811D0/
      DATA QQP    ( 80)/       1.2164033D0/
      DATA AMP    ( 80)/       0.2434664D0/
      DATA ADP    ( 80)/       0.4515472D0/
      DATA AQP    ( 80)/       0.2618394D0/
      DATA FN1P   ( 80,1)/       1.0827200D0/
      DATA FN2P   ( 80,1)/       6.4965980D0/
      DATA FN3P   ( 80,1)/       1.1951460D0/
      DATA FN1P   ( 80,2)/      -0.0965530D0/
      DATA FN2P   ( 80,2)/       3.9262810D0/
      DATA FN3P   ( 80,2)/       2.6271600D0/
!=========================================================================
!     Thallium parameters
!=========================================================================
!                    DATA FOR ELEMENT 81        THALLIUM
!      DATA REFPM3(81)/ ' Tl: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1(ACCEPTED)                      '/
      DATA USSP   ( 81)/     -30.0531700D0/
      DATA UPPP   ( 81)/     -26.9206370D0/
      DATA BETASP ( 81)/      -1.0844950D0/
      DATA BETAPP ( 81)/      -7.9467990D0/
      DATA ZSP    ( 81)/       6.8679210D0/
      DATA ZPP    ( 81)/       1.9694450D0/
      DATA ALPP   ( 81)/       1.3409510D0/
      DATA EISOLP ( 81)/     -56.6492050D0/
      DATA GSSP   ( 81)/      10.4604120D0/
      DATA GSPP   ( 81)/      11.2238830D0/
      DATA GPPP   ( 81)/       4.9927850D0/
      DATA GP2P   ( 81)/       8.9627270D0/
      DATA HSPP   ( 81)/       2.5304060D0/
      DATA DDP    ( 81)/       0.0781362D0/
      DATA QQP    ( 81)/       1.5317110D0/
      DATA AMP    ( 81)/       0.3844326D0/
      DATA ADP    ( 81)/       2.5741815D0/
      DATA AQP    ( 81)/       0.2213264D0/
      DATA FN1P   ( 81,1)/      -1.3613990D0/
      DATA FN2P   ( 81,1)/       3.5572260D0/
      DATA FN3P   ( 81,1)/       1.0928020D0/
      DATA FN1P   ( 81,2)/      -0.0454010D0/
      DATA FN2P   ( 81,2)/       2.3069950D0/
      DATA FN3P   ( 81,2)/       2.9650290D0/
!=========================================================================
!     Lead parameters.
!=========================================================================
      DATA ALPM(82)   /   1.7283330D0/
      DATA EISOLM(82) /-105.8345040D0/
      DATA BETASM(82) /  -8.0423870D0/
      DATA BETAPM(82) /  -3.0000000D0/
      DATA ZSM(82)    /   2.4982860D0/
      DATA ZPM(82)    /   2.0820710D0/
      DATA DDM(82)    /   1.5526624D0/
      DATA QQM(82)    /   1.4488558D0/
      DATA AMM(82)    /   0.3601617D0/
      DATA ADM(82)    /   0.3239309D0/
      DATA AQM(82)    /   0.3502057D0/
      DATA USSM(82)   / -47.3196920D0/
      DATA UPPM(82)   / -28.8475600D0/
      DATA GSSM(82)   /   9.8000000D0/
      DATA GSPM(82)   /   8.3000000D0/
      DATA GPPM(82)   /   7.3000000D0/
      DATA GP2M(82)   /   6.5000000D0/
      DATA HSPM(82)   /   1.3000000D0/
!      DATA REFPM3(82)/ ' Pb: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1(ACCEPTED)                      '/
      DATA USSP   ( 82)/     -30.3227560D0/
      DATA UPPP   ( 82)/     -24.4258340D0/
      DATA BETASP ( 82)/      -6.1260240D0/
      DATA BETAPP ( 82)/      -1.3954300D0/
      DATA ZSP    ( 82)/       3.1412890D0/
      DATA ZPP    ( 82)/       1.8924180D0/
      DATA ALPP   ( 82)/       1.6200450D0/
      DATA EISOLP ( 82)/     -73.4660775D0/
      DATA GSSP   ( 82)/       7.0119920D0/
      DATA GSPP   ( 82)/       6.7937820D0/
      DATA GPPP   ( 82)/       5.1837800D0/
      DATA GP2P   ( 82)/       5.0456510D0/
      DATA HSPP   ( 82)/       1.5663020D0/
      DATA DDP    ( 82)/       0.9866290D0/
      DATA QQP    ( 82)/       1.5940562D0/
      DATA AMP    ( 82)/       0.2576991D0/
      DATA ADP    ( 82)/       0.4527678D0/
      DATA AQP    ( 82)/       0.2150175D0/
      DATA FN1P   ( 82,1)/      -0.1225760D0/
      DATA FN2P   ( 82,1)/       6.0030620D0/
      DATA FN3P   ( 82,1)/       1.9015970D0/
      DATA FN1P   ( 82,2)/      -0.0566480D0/
      DATA FN2P   ( 82,2)/       4.7437050D0/
      DATA FN3P   ( 82,2)/       2.8618790D0/
!=========================================================================
!     Bismuth parameters
!=========================================================================
!                    DATA FOR ELEMENT 83        BISMUTH
!      DATA REFPM3(83)/ ' Bi: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1(ACCEPTED)                      '/
      DATA USSP   ( 83)/     -33.4959380D0/
      DATA UPPP   ( 83)/     -35.5210260D0/
      DATA BETASP ( 83)/      -5.6072830D0/
      DATA BETAPP ( 83)/      -5.8001520D0/
      DATA ZSP    ( 83)/       4.9164510D0/
      DATA ZPP    ( 83)/       1.9349350D0/
      DATA ALPP   ( 83)/       1.8574310D0/
      DATA EISOLP ( 83)/    -109.2774910D0/
      DATA GSSP   ( 83)/       4.9894800D0/
      DATA GSPP   ( 83)/       6.1033080D0/
      DATA GPPP   ( 83)/       8.6960070D0/
      DATA GP2P   ( 83)/       8.3354470D0/
      DATA HSPP   ( 83)/       0.5991220D0/
      DATA DDP    ( 83)/       0.2798609D0/
      DATA QQP    ( 83)/       1.5590294D0/
      DATA AMP    ( 83)/       0.1833693D0/
      DATA ADP    ( 83)/       0.6776013D0/
      DATA AQP    ( 83)/       0.2586520D0/
      DATA FN1P   ( 83,1)/       2.5816930D0/
      DATA FN2P   ( 83,1)/       5.0940220D0/
      DATA FN3P   ( 83,1)/       0.4997870D0/
      DATA FN1P   ( 83,2)/       0.0603200D0/
      DATA FN2P   ( 83,2)/       6.0015380D0/
      DATA FN3P   ( 83,2)/       2.4279700D0/
!=========================================================================
!     Start parameter declarations for GHO atoms.
!=========================================================================
      DATA ALFA(91)  /   2.6482740D0/, ALPM(91)   /   2.5463800D0/
      DATA EISOL(91) /-120.8157940D0/, EISOLM(91) /-120.5006060D0/
      DATA BETAS(91) /  -5.5005241D0/, BETASM(91) / -18.9850440D0/
      DATA BETAP(91) / -14.6666377D0/, BETAPM(91) /  -7.9341220D0/
      DATA ZS(91)    /   1.8086650D0/, ZSM(91)    /   1.7875370D0/
      DATA ZP(91)    /   1.6851160D0/, ZPM(91)    /   1.7875370D0/
      DATA DD(91)    /   0.8236736D0/, DDM(91)    /   0.8074662D0/
      DATA QQ(91)    /   0.7268015D0/, QQM(91)    /   0.6851578D0/
      DATA BDD(91,1)    /   0.4494671D0/, AMM(91)    /   0.4494671D0/
      DATA BDD(91,2)    /   0.6082946D0/, ADM(91)    /   0.6149474D0/
      DATA BDD(91,3)    /   0.6423492D0/, AQM(91)    /   0.6685897D0/
      DATA USS(91)   / -52.0286580D0/, USSM(91)   / -52.2797450D0/
      DATA UPP(91)   / -38.7031115D0/, UPPM(91)   / -39.2055580D0/
      DATA GSS(91)   /  12.2300000D0/, GSSM(91)   /  12.2300000D0/
      DATA GSP(91)   /  11.4700000D0/, GSPM(91)   /  11.4700000D0/
      DATA GPP(91)   /  11.0800000D0/, GPPM(91)   /  11.0800000D0/
      DATA GP2(91)   /   9.8400000D0/, GP2M(91)   /   9.8400000D0/
      DATA HSP(91)   /   2.4300000D0/, HSPM(91)   /   2.4300000D0/
!
      DATA FN1(91,1) /  0.0113550D0/
      DATA FN2(91,1) /  5.0000000D0/
      DATA FN3(91,1) /  1.6000000D0/
      DATA FN1(91,2) /  0.0459240D0/
      DATA FN2(91,2) /  5.0000000D0/
      DATA FN3(91,2) /  1.8500000D0/
      DATA FN1(91,3) / -0.0200610D0/
      DATA FN2(91,3) /  5.0000000D0/
      DATA FN3(91,3) /  2.0500000D0/
      DATA FN1(91,4) / -0.0012600D0/
      DATA FN2(91,4) /  5.0000000D0/
      DATA FN3(91,4) /  2.6500000D0/
!      DATA REFPM3 ( 6)/'  C: (PM3): J. J. P. STEWART, J. COMP. CHEM.
!     1 10, 209 (1989).                '/
      DATA USSP   ( 91)/     -47.2703200D0/
      DATA UPPP   ( 91)/     -35.2877112D0/
      DATA BETASP ( 91)/      -2.3820030D0/
      DATA BETAPP ( 91)/     -14.7041325D0/
      DATA ZSP    ( 91)/       1.5650850D0/
      DATA ZPP    ( 91)/       1.8423450D0/
      DATA ALPP   ( 91)/       2.7078070D0/
      DATA EISOLP ( 91)/    -111.2299170D0/
      DATA GSSP   ( 91)/      11.2007080D0/
      DATA GSPP   ( 91)/      10.2650270D0/
      DATA GPPP   ( 91)/      10.7962920D0/
      DATA GP2P   ( 91)/       9.0425660D0/
      DATA HSPP   ( 91)/       2.2909800D0/
      DATA DDP    ( 91)/       0.8332396D0/
      DATA QQP    ( 91)/       0.6647750D0/
      DATA AMP    ( 91)/       0.4116394D0/
      DATA ADP    ( 91)/       0.5885862D0/
      DATA AQP    ( 91)/       0.7647667D0/
      DATA FN1P   ( 91,1)/       0.0501070D0/
      DATA FN2P   ( 91,1)/       6.0031650D0/
      DATA FN3P   ( 91,1)/       1.6422140D0/
      DATA FN1P   ( 91,2)/       0.0507330D0/
      DATA FN2P   ( 91,2)/       6.0029790D0/
      DATA FN3P   ( 91,2)/       0.8924880D0/
!=========================================================================
!     End of parameter declarations.
!=========================================================================
#endif /* (quantum)*/

end module am1parm

