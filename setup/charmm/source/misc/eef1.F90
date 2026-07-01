module eef1_mod
! corrections for parallel IMM1
  use chm_kinds
  use dimens_fcm
  implicit none
  character(len=*),private,parameter :: file_name   ="eef1.src"

#if KEY_ASPENER==1 /*main_eef1*/
   !
   ! For the solvation calculations
   !
   !       SLVITC(atom types)= pointers to the correct entry in VOLM,GREF
   !       VOLM(atom types)=   volumes for each atom type
   !       GREF(atom types)=   reference solvation free energies
   !       GFREE(atom types)=  solvation free energy of "free" group
   !

   !---mfc--- Needs allocation routines

   INTEGER SLVITC(maxatc)
   INTEGER NSMTH,NSMP
   INTEGER, PARAMETER :: MAXPHIDIM=200 ! Maximum size of PB box.
   INTEGER SHPEBND,PBNX,PBNY,PBNZ,PBNCEL
   real(chm_real) :: VOLM(MAXATC*2),GREF(MAXATC*2), &
         CTFSLV,GFREE(MAXATC*2),SIGW(MAXATC*2), &
         FSIGW(MAXATC*2),HREF(MAXATC*2),CPREF(MAXATC*2), &
         MYEXP(350), &
         AEMPIR,width, &
         RDEBYE,GALPHA,PSI0,GKAPPA,OFFST,TEMPR,VOLT,ACONS, &
         RCYL,RCRC,RPRB,APRB, &
         PBDX,PBDY,PBDZ,PBX0,PBY0,PBZ0,PBV,RADU

   real(chm_real),allocatable,dimension(:) :: GSOLV,VOLMI, &
         GREFI,GFREEI,SIGWI,FSIGWI, &
         VDWRI,&
         GREFI2,GFREEI2,HREFI,HREFI2, &
         CPREFI,CPREFI2,SIGWI2,FSIGWI2, &
         LAM,DFDX,DFDY,DFDZ,WELECA,PBPHI

   LOGICAL DOEEF1
   LOGICAL LMEMBR
   LOGICAL LDEBYE
   LOGICAL LGOUY,LVOLT,LINF
   LOGICAL LCYL,LPOR,LPRB,LCRC,LEPOR,LPB
   LOGICAL LCURV,LVES,LTUB,LGC4CURVE

   CHARACTER(len=4) :: SLVNT,SLVNT2

   ! Lateral pressure and dipole potential
   INTEGER NSLAB
   REAL(chm_real) PLAT(600), RLAT(100)
   REAL(chm_real) MDP,CMD,APL,PL,XPE,TSLAB,TML
   REAL(chm_real) CLP1,CLP2,CLP3,CLP4,CLP5,CLP6,LAMBDA
   LOGICAL LLAT,LDP


contains
!
! Subroutines for EEF1 by Themis Lazaridis
!
! July 1999: added support for INTE command and IMAGES
!
! Modified for compatibility with active atom selections
!          --RJP March 2000
!
! Added analytical second derivatives, I.A.
!
! NOTE:
! This code does not pass test first check when using lookup table
! for EXP function (MYEXP). But this may be OK ??
! Below are 2 commented statements (EXRI,EXRJ) which do pass this check,
! when activated. (NOTE: Deactivate previous 2 lines for speed)
!
! Code passes TEST SECOND also only when the two commented statements
! reffered to above are activated. I.A.
!
! NOTE ON THE ABOVE NOTE (May 2004, TL):
! Test first fails not because the analytical derivatives are incorrect,
! but because the NUMERICAL derivatives are incorrect when a discretized
! lookup table is used for the EXP function. There is no problem.
!
! May 2004: Implicit Membrane Model 1 added, including a Gouy-Chapman term
! for modeling charged membranes and transmembrane voltage.
!
! Oct 2013: Let's make the exact EXP the default. If anyone needs
! extra speed, they can use the lookup table by flipping the comments
! at lines ~ 675,679
!                                              Themis Lazaridis
!
! May 2010: Implicit Membrane Model 1 - Pore added (cylindrical,
! parabolic and circular pore).  Maja Mihajlovic
!
! May 2018: Implicit Membrane Model 1 extension to curved membranes. Binod Nepal
!
!
   SUBROUTINE EEF1
  use consta
  use psf
  use comand
  use exfunc
  use param
  use stream
  use string
  use number
  use parallel
  use machutil
  use machio, only:vopen

      implicit none
      CHARACTER(len=4) WRD
      INTEGER UNPAR,I
! Gouy-Chapman variables
      INTEGER VALENCE
      real(chm_real) ANFR,AREA,CONC,RVAL
      ! Poisson-Boltzmann variables
      CHARACTER(len=256) PHIFILE
      INTEGER PHIFNLEN,IUPHI
!B010606b.oe Fix for IBMSP compilation MYEXP -> THEEXP
      real(chm_real) THEEXP(350)
      DATA THEEXP/0.99998,0.99978,0.99938,0.99878,0.99798,0.99698, &
            0.99578,0.99439,0.99280,0.99102,0.98904,0.98686,0.98450, &
            0.98194,0.97919,0.97626,0.97314,0.96984,0.96635,0.96269, &
            0.95885,0.95483,0.95064,0.94627,0.94174,0.93704,0.93218, &
            0.92716,0.92199,0.91665,0.91117,0.90554,0.89976,0.89384, &
            0.88779,0.88159,0.87527,0.86882,0.86224,0.85554,0.84872, &
            0.84179,0.83475,0.82760,0.82035,0.81300,0.80555,0.79802, &
            0.79039,0.78268,0.77490,0.76703,0.75910,0.75109,0.74303, &
            0.73490,0.72671,0.71847,0.71019,0.70186,0.69349,0.68508, &
            0.67663,0.66816,0.65966,0.65114,0.64261,0.63405,0.62549, &
            0.61691,0.60834,0.59976,0.59119,0.58262,0.57406,0.56551, &
            0.55698,0.54847,0.53998,0.53151,0.52308,0.51467,0.50630, &
            0.49797,0.48967,0.48142,0.47321,0.46504,0.45693,0.44887, &
            0.44086,0.43291,0.42502,0.41719,0.40942,0.40171,0.39407, &
            0.38650,0.37900,0.37157,0.36421,0.35693,0.34972,0.34259, &
            0.33554,0.32856,0.32167,0.31486,0.30813,0.30149,0.29493, &
            0.28845,0.28206,0.27576,0.26954,0.26341,0.25737,0.25142, &
            0.24556,0.23978,0.23410,0.22850,0.22299,0.21757,0.21224, &
            0.20700,0.20185,0.19679,0.19181,0.18693,0.18213,0.17742, &
            0.17280,0.16826,0.16381,0.15945,0.15517,0.15098,0.14687, &
            0.14284,0.13890,0.13503,0.13125,0.12755,0.12393,0.12039, &
            0.11692,0.11354,0.11023,0.10699,0.10383,0.10074,0.09772, &
            0.09478,0.09190,0.08910,0.08636,0.08369,0.08109,0.07855, &
            0.07608,0.07367,0.07132,0.06903,0.06680,0.06463,0.06252, &
            0.06047,0.05847,0.05653,0.05464,0.05280,0.05102,0.04928, &
            0.04760,0.04596,0.04437,0.04283,0.04133,0.03987,0.03846, &
            0.03710,0.03577,0.03449,0.03324,0.03203,0.03086,0.02973, &
            0.02863,0.02757,0.02654,0.02555,0.02458,0.02365,0.02275, &
            0.02188,0.02104,0.02023,0.01944,0.01869,0.01795,0.01725, &
            0.01656,0.01590,0.01527,0.01465,0.01406,0.01349,0.01294, &
            0.01241,0.01190,0.01141,0.01094,0.01048,0.01004,0.00962, &
            0.00921,0.00882,0.00844,0.00808,0.00773,0.00740,0.00708, &
            0.00677,0.00647,0.00619,0.00592,0.00565,0.00540,0.00516, &
            0.00493,0.00470,0.00449,0.00429,0.00409,0.00390,0.00372, &
            0.00355,0.00339,0.00323,0.00308,0.00293,0.00279,0.00266, &
            0.00253,0.00241,0.00230,0.00219,0.00208,0.00198,0.00188, &
            0.00179,0.00170,0.00162,0.00154,0.00146,0.00139,0.00132, &
            0.00125,0.00119,0.00113,0.00107,0.00102,0.00097,0.00092, &
            0.00087,0.00082,0.00078,0.00074,0.00070,0.00066,0.00063, &
            0.00060,0.00056,0.00053,0.00051,0.00048,0.00045,0.00043, &
            0.00040,0.00038,0.00036,0.00034,0.00032,0.00031,0.00029, &
            0.00027,0.00026,0.00024,0.00023,0.00022,0.00020,0.00019, &
            0.00018,0.00017,0.00016,0.00015,0.00014,0.00014,0.00013, &
            0.00012,0.00011,0.00011,0.00010,0.00009,0.00009,0.00008, &
            0.00008,0.00007,0.00007,0.00007,0.00006,0.00006,0.00005, &
            0.00005,0.00005,0.00004,0.00004,0.00004,0.00003,0.00003, &
            0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, &
            0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, &
            0.0/

      ! On call to set-up eef1 we can allocate needed arrays. cb3
      if(.not.allocated(GSOLV)) call allocate_eef1

! assign each theexp to myexp
      DO I=1,350
         MYEXP(I)=THEEXP(I)
      ENDDO

      WRD= NEXTA4(COMLYN,COMLEN)
      IF (WRD.eq.'PRIN') THEN
         CALL SLVPRINT(COMLYN,COMLEN)
      ELSE
         DOEEF1=.true.
         CTFSLV= 81.0d0                           ! 9^2
         TEMPR= GTRMF(COMLYN,COMLEN,'TEMP',ROOMT)
         LMEMBR = (INDXA(COMLYN,COMLEN,'MEMB') .GT. 0)
         RDEBYE= GTRMF(COMLYN,COMLEN,'IONI',ZERO)
         IF (RDEBYE.GT.1.E-10) THEN
            LDEBYE=.TRUE.
! The following gives Debye length in A, assuming water as solvent
            RDEBYE= SQRT(0.0316*TEMPR/RDEBYE)
            WRITE (OUTU,*) 'Debye Length is: ', RDEBYE
         ENDIF
         SLVNT= GTRMA(COMLYN,COMLEN,'SLVT')
         IF (SLVNT.eq.'    ') SLVNT= 'WATE'              !default water
         IF (LMEMBR) THEN
            SLVNT2= GTRMA(COMLYN,COMLEN,'SLV2')
            WIDTH= GTRMF(COMLYN,COMLEN,'WIDT',THIRTY)
            NSMTH= GTRMI(COMLYN,COMLEN,'NSMT',10)
            NSMP= GTRMI(COMLYN,COMLEN,'NSMP',10)
            AEMPIR=0.85
            AEMPIR= GTRMF(COMLYN,COMLEN,'AEMP',AEMPIR)
            LGOUY = (INDXA(COMLYN,COMLEN,'GOUY') .GT. 0)      !oct03
            LINF = (INDXA(COMLYN,COMLEN,'LINF') .GT. 0)
            RADU = GTRMF(COMLYN,COMLEN,'RADU',ZERO)
            IF (RADU.GT.ZERO) THEN
               LCURV=.TRUE.
               LTUB=(INDXA(COMLYN,COMLEN,'TUBE') .GT. 0)
               IF (.NOT.LTUB) LVES=.TRUE.
            ENDIF
            LGC4CURVE = (INDXA(COMLYN,COMLEN,'GCCU') .GT. 0)
            VOLT = GTRMF(COMLYN,COMLEN,'VOLT',ZERO)
            IF (abs(VOLT).GT.ZERO) LVOLT=.TRUE.
! Ro (RCYL) for a cylindrical pore
            RCYL = GTRMF(COMLYN,COMLEN,'RCYL',ZERO)
            LCYL = (RCYL.GT.ZERO)
            IF (LCYL) LEPOR = (INDXA(COMLYN,COMLEN,'EPOR') .GT. 0)
! Ro (RPRB) & k (APRB) for a parabolic pore
            RPRB = GTRMF(COMLYN,COMLEN,'RPRB',ZERO)
            APRB = GTRMF(COMLYN,COMLEN,'APRB',ONE)
            LPRB = (RPRB.GT.ZERO)
! Ro (RCRC) for a circular pore
            RCRC = GTRMF(COMLYN,COMLEN,'RCRC',ZERO)
            LCRC = (RCRC.GT.ZERO)
            LPOR = (LCYL .OR. LPRB .OR. LCRC)
            IF (LGOUY.OR.LVOLT) THEN
               CONC= GTRMF(COMLYN,COMLEN,'CONC',ONE)
               VALENCE= GTRMI(COMLYN,COMLEN,'VALE',1)
               RVAL=FLOAT(VALENCE)
               GKAPPA= 5.622667*RVAL*SQRT(CONC/TEMPR)     !1/debye length
            ENDIF
            IF (LGOUY) THEN
               ANFR=0.3
               ANFR= GTRMF(COMLYN,COMLEN,'ANFR',ANFR)
               AREA=70.0
               AREA= GTRMF(COMLYN,COMLEN,'AREA',AREA)
               OFFST= GTRMF(COMLYN,COMLEN,'OFFS',THREE)
               PSI0=-2334.2*ANFR/AREA/SQRT(TEMPR*CONC)
               GALPHA= LOG(PSI0+SQRT(PSI0**2+1))/RVAL
               PSI0=1.7235D-4*TEMPR/RVAL*LOG(PSI0+SQRT(PSI0**2+1))
               IF (PRNLEV .GT. 9) WRITE (6,*) 'Psi0 (V) ', PSI0
               GALPHA= (EXP(GALPHA)-1.0)/(EXP(GALPHA)+1.0)
            ENDIF
            IF (LVOLT) THEN
               ACONS= VOLT/(2.+40.*GKAPPA*WIDTH)
            ENDIF
            !Parameters for PB energy
            PHIFNLEN=0
            IUPHI=GTRMI(COMLYN,COMLEN,'IUPHI',91)
            SHPEBND=GTRMI(COMLYN,COMLEN,'BDCD',0)
            CALL GTRMWD(COMLYN,COMLEN,'PHI',3,PHIFILE,256,PHIFNLEN)
            IF (PHIFNLEN .GT. 0) THEN
                CALL VOPEN(IUPHI,PHIFILE,'FORMATTED','READ',LPB,0)
                LPB=.NOT. LPB ! OPEN NOT FAIL
                IF (LPB) THEN
                    CALL RDPHI(IUPHI,PBPHI,PBNX,PBNY,PBNZ,PBNCEL,&
                               PBDX,PBDY,PBDZ,PBX0,PBY0,PBZ0)
                    LGOUY=.TRUE.
                ELSE
                    CALL WRNDIE(-5,'<IMM1 PB>','Cannot open the given PB file!')
                ENDIF
            ENDIF

            LLAT = (INDXA(COMLYN, COMLEN, 'LAT') .GT. 0)
            IF (LLAT) THEN
              CMD = 5.3    ! Compressibility modulus, Ka per slice (240/45.4)
              CMD = GTRMF(COMLYN, COMLEN, 'CMD', CMD)
              APL = 70.0
              APL = GTRMF(COMLYN, COMLEN, 'APL', APL)
              PL = 0.01    ! Protein/lipid ratio
              PL = GTRMF(COMLYN, COMLEN, 'PL', PL)
              XPE = 0.0    ! Mole fraction of PE
              XPE = GTRMF(COMLYN, COMLEN, 'XPE', XPE)
              LAMBDA = 0.5 ! expansion coefficient (actually 1-lambda)
              LAMBDA = GTRMF(COMLYN, COMLEN, 'LAMBDA', LAMBDA)

            ENDIF

            MDP = 0.0    ! Dipole potential at membrane center
            MDP = GTRMF(COMLYN, COMLEN, 'MDP', MDP)
            LDP = (MDP .GT. 0.0)

         ENDIF   !LMEMBR
         COMLYN="READ CARD "//COMLYN
         COMLEN=COMLEN+10
         CALL OPNLGU(COMLYN,COMLEN,UNPAR)
         CALL XTRANE(COMLYN,COMLEN,'EEF1')
         CALL RDSLVPAR(ATC,NATC,SLVITC,VOLM,GREF,MAXATC,COMLYN, &
               COMLEN,MXCMSZ,GFREE,UNPAR,TEMPR,SIGW,HREF,CPREF,SLVNT)

        IF (LLAT) CALL INILATPRES

         DO I=1,NATC
            FSIGW(I)= TWOPI*SQRT(PI)*SIGW(I)
            IF (SIGW(i).GT.RSMALL) THEN
               FSIGW(I)= ONE/FSIGW(I)
               SIGW(I)= ONE/SIGW(I)
            ENDIF
         ENDDO
         DO I=1,NATOM
            GREFI(I)= GREF(SLVITC(IAC(I)))
            HREFI(I)= HREF(SLVITC(IAC(I)))
            CPREFI(I)= CPREF(SLVITC(IAC(I)))
            GFREEI(I)= GFREE(SLVITC(IAC(I)))
            VOLMI(I)= VOLM(SLVITC(IAC(I)))
            SIGWI(I)= SIGW(SLVITC(IAC(I)))
            FSIGWI(I)= FSIGW(SLVITC(IAC(I)))
            VDWRI(I)= VDWR(ITC(IAC(I)))
         ENDDO
         IF (LMEMBR) THEN
            REWIND (UNIT=UNPAR)
            CALL RDSLVPAR(ATC,NATC,SLVITC,VOLM,GREF,MAXATC,COMLYN, &
                  COMLEN,MXCMSZ,GFREE,UNPAR,TEMPR,SIGW,HREF,CPREF,SLVNT2)
            DO I=1,NATC
               FSIGW(I)= 2.d+0*3.14159d+0*SQRT(3.14159d+0)*SIGW(I)
               IF (SIGW(I).GT.1.d-10) THEN
                  FSIGW(I)= 1.d0/FSIGW(I)
                  SIGW(I)= 1.d0/SIGW(I)
               ENDIF
            ENDDO
            DO I=1,NATOM
               GREFI2(I)= GREF(SLVITC(IAC(I)))
               GFREEI2(I)= GFREE(SLVITC(IAC(I)))
               HREFI2(I)= HREF(SLVITC(IAC(I)))
               CPREFI2(I)= CPREF(SLVITC(IAC(I)))
!              VOLMI(I)= VOLM(SLVITC(IAC(I)))        Volume,vdwr the same
!              VDWRI(I)= VDWR(ITC(IAC(I)))
               SIGWI2(I)= SIGW(SLVITC(IAC(I)))
               FSIGWI2(I)= FSIGW(SLVITC(IAC(I)))
            ENDDO
         ENDIF
      ENDIF
      RETURN
   END SUBROUTINE EEF1

   SUBROUTINE EEF1EN(EU,X,Y,Z,DX,DY,DZ,QECONT,ECONT,IFRSTA,NATOMX, &
         IFRSTG,NGRPX,JNBG,INBLOG,INB14,IBLO14,LINTE, &
         QSECD, &
         DD1,IUPT)
!
!     EU   - Energy to be returned
!     X,Y,Z - Coordinates for energy evaluation
!     DX,DY,DZ - Forces.
!     QECONT - Flag for analysis (0-no analysis,>0=analysis)
!     ECONT(NATOMX) - Analysis array to be filled if QECONT>0.
!     NATOMX - Number of atoms
!
!     Author: T.Lazaridis
!     Second derivatives by I.A.
!
  use number

  use psf
  use param
  use parallel
  use stream
  use actclus_mod,only: qlbycc,acaflg
! ARD 00-10-14
  use block_ltm
      implicit none
      real(chm_real) EU
      real(chm_real) X(*),Y(*),Z(*)
      real(chm_real) DX(*),DY(*),DZ(*)
      LOGICAL QECONT
      real(chm_real) ECONT(:)
      INTEGER IFRSTA,NATOMX,IFRSTG,NGRPX
      INTEGER JNBG(:),INBLOG(:),INB14(:),IBLO14(:)
      LOGICAL LINTE
      LOGICAL QSECD
      real(chm_real) F(NATOMX)
      real(chm_real),optional :: DD1(*)
      integer,optional :: IUPT(*)

      INTEGER LBIN
      INTEGER I,J,K
      real(chm_real) EXRI,EXRJ,XDISTI,XDISTJ,RSQ,ARSQ,RDIST, &
            ARDIST,ARDIST3
      real(chm_real) GRI,VI,GRJ,VJ
      real(chm_real) VGIJ,VGJI,FDERI,FDERJ,GSI,GSJ
      real(chm_real) SIGI,SIGJ,FSIGI,FSIGJ
      real(chm_real) TMP,DPSI
      INTEGER NB,ITEMP,NPR,IS,IQ,JSLC,JQLC,JRS,JRSPR,NXI,NXIMAX,JSX
      INTEGER IRS,INBX,ICNT
      LOGICAL LEXCL
      real(chm_real) DXI,DYI,DZI,DXIT,DYIT,DZIT,ACC
      CHARACTER(len=1) STR1
      CHARACTER(len=3) STR3
      real(chm_real) FF
      real(chm_real) EGOUY,TMPE,PSI,CGGOUY,EVOLT
! PORE MODEL
      real(chm_real) RC(natomx),DFDR(natomx)
      real(chm_real) DGDR,G
      real(chm_real) RPOR,DRDZ(natomx),TMP2
! For EPORE
      real(chm_real) AREA,FR,DFRDX,DFRDY,FZ,DFZDZ
      real(chm_real) AREA_CONTB,CONST1,BETA,GAMMA,SIGMA,EPORE
! PB energy
      real(chm_real) SHPEV,DSHPEDX,DSHPEDY,DSHPEDZ
      real(chm_real) BV,DPBDX,DPBDY,DPBDZ,PMX,PMY,PMZ
      LOGICAL LOUTSIDE ! if outside the PB region.
! Curved membrane
      real(chm_real) TMPX,TMPY,TMPZ,TMPR, theta, phi, DJ
      real(chm_real) theta1,theta2,phi1,phi2,TMPR1,TMPR2, DK1,DK2
! Local variables for second derivatives. I.A.
      real(chm_real) EFDERI,EFDERJ,FCTI,FCTJ,ARDIST6,ETMP
      real(chm_real) AXX,AYY,AZZ,AXY,AXZ,AYZ
      INTEGER IADD,II,JJ
      LOGICAL LSECD,LCHA
! Variables for approximate solutions of the Poisson–Boltzmann Equation in Curved Membranes
! Tuinier, Approximate solutions to the PB equation in spherical and cylindrical geometry.
! Journal of Colloid and Interface Science. 2003, 258 (1), 45-49.
! We have extended the approximation for inside a sphere and a cylinder. B.Nepal,A.Sepehri
! If "LGC4CURVE" is True, the Gouy-Chapman solution is used, good for low curvatures
      real(chm_real) OMEG_SPH,OMEG_CYL
      real(chm_real) PSI_LOW,PSI_HIGH,RADU_IN,RADU_OUT
      real(chm_real) DIFF_LOW,DIFF_HIGH,TANH_PSI0
      real(chm_real) DTANH_PSI0,TANH_PLUS,TANH_MINUS
      real(chm_real) GALPHA_PLUS,GALPHA_MINUS
      real(chm_real) PSI0_DIMLESS
      real(chm_real) EXTRA_TERM,DEXTRA_TERM
!
! -----------------------------
      LSECD = QSECD .and. present(DD1) .and. present(IUPT)
      EU = ZERO

! For EPORE
       IF(LCYL .AND. LEPOR) THEN
           GAMMA = 50.05*0.001439    !interfacial tension in kcal/mol A2
           CONST1 = 2*3.141*RCYL*WIDTH*GAMMA
           SIGMA=0.7/2.35842         !0.7 is FWHM of Gaussian. Sigma the st dev
           BETA=2*SIGMA**2
           AREA_CONTB = 0.0
       ENDIF

! Solvation calculation
      DO I=IFRSTA,NATOMX
         GSOLV(I)= ZERO
         IF (LMEMBR) THEN
            IF (LCURV) THEN
               IF (LTUB) THEN
                  TMPR= SQRT(Y(I)**2+Z(I)**2)
               ELSE
                  TMPR= SQRT(X(I)**2+Y(I)**2+Z(I)**2)
                  TMPX = X(I)/TMPR
               ENDIF
               TMP = ABS(TMPR-RADU)*2.0/WIDTH
               TMPY = Y(I)/TMPR
               TMPZ = Z(I)/TMPR
            ELSE
               TMP = ABS(Z(I))*2.0/WIDTH
            ENDIF
            F(I)= TMP**NSMTH/(1.0+TMP**NSMTH)
            IF (LINF) THEN  ! Linear switching function for the charged sidechains
                LCHA=.FALSE.
                STR3= ATC(IAC(I))
                IF ((STR3.EQ.'NH3').OR.(STR3.EQ.'NC2').OR.(STR3.EQ.'OC ')) LCHA=.TRUE.
                IF (LCHA) F(I)= MIN(TMP,1.0)
            ENDIF
! PORE MODEL
            FF = F(I)
            IF (LPOR) THEN
              IF (LCYL) THEN
                 RPOR=RCYL
              ENDIF
! parabolic pore: R=Ro+k(z'**2)
! k is APRB
              IF (LPRB) THEN
                 RPOR=RPRB+APRB*TMP**2
              ENDIF

! circular pore: R=Ro+T/2*(1-sqrt(1-z'**2))
              IF (LCRC) THEN
! if z' .gt. 1, R is NAN due to the sqrt of a negative
! number. To avoid this, z' is set to 1 (this is TMP2) when z' > 1
! (this is TMP)
                 TMP2=TMP
                 IF (TMP .GT. ONE) TMP2=ONE
                 RPOR=RCRC+(WIDTH/TWO)*(ONE-SQRT(ONE-TMP2**2))
              ENDIF
              RC(I)= SQRT(X(I)**2+Y(I)**2)/RPOR
              G = ONE- RC(I)**NSMP/(ONE+RC(I)**NSMP)
              F(I) = F(I) + G -F(I)*G
              IF (LCYL .AND. LEPOR) THEN
                   AREA = 3.141*VDWRI(I)**2
                   FR  = EXP(-(RC(I)*RPOR - RPOR)**2/BETA) !gaussian
                   DFRDX = -2*X(I)*(RC(I)*RPOR - RPOR)*FR/(BETA*RC(I)*RPOR)
                   DFRDY = -2*Y(I)*(RC(I)*RPOR - RPOR)*FR/(BETA*RC(I)*RPOR)
                   FZ = 1 - FF
                   DFZDZ = -(2.*sign(ONE,Z(I))/WIDTH)* &
                   FLOAT(NSMTH)*TMP**(NSMTH-1)/(1.0+TMP**NSMTH)**2
                   AREA_CONTB = AREA_CONTB + AREA*FR*FZ*GAMMA
!                   write (6,*) 'area,fr,fz,gamma',area,fr,fz,gamma
#if KEY_PARALLEL==1
#if KEY_VIBPARA==0
            if (mynod.eq.0) then
#endif
#endif
                   DX(I) = DX(I) -  AREA*FZ*DFRDX*GAMMA
                   DY(I) = DY(I) -  AREA*FZ*DFRDY*GAMMA
                   DZ(I) = DZ(I) -  AREA*FR*DFZDZ*GAMMA
#if KEY_PARALLEL==1
#if KEY_VIBPARA==0
            endif
#endif
#endif
              ENDIF
            ENDIF
            LAM(I)= AEMPIR + (1.0-AEMPIR)*F(I)
            IF (LCURV) THEN
              DFDZ(I) = (2.*sign(ONE,(TMPR-RADU))/WIDTH)* &
                 FLOAT(NSMTH)*TMP**(NSMTH-1)/(1.0+TMP**NSMTH)**2
              DJ= (GREFI(I)-GREFI2(I))*DFDZ(I)
            ELSE
              DFDZ(I)= (2.*sign(ONE,Z(I))/WIDTH)* &
                 FLOAT(NSMTH)*TMP**(NSMTH-1)/(1.0+TMP**NSMTH)**2
            ENDIF
            IF (LINF .AND. LCHA) THEN
              IF (TMP.GT.1.0) THEN
                DFDZ(I)= ZERO
              ELSE
                DFDZ(I)= (2.*sign(ONE,Z(I))/WIDTH)
              ENDIF
            ENDIF
! PORE MODEL
            IF (LPOR) THEN
! for all pores
              DFDZ(I)= DFDZ(I)*(ONE-G)
              DGDR= -FLOAT(NSMP)*RC(I)**(NSMP-1)/(ONE+RC(I)**NSMP)**2
              DFDR(I)= DGDR*(ONE-FF)
! parabolic pore
! DRDZ is dr'/dz where r'=r/RPOR
              IF (LPRB) THEN
                 DRDZ(I)= -TWO*(TWO*sign(ONE,Z(I))/WIDTH)* &
                      RC(I)*APRB*TMP/RPOR
                 DFDZ(I)= DFDZ(I)+DFDR(I)*DRDZ(I)
              ENDIF
! circular pore
              IF (LCRC) THEN
! To avoid devision by 0 and sqrt of a negative number,
! DRDZ is set to 0 at z'>= 1
                  IF ( TMP .LT. ONE) THEN
                    DRDZ(I)= -RC(I)*TMP/(RPOR*SQRT(ONE-TMP**2))
                 ELSE
                    DRDZ(I) = ZERO
                 ENDIF
                 DFDZ(I)= DFDZ(I)+DFDR(I)*DRDZ(I)
              ENDIF
            ENDIF
!           The contribution to the z derivative due to self-energy
#if KEY_PARALLEL==1
#if KEY_VIBPARA==0
            if (mynod.eq.0) then
#endif
#endif
            IF (LCURV) THEN
               IF (LTUB) THEN
                  TMPR= SQRT(Y(I)**2+Z(I)**2)
                  DY(I)=DY(I)+DJ*Y(I)/TMPR
                  DZ(I)=DZ(I)+DJ*Z(I)/TMPR
               ELSE
                  TMPR= SQRT(X(I)**2+Y(I)**2+Z(I)**2)
                  DX(I)= DX(I)+DJ*X(I)/TMPR
                  DY(I)=DY(I)+DJ*Y(I)/TMPR
                  DZ(I)=DZ(I)+DJ*Z(I)/TMPR
               ENDIF
            ELSE
               DZ(I)= DZ(I)+ DFDZ(I)*(GREFI(I)-GREFI2(I))
            ENDIF
#if KEY_PARALLEL==1
#if KEY_VIBPARA==0
            endif
#endif
#endif
! all pores
            IF (LPOR) THEN
              DFDX(I)= DFDR(I)*X(I)/RC(I)/RPOR**2
              DFDY(I)= DFDR(I)*Y(I)/RC(I)/RPOR**2
#if KEY_PARALLEL==1
#if KEY_VIBPARA==0
            if (mynod.eq.0) then
#endif
#endif
              DX(I)= DX(I) + (GREFI(I)-GREFI2(I))*DFDX(I)
              DY(I)= DY(I) + (GREFI(I)-GREFI2(I))*DFDY(I)
#if KEY_PARALLEL==1
#if KEY_VIBPARA==0
            endif
#endif
#endif
            ENDIF
         ENDIF
      ENDDO
! add EPORE
      IF (LEPOR) THEN
        EPORE = CONST1 - AREA_CONTB
#if KEY_PARALLEL==1
#if KEY_VIBPARA==0
            if (mynod.eq.0) then
#endif
#endif
        EU = EU + EPORE
        IF (PRNLEV.GT.9) WRITE(6,*) 'Epore is',EPORE,'Naked pore:',CONST1
#if KEY_PARALLEL==1
#if KEY_VIBPARA==0
            endif
#endif
#endif
      ENDIF

!     LOOP OVER PAIRS OF GROUPS IN GROUPS LIST
!
      EGOUY=0.d0
      EVOLT=0.d0
      NB=0
      ITEMP=0
      LOUTSIDE=.TRUE.
      DO IRS=IFRSTG,NGRPX
         NPR=INBLOG(IRS)-ITEMP
         ITEMP=INBLOG(IRS)
         IS=IGPBS(IRS)+1
         IQ=IGPBS(IRS+1)
! Gouy-Chapman term
! This calculation uses the full charge of ionic residues. They are
! recognized only based on partial charge, so the partial charge
! has to be unique in the topology file!
#if KEY_PARALLEL==1
#if KEY_VIBPARA==0
            if (mynod.eq.0) then
#endif
#endif
         IF ((LGOUY.OR.LVOLT) .AND. .NOT. LINTE) THEN
            DO I=IS,IQ
               IF (abs(CG(I)+0.9).LT.1d-6) THEN          !Lys or Nt
                  CGGOUY=+0.1d0
               ELSE IF (abs(CG(I)+0.121).LT.1d-6) THEN   !ARG
                  CGGOUY=+0.379d0
               ELSE IF (abs(CG(I)-0.45).LT.1d-6) THEN    !HSC/HSP,GLYP,PROP
                  CGGOUY=+0.95d0
               ELSE IF (abs(CG(I)-1.0).LT.1d-6) THEN     !ASP,GLU,Ct
                  CGGOUY= 0.d0
               ELSE
                  CGGOUY=CG(I)
               ENDIF
               IF (LGOUY) THEN
                   IF (LPB) THEN
                       CALL GETPHI(PBPHI,PBNX,PBNY,PBNZ,PBNCEL,PBDX,PBDY,PBDZ, &
                                   PBX0,PBY0,PBZ0,X(I),Y(I),Z(I),PBV, &
                                   DPBDX,DPBDY,DPBDZ,LOUTSIDE)
                   ENDIF
                   IF (.NOT. LOUTSIDE) THEN
                       ! If not outside PB calculation region
                       EGOUY= EGOUY+1.9899e-3*TEMPR*CGGOUY*PBV
                       WELECA(I)=1.9899e-3*TEMPR*CGGOUY*PBV
                       DX(I)=DX(I)+1.9899e-3*TEMPR*DPBDX*CGGOUY
                       DY(I)=DY(I)+1.9899e-3*TEMPR*DPBDY*CGGOUY
                       DZ(I)=DZ(I)+1.9899e-3*TEMPR*DPBDZ*CGGOUY
                   ELSE
                        ! Outside then use Gouy-Chapman
                      IF (LCURV) THEN
                         IF (LTUB) THEN
                            TMPR=SQRT(Y(I)**2+Z(I)**2)
                            TMP=ABS(TMPR-RADU)-WIDTH/2.0-OFFST
                         ELSE
                             TMPR=SQRT(X(I)**2+Y(I)**2+Z(I)**2)
                             TMP=ABS(TMPR-RADU)-WIDTH/2.0-OFFST
                         ENDIF
                      ELSE
                         TMP=ABS(Z(I))-WIDTH/2.0-OFFST
                      ENDIF
                      IF (TMP .LE. 0.0) THEN
                         PSI=PSI0*23.06
                      ELSE
                         TMPE=EXP(-GKAPPA*TMP)
                         PSI=3.974d-3*TEMPR*LOG((1.+GALPHA*TMPE)/(1.-GALPHA*TMPE))
                         IF (LCURV) THEN
                           IF (LGC4CURVE) THEN ! Use Gouy-Chapman model for cylinder and sphere
                             DJ = CGGOUY*3.974d-3*TEMPR*(1.-GALPHA*TMPE)/ &
              (1.+GALPHA*TMPE)*2.0*GALPHA/(1.-GALPHA*TMPE)**2*(-GKAPPA*TMPE* &
                 SIGN(ONE,(TMPR-RADU)))
                             IF (LTUB) THEN
                                TMPR= SQRT(Y(I)**2+Z(I)**2)
                                DY(I)=DY(I)+DJ*Y(I)/TMPR
                                DZ(I)=DZ(I)+DJ*Z(I)/TMPR
                             ELSE
                                TMPR= SQRT(X(I)**2+Y(I)**2+Z(I)**2)
                                DX(I)=DX(I)+DJ*X(I)/TMPR
                                DY(I)=DY(I)+DJ*Y(I)/TMPR
                                DZ(I)=DZ(I)+DJ*Z(I)/TMPR
                             ENDIF
                           ELSE
!  Use the approximate solution for cylinder and sphere
                             PSI0_DIMLESS = 0.5*LOG((1.+GALPHA)/(1.-GALPHA)) ! PSI0/4
                             IF (LTUB) THEN
                               TMPR= SQRT(Y(I)**2+Z(I)**2)
                               OMEG_CYL = 3.0
                               IF (TMPR .GT. RADU) THEN ! Outside a cylinder
                                  RADU_OUT = TMPR - TMP
                                  OMEG_CYL = exp(-OMEG_CYL*GKAPPA*RADU_OUT)
                                  GALPHA_PLUS = 1.+GALPHA*TMPE
                                  GALPHA_MINUS = 1.-GALPHA*TMPE
                                  EXTRA_TERM = BESSK0(GKAPPA*TMPR)/ &
                         (BESSK0(GKAPPA*RADU_OUT)*TMPE)
                                  PSI_LOW = EXTRA_TERM*LOG(GALPHA_PLUS/ &
                         GALPHA_MINUS)
                                  DEXTRA_TERM = (-GKAPPA*TMPE*(BESSK1(GKAPPA*TMPR)- &
                         BESSK0(GKAPPA*TMPR)))/(BESSK0(GKAPPA*RADU_OUT)*TMPE**2)
                                  DIFF_LOW = DEXTRA_TERM*LOG(GALPHA_PLUS/ &
                         GALPHA_MINUS)+EXTRA_TERM*(-2.*GALPHA*GKAPPA*TMPE*GALPHA_MINUS/ &
                         (GALPHA_PLUS*GALPHA_MINUS**2))
                                  EXTRA_TERM = BESSK0(GKAPPA*TMPR)/ &
                         BESSK0(GKAPPA*RADU_OUT)
                                  DEXTRA_TERM = -GKAPPA*BESSK1(GKAPPA*TMPR)/ &
                         BESSK0(GKAPPA*RADU_OUT)
                                  TANH_PSI0 = TANH(PSI0_DIMLESS*EXTRA_TERM)
                                  TANH_PLUS = 1.+TANH_PSI0*TMPE
                                  TANH_MINUS = 1.-TANH_PSI0*TMPE
                                  PSI_HIGH = LOG(TANH_PLUS/TANH_MINUS)/TMPE
                                  PSI=3.974d-3*TEMPR*(PSI_LOW*(1.-OMEG_CYL)+ &
                         PSI_HIGH*OMEG_CYL)
                                  DTANH_PSI0 = PSI0_DIMLESS*DEXTRA_TERM* &
                         (1.-TANH_PSI0**2)
                                  DIFF_HIGH = (2.*TMPE*(DTANH_PSI0- &
                         TANH_PSI0*GKAPPA)*TANH_MINUS)/(TANH_PLUS*TANH_MINUS**2*TMPE)+ &
                         GKAPPA*LOG(TANH_PLUS/TANH_MINUS)/TMPE
                                  DJ = CGGOUY*3.974d-3*TEMPR*(DIFF_LOW* &
                         (1.-OMEG_CYL)+DIFF_HIGH*OMEG_CYL)*(SIGN(ONE,(TMPR-RADU)))
                               ELSE ! Inside a cylinder
                                  RADU_IN = TMPR + TMP
                                  OMEG_CYL = exp(-OMEG_CYL*GKAPPA*RADU_IN)
                                  GALPHA_PLUS = 1.+GALPHA*TMPE
                                  GALPHA_MINUS = 1.-GALPHA*TMPE
                                  EXTRA_TERM = BESSI0(GKAPPA*TMPR)/ &
                         (BESSI0(GKAPPA*RADU_IN)*TMPE)
                                  PSI_LOW = EXTRA_TERM*LOG(GALPHA_PLUS/ &
                         GALPHA_MINUS)
                                  DEXTRA_TERM = (-GKAPPA*TMPE*(BESSI1(GKAPPA*TMPR)- &
                         BESSI0(GKAPPA*TMPR)))/(BESSI0(GKAPPA*RADU_IN)*TMPE**2)
                                  DIFF_LOW = DEXTRA_TERM*LOG(GALPHA_PLUS/ &
                         GALPHA_MINUS)+EXTRA_TERM*(-2.*GALPHA*GKAPPA*TMPE*GALPHA_MINUS/ &
                         (GALPHA_PLUS*GALPHA_MINUS**2))
                                  EXTRA_TERM = BESSI0(GKAPPA*TMPR)/ &
                         BESSI0(GKAPPA*RADU_IN)
                                  DEXTRA_TERM = -GKAPPA*BESSI1(GKAPPA*TMPR)/ &
                         BESSI0(GKAPPA*RADU_IN)
                                  TANH_PSI0 = TANH(PSI0_DIMLESS*EXTRA_TERM)
                                  TANH_PLUS = 1.+TANH_PSI0*TMPE
                                  TANH_MINUS = 1.-TANH_PSI0*TMPE
                                  PSI_HIGH = LOG(TANH_PLUS/TANH_MINUS)/TMPE
                                  PSI=3.974d-3*TEMPR*(PSI_LOW*(1.-OMEG_CYL)+ &
                         PSI_HIGH*OMEG_CYL)
                                  DTANH_PSI0 = PSI0_DIMLESS*DEXTRA_TERM* &
                         (1.-TANH_PSI0**2)
                                  DIFF_HIGH = (2.*TMPE*(DTANH_PSI0- &
                         TANH_PSI0*GKAPPA)*TANH_MINUS)/(TANH_PLUS*TANH_MINUS**2*TMPE)+ &
                         GKAPPA*LOG(TANH_PLUS/TANH_MINUS)/TMPE
                                  DJ = CGGOUY*3.974d-3*TEMPR*(DIFF_LOW* &
                         (1.-OMEG_CYL)+DIFF_HIGH*OMEG_CYL)*(SIGN(ONE,(TMPR-RADU)))
                               ENDIF
                                 DY(I)=DY(I)+DJ*Y(I)/TMPR
                                 DZ(I)=DZ(I)+DJ*Z(I)/TMPR
                             ELSE
                               TMPR= SQRT(X(I)**2+Y(I)**2+Z(I)**2)
                               OMEG_SPH = 0.3
                               IF (TMPR .GT. RADU) THEN  ! Outside a sphere
                                  RADU_OUT = TMPR - TMP
                                  OMEG_SPH = exp(-OMEG_SPH*GKAPPA*RADU_OUT)
                                  GALPHA_PLUS = 1.+GALPHA*TMPE
                                  GALPHA_MINUS = 1.-GALPHA*TMPE
                                  EXTRA_TERM = RADU_OUT/TMPR
                                  PSI_LOW = EXTRA_TERM*LOG(GALPHA_PLUS/ &
                         GALPHA_MINUS)
                                  DEXTRA_TERM = -RADU_OUT/TMPR**2
                                  DIFF_LOW = DEXTRA_TERM*LOG(GALPHA_PLUS/ &
                         GALPHA_MINUS)+EXTRA_TERM*(-2.*GALPHA*GKAPPA*TMPE*GALPHA_MINUS/ &
                         (GALPHA_PLUS*GALPHA_MINUS**2))
                                  TANH_PSI0 = TANH(PSI0_DIMLESS*EXTRA_TERM)
                                  TANH_PLUS = 1.+TANH_PSI0*TMPE
                                  TANH_MINUS = 1.-TANH_PSI0*TMPE
                                  PSI_HIGH = LOG(TANH_PLUS/TANH_MINUS)
                                  PSI=3.974d-3*TEMPR*(PSI_LOW*(1.-OMEG_SPH)+ &
                         PSI_HIGH*OMEG_SPH)
                                  DTANH_PSI0 = PSI0_DIMLESS*DEXTRA_TERM* &
                         (1.-TANH_PSI0**2)
                                  DIFF_HIGH = (2.*TMPE*(DTANH_PSI0- &
                         TANH_PSI0*GKAPPA)*TANH_MINUS)/(TANH_PLUS*TANH_MINUS**2)
                                  DJ = CGGOUY*3.974d-3*TEMPR*(DIFF_LOW* &
                         (1.-OMEG_SPH)+DIFF_HIGH*OMEG_SPH)*(SIGN(ONE,(TMPR-RADU)))
                               ELSE ! Inside a sphere
                                  RADU_IN = TMPR + TMP
                                  OMEG_SPH = exp(-OMEG_SPH*GKAPPA*RADU_IN)
                                  GALPHA_PLUS = 1.+GALPHA*TMPE
                                  GALPHA_MINUS = 1.-GALPHA*TMPE
                                  EXTRA_TERM = (SINH(GKAPPA*TMPR)/TMPR)/ &
                         (SINH(GKAPPA*RADU_IN)*TMPE/RADU_IN)
                                  PSI_LOW = EXTRA_TERM*LOG(GALPHA_PLUS/ &
                         GALPHA_MINUS)
                                  DEXTRA_TERM = (1./(SINH(GKAPPA*RADU_IN)/ &
                         RADU_IN))*((-GKAPPA*COSH(GKAPPA*TMPR)*TMPR*TMPE+ &
                         SINH(GKAPPA*TMPR)*TMPE*(1.+GKAPPA*TMPR))/(TMPR**2*TMPE**2))
                                  DIFF_LOW = DEXTRA_TERM*LOG(GALPHA_PLUS/ &
                         GALPHA_MINUS)+EXTRA_TERM*(-2.*GALPHA*GKAPPA*TMPE*GALPHA_MINUS/ &
                         (GALPHA_PLUS*GALPHA_MINUS**2))
                                  TANH_PSI0 = TANH(PSI0_DIMLESS*EXTRA_TERM)
                                  TANH_PLUS = 1.+TANH_PSI0*TMPE
                                  TANH_MINUS = 1.-TANH_PSI0*TMPE
                                  PSI_HIGH = LOG(TANH_PLUS/TANH_MINUS)
                                  PSI=3.974d-3*TEMPR*(PSI_LOW*(1.-OMEG_SPH)+ &
                         PSI_HIGH*OMEG_SPH)
                                  DTANH_PSI0 = PSI0_DIMLESS*DEXTRA_TERM* &
                         (1.-TANH_PSI0**2)
                                  DIFF_HIGH = (2.*TMPE*(DTANH_PSI0- &
                         TANH_PSI0*GKAPPA)*TANH_MINUS)/(TANH_PLUS*TANH_MINUS**2)
                                  DJ = CGGOUY*3.974d-3*TEMPR*(DIFF_LOW* &
                         (1.-OMEG_SPH)+DIFF_HIGH*OMEG_SPH)*(SIGN(ONE,(TMPR-RADU)))
                               ENDIF
                                  DX(I)=DX(I)+DJ*X(I)/TMPR
                                  DY(I)=DY(I)+DJ*Y(I)/TMPR
                                  DZ(I)=DZ(I)+DJ*Z(I)/TMPR
                             ENDIF ! IF LTUB
                           ENDIF ! IF LGC4CURVE
                         ELSE ! IF LCURV
                           DZ(I)=DZ(I)+CGGOUY*3.974d-3*TEMPR*(1.-GALPHA*TMPE) &
                           /(1.+GALPHA*TMPE)*2.0*GALPHA/(1.-GALPHA*TMPE)**2* &
                           (-GKAPPA*TMPE*SIGN(ONE,Z(I)))
                         ENDIF
                      ENDIF
                      EGOUY= EGOUY+PSI*CGGOUY
                      WELECA(I)=PSI*CGGOUY
                   ENDIF
               ENDIF
               IF (LVOLT) THEN
                  IF (Z(I).LE.-WIDTH/2) THEN
                     TMP=Z(I)+WIDTH/2
                     TMPE=EXP(GKAPPA*TMP)
                     PSI=ACONS*TMPE*23.06
                     DPSI=PSI*GKAPPA
                  ELSE IF (Z(I).GE.WIDTH/2) THEN
                     TMP=Z(I)-WIDTH/2
                     TMPE=EXP(-GKAPPA*TMP)
                     PSI=(VOLT-ACONS*TMPE)*23.06
                     DPSI=GKAPPA*ACONS*TMPE*23.06
                  ELSE
                     TMP=Z(I)+WIDTH/2
                     PSI=ACONS*(40.*GKAPPA*TMP+1.0)*23.06
                     DPSI=GKAPPA*ACONS*40.0*23.06
                  ENDIF
                  EVOLT=EVOLT+PSI*CGGOUY
                  DZ(I)=DZ(I)+DPSI*CGGOUY
                  WELECA(I)=PSI*CGGOUY
               ENDIF
            ENDDO
         ENDIF
#if KEY_PARALLEL==1
#if KEY_VIBPARA==0
            endif
#endif
#endif
! Gouy end
         DO JRSPR=1,NPR
            NB=NB+1
            JRS=JNBG(NB)
            LEXCL=(JRS.LT.0)
            IF(LEXCL) JRS=-JRS
            JSLC=IGPBS(JRS)+1
            JQLC=IGPBS(JRS+1)
!
!     PROCESS THIS GROUP PAIR
!
            DO I=IS,IQ

               STR1= ATC(IAC(I))
               IF (STR1.EQ.'H') CYCLE  !skip hydrogens
               IF (LEXCL) THEN
                  IF (I.GT.1) THEN
                     NXI=IBLO14(I-1)+1
                  ELSE
                     NXI=1
                  ENDIF
                  NXIMAX=IBLO14(I)
                  IF (IS.EQ.JSLC) THEN
                     JSX= I+1
                  ELSE
                     JSX= JSLC
                  ENDIF
               ELSE
                  JSX= JSLC
               ENDIF
               VI= VOLMI(I)
               IF (LMEMBR) THEN
                  GRI= GFREEI2(I)+ F(I)*(GFREEI(I)-GFREEI2(I))
                  SIGI= 1./SIGWI2(I)+ F(I)*(1./SIGWI(I)-1./SIGWI2(I))
                  SIGI= 1./SIGI
                  FSIGI= 1./FSIGWI2(I)+ F(I)*(1./FSIGWI(I)-1./FSIGWI2(I))
                  FSIGI= 1./FSIGI
               ELSE
                  GRI= GFREEI(I)
                  SIGI= SIGWI(I)
                  FSIGI= FSIGWI(I)
               ENDIF
               loop80: DO J=JSX,JQLC
                  STR1= ATC(IAC(J))
                  IF (STR1.EQ.'H') CYCLE loop80
                  IF (LEXCL) THEN
!  CHECK ATOM EXCLUSION LIST FOR EXCLUSIONS
                     DO WHILE (NXI .LE. NXIMAX)
                        IF (INB14(NXI).LT.0) THEN
                           INBX=-INB14(NXI)
                           IF(J.LE.INBX) EXIT
                        ELSE
                           IF(J.EQ.INB14(NXI)) CYCLE loop80
                           IF(J.LT.INB14(NXI)) EXIT
                        ENDIF
                        NXI=NXI+1
                     ENDDO
                  ENDIF
! not excluded, calculate solvation
                  DXI=X(I)-X(J)
                  DYI=Y(I)-Y(J)
                  DZI=Z(I)-Z(J)
                  RSQ=DXI*DXI+DYI*DYI+DZI*DZI
                  IF (RSQ.GT.CTFSLV) CYCLE loop80
                  ARSQ=ONE/RSQ
                  RDIST= SQRT(RSQ)
                  ARDIST= ONE/RDIST
                  ARDIST3= TWO*ARDIST*ARDIST*ARDIST
                  IF(LSECD) ARDIST6= ARSQ*ARSQ*ARSQ

                  VJ= VOLMI(J)
                  IF (LMEMBR) THEN
                     GRJ= GFREEI2(J)+ F(J)*(GFREEI(J)-GFREEI2(J))
                     SIGJ= 1./SIGWI2(J)+ F(J)*(1./SIGWI(J)-1./SIGWI2(J))
                     FSIGJ=1./FSIGWI2(J)+F(J)*(1./FSIGWI(J)-1./FSIGWI2(J))
                     SIGJ= 1./SIGJ
                     FSIGJ= 1./FSIGJ
                  ELSE
                     GRJ= GFREEI(J)
                     SIGJ= SIGWI(J)
                     FSIGJ= FSIGWI(J)
                  ENDIF
                  XDISTI= ABS((RDIST-VDWRI(I))*SIGI) !reduced distance
!                  LBIN= IDINT(XDISTI*100)+1
!                  EXRI= MYEXP(LBIN)
                  EXRI=EXP(-XDISTI*XDISTI)
                  XDISTJ= ABS((RDIST-VDWRI(J))*SIGJ)
!                  LBIN= IDINT(XDISTJ*100)+1
!                  EXRJ= MYEXP(LBIN)
                  EXRJ=EXP(-XDISTJ*XDISTJ)
                  VGJI= VJ*GRI
                  VGIJ= VI*GRJ

! ARD 00-10-14
! Moved EU sum into loop for Monte Carlo which uses only a partial
! INBLOG list.  Also put in NOFORC flag to skip force calculation.
! Note:  if IFRSTA>1 and NATOMX<NATOM not all GSOLV will be initialized.
                  GSI = VGJI*EXRI*ARSQ*FSIGI
                  GSJ = VGIJ*EXRJ*ARSQ*FSIGJ
                  GSOLV(I)= GSOLV(I) - GSI
                  GSOLV(J)= GSOLV(J) - GSJ
                  EU = EU - GSI - GSJ

#if KEY_BLOCK==1
                  IF (.NOT. NOFORC) THEN
#endif

                     FDERI= EXRI*(XDISTI*SIGI+ARDIST)*ARDIST3*FSIGI
                     FDERJ= EXRJ*(XDISTJ*SIGJ+ARDIST)*ARDIST3*FSIGJ
                     TMP= FDERI*VGJI+FDERJ*VGIJ
                     DXIT= DXI*TMP
                     DYIT= DYI*TMP
                     DZIT= DZI*TMP
                     DX(I)= DX(I)+ DXIT
                     DY(I)= DY(I)+ DYIT
                     DZ(I)= DZ(I)+ DZIT
                     DX(J)= DX(J)- DXIT
                     DY(J)= DY(J)- DYIT
                     DZ(J)= DZ(J)- DZIT
! The contributions to z derivatives
                     IF (LMEMBR) THEN
                        IF (LCURV) THEN
                          IF (LTUB) THEN
                            TMPR1= SQRT(Y(I)**2+Z(I)**2)
                            TMPR2= SQRT(Y(J)**2+Z(J)**2)

                            DK1 = VJ*ARSQ*FSIGI*EXRI*DFDZ(I)* &
                        (GFREEI(I)-GFREEI2(I)+ GRI*SIGI*(2.*XDISTI**2 -1.0)* &
                              (1./SIGWI(I)-1./SIGWI2(I)))
                            DK2= VI*ARSQ*FSIGJ*EXRJ*DFDZ(J)* &
                        (GFREEI(J)-GFREEI2(J)+ GRJ*SIGJ*(2.*XDISTJ**2 -1.0)* &
                              (1./SIGWI(J)-1./SIGWI2(J)))
                            DY(I)=DY(I)-DK1*Y(I)/TMPR1
                            DZ(I)=DZ(I)-DK1*Z(I)/TMPR1
                            DY(J)=DY(J)-DK2*Y(J)/TMPR2
                            DZ(J)=DZ(J)-DK2*Z(J)/TMPR2
                          ELSE
                            TMPR1= SQRT(X(I)**2+Y(I)**2+Z(I)**2)

                            TMPR2= SQRT(X(J)**2+Y(J)**2+Z(J)**2)
                            DK1 = VJ*ARSQ*FSIGI*EXRI*DFDZ(I)* &
                        (GFREEI(I)-GFREEI2(I)+ GRI*SIGI*(2.*XDISTI**2 -1.0)* &
                              (1./SIGWI(I)-1./SIGWI2(I)))
                            DK2= VI*ARSQ*FSIGJ*EXRJ*DFDZ(J)* &
                        (GFREEI(J)-GFREEI2(J)+ GRJ*SIGJ*(2.*XDISTJ**2 -1.0)* &
                              (1./SIGWI(J)-1./SIGWI2(J)))
                            DX(I)=DX(I)-DK1*X(I)/TMPR1
                            DY(I)=DY(I)-DK1*Y(I)/TMPR1
                            DZ(I)=DZ(I)-DK1*Z(I)/TMPR1
                            DX(J)=DX(J)-DK2*X(J)/TMPR2
                            DY(J)=DY(J)-DK2*Y(J)/TMPR2
                            DZ(J)=DZ(J)-DK2*Z(J)/TMPR2
                          ENDIF
                        ELSE
                          DZ(I)= DZ(I) - VJ*ARSQ*FSIGI*EXRI*DFDZ(I)* &
                          (GFREEI(I)-GFREEI2(I)+ GRI*SIGI*(2.*XDISTI**2 -1.0)* &
                          (1./SIGWI(I)-1./SIGWI2(I)))
                          DZ(J)= DZ(J) - VI*ARSQ*FSIGJ*EXRJ*DFDZ(J)* &
                          (GFREEI(J)-GFREEI2(J)+ GRJ*SIGJ*(2.*XDISTJ**2 -1.0)* &
                          (1./SIGWI(J)-1./SIGWI2(J)))
                        ENDIF

!  PORE MODEL
                        IF (LPOR) THEN
                           DX(I)= DX(I) - VJ*ARSQ*FSIGI*EXRI* &
                                  (GFREEI(I)-GFREEI2(I))*DFDX(I)
                           DX(J)= DX(J) - VI*ARSQ*FSIGJ*EXRJ* &
                                  (GFREEI(J)-GFREEI2(J))*DFDX(J)
                           DY(I)= DY(I) - VJ*ARSQ*FSIGI*EXRI* &
                                  (GFREEI(I)-GFREEI2(I))*DFDY(I)
                           DY(J)= DY(J) - VI*ARSQ*FSIGJ*EXRJ* &
                                  (GFREEI(J)-GFREEI2(J))*DFDY(J)
                        ENDIF
                     ENDIF
#if KEY_BLOCK==1
                  ENDIF
#endif
! Calculate second derivatives, I.A.
!...##IF lsecd (lsecd_main1)
                  IF(LSECD) THEN
                     FCTI=ARDIST6*(RSQ*SIGI*SIGI- &
                           3.D0*XDISTI*RDIST*SIGI-4.D0) &
                           -SIGI*ARDIST3*XDISTI*ARDIST*( &
                           XDISTI*SIGI+ARDIST)
                     FCTJ=ARDIST6*(RSQ*SIGJ*SIGJ- &
                           3.D0*XDISTJ*RDIST*SIGJ-4.D0) &
                           -SIGJ*ARDIST3*XDISTJ*ARDIST*( &
                           XDISTJ*SIGJ+ARDIST)
                     EFDERI=FSIGI*EXRI*FCTI
                     EFDERJ=FSIGJ*EXRJ*FCTJ
                     ETMP=TWO*(EFDERI*VGJI+EFDERJ*VGIJ)

                     AXX=DXI*DXI*ETMP+TMP ! was AXX=DXI*DXI*DDF+DF
                     AYY=DYI*DYI*ETMP+TMP
                     AZZ=DZI*DZI*ETMP+TMP
                     AXY=DXI*DYI*ETMP
                     AXZ=DXI*DZI*ETMP
                     AYZ=DYI*DZI*ETMP

                     II=3*I-2
                     JJ=3*J-2

                     IADD=IUPT(II)+II
                     DD1(IADD)=DD1(IADD)+AXX
                     IADD=IUPT(II+1)+II+1
                     DD1(IADD)=DD1(IADD)+AYY
                     IADD=IUPT(II+2)+II+2
                     DD1(IADD)=DD1(IADD)+AZZ
                     IADD=IUPT(II)+II+1
                     DD1(IADD)=DD1(IADD)+AXY
                     IADD=IUPT(II)+II+2
                     DD1(IADD)=DD1(IADD)+AXZ
                     IADD=IUPT(II+1)+II+2
                     DD1(IADD)=DD1(IADD)+AYZ

                     IADD=IUPT(JJ)+JJ
                     DD1(IADD)=DD1(IADD)+AXX
                     IADD=IUPT(JJ+1)+JJ+1
                     DD1(IADD)=DD1(IADD)+AYY
                     IADD=IUPT(JJ+2)+JJ+2
                     DD1(IADD)=DD1(IADD)+AZZ
                     IADD=IUPT(JJ)+JJ+1
                     DD1(IADD)=DD1(IADD)+AXY
                     IADD=IUPT(JJ)+JJ+2
                     DD1(IADD)=DD1(IADD)+AXZ
                     IADD=IUPT(JJ+1)+JJ+2
                     DD1(IADD)=DD1(IADD)+AYZ

                     IF (JJ.LT.II) THEN
                        IADD=IUPT(JJ)+II
                        DD1(IADD)=DD1(IADD)-AXX
                        IADD=IUPT(JJ+1)+II+1
                        DD1(IADD)=DD1(IADD)-AYY
                        IADD=IUPT(JJ+2)+II+2
                        DD1(IADD)=DD1(IADD)-AZZ
                        IADD=IUPT(JJ)+II+1
                        DD1(IADD)=DD1(IADD)-AXY
                        IADD=IUPT(JJ+1)+II
                        DD1(IADD)=DD1(IADD)-AXY
                        IADD=IUPT(JJ)+II+2
                        DD1(IADD)=DD1(IADD)-AXZ
                        IADD=IUPT(JJ+2)+II
                        DD1(IADD)=DD1(IADD)-AXZ
                        IADD=IUPT(JJ+1)+II+2
                        DD1(IADD)=DD1(IADD)-AYZ
                        IADD=IUPT(JJ+2)+II+1
                        DD1(IADD)=DD1(IADD)-AYZ
                     ELSE
                        IADD=IUPT(II)+JJ
                        DD1(IADD)=DD1(IADD)-AXX
                        IADD=IUPT(II+1)+JJ+1
                        DD1(IADD)=DD1(IADD)-AYY
                        IADD=IUPT(II+2)+JJ+2
                        DD1(IADD)=DD1(IADD)-AZZ
                        IADD=IUPT(II+1)+JJ
                        DD1(IADD)=DD1(IADD)-AXY
                        IADD=IUPT(II)+JJ+1
                        DD1(IADD)=DD1(IADD)-AXY
                        IADD=IUPT(II+2)+JJ
                        DD1(IADD)=DD1(IADD)-AXZ
                        IADD=IUPT(II)+JJ+2
                        DD1(IADD)=DD1(IADD)-AXZ
                        IADD=IUPT(II+2)+JJ+1
                        DD1(IADD)=DD1(IADD)-AYZ
                        IADD=IUPT(II+1)+JJ+2
                        DD1(IADD)=DD1(IADD)-AYZ
                     ENDIF
                  ENDIF
!...##ENDIF  (lsecd_main1)
               ENDDO loop80
            ENDDO
         ENDDO
      ENDDO

#if KEY_PARALLEL==1
#if KEY_VIBPARA==0
      if (mynod.eq.0) then
#endif
#endif
      IF (LMEMBR) THEN
        IF (LLAT) THEN
          CALL CALLATPRES(IFRSTA, NATOMX, X, Y, Z, EU, DZ, DX, DY)
        ENDIF
        IF (LDP) THEN
          CALL CALDPPT(IFRSTA, NATOMX, Z, EU, DZ)
        ENDIF
      ENDIF
#if KEY_PARALLEL==1
#if KEY_VIBPARA==0
      endif
#endif
#endif

! AvdV 12/5/01
#if KEY_PARALLEL==1
#if KEY_VIBPARA==0
      call gcomb(gsolv,natomx)
#endif
#endif

! ARD 00-10-14
! Moved addition of GREFI contribution down here due to EU move to loop.
! AvdV 12/05/01
! Add the GREFI only to one node for parallel runs to avoid double counting.
#if KEY_PARALLEL==1
#if KEY_VIBPARA==0
      if (mynod.eq.0) then
#endif
#endif
         IF (.NOT. LINTE) THEN
            DO I=IFRSTA,NATOMX
               IF (LMEMBR) THEN
                  GSOLV(I) = GSOLV(I) + GREFI2(I)+ F(I)*(GREFI(I)-GREFI2(I))
               ELSE
                  GSOLV(I) = GSOLV(I) + GREFI(I)
               ENDIF
               IF(QLBYCC) THEN
                  IF (ACAFLG(I).EQ.1) THEN  !if atom is active
                     IF (LMEMBR) THEN
                        EU = EU + GREFI2(I)+ F(I)*(GREFI(I)-GREFI2(I))
                     ELSE
                        EU = EU + GREFI(I)
                     ENDIF
                  ENDIF
               ELSE
                  IF (LMEMBR) THEN
                     EU = EU + GREFI2(I)+ F(I)*(GREFI(I)-GREFI2(I))
                  ELSE
                     EU = EU + GREFI(I)
                  ENDIF
               ENDIF
            ENDDO
            IF (LGOUY) THEN
               EU=EU+EGOUY
               IF (PRNLEV .GT. 9) WRITE (6,*) 'Egouy is ', EGOUY
            ENDIF
            IF (LVOLT) THEN
               EU=EU+EVOLT
               IF (PRNLEV .GT. 9) WRITE (6,*) 'Evolt is ', EVOLT
            ENDIF
         ENDIF
#if KEY_PARALLEL==1
#if KEY_VIBPARA==0
      endif
#endif
#endif

! AvdV 12/05/01
! The following code is fine for both parallel and serial
      IF (QECONT) THEN
         DO I=IFRSTA,NATOMX
! added if test--RJP
            IF(QLBYCC) THEN
               IF (ACAFLG(I).EQ.1) ECONT(I)=GSOLV(I) !if atom is active
            ELSE
               ECONT(I)=ECONT(I)+GSOLV(I)
            ENDIF
         ENDDO
      ENDIF
      RETURN
   END SUBROUTINE EEF1EN

   SUBROUTINE SLVPRINT(COMLYN,COMLEN)
! Calls EEF1EN and then prints out the solvation energies of each atom
!
  use chm_types
  use psf
  use coord
  use deriv
  use stream
  use param
  use econtmod
  use exfunc
  use bases_fcm
  use inbnd
  use parallel
  use number
  use string

      implicit none
      real(chm_real) EU,TOTALI,TOTARO,TOTPOL,TMP,F,HRF,CPRF
      real(chm_real) G,TOTREF,RPOR,TMP2,RC
      real(chm_real) HSOLV,CPSOLV,HALI,HARO,HPOL,CPALI,CPARO,CPPOL,GRF
      CHARACTER(len=*) COMLYN
      INTEGER ISEG,IRES,I,COMLEN
      CHARACTER(len=2) STR2
      CHARACTER(len=3) STR3
      LOGICAL LREF,LCHA

      LREF = (INDXA(COMLYN,COMLEN,'GREF') .GT. 0)
      CALL EEF1EN(EU,X,Y,Z,DX,DY,DZ,.FALSE.,ECONT,1,NATOM,1,NGRP, &
            BNBND%JNBG,BNBND%INBLOG, &
            BNBND%INB14,BNBND%IBLO14,.FALSE., &
            .FALSE. &
            )

#if KEY_PARALLEL==1
#if KEY_VIBPARA==0

      CALL GCOMB(EU,1)
! AvdV 12/5/01
! Removed the gcomb statement: this is already done in eef1en
!cc        CALL GCOMB(GSOLV,NATOM)
      IF (MYNOD.NE.0) RETURN
#endif
#endif

      IF (IOLEV.LE.0) RETURN

      WRITE (OUTU,100) EU
100   FORMAT ('SLVPRINT> TOTAL SOLVATION FREE ENERGY: ',F15.3, &
            ' Kcal/mol')
      WRITE (OUTU,*) &
      'SLVPRINT>                            G           H         CP        Welec'
      TOTALI= 0.0d+0
      TOTARO= 0.0d+0
      TOTPOL= 0.0d+0
      HALI= 0.0d+0
      HARO= 0.0d+0
      HPOL= 0.0d+0
      CPALI= 0.0d+0
      CPARO= 0.0d+0
      CPPOL= 0.0d+0
      TOTREF= 0.0d+0
      DO ISEG=1,NSEG
         DO IRES=NICTOT(ISEG)+1,NICTOT(ISEG+1)
            DO I=IBASE(IRES)+1,IBASE(IRES+1)
              ! IF (AMASS(I).LT.1.1) CYCLE
              ! Allow printing Hydrogen because they have electrostatic
              ! energy(PB or Gouy or Volt) not equal to zero.
                IF (LMEMBR) THEN
                  IF (LCURV) THEN
                     TMP = ABS(SQRT(X(I)**2+Y(I)**2+Z(I)**2)-RADU)*2/WIDTH
                  ELSE
                     TMP = ABS(Z(I))*2.0/WIDTH
                  ENDIF
                  F= TMP**NSMTH/(1.0+TMP**NSMTH)
                  IF (LINF) THEN
                    LCHA=.FALSE.
                    STR3= ATC(IAC(I))
                    IF ((STR3.EQ.'NH3').OR.(STR3.EQ.'NC2').OR.(STR3.EQ.'OC ')) LCHA=.TRUE.
                    IF (LCHA) F = MIN(TMP,1.0)
                  ENDIF
! PORE MODEL
                  IF (LPOR) THEN
                     IF (LCYL) THEN
                        RPOR= RCYL
                     ENDIF
                     IF (LPRB) THEN
                        RPOR= RPRB+APRB*TMP**2
                     ENDIF
                     IF (LCRC) THEN
                        TMP2=TMP
                        IF (TMP .GT. ONE) TMP2=ONE
                        RPOR= RCRC+(WIDTH/TWO)*(ONE-SQRT(ONE-TMP2**2))
                     ENDIF
                     RC= SQRT(X(I)**2+Y(I)**2)/RPOR
                     G = ONE- RC**NSMP/(ONE+RC**NSMP)
                     F = F + G -F*G
                  ENDIF
                  GRF = GREFI2(I)+ F*(GREFI(I)-GREFI2(I))
                  HRF = HREFI2(I)+ F*(HREFI(I)-HREFI2(I))
                  CPRF = CPREFI2(I)+ F*(CPREFI(I)-CPREFI2(I))
               ELSE
                  GRF= GREFI(I)
                  HRF= HREFI(I)
                  CPRF= CPREFI(I)
               ENDIF
               IF (DABS(GRF).GT.1.d-10) THEN
                  HSOLV= HRF*GSOLV(I)/GRF
                  CPSOLV= CPRF*GSOLV(I)/GRF
               ELSE
                  HSOLV= 0.d0
                  CPSOLV= 0.d0
               ENDIF
               WRITE(OUTU,'(I5,1X,A,1X,A,1X,A,1X,A,1X,A4,1X,4f11.3)') &
                     I,SEGID(ISEG)(1:idleng),RESID(IRES)(1:idleng), &
                     RES(IRES)(1:idleng),ATYPE(I)(1:idleng),ATC(IAC(I)), &
                     GSOLV(I),HSOLV,CPSOLV,WELECA(I)
               TOTREF= TOTREF+GSOLV(I)
               STR2= ATC(IAC(I))                        !Want first character
               IF (STR2.EQ.'CH') THEN
                  TOTALI= TOTALI+GSOLV(I)
                  HALI= HALI+HSOLV
                  CPALI= CPALI+CPSOLV
               ELSEIF (STR2.EQ.'CR') THEN
                  TOTARO= TOTARO+GSOLV(I)
                  HARO= HARO+HSOLV
                  CPARO= CPARO+CPSOLV
               ELSE
                  TOTPOL= TOTPOL+GSOLV(I)
                  HPOL= HPOL+HSOLV
                  CPPOL= CPPOL+CPSOLV
               ENDIF
            ENDDO
         ENDDO
      ENDDO
      WRITE (OUTU,101) TOTALI
      WRITE (OUTU,102) TOTARO
      WRITE (OUTU,103) TOTPOL
101   FORMAT ('SLVPRINT> TOTAL ALIPHATIC : ',F15.3,' Kcal/mol')
102   FORMAT ('SLVPRINT> TOTAL AROMATIC  : ',F15.3,' Kcal/mol')
103   FORMAT ('SLVPRINT> TOTAL POLAR     : ',F15.3,' Kcal/mol')
      WRITE (OUTU,105) 'SLVPRINT> ENTHALPY (ALIPHATIC,AROMATIC,POLAR)' &
            ,HALI+HARO+HPOL,HALI,HARO,HPOL
      WRITE (OUTU,105) 'SLVPRINT>       CP (ALIPHATIC,AROMATIC,POLAR)' &
            ,CPALI+CPARO+CPPOL,CPALI,CPARO,CPPOL
! M 05/04/10
      IF (LREF) WRITE (OUTU,105) &
        'SLVPRINT> TOTAL REFERENCE SOLVATION F ENERGY',TOTREF
105   FORMAT (A,F15.3,' (',3F15.3,' )')
      RETURN
   END SUBROUTINE SLVPRINT

   SUBROUTINE RDSLVPAR(ATC,NATC,SLVITC,VOLM,GREF,MAXATCL,COMLYN,&
         COMLEN,MXCMSZL,GFREE,UNPAR,TEMPR,SIGW,HREF,CPREF,SLVNT)
! This reads the file solvpar.inp
! Modeled it after PARRDR
!        GREF(i=1,atom types)= reference solvation free energy of an atom
!        GFREE(i=1,atom types)= free group solvation free energy
!        VOLM(i=1,atom types)= volume of each atom
!        SLVITC(atom type)= like ITC for NB terms, points to the correct index
!                            in GREF,VOLM, i.e., the Gref of atom J is
!                                GREF(SLVITC(IAC(J)))
!
  use exfunc
  use stream
  use string
  use consta
  use number
  use parallel
      implicit none

      INTEGER NAT,MAXATCL,NATC,MXCMSZL,COMLEN,I,J,UNPAR
      INTEGER SLVITC(*)
      CHARACTER(len=*) COMLYN
      CHARACTER(len=*) ATC(:)
      CHARACTER(len=4) WRD,AI
      LOGICAL EOF
      real(chm_real) VOLM(MAXATCL*2),GREF(MAXATCL*2),GFREE(MAXATCL*2)
      real(chm_real) SIGW(MAXATCL*2),HREF(MAXATCL*2),CPREF(MAXATCL*2)
      real(chm_real) V,G,GF,H,CP,RATIO,TEMPR,SG
      CHARACTER(len=4) SLVNT
      INTEGER ICHECK(NATC)
!
#if KEY_PARALLEL==1
!cc        IF(MYNOD.GT.0) GOTO 100
#endif
!
      NAT=0
      ICHECK=0
! Get input line
      EOF= .FALSE.
      DO
         CALL RDCMND(COMLYN,MXCMSZL,COMLEN,UNPAR,EOF,.TRUE., &
               .FALSE.,'SETUPSLV> ')
         CALL TRIME(COMLYN,COMLEN)
         IF (EOF) THEN      !only one solvent is there
            REWIND (UNIT=UNPAR)
            EXIT
         ENDIF
         WRD= NEXTA4(COMLYN,COMLEN)
         IF (WRD == SLVNT) EXIT
      ENDDO
! We have found the right solvent. Proceed with parameters
      DO
         EOF= .FALSE.
         CALL RDCMND(COMLYN,MXCMSZL,COMLEN,UNPAR,EOF,.TRUE., &
               .FALSE.,'SETUPSLV> ')
         CALL TRIME(COMLYN,COMLEN)
         IF (EOF) EXIT
         IF (.NOT.(COMLEN.GT.0)) CYCLE
! Data processing
         WRD= NEXTA4(COMLYN,COMLEN)
         IF (WRD.EQ.'END') EXIT
         AI=WRD
         V=NEXTF(COMLYN,COMLEN)
         G=NEXTF(COMLYN,COMLEN)
         GF=NEXTF(COMLYN,COMLEN)
         H=NEXTF(COMLYN,COMLEN)
         CP=NEXTF(COMLYN,COMLEN)
         SG=NEXTF(COMLYN,COMLEN)
         IF (DABS(G).LT.RSMALL) THEN
            RATIO= ZERO
         ELSE
            RATIO= GF/G
         ENDIF
         CALL TRIME(COMLYN,COMLEN)
         NAT= NAT+1
         I=0
         DO J=1,NATC
            IF(EQWDWC(ATC(J),AI)) THEN
               I=J
               ICHECK(J)=1
               SLVITC(I)= NAT
               VOLM(NAT)= V
               G= G -(TEMPR-ROOMT)*(H-G)/ROOMT -CP* &
                     (TEMPR*DLOG(TEMPR/ROOMT)-(TEMPR-ROOMT))/THOSND
               GREF(NAT)= G
               GFREE(NAT)= G*RATIO
               HREF(NAT)= H + CP*(TEMPR-ROOMT)/THOSND
               CPREF(NAT)= CP
               SIGW(NAT)= SG
            ENDIF
         ENDDO
         IF(I.EQ.0) THEN
            WRITE(OUTU,150) AI
150         FORMAT(' RDSLVPAR> WARNING: ATOM FOR SOLV ', &
                  A4,' DOESNT EXIST')
            NAT= NAT-1
         ENDIF
         CALL XTRANE(COMLYN,COMLEN,'RDSLVPAR')
      ENDDO
!Check that all atom types have a solvation parameter
      DO J=1,NATC
        IF(ATC(J).EQ." ") CYCLE
        IF(ICHECK(J).EQ.0) THEN
            WRITE(OUTU,160) ATC(J)
160         FORMAT(' RDSLVPAR> WARNING: NO SOLV PAR FOR ',A8)
        ENDIF
      ENDDO
#if KEY_PARALLEL==1
      CALL PSND4(NAT,1)
      CALL PSND4(SLVITC,NATC)
      CALL PSND8(VOLM,NAT)
      CALL PSND8(GREF,NAT)
      CALL PSND8(GFREE,NAT)
      CALL PSND8(HREF,NAT)
      CALL PSND8(CPREF,NAT)
      CALL PSND8(SIGW,NAT)
#endif
      RETURN
   END SUBROUTINE RDSLVPAR

   SUBROUTINE EEF1IM(EIMSLV,BNBND,BIMAG,BIMAGX, &
         IGPBS,IGPTYP,IAC,DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
         NATOMX)
!
!     Solvation free energy for images
!     Based on EIMNBD
!
!     T. Lazaridis  Jun 99
!
  use chm_types
  use exfunc
  use number
  use inbnd
  use image
  use timerm
  use stream
  use datstr
  use machutil,only:timre,timrb
!---   use nbutil_module,only:setbnd,getbnd
      implicit none

      real(chm_real) EIMSLV,ESLV
      type(nonbondDataStructure) BNBND
      type(imageDataStructure) BIMAG
      type(imageDataStructure) BIMAGX

      INTEGER IGPBS(:),IGPTYP(:),IAC(:)
      real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
      LOGICAL QECONT
      real(chm_real) ECONT(:)
      INTEGER NATOMX !RJP

      INTEGER I,IMX

!     Set up a dummy data structure for enbond.
      type(nonbondDataStructure) BDUMMY

      IF(NTRANS.EQ.0) RETURN
      CALL ALIASDT_nbond(BDUMMY,BNBND)

!     Make sure counters, flags and cuttoffs, etc are correct.
      CALL GETBND(BNBND,.TRUE.)

      EIMSLV=ZERO
      ESLV=ZERO

      CALL NBSET_B14_FROM_IMG(BDUMMY, BIMAG)
      CALL NBSET_G14_FROM_IMG(BDUMMY, BIMAG)
      NNG14=BIMAG%NIMING
!
!     Check if any self-energy terms are present
!
      IF (BIMAGX%NIMNBS.GT.0 .OR. BIMAGX%NIMNBX.GT.0) THEN
!       Self terms are present
         IMX=NATIM
         DO I=1,IMX
            DX(I)=DX(I)*TWO
            DY(I)=DY(I)*TWO
            DZ(I)=DZ(I)*TWO
         ENDDO

         CALL NBSET_FROM_IMG_SX(BDUMMY, BIMAGX)
         NNNB =BIMAGX%NIMNBS
         NNNBG=BIMAGX%NIMNBX

         NNB14= 0

         CALL SETBND(BDUMMY)

         CALL EEF1EN(ESLV,X,Y,Z,DX,DY,DZ,QECONT,ECONT,1,NATIM,1,NIMGRP, &
               BDUMMY%JNBG,BDUMMY%INBLOG, &
               BDUMMY%INB14,BDUMMY%IBLO14,.TRUE., &
               .FALSE. &
               )

         ESLV=ESLV*HALF
         DO I=1,IMX
            DX(I)=DX(I)*HALF
            DY(I)=DY(I)*HALF
            DZ(I)=DZ(I)*HALF
         ENDDO
      ENDIF
!
!     compute image nonbonded energies
!
      IF (BIMAGX%NIMNB.GT.0 .OR. BIMAGX%NIMNBG.GT.0) THEN

         NNB14=BIMAGX%NIMINB

         CALL NBSET_FROM_IMG_G(BDUMMY, BIMAGX)
         NNNB =BIMAGX%NIMNB
         NNNBG=BIMAGX%NIMNBG

         CALL SETBND(BDUMMY)

         CALL EEF1EN(EIMSLV,X,Y,Z,DX,DY,DZ,QECONT,ECONT,1,NATIM,1,NIMGRP, &
               BDUMMY%JNBG,BDUMMY%INBLOG, &
               BDUMMY%INB14,BDUMMY%IBLO14,.TRUE., &
               .FALSE. &
               )

         IF (TIMER.GT.1) THEN
            IF(PRNLEV.GE.2) WRITE(OUTU,'(1X,A)') &
                  'IMAGE EEF1 TIMES:'
            CALL TIMRE
            CALL TIMRB
         ENDIF
      ENDIF

      EIMSLV=EIMSLV+ESLV

      CALL FREEDT_nbond(BDUMMY)

      RETURN
   END SUBROUTINE EEF1IM

  subroutine allocate_eef1()
    use memory
    character(len=*),parameter :: routine_name="allocate_eef1"

    call chmalloc(file_name,routine_name,'GSOLV   ',maxaIM,crl=GSOLV  )
    call chmalloc(file_name,routine_name,'VOLMI   ',maxaIM,crl=VOLMI  )
    call chmalloc(file_name,routine_name,'GREFI   ',maxaIM,crl=GREFI  )
    call chmalloc(file_name,routine_name,'GFREEI  ',maxaIM,crl=GFREEI )
    call chmalloc(file_name,routine_name,'SIGWI   ',maxaIM,crl=SIGWI  )
    call chmalloc(file_name,routine_name,'FSIGWI  ',maxaIM,crl=FSIGWI )
    call chmalloc(file_name,routine_name,'VDWRI   ',maxaIM,crl=VDWRI  )
    call chmalloc(file_name,routine_name,'GREFI2  ',maxa  ,crl=GREFI2 )
    call chmalloc(file_name,routine_name,'GFREEI2 ',maxa  ,crl=GFREEI2)
    call chmalloc(file_name,routine_name,'HREFI   ',maxa  ,crl=HREFI  )
    call chmalloc(file_name,routine_name,'HREFI2  ',maxa  ,crl=HREFI2 )
    call chmalloc(file_name,routine_name,'CPREFI  ',maxa  ,crl=CPREFI )
    call chmalloc(file_name,routine_name,'CPREFI2 ',maxa  ,crl=CPREFI2)
    call chmalloc(file_name,routine_name,'SIGWI2  ',maxa  ,crl=SIGWI2 )
    call chmalloc(file_name,routine_name,'FSIGWI2 ',maxa  ,crl=FSIGWI2)
    call chmalloc(file_name,routine_name,'LAM     ',maxa  ,crl=LAM    )
    call chmalloc(file_name,routine_name,'DFDX    ',maxa  ,crl=DFDX   )
    call chmalloc(file_name,routine_name,'DFDY    ',maxa  ,crl=DFDY   )
    call chmalloc(file_name,routine_name,'DFDZ    ',maxa  ,crl=DFDZ   )
    call chmalloc(file_name,routine_name,'WELECA  ',maxa  ,crl=WELECA )
    call chmalloc(file_name,routine_name,'PBPHI   ',MAXPHIDIM*MAXPHIDIM*MAXPHIDIM,crl=PBPHI )
    return
  end subroutine allocate_eef1

  SUBROUTINE RDPHI(READU,PBPHI,PNX,PNY,PNZ, &
       NCEL,PDX,PDY,PDZ,X0,Y0,Z0)
  use parallel
        INTEGER READU
        INTEGER PNX,PNY,PNZ,NCEL
        REAL(chm_real) PDX,PDY,PDZ,X0,Y0,Z0
        REAL(chm_real) PBPHI(*)
!       LOCAL VARIABLE
        REAL(chm_real) TMP
        CHARACTER*80 STRING
        CHARACTER*70 TMPS
        INTEGER NLINE,REMAIN,I,J
100     CONTINUE
        READ(READU,'(A70)') STRING
        IF (INDEX(STRING,'#') .EQ. 1) GOTO 100
        READ(STRING,'(A35,3(I4))') TMPS,PNX,PNY,PNZ
        READ(READU,*) TMPS,X0,Y0,Z0
        READ(READU,*) TMPS,PDX,TMP,TMP
        READ(READU,*) TMPS,TMP,PDY,TMP
        READ(READU,*) TMPS,TMP,TMP,PDZ
        READ(READU,*) TMPS
        READ(READU,*) TMPS
        NCEL=PNX*PNY*PNZ;
        NLINE=NCEL/3
        REMAIN=MOD(NCEL,3)
        J=1
        DO 101 I = 1,NLINE
          READ(READU,*) PBPHI(J),PBPHI(J+1),PBPHI(J+2)
          J=J+3
101     CONTINUE
        IF (REMAIN .EQ. 1) THEN
           READ(READU,*) PBPHI(J)
           J=J+1
        ELSEIF (REMAIN .EQ. 2) THEN
           READ(READU,*) PBPHI(J),PBPHI(J+1)
           J=J+2
        ENDIF
#if KEY_PARALLEL==1
        CALL PSND8(PBPHI,MAXPHIDIM*MAXPHIDIM*MAXPHIDIM)
#endif
        RETURN
  END SUBROUTINE RDPHI

  SUBROUTINE GETPHI(PHI,PNX,PNY,PNZ, &
       NCEL,PDX,PDY,PDZ,X0,Y0,Z0,XPOS,YPOS,ZPOS, &
       V,DVDX,DVDY,DVDZ,LOUTSIDE)
        REAL(chm_real) PHI(*)
        INTEGER PNX,PNY,PNZ,NCEL
        REAL(chm_real) PDX,PDY,PDZ,X0,Y0,Z0,XPOS,YPOS,ZPOS
        REAL(chm_real) XL,XH,YL,YH,ZL,ZH
        REAL(chm_real) IXF,IYF,IZF,V,PX,PY,PZ,DVDX,DVDY,DVDZ
        INTEGER IX,IY,IZ,PINDEX
        LOGICAL LOUTSIDE
        XL=X0
        YL=Y0
        ZL=Z0
        IXF=(XPOS-XL)/PDX
        IYF=(YPOS-YL)/PDY
        IZF=(ZPOS-ZL)/PDZ
        IX=FLOOR(IXF)
        IY=FLOOR(IYF)
        IZ=FLOOR(IZF)
        PX=IXF-FLOAT(IX)
        PY=IYF-FLOAT(IY)
        PZ=IZF-FLOAT(IZ)
        ! peroidic boundary
        IF( SHPEBND .EQ. 0) THEN
            IF(IX .GE. 0) THEN
             IX=MOD(IX,PNX-1)
            ELSE
             IX=PNX-1+MOD(IX,PNX-1)
            ENDIF
            IF(IY .GE. 0) THEN
             IY=MOD(IY,PNY-1)
            ELSE
             IY=PNY-1+MOD(IY,PNY-1)
            ENDIF
            IF(IZ .GE. 0) THEN
             IZ=MOD(IZ,PNZ-1)
            ELSE
             IZ=PNZ-1+MOD(IZ,PNZ-1)
            ENDIF
        ENDIF
        ! Outside use Analytic Function of Imm1
        IF(SHPEBND .EQ. 1) THEN
             IF(IX .LT. 0 .OR. IX .GT. PNX-1) THEN
              LOUTSIDE = .TRUE.
             RETURN
             ENDIF
             IF(IY .LT. 0 .OR. IY .GT. PNY-1) THEN
              LOUTSIDE = .TRUE.
             RETURN
             ENDIF
             IF(IZ .LT. 0 .OR. IZ .GT. PNZ-1) THEN
              LOUTSIDE = .TRUE.
             RETURN
             ENDIF
        ENDIF
        LOUTSIDE = .FALSE.
        PINDEX=IX*PNY*PNZ+IY*PNZ+IZ+1! trilinear interpolation of phi to get the value at (px,py,pz)
         V=PX*PY*PZ*PHI(PINDEX+PNY*PNZ+PNZ+1)
         V=V+PX*PY*(1-PZ)*PHI(PINDEX+PNY*PNZ+PNZ)
         V=V+PX*(1-PY)*PZ*PHI(PINDEX+PNY*PNZ+1)
         V=V+PX*(1-PY)*(1-PZ)*PHI(PINDEX+PNY*PNZ)
         V=V+(1-PX)*PY*PZ*PHI(PINDEX+PNZ+1)
         V=V+(1-PX)*PY*(1-PZ)*PHI(PINDEX+PNZ)
         V=V+(1-PX)*(1-PY)*PZ*PHI(PINDEX+1)
         V=V+(1-PX)*(1-PY)*(1-PZ)*PHI(PINDEX)
         ! trilinear interpolation of phi to get to get the derivatives at (px,py,pz)
         DVDX=PY*PZ*PHI(PINDEX+PNY*PNZ+PNZ+1)
         DVDX=DVDX+PY*(1-PZ)*PHI(PINDEX+PNY*PNZ+PNZ)
         DVDX=DVDX+(1-PY)*PZ*PHI(PINDEX+PNY*PNZ+1)
         DVDX=DVDX+(1-PY)*(1-PZ)*PHI(PINDEX+PNY*PNZ)
         DVDX=DVDX-PY*PZ*PHI(PINDEX+PNZ+1)
         DVDX=DVDX-PY*(1-PZ)*PHI(PINDEX+PNZ)
         DVDX=DVDX-(1-PY)*PZ*PHI(PINDEX+1)
         DVDX=DVDX-(1-PY)*(1-PZ)*PHI(PINDEX)
         DVDX=DVDX/PDX
         DVDY=PX*PZ*PHI(PINDEX+PNY*PNZ+PNZ+1)
         DVDY=DVDY+PX*(1-PZ)*PHI(PINDEX+PNY*PNZ+PNZ)
         DVDY=DVDY-PX*PZ*PHI(PINDEX+PNY*PNZ+1)
         DVDY=DVDY-PX*(1-PZ)*PHI(PINDEX+PNY*PNZ)
         DVDY=DVDY+(1-PX)*PZ*PHI(PINDEX+PNZ+1)
         DVDY=DVDY+(1-PX)*(1-PZ)*PHI(PINDEX+PNZ)
         DVDY=DVDY-(1-PX)*PZ*PHI(PINDEX+1)
         DVDY=DVDY-(1-PX)*(1-PZ)*PHI(PINDEX)
         DVDY=DVDY/PDY
         DVDZ=PX*PY*PHI(PINDEX+PNY*PNZ+PNZ+1)
         DVDZ=DVDZ-PX*PY*PHI(PINDEX+PNY*PNZ+PNZ)
         DVDZ=DVDZ+PX*(1-PY)*PHI(PINDEX+PNY*PNZ+1)
         DVDZ=DVDZ-PX*(1-PY)*PHI(PINDEX+PNY*PNZ)
         DVDZ=DVDZ+(1-PX)*PY*PHI(PINDEX+PNZ+1)
         DVDZ=DVDZ-(1-PX)*PY*PHI(PINDEX+PNZ)
         DVDZ=DVDZ+(1-PX)*(1-PY)*PHI(PINDEX+1)
         DVDZ=DVDZ-(1-PX)*(1-PY)*PHI(PINDEX)
         DVDZ=DVDZ/PDZ
  END SUBROUTINE GETPHI

  SUBROUTINE INILATPRES
  !
  ! Initialize for lateral pressure calculation
  !
    use consta

    INTEGER I

    TML   = 30.0D0
    TSLAB = 0.1D0 ! slab thickness
    NSLAB = IDINT(2*TML/TSLAB)
    CLP1  = 2.0*PI*TSLAB
    CLP2  = 1.434D-5*TSLAB
    CLP3  = PI*TSLAB*TSLAB
    CLP4  = PL/APL
    CLP5  = 100.0*CMD*CLP4
    CLP6  = 100.0*CMD/CLP4

    ! Atomic radius from EEF1 volume. Increased by 10% empirically
    DO I = 1, 32
      RLAT(I) = 1.1*(0.2387324*VOLM(I))**0.333
    ENDDO

    CALL GENLATPRES

  END SUBROUTINE INILATPRES

    SUBROUTINE CALLATPRES(IFRSTA, NATOMX, X, Y, Z, EU, DZ, DX, DY)
  !
  ! Interaction between membrane lateral pressure and peptide cross-sectional area
  !
    use consta
    use psf
    use number
    use stream

    implicit none
    ! Global
    INTEGER IFRSTA, NATOMX
    REAL(chm_real) EU, X(*), Y(*), Z(*), DZ(*), DX(*), DY(*)
     ! Local, expand DADZ if more than 300 atoms
    INTEGER I, K, ZN, RN, RN2, DIST, BL, BH, LOC
    REAL(chm_real) EPS,TMP1,TMP2,PRES,APWSR,ELAT,MTML,TMPZ,TMPR,DTMPZ
    REAL(chm_real) AP(600), DEDA(600), DADZ(600*maxa),ELAT1,ELAT2

    EPS  = 1.0D-8
    MTML = TML-2.2

    ! Initialize slabs
    ELAT = ZERO
    ELAT1 = ZERO
    ELAT2 = ZERO
    DO K = 1, NSLAB
      AP(K) = ZERO
      DEDA(K) = ZERO
      DO I = IFRSTA, NATOMX
        DADZ(NSLAB*(I-1)+K) = ZERO
      ENDDO
    ENDDO

    ! Peptide area and dA/dZ
    DO I = IFRSTA, NATOMX
      IF (LCURV) THEN
        IF (LTUB) THEN
          TMPZ=SQRT(Y(I)**2+Z(I)**2)
        ELSE
          TMPZ=SQRT(X(I)**2+Y(I)**2+Z(I)**2)
        ENDIF
        TMPZ=TMPZ-RADU
      ELSE
        TMPZ=Z(I)
      ENDIF
      IF (ABS(TMPZ) .LT. MTML) THEN
        LOC = NSLAB*(I-1)
        ZN = IDINT((TMPZ+TML)/TSLAB)+1
        RN = IDINT(RLAT(SLVITC(IAC(I)))/TSLAB)
        RN2 = RN*RN
        BL = ZN-RN
        BH = ZN+RN
        DO K = BL, BH
          DIST = ZN-K
          AP(K) = AP(K)+CLP3*(RN2-DIST*DIST)
          DADZ(LOC+K) = -CLP1*DIST
        ENDDO
      ENDIF
    ENDDO

    ! Lateral pressure energy and dE/dA
    DO K = 1, NSLAB
      APWSR = LAMBDA*AP(K)
      PRES = PLAT(K)

      ! Linear
      ! ELAT = ELAT+(PRES+0.5*CLP5*APWSR)*APWSR
      ! DEDA(K) = 0.5*CLP2*(PRES+CLP5*APWSR)

      ! Logarithmic
      TMP1 = ONE-CLP4*APWSR
      TMP2 = LOG(TMP1)
      ELAT = ELAT+PRES*APWSR+CLP6*(TMP1*(TMP2-ONE)+ONE)
      ELAT1 = ELAT1+ PRES*APWSR
      ELAT2 = ELAT2+ CLP6*(TMP1*(TMP2-ONE)+ONE)
      DEDA(K) = 0.5*CLP2*(PRES-100.0*CMD*TMP2)
    ENDDO

    ELAT = CLP2*ELAT
    ELAT1 = CLP2*ELAT1
    ELAT2 = CLP2*ELAT2
    EU = EU+ELAT

    ! First derivative (dE/dZ=dE/dA*dA/dZ)
    DO I = IFRSTA, NATOMX
       IF (LCURV) THEN
         IF (LTUB) THEN
           TMPR=SQRT(Y(I)**2+Z(I)**2)
           TMPZ=TMPR-RADU
           IF (ABS(TMPZ) .LT. MTML) THEN
             LOC = NSLAB*(I-1)
             ZN = IDINT((TMPZ+TML)/TSLAB)+1
             RN = IDINT(RLAT(SLVITC(IAC(I)))/TSLAB)
             BL = ZN-RN
             BH = ZN+RN
             DO K = BL, BH
               DTMPZ = DEDA(K)*DADZ(LOC+K)
               DY(I)=DY(I)+Y(I)*DTMPZ/TMPR
               DZ(I)=DZ(I)+Z(I)*DTMPZ/TMPR
             ENDDO
           ENDIF
         ELSE
           TMPR=SQRT(X(I)**2+Y(I)**2+Z(I)**2)
           TMPZ=TMPR-RADU
           IF (ABS(TMPZ) .LT. MTML) THEN
             LOC = NSLAB*(I-1)
             ZN = IDINT((TMPZ+TML)/TSLAB)+1
             RN = IDINT(RLAT(SLVITC(IAC(I)))/TSLAB)
             BL = ZN-RN
             BH = ZN+RN
             DO K = BL, BH
               DTMPZ = DEDA(K)*DADZ(LOC+K)
               DX(I)= DX(I)+X(I)*DTMPZ/TMPR
               DY(I)= DY(I)+Y(I)*DTMPZ/TMPR
               DZ(I)= DZ(I)+Z(I)*DTMPZ/TMPR
             ENDDO
           ENDIF
         ENDIF
       ELSE
         IF (ABS(Z(I)) .LT. MTML) THEN
           LOC = NSLAB*(I-1)
           ZN = IDINT((Z(I)+TML)/TSLAB)+1
           RN = IDINT(RLAT(SLVITC(IAC(I)))/TSLAB)
           BL = ZN-RN
           BH = ZN+RN
           DO K = BL, BH
             DZ(I) = DZ(I)+DEDA(K)*DADZ(LOC+K)
           ENDDO
         ENDIF
       ENDIF
    ENDDO

    ! Print area
    IF (PRNLEV .GT. 9) THEN
      WRITE(*, 750)
      WRITE(*, 751)
      DO K = 1, NSLAB
        WRITE(*, 752) K, PLAT(K), AP(K), PLAT(K)*AP(K)*LAMBDA*CLP2
      ENDDO
      WRITE(*,753)
750   FORMAT('     PLAT:     NSLAB         P        AP')
751   FORMAT('----------------------------------------')
752   FORMAT('     PLAT>',I10,3F10.3)
753   FORMAT('----------------------------------------')
    ENDIF

    ! Print energy
    IF (PRNLEV .GT. 9) THEN
      WRITE(*, 754) ELAT
      WRITE(*,*) ELAT1,ELAT2
754   FORMAT('LATERAL PRESSURE ENERGY>', F12.8)
    ENDIF

  END SUBROUTINE CALLATPRES
  SUBROUTINE CALDPPT(IFRSTA, NATOMX, Z, EU, DZ)
  !
  ! Interaction between membrane dipole potential and peptide dipole moment
  !
    use consta
    use psf
    use stream

    implicit none
    ! Global
    INTEGER IFRSTA, NATOMX
    REAL(chm_real) EU, Z(*), DZ(*)
    ! Local
    INTEGER I
    REAL(chm_real) FDP, GDP, Z1, Z2, ZEG, ZEF, ADP, CGDP, EDP, MDP1

    MDP1 = 0.0231*MDP
    ADP = 6.23D-7 ! control shape with b = 5.5
    EDP = 0.0

    DO I = IFRSTA, NATOMX
      Z1  = ABS(Z(I))
      Z2  = Z1*Z1
      ZEF = Z2*Z2*Z1*SQRT(Z1)
      ZEG = Z2*Z2*SQRT(Z1)
      FDP = MDP1/(1+ADP*ZEF) ! dipole potential
      GDP = -5.5*SIGN(1.0d0, Z1)*ADP*MDP1*ZEG/((1+ADP*ZEF)*(1+ADP*ZEF)) ! first derivative

      ! Add side-chain charges
      IF (abs(CG(I)+0.9).LT.1d-6) THEN ! LYS or NT
        CGDP = +0.1d0
      ELSE IF (abs(CG(I)+0.121).LT.1d-6) THEN ! ARG
        CGDP = +0.38d0
      ELSE IF (abs(CG(I)-0.45).LT.1d-6) THEN ! HSC/HSP,GLYP,PROP
        CGDP = +0.95d0
      ELSE IF (abs(CG(I)-1.0).LT.1d-6) THEN ! ASP,GLU,CT
        CGDP = 0.d0
      ELSE
        CGDP = CG(I)
      ENDIF

      EU = EU+CGDP*FDP
      EDP = EDP+CGDP*FDP
      DZ(I) = DZ(I)+CGDP*GDP
    ENDDO

    ! Print energy
    IF (PRNLEV .GT. 9) THEN
      WRITE(*, 755) EDP
755   FORMAT('DIPOLE POTENTIAL ENERGY>', F12.8)
    ENDIF

  END SUBROUTINE CALDPPT

  SUBROUTINE GENLATPRES
  !
  ! This subroutine calculates the lateral pressure profile at various
  ! molar fractions of DOPE for mixed DOPC/DOPE bilayers. The calculation
  ! is based on the elasticity theory of membranes (Helfrich W, 1981)
  !
    use consta

    implicit none
    INTEGER I, J
    REAL(chm_real) MEANMOD,GAUSSMOD,CURPE,CURPC,DELTA,PRECISION,CUTOFF,CURTOTAL
    REAL(chm_real) TMP1,TMP2,DENOM,ZSLAB,MAGN(3),POS(3),WIDTH(3),M1(3),M2(3),WD(3)
    REAL(chm_real) TMEM,XTMP,GTMP,DAZ

    ! Parameters of curvature and elasticity (Marsh D, Biophys J,2007,93: 3884)
    MEANMOD = 4.0D5         ! Bending modulus, in bar*Ang^3
    GAUSSMOD = -0.8*MEANMOD ! Gaussian modulus
    CURPE = -0.0431         ! PE spontaneous curvature, 1/Ang
    CURPC = -0.0066         ! PC spontaneous curvature
    DELTA = 13.5            ! position of pivot plane, Ang
    TMEM = 45.4             ! Thickness of entire membrane, Ang

    ! Parameters for Gaussian functions. Subscript 1, 2, 3 corresponds to
    ! acyl chain, interface, and headgroup. Values are estimated based on
    ! literature (Ollila OHS, J Struct Biol, 2007, 159: 311), magn(1) and
    ! magn(3) are temporarily set to 0 and will be determined later based
    ! on elasticity.
    DATA MAGN/0, -1200, 0/
    DATA POS/9.5, 15.5, 19.5/
    DATA WIDTH/35.0, 10.0, 10.0/

    ! Decay of Gaussian function
    PRECISION = 0.02
    CUTOFF = 2*SQRT(LOG(1/PRECISION))

    ! Spontaneous curvature of mixed monolayer
    CURTOTAL = (1-XPE)*CURPC+XPE*CURPE

    ! Calculate first and second moments
    DO I = 1, 3
      WD(I) = WIDTH(I)/CUTOFF
      M1(I) = POS(I)*WD(I)
      M2(I) = WD(I)*((POS(I)-DELTA)*(pos(I)-DELTA)+0.5*WD(I)*WD(I))
    ENDDO

    ! Calculate MAGN(1) and MAGN(3)
    TMP1 = MEANMOD*CURTOTAL/SQRT(PI)-M1(2)*MAGN(2)
    TMP2 = -GAUSSMOD/SQRT(PI)-M2(2)*MAGN(2)
    DENOM = M1(1)*M2(3)-M2(1)*M1(3)
    MAGN(1) = (M2(3)*TMP1-M1(3)*TMP2)/DENOM
    MAGN(3) = (M1(1)*TMP2-M2(1)*TMP1)/DENOM

    DO I = 1, NSLAB
      ZSLAB = -29.95+TSLAB*I
      PLAT(I) = 0.0
      DO J = 1, 3
        PLAT(I) = PLAT(I)+MAGN(J)*EXP(-(ZSLAB-POS(J))* &
            (ZSLAB-POS(J))/(WD(J)*WD(J)))
        PLAT(I) = PLAT(I)+MAGN(J)*EXP(-(ZSLAB+POS(J))* &
            (ZSLAB+POS(J))/(WD(J)*WD(J)))
      ENDDO
      IF (LCURV) THEN
        IF (LTUB) THEN
          XTMP=TMEM/4.0/RADU
        ELSE
          XTMP=TMEM/(2.0*RADU)*(1.0+TMEM/6.0/RADU)
        ENDIF
        GTMP=1.0
        TMP1=ZSLAB/RADU
        IF (LTUB) THEN
          DAZ=(1.0+TMP1)*GTMP -1.0
        ELSE
          DAZ=(1.0+2*TMP1+TMP1**2)*GTMP - 1.0
        ENDIF
        ZSLAB=ABS(ZSLAB)
        IF (ZSLAB.GT.POS(3)) THEN
          PLAT(I)=PLAT(I)-CMD*100.0*DAZ*EXP(-(ZSLAB-POS(3))* &
            (ZSLAB-POS(3))/(WD(3)*WD(3)))
        ELSE
          PLAT(I)=PLAT(I)-CMD*100.0*DAZ
        ENDIF
      ENDIF
    ENDDO

  END SUBROUTINE GENLATPRES

! Bessel Functions used by approximate solutions of the Poisson–Boltzmann equation in
! cylindrical coordinates
!     Bessel Function of the 2nd kind of order zero
!
      Real(chm_real) FUNCTION BESSK0(X)
  use consta
  use number
      IMPLICIT NONE
      Real(chm_real) X,Y,AX,P1,P2,P3,P4,P5,P6,P7,Q1,Q2,Q3,Q4,Q5,Q6,Q7
      DATA P1,P2,P3,P4,P5,P6,P7/-0.57721566D0,0.42278420D0,0.23069756D0, &
      0.3488590D-1,0.262698D-2,0.10750D-3,0.74D-5/
      DATA Q1,Q2,Q3,Q4,Q5,Q6,Q7/1.25331414D0,-0.7832358D-1,0.2189568D-1, &
      -0.1062446D-1,0.587872D-2,-0.251540D-2,0.53208D-3/
      IF(X.EQ.0.D0) THEN
      BESSK0=1.D30
      RETURN
      ENDIF
      IF(X.LE.2.D0) THEN
      Y=X*X/4.D0
      AX=-LOG(X/2.D0)*BESSI0(X)
      BESSK0=AX+(P1+Y*(P2+Y*(P3+Y*(P4+Y*(P5+Y*(P6+Y*P7))))))
      ELSE
      Y=(2.D0/X)
      AX=EXP(-X)/DSQRT(X)
      BESSK0=AX*(Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*(Q5+Y*(Q6+Y*Q7))))))
      ENDIF
      RETURN
      END FUNCTION BESSK0
!
!     Bessel Function of the 2nd kind of order one
!
      Real(chm_real) FUNCTION BESSK1(X)
  use consta
  use number
      IMPLICIT NONE
      Real(chm_real) X,Y,AX,P1,P2,P3,P4,P5,P6,P7,Q1,Q2,Q3,Q4,Q5,Q6,Q7
      DATA P1,P2,P3,P4,P5,P6,P7/1.D0,0.15443144D0,-0.67278579D0,  &
      -0.18156897D0,-0.1919402D-1,-0.110404D-2,-0.4686D-4/
      DATA Q1,Q2,Q3,Q4,Q5,Q6,Q7/1.25331414D0,0.23498619D0,-0.3655620D-1, &
      0.1504268D-1,-0.780353D-2,0.325614D-2,-0.68245D-3/
      IF(X.EQ.0.D0) THEN
      BESSK1=1.D32
      RETURN
      ENDIF
      IF(X.LE.2.D0) THEN
      Y=X*X/4.D0
      AX=LOG(X/2.D0)*BESSI1(X)
      BESSK1=AX+(1.D0/X)*(P1+Y*(P2+Y*(P3+Y*(P4+Y*(P5+Y*(P6+Y*P7))))))
      ELSE
      Y=(2.D0/X)
      AX=EXP(-X)/DSQRT(X)
      BESSK1=AX*(Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*(Q5+Y*(Q6+Y*Q7))))))
      ENDIF
      RETURN
      END FUNCTION BESSK1
!
!     Bessel Function of the 1st kind of order zero
!
      Real(chm_real) FUNCTION BESSI0(X)
  use consta
  use number
      IMPLICIT NONE
      Real(chm_real) X,Y,P1,P2,P3,P4,P5,P6,P7,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,AX,BX
      DATA P1,P2,P3,P4,P5,P6,P7/1.D0,3.5156229D0,3.0899424D0,1.2067429D0,  &
      0.2659732D0,0.360768D-1,0.45813D-2/
      DATA Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9/0.39894228D0,0.1328592D-1,  &
      0.225319D-2,-0.157565D-2,0.916281D-2,-0.2057706D-1,  &
      0.2635537D-1,-0.1647633D-1,0.392377D-2/
      IF(ABS(X).LT.3.75D0) THEN
      Y=(X/3.75D0)**2
      BESSI0=P1+Y*(P2+Y*(P3+Y*(P4+Y*(P5+Y*(P6+Y*P7)))))
      ELSE
      AX=ABS(X)
      Y=3.75D0/AX
      BX=EXP(AX)/DSQRT(AX)
      AX=Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*(Q5+Y*(Q6+Y*(Q7+Y*(Q8+Y*Q9)))))))
      BESSI0=AX*BX
      ENDIF
      RETURN
      END FUNCTION BESSI0
!     Bessel Function of the 1st kind of order one
!
      Real(chm_real) FUNCTION BESSI1(X)
  use consta
  use number
      IMPLICIT NONE
      Real(chm_real) X,Y,P1,P2,P3,P4,P5,P6,P7,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,AX,BX
      DATA P1,P2,P3,P4,P5,P6,P7/0.5D0,0.87890594D0,0.51498869D0,  &
      0.15084934D0,0.2658733D-1,0.301532D-2,0.32411D-3/
      DATA Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9/0.39894228D0,-0.3988024D-1, &
      -0.362018D-2,0.163801D-2,-0.1031555D-1,0.2282967D-1,        &
      -0.2895312D-1,0.1787654D-1,-0.420059D-2/
      IF(ABS(X).LT.3.75D0) THEN
      Y=(X/3.75D0)**2
      BESSI1=X*(P1+Y*(P2+Y*(P3+Y*(P4+Y*(P5+Y*(P6+Y*P7))))))
      ELSE
      AX=ABS(X)
      Y=3.75D0/AX
      BX=EXP(AX)/DSQRT(AX)
      AX=Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*(Q5+Y*(Q6+Y*(Q7+Y*(Q8+Y*Q9)))))))
      BESSI1=AX*BX
      ENDIF
      RETURN
      END FUNCTION BESSI1
!

#endif /* (main_eef1)*/

end module eef1_mod
