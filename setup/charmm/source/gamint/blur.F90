module gmsblur
  use chm_kinds
  implicit none

contains

#if KEY_GAMESS==1
  SUBROUTINE BLURIN(H,LL2)
    !-----------------------------------------------------------------------
    !
    !! not good to be here.... unfortunately!!   use gamess_fcm
    real(chm_real) MOROKM, DENMAX,ZNUC,CX,CY,CZ,RR,AI,ARRI,AXI,AYI,AZI
    real(chm_real) CSI,CPI,CDI,CFI,CGI,AJ,AA,AA1,DUM,FAC,RHO
    real(chm_real) CSJ,CPJ,CDJ,CFJ,CGJ,AX,AY,AZ,DUM1,DUM2,AAX,AAY,AAZ
    real(chm_real) UU,WW,TT
    INTEGER ISTART,IEND,JSTART,LOCIJ,NATST,NATED,ISAVE,L1,L2
    INTEGER NINT,NSCHWZ,IPCOUNT,IBLUR,IC,I,I1,I2,LOCI,J,J1,J2
    INTEGER LOCJ,IJ,MAX,NX,NY,NZ,IEXCH,ISH,JSH,KSH,LSH,JGMAX,IG
    INTEGER NN,MM,K,IN,JN,LI,LJ,JG
    !
    LOGICAL IANDJ,NORM,DOUBLE,GOPARR,DSKWRK,MASWRK,QBARI,QSHORT
    !
    INTEGER LL2
    real(chm_real)  H(LL2)
    real(chm_real)  VBLK(784),FT(784),DIJ(784),zvblk(784)
    INTEGER IJX(784),IJY(784),IJZ(784)
    real(chm_real)  XIN(343),YIN(343),ZIN(343)
    INTEGER IX(84),IY(84),IZ(84),JX(84),JY(84),JZ(84)
    !
    INTEGER, parameter :: MXSH=5000,MXGTOT=20000
    INTEGER, parameter :: MXATM=2000, MXCHRM=360720
    !
    real(chm_real) XCHM,YCHM,ZCHM,DXELMM,DYELMM,DZELMM,QCHM
    real(chm_real) QQCHG,FQQCHG
    INTEGER NCHMAT,KCHRMM,KGUES,NGAMES,NQQCHG,KHFDFT
    COMMON /CHMGMS/ XCHM(MXCHRM),YCHM(MXCHRM),ZCHM(MXCHRM), &
         DXELMM(MXCHRM),DYELMM(MXCHRM),DZELMM(MXCHRM), &
         QCHM(MXCHRM),NCHMAT,KCHRMM,KGUES,KHFDFT
    COMMON /CGAMES/ NGAMES,NQQCHG,QQCHG(MXATM),FQQCHG(MXATM)
    LOGICAL QGMREM,QGMEXG,QINIGM,QBLUCH,QNOGU
    COMMON /GAMESL/QGMREM,QGMEXG,QINIGM,QBLUCH,QNOGU
    INTEGER,PARAMETER :: MAXBLU=360720
    INTEGER NBLUCH,IBLUCH
    real(chm_real) EBLUCH,CBLUCH
    COMMON /BLUR/EBLUCH(MAXBLU),CBLUCH(MAXBLU),IBLUCH(MAXBLU),NBLUCH
    real(chm_real) CGBLCH,SGBLCH
    COMMON /MBLUR/ CGBLCH(MAXBLU), SGBLCH(MAXBLU)
    !--- !
    real(chm_real) QX,QY,QZ,CKN,CLN,AK
    COMMON /CBARI/ QX,QY,QZ,CKN,CLN,AK
    !
    INTEGER NAT,ICH,MUL,NUM,NNP,NE,NA,NB,IR,IW,IP,IS,IPK,IDAF,NAV,IODA
    real(chm_real) ZAN,C,EX,CS,CP,CD,CF,CG,XX,U,W,RUNTYP,EXETYP
    real(chm_real) CH,CI,VLAMB,SCREEN
    real(chm_real) XINT,YINT,ZINT,TAA,X0,Y0,Z0,XI,YI,ZI,XJ,YJ,ZJ,TOL
    INTEGER KSTART,KATOM,KTYPE,KNG,KLOC,KMIN,KMAX,NSHELL,NI,NJ,IAN
    INTEGER NPRINT,ITOL,ICUT,NORMF,NORMP,NOPK,ME,MASTER,NPROC,IBTYP
    INTEGER IPTIM,NROOTS,NEVALS,II,JJ,LIT,LJT,MINI,MINJ,MAXI,MAXJ
    integer NGLEVL,NHLEVL
    COMMON /INFOA / NAT,ICH,MUL,NUM,NNP,NE,NA,NB,ZAN(MXATM), &
         C(3,MXATM),IAN(MXATM)
    COMMON /IOFILE/ IR,IW,IP,IS,IPK,IDAF,NAV,IODA(950)
    COMMON /NSHEL / EX(MXGTOT),CS(MXGTOT),CP(MXGTOT),CD(MXGTOT), &
         CF(MXGTOT),CG(MXGTOT),CH(MXGTOT),CI(MXGTOT), &
         KSTART(MXSH),KATOM(MXSH),KTYPE(MXSH),KNG(MXSH), &
         KLOC(MXSH),KMIN(MXSH),KMAX(MXSH),NSHELL
    COMMON /OUTPUT/ NPRINT,ITOL,ICUT,NORMF,NORMP,NOPK
    COMMON /PAR   / ME,MASTER,NPROC,IBTYP,IPTIM,GOPARR,DSKWRK,MASWRK
    COMMON /ROOT  / XX,U(13),W(13),NROOTS
    COMMON /RUNOPT/ RUNTYP,EXETYP,NEVALS,NGLEVL,NHLEVL
    COMMON /SCINP / VLAMB,SCREEN
    COMMON /STV   / XINT,YINT,ZINT,TAA,X0,Y0,Z0, &
         XI,YI,ZI,XJ,YJ,ZJ,NI,NJ
    COMMON /SYMIND/ TOL,II,JJ,LIT,LJT,MINI,MINJ,MAXI,MAXJ,IANDJ
    !
    !     These are taken from int2a.src:
    !-----------------------------------
    !                                 NORG needed in SHELLS !
    real(chm_real) GPOPLE,QQ4
    INTEGER NORG,LITQ,LJTQ,LKTQ,LLT,LOCIQ,LOCJQ,LOCKQ,LOCLQ,MINIQ
    INTEGER MINJQ,MINKQ,MINLQ,MAXIQ,MAXJQ,MAXKQ,MAXLQ
    INTEGER NIJQ,IJQ,KLQ,IJKL
    COMMON /GOUT  / GPOPLE(768),NORG
    COMMON /SHLNOS/ QQ4,LITQ,LJTQ,LKTQ,LLT,LOCIQ,LOCJQ,LOCKQ,LOCLQ, &
         MINIQ,MINJQ,MINKQ,MINLQ,MAXIQ,MAXJQ,MAXKQ,MAXLQ, &
         NIJQ,IJQ,KLQ,IJKL
    INTEGER,PARAMETER :: MXGSH=30, MXG2=MXGSH*MXGSH
    !
    real(chm_real) DDIJ(16*MXG2)
    !
    !-----------------------------------
    real(chm_real),parameter :: &
         ZERO=0.0_chm_real, PT5=0.5_chm_real, ONE=1.0_chm_real, &
         TWO=2.0_chm_real, THREE=3.0_chm_real, FIVE=5.0_chm_real, &
         SEVEN=7.0_chm_real,NINE=9.0_chm_real,ELEVEN=11.0_chm_real,&
         E999=999.0_chm_real, THIRTN=13.0_chm_real, &
         FIFTEN=15.0_chm_real, PI212=1.1283791670955_chm_real, &
         SQRT3=1.73205080756888_chm_real, &
         SQRT5=2.23606797749979_chm_real, &
         SQRT7=2.64575131106459_chm_real, &
         RLN10=2.30258_chm_real,PI=3.14159265358979323844_chm_real, &
         PI32=5.56832799683170_chm_real
    !
    DATA JX / 0, 1, 0, 0, 2, 0, 0, 1, 1, 0, &
         3, 0, 0, 2, 2, 1, 0, 1, 0, 1, &
         4, 0, 0, 3, 3, 1, 0, 1, 0, 2, &
         2, 0, 2, 1, 1, &
         5, 0, 0, 4, 4, 1, 0, 1, 0, 3, &
         3, 2, 0, 2, 0, 3, 1, 1, 2, 2, &
         1,     &
         6, 0, 0, 5, 5, 1, 0, 1, 0, 4, &
         4, 2, 0, 2, 0, 4, 1, 1, 3, 3, &
         0, 3, 3, 2, 1, 2, 1, 2/
    DATA IX / 1, 8, 1, 1,15, 1, 1, 8, 8, 1, &
         22, 1, 1,15,15, 8, 1, 8, 1, 8, &
         29, 1, 1,22,22, 8, 1, 8, 1,15, &
         15, 1,15, 8, 8,  &
         36, 1, 1,29,29, 8, 1, 8, 1,22,  &
         22,15, 1,15, 1,22, 8, 8,15,15, &
         8, &
         43, 1, 1,36,36, 8, 1, 8, 1,29, &
         29,15, 1,15, 1,29, 8, 8,22,22, &
         1,22,22,15, 8,15, 8,15/ 
    DATA JY / 0, 0, 1, 0, 0, 2, 0, 1, 0, 1, &
         0, 3, 0, 1, 0, 2, 2, 0, 1, 1, &
         0, 4, 0, 1, 0, 3, 3, 0, 1, 2, &
         0, 2, 1, 2, 1,  &
         0, 5, 0, 1, 0, 4, 4, 0, 1, 2, &
         0, 3, 3, 0, 2, 1, 3, 1, 2, 1, &
         2,  &
         0, 6, 0, 1, 0, 5, 5, 0, 1, 2, &
         0, 4, 4, 0, 2, 1, 4, 1, 3, 0, &
         3, 2, 1, 3, 3, 1, 2, 2/
    DATA IY / 1, 1, 8, 1, 1,15, 1, 8, 1, 8, &
         1,22, 1, 8, 1,15,15, 1, 8, 8, &
         1,29, 1, 8, 1,22,22, 1, 8,15, &
         1,15, 8,15, 8, &
         1,36, 1, 8, 1,29,29, 1, 8,15, &
         1,22,22, 1,15, 8,22, 8,15, 8, &
         15, &
         1,43, 1, 8, 1,36,36, 1, 8,15, &
         1,29,29, 1,15, 8,29, 8,22, 1, &
         22,15, 8,22,22, 8,15,15/ 
    DATA JZ / 0, 0, 0, 1, 0, 0, 2, 0, 1, 1, &
         0, 0, 3, 0, 1, 0, 1, 2, 2, 1, &
         0, 0, 4, 0, 1, 0, 1, 3, 3, 0, &
         2, 2, 1, 1, 2,  &
         0, 0, 5, 0, 1, 0, 1, 4, 4, 0, &
         2, 0, 2, 3, 3, 1, 1, 3, 1, 2, &
         2, &
         0, 0, 6, 0, 1, 0, 1, 5, 5, 0, &
         2, 0, 2, 4, 4, 1, 1, 4, 0, 3, &
         3, 1, 2, 1, 2, 3, 3, 2/
    DATA IZ / 1, 1, 1, 8, 1, 1,15, 1, 8, 8, &
         1, 1,22, 1, 8, 1, 8,15,15, 8, &
         1, 1,29, 1, 8, 1, 8,22,22, 1, &
         15,15, 8, 8,15, &
         1, 1,36, 1, 8, 1, 8,29,29, 1, &
         15, 1,15,22,22, 8, 8,22, 8,15, &
         15, &
         1, 1,43, 1, 8, 1, 8,36,36, 1, &
         15, 1,15,29,29, 8, 8,29, 1,22, &
         22, 8,15, 8,15,22,22,15/
    !
    DATA MOROKM/8HMOROKUMA/
    !
    !     ----- COMPUTE ADDITIONAL H INTEGRALS DUE TO CHARMM FIELD -----
    !
    TOL = RLN10*ITOL
    NORM = NORMF .NE. 1 .OR. NORMP .NE. 1
    !
    !     ----- RESET SOME PARAMETERS FOR MOROKUMA DECOMPOSITIONS -----
    !     ISAVE .EQ. 0 : SAVE S, H, AND T TO DAF 12, 11, AND 13
    !     ISAVE .EQ. 1 : SAVE S, H, AND T TO DAF 12, 11, AND 13
    !                    AND SAVE S AND H TO DAF 312 AND 311
    !     NOTE THAT LL2 IS ALWAYS (NUM*NUM+NUM)/2,
    !     L1,L2 MAY BE SMALLER THAN USUAL FOR A MONOMER IN A MOROKUMA RUN
    !
    IF (RUNTYP.EQ.MOROKM) THEN
       CALL STINT1(ISTART,IEND,JSTART,LOCIJ,NATST,NATED,ISAVE,L1,L2)
    ELSE
       ISTART = 1
       IEND   = NSHELL
       JSTART = 1
       LOCIJ  = 0
       NATST  = NAT+1
       NATED  = NAT+NCHMAT
       ISAVE  = 0
       L1 = NUM
       L2 = (NUM*(NUM+1))/2
    END IF
    !
    NINT  = 0
    NSCHWZ= 0
    DENMAX = ZERO
    !
    !     ----- INTIALIZE PARALLEL -----
    !
    IPCOUNT = ME - 1
    !
    !     This is the loop for external charges. Because
    !     the ``blur'' method needs data before the inner loop
    !     the loop over the charge
    !     centers has to be taken out from the inner loop.
    !
    !     -NCHMAT- IS NONZERO IF THERE ARE EXTERNAL CHARGES WHICH
    !     PERTURB THE SYSTEM, SUCH AS IF CHARMM IS IN USE.  NOTE
    !     THAT THERE IS ALSO A NUCLEAR REPULSION TERM WHICH IS NOT
    !     INCLUDED HERE, IT IS IN THE CHARMM INTERFACE CODE.
    !
    !
    IBLUR=1
    loop460:DO IC = NATST,NATED
       ZNUC = -QCHM(IC-NAT)
       CX = XCHM(IC-NAT)
       CY = YCHM(IC-NAT)
       CZ = ZCHM(IC-NAT)
       !
       QBARI=.FALSE.
       QSHORT=.FALSE.
       AK=ZERO
       IF(QBLUCH.AND.(IBLUCH(IBLUR).EQ.(IC-NAT))) THEN
          AK=PT5/SGBLCH(IBLUR)/SGBLCH(IBLUR)
          ZNUC=-CGBLCH(IBLUR)
          QBARI=.TRUE.
          QSHORT=.FALSE.
          IBLUR=IBLUR+1
       ENDIF
       !        This is the normalization factor
       CKN=TWO*AK/PI
       CKN=CKN*CKN*CKN
       CKN=SQRT(CKN)
       CKN=SQRT(CKN)
       CLN=ZNUC*CKN
       QX=CX
       QY=CY
       QZ=CZ
       !
       !     ----- I SHELL -----
       !
       loop720:DO II = ISTART,IEND
          I = KATOM(II)
          XI = C(1,I)
          YI = C(2,I)
          ZI = C(3,I)
          I1 = KSTART(II)
          I2 = I1+KNG(II)-1
          LIT = KTYPE(II)
          MINI = KMIN(II)
          MAXI = KMAX(II)
          LOCI = KLOC(II)-MINI-LOCIJ
          !      
          !     ----- J SHELL -----
          ! 
          loop700: DO JJ = JSTART,II
             !
             !     ----- GO PARALLEL! -----
             !
             IF (GOPARR) THEN
                IPCOUNT = IPCOUNT + 1
                IF (MOD(IPCOUNT,NPROC).NE.0) cycle loop700
             END IF
             J = KATOM(JJ)
             XJ = C(1,J)
             YJ = C(2,J)
             ZJ = C(3,J)
             J1 = KSTART(JJ)
             J2 = J1+KNG(JJ)-1
             LJT = KTYPE(JJ)
             MINJ = KMIN(JJ)
             MAXJ = KMAX(JJ)
             LOCJ = KLOC(JJ)-MINJ-LOCIJ
             NROOTS = (LIT+LJT-2)/2+1
             RR = (XI-XJ)**2+(YI-YJ)**2+(ZI-ZJ)**2
             IANDJ = II .EQ. JJ
             !
             !     ----- PREPARE INDICES FOR PAIRS OF (I,J) FUNCTIONS
             !
             IJ = 0
             MAX = MAXJ
             DO I = MINI,MAXI
                NX = IX(I)
                NY = IY(I)
                NZ = IZ(I)
                IF (IANDJ) MAX = I
                DO J = MINJ,MAX
                   IJ = IJ+1
                   IJX(IJ) = NX+JX(J)
                   IJY(IJ) = NY+JY(J)
                   IJZ(IJ) = NZ+JZ(J)
                   IF  (J.LE. 1)                FT(IJ) = THREE
                   IF ((J.GE. 2).AND.(J.LE. 4)) FT(IJ) = FIVE
                   IF ((J.GE. 5).AND.(J.LE.10)) FT(IJ) = SEVEN
                   IF ((J.GE.11).AND.(J.LE.20)) FT(IJ) = NINE
                   IF ((J.GE.21).AND.(J.LE.35)) FT(IJ) = ELEVEN
                   IF ((J.GE.36).AND.(J.LE.56)) FT(IJ) = THIRTN
                   IF ((J.GE.57).AND.(J.LE.84)) FT(IJ) = FIFTEN
                enddo
             enddo
             !
             zVBLK(1:ij) = ZERO
             VBLK(1:ij) = ZERO
             !
             IF (QBARI.AND.(.NOT.QSHORT)) THEN
                !
                !     ----- K,L SHELL ----- For BARI ignore the loops and set 
                !                           the indeces to be always 1
                !
                IEXCH = 1
                ISH = II
                JSH = JJ
                KSH = 1
                LSH = 1
                QQ4 = ONE
                !
                !        ----- COMPUTE TWO-ELECTRON INTEGRALS ----
                !
                !     USE HONDO RYS POLYNOMIAL CODE FOR OTHER BLOCKS
                !
                !        ----- GET INFORMATION ABOUT ISH AND JSH -----
                !        ----- FORM PAIRS OF PRIMITIVES FROM ISH AND JSH -----
                !        ----- GET INFORMATION ABOUT KSH AND LSH -----
                !
                NORG=0
                CALL SHELLS(1,ISH,JSH,KSH,LSH)
                CALL IJPRIM1(DDIJ)
                !     Info which is needed, but is missing because call to
                !     SHELLS(2,ISH,JSH,KSH,LSH) would be meaningless for BARI
                !     Provide here:
                IJKL=IJ
                !
                !        ----- DO INTEGRAL BATCH, SSSS IS A SPECIAL CASE -----
                !                                 but general can handle them too!
                !
                CALL BARIGN(VBLK,DDIJ)
             else
                !
                !     ----- I PRIMITIVE
                !
                JGMAX = J2
                loop520: DO IG = I1,I2
                   AI = EX(IG)
                   ARRI = AI*RR
                   AXI = AI*XI
                   AYI = AI*YI
                   AZI = AI*ZI
                   CSI = CS(IG)
                   CPI = CP(IG)
                   CDI = CD(IG)
                   CFI = CF(IG)
                   CGI = CG(IG)
                   !
                   !     ----- J PRIMITIVE
                   !
                   IF (IANDJ) JGMAX = IG
                   loop500: DO JG = J1,JGMAX
                      AJ = EX(JG)
                      AA = AI+AJ
                      AA1 = ONE/AA
                      DUM = AJ*ARRI*AA1
                      IF (QSHORT.AND.QBARI.AND.(AK.EQ.ZERO)) cycle loop500
                      IF (DUM .GT. TOL) cycle loop500
                      FAC = EXP(-DUM)
                      CSJ = CS(JG)
                      CPJ = CP(JG)
                      CDJ = CD(JG)
                      CFJ = CF(JG)
                      CGJ = CG(JG)
                      AX = (AXI+AJ*XJ)*AA1
                      AY = (AYI+AJ*YJ)*AA1
                      AZ = (AZI+AJ*ZJ)*AA1
                      !
                      !     ----- DENSITY FACTOR
                      !
                      DOUBLE=IANDJ.AND.IG.NE.JG
                      MAX = MAXJ
                      NN = 0
                      DUM1 = ZERO
                      DUM2 = ZERO
                      loop220: DO I = MINI,MAXI
                         IF (I.EQ.1) DUM1=CSI*FAC
                         IF (I.EQ.2) DUM1=CPI*FAC
                         IF (I.EQ.5) DUM1=CDI*FAC
                         IF ((I.EQ. 8).AND.NORM) DUM1=DUM1*SQRT3
                         IF (I.EQ.11) DUM1=CFI*FAC
                         IF ((I.EQ.14).AND.NORM) DUM1=DUM1*SQRT5
                         IF ((I.EQ.20).AND.NORM) DUM1=DUM1*SQRT3
                         IF (I.EQ.21) DUM1=CGI*FAC
                         IF ((I.EQ.24).AND.NORM) DUM1=DUM1*SQRT7
                         IF ((I.EQ.30).AND.NORM) DUM1=DUM1*SQRT5/SQRT3
                         IF ((I.EQ.33).AND.NORM) DUM1=DUM1*SQRT3
                         IF (IANDJ) MAX = I
                         loop200: DO J = MINJ,MAX
                            IF (J.EQ.1) THEN
                               DUM2=DUM1*CSJ
                               IF (DOUBLE) THEN
                                  IF (I.LE.1) THEN
                                     DUM2=DUM2+DUM2
                                  ELSE
                                     DUM2=DUM2+CSI*CPJ*FAC
                                  END IF
                               END IF
                            ELSE IF (J.EQ.2) THEN
                               DUM2=DUM1*CPJ
                               IF (DOUBLE) DUM2=DUM2+DUM2
                            ELSE IF (J.EQ.5) THEN
                               DUM2=DUM1*CDJ
                               IF (DOUBLE) DUM2=DUM2+DUM2
                            ELSE IF ((J.EQ.8).AND.NORM) THEN
                               DUM2=DUM2*SQRT3
                            ELSE IF (J.EQ.11) THEN
                               DUM2=DUM1*CFJ
                               IF (DOUBLE) DUM2=DUM2+DUM2
                            ELSE IF ((J.EQ.14).AND.NORM) THEN
                               DUM2=DUM2*SQRT5
                            ELSE IF ((J.EQ.20).AND.NORM) THEN
                               DUM2=DUM2*SQRT3
                            ELSE IF (J.EQ.21) THEN
                               DUM2=DUM1*CGJ
                               IF (DOUBLE) DUM2=DUM2+DUM2
                            ELSE IF ((J.EQ.24).AND.NORM) THEN
                               DUM2=DUM2*SQRT7
                            ELSE IF ((J.EQ.30).AND.NORM) THEN
                               DUM2=DUM2*SQRT5/SQRT3
                            ELSE IF ((J.EQ.33).AND.NORM) THEN
                               DUM2=DUM2*SQRT3
                            END IF
                            NN = NN+1
                            DIJ(NN) = DUM2
                         enddo loop200
                      enddo loop220
                      !
                      !     ----- NUCLEAR ATTRACTION
                      !
                      !                    PI212 = TWO/SQRT(PI), PI32 = PI**(3/2)
                      !
                      DUM = PI212*AA1
                      IF(QBARI.AND.QSHORT) &
                           DUM=DUM*PI32*CKN*CKN/(TWO*AK*SQRT(AA+TWO*AK))
                      DIJ(1:ij) = DIJ(1:ij)*DUM
                      !
                      AAX = AA*AX
                      AAY = AA*AY
                      AAZ = AA*AZ
                      !
                      IF (QBARI.AND.QSHORT) THEN
                         RHO=AA*TWO*AK/(AA+AK+AK)
                         XX = RHO*((AX-CX)**2+(AY-CY)**2+(AZ-CZ)**2)
                      ELSE
                         XX = AA*((AX-CX)**2+(AY-CY)**2+(AZ-CZ)**2)
                      ENDIF
                      IF (NROOTS.LE.3) CALL RT123
                      IF (NROOTS.EQ.4) CALL ROOT4
                      IF (NROOTS.EQ.5) CALL ROOT5
                      MM = 0
                      DO K = 1,NROOTS
                         IF(QBARI.AND.QSHORT) THEN
                            UU = RHO*U(K)
                         ELSE
                            UU = AA*U(K)
                         ENDIF
                         WW = W(K)*ZNUC
                         TT = ONE/(AA+UU)
                         TAA = SQRT(TT)
                         X0 = (AAX+UU*CX)*TT
                         Y0 = (AAY+UU*CY)*TT
                         Z0 = (AAZ+UU*CZ)*TT
                         IN = -7+MM
                         DO I = 1,LIT
                            IN = IN+7
                            NI = I
                            DO J = 1,LJT
                               JN = IN+J
                               NJ = J
                               CALL STVINT
                               XIN(JN) = XINT
                               YIN(JN) = YINT
                               ZIN(JN) = ZINT*WW
                            enddo
                         enddo
                         MM = MM+49
                      enddo
                      DO I = 1,IJ
                         NX = IJX(I)
                         NY = IJY(I)
                         NZ = IJZ(I)
                         DUM = ZERO
                         MM = 0
                         DO K = 1,NROOTS
                            DUM = DUM+XIN(NX+MM)*YIN(NY+MM)*ZIN(NZ+MM)
                            MM = MM+49
                         enddo
                         VBLK(I) = VBLK(I) + DUM*DIJ(I)
                      enddo
                      !
                      !     ----- END OF PRIMITIVE LOOPS -----
                      !
                   enddo loop500
                enddo loop520
             endif
             !
             !     ----- COPY BLOCK INTO H-CORE, OVERLAP, AND KINETIC ENERGY MATRICES
             !
             MAX = MAXJ
             NN = 0
             DO I = MINI,MAXI
                LI = LOCI+I
                IN = (LI*(LI-1))/2
                IF (IANDJ) MAX = I
                DO J = MINJ,MAX
                   LJ = LOCJ+J
                   JN = LJ+IN
                   NN = NN+1
                   H(JN) = H(JN) + VBLK(NN)
                enddo
             enddo
             !
             !     ----- END OF SHELL LOOPS -----
             !
          enddo loop700
       enddo loop720
    enddo loop460
    !
    RETURN
  END subroutine blurin
  !
  SUBROUTINE BARI00(GHONDO,DDIJ)
    !
    LOGICAL IANDJ,KANDL,SAME,OUT
    !
    integer,parameter :: MXGSH=30, MXG2=MXGSH*MXGSH
    !
    real(chm_real) GHONDO(*),DDIJ(16*MXG2)
    !
    !     For blur
    real(chm_real) qx,qy,qz,ckn,cln,ak
    COMMON /CBARI/ QX,QY,QZ,CKN,CLN,AK
    !
    real(chm_real) a,r,x1,y1,z1,ijd
    COMMON /IJGNRL/ A(MXG2),R(MXG2),X1(MXG2),Y1(MXG2),Z1(MXG2), &
         IJD(784)
    real(chm_real) gpople,norg,ag,csa,cpa,cda,cfa,cga,cha,cia
    real(chm_real) bg,csb,cpb,cdb,cfb,cgb,chb,cib
    real(chm_real) cg,csc,cpc,cdc,cfc,cgc,chc,cic
    real(chm_real) dg,csd,cpd,cdd,cfd,cgd,chd,cid,xi,yi,zi,xj,yj,zj,rri
    real(chm_real) xk,yk,zk,xl,yl,zl,rrk
    integer nga,ngb,ngc,ngd
    COMMON /GOUT  / GPOPLE(768),NORG
    COMMON /MISC  / IANDJ,KANDL,SAME
    COMMON /SHLINF/ AG(MXGSH),CSA(MXGSH),CPA(MXGSH),CDA(MXGSH), &
         CFA(MXGSH),CGA(MXGSH),CHA(mxgsh),CIA(mxgsh), &
         BG(MXGSH),CSB(MXGSH),CPB(MXGSH),CDB(MXGSH), &
         CFB(MXGSH),CGB(MXGSH),CHb(mxgsh),CIb(mxgsh), &
         CG(MXGSH),CSC(MXGSH),CPC(MXGSH),CDC(MXGSH), &
         CFC(MXGSH),CGC(MXGSH),CHc(mxgsh),CIc(mxgsh), &
         DG(MXGSH),CSD(MXGSH),CPD(MXGSH),CDD(MXGSH), &
         CFD(MXGSH),CGD(MXGSH),CHd(mxgsh),CId(mxgsh), &
         XI,YI,ZI,XJ,YJ,ZJ,RRI,XK,YK,ZK,XL,YL,ZL,RRK, &
         NGA,NGB,NGC,NGD
    real(chm_real) qq4,tol,cutoff
    integer lit,ljt,lkt,llt,loci,locj,lock,locl,mini,minj,mink,minl
    integer maxi,maxl,maxk,maxj,nij,ij,kl,ijkl,icount
    COMMON /SHLNOS/ QQ4,LIT,LJT,LKT,LLT,LOCI,LOCJ,LOCK,LOCL, &
         MINI,MINJ,MINK,MINL,MAXI,MAXJ,MAXK,MAXL, &
         NIJ,IJ,KL,IJKL
    COMMON /SHLT  / TOL,CUTOFF,ICOUNT,OUT
    !
    real(chm_real),PARAMETER :: PI252=34.986836655250_chm_real, &
         PIE4=7.85398163397448_chm_real, &
         ZERO=0.0_chm_real,ONE=1.0_chm_real
    !
    !     SPECIAL SSSS INTEGRAL ROUTINE WHEN USING HONDO INTEGRALS
    !
    real(chm_real) aa,ab,bb,bbinv,bbrrk,bbx,bby,bbz,bk,bl,brrk,bxk
    real(chm_real) byk,bzk,csk,d2,dum,e,expe,f1,ggout,sum,ww1
    real(chm_real) x,xinv,xx,y
    integer n,nn
    !
    GGOUT = ZERO
    BK = AK
    RRK=ZERO
    BRRK = ZERO
    BXK = BK*QX
    BYK = BK*QY
    BZK = BK*QZ
    CSK = CKN
    BL = AK
    BB = BK+BL
    BBINV = ONE/BB
    DUM = BL*BRRK*BBINV
    IF (DUM .GT. TOL) GO TO 280
    BBRRK = DUM
    D2 = CLN*CSK*BBINV
    BBX = QX
    BBY = QY
    BBZ = QZ
    SUM = ZERO
    NN = 1
    loop260: DO N = 1,NIJ
       DUM = BBRRK+R(N)
       IF (DUM .GT. TOL) cycle loop260
       EXPE = EXP(-DUM)
       AA = A(N)
       AB = AA+BB
       DUM = X1(N)-BBX
       XX = DUM*DUM
       DUM = Y1(N)-BBY
       XX = DUM*DUM+XX
       DUM = Z1(N)-BBZ
       XX = DUM*DUM+XX
       X = XX*AA*BB/AB
       !
       IF (X .GT. 5.0_chm_real) GO TO 160
       IF (X .GT. 1.0_chm_real) GO TO 120
       IF (X .GT. 3.0D-07) GO TO 100
       WW1 = 1.0_chm_real-X/3.0_chm_real
       GO TO 240
       !
100    CONTINUE
       F1 = ((((((((-8.36313918003957D-08*X+1.21222603512827D-06 )*X- &
            1.15662609053481D-05 )*X+9.25197374512647D-05 )*X- &
            6.40994113129432D-04 )*X+3.78787044215009D-03 )*X- &
            1.85185172458485D-02 )*X+7.14285713298222D-02 )*X- &
            1.99999999997023D-01 )*X+3.33333333333318D-01
       WW1 = (X+X)*F1+EXP(-X)
       GO TO 240
       !
120    CONTINUE
       IF (X .GT. 3.0_chm_real) GO TO 140
       Y = X-2.0_chm_real
       F1 = ((((((((((-1.61702782425558D-10*Y+1.96215250865776D-09 )*Y- &
            2.14234468198419D-08 )*Y+2.17216556336318D-07 )*Y- &
            1.98850171329371D-06 )*Y+1.62429321438911D-05 )*Y- &
            1.16740298039895D-04 )*Y+7.24888732052332D-04 )*Y- &
            3.79490003707156D-03 )*Y+1.61723488664661D-02 )*Y- &
            5.29428148329736D-02 )*Y+1.15702180856167D-01
       WW1 = (X+X)*F1+EXP(-X)
       GO TO 240
       !
140    CONTINUE
       Y = X-4.0_chm_real
       F1 = ((((((((((-2.62453564772299D-11*Y+3.24031041623823D-10 )*Y- &
            3.614965656163D-09)*Y+3.760256799971D-08)*Y- &
            3.553558319675D-07)*Y+3.022556449731D-06)*Y- &
            2.290098979647D-05)*Y+1.526537461148D-04)*Y- &
            8.81947375894379D-04 )*Y+4.33207949514611D-03 )*Y- &
            1.75257821619926D-02 )*Y+5.28406320615584D-02
       WW1 = (X+X)*F1+EXP(-X)
       GO TO 240
       !
160    CONTINUE
       IF (X .GT. 15.0_chm_real) GO TO 200
       E = EXP(-X)
       IF (X .GT. 10.0_chm_real) GO TO 180
       XINV = ONE/X
       WW1 = (((((( 4.6897511375022D-01*XINV-6.9955602298985D-01)*XINV + &
            5.3689283271887D-01)*XINV-3.2883030418398D-01)*XINV + &
            2.4645596956002D-01)*XINV-4.9984072848436D-01)*XINV - &
            3.1501078774085D-06)*E + SQRT(PIE4*XINV)
       GO TO 240
       !
180    CONTINUE
       XINV = ONE/X
       WW1 = (((-1.8784686463512D-01*XINV+2.2991849164985D-01)*XINV &
            -4.9893752514047D-01)*XINV-2.1916512131607D-05)*E &
            + SQRT(PIE4*XINV)
       GO TO 240
       !
200    CONTINUE
       IF (X < 33.0_chm_real) then !GO TO 220
          XINV = ONE/X
          E = EXP(-X)
          WW1 = (( 1.9623264149430D-01*XINV-4.9695241464490D-01)*XINV - &
               6.0156581186481D-05)*E + SQRT(PIE4*XINV)

       else
          WW1 = SQRT(PIE4/X)
       endif
       !
240    SUM = SUM+DDIJ(NN)*WW1*EXPE/SQRT(AB)
       NN = NN+16
    enddo loop260
    GGOUT = GGOUT+D2*SUM
280 CONTINUE
    GHONDO(1) = GGOUT*PI252*QQ4
    RETURN
  END subroutine bari00
  !
  SUBROUTINE BARIGN(GHONDO,DDIJ)
    !
    real(chm_real) GHONDO(*),DDIJ(*)
    !
    LOGICAL IANDJ,KANDL,SAME,OUT,NORM,DOUBLE
    !
    integer,parameter :: MXGSH=30, MXG2=MXGSH*MXGSH
    !
    real(chm_real) QX,QY,QZ,CKN,CLN,AK
    COMMON /CBARI/ QX,QY,QZ,CKN,CLN,AK
    !
    integer ijd,nprint,itol,icut,normf,normp,nopk,nroots
    real(chm_real) dkl, dij,aa,r,x1,y1,z1,xx,u,w
    COMMON /DENS  / DKL(784),DIJ(784)
    COMMON /IJGNRL/ AA(MXG2),R(MXG2),X1(MXG2),Y1(MXG2),Z1(MXG2), &
         IJD(784)
    COMMON /MISC  / IANDJ,KANDL,SAME
    COMMON /OUTPUT/ NPRINT,ITOL,ICUT,NORMF,NORMP,NOPK
    COMMON /ROOT  / XX,U(13),W(13),NROOTS
    integer in,kn,ni,nj,nk,nl,nmax,mmax
    real(chm_real) bp01,b00,b10,xcp00,xc00,ycp00,yc00,zcp00,zc00,f00
    real(chm_real)  DXIJ,DYIJ,DZIJ,DXKL,DYKL,DZKL
    COMMON /SETINT/ IN(13),KN(13),NI,NJ,NK,NL,NMAX,MMAX, &
         BP01,B00,B10,XCP00,XC00,YCP00,YC00,ZCP00,ZC00,F00, &
         DXIJ,DYIJ,DZIJ,DXKL,DYKL,DZKL
    real(chm_real) gpople,norg,ag,csa,cpa,cda,cfa,cga,cha,cia
    real(chm_real) bg,csb,cpb,cdb,cfb,cgb,chb,cib
    real(chm_real) cg,csc,cpc,cdc,cfc,cgc,chc,cic
    real(chm_real) dg,csd,cpd,cdd,cfd,cgd,chd,cid,xi,yi,zi,xj,yj,zj,rri
    real(chm_real) xk,yk,zk,xl,yl,zl,rrk
    integer nga,ngb,ngc,ngd
    COMMON /SHLINF/ AG(MXGSH),CSA(MXGSH),CPA(MXGSH),CDA(MXGSH), &
         CFA(MXGSH),CGA(MXGSH),CHA(mxgsh),CIA(mxgsh), &
         BG(MXGSH),CSB(MXGSH),CPB(MXGSH),CDB(MXGSH), &
         CFB(MXGSH),CGB(MXGSH),CHb(mxgsh),CIb(mxgsh), &
         CG(MXGSH),CSC(MXGSH),CPC(MXGSH),CDC(MXGSH), &
         CFC(MXGSH),CGC(MXGSH),CHc(mxgsh),CIc(mxgsh), &
         DG(MXGSH),CSD(MXGSH),CPD(MXGSH),CDD(MXGSH), &
         CFD(MXGSH),CGD(MXGSH),CHd(mxgsh),CId(mxgsh), &
         XI,YI,ZI,XJ,YJ,ZJ,RRI,XK,YK,ZK,XL,YL,ZL,RRK, &
         NGA,NGB,NGC,NGD

    integer LIT,LJT,LKT,LLT,LOCI,LOCJ,LOCK,LOCL,icount
    integer MINI,MINJ,MINK,MINL,MAXI,MAXJ,MAXK,MAXL,NIJ,IJ,KL,IJKL
    real(chm_real) qq4,tol,cutoff
    COMMON /SHLNOS/ QQ4,LIT,LJT,LKT,LLT,LOCI,LOCJ,LOCK,LOCL, &
         MINI,MINJ,MINK,MINL,MAXI,MAXJ,MAXK,MAXL, &
         NIJ,IJ,KL,IJKL
    COMMON /SHLT  / TOL,CUTOFF,ICOUNT,OUT
    !
    INTEGER IN1(9)
    !
    real(chm_real), parameter :: &
         SQRT3=1.73205080756888_chm_real, &
         SQRT5=2.23606797749979_chm_real, &
         SQRT7=2.64575131106459_chm_real, &
         PI252=34.986836655250_chm_real, &
         ZERO=0.0_chm_real, HALF=0.5_chm_real,ONE=1.0_chm_real
    !
    real(chm_real) a,aandb,ab,akxk,akyk,akzk,al,axai,axak,ayai,ayak
    real(chm_real) azai,azak,b,bbrrk,binv,brrk,bxbi,bxbk,bybi,bybk
    real(chm_real) bzbi,bzbk,c1x,c1y,c1z,c2x,c2y,c2z,c3x,c3y,c3z
    real(chm_real) c4x,c4y,c4z,csk,csl,dum,duminv,expe,factor
    real(chm_real) dm2inv, rho, u2, xa, xb, ya, yb, za, zb, d1
    integer i, k, m,max, mm, n, nn
    !
    !     GENERAL INTEGRAL ROUTINE FOR SPD FUNCTIONS
    !
    !     PI252 = PI**2.5*2
    FACTOR = PI252*QQ4
    NI = LIT-1
    NJ = LJT-1
    NK = 0
    NL = 0
    DXIJ = XI-XJ
    DYIJ = YI-YJ
    DZIJ = ZI-ZJ
    DXKL = ZERO
    DYKL = ZERO
    DZKL = ZERO
    NMAX = NI+NJ
    MMAX = NK+NL
    MAX = NMAX+1
    DO I = 1,MAX
       N = I-1
       IF (N .LE. NI) IN1(I) = 125*N+1
       IF (N .GT. NI) IN1(I) = 125*NI+25*(N-NI)+1
    enddo
    MAX = MMAX+1
    DO K = 1,MAX
       N = K-1
       IF (N .LE. NK) KN(K) = 5*N
       IF (N .GT. NK) KN(K) = 5*NK+N-NK
    enddo
    !     KN(2) is actually accessed in XYZINT, but then never used!
    KN(2)=0
    !
    !     ----- K,L PRIMITIVE - loops deleted
    !     some variables here are maybe not needed (RRK,...)
    !
    RRK = ZERO
    BRRK = ZERO
    AKXK = AK*QX
    AKYK = AK*QY
    AKZK = AK*QZ
    CSK = CKN*FACTOR
    AL = AK
    B = AK+AL
    BINV = ONE/B
    BBRRK = AL*BRRK*BINV
    CSL = CLN
    XB = QX
    YB = QY
    ZB = QZ
    BXBK = ZERO
    BYBK = ZERO
    BZBK = ZERO
    BXBI = B*(XB-XI)
    BYBI = B*(YB-YI)
    BZBI = B*(ZB-ZI)
    !
    !           ----- DENSITY FACTOR
    !
    DKL(1) = CSK*BINV*CSL
    !
    !           ----- PAIR OF I,J PRIMITIVES
    !
    NN = 0
    loop440: DO N = 1,NIJ
       DUM = BBRRK+R(N)
       DO I = 1,IJ
          DIJ(I) = DDIJ(IJD(I)+NN)
       enddo
       A = AA(N)
       AB = A*B
       AANDB = A+B
       EXPE = EXP(-DUM)/SQRT(AANDB)
       RHO = AB/AANDB
       XA = X1(N)
       YA = Y1(N)
       ZA = Z1(N)
       XX = RHO*((XA-XB)*(XA-XB) + (YA-YB)*(YA-YB) &
            + (ZA-ZB)*(ZA-ZB))
       AXAK = A*(XA-QX)
       AYAK = A*(YA-QY)
       AZAK = A*(ZA-QZ)
       AXAI = A*(XA-XI)
       AYAI = A*(YA-YI)
       AZAI = A*(ZA-ZI)
       C1X = BXBK+AXAK
       C2X = A*BXBK
       C3X = BXBI+AXAI
       C4X = B*AXAI
       C1Y = BYBK+AYAK
       C2Y = A*BYBK
       C3Y = BYBI+AYAI
       C4Y = B*AYAI
       C1Z = BZBK+AZAK
       C2Z = A*BZBK
       C3Z = BZBI+AZAI
       C4Z = B*AZAI
       !
       !              ----- ROOTS AND WEIGHTS FOR QUADRATURE
       !
       IF (NROOTS .LE. 3) CALL RT123
       IF (NROOTS .EQ. 4) CALL ROOT4
       IF (NROOTS .EQ. 5) CALL ROOT5
       IF (NROOTS .GE. 6) CALL ROOT6
       MM = 0
       MAX = NMAX+1
       !
       !              COMPUTE TWO-ELECTRON INTEGRALS FOR EACH ROOT
       !
       DO M = 1,NROOTS
          U2 = U(M)*RHO
          F00 = EXPE*W(M)
          DO I = 1,MAX
             IN(I) = IN1(I)+MM
          enddo
          DUMINV = ONE/(AB+U2*AANDB)
          DM2INV = HALF*DUMINV
          BP01 = (A+U2)*DM2INV
          B00 = U2*DM2INV
          B10 = (B+U2)*DM2INV
          XCP00 = (U2*C1X+C2X)*DUMINV
          XC00 = (U2*C3X+C4X)*DUMINV
          YCP00 = (U2*C1Y+C2Y)*DUMINV
          YC00 = (U2*C3Y+C4Y)*DUMINV
          ZCP00 = (U2*C1Z+C2Z)*DUMINV
          ZC00 = (U2*C3Z+C4Z)*DUMINV
          CALL XYZINT1
          MM = MM+625
       enddo
       !
       !              ----- FORM (I,J//K,L) INTEGRALS OVER FUNCTIONS
       !
       CALL FORMS1(GHONDO)  
       NN = NN+16
    enddo loop440
    !
    RETURN
  END subroutine barign
  !
  SUBROUTINE FORMS1(GHONDO)
    !-----------------------------------------------------------------------
    !
    !     This routine is taken from int2a.src(forms) and modified for blur
    !     functions. Note that IJX(),IJY(),IJZ() are different than
    !     the ones in SUBROUTINE BLURIN, because XIN(),YIN(),ZIN()
    !     are calculated that way....
    !
    real(chm_real) GHONDO(*)
    !
    real(chm_real) dkl, dij, qq4
    integer ijgt,ijx,ijy,ijz,ik,klgt,klx,kly,klz
    COMMON /DENS  / DKL(784),DIJ(784)
    COMMON /INTDEX/ IJGT(784),IJX(784),IJY(784),IJZ(784),IK(784), &
         KLGT(784),KLX(784),KLY(784),KLZ(784)
    integer LIT,LJT,LKT,LLT,LOCI,LOCJ,LOCK,LOCL,MINI,MINJ,MINK,MINL
    integer MAXI,MAXJ,MAXK,MAXL,NIJ,IJ,KL,IJKL,nroots
    COMMON /SHLNOS/ QQ4,LIT,LJT,LKT,LLT,LOCI,LOCJ,LOCK,LOCL, &
         MINI,MINJ,MINK,MINL,MAXI,MAXJ,MAXK,MAXL, &
         NIJ,IJ,KL,IJKL
    real(chm_real) xx,u,w,xin,yin,zin
    COMMON /ROOT  / XX,U(13),W(13),NROOTS
    COMMON /XYZ   / XIN(31213),YIN(31213),ZIN(31213)
    !
    real(chm_real) d1
    integer i,nx,ny,nz
    !
    !     ----- FORM INTEGRALS OVER FUNCTIONS -----
    !     DIMENSIONING XIN(81,5), AND ROLLING UP THE COMPUTATION
    !     OF GHONDO IN A LOOP OF LENGTH NROOTS ADDS 33 SECONDS TO
    !     A 240 SECOND INTEGRAL COMPUTATION JOB.  LEAVE IT UNROLLED.
    !
    !  GO TO (100,140,180,220,260,300,340,380,420),NROOTS
    !  cases   1   2   3   4   5   6   7   8   9
    select case(nroots)
    case(1)
       DO I = 1,IJ
          D1 = DIJ(I)
          NX = IJX(I)
          NY = IJY(I)
          NZ = IJZ(I)
          GHONDO(I)=( XIN(NX )*YIN(NY )*ZIN(NZ ) )*D1*DKL(1)+GHONDO(I)
       enddo

    case(2)
       DO I = 1,IJ
          D1 = DIJ(I)
          NX = IJX(I)
          NY = IJY(I)
          NZ = IJZ(I)
          GHONDO(I)=( XIN(NX )*YIN(NY )*ZIN(NZ ) +XIN(NX+625)*YIN(NY+625)* &
               ZIN(NZ+625) )*D1*DKL(1)+GHONDO(I)
       enddo

    case(3)
       DO I = 1,IJ
          D1 = DIJ(I)
          NX = IJX(I)
          NY = IJY(I)
          NZ = IJZ(I)
          GHONDO(I) = ( XIN(NX )*YIN(NY )*ZIN(NZ ) +XIN(NX+625)*YIN(NY+625)* &
               ZIN(NZ+625) +XIN(NX+1250)*YIN(NY+1250)*ZIN(NZ+1250) )*D1* &
               DKL(1)+GHONDO(I)
       enddo

    case(4)
       DO I = 1,IJ
          D1 = DIJ(I)
          NX = IJX(I)
          NY = IJY(I)
          NZ = IJZ(I)
          GHONDO(I) = ( XIN(NX )*YIN(NY )*ZIN(NZ ) +XIN(NX+625)*YIN(NY+625)* &
               ZIN(NZ+625) +XIN(NX+1250)*YIN(NY+1250)*ZIN(NZ+1250) +XIN(NX+ &
               1875)*YIN(NY+1875)*ZIN(NZ+1875) )*D1*DKL(1)+GHONDO(I)
       enddo

    case(5)
       DO I = 1,IJ
          D1 = DIJ(I)
          NX = IJX(I)
          NY = IJY(I)
          NZ = IJZ(I)
          GHONDO(I) = ( XIN(NX )*YIN(NY )*ZIN(NZ ) +XIN(NX+625)*YIN(NY+625)* &
               ZIN(NZ+625) +XIN(NX+1250)*YIN(NY+1250)*ZIN(NZ+1250) +XIN(NX+ &
               1875)*YIN(NY+1875)*ZIN(NZ+1875) +XIN(NX+2500)*YIN(NY+2500)* &
               ZIN(NZ+2500) )*D1*DKL(1)+GHONDO(I)
       enddo

    case(6)
       DO I = 1,IJ
          D1 = DIJ(I)
          NX = IJX(I)
          NY = IJY(I)
          NZ = IJZ(I)
          GHONDO(I) = ( XIN(NX )*YIN(NY )*ZIN(NZ ) +XIN(NX+625)*YIN(NY+625)* &
               ZIN(NZ+625) +XIN(NX+1250)*YIN(NY+1250)*ZIN(NZ+1250) +XIN(NX+ &
               1875)*YIN(NY+1875)*ZIN(NZ+1875) +XIN(NX+2500)*YIN(NY+2500)* &
               ZIN(NZ+2500) +XIN(NX+3125)*YIN(NY+3125)*ZIN(NZ+3125)) &
               *D1*DKL(1)+GHONDO(I)
       enddo

    case(7)
       DO I = 1,IJ
          D1 = DIJ(I)
          NX = IJX(I)
          NY = IJY(I)
          NZ = IJZ(I)
          GHONDO(I) = ( XIN(NX )*YIN(NY )*ZIN(NZ ) +XIN(NX+625)*YIN(NY+625)* &
               ZIN(NZ+625) +XIN(NX+1250)*YIN(NY+1250)*ZIN(NZ+1250) +XIN(NX+ &
               1875)*YIN(NY+1875)*ZIN(NZ+1875) +XIN(NX+2500)*YIN(NY+2500)* &
               ZIN(NZ+2500) +XIN(NX+3125)*YIN(NY+3125)*ZIN(NZ+3125) +XIN(NX+ &
               3750)*YIN(NY+3750)*ZIN(NZ+3750)) &
               *D1*DKL(1)+GHONDO(I)
       enddo

    case(8)
       DO I = 1,IJ
          D1 = DIJ(I)
          NX = IJX(I)
          NY = IJY(I)
          NZ = IJZ(I)
          GHONDO(I) = ( XIN(NX )*YIN(NY )*ZIN(NZ ) +XIN(NX+625)*YIN(NY+625)* &
               ZIN(NZ+625) +XIN(NX+1250)*YIN(NY+1250)*ZIN(NZ+1250) +XIN(NX+ &
               1875)*YIN(NY+1875)*ZIN(NZ+1875) +XIN(NX+2500)*YIN(NY+2500)* &
               ZIN(NZ+2500) +XIN(NX+3125)*YIN(NY+3125)*ZIN(NZ+3125) +XIN(NX+ &
               3750)*YIN(NY+3750)*ZIN(NZ+3750) +XIN(NX+4375)*YIN(NY+4375)* &
               ZIN(NZ+4375))*D1*DKL(1)+GHONDO(I)
       enddo

    case(9)
       DO  I = 1,IJ
          D1 = DIJ(I)
          NX = IJX(I)
          NY = IJY(I)
          NZ = IJZ(I)
          GHONDO(I) = ( XIN(NX )*YIN(NY )*ZIN(NZ ) +XIN(NX+625)*YIN(NY+625)* &
               ZIN(NZ+625) +XIN(NX+1250)*YIN(NY+1250)*ZIN(NZ+1250) +XIN(NX+ &
               1875)*YIN(NY+1875)*ZIN(NZ+1875) +XIN(NX+2500)*YIN(NY+2500)* &
               ZIN(NZ+2500) +XIN(NX+3125)*YIN(NY+3125)*ZIN(NZ+3125) +XIN(NX+ &
               3750)*YIN(NY+3750)*ZIN(NZ+3750) +XIN(NX+4375)*YIN(NY+4375)* &
               ZIN(NZ+4375) +XIN(NX+5000)*YIN(NY+5000)*ZIN(NZ+5000)) &
               *D1*DKL(1)+GHONDO(I)
       end DO
    end select
    RETURN
  END subroutine forms1
  !
  SUBROUTINE IJPRIM1(DDIJ)
    !------------------------------------------------------------------
    LOGICAL IANDJ,KANDL,SAME,OUT,NORM
    !
    integer,parameter :: MXGSH=30, MXG2=MXGSH*MXGSH
    !
    real(chm_real) DDIJ(16*MXG2)
    !
    real(chm_real) a,r,x1,y1,z1
    integer ijd,nprint,itol,icut,normf,normp,nopk
    COMMON /IJGNRL/ A(MXG2),R(MXG2),X1(MXG2),Y1(MXG2),Z1(MXG2), &
         IJD(784)
    COMMON /MISC  / IANDJ,KANDL,SAME
    COMMON /OUTPUT/ NPRINT,ITOL,ICUT,NORMF,NORMP,NOPK
    real(chm_real) ag,csa,cpa,cda,cfa,cga,cha,cia
    real(chm_real) bg,csb,cpb,cdb,cfb,cgb,chb,cib
    real(chm_real) cg,csc,cpc,cdc,cfc,cgc,chc,cic
    real(chm_real) dg,csd,cpd,cdd,cfd,cgd,chd,cid
    real(chm_real)  XI,YI,ZI,XJ,YJ,ZJ,RRI,XK,YK,ZK,XL,YL,ZL,RRK
    integer nga,ngb,ngc,ngd
    COMMON /SHLINF/ AG(MXGSH),CSA(MXGSH),CPA(MXGSH),CDA(MXGSH), &
         CFA(MXGSH),CGA(MXGSH),CHA(mxgsh),CIA(mxgsh), &
         BG(MXGSH),CSB(MXGSH),CPB(MXGSH),CDB(MXGSH), &
         CFB(MXGSH),CGB(MXGSH),CHb(mxgsh),CIb(mxgsh), &
         CG(MXGSH),CSC(MXGSH),CPC(MXGSH),CDC(MXGSH), &
         CFC(MXGSH),CGC(MXGSH),CHc(mxgsh),CIc(mxgsh), &
         DG(MXGSH),CSD(MXGSH),CPD(MXGSH),CDD(MXGSH), &
         CFD(MXGSH),CGD(MXGSH),CHd(mxgsh),CId(mxgsh), &
         XI,YI,ZI,XJ,YJ,ZJ,RRI,XK,YK,ZK,XL,YL,ZL,RRK, &
         NGA,NGB,NGC,NGD
    real(chm_real) qq4,tol,cutoff
    integer LIT,LJT,LKT,LLT,LOCI,LOCJ,LOCK,LOCL,icount
    integer MINI,MINJ,MINK,MINL,MAXI,MAXJ,MAXK,MAXL,NIJ,IJ,KL,IJKL
    COMMON /SHLNOS/ QQ4,LIT,LJT,LKT,LLT,LOCI,LOCJ,LOCK,LOCL, &
         MINI,MINJ,MINK,MINL,MAXI,MAXJ,MAXK,MAXL, &
         NIJ,IJ,KL,IJKL
    COMMON /SHLT  / TOL,CUTOFF,ICOUNT,OUT
    !
    real(chm_real),PARAMETER :: SQRT3=1.73205080756888_chm_real, &
         SQRT5=2.23606797749979_chm_real, &
         SQRT7=2.64575131106459_chm_real, &
         ZERO=0.0_chm_real, ONE=1.0_chm_real
    !
    real(chm_real) aa,aainv,ai,aj,arri,axi,ayi,azi,cdi,cdj,cfi
    real(chm_real) cfj,cgi,cgj,cpi,cpj,csi,csj,dum,dum1,dum2
    integer i,ia,j,jb,jbmax,max,n,nm,nn
    !
    NORM = NORMF .NE. 1 .OR. NORMP .NE. 1
    TOL=75.0_chm_real
    MAX = MAXJ
    N = 0
    NN = 0
    NM = -2**20
    DO I = MINI,MAXI
       !GO TO (100,100,120,120,100,120,120,100,120,120, &
       !        1   2   3   4   5   6   7   8   9   10
       !     100,120,120,100,120,120,120,120,120,100, &
       !     11   2   3   4   5   6   7   8   9   20
       !     100,120,120,100,120,120,120,120,120,100, &
       !     21   2   3   4   5   6   7   8   9   30
       !     120,120,100,120,120),I
       !     31   2   3   4   5
       select case(i)
       case(1:2,5,8,11,14,20:21,24,30,33)
          NM = NN
       end select
       NN = NM
       IF (IANDJ) MAX = I
       DO J = MINJ,MAX
          !GO TO (140,140,160,160,140,160,160,140,160,160, &
          !        1   2   3   4   5   6   7   8   9   10
          !     140,160,160,140,160,160,160,160,160,140, &
          !     11   2   3   4   5   6   7   8   9   20
          !     140,160,160,140,160,160,160,160,160,140, &
          !     21   2   3   4   5   6   7   8   9   30
          !     160,160,140,160,160),J
          !      31  2   3   4   5
          select case(j)
          case(1:2,5,8,11,14,20:21,24,30,33)
             NN = NN+1
          end select
          N = N+1
          IJD(N) = NN
       enddo
    enddo
    !
    !     ----- I PRIMITIVE
    !
    NIJ = 0
    JBMAX = NGB
    loop540:DO IA = 1,NGA
       AI = AG(IA)
       ARRI = AI*RRI
       AXI = AI*XI
       AYI = AI*YI
       AZI = AI*ZI
       CSI = CSA(IA)
       CPI = CPA(IA)
       CDI = CDA(IA)
       CFI = CFA(IA)
       CGI = CGA(IA)
       !
       !        ----- J PRIMITIVE
       !
       IF (IANDJ) JBMAX = IA
       loop520:DO JB = 1,JBMAX
          AJ = BG(JB)
          AA = AI+AJ
          AAINV = ONE/AA
          DUM = AJ*ARRI*AAINV
          IF (DUM .GT. TOL) cycle loop520
          CSJ = CSB(JB)
          CPJ = CPB(JB)
          CDJ = CDB(JB)
          CFJ = CFB(JB)
          CGJ = CGB(JB)
          NM = 16*NIJ
          NN = NM
          NIJ = NIJ+1
          R(NIJ) = DUM
          A(NIJ) = AA
          X1(NIJ) = (AXI+AJ*XJ)*AAINV
          Y1(NIJ) = (AYI+AJ*YJ)*AAINV
          Z1(NIJ) = (AZI+AJ*ZJ)*AAINV
          !
          !           ----- DENSITY FACTOR
          !
          DUM1 = ZERO
          DUM2 = ZERO
          loop420:DO I = MINI,MAXI
             !        1   2   3   4   5   6   7   8   9   10
             !GO TO (200,220,420,420,240,420,420,260,420,420, &
             !       261,420,420,262,420,420,420,420,420,263, &
             !       264,420,420,265,420,420,420,420,420,266, &
             !       420,420,267,420,420),I
             select case(i)
             case(1)
                DUM1 = CSI*AAINV
                IF (IANDJ) MAX = I
             case(2)
                DUM1 = CPI*AAINV
                IF (IANDJ) MAX = I
             case(5)
                DUM1 = CDI*AAINV
                IF (IANDJ) MAX = I
             case(8)
                IF (NORM) DUM1 = DUM1*SQRT3
                IF (IANDJ) MAX = I
             case(11)
                DUM1 = CFI*AAINV
                IF (IANDJ) MAX = I
             case(14)
                IF (NORM) DUM1 = DUM1*SQRT5
                IF (IANDJ) MAX = I
             case(20)
                IF (NORM) DUM1 = DUM1*SQRT3
                IF (IANDJ) MAX = I
             case(21)
                DUM1 = CGI*AAINV
                IF (IANDJ) MAX = I
             case(24)
                IF (NORM) DUM1 = DUM1*SQRT7
                IF (IANDJ) MAX = I
             case(30)
                IF (NORM) DUM1 = DUM1*SQRT5/SQRT3
                IF (IANDJ) MAX = I
             case(33)
                IF (NORM) DUM1 = DUM1*SQRT3
                IF (IANDJ) MAX = I
             case default
                cycle loop420
             end select
             loop400: DO J = MINJ,MAX
                !        1   2   3   4   5   6   7   8   9   10
                !GO TO (300,320,400,400,340,400,400,360,400,400, &
                !       361,400,400,362,400,400,400,400,400,363, &
                !       364,400,400,365,400,400,400,400,400,366, &
                !       400,400,367,400,400),J
                select case(j)
                case(1)
                   DUM2 = DUM1*CSJ
                case(2)
                   DUM2 = DUM1*CPJ
                case(5)
                   DUM2 = DUM1*CDJ
                case(8)
                   IF (NORM) DUM2 = DUM2*SQRT3
                case(11)
                   DUM2 = DUM1*CFJ
                case(14)
                   IF (NORM) DUM2 = DUM2*SQRT5
                case(20)
                   IF (NORM) DUM2 = DUM2*SQRT3
                case(21)
                   DUM2 = DUM1*CGJ
                case(24)
                   IF (NORM) DUM2 = DUM2*SQRT7
                case(30)
                   IF (NORM) DUM2 = DUM2*SQRT5/SQRT3
                case(33)
                   IF (NORM) DUM2 = DUM2*SQRT3
                case default
                   cycle loop400
                end select
                NN = NN+1
                DDIJ(NN) = DUM2
             enddo loop400
          enddo loop420
          IF ( .NOT. IANDJ) cycle loop520
          IF (IA .EQ. JB) cycle loop520

          ! GO TO (500,440,460,455,450),LIT

          select case(lit)
          case(1)
             DDIJ(NM+1) = DDIJ(NM+1)+DDIJ(NM+1)
          case(2)   
             IF (MINI .EQ. 2) then
                DDIJ(NM+1) = DDIJ(NM+1)+DDIJ(NM+1)
             else
                DDIJ(NM+2) = DDIJ(NM+2)+CSI*CPJ*AAINV
                DDIJ(NM+3) = DDIJ(NM+3)+DDIJ(NM+3)
                DDIJ(NM+1) = DDIJ(NM+1)+DDIJ(NM+1)
             endif
          case(3)
             DDIJ(NM+2) = DDIJ(NM+2)+DDIJ(NM+2)
             DDIJ(NM+3) = DDIJ(NM+3)+DDIJ(NM+3)
             DDIJ(NM+1) = DDIJ(NM+1)+DDIJ(NM+1)
          case(4)
             DDIJ(NM+6) = DDIJ(NM+6)+DDIJ(NM+6)
             DDIJ(NM+5) = DDIJ(NM+5)+DDIJ(NM+5)
             DDIJ(NM+4) = DDIJ(NM+4)+DDIJ(NM+4)
             DDIJ(NM+2) = DDIJ(NM+2)+DDIJ(NM+2)
             DDIJ(NM+3) = DDIJ(NM+3)+DDIJ(NM+3)
             DDIJ(NM+1) = DDIJ(NM+1)+DDIJ(NM+1)
          case(5)
             DDIJ(NM+10) = DDIJ(NM+10)+DDIJ(NM+10)
             DDIJ(NM+9) = DDIJ(NM+9)+DDIJ(NM+9)
             DDIJ(NM+8) = DDIJ(NM+8)+DDIJ(NM+8)
             DDIJ(NM+7) = DDIJ(NM+7)+DDIJ(NM+7)
             DDIJ(NM+6) = DDIJ(NM+6)+DDIJ(NM+6)
             DDIJ(NM+5) = DDIJ(NM+5)+DDIJ(NM+5)
             DDIJ(NM+4) = DDIJ(NM+4)+DDIJ(NM+4)
             DDIJ(NM+2) = DDIJ(NM+2)+DDIJ(NM+2)
             DDIJ(NM+3) = DDIJ(NM+3)+DDIJ(NM+3)
             DDIJ(NM+1) = DDIJ(NM+1)+DDIJ(NM+1)
          end select

          !-- 440          IF (MINI .EQ. 2) GO TO 500
          !--           DDIJ(NM+2) = DDIJ(NM+2)+CSI*CPJ*AAINV
          !--           GO TO 480
          !-- 450       DDIJ(NM+10) = DDIJ(NM+10)+DDIJ(NM+10)
          !--           DDIJ(NM+9) = DDIJ(NM+9)+DDIJ(NM+9)
          !--           DDIJ(NM+8) = DDIJ(NM+8)+DDIJ(NM+8)
          !--           DDIJ(NM+7) = DDIJ(NM+7)+DDIJ(NM+7)
          !-- 455       DDIJ(NM+6) = DDIJ(NM+6)+DDIJ(NM+6)
          !--           DDIJ(NM+5) = DDIJ(NM+5)+DDIJ(NM+5)
          !--           DDIJ(NM+4) = DDIJ(NM+4)+DDIJ(NM+4)
          !-- 460       DDIJ(NM+2) = DDIJ(NM+2)+DDIJ(NM+2)
          !-- 480       DDIJ(NM+3) = DDIJ(NM+3)+DDIJ(NM+3)
          !-- 500       DDIJ(NM+1) = DDIJ(NM+1)+DDIJ(NM+1)
       enddo loop520
    enddo loop540
    RETURN
  END subroutine ijprim1
  !
  SUBROUTINE XYZINT1
    !
    LOGICAL N0,N1,M0,M1
    !
    integer i,k,NIMAX,NJMAX,NKMAX,NLMAX,NMAX,MMAX
    real(chm_real) BP01,B00,B10,XCP00,XC00,YCP00,YC00,ZCP00,ZC00,F00
    real(chm_real) DXIJ,DYIJ,DZIJ,DXKL,DYKL,DZKL
    COMMON /SETINT/ I(13),K(13),NIMAX,NJMAX,NKMAX,NLMAX,NMAX,MMAX &
         ,BP01,B00,B10,XCP00,XC00,YCP00,YC00,ZCP00,ZC00,F00 &
         ,DXIJ,DYIJ,DZIJ,DXKL,DYKL,DZKL
    real(chm_real) xint,yint,zint
    COMMON /XYZ   / XINt(31213),YINt(31213),ZINt(31213)
    !
    real(chm_real),PARAMETER :: ZERO=0.0_chm_real, ONE=1.0_chm_real
    !
    real(chm_real) c01,c10,cp01,cp10
    integer i1,i2,i3,i4,i5,ia,ib,k2,k3,k4,km,m,min,n,ni,nj,nk,nl
    !
    N0 = NMAX .EQ. 0
    N1 = NMAX .LE. 1
    M0 = MMAX .EQ. 0
    M1 = MMAX .LE. 1
    !
    !     ----- I(0,0) -----
    !
    I1 = I(1)
    XINT(I1) = ONE
    YINT(I1) = ONE
    ZINT(I1) = F00
    IF (N0 .AND. M0) RETURN
    I2 = I(2)
    K2 = K(2)
    CP10 = B00
    IF (N0) GO TO 100
    !
    !     ----- I(1,0) -----
    !
    XINT(I2) = XC00
    YINT(I2) = YC00
    ZINT(I2) = ZC00*F00
    IF (M0) GO TO 120
    !
    !     ----- I(0,1) -----
    !
100 I3 = I1+K2
    XINT(I3) = XCP00
    YINT(I3) = YCP00
    ZINT(I3) = ZCP00*F00

    IF (.not.N0) then !GO TO 120
       !
       !     ----- I(1,1) -----
       !
       I3 = I2+K2
       XINT(I3) = XCP00*XINT(I2)+CP10
       YINT(I3) = YCP00*YINT(I2)+CP10
       ZINT(I3) = ZCP00*ZINT(I2)+CP10*F00
    endif
120 continue
    IF (.not.N1)then  ! GO TO 180
       C10 = ZERO
       I3 = I1
       I4 = I2
       loop160: DO N = 2,NMAX
          C10 = C10+B10
          !
          !     ----- I(N,0) -----
          !
          I5 = I(N+1)
          XINT(I5) = C10*XINT(I3)+XC00*XINT(I4)
          YINT(I5) = C10*YINT(I3)+YC00*YINT(I4)
          ZINT(I5) = C10*ZINT(I3)+ZC00*ZINT(I4)
          IF (.not.M0) then  !GO TO 140
             CP10 = CP10+B00
             !
             !     ----- I(N,1) -----
             !
             I3 = I5+K2
             XINT(I3) = XCP00*XINT(I5)+CP10*XINT(I4)
             YINT(I3) = YCP00*YINT(I5)+CP10*YINT(I4)
             ZINT(I3) = ZCP00*ZINT(I5)+CP10*ZINT(I4)
          endif
          I3 = I4
          I4 = I5
       enddo loop160
    endif
    IF (.not.M1)then   ! GO TO 240
       CP01 = ZERO
       C01 = B00
       I3 = I1
       I4 = I1+K2
       loop220:DO  M = 2,MMAX
          CP01 = CP01+BP01
          !
          !     ----- I(0,M) -----
          !
          I5 = I1+K(M+1)
          XINT(I5) = CP01*XINT(I3)+XCP00*XINT(I4)
          YINT(I5) = CP01*YINT(I3)+YCP00*YINT(I4)
          ZINT(I5) = CP01*ZINT(I3)+ZCP00*ZINT(I4)
          IF (.not.N0)then    ! GO TO 200
             C01 = C01+B00
             !
             !     ----- I(1,M) -----
             !
             I3 = I2+K(M+1)
             XINT(I3) = XC00*XINT(I5)+C01*XINT(I4)
             YINT(I3) = YC00*YINT(I5)+C01*YINT(I4)
             ZINT(I3) = ZC00*ZINT(I5)+C01*ZINT(I4)
          endif
          I3 = I4
          I4 = I5
       enddo loop220
    endif
    IF (.not.(N1 .OR. M1) ) then   !GO TO 300
       !
       !     ----- I(N,M) -----
       !
       C01 = B00
       K3 = K2
       loop280: DO M = 2,MMAX
          K4 = K(M+1)
          C01 = C01+B00
          I3 = I1
          I4 = I2
          C10 = B10
          loop260: DO N = 2,NMAX
             I5 = I(N+1)
             XINT(I5+K4) = C10*XINT(I3+K4)+XC00*XINT(I4+K4)+C01*XINT(I4+K3)
             YINT(I5+K4) = C10*YINT(I3+K4)+YC00*YINT(I4+K4)+C01*YINT(I4+K3)
             ZINT(I5+K4) = C10*ZINT(I3+K4)+ZC00*ZINT(I4+K4)+C01*ZINT(I4+K3)
             C10 = C10+B10
             I3 = I4
             I4 = I5
          enddo loop260
          K3 = K4
       enddo loop280
    endif
    IF (NJMAX /= 0) then   ! GO TO 440
       !
       !     ----- I(NI,NJ,M) -----
       !
       M = 0
       I5 = I(NMAX+1)
320    MIN = NIMAX
       KM = K(M+1)
340    N = NMAX
       I3 = I5+KM
360    I4 = I(N)+KM
       XINT(I3) = XINT(I3)+DXIJ*XINT(I4)
       YINT(I3) = YINT(I3)+DYIJ*YINT(I4)
       ZINT(I3) = ZINT(I3)+DZIJ*ZINT(I4)
       I3 = I4
       N = N-1
       IF (N .GT. MIN) GO TO 360
       MIN = MIN+1
       IF (MIN .LT. NMAX) GO TO 340
       IF (NIMAX /= 0) then ! GO TO 420
          I3 = 25+KM+I1
          loop400: DO NJ = 1,NJMAX
             I4 = I3
             loop380: DO NI = 1,NIMAX
                XINT(I4) = XINT(I4+100)+DXIJ*XINT(I4-25)
                YINT(I4) = YINT(I4+100)+DYIJ*YINT(I4-25)
                ZINT(I4) = ZINT(I4+100)+DZIJ*ZINT(I4-25)
                I4 = I4+125
             enddo loop380
             I3 = I3+25
          enddo loop400
       endif
       M = M+1
       IF (M .LE. MMAX) GO TO 320
    endif
    IF (NLMAX /= 0) then ! GO TO 600
       !
       !     ----- I(NI,NJ,NK,NL) -----
       !
       I5 = K(MMAX+1)
       IA = I1
       NI = 0
460    NJ = 0
       IB = IA
480    MIN = NKMAX
500    M = MMAX
       I3 = IB+I5
520    I4 = IB+K(M)
       XINT(I3) = XINT(I3)+DXKL*XINT(I4)
       YINT(I3) = YINT(I3)+DYKL*YINT(I4)
       ZINT(I3) = ZINT(I3)+DZKL*ZINT(I4)
       I3 = I4
       M = M-1
       IF (M .GT. MIN) GO TO 520
       MIN = MIN+1
       IF (MIN .LT. MMAX) GO TO 500
       IF (NKMAX /= 0)then   ! GO TO 580
          I3 = IB+1
          loop560: DO NL = 1,NLMAX
             I4 = I3
             loop540: DO NK = 1,NKMAX
                XINT(I4) = XINT(I4+4)+DXKL*XINT(I4-1)
                YINT(I4) = YINT(I4+4)+DYKL*YINT(I4-1)
                ZINT(I4) = ZINT(I4+4)+DZKL*ZINT(I4-1)
                I4 = I4+5
             enddo loop540
             I3 = I3+1
          enddo loop560
       endif
       NJ = NJ+1
       IB = IB+25
       IF (NJ .LE. NJMAX) GO TO 480
       NI = NI+1
       IA = IA+125
       IF (NI .LE. NIMAX) GO TO 460
    endif
    RETURN
  end subroutine xyzint1
  !
#else /**/
  SUBROUTINE BLUR_NULL
    RETURN
  END subroutine blur_null
#endif 

endmodule gmsblur


