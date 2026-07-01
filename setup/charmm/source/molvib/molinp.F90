#if KEY_MOLVIB==1
SUBROUTINE MOLINP(JCONT,NUMAT,NQ,NIC,NIC0,IPRNT,NULL,NSTRT, &
     NAT,NAT3,NQ2,IGLEV,ICANO,IFPED,IFTRRM, &
     IZMOL,IFCRYS,IFSTEP,ISTCOR,STPSIZ, &
     IFFMAT,IFLMAT,IFEXPF,IFSYMM,IFTRAN, &
     NQM,NATM,NAT3M,IZMAX,U1,U2,FS,FX, &
     LX,W2,V1,V2,V3,ATYP,CUTPED,CALFRQ,W1, &
     ISTRM,OUTU,NGMAX,MAXSYM,NBLMAX, &
     LGRUP,IGRUP,KSYMB,NGRUP,NSYMB, &
     SYMB,SPED,SBLOCK,IPTA,IBLOCK,NBLOCK,IPTB,QSEL, &
     MNAT,X,AMASS,EXPFRQ,ICTYP,IATI,IATJ,IATK,IATL)
  !
  !  Input section for program MOLVIB
  !  Input description : molinp.dsc
  !
  !  Note: For GAUS and GFX options remember that the eigenvectors LX
  !        output by GAUSSIAN are in the 'standard orientation', while
  !        the FX matrix is given in the 'Z matrix orientation'.
  !        This may cause some confusion as different cartesian
  !        coordinates have to be used for different applications.
  !
  !  NAT - no. of atoms
  !  NQ  - dimension of the G matrix for diagonalization
  !  NIC0 - initial no. of internal coords ("primitive IC's")
  !  NIC - no. of internal coordinates after presymmetrization
  !        of G by matrix U1
  !  IZMOL - no. of molecules, =1 for molecular vibrations
  !                            = no. of molecules in unit cell for CRYS
  !
  !   Maximum dimensions : all arrays except AMASS and X
  !   should have dimensions smaller than NQM;
  !   AMASS and X : smaller than NAT3M
  !   PRINCI - smaller than IZMAX*3, TOMM - smaller than IZMAX
  !
  !     IZMAX = max. no. of molecules in unit cell
  !     NQ2M=NQM**2
  !     NAT3M=NAT*3
  !     ISTRM - input file no.
  !     OUTU  - output file no.
  !-------------------------------------------------------------------
  !    K.Kuczera & J.Wiorkiewicz-Kuczera   27-Dec-1988
  !                              update    03-Apr-1991
  !-------------------------------------------------------------------
  !
  ! ... data from molvib.f90
  use chm_kinds
  use machio,only:vopen
  implicit none
  INTEGER NQM,NATM,NAT3M,IZMAX
  real(chm_real) U1(NQM,NQM),U2(NQM,NQM),FS(NQM,NQM),FX(NQM,NQM)
  real(chm_real) LX(NQM,NQM),W2(NQM,NQM),V1(NQM),V2(NQM),V3(NQM)
  real(chm_real) X(NAT3M),AMASS(NAT3M),EXPFRQ(NQM)
  real(chm_real) CALFRQ(NQM),W1(NQM,NQM)
  INTEGER MNAT(IZMAX)
  INTEGER ICTYP(NQM),IATI(NQM),IATJ(NQM),IATK(NQM),IATL(NQM)
  CHARACTER(len=*) ATYP(NAT3M)
  ! ... data from ped.f90
  INTEGER NGMAX,MAXSYM,NBLMAX
  INTEGER LGRUP(NGMAX),IGRUP(NGMAX,MAXSYM),KSYMB(MAXSYM)
  INTEGER NGRUP,NSYMB
  CHARACTER(len=*) SYMB(MAXSYM)
  CHARACTER(len=*) SPED(NBLMAX)
  CHARACTER(len=*) SBLOCK(NBLMAX)
  INTEGER IPTA(NBLMAX),IBLOCK(NBLMAX),NBLOCK
  INTEGER IPTB(NBLMAX)
  LOGICAL QSEL(NBLMAX)
  real(chm_real) CUTPED
  !
  CHARACTER(len=80) TITLE
  CHARACTER(len=4) WORD,JCONT
  CHARACTER(len=16) :: TYP(16)=(/ &
       'BOND STRETCH    ','ANGLE BEND      ', &
       'OUT OF PLANE WAG','TORSION         ', &
       'LINEAR BEND IP  ','LINEAR BEND OOPL', &
       '                ','                ', &
       '                ','                ', &
       'X TRANSLATION   ','Y TRANSLATION   ', &
       'Z TRANSLATION   ','X ROTATION      ', &
       'Y ROTATION      ','Z ROTATION      '/)
  INTEGER ISS(4)
  CHARACTER(len=8) SSS(4)
  !
  INTEGER ISTRM,OUTU,IFPED
  INTEGER I,J,K,L,M,N,KK,LL,KL,KKI,KKJ,KKK,KKL,LK,ML,I1,JJ
  INTEGER IT1,IT2,IT3,IT4,IU,IZU,IFU,INU,IUU,N1,N2,IZU1,NUMAT
  INTEGER IZMOL,IPRNT,IFTRRM,IFSTEP,IFFMAT,IFLMAT,IFEXPF,IFSYMM
  INTEGER IGLEV,IFCRYS,ICANO,ISTCOR,IFTRAN
  INTEGER NQ,NIC,NIC0,NQ2,IZIC,NIC00,NAT0,IZIC1,NAT,IFC, NAT3
  INTEGER IFF,ISF,IUF,IFL,IUL,NFRQ
  INTEGER NULL,NSTRT
  real(chm_real)  STPSIZ,ST,CUTPET,FCTR
  LOGICAL ERR
  !
  ! ... setup defaults
  IZMOL=1
  IPRNT=2
  CUTPED=0.03
  IFTRRM=0
  IFSTEP=0
  IFFMAT=0
  IFLMAT=0
  IFEXPF=0
  IFSYMM=0
  NBLOCK=1
  SBLOCK(1)='    '
  IFTRAN=-1
  !
  ! ... Check dimension consistency in the .f90 files
  IF(NATM.GT.NQM .OR. NAT3M.GT.NQM) THEN
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' **** Error: in molvib dimensions'
     WRITE(OUTU,*) ' **** make all parameters .LE. NQM '
     WRITE(OUTU,*)
     CALL WRNDIE(-4,'<MOLINP>','CHECK DIMENSIONS')
  END IF

  !  Print header
  !
  WRITE(OUTU,*)
  WRITE(OUTU,*)
  WRITE(OUTU,*)
  WRITE(OUTU,*) '     MOLVIB : PROGRAM FOR MOLECULAR ', &
       'VIBRATIONAL SPECTROSCOPY'
  WRITE(OUTU,*)
  WRITE(OUTU,*)
  WRITE(OUTU,*)
  !
  !   Open input file
  !
  NUMAT=0
  !     IU=99
  IU=ISTRM
  !     OPEN (UNIT=IU,NAME='mol.inp',FORM='FORMATTED',
  !    1      STATUS ='OLD',READONLY)
  !
  !  Read title
  !
10 READ(IU,900) TITLE
  IF(TITLE(1:1).EQ.'*') THEN
     WRITE(OUTU,901) TITLE
     GOTO 10
  END IF
  BACKSPACE IU
900 FORMAT(A80)
901 FORMAT(1X,A80)
  !
  !  Start input loop : read through control cards and data
  !

20 READ(IU,910) WORD,IT1,IT2,IT3,IT4
910 FORMAT(A4,1X,4I5)
  IF(IPRNT.GE.2) WRITE(OUTU,911) WORD,IT1,IT2,IT3,IT4
911 FORMAT(1X,A4,1X,4I5)

  !
  ! ... Read main control variable
  !

  IF(WORD.EQ.'G   ') THEN
     JCONT=WORD
     IGLEV=IT1
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     WRITE(OUTU,*) '       SELECTED OPTION : REDUNDANCY ', &
          'ANALYSIS, LEVEL = ',IGLEV
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     GOTO 20
  END IF
  IF(WORD.EQ.'GF  ') THEN
     JCONT=WORD
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     WRITE(OUTU,*) '  SELECTED OPTION : WILSON GF METHOD IN ', &
          'INTERNAL COORDINATES '
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     GOTO 20
  END IF
  IF(WORD.EQ.'GAUS') THEN
     JCONT=WORD
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     WRITE(OUTU,*) '  SELECTED OPTION : LX -> LS TRANSFORM', &
          'ATION FROM GAUSSIAN INPUT '
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     GOTO 20
  END IF
  IF(WORD.EQ.'GFX ') THEN
     JCONT=WORD
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     WRITE(OUTU,*) '  SELECTED OPTION : VIBRATIONAL EIGENVALUE ', &
          'PROBLEM IN CARTESIAN COORDINATES'
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     GOTO 20
  END IF
  IF(WORD.EQ.'CRYS') THEN
     JCONT=WORD
     IZMOL=IT1
     IFCRYS=IT2
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     WRITE(OUTU,*) '  SELECTED OPTION: CRYSTAL VIBRATIONS ', &
          'FROM CARTESIAN FORCE CONSTANTS FX '
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     IF(IZMOL.GT.IZMAX) THEN
        WRITE(OUTU,*) ' **** Error : IZMOL exceeds limit'
        CALL WRNDIE(-4,'<MOLINP>','CHECK CRYS CARD')
     END IF
     GOTO 20
  END IF
  !
  ! ... 'KANO' keyword
  !
  IF(WORD.EQ.'KANO') THEN
     ICANO=IT1
     IF(ICANO.GT.1) JCONT=WORD
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' CANONICAL FORCE FIELD EVALUATION WITH', &
          ' ICANO =',ICANO
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     GOTO 20
  END IF
  !
  ! ... 'STEP' keyword
  !
  IF(WORD.EQ.'STEP') THEN
     IFSTEP=IT1
     ISTCOR=IT2
     IFFMAT=IT3
     IFLMAT=IT4
     JCONT=WORD
     IF(IFSTEP.LE.0 .OR. IFSTEP.GT.3) THEN
        WRITE(OUTU,*) ' **** Error: IFSTEP=',IFSTEP,' is illegal'
        WRITE(OUTU,*)
        CALL WRNDIE(-4,'<MOLINP>','CHECK INPUT')
     END IF
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' GENERATION OF DISPLACED CARTESIAN', &
          ' COORDINATES'
     WRITE(OUTU,*)
     IF(IFSTEP.EQ.1) THEN
        WRITE(OUTU,*) '   STEP along cartesian eigenvector #', &
             ISTCOR
     END IF
     IF(IFSTEP.EQ.2) THEN
        WRITE(OUTU,*) '   STEP along internal eigenvector #', &
             ISTCOR
     END IF
     IF(IFSTEP.EQ.3) THEN
        WRITE(OUTU,*) '   STEP along internal coordinate #', &
             ISTCOR
     END IF
     WRITE(OUTU,*)
     READ(IU,*) STPSIZ
     WRITE(OUTU,909) STPSIZ
     WRITE(OUTU,*)
     GOTO 20
  END IF
909 FORMAT(1X,'   Step size = ',F10.6)
  !
  ! ... Read dimension variables
  !
  IF(WORD.EQ.'DIM ') THEN
     !       NQ=IT1
     !       NIC=IT2
     !       NIC0=IT3
     IBLOCK(1)=NQ
     WRITE(OUTU,*)
     WRITE(OUTU,'(A,3I8)') ' The dimensions are : NQ,NIC,NIC0 = ', &
          NQ,NIC,NIC0
     NQ2=NQ*NQ
     WRITE(OUTU,*) ' NQ2 = ',NQ2
     WRITE(OUTU,*)
     IF(NQ.GT.NQM .OR. NIC.GT.NQM .OR. NIC0.GT.NQM) THEN
        WRITE(OUTU,*)
        WRITE(OUTU,*) ' **** Error : dimensions exceed limit '
        WRITE(OUTU,*)
        CALL WRNDIE(-4,'<MOLINP>','CHECK DIMENSIONS')
     END IF
     GOTO 20
  END IF
  !
  ! ... Read in 'MNAT' data: nos. of atoms of the component molecules
  ! ... in a mixed crystal (or molecular complex)
  !
  IF(WORD.EQ.'MNAT') THEN
     READ(IU,932) (MNAT(I),I=1,IZMOL)
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' The system contains',IZMOL, &
          ' molecules; with following # of atoms:'
     WRITE(OUTU,933)  (MNAT(I),I=1,IZMOL)
     WRITE(OUTU,*)
     GOTO 20
  END IF
  !
  ! ... Read in 'TEST' card : raise print volume to maximum
  !
  IF(WORD.EQ.'TEST') THEN
     IPRNT=5
     GOTO 20
  END IF
  !
  ! ... Read in 'PRNT' card : set print volume
  !
  IF(WORD.EQ.'PRNT') THEN
     IPRNT=IT1
     GOTO 20
  END IF
  !
  ! ... Read atomic cartesian coordinates and masses
  ! ... IFC=0 - free format; IFC=1 - CHARMM format
  !
  IF(WORD.EQ.'CART') THEN
     !       NAT=IT1
     IFC=IT2
     !       NAT3=3*NAT
     IF(NAT.GT.NATM .OR. NAT3.GT.NAT3M) THEN
        WRITE(OUTU,*)
        WRITE(OUTU,*) ' **** Error: number of atoms exceeds limit'
        WRITE(OUTU,*)
        CALL WRNDIE(-4,'<MOLINP>','CHECK DIMENSIONS')
     END IF
     IF(IFC.EQ.0) CALL RDCART(X,AMASS,NAT,NAT3,IU,OUTU)
     IF(IFC.EQ.1) CALL RDCARC(X,AMASS,NAT,NAT3,IU,OUTU)
     IF(IFC.LT.0 .OR. IFC.GT.1) THEN
        WRITE(OUTU,*)
        WRITE(OUTU,*) ' Warning : wrong coordinate read format', &
             ' ,IFC= ',IFC
        WRITE(OUTU,*)
     END IF
     IF(IPRNT.GE.5) THEN
        WRITE(OUTU,*)
        WRITE(OUTU,*) '     Testing cartesian coordinate input'
        WRITE(OUTU,*)
        WRITE(OUTU,*) '   Atom #       X         Y         Z', &
             '       M/Z/A '
        DO I=1,NAT
           WRITE(OUTU,921) I,X(I),X(I+NAT),X(I+2*NAT),AMASS(I)
        END DO
     END IF
     GOTO 20
  END IF
921 FORMAT(3X,I3,3X,4F10.5)
  !
  ! ... Assign masses
  !
  IF(WORD.EQ.'MASA') THEN
     NAT3=3*NAT
     CALL MAMUA(NAT,NAT3,ST,ATYP,X,AMASS,OUTU)
     IF(ST .EQ. -1.) GOTO 9999
     GOTO 20
  END IF
  !
  IF(WORD.EQ.'MASZ') THEN
     NAT3=3*NAT
     CALL MAMUZ(NAT,NAT3,ST,ATYP,X,AMASS,OUTU)
     IF(ST .EQ. -1.) GOTO 9999
     GOTO 20
  END IF
  !
  ! ... Read IC definitions
  !
  IF(WORD.EQ.'IC  ') THEN
     IZIC=IT1
     IF(IPRNT.GE.1) THEN
        IF(IZIC.GT.1) THEN
           WRITE(OUTU,*)
           WRITE(OUTU,*) ' Internal/external coordinates for Z='
           WRITE(OUTU,*) IZIC,' molecules will be autogenerated ', &
                ' from first asymmetric unit'
           WRITE(OUTU,*)
        END IF
        WRITE(OUTU,*)
        WRITE(OUTU,*) ' The internal/external coordinates'
        WRITE(OUTU,*)
        WRITE(OUTU,*) ' No.        Type                   I     J', &
             '     K     L'
        WRITE(OUTU,*)
     END IF
     NIC00=NIC0
     IF(IZIC.GT.1) THEN
        NIC00=NIC0/IZIC
        IF(IZIC*NIC00.NE.NIC0) THEN
           WRITE(OUTU,*) ' **** Error in IZIC '
           CALL WRNDIE(-4,'<MOLINP>','CHECK DIMENSIONS')
        END IF
     END IF
     !
     DO  I=1,NIC00
        READ(IU,*) ICTYP(I),IATI(I),IATJ(I),IATK(I),IATL(I)
     END DO
     !
     ! ... Autogeneration of unit cell IC's from asymmetric unit
     !
     IF(IZIC.GT.1) THEN
        NAT0=NAT/IZIC
        IZIC1=IZIC-1
        LL=0
        KK=0
        DO L=1,IZIC1
           LL=LL+NIC00
           KK=KK+NAT0
           DO  K=1,NIC00
              KL=K+LL
              ICTYP(KL)=ICTYP(K)
              IF(ICTYP(K).LE.10) THEN
                 KKI=KK
                 IF(IATI(K).EQ.0) KKI=0
                 IATI(KL) =IATI(K)+KKI
                 KKJ=KK
                 IF(IATJ(K).EQ.0) KKJ=0
                 IATJ(KL) =IATJ(K)+KKJ
                 KKK=KK
                 IF(IATK(K).EQ.0) KKK=0
                 IATK(KL) =IATK(K)+KKK
                 KKL=KK
                 IF(IATL(K).EQ.0) KKL=0
                 IATL(KL) =IATL(K)+KKL
              END IF
              IF(ICTYP(K).GE.11 .AND. ICTYP(K).LE.16) THEN
                 IATI(KL)=IATI(K)+L
                 IATJ(KL)=IATJ(K)
                 IATK(KL)=IATK(K)
                 IATL(KL)=IATL(K)
              END IF
           END DO
        END DO
     END IF
     !            !   (IZIC.GT.1)
     !
     ! ... Printout
     !
     IF(IPRNT.GE.1) THEN
        DO  I=1,NIC0
           JJ=ICTYP(I)
           WRITE(OUTU,923) &
                I,JJ,TYP(JJ),IATI(I),IATJ(I),IATK(I),IATL(I)
        END DO
     END IF
     !
     GOTO 20
  END IF
923 FORMAT(1X,I3,3X,I3,2X,A16,2X,4I6)
  !
  ! ... Read in U matrix and (optionally) normalize rows
  ! ... Only SS format supported at present (IFU=0)
  !
  IF(WORD.EQ.'UMAT') THEN
     NUMAT=NUMAT+1
     IZU=IT4
     IFU=0
     INU=IT2
     IUU=IU
     IF(IT3.NE.IU .AND. IT3.NE.0) THEN
        IUU=IT3
        IF(NUMAT.EQ.1) THEN
           !C
           WRITE(OUTU,*) ' Reading U1 matrix from "u1.dat" ON unit = ',IUU
           CALL VOPEN(IUU,'u1.dat','FORMATTED','READ',ERR,0)
           !        OPEN(UNIT=IUU,FORM='FORMATTED',
           !     1         STATUS='OLD',READONLY)
           !C
        END IF
        IF(NUMAT.EQ.2) THEN
           !C
           WRITE(OUTU,*) ' Reading U2 matrix from "u2.dat" on unit = ',IUU
           CALL VOPEN(IUU,'u2.dat','FORMATTED','READ',ERR,0)
           !        OPEN(UNIT=IUU,FORM='FORMATTED',
           !     1         STATUS='OLD',READONLY)
           !C
        END IF
     END IF

     ! ...  Specify dimensions for U1 and U2 cases

     IF(NUMAT.EQ.1) THEN
        N1=NIC
        N2=NIC0
        IF(IZU.GT.1) THEN
           N1=N1/IZU
           N2=N2/IZU
           WRITE(OUTU,*) ' U matrix for unit cell will be', &
                ' generated from asymmetric unit'
           IF(IZU*N1.NE.NIC) THEN
              WRITE(OUTU,*) ' **** Error in IZU '
              CALL WRNDIE(-4,'<MOLINP>','CHECK DIM and UMAT CARDS')
           END IF
        END IF
        CALL RDMAT(U1,NIC,NIC0,NQM,IUU,IPRNT)
        !
        ! ... Autogenerate U for unit cell from assymetric unit
        !
        IF(IZU.GT.1) THEN
           IZU1=IZU-1
           KK=0
           LL=0
           DO K=1,IZU1
              KK=KK+N1
              LL=LL+N2
              DO L=1,N1
                 DO M=1,N2
                    LK=L+KK
                    ML=M+LL
                    U1(LK,ML)=U1(L,M)
                 END DO
              END DO
           END DO
        END IF
        !              ! (IZU.GT.1)
        !
        N1=NIC
        N2=NIC0
        !
        WRITE(OUTU,*) ' U matrix has been read : INU = ',INU
        IF(IPRNT.GE.3) THEN
           WRITE(OUTU,*) 'U1 matrix'
           DO  I=1,N1
              WRITE(OUTU,1000) I,(U1(I,J),J=1,N2)
           END DO
        END IF
        IF(INU.NE.0) THEN
           CALL NORMR(U1,N1,N2,NQM)
           IF(IPRNT.GE.2) THEN
              WRITE(OUTU,*) 'Normalized U1 matrix'
              DO  I=1,N1
                 WRITE(OUTU,1000) I,(U1(I,J),J=1,N2)
              END DO
           END IF
        END IF
     END IF
     !            !  (NUMAT.EQ.1)
     !
     IF(NUMAT.EQ.2) THEN
        N1=NQ
        N2=NIC
        IF(IZU.GT.1) THEN
           N1=N1/IZU
           N2=N2/IZU
           WRITE(OUTU,*) ' U matrix for unit cell will be ', &
                ' generated from asymmetric unit'
           IF(IZU*N1.NE.NQ) THEN
              WRITE(OUTU,*) ' **** Error in IZU '
              CALL WRNDIE(-4,'<MOLINP>','CHECK DIM and UMAT CARDS')
           END IF
        END IF
        CALL RDMAT(U2,NQ,NIC,NQM,IUU,IPRNT)
        !
        ! ... Autogenerate U for unit cell from assymetric unit
        !
        IF(IZU.GT.1) THEN
           IZU1=IZU-1
           KK=0
           LL=0
           DO K=1,IZU1
              KK=KK+N1
              LL=LL+N2
              DO L=1,N1
                 DO M=1,N2
                    LK=L+KK
                    ML=M+LL
                    U2(LK,ML)=U2(L,M)
                 END DO
              END DO
           END DO
        END IF
        !              ! (IZU.GT.1)
        !
        N1=NQ
        N2=NIC
        !
        WRITE(OUTU,*) ' U matrix has been read : INU = ',INU
        IF(IPRNT.GE.3) THEN
           WRITE(OUTU,*) 'U2 matrix'
           DO  I=1,N1
              WRITE(OUTU,1000) I,(U2(I,J),J=1,N2)
           END DO
        END IF
        IF(INU.NE.0) THEN
           CALL NORMR(U2,N1,N2,NQM)
           IF(IPRNT.GE.2) THEN
              WRITE(OUTU,*) 'Normalized U2 matrix'
              DO  I=1,N1
                 WRITE(OUTU,1000) I,(U2(I,J),J=1,N2)
              END DO
           END IF
        END IF
     END IF
     !              ! (NUMAT.EQ.2)
     !
     GOTO 20
  END IF
1000 FORMAT(1X,I3,8F9.5,12(/4X,8F9.5))

  !
  ! ... Read F matrix
  ! ... SS format for IFF=0, GAUSSIAN format for IFF=1
  ! ... CHARMM format for IFF=2
  !
  IF(WORD.EQ.'FMAT') THEN
     IFF=IT1
     ISF=IT2
     IUF=IU
     !
     IF(IT3.NE.IU .AND. IT3.NE.0) THEN
        IUF=IT3
        !C
        WRITE(OUTU,*) ' Reading F matrix from "f.dat" on unit = ',IUF
        CALL VOPEN(IUF,'f.dat','FORMATTED','READ',ERR,0)
        !        OPEN(UNIT=IUF,FORM='FORMATTED',
        !     1         STATUS='OLD',READONLY)
        !C
     END IF
     !
     ! ... for IFF=0 read in FS matrix in IC's
     ! ...   FS dimensions are set by the IFTRAN variable:
     ! ...   IFTRAN=0 - FS is in primitive IC's R  (NIC0xNIC0)
     ! ...         =1 - FS is in S1=U1*R coords    (NICxNIC)
     ! ...         =2 - FS is in S2=U2*U1*R coords (NQxNQ)
     ! ... for IFF=1 read in NAT3xNAT3 cartesian fc matrix FX
     !
     IF(IFF.EQ.0) THEN
        !
        IF(IFTRAN.LT.0) IFTRAN=NUMAT
        !
        WRITE(OUTU,*)
        IF(IFTRAN.EQ.2) THEN
           WRITE(OUTU,*) ' F matrix in independent ICs S2=U2*U1*R'
           N1=NQ
        END IF
        IF(IFTRAN.EQ.1) THEN
           WRITE(OUTU,*) ' F matrix in independent ICs S1=U1*R'
           N1=NIC
        END IF
        IF(IFTRAN.EQ.0) THEN
           WRITE(OUTU,*) ' F matrix in primitive ICs R'
           N1=NIC0
        END IF
        WRITE(OUTU,*)
        !
        CALL RDMAT(FS,N1,N1,NQM,IUF,IPRNT)
        !
        IF(IPRNT.GE.2) THEN
           DO I=1,N1
              WRITE(OUTU,1000) I,(FS(I,J),J=1,N1)
           END DO
        END IF
     END IF
     !
     IF(IFF.EQ.1) THEN
        IF(IPRNT.GE.2) THEN
           WRITE(OUTU,*)
           WRITE(OUTU,*) ' F matrix in cartesian coordinates'
           WRITE(OUTU,*)
        END IF
        CALL RFORC(FX,W1,NAT,NAT3,NQM,IUF,IPRNT,OUTU)
        IF(IPRNT.GE.2) THEN
           DO I=1,NAT3
              WRITE(OUTU,1000) I,(FX(I,J),J=1,NAT3)
           END DO
        END IF
     END IF
     IF(IFF.EQ.2) THEN
        IF(IPRNT.GE.2) THEN
           WRITE(OUTU,*)
           WRITE(OUTU,*) ' F matrix in cartesian coordinates'
           WRITE(OUTU,*)
        END IF
        CALL RDSECO(FX,W1,V1,V2,V3,NAT,NAT3,NQM,IUF,IPRNT,OUTU)
        IF(IPRNT.GE.2) THEN
           DO I=1,NAT3
              WRITE(OUTU,1000) I,(FX(I,J),J=1,NAT3)
           END DO
        END IF
     END IF

     ! ... Symmetrize FS matrix for specified upper triangle

     IF(ISF.GT.0)  THEN
        DO  I=1,NQ
           I1=I+1
           DO  J=I1,NQ
              FS(J,I)=FS(I,J)
           END DO
        END DO
        IF(IPRNT.GT.2) THEN
           WRITE(OUTU,*)
           WRITE(OUTU,*) ' The symmetrized FS matrix'
           WRITE(OUTU,*)
           DO I=1,NQ
              WRITE(OUTU,1000) I,(FS(I,J),J=1,NQ)
           END DO
        END IF
     END IF

     GOTO 20
  END IF
  !
  ! ... Read in cartesian eigenvectors
  ! ... IFL=0 - GAUSSIAN format; IFL=1 - CHARMM binary format
  ! ... Note: this matrix should contain only the vibrational
  ! ... eigenvalues, i.e. dimensions are (NAT3,NQ)
  !
  IF(WORD.EQ.'LX  ') THEN

     IFL=IT1
     IUL=IU
     !
     IF(IFL.EQ.0) THEN
        IF(IT3.NE.IU .AND. IT3.NE.0) THEN
           IUL=IT3
           !C
           WRITE(OUTU,*)' Reading GAUSSIAN LX matrix from "lx.dat" unit =' &
                ,IUL
           CALL VOPEN(IUL,'lx.dat','FORMATTED','READ',ERR,0)
           !        OPEN(UNIT=IUL,FORM='FORMATTED',
           !     1         STATUS='OLD',READONLY)
           !C
        END IF
        !
        CALL REIGEN(NAT,NAT3,NQ,NQM,W2,LX,NFRQ,CALFRQ,IUL, &
             IPRNT,OUTU)
        CALL LXNCHK(LX,NQ,NAT3,NQM,NQ,AMASS,V1,V2,IPRNT,OUTU)
        !
        IF(NFRQ.NE.NQ) THEN
           WRITE(OUTU,*) ' Warning : REIGEN read ',NFRQ,' frequencies'
        END IF
        GOTO 20
     END IF
     !
     IF(IFL.EQ.1) THEN
        IF(IT3.NE.IU .AND. IT3.NE.0) THEN
           IUL=IT3
           !C
           WRITE(OUTU,*) ' Reading CHARMM LX matrix from "lx.dat" unit =' &
                ,IUL
           CALL VOPEN(IUL,'lx.dat','FORMATTED','READ',ERR,0)
           !        OPEN(UNIT=IUL,FORM='UNFORMATTED',
           !     1         STATUS='OLD',READONLY)
           !C
        END IF
        !
        CALL REIGCH(NQ,NAT,NAT3,IPRNT,IUL, &
             NQM,W1,LX,V1,V2,CALFRQ,OUTU)
        CALL LXNCHK(LX,NQ,NAT3,NQM,NQ,AMASS,V1,V2,IPRNT,OUTU)
        !
        GOTO 20
     END IF
     !
     IF(IFL.LE.0 .OR. IFL.GT.1) THEN
        WRITE(OUTU,*)
        WRITE(OUTU,*) ' Warning : wrong eigenvector format, IFL=', &
             IFL
        WRITE(OUTU,*)
     END IF
  END IF

  !
  ! ... Read in dimension of null basis for redundancy analysis
  !

  IF(WORD.EQ.'NULL') THEN
     NULL=IT1
     NSTRT=IT2
     WRITE(OUTU,*) ' U2 matrix will contain NULL = ',NULL, &
          ' space basis vectors and NSTRT = ',NSTRT,' test vectors'
     GOTO 20
  END IF
  !
  ! ... PED analysis section
  ! ... NGRUP - no.of coordinate groups
  ! ... LGRUP(I),I=1,NGRUP - no. of coords in each group
  ! ... IGRUP(I,J), I=1,NGRUP;J=1,LGRUP(I) - the coor. numbers of the
  ! ...             coordinates belongng to group I
  ! ... NSYMB - no. of symbols definng coordinate types
  ! ... SYMB(I),I=1,NSYMB -  table of the symbols
  ! ... KSYMB(K),K=1,NQ - pointer into the SYMB table for coordinate
  ! ...                   number K
  !
  IF(WORD.EQ.'PED ') THEN
     DO J=1,NQ
        KSYMB(J)=0
     END DO
     NGRUP=IT1
     !
     CUTPET=IT2
     CUTPET=CUTPET/100
     IF(CUTPET.LE.0.0 .OR. CUTPET.GT.1.0) THEN
        WRITE(OUTU,*) ' **** Warning : PED cutoff is incorrect'
        WRITE(OUTU,*) '      Using default value '
     ELSE
        CUTPED=CUTPET
     END IF
     WRITE(OUTU,*)
     WRITE(OUTU,1090) ' The PED cutoff will be CUTPED =',CUTPED
1090 FORMAT(A,F8.3)
     !
     IF(IPRNT.GE.1) THEN
        WRITE(OUTU,*)
        WRITE(OUTU,*) ' Reading PED analysis information'
        WRITE(OUTU,*)
        WRITE(OUTU,*) ' The following coordinate groups are defined '
        WRITE(OUTU,*) '      Ngrup = ',NGRUP
     END IF
     IF(NGRUP.GT.NGMAX) THEN
        WRITE(OUTU,*) ' **** Error : Too many groups NGRUP=',NGRUP
        CALL WRNDIE(-4,'<MOLINP>','CHECK PED CARD')
     END IF
     IF(NGRUP.GT.0) THEN
        DO I=1,NGRUP
           READ(IU,932) N,(IGRUP(I,J),J=1,N)
           IF(N.GT.MAXSYM) THEN
              WRITE(OUTU,*) ' **** Error : Too many coordinates in group'
              WRITE(OUTU,*) ' **** I,N =',I,N
              CALL WRNDIE(-4,'<MOLINP>','CHECK PED DATA')
           END IF
           LGRUP(I)=N
           IF(IPRNT.GE.1) WRITE(OUTU,933) (IGRUP(I,J),J=1,N)
        END DO
     END IF
     !
932  FORMAT(20I3)
933  FORMAT(1X,20I3)
     !
     IF(IPRNT.GE.1) THEN
        WRITE(OUTU,*) ' The following are coordinate symbols'
     END IF
     !
     ! ... Read in individual symbols and set up SYMB and KSYMB arrays
     !
     NSYMB=0
2100 READ(IU,934) (ISS(I),SSS(I),I=1,4)
     IF(IPRNT.GE.1) WRITE(OUTU,935) (ISS(I),SSS(I),I=1,4)
     DO I=1,4
        IF(ISS(I).LT.0) GOTO 2110
        IF(ISS(I).GT.0) THEN
           NSYMB=NSYMB+1
           IF(ISS(I).GT.NQ) THEN
              WRITE(OUTU,*)
              WRITE(OUTU,*) ' **** Error: ISS =',ISS(I), &
                   ' exceeds limit'
              WRITE(OUTU,*)
              CALL WRNDIE(-4,'<MOLINP>','CHECK INPUT')
           END IF
           KSYMB(ISS(I))=NSYMB
           SYMB(NSYMB)=SSS(I)
        END IF
     END DO
     !
     GOTO 2100
     !
2110 CONTINUE
     !
934  FORMAT(4(I3,2X,A8,2X))
935  FORMAT(1X,4(I3,2X,A8,2X))
     IF(MAXSYM.LT.NQ) THEN
        WRITE(OUTU,*)
        WRITE(OUTU,*) ' **** Error: MAXSYM is too small '
        WRITE(OUTU,*)
        CALL WRNDIE(-4,'<MOLINP>','CHECK DIMENSIONS')
     END IF
     IF(NSYMB.GT.MAXSYM) THEN
        WRITE(OUTU,*)
        WRITE(OUTU,*) ' **** Error: too many symbols '
        WRITE(OUTU,*)
        CALL WRNDIE(-4,'<MOLINP>','CHECK DIMENSIONS')
     END IF
     !
     ! ... For each group set table pointers to those of first coordinate
     ! ... in the group
     !
     IF(NGRUP.GT.0) THEN
        DO I=1,NQ
           DO J=1,NGRUP
              IF(I.EQ.IGRUP(J,1)) THEN
                 N=LGRUP(J)
                 DO K=2,N
                    KK=IGRUP(J,K)
                    KSYMB(KK)=KSYMB(I)
                 END DO
              END IF
           END DO
        END DO
     END IF
     !
     ! ... Check that all coords have symbols defined
     !
     IFPED=1
     DO I=1,NQ
        IF(KSYMB(I).LE.0) THEN
           WRITE(OUTU,*) ' Warning : undefined symbol for IC= ',I
           IFPED=0
        END IF
     END DO
     IF(IPRNT.GE.1) THEN
        WRITE(OUTU,*)
        WRITE(OUTU,*) ' Final symbol assignemts to ICs'
        WRITE(OUTU,*)
        DO I=1,NQ
           WRITE(OUTU,936) I,SYMB(KSYMB(I))
        END DO
     END IF
936  FORMAT(12X,I3,2X,A8)
     !
     GOTO 20
  END IF
  !
  ! ... Scaling of FX matrix
  !
  IF(WORD.EQ.'SCAL') THEN
     READ(IU,965) FCTR
965  FORMAT(F10.6)
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' The FX matrix will be scaled by FCTR =' &
          ,FCTR
     WRITE(OUTU,*)
     DO I=1,NAT3
        DO J=1,NAT3
           FX(I,J)=FX(I,J)*FCTR
        END DO
     END DO
     IF(IPRNT.GE.3) THEN
        WRITE(OUTU,*)
        WRITE(OUTU,*) ' The scaled FX matrix '
        WRITE(OUTU,*)
        DO I=1,NAT3
           WRITE(OUTU,1000) I,(FX(I,J),J=1,NAT3)
        END DO
     END IF

     GOTO 20
  END IF
  !
  ! ... Read in 'TRRM' card
  !
  IF(WORD.EQ.'TRRM') THEN
     IFTRRM=1
     GOTO 20
  END IF
  !
  ! ... Read in 'IFTR' card
  !
  IF(WORD.EQ.'IFTR') THEN
     IFTRAN=IT1
     GOTO 20
  END IF
  !
  ! ... Read in 'EXPF' card
  !

  IF(WORD.EQ.'EXPF') THEN
     IFEXPF=1
     WRITE(OUTU,*)
     WRITE(OUTU,*) NQ,' reference frequencies will be read in'
     WRITE(OUTU,*)
     DO I=1,NQ
        READ(IU,*) EXPFRQ(I)
     END DO
     IF(IPRNT.GE.4) THEN
        WRITE(OUTU,*)
        WRITE(OUTU,*) ' The reference frequncies in cm-1'
        WRITE(OUTU,*)
        DO I=1,NQ
           WRITE(OUTU,1056) I,EXPFRQ(I)
        END DO
1056    FORMAT(6X,I4,4X,F12.1)
     END IF
     GOTO 20
  END IF
  !
  ! ... Read in 'SYMM' card
  !

  IF(WORD.EQ.'SYMM') THEN
     IFSYMM=1
     NBLOCK=IT1
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' Symmetry blocking will be used , NBLOCK=' &
          ,NBLOCK
     WRITE(OUTU,*)
     READ(IU,1045) (IBLOCK(J),J=1,NBLOCK)
     READ(IU,1047) (SBLOCK(J),J=1,NBLOCK)
     IF(IPRNT.GE.0) THEN
        WRITE(OUTU,*)
        WRITE(OUTU,*) ' The symmetry block dimensions and symbols:'
        WRITE(OUTU,*)
        WRITE(OUTU,1046) (IBLOCK(J),J=1,NBLOCK)
        WRITE(OUTU,1048) (SBLOCK(J),J=1,NBLOCK)
     END IF
1045 FORMAT(20I4)
1047 FORMAT(20A4)
1046 FORMAT(1X,20I4)
1048 FORMAT(1X,20A4)
     GOTO 20
  END IF
  !
  ! ... Return to main program
  !

  IF(WORD.EQ.'END ') THEN
     RETURN
  END IF

  !
  ! ... Process unrecognized statements
  !
  WRITE(OUTU,*) ' **** Error : unrecognized statement **** '
  WRITE(OUTU,*) ' WORD,IT1,IT2,IT3 = ', WORD,IT1,IT2,IT3
9999 CONTINUE
  CALL WRNDIE(-4,'<MOLINP>','CHECK INPUT')
  !
  RETURN
END SUBROUTINE MOLINP
#else /**/
SUBROUTINE NULL_MI
  RETURN
END SUBROUTINE NULL_MI
#endif /*  MOLVIB*/

