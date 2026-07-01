#if KEY_MOLVIB==1
SUBROUTINE RDCART(X,AMASS,NAT,NAT3,IU,IO)
  !
  ! ... Read atomic cartesian coordinates in format compatible
  ! ... with GAUSSIAN output. Masses also read
  !
  use chm_kinds
  implicit none
  INTEGER NAT,NAT3,IU,I,J,K,IO
  real(chm_real) X(NAT3),AMASS(NAT3),XX(4)
  !
  WRITE(IO,*)
  WRITE(IO,*) 'The cartesian coordinates and mass/atomic numbers'
  WRITE(IO,*)
  !
  DO I=1,NAT
     READ(IU,*) (XX(J),J=1,4)
     DO K=1,3
        AMASS(NAT*(K-1)+I)=XX(4)
        X(NAT*(K-1)+I)=XX(K)
     ENDDO
     WRITE(IO,920) I,(XX(J),J=1,4)
  ENDDO
  !
920 FORMAT(1X,I3,4X,4F12.6)
  !
  RETURN
END SUBROUTINE RDCART

SUBROUTINE RDCARC(X,AMASS,NAT,NAT3,IU,IO)
  !
  ! ... Read atomic cartesian coordinates in CHARMM format
  ! ... X,Y,Z extracted from CHARMM fromat file
  ! ... Mass specification read in from the WMAIN field; as in RDCART,
  ! ... this may be either the atom mass in amu or the atom number or the
  ! ... mass number, which will enable mass definition by the 'MASZ' or 'MASA'
  ! ... control cards in MOLINP.
  !
  use chm_kinds
  implicit none
  INTEGER NAT,NAT3,IU,I,J,K,IO
  real(chm_real) X(NAT3),AMASS(NAT3),XX(4)
  CHARACTER(len=72) TITLE
  !
  WRITE(IO,*)
  WRITE(IO,*) 'The cartesian coordinates and mass/atomic numbers'
  WRITE(IO,*) '              in CHARMM format'
  WRITE(IO,*)
  !
  DO I=1,NAT
     READ(IU,900) TITLE(1:20),(XX(J),J=1,3),TITLE(50:60),XX(4)
     DO K=1,3
        AMASS(NAT*(K-1)+I)=XX(4)
        X(NAT*(K-1)+I)=XX(K)
     ENDDO
     WRITE(IO,920) TITLE(1:20),(XX(J),J=1,3),TITLE(50:60),XX(4)
  ENDDO
  !
900 FORMAT(A20,3F10.5,A10,F10.5)
920 FORMAT(1X,A20,3F10.5,A10,F10.5)
  !
  RETURN
END SUBROUTINE RDCARC

SUBROUTINE MAMUA(NAT,NAT3,ST,ATYP,X,AMASS,IO)
  !
  !      GIVES MASSES IN ATOMIC UNITS WHEN MASS NUMBERS ARE GIVEN
  !
  use chm_kinds
  implicit none
  INTEGER NAT,NAT3,I,IO
  real(chm_real) M,MM,ST,AM,EPSILON,XMAS
  real(chm_real) X(NAT3),AMASS(NAT3)
  CHARACTER(len=*) ATYP(NAT3)
  CHARACTER(len=4) :: TYP(15)=(/'H1  ','D   ','T   ','C12 ','C13 ','N14 ','N15 ', &
       'O16 ','O17 ','O18 ','F19 ','P31 ','S32 ','S33 ','S34 '/)
  !
  EPSILON=0.0001
  AM=0.
  ST=0.
  DO I=1,NAT3
     XMAS=AMASS(I)
     M=1.+ EPSILON
     MM=1.- EPSILON
     IF(XMAS.GT.MM .AND. XMAS.LT.M) AM=1.007825
     IF(XMAS.GT.MM .AND. XMAS.LT.M) ATYP(I)=TYP(1)
     M=2.+EPSILON
     MM=2.-EPSILON
     IF(XMAS.GT.MM .AND. XMAS.LT.M) AM=2.014102
     IF(XMAS.GT.MM .AND. XMAS.LT.M) ATYP(I)=TYP(2)
     M=3.+EPSILON
     MM=3.-EPSILON
     IF(XMAS.GT.MM .AND. XMAS.LT.M) AM=3.016050
     IF(XMAS.GT.MM .AND. XMAS.LT.M) ATYP(I)=TYP(3)
     M=12.+EPSILON
     MM=12-EPSILON
     IF(XMAS.GT.MM .AND. XMAS.LT.M) AM=12.00000
     IF(XMAS.GT.MM .AND. XMAS.LT.M) ATYP(I)=TYP(4)
     M=13.+EPSILON
     MM=13.-EPSILON
     IF(XMAS.GT.MM .AND. XMAS.LT.M) AM=13.003354
     IF(XMAS.GT.MM .AND. XMAS.LT.M) ATYP(I)=TYP(5)
     M=14.+EPSILON
     MM=14.-EPSILON
     IF(XMAS.GT.MM .AND. XMAS.LT.M) AM=14.003074
     IF(XMAS.GT.MM .AND. XMAS.LT.M) ATYP(I)=TYP(6)
     M=15.+EPSILON
     MM=15.-EPSILON
     IF(XMAS.GT.MM .AND. XMAS.LT.M) AM=15.000108
     IF(XMAS.GT.MM .AND. XMAS.LT.M) ATYP(I)=TYP(7)
     M=16.+ EPSILON
     MM=16.- EPSILON
     IF(XMAS.GT.MM .AND. XMAS.LT.M) AM=15.994915
     IF(XMAS.GT.MM .AND. XMAS.LT.M) ATYP(I)=TYP(8)
     M=17.+EPSILON
     MM=17.-EPSILON
     IF(XMAS.GT.MM .AND. XMAS.LT.M) AM=15.999133
     IF(XMAS.GT.MM .AND. XMAS.LT.M) ATYP(I)=TYP(9)
     M=18.+EPSILON
     MM=18.-EPSILON
     IF(XMAS.GT.MM .AND. XMAS.LT.M) AM=15.999160
     IF(XMAS.GT.MM .AND. XMAS.LT.M) ATYP(I)=TYP(10)
     M=19.+EPSILON
     MM=19.-EPSILON
     IF(XMAS.GT.MM .AND. XMAS.LT.M) AM=18.998405
     IF(XMAS.GT.MM .AND. XMAS.LT.M) ATYP(I)=TYP(11)
     M=31.+EPSILON
     MM=31.-EPSILON
     IF(XMAS.GT.MM .AND. XMAS.LT.M) AM=30.973765
     IF(XMAS.GT.MM .AND. XMAS.LT.M) ATYP(I)=TYP(12)
     M=32.+EPSILON
     MM=32.-EPSILON
     IF(XMAS.GT.MM .AND. XMAS.LT.M) AM=31.972074
     IF(XMAS.GT.MM .AND. XMAS.LT.M) ATYP(I)=TYP(13)
     M=33.+EPSILON
     MM=33.-EPSILON
     IF(XMAS.GT.MM .AND. XMAS.LT.M) AM=32.971462
     IF(XMAS.GT.MM .AND. XMAS.LT.M) ATYP(I)=TYP(14)
     M=34.+EPSILON
     MM=34.-EPSILON
     IF(XMAS.GT.MM .AND. XMAS.LT.M) AM=33.967865
     IF(XMAS.GT.MM .AND. XMAS.LT.M) ATYP(I)=TYP(15)

     !
     IF(AM.LE.EPSILON) THEN
        ST=-1.
        WRITE(IO,1000) AMASS(I)
1000    FORMAT('MASS NUMBER',1X,F4.1,1X, &
             'IS NOT SUPPORTED BY THIS PROGRAM')
        GOTO 9999
     ENDIF
     !
     AMASS(I)=AM
  ENDDO
  !
  WRITE(IO,*) 'THE FOLLOWING MASSES WILL BE USED'
  DO I=1,NAT
     WRITE(IO,2000) I,ATYP(I),AMASS(I)
  ENDDO
2000 FORMAT('ATOM #',2X,I4,2X,'IS NUCLIDE',2X,A4,2X,'WITH MASS', &
       2X,F9.6,1X,'A.M.U.')
9999 RETURN
END SUBROUTINE MAMUA

SUBROUTINE MAMUZ(NAT,NAT3,ST,ATYP,X,AMASS,IO)
  !
  !      GIVES MASSES IN ATOMIC UNITS WHEN ATOMIC NUMBERS ARE GIVEN
  !
  !      N.B. Array ATYP has fixed dimension, temporarily
  !           Remove MAXATP and error check when this is remedied
  !
  use chm_kinds
  implicit none
  INTEGER NAT,NAT3,I,IO
  real(chm_real) M,MM,ST,AM,EPS
  real(chm_real) AMASS(NAT3),X(NAT3)
  CHARACTER(len=*) ATYP(NAT3)
  CHARACTER(len=4) :: TYP(7)=(/'H1  ','C12 ','N14 ','O16 ','F19 ','P31 ','S32 '/)
  !
  !
  EPS=0.0001
  ST=0.
  AM=0.
  DO I=1,NAT3
     M=1.+EPS
     MM= 1.-EPS
     IF(AMASS(I).GT.MM .AND. AMASS(I).LT.M) AM=1.007825
     IF(AMASS(I).GT.MM .AND. AMASS(I).LT.M) ATYP(I)=TYP(1)
     M=6.+EPS
     MM=6.-EPS
     IF(AMASS(I).GT.MM .AND. AMASS(I).LT.M) AM=12.00000
     IF(AMASS(I).GT.MM .AND. AMASS(I).LT.M) ATYP(I)=TYP(2)
     M=7.+EPS
     MM=7.-EPS
     IF(AMASS(I).GT.MM .AND. AMASS(I).LT.M) AM=14.003074
     IF(AMASS(I).GT.MM .AND. AMASS(I).LT.M) ATYP(I)=TYP(3)
     M=8.+EPS
     MM=8.-EPS
     IF(AMASS(I).GT.MM .AND. AMASS(I).LT.M) AM=15.994915
     IF(AMASS(I).GT.MM .AND. AMASS(I).LT.M) ATYP(I)=TYP(4)
     M=9.+EPS
     MM=9.-EPS
     IF(AMASS(I).GT.MM .AND. AMASS(I).LT.M) AM=18.998405
     IF(AMASS(I).GT.MM .AND. AMASS(I).LT.M) ATYP(I)=TYP(5)
     M=15.+EPS
     MM=15.-EPS
     IF(AMASS(I).GT.MM .AND. AMASS(I).LT.M) AM=30.973765
     IF(AMASS(I).GT.MM .AND. AMASS(I).LT.M) ATYP(I)=TYP(6)
     M=16.+EPS
     MM=16.-EPS
     IF(AMASS(I).GT.MM .AND. AMASS(I).LT.M) AM=31.972074
     IF(AMASS(I).GT.MM .AND. AMASS(I).LT.M) ATYP(I)=TYP(7)

     !
     IF(AM.LE.EPS) THEN
        ST=-1.
        WRITE(IO,1000) AMASS(I)
1000    FORMAT('ATOMIC MASS FOR ATOMIC NUMBER',1X,F4.1,1X, &
             'IS NOT SUPPORTED BY THIS PROGRAM')
        GOTO 9999
     ENDIF
     !
     AMASS(I)=AM
  ENDDO

  !
  WRITE(IO,*) 'THE FOLLOWING MASSES WILL BE USED'
  DO I=1,NAT
     WRITE(IO,2000) I,ATYP(I),AMASS(I)
  ENDDO
2000 FORMAT('ATOM #',2X,I4,2X,'IS NUCLIDE',2X,A4,2X,'WITH MASS', &
       2X,F9.6,1X,'A.M.U.')
9999 RETURN
END SUBROUTINE MAMUZ

SUBROUTINE RDMAT(U,NQ,NIC,NQMAX,IU,IPRNT)
  !
  ! ... Read in U matrix in Schachtschneider format
  ! ... (NQ,NIC) - actual dimensions of U
  ! ... NQMAX - first dimension of U in calling subroutine
  !
  use chm_kinds
  implicit none
  INTEGER NQ,NIC,NQMAX,IU,IPRNT
  real(chm_real) U(NQMAX,NQMAX),DAT(4)
  INTEGER I1(4),I2(4),I,J,K,L
  !
  DO I=1,NQ
     DO J=1,NIC
        U(I,J)=0.0
     ENDDO
  ENDDO
  !
30 READ(IU,1010) (I1(J),I2(J),DAT(J),J=1,4)
  IF(IPRNT.GT.2) WRITE(*,1011) (I1(J),I2(J),DAT(J),J=1,4)
  DO J=1,4
     IF(I1(J).LT.0) GOTO 45
     IF(I1(J).GT.0) THEN
        K=I1(J)
        L=I2(J)
        U(K,L)=DAT(J)
     ENDIF
  ENDDO
  GOTO 30
45 CONTINUE
  !
1010 FORMAT(4(2I3,F12.6))
1011 FORMAT(1X,4(2I3,F12.6))
  RETURN
END SUBROUTINE RDMAT

SUBROUTINE WRMAT(U,NQ,NIC,NQMAX,IU,IPRNT)
  !
  ! ... WRITE out U matrix in Schachtschneider format
  ! ... (NQ,NIC) - actual dimensions of U
  ! ... NQMAX - first dimension of U in calling subroutine
  ! ... Elememnts smaller than EPS are set to zero
  !
  use chm_kinds
  implicit none
  INTEGER NQ,NIC,NQMAX,IU,IPRNT
  real(chm_real) :: U(NQMAX,NQMAX),DAT(4),EPS=1.0e-4_chm_real
  INTEGER I1(4),I2(4),I,J,II,JJ
  !
  II=0
  DO I=1,NQ
     DO J=1,NIC
        IF(ABS(U(I,J)).GT.EPS) THEN
           II=II+1
           I1(II)=I
           I2(II)=J
           DAT(II)=U(I,J)
           IF(II.EQ.4) THEN
              WRITE(IU,1010) (I1(JJ),I2(JJ),DAT(JJ),JJ=1,4)
              II=0
           ENDIF
        ENDIF
     ENDDO
  ENDDO
  IF(II.GT.0) WRITE(IU,1010) (I1(JJ),I2(JJ),DAT(JJ),JJ=1,II)
  II=-1
  WRITE(IU,1010) II
  !
1010 FORMAT(4(2I3,F12.6))
  RETURN
END SUBROUTINE WRMAT

SUBROUTINE NORMR(U,NQ,NIC,NQMAX)
  !
  ! ... Normalize rows of matrix U
  ! ... (NQ,NIC) - actual dimensions of U
  ! ... NQMAX - first dimension of U in calling subroutine
  !
  use chm_kinds
  implicit none
  INTEGER NQ,NIC,NQMAX
  real(chm_real) U(NQMAX,NQMAX),S
  INTEGER I,J
  !
  DO I=1,NQ
     S=0.0
     DO J=1,NIC
        S=S+U(I,J)*U(I,J)
     ENDDO
     S=SQRT(S)
     IF(S.LT.0.000001) THEN
        WRITE(*,*) ' *** Error in NORMR ***'
        WRITE(*,*) '     Norm too small for U row no. ',I
        CALL WRNDIE(-4,'<NORMR>',' RE-NORMALIZE')
     ENDIF
     DO J=1,NIC
        U(I,J)=U(I,J)/S
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE NORMR

SUBROUTINE RFORC(F,FX,NAT,NAT3,NQM,IU,IPRNT,IO)
  !
  !     READING FORCE CONSTANTS IN CARTESIAN COORDINATES
  !     AS IN GAUSSIAN90 OUTPUT
  !     ROWS AND COLUMNS PERMUTED TO GET CARTESIAN COORDS
  !     IN FORM [X1,X2,...,XN,Y1,...,YN,Z1,...,ZN] AS IN BMAT
  !     VALUES CONVERTED FROM H/BOHR**2 TO MDYNE/A
  !
  use chm_kinds
  use stream
  implicit none
  INTEGER NAT,NAT3,NQM,IU,IPRNT,IO,IA,M
  CHARACTER(len=4) DUMMY(20)
  INTEGER IAT(5),N1,N2,I,II,IFIRST,ILAST,J,JJ,KK,IM,JM
  real(chm_real) F(NQM,NQM),FX(NQM,NQM),FXA(5),SCAL
  !
  DO N1=1,NAT3
     DO N2=1,NAT3
        FX(N1,N2)=0.
     ENDDO
  ENDDO
  READ(IU,8000) DUMMY
8000 FORMAT(20A4)
110 CONTINUE
  DO I=1,5
     IAT(I)=0
  ENDDO
  READ(IU,1008) (IAT(I),I=1,5)
1008 FORMAT(5(10X,I4))
  IFIRST=IAT(1)
  ILAST=IAT(5)
  IF(ILAST .NE. 0) GOTO 117
  DO I=1,5
     II=6-I
     IF(IAT(II) .EQ. 0) GOTO 116
     ILAST=IAT(II)
     GOTO 117
116  CONTINUE
  ENDDO
117 CONTINUE
  DO I=IFIRST,NAT3
     READ(IU,1002) IA,(FXA(KK),KK=1,5)
1002 FORMAT(I4,5E14.6)
     KK=0
     DO J=IFIRST,ILAST
        KK=KK+1
        FX(I,J)=FXA(KK)
     ENDDO
  ENDDO
  IF(ILAST .LT. NAT3) GOTO 110
  DO J=1,NAT3
     DO I=J,NAT3
        FX(J,I)=FX(I,J)
     ENDDO
  ENDDO
  !      IF(IOLEV.GT.0) CLOSE(UNIT=IU)
  !
  IF(IPRNT.GT.1) THEN
     WRITE(IO,*) 'FORCE CONSTANT MATRIX IN CARTESIAN COORDINATES'
     WRITE(IO,3004) ((I,J,FX(I,J),J=1,NAT3),I=1,NAT3)
3004 FORMAT(3('FX(',I2,',',I2,')=',E14.6,4X))
  ENDIF

  !     PERMUTING ROWS IN THE F MATRIX TO GET FORMAT USED IN BMAT

  II=0
  DO I=1,NAT
     DO M=1,3
        II=II+1
        IM=(M-1)*NAT+I
        DO J=1,NAT3
           F(IM,J)=FX(II,J)
        ENDDO
     ENDDO
  ENDDO
  !
  !     PERMUTING COLUMNS IN THE F MATRIX TO GET FORMAT USED IN BMAT

  JJ=0
  DO J=1,NAT
     DO M=1,3
        JJ=JJ+1
        JM=(M-1)*NAT+J
        DO I=1,NAT3
           FX(I,JM)=F(I,JJ)
        ENDDO
     ENDDO
  ENDDO
  !
  SCAL=15.5675
  !     SCAL CONVERTS FORCE CONSTANTS FROM HARTEE/BOHR TO MDYN/A
  DO I=1,NAT3
     DO J=1,NAT3
        F(I,J)=SCAL*FX(I,J)
     ENDDO
  ENDDO
  !
  IF(IPRNT.GT.1) THEN
     WRITE(IO,*) &
          'PERMUTED FORCE CONSTANT MATRIX IN CARTESIAN COORDINATES [mdyn/A]'
     WRITE(IO,3004) ((I,J,F(I,J),J=1,NAT3),I=1,NAT3)
  ENDIF
  !
  RETURN
END SUBROUTINE RFORC

SUBROUTINE REIGEN(NA,NF,NFI,NQM,LS,LX,NFRQ,FREQ,IU,IPRNT,IO)
  !
  !     READING EIGENVECTORS IN CARTESIAN COORDINATES
  !     AS IN GAUSSIAN84 OUTPUT
  !
  use chm_kinds
  use stream
  implicit none
  CHARACTER(len=4) DUMMY(20)
  INTEGER NA,NF,NFI,NQM,NFRQ,IU,IPRNT,IO
  real(chm_real) FREQ(NQM)
  real(chm_real) LX(NQM,NQM),LS(NQM,NQM)
  INTEGER IAT(5),JUNK(3),I,IFIRST,ILAST,II,KK,LL,IM,J
  INTEGER IUU,M
  real(chm_real) LXA(5)
  !
  READ(IU,8000) DUMMY
  IF(IPRNT.GE.4) WRITE(IO,2000) DUMMY
8000 FORMAT(20A4)
2000 FORMAT(1X,20A4)
210 CONTINUE
  READ(IU,1003) (IAT(I),I=1,5)
  IF(IPRNT.GE.4) WRITE(IO,1003) (IAT(I),I=1,5)
1003 FORMAT(18X,5(6X,I4))
  IFIRST=IAT(1)
  ILAST=IAT(5)
  IF(ILAST .NE. 0) GOTO 217
  DO I=1,5
     II=I-1
     II=5-II
     IF(IAT(II) .EQ. 0) GOTO 216
     ILAST=IAT(II)
     GOTO 217
216  CONTINUE
  ENDDO
217 CONTINUE
  READ(IU,8000) DUMMY
  IF(IPRNT.GE.4) WRITE(IO,2000) DUMMY
  READ(IU,7999) (FREQ(LL),LL=IFIRST,ILAST)
  IF(IPRNT.GE.4) WRITE(IO,7999) (FREQ(LL),LL=IFIRST,ILAST)
7999 FORMAT(23X,5F10.4)
  READ(IU,8000) DUMMY
  IF(IPRNT.GE.4) WRITE(IO,2000) DUMMY
  DO I=1,NF
     READ(IU,1000) (JUNK(IUU),IUU=1,3),(LXA(KK),KK=1,5)
     IF(IPRNT.GE.4) THEN
        WRITE(IO,1000) (JUNK(IUU),IUU=1,3),(LXA(KK),KK=1,5)
     ENDIF
1000 FORMAT(I4,2(4X,I2),7X,5F10.5)
     KK=0
     DO J=IFIRST,ILAST
        KK=KK+1
        LS(I,J)=LXA(KK)
     ENDDO
  ENDDO
  IF(ILAST .LT. NFI) GOTO 210
  !      IF(IOLEV.GT.0) CLOSE(UNIT=IU)
  !
  NFRQ=ILAST
  WRITE(IO,*)
  WRITE(IO,*) ' The following eigenvectors have been read'
  WRITE(IO,*)
  WRITE(IO,3005) ((I,J,LS(I,J),J=1,NFI),I=1,NF)
3005 FORMAT(3('LX(',I2,',',I2,')=',F8.5,2X))
  !
  !     PERMUTING LS TO GET LX IDENTICAL TO THAT USED IN SUBROUTINE
  !     BMAT
  !
  II=0
  DO I=1,NA
     DO M=1,3
        II=II+1
        IM=(M-1)*NA+I
        DO J=1,NFI
           LX(IM,J)=LS(II,J)
        ENDDO
     ENDDO
  ENDDO
  !
  IF(IPRNT.GT.2) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' Final LX matrix'
     WRITE(IO,*)
     WRITE(IO,3006) ((I,J,LX(I,J),J=1,NFI),I=1,NF)
  ENDIF
3006 FORMAT(3('LX(',I2,',',I2,')=',F8.5,2X))
  RETURN
END SUBROUTINE REIGEN

SUBROUTINE LXNCHK(LX,NC,NAT3,NQM,NQ,AMASS,V1,V2,IPRNT,IO)
  !
  !   Analyzes normalization of the cartesian eigenvector matrix
  !   LX. Application - check form of LX that has been supplied
  !   to MOLVIB  from outside.
  !
  !   LX(NAT3,NC) - cartesian eigenvectors in columns; NAT3=3*NAT
  !       NC is either NAT3 or NQ, if only vibrational eigenvectors
  !       are provided
  !   NQM - dimension of LX in calling subroutine
  !   NQ  - no. of vibrational degrees of freedom
  !   AMASS - array of atomic masses [amu]
  !   V1,V2 - work arrays
  !   IPRNT - print level
  !
  use chm_kinds
  implicit none
  INTEGER NC,NAT3,NQM,NQ,IPRNT,IO
  real(chm_real) LX(NQM,NQM),AMASS(NAT3),V1(NAT3),V2(NAT3),S,EPS
  INTEGER I,J
  LOGICAL QC1,QR1,QRM,QVIB,QCM
  !
  ! ... First check how many columns there are
  !
  WRITE(IO,*) ' LXNCHK : analyzing normalization of LX'
  WRITE(IO,*)
  QC1=.FALSE.
  QR1=.FALSE.
  QCM=.FALSE.
  IF(NC.EQ.NAT3) THEN
     QVIB=.FALSE.
     WRITE(IO,*) '  Full LX matrix provided'
  ENDIF
  IF(NC.EQ.NQ) THEN
     QVIB=.TRUE.
     WRITE(IO,*) '  Vibrational eigenvectors provided'
  ENDIF
  IF(NC.NE.NAT3 .AND. NC.NE.NQ) THEN
     WRITE(IO,*) ' **** Error in LXNCHK : incorrect LX', &
          ' dimension'
     WRITE(IO,*)
     CALL WRNDIE(-4,'<LXNCHK>',' SO LONG')
  ENDIF
  !
  ! ... Calculate column norms
  !
  EPS=1.0D-6
  DO J=1,NC
     S=0.0D0
     DO I=1,NAT3
        S=S+LX(I,J)**2
     ENDDO
     S=SQRT(S)
     IF(S.LT.EPS) THEN
        WRITE(IO,*) ' **** Norm too small, J=',J
     ENDIF
     V1(J)=S
  ENDDO  ! J=1,NC
  !
  EPS=1.0D-2
  QC1=.TRUE.
  DO J=1,NC
     IF(ABS(V1(J)-1.0D0).GT.EPS) QC1=.FALSE.
  ENDDO
  IF(QC1) THEN
     WRITE(IO,*) ' The columns of LX are normalized to 1'
     WRITE(IO,*)
  ENDIF
  !
  IF(IPRNT.GT.2) THEN
     WRITE(IO,*) ' The LX matrix column norms: '
     WRITE(IO,900) (V1(J),J=1,NC)
     WRITE(IO,*)
  ENDIF
900 FORMAT(10F8.4)
  !
  IF(QVIB) RETURN
  !
  ! ... Calculate row norms
  !
  EPS=1.0D-6
  DO I=1,NAT3
     S=0.0D0
     DO J=1,NC
        S=S+LX(I,J)**2
     ENDDO
     S=SQRT(S)
     IF(S.LT.EPS) THEN
        WRITE(IO,*) ' **** Norm too small, I=',I
     ENDIF
     V2(I)=S
  ENDDO  !  I=1,NAT3
  !
  EPS=1.0D-2
  QR1=.TRUE.
  DO I=1,NAT3
     IF(ABS(V2(I)-1.0D0).GT.EPS) QR1=.FALSE.
  ENDDO
  IF(QR1) THEN
     WRITE(IO,*) ' The rows of LX are normalized to 1'
     WRITE(IO,*)
  ENDIF
  IF(IPRNT.GT.2) THEN
     WRITE(IO,*) ' The LX matrix row norms: '
     WRITE(IO,900) (V2(I),I=1,NAT3)
     WRITE(IO,*)
  ENDIF
  !
  DO I=1,NAT3
     V2(I)=V2(I)*AMASS(I)
  ENDDO
  QRM=.TRUE.
  DO I=1,NAT3
     IF(ABS(V2(I)-1.0D0).GT.EPS) QRM=.FALSE.
  ENDDO
  IF(QRM) THEN
     WRITE(IO,*) ' The rows of LX are normalized to M-1'
     WRITE(IO,*)
  ENDIF
  IF(IPRNT.GT.2) THEN
     DO I=1,NAT3
        V2(I)=1.0D0/AMASS(I)
     ENDDO
     WRITE(IO,*) ' The inverse masses:'
     WRITE(IO,900) (V2(I),I=1,NAT3)
     WRITE(IO,*)
  ENDIF
  !
  IF(.NOT.(QC1.OR.QR1.OR.QRM)) THEN
     WRITE(IO,*) ' Normalization of LX is unknown'
     WRITE(IO,*)
  ENDIF
  !
  RETURN
END SUBROUTINE LXNCHK

SUBROUTINE RDSECO(FX,FS,V1,V2,V3,NAT,NAT3,NQM,IU,IPRNT,IO)
  !
  !     Reading CHARMM formatted SECO file to get F matrix
  !     in cartesian coordinates
  !
  use chm_kinds
  implicit none
  INTEGER NAT,NAT3,NQM,IU,IPRNT,IO
  real(chm_real) FX(NQM,NQM),ENERGY,SCAL
  real(chm_real) FS(NQM,NQM),V1(NQM),V2(NQM),V3(NQM)
  INTEGER NATOM,I,J,K,II,I1,JJ,IM,JM,M
  !
  WRITE(IO,*)
  WRITE(IO,*) ' Reading CHARMM second derivative file '
  WRITE(IO,*)
  !
  ! ... 1. Read NATOM and ENERGY
  !
  READ(IU,900) NATOM
  READ(IU,901) ENERGY
  WRITE(IO,902) NATOM,ENERGY
900 FORMAT(I5)
901 FORMAT(3F20.10)
902 FORMAT(2X,' NATOM, ENERGY = ', I5, F16.6)
  IF(NATOM.NE.NAT) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' **** Error in RDSECO : NAT .NE. NATOM '
     WRITE(IO,*)
     CALL WRNDIE(-4,'<RDSECO>','CHECK NUMBER OF ATOMS')
  ENDIF
  !
  ! ... 2. Read first derivatives into dummy arrays V1,V2,V3
  !
  READ(IU,901) (V1(K),V2(K),V3(K),K=1,NAT)
  IF(IPRNT.GE.1) THEN
     WRITE(IO,*)
     WRITE(IO,*) '        The cartesian  energy gradient '
     WRITE(IO,*) '  Atom #         FX          FY          FZ'
     WRITE(IO,*)
     DO I=1,NAT
        WRITE(IO,911) I,V1(I),V2(I),V3(I)
     ENDDO
  ENDIF
911 FORMAT(2X,I3,2X,3F12.6)
  !
  ! ... 3. Read in upper triangle of CHARMM FX matrix
  !
  DO I=1,NAT3
     READ(IU,920) (FX(I,K),K=I,NAT3)
  ENDDO
920 FORMAT(F20.10)
  !
  ! ... 4. Symmetrize
  !
  DO I=1,NAT3
     I1=I+1
     DO J=I1,NAT3
        FX(J,I)=FX(I,J)
     ENDDO
  ENDDO
  !
  ! ... Permute rows and columns to agree with BMAT
  ! ... and convert from [kcal/mol*A**2] to [mdyne/A]
  !
  II=0
  DO I=1,NAT
     DO M=1,3
        II=II+1
        IM=(M-1)*NAT+I
        DO J=1,NAT3
           FS(IM,J)=FX(II,J)
        ENDDO
     ENDDO
  ENDDO
  !
  JJ=0
  DO J=1,NAT
     DO M=1,3
        JJ=JJ+1
        JM=(M-1)*NAT+J
        DO I=1,NAT3
           FX(I,JM)=FS(I,JJ)
        ENDDO
     ENDDO
  ENDDO
  !
  SCAL=143.94D0
  DO I=1,NAT3
     DO J=1,NAT3
        FX(I,J)=FX(I,J)/SCAL
     ENDDO
  ENDDO
  !
  ! ... There are still cartesian coords left in this file
  ! ... but reading stops. If necessary, get X,Y,Z in the same format
  ! ... as forces.
  RETURN
END SUBROUTINE RDSECO

SUBROUTINE REIGCH(NQ,NAT,NAT3,IPRNT,IU, &
     NQM,W1,LX,V1,V2,DD,IO)
  !
  !  Subroutine reads in CHARMM binary file containing vibrational eigenvectors
  !  in cartesian coordinates. It is assumed that the first six normal modes,
  !  representing translations and rotations are not in the file.
  !  The number of modes and number of atoms have t agree with previous
  !  specifications.
  !  NQ - no. of normal modes (vibrational degrees of freedom)
  !  NAT - no. f atoms, NAt3=3*NAT
  !  CFACT - conversion factor from sqrt(eigenvalues) to cm-1
  !
  use chm_kinds
  implicit none
  INTEGER NQ,NAT,NAT3,IPRNT,IU,NQM,IO
  real(chm_real) W1(NQM,NQM),LX(NQM,NQM),V1(NQM),V2(NQM),DD(NQM)
  CHARACTER(len=4) HDR
  CHARACTER(len=80) TITL(32)
  INTEGER ICNTRL(20),NTITL,I,J,K,II,IK,NNEG
  real(chm_real) :: CFACT=108.592_chm_real
  !
  ! ... Header
  !
  READ(IU) HDR,ICNTRL
  WRITE(IO,*) HDR,ICNTRL
  IF(HDR.NE.'NMDS') THEN
     WRITE(IO,*)
     WRITE(IO,*) ' **** REIGCH : Warning, header incorrect '
     WRITE(IO,*)
  ENDIF
  IF(ICNTRL(1).NE.NQ) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' **** REIGCH : Warning, dimensions do not match '
     WRITE(IO,*) ' Resetting NQ from ',NQ,'  to ',ICNTRL(1)
     WRITE(IO,*)
     NQ=ICNTRL(1)
     IF(NQ.GT.NQM) CALL WRNDIE(-4,'<REIGCH>','NQM exceeded')
  ENDIF
  IF(ICNTRL(2).NE.NAT3 .OR. ICNTRL(3).NE.NAT) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' **** REIGCH : Warning, atom numbers do not match '
     WRITE(IO,*) ' NAT,NAT3 = ',NAT,NAT3, &
          '   Input = ',ICNTRL(3),ICNTRL(2)
     WRITE(IO,*)
     CALL WRNDIE(-4,'<REIGCH>',' SO LONG')
  ENDIF
  !
  ! ... 2.  CHARMM binary title
  !
  READ(IU) NTITL,(TITL(I),I=1,NTITL)
  DO I=1,NTITL
     WRITE(IO,*) TITL(I)
  ENDDO
  !
  ! ... 3. CHARMM masses, put in temp. array and not used further
  !
  READ(IU) (V1(J),J=1,NAT)
  !
  ! ... 4. CHARMM eigenvalues; converted to cm-1
  !
  READ(IU) (V2(J),J=1,NQ)
  NNEG=0
  DO J=1,NQ
     DD(J)=CFACT*SQRT(ABS(V2(J)))
     IF(V2(J).LT.0) NNEG=NNEG+1
  ENDDO
  IF(NNEG.NE.0) THEN
     WRITE(IO,*) ' REIGCH: read ',NNEG,' negative eigenvalues '
  ENDIF
  !
  ! ... 5. CHARMM cartesian eigenvectors
  !
  DO I=1,NQ
     READ(IU) (W1(J,I),J=1,NAT3)
  ENDDO
  !
  ! ... Permute rows of LS to transform to BMAT variant of cartesian
  ! ... coords, i.e. [X1,....,XN,Y1,...YN,Z1,...,ZN] from the
  ! ... CHARMM type  [X1,Y1,Z1,...,XN,YN,ZN].
  II=0
  DO I=1,NAT
     DO K=1,3
        II=II+1
        IK=(K-1)*NAT+I
        DO J=1,NQ
           LX(IK,J)=W1(II,J)
        ENDDO
     ENDDO
  ENDDO
  !
  IF(IPRNT.GE.2) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' Vibrational eigenvalues and eigenvectors LX '
     WRITE(IO,*) '             in MOLVIB form '
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,900) I,DD(I),(LX(J,I),J=1,NAT3)
     ENDDO
  ENDIF
900 FORMAT(1X,I3,2X,F8.1,2X,10F9.5,12(/16X,10F9.5))
  !
  RETURN
END SUBROUTINE REIGCH
#else /**/
SUBROUTINE NULL_MO
  RETURN
END SUBROUTINE NULL_MO
#endif /*  MOLVIB*/

