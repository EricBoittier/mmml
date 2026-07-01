#if KEY_NOMISC==0
SUBROUTINE MKALIS(NATOM,FLAGS,MTESTA,TESTAT,NTESTA)
  !-----------------------------------------------------------------------
  !     Routine makes TESTAT array from selection array FLAGS
  !
  use chm_kinds
  implicit none
  !     input/output
  INTEGER   NATOM
  INTEGER FLAGS(NATOM)
  INTEGER   MTESTA
  INTEGER   TESTAT(MTESTA)
  INTEGER   NTESTA
  INTEGER   I
  !
  !     begin
  NTESTA=0
  DO I=1,NATOM
     IF (FLAGS(I).EQ.1) THEN
        NTESTA=NTESTA+1
        IF (NTESTA.GT.MTESTA) THEN
           CALL WRNDIE(-1,'MKALIS', &
                'maximum number of table atoms exceeded')
        ENDIF
        TESTAT(NTESTA)=I
     ENDIF
  ENDDO
  !
  RETURN
END SUBROUTINE MKALIS
#endif 

SUBROUTINE SBINTG
  !-----------------------------------------------------------------------
  !     Routine integrates forces to get potential and generates
  !     cubic spline approximation of the potential.
  !     Use after SBSPHR.
  !
  !      Syntax:
  !      > SBOUnd POTEntial INPUt <integer> OUTPut <integer>
  !
  !      Axel Brunger, Cambridge, 2-AUG-83
  !      ---------------------------------
  !
  use chm_kinds
  use dimens_fcm
  use memory
  use exfunc
  !
  use comand
  use ctitla
  use stream
  use string

  implicit none
  !     local
  real(chm_real),allocatable,dimension(:) :: R
  real(chm_real),allocatable,dimension(:) :: AFORCE
  real(chm_real),allocatable,dimension(:) :: APOT
  real(chm_real),allocatable,dimension(:) :: CSPLIN
  INTEGER IUNIT,OUNIT
  !
  INTEGER NR, NTESTA
  INTEGER I
  LOGICAL EOF
  !
#if KEY_NOMISC==1
  CALL WRNDIE(-1,'<SBINTG>','SBOUND code not compiled.')
#else /**/
  !
  !     begin
  IUNIT=GTRMI(COMLYN,COMLEN,'INPU',5)
  OUNIT=GTRMI(COMLYN,COMLEN,'OUTP',OUTU)
  !
  IF(PRNLEV.GE.2) WRITE(OUTU,1000) IUNIT
1000 FORMAT(' SBINTG: Force table being read from unit',I4)
  !
  IF(IOLEV.GT.0) CALL RDTITL(TITLEB,NTITLB,IUNIT,0)
  !
  EOF=.FALSE.
  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE.,.FALSE.,' ')
  !
  NR=GTRMI(COMLYN,COMLEN,'ROWS',0)
  NTESTA=GTRMI(COMLYN,COMLEN,'COLU',0)-1
  IF (NR.EQ.0.OR.NTESTA.EQ.0) THEN
     CALL WRNDIE(0,'SBINTG','Error reading force table')
  ENDIF
  !
  IF(IOLEV.GT.0) THEN
     DO I=1,NTESTA+1
        READ(IUNIT,*)
     ENDDO
  ENDIF
  !
  !     dynamic
  call chmalloc('sbound.src','SBINTG','R',NR,crl=R)
  call chmalloc('sbound.src','SBINTG','AFORCE',NTESTA*NR,crl=AFORCE)
  call chmalloc('sbound.src','SBINTG','APOT',NTESTA*NR,crl=APOT)
  call chmalloc('sbound.src','SBINTG','CSPLIN',NTESTA*NR*3,crl=CSPLIN)
  !
  call SBINT2(IUNIT,OUNIT,NR,NTESTA,R,AFORCE,APOT,CSPLIN)
  !
  call chmdealloc('sbound.src','SBINTG','R',NR,crl=R)
  call chmdealloc('sbound.src','SBINTG','AFORCE',NTESTA*NR,crl=AFORCE)
  call chmdealloc('sbound.src','SBINTG','APOT',NTESTA*NR,crl=APOT)
  call chmdealloc('sbound.src','SBINTG','CSPLIN',NTESTA*NR*3,crl=CSPLIN)
  !
  RETURN
END SUBROUTINE SBINTG

SUBROUTINE SBINT2(IUNIT,OUNIT,NR,NTESTA,R,AFORCE,APOT,CSPLIN)
  !-----------------------------------------------------------------------
  !     see SBINTG above
  !
  use chm_kinds
  !     input/output
  use ctitla
  use stream
  use parallel,only: psnd8
  implicit none
  INTEGER IUNIT,OUNIT,NR,NTESTA
  real(chm_real)  R(NR), AFORCE(NR,NTESTA), APOT(NR,NTESTA)
  real(chm_real)  CSPLIN(NR,3,NTESTA)
  !     local
  INTEGER IR, ITEST, IER, J
  !
  !     begin
  IF (IOLEV.GE.0) THEN      ! MFC, 11-Feb-99
     DO IR=1,NR
        READ(IUNIT,*) R(IR),(AFORCE(IR,ITEST),ITEST=1,NTESTA)
     ENDDO
  ENDIF                     ! MFC, 11-Feb-99
  !
#if KEY_PARALLEL==1
  CALL PSND8(R,NR)
  DO ITEST=1,NTESTA
     CALL PSND8(AFORCE(1,ITEST),NR)
  ENDDO
#endif 
  !
  !     loop over all test atoms
  DO ITEST=1,NTESTA
     !
     !     now do integration of AFORCE using the trapezoidal rule
     APOT(1,ITEST)=0.0
     DO IR=2,NR
        APOT(IR,ITEST)=APOT(IR-1,ITEST)- 0.5 * (R(IR)-R(IR-1)) * &
             (AFORCE(IR,ITEST)+AFORCE(IR-1,ITEST))
     ENDDO
     !
     !     now generate cubic splines
     IER=0
     CALL SPLINR(R,APOT(1,ITEST),NR,CSPLIN(1,1,ITEST),NR,IER)
     IF (IER.NE.0 .AND. WRNLEV.GE.2) WRITE(OUTU,1000) IER
     !
  ENDDO
1000 FORMAT(' SBINT2: SPLINR terminated with error IER=',I6)
  !
  IF(OUNIT.NE.OUTU .AND. IOLEV.LT.0) RETURN
  IF(OUNIT.EQ.OUTU .AND. PRNLEV.LE.2) RETURN
  !
  !     write output file
  CALL WRTITL(TITLEB,NTITLB,OUNIT,0)
  !
  DO ITEST=1,NTESTA
     WRITE(OUNIT,2000) NR, 5
     DO IR=1,NR
        WRITE(OUNIT,2020) R(IR),APOT(IR,ITEST), &
             (CSPLIN(IR,J,ITEST),J=1,3)
     ENDDO
  ENDDO
2000 FORMAT(' ROWS ',I10,'     COLUmns ',I10,/, &
          ' COLUmn    1          radius',/, &
          ' COLUmn    2          potential',/, &
          ' COLUmn    3          spline C1',/, &
          ' COLUmn    4          spline C2',/, &
          ' COLUmn    5          spline C3')
2020 FORMAT(1X,5(G18.10,1X))
  !
  RETURN
END SUBROUTINE SBINT2

SUBROUTINE BNDSET
  !-----------------------------------------------------------------------
  !     Solvent boundary routine to set boundary geometry
  !     and specify the atoms referring to the tables.
  !
  !      Syntax:
  !      > SBOUnd SET XREF <real> YREF <real> ZREF <real>
  !      ASSIgn <table number> <selection-syntax>
  !
  !      Axel Brunger, Cambridge, JUL-83
  !      -------------------------------
  !
  use chm_kinds
  use dimens_fcm
  use memory
  use sbound
  use comand
  use coord
  use psf
  use select
  use string
  use stream

  implicit none
  !     local
  integer,allocatable,dimension(:) :: FLAGS
  LOGICAL DONE
  INTEGER ITABL
  CHARACTER(len=4) WRD
  !
  !     begin
  DONE=.FALSE.
10 CONTINUE
  WRD=NEXTA4(COMLYN,COMLEN)
  IF (WRD.EQ.'XREF') THEN
     SSXREF=NEXTF(COMLYN,COMLEN)
  ELSE IF (WRD.EQ.'YREF') THEN
     SSYREF=NEXTF(COMLYN,COMLEN)
  ELSE IF (WRD.EQ.'ZREF') THEN
     SSZREF=NEXTF(COMLYN,COMLEN)
  ELSE IF (WRD.EQ.'ASSI') THEN
     ITABL=NEXTI(COMLYN,COMLEN)
     call chmalloc('sbound.src','BNDSET','FLAGS',NATOM,intg=FLAGS)
     call SELCTA(COMLYN,COMLEN,FLAGS,X,Y,Z,WMAIN,.TRUE.)
     call MKALIS(NATOM,FLAGS,NMCATM,CATOM(1,ITABL),NCATOM(ITABL))
     call chmdealloc('sbound.src','BNDSET','FLAGS',NATOM,intg=FLAGS)

  ELSE IF (WRD.EQ.'    ') THEN
     DONE=.TRUE.
  ELSE
     DONE=.TRUE.
     CALL JOINWD(COMLYN,MXCMSZ,COMLEN,WRD,4)
  ENDIF
  IF (.NOT.(DONE)) GOTO 10
  CALL XTRANE(COMLYN,COMLEN,'BNDSET')
  !
  DO ITABL=1,NCTABL
     IF(PRNLEV.GE.2) WRITE(OUTU,9000) ' BNDSET: ',NCATOM(ITABL), &
          ' atoms have been assigned to table ',ITABL
  ENDDO
  !
  IF(PRNLEV.GE.2) WRITE(OUTU,9010) &
       ' BNDSET: Solvent boundary reference point = ( ', &
       SSXREF,',',SSYREF,',',SSZREF,' )'
  QBOUND=.TRUE.
9000 FORMAT(A,I8,A,I6)
9010 FORMAT(A,F8.3,A,F8.3,A,F8.3,A)
  RETURN
END SUBROUTINE BNDSET

SUBROUTINE SBREAD
  !-----------------------------------------------------------------------
  !     This routine reads in the boundary potential splines
  !     for the boundary on h and o atoms in water.
  !
  !      Syntax:
  !      > SBOUnd READ UNIT <integer>
  !
  !      Axel Brunger, Cambridge, 3-AUG-83
  !      ---------------------------------
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  use ctitla
  use comand
  use sbound
  use stream
  use string
  use parallel,only: psnd4,psnd8

  implicit none
  !     local
  INTEGER IUNIT, IR, J, NCOLU
  LOGICAL EOF
  !
  !     begin
  IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',5)
  IF(PRNLEV.GE.2) WRITE(OUTU,100) IUNIT
100 FORMAT(' SBREAD: boundary potential spline being read from unit', &
       I4)
  !
  IF(IOLEV.GT.0) THEN
     CALL RDTITL(TITLEB,NTITLB,IUNIT,0)
     !
     NCTABL=0
     EOF=.FALSE.
200  CONTINUE
     ! cannot use compress here, in case we are running parallel version,
     ! we don't want this line to be broadcast.
     CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.FALSE.,.FALSE.,' ')
     IF (.NOT.EOF) THEN
        NCTABL=NCTABL+1
        NCATOM(NCTABL)=0
        !
        NR=GTRMI(COMLYN,COMLEN,'ROWS',0)
        NCOLU=GTRMI(COMLYN,COMLEN,'COLU',0)
        !
        IF (NR.EQ.0.OR.NCOLU.NE.NSPLIN+2) THEN
           CALL WRNDIE(0,'SBINTG','Error reading potential spline')
        ENDIF
        IF (NR.GT.NMFTAB) CALL WRNDIE(-1,'SBINTG','NMFTAB exceeded')
        DO J=1,NSPLIN+2
           READ(IUNIT,*,ERR=999,END=999)
        ENDDO
        DO IR=1,NR
           READ(IUNIT,*,ERR=999,END=999) SBR(IR,NCTABL), &
                APOT(IR,NCTABL),(CSPLIN(IR,J,NCTABL),J=1,3)
        ENDDO
     ENDIF
     IF (.NOT.(EOF)) GOTO 200
     !
     IF(PRNLEV.GE.2) WRITE(OUTU,'(A,I8,A)') &
          ' SBREAD: ',NCTABL,' cubic spline tables were read'
  ENDIF
  !
#if KEY_PARALLEL==1
  CALL PSND4(NR,1)
  CALL PSND4(NCTABL,1)
  CALL PSND8(SBR(1,NCTABL),NR)
  CALL PSND8(APOT(1,NCTABL),NR)
  DO J=1,3
     CALL PSND8(CSPLIN(1,J,NCTABL),NR)
  ENDDO
  DO J=1,NCTABL
     NCATOM(J)=0
  ENDDO
#endif 
  !
  RETURN
999 CALL WRNDIE(-3,'SBREAD','Error during read')
  RETURN
END SUBROUTINE SBREAD

SUBROUTINE BNDRY(EBOUND,X,Y,Z,DX,DY,DZ,QECONT,ECONT, &
     DD1,IUPT,QSECD)
  !-----------------------------------------------------------------------
  !     This subroutine computes the boundary force and potential on
  !     specified atoms.
  !     The boundary is represented by cubic splines stored SBOUND.FCM.
  !
  !      Axel Brunger, Cambridge, 3-AUG-83
  !      ---------------------------------
  !
  use chm_kinds
  use dimens_fcm
  use number
  !
  use psf
  use sbound
  use stream
#if KEY_DIMB==1
  use dimb  
#endif
  implicit none
  !
  real(chm_real) EBOUND, DD1(*)
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  LOGICAL QECONT
  real(chm_real) ECONT(*)
  INTEGER IUPT(*)
  LOGICAL QSECD
  !     local
  INTEGER ITABL, I, II, IADD, IAT
  real(chm_real) DELX, DELY, DELZ, E, DE, DDE, R
  !     begin
  EBOUND=0.0
  IF (QBOUND) THEN
     !
     DO ITABL=1,NCTABL
        IF (NCATOM(ITABL).LE.0) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,'(A,I6)') &
                ' BNDRY: zero atoms selected for table ',ITABL
           CALL DIEWRN(-1)
        ENDIF
        DO I=1,NCATOM(ITABL)
           IAT=CATOM(I,ITABL)
           E=0.0
           DE=0.0
           DDE=0.0
           !
           DELX=X(IAT)-SSXREF
           DELY=Y(IAT)-SSYREF
           DELZ=Z(IAT)-SSZREF
           R=(DELX*DELX+DELY*DELY+DELZ*DELZ)
           !
           !     we assume that the radial potential and all derivatives are
           !     zero at R=0
           IF (R.GT.RSMALL) THEN
              R=SQRT(R)
              DELX=DELX/R
              DELY=DELY/R
              DELZ=DELZ/R
              !
              IF (R.GE.SBR(1,ITABL)) THEN
                 !
                 !     get radial potential and its derivatives
                 CALL ESPLIN(NR,SBR(1,ITABL),APOT(1,ITABL),NMFTAB, &
                      CSPLIN(1,1,ITABL),R,E,DE,DDE)
              ENDIF
              !
              !     energy
              EBOUND=EBOUND+E
              IF(QECONT) ECONT(IAT)=ECONT(IAT)+E
              !
              !     first derivatives
              DX(IAT)=DX(IAT)+DE*DELX
              DY(IAT)=DY(IAT)+DE*DELY
              DZ(IAT)=DZ(IAT)+DE*DELZ
              !
              !     now the second derivatives
              IF (QSECD) THEN

#if KEY_DIMB==1
                 IF(QCMPCT) THEN
                    CALL BNDCMP(IAT,DE,DDE,DELX,DELY,DELZ,R,DD1, &
                         PINBCM,PJNBCM)
                 ELSE
#endif /*  DIMB*/

                    II=3*IAT-2
                    IADD=IUPT(II)+II
                    DD1(IADD)=DD1(IADD)+DDE*DELX*DELX+DE*(1.0-DELX*DELX)/R
                    IADD=IUPT(II+1)+II+1
                    DD1(IADD)=DD1(IADD)+DDE*DELY*DELY+DE*(1.0-DELY*DELY)/R
                    IADD=IUPT(II+2)+II+2
                    DD1(IADD)=DD1(IADD)+DDE*DELZ*DELZ+DE*(1.0-DELZ*DELZ)/R
                    IADD=IUPT(II)+II+1
                    DD1(IADD)=DD1(IADD)+DDE*DELX*DELY-DE*DELX*DELY/R
                    IADD=IUPT(II)+II+2
                    DD1(IADD)=DD1(IADD)+DDE*DELX*DELZ-DE*DELX*DELZ/R
                    IADD=IUPT(II+1)+II+2
                    DD1(IADD)=DD1(IADD)+DDE*DELY*DELZ-DE*DELY*DELZ/R

#if KEY_DIMB==1
                 ENDIF
#endif /*  DIMB*/

              ENDIF
           ENDIF
           !
        ENDDO
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE BNDRY

SUBROUTINE ESPLIN(NX,X,Y,IC,C,U,E,DE,DDE)
  !-----------------------------------------------------------------------
  !     Cubic spline function, first and second derivative evaluator:
  !      E  = ( ( C(J,3)*D+C(J,2) )*D+C(J,1) )*D + Y(J)
  !      DE = ( 3.0*C(J,3)*D+2.0*C(J,2) )*D + C(J,1)
  !      DDE= 6.0*C(J,3)*D + 2.0*C(J,2)
  !     where X(J) .le. U .lt. X(J+1) and D = U-X(J).
  !     If U .gt. X(NX) a quadratic extrapolation is being performed
  !
  !      Axel Brunger, Cambridge, 5-AUG-83.
  !      ---------------------------------
  !
  use chm_kinds
  implicit none
  INTEGER NX
  real(chm_real)  X(NX), Y(NX)
  INTEGER IC
  real(chm_real)  C(IC,3), U, E, DE, DDE
  !     local
  INTEGER ISPLIN, NXM1
  real(chm_real)  D, CEXTR0, CEXTR1, CEXTR2, DXNXM1, SPP
  !
  !     begin
  IF (U.LT.X(1)) THEN
     CALL WRNDIE(-2,'<ESPLIN>', &
          'No extrapolation on left side allowed')
  ELSE IF (U.GT.X(NX)) THEN
     !
     !     do quadratic extrapolation on right side
     NXM1=NX-1
     DXNXM1=X(NX)-X(NXM1)
     SPP=(-3.0*C(NXM1,3)*DXNXM1+C(NXM1,2))
     CEXTR2=SPP+SPP
     CEXTR1=(SPP+C(NXM1,2))*DXNXM1+C(NXM1,1)-CEXTR2*X(NX)
     CEXTR0=Y(NX)-(0.5*CEXTR2*X(NX)+CEXTR1)*X(NX)
     E=(0.5*CEXTR2*U+CEXTR1)*U+CEXTR0
     DE=CEXTR2*U+CEXTR1
     DDE=CEXTR2
  ELSE
     !
     !     do spline interpolation
     ISPLIN=2
     IF (U.GT.X(ISPLIN)) THEN
10      CONTINUE
        ISPLIN=ISPLIN+1
        IF (U.GT.X(ISPLIN)) GOTO 10
     ENDIF
     ISPLIN=ISPLIN-1
     D=U-X(ISPLIN)
     E=((C(ISPLIN,3)*D+C(ISPLIN,2))*D+C(ISPLIN,1))*D+Y(ISPLIN)
     SPP=3.0*C(ISPLIN,3)*D+C(ISPLIN,2)
     DE=(SPP+C(ISPLIN,2))*D+C(ISPLIN,1)
     DDE=SPP+SPP
     !
  ENDIF
  !
  RETURN
END SUBROUTINE ESPLIN

SUBROUTINE SPLINR(X,Y,NX,C,IC,IERROR)
  !-----------------------------------------------------------------------
  !     This routine computes the cubic spline coefficients for an ordered
  !     set of X's and Y's, Y=F(X).  It is modeled after the IMSL routine
  !     ICSCCU.
  !     This FLECS version was created by Charles L. Brooks III.
  !
  !     The arguments are;
  !      X      - Vector of length NX holding the ordered abscissae.
  !               Note that X(I) .LT. X(I+1), I=1,NX-1
  !      Y      - vector of length NX containing the function values.
  !      NX     - Number of elements in X and Y.  NX must be .ge. 2.
  !      C      - Matrix of cubic spline coefficients.  The coefficients are
  !               stored such the approximation for Y at T is
  !               S(T) = ((C(I,3)*D+C(I,2))*D+C(I,1))*D+Y(I)
  !               where X(I) .LE. T .LT. X(I+1) and D = T-X(I).
  !      IC     - Row dimension of the matrix C as specified in the
  !               dimension statement of the calling routine
  !               C(IC,3), with IC .ge. NX-1, the maximum dimension of C.
  !      IERROR - Error parameter.
  !
  !     Terminal error indicators
  !      IERROR = 129, IC .lt. NX-1.
  !      IERROR = 130, NX .lt. 2.
  !      IERROR = 131, input abscissa are not ordered
  !-----------------------------------------------------------------------
  use chm_kinds
  implicit none
  !
  !     SPECIFICATIONS FOR ARGUMENTS
  INTEGER  NX,IC,IERROR
  real(chm_real)   X(NX),Y(NX),C(IC,3)
  !     LOCAL VARIABLES
  INTEGER  IM1,I,JJ,J,MM1,MP1,M,NM1,NM2
  real(chm_real)   DIV1,DIV3,DT,G,CNX(3)
  !     BEGIN
  NM1=NX-1
  IERROR=129
  IF (IC .LT. NM1) GOTO 1000
  IERROR=130
  IF (NX .LT. 2) GOTO 1000
  IERROR=131
  IF (NX .EQ. 2) GOTO 45
  !
  do i=1,ic
     do j=1,3
        c(i,j) = 0.0
     enddo
  enddo

  DO M=2, NM1
     MM1=M-1
     C(M,2)=X(M)-X(MM1)
     IF (C(M,2).LE.0.0) GOTO 1000
     C(M,3)=(Y(M)-Y(MM1))/C(M,2)
  ENDDO
  CNX(2)=X(NX)-X(NM1)
  IF (CNX(2).LE.0.0) GOTO 1000
  CNX(3)=(Y(NX)-Y(NM1))/CNX(2)
  IERROR=0
  NM2=NX-2
  IF (NX .GT. 3) GOTO 10
  C(1,3)=CNX(2)
  C(1,2)=C(2,2)+CNX(2)
  C(1,1)=((C(2,2)+2.0*C(1,2))*C(2,3)*CNX(2)+C(2,2)**2*CNX(3)) &
       /C(1,2)
  GOTO 20
10 C(1,3)=C(3,2)
  C(1,2)=C(2,2)+C(3,2)
  C(1,1)=((C(2,2)+2.0*C(1,2))*C(2,3)*C(3,2)+C(2,2)**2*C(3,3)) &
       /C(1,2)
  DO M=2, NM2
     MP1=M+1
     MM1=M-1
     G=-C(MP1,2)/C(MM1,3)
     C(M,1)=G*C(MM1,1)+3.0*C(M,2)*C(MP1,3)+3.0*C(MP1,2)*C(M,3)
     C(M,3)=G*C(MM1,2)+2.0*C(M,2)+2.0*C(MP1,2)
  ENDDO
20 G=-CNX(2)/C(NM2,3)
  C(NM1,1)=G*C(NM2,1)+3.0*C(NM1,2)*CNX(3)+3.0*CNX(2)*C(NM1,3)
  C(NM1,3)=G*C(NM2,2)+2.0*C(NM1,2)+2.0*CNX(2)
  IF (NX.GT.3) GOTO 25
  CNX(1)=2.0*CNX(3)
  CNX(3)=1.0
  G=-1.0/C(NM1,3)
  GOTO 30
25 G=C(NM1,2)+CNX(2)
  CNX(1)=((CNX(2)+2.0*G)*CNX(3)*C(NM1,2)+CNX(2)**2* &
       (Y(NM1)-Y(NX-2))/C(NM1,2))/G
  G=-G/C(NM1,3)
  CNX(3)=C(NM1,2)
30 CNX(3)=G*C(NM1,2)+CNX(3)
  CNX(1)=(G*C(NM1,1)+CNX(1))/CNX(3)
  C(NM1,1)=(C(NM1,1)-C(NM1,2)*CNX(1))/C(NM1,3)
  DO JJ=1, NM2
     J=NM1-JJ
     C(J,1)=(C(J,1)-C(J,2)*C(J+1,1))/C(J,3)
  ENDDO
  DO I=2, NM1
     IM1=I-1
     DT=C(I,2)
     DIV1=(Y(I)-Y(IM1))/DT
     DIV3=C(IM1,1)+C(I,1)-2.0*DIV1
     C(IM1,2)=(DIV1-C(IM1,1)-DIV3)/DT
     C(IM1,3)=DIV3/DT**2
  ENDDO
  DT=CNX(2)
  DIV1=(Y(NX)-Y(NM1))/DT
  DIV3=C(NM1,1)+CNX(1)-2.0*DIV1
  C(NM1,2)=(DIV1-C(NM1,1)-DIV3)/DT
  C(NM1,3)=DIV3/DT**2
  GOTO 1000
45 IF (X(1) .GE. X(2)) GOTO 1000
  IERROR=0
  C(1,1)=(Y(2)-Y(1))/(X(2)-X(1))
  C(1,2)=0.0
  C(1,3)=0.0
1000 CONTINUE
#endif 
  RETURN
END SUBROUTINE SPLINR

