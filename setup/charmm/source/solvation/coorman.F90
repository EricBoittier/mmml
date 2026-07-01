#if KEY_RISM==0
SUBROUTINE NULL_MKOM
  RETURN
END SUBROUTINE NULL_MKOM
#else /**/
SUBROUTINE MKOMEG(X,Y,Z,SEGID,NSITE,INTRA,DIM,OMEGK,OMEGKI,JOB)
  !-----------------------------------------------------------------------
  !     This subroutine calculates the intramolecular omega matrix
  !     and its inverse in Fourier k-space.

  use chm_kinds
  use number
  use rism
  use fft
  use stream
  implicit none
  real(chm_real) X(*),Y(*),Z(*)
  CHARACTER(len=*) SEGID(*)
  INTEGER NSITE

  !     Pointer array
  INTEGER DIM
  INTEGER INTRA(DIM,DIM)

  !     Intramolecular functions
  real(chm_real) OMEGK(DVECT,*),OMEGKI(DVECT,*),DIJ
  CHARACTER(len=5) JOB

  !     Matrix to invert
  real(chm_real) RM(DSITE,DSITE),RMI(DSITE,DSITE)
  INTEGER IK,I,J

  WRITE(OUTU,100)
100 FORMAT(1X,'Making the intramolecular correlation matrix')

  DO IK=NFIRST,NPOINT
     DO I=1,NSITE
        ! 
        ! Set the diagonal elements to 1.0
        !
        OMEGK(IK,INTRA(I,I))=ONE
        DO J=1,I-1
           !
           ! Check if the sites belong to different molecules
           !
           IF(SEGID(I).NE.SEGID(J))THEN
              OMEGK(IK,INTRA(I,J))=ZERO
              !
              ! or to the same molecule
              !
           ELSE 
              DIJ=SQRT((X(I)-X(J))**2+(Y(I)-Y(J))**2+(Z(I)-Z(J))**2)
              OMEGK(IK,INTRA(I,J))=SIN(RK(IK)*DIJ)/(RK(IK)*DIJ)
           ENDIF
        ENDDO
     ENDDO
  ENDDO

  IF(JOB.EQ.'INVRS') THEN
     WRITE(OUTU,101)
101  FORMAT(1X,'The inverse is being calculated',/)
     DO IK=NFIRST,NPOINT
        DO I=1,NSITE
           DO J=1,NSITE
              RM(I,J)=OMEGK(IK,INTRA(I,J))
           ENDDO
        ENDDO
        CALL INVRS(RM,RMI,NSITE)
        DO I=1,NSITE
           DO J=1,I
              OMEGKI(IK,INTRA(I,J))=RMI(I,J)
           ENDDO
        ENDDO
     ENDDO
  ENDIF

  RETURN
END SUBROUTINE MKOMEG

SUBROUTINE EDTZM
  !-----------------------------------------------------------------------
  !     This subroutine edits the Z-matrix according to the instructions of
  !     the input file
  use chm_kinds
  use dimens_fcm
  use comand
  use string
  use rism
  use struc
  use distri
  implicit none

  INTEGER IU,IOFF

  IF(CHECQUE(COMLYN,'SOLVENT')) THEN
     CALL EDTZM2(IZMAT,ZMAT,X,Y,Z,NSITV)

  ELSEIF(CHECQUE(COMLYN,'SOLUTE'))THEN
     IU=1
     IF(DOTRIM(COMLYN)) IU=NEXTI(COMLYN,COMLEN)
     IF(IU.EQ.0) RETURN
     IOFF=DSITV+(IU-1)*DSITU+1
     CALL EDTZM2(IZMAT(1,1,IU+1),ZMAT(1,1,IU+1), &
          X(IOFF),Y(IOFF),Z(IOFF),NSITU(IU))

  ELSE
     CALL EDTZM2(IZMAT,ZMAT,X,Y,Z,NSITV)

  ENDIF
  RETURN
END SUBROUTINE EDTZM

SUBROUTINE EDTZM2(IZMAT,ZMAT,X,Y,Z,NSITE)
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use comand
  use stream
  use string
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) ZMAT(3,*)
  INTEGER IZMAT(4,*),NSITE

  LOGICAL EOF
  INTEGER IENTRY
  real(chm_real)  BOND,THETA,PHI
  !
  !     initialize logical
  EOF = .FALSE.

1000 CONTINUE
  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE.,.TRUE., &
       'EDTZM> ')
  IF(EOF) GOTO 9000
  IF(COMLEN.LE.0) GOTO 1000
  IF(CHECQUE(COMLYN,'END'))THEN
     CALL ZCONSTR(.TRUE.,1,NSITE,IZMAT,ZMAT,X,Y,Z)
     RETURN
  ENDIF

  !     Read the Z matrix modification <entry> bond theta phi
  !
  IENTRY=GTRMI(COMLYN,COMLEN,'ENTR',-1)
  IF((IENTRY.EQ.-1).OR.(IENTRY.GT.NSITE)) RETURN
  BOND =GTRMF(COMLYN,COMLEN,'BOND', ZMAT(1,IENTRY))
  THETA=GTRMF(COMLYN,COMLEN,'THETA',ZMAT(2,IENTRY))
  PHI  =GTRMF(COMLYN,COMLEN,'PHI',  ZMAT(3,IENTRY))
  ZMAT(1,IENTRY)=BOND
  ZMAT(2,IENTRY)=THETA
  ZMAT(3,IENTRY)=PHI
  GOTO 1000
  RETURN
  !
9000 CONTINUE
  WRITE (OUTU,'(A)') ' RISM:EDTZM2> ERROR: End-of-File reached.'
  RETURN
END SUBROUTINE EDTZM2
#endif 


