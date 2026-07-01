SUBROUTINE NULL_DIMBD
  RETURN
END SUBROUTINE NULL_DIMBD

#if KEY_DIMB==1 /*dimbcray_main*/
#if KEY_EISPACK==1 /*eispack_main*/

SUBROUTINE DIASCI(NDIM,MAT,EIGVA,EIGVE,FV1,FV2)
  !

  !-----------------------------------------------------------------------
  !
  !     ++++  THE SCILIB LIBRARY IS REQUIRED FOR CRAY +++++++
  !
  !     07-Feb-1994 David Perahia
  !     15-Dec-1994 Herman van Vlijmen
  !
  !     Diagonalize the real symmetric matrix MAT.
  !     MAT contains only the lower triangle elements.
  !     NDIM = order of the matrix MAT
  !     EIGVA = eigenvalues
  !     EIGVE = eigenvectors
  !     FV1,FV2 = working arrays
  !     MATZ=0 -> only eigenvalues
  !       ELSE -> eigenvalues and eigenvectors
  !     DIMENSIONS : FV1(NDIM),FV2(NDIM)
  !-----------------------------------------------------------------------
  use chm_kinds
  use stream
  implicit none
  ! Passed variables
  INTEGER NDIM
  real(chm_real) MAT(*),EIGVA(*),EIGVE(NDIM,*),FV1,FV2
  ! Local variables
  INTEGER IERR,MATZ,NM,NV

  ! Begin
  IF(PRNLEV >= 2) WRITE(OUTU,103)
103 FORMAT(/' DIASCI: Subroutine RSP of EISPACK is used for', &
       ' diagonalization')

  NM=NDIM
  NV=NDIM*(NDIM+1)/2
  MATZ=1
  CALL RSP(NM,NDIM,NV,MAT,EIGVA,MATZ,EIGVE,FV1,FV2,IERR)
  IF(PRNLEV >= 2) WRITE(OUTU,105)
105 FORMAT(' DIASCI: Subroutine RSP completed')

  IF(IERR > 0) THEN
     CALL WRNDIE(0,'<DIASCI>','Error in RSP')
     IF(IERR == 10*NDIM) THEN
        CALL WRNDIE(0,'<DIASCI>','N is greater than NM')
     ELSEIF(IERR == 20*NDIM) THEN
        CALL WRNDIE(0,'<DIASCI>','N is less than N*(N+1)/2')
     ELSEIF(IERR <= NDIM) THEN
        WRITE(OUTU,113) IERR
        CALL WRNDIE(0,'<DIASCI>','Wrong eigenvalues')
     ENDIF
  ENDIF
113 FORMAT(/' Warning from DIASCI: The ',I5,'th eigenvalue and', &
       ' beyond are wrong')

  RETURN
END SUBROUTINE DIASCI

!
SUBROUTINE DIASCR(NDIM,M,MAT,EIGVA,EIGVE,D,E,E2,BD,IND, &
     FV1,FV2,FV3,FV4,FV6)
  !-----------------------------------------------------------------------
  !     07-Feb-1994 David Perahia
  !     15-Dec-1994 Herman van Vlijmen
  !
  !     Diagonalize the real symmetric matrix MAT.
  !     MAT contains only the lower triangle elements.
  !     Only the lowest eigenvalues and eigenvectors are produced.
  !     NDIM = order of the matrix MAT
  !     EIGVA = eigenvalues
  !     EIGVE = eigenvectors
  !     FV1,FV2,FV3,FV4,FV6 = working arrays
  !     DIMENSIONS : FV1(NDIM),FV2(NDIM)
  !     D = diagonal elements of the tridiagonal matrix
  !     E = subdiagonal elements of the tridiagonal matrix
  !     E2 = squares of the corresponding elements of E
  !     EPS1 = theoretical absolute error tolerance for the
  !            computed eigenvalues
  !     M = number of lowest eigenvalues and eigenvectors to be found
  !-----------------------------------------------------------------------
  use chm_kinds
  use stream
  implicit none
  ! Passed variables
  INTEGER NDIM,M,IND(*)
  real(chm_real) MAT(*),EIGVA(*),EIGVE(NDIM,*)
  real(chm_real) FV1(*),FV2(*),FV3(*),FV4(*),FV6(*)
  real(chm_real) D(*),E(*),E2(*),BD(*)
  ! Local variables
  real(chm_real) EPS1
  INTEGER IERR,MATZ,IDEF,KERR,NM,NV

  ! Begin
  IF(PRNLEV >= 2) WRITE(OUTU,103)
103 FORMAT(/' DIASCR: Subroutine RATQR of EISPACK is used for', &
       ' diagonalization')

  NM=NDIM
  NV=NDIM*(NDIM+1)/2
  MATZ=1
  EPS1=-1.0
  IDEF=0
  CALL TRED3(NM,NV,MAT,D,E,E2)
  CALL RATQR(NM,EPS1,D,E,E2,M,EIGVA,IND,BD,.TRUE.,IDEF,IERR)
  IF(IERR > 0) THEN
     CALL WRNDIE(0,'<DIASCR>','Error in RATQR')
     IF(IERR <= 6*NDIM) THEN
        KERR=IERR-5*NDIM
        WRITE(OUTU,113) KERR
        CALL WRNDIE(0,'<DIASCR>','Wrong eigenvalue')
     ENDIF
  ENDIF
113 FORMAT(/' Warning from DIASCR: The ',I5,'th eigenvalue is not', &
       ' evaluated correctly')

  E2(1)=0.D0
  CALL TINVIT(NM,NDIM,D,E,E2,M,EIGVA,IND,EIGVE,IERR, &
       FV1,FV2,FV3,FV4,FV6)
  IF(IERR /= 0) THEN
     CALL WRNDIE(0,'<DIASCR>','Error in TINVIT')
     KERR=-IERR
     WRITE(OUTU,115) KERR
     CALL WRNDIE(0,'<DIASCR>','Wrong eigenvalue')
  ENDIF
115 FORMAT(/' Warning from DIASCR:',I5,'th eigenvalue fails to', &
       ' converge in 5 iterations')

  CALL TRBAK3(NM,NDIM,NV,MAT,M,EIGVE)
  IF(PRNLEV >= 2) WRITE(OUTU,117)
117 FORMAT(' DIASCI: Diagonalization with RATQR completed')

  RETURN
END SUBROUTINE DIASCR

!
SUBROUTINE GENLSC(DU,DL,NDIM)
  !-----------------------------------------------------------------------
  !     30-Dec-1993 David Perahia
  !     15-Dec-1994 Herman van Vlijmen
  !
  !     This routine generates the lower triangular matrix DL 
  !     from the upper triangular matrix DU.
  !     NDIM is the order of the matrix
  !-----------------------------------------------------------------------
  use chm_kinds
  implicit none
  ! Passed variables
  INTEGER NDIM
  real(chm_real) DU(*),DL(*)
  ! Local variables
  INTEGER KU,KL,I,J

  ! Begin
  KL=0
  DO I=1,NDIM
     KU=I
     DO J=1,I
        KL=KL+1
        DL(KL)=DU(KU)
        KU=KU+NDIM-J
     ENDDO
  ENDDO
  !
  return
end SUBROUTINE GENLSC
#endif /* (eispack_main)*/
#endif /* (dimbcray_main)*/

