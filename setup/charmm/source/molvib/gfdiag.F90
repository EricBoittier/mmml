#if KEY_MOLVIB==1
SUBROUTINE GFDIAG(NAT,NAT3,NQ,NQ2,NQQ,IPRNT,IFDD,JCONT, &
     NQM,G,W1,W2,LS,FS,DD,V1,CALFRQ,IO)
  !
  !  Diagonalize GF matrix 
  !  two-stage procedure : first diagonalize G=W*GD*WT
  !  then transform F by [G+1/2]=W*[SQRT(GD)]*WT
  !  to F'= [G+1/2]*F*[G+1/2] and diagonalize again
  !
  !  G  - the Wilson G matrix in S coords; (NQ,NQ)
  !  F - force constant matrix in S coords; (NQ,NQ)
  !  W1 -  work array for G eigenvectors
  !  LS - vibrational eigenvectors in S coords G*F*L=L*DD; (NQ,NQ)
  !  DD - eigenvalues
  !  CALFRQ - vibrational frequencies in cm-1
  !
  !  dimensions:
  !  NAT - number of atoms,  NAT3= 3*NAT
  !  NQ  - number of vibrational degrees of freedom
  !  NQ2=NQ*(NQ+1)/2 - symmetric array storage (unused)
  !  NQQ=NQ*NQ
  !  NQM - physical dimension of arrays in calling subroutine
  !  FACT - conversion factor to cm-1; for frequencies
  !
  use chm_kinds
  implicit none
  INTEGER NAT,NAT3,NQ,NQ2,NQQ,IPRNT,IFDD,NQM,IO
  real(chm_real) G(NQM,NQM),W1(NQM,NQM),LS(NQM,NQM),DD(NQM)
  real(chm_real) W2(NQM,NQM),FS(NQM,NQM),V1(NQM),CALFRQ(NQM)
  !
  CHARACTER(len=4) JCONT
  INTEGER I,J,K,L,NNEG
  ! ... Fact is the conversion factor from SQRT(eigenvalue) to cm-1
  real(chm_real) :: FACT=1302.80_chm_real, S
  real(chm_real) ESIGN
  !
  WRITE(IO,*) 
  WRITE(IO,*) '      Start GF diagonalization routine'
  WRITE(IO,*) 
  !
  ! ... Load G into W1 and diagonalize; eigenvectors will overwrite W1
  !
  DO I=1,NQ
     DO J=1,NQ
        W1(I,J)=G(I,J)
     ENDDO
  ENDDO
  !
  CALL HSHLDR(W1,DD,V1,NQ,NQM)
  !

  IF(JCONT.EQ.'G   ' .OR. IPRNT.GT.4) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' G matrix eigenvectors I,(W1(J,I),J=1,NQ)'
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,1005) I,(W1(J,I),J=1,NQ)
     ENDDO
  ENDIF
  !
  ! ... Print G eigenvalues
  !
  WRITE(IO,*)
  WRITE(IO,*) ' G eigenvalues '
  WRITE(IO,*)
  WRITE(IO,1000) (DD(J),J=1,NQ)
1000 FORMAT(1X,8F9.6,12(/1X,8F9.6))
  ! ...testng segment
  IF(IPRNT.GT.4) THEN
     DO I=1,NQ
        DO J=1,NQ
           S=0.0
           DO K=1,NQ
              DO L=1,NQ
                 S = S + G(K,L)*W1(K,I)*W1(L,J)
              ENDDO
           ENDDO
           IF(I.NE.J.AND.ABS(S).GT.0.0001) WRITE(IO,*)'TEST: S too big'
           LS(I,J)=S 
        ENDDO
     ENDDO
     WRITE(IO,*)
     WRITE(IO,*) ' Test of G diagonalization: the WT*G*W matrix'
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,1005) I,(LS(I,J),J=1,NQ)
     ENDDO
  ENDIF
  ! ...end testing segment
  !
  ! ... Finished if G diagonalization only
  !
  IF(IFDD.EQ.0) RETURN
  !
  ! ... now calculate square root of G matrix = [G+1/2]
  !
  NNEG=0
  DO I=1,NQ
     IF(DD(I).LT.1.0D-16) NNEG=NNEG+1
     DD(I)=SQRT(ABS(DD(I)))
  ENDDO
  IF(NNEG.NE.0) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' **** G matrix is not positive definite '
     WRITE(IO,*) ' ****   Redefine internal coordinates '
     WRITE(IO,*)
     CALL WRNDIE(-4,'<GFDIAG>',' SO LONG')
  ENDIF
  DO I=1,NQ
     DO J=1,NQ
        S=0.0
        DO K=1,NQ
           S = S + DD(K)*W1(I,K)*W1(J,K)
        ENDDO
        W2(I,J)=S
     ENDDO
  ENDDO

  IF(IPRNT.GT.4) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' Square root of G matrix'
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,1005) I,(W2(I,J),J=1,NQ)
     ENDDO
  ENDIF

  ! ... Transform F by G1/2
  !
  DO I=1,NQ
     DO J=1,NQ
        S=0.0
        DO K=1,NQ
           DO L=1,NQ
              S = S + FS(K,L)*W2(I,K)*W2(L,J)
           ENDDO
        ENDDO
        W1(I,J)=S
     ENDDO
  ENDDO
  !
  IF(IPRNT.GT.4) THEN
     WRITE(IO,*)
     WRITE(IO,*) 'Transformed F matrix'
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,1005) I,(W1(I,J),J=1,NQ)
     ENDDO
  ENDIF
  !
  ! ... Diagonalize transformed F loaded into W1
  !
  CALL HSHLDR(W1,DD,V1,NQ,NQM)
  !
  ! ... Now transform eigenvectors back to S coordinates
  ! ... and eigenvalues to cm-1
  !
  IF(IPRNT.GT.4) THEN
     WRITE(IO,*)
     WRITE(IO,*) 'Eigenvectors of F'
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,1005) I,(W1(I,J),J=1,NQ)
     ENDDO
  ENDIF
  !
  DO I=1,NQ
     DO J=1,NQ
        S=0.0
        DO K=1,NQ
           S = S + W2(I,K)*W1(K,J)
        ENDDO
        LS(I,J)=S
     ENDDO
  ENDDO
  NNEG=0
  DO I=1,NQ
     ESIGN=1.0D0
     IF(DD(I).LT.0.0D0) THEN
        NNEG=NNEG+1
        ESIGN=-1.0D0
     ENDIF
     CALFRQ(I) = ESIGN*FACT*SQRT(ABS(DD(I)))
  ENDDO
  !
  ! ... Output
  !
  IF(IPRNT.GE.1) THEN
     WRITE(IO,*)
     WRITE(IO,*) 'Frequencies and eigenvectors in internal coords'
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,1010) I,CALFRQ(I),(LS(J,I),J=1,NQ)
     ENDDO
  ENDIF
1005 FORMAT(1X,I3,8F9.4,12(/4X,8F9.4))
1010 FORMAT(1X,I3,F10.1,10F9.4,12(/14X,10F9.4))
  !
  IF(NNEG.NE.0) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' **** Warning : ',NNEG,' negative eigenvalues' 
     WRITE(IO,*)
  ENDIF
  !
  RETURN 
END SUBROUTINE GFDIAG

SUBROUTINE GFDIAK(NAT,NAT3,NQ,NQ2,NQQ,IPRNT,IFDD,JCONT, &
     NQM,G,FS,LS,W1,W2,V1,DD,CALFRQ,IO)
  !
  !  Diagonalize GF matrix 
  !  two-stage procedure : first diagonalize G=W*GD*WT
  !  then transform F by [G+1/2]=W*[SQRT(GD)]*WT
  !  to F'= [G+1/2]*F*[G+1/2] and diagonalize again
  !  This is essentially the same as GFDIAG. The difference is that
  !  this routine will perform a generalized sqare root of G,
  !  ignoring zero eigenvalues
  !
  !  G  - the Wilson G matrix in S coords; (NQ,NQ)
  !  F - force constant matrix in S coords; (NQ,NQ)
  !  W1 -  work array for G eigenvectors
  !  LS - vibrational eigenvectors in S coords G*F*L=L*DD; (NQ,NQ)
  !  DD - eigenvalues
  !  CALFRQ - vibrational frequencies in cm-1
  !
  !  dimensions:
  !  NAT - number of atoms,  NAT3= 3*NAT
  !  NQ  - number of vibrational degrees of freedom
  !  NQ2=NQ*(NQ+1)/2 - symmetric array storage (unused)
  !  NQQ=NQ*NQ
  !  NQM - physical dimension of arrays in calling subroutine
  !  FACT - conversion factor to cm-1; for frequencies
  !
  use chm_kinds
  implicit none
  INTEGER NAT,NAT3,NQ,NQ2,NQQ,IPRNT,IFDD,NQM,IO
  real(chm_real) G(NQM,NQM),FS(NQM,NQM),LS(NQM,NQM),W1(NQM,NQM)
  real(chm_real) W2(NQM,NQM),DD(NQM),V1(NQM),CALFRQ(NQM)
  INTEGER I,J,K,L,NNEG
  !
  CHARACTER(len=4) JCONT
  ! ... Fact is the conversion factor from SQRT(eigenvalue) to cm-1
  ! ... Also occurs in FSGL.FOR
  real(chm_real) ESIGN
  real(chm_real) :: FACT=1302.80_chm_real, S
  !
  WRITE(IO,*) 
  WRITE(IO,*) '      Start GF diagonalization routine'
  WRITE(IO,*) 
  !
  ! ... Load G into W1 and diagonalize; eigenvalues will overwrite W1
  !
  DO I=1,NQ
     DO J=1,NQ
        W1(I,J)=G(I,J)
     ENDDO
  ENDDO
  !
  CALL HSHLDR(W1,DD,V1,NQ,NQM)
  !
  IF(JCONT.EQ.'G   ' .OR. IPRNT.GT.4) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' G matrix eigenvectors I,(W1(J,I),J=1,NQ)'
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,1005) I,(W1(J,I),J=1,NQ)
     ENDDO
  ENDIF
  !
  ! ... Test
  !
  WRITE(IO,*)
  WRITE(IO,*) ' G eigenvalues '
  WRITE(IO,*)
  WRITE(IO,1000) (DD(J),J=1,NQ)
1000 FORMAT(1X,8F9.6,12(/1X,8F9.6))
  !
  ! ... Finished if G diagonalization only
  !
  IF(IFDD.EQ.0) RETURN
  !
  ! ... now calculate square root of G matrix = [G+1/2]
  ! ... zero eigenvalues allowed here
  !
  NNEG=0
  DO I=1,NQ
     IF(DD(I).LT.1.0D-16) THEN
        NNEG=NNEG+1
        DD(I)=0.0D0
     ENDIF
     DD(I)=SQRT(ABS(DD(I)))
  ENDDO
  IF(NNEG.NE.0) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' **** GFDIAK: G is not positive definite ****'
     WRITE(IO,*) ' Found ',NNEG,' nonpositive eigenvalues'
     WRITE(IO,*)
  ENDIF
  DO I=1,NQ
     DO J=1,NQ
        S=0.0
        DO K=1,NQ
           S = S + DD(K)*W1(I,K)*W1(J,K)
        ENDDO
        W2(I,J)=S
     ENDDO
  ENDDO

  IF(IPRNT.GT.4) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' Square root of G matrix'
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,1005) I,(W2(I,J),J=1,NQ)
     ENDDO
  ENDIF

  ! ... Transform F
  !
  DO I=1,NQ
     DO J=1,NQ
        S=0.0
        DO K=1,NQ
           DO L=1,NQ
              S = S + FS(K,L)*W2(I,K)*W2(L,J)
           ENDDO
        ENDDO
        W1(I,J)=S
     ENDDO
  ENDDO
  !
  IF(IPRNT.GT.4) THEN
     WRITE(IO,*)
     WRITE(IO,*) 'Transformed F matrix'
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,1005) I,(W1(I,J),J=1,NQ)
     ENDDO
  ENDIF
  !
  ! ... Diagonalize transformed F loaded into W1
  !
  CALL HSHLDR(W1,DD,V1,NQ,NQM)
  !
  ! ... Now transform eigenvectors back to S coordinates
  ! ... and eigenvalues to cm-1
  !
  IF(IPRNT.GT.4) THEN
     WRITE(IO,*)
     WRITE(IO,*) 'Eigenvectors of F'
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,1005) I,(W1(I,J),J=1,NQ)
     ENDDO
  ENDIF
  !
  DO I=1,NQ
     DO J=1,NQ
        S=0.0
        DO K=1,NQ
           S = S + W2(I,K)*W1(K,J)
        ENDDO
        LS(I,J)=S
     ENDDO
  ENDDO
  NNEG=0
  DO I=1,NQ
     ESIGN=1.0D0
     IF(DD(I).LT.0.0D0) THEN
        NNEG=NNEG+1
        ESIGN=-1.0D0
     ENDIF
     CALFRQ(I) = ESIGN*FACT*SQRT(ABS(DD(I)))
  ENDDO
  !
  ! ... Output
  !
  IF(IPRNT.GE.1) THEN
     WRITE(IO,*)
     WRITE(IO,*) 'Frequencies and eigenvectors in internal coords'
     WRITE(IO,*)
     DO I=1,NQ
        WRITE(IO,1010) I,CALFRQ(I),(LS(J,I),J=1,NQ)
     ENDDO
  ENDIF
1005 FORMAT(1X,I3,8F9.6,12(/4X,8F9.6))
1010 FORMAT(1X,I3,F10.1,10F9.4,12(/14X,10F9.4))
  !
  IF(NNEG.NE.0) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' **** Warning : ',NNEG,' negative eigenvalues' 
     WRITE(IO,*)
  ENDIF
  !
  RETURN 
END SUBROUTINE GFDIAK

SUBROUTINE GFX(NAT,NAT3,NQ,NQ2,NQQ,IPRNT,IFDD,JCONT, &
     NQM,FX,LX,W1,LS,DD,V1,AMASS,CALFRQ,IO)
  !
  !  Vibrational problem in cartesian coordinates
  !
  !  FX - force constant matrix in X coords; (NAt3,NAT3)
  !  AMASS - array of atomic masses
  !  W1 -  work array 
  !  LX - vibrational eigenvectors in X coords (NAT3,NQ)
  !  DD - eigenvalues
  !  CALFRQ - vibrational frequencies in cm-1
  !
  !  dimensions:
  !  NAT - number of atoms,  NAT3= 3*NAT
  !  NQ  - number of vibrational degrees of freedom
  !  NQ2=NQ*(NQ+1)/2 - symmetric array storage (unused)
  !  NQQ=NQ*NQ
  !  
  !  FACT - conversion factor to cm-1; for frequencies
  !
  !  NB: look out for dimension, must be NAT3.LE.NQM
  !
  use chm_kinds
  implicit none
  INTEGER NQM,NAT,NAT3,NQ,NQ2,NQQ,IPRNT,IFDD,NNEG,IO
  real(chm_real) FX(NQM,NQM),LX(NQM,NQM),W1(NQM,NQM),LS(NQM,NQM)
  real(chm_real) DD(NQM),V1(NQM),AMASS(NAT3),CALFRQ(NQM)
  !
  INTEGER I,J,K,L
  CHARACTER(len=4) JCONT
  real(chm_real) ESIGN
  real(chm_real) :: FACT=1302.80_chm_real, S
  !
  WRITE(IO,*)
  WRITE(IO,*) '      Starting FX diagonalization '
  WRITE(IO,*)
  !
  ! ... Scale FX by square roots of masses - place it in W1
  !
  DO I=1,NAT3
     DO J=1,NAT3
        W1(I,J) = FX(I,J)/SQRT(AMASS(I)*AMASS(J))
     ENDDO
  ENDDO

  IF(IPRNT.GT.4) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' Mass-scaled FX matrix'
     WRITE(IO,*)
     DO I=1,NAT3
        WRITE(IO,1005) I,(W1(I,J),J=1,NAT3)
     ENDDO
  ENDIF

  !
  ! ... Diagonalize the transformed F loaded into W1
  !
  CALL HSHLDR(W1,DD,V1,NAT3,NQM)
  !
  ! ... Test
  !
  IF(IPRNT.GT.4) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' Test of FX diagonalization: eigenvalues and', &
          ' the WT*FX*W matrix'
     WRITE(IO,*)
     WRITE(IO,1000) (DD(J),J=1,NAT3)
     WRITE(IO,*)
     DO I=1,NAT3
        DO J=1,NAT3
           S=0.0
           DO K=1,NAT3
              DO L=1,NAT3
                 S = S + FX(K,L)*W1(K,I)*W1(L,J)
              ENDDO
           ENDDO
           IF(I.NE.J.AND.ABS(S).GT.0.0001) WRITE(IO,*)'TEST: S too big'
           LS(I,J)=S 
        ENDDO
     ENDDO
     DO I=1,NAT3
        WRITE(IO,1005) I,(LS(I,J),J=1,NAT3)
     ENDDO
  ENDIF
  ! ...end testing segment


  !
  ! ... Now transform eigenvectors back to X coordinates
  ! ... and eigenvalues to frequencies in cm-1
  !
  IF(IPRNT.GT.4) THEN
     WRITE(IO,*)
     WRITE(IO,*) ' Mass-scaled eigenvectors of FX'
     WRITE(IO,*)
     DO I=1,NAT3
        WRITE(IO,1005) I,(W1(I,J),J=1,NAT3)
     ENDDO
  ENDIF
  !
  DO I=1,NAT3
     DO J=1,NAT3
        LX(I,J) = W1(I,J)/SQRT(AMASS(I))
     ENDDO
  ENDDO
  !
  NNEG=0
  DO I=1,NAT3
     ESIGN=1.0D0
     IF(DD(I).LT.0.0D0) THEN
        NNEG=NNEG+1
        ESIGN=-1.0D0 
     ENDIF
     CALFRQ(I) = ESIGN*FACT*SQRT(ABS(DD(I)))
  ENDDO
  !
  IF(NNEG.GT.0) THEN
     WRITE(IO,*)
     WRITE(IO,*)  ' **** Found ',NNEG,'  negative eigenvalues ****'
     WRITE(IO,*)
  ENDIF
  !
  ! ... Output
  !
  IF(IPRNT.GE.1) THEN
     WRITE(IO,*)
     WRITE(IO,*) 'Frequencies and eigenvectors in cartesian coords'
     WRITE(IO,*)
     DO I=1,NAT3
        WRITE(IO,1010) I,CALFRQ(I),(LX(J,I),J=1,NAT3)
     ENDDO
  ENDIF
1000 FORMAT(1X,8F9.6,12(/1X,8F9.6))
1005 FORMAT(1X,I3,8F9.4,12(/4X,8F9.4))
1010 FORMAT(1X,I3,F10.1,10F9.4,5(/14X,10F9.4))
  !
  RETURN 
END SUBROUTINE GFX
#else /**/
SUBROUTINE NULL_GF
  RETURN 
END SUBROUTINE NULL_GF
#endif /*  MOLVIB*/


