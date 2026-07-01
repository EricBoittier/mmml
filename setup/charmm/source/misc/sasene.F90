#if KEY_SASAE==1
SUBROUTINE SASENE(NAT, &
     NATIM, &
     SASWBL,SASWNB, &
     CHMIBL,CHMINB, &
     IMAIBL,IMAINB, &
     IMATTR, &
     SASIBL,SASINB, &
     SASLCT, &
     SASIDX, &
     SASACS)
  !-----------------------------------------------------------------------
  !     This routine calculates the mean solvation energy and its
  !     derivatives based on the method described in the paper
  !     Hasel, W.; Hendrickson, T. F.; Still, W. C.; A Rapid Approximation
  !     to the Solvent Accessible Surface Areas of Atoms; Tetrahedron
  !     Computer Methodology; Vol. 1; No. 2; 103-116; 1988.
  !        See source/sasa/sasa.f90 for a description of variables.
  !
  !     Author: Urs Haberthuer. The code of this routine is based on a
  !             previous code by Joannis Apostolakis.

  use chm_kinds
  use dimens_fcm
  use number
  use consta
  use coord
  use deriv
  use energym
  use sasa
  implicit none

  INTEGER NAT
  INTEGER NATIM
  INTEGER SASWBL(*),SASWNB(*)
  INTEGER CHMIBL(*),CHMINB(*)
  INTEGER IMAIBL(*),IMAINB(*)
  INTEGER IMATTR(*)
  INTEGER SASIBL(*),SASINB(*)
  INTEGER SASLCT(*)
  INTEGER SASIDX(*)
  real(chm_real)  SASACS(*)


  real(chm_real)  MST
  real(chm_real)  DMSTPS
  real(chm_real)  DMSTPX,DMSTPY,DMSTPZ
  real(chm_real)  BMN,BNM
  real(chm_real)  DBMNP,DBNMP
  real(chm_real)  XMN,YMN,ZMN
  real(chm_real)  DMN1,DMN2
  real(chm_real)  TMN,TNM
  real(chm_real)  T1MN,T2MN
  real(chm_real)  T3MN,T4MN,T5MN,T3NM,T4NM,T5NM
  real(chm_real)  CON
  real(chm_real)  TMP
  INTEGER A,B,C,D,E,F
  INTEGER M,N,U,V
  INTEGER MI,NI
  INTEGER I,J,K

  !     MST - The mean solvation energy.

  MST=ZERO

  !     Set the solvent accessible surface area of each atom that has been
  !     selected by the user for SASA equal to the area of its first
  !     solvation shell. If the atom has not been selected the value 0 is
  !     assigned.

  DO I=1,NAT
     IF (SASLCT(I) .NE. 0) THEN
        SASACS(I)=SASASS(SASIDX(I))
     ELSE
        SASACS(I)=ZERO
     ENDIF
  ENDDO

  !     Calculate the mean solvation energy. Go through all the pairs in
  !     the nonbond pair list and in the SASA pair list. Only pairs where
  !     both atoms have been selected by the user for SASA are taken into
  !     account. All other pairs are neglected.

  DO I=1,NAT
     IF (SASLCT(I) .NE. 0) THEN
        IF (I .EQ. 1) THEN
           A=1
           C=1
           E=1
        ELSE
           A=CHMIBL(I-1)+1
           C=SASIBL(I-1)+1
           E=SASWBL(I-1)+1
        ENDIF
        B=CHMIBL(I)
        D=SASIBL(I)
        F=SASWBL(I)

        !     Go through all the pairs in the nonbond pair list.

        DO J=A,B
           M=I
           N=ABS(CHMINB(J))
           IF (SASLCT(N) .NE. 0) THEN
              U=SASIDX(M)
              V=SASIDX(N)

              !     Check if the M-N pair is also in the SASWNB/SASWBL pair list. If
              !     so, it's a 1-2 pair and the 1-2 connectivity parameter has to be
              !     used. If not, it's a 1-n pair with n greater than 2 and the
              !     connectivity parameter for more distant pairs is to be applied.

              K=E
              DO WHILE ((K .LE. F) .AND. (N .GT. SASWNB(K)))
                 K=K+1
              ENDDO
              IF       ((K .LE. F) .AND. (N .EQ. SASWNB(K))) THEN
                 CON=SASFCO
              ELSE
                 CON=SASNCO
              ENDIF

              !     Calculate the contribution of the M-N pair to the reduction of
              !     the solvent accessible surface areas of atoms M and N.

              XMN=X(M)-X(N)
              YMN=Y(M)-Y(N)
              ZMN=Z(M)-Z(N)
              DMN2=XMN**2+YMN**2+ZMN**2
              IF (DMN2 .LT. SASSUM(U,V)**2) THEN
                 DMN1     =SQRT(DMN2)
                 T1MN     =PI*(SASSUM(U,V)-DMN1)
                 T2MN     =SASDIF(U,V)/DMN1
                 BMN      =SASRSS(U)*T1MN*(ONE-T2MN)
                 BNM      =SASRSS(V)*T1MN*(ONE+T2MN)
                 TMN      =SASPAT(U)*CON*BMN/SASASS(U)
                 TNM      =SASPAT(V)*CON*BNM/SASASS(V)
                 SASACS(M)=SASACS(M)*(ONE-TMN)
                 SASACS(N)=SASACS(N)*(ONE-TNM)
              ENDIF
           ENDIF
        ENDDO

        !     Go through all the pairs in the SASA pair list.

        DO J=C,D
           M=I
           N=ABS(SASINB(J))
           IF (SASLCT(N) .NE. 0) THEN
              U=SASIDX(M)
              V=SASIDX(N)

              !     Check if the M-N pair is also in the SASWNB/SASWBL pair list. If
              !     so, it's a 1-2 pair and the 1-2 connectivity parameter has to be
              !     used. If not, it's a 1-n pair with n greater than 2 and the
              !     connectivity parameter for more distant pairs is to be applied.

              K=E
              DO WHILE ((K .LE. F) .AND. (N .GT. SASWNB(K)))
                 K=K+1
              ENDDO
              IF       ((K .LE. F) .AND. (N .EQ. SASWNB(K))) THEN
                 CON=SASFCO
              ELSE
                 CON=SASNCO
              ENDIF

              !     Calculate the contribution of the M-N pair to the reduction of
              !     the solvent accessible surface areas of atoms M and N.

              XMN=X(M)-X(N)
              YMN=Y(M)-Y(N)
              ZMN=Z(M)-Z(N)
              DMN2=XMN**2+YMN**2+ZMN**2
              IF (DMN2 .LT. SASSUM(U,V)**2) THEN
                 DMN1     =SQRT(DMN2)
                 T1MN     =PI*(SASSUM(U,V)-DMN1)
                 T2MN     =SASDIF(U,V)/DMN1
                 BMN      =SASRSS(U)*T1MN*(ONE-T2MN)
                 BNM      =SASRSS(V)*T1MN*(ONE+T2MN)
                 TMN      =SASPAT(U)*CON*BMN/SASASS(U)
                 TNM      =SASPAT(V)*CON*BNM/SASASS(V)
                 SASACS(M)=SASACS(M)*(ONE-TMN)
                 SASACS(N)=SASACS(N)*(ONE-TNM)
              ENDIF
           ENDIF
        ENDDO
     ENDIF
  ENDDO

  !     Begin image.

  DO I=NAT+1,NATIM
     MI=IMATTR(I)
     IF (SASLCT(MI) .NE. 0) THEN
        IF (I .EQ. 1) THEN
           A=1
        ELSE
           A=IMAIBL(I-1)+1
        ENDIF
        B=IMAIBL(I)
        DO J=A,B
           M=I
           N=ABS(IMAINB(J))
           NI=N
           IF ((SASLCT(NI) .NE. 0)) THEN
              U=SASIDX(MI)
              V=SASIDX(NI)
              CON=SASNCO
              XMN=X(M)-X(N)
              YMN=Y(M)-Y(N)
              ZMN=Z(M)-Z(N)
              DMN2=XMN**2+YMN**2+ZMN**2
              IF (DMN2 .LT. SASSUM(U,V)**2) THEN
                 DMN1      =SQRT(DMN2)
                 T1MN      =PI*(SASSUM(U,V)-DMN1)
                 T2MN      =SASDIF(U,V)/DMN1
                 BMN       =SASRSS(U)*T1MN*(ONE-T2MN)
                 BNM       =SASRSS(V)*T1MN*(ONE+T2MN)
                 TMN       =SASPAT(U)*CON*BMN/SASASS(U)
                 TNM       =SASPAT(V)*CON*BNM/SASASS(V)
                 SASACS(MI)=SASACS(MI)*(ONE-TMN)
                 SASACS(NI)=SASACS(NI)*(ONE-TNM)
              ENDIF
           ENDIF
        ENDDO
     ENDIF
  ENDDO

  !     End image.

  !     SASA solvation energy.

  DO I=1,NAT
     IF (SASLCT(I) .NE. 0) THEN
        U=SASIDX(I)
        TMP=SASSIG(U)
        MST=MST+TMP*SASACS(I)
     ENDIF
  ENDDO
  ETERM(SASTRM)=MST

  IF (QSRFC) THEN
     DO I=1,NAT
        IF (SASLCT(I) .NE. 0) THEN
           WMAIN(I)=SASACS(I)
        ENDIF
     ENDDO
  ENDIF

  !     Calculate the derivatives of the mean solvation energy. Go through
  !     all the pairs in the nonbond pair list and in the SASA pair list.
  !     Only pairs where both atoms have been selected by the user for
  !     SASA are taken into account. All other pairs are neglected.

  DO I=1,NAT
     IF (SASLCT(I) .NE. 0) THEN
        IF (I .EQ. 1) THEN
           A=1
           C=1
           E=1
        ELSE
           A=CHMIBL(I-1)+1
           C=SASIBL(I-1)+1
           E=SASWBL(I-1)+1
        ENDIF
        B=CHMIBL(I)
        D=SASIBL(I)
        F=SASWBL(I)

        !     Go through all the pairs in the nonbond pair list.

        DO J=A,B
           M=I
           N=ABS(CHMINB(J))
           IF (SASLCT(N) .NE. 0) THEN
              U=SASIDX(M)
              V=SASIDX(N)

              !     Check if the M-N pair is also in the SASWNB/SASWBL pair list. If
              !     so, it's a 1-2 pair and the 1-2 connectivity parameter has to be
              !     used. If not, it's a 1-n pair with n greater than 2 and the
              !     connectivity parameter for more distant pairs is to be applied.

              K=E
              DO WHILE ((K .LE. F) .AND. (N .GT. SASWNB(K)))
                 K=K+1
              ENDDO
              IF       ((K .LE. F) .AND. (N .EQ. SASWNB(K))) THEN
                 CON=SASFCO
              ELSE
                 CON=SASNCO
              ENDIF

              !     Calculate the contribution of the M-N pair to the derivatives
              !     of the mean solvation energy with respect to the atoms M and N.

              XMN=X(M)-X(N)
              YMN=Y(M)-Y(N)
              ZMN=Z(M)-Z(N)
              DMN2=XMN**2+YMN**2+ZMN**2
              IF (DMN2 .LT. SASSUM(U,V)**2) THEN
                 DMN1  =SQRT(DMN2)
                 T1MN  =PI*(SASSUM(U,V)-DMN1)
                 T2MN  =SASDIF(U,V)/DMN1
                 BMN   =SASRSS(U)*T1MN*(ONE-T2MN)
                 BNM   =SASRSS(V)*T1MN*(ONE+T2MN)
                 DBMNP =PI*SASRSS(U)*(DMN2-SASPRD(U,V))
                 DBNMP =PI*SASRSS(V)*(DMN2+SASPRD(U,V))
                 TMP   =SASSIG(U)
                 T3MN  =TMP*SASACS(M)*DBMNP
                 T4MN  =SASASS(U)/(SASPAT(U)*CON)-BMN
                 T5MN  =T3MN/T4MN
                 TMP   =SASSIG(V)
                 T3NM  =TMP*SASACS(N)*DBNMP
                 T4NM  =SASASS(V)/(SASPAT(V)*CON)-BNM
                 T5NM  =T3NM/T4NM
                 DMSTPS=(T5MN+T5NM)/(DMN1*DMN2)
                 DMSTPX=XMN*DMSTPS
                 DMSTPY=YMN*DMSTPS
                 DMSTPZ=ZMN*DMSTPS
                 DX(M) =DX(M)+DMSTPX
                 DY(M) =DY(M)+DMSTPY
                 DZ(M) =DZ(M)+DMSTPZ
                 DX(N) =DX(N)-DMSTPX
                 DY(N) =DY(N)-DMSTPY
                 DZ(N) =DZ(N)-DMSTPZ
              ENDIF
           ENDIF
        ENDDO

        !     Go through all the pairs in the SASA pair list.

        DO J=C,D
           M=I
           N=ABS(SASINB(J))
           IF (SASLCT(N) .NE. 0) THEN
              U=SASIDX(M)
              V=SASIDX(N)

              !     Check if the M-N pair is also in the SASWNB/SASWBL pair list. If
              !     so, it's a 1-2 pair and the 1-2 connectivity parameter has to be
              !     used. If not, it's a 1-n pair with n greater than 2 and the
              !     connectivity parameter for more distant pairs is to be applied.

              K=E
              DO WHILE ((K .LE. F) .AND. (N .GT. SASWNB(K)))
                 K=K+1
              ENDDO
              IF       ((K .LE. F) .AND. (N .EQ. SASWNB(K))) THEN
                 CON=SASFCO
              ELSE
                 CON=SASNCO
              ENDIF

              !     Calculate the contribution of the M-N pair to the derivatives
              !     of the mean solvation energy with respect to the atoms M and N.

              XMN=X(M)-X(N)
              YMN=Y(M)-Y(N)
              ZMN=Z(M)-Z(N)
              DMN2=XMN**2+YMN**2+ZMN**2
              IF (DMN2 .LT. SASSUM(U,V)**2) THEN
                 DMN1  =SQRT(DMN2)
                 T1MN  =PI*(SASSUM(U,V)-DMN1)
                 T2MN  =SASDIF(U,V)/DMN1
                 BMN   =SASRSS(U)*T1MN*(ONE-T2MN)
                 BNM   =SASRSS(V)*T1MN*(ONE+T2MN)
                 DBMNP =PI*SASRSS(U)*(DMN2-SASPRD(U,V))
                 DBNMP =PI*SASRSS(V)*(DMN2+SASPRD(U,V))
                 TMP   =SASSIG(U)
                 T3MN  =TMP*SASACS(M)*DBMNP
                 T4MN  =SASASS(U)/(SASPAT(U)*CON)-BMN
                 T5MN  =T3MN/T4MN
                 TMP   =SASSIG(V)
                 T3NM  =TMP*SASACS(N)*DBNMP
                 T4NM  =SASASS(V)/(SASPAT(V)*CON)-BNM
                 T5NM  =T3NM/T4NM
                 DMSTPS=(T5MN+T5NM)/(DMN1*DMN2)
                 DMSTPX=XMN*DMSTPS
                 DMSTPY=YMN*DMSTPS
                 DMSTPZ=ZMN*DMSTPS
                 DX(M) =DX(M)+DMSTPX
                 DY(M) =DY(M)+DMSTPY
                 DZ(M) =DZ(M)+DMSTPZ
                 DX(N) =DX(N)-DMSTPX
                 DY(N) =DY(N)-DMSTPY
                 DZ(N) =DZ(N)-DMSTPZ
              ENDIF
           ENDIF
        ENDDO
     ENDIF
  ENDDO

  !     Begin image.

  DO I=NAT+1,NATIM
     MI=IMATTR(I)
     IF (SASLCT(MI) .NE. 0) THEN
        IF (I .EQ. 1) THEN
           A=1
        ELSE
           A=IMAIBL(I-1)+1
        ENDIF
        B=IMAIBL(I)
        DO J=A,B
           M=I
           N=ABS(IMAINB(J))
           NI=N
           IF ((SASLCT(NI) .NE. 0)) THEN
              U=SASIDX(MI)
              V=SASIDX(NI)
              CON=SASNCO
              XMN=X(M)-X(N)
              YMN=Y(M)-Y(N)
              ZMN=Z(M)-Z(N)
              DMN2=XMN**2+YMN**2+ZMN**2
              IF (DMN2 .LT. SASSUM(U,V)**2) THEN
                 DMN1  =SQRT(DMN2)
                 T1MN  =PI*(SASSUM(U,V)-DMN1)
                 T2MN  =SASDIF(U,V)/DMN1
                 BMN   =SASRSS(U)*T1MN*(ONE-T2MN)
                 BNM   =SASRSS(V)*T1MN*(ONE+T2MN)
                 DBMNP =PI*SASRSS(U)*(DMN2-SASPRD(U,V))
                 DBNMP =PI*SASRSS(V)*(DMN2+SASPRD(U,V))
                 TMP   =SASSIG(U)
                 T3MN  =TMP*SASACS(MI)*DBMNP
                 T4MN  =SASASS(U)/(SASPAT(U)*CON)-BMN
                 T5MN  =T3MN/T4MN
                 TMP   =SASSIG(V)
                 T3NM  =TMP*SASACS(NI)*DBNMP
                 T4NM  =SASASS(V)/(SASPAT(V)*CON)-BNM
                 T5NM  =T3NM/T4NM
                 DMSTPS=(T5MN+T5NM)/(DMN1*DMN2)
                 DMSTPX=XMN*DMSTPS
                 DMSTPY=YMN*DMSTPS
                 DMSTPZ=ZMN*DMSTPS
                 DX(MI)=DX(MI)+DMSTPX
                 DY(MI)=DY(MI)+DMSTPY
                 DZ(MI)=DZ(MI)+DMSTPZ
                 DX(NI)=DX(NI)-DMSTPX
                 DY(NI)=DY(NI)-DMSTPY
                 DZ(NI)=DZ(NI)-DMSTPZ
              ENDIF
           ENDIF
        ENDDO
     ENDIF
  ENDDO
  !     End image.
  return
end SUBROUTINE SASENE

#endif 
SUBROUTINE NULL_SASAENE
  RETURN
END SUBROUTINE NULL_SASAENE

