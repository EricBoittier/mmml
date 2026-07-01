#if KEY_MOLVIB==1
SUBROUTINE BRUSOR(ZPED,IPTA,QSEL,NPED)
  !
  ! ... Brutal sorting routine. Use only for  symbolic PED sorting, when
  ! ... there are no more than 3-6 elements.
  ! ... I am just too lazy to look up a better algorithm (KK)
  ! ... On output IPTA(1),IPTA(2),... will contain the index numbers
  ! ... of the largest, second largest,... elements of positive array
  ! ... ZPED
  !
  use chm_kinds
  implicit none
  INTEGER NPED,I,J,JMAX
  real(chm_real) ZPED(NPED),ZMAX
  INTEGER IPTA(NPED)
  LOGICAL QSEL(NPED)
  !
  DO I=1,NPED
     QSEL(I)=.TRUE.
  END DO
  !
  DO I=1,NPED
     ZMAX=-1.0
     JMAX=-1
     DO J=1,NPED
        IF(ABS(ZPED(J)).GT.ABS(ZMAX) .AND. QSEL(J)) THEN
           ZMAX=ZPED(J)
           JMAX=J
        END IF
     END DO
     IPTA(I)=JMAX
     QSEL(JMAX)=.FALSE.
  END DO
  !
  RETURN
END SUBROUTINE BRUSOR

SUBROUTINE DIFFRQ(FCALC,FEXP,NQ,ILO,IHI,S,IO)
  !
  !   Calculate the deviations of the two sets of vibrational
  !   frequencies (in cm-1). Eg. FCALC could contain the calculated
  !   and FEXP the experimental frequencies.
  !   NQ - no. of frequencies
  !
  use chm_kinds
  implicit none
  INTEGER NQ,ILO,IHI,IO,I
  real(chm_real) FCALC(NQ),FEXP(NQ),S
  !
  S=0.0D0
  WRITE(IO,*)
  WRITE(IO,*) ' Comparison of calculated and reference freqs'
  WRITE(IO,*)
  WRITE(IO,*) '  #      Calc.      Ref.       Ref.-Calc. '
  WRITE(IO,*)
  DO I=ILO,IHI
     S=S+(FCALC(I)-FEXP(I))**2
     WRITE(IO,900) I,FCALC(I),FEXP(I),FEXP(I)-FCALC(I)
  END DO
  WRITE(IO,*)
  WRITE(IO,*)  ' The sum of squares and rms deviation'
  WRITE(IO,910) S,SQRT(S/(IHI-ILO+1))
  WRITE(IO,*)
  !
900 FORMAT(1X,I3,2X,2F10.1,4X,F10.1)
910 FORMAT(11X,2F12.2)
  RETURN
END SUBROUTINE DIFFRQ

SUBROUTINE GTRAN(G,U,W,NI,NF,NMAX,IPRNT,IO)
  !
  !   Similarity transformation of matrix G by matrix U
  !   on input : G is NIxNI
  !   on output: G=U*G*UT is NFXNF
  !   W - work array
  !
  use chm_kinds
  implicit none
  INTEGER  NMAX
  real(chm_real) G(NMAX,NMAX),U(NMAX,NMAX),W(NMAX,NMAX),S
  INTEGER NI,NF,I,J,K,IPRNT,IO
  !
  DO I=1,NF
     DO J=1,NI
        S=0.0
        DO K=1,NI
           S = S + U(I,K)*G(K,J)
        END DO
        W(I,J)=S
     END DO
  END DO
  !
  DO I=1,NF
     DO J=1,NF
        S=0.0
        DO K=1,NI
           S = S + W(I,K)*U(J,K)
        END DO
        G(I,J)=S
     END DO
  END DO
  !
  IF(IPRNT.GE.2) THEN
     WRITE(IO,*) ' The transformed G matrix'
     WRITE(IO,*)
     DO I=1,NF
        WRITE(IO,1006) I,(G(I,J),J=1,NF)
     END DO
  END IF
1006 FORMAT(1X,I3,8F9.3,12(/4X,8F9.3))
  !
  RETURN
END SUBROUTINE GTRAN

SUBROUTINE RMVTR0(X,W1,W2,NQM,NQ,NAT,NAT3,AMASS,CALFRQ,IPRNT)
  !
  !     Remove translation-rotation contributions from vibrational
  !     eigenvectors in cartesian coordinates
  !     Differece from RMVTR : only vibrational eigenvectors used
  !
  !     The translational eigenvectors are taken to be (for each atom):
  !     m*[1,0,0], m[0,1,0] and m[0,0,1]
  !     while the infinitesimal CM rotation generators are:
  !     m*[0,-z,y], m*[z,0,-x] and m*[-y,x,0]
  !
  !     X - contains cartesian coordinates [x1,...,xn,y1,...yn,z1,...zn]
  !     W1 - contains cartesian eigenvectors in columns;
  !          1...6 are transl. and rot., 7...NAT3 are vibrations
  !     W2 - colums contain the translation and rotation generators
  !     NAT - no. of atoms, NAT3=3*NAT
  !     NQ = NAT3-6
  !     NQM - array dimensions in calling subroutine
  !
  use chm_kinds
  implicit none
  INTEGER NQM,NQ,NAT,NAT3,IPRNT
  real(chm_real) X(NAT3),W1(NQM,NQM),W2(NQM,NQM),AMASS(NAT3)
  real(chm_real) CALFRQ(NQ),S
  !
  INTEGER I,J,K,K1,K2,NVIB
  !
  WRITE(*,*) ' RMVTR0 : removing transl-rot', &
       ' contribution to cartesian eigenvectors LX'
  WRITE(*,*)
  !
  ! ... Fill W2 columns with CM transl. and rot. generators
  !
  DO K=1,NAT
     K1=K+NAT
     K2=K1+NAT
     W2(K,1)  = 1.0
     W2(K1,2) = 1.0
     W2(K2,3) = 1.0
     W2(K1,4) = -X(K2)
     W2(K2,4) =  X(K1)
     W2(K,5)  =  X(K2)
     W2(K2,5) = -X(K)
     W2(K,6)  = -X(K1)
     W2(K1,6) =  X(K)
  END DO
  DO K=1,NAT3
     DO J=1,6
        W2(K,J) = W2(K,J)*AMASS(K)
     END DO
  END DO
  !
  ! ... Normalize
  !
  DO J=1,6
     S=0.0
     DO K=1,NAT3
        S = S + W2(K,J)**2
     END DO
     S=SQRT(S)
     IF(S.LT.1.0D-6) THEN
        WRITE(6,*) ' Warning from RMVTR : norm too small for J=',J
     END IF
     DO K=1,NAT3
        W2(K,J) = W2(K,J)/S
     END DO
  END DO
  !
  ! ... Subtract projections from vibrational eigenvectors
  !
  NVIB=NAT3-6
  DO I=1,NVIB
     DO J=1,6
        S=0.0
        DO K=1,NAT3
           S = S + W1(K,I)*W2(K,J)
        END DO
        DO K=1,NAT3
           W1(K,I) = W1(K,I) - S*W2(K,J)
        END DO
     END DO
  END DO
  !
  ! ... Re-normalize eigenvectors
  !
  DO I=1,NVIB
     S=0.0
     DO K=1,NAT3
        S = S + W1(K,I)**2
     END DO
     IF(S.LT.1.0D-12) THEN
        WRITE(*,*) ' RMVTR0 warning: norm too small for I=',I
        S=1.0D-12
     END IF
     S=SQRT(S)
     DO K=1,NAT3
        W1(K,I)=W1(K,I)/S
     END DO
  END DO
  IF(IPRNT.GE.1) THEN
     WRITE(*,*)
     WRITE(*,*) ' The LX matrix after projection on vibrational', &
          ' subspace [LX(K,I),K=1,NAT3]'
     WRITE(*,*)
     DO I=1,NVIB
        WRITE(*,900) I,CALFRQ(I),(W1(K,I),K=1,NAT3)
     END DO
  END IF
900 FORMAT(1X,I3,2X,F8.1,2X,10F9.5,12(/16X,10F9.5))
  RETURN
END SUBROUTINE RMVTR0

SUBROUTINE SYMSOR(W,DD,NQ,NQM,IBLOCK,NBLOCK,IPTB)
  !
  !    New version : does not permute DD elements or W rows, pointers used.
  !    This subroutine sorts eigenvectors acording to ascending eigenvalues
  !    within each symmetry block separately.
  !    IBLOCK[1...NBLOCK] - dimensions of symmetry blocks
  !    IPTB - pointer array for rearranging PED rows
  !    W - the PED matrix from subroutine PED
  !    DD - the eigenvalues, on input - in ascending order
  !    NQ - no. of vibrational degrees of freedom
  !    NQM - W dimension in calling routine
  !    Warning: For E (or F,... etc) representations non-mixing symmetry
  !             coordinates must be selected, otherwise the blocking algorithm
  !             may fail.
  !
  use chm_kinds
  implicit none
  INTEGER NQM,NQ
  real(chm_real) W(NQM,NQM),DD(NQ),S,ZMIN
  INTEGER NBLOCK
  INTEGER IBLOCK(NBLOCK),IPTB(NQ)
  INTEGER I,IC,ILO,IHI,IBLI,IPIC,ID,IS,IHJ,J,JP,K,KP,KMIN,L
  !
  ! ... First: de-sort the freqs and PED rows so that those of block #1 come
  ! ... first, block #2 netx, ... etc
  ! ... IC counts the current row, ILO,IHI are first and last coord in block I
  !
  DO I=1,NQ
     IPTB(I)=I
  END DO
  IC=0
  IHI=0
  !
  DO I=1,NBLOCK
     IBLI=IBLOCK(I)
     ILO=IHI+1
     IHI=IHI+IBLI
     !      WRITE(6,*) ' I,IBLI,ILO,IHI=',I,IBLI,ILO,IHI
     !
     DO J=1,IBLI
        IC=IC+1
        IPIC=IPTB(IC)
        !
        ! ... check if IC belongs to block I; if yes - leave row IC as is
        S=0.0D0
        DO L=ILO,IHI
           S=S+W(IPIC,L)
        END DO
        IF(S.GT.0.5) THEN
           !       WRITE(6,*) '  =Row IC belongs to block I; I,J=',I,J
           GOTO 66
        END IF
        !
        ! ... if not - find the next row belonging to I and permute it with IC
        ID=IC+1
        DO K=ID,NQ
           KP=IPTB(K)
           S=0.0D0
           DO L=ILO,IHI
              S=S+W(KP,L)
           END DO
           !       WRITE(*,*) '   J,IC,K,S=',J,IC,K,S
           !
           IF(S.GT.0.9) THEN
              !       WRITE(*,*) '     Permuting rows:',IC,K
              IS=IPTB(IC)
              IPTB(IC)=IPTB(K)
              IPTB(K)=IS
              GOTO 66
           END IF
           !
        END DO   ! K=ID,NQ
66      CONTINUE
        !
     END DO     ! J=1,IBLI
     !
  END DO       ! I=1,NBLOCK
  !
  ! ... Test operation
  !      WRITE(*,*)
  !      WRITE(*,909) (IPTB(I),I=1,NQ)
  !      WRITE(*,*)
  !      WRITE(*,*) ' The PED matrix after blocking operation'
  !      WRITE(*,*)
  !      DO I=1,NQ
  !        II=IPTB(I)
  !        WRITE(*,908) I,DD(II),(W(II,J),J=1,NQ)
  !      END DO
  !908   FORMAT(1X,I3,F8.1,10F6.2,12(/12X,10F6.2))
  !909   FORMAT(1X,20I3,5(/1X,20I3))
  !
  ! ... Now sort frequencies within each block
  ! ... Loop over symmetry blocks, ILO and IHI are block boundaries
  !
  IHI=0
  DO I=1,NBLOCK
     IBLI=IBLOCK(I)
     ILO=IHI+1
     IHI=IHI+IBLI
     !
     ! ... Sort each block: ascending frequencies
     !
     IHJ=IHI-1
     DO J=ILO,IHJ
        JP=IPTB(J)
        ZMIN=1000000.0
        DO K=J,IHI
           KP=IPTB(K)
           IF(DD(KP).LT.ZMIN) THEN
              ZMIN=DD(KP)
              KMIN=K
           END IF
        END DO
        ! ... Permute frequencies
        !       WRITE(*,*) '  Permuting rows J,KMIN=',J,KMIN
        IS=IPTB(J)
        IPTB(J)=IPTB(KMIN)
        IPTB(KMIN)=IS
        !
     END DO  ! J=ILO,IHI
     !
  END DO    ! I=1,NBLOCK
  !
  ! ... Test operation
  !      WRITE(*,*)
  !      WRITE(*,909) (IPTB(I),I=1,NQ)
  !      WRITE(*,*)
  !      WRITE(*,*) ' The PED matrix after sorting within blocks'
  !      WRITE(*,*)
  !      DO I=1,NQ
  !        II=IPTB(I)
  !        WRITE(6,908) I,DD(II),(W(II,J),J=1,NQ)
  !      END DO
  !
  RETURN
END SUBROUTINE SYMSOR

SUBROUTINE HSHLDR(A,DD,VV,N,NM)
  !
  !   Householder diagonalization of a symmetric matrix, using
  !   code from "Numerical Recipes" by W.H.Press, B.P.Flannery,
  !   S.A.Teukolsky and W.T.Vetterling, Cambridge University Press,
  !   Cambridge, 1986.
  !
  !    A - on input: matrix to be diagonalized
  !        on output: eigenvectors in columns, ordered by ascending
  !                   eigenvalues
  !    DD - eigenvalues
  !    VV - work vector
  !    N  - current dimension
  !    NM - dimension of A in calling routine
  !
  use chm_kinds
  implicit none
  INTEGER N,NM
  real(chm_real) A(NM,NM),DD(NM),VV(NM)
  !
  CALL TREDTWO(A,N,NM,DD,VV)
  CALL TQLI(DD,VV,N,NM,A)
  CALL EIGSRTA(DD,A,N,NM)
  !
  RETURN
END SUBROUTINE HSHLDR

SUBROUTINE EIGSRTA(D,V,N,NP)
  !
  !  Modified to give ascending order of eigenvalues
  !  and to real(chm_real)
  !
  use chm_kinds
  implicit none
  INTEGER N,NP
  real(chm_real) D(NP),V(NP,NP)
  INTEGER I,J,K
  real(chm_real) P
  !
  DO I=1,N-1
     K=I
     P=D(I)
     DO J=I+1,N
        IF(D(J).LE.P)THEN
           K=J
           P=D(J)
        ENDIF
     enddo
     IF(K.NE.I)THEN
        D(K)=D(I)
        D(I)=P
        DO J=1,N
           P=V(J,I)
           V(J,I)=V(J,K)
           V(J,K)=P
        enddo
     ENDIF
  enddo
  RETURN
END SUBROUTINE EIGSRTA

SUBROUTINE TQLI(D,E,N,NP,Z)
  !
  ! Modified to real(chm_real)
  !
  use chm_kinds
  implicit none
  INTEGER N,NP
  real(chm_real) D(NP),E(NP),Z(NP,NP)
  INTEGER I,ITER,K,L,M
  real(chm_real) B,C,DD,F,G,P,R,S
  !
  IF (N.GT.1) THEN
     DO I=2,N
        E(I-1)=E(I)
     enddo
     E(N)=0.
     loop15: DO L=1,N
        ITER=0
1       DO M=L,N-1
           DD=ABS(D(M))+ABS(D(M+1))
           IF (ABS(E(M))+DD.EQ.DD) GO TO 2
        enddo
        M=N
2       IF(M.NE.L)THEN
           IF(ITER.EQ.30) THEN
              CALL WRNDIE(-4,'<TQLI>','too many iterations')
              RETURN
           ENDIF
           ITER=ITER+1
           G=(D(L+1)-D(L))/(2.*E(L))
           R=SQRT(G**2+1.)
           G=D(M)-D(L)+E(L)/(G+SIGN(R,G))
           S=1.
           C=1.
           P=0.
           loop14:DO I=M-1,L,-1
              F=S*E(I)
              B=C*E(I)
              IF(ABS(F).GE.ABS(G))THEN
                 C=G/F
                 R=SQRT(C**2+1.)
                 E(I+1)=F*R
                 S=1./R
                 C=C*S
              ELSE
                 S=F/G
                 R=SQRT(S**2+1.)
                 E(I+1)=G*R
                 C=1./R
                 S=S*C
              ENDIF
              G=D(I+1)-P
              R=(D(I)-G)*S+2.*C*B
              P=S*R
              D(I+1)=G+P
              G=C*R-B
              !     Omit lines from here >>>
              DO K=1,N
                 F=Z(K,I+1)
                 Z(K,I+1)=S*Z(K,I)+C*F
                 Z(K,I)=C*Z(K,I)-S*F
              enddo
              !     >>> to here when finding only eigenvalues.
           enddo loop14
           D(L)=D(L)-P
           E(L)=G
           E(M)=0.
           GO TO 1
        ENDIF
     enddo loop15
  ENDIF
  RETURN
END SUBROUTINE TQLI

SUBROUTINE TREDTWO(A,N,NP,D,E)
  !
  ! Modified to real(chm_real)
  !
  use chm_kinds
  implicit none
  INTEGER N,NP
  real(chm_real) A(NP,NP),D(NP),E(NP)
  INTEGER I,J,K,L
  real(chm_real) F,G,H,HH,SCALE
  !
  IF(N.GT.1)THEN
     loop18:DO I=N,2,-1
        L=I-1
        H=0.
        SCALE=0.
        IF(L.GT.1)THEN
           DO K=1,L
              SCALE=SCALE+ABS(A(I,K))
           enddo
           IF(SCALE.EQ.0.)THEN
              E(I)=A(I,L)
           ELSE
              DO K=1,L
                 A(I,K)=A(I,K)/SCALE
                 H=H+A(I,K)**2
              enddo
              F=A(I,L)
              G=-SIGN(SQRT(H),F)
              E(I)=SCALE*G
              H=H-F*G
              A(I,L)=F-G
              F=0.
              loop15:DO J=1,L
                 !     Omit following line if finding only eigenvalues
                 A(J,I)=A(I,J)/H
                 !
                 G=0.
                 DO K=1,J
                    G=G+A(J,K)*A(I,K)
                 enddo
                 IF(L.GT.J)THEN
                    DO K=J+1,L
                       G=G+A(K,J)*A(I,K)
                    enddo
                 ENDIF
                 E(J)=G/H
                 F=F+E(J)*A(I,J)
              enddo loop15
              HH=F/(H+H)
              DO J=1,L
                 F=A(I,J)
                 G=E(J)-HH*F
                 E(J)=G
                 DO K=1,J
                    A(J,K)=A(J,K)-F*E(K)-G*A(I,K)
                 enddo
              enddo
           ENDIF
        ELSE
           E(I)=A(I,L)
        ENDIF
        D(I)=H
     enddo loop18
  ENDIF
  !     Omit following line if finding only eigenvalues.
  D(1)=0.
  !
  E(1)=0.
  loop23:DO I=1,N
     !     Delete lines from here >>>
     L=I-1
     IF(D(I).NE.0.)THEN
        DO J=1,L
           G=0.
           DO K=1,L
              G=G+A(I,K)*A(K,J)
           enddo
           DO K=1,L
              A(K,J)=A(K,J)-G*A(K,I)
           enddo
        enddo
     ENDIF
     !     >>> to here when finding only eigenvalues.
     D(I)=A(I,I)
     !     Also delete lines from here >>>
     A(I,I)=1.
     IF(L.GE.1)THEN
        DO J=1,L
           A(I,J)=0.
           A(J,I)=0.
        enddo
     ENDIF
     !     >>> to here when finding only eigenvalues.
  enddo loop23
  RETURN
END SUBROUTINE TREDTWO

SUBROUTINE MATNVN(A,Y,VV,INDX,N,NP)
  !
  !    Matrix inversion using LU decomposition
  !    from "Numerical recipes", W.H.Press, B.P.Flannery,
  !    S.A.Teukolsky and W.T.Vetterling, Cambridge University
  !    Press, 1986/1992.
  !
  ! Modified to real(chm_real)
  !
  !    A - input matrix, destroyed on return
  !    Y - output matrix = inverse of A
  !    N - order of A,B (actual dimension)
  !    NP - first dimension of A in calling routine
  !
  use chm_kinds
  use number

  implicit none
  INTEGER I,J,N,NP
  INTEGER INDX(NP)
  real(chm_real) A(NP,NP),Y(NP,NP),VV(NP)
  real(chm_real) D
  !
  ! ... Setup Y as identity matrix
  !
  Y(1:n,1:n)=zero
  DO I=1,N
     Y(I,I)=one
  enddo
  !
  ! ... Perform LU decomposition
  !
  CALL LUDCM(A,VV,N,NP,INDX,D)
  !
  ! ... Find inverse by columns
  !
  DO J=1,N
     CALL LUBKS(A,N,NP,INDX,Y(1,J))
  enddo
  RETURN
END SUBROUTINE MATNVN

SUBROUTINE LUDCM(A,VV,N,NP,INDX,D)
  !
  ! Modified to real(chm_real)
  !
  use chm_kinds
  implicit none
  real(chm_real),parameter :: TINY=1.0D-20
  INTEGER N,NP,INDX(NP)
  real(chm_real) A(NP,NP),VV(NP)
  INTEGER I,IMAX,J,K
  real(chm_real) AAMAX,D,DUM,SUM
  !
  D=1.
  DO I=1,N
     AAMAX=0.
     DO J=1,N
        IF (ABS(A(I,J)).GT.AAMAX) AAMAX=ABS(A(I,J))
     enddo
     IF(AAMAX.EQ.0.) THEN
        CALL WRNDIE(-4,'<MATNVN>','Singular matrix')
        RETURN
     ENDIF
     VV(I)=1./AAMAX
  enddo
  loop19:DO J=1,N
     IF (J.GT.1) THEN
        DO I=1,J-1
           SUM=A(I,J)
           IF (I.GT.1)THEN
              DO K=1,I-1
                 SUM=SUM-A(I,K)*A(K,J)
              enddo
              A(I,J)=SUM
           ENDIF
        enddo
     ENDIF
     AAMAX=0.
     DO I=J,N
        SUM=A(I,J)
        IF (J.GT.1)THEN
           DO K=1,J-1
              SUM=SUM-A(I,K)*A(K,J)
           enddo
           A(I,J)=SUM
        ENDIF
        DUM=VV(I)*ABS(SUM)
        IF (DUM.GE.AAMAX) THEN
           IMAX=I
           AAMAX=DUM
        ENDIF
     enddo
     IF (J.NE.IMAX)THEN
        DO K=1,N
           DUM=A(IMAX,K)
           A(IMAX,K)=A(J,K)
           A(J,K)=DUM
        enddo
        D=-D
        VV(IMAX)=VV(J)
     ENDIF
     INDX(J)=IMAX
     IF(J.NE.N)THEN
        IF(A(J,J).EQ.0.)A(J,J)=TINY
        DUM=1./A(J,J)
        DO I=J+1,N
           A(I,J)=A(I,J)*DUM
        enddo
     ENDIF
  enddo loop19
  IF(A(N,N).EQ.0.)A(N,N)=TINY
  RETURN
END SUBROUTINE LUDCM

SUBROUTINE LUBKS(A,N,NP,INDX,B)
  !
  ! Modified to real(chm_real)
  !
  use chm_kinds
  implicit none
  INTEGER N,NP
  real(chm_real) A(NP,NP),B(N)
  INTEGER INDX(N)
  INTEGER I,II,J,LL
  real(chm_real) SUM
  !
  II=0
  DO I=1,N
     LL=INDX(I)
     SUM=B(LL)
     B(LL)=B(I)
     IF (II.NE.0)THEN
        DO J=II,I-1
           SUM=SUM-A(I,J)*B(J)
        enddo
     ELSE IF (SUM.NE.0.) THEN
        II=I
     ENDIF
     B(I)=SUM
  enddo
  DO I=N,1,-1
     SUM=B(I)
     IF(I.LT.N)THEN
        DO J=I+1,N
           SUM=SUM-A(I,J)*B(J)
        enddo
     ENDIF
     B(I)=SUM/A(I,I)
  enddo
  !
  return
end SUBROUTINE LUBKS

#endif /*  MOLVIB*/
SUBROUTINE NULL_MU
  RETURN
END SUBROUTINE NULL_MU

