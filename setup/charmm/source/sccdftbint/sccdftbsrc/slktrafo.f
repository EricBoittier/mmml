!
! Geometrical transformation of ss-Overlapp and Hamilton matrix elements
!
      SUBROUTINE SKSS(X,X2,I,J,R2,IOVPAR,EM,NE)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION X(6),X2(6),PARM(13),EM(1,1)
      EXTERNAL IOVPAR
!
        ID=IOVPAR(I,J,R2,PARM)
        EM(1,1)=PARM(10)
        RETURN
      END

!
! Geometrical transformation of sp-Overlapp and Hamilton matrix elements
!
      SUBROUTINE SKSP(X,X2,I,J,R2,IOVPAR,EM,EMT,NE)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION X(6),X2(6),PARM(13),EM(NE,3),EMT(NE,3)
      EXTERNAL IOVPAR
!
        ID=IOVPAR(I,J,R2,PARM)
        DO 10 L=1,3
          EM(1,L)=X(L)*PARM(9)
          EMT(L,1)=-EM(1,L)
   10   CONTINUE
        RETURN
      END

!
! Geometrical transformation of sd-Overlapp and Hamilton matrix elements
!
      SUBROUTINE SKSD(X,X2,I,J,R2,IOVPAR,EM,EMT,NE)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION X(6),X2(6),PARM(13),EM(NE,5),EMT(NE,5),ES(5)
      EXTERNAL IOVPAR
!
        R3=SQRT(3.0)
        D4=X2(3)-0.5*(X2(1)+X2(2))
        D5=X2(1)-X2(2)
        ID=IOVPAR(I,J,R2,PARM)
!
        DO 20 L=1,3
          ES(L)=R3*X(L)*X(L+1)
   20   CONTINUE
        ES(4)=0.5*R3*D5
        ES(5)=D4
        DO 21 L=1,5
          EM(1,L)=ES(L)*PARM(8)
          EMT(L,1)=EM(1,L)
   21   CONTINUE
        RETURN
      END

!
! Geometrical transformation of pp-Overlapp and Hamilton matrix elements
!
      SUBROUTINE SKPP(X,X2,I,J,R2,IOVPAR,EM,NE)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION X(6),DM(6),X2(6),PARM(13),EM(NE,3),EPP(6)
      EXTERNAL IOVPAR
!
        ID=IOVPAR(I,J,R2,PARM)
        DO 30 L=1,3
          EPP(L)=X2(L)
          EPP(L+3)=X(L)*X(L+1)
   30   CONTINUE
        DO 31 L=1,3
          HP=EPP(L)
          DM(L)=HP*PARM(6)+(1.0-HP)*PARM(7)
   31   CONTINUE
        DO 32 L=4,6
          DM(L)=EPP(L)*(PARM(6)-PARM(7))
   32   CONTINUE
        DO 33 IR=1,3
          DO 34 IS=1,IR
            II=IR-IS
            K=3*II-(II*(II-1))/2+IS
            EM(IS,IR)=DM(K)
            EM(IR,IS)=DM(K)
   34     CONTINUE
   33   CONTINUE
        RETURN
      END

!
! Geometrical transformation of pd-Overlapp and Hamilton matrix elements
!
      SUBROUTINE SKPD(X,X2,I,J,R2,IOVPAR,EM,EMT,NE)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION X(6),X2(6),PARM(13),EM(NE,5),EMT(NE,5),EPD(13,2),DM(15)
      EXTERNAL IOVPAR
!
        R3=SQRT(3.0)
        D3=X2(1)+X2(2)
        D4=X2(3)-0.5*D3
        D5=X2(1)-X2(2)
        D6=X(1)*X(2)*X(3)
        ID=IOVPAR(I,J,R2,PARM)
        DO 40 L=1,3
          EPD(L,1)=R3*X2(L)*X(L+1)
          EPD(L,2)=X(L+1)*(1.0-2.0*X2(L))
          EPD(L+4,1)=R3*X2(L)*X(L+2)
          EPD(L+4,2)=X(L+2)*(1.0-2.0*X2(L))
          EPD(L+7,1)=0.5*R3*X(L)*D5
          EPD(L+10,1)=X(L)*D4
   40   CONTINUE
        EPD(4,1)=R3*D6
        EPD(4,2)=-2.0*D6
        EPD(8,2)=X(1)*(1.0-D5)
        EPD(9,2)=-X(2)*(1.0+D5)
        EPD(10,2)=-X(3)*D5
        EPD(11,2)=-R3*X(1)*X2(3)
        EPD(12,2)=-R3*X(2)*X2(3)
        EPD(13,2)=R3*X(3)*D3
        DO 41 L=1,15
          DM(L)=0.0
   41   CONTINUE
        DO 42 M=1,2
          DM(1)=DM(1)+EPD(1,M)*PARM(M+3)
          DM(2)=DM(2)+EPD(6,M)*PARM(M+3)
          DM(3)=DM(3)+EPD(4,M)*PARM(M+3)
          DM(5)=DM(5)+EPD(2,M)*PARM(M+3)
          DM(6)=DM(6)+EPD(7,M)*PARM(M+3)
          DM(7)=DM(7)+EPD(5,M)*PARM(M+3)
          DM(9)=DM(9)+EPD(3,M)*PARM(M+3)
          DO 43 L=8,13
            DM(L+2)=DM(L+2)+EPD(L,M)*PARM(M+3)
   43     CONTINUE
   42   CONTINUE
        DM(4)=DM(3)
        DM(8)=DM(3)
        DO 44 IR=1,5
          DO 45 IS=1,3
            K=3*(IR-1)+IS
            EMT(IR,IS)=-DM(K)
            EM(IS,IR)=DM(K)
   45     CONTINUE
   44   CONTINUE
        RETURN
      END

!
! Geometrical transformation of dd-Overlapp and Hamilton matrix elements
!
      SUBROUTINE SKDD(X,X2,I,J,R2,IOVPAR,EM,NE)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION X(6),X2(6),PARM(13),EM(NE,5),E(15,3),DM(15),DD(3)
      EXTERNAL IOVPAR
!
        R3=SQRT(3.0)
        D3=X2(1)+X2(2)
        D4=X2(3)-0.5*D3
        D5=X2(1)-X2(2)
        ID=IOVPAR(I,J,R2,PARM)
        DO 50 L=1,3
          E(L,1)=X2(L)*X2(L+1)
          E(L,2)=X2(L)+X2(L+1)-4.0*E(L,1)
          E(L,3)=X2(L+2)+E(L,1)
          E(L,1)=3.0*E(L,1)
   50   CONTINUE
        E(4,1)=D5*D5
        E(4,2)=D3-E(4,1)
        E(4,3)=X2(3)+0.25*E(4,1)
        E(4,1)=0.75*E(4,1)
        E(5,1)=D4*D4
        E(5,2)=3.0*X2(3)*D3
        E(5,3)=D3*D3*0.75
        DD(1)=X(1)*X(3)
        DD(2)=X(2)*X(1)
        DD(3)=X(3)*X(2)
        DO 51 L=1,2
          E(L+5,1)=3.0*X2(L+1)*DD(L)
          E(L+5,2)=DD(L)*(1.0-4.0*X2(L+1))
          E(L+5,3)=DD(L)*(X2(L+1)-1.0)
   51   CONTINUE
        E(8,1)=DD(1)*D5*1.5
        E(8,2)=DD(1)*(1.0-2.0*D5)
        E(8,3)=DD(1)*(0.5*D5-1.0)
        E(9,1)=D5*0.5*D4*R3
        E(9,2)=-D5*X2(3)*R3
        E(9,3)=D5*0.25*(1.0+X2(3))*R3
        E(10,1)=X2(1)*DD(3)*3.0
        E(10,2)=(0.25-X2(1))*DD(3)*4.0
        E(10,3)=DD(3)*(X2(1)-1.0)
        E(11,1)=1.5*DD(3)*D5
        E(11,2)=-DD(3)*(1.0+2.0*D5)
        E(11,3)=DD(3)*(1.0+0.5*D5)
        E(13,3)=0.5*D5*DD(2)
        E(13,2)=-2.0*DD(2)*D5
        E(13,1)=E(13,3)*3.0
        E(12,1)=D4*DD(1)*R3
        E(14,1)=D4*DD(3)*R3
        E(15,1)=D4*DD(2)*R3
        E(15,2)=-2.0*R3*DD(2)*X2(3)
        E(15,3)=0.5*R3*(1.0+X2(3))*DD(2)
        E(14,2)=R3*DD(3)*(D3-X2(3))
        E(14,3)=-R3*0.5*DD(3)*D3
        E(12,2)=R3*DD(1)*(D3-X2(3))
        E(12,3)=-R3*0.5*DD(1)*D3
        DO 52 L=1,15
          DM(L)=0.0
          DO 53 M=1,3
            DM(L)=DM(L)+E(L,M)*PARM(M)
   53     CONTINUE
   52   CONTINUE
        DO 54 IR=1,5
          DO 55 IS=1,IR
            II=IR-IS
            K=5*II-(II*(II-1))/2+IS
            EM(IR,IS)=DM(K)
            EM(IS,IR)=DM(K)
   55     CONTINUE
   54   CONTINUE
        RETURN
      END

