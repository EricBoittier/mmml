      SUBROUTINE SELFS(I,J,R2,IOVPAR,EM,NE)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION PARM(13),EM(NE,1)
      EXTERNAL IOVPAR
!
        ID=IOVPAR(I,J,R2,PARM)
        EM(1,1)=PARM(13)
        RETURN
      END


      SUBROUTINE SELFP(I,J,R2,IOVPAR,EM,NE)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION PARM(13),EM(NE,3)
      EXTERNAL IOVPAR
!
        ID=IOVPAR(I,J,R2,PARM)
        DO 11 L=1,3
          DO 10 M=1,3
            EM(L,M)=0.0
10        CONTINUE  
          EM(L,L)=PARM(12)
11      CONTINUE
        RETURN
      END


      SUBROUTINE SELFD(I,J,R2,IOVPAR,EM,NE)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION PARM(13),EM(NE,5)
      EXTERNAL IOVPAR
!
        ID=IOVPAR(I,J,R2,PARM)
        DO 11 L=1,5
          DO 10 M=1,5
            EM(L,M)=0.0
10        CONTINUE
          EM(L,L)=PARM(11)
11      CONTINUE
        RETURN
      END

