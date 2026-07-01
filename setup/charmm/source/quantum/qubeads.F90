!     Dan T. Major, 09/2011
!     Questions to: majort@biu.ac.il
#if KEY_QUANTUM==1 || KEY_SQUANTM==1 || KEY_SCCDFTB==1 || KEY_GAMESSUK==1 || KEY_QCHEM==1 || KEY_QTURBO==1 /*quant*/

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE PIBGEN(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQ)
  use chm_kinds
  use number
  implicit none

  INTEGER :: NPIATM,NBEADSQ
  real(chm_real) CBDSX(NBEADSQ,NPIATM),CBDSY(NBEADSQ,NPIATM),CBDSZ(NBEADSQ,NPIATM)
  !
  !   Place all beads at origin
  CBDSX(1:NBEADSQ,1:NPIATM) = ZERO
  CBDSY(1:NBEADSQ,1:NPIATM) = ZERO
  CBDSZ(1:NBEADSQ,1:NPIATM) = ZERO

  RETURN
END SUBROUTINE PIBGEN

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Random cubical MC move
!  MC step is a multiple of De Broglie wavelength
SUBROUTINE MCMV(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQ,IBMOVE,NBMOVE,LAMBDA,LAMBDA2,PIMCMV)
  use chm_kinds
  use qub_m, only: ranumc
  implicit none

  !   Passed arguments
  INTEGER :: NPIATM,NBEADSQ,NBMOVE
  INTEGER :: IBMOVE(NBEADSQ)
  real(chm_real)  CBDSX(NBEADSQ,NPIATM),CBDSY(NBEADSQ,NPIATM),CBDSZ(NBEADSQ,NPIATM)
  real(chm_real)  LAMBDA(NPIATM),LAMBDA2(NPIATM),PIMCMV
  !   Local variables
  INTEGER :: I,J,KK

  DO J = 1,NBMOVE
     KK = IBMOVE(J)
     DO I = 1,NPIATM
        CBDSX(KK,I) = CBDSX(KK,I)+PIMCMV*(-LAMBDA(I)+RANUMC()*LAMBDA2(I))
        CBDSY(KK,I) = CBDSY(KK,I)+PIMCMV*(-LAMBDA(I)+RANUMC()*LAMBDA2(I))
        CBDSZ(KK,I) = CBDSZ(KK,I)+PIMCMV*(-LAMBDA(I)+RANUMC()*LAMBDA2(I))
        !  mDTM
        !           write(6,'(A,3I4,F10.5)')'x',i,j,kk,cbdsx(kk,i)
     ENDDO
  ENDDO

  RETURN
END SUBROUTINE MCMV

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!   Sample according to bisection algorithm (heat-bath transition rule)
!   Ref: Levy, P. Compositio Math. 7, 283, 1939.
!        Ceperley, D.M.; Pollock, E.L. Monte Carlo methods in theoretical
!        physics, page. 35. ETS Editrice, Pisa. 1992.
!
SUBROUTINE BISECT(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQ,IBMOVE,NBMOVE,LAMBDA,KLEV)
  use chm_kinds
  use qub_m, only: gausdev
  implicit none

  !   Passed arguments
  INTEGER :: NPIATM,NBEADSQ,NBMOVE,KLEV
  INTEGER :: IBMOVE(NBEADSQ)
  real(chm_real)  CBDSX(NBEADSQ,NPIATM),CBDSY(NBEADSQ,NPIATM),CBDSZ(NBEADSQ,NPIATM)
  real(chm_real)  LAMBDA(NPIATM)
  !   Local variables
  real(chm_real)  MIDPNTX,MIDPNTY,MIDPNTZ,RR,LBSECT,RTWO
  INTEGER :: I,J,K,KK,LEV
  INTEGER :: NUMBSECT,BBSECT,IBEAD,FBEAD,MBEAD

  RTWO = 0.5D0
  DO I = 1,NBMOVE
     KK = IBMOVE(I)
     DO LEV = KLEV,1,-1
        NUMBSECT = 2**(KLEV-LEV)
        LBSECT   = SQRT(DBLE(2**(LEV-1)))
        BBSECT   = 2**LEV
        IBEAD = KK
        FBEAD = IBEAD + BBSECT
        MBEAD = (IBEAD + FBEAD)/2
        DO J = 1,NUMBSECT
           IF (IBEAD.GT.NBEADSQ) IBEAD = IBEAD - NBEADSQ
           IF (FBEAD.GT.NBEADSQ) FBEAD = FBEAD - NBEADSQ
           IF (MBEAD.GT.NBEADSQ) MBEAD = MBEAD - NBEADSQ
           !               write(*,*)'beads',IBEAD,FBEAD,MBEAD,LEV,LBSECT
           DO K = 1,NPIATM
              MIDPNTX = (CBDSX(IBEAD,K)+CBDSX(FBEAD,K))*RTWO
              CBDSX(MBEAD,K) = GAUSDEV(LAMBDA(K)*LBSECT,MIDPNTX)
              MIDPNTY = (CBDSY(IBEAD,K)+CBDSY(FBEAD,K))*RTWO
              CBDSY(MBEAD,K) = GAUSDEV(LAMBDA(K)*LBSECT,MIDPNTY)
              MIDPNTZ = (CBDSZ(IBEAD,K)+CBDSZ(FBEAD,K))*RTWO
              CBDSZ(MBEAD,K) = GAUSDEV(LAMBDA(K)*LBSECT,MIDPNTZ)
              !     write(*,'(A,4F10.5)')'bisct',CBDSX(IBEAD,K),CBDSX(FBEAD,K),MIDPNTX,CBDSX(MBEAD,K)
           ENDDO
           IBEAD = IBEAD + BBSECT
           FBEAD = FBEAD + BBSECT
           MBEAD = MBEAD + BBSECT
        ENDDO
     ENDDO
     !        write(6,'(A,3I,2F10.5)')'x',i,j,kk,cbdsx(kk,i),midpntx
  ENDDO

  RETURN
END SUBROUTINE BISECT

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!   Sample according to the staging algorithm
!   Ref: Sprik, M.; Klein, M. L.; Chandler, D.
!        Phys. Rev. B 1985, 31, 4234
!
SUBROUTINE STAGING(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQ,IBMOVE,NBMOVE,LAMBDA,NBEADM)
  use chm_kinds
  use number, only : two
  use qub_m, only: gausdev
  implicit none

  ! Passed arguments
  INTEGER :: NPIATM,NBEADSQ,NBMOVE,NBEADM
  INTEGER :: IBMOVE(NBEADSQ)
  real(chm_real) CBDSX(NBEADSQ,NPIATM),CBDSY(NBEADSQ,NPIATM),CBDSZ(NBEADSQ,NPIATM),LAMBDA(NPIATM)
  ! Local variables
  real(chm_real) NXTPNTX,NXTPNTY,NXTPNTZ
  INTEGER :: I,J,II,JJ
  INTEGER :: IBEAD,FBEAD,JBEAD

  ! Place beads using staging coefficients and random gaussian displacement
  ! Sample NBEADS-1 beads between JBEAD and FBEAD
  DO II = 1,NBMOVE
  !         IBEAD = IBMOVE(II)
     DO I = 1,NPIATM
        IBEAD = IBMOVE(II)
        JBEAD = IBEAD - 1
        FBEAD = IBEAD + NBEADM - 1
        IF (JBEAD.LE.0) JBEAD = JBEAD + NBEADSQ
        IF (FBEAD.GT.NBEADSQ) FBEAD = FBEAD - NBEADSQ
        JJ = NBEADM
  ! Initialize first bead at origin
  !         CBDSX(IBEAD,I) = ZERO
        DO J = 1,NBEADM-1
           IF (IBEAD.GT.NBEADSQ) IBEAD = IBEAD - NBEADSQ
           IF (JBEAD.GT.NBEADSQ) JBEAD = JBEAD - NBEADSQ
           JJ = JJ - 1
           NXTPNTX = (CBDSX(JBEAD,I)*DBLE(JJ)+CBDSX(FBEAD,I))/DBLE(JJ+1)
           NXTPNTY = (CBDSY(JBEAD,I)*DBLE(JJ)+CBDSY(FBEAD,I))/DBLE(JJ+1)
           NXTPNTZ = (CBDSZ(JBEAD,I)*DBLE(JJ)+CBDSZ(FBEAD,I))/DBLE(JJ+1)
           CBDSX(IBEAD,I) = GAUSDEV(LAMBDA(I)*SQRT(two*DBLE(JJ)/(DBLE(JJ+1))),NXTPNTX)
           CBDSY(IBEAD,I) = GAUSDEV(LAMBDA(I)*SQRT(two*DBLE(JJ)/(DBLE(JJ+1))),NXTPNTY)
           CBDSZ(IBEAD,I) = GAUSDEV(LAMBDA(I)*SQRT(two*DBLE(JJ)/(DBLE(JJ+1))),NXTPNTZ)
  !            write(6,'(A,6I3,2F10.5)')'x',i,j,ibead,jbead,fbead,jj,
  !     &           cbdsx(ibead,i),nxtpntx
           IBEAD = IBEAD + 1
           JBEAD = JBEAD + 1
        ENDDO
     ENDDO
  ENDDO

  RETURN
  END SUBROUTINE STAGING

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!   Sample according to the staging algorithm
!   Ref: Sprik, M.; Klein, M. L.; Chandler, D.
!        Phys. Rev. B 1985, 31, 4234
!   This version performs open chain staging 
!   Ref: Morrone, J. A.; Srinivasan, V.; Sebastiani, D.; Car, R.
!        J. Chem. Phys. 2007, 126, 234504
!
SUBROUTINE OSTAGING(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQ,LAMBDA,NBEADM,EED,ISTR,ISTRVEC)

  use chm_kinds
  use number, only : zero,two
  use qub_m, only: gausdev
  implicit none

  ! Passed arguments
  INTEGER :: NPIATM,NBEADSQ,NBEADM
  real(chm_real) EED,SFAC   ! End-to-end bead-distance
  real(chm_real) CBDSX(NBEADSQ,NPIATM),CBDSY(NBEADSQ,NPIATM),CBDSZ(NBEADSQ,NPIATM),LAMBDA(NPIATM),ISTRVEC(4)
  logical :: ISTR   ! Flag for isotropic sampling. If true, will sample open chain along donor-acceptor axis
  ! Local variables
  real(chm_real) NXTPNTX,NXTPNTY,NXTPNTZ
  INTEGER :: I,J,JJ
  INTEGER :: IBEAD,FBEAD,JBEAD

  ! Place beads using staging coefficients and random gaussian displacement
  ! Sample NBEADS-1 beads between JBEAD and FBEAD
  ! Since open PI sample P+1 beads, we place the first bead at the origin and sample
  ! the remaining P beads. In the current implementation, the first bead is located
  ! at the classical position and is therefore not treated explicitly.
  DO I = 1,NPIATM   ! It is assumed that NPIATM=1, as momentum distribution is a one-particle property
     IBEAD = 2
     JBEAD = IBEAD - 1
     FBEAD = NBEADM
     JJ = NBEADM - 1
  ! Initialize first and last beads
     CBDSX(JBEAD,I) = ZERO
     CBDSY(JBEAD,I) = ZERO
     CBDSZ(JBEAD,I) = ZERO
     CBDSX(FBEAD,I) = GAUSDEV(LAMBDA(I)*SQRT(two*DBLE(JJ)),CBDSX(JBEAD,I))
     CBDSY(FBEAD,I) = GAUSDEV(LAMBDA(I)*SQRT(two*DBLE(JJ)),CBDSY(JBEAD,I))
     CBDSZ(FBEAD,I) = GAUSDEV(LAMBDA(I)*SQRT(two*DBLE(JJ)),CBDSZ(JBEAD,I))
     EED = SQRT(CBDSX(FBEAD,I)**2 + CBDSY(FBEAD,I)**2 + CBDSZ(FBEAD,I)**2)  ! End-2-end distance
     IF (ISTR) THEN
        ! ISTRVEC is isotropic vector: (vecx,vecy,vecz,length_vec)
        SFAC = EED / ISTRVEC(4)   ! Scaling factor
        CBDSX(FBEAD,I) = ISTRVEC(1)*SFAC
        CBDSY(FBEAD,I) = ISTRVEC(2)*SFAC
        CBDSZ(FBEAD,I) = ISTRVEC(3)*SFAC
!    write(*,*)'istr=',ISTRVEC(1),ISTRVEC(2),ISTRVEC(3),CBDSX(FBEAD,I),CBDSY(FBEAD,I),CBDSZ(FBEAD,I)
     ENDIF
     DO J = 2,NBEADM-1
        JJ = JJ - 1
        NXTPNTX = (CBDSX(JBEAD,I)*DBLE(JJ)+CBDSX(FBEAD,I))/DBLE(JJ+1)
        NXTPNTY = (CBDSY(JBEAD,I)*DBLE(JJ)+CBDSY(FBEAD,I))/DBLE(JJ+1)
        NXTPNTZ = (CBDSZ(JBEAD,I)*DBLE(JJ)+CBDSZ(FBEAD,I))/DBLE(JJ+1)
        CBDSX(IBEAD,I) = GAUSDEV(LAMBDA(I)*SQRT(two*DBLE(JJ)/(DBLE(JJ+1))),NXTPNTX)
        CBDSY(IBEAD,I) = GAUSDEV(LAMBDA(I)*SQRT(two*DBLE(JJ)/(DBLE(JJ+1))),NXTPNTY)
        CBDSZ(IBEAD,I) = GAUSDEV(LAMBDA(I)*SQRT(two*DBLE(JJ)/(DBLE(JJ+1))),NXTPNTZ)
        IBEAD = IBEAD + 1
        JBEAD = JBEAD + 1
  !            write(6,'(A,6I3,2F10.5)')'x',i,j,ibead,jbead,fbead,jj,
  !     &           cbdsx(ibead,i),nxtpntx
     ENDDO
  ENDDO

  RETURN
  END SUBROUTINE OSTAGING

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE FPICNT(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQ,NBEADSQQ)
  use chm_kinds
  use number
  implicit none

  INTEGER :: NPIATM,NBEADSQ,NBEADSQQ
  real(chm_real) CBDSX(NBEADSQQ,NPIATM),CBDSY(NBEADSQQ,NPIATM),CBDSZ(NBEADSQQ,NPIATM)
  !
  real(chm_real) XC,YC,ZC,RNBEADS
  INTEGER :: I,J,K
  LOGICAL :: QOSTAGE
  !
  !      real(chm_real) xtest

  QOSTAGE = .FALSE.
  IF (NBEADSQQ > NBEADSQ) QOSTAGE = .TRUE.    ! Open paths - centroid definition different
  RNBEADS = ONE/DBLE(NBEADSQ)
  DO I = 1,NPIATM
     XC = ZERO
     YC = ZERO
     ZC = ZERO
     K = 1
     IF (QOSTAGE) THEN
        K = 2
        XC = 0.5D0*(CBDSX(1,I)+CBDSX(NBEADSQQ,I))
        YC = 0.5D0*(CBDSY(1,I)+CBDSY(NBEADSQQ,I))
        ZC = 0.5D0*(CBDSZ(1,I)+CBDSZ(NBEADSQQ,I))
     ENDIF
     DO J = K,NBEADSQ
        XC = XC+CBDSX(J,I)
        YC = YC+CBDSY(J,I)
        ZC = ZC+CBDSZ(J,I)
     ENDDO
     XC = XC*RNBEADS
     YC = YC*RNBEADS
     ZC = ZC*RNBEADS
     DO J = 1,NBEADSQQ
        CBDSX(J,I) = CBDSX(J,I)-XC
        CBDSY(J,I) = CBDSY(J,I)-YC
        CBDSZ(J,I) = CBDSZ(J,I)-ZC
     ENDDO
     !            xtest = 0.0
     !            do j = 1,nbeadsq
     !               xtest = xtest+cbdsx(j,i)
     !               write(66,'(3F10.5)') cbdsx(j,i),cbdsy(j,i),cbdsz(j,i)
     !            enddo
     !            write(66,'(A,I4,F10.5)') 'center of x for i',i,xtest
  ENDDO
  !
  RETURN
END SUBROUTINE FPICNT

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE FPICNT2(CBDSX,CBDSY,CBDSZ,XBOLD,YBOLD,ZBOLD,NPIATM,NBEADSQ,NBMOVE,IBMOVE)
  use chm_kinds
  use number
  implicit none

  INTEGER :: NPIATM,NBEADSQ,NBMOVE,IBMOVE(NBEADSQ)
  real(chm_real) CBDSX(NBEADSQ,NPIATM),CBDSY(NBEADSQ,NPIATM),CBDSZ(NBEADSQ,NPIATM)
  real(chm_real) XBOLD(NBEADSQ,NPIATM),YBOLD(NBEADSQ,NPIATM),ZBOLD(NBEADSQ,NPIATM)
  !
  real(chm_real) XDEL(NPIATM),YDEL(NPIATM),ZDEL(NPIATM)
  real(chm_real) RNBMOVE
  INTEGER :: I,J,KK
  !      real(chm_real) xtest

  !  Enforce centroid constraint for the moved particles
  ! DTM testing enforcing for all particles
  RNBMOVE = ONE/NBMOVE
  DO I = 1,NPIATM
     XDEL(I) = ZERO
     YDEL(I) = ZERO
     ZDEL(I) = ZERO
     DO J = 1,NBMOVE
        KK = IBMOVE(J)
        XDEL(I) = XDEL(I)+ CBDSX(KK,I) - XBOLD(J,I)
        YDEL(I) = YDEL(I)+ CBDSY(KK,I) - YBOLD(J,I)
        ZDEL(I) = ZDEL(I)+ CBDSZ(KK,I) - ZBOLD(J,I)
     ENDDO
  ENDDO
  !
  DO I = 1,NPIATM
     XDEL(I) = XDEL(I)*RNBMOVE
     YDEL(I) = YDEL(I)*RNBMOVE
     ZDEL(I) = ZDEL(I)*RNBMOVE
     DO J = 1,NBMOVE
        KK = IBMOVE(J)
        CBDSX(KK,I) = CBDSX(KK,I) - XDEL(I)
        CBDSY(KK,I) = CBDSY(KK,I) - YDEL(I)
        CBDSZ(KK,I) = CBDSZ(KK,I) - ZDEL(I)
     ENDDO
     !         write(6,'(A,I,3F10.5)') 'moves:',i,xdel(i),ydel(I),zdel(I)
     !            xtest = 0.0
     !            do j = 1,nbeadsq
     !               xtest = xtest+cbdsx(j,i)
     !            enddo
     !            write(6,'(A,I,F10.5)') 'center of x for i',i,xtest
  ENDDO

  RETURN
END SUBROUTINE FPICNT2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE MASSPERT(CBDSX,CBDSY,CBDSZ,PCBDSX,PCBDSY,PCBDSZ,NPIATM,NBEADSQ,MSQRAT)
  !
  use chm_kinds
  implicit none
  !...  use number

  INTEGER :: NPIATM,NBEADSQ
  real(chm_real) MSQRAT(NPIATM)
  real(chm_real) CBDSX(NBEADSQ,NPIATM),CBDSY(NBEADSQ,NPIATM),CBDSZ(NBEADSQ,NPIATM), &
       PCBDSX(NBEADSQ,NPIATM),PCBDSY(NBEADSQ,NPIATM),PCBDSZ(NBEADSQ,NPIATM)
  !
  INTEGER :: I
  !            integer :: J
  !            real(chm_real) xtest

  DO I = 1,NPIATM
     !         write(*,*)'MSQRAT> ',MSQRAT(I)
     PCBDSX(1:NBEADSQ,I) = CBDSX(1:NBEADSQ,I)*MSQRAT(I)
     PCBDSY(1:NBEADSQ,I) = CBDSY(1:NBEADSQ,I)*MSQRAT(I)
     PCBDSZ(1:NBEADSQ,I) = CBDSZ(1:NBEADSQ,I)*MSQRAT(I)
     !            xtest = 0.0d0
     !            do j = 1,nbeadsq
     !               xtest = xtest+cbdsx(j,i)
     !            enddo
     !            write(6,'(A,I5,F10.5)') 'center of x for i',i,xtest
  ENDDO

  RETURN
END SUBROUTINE MASSPERT

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Write out quantized atoms (only beads) in pdb format
! Uses NMR format to write multiple models
SUBROUTINE BWRITE(IUNIT,X,Y,Z,CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQ,IPIATM,WRTBDS,NCONF)
  use chm_kinds
  use dimens_fcm
  use psf
  implicit none

  INTEGER IUNIT,NPIATM,NBEADSQ,IPIATM(NPIATM),NCONF
  real(chm_real)  X(*),Y(*),Z(*)
  real(chm_real) CBDSX(NBEADSQ,NPIATM),CBDSY(NBEADSQ,NPIATM),CBDSZ(NBEADSQ,NPIATM)
  !
  INTEGER :: I,J,JJ
  CHARACTER(len=8) ATYPEI
  LOGICAL WRTBDS

  IF (WRTBDS) THEN
     WRITE(IUNIT,'(A,I7)')'REMARK Path-Integral config # ',NCONF
     WRITE(IUNIT,'(A,I4)')'REMARK Number of beads ',NBEADSQ
     WRITE(IUNIT,'(A,I4)')'REMARK Number of PI atoms ',NPIATM
     DO I = 1,NPIATM
        WRITE(IUNIT,'(A)') 'MODEL'
        JJ = IPIATM(I)
        ! shift atom names when they exceed 3 characters
        IF (ATYPE(JJ)(4:4).EQ.' ') THEN
           ATYPEI=' '//ATYPE(JJ)(1:3)
        ELSE
           ATYPEI=ATYPE(JJ)
        ENDIF
        DO J = 1,NBEADSQ
           WRITE(IUNIT,'(A,I5,1X,A4,1X,A4,2X,I4,3X,3F8.3,2F6.2,6X,A4)') &
                'ATOM  ',(I-1)*NBEADSQ+J,ATYPEI,'BEAD',J, &
                X(JJ)+CBDSX(J,I),Y(JJ)+CBDSY(J,I),Z(JJ)+CBDSZ(J,I),1.0,0.0,'BEAD'
        ENDDO
        WRITE(IUNIT,'(A)') 'ENDMDL'
     ENDDO
  ELSE
     WRITE(IUNIT,'(A)') 'END'
     CLOSE(IUNIT)
  ENDIF

  RETURN
END SUBROUTINE BWRITE

! The following routine is obsolete, but neatly written...
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE PIBGEN2(CBDSX,CBDSY,CBDSZ,AMASS,NPIATM,NBEADSQ,NBEADSQQ,IPIATM)
  use chm_kinds
  use consta
  use number
  use qub_m, only: ranumc
  implicit none

  INTEGER NPIATM,NBEADSQ,NBEADSQQ,IPIATM(NPIATM)
  real(chm_real) CBDSX(NBEADSQQ,NPIATM),CBDSY(NBEADSQQ,NPIATM),CBDSZ(NBEADSQQ,NPIATM),AMASS(*)
  !
  real(chm_real) RHV,RLT,PCENT,DANGLE,ANGLE,R
  INTEGER :: I,IA,IB,IC,J
  !
  RHV = 0.05D0
  RLT = 0.30D0
  PCENT = 0.15D0
  DANGLE = TWOPI/DBLE(NBEADSQQ)
  DO I = 1,NPIATM
     IA = INT(three*RANUMC())+1
     IB = INT(three*RANUMC())+1
     IF(IA.EQ.IB) THEN
        IB = IA+1
        IF(IB.GT.3) IB = IA-1
     ENDIF
     IF(IA+IB.EQ.3) IC = 3
     IF(IA+IB.EQ.4) IC = 2
     IF(IA+IB.EQ.5) IC = 1
     ANGLE = ZERO
     R = RLT
     IF(AMASS(IPIATM(I)).GT.TWO) R = RHV
     DO J = 1,NBEADSQ
        CBDSX(J,I) = ZERO
        CBDSY(J,I) = ZERO
        CBDSZ(J,I) = ZERO
        select case (IA)
        case (1)
           CBDSX(J,I) = R*COS(ANGLE)*(one+(two*RANUMC()-one)*PCENT)
        case (2)
           CBDSY(J,I) = R*COS(ANGLE)*(one+(two*RANUMC()-one)*PCENT)
        case (3)
           CBDSZ(J,I) = R*COS(ANGLE)*(one+(two*RANUMC()-one)*PCENT)
        end select
        select case (IB)
        case (1)
           CBDSX(J,I) = R*SIN(ANGLE)*(one+(two*RANUMC()-one)*PCENT)
        case (2)
           CBDSY(J,I) = R*SIN(ANGLE)*(one+(two*RANUMC()-one)*PCENT)
        case (3)
           CBDSZ(J,I) = R*SIN(ANGLE)*(one+(two*RANUMC()-one)*PCENT)
        end select
        select case (IC)
        case (1)
           CBDSX(J,I) = (two*RANUMC()-one)*PCENT
           CBDSX(J,I) = zero
        case (2)
           CBDSY(J,I) = (two*RANUMC()-one)*PCENT
           CBDSY(J,I) = zero
        case (3)
           CBDSZ(J,I) = (two*RANUMC()-one)*PCENT
           CBDSZ(J,I) = zero
        end select
        ANGLE = ANGLE+DANGLE
        !   mDTM Print bead coordinates
        !            write(*,'(A,I7,3F10.5)') 'Bead: ',J,CBDSX(J,I),CBDSY(J,I),CBDSZ(J,I)
     ENDDO
  ENDDO
  !
  !  Center around centroid position
  !
  CALL FPICNT(CBDSX,CBDSY,CBDSZ,NPIATM,NBEADSQ,NBEADSQQ)
  !
  RETURN
END SUBROUTINE PIBGEN2

#else /* (quant)*/
SUBROUTINE NULL_SQUB
  RETURN
END SUBROUTINE NULL_SQUB
#endif /* (quant)*/


