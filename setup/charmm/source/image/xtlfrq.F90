module xtlfrq_m
  implicit none

contains

SUBROUTINE XTLFRQ(X,Y,Z,NFREQ,DDF,FREQ,EVAL,EVEC,QMASS,QXGRP)
  !
  !     This subroutine calculates the second derivative matrix for a
  !     crystal and then diagonalises it to give the lattice vibrations.
  !
  !     M. J. Field - April 1986.
  !     Updated October 1990.
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use exfunc
  use number
  !
  use bases_fcm
  use consta
  use deriv
  use energym
  use hbondm
  use memory
  use image
  use inbnd
  use parallel,only: gcomb
  use psf
  use stream
  implicit none
  !
  !     Passed variables.
  !
  real(chm_real),allocatable,dimension(:) :: DDM, DD1X, DD2X
  real(chm_real),allocatable,dimension(:) :: DDF1
  real(chm_real),allocatable,dimension(:) :: work1, work2, work3, work4, work5, work6, work7
  INTEGER NFREQ
  LOGICAL QMASS, QXGRP
  real(chm_real) :: DDF(:), EVAL(*), EVEC(*), FREQ(*)
  real(chm_real) :: X(:), Y(:), Z(:)
  !
  !     Local variables.
  !
  INTEGER I, IER, NSTORE, XSTORE
  LOGICAL QERROR
  !
  !     Pointers.
  !
  !
  !     Check that a crystal is defined.
  !
  QERROR = (XNSYMM.NE.1) .OR. (NTRANS.LE.0)
  IF (QERROR) CALL WRNDIE(-5,'<XTLFRQ>', &
       'A crystal has not been defined.')
  !
  !     Check that there is no image patching.
  !
  QERROR = (NIMBON .GT. 0) .OR. (NIMANG .GT. 0) .OR. &
       (NIMDIH .GT. 0) .OR. (NIMIMP .GT. 0) .OR. &
       (NIMHB  .GT. 0)
  IF (QERROR) CALL WRNDIE(-5,'<XTLFRQ>','IMPATCH not allowed.')
  !
  IF (.NOT.(QXGRP)) THEN
     !
     !       Update the image atom non-bond lists.
     !
#if KEY_IMCUBES==1
     CALL XNBLST(lbycbim)
#else /**/
     CALL XNBLST
#endif 
     !
     !       Allocate scratch space.
     !
     call chmalloc('xtlfrq.src','XTLFRQ','DDM',NATOM,crl=DDM)
     call chmalloc('xtlfrq.src','XTLFRQ','DD1X',6*XATIM,crl=DD1X)
     call chmalloc('xtlfrq.src','XTLFRQ','DD2X',6*XNNNB,crl=DD2X)
     dd1x = zero
     CALL FILDDM(DDM,AMASS,NATOM,.FALSE.)
  ENDIF
  !
  call chmalloc('xtlfrq.src','XTLFRQ','WORK1',NFREQ+1,crl=WORK1)
  call chmalloc('xtlfrq.src','XTLFRQ','WORK2',NFREQ+1,crl=WORK2)
  call chmalloc('xtlfrq.src','XTLFRQ','WORK3',NFREQ+1,crl=WORK3)
  call chmalloc('xtlfrq.src','XTLFRQ','WORK4',NFREQ+1,crl=WORK4)
  call chmalloc('xtlfrq.src','XTLFRQ','WORK5',NFREQ+1,crl=WORK5)
  call chmalloc('xtlfrq.src','XTLFRQ','WORK6',NFREQ+1,crl=WORK6)
  call chmalloc('xtlfrq.src','XTLFRQ','WORK7',NFREQ+1,crl=WORK7)
  !
  !     Calculate the energy, forces and fill the SD element arrays.
  !
  NSTORE = NTRANS
  XSTORE = XNSYMM
  NTRANS = 0
  XNSYMM = 0
  CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1,NFREQ,DDF)
  IF(PRNLEV.GE.2) CALL PRINTE(OUTU,EPROP,ETERM,'VIBR','ENR', &
       .TRUE.,0,ZERO,ZERO, .TRUE.)
#if KEY_PARALLEL==1
  CALL VDGBR(DX,DY,DZ,1)
  CALL GCOMB(DDF,(NFREQ*(NFREQ+1))/2)
#endif 
  NTRANS = NSTORE
  XNSYMM = XSTORE
  !
  !     Calculate the crystal derivatives.
  !
  IF (QXGRP) THEN
     !
     !       Reorder the second derivative matrix.
     !
     call chmalloc('xtlfrq.src','XTLFRQ','DDF1',(NFREQ*(NFREQ+1))/2,crl=DDF1)

     DDF1(1:(NFREQ*(NFREQ+1))/2 ) = DDF(1:(NFREQ*(NFREQ+1))/2 )
     CALL XTLFRR ( DDF, DDF1, NFREQ )
     call chmdealloc('xtlfrq.src','XTLFRQ','DDF1',(NFREQ*(NFREQ+1))/2,crl=DDF1)
     !
     !       Calculate the interactions.
     !
     CALL EXTAL2(ETERM(IMELEC),ETERM(IMVDW),0,DDF)

     EPROP(EPOT) = EPROP(EPOT) + ETERM(IMVDW) + ETERM(IMELEC)
     CALL XTLFRM(DDF)
     !
     !       Diagonalise.
     !
     IER=0
     CALL EIGRS(DDF,NFREQ,1,EVAL,EVEC,NFREQ,IER)
     IF(IER.GT.0 .AND. WRNLEV.GE.2) WRITE (OUTU,'(A,I5)') &
          ' XTLFRQ> Possible diagonalisation error. IER = ',IER
  ELSE
     CALL ECRYS2(DD1X,DD2X)
     !
     !       Build the full derivative matrix ready for diagonalisation.
     !
     CALL MAKXSD(NATOM,NTRANS,XNSYMM,XATIM,DD1X,DD2X, &
          DDF,DDM,XINBLO,XJNB,QMASS)
     !
     !       Diagonalise.
     !
     CALL DIAGQ(NFREQ,NFREQ,DDF,EVEC,WORK1,WORK2, &
          WORK3,WORK4,EVAL,WORK5, &
          WORK6,WORK7,0)
  ENDIF
  !
  !     Print the energy.
  !
  IF(PRNLEV.GE.2) CALL PRINTE(OUTU,EPROP,ETERM,'VIBR','ENR', &
       .TRUE.,0,ZERO,ZERO,.TRUE.)
  !
  !     Calculate the frequencies.
  !
  DO I=1,NFREQ
     FREQ(I)  = CNVFRQ * SQRT(ABS(EVAL(I)))
     IF (EVAL(I) .LT. ZERO) FREQ(I) = -FREQ(I)
  end DO
  !
  IF(PRNLEV.GE.2) THEN
     WRITE (OUTU,'(/15X,''Diagonalization Completed'')')
     WRITE (OUTU,'(/15X,''Frequencies''/)')
     WRITE (OUTU,'(5(I4,F12.6))') (I,FREQ(I),I = 1,NFREQ)
  ENDIF
  !
  !     Free storage space.
  !
  call chmdealloc('xtlfrq.src','XTLFRQ','DDM',NATOM,crl=DDM)
  call chmdealloc('xtlfrq.src','XTLFRQ','DD1X',6*XATIM,crl=DD1X)
  call chmdealloc('xtlfrq.src','XTLFRQ','DD2X',6*XNNNB,crl=DD2X)
  call chmdealloc('xtlfrq.src','XTLFRQ','WORK1',NFREQ+1,crl=WORK1)
  call chmdealloc('xtlfrq.src','XTLFRQ','WORK2',NFREQ+1,crl=WORK2)
  call chmdealloc('xtlfrq.src','XTLFRQ','WORK3',NFREQ+1,crl=WORK3)
  call chmdealloc('xtlfrq.src','XTLFRQ','WORK4',NFREQ+1,crl=WORK4)
  call chmdealloc('xtlfrq.src','XTLFRQ','WORK5',NFREQ+1,crl=WORK5)
  call chmdealloc('xtlfrq.src','XTLFRQ','WORK6',NFREQ+1,crl=WORK6)
  call chmdealloc('xtlfrq.src','XTLFRQ','WORK7',NFREQ+1,crl=WORK7)
  !
  RETURN
END SUBROUTINE XTLFRQ

SUBROUTINE MAKXSD(NATOM,NTRANS,XNSYMM,XATIM,DD1X,DD2X,DDF, &
     DDM,XINBLO,XJNB,QMASS)
  !
  !     This subroutine constructs the crystal second derivative
  !     matrix when the wave vector ,k, is zero.
  !
  use chm_kinds
#if KEY_DEBUG==1
  use stream
#endif 
  implicit none
  !
  INTEGER   XJNB(*)
  INTEGER   NATOM, NTRANS, XATIM, XINBLO(*), XNSYMM
  LOGICAL   QMASS
  real(chm_real)    DD2X(*)
  real(chm_real)    DDF(*), DDM(*), DD1X(*)
  !
  INTEGER   I, IIADD, IJADD, IPT, IPTX, ITEMP, I1, I2, I3
  INTEGER   J, JPR, J1, J2, J3, NATOM3, NB, NPR
  real(chm_real)    MI, MIJ
  !
  real(chm_real), PARAMETER :: HALF = 0.5D0
  !
  NATOM3 = 3 * NATOM
  !
  !     Do the XNSYMM .eq. 1 option first.
  !
  IF (XNSYMM .EQ. 1) THEN
     !
     !       Add in the diagonal terms to the matrix.
     !
     IIADD = 0
     IPT   = 1
     DO I=1,NATOM
        I3 = NATOM3 - I*3 + 2
        DDF(IPT)     = DDF(IPT)     + DD1X(IIADD + 1)
        DDF(IPT + 1) = DDF(IPT + 1) + DD1X(IIADD + 2)
        DDF(IPT + 2) = DDF(IPT + 2) + DD1X(IIADD + 4)
        IPT = IPT + I3 + 1
        DDF(IPT)     = DDF(IPT)     + DD1X(IIADD + 3)
        DDF(IPT + 1) = DDF(IPT + 1) + DD1X(IIADD + 5)
        IPT = IPT + I3
        DDF(IPT)     = DDF(IPT)     + DD1X(IIADD + 6)
        IIADD = IIADD + 6
        IPT   = IPT + I3 - 1
     enddo
#if KEY_DEBUG==1
     IF(PRNLEV.GE.2) WRITE(OUTU,'(A)') ' MAKXSD> Done Diagonal Terms.'
#endif 
     !
     !       Determine the off-diagonal contribution to the matrix.
     !
     IJADD = 0
     ITEMP = 0
     NB   = 0
     DO I=1,XATIM
        NPR   = XINBLO(I) - ITEMP
        ITEMP = XINBLO(I)
        I1    = I - ((I-1)/NATOM)*NATOM
        IF (NPR .GT. 0) THEN
           DO JPR=1,NPR
              NB  = NB + 1
              J1  = XJNB(NB)
              IF (J1 .LT. 0) J1 = -J1
              IF (I1 .EQ. J1) THEN
                 I2  = 3*(I1 - 1)
                 I3  = NATOM3 - 3*I1 + 2
                 IPT = 1 + I2*NATOM3 - (I2*(I2 - 1))/2
                 DDF(IPT)     = DDF(IPT)     + DD2X(IJADD + 1)
                 DDF(IPT + 1) = DDF(IPT + 1) + DD2X(IJADD + 2)
                 DDF(IPT + 2) = DDF(IPT + 2) + DD2X(IJADD + 4)
                 IPT = IPT + I3 + 1
                 DDF(IPT)     = DDF(IPT)     + DD2X(IJADD + 3)
                 DDF(IPT + 1) = DDF(IPT + 1) + DD2X(IJADD + 5)
                 IPT = IPT + I3
                 DDF(IPT)     = DDF(IPT)     + DD2X(IJADD + 6)
              ELSE
                 IF (J1 .GT. I1) THEN
                    I2 = I1
                    J2 = J1
                 ELSE
                    I2 = J1
                    J2 = I1
                 ENDIF
                 I3   = 3*I2 - 2
                 IPTX = ((I3 - 1)*(NATOM3 + NATOM3 - I3)) / 2
                 J3   = 3*J2 - 2
                 IPT  = IPTX + J3
                 DDF(IPT)     = DDF(IPT)     + HALF * DD2X(IJADD + 1)
                 DDF(IPT + 1) = DDF(IPT + 1) + HALF * DD2X(IJADD + 2)
                 DDF(IPT + 2) = DDF(IPT + 2) + HALF * DD2X(IJADD + 4)
                 IPT  = IPT + NATOM3 - I3
                 DDF(IPT)     = DDF(IPT)     + HALF * DD2X(IJADD + 2)
                 DDF(IPT + 1) = DDF(IPT + 1) + HALF * DD2X(IJADD + 3)
                 DDF(IPT + 2) = DDF(IPT + 2) + HALF * DD2X(IJADD + 5)
                 IPT  = IPT + NATOM3 - I3 - 1
                 DDF(IPT)     = DDF(IPT)     + HALF * DD2X(IJADD + 4)
                 DDF(IPT + 1) = DDF(IPT + 1) + HALF * DD2X(IJADD + 5)
                 DDF(IPT + 2) = DDF(IPT + 2) + HALF * DD2X(IJADD + 6)
              ENDIF
              IJADD = IJADD + 6
           enddo
        ENDIF
     enddo
     !
  ELSE
     CALL WRNDIE(-5,'<MAKXSD>','XNSYMM > 1 code unavailable.')
  ENDIF
#if KEY_DEBUG==1
  IF(PRNLEV.GE.2) WRITE(OUTU,'(A)') &
       ' MAKXSD> Done Off-diagonal Terms.'
#endif 

  !
  !     Mass-weight the matrix.
  !
  IF (QMASS) THEN
     IPT = 0
     DO I=1,NATOM
        MI = DDM(I)
        I2 = 4
        DO I1=1,3
           I2  = I2 - 1
           MIJ = MI * MI
           DO J1=1,I2
              IPT = IPT + 1
              DDF(IPT) = MIJ * DDF(IPT)
           enddo
           DO J=(I+1),NATOM
              MIJ = MI * DDM(J)
              DO J1=1,3
                 IPT = IPT + 1
                 DDF(IPT) = MIJ * DDF(IPT)
              enddo
           enddo
        enddo
     enddo
#if KEY_DEBUG==1
     IF(PRNLEV.GE.2) WRITE(OUTU,'(A)') &
          ' MAKXSD> Done the Mass Weighting.'
#endif 

  ENDIF
  !
  RETURN
END SUBROUTINE MAKXSD

SUBROUTINE PHONON(X,Y,Z,NSTEP,KSTRT,KSTP,NFREQ,DDFC,DDF, &
     FREQ,EVAL,EVEC,QMASS)
  !
  !     This subroutine calculates the second derivative matrix for a
  !     crystal and then diagonalises it to give the lattice vibrations.
  !     Any value of the wave vector, k, can be accommodated.
  !
  !     M. J. Field - April 1986.
  !     Updated October 1990.
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use exfunc
  use number
  !
  use bases_fcm
  use consta
  use deriv
  use energym
  use hbondm
  use memory
  use image
  use inbnd
  use parallel,only: gcomb
  use psf
  use stream
  implicit none
  !
  !     Passed variables.
  !
  real(chm_real),allocatable,dimension(:) :: DDM
  real(chm_real),allocatable,dimension(:) :: DD1X
  real(chm_real),allocatable,dimension(:) :: DD2X
  real(chm_real),allocatable,dimension(:) :: WORK
  complex(chm_cmpx) :: DDFC(*), EVEC(*)
  INTEGER    NFREQ,NSTEP
  LOGICAL    QMASS
  real(chm_real) :: DDF(:), EVAL(*), FREQ(*), KSTP(3), KSTRT(3)
  real(chm_real) :: X(:), Y(:), Z(:)
  !
  !     Local variables.
  !
  INTEGER    I, J, IER, IPT, ISPACE, ISTEP, JPT
  INTEGER    NATOM3, NSTORE, XSTORE
  LOGICAL    QERROR
  real(chm_real)     K(3)
  !
  !     Pointers.
  !
  INTEGER    ADDFRQ, ADDVEC
  !
  !     Check that a P1 crystal is defined.
  !
  QERROR = (XNSYMM.NE.1) .OR. (NTRANS.LE.0)
  IF (QERROR) CALL WRNDIE(-5,'<PHONON>', &
       'A P1 symmetry crystal has not been defined.')
  !
  !     Check that there is no image patching.
  !
  QERROR = (NIMBON .GT. 0) .OR. (NIMANG .GT. 0) .OR. &
       (NIMDIH .GT. 0) .OR. (NIMIMP .GT. 0) .OR. &
       (NIMHB  .GT. 0)
  IF (QERROR) CALL WRNDIE(-5,'<PHONON>','IMPATCH not allowed.')
  !
  !     Update the image atom non-bond lists.
  !
#if KEY_IMCUBES==1
#if KEY_IMCUBES==1
  CALL XNBLST(lbycbim)              
#endif
#else /**/
  CALL XNBLST
#endif 
  !
  !     Allocate scratch space.
  !
  NATOM3 = 3 * NATOM
  !
  call chmalloc('xtlfrq.src','PHONON','DDM',NATOM,crl=DDM)
  call chmalloc('xtlfrq.src','PHONON','DD1X',6*XATIM,crl=DD1X)
  call chmalloc('xtlfrq.src','PHONON','DD2X',6*XNNNB,crl=DD2X)
  call chmalloc('xtlfrq.src','PHONON','WORK',6*NATOM3,crl=WORK)
  dd1x = zero
  CALL FILDDM(DDM,AMASS,NATOM,.FALSE.)
  !
  !     Calculate the energy, forces and fill the SD element arrays.
  !
  NSTORE = NTRANS
  XSTORE = XNSYMM
  NTRANS = 0
  XNSYMM = 0
  CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1,NATOM3,DDF)
#if KEY_PARALLEL==1
  CALL VDGBR(DX,DY,DZ,1)
  CALL GCOMB(DDF,(NATOM3*(NATOM3+1))/2)
#endif 
  NTRANS = NSTORE
  XNSYMM = XSTORE
  !
  !     Calculate the crystal derivatives.
  !
  CALL ECRYS2(DD1X,DD2X)
  !
  !     Print the energy.
  !
  IF(PRNLEV.GE.2) CALL PRINTE(OUTU,EPROP,ETERM,'VIBR','ENR', &
       .TRUE.,0,ZERO,ZERO,.TRUE.)
  !
  !     Loop over the wave vector values.
  !
  DO I = 1,3
     K(I) = TWOPI*(KSTRT(I)-KSTP(I))
  enddo
  DO ISTEP = 1,NSTEP
     DO I = 1,3
        K(I) = K(I)+TWOPI*KSTP(I)
     enddo
     !
     !       Initialise the complex matrix for this k-value. It is to be noted
     !       that the matrices are stored in different forms (because of EIGCH)
     !
     IPT = 0
     DO I = 1,NATOM3
        ISPACE = NATOM3
        JPT    = I
        DO J = 1,I
           IPT       = IPT + 1
           DDFC(IPT) = cmplx(DDF(JPT),ZERO,chm_cmpx)
           ISPACE    = ISPACE - 1
           JPT       = JPT + ISPACE
        enddo
     enddo
     !
     !       Build the full derivative matrix ready for diagonalisation.
     !
     CALL MAKPSD(NATOM,XNSYMM,XATIM,XNOP,XNA,XNB,XNC,K, &
          DD1X,DD2X,DDFC,DDM, &
          XINBLO,XJNB,QMASS)
     !
     !       Determine array addresses.
     !
     ADDFRQ=NPHONS*(ISTEP-1)
     ADDVEC=NPHONS*NPHONS*(ISTEP-1)
     !
     !       Diagonalise the completed matrix to determine the normal modes
     !       and the eigenvalues and then print the frequencies out.
     !
     IER = 0
     EVAL(1:natom3) = ZERO
     CALL EIGCH(DDFC,NATOM3,1,EVAL(ADDFRQ+1),EVEC(ADDVEC+1),NATOM3, &
          WORK,IER)

     IF (.NOT.(IER .EQ. 0) .AND. WRNLEV.GE.2) THEN
        WRITE (OUTU,'(A,A,I5,A)') &
             'PHONON> Probable diagonalisation error.', &
             'EIGCH error number = ',IER,'.'
     ENDIF
     !
     DO I = 1,NATOM3
        FREQ(I+ADDFRQ)  = CNVFRQ * SQRT(ABS(EVAL(I+ADDFRQ)))
        IF (EVAL(I+ADDFRQ) .LT. ZERO) FREQ(I+ADDFRQ) = -FREQ(I &
             +ADDFRQ)
     enddo
     !
     IF(PRNLEV.GE.2) THEN
        WRITE (OUTU,'(/15X,''Diagonalization Completed'')')
        WRITE (OUTU,'(/15X,''Frequencies for K = '',3F8.4,/)') &
             ((K(I) / TWOPI), I = 1,3)
        WRITE (OUTU,'(5(I4,F12.6))') (I,FREQ(I+ADDFRQ),I = 1,NATOM3)
     ENDIF
  enddo
  !
  !     Free storage space.
  !
  call chmdealloc('xtlfrq.src','PHONON','DDM',NATOM,crl=DDM)
  call chmdealloc('xtlfrq.src','PHONON','DD1X',6*XATIM,crl=DD1X)
  call chmdealloc('xtlfrq.src','PHONON','DD2X',6*XNNNB,crl=DD2X)
  call chmdealloc('xtlfrq.src','PHONON','WORK',6*NATOM3,crl=WORK)
  !
  RETURN
END SUBROUTINE PHONON

SUBROUTINE MAKPSD(NATOM,XNSYMM,XATIM,XNOP,XNA,XNB,XNC,K, &
     DD1X,DD2X,DDFC,DDM,XINBLO,XJNB,QMASS)
  !
  !     This subroutine constructs the crystal second derivative
  !     matrix for a general value of the wave vector ,k.
  !
  use chm_kinds
  use number,only:half,zero
  implicit none
  !
  complex(chm_cmpx) :: DDFC(*)
  INTEGER    XJNB(*), XNA(*), XNB(*), XNC(*), XNOP(*)
  INTEGER    NATOM, XATIM, XINBLO(*), XNSYMM
  LOGICAL    QMASS
  real(chm_real)     DD2X(*)
  real(chm_real)     DDM(*), DD1X(*), K(3)
  !
  complex(chm_cmpx) :: CFACT1, CFACT2
  INTEGER    I, IIADD, IJADD, IPT, ITEMP, ITRANS
  INTEGER    I1, I2, I3, J, JPR, J1, J2, J3, NB, NPR
  real(chm_real)     A, B, C, DOTKR, MI, MIJ
  !
  !     Do the XNSYMM .eq. 1 option first.
  !
  IF (XNSYMM .EQ. 1) THEN
     !
     !       Add in the diagonal terms to the matrix.
     !
     IIADD = 0
     IPT   = 1
     DO I = 1,NATOM
        I3 = 3 * (I - 1)
        DDFC(IPT)     = DDFC(IPT)     + DD1X(IIADD + 1)
        IPT = IPT + I3 + 1
        DDFC(IPT)     = DDFC(IPT)     + DD1X(IIADD + 2)
        DDFC(IPT + 1) = DDFC(IPT + 1) + DD1X(IIADD + 3)
        IPT = IPT + I3 + 2
        DDFC(IPT)     = DDFC(IPT)     + DD1X(IIADD + 4)
        DDFC(IPT + 1) = DDFC(IPT + 1) + DD1X(IIADD + 5)
        DDFC(IPT + 2) = DDFC(IPT + 2) + DD1X(IIADD + 6)
        IIADD = IIADD + 6
        IPT   = IPT + I3 + 6
     enddo
     !
     !       Determine the off-diagonal contribution to the matrix.
     !
     IJADD = 0
     ITEMP = 0
     NB    = 0
     DO I = 1,XATIM
        NPR   = XINBLO(I) - ITEMP
        ITEMP = XINBLO(I)
        I1    = I - ((I-1)/NATOM)*NATOM
        IF (NPR .GT. 0) THEN
           !
           !           Determine the complex factor.
           !
           ITRANS = (I - 1) / NATOM
           A = XNA(ITRANS)
           B = XNB(ITRANS)
           C = XNC(ITRANS)
           DOTKR  = K(1) * A + K(2) * B + K(3) * C
           CFACT1 = EXP(cmplx(ZERO,DOTKR,chm_cmpx))
           !
           DO JPR = 1,NPR
              NB  = NB + 1
              J1  = XJNB(NB)
              IF (J1 .LT. 0) J1 = -J1
              IF (I1 .EQ. J1) THEN
                 I3  = 3 * (I1 - 1)
                 IPT = ((I3 + 1) * (I3 + 2)) / 2
                 DDFC(IPT)   = DDFC(IPT)     + CFACT1 * DD2X(IJADD + 1)
                 IPT = IPT + I3 + 1
                 DDFC(IPT)   = DDFC(IPT)     + CFACT1 * DD2X(IJADD + 2)
                 DDFC(IPT+1) = DDFC(IPT + 1) + CFACT1 * DD2X(IJADD + 3)
                 IPT = IPT + I3 + 2
                 DDFC(IPT)   = DDFC(IPT)     + CFACT1 * DD2X(IJADD + 4)
                 DDFC(IPT+1) = DDFC(IPT + 1) + CFACT1 * DD2X(IJADD + 5)
                 DDFC(IPT+2) = DDFC(IPT + 2) + CFACT1 * DD2X(IJADD + 6)
              ELSE
                 IF (J1 .GT. I1) THEN
                    I2 = J1
                    J2 = I1
                    CFACT2 = HALF * CONJG(CFACT1)
                 ELSE
                    I2 = I1
                    J2 = J1
                    CFACT2 = HALF * CFACT1
                 ENDIF
                 I3  = 3 * (I2 - 1)
                 J3  = 3 * (J2 - 1)
                 IPT = (I3 * (I3 + 1)) / 2 + J3 + 1
                 DDFC(IPT)   = DDFC(IPT)     + CFACT2 * DD2X(IJADD + 1)
                 DDFC(IPT+1) = DDFC(IPT + 1) + CFACT2 * DD2X(IJADD + 2)
                 DDFC(IPT+2) = DDFC(IPT + 2) + CFACT2 * DD2X(IJADD + 4)
                 IPT  = IPT + I3 + 1
                 DDFC(IPT)   = DDFC(IPT)     + CFACT2 * DD2X(IJADD + 2)
                 DDFC(IPT+1) = DDFC(IPT + 1) + CFACT2 * DD2X(IJADD + 3)
                 DDFC(IPT+2) = DDFC(IPT + 2) + CFACT2 * DD2X(IJADD + 5)
                 IPT  = IPT + I3 + 2
                 DDFC(IPT)   = DDFC(IPT)     + CFACT2 * DD2X(IJADD + 4)
                 DDFC(IPT+1) = DDFC(IPT + 1) + CFACT2 * DD2X(IJADD + 5)
                 DDFC(IPT+2) = DDFC(IPT + 2) + CFACT2 * DD2X(IJADD + 6)
              ENDIF
              IJADD = IJADD + 6
           enddo
        ENDIF
     enddo
     !
  ELSE
     CALL WRNDIE(-5,'<MAKPSD>','XNSYMM > 1 code unavailable.')
  ENDIF
  !
  !     Mass-weight the matrix.
  !
  IF (QMASS) THEN
     IPT = 0
     DO I = 1,NATOM
        MI = DDM(I)
        DO I1 = 1,3
           DO J  = 1,(I - 1)
              MIJ = MI * DDM(J)
              DO J1 = 1,3
                 IPT = IPT + 1
                 DDFC(IPT) = MIJ * DDFC(IPT)
              enddo
           enddo
           MIJ = MI * MI
           DO J1 = 1,I1
              IPT = IPT + 1
              DDFC(IPT) = MIJ * DDFC(IPT)
           enddo
        enddo
     enddo
  ENDIF
  !
  RETURN
END SUBROUTINE MAKPSD

SUBROUTINE XNBLST( &
#if KEY_IMCUBES==1
     lbycbim &             
#endif
     )
  !
  !     This subroutine modifies the image non-bond lists so
  !     that they are more suitable for calculation of the second
  !     derivatives of a crystal or image system. The charge and
  !     atom indexing arrays are also expanded.
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use exfunc
  !
  use bases_fcm
  use memory
  use image
  use psf
  implicit none
  !
  INTEGER  I, IATOM, ITRAN, NBGUES
  LOGICAL  QFREE
#if KEY_IMCUBES==1
  logical lbycbim              
#endif
  !
  !     Deallocate space for last lists.
  !
  QFREE = (XNNNB  .GT. 0)      .AND. &
       (allocated(xinblo)) .AND. (allocated(XJNB)) .AND. &
       (allocated(XMATPT)) .AND. (allocated(XMATTR))
  !
  IF (QFREE) THEN
     call chmdealloc('xtlfrq.src','XNBLST','XINBLO',XATIM,intg=XINBLO)
     call chmdealloc('xtlfrq.src','XNBLST','XJNB',XNNNB,intg=XJNB)
     call chmdealloc('xtlfrq.src','XNBLST','XMATPT',NTRANS,intg=XMATPT)
     call chmdealloc('xtlfrq.src','XNBLST','XMATTR',XATIM,intg=XMATTR)
  ENDIF
  !
  !     Allocate space for expanded lists.
  !
  XATIM  = NATOM * (NTRANS + 1)
  NBGUES = 2 * BIMAG%NIMNB + BIMAG%NIMNBS
  !
  call chmalloc('xtlfrq.src','XNBLST','XINBLO',XATIM,intg=XINBLO)
  call chmalloc('xtlfrq.src','XNBLST','XJNB',NBGUES,intg=XJNB)
  call chmalloc('xtlfrq.src','XNBLST','XMATPT',NTRANS,intg=XMATPT)
  call chmalloc('xtlfrq.src','XNBLST','XMATTR',XATIM,intg=XMATTR)
  !
  CALL XNBLS2(NATOM,XNNNB,NTRANS,IMINV,BIMAG%IMATPT, &
       BIMAG%IMBLO,BIMAG%IMJNB, &
       BIMAG%IMBLOS,BIMAG%IMJNBS, &
       XINBLO,XJNB,LIMINV,XMATPT, &
       XMATTR &
#if KEY_IMCUBES==1
       ,natim,lbycbim,xatim             & 
#endif
       )
  !
  IF (XNNNB .NE. NBGUES) THEN
     CALL WRNDIE(-5,'<XNBLST>','Non-bond list overflow.')
  ENDIF
  !
  I = NATOM
  DO ITRAN = 1,NTRANS
     DO IATOM = 1,NATOM
        I = I + 1
        CG(I)  = CG(IATOM)
        IAC(I) = IAC(IATOM)
     enddo
  enddo
  !
  RETURN
END SUBROUTINE XNBLST

SUBROUTINE XNBLS2(NATOM,XNNNB,NTRANS,IMINV,IMATPT,IMBLO,JNB, &
     IMBLOS,JNBS,XINBLO,XJNB,LIMINV,XMATPT,XMATTR &
#if KEY_IMCUBES==1
     ,natim,lbycbim,xatim                  & 
#endif
     )
  !
  !     Creates expanded image non-bond lists from the normal
  !     CHARMM ones.
  !
  use chm_kinds
  implicit none
  !
  INTEGER IMATPT(*),JNB(*),JNBS(*),XJNB(*),XMATPT(*),XMATTR(*)
  INTEGER IMBLO(*),IMBLOS(*),IMINV(*), XINBLO(*)
  INTEGER XNNNB,NATOM,NTRANS
  LOGICAL LIMINV
  !
  INTEGER   I, IATOM, INCI1, INCI2, INCJ1, ITEMP, ITRANS
  INTEGER   J, JATOM, JPR, JTRANS, K, L, NB, NBNEW, NPR
#if KEY_IMCUBES==1
#if KEY_IMCUBES==1
  logical lbycbim             
#endif
  integer natim,xatim
#endif 
  !
  XINBLO(1:natom) = 0
  !
  NBNEW = 0
  DO ITRANS = 1,NTRANS
     JTRANS = IMINV(ITRANS)
     IF (ITRANS .GT. 1) THEN
        INCI1  = IMATPT(ITRANS - 1)
     ELSE
        INCI1 = NATOM
     ENDIF
     IF (JTRANS .GT. 1) THEN
        INCJ1  = IMATPT(JTRANS - 1)
     ELSE
        INCJ1 = NATOM
     ENDIF
     INCI2  = ITRANS * NATOM
     !
     IF (ITRANS .EQ. JTRANS) THEN
        DO IATOM = 1,NATOM
           DO JATOM = 1,NATOM
              IF (IATOM .EQ. JATOM) THEN
                 NB  = IMBLOS(INCI1 + IATOM - 1)
                 NPR = IMBLOS(INCI1 + IATOM) - NB
                 IF (NPR .EQ. 1 .AND. IABS(JNBS(NB+1)) .EQ. JATOM) THEN
                    NBNEW = NBNEW + 1
                    XJNB(NBNEW) = JATOM
                 ENDIF
              ELSE
                 IF (IATOM .GT. JATOM) THEN
                    I = JATOM
                    J = IATOM
                 ELSE
                    I = IATOM
                    J = JATOM
                 ENDIF
                 NB  = IMBLO(INCI1 + I - 1)
#if KEY_IMCUBES==1
                 if(lbycbim)nb=IMBLO(INCI1 + natim)             
#endif
                 NPR = IMBLO(INCI1 + I) - NB
                 !               APPEND-NEW-ENTRY-TO-NONBOND-LIST
                 DO K = 1,NPR
                    NB = NB + 1
                    L  = JNB(NB)
                    IF (IABS(L) .EQ. J) THEN
                       NBNEW       = NBNEW + 1
                       XJNB(NBNEW) = JATOM
                    ENDIF
                 enddo
              ENDIF
           enddo
           XINBLO(INCI2 + IATOM) = NBNEW
        enddo
        !
     ELSE IF ((ITRANS .NE. JTRANS) .AND. (.NOT. LIMINV)) THEN
        DO IATOM = 1,NATOM
           DO JATOM = 1,NATOM
              IF (IATOM .EQ. JATOM) THEN
                 NB  = IMBLOS(INCI1 + IATOM - 1)
                 NPR = IMBLOS(INCI1 + IATOM) - NB
                 IF (NPR .EQ. 1 .AND. IABS(JNBS(NB+1)) .EQ. JATOM) THEN
                    NBNEW = NBNEW + 1
                    XJNB(NBNEW) = JATOM
                 ENDIF
              ELSE
                 IF (IATOM .GT. JATOM) THEN
                    I = JATOM
                    J = IATOM
                    NB  = IMBLO(INCJ1 + I - 1)
#if KEY_IMCUBES==1
                    if(lbycbim)nb=IMBLO(INCI1 + I + natim)          
#endif
                    NPR = IMBLO(INCJ1 + I) - NB
                 ELSE
                    I = IATOM
                    J = JATOM
                    NB  = IMBLO(INCI1 + I - 1)
#if KEY_IMCUBES==1
                    if(lbycbim)nb=IMBLO(INCI1 + I + natim)      
#endif
                    NPR = IMBLO(INCI1 + I) - NB
                 ENDIF
                 !               APPEND-NEW-ENTRY-TO-NONBOND-LIST
                 DO K = 1,NPR
                    NB = NB + 1
                    L  = JNB(NB)
                    IF (IABS(L) .EQ. J) THEN
                       NBNEW       = NBNEW + 1
                       XJNB(NBNEW) = JATOM
                    ENDIF
                 enddo
              ENDIF
           enddo
           XINBLO(INCI2 + IATOM) = NBNEW
        enddo
        !
     ELSE IF (ITRANS .GT. JTRANS) THEN
        DO IATOM = 1,NATOM
           DO JATOM = 1,NATOM
              I   = JATOM
              J   = IATOM
              NB  = IMBLO(INCJ1 + I - 1)
#if KEY_IMCUBES==1
              if(lbycbim)nb=IMBLO(INCJ1 + I + natim)           
#endif
              NPR = IMBLO(INCJ1 + I) - NB
              !             APPEND-NEW-ENTRY-TO-NONBOND-LIST
              DO K = 1,NPR
                 NB = NB + 1
                 L  = JNB(NB)
                 IF (IABS(L) .EQ. J) THEN
                    NBNEW       = NBNEW + 1
                    XJNB(NBNEW) = JATOM
                 ENDIF
              enddo
           enddo
           XINBLO(INCI2 + IATOM) = NBNEW
        enddo
        !
     ELSE IF (ITRANS .LT. JTRANS) THEN
        ITEMP = IMBLO(INCI1)
        NB    = ITEMP
        DO IATOM = 1,NATOM
#if KEY_IMCUBES==1
           if(lbycbim)itemp=IMBLO(INCI1 + IATOM + natim)
           if(lbycbim)nb=itemp
#endif 
           NPR   = IMBLO(INCI1 + IATOM) - ITEMP
           ITEMP = IMBLO(INCI1 + IATOM)
           DO JPR = 1,NPR
              NB          = NB    + 1
              NBNEW       = NBNEW + 1
              XJNB(NBNEW) = JNB(NB)
           enddo
           XINBLO(INCI2 + IATOM) = NBNEW
        enddo
     ENDIF
  enddo
  !
  XNNNB = NBNEW
  !
  !     Create modified IMATPT and IMATTR arrays.
  !
  IATOM = NATOM
  DO ITRANS = 1,NTRANS
     IATOM = IATOM + NATOM
     XMATPT(ITRANS) = IATOM
  enddo
  !
  XMATTR(1:nATOM) = 0
  I = NATOM
  DO ITRANS = 1,NTRANS
     DO IATOM = 1,NATOM
        I = I + 1
        XMATTR(I) = IATOM
     enddo
  enddo
  !
  RETURN
END SUBROUTINE XNBLS2

SUBROUTINE ECRYS2(DD1X,DD2X)
  !
  !     Calculate the second derivatives on the system due to the images.
  !
#if KEY_FLUCQ==1
  use flucqm,only:fqcfor          
#endif
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  !
  use bases_fcm
  use coord
  use deriv
  use econtmod
  use energym
  use exelecm
  use image
  use image_routines_module, only: transi, transo
  use eimg
  use inbnd
  use param
  use psf
  use memory
#if KEY_FLUCQ==1
  use flucq
#endif 
  use prssre
  implicit none
  !
  integer,allocatable,dimension(:) :: IOFF
  real(chm_real) DD1X(*),DD2X(*)
  real(chm_real),parameter :: DAT0(1) = (/ ZERO /)
  INTEGER I
  !
  !     Construct image atoms.
  !
  CALL TRANSO(X,Y,Z,DX,DY,DZ,.TRUE.,QECONT,ECONT,NATOM,NTRANS, &
       IMTRNS,XMATPT,XMATTR,NOROT,XATIM &
#if KEY_FLUCQ==1
       ,QFLUC,CG,FQCFOR    & 
#endif
       )
  !
  !     Calculate energy terms.
  !
  DO I=1,XATIM
     DX(I)=DX(I)*TWO
     DY(I)=DY(I)*TWO
     DZ(I)=DZ(I)*TWO
  enddo
  !
  call chmalloc('xtlfrq.src','ECRYS2','IOFF',NATC,intg=IOFF)
  CALL EVDWSD(ETERM(IMVDW),ETERM(IMELEC),XATIM,XJNB, &
       XINBLO,CG,CNBA,CNBB,MAXCN,IAC,ITC,NATC,IOFF, &
       LELEC,LVDW,LGROUP,QEXTND,LCONS,LSHFT,LVATOM,LVSHFT, &
       .FALSE.,DX,DY,DZ,X,Y,Z,0,-1,0,0,DAT0,ZERO,ZERO, &
       CTONNB,CTOFNB,EPS,E14FAC, &
       DD1X,DD2X,.FALSE.,.FALSE.)
  call chmdealloc('xtlfrq.src','ECRYS2','IOFF',NATC,intg=IOFF)
  !
  ETERM(IMVDW)=ETERM(IMVDW)*HALF
  ETERM(IMELEC)=ETERM(IMELEC)*HALF
  EPROP(EPOT)=EPROP(EPOT)+ETERM(IMELEC)+ETERM(IMVDW)
  DO I=1,XATIM
     DX(I)=DX(I)*HALF
     DY(I)=DY(I)*HALF
     DZ(I)=DZ(I)*HALF
  enddo
  !
  !     Calculate lattice first derivatives.
  !
  CALL VIRAL(EPROP(VIRI),EPRESS(VIXX:VIZZ),1,NATOMT,X,Y,Z,DX,DY,DZ)
  !
  !     Calculate forces on the image atoms.
  !
  CALL TRANSI(X,Y,Z,DX,DY,DZ,QECONT,ECONT,NATOM,NTRANS, &
       IMTRNS,XMATPT,XMATTR,NOROT,XATIM, &
       IMINV,IMFORC,IMTORQ &
#if KEY_FLUCQ==1
       ,QFLUC,FQCFOR    & 
#endif
       )
  !
  CALL VIRTOT(EPRESS(VIXX:VIZZ),EPRESS(VEXX:VEZZ),NATOM,X,Y,Z,DX,DY,DZ)
  IF(NTRANS.GT.0) &
       CALL LATTFD(XTLTYP,XTLABC,EPRESS(VEXX:VEZZ),DXTL,XTLREF)
  !
  !     Scale the second derivative terms.
  !
  DO I=1,6*XATIM
     DD1X(I)=HALF*DD1X(I)
  enddo
  DO I=1,6*XNNNB
     DD2X(I)=HALF*DD2X(I)
  enddo
  !
  RETURN
END SUBROUTINE ECRYS2

SUBROUTINE EVDWSD(ENB,EEL,NATOM,JNB,INBLO,CG,CNBA,CNBB,MAXCN, &
     IAC,ITC,NATC,IOFF, &
     LELEC,LVDW,LGROUP,LEXTND,LCONS,LSHFT,LVATOM,LVSHFT,LVSIGM, &
     DX,DY,DZ,X,Y,Z,IPRINT,IUNIT,IMAXP,ANALYS,DATA, &
     SIGON,SIGOFF,CTONNB,CTOFNB,EPS,E14FAC, &
     DD1,DD2,NSECD,DIAGSD)
  !
  !     THIS ROUTINE CALCULATES NON BONDED INTERACTION ENERGIES AND FORCES
  !
  !
  !     ENB    - vdw energy returned
  !     EEL    - electrostatic energy returned
  !     NATOM  - number of atoms
  !     JNB    - nonbond pair list  (INBLO(NATOM))
  !     INBLO  - pointers into JNB  (NATOM)
  !     CG     - charges  (NATOM)
  !     CNBA   - vdw well depths  (MAXCN*2)
  !     CNBB   - vdw well distance (sigma) values  (MAXCN*2)
  !     MAXCN  - offset for 1-4 interaction in CNBA and CNBB
  !     LELEC,,,,,,,,LVSIGM - logical flags used in BNBND.FCM
  !     ITC(IAC(J))  - lookup pointers into CNBA and CNBB  (NATOM)
  !     IPRINT,IMAXP - print option specifiers
  !     ANALYS,DATA  - analysis option items
  !     SIGON,SIGOFF - vdw switching funtion specifiers (in sigma space)
  !     CTONNB,CTOFNB - switching function specifiers in real space
  !     EPS - dielectric constant
  !     DD1,DD2,NSECD,DIAGSD  - second derivative specifiers
  !
  !
  !     By Bernard R. Brooks   5/7/81
  !
  use chm_kinds
  use consta
  use machutil,only:die
  implicit none
  !
  real(chm_real) ENB,EEL
  INTEGER NATOM
  INTEGER JNB(*)
  INTEGER INBLO(*)
  real(chm_real) CG(*),CNBA(*),CNBB(*)
  INTEGER MAXCN
  INTEGER IAC(*),ITC(*)
  INTEGER NATC,IOFF(*)
  LOGICAL LELEC,LVDW,LGROUP,LEXTND,LCONS,LSHFT,LVATOM,LVSHFT,LVSIGM
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  INTEGER IPRINT,IUNIT,IMAXP,ANALYS
  real(chm_real) DATA(*),SIGON,SIGOFF,CTONNB,CTOFNB,EPS,E14FAC
  real(chm_real) DD1(*)
  real(chm_real) DD2(*)
  LOGICAL NSECD,DIAGSD
  !
  real(chm_real) ETEMP1,ETEMP2,ENBPR,EELPR
  real(chm_real) C2OFNB,C4ROF,C2ROF2,CHROF2,C2ONNB
  real(chm_real) RUL3,RUL12,SIGSHF,SGSHSQ,ASH6,BSH6,ASH6D,ASH6DD
  real(chm_real) SIGONS,SIGOFS,SGONSQ,SIGL3,SIGL12,CGF,CGT
  real(chm_real) FDXI,FDYI,FDZI,CRXI,CRYI,CRZI,CGT2,DXI,DYI,DZI
  real(chm_real) S,SIG2,R2,SIG6,SIG12,RIJL,RIJU,FUNCT,DFN
  real(chm_real) EN,DEN,DF,DDF,SIGR2,G2,R1,G1,G3,DXIT,DYIT,DZIT
  real(chm_real) AXX,AYY,AZZ,AXY,AXZ,AYZ,E,SQ
  INTEGER IQ,I,NB,IJADD,ITEMP,J,NPR,I1,IACI,JPR,J1,IC,JJADD,IIADD
  !
  !     FLAGS:  LANALD - DO ANALYSIS
  !     NSECD  - NO SECOND DERIVATIVES CALCULATED
  !     DIAGSD - CALCULATE ONLY DIAGONAL SECOND DERIVATIVES
  !     ELECFG - DO ELECTROSTATICS HERE
  !     NSECD  - IF FALSE, DO SECOND DERIVATIVES
  !     VDWFG  - DO VDW INTERACTION
  !
  !     RSHFT  - DO R-DIELECTRIC WITH SHIFTED POTENTIAL
  !     RSWIT  - DO R-DIELECTRIC WITH SWITCHING FUNCTIONS (DEFAULT)
  !     CSHFT  - DO CONSTANT DIEL. WITH SHIFTED POTENTIAL
  !     CSWIT  - DO CONSTANT DIEL. WITH SWITCHING FUNCTIONS
  !
  !     DSHFT - VDW DISTANCE SHIFTING
  !     DSWIT - VDW DISTANCE SWITCHING (DEFAULT)
  !     SSHFT - VDW SIGMA SHIFTING
  !     SSWIT - VDW SIGMA SWITCHING
  !
  !
  LOGICAL LANALD,RSHFT,RSWIT,CSHFT,CSWIT,LSECD,ELECFG
  LOGICAL VDWFG,SSHFT,SSWIT,DSWIT,DSHFT,ELCFG,LUSED
  !
  LSECD=.NOT.NSECD
  RSHFT=.NOT.LCONS .AND. LSHFT
  RSWIT=.NOT.LCONS .AND. .NOT.LSHFT
  CSWIT=LCONS .AND. .NOT.LSHFT
  CSHFT=LCONS .AND. LSHFT
  !
  ELECFG=(LELEC.AND. (.NOT.LGROUP)) .AND. (EPS.NE.0.0)
  ELCFG=.FALSE.
  VDWFG=LVDW.AND.LVATOM
  IF (VDWFG) THEN
     SSHFT=LVSHFT .AND. LVSIGM
     SSWIT=.NOT.LVSHFT .AND. LVSIGM
     DSHFT=LVSHFT .AND. .NOT.LVSIGM
     DSWIT=.NOT.LVSHFT .AND. .NOT.LVSIGM
  ELSE
     SSHFT=.FALSE.
     SSWIT=.FALSE.
     DSWIT=.FALSE.
     DSHFT=.FALSE.
  ENDIF
  !
  LANALD=(ANALYS.NE.0 .OR. IPRINT.NE.0).OR.LSECD
  !
  IF (.NOT.(ELECFG.OR.VDWFG)) THEN
     IF(LSECD) THEN
        IF (.NOT.(DIAGSD)) THEN
           IQ=6*INBLO(NATOM)
           DD2(1:iq)=0.0
        ENDIF
     ENDIF
     RETURN
  ENDIF
  !
  !
  NB=0
  !
  IF (LSHFT) THEN
     !       SHIFTED DIELECTRIC COEFFICIENTS
     C2OFNB=CTOFNB*CTOFNB
     C4ROF=1./(C2OFNB*C2OFNB)
     C2ROF2=-2.0/C2OFNB
     CHROF2=-0.5/C2OFNB
  ELSE
     !     SWITCHING ELECTROSTATIC OPTIONS
     C2ONNB=CTONNB*CTONNB
     C2OFNB=CTOFNB*CTOFNB
     IF (CTOFNB.GT.CTONNB) THEN
        RUL3=1./(C2OFNB-C2ONNB)**3
        RUL12=RUL3*12.0
     ENDIF
  ENDIF
  !
  IF (SSHFT) THEN
     !     FOR VDW SIGMA SHIFTING
     SIGSHF=SIGOFF
     SGSHSQ=1.0/(SIGSHF*SIGSHF)
     ASH6=2.0*(SGSHSQ**9 -SGSHSQ**6)
     BSH6=3.0*SGSHSQ**6 -4.0*SGSHSQ**3
     ASH6D=6.0*ASH6
     ASH6DD=5.0*ASH6D
     !
  ELSE IF (SSWIT) THEN
     !     VAN DER WAAL SIGMA SWITCHING COEFFICIENTS
     SIGSHF=SIGOFF
     SGSHSQ=1.0/(SIGSHF*SIGSHF)
     SIGONS=SIGON*SIGON
     SIGOFS=SIGOFF*SIGOFF
     SGONSQ=1.0/SIGONS
     IF (SIGOFS.GT.SIGONS) THEN
        SIGL3=1./(SIGOFS-SIGONS)**3
        SIGL12=SIGL3*12
     ENDIF
     !
  ELSE IF (DSWIT) THEN
     !     VAN DER WAAL DISTANCE SWITCHING COEFFICIENTS
     C2ONNB=CTONNB*CTONNB
     C2OFNB=CTOFNB*CTOFNB
     IF (CTOFNB.GT.CTONNB) THEN
        RUL3=1./(C2OFNB-C2ONNB)**3
        RUL12=RUL3*12.0
     ENDIF
  ELSE IF (DSHFT) THEN
     !     VAN DER WAAL DISTANCE SHIFTING
     C2OFNB=CTOFNB*CTOFNB
  ELSE
     IF(VDWFG) CALL DIE
  ENDIF
  !
  IJADD=0
  ITEMP=0
  ENB=0.0
  EEL=0.0
  IF(ELECFG) CGF=CCELEC/EPS
  LUSED=.FALSE.
  !
  !     Initialize the code look up offsets
  !
  J=0
  DO I=1,NATC
     IOFF(I)=J
     J=J+I
  enddo
  !
  !
  IF (IPRINT.GT.0) WRITE(IUNIT,25)
25 FORMAT(//7X,'ATOM ATOM PAIR  DISTANCE    ELECT     VDW    ', &
       'DERIVATIVE CHARGE(I) CHARGE(J)'/)
  !
  !     DO VDW AND ELECTROSTATIC TERMS AS REQUESTED
  !
  loop100: DO I=1,NATOM
     ETEMP1=0.0
     ETEMP2=0.0
     NPR=INBLO(I)-ITEMP
     ITEMP=INBLO(I)
     IF (NPR.EQ.0) GOTO 55
     I1=ITC(IAC(I))
     IACI=IOFF(I1)
     !
     IF (ELECFG) THEN
        CGT=CGF*CG(I)
        ELCFG=(CGT.NE.0.0)
     ENDIF
     !
     !     USE FDXI,FDYI,FDZI FOR ITH COMPONENT OF FORCE VECTORS
     !     USE CRXI,CRYI,CRZI FOR ITH COMPONENT OF THE COORDINATES
     !
     FDXI=DX(I)
     FDYI=DY(I)
     FDZI=DZ(I)
     CRXI=X(I)
     CRYI=Y(I)
     CRZI=Z(I)
     !
     loop40: DO JPR=1,NPR
        NB=NB+1
        IF (JNB(NB).LT.0) THEN
           CGT2=CGT*E14FAC
           J=-JNB(NB)
           J1=ITC(IAC(J))
           IF (I1.LT.J1) THEN
              IC=IOFF(J1)+I1+MAXCN
           ELSE
              IC=IACI+J1+MAXCN
           ENDIF
        ELSE
           CGT2=CGT
           J=JNB(NB)
           J1=ITC(IAC(J))
           IF (I1.LT.J1) THEN
              IC=IOFF(J1)+I1
           ELSE
              IC=IACI+J1
           ENDIF
        ENDIF
        !
        DXI=CRXI-X(J)
        DYI=CRYI-Y(J)
        DZI=CRZI-Z(J)
        S=DXI*DXI+DYI*DYI+DZI*DZI
        !
        !     TO COMPUTE VDW INTERACTION FOR THIS PAIR
        !
        !
        IF (DSWIT) THEN
           !     VAN DER WAAL DISTANCE SWITCHING FUNCTION
           !
           IF (S.LT.C2OFNB) THEN
              SIG2=CNBA(IC)/S
              R2=1.0/S
              SIG6=SIG2*SIG2*SIG2
              SIG12=SIG6*SIG6
              IF (S.GT.C2ONNB) THEN
                 RIJL=C2ONNB-S
                 RIJU=C2OFNB-S
                 FUNCT=RIJU*RIJU*(RIJU-3*RIJL)*RUL3
                 DFN=RIJL*RIJU*RUL12
                 EN=CNBB(IC)*(SIG12-SIG6-SIG6)
                 ENBPR=(FUNCT*EN)
                 DEN=CNBB(IC)*R2*12.0*(SIG6-SIG12)
                 DF=DFN*EN+FUNCT*DEN
                 IF(LSECD) THEN
                    DDF=FUNCT*CNBB(IC)*(156.*SIG12-84.0*SIG6)*R2*R2+ &
                         2.*DFN*DEN+EN*(DFN*R2-2.*(RIJU+RIJL)*RUL12)
                 ENDIF
              ELSE
                 ENBPR=(CNBB(IC)*(SIG12-SIG6-SIG6))
                 DF=CNBB(IC)*R2*12.0*(SIG6-SIG12)
                 IF(LSECD) DDF=CNBB(IC)*(156.*SIG12-84.0*SIG6)*R2*R2
              ENDIF
              ETEMP1=ETEMP1+ENBPR
              LUSED=.TRUE.
           ENDIF
           !
        ELSE IF (SSHFT) THEN
           !     VAN DER WAAL SHIFTED POTENTIAL
           !     E= EMIN*( (S/R)**12 - (S/R)**6 + ASH6*(R/S)**6 - BSH6)
           !
           SIG2=CNBA(IC)/S
           IF (SIG2.GT.SGSHSQ) THEN
              R2=1.0/S
              SIG6=SIG2*SIG2*SIG2
              SIG12=SIG6*SIG6
              ENBPR=(CNBB(IC)*(SIG12-SIG6-SIG6+ASH6/SIG6-BSH6))
              DF=CNBB(IC)*R2*(12.0*(SIG6-SIG12)+ASH6D/SIG6)
              IF(LSECD) DDF=CNBB(IC)*R2*R2*(156.0*SIG12-84.0*SIG6 &
                   +ASH6DD/SIG6)
              ETEMP1=ETEMP1+ENBPR
              LUSED=.TRUE.
           ENDIF
           !
        ELSE IF (SSWIT) THEN
           !     VAN DER WAAL SIGMA SWITCHING FUNCTION
           !
           SIG2=CNBA(IC)/S
           IF (SIG2.GT.SGSHSQ) THEN
              R2=1.0/S
              SIG6=SIG2*SIG2*SIG2
              SIG12=SIG6*SIG6
              IF (SIG2.LT.SGONSQ) THEN
                 SIGR2=1.0/SIG2
                 RIJL=SIGONS-SIGR2
                 RIJU=SIGOFS-SIGR2
                 FUNCT=RIJU*RIJU*(RIJU-3*RIJL)*SIGL3
                 DFN=RIJL*RIJU*SIGL12/CNBA(IC)
                 EN=CNBB(IC)*(SIG12-SIG6-SIG6)
                 ENBPR=(FUNCT*EN)
                 DEN=CNBB(IC)*R2*12.0*(SIG6-SIG12)
                 DF=DFN*EN+FUNCT*DEN
                 IF(LSECD) THEN
                    DDF=FUNCT*CNBB(IC)*(156.*SIG12-84.0*SIG6)*R2*R2+ &
                         2.*DFN*DEN+EN*(DFN*R2-2.*(RIJU+RIJL)*RUL12 &
                         /CNBA(IC)**2)
                 ENDIF
              ELSE
                 ENBPR=(CNBB(IC)*(SIG12-SIG6-SIG6))
                 DF=CNBB(IC)*R2*12.0*(SIG6-SIG12)
                 IF(LSECD) DDF=CNBB(IC)*(156.*SIG12-84.0*SIG6)*R2*R2
              ENDIF
              ETEMP1=ETEMP1+ENBPR
              LUSED=.TRUE.
           ENDIF
           !
        ELSE IF (DSHFT) THEN
           !     VAN DER WAAL DISTANCE SHIFTED FUNCTION
           !     E= CA/R**12 - CB/R**6 -CC*R**6 +CD
           !
           IF (S.LT.C2OFNB) THEN
              SIG2=CNBA(IC)/S
              R2=1.0/S
              SIG6=SIG2*SIG2*SIG2
              SIG12=SIG6*SIG6
              SGSHSQ=CNBA(IC)/C2OFNB
              ASH6=2.0*(SGSHSQ**9 -SGSHSQ**6)/SIG6
              BSH6=3.0*SGSHSQ**6 -4.0*SGSHSQ**3
              !
              ENBPR=(CNBB(IC)*(SIG12-SIG6-SIG6+ASH6-BSH6))
              DF=CNBB(IC)*R2*(12.0*(SIG6-SIG12)+6.0*ASH6)
              IF(LSECD) DDF=CNBB(IC)*R2*R2*(156.0*SIG12-84.0*SIG6 &
                   +30.0*ASH6)
              ETEMP1=ETEMP1+ENBPR
              LUSED=.TRUE.
           ENDIF
           !
        ENDIF
        !
        IF (ELCFG) THEN
           !
           !     TO DO-ELECTROSTATICS-FOR-THIS-PAIR
           !
           EELPR=0.0
           IF (.NOT.(CG(J).EQ.0.0)) THEN
              IF(S.LT.C2OFNB) THEN
                 IF (.NOT.(LUSED)) THEN
                    R2=1.0/S
                    DF=0.0
                    DDF=0.0
                    LUSED=.TRUE.
                    ENBPR=0.0
                 ENDIF
                 IF (RSWIT) THEN
                    !
                    !     TO DO-SWITCHING-FUNCTION-INNER-SECTION
                    G2=CGT2*CG(J)*R2
                    IF (S.GT.C2ONNB) THEN
                       RIJL=C2ONNB-S
                       RIJU=C2OFNB-S
                       FUNCT=RIJU*RIJU*(RIJU-3*RIJL)*RUL3
                       DFN=RIJL*RIJU*RUL12
                       EELPR=(FUNCT*G2)
                       DEN=R2*(-2.*G2)
                       DF=DF+DFN*G2+FUNCT*DEN
                       IF(LSECD) THEN
                          DDF=DDF+(FUNCT*6.*G2)*R2*R2+2.*DFN*DEN+ &
                               G2*(DFN*R2-2.*(RIJU+RIJL)*RUL12)
                       ENDIF
                    ELSE
                       EELPR=G2
                       DF=DF+R2*(-2.*G2)
                       IF(LSECD) THEN
                          DDF=DDF+6.*G2*R2*R2
                       ENDIF
                    ENDIF
                    ETEMP2=ETEMP2+EELPR
                    !
                 ELSE IF (CSHFT) THEN
                    !
                    !     TO DO-CONSTANT-SHIFT-INNER-SECTION
                    !     FOR FUNCTION SHIFTING, THE FORM OF THE POTENTIAL IS
                    !     EEL=QI*QJ/EPS*(1./R)*(1.0 - 2.0*R**2/CTOFNB**2 + R**4/CTOFNB**4)
                    !     EEL=0.0  ( R > CTOFNB )
                    !
                    R1=SQRT(R2)
                    G1=CGT2*CG(J)*R1
                    G2=G1*S*C2ROF2
                    G3=G2*S*CHROF2
                    EELPR=(G1+G2+G3)
                    DF=DF+R2*(G2-G1+3.0*G3)
                    IF(LSECD) DDF=DDF+(2.*G1+6.*G3)*R2*R2
                    ETEMP2=ETEMP2+EELPR
                    !
                 ELSE IF (RSHFT) THEN
                    !
                    !     TO DO-SHIFT-INNER-SECTION
                    !     FOR FUNCTION SHIFTING, THE FORM OF THE POTENTIAL IS
                    !     EEL=QI*QJ/EPS*(1./R**2 + R**2/(CTOFNB**4) - 2.0/CTOFNB**2)
                    !     EEL=0.0  ( R > CTOFNB )
                    !
                    G2=CGT2*CG(J)
                    G1=G2*S*C4ROF
                    G3=G2*C2ROF2
                    G2=G2*R2
                    EELPR=(G1+G2+G3)
                    DF=DF+R2*(2.*(G1-G2))
                    IF(LSECD) DDF=DDF+(6.*G2+2.*G1)*R2*R2
                    ETEMP2=ETEMP2+EELPR
                    !
                 ELSE IF (CSWIT) THEN
                    !
                    !     TO DO-CONSTANT-DIELECTRIC-INNER-SECTION
                    R1=SQRT(R2)
                    G1=CGT2*CG(J)*R1
                    IF (S.GT.C2ONNB) THEN
                       RIJL=C2ONNB-S
                       RIJU=C2OFNB-S
                       FUNCT=RIJU*RIJU*(RIJU-3*RIJL)*RUL3
                       DFN=RIJL*RIJU*RUL12
                       EELPR=(FUNCT*G1)
                       DEN=-R2*G1
                       DF=DF+DFN*G1+FUNCT*DEN
                       IF(LSECD) THEN
                          DDF=DDF+FUNCT*2.*G1*R2*R2+2.*DFN*DEN+ &
                               G1*(DFN*R2-2.*(RIJU+RIJL)*RUL12)
                       ENDIF
                    ELSE
                       EELPR=(G1)
                       DF=DF-R2*G1
                       IF(LSECD) THEN
                          DDF=DDF+2.*G1*R2*R2
                       ENDIF
                    ENDIF
                    ETEMP2=ETEMP2+EELPR
                    !
                 ELSE
                    CALL DIE
                 ENDIF
              ENDIF
           ENDIF
        ENDIF
        !
        IF (LUSED) THEN
           !
           DXIT=DXI*DF
           DYIT=DYI*DF
           DZIT=DZI*DF
           FDXI=FDXI+DXIT
           FDYI=FDYI+DYIT
           FDZI=FDZI+DZIT
           DX(J)=DX(J)-DXIT
           DY(J)=DY(J)-DYIT
           DZ(J)=DZ(J)-DZIT
           IF (LANALD) THEN
              !     TO CALCULATE-SECOND-DERIVATIVES
              IF (LSECD) THEN
                 !
                 DDF=DDF-DF*R2
                 JJADD=(J-1)*6
                 IIADD=(I-1)*6
                 !
                 !     NOW UPDATE DERIVATIVE MATRICIES
                 !
                 AXX=DXI*DXI*DDF+DF
                 AYY=DYI*DYI*DDF+DF
                 AZZ=DZI*DZI*DDF+DF
                 AXY=DXI*DYI*DDF
                 AXZ=DXI*DZI*DDF
                 AYZ=DYI*DZI*DDF
                 !
                 DD1(IIADD+1)=DD1(IIADD+1)+AXX
                 DD1(IIADD+3)=DD1(IIADD+3)+AYY
                 DD1(IIADD+6)=DD1(IIADD+6)+AZZ
                 DD1(IIADD+2)=DD1(IIADD+2)+AXY
                 DD1(IIADD+4)=DD1(IIADD+4)+AXZ
                 DD1(IIADD+5)=DD1(IIADD+5)+AYZ
                 !
                 DD1(JJADD+1)=DD1(JJADD+1)+AXX
                 DD1(JJADD+3)=DD1(JJADD+3)+AYY
                 DD1(JJADD+6)=DD1(JJADD+6)+AZZ
                 DD1(JJADD+2)=DD1(JJADD+2)+AXY
                 DD1(JJADD+4)=DD1(JJADD+4)+AXZ
                 DD1(JJADD+5)=DD1(JJADD+5)+AYZ
                 !
                 IF (.NOT.(DIAGSD)) THEN
                    DD2(IJADD+1)=-AXX
                    DD2(IJADD+3)=-AYY
                    DD2(IJADD+6)=-AZZ
                    DD2(IJADD+2)=-AXY
                    DD2(IJADD+4)=-AXZ
                    DD2(IJADD+5)=-AYZ
                    IJADD=IJADD+6
                 ENDIF
              ELSE
                 IF (ANALYS.NE.0) THEN
                    !     TO STORE-ANALYSIS-INFORMATION
                    !     ANALYS IS DECODED AS FOLLOWS:
                    !
                    !     0) DO NO ANALYSIS, CALCULATE FORCES
                    !        IF GREATER THAN 0 DO ONLY ANALYSIS AND DO NOT REFERENCE FORCE
                    !        ARRAYS, DX, DY, OR DZ.
                    !     1) VAN DER WAALS ENERGY. DIVIDE EVENLY THE INTERACTION BETWEEN
                    !        EACH PAIR AMONG THE TWO ATOMS AND SUM EACH CONTRIBUTION FOR
                    !        ALL INTERACTIONS.
                    !     2) ELECTROSTATIC ENERGY. DIVIDE ENERGY AS DESCRIBED FOR 1.
                    !     3) TOTAL NONBONDED INTERACTION ENERGY. SUM OF 1 AND 2. DIVIDE
                    !        ENERGY AS ABOVE.
                    !
                    E=0.0
                    IF (ANALYS.EQ.1) THEN
                       E=ENBPR
                    ELSE IF (ANALYS.EQ.2) THEN
                       IF(ELCFG) E=EELPR
                    ELSE IF (ANALYS.EQ.3) THEN
                       IF (ELCFG) THEN
                          E=ENBPR+EELPR
                       ELSE
                          E=ENBPR
                       ENDIF
                    ENDIF
                    E=E*0.5
                    DATA(I)=DATA(I)+E
                    DATA(J)=DATA(J)+E
                 ENDIF
                 IF (IPRINT.NE.0) THEN
                    !     TO PRINT-OUT-INFORMATION
                    IF(I.LE.IMAXP.AND.J.LE.IMAXP) THEN
                       SQ=SQRT(S)
                       WRITE(IUNIT,35) I,J,JPR,SQ,EELPR,ENBPR, &
                            DF*SQ,CG(I),CG(J)
35                     FORMAT(5X,3I5,7F10.4)
                    ENDIF
                 ENDIF
              ENDIF
           ENDIF
           !
           LUSED=.FALSE.
        ELSE
           IF(LSECD) THEN
              IF (.NOT.(DIAGSD)) THEN
                 DD2(IJADD+1)=0.0
                 DD2(IJADD+2)=0.0
                 DD2(IJADD+3)=0.0
                 DD2(IJADD+4)=0.0
                 DD2(IJADD+5)=0.0
                 DD2(IJADD+6)=0.0
                 IJADD=IJADD+6
              ENDIF
           ENDIF
        ENDIF
     enddo loop40
     !
     !
     !       RESTORE ITH COMPONENT OF FORCE IN THE ARRAY
     !
     DX(I)=FDXI
     DY(I)=FDYI
     DZ(I)=FDZI
     ENB=ENB+ETEMP1
     EEL=EEL+ETEMP2
     IF(IPRINT.GT.0) THEN
        IF(I.LE.IMAXP) WRITE(IUNIT,45) I,ETEMP2,ETEMP1
45      FORMAT(90X,I5,5X,2F10.4)
     ENDIF
55   CONTINUE
  enddo loop100
  !
  IF (IPRINT.GT.0) WRITE(IUNIT,60) ENB,EEL
60 FORMAT (' FROM EVDW:  VDW ENERGY',F10.4,' ELEC ENERGY',F10.4)
  RETURN
  !
END SUBROUTINE EVDWSD

SUBROUTINE PRNTDC(COMLYN,COMLEN,UNIT)
  !
  !     Print the dispersion curves.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  !
  use coord
  use memory
  use image
  use psf
  use select
  use string
  implicit none
  ! . Passed variables.
  integer,allocatable,dimension(:) :: ISLCT
  complex(chm_cmpx),allocatable,dimension(:) :: WORK
  CHARACTER COMLYN*(*)
  INTEGER   COMLEN, UNIT
  ! . Local variables.
  INTEGER  ISTRT, ISTOP, KSTRT, KSTOP
  real(chm_real)   FACT
  !
  IF(XNSYMM.LE.0.OR.XNSYMM.GT.1.OR.NPHONS.LE.0) RETURN
  ! . Do some initialisation.
  ! . Get the mode specification.
  FACT  = GTRMF (COMLYN, COMLEN, 'FACT', ONE)
  ISTRT = GTRMI (COMLYN, COMLEN, 'MODE', 1)
  ISTOP = GTRMI (COMLYN, COMLEN, 'THRU', NPHONS)
  !
  KSTRT = GTRMI (COMLYN, COMLEN, 'KPTS', 1)
  KSTOP = GTRMI (COMLYN, COMLEN, 'TO', NKPTS)
  !
  IF (ISTRT.GT.NPHONS .OR. ISTRT.GT.ISTOP .OR. ISTRT.LT.1) THEN
     ISTRT = 1
  ENDIF
  IF (ISTOP.GT.NPHONS .OR. ISTOP.LT.ISTRT .OR. ISTOP.LT.1) THEN
     ISTOP = NPHONS
  ENDIF
  !
  IF (KSTRT.GT.NKPTS .OR. KSTRT.GT.KSTOP .OR. KSTRT.LT.1) THEN
     KSTRT = 1
  ENDIF
  IF (KSTOP.GT.NKPTS .OR. KSTOP.GT.KSTRT .OR. KSTOP.LT.1) THEN
     KSTOP = NKPTS
  ENDIF
  !
  call chmalloc('xtlfrq.src','PRNTDC','ISLCT',NATOM,intg=ISLCT)
  call chmalloc('xtlfrq.src','PRNTDC','WORK',NPHONS,cmpx=WORK)
  !
  CALL SELCTA (COMLYN, COMLEN, ISLCT, X, Y, Z, WMAIN, &
       .TRUE.)
  CALL PRNTD2 (KSTRT, KSTOP, ISTRT, ISTOP, ISLCT, KSTART, &
       KSTEP, NPHONS, XEVECP, XFREQP, &
       WORK, FACT, UNIT)
  call chmdealloc('xtlfrq.src','PRNTDC','ISLCT',NATOM,intg=ISLCT)
  call chmdealloc('xtlfrq.src','PRNTDC','WORK',NPHONS,cmpx=WORK)
  !
  RETURN
END SUBROUTINE PRNTDC

SUBROUTINE PRNTD2 (KSTRT, KSTOP, ISTRT, ISTOP, ISLCT, KSTART, &
     KSTEP, NPHONS, EVECP, FREQP, WORK, FACT, UNIT)
  !
  !     Print the dispersion curves.
  !
  use chm_kinds
  use dimens_fcm
  !
  use psf
  use chutil,only:getres
  implicit none
  !
  INTEGER    ISLCT(*), ISTOP, ISTRT, UNIT, KSTOP, KSTRT, NPHONS
  COMPLEX(chm_cmpx) EVECP(NPHONS,NPHONS,*), WORK(*)
  real(chm_real)     FACT, FREQP(NPHONS,*), KSTART(3), KSTEP(3)
  !
  INTEGER    I, IATOM, IIRES, IMODE, IKPTS, IPT, J
  real(chm_real)     XKVEC(3)
  ! . Write the header.
  WRITE (UNIT,'(A)') ' PRNTDC> Dispersion Curves :'
  ! . Write out the modes.
  loop80: DO IKPTS = KSTRT,KSTOP
     DO I = 1,3
        XKVEC(I) = KSTART(I) + (IKPTS - 1) * KSTEP(I)
     ENDDO
     WRITE (UNIT,'(A,3F10.5,A)') &
          ' PRNTDC> Frequencies for the k-vector ',(XKVEC(I),I = 1,3),':'
     DO IMODE = ISTRT,ISTOP
        WRITE (UNIT,'(A,I5,A,F12.6)') &
             ' PRNTDC> Vibrational mode ',IMODE,' Frequency = ', &
             FREQP(IMODE,IKPTS)
        !
        DO I = 1,NPHONS
           WORK(I) = EVECP(I,IMODE,IKPTS)
        ENDDO
        CALL CNORML(WORK,NPHONS)
        IPT = 0
        DO IATOM = 1,NATOM
           DO J = 1,3
              IPT = IPT + 1
              WORK(IPT) = FACT * WORK(IPT) / SQRT(AMASS(IATOM))
           enddo
        enddo
        !
        DO IATOM = 1,NATOM
           IF (ISLCT(IATOM) .GT. 0) THEN
              I = IATOM
              IIRES = GETRES(I, IBASE, NRES)
              I = 3 * (IATOM - 1)
              WRITE (UNIT,'(10X,2I5,1X,A4,1X,A4,6F10.5)') &
                   IATOM, IIRES, RES(IIRES), ATYPE(IATOM), &
                   WORK(I + 1), WORK(I + 2), WORK(I + 3)
           ENDIF
        enddo
     enddo
  enddo loop80
  !
  RETURN
END SUBROUTINE PRNTD2

SUBROUTINE WRITDC (UNIT, NATOM, NKPTS, NPHONS, AMASS, EVALP, &
     EVECP)
  !
  !     Write the dispersion curves to a file.
  !
  use chm_kinds
  use dimens_fcm
  !
  use ctitla
  implicit none
  !
  INTEGER    UNIT, NATOM, NKPTS, NPHONS
  COMPLEX(chm_cmpx) EVECP(NPHONS,NPHONS,*)
  real(chm_real)     AMASS(*), EVALP(NPHONS,*)

  INTEGER I, IKPTS, J
  !
  WRITE (UNIT) 'DISP', NKPTS, NPHONS, NATOM

  CALL WRTITL (TITLEA, NTITLA, UNIT, -1)
  WRITE (UNIT) (AMASS(I) ,I = 1,NATOM)
  DO IKPTS = 1,NKPTS
     WRITE (UNIT) (EVALP(I,IKPTS) ,I = 1,NPHONS)
     DO I = 1,NPHONS
        WRITE (UNIT) (EVECP(J,I,IKPTS), J = 1,NPHONS)
     ENDDO
  ENDDO
  !
  RETURN
END SUBROUTINE WRITDC

SUBROUTINE CNORML (A,N)
  !
  !     Normalise a Complex*16 vector.
  !
  use chm_kinds
  use number
  implicit none
  !
  complex(chm_cmpx) :: A(*), NORM
  INTEGER    I, N
  real(chm_real)     CIMAG, CREAL
  !
  CIMAG = ZERO
  CREAL = ZERO
  DO I = 1,N
     CIMAG = CIMAG + aimag(A(I)) * aimag(A(I))
     CREAL = CREAL + DBLE(A(I)) * DBLE(A(I))
  enddo
  !
  NORM = cmplx(ONE / SQRT(CIMAG + CREAL), ZERO, chm_cmpx)
  !
  A(1:n) = NORM * A(1:n)
  !
  RETURN
END SUBROUTINE CNORML

SUBROUTINE XFRQWR (COMLYN, COMLEN, UNIT)
  !
  !     Process the crystal vibration write options.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  use image
  use psf
  use string
  implicit none
  !
  CHARACTER COMLYN*(*)
  INTEGER   COMLEN, UNIT
  !
  INTEGER   ISTOP, ISTRT
  !
  IF (XNSYMM.LE.0.OR.NFREQX.LE.0) RETURN
  !
  ISTRT = GTRMI (COMLYN, COMLEN, 'MODE', 1)
  ISTOP = GTRMI (COMLYN, COMLEN, 'THRU', NFREQX)
  IF (ISTRT .LT. 1 .OR. ISTRT .GT. NFREQX) ISTRT = 1
  IF (ISTOP.LT.ISTRT .OR. ISTOP.GT.NFREQX) ISTOP = NFREQX
  CALL XFRQW2 (UNIT, ISTRT, ISTOP, NATOM, NFREQX, AMASS, &
       XEVAL, XEVEC)
  !
  RETURN
END SUBROUTINE XFRQWR

SUBROUTINE XFRQW2 (UNIT, ISTRT, ISTOP, NATOM, NVIB, AMASS, &
     EVAL, EVEC)
  !
  !     Write the crystal normal modes to a file.
  !
  use chm_kinds
  use dimens_fcm
  !
  use ctitla
  implicit none
  !
  CHARACTER(len=4) HEADER
  INTEGER     ISTOP, ISTRT, UNIT, NATOM, NVIB
  real(chm_real)      AMASS(*), EVAL(*), EVEC(NVIB,*)

  INTEGER I, J, NSET, SCRATCH(17)
  !
  NSET = NVIB / (3 * NATOM)
  !
  HEADER = 'NMDS'
  DO I=1,17
     SCRATCH(I)=0
  ENDDO
  WRITE (UNIT) HEADER, (ISTOP - ISTRT + 1), NVIB, &
       (NSET * NATOM), (SCRATCH(I), I = 1,17)
  CALL WRTITL (TITLEA, NTITLA, UNIT, -1)
  WRITE (UNIT) ((AMASS(I) ,I = 1,NATOM), J = 1,NSET)
  WRITE (UNIT) (EVAL(I) ,I = ISTRT,ISTOP)
  DO I = ISTRT,ISTOP
     WRITE (UNIT) (EVEC(J,I), J = 1,NVIB)
  ENDDO
  !
  RETURN
END SUBROUTINE XFRQW2

SUBROUTINE EXTAL2 (EELEC, EVDW, NDIM, DDF)
  !
  !     CALLING ROUTINE FOR EXTL2A.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  use memory
  use image
  use psf
  implicit none
  !
  real(chm_real),allocatable,dimension(:) :: XG0
  real(chm_real),allocatable,dimension(:) :: XG1
  real(chm_real),allocatable,dimension(:) :: XG2
  real(chm_real),allocatable,dimension(:) :: YG0
  real(chm_real),allocatable,dimension(:) :: YG1
  real(chm_real),allocatable,dimension(:) :: YG2
  real(chm_real),allocatable,dimension(:) :: ZG0
  real(chm_real),allocatable,dimension(:) :: ZG1
  real(chm_real),allocatable,dimension(:) :: ZG2
  real(chm_real),allocatable,dimension(:) :: XTLX0
  real(chm_real),allocatable,dimension(:) :: XTLY0
  real(chm_real),allocatable,dimension(:) :: XTLZ0
  real(chm_real),allocatable,dimension(:) :: XTLX1
  real(chm_real),allocatable,dimension(:) :: XTLY1
  real(chm_real),allocatable,dimension(:) :: XTLZ1
  real(chm_real),allocatable,dimension(:) :: XTLDX
  real(chm_real),allocatable,dimension(:) :: XTLDY
  real(chm_real),allocatable,dimension(:) :: XTLDZ
  real(chm_real),allocatable,dimension(:) :: ALDX
  real(chm_real),allocatable,dimension(:) :: ALDY
  real(chm_real),allocatable,dimension(:) :: ALDZ
  real(chm_real),allocatable,dimension(:) :: XLDX
  real(chm_real),allocatable,dimension(:) :: XLDY
  real(chm_real),allocatable,dimension(:) :: XLDZ
  INTEGER NDIM
  real(chm_real)  DDF(*), EELEC, EVDW
  !
  INTEGER ISYMM, ITRAN, NOPS, XNSYMT(MAXSYM)
  !
  !     ALLOCATE SCRATCH SPACE FOR THE CRYSTAL ARRAYS.
  !
  call chmalloc('xtlfrq.src','EXTAL2','XG0',NGRP,crl=XG0)
  call chmalloc('xtlfrq.src','EXTAL2','XG1',NGRP,crl=XG1)
  call chmalloc('xtlfrq.src','EXTAL2','XG2',NGRP,crl=XG2)
  call chmalloc('xtlfrq.src','EXTAL2','YG0',NGRP,crl=YG0)
  call chmalloc('xtlfrq.src','EXTAL2','YG1',NGRP,crl=YG1)
  call chmalloc('xtlfrq.src','EXTAL2','YG2',NGRP,crl=YG2)
  call chmalloc('xtlfrq.src','EXTAL2','ZG0',NGRP,crl=ZG0)
  call chmalloc('xtlfrq.src','EXTAL2','ZG1',NGRP,crl=ZG1)
  call chmalloc('xtlfrq.src','EXTAL2','ZG2',NGRP,crl=ZG2)
  call chmalloc('xtlfrq.src','EXTAL2','XTLX0',NATOM,crl=XTLX0)
  call chmalloc('xtlfrq.src','EXTAL2','XTLY0',NATOM,crl=XTLY0)
  call chmalloc('xtlfrq.src','EXTAL2','XTLZ0',NATOM,crl=XTLZ0)
  call chmalloc('xtlfrq.src','EXTAL2','XTLX1',NATOM,crl=XTLX1)
  call chmalloc('xtlfrq.src','EXTAL2','XTLY1',NATOM,crl=XTLY1)
  call chmalloc('xtlfrq.src','EXTAL2','XTLZ1',NATOM,crl=XTLZ1)
  call chmalloc('xtlfrq.src','EXTAL2','XTLDX',NATOM,crl=XTLDX)
  call chmalloc('xtlfrq.src','EXTAL2','XTLDY',NATOM,crl=XTLDY)
  call chmalloc('xtlfrq.src','EXTAL2','XTLDZ',NATOM,crl=XTLDZ)
  call chmalloc('xtlfrq.src','EXTAL2','ALDX',NATOM,crl=ALDX)
  call chmalloc('xtlfrq.src','EXTAL2','ALDY',NATOM,crl=ALDY)
  call chmalloc('xtlfrq.src','EXTAL2','ALDZ',NATOM,crl=ALDZ)
  call chmalloc('xtlfrq.src','EXTAL2','XLDX',NATOM,crl=XLDX)
  call chmalloc('xtlfrq.src','EXTAL2','XLDY',NATOM,crl=XLDY)
  call chmalloc('xtlfrq.src','EXTAL2','XLDZ',NATOM,crl=XLDZ)
  !
  !     Fill XNSYMT.
  !
  DO ISYMM = 1,XNSYMM
     NOPS = 0
     DO ITRAN = 1,NTRANS
        IF ( XNOP(ITRAN) .EQ. ISYMM ) NOPS = NOPS + 1
     enddo
     XNSYMT(ISYMM) = NOPS
  enddo
  !
  CALL EXTL2A (EELEC, EVDW, DDF, XNA, XNB, XNC, &
       XG0, YG0, ZG0, &
       XG1, YG1, ZG1, &
       XG2, YG2, ZG2, &
       XTLX0, XTLY0, XTLZ0, &
       XTLX1, XTLY1, XTLZ1, &
       XTLDX, XTLDY, XTLDZ, &
       ALDX, ALDY, ALDZ, &
       XLDX, XLDY, XLDZ, &
       XNSYMT )
  !
  !     FREE SCRATCH SPACE.
  !
  call chmdealloc('xtlfrq.src','EXTAL2','XG0',NGRP,crl=XG0)
  call chmdealloc('xtlfrq.src','EXTAL2','XG1',NGRP,crl=XG1)
  call chmdealloc('xtlfrq.src','EXTAL2','XG2',NGRP,crl=XG2)
  call chmdealloc('xtlfrq.src','EXTAL2','YG0',NGRP,crl=YG0)
  call chmdealloc('xtlfrq.src','EXTAL2','YG1',NGRP,crl=YG1)
  call chmdealloc('xtlfrq.src','EXTAL2','YG2',NGRP,crl=YG2)
  call chmdealloc('xtlfrq.src','EXTAL2','ZG0',NGRP,crl=ZG0)
  call chmdealloc('xtlfrq.src','EXTAL2','ZG1',NGRP,crl=ZG1)
  call chmdealloc('xtlfrq.src','EXTAL2','ZG2',NGRP,crl=ZG2)
  call chmdealloc('xtlfrq.src','EXTAL2','XTLX0',NATOM,crl=XTLX0)
  call chmdealloc('xtlfrq.src','EXTAL2','XTLY0',NATOM,crl=XTLY0)
  call chmdealloc('xtlfrq.src','EXTAL2','XTLZ0',NATOM,crl=XTLZ0)
  call chmdealloc('xtlfrq.src','EXTAL2','XTLX1',NATOM,crl=XTLX1)
  call chmdealloc('xtlfrq.src','EXTAL2','XTLY1',NATOM,crl=XTLY1)
  call chmdealloc('xtlfrq.src','EXTAL2','XTLZ1',NATOM,crl=XTLZ1)
  call chmdealloc('xtlfrq.src','EXTAL2','XTLDX',NATOM,crl=XTLDX)
  call chmdealloc('xtlfrq.src','EXTAL2','XTLDY',NATOM,crl=XTLDY)
  call chmdealloc('xtlfrq.src','EXTAL2','XTLDZ',NATOM,crl=XTLDZ)
  call chmdealloc('xtlfrq.src','EXTAL2','ALDX',NATOM,crl=ALDX)
  call chmdealloc('xtlfrq.src','EXTAL2','ALDY',NATOM,crl=ALDY)
  call chmdealloc('xtlfrq.src','EXTAL2','ALDZ',NATOM,crl=ALDZ)
  call chmdealloc('xtlfrq.src','EXTAL2','XLDX',NATOM,crl=XLDX)
  call chmdealloc('xtlfrq.src','EXTAL2','XLDY',NATOM,crl=XLDY)
  call chmdealloc('xtlfrq.src','EXTAL2','XLDZ',NATOM,crl=XLDZ)
  !
  RETURN
END SUBROUTINE EXTAL2

SUBROUTINE EXTL2A (EELEC, EVDW, DDF, XXNA, XXNB, XXNC, &
     XG0, YG0, ZG0, XG1, YG1, ZG1, XG2, YG2, ZG2, &
     XTLX0, XTLY0, XTLZ0, XTLX1, XTLY1, XTLZ1, &
     XTLDX, XTLDY, XTLDZ, ALDX, ALDY, ALDZ, &
     XLDX, XLDY, XLDZ, XNSYMT )
  !
  !     CALCULATE THE ENERGIES AND FORCES ON A SYSTEM DUE TO ITS
  !     CRYSTAL ENVIRONMENT. A GROUP BASED CUTOFF IS USED. A GROUP
  !     SWITCHING TERM IS APPLIED TO ANY INTERACTIONS THAT LIE WITHIN
  !     THE ON AND OFF CUTOFF DISTANCES.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  use coord
  use deriv
  use image
  use param
  use psf
  implicit none
  !
  INTEGER XXNA(*), XXNB(*), XXNC(*), XNSYMT(XNSYMM)
  real(chm_real)  DDF(*), EELEC, EVDW, XG0(*), XG1(*), XG2(*), &
       YG0(*), YG1(*), YG2(*), ZG0(*), ZG1(*), ZG2(*), &
       XTLX0(*), XTLY0(*), XTLZ0(*), &
       XTLX1(*), XTLY1(*), XTLZ1(*), &
       XTLDX(*), XTLDY(*), XTLDZ(*), &
       ALDX(*), ALDY(*), ALDZ(*), &
       XLDX(*), XLDY(*), XLDZ(*)
  !
  INTEGER I, IGRP, ISTOP, ISTRT, ISYMM, ITRAN, IXTRAN, J, &
       NUMAT
  LOGICAL OK
  real(chm_real)  DA(3,3), EETEMP, EVTEMP, NA, NB, NC, R(3,3), &
       S(3), T(3), WRKR(3,3), WRKT(3), XFACT, XGCEN, &
       XTLINV(6), YGCEN, ZGCEN
  !
  real(chm_real), PARAMETER :: ZERO = 0.D0, HALF = 0.5D0, ONE = 1.D0
  !
  !     GET THE INVERSE LATTICE VECTOR MATRIX.
  !
  CALL INVT33S(XTLINV,XTLABC,OK)
  IF (.NOT. OK) THEN
     CALL WRNDIE(-5,'<EXTAL1>','METRIC MATRIX SINGULAR.')
  ENDIF
  !
  XFACT = ONE / MAXSYM
  !
  !     CALCULATE THE CENTRES OF GEOMETRY FOR EACH GROUP.
  !
  DO IGRP = 1,NGRP
     ISTRT = IGPBS(IGRP) + 1
     ISTOP = IGPBS(IGRP + 1)
     NUMAT = ISTOP - ISTRT + 1
     XGCEN = ZERO
     YGCEN = ZERO
     ZGCEN = ZERO
     DO I = ISTRT,ISTOP
        XGCEN = XGCEN + X(I)
        YGCEN = YGCEN + Y(I)
        ZGCEN = ZGCEN + Z(I)
     enddo
     XG0(IGRP) = XGCEN / NUMAT
     YG0(IGRP) = YGCEN / NUMAT
     ZG0(IGRP) = ZGCEN / NUMAT
  enddo
  !
  DA(1:3,1:3) = ZERO
  !
  !     LOOP OVER THE CRYSTAL SYMMETRY OPERATIONS.
  !
  ITRAN = 0
  loop170: DO ISYMM = 1,XNSYMM
     !
     !     GET THE SYMMETRY TRANSFORMATION.
     !
     DO I = 1,3
        DO J = 1,3
           R(J,I) = XSYMOP(J,I,ISYMM)
        ENDDO
     ENDDO
     !
     CALL MULNXNFL(WRKR,R,XTLINV,3)
     CALL MULNXNLF(R,XTLABC,WRKR,3)
     !
     !     FIND THE TRANSFORMED GROUP CENTRES.
     !
     DO I = 1,NGRP
        XG1(I) = R(1,1)*XG0(I) + R(1,2)*YG0(I) + R(1,3)*ZG0(I)
        YG1(I) = R(2,1)*XG0(I) + R(2,2)*YG0(I) + R(2,3)*ZG0(I)
        ZG1(I) = R(3,1)*XG0(I) + R(3,2)*YG0(I) + R(3,3)*ZG0(I)
     enddo
     !
     !     FIND THE COORDINATES.
     !
     DO I = 1,NATOM
        XTLX0(I) = R(1,1)*X(I) + R(1,2)*Y(I) + R(1,3)*Z(I)
        XTLY0(I) = R(2,1)*X(I) + R(2,2)*Y(I) + R(2,3)*Z(I)
        XTLZ0(I) = R(3,1)*X(I) + R(3,2)*Y(I) + R(3,3)*Z(I)
     enddo
     !
     DO I = 1,3
        WRKT(I) = XFACT * XSYMOP(I,4,ISYMM)
     ENDDO
     !
     !     LOOP OVER THE TRANSLATIONS FOR EACH SYMMETRY OPERATION.
     !
     loop160: DO IXTRAN = 1,XNSYMT(ISYMM)
        !
        ITRAN = ITRAN + 1
        !
        NA = WRKT(1) + XXNA(ITRAN)
        NB = WRKT(2) + XXNB(ITRAN)
        NC = WRKT(3) + XXNC(ITRAN)
        !
        !RCZ            DO 80 I = 1,3
        !RCZ               T(I) = NA * XTLABC(I,1) + NB * XTLABC(I,2) +
        !RCZ     *                NC * XTLABC(I,3)
        !RCZ   80       CONTINUE
        T(1) = NA*XTLABC(1) + NB*XTLABC(2) + NC*XTLABC(4)
        T(2) = NA*XTLABC(2) + NB*XTLABC(3) + NC*XTLABC(5)
        T(3) = NA*XTLABC(4) + NB*XTLABC(5) + NC*XTLABC(6)
        !
        DO I = 1,NGRP
           XG2(I) = XG1(I) + T(1)
           YG2(I) = YG1(I) + T(2)
           ZG2(I) = ZG1(I) + T(3)
        enddo
        !
        DO I = 1,NATOM
           XTLX1(I) = XTLX0(I) + T(1)
           XTLY1(I) = XTLY0(I) + T(2)
           XTLZ1(I) = XTLZ0(I) + T(3)
        enddo
        !
        !     CALCULATE THE ENERGIES AND FORCES FOR EACH TRANSFORMATION.
        !
        XTLDX(1:natom) = ZERO
        XTLDY(1:natom) = ZERO
        XTLDZ(1:natom) = ZERO
        !
        EETEMP = ZERO
        EVTEMP = ZERO
        !
        CALL EXTNB2(EETEMP, EVTEMP, DDF, ISYMM, XG0, YG0, ZG0, &
             XG2, YG2, ZG2, XTLX1, XTLY1, XTLZ1, &
             XTLDX, XTLDY, XTLDZ, ALDX, ALDY, ALDZ, &
             XLDX, XLDY, XLDZ)
        !
        EELEC = EELEC + EETEMP
        EVDW  = EVDW  + EVTEMP
        !
        !     COMPRESS THE FORCES INTO THE PROPER ARRAYS.
        !
        DO I = 1,NATOM
           DX(I) = DX(I) + R(1,1) * XTLDX(I) + &
                R(2,1) * XTLDY(I) + R(3,1) * XTLDZ(I)
           DY(I) = DY(I) + R(1,2) * XTLDX(I) + &
                R(2,2) * XTLDY(I) + R(3,2) * XTLDZ(I)
           DZ(I) = DZ(I) + R(1,3) * XTLDX(I) + &
                R(2,3) * XTLDY(I) + R(3,3) * XTLDZ(I)
        enddo
        !
        !     CALCULATE THE DERIVATIVES WITH RESPECT TO THE LATTICE PARAMETERS.
        !
        DO I = 1,3
           S(I) = ZERO
        ENDDO
        DO I = 1,NATOM
           S(1) = S(1) + XTLDX(I)
           S(2) = S(2) + XTLDY(I)
           S(3) = S(3) + XTLDZ(I)
        ENDDO
        DO I = 1,3
           DA(I,1) = DA(I,1) + NA * S(I)
           DA(I,2) = DA(I,2) + NB * S(I)
           DA(I,3) = DA(I,3) + NC * S(I)
        ENDDO
        !
     enddo loop160
  enddo loop170
  ! . Calculate the lattice first derivatives.
  DXTL(1:6) = ZERO
  DXTL(1) = DA(3,3)
  IF (XTLTYP .NE. 'CUBI') THEN
     DXTL(2) = DA(2,2)
     IF (XTLTYP .NE. 'HEXA' .AND. XTLTYP .NE. 'TETR') THEN
        DXTL(3) = DA(1,1)
        IF (XTLTYP .NE. 'ORTH') THEN
           DXTL(4) = DA(1,3)
           IF (XTLTYP .NE. 'MONO') THEN
              DXTL(5) = DA(2,1)
              DXTL(6) = DA(2,3)
           ENDIF
        ENDIF
     ENDIF
  ENDIF
  !
  RETURN
END SUBROUTINE EXTL2A

SUBROUTINE EXTNB2(EELEC, EVDW, DDF, ISYMM, &
     XG0, YG0, ZG0, XG2, YG2, ZG2, &
     XTLX, XTLY, XTLZ, XTLDX, XTLDY, XTLDZ, &
     ALDX, ALDY, ALDZ, XLDX, XLDY, XLDZ)
  !
  !     EXTNB2 CALCULATES THE CRYSTAL NON-BONDED INTERACTIONS. NO PAIR
  !     LIST IS REQUIRED. THE INTERACTIONS ARE CALCULATED ON A GROUP BY
  !     GROUP BASIS AND A GROUP SWITCHING FUNCTION IS APPLIED TO THOSE
  !     INTERACTIONS THAT LIE BETWEEN THE ON AND OFF CUTOFF DISTANCES.
  !     THERE ARE NO 1,4 INTERACTIONS FOR CRYSTALS.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  use consta
  use coord
  use deriv
  use inbnd
  use param
  use psf
  implicit none
  !
  INTEGER ISYMM
  real(chm_real)  DDF(*), EELEC, EVDW, XG0(*), YG0(*), ZG0(*), &
       XG2(*), YG2(*), ZG2(*), XTLX(*), XTLY(*), XTLZ(*), &
       XTLDX(*), XTLDY(*), XTLDZ(*), ALDX(*), ALDY(*), ALDZ(*), &
       XLDX(*), XLDY(*), XLDZ(*)
  !
  INTEGER I1, I3, IACI, IAGRP, IATOM, IC, IPT, ISTOPA, ISTOPX, &
       ISTRTA, ISTRTX, IXGRP, IXTAL, J1, J3, JATOM, NUMATA, &
       NUMATX
  real(chm_real)  AXX, AXY, AXZ, AYY, AYZ, AZZ, C2OFNB, C2ONNB,  &
       CGF, CGT, &
       D2F, DELTGX, DELTGY, DELTGZ, DELTX, DELTY, DELTZ, D2FN, &
       DF, DFA, DFN, DFX, DXTEMP, DYTEMP, DZTEMP, EELEC1, ETEMP1, &
       ETEMP2, EVDW1, FACT, FUNCT, RIJ2, R2, RIJL, RIJU, RUL3, &
       RUL12, S, SGCENT, SIG2, SIG6, SIG12, XXTL, YXTL, ZXTL
  !
  real(chm_real), PARAMETER :: ZERO = 0.D0, HALF = 0.5D0, ONE = 1.D0, TWO = 2.D0, &
       THREE = 3.D0, SIX = 6.D0, TWELVE = 12.D0, &
       EIGHT4 = 84.D0, ONE56 = 156.D0
  !
  !     SET UP SOME INITIAL PARAMETERS.
  !
  CGF    = CCELEC / EPS
  C2OFNB = CTOFNB * CTOFNB
  C2ONNB = CTONNB * CTONNB
  RUL3   = ONE / (C2OFNB - C2ONNB) **3
  RUL12  = TWELVE * RUL3
  !
  !     DO OUTER LOOPS OVER GROUPS.
  !
  loop180: DO IXGRP = 1,NGRP
     !
     ISTRTX = IGPBS(IXGRP) + 1
     ISTOPX = IGPBS(IXGRP + 1)
     NUMATX = ISTOPX - ISTRTX + 1
     !
     loop170:DO IAGRP = 1,NGRP
        DELTGX = XG2(IXGRP) - XG0(IAGRP)
        DELTGY = YG2(IXGRP) - YG0(IAGRP)
        DELTGZ = ZG2(IXGRP) - ZG0(IAGRP)
        SGCENT = DELTGX*DELTGX + DELTGY*DELTGY + DELTGZ*DELTGZ
        !
        IF (SGCENT .LT. C2OFNB) THEN
           ISTRTA = IGPBS(IAGRP) + 1
           ISTOPA = IGPBS(IAGRP + 1)
           NUMATA = ISTOPA - ISTRTA + 1
           !
           IF (SGCENT .GT. C2ONNB) THEN
              RIJL  = C2ONNB - SGCENT
              RIJU  = C2OFNB - SGCENT
              RIJ2  = ONE / SGCENT
              FUNCT = RIJU * RIJU * (RIJU - THREE * RIJL) * RUL3
              DFN   = RIJL * RIJU * RUL12
              D2FN  = DFN * RIJ2 - TWO * (RIJL + RIJU) * RUL12
              XLDX(istrtx:istopx) = ZERO
              XLDY(istrtx:istopx) = ZERO
              XLDZ(istrtx:istopx) = ZERO
              ALDX(istrta:istopa) = ZERO
              ALDY(istrta:istopa) = ZERO
              ALDZ(istrta:istopa) = ZERO
           ELSE
              FUNCT = ONE
           ENDIF
           ETEMP1 = ZERO
           ETEMP2 = ZERO
           !
           !     LOOP OVER CRYSTAL ATOMS.
           !
           loop40:DO IXTAL = ISTRTX,ISTOPX
              XXTL = XTLX(IXTAL)
              YXTL = XTLY(IXTAL)
              ZXTL = XTLZ(IXTAL)
              CGT  = CG(IXTAL) * CGF
              I1   = ITC(IAC(IXTAL))
              I3   = 3 * (IXTAL - 1) + 3 * NATOM * (ISYMM - 1)
              IACI = (I1 * (I1 - 1)) / 2
              !
              !     LOOP OVER SYSTEM ATOMS.
              !
              loop30:DO IATOM = ISTRTA,ISTOPA
                 J1 = ITC(IAC(IATOM))
                 IF (I1 .LT. J1) THEN
                    IC = (J1 * (J1 - 1)) / 2 + I1
                 ELSE
                    IC = IACI + J1
                 ENDIF
                 J3 = 3 * (IATOM - 1)
                 DELTX  = XXTL - X(IATOM)
                 DELTY  = YXTL - Y(IATOM)
                 DELTZ  = ZXTL - Z(IATOM)
                 S   = DELTX*DELTX + DELTY*DELTY + DELTZ*DELTZ
                 R2  = ONE / S
                 !
                 !     DO THE VDW INTERACTION FOR THIS ATOM PAIR.
                 !
                 SIG2  = CNBA(IC) * R2
                 SIG6  = SIG2 * SIG2 * SIG2
                 SIG12 = SIG6 * SIG6
                 EVDW1 = CNBB(IC) * (SIG12 - SIG6 - SIG6)
                 DF  = CNBB(IC) * R2 * TWELVE * (SIG6 - SIG12)
                 D2F = CNBB(IC) * (ONE56 * SIG12 - &
                      EIGHT4 * SIG6) * R2 * R2
                 !
                 !     DO ELECTROSTATICS FOR THIS PAIR.
                 !
                 EELEC1 = CGT * CG(IATOM) * SQRT(R2)
                 DF  = DF - R2 * EELEC1
                 D2F = D2F + TWO * EELEC1 * R2 * R2
                 !
                 !     ACCUMULATE THE ENERGIES.
                 !
                 ETEMP1 = ETEMP1 + EELEC1
                 ETEMP2 = ETEMP2 + EVDW1
                 !
                 !     ACCUMULATE THE GROUP DERIVATIVES FOR LATER.
                 !
                 IF (SGCENT .GT. C2ONNB) THEN
                    DXTEMP = DELTX * DF
                    DYTEMP = DELTY * DF
                    DZTEMP = DELTZ * DF
                    ALDX(IATOM) = ALDX(IATOM) - DXTEMP
                    ALDY(IATOM) = ALDY(IATOM) - DYTEMP
                    ALDZ(IATOM) = ALDZ(IATOM) - DZTEMP
                    XLDX(IXTAL) = XLDX(IXTAL) + DXTEMP
                    XLDY(IXTAL) = XLDY(IXTAL) + DYTEMP
                    XLDZ(IXTAL) = XLDZ(IXTAL) + DZTEMP
                 ENDIF
                 !
                 !     ACCUMULATE THE FORCES.
                 !
                 DXTEMP = HALF * FUNCT * DELTX * DF
                 DYTEMP = HALF * FUNCT * DELTY * DF
                 DZTEMP = HALF * FUNCT * DELTZ * DF
                 !
                 XTLDX(IXTAL) = XTLDX(IXTAL) + DXTEMP
                 XTLDY(IXTAL) = XTLDY(IXTAL) + DYTEMP
                 XTLDZ(IXTAL) = XTLDZ(IXTAL) + DZTEMP
                 !
                 DX(IATOM) = DX(IATOM) - DXTEMP
                 DY(IATOM) = DY(IATOM) - DYTEMP
                 DZ(IATOM) = DZ(IATOM) - DZTEMP
                 !
                 !     CALCULATE THE SECOND DERIVATIVE TERMS.
                 !
                 IF (.NOT. (ISYMM .EQ. 1 .AND. &
                      IXTAL .EQ. IATOM)) THEN
                    D2F = (D2F - DF * R2) * FUNCT
                    AXX = DELTX * DELTX * D2F + DF * FUNCT
                    AYY = DELTY * DELTY * D2F + DF * FUNCT
                    AZZ = DELTZ * DELTZ * D2F + DF * FUNCT
                    AXY = DELTX * DELTY * D2F
                    AXZ = DELTX * DELTZ * D2F
                    AYZ = DELTY * DELTZ * D2F
                    !
                    !     ADD IN DIAGONAL TERMS FOR THE SYSTEM ATOM (IATOM).
                    !
                    IPT = ((J3 + 1) * (J3 + 2)) / 2
                    DDF(IPT) = DDF(IPT) + AXX
                    !
                    IPT = IPT + J3 + 1
                    DDF(IPT)   = DDF(IPT)   + AXY
                    DDF(IPT+1) = DDF(IPT+1) + AYY
                    !
                    IPT = IPT + J3 + 2
                    DDF(IPT)   = DDF(IPT)   + AXZ
                    DDF(IPT+1) = DDF(IPT+1) + AYZ
                    DDF(IPT+2) = DDF(IPT+2) + AZZ
                    !
                    !     ADD IN THE OFF-DIAGONAL TERMS.
                    !
                    IF (J3 .LT. I3) THEN
                       IPT = (I3 * (I3 + 1)) / 2 + J3 + 1
                       DDF(IPT)   = DDF(IPT)   - AXX
                       DDF(IPT+1) = DDF(IPT+1) - AXY
                       DDF(IPT+2) = DDF(IPT+2) - AXZ
                       !
                       IPT = IPT + I3 + 1
                       DDF(IPT)   = DDF(IPT)   - AXY
                       DDF(IPT+1) = DDF(IPT+1) - AYY
                       DDF(IPT+2) = DDF(IPT+2) - AYZ
                       !
                       IPT = IPT + I3 + 2
                       DDF(IPT)   = DDF(IPT)   - AXZ
                       DDF(IPT+1) = DDF(IPT+1) - AYZ
                       DDF(IPT+2) = DDF(IPT+2) - AZZ
                    ENDIF
                 ENDIF
                 !
              enddo loop30
           enddo loop40
           !
           !     REMOVE FORCES DUE TO THE SWITCHING TERM.
           !
           IF (SGCENT .GT. C2ONNB) THEN
              IF (.NOT. (ISYMM .EQ. 1 .AND. &
                   (IXGRP .EQ. IAGRP))) THEN
                 DF  = DFN * (ETEMP1 + ETEMP2)
                 DFA = HALF * DF / NUMATA
                 DFX = HALF * DF / NUMATX
                 DXTEMP = DELTGX * DFA
                 DYTEMP = DELTGY * DFA
                 DZTEMP = DELTGZ * DFA
                 DO IATOM = ISTRTA,ISTOPA
                    DX(IATOM) = DX(IATOM) - DXTEMP
                    DY(IATOM) = DY(IATOM) - DYTEMP
                    DZ(IATOM) = DZ(IATOM) - DZTEMP
                 enddo
                 DXTEMP = DELTGX * DFX
                 DYTEMP = DELTGY * DFX
                 DZTEMP = DELTGZ * DFX
                 DO IXTAL = ISTRTX,ISTOPX
                    XTLDX(IXTAL) = XTLDX(IXTAL) + DXTEMP
                    XTLDY(IXTAL) = XTLDY(IXTAL) + DYTEMP
                    XTLDZ(IXTAL) = XTLDZ(IXTAL) + DZTEMP
                 enddo
                 !
                 !     MODIFY THE SECOND DERIVATIVES BECAUSE OF THE SWITCHING TERM.
                 !
                 D2F = (D2FN - DFN * RIJ2) * (ETEMP1 + ETEMP2)
                 AXX = DELTGX * DELTGX * D2F + DF
                 AYY = DELTGY * DELTGY * D2F + DF
                 AZZ = DELTGZ * DELTGZ * D2F + DF
                 AXY = DELTGX * DELTGY * D2F
                 AXZ = DELTGX * DELTGZ * D2F
                 AYZ = DELTGY * DELTGZ * D2F
                 !
                 !     ADD IN THE DIAGONAL TERMS.
                 !
                 FACT = ONE / (NUMATA * NUMATA)
                 DO IATOM = ISTRTA, ISTOPA
                    I3 = 3 * (IATOM - 1)
                    DO JATOM = ISTRTA,(IATOM-1)
                       J3 = 3 * (JATOM - 1)
                       IPT = (I3 * (I3 + 1)) / 2 + J3 + 1
                       DDF(IPT)   = DDF(IPT)   + FACT * AXX
                       DDF(IPT+1) = DDF(IPT+1) + FACT * AXY
                       DDF(IPT+2) = DDF(IPT+2) + FACT * AXZ
                       IPT = IPT + I3 + 1
                       DDF(IPT)   = DDF(IPT)   + FACT * AXY
                       DDF(IPT+1) = DDF(IPT+1) + FACT * AYY
                       DDF(IPT+2) = DDF(IPT+2) + FACT * AYZ
                       IPT = IPT + I3 + 2
                       DDF(IPT)   = DDF(IPT)   + FACT * AXZ
                       DDF(IPT+1) = DDF(IPT+1) + FACT * AYZ
                       DDF(IPT+2) = DDF(IPT+2) + FACT * AZZ
                    enddo
                    IPT = ((I3 + 1) * (I3 + 2)) / 2
                    DDF(IPT) = DDF(IPT) + FACT * AXX
                    IPT = IPT + I3 + 1
                    DDF(IPT)   = DDF(IPT)   + FACT * AXY
                    DDF(IPT+1) = DDF(IPT+1) + FACT * AYY
                    IPT = IPT + I3 + 2
                    DDF(IPT)   = DDF(IPT)   + FACT * AXZ
                    DDF(IPT+1) = DDF(IPT+1) + FACT * AYZ
                    DDF(IPT+2) = DDF(IPT+2) + FACT * AZZ
                 enddo
                 !
                 !     MODIFY THE CROSS-TERMS.
                 !
                 FACT = ONE / (NUMATA * NUMATX)
                 loop100:DO IXTAL = ISTRTX, ISTOPX
                    I3 = 3 * (IXTAL - 1) + 3 * NATOM * (ISYMM - 1)
                    DO IATOM = ISTRTA,ISTOPA
                       J3 = 3 * (IATOM - 1)
                       IF (J3 .LT. I3) THEN
                          IPT = (I3 * (I3 + 1)) / 2 + J3 + 1
                          DDF(IPT)   = DDF(IPT)   - FACT * AXX
                          DDF(IPT+1) = DDF(IPT+1) - FACT * AXY
                          DDF(IPT+2) = DDF(IPT+2) - FACT * AXZ
                          IPT = IPT + I3 + 1
                          DDF(IPT)   = DDF(IPT)   - FACT * AXY
                          DDF(IPT+1) = DDF(IPT+1) - FACT * AYY
                          DDF(IPT+2) = DDF(IPT+2) - FACT * AYZ
                          IPT = IPT + I3 + 2
                          DDF(IPT)   = DDF(IPT)   - FACT * AXZ
                          DDF(IPT+1) = DDF(IPT+1) - FACT * AYZ
                          DDF(IPT+2) = DDF(IPT+2) - FACT * AZZ
                       ENDIF
                    enddo
                 enddo loop100
                 !
                 !     DO THE FIRST DERIVATIVE TERMS.
                 !
                 FACT = - DFN / NUMATA
                 DO IATOM = ISTRTA,ISTOPA
                    ALDX(IATOM) = FACT * ALDX(IATOM)
                    ALDY(IATOM) = FACT * ALDY(IATOM)
                    ALDZ(IATOM) = FACT * ALDZ(IATOM)
                 enddo
                 DO IXTAL = ISTRTX,ISTOPX
                    XLDX(IXTAL) = FACT * XLDX(IXTAL)
                    XLDY(IXTAL) = FACT * XLDY(IXTAL)
                    XLDZ(IXTAL) = FACT * XLDZ(IXTAL)
                 enddo
                 !
                 DO IATOM = ISTRTA, ISTOPA
                    I3 = 3 * (IATOM - 1)
                    DO JATOM = ISTRTA,(IATOM-1)
                       J3 = 3 * (JATOM - 1)
                       IPT = (I3 * (I3 + 1)) / 2 + J3 + 1
                       DDF(IPT)   = DDF(IPT) + DELTGX * &
                            (ALDX(IATOM) + ALDX(JATOM))
                       DDF(IPT+1) = DDF(IPT+1) + (DELTGY * &
                            ALDX(IATOM) + DELTGX * ALDY(JATOM))
                       DDF(IPT+2) = DDF(IPT+2) + (DELTGZ * &
                            ALDX(IATOM) + DELTGX * ALDZ(JATOM))
                       IPT = IPT + I3 + 1
                       DDF(IPT)   = DDF(IPT) + (DELTGX * &
                            ALDY(IATOM) + DELTGY * ALDX(JATOM))
                       DDF(IPT+1) = DDF(IPT+1) + DELTGY * &
                            (ALDY(IATOM) + ALDY(JATOM))
                       DDF(IPT+2) = DDF(IPT+2) + (DELTGZ * &
                            ALDY(IATOM) + DELTGY * ALDZ(JATOM))
                       IPT = IPT + I3 + 2
                       DDF(IPT)   = DDF(IPT) + (DELTGX * &
                            ALDZ(IATOM) + DELTGZ * ALDX(JATOM))
                       DDF(IPT+1) = DDF(IPT+1) + (DELTGY * &
                            ALDZ(IATOM) + DELTGZ * ALDY(JATOM))
                       DDF(IPT+2) = DDF(IPT+2) + DELTGZ * &
                            (ALDZ(IATOM) + ALDZ(JATOM))
                    enddo
                    IPT = ((I3 + 1) * (I3 + 2)) / 2
                    DDF(IPT) = DDF(IPT) + TWO * DELTGX * &
                         ALDX(IATOM)
                    IPT = IPT + I3 + 1
                    DDF(IPT)   = DDF(IPT) + (DELTGX * &
                         ALDY(IATOM) + DELTGY * ALDX(IATOM))
                    DDF(IPT+1) = DDF(IPT+1) + TWO * DELTGY * &
                         ALDY(IATOM)
                    IPT = IPT + I3 + 2
                    DDF(IPT)   = DDF(IPT) + (DELTGX * &
                         ALDZ(IATOM) + DELTGZ * ALDX(IATOM))
                    DDF(IPT+1) = DDF(IPT+1) + (DELTGY * &
                         ALDZ(IATOM) + DELTGZ * ALDY(IATOM))
                    DDF(IPT+2) = DDF(IPT+2) + TWO * DELTGZ * &
                         ALDZ(IATOM)
                 enddo
                 !
                 !     MODIFY THE CROSS-TERMS.
                 !
                 FACT = -NUMATA
                 FACT = FACT / NUMATX
                 DO IATOM = ISTRTA,ISTOPA
                    ALDX(IATOM) = FACT * ALDX(IATOM)
                    ALDY(IATOM) = FACT * ALDY(IATOM)
                    ALDZ(IATOM) = FACT * ALDZ(IATOM)
                 enddo
                 DO IXTAL = ISTRTX, ISTOPX
                    I3 = 3 * (IXTAL - 1) + 3 * NATOM * (ISYMM - 1)
                    DO IATOM = ISTRTA,ISTOPA
                       J3 = 3 * (IATOM - 1)
                       IF (J3 .LT. I3) THEN
                          IPT = (I3 * (I3 + 1)) / 2 + J3 + 1
                          DDF(IPT)   = DDF(IPT) + DELTGX * &
                               (XLDX(IXTAL) + ALDX(IATOM))
                          DDF(IPT+1) = DDF(IPT+1) + DELTGY * &
                               XLDX(IXTAL) + DELTGX * ALDY(IATOM)
                          DDF(IPT+2) = DDF(IPT+2) + DELTGZ * &
                               XLDX(IXTAL) + DELTGX * ALDZ(IATOM)
                          IPT = IPT + I3 + 1
                          DDF(IPT)   = DDF(IPT) + DELTGX * &
                               XLDY(IXTAL) + DELTGY * ALDX(IATOM)
                          DDF(IPT+1) = DDF(IPT+1) + DELTGY * &
                               (XLDY(IXTAL) + ALDY(IATOM))
                          DDF(IPT+2) = DDF(IPT+2) + DELTGZ * &
                               XLDY(IXTAL) + DELTGY * ALDZ(IATOM)
                          IPT = IPT + I3 + 2
                          DDF(IPT)   = DDF(IPT) + DELTGX * &
                               XLDZ(IXTAL) + DELTGZ * ALDX(IATOM)
                          DDF(IPT+1) = DDF(IPT+1) + DELTGY * &
                               XLDZ(IXTAL) + DELTGZ * ALDY(IATOM)
                          DDF(IPT+2) = DDF(IPT+2) + DELTGZ * &
                               (XLDZ(IXTAL) + ALDZ(IATOM))
                       ENDIF
                    enddo
                 enddo
                 !
              ENDIF
              ETEMP1 = ETEMP1 * FUNCT
              ETEMP2 = ETEMP2 * FUNCT
           ENDIF
           EELEC = EELEC + ETEMP1
           EVDW  = EVDW  + ETEMP2
           !
        ENDIF
     enddo loop170
  enddo loop180
  !
  EELEC = HALF * EELEC
  EVDW  = HALF * EVDW
  !
  RETURN
END SUBROUTINE EXTNB2

SUBROUTINE XTLFRR ( DDF, DDF1, N )
  !
  !     Reorder a second derivative matrix.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  implicit none
  !
  INTEGER N
  real(chm_real)  DDF(*),DDF1(*)
  !
  INTEGER I,IPT,ISPACE,J,JPT
  !
  IPT = 0
  DO I = 1,N
     ISPACE = N
     JPT    = I
     DO J = 1,I
        IPT      = IPT + 1
        DDF(IPT) = DDF1(JPT)
        ISPACE   = ISPACE - 1
        JPT      = JPT + ISPACE
     enddo
  enddo
  RETURN
END SUBROUTINE XTLFRR

SUBROUTINE XTLFRM ( DDF )
  !
  !     Mass-weight the second derivative matrix.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  !
  use image
  use psf
  implicit none
  !
  real(chm_real)  DDF(*)
  !
  INTEGER I,IATOM,IPT,ISYMM,J,JATOM,JSYMM
  real(chm_real)  MI,MIJ
  !
  ipt = 0
  Do iatom = 1,natom
     mi = one / sqrt(amass(iatom))
     Do i = 1,3
        Do jatom = 1,(iatom - 1)
           mij = mi / sqrt(amass(jatom))
           Do j = 1,3
              ipt = ipt + 1
              ddf(ipt) = mij * ddf(ipt)
           enddo
        enddo
        mij = mi * mi
        Do j = 1,i
           ipt = ipt + 1
           ddf(ipt) = mij * ddf(ipt)
        enddo
     enddo
  enddo
  !
  Do isymm = 2,xnsymm
     Do iatom = 1,natom
        mi = one / sqrt(amass(iatom))
        Do i = 1,3
           Do jsymm = 1,(isymm - 1)
              Do jatom = 1,natom
                 mij = mi / sqrt(amass(jatom))
                 Do j = 1,3
                    ipt = ipt + 1
                    ddf(ipt) = mij * ddf(ipt)
                 enddo
              enddo
           enddo
           Do jatom = 1,(iatom - 1)
              mij = mi / sqrt(amass(jatom))
              Do j = 1,3
                 ipt = ipt + 1
                 ddf(ipt) = mij * ddf(ipt)
              enddo
           enddo
           mij = mi * mi
           Do j = 1,i
              ipt = ipt + 1
              ddf(ipt) = mij * ddf(ipt)
           enddo
        enddo
     enddo
  enddo
  return
end SUBROUTINE XTLFRM
!

end module xtlfrq_m

