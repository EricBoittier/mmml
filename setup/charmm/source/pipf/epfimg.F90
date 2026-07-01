#if KEY_PIPF==1 /*pipf_main*/
SUBROUTINE EPFIMG(CG,MAXROW,IAC,ITC,DX,DY,DZ,X,Y,Z,ALP)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE ALLOCATES MEMORY FOR COMPUTING POLARIZATION
  !     ENERGIES AND FORCES USING THE ITERATIVE PROCEDURE WHERE
  !     PRIODIC CONDITION IS TREATED BY USING IMAGE ATOMS. THE ACUTAL
  !     CACLULATION IS DONE BY SUBROUTINE EPFIMG2
  !
  !     IFRSTAP - first atom to look at in INBLOP
  !     NATOMP  - last atom to look at in INBLOP (number of atoms)
  !     IFRSTAI - first atom to look at in INBLOI 
  !     NATOMI  - last atom to look at in INBLOI (number of atoms)
  !     JNB    - nonbond pair list  (INBLO(NATOM))
  !     INBLO  - pointers into JNB  (NATOM)
  !     CG     - charges  (NATOM)
  !     MAXROW  - offset for 1-4 interaction
  !     LELEC - logical flags used in BNBND.FCM
  !     CTONNB,CTOFNB - switching function specifiers in real space
  !     EPS - dielectric constant
  !     EPOL   - polarization energy
  !     ALP    - polarizability (passed in param.f90)
  !----------------------------------------------------------------------
  use chm_kinds
  use chm_types
  use bases_fcm
  use consta
  use dimens_fcm
  use number
  use memory
  use exfunc
  use energym
  use inbnd
  use image
  use pipfm
  implicit none
  !
  real(chm_real) EPOL
  real(chm_real) CG(*)
  INTEGER MAXROW,IAC(*),ITC(*)
  !     INTEGER IOFF(*)
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  real(chm_real) ALP(*)
  !
  real(chm_real),allocatable,dimension(:) :: IDDT
  real(chm_real),allocatable,dimension(:) :: IEFIELD
  real(chm_real),allocatable,dimension(:) :: IEIND
  real(chm_real),allocatable,dimension(:) :: ITXWWPP
  real(chm_real),allocatable,dimension(:) :: ITXWWIP
  integer,allocatable,dimension(:) :: IIPOL

  INTEGER I,J,NB,ITEMP,NPR,N3,NBPP,NBIP,NB6PP,NB6IP
  !
  CALL GETNB(BNBND%INBLO,IFRSTAPP,NATOMPP,NBPP)
  CALL GETNB(BIMAG%IMBLO,IFRSTAIP,NATOMIP,NBIP)

  !
  !     UPDATE THE TOTAL NUMBER OF ATOMS INCLUDING IMAGE ATOMS
  !     (WE USE A LOCAL COPY OF THAT NUMBER)
  !
  NPFIM = NATIM
  !
  !     ALLOCATE MEMORY 
  !
  N3 = 3*NPFIM
  NB6PP = 6*NBPP
  NB6IP = 6*NBIP
  IF(PFMODE .EQ. 1) THEN
     if (.not.allocated(INDP)) call wrndie(-5,'<EPFIMG>', &
          'INDP not allocated for PFMODE=1.')
  ELSE
     call chmalloc('epfimg.src','EPFIMG','INDP',3,NPFIM,crl=INDP)
  ENDIF

  call chmalloc('epfimg.src','EPFIMG','IDDT',N3,crl=IDDT)
  call chmalloc('epfimg.src','EPFIMG','IEFIELD',N3,crl=IEFIELD)
  call chmalloc('epfimg.src','EPFIMG','IEIND',N3,crl=IEIND)
  call chmalloc('epfimg.src','EPFIMG','ITXWWPP',NB6PP,crl=ITXWWPP)
  call chmalloc('epfimg.src','EPFIMG','ITXWWIP',NB6IP,crl=ITXWWIP)
  call chmalloc('epfimg.src','EPFIMG','IIPOL',NPFIM,intg=IIPOL)

  call EPFIMG2(BNBND%JNB,BNBND%INBLO,BIMAG%IMJNB, &
       BIMAG%IMBLO,BIMAG%IMATTR,CG,MAXROW,IAC,ITC,LELEC, &
       LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS, &
       E14FAC,EPOL,ALP,IDDT,IEFIELD,IEIND,ITXWWPP, &
       ITXWWIP,IIPOL)
  !
  !     FREE MEMORY
  !
  IF (PFMODE .NE. 1) THEN
     call chmdealloc('epfimg.src','EPFIMG','INDP',3,NPFIM,crl=INDP)
  ENDIF

  call chmdealloc('epfimg.src','EPFIMG','IDDT',N3,crl=IDDT)
  call chmdealloc('epfimg.src','EPFIMG','IEFIELD',N3,crl=IEFIELD)
  call chmdealloc('epfimg.src','EPFIMG','IEIND',N3,crl=IEIND)
  call chmdealloc('epfimg.src','EPFIMG','ITXWWPP',NB6PP,crl=ITXWWPP)
  call chmdealloc('epfimg.src','EPFIMG','ITXWWIP',NB6IP,crl=ITXWWIP)
  call chmdealloc('epfimg.src','EPFIMG','IIPOL',NPFIM,intg=IIPOL)

  RETURN
END SUBROUTINE EPFIMG
!
SUBROUTINE EPFIMG2(JNBPP,INBLOPP,JNBIP,INBLOIP, &
     IMATTR,CG,MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT, &
     LFSWT,LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
     EPOL,ALP,DDT,EFIELD,EIND,TXWWPP,TXWWIP,IPOL    &
     !    &  ,IOFF
     )
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CALCULATES POLARIZATION ENERGIES AND FORCES
  !     VIA THE SELF-CONSISTENT ITERATIVE PROCESS
  !
  !     JNB    - nonbond pair list  (INBLO(NATOM))
  !     INBLO  - pointers into JNB  (NATOM)
  !     CG     - charges  (NATOM)
  !     MAXROW  - offset for 1-4 interaction 
  !     LELEC - logical flags used in BNBND.FCM
  !     CTONNB,CTOFNB - switching function specifiers in real space
  !     EPS - dielectric constant
  !     EPOL   - polarization energy
  !     ALP    - polarizability (passed in param.f90)
  !
  !     INDP - induced dipole for current iteration (pipf_ltm)
  !     DDT - induced dipole for previous iteration
  !     EFIELD - permenent electric field
  !     EIND - induced electric field
  !     TXWW - rank 2 polar tensor
  !     IPOL - tag to mark an atom as a dipole center
  !----------------------------------------------------------------------
  use chm_kinds
  use consta
  use dimens_fcm
  use number
  use exfunc
  use energym
  use stream
  use pipfm
  implicit none
  !
  real(chm_real) EPOL
  INTEGER JNBPP(*),JNBIP(*)
  INTEGER INBLOPP(*),INBLOIP(*)
  INTEGER IMATTR(*)
  real(chm_real) CG(*)
  INTEGER MAXROW,IAC(*),ITC(*)
  !     INTEGER IOFF(*)
  LOGICAL LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  real(chm_real) CTONNB,CTOFNB,EPS,E14FAC
  real(chm_real) ALP(*)
  !
  INTEGER NITER, NCOUNT
  INTEGER I,J,JPR,NB,NPR,ITEMP,I1
  real(chm_real) EFIELD(3,*),TXWWPP(6,*),TXWWIP(6,*)
  real(chm_real) DDT(3,*),EIND(3,*)
  real(chm_real) DMUNEW,DDIFF
  real(chm_real) E14M1,E14F,CGT2,CGT
  INTEGER IPOL(*)
  !
  !      GET THE ZEROTH ORDER FIELD AND POLAR TENSOR
  !
  CALL GZERO(EFIELD,3,NPFIM)

  CALL ETINIT(IFRSTAPP,NATOMPP,JNBPP,INBLOPP,CG, &
       DX,DY,DZ,X,Y,Z,EFIELD,TXWWPP,PFCTOF, &
       ALP,DPFAC,IAC,ITC,NPDAMP,QPFEX)

  CALL ETINIT(IFRSTAIP,NATOMIP,JNBIP,INBLOIP,CG, &
       DX,DY,DZ,X,Y,Z,EFIELD,TXWWIP,PFCTOF, &
       ALP,DPFAC,IAC,ITC,NPDAMP,QPFEX)
  !
  !     TRANSFER THE PERMENENT FIELD ON IMAGE ATOM TO 
  !     PRIMARY ATOMS
  ! 
  CALL DPTRANSI(EFIELD,NPFPR,NPFIM,IMATTR)
  !
  !     SET DIPOLE CENTER TAG TO ZERO
  !
  DO I = 1, NPFIM
     IPOL(I) = 0
  END DO
  !
  !     ZERO INITIAL INDUCED DIPOLE 
  !
  IF(PFMODE .EQ. 1) THEN
     IF(QFSTDP) THEN
        QFSTDP=.FALSE.
        CALL GZERO(INDP,3,NPFIM)
     ENDIF
  ELSE
     CALL GZERO(INDP,3,NPFIM)
  ENDIF

  NITER = 0
100 NITER = NITER+1
  CALL COORTR(DDT,INDP,3*NPFIM)
  CALL GZERO(EIND,3,NPFIM)
  !
  !=======================================================================
  !   MAIN LOOP BEGIN (P-P Non-bonded list)
  !=======================================================================
  NB = 0
  ITEMP = 0
  DO I=IFRSTAPP,NATOMPP
     NPR=INBLOPP(I)-ITEMP
     ITEMP=INBLOPP(I)
     IF(NPR.GT.0) THEN
        DO JPR=1,NPR
           NB=NB+1
           IF (JNBPP(NB).LT.0) THEN
              J=-JNBPP(NB)
           ELSE
              J=JNBPP(NB)
           ENDIF
           !
           !      COMPUTE INDUCED FIELD
           !      SINCE WE USE NONBONDED PAIR (I<J), WE NEED TO INCLUDE
           !      CONTRIBUTION TO J FOR I>J PAIR EXPLICITLY 
           !
           !      NOTE:  TXWW(1:6,NB) = TXWW(1:6,I,J) = TXWW(1:6,J,I)
           !
           EIND(1,I) = EIND(1,I)+TXWWPP(1,NB)*INDP(1,J) &
                +TXWWPP(2,NB)*INDP(2,J)+TXWWPP(4,NB)*INDP(3,J)   
           EIND(2,I) = EIND(2,I)+TXWWPP(2,NB)*INDP(1,J) &
                +TXWWPP(3,NB)*INDP(2,J)+TXWWPP(5,NB)*INDP(3,J)
           EIND(3,I) = EIND(3,I)+TXWWPP(4,NB)*INDP(1,J) &
                +TXWWPP(5,NB)*INDP(2,J)+TXWWPP(6,NB)*INDP(3,J)      

           EIND(1,J) = EIND(1,J)+TXWWPP(1,NB)*INDP(1,I) &
                +TXWWPP(2,NB)*INDP(2,I)+TXWWPP(4,NB)*INDP(3,I)
           EIND(2,J) = EIND(2,J)+TXWWPP(2,NB)*INDP(1,I) &
                +TXWWPP(3,NB)*INDP(2,I)+TXWWPP(5,NB)*INDP(3,I)
           EIND(3,J) = EIND(3,J)+TXWWPP(4,NB)*INDP(1,I) &
                +TXWWPP(5,NB)*INDP(2,I)+TXWWPP(6,NB)*INDP(3,I)   

           !       MARK J AS A DIPOLE CENTER
           IF (NITER .EQ. 1) IPOL(J) = 1

           !
           ! END JPR LOOP
           !
        ENDDO

        !       MARK I AS A DIPOLE CENTER
        IF (NITER .EQ. 1) IPOL(I) = 1

     ENDIF
     !
     ! END MAIN LOOP
     !
  ENDDO

  !=======================================================================
  !   MAIN LOOP BEGIN (I-P Non-bonded list)
  !=======================================================================
  NB = 0
  ITEMP = 0
  DO I=IFRSTAIP,NATOMIP
     NPR=INBLOIP(I)-ITEMP
     ITEMP=INBLOIP(I)
     IF(NPR.GT.0) THEN
        DO JPR=1,NPR
           NB=NB+1
           IF (JNBIP(NB).LT.0) THEN
              J=-JNBIP(NB)
           ELSE
              J=JNBIP(NB)
           ENDIF
           !
           !      COMPUTE INDUCED FIELD
           !      SINCE WE USE NONBONDED PAIR (I<J), WE NEED TO INCLUDE
           !      CONTRIBUTION TO J FOR I>J PAIR EXPLICITLY
           !
           !      NOTE:  TXWW(1:6,NB) = TXWW(1:6,I,J) = TXWW(1:6,J,I)
           !
           EIND(1,I) = EIND(1,I)+TXWWIP(1,NB)*INDP(1,J) &
                +TXWWIP(2,NB)*INDP(2,J)+TXWWIP(4,NB)*INDP(3,J)
           EIND(2,I) = EIND(2,I)+TXWWIP(2,NB)*INDP(1,J) &
                +TXWWIP(3,NB)*INDP(2,J)+TXWWIP(5,NB)*INDP(3,J)
           EIND(3,I) = EIND(3,I)+TXWWIP(4,NB)*INDP(1,J) &
                +TXWWIP(5,NB)*INDP(2,J)+TXWWIP(6,NB)*INDP(3,J)   

           EIND(1,J) = EIND(1,J)+TXWWIP(1,NB)*INDP(1,I) &
                +TXWWIP(2,NB)*INDP(2,I)+TXWWIP(4,NB)*INDP(3,I)
           EIND(2,J) = EIND(2,J)+TXWWIP(2,NB)*INDP(1,I) &
                +TXWWIP(3,NB)*INDP(2,I)+TXWWIP(5,NB)*INDP(3,I)
           EIND(3,J) = EIND(3,J)+TXWWIP(4,NB)*INDP(1,I) &
                +TXWWIP(5,NB)*INDP(2,I)+TXWWIP(6,NB)*INDP(3,I)

           !       MARK J AS A DIPOLE CENTER
           IF (NITER .EQ. 1) IPOL(J) = 1
           !
           ! END JPR LOOP
           !
        ENDDO

        !       MARK I AS A DIPOLE CENTER
        IF (NITER .EQ. 1) IPOL(I) = 1

     ENDIF
     !
     ! END MAIN LOOP
     !
  ENDDO

  !
  ! TRANSFER THE INDUCED FIELD ON IMAGE ATOMS TO PRIMARY ATOMS
  !
  CALL DPTRANSI(EIND,NPFPR,NPFIM,IMATTR)

  !
  ! NEW INDUCED DIPOLES FOR PRIMARY ATOMS
  ! SIGN FOLLOWS EQ. (4) in CHEM. PHYS. LETT. 1990, 166, 180.
  !
  NCOUNT = 0
  DMUNEW = 0.0D+00
  DO I=1,NPFPR
     IF(IPOL(I).EQ.1) THEN 
        NCOUNT = NCOUNT + 1
        I1=ITC(IAC(I))
        INDP(1,I) = ALP(I1)*(EFIELD(1,I)+EIND(1,I))
        INDP(2,I) = ALP(I1)*(EFIELD(2,I)+EIND(2,I))   
        INDP(3,I) = ALP(I1)*(EFIELD(3,I)+EIND(3,I))
        DMUNEW = DMUNEW+(INDP(1,I)-DDT(1,I))**2+ &
             (INDP(2,I)-DDT(2,I))**2+ &
             (INDP(3,I)-DDT(3,I))**2
     ENDIF
  ENDDO

  !
  ! COPY PRIMARY DIPOLE TO IMAGE ATOMS
  !
  CALL CPDPIMG(IMATTR,INDP)

  ! DIPOLE MOMENT UNIT, CONVERT ATOMIC CHARGE AND aNGSTROM TO DEBYE
  !
  DDIFF= 4.8028132D+00*DSQRT(DMUNEW/DBLE(NCOUNT))           

  IF(NITER.GT.15) THEN
     IF (WRNLEV .GE. 2) THEN
        WRITE(OUTU,*) 'CONVERGENCE PROBLEMS in EPIPF:',NITER
        WRITE(OUTU,10) NITER, DDIFF, DTHRES
     ENDIF
10   FORMAT(1X,'EPIPF ITER', I3, ':', E15.5, E15.5)
     if(niter.gt.ITRMX.AND.DDIFF.GT.10.D+00*DTHRES) THEN
        EPOL = 99.99D+00
        IF (WRNLEV .GE. 2) THEN
           WRITE(OUTU,*) 'CONVERGENCE FAILED - GIVE A LARGE ENERGY'
        ENDIF
        RETURN
     ENDIF
  ENDIF
  !
  ! CONVERGE BOTH DIPOLE AND ENERGY
  !
  IF(DDIFF.GT.DTHRES) GOTO 100

  !
  !     FINAL POLARIZATION ENERGY
  !
  EPOL = 0.0D+00
  DO I = 1,NPFPR
     IF(IPOL(I).EQ.1) THEN
        EPOL = EPOL+INDP(1,I)*EFIELD(1,I)+INDP(2,I) &
             *EFIELD(2,I)+INDP(3,I)*EFIELD(3,I)
     ENDIF
  ENDDO
  !
  ! ENERGY, CONVERT SI TO KCAL/MOL
  !
  EPOL = -0.5D0*CCELEC*EPOL
  !
  IF (PRNLEV .GE. 7) THEN
     WRITE(OUTU,*) 'Polarization Energy (EPOL):', EPOL
  ENDIF
  !
  ! STORE EPOL TO ENERGY TERM COMMON BLOCK
  ! --- BETTER TO MOVE TO THE ENBOND ARGUMENT LIST 
  !     MODIFIED TO CUMULATIVE ADDITION TO HANDEL BOTH 
  !     PRIMARY AND IMAGE POLARIZATION ENERGY
  !         ETERM(PIPF) = EPOL
  !
  ETERM(PIPF) = ETERM(PIPF)+EPOL
  !
  ! COMPUTE THE PIPF FORCE
  !
  CALL DPFIMG(IFRSTAPP,NATOMPP,JNBPP,INBLOPP, &
       IFRSTAIP,NATOMIP,JNBIP,INBLOIP,IMATTR,CG, &
       MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
       LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
       TXWWPP,TXWWIP,INDP,PFCTOF, &
       ALP,DPFAC,NPDAMP,QPFEX &
       !    &  ,IOFF
       )

  !
  ! PRINT OUT DIPOLE INFOR
  !
  IF (NAVDIP .GT. 0) THEN
     CALL PRDIP(NPFPR,NAVDIP,X,Y,Z,CG,INDP)
  END IF

  RETURN
END SUBROUTINE EPFIMG2

!---------------------------------------------
SUBROUTINE GETNB(INBLO,IFRSTA,NATOM,NB)
  !---------------------------------------------
  INTEGER INBLO(*),NB,IFRSTA,NATOM
  INTEGER ITEMP,NPR
  !
  NB = 0
  ITEMP = 0
  DO I=IFRSTA,NATOM
     NPR=INBLO(I)-ITEMP
     ITEMP=INBLO(I)
     IF(NPR.GT.0) NB = NB + NPR
  END DO
  !
  RETURN
END SUBROUTINE GETNB

#endif /* (pipf_main)*/
subroutine epfimg_dummy()
  return
end subroutine epfimg_dummy

