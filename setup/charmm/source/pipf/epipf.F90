#if KEY_PIPF==1 /*pipf_main*/
SUBROUTINE EPFDRV(IFRSTA,NATOM,JNB,INBLO,CG, &
     MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT,                   &
     LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC,ALP, &
     DD1,IUPT,QSECD &
     )
  !-----------------------------------------------------------------------
  !     THIS IS THE PIPF ENERGY/FORCE DRIVER ROUTINE, WHICH BRANCHES THE 
  !     PIPF CALCULATIONS TO TWO CASES:
  !       1. DYNAMICAL INDUCED DIPOLE MOMENT IN THE LAGRANGIAN (EPFDY)
  !       2. SELF-CONSISTENT ITERATIVELY DETERMINED DIPOLE (EPIPF)
  !       3. CALCULATE INDUCED DIPOLE BY MATRIX INVERSE (EPFINV)
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
  INTEGER IFRSTA, NATOM,IUPT(*)
  INTEGER JNB(*)
  INTEGER INBLO(*)
  real(chm_real) CG(*)
  INTEGER MAXROW,IAC(*),ITC(*)
  !     INTEGER IOFF(*)
  LOGICAL LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT,QSECD
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  real(chm_real) CTONNB,CTOFNB,EPS,E14FAC
  real(chm_real) ALP(*),DD1(*)
  !
  ! SKIP EPOL IF SPECIFIED.
  ! --- Better to move to the ENBOND argument list
  !

  IF (.NOT. QETERM(PIPF)) RETURN

  ! SKIP EPOL IF PIPF IS NOT SPECIFIED EXPLICITLY
  IF (.NOT. QPIPF) RETURN

  IF (QPFDYN) THEN
     !
     ! OBTAIN AN FIRST ORDER DIPOLE FOR THE FIRST DYNAMICS STEP
     !
     IF (NUFRS .GT. 1 .AND. QDYFST) THEN
        CALL DPFST(IFRSTA,NATOM,JNB,INBLO,CG, &
             MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
             LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC,ALP    &
             )
        QDYFST = .FALSE.
     ENDIF
     !
     ! COMPUTE ENERGY AND FORCE USING DYNAMICAL DIPOLES
     !
     CALL EPFDY(IFRSTA,NATOM,JNB,INBLO,CG, &
          MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
          LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC,ALP &
                                !    &     ,IOFF
          )
  ELSEIF(QMINV) THEN
     !
     !     CALCULATE DIPOLE BY MATRIX INVERSION
     !
     IF (NPFIM .GT. NPFPR) THEN
        CALL WRNDIE(-1,'<PFINT>', &
             'matrix inverse with image atoms not available')
        RETURN
     ELSE
        CALL EPFINV(IFRSTA,NATOM,JNB,INBLO,CG, &
             MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
             LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC,ALP, &
             .TRUE. &
                                !    &           ,IOFF
             )
        IF (QVPOL.AND.QSECD) THEN
           CALL VPIPF(IFRSTA,NATOM,JNB,INBLO,CG, &
                MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
                LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC,ALP, &
                DD1,IUPT,QSECD &
                                !    &           ,IOFF
                )
        ENDIF
     ENDIF
  ELSE

     !
     ! CLASSICAL SELF-CONSISTENT ITERATIVE PROCEDURE
     !
     IF (NPFIM .GT. NPFPR) THEN
        !
        ! IMAGE ATOM PRESENT, WE DEFER THE CALCULATION UNTIL THE
        ! PRIMARY AND IMAGE NONBONDED LIST ARE AVAILABLE 
        !
        IF (NATOM .LE. NPFPR) THEN 
           IFRSTAPP = IFRSTA 
           NATOMPP = NATOM
        ELSE 
           IFRSTAIP = IFRSTA
           NATOMIP = NATOM
           CALL EPFIMG(CG,MAXROW,IAC,ITC,DX,DY,DZ,X,Y,Z,ALP    &
                                !    &         ,IOFF
                )
        ENDIF
     ELSE
        CALL EPIPF(IFRSTA,NATOM,JNB,INBLO,CG, &
             MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
             LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC,ALP    &
                                !    &      ,IOFF
             )
     ENDIF
  ENDIF
  !
  RETURN 
END SUBROUTINE EPFDRV

SUBROUTINE EPIPF(IFRSTA,NATOM,JNB,INBLO,CG, &
     MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
     LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC,ALP &
                                !    &  ,IOFF
     )
  !-----------------------------------------------------------------------
  !     THIS ROUTINE ALLOCATES MEMORY FOR COMPUTING POLARIZATION
  !     ENERGIES AND FORCES USING THE ITERATIVE PROCEDURE. THE ACTUAL 
  !     CACLULATION IS DONE BY SUBROUTINE EPIPF
  !
  !     IFRSTA - first atom to look at in INBLO
  !     NATOM  - last atom to look at in INBLO (number of atoms)
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
  use consta
  use dimens_fcm
  use number
  use memory
  use exfunc
  use energym
  use pipfm
  implicit none
  !
  real(chm_real) EPOL
  INTEGER IFRSTA, NATOM
  INTEGER JNB(*)
  INTEGER INBLO(*)
  real(chm_real) CG(*)
  INTEGER MAXROW,IAC(*),ITC(*)
  !     INTEGER IOFF(*)
  LOGICAL LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  real(chm_real) CTONNB,CTOFNB,EPS,E14FAC
  real(chm_real) ALP(*)
  !
  INTEGER I,J,NB,ITEMP,NPR,N3,NB6
  real(chm_real),allocatable,dimension(:) :: IDMUIND
  real(chm_real),allocatable,dimension(:) :: IDDT
  real(chm_real),allocatable,dimension(:) :: IEFIELD
  real(chm_real),allocatable,dimension(:) :: IEIND
  real(chm_real),allocatable,dimension(:) :: ITXWW
  integer,allocatable,dimension(:) :: IIPOL
  !
  NB = 0
  ITEMP = 0

  DO I=IFRSTA,NATOM
     NPR=INBLO(I)-ITEMP
     ITEMP=INBLO(I)
     IF(NPR.GT.0) NB = NB + NPR
  END DO
  !
  !     ALLOCATE MEMORY 
  !
  N3 = 3*NATOM 
  NB6 = 6*NB
  call chmalloc('epipf.src','EPIPF','IDMUIND',N3,crl=IDMUIND)
  call chmalloc('epipf.src','EPIPF','IDDT',N3,crl=IDDT)
  call chmalloc('epipf.src','EPIPF','IEFIELD',N3,crl=IEFIELD)
  call chmalloc('epipf.src','EPIPF','IEIND',N3,crl=IEIND)
  call chmalloc('epipf.src','EPIPF','ITXWW',NB6,crl=ITXWW)
  call chmalloc('epipf.src','EPIPF','IIPOL',NATOM,intg=IIPOL)

  call EPIPF2(IFRSTA,NATOM,JNB,INBLO,CG,MAXROW,IAC,ITC,LELEC,LCONS, &
       LSHFT,LVSHFT,LFSWT,LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
       EPOL,ALP,IDMUIND,IDDT,IEFIELD,IEIND,ITXWW,IIPOL)
  !
  !     FREE MEMORY
  !
  call chmdealloc('epipf.src','EPIPF','IDMUIND',N3,crl=IDMUIND)
  call chmdealloc('epipf.src','EPIPF','IDDT',N3,crl=IDDT)
  call chmdealloc('epipf.src','EPIPF','IEFIELD',N3,crl=IEFIELD)
  call chmdealloc('epipf.src','EPIPF','IEIND',N3,crl=IEIND)
  call chmdealloc('epipf.src','EPIPF','ITXWW',NB6,crl=ITXWW)
  call chmdealloc('epipf.src','EPIPF','IIPOL',NATOM,intg=IIPOL)

  RETURN
END SUBROUTINE EPIPF
!
SUBROUTINE EPIPF2(IFRSTA,NATOM,JNB,INBLO,CG, &
     MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
     LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
     EPOL,ALP,DMUIND,DDT,EFIELD,EIND,TXWW,IPOL &
                                !    &  ,IOFF
     )
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CALCULATES POLARIZATION ENERGIES AND FORCES
  !     VIA THE SELF-CONSISTENT ITERATIVE PROCESS
  !
  !     IFRSTA - first atom to look at in INBLO
  !     NATOM  - last atom to look at in INBLO (number of atoms)
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
  !     DMUIND - induced dipole for current iteration
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
  INTEGER IFRSTA, NATOM
  INTEGER JNB(*)
  INTEGER INBLO(*)
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
  real(chm_real) DMUIND(3,*),EFIELD(3,*),TXWW(6,*)   
  real(chm_real) DDT(3,*),EIND(3,*)
  real(chm_real) DMUNEW,DDIFF
  real(chm_real) E14M1,E14F,CGT2,CGT
  INTEGER IPOL(*)
  !
  !      GET THE ZEROTH ORDER FIELD AND POLAR TENSOR
  !
  CALL GZERO(EFIELD,3,NATOM)
  CALL ETINIT(IFRSTA,NATOM,JNB,INBLO,CG, &
       DX,DY,DZ,X,Y,Z,EFIELD,TXWW,PFCTOF, &
       ALP,DPFAC,IAC,ITC,NPDAMP,QPFEX)
  !
  !     SET DIPOLE CENTER TAG TO ZERO
  !
  DO I = 1, NATOM
     IPOL(I) = 0
  END DO
  !
  !     ZERO INITIAL INDUCED DIPOLE 
  !
  CALL GZERO(DMUIND,3,NATOM)

  NITER = 0
100 NITER = NITER+1
  CALL COORTR(DDT,DMUIND,3*NATOM)
  CALL GZERO(EIND,3,NATOM)
  !
  NB = 0
  ITEMP = 0
  !
  !=======================================================================
  !   MAIN LOOP BEGIN
  !=======================================================================
  DO I=IFRSTA,NATOM
     NPR=INBLO(I)-ITEMP
     ITEMP=INBLO(I)
     IF(NPR.GT.0) THEN

        !         I1=ITC(IAC(I))
        !         IACI=IOFF(I1)

        !
        ! not sure about thie section (group-group?)
        !         IF (ELECFG) THEN
        !           CGT=CGF*CG(I)
        !           ELCFG=(CGT.NE.0.0)
        !         ELSE
        !           CGT=ZERO
        !         ENDIF
        !

        !
        DO JPR=1,NPR
           NB=NB+1
           IF (JNB(NB).LT.0) THEN
              CGT2=CGT*E14FAC
              E14F = E14M1
              J=-JNB(NB)

              !             J1=ITC(IAC(J))
              !             IF (I1.LT.J1) THEN
              !               IC=IOFF(J1)+I1+MAXROW
              !             ELSE
              !               IC=IACI+J1+MAXROW
              !             ENDIF

           ELSE
              CGT2=CGT
              E14F=ZERO
              J=JNB(NB)

              !             J1=ITC(IAC(J))
              !             IF (I1.LT.J1) THEN
              !               IC=IOFF(J1)+I1
              !             ELSE
              !               IC=IACI+J1
              !             ENDIF

           ENDIF
           !
           !      COMPUTE INDUCED FIELD
           !      SINCE WE USE NONBONDED PAIR (I<J) (IMAGE LIST I>J), WE NEED 
           !      TO INCLUDE CONTRIBUTION TO J FOR I>J (IMAGE LIST I<J) PAIR 
           !      EXPLICITLY 
           !
           !      NOTE:  TXWW(1:6,NB) = TXWW(1:6,I,J) = TXWW(1:6,J,I)
           !
           EIND(1,I) = EIND(1,I)+TXWW(1,NB)*DMUIND(1,J) &
                +TXWW(2,NB)*DMUIND(2,J)+TXWW(4,NB)*DMUIND(3,J)   
           EIND(2,I) = EIND(2,I)+TXWW(2,NB)*DMUIND(1,J) &
                +TXWW(3,NB)*DMUIND(2,J)+TXWW(5,NB)*DMUIND(3,J)
           EIND(3,I) = EIND(3,I)+TXWW(4,NB)*DMUIND(1,J) &
                +TXWW(5,NB)*DMUIND(2,J)+TXWW(6,NB)*DMUIND(3,J)   

           EIND(1,J) = EIND(1,J)+TXWW(1,NB)*DMUIND(1,I) &
                +TXWW(2,NB)*DMUIND(2,I)+TXWW(4,NB)*DMUIND(3,I)
           EIND(2,J) = EIND(2,J)+TXWW(2,NB)*DMUIND(1,I) &
                +TXWW(3,NB)*DMUIND(2,I)+TXWW(5,NB)*DMUIND(3,I)
           EIND(3,J) = EIND(3,J)+TXWW(4,NB)*DMUIND(1,I) &
                +TXWW(5,NB)*DMUIND(2,I)+TXWW(6,NB)*DMUIND(3,I)   

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
  !      NEW INDUCED DIPOLES
  !      SIGN FOLLOWS EQ. (4) in CHEM. PHYS. LETT. 1990, 166, 180.
  !
  NCOUNT = 0
  DMUNEW = 0.0D+00
  DO I=IFRSTA,NATOM
     IF(IPOL(I).EQ.1) THEN
        NCOUNT = NCOUNT + 1
        I1=ITC(IAC(I))
        DMUIND(1,I) = ALP(I1)*(EFIELD(1,I)+EIND(1,I))
        DMUIND(2,I) = ALP(I1)*(EFIELD(2,I)+EIND(2,I))
        DMUIND(3,I) = ALP(I1)*(EFIELD(3,I)+EIND(3,I))
        DMUNEW = DMUNEW+(DMUIND(1,I)-DDT(1,I))**2+ &
             (DMUIND(2,I)-DDT(2,I))**2+ &
             (DMUIND(3,I)-DDT(3,I))**2
     ENDIF
  ENDDO

  ! DIPOLE MOMENT UNIT, CONVERT ATOMIC CHARGE AND aNGSTROM TO DEBYE
  !
  DDIFF= 4.8028132D+00*DSQRT(DMUNEW/DBLE(NCOUNT))           

  IF(NITER.GT.15) THEN
     IF (WRNLEV.GE.2) THEN 
        WRITE(OUTU,*) 'CONVERGENCE PROBLEMS in EPIPF:',NITER
        WRITE(OUTU,10) NITER, DDIFF, DTHRES
     ENDIF
10   FORMAT(1X,'EPIPF ITER', I3, ':', E15.5, E15.5)
     if(niter.gt.ITRMX.AND.DDIFF.GT.10.D+00*DTHRES) THEN
        EPOL = 99.99D+00
        IF (WRNLEV.GE.2) THEN
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
  DO I = IFRSTA,NATOM
     IF(IPOL(I).EQ.1) THEN
        EPOL = EPOL+DMUIND(1,I)*EFIELD(1,I)+DMUIND(2,I) &
             *EFIELD(2,I)+DMUIND(3,I)*EFIELD(3,I)
     ENDIF
  ENDDO
  !
  ! ENERGY, CONVERT SI TO KCAL/MOL
  !
  EPOL = -0.5D0*CCELEC*EPOL
  !
  IF(PRNLEV.GE.7) THEN
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
  CALL DPIPF(IFRSTA,NATOM,JNB,INBLO,CG, &
       MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
       LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
       TXWW,DMUIND,PFCTOF, &
       ALP,DPFAC,NPDAMP,QPFEX &
                                !    &  ,IOFF
       )

  !
  ! PRINT OUT DIPOLE INFOR
  !
  IF (NAVDIP .GT. 0) THEN
     CALL PRDIP(NATOM,NAVDIP,X,Y,Z,CG,DMUIND)
  END IF

  RETURN
END SUBROUTINE EPIPF2

!
SUBROUTINE ETINIT(IFRSTA,NATOM,JNB,INBLO,CG, &
     DX,DY,DZ,X,Y,Z,EZERO,TXWWT,PFCTOF, &
     ALP,DPFAC,IAC,ITC,NPDAMP,QPFEX)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CALCULATES INITIAL ELECTRIC FIELD AND
  !     DIPOLE TENSOR
  !
  !     IFRSTA - first atom to look at in INBLO
  !     NATOM  - last atom to look at in INBLO (number of atoms)
  !     JNB    - nonbond pair list  (INBLO(NATOM))
  !     INBLO  - pointers into JNB  (NATOM)
  !     CG     - charges  (NATOM)
  !     MAXROW  - offset for 1-4 interaction
  !     LELEC,,,,,,,, - logical flags used in BNBND.FCM
  !     CTONNB,CTOFNB - switching function specifiers in real space
  !     EPS - dielectric constant
  !     EPOL   - polarization energy
  !     ALP    - polarizability
  !     EZERO  - permanent electric field
  !     TXWWT  - dipole tensor
  !     PFCTOF - polariziation energy cut-off
  !     IAC,ITC - polarizability index
  !     NPDAMP - polarization damping option
  !     DPFAC - damping factor (gaussian width in Thole roh2)
  !     QPFEX - tag to exclude 1-4 polarization
  !----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use number
  use exfunc
#if KEY_PBOUND==1
  use pbound  
#endif
  implicit none
#if KEY_PBOUND==1
  real(chm_real) CORR  
#endif
  !
  INTEGER IFRSTA, NATOM
  INTEGER JNB(*)
  INTEGER INBLO(*)
  real(chm_real) CG(*)
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  real(chm_real) EZERO(3,NATOM), TXWWT(6,*)
  !
  INTEGER I,J,JPR,NB,NPR,ITEMP
  real(chm_real) XX,YY,ZZ,R1,R2,R3,QR3I,QR3J,    &
       RR3,RR3XX,RR3YY,RR3ZZ
  !
  real(chm_real) ALP(*),DPFAC,U3,AU3,FLMD3,FLMD5
  INTEGER IAC(*),ITC(*),NPDAMP,I1,J1
  !
  real(chm_real) RIJ2,PFCTOF,PFCTOF2
  LOGICAL QPFEX
  PFCTOF2=PFCTOF*PFCTOF
  !
  !     DO I = IFRSTA,NATOM 
  !        EZERO(1,I) = 0.0D+00
  !        EZERO(2,I) = 0.0D+00
  !        EZERO(3,I) = 0.0D+00
  !     ENDDO
  !
  NB = 0
  ITEMP = 0

  ! DO BLOCK EXPANSION OF CODE TO IMPROVE EFFICIENCY
  IF(PFCTOF.LE.0.0D0 .AND. NPDAMP.EQ.0) THEN
#undef PIPF_CTOF
#undef PIPF_DAMP
#include "epipf.inc"
  ELSE IF(PFCTOF.GT.0.0D0 .AND. NPDAMP.EQ.0) THEN
#define PIPF_CTOF 1
#include "epipf.inc"
#undef PIPF_CTOF
  ELSE IF(PFCTOF.LE.0.0D0 .AND. NPDAMP.EQ.1) THEN
#define PIPF_DAMP 1
#include "epipf.inc"
  ELSE
#define PIPF_CTOF 1
#include "epipf.inc"
#undef PIPF_CTOF
#undef PIPF_DAMP
  ENDIF
  RETURN
END SUBROUTINE ETINIT

!-------------------------------------------------------------------
SUBROUTINE COORTR(A,B,N)
  !-------------------------------------------------------------------
  use chm_kinds
  implicit none
  INTEGER N
  real(chm_real) A(*),B(*)
  !
  INTEGER I
  DO I = 1,N
     A(I) = B(I)
  ENDDO
  RETURN 
END SUBROUTINE COORTR

!------------------------------------------------------------------
SUBROUTINE GZERO(A,N,M)
  !------------------------------------------------------------------
  use chm_kinds
  use number
  implicit none
  !     ZERO THE ARRAY A(N,M). FOR TWO DIMENSIONAL ARRAYS,
  !     ALWAYS ZERO THE ENTIRE ARRAY.
  !
  INTEGER N,M
  real(chm_real) A(*)
  !
  INTEGER J,K
  K = N
  IF (M.GT.0) K = N*M
  A(1:k) = zero
  RETURN
END SUBROUTINE GZERO

!------------------------------------------------------------------
SUBROUTINE PRDIP(NATOM,NAVDIP,X,Y,Z,CG,DMUIND)
  !------------------------------------------------------------------
  use chm_kinds
  use consta
  use dimens_fcm
  use number
  use stream
  implicit none
  !
  INTEGER NATOM, NAVDIP
  real(chm_real) X(*), Y(*), Z(*), CG(*), DMUIND(3,*)
  !
  INTEGER NMOL,I,J,K
  real(chm_real) PDP,DPI,TDP,PDPX,PDPY,PDPZ,DPIX,DPIY,DPIZ, &
       TDPX,TDPY,TDPZ,ANGPI,AMPP,AMPI,AMPT

  NMOL = NATOM/NAVDIP
  PDP = 0.0D0
  DPI = 0.0D0
  TDP = 0.0D0
  ANGPI = 0.0D0
  K = 0
  DO I = 1, NMOL
     PDPX = 0.0D0
     PDPY = 0.0D0
     PDPZ = 0.0D0
     DPIX = 0.0D0
     DPIY = 0.0D0
     DPIZ = 0.0D0
     DO J = 1, NAVDIP
        K = K + 1
        PDPX = PDPX + X(K)*CG(K)
        PDPY = PDPY + Y(K)*CG(K)
        PDPZ = PDPZ + Z(K)*CG(K)
        DPIX = DPIX + DMUIND(1,K)
        DPIY = DPIY + DMUIND(2,K)
        DPIZ = DPIZ + DMUIND(3,K)
        TDPX = PDPX + DPIX
        TDPY = PDPY + DPIY
        TDPZ = PDPZ + DPIZ
     END DO
     AMPP = SQRT(PDPX*PDPX+PDPY*PDPY+PDPZ*PDPZ)
     AMPI = SQRT(DPIX*DPIX+DPIY*DPIY+DPIZ*DPIZ)
     AMPT = SQRT(TDPX*TDPX+TDPY*TDPY+TDPZ*TDPZ)
     PDP = PDP + AMPP
     DPI = DPI + AMPI
     TDP = TDP + AMPT
     ANGPI = ANGPI + ACOS((PDPX*DPIX+PDPY*DPIY+PDPZ*DPIZ)/ &
          (AMPP*AMPI))/PI*180.0D0
  END DO
  PDP = PDP / DBLE(NMOL) * 4.8028132D+00
  DPI = DPI / DBLE(NMOL) * 4.8028132D+00
  TDP = TDP / DBLE(NMOL) * 4.8028132D+00
  ANGPI = ANGPI / NMOL
  WRITE(OUTU,200) TDP, PDP, DPI, ANGPI
200 FORMAT(' Ave dips (db): tot, perm, ind, theta = ', &
       F7.3, F7.3, F7.3, F8.3)
  RETURN
END SUBROUTINE PRDIP
#endif /* (pipf_main)*/
subroutine epipf_dummy()
  return
end subroutine epipf_dummy
