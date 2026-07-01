#if KEY_PIPF==1 /*pipf_main*/
SUBROUTINE EPFINV(IFRSTA,NATOM,JNB,INBLO,CG, &
     MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
     LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC,ALP, &
     QPOLF &
     )
  !-----------------------------------------------------------------------
  !     THIS ROUTINE ALLOCATES MEMORY FOR COMPUTING POLARIZATION
  !     ENERGIES AND FORCES USING THE MATRIX INVERSION PROCEDURE. THE  
  !     ACTUAL CACLULATION IS DONE BY SUBROUTINE EPFINV2
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
  !     QPOLF  - flag to calculate force
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
  LOGICAL QPOLF
  !
  INTEGER I,J,NB,ITEMP,NPR,N3,NB6
  INTEGER NN3,NITPR,NITPR6,I1

  real(chm_real),allocatable,dimension(:) :: IMTXA
  real(chm_real),allocatable,dimension(:) :: IMTXB
  real(chm_real),allocatable,dimension(:) :: IDMUIND
  real(chm_real),allocatable,dimension(:) :: IEFIELD
  real(chm_real),allocatable,dimension(:) :: ITXWW
  real(chm_real),allocatable,dimension(:) :: IVECE
  real(chm_real),allocatable,dimension(:) :: IVV
  integer,allocatable,dimension(:) :: IINDXX
  real(chm_real),allocatable,dimension(:) :: ITTSR

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
  NN3 = N3*N3
  NITPR = NATOM*(NATOM - 1)
  NITPR6 = 6*NITPR 
  NB6 = 6*NB

  call chmalloc('epfinv.src','EPFINV','IMTXA',NN3,crl=IMTXA)
  call chmalloc('epfinv.src','EPFINV','IMTXB',NN3,crl=IMTXB)
  call chmalloc('epfinv.src','EPFINV','IDMUIND',N3,crl=IDMUIND)
  call chmalloc('epfinv.src','EPFINV','IEFIELD',N3,crl=IEFIELD)
  call chmalloc('epfinv.src','EPFINV','ITXWW',NB6,crl=ITXWW)
  call chmalloc('epfinv.src','EPFINV','IVECE',N3,crl=IVECE)
  call chmalloc('epfinv.src','EPFINV','IVV',N3,crl=IVV)
  call chmalloc('epfinv.src','EPFINV','IINDXX',N3,intg=IINDXX)

  call EPFINV2(IFRSTA,NATOM,JNB,INBLO,CG,MAXROW,IAC,ITC,LELEC,LCONS, &
       LSHFT,LVSHFT,LFSWT,LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
       EPOL,ALP,IDMUIND,IEFIELD,ITXWW,IMTXA,IMTXB,IVECE, &
       IVV,IINDXX,N3,QPOLF)
  !
  !     FREE MEMORY
  !
  call chmdealloc('epfinv.src','EPFINV','IEFIELD',N3,crl=IEFIELD)
  call chmdealloc('epfinv.src','EPFINV','ITXWW',NB6,crl=ITXWW)
  call chmdealloc('epfinv.src','EPFINV','IDMUIND',N3,crl=IDMUIND)

  IF (QMPOL) THEN
     call chmalloc('epfinv.src','EPFINV','ITTSR',NITPR6,crl=ITTSR)
     call MPOLAR(IFRSTA,NATOM,IAC,ITC,X,Y,Z,ALP,ITTSR,IMTXA, &
          IMTXB,IVECE,IVV,IINDXX,N3)
     call chmdealloc('epfinv.src','EPFINV','ITTSR',NITPR6,crl=ITTSR)
  ENDIF

  call chmdealloc('epfinv.src','EPFINV','IMTXA',NN3,crl=IMTXA)
  call chmdealloc('epfinv.src','EPFINV','IMTXB',NN3,crl=IMTXB)
  call chmdealloc('epfinv.src','EPFINV','IVECE',N3,crl=IVECE)
  call chmdealloc('epfinv.src','EPFINV','IVV',N3,crl=IVV)
  call chmdealloc('epfinv.src','EPFINV','IINDXX',N3,intg=IINDXX)

  RETURN
END SUBROUTINE EPFINV
!
!
SUBROUTINE EPFINV2(IFRSTA,NATOM,JNB,INBLO,CG, &
     MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
     LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
     EPOL,ALP,DMUIND,EFIELD,TXWW, &
     MTXA,MTXB,VECE,VV,INDXX,N3,QPOLF &
     !    &  ,IOFF
     )
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CALCULATES POLARIZATION ENERGIES AND FORCES
  !     VIA THE MATRIX INVERSE PROCESS
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
  !     MTXA - matrix used to store polarizability tensor
  !     MTXB - matrix used to store inverse of MTXA
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
  INTEGER N3,INDXX(*)
  LOGICAL QPOLF
  !
  INTEGER NITER, NCOUNT 
  INTEGER I,J,JPR,NB,NPR,ITEMP,I1
  real(chm_real) DMUIND(3,*),EFIELD(3,*),TXWW(6,*)
  real(chm_real) MTXA(N3,N3),MTXB(N3,N3),VECE(N3),VV(N3)
  real(chm_real) DMUNEW,DDIFF
  real(chm_real) E14M1,E14F,CGT2,CGT
  INTEGER I3,J3,IX,IY,IZ,JX,JY,JZ,NN3
  real(chm_real) ALPINV,MPOL,MPOLX,MPOLY,MPOLZ
  !
  NN3 = N3*N3
  !
  !     INITIATE MTXA, DIAGONAL ELEMENTS ARE 1/ALP
  !
  DO I = IFRSTA,NATOM
     I1=ITC(IAC(I))
     I3 = 3*I
     IX = I3 - 2
     IY = I3 - 1
     IZ = I3
     DO J = I,NATOM
        J3 = 3*J
        JX = J3 - 2
        JY = J3 - 1
        JZ = J3
        IF (I .EQ. J) THEN
           ALPINV = 1.0D0/ALP(I1)
           MTXA(IX,JX) = ALPINV
           MTXA(IY,JY) = ALPINV
           MTXA(IZ,JZ) = ALPINV
           MTXA(IX,JY) = 0.0D+00
           MTXA(IX,JZ) = 0.0D+00
           MTXA(IY,JZ) = 0.0D+00
        ELSE
           MTXA(IX,JX) = 0.0D+00
           MTXA(IY,JY) = 0.0D+00
           MTXA(IZ,JZ) = 0.0D+00
           MTXA(IX,JY) = 0.0D+00
           MTXA(IY,JX) = 0.0D+00
           MTXA(IX,JZ) = 0.0D+00
           MTXA(IZ,JX) = 0.0D+00
           MTXA(IY,JZ) = 0.0D+00
           MTXA(IZ,JY) = 0.0D+00
        ENDIF
     ENDDO
  ENDDO
  !
  !      GET THE ZEROTH ORDER FIELD AND POLAR TENSOR
  !
  CALL GZERO(EFIELD,3,NATOM)
  CALL ETINIT(IFRSTA,NATOM,JNB,INBLO,CG, &
       DX,DY,DZ,X,Y,Z,EFIELD,TXWW,PFCTOF, &
       ALP,DPFAC,IAC,ITC,NPDAMP,QPFEX)

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
        I3 = 3*I
        IX = I3 - 2
        IY = I3 - 1
        IZ = I3 
        DO JPR=1,NPR
           NB=NB+1
           IF (JNB(NB).LT.0) THEN
              CGT2=CGT*E14FAC
              E14F = E14M1
              J=-JNB(NB)
           ELSE
              CGT2=CGT
              E14F=ZERO
              J=JNB(NB)
           ENDIF
           J3 = 3*J
           JX = J3 - 2
           JY = J3 - 1
           JZ = J3 
           MTXA(IX,JX) = -1.0D0*TXWW(1,NB)
           MTXA(IX,JY) = -1.0D0*TXWW(2,NB)
           MTXA(IY,JX) = -1.0D0*TXWW(2,NB)
           MTXA(IY,JY) = -1.0D0*TXWW(3,NB)
           MTXA(IX,JZ) = -1.0D0*TXWW(4,NB)
           MTXA(IZ,JX) = -1.0D0*TXWW(4,NB)
           MTXA(IY,JZ) = -1.0D0*TXWW(5,NB)
           MTXA(IZ,JY) = -1.0D0*TXWW(5,NB)
           MTXA(IZ,JZ) = -1.0D0*TXWW(6,NB)
           !
           ! END JPR LOOP
           !
        ENDDO

     ENDIF
     !
     ! END MAIN LOOP
     !
  ENDDO
  !
  !     BUILD THE INTIRE MTXA FROM UPPER TRIANGLE MATRIX
  DO I = IFRSTA,N3
     DO J = IFRSTA,I - 1
        MTXA(I,J) = MTXA(J,I)
     ENDDO
  ENDDO

  !
  !     INVERT MTXA BY LU DECOMPOSITION 
  !
  CALL MATNVN(MTXA,MTXB,VV,INDXX,N3,N3) 
  !
  !     ZERO THE INDUCED DIPOLE
  !
  DO I = IFRSTA,NATOM
     DMUIND(1,I) = 0.0D+00
     DMUIND(2,I) = 0.0D+00
     DMUIND(3,I) = 0.0D+00
  ENDDO
  !    
  !     CALCULATE INDUCED DIPOLES
  !
  DO I=IFRSTA,NATOM
     I3 = 3*I
     IX = I3 - 2
     IY = I3 - 1
     IZ = I3
     DO J=IFRSTA,NATOM
        J3 = 3*J
        JX = J3 - 2
        JY = J3 - 1
        JZ = J3

        DMUIND(1,I) = DMUIND(1,I) + MTXB(IX,JX) &
             *EFIELD(1,J) &
             + MTXB(IX,JY)*EFIELD(2,J) &
             + MTXB(IX,JZ)*EFIELD(3,J)
        DMUIND(2,I) = DMUIND(2,I) + MTXB(IY,JX) &
             *EFIELD(1,J) &
             + MTXB(IY,JY)*EFIELD(2,J) &
             + MTXB(IY,JZ)*EFIELD(3,J)
        DMUIND(3,I) = DMUIND(3,I) + MTXB(IZ,JX) &
             *EFIELD(1,J) &
             + MTXB(IZ,JY)*EFIELD(2,J) &
             + MTXB(IZ,JZ)*EFIELD(3,J)
     ENDDO
  ENDDO
  !
  !     FINAL POLARIZATION ENERGY
  !
  EPOL = 0.0D+00
  DO I = IFRSTA,NATOM
     EPOL = EPOL+DMUIND(1,I)*EFIELD(1,I)+DMUIND(2,I) &
          *EFIELD(2,I)+DMUIND(3,I)*EFIELD(3,I)
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
  !
  ETERM(PIPF) = ETERM(PIPF)+EPOL
  !
  ! COMPUTE THE PIPF FORCE
  !
  IF(QPOLF) THEN
     CALL DPIPF(IFRSTA,NATOM,JNB,INBLO,CG, &
          MAXROW,IAC,ITC,LELEC,LCONS,LSHFT,LVSHFT,LFSWT, &
          LVFSWT,DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
          TXWW,DMUIND,PFCTOF, &
          ALP,DPFAC,NPDAMP,QPFEX &
          !    &  ,IOFF
          )
  ENDIF

  !
  ! PRINT OUT DIPOLE INFOR
  !
  IF (NAVDIP .GT. 0) THEN
     CALL PRDIP(NATOM,NAVDIP,X,Y,Z,CG,DMUIND)
  ENDIF

  RETURN
END SUBROUTINE EPFINV2

!

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
SUBROUTINE MPOLAR(IFRSTA,NATOM,IAC,ITC,X,Y,Z, &
     ALP,TTSR,MTXA,MTXB,VECE,VV,INDXX,N3 &
     )
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CALCULATES MOLECULAR POLARIZABILITY BY MATRIX INVERSE 
  !
  !     IFRSTA - first atom to look at in INBLO
  !     NATOM  - last atom to look at in INBLO (number of atoms)
  !     ALP    - polarizability (passed in param.f90)
  !     TTSR - rank 2 polar tensor
  !     MTXA - matrix used to store polarizability tensor
  !     MTXB - matrix used to store inverse of MTXA
  !----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use stream
  use pipfm
  implicit none
  !
  real(chm_real) EPOL
  INTEGER IFRSTA, NATOM
  INTEGER MAXROW,IAC(*),ITC(*)
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) ALP(*)
  INTEGER N3,INDXX(*)
  !
  INTEGER I,J,JPR,NB,I1
  real(chm_real) TTSR(6,*)
  real(chm_real) MTXA(N3,N3),MTXB(N3,N3),VECE(N3),VV(N3)
  INTEGER I3,J3,IX,IY,IZ,JX,JY,JZ,NN3
  real(chm_real) ALPINV,MPOL,MPOLX,MPOLY,MPOLZ
  !
  NB = 0
  NN3 = N3*N3
  !
  !     INITIATE MTXA, DIAGONAL ELEMENTS ARE 1/ALP
  !
  DO I = IFRSTA,NATOM
     I1=ITC(IAC(I))
     I3 = 3*I
     IX = I3 - 2
     IY = I3 - 1
     IZ = I3
     DO J = I,NATOM
        J3 = 3*J
        JX = J3 - 2
        JY = J3 - 1
        JZ = J3
        IF (I .EQ. J) THEN
           ALPINV = 1.0D0/ALP(I1)
           MTXA(IX,JX) = ALPINV
           MTXA(IY,JY) = ALPINV
           MTXA(IZ,JZ) = ALPINV
           MTXA(IX,JY) = 0.0D+00
           MTXA(IY,JX) = 0.0D+00
           MTXA(IX,JZ) = 0.0D+00
           MTXA(IZ,JX) = 0.0D+00
           MTXA(IY,JZ) = 0.0D+00
           MTXA(IZ,JY) = 0.0D+00
        ELSE
           MTXA(IX,JX) = 0.0D+00
           MTXA(IY,JY) = 0.0D+00
           MTXA(IZ,JZ) = 0.0D+00
           MTXA(IX,JY) = 0.0D+00
           MTXA(IY,JX) = 0.0D+00
           MTXA(IX,JZ) = 0.0D+00
           MTXA(IZ,JX) = 0.0D+00
           MTXA(IY,JZ) = 0.0D+00
           MTXA(IZ,JY) = 0.0D+00
        ENDIF
     ENDDO
  ENDDO
  CALL INTTSR(IFRSTA,NATOM,X,Y,Z,TTSR,PFCTOF, &
       ALP,DPFAC,IAC,ITC,NPDAMP,QPFEX)

  DO I = IFRSTA,NATOM-1
     I3 = 3*I
     IX = I3 - 2
     IY = I3 - 1
     IZ = I3
     DO J = I+1,NATOM
        NB = NB + 1
        J3 = 3*J
        JX = J3 - 2
        JY = J3 - 1
        JZ = J3
        MTXA(IX,JX) = -1.0D0*TTSR(1,NB)
        MTXA(IX,JY) = -1.0D0*TTSR(2,NB)
        MTXA(IY,JX) = -1.0D0*TTSR(2,NB)
        MTXA(IY,JY) = -1.0D0*TTSR(3,NB)
        MTXA(IX,JZ) = -1.0D0*TTSR(4,NB)
        MTXA(IZ,JX) = -1.0D0*TTSR(4,NB)
        MTXA(IY,JZ) = -1.0D0*TTSR(5,NB)
        MTXA(IZ,JY) = -1.0D0*TTSR(5,NB)
        MTXA(IZ,JZ) = -1.0D0*TTSR(6,NB)
     ENDDO
  ENDDO

  DO I = IFRSTA,N3
     DO J = IFRSTA,I - 1
        MTXA(I,J) = MTXA(J,I)
     ENDDO
  ENDDO

  !
  !     INVERT MTXA BY LU DECOMPOSITION
  !
  CALL MATNVN(MTXA,MTXB,VV,INDXX,N3,N3)
  !
  !
  !     CALCULATE MOLECULAR POLARIZABILITY
  !
  !
  CALL MPOLAR2(MTXB,N3,N3,MPOL,MPOLX,MPOLY,MPOLZ)
  WRITE(OUTU,200) MPOL,MPOLX,MPOLY,MPOLZ
200 FORMAT(' Molecular polarizability (MPOL) tot,x,y,z :' &
       ,F8.4,F8.4,F8.4,F8.4) 

  RETURN
END SUBROUTINE MPOLAR

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
SUBROUTINE MPOLAR2(MTXB,N,M,MPOL,MPOLX,MPOLY,MPOLZ)
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !
  !    THIS SUBROUTINE CALCULATES MOLECULAR POLARIZABILITY
  !    BY SUM OVER ALL THE ELEMENTS OF THE MOLECULAR
  !    POLARIZABILITY TENSOR
  !    MTXB  -  MOLECULAR POLARIZABILITY TENSOR 
  !    N     -  NUMBER OF ROWS IN MTXB
  !    M     -  NUMBER OF COLUMN IN MTXB
  !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  !
  use chm_kinds
  implicit none
  INTEGER N,M
  real(chm_real)  MPOL,MPOLX,MPOLY,MPOLZ,MTXB(N,M)
  !
  INTEGER NATOM,I,J,I3,J3,IX,IY,IZ,JX,JY,JZ
  real(chm_real)  TMOL(3,3),EVAL(3),EVEC(3,3)
  !

  NATOM = INT(N/3) 

  DO I = 1,3
     DO J = 1,3
        TMOL(I,J) = 0.0D+00
     ENDDO
  ENDDO

  DO I = 1,NATOM
     I3 = 3*I
     IX = I3 - 2
     IY = I3 - 1
     IZ = I3        
     DO J = 1,NATOM
        J3 = 3*J
        JX = J3 - 2
        JY = J3 - 1
        JZ = J3
        TMOL(1,1) = TMOL(1,1) + MTXB(IX,JX)
        TMOL(1,2) = TMOL(1,2) + MTXB(IX,JY)
        TMOL(2,1) = TMOL(2,1) + MTXB(IY,JX)
        TMOL(1,3) = TMOL(1,3) + MTXB(IX,JZ)
        TMOL(3,1) = TMOL(3,1) + MTXB(IZ,JX)
        TMOL(2,2) = TMOL(2,2) + MTXB(IY,JY)
        TMOL(2,3) = TMOL(2,3) + MTXB(IY,JZ)
        TMOL(3,2) = TMOL(3,2) + MTXB(IZ,JY)
        TMOL(3,3) = TMOL(3,3) + MTXB(IZ,JZ)
     ENDDO
  ENDDO

  CALL TREDTWO(TMOL,3,3,EVAL,EVEC)
  CALL TQLI(EVAL,EVEC,3,3,TMOL)

  ! get eigenvalues
  MPOLX = EVAL(1)
  MPOLY = EVAL(2)
  MPOLZ = EVAL(3)

  MPOL = (EVAL(1) + EVAL(2) + EVAL(3))/3.0D0

  RETURN

END SUBROUTINE MPOLAR2
!

SUBROUTINE INTTSR(IFRSTA,NATOM,X,Y,Z,TTSR,PFCTOF, &
     ALP,DPFAC,IAC,ITC,NPDAMP,QPFEX)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CALCULATES INITIAL DIPOLE INTERACTION TENSOR FOR THE
  !     CALCULATION OF MOLECULAR POLARIZABILITY 
  !
  !     IFRSTA - first atom to look at in INBLO
  !     ALP    - polarizability
  !     TTSR   - dipole tensor
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
  INTEGER IFRSTA,NATOM
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) TTSR(6,*)
  !
  INTEGER I,J,JPR,NB,NPR,ITEMP
  real(chm_real) XX,YY,ZZ,R1,R2,R3,QR3I,QR3J, &
       RR3,RR3XX,RR3YY,RR3ZZ
  !
  real(chm_real) ALP(*),DPFAC,U3,AU3,FLMD3,FLMD5
  INTEGER IAC(*),ITC(*),NPDAMP,I1,J1
  !
  real(chm_real) RIJ2,PFCTOF,PFCTOF2
  LOGICAL QPFEX
  PFCTOF2=PFCTOF*PFCTOF
  NB = 0
  ITEMP = 0

  ! DO BLOCK EXPANSION OF CODE TO IMPROVE EFFICIENCY
  IF(PFCTOF.LE.0.0D0 .AND. NPDAMP.EQ.0) THEN
#undef PIPF_CTOF
#undef PIPF_DAMP
#include "epfinv.inc"
  ELSE IF(PFCTOF.GT.0.0D0 .AND. NPDAMP.EQ.0) THEN
#define PIPF_CTOF 1
#include "epfinv.inc"
#undef PIPF_CTOF
  ELSE IF(PFCTOF.LE.0.0D0 .AND. NPDAMP.EQ.1) THEN
#define PIPF_DAMP 1
#include "epfinv.inc"
  ELSE
#define PIPF_CTOF 1
#include "epfinv.inc"
#undef PIPF_CTOF
#undef PIPF_DAMP
  ENDIF
  RETURN
END SUBROUTINE INTTSR
#endif /* (pipf_main)*/

subroutine epfinv_dummy()
  return
end subroutine epfinv_dummy
