#if KEY_QUANTUM==1
SUBROUTINE ITER (H,W,WJ,WK,X,Y,Z,XI,YI,ZI, &
     ENUCLR,EE,FULSCF)
  !
  !     The SCF calculation is carried out here. ITER deals
  !     with the storage allocation for ITER2 which actually
  !     performs the calculation.
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  use sizes
  !
  use quantm
  use scfblk
  use qmlinkm
  implicit none
  !
  LOGICAL FULSCF
  real(chm_real)  EE, ENUCLR, H(*), W(*), WJ(*), WK(*)
  real(chm_real)  X(*), Y(*), Z(*), XI(*), YI(*), ZI(*)
  !
  INTEGER LINEAR, LPULAY, LPULY2
  !
  INTEGER NORBS2
  !
  !     Define some constants. LPULAY is set to 6 times the size of the
  !     density matrix size so that several old matrices can be stored
  !     and then used by the PULAY converger.
  !
  LINEAR = (NORBS * (NORBS + 1)) / 2
  LPULAY = 6 * LINEAR
  LPULY2 = MAX(36,NORBS)
  NORBS2 = NORBS*NORBS


  !     Check memory allocation. 
  Call Check_working_memory_iter(NORBS,LINEAR,LPULAY,LPULY2,NORBS2)
  !
  !     Copy the old density matrices to POLD1 and PBOLD1.
  !
  pold1_keep(1:linear) = PAOLD (1:LINEAR)
  pbold1_keep(1:linear) = PBOLD(1:LINEAR)
  !
  CALL ITER2(H,W,WJ,WK,ENUCLR,EE,FULSCF,X,Y,Z,XI,YI,ZI,PDENS,PDENSA,PDENSB, &
       POLD1_keep,PBOLD1_keep,CALPHA, EIGSA,CBETA,EIGSB,FA,FB, &
       AR1_keep,AR2_keep,AR3_keep,AR4_keep,BR1_keep,BR2_keep,BR3_keep,BR4_keep, &
       POLD2_keep,POLD3_keep,PBOLD2_keep,PBOLD3_keep,WORK_keep, &
       FHB_keep,CHB_keep,PAHB_keep,LPULAY, &
       FBHB_keep,CBHB_keep,PBHB_keep,PAPRE_keep,PBPRE_keep)     ! For UHF-GHO and density damping
  !
  !     Save the old density matrices.
  !
  paold(1:linear) = POLD1_keep (1:LINEAR)
  pbold(1:linear) = PBOLD1_keep(1:LINEAR)
  IF(QMLINK) THEN
     calpha(1:norbs2) = CHB_keep(1:NORBS2)
     pho(1:linear)    = PAHB_keep(1:LINEAR)
     !
     ! For UHF-GHO ... PJ 12/2002
     !
     cbeta(1:norbs2) = CBHB_keep(1:NORBS2)
     pbho(1:linear) = PBHB_keep(1:LINEAR)
  ENDIF
  !
  RETURN
END SUBROUTINE ITER
!
SUBROUTINE ITER2(H,W,WJ,WK,ENUCLR,EE,FULSCF,X,Y,Z,XI,YI,ZI,P,PA,PB, &
     POLD,PBOLDH,C,EIGS,CBETAH,EIGB,F,FBH, &
     AR1,AR2,AR3,AR4,BR1,BR2,BR3,BR4, &
     POLD2,POLD3,PBOLD2,PBOLD3,WORK, &
     FHB,CHB,PAHB,LPULAY, &
     FBHB,CBHB,PBHB,PAPRE,PBPRE)     ! For UHF-GHO and density damping
  !
  !     Performs the SCF calculation.
  !
  use chm_kinds
  use dimens_fcm
  use number
  !
  use quantm
  use scfblk
  use stream
  use sizes
  use qmlinkm
  use psf
  !
  ! for parallel run
#if KEY_PARALLEL==1
  use parallel       
#endif
#if KEY_PARALLEL==1 && KEY_MPI==1
  use mpi            
#endif
  implicit none
  !
  EXTERNAL MNHELECT, MECI
  real(chm_real)   MNHELECT, MECI
#if KEY_UNUSED==1 /*capcor_unused*/
  EXTERNAL CAPCOR
  real(chm_real)   CAPCOR
#endif /* (capcor_unused)*/
  !
  INTEGER LPULAY
  LOGICAL FULSCF
  real(chm_real):: AR1(*), AR2(*), AR3(*), AR4(*), BR1(*), BR2(*), &
       BR3(*), BR4(*), C(*), CBETAH(*), EE, EIGB(*), EIGS(*), &
       ENUCLR, F(*), FBH(*), H(*), P(*), PA(*),PB(*),PBOLDH(*), &
       PBOLD2(*), PBOLD3(*), POLD(*), POLD2(*), POLD3(*), W(*), &
       WJ(*), WK(*), WORK(*), FHB(*), CHB(*), PAHB(*), &
       X(*), Y(*), Z(*), XI(*), YI(*), ZI(*)
  !
  real(chm_real):: FBHB(*), CBHB(*), PBHB(*), PAPRE(*), PBPRE(*)  ! For UHF-GHO and density damping ...
  !
  INTEGER  I, IALP, IBET, IPT, IREDY, J, JALP, JBET, LINEAR, &
       MODEA, MODEB, NA1EL, NA2EL, NB1EL, NB2EL, NITER
  LOGICAL  BFRST, DEBUG, FRST, HALFE, INCITR, MAKEA, MAKEB, &
       PRT1EL, PRTDEN, PRTEIG, PRTENG, PRTFOK, PRTPL, PRTVEC, &
       READY
  integer :: ilocal
  real(chm_real):: DIFF, ECI, EOLD, ESCF, RANDOM,SELCON,SHIFT,TRANS
  !
  real(chm_real),parameter :: BFACT = -1000.2D0, SFACT = 23.061D0, &
       TEN2 = 100.D0,BIG2 = 9999.D0, BIG1 = 99999.D0, &
       rthree = one/three

  !   Variables for QM-boundary atoms
  INTEGER II, JJ, I1, J1, IJ, K, L, KK1, KK2, KK3
  INTEGER NORBHB, IORB1B, NAOS, LINAO, IQATM
  real(chm_real)  XBT1,XBT2,XBT3
#if KEY_PARALLEL==1
#if KEY_MPI==1
  integer ireq,ierr,istatus
#endif 
#endif 

  !
  ! go parallel
  INTEGER ATFRST,ATLAST,NODELE,NORBS2
  INTEGER ATFRS2,ATLAS2,NODEL2
  ! debug 
  INTEGER NORBSX,LINEARX
  !
  !     Do some initialisation.
  !
  IREDY  = 1
  LINEAR = (NORBS * (NORBS + 1)) / 2
  NORBS2 = NORBS * NORBS
  NA1EL  = NALPHA + NOPEN
  NA2EL  = NCLOSE
  NB1EL  = NBETA + NOPEN
  NB2EL  = 0
  ! debug
  LINEARX= LINEAR
  NORBSX = NORBS
  !
  !   LINK ATOM HAS TO BE NON-HYDROGEN
  IF(QMLINK) THEN
     NORBHB = NORBS - 3*MQMLNK
     NAOS = NORBHB-MQMLNK
     LINAO = (NAOS*(NAOS+1))/2
  ENDIF
  !
  HALFE  = (NOPEN .NE. NCLOSE)
  MAKEA  = .TRUE.
  MAKEB  = .TRUE.
  READY  = .FALSE.
  !
  EOLD   = TEN2
  TRANS  = TENM1
  SHIFT  = ZERO
  JBET   = 0
  JALP   = 0
  !
  !     Zero out values
  !
  DEBUG  = .FALSE.
  PRTEIG = .FALSE.
  PRTENG = .FALSE.
  PRTPL  = .FALSE.
  PRT1EL = .FALSE.
  PRTDEN = .FALSE.
  PRTFOK = .FALSE.
  PRTVEC = .FALSE.
  !
  !     Determine some printing options.
  !
  DEBUG  = INDEX(KEYWRD,'ITER') .NE. 0
  PRTEIG = INDEX(KEYWRD,'EIG2') .NE. 0
  PRTENG = INDEX(KEYWRD,'ENERGY') .NE. 0
  PRTPL  = INDEX(KEYWRD,' PL ') .NE. 0
  IF (INDEX(KEYWRD,'DEBUG') .NE. 0) THEN
     PRT1EL = INDEX(KEYWRD,'1ELEC') .NE. 0
     PRTDEN = INDEX(KEYWRD,'DENSITY') .NE. 0
     PRTFOK = INDEX(KEYWRD,'FOCK') .NE. 0
     PRTVEC = INDEX(KEYWRD,'VECT') .NE. 0
  ENDIF
  !
  IF (NEWDG) NEWDG = (ABS(BSHIFT) .LT. TENM3)
  SELCON = SCFCRT * SFACT
  IF (PLTEST .LT. SCFCRT) PLTEST = SCFCRT
  IF (NALPHA .NE. NBETA .OR. .NOT. UHF) PLTEST = TENM3
  IF (DEBUG.AND.PRNLEV.GE.2) WRITE (OUTU,'(A,F16.7)') ' ITER> SELCON  = ',SELCON
  IF (PRT1EL) THEN
     IF(PRNLEV.GE.2) WRITE(OUTU,'(A)') ' ITER> One-electron matrix on entry :' 
     CALL MNVECPRT(H,NORBS)
  ENDIF
  ! debug
  ! renormalize if any density matrix elements have been truncated during
  ! scf. -> Temporay, ignore the renormalization
  IF(UHF) THEN
     ! alpha /beta density
     L=0
     DO I=1,NORBSX
        DO J=1,I-1
           L=L+1
           PA(L)=ZERO
           PB(I)=ZERO
        END DO
        L=L+1
     END DO
     DO I=1,LINEARX
        P(I)     = PA(I) + PB(I)
        POLD(I)  = ZERO
        PBOLDH(I)= PB(I)
     END DO
  ELSE
     L=0
     DO I=1,NORBSX
        DO J=1,I-1
           L=L+1
           P(I)=ZERO
        END DO
        L=L+1
     END DO
     DO I=1,LINEARX
        PA(I)=HALF*P(I)
        PB(I)=PA(I)
        POLD(I)=P(I)
     END DO
  END IF
  ! end
  !
100 NITER  = 0
  BFRST  = .TRUE.
  FRST   = .TRUE.
  INCITR = .TRUE.
  IF (CAMKIN) THEN
     MODEA = 1
     MODEB = 1
  ELSE
     MODEA = 0
     MODEB = 0
  ENDIF
  !=========================================================================
  !     Start the SCF loop here.
  !=========================================================================
200 INCITR = (MODEA .NE. 3) .AND. (MODEB .NE. 3)
  IF (INCITR) NITER = NITER + 1
  !=========================================================================
  !     Invoke extra convergers if SCF calculation is having difficulty.
  !=========================================================================
  IF (NITER .GT. (ITRMAX - 10) .AND. .NOT. ALLCON) THEN
     IF(PRNLEV.GE.2) WRITE (OUTU,'(A)') ' ITER> All convergers forced on and iteration counter reset.'
     IREDY  = -4
     ALLCON = .TRUE.
     CAMKIN = (.NOT. HALFE)
     NEWDG  = .FALSE.
     OKPULY = .TRUE.
     BSHIFT = BFACT
     EOLD   = TEN2
     GOTO 100
  ENDIF
  !=========================================================================
  !     Make the alpha Fock matrix.
  !=========================================================================
  ! go parallel
#if KEY_PARALLEL==1 /*paramain*/
  NODELE = LINEAR/ NUMNOD
  NODEL2 = NORBS / NUMNOD
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
  ATFRST=MYNOD*NODELE + 1
  ATLAST=(MYNOD+1)*NODELE
  !
  ATFRS2=MYNOD*NODEL2 + 1
  ATLAS2=(MYNOD+1)*NODEL2
  !
  IF(MYNOD.EQ.(NUMNOD-1)) THEN 
     ATLAST=LINEAR
     ATLAS2=NORBS
  END IF
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
  ATFRST=1
  ATLAST=LINEAR
  !
  ATFRS2=1
  ATLAS2=NORBS
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
  !  DTM PI
  IF (QMPI) THEN
     ATFRST=1
     ATLAST=LINEAR
     !
     ATFRS2=1
     ATLAS2=NORBS
  ENDIF
#else /* (paramain)*/
  ATFRST=1
  ATLAST=LINEAR
  !
  ATFRS2=1
  ATLAS2=NORBS
#endif /* (paramain)*/
  ! 
  ! initialie F matrix
  F(1:NORBS2) = ZERO
  IF(UHF) FBH(1:NORBS2) = ZERO
  !
  IF (BSHIFT .NE. ZERO) THEN
     IPT = 0
     SHIFT = BSHIFT * (NITER + ONE) ** (-ONEPT5)
#if KEY_PARALLEL==1
     !  DTM PI
     IF (.NOT. QMPI) THEN
        DO I = 1, (ATFRS2-1)
           IPT=IPT+I
        END DO
     ENDIF
#endif /* */
     DO I = ATFRS2, ATLAS2   ! I=1,NORBS
        DO J=1,I
           IPT = IPT + 1
           F(IPT) = H(IPT) + SHIFT * PA(IPT)
        END DO
        F(IPT) = F(IPT) - SHIFT
     END DO
  ELSE IF (NITER .LT. 2 .AND. FULSCF) THEN
     RANDOM = TENM3
     DO I=ATFRST, ATLAST     ! I=1,LINEAR
        RANDOM = -RANDOM
        F(I) = H(I)           ! namkh: old-style: F(I) = H(I) + RANDOM 
     END DO
  ELSE
     F(ATFRST:ATLAST) = H(ATFRST:ATLAST)    ! I=1,LINEAR
  ENDIF

300 CONTINUE
  !
  CALL MNFOCK2(F,P,PA,W,WJ,WK,NATQM,N1ST,NMIDLE,NLAST)
  CALL MNFOCK1(F,P,PA,PB,X,Y,Z,XI,YI,ZI)
  !
  ! go parallel
#if KEY_PARALLEL==1
  !  DTM PI
  IF(NUMNOD.NE.1 .AND. .NOT. QMPI) THEN
     call psync()
     call GCOMB(F,LINEAR)
  END IF
#endif 
  !
  !     TRANSFORM F INTO HB FOR QM LINK ATOM....JG 5/17/97
  IF(QMLINK) CALL MNFTOFHB4(F,FHB,MBT,NATQM,MQMLNK,NORBS,N1ST,NMIDLE,NLAST)

  IF (PRTFOK) THEN
     IF(PRNLEV.GE.2) WRITE (OUTU,'(A,I4)') ' ITER2> Alpha Fock matrix on iteration ',NITER
     CALL MNVECPRT(F,NORBS)
     IF(QMLINK) THEN
        IF(PRNLEV.GE.2) WRITE(OUTU,'(A,I4)') ' ITER2> Fhb: NORBHB= ',NORBHB
        CALL MNVECPRT(FHB,NORBHB)
        IF(PRNLEV.GE.2) WRITE(OUTU,'(A)') ' ITER2> Density Matrices, p,pa:'
        CALL MNVECPRT(P,NORBS)
        CALL MNVECPRT(PA,NORBS)
     ENDIF
  ENDIF
  !=========================================================================
  !     Make the beta Fock matrix.
  !=========================================================================
  IF (UHF) THEN
     !
     ! commented, UHF-GHO has been implemented ... PJ 12/2002
     !       IF(QMLINK) CALL WRNDIE(-5,'ITER2>','OPEN SHELL NOT IMPLEMENTED')
     !
     IF (SHIFT .NE. ZERO) THEN
        IPT = 0
#if KEY_PARALLEL==1
        !  DTM PI
        IF (.NOT. QMPI) THEN
           DO I = 1, (ATFRS2-1)
              IPT=IPT+I
           END DO
        ENDIF
#endif 
        DO I = ATFRS2, ATLAS2   ! I=1,NORBS
           DO J=1,I
              IPT = IPT + 1
              FBH(IPT) = H(IPT) + SHIFT * PB(IPT)
           END DO
           FBH(IPT) = FBH(IPT) - SHIFT
        END DO
     ELSE IF (NITER .LT. 2 .AND. FULSCF) THEN
        RANDOM = TENM3
        DO I=ATFRST, ATLAST     ! I=1,LINEAR
           RANDOM = -RANDOM
           FBH(I) = H(I)             ! namkh: old-style: FBH(I) = H(I) + RANDOM
        END DO
     ELSE
        FBH(ATFRST:ATLAST) = H(ATFRST:ATLAST)    ! I=1,LINEAR
     ENDIF
     !
     CALL MNFOCK2(FBH,P,PB,W,WJ,WK,NATQM,N1ST,NMIDLE,NLAST)
     CALL MNFOCK1(FBH,P,PB,PA,X,Y,Z,XI,YI,ZI)
     !
     ! go parallel
#if KEY_PARALLEL==1
     IF(NUMNOD.NE.1 .AND. .NOT. QMPI) THEN
        call psync()
        call GCOMB(FBH,LINEAR)
     END IF
#endif 
     !
     !
     ! for GHO, transform beta Fock matrix to hybrid basis ... PJ 12/2002
     !
     IF (QMLINK) CALL MNFTOFHB4(FBH,FBHB,MBT,NATQM,MQMLNK,NORBS,N1ST,NMIDLE,NLAST)
     !
     IF (PRTFOK) THEN
        IF(PRNLEV.GE.2) WRITE (OUTU,'(A,I4)') ' ITER2> Beta Fock matrix on iteration ',NITER
        CALL MNVECPRT(FBH,NORBS)
        !
        ! UHF-GHO debug ... PJ 12/2002
        !
        IF(QMLINK) THEN
           IF(PRNLEV.GE.2) WRITE(OUTU,'(A,I4)') ' ITER2> Fbhb: NORBHB= ',NORBHB
           CALL MNVECPRT(FBHB,NORBHB)
           IF(PRNLEV.GE.2) WRITE(OUTU,'(A)') ' ITER2> Density Matrices, p,pb:'
           CALL MNVECPRT(P,NORBS)
           CALL MNVECPRT(PB,NORBS)
        ENDIF
     ENDIF
  ENDIF          ! (UHF)
  !=========================================================================
  !     Calculate the energy in Kcal/mole.
  !=========================================================================
  IF (.NOT.FULSCF) GOTO 400
  EE = MNHELECT(NORBS,PA,H,F,.TRUE.)
  IF (UHF) THEN
     EE = EE + MNHELECT(NORBS,PB,H,FBH,.TRUE.)
  ELSE
     EE = TWO * EE
  ENDIF
  !
  ESCF = EE + ENUCLR
#if KEY_UNUSED==1 /*capcor_unused*/
  !     EE = EE + CAPCOR(NAT,N1ST,NLAST,NATQM,P,H)
#endif /* (capcor_unused)*/
  !
  ! go parallel
#if KEY_PARALLEL==1
  IF(NUMNOD.NE.1 .AND. .NOT. QMPI) THEN
     call psync()
     call GCOMB(ESCF,1)
  END IF
#endif 
  !
  ESCF = (ESCF + SHIFT*(NOPEN-NCLOSE)*PT25)*SFACT + ATHEAT
  !=========================================================================
  !     Self-consistency tests.
  !=========================================================================
  IF (INCITR) THEN
     DIFF = ESCF - EOLD
     IF (PL .LT. PLTEST .AND. ABS(DIFF) .LT. SELCON .AND. READY) THEN
        IF (ABS(SHIFT) .LT. TENM10) GOTO 400
        SHIFT = ZERO
        F(ATFRST:ATLAST) = H(ATFRST:ATLAST)     ! I=1,LINEAR
        MAKEA = .TRUE.
        MAKEB = .TRUE.
        GOTO 300
     END IF
     READY = (IREDY .GT. 0 .AND. ABS(DIFF) .LT. TEN * SELCON)
     IREDY = IREDY + 1
  ENDIF
  IF(INCITR) EOLD = ESCF
  !
  IF (PRTPL) THEN
     IF (ABS(ESCF) .GT. BIG1) ESCF = BIG1
     IF (ABS(DIFF) .GT. BIG2) DIFF = ZERO
     IF (INCITR.AND.PRNLEV.GE.2) THEN
        WRITE (OUTU,'(A,I4,A,2E10.3,/,8X,A,F14.7,A,F13.7)') &
             ' ITER> Iteration number = ',NITER,' PLS = ',PL,PLB, &
             ' Energy = ',ESCF,' DeltaE = ',DIFF
     END IF
  ENDIF
  !=========================================================================
  !     Invoke the Camp-King converger for the alpha Fock matrix.
  !=========================================================================
  IF (NITER .GT. 2 .AND. CAMKIN .AND. MAKEA) THEN
     CALL INTERP(NORBS,NA1EL,(NORBS-NA1EL),MODEA,(ESCF/SFACT),F,C,AR1,AR2,AR3,AR4,AR1)
  END IF
  MAKEB = .FALSE.
  If(MODEA .ne. 3) then
     MAKEB = .TRUE.
     !=========================================================================
     !     Invoke Pulay's converger for the alpha density matrix.
     !=========================================================================
     IF (NEWDG) THEN
        if (OKPULY .AND. MAKEA .AND. IREDY .GT. 1) then
           CALL PULAY(F,PA,NORBS,POLD,POLD2,POLD3,JALP,IALP,LPULAY,FRST,PL)
        end if
        !=========================================================================
        !     Diagonalise the alpha or RHF secular determinant.
        !=========================================================================
        IF (HALFE .OR. CAMKIN) THEN
           IF(QMLINK) THEN
              CALL RSPQM(FHB,NORBHB,NORBHB,EIGS,CHB)
           ELSE
              CALL RSPQM(F,NORBS,NORBS,EIGS,C)
           ENDIF
        ELSE
           IF(QMLINK) THEN
              CALL RSPQM(FHB,NORBHB,NORBHB,EIGS,CHB)
           ELSE
              ! namkh 04/05/05
              ! turn off fast diagonalization routine to avoid conflict with PULAY
              ! use slow diagonalization routine
              IF ((NEWDG.AND.OKPULY).OR.(INCITR.AND.READY)) THEN
                 CALL RSPQM(F,NORBS,NORBS,EIGS,C)
              ELSE
                 CALL SQUARE(F,NORBS)
                 CALL DIAG (F,C,NA1EL,EIGS,NORBS,NORBS,WORK)
              END IF
           ENDIF
        ENDIF
     ELSE
        IF(QMLINK) THEN
           CALL RSPQM(FHB,NORBHB,NORBHB,EIGS,CHB)
        ELSE
           CALL RSPQM(F,NORBS,NORBS,EIGS,C)
        ENDIF
     ENDIF
     !
     !      EXPAND CHB TO C...JG 5/17/97
     !      C IS NEEDED BY MULLIKEN....JG 1/19/98
     !      MOVED TO MULLIKEN....      JG 3/11/98
     IF(QMLINK) THEN
        CALL CALDENS(CHB,NORBHB,NORBHB,NA2EL,NA1EL,FRACT,PAHB)
        !
        ! alpha density damping for UHF GHO ... PJ 12/2002
        !    !  Pi = a*Pi-1 + (1-a)Pi, default a = 0.0 (no damping)
        !    !  only use damping when denisty change is 100 times greater
        !       than the convergence criterion. This is added to ensure
        !       last several iterations are carried out without any damping.
        !
        IF (QMLINK .AND. UHF)  THEN
           ilocal = NORBHB*(NORBHB+1)/2
           IF (NITER .GE. 2 .AND. PL .GT. TEN2*PLTEST) THEN
              PAHB(1:ilocal) = PALPHA*PAPRE(1:ilocal) + (1-PALPHA)*PAHB(1:ilocal)
           END IF
           PAPRE(1:ilocal) = PAHB(1:ilocal)
        END IF
        !
        !   Relocate the positions of active hybrid orbitals if
        !   there are more than one QM-boundary atom.
        DO I = NORBHB,NAOS+2,-1
           IQATM  = I-NAOS
           II     = I*(I-1)/2
           IORB1B = NAOS+4*(IQATM-1)
           JJ     = IORB1B*(IORB1B+1)/2
           DO J = 1,NAOS
              II = II+1
              JJ = JJ+1
              PAHB(JJ) = PAHB(II)
              PAHB(II) = zero
           END DO
           !
           !   HB-HB blocks
           DO J = 1,IQATM
              II = II+1
              JJ = JJ+1
              PAHB(JJ) = PAHB(II)
              PAHB(II) = zero
              IF(J.NE.IQATM) THEN
                 PAHB(JJ+1:JJ+3) = zero
                 JJ = JJ+3
              ENDIF
           END DO
           !   The rest three auxiliary orbitals
           DO J = 2,4
              PAHB(JJ+1:jj+IORB1B+J) = zero
              JJ = JJ+IORB1B+J
              PAHB(JJ) = one-QMATMQM(IQATM)*rthree     ! /three
              !
              ! UHF-GHO auxiliary alpha density ... PJ 12/2002
              !
              IF (UHF) PAHB(JJ) = (one-QMATMQM(IQATM)*rthree)*half   ! /three and /2.0D0
           END DO
        ENDDO
        !
        !   Auxiliary density for the first QM-boundary atom
        K = NAOS+2              ! N1ST(IMQLINK(1))+1
        L = NAOS+4                 ! NLAST(IMQLINK(1))
        DO I = K,L
           JJ = I*(I-1)/2
           PAHB(JJ+1:jj+I)=zero
           PAHB(JJ+I) = one-QMATMQM(1)*rthree  ! /3.0D0
           !
           ! UHF-GHO auxiliary alpha density ... PJ 12/2002
           !
           IF (UHF) PAHB(JJ+I) = (one-QMATMQM(1)*rthree)*half  ! /3.0D0)/2.0D0
        ENDDO
        !
        !   AO blocks...Not affected by orbital transformation
        P(1:LINAO) = PAHB(1:LINAO)
        IJ = LINAO
        !   Loop over QM-boundary atoms
        DO K = 1,MQMLNK
           J1 = 16*(K-1)
           I1 = NAOS+4*(K-1)
           IORB1B = I1*(I1+1)/2
           !
           !      FOR EACH BOUNDARY ATOM, THERE ARE FOUR AOs.
           !      BOUNDARY ATOMS MUST BE NON-HYDROGEN ATOMS...
           L = 0
           DO I = 1,4
              !
              !      SINCE ONLY ONE HYBRID-ORBITAL DENSITY ON THE BOUNDARY ATOM IS NON-ZERO
              !      SUM OVER ORBITALS IS NOT NEEDED
              XBT1 = MBT(J1+I)                   ! MBTM(J1+4*(I-1)+1)
              P(IJ+1:IJ+NAOS) = PAHB(IORB1B+1:IORB1B+NAOS)*XBT1
              IJ   = IJ+NAOS 
              !
              !   BOUNDARY ATOM-OTHER BOUNDARY ATOM BLOCKS
              DO L = 1,K-1
                 KK1 = 16*(L-1)
                 KK3 = IORB1B+NAOS+4*(L-1)+1
                 XBT2= PAHB(KK3)*XBT1 
                 P(IJ+1:IJ+4) = XBT2*MBT(KK1+1:KK1+4)   ! MBTM(KK1+4*(J-1)+1), J=1,4
                 IJ  = IJ+4
              ENDDO
              !
              !   BOUNDARY ATOM SELF BLOCK
              KK1 = 4*(I-1)+J1
              DO J = 1,I
                 IJ = IJ+1
                 KK2 = 4*(J-1)+J1
                 P(IJ) = zero
                 DO L = 1,4
                    KK3 = (I1+L)*(I1+L+1)/2
                    P(IJ) = P(IJ)+PAHB(KK3)*MBTM(KK1+L)*MBTM(KK2+L)
                 END DO
              ENDDO
           ENDDO
        ENDDO
     ENDIF
     !
     ! GHO alpha density matrix for UHF, already collected in P ... PJ 12/2002
     !
     IF (UHF) THEN
        ilocal=NORBS*(NORBS+1)/2
        PA(1:ilocal) = P(1:ilocal)
     END IF

     !         do i=1,NORBS*(NORBS+1)/2
     !            write(6,'(I4,F15.7,F15.7)') i,p(i),pahb(i)
     !         end do

     !-------------------------------------------------------------------------
     IF (PRTVEC) THEN
        IF(PRNLEV.GE.2) WRITE (OUTU,'(A,I4)') ' ITER> Alpha eigenvectors and eigenvalues on iteration ', NITER
        CALL MATOUT(C,EIGS,NORBS,NORBS,NORBS)
     ELSE IF (PRTEIG.AND.PRNLEV.GE.2) THEN
        WRITE(OUTU,'(A,I4)') ' ITER> Alpha eigenvalues on iteration ',NITER
        WRITE (OUTU,'(8X,6G13.6)') (EIGS(I) , I = 1,NORBS)
     ENDIF
  End if           ! MODEA .ne. 3
  IF (IFILL .NE. 0) CALL SWAP(C,NORBS,NORBS,NA2EL,IFILL)
  !=========================================================================
  !     Calculate the alpha density matrix.
  !=========================================================================
  IF(.NOT. QMLINK) THEN
     IF (UHF) THEN
        CALL CALDENS( C,NORBS, NORBS, NA2EL,NA1EL, FRACT, PA)
     ELSE
        CALL CALDENS( C,NORBS, NORBS, NA2EL,NA1EL, FRACT, P)
     ENDIF
  ENDIF
  IF (MODEA .NE. 3 .AND. .NOT. (NEWDG .AND. OKPULY)) THEN
     CALL CNVG(P, POLD, POLD2, NORBS, NITER, PL,.FALSE.)
  ENDIF
  !=========================================================================
  !     Invoke the Camp-King converger for the beta Fock matrix.
  !=========================================================================
  IF (UHF) THEN
     If (NITER .GT. 2 .AND. CAMKIN .AND. MAKEB) then
        CALL INTERP(NORBS,NB1EL,(NORBS-NB1EL),MODEB,(ESCF/SFACT),FBH,CBETAH,BR1,BR2,BR3,BR4,BR1)
     End if
     MAKEA = .FALSE.
     If(MODEB .ne. 3) then
        MAKEA = .TRUE.
        !=========================================================================
        !     Invoke Pulay's converger for the beta density matrix.
        !=========================================================================
        IF (NEWDG) THEN
           IF (OKPULY .AND. MAKEB .AND. IREDY .GT. 1)  &
                CALL PULAY(FBH,PB,NORBS,PBOLDH,PBOLD2,PBOLD3,JBET,IBET,LPULAY,BFRST,PLB)
           !=======================================================================
           !     Diagonalise the beta secular determinant.
           !=======================================================================
           IF (HALFE .OR. CAMKIN) THEN
              !
              ! For UHF-GHO ... PJ 12/2002
              IF (QMLINK) THEN
                 CALL RSPQM(FBHB,NORBHB,NORBHB,EIGB,CBHB)
              ELSE
                 CALL RSPQM(FBH,NORBS,NORBS,EIGB,CBETAH)
              ENDIF
           ELSE
              !
              ! For UHF-GHO ... PJ 12/2002
              IF (QMLINK) THEN
                 CALL RSPQM(FBHB,NORBHB,NORBHB,EIGB,CBHB)
              ELSE
                 ! namkh 04/05/05
                 ! turn off fast diagonalization routine to avoid conflict with PULAY
                 ! use slow diagonalization routine 
                 IF ((NEWDG.AND.OKPULY).OR.(INCITR.AND.READY)) THEN
                    CALL RSPQM(FBH,NORBS,NORBS,EIGB,CBETAH)
                 ELSE
                    CALL SQUARE(FBH,NORBS)
                    CALL DIAG (FBH,CBETAH,NB1EL,EIGB,NORBS,NORBS,WORK)
                 END IF
                 ! orig
                 !                   CALL SQUARE(FBH,NORBS)
                 !                   CALL DIAG (FBH,CBETAH,NB1EL,EIGB,NORBS,NORBS,WORK)
              ENDIF     ! QMLINK
           ENDIF        ! HALFE .OR. CAMKIN
        ELSE            ! NEWDG
           !
           ! For UHF-GHO ... PJ 12/2002
           IF (QMLINK) THEN
              CALL RSPQM(FBHB,NORBHB,NORBHB,EIGB,CBHB)
           ELSE
              CALL RSPQM(FBH,NORBS,NORBS,EIGB,CBETAH)
           ENDIF
        ENDIF           ! NEWDG
        !
        ! Calculate beta density matrix for UHF-GHO ... PJ 12/2002
        !
        IF(QMLINK) THEN
           CALL CALDENS(CBHB,NORBHB,NORBHB,NB2EL,NB1EL,FRACT,PBHB)

           ! do beta density damping for UHF-GHO
           !    ! Pi = a*Pi-1 + (1-a)Pi, default a = 0.0 (no damping)
           !    ! only use damping when denisty change is 100 times greater
           !      than the convergence criterion. This is added to ensure
           !      last several iterations are carried out without any damping.
           IF (QMLINK .AND. UHF)  THEN
              ilocal=NORBHB*(NORBHB+1)/2
              IF (NITER .GE. 2 .AND. PLB .GT. TEN2*PLTEST) THEN
                 PBHB(1:ilocal) = PALPHA*PBPRE(1:ilocal) + (1-PALPHA)*PBHB(1:ilocal)
              END IF
              PBPRE(1:ilocal) = PBHB(1:ilocal)
           END IF

           !   Relocate the positions of active hybrid orbitals if
           !   there are more than one QM-boundary atom.
           DO I = NORBHB,NAOS+2,-1
              IQATM = I-NAOS
              II = I*(I-1)/2
              IORB1B = NAOS+4*(IQATM-1)
              JJ = IORB1B*(IORB1B+1)/2
              DO J = 1,NAOS
                 II = II+1
                 JJ = JJ+1
                 PBHB(JJ) = PBHB(II)
                 PBHB(II) = zero
              ENDDO
              !
              !   HB-HB blocks
              DO J = 1,IQATM
                 II = II+1
                 JJ = JJ+1
                 PBHB(JJ) = PBHB(II)
                 PBHB(II) = zero
                 IF(J.NE.IQATM) THEN
                    PBHB(JJ+1:JJ+3) = zero
                    JJ = JJ+3
                 ENDIF
              ENDDO
              !   The rest three auxiliary orbitals
              DO J = 2,4
                 PBHB(JJ+1:jj+IORB1B+J)=zero
                 JJ = JJ+IORB1B+J
                 PBHB(JJ) = (one-QMATMQM(IQATM)*rthree)*half  ! /3.0D0)/2.0D0
              ENDDO
           ENDDO

           !
           !   Auxiliary density for the first QM-boundary atom
           K = NAOS+2              ! N1ST(IMQLINK(1))+1
           L = NAOS+4                 ! NLAST(IMQLINK(1))
           DO I = K,L
              JJ = I*(I-1)/2
              PBHB(JJ+1:jj+I)=zero
              PBHB(JJ+I) = (one-QMATMQM(1)*rthree)*half  ! /3.0D0)/2.0D0
           ENDDO
           !
           !   AO blocks...Not affected by orbital transformation
           PB(1:LINAO) = PBHB(1:LINAO)

           IJ = LINAO
           !   Loop over QM-boundary atoms
           DO K = 1,MQMLNK
              J1 = 16*(K-1)
              I1 = NAOS+4*(K-1)
              IORB1B = I1*(I1+1)/2
              !
              !      FOR EACH BOUNDARY ATOM, THERE ARE FOUR AOs.
              !      BOUNDARY ATOMS MUST BE NON-HYDROGEN ATOMS...
              L = 0
              DO I = 1,4
                 !
                 !      SINCE ONLY ONE HYBRID-ORBITAL DENSITY ON THE BOUNDARY ATOM IS NON-ZERO
                 !      SUM OVER ORBITALS IS NOT NEEDED
                 XBT1 = MBT(J1+I)                   ! MBTM(J1+4*(I-1)+1)
                 PB(IJ+1:IJ+NAOS) = PBHB(IORB1B+1:IORB1B+NAOS)*XBT1
                 IJ = IJ+NAOS
                 !
                 !   BOUNDARY ATOM-OTHER BOUNDARY ATOM BLOCKS
                 DO L = 1,K-1
                    KK1 = 16*(L-1)
                    KK3 = IORB1B+NAOS+4*(L-1)+1
                    XBT2= PBHB(KK3)*XBT1
                    PB(IJ+1:IJ+4) = XBT2*MBT (KK1+1:KK1+4)   ! MBTM(KK1+4*(J-1)+1)
                    IJ = IJ+4
                 END DO
                 !
                 !   BOUNDARY ATOM SELF BLOCK
                 KK1 = 4*(I-1)+J1
                 DO J = 1,I
                    IJ = IJ+1
                    KK2 = 4*(J-1)+J1
                    PB(IJ) = zero
                    DO L = 1,4
                       KK3 = (I1+L)*(I1+L+1)/2
                       PB(IJ) = PB(IJ)+PBHB(KK3)*MBTM(KK1+L)*MBTM(KK2+L)
                    END DO
                 END DO
              END DO  ! I = 1,4
           END DO     ! K = 1,MQMLNK
        END IF        ! QMLINK
        !
        IF (PRTVEC) THEN
           IF(PRNLEV.GE.2) WRITE (OUTU,'(A,I4)') ' ITER> Beta eigenvectors and eigenvalues on iteration', NITER
           CALL MATOUT(CBETAH,EIGB,NORBS,NORBS,NORBS)
        ELSE IF (PRTEIG.AND.PRNLEV.GE.2) THEN
           WRITE (OUTU,'(A,I4)') ' ITER> Beta eigenvalues on iteration ',NITER
           WRITE (OUTU,'(8X,6G13.6)') (EIGB(I) , I = 1,NORBS)
        ENDIF
     End if           ! MODEB .ne. 3
     !=========================================================================
     !     Calculate the beta density matrix.
     !=========================================================================
     !
     ! For UHF-GHO ... PJ 12/2002
     IF (.NOT. QMLINK) CALL CALDENS(CBETAH,NORBS,NORBS,NB2EL,NB1EL,FRACT,PB)
     IF (.NOT. (NEWDG.AND.OKPULY)) CALL CNVG(PB,PBOLDH,PBOLD2,NORBS,NITER,PLB,.TRUE.)
  ENDIF      ! UHF
  !=========================================================================
  !     Calculate the total density matrix.
  !=========================================================================
  IF (UHF) THEN
     P(1:LINEAR) = PA(1:LINEAR) + PB(1:LINEAR)
  ELSE
     do I=1,LINEAR
        PA(I) = HALF * P(I)
        PB(I) = PA(I)
     end do
  ENDIF
  !
  IF (PRTDEN) THEN
     IF(PRNLEV.GE.2) WRITE (OUTU,'(A,I4)') ' ITER> Total density matrix on iteration ',NITER
     CALL MNVECPRT (P,NORBS)
  ENDIF
  !
  OKNEWD = (PL .LT. SELCON .OR. OKNEWD)
  NEWDG  = (PL .LT. TRANS .AND. OKNEWD .OR. NEWDG)
  IF (PL .LT. (TRANS / THREE)) OKNEWD = .TRUE.
  !
  IF (NITER .GE. ITRMAX) THEN
     If(PRNLEV.GE.2) then
        WRITE (OUTU,'(A)') ' ITER> ***** UNABLE TO ACHIEVE SELF-CONSISTENCE *****'
        WRITE (OUTU,'(8X,A,E12.4,A,2E12.4)') ' DeltaE = ',DIFF,' and deltaP = ',PL,PLB
     End if
     CALL WRNDIE(-5,'<ITER2>','UNABLE TO ACHIEVE SELF-CONSISTENCE.')
  ENDIF
  !
  GOTO 200
  !=========================================================================
  !     Exit the loop - SCF achieved.
  !=========================================================================
400 CONTINUE

  EE = MNHELECT(NORBS,PA,H,F,.TRUE.)
  IF (UHF) THEN
     EE = EE + MNHELECT(NORBS,PB,H,FBH,.TRUE.)
  ELSE
     !        EE = TWO * EE + PT25 * SHIFT * (NOPEN - NCLOSE)
     EE = TWO * EE
  ENDIF
  !
  ! go parallel
#if KEY_PARALLEL==1
  !  DTM PI
  IF(NUMNOD.NE.1 .AND. .NOT. QMPI) THEN
     call psync()
     call GCOMB(EE,1)
  END IF
#endif 
  !
  IF (.NOT.UHF) EE = EE + PT25 * SHIFT * (NOPEN - NCLOSE)
  !
#if KEY_UNUSED==1 /*capcor_unused*/
  !      EE = EE + CAPCOR(NAT,N1ST,NLAST,NATQM,P,H)
#endif /* (capcor_unused)*/
  !
  !     Do exact diagonalisations of the Fock matrices using POLD2 as a
  !     temporary array.
  !
  IF(.NOT.QMLINK .AND. (QFIRST .OR. ABS(SHIFT) .GT. TENM5 .OR. HALFE)) THEN
     POLD2(1:LINEAR) = F(1:LINEAR)
     CALL RSPQM(POLD2,NORBS,NORBS,EIGS,C)
     IF (UHF) THEN
        POLD2(1:LINEAR)=FBH(1:LINEAR)
        CALL RSPQM(POLD2,NORBS,NORBS,EIGB,CBETAH)
     ENDIF
     !
     IF (CI .OR. HALFE) THEN
        ECI = MECI(EIGS,C,CBETAH,EIGB, NORBS,NMOS,NCIS,.FALSE.,1,P,PA,PB)
        EE  = EE + ECI
     ENDIF
  ENDIF
  !
  IF (DEBUG.AND.PRNLEV.GE.2) WRITE (OUTU,'(A,I4)') ' ITER> Number of iterations = ',NITER
  !
  IF (ALLCON .AND. ABS(BSHIFT - BFACT) .LT. TENM2) THEN
     ALLCON = .FALSE.
     CAMKIN = .FALSE.
     NEWDG  = .FALSE.
     OKPULY = .FALSE.
     BSHIFT = ZERO
  END IF
  IF (HALFE) BSHIFT = ZERO
  QFIRST = .FALSE.
  !
  RETURN
END SUBROUTINE ITER2
!
SUBROUTINE MNCHRGE(P,Q)
  !***********************************************************************
  !      CHRGE STORES IN Q THE TOTAL ELECTRON DENSITIES ON THE ATOMS
  !      ON INPUT P      = DENSITY MATRIX
  !      ON OUTPUT Q     = ATOM ELECTRON DENSITIES
  !***********************************************************************
  use chm_kinds
  use number
  use dimens_fcm
  !
  use sizes
  use quantm
  implicit none
  !
  real(chm_real) P(*),Q(*)
  !
  INTEGER I,J,K,IA,IB
  !
  K=0
  Do I=1,NATQM
     IA=N1ST(I)
     IB=NLAST(I)
     Q(I)=ZERO
     Do J=IA,IB
        K=K+J
        Q(I)=Q(I)+P(K)
     End do
  End do
  !
  RETURN
END SUBROUTINE MNCHRGE

SUBROUTINE MNDCART(COORD,DXYZ,P,PA,PB,UHF,ISTORE)
  !***********************************************************************
  !
  !    DCART CALCULATES THE DERIVATIVES OF THE ENERGY WITH RESPECT TO THE
  !          CARTESIAN COORDINATES. THIS IS DONE BY FINITE DIFFERENCES.
  !
  !    THE MAIN ARRAYS IN DCART ARE:
  !        DXYZ   ON EXIT CONTAINS THE CARTESIAN DERIVATIVES.
  !
  !***********************************************************************
  use chm_kinds
  use dimens_fcm
  use number,only : zero, one
  use consta,only : EV_TO_KCAL
  !
  use stream
  use sizes
  use quantm
  !
  ! for parallel run
#if KEY_PARALLEL==1
  use parallel
#endif 
  implicit none
  !
  LOGICAL UHF
  real(chm_real) COORD(3,*),DXYZ(3,*),ISTORE(*), P(*), PA(*), PB(*)
  !
  real(chm_real) PDI(171),PADI(171),PBDI(171), CDI(3,2), ENG(3)
  INTEGER NDI(2)
  !
  real(chm_real) AA,EE,DERIV,CREFT
  INTEGER IF,JF,II,IJ,JJ,IL,IM,JL,JM,KREP,IM1,I,J,K,L,IREAD
  !
  logical :: MAKEP
  logical, save :: FIRST=.true.
  logical, save :: DEBUG1,ANADER,FORCE

  real(chm_real) :: CHNGE=1.0D-4, &     ! old: 1.0D-6, CHNGE IS A MACHINE-PRECISION DEPENDENT CONSTANT
       CHNGE2=5.0D-5, &    !      5.0D-7, CHNGE2=CHNGE/2 
       RCHNGE=1.0D4, &     !      1.0D6
       RCHNGE2=2.0D4       !      2.0D6
  !
  ! go parallel
  INTEGER ATFRST,ATLAST,NODNQM,NTOTERM,NTERM
  !
  IF (FIRST) THEN
     ANADER = (INDEX(KEYWRD,'ANALYT') .NE. 0)
     DEBUG1 = (INDEX(KEYWRD,'DCART') .NE. 0)
     FORCE = (INDEX(KEYWRD,'PRECISE')+INDEX(KEYWRD,'FORCE') .NE. 0)
     FIRST = .FALSE.
  ENDIF
  DXYZ(1:3,1:NATQM)=zero
  !
  KREP  = 0
  IREAD = 0
  ! go parallel
  nterm = 0
  ntoterm=0
  do i=2,natqm
     ntoterm=ntoterm+(i-1)
  end do
  !
  ! go parallel
#if KEY_PARALLEL==1 /*paramain*/
  nodnqm = ntoterm /numnod
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
  ATFRST=MYNOD*NODNQM + 1
  ATLAST=(MYNOD+1)*NODNQM
  IF(MYNOD.EQ.(NUMNOD-1)) THEN
     ATLAST=ntoterm
  END IF
  IF(ATLAST.GT.ntoterm.AND.ATFRST.LE.ntoterm) ATLAST=ntoterm
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
  ATFRST=1
  ATLAST=ntoterm
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
  !  DTM PI
  IF (QMPI) THEN
     ATFRST=1
     ATLAST=ntoterm
  ENDIF
#else /* (paramain)*/
  ATFRST=1
  ATLAST=ntoterm
#endif /* (paramain)*/
  !
  loopII: Do II=1,NATQM
     IM1=II
     IF=N1ST(II)
     IM=NMIDLE(II)
     IL=NLAST(II)
     NDI(2)=NAT(II)
     !         IF (NDI(2) .EQ. 0) GOTO 120
     CDI(1:3,2)=COORD(1:3,II)

     loopJJ: Do JJ=1,IM1
        !  FORM DIATOMIC MATRICES
        JF=N1ST(JJ)
        JM=NMIDLE(JJ)
        JL=NLAST(JJ)
        !   GET FIRST ATOM
        NDI(1)=NAT(JJ)
        !            IF (NDI(1) .EQ. 0) GOTO 120
        MAKEP=.TRUE.
        CDI(1:3,1)=COORD(1:3,JJ)
        if(.NOT. MAKEP) then
           continue
        else
           MAKEP=.FALSE.
           IJ=0
           do I=JF,JL
              K=I*(I-1)/2+JF-1
              do J=JF,I
                 IJ=IJ+1
                 K =K+1
                 PADI(IJ)=PA(K)
                 PBDI(IJ)=PB(K)
                 PDI(IJ) =P(K)
              end do
           end do
           ! GET SECOND ATOM FIRST ATOM INTERSECTION
           do I=IF,IL
              L=I*(I-1)/2
              K=L+JF-1
              do J=JF,JL
                 IJ=IJ+1
                 K =K+1
                 PADI(IJ)=PA(K)
                 PBDI(IJ)=PB(K)
                 PDI(IJ) =P(K)
              end do
              K=L+IF-1
              do L=IF,I
                 K =K+1
                 IJ=IJ+1
                 PADI(IJ)=PA(K)
                 PBDI(IJ)=PB(K)
                 PDI(IJ) =P(K)
              end do
           end do
        end if   ! .NOT. MAKEP
        If(II.ne.JJ) then      ! IF(II.EQ.JJ) GOTO  120
           !
#if KEY_PARALLEL==1
           !  DTM PI
           IF (.NOT. QMPI) THEN
              nterm=nterm+1
              if (nterm.lt.atfrst) then
                 if(ANADER) iread=iread+1
                 cycle loopJJ                ! goto 120
              else if(nterm.gt.atlast) then
                 EXIT loopII                 ! goto 220
              end if
           ENDIF
#endif 
           IF (ANADER) THEN
              IREAD = IREAD + 1
              CALL MNANALYT (PDI,PADI,PBDI,CDI,NDI,JF,JL,IF,IL,NORBS,ENG,KREP,ISTORE,IREAD)
              DXYZ(1:3,II) = DXYZ(1:3,II) + ENG(1:3)
              DXYZ(1:3,JJ) = DXYZ(1:3,JJ) - ENG(1:3)
           ELSE
              IF(FORCE) THEN
                 DO K =1 ,3
                    CREFT   =CDI(K,2)
                    CDI(K,2)=CREFT  + CHNGE2
                    CALL MNDHC(PDI,PADI,PBDI,CDI,NDI,JF,JM,JL,IF,IM,IL,AA,UHF)
                    CDI(K,2)=CREFT  - CHNGE2
                    CALL MNDHC(PDI,PADI,PBDI,CDI,NDI,JF,JM,JL,IF,IM,IL,EE,UHF)
                    CDI(K,2)=CREFT
                    DERIV=(AA-EE)*EV_TO_KCAL*RCHNGE2             ! 23.061D0
                    DXYZ(K,II)=DXYZ(K,II)-DERIV
                    DXYZ(K,JJ)=DXYZ(K,JJ)+DERIV
                 END DO
              ELSE
                 DO K =1 ,3
                    CREFT   =CDI(K,2)
                    CDI(K,2)=CREFT  + CHNGE
                    CALL MNDHC(PDI,PADI,PBDI,CDI,NDI,JF,JM,JL,IF,IM,IL,AA,UHF)
                    CDI(K,2)=CREFT  - CHNGE
                    CALL MNDHC(PDI,PADI,PBDI,CDI,NDI,JF,JM,JL,IF,IM,IL,EE,UHF)
                    CDI(K,2)=CREFT
                    DERIV=(AA-EE)*EV_TO_KCAL*RCHNGE              ! 23.061D0
                    DXYZ(K,II)=DXYZ(K,II)-DERIV
                    DXYZ(K,JJ)=DXYZ(K,JJ)+DERIV
                 END DO
              END IF
              !
              ! start *****************************************************************************
              !                  IF( .NOT. FORCE) THEN
              !                     CDI(1:3,1)=CDI(1:3,1)+CHNGE2
              !CDCC#                     CALL MNDHC(PDI,PADI,PBDI,CDI,NDI,JF,JM,JL,IF,IM,IL,NORBS,AA,UHF)
              !                     CALL MNDHC(PDI,PADI,PBDI,CDI,NDI,JF,JM,JL,IF,IM,IL,AA,UHF)
              !CDCC+
              !                  ENDIF
              !                  DO K=1,3
              !                     IF( FORCE )THEN
              !                        CDI(K,2)=CDI(K,2)-CHNGE2
              !CDCC#                        CALL MNDHC(PDI,PADI,PBDI,CDI,NDI,JF,JM,JL,IF,IM,IL,NORBS,AA,UHF)
              !                        CALL MNDHC(PDI,PADI,PBDI,CDI,NDI,JF,JM,JL,IF,IM,IL,AA,UHF)
              !                     ENDIF
              !                     CDI(K,2)=CDI(K,2)+CHNGE
              !CDCC#                     CALL MNDHC(PDI,PADI,PBDI,CDI,NDI,JF,JM,JL,IF,IM,IL,NORBS,EE,UHF)
              !                     CALL MNDHC(PDI,PADI,PBDI,CDI,NDI,JF,JM,JL,IF,IM,IL,EE,UHF)
              !                     CDI(K,2)=CDI(K,2)-CHNGE2
              !                     IF( .NOT. FORCE) CDI(K,2)=CDI(K,2)-CHNGE2
              !                     DERIV=(AA-EE)*46.122D0/CHNGE
              !                     DXYZ(K,II)=DXYZ(K,II)+DERIV
              !                     DXYZ(K,JJ)=DXYZ(K,JJ)-DERIV
              !                  END DO
              ! end  ******************************************************************************
              !
           ENDIF      ! ANADER
           !#             WRITE(6,*)' WHOLE OF DXYZ',II,JJ,IK,JK,KL
           !#             WRITE(6,'(3(3F17.5,/),/)')((DXYZ(J,I),J=1,3),I=1,18)

        End if         ! II.ne.JJ
120     Continue
     End do loopJJ     ! JJ=1,IM1
  End do  loopII       ! II=1,NATQM 
  !
  ! go parallel: out of the do-loop 
220 continue       ! nterm.gt.atlast
  !
  If (DEBUG1 .and. PRNLEV.GE.2) then 
     WRITE(6,'(//10X,''CARTESIAN COORDINATE DERIVATIVES'',//3X,''ATOM  AT. NO.'',5X,''X'',12X,''Y'',12X,''Z'',/)')
     WRITE(6,'(2I6,F13.6,2F13.6)') (I,NAT(I),(DXYZ(J,I),J=1,3),I=1,NATQM)
  End if
  RETURN
END SUBROUTINE MNDCART

SUBROUTINE MNFOCK1(F, PTOT, PA, PB, X, Y, Z, XI, YI, ZI)
  ! *********************************************************************
  !
  ! *** COMPUTE THE REMAINING CONTRIBUTIONS TO THE ONE-CENTRE ELEMENTS.
  !
  ! *********************************************************************
  !
  use chm_kinds
  use dimens_fcm
  use number, only : half,zero,one,two
  use consta, only : BOHRR,EV_TO_KCAL,AU_TO_EV
  !
  use sizes
  use quantm
  ! namkh 08/08/04
  ! QM/MM-Ewald
  use am1parm
  !
  ! for parallel run
#if KEY_PARALLEL==1
  use parallel
#endif 
  implicit none
  !
  INTEGER I1,I2,itemp
  !
  real(chm_real) F(*), PTOT(*), PA(*), PB(*)
  real(chm_real) X(*), Y(*), Z(*), XI(*), YI(*), ZI(*)
  !
  real(chm_real) QTOT(MAXQM1), QA(MAXQM1), QB(MAXQM1)
  !
  real(chm_real) PAPOP,PBPOP,DTPOP,PTPOP,DAPOP,DBPOP
  INTEGER IPLUS,IA,IB,IC,KA,II,NI,I,J,L,M,IMINUS,ICC

  real(chm_real) RFLAMB,temp0,temp1,temp2
  real(chm_real), parameter :: A0C  = BOHRR, &      ! 0.529167D0, A->B
       EVC  = AU_TO_EV      ! 27.21D0   , cal/mol -> EV
  !
  ! go parallel
  INTEGER ATFRST,ATLAST,NODELE
  real(chm_real)  EMPOTF
  !

  RFLAMB=one
  IF(QMPERT) RFLAMB=RLAMBF
  !
  ! go parallel
#if KEY_PARALLEL==1 /*paramain*/
  NODELE = NATQM / NUMNOD
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
  ATFRST=MYNOD*NODELE + 1
  ATLAST=(MYNOD+1)*NODELE
  IF(MYNOD.EQ.(NUMNOD-1)) ATLAST=NATQM
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
  ATFRST=1
  ATLAST=NATQM
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
  !  DTM PI
  IF (QMPI) THEN
     ATFRST=1
     ATLAST=NATQM
  ENDIF
#else /* (paramain)*/
  ATFRST=1
  ATLAST=NATQM
#endif /* (paramain)*/
  !
  CALL MNCHRGE(PTOT,QTOT)
  CALL MNCHRGE(PA,QA)
  QB(1:NATQM)=QTOT(1:NATQM)-QA(1:NATQM)
  !
  ! QM/MM-Ewald 
  CHAG(1:NATQM)=CORE(NAT(1:NATQM))-QTOT(1:NATQM)
  !
  !
  ! namkh 08/08/04
  ! QM/MM-Ewald
  IF(LQMEWD) THEN
     CALL QEWALDP(X,Y,Z,XI,YI,ZI,.TRUE.)
     !
     DO I = 1, NATQM
        IA=N1ST(I)
        IB=NLAST(I)
        EMPOTF = RFLAMB*(EMPOT(I)+ESLF(I))*EVC*A0C
        DO I1=IA,IB
           I2=I1*(I1-1)/2+I1
           F(I2) = F(I2)-EMPOTF
        END DO
     END DO
  END IF
  !
  loopII: Do II=ATFRST, ATLAST   ! II=1,NATQM
     IA=N1ST(II)
     IB=NMIDLE(II)
     IC=NLAST(II)
     NI=NAT(II)
     DTPOP=zero
     DAPOP=zero
     PTPOP=zero
     PAPOP=zero

     itemp = IC-IA+2
     if(itemp.le.1 .or.  itemp.ge.11) then  ! ===1, or 11=== : 120 or exit loop
        cycle loopII
     else 
        if(itemp.ge.6) then                 ! 6:10: goto 40
           DTPOP=PTOT((IC*(IC+1))/2)+PTOT(((IC-1)*(IC))/2)+PTOT(((IC-2)*(IC-1))/2)+PTOT(((IC-3)*(IC-2))/2) &
                +PTOT(((IC-4)*(IC-3))/2)
           DAPOP=PA((IC*(IC+1))/2)  +PA(((IC-1)*(IC))/2)  +PA(((IC-2)*(IC-1))/2)  +PA(((IC-3)*(IC-2))/2)  &
                +PA(((IC-4)*(IC-3))/2)
        end if
        if(itemp.ge.3) then                 ! 3:5: goto 50
           PTPOP=PTOT((IB*(IB+1))/2)+PTOT(((IB-1)*(IB))/2)+PTOT(((IB-2)*(IB-1))/2)
           PAPOP=PA((IB*(IB+1))/2)  +PA(((IB-1)*(IB))/2)  +PA(((IB-2)*(IB-1))/2)
        end if
        DBPOP=DTPOP-DAPOP                      ! 2: goto 60
        PBPOP=PTPOP-PAPOP
     end if
     !
     !     F(S,S)
     !
     KA   =(IA*(IA+1))/2
     F(KA)= F(KA)+PB(KA)*GSS(NI)+PTPOP*GSP(NI)-PAPOP*HSP(NI)+DTPOP*GSD(NI) 
     if (NI.GE.3) then  
        IPLUS=IA+1
        L=KA
        do J=IPLUS,IB
           M=L+IA
           L=L+J
           !
           !     F(P,P)
           !
           F(L)=F(L)+PTOT(KA)*GSP(NI)-PA(KA)*HSP(NI)+PB(L)*GPP(NI)+(PTPOP-PTOT(L))*GP2(NI) &
                -half*(PAPOP-PA(L))*(GPP(NI)-GP2(NI))         + DTPOP*GPD(NI)
           !
           !     F(S,P)
           !
           F(M)=F(M)+two*PTOT(M)*HSP(NI)-PA(M)*(HSP(NI)+GSP(NI))
        end do
        !
        !     F(P,P*)
        !
        IMINUS=IB-1
        do J=IPLUS,IMINUS
           ICC=J+1
           do L=ICC,IB
              M=(L*(L-1))/2+J
              F(M)=F(M)+PTOT(M)*(GPP(NI)-GP2(NI))-half*PA(M)*(GPP(NI)+GP2(NI))
           end do
        end do
        do J=IB+1,IC
           M=(J*(J+1))/2
           F(M)=F(M)+PTOT(KA)*GSD(NI)+PTPOP*GPD(NI)+(DTPOP-PA(M))*GDD(NI) 
        end do
     end if      ! NI.LT.3
  End do loopII
  RETURN
END SUBROUTINE MNFOCK1

SUBROUTINE MNHCORE (COORD,H,W, WJ,WK,ENUCLR,ISTORE)
  !
  use chm_kinds
  use dimens_fcm
  use number, only : half,zero,one,two
  !
  use stream
  use sizes
  use quantm
  !
  ! for parallel run
#if KEY_PARALLEL==1
  use parallel
#endif 
  implicit none
  !
  real(chm_real) COORD(3,*),H(*), WJ(N2ELEC), WK(N2ELEC), W(N2ELEC)
  real(chm_real) ISTORE(*)
  !************************************************************************
  !
  !   HCORE GENERATES THE ONE-ELECTRON MATRIX AND TWO ELECTRON INTEGRALS
  !         FOR A GIVEN MOLECULE WHOSE GEOMETRY IS DEFINED IN CARTESIAN
  !         COORDINATES.
  !
  !  ON INPUT  COORD   = COORDINATES OF THE MOLECULE.
  !
  !  ON OUTPUT  H      = ONE-ELECTRON MATRIX.
  !             W      = TWO-ELECTRON INTEGRALS.
  !             ENUCLR = NUCLEAR ENERGY
  !***********************************************************************
  !
  real(chm_real) HALF_local,ENUC,ENUCLR
  INTEGER IA,IB,JA,IC,JB,JC,II,JJ,NI,NJ,KR,IM1
  INTEGER I,J,I1,I2,IREAD,J1,J2
  !
  real(chm_real) E1B(10),E2A(10),DI(9,9), WJD(100), WKD(100)
  logical, save :: FIRST=.TRUE.
  logical, save :: DEBUG1

  Integer,parameter :: IONE=1
  real(chm_real),parameter :: CUTOFF=1.0D10
  !
  ! go parallel
  INTEGER nterm
  INTEGER :: ATFRST,ATLAST,NODNQM,NII,NJJ,KKJ(10)=(/1,3,6,10,15,21,28,36,45,55/)
  !
  IF (FIRST) THEN
     FIRST=.FALSE.
     DEBUG1=(INDEX(KEYWRD,'HCORE') .NE. 0)
  ENDIF
  !
  ! go parallel
#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
  ATFRST=MYNOD + 1
  ATLAST=NATQM
  nterm = numnod - 1
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
  ATFRST=1
  ATLAST=NATQM  
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
  !  DTM PI
  IF (QMPI) THEN
     ATFRST=1
     ATLAST=NATQM
  ENDIF
#else /* (paramain)*/
  ATFRST=1
  ATLAST=NATQM   
#endif /* (paramain)*/
  ! 
  !     H(1:(NORBS*(NORBS+1))/2) = zero     ! This been done in qmene.src
  IREAD = 0
  ENUCLR= zero
  KR    = 1
  !
  ! go parallel
#if KEY_PARALLEL==1
#if KEY_PARAFULL==1
  !  DTM PI
  IF (.NOT. QMPI) THEN
     do i=1,ATFRST-1
        iread = iread + (i-ione)
        nii = NMIDLE(I)-N1ST(I)+1
        nii = min(nii,4)
        do j=1,(i-ione)
           njj=NMIDLE(J)-N1ST(J)+1
           njj=MIN(njj,4)
           KR=KR+KKJ(NII)*KKJ(NJJ)
        end do
     end do
  ENDIF
#endif 
#endif 
  !
  ! go parallel
  Do I=ATFRST, ATLAST   ! I = 1,NATQM
     IA=N1ST(I)
     IB=NLAST(I)
     IC=NMIDLE(I)
     NI=NAT(I)
     ! 
#if KEY_PARALLEL==1
#if KEY_PARAFULL==1
     !  DTM PI
     IF (.NOT. QMPI) THEN
        nterm=nterm+1
        if(nterm.ne.numnod) then
           iread = iread + (i-ione)
           nii   = MIN((IC-IA+1),4)
           do j =1,(i-ione)
              njj=NMIDLE(J)-N1ST(J)+1
              njj=MIN(njj,4)
              KR=KR+KKJ(NII)*KKJ(NJJ)
           end do
           goto 110
        else 
           nterm= 0    
        end if
     ENDIF
#endif 
#endif 
     !
     ! FIRST WE FILL THE DIAGONALS, AND OFF-DIAGONALS ON THE SAME ATOM
     !
     do I1=IA,IB
        I2=I1*(I1-1)/2+IA-1
        do J1=IA,I1
           I2=I2+1
           H(I2)=H(I2)+zero
        end do
        H(I2)=USPD(I1)
     end do
     IM1=I-IONE
     !       write(6,*) 'hcore:',i,nat(i),ia,ib,natqm
     do J=1,IM1
        if(I.EQ.J) then
           HALF_local=half
        else
           HALF_local=one
        end if
        JA=N1ST(J)
        JB=NLAST(J)
        JC=NMIDLE(J)
        NJ=NAT(J)
        CALL MNH1ELEC(NI,NJ,COORD(1,I),COORD(1,J),DI)
        !
        !   FILL THE ATOM-OTHER ATOM ONE-ELECTRON MATRIX<PSI(LAMBDA)|PSI(SIGMA)>
        !
        I2=0
        do I1=IA,IB
           II=I1*(I1-1)/2+JA-1
           I2=I2+1
           J2=0
           JJ=MIN(I1,JB)
           do J1=JA,JJ
              II=II+1
              J2=J2+1
              H(II)=H(II)+DI(I2,J2)
           end do
        end do
        !
        !   CALCULATE THE TWO-ELECTRON INTEGRALS, W; THE ELECTRON NUCLEAR TERMS
        !   E1B AND E2A; AND THE NUCLEAR-NUCLEAR TERM ENUC.
        !
        IREAD = IREAD + 1
        CALL MNROTATE(NI,NJ,COORD(1,I),COORD(1,J),W(KR),KR,E1B,E2A,ENUC,CUTOFF,ISTORE,IREAD)
        ENUCLR = ENUCLR + ENUC
        !
        !   ADD ON THE ELECTRON-NUCLEAR ATTRACTION TERM FOR ATOM I.
        !
        I2=0
        do I1=IA,IC
           II=I1*(I1-1)/2+IA-1
           do J1=IA,I1
              II=II+1
              I2=I2+1
              H(II)=H(II)+E1B(I2)*HALF_local
           end do
        end do
        do I1=IC+1,IB
           II=(I1*(I1+1))/2
           H(II)=H(II)+E1B(1)*HALF_local
        end do
        !
        !   ADD ON THE ELECTRON-NUCLEAR ATTRACTION TERM FOR ATOM J.
        !
        I2=0
        do I1=JA,JC
           II=I1*(I1-1)/2+JA-1
           do J1=JA,I1
              II=II+1
              I2=I2+1
              H(II)=H(II)+E2A(I2)*HALF_local
           end do
        end do
        do I1=JC+1,JB
           II=(I1*(I1+1))/2
           H(II)=H(II)+E2A(1)*HALF_local
        end do
     end do     ! J=1,IM1
110  CONTINUE
  End do        ! I=ATFRST, ATLAST
  If(DEBUG1) then
     IF(PRNLEV.GE.2) WRITE(6,'(//10X,''ONE-ELECTRON MATRIX FROM HCORE'')')
     CALL MNVECPRT(H,NORBS)
     J=MIN(400,KR)
     IF(PRNLEV.GE.2) THEN
        WRITE(6,'(//10X,''TWO-ELECTRON MATRIX IN HCORE''/)')
        WRITE(6,120)(W(I),I=1,J)
     END IF
  End if
120 FORMAT(10F8.4)
  RETURN
END SUBROUTINE MNHCORE

SUBROUTINE MATOUT (A,B,NC,NR,NDIM)
  !**********************************************************************
  !
  !      MATOUT PRINTS A SQUARE MATRIX OF EIGENVECTORS AND EIGENVALUES
  !
  !    ON INPUT A CONTAINS THE MATRIX TO BE PRINTED.
  !             B CONTAINS THE EIGENVALUES.
  !             NC NUMBER OF MOLECULAR ORBITALS TO BE PRINTED.
  !             NR IS THE SIZE OF THE SQUARE ARRAY TO BE PRINTED.
  !             NDIM IS THE ACTUAL SIZE OF THE SQUARE ARRAY "A".
  !             N1ST AND NLAST CONTAIN ATOM ORBITAL COUNTERS.
  !             NAT = ARRAY OF ATOMIC NUMBERS OF ATOMS.
  !
  !
  !***********************************************************************
  !
  use chm_kinds
  use dimens_fcm
  !
  use am1parm,only:elemnt
  use stream
  use sizes
  use quantm
  implicit none
  INTEGER NC,NR,NDIM
  real(chm_real) A(NDIM,NDIM), B(NDIM)
  !
  CHARACTER(len=2) :: ATORBS(9)=(/' S','PX','PY','PZ','X2','XZ','Z2','YZ','XY'/)
  CHARACTER(len=2) ITEXT(MAXORB), JTEXT(MAXORB)
  INTEGER NATOM(MAXORB)
  !
  INTEGER KA,KB,LA,KC,LB,LC,I,J,K,L,JHI,JLO
  !
  if(NATQM.EQ.0 .or. NLAST(NATQM).NE.NR) then
     NR=ABS(NR)
     do I=1,NR
        ITEXT(I)='  '
        JTEXT(I)='  '
        NATOM(I)=I
     end do
  else
     do I=1,NATQM
        JLO=N1ST(I)
        JHI=NLAST(I)
        L=NAT(I)
        K=0
        do J=JLO,JHI
           K=K+1
           ITEXT(J)=ATORBS(K)
           JTEXT(J)=ELEMNT(L)
           NATOM(J)=I
        end do
     end do
  end if

  KA=1
  KC=6
60 KB=MIN0(KC,NC)
  IF(PRNLEV.GE.2) THEN
     WRITE (6,100) (I,I=KA,KB)
     IF(B(1).NE.0.D0)WRITE (6,110) (B(I),I=KA,KB)
     WRITE (6,120)
  END IF
  LA=1
  LC=40
70 LB=MIN0(LC,NR)
  IF(PRNLEV.GE.2) THEN
     do I=LA,LB
        IF(ITEXT(I).EQ.' S')WRITE(6,120)
        WRITE (6,130) ITEXT(I),JTEXT(I),NATOM(I),(A(I,J),J=KA,KB)
     end do
  END IF
  if(LB.ne.NR) then
     LA=LC+1
     LC=LC+40
     IF(PRNLEV.GE.2) WRITE (6,140)
     GO TO 70
  end if
  if (KB.EQ.NC) then
     RETURN
  else
     KA=KC+1
     KC=KC+6
     IF (NR.GT.25.AND.PRNLEV.GE.2) WRITE (6,140)
     GO TO 60
  end if
  !
100 FORMAT (////,3X,9H ROOT NO.,I5,9I12)
110 FORMAT (/8X,10F12.5)
120 FORMAT (2H  )
130 FORMAT (1H ,2(1X,A2),I3,F10.5,10F12.5)
140 FORMAT (1H1)
  !
END SUBROUTINE MATOUT
!
FUNCTION MECI(EIGS,COEFF,COEFFS,EIGA,N,NMOS,IDUMMY,FINISH,LROOT,P,PA,PB) result(meci_rtn)
  !***********************************************************************
  !*
  !*                 PROGRAM MECI
  !*
  !*   A MULTI-ELECTRON CONFIGURATION INTERACTION CALCULATION
  !*
  !*   WRITTEN BY JAMES J. P. STEWART, AT THE
  !*              FRANK J. SEILER RESEARCH LABORATORY
  !*              USAFA, COLORADO SPRINGS, CO 80840
  !*
  !*              1985
  !*
  !***********************************************************************
  !
  use chm_kinds
  use dimens_fcm
  !
  use stream
  use sizes
  use quantm
  implicit none
  !
  !   MATRICES FOR SEC.DET., VECTORS, AND EIGENVALUES.
  !
  real(chm_real) P(*), PA(*), PB(*), meci_rtn
  real(chm_real) CIMAT(NMECI**4), CONF(NMECI**4), EIG(NMECI**2), DIAG(2*NMECI**3)
  !
  !   MATRICES TO HOLD ELECTRON CONFIGURATIONS
  !
  INTEGER MICROA(NMECI,2*NMECI**3), MICROB(NMECI,2*NMECI**3), &
       IOCCA1(NMECI), IOCCA2(NMECI), IOCCB1(NMECI), &
       IOCCB2(NMECI), NALPHAO(NMECI**2)
  !
  !   MATRICES FOR PERMUTATION WORK
  !
  INTEGER NFA(2*NMECI), NPERMA(NMECI,6*NMECI), &
       NPERMB(NMECI,6*NMECI)
  !
  !   MATRICES FOR ONE AND TWO ELECTRON INTEGRALS, AND M.O.S
  !
  INTEGER N
  real(chm_real):: EIGA(MAXORB), EIGS(MAXORB), RJKAA(NMECI,NMECI), &
       RJKAB(NMECI,NMECI), COEFF(N,N),  COEFFS(N,N)
  !
  !   SPIN MATRICES
  !
  real(chm_real)  SPIN(NMECI**2)
  !
  INTEGER NDOWN,IROOT,KROOT,LROOT,II,JI,NE,IK,IL,LI,IN,IU,IX,IY, &
       LIMA,LIMB,MDIM,KALPHA,NMOS,KDELTA,NUPP,MAXVC,J,IOFSET, &
       L,NATOMS,IDUMMY,LAB,IUJ,I1,J1,K1,KBETA,L1,NELEC,NLEFT
  integer :: nterm_local
  real(chm_real) SMULT,AABABC,AABACD,BABBBC,AABBCD,BABBCD, &
       XX,SUMM,X,Y, &
       GSE,AMS,DIAGI,SUM
  !
  LOGICAL :: DEBUG1,  LARGE1, PRNT, FIRST=.TRUE., LSPIN, LSPIN1, FINISH, &
       FIRST1=.TRUE., BIGPRT, SING, DOUB, TRIP, QUAR, QUIN, SEXT, LAST1, &
       PRNT2
  CHARACTER(len=8) :: TSPIN(7)=(/'SINGLET ','DOUBLET ','TRIPLET ','QUARTET ','QUINTET ', &
       'SEXTET  ','SEPTET  '/)
  character(len=80) LINE
  integer:: I,K
  !***MFC ERROR KDELTA is not set before it is used
  !*** setting this to zero till someone sets it to the correct value 11/02
  kdelta=0
  !***MFC
  IF(FIRST)THEN
     FIRST=.FALSE.
     LAST1=(INDEX(KEYWRD,'1SCF').NE.0)
     LSPIN1=(INDEX(KEYWRD,'ESR').NE.0)
     MDIM=NMECI**2
     DEBUG1=(INDEX(KEYWRD,'DEBUG').NE.0)
     PRNT2=(INDEX(KEYWRD,'MECI').NE.0)
     DEBUG1=(DEBUG1.AND.PRNT2)
     LARGE1=(INDEX(KEYWRD,'LARGE').NE.0)
     J=(NCLOSE+NOPEN+1)/2-(NMOS-1)/2
     L=0
     nterm_local=NCLOSE-J+1
     OCCA_meci(L+1:L+nterm_local)=1.0D0
     L=L+nterm_local
     nterm_local=NOPEN-NCLOSE
     OCCA_meci(L+1:L+nterm_local)=FRACT*0.5D0
     L=L+nterm_local
     nterm_local=(J+NMOS-1)-NOPEN
     OCCA_meci(L+1:L+nterm_local)=0.0d0
     L=L+nterm_local 
     !#       WRITE(6,'('' INITIAL ORBITAL OCCUPANCIES'')')
     !#       WRITE(6,'(6F12.6)')(OCCA_meci(L),L=1,NMOS)
     SING=(INDEX(KEYWRD,'SINGL')+INDEX(KEYWRD,'EXCI')+INDEX(KEYWRD,'BIRAD').NE.0)
     DOUB=(INDEX(KEYWRD,'DOUBL').NE.0)
     TRIP=(INDEX(KEYWRD,'TRIPL').NE.0)
     QUAR=(INDEX(KEYWRD,'QUART').NE.0)
     QUIN=(INDEX(KEYWRD,'QUINT').NE.0)
     SEXT=(INDEX(KEYWRD,'SEXTE').NE.0)
     SMULT=-.5D0
     IF(SING) SMULT=0.00D0
     IF(DOUB) SMULT=0.75D0
     IF(TRIP) SMULT=2.00D0
     IF(QUAR) SMULT=3.75D0
     IF(QUIN) SMULT=6.00D0
     IF(SEXT) SMULT=8.75D0

     !#      WRITE(6,'('' ORBITAL COUNTERS, PER ATOM, FIRST LAST'')')
     !#      WRITE(6,'(I40,I6)')(N1ST(I),NLAST(I),I=1,NATQM)
     X=0.D0
     do J=1,NMOS
        X=X+OCCA_meci(J)
     end do
     XX=X+X
     NE=XX+0.5D0
     NELEC=(NELECS-NE+1)/2
     NLEFT=NORBS-NMOS-NELEC
  ENDIF      ! FIRST
  PRNT=(DEBUG1.OR.FINISH.AND.PRNT2)
  BIGPRT=(PRNT.AND.LARGE1)
  LAST1=(LAST1.OR.FINISH)
  do I=1,NMOS
     IN=I+NELEC
     COEFFS(1:NORBS,I)=COEFF(1:NORBS,IN)
     EIGA(I)=EIGS(IN)
  end do
  LSPIN=(LSPIN1.AND. FINISH)
  IF(BIGPRT.AND.PRNLEV.GE.2)THEN
     WRITE(6,'(''  INITIAL EIGENVALUES'')')
     WRITE(6,'(5F12.6)')(EIGA(I),I=1,NMOS)
     WRITE(6,'(//10X,''NUMBER OF ELECTRONS IN C.I. ='',F5.1)')XX
  ENDIF
  KALPHA=(NE+1)/2
  KBETA=NE-KALPHA
  IF( BIGPRT .AND. PRNLEV.GE.2 ) THEN
     WRITE(6,'(//10X,''EIGENVECTORS'',/)')
     do I=1,NORBS
        WRITE(6,'(6F12.6)')(COEFFS(I,J),J=1,NMOS)
     end do
  END IF
  XY_meci(1:NMOS,1:NMOS,1:NMOS,1:NMOS)=100.0D0
  NFA(2)=1
  NFA(1)=1
  do I=3,NMECI+1
     NFA(I)=NFA(I-1)*(I-1)
  end do
  XY_meci(1:NMOS,1:NMOS,1:NMOS,1:NMOS)=100.0D0
  do I=1,NMOS
     I1=I
     do J=1,I
        J1=J
        do K=1,NMOS
           K1=K
           do L=1,K
              L1=L
              CALL IJKL(I1,K1,J1,L1,X,COEFFS,NORBS)
           end do
        end do
     end do
  end do
  do J=1,NMOS
     do I=1,NMOS
        RJKAA(I,J)=XY_meci(I,I,J,J)-XY_meci(I,J,I,J)
        RJKAB(I,J)=XY_meci(I,I,J,J)
     end do
  end do
  do I=1,NMOS
     X=0.0D0
     do J=1,NMOS
        X=X+(RJKAA(I,J)+RJKAB(I,J))*OCCA_meci(J)
     end do
     EIGA(I)=EIGA(I)-X
     !#      IF(ABS(OCCA_meci(I)-0.5D0).LT.1.D-4)EIGA(I)=EIGA(I)+XY_meci(I,I,I,I)*0.25D0
  end do
  IF(BIGPRT.AND.PRNLEV.GE.2) THEN
     WRITE(6,150)
150  FORMAT(/,5X,'EIGENVALUES AFTER REMOVAL OF INTER-ELECTRONIC INTERACTIONS',/)
     WRITE(6,'(6F12.6)')(EIGA(I),I=1,NMOS)
     WRITE(6,'(///10X,''TWO-ELECTRON J-INTEGRALS'',/)')
     do I1=1,NMOS
        WRITE(6,'(10F10.4)')(RJKAB(I1,J1),J1=1,NMOS)
     end do
     WRITE(6,'(///10X,''TWO-ELECTRON K-INTEGRALS'',/)')
     do I1=1,NMOS
        WRITE(6,'(10F10.4)')(RJKAB(I1,J1)-RJKAA(I1,J1),J1=1,NMOS)
     end do
  ENDIF
  NATOMS=NATQM
  RJKAA(1:NMOS,1:NMOS)=0.5D0*RJKAA(1:NMOS,1:NMOS)
  IF(FIRST1) THEN
     I=INDEX(KEYWRD,'MICROS')
     IF (I .GT. 0) CALL WRNDIE(-5,'<MECI>','MICROS not supported.')
     K=KDELTA
     IF (BIGPRT.AND.PRNLEV.GE.2) WRITE(6,220) K
220  FORMAT(/////10X,11H DELTA S = ,I4)
     NUPP=KALPHA+K
     NDOWN=KBETA-K
     AMS=(NUPP-NDOWN)*0.5D0
     IF (PRNT.AND.PRNLEV.GE.2) WRITE(6,230) AMS
230  FORMAT(10X,'COMPONENT OF SPIN  = ',F4.1)
     !
     ! WRITE SOME DEBUGGING INFORMATION
     !
     !#      WRITE(6,*)'NE=',NE,' NUPP=',NUPP,' NDOWN=',NDOWN,
     !#     &' KALPHA=',KALPHA,' KBETA=',KBETA,' K=',K
     !
     if(NUPP*NDOWN.lt.0) then
        IF(PRNLEV.GE.2) WRITE(6,240)
240     FORMAT(/10X,28H IMPOSSIBLE VALUE OF DELTA S/)
        CALL WRNDIE(-5,'<MECI>','IMPOSSIBLE VALUE OF DELTA S')
     end if
     LIMA=NFA(NMOS+1)/(NFA(NUPP+1)*NFA(NMOS-NUPP+1))
     LIMB=NFA(NMOS+1)/(NFA(NDOWN+1)*NFA(NMOS-NDOWN+1))
     LAB=LIMA*LIMB
     IF (PRNT.AND.PRNLEV.GE.2) WRITE(6,260) LAB
260  FORMAT(//10X,35H NO OF CONFIGURATIONS CONSIDERED = ,I4)
     !#         IF(LAB.LT.101) GOTO 240
     !#         WRITE(6,230)
     !#  230    FORMAT(10X,24H TOO MANY CONFIGURATIONS/)
     !#         GOTO 160
     !#  240    CONTINUE
     CALL PERM(NPERMA, NUPP, NMOS, NMECI, LIMA)
     CALL PERM(NPERMB, NDOWN, NMOS, NMECI, LIMB)
     K=0
     do I=1,LIMA
        do J=1,LIMB
           K=K+1
           MICROA(1:NMOS,K)=NPERMA(1:NMOS,I)
           MICROB(1:NMOS,K)=NPERMB(1:NMOS,J)
        end do
     end do
     LIMA=LAB
     LIMB=LAB
  END IF         ! FIRST1

  GSE=0.0D0
  do I=1,NMOS
     !#         IF(ABS(OCCA_meci(I)-0.5D0).LT.0.01D0)GSE=GSE-0.25D0*XY_meci(I,I,I,I)
     GSE=GSE+EIGA(I)*OCCA_meci(I)*2.D0
     GSE=GSE+XY_meci(I,I,I,I)*OCCA_meci(I)*OCCA_meci(I)
     do J=I+1,NMOS
        GSE=GSE+2.D0*(2.D0*XY_meci(I,I,J,J) - XY_meci(I,J,I,J))*OCCA_meci(I)*OCCA_meci(J)
     end do
  end do
  !#    IF(PRNT)WRITE(6,'('' GROUND STATE ENERGY:'',F13.6,'' E.V.'')')GSE
  !     ..........
  IF( PRNT .AND. PRNLEV.GE.2 ) WRITE(6,'(//10X,''CONFIGURATIONS CONSIDERED IN C.I.'',//)')
  J=0
  do I=1,LAB
     DIAG(I)=DIAGI(MICROA(1,I),MICROB(1,I),EIGA,XY_meci,NMOS)-GSE
  end do
320 CONTINUE
  if(LAB.gt.MDIM) then
     X=-100.D0
     do I=1,LAB
        IF(DIAG(I).GT.X)THEN
           X=DIAG(I)
           J=I
        ENDIF
     end do
     IF(J.NE.LAB) THEN
        do I=J,LAB
           I1=I+1
           MICROA(1:NMOS,I)=MICROA(1:NMOS,I1)
           MICROB(1:NMOS,I)=MICROB(1:NMOS,I1)
           DIAG(I)=DIAG(I1)
        end do
     ENDIF
     LAB=LAB-1
     GOTO 320
  end if
  !
  !  MAIN LOOP TO FILL SECULAR DETERMINANT
  !
  IK=0
  do I=1,LAB
     K=0
     do J=1,NMOS
        IOCCB1(J)=MICROB(J,I)
        IOCCA1(J)=MICROA(J,I)
        K=K+IOCCA1(J)
     end do
     NALPHAO(I)=K
     IF(PRNT.AND.PRNLEV.GE.2)  THEN
        WRITE(6,'(/10X,I4,6X,10I4)') I,(IOCCA1(K),K=1,NMOS)
        WRITE(6,'(20X,10I4)')(IOCCB1(K),K=1,NMOS)
     ENDIF
     IS_meci=2
     !
     !   INNER LOOP TO FILL SECULAR DETERMINANT
     !
     do K=1,I
        IK=IK+1
        CIMAT(IK)=I*1.D-15+J*.1E-17
        IX=0
        IY=0
        ILOOP_meci=I
        JLOOP_meci=K
        do J=1,NMOS
           IOCCB2(J)=MICROB(J,K)
           IOCCA2(J)=MICROA(J,K)
           IX=IX+ABS(IOCCA1(J)-IOCCA2(J))
           IY=IY+ABS(IOCCB1(J)-IOCCB2(J))
        end do
        !
        !                              CHECK IF MATRIX ELEMENT HAS TO BE ZERO
        !
        if(IX+IY.GT.4 .OR. NALPHAO(I).NE.NALPHAO(K)) then
           continue
        else
           IF(IX+IY.EQ.4) THEN
              IF(IX.EQ.0)THEN
                 CIMAT(IK)=BABBCD(IOCCA1, IOCCB1, IOCCA2, IOCCB2, NMOS)
              ELSEIF(IX.EQ.2)THEN
                 CIMAT(IK)=AABBCD(IOCCA1, IOCCB1, IOCCA2, IOCCB2, NMOS)
              ELSE
                 CIMAT(IK)=AABACD(IOCCA1, IOCCB1, IOCCA2, IOCCB2, NMOS)
              ENDIF
           ELSEIF(IX.EQ.2)THEN
              CIMAT(IK)=AABABC(IOCCA1, IOCCB1, IOCCA2, IOCCB2, NMOS)
           ELSEIF(IY.EQ.2)THEN
              CIMAT(IK)=BABBBC(IOCCA1, IOCCB1, IOCCA2, IOCCB2, NMOS)
           ELSE
              CIMAT(IK)=DIAG(I)
              X=0.0D0
              do J=1,NMOS
                 X=X+IOCCA1(J)*IOCCB1(J)
              end do
              SPIN(I)=(-(XX-2*NALPHAO(I))*(XX-2*NALPHAO(I))+X*4.D0)
           ENDIF
        end if    ! IX+IY.GT.4 .OR. NALPHAO(I).NE.NALPHAO(K)
     end do       ! K=1,I
     ISPQR_meci(I,1)=IS_meci-1
  end do          ! I=1,LAB
  IF(BIGPRT)THEN
     IF(PRNLEV.GE.2) WRITE(6,'(//,'' C.I. MATRIX'')')
     CALL MNVECPRT(CIMAT,LAB)
  ELSE
     IF(PRNT.AND.PRNLEV.GE.2) THEN
        WRITE(6,'(//,'' DIAGONAL OF C.I. MATRIX'')')
        WRITE(6,'(5F13.6)')(CIMAT((I*(I+1))/2),I=1,LAB)
     END IF
  ENDIF
  CALL RSPQM(CIMAT,LAB,LAB,EIG,CONF)
  !
  !   DECIDE WHICH ROOT TO EXTRACT
  !
  KROOT=0
  IF(SMULT.LT.0.1D0)THEN
     MECI_RTN=EIG(LROOT)
     KROOT=LROOT
  ENDIF
  IF(BIGPRT)  THEN
     IF(PRNLEV.GE.2) WRITE(6,'(//20X,''STATE VECTORS'',//)')
     CALL MATOUT(CONF,EIG,LAB,LAB,LAB)
  ENDIF
  IF(PRNT.AND.PRNLEV.GE.2)THEN
     WRITE(6,420)
420  FORMAT(///,' STATE ENERGIES ',' EXPECTATION VALUE OF S**2  S FROM S**2=S(S+1)',//)
  ENDIF
  IROOT=0
  do I=1,LAB
     X=0.5D0*XX
     II=(I-1)*LAB
     do J=1,LAB
        JI=J+II
        X=X-CONF(JI)*CONF(JI)*SPIN(J)*0.25D0
        K=ISPQR_meci(J,1)
        if(K.ne.1) then
           do K=2,K
              LI=ISPQR_meci(J,K)+II
              X=X+CONF(JI)*CONF(LI)*2.D0
           end do
        end if
     end do
     Y=(-1.D0+SQRT(1.D0+4.D0*X))*0.5D0
     IF(ABS(SMULT-X).LT.0.01D0)THEN
        IROOT=IROOT+1
        IF(IROOT.EQ.LROOT) THEN
           IF(KROOT.EQ.0)KROOT=I
           MECI_RTN=EIG(I)
        ENDIF
     ENDIF
     J=Y*2.D0+1.5D0
     IF(PRNT.AND.PRNLEV.GE.2) WRITE(6,470) I,EIG(I),TSPIN(J),X,Y
470  FORMAT(I5,F12.6,3X,A8,F15.5,F10.5)
  end do
  !#      M=0
  !#      do I=1,NMOS
  !#         WRITE(6,*)
  !#         do J=1,NMOS
  !#            WRITE(6,*)
  !#            do K=1,NMOS
  !#               WRITE(6,'(4I2,8F12.6)')I,J,K,M,(XY_meci(I,J,K,L),L=1,NMOS)
  !#            end do
  !#         end do
  !#      end do
  IF(LAST1)THEN
     !
     !   REFORM DENSITY MATRIX
     !
     !#      WRITE(6,*)' NO. OF ELECTRONS IN C.I.:',NE
     K=(KROOT-1)*LAB
     do I=1,NMOS
        SUM=0.D0
        do J=1,LAB
           SUM=SUM+(MICROA(I,J)+MICROB(I,J))*CONF(J+K)**2
        end do
        EIGA(I)=SUM-OCCA_meci(I)*2.D0
     end do
     !#      WRITE(6,*)'  ORBITAL OCCUPANCY'
     !#      WRITE(6,'(6F12.6)')(EIGA(I),I=1,NMOS)
     L=0
     do I=1,NORBS
        do J=1,I
           SUM=0.D0
           do K=1,NMOS
              SUM=SUM+EIGA(K)*COEFFS(I,K)*COEFFS(J,K)
           end do
           L=L+1
           P(L)=P(L)+SUM
        end do
     end do
  ENDIF
  MAXVC=0
  IF(LSPIN)MAXVC=MIN(4,LAB)
  IF(LSPIN.AND.(NE/2)*2.EQ.NE.AND.PRNLEV.GE.2) THEN
     WRITE(6,'(''   ESR SPECIFIED FOR AN EVEN-ELECTRON SYSTEM'')')
  ENDIF
  COEFFS(1:NORBS,1:NMOS)=COEFFS(1:NORBS,1:NMOS)**2
  do IUJ=1,MAXVC
     IOFSET=(IUJ-1)*LAB
     IF(PRNLEV.GE.2) THEN
        WRITE(6,'(//,''      MICROSTATE CONTRIBUTIONS TO '',''STATE EIGENFUNCTION'',I3)') IUJ 
        WRITE(6,'(5F13.6)')(CONF(I+IOFSET),I=1,LAB)
     END IF
     CONF(1:LAB)=CONF(1+IOFSET:LAB+IOFSET)**2
     !                                             SECOND VECTOR!
     do I=1,NMOS
        SUM=0.D0
        do J=1,LAB
           SUM=SUM+(MICROA(I,J)-MICROB(I,J))*CONF(J)
        end do
        EIGA(I)=SUM
     end do
     IF(PRNLEV.GE.2) THEN
        WRITE(6,'(/,''    SPIN DENSITIES FROM EACH M.O., ENERGY:'',F7.3)')EIG(IUJ)
        WRITE(6,'(5F12.6)') (EIGA(I),I=1,NMOS)
        WRITE(6,*)
        WRITE(6,*)'     SPIN DENSITIES FROM EACH ATOMIC ORBITAL'
        WRITE(6,*)'                              S        PX        PY        PZ        TOTAL'
     END IF
     do I=1,NATOMS
        IL=N1ST(I)
        IU=NLAST(I)
        L=0
        SUMM=0.D0
        do K=IL,IU
           L=L+1
           SUM=0.D0
           do J=1,NMOS
              SUM=SUM+COEFFS(K,J)*EIGA(J)
           end do
           SUMM=SUMM+SUM
           EIGS(L)=SUM
        end do
        IF(PRNLEV.GE.2) THEN
           IF(L.EQ.4)THEN
              WRITE(6,'(''  ATOM'',I4,''    SPIN DENSITY  '',5F10.7)') I,(EIGS(K),K=1,L),SUMM
           ELSE
              WRITE(6,'(''  ATOM'',I4,''    SPIN DENSITY  '',F10.7,30X,F10.7)')I,EIGS(1),SUMM
           ENDIF
        END IF
     end do ! I=1,NATOMS
  end do    ! IUJ=1,MAXVC
  RETURN
END FUNCTION MECI

FUNCTION SPCG(C1,C2,C3,C4,W,WJ) result(spcg_rtn)
  !********************************************************************
  !
  !     SPCG CALCULATES THE REPULSION BETWEEN ELECTRON 1 IN MOLECULAR
  !     ORBITALS C1 AND C2 AND ELECTRON 2 IN M.O.S C3 AND C4 FOR THE
  !     VALENCE SP SHELL AT AN MNDO OR MINDO/3 LEVEL.
  !
  !                            USAGE
  !      XJ=SPCG(C(1,I),C(1,J),C(1,K),C(1,L))
  !  OR, XJ=<I(1),J(1)/K(2),L(2)>
  !
  !    ON INPUT C1    THE FIRST COLUMN MOLECULAR ORBITAL OF ELECTRON ONE.
  !             C2        SECOND
  !             C3        FIRST                                      TWO.
  !             C4        SECOND
  !
  !   ON OUTPUT SPCG   =   <C1(1)*C2(1)/C3(2)*C4(2)>
  !*********************************************************************
  !
  ! ***   MNDO OPTION   ***
  !
  use chm_kinds
  use dimens_fcm
  !
  use am1parm,only: GSS,GSP,GPP,GP2,HSP,GSD,GPD,GDD
  use sizes
  use quantm
  implicit none
  real(chm_real) spcg_rtn
  real(chm_real) C1(*),C2(*),C3(*),C4(*),W(*), WJ(*)

  INTEGER IA,IB,JA,JB,II,JJ,KK,IS,IX,IY,IZ,IS1,I,J,K,L,IMINUS,I1,J1,IZN,K1
  real(chm_real) AABABC,AABACD,BABBBC,AABBCD,BABBCD,WINT,TEMP1,DIAGI,ATEMP
  !
  SPCG_rtn=0.0
  KK=0
  iiloop: DO II=1,NATQM
     IA=N1ST(II)
     IB=NLAST(II)
     IMINUS=II-1
     jjloop: DO JJ=1,IMINUS
        JA=N1ST(JJ)
        JB=NLAST(JJ)
        i1loop: DO I=IA,IB
           j1loop: DO J=IA,I
              kloop: DO K=JA,JB
                 lloop: DO L=JA,K
                    KK=KK+1
                    WINT=W(KK)
                    SPCG_RTN=SPCG_RTN + WINT*(C1(I)*C2(J)*C3(K)*C4(L) + C1(K)*C2(L)*C3(I)*C4(J))
                    IF(I.NE.J) SPCG_RTN=SPCG_RTN + WINT*(C1(J)*C2(I)*C3(K)*C4(L) + C1(K)*C2(L)*C3(J)*C4(I))
                    IF(K.NE.L) SPCG_RTN=SPCG_RTN + WINT*(C1(I)*C2(J)*C3(L)*C4(K) + C1(L)*C2(K)*C3(I)*C4(J)) 
                    IF((I.NE.J).AND.(K.NE.L))SPCG_RTN=SPCG_RTN + WINT*( C1(J)*C2(I)*C3(L)*C4(K) &
                         +C1(L)*C2(K)*C3(J)*C4(I) )
                 enddo lloop
              enddo kloop
           enddo j1loop
        enddo i1loop
     enddo jjloop
  enddo iiloop
  !
  ATEMP=SPCG_RTN
  IS1=0
  do I1=1,NATQM
     IS1=IS1+1
     IZN=NAT(I1)
     !
     !      (SS/SS)
     !
     SPCG_RTN=SPCG_RTN+C1(IS1)*C2(IS1)*C3(IS1)*C4(IS1)*GSS(IZN)
     if(IZN.LT.3) then
        continue
     else
        IS=IS1
        IS1=IS1+1
        IX=IS1
        IS1=IS1+1
        IY=IS1
        IS1=IS1+1
        IZ=IS1
        SPCG_RTN=SPCG_RTN+GPP(IZN)*( C1(IX)*C2(IX)*C3(IX)*C4(IX)+C1(IY)*C2(IY)*C3(IY)*C4(IY) &
             +C1(IZ)*C2(IZ)*C3(IZ)*C4(IZ) )
        SPCG_RTN=SPCG_RTN+GSP(IZN)*( C1(IS)*C2(IS)*C3(IX)*C4(IX)+C1(IS)*C2(IS)*C3(IY)*C4(IY) &
             +C1(IS)*C2(IS)*C3(IZ)*C4(IZ)+C1(IX)*C2(IX)*C3(IS)*C4(IS) &
             +C1(IY)*C2(IY)*C3(IS)*C4(IS)+C1(IZ)*C2(IZ)*C3(IS)*C4(IS) )
        SPCG_RTN=SPCG_RTN+GP2(IZN)*( C1(IX)*C2(IX)*C3(IY)*C4(IY)+C1(IX)*C2(IX)*C3(IZ)*C4(IZ) &
             +C1(IY)*C2(IY)*C3(IZ)*C4(IZ)+C1(IY)*C2(IY)*C3(IX)*C4(IX) &
             +C1(IZ)*C2(IZ)*C3(IX)*C4(IX)+C1(IZ)*C2(IZ)*C3(IY)*C4(IY) )
        TEMP1=HSP(IZN)
        do J1=IX,IZ
           SPCG_RTN=SPCG_RTN+TEMP1*( C1(IS)*C2(J1)*C3(J1)*C4(IS)+C1(IS)*C2(J1)*C3(IS)*C4(J1) &
                +C1(J1)*C2(IS)*C3(IS)*C4(J1)+C1(J1)*C2(IS)*C3(J1)*C4(IS) )
        end do
        TEMP1=0.5D0*(GPP(IZN)-GP2(IZN))
        do J1=IX,IZ
           do K1=IX,IZ
              if(J1.ne.K1) SPCG_RTN=SPCG_RTN+TEMP1*( C1(J1)*C2(K1)*C3(J1)*C4(K1)+C1(J1)*C2(K1)*C3(K1)*C4(J1) )
           end do
        end do
     end if
  end do
  RETURN
END FUNCTION SPCG

SUBROUTINE MNVECPRT (A,NUMB)
  !**********************************************************************
  !
  !  VECPRT PRINTS A LOWER-HALF TRIANGLE OF A SQUARE MATRIX, THE
  !         LOWER-HALF TRIANGLE BEING STORED IN PACKED FORM IN THE
  !         ARRAY "A"
  !
  ! ON INPUT:
  !      A      = ARRAY TO BE PRINTED
  !      NUMB   = SIZE OF ARRAY TO BE PRINTED
  !(REF) NATQM  = NUMBER OF ATOMS IN THE MOLECULE (THIS IS NEEDED TO
  !               DECIDE IF AN ATOMIC ARRAY OR ATOMIC ORBITAL ARRAY IS
  !               TO BE PRINTED
  !(REF) NAT    = LIST OF ATOMIC NUMBERS
  !(REF) N1ST   = LIST OF ORBITAL COUNTERS
  !(REF) NLAST  = LIST OF ORBITAL COUNTERS
  !
  !  NONE OF THE ARGUMENTS ARE ALTERED BY THE CALL OF VECPRT
  !
  !*********************************************************************
  !
  use am1parm,only:elemnt
  use chm_kinds
  use dimens_fcm
  !
  use stream
  use sizes
  use quantm
  implicit none
  real(chm_real)  A(*)
  !
  INTEGER NATOM(MAXORB)
  CHARACTER(len=6) LINE(21)
  CHARACTER(len=2) :: ATORBS(9)=(/' S','PX','PY','PZ','X2','XZ','Z2','YZ','XY'/)
  CHARACTER(len=2) ITEXT(MAXORB), JTEXT(MAXORB)
  !
  INTEGER LIMIT,MA,NA,KK,LL,NUMB,I,J,K,L,M,N,JHI,JLO
  real(chm_real) AABABC,AABACD,BABBBC,AABBCD,BABBCD,SPCG,DIAGI
  !
  IF(NATQM.NE.0.AND.NATQM.EQ.NUMB) THEN
     !
     !    PRINT OVER ATOM COUNT
     !
     do I=1,NATQM
        ITEXT(I)='  '
        JTEXT(I)=ELEMNT(NAT(I))
        NATOM(I)=I
     end do
  ELSE
     IF (NATQM.NE.0.AND.NLAST(NATQM) .EQ. NUMB) THEN
        do I=1,NATQM
           JLO=N1ST(I)
           JHI=NLAST(I)
           L=NAT(I)
           K=0
           do J=JLO,JHI
              K=K+1
              ITEXT(J)=ATORBS(K)
              JTEXT(J)=ELEMNT(L)
              NATOM(J)=I
           end do
        end do
     ELSE
        NUMB=ABS(NUMB)
        do I=1,NUMB
           ITEXT(I) = '  '
           JTEXT(I) = '  '
           NATOM(I)=I
        end do
     ENDIF
  END IF
  NUMB=ABS(NUMB)
  LINE(1:21)='------'
  LIMIT=(NUMB*(NUMB+1))/2
  KK=8
  NA=1
60 LL=0
  M=MIN0((NUMB+1-NA),6)
  MA=2*M+1
  M=NA+M-1
  IF(PRNLEV.GE.2) THEN
     WRITE(6,100)(ITEXT(I),JTEXT(I),NATOM(I),I=NA,M)
     WRITE(6,110) (LINE(K),K=1,MA)
  END IF
  do I=NA,NUMB
     LL=LL+1
     K=(I*(I-1))/2
     L=MIN0((K+M),(K+I))
     K=K+NA
     if((KK+LL).gt.50) then   
        IF(PRNLEV.GE.2) THEN
           WRITE (6,120)
           WRITE (6,100) (ITEXT(N),JTEXT(N),NATOM(N),N=NA,M)
           WRITE (6,110) (LINE(N),N=1,MA)
        END IF
        KK=4
        LL=0
     end if
     IF(PRNLEV.GE.2) WRITE (6,130) ITEXT(I),JTEXT(I),NATOM(I),(A(N),N=K,L)
  end do
  if (L.lt.LIMIT) then
     KK=KK+LL+4
     NA=M+1
     IF ((KK+NUMB+1-NA).LE.50) GO TO 60
     KK=4
     WRITE (6,120)
     GO TO 60
  end if
  !
100 FORMAT (1H0/9X,10(2X,A2,1X,A2,I3,1X))
110 FORMAT (1H ,21A6)
120 FORMAT (1H1)
130 FORMAT (1H ,A2,1X,A2,I3,10F11.6)
  !
  RETURN
END SUBROUTINE MNVECPRT

SUBROUTINE MNFTOFHB4(F,FHB,MBT,NATQM,MQMLNK,NORBS,N1ST,NMIDLE,NLAST)
  !
  !   TRANSFORMS A FULL FOCK MATRIX IN AO BASIS INTO ACTIVE HO BASIS
  !   ON INPUT:
  !      F      FOCK MATRIX IN AO, LOWER TRIANGLE
  !      MBT    TRANSFORMATION MATRIX FOR EACH BOUNDARY ATOM, 4x4,MQMLNK   
  !      N1ST,NMIDLE,NLAST - STANDARD MOPAC ARRAY
  !      NATQM  NUMBER OF QM ATOMS
  !      MQMLNK NUMBER OF GHO BOUNDARY ATOMS
  !
  !   ON OUTPUT:
  !      FHB    FOCK MATRIX IN HO. INCLUDES ONLY ACTIVE ORBITALS
  !
  use chm_kinds
  implicit none
  INTEGER NATQM,MQMLNK,NORBS,N1ST(*),NMIDLE(*),NLAST(*)
  real(chm_real)  F(*),FHB(*),MBT(16,*)
  !
  INTEGER I,J,K,L,II,JJ,IJ,IA,IB,JA,JB,I1,IAMONE,INDF,IAL
  real(chm_real)  FTMP(10),FTMP2(10)

  logical, save :: FIRST=.TRUE.
  integer, save :: NORBAO,LIN1,NACTATM


  IF(FIRST) THEN
     NORBAO = NORBS-4*MQMLNK
     LIN1 = NORBAO*(NORBAO+1)/2
     NACTATM = NATQM-MQMLNK
     FIRST = .FALSE.
  ENDIF
  !
  !      CORE PART NOT AFFECTED
  !
  FHB(1:LIN1) = F(1:LIN1)

  !   Loop over QM-boundary atoms for orbitals to be transformed.
  DO I = 1,MQMLNK
     !   F(mu,l), AO-HO block
     !   Only one active HO per QM-boundary atom.
     I1 = NORBAO+I
     I1 = I1*(I1-1)/2
     II = NACTATM+I
     IA = N1ST(II)
     IB = NLAST(II)
     IAMONE = IA-1
     IJ = I1
     DO J = 1,NORBAO
        IJ = IJ+1
        FHB(IJ) = 0.0D0
        DO K = IA,IB
           FHB(IJ) = FHB(IJ)+MBT(K-IA+1,I)*F(J+K*(K-1)/2)
        ENDDO
     ENDDO
     !   F(l,l'), HO-other HO block
     DO J = 1,I-1
        JJ = NACTATM+J
        JA = N1ST(JJ)
        IJ = IJ+1
        FHB(IJ) = 0.0D0
        DO L = 1,4
           FTMP(L) = 0.0D0
           IAL = IA+L-1
           INDF = JA-1+IAL*(IAL-1)/2
           DO K = 1,4
              FTMP(L) = FTMP(L)+F(INDF+K)*MBT(K,J)
           ENDDO
           FHB(IJ) = FHB(IJ)+FTMP(L)*MBT(L,I)
        ENDDO
     ENDDO
     !   F(l,l), HO-HO corner block
     L = 0
     DO J = IA,IB
        JA = J*(J-1)/2
        DO K = IA,J
           L = L+1
           FTMP(L) = F(K+JA)
           FTMP2(L) = 0.0D0
        ENDDO
     ENDDO
     CALL MNVBFTN4(FTMP,FTMP2,MBT(1,I),4)
     FHB(IJ+1) = FTMP2(1)
  ENDDO
  !
  RETURN
END SUBROUTINE MNFTOFHB4

SUBROUTINE Check_working_memory_iter(NORBS,LINEAR,LPULAY,LPULY2,NORBS2)
  !
  !     Allocate/deallocate needed memory, which is used in ITER routine calls.
  !
  use chm_kinds
  use memory
  use scfblk, only : LINEAR_old,LPULAY_old,LPULY2_old,NORBS2_old, &
       AR1_keep,AR2_keep,AR3_keep,AR4_keep,   &
       BR1_keep,BR2_keep,BR3_keep,BR4_keep,   &
       POLD1_keep,POLD2_keep,POLD3_keep,      &
       PBOLD1_keep,PBOLD2_keep,PBOLD3_keep,   &
       WORK_keep,FHB_keep,CHB_keep,PAHB_keep, &
       FBHB_keep,CBHB_keep,PBHB_keep,         &
       PAPRE_keep,PBPRE_keep
  implicit none
  integer :: NORBS,LINEAR,LPULAY,LPULY2,NORBS2

  !     Allocate storage.
  !
100 continue
  if(LINEAR.gt.LINEAR_old) then
     LINEAR_old = ((NORBS+15) * ((NORBS+15) + 1)) / 2
     LPULAY_old = 6 * LINEAR_old
     NORBS2_old = (NORBS+15)*(NORBS+15)
     if(allocated(AR1_keep))  call chmdealloc('qmjunc.src','Check_working_memory_iter','AR1_keep',size(AR1_keep), &
          crl=AR1_keep)
     if(allocated(AR2_keep))  call chmdealloc('qmjunc.src','Check_working_memory_iter','AR2_keep',size(AR2_keep), &
          crl=AR2_keep)
     if(allocated(AR3_keep))  call chmdealloc('qmjunc.src','Check_working_memory_iter','AR3_keep',size(AR3_keep), &
          crl=AR3_keep)
     if(allocated(AR4_keep))  call chmdealloc('qmjunc.src','Check_working_memory_iter','AR4_keep',size(AR4_keep), &
          crl=AR4_keep)
     if(allocated(BR1_keep))  call chmdealloc('qmjunc.src','Check_working_memory_iter','BR1_keep',size(BR1_keep), &
          crl=BR1_keep)
     if(allocated(BR2_keep))  call chmdealloc('qmjunc.src','Check_working_memory_iter','BR2_keep',size(BR2_keep), &
          crl=BR2_keep)
     if(allocated(BR3_keep))  call chmdealloc('qmjunc.src','Check_working_memory_iter','BR3_keep',size(BR3_keep), &
          crl=BR3_keep)
     if(allocated(BR4_keep))  call chmdealloc('qmjunc.src','Check_working_memory_iter','BR4_keep',size(BR4_keep), &
          crl=BR4_keep)
     if(allocated(PAHB_keep)) call chmdealloc('qmjunc.src','Check_working_memory_iter','PAHB_keep',size(PAHB_keep),&
          crl=PAHB_keep)

     if(allocated(POLD1_keep))  call chmdealloc('qmjunc.src','Check_working_memory_iter','POLD1_keep', &
          size(POLD1_keep),crl=POLD1_keep)
     if(allocated(POLD2_keep))  call chmdealloc('qmjunc.src','Check_working_memory_iter','POLD2_keep', &
          size(POLD2_keep),crl=POLD2_keep)
     if(allocated(PBOLD1_keep)) call chmdealloc('qmjunc.src','Check_working_memory_iter','PBOLD1_keep', &
          size(PBOLD1_keep),crl=PBOLD1_keep)
     if(allocated(PBOLD2_keep)) call chmdealloc('qmjunc.src','Check_working_memory_iter','PBOLD2_keep', &
          size(PBOLD2_keep),crl=PBOLD2_keep)

     if(allocated(WORK_keep)) call chmdealloc('qmjunc.src','Check_working_memory_iter','WORK_keep', &
          size(WORK_keep),crl=WORK_keep)
     if(allocated(FHB_keep))  call chmdealloc('qmjunc.src','Check_working_memory_iter','FHB_keep', &
          size(FHB_keep),crl=FHB_keep)
     if(allocated(CHB_keep))  call chmdealloc('qmjunc.src','Check_working_memory_iter','CHB_keep', &
          size(CHB_keep),crl=CHB_keep)

     ! For UHF-GHO and density damping ... PJ 12/2002
     if(allocated(FBHB_keep))  call chmdealloc('qmjunc.src','Check_working_memory_iter','FBHB_keep', &
          size(FBHB_keep),crl=FBHB_keep)
     if(allocated(CBHB_keep))  call chmdealloc('qmjunc.src','Check_working_memory_iter','CBHB_keep', &
          size(CBHB_keep),crl=CBHB_keep)
     if(allocated(PBHB_keep))  call chmdealloc('qmjunc.src','Check_working_memory_iter','PBHB_keep', &
          size(PBHB_keep),crl=PBHB_keep)
     if(allocated(PAPRE_keep)) call chmdealloc('qmjunc.src','Check_working_memory_iter','PAPRE_keep', &
          size(PAPRE_keep),crl=PAPRE_keep)
     if(allocated(PBPRE_keep)) call chmdealloc('qmjunc.src','Check_working_memory_iter','PBPRE_keep', &
          size(PBPRE_keep),crl=PBPRE_keep)
  end if

  if(LPULY2.gt.LPULY2_old) then
     LPULY2_old=LPULY2+50
     if(allocated(POLD3_keep))  call chmdealloc('qmjunc.src','Check_working_memory_iter','POLD3_keep', &
          size(POLD3_keep),crl=POLD3_keep)
     if(allocated(PBOLD3_keep)) call chmdealloc('qmjunc.src','Check_working_memory_iter','PBOLD3_keep', &
          size(PBOLD3_keep),crl=PBOLD3_keep)
  end if

  !     sanity check...
  if(LINEAR_old.le.LINEAR .or. LPULAY_old.le.LPULAY .or.NORBS2_old.le.NORBS2 .or. LPULY2_old.le.LPULY2) then
     call WRNDIE(-1,'Check_working_array_iter>','Memory allocation error for scratch arrays')
     goto 100
  end if
  if(.not.allocated(AR1_keep))    call chmalloc('qmjunc.src','Check_working_memory_iter','AR1_keep',LINEAR_old, &
       crl=AR1_keep)
  if(.not.allocated(AR2_keep))    call chmalloc('qmjunc.src','Check_working_memory_iter','AR2_keep',LINEAR_old, &
       crl=AR2_keep)
  if(.not.allocated(AR3_keep))    call chmalloc('qmjunc.src','Check_working_memory_iter','AR3_keep',LINEAR_old, &
       crl=AR3_keep)
  if(.not.allocated(AR4_keep))    call chmalloc('qmjunc.src','Check_working_memory_iter','AR4_keep',LINEAR_old, &
       crl=AR4_keep)
  if(.not.allocated(BR1_keep))    call chmalloc('qmjunc.src','Check_working_memory_iter','BR1_keep',LINEAR_old, &
       crl=BR1_keep)
  if(.not.allocated(BR2_keep))    call chmalloc('qmjunc.src','Check_working_memory_iter','BR2_keep',LINEAR_old, &
       crl=BR2_keep)
  if(.not.allocated(BR3_keep))    call chmalloc('qmjunc.src','Check_working_memory_iter','BR3_keep',LINEAR_old, &
       crl=BR3_keep)
  if(.not.allocated(BR4_keep))    call chmalloc('qmjunc.src','Check_working_memory_iter','BR4_keep',LINEAR_old, &
       crl=BR4_keep)
  if(.not.allocated(POLD1_keep))  call chmalloc('qmjunc.src','Check_working_memory_iter','POLD1_keep',LPULAY_old, &
       crl=POLD1_keep)
  if(.not.allocated(POLD2_keep))  call chmalloc('qmjunc.src','Check_working_memory_iter','POLD2_keep',LPULAY_old, &
       crl=POLD2_keep)
  if(.not.allocated(POLD3_keep))  call chmalloc('qmjunc.src','Check_working_memory_iter','POLD3_keep',LPULY2_old, &
       crl=POLD3_keep)
  if(.not.allocated(PBOLD1_keep)) call chmalloc('qmjunc.src','Check_working_memory_iter','PBOLD1_keep',LPULAY_old, &
       crl=PBOLD1_keep)
  if(.not.allocated(PBOLD2_keep)) call chmalloc('qmjunc.src','Check_working_memory_iter','PBOLD2_keep',LPULAY_old, &
       crl=PBOLD2_keep)
  if(.not.allocated(PBOLD3_keep)) call chmalloc('qmjunc.src','Check_working_memory_iter','PBOLD3_keep',LPULY2_old, &
       crl=PBOLD3_keep)
  if(.not.allocated(WORK_keep))   call chmalloc('qmjunc.src','Check_working_memory_iter','WORK_keep',NORBS2_old, &
       crl=WORK_keep)
  if(.not.allocated(FHB_keep))    call chmalloc('qmjunc.src','Check_working_memory_iter','FHB_keep',NORBS2_old, &
       crl=FHB_keep)
  if(.not.allocated(CHB_keep))    call chmalloc('qmjunc.src','Check_working_memory_iter','CHB_keep',NORBS2_old, &
       crl=CHB_keep)
  if(.not.allocated(PAHB_keep))   call chmalloc('qmjunc.src','Check_working_memory_iter','PAHB_keep',LINEAR_old, &
       crl=PAHB_keep)

  if(.not.allocated(FBHB_keep))  call chmalloc('qmjunc.src','Check_working_memory_iter','FBHB_keep',NORBS2_old, &
       crl=FBHB_keep)
  if(.not.allocated(CBHB_keep))  call chmalloc('qmjunc.src','Check_working_memory_iter','CBHB_keep',NORBS2_old, &
       crl=CBHB_keep)
  if(.not.allocated(PBHB_keep))  call chmalloc('qmjunc.src','Check_working_memory_iter','PBHB_keep',LINEAR_old, &
       crl=PBHB_keep)
  if(.not.allocated(PAPRE_keep)) call chmalloc('qmjunc.src','Check_working_memory_iter','PAPRE_keep',LINEAR_old, &
       crl=PAPRE_keep)
  if(.not.allocated(PBPRE_keep)) call chmalloc('qmjunc.src','Check_working_memory_iter','PBPRE_keep',LINEAR_old, &
       crl=PBPRE_keep)
  return
end SUBROUTINE Check_working_memory_iter

#endif 
SUBROUTINE NULL_QJ
  RETURN
END SUBROUTINE NULL_QJ

