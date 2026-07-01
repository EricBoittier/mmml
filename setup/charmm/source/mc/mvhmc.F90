module mcmvhmc
#if KEY_MC==1
contains

  SUBROUTINE MVHMC(COMLYN,COMLEN,NMVATM, &
       IMVNGP,MDXP,MBONDT,QBND, &
       ARMLIM,ARMMAX,ANISO,RMDX,X,Y,Z,WMAIN)
    !
    !       Determines move specific information for Hybrid Monte Carlo moves.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use chm_types
    use dimens_fcm
    use psf
    use selctam
    use select
    use memory
    use number
    use mcmvutil, only: fillqb, imvlst
    implicit none
    !
    !       Passed Variables
    !
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    type(iptr_ptr) :: IMVNGP
    INTEGER NMVATM, MBONDT
    type(chm_ptr) :: MDXP
    real(chm_real)  RMDX, ARMMAX, X(*), Y(*), Z(*), WMAIN(*)
    LOGICAL ARMLIM, ANISO, QBND(MBONDT)

    integer,allocatable,dimension(:) :: IFP
    INTEGER I
    type(chm_iptr) :: LISTP
    CHARACTER(len=4) WRD

    !       Fill in logicals for which bonded energy terms must be calculated.
    CALL FILLQB(QBND,.FALSE.,.FALSE.,.FALSE.,.FALSE.,.FALSE.)

    !       Get the SELE...END statement that defines the moveable residues.
    call chmalloc('mvhmc.src','MVHMC','IFP',NATOM,intg=IFP)
    IFP(1:NATOM) = 0
    CALL SELCTA(COMLYN,COMLEN,IFP,X,Y,Z,WMAIN,.TRUE.)

    !       Count the number of elements that will be in the list.
    !       If there are no move locations, no point in the move, so return.
    NMVATM = 0
    DO I = 1, NATOM
       IF (IFP(I) == 1) NMVATM = NMVATM + 1
    ENDDO
    IF (NMVATM > 0) THEN

       NMVATM = 1

       !       Allocate the space
       allocate(IMVNGP%A(NMVATM))
       call chmalloc('mvhmc.src','MVHMC','MDXP',NMVATM,crlp=MDXP%A)

       CALL IMVLST(LISTP, NATOM, IFP)
       IMVNGP%A(NMVATM)%A => LISTP%A

       MDXP%A(NMVATM) = RMDX

       ARMLIM = .FALSE.

    ENDIF
    !       Free the selection array
    call chmdealloc('mvhmc.src','MVHMC','IFP',NATOM,intg=IFP)

    RETURN
  END SUBROUTINE MVHMC

  SUBROUTINE MKHMC(TEMPI,ISDMC,TKELV,NMDSTP,TIMEMD,IDX,IMVNG,ALLMV, &
       IGAMMA,QTSALL,TSEMIN,TSQ,BNBND,BIMAG,X,Y,Z &
#if KEY_MEHMC==1
       ,ISTEP,MEUPDT,CSIME,WGHTME,QMEUPD,BIASME  & 
#endif
       )
    !
    !       Applies hybrid Monte Carlo moves
    !
    !       The routine is implemented in such a way that the overhead will
    !       become prohibitively costly if only a small part of the system
    !       is moving.  However, I do not think that there is any point in
    !       implementing it in a less overhead heavy way due to the structure
    !       of the dynamics routine called.
    !
    !       Aaron R. Dinner
    !
    use new_timer,only:timer_start,timer_stop,t_dcntrl        

#if KEY_CHEQ==1
    use cheq,only: pmassq                  
#endif

    use chm_kinds
    use chm_types
    use dimens_fcm
    use block_ltm
    use consta
    use mehmc
    use number
    use psf
    use reawri
    use memory
    use stream
#if KEY_TSM==1
    use tsmh  
#endif
    use mcmvutil, only: tagatm
#if KEY_DHDGB==1
    use dhdgb,only:totals
#endif
    use dynutil, only: assvel
    implicit none

    INTEGER ISDMC, NMDSTP, IDX, ALLMV(:)
    type(chm_iptr) :: IMVNG(:)
    real(chm_real),dimension(:) :: IGAMMA
    type(nonbondDataStructure) BNBND
    type(imageDataStructure) BIMAG
    real(chm_real)  TEMPI, TKELV, TIMEMD, X(*), Y(*), Z(*)
    real(chm_real)  TSEMIN, TSQ
    LOGICAL QTSALL
#if KEY_MEHMC==1
    INTEGER ISTEP, MEUPDT
    real(chm_real)  CSIME, WGHTME, BIASME
    LOGICAL QMEUPD
#endif 
# if KEY_DHDGB==1
    real(chm_real) DUM_FHDGB1(TOTALS),DUM_FHDGB2(TOTALS)
#endif
    !
    real(chm_real),allocatable,dimension(:) :: XOLD, YOLD, ZOLD
    real(chm_real),allocatable,dimension(:) :: XNEW, YNEW, ZNEW
    real(chm_real),allocatable,dimension(:) :: VX, VY, VZ, VK
    integer,allocatable,dimension(:) :: ISKP
    INTEGER IGVOPT, NPRIV, NPRIVOLD
    INTEGER PRNTMP, HIMOVE
    real(chm_real)  JHTEMP, BETA
#if KEY_MEHMC==1
    real(chm_real)  BPOLD, BPNEW 
#endif
    LOGICAL RUNOK
    !
#if KEY_CHEQ==1
    real(chm_real),allocatable,dimension(:) :: ICGNEW, ICGOLD, IVCG
#endif 
    real(chm_real)  MCGTKE

#if KEY_TSALLIS==1
    IF (QTSALL) BETA = ONE / (KBOLTZ * TKELV)
#else /**/
    IF (QTSALL) CALL WRNDIE (-2, '<MKHMC>', &
         'TSALLIS DYNAMICS CODE NOT COMPILED')
#endif 

    !       Make the print level low to suppress massive dynamics output
    PRNTMP = PRNLEV
    PRNLEV = 0

    !       Turn on the force calculations
#if KEY_BLOCK==1
    NOFORC = .FALSE.  
#endif

    !       Set the timestep via a common variable
    DELTA = TIMEMD/TIMFAC

    call chmalloc('mvhmc.src','MKHMC','XOLD',NATOM,crl=XOLD)
    call chmalloc('mvhmc.src','MKHMC','YOLD',NATOM,crl=YOLD)
    call chmalloc('mvhmc.src','MKHMC','ZOLD',NATOM,crl=ZOLD)
    call chmalloc('mvhmc.src','MKHMC','XNEW',NATOM,crl=XNEW)
    call chmalloc('mvhmc.src','MKHMC','YNEW',NATOM,crl=YNEW)
    call chmalloc('mvhmc.src','MKHMC','ZNEW',NATOM,crl=ZNEW)
    call chmalloc('mvhmc.src','MKHMC','VX',NATOM,crl=VX)
    call chmalloc('mvhmc.src','MKHMC','VY',NATOM,crl=VY)
    call chmalloc('mvhmc.src','MKHMC','VZ',NATOM,crl=VZ)
    call chmalloc('mvhmc.src','MKHMC','VK',NATOM,crl=VK)

    !       For SHAKE
    call chmalloc('mvhmc.src','MKHMC','ISKP',NATOM,intg=ISKP)
#if KEY_CHEQ==1
    call chmalloc('mvhmc.src','MKHMC','ICGNEW',NATOM,crl=ICGNEW)
    call chmalloc('mvhmc.src','MKHMC','ICGOLD',NATOM,crl=ICGOLD)
    call chmalloc('mvhmc.src','MKHMC','IVCG',NATOM,crl=IVCG)
#endif 
    !       Write over the IMOVE array to fix atoms
    IMOVE(1:NATOM) = 1
    CALL TAGATM(IMOVE, IMVNG(IDX)%A, .TRUE.,0)

    !       Assign the velocities
    !       This also sets IGVOPT = 2
    !       IASVEL set to 1
    CALL ASSVEL(TKELV,X,Y,Z, VX, VY, VZ, &
         AMASS,ISDMC,1,IGVOPT,NATOM,IMOVE &
#if KEY_TSM==1
         ,BACKLS                                    & 
#endif
#if KEY_DHDGB==1
!AP/MF
    , .FALSE., DUM_FHDGB2, 0.0_8, 0 &
#endif
         )

#if KEY_MEHMC==1
    !       If necessary, update the bias vector and shift velocities
    CALL MEBUPD(QMEUPD,BVECMX,BVECMY,BVECMZ,VMEAVX,VMEAVY,VMEAVZ, &
         VMEOLX,VMEOLY,VMEOLZ,CSIME,MEUPDT,ISTEP,NATOM)
    IF (MEUPDT .GT. 0) THEN
       CALL ADBIAS(ISDMC,BVECMX,BVECMY,BVECMZ,NATOM, &
            VX, VY, VZ)
       BPOLD = GTBPVC(NATOM,TKELV,BVECMX,BVECMY,BVECMZ, &
            VX, VY, VZ)
       RMEHMC = WGHTME
    ENDIF
#endif 

    !       NDEGF  is set to 1   --- should only affect averages
    !       IPRFRQ is set to 999 --- should only affect averages
    NPRIV = 0
    NPRIVOLD = 0

    !       dynamc uses child timers of T_dcntrl; so T_dcntrl is started.
    CALL TIMER_START(T_DCNTRL)                           
    CALL DYNAMC(VX, VY, VZ, VK, &
         XNEW, YNEW, ZNEW, &
         XOLD, YOLD, ZOLD, &
#if KEY_CHEQ==1
         CG, ICGNEW, ICGOLD, IVCG,  & 
         pMASSQ,                                & 
#endif
#if KEY_PIPF==1
         0,0,0,0, &        
#endif
         AMASS,NPRIV,NPRIVOLD,3*NATOM-6,IGVOPT, &
         IMOVE, ISKP, 0,0, &
         NATOM,BNBND,BIMAG,1,NMDSTP,999,0, &
         JHTEMP, IGAMMA, .FALSE.,0,0,0,0, &
#if KEY_CHEQ==1
         0,                                        & 
#endif
         NDRUDE,ISDRUDE, &
         .FALSE.,RUNOK &
#if KEY_TSM==1
         ,0,0,0,0                                  & 
#endif
#if KEY_BLOCK==1
         ,.FALSE.                                  & 
#endif
#if KEY_TNPACK==1
         ,.FALSE.,.FALSE.,.FALSE.,0,0,0,0,0        & 
#endif
#if KEY_TSALLIS==1
         ,QTSALL,TSEMIN,TSQ,BETA                   & 
#endif
#if KEY_TPS==1
         ,.FALSE.,0,0,0,0,0,0,0,0,.FALSE.,.FALSE.,.FALSE. &
         ,0,0,0,0,0 &
#endif 
         )
    CALL TIMER_STOP(T_DCNTRL)                            

#if KEY_MEHMC==1
    !       Calculate the weighting factor for the acceptance criterion
    IF (MEUPDT .GT. 0) THEN
       BPNEW  = GTBPVC(NATOM,TKELV,BVECMX,BVECMY,BVECMZ, &
            VX, VY, VZ)
       BIASME = (EXP(BPNEW) + EXP(-BPNEW))/(EXP(BPOLD) + EXP(-BPOLD))
    ENDIF
#endif 

    !       Restore the IMOVE array
    IMOVE(1:NATOM) = 1
    CALL TAGATM(IMOVE,ALLMV,.TRUE.,0)

#if KEY_BLOCK==1
    NOFORC = .TRUE.  
#endif

    !       Explicitly pass back the initial kinetic energy
    TEMPI = HMCKEI

    PRNLEV = PRNTMP

#if KEY_CHEQ==1
    call chmdealloc('mvhmc.src','MKHMC','IVCG',NATOM,crl=IVCG)
    call chmdealloc('mvhmc.src','MKHMC','ICGOLD',NATOM,crl=ICGOLD)
    call chmdealloc('mvhmc.src','MKHMC','ICGNEW',NATOM,crl=ICGNEW)
#endif 
    call chmdealloc('mvhmc.src','MKHMC','ISKP',NATOM,intg=ISKP)

    call chmdealloc('mvhmc.src','MKHMC','VK',NATOM,crl=VK)
    call chmdealloc('mvhmc.src','MKHMC','VZ',NATOM,crl=VZ)
    call chmdealloc('mvhmc.src','MKHMC','VY',NATOM,crl=VY)
    call chmdealloc('mvhmc.src','MKHMC','VX',NATOM,crl=VX)
    call chmdealloc('mvhmc.src','MKHMC','ZNEW',NATOM,crl=ZNEW)
    call chmdealloc('mvhmc.src','MKHMC','YNEW',NATOM,crl=YNEW)
    call chmdealloc('mvhmc.src','MKHMC','XNEW',NATOM,crl=XNEW)
    call chmdealloc('mvhmc.src','MKHMC','ZOLD',NATOM,crl=ZOLD)
    call chmdealloc('mvhmc.src','MKHMC','YOLD',NATOM,crl=YOLD)
    call chmdealloc('mvhmc.src','MKHMC','XOLD',NATOM,crl=XOLD)

    RETURN
  END SUBROUTINE MKHMC

  SUBROUTINE HMCPAR(COMLYN,COMLEN,DELTAT,NMDSTP,LINIT &
#if KEY_MEHMC==1
       ,MEUPDT,CSIME,RUPDME  & 
#endif
       )
    !
    !       Reads special parameters associated with Hybrid
    !       Monte Carlo moves.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use exfunc
    use number
    use contrl
    use string

    implicit none

    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    INTEGER NMDSTP
    real(chm_real)  DELTAT
    LOGICAL LINIT
#if KEY_MEHMC==1
    INTEGER MEUPDT
    real(chm_real)  CSIME, RUPDME
#endif 

    IF (LINIT) THEN
       NMDSTP = 0
       DELTAT = ZERO
#if KEY_MEHMC==1
       MEUPDT = 0
       CSIME  = ZERO
       RUPDME = ZERO
#endif 
    ENDIF

    DELTAT   = GTRMF(COMLYN,COMLEN,'TIME', DELTAT)
    NMDSTP   = GTRMI(COMLYN,COMLEN,'NMDS', NMDSTP)
#if KEY_MEHMC==1
    MEUPDT   = GTRMI(COMLYN,COMLEN,'MEUP', MEUPDT)
    CSIME    = GTRMF(COMLYN,COMLEN,'MEFA', CSIME)
    RUPDME   = GTRMF(COMLYN,COMLEN,'MEWE', RUPDME)
#else /**/
    IF ((INDXA(COMLYN, COMLEN,'MEUP') .GT. 0) .OR. &
         (INDXA(COMLYN, COMLEN,'MEFA') .GT. 0) .OR. &
         (INDXA(COMLYN, COMLEN,'MEWE') .GT. 0)) THEN
       CALL WRNDIE (-2, '<HMCPAR>', 'MEHMC CODE NOT COMPILED')
    ENDIF
#endif 

    RETURN
  END SUBROUTINE HMCPAR

#if KEY_MEHMC==1
  SUBROUTINE MEINIT(QMEUPD,VMEAVX,VMEAVY,VMEAVZ,NMVTYP,NMDSTP, &
       MEUPDT,NATOM)
    !
    !       Initialize quantities for Momentum-Enhanced HMC
    !
    !       More than one type of MEHMC move will break this code.
    !
    use chm_kinds
    use number
    implicit none
    !
    INTEGER NMVTYP, NMDSTP(NMVTYP), MEUPDT(NMVTYP), NATOM
    real(chm_real)  VMEAVX(*), VMEAVY(*), VMEAVZ(*)
    LOGICAL QMEUPD
    !
    INTEGER I

    !       Use QMEUPD and see if any moves are MEHMC
    QMEUPD = .FALSE.
    DO I = 1, NMVTYP
       IF ((NMDSTP(I).LT.0).AND.(MEUPDT(I).GT.0)) THEN
          QMEUPD = .TRUE.
       ENDIF
    ENDDO

    !       If MEHMC, initialize the average vectors.
    !       The B vector will get initialized in MKHMC if QMEUPD is true.
    IF (QMEUPD) THEN
       DO I = 1, NATOM
          VMEAVX(I) = ZERO
          VMEAVY(I) = ZERO
          VMEAVZ(I) = ZERO
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE MEINIT

  SUBROUTINE MEBUPD(QMEUPD,BVECMX,BVECMY,BVECMZ,VMEAVX,VMEAVY, &
       VMEAVZ,VMEOLX,VMEOLY,VMEOLZ,CSIME,MEUPDT, &
       ISTEP,NATOM)
    !
    !       Update the B vector for MEHMC and store the old average
    !
    use chm_kinds
    implicit none
    !
    INTEGER MEUPDT, ISTEP, NATOM
    real(chm_real)  BVECMX(*), BVECMY(*), BVECMZ(*)
    real(chm_real)  VMEAVX(*), VMEAVY(*), VMEAVZ(*)
    real(chm_real)  VMEOLX(*), VMEOLY(*), VMEOLZ(*)
    real(chm_real)  CSIME
    LOGICAL QMEUPD
    !
    INTEGER I

    !       If QMEUPD was true in the previous step, update the B vector.
    !       This is done here so that a step starts and ends with the same
    !       B vector.
    !
    !       The ASSVEL routine divides the width of the distribution by the
    !       mass (smaller mass has larger velocity so same KE).  I am not
    !       sure whether it is better to bias with the average velocity or
    !       momentum.  I'll do velocity for now.

    IF (QMEUPD) THEN
       DO I = 1, NATOM
          BVECMX(I) = CSIME*VMEAVX(I)
          BVECMY(I) = CSIME*VMEAVY(I)
          BVECMZ(I) = CSIME*VMEAVZ(I)
       ENDDO
    ENDIF

    IF (MEUPDT .GT. 0) THEN

       !         Store the old average velocities in VMEOLi for rejected steps
       DO I = 1, NATOM
          VMEOLX(I) = VMEAVX(I)
          VMEOLY(I) = VMEAVY(I)
          VMEOLZ(I) = VMEAVZ(I)
       ENDDO

       QMEUPD = (MOD(ISTEP,MEUPDT) .EQ. 0)
    ELSE
       QMEUPD = .FALSE.
    ENDIF

    RETURN
  END SUBROUTINE MEBUPD

  SUBROUTINE ADBIAS(ISEED,BX,BY,BZ,NATOM,VX,VY,VZ)
    !
    !       Add bias for MEHMC.
    !
    !       Aaron R. Dinner
    !
    use clcg_mod,only:random
    use chm_kinds
    use number
    use exfunc
    implicit none

    INTEGER ISEED, NATOM
    real(chm_real)  VX(*), VY(*), VZ(*), BX(*), BY(*), BZ(*), TKELV
    !
    INTEGER I

    IF (RANDOM(ISEED) .GT. HALF) THEN
       DO I = 1, NATOM
          VX(I) = VX(I) + BX(I)
          VY(I) = VY(I) + BY(I)
          VZ(I) = VZ(I) + BZ(I)
       ENDDO
    ELSE
       DO I = 1, NATOM
          VX(I) = VX(I) - BX(I)
          VY(I) = VY(I) - BY(I)
          VZ(I) = VZ(I) - BZ(I)
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE ADBIAS

  FUNCTION GTBPVC(NATOM,TKELV,BX,BY,BZ,VX,VY,VZ)  &
       result(gtbpvc_result)
    !
    !       Get the dot product of B and V for MEHMC
    !
    use chm_kinds
    use consta
    use vector
    implicit none
    !
    real(chm_real) :: gtbpvc_result
    INTEGER NATOM
    real(chm_real)  TKELV, BX(*), BY(*), BZ(*), VX(*), VY(*), VZ(*)
    !
    GTBPVC_RESULT = DOTVEC(VX,BX,NATOM)
    GTBPVC_RESULT = GTBPVC_RESULT + DOTVEC(VY,BY,NATOM)
    GTBPVC_RESULT = GTBPVC_RESULT + DOTVEC(VZ,BZ,NATOM)
    GTBPVC_RESULT = GTBPVC_RESULT / (KBOLTZ * TKELV)
    RETURN
  END FUNCTION GTBPVC

  SUBROUTINE MERJCT(VMEAVX,VMEAVY,VMEAVZ,VMEOLX,VMEOLY,VMEOLZ, &
       NMDSTP,WGHTME,NATOM)
    !
    !         Average in zeros to the guiding vector for rejected moves.
    !
    use chm_kinds
    use number
    implicit none
    !
    INTEGER NATOM, NMDSTP
    real(chm_real)  VMEAVX(*), VMEAVY(*), VMEAVZ(*)
    real(chm_real)  VMEOLX(*), VMEOLY(*), VMEOLZ(*)
    real(chm_real)  WGHTME
    !
    INTEGER I
    real(chm_real)  FACT

    FACT = (ONE - WGHTME)**NMDSTP

    DO I = 1, NATOM

       VMEAVX(I) = VMEOLX(I)*FACT
       VMEAVY(I) = VMEOLY(I)*FACT
       VMEAVZ(I) = VMEOLZ(I)*FACT

    ENDDO

    RETURN
  END SUBROUTINE MERJCT
#endif 

  SUBROUTINE HMCARM(MVTYPE,NTRY,IDX,GAMMA,DELTAT,X,Y,Z)
    !
    !       If DMAX changed due to ARM/DOMC (as indicated by
    !       NTRY == 0), update the GAMMA arrays.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use chm_types
    use consta
    use number
    implicit none
    INTEGER MVTYPE, NTRY(*), IDX
    real(chm_real)  X(*), Y(*), Z(*), DELTAT
    type(chm_ptr) :: GAMMA
    LOGICAL QHMCF
    IF (MVTYPE .EQ. 6) THEN
       IF (NTRY(IDX) .EQ. 0) THEN
          CALL LNGFIL(0,0,GAMMA%A,ZERO,DELTAT/TIMFAC,0,0,0,0,X,Y,Z &
#if KEY_ACE==1
               ,0,0,.FALSE.         & 
#endif
               )
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE HMCARM

#endif 
end module mcmvhmc

