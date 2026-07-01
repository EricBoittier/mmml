#if KEY_OLDDYN==1
SUBROUTINE DYNAMCV(VX,VY,VZ,VK,XNEW,YNEW,ZNEW, &
     XOLD,YOLD,ZOLD,AMASS, NPRIV,NDEGF,IGVOPT, &
     IMOVE,ISKP,FREEAT,NFREAT,NATOMX,BNBND, &
     BIMAG,ISTART,ISTOP,IPRFRQ,IDYNPR,JHTEMP, &
     GAMMA,RFD,RFT,QAVER,NAVER,XAVE,YAVE,ZAVE, &
     NDRUDE,ISDRUDE, &
     QKUHEAD &
#if KEY_TSM==1
     ,REACLS,PRODLS,PIGGLS,BACKLS &    
#endif
     ,XNSE,YNSE,ZNSE &
     )
  !
  !     PERFORMS MOLECULAR DYNAMICS INTEGRATION WITH NO FRILLS SUCH AS
  !     TEMPERATURE ADJUSTMENT, LIST UPDATES, AND THE LIKE.
  !
  !     This routine handles the actual integration algorithms.  It also
  !     keeps track of averages and fluctuations during the run.
  !
  !     Authors: Everybody
  !     Overhauled by Bernie Brooks
  !     Langevin Dynamics included by Charlie Brooks and Axel Brunger
  !     Nose Dynamics included by Masa Watanabe
  !

#if KEY_CHEQ==1
  use cheq,only:qcg         
#endif

  use chm_kinds
  use chm_types
  use dimens_fcm
  use reawri
#if KEY_SCCDFTB==1
  use blockscc_fcm
  use sccdftb
#endif 
#if KEY_BLOCK==1
  use block_fcm, only : prhybh, nblock
!ldm
  use lambdam,only: qldm, ldm_init_dynam, &
       ldm_prop1_dynamcv, &
       nsavl, &
       ldm_prop2_dynamcv, &
       ldm_reset_dynam
#endif 
  use ctitla
  use contrl
  use coord
  use cvio
  use deriv
  use dynio
  use energym
  use averfluc
  use number
  use consta
  use shake
  use stream
  use tsms_mod
  use icfix
  use icpert
  use nose_mod
#if KEY_FLUCQ==1
  use flucq     
#endif
  use phmd
  use holonom,only:holonoma
#if KEY_DYNVV2==1
  use dynvv2, only:optimizedrude     
#endif
#if KEY_PERT==1
  use pert
  use pshake
#endif 
#if KEY_ADUMB==1
  use umb       
#endif
#if KEY_GAMUS==1
  use gamusmodule 
#endif
  ! FBS MOD ********************************************
  use dmcons
  use rgym
  ! END FBS MOD ****************************************
#if KEY_PARALLEL==1
  use parallel,only:lnnod     
#endif
#if KEY_SQUANTM==1
  use squantm, only: LTRBOMD,q_apply_tr_bomd,i_md_step,i_md_step_old  
#endif
  use machutil,only:die
  use heurist,only:updeci
  use prssre
!QC_Ito_UW0616
#if KEY_RXNCOR==1
  use rxncom, only: UMBMDSTEP
#endif

  implicit none
  !
#if KEY_SCCDFTB==1
  real(chm_real) dvdltmp    
#endif
  real(chm_real) VX(*),VY(*),VZ(*),VK(*),XNEW(*),YNEW(*),ZNEW(*)
  real(chm_real) XOLD(*),YOLD(*),ZOLD(*)
  real(chm_real) AMASS(*)
  INTEGER NPRIV,NDEGF,IGVOPT
  INTEGER IMOVE(*),ISKP(*)
  INTEGER FREEAT(*),NFREAT,NATOMX
  !!      INTEGER BNBND(*),BIMAG(*)
  type(nonbondDataStructure) BNBND
  type(imageDataStructure) BIMAG

  INTEGER ISTART,ISTOP,IPRFRQ,IDYNPR
  real(chm_real) GAMMA(*),RFD(*),RFT(*)
  INTEGER NDRUDE
  LOGICAL ISDRUDE(*)
  LOGICAL QAVER,QKUHEAD
  INTEGER NAVER
  real(chm_real) XAVE(*),YAVE(*),ZAVE(*)
  real(chm_real) JHTEMP
#if KEY_TSM==1
  INTEGER REACLS(*),PRODLS(*),PIGGLS(*),BACKLS(*)
#endif 
  real(chm_real) XNSE(*),YNSE(*),ZNSE(*)
  !
  real(chm_real) DELTA2,FACT,DNUM,DNM1,DNP1,DNPM1
  real(chm_real) DELTAS,TIME,TEMPI,AKMATI,ALPHA,TOTEPR
  INTEGER NATOM,IST1,ISTEP,ISTPSA,NUMSTP,I,K
  LOGICAL OK
  !
  !
  INTEGER J,NIT
  real(chm_real) RRS,QK1,QK2,QV1
  real(chm_real)  SV0(MAXNOS),SNM0(MAXNOS)
  real(chm_real)  TMPN(MAXNOS),EPTKE1(MAXNOS),EPTKX(MAXNOS)
  real(chm_real)  EPT01,SAX,QQK,SSL,SA1X,SA2X
  !
  INTEGER FSTEP,SSTEP,AVG,FLUC
  PARAMETER(FSTEP=0,SSTEP=1,AVG=2,FLUC=3)
  !
  real(chm_real) FLUCTD,RVAL
  !
  !
  !     TIMFAC is the conversion factor from AKMA time to picoseconds.
  !
#if KEY_TSM==1
  real(chm_real) PUMEP,RLAMBDA,PLAMBDA,ORLAMBDA,OPLAMBDA
#endif 
  LOGICAL QICP, QOK
  DATA TOTEPR/-9999.0D0/

  !mw...02-Aug-95, Masa Watanabe and Stefan Boresch to fix NOSE restart
  !                problem on HP
  SAVE SV0
  !mw...
#if KEY_PARALLEL==1
  IF (LNNOD() > 1) THEN
     CALL WRNDIE(-4,'<DYNAMCV>','Old integrator is not parallel.')
  ENDIF
#endif 
  !
#if KEY_TSM==1
  IF (QTSM) THEN
     RLAMBDA = (ONE-LAMBDA)**LPOWER
     PLAMBDA = LAMBDA**LPOWER
  ENDIF
  QICP=(IICP > 0)
#endif 
  !
  NATOM=NATOMX
  !
#if KEY_MNDO97==1
  ! namkh 09/12/03 for MNDO97 printing setup
  ISTEPQM = 0
#endif 
  !
  IF (NDRUDE > 0) THEN
     !        Put back the Drude masses on their heavy atoms.  (The motion of
     !        the Drude particles will be solved iteratively, by calling
     !        subroutine "OptimizeDrude" before every call to "ENERGY".)
     !        The IMOVE variable of the Drude particles is temporarily set to
     !        -1 so that the integration code will ignore them.
     DO I=1,NATOM
        IF (ISDRUDE(I) .AND. AMASS(I) > ZERO) THEN
           IF (PRNLEV > 5) THEN
              WRITE(OUTU,*) 'MASS',AMASS(I),' :',I,' ->',I-1
           ENDIF
           AMASS(I-1) = AMASS(I-1) + AMASS(I)
           AMASS(I) = ZERO
           IF (IMOVE(I-1) == -1) AMASS(I-1) = ZERO
        ENDIF
        IF (ISDRUDE(I) .AND. IMOVE(I) == 0) IMOVE(I) = -1
     ENDDO
  ENDIF
  !
#if KEY_CHEQ==1
  !  Check to see if Charge Equilibration is on
  IF(QCG) THEN
     WRITE(OUTU,*) " DYNAMCV:  CHEQ not supported with ORIG integrator"
     CALL WRNDIE(-2,'<DYNAMCV>', &
          'CHEQ NOT YET SUPPORTED w/ORIG')
  ENDIF
#endif 


  DELTAS=HALF*DELTA*DELTA
  DELTA2=DELTA*DELTA
  IST1=ISTART-1
  !
  IF(QAVER) THEN
     IF(NAVER == 0) THEN
        DO I=1,NATOM
           XAVE(I)=ZERO
           YAVE(I)=ZERO
           ZAVE(I)=ZERO
        ENDDO
     ENDIF
  ENDIF
  !
#if KEY_BLOCK==1 /*ldm*/
  !:    set BLCOEF = LAMBDA**2 before update the lambdas or call energy
  !:    lambda(1) == 1 by the design see block command
  if(qldm) call ldm_init_dynam(nblock) 
#endif /*  LDM*/
  !
  IF(IGVOPT >= 3 .AND. (IDYNPR == 0.OR.JHSTRT.EQ.0)) THEN
     !
     DO K=1,NATOM
        XNEW(K)=X(K)
        YNEW(K)=Y(K)
        ZNEW(K)=Z(K)
        X(K)=XOLD(K)
        Y(K)=YOLD(K)
        Z(K)=ZOLD(K)
     ENDDO
#if KEY_TSM==1
     IF(QTSM.AND.PIGSET) THEN
        CALL PIGCVSET(X,Y,Z)
        CALL PIGCVSET(XNEW,YNEW,ZNEW)
        CALL PIGCVSET(XOLD,YOLD,ZOLD)
     ENDIF
#endif 
     !
     !        Get previous energy for printing (just as a check)
#if KEY_DYNVV2==1
     IF (NDRUDE > 0) THEN
        CALL OptimizeDrude( TENM5,500, &
             XOLD,YOLD,ZOLD, VX,VY,VZ, BNBND,BIMAG, &
             DX,DY,DZ, .TRUE. )
     ENDIF
#endif 
     QDYNCALL=.TRUE.
     CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
     QDYNCALL=.FALSE.
     DO K=1,NATOM
        X(K)=XNEW(K)
        Y(K)=YNEW(K)
        Z(K)=ZNEW(K)
     ENDDO
     !
  ENDIF
  !
  IF(IGVOPT == 2) THEN
#if KEY_TSM==1
     IF(QTSM.AND.PIGSET) CALL PIGCVSET(XOLD,YOLD,ZOLD)
#endif 
     !
     !        This is the 2-step VERLET
     DO K=1,NATOM
        X(K)=XOLD(K)
        Y(K)=YOLD(K)
        Z(K)=ZOLD(K)
     ENDDO

     IF(QHOLO) THEN
        !     Do shake just in case coordinates don't fit constraints
        CALL HOLONOMA(X,Y,Z,XOLD,YOLD,ZOLD,.TRUE.,.TRUE.,QOK)
     ENDIF
     !
#if KEY_DYNVV2==1
     IF (NDRUDE > 0) THEN
        CALL OptimizeDrude( TENM5,500, &
             XOLD,YOLD,ZOLD, VX,VY,VZ, BNBND,BIMAG, &
             DX,DY,DZ, .TRUE. )
     ENDIF
#endif 
     CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
     !
     !
     IGVOPT=3
     DO I=1,NATOM
        IF (IMOVE(I) == 0) THEN
           FACT=DELTAS/AMASS(I)
           XOLD(I)=X(I)
           X(I)=X(I)+VX(I)*DELTA-DX(I)*FACT
           YOLD(I)=Y(I)
           Y(I)=Y(I)+VY(I)*DELTA-DY(I)*FACT
           ZOLD(I)=Z(I)
           Z(I)=Z(I)+VZ(I)*DELTA-DZ(I)*FACT
        ENDIF
     ENDDO
     !
#if KEY_BLOCK==1 /*ldm*/
     if(qldm) call ldm_prop1_dynamcv(nblock, delta)
#endif /*  LDM*/
     IF(QHOLO) THEN
        CALL HOLONOMA(X,Y,Z,XOLD,YOLD,ZOLD,.TRUE.,.TRUE.,QOK)
     ENDIF
     !
  ENDIF
  !
  IF(ISTOP < ISTART) CALL DIE
  IF(MOD(IST1,IPRFRQ) == 0) THEN
     call avfl_reset()
     FITA = ZERO
  ENDIF
  !
  IF (IDYNPR == 0 .OR. JHSTRT.EQ.0) THEN
     !      WRITE OUT INITIAL CONDITIONS
     ISTEP=ISTART-1
     TIME=TIMFAC*DELTA*NPRIV
     EPROP(TOTKE)=ZERO
#if KEY_TSM==1
     !   zero back atom velocities so that they are not counted in the virial
     IF(QTSM.AND.PIGSET) CALL BACK0(VX,VY,VZ)
#endif 
     DO I=1,NATOM
        IF(IMOVE(I) == 0) THEN
           TEMPI = VX(I)**2 + VY(I)**2 + VZ(I)**2
           TEMPI=AMASS(I)*TEMPI
           EPROP(TOTKE)=EPROP(TOTKE)+TEMPI
        ENDIF
     ENDDO
     !
     ! . Calculate the RMS gradient and work.
     RVAL = ZERO
     DO I=1,NATOM
        RVAL=RVAL+DX(I)*DX(I)+DY(I)*DY(I)+DZ(I)*DZ(I)
     ENDDO
     EPROP(GRMS) = SQRT(RVAL/NDEGF)
     !
     ! . Calculate the pressures.
     CALL PRSEXT
     CALL PRSINT(NATOM, AMASS, VX, VY, VZ, ONE, VX, VX, VX, ZERO)
     ! APH: Note VX, VX, VX here ------------------|
     !      are just dummies
     !
     QK2=ZERO
     QV1=ZERO
     IF(QNOSE) THEN
        DO I=1,NOBL
           IF (SQM(I) == ZERO) SQM(I) = RBIG
           QK1=SQM(I)*SV0(I)
           QK2=QK2+QK1*QK1*0.5/SQM(I)
           QV1=QV1+KBOLTZ*NDGN(I)*RTMPR(I)*SN12(I)
        ENDDO
     ENDIF
     !
     EPROP(TOTKE)=EPROP(TOTKE)/TWO
     EPROP(TOTE)=EPROP(EPOT)+EPROP(TOTKE)+QK2+QV1
     IF(QNOSE) THEN
        EPROP(HFCTE)=EPROP(EPOT)+EPROP(TOTKE)
        EPROP(EHFC)=QK2+QV1
     ELSE
        EPROP(HFCTE)=EPROP(TOTE)
        EPROP(EHFC)=ZERO
     ENDIF
     EPROP(TEMPS)=TWO*EPROP(TOTKE)/(KBOLTZ*NDEGF)
#if KEY_FLUCQ==1
     ! Calculate charge velocities, and the FlucQ contribution to energy
     IF (QFLUC) CALL FQCNEW(0,NPRINT,1,NATOM)
#endif 

#if KEY_PHMD==1
     IF (QPHMD) THEN
        CALL UpdatePHMD(0, 0, IStep, 0)
     ENDIF
#endif 
     !
     IF(PRNLEV >= 2) CALL PRINTE(OUTU, EPROP, ETERM, 'DYNA', &
          'DYN', .TRUE.,ISTEP, TIME, ZERO, .TRUE.)
  ENDIF

!  if(LTRBOMD) q_apply_tr_bomd=.true.   ! for TR-BOMD, start applying TR-BOMD
  !
  !      INITIALIZE ACCUM VARIABLES
  IF(JHSTRT == 0) THEN
     DO I=1,NATOM
        VK(I)=ZERO
     ENDDO
     JHTEMP=ZERO
     FITP=ZERO
     call avfl_reset_lt()
  ENDIF
  !
  !     THIS IS THE MAIN LOOP FOR DYNAMICS.
  !
  DO ISTEP=ISTART,ISTOP
!QC_Ito_UW0616
#if KEY_RXNCOR==1
  UMBMDSTEP=ISTEP
#endif

     !
#if KEY_MNDO97==1
     ! namkh 09/12/03 for MNDO97 printing setup
     ISTEPQM=ISTEP
#endif 
#if KEY_SQUANTM==1
         if(LTRBOMD) then
            i_md_step=i_md_step+1         ! for TR-BOMD
            i_md_step_old=i_md_step_old+1
         end if
#endif 
     JHSTRT=JHSTRT+1
     NPRIV=NPRIV+1
     ! FBS MOD ************************************
#if KEY_DMCONS==1
     DSTEP = NPRIV
#endif 
#if KEY_RGYCONS==1
     DSTEPRG = NPRIV
#endif 
     ! END FBS MOD ********************************
     !
#if KEY_FLUCQ==1
     ! Compute new charges
     IF (QFLUC) CALL FQCHPR(ISTEP)
#endif 
#if KEY_PHMD==1
     IF (QPHMD) THEN
        CALL UpdatePHMD(1, 1, IStep, 0)
     ENDIF
#endif 
     IF(QAVER) THEN
        NAVER=NAVER+1
        DO I=1,NATOM
           XAVE(I)=XAVE(I)+X(I)
           YAVE(I)=YAVE(I)+Y(I)
           ZAVE(I)=ZAVE(I)+Z(I)
        ENDDO
     ENDIF
     !
     IF(NSAVC > 0) THEN
        IF (MOD(ISTEP,NSAVC) == 0) THEN
           IF(QAVER) THEN
              DNUM=ONE
              DNUM=DNUM/NAVER
              DO I=1,NATOM
                 XAVE(I)=XAVE(I)*DNUM
                 YAVE(I)=YAVE(I)*DNUM
                 ZAVE(I)=ZAVE(I)*DNUM
              ENDDO
#if KEY_TSM==1
              IF(QTSM.AND.PIGSET) CALL PIGCVSET(XAVE,YAVE,ZAVE)
#endif 
              CALL WRITCV(XAVE,YAVE,ZAVE, &
#if KEY_CHEQ==1
                   (/ ZERO /), .FALSE., &  
#endif
                   NATOM,FREEAT,NFREAT,NPRIV, &
                   ISTEP,NDEGF,DELTA,NSAVC,NSTEP,TITLEA, &
                   NTITLA,IUNCRD,.FALSE.,.FALSE., (/ 0 /), &
                   .FALSE., (/ ZERO /))
              NAVER=0
              DO I=1,NATOM
                 XAVE(I)=ZERO
                 YAVE(I)=ZERO
                 ZAVE(I)=ZERO
              ENDDO
           ELSE
#if KEY_TSM==1
              IF(QTSM.AND.PIGSET) CALL PIGCVSET(X,Y,Z)
#endif 
              CALL WRITCV(X,Y,Z, &
#if KEY_CHEQ==1
                   (/ ZERO /), .FALSE., &  
#endif
                   NATOM,FREEAT,NFREAT,NPRIV,ISTEP, &
                   NDEGF,DELTA,NSAVC,NSTEP,TITLEA,NTITLA, &
                   IUNCRD,.FALSE.,.FALSE., (/ 0 /), .FALSE., (/ ZERO /))
           ENDIF
        ENDIF
     ENDIF
#if KEY_BLOCK==1 /*ldm*/
     if(nsavl > 0 .and. iolev.gt.0) then
        if(mod(istep,nsavl) == 0) then
           call writld(nblock,npriv, &
                istep,nstep, &
                delta)
        endif
     endif
#endif 
     !
     ! Do list updates if appropriate (Image centering within dynamics
     ! doesn't work with the old integrator).
     IF (NDRUDE > 0) THEN
        !           For the list update to work properly, the Drude particles
        !           cannot have IMOVE = -1.
        DO I=1,NATOM
           IF (ISDRUDE(I) .AND. IMOVE(I) == -1) IMOVE(I) = 0
        ENDDO
     ENDIF
     CALL UPDECI(ISTEP-1,X,Y,Z,WMAIN, &
          -1,XOLD,YOLD,ZOLD,VX,VY,VZ)
     IF (NDRUDE > 0) THEN
        DO I=1,NATOM
           IF (ISDRUDE(I) .AND. IMOVE(I) == 0) IMOVE(I) = -1
        ENDDO
     ENDIF
     !
#if KEY_DYNVV2==1
     IF (NDRUDE > 0) THEN
        CALL OptimizeDrude( TENM5,500, &
             XOLD,YOLD,ZOLD, VX,VY,VZ, BNBND,BIMAG, &
             DX,DY,DZ, .TRUE. )
     ENDIF
#endif 
     CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
#if KEY_TSM==1
     IF(QTSM.AND.PIGSET) CALL PIGFSET(DX,DY,DZ)
#endif 
     IF(ILANG == 1) CALL DLNGVV(RFD,NATOM,RFT,DX,DY,DZ,ISEED)
     !
     IF(ILANG == 1) THEN
        DO I=1,NATOM
           IF (IMOVE(I)  ==  0) THEN
              FACT=DELTA2/AMASS(I)
              ALPHA=ONE/(ONE+GAMMA(I))
              XNEW(I)=(X(I)+X(I)-XOLD(I)-DX(I)*FACT+ &
                   GAMMA(I)*XOLD(I))*ALPHA
              YNEW(I)=(Y(I)+Y(I)-YOLD(I)-DY(I)*FACT+ &
                   GAMMA(I)*YOLD(I))*ALPHA
              ZNEW(I)=(Z(I)+Z(I)-ZOLD(I)-DZ(I)*FACT+ &
                   GAMMA(I)*ZOLD(I))*ALPHA
           ENDIF
        ENDDO
     ELSE
        DO I=1,NATOM
           IF (IMOVE(I) == 0) THEN
              FACT=DELTA2/AMASS(I)
              XNEW(I)=X(I)+X(I)-XOLD(I)-DX(I)*FACT
              YNEW(I)=Y(I)+Y(I)-YOLD(I)-DY(I)*FACT
              ZNEW(I)=Z(I)+Z(I)-ZOLD(I)-DZ(I)*FACT
              XNSE(I)=XNEW(I)
              YNSE(I)=YNEW(I)
              ZNSE(I)=ZNEW(I)
           ENDIF
        ENDDO
     ENDIF
     !
#if KEY_BLOCK==1 /*ldm*/
     if(qldm) call ldm_prop2_dynamcv(nblock, delta)
#endif /*  LDM*/
     !sb      Have to change Masa's code here; need this call to SHAKE below
     !         IF(QHOLO.AND.(.NOT.QNOSE)) THEN
     IF (QHOLO) THEN
        !             DO I=1,NATOM
        !             XXI(I) = XNEW(I)
        !             YYI(I) = YNEW(I)
        !             ZZI(I) = ZNEW(I)
        !             ENDDO
        CALL HOLONOMA(XNEW,YNEW,ZNEW,X,Y,Z,.TRUE.,.TRUE.,QOK)
#if KEY_PERT==1
        IF (QPSHAKE) CALL EPCONS(DELTA2)
#endif 
        !            DO I=1,NATOM
        !            FACT=-AMASS(I)/DELTA2
        !            VX(I)=(XNEW(I)-XXI(I))*FACT
        !            VY(I)=(YNEW(I)-YYI(I))*FACT
        !            VZ(I)=(ZNEW(I)-ZZI(I))*FACT
        !            ENDDO
        !            CALL VIRSHK(EPRESS(VIXX:VIZZ),NATOM,X,Y,Z,VX,VY,VZ)
     ENDIF
     FACT=TWO*DELTA
     DO I=1,NATOM
        IF (IMOVE(I) <= 0) THEN
           VX(I)=(XNEW(I)-XOLD(I))/FACT
           VY(I)=(YNEW(I)-YOLD(I))/FACT
           VZ(I)=(ZNEW(I)-ZOLD(I))/FACT
        ENDIF
     ENDDO
     !
#if KEY_TSM==1
     IF(QTSM.AND.PIGSET) THEN
        CALL PIGCVSET(XNEW,YNEW,ZNEW)
        CALL PIGCVSET(X,Y,Z)
        CALL PIGCVSET(XOLD,YOLD,ZOLD)
     ENDIF
#endif 
     !
     EPROP(TOTKE)=ZERO
#if KEY_TSM==1
     IF (QTSM.AND.PIGSET) CALL BACK0(VX,VY,VZ)
#endif 
     IF(QNOSE) THEN
        DO I=1,NOBL
           EPTKX(I)=ZERO
        ENDDO
     ENDIF
     !
     DO I=1,NATOM
        IF(IMOVE(I) == 0) THEN
           TEMPI = VX(I)**2 + VY(I)**2 + VZ(I)**2
           TEMPI=AMASS(I)*TEMPI
           VK(I)=VK(I)+TEMPI
           EPROP(TOTKE)=EPROP(TOTKE)+TEMPI
           IF(QNOSE) THEN
              J = INLCKP(I)
              EPTKX(J)=EPTKX(J)+TEMPI
           ENDIF
        ENDIF
     ENDDO
     !
     ! NOSE DYNAMICS FOR CONSTANT TEMPERATURE
     !
     IF(QNOSE) THEN
        DO I = 1,NOBL
           SV0(I)=(3.0*SN11(I)-4.0*SN12(I)+SN13(I))/(2.0*DELTA)
        ENDDO
        !
        NIT = 0
11111   CONTINUE
        NIT = NIT + 1
        DO I=1,NOBL
           SA1X=EPTKX(I)-NDGN(I)*KBOLTZ*RTMPR(I)
           SAX = SA1X/SQM(I)
           SNM0(I)=-SN12(I)+2.0*SN11(I)+DELTA2*SAX
        ENDDO
        !
        DO I = 1, NATOM
           IF (IMOVE(I)  ==  0) THEN
              J = INLCKP(I)
              RRS=SV0(J)
              XNEW(I)=XNSE(I)-RRS*DELTA2*VX(I)
              YNEW(I)=YNSE(I)-RRS*DELTA2*VY(I)
              ZNEW(I)=ZNSE(I)-RRS*DELTA2*VZ(I)
           ENDIF
        ENDDO
        IF(QHOLO) THEN
           !          DO I=1,NATOM
           !          XXI(I) = XNEW(I)
           !          YYI(I) = YNEW(I)
           !          ZZI(I) = ZNEW(I)
           !          ENDDO
           CALL HOLONOMA(XNEW,YNEW,ZNEW,X,Y,Z,.TRUE.,.TRUE.,QOK)
           !          DO I=1,NATOM
           !          FACT=-AMASS(I)/DELTA2
           !          VX(I)=(XNEW(I)-XXI(I))*FACT
           !          VY(I)=(YNEW(I)-YYI(I))*FACT
           !          VZ(I)=(ZNEW(I)-ZZI(I))*FACT
           !          ENDDO
           !         CALL VIRSHK(EPRESS(VIXX:VIZZ),NATOM,X,Y,Z,VX,VY,VZ)
        ENDIF
        !
        FACT=TWO*DELTA
        DO I=1,NATOM
           IF (IMOVE(I) <= 0) THEN
              VX(I)=(XNEW(I)-XOLD(I))/FACT
              VY(I)=(YNEW(I)-YOLD(I))/FACT
              VZ(I)=(ZNEW(I)-ZOLD(I))/FACT
           ENDIF
        ENDDO
        !
        DO I=1,NOBL
           EPTKE1(I) = EPTKX(I)
           EPTKX(I) = ZERO
        ENDDO
        !
        EPROP(TOTKE)=ZERO
        !
        DO I=1,NATOM
           IF(IMOVE(I) == 0) THEN
              TEMPI = VX(I)**2 + VY(I)**2 + VZ(I)**2
              TEMPI=AMASS(I)*TEMPI
              VK(I)=VK(I)+TEMPI
              EPROP(TOTKE)=EPROP(TOTKE)+TEMPI
              J = INLCKP(I)
              EPTKX(J)=EPTKX(J)+TEMPI
           ENDIF
        ENDDO
        !
        DO I = 1,NOBL
           SV0(I)=0.5*(SNM0(I) - SN12(I))/DELTA
        ENDDO
        !
        !   CHECK CONVERGENCE FOR NOSE (AND ANDERSEN) METHOD
        EPT01=ZERO
        DO I=1,NOBL
           EPT01=EPT01+DABS(EPTKE1(I)-EPTKX(I))
        ENDDO
        EPT01=EPT01/NOBL
        IF(EPT01 <= TOLNS) GO TO 22211
        !
        IF(NIT <= NSCYC) GO TO 11111
22211   CONTINUE
        !
        ! end of convergence check
        !
        DO I = 1,NOBL
           SN13(I) = SN12(I)
           SN12(I) = SN11(I)
           SN11(I) = SNM0(I)
        ENDDO
     ENDIF
     !
     ! . Calculate the RMS gradient and work.
     RVAL = ZERO
     DO I=1,NATOM
        RVAL=RVAL+DX(I)*DX(I)+DY(I)*DY(I)+DZ(I)*DZ(I)
     ENDDO
     EPROP(GRMS) = SQRT(RVAL/NDEGF)
     !
     ! . Calculate the pressures.
     CALL PRSEXT
     CALL PRSINT(NATOM, AMASS, VX, VY, VZ, ONE, VX, VX, VX, ZERO)
     ! APH: Note VX, VX, VX here ------------------|
     !      are just dummies
     !
     !   CHECKING SCALING PARAMETER
     !
     IF(QNOSE) THEN
        SSL=ZERO
        QK2=ZERO
        DO I = 1, NOBL
           QK1 = SQM(I) * SV0(I)
           QQK = QK1 * QK1 * 0.5 / SQM(I)
           QK2 = QK2 + QQK
           SSL = SSL + RTMPR(I)*NDGN(I)*SN12(I)
        ENDDO
        QV1=KBOLTZ*SSL
     ELSE
        QK2 = ZERO
        QV1 = ZERO
     ENDIF

     !
     EPROP(TOTKE)=EPROP(TOTKE)/TWO
     EPROP(TOTE)=EPROP(EPOT)+EPROP(TOTKE)+QK2+QV1
     !
     IF(QNOSE) THEN
        EPROP(HFCTE)=EPROP(EPOT)+EPROP(TOTKE)
        EPROP(EHFC)=QK2+QV1
     ELSE
        EPROP(HFCTE)=EPROP(TOTE)
        EPROP(EHFC)=ZERO
     ENDIF
     !
#if KEY_FLUCQ==1
     ! Calculate charge velocities, and the FlucQ contribution to energy
     IF (QFLUC) CALL FQCNEW(ISTEP,NPRINT,1,NATOM)
#endif 
     !
#if KEY_PERT==1
     !sb      Now add everything up for PERT/SHAKE
     IF (QPERT.AND.QACCUM) CALL EPSUM
#endif 
     !
     !        CHECK THE CHANGE IN TOTAL ENERGY TO BE SURE EVERYTHING IS OK.
     !
     IF ((.NOT.QNOSE).AND.(JHSTRT > 2)) THEN
        IF (ABS(TOTEPR-EPROP(TOTE)) >  &
             MAX(ECHECK,0.1D0*EPROP(TOTKE))) THEN
           IF (TOTEPR  /=  -9999.0) THEN
              IF(WRNLEV >= 2) WRITE (OUTU,2000) &
                   ECHECK,TOTEPR,EPROP(TOTE),EPROP(TOTKE)
2000          FORMAT(' TOTAL ENERGY CHANGE EXCEDED'/G12.2, &
                   ' KCAL AND 10% OF THE TOTAL KINETIC ENERGY', &
                   ' IN THE LAST STEP'/' PREVIOUS EPROP(EPOT) =',G14.4, &
                   ' CURRENT EPROP(EPOT) =',G14.4,' KINETIC =',G14.4)
              CALL WRNDIE(-2,'<DYNAMC>', &
                   'ENERGY CHANGE TOLERENCE EXCEEDED')
           ENDIF
        ENDIF
     ENDIF
     TOTEPR=EPROP(TOTE)
     TIME=TIMFAC*DELTA*NPRIV
     AKMATI=DELTA*NPRIV
     EPROP(TEMPS)=TWO*EPROP(TOTKE)/(KBOLTZ*NDEGF)
     !
     IF(QNOSE) THEN
        DO I=1,NOBL
           TMPN(I)=EPTKX(I)/(KBOLTZ*NDGN(I))
        ENDDO
     ENDIF
     !
     JHTEMP=JHTEMP+EPROP(TEMPS)
     call avfl_update(EPROP, ETERM, EPRESS)
     ISTPSA=MOD(IST1,IPRFRQ)+ISTEP-IST1
     FITA=FITA+ISTPSA*EPROP(TOTE)
     FITP=FITP+JHSTRT*EPROP(TOTE)
     !
     IF(NSAVV > 0) THEN
        IF(MOD(ISTEP,NSAVV) == 0) THEN
#if KEY_TSM==1
           IF(QTSM.AND.PIGSET) CALL BACK0(VX,VY,VZ)
#endif 
           CALL WRITCV(VX,VY,VZ, &
#if KEY_CHEQ==1
                (/ ZERO /), .FALSE., &  
#endif
                NATOM,FREEAT,NFREAT,NPRIV,ISTEP, &
                NDEGF,DELTA,NSAVV,NSTEP,TITLEA,NTITLA, &
                IUNVEL,.TRUE.,.FALSE., (/ 0 /), .FALSE., (/ ZERO /))
        ENDIF
     ENDIF
     !
     IF((IUNOS >= 0).AND.QNOSE) THEN
        IF(MOD(ISTEP,NSNOS) == 0) THEN
           WRITE(IUNOS,166) NPRIV,TIME,EPROP(TOTE)
           WRITE(IUNOS,167) (SN11(I),TMPN(I),I=1,NOBL)
167        FORMAT(2F16.6)
166        FORMAT(I16,2F16.6)
           CALL GFLUSH(IUNOS)
        ENDIF
     ENDIF
     !
     ! . Adaptive umbrella sampling, write out umbrella coordinate
#if KEY_ADUMB==1
     IF((NSAVC > 0).AND.(WCUNUM.GT.0)) THEN
        IF (MOD(ISTEP,NSAVC) == 0) THEN
           CALL UMWRUC(ISTEP)
        ENDIF
     ENDIF
#endif 
      !GAMUS, write out reaction coordinates
#if KEY_GAMUS==1
      IF((DOGAMUS).AND.(IGAMUSW.GT.0))THEN
         IF(MOD(ISTEP,IGAMUSF).EQ.0)THEN
            CALL GAMUSW(GAMUSQ,GAMUSDQ)
         ENDIF
      ENDIF
#endif 
     !
     IF(MOD(ISTEP,NPRINT) == 0) THEN
        IF(PRNLEV >= 2) THEN
           CALL PRINTE(OUTU, EPROP, ETERM, 'DYNA', 'DYN', .FALSE., &
                ISTEP, TIME, ZERO, .TRUE.)
        ENDIF
        CALL WRETERM(NPRIV,TIME,QKUHEAD)
     ENDIF
     !
#if KEY_TSM==1
     ! This section is for conformational thermodynamic integration
     ! It is placed here on purpose - the routines use the forces DX,DY,DZ
     ! calculated at the X,Y,Z coordinates.   K. Kuczera
     ! DYNICT -- standard TSM + one-dimensional TI added on (KK)
     ! DYNICM -- multi dimensional conformational TI, no TP (KK)
     !
     IF(QCFTI) THEN
        CALL DYNICT(X,Y,Z,NATOM,NPRIV,AKMATI,EPROP(TOTE), &
             EPROP(TOTKE),IHPCFTI(1)%a,IHPCFTI(2)%a,IHPCFTI(3)%a, &
             IHPCFTI(4)%a,IHPCFTI(5)%a,IHPCFTI(6)%a, &
             IHPCFTI(7)%a,IHPCFTI(8)%a,IHPCFTI(9)%a, &
             IHPCFTI(10)%a,IHPCFTI(11)%a,IHPCFTI(12)%a)
     END IF
     IF(QCFTM) THEN
        CALL DYNICM(X,Y,Z,NATOM,NPRIV,AKMATI,EPROP(TOTE),EPROP(TOTKE))
     END IF
#endif 
     !
     DO I=1,NATOM
        IF (IMOVE(I) <= 0) THEN
           XOLD(I)=X(I)
           X(I)=XNEW(I)
           YOLD(I)=Y(I)
           Y(I)=YNEW(I)
           ZOLD(I)=Z(I)
           Z(I)=ZNEW(I)
        ENDIF
     ENDDO
#if KEY_BLOCK==1
     !.ab.Print out HybridH. Use the #BLOCK#
     IF(IOLEV >= 0) CALL PRHYBH()
     !.ab.
#endif 
#if KEY_TSM==1
     IF(QTSM.AND.PIGSET) THEN
        CALL PIGCVSET(X,Y,Z)
        CALL PIGCVSET(XOLD,YOLD,ZOLD)
        CALL PIGCVSET(XNEW,YNEW,ZNEW)
     ENDIF
     !            !write out
     IF(QTSM) THEN
        IF(SAVEP) THEN
           IF(MOD(NPRIV,PERFRQ) == 0) THEN
              IF(NPUMB /= 0) THEN
                 CALL PUMEPHI(PUMEP,X,Y,Z)
                 IF(IOLEV > 0) WRITE(PUNITX,3550) &
                      NPRIV,AKMATI,LAMBDA,EPROP(EPOT),VPRTTR,VPRTTP, &
                      VPRTNR,VPRTNP,VPRTVR,VPRTVP,VPRTER,VPRTEP, &
                      EPROP(TOTE),EPROP(TOTKE),PUMEP
3550             FORMAT(I12,2(1X,1PG24.16E2),/,3(2(1PG24.16E2,1X), &
                      1PG24.16E2,/),2(1PG24.16E2,1X),1PG24.16E2)
                 CALL GFLUSH(PUNITX)
              ELSE
                 IF(IOLEV > 0) WRITE(PUNITX,3575) &
                      NPRIV,AKMATI,LAMBDA,EPROP(EPOT),VPRTTR,VPRTTP, &
                      VPRTNR,VPRTNP,VPRTVR,VPRTVP,VPRTER,VPRTEP, &
                      EPROP(TOTE),EPROP(TOTKE)
3575             FORMAT(I12,2(1X,1PG24.16E2),/,3(2(1PG24.16E2,1X), &
                      1PG24.16E2,/),1PG24.16E2,1X,1PG24.16E2)
                 CALL GFLUSH(PUNITX)
              ENDIF
              ! machine dependent routine to flush output buffer
              CALL GFLUSH(PUNITX)
           ENDIF
        ENDIF
     ENDIF
     IF (SLOWST) THEN
        ORLAMBDA = RLAMBDA
        OPLAMBDA = PLAMBDA
        LAMBDA = LAMBDA + DLMBDA
        RLAMBDA = (ONE-LAMBDA)**LPOWER
        PLAMBDA = LAMBDA**LPOWER
        ASLOW = ASLOW + VPRTTR*(RLAMBDA - ORLAMBDA) + &
             VPRTTP*(PLAMBDA - OPLAMBDA)
        IF(PIGSET) CALL PIGMSET(AMASS,.FALSE.)
     ENDIF
     ! Calculate the ic perturbation interaction energies ... pure TSM
     IF(QICP .AND..NOT.(QCFTI .OR. QCFTM)) THEN
        CALL DYNICP(X,Y,Z,NATOM,NPRIV,AKMATI,EPROP(TOTE),EPROP(TOTKE), &
             IHPCFTI(1)%a,IHPCFTI(2)%a,IHPCFTI(3)%a)
     ENDIF
     !
     IF(QTSM.AND.PIGSET) THEN
        CALL PIGCVSET(XNEW,YNEW,ZNEW)
        CALL PIGCVSET(X,Y,Z)
        CALL PIGCVSET(XOLD,YOLD,ZOLD)
     ENDIF
#endif 

#if KEY_SCCDFTB==1
     if(qsccb.and.(qlamda.or.qpkac)) then
        if((istep >= sccpass).and.(mod((istep-sccpass),sccstep) == 0)) &
             then
           icntdyn=icntdyn+1
           dvdltmp=dvdlb+dvdla+dvdlp+dvdle+dvdlv+dvdlscc+dvdlub &
                +dvdlip
           dvdl=dvdl+dvdltmp
           write(outpka,'(1x,"DVDL= ",F12.8," at step ",I6)') &
                dvdltmp                  ,icntdyn

           if(mod(icntdyn,tiav) == 0) then
              iavti=iavti+1
              dtmp=dvdl/icntdyn
              dtmp1=dtmp-dtmp1
              write(outpka,*) '<dvdl> at ',iavti,'th average = ',dtmp
              dtmp1=dtmp
           endif
        endif
     endif
#endif 
  ENDDO
  IF (NDRUDE > 0) THEN
     !        Reset the IMOVE variable of the Drude particles to 0.
     !        The Drude particles are left masseless, however.
     DO I=1,NATOM
        IF (ISDRUDE(I) .AND. IMOVE(I) == -1) IMOVE(I) = 0
     ENDDO
  ENDIF
  ! END OF MAIN LOOP
  !

!  if(LTRBOMD) q_apply_tr_bomd=.false.   ! for TR-BOMD, do not apply TR-BOMD.

#if KEY_BLOCK==1 /*ldm*/
  !     reset lambdas to time t: the reason is that at the end of
  !     dcntrl.src, when ORIG is true, x is reset to xold
  !     set bldold to bxlamb(t+dt) for restart file
  if(qldm) call ldm_reset_dynam(nblock)
#endif /*  LDM*/
  IF (ISTPSA < IPRFRQ) RETURN
  !
  !     Now do long time fluctuations
  !
  call avfl_update_lt()
  NUMSTP = ISTPSA
  DNUM   = NUMSTP
  DNM1   = DNUM - ONE
  DNP1   = DNUM + ONE
  DNPM1  = DNM1 * DNP1
  IF(NUMSTP > 1) THEN
     DRIFTA = (TWO*FITA-DNP1*EPRPA(TOTE))*SIX/(DNPM1*DNUM)
     EAT0A  = EPRPA(TOTE)/DNUM-DRIFTA*DNP1/TWO
     IF((EPRP2A(TOTE)/DNUM-(EPRPA(TOTE)/DNUM)**2) > ZERO) THEN
        CORRA=DNPM1/(EPRP2A(TOTE)/DNUM-(EPRPA(TOTE)/DNUM)**2)/TWELVE
        CORRA=DRIFTA*SQRT(CORRA)
     ELSE
        CORRA=ZERO
     ENDIF
     !
     ! . Compute statistics.
     call avfl_compute(DNUM)
  ENDIF
  IF(PRNLEV >= 4) THEN
     call avfl_print_aver(NUMSTP, TIME)
     call avfl_print_fluc(NUMSTP, TIME)
  ENDIF
  DRIFTP=DRIFTA
  EAT0P=EAT0A
  CORRP=CORRA
  IF (JHSTRT <= NUMSTP) GOTO 190
  NUMSTP=JHSTRT
  DNUM=NUMSTP
  DNM1=DNUM-ONE
  DNP1=DNUM+ONE
  DNPM1=DNM1*DNP1
  IF(NUMSTP > 1) THEN
     DRIFTP = (TWO*FITP - DNP1*EPRPP(TOTE))*SIX/(DNPM1*DNUM)
     EAT0P  = EPRPP(TOTE)/DNUM-DRIFTP*DNP1/TWO
     IF((EPRP2P(TOTE)/DNUM-(EPRPP(TOTE)/DNUM)**2) > ZERO) THEN
        CORRP=DNPM1/(EPRP2P(TOTE)/DNUM-(EPRPP(TOTE)/DNUM)**2)/TWELVE
        CORRP=DRIFTP*SQRT(CORRP)
     ELSE
        CORRP=ZERO
     ENDIF
  ENDIF
  call avfl_compute_lt(DNUM)
  IF(PRNLEV >= 4)  THEN
     call avfl_print_aver_lt(NUMSTP, TIME)
     call avfl_print_fluc_lt(NUMSTP, TIME)
  ENDIF
  !
190 CONTINUE
  IF(NUMSTP <= 1 .OR. JHSTRT.LE.1 .OR. PRNLEV < 2) RETURN
  WRITE(OUTU,195)DRIFTA,DRIFTP,EAT0A,EAT0P,CORRA,CORRP
195 FORMAT(/5X,'DRIFT/STEP (LAST-TOTAL): ',1P,2G17.8, &
       /5X,'EPROP(EPOT) AT STEP 0  : ',1P,2G17.8, &
       /5X,'CORR. COEFFICIENT      : ',1P,2G17.8)
  return
end SUBROUTINE DYNAMCV
#endif /*  OLDDYN*/

SUBROUTINE NULL_CV
  RETURN
END SUBROUTINE NULL_CV

