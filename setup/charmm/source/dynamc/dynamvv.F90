SUBROUTINE DYNAMVV(VX,VY,VZ,VK,XNEW,YNEW,ZNEW, &
     XOLD,YOLD,ZOLD, &
#if KEY_DHDGB==1
!AP/MF
     VS_DHDGB,SDEFNEW,SDEFOLD, &
     SDEFLD,VK_DHDGB, &
#endif
#if KEY_CHEQ==1
     CG, CGNEW, CGOLD, VCG,PMASSQ,         & 
#endif
#if KEY_PIPF==1
     UINDN,UINDO,VUIND,PMASSU, &   
#endif
     AMASS,NPRIV,NDEGF,IGVOPT, &
     IMOVE,ISKP,FREEAT,NFREAT,NATOMX, &
     BNBND,BIMAG, &
     ISTART,ISTOP,IPRFRQ,IDYNPR,JHTEMP, &
     GAMMA,RFD,RFT,QAVER,NAVER,XAVE,YAVE,ZAVE, &
#if KEY_CHEQ==1
     CGAVE,                                & 
#endif
     QKUHEAD &
#if KEY_MTS==1
     ,XMI,YMI,ZMI,XMM,YMM,ZMM &  
#endif
     ,XLD,YLD,ZLD &
     )
  !
  !
  !     PERFORMS MOLECULAR DYNAMICS INTEGRATION WITH NO FRILLS SUCH AS
  !     TEMPERATURE ADJUSTMENT, LIST UPDATES, AND THE LIKE
  !
  !     Velocity Verlet Algorithms
  !     This routine handles the actual integration algorithms.  It also
  !     keeps track of averages and fluctuations during the run.
  !     Authors: Masa Watanabe
  !     Nose-Hoover and Multiple time scaled methods inclued by
  !     Masa Watanabe
  !
  !     2/15/98 Update to work in parallel platforms by Masa Watanabe

#if KEY_CHEQ==1
  use cheq,only: qcg,cheqnhso,kecgbath,FQNHSBATH,FQNHSOBATH, &
       cheqnhs,cheqtemp,fqnhswat,fqnhsowat, massq,noseflag, &
       nqbaths,NDGFBATH,IFIRSTBATH,fqnhmbath,ilastbath,      &
#if KEY_PARALLEL==1
       cgcpy, &           
#endif
       FQTEMPBATH,   &
       DCH,SUMDCH,DDCH
#endif 

  use chm_kinds
  use chm_types
  use dimens_fcm
  use reawri
#if KEY_SCCDFTB==1
  use blockscc_fcm  
#endif
#if KEY_DHDGB==1
!AP/MF
  use dhdgb,only:sdef,samass,totals,qfhdgb
  use derivdhdgb
#endif
#if KEY_SCCDFTB==1
  use sccdftb   
#endif
#if KEY_BLOCK==1
  use block_fcm, only : prhybh, nblock

!ldm
  use lambdam,only: qldm, nsavl,titlel, ntitll, iunldm, &
       ibvidi, ibvidj, ibclas, irreup, irrlow, ikbias, ipbias, nbiasv, &
       ldm_init_dynam,ldm_prop1_dynamvv,ldm_prop2_dynamvv, ldm_reset_dynam
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
  use icfix
  use icpert
  use nose_mod
#if KEY_ADUMB==1
  use umb
#endif 
#if KEY_GAMUS==1
  use gamusmodule 
#endif
  ! JG 5/2002
#if KEY_QUANTUM==1
  use sizes
  use quantm
#endif 
#if KEY_SQUANTM==1
  use squantm          
#endif
#if KEY_PARALLEL==1
  use parallel    
#endif
  ! FBS MOD ********************************************
  use dmcons
  use rgym
  ! END FBS MOD ****************************************
  use tbmts
#if KEY_PIPF==1
  use pipfm     
#endif
  use holonom,only:holonoma
  use machutil,only:eclock,die
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec    
#endif
  use heurist,only:updeci
  use prssre
!QC_Ito_UW0616
#if KEY_RXNCOR==1
  use rxncom, only: UMBMDSTEP
#endif

#if KEY_MNDO97==1
  use qm1_info, only : qm_control_r
#endif
  implicit none
#if KEY_DHDGB==1
  real(chm_real) VS_DHDGB(*),SDEFNEW(*),SDEFOLD(*)
  real(chm_real) SDEFLD(*),VK_DHDGB(*)
#endif
  !
  !
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
  LOGICAL QAVER,QKUHEAD
  INTEGER NAVER
  real(chm_real) XAVE(*),YAVE(*),ZAVE(*)
#if KEY_MTS==1
  real(chm_real) XMI(*),YMI(*),ZMI(*)
  real(chm_real) XMM(*),YMM(*),ZMM(*)
#endif 
  real(chm_real) XLD(*),YLD(*),ZLD(*)
  real(chm_real) JHTEMP

#if KEY_BLOCK==1
  real(chm_real)  RNORMC   /*ldm     local variables*/
#endif

  real(chm_real) DELTA2,FACT,DNUM,DNM1,DNP1,DNPM1
  real(chm_real) DELTAS,TIME,TEMPI,AKMATI,ALPHA,TOTEPR
  INTEGER NATOM,IST1,ISTEP,ISTPSA,NUMSTP,I,J,K
  INTEGER NIT2,NIT
  LOGICAL OK
#if KEY_SCCDFTB==1
  real(chm_real) dvdltmp
#endif 

#if KEY_CHEQ==1
  real(chm_real) CG(*),CGNEW(*),CGOLD(*),VCG(*),CGAVE(*), &
       KECGPROT,KEIND
  real(chm_real) KECGCPROT, KECGCWAT,KECGWAT,VCONST,PARTDCH, &
       FQMV2,KECGH2O
  real(chm_real) VEL,SD,KECGTMP, KECGC,SD1,SD2, &
       KECG,KECGOX,KECGH1,KECGH2
  real(chm_real) FQNHA,MTS3,KECGNMA
  real(chm_real) NHSDIF,MV2TMP,CMASS,CHEQNHTL,NHPOT,NHKIN
  real(chm_real) FQNHM2,FQNHS2,FQNHS2O,NHS2DIF,FQNHS2A,FQNHS2N,ECONS
  real(chm_real) NHSDIFWAT,MV2TMPWAT,FQNHMWAT,FQNHAWAT,FQCMV2WAT, &
       FQCDGFWAT,FQNHSNWAT,MASINV,LAMSRF,SUMKE
  real(chm_real) ETA(10),ETAD(10),ETADD(10),ETANEW(10), &
       ETAOLD(10),META(10),ETA1DIF,KECGSOL
  real(chm_real) MRENO(12,12),KENOND
  real(chm_real) FQNHABATH(10), FQNHSNBATH(10), NHSDIFBATH(10), &
       MAXERROR,MAXEOLD, &
       MV2TMPBATH(10),ERROR
  real(chm_real) PMASSQ(NATOMX)
  INTEGER J1, I1
  INTEGER NHITR,CHEQNHMX,IBS,LLL,LLLL,LL,CHEQCDGF
  INTEGER NCHAINS
  INTEGER IMASS,JMASS, itest
  INTEGER L,IMAX
  LOGICAL QCHEQRDPRM
#endif 
  ! PJ 06/2005
#if KEY_PIPF==1
  real(chm_real) UINDN(3,*),UINDO(3,*), &
       VUIND(3,*),PMASSU(*)
#endif 
  !
#if KEY_PARALLEL==1
  real(chm_real) GCARR(11),TIMMER
#endif 
  real(chm_real)  TMPN(MAXNOS),EPTKE1(MAXNOS),EPTKX(MAXNOS)
  real(chm_real)  SNHV1(MAXNOS)
  real(chm_real)  ECSUM,FACT1,SA1X,SA2X,SA3X
  real(chm_real)  RAVL,TOTKEN,TOTKEO,SAX
  real(chm_real)  SS1X,SS2X,SS3X,SNHF1
  real(chm_real)  QK1,QK2,QV1,QV2,FACT2
  real(chm_real)  EMTS(LENENT)
  INTEGER NNQ, NMTS0
  INTEGER ATFRST,ATLAST
  LOGICAL QOK
#if KEY_CHEQ==1
  real(chm_real) FACTCG,FACTCG1,FACTCG2,ALPHACG2
  LOGICAL QSECD,LEWALD
#endif 

  LOGICAL DUMM
  !lb.. 01-AUG-98
  SAVE EPTKE1 , EPTKX,TOTKEN,TOTKEO,RAVL,TEMPI,QK1,QK2
  SAVE ISTEP
  !..
  INTEGER FSTEP,SSTEP,AVG,FLUC
  PARAMETER(FSTEP=0,SSTEP=1,AVG=2,FLUC=3)
  !
  real(chm_real) FLUCTD
  !
  !     TIMFAC is the conversion factor from AKMA time to picoseconds.
  !
  DATA TOTEPR/-9999.0D0/
  !
  DUMM=.FALSE.
  !
  NATOM=NATOMX
  !
#if KEY_MNDO97==1
  ! namkh 09/12/03 for MNDO97 printing setup
  ISTEPQM = ISTART-1
#endif 
  !
#if KEY_DOMDEC==1
  if (q_domdec) then
     call wrndie(-5,'<dynamvv>','DOMDEC not implemented for dynamvv')
  else
#endif 
     atfrst = 1
     atlast = natom
#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
     ATFRST=1+IPARPT(MYNOD)
     ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
     ATFRST=1
     ATLAST=NATOM
#endif /* (parfmain)*/
#else /* (paramain)*/
     ATFRST=1
     ATLAST=NATOM
#endif /* (paramain)*/
#if KEY_DOMDEC==1
  endif  
#endif

#if KEY_CHEQ==1
  IF (QCG) THEN

     IF (IBS == 0) THEN
        CHEQNHS = 1.0
        CHEQNHSO = 1.0
        FQNHS2 = 1.0  ! second chain
        FQNHS2O = 1.0  ! second chain
        NHSDIF = 0.1
        FQNHSWAT = 1.0
        FQNHSOWAT = 1.0
        NCHAINS = 3
        !   test for multiple chain NOSE HOOVER
        DO I=1,NCHAINS
           ETA(I)=1.0
           ETAD(I)=0.0
           ETADD(I)=0.0
           ETAOLD(I)=1.0
           ETANEW(I)=1.0
           META(I) = 0.005
        ENDDO

     ENDIF
     IBS = 1

     IF(NPRIV == 0) THEN

        FQNHSWAT = 1.0
        FQNHSOWAT = 1.0
        DO I = 1,NATOM
           CGOLD(I) = CG(I)
        ENDDO
     ENDIF

  ENDIF

#endif 


#if KEY_CHEQ==1
#if KEY_PARALLEL==1
  IF (MYNOD == 0) THEN
#endif 
     IF (QCG .and. prnlev > 5) THEN
        write(outu,*) " Charge Bath Temperatures"
        write(OUTU,189)((KECGBATH(L)/(KBOLTZ*NDGFBATH(L))) &
             ,L=1,NQBATHS)
        write(OUTU,*) "Charge KE = ", KECG
     ENDIF
#if KEY_PARALLEL==1
  ENDIF
#endif 

189 format(2(f10.5,1x))

#endif 

  !
  ! First (N)
  !
  DELTAS=HALF*DELTA
  DELTA2=DELTA*DELTAS

#if KEY_CHEQ==1
  IF (QCG) THEN
     FACTCG=DELTA2/MASSQ
     FACTCG1=DELTAS/MASSQ
  ENDIF
#endif 



  !
  ! Second (M)
  !
#if KEY_MTS==1
  SS1X=DELTA*NMTS1
#else /**/
  SS1X=DELTA
#endif 
  SS2X=HALF*SS1X
  SS3X=SS1X*SS2X
  !
#if KEY_MTS==1
  !
  !  Multitime Dt=N*M*dt
  !
  IF (.NOT. QTBMTS) NMTS=1
  NMTS0=NMTS
#else /**/
  NMTS0=1
#endif 
  !
  SA1X=DELTA*NMTS0
  SA2X=HALF*SA1X
  SA3X=SA1X*SA2X
  !
  IST1=ISTART-1
  !
  IF(QAVER) THEN
     IF(NAVER == 0) THEN
        do i=atfrst,atlast
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
              XAVE(I)=ZERO
              YAVE(I)=ZERO
              ZAVE(I)=ZERO
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
     ENDIF
  ENDIF
  !
  DO I=1,LENENT
     EMTS(I)=ZERO
  ENDDO
  !
  !
#if KEY_BLOCK==1 /*ldm*/
  !:    set BLCOEF = LAMBDA before update the lambdas or call energy
  !:    lambda(1) == 1 by the design see block command
  if(qldm) call ldm_init_dynam(nblock)
#endif /*  LDM*/
  !
  IF(IGVOPT >= 3 .AND. (IDYNPR == 0.OR.JHSTRT.EQ.0)) THEN
#if KEY_MTS==1 /*mts*/
     !
     !  Multiple Time Scale (Initial Energy)
     !
     IF (QTBMTS) THEN
        IF(NMTS2 >= 1) THEN
           ENE2=.TRUE.
           CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
           DO I=1,LENENT
              EMTS(I)=ETERM(I)
           ENDDO
           do i=atfrst,atlast
#if KEY_PARASCAL==1
              IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                 XMM(I)=DX(I)
                 YMM(I)=DY(I)
                 ZMM(I)=DZ(I)
#if KEY_PARASCAL==1
              ENDIF
#endif 
           ENDDO
        ENDIF
        ENE1=.TRUE.
        CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
        DO I=1,LENENT
           EMTS(I)=EMTS(I)+ETERM(I)
        ENDDO
        do i=atfrst,atlast
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
              XMI(I)=DX(I)
              YMI(I)=DY(I)
              ZMI(I)=DZ(I)
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
        ENE3=.TRUE.
        CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
        ECSUM=ZERO
        DO I=1,LENENT
           ETERM(I)=ETERM(I)+EMTS(I)
           ECSUM=ETERM(I)+ECSUM
        ENDDO
        EPROP(EPOT)=ECSUM
     ELSE
#endif /* (mts)*/
        !       Get previous energy for printing (just as a check)
        CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0 &
#if KEY_DHDGB==1
!AP/MF
                   ,SDEFin=SDEF,DS_DHDGBout=DS_DHDGB &
#endif
      )
#if KEY_MTS==1
     ENDIF
#endif 
     !
     CALL GRAM( &
#if KEY_MTS==1
          XMI,YMI,ZMI,XMM,YMM,ZMM, &   
#endif
          DX,DY,DZ &
#if KEY_DHDGB==1
          ,DS_DHDGB &
#endif
    )
  ENDIF
  !
  IF(IGVOPT == 2) THEN

     !
     !       This is the 2-step VERLET
     do i=atfrst,atlast
#if KEY_PARASCAL==1
        IF(JPBLOCK(I) == MYNOD) THEN
#endif 
           XOLD(I)=X(I)
           YOLD(I)=Y(I)
           ZOLD(I)=Z(I)
#if KEY_PARASCAL==1
        ENDIF
#endif 
     ENDDO
     !
#if KEY_DHDGB==1
!AP/MF
     IF (QFHDGB) THEN
         DO I=1,TOTALS
            SDEFOLD(I)=SDEF(I)
         ENDDO
     ENDIF
#endif
     IF(QHOLO) THEN
        !         Do shake just in case coordinates don't fit constraints
        CALL HOLONOMA(X,Y,Z,XOLD,YOLD,ZOLD,.TRUE.,.TRUE.,QOK)
        !
#if KEY_PARALLEL==1
        CALL VDGBR(X,Y,Z,0)  
#endif

        !
        !        DO I=1,NATOM
        !        FACT=-AMASS(I)/(2.0*DELTA2)
        !        VX1(I)=(X(I)-XOLD(I))*FACT
        !        VY1(I)=(Y(I)-YOLD(I))*FACT
        !        VZ1(I)=(Z(I)-ZOLD(I))*FACT
        !        ENDDO
        !        ENDIF
        !
#if KEY_MTS==1
        IF (QTBMTS) THEN
           DUMM=.TRUE.
           QTBMTS = .FALSE.
        ENDIF
#endif 
        CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0 &
#if KEY_DHDGB==1
!AP/MF
             ,SDEFin=SDEF,DS_DHDGBout=DS_DHDGB &
#endif
       )
#if KEY_MTS==1
        IF(DUMM) THEN
           DUMM = .FALSE.
           QTBMTS = .TRUE.
        ENDIF
#endif 
        !
        do i=atfrst,atlast
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
              IF(IMOVE(I) == 0) THEN
                 !           FACT1=2.0*DELTAS/AMASS(I)
                 FACT1=DELTAS/AMASS(I)
                 VX(I)=VX(I)-FACT1*DX(I)
                 VY(I)=VY(I)-FACT1*DY(I)
                 VZ(I)=VZ(I)-FACT1*DZ(I)
              ENDIF
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
#if KEY_DHDGB==1
!AP/MF
        IF (QFHDGB) THEN
            DO I=1,TOTALS
               FACT1=DELTAS/SAMASS
               VS_DHDGB(I)=VS_DHDGB(I)-FACT1*DS_DHDGB(I)
            ENDDO
        ENDIF
#endif
        !--------------------------------------------
        do i=atfrst,atlast
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
              IF(IMOVE(I) == 0) THEN
                 XNEW(I)=X(I)+DELTA*VX(I)
                 YNEW(I)=Y(I)+DELTA*VY(I)
                 ZNEW(I)=Z(I)+DELTA*VZ(I)
                 ! Bugfix AvdV: to avoid shake errors upon restart when atoms are fixed.
              else
                 xnew(i)=x(i)
                 ynew(i)=y(i)
                 znew(i)=z(i)
              ENDIF
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
#if KEY_DHDGB==1
!AP/MF
        IF (QFHDGB) THEN
            DO I=1,TOTALS
               SDEFNEW(I)=SDEF(I)+DELTA*VS_DHDGB(I)
            ENDDO
        ENDIF
#endif
        !
        CALL DYNASHK(VX,VY,VZ,XNEW,YNEW,ZNEW,AMASS,IMOVE, &
             ISKP,NATOM,DELTA)
        !
        do i=atfrst,atlast
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
              X(I)=XNEW(I)
              Y(I)=YNEW(I)
              Z(I)=ZNEW(I)
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
#if KEY_DHDGB==1
!AP/MF
        IF (QFHDGB) THEN
            DO I=1,TOTALS
               SDEF(I)=SDEFNEW(I)
            ENDDO
        ENDIF
#endif
#if KEY_PARALLEL==1
        CALL VDGBR(X,Y,Z,0)
#endif 
     ENDIF
     !
#if KEY_BLOCK==1 /*ldm*/
     if(qldm) call ldm_prop1_dynamvv(nblock, delta)
#endif /*  LDM*/
     !
#if KEY_MTS==1 /*mts*/
     !
     ! Multiple Time Scale (Initial Energy)
     !
     IF (QTBMTS) THEN
        !
        !- MEDIUM FORCE
        IF(NMTS2 >= 1) THEN
           ENE2=.TRUE.
           CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
           DO I=1,LENENT
              EMTS(I)=ETERM(I)
           ENDDO
           do i=atfrst,atlast
#if KEY_PARASCAL==1
              IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                 XMM(I)=DX(I)
                 YMM(I)=DY(I)
                 ZMM(I)=DZ(I)
#if KEY_PARASCAL==1
              ENDIF
#endif 
           ENDDO
        ELSE
           do i=atfrst,atlast
#if KEY_PARASCAL==1
              IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                 XMM(I)=ZERO
                 YMM(I)=ZERO
                 ZMM(I)=ZERO
#if KEY_PARASCAL==1
              ENDIF
#endif 
           ENDDO
        ENDIF
        !
        ! - FASTEST VERING FORCE
        ENE1=.TRUE.
        CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
        DO I=1,LENENT
           EMTS(I)=EMTS(I)+ETERM(I)
        ENDDO
        do i=atfrst,atlast
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
              XMI(I)=DX(I)
              YMI(I)=DY(I)
              ZMI(I)=DZ(I)
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
        !
        !- SLOW VERING FORCE
        ENE3=.TRUE.
        CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
        ECSUM=ZERO
        DO I=1,LENENT
           ETERM(I)=ETERM(I)+EMTS(I)
           ECSUM=ETERM(I)+ECSUM
        ENDDO
        EPROP(EPOT)=ECSUM
     ELSE
#endif /* (mts)*/
        CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0 &
#if KEY_DHDGB==1
!AP/MF
                   ,SDEFin=SDEF,DS_DHDGBout=DS_DHDGB &
#endif
               )

#if KEY_MTS==1
     ENDIF
#endif 
     CALL GRAM( &
#if KEY_MTS==1
          XMI,YMI,ZMI,XMM,YMM,ZMM, &    
#endif
          DX,DY,DZ &
#if KEY_DHDGB==1
!AP/MF
          ,DS_DHDGB &
#endif
      )
     !
     IGVOPT=3
     !
#if KEY_MTS==1
     IF (QTBMTS) THEN
        do i=atfrst,atlast
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
              IF(IMOVE(I) == 0) THEN
                 FACT1=DELTAS/AMASS(I)
                 FACT=SS2X/AMASS(I)
                 FACT2=SA2X/AMASS(I)
                 VX(I)=VX(I)-FACT1*XMI(I)-FACT*XMM(I)- &
                      FACT2*DX(I)
                 VY(I)=VY(I)-FACT1*YMI(I)-FACT*YMM(I)- &
                      FACT2*DY(I)
                 VZ(I)=VZ(I)-FACT1*ZMI(I)-FACT*ZMM(I)- &
                      FACT2*DZ(I)
              ENDIF
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
     ELSE
#endif 
        do i=atfrst,atlast
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
              IF(IMOVE(I) == 0) THEN
                 FACT1=DELTAS/AMASS(I)
                 VX(I)=VX(I)-FACT1*DX(I)
                 VY(I)=VY(I)-FACT1*DY(I)
                 VZ(I)=VZ(I)-FACT1*DZ(I)
              ENDIF
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
#if KEY_DHDGB==1
!AP/MF
        IF (QFHDGB) THEN
            DO I=1,TOTALS
               FACT1=DELTAS/SAMASS
               VS_DHDGB(I)=VS_DHDGB(I)-FACT1*DS_DHDGB(I)
            ENDDO
        ENDIF
#endif
        !
#if KEY_MTS==1
     ENDIF
#endif 
#if KEY_BLOCK==1 /*ldm*/
     if(qldm) call ldm_prop2_dynamvv(nblock, delta)
#endif /*  LDM*/

  ENDIF
  !
  IF(ISTOP < ISTART) CALL DIE
  IF(MOD(IST1,IPRFRQ) == 0) THEN
     call avfl_reset()
     FITA = ZERO
  ENDIF
  !
  IF (IDYNPR == 0 .OR. JHSTRT.EQ.0) THEN
     IF(QNOSE) THEN
        DO I=1,NOBL
           EPTKX(I)=ZERO
        ENDDO
     ENDIF
     !
     TOTKEN=ZERO
     TEMPI=ZERO
#if KEY_DHDGB==1
!AP/MF
     IF (QFHDGB) THEN
         DO J=1,TOTALS
            RAVL=SAMASS*VS_DHDGB(J)**2
            TEMPI=TEMPI+SAMASS*VS_DHDGB(J)**2
            IF (QNOSE) THEN
                EPTKX(1)=EPTKX(1)+RAVL
            ENDIF
         ENDDO
      ENDIF
#endif
     do i=atfrst,atlast
#if KEY_PARASCAL==1
        IF(JPBLOCK(I) == MYNOD) THEN
#endif 
           IF(IMOVE(I) == 0) THEN
              RAVL = AMASS(I)*(VX(I)**2+VY(I)**2+VZ(I)**2)
              TEMPI=TEMPI+RAVL
              IF(QNOSE) THEN
                 J = INLCKP(I)
                 EPTKX(J)=EPTKX(J)+RAVL
              ENDIF
           ENDIF
#if KEY_PARASCAL==1
        ENDIF
#endif 
     ENDDO
#if KEY_PARALLEL==1
     IF(QNOSE) THEN
        GCARR(1) = TEMPI
        DO I = 1,NOBL
           GCARR(I+1) = EPTKX(I)
        ENDDO
        CALL GCOMB(GCARR,NOBL+1)
        TEMPI = GCARR(1)
        DO I = 1, NOBL
           EPTKX(I) = GCARR(I+1)
        ENDDO
     ELSE
        GCARR(1) = TEMPI
        CALL GCOMB(GCARR,1)
        TEMPI = GCARR(1)
     ENDIF
#endif 
     !
     !       WRITE OUT INITIAL CONDITIONS
     !
     ISTEP=ISTART-1
     TIME=TIMFAC*DELTA*NMTS0*NPRIV
     !       . Calculate the pressures.
     CALL PRSEXT
     CALL PRSINT(NATOM,AMASS,VX,VY,VZ,ONE,VX,VX,VX,ZERO)
     ! APH: Note VX, VX, VX here ------------|
     !      are just dummies
     !
     QK1=ZERO
     QK2=ZERO
     IF(QNOSE) THEN
        DO I = 1, NOBL
           SNHF(I)=EPTKX(I)-NDGN(I)*KBOLTZ*RTMPR(I)
           QK1=QK1+0.5*SQM(I)*SNHV(I)*SNHV(I)
           QK2=QK2+NDGN(I)*KBOLTZ*SNH(I)*RTMPR(I)
        ENDDO
     ENDIF
     !
     EPROP(TOTKE)=TEMPI/TWO
     EPROP(TOTE)=EPROP(EPOT)+EPROP(TOTKE)+QK1+QK2
     !
     IF(QNOSE) THEN
        EPROP(HFCTE)=EPROP(EPOT)+EPROP(TOTKE)
        EPROP(EHFC)=QK1+QK2
     ELSE
        EPROP(HFCTE)=ZERO
        EPROP(EHFC)=ZERO
     ENDIF
     EPROP(TEMPS)=TEMPI/(KBOLTZ*NDEGF)
     CALL PRINTE(OUTU, EPROP, ETERM, 'DYNA', 'DYN', .TRUE., &
          ISTEP, TIME, ZERO, .TRUE.)
  ENDIF
  !
  !     INITIALIZE ACCUM VARIABLES
#if KEY_DHDGB==1
!AP/MF
  IF (QFHDGB) THEN
      DO I=1,TOTALS
         VK_DHDGB(I)=ZERO
      ENDDO
  ENDIF
#endif
  IF(JHSTRT == 0) THEN
     do i=atfrst,atlast
#if KEY_PARASCAL==1
        IF(JPBLOCK(I) == MYNOD) THEN
#endif 
           VK(I)=ZERO
#if KEY_PARASCAL==1
        ENDIF
#endif 
     ENDDO
     JHTEMP=ZERO
     FITP=ZERO
     call avfl_reset_lt()
#if KEY_QUANTUM==1
     ! JG 5/2002
     IF(QMPERT.OR.QDECOM) THEN
        DO I = 1,LENQEQ
           EQPRP(I) = ZERO
           EQPR2P(I) = ZERO
        ENDDO
     ENDIF
#endif 
#if KEY_SQUANTM==1
     IF(QMFEP .or. QMSOLV) THEN
        DO I = 1, LENQEQ
           EQPRP(I) = ZERO
           EQPR2P(I) = ZERO
        END DO
     END IF
#endif 
     !
  ENDIF
  !
#if KEY_QUANTUM==1
  ! JG 5/2002
  IF (CHDYN) THEN
     IF (ISTART == 1) THEN
        CALL DYNDEN('INIT',1)
     END IF
  END IF
#endif
#if KEY_MNDO97==1 
  !!!qm_control_r%md_run =.true.        ! this is MD run.
  !!!qm_control_r%md_run =.false.       ! to turn off this. (See the note in qmmm_interface.src.)
#endif
  !
  !-------------------------------------------------------
  !     THIS IS THE MAIN LOOP FOR DYNAMICS.
  !-------------------------------------------------------
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
     !
#if KEY_PARALLEL==1
     CALL PSYNC()
     TIMMER=ECLOCK()
#endif 
     !
#if KEY_MTS==1
     IF (QTBMTS) THEN
        DO I=1,LENENT
           EMTS(I)=ZERO
        ENDDO
     ENDIF
#endif 
     !
     JHSTRT=JHSTRT+1
     !
     NPRIV=NPRIV+1
     !
     !
#if KEY_PARALLEL==1
     TMERI(TIMDCNTRL) = TMERI(TIMDCNTRL)+ECLOCK()-TIMMER
     CALL PSYNC()
     TIMMER=ECLOCK()
     !
     TMERI(TIMGCOMM) = TMERI(TIMGCOMM)+ECLOCK()-TIMMER
     CALL PSYNC()
     TIMMER=ECLOCK()
#endif 
     !
     ! FBS MOD ************************************
#if KEY_DMCONS==1
     DSTEP = NPRIV
#endif 
#if KEY_RGYCONS==1
     DSTEPRG = NPRIV
#endif 
     ! END FBS MOD ********************************
     !
     IF(QAVER) THEN
        NAVER=NAVER+1
        do i=atfrst,atlast
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
              XAVE(I)=XAVE(I)+X(I)
              YAVE(I)=YAVE(I)+Y(I)
              ZAVE(I)=ZAVE(I)+Z(I)
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
     ENDIF
     !
     IF(NSAVC > 0) THEN
        IF (MOD(ISTEP,NSAVC) == 0) THEN
           IF (QAVER) THEN
              DNUM=ONE
              DNUM=DNUM/NAVER
              !
              do i=atfrst,atlast
#if KEY_PARASCAL==1
                 IF(JPBLOCK(I) == MYNOD) THEN  
#endif
                    XAVE(I)=XAVE(I)*DNUM
                    YAVE(I)=YAVE(I)*DNUM
                    ZAVE(I)=ZAVE(I)*DNUM
#if KEY_PARASCAL==1
                 ENDIF  
#endif
              ENDDO
              !
#if KEY_PARALLEL==1
              CALL VDGBR(XAVE,YAVE,ZAVE,1)  
#endif
              !
              CALL WRITCV(XAVE,YAVE,ZAVE, &
#if KEY_CHEQ==1
                   CGAVE,QCG,                         & 
#endif
                   NATOM,FREEAT,NFREAT,NPRIV, &
                   ISTEP,NDEGF,SA1X,NSAVC,NSTEP,TITLEA,NTITLA,IUNCRD, &
                   .FALSE.,.FALSE., (/ 0 /), .FALSE., (/ ZERO /))
              NAVER=0
              do i=atfrst,atlast
#if KEY_PARASCAL==1
                 IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                    XAVE(I)=ZERO
                    YAVE(I)=ZERO
                    ZAVE(I)=ZERO
#if KEY_PARASCAL==1
                 ENDIF
#endif 
              ENDDO
           ELSE
#if KEY_PARALLEL==1
              CALL VDGBR(XAVE,YAVE,ZAVE,2)  
#endif
              CALL WRITCV(X,Y,Z, &
#if KEY_CHEQ==1
                   (/ ZERO /), .FALSE., &  
#endif
                   NATOM,FREEAT,NFREAT,NPRIV,ISTEP,NDEGF, &
                   SA1X,NSAVC,NSTEP,TITLEA,NTITLA,IUNCRD,.FALSE., &
                   .FALSE., (/ 0 /), .FALSE., (/ ZERO /))
#if KEY_DHDGB==1
!AP/MF
              IF ((IUDHDGB .GE. 0) .AND. QFHDGB) THEN
                 IF(MOD(ISTEP,NSAVC) .EQ.0) THEN
                       WRITE(IUDHDGB,102) SDEF(1),SDEF(2),SDEF(3),SDEF(4), &
                                SDEF(5), &
                                SDEF(6),SDEF(7),SDEF(8),SDEF(9),SDEF(10)
102        FORMAT(10F7.3)
                       CALL GFLUSH(IUDHDGB)
                 ENDIF
              ENDIF
#endif

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
     CALL UPDECI(ISTEP-1,X,Y,Z,WMAIN, &
          !     $              -1,XOLD,YOLD,ZOLD,VX,VY,VZ)
          1,XOLD,YOLD,ZOLD,VX,VY,VZ)
     !
#if KEY_MTS==1
     IF (QTBMTS) THEN
        do i=atfrst,atlast
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
              IF(IMOVE(I) == 0) THEN
                 FACT=0.5*SA2X/AMASS(I)
                 VX(I)=VX(I)-FACT*DX(I)
                 VY(I)=VY(I)-FACT*DY(I)
                 VZ(I)=VZ(I)-FACT*DZ(I)
              ENDIF
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
     ENDIF
#endif 
     !
     IF(QNOSE) THEN
        DO I=1,NOBL
           SNHV1(I)=SNHV(I)+SA2X*SNHF(I)/SQM(I)
           SNH(I)=SNH(I)+SA1X*SNHV1(I)
        ENDDO
        !
        do i=atfrst,atlast
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
              IF(IMOVE(I) == 0) THEN
                 J = INLCKP(I)
                 FACT=SA2X*SNHV(J)
                 !        VX(I)=VX(I)*FACT
                 !        VY(I)=VY(I)*FACT
                 !        VZ(I)=VZ(I)*FACT
                 VX(I)=VX(I)-FACT*VX(I)
                 VY(I)=VY(I)-FACT*VY(I)
                 VZ(I)=VZ(I)-FACT*VZ(I)
              ENDIF
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
     ENDIF
     !
#if KEY_DHDGB==1
!AP/MF
     IF (QFHDGB) THEN
        DO I=1,TOTALS
           FACT=SA2X*SNHV(1)
           VS_DHDGB(I)=VS_DHDGB(I)-FACT*VS_DHDGB(I)
        ENDDO
     ENDIF
#endif
#if KEY_MTS==1
     IF (QTBMTS) THEN
        do i=atfrst,atlast
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
              IF(IMOVE(I) == 0) THEN
                 FACT=0.5*SA2X/AMASS(I)
                 VX(I)=VX(I)-FACT*DX(I)
                 VY(I)=VY(I)-FACT*DY(I)
                 VZ(I)=VZ(I)-FACT*DZ(I)
              ENDIF
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
        !
        do i=atfrst,atlast
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
              IF(IMOVE(I) == 0) THEN
                 IF(IMTS(I) < 0) THEN
                    XNEW(I)=X(I)+SA1X*VX(I)
                    YNEW(I)=Y(I)+SA1X*VY(I)
                    ZNEW(I)=Z(I)+SA1X*VZ(I)
                 ENDIF
                 ! Bugfix AvdV: to avoid shake errors upon restart when atoms are fixed.
              else
                 if (imts(i) < 0) then
                    xnew(i)=x(i)
                    ynew(i)=y(i)
                    znew(i)=z(i)
                 endif
              ENDIF
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
        !
        !------------------------------------------------
        IF(QHOLO.AND.(.NOT.SLFG)) THEN
           CALL DYNASHK(VX,VY,VZ,XNEW,YNEW,ZNEW,AMASS, &
                IMOVE,ISKP,NATOM,SA1X)
        ENDIF
        !------------------------------------------------
        !
        IF(NMTS2 >= 1) THEN
           do i=atfrst,atlast
#if KEY_PARASCAL==1
              IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                 IF(IMOVE(I) == 0) THEN
                    DX(I)=XMM(I)
                    DY(I)=YMM(I)
                    DZ(I)=ZMM(I)
                 ENDIF
#if KEY_PARASCAL==1
              ENDIF
#endif 
           ENDDO
        ENDIF
        !
     ENDIF
#endif 
     !==================================================
     do i=atfrst,atlast
#if KEY_PARASCAL==1
        IF(JPBLOCK(I) == MYNOD) THEN
#endif 
           IF(IMOVE(I) == 0) THEN
              XOLD(I)=X(I)
              YOLD(I)=Y(I)
              ZOLD(I)=Z(I)
           ENDIF
#if KEY_PARASCAL==1
        ENDIF
#endif 
     ENDDO
#if KEY_DHDGB==1
!AP/MF
     IF (QFHDGB) THEN
         DO I=1,TOTALS
            SDEFOLD(I)=SDEF(I)
         ENDDO
     ENDIF
#endif
     !===================================================
#if KEY_MTS==1
     NIT2=0
2222 CONTINUE
     IF (QTBMTS) THEN
        NIT2=NIT2+1
        IF(NMTS2 >= 1) THEN
           !---------------------------------------------------
           do i=atfrst,atlast
#if KEY_PARASCAL==1
              IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                 IF(IMTS(I) < 0) GOTO 2985
                 IF (IMOVE(I) == 0) THEN
                    FACT1=SS2X/AMASS(I)
                    VX(I)=VX(I)-FACT1*DX(I)
                    VY(I)=VY(I)-FACT1*DY(I)
                    VZ(I)=VZ(I)-FACT1*DZ(I)
                 ENDIF
2985             CONTINUE
#if KEY_PARASCAL==1
              ENDIF
#endif 
           ENDDO
        ENDIF
        !---------------------------------------------------
        !         IF(TBMTS.AND.SLFG) THEN
        !           DO I=1,NATOM
        !           IF((IMTM(I) /= 2).OR.(IMTS(I) < 0)) GOTO 989
        !           IF (IMOVE(I) == 0) THEN
        !            XNEW(I)=X(I)+SS1X*VX(I)
        !            YNEW(I)=Y(I)+SS1X*VY(I)
        !            ZNEW(I)=Z(I)+SS1X*VZ(I)
        !           ENDIF
        !989        CONTINUE
        !           ENDDO
        !
        !         IF (QHOLO) THEN
        !         CALL DYNASHK(VX,VY,VZ,XNEW,YNEW,ZNEW,AMASS,IMOVE,
        !     &        ISKP,NATOM,SS1X)
        !         ENDIF
        !
        !         DO I=1,NATOM
        !         IF((IMOVE(I) == 0).AND.(IMTM(I).EQ.2)) THEN
        !         X(I)=XNEW(I)
        !         Y(I)=YNEW(I)
        !         Z(I)=ZNEW(I)
        !         ENDIF
        !         ENDDO
        !        ENDIF
        !
        !----------------------------------------------------
        !
        do i=atfrst,atlast
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
              IF(IMOVE(I) == 0) THEN
                 DX(I)=XMI(I)
                 DY(I)=YMI(I)
                 DZ(I)=ZMI(I)
              ENDIF
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
     ENDIF
     !
     !----------------------------------------------------
     NIT=0
1111 CONTINUE
     IF (QTBMTS) NIT = NIT + 1
#endif 
     do i=atfrst,atlast
#if KEY_PARASCAL==1
        IF(JPBLOCK(I) == MYNOD) THEN
#endif 
#if KEY_MTS==1
           IF (QTBMTS .AND. IMTS(I) < 0) GOTO 985
#endif 
           IF(IMOVE(I) == 0) THEN
              FACT1=DELTAS/AMASS(I)
              VX(I)=VX(I)-FACT1*DX(I)
              VY(I)=VY(I)-FACT1*DY(I)
              VZ(I)=VZ(I)-FACT1*DZ(I)
           ENDIF
985        CONTINUE
#if KEY_PARASCAL==1
        ENDIF
#endif 
     ENDDO
     !
#if KEY_DHDGB==1
!AP/MF
     IF (QFHDGB) THEN
         DO I=1,TOTALS
            FACT1=DELTAS/SAMASS
            VS_DHDGB(I)=VS_DHDGB(I)-FACT1*DS_DHDGB(I)
         ENDDO
     ENDIF
#endif

     do i=atfrst,atlast
#if KEY_PARASCAL==1
        IF(JPBLOCK(I) == MYNOD) THEN
#endif 
#if KEY_MTS==1
           IF (QTBMTS .AND. IMTS(I) < 0) GOTO 980
#endif 
           IF (IMOVE(I) == 0) THEN
              XNEW(I)=X(I)+DELTA*VX(I)
              YNEW(I)=Y(I)+DELTA*VY(I)
              ZNEW(I)=Z(I)+DELTA*VZ(I)
              ! Bugfix AvdV: to avoid shake errors upon restart when atoms are fixed.
           else
              xnew(i)=x(i)
              ynew(i)=y(i)
              znew(i)=z(i)
           ENDIF
980        CONTINUE
#if KEY_PARASCAL==1
        ENDIF
#endif 
     ENDDO
     !
#if KEY_DHDGB==1
!AP/MF
     IF (QFHDGB) THEN
         DO I=1,TOTALS
            SDEFNEW(I)=SDEF(I)+DELTA*VS_DHDGB(I)
         ENDDO
     ENDIF
#endif
#if KEY_BLOCK==1 /*ldm*/
     if(qldm) call ldm_prop1_dynamvv(nblock, delta)
#endif /*  LDM*/
     !
     !---------------------------------------------------------------
#if KEY_MTS==1
     IF ((.NOT. QTBMTS) .OR. (QTBMTS .AND. SLFG)) THEN
#endif 
        IF (QHOLO) THEN
           !            CALL ADDVIR
           CALL DYNASHK(VX,VY,VZ,XNEW,YNEW,ZNEW,AMASS,IMOVE, &
                ISKP,NATOM,DELTA)
        ENDIF
#if KEY_MTS==1
     ENDIF
#endif 
     !---------------------------------------------------------------
     !          DO I=1,NATOM
     !...##IF MTS
     !           IF(TBMTS.AND.(IMTS(I) < 0)) GOTO 985
     !...##ENDIF
     !           IF (IMOVE(I) == 0) THEN
     !           FACT1=DELTAS/AMASS(I)
     !           VX(I)=VX(I)-FACT1*DX(I)
     !           VY(I)=VY(I)-FACT1*DY(I)
     !           VZ(I)=VZ(I)-FACT1*DZ(I)
     !           ENDIF
     !985        CONTINUE
     !           ENDDO
     !
     do i=atfrst,atlast
#if KEY_PARASCAL==1
        IF(JPBLOCK(I) == MYNOD) THEN
#endif 
           IF(IMOVE(I) == 0) THEN
              X(I)=XNEW(I)
              Y(I)=YNEW(I)
              Z(I)=ZNEW(I)
           ENDIF
#if KEY_PARASCAL==1
        ENDIF
#endif 
     ENDDO
     !
#if KEY_DHDGB==1
!AP/MF
      IF (QFHDGB) THEN
          DO I=1,TOTALS
             SDEF(I)=SDEFNEW(I)
          ENDDO
      ENDIF
#endif
#if KEY_PARALLEL==1
     CALL VDGBR(X,Y,Z,0)  
#endif
     !
#if KEY_MTS==1
     IF (QTBMTS) THEN
        ENE1=.TRUE.
        CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
     ELSE
#endif 
        QDYNCALL=.TRUE.
        CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1 &
#if KEY_DHDGB==1
!AP/MF
                    ,SDEFin=SDEF,DS_DHDGBout=DS_DHDGB &
#endif
        )
        QDYNCALL=.FALSE.
        CALL GRAM( &
#if KEY_MTS==1
             XMI,YMI,ZMI,XMM,YMM,ZMM, & 
#endif
             DX,DY,DZ &
#if KEY_DHDGB==1
            ,DS_DHDGB &
#endif
        )
#if KEY_MTS==1
     ENDIF
#endif 
     !
     do i=atfrst,atlast
#if KEY_PARASCAL==1
        IF(JPBLOCK(I) == MYNOD) THEN
#endif 
#if KEY_MTS==1
           IF (QTBMTS .AND. IMTS(I) < 0) GOTO 979
#endif 
           IF (IMOVE(I) == 0) THEN
              FACT=DELTAS/AMASS(I)
              VX(I)=VX(I)-DX(I)*FACT
              VY(I)=VY(I)-DY(I)*FACT
              VZ(I)=VZ(I)-DZ(I)*FACT
              XLD(I)=VX(I)
              YLD(I)=VY(I)
              ZLD(I)=VZ(I)
           ENDIF
979        CONTINUE
#if KEY_PARASCAL==1
        ENDIF
#endif 
     ENDDO
     !
#if KEY_DHDGB==1
!AP/MF
     IF (QFHDGB) THEN
         DO I=1,TOTALS
            FACT=DELTAS/SAMASS
            VS_DHDGB(I)=VS_DHDGB(I)-DS_DHDGB(I)*FACT
            SDEFLD(I)=VS_DHDGB(I)
         ENDDO
     ENDIF
#endif
#if KEY_BLOCK==1 /*ldm*/
     if(qldm) call ldm_prop2_dynamvv(nblock, delta)
#endif /*  LDM*/
     !
#if KEY_MTS==1
     IF (QTBMTS) THEN
        IF(NIT < NMTS1) GOTO 1111
        !
        DO I=1,LENENT
           EMTS(I)=ETERM(I)
        ENDDO
        do i=atfrst,atlast
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
              XMI(I)=DX(I)
              YMI(I)=DY(I)
              ZMI(I)=DZ(I)
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
        !==========================================================
        IF(NMTS2 >= 1) THEN
           ENE2=.TRUE.
           CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
           do i=atfrst,atlast
#if KEY_PARASCAL==1
              IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                 IF(IMTS(I) < 0) GOTO 279
                 IF (IMOVE(I) == 0) THEN
                    FACT=SS2X/AMASS(I)
                    VX(I)=VX(I)-DX(I)*FACT
                    VY(I)=VY(I)-DY(I)*FACT
                    VZ(I)=VZ(I)-DZ(I)*FACT
                    XLD(I)=VX(I)
                    YLD(I)=VY(I)
                    ZLD(I)=VZ(I)
                 ENDIF
279              CONTINUE
#if KEY_PARASCAL==1
              ENDIF
#endif 
           ENDDO
           !
           IF(NIT2 < NMTS2) GOTO 2222
           !
           DO I=1,LENENT
              EMTS(I)=EMTS(I)+ETERM(I)
           ENDDO
           do i=atfrst,atlast
#if KEY_PARASCAL==1
              IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                 XMM(I)=DX(I)
                 YMM(I)=DY(I)
                 ZMM(I)=DZ(I)
#if KEY_PARASCAL==1
              ENDIF
#endif 
           ENDDO
        ENDIF
        !
        !=================================================================
        ENE3=.TRUE.
        CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
        ECSUM=ZERO
        DO I=1,LENENT
           ETERM(I)=ETERM(I)+EMTS(I)
           ECSUM=ETERM(I)+ECSUM
        ENDDO
        EPROP(EPOT)=ECSUM
        !
        CALL GRAM( &
#if KEY_MTS==1
             XMI,YMI,ZMI,XMM,YMM,ZMM, &   
#endif
             DX,DY,DZ &
#if KEY_DHDGB==1
!AP/MF
            ,DS_DHDGB &
#endif

        )
        !
        do i=atfrst,atlast
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
              IF(IMOVE(I) == 0) THEN
                 FACT=0.5*SA2X/AMASS(I)
                 VX(I)=VX(I)-FACT*DX(I)
                 VY(I)=VY(I)-FACT*DY(I)
                 VZ(I)=VZ(I)-FACT*DZ(I)
                 XLD(I)=VX(I)
                 YLD(I)=VY(I)
                 ZLD(I)=VZ(I)
              ENDIF
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
     ENDIF
#endif 
     !
     !
     NNQ=0
8818 CONTINUE
     NNQ=NNQ+1
     !
     IF(QNOSE) THEN
        !        FACT=DEXP(-SA2X*SNHV)
        !        FACT=SA2X*SNHV
        do i=atfrst,atlast
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
              IF(IMOVE(I) == 0) THEN
                 J = INLCKP(I)
                 FACT=SA2X*SNHV(J)
                 VX(I)=XLD(I)-VX(I)*FACT
                 VY(I)=YLD(I)-VY(I)*FACT
                 VZ(I)=ZLD(I)-VZ(I)*FACT
              ENDIF
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
#if KEY_DHDGB==1
!AP/MF
        IF (QFHDGB) THEN
            DO I=1,TOTALS
               FACT=SA2X*SNHV(1)
               VS_DHDGB(I)=SDEFLD(I)-VS_DHDGB(I)*FACT
            ENDDO
        ENDIF
#endif
     ENDIF
     !
#if KEY_MTS==1
     IF (QTBMTS) THEN
        do i=atfrst,atlast
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
              IF(IMOVE(I) == 0) THEN
                 FACT=0.5*SA2X/AMASS(I)
                 VX(I)=VX(I)-FACT*DX(I)
                 VY(I)=VY(I)-FACT*DY(I)
                 VZ(I)=VZ(I)-FACT*DZ(I)
              ENDIF
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
     ENDIF
#endif 
     !
#if KEY_TSM==1
     ! This section is for conformational thermodynamic integration
     ! To work correctly the routines need to have the correct total forces
     ! corresponding to structure X,Y,Z in DX, DY, DZ. K. Kuczera
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
     IF(QNOSE) THEN
        !
        ! NOSE-HOOVER
        !
        TEMPI=ZERO
        DO I = 1,NOBL
           EPTKE1(I)=EPTKX(I)
           EPTKX(I)=ZERO
        ENDDO
        !
#if KEY_DHDGB==1
!AP/MF
        IF (QFHDGB) THEN
            DO I=1,TOTALS
               RAVL=SAMASS*VS_DHDGB(I)**2
               TEMPI=TEMPI+RAVL
               IF (QNOSE) THEN
                  EPTKX(1)=EPTKX(1)+RAVL
               ENDIF
            ENDDO
        ENDIF
#endif
        do i=atfrst,atlast
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif 
              IF(IMOVE(I) == 0) THEN
                 RAVL=AMASS(I)*(VX(I)**2+VY(I)**2+VZ(I)**2)
                 TEMPI=TEMPI+RAVL
                 J = INLCKP(I)
                 EPTKX(J)=EPTKX(J)+RAVL
              ENDIF
#if KEY_PARASCAL==1
           ENDIF
#endif 
        ENDDO
        !
#if KEY_PARALLEL==1
        GCARR(1) = TEMPI
        DO I = 1,NOBL
           GCARR(I+1) = EPTKX(I)
        ENDDO
        CALL GCOMB(GCARR,NOBL+1)
        TEMPI = GCARR(1)
        DO I = 1, NOBL
           EPTKX(I) = GCARR(I+1)
        ENDDO
#endif 
        !
        DO I=1,NOBL
           SNHF1=EPTKX(I)-NDGN(I)*KBOLTZ*RTMPR(I)
           SNHV(I)=SNHV1(I)+SA2X*SNHF1/SQM(I)
        ENDDO
        !-----------------------------------------------------
        ! following correction of constraint forces
        ! Usually these are very small. We can ignore these for
        ! the calculation of the velocity
        !-------------------------------------------------------
        !           IF(QHOLO) THEN
        !.##IF MTS
        !             IF(TBMTS) THEN
        !             FACT1=SA3X
        !             SAX=SA1X
        !             ELSE
        !.##ENDIF
        !             FACT1=DELTA2
        !             SAX=DELTA
        !.##IF MTS
        !             ENDIF
        !.##ENDIF
        !           NIT=0
        !9919      CONTINUE
        !          NIT=NIT+1
        !          DO I=1,NATOM
        !.##IF MTS
        !          IF (TBMTS.AND.(IMTS(I) > 0)) GOTO 199
        !.##ENDIF
        !          IF (IMOVE(I) == 0) THEN
        !          FACT=FACT1/AMASS(I)
        !          FACT2=FACT1*SNHV
        !          XNEW(I)=X(I)+SAX*VX(I)-DX(I)*FACT-FACT2*VX(I)
        !          YNEW(I)=Y(I)+SAX*VY(I)-DY(I)*FACT-FACT2*VY(I)
        !          ZNEW(I)=Z(I)+SAX*VZ(I)-DZ(I)*FACT-FACT2*VZ(I)
        !          ENDIF
        !199       CONTINUE
        !          ENDDO
        !       CALL VIRSHK(EPRESS(VIXX:VIZZ),NATOM,X,Y,Z,VX1,VY1,VZ1)
        !        CALL DYNASHK(VX,VY,VZ,XNEW,YNEW,ZNEW,AMASS,
        !     1   IMOVE,ISKP,NATOM,SAX,VX1,VY1,VZ1)
        !       CALL VIRSHK(EPRESS(VIXX:VIZZ),NATOM,X,Y,Z,VX1,VY1,VZ1)
        !        IF(NIT < 4) GOTO 9919
        !        ENDIF
        !----------------------------------------------------------
        QV2=ZERO
        DO I=1,NOBL
           QV2=QV2+ABS(EPTKE1(I)-EPTKX(I))
        ENDDO
        QV2=QV2/NOBL
        IF(QV2 <= TOLNS) GOTO 8919
        IF(NNQ < NSCYC) GOTO 8818
     ENDIF
     !
     !-------------------------------------------------------
8919 CONTINUE
     TEMPI=ZERO
     TOTKEN=ZERO
     IF(QNOSE) THEN
        DO I=1,NOBL
           EPTKX(I)=ZERO
        ENDDO
     ENDIF
     !
#if KEY_DHDGB==1
!AP/MF
     IF (QFHDGB) THEN
         DO I=1,TOTALS
            RAVL=SAMASS*VS_DHDGB(I)**2
            TEMPI=TEMPI+RAVL
            VK_DHDGB(I)=VK_DHDGB(I)+RAVL
            IF (QNOSE) THEN
                EPTKX(1)=EPTKX(1)+RAVL
            ENDIF
         ENDDO
     ENDIF
#endif
     do i=atfrst,atlast
#if KEY_PARASCAL==1
        IF(JPBLOCK(I) == MYNOD) THEN
#endif 
           IF(IMOVE(I) == 0) THEN
              RAVL=AMASS(I)*(VX(I)**2 + VY(I)**2 + VZ(I)**2)
              TEMPI=TEMPI+RAVL
              VK(I)=VK(I)+RAVL
              IF(QNOSE) THEN
                 J = INLCKP(I)
                 EPTKX(J)=EPTKX(J)+RAVL
              ENDIF
           ENDIF
#if KEY_PARASCAL==1
        ENDIF
#endif 
     ENDDO
     !
#if KEY_PARALLEL==1
     IF(QNOSE) THEN
        GCARR(1) = TEMPI
        DO I = 1,NOBL
           GCARR(I+1) = EPTKX(I)
        ENDDO
        CALL GCOMB(GCARR,NOBL+1)
        TEMPI = GCARR(1)
        DO I = 1, NOBL
           EPTKX(I) = GCARR(I+1)
        ENDDO
     ELSE
        GCARR(1) = TEMPI
        CALL GCOMB(GCARR,1)
        TEMPI = GCARR(1)
     ENDIF
#endif 
     !
     IF(QNOSE) THEN
        DO I = 1,NOBL
           SNHF(I)=EPTKX(I)-NDGN(I)*KBOLTZ*RTMPR(I)
           SNHV(I)=SNHV1(I)+SA2X*SNHF(I)/SQM(I)
        ENDDO
     ENDIF
     !
     !       . Calculate the pressures.
     CALL PRSEXT
     CALL PRSINT(NATOM,AMASS,VX,VY,VZ,ONE,VX,VX,VX,ZERO)
     ! APH: Note VX, VX, VX here -----------|
     !      are just dummies
     !
     QK1=ZERO
     QK2=ZERO
     IF(QNOSE) THEN
        DO I=1,NOBL
           QK1=QK1+0.5*SQM(I)*SNHV(I)*SNHV(I)
           QK2=QK2+NDGN(I)*KBOLTZ*RTMPR(I)*SNH(I)
        ENDDO
     ENDIF

#if KEY_CHEQ==1
     EPROP(CGKE) = HALF*KECG
#endif 

     !
     EPROP(TOTKE)=TEMPI/TWO
     EPROP(TOTE)=EPROP(EPOT)+EPROP(TOTKE)+QK2+QK1
     !
     IF(QNOSE) THEN
        EPROP(HFCTE)=EPROP(EPOT)+EPROP(TOTKE)
        EPROP(EHFC)=QK2+QK1
     ENDIF
     !
     !       CHECK THE CHANGE IN TOTAL ENERGY TO BE SURE EVERYTHING IS OK.
     !
     IF ((.NOT.QNOSE).AND.(JHSTRT > 2)) THEN
        IF (ABS(TOTEPR-EPROP(TOTE)) > MAX(ECHECK,0.1D0*EPROP &
             (TOTKE))) THEN
           IF (TOTEPR  /=  -9999.0) THEN
              WRITE (OUTU,2000) ECHECK,TOTEPR,EPROP(TOTE),EPROP(TOTKE)
2000          FORMAT(' TOTAL ENERGY CHANGE EXCEDED'/G12.2, &
                   ' KCAL AND 10% OF THE TOTAL KINETIC ENERGY IN THE ' &
                   //'LAST STEP'/ &
                   ' PREVIOUS EPROP(EPOT) =',G14.4,' CURRENT EPROP(EPOT)' &
                   //' =',G14.4, &
                   ' KINETIC =',G14.4)
#if KEY_PARALLEL==1
              CALL VDGBR(X,Y,Z,1)  
#endif
              !
              CALL WRNDIE(-2,'<DYNAMC>','ENERGY CHANGE TOLERENCE ' &
                   //'EXCEEDED')
           ENDIF
        ENDIF
     ENDIF
     !
     TOTEPR=EPROP(TOTE)
     TIME=TIMFAC*DELTA*NMTS0*NPRIV
     AKMATI=DELTA*NMTS0*NPRIV
     EPROP(TEMPS)=TEMPI/(KBOLTZ*NDEGF)
     !
     IF(QNOSE) THEN
        DO I=1,NOBL
           TMPN(I)=EPTKX(I)/(KBOLTZ*NDGN(I))
        ENDDO
     ENDIF
     !
     JHTEMP=JHTEMP+EPROP(TEMPS)
     call avfl_update(EPROP, ETERM, EPRESS)
     ! JG 5/2002
#if KEY_QUANTUM==1
     IF(QMPERT) THEN
        DO I = QPT1,QPT2
           EEQTRM(I) = EXP((ETERM(QMEL)-EEQTRM(I))*BETA_QMMM)
           EQPRA(I) = EQPRA(I)+EEQTRM(I)
           EQPR2A(I) = EQPR2A(I)+EEQTRM(I)*EEQTRM(I)
        ENDDO
     ENDIF
     IF(QDECOM) THEN
        !  QVER - QGAS are sequentially defined, quantm.f90, and ENERIN
        DO I = QVER,QGAS
           EQPRA(I) = EQPRA(I)+EEQTRM(I)
           EQPR2A(I) = EQPR2A(I)+EEQTRM(I)*EEQTRM(I)
        ENDDO
     ENDIF
     IF (CHDYN) THEN
        CALL DYNDEN('ACCU',ISTEP)
     END IF
#endif 
#if KEY_SQUANTM==1
     IF(QMFEP .or. QMSOLV) THEN
        DO I = QPT1,QPT2
           EEQTRM(I) = EXP((ETERM(QMEL)-EEQTRM(I))*beta_qmmm_fep)
           EQPRA(I)  = EQPRA(I)+EEQTRM(I)
           EQPR2A(I) = EQPR2A(I)+EEQTRM(I)*EEQTRM(I)
        END DO
     END IF
#endif 

     !
     ISTPSA=MOD(IST1,IPRFRQ)+ISTEP-IST1
     FITA=FITA+ISTPSA*EPROP(TOTE)
     FITP=FITP+JHSTRT*EPROP(TOTE)
     !

     !   DO CHARGE UPDATE

     !  COmpute new charges

#if KEY_CHEQ==1

     IF (QCG) THEN

#if KEY_PARALLEL==1
        !        IF (MYNOD == 0) THEN 
#endif

        IF (NOSEFLAG == 1) THEN


           CHEQNHMX  = 1000
           CHEQNHTL = 0.000001
           DO K = 1, NQBATHS
              KECGBATH(K) = 0.0
              DO L = IFIRSTBATH(K), ILASTBATH(K)
                 KECGBATH(K)=KECGBATH(K)+PMASSQ(L)*VCG(L)*VCG(L)
              ENDDO
              !          NDGFBATH(K) = ILASTBATH(K)-IFIRSTBATH(K) + 1
           ENDDO


           ! Do Nose-Hoover iterations if necessary
           IF (FQNHMBATH(1) /= ZERO) THEN
              DO NHITR=1,CHEQNHMX


                 DO K = 1,NQBATHS
                    CHEQTEMP=FQTEMPBATH(K)
                    CHEQCDGF = NDGFBATH(K) ! ILASTBATH(K)-IFIRSTBATH(K) + 1
                    FQNHABATH(K) = (KECGBATH(K)-(CHEQCDGF)*KBOLTZ*CHEQTEMP)/ &
                         FQNHMBATH(K)
                    FQNHSNBATH(K)=TWO*FQNHSBATH(K)-FQNHSOBATH(K) &
                         +DELTA**2*FQNHABATH(K)
                    NHSDIFBATH(K)=FQNHSNBATH(K)-FQNHSOBATH(K)

                 ENDDO


                 ! Propagate the charges by standard Verlet, including the calculated
                 ! scaling constant, and get a new estimate for mv**2

                 MAXERROR = -1000.0
                 MAXEOLD = MAXERROR
                 IMAX=0
                 DO K = 1, NQBATHS
                    MV2TMPBATH(K)=0.0
                    DO L = IFIRSTBATH(K),ILASTBATH(K)
                       CMASS=PMASSQ(L)
                       CGNEW(L)=(TWO*CG(L)-CGOLD(L)*(ONE-0.25d0*NHSDIFBATH(K))- &
                            DELTA**2*DCH(L)/CMASS)/(ONE+0.25d0*NHSDIFBATH(K))
                       VCG(L)=(CGNEW(L)-CGOLD(L))/(TWO*DELTA)
                       MV2TMPBATH(K)=  MV2TMPBATH(K) +CMASS*VCG(L)*VCG(L)
                    ENDDO

                    ERROR = ABS(MV2TMPBATH(K)-KECGBATH(K))
                    MAXERROR = MAX(ERROR,MAXERROR)
                    IF(MAXERROR /= MAXEOLD) IMAX=K
                    MAXEOLD = MAXERROR
                 ENDDO



                 ! If the new value of mv**2 is close enough to the old, we have reached
                 ! self-consistency and can continue. Otherwise, do another Nose iteration

                 IF  (MAXERROR < CHEQNHTL) THEN


                    ! Add the Nose-Hoover contributions to the energies
                    !                  IF (MOD(ISTEP,100) == 0) THEN
                    !                  WRITE(69,10) NHITR,CHEQNHS,NHSDIF/(TWO*DELTA),
                    !     &                            FQNHA,FQCMV2/TWO
                    !19                FORMAT(' FQCHP2> ',I4,' iterations; S= ',F9.3,
                    !     &                 ' dS/dt= ',F9.3,' d2S/dt2= ',F9.3,' KE=',F9.3)
                    !                ENDIF

                    KECG=ZERO
                    DO K=1,NQBATHS
                       KECGBATH(K) = MV2TMPBATH(K)
                       FQNHSOBATH(K) = FQNHSBATH(K)
                       FQNHSBATH(K) = FQNHSNBATH(K)
                       KECG = KECG + KECGBATH(K)
                    ENDDO
                    GOTO 1039
                 ENDIF

                 DO K = 1,NQBATHS
                    KECGBATH(K) = MV2TMPBATH(K)
                 ENDDO

              ENDDO
              CALL WRNDIE(-3,'<FQCNW2>', &
                   'Maximum Nose-Hoover iterations exceeded')
              !     &        KECG, KESOLUTE, KEWATER=',KECG,KECGSOL,MV2TMPWAT)
           ELSE

           ENDIF

           !  UPDATE CHARGES

1039       CONTINUE
           DO I = 1,NATOM
              CGOLD(I)=CG(I)
              CG(I) = CGNEW(I)
           ENDDO




        ELSEIF (NOSEFLAG == 2) THEN

           IF(QCG) THEN
              KECG=0.0
              DO I=1,NATOM

                 CGNEW(I) = 2.0*CG(I) - CGOLD(I) - &
                      (DELTA**2)*DCH(I)/PMASSQ(I)


                 VCG(I)=(CGNEW(I)-CGOLD(I))/(2.0*DELTA)

                 KECG=KECG+PMASSQ(I)*VCG(I)*VCG(I)

                 CGOLD(I)= CG(I)
                 CG(I)= CGNEW(I)

              ENDDO
           ENDIF


        ELSEIF (NOSEFLAG == 0) THEN
           !     do nothing
        ELSE
        ENDIF   !  CONDITIONAL ON NOSEFLAG



#if KEY_PARALLEL==1
        !         ENDIF 
#endif

     ELSE   ! CONDITIONAL ON QCG
     ENDIF    ! CONDITIONAL ON QCG


#if KEY_PARALLEL==1
     !      IF (MYNOD == 0) THEN
     IF (QCG) THEN
#if KEY_DOMDEC==1
        if (q_domdec) CALL WRNDIE(-5,'<DYNAMC>','NOT YET IMPLEMENTED FOR DOMDEC') 
#endif
        CALL VDGBRE(CG,IPARPT)
        CALL VDGBRE(VCG,IPARPT)
#if KEY_PARALLEL==1
        !        CALL VDGBRE(CG,IPARPT) 
#endif
#if KEY_PARALLEL==1
        !        CALL VDGBR(CG,cg,cg,0) 
#endif
        CALL CGCPY(CG)
     ENDIF
     !      ENDIF
#endif 
#endif /*    end conditional on CHEQ*/
     ! PJ 06/2005
#if KEY_PIPF==1
     IF (QPIPF .AND. QPFDYN) THEN
        CALL PFDYN(UINDO,UINDN,VUIND,PMASSU,DELTA)
     ENDIF
#endif 
     IF(NSAVV > 0) THEN
        IF(MOD(ISTEP,NSAVV) == 0) THEN
           CALL WRITCV(VX,VY,VZ, &
#if KEY_CHEQ==1
                VCG,QCG,                         & 
#endif
                NATOM,FREEAT,NFREAT,NPRIV,ISTEP,NDEGF, &
                SA1X,NSAVV,NSTEP,TITLEA,NTITLA,IUNVEL,.TRUE., &
                .FALSE., (/ 0 /), .FALSE., (/ ZERO /))
        ENDIF
     ENDIF
     !
     IF((IUNOS >= 0).AND.QNOSE) THEN
        IF(MOD(ISTEP,NSNOS) == 0) THEN
           WRITE(IUNOS,166) NPRIV,TIME,EPROP(TOTE)
           WRITE(IUNOS,167) (SNH(I),TMPN(I),I=1,NOBL)
167        FORMAT(2F16.6)
           !          machine dependent call to flush output buffers
           CALL GFLUSH(IUNOS)
166        FORMAT(I16,2F16.6)
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
#if KEY_BLOCK==1
     !.ab.Print out HybridH. Use the #BLOCK# (seems to be the right place...).
     IF(IOLEV >= 0) CALL PRHYBH()
     !.ab.
#endif 

     IF(MOD(ISTEP,NPRINT) == 0) THEN
        IF (PRNLEV >= 2) &
             CALL PRINTE(OUTU, EPROP, ETERM, 'DYNA', 'DYN', .FALSE., &
             ISTEP, TIME, ZERO, .TRUE.)
        !
        CALL WRETERM(NPRIV,TIME,QKUHEAD)
        !
     ENDIF

#if KEY_SCCDFTB==1
#if KEY_PARALLEL==1
     IF (MYNOD == 0) THEN
#endif 

        if(qsccb.and.(qlamda.or.qpkac)) then
           if((istep >= sccpass).and.(mod((istep-sccpass),sccstep) == 0)) &
                then
              icntdyn=icntdyn+1
              dvdltmp=  dvdlb+dvdla+dvdlp+dvdle+dvdlv+dvdlscc+dvdlub &
                   +dvdlip
              dvdl=dvdl+dvdltmp
! QC: UW_2017: correct for a minor bug (reported by Puja)
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
#if KEY_PARALLEL==1
     ENDIF
#endif 

#endif 
  ENDDO
  ! Main loop End
  !----------------------------------------------------------------
  !----------------------------------------------------------------
  !
#if KEY_PARALLEL==1
  CALL VDGBR(VX,VY,VZ,1)        
#endif
#if KEY_PARALLEL==1
  CALL VDGBR(XOLD,YOLD,ZOLD,1)  
#endif
  !
#if KEY_BLOCK==1 /*ldm*/
  !     reset lambdas to time t: the reason is that at the end of
  !     dcntrl.src, when ORIG is true, x is reset to xold
  !     set BLDOLD to bxlamb(t+dt) for restart file
  if(qldm) call ldm_reset_dynam(nblock)
#endif /*  LDM*/
  IF (ISTPSA < IPRFRQ) RETURN
  !
  !     Now do long time fluctuations
  !
  call avfl_update_lt()
  ! JG 5/2002
#if KEY_QUANTUM==1
  IF(QMPERT) THEN
     EQPRP(QPT1) = EQPRP(QPT1)+EQPRA(QPT1)
     EQPR2P(QPT1) = EQPR2P(QPT1)+EQPR2A(QPT1)
     EQPRP(QPT2) = EQPRP(QPT2)+EQPRA(QPT2)
     EQPR2P(QPT2) = EQPR2P(QPT2)+EQPR2A(QPT2)
  ENDIF
  IF(QDECOM) THEN
     !  QVER - QGAS are sequentially defined, quantm.f90, and ENERIN
     DO I = QVER,QGAS
        EQPRP(I) = EQPRP(I)+EQPRA(I)
        EQPR2P(I) = EQPR2P(I)+EQPR2A(I)
     ENDDO
  ENDIF
#endif 
#if KEY_SQUANTM==1
  IF(QMFEP .OR. QMSOLV) THEN
     EQPRP(QPT1)  = EQPRP(QPT1) +EQPRA(QPT1)
     EQPR2P(QPT1) = EQPR2P(QPT1)+EQPR2A(QPT1)
     EQPRP(QPT2)  = EQPRP(QPT2) +EQPRA(QPT2)
     EQPR2P(QPT2) = EQPR2P(QPT2)+EQPR2A(QPT2)
  END IF
#endif 
!!#if KEY_MNDO97==1
!!  qm_control_r%md_run =.false.        ! end of MD run.
!!#endif
  !
  NUMSTP = ISTPSA
  DNUM   = NUMSTP
  DNM1   = DNUM - ONE
  DNP1   = DNUM + ONE
  DNPM1  = DNM1 * DNP1
  IF(NUMSTP > 1) THEN
     DRIFTA = (TWO*FITA-DNP1*EPRPA(TOTE))*SIX/(DNPM1*DNUM)
     EAT0A  = EPRPA(TOTE)/DNUM-DRIFTA*DNP1/TWO
     IF ((EPRP2A(TOTE)/DNUM-(EPRPA(TOTE)/DNUM)**2) > ZERO) THEN
        CORRA=DNPM1/(EPRP2A(TOTE)/DNUM-(EPRPA(TOTE)/DNUM)**2)/TWELVE
        CORRA=DRIFTA*SQRT(CORRA)
     ELSE
        CORRA=ZERO
     ENDIF
     !
     !       . Compute statistics.
     call avfl_compute(DNUM)
     ! JG 5/2002
#if KEY_QUANTUM==1
     IF(QMPERT) THEN
        DO I = QPT1,QPT2
           EQPRA(I) = EQPRA(I) / DNUM
           FLUCTD = EQPR2A(I)/DNUM - EQPRA(I)**2
           EQPR2A(I) = ZERO
           IF(FLUCTD > ZERO) EQPR2A(I) = SQRT(FLUCTD)
           IF(EQPRA(I) > ZERO) &
                EQPRA(I) = -LOG(EQPRA(I))/BETA_QMMM
           IF(EQPR2A(I) > ZERO) &
                EQPR2A(I) = -LOG(EQPR2A(I))/BETA_QMMM
        ENDDO
     ENDIF
     IF(QDECOM) THEN
        !  QVER - QGAS are sequentially defined, quantm.f90, and ENERIN
        DO I = QVER,QGAS
           EQPRA(I) = EQPRA(I) / DNUM
           FLUCTD = EQPR2A(I)/DNUM - EQPRA(I)**2
           EQPR2A(I) = ZERO
           IF(FLUCTD > ZERO) EQPR2A(I) = SQRT(FLUCTD)
        ENDDO
     ENDIF
#endif 
#if KEY_SQUANTM==1
     IF(QMFEP .OR. QMSOLV) THEN
        DO I = QPT1,QPT2
           EQPRA(I)  = EQPRA(I) / DNUM
           FLUCTD    = EQPR2A(I)/DNUM - EQPRA(I)**2
           EQPR2A(I) = ZERO
           IF(FLUCTD > ZERO) EQPR2A(I) = SQRT(FLUCTD)
           IF(EQPRA(I) > ZERO) &
                EQPRA(I) = -LOG(EQPRA(I))/beta_qmmm_fep
           IF(EQPR2A(I) > ZERO) &
                EQPR2A(I) = -LOG(EQPR2A(I))/beta_qmmm_fep
        END DO
     END IF
#endif 
     !
  ENDIF
  !
  !. Print out the results
  IF(PRNLEV  >=  2) THEN
     call avfl_print_aver(NUMSTP, TIME)
     call avfl_print_fluc(NUMSTP, TIME)
     ! JG 5/2002
#if KEY_QUANTUM==1
     IF(QMPERT.OR.QDECOM) THEN
        WRITE(OUTU,'(A,I8,A)') &
             ' QMPERT> QM FEP for the last ',NUMSTP,' steps:'
        CALL PRQFEP(OUTU,NUMSTP,TIME)
     ENDIF
     IF (CHDYN) THEN
        CALL DYNDEN('PRIN',ISTEP)
     END IF
#endif 
#if KEY_SQUANTM==1
     IF(QMFEP .OR. QMSOLV) THEN
        IF(QMFEP) THEN
           WRITE(OUTU,'(A,I8,A)') &
                ' QMFEP > QM FEP for the last ',NUMSTP,' steps:'
        ELSE IF(QMSOLV) THEN
           WRITE(OUTU,'(A,I8,A)') &
                ' QMSOLV> QM FEP for the last ',NUMSTP,' steps:'
        END IF
        CALL PRQFEP_SQ(NUMSTP)
     END IF
#endif 
  ENDIF
  !
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
     IF ((EPRP2P(TOTE)/DNUM-(EPRPP(TOTE)/DNUM)**2) > ZERO) THEN
        CORRP=DNPM1/(EPRP2P(TOTE)/DNUM-(EPRPP(TOTE)/DNUM)**2)/TWELVE
        CORRP=DRIFTP*SQRT(CORRP)
     ELSE
        CORRP=ZERO
     ENDIF
  ENDIF
  call avfl_compute_lt(DNUM)
  ! JG 5/2002
#if KEY_QUANTUM==1
  IF(QMPERT) THEN
     DO I = QPT1,QPT2
        EQPRA(I) = EQPRP(I) / DNUM
        FLUCTD = EQPR2P(I)/DNUM - EQPRA(I)**2
        EQPR2A(I) = ZERO
        IF(FLUCTD > ZERO) EQPR2A(I) = SQRT(FLUCTD)
        IF(EQPRA(I) > ZERO) &
             EQPRA(I) = -LOG(EQPRA(I))/BETA_QMMM
        IF(EQPR2A(I) > ZERO) &
             EQPR2A(I) = -LOG(EQPR2A(I))/BETA_QMMM
     ENDDO
  ENDIF
  IF(QDECOM) THEN
     DO I = QVER,QGAS
        EQPRA(I) = EQPRP(I) / DNUM
        FLUCTD = EQPR2P(I)/DNUM - EQPRA(I)**2
        EQPR2A(I) = ZERO
        IF(FLUCTD > ZERO) EQPR2A(I) = SQRT(FLUCTD)
     ENDDO
  ENDIF
#endif 
#if KEY_SQUANTM==1
  IF(QMFEP .OR. QMSOLV) THEN
     DO I = QPT1,QPT2
        EQPRA(I)  = EQPRP(I) / DNUM
        FLUCTD    = EQPR2P(I)/DNUM - EQPRA(I)**2
        EQPR2A(I) = ZERO
        IF(FLUCTD > ZERO) EQPR2A(I) = SQRT(FLUCTD)
        IF(EQPRA(I) > ZERO) &
             EQPRA(I) = -LOG(EQPRA(I))/beta_qmmm_fep
        IF(EQPR2A(I) > ZERO) &
             EQPR2A(I) = -LOG(EQPR2A(I))/beta_qmmm_fep
     END DO
  END IF
#endif 
  !
  !. Print out results
  IF(PRNLEV >= 2) THEN
     call avfl_print_aver_lt(NUMSTP, TIME)
     call avfl_print_fluc_lt(NUMSTP, TIME)
     ! JG 5/2002
#if KEY_QUANTUM==1
     IF(QMPERT.OR.QDECOM) THEN
        WRITE (OUTU,'(A,I8,A,F9.1,A)') &
             ' QMPERT> QM FEP for the last ', NUMSTP,' steps:(', &
             DNUM,' )'
        WRITE(OUTU,'(A,I8,A)') &
             ' QMPERT> QM FEP for the last ',NUMSTP,' steps:'
        CALL PRQFEP(OUTU,NUMSTP,TIME)
     ENDIF
#endif 
#if KEY_SQUANTM==1
     IF(QMFEP .OR. QMSOLV) THEN
        IF(QMFEP) THEN
           WRITE(OUTU,'(A,I8,A,F9.1,A)') &
                ' QMFEP > QM FEP for the last ', NUMSTP,' steps:(',DNUM,' )'
        ELSE IF(QMSOLV) THEN
           WRITE(OUTU,'(A,I8,A,F9.1,A)') &
                ' QMSOLV> QM FEP for the last ', NUMSTP,' steps:(',DNUM,' )'
        END IF
        CALL PRQFEP_SQ(NUMSTP)
     END IF
#endif 
     !
  ENDIF
  !
190 CONTINUE
  IF(NUMSTP <= 1 .OR. JHSTRT.LE.1 .OR. PRNLEV < 3) RETURN
  !
  WRITE(OUTU,195)DRIFTA,DRIFTP,EAT0A,EAT0P,CORRA,CORRP
195 FORMAT(/5X,'DRIFT/STEP (LAST-TOTAL): ',1P,2G17.8, &
       /5X,'EPROP(EPOT) AT STEP 0  : ',1P,2G17.8, &
       /5X,'CORR. COEFFICIENT      : ',1P,2G17.8)
  RETURN
END SUBROUTINE DYNAMVV

