module dynio
  use chm_kinds
  use number
  implicit none

  real(chm_real), parameter :: uind0(3,1) &
       = reshape((/ ZERO, ZERO, ZERO /), (/ 3, 1 /))

contains

SUBROUTINE WRXYZ(IUN,X,Y,Z,VX,VY,VZ,DX,DY,DZ,ISTEP)
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use consta
  use number
  use energym
  use reawri
  use stream
#if KEY_MULTICOM==1 /*  VO stringm */
  use multicom_aux   
#endif
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*),VX(*),VY(*),VZ(*),DX(*),DY(*),DZ(*)
  INTEGER IUN,ISTEP,I
  real(chm_real) T,U
  !
  if (IUN < 0 .OR. (IOLEV.LT.0 &
#if KEY_MULTICOM==1 /*  VO stringm */
 &                             .and.(ME_LOCAL.ne.0)    &      
#endif
 &                             )) RETURN
  IF(MXYZ <  0) RETURN
  !
  IF(MXYZ >  0) THEN 
  WRITE(IUN,'(I5)')NATOM
  write(IUN,'(5E25.15)') ISTEP*DELTA*TIMFAC, &
       EPROP(TOTE),EPROP(TOTKE),EPROP(EPOT),EPROP(TEMPS)
  ELSE 
    IF(ISTEP.EQ.1) THEN 
      WRITE(IUN,'(A)') '!     Step #            Time                     Etot                    Etotke                    Epot                    Temp'
    ENDIF 
  write(IUN,'(I12,5E25.15)') ISTEP,ISTEP*DELTA*TIMFAC, &
       EPROP(TOTE),EPROP(TOTKE),EPROP(EPOT),EPROP(TEMPS)
  ENDIF 
  DO I=1,NATOM
     IF(MXYZ == 1) &
          WRITE(IUN,'(A2,1X,3E25.15)') ATYPE(I)(1:2),X(I),Y(I),Z(I)
     IF(MXYZ == 2) &
          WRITE(IUN,'(A2,1X,6E25.15)') ATYPE(I)(1:2),X(I),Y(I),Z(I), &
          VX(I),VY(I),VZ(I)
     IF(MXYZ == 3) &
          WRITE(IUN,'(A2,1X,9E25.15)') ATYPE(I)(1:2),X(I),Y(I),Z(I), &
          VX(I),VY(I),VZ(I),DX(I),DY(I),DZ(I)
     IF(MXYZ == 4) &
          WRITE(IUN,'(A2,1X,6E25.15)') ATYPE(I)(1:2),X(I),Y(I),Z(I), &
          DX(I),DY(I),DZ(I)
  ENDDO
  !
  RETURN
END SUBROUTINE WRXYZ


SUBROUTINE WRIDYN(U,NATOM,XOLD,YOLD,ZOLD,X,Y,Z,VX,VY,VZ, &
#if KEY_CHEQ==1
     CG,CGOLD,VCG,QCG,                     & 
#endif
#if KEY_PIPF==1
     UIND,UINDO,VUIND,QPFDYN,              & 
#endif
#if KEY_PIPF==1
     NPFBATHS,PFNHSBATH,PFNHSOBATH,        & 
#endif
#if KEY_DYNVV2==1
     QCONSTRAINTS,DXC,DYC,DZC,             & 
#endif
     NPRIV,JHSTRT,NDEGF,NSTEP, &
     NSAVC,NSAVV,SEED,AVETEM,ISTPSA,LDYNA &
#if KEY_BLOCK==1
     ,QLMC,QLDM,NBLOCK,BXLAMB,BLDOLD,BVLAMB,NSAVL &    /*ldm*/
#endif
#if KEY_FOURD==1
     ,VFD,FDOLD                            & 
#endif
#if KEY_SCCDFTB==1
     ,qlamda,qpkac,icntdyn,iavti,dvdl,dvdlav,dtmp1 &  
#endif
#if KEY_DHDGB==1
!AP/MF
     ,QFHDGB,TOTALS,SDEFOLD,SDEF,VS_DHDGB &
#endif
     )
  !
  ! Write a formatted restart file with a maximum record length
  ! of 80 characters. For floating point numbers the format
  ! D22.15 was chosen. This should give enough precision in most
  ! cases.  Only the absolute minimum of information is written
  ! to the restart files.
  !
  ! Axel Brunger, 30-SEP-1984
  ! LDM code added by James Kong, 22-JUL-1996
  ! =========================
  !
  use chm_kinds
  use dimens_fcm
  ! input/output
  use energym
  use averfluc
  use avfl_ucell
  use fourdm
  use stream
  use ctitla
  use image
  use version
  use nose_mod
  use pathm
  use replica_ltm
  use parallel
  use repdstr
  use string
  use rndnum
  use clcg_mod
  use memory
#if KEY_MULTICOM==1 /*  VO stringm */
  use multicom_aux    
#endif
#if KEY_BLOCK==1
  use lambdam,only : qmld, msld_write_restart  /*ldm*/
#endif
#if KEY_FLUCQ==1
  use flucq
#endif 
  use phmd

  implicit none

#if KEY_DHDGB==1
!AP/MF
  INTEGER,optional ::  TOTALS
  REAL(chm_real),optional :: SDEFOLD(*),SDEF(*),VS_DHDGB(*)
  LOGICAL,optional :: QFHDGB
  CHARACTER(len=4) DHDG
#endif
  INTEGER NATOM, U
  real(chm_real) XOLD(*), YOLD(*), ZOLD(*), X(*), Y(*), Z(*)
  real(chm_real) VX(*), VY(*), VZ(*)
#if KEY_CHEQ==1
  real(chm_real) CG(*),CGOLD(*),VCG(*)
  LOGICAL QCG
#endif 
  ! PJ 06/2005
#if KEY_PIPF==1
  real(chm_real) UIND(3,*),UINDO(3,*),VUIND(3,*)
  INTEGER J
  LOGICAL QPFDYN
  INTEGER NPFBATHS
  real(chm_real) PFNHSBATH(*),PFNHSOBATH(*)
#endif 
#if KEY_DYNVV2==1
  LOGICAL QCONSTRAINTS                      
#endif
#if KEY_DYNVV2==1
  real(chm_real) DXC(*), DYC(*), DZC(*)             
#endif
  INTEGER NPRIV, JHSTRT, NDEGF, NSTEP, NSAVC, NSAVV
  real(chm_real) AVETEM,SEED
  INTEGER ISTPSA, LDYNA
  character(len=4)  NSED
  integer,allocatable,dimension(:) :: lrngseed,allseeds
  integer rngstream,nrandl,numnodl,countl
  character(len=64) :: seed_format
  character(len=10) :: seed_format_cpus
#if KEY_BLOCK==1 /*ldm*/
  logical qlmc, qldm
  INTEGER NBLOCK, NSAVL
  real(chm_real) BXLAMB(:), BVLAMB(:), BLDOLD(:)
#else /*   ldm*/
  INTEGER NSAVL
  PARAMETER (NSAVL=0)
#endif /*  ldm*/
#if KEY_FOURD==1
  real(chm_real) VFD(*), FDOLD(*)
#endif 
#if KEY_SCCDFTB==1
  logical qlamda,qpkac
  integer icntdyn, iavti
  real(chm_real) dvdl,dvdlav,dtmp1
#endif 
!
  ! local
  INTEGER I
  ! begin
  NSED='    '
  IF(QNOSE) NSED='NOSE'
#if KEY_DHDGB==1
!AP/MF
  DHDG='    '
  IF(PRESENT(QFHDGB)) THEN
    IF(QFHDGB) DHDG='DHDG'
  ENDIF
#endif
  !
  ! get the seeds to write in this file first.
  !
  ! This code is before iolev < 0, since in parallel we need 
  ! seeds from every CPU
  !
#if KEY_PARALLEL==1
  numnodl=numnod
#else
  numnodl=1
#endif
  !
  if((rngchoice==0).or.qoldrandom.or.qbrokenclcg)then
     nrandl=1
     call chmalloc('dynio.src','WRIDYN','lrngseed',nrandl,intg=lrngseed)
     lrngseed(1) = seed ! hope this is OK, mixup real vs. integer!
  elseif (rngchoice==1) then
     rngstream=1 ! check this one!! maybe we are using 2??
     nrandl=nrand
     call chmalloc('dynio.src','WRIDYN','lrngseed',nrandl,intg=lrngseed)
     call getseed(rngstream,lrngseed)
  elseif (rngchoice==2) then
     nrandl=nrand
     call chmalloc('dynio.src','WRIDYN','lrngseed',nrandl,intg=lrngseed)
     call random_seed(get=lrngseed)
  else
     nrandl=1 ! to make chmdealloc always happy
     call chmalloc('dynio.src','WRIDYN','lrngseed',nrandl,intg=lrngseed)
  endif
  ! For parallel collect from all the CPU's
   call chmalloc('dynio.src','WRIDYN','allseeds',nrandl*numnodl,intg=allseeds)
#if KEY_PARALLEL==1
   allseeds=0
   allseeds(mynod*nrandl+1:mynodp*nrandl)=lrngseed(1:nrandl)
   call igcomb(allseeds,nrandl*numnodl)
#else
   allseeds(1:nrandl)=lrngseed(1:nrandl)
#endif
  !
  !----------  Return if not an i/o node -----------------------------------
!MFC-- this code probably no longer needed, will remove with ensemble convert done
!-- ##IFN ENSEMBLE (ens_case)
  ! for not parallel ensemble, don't return, all nodes write output.
#if KEY_REPDSTR==1 /*repdstr*/
  if(qrepdstr) then
     if(mynod > 0) return
  else
#endif /* (repdstr)*/
     if(iolev < 0 &
#if KEY_MULTICOM==1 /*  VO stringm */
  &                .and. (ME_LOCAL.NE.0) &           
#endif
  &                ) return
#if KEY_REPDSTR==1  /*repdstr*/
  endif                            
#endif
!-- ##ELSE (ens_case)
!--   ! for parallel ensemble, only masters of each ensemble member write output.
#if KEY_PARALLEL==1
!--   if (iolev < 0 ) return     
#endif
!-- ##ENDIF (ens_case)



  if (reallow) then   
     REWIND(UNIT=U)
  endif 
#if KEY_DHDGB==1
!AP/MF
  WRITE(U,'(A4,2I6,2X,A4,2X,A4,A4)') &
       'REST',VERNUM,LDYNA,XTLTYP,NSED,DHDG
#else /**/
  WRITE(U,'(A4,2I6,2X,A4,2X,A4)') &
       'REST',VERNUM,LDYNA,XTLTYP,NSED
#endif          
  !
!  WRITE(U,'(A4,2I6,2X,A4,2X,A4)') &
!       'REST',VERNUM,LDYNA,XTLTYP,NSED
  CALL WRTITL(TITLEA,NTITLA,0,+2)
  WRITE(U,'(/I8,A)') NTITLA,' !NTITLE followed by title'
  WRITE(U,'(A)') (TITLEA(I),I=1,NTITLA)
  !
  !
#if KEY_DHDGB==1
!AP/MF
  IF(PRESENT(QFHDGB)) THEN
    IF(QFHDGB) THEN
      WRITE(U,'(/A)') ' !DHDGB PARAMETERS'
      WRITE(U,'(3D22.15)') (SDEFOLD(I),SDEF(I),VS_DHDGB(I),I=1,TOTALS)
    ENDIF
  ENDIF
!  IF(QFHDGB) THEN
!    WRITE(U,'(/A)') ' !DHDGB PARAMETES'
!    WRITE(U,'(3D22.15)') (SDEFOLD(I),SDEF(I), &
!                          VS_DHDGB(I),I=1,TOTALS)
!  ENDIF


#endif
  IF(QNOSE) THEN
     WRITE(U,'(/A)') ' !NOSE-HOOVER PARAMETES'
     WRITE(U,'(3D22.15)') (SN11(I),SN12(I),SN13(I),I=1,MAXNOS)
     WRITE(U,'(3D22.15)') (SNH(I),SNHV(I),SNHF(I),I=1,MAXNOS)
     ! BEGIN TPCONTROL (G. Lamoureux)
     !     IF (QPCONFULL) THEN
     WRITE(U,'(3D22.15)') (ETA_CP(1,I),I=1,3)
     WRITE(U,'(3D22.15)') (ETA_CP(2,I),I=1,3)
     WRITE(U,'(3D22.15)') (ETA_CP(3,I),I=1,3)
     WRITE(U,'(3D22.15)') (ZETA_CP(1,I),I=1,3)
     WRITE(U,'(3D22.15)') (ZETA_CP(2,I),I=1,3)
     WRITE(U,'(3D22.15)') (ZETA_CP(3,I),I=1,3)
     WRITE(U,'(3D22.15)') (G_CP(1,I),I=1,3)
     WRITE(U,'(3D22.15)') (G_CP(2,I),I=1,3)
     WRITE(U,'(3D22.15)') (G_CP(3,I),I=1,3)
     !     ELSE
     !        WRITE(U,'(3D22.15)') ETA_CP(1,1),ZETA_CP(1,1),G_CP(1,1)
     !     ENDIF
     WRITE(U,'(3D22.15)') (RX1_CP(1,I),I=1,3)
     WRITE(U,'(3D22.15)') (RX1_CP(2,I),I=1,3)
     WRITE(U,'(3D22.15)') (RX1_CP(3,I),I=1,3)
     WRITE(U,'(3D22.15)') (RX2_CP(1,I),I=1,3)
     WRITE(U,'(3D22.15)') (RX2_CP(2,I),I=1,3)
     WRITE(U,'(3D22.15)') (RX2_CP(3,I),I=1,3)
     WRITE(U,'(3D22.15)') (RV1_CP(1,I),I=1,3)
     WRITE(U,'(3D22.15)') (RV1_CP(2,I),I=1,3)
     WRITE(U,'(3D22.15)') (RV1_CP(3,I),I=1,3)
     WRITE(U,'(3D22.15)') (RV2_CP(1,I),I=1,3)
     WRITE(U,'(3D22.15)') (RV2_CP(2,I),I=1,3)
     WRITE(U,'(3D22.15)') (RV2_CP(3,I),I=1,3)
     ! END TPCONTROL (G. Lamoureux)
  ENDIF
  !
  IF(XTLTYP /= '    ') THEN
     WRITE(U,'(/A)') ' !CRYSTAL PARAMETERS'
     WRITE(U,'(3D22.15)') XTLABC
     WRITE(U,'(3D22.15)') HDOT
     WRITE(U,'(3D22.15)') PNH, PNHV, PNHF
     WRITE(U,'(3D22.15)') UC1A
     WRITE(U,'(3D22.15)') UC2A
     WRITE(U,'(3D22.15)') UC1B
     WRITE(U,'(3D22.15)') UC2B
     WRITE(U,'(3D22.15)') GRAD1A, GRAD1B, GRAD2A, GRAD2B
  ENDIF
  !
  WRITE(U,'(/A)') &
       ' !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL'
  !
  ! write collected seeds here....
  ! string format is needed
  write(seed_format_cpus,'(i10)')nrandl*numnodl
  countl=10
  call trima(seed_format_cpus,countl)
  seed_format='(7I12,D22.15,I12,2I22,'//seed_format_cpus(1:countl)//'I22)'
  WRITE(U,seed_format) &
       NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL,numnodl, &
       nrandl,allseeds(1:nrandl*numnodl)
  !
  call chmdealloc('dynio.src','WRIDYN','lrngseed',nrandl,intg=lrngseed)
  call chmdealloc('dynio.src','WRIDYN','allseeds',nrandl*numnodl,intg=allseeds)
  !
  ! write current flags, energies and statistics
  WRITE(U,'(/A)') ' !ENERGIES and STATISTICS'
  WRITE(U,'(128L1)') (QEPROP(I), I = 1,LENENP)
  WRITE(U,'(128L1)') (QETERM(I), I = 1,LENENT)
  WRITE(U,'(I8,3D22.15)') ISTPSA,FITA,FITP,AVETEM
  WRITE(U,'(3D22.15)') (EPROP(I),EPRPP(I),EPRP2P(I),I=1,LENENP)
  WRITE(U,'(2D22.15)') (EPRPA(I),EPRP2A(I),I=1,LENENP)
  WRITE(U,'(3D22.15)') (ETERM(I),ETRMP(I),ETRM2P(I),I=1,LENENT)
  WRITE(U,'(2D22.15)') (ETRMA(I),ETRM2A(I),I=1,LENENT)
  WRITE(U,'(3D22.15)') (EPRESS(I),EPRSP(I),EPRS2P(I),I=1,LENENV)
  WRITE(U,'(2D22.15)') (EPRSA(I),EPRS2A(I),I=1,LENENV)
  !
  ! write positions and velocities
#if KEY_FOURD==1 /*4dwrite*/
  IF(DIM4) THEN
     WRITE(U,'(/A)')  ' !XOLD, YOLD, ZOLD, FDOLD'
     WRITE(U,'(3D22.15)') &
          (XOLD(I),YOLD(I),ZOLD(I),FDOLD(I),I=1,NATOM)
     WRITE(U,'(/A)') ' !VX, VY, VZ, VFD'
     WRITE(U,'(3D22.15)') (VX(I),VY(I),VZ(I),VFD(I),I=1,NATOM)
     WRITE(U,'(/A)') ' !X, Y, Z, FDIM'
     WRITE(U,'(3D22.15)') (X(I),Y(I),Z(I),FDIM(I),I=1,NATOM)
  ELSE
#endif /* (4dwrite)*/
     WRITE(U,'(/A)')  ' !XOLD, YOLD, ZOLD'
     WRITE(U,'(3D22.15)') &
          (XOLD(I),YOLD(I),ZOLD(I),I=1,NATOM)
     WRITE(U,'(/A)') ' !VX, VY, VZ'
     WRITE(U,'(3D22.15)') (VX(I),VY(I),VZ(I),I=1,NATOM)
     WRITE(U,'(/A)') ' !X, Y, Z'
     WRITE(U,'(3D22.15)') (X(I),Y(I),Z(I),I=1,NATOM)
#if KEY_FOURD==1 /*4dendif*/
  ENDIF
#endif /* (4dendif)*/
#if KEY_DYNVV2==1 /*dynvv2_write*/
  ! BEGIN DYNA VV2 (G. Lamoureux)
  IF (QCONSTRAINTS) THEN
     WRITE(U,'(/A)') ' !DXC, DYC, DZC'
     WRITE(U,'(3D22.15)') (DXC(I),DYC(I),DZC(I),I=1,NATOM)
  ENDIF
  ! END DYNA VV2 (G. Lamoureux)
#endif /* (dynvv2_write)*/
#if KEY_FLUCQ==1
  IF(QFLUC) CALL FQRWRI(U)   
#endif
#if KEY_PHMD==1
  IF(QPHMD) CALL PHMDWRIT(U) 
#endif
#if KEY_BLOCK==1 /*ldm*/
  if (qmld) then
     call msld_write_restart(u,nblock)
  elseIF(QLDM .or.qlmc) THEN
     WRITE(U,'(/A)')  ' !LAMBDAOLD'
     WRITE(U,'(1D22.15)') (BLDOLD(I),I=1,NBLOCK)
     WRITE(U,'(/A)')  ' !LAMBDA_V'
     WRITE(U,'(1D22.15)') (BVLAMB(I),I=1,NBLOCK)
     WRITE(U,'(/A)')  ' !LAMBDA'
     WRITE(U,'(1D22.15)') (BXLAMB(I),I=1,NBLOCK)
  ENDIF
#endif /* ldm*/
#if KEY_CHEQ==1
  IF (QCG) WRITE(U,'(/A)') ' !CG,CGOLD,VCG'
  IF (QCG) WRITE(U,'(3D22.15)') &
       (CG(I),CGOLD(I),VCG(I),I=1,NATOM)
#endif 
  ! PJ 06/2005
#if KEY_PIPF==1
  IF (QPFDYN) WRITE(U,'(/A)') ' !UIND,UINDO,VUIND'
  IF (QPFDYN) THEN
     DO I = 1, NATOM
        WRITE(U,'(3D22.15)') &
             (UIND(J,I),UINDO(J,I),VUIND(J,I),J=1,3)
     ENDDO
  ENDIF
  IF (QPFDYN) WRITE(U,'(/A)') ' !PFNHSBATH, PFNHSOBATH'
  IF (QPFDYN) THEN
     DO I = 1, NPFBATHS
        WRITE(U,'(2D22.15)') PFNHSBATH(I),PFNHSOBATH(I)
     ENDDO
  ENDIF
#endif 
#if KEY_SCCDFTB==1
  IF(qlamda.or.qpkac) THEN
     WRITE(U,'(/A)') ' !SCCDFTB-TI Parameters'
     WRITE(U,'(/A)') ' !ICNTDYN,IAVTI'
     WRITE(U,'(I8,I8)') icntdyn,iavti
     WRITE(U,'(/A)')  ' !dvdl,dvdlav,dtmp1'
     WRITE(U,'(3D22.15)') dvdl,dvdlav,dtmp1
  ENDIF
#endif 
#if KEY_REPLICA==1
#if KEY_RPATH==1
  IF(QPROPT.AND.(NREPL > 0))THEN
     WRITE(U,'(/A)') ' !OFF-PATH PMF,FLUC'
     WRITE(U,'(I12)')NPCALL
     WRITE(U,'(3D22.15)')(PMF(I),I=1,NREPL)
     WRITE(U,'(3D22.15)')(FLUC(I),I=1,NREPL)
  ENDIF
#endif 
#endif 
  !
  ! ready
  !
  !++  LN MOD /APR 90
  !     Make sure everything is put on disk (which is needed on some
  !     machines in case of a job crash
  CALL SAVEIT(U)
  !--

  IF(PRNLEV >= 2) WRITE(OUTU,'(A,I8)') &
       ' WRIDYN: RESTart file was written at step', NPRIV
  !
  RETURN
END SUBROUTINE WRIDYN

SUBROUTINE READYN(U,NATOM,XOLD,YOLD,ZOLD,X,Y,Z,VX,VY,VZ, &
#if KEY_CHEQ==1
     CG,CGOLD,VCG,QCG,                     & 
#endif
#if KEY_PIPF==1
     UIND,UINDO,VUIND,QPFDYN,              & 
#endif
#if KEY_PIPF==1
     NPFBATHS,PFNHSBATH,PFNHSOBATH,        & 
#endif
#if KEY_DYNVV2==1
     QCONSTRAINTS,DXC,DYC,DZC,             & 
#endif
     NPRIV, &
     JHSTRT,NDEGF,NSTEP,NSAVC,NSAVV,SEED,AVETEM, &
     ISTPSA,LDYNA &
#if KEY_BLOCK
     ,QLMC,QLDM,NBLOCK,BXLAMB,BLDOLD,BVLAMB,NSAVL &   !BLOCK  !ldm
#endif
#if KEY_FOURD==1
     ,VFD,FDOLD                            & 
#endif
#if KEY_SCCDFTB==1
     ,qlamda,qpkac,qsccres,icntdyn,iavti,dvdl,dvdlav,dtmp1 &   
#endif
#if KEY_DHDGB==1
!AP/MF
     ,QFHDGB,TOTALS,SDEFOLD,SDEF,VS_DHDGB  &
#endif
     )
  !-----------------------------------------------------------------------
  !     Read a formatted dynamics RESTart file
  !     30-SEP-84 Axel Brunger
  !     12-OCT-91 Bernie R. Brooks, modify 7I8 to 7I12
  !     22-OCT-91 Youngdo Won, enable to read C21 restart files
  !     22-JUL-96 James Kong for lambda dynamics restart files
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use number
  ! input/output
  use energym
  use averfluc
  use avfl_ucell
  use fourdm
  use ctitla
  use stream
  use string

  use image
  use prssre, only: QP21XCEN
  use version
  use nose_mod
  use replica_ltm
  use rndnum
  use parallel    ! mh050712
  use repdstr     ! mh050712
#if KEY_GRAPE==1
  use grape, only: lfmm,igrape
  use grapemod, only: save_igrape
#endif
  use pathm
#if KEY_FLUCQ==1
  use flucq     
#endif
  use phmd
  use gbsw
  use gbmv
  use clcg_mod,only:clcginit, setseed
  use memory
#if KEY_MULTICOM==1 /*  VO stringm */
  use multicom_aux   
#endif
#if KEY_BLOCK==1
  use lambdam,only:qmld,msld_read_restart,msld_restart_broadcast  /*ldm*/
#endif

  implicit none

  INTEGER U, NATOM
  real(chm_real) XOLD(*), YOLD(*), ZOLD(*), X(*), Y(*), Z(*)
  real(chm_real) VX(*), VY(*), VZ(*)
#if KEY_CHEQ==1
  real(chm_real) CG(*),CGOLD(*),VCG(*)
  LOGICAL QCG
#endif 
  ! PJ 06/2005
#if KEY_PIPF==1
  real(chm_real) UIND(3,*),UINDO(3,*),VUIND(3,*)
  LOGICAL QPFDYN
  INTEGER NPFBATHS
  real(chm_real) PFNHSBATH(*),PFNHSOBATH(*)
#endif 
#if KEY_DYNVV2==1
  LOGICAL QCONSTRAINTS                        
#endif
#if KEY_DYNVV2==1
  real(chm_real) DXC(*), DYC(*), DZC(*)               
#endif
  INTEGER NPRIV, JHSTRT, NDEGF, NSTEP, NSAVC, NSAVV
  real(chm_real) SEED
  real(chm_real) AVETEM
  INTEGER ISTPSA, LDYNA
  real(chm_real),allocatable,dimension(:,:,:) :: TRANSF

#if KEY_BLOCK==1 /*ldm*/
  LOGICAL QLDM
  LOGICAL QLMC
  INTEGER NBLOCK, NSAVL
  real(chm_real) BXLAMB(*), BVLAMB(*), BLDOLD(*)
#else /*  ldm*/
  INTEGER NSAVL
#endif /* ldm*/

#if KEY_FOURD==1
  real(chm_real) VFD(*), FDOLD(*)
#endif 
#if KEY_SCCDFTB==1
  logical qlamda,qsccres,qpkac
  integer icntdyn, iavti
  real(chm_real) dvdl,dvdlav,dtmp1
#endif 

  ! local
  INTEGER I, J, NATOMQ, ILENEP, ILENET, IVERS, LDYNAR
  real(chm_real) :: xtla
  character(len=1) BIT
  character(len=4) HDR, XTLTPR
  character(len=128) LINE
  character(len=4) NSEO
  LOGICAL QET, MISMAT
  character(len=512*22) rngline
  integer numnodl,nrandl,numnodlr,nrandlr,rngstream,countl
  integer,allocatable,dimension(:) :: allseeds
  character(len=64) :: seed_format
  character(len=10) :: seed_format_cpus
  logical quseseeds
#if KEY_DHDGB==1
!AP/MF
  CHARACTER(len=4) DHDG
  REAL(chm_real),optional :: SDEF(*),SDEFOLD(*)
  REAL(chm_real),optional :: VS_DHDGB(*)
  INTEGER,optional :: TOTALS
  LOGICAL,optional :: QFHDGB
#endif
  ! begin

#if KEY_PARALLEL==1
  numnodl=numnod
#else
  numnodl=1
#endif
  nrandl=nrand
  call chmalloc('dynio.src','WRIDYN','allseeds',nrandl*numnodl,intg=allseeds)

  IF(IOLEV < 0 &
#if KEY_MULTICOM==1 /*  VO stringm */
  &            .and. ( ME_LOCAL.ne.0 ) &             
#endif
  &            ) GOTO 900
  !
  if (reallow) then 
     REWIND(UNIT=U)
  endif             
#if KEY_DHDGB==1
!AP/MF
  READ(U,'(A4,2I6,2X,A4,2X,A4,A4)',END=9) &
       HDR,IVERS,LDYNAR,XTLTPR,NSEO,DHDG
#else /**/
  READ(U,'(A4,2I6,2X,A4,2X,A4,A4)',END=9) &
       HDR,IVERS,LDYNAR,XTLTPR,NSEO
#endif

!  READ(U,'(A4,2I6,2X,A4,2X,A4)',END=9) &
!       HDR,IVERS,LDYNAR,XTLTPR,NSEO
  IF(HDR /= 'REST') THEN
     IF(WRNLEV >= 2) WRITE(OUTU,'(A)') &
          ' READYN> ERROR: wrong file header'
     GOTO 8
  ENDIF
  !
  READ(U,'(/I8)',END=9) NTITLB
  READ(U,'(A)',END=9) (TITLEB(I),I=1,NTITLB)
  IF(PRNLEV >= 2) CALL WRTITL(TITLEB,NTITLB,OUTU,+1)
  !
#if KEY_DHDGB==1
!AP/MF
  IF(DHDG /= '    ' .AND. PRESENT(QFHDGB)) THEN
     QFHDGB=.TRUE.
     READ(U,'(/A)',END=9) LINE
     READ(U,'(3D22.15)',END=9)(SDEFOLD(I),SDEF(I),VS_DHDGB(I), &
                               I=1,TOTALS)
  ENDIF
#endif
  IF(NSEO /= '    ') THEN
     READ(U,'(/A)',END=9) LINE
     READ(U,'(3D22.15)',END=9) (SN11(I),SN12(I),SN13(I),I=1,MAXNOS)
     READ(U,'(3D22.15)',END=9) (SNH(I),SNHV(I),SNHF(I),I=1,MAXNOS)
     ! BEGIN TPCONTROL (G. Lamoureux)
     !     IF (QPCONFULL) THEN
     READ(U,'(3D22.15)',END=9) (ETA_CP(1,I),I=1,3)
     READ(U,'(3D22.15)',END=9) (ETA_CP(2,I),I=1,3)
     READ(U,'(3D22.15)',END=9) (ETA_CP(3,I),I=1,3)
     READ(U,'(3D22.15)',END=9) (ZETA_CP(1,I),I=1,3)
     READ(U,'(3D22.15)',END=9) (ZETA_CP(2,I),I=1,3)
     READ(U,'(3D22.15)',END=9) (ZETA_CP(3,I),I=1,3)
     READ(U,'(3D22.15)',END=9) (G_CP(1,I),I=1,3)
     READ(U,'(3D22.15)',END=9) (G_CP(2,I),I=1,3)
     READ(U,'(3D22.15)',END=9) (G_CP(3,I),I=1,3)
     !     ELSE
     !        READ(U,'(3D22.15)',END=9) ETA_CP(1,1),ZETA_CP(1,1),G_CP(1,1)
     !     ENDIF
     READ(U,'(3D22.15)',END=9) (RX1_CP(1,I),I=1,3)
     READ(U,'(3D22.15)',END=9) (RX1_CP(2,I),I=1,3)
     READ(U,'(3D22.15)',END=9) (RX1_CP(3,I),I=1,3)
     READ(U,'(3D22.15)',END=9) (RX2_CP(1,I),I=1,3)
     READ(U,'(3D22.15)',END=9) (RX2_CP(2,I),I=1,3)
     READ(U,'(3D22.15)',END=9) (RX2_CP(3,I),I=1,3)
     READ(U,'(3D22.15)',END=9) (RV1_CP(1,I),I=1,3)
     READ(U,'(3D22.15)',END=9) (RV1_CP(2,I),I=1,3)
     READ(U,'(3D22.15)',END=9) (RV1_CP(3,I),I=1,3)
     READ(U,'(3D22.15)',END=9) (RV2_CP(1,I),I=1,3)
     READ(U,'(3D22.15)',END=9) (RV2_CP(2,I),I=1,3)
     READ(U,'(3D22.15)',END=9) (RV2_CP(3,I),I=1,3)
     ! END TPCONTROL (G. Lamoureux)
  ENDIF
  !
  IF(XTLTPR /= '    ') THEN
     IF(XTLTYP /= XTLTPR) THEN
        CALL WRNDIE(0,'<READYN>','Crystal types do not match')
     ENDIF
     READ(U,'(/A)',END=9) LINE
     READ(U,'(3D22.15)',END=9) XTLABC
     IF(QP21XCEN) IMXCEN = 0.25*XTLABC(1)
     IF(IVERS > 23) THEN
        READ(U,'(3D22.15)',END=9) HDOT
        READ(U,'(3D22.15)',END=9) PNH, PNHV, PNHF
     ENDIF
     READ(U,'(3D22.15)',END=9) UC1A
     READ(U,'(3D22.15)',END=9) UC2A
     READ(U,'(3D22.15)',END=9) UC1B
     READ(U,'(3D22.15)',END=9) UC2B
     READ(U,'(3D22.15)',END=9) GRAD1A, GRAD1B, GRAD2A, GRAD2B
  ENDIF
  !
  READ(U,'(/A)',END=9) LINE
  IF (IVERS < 22) THEN
     READ(U,'(7I8,D22.15,I8)',END=9) &
          NATOMQ,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED, NSAVL
  ELSE

     !for the new random number generators the info at the end of line
     !needs to be parsed separately. See code below in the broadcast
     !section

     write(seed_format_cpus,'(i10)')nrandl*numnodl
     countl=10
     call trima(seed_format_cpus,countl)
     seed_format='(7I12,D22.15,I12,2I22,'//seed_format_cpus(1:countl)//'I22)'
     READ(U,seed_format,END=977) &
          NATOMQ,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL, &
          numnodlr,nrandlr,allseeds(1:nrandl*numnodl)
977  continue  ! reading the line past end of it happens a lot here: ignoring it!
#if KEY_PARALLEL==0
     !protect if it is the same number
     rngseeds(1:nrandlr) = allseeds(1:nrandlr)
#endif
  ENDIF
  IF (NATOM /= NATOMQ) THEN
     IF(WRNLEV >= 2) WRITE(OUTU,'(A)') &
          ' %READYN-ERR: NATOM - psf mismatch'
     GOTO 8
  ENDIF
  !
  !-----------------------------------------------------------------------
  ! Read current flags, energies and statistics
  !
  ! Just in case we're reading on old restart file, zero the arrays
  ! so what isn't read has zeros.
  EPROP(1:lenenp)=ZERO
  ETERM(1:lenent)=ZERO
  EPRESS(1:lenenv)=ZERO
  call avfl_reset_lt()
  call avfl_reset()
  !
  READ(U,'(/A)',END=9) LINE
  READ(U,'(A)',END=9) LINE  ! read QEPROP
  ILENEP=LEN(LINE)
  CALL TRIMA(LINE,ILENEP)
  ! Check to see if the energy property flags match
  J=MIN(ILENEP,LENENP)
  MISMAT=.FALSE.
  DO I=1,J
     BIT=LINE(I:I)
     READ(BIT,'(L1)') QET
     IF(QET.NEQV.QEPROP(I)) MISMAT=.TRUE.
  ENDDO
  IF(MISMAT) CALL WRNDIE(-1,'<READYN>', &
       'Energy property flags in the restart file do not match')
  !
  READ(U,'(A)',END=9) LINE  ! read QETERM
  ILENET=LEN(LINE)
  CALL TRIMA(LINE,ILENET)
  ! Check to see if the energy term flags match
  J=MIN(ILENET,LENENT)
  MISMAT=.FALSE.
  DO I=1,J
     BIT=LINE(I:I)
     READ(BIT,'(L1)') QET
     IF(QET.NEQV.QETERM(I)) MISMAT=.TRUE.
  ENDDO
  IF(MISMAT) CALL WRNDIE(-1,'<READYN>', &
       'Energy term flags in the restart file do not match')
  !
  IF (ILENEP > LENENP .OR. ILENET.GT.LENENT) THEN
     ! A future restart file??
     CALL WRNDIE(-4,'<READYN>', &
          'Cannot read a future version restart file')
  ELSE IF (ILENEP == LENENP .AND. ILENET.EQ.LENENT) THEN
     ! A current restart file
     READ(U,'(I8,3D22.15)',END=9) ISTPSA,FITA,FITP,AVETEM
     READ(U,'(3D22.15)',END=9) (EPROP(I),EPRPP(I),EPRP2P(I),I=1, &
          LENENP)
     READ(U,'(2D22.15)',END=9) (EPRPA(I),EPRP2A(I),I=1,LENENP)
     READ(U,'(3D22.15)',END=9) (ETERM(I),ETRMP(I),ETRM2P(I),I=1, &
          LENENT)
     READ(U,'(2D22.15)',END=9) (ETRMA(I),ETRM2A(I),I=1,LENENT)
     READ(U,'(3D22.15)',END=9) (EPRESS(I),EPRSP(I),EPRS2P(I), &
          I=1,LENENV)
     READ(U,'(2D22.15)',END=9) (EPRSA(I),EPRS2A(I),I=1,LENENV)
  ELSE IF (ILENEP >= 50 .AND. ILENET.GE.50) THEN
     ! A post version 21 restart file
     READ(U,'(I8,3D22.15)',END=9) ISTPSA,FITA,FITP,AVETEM
     READ(U,'(3D22.15)',END=9) (EPROP(I),EPRPP(I),EPRP2P(I),I=1, &
          ILENEP)
     READ(U,'(2D22.15)',END=9) (EPRPA(I),EPRP2A(I),I=1,ILENEP)
     READ(U,'(3D22.15)',END=9) (ETERM(I),ETRMP(I),ETRM2P(I),I=1, &
          ILENET)
     READ(U,'(2D22.15)',END=9) (ETRMA(I),ETRM2A(I),I=1,ILENET)
     IF(IVERS > 23) THEN
        READ(U,'(3D22.15)',END=9) (EPRESS(I),EPRSP(I),EPRS2P(I), &
             I=1,LENENV)
        READ(U,'(2D22.15)',END=9) (EPRSA(I),EPRS2A(I),I=1,LENENV)
     ENDIF
  ELSE
     ! an old restart file - no go....
     CALL WRNDIE(-4,'<READYN>', &
          'Cannot read a version 21 or earlier restart file')
  ENDIF
  !
  !-----------------------------------------------------------------------
  ! read positions and velocities
#if KEY_FOURD==1 /*4dread*/
  IF(DIM4.AND.(REST4 == 1)) THEN
     READ(U,'(/A)',END=9) LINE
     READ(U,'(3D22.15)',END=9) &
          (XOLD(I),YOLD(I),ZOLD(I),FDOLD(I),I=1,NATOM)
     READ(U,'(/A)',END=9) LINE
     READ(U,'(3D22.15)',END=9) &
          (VX(I),VY(I),VZ(I),VFD(I),I=1,NATOM)
     READ(U,'(/A)',END=9) LINE
     READ(U,'(3D22.15)',END=9) (X(I),Y(I),Z(I),FDIM(I),I=1,NATOM)
  ELSE
#endif /* (4dread)*/
     READ(U,'(/A)',END=9) LINE
     READ(U,'(3D22.15)',END=9) (XOLD(I),YOLD(I),ZOLD(I),I=1,NATOM)
     READ(U,'(/A)',END=9) LINE
     READ(U,'(3D22.15)',END=9) (VX(I),VY(I),VZ(I),I=1,NATOM)
     READ(U,'(/A)',END=9) LINE
     READ(U,'(3D22.15)',END=9) (X(I),Y(I),Z(I),I=1,NATOM)
#if KEY_FOURD==1 /*4dendif*/
  ENDIF
#endif /* (4dendif)*/
#if KEY_DYNVV2==1 /*dynvv2_read*/
  ! BEGIN DYNA VV2 (G. Lamoureux)
  IF (QCONSTRAINTS) THEN
     READ(U,'(/A)',END=9) LINE
     READ(U,'(3D22.15)',END=9) (DXC(I),DYC(I),DZC(I),I=1,NATOM)
  ENDIF
  ! END DYNA VV2 (G. Lamoureux)
#endif /* (dynvv2_read)*/
#if KEY_FLUCQ==1
  IF(QFLUC) CALL FQRREA(U)   
#endif
#if KEY_BLOCK==1 /*ldm*/
  if(qmld) then
     call msld_read_restart(u,nblock)
  elseIF(QLDM .or. QLMC) THEN
     READ(U,'(/A)',END=9) LINE
     READ(U,'(1D22.15)',END=9) (BLDOLD(I),I=1,NBLOCK)
     READ(U,'(/A)',END=9) LINE
     READ(U,'(1D22.15)',END=9) (BVLAMB(I),I=1,NBLOCK)
     READ(U,'(/A)',END=9) LINE
     READ(U,'(1D22.15)',END=9) (BXLAMB(I),I=1,NBLOCK)
  ENDIF
#endif /*  ldm*/
#if KEY_CHEQ==1
  IF (QCG) READ(U,'(/A)',END=9) LINE
  IF (QCG) READ(U,'(3D22.15)',END=9) &
       (CG(I),CGOLD(I),VCG(I),I=1,NATOM)
#endif 
  ! PJ 06/2005
#if KEY_PIPF==1
  IF (QPFDYN) READ(U,'(/A)',END=9) LINE
  IF (QPFDYN) THEN
     DO I = 1, NATOM
        READ(U,'(3D22.15)',END=9) &
             (UIND(J,I),UINDO(J,I),VUIND(J,I),J=1,3)
     ENDDO
  ENDIF
  IF (QPFDYN) READ(U,'(/A)',END=9) LINE
  IF (QPFDYN) THEN
     DO I = 1, NPFBATHS
        READ(U,'(2D22.15)') PFNHSBATH(I),PFNHSOBATH(I)
     ENDDO
  ENDIF
#endif 
#if KEY_SCCDFTB==1
  if((qlamda.or.qpkac).and.qsccres) then
     READ(U,'(/A)',END=9) LINE
     READ(U,'(/A)',END=9) LINE
     READ(U,'(I8,I8)') icntdyn,iavti
     READ(U,'(/A)',END=9) LINE
     READ(U,'(3D22.15)',END=9) dvdl,dvdlav,dtmp1
  ENDIF
#endif 
#if KEY_REPLICA==1
#if KEY_RPATH==1
  IF(QPROPT.AND.(NREPL > 0))THEN
     READ(U,'(/A)',END=9) LINE
     READ(U,'(I12)',END=9)NPCALL
     READ(U,'(3D22.15)',END=9)(PMF(I),I=1,NREPL)
     READ(U,'(3D22.15)',END=9)(FLUC(I),I=1,NREPL)
  ENDIF
#endif 
#endif 

  !
  ! Correct for different restart file formats
  !
#if KEY_NIH==1
  ! If at nih assume older file uses leap-frog
  IF(LDYNAR == 0) LDYNAR=1
#else /**/
  ! If not at nih assume older file uses verlet
  IF(LDYNAR == 0) LDYNAR=-1
#endif 
  IF(LDYNA /= LDYNAR) THEN
     IF (LDYNA > 0) THEN
        ! switch from verlet to leap-frog
        X(1:natom)=X(1:natom)-XOLD(1:natom)
        Y(1:natom)=Y(1:natom)-YOLD(1:natom)
        Z(1:natom)=Z(1:natom)-ZOLD(1:natom)
     ELSE
        ! switch from leap-frog to verlet
        X(1:natom)=X(1:natom)+XOLD(1:natom)
        Y(1:natom)=Y(1:natom)+YOLD(1:natom)
        Z(1:natom)=Z(1:natom)+ZOLD(1:natom)
     ENDIF
  ENDIF
#if KEY_BLOCK==1 /*ldm*/
  IF(QLDM) THEN
     !     correct for different restart file formats
     !     Since bldold(i)=lambda(t+dt), bxlamb(i)=lambda(t)
     !     were stored in all algorithms, we have to switch the lambdas such
     !     that bxlamb(i) = lambda(t+dt).
     !     velocity-verlet: v(i)=v(t+dt)
     !     Verlet: bvlamb(i) = lambda(t)
     !     Leap_frog: bvlamb(i) = v(t+dt/2)
     BXLAMB(1:nblock) = BLDOLD(1:nblock)
  ENDIF
#endif /*   ldm*/
#if KEY_PHMD==1
  IF(QGBSW.or.QGBMV)THEN
    IF (QPHMD)THEN
        CALL PHMDREAD(U) 
    ENDIF
  ENDIF
#endif 
#if KEY_GRAPE==1
  ! store velocities for FMM export
  !write(*,*)'before writing velocities: igrape, save_igrape', igrape, save_igrape
!  if(lfmm.and.(save_igrape==7))write(7,'(3d28.18)')(xold(i),yold(i),zold(i),i=1,natom)
!  if(lfmm.and.(save_igrape==7))write(7,'(3d28.18)')(vx(i),vy(i),vz(i),i=1,natom)
!  if(lfmm.and.(save_igrape==7))write(7,'(3d28.18)')(x(i),y(i),z(i),i=1,natom)
  if(lfmm)write(7,'(3d28.18)')(xold(i),yold(i),zold(i),i=1,natom)
  if(lfmm)write(7,'(3d28.18)')(vx(i),vy(i),vz(i),i=1,natom)
  if(lfmm)write(7,'(f20.10)') (npriv+nstep)/thosnd ! assume 1fs step
  !if(lfmm)write(7,'(3d28.18)')(x(i),y(i),z(i),i=1,natom)
#endif
  !
  ! ready
  IF(PRNLEV >= 2) WRITE(OUTU,'(2A,I8)') &
       ' READYN> dynamics restart file was read. ', &
       'Current step=',NPRIV
  GOTO 900
  !
  !-error-handling
8 CONTINUE
  CALL WRNDIE(-3,'<READYN>','ERROR during read at')
  GOTO 900
  !
9 CONTINUE
  CALL WRNDIE(-3,'<READYN>','EOF during read')
  !
900 CONTINUE
  !
#if KEY_PARALLEL==1
#if KEY_MULTICOM==1 /*  VO stringm */
  if (SIZE_LOCAL.gt.1) then        
#endif
  ! nose
  CALL PSND8(SN11,MAXNOS)
  CALL PSND8(SN12,MAXNOS)
  CALL PSND8(SN13,MAXNOS)
  CALL PSND8(SNH,MAXNOS)
  CALL PSND8(SNHV,MAXNOS)
  CALL PSND8(SNHF,MAXNOS)
  ! BEGIN TPCONTROL (G. Lamoureux)
#if KEY_DYNVV2==1
  CALL PSND8m(ETA_CP,9)
  CALL PSND8m(ZETA_CP,9)
  CALL PSND8m(G_CP,9)
  CALL PSND8m(RX1_CP,9)
  CALL PSND8m(RX2_CP,9)
  CALL PSND8m(RV1_CP,9)
  CALL PSND8m(RV2_CP,9)
  CALL PSND8(DXC,NATOM)
  CALL PSND8(DYC,NATOM)
  CALL PSND8(DZC,NATOM)
#endif 
  ! END TPCONTROL (G. Lamoureux)
  !
  CALL PSND8(XTLABC,6)
  CALL PSND8(HDOT,6)
  CALL PSND8(PNH,1)
  CALL PSND8(PNHV,1)
  CALL PSND8(PNHF,1)
  CALL PSND8(UC1A,6)
  CALL PSND8(UC1B,6)
  CALL PSND8(UC2A,6)
  CALL PSND8(UC2B,6)
  CALL PSND8(GRAD1A,1)
  CALL PSND8(GRAD1B,1)
  CALL PSND8(GRAD2A,1)
  CALL PSND8(GRAD2B,1)
  CALL PSND4(NPRIV,1)
  CALL PSND4(NSTEP,1)
  CALL PSND4(NSAVC,1)
  CALL PSND4(NSAVV,1)
  CALL PSND4(JHSTRT,1)
  CALL PSND4(NDEGF,1)
  CALL PSND8(SEED,1)
!-  call psnd4(rngseeds,nrand) ! Tim Miller 01-03-2013

  ! Parallel random generator restore section
  !
  ! 3 cases:
  ! 1. oldrandom: uses seed number independent of the new RNG
  !    fix it only for the parallel, so not everybody has the same number
  !    testcase ???


  ! 1. nrandlr == 0 or numnodlr == 0 : dont do anything new here
  !   if parallel then it has to ignore the call to clcginit!!!
  ! 2. nrandlr /=0 and numnodlr /= 0: new stuff present
  ! 3. -prevclcg ??? -oldrandom ???

  ! broadcast and process saved seeds:
  CALL PSND4(nrandlr,1)   ! get nrand on every CPU
  CALL PSND4(numnodlr,1)  ! get numnod on every CPU
  ! if nrandl,numnodlr are zero we have an old restart file
  ! quseseeds false means use the seeds that comes from initialization
  ! which by default each cpu has its own based on the system time and adjusted
  ! by cpu number.
  quseseeds=.true.
  if(nrandl /= nrandlr)quseseeds=.false.   ! using different RNG !!!
  if(numnodl /= numnodlr)quseseeds=.false. ! using different processor numbers
  if((nrandlr == 0).or.(numnodlr == 0))quseseeds=.false.
  if(quseseeds)then
     call psnd4(allseeds,nrandl*numnodl)
     rngseeds(1:nrand)=allseeds(mynod*nrandl+1:mynodp*nrandl)
  else
     if(prnlev>=2)then
        write(outu,'(a)')' READYN> Stored seeds for random number generator ignored.'
        write(outu,'(a)')' READYN> Maybe restarting from different count of processors.'
     endif
  endif

  ! X. Qian for CLCG states restore backwards compatibility:
  !         for previous versions' restart files, do nothing.
  !
  !     This code is basically not restartable in parallel
  !     If somebody needs this functionality [s]he can put it in!!!
  !     For now we just take the ISEED variable and do as it is
  !     elsewhere in the code:
  if(qbrokenclcg.or.qoldrandom) then
     seed=rngseeds(1)  ! this works in parallel too!
     if (.not.qoldrng) CALL CLCGINIT(INT(SEED)) !yw 05-Aug-2008
  endif
  ! X. Qian for CLCG
  !
  ! we got some more seeds then with the old random numbers
  ! generator also additionaly first number is numnod and then
  ! nseed numbers.  After this there are nseed*numnod numbers at the
  ! end of the old line (in case of parallel runs)
  !
  ! NOTE2: Compatibility issues:
  !
  !        we can read more info than it was written on the line
  !        but the values would be set to 0.
  !
  ! This is moved from read section
  if(.not.(qbrokenclcg.or.qoldrandom)) then
     ! first check if we got any real data from restart file.
     ! older restart files will have nothing here...
     if ((numnodlr /= 0) .and. (nrandlr /= 0)) then
        ! we have the data
        !
        if(rngchoice==1)then
           rngstream=1 ! check this one!! maybe we are using 2??
           call setseed(rngstream,rngseeds)
!           write(150,*)'setting random seeds after reading from restart file'
        endif
        if(rngchoice==2)then
           call random_seed(put=rngseeds)
        endif
     endif
  endif

!redundant here:  if (.not.qoldrng) CALL CLCGINIT(INT(SEED)) !yw 05-Aug-2008

  call chmdealloc('dynio.src','WRIDYN','allseeds',nrandl*numnodl,intg=allseeds)

  CALL PSND4(ISTPSA,1)
  CALL PSND8(FITA,1)
  CALL PSND8(FITP,1)
  CALL PSND8(AVETEM,1)
  CALL PSND8(EPROP,LENENP)
  CALL PSND8(EPRPP,LENENP)
  CALL PSND8(EPRP2P,LENENP)
  CALL PSND8(EPRPA,LENENP)
  CALL PSND8(EPRP2A,LENENP)
  CALL PSND8(ETERM,LENENT)
  CALL PSND8(ETRMP,LENENT)
  CALL PSND8(ETRM2P,LENENT)
  CALL PSND8(ETRMA,LENENT)
  CALL PSND8(ETRM2A,LENENT)
  CALL PSND8(XOLD,NATOM)
  CALL PSND8(YOLD,NATOM)
  CALL PSND8(ZOLD,NATOM)
  CALL PSND8(VX,NATOM)
  CALL PSND8(VY,NATOM)
  CALL PSND8(VZ,NATOM)
  CALL PSND8(X,NATOM)
  CALL PSND8(Y,NATOM)
  CALL PSND8(Z,NATOM)
#if KEY_CHEQ==1
  IF(QCG) THEN
     CALL PSND8(CG,NATOM)
     CALL PSND8(VCG,NATOM)
     CALL PSND8(CGOLD,NATOM)
  ENDIF
#endif 
#if KEY_BLOCK==1
  if(qmld) then
     call msld_restart_broadcast(nblock)
  endif
#endif 
#if KEY_PHMD==1
  IF(QPHMD)THEN
     CALL PSND8(CG,NATOM)
     CALL PSND8(PH_THETA,NTITR)
  ENDIF
#endif 
  CALL PSNDC(XTLTPR,1)
#if KEY_FLUCQ==1
  CALL FQRUPD   
#endif
#if KEY_FOURD==1 /*4dpar*/
  IF(DIM4.AND.(REST4 == 1)) THEN
     CALL PSND8(FDOLD,NATOM)
     CALL PSND8(VFD,NATOM)
     CALL PSND8(FDIM,NATOM)
  ENDIF
#endif /* (4dpar)*/
#if KEY_MULTICOM==1 /*  VO */
  endif                         
#endif
#endif 
  !
  IF(XTLTPR /= '    ') THEN
     CALL XTLLAT(XUCELL,XTLABC)
     CALL XTLMSR(XUCELL)
     call chmalloc('dynio.src','READYN','TRANSF',3,4,XNSYMM,crl=TRANSF)
     CALL IMFILL(TRANSF,.FALSE.)
     call chmdealloc('dynio.src','READYN','TRANSF',3,4,XNSYMM,crl=TRANSF)
  ENDIF
  !

  RETURN
END SUBROUTINE READYN

SUBROUTINE WRETERM(NPRIV,TIME,QHEADER)
  !
  ! Write some energy terms and properties to file specified by KUNIT
  ! QHEADER should be TRUE on first call if the header containing
  ! a brief file format description is to be written.
  ! Note that QHEADER is set to FALSE once this has been done.
  ! L. Nilsson, Karolinska institutet, March 1998
  !
  !
  use chm_kinds
  !
  use energym
  use reawri
  use stream
  use version
#if KEY_MULTICOM==1 /*  VO string */
  use multicom_aux                         
#endif
  implicit none
  INTEGER NPRIV
  real(chm_real) TIME
  LOGICAL QHEADER
  !
  INTEGER NHEADER,NLINES
  !
  IF(KUNIT < 0 .OR. (IOLEV <= 0 &
#if KEY_MULTICOM==1
     &  .and. (ME_LOCAL.ne.0)   &           
#endif
     &                          )) RETURN
  !
  IF(QHEADER)THEN
     !
     ! Try to keep the NHEADER /number of lines in this header/
     ! and NLINES /number of lines in each record/ up to date!!!!!
     ! The version number is printed as a negative number so that
     ! one can immediately tell the difference from pre v26 files,
     ! where this first number is >0 (=NPRIV)
     NLINES=4
     ! I haven't figured out the names for the pressures, so we leave
     ! those out for the time being
     !
     NHEADER=NLINES+1
     IF(QCNSTP) NLINES=NLINES+3
     IF(QCNSTP .AND. IOLEV > 1) NLINES=NLINES+3
     WRITE(KUNIT, 150)  NHEADER, NLINES, -VERNUM
150  FORMAT(2I3,I10)
     WRITE(KUNIT, 155) 'STEP','TIME',CEPROP(TOTE),CEPROP(TOTKE), &
          CEPROP(EPOT),'EP-K', &
          CEPROP(TEMPS),CETERM(BOND),CETERM(ANGLE), &
          CETERM(DIHE),CETERM(IMDIHE),CETERM(VDW), &
          CETERM(ELEC),CETERM(HBOND),CETERM(CHARM), &
          CETERM(DMC),CETERM(RGY)
155  FORMAT(5A5)
     QHEADER=.FALSE.
  ENDIF
  WRITE(KUNIT,160) NPRIV,TIME,EPROP(TOTE),EPROP(TOTKE), &
       EPROP(EPOT),EPROP(EPOT)-EPROP(TOTKE), &
       EPROP(TEMPS),ETERM(BOND),ETERM(ANGLE), &
#if KEY_CMAP==1
       ETERM(DIHE)+ETERM(CMAP),ETERM(IMDIHE),ETERM(VDW), &   
#endif
#if KEY_CMAP==0
       ETERM(DIHE),ETERM(IMDIHE),ETERM(VDW), &               
#endif
       ETERM(ELEC),ETERM(HBOND),ETERM(CHARM), &
       ETERM(DMC),ETERM(RGY)
  IF (QCNSTP) THEN
     WRITE(KUNIT,170) REFP, &
          EPRESS(PEXX),EPRESS(PEXY),EPRESS(PEXZ), &
          EPRESS(PEYX),EPRESS(PEYY),EPRESS(PEYZ), &
          EPRESS(PEZX),EPRESS(PEZY),EPRESS(PEZZ)
     IF (IOLEV > 1)  &
          WRITE(KUNIT,170) &
          EPRESS(PIXX),EPRESS(PIXY),EPRESS(PIXZ), &
          EPRESS(PIYX),EPRESS(PIYY),EPRESS(PIYZ), &
          EPRESS(PIZX),EPRESS(PIZY),EPRESS(PIZZ)
  ENDIF
#if KEY_LONGLINE==1
160 FORMAT(I16,19F16.4)
170 FORMAT(15F16.4)
#else /**/
160 FORMAT(I16,4F16.4,/,5F16.4,/,5F16.4,:/,5F16.4)
170 FORMAT(5F16.4)
#endif 
  ! machine dependent call to flush output buffers
  CALL GFLUSH(KUNIT)
  RETURN
END SUBROUTINE WRETERM


#if KEY_BLOCK==1 /*ldm_routines*/

SUBROUTINE  WRITLD(NBLOCK,NPRIV,ISTEP,NSTEP,delta)
  !
  !     WRITES A SET OF LAMBDA VARIABLES FOR A SINGLE DYNAMICS STEP.
  !
  use lambdam,only: bixlam, nsavl, iunldm &
       ,titlel, ntitll &
       ,ibvidi, ibvidj, ibclas &
       ,irreup, irrlow, ikbias &
       ,ipbias, nbiasv &
       ,qthetadm,theta

  use chm_kinds
  use dimens_fcm
  use stream
  use image
  use version
#if KEY_MULTICOM==1 /*  VO stringm */
  use multicom_aux        
#endif
  !
  !joined with block   use blockappend     ! yw070702 Theta-dynamics

  implicit none
  integer :: j
  integer, intent(in) :: nblock,npriv,istep, nstep
  real(chm_real), intent(in) :: delta
  real(chm_real4) :: delta4
  !
  integer :: icntrl(20),ifile,nfile,i
  logical :: error
  character(len=4) :: hdr,hdrc
  PARAMETER (HDRC='LAMB')

  !
  ! save all variables between calls to WRITCV
  SAVE
  !
  IF(IUNLDM < 0 .OR. (IOLEV < 0   &
#if KEY_MULTICOM==1 /*  VO stringm */
    &                           .and. (ME_LOCAL.ne.0)     &           
#endif
    &                           )) RETURN
  !
  IFILE=ISTEP/NSAVL
  NFILE=NSTEP/NSAVL
  !
  IF (IFILE <= 0) THEN
     !       ERROR IN INITIATION
     IF(WRNLEV >= 2) WRITE(OUTU,44) NBLOCK,NPRIV,ISTEP, &
          NSAVL,NSTEP,NTITLl,IUNLDM
44   FORMAT(' ** ERROR IN WRITLD INITIATION **',/ &
          '  NBLOCK,NPRIV,ISTEP,', &
          'NSAVL,NSTEP,NTITLE,IUNLDM'/,12I6)
     CALL DIEWRN(-3)
  ELSE IF (IFILE == 1) THEN
     ICNTRL(1:20) = 0
     ICNTRL(1)=NFILE
     ICNTRL(2)=NPRIV
     ICNTRL(3)=NSAVL
     ICNTRL(4)=NSTEP

     !
     HDR=HDRC
     WRITE(IUNLDM) HDR,ICNTRL
     CALL WRTITL(TITLEl,NTITLl,IUNLDM,-1)
     WRITE(IUNLDM) NBIASV
     WRITE(IUNLDM)(iBVIDI(I),iBVIDJ(I),iBCLAS(I), &
          iRREUP(I),iRRLOW(I),iKBIAS(I), &
          iPBIAS(I), I=1,NBIASV)
     WRITE(IUNLDM) NBLOCK
     !       write out lambda**2 instead of lambda itself !!!!
#if KEY_SINGLE==1
     WRITE(IUNLDM) (BiXLAM(I)*BiXLAM(I),I=1,NBLOCK)
  ELSE
     WRITE(IUNLDM) (BiXLAM(I)*BiXLAM(I),I=1,NBLOCK)
#else /**/
     IF(QTHETADM) THEN                      ! wy070702
        WRITE(IUNLDM) THETA,(SIN(THETA))**2   !
     ELSE                                   !
        WRITE(IUNLDM) (SNGL(BiXLAM(I)*BiXLAM(I)),I=1,NBLOCK)
     ENDIF                                  !
  ELSE
     IF(QTHETADM) THEN                      !
        WRITE(IUNLDM) THETA,(SIN(THETA))**2   !
     ELSE                                   !
        WRITE(IUNLDM) (SNGL(BiXLAM(I)*BiXLAM(I)),I=1,NBLOCK)
     ENDIF                                  ! wy070702
#endif 
  ENDIF
  !
  !++  LN MOD /APR 90
  !     Make sure everything is put on disk (which is needed on some
  !     machines in case of a job crash
  CALL SAVEIT(IUNLDM)
  !--
  IF(IFILE < NFILE) RETURN
  IF(PRNLEV >= 2) THEN
     WRITE(OUTU,101) (ICNTRL(I),I=1,3),IUNLDM
101  FORMAT(/2X,I5,'   LAMBDA SETS STARTING FROM',/, &
          5X,'STEP NO ',I8,'   FOR EVERY ',I5,'  STEPS',/, &
          5X,'WRITTEN ON UNIT',I5,/)
  ENDIF
  CALL VCLOSE(IUNLDM,'KEEP',ERROR)
  RETURN
END SUBROUTINE WRITLD
#endif /* (ldm_routines)*/

end module dynio

!============================================================================

!=======================================================================================

   subroutine ldm_gticnt(iunit,hdr,icntrl)
!
! Extract ICNTRL information from lambda-dynamics lambda file
! on unit IUNIT (which is REWOUND).
!  HDR      type of trajectory (CORD or VELD)
!  ICNTRL   the whole ICNTRL array
!
!  Lennart Nilsson, Karolinska Institutet, NOV-87
!  Print and substitution added - BRB, NIH, MAR-98
!
      use chm_kinds
      use consta
      use stream
#if KEY_MULTICOM==1 /*  VO stringm */
      use multicom_aux           
#endif
      use param_store, only: set_param
      use parallel,only:psnd4,psnd8

      implicit none
!
      INTEGER ICNTRL(20),IUNIT
      integer int8traju
      CHARACTER*4 HDR
!
      INTEGER I
      real(chm_real) DELTA
      REAL(chm_real4) DELTA4
!
      IF(IOLEV.GT.0 &
#if KEY_MULTICOM==1
    &               .or. (ME_LOCAL.eq.0) &          
#endif
    &               ) THEN
         int8traju=0     !assume 32 bit ints
         IF (reallow)    REWIND(IUNIT)
      ENDIF              
            READ(IUNIT) HDR,ICNTRL
            ! store bytes of icntrl(10) in delta4
            delta4 = transfer(icntrl(10), delta4)

#if KEY_PARALLEL==1
      CALL PSNDC(HDR,1)
      CALL PSND4(ICNTRL,20)
#endif 
!
      DELTA=DELTA4*TIMFAC ! switch from AKMA time to picoseconds - BRB
#if KEY_PARALLEL==1
      CALL PSND8(delta,1)                               
#endif
!
      IF(PRNLEV.GE.2) THEN
        WRITE(OUTU,26) IUNIT,(ICNTRL(I),I=1,4)
  26    FORMAT(/' READING TRAJECTORY FROM UNIT',I4,/ &
        '   NUMBER OF LAMBDA VALUES IN FILE:    ',I8,/ &
        '   NUMBER OF PREVIOUS DYNAMICS STEPS:  ',I8,/ &
        '   FREQUENCY FOR SAVING LAMBDA VALUES: ',I8,/ &
        '   NUMBER OF STEPS FOR CREATION RUN:   ',I8,/)
        WRITE(OUTU,27) ICNTRL(8),DELTA
  27    FORMAT( &
        '   NUMBER OF DEGREES OF FREEDOM:     ',I8,/ &
        '   THE INTEGRATION TIME STEP (PS):',F11.4)
        WRITE(OUTU,29) (ICNTRL(i), i = 12,20)
  29    FORMAT( &
        '   SUBSTITUENTS ON EACH SITE:  ', 10i5)
  28    FORMAT(A)

      ENDIF

! set some substitution parameters
      CALL set_param('NFILE',ICNTRL(1))
      CALL set_param('START',ICNTRL(2))
      CALL set_param('SKIP', ICNTRL(3))
      CALL set_param('NSTEP',ICNTRL(4))
      CALL set_param('NDEGF',ICNTRL(8))
      call set_param('DELTA',DELTA)

      DELTA=DELTA*TIMFAC

   END subroutine ldm_gticnt

!============================================================================

