subroutine dynamc(vx, vy, vz, vk, xnew, ynew, znew, &
     xold, yold, zold, &
#if KEY_CHEQ==1
     cg,cgnew,cgold,vcg,pmassq, &
#endif
#if KEY_PIPF==1
     uindn,uindo,vuind,pmassu, &
#endif
     amass, npriv, nprivold, &
     ndegf, igvopt, &
     imove, iskp, freeat, nfreat, natomx, bnbnd, &
     bimag, istart, istop, iprfrq, idynpr, jhtemp, &
     gamma, qaver, naver, xave, yave, zave,  &
#if KEY_CHEQ==1
     cgave, &
#endif
     qkuhead, &
     runok &
#if KEY_TSM==1
     ,reacls,prodls,piggls,backls &
#endif
#if KEY_BLOCK==1
     ,qprntvl      &
#endif
#if KEY_TNPACK==1
     ,qeuler,qestrt,qeharm,keharm,rxharm,ryharm,rzharm,lieff &
#endif
#if KEY_TSALLIS==1
     ,qtsall,tsemin,tsq,tsbeta         &
#endif
#if KEY_TPS==1
     ,qtps,inbcut &
#endif
     )
  !
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

#if KEY_LIBRARY == 1
  use api_ktable, only: &
       fill_ktable, &
       ktable_init, ktable_set_names, &
       ktable_add_row
  use api_velos, only: &
       fill_velos, velos_init, velos_set_names, velos_add_rows
#if KEY_BLOCK == 1
  use lambdam, only: nbiasv
  use api_lambdata, only: &
       fill_lambdata, lambdata_init, lambdata_set_names, lambdata_add_rows
  use api_msldata, only: &
       fill_msldata, msldata_init, msldata_set_names, msldata_add_rows
#endif
#endif /* KEY_LIBRARY */
  
#if KEY_RMD==1
  use cross, only: CURRENTGAMMA,SAVEDSX,SAVEDSY,SAVEDSZ, &
       SAVEDSOX,SAVEDSOY,SAVEDSOZ,SAVEDXX, &
       SAVEDXY,SAVEDXZ,SAVEDGAMMA,CTIME,NCRUN,XSTRT
#endif
  use new_timer,only:timer_start,timer_stop,timer_stpstrt,  &
#if KEY_DIMS==1
       T_DIMS,T_DIMNM,T_DIMCOMB,   &
#endif
#if KEY_DOMDEC==1
       T_crdcomm2, &
#endif
       T_dynamc,T_dcntrl,T_crdcomm,T_list,T_constp,T_constt,&
       T_commvar,T_saveshake,T_heur,T_tvir
  !
#if KEY_STRINGM==1 /*  VO stringm */
  use sm_config, only: &
  &   smcv_on, voronoi_hist_on, voronoi_wrong_cell, &
  &   restraint_force_on, compute_whereami, ftsm_on
  use smcv_master, only: smcv_main, smcv_voronoi_whereami,&
  &   smcv_voronoi_compute
  use ftsm, only: ftsm_main
  use cv_common, only: cv
  use sm_var, only: mestring
  use ftsm_var, only : ftsm_mini_on
  use ftsm_voronoi, only : ftsm_voronoi_whereami_compute,&
  &   ftsm_voronoi_whereami, ftsm_voronoi_map,             &
  &   ftsm_voronoi_check, ftsm_voronoi_print_map
  use multicom_aux
  !
#endif /* VO stringm */
#if KEY_CHEQ==1
  use cheq, only: kecgbath,FQNHSBATH,FQNHSOBATH,qcg,massq,     &
       cheqnhs,cheqnhso, fqnhswat, fqnhsowat,nqbaths,noseflag, &
       ndgfbath,ifirstbath,fqnhmbath,qpartbin,cheqtemp, &
       ilastbath,fqtempbath,   &
       DCH,SUMDCH,DDCH                                     &
#if KEY_PARALLEL==1
       ,cgcpy  &
#endif
       ;                    ! end continuation when not PARALLEL
#endif

  use chm_kinds
  use chm_types
  use dimens_fcm
  use vector
  use number
#if KEY_BLOCK==1
  use block_fcm, only : loadl, prhybh
  use block_ltm, only: qhybh, iblckp, nblock
! ldm
  use lambdam,only: &
       qldm, qlmc, ldm_init_dynamc, &
       ldm_reset_dynamc, &
       qmld, msld_firsttheta, ldm_prop1_dynamc, &
       msld_add_kineticenergy, &
       nsavl, msld_writld, &
       rstp, ldm_rstp_dynamc, &
       iresd, lstrt, nrst, ilapot, biptnlam, bixlam, ilaf, bfrst, lmdcoef, &
       msld_updatetheta, ldm_prop2_dynamc, &
       msld_add_kineticenergy, &
       bldold, bivlam, &
       ldm_write_dynamc, &
       ldm_reset3_dynamc, &
       blcoee, &
       ldm_reset2_dynamc
#endif /*   (ldm)*/
#if KEY_SCCDFTB==1
  use blockscc_fcm
#endif
#if KEY_SCCDFTB==1
  use sccdftb
#endif
#if KEY_PIPF==1
  use pipfm
#endif
  use reawri
  use clcg_mod,only:random,clcginit
  use ctitla
  use contrl
  use coord
  use coordc
  use cveloci_mod
  use cvio
  use deriv
  use dynio
  use energym
  use averfluc
  use avfl_ucell
  use icfix
  use icpert
  use image
  use inbnd
  use consta
  use mehmc
  use pull_mod
  use rndnum
  use rxncom
#if KEY_RXNCONS==1
  use rxncons
#endif
#if KEY_SGLD==1
  use sgld,only:sgfavg,sgldw,sgmdw,prntsg,qsgld,qsgmd, sgld_ave_params
#endif
#if KEY_EMAP==1
  use emapmod,only:lemapld
#endif
  use shake
  use stream
#if KEY_TSM==1
  use tsms_mod
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
#if KEY_FLUCQ==1
  use flucq
#endif
  use phmd
#if KEY_QUANTUM==1
  use sizes
  use quantm
#endif
#if KEY_AXD==1
  use axd_module
#endif
  use gamess_fcm
  use grape
#if KEY_SQUANTM==1
  use squantm
#endif
#if KEY_PARALLEL==1
  use parallel
#endif
  use spacdec
  use repdstr    ! mh050712
  use holonom,only:holonoma
  ! FBS MOD ********************************************
  use dmcons
  use rgym
  ! END FBS MOD ****************************************
#if KEY_ENSEMBLE==1
  use ensemble
#endif
#if KEY_DIMS==1
  use dims
#endif
#if KEY_TMD==1
  use tmd
#endif
  use memory
  use dims

  use machutil,only:eclock,die
#if KEY_TPS==1
  use tps
#endif
#if KEY_DOMDEC==1
  use domdec_common,only:atoml,natoml,q_domdec,groupl,&
#if KEY_DOMDEC_GPU==1
       q_gpu, &
#endif
       zonelist, q_split, q_print_output, cpu_vendor, CPU_AMD
  use heurist,only:q_donb_updeci
  use domdec_bonded,only:q_bonded,check_bonded_14_totals,&
       nin14tbl, nex14tbl, nex14tholetbl, nhypertbl, &
#if KEY_CMAP==1
       ncmaptbl, &
#endif
       nbondtbl,nangletbl,ndihetbl,nimdihetbl
#if KEY_LONEPAIR==1
  use domdec_lonepair,only:q_lonepr, check_lonepr_total, nloneprlist
#endif
  use domdec_aniso,only:q_aniso, check_aniso_total, nanisolist
  use domdec_dlb,only:q_load_balance
  use domdec_d2d_comm,only:transfer_coord,copy_to_all
  use domdec_local,only:update_local_coord
  use domdec_d2r_comm,only:send_coord_to_recip
  use psf,only:nbond,ntheta,nphi,nimphi,&
       qhardwall,l_wall,&
#if KEY_CMAP==1
       ncrterm,&
#endif
       igpbs
  use ewald,only:lewald
  use pme_module,only:qpme
  use groupxfast,only:group, group_out
  use enbxfast,only:q_in_dynamc_main_loop
#else
  use psf,only:qhardwall, l_wall
#endif
#if KEY_DOMDEC_GPU==1
  use domdec_util_gpu_mod,only:range_start, range_stop
#endif
#if KEY_ALLMPI==1
  use mpi
#endif
  use heurist,only: updeci, heuristic_check, reuris
  use prssre
#if KEY_BLOCK==1
  use dcntrl_mod, only: sum_potential
#endif
  use consph, only: phwrirsvr
  use pucker_mod,only: pucker_constraint_output,ncspuck
#if KEY_MNDO97==1
  use qm1_info, only : qm_control_r
#endif
#if KEY_DIMS==1
  use param_store, only: set_param
#endif
  use image_routines_module, only: imcent
  
  implicit none

#if KEY_SCCDFTB==1
  !     Local variable
  real(chm_real) dvdltmp
#endif

  real(chm_real) VX(*),VY(*),VZ(*),VK(*),XNEW(*),YNEW(*),ZNEW(*)
  real(chm_real) XOLD(*),YOLD(*),ZOLD(*)
  integer K ! needed everywhere
#if KEY_CHEQ==1
  real(chm_real) CG(*)
  real(chm_real) CGNEW(*),CGOLD(*),VCG(*),CGAVE(*)
  real(chm_real) KECG
  real(chm_real) FQNHA
  real(chm_real) NHSDIF,MV2TMP,CMASS,CHEQNHTL,NHPOT,NHKIN
  real(chm_real) FQNHM2,FQNHS2,FQNHS2O,NHS2DIF,FQNHS2A,FQNHS2N,ECONS
  real(chm_real) ETA(10),ETAD(10),ETADD(10),ETANEW(10), &
       ETAOLD(10),META(10),ETA1DIF,KECGSOL
  real(chm_real) FQNHABATH(10), FQNHSNBATH(10), NHSDIFBATH(10), &
       MAXERROR,MAXEOLD, &
       MV2TMPBATH(10),ERROR
  INTEGER J1, I1
  INTEGER NHITR,CHEQNHMX,IBS,IMTS,LLL,LLLL,inner,LL,CHEQCDGF
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
  real(chm_real) AMASS(*)
  INTEGER NPRIV,NPRIVOLD,NDEGF,IGVOPT
  INTEGER IMOVE(*),ISKP(*)
  INTEGER FREEAT(*),NFREAT,NATOMX
  !!  INTEGER BNBND(*),BIMAG(*)
  type(nonbondDataStructure) BNBND
  type(imageDataStructure) BIMAG
  INTEGER ISTART,ISTOP,IPRFRQ,IDYNPR
  real(chm_real) GAMMA(*)
  LOGICAL QAVER
  INTEGER NAVER
  real(chm_real) XAVE(*),YAVE(*),ZAVE(*)
  real(chm_real) JHTEMP

  LOGICAL QKUHEAD,RUNOK,OK
#if KEY_TSM==1
  INTEGER REACLS(*),PRODLS(*),PIGGLS(*),BACKLS(*)
#endif
#if KEY_BLOCK==1
  LOGICAL QPRNTVL
  real(chm_real) V_TSM(8)
#endif /*  BLOCK*/
#if KEY_TNPACK==1
  LOGICAL QEULER,QESTRT,QEHARM
  real(chm_real) KEHARM(*),RXHARM(*),RYHARM(*),RZHARM(*),LIEFF
#endif

  !
  integer ia,is,iq
#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
  integer ig
  real(chm_real) gcarr(9+6+18)
  integer ipt
  integer nbond_sum, ntheta_sum, nphi_sum, nimphi_sum, ncrterm_sum, nin14_sum, nex14_sum
  integer nex14thole_sum, nhyper_sum
#if KEY_LONEPAIR==1
  integer nlonepr_sum
#endif
  integer naniso_sum
#else /**/
  real(chm_real) GCARR(10)
#endif
  real(chm_real) GCARR2(2)
  real(chm_real) TIMMER
#if KEY_ALLMPI==1
  integer status
  real(chm_real),allocatable,dimension(:) :: crdbuf
#endif
#endif

#if KEY_CHEQ==1
  real(chm_real) FACTCG,FACTCG1,FACTCG2,ALPHACG2,PMASSQ(NATOMX)
  LOGICAL QSECD
#endif
#if KEY_DIMS==1
  LOGICAL DIMSCARTON,QPRINT
  real(chm_real) OMSC
#endif
  real(chm_real) DELTA2,FACT,DNUM,DNM1,DNP1,DNPM1
  real(chm_real) DELTAS,TIME,RVAL,AKMATI,ALPHA,TEMPI,EWRK
  real(chm_real) VXI,VYI,VZI

  INTEGER NATOM,IST1,ISTEP,ISTPSA,NUMSTP,I,J,NATOM2,NATOM3
  INTEGER ATFRST,ATLAST
  logical :: ATMASK(NATOMX)

  real(chm_real) FLUCTD,TOTKEO,TOTKEN,GNORM
  real(chm_real) EPPRTMP(LENENP), ETPRTMP(LENENT), EVPRTMP(LENENV)
  CHARACTER(len=40) STR
  INTEGER STRLN
  LOGICAL QICP,QOK
#if KEY_TMD==1
  real(chm_real) tol_corr,lagsum
  !%% AvdV addition
  real(chm_real) teterm(lenent),teprop(lenenp),tepress(lenenv),tmde
  logical tmdsh1
  real(chm_real),allocatable,dimension(:) :: itmptx,itmpty,itmptz
  !%% AvdV end addition
#endif

#if KEY_TSALLIS==1
  !     Passed
  real(chm_real)  TSEMIN, TSQ, TSBETA
  LOGICAL QTSALL
  !     Local
  real(chm_real)  TSFACT
#endif
#if KEY_TPS==1
  !     Passed
  INTEGER REJCTP
  INTEGER INBCUT
  LOGICAL QTPS
  !     Local
  INTEGER STEMP
#endif

#if KEY_TSM==1
  real(chm_real) PUMEP,RLAMBDA,PLAMBDA,ORLAMBDA,OPLAMBDA
#endif
  integer a0,a1
#if KEY_DOMDEC==1
  real(chm_real), allocatable, dimension(:) :: XTMP,YTMP,ZTMP
  real(chm_real) XDM1(1),YDM1(1),ZDM1(1)
  real(chm_real) XDM2(1),YDM2(1),ZDM2(1)

#endif

  SAVE

  ! TIMFAC is the conversion factor from AKMA time to picoseconds.
  !
  !=======================================================================
  ! Startup code

  !%% AvdV addition
#if KEY_TMD==1
  if(qtmd)then
     if((itmd > 0).and.(ftmd <= 0))ftmd=nprint
  endif
#endif
  !%% AvdV end addition
#if KEY_DOMDEC==1
   IF(q_domdec .and. LIMCEN .AND. NTRANS > 0)THEN
      call chmalloc('dynamc.src','DYNAMC','xtmp',natomx,crl=xtmp)
      call chmalloc('dynamc.src','DYNAMC','ytmp',natomx,crl=ytmp)
      call chmalloc('dynamc.src','DYNAMC','ztmp',natomx,crl=ztmp)
    ENDIF
#endif
  call timer_start(T_dynamc)

#if KEY_MNDO97==1
  ! namkh 09/12/03 for MNDO97 printing setup
  ISTEPQM = ISTEP-1
#endif

#if KEY_TSM==1
  IF (QTSM) THEN
     RLAMBDA = (ONE-LAMBDA)**LPOWER
     PLAMBDA = LAMBDA**LPOWER
  ENDIF
  QICP=(IICP > 0)
#endif
#if KEY_BLOCK==1 /*ldm1*/ /*ldm*/
  if(qldm .or. qlmc) call ldm_init_dynamc(nblock, finalt, jhstrt, istep, igvopt)
#endif /* (ldm1)*/

  NATOM=NATOMX
  NATOM2=NATOM+NATOM
  NATOM3=NATOM2+NATOM

#if KEY_DOMDEC==1
  if (q_domdec) then
     atfrst = 1
     atlast = natoml
  else
#endif
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

  do I = 1, NATOMX
#if KEY_PARASCAL==1
     ATMASK(I) = (JPBLOCK(I) == MYNOD)
#elif KEY_SPACDEC==1
     ATMASK(I) = (ICPUMAP(I) == MYNOD)
#else /**/
     ATMASK(I) = .true.
#endif
  enddo

  DELTAS=HALF*DELTA*DELTA
  DELTA2=DELTA*DELTA
#if KEY_CHEQ==1
  IF (QCG) THEN
     FACTCG=DELTA2/MASSQ
     FACTCG1=DELTAS/MASSQ
  ENDIF
#endif
  IST1=ISTART-1

  do ia=atfrst,atlast
#if KEY_DOMDEC==1
     if (q_domdec) then
        i = atoml(ia)
     else
#endif
        i = ia
#if KEY_DOMDEC==1
     endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
     IF (ATMASK(I)) THEN
#endif
        XCOMP(I)=X(I)
        YCOMP(I)=Y(I)
        ZCOMP(I)=Z(I)
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
     ENDIF
#endif
  ENDDO
  !
#if KEY_STRINGM==1 /*  VO stringm */
  if (voronoi_hist_on) call vdgbr(xcomp, ycomp, zcomp,1)
#endif
  !
  IF(QAVER) THEN
     IF(NAVER == 0) THEN
#if KEY_DOMDEC==1
        if (q_domdec) then
           xave(1:natom) = zero
           yave(1:natom) = zero
           zave(1:natom) = zero
        else
#endif
           do i=atfrst,atlast
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              IF (ATMASK(I)) THEN
#endif
                 XAVE(I)=ZERO
                 YAVE(I)=ZERO
                 ZAVE(I)=ZERO
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              ENDIF
#endif
           ENDDO
#if KEY_DOMDEC==1
        endif
#endif
#if KEY_CHEQ==1
        IF (QCG) THEN
#if KEY_DOMDEC==1
           if (q_domdec) then
              cgave(1:natom) = zero
           else
#endif
              do i=atfrst,atlast
#if KEY_PARASCAL==1
                 IF(JPBLOCK(I) == MYNOD) THEN
#endif
                    CGAVE(I)=ZERO
#if KEY_PARASCAL==1
                 ENDIF
#endif
              ENDDO
#if KEY_DOMDEC==1
           endif
#endif
        ENDIF
#endif
     ENDIF
  ENDIF
#if KEY_TSM==1 /*tsmtest*/
  IF(QTSM.AND.PIGSET) THEN
     !     x,y,z and dx,dy,dz will have been taken care of in energy.
     !     xold,yold,zold and vx,vy,vz are velocities so use back0.
     CALL BACK0(XOLD,YOLD,ZOLD)
     CALL BACK0(VX,VY,VZ)
  ENDIF
#if KEY_TNPACK==1 /*eulertsm*/
  IF(QEULER.AND.QTSM) CALL WRNDIE(-3,'<DYNAMC>', &
       'EULER and TSM are not compatible')
#endif /* (eulertsm)*/
#endif /* (tsmtest)*/
#if KEY_TNPACK==1 /*eulersetup*/
  IF(QEULER.AND.QCNSTT) CALL WRNDIE(-3,'<DYNAMC>', &
       'EULER and constant temperature not compatible')
  IF(QEULER.AND.QHOLO) CALL WRNDIE(0,'<DYNAMC>', &
       'EULER and SHAKE are untested for compatibility')

  ! Setup the restraint force constants for Euler integration.
  IF(QEULER) THEN
     QEHARM = .FALSE.

     do ia=atfrst,atlast
#if KEY_DOMDEC==1
        if (q_domdec) then
           i = atoml(ia)
        else
#endif
           i = ia
#if KEY_DOMDEC==1
        endif
#endif
#if KEY_PARASCAL==1
        IF(JPBLOCK(I) == MYNOD) THEN
#endif
           IF(GAMMA(I+NATOM) > ZERO) THEN
              KEHARM(I) = HALF/GAMMA(I+NATOM)
           ELSE
              KEHARM(I) = ZERO
           ENDIF
#if KEY_PARASCAL==1
        ENDIF
#endif
     ENDDO
  ENDIF
#endif /* (eulersetup)*/

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


#if KEY_CHEQ==1 /*cheq*/
#if KEY_PARALLEL==1
  IF (MYNOD == 0) THEN
#endif
     IF (QCG .and. prnlev > 5 ) THEN
        write(outu,*) " Charge Bath Temperatures"
        write(OUTU,189)((KECGBATH(L)/(KBOLTZ*NDGFBATH(L))) &
             ,L=1,NQBATHS)
        write(OUTU,*) "Charge KE = ", KECG
     ENDIF
#if KEY_PARALLEL==1
  ENDIF
#endif

189 format(2(f10.5,1x))

#endif /* (cheq)*/

  !=======================================================================
  !
  ! This is the 3-step VERLET
  IF(IGVOPT >= 3) THEN
     !
     !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
#if KEY_BLOCK==1 /*LDM*/
     ! Banba 01/11/99
     ! When reading restart file was not read, lambda (t+dt) should be used
     ! This correction is not effecient
     if (qldm) call ldm_reset_dynamc(nblock)
#endif /*  LDM*/
     !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
     ! Restore old step (just as a total energy check).
     do ia=atfrst,atlast
#if KEY_DOMDEC==1
        if (q_domdec) then
           i = atoml(ia)
        else
#endif
           i = ia
#if KEY_DOMDEC==1
        endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
        IF (ATMASK(I)) THEN
#endif

           FACT=ONE/GAMMA(I+NATOM3)

              XNEW(I)=VX(I)*FACT-XOLD(I)
              YNEW(I)=VY(I)*FACT-YOLD(I)
              ZNEW(I)=VZ(I)*FACT-ZOLD(I)

#if KEY_SPACDEC==1 || KEY_PARASCAL==1
        ENDIF
#endif
     ENDDO
#if KEY_TSM==1
     ! xnew,ynew,znew are velocities so use back0
     IF(QTSM.AND.PIGSET) CALL BACK0(XNEW,YNEW,ZNEW)
#endif

     IF(IDYNPR == 0.OR.JHSTRT.EQ.0) THEN
        ! Get previous energy for printing (just as a check)

#if KEY_DOMDEC==1
        if (q_domdec) then
           call transfer_coord(x, y, z, .false.)
           call update_local_coord(x, y, z)
        endif
#endif
        call timer_stop(T_dynamc)
        call timer_stop(T_dcntrl)
#if KEY_DOMDEC==1
        ! Communicate coordinates to recip CPUs
        if (q_domdec .and. q_split) then
           call timer_start(T_crdcomm2)
           call send_coord_to_recip(x, y, z, .true.)
           call timer_stop(T_crdcomm2)
        endif
#endif
        CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
     !
#if KEY_STRINGM==1 /*  VO stringm */
!================ call string method routines ========================
      if (smcv_on) call smcv_main(x,y,z,xcomp,ycomp,zcomp,&
                                  amass(1:natom),dx,dy,dz,istart-1)
      if (ftsm_on) call ftsm_main(x(1:natom),y(1:natom),z(1:natom),             &
     &                            xcomp(1:natom),ycomp(1:natom),zcomp(1:natom), &
     &                            dx(1:natom),dy(1:natom),dz(1:natom),          &
     &                            amass(1:natom),                               &
     &                            istart-1,wmain(1:natom),bnbnd,bimag)
!=====================================================================
#endif
        call timer_start(T_dcntrl)
        call timer_start(T_dynamc)
#if KEY_SGLD==1
        !  WXW    perform force average
        IF(QSGLD.OR.QSGMD)THEN
           CALL SGFAVG(ATFRST,ATLAST,EPROP(EPOT),IMOVE, &
                DX,DY,DZ)
        ENDIF
#endif

        IF(QHOLO) THEN
           ! Add shake "force" to the internal virial.
           do ia=atfrst,atlast
#if KEY_DOMDEC==1
              if (q_domdec) then
                 i = atoml(ia)
              else
#endif
                 i = ia
#if KEY_DOMDEC==1
              endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              IF (ATMASK(I)) THEN
#endif
                 FACT=GAMMA(I+NATOM)
                 ALPHA=GAMMA(I+NATOM2)
                 IF (GAMMA(I+NATOM) /= ZERO) THEN
                    RVAL=ONE/GAMMA(I+NATOM)
                 ELSE
                    RVAL=ZERO
                 ENDIF
                 X(I)=(ALPHA*XNEW(I)-DX(I)*FACT-XOLD(I))*RVAL
                 Y(I)=(ALPHA*YNEW(I)-DY(I)*FACT-YOLD(I))*RVAL
                 Z(I)=(ALPHA*ZNEW(I)-DZ(I)*FACT-ZOLD(I))*RVAL

#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              ENDIF
#endif
           ENDDO
#if KEY_TSM==1
           ! We don't want the "back atom" forces (in vx) contributing to the
           ! virial and we need to zero the shake corrected "back atom" half-step
           ! velocities (in xnew).
           IF(QTSM.AND.PIGSET) THEN
              CALL BACK0(X,Y,Z)
           ENDIF
#endif
           CALL VIRSHK(EPRESS(VIXX:VIZZ),NATOM,XCOMP,YCOMP,ZCOMP,X,Y,Z)
        ENDIF
     ENDIF
  ENDIF

  !=======================================================================

  igvopt2: IF(IGVOPT == 2) THEN

     ! This is the 2-step VERLET
     !    X,Y,Z          - current coordinates
     !    XOLD,YOLD,ZOLD - ignored
     !    VX,VY,VZ       - velocity to use

     IF(QHOLO) THEN
        ! Do shake just in case coordinates don't fit constraints
        call timer_stop(T_dynamc)
        call timer_stop(T_dcntrl)
        CALL HOLONOMA(X,Y,Z,XCOMP,YCOMP,ZCOMP,.TRUE.,.TRUE.,QOK)
        call timer_start(T_dcntrl)
        call timer_start(T_dynamc)
#if KEY_PARALLEL==1
        !        CALL PSYNC()    !APH: PSYNC commented out for performance
#if KEY_DOMDEC==1
        if (.not.q_domdec) then
#endif
           TIMMER=ECLOCK()
#if KEY_DOMDEC==1
        endif
#endif
#if KEY_SPACDEC==1
        CALL SPACSR(X,Y,Z)
#else /**/
#if KEY_DOMDEC==1
        if (q_domdec) then
           call transfer_coord(x, y, z, .false.)
           call update_local_coord(x, y, z)
        else
#endif
           CALL VDGBR(X,Y,Z,0)
#if KEY_DOMDEC==1
        endif
#endif
#endif
#if KEY_DOMDEC==1
        if (.not.q_domdec) then
#endif
           TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
           !        CALL PSYNC()  !APH: PSYNC commented out for performance
           TIMMER=ECLOCK()
#if KEY_DOMDEC==1
        endif
#endif
#endif
        do ia=atfrst,atlast
#if KEY_DOMDEC==1
           if (q_domdec) then
              i = atoml(ia)
           else
#endif
              i = ia
#if KEY_DOMDEC==1
           endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           IF (ATMASK(I)) THEN
#endif
              XCOMP(I)=X(I)
              YCOMP(I)=Y(I)
              ZCOMP(I)=Z(I)
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           ENDIF
#endif
        ENDDO
     !
#if KEY_STRINGM==1 /*  VO */
        if (voronoi_hist_on) call vdgbr(xcomp, ycomp, zcomp,1)
#endif
     !
#if KEY_DOMDEC==1
     else
        ! We have to update DOMDEC local coordinates here
        if (q_domdec) then
           call update_local_coord(x, y, z)
        endif
#endif /* domdec */
     ENDIF

     call timer_stop(T_dynamc)
     call timer_stop(T_dcntrl)
#if KEY_DOMDEC==1
     ! Communicate coordinates to recip CPUs
     if (q_domdec .and. q_split) then
        call timer_start(T_crdcomm2)
        call send_coord_to_recip(x, y, z, .true.)
        call timer_stop(T_crdcomm2)
     endif
#endif
     CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
#if KEY_STRINGM==1 /*  VO stringm */
!================ call string method routines ========================
      if (smcv_on) call smcv_main(x,y,z,xcomp,ycomp,zcomp,&
                                  amass(1:natom),dx,dy,dz,istart-1)
      if (ftsm_on) call ftsm_main(x(1:natom),y(1:natom),z(1:natom),             &
     &                            xcomp(1:natom),ycomp(1:natom),zcomp(1:natom), &
     &                            dx(1:natom),dy(1:natom),dz(1:natom),          &
     &                            amass(1:natom),                               &
     &                            istart-1,wmain(1:natom),bnbnd,bimag)
!=====================================================================
#endif
     call timer_start(T_dcntrl)
     call timer_start(T_dynamc)
#if KEY_SGLD==1
     !  WXW    perform force average
     IF(QSGLD.OR.QSGMD)THEN
        CALL SGFAVG(ATFRST,ATLAST,EPROP(EPOT),IMOVE, &
             DX,DY,DZ)
     ENDIF
#endif

     IF (ILANG  ==  1) THEN
#if KEY_TPS==1 /*tps_seed_swap_1*/
        !          M. Hagan and A. R. Dinner, May 2003
        !          For TPS, use saved seed to get Langevin forces.  It is necessary
        !          to use the same seed that was used in the original 3-step dynamics
        !          or the same displacement vectors will not be the same, and TPS
        !          moves will fail.  The sign of the saved seed denotes the integration
        !          direction.
        ! /MH09/   this is not implemented with the new random generators
        !          the organization of the code around seeds should be modified
        !          for new stuff :-(
        !          For now we dont modify anything here
        IF (QTPS) THEN
           STEMP = ISEED
           ISEED = ABS(ITPSSD)
           ! This probably will not work with the newrng code??
           ! but works with the old random anyway.
           if (.not.qoldrng) then  !yw 05-Aug-2008
              CALL CLCGINIT(ISEED)
              ISEED=1
           endif
        ENDIF
#endif /* (tps_seed_swap_1)*/

           CALL DLNGV(GAMMA,ISEED)

#if KEY_TPS==1 /*tps_seed_swap_2*/
        IF (QTPS) THEN
           ISEED = STEMP
           if (.not.qoldrng) then  !yw 05-Aug-2008
              CALL CLCGINIT(ISEED)
              ISEED=1
           endif
        ENDIF
#endif /* (tps_seed_swap_2)*/
     ENDIF
     IGVOPT=3
     !lb...B980830
     !  Fix to correct uninitialized variables in parallel runs
     !  clbiii/mfc - 9/28/98
     DO I=1, NATOM
        XOLD(I) = ZERO
        YOLD(I) = ZERO
        ZOLD(I) = ZERO
        XNEW(I) = ZERO
        YNEW(I) = ZERO
        ZNEW(I) = ZERO
     ENDDO
     !lb...
     do ia=atfrst,atlast
#if KEY_DOMDEC==1
        if (q_domdec) then
           i = atoml(ia)
        else
#endif
           i = ia
#if KEY_DOMDEC==1
        endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
        IF (ATMASK(I)) THEN
#endif
           IF (IMOVE(I) == 0) THEN
              FACT=DELTAS/AMASS(I) ! corrected MFH and ARD May 2003
              ALPHA=TWO*GAMMA(I+NATOM2)*GAMMA(I+NATOM3)*DELTA2
#if KEY_TPS==1
              !              If we are shooting in the opposite direction than the
              !              original integration from which these velocities are taken,
              !              we have to reverse the two ALPHAs due to the assymmetry in
              !              the forward and backwards displacement formulas for Langevin
              !              dynamics.
              IF (QTPS .AND. (PDIR*ITPSSD  <  0)) THEN
                 ALPHA=TWO*GAMMA(I+NATOM3)*DELTA2
              ENDIF
#endif
              XOLD(I)=VX(I)*ALPHA-DX(I)*FACT
              YOLD(I)=VY(I)*ALPHA-DY(I)*FACT
              ZOLD(I)=VZ(I)*ALPHA-DZ(I)*FACT

              ALPHA=ALPHA/GAMMA(I+NATOM2) ! better factor
#if KEY_TPS==1
              IF (QTPS .AND. (PDIR*ITPSSD  <  0)) THEN
                 ALPHA=TWO*GAMMA(I+NATOM2)*GAMMA(I+NATOM3)*DELTA2
              ENDIF
#endif
              XNEW(I)=VX(I)*ALPHA+DX(I)*FACT
              YNEW(I)=VY(I)*ALPHA+DY(I)*FACT
              ZNEW(I)=VZ(I)*ALPHA+DZ(I)*FACT

           ENDIF
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
        ENDIF
#endif
     ENDDO
     !lb...B980830
#if KEY_CVELOCI==1 /*cvel_init*/
     !.DNC.check for cveloci; ZERO XOLD AND XNEW
     !                      SO VELOCITIES (KE) = 0 FOR CVELOCI
     IF(LCVEL) THEN
        DO J=1,NCVEL
           I=FCVEL(J)
           IF (PrnLev >= 5) WRITE(OUTU,1120) I
           XOLD(I)=ZERO
           YOLD(I)=ZERO
           ZOLD(I)=ZERO
           XNEW(I)=ZERO
           YNEW(I)=ZERO
           ZNEW(I)=ZERO
        ENDDO
1120    FORMAT(/' DYNAMC>  Constant velocity applied to atom number',I8)
        IF(NCVEL > 5 .AND. PrnLev >= 5) WRITE(OUTU,1121)
1121    FORMAT(/' DYNAMC>  Constant velocity for additional atoms...',/)
     ENDIF
#endif /* (cvel_init)*/

#if KEY_BLOCK==1 /*ldm*/
     if (qmld) call msld_firsttheta(nblock,delta)
     if(qldm .or. qlmc) call ldm_prop1_dynamc(nblock,istep,delta)
#endif /* (ldm)*/

#if KEY_TSM==1
     ! Now xold and xnew are velocities.
     IF(QTSM.AND.PIGSET) THEN
        CALL BACK0(XOLD,YOLD,ZOLD)
        CALL BACK0(XNEW,YNEW,ZNEW)
     ENDIF
#endif

     !SB should probably call hardwall here too; however, the user should ensure that
     !   the starting coors are meaningful

     ! Process SHAKE if requested (not the most efficient way)
     IF(QHOLO) THEN
        do ia=atfrst,atlast
#if KEY_DOMDEC==1
           if (q_domdec) then
              i = atoml(ia)
           else
#endif
              i = ia
#if KEY_DOMDEC==1
           endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           IF (ATMASK(I)) THEN
#endif
              X(I)=XCOMP(I)-XNEW(I)
              Y(I)=YCOMP(I)-YNEW(I)
              Z(I)=ZCOMP(I)-ZNEW(I)
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           ENDIF
#endif
        ENDDO
        call timer_stop(T_dynamc)
        call timer_stop(T_dcntrl)
        CALL HOLONOMA(X,Y,Z,XCOMP,YCOMP,ZCOMP,.TRUE.,.TRUE.,QOK)
        call timer_start(T_dcntrl)
        call timer_start(T_dynamc)
        do ia=atfrst,atlast
#if KEY_DOMDEC==1
           if (q_domdec) then
              i = atoml(ia)
           else
#endif
              i = ia
#if KEY_DOMDEC==1
           endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           IF (ATMASK(I)) THEN
#endif
              XNEW(I)=XCOMP(I)-X(I)
              YNEW(I)=YCOMP(I)-Y(I)
              ZNEW(I)=ZCOMP(I)-Z(I)
              X(I)=XCOMP(I)+XOLD(I)
              Y(I)=YCOMP(I)+YOLD(I)
              Z(I)=ZCOMP(I)+ZOLD(I)
              FACT=GAMMA(I+NATOM)
              ALPHA=GAMMA(I+NATOM2)
              VX(I)=ALPHA*XNEW(I)-DX(I)*FACT
              VY(I)=ALPHA*YNEW(I)-DY(I)*FACT
              VZ(I)=ALPHA*ZNEW(I)-DZ(I)*FACT

#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           ENDIF
#endif
        ENDDO
        call timer_stop(T_dynamc)
        call timer_stop(T_dcntrl)
        CALL HOLONOMA(X,Y,Z,XCOMP,YCOMP,ZCOMP,.TRUE.,.TRUE.,QOK)
        call timer_start(T_dcntrl)
        call timer_start(T_dynamc)
#if KEY_DOMDEC==1
        if (q_domdec) then
           call transfer_coord(x, y, z, .false.)
           call update_local_coord(x, y, z)
        endif
#endif
        do ia=atfrst,atlast
#if KEY_DOMDEC==1
           if (q_domdec) then
              i = atoml(ia)
           else
#endif
              i = ia
#if KEY_DOMDEC==1
           endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           IF (ATMASK(I)) THEN
#endif

                 XOLD(I)=X(I)-XCOMP(I)
                 YOLD(I)=Y(I)-YCOMP(I)
                 ZOLD(I)=Z(I)-ZCOMP(I)
                 IF (GAMMA(I+NATOM) /= ZERO) THEN
                    FACT=MINONE/GAMMA(I+NATOM)
                 ELSE
                    FACT=ZERO
                 ENDIF
                 VX(I)=(XOLD(I)-VX(I))*FACT
                 VY(I)=(YOLD(I)-VY(I))*FACT
                 VZ(I)=(ZOLD(I)-VZ(I))*FACT

#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           ENDIF
#endif
        ENDDO
        ! Add shake "force" to the internal virial.
#if KEY_TSM==1
        ! We don't want the "back atom" forces (in vx) contributing to the
        ! virial and we need to zero the shake corrected "back atom" half-step
        ! velocities (in xnew).
        IF(QTSM.AND.PIGSET) THEN
           CALL BACK0(VX,VY,VZ)
        ENDIF
#endif
        CALL VIRSHK(EPRESS(VIXX:VIZZ),NATOM,XCOMP,YCOMP,ZCOMP,VX,VY,VZ)

        do ia=atfrst,atlast
#if KEY_DOMDEC==1
           if (q_domdec) then
              i = atoml(ia)
           else
#endif
              i = ia
#if KEY_DOMDEC==1
           endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           IF (ATMASK(I)) THEN
#endif
              FACT=GAMMA(I+NATOM3)
              VX(I)=(XNEW(I)+XOLD(I))*FACT
              VY(I)=(YNEW(I)+YOLD(I))*FACT
              VZ(I)=(ZNEW(I)+ZOLD(I))*FACT

#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           ENDIF
#endif
        ENDDO
#if KEY_TSM==1
        ! Xold, xnew and vx are velocities.
        IF(QTSM.AND.PIGSET) THEN
           CALL BACK0(XOLD,YOLD,ZOLD)
           CALL BACK0(XNEW,YNEW,ZNEW)
           CALL BACK0(VX,VY,VZ)
        ENDIF
#endif
     ENDIF

     EPROP(PEPR) = FMARK
     EPROP(KEPR2) = FMARK
     IF(QCNSTP) THEN
        EPROP(XTLKP2) = FMARK
        EPROP(XTLKEP) = FMARK
        EPROP(XTLPEP) = FMARK
     ENDIF

  ENDIF igvopt2

  !=======================================================================
  ! Generic code for both 3-step and 2-step continuation
  !
  ! compute initial portion of the kinetic energy
  ! compute temperature

  IF (IDYNPR == 0 .OR. JHSTRT.EQ.0) THEN
     TOTKEO=ZERO
     TOTKEN=ZERO
     TEMPI=ZERO

     do ia=atfrst,atlast
#if KEY_DOMDEC==1
        if (q_domdec) then
           i = atoml(ia)
        else
#endif
           i = ia
#if KEY_DOMDEC==1
        endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
        IF (ATMASK(I)) THEN
#endif

           TOTKEO=TOTKEO+AMASS(I)*(XOLD(I)**2+YOLD(I)**2+ZOLD(I)**2)
           TOTKEN=TOTKEN+AMASS(I)*(XNEW(I)**2+YNEW(I)**2+ZNEW(I)**2)
           TEMPI=TEMPI+AMASS(I)*(VX(I)**2+VY(I)**2+VZ(I)**2)

#if KEY_SPACDEC==1 || KEY_PARASCAL==1
        ENDIF
#endif
     ENDDO

#if KEY_BLOCK==1
     if (qmld) call msld_add_kineticenergy(nblock, tempi)  /*ldm*/
#endif
     ! . Calculate the RMS gradient and work.
     EWRK = ZERO
     RVAL = ZERO
     do ia=atfrst,atlast
#if KEY_DOMDEC==1
        if (q_domdec) then
           i = atoml(ia)
        else
#endif
           i = ia
#if KEY_DOMDEC==1
        endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
        IF (ATMASK(I)) THEN
#endif
           EWRK = EWRK + DX(I)*VX(I) + DY(I)*VY(I) + DZ(I)*VZ(I)
           RVAL = RVAL + DX(I)*DX(I) + DY(I)*DY(I) + DZ(I)*DZ(I)
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
        ENDIF
#endif
     ENDDO
     EWRK=EWRK*DELTA

#if KEY_PARALLEL==1
     ! double counting      TMERI(TIMDCNTRL)=TMERI(TIMDCNTRL)+ECLOCK()-TIMMER
     !     CALL PSYNC()  !APH: PSYNC commented out for performance
#if KEY_DOMDEC==1
     if (.not.q_domdec) then
#endif
        TIMMER=ECLOCK()
#if KEY_DOMDEC==1
     endif
#endif
     GCARR(1)=TOTKEO
     GCARR(2)=TOTKEN
     GCARR(3)=TEMPI
     GCARR(4)=EWRK
     GCARR(5)=RVAL
     CALL GCOMB(GCARR,5)
     TOTKEO=GCARR(1)
     TOTKEN=GCARR(2)
     TEMPI=GCARR(3)
     EWRK=GCARR(4)
     RVAL=GCARR(5)

#if KEY_DOMDEC==1
     if (.not.q_domdec) then
#endif
        TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
        !     CALL PSYNC()    !APH: PSYNC commented out for performance
        TIMMER=ECLOCK()
#if KEY_DOMDEC==1
     endif
#endif
#endif
     EPROP(GRMS)=ZERO
     IF(RVAL > ZERO) EPROP(GRMS) = SQRT(RVAL/NDEGF)

     ! Calculate the pressures.
     CALL PRSEXT
     CALL PRSINT(NATOM,AMASS,XOLD,YOLD,ZOLD,DELTA, &
          XNEW,YNEW,ZNEW,DELTA)

     ! WRITE OUT INITIAL CONDITIONS

     ISTEP=ISTART-1
     AKMATI=DELTA*NPRIV
     TIME=TIMFAC*AKMATI
     ! Possibly time-dependent PULLing forces need to know time
     PTIME=TIME
#if KEY_RMD==1
     CTIME=TIME
#endif

     TOTKEN = HALF*TOTKEN/DELTA2
     TOTKEO = HALF*TOTKEO/DELTA2
     EPROP(HFCKE) = HALF * (TOTKEO+TOTKEN)
     EPROP(TOTKE) = TEMPI*HALF
     IF(EPROP(KEPR2) == FMARK) EPROP(KEPR2)=TOTKEN + EWRK
     IF(EPROP(PEPR) == FMARK) EPROP(PEPR)=EPROP(EPOT) - EWRK
     EPROP(EHFC) = (TOTKEN - EPROP(TOTKE) + EPROP(PEPR) - &
          EPROP(EPOT))/THREE + (EPROP(KEPR2)- TOTKEO)/TWELVE

     EPROP(HFCTE) = EPROP(EPOT) + EPROP(TOTKE) + EPROP(EHFC)
     EPROP(TOTE)  = EPROP(EPOT) + EPROP(TOTKE)
     EPROP(TEMPS) = TEMPI / (KBOLTZ*NDEGF)

#if KEY_FLUCQ==1
     ! Calculate charge velocities, and the FlucQ contribution to energy
     IF (QFLUC) CALL FQCNEW(0,NPRINT,ATFRST,ATLAST)
#endif
#if KEY_PHMD==1
     IF ((QPHMD).AND.(NPRIV  ==  0)) THEN
        CALL UpdatePHMD(0, 1, 0, 0)
     ENDIF
#endif

     IF(QCNSTP) THEN
        IF(EPROP(XTLPEP) == FMARK) EPROP(XTLPEP) = EPROP(PEPR)
        IF(EPROP(XTLKP2) == FMARK) EPROP(XTLKP2) = EPROP(KEPR2)
     ENDIF

     EPROP(TEPR)  = EPROP(HFCTE)
     EPROP(PEPR)  = EPROP(EPOT)
     EPROP(KEPR2) = TOTKEN
     EPROP(KEPR)  = TOTKEO
     
     IF(PRNLEV > 2) THEN
        CALL PRINTE(OUTU, EPROP, ETERM,'DYNA','DYN', &
             .TRUE.,ISTEP, TIME, ZERO, .TRUE.)
#if KEY_SGLD==1 /*sgldprint*/
        IF(QSGLD.OR.QSGMD)CALL PRNTSG(OUTU,.TRUE.,'DYNA',.TRUE.)
#endif /* (sgldprint)*/
        !adm jr.
        IF (QCNSTP) THEN
           RVAL=DOTVEC(DXTL,DXTL,XDIM) / XDIM
           GNORM = ZERO
           IF(RVAL > ZERO) GNORM = SQRT(RVAL)
           CALL PRNXTLD(OUTU,'DYNA',XTLTYP,XUCELL,.TRUE.,GNORM, &
                .TRUE.,EPRESS)
        ENDIF
      ENDIF

     !        ARD 00-08-15
     !        Save the initial kinetic energy for the acceptance criterion
     !        in Hybrid Monte Carlo (see mc.src).
#if KEY_MC==1
     HMCKEI = EPROP(TOTKE)
#endif

  ENDIF

  ! Initialize accum variables
  IF(ISTOP < ISTART) CALL DIE
  IF(MOD(IST1,IPRFRQ) == 0) THEN
     call avfl_reset()
     FITA = ZERO
     ! Unit cell and gradient terms
     if (QCNSTP) call avfl_ucell_reset()
#if KEY_QUANTUM==1
     IF(QMPERT.OR.QDECOM) THEN
        DO I = 1,LENQEQ
           EQPRA(I) = ZERO
           EQPR2A(I) = ZERO
        ENDDO
     ENDIF
#endif
#if KEY_SQUANTM==1
     IF(QMFEP .or. QMSOLV) THEN
        DO I = 1, LENQEQ
           EQPRA(I)  = ZERO
           EQPR2A(I) = ZERO
        END DO
     END IF
#endif
  ENDIF
  ! Initialize accum variables
  IF (JHSTRT == 0) THEN
     do ia=atfrst,atlast
#if KEY_DOMDEC==1
        if (q_domdec) then
           i = atoml(ia)
        else
#endif
           i = ia
#if KEY_DOMDEC==1
        endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
        IF (ATMASK(I)) THEN
#endif
           VK(I)=ZERO
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
        ENDIF
#endif
     ENDDO
     JHTEMP=ZERO
     FITP=ZERO
     call avfl_reset_lt()
     ! Unit cell and gradient terms for long term averages
     if (QCNSTP) call avfl_ucell_reset_lt()
#if KEY_QUANTUM==1
     IF(QMPERT.OR.QDECOM) THEN
        DO I = 1,LENQEQ
           EQPRP(I) = ZERO
           EQPR2P(I) = ZERO
        ENDDO
     ENDIF
     ! JG 5/2002
     IF (CHDYN) THEN
        IF (ISTART == 1) THEN
           CALL DYNDEN('INIT',1)
        ENDIF
     ENDIF
#endif
#if KEY_SQUANTM==1
     IF(QMFEP .or. QMSOLV) THEN
        DO I = 1,LENQEQ
           EQPRP(I)  = ZERO
           EQPR2P(I) = ZERO
        END DO
     END IF
#endif
  ENDIF
#if KEY_DIMS==1
  OMSC=0.0
#endif

#if KEY_SQUANTM==1
!  if(LTRBOMD) q_apply_tr_bomd=.true.   ! for TR-BOMD, start applying TR-BOMD
#endif
#if KEY_MNDO97==1
  qm_control_r%md_run =.true.        ! this is MD run.
#endif
  !=======================================================================
  ! THIS IS THE MAIN LOOP FOR DYNAMICS.
  !
  ! within 3 step Verlet at this point;
  !    X,Y,Z          - scratch array for coordinates
  !    XCOMP,YCOMP,ZCOMP - current coordinates  (Xn-1)
  !    XOLD,YOLD,ZOLD - next step vector        (Un-half)
  !    XNEW,YNEW,ZNEW - previous step vector    (Un-3halves)
  !    VX,VY,VZ       - Filled with velocities  (Vn-1)
  !    GAMMA(NATOM,1) - RDF (std.dev. of random force)
  !                   SQRT(2.0*AMASS(I)*GAM*KBT)/DELTA
  !    GAMMA(NATOM,2) - BETA  ( dx scale factor)
  !                   DELTA*DELTA/((ONE+GAM*HALF)*AMASS(I))
  !    GAMMA(NATOM,3) - ALPHA ( x-xold scale factor)
  !                   (ONE-GAM*HALF)/(ONE+GAM*HALF)
  !    GAMMA(NATOM,4) - Velocity compute scale factor
  !                   HALF*SQRT(ONE+GAM*HALF)/DELTA
  !

  call heuristic_check(istart-1, xcomp, ycomp, zcomp, xold, yold, zold, .true.)

#if KEY_PARALLEL==1
  CALL GCOMB(reuris,1)
#endif

  mainloop: DO ISTEP=ISTART,ISTOP

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('Before energy call (dynamc)')
#endif
#if KEY_SCCDFTB==1
     igeometry = ISTEP
#endif
#if KEY_SCCDFTB==1
     IMD = 1
#endif

#if KEY_DOMDEC==1
     if (.not.q_domdec) then
#endif
#if KEY_PARALLEL==1
        TIMMER=ECLOCK()
#endif
#if KEY_DOMDEC==1
     endif
#endif

#if KEY_RMD==1 /*rmd*/

     IF(NCRUN > 0) THEN
        ! Backup current GAMMA()
        DO I=1,NATOMX
           CURRENTGAMMA(I)=GAMMA(I)
           CURRENTGAMMA(I+NATOM)=GAMMA(I+NATOM)
           CURRENTGAMMA(I+NATOM2)=GAMMA(I+NATOM2)
           CURRENTGAMMA(I+NATOM3)=GAMMA(I+NATOM3)
        ENDDO
     ENDIF

     IF((NCRUN > 0) .AND. XSTRT) THEN
        DO I=1,NATOMX
           XOLD(I)=SAVEDSX(I)
           YOLD(I)=SAVEDSY(I)
           ZOLD(I)=SAVEDSZ(I)
           XNEW(I)=SAVEDSOX(I)
           YNEW(I)=SAVEDSOY(I)
           ZNEW(I)=SAVEDSOZ(I)
           XCOMP(I)=SAVEDXX(I)
           YCOMP(I)=SAVEDXY(I)
           ZCOMP(I)=SAVEDXZ(I)
           GAMMA(I)=SAVEDGAMMA(I)
           GAMMA(I+NATOM)=SAVEDGAMMA(I+NATOM)
           GAMMA(I+NATOM2)=SAVEDGAMMA(I+NATOM2)
           GAMMA(I+NATOM3)=SAVEDGAMMA(I+NATOM3)
        enddo
        XSTRT=.FALSE.
     ENDIF

#endif /* (rmd)*/

#if KEY_AXD==1
     IF (LAXD) THEN
        CALL LEAPAXD(XCOMP,YCOMP,ZCOMP,XOLD,YOLD,ZOLD,XNEW,YNEW,ZNEW,AMASS,NATOMX,ISTEP,NSTEP)
     ENDIF
#endif

     !%% AvdV addition
#if KEY_TMD==1
#if KEY_DIMS==1
     IF(QDIMS) THEN
        IF(QTMD.AND..NOT.DTMD) THEN
           QTMD=.FALSE.
           DTMD=.TRUE.
        ENDIF
     ENDIF
#endif
     if(qtmd)then
        tmdsh1=.true.
        tmdwr=.false.
        if(itmd > 0)then
           IF(MOD(ISTEP,ftmd) == 0)tmdwr=.true.
        endif
     endif
#endif
     !%% AvdV end addition

#if KEY_SCCDFTB==1
#if KEY_PARALLEL==1
     IF (MYNOD == 0) THEN
#endif

        if(qsccb.and.(qlamda.or.qpkac)) then
           if((istep >= sccpass).and.(mod((istep-sccpass),sccstep) == 0)) &
                then
              icntdyn=icntdyn+1
              dvdltmp=dvdlb+dvdla+dvdlp+dvdle+dvdlv+dvdlscc+dvdlub &
                   +dvdlip
              dvdl=dvdl+dvdltmp
              write(outpka,'(1x,"DVDL= ",F14.8," at step ",I6)') &
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

#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
     if (.not.q_domdec) then
#endif
        TMERI(TIMDCNTRL)=TMERI(TIMDCNTRL)+ECLOCK()-TIMMER
        !     CALL PSYNC()     !APH: PSYNC commented out for performance
        TIMMER=ECLOCK()
#if KEY_DOMDEC==1
     endif
#endif
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
     ! Compute new atom positions
#if KEY_SPACDEC==1 /*spacdec*/
     a0=1
     a1=nmyatm
#else /* (spacdec)*/
     a0=atfrst
     a1=atlast
#endif /* (spacdec)*/

#if KEY_DOMDEC==1
     if (q_domdec) then
!$omp parallel do schedule(static) private(ia, i)
           do ia=a0,a1
              i = atoml(ia)
              X(I)=XCOMP(I)+XOLD(I)
              Y(I)=YCOMP(I)+YOLD(I)
              Z(I)=ZCOMP(I)+ZOLD(I)
           enddo
!$omp end parallel do
     else
#endif
        do ia=a0,a1
#if KEY_SPACDEC==1 /*spacdec*/
           I=MYSATARR(ia)
#else /* (spacdec)*/
           i = ia
#endif /* (spacdec)*/
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif
              X(I)=XCOMP(I)+XOLD(I)
              Y(I)=YCOMP(I)+YOLD(I)
              Z(I)=ZCOMP(I)+ZOLD(I)
#if KEY_PARASCAL==1
           ENDIF
#endif
        enddo
#if KEY_DOMDEC==1
     endif
#endif
!!$     do ia=a0,a1
!!$##IF DOMDEC
!!$        if (q_domdec) then
!!$           i = atoml(ia)
!!$        else
!!$##ENDIF
!!$##IF SPACDEC (spacdec)
!!$           I=MYSATARR(ia)
!!$##ELSE (spacdec)
!!$           i = ia
!!$##ENDIF (spacdec)
#if KEY_DOMDEC==1
!!$        endif
#endif
#if KEY_PARASCAL==1
!!$        IF(JPBLOCK(I) == MYNOD) THEN
#endif
!!$           X(I)=XCOMP(I)+XOLD(I)
!!$           Y(I)=YCOMP(I)+YOLD(I)
!!$           Z(I)=ZCOMP(I)+ZOLD(I)
#if KEY_PARASCAL==1
!!$        ENDIF
#endif
!!$     enddo
!
#if KEY_STRINGM==1 /* (string_method_voronoi)  VO stringm */
     if (smcv_on) then
       if (voronoi_hist_on.and..not.restraint_force_on) then ! will be off for exploration dynamics
     !    compute whereami: this code will not work if a replica was allowed to cross just before
     !    calculation ended, so that its 'cv%voronoi_whereami' is the new cell, but the actual coords
     !    (xcomp...) correspond to the old cell;
        if (compute_whereami) then
         call vdgbr(xcomp, ycomp, zcomp, 1) ! need to know all coordinates
         call vdgbr(xnew, ynew, znew, 1)
         call smcv_voronoi_whereami( &
          xcomp(1:natom)-xnew(1:natom),ycomp(1:natom)-ynew(1:natom),&
          zcomp(1:natom)-znew(1:natom),amass(1:natom)) ! use coordinates from previous step
     !
         if (any(cv%voronoi_map.ne.-1)) then
          if (cv%voronoi_map(mestring+1).ne.cv%voronoi_whereami) then
     !    old coordinates inconsistent: check new coordinates: perhaps replica has been reflected
           call smcv_voronoi_whereami(xcomp(1:natom),ycomp(1:natom),&
                                      zcomp(1:natom),amass(1:natom)) ! current coordinates
     !
           if (cv%voronoi_map(mestring+1).ne.cv%voronoi_whereami) then
!
            if (ME_LOCAL.eq.0) then
             i=prnlev
             prnlev=5
             call wrndie(-3,'<DYNAMC>', &
         'SMCV VORONOI MAP INCONSISTENT WITH CURRENT COORDINATES.')
             prnlev=i
             return
            endif  ! ME_LOCAL
           endif ! current coordinates
          endif ! old coordinates
         endif ! voronoi_map.ne.-1
         compute_whereami=.false.
        endif ! compute_whereami
     !
     !  check replica cell
        if (.not.smcv_voronoi_compute(xcomp,ycomp,zcomp,&
            amass(1:natom),istep)) then ! check V. cell
         if (voronoi_wrong_cell) then
           if (ME_LOCAL.eq.0) &
           write(outu,770) ME_STRNG, istep
 770  FORMAT('DYNAMC> WARNING: SMCV REPLICA ',I5, &
       ' INSIDE WRONG VORONOI CELL AT STEP',I9)
         endif
         voronoi_wrong_cell=.true.
     !========================================
     !REFLECT :
#if KEY_SPACDEC==1
         a0=1 ;     a1=nmyatm
#endif
         a0=atfrst; a1=atlast  !.not.SPACDEC
         do ia=a0,a1
#if KEY_SPACDEC==1
          I=MYSATARR(ia)
#elif KEY_DOMDEC==1 /*  VO NB: currently no support for DOMDEC in stringm */
          if (q_domdec) i=atoml(i)
#elif KEY_PARASCAL==1
          IF(JPBLOCK(I) == MYNOD) THEN
#else
            i = ia
#endif
            FACT=-XOLD(I)   ! checking xcomp
            XOLD(I)=-XNEW(I)
            XNEW(I)=FACT
            FACT=-YOLD(I)
            YOLD(I)=-YNEW(I)
            YNEW(I)=FACT
            FACT=-ZOLD(I)
            ZOLD(I)=-ZNEW(I)
            ZNEW(I)=FACT
     !       compute new positions (note: if the voronoi test is on xcomp [vs x] then for one iteration, we actually are in another cell)
     !       currently, checking xcomp is preferable because it is more accurate (should retain accuracy of the verlet)
     !       otherwise (with x) we are not generating correct momenta ( xold should be computed afresh, but it is not )
            X(I)=XCOMP(I)  + XOLD(I) ! checking xcomp
            Y(I)=YCOMP(I)  + YOLD(I)
            Z(I)=ZCOMP(I)  + ZOLD(I)

#if KEY_PARASCAL==1
          ENDIF
#endif
         ENDDO
     !
        else ! smcv_voronoi_compute
          voronoi_wrong_cell=.false.
        endif ! not voronoi_compute
       endif ! voronoi_hist_on
     endif ! smcv_on
     !================ end SMCV ==========================
     !================ begin FTSM / Voronoi. There is currently code duplication
     if (ftsm_on) then
       if (voronoi_hist_on) then
        if (compute_whereami) then
         call vdgbr(xcomp, ycomp, zcomp, 1) ! need to know all coordinates
         call vdgbr(xnew, ynew, znew, 1)
         call ftsm_voronoi_whereami_compute( &
&          xcomp(1:natom)-xnew(1:natom),ycomp(1:natom)-ynew(1:natom),&
&          zcomp(1:natom)-znew(1:natom)) ! use coordinates from previous step
     !
         if (all(ftsm_voronoi_map.ne.-1)) then
          if (ftsm_voronoi_map(mestring+1).ne.ftsm_voronoi_whereami) &
     !    old coordinates inconsistent: check current coordinates: perhaps replica has been reflected
&           call ftsm_voronoi_whereami_compute(xcomp(1:natom),ycomp(1:natom),&
&                                              zcomp(1:natom)) ! current coordinates
          if (ftsm_voronoi_map(mestring+1).ne.ftsm_voronoi_whereami) then
     !    current coordinates are also inconsistent: this should never happen, unless
     !    the restart procedure is broken (somewhere above this line, xold, xcomp, and xnew are modified)
     !    as a last (ad hoc) resort before bailing out, try the new coords (maybe the replica has just crossed back)
           call vdgbr(x, y, z, 1)
           call ftsm_voronoi_whereami_compute(x(1:natom),y(1:natom),&
&                                               z(1:natom)) ! current coordinates
          endif
     !
     ! all processors decide whether coordinates are consistent
          if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) &
&          call MPI_ALLREDUCE(ftsm_voronoi_map(mestring+1).eq.ftsm_voronoi_whereami, ok, 1, MPI_LOGICAL, MPI_LAND, MPI_COMM_STRNG, i)
     ! broadcast to slaves
          if (SIZE_LOCAL.gt.1) call MPI_BCAST(ok, 1, MPI_LOGICAL, 0, MPI_COMM_LOCAL, i)
          if (.not.ok) then
     ! output map
           if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) &
&            call MPI_ALLGATHER(ftsm_voronoi_whereami, 1, MPI_INTEGER, ftsm_voronoi_map, 1, MPI_INTEGER, MPI_COMM_STRNG, i)
           if (ME_STRNG.eq.0) then
            call open_file(i, 'VMAP.DBG', 'FORMATTED', 'WRITE')
            call ftsm_voronoi_print_map(i)
            call VCLOSE(i, 'KEEP', j)
           endif
     !
           if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) call MPI_BARRIER(MPI_COMM_STRNG, i)
           if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL) call MPI_BARRIER(MPI_COMM_LOCAL, i)
     !
           call wrndie(-3,'<DYNAMC>', 'FTSM VORONOI MAP INCONSISTENT WITH CURRENT COORDINATES. (WROTE VMAP.DBG)')
           return
          endif ! not ok
          compute_whereami=.false.
         endif ! map =/= -1
        endif  ! compute_whereami
     !
        if (.not.ftsm_voronoi_check(xcomp,ycomp,zcomp,istep)) then ! still in the current cell?
         if (voronoi_wrong_cell) then
           if (ME_LOCAL.eq.0) &
           write(outu,771) ME_STRNG, istep
 771  FORMAT('DYNAMC> WARNING: FTSM REPLICA ',I5, &
       ' INSIDE WRONG VORONOI CELL AT STEP',I9)
         endif
         voronoi_wrong_cell=.true.
     !========================================
     !REFLECT :
#if KEY_SPACDEC==1
         a0=1 ;     a1=nmyatm
#endif
         a0=atfrst; a1=atlast  !.not.SPACDEC
         do ia=a0,a1
#if KEY_SPACDEC==1
          I=MYSATARR(ia)
#elif KEY_DOMDEC==1 /*  VO NB: currently no support for DOMDEC in stringm */
          if (q_domdec) i=atoml(i)
#elif KEY_PARASCAL==1
          IF(JPBLOCK(I) == MYNOD) THEN
#else
            i = ia
#endif
            FACT=-XOLD(I)
            XOLD(I)=-XNEW(I)
            XNEW(I)=FACT
            FACT=-YOLD(I)
            YOLD(I)=-YNEW(I)
            YNEW(I)=FACT
            FACT=-ZOLD(I)
            ZOLD(I)=-ZNEW(I)
            ZNEW(I)=FACT
            X(I)=XCOMP(I)  + XOLD(I)
            Y(I)=YCOMP(I)  + YOLD(I)
            Z(I)=ZCOMP(I)  + ZOLD(I)
#if KEY_PARASCAL==1
          ENDIF
#endif
         ENDDO
        else ! not ftsm_voronoi_check
          voronoi_wrong_cell=.false.
        endif ! not ftsm_voronoi_check
       endif ! voronoi_hist_on
     endif ! ftsm_on
     !======================== end FTSM ===========================
#endif /*(string_method_voronoi) */

#if KEY_CHEQ==1 /*cheq*/

     IF (QCG) THEN

        IF (NOSEFLAG == 1) THEN

           CHEQNHMX  = 1000
           CHEQNHTL = 0.000001
           DO K = 1, NQBATHS
              KECGBATH(K) = 0.0
              DO L = IFIRSTBATH(K), ILASTBATH(K)
                 KECGBATH(K)=KECGBATH(K)+PMASSQ(L)*VCG(L)*VCG(L)
              ENDDO
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
           ELSE

           ENDIF

           !  UPDATE CHARGES

1039       CONTINUE



           DO I = 1,NATOM
              IF (QPARTBIN(I) /= 0) THEN
                 CGOLD(I)=CG(I)
                 CG(I) = CGNEW(I)
              ENDIF
           ENDDO




        ELSEIF (NOSEFLAG == 2) THEN

           IF(QCG) THEN
              KECG=0.0
              DO I=1,NATOM

                 CGNEW(I) = 2.0*CG(I) - CGOLD(I) -  &
                      (DELTA**2)*DCH(I)/PMASSQ(I)

                 VCG(I)=(CGNEW(I)-CGOLD(I))/(2.0*DELTA)

                 KECG=KECG+PMASSQ(I)*VCG(I)*VCG(I)

                 IF (QPARTBIN(I) /= 0) THEN
                    CGOLD(I)= CG(I)
                    CG(I)= CGNEW(I)
                 ENDIF

              ENDDO
           ENDIF


        ELSEIF (NOSEFLAG == 0) THEN
           !     do nothing
        ELSE
        ENDIF   !  CONDITIONAL ON NOSEFLAG

     ELSE   ! CONDITIONAL ON QCG
     ENDIF    ! CONDITIONAL ON QCG

#if KEY_PARALLEL==1
     IF (QCG) THEN
        CALL VDGBRE(CG,IPARPT)
        CALL VDGBRE(VCG,IPARPT)
        CALL CGCPY(CG)
     ENDIF
#endif
     ! end conditional on CHEQ
#endif /*  (cheq)*/
     ! PJ 06/2005
#if KEY_PIPF==1
     IF (QPIPF .AND. QPFDYN) THEN
        CALL PFDYN(UINDO,UINDN,VUIND,PMASSU,DELTA)
     ENDIF
#endif
#if KEY_CVELOCI==1 /*cvel_main*/
     !-DNC Do constant velocity
     IF(LCVEL) THEN
        DO J=1,NCVEL
           I=FCVEL(J)
           X(I)=XCOMP(I)+DELTA*TIMFAC*VCVEL(1,J)
           Y(I)=YCOMP(I)+DELTA*TIMFAC*VCVEL(2,J)
           Z(I)=ZCOMP(I)+DELTA*TIMFAC*VCVEL(3,J)
        ENDDO
     ENDIF
#endif /* (cvel_main)*/
     ! If constant pressure is enabled
     IF (QCNSTP) THEN
        call timer_start(T_constp)
        !RCZ 92/06/01 - modify REFP if pressure was requested to be linearly
        !               modified i.e. if REFPI /= REFPF
        ! (in progress
        ! - I am not quite sure how to calculate FACT in the case of restart)
        IF(QCNSTPL) THEN
           FACT=NPRIV-NPRIVOLD
           IF(NSTEP /= 0) FACT=FACT/NSTEP
           DO I=1,6
              REFP(I)=(ONE-FACT)*REFPI(I)+FACT*REFPF(I)
           ENDDO
           IF(PRNLEV > 5) THEN
              WRITE(OUTU,'(A,3I5,F10.5)') &
                   ' ISTART,ISTOP,ISTEP,FACT=',ISTART,ISTOP,ISTEP,FACT
              WRITE(OUTU,'(A,6F8.1)') ' REFP=',REFP
           ENDIF
        ENDIF
#if KEY_TSM==1
        IF(QTSM.AND.PIGSET) CALL PIGCVSET(X,Y,Z)
#endif /*  TSM*/
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_start('pscale (dynamc)')
#endif
        CALL PSCALE(X,Y,Z,XOLD,YOLD,ZOLD,VX,VY,VZ, &
             XNEW,YNEW,ZNEW,XCOMP,YCOMP,ZCOMP,GAMMA,ISKP,NDEGF,NPRIV)
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_stop()
#endif
        call timer_stop(T_constp)
     ENDIF
#if KEY_TSM==1
     !     Just in case.
     IF(QTSM.AND.PIGSET) CALL PIGCVSET(X,Y,Z)
#endif /*  TSM*/

#if KEY_PARALLEL==1 /*parallel*/
#if KEY_DOMDEC==1
     if (.not.q_domdec) then
#endif
        TMERI(TIMDCNTRL)=TMERI(TIMDCNTRL)+ECLOCK()-TIMMER
        !     CALL PSYNC()     !APH: PSYNC commented out for performance
        TIMMER=ECLOCK()
        call timer_stpstrt(T_dynamc,T_crdcomm)
#if KEY_ALLMPI==1 /*allmpi*/
        !      print *,"Coords MPI_ALLGATHERV 8"
        CALL MPI8ALLGATHERV(X(IPARPT(MYNOD)+1),NPARPT(MYNOD), &
             MPI_DOUBLE_PRECISION, &
             X,NPARPT,IPARPT, &
             MPI_DOUBLE_PRECISION,COMM_CHARMM,STATUS)
        CALL MPI8ALLGATHERV(Y(IPARPT(MYNOD)+1),NPARPT(MYNOD), &
             MPI_DOUBLE_PRECISION, &
             Y,NPARPT,IPARPT, &
             MPI_DOUBLE_PRECISION,COMM_CHARMM,STATUS)
        CALL MPI8ALLGATHERV(Z(IPARPT(MYNOD)+1),NPARPT(MYNOD), &
             MPI_DOUBLE_PRECISION, &
             Z,NPARPT,IPARPT, &
             MPI_DOUBLE_PRECISION,COMM_CHARMM,STATUS)
#else /*   (allmpi)*/
#if KEY_SPACDEC==1 /*spacedec*/
        CALL SPACSR(X,Y,Z)
#else /*         (spacedec)*/
        CALL VDGBR(X,Y,Z,0)
#endif /*        (spacedec)*/
#endif /* (allmpi)*/
        call timer_stpstrt(T_crdcomm,T_dynamc)
#if KEY_DOMDEC==1
        if (.not.q_domdec) then
#endif
           TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
           TIMMER=ECLOCK()
#if KEY_DOMDEC==1
        endif
#endif
#if KEY_DOMDEC==1
     endif
#endif
#endif /* (parallel)*/

     IF(QAVER) THEN
        NAVER=NAVER+1
        do ia=atfrst,atlast
#if KEY_DOMDEC==1
           if (q_domdec) then
              i = atoml(ia)
           else
#endif
              i = ia
#if KEY_DOMDEC==1
           endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           IF (ATMASK(I)) THEN
#endif
              XAVE(I)=XAVE(I)+X(I)
              YAVE(I)=YAVE(I)+Y(I)
              ZAVE(I)=ZAVE(I)+Z(I)
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           ENDIF
#endif
        ENDDO
     ENDIF

     IF(NSAVC > 0) THEN
        IF (MOD(ISTEP,NSAVC) == 0) THEN
           IF(QAVER) THEN
              DNUM=ONE / NAVER
              do ia=atfrst,atlast
#if KEY_DOMDEC==1
                 if (q_domdec) then
                    i = atoml(ia)
                 else
#endif
                    i = ia
#if KEY_DOMDEC==1
                 endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
                 IF (ATMASK(I)) THEN
#endif
                    XAVE(I)=XAVE(I)*DNUM
                    YAVE(I)=YAVE(I)*DNUM
                    ZAVE(I)=ZAVE(I)*DNUM
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
                 ENDIF
#endif
              ENDDO
#if KEY_TSM==1
              IF(QTSM.AND.PIGSET) CALL PIGCVSET(XAVE,YAVE,ZAVE)
#endif

#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
              if (.not.q_domdec) then
#endif
                 TMERI(TIMDCNTRL)=TMERI(TIMDCNTRL)+ECLOCK()-TIMMER
                 !              CALL PSYNC()      !APH: PSYNC commented out for performance
                 TIMMER=ECLOCK()
#if KEY_DOMDEC==1
              endif
#endif
#if KEY_SPACDEC==1
              CALL SPACBR(XAVE,NATOM,ICPUMAP)
              CALL SPACBR(YAVE,NATOM,ICPUMAP)
              CALL SPACBR(ZAVE,NATOM,ICPUMAP)
#else /**/
#if KEY_DOMDEC==1
              if (q_domdec) then
                 call copy_to_all(xave,yave,zave)
                 IF(LIMCEN .AND. NTRANS > 0)THEN
                    XTMP=XAVE(1:NATOM)
                    YTMP=YAVE(1:NATOM)
                    ZTMP=ZAVE(1:NATOM)
                    CALL IMCENT(IMXCEN,IMYCEN,IMZCEN,bimag%IMCENF,NTRANS,IMTRNS, &
                         IMNAME,XAVE,YAVE,ZAVE,0,XDM1,YDM1,ZDM1,XDM2,YDM2,ZDM2,.false., &
                         NumLattice,ILATT,lattice_vector)
                    CALL IMCENT(IMXCEN,IMYCEN,IMZCEN,bimag%IMCENF,NTRANS,IMTRNS, &
                         IMNAME,XAVE,YAVE,ZAVE,0,XDM1,YDM1,ZDM1,XDM2,YDM2,ZDM2,.false., &
                         NumLattice,ILATT,lattice_vector)
                 ENDIF
              else
#endif
                 CALL VDGBR(XAVE,YAVE,ZAVE,1)
#if KEY_DOMDEC==1
              endif
#endif
#endif
#if KEY_DOMDEC==1
              if (.not.q_domdec) then
#endif
                 TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
                 TIMMER=ECLOCK()
#if KEY_DOMDEC==1
              endif
#endif
#endif
              !C MH05: this is already in WRITCV!!               IF(IOLEV > 0) THEN
              CALL WRITCV(XAVE,YAVE,ZAVE, &
#if KEY_CHEQ==1
                   CGAVE,QCG,                     &
#endif
                   NATOM,FREEAT,NFREAT,NPRIV, &
                   ISTEP,NDEGF,DELTA,NSAVC,NSTEP,TITLEA, &
                   NTITLA,IUNCRD,.FALSE.,.FALSE., (/ 0 /), &
                   .FALSE., (/ ZERO /))
              !C MH05               ENDIF
              NAVER=0

#if KEY_DOMDEC==1
!cheaper way is to do it as for X,Y,Z just below
              IF(q_domdec .and. LIMCEN .AND. NTRANS > 0)THEN
                  XAVE(1:NATOM)=XTMP(1:NATOM)
                  YAVE(1:NATOM)=YTMP(1:NATOM)
                  ZAVE(1:NATOM)=ZTMP(1:NATOM)
              ENDIF
#endif
              do ia=atfrst,atlast
#if KEY_DOMDEC==1
                 if (q_domdec) then
                    i = atoml(ia)
                 else
#endif
                    i = ia
#if KEY_DOMDEC==1
                 endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
                 IF (ATMASK(I)) THEN
#endif
                    XAVE(I)=ZERO
                    YAVE(I)=ZERO
                    ZAVE(I)=ZERO
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
                 ENDIF
#endif
              ENDDO
           ELSE
              !C MH05 WRITCV already has IOLEV check:   IF(IOLEV > 0) THEN
#if KEY_TSM==1
              IF((IOLEV > 0).AND.(QTSM.AND.PIGSET))CALL PIGCVSET(X,Y,Z)
#endif
#if KEY_DOMDEC==1
              !              call copy_to_root(x, y, z)
#endif
#if KEY_DOMDEC==1
              if (q_domdec) call copy_to_all(x, y, z)
#endif
#if KEY_DOMDEC==1
              IF(q_domdec .and. LIMCEN .AND. NTRANS > 0)THEN
                 XTMP=X(1:NATOM)
                 YTMP=Y(1:NATOM)
                 ZTMP=Z(1:NATOM)
! NOTE!!! This should be done with checking if more calls are necessary, until
! we have evertything back in the box.
!Don't let IMCENT know we are doing dynamics, so VX and XOLD do not get changed
                 CALL IMCENT(IMXCEN,IMYCEN,IMZCEN,bimag%IMCENF,NTRANS,IMTRNS, &
                      IMNAME,XTMP,YTMP,ZTMP,0,XDM1,YDM1,ZDM1,XDM2,YDM2,ZDM2,.false., &
                      NumLattice,ILATT,lattice_vector)
                 CALL IMCENT(IMXCEN,IMYCEN,IMZCEN,bimag%IMCENF,NTRANS,IMTRNS, &
                      IMNAME,XTMP,YTMP,ZTMP,0,XDM1,YDM1,ZDM1,XDM2,YDM2,ZDM2,.false., &
                      NumLattice,ILATT,lattice_vector)
                 CALL WRITCV(XTMP,YTMP,ZTMP, &
#if KEY_CHEQ==1
                   CG,QCG,                        &
#endif
                   NATOM,FREEAT,NFREAT,NPRIV, &
                   ISTEP,NDEGF,DELTA,NSAVC,NSTEP,TITLEA, &
                   NTITLA,IUNCRD,.FALSE.,.FALSE., (/ 0 /), &
                   .FALSE., (/ ZERO /))

              ELSE
#endif
              IF(IOLEV > 0) CALL PHWRIRSVR(X,Y,Z &
#if KEY_CHEQ==1
                  ,CG &
#endif
              )
              CALL WRITCV(X,Y,Z, &
#if KEY_CHEQ==1
                   CG,QCG,                        &
#endif
                   NATOM,FREEAT,NFREAT,NPRIV, &
                   ISTEP,NDEGF,DELTA,NSAVC,NSTEP,TITLEA, &
                   NTITLA,IUNCRD,.FALSE.,.FALSE., (/ 0 /), &
                   .FALSE., (/ ZERO /))
#if KEY_DOMDEC==1
              ENDIF
#endif
              !C MH05               ENDIF
           ENDIF
        ENDIF
     ENDIF
#if KEY_BLOCK==1 /*msld*/
     if (qmld .AND. NSAVL > 0 .AND. IOLEV.GT.0) then
        if (MOD(ISTEP,NSAVL) == 0) then
           CALL msld_writld(NBLOCK,NPRIV, &
                ISTEP,NSTEP, &
                delta)
        endif
     endif
     IF((QLDM.or.QLMC).AND.NSAVL > 0 .AND. IOLEV.GT.0) THEN
        IF(MOD(ISTEP,NSAVL) == 0) THEN
           CALL WRITLD(NBLOCK,NPRIV, &
                ISTEP,NSTEP, &
                delta)
        ENDIF
     ENDIF

#if KEY_LIBRARY == 1
     if (fill_lambdata .and. (istep .eq. istop)) then
        call lambdata_init(nbiasv, nblock)
        call lambdata_set_names()
        call lambdata_add_rows(istep, time)
     end if

     if (fill_msldata .and. (istep .eq. istop)) then
        call msldata_init(nstep)
        call msldata_set_names()
        call msldata_add_rows(istep, time)
     end if
#endif
#endif /* (msld)*/

     ! Do list updates if appropriate
     call timer_stop(T_dynamc)
     call timer_stop(T_dcntrl)
     call timer_start(T_list)

#if KEY_DOMDEC==1
     if (.not.q_domdec) then
#endif
#if KEY_PARALLEL==1
        TMERI(TIMDCNTRL)=TMERI(TIMDCNTRL)+ECLOCK()-TIMMER
#endif
#if KEY_DOMDEC==1
     endif
#endif

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('Updeci (dynamc)')
#endif

#if KEY_DOMDEC==1
     q_in_dynamc_main_loop = .true.
     q_print_output = .false.
#if KEY_MC==1
     if (nprint  >  0) then
#endif
        if(mod(istep,nprint) == 0) then
           q_print_output = .true.
        endif
#if KEY_MC==1
     endif
#endif

#endif
     CALL UPDECI(ISTEP-1,X,Y,Z,WMAIN,2,XOLD,YOLD,ZOLD,(/zero/),(/zero/),(/zero/))
#if KEY_DOMDEC==1
     q_in_dynamc_main_loop = .false.
#endif

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

     ! For domdec, we must update atlast after update_groupl has been called in updeci
     ! call above
#if KEY_DOMDEC==1
     if (q_domdec) atlast = natoml
#endif

#if KEY_DOMDEC==1
     if (.not.q_domdec) then
#endif
#if KEY_PARALLEL==1
        TIMMER=ECLOCK()
#endif
#if KEY_DOMDEC==1
     endif
#endif

     call timer_stop(T_list)

#if KEY_DOMDEC==1
     ! Communicate coordinates to recip CPUs
     ! NOTE: This has to be done AFTER call to UPDECI !
     if (q_domdec .and. q_split) then
        call timer_start(T_crdcomm2)
        ! APH NOTE: (x, y, z) coordinates are current here because we just came back from
        !           UPDECI
        call send_coord_to_recip(x, y, z, q_donb_updeci)
        call timer_stop(T_crdcomm2)
     endif
#endif

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('DIMS (dynamc)')
#endif
#if KEY_DIMS==1

     !     DIMS CONTROL LOOP

     IF(QDIMS) THEN
        DIMSON=.FALSE.
        DIMSCARTON=.FALSE.
        !     DIMSDONE Modifies variable DDONE
        QPRINT=MOD(ISTEP,NPRINT) == 0
        CALL DIMSDONE(X,Y,Z,WMAIN,QPRINT,ISTEP, &
             ATFRST,ATLAST)
        IF(DIMNM) THEN
           IF((MOD(ISTEP,NMSKIP) == 0.OR.(NMFIRST.AND.ISTEP.EQ.1)) &
                .AND.DIMNMON) THEN
              call timer_start(T_DIMNM)
              CALL DIMSNM()
              call timer_stop(T_DIMNM)
              MDISA=.FALSE.
              DIMSON=.TRUE.
              IF(DDFIX) THEN
                 CALL DIMSDYNFIX(IMOVE,1)
              ENDIF
           ELSE IF(DDFIX) THEN
              CALL DIMSDYNFIX(IMOVE,0)
           ENDIF
        ELSE IF(DCART) THEN
           DIMSCARTON=.TRUE.
        ELSE IF(DHARD) THEN
           DIMSON=.TRUE.
           CALL DIMSHARD(X,Y,Z, &
                DIMTARGXA,DIMTARGYA, DIMTARGZA, &
                ISTEP,NSTEP,NIDIMS,DIMSIDX, &
                DXDIMS,DYDIMS,DZDIMS,IDIMS, &
                ATFRST,ATLAST)
        ENDIF
        IF(DDONE) THEN
           ISTOP=ISTEP
           exit mainloop
        ENDIF
     ENDIF
     !     UPDATE DIMSNM POSITIONS

     IF(QDIMS) THEN
        IF(DIMNM) THEN
           I=MOD(ISTEP,BSKIP)
           IF(I <= NBIAS &
                .AND.(ISTEP >= NMSKIP.OR.NMFIRST) &
                .AND.DIMNMON) THEN
              call timer_start(T_DIMCOMB)
              CALL SCORENMMOVE(GAMMA,X,Y,Z,WMAIN,I,ATFRST,ATLAST)
              call timer_stop(T_DIMCOMB)
              DIMSON=.TRUE.
           ElSE
              DIMSON=.FALSE.
           ENDIF
        ENDIF
        IF(DIMSON) THEN
           !     -- Just an extra ENERGY CALL needed to compute dims score
           CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
           !     Unbiased energy
           CALL UPDATEDIMSEQ(EPROP(EPOT))
           CALL UPDATEDIMSPOS(X,Y,Z,ATFRST,ATLAST)
        ENDIF
        IF(DIMNM.AND.NMPRINT) THEN
           IF(ISTEP > NMSKIP.AND.DIMNMON) THEN
              CALL DIMSWRITNM(X,Y,Z, &
                   DXDIMS,DYDIMS,DZDIMS, &
                   XScratch,YScratch,ZScratch, &
                   IUDIMSNM, &
                   NSAVC, DELTA, ISTEP, NSTEP,0)
           ELSE
              CALL DIMSWRITNM(X,Y,Z, &
                   DXDIMS,DYDIMS,DZDIMS, &
                   XScratch,YScratch,ZScratch, &
                   IUDIMSNM, &
                   NSAVC, DELTA, ISTEP, NSTEP,1)
           ENDIF
        ENDIF
     ENDIF
#endif

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

#if KEY_RXNCOR==1
     ! yj070912.bfx: MDSTEP2 is moved up (see rxnene.src)
     !     umbmdstep is a variable saved in common/rxncmi of rxncom.f90
     UMBMDSTEP=ISTEP
#endif

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('Energy (dynamc)')
#endif

#if KEY_DOMDEC==1
     if (.not.q_domdec) then
#endif
#if KEY_PARALLEL==1
        TMERI(TIMDCNTRL)=TMERI(TIMDCNTRL)+ECLOCK()-TIMMER
#endif
#if KEY_DOMDEC==1
     endif
#endif

     call set_qdyncall()
     call energy(x,y,z,dx,dy,dz,bnbnd,bimag,1)
     qdyncall=.false.

     !
#if KEY_STRINGM==1 /*  VO stringm */
     !=========================================
     if (smcv_on) call smcv_main(x,y,z,xcomp,ycomp,zcomp,&
     &                            amass(1:natom),dx,dy,dz,istep)
     if (ftsm_on) call ftsm_main(x(1:natom),y(1:natom),z(1:natom),&
     &                            xcomp(1:natom),ycomp(1:natom),zcomp(1:natom),&
     &                            dx(1:natom),dy(1:natom),dz(1:natom),&
     &                            amass(1:natom),&
     &                            istep,wmain(1:natom),bnbnd,bimag)
     !=========================================
#endif /* stringm */
     !
     call timer_start(T_dcntrl)
     call timer_start(T_dynamc)
#if KEY_DOMDEC==1
     if (.not.q_domdec) then
#endif
#if KEY_PARALLEL==1
        TIMMER=ECLOCK()
#endif
#if KEY_DOMDEC==1
     endif
#endif

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

#if KEY_SGLD==1
     !  WXW    perform force average
     IF(QSGLD.OR.QSGMD)THEN
        CALL SGFAVG(ATFRST,ATLAST,EPROP(EPOT),IMOVE, &
             DX,DY,DZ)
     ENDIF
#endif

#if KEY_DIMS==1
     IF(DIMSON) THEN
        !     Biased Energy
        CALL UPDATEDIMSED(EPROP(EPOT))
        CALL DIMSSCORE(PRNLEV >= 3)
     ENDIF
#endif /* DIMS*/
#if KEY_BLOCK==1 /*ldm_2*/
     if(rstp) call ldm_rstp_dynamc(natom, dx, dy, dz, nblock, iblckp)
     if(nsavl > 0 .and. iolev.gt.0)then
        if(mod(istep,nsavl) == 0) then
           if(iresd /= 0) then
              ! print out the total harmonic restraining potential
              write(iresd,'(i7,2x,e12.4)') istep/nsavl, eterm(resd)
           endif
           if(ilapot /= 0) then
              ! print out the partial potential vi
              write(ilapot,'(i5,2x,100e12.4)') istep/nsavl, &
                   (biptnlam(i),i=lstrt, nblock)
           endif

           if(ilaf /= 0)then
              ! print out the restraining potential
              if(nrst /= 2)then
                 write(ilaf,'(i5,2x,100e12.4)')istep/nsavl, &
                      (lmdcoef*biptnlam(i)*(1.0-bixlam(i)**2),i=lstrt, nblock)
              else
                 write(ilaf,'(i5,2x,100e12.4)')istep/nsavl, &
                      (lmdcoef*bfrst(i)*(1.0-bixlam(i)**2),i=lstrt, nblock)
              endif
           endif
        endif
     endif
#endif /*  (ldm_2)*/

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('Copy x to xcomp')
#endif

#if KEY_SPACDEC==1 /*spacdec*/
     a0=1
     a1-nmyatm
#else /* (spacdec)*/
     a0=atfrst
     a1=atlast
#if KEY_STRINGM==1 /*  VO   Voronoi tests require complete xcomp */
     if (voronoi_hist_on) then ; a0=1; a1=natom ; endif
#endif
#endif /* (spacdec)*/

#if KEY_DOMDEC==1
     if (q_domdec) then
!$omp parallel do schedule(static) private(ia, i)
        do ia=a0,a1
           i = atoml(ia)
           XCOMP(I)=X(I)
           YCOMP(I)=Y(I)
           ZCOMP(I)=Z(I)
        enddo
!$omp end parallel do
     else
#endif
        do ia=a0,a1
#if KEY_SPACDEC==1 /*spacdec*/
           I=MYSATARR(ia)
#else /* (spacdec)*/
           i = ia
#endif /* (spacdec)*/
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif
              XCOMP(I)=X(I)
              YCOMP(I)=Y(I)
              ZCOMP(I)=Z(I)
#if KEY_PARASCAL==1
           ENDIF
#endif
        enddo
#if KEY_DOMDEC==1
     endif
#endif
!!$     do ia=a0,a1
!!$##IF DOMDEC
!!$        if (q_domdec) then
!!$           i = atoml(ia)
!!$        else
!!$##ENDIF
!!$##IF SPACDEC (spacdec)
!!$           I=MYSATARR(ia)
!!$##ELSE (spacdec)
!!$           i = ia
!!$##ENDIF (spacdec)
#if KEY_DOMDEC==1
!!$        endif
#endif
#if KEY_PARASCAL==1
!!$        IF(JPBLOCK(I) == MYNOD) THEN
#endif
!!$           XCOMP(I)=X(I)
!!$           YCOMP(I)=Y(I)
!!$           ZCOMP(I)=Z(I)
#if KEY_PARASCAL==1
!!$        ENDIF
#endif
!!$     enddo

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
     if (.not.q_domdec) then
#endif
        TMERI(TIMDCNTRL)=TMERI(TIMDCNTRL)+ECLOCK()-TIMMER
        !      CALL PSYNC()      !APH: PSYNC commented out for performance
        TIMMER=ECLOCK()
#if KEY_DOMDEC==1
     endif
#endif
#endif

#if KEY_TNPACK==1 /*eulermain1*/
     IF(QEULER) THEN
        ! save current forces and clear
        DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif
              !           save current forces
              VX(I)=DX(I)
              VY(I)=DY(I)
              VZ(I)=DZ(I)
              !           clear the forces to facilitate getting restraint reference
              DX(I)=ZERO
              DY(I)=ZERO
              DZ(I)=ZERO
#if KEY_PARASCAL==1
           ENDIF
#endif
        ENDDO
        !     save current energies and properties
        EPPRTMP(:) = EPROP(:)
        ETPRTMP(:) = ETERM(:)
        EVPRTMP(:) = EPRESS(:)
     ENDIF
#endif /* (eulermain1)*/

     ! compute u(n+1/2)
#if KEY_TPS==1 /*tps_save_seed*/
     !     Save seed (and integration direction) for TPS
     IF (QTPS) ITPSSD = ISEED*PDIR
#endif /*  (tps_save_seed)*/
     IF(ILANG == 1) THEN
#if KEY_DIMS==1
        IF(DIMSCARTON) THEN
           CALL DIMSDLNGV(GAMMA,ISEED, &
                XCOMP,YCOMP,ZCOMP, XOLD, YOLD, ZOLD, &
                DIMTARGXA, DIMTARGYA, DIMTARGZA, &
                XScratch,YSCratch,ZScratch, &
                DXDIMS,DYDIMS,DZDIMS, &
                XNEW,YNEW,ZNEW, &
                WMAIN,NATOM2,NATOM3,DWIDTH,QPRINT, &
                ATFRST,ATLAST,IDIMS)
        ELSE
#endif

           CALL DLNGV(GAMMA,ISEED)

#if KEY_DIMS==1
        ENDIF
#endif
     ENDIF
#if KEY_SGLD==1
     !WXW    considering the guiding effect
     IF(QSGLD.OR.QSGMD)THEN
        IF(QSGMD)THEN
           CALL SGMDW(ATFRST,ATLAST,DELTA,NDEGF,EPROP(TEMPS), &
                 IMOVE,AMASS,X,Y,Z,XOLD,YOLD,ZOLD,DX,DY,DZ)
        ELSE
           CALL SGLDW(ATFRST,ATLAST,DELTA,NDEGF,EPROP(TEMPS),&
                 IMOVE,AMASS,X,Y,Z,XOLD,YOLD,ZOLD,DX,DY,DZ)
        ENDIF
     ENDIF
#endif
#if KEY_EMAP==1
     !WXW    move map objects
     IF(LEMAPLD)THEN
        CALL EMAPLD(DELTA)
     ENDIF
#endif

#if KEY_TSALLIS==1 /*TsallisMD*/
     !     ARD 01-06-12
     !     Tsallis weighted dynamics
     IF (QTSALL) THEN
        !       TSQ is (1 - q), TSBETA is beta, TSEMIN is minimum E
        !       TSFACT is q / (1 - ((1-q)*beta*(E-Emin)))
        TSFACT = (ONE - TSQ) / &
             (ONE - (TSQ*TSBETA*(EPROP(EPOT)-TSEMIN)))
        DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif
              FACT=GAMMA(I+NATOM)*TSFACT
              ALPHA=GAMMA(I+NATOM2)
              XNEW(I)=ALPHA*XOLD(I)-DX(I)*FACT
              YNEW(I)=ALPHA*YOLD(I)-DY(I)*FACT
              ZNEW(I)=ALPHA*ZOLD(I)-DZ(I)*FACT
#if KEY_PARASCAL==1
           ENDIF
#endif
        ENDDO
     ELSE
#endif /*      (TsallisMD)*/

#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_start('Calculate new step vector xnew')
#endif

#if KEY_SPACDEC==1 /*spacdec*/
        a0=1
        a1=nmyatm
#else /* (spacdec)*/
        a0=atfrst
        a1=atlast
#endif /* (spacdec)*/

#if KEY_DOMDEC==1
        if (q_domdec) then

!$omp parallel do schedule(static) private(ia, i, fact, alpha)
              do ia=a0,a1
                 i = atoml(ia)
                 FACT=GAMMA(I+NATOM)
                 ALPHA=GAMMA(I+NATOM2)
                 XNEW(I)=ALPHA*XOLD(I)-DX(I)*FACT
                 YNEW(I)=ALPHA*YOLD(I)-DY(I)*FACT
                 ZNEW(I)=ALPHA*ZOLD(I)-DZ(I)*FACT
              enddo
!$omp end parallel do
        else
#endif
           do ia=a0,a1
#if KEY_SPACDEC==1 /*spacdec*/
              I=MYSATARR(ia)
#else /* (spacdec)*/
              i = ia
#endif /* (spacdec)*/
#if KEY_PARASCAL==1
              IF(JPBLOCK(I) == MYNOD) THEN
#endif
                 FACT=GAMMA(I+NATOM)
                 ALPHA=GAMMA(I+NATOM2)
                 XNEW(I)=ALPHA*XOLD(I)-DX(I)*FACT
                 YNEW(I)=ALPHA*YOLD(I)-DY(I)*FACT
                 ZNEW(I)=ALPHA*ZOLD(I)-DZ(I)*FACT

#if KEY_PARASCAL==1
              ENDIF
#endif
           enddo
#if KEY_DOMDEC==1
        endif
#endif

#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_stop()
#endif

#if KEY_TSALLIS==1
     ENDIF
#endif

#if KEY_BLOCK==1 /*msld*/ /*LDM*/
     if (qmld) call msld_updatetheta(nblock,delta)
     if(qldm .or. qlmc) call ldm_prop2_dynamc(nblock,istep,delta)
#endif /* (msld)  LDM*/
#if KEY_TSM==1
     IF(QTSM.AND.PIGSET) CALL BACK0(XNEW,YNEW,ZNEW)
#endif

#if KEY_TNPACK==1 /*eulermain2*/
     IF(QEULER) THEN
        ! Do implicit Euler projection and minimization
        do ia=atfrst,atlast
#if KEY_DOMDEC==1
           if (q_domdec) then
              i = atoml(ia)
           else
#endif
              i = ia
#if KEY_DOMDEC==1
           endif
#endif
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif
              !           compute reference position
              RXHARM(I)=XCOMP(I)+XNEW(I)
              RYHARM(I)=YCOMP(I)+YNEW(I)
              RZHARM(I)=ZCOMP(I)+ZNEW(I)
              write(outu,33) 'x curr',xcomp(i),ycomp(i),zcomp(i)
              write(outu,33) 'x ref',rxharm(i),ryharm(i),rzharm(i)
33            format(a10,3f20.10)
              !           generate improved starting coordinates for minimization
              !           based on current forces.
              FACT=GAMMA(I+NATOM)
              X(I)=RXHARM(I)-VX(I)*FACT*LIEFF
              Y(I)=RYHARM(I)-VY(I)*FACT*LIEFF
              Z(I)=RZHARM(I)-VZ(I)*FACT*LIEFF
              write(outu,33) 'x guess',x(i),y(i),z(i)
              !           save current forces including random forces.
              VX(I)=VX(I)+DX(I)
              VY(I)=VY(I)+DY(I)
              VZ(I)=VZ(I)+DZ(I)
              write(outu,33) 'force',vx(i),vy(i),vz(i)
#if KEY_PARASCAL==1
           ENDIF
#endif
        ENDDO

        !        minimize to get next integration point.
        STR='NPRINT 1 NSTEP 10 STEP 0.0001'
        STRLN=32
        QEHARM=.TRUE.
        !C         CALL ABNER(STR,STRLN)
        !C         CALL POWELL(STR,STRLN,X,Y,Z)
        CALL TNDRIV(STR,STRLN,QESTRT)

        do ia=atfrst,atlast
#if KEY_DOMDEC==1
           if (q_domdec) then
              i = atoml(ia)
           else
#endif
              i = ia
#if KEY_DOMDEC==1
           endif
#endif
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif
              !           restore original forces
              DX(I)=VX(I)
              DY(I)=VY(I)
              DZ(I)=VZ(I)
              !           restore original coordintes
              !           generate new step from minimized coordinates
              XNEW(I)=X(I)-XCOMP(I)
              YNEW(I)=Y(I)-YCOMP(I)
              ZNEW(I)=Z(I)-ZCOMP(I)
#if KEY_PARASCAL==1
           ENDIF
#endif
        ENDDO
        !        restore properties and energies.
        EPROP(:) = EPPRTMP(:)
        ETERM(:) = ETPRTMP(:)
        EPRESS(:) = EVPRTMP(:)
        QEHARM=.FALSE.
     ENDIF
#endif /* (eulermain2)*/

     call timer_start(T_saveshake)

     ! Save coordinates for SHAKE (if present).
     IF(QHOLO) THEN
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_start('Holonoma (1) (dynamc)')
#endif
#if KEY_DOMDEC==1
        if (q_domdec) then
           if (cpu_vendor == CPU_AMD) then
!$omp parallel private(ia, i)
!$omp do schedule(static)
              do ia=atfrst,atlast
                 i = atoml(ia)
                 VX(I)=XNEW(I)
                 VY(I)=YNEW(I)
                 VZ(I)=ZNEW(I)
              enddo
!$omp end do nowait
!$omp do schedule(static)
              do ia=atfrst,atlast
                 i = atoml(ia)
                 X(I)=XNEW(I)+XCOMP(I)
                 Y(I)=YNEW(I)+YCOMP(I)
                 Z(I)=ZNEW(I)+ZCOMP(I)
              enddo
!$omp end do
!$omp end parallel
           else
!$omp parallel do schedule(static) private(ia, i)
              do ia=atfrst,atlast
                 i = atoml(ia)
                 VX(I)=XNEW(I)
                 VY(I)=YNEW(I)
                 VZ(I)=ZNEW(I)
                 X(I)=XNEW(I)+XCOMP(I)
                 Y(I)=YNEW(I)+YCOMP(I)
                 Z(I)=ZNEW(I)+ZCOMP(I)
              enddo
!$omp end parallel do
           endif
        else
#endif
           do i=atfrst,atlast
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              IF (ATMASK(I)) THEN
#endif
                 VX(I)=XNEW(I)
                 VY(I)=YNEW(I)
                 VZ(I)=ZNEW(I)
                 X(I)=XNEW(I)+XCOMP(I)
                 Y(I)=YNEW(I)+YCOMP(I)
                 Z(I)=ZNEW(I)+ZCOMP(I)
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              ENDIF
#endif
           enddo
#if KEY_DOMDEC==1
        endif
#endif
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_stop()
#endif
        ! Process SHAKE if present (not the most efficient way).
        call timer_stop(T_dynamc)
        call timer_stop(T_dcntrl)
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_start('Holonoma (2) (dynamc)')
#endif
        CALL HOLONOMA(X,Y,Z,XCOMP,YCOMP,ZCOMP,.TRUE.,.TRUE.,QOK)
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_stop()
#endif
#if KEY_RXNCONS==1 /*jc070701*/
        ! Print out lagrange multiplier, Z^-1/2, and G
        IF(LRXCNS) THEN
#if KEY_PARALLEL==1
           IF(MYNOD == 0) THEN
#endif
              IF(IUNLAG > 1.AND.NSAVLG.GT.0.AND. &
                   (MOD(ISTEP,NSAVLG) == 0)) THEN
                 WRITE(IUNLAG,222) ISTEP,LAGRANM/DELTA2,FRCONS, &
                      ZFAC,GFAC,RCNSDL,RCNSDS,RCNSDSM, &
                      RCNSDC,RCNSDCM,RCNSDR,RCNSDRC,RCNSAMF
              ENDIF
#if KEY_PARALLEL==1
           ENDIF
#endif
        ENDIF
222     FORMAT(I10,2E16.6,2F12.4,8F16.6)
#endif /* (jc070701)*/
        call timer_start(T_dcntrl)
        call timer_start(T_dynamc)
#if KEY_PERT==1
        !sb      Have to actually calculate the constraint correction
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_start('Holonoma (3) (dynamc)')
#endif
        IF (QPSHAKE) CALL EPCONS(DELTA2)
#endif

#if KEY_DOMDEC==1
        if (q_domdec) then
!$omp parallel do schedule(static) private(ia, i, fact)
              do ia=atfrst,atlast
                 i = atoml(ia)
                 XNEW(I)=X(I)-XCOMP(I)
                 YNEW(I)=Y(I)-YCOMP(I)
                 ZNEW(I)=Z(I)-ZCOMP(I)
                 IF (GAMMA(I+NATOM) /= ZERO) THEN
                    FACT=MINONE/GAMMA(I+NATOM)
                 ELSE
                    FACT=ZERO
                 ENDIF
                 VX(I)=(XNEW(I)-VX(I))*FACT
                 VY(I)=(YNEW(I)-VY(I))*FACT
                 VZ(I)=(ZNEW(I)-VZ(I))*FACT
              enddo
!$omp end parallel do

        else
#endif
           do i=atfrst,atlast
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              IF (ATMASK(I)) THEN
#endif
                    XNEW(I)=X(I)-XCOMP(I)
                    YNEW(I)=Y(I)-YCOMP(I)
                    ZNEW(I)=Z(I)-ZCOMP(I)
                    IF (GAMMA(I+NATOM) /= ZERO) THEN
                       FACT=MINONE/GAMMA(I+NATOM)
                    ELSE
                       FACT=ZERO
                    ENDIF
                    VX(I)=(XNEW(I)-VX(I))*FACT
                    VY(I)=(YNEW(I)-VY(I))*FACT
                    VZ(I)=(ZNEW(I)-VZ(I))*FACT

#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              ENDIF
#endif
           enddo
#if KEY_DOMDEC==1
        endif
#endif
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_stop()
#endif
        ! Add shake "force" to the internal virial.
#if KEY_TSM==1
        ! We don't want the "back atom" forces (in vx) contributing to the
        ! virial and we need to zero the shake corrected "back atom" half-step
        ! velocities (in xnew).
        IF(QTSM.AND.PIGSET) THEN
           CALL BACK0(XNEW,YNEW,ZNEW)
           CALL BACK0(VX,VY,VZ)
        ENDIF
#endif
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_start('Holonoma (4) (dynamc)')
#endif
#if KEY_DOMDEC==1
        if (q_domdec) then
           ! virshk fills in gcarr(1:9)
           CALL VIRSHK(EPRESS(VIXX:VIZZ),NATOM,XCOMP,YCOMP,ZCOMP,VX,VY,VZ,gcarr,.true.)
        else
#endif
           CALL VIRSHK(EPRESS(VIXX:VIZZ),NATOM,XCOMP,YCOMP,ZCOMP,VX,VY,VZ)
#if KEY_DOMDEC==1
        endif
#endif
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_stop()
#endif
     ENDIF

     call timer_stop(T_saveshake)

     ! Constant temperature scaling if requested.
     const_temp: IF(QCNSTT) THEN

        call timer_start(T_constt)
        TEMPI=ZERO
        TOTKEN=ZERO

        do ia=atfrst,atlast
#if KEY_DOMDEC==1
           if (q_domdec) then
              i = atoml(ia)
           else
#endif
              i = ia
#if KEY_DOMDEC==1
           endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           IF (ATMASK(I)) THEN
#endif
              FACT=GAMMA(I+NATOM3)
              VXI=(XNEW(I)+XOLD(I))*FACT
              VYI=(YNEW(I)+YOLD(I))*FACT
              VZI=(ZNEW(I)+ZOLD(I))*FACT
              TEMPI=TEMPI+AMASS(I)*(VXI**2+VYI**2+VZI**2)
              TOTKEN=TOTKEN+AMASS(I)*(XNEW(I)**2+YNEW(I)**2+ZNEW(I)**2)
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           ENDIF
#endif
        ENDDO
        TEMPI = TEMPI / (KBOLTZ*NDEGF)
        TOTKEN = TOTKEN / (KBOLTZ*NDEGF*DELTA2)
#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
        if (.not.q_domdec) then
#endif
           TMERI(TIMDCNTRL)=TMERI(TIMDCNTRL)+ECLOCK()-TIMMER
           !         CALL PSYNC()       !APH: PSYNC commented out for performance
           TIMMER=ECLOCK()
#if KEY_DOMDEC==1
        endif
#endif

        GCARR2(1)=TOTKEN
        GCARR2(2)=TEMPI
        CALL GCOMB(GCARR2,2)
        TOTKEN=GCARR2(1)
        TEMPI=GCARR2(2)

#if KEY_DOMDEC==1
        if (.not.q_domdec) then
#endif
           TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
           TIMMER=ECLOCK()
#if KEY_DOMDEC==1
        endif
#endif
#endif
        IF(TOTKEN < ONE) TOTKEN=ONE

        IF(QCNSTTX) THEN
           FACT=SQRT(TREF/TEMPI)

           do ia=atfrst,atlast
#if KEY_DOMDEC==1
              if (q_domdec) then
                 i = atoml(ia)
              else
#endif
                 i = ia
#if KEY_DOMDEC==1
              endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              IF (ATMASK(I)) THEN
#endif
                 XNEW(I) = (TWO*FACT-ONE)*XOLD(I)+FACT * (XNEW(I)-XOLD(I))
                 YNEW(I) = (TWO*FACT-ONE)*YOLD(I)+FACT * (YNEW(I)-YOLD(I))
                 ZNEW(I) = (TWO*FACT-ONE)*ZOLD(I)+FACT * (ZNEW(I)-ZOLD(I))
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              ENDIF
#endif
           ENDDO
        ELSE
           FACT=SQRT(ONE+(TIMEST/TCOUPL)*(TREF-TEMPI)/TOTKEN)

           do ia=atfrst,atlast
#if KEY_DOMDEC==1
              if (q_domdec) then
                 i = atoml(ia)
              else
#endif
                 i = ia
#if KEY_DOMDEC==1
              endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              IF (ATMASK(I)) THEN
#endif
                 XNEW(I) = FACT * XNEW(I)
                 YNEW(I) = FACT * YNEW(I)
                 ZNEW(I) = FACT * ZNEW(I)
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              ENDIF
#endif
           ENDDO
        ENDIF

#if KEY_TMD==1 /*tmd*/
        ! In the case that only TMD is present, not the shake.
        ! Note, nothing else is supported in this option. jma

        IF(QTMD .AND. (.NOT. QSHAKE)) THEN
           do ia=atfrst,atlast
#if KEY_DOMDEC==1
              if (q_domdec) then
                 i = atoml(ia)
              else
#endif
                 i = ia
#if KEY_DOMDEC==1
              endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              IF (ATMASK(I)) THEN
#endif
                 X(I)=XNEW(I)+XCOMP(I)
                 Y(I)=YNEW(I)+YCOMP(I)
                 Z(I)=ZNEW(I)+ZCOMP(I)
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              ENDIF
#endif
           ENDDO

           !           CALL HOLONOMA(X,Y,Z,XCOMP,YCOMP,ZCOMP,.TRUE.,.TRUE.,QOK)

           ! Take care of the distance increment. It is done here to
           ! avoid the trouble with the interation between shake and shift.
           if(tmdfmax2 < zero)tmdrhof=tmdrhof-drho  ! avv050701
           !            IF(prnlev > 5) write(outu,*)'DYNAMC:  tmdrhof,drho  ',
           !     &                                             tmdrhof, drho
           IF(.NOT.QZETA) THEN
              !%% AvdV changes
              call shift_tmd(x,y,z,xcomp,ycomp,zcomp, &
                   ixtar,iytar,iztar, &
                   amass,natom,tol_corr, &
                   jstmd, &
                   bnbnd,bimag,teterm,teprop,tepress,tmde, &
                   itmdx,itmdy,itmdz, &
                   istep,tmdsh1)
              tmdsh1=.false.
              !%% AvdV end changes
           ELSE
              if(drho > ZERO .and. tmdrhof < zetatg) tmdrhof=zetatg ! DINC > 0
              if(drho < ZERO .and. tmdrhof > zetatg) tmdrhof=zetatg ! DINC < 0
              call preshz(xcomp,ycomp,zcomp, &
                   ixtar,iytar,iztar, &
                   ixtar2,iytar2,iztar2, &
                   amass, &
                   jstmd, &
                   ixgtmd,iygtmd,izgtmd, &
                   lagsum)
              call shiftz(x,y,z,xcomp,ycomp,zcomp, &
                   ixtar,iytar,iztar, &
                   ixtar2,iytar2,iztar2, &
                   amass,tol_corr,               & ! fix posted on charmm.org
                   jstmd, &
                   ix1tmd,iy1tmd,iz1tmd, &
                   ix2tmd,iy2tmd,iz2tmd, &
                   ixgtmd,iygtmd,izgtmd, &
                   lagsum)
           ENDIF

           do ia=atfrst,atlast
#if KEY_DOMDEC==1
              if (q_domdec) then
                 i = atoml(ia)
              else
#endif
                 i = ia
#if KEY_DOMDEC==1
              endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              IF (ATMASK(I)) THEN
#endif
                 XNEW(I)=X(I)-XCOMP(I)
                 YNEW(I)=Y(I)-YCOMP(I)
                 ZNEW(I)=Z(I)-ZCOMP(I)
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              ENDIF
#endif
           ENDDO

           GOTO 100 !Jump over the next shake option. jma

        ENDIF !(QTMD .and. (.not. QSHAKE))
#endif /* (tmd)*/

        ! Process SHAKE if present (not the most efficient way).
        IF(QHOLO) THEN
           do ia=atfrst,atlast
#if KEY_DOMDEC==1
              if (q_domdec) then
                 i = atoml(ia)
              else
#endif
                 i = ia
#if KEY_DOMDEC==1
              endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              IF (ATMASK(I)) THEN
#endif
                 X(I)=XNEW(I)+XCOMP(I)
                 Y(I)=YNEW(I)+YCOMP(I)
                 Z(I)=ZNEW(I)+ZCOMP(I)
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              ENDIF
#endif
           ENDDO

#if KEY_TMD==1

           ! Put the SHIFT routine here to include the constant temperature
           ! dynamics and the bond length SHAKE. Do the iteration between
           ! the SHIFT and SHAKE until both are satisfied.

           ! Take care of the distance increment. It is done here to
           ! avoid the trouble with the interation between shake and shift.
           if(tmdfmax2 < zero)tmdrhof=tmdrhof-drho  ! avv050701
           IF(QZETA) THEN
              if(drho > ZERO .and. tmdrhof < zetatg) tmdrhof=zetatg ! DINC > 0
              if(drho < ZERO .and. tmdrhof > zetatg) tmdrhof=zetatg ! DINC < 0
              call preshz(xcomp,ycomp,zcomp, &
                   ixtar,iytar,iztar, &
                   ixtar2,iytar2,iztar2, &
                   amass, &
                   jstmd, &
                   ixgtmd,iygtmd,izgtmd, &
                   lagsum)
           ENDIF
11         continue
#endif

           call timer_stop(T_dynamc)
           call timer_stop(T_dcntrl)
           CALL HOLONOMA(X,Y,Z,XCOMP,YCOMP,ZCOMP,.TRUE.,.TRUE.,QOK)
           call timer_start(T_dcntrl)
           call timer_start(T_dynamc)
#if KEY_TMD==1
           IF(QTMD) THEN
              IF(.NOT.QZETA) THEN
                 !%% AvdV changes
                 if(tmdsh1.or.(tmdfmax2 < zero))then
                    call shift_tmd(x,y,z,xcomp,ycomp,zcomp, &
                         ixtar,iytar,iztar, &
                         amass,natom,tol_corr, &
                         jstmd, &
                         bnbnd,bimag,teterm,teprop,tepress,tmde, &
                         itmdx,itmdy,itmdz, &
                         istep,tmdsh1)
                    tmdsh1=.false.
                    if(tol_corr  >  tenm6) goto 11
                 endif
                 !%% AvdV end changes
              ELSE
                 call shiftz(x,y,z,xcomp,ycomp,zcomp, &
                      ixtar,iytar,iztar, &
                      ixtar2,iytar2,iztar2, &
                      amass,tol_corr, &
                      jstmd, &
                      ix1tmd,iy1tmd,iz1tmd, &
                      ix2tmd,iy2tmd,iz2tmd, &
                      ixgtmd,iygtmd,izgtmd, &
                      lagsum)
                 if(tol_corr  >  tenm6) goto 11
              ENDIF
           ENDIF

#endif

           do ia=atfrst,atlast
#if KEY_DOMDEC==1
              if (q_domdec) then
                 i = atoml(ia)
              else
#endif
                 i = ia
#if KEY_DOMDEC==1
              endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              IF (ATMASK(I)) THEN
#endif
                 XNEW(I)=X(I)-XCOMP(I)
                 YNEW(I)=Y(I)-YCOMP(I)
                 ZNEW(I)=Z(I)-ZCOMP(I)
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
              ENDIF
#endif
           ENDDO
        ENDIF
#if KEY_TSM==1
        IF(QTSM.AND.PIGSET) CALL BACK0(XNEW,YNEW,ZNEW)
#endif

#if KEY_TMD==1
100     CONTINUE
        !%% AvdV addition
        if(qtmd.and.tmdwr)then
           call chmalloc('dynamc.src','DYNAMC','itmptx',natom,crl=itmptx)
           call chmalloc('dynamc.src','DYNAMC','itmpty',natom,crl=itmpty)
           call chmalloc('dynamc.src','DYNAMC','itmptz',natom,crl=itmptz)
           call tmdan(ixtar,iytar,iztar, &
                itmdx,itmdy,itmdz,bnbnd,bimag, &
                tmde,teterm,teprop,tepress, &
                jstmd,istep,atfrst,atlast, &
                itmptx,itmpty,itmptz)
           tmdwr=.false.
           call chmdealloc('dynamc.src','DYNAMC','itmptx',natom,crl=itmptx)
           call chmdealloc('dynamc.src','DYNAMC','itmpty',natom,crl=itmpty)
           call chmdealloc('dynamc.src','DYNAMC','itmptz',natom,crl=itmptz)
        endif
        !%% AvdV end addition
#endif

        call timer_stop(T_constt)
     ENDIF const_temp

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('Compute velocities (dynamc)')
#endif

     ! compute the velocities
     !         write(*,*)'DYNAMC-24>'
#if KEY_SPACDEC==1 /*spacdec*/
     a0=1
     a1=nmyatm
#else /* (spacdec)*/
     a0=atfrst
     a1=atlast
#endif /* (spacdec)*/

#if KEY_DOMDEC==1
     if (q_domdec) then

!$omp parallel do schedule(static) private(ia, i, fact)
           do ia=1,natoml
              i = atoml(ia)
              FACT=GAMMA(I+NATOM3)
              VX(I)=(XNEW(I)+XOLD(I))*FACT
              VY(I)=(YNEW(I)+YOLD(I))*FACT
              VZ(I)=(ZNEW(I)+ZOLD(I))*FACT
           enddo
!$omp end parallel do

     else
#endif
        do ia=a0,a1
#if KEY_SPACDEC==1 /*spacdec*/
           I=MYSATARR(ia)
#else /* (spacdec)*/
           i = ia
#endif /* (spacdec)*/
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif
              FACT=GAMMA(I+NATOM3)
              VX(I)=(XNEW(I)+XOLD(I))*FACT
              VY(I)=(YNEW(I)+YOLD(I))*FACT
              VZ(I)=(ZNEW(I)+ZOLD(I))*FACT

#if KEY_PARASCAL==1
           ENDIF
#endif
        ENDDO
#if KEY_DOMDEC==1
     endif
#endif


#if KEY_MEHMC==1 /*mehmc_add*/
     !     If doing MEHMC, add the velocities to the averages
     IF (RMEHMC  >  ZERO) THEN
        do ia=atfrst,atlast
#if KEY_DOMDEC==1
           if (q_domdec) then
              i = atoml(ia)
           else
#endif
              i = ia
#if KEY_DOMDEC==1
           endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           IF (ATMASK(I)) THEN
#endif
              VMEAVX(I)=(ONE-RMEHMC)*VMEAVX(I) + RMEHMC*VX(I)
              VMEAVY(I)=(ONE-RMEHMC)*VMEAVY(I) + RMEHMC*VY(I)
              VMEAVZ(I)=(ONE-RMEHMC)*VMEAVZ(I) + RMEHMC*VZ(I)
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           ENDIF
#endif
        ENDDO
     ENDIF
#endif /*    (mehmc_add)*/

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

#if KEY_TSM==1
     IF(QTSM.AND.PIGSET) CALL BACK0(VX,VY,VZ)
#endif

#if KEY_PERT==1
     IF(QPERT) THEN
        IF(QACCUM) CALL EPSUM
        CALL EWORKT(VX,VY,VZ,DX,DY,DZ)
     ENDIF
#endif

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('Heuristic_check (dynamc)')
#endif

     call timer_start(T_heur)
     call heuristic_check(istep, xcomp, ycomp, zcomp, xnew, ynew, znew, .true.)
     call timer_stop(T_heur)

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

     call timer_start(T_tvir)

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('Compute temperature (dynamc)')
#endif

     ! compute temperature (2*ke) and  virial contribution (ke)
     TEMPI=ZERO
     TOTKEN=ZERO

#if KEY_DOMDEC==1
     if (q_domdec) then

!$omp parallel do schedule(static) private(ia, i, rval) reduction(+:tempi,totken)
           do ia=1,natoml
              i = atoml(ia)
              RVAL=AMASS(I)*(VX(I)**2+VY(I)**2+VZ(I)**2)
              VK(I)=VK(I)+RVAL
              TEMPI=TEMPI+RVAL
              TOTKEN=TOTKEN+AMASS(I)*(XNEW(I)**2+YNEW(I)**2+ZNEW(I)**2)
           ENDDO
!$omp end parallel do

     else
#endif
        do i=atfrst,atlast
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           IF (ATMASK(I)) THEN
#endif
              RVAL=AMASS(I)*(VX(I)**2+VY(I)**2+VZ(I)**2)
              VK(I)=VK(I)+RVAL
              TEMPI=TEMPI+RVAL
              TOTKEN=TOTKEN+AMASS(I)*(XNEW(I)**2+YNEW(I)**2+ZNEW(I)**2)

#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           ENDIF
#endif
        ENDDO
#if KEY_DOMDEC==1
     endif
#endif

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('Compute RMS (dynamc)')
#endif

#if KEY_BLOCK==1
     if (qmld) then
        call msld_add_kineticenergy(nblock,tempi)  /*ldm*/
     endif
#endif
     ! . Calculate the RMS gradient and work.
     EWRK = ZERO
     RVAL = ZERO
#if KEY_DOMDEC==1
     if (q_domdec) then

!$omp parallel do schedule(static) private(ia, i) reduction(+:ewrk,rval)
        do ia=1,natoml
           i = atoml(ia)
           EWRK = EWRK + DX(I)*VX(I) + DY(I)*VY(I) + DZ(I)*VZ(I)
           RVAL = RVAL + DX(I)*DX(I) + DY(I)*DY(I) + DZ(I)*DZ(I)
        ENDDO
!$omp end parallel do
     else
#endif
        do i=atfrst,atlast
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           IF (ATMASK(I)) THEN
#endif
              EWRK = EWRK + DX(I)*VX(I) + DY(I)*VY(I) + DZ(I)*VZ(I)
              RVAL = RVAL + DX(I)*DX(I) + DY(I)*DY(I) + DZ(I)*DZ(I)
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
           ENDIF
#endif
        ENDDO
#if KEY_DOMDEC==1
     endif
#endif

     EWRK=EWRK*DELTA

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('Compute bunch of stuff (dynamc)')
#endif

     ! Calculate the pressures.
     CALL PRSEXT
#if KEY_DOMDEC==1
     if (.not.q_domdec) then
#endif
        CALL PRSINT(NATOM,AMASS,XOLD,YOLD,ZOLD,DELTA, &
             XNEW,YNEW,ZNEW,DELTA)
#if KEY_DOMDEC==1
     endif
#endif

     call timer_stop(T_tvir)

#if KEY_PARALLEL==1 /*parallel*/
     call timer_start(T_commvar)
#if KEY_DOMDEC==1 /*domdec*/
     if (q_domdec) then
        ! NOTE: ipt points to the next available position
        ipt = 1

        ! If called, virshk filled gcarr(1:9)
        if (qholo .and. q_domdec) then
           ipt = ipt + 9
        endif

#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_start('prsint')
#endif
        ! PRSINT: gcarr(ipt+1:ipt+6) = vpart(1:6)
        CALL PRSINT(NATOM,AMASS,XOLD,YOLD,ZOLD,DELTA, &
             XNEW,YNEW,ZNEW,DELTA,gcarr(ipt:ipt+5),.false.)
        ipt = ipt + 6
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_stop()
#endif

        gcarr(ipt)=totken
        ipt = ipt + 1
        gcarr(ipt)=tempi
        ipt = ipt + 1
        gcarr(ipt)=ewrk
        ipt = ipt + 1
        gcarr(ipt)=rval
        ipt = ipt + 1
        gcarr(ipt)=reuris
        ipt = ipt + 1

        if (q_bonded) then
           gcarr(ipt) = real(nbondtbl)
           ipt = ipt + 1
           gcarr(ipt) = real(nangletbl)
           ipt = ipt + 1
           gcarr(ipt) = real(ndihetbl)
           ipt = ipt + 1
           gcarr(ipt) = real(nimdihetbl)
           ipt = ipt + 1
           gcarr(ipt) = real(nin14tbl)
           ipt = ipt + 1
           gcarr(ipt) = real(nex14tbl)
#if KEY_CMAP==1
           ipt = ipt + 1
           gcarr(ipt) = real(ncmaptbl)
#endif
           ipt = ipt + 1
           gcarr(ipt) = real(nex14tholetbl)
           ipt = ipt + 1
           gcarr(ipt) = real(nhypertbl)
           ipt = ipt + 1
        endif

#if KEY_LONEPAIR==1
        if (q_lonepr) then
           gcarr(ipt) = real(nloneprlist)
           ipt = ipt + 1
        endif
#endif

        if (q_aniso) then
           gcarr(ipt) = real(nanisolist)
           ipt = ipt + 1
        endif

        ! Done, decrease ipt by one to get the total count
        ipt = ipt - 1

        if (ipt > size(gcarr)) then
           call wrndie(-5,'<dynamc>','gcarr array size exceeded')
        endif

#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_start('gcomb')
#endif
        CALL GCOMB(GCARR,IPT)
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_stop()
#endif

        ipt = 1

        if (qholo .and. q_domdec) then
#if KEY_DOMDEC_GPU==1
           if (q_gpu) call range_start('virshk')
#endif
           CALL VIRSHK(EPRESS(VIXX:VIZZ),NATOM,XCOMP,YCOMP,ZCOMP,VX,VY,VZ,gcarr,.false.)
#if KEY_DOMDEC_GPU==1
           if (q_gpu) call range_stop()
#endif
           ipt = ipt + 9
        endif

#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_start('prsint')
#endif
        ! PRSINT: gcarr(ipt:ipt+5) = vpart(1:6)
        CALL PRSINT(NATOM,AMASS,XOLD,YOLD,ZOLD,DELTA, &
             XNEW,YNEW,ZNEW,DELTA,gcarr(ipt:ipt+5),.true.)
        ipt = ipt + 6
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_stop()
#endif

        totken=gcarr(ipt)
        ipt = ipt + 1
        tempi=gcarr(ipt)
        ipt = ipt + 1
        ewrk=gcarr(ipt)
        ipt = ipt + 1
        rval=gcarr(ipt)
        ipt = ipt + 1
        reuris=gcarr(ipt)
        ipt = ipt + 1

        if (q_bonded) then
           nbond_sum = int(gcarr(ipt))
           ipt = ipt + 1
           ntheta_sum = int(gcarr(ipt))
           ipt = ipt + 1
           nphi_sum = int(gcarr(ipt))
           ipt = ipt + 1
           nimphi_sum = int(gcarr(ipt))
           ipt = ipt + 1
           nin14_sum = int(gcarr(ipt))
           ipt = ipt + 1
           nex14_sum = int(gcarr(ipt))
#if KEY_CMAP==1
           ipt = ipt + 1
           ncrterm_sum = int(gcarr(ipt))
#endif
           ipt = ipt + 1
           nex14thole_sum = int(gcarr(ipt))
           ipt = ipt + 1
           nhyper_sum = int(gcarr(ipt))
           ipt = ipt + 1
        endif

#if KEY_LONEPAIR==1
        if (q_lonepr) then
           nlonepr_sum = int(gcarr(ipt))
           ipt = ipt + 1
        endif
#endif

        if (q_aniso) then
           naniso_sum = int(gcarr(ipt))
           ipt = ipt + 1
        endif

        if (q_bonded) then
           call check_bonded_14_totals(nbond_sum, ntheta_sum, nphi_sum, nimphi_sum, &
                nin14_sum, nex14_sum, &
#if KEY_CMAP==1
                ncrterm_sum, &
#endif
                nex14thole_sum, nhyper_sum)
        endif

#if KEY_LONEPAIR==1
        if (q_lonepr) then
           call check_lonepr_total(nlonepr_sum)
        endif
#endif

        if (q_aniso) then
           call check_aniso_total(naniso_sum)
        endif

     else
#endif /* (domdec)*/
        TMERI(TIMDCNTRL)=TMERI(TIMDCNTRL)+ECLOCK()-TIMMER
        TIMMER=ECLOCK()
        GCARR(1)=TOTKEN
        GCARR(2)=TEMPI
        GCARR(3)=EWRK
        GCARR(4)=RVAL
        GCARR(5)=reuris
        CALL GCOMB(GCARR,5)
        TOTKEN=GCARR(1)
        TEMPI=GCARR(2)
        EWRK=GCARR(3)
        RVAL=GCARR(4)
        reuris=GCARR(5)
        TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
        TIMMER=ECLOCK()
#if KEY_DOMDEC==1
     endif
#endif
     call timer_stop(T_commvar)
#endif /* (parallel)*/

     EPROP(GRMS)=ZERO
     IF(RVAL > ZERO) EPROP(GRMS) = SQRT(RVAL/NDEGF)

     !C      EPROP(WRKE) = EWRK

#if KEY_CHEQ==1
     EPROP(CGKE)=HALF*KECG
#endif

     TOTKEN = HALF*TOTKEN/DELTA2
     EPROP(HFCKE) = HALF * (TOTKEN + EPROP(KEPR))
     EPROP(TOTKE)= TEMPI*HALF
     ! compute total energy with correct high frequency correction
     EPROP(EHFC) = (EPROP(KEPR) - EPROP(TOTKE) + EPROP(PEPR) - &
          EPROP(EPOT))/THREE + (EPROP(KEPR2)- TOTKEN)/TWELVE

     !      EPROP(GRMS)  = (EPROP(KEPR2) + 4*EPROP(PEPR) +6*EPROP(KEPR)+
     !     &                4*EPROP(EPOT) + TOTKEN )/8.0
     !
     EPROP(HFCTE) = EPROP(EPOT) + EPROP(TOTKE) + EPROP(EHFC)
     EPROP(TOTE)  = EPROP(EPOT) + EPROP(TOTKE)
     EPROP(KEPR2) = EPROP(KEPR)
     EPROP(KEPR)  = TOTKEN
#if KEY_FLUCQ==1
     ! Calculate charge velocities, and the FlucQ contribution to energy
     IF (QFLUC) CALL FQCNEW(ISTEP,NPRINT,ATFRST,ATLAST)
#endif
#if KEY_PHMD==1
     IF (QPHMD) THEN
        CALL UpdatePHMD(0, 1, IStep, 0)
     ENDIF
#endif

     !     CHECK THE CHANGE IN TOTAL ENERGY TO BE SURE EVERYTHING IS OK.

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('Check total energy')
#endif
     chk_tote: if (jhstrt > 2.and.echeck.gt.0.0) then
        if (abs(eprop(tepr)-eprop(hfcte)) >  &
             max(echeck,ptone*eprop(hfcke))) then
#if KEY_ENSEMBLE==1
           if (jrswap.and.(mod(istep,ensfrq) < 10)) then
              write (outu,'("mod(istep,ensfrq)=",i6)') mod(istep,ensfrq)
              if(wrnlev >= 2) write(outu,2000) &
                   echeck,eprop(tepr),eprop(hfcte),eprop(hfcke)
              call flush(outu)
           else
#endif
              IF(WRNLEV >= 2) then
#if KEY_STRINGM==1 /*  VO stringm */
                WRITE(OUTU,'(A,I9)') ' GLOBAL RANK : ',ME_GLOBAL
                WRITE(OUTU,'(A,I9)') ' LOCAL RANK  : ',ME_LOCAL
                if (ftsm_on.or.smcv_on) WRITE(OUTU,'(A,I9)') ' STRING RANK : ',ME_STRNG
#endif /* VO */
              WRITE(OUTU,2000) &
                   ECHECK,EPROP(TEPR),EPROP(HFCTE),EPROP(HFCKE)
2000          FORMAT(' TOTAL ENERGY CHANGE EXCEEDED'/G12.2, &
                   ' KCAL AND 10% OF THE TOTAL KINETIC ENERGY IN THE LAST STEP'/ &
                   ' PREVIOUS E =',G14.4,' CURRENT E =',G14.4,' KINETIC =',G14.4)
              ENDIF
              ! LNI changed what is written in c28a1->a2 November 1999.
              ! Write out restart file with bad coordinates in it. Use the previous
              !  position instead of displacement.  Then get out!
              do ia=atfrst,atlast
#if KEY_DOMDEC==1
                 if (q_domdec) then
                    i = atoml(ia)
                 else
#endif
                    i = ia
#if KEY_DOMDEC==1
                 endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
                 IF (ATMASK(I)) THEN
#endif
                    ! X,Y,Zold become previous position
                    XOLD(I)=XCOMP(I)-XOLD(I)
                    YOLD(I)=YCOMP(I)-YOLD(I)
                    ZOLD(I)=ZCOMP(I)-ZOLD(I)
                    ! X,Y,Z become current position
                    X(I)=XCOMP(I)
                    Y(I)=YCOMP(I)
                    Z(I)=ZCOMP(I)
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
                 ENDIF
#endif
              ENDDO
#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
              if (.not.q_domdec) then
#endif
                 TMERI(TIMDCNTRL)=TMERI(TIMDCNTRL)+ECLOCK()-TIMMER
                 !               CALL PSYNC()     !APH: PSYNC commented out for performance
                 TIMMER=ECLOCK()
#if KEY_DOMDEC==1
              endif
#endif
#if KEY_DOMDEC==1 /*domdec*/
              if (q_domdec) then
                 !               call copy_to_root(xold, yold, zold)
                 !               call copy_to_root(vx, vy, vz)
                 !               call copy_to_root(x, y, z)
                 call copy_to_all(xold, yold, zold)
                 call copy_to_all(vx, vy, vz)
                 call copy_to_all(x, y, z)
              else
#endif /* (domdec)*/
#if KEY_SPACDEC==1
                 CALL SPACBR(XOLD,NATOM,ICPUMAP)
                 CALL SPACBR(YOLD,NATOM,ICPUMAP)
                 CALL SPACBR(ZOLD,NATOM,ICPUMAP)
                 CALL SPACBR(VX,  NATOM,ICPUMAP)
                 CALL SPACBR(VY,  NATOM,ICPUMAP)
                 CALL SPACBR(VZ,  NATOM,ICPUMAP)
                 CALL SPACBR(X,   NATOM,ICPUMAP)
                 CALL SPACBR(Y,   NATOM,ICPUMAP)
                 CALL SPACBR(Z,   NATOM,ICPUMAP)
#endif
                ! In the case of the original MPI parallelism
                ! do not gather the positions or velocities
                ! or make any more MPI calls as this will
                ! result in a deadlock
                !
                ! This is where vanilla PARALLEL stuff should
                ! be done, but DO NOT put a blocking MPI call
                ! here
#if KEY_DOMDEC==1
              endif
#endif
#if KEY_DOMDEC==1
              if (.not.q_domdec) then
#endif
                 TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
                 TIMMER=ECLOCK()
#if KEY_DOMDEC==1
              endif
#endif
#endif
              IF(IUNWRI > 0 &
#if KEY_STRINGM==0 /*  VO stringm */
     &                      .AND. IOLEV.GT.0 &
#endif
     &                      ) THEN
#if KEY_BLOCK==1 /*ldm_reset*/
                 if(qldm .or. qlmc) call ldm_reset2_dynamc(nblock)
#endif /* (ldm_reset)*/
#if KEY_DOMDEC==1
                 !                  call copy_to_root(x, y, z)
#endif
#if KEY_DOMDEC==1
                 if (q_domdec) then
                    call copy_to_all(x, y, z)
                 endif
#endif
                 CALL WRIDYN(IUNWRI,NATOM,X,Y,Z,XOLD,YOLD,ZOLD,VX,VY,VZ, &
#if KEY_CHEQ==1
                      (/ ZERO /), (/ ZERO /), (/ ZERO /), .FALSE., &
#endif
                                ! PJ 06/2005
#if KEY_PIPF==1
                      uind0, uind0, uind0,.FALSE.,0,(/ ZERO /),(/ ZERO /),  &
#endif
#if KEY_DYNVV2==1
                      .FALSE., (/ ZERO /), (/ ZERO /), (/ ZERO /), &
#endif
                      NPRIV,JHSTRT,NDEGF,NSTEP, &
                      NSAVC,NSAVV,ZERO,ZERO,0,-1 &
#if KEY_BLOCK==1
                      ,QLMC ,QLDM,NBLOCK,BIXLAM,BLDOLD,BiVLAM,NSAVL &     /*ldm*/
#endif
#if KEY_FOURD==1
                      , (/ ZERO /), (/ ZERO /) &
#endif
#if KEY_SCCDFTB==1
                      ,qlamda,qpkac,icntdyn,iavti,dvdl,dvdlav,dtmp1 &
#endif
                      )
              ENDIF

              IF(PRNLEV >= 2)THEN
                 WRITE(OUTU,'(/,3(/10X,A)//)' )  &
                      'Writing RESTART FILE with previous and current coordinates,', &
                      'which may be read by: READ COOR DYNR .... (see io.doc).', &
                      'NOTE!! THIS FILE  C A N N O T  BE USED TO RESTART A RUN!!!'
              ENDIF
              call timer_stop(t_dynamc)
#if KEY_STRINGM==1 /*  VO synchronize locally to make sure restart has been written properly */
              call mpi_barrier(MPI_COMM_LOCAL, i)
#endif
              CALL WRNDIE(-2,'<DYNAMC>', &
                   'ENERGY CHANGE TOLERANCE EXCEEDED')
#if KEY_TMD==1
              if(.not.qtmd) then
#endif
                 runok=.false.
#if KEY_DOMDEC==1
                 call dealloc_xtmp()
#endif
                 return
#if KEY_TMD==1
              endif
#endif
              call timer_start(t_dynamc)

#if KEY_ENSEMBLE==1
           endif
#endif
        endif
     endif chk_tote

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

     eprop(tepr) = eprop(hfcte)
     eprop(pepr) = eprop(epot)

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('Rest')
#endif

     time   = timfac*delta*npriv
     ! Possibly time-dependent PULLing forces need to know the time
     ptime  = time
#if KEY_RMD==1
     ctime=time
#endif

     eprop(temps) = tempi / (kboltz*ndegf)

#if KEY_BLOCK==1 /*notqmld*/ /*ldm*/
     if (.not.qmld) call ldm_write_dynamc(nblock, ndegf, istep, tempi, eprop(temps))
#endif /* (notqmld)  LDM*/
     jhtemp = jhtemp + eprop(temps)
     call avfl_update(eprop, eterm, epress)
     ! Unit cell and gradient terms
     if (qcnstp) call avfl_ucell_update()
#if KEY_QUANTUM==1
     if(qmpert) then
        do i = qpt1,qpt2
           eeqtrm(i) = exp((eterm(qmel)-eeqtrm(i))*beta_qmmm)
           eqpra(i) = eqpra(i)+eeqtrm(i)
           eqpr2a(i) = eqpr2a(i)+eeqtrm(i)*eeqtrm(i)
        enddo
     endif
     if(qdecom) then
        !  qver - qgas are sequentially defined, quantm.f90, and enerin
        do i = qver,qgas
           eqpra(i) = eqpra(i)+eeqtrm(i)
           eqpr2a(i) = eqpr2a(i)+eeqtrm(i)*eeqtrm(i)
        enddo
     endif
     ! jg 5/2002 average charges during qm/mm md
     if (chdyn) then
        call dynden('accu',istep)
     endif
#endif
#if KEY_SQUANTM==1
     if(qmfep .or. qmsolv) then
        do i = qpt1,qpt2
           eeqtrm(i) = exp((eterm(qmel)-eeqtrm(i))*beta_qmmm_fep)
           eqpra(i)  = eqpra(i)+eeqtrm(i)
           eqpr2a(i) = eqpr2a(i)+eeqtrm(i)*eeqtrm(i)
        end do
     end if
#endif

     istpsa=mod(ist1,iprfrq)+istep-ist1
     fita = fita + istpsa * eprop(hfcte)
     fitp = fitp + jhstrt * eprop(hfcte)

     if(nsavv > 0)then
        if(mod(istep,nsavv) == 0) then
#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
           if (.not.q_domdec) then
#endif
              tmeri(timdcntrl)=tmeri(timdcntrl)+eclock()-timmer
              !            CALL PSYNC()     !APH: PSYNC commented out for performance
              timmer=eclock()
#if KEY_DOMDEC==1
           endif
#endif
#if KEY_DOMDEC==1 /*domdec*/
           if (q_domdec) then
              !            call copy_to_root(vx, vy, vz)
              call copy_to_all(vx, vy, vz)
           else
#endif /* (domdec)*/
#if KEY_SPACDEC==1
              CALL SPACBR(VX,NATOM,ICPUMAP)
              CALL SPACBR(VY,NATOM,ICPUMAP)
              CALL SPACBR(VZ,NATOM,ICPUMAP)
#else /**/
              CALL VDGBR(VX,VY,VZ,1)
#endif
#if KEY_DOMDEC==1
           endif
#endif
#if KEY_DOMDEC==1
           if (.not.q_domdec) then
#endif
              TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
              TIMMER=ECLOCK()
#if KEY_DOMDEC==1
           endif
#endif
#endif
           !C MH05 WRITCV has IOLEV inside          IF(IOLEV > 0) THEN
           CALL WRITCV(VX,VY,VZ, &
#if KEY_CHEQ==1
                VCG,QCG,                               &
#endif
                NATOM,FREEAT,NFREAT,NPRIV,ISTEP,NDEGF, &
                DELTA,NSAVV,NSTEP,TITLEA,NTITLA,IUNVEL,.TRUE., &
                .FALSE., (/ 0 /), .FALSE., (/ ZERO /))
           !C MH05          ENDIF

        ENDIF
     ENDIF

#if KEY_LIBRARY == 1
     if (fill_velos .and. (istep .eq. istop)) then
        call velos_init()
        call velos_set_names()
        call velos_add_rows(istep, time, vx, vy, vz)
     end if
#endif

     !     mdstep is a global variable in common/contrl/
     MDSTEP=ISTEP

     !     Save ab initio charges in the same format as coordinates
     !     QVEL=.FALSE. (maybe change to .TRUE. ???)
     !     /For frequencies we need velocities and charges/
     !     This code doesn't deal with non-free QM atoms, ie
     !     charges will be fixed too, although they shouldn't be.
#if KEY_GAMESS==1
     if (NSAVQ > 0) then
        if (MOD(ISTEP, NSAVQ) == 0) then
           CALL WRITCV(QMCMUL,QMCLOW,QMCKOL, &
#if KEY_CHEQ==1
                VCG,QCG, &
#endif
                NATOM,FREEAT,NFREAT,NPRIV,ISTEP,NDEGF, &
                DELTA,NSAVQ,NSTEP,TITLEA,NTITLA,IUNQMC,.FALSE., &
                .FALSE., (/ 0 /), .FALSE., (/ ZERO /))
        end if
     end if
#endif /* KEY_GAMESS */

     !     At this point we can write X,VX, and DX if needed:
     if (NSAVX > 0) then
        if (MOD(ISTEP, NSAVX) == 0) then
          CALL WRXYZ(IUNXYZ,X,Y,Z,VX,VY,VZ,DX,DY,DZ,ISTEP)
       end if
    end if

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
#if KEY_TPS==1
     !     M. Hagan and A. R. Dinner, May 2003
     IF (QTPS) THEN
        !       Check whether or not the path is in Basin B (for H sampling).
        !       The order parameter values were calculated during the energy call.
        CALL TPHUPD()
        !       Store the coordinates (which are for the next step).
        !       Subtract PDIR since it is flipped.
        ITPSST = ITPSST - PDIR
        CALL TPSUPD(NATOM,XCOMP,YCOMP,ZCOMP, &
             VX,VY,VZ)
     ELSE IF (INBCUT  >  0) THEN
        CALL TPCMMT()
        IF (ABS(IBASIN)  >=  INBCUT) THEN
#if KEY_DOMDEC==1
           call dealloc_xtmp()
#endif
           call timer_stop(T_dynamc)
           RETURN
        ENDIF
     ENDIF
#endif
#if KEY_SGLD==1 /*sgldprint*/
     IF (NSAVC > 0 .AND. PRNLEV > 0) THEN
        IF ((MOD(ISTEP, NSAVC) == 0) .AND. (QSGLD .OR. QSGMD)) THEN
           CALL PRINTE(OUTU, EPROP, ETERM, 'TRAJ', 'DYN', .FALSE., &
                ISTEP, TIME, ZERO, .TRUE.)
           CALL PRNTSG(OUTU,.FALSE.,'TRAJ',.FALSE.)
        END IF
     END IF
#endif /* (sgldprint)*/

#if KEY_MC==1
     IF (NPRINT  >  0) THEN
#endif
        IF(MOD(ISTEP,NPRINT) == 0) THEN
           IF(PRNLEV > 0) THEN
              CALL PRINTE(OUTU, EPROP, ETERM, 'DYNA', 'DYN', .FALSE., &
                   ISTEP, TIME, ZERO, .TRUE.)

#if KEY_SGLD==1 /*sgldprint*/
              IF((QSGLD.OR.QSGMD).AND.PRNLEV>=2)CALL PRNTSG(OUTU,.FALSE.,'DYNA',.FALSE.)
#endif /* (sgldprint)*/
              !adm jr.
              IF (QCNSTP) THEN
                 RVAL=DOTVEC(DXTL,DXTL,XDIM)
                 GNORM = ZERO
                 IF(RVAL > ZERO) GNORM = SQRT(RVAL/XDIM)
                 CALL PRNXTLD(OUTU,'DYNA',XTLTYP,XUCELL,.TRUE.,GNORM, &
                      .TRUE.,EPRESS)
              ENDIF
           ENDIF

           CALL WRETERM(NPRIV,TIME,QKUHEAD)
        ENDIF
#if KEY_MC==1
     ENDIF
#endif

#if KEY_LIBRARY == 1
     if (fill_ktable .and. (istep .eq. istop)) then
        call ktable_init()
        call ktable_set_names()
        call ktable_add_row(npriv, time)
     end if
#endif

#if KEY_TSM==1
     ! This section is for conformational thermodynamic integration
     ! It is placed here on purpose - the routines use the forces DX,DY,DZ
     ! calculated at the X,Y,Z coordinates.   K. Kuczera
     ! DYNICT -- standard TSM + one-dimensional TI added on (KK)
     ! DYNICM -- multi dimensional conformational TI, no TP (KK)

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

#if KEY_TSM==1
     IF(QTSM.AND.PIGSET) CALL PIGCVSET(XNEW,YNEW,ZNEW)
#endif

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

     !         write(*,*)'DYNAMC-27>'
#if KEY_SPACDEC==1
     a0=1
     a1=nmyatm
#else /**/
     a0=atfrst
     a1=atlast
#endif

#if KEY_DOMDEC==1
     if (q_domdec) then
!$omp parallel do schedule(static) private(ia, i, fact)
        do ia=a0,a1
           i = atoml(ia)
           FACT=XOLD(I)
           XOLD(I)=XNEW(I)
           XNEW(I)=FACT
           FACT=YOLD(I)
           YOLD(I)=YNEW(I)
           YNEW(I)=FACT
           FACT=ZOLD(I)
           ZOLD(I)=ZNEW(I)
           ZNEW(I)=FACT
        enddo
!$omp end parallel do
     else
#endif
        do ia=a0,a1
#if KEY_SPACDEC==1
           I=MYSATARR(ia)
#else /**/
           i = ia
#endif
#if KEY_PARASCAL==1
           IF(JPBLOCK(I) == MYNOD) THEN
#endif
              FACT=XOLD(I)
              XOLD(I)=XNEW(I)
              XNEW(I)=FACT
              FACT=YOLD(I)
              YOLD(I)=YNEW(I)
              YNEW(I)=FACT
              FACT=ZOLD(I)
              ZOLD(I)=ZNEW(I)
              ZNEW(I)=FACT
#if KEY_PARASCAL==1
           ENDIF
#endif
        enddo
#if KEY_DOMDEC==1
     endif
#endif
!!$     do ia=a0,a1
!!$##IF DOMDEC
!!$        if (q_domdec) then
!!$           i = atoml(ia)
!!$        else
!!$##ENDIF
!!$##IF SPACDEC
!!$        I=MYSATARR(ia)
!!$##ELSE
!!$        i = ia
!!$##ENDIF
#if KEY_DOMDEC==1
!!$        endif
#endif
#if KEY_PARASCAL==1
!!$        IF(JPBLOCK(I) == MYNOD) THEN
#endif
!!$           FACT=XOLD(I)
!!$           XOLD(I)=XNEW(I)
!!$           XNEW(I)=FACT
!!$           FACT=YOLD(I)
!!$           YOLD(I)=YNEW(I)
!!$           YNEW(I)=FACT
!!$           FACT=ZOLD(I)
!!$           ZOLD(I)=ZNEW(I)
!!$           ZNEW(I)=FACT
#if KEY_PARASCAL==1
!!$        ENDIF
#endif
!!$     enddo

#if KEY_BLOCK==1
     ! was BANBA
     IF (QPRNTVL)  THEN
        IF(MOD(NPRIV,IBLCKFEP) == 0) THEN

           CALL SUM_POTENTIAL(V_TSM)

           IF(IOLEV >= 0) WRITE(NBLCKFEP,4550) &
                NPRIV,AKMATI,1.0-BLCOEe(2),EPROP(EPOT),V_TSM(1),V_TSM(2), &
                V_TSM(3),V_TSM(4),V_TSM(5),V_TSM(6),V_TSM(7),V_TSM(8), &
                EPROP(TOTE),EPROP(TOTKE)
           CALL GFLUSH(NBLCKFEP)
        ENDIF
     ENDIF
4550 FORMAT(I12,2(1X,1PG24.16E2),/,3(2(1PG24.16E2,1X), &
          1PG24.16E2,/),2(1PG24.16E2,1X),1PG24.16E2)
     !.ab.Print out HybridH. Use the #BLOCK#
     IF(IOLEV >= 0) CALL PRHYBH()
     !.ab.
#endif /*  BLOCK*/

#if KEY_TSM==1 /*tsm*/
     IF(QTSM) THEN
        IF(SAVEP) THEN
           IF(MOD(NPRIV,PERFRQ) == 0) THEN
              IF (NPUMB /= 0) THEN
                 CALL PUMEPHI(PUMEP,XCOMP,YCOMP,ZCOMP)
                 IF(IOLEV >= 0) WRITE(PUNITX,3550) &
                      NPRIV,AKMATI,LAMBDA,EPROP(EPOT),VPRTTR,VPRTTP,VPRTNR, &
                      VPRTNP,VPRTVR,VPRTVP,VPRTER,VPRTEP,EPROP(TOTE), &
                      EPROP(TOTKE),PUMEP
3550             FORMAT(I12,2(1X,1PG24.16E2),/,3(2(1PG24.16E2,1X), &
                      1PG24.16E2,/),2(1PG24.16E2,1X),1PG24.16E2)
              ELSE
                 IF(IOLEV >= 0) WRITE(PUNITX,3575) &
                      NPRIV,AKMATI,LAMBDA,EPROP(EPOT),VPRTTR,VPRTTP,VPRTNR, &
                      VPRTNP,VPRTVR,VPRTVP,VPRTER,VPRTEP,EPROP(TOTE), &
                      EPROP(TOTKE)
3575             FORMAT(I12,2(1X,1PG24.16E2),/,3(2(1PG24.16E2,1X), &
                      1PG24.16E2,/),1PG24.16E2,1X,1PG24.16E2)
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
        IF(QTSM.AND.PIGSET) CALL PIGMSET(AMASS,.FALSE.)
     ENDIF

     ! Calculate the ic perturbation interaction energies ..normal TSM
     IF(QICP .AND. .NOT.(QCFTI.OR.QCFTM)) THEN
        ! The next line is changed for parallel!!!!
#if KEY_PARALLEL==1 /*parallel*/
#if KEY_SPACDEC==1 /*spacdec*/
        CALL SPACBR(XCOMP,NATOM,ICPUMAP)
        CALL SPACBR(YCOMP,NATOM,ICPUMAP)
        CALL SPACBR(ZCOMP,NATOM,ICPUMAP)
#else /* (spacdec)*/
#if KEY_DOMDEC==1
        if (q_domdec) then
           call copy_to_all(xcomp, ycomp, zcomp)
        else
#endif
           CALL VDGBR(XCOMP,YCOMP,ZCOMP,1)
#if KEY_DOMDEC==1
        endif
#endif
#endif /* (spacdec)*/
#endif /* (parallel)*/
        CALL DYNICP(XCOMP,YCOMP,ZCOMP,NATOM,NPRIV,AKMATI, &
             EPROP(TOTE),EPROP(TOTKE), &
             IHPCFTI(1)%a,IHPCFTI(2)%a,IHPCFTI(3)%a)
     ENDIF
#endif /* (tsm)*/
#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
     if (.not.q_domdec) then
#endif
        TMERI(TIMDCNTRL)=TMERI(TIMDCNTRL)+ECLOCK()-TIMMER
        TIMMER=ECLOCK()
#if KEY_DOMDEC==1
     endif
#endif
#endif
#if KEY_TMD==1
     !     check if the requested final rmsd has been reached: if so, quit
     if(qtmd.and.(.not.qzeta).and.(tmdrhof <= tmdfrms))then
        istop=istep
        IF(PRNLEV > 0) THEN
           CALL PRINTE(OUTU, EPROP, ETERM, 'DYNA', 'DYN', .FALSE., &
                ISTEP, TIME, ZERO, .TRUE.)
        ENDIF
        write(outu,'(/" THE REQUESTED RMSD OF ",f12.5, &
             " HAS BEEN REACHED AT STEP ",I8)') &
             tmdfrms,istep
        exit mainloop
     endif
#endif

#if KEY_ENSEMBLE==1
     if (jrswap) then
        if (mod(istep,ensfrq) == 0) then
           call ensswl(xnew,ynew,znew,xold,yold,zold,xcomp,ycomp,zcomp,natom)
           call ensout
           if (ensas) then
              ! assign velocities after ensemble swap try
              ! (regardless if the swap succeeded or not)
              if (mynod == 0) then
                 do i=1,natom
                    if(imove(i) == 0) then
                       call gaussi(zero,sqrt(ensmyt*kboltz/amass(i)),vx(i),iseed,1)
                       call gaussi(zero,sqrt(ensmyt*kboltz/amass(i)),vy(i),iseed,1)
                       call gaussi(zero,sqrt(ensmyt*kboltz/amass(i)),vz(i),iseed,1)
                    else
                       vx(i)=zero
                       vy(i)=zero
                       vz(i)=zero
                    endif
                 enddo
              endif
              call psnd8(vx,natom)
              call psnd8(vy,natom)
              call psnd8(vz,natom)
              do i=1,natom
                 fact=one/gamma(i+natom3)
                 xnew(i)=vx(i)*fact+xold(i)
                 ynew(i)=vy(i)*fact+yold(i)
                 znew(i)=vz(i)*fact+zold(i)
              enddo
           endif
        endif
     endif
#endif
#if KEY_DIMS==1
     IF(QDIMS) THEN
        !     Compute Onsager-Machlup Score for current move
        DO I=ATFRST,ATLAST
           FACT=AMASS(I)
           OMSC=OMSC+ &
                ( (FACT*VX(I)-DX(I))**2 &
                + (FACT*VY(I)-DY(I))**2 &
                + (FACT*VZ(I)-DZ(I))**2)  &
                * 0.5
        ENDDO
        IF(MOD(ISTEP,NPRINT) == 0) &
             WRITE(OUTU,'(A9,E12.4)') 'OMSCORE> ',OMSC
        call set_param('OMSCORE',OMSC)
     ENDIF
#endif
     if(ncspuck > 0 ) call pucker_constraint_output(istep)

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

  ENDDO mainloop


! THIS IS THE END OF THE MAIN LOOP ! DO ISTEP=ISTART,ISTOP
!=======================================================================

#if KEY_SQUANTM==1
!  if(LTRBOMD) q_apply_tr_bomd=.false.   ! for TR-BOMD, do not apply TR-BOMD.
#endif

!!#if KEY_MNDO97==1
!!  qm_control_r%md_run =.false.        ! end of MD run.
!!#endif

  do ia=atfrst,atlast
#if KEY_DOMDEC==1
     if (q_domdec) then
        i = atoml(ia)
     else
#endif
        i = ia
#if KEY_DOMDEC==1
     endif
#endif
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
     IF (ATMASK(I)) THEN
#endif
        X(I)=XCOMP(I)
        Y(I)=YCOMP(I)
        Z(I)=ZCOMP(I)
#if KEY_SPACDEC==1 || KEY_PARASCAL==1
     ENDIF
#endif
  ENDDO

#if KEY_BLOCK==1 /*ldm*/
  !     reset lambdas to time t and bldold to t + dt before writing
  !     restart file
  if(qldm .or. qlmc) call ldm_reset3_dynamc(nblock)
#endif /*  LDM*/

  RUNOK=.TRUE.
  IF (ISTPSA < IPRFRQ) then
#if KEY_DOMDEC==1
     call dealloc_xtmp()
#endif
     call timer_stop(T_dynamc)
     RETURN
  endif

  !     Now do long time fluctuations

  call avfl_update_lt()
  ! Unit cell and gradient long terms
  if (QCNSTP) call avfl_ucell_update_lt()
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

  NUMSTP = ISTPSA
  DNUM   = NUMSTP
  DNM1   = DNUM - ONE
  DNP1   = DNUM + ONE
  DNPM1  = DNM1 * DNP1
  IF(NUMSTP > 1) THEN
     DRIFTA = (TWO*FITA-DNP1*EPRPA(HFCTE))*SIX/(DNPM1*DNUM)
     EAT0A  = EPRPA(HFCTE)/DNUM-DRIFTA*DNP1/TWO
     RVAL = EPRP2A(HFCTE)/DNUM - (EPRPA(HFCTE)/DNUM)**2
     IF(RVAL > ZERO) THEN
        CORRA=DNPM1/RVAL/TWELVE
        CORRA=DRIFTA*SQRT(CORRA)
     ELSE
        CORRA=ZERO
     ENDIF

     ! . Compute statistics.
     call avfl_compute(DNUM)
     ! Calculate average unit cell and gradient terms
     if (QCNSTP) call avfl_ucell_compute(DNUM)
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
           EQPRA(I) = EQPRA(I) / DNUM
           FLUCTD   = EQPR2A(I)/DNUM - EQPRA(I)**2
           EQPR2A(I)= ZERO
           IF(FLUCTD > ZERO) EQPR2A(I) = SQRT(FLUCTD)
           IF(EQPRA(I) > ZERO) &
                EQPRA(I) = -LOG(EQPRA(I))/beta_qmmm_fep
           IF(EQPR2A(I) > ZERO) &
                EQPR2A(I) = -LOG(EQPR2A(I))/beta_qmmm_fep
        END DO
     END IF
#endif

  ENDIF

#if KEY_SGLD==1 /* sgld store params */
  if (qsgld .or. qsgmd) call sgld_ave_params()
#endif /* sgld store params */

  ! . Print out the results.
  IF(PRNLEV >= 2) THEN
     call avfl_print_aver(NUMSTP, TIME, tag='DYNAMC>')
#if KEY_SGLD==1 /*sgldprint*/
        IF((QSGLD.OR.QSGMD).AND.PRNLEV>=2)CALL PRNTSG(OUTU,.FALSE.,'AVER',.TRUE.)
#endif /* (sgldprint)*/
     if (QCNSTP) call avfl_ucell_print_aver(NUMSTP)
     call avfl_print_fluc(NUMSTP, TIME, tag='DYNAMC>')
#if KEY_SGLD==1 /*sgldprint*/
        IF((QSGLD.OR.QSGMD).AND.PRNLEV>=2)CALL PRNTSG(OUTU,.FALSE.,'FLUC',.TRUE.)
#endif /* (sgldprint)*/
     if (QCNSTP) call avfl_ucell_print_fluc(NUMSTP)
#if KEY_QUANTUM==1
     ! JG 12/00
     IF(QMPERT.OR.QDECOM) THEN
        WRITE(OUTU,'(A,I8,A)') &
             ' QMPERT> QM FEP for the last ',NUMSTP,' steps:'
        CALL PRQFEP(OUTU,NUMSTP,TIME)
     ENDIF
     ! JG 5/2002
     IF (CHDYN) THEN
        CALL DYNDEN('PRIN',ISTEP)
     ENDIF
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

  DRIFTP = DRIFTA
  EAT0P  = EAT0A
  CORRP  = CORRA
  IF (JHSTRT <= NUMSTP) GOTO 190
  NUMSTP = JHSTRT
  DNUM   = NUMSTP
  DNM1   = DNUM - ONE
  DNP1   = DNUM + ONE
  DNPM1  = DNM1 * DNP1
  IF(NUMSTP > 1) THEN
     DRIFTP = (TWO*FITP - DNP1*EPRPP(HFCTE))*SIX/(DNPM1*DNUM)
     EAT0P  = EPRPP(HFCTE)/DNUM-DRIFTP*DNP1/TWO
     RVAL=EPRP2P(HFCTE)/DNUM - (EPRPP(HFCTE)/DNUM)**2
     IF(RVAL > ZERO) THEN
        CORRP=DNPM1/RVAL/TWELVE
        CORRP=DRIFTP*SQRT(CORRP)
     ELSE
        CORRP=ZERO
     ENDIF
  ENDIF

  call avfl_compute_lt(DNUM)
  ! Unit cell and gradient long terms
  if (QCNSTP) call avfl_ucell_compute_lt(DNUM)
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

! . Print out the results.
  IF(PRNLEV >= 2) THEN
     call avfl_print_aver_lt(NUMSTP, TIME, tag='DYNAMC>')
     if (QCNSTP) call avfl_ucell_print_aver_lt(NUMSTP)
     call avfl_print_fluc_lt(NUMSTP, TIME, tag='DYNAMC>')
     if (QCNSTP) call avfl_ucell_print_fluc_lt(NUMSTP)
#if KEY_QUANTUM==1
     ! JG 12/00
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

  ENDIF

190 CONTINUE
  IF(NUMSTP <= 1 .OR. JHSTRT.LE.1 .OR. PRNLEV < 3)then
#if KEY_DOMDEC==1
     call dealloc_xtmp()
#endif
     call timer_stop(T_dynamc)
     RETURN
  endif
  WRITE(OUTU,195)DRIFTA,DRIFTP,EAT0A,EAT0P,CORRA,CORRP
195 FORMAT(/5X,'DRIFT/STEP (LAST-TOTAL): ',1P,2G17.8, &
             /5X,'E AT STEP 0            : ',1P,2G17.8, &
             /5X,'CORR. COEFFICIENT      : ',1P,2G17.8)
#if KEY_DOMDEC==1
  call dealloc_xtmp()
#endif
  call timer_stop(T_dynamc)

  RETURN

contains

  subroutine dealloc_xtmp()
#if KEY_DOMDEC==1
     if (allocated(xtmp)) then
        call chmdealloc('dynamc.src','DYNAMC','xtmp',natomx,crl=xtmp)
        call chmdealloc('dynamc.src','DYNAMC','ytmp',natomx,crl=ytmp)
        call chmdealloc('dynamc.src','DYNAMC','ztmp',natomx,crl=ztmp)
     endif
#endif
  end subroutine dealloc_xtmp

  ! *
  ! * Sets the qdyncall -variable:
  ! * Set to .false. if any printing of results is happening
  ! *
  subroutine set_qdyncall()
    implicit none
    QDYNCALL=.TRUE.
#if KEY_SGLD==1 /*sgldprint*/
     if (nsavc > 0) then
        if((mod(istep, nsavc) == 0) .and. (qsgld .or. qsgmd)) &
           qdyncall = .false.
     end if
#endif /* (sgldprint)*/

#if KEY_MC==1
     IF (NPRINT  >  0) THEN
#endif
        IF(MOD(ISTEP,NPRINT) == 0) THEN
           qdyncall = .false.
        endif
#if KEY_MC==1
     ENDIF
#endif

     return
  end subroutine set_qdyncall

END subroutine dynamc
