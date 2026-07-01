!> Don't call old_ENERGY, use energym and call energy,
!> which has the same arguments and an explicit interface.
SUBROUTINE old_ENERGY(X, Y, Z, DX, DY, DZ, BNBND, BIMAG, &
     NDD1, DD1, IUPT, KUPT, QSECD, ICALL &
#if KEY_DHDGB==1
!AP/MF
     ,SDEF,DS_DHDGB &
#endif
)
  !-----------------------------------------------------------------------
  !       CALCULATES THE ENERGY AND FORCES FOR A STRUCTURE.
  !     The total energy and individual energy contributions are
  !     returned in the ENERGY.FCM common block.
  !
  !      X,Y,Z         - Coordinates
  !      DX,DY,DZ      - Forces returned
  !      BNBND,BIMAG   - Nonbond and Images data structure bases
  !      NDD1            The dimension of the second derivative matrix.
  !      DD1           - Second derivative arrays
  !      QSECD         - Second derivative flags
  !      ICALL         - ECALLS increment
  !      FDIM,DFDIM    - Coordinates and forces in 4th DIM
  !      DIM4          - Fourth dimension flag
  !
  !     By Bernard R. Brooks (and others)  1981 to 1983
  !     Jay Banks: added MMFF interfaces - October 1995
  !     SPASIBA Force Field added by P. Lagant and R. Stote (11/01)
  !     SPASIBA Force Field removed for c34b2 and c35a1 releases
  !
  !......................................................................
  !     For parallel communication calls search for: paramain and paraend
  !     pref.dat keyword labels
  !......................................................................
  !
  !-----------------------------------------------------------------------

#if KEY_ZEROM==1
  use zdata_mod,only: QZMOD
#endif
#if KEY_EPMF==1
  use epmf,only: totepmf
#endif
#if KEY_PRIMO==1
  use primomodule,only: eprimo
#endif
#if KEY_DENBIAS==1
  use denbias, only : qdenbias, denbias_vol_density
#endif
  use ewald, only: lewald,ewvirial,kappa,                  &
#if KEY_MNDO97==1
                   kspace_qmmm_prep,KSPACE_qmmm_get_force, &
#endif
                   kspace
  use genborn, only: qgenborn, fillalpgb, gbsolv
#if KEY_BLOCK==1
  use genborn, only: fillalpgb1, fillalpgb2, gbsolv1, gbsolv2
#endif

  use pme_module, only: qpme
  use gb_common,only: igentype
  use gbmv, only: rungbmv,rungbmvgrid,qgbmv

#if KEY_OPENMM==1
  use omm_gbsw, only : qphmd_omm
#endif
  use inbnd, only : qgbsw_omm,qgbmv_omm
  use gbsw, only: gb_lookup, gb_solv1,gb_surf1,qgbsw,qgvdw,      &
     gvdw,sgamma,gbsurf,ngbsel
#if KEY_LOOKUP==1
  use LOOKUP,only:elookup,qlookup,nwwo,iwwo,iwoonbl,jwoonbl,ivunbl,jvunbl,&
                  iuunbl,juunbl,iwwenr,ewweel,ewwenb
#endif
#if KEY_OVERLAP==1
  use eolapmod, only:olapener
#endif
  use new_timer,only:timer_start,timer_stop,timer_stpstrt,       &
     T_energy,T_inte,T_nonbon,T_gborn,T_bond,T_angle,          &
     T_ephi,t_restr,T_ecomm,T_ewald,T_rec,T_fcomm,             &
     T_ips,T_ipsex,T_ipsa,                                     &
#if KEY_DOMDEC==1
     T_fcomm2,                                                 &
     T_dlb,                                                    &
#endif
     T_facts

#if KEY_CHEQ==1
  use cheq, only: qcg,qnoco,gqt4p,qcginv,qcgmine,     &
     cpcgimg,qcgset,qbener,qcgtransi,gbfoncheq,qnorm,  &
     ddcg,qnocod,   &
     DCH,SUMDCH,DDCH
#endif
  use nb_module
#if KEY_NBIPS==1
  use aips_module,only: eipssys,aipsfin,aipspbc
#endif
#if KEY_MSCALE==1
  use mscalemod, only: qmscale,emscale
#endif
#if KEY_REPDSTR==1
  use repdstrmod
#endif
#if KEY_RPATH==1
  use epathmod
#endif
#if KEY_RMD==1
  use cross, only: ecross,NCRUN
#endif
#if KEY_MRMD==1
  use mrmd_fcm,only:emrmd,mrmd_active
#endif
#if KEY_MMPT==1
  use mmpt_fcm, only: emmpt
#endif
#if KEY_MSMMPT==1
  use msmmpt_fcm, only: emsmmpt
#endif
#if KEY_FLUCQ==1
  use flucqm, only: fqcfor
#endif
#if KEY_TSALLIS==1
  use tsallis_module
#endif
#if KEY_OVERLAP==1
     !C  use eolapmod, only: olapener
#endif
#if KEY_FACTS==1
  use facts_module
#endif
#if KEY_AFM==1
  use afm_module,only: afm, lafm
#endif
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
  use gukini_mod,only: gukene
#if KEY_QCHEM==1
  use gukini_mod,only: microcorr
#endif
#endif
  use dcm_fcm,only: qdcm,add_dcm_recip_f ! DCM module

#if KEY_CADPAC == 1
  use cadpac_mod, only: cadene
#endif

  use chm_kinds
  use chm_types
  use dimens_fcm
  use memory
  use number
#if KEY_ACTBOND==1
  use actclus_mod,only:QUREYBR
#endif
#if KEY_APBS==1
  use iapbs
#endif
  use block_ltm
#if KEY_SCCDFTB==1
  use blockscc_fcm
#endif
  use lambdam
  use cnst_fcm,only:ccbic,cctic,ccpic,cciic,ihset,kbexpn,kqcnst,kqexpn,lcic,lqmass,lupper,numhsets,&
#if KEY_CPATH==1
       pathn, &
#endif
       qcnstr,qhnort,qhnotr,refx,refy,refz,kcnstr,typhset,xhscale,yhscale,zhscale,kcexpn,&
       qqcnst, numpca, pcax, pcay, pcaz
!  use cstran_mod
  use code
  use conshelix_m
#if KEY_SPACDEC==1 || KEY_TSALLIS==1
  use contrl
#endif
  use ecntrl
  use econtmod
#if KEY_ASPENER==1
  use eef1_mod
#endif
  use eintern
  use eintern_fast
#if KEY_EMAP==1
  use emapmod
#endif
  use enbond_mod
  use energym
  use energy_util,only:set_mtsflags, calc_adumb_potential, calc_dmcons, write_dmcons, &
       calc_umbrella_potential, calc_dihe_restraints, calc_noe_restraints, calc_redcns, &
       calc_dbias, calc_hmcm, zero_energy_terms
  use euler
  use eutil
  use exelecm
  use fast
  use fourdm
  use gamess_fcm,only: qmused,qmicro,qfrstmicit,qwriteinp
  use gbim, only: qgbimb, FillAlpGBIM, GBSolvIM
  use grape
#if KEY_GRAPE==1
  use grapemod,only: qgpuene,ksdx,ksdy,ksdz,gpuinblo,ene_c,ene_v,vdw_im,ele_im
#endif
  use hbondm
  use holonom,only:holonomf
#if KEY_HQBM==1
  use hqbmm, only: hqbias
#endif
#if KEY_ABPO==1
  use abpo_ltm
#endif
#if KEY_ABPO==1
  use abpo, only: abpo_ener
#endif
  use image
  use image_routines_module, only: &
#if KEY_BLOCK==1 /*ldm_image*/
       transrstp, &
#endif
       transi, transo
  use eimg
  use inbnd
#if KEY_MULTCAN==1
  use mltcanon,only:genensemble,qmltcan
#endif
#if KEY_MNDO97==1
  use mndo97
#endif
  use nbips
  use param
  use pathm
#if KEY_PATHINT==1
  use mpathint, only: qpint, epint
#endif
#if KEY_PBEQ==1
  use pbeq, only: qpbeq,qpbf,npbeq,qgsbp,pbforce,gsbp0,qgas1,qsmbp,smbp0
#endif
  use pbound
  use pert
#if KEY_POLAR==1
  use polarm, only: qpolar, polar1
#endif
#if KEY_PRIMSH==1
  use primsh, only: qshel, pshel
#endif
  use psf
  use pull_mod, only: epull
  use quantm
  use replica_mod
  use rush_mod
#if KEY_RXNCOR==1
  use rxncom,only:rxnind
#endif
  use sbound
#if KEY_SCCDFTB==1
  use sccdftb
#endif
#if KEY_SCCDFTB==1
  use sccdftbsrc                /* qc_010110*/
#endif
  use shake
  use stream
  use shapes
#if KEY_SQUANTM==1
  use squantm
#endif
  use surface
  use surfmemb
  use timerm
  use tsms_mod
  use tbmts
  use usermod,only:usracm,usere
  use grid_dock
  use olap
  use nbthole
#if KEY_SSNMR==1
  use ssnmr
#endif
#if KEY_RDC==1
  use rdc, only: RDC1,QRDC
#endif
     !
#if KEY_NOMISC==0 /*nomisc*/
  use mmfp
  use ssbpm, only: qssbp,ssbp1
#endif /* (nomisc)*/
#if KEY_PARALLEL==1
  use parallel
#endif
  use ensemble
#if KEY_ENSEMBLE==1
  use evb_mod,only: qevb, evb
#endif
!  use repdstr
     !
  use ffieldm
#if KEY_MMFF==1 /*mmff*/
  use mmffm
  use escalar_mm
  use vangle_mm
#endif /* (mmff)*/
#if KEY_ADUMB==1 /*adumb*/
  use umb
  use umbcor
#endif /*  (adumb)*/
#if KEY_GAMUS==1
  use gamusmodule
#endif
#if KEY_PIPF==1
  use pipfm
#endif
     !
#if KEY_FLUCQ==1
  use flucq
#endif
     !
  use phmd
     !
  use dmcons
#if KEY_RGYCONS==1
  use rgym, only: qrgy, ergy
#endif
     !
#if KEY_ESTATS==1
  use estats_mod
#endif
#if KEY_SASAE==1
  use sasa
#endif
  use machutil,only:eclock,wrttim,die
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec,zonelist,groupl,ndirect,q_split
  use domdec_d2d_comm,only:transfer_force
  use domdec_d2r_comm,only:probe_results_from_recip, wait_results_from_recip, &
       unpack_results_from_recip
  use domdec_r2d_comm,only:comm_coord_among_recip, comm_force_among_recip, zero_recip_force, &
       recipforcex, recipforcey, recipforcez, reduce_recip_forces
  use groupxfast,only:group, group_out
  use pme_module, only: pme_self_energy
#if KEY_LJPME==1
  use pme_module, only: ljpme_self_energy
  use pmeutil, only: dkappa, qljpme
#endif
  use enbxfast,only:calc_virial
  use domdec_local,only:unpack_forces
  use ebonded_domdec,only:eureyb_domdec, thole_ex14_domdec
  use energy_util,only:init_auxdata, read_from_auxdata
  use domdec_dr_common,only:q_recip_node
#endif
#if KEY_DOMDEC_GPU==1
  use domdec_local,only:sforce, combine_gpu_forces_to_global_forces
  use domdec_common, only: gpu_code_version, q_gpu, q_split
  use domdec_util_gpu_mod,only:range_start, range_stop, reduce_force_virial_gpu, &
       copy_force_virial_energy_from_gpu, wait_force_virial_energy_from_gpu
  use enb_core_gpu_mod,only:read_nonbond_energy_gpu, read_direct_virial_gpu
  use nbrecip_gpu_mod,only:read_recip_energy_gpu, read_recip_virial_gpu
  use ebonded_domdec,only:ebonded_gpu
  use bonded_gpu_mod,only:read_bonded_energy_gpu
#if KEY_BLOCK==1
  use domdec_block,only:combine_biflam_from_gpu
#endif
#endif
#if KEY_ALLMPI==1
  use mpi
#endif
#if KEY_LARMORD==1
  use larmord, only : qlarmord, ener_larmord
#endif
#if KEY_VALBOND==1
  use valbond
#endif
  use drude,only:eanisotropy, thole_nbx, thole_nb, ehyper
  use vector
  use pucker_mod,only:epucker,ncspuck
#if KEY_CONSHELIX==1 /*conshelix*/
  use conshelix_fcm
#endif /* (conshelix)*/
  use gopair, only : qgopair, EGoPair
  use prssre
#if KEY_PNM==1
  use pnm, only : pnm_ene
#endif
#if KEY_MNDO97==1
  use qmmmewald_module, only: qmmm_ewald_r
#endif
#if KEY_DHDGB==1
!AP/MF
  use dhdgb,only:totals,qfhdgb
#endif
! #if KEY_MOBHY==1
  use mobhy_mod, only: ENMOBHY,LMOBHY
! #endif
use extbond 
use triakern ! Pezzella 10.12.2019
use fullkern ! Pezzella 11.07.2020 

use api_func, only: func_call, func_is_set, mlpot_call, mlpot_is_set

!
  implicit none
  !

#if KEY_DENBIAS==1
  real(chm_real) :: ener_denbias
#endif

#if KEY_CONSHELIX==1
  INTEGER NNSEL(2)
  real(chm_real) ENEMIND,ENETANE,ENERANE,ENEANG
  LOGICAL OMLLIMIT
#endif
  real(chm_real),allocatable,dimension(:) :: lfrand
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DX(*),DY(*),DZ(*)
#if KEY_DHDGB==1
!AP/MF
  real(chm_real) :: SDEF(*)
  real(chm_real) :: DS_DHDGB(*)
#endif
!!  INTEGER BNBND(*),BIMAG(*)
  type(nonbondDataStructure) BNBND
  type(imageDataStructure) BIMAG

  INTEGER NDD1, ICALL
  real(chm_real) DD1(*)
  integer iupt(*), kupt(*)
  LOGICAL QSECD
#if KEY_ASPENER==1
  real(chm_real) EIMSLV
#endif
  real(chm_real) EANIS                              ! (E. Harder)
  logical q_calc_aniso

  integer,dimension(1:2) :: gbmv_dummyarr
  INTEGER GbFixed

  integer iwr
#if KEY_PARALLEL==1 /*pll*/
  INTEGER NALLWK
  !.ab.HybH.for//brodcasting+qtest.      PARAMETER (NALLWK=LENENT+LENENV+10)
  PARAMETER (NALLWK=3*LENENT+LENENV+10)
  real(chm_real) ALLWRK(NALLWK)
  real(chm_real) TIMMER, TIMME1
  INTEGER IPT
#if KEY_ALLMPI==1 /*allmpi*/
  integer :: status
  real(chm_real),allocatable,dimension(:) :: frcbuf
#endif /* (allmpi)*/
#endif /* (pll)*/
  !
#if KEY_MMFF==1 /*mmff*/
  INTEGER DERIVS
#endif /* (mmff)*/
  !
#if KEY_CHEQ==1 /*cheq*/
  !      INTEGER J,I1,K
  INTEGER MYJ, IJUNK,JMYK
  real(chm_real) TJUNK
#endif /* (cheq)*/
  ! PJ 06/2005
#if KEY_SCCDFTB==1
  INTEGER N,J,II
#endif

#if KEY_SCCDFTB==1 || KEY_CONSHELIX==1
  integer :: k
#endif

  INTEGER MSCINIT
  INTEGER J1,J3
  INTEGER I,ATFRST,ATLAST
  !.ab.Note: ATFRST,ATLAST were to be removed.
  !.ab.      Now Found in EEXIPS(),AIPSFIN(),AIPSPBC()...
  !.ab.Utile.
  LOGICAL QTEST
  real(chm_real)  ECSUM,ECHARMBF
  !WXW  IPS energy term
#if KEY_NBIPS==1
  real(chm_real)  EIPSVDW,EIPSELE,ENBAIPS,EELAIPS
#endif
  LOGICAL TBM2,TBM4,TBHY3,TBHY4
  LOGICAL QFARRAY,QDIM4
  LOGICAL QOK
  LOGICAL QOKANGLE
#if KEY_LRVDW==1
  real(chm_real) ct3,lrcn
#endif
  integer,dimension(1)::idum
#if KEY_LOOKUP==1
  real(chm_real) ENBW,EELW
#endif
#if KEY_AFM==1
  real(chm_real) eafm
#endif
  real(chm_real) :: cont_diff, cont_tol
  ! H Kamberaj (Tsallis MD) November 2007
#if KEY_TSALLIS==1
  integer :: ierror
#endif
  real(chm_real) FRANDTHETA,THETAF2,THETAF3      !New THETA-DYNAMICS
  real(chm_real) adum(1)

  real(chm_real) vpress(9)  ! For DMCONS
  real(chm_real) dmc_ener(ndmc), dmc_rho(ndmc) ! For DMCONS

#if KEY_DOMDEC==1
  integer ig,is,iq
  logical got_recip_results
#endif
#if KEY_ZEROM==1
  real(chm_real) :: TEMP0(1)
#endif
#if KEY_MNDO97==1
  logical :: q_ewald_all
#endif
  real(chm_real),dimension(:),allocatable :: dx_a,dy_a,dz_a
#if KEY_LJPME==1
  integer :: atom, ljptr
  real(chm_real) :: ljvdwr, ljeff
#endif
!end of declarations-------------------------------------------------------

  call timer_start(T_energy)
  call timer_start( T_inte)


  ! These update function of the EXTBOND and TRIAKERN module, respectively,
  ! deactivate the potential energy and force contribution of bonds and angles
  ! wich are handled by a custom potential. 
  ! It sets the respective element in the ICB and ICT array to zero, which
  ! would point to the proper force constant in the respective parameter 
  ! arrays. If zero, there bond and angle potential contribution is skipped.
  ! More convenient would be the addition of these function to the
  ! CODES subroutine in UPDATE.F90, which handle the generation of the ICB and
  ! ICT arrays and would just need a one-time execution.
  ! Doing so, however, lead to the deletion of the bonds (and angle, untested) 
  ! in the psf file and thus to further issues, such as non-bonding 
  ! contributions. 
  ! Kai Toepfer - DEC 2023
  if(qxtbd) call ext_bond_update()
  if(qkern) call tria_kern_update()
  
#if KEY_DMCONS==1
  ! these must be initialized or test first
  ! returns bad first derivatives in parallel runs
  dmc_ener(1:ndmc) = zero
  dmc_rho(1:ndmc) = zero
#endif

  if(.not. allocated(ccnba))then
     if (prnlev >= 2) write(outu,*)"ENERGY    CCNBA not allocated"
     !
     ! MH07: problem with empty nonbond list when doing pure QM calcs!
     !       make it so CCNBA is always at least one !?
     !         stop
  endif
#if KEY_DOMDEC==1
  ! Sanity check: DMCONS and RXNCOR require constraint node = PME
  if (q_domdec .and. .not.qpme) then
#if KEY_DMCONS==1
    if (qdmc) call wrndie(-5,'<energy>','Under DOMDEC use of DMCONS requires PME')
#endif
#if KEY_RXNCOR==1
     if (rxnind /= 0 .and. qeterm(umbr)) &
       call wrndie(-5,'<energy>','Under DOMDEC use of RXNCOR requires PME')
#endif
  endif
  ! Set auxdata
  if (q_domdec .and. q_split .and. QPME) then
     ! Initialize auxdata to correct size
     call init_auxdata(nauxdata, auxdata)
  endif
#endif

  ! Define the atom bounds for this processor. !+xw050623
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
  ATFRST=1+IPARPT(MYNOD)
  ATLAST=IPARPT(MYNODP)
#else /* (parfmain)*/
  ATFRST=1
  ATLAST=NATOM
#endif /* (parfmain)                           -xw050623*/

#if KEY_MNDO97==1
  ! to control the kspace call between pure mm and qm/mm-pme part.
  q_ewald_all  =.true.
#endif

  call set_mtsflags()

  !
  !     Check if we have enough time and no interruptions
  !
  CALL CHKLIM
  IF(ATLIM) then
     call timer_stop( T_inte)
     call timer_stop(T_Energy)
     RETURN
  endif

#if KEY_MMFF==1 /*mmff*/
  DERIVS=1
  IF(QSECD) DERIVS=3
#endif /*  (mmff)*/
  !
  IF (TIMER .GT. 0) CALL WRTTIM('Time into energy:')
  !
  ! . Increment the energy calls counters. ECALLS is used for updating.
  !   TOT1ST counts all calls where the first  derivative gets computed.
  !   TOT2ND counts all calls where the second derivative gets computed.
  ECALLS = ECALLS + ICALL
  !
#if KEY_PARALLEL==1 /*pll*/
!  CALL PSYNC()           !APH: this PSYNC could be commented
#if KEY_DOMDEC==1
  if (.not.q_domdec) then
#endif
     TIMMER = ECLOCK()
#if KEY_DOMDEC==1
  endif
#endif
#endif /*  (pll)*/
#if KEY_TSM==1 /*tsm*/
  !     Make sure that "back" atoms have the same coordinates as the "piggy"
  !     atoms.
  IF(QTSM.AND.PIGSET) CALL PIGCVSET(X,Y,Z)
#endif /*  (tsm)*/
  !
#if KEY_BLOCK==1 /*block*/
!ldm
  IF (.NOT.NOFORC) TOT1ST = TOT1ST + 1
  IF (QPRNTV)  THEN
     CALL SETFLA(VBBOND,NBLOCK)
     CALL SETFLA(VBANG ,NBLOCK)
     CALL SETFLA(VBTORS,NBLOCK)
     CALL SETFLA(VBIMPR,NBLOCK)
#if KEY_CMAP==1
     CALL SETFLA(VBCMAP,NBLOCK)
#endif
     CALL SETFLA(VBGENB,NBLOCK)
     CALL SETFLA(VBELEC,NBLOCK)
     CALL SETFLA(VBVDW ,NBLOCK)
  ENDIF
!ldm
#else /*    (block)*/
  TOT1ST = TOT1ST + 1
#endif /*  (block)*/
#if KEY_MSCALE==1
  !
  !     MSCALE: Send the coordinates to start calculating on subsystems
  MSCINIT=1
  IF(QMSCALE) CALL EMSCALE(MSCINIT,NATOM,LENENT, &
       X,Y,Z,DX,DY,DZ,ETERM,EPRESS,EPROP,QSECD,DD1,QDYNCALL)

#endif
#if KEY_BLOCK==1 /*ldm*/
  !     INITIALIZE THE LAGRANGE MULTIPLIER AND FORCES
  if(qmld) then
#if KEY_DOMDEC_GPU==1
     ! In domdec_gpu, msld_setblcoef is called in domdec/enbxfast.src
     if (.not.q_domdec .or. .not.q_gpu) &
#endif
          call msld_setblcoef(nblock,ninter,bixlam,blcoep)
  endif
  IF(QLDM.or.QLMC) THEN
     CALL SETFLA(BIFLAM,NBLOCK)
     IF(QTHETADM) THETAF=0.0                    !New THETA-DYNAMICS
     IF(RSTP)THEN
        IF(NRST.EQ.2) CALL SETFLA(BFRST,NBLOCK)
        IF(NRST.EQ.1)THEN
#if KEY_PBOUND==1
           IF(.NOT.QBOUN) THEN
#endif
                 IF(NTRANS.GT.0) CALL WRNDIE(-3,'<ENERGY>', &
                      'IMAGE DOES NOT WORK WITH NRST=1' )
#if KEY_PBOUND==1
           ENDIF
#endif
           call SETFLA(ENVDX,NATOM*(NBLOCK-1))
           call SETFLA(ENVDY,NATOM*(NBLOCK-1))
           call SETFLA(ENVDZ,NATOM*(NBLOCK-1))
        ELSE
           call SETFLA(ENVDX,NATOM)
           call SETFLA(ENVDY,NATOM)
           call SETFLA(ENVDZ,NATOM)
        ENDIF
     ENDIF
  ENDIF
#endif /* (ldm)*/
  IF(QSECD) TOT2ND = TOT2ND + 1

  ! Initialize TSALLIS MD (HK,2007)
#if KEY_TSALLIS==1 /*tsallis_init*/
  if (qttsall) then
     allocate(tdx(natom), stat = ierror)
     if (ierror /= 0) then
        write(outu,'("Natom=  ", I10)') natom
        call wrndie(-4,'<ENERGY>', &
             'Abort: unable to allocate TDX')
     endif
     allocate(tdy(natom), stat = ierror)
     if (ierror /= 0) then
        write(outu,'("Natom=  ", I10)') natom
        call wrndie(-4,'<ENERGY>', &
             'Abort: unable to allocate TDY')
     endif
     allocate(tdz(natom), stat = ierror)
     if (ierror /= 0) then
        write(outu,'("Natom=  ", I10)') natom
        call wrndie(-4,'<ENERGY>', &
             'Abort: unable to allocate TDZ')
     endif
     tdx=zero
     tdy=zero
     tdz=zero
     TSNDF=0
     EBTORS=ZERO
  endif
#endif /* (tsallis_init)*/

#if KEY_PERT==1 /*pert*/
  IF(QPERT) THEN
     !.ab.
#if KEY_BLOCK==1
     IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
          'HYBH and PERT incompatible.')
#endif
     !.ab.
     CALL EPERT(X, Y, Z, DX, DY, DZ ,QECONT, ECONT, &
          NDD1, DD1, QSECD, ICALL)
     call timer_stop( T_inte)
     call timer_stop(T_Energy)
     !
     ! Remove any forces on zero mass atom (only)
     IF(QHOLO) THEN
        CALL HOLONOMF(DX,DY,DZ,X,Y,Z,.FALSE.,.TRUE.,QOK)
     ENDIF
#if KEY_BLOCK==1
     if(.not.QLDM) then     /*LDM  Cc New PBLOCK*/
#endif
        RETURN
#if KEY_BLOCK==1 /*ldm*/
     else          !Cc New PBLOCK
        goto 1967  !Cc New PBLOCK go directly to calculate the lamda force
     endif         !Cc New PBLOCK
#endif
  ENDIF
#endif /*   (pert)*/
  !
  ! . Zero the energy terms calculated in this routine.
  call zero_energy_terms()
#if KEY_SCCDFTB==1
  if(qsccb) then
     dvdlb=0.0d0
     dvdlub=0.0d0
     dvdla=0.0d0
     dvdlp=0.0d0
     dvdlip=0.0d0
     dvdlcp=0.0d0
     dvdle=0.0d0
     dvdlv=0.0d0
     dvdlscc=0.0d0
  endif
#endif
  !
  ! . Zero out the force arrays
#if KEY_DOMDEC==1
  if (q_domdec) then
!$omp parallel do schedule(static) private(i, ig, is, iq)
     do i=1,zonelist(8)
        ig = groupl(i)
        call group_out(group(ig), is, iq)
        dx(is:iq) = zero
        dy(is:iq) = zero
        dz(is:iq) = zero
     enddo
!$omp end parallel do
  else
#endif
     DO I=1,NATOM
        DX(I)=ZERO
        DY(I)=ZERO
        DZ(I)=ZERO
#if KEY_FOURD==1
        IF(DIM4) DFDIM(I)=ZERO
#endif
        ! PJ 06/2005
#if KEY_PIPF==1
        IF (QPIPF .AND. QPFDYN) THEN
           CALL DUZERO(I,DUIND)
        ENDIF
#endif
     ENDDO
#if KEY_DOMDEC==1
  endif
#endif
#if KEY_DHDGB==1
!AP/MF
     IF (QFHDGB) THEN
        DO I=1,TOTALS
           DS_DHDGB(I)=ZERO
        ENDDO
     ENDIF
#endif
#if KEY_CHEQ==1
     !  Zero force charges on entering energy routine
     if(qcg) DCH(1:natom) = ZERO
#endif
#if KEY_FLUCQ==1 /*fluc*/
  IF (QFLUC) THEN
     ! Clear charge force array
     FQCFOR(1:NATOM) = ZERO
  ENDIF
#endif /*    (fluc)*/
  !
  !--------------------------------------------------------
  ! Setup counters for energy counts for MTS use.
  TBM2=.TRUE.   ! calculate some angles and bonds
  TBM4=.TRUE.   ! calculate some dihedrals and impropers
  TBHY3=.FALSE. ! flag to do faster part of nonbond list
  J1=1
  J3=1
#if KEY_MTS==1 /*mts*/
  IF (QTBMTS) THEN
     CALL SELMTS(TBM2,TBM4,J1,J3,.FALSE.)
  ELSE
#endif /*     (mts)*/
     NBONM=NBOND
     NTHETM=NTHETA
     NTHUBM=NTHETA
#if KEY_MTS==1
  ENDIF
#endif
  !
  !--------------------------------------------------------
  ! Zero the energy contribution array.
  IF(QECONT) THEN
     DO I=1,NATOM
        ECONT(I)=ZERO
     ENDDO
  ENDIF
  !--------------------------------------------------------
  !
  ! Set up second derivative arrays
  IF(QSECD) THEN
#if KEY_DOMDEC==1
     if (q_domdec) then
        call wrndie(-5,'<energy>','Second derivatives not supported in DOMDEC')
     endif
#endif
     IF(NDD1.LE.0) CALL DIE
     CALL FILUPT(IUPT,NDD1)
     CALL FILUPT(KUPT,NDD1)
#if KEY_MMFF==1 /*mmff*/
     !
     !        LTSD - keeps temporary SD array generated by MMFF
     !               we need it because MMFF generates SD array
     !               as lower triangle whereas charmm needs
     !               upper triangle.
     !
     if(FFIELD.eq.MMFF) then
        call chmalloc('energy.src','ENERGY','LTSD',3*NATOM*(3*NATOM+1)/2,crl=LTSD)
        LTSD=zero
     endif
#endif /*  (mmff)*/
  ENDIF
  !
  ! . Construct coordinates for all image atoms.
  ! . Do it here so user energy routines will work with images.
  !
#if KEY_PBOUND==1
  if(.not.qboun) then
#endif
        IF(NTRANS.GT.0) THEN
#if KEY_DOMDEC==1
           if(.not.q_domdec) then
#endif
              ! . Construct the coordinates.
              CALL TRANSO(X,Y,Z,DX,DY,DZ,.TRUE.,QECONT,ECONT,NATOM,NTRANS, &
                   IMTRNS,BIMAG%IMATPT,BIMAG%IMATTR, &
                   NOROT,NATIM &
#if KEY_FLUCQ==1
                   ,QFLUC,CG,FQCFOR    &
#endif
                   )
#if KEY_CHEQ==1 /*cheq*/
              ! . Update charges on image atoms
              CALL CPCGIMG(BIMAG%IMATTR)
#endif /*    (cheq)*/
              ! PJ 06/2005
#if KEY_PIPF==1
              ! . Update dipoles on image atoms
              IF (QPIPF .AND. QPFDYN) THEN
                 CALL CPDPIMG(BIMAG%IMATTR,UIND)
              ENDIF
#endif
#if KEY_DOMDEC==1
           endif
#endif
           ! . Calculate the volume of the system.
           CALL GETVOL(EPROP(VOLUME))
        ENDIF
#if KEY_PBOUND==1
  endif
#endif

#if KEY_CHEQ==1 /*cheq*/
  IF (QCG.AND.QETERM(ELEC)) THEN
     IF (QCGINV.AND.LFAST.LT.0) THEN
        CALL WRNDIE(3,'<ENERGY>','CGINV not supported without FAST')
        QCGINV=.FALSE.
     ENDIF
     IF (QCGINV.AND.NATIM.GT.NATOM) THEN
        CALL WRNDIE(3,'<ENERGY>','CGINV not supported with images')
        WRITE(OUTU,*) 'Charge inversion computed with real atoms only'
     ENDIF
     !  Call routine to find equilibrium charges for this configuration by
     !  solution of linear equations (if QCGINV=T), initialize charge forces.
     CALL QCGSET(NATOM,X,Y,Z,DX,DY,DZ,CG, &
          BNBND%INBLO,BNBND%IBLO14, &
          BNBND%INB14,BNBND%JNB,IAC,LELEC, &
#if KEY_WCA==1
          LSOFTCORE0,SCCUTR0,WCA,     &
#endif
          OUTU)
     !  Make sure charges are updated in "fast" charge arrays cgx.
     ! ??? the CGX array no longer exists.  make sure charges are getting updated
     !        IF (NTRANS .GT. 0) THEN
     !          CALL CGCOPY(CG,CGX,NATOM,NATIM)
     !        ENDIF
  ELSE
     EPROP(CGPOT)=ZERO
  ENDIF
#endif /*   (cheq)*/
  !
  !=======================================================================
  !========= START OF ENERGY TERM CALCULATIONS ===========================
  !=======================================================================
  !
  !  IMPORTANT NOTE TO DEVELOPERS:
  !     The order of energy terms is important for the calculation of the
  !     virial. Before the call to VIRIAL, only conservative force terms
  !     may appear (i.e. those that do not involve an externally applied
  !     force or torque).  Below the call to VIRIAL, should ONLY appear
  !     energy terms involving an external force on the system
  !     (KSPACE is an exception to this rule).
  !
  !     Other methods may have specific placement needs as described...
  !     Please read and think before you put your new thing in here! - BRB
  !
  !-------------------------------------------------------------------
  ! Calculate bead energy for path integrals.
  ! Must be done before other energy terms since it might change
  ! the coordinates of some atoms.
#if KEY_PATHINT==1
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
     IF (QETERM(PINT) .AND. QPINT) THEN
        ! UW_05:Haibo Yu
#if KEY_PARALLEL==1
        !        IF(NUMNOD.GT.1) CALL WRNDIE(-4,'<ENERGY>',
#endif
#if KEY_PARALLEL==1
        !    &                   'No parallel code for EPINT')
#endif
#if KEY_PARALLEL==1
        IF(MYNOD.EQ.INODE(19)) THEN
#endif
           !.ab.
#if KEY_BLOCK==1
           IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
                'HYBH and EPINT incompatible.')
#endif
           !.ab.
           CALL EPINT(X,Y,Z,DX,DY,DZ)
#if KEY_PARALLEL==1
        ENDIF
#endif
     ENDIF
#if KEY_MTS==1
  ENDIF
#endif
#endif
  !-------------------------------------------------------------------
  !  User supplied energy term
  !
  !     MTS note: USERE is always called.  The user is responsible for
  !     selecting when to calculate this term within USERE. - BRB
  !
  !     Energy terms should not be calculated before USERE, since it
  !     is sometimes used to setup data or coordinates....
  !     There is another user routine called at the end of the energy
  !     calculation (USRACM) for post processing and statistics...
  !
#if KEY_LIBRARY == 1
  if (qeterm(user) .and. func_is_set()) then
     call func_call(eterm(user) &
          , natom &
          , x, y, z &
          , dx, dy, dz)
  end if
  ! Addition Kai Toepfer May 2022
  if (qeterm(user) .and. mlpot_is_set()) then
    call mlpot_call(&
        eterm(user), &
        natom, ntrans, natim, &
        x, y, z, &
        dx, dy, dz, &
        bnbnd%jnb, bnbnd%inblo, &
        bimag%imattr, bimag%imjnb, bimag%imblo)
  end if
#else /* KEY_LIBRARY */
#if KEY_PARALLEL==1
  IF(MYNOD.EQ.0) THEN
#endif
     IF (QETERM(USER)) THEN
        !.ab. Normally, empty routine...
#if KEY_BLOCK==1
        !            IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>',
        !     $           'HYBH and USERE currently incompatible.')
#endif
        !.ab.
        CALL USERE(ETERM(USER),X,Y,Z,DX,DY,DZ,QECONT,ECONT,NATOM)
        IF (TIMER.GT.1) CALL WRTTIM('User energy times:')
     ENDIF
! #if KEY_MOBHY==1
     IF (LMOBHY) CALL ENMOBHY(ETERM(ASP),X,Y,Z,DX,DY,DZ,QECONT,ECONT,NATOM)
! #endif
#if KEY_PARALLEL==1
  ENDIF
#endif
#endif /* KEY_LIBRARY */

  !---------------------CROSS----------------------------------------
#if KEY_RMD==1
#if KEY_PARALLEL==1
  IF(MYNOD.EQ.0) THEN
#endif
     IF((NCRUN.GT.0).AND.QETERM(CROS)) THEN
        !.ab.
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and ECROS currently incompatible.')
#endif
        !.ab.
        CALL ECROSS(ETERM(CROS),X,Y,Z,DX,DY,DZ,NATOM)
     ENDIF
#if KEY_PARALLEL==1
  ENDIF
#endif
#endif
  !---------------------MRMD----------------------------------------
#if KEY_MRMD==1
#if KEY_PARALLEL==1
  IF(MYNOD.EQ.0) THEN
#endif
     IF(mrmd_active.AND.QETERM(MRMD)) THEN
        !.ab.
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and MRMD currently incompatible.')
#endif
        !.ab.
        CALL EMRMD(ETERM(MRMD),X,Y,Z,DX,DY,DZ,NATOM)
     ENDIF
#if KEY_PARALLEL==1
  ENDIF
#endif
#endif
!----------------------MMPT--------------------------------------------
#if KEY_MMPT==1
#if KEY_PARALLEL==1
    IF(MYNOD.EQ.0) THEN
#endif
      IF(QETERM(MMPT)) THEN
        CALL EMMPT(ETERM(MMPT),X,Y,Z,DX,DY,DZ,NATOM)
      ENDIF
#if KEY_PARALLEL==1
    ENDIF
#endif
#endif 
!----------------------MSMMPT------------------------------------------
#if KEY_MSMMPT==1
#if KEY_PARALLEL==1
    IF(MYNOD.EQ.0) THEN   
#endif
      IF(QETERM(MSMMPT)) THEN
        CALL EMSMMPT(ETERM(MSMMPT),X,Y,Z,DX,DY,DZ,NATOM)
      ENDIF
#if KEY_PARALLEL==1
    ENDIF                 
#endif
#endif
!----------------------VALBOND-----------------------------------------
#if KEY_VALBOND==1
#if KEY_PARALLEL==1
    IF(MYNOD.EQ.0) THEN
#endif
        CALL EANGVB(ETERM(VALB),X,Y,Z,DX,DY,DZ,NATOM)
#if KEY_PARALLEL==1
    ENDIF
#endif
#endif
  !---------------------PNM-------------------------------------------
#if KEY_PNM==1 /*pnm*/
  ! plastic network energy, 0504PJ07
     IF (QETERM(PNME)) THEN
        !.ab.
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and PNMENE currently incompatible.')
#endif
        !.ab.
        CALL PNM_ENE(ETERM(PNME),X,Y,Z,DX,DY,DZ,QECONT,ECONT,NATOM)
        IF (TIMER.GT.1) CALL WRTTIM('User energy times:')
     ENDIF
#endif /* (pnm)*/
  !-----------------------------------------------------------------------
#if KEY_PARALLEL==1 /*pll*/
#if KEY_DOMDEC==1
  if (.not.q_domdec) then
#endif
     ! get energies
     TMERI(TIMINTE) = TMERI(TIMINTE) + ECLOCK()-TIMMER
     TIMMER = ECLOCK()
     !  CALL PSYNC()          !APH: this psync could be commented out
     TMERI(TIMWAIT) = TMERI(TIMWAIT) + ECLOCK()-TIMMER
     TIMMER = ECLOCK()
#if KEY_DOMDEC==1
  endif
#endif
#endif /*        (pll)*/
  !  Generalized Born Solvation energy term
  !
  !  First call set-up and get alphas and sigmas for this config
  !

  IF (QGBMV) THEN
     CALL timer_stpstrt(T_inte,T_nonbon)
  ENDIF

  IF (QGenBorn) THEN
     !.ab.Not compatible. Would probably be interesting. Contact A.Blondel.
#if KEY_BLOCK==1
     IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
          'HYBH and GenBorn incompatible, see code.')
#endif
     !.ab.
     CALL timer_stpstrt(T_inte,T_nonbon)
     CALL timer_start(T_gborn)
     If ( QSecD ) Call WrnDie(-3,'<ENERGY>', &
          'No 2nd derivitives w/ Generalized Born.')
#if KEY_BLOCK==1 /*ldm*/
     If ( RSTP ) Call WrnDie(-3,'<ENERGY>', &
          'No restraining potential w/ Generalized Born.')
#endif
     If ( NTrans .gt. 0 ) Call WrnDie(-3,'<ENERGY>', &
          'Images not implemented w/ Generalized Born.')

#if KEY_BLOCK==1 /*block*/
     ! Generalized Born with Block (Type 1)
     IF(IGenType .EQ. 1) THEN
        IF (QBLOCK) THEN

           Call FillAlpGB1( X, Y, Z, &
                BNBND%JnB, BNBND%INblO, &
                BNBND%InB14, BNBND%IbLo14, &
                .false., idum, idum, GbFixed )
           IF (TIMER.GT.1) &
                CALL WRTTIM('Generalized Born set-up times:')


           Call GBSolv1( ETERM(GBEnr), X, Y, Z, DX, DY, DZ, &
                BNBND%JnB, BNBND%INblO, &
                BNBND%InB14, BNBND%IbLo14, &
                .false., idum, idum, GbFixed )
           IF (TIMER.GT.1) &
                CALL WRTTIM('Generalized Born energy times:')
        ELSE
           CALL WRNDIE(-3,'<ENERGY>', &
                'BLOCK should be on when IGenType = 1.')
        ENDIF
        !-------------------------------------------------------------------
        ! Generalized Born with Block (Type 2)
     ELSE IF (IGenType .EQ. 2) THEN
        IF (QBLOCK) THEN
           Call FillAlpGB2( X, Y, Z, &
                BNBND%JnB, BNBND%INblO, &
                BNBND%InB14, BNBND%IbLo14, &
                .false., idum, idum, GbFixed )
           IF (TIMER.GT.1) &
                CALL WRTTIM('Generalized Born set-up times:')
           !
           !
           Call GBSolv2( ETERM(GBEnr), X, Y, Z, DX, DY, DZ, &
                BNBND%JnB, BNBND%INblO, &
                BNBND%InB14, BNBND%IbLo14, &
                .false., idum, idum, GbFixed )
           IF (TIMER.GT.1) &
                CALL WRTTIM('Generalized Born energy times:')
        ELSE
           CALL WRNDIE(-3,'<ENERGY>', &
                'BLOCK should be on when IGenType = 2.')
        ENDIF
     ELSE
#endif /* (block)*/

        !-------------------------------------------------------------------
        ! Common Generalized Born
        Call FillAlpGB( X, Y, Z, &
             BNBND%JnB, BNBND%INblO, &
             BNBND%InB14, BNBND%IbLo14, &
             .false., idum, idum &
#if KEY_GBFIXAT==1
             ,GbFixed                   &
#endif
             )

        IF (TIMER.GT.1) &
             CALL WRTTIM('Generalized Born set-up times:')
        !
        !
        Call GBSolv( ETERM(GBEnr), X, Y, Z, DX, DY, DZ,  &
             BNBND%JnB, BNBND%INblO, &
             BNBND%InB14, BNBND%IbLo14, &
             .false., idum, idum &
#if KEY_GBFIXAT==1
             ,GbFixed                       &
#endif
             )
        IF (TIMER.GT.1) &
             CALL WRTTIM('Generalized Born energy times:')
#if KEY_BLOCK==1
     ENDIF
#endif /*  BLOCK*/
  ELSE IF (IGenType .eq. 20 .and. (.not. qgbmv_omm)) THEN
  ! qgbmv_omm: T -> call omm_energy in OpenMM plugins directory
  !            F -> call RunGBMV in energy directory
     CALL timer_start(T_gborn)
     Call RunGBMV( ETERM(GBEnr), &
          BNBND%JnB, BNBND%INblO, &
          BNBND%InB14, BNBND%IbLo14, &
          .false.,gbmv_dummyarr,gbmv_dummyarr &
          )
     CALL timer_stop(T_gborn)
     CALL timer_stpstrt(T_nonbon,T_inte)
  ELSE IF (IGenType .eq. 21 .and. (.not. qgbmv_omm)) THEN
     CALL timer_start(T_gborn)
     Call RunGBMVGrid( ETERM(GBEnr), &
          BNBND%JnB, BNBND%INblO, &
          BNBND%InB14, BNBND%IbLo14 &
          )
     CALL timer_stop(T_gborn)
     CALL timer_stpstrt(T_nonbon,T_inte)
  ENDIF
  IF (QGenBorn) THEN
     CALL timer_stop(T_gborn)
     CALL timer_stpstrt(T_nonbon,T_inte)
  ENDIF

#if KEY_PARALLEL==1 /*pll*/
#if KEY_DOMDEC==1
  if (.not.q_domdec) then
#endif
     ! get energies
     TMERI(TIMEXTE) = TMERI(TIMEXTE) + ECLOCK()-TIMMER
     TIMMER = ECLOCK()
#if KEY_DOMDEC==1
  endif
#endif
#endif /*         (pll)*/
  !------------------------------------------------------------------
  !  GBIM Generalized Born w/ Implicit Membrane
  !
  !  First call set-up and get alphas and sigmas for this config
  !
  IF (QGBIMb) THEN
     !.ab.Not compatible. Could be interesting for ThermIntg. A.Blondel.
#if KEY_BLOCK==1
     IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
          'HYBH and GenBorn+ImpMembr. currently incompatible.')
#endif
     !.ab.
     If ( QSecD ) Call WrnDie(-3,'<ENERGY>', &
          'No 2nd derivitives w/ GBIM.')
     If ( NTrans .gt. 0 ) Call WrnDie(-3,'<ENERGY>', &
          'Images not implemented w/ GBIM.')
     !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     Call FillAlpGBIM( X, Y, Z, &
          BNBND%JnB, BNBND%INblO, &
          BNBND%InB14, BNBND%IbLo14, &
          .false.)
     IF (TIMER.GT.1)  &
          CALL WRTTIM('Generalized Born set-up times:')
     !
     Call GBSolvIM( ETERM(GBEnr), X, Y, Z, DX, DY, DZ,  &
          BNBND%JnB, BNBND%INblO, &
          BNBND%InB14, BNBND%IbLo14, &
          .false.)
     IF (TIMER.GT.1)  &
          CALL WRTTIM('Generalized Born energy times:')
  ENDIF
  !-------------------------------------------------------------------
#if KEY_TSM==1 /*tsm*/
  ! shf 4/4/90 thermodynamic simulation method
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
     IF(QTSM .AND. QETERM(TSM)) THEN
        !.ab.
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and TSM incompatible.')
#endif
        !.ab.
        CALL TSME(X,Y,Z,DX,DY,DZ,.TRUE.)
        IF (TIMER.GT.1) CALL WRTTIM('TSM energy times:')
     ENDIF
#if KEY_MTS==1
  ENDIF
#endif
#endif /*  (tsm)*/
  !-------------------------------------------------------------------
  ! All of the QM/MM terms are here...
#if KEY_QUANTUM==1 /*quantumTerms*/
  ! . Use Quantum Mechanical / Molecular Mechanical Force Field
  IF (NATQM.GT.0) THEN
     !.ab.Nothing done to date for QM in HybH... A.Blondel.
#if KEY_BLOCK==1
     IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
          'HYBH and QM EAMPAC currently incompatible.')
#endif
     !.ab.
#if KEY_MTS==1 /*mtsQ*/
        IF (QTBMTS) THEN
           IF(ENE1) THEN
#if KEY_PARALLEL==1
              !CCC        IF(MYNOD.EQ.INODE(2)) THEN
#endif
              ! The parallel has been handled within EAMPAC
              ! namkh 09/20/04
              IF (QETERM(QMEL)) CALL EAMPAC(ETERM(QMEL),X,Y,Z,DX,DY,DZ)
              IF (NATOM.GT.NATQM .AND. QETERM(QMVDW)) &
                   CALL EVDWQM(ETERM(QMVDW),X,Y,Z,DX,DY,DZ)
#if KEY_PARALLEL==1
              IF(MYNOD.EQ.INODE(2)) THEN
#endif
                 IF (TIMER.GT.1) &
                      CALL WRTTIM('MTS QM & QM/MM energy times:')
#if KEY_PARALLEL==1
              ENDIF
#endif
           ENDIF
        ELSE
#endif /* (mtsQ)*/
#if KEY_MTS==1
           IF(ENE3) THEN
#endif
#if KEY_PARALLEL==1
              !CCC    IF(MYNOD.EQ.INODE(2)) THEN
#endif
              ! The parallel has been handled within EAMPAC
              ! namkh 09/20/04
              IF (QETERM(QMEL)) CALL EAMPAC(ETERM(QMEL),X,Y,Z,DX,DY,DZ)
              IF (NATOM.GT.NATQM .AND. QETERM(QMVDW)) &
                   CALL EVDWQM(ETERM(QMVDW),X,Y,Z,DX,DY,DZ)
#if KEY_PARALLEL==1
              IF(MYNOD.EQ.INODE(2)) THEN
#endif
                 IF (TIMER.GT.1) CALL WRTTIM('QM & QM/MM energy times:')
#if KEY_PARALLEL==1
              ENDIF
#endif
#if KEY_MTS==1
           ENDIF
#endif
#if KEY_MTS==1 /*mtsQ*/
        ENDIF
#endif /*  (mtsQ)*/
  ENDIF
#endif /*   (quantumTerms)*/
  !-------------------------------------------------------------------
#if KEY_GAMESSUK==1 || KEY_GAMESS==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
  !  Ab initio energy from GAMESS (UK version, US version, Q-chem)
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
     !.ab.Nothing done to date for QM in HybH... A.Blondel.
#if KEY_BLOCK==1
     if(qmused)then
     IF ((QHYBH).AND.(QETERM(QMEL))) CALL WRNDIE(-5,'<ENERGY>', &
          'HYBH and QM GUKENE currently incompatible.')
     endif
#endif
     !.ab.
     !
! JZ_UW12: For SMBP, skip QM call here and call it from SMBP0
#if KEY_SMBP==1
     IF (.NOT.(QPBEQ .AND. QETERM(SMBP) .AND. QSMBP)) THEN
#endif
     IF (QETERM(QMEL)) CALL GUKENE(ETERM(QMEL),X,Y,Z,DX,DY,DZ,CG, &
          AMASS,IAC,NATOM,NDD1,DD1,QSECD,IUPT,KUPT)
#if KEY_SMBP==1
     ENDIF
#endif
#if KEY_QCHEM==1
     IF(QWRITeinp) THEN
       QWRITeinp=.false.
       RETURN
     ENDIF
#endif
#if KEY_MTS==1
  ENDIF
#endif
#endif

#if KEY_CADPAC==1
  ! Energy from CADPAC
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
     !.ab.Nothing done to date for QM in HybH... A.Blondel.
#if KEY_BLOCK==1
     IF ((QHYBH).AND.(QETERM(QMEL))) CALL WRNDIE(-5,'<ENERGY>', &
          'HYBH and QM CADENE currently incompatible.')
#endif
     !.ab.
     IF (QETERM(QMEL)) CALL CADENE(ETERM(QMEL),X,Y,Z,DX,DY,DZ,NATOM)
#if KEY_MTS==1
  ENDIF
#endif
#endif /* KEY_CADPAC */

#if KEY_MNDO97==1
  ! Energy from MNDO97
  IF(NUMAT.GT.0) THEN
#if KEY_MTS==1
     IF(ENE3) THEN
#endif
        !.ab.Nothing done to date for QM in HybH... A.Blondel.
#if KEY_BLOCK==1
        if(qmused)then
        IF ((QHYBH).AND.((QETERM(QMEL)).OR.(QETERM(QMVDW)))) &
             CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and QM MNDENE currently incompatible.')
        endif
#endif

        ! memory allocation
        allocate(dx_a(natom))
        allocate(dy_a(natom))
        allocate(dz_a(natom))
        ! step1.
        ! fist preparation step: needed for kspace term evaluations.
        IF (QETERM(QMEL)) CALL MNDENE_STEP1(X,Y,Z)

! P. Ojeda 2016 Hybrid QM/MM implementation
!$omp parallel NUM_THREADS(2)
!$omp sections
!$omp section
        ! K_space term done here.. for qm/mm
#if KEY_DOMDEC==1
        if (.not.q_domdec .or. (q_domdec .and. .not.q_split)) then
#endif
           IF(LEWALD .and. QPME) THEN
              call timer_start(T_nonbon)
              call timer_start(T_ewald)
              call timer_start(T_rec)
#if KEY_DOMDEC==1
              if (.not.q_domdec) then
#endif
#if KEY_PARALLEL==1
                 TMERI(TIMINTE) = TMERI(TIMINTE) + ECLOCK()-TIMMER
                 TIMMER = ECLOCK()
#endif
#if KEY_DOMDEC==1
              endif
#endif

#if KEY_BLOCK==1
              IF (QHYBH) QBLOCK=.FALSE.
#endif
              ! for kspace call control.
              q_ewald_all =.false.
              !do i=1,natom
              !   qmmm_ewald_r%d_ewald_mm(1:3,i)=zero
              !end do

              ! domdec has yet been supported..
              CALL KSPACE_qmmm_prep(ETERM(EWKSUM),ETERM(EWSELF),ETERM(EWQCOR),ETERM(EWUTIL),&
                      QETERM(EWKSUM),QETERM(EWSELF),QETERM(EWQCOR),QETERM(EWUTIL), &
                      X,Y,Z,qmmm_ewald_r%d_ewald_mm(1:3,1:natom),NATOM,CG,CGTOT &
#if KEY_FLUCQ==1
                      ,QFLUC,FQCFOR     &
#endif
                      )
#if KEY_BLOCK==1
              IF (QHYBH) QBLOCK=.TRUE.
#endif
              call timer_stop(T_rec)
              call timer_stop(T_ewald)
              call timer_stop(T_nonbon)
#if KEY_DOMDEC==1
              if (.not.q_domdec) then
#endif
#if KEY_PARALLEL==1
                 TMERI(TIMEXTE) = TMERI(TIMEXTE) + ECLOCK()-TIMMER
                 TIMMER = ECLOCK()
#endif
#if KEY_DOMDEC==1
              end if
#endif
            END IF
#if KEY_DOMDEC==1
        end if
#endif
#if KEY_PARALLEL==1
        ! (see below..)
        if(numnod>num_cpus .and. QETERM(QMEL)) CALL MNDENE_STEP2(X,Y,Z)
#endif
        !------------------------END SECTION 1----------------------------!

!$omp section
#if KEY_PARALLEL==1
        if(numnod>num_cpus) then
           ! all nonbonded energies are evaluated here.
           call nonbonded_energy()
        else
#endif
           ! for mm-mm image/bonded terms.
           call bonded_energy()
           call nonbonded_image_energy()

           ! probably here... so no need to do the sum below.
           IF (NATOM.GT.NUMAT .AND. QETERM(QMVDW)) &
                   CALL EVDWQM(ETERM(QMVDW),X,Y,Z,DX,DY,DZ)

           ! 2) Step 2.
           IF (QETERM(QMEL)) CALL MNDENE_STEP2(X,Y,Z)
#if KEY_PARALLEL==1
        end if
#endif
        !-------------------------END SECTION 2---------------------------!
!$omp end sections
!$omp end parallel
        !======================END OPENMP PARALLEL========================!

        ! now compute energy (and gradients).
        IF (QETERM(QMEL)) CALL MNDENE_STEP3(ETERM(QMEL),X,Y,Z,DX,DY,DZ)

        ! gradients computation.
!$omp parallel  NUM_THREADS(2)
!$omp sections
!$omp section
        !
        do i=1,natom
           dx_a(i)=zero
           dy_a(i)=zero
           dz_a(i)=zero
        end do

        ! gradients computation and gho/leps energies/gradients..
#if KEY_PARALLEL==1
        if(numnod>num_cpus) then
           IF (QETERM(QMEL)) CALL MNDENE_STEP4_1(x,y,z,dx_a,dy_a,dz_a)
        else
#endif
           IF (QETERM(QMEL)) CALL MNDENE_STEP4(ETERM(QMEL),x,y,z,dx_a,dy_a,dz_a)
#if KEY_PARALLEL==1
        end if
#endif
        !------------------------END SECTION 1----------------------------!

!$omp section
        !
#if KEY_PARALLEL==1
        if(numnod>num_cpus) then
           IF (QETERM(QMEL)) CALL MNDENE_STEP4_2(ETERM(QMEL),x,y,z,dx,dy,dz)

           ! probably here... so no need to do the sum below.
           IF (NATOM.GT.NUMAT .AND. QETERM(QMVDW)) &
                   CALL EVDWQM(ETERM(QMVDW),X,Y,Z,DX,DY,DZ)

           ! for mm-mm image/bonded terms.
           call bonded_energy()
           call nonbonded_image_energy()
        else
#endif
           ! see here... that all nonbonded energies are evaluated here.
           call nonbonded_energy()
#if KEY_PARALLEL==1
        end if
#endif
        !-------------------------END SECTION 2---------------------------!
!$omp end sections

        ! This is done here as dx/dy/dz are split into dx_a/dy_a/dz_a for nonbonded_energy terms.
        ! add forces into the main force array.
!$omp do
        do i=1,natom
           dx(i) = dx(i) + dx_a(i)
           dy(i) = dy(i) + dy_a(i)
           dz(i) = dz(i) + dz_a(i)
        end do
!$omp end do
!$omp end parallel
        ! memory deallocation.
        if(allocated(dx_a)) deallocate(dx_a)
        if(allocated(dy_a)) deallocate(dy_a)
        if(allocated(dz_a)) deallocate(dz_a)

#if KEY_PARALLEL==1
        IF(MYNOD.EQ.0) THEN
#endif
           IF (TIMER.GT.1) CALL WRTTIM('QM & QM/MM energy times:')
#if KEY_PARALLEL==1
        ENDIF
#endif
#if KEY_MTS==1
     ENDIF
#endif
  ENDIF
#endif
  !-------------------------------------------------------------------
#if KEY_SCCDFTB==1
  ! Energy from SCCDFTB
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
     !OLD  QC: If we run in parallel mode - only the master will do SCCDFTB
     !OLD  because SCCDFTB is not parallelized.
     !OLD  but if we have the PARASCC mode, we allow everynode to call
     !OLD  SCCTBENE because we would use the replica mode which
     !OLD  assigns each replica to a different node.
     !     QC:UW_031205 - We allow parallel here because we can split up the
     !     eWald calculations. In the future, we might consider splitting
     !     up the MM atoms (NOT URGENT THOUGH).
     !UW_031205##IF PARALLEL
     !UW_031205      IF (MYNOD.EQ.0) THEN
     !UW_031205##ENDIF
     IF (QETERM(QMEL)) THEN
        !.ab.Nothing done to date for QM in HybH... A.Blondel.
#if KEY_BLOCK==1
        if(qmused)then
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and QM SCCTBENE currently incompatible.')
        endif
#endif
        !.ab.
        ! for lamda from guohui
        if (qlamda) then
           ! ========================= STATE A ==============================
           ! restoring control parameters
           nel=nela
           do n=1,nndim
              izp2(n)=izp2a(n)
           enddo
           TELEC=TELEC1
           SCFTOL=SCFTOL1
           MXITSCF=MXITSCF1
           LPRVEC=LPRVEC1
           ICHSCC=ICHSCC1
           nscctc=nscctc1
           do ii=1,nscctc1
              scctyp(ii)=scctyp1(ii)
              scczin(ii)=scczin1(ii)
              sccatm(ii)=sccatm1(ii)
           enddo
           do ii=1,nscctc1
              iqmlst(ii,1)=iqmlst1(ii,1)
           enddo
           !            Zero out the CG array if Mulliken has been called
           IF (LMULIK) THEN
              DO J=1,NSCCRP
                 DO I=1,NSCCTC
                    CG(IQMLST(I,J))=ZERO
                 ENDDO
              ENDDO
           ENDIF
           !MG_UW1211: for spin-polarization get the right number of electrons
           lcolspin=lcolspin1
           if (lcolspin) then
             nelup=nelupa
             neldown=neldowna
           endif
           !MG_UW1211: bookkeeping of charges (globally stored at ltm/sccdftbsrc_ltm.src)
           if (lldep) then
             do i=1,3*nscctc
               ql(i)=qla(i)
             enddo
           else
             do i=1,nscctc
               qmat(i)=qmata(i)
             enddo
           endif
           if (lcolspin) then
             do i=1,3*nscctc
               qlup(i)=qlupa(i)
               qldown(i)=qldowna(i)
             enddo
           endif

           scal=1.0d0-scclamd

           ! QC:UW_04: set for GSBP option
#if KEY_PBEQ==1
#if KEY_GSBP==1 || KEY_SMBP==1
           !     if SCC-GSBP is needed, skip the SCC calculations here
           !     and instead do them inside GSBP0 - because we need the GSBP
           !     related quantities to be computed first
           !     JZ_UW12: Added SMBP
           IF (.NOT.(QPBEQ .AND. (QETERM(GSBP) .OR. QETERM(SMBP)) .AND. &
              (QGSBP .OR. QSMBP))) THEN
#endif
#endif
!             XIAO_QC_UW0609 (add force flag to SCCTBENE)
              CALL SCCTBENE(ETERM1,X,Y,Z,DX,DY,DZ,NATOM,.true.)

              !MG_UW1211: bookkeeping of charges (globally stored at ltm/sccdftbsrc_ltm.src)
              if (lldep) then
                do i=1,3*nscctc
                  qla(i)=ql(i)
                enddo
              else
                do i=1,nscctc
                  qmata(i)=qmat(i)
                enddo
              endif
              if (lcolspin) then
                do i=1,3*nscctc
                  qlupa(i)=qlup(i)
                  qldowna(i)=qldown(i)
                enddo
              endif
#if KEY_PBEQ==1
#if KEY_GSBP==1 || KEY_SMBP==1
           ENDIF
#endif
#endif

           ! ========================= STATE B ==============================
           ! restoring control parameters
           ! restoring parameters given by dylcao to sth used in eglcao for B
           nel=nelb
           do n=1,nndim
              izp2(n)=izp2b(n)
           enddo

           TELEC=TELEC2
           SCFTOL=SCFTOL2
           MXITSCF=MXITSCF2
           LPRVEC=LPRVEC2
           ICHSCC=ICHSCC2
           nscctc=nscctc2
           do ii=1,nscctc2
              scctyp(ii)=scctyp2(ii)
              scczin(ii)=scczin2(ii)
              sccatm(ii)=sccatm2(ii)
           enddo
           do ii=1,nscctc2
              iqmlst(ii,1)=iqmlst2(ii,1)
           enddo
           !            Zero out the CG array if Mulliken has been called
           !            We need to do this for both state A and B
           IF (LMULIK) THEN
              DO J=1,NSCCRP
                 DO I=1,NSCCTC
                    CG(IQMLST(I,J))=ZERO
                 ENDDO
              ENDDO
           ENDIF
           !MG_UW1211: for spin-polarization get the right number of electrons
           lcolspin=lcolspin2
           if (lcolspin) then
             nelup=nelupb
             neldown=neldownb
           endif
           !MG_UW1211: bookkeeping of charges (globally stored at ltm/sccdftbsrc_ltm.src)
           if (lldep) then
             do i=1,3*nscctc
               ql(i)=qlb(i)
             enddo
           else
             do i=1,nscctc
               qmat(i)=qmatb(i)
             enddo
           endif
           if (lcolspin) then
             do i=1,3*nscctc
               qlup(i)=qlupb(i)
               qldown(i)=qldownb(i)
             enddo
           endif

           scal=scclamd

#if KEY_PBEQ==1
#if KEY_GSBP==1 || KEY_SMBP==1
           !     if SCC-GSBP is needed, skip the SCC calculations here
           !     and instead do them inside GSBP0 - because we need the GSBP
           !     related quantities to be computed first
           !     JZ_UW12: Added SMBP
           IF (.NOT.(QPBEQ .AND. (QETERM(GSBP) .OR. QETERM(SMBP)) .AND. &
              (QGSBP .OR. QSMBP))) THEN
#endif
#endif
!             XIAO_QC_UW0609 (add force flag to SCCTBENE)
!              CALL SCCTBENE(ETERM1,X,Y,Z,DX,DY,DZ,NATOM,.true.)
              CALL SCCTBENE(ETERM2,X,Y,Z,DX,DY,DZ,NATOM,.true.)
!is it eterm2???? /MH09/              CALL SCCTBENE(ETERM2,X,Y,Z,DX,DY,DZ,NATOM)

              !MG_UW1211: bookkeeping of charges (globally stored at ltm/sccdftbsrc_ltm.src)
              if (lldep) then
                do i=1,3*nscctc
                  qlb(i)=ql(i)
                enddo
              else
                do i=1,nscctc
                  qmatb(i)=qmat(i)
                enddo
              endif
              if (lcolspin) then
                do i=1,3*nscctc
                  qlupb(i)=qlup(i)
                  qldownb(i)=qldown(i)
                enddo
              endif

              ! ===================================================================
              ! Scale the total energies
              ! forces have been scaled in SCCTBENE
              ETERM(QMEL)=(1.0d0-scclamd)*ETERM1+scclamd*ETERM2

              ! Update free energy derivatives if necessary
              ! ----------------------------------------------------------------
              if(qsccb) dvdlscc=-ETERM1+ETERM2

#if KEY_PBEQ==1
#if KEY_GSBP==1 || KEY_SMBP==1
           ENDIF
#endif
#endif

           ! ----------------------------------------------------------------
        else
           !          Single-state calculations
           !          Zero out the CG array if Mulliken has been called
           !          We need to do this for both state A and B
           IF (LMULIK) THEN
              DO J=1,NSCCRP
                 DO I=1,NSCCTC
                    CG(IQMLST(I,J))=ZERO
                 ENDDO
              ENDDO
           ENDIF
!     XIAO_QC_UW0609 (add force flag to SCCTBENE)
#if KEY_PBEQ==1
      IF (.NOT. QPBEQ ) THEN
#if KEY_GSBP==1 || KEY_SMBP==1
           !     if SCC-GSBP is needed, skip the SCC calculations here
           !     and instead do them inside GSBP0 - because we need the GSBP
           !     related quantities to be computed first
           !     JZ_UW12: Added SMBP
           IF (.NOT.(QPBEQ .AND. (QETERM(GSBP) .OR. QETERM(SMBP)) .AND. &
              (QGSBP .OR. QSMBP))) THEN
#endif
#endif
              CALL SCCTBENE(ETERM(QMEL),X,Y,Z,DX,DY,DZ,NATOM,.true.)
#if KEY_PBEQ==1
#if KEY_GSBP==1 || KEY_SMBP==1
           ENDIF
#endif
      ENDIF
#endif
           ! ----------------------------------------------------------------
        endif ! qlamda

     ENDIF ! QETERM
     !UW_031205##IF PARALLEL
     !UW_031205      ENDIF
     !UW_031205##ENDIF
#if KEY_MTS==1
  ENDIF
#endif
#endif
   !----------------------------------------------------------------
#if KEY_SSNMR==1
   ! SSNMR restraints (Jinhyuk Lee and Wonpil Im)
#if KEY_PARALLEL==1
      IF(MYNOD.EQ.0) THEN
#endif
         IF((CSNUM.GT.0) .AND. QETERM(ECS)) THEN
            CALL CSCNS(ETERM(ECS),DX,DY,DZ,X,Y,Z, &
                       FORCs,expcs,csnum,CSIPT,csinm,cslis, &
                       sigma,phics, &
                       kmincs,plcs,kmaxcs,pucs,rsw,expt, &
                       knew,rswl,lsofta,ldipc,nudc, &
                       lanal,stoag,numcat,ldcabs, &
                       lconsnu,rmsdval,rmsdvdc, &
                       iunij)
            IF (TIMER.GT.1) CALL WRTTIM('CS constraint energy times:')
         ENDIF
#if KEY_PARALLEL==1
      ENDIF
#endif
#endif
#if KEY_RDC==1
   ! RDC restraints.
#if KEY_PARALLEL==1
   IF (MYNOD.EQ.0) THEN
#endif
           IF(QRDC) THEN
             call RDC1(NATOM,X,Y,Z,AMASS,ETERM(ERDC),DX,DY,DZ)
             IF (TIMER.GT.1) CALL WRTTIM('RDC constraint energy times:')
           ENDIF
#if KEY_PARALLEL==1
   ENDIF
#endif
#endif
  !-------------------------------------------------------------------
#if KEY_SQUANTM==1
  ! Energy from SQUANTM
  IF (NATQM(1).GT.0) THEN
#if KEY_MTS==1
     IF(ENE3) THEN
#endif
        !.ab.Nothing done to date for QM in HybH... A.Blondel.
#if KEY_BLOCK==1
        if(qmused)then
        IF ((QHYBH).AND.((QETERM(QMEL)).OR.(QETERM(QMVDW)))) &
             CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and QM SQMMME currently incompatible.')
        endif
#endif
        !.ab.
        IF (QETERM(QMEL)) CALL SQMMME(ETERM(QMEL),X,Y,Z,DX,DY,DZ)

        IF (NATOM.GT.NATQM(1) .AND. QETERM(QMVDW)) &
             CALL EVDWQM(ETERM(QMVDW),X,Y,Z,DX,DY,DZ)
#if KEY_PARALLEL==1
        IF(MYNOD.EQ.0) THEN
#endif
           IF (TIMER.GT.1) CALL WRTTIM('QM & QM/MM energy times:')
#if KEY_PARALLEL==1
        ENDIF
#endif
#if KEY_MTS==1
     ENDIF
#endif
  END IF
#endif

#if KEY_DOMDEC_GPU==1
  if (q_gpu) then
     ! For DOMDEC_GPU, we need to calculate non-bonded energy before bonded energy:
     call timer_stpstrt(T_inte, T_nonbon)
     call nonbonded_energy()
     call timer_stpstrt(T_nonbon, T_inte)
     call bonded_energy()
  else
#endif
     ! Traditional CHARMM order:
#if KEY_MNDO97==0  /* MNDO97 related */
     call bonded_energy()
     call nonbonded_energy()
#else              /* MNDO97 related */
     if(numat<=0) then
        ! No qm/mm is setup. So, pure MM calc.
        call bonded_energy()
        call nonbonded_energy()
     end if
#endif             /* MNDO97 related */
#if KEY_DOMDEC_GPU==1
  endif
#endif
  !
  !================= RESTRAINT ENERGY TERMS ================================
  !-----------------------------------------------------------------------
  ! . Dihedral restraints.
  call calc_dihe_restraints(eterm(cdihe), x, y, z, dx, dy, dz, dd1, iupt, qsecd)
  !-----------------------------------------------------------------------
  ! . Puckering restraints.
#if KEY_MTS==1
  IF(ENE2) THEN
#endif
     IF(NCSPUCK > 0 .AND. QETERM(CPUCK)) THEN
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and CPUCK incompatible, see code.')
#endif
#if KEY_DOMDEC==1
        if (q_domdec) CALL WRNDIE(-5,'<ENERGY>','PUCKERING RESTRAINTS NOT READY FOR DOMDEC')
#endif
        call epucker(eterm(cpuck), dx, dy, dz, x, y, z)

        IF (TIMER.GT.1) &
             CALL WRTTIM('Puckering constraint energy times:')
     ENDIF
#if KEY_MTS==1
  ENDIF
#endif
  !
  !-----------------------------------------------------------------------
  !
  !-----------------------------------------------------------------------
  ! . IC restraints.
#if KEY_MTS==1
  IF(ENE2) THEN
#endif
     IF(LCIC.AND.QETERM(CINTCR)) THEN
        !.ab.Note: idem CDIHE. Contact A.Blondel.
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and CINTCR incompatible, see code.')
#endif
#if KEY_DOMDEC==1
        if (q_domdec) CALL WRNDIE(-5,'<ENERGY>','IC RESTRAINTS NOT READY FOR DOMDEC')
#endif
        CALL EICCON(ETERM(CINTCR),DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
             CCBIC,CCTIC,CCPIC,CCIIC,DD1,IUPT,QSECD, &
             KBEXPN,LUPPER)
        IF (TIMER.GT.1) CALL WRTTIM('IC constraint energy times:')
     ENDIF

  if(qxtbd.and.qeterm(bond).and.(nxtbd>0))then
    call ext_bond_ener(eterm(bond),x,y,z,dx,dy,dz)
  endif
  if(qkern.and.qeterm(bond).and.(nkern>0))then
    call tria_kern_ener(eterm(bond),x,y,z,dx,dy,dz)
  endif
 if(qflmk) then !Pezzella 11.07.2020 
    call fullkern_ener(eterm(bond),x,y,z,dx,dy,dz)
 endif 
#if KEY_MTS==1
  ENDIF
#endif
  !
  !-----------------------------------------------------------------------
#if KEY_SHAPES==1 /*shapener*/
  ! . Shape restraint energy.
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.INODE(4)) THEN
#endif
        IF((NUMESH.GT.0) .AND. QETERM(SHAP)) THEN
           !.ab.No clear reason to use with TI.
#if KEY_BLOCK==1
           IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
                'HYBH and SHAP incompatible.')
#endif
#if KEY_DOMDEC==1
           if (q_domdec) CALL WRNDIE(-5,'<ENERGY>','SHAPE RESTRAINTS NOT READY FOR DOMDEC')
#endif
           CALL ESHAPE(ETERM(SHAP),DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
                DD1,IUPT,QSECD)
           IF (TIMER.GT.1) &
                CALL WRTTIM('Shape restraint energy times:')
        ENDIF
#if KEY_PARALLEL==1
     ENDIF
#endif
#if KEY_MTS==1
  ENDIF
#endif
#endif /* (shapener)*/
  !-----------------------------------------------------------------------
#if KEY_NOMISC==0 /*nomisc2*/
  ! . NOE restraints.
  call calc_noe_restraints(eterm(noe), x, y, z, dx, dy, dz, dd1, iupt, qsecd)
!!$#if KEY_MTS==1
!!$  IF(ENE2) THEN
!!$#endif
!!$#if KEY_PARALLEL==1
!!$     IF(MYNOD.EQ.INODE(5)) THEN
!!$#endif
!!$        IF((NOENUM.GT.0) .AND. QETERM(NOE)) THEN
!!$           !.ab.
!!$#if KEY_BLOCK==1
!!$           IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
!!$                'HYBH and NOE incompatible.')
!!$#endif
!!$#if KEY_DOMDEC==1
!!$           if (q_domdec) CALL WRNDIE(-5,'<ENERGY>','NOE RESTRAINTS NOT READY FOR DOMDEC')
!!$#endif
!!$           CALL NOECNS(ETERM(NOE),DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
!!$                NOENUM,NOESCA,NOEIPT,NOEJPT,NOEINM,NOEJNM, &
!!$                NOELIS,NOEEXP,NOERMN,NOEKMN,NOERMX,NOEKMX, &
!!$                NOEFMX,NOETCN,NOEAVE,NOEMIN,DD1,IUPT,QSECD &
!!$                ,NOERSW,NOESEX,NOERAM                        & !CJH, NOE_SOFT
!!$#if KEY_PNOE==1
!!$                , IsPNOE, C0X, C0Y, C0Z                 &
!!$#endif
!!$#if KEY_PNOE==1
!!$                , MVPNOE,OC0X,OC0Y,OC0Z                 &
!!$#endif
!!$#if KEY_PNOE==1
!!$                ,TC0X,TC0Y,TC0Z                 &
!!$#endif
!!$#if KEY_PNOE==1
!!$                , NMPNOE, IMPNOE                        &
!!$#endif
!!$                )
!!$           IF (TIMER.GT.1) CALL WRTTIM('NOE constraint energy times:')
!!$        ENDIF
!!$#if KEY_PARALLEL==1
!!$     ENDIF
!!$#endif
!!$#if KEY_MTS==1
!!$  ENDIF
!!$#endif
  !-----------------------------------------------------------------------
  ! . General distance restraints.
  call calc_redcns(eterm(resd), x, y, z, dx, dy, dz, dd1, iupt, qsecd)
!!$#if KEY_MTS==1
!!$  IF(ENE2) THEN
!!$#endif
!!$     IF((REDNUM.GT.0) .AND. QETERM(RESD)) THEN
!!$        !.ab.
!!$#if KEY_BLOCK==1
!!$        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
!!$             'HYBH and RESD incompatible.')
!!$#endif
!!$#if KEY_DOMDEC==1
!!$        if (q_domdec) CALL WRNDIE(-5,'<ENERGY>','GENERAL DISTANCE RESTRAINTS NOT READY FOR DOMDEC')
!!$#endif
!!$        CALL REDCNS(ETERM(RESD),DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
!!$             REDNUM,REDSCA,REDIPT,REDILIS,REDKLIS, &
!!$             REDKVAL,REDRVAL,REDEVAL,REDIVAL,REDMVAL, &
!!$             DD1,IUPT,QSECD)
!!$        IF (TIMER.GT.1) &
!!$             CALL WRTTIM('Distance restraint energy times:')
!!$     ENDIF
!!$#if KEY_MTS==1
!!$  ENDIF
!!$#endif
#endif /* (nomisc2)*/
  !
  !-----------------------------------------------------------------------
#if KEY_FOURD==1 /*4defour*/
  ! . Fourth dimension energy term called:
  !    Note:EPROP(EPOT4) for dynamc and ETERM(BK4D) for energy are equal.
#if KEY_MTS==1
  IF(ENE2) THEN
#endif
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.INODE(7)) THEN
#endif
        IF (DIM4 .AND. QETERM(BK4D)) THEN
           !.ab.
#if KEY_BLOCK==1
           IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
                'HYBH and EPOT4 incompatible.')
#endif
#if KEY_DOMDEC==1
           if (q_domdec) CALL WRNDIE(-5,'<ENERGY>','FOURD NOT READY FOR DOMDEC')
#endif
           !.ab.
           CALL EFOUR(ETERM(BK4D),NATOM,AMASS)
           EPROP(EPOT4)=ETERM(BK4D)
        ENDIF
#if KEY_PARALLEL==1
     ENDIF
#endif
#if KEY_MTS==1
  ENDIF
#endif
#endif /* (4defour)*/
  !
  call timer_stop(  T_restr)
  call timer_stpstrt( T_inte,T_nonbon)
  !================= IMAGE ENERGY TERMS ==================================
#if KEY_MNDO97==0  /* MNDO97 related */
    ! when MNDO97 is not used.
    call nonbonded_image_energy()
#else             /* MNDO97 related */
    ! pure MM case.
    if(numat<=0) call nonbonded_image_energy()
#endif             /* MNDO97 related */
  call timer_stop( T_nonbon)
  !-----------------------------------------------------------------------
  !
  !
  ! . Adaptive Umbrella
  ! . C.BARTELS 1996/97/98
  ! . energy terms calculated below this point will not be included into
  !   the energy umbrella (e.g., K-Space part of the ewald sum).
#if KEY_ADUMB==1
  !JMS 8/2012 -- DOMDEC/ADUMB integration
  !The code that was here has been transferred to the subroutine.
  !This must be called on both direct and reciprocal nodes in order that
  !both keep track of the histograms and do WHAM if necessary.  But since the
  !adaptive umbrella energy and forces on the reciprocal node (which are calculated using
  !up to date coordinates) are transferred to the direct node, they are the ones that count.
  call calc_adumb_potential(eterm(adumb),x,y,z,dx,dy,dz,ecalls)
#endif
#if KEY_GAMUS==1
      IF(DOGAMUS)THEN
#if KEY_PARALLEL==1
        if(mynod==0)then
#endif
         CAll GAMUSEN(NATOM,X,Y,Z,DX,DY,DZ,ETERM(GAMUS))
#if KEY_PARALLEL==1
        endif
#endif
      ENDIF
#endif
#if KEY_LARMORD==1
      IF(qlarmord)THEN
#if KEY_PARALLEL==1
        if(mynod==0)then
#endif
          call ener_larmord(eterm(cspres), x, y, z, dx, dy, dz)
#if KEY_PARALLEL==1
        endif
#endif
      ENDIF
#endif
  !
  !-----------------------------------------------------------------------
  ! . Harmonic restraint terms.
  !   do just the best-fit and relative types.
#if KEY_MTS==1
  IF(ENE1) THEN
#endif
     IF(QCNSTR.AND.QETERM(CHARM)) THEN
        !.ab.Note: In fact could be there, but distort the 'context'
        !.ab..of the molecule which is contradictory with TI.
        !.ab..Comment out the lines below to use it at your own risks.
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and CNSTR not really compatible, see code.')
#endif
        CALL ECNSTR(ECHARMBF,QCNSTR,REFX,REFY,REFZ,KCNSTR,NATOM, &
             KCEXPN,XHSCALE,YHSCALE,ZHSCALE,-1, &
             NUMHSETS,TYPHSET,IHSET,QHNORT,QHNOTR, &
             X,Y,Z,DX,DY,DZ, &
             QECONT,ECONT,DD1,IUPT,QSECD &
             ,(/0/),(/0,0/),(/0,0/),(/0,0/) &
             )

        IF (TIMER.GT.1) &
             CALL WRTTIM('Harmonic constraint energy times:')
     ENDIF
#if KEY_MTS==1
  ENDIF
#endif
  !
  !-----------------------------------------------------------------------
#if KEY_NOMISC==0 /*misc_energy_1*/
  !-----------------------------------------------------------------------
  ! . Umbrella potential term
  call calc_umbrella_potential(eterm(umbr), vpress, .false.)
  !
  !-----------------------------------------------------------------------
  ! . Reaction field terms.
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.INODE(10)) THEN
#endif
        IF (QRXNFL .AND. QETERM(RXNFLD) .AND. RXNMOD.EQ.'ENERGY') THEN
           !.ab.
#if KEY_BLOCK==1
           IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
                'HYBH and RXNFLD incompatible.')
#endif
           !.ab.
           CALL ERXNFL(ETERM(RXNFLD),NATOM,CG,X,Y,Z,DX,DY,DZ, &
                adum,adum,adum,adum,0,RXNORD,EPS,EPSEXT,RXNSHL, &
                MPMMNT,RXMMNT,ERXMNT,LEGPN,LEGPND)
           IF (TIMER.GT.1) CALL WRTTIM('Reaction field energy times:')
        ENDIF
#if KEY_PARALLEL==1
     ENDIF
#endif
#if KEY_MTS==1
  ENDIF
#endif
  !
  !-----------------------------------------------------------------------
#if KEY_POLAR==1
  !BR...08-JUL-96, Stillinger-David Polarization Model for Water
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.INODE(11)) THEN
#endif
        IF (QPOLAR .AND. QETERM(POLAR)) THEN
#if KEY_BLOCK==1
           IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
                'HYBH and POLAR incompatible.')
#endif
           CALL POLAR1(ETERM(POLAR),NATOM,X,Y,Z,DX,DY,DZ, &
                NATC,IAC,ITC,CNBA,CNBB,CG)
           IF (TIMER.GT.1) &
                CALL WRTTIM('Polarization model energy times:')
        ENDIF
#if KEY_PARALLEL==1
     ENDIF
#endif
#if KEY_MTS==1
  ENDIF
#endif
#endif
  !
  !-----------------------------------------------------------------------
#if KEY_DMCONS==1
  call calc_dmcons(dmc_ener, dmc_rho, x, y, z, dx, dy, dz, qsecd, vpress, .false.)
  eterm(dmc)=SUM(dmc_ener)
#endif
  !-----------------------------------------------------------------------
#if KEY_RGYCONS==1
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.0) THEN
#endif
        IF(QRGY.AND.(.NOT.QSECD)) THEN
           !.ab.
#if KEY_BLOCK==1
           IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
                'HYBH and RGY incompatible.')
#endif
           !.ab.
           CALL ERGY(ETERM(RGY),X,Y,Z,DX,DY,DZ)
           IF (TIMER.GT.1) CALL WRTTIM('RGY constraint energy times:')
        ENDIF
#if KEY_PARALLEL==1
     ENDIF
#endif
#if KEY_MTS==1
  ENDIF
#endif
#endif
  !-----------------------------------------------------------------------
  !
#endif /* (misc_energy_1)  NOMISC*/
  !
  !
  !-----------------------------------------------------------------------
#if KEY_REPLICA==1
#if KEY_RPATH==1
  ! . REPLICA/PATH restraints. (with NEB options - jwchu)
  IF(QPATH) THEN
     !.ab.This covers all calls to epath (see below).
#if KEY_BLOCK==1
     IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
          'HYBH and PRMS/PANG incompatible.')
#endif
     !.ab.
     IF(.NOT.QPROPT) THEN
        IF(.NOT.(QPNEB.OR.QPTAU)) THEN
           ! Calculate path energy and forces .  Since we
           ! need to project all true forces out. jwchu 6/02
           CALL EPATH(ETERM(PRMS),ETERM(PANG),QETERM(PRMS), &
                QETERM(PANG),DX,DY,DZ,X,Y,Z,QSECD,ICALL)
        ENDIF
     ENDIF
  ENDIF
#endif
#endif

! RunPHMD is called after the calling of KSPACE
! !------------------------------------------------------------------------
! ! Call RunPHMD before the call to VIRIAL -- Wei Chen 2015
! ! Call RunPHMD if running explicit solvent, virial is not right and
! ! pressure is not accurate - J. Wallace
!#if KEY_PHMD==1
!     IF(QPHMD .and. (.not. QGBSW) .and. (.not. QGBMV))THEN
!        CALL RunPHMD(ETERM(PHEnr), &
!             BNBND%JnB,BNBND%INblO, &
!             BNBND%InB14,BNBND%IbLo14, &
!             CCNBA,CCNBB,IACNB,NITCC2,LOWTP, &
!             DX,DY,DZ, &
!             natim,ntrans,imtrns,iminv,NOROT, &
!             BIMAG%imatpt,BIMAG%imattr, &
!             BIMAG%IMJNB,BIMAG%IMBLO, &
!             BIMAG%NIMNBS,BIMAG%IMJNBS, &
!             BIMAG%IMBLOS)
!      ENDIF
!#endif
!  !-----------------------------------------------------------------------
#if KEY_LRVDW==1
  !   LongRange vdw contributions (beyond CUTNB)
  IF (LLRVDW) THEN
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.0) THEN
#endif
        ct3=one/(ctofnb*ctofnb*ctofnb)
        lrcn = (lrcb + lrca*ct3*ct3*third) *lrvdw_const*ct3
        ETERM(ELRC) = LRCn/EPROP(VOLUME)
        lrcn = (lrcb + two*lrca*ct3*ct3*third)*two *lrvdw_const*ct3
        PVLRC = LRCn/EPROP(VOLUME)
#if KEY_PARALLEL==1
     ENDIF
#endif
  ELSE IF (LLRVDW_MS) THEN                               ! LRC using Shirts's algorithm
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.0) THEN
#endif
        ETERM(ELRC) = lrvdw_const_ms/EPROP(VOLUME)
        PVLRC = lrvdw_const_ms/EPROP(VOLUME)
#if KEY_PARALLEL==1
     ENDIF
#endif
  else
     ETERM(ELRC) = zero
     pvlrc=zero
  ENDIF
#endif
  !
  ! . The total internal virial.
  !write(outu,*)'ENERGY>before viral(),natom,natomt,viri=',natom,natomt,eprop(viri)
  !write(outu,'(a,9f12.4)')'ENERGY>before viral(),vixx:vizz=',(epress(i),i=vixx,vizz)
  !write(50,'(3f20.7)')(x(i),y(i),z(i),i=1,natomt)
  !write(51,'(3f20.7)')(dx(i),dy(i),dz(i),i=1,natomt)
#if KEY_GRAPE==1
  ! By now all the GPU/GRAPE terms are finished so fill up the
  ! energy terms here:
  if(lgrape.and.ntrans>0) then
    eterm(elec) = ene_c - ele_im
    eterm(vdw) = ene_v - vdw_im
    eterm(imelec) = ele_im
    eterm(imvdw) = vdw_im
  endif
  ! to be fixed for parallel pressure !! FIXME
  if(lgrape.and.(mod(igrape,10)==3))then
     do i=natom+1,natomt
        if(gpuinblo(i)==0)then
           iwr=bimag%imattr(i)
           dx(iwr)=dx(iwr)+dx(i)
           dy(iwr)=dy(iwr)+dy(i)
           dz(iwr)=dz(iwr)+dz(i)
           dx(i)=zero
           dy(i)=zero
           dz(i)=zero
        endif
     enddo
  endif
#endif
#if KEY_DOMDEC==1
  if (q_domdec) then
     ! Unpack, reduce, etc. packed forces to (dx, dy, dz)
     call unpack_forces(dx, dy, dz)
     ! Calculate virial
     call calc_virial(eprop(viri), epress(vixx:vizz), x, y, z, dx, dy, dz, &
          groupl, zonelist(8), .true.)
  else
#endif
#if KEY_ZEROM==1
  if(.not.QZMOD) then
#endif
     CALL VIRAL(EPROP(VIRI),EPRESS(VIXX:VIZZ),1,NATOMT,X,Y,Z,DX,DY,DZ)
#if KEY_ZEROM==1
  endif
#endif
#if KEY_DOMDEC==1
  endif
#endif

#if KEY_LRVDW==1
  !   ADD in long range correction to diagonal virial elements

  IF (LLRVDW .OR. LLRVDW_MS) THEN
     EPRESS(VIXX) = EPRESS(VIXX)+ PVLRC
     EPRESS(VIYY) = EPRESS(VIYY)+ PVLRC
     EPRESS(VIZZ) = EPRESS(VIZZ)+ PVLRC
  ENDIF
#endif
#if KEY_NBIPS==1 /*nbips_press*/
  IF(QIPS)THEN
     ! Calculate grid based anisotropic IPS interaction
     IF(QAIPS)THEN
        call timer_start(T_nonbon)
        call timer_start(T_ips)
        call timer_start(T_ipsa)
        IF(QIPSFIN)THEN
           CALL AIPSFIN(ENBAIPS,EELAIPS,ATFRST,ATLAST,NATOM, &
                QETERM(VDW),QETERM(ELEC),LVIPS,LEIPS, &
                EPS,CG,X,Y,Z,DX,DY,DZ)
        ELSE
           CALL AIPSPBC(ENBAIPS,EELAIPS,ATFRST,ATLAST,NATOM, &
                LEWALD,QETERM(VDW),QETERM(ELEC),LVIPS,LEIPS, &
                EPS,CG,X,Y,Z,DX,DY,DZ)
        ENDIF
        IF(TIMER.GT.1) CALL WRTTIM('AIPS times:')
        ETERM(VDW)=ETERM(VDW)+ENBAIPS
        ETERM(ELEC)=ETERM(ELEC)+EELAIPS
     ENDIF
     CALL ADDVEC(EPRESS(VIXX:VIZZ),PIPSVIR,EPRESS(VIXX:VIZZ),9)
     EPROP(VIRI) = EPROP(VIRI) + &
          (PIPSVIR(1)+PIPSVIR(5)+PIPSVIR(9))/THREE
     call timer_stop(T_ipsa)
     call timer_stop(T_ips)
     call timer_stop(T_nonbon)
  ENDIF
#endif /*  (nbips_press)*/

#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('Reciprocal energy')
#endif

  !
  !  IMPORTANT NOTE TO DEVELOPERS:
  !     Above this call to VIRIAL should only appear energy terms which
  !     involve forces between atoms.  At this point, the total force
  !     and torque on the system must be zero!
  !
  !     Below this call to VIRIAL, should ONLY appear energy terms
  !     involving an external force on the system (KSPACE is an exception).
  !
  !=======================================================================

  !  K-space sum forces include both internal and external virial components.
  !  Reciprocal space part of ewald sum calculation
#if KEY_DOMDEC==1
  if (.not.q_domdec .or. (q_domdec .and. q_recip_node)) then
#endif
#if KEY_MTS==1
     IF(ENE3) THEN
#endif
        IF(LEWALD) THEN
           call timer_start(T_nonbon)
           call timer_start(T_ewald)
           call timer_start(T_rec)
#if KEY_DOMDEC==1
           if (.not.q_domdec) then
#endif
#if KEY_PARALLEL==1
              TMERI(TIMINTE) = TMERI(TIMINTE) + ECLOCK()-TIMMER
#endif
#if KEY_PARALLEL==1
              TIMMER = ECLOCK()
#endif
#if KEY_DOMDEC==1
           endif
#endif
           !.ab.HYBH supported. Special routine for forces (calculate dH/dlambda)...
           !+.ab.HYBH for the time being we assume that if QHYBH=.true. we don't use
           !+.ab. other block functionnalities -> block = false, restored after....
           !+.ab. to avoid interferrence with block introduced at the same time.
#if KEY_BLOCK==1
           IF (QHYBH) QBLOCK=.FALSE.
#endif
#if KEY_GRAPE==1
           !write(outu,*)'energy>lgrape,natom=',lgrape,natom
           !write(outu,*)'energy>ksdx?=',allocated(ksdx)
           !write(57,'(3f20.7)')(dx(iwr),dy(iwr),dz(iwr),iwr=1,natim)
           IF(LGRAPE) THEN
              !
#if KEY_PARALLEL==1
              if(qsplit)then
                 if(mynodg == (numnodg-1)) then
                    call gsen(0,1,ksdx,natom*8)
                    call gsen(0,1,ksdy,natom*8)
                    call gsen(0,1,ksdz,natom*8)
                    call gsen(0,1,ewvirial,9*8)
                    call gsen(0,1,eterm(ewksum),8)
                 endif
                 if(mynodg == 0) then
                    call grec(numnodg-1,1,ksdx,natom*8)
                    call grec(numnodg-1,1,ksdy,natom*8)
                    call grec(numnodg-1,1,ksdz,natom*8)
                    call grec(numnodg-1,1,ewvirial,9*8)
                    call grec(numnodg-1,1,eterm(ewksum),8)
                 endif
              endif
#endif
              !
              if(mod(igrape,10)==3) then
                 do iwr=1,natom
                    dx(iwr)=dx(iwr)+ksdx(iwr)
                    dy(iwr)=dy(iwr)+ksdy(iwr)
                    dz(iwr)=dz(iwr)+ksdz(iwr)
                 enddo
              endif
           ELSE
#endif
            if(qdcm) then
              call add_dcm_recip_f(ewvirial,X,Y,Z)
            else ! skip for DCM module
#if KEY_DOMDEC==1
#if KEY_DOMDEC_GPU==1
              if (q_gpu) call range_start('comm_coord_among_recip')
#endif
              if (q_domdec .and. qpme) call comm_coord_among_recip(natom, x, y, z, cg)
#if KEY_DOMDEC_GPU==1
              if (q_gpu) call range_stop()
#endif
              if (q_domdec .and. qpme .and. ndirect > 1) then
#if KEY_DOMDEC_GPU==1
                 if (q_gpu) call range_start('zero_recip_force')
#endif
                 call zero_recip_force(recipforcex, recipforcey, recipforcez)
#if KEY_DOMDEC_GPU==1
                 if (q_gpu) call range_stop()
#endif
#if KEY_MNDO97==1
                 if(q_ewald_all) then
                    ! it is the regular kspace term calculation.
#endif
                 CALL KSPACE(ETERM(EWKSUM),ETERM(EWSELF),ETERM(EWQCOR),ETERM(EWUTIL),&
                      QETERM(EWKSUM),QETERM(EWSELF),QETERM(EWQCOR),QETERM(EWUTIL), &
                      X,Y,Z,recipforcex,recipforcey,recipforcez,NATOM,CG,CGTOT &
#if KEY_LJPME==1
                      ,ljc6 &
#endif
#if KEY_FLUCQ==1
                      ,QFLUC,FQCFOR     &
#endif
                      )
#if KEY_MNDO97==1
                 else
                    ! for qm/mm-pme part, we split the KSPACE routine into
                    ! 1) energy/gradient + qm/mm-pme values above.
                    ! 2) here, we bring gradient into the main dx, dy, dz arrays + viral terms.
                    CALL KSPACE_qmmm_get_force(DX,DY,DZ,ewvirial,NATOM)
                 end if
#endif
              else
#endif
#if KEY_MNDO97==1
                 if(q_ewald_all) then
                    ! it is the regular kspace term calculation.
#endif
                 CALL KSPACE(ETERM(EWKSUM),ETERM(EWSELF),ETERM(EWQCOR),ETERM(EWUTIL),&
                      QETERM(EWKSUM),QETERM(EWSELF),QETERM(EWQCOR),QETERM(EWUTIL), &
                      X,Y,Z,DX,DY,DZ,NATOM,CG,CGTOT &
#if KEY_LJPME==1
                      ,ljc6 &
#endif
#if KEY_FLUCQ==1
                      ,QFLUC,FQCFOR     &
#endif
                      )
#if KEY_MNDO97==1
                 else
                    ! for qm/mm-pme part, we split the KSPACE routine into
                    ! 1) energy/gradient + qm/mm-pme values above.
                    ! 2) here, we bring gradient into the main dx, dy, dz arrays + viral terms.
                    CALL KSPACE_qmmm_get_force(DX,DY,DZ,ewvirial,NATOM)
                 end if
#endif
#if KEY_DOMDEC==1
              endif
#endif
#if KEY_GRAPE==1
           ENDIF
#endif
#if KEY_BLOCK==1
           IF (QHYBH) QBLOCK=.TRUE.
#endif

           endif !if QDCM (DCM module)
           ! Correct the pressure due to the k-space sum.
           CALL ADDVEC(EPRESS(VIXX:VIZZ),EWVIRIAL,EPRESS(VIXX:VIZZ),9)
           EPROP(VIRI) = EPROP(VIRI) + &
                (EWVIRIAL(1)+EWVIRIAL(5)+EWVIRIAL(9))/THREE
           call timer_stop(T_rec)
           call timer_stop(T_ewald)
           call timer_stop(T_nonbon)
#if KEY_DOMDEC==1
           if (.not.q_domdec) then
#endif
#if KEY_PARALLEL==1
              TMERI(TIMEXTE) = TMERI(TIMEXTE) + ECLOCK()-TIMMER
#endif
#if KEY_PARALLEL==1
              TIMMER = ECLOCK()
#endif
#if KEY_DOMDEC==1
           endif
#endif
        ENDIF
#if KEY_MTS==1
     ENDIF
#endif
#if KEY_DOMDEC==1
  else
     if (lewald) then
#if KEY_DOMDEC_GPU==1
       if (.not. (q_split .and. q_gpu .and. (gpu_code_version == 2))) then
#endif
         call pme_self_energy(eterm(ewself),qeterm(ewself),kappa,cg &
                 ,natom,.false. & ! For DCM module
              )
#if KEY_DOMDEC_GPU==1
       end if
#endif
     endif
#if KEY_LJPME==1
     if(qljpme) call ljpme_self_energy(ETERM(LJSEL), QETERM(LJSEL), dkappa, ljc6)
#endif
     call getvol(eprop(volume))
  endif
#endif

#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()
#endif

 !------------------------------------------------------------------------
 ! Call RunPHMD if running explicit solvent, virial is not right and
 ! pressure is broken - J. Wallace
#if KEY_PHMD==1
     IF(QPHMD .and. (.not. QGBSW) .and. (.not. QGBMV))THEN
        CALL RunPHMD(ETERM(PHEnr), &
             BNBND%JnB,BNBND%INblO, &
             BNBND%InB14,BNBND%IbLo14, &
             CCNBA,CCNBB,IACNB,NITCC2,LOWTP, &
             DX,DY,DZ, &
             natim,ntrans,imtrns,iminv,NOROT, &
             BIMAG%imatpt,BIMAG%imattr, &
             BIMAG%IMJNB,BIMAG%IMBLO, &
             BIMAG%NIMNBS,BIMAG%IMJNBS, &
             BIMAG%IMBLOS)
      ENDIF
#endif
  !-----------------------------------------------------------------------


  !
  !=======================================================================
  ! . External (or boundary) terms.
  !
  !-------------------------------------------------------------------
#if KEY_GRID==1
  !  Grid based vdW and Elec energy terms
  !
#if KEY_PARALLEL==1
  IF(MYNOD.EQ.0) THEN
#endif
     IF (QGrid .and. QGridOK) THEN
        !            Write(6,'(a)')' Call GridEnergy'
        !.ab.
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and GridEnergy incompatible.')
#endif
        Call GridEnergy(ETerm(GrvdW),ETerm(GrElec),ETerm(GrHBON), &
             XGridCenter, YGridCenter, ZGridCenter, &
             XGridMax, YGridMax, ZGridMax, DGrid, &
             GridForce, Cg, X, Y, Z, Dx, DY, DZ, &
             QETERM(GrvdW), QETERM(GrElec), QETERM(GrHBON), &
             QGrHBon,0,0,.false.,GridNoSelection)

        IF (TIMER.GT.1) &
             CALL WRTTIM('Grid potential energy times:')
     ENDIF
#if KEY_PARALLEL==1
  ENDIF
#endif
#endif
  !-----------------------------------------------------------------------
  ! . Harmonic restraint terms.
  !   do just the absolute types (not relative and best-fit).
#if KEY_MTS==1
  IF(ENE1) THEN
#endif
     IF(QCNSTR.AND.QETERM(CHARM)) THEN
        !.ab.
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and ECNSTR-C.HARM incompatible.')
#endif
        CALL ECNSTR(ETERM(CHARM),QCNSTR,REFX,REFY,REFZ,KCNSTR,NATOM, &
                KCEXPN,XHSCALE,YHSCALE,ZHSCALE,1, &
                NUMHSETS,TYPHSET,IHSET,QHNORT,QHNOTR, &
                X,Y,Z,DX,DY,DZ, &
                QECONT,ECONT,DD1,IUPT,QSECD &
                ,(/0/),(/0,0/),(/0,0/),(/0,0/) &
                )
        IF (TIMER.GT.1) &
             CALL WRTTIM('Harmonic constraint energy times:')
        !
        ETERM(CHARM)=ETERM(CHARM)+ECHARMBF
        !
     ENDIF
#if KEY_MTS==1
  ENDIF
#endif
  !-----------------------------------------------------------------------
  ! . PCA Harmonic restraint terms.
  !   do just the absolute types (not relative and best-fit).
#if KEY_MTS==1
  IF(ENE1) THEN
#endif
     IF(QCNSTR.AND.QETERM(PCHARM).and.numpca>0) THEN
        !.ab.
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and ECNSTR-C.HARM incompatible.')
#endif
        CALL ECNSTR(ETERM(PCHARM),QCNSTR,REFX,REFY,REFZ,KCNSTR,NATOM, &
                KCEXPN,XHSCALE,YHSCALE,ZHSCALE,2, &
                NUMHSETS,TYPHSET,IHSET,QHNORT,QHNOTR, &
                X,Y,Z,DX,DY,DZ, &
                QECONT,ECONT,DD1,IUPT,QSECD &
                ,numpca,pcax,pcay,pcaz &
                )
        IF (TIMER.GT.1) &
             CALL WRTTIM('PCA Harmonic constraint energy times:')
        !
        !
     ENDIF
#if KEY_MTS==1
  ENDIF
#endif
  !
  !-----------------------------------------------------------------------
#if KEY_TNPACK==1 /*eharm*/
  ! . Second harmonic restraint term (for implicit Euler integration).
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
     IF(QEHARM.AND.QETERM(EHARM)) THEN
        !.ab.
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and ECNSTR-E.HARM incompatible.')
#endif
        CALL ECNSTR(ETERM(EHARM),QEHARM,RXHARM,RYHARM, &
             RZHARM,KEHARM,NATOM, (/ 2 /), &
             (/ ONE /), (/ ONE /), (/ ONE /), 1, 1, (/ 0 /), IHHARM, (/ .false. /), (/ .false. /), &
             X,Y,Z,DX,DY,DZ,QECONT,ECONT,DD1, &
             IUPT,QSECD &
             ,(/0/),(/0,0/),(/0,0/),(/0,0/) &
             )
        IF (TIMER.GT.1) &
             CALL WRTTIM('Harmonic restraint energy times:')
     ENDIF
#if KEY_MTS==1
  ENDIF
#endif
#endif /* (eharm)*/
  !
  !-----------------------------------------------------------------------
#if KEY_NOMISC==0 /*misc_energy_2*/
  !
  !-----------------------------------------------------------------------

  !smg/mf MSU 2008-----------------------------EPMF------------------------
#if KEY_EPMF==1
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
        IF(QETERM(PMF1D) .OR. QETERM(PMF2D) ) THEN
           CALL TOTEPMF(ETERM(PMF1D),ETERM(PMF2D),X,Y,Z,DX,DY,DZ)
           IF (TIMER.GT.1) CALL WRTTIM('EPMF times:')
        ENDIF
#if KEY_MTS==1
  ENDIF
#endif
#endif
  !smg/mf MSU 2008--------------------EPMF-----------------------------------


  !smg/mf MSU 2010-----------------------------PRIMO------------------------
#if KEY_PRIMO==1
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
        IF(QETERM(PRIMO)) THEN
           CALL EPRIMO(ETERM(PRIMO),X,Y,Z,DX,DY,DZ)
           IF (TIMER.GT.1) CALL WRTTIM('EPMF times:')
        ENDIF
#if KEY_MTS==1
  ENDIF
#endif
#endif

  !VM/MF 2014 -------------------DENBIAS-------------------
#if KEY_DENBIAS==1
   if (qdenbias) then
       call calc_dbias(ener_denbias,.true.,.false.)
       !write(outu,*)'energy.src DBIAS: ',ETERM(DBIAS)
       ETERM(CHARM)=ETERM(CHARM) + ener_denbias
   endif
#endif

#if KEY_HMCOM==1
  ! . Center of mass harmonic constraint
  call calc_hmcm(eterm(hmcm), x, y, z, dx, dy, dz)
#endif

#if KEY_CPATH==1
  ! . PATH Constraint
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.INODE(8)) THEN
#endif
        IF(PATHN.GT.0 .AND.QETERM(PATH)) THEN
           CALL PATHEPROJ (ETERM(PATH),X,Y,Z,DX,DY,DZ)
           IF (TIMER.GT.1) &
                CALL WRTTIM('PATH constraint times:')
        ENDIF
#if KEY_PARALLEL==1
     ENDIF
#endif
#if KEY_MTS==1
  ENDIF
#endif
#endif

  ! . Quartic droplet constraints.
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.INODE(8)) THEN
#endif
        IF(QQCNST.AND.QETERM(CQRT)) THEN
           !.ab.
#if KEY_BLOCK==1
           IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
                'HYBH and Quartic droplet constraints incompatible.')
#endif
           !.ab.
           CALL EQUART(ETERM(CQRT),X,Y,Z,NATOM,LQMASS,AMASS,DX,DY,DZ, &
                QECONT,ECONT,KQCNST,KQEXPN,QSECD)
           IF(TIMER.GT.1) &
                CALL WRTTIM('Droplet constraint energy times:')
        ENDIF
#if KEY_PARALLEL==1
     ENDIF
#endif
#if KEY_MTS==1
  ENDIF
#endif
  !
  !-----------------------------------------------------------------------
  ! . PULLing forces, 23-JUL-96 LN
  !
#if KEY_MTS==1
  IF(ENE2) THEN
#endif
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.INODE(12)) THEN
#endif
        IF(QETERM(PULL)) THEN
           !.ab.
#if KEY_BLOCK==1
           !         IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>',
           !     $        'HYBH and EPULL incompatible.')
#endif
           !.ab.
           CALL EPULL(ETERM(PULL),DX,DY,DZ,X,Y,Z)
           IF (TIMER.GT.1) CALL WRTTIM('PULL energy times:')
        ENDIF
#if KEY_PARALLEL==1
     ENDIF
#endif
#if KEY_MTS==1
  ENDIF
#endif
  !
  !-----------------------------------------------------------------------
  ! . Stochastic boundary terms.
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.INODE(13)) THEN
#endif
        IF(QBOUND.AND.QETERM(SBNDRY)) THEN
           !.ab.
#if KEY_BLOCK==1
           !         IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>',
           !     $        'HYBH and BNDRY incompatible.')
#endif
           CALL BNDRY(ETERM(SBNDRY),X,Y,Z,DX,DY,DZ,QECONT,ECONT, &
                DD1,IUPT,QSECD)
           IF(TIMER.GT.1) CALL WRTTIM('Solvent boundary energy times:')
        ENDIF
#if KEY_PARALLEL==1
     ENDIF
#endif
#if KEY_MTS==1
  ENDIF
#endif
  !
  !-----------------------------------------------------------------------
  ! . Mean-Field-Potentials and miscelaneous constraints (B.Roux)
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.0) THEN             /* MMFP fix 11/08*/
#endif
#if KEY_PARALLEL==1
        !!        IF(MYNOD.EQ.INODE(14)) THEN
#endif
        IF(QGEO.AND.QETERM(GEO)) THEN
           !.ab.
#if KEY_BLOCK==1
           IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
                'HYBH and GEO incompatible.')
#endif
           !.ab.
           CALL GEO2(ETERM(GEO),NATOM,X,Y,Z,DX,DY,DZ, &
                LSTGEO,NGEO,NTGEO, &
                IGEO,JGEO,AMASS, &
                XRGEO,YRGEO,ZRGEO,TRGEO, &
                XDGEO,YDGEO,ZDGEO,DRGEO, &
                DTGEO,FCGEO,P1GEO,P2GEO, &
                P3GEO,IUGEO)
           IF(TIMER.GT.1) &
                CALL WRTTIM('Mean-Field-Potential energy times:')
        ENDIF
#if KEY_PARALLEL==1
     ENDIF
#endif
#if KEY_MTS==1
  ENDIF
#endif
!-----------------------------------------------------------------------
!     EPR distance restraints (B.Roux, 2010)
#if KEY_MTS==1
      IF(ENE3) THEN
#endif
         IF (QEPRR) THEN
            CALL EPR002(X, Y, Z, DX, DY, DZ, ETERM(GEO), kenpp,  &
                       nrep_EPR, nspinlabels, Hspinlabel,       &
                       nptot, Hspinlb1, Hspinlb2,               &
                       Hpcalc, Hpexp, nppp, delppp, sig2ppp,    &
                       HdistE,.false.,qtvec,tvec)

         ENDIF
#if KEY_MTS==1
      ENDIF
#endif
  !-----------------------------------------------------------------------
  !     Solvent boundary potential of D.Beglov and B.Roux
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
        IF (QSSBP .AND. QETERM(SSBP) ) THEN
           !.ab.
#if KEY_BLOCK==1
           IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
                'HYBH and SSBP incompatible.')
#endif
           !.ab.
           CALL SSBP1(NATOM,X,Y,Z,ETERM(SSBP),CG,DX,DY,DZ)
           IF (TIMER.GT.1) &
                CALL WRTTIM('Solvent-Boundary-Potential energy times:')
        ENDIF
#if KEY_MTS==1
  ENDIF
#endif
  !-----------------------------------------------------------------------
#if KEY_CONSHELIX==1
  !     Helix-Helix Constraints (Jinhyuk Lee and Wonpil Im; Taehoon Kim)
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
#if KEY_PARALLEL==1
     IF(MYNOD == 0) THEN
#endif
        IF(QCONSH.AND.(.NOT.QSECD)) THEN
           IF(PRNLEV >= 6) THEN
              WRITE(6,*) 'Number of CONS HELIX or COM :: ',CHNUM
           ENDIF
           ! initialize constraints energy
           ENEMIND=ZERO
           ENEANG=ZERO
           ENETANE=ZERO
           ENERANE=ZERO

           DO I=1,CHNUM
              OCHNUM=I
              LPARL(I)=.FALSE.
              LLIMIT(I)=.FALSE.
              LAZAXIS(I)=.FALSE.

              DO K=1,2
                 NNSEL(K)=NSEL(K,I)
              ENDDO

              IF(LCOMM(I).AND.QETERM(ECHDL)) THEN
                 IF(PRNLEV >= 6) WRITE(OUTU,*) 'COM-COM CONSTRAINTS'
                 CALL ECHE(ETERM(ECHDL),X,Y,Z,DX,DY,DZ,HDIST(I),DFOR(I),NNSEL, &
                           ASLCT(I)%a,BSLCT(I)%a,AMASS,CHUNIT(I),CHSTEP(I), &
                           ENEMIND,LQSTPRT,CCOUNT)
              ELSE
                 IF(PRNLEV >= 6) WRITE(OUTU,*) 'Helix-Helix Constraints'
                 !
                 ! GET HELIX INFORMATION
                 !
                 LLDNA=LDNA(I)
                 LLBHG=LBHG(I)
                 LLBHP=LBHP(I)

                 CALL CALHELC(X,Y,Z,I)

                 OMLLIMIT=LLIMIT(I)

                 IF(.NOT.XANG(I).AND.QETERM(ECHDL)) THEN
                    IF(LPARL(I)) THEN
                       WRITE(OUTU,*) 'PARALLEL OR LIMITS: CONST. #',I,LPARL(I),LLIMIT(I)
                    ELSE
                       CALL ECHEMD(ETERM(ECHDL),X,Y,Z,DX,DY,DZ,CRVEC,HDIST(I),DFOR(I),NNSEL, &
                                   ASLCT(I)%a,BSLCT(I)%a,AMASS,CTVEC,CPRVEC,CAVEC,CU,CEV,CBVEC, &
                                   CEVEC,CSVAL,CPVEC,OMLLIMIT,CHUNIT(I),CHSTEP(I),ENEMIND, &
                                   LQSTPRT,MCOUNT,lbhg(i),lbhp(i))
                    ENDIF
                 ELSE
                    IF(LOHEL(I)) THEN
                       IF(.NOT.LROTH(I).AND.QETERM(ECHTN)) THEN
                          IF(PRNLEV >= 6) WRITE(OUTU,*) 'TILT Angle Force Calculation'
                          ! call subroutine of tilt angle
                          CALL ECHTA(ETERM(ECHTN),X,Y,Z,DX,DY,DZ,ANGL(I),AFOR(I),NNSEL, &
                                     ASLCT(I)%a,BSLCT(I)%a,AMASS,CAVEC,CU,CEV,CBVEC,CEVEC, &
                                     CPVEC,CHUNITA(I),CHSTEPA(I),ENETANE,LQSTPRT,TCOUNT, &
                                     LTAX(I),LTAY(I),LTAZ(I),LBHG(I),LBHP(I))
                       ENDIF
                       IF(LROTH(I).AND.QETERM(ECHRN)) THEN
                          ! call subroutine of rotation angle along helical axis
                          IF(LAZAXIS(I)) THEN
                             WRITE(OUTU,*) 'HELIX is along to z axis: CONST. #',I,LAZAXIS(I)
                          ELSE
                             IF(PRNLEV >= 6) WRITE(OUTU,*) 'Rotation Angle Force Calculation'
                             CALL ECHro(ETERM(ECHRN),X,Y,Z,DX,DY,DZ,ANGL(I),AFOR(I),NNSEL, &
                                        ASLCT(I)%a,BSLCT(I)%a,AMASS,CPRVEC,CAVEC,CU,CEV, &
                                        CBVEC,CEVEC,CPVEC,CHUNITA(I),CHSTEPA(I),REFA(I), &
                                        ENERANE,LQSTPRT,RCOUNT,LHARM(I),LBHG(I),LBHP(I), &
                                        LRAX(I),LRAY(I),LRAZ(I))
                          ENDIF
                       ENDIF
                    ELSE
                       IF(QETERM(ECHAN)) THEN
                          IF(LPARL(I)) THEN
                             WRITE(OUTU,*) 'PARALLEL OR LIMITS: CONST. #',I,LPARL(I),LLIMIT(I)
                          ELSE
                             ! crossing angle
                             IF(LCROS(I)) THEN
                                IF(PRNLEV >= 6) WRITE(OUTU,*) 'CROSS Angle Force Calculation'
                                ! add cprvec parameter to echxa subroutine
                                CALL ECHXA(ETERM(ECHAN),X,Y,Z,DX,DY,DZ,ANGL(I),AFOR(I),NNSEL, &
                                        ASLCT(I)%a,BSLCT(I)%a,AMASS,CTVEC,cprvec,CAVEC,CU,CEV, &
                                        CBVEC,CEVEC,CSVAL,CPVEC,OMLLIMIT,CHUNITA(I),CHSTEPA(I), &
                                        ENEANG,LQSTPRT,XCOUNT)
                             ELSE
                                ! hinge angle
                                IF(PRNLEV >= 6) WRITE(OUTU,*) 'HINGE Angle Force Calculation'
                                CALL ECHHA(ETERM(ECHAN),X,Y,Z,DX,DY,DZ,ANGL(I),AFOR(I),NNSEL,  &
                                        ASLCT(I)%a,BSLCT(I)%a,AMASS,CAVEC,CU,CEV,CBVEC,CEVEC, &
                                        CPVEC,CHUNITA(I),CHSTEPA(I),ENEANG,LQSTPRT,HBCOUNT, &
                                        LHMOD(I))
                             ENDIF
                          ENDIF
                       ENDIF
                    ENDIF
                 ENDIF
              ENDIF
              ! FOR REAL MDSTEP
              LQSTPRT=.FALSE.
           ENDDO
#if KEY_PARALLEL==1
        ENDIF
#endif
#if KEY_MTS==1
     ENDIF
#endif
  ENDIF
#endif

  !     GB with a simple switching function
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
     IF (QGBSW .and. (.not. qgbsw_omm)) THEN
        !.ab.
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and GBSW incompatible.')
#endif
        !.ab.

        CALL timer_start(T_nonbon)
        CALL timer_start(T_gborn)

        if(prnlev > 6 ) write(outu,*)  &
             "ENERGY calling GB_LOOKUP"
        call GB_LOOKUP(X,Y,Z)

        IF (TIMER.GT.1) &
             CALL WRTTIM('Generalized Born set-up times:')
        if(prnlev > 6 ) write(outu,*)  &
             "ENERGY using GB_SOLV1"
        call GB_SOLV1(ngbsel,natim,cg,AMASS, &
             ETERM(GBEnr),dx,dy,dz, &
             BNBND%JnB,BNBND%INblO, &
             BNBND%InB14,BNBND%IbLo14, &
             ntrans,imtrns,iminv,NOROT, &
             BIMAG%imatpt,BIMAG%imattr, &
             BIMAG%IMJNB,BIMAG%IMBLO, &
             BIMAG%NIMNBS,BIMAG%IMJNBS, &
             BIMAG%IMBLOS)

        if(QGvdW) ETERM(ASP)=gvdw

        IF (TIMER.GT.1) &
             CALL WRTTIM('Generalized Born energy times:')

        if(sgamma.gt.0.0D0) then
           if(prnlev > 6 ) write(outu,*)  &
                "ENERGY using GB_SURF1"
           call GB_SURF1(natom,atype,natim, &
                PTONE, AMASS, &
                dx,dy,dz, &
                BIMAG%imattr,imtrns,NOROT)

           ETERM(ASP)= ETERM(ASP)+gbsurf*sgamma

           IF (TIMER.GT.1) &
                CALL WRTTIM('Nonpolar solvation energy times:')

        endif

        CALL timer_stop(T_gborn)
        CALL timer_stop(T_nonbon)
     ENDIF
#if KEY_MTS==1
  ENDIF
#endif

  !-----------------------------------------------------------------------
#if KEY_PHMD==1 /* phmd */
  IF (QPHMD &
#if KEY_OPENMM
    .and. (.not. qphmd_omm) &
#endif
  ) THEN
     !.ab.
#if KEY_BLOCK==1
     IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
          'HYBH and PHMD incompatible.')
#endif
     !.ab.
     ! Modified by Wei Chen 2015
    if (QGBSW) then
     CALL RunPHMD(ETERM(PHEnr), &
          BNBND%JnB,BNBND%INblO, &
          BNBND%InB14,BNBND%IbLo14, &
          CCNBA,CCNBB,IACNB,NITCC2,LOWTP, &
          DX,DY,DZ, &
          natim,ntrans,imtrns,iminv,NOROT, &
          BIMAG%imatpt,BIMAG%imattr, &
          BIMAG%IMJNB,BIMAG%IMBLO, &
          BIMAG%NIMNBS,BIMAG%IMJNBS, &
          BIMAG%IMBLOS)
    endif
    if (QGBMV) then
     CALL RunPHMD(ETERM(PHEnr), &
          BNBND%JnB,BNBND%INblO, &
          BNBND%InB14,BNBND%IbLo14, &
          CCNBA,CCNBB,IACNB,NITCC2,LOWTP, &
          DX,DY,DZ, &
          natim,ntrans,imtrns,iminv,NOROT, &
          BIMAG%imatpt,BIMAG%imattr, &
          BIMAG%IMJNB,BIMAG%IMBLO, &
          BIMAG%NIMNBS,BIMAG%IMJNBS, &
          BIMAG%IMBLOS)
    endif
  ENDIF
#endif /* phmd */
  !------------------------------------------------------------------
#if KEY_PBEQ==1
  !     Poisson-Boltzmann Solvation potential
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.INODE(16)) THEN
#endif
        IF (QPBEQ .AND. QETERM(PBELEC) .AND. QETERM(PBNP) &
             .AND. QPBF) THEN
           !.ab.
#if KEY_BLOCK==1
           IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
                'HYBH and PBS pot. incompatible.')
#endif
           !.ab.
!           XIAO_QC_UW0609 change calling sequence with SCC

           CALL PBFORCE(NATOM,X,Y,Z,CG,ETERM(PBELEC),ETERM(PBNP), &
                DX,DY,DZ,NPBEQ,ecalls,.false. &
#if KEY_SCCDFTB==1
                ,QETERM(QMEL),ETERM(QMEL),QGAS1 &
#endif
                )
           IF (TIMER.GT.1) &
                CALL WRTTIM('Poisson Boltzmann Solvation energy times:')
        ENDIF
#if KEY_PARALLEL==1
     ENDIF
#endif
#if KEY_MTS==1
  ENDIF
#endif
#if KEY_GSBP==1
  !     Generalized Solvent Boundary Potential (GSBP)
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
        IF (QPBEQ .AND. QETERM(GSBP) .AND. QGSBP) THEN
           !.ab.
#if KEY_BLOCK==1
           IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
                'HYBH and GSBP incompatible.')
#endif
           !.ab.
           CALL GSBP0(NATOM,X,Y,Z,CG,ETERM(GSBP),DX,DY,DZ, &
                ecalls,.false. &
#if KEY_SCCDFTB==1
           ,QETERM(QMEL),ETERM(QMEL) &
#endif
           )

           IF (TIMER.GT.1) &
                CALL WRTTIM('GSBP energy times:')
        ENDIF
#if KEY_MTS==1
  ENDIF
#endif
#endif
!     JZ_UW12: Added SMBP
#if KEY_SMBP==1
!     Solvent Macromolecule Boundary Potential (SMBP)
#if KEY_MTS==1
      IF(ENE3) THEN
#endif
          IF (QPBEQ .AND. QETERM(SMBP) .AND. QSMBP) THEN
             !.ab.
#if KEY_BLOCK==1
            IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
                 'HYBH and SMBP incompatible.')
#endif
            !.ab.
            CALL SMBP0(NATOM,X,Y,Z,CG,ETERM(SMBP),DX,DY,DZ, &
                       ecalls,.false. &
#if KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_G09==1
                      ,ETERM(QMEL) &
#endif
                      )

            IF (TIMER.GT.1) &
                 CALL WRTTIM('SMBP energy times:')
          ENDIF
#if KEY_MTS==1
      ENDIF
#endif
#endif
#endif
  !
  !-----------------------------------------------------------------------
#if KEY_APBS==1
  !    APBS calculated forces
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.INODE(20)) THEN
#endif
        IF (QETERM(PBELEC) .AND. QETERM(PBNP) .AND. QFAPBS ) THEN
           !.ab.
#if KEY_BLOCK==1
           IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
                'HYBH and APBS incompatible.')
#endif
           !.ab.
           CALL apbsfrc(NATOM,X,Y,Z,CG,ETERM(PBELEC),ETERM(PBNP), &
                DX,DY,DZ,ecalls, .false.)
           IF (TIMER.GT.1) &
                CALL WRTTIM(' APBS Solvation energy times:')
        ENDIF
#if KEY_PARALLEL==1
     ENDIF
#endif
#if KEY_MTS==1
  ENDIF
#endif

#endif
  !
  !-----------------------------------------------------------------------
#if KEY_FACTS==1
  if (fctrun) then
     call timer_start(T_nonbon)
     call timer_start(T_facts)
     !.ab.
#if KEY_BLOCK==1
     IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
          'HYBH and FACTS incompatible.')
      ! ------------
      ! FACTS-BLOCK
      if(qblock) then
         ! write(outu,*) 'energy.src: QBLOCK'
         ! write(outu,*) 'energy.src: call fctblk()'
         call fctblk(eterm(ifctpol)                ,&
                     eterm(ifctnpl)                ,&
                     natom                         ,&
                     BNBND%inblo   ,BNBND%jnb      ,&
                     FCTBND%fct1ilo,FCTBND%fct1jnb ,&
                     FCTBND%fct2ilo,FCTBND%fct2jnb ,&
                     natim                         ,&
                     BIMAG%imblo   ,BIMAG%imjnb    ,&
                     BIMAG%imattr                  ,&
                     FCTBND%fct3ilo,FCTBND%fct3jnb ,&
                     FCTBND%fct4ilo,FCTBND%fct4jnb ,&
                     iblckp,blcoee,blcoev )

         ! write(outu,*) 'energy.src: fctblk() call finished'
         ! write(outu,*) 'FCTPOL: ', eterm(ifctpol), 'FCTNPL: ',eterm(ifctnpl)

         call timer_stop(T_facts)
         call timer_stop(T_nonbon)
      else
#endif
         !.ab.
         call fctene(eterm(ifctpol)                 ,&
                     eterm(ifctnpl)                 ,&
                     natom                          ,&
                     BNBND%inblo ,BNBND%jnb         ,&
                     FCTBND%fct1ilo,FCTBND%fct1jnb  ,&
                     FCTBND%fct2ilo,FCTBND%fct2jnb  ,&
                     natim                          ,&
                     BIMAG%imblo ,BIMAG%imjnb       ,&
                     BIMAG%imattr                   ,&
                     FCTBND%fct3ilo,FCTBND%fct3jnb  ,&
                     FCTBND%fct4ilo,FCTBND%fct4jnb)
         call timer_stop(T_facts)
         call timer_stop(T_nonbon)
#if KEY_BLOCK==1
      endif
#endif
   endif
#endif
  !-----------------------------------------------------------------------
#if KEY_SASAE==1
  IF (QSASA) THEN
     !.ab.
#if KEY_BLOCK==1
     IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
          'HYBH and SASAE incompatible.')
#endif
     CALL SASENE(NATOM              , &
          NATIM              , &
          PSASWBL,PSASWNB, &
          BNBND%INBLO ,BNBND%JNB  , &
          BIMAG%IMBLO ,BIMAG%IMJNB, &
          BIMAG%IMATTR, &
          PSASIBL,PSASINB, &
          PSASLCT, &
          PSASIDX, &
          PSASACS )
  ENDIF
#endif
  !-----------------------------------------------------------------------
#if KEY_PRIMSH==1
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
     IF (QSHEL.AND. QETERM(SHEL))THEN
#if KEY_PARALLEL==1
        IF(MYNOD.EQ.0) THEN
#endif
           !.ab.
#if KEY_BLOCK==1
           IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>','HYBH and SHEL incompatible.')
#endif
           !.ab.
            CALL PSHEL(VDWR,IAC,ITC,X,Y,Z, &
                       ETERM(SHEL),DX,DY,DZ,NWAT,NPATM,TOTV)
#if KEY_PARALLEL==1
        ENDIF
#endif
        EPROP(VOLUME) = TOTV
#if KEY_PARALLEL==1
        CALL PSND8(EPROP(VOLUME),1)
#endif
     ENDIF
#if KEY_MTS==1
  ENDIF
#endif
#endif
  !-----------------------------------------------------------------------
  ! VMOD
#if KEY_PARALLEL==1
  IF(MYNOD.EQ.0) THEN
#endif
     IF(QVMOD.AND.QETERM(VMOD)) THEN
        !.ab.
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and VMOD incompatible.')
#endif
        CALL VMOD2(AMASS,RVMASS,RVMAST,ETERM(VMOD),NATOMV, &
             X,Y,Z,DX,DY,DZ, &
             XINI,YINI,ZINI, &
             XMODN,YMODN,ZMODN, &
             QN,QMODN,KMODNU,ISLCTV, &
             KCGRA,KROTA,NTMD)
     ENDIF
#if KEY_PARALLEL==1
  ENDIF
#endif
  !-----------------------------------------------------------------------
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.INODE(18)) THEN
#endif
        IF(QMDIP.AND.QETERM(MDIP)) THEN
           !.ab.
#if KEY_BLOCK==1
           IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
                'HYBH and MDIP incompatible.')
#endif
           CALL MDIP2(ETERM(MDIP),NATOM,X,Y,Z,DX,DY,DZ,CG, &
                LSTMDIP,NMDIP,NTMDIP, &
                XRMDIP,YRMDIP,ZRMDIP, &
                XDMDIP,YDMDIP,ZDMDIP, &
                FCMDIP,D0MDIP,PWMDIP)
           IF(TIMER.GT.1) &
                CALL WRTTIM('Mean-Field-Potential energy times:')
        ENDIF
#if KEY_PARALLEL==1
     ENDIF
#endif
#if KEY_MTS==1
  ENDIF
#endif
  !-----------------------------------------------------------------------
#if KEY_HQBM==1
  IF (HQBM) THEN
     ehqbm=zero
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.0) THEN
#endif
        !     Added by Emanuele Paci 02-JUL-1997
        !.ab.
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and HQBM incompatible.')
#endif
        !.ab.
        CALL HQBIAS(EHQBM,X,Y,Z,DX,DY,DZ,NATOM,.true.)
        IF (TIMER.GT.1) CALL WRTTIM('User energy times:')
#if KEY_PARALLEL==1
     ENDIF
#endif
  ENDIF
#endif
#if KEY_ABPO==1
     IF (Q_ABPO_ENER) CALL ABPO_ENER(X,Y,Z,DX,DY,DZ)
#endif
#if KEY_AFM==1
  !     Added by Emanuele Paci 20-Jun-2003
  IF (LAFM) THEN
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.0) THEN
#endif
        !.ab.
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and AFM incompatible.')
#endif
        !.ab.
        CALL AFM(EAFM,X,Y,Z,DX,DY,DZ,NATOM)
        IF (TIMER.GT.1) CALL WRTTIM('User energy times:')
#if KEY_PARALLEL==1
     ENDIF
#endif
  ENDIF
#endif
#if KEY_EMAP==1
      IF (LEMAPENG) THEN
!      EMAP potential
            CALL EMAPFORCE(ETERM(EEMAP),NATOM,X,Y,Z,AMASS,DX,DY,DZ)
            IF (TIMER.GT.1) CALL WRTTIM('EMAP energy times:')
      ENDIF
#endif
  !-----------------------------------------------------------------------
  !
#endif /* (misc_energy_2)  NOMISC*/
  !
  !
  !========= END OF ENERGY TERM CALCULATIONS =============================
  !=======================================================================

#if KEY_LOOKUP==1
  ! Turn off energy calculation flag if necessary
  IF(QLOOKUP .AND. IWWENR.EQ.1) IWWENR= -1
#endif
#if KEY_SCCDFTB==1
  ! QC_UW04: Now transfer the mulliken population over to the CG array
  ! which are convenient for printint/manipulations
  ! NOTE BENE: should NOT affect energy except for IMAGE????
  ! In the case of FEP, this corresponds to the state-B population

  IF (LMULIK) THEN
     DO J=1,NSCCRP
        DO I=1,NSCCTC
           CG(IQMLST(I,J))=QMULI2(I,J)
        ENDDO
     ENDDO
  ENDIF
#endif
  !=======================================================================

  !
#if KEY_DOMDEC==1
  if (.not.q_domdec) then
#endif
#if KEY_PBOUND==1
  IF(.NOT.QBOUN) THEN
#endif
        ! . Backtransform image forces
        IF(NTRANS.GT.0) THEN
           !
           ! . Calculate crystal lattice first derivatives.
           CALL TRANSI(X,Y,Z,DX,DY,DZ,QECONT,ECONT,NATOM,NTRANS, &
                IMTRNS,BIMAG%IMATPT,BIMAG%IMATTR, &
                NOROT,NATIM,IMINV,IMFORC,IMTORQ &
#if KEY_FLUCQ==1
                ,QFLUC,FQCFOR    &
#endif
                )
        ENDIF
#if KEY_CHEQ==1
        ! transform charge forces from image atoms onto real atoms
        IF(QCG) THEN
           CALL QCGTRANSI(DCH,NATOM,NATIM,BIMAG%IMATTR)
        ENDIF
#endif
        ! PJ 06/2005
#if KEY_PIPF==1
        ! transform dipole forces from image atoms onto real atoms
        IF(QPIPF .AND. QPFDYN) THEN
           CALL DPTRANSI(DUIND,NATOM,NATIM,BIMAG%IMATTR)
        ENDIF
#endif
#if KEY_PBOUND==1
  ENDIF
#endif
#if KEY_DOMDEC==1
  endif
#endif

  !
  !=======================================================================
#if KEY_FLUCQ==1
  ! Calculates charge forces from electronegativities, and updates parallel
  ! nodes if necessary
  IF (QFLUC) CALL FQFORC
#endif

  ! . Finish up.
  IF (TIMER .EQ. 1) THEN
     CALL WRTTIM('Total energy times:')
#if KEY_MTS==1
     IF (QTBMTS) THEN
        IF(ENE1) CALL WRTTIM('MTS FIRST  CALL TIMES:')
        IF(ENE2) CALL WRTTIM('MTS MEDIUM CALL TIMES:')
        IF(ENE3) CALL WRTTIM('MTS SLOW   CALL TIMES:')
     ENDIF
#endif
  ENDIF
  !
  !---------------------------------------------------------------

#if KEY_PARALLEL==1 /*paramain*/
  !
  ! H Kamberaj (MD using scaling of torsions, November 2007)
#if KEY_TSALLIS==1
  IF (QTTSALL) THEN
     DO I=mynodp,natom,numnod
        DX(I)=DX(I) + TSFACT * TDX(I)
        DY(I)=DY(I) + TSFACT * TDY(I)
        DZ(I)=DZ(I) + TSFACT * TDZ(I)
     ENDDO
  ENDIF
#endif

#if KEY_DOMDEC==1
  got_recip_results = .false.
#if KEY_DOMDEC_GPU==1
  if (q_gpu) then
     if (q_split .and. QPME) then
        call range_start('wait_results_from_recip')
        call wait_results_from_recip()
        call range_stop()
        got_recip_results = .true.
     endif
     !if ((qeterm(vdw) .or. qeterm(elec)) .or. qeterm(ewexcl) .or. &
     !     (gpu_code_version==2 .and. &
     !     ((q_recip_node .and. (qeterm(ewksum) .or. qeterm(ewself))) .or. qeterm(bond) .or. &
     !     qeterm(angle) .or.  qeterm(ureyb) .or. qeterm(dihe) .or. qeterm(imdihe)))) then
     ! Reduce forces and virial (latter only if required) on device
     call reduce_force_virial_gpu()
     ! Copy forces, virial and energy to host
     call copy_force_virial_energy_from_gpu()
     ! Wait for copying to finish
     call wait_force_virial_energy_from_gpu()
     ! Combine GPU forces to the global force
     call combine_gpu_forces_to_global_forces(dx, dy, dz)
#if KEY_BLOCK==1
     if (qmld) then
        ! Combine biflam and biflam2 from GPU to the corresponding CPU arrays
        call combine_biflam_from_gpu()
     endif
#endif
     !endif
     ! Read energies and virial that was calculated by GPU
     call read_direct_virial_gpu(eprop(viri), epress(vixx:vizz))
     call read_nonbond_energy_gpu(eterm(vdw), eterm(elec), eterm(ewexcl))
     if (gpu_code_version==2) then
        if (q_recip_node .and. qpme) then
           call read_recip_virial_gpu(ewvirial)
           call read_recip_energy_gpu(eterm(ewksum), eterm(ewself))
           ! Correct the pressure due to the k-space sum.
           CALL ADDVEC(EPRESS(VIXX:VIZZ),EWVIRIAL,EPRESS(VIXX:VIZZ),9)
           EPROP(VIRI) = EPROP(VIRI) + &
                (EWVIRIAL(1)+EWVIRIAL(5)+EWVIRIAL(9))/THREE
        endif
        call read_bonded_energy_gpu(eterm(bond), eterm(angle), eterm(ureyb), &
             eterm(dihe), eterm(imdihe))
     endif
  endif
#endif
#endif
  !-----------------------------------------------------------------------

  ! Process code for parallelization
  call timer_start(T_fcomm)
#if KEY_DOMDEC==1
  if (.not.q_domdec) then
#endif
     TMERI(TIMINTE) = TMERI(TIMINTE) + ECLOCK()-TIMMER
     TIMMER = ECLOCK()
     !  CALL PSYNC()                 !APH: this psync could be commented out
     TMERI(TIMWAIT) = TMERI(TIMWAIT) + ECLOCK()-TIMMER
     TIMMER = ECLOCK()
#if KEY_DOMDEC==1
  endif
#endif
  !
  IMERI = IMERI + 1
  !
  ! Combine forces.
#if KEY_DOMDEC==1
  if (q_domdec) then
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('transfer_force')
#endif
     call transfer_force(dx, dy, dz)
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif
  else
#endif
#if KEY_ALLMPI==1 /*allmpi*/
     !      print *,"Forces  MPI_REDUCE_SCATTER 8"
     CALL MPI8REDUCE_SCATTER(dx,dx(IPARPT(MYNOD)+1),NPARPT, &
          MPI_DOUBLE_PRECISION,MPI_SUM,COMM_CHARMM, &
          STATUS)
     CALL MPI8REDUCE_SCATTER(dy,DY(IPARPT(MYNOD)+1),NPARPT, &
          MPI_DOUBLE_PRECISION,MPI_SUM,COMM_CHARMM, &
          STATUS)
     CALL MPI8REDUCE_SCATTER(dz,DZ(IPARPT(MYNOD)+1),NPARPT, &
          MPI_DOUBLE_PRECISION,MPI_SUM,COMM_CHARMM, &
          STATUS)
#else /*           (allmpi)*/
#if KEY_SPACDEC==1 /*spacedec*/
     !     This call could go into VDGSUM ???
     !      write(*,*)'ENERGY-0>before forcsr:me=',mynod
     CALL FORCSR(DX,DY,DZ)
     !      write(*,*)'ENERGY-0>after forcsr:me=',mynod
#else /*      (spacedec)*/
if(.not.NOFORC) then
     CALL VDGSUM(DX,DY,DZ,0)
endif
#endif /*     (spacedec)*/
#endif /*     (allmpi)*/
#if KEY_DOMDEC==1
  endif
#endif
  call timer_stop( T_fcomm )

#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('Rest of energy')
#endif

#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('Recip communication')
#endif

#if KEY_DOMDEC==1 /*domdec*/
  ! Add k-space forces, energies, and virial from recip CPUs to direct CPUs
  if (q_domdec .and. q_split .and. QPME) then
     call timer_start(T_fcomm2)
     if (.not. got_recip_results) call wait_results_from_recip()
     call unpack_results_from_recip(dx, dy, dz, auxdata, nauxdata)
     ewvirial(1:9) = auxdata(1:9)
     eterm(ewksum) = auxdata(10)
     eterm(ewqcor) = auxdata(11)
     eterm(ewutil) = auxdata(12)
     nauxdata = 12
#if KEY_DOMDEC_GPU==1
     if (q_gpu .and. gpu_code_version==2) then
       eterm(ewself) = auxdata(13)
       nauxdata = 13
     end if
#endif
#if KEY_LJPME==1
     if(qljpme) then
       eterm(ljrec) = auxdata(14)
       nauxdata = 14
     endif
#endif
     call timer_stop(T_fcomm2)
     ! Read data from auxdata(13:)
     vpress(1:9) = zero
     call read_from_auxdata(nauxdata, auxdata, dmc_ener, dmc_rho, vpress)
     ! Add vpress to epress
     epress(vixx:vizz) = epress(vixx:vizz) + vpress(1:9)
     ! Add to eprop(viri)
     eprop(viri) = eprop(viri) + (vpress(1) + vpress(5) + vpress(9))/three
#if KEY_ADUMB==1
     if (numbr > 0 .and. mynod == 0) then
        !write(*,*) "energy: mynod, eterm(adumb), coum = ",mynod,eterm(adumb),coum
        !We record statistics and do wham on both the direct and reciprocal nodes.
        !It isn't very time consuming and this
        !avoids having to have a separate message pass just for this.
        IF(STONUM) THEN
           call UMSTAT(STFPUM,STUPUM,STSFUM,STHIUM,STEMUM, &
                HPCRRD,HPTOAV,HPCRMS1,HPCRMS2,HPTORM1,HPTORM2)
        ENDIF
     endif
#endif
     ! Correct the pressure due to the k-space sum.
     epress(vixx:vizz) = epress(vixx:vizz) + ewvirial(1:9)
     EPROP(VIRI) = EPROP(VIRI) + (EWVIRIAL(1)+EWVIRIAL(5)+EWVIRIAL(9))/THREE
  elseif (q_domdec .and. .not.q_split .and. qpme .and. ndirect > 1) then
     ! In this case direct nodes are also reciprocal nodes
     call comm_force_among_recip(recipforcex, recipforcey, recipforcez)
     call reduce_recip_forces(recipforcex, recipforcey, recipforcez, dx, dy, dz)
  endif
#endif /* (domdec)*/

#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()
#endif
#if KEY_DMCONS==1
  ! Write dmcons to file
  ! NOTE for domdec: this has to be done after recv_results_from_recip
  !                  because that is where the root node gets the values
  !                  of dmc_ener and dmc_rho
  if (qdmc) then
     call write_dmcons(dmc_ener, dmc_rho, qsecd)
  end if
#endif
#if KEY_CHEQ==1
  if(qcg)Call GComb(DCH,Natom)
#endif
#if KEY_DOMDEC==1
  if (.not.q_domdec) then
#endif
     TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
     !  CALL PSYNC()               !APH: this psync could be commented out
     TIMMER = ECLOCK()
#if KEY_DOMDEC==1
  endif
#endif
  !
  ! NOTES:
  !     Distributed vector broadcast of forces done in calling routines.
  !     Combine Hessian explicitly in calling routines if needed.
  !
#else /*  (paramain)*/
#if KEY_DMCONS==1
  ! Write dmcons to file
  if (qdmc) then
     call write_dmcons(dmc_ener, dmc_rho, qsecd)
  end if
#endif
  ! ---------------------
  ! H Kamberaj (MD using scaling of torsions, November 2007)
#if KEY_TSALLIS==1
  IF (QTTSALL) THEN
     DO I=1,NATOM
        DX(I)=DX(I) + TSFACT * TDX(I)
        DY(I)=DY(I) + TSFACT * TDY(I)
        DZ(I)=DZ(I) + TSFACT * TDZ(I)
     ENDDO
  ENDIF
#endif
  !--------------------
#endif /* (paramain)*/
  ! Free memory Space
#if KEY_TSALLIS==1
  IF (QTTSALL) THEN
     if (prnlev .ge. 6) &
          write(outu,'("Free the allocated memory ofTDX,TDY,TDX")')
     deallocate(tdx, stat=ierror)
     if (ierror /= 0) &
          call wrndie(-4,'<ENERGY>', 'unable to deallocate TDX')
     deallocate(tdy, stat=ierror)
     if (ierror /= 0) &
          call wrndie(-4,'<ENERGY>', 'unable to deallocate TDY')
     deallocate(tdz, stat=ierror)
     if (ierror /= 0) &
          call wrndie(-4,'<ENERGY>', 'unable to deallocate TDZ')
  ENDIF
#endif

#if KEY_CHEQ==1 /*cheq0*/
  IF(QCG) THEN
     !...##IF PARALLEL
     !         IF (MYNOD.EQ.0) THEN
     !...##ENDIF
     IF (QGenBorn .OR. QGBMV) CALL GBFONCHEQ(X,Y,Z)
     CALL QNORM(DCH,GQT4P)
     IF (QCGMINE.AND.QSECD) THEN
        SUMDCH=0.0
        DO I=1,NATOM
           SUMDCH=SUMDCH+DCH(I)*DCH(I)
        ENDDO
        CALL DDCG(BNBND,NATOM,CG,X,Y,Z)
     ENDIF
     !...##IF PARALLEL
     !       ENDIF
     !...##ENDIF
  ENDIF
#endif /* (cheq0)*/
#if KEY_DOMDEC==1
  ! DOMDEC does not currently support IMOVE. This has to be implemented more
  ! efficiently before DOMDEC should support it. APH 10/21/2014
  if (.not.q_domdec) then
#endif
  !---------------------------------------------------------------
  ! . Remove forces on fixed atoms
     if(.not.NOFORC) then
        DO I=1,NATOM
           IF(IMOVE(I).GT.0) THEN
              DX(I)=ZERO
              DY(I)=ZERO
              DZ(I)=ZERO
#if KEY_FOURD==1
              IF(DIM4) DFDIM(I)=ZERO
#endif
           ENDIF
        ENDDO
     endif
#if KEY_DOMDEC==1
  endif
#endif
#if KEY_CHEQ==1
  IF (QNOCO) THEN
     CALL QNOCOD(DX,DY,DZ,NATOM)
  ENDIF
#endif
  !
  !---------------------------------------------------------------
#if KEY_TSM==1
  !     Now add the back atoms forces to the piggy atom
  !     and zero the back atom forces.
  IF(QTSM.AND.PIGSET) CALL PIGFSET(DX,DY,DZ)
#endif /*  TSM*/
  ! shf - moved this below fixed atom force removal 1/92
  !
  !---------------------------------------------------------------
  ! . Final virial computation calculation.
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('virtot')
#endif
  CALL VIRTOT(EPRESS(VIXX:VIZZ),EPRESS(VEXX:VEZZ),NATOM,X,Y,Z,DX,DY,DZ)
#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()
#endif
  !
  IF(NTRANS.GT.0 .AND. XTLTYP.NE.' ') &
       CALL LATTFD(XTLTYP,XTLABC,EPRESS(VEXX:VEZZ),DXTL,XTLREF)
  !
  !---------------------------------------------------------------
#if KEY_MTS==1 /*mts_reset*/
  ENE1=.FALSE.
  ENE2=.FALSE.
  ENE3=.FALSE.
#endif /* (mts_reset)*/
#if KEY_ZEROM==1 /*zmod ener*/
  if(QZMOD) then
     ECSUM = ZERO  ! initialize 24-Jun-2014 bp
     DO I = 1,LENENT
       ECSUM = ECSUM + ETERM(I)
     ENDDO
#if KEY_PARALLEL==1 /* para zmod */
     TEMP0(1) = ECSUM
     call GCOMB(TEMP0,1)
     EPROP(EPOT)=TEMP0(1)
#else              /* para zmod */   !serial code
     EPROP(EPOT)=ECSUM
#endif             /* para zmod */
  else                             !if not in zmod, do the usual
#endif          /*zmod ener*/
  !---------------------------------------------------------------
  !
#if KEY_PARALLEL==1 /*paraend*/
  ! Process more code for parallelization
  call timer_start(T_ecomm)
#if KEY_DOMDEC==1
  if (.not.q_domdec) then
#endif
     TMERI(TIMINTE) =TMERI(TIMINTE) + ECLOCK()-TIMMER
     TIMMER = ECLOCK()
#if KEY_DOMDEC==1
  endif
#endif
     !
     ! Pack assorted results into a single array for efficiency
     ! in using the global sum operation.
     !
     !      VIRI    - internal virial
     !      VIRE    - external virial
     !      VIRKE   - virial energy ( <f|r> )
     !
     IPT=1
     ALLWRK(IPT)=EPROP(VIRI)
     IPT=IPT+1
     ALLWRK(IPT)=EPROP(VIRE)
     IPT=IPT+1
     ALLWRK(IPT)=EPROP(VIRKE)
#if KEY_CHEQ==1
     IF(QCG) THEN
        IPT=IPT+1
        ALLWRK(IPT)=EPROP(CGPOT)
     ENDIF !GG: Not called in MSLD
#endif
     DO I=1,LENENT
        IPT=IPT+1
        ALLWRK(IPT)=ETERM(I)
     ENDDO
     !.ab.Pack&broadcast HybH contribs.
#if KEY_BLOCK==1
     IF (QHYBH) THEN  !GG: Not called in MSLD
        DO I=1,LENENT
           IPT=IPT+1
           ALLWRK(IPT)=ETERMR(I)
        ENDDO
        DO I=1,LENENT
           IPT=IPT+1
           ALLWRK(IPT)=ETERMP(I)
        ENDDO
     ENDIF
#endif
     !.ab.
! Changed by Lazaridis in May23. Only EPRESS(1:VIZZ) are calculated here.
!    DO I=1,LENENV
     DO I=1,VIZZ
        IPT=IPT+1
        ALLWRK(IPT)=EPRESS(I)
     ENDDO
     DO I=1,6
        IPT=IPT+1
        ALLWRK(IPT)=DXTL(I)
     ENDDO
     !
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('gcomb')
#endif
     CALL GCOMB(ALLWRK,IPT)  !GG: GCOMB=Global Combine Energy
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif
  !
     IF(QECONT)CALL GCOMB(ECONT,NATOM)    ! jwchu
     !
     IPT=1
     EPROP(VIRI)=ALLWRK(IPT)
     IPT=IPT+1
     EPROP(VIRE)=ALLWRK(IPT)
     IPT=IPT+1
     EPROP(VIRKE)=ALLWRK(IPT)
#if KEY_CHEQ==1
     IF(QCG) THEN  !GG: Not called in MSLD
        IPT=IPT+1
        EPROP(CGPOT)=ALLWRK(IPT)
     ENDIF
#endif
     DO I=1,LENENT
        IPT=IPT+1
#if KEY_VIBPARA==0
        ETERM(I)=ALLWRK(IPT)
#else /**/
        ETERM(I)=ALLWRK(IPT)/numnod
#endif
     ENDDO
     !.ab.
#if KEY_BLOCK==1
     IF (QHYBH) THEN   !GG: Not called in MSLD
#if KEY_VIBPARA==1
        CALL WRNDIE(-5,'<ENERGY>','HYBH and VIBPARA incompatible.')
#endif
        DO I=1,LENENT
           IPT=IPT+1
           ETERMR(I)=ALLWRK(IPT)
        ENDDO
        DO I=1,LENENT
           IPT=IPT+1
           ETERMP(I)=ALLWRK(IPT)
        ENDDO
     ENDIF
#endif
     !.ab.
!see above, Lazaridis, May23
!    DO I=1,LENENV
     DO I=1,VIZZ
        IPT=IPT+1
        EPRESS(I)=ALLWRK(IPT)
     ENDDO
     DO I=1,6
        IPT=IPT+1
        DXTL(I)=ALLWRK(IPT)
     ENDDO
  !
  ! timing stuff
#if KEY_DOMDEC==1
  if (.not.q_domdec) then
#endif
     TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
#if KEY_DOMDEC==1
  endif
#endif
  call timer_stop( T_ecomm )
#endif /* (paraend)*/
#if KEY_MSCALE==1
  !
  !     Get and scale the data from subsystems:
  MSCINIT=0
  IF(QMSCALE) CALL EMSCALE(MSCINIT,NATOM,LENENT, &
       X,Y,Z,DX,DY,DZ,ETERM,EPRESS,EPROP,QSECD,DD1,QDYNCALL)
#endif
  !
  !----------------------------------------------------------------------
  !   Correct virial and remove any forces on zero mass atom (only)
  !   NOTE for PARALLEL: HOLONOMF calls VIRSHK so this must be after
  !                      GCOMB(ALLWRK,...)
  IF(QHOLO) THEN
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('Holonomf')
#endif
     CALL HOLONOMF(DX,DY,DZ,X,Y,Z,.FALSE.,.TRUE.,QOK)
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif
  ENDIF
  !
  !----------------------------------------------------------------------
#if KEY_REPLICA==1
#if KEY_RPATH==1
  ! . REPLICA/PATH restraints. (with NEB options - jwchu)
  IF(QPATH) THEN
     IF((QPROPT).OR.(QPNEB.OR.QPTAU)) THEN
        ! Calculate path energy and forces .  Since we
        ! need to project all true forces out. jwchu 6/02
        CALL EPATH(ETERM(PRMS),ETERM(PANG),QETERM(PRMS), &
             QETERM(PANG),DX,DY,DZ,X,Y,Z,QSECD,ICALL)
     ENDIF
  ENDIF
#endif
#endif
  !
  !---------------------------------------------------------------
#if KEY_PARALLEL==1
  IF(MYNOD.EQ.0) THEN
#endif
     IF(QMICRO) THEN
#if KEY_QCHEM==1
        CALL MICROCORR(ETERM,LENENT,DX,DY,DZ)
#endif
     ENDIF
#if KEY_PARALLEL==1
  ENDIF
#endif
  IF(QMICRO) THEN
     QFRSTMICIT=.false.
  ENDIF
  !---------------------------------------------------------------
#if KEY_DOMDEC==1
!  if (.not.q_domdec .or. .not.QDYNCALL) then
#endif
#if KEY_MTS==1
  IF (.NOT. QTBMTS) THEN
#endif
     ! . Add all the terms to get the total energy.
     ECSUM = ZERO
     DO I = 1,LENENT
        ECSUM = ECSUM + ETERM(I)
     ENDDO
     !---------------------------------------------------------------
#if KEY_HQBM==1
     !    ep: HQBM implies an energy which is small but not neglible
     !        which needs to be considered in the total energy.
     !        Doing as follows I do not have to modify LENENT
     IF (HQBM) THEN
        ECSUM=ECSUM+EHQBM
     ENDIF
#endif
#if KEY_AFM==1
     IF (LAFM) THEN
        ECSUM=ECSUM+EAFM
     ENDIF
#endif
     !---------------------------------------------------------------
#if KEY_BLOCK==1
     if (qmld) call msld_add_potentialenergy(nblock,bixlam,ecsum)
        !/*ldm GG adds potential energy of Gbias and kbias into ECSUM*/
     if (qmld) call msld_add_restraints(nblock,x,y,z,dx,dy,dz,ecsum)
#endif
     EPROP(EPOT) = ECSUM
     !
#if KEY_MTS==1
  ENDIF
#endif
#if KEY_DOMDEC==1
!  endif
#endif

#if KEY_ZEROM==1   /* zmod ener */
!  endif
#endif             /* zmod ener */
  !
  !---------------------------------------------------------------
#if KEY_MULTCAN==1
  ! . Generalized (Multicanonical) restraint terms.
  IF(QMltCan) THEN
     CALL GenEnsemble(EPROP(EPOT),ETERM(CHARM),NATOM, &
          DX,DY,DZ)
     IF (TIMER.GT.1) &
          CALL WRTTIM('Generalized Ensemble restraint energy times:')
     !
     !
  ENDIF
  !
  !-----------------------------------------------------------------------
#endif
  !
  !---------------------------------------------------------------
  ! . Check energy contribution totals
  IF(LFAST.GE.0) THEN
     CONTINUE
#if KEY_MTS==1
  ELSE IF (QTBMTS) THEN
     CONTINUE
#endif
  ELSE IF(QECONT) THEN
#if KEY_PARALLEL==1
    IF(MYNOD.EQ.0) THEN
#endif
     ECSUM=ZERO
#if KEY_NBIPS==1
     IF(QIPS)ECSUM=EIPSSNB+EIPSSEL+ENBAIPS+EELAIPS
#endif
     DO I=1,NATOM
        ECSUM=ECSUM+ECONT(I)
     ENDDO
     ! Check if use econt with gamessuk   jwchu
#if KEY_RPATH==1
     cont_diff = ECSUM - EPROP(EPOT) + ETERM(PRMS) + ETERM(PANG)
#else /**/
     cont_diff = ECSUM - EPROP(EPOT)
#endif
#if KEY_GAMESSUK==1
     cont_tol = PT0005
#else /**/
     cont_tol = TENM5
#endif
     IF (abs(cont_diff) > cont_tol) THEN
        IF(WRNLEV.GE.2) WRITE(OUTU,432) ECSUM,EPROP(EPOT)
432     FORMAT(' ENERGY: Contribution totals do not match the', &
             ' energy.',/'         SUM=',F14.6,'  ENERGY=',F14.6)
     ENDIF
#if KEY_PARALLEL==1
    ENDIF
#endif
  ENDIF
  !---------------------------------------------------------------
  !
  IF (TIMER.GT.1) CALL WRTTIM('Backtransform and fix atom times:')
  !
  !---------------------------------------------------------------
  ! Do user accumulation of properties.
  CALL USRACM(NATOM,X,Y,Z,DX,DY,DZ,QECONT,ECONT,ICALL)
  IF (TIMER.GT.1) CALL WRTTIM('User accumulation times:')
  !
  !---------------------------------------------------------------
#if KEY_MMFF==1
  if(FFIELD.eq.MMFF .and. QSECD) then
     call addltou(DD1,LTSD,3*NATOM)
     call chmdealloc('energy.src','ENERGY','LTSD',3*NATOM*(3*NATOM+1)/2,crl=LTSD)
  endif
#endif

  !---------------------------------------------------------------

1967 continue  !Cc New PBLOCK

#if KEY_BLOCK==1 /*ldm*/
  if(qmld) call msld_add_force(nblock)
  IF(QLMC) &
       CALL SAVPOTEN(BIFLAM,BIPTNLAM, &
       BIELAM,NBLOCK,LSTRT)
  IF(QLDM) THEN
     call chmalloc('energy.src','ENERGY','lfrand',nblock,crl=lfrand)
     CALL SAVPOTEN(BIFLAM,BIPTNLAM,BIELAM,NBLOCK,LSTRT)
     IF(.NOT. RSTP .OR. MCRST)THEN
        IF(QTHETADM) THEN                             !New THETA-DYNAMICS
           THETAF2 = BIFLAM(2)                        !New THETA-DYNAMICS
           THETAF3 = BIFLAM(3)                        !New THETA-DYNAMICS
           CALL DLNGV2THETA(GAMMATHETA,FRANDTHETA)    !New THETA-DYNAMICS
           THETAF=-(THETAF3-THETAF2)*SIN(2.0*THETA)+FRANDTHETA !New THETA-
        ELSE                                          !New THETA-DYNAMICS
           CALL COMFLA(BIFLAM,NBLOCK, BIXLAM,BIVLAM,BIMLAM,BIELAM &
                ,LSTRT,IGAMMALD,BIBLAM,ILALDM,LFRAND)
        ENDIF                                         !New THETA-DYNAMICS
     ELSE
        CALL COMFLA2(BIFLAM,NBLOCK,BIXLAM,BIVLAM,BIMLAM,BIELAM &
             ,LMDCOEF,NRST,BFRST,LSTRT &
             ,IGAMMALD,BIBLAM,ILALDM,LFRAND)
     ENDIF
     call chmdealloc('energy.src','ENERGY','lfrand',nblock,crl=lfrand)
  ENDIF
  !     Add biasing potentials
  IF(QLDB) THEN
     CALL FVBIAS(NBIASV,IBVIDI,IBVIDJ,IBCLAS, &
          IRREUP,IRRLOW,IKBIAS,IPBIAS, &
          BIFLAM, BIXLAM)
  ENDIF
#if KEY_PBOUND==1
  IF(.NOT.QBOUN) THEN
#endif
     ! . Backtransform image forces
     IF(NTRANS.GT.0) THEN
        !
        ! . Calculate crystal lattice first derivatives.
        IF (RSTP) THEN
           CALL TRANSRSTP(X,Y,Z,ENVDX,ENVDY,ENVDZ, &
                NATOM,NTRANS,IMTRNS,BIMAG%IMATPT, &
                BIMAG%IMATTR,NOROT,NATIM,IMINV)
        ENDIF
     ENDIF
#if KEY_PBOUND==1
  ENDIF
#endif
#endif /* (ldm)*/
#if KEY_ENSEMBLE==1
  IF (QENSEXP) THEN
     CALL ENSEXPAVG(X, Y, Z, DX,DY,DZ,EPROP(EPOT),NATOM)
  else if (qevb) then
     call evb(x, y, z, dx, dy, dz, eprop(epot), natom)
  ENDIF
#endif
  !---------------------------------------------------------------
#if KEY_ESTATS==1 /*estats*/
  IF (LESTAT) THEN
     IENCAL = IENCAL + 1
     IF (IENCAL.LE.MXENCL) THEN !if not past end
        IF (IENCAL.GT.ESTRCL) THEN !if after start
           IF (MOD((IENCAL-ESTRCL),ESTMOD).EQ.0) THEN
              CALL ESTACAL
           ENDIF
           IF (LSTPRI) THEN
              IF ((MOD((IENCAL-ESTRCL),PRNMOD).EQ.0).AND.(PRNLEV &
                   .GT.2))  CALL PRNSTAT
           ENDIF
           IF (LPRNTU) THEN
              IF ((MOD((IENCAL-ESTRCL),IPRNMD).EQ.0).AND.(PRNLEV &
                   .GT.2))  CALL PRNSTATU
           ENDIF
           IF (LPRNTE) THEN
              IF ((MOD((IENCAL-ESTRCL),EPRNMD).EQ.0).AND.(PRNLEV &
                   .GT.2))  CALL PRNENERU
           ENDIF
           IF (IENCAL.EQ.MXENCL) CALL STOPSTAT
        ENDIF
     ENDIF
  ENDIF
#endif /* (estats)*/
  !---------------------------------------------------------------
#if KEY_SCCDFTB==1
  if(qsccb) then
     if(qstop.and.(.not.qpkac)) then
        dvdlb=0.0d0
        dvdlub=0.0d0
        dvdla=0.0d0
        dvdlp=0.0d0
        dvdlip=0.0d0
        dvdle=0.0d0
        dvdlv=0.0d0
     else if(qstop.and.qpkac.and.(idxstp.eq.1)) then
        !        just for dummy Hydrogen.
        dvdlub=0.0d0
        dvdlp=dvdlcp
        dvdlip=0.0d0
        dvdle=0.0d0
        dvdlv=0.0d0
     else if(qpkac.and.(idxstp.eq.2)) then
        dvdlb=0.0d0
        dvdlub=0.0d0
        dvdla=0.0d0
        dvdlp=0.0d0
        dvdlip=0.0d0
        dvdle=dvdlv1
        dvdlscc=0.0d0
     endif
  endif
#endif
  call timer_stop(T_energy)

#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()
#endif
  RETURN

contains

  subroutine bonded_energy()
    implicit none

#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('Bonded energy')
#endif

  !
  !================== INTERNAL ENERGY TERMS ==============================
  ! . Internal terms.
  ! . Use the FAST option if possible.
  IF (FASTER.GE.0) CALL FASTST(X,Y,Z,DX,DY,DZ,QSECD)
  QFARRAY=.TRUE.
  !
  !-------------------------------------------------------------------
  ! . Bond terms.
  call timer_start(  T_bond)
  IF(NBOND.GT.0.AND.QETERM(BOND)) THEN
#if KEY_SCCDFTB==1
     if(qsccb) idxbnd=0
#endif
     CALL EBONDC(ETERM(BOND),IB(J1:),JB(J1:), &
          ICB(J1:),NBONM,CBC,CBB,DX,DY,DZ,X,Y,Z, &
          DD1,IUPT,QSECD,NATOMT)
     IF (TIMER.GT.1) CALL WRTTIM('Bond energy times:')
  ! r328 Drude-Block move ENDIF below
  !     anisotropy for drudes by Ed Harder (Sept 2005)
  if (qdrude) then         ! Added by APH 7/16/2014
       q_calc_aniso = .false.
#if KEY_DOMDEC==1
       if (q_domdec) then
          q_calc_aniso = .true.
       else
#endif
#if KEY_PARALLEL==1
          if (mynod.eq.0) then
             q_calc_aniso = .true.
          endif
#else
       q_calc_aniso = .true.
#endif
#if KEY_DOMDEC==1
       endif
#endif
       if (q_calc_aniso) then
          CALL EANISOTROPY(EANIS,X,Y,Z,DX,DY,DZ,NATOM)
          ETERM(BOND)=ETERM(BOND)+EANIS
       endif
    endif
    ENDIF ! r328 Drude-Block
  call timer_stpstrt(  T_bond,T_angle)
  !-------------------------------------------------------------------
  ! . Angle terms.
  QOKANGLE = NTHETA > 0 .AND. QETERM(ANGLE)
#if KEY_CFF==1
  if (FFIELD == CFF) QOKANGLE = NTHETA > 0 .AND. (QETERM(ANGLE) &
       .OR. QETERM(BNDBND) .OR. QETERM(STRB) .OR. QETERM(STRSTR))
#endif
  IF (QOKANGLE) THEN
     CALL EANGLC(ETERM(ANGLE),IT(J3:),JT(J3:),KT(J3:), &
          ICT(J3:),NTHETM,DX,DY,DZ,X,Y,Z, &
          DD1,IUPT,QSECD, &
          IAC,NATOMT,QFARRAY, &
          ICB,ETERM(BNDBND),ETERM(STRB),ETERM(STRSTR))
     IF (TIMER.GT.1) CALL WRTTIM('Angle energy times:')
  ENDIF
  call timer_stpstrt(  T_angle,T_ephi)
  !--------------------------------------------------------------------
  ! . Urey-Bradley terms.
#if KEY_CFF==1
  if(FFIELD.ne.CFF) then
#endif
     IF(NTHETA.GT.0.AND.QETERM(UREYB)) THEN
#if KEY_SCCDFTB==1
        if(qsccb) idxbnd=1
#endif
#if KEY_ACTBOND==1
        QUREYBR = .TRUE.
#endif
#if KEY_BLOCK==1
        if (qmld) qldm_ureybr = .true.                /*ldm*/
#endif
#if KEY_DOMDEC==1
        if (q_domdec) then
           call eureyb_domdec(it(j3:), kt(j3:), ict(j3:), ctub, ctuc, eterm(ureyb))
        else
#endif
           CALL EBONDC(ETERM(UREYB),IT(J3:),KT(J3:), &
                ICT(J3:),NTHUBM,CTUC,CTUB, &
                DX,DY,DZ,X,Y,Z,DD1,IUPT,QSECD,NATOMT)
#if KEY_DOMDEC==1
        endif
#endif
        IF (TIMER.GT.1) CALL WRTTIM('Urey-Bradley energy times:')
#if KEY_ACTBOND==1
        QUREYBR = .FALSE.
#endif
#if KEY_BLOCK==1
        if (qmld) qldm_ureybr = .false.                /*ldm*/
#endif
     ENDIF
#if KEY_CFF==1
  endif
#endif
  !--------------------------------------------------------------------
#if KEY_MMFF==1
#if KEY_MTS==1
  IF(ENE2) THEN
#endif
     IF (FFIELD.EQ.MMFF) THEN
        !.ab.
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and MMFF incompatible.')
#endif
        !.ab.
        ! . Out-off-plane energy (MMFF)
        IF(NTHETA.GT.0.AND.QETERM(OOPL)) THEN
           IF (LFAST.GE.0) THEN
              CALL WRNDIE(-3,'<ENERGY>','No MMFF FAST code compiled.')
           ELSE
              CALL EOOPL(ETERM(OOPL),IT(J3),JT(J3),KT(J3),LTHETA(J3), &
                   icoop(J3),ntheta,OoplFC, &
                   X,Y,Z,DX,DY,DZ,LTSD,DERIVS, &
                   QECONT,ECONT,0, (/ 0 /))
           ENDIF
           IF (TIMER.GT.1) CALL WRTTIM('Out-of-Plane energy times:')
        ENDIF
        !
        !--------------------------------------------------------------------
        ! . Strech-Bend coupling energy (MMFF)
        IF(NTHETA.GT.0.AND.QETERM(STRB)) THEN
           IF (LFAST.GE.0) THEN
              CALL WRNDIE(-3,'<ENERGY>','No MMFF FAST code compiled.')
           ELSE
              CALL ESTRBND(ETERM(STRB),IT(J3),JT(J3),KT(J3),ICT(J3), &
                   ntheta,CTB, &
                   IB(J1),JB(J1),ICB(J1),CBB,StrbList,ICSTBN,STBNP, &
                   X,Y,Z,DX,DY,DZ,LTSD,DERIVS, &
                   QECONT,ECONT,0, (/ 0 /))
           ENDIF
           IF (TIMER.GT.1) CALL WRTTIM('Stretch-Bend energy times:')
        ENDIF
     ENDIF
#if KEY_MTS==1
  ENDIF
#endif
#endif /*  MMFF*/
  !
  !---------------------------------------------------------------------
  ! . Setup counts for dihedrals and improper diredrals for MTS.
#if KEY_MTS==1
  IF (QTBMTS) THEN
     NPHM=0
     NIMPHM=0
#if KEY_CMAP==1
     NPCTM=0
#endif
     IF(TBM4) THEN
        IF(ENE3) THEN
           NPHM=NPHS
           NIMPHM=NIMPHS
#if KEY_CMAP==1
           NPCTM=NPCTS
#endif
        ELSE
           NPHM=NPHF
           NIMPHM=NIMPHF
#if KEY_CMAP==1
           NPCTM=NPCTF
#endif
        ENDIF
     ENDIF
  ELSE
#endif
     NPHM=NPHI
     NIMPHM=NIMPHI
#if KEY_CMAP==1
     NPCTM=NCRTERM
#endif

#if KEY_MTS==1
  ENDIF
#endif
  !
  !
  !---------------------------------------------------------------------
  ! . Proper dihedral terms.
  IF(NPHM.GT.0.AND.QETERM(DIHE)) THEN
#if KEY_FOURD==0
     QDIM4=.FALSE.
#endif
#if KEY_FOURD==1
     QDIM4=(DIM4ON(DIHE).EQ.1)
#endif
#if KEY_SCCDFTB==1
     if(qsccb) idxphi=0
#endif
     !.ab. HybH. (see AB.J.Comp.Chem2004,25,985-993).
     !.ab.Note on HybH: Not modifications on Bonds,Angle&UB.
     !.ab.Real modifications start here.
     !.ab. Currently only fastscalar routine, and PME for Ewald
     !.ab. supported. Contact A.Blondel.
#if KEY_BLOCK==1
     QTEST=.FALSE.
#if KEY_FOURD==1
     QTEST=QDIM4
#endif
#if KEY_CFF==1
     QTEST=(QTEST.OR.(FFIELD.EQ.CFF))
#endif
     IF ((QHYBH).AND.((LFAST.LT.0).OR.QTEST)) &
          CALL WRNDIE(-5,'<ENERGY>', &
          'HYBH only suported for fastscalar, see code.')
     IF ((QHYBH).AND.(LEWALD.AND..NOT.QPME)) &
          CALL WRNDIE(-5,'<ENERGY>', &
          'HYBH only suported for PMEwald, see code.')
     !.ab. IHYBH used to assign output for dH/dlambda...
     IF (QHYBH) IHYBH=DIHE
#endif
     !.ab.
     CALL EPHIC(ETERM(DIHE),IP,JP,KP,LP,ICP,NPHM, &
          DX,DY,DZ,X,Y,Z,DD1,IUPT,QSECD,NATOMT, &
          QDIM4,QFARRAY,IB,ICB,ICT,ETERM(BBT),ETERM(BNDTW), &
          ETERM(EBST),ETERM(MBST),ETERM(SST) &
          )
     IF (TIMER.GT.1) CALL WRTTIM('Proper dihedral energy times:')
  ENDIF
  !---------------------------------------------------------------------
  ! . Improper dihedral terms.
  IF(NIMPHM.GT.0.AND.QETERM(IMDIHE)) THEN
#if KEY_SCCDFTB==1
     if(qsccb) idxphi=1
#endif
     !.ab. output for dH/dlambda...
#if KEY_BLOCK==1
     IF (QHYBH) IHYBH=IMDIHE
#endif
     !.ab.
     CALL EIMPHIC(ETERM(IMDIHE),IM,JM,KM,LM,ICI,NIMPHM, &
          DX,DY,DZ,X,Y,Z,DD1,IUPT,QSECD,NATOMT, &
          QFARRAY)
     IF(TIMER.GT.1)CALL WRTTIM('Improper dihedral energy times:')
  ENDIF
  !
#if KEY_CMAP==1
  !---------------------------------------------------------------------
  ! . Crossterms
  IF(NPCTM.GT.0.AND.QETERM(CMAP)) THEN
     !.ab. output for dH/dlambda... What about CMAP....
     !.ab. (actually nothing done !).
#if KEY_BLOCK==1
     IF (QHYBH) IHYBH=CMAP
#endif
     !.ab.
     CALL ECMAPC(ETERM(CMAP),I1CT,J1CT,K1CT,L1CT, &
          I2CT,J2CT,K2CT,L2CT, &
          ICCT,NPCTM, &
          DX,DY,DZ,X,Y,Z,DD1,IUPT,QSECD, &
          NATOMT,QFARRAY &
          )
     IF(TIMER.GT.1)CALL WRTTIM('Cross-term energy times:')
  ENDIF

  !
  ! ----------------------------------------------------------------------
  ! H Kamberaj (March 2007)
#if KEY_TSALLIS==1
  IF (QTTSALL) THEN
     ! Add bias part of energy to total energy
     IF (NPHM .GT. 0 .AND. QETERM(DIHE) ) THEN
        ETERM(DIHE)=ETERM(DIHE)+EBTORS(1)
     ENDIF
#if KEY_CMAP==1
     IF (NPCTM .GT. 0 .AND. QETERM(CMAP)) THEN
        ETERM(CMAP)=ETERM(CMAP)+EBTORS(2)
     ENDIF
#endif
#if KEY_PARALLEL==1
     CALL IGCOMB(TSNDF,1)
     CALL GCOMB(EBTORS,2)
#endif
     IF (PRNLEV .GT. 6) THEN
        WRITE(OUTU,'("Tsallis degrees of freedom:  ",I5)') &
             TSNDF
     ENDIF
     IF ((ONE-TSQ).GT.(ONE+ONE/(ONE*TSNDF))) THEN
        call wrndie(-4, '<ENERGY>', &
             'Decrease Tsallis Scaling Factor')
     ENDIF
     EBIAS = EBTORS(1) + EBTORS(2)
     IF (EBIAS .LE. TSEMIN ) THEN
        WRITE(OUTU,'("ebias, tsemin:  ", 2F12.5)') &
             ebias, tsemin
        CALL WRNDIE(-4,'<ENERGY>', &
             'TSALLIS DYNAMICS:ENERGY LESS THAN MIN VALUE')
     ENDIF
     etsbias=-((one-tsq)/(tsbeta*tsq))*log(one- &
          tsbeta*tsq*(ebias-tsemin))

     TSFACT=(ONE-TSQ)/(ONE-TSBETA*TSQ*(EBIAS-TSEMIN))
  ENDIF
#endif
#endif
  !-----------------------------------------------------------------------
#if KEY_TSM==1
  !---  Correct the internal energy terms for TSM.
  IF (QTSM) THEN
     ETERM(BOND)  =  ETERM(BOND) + TSMTRM(BOND)
     ETERM(ANGLE) =  ETERM(ANGLE) + TSMTRM(ANGLE)
     ETERM(DIHE)  =  ETERM(DIHE) + TSMTRM(DIHE)
     ETERM(IMDIHE) = ETERM(IMDIHE) + TSMTRM(IMDIHE)
#if KEY_CMAP==1
     ETERM(CMAP) = ETERM(CMAP) + TSMTRM(CMAP)
#endif
  ENDIF
#endif
#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
  if (.not.q_domdec) then
#endif
     ! get energies
     TMERI(TIMINTE) = TMERI(TIMINTE) + ECLOCK()-TIMMER
     TIMMER = ECLOCK()
#if KEY_DOMDEC==1
  endif
#endif
#endif

  call timer_stop(  T_ephi)
  call timer_stpstrt( T_inte,T_nonbon)
  !
  !---------------------------------------------------------------------
  ! ASPENER is moved here, before ENBOND because it calculates LAMD
  ! which is needed in the electrostatic calculation in IMM1
  QTEST=.FALSE.
#if KEY_ASPMEMB==1
  QTEST=SOLVMEMB_ENTRD
#endif
#if KEY_ASPENER==1
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
     !.ab.Not compatible. Would probably be interesting. Contact A.Blondel.
     !.ab.This should also take care of ASPMEMB (below)...
#if KEY_BLOCK==1
     QTEST=QTEST.OR.SOLVENT_ENTERED
     QTEST=QTEST.AND.QETERM(ASP)
     IF (QHYBH.AND.QTEST) CALL WRNDIE(-5,'<ENERGY>', &
          'HYBH and ASPENER incompatible, see code.')
#endif
     !.ab.
     IF(SOLVENT_ENTERED.AND.QETERM(ASP)) THEN

        CALL ASPENR(ETERM(ASP),X,Y,Z,DX,DY,DZ,QECONT,ECONT)
        IF (TIMER.GT.1) CALL WRTTIM('ASP surface energy times:')
     ELSEIF (DOEEF1.AND.QETERM(ASP)) THEN
        CALL EEF1EN(ETERM(ASP),X,Y,Z,DX,DY,DZ,QECONT,ECONT,1,NATOM, &
             1,NGRP,BNBND%JNBG,BNBND%INBLOG, &
             BNBND%INB14,BNBND%IBLO14,.FALSE., &
             QSECD, &
             DD1,IUPT)

        IF (TIMER.GT.1) CALL WRTTIM('EEF1 energy times:')

#if KEY_PBOUND==1
        if(.not.qboun) then
#endif
           IF(NTRANS.GT.0) THEN
              !  added "NATOM" to argument list - RJP
              CALL EEF1IM(EIMSLV,BNBND,BIMAG,BIMAG, &
                   IGPBS,IGPTYP,IAC,DX,DY,DZ,X,Y,Z,QECONT,ECONT,NATOM)
              ! EIMSLV is the amount of solvation free energy of primary atoms
              ! excluded by image atoms
              ETERM(ASP)= ETERM(ASP)+ EIMSLV
           ENDIF
#if KEY_PBOUND==1
        endif
#endif
     ENDIF
#if KEY_MTS==1
  ENDIF
#endif
#endif /*  ASPENER*/

#if KEY_DOMDEC_GPU==1
  if (q_gpu .and. gpu_code_version==2) then
     call ebonded_gpu()
  endif
#endif

#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()
#endif

    return
  end subroutine bonded_energy

  subroutine nonbonded_energy()
    implicit none
    logical q_do_ehyper

#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_start('Non-bonded energy')
#endif

  !================= NONBOND ENERGY TERMS ================================
  ! . Non-bond terms.
  IF (NATOM .GT. 0) THEN
     !-----------------------------------------------------------------------
     !
#if KEY_NBIPS==1 /*nbips*/
     ! See if IPS algorithm is in use. if so, calculate IPS of excluded atoms.
     !
     IF(QIPS) THEN
        !.ab.Not compatible. Would probably be interesting. Contact A.Blondel.
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and Isotropic Periodic Sums incompatible, see code.')
#endif
        !.ab.

        !WXW  IPS for excluded pairs
        !  3D periodic IPS calculation
        CALL timer_start(T_ips)
        CALL timer_start(T_ipsex)
        CALL  EIPSSYS(EIPSVDW,EIPSELE,ATFRST,ATLAST,NATOM,NATC,  &
             QETERM(VDW),QETERM(ELEC),LVIPS,LEIPS,LCONS,LDBEXP,  &
             BNBND%INB14,BNBND%IBLO14,     &
             CG,RSCLF,CNBA,CNBB,IAC,ITC,MAXCN,EPS,E14FAC,    &
             EPROP(VOLUME),CGSUM,CG2SUM,A2SUM,C2SUM,AIJSUM,CIJSUM,  &
             DX,DY,DZ,X,Y,Z,QECONT,ECONT,DD1,IUPT,QSECD,ICALL)
        !write(*,*)'EEXIPS:',mynod,atfrst,atlast,natom,eipsvdw,eipsele
        CALL timer_stop(T_ipsex)
        CALL timer_stop(T_ips)
        IF(TIMER.GT.1) CALL WRTTIM('Excluded pair IPS times:')
     ENDIF
#endif /* (nbips)*/

     !
     !-----------------------------------------------------------------------
#if KEY_MTS==1
     IF (QTBMTS) THEN
        !.ab.Not compatible. Would probably be interesting. Contact A.Blondel.
        !.ab.Not clear what should be done.
        !.ab.Probably sum dH/dlambda for main steps only....
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and MTS incompatible, see code.')
#endif
        !.ab.

        IF(ENE1.AND.(TBHY1.AND.(.NOT.TBHY2))) TBHY3=.TRUE.
        IF(ENE2) THEN
           IF((TBHY1.AND.TBHY2).OR.SLFG) TBHY3=.TRUE.
        ENDIF
     ENDIF
     !
     IF(TBHY3) THEN
        IF(SLFG) SLFG1=.TRUE.
        CALL ENBOND(ETERM(VDW),ETERM(ELEC),BNBNM1,1, &
             NATOM,CG,RSCLF,NGRP,IGPBS,IGPTYP,IAC,IACNB, &
             DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
             QETERM(EWEXCL),ETERM(EWEXCL), &
             DD1,IUPT,QSECD,QETERM(VDW),QETERM(ELEC), &
#if KEY_FLUCQ==1
             QFLUC,FQCFOR,     &
#endif
             QETERM(ST2),NST2,ETERM(ST2),QEXTND &
#if KEY_WCA==1
             ,LSOFTCORE0,SCCUTR0,WCA  &
#endif
#if KEY_LJPME==1
             ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
             )

        SLFG1=.FALSE.
        IF (TIMER.GT.1) CALL WRTTIM('MTS1: Non-bond energy times:')
     ELSE
        IF(ENE3) THEN
           IF(TBHY1.OR.SLFG) THEN
              IF(SLFG) SLFG2=.TRUE.
              CALL ENBOND(ETERM(VDW),ETERM(ELEC),BNBNM2,1, &
                   NATOM,CG,RSCLF,NGRP,IGPBS,IGPTYP,IAC,IACNB, &
                   DX,DY,DZ,X,Y,Z,QECONT,ECONT,.FALSE.,ZERO, &
                   DD1,IUPT,QSECD,QETERM(VDW),QETERM(ELEC), &
#if KEY_FLUCQ==1
                   QFLUC,FQCFOR,     &
#endif
                   QETERM(ST2),NST2,ETERM(ST2),QEXTND &
#if KEY_WCA==1
                   ,LSOFTCORE0,SCCUTR0,WCA  &
#endif
#if KEY_LJPME==1
                   ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
                   )

              SLFG2=.FALSE.
              IF (TIMER.GT.1) CALL WRTTIM('MTS2: Non-bond energy times:')
           ELSE
#endif
#if KEY_SCCDFTB==1
                 if(qsccb) idxnbd=0
#endif
                 !.ab.Here, we rely on the fact that ELEC=VDW+1 (It is checked).
                 !.ab. and other tests.
#if KEY_BLOCK==1
                 IF (QHYBH) IHYBH=VDW
#endif
                 !.ab.
#if KEY_GRAPE==1
                 qgpuene=.true.
#endif
                 CALL ENBOND(ETERM(VDW),ETERM(ELEC),BNBND,1, &  !GG: MSLD calls this subroutine
                      NATOM,CG,RSCLF,NGRP,IGPBS,IGPTYP,IAC,IACNB, &
                      DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
                      QETERM(EWEXCL),ETERM(EWEXCL), &
                      DD1,IUPT,QSECD,QETERM(VDW),QETERM(ELEC), &
#if KEY_FLUCQ==1
                      QFLUC,FQCFOR,     &
#endif
                      QETERM(ST2),NST2,ETERM(ST2),QEXTND &
#if KEY_WCA==1
                      ,LSOFTCORE0,SCCUTR0,WCA  &
#endif
#if KEY_LJPME==1
                      ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
                      )
                 IF (TIMER.GT.1) CALL WRTTIM('Non-bond energy times:')

#if KEY_MTS==1
           ENDIF
        ENDIF
     ENDIF
#endif
     !
     !       IF (TIMER.GT.1) CALL WRTTIM('Non-bond energy times:')
  ENDIF

  !
#if KEY_LOOKUP==1
  !-----------------------------------------------------------------------
  !LNI -  non-bond lookups
  IF(QLOOKUP)THEN
     !.ab.Not compatible. Contact A.Blondel for development.
#if KEY_BLOCK==1
     IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
          'HYBH and LOOKUP incompatible, see code.')
#endif
     !.ab.
     CALL ELOOKUP(ENBW,EELW, NWWO,&
          IWWO,IWOONBL,JWOONBL, &
          NATOM,IVUNBL,JVUNBL,IUUNBL,JUUNBL)
     IF(IWWENR.GE.0)THEN
        EWWEEL=EELW
        EWWENB=ENBW
     ELSE
        EELW=EWWEEL
        ENBW=EWWENB
     ENDIF
     ETERM(VDW)=ETERM(VDW)+ENBW
     ETERM(ELEC)=ETERM(ELEC)+EELW
     IF(TIMER.GT.1) CALL WRTTIM('Lookup energy times:')
  ENDIF
#endif
  if(qgopair) then
     call EGoPair(eterm(vdw),dx, dy, dz, x, y, z)
     if(timer > 1) call wrttim('Go pair energy times:')
  endif
  !-----------------------------------------------------------------------
#if KEY_TSM==1
  !---  Correct the primary nonbond energy terms for TSM.
  IF (QTSM) THEN
     ETERM(VDW)  =  ETERM(VDW) + TSMTRM(VDW)
     ETERM(ELEC) =  ETERM(ELEC) + TSMTRM(ELEC)
  ENDIF
#endif
  !-----------------------------------------------------------------------
#if KEY_NBIPS==1 /*nbips_add*/
  IF(QIPS) THEN                                          ! xw050623
     ETERM(VDW)  =  ETERM(VDW) + EIPSVDW
     ETERM(ELEC) =  ETERM(ELEC) + EIPSELE
  ENDIF
#endif /*  (nbips_add)*/
  !-----------------------------------------------------------------------
#if KEY_NOMISC==0
  ! . Extended electrostatics terms.
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
     IF(QEXTND .AND. QETERM(EXTNDE)) THEN
        !.ab.Not compatible. Has to define how to conceive TI with EXTE.
        !.ab. Contact A.Blondel for development.
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and Extended electrostatics incompatible, see code.')
#endif
        !.ab.
        CALL EXELEC(ETERM(EXTNDE), NATOM, X, Y, Z, DX, DY, DZ, &
             QXGRAD, WRNMXD, ATSX, ATSY, ATSZ, &
             ATPOT, ATFX, ATFY, &
             ATFZ,  ATGXX, ATGYY, &
             ATGZZ, ATGXY, ATGYZ, &
             ATGZX, OUTU)
        IF(TIMER.GT.1) CALL WRTTIM('Extended electrostatics times:')
     ENDIF
#if KEY_MTS==1
  ENDIF
#endif
#endif

  ! BEGIN Thole (G. Lamoureux)
     IF (QDRUDE .AND. QETERM(ELEC)) THEN
        !.ab.Not compatible. Has to define how to conceive TI with DRUDE.
        !.ab. Contact A.Blondel for development.
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and DRUDE incompatible, see code.')
#endif
        !.ab.

125     FORMAT(' ENERGY: Calling routine ',A)
        IF (PRNLEV.GT.6) WRITE(OUTU,125) 'THOLE_NBX'
#if KEY_DOMDEC==1
       if (q_domdec) then
          call thole_ex14_domdec(eterm(elec))
       else
#endif
          CALL THOLE_NBX(ISDRUDE,ALPHADP,THOLEI,CG, &
               ETERM(ELEC),BNBND%INB14,BNBND%IBLO14, &
               X,Y,Z,DX,DY,DZ,DD1,IUPT,QSECD,NATOM)
#if KEY_DOMDEC==1
       endif
#endif

        IF (TIMER.GT.1) CALL WRTTIM('Dipole-dipole energy times:')
            IF (PRNLEV.GT.6) WRITE(OUTU,125) 'THOLE_NB'
       CALL THOLE_NB(ISDRUDE,ALPHADP,THOLEI,CG, &
                1, NBTHOLP, &
                NBTHOL1, NBTHOL2, NBTHOL3, &
                NBTHOLXIJ, &
            ETERM(ELEC),X,Y,Z,DX,DY,DZ)
            IF (TIMER.GT.1) CALL WRTTIM('Thole-shielding energy times:')

!     Hyperpolarizability
       IF(QHYPER) THEN
          q_do_ehyper = .true.
#if KEY_DOMDEC==1
          if (.not.q_domdec) then
#endif
#if KEY_PARALLEL==1
             if (mynod == 0) then
                q_do_ehyper = .true.
             else
                q_do_ehyper = .false.
             endif
#endif
#if KEY_DOMDEC==1
          endif
#endif
          if (q_do_ehyper) then
             CALL EHYPER(NATOM,ISDRUDE,HORDER,KHYPER,RHYPER,ETERM(BOND), &
                  X,Y,Z,DX,DY,DZ)
          endif
       ENDIF
     ENDIF
  ! END Thole (G. Lamoureux)

  !-----------------------------------------------------------------------
  ! . Hydrogen-bond terms.
#if KEY_MTS==1
  IF(ENE2) THEN
#endif
#if KEY_PARALLEL==1
        IF(MYNOD.EQ.INODE(3)) THEN
#endif
           IF(NHB.GT.0.AND.QETERM(HBOND)) THEN
              !.ab.
#if KEY_BLOCK==1
              IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
                   'HYBH and HBOND incompatible.')
#endif
              CALL EHBND(ETERM(HBOND),ECALLS,X,Y,Z,DX,DY,DZ, &
                   QECONT,ECONT,DD1,IUPT,QSECD)
              IF (TIMER.GT.1) CALL WRTTIM('Hydrogen-bond energy times:')
              !             Correct the primary hbond energy terms for TSM.
#if KEY_TSM==1
              IF (QTSM) ETERM(HBOND)  =  ETERM(HBOND) + TSMTRM(HBOND)
#endif
           ENDIF
#if KEY_PARALLEL==1
        ENDIF
#endif
#if KEY_MTS==1
  ENDIF
#endif
  !
  call timer_stpstrt( T_nonbon,T_inte)
  call timer_start(  T_restr)
#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
  if (.not.q_domdec) then
#endif
     TMERI(TIMEXTE)=TMERI(TIMEXTE)+ECLOCK()-TIMMER
     TIMMER = ECLOCK()
#if KEY_DOMDEC==1
  endif
#endif
#endif

! begin rush
#if KEY_MTS==1
  IF(ENE2) THEN
#endif
     IF(   QRUSH .and. &
          ( QETERM( rushRepu ) .or. QETERM( rushPhob ) &
          .or. QETERM( rushHbnd ) .or. QETERM( rushBdon ) &
          .or. QETERM( rushBacc ) .or. QETERM( rushArom ) ) )THEN
#if KEY_PARALLEL==1
        if(NUMNOD .gt. 1) call WRNDIE(-4,'<ENERGY>',        &
#endif
#if KEY_PARALLEL==1
             'No parallel code for RUSH')
#endif
        !.ab.Not compatible. Could be done if was interesting. Contact A.Blondel..
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and RUSH incompatible, see code.')
#endif
        !.ab.
        call rush_energy(  &
             ETERM( rushRepu ),  ETERM( rushPhob ), &
             ETERM( rushHbnd ),  ETERM( rushBdon ), &
             ETERM( rushBacc ),  ETERM( rushArom ), &
             QETERM( rushRepu ), QETERM( rushPhob ), &
             QETERM( rushHbnd ), QETERM( rushBdon ), &
             QETERM( rushBacc ), QETERM( rushArom ), &
             X,  Y,  Z, &
             DX, DY, DZ, &
             NATOM, .false. )
#if KEY_MTS==1
     ENDIF
#endif
  ENDIF
! end rush

#if KEY_DOMDEC_GPU==1
  if (q_gpu) call range_stop()
#endif

    return
  end subroutine nonbonded_energy

    subroutine nonbonded_image_energy()
    implicit none

  !================= IMAGE ENERGY TERMS ==================================
  !
#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
  if (.not.q_domdec) then
#endif
     TMERI(TIMINTE)=TMERI(TIMINTE)+ECLOCK()-TIMMER
     TIMMER = ECLOCK()
#if KEY_DOMDEC==1
  endif
#endif
#endif
#if KEY_DOMDEC==1
  if (.not.q_domdec) then
#endif
#if KEY_PBOUND==1
  if(.not.qboun) then
#endif
           ! . Image terms.
           IF(NTRANS .GT. 0) THEN
              !.ab.HYBH: Apparently compatible... Term assignment done in EIMAGE.
              CALL EIMAGE(X,Y,Z,DX,DY,DZ,QECONT,ECONT,QSECD)

              !
#if KEY_MTS==1
              IF(TBHY3) THEN
                 IF(SLFG) SLFG1=.TRUE.
                 CALL EIMNBD(ETERM(IMVDW),ETERM(IMELEC),BNBNM1,BIMAG, &
                      BIMTS1,1,NATIM,CG,RSCLF,NIMGRP, &
                      IGPBS,IGPTYP,IAC,IACNB,DX,DY,DZ,X,Y,Z, &
                      QECONT,ECONT,QETERM(EWEXCL),ETERM(EWEXCL), &
                      QETERM(IMVDW),QETERM(IMELEC), &
#if KEY_FLUCQ==1
                      QFLUC,FQCFOR,    &
#endif
                      QETERM(IMST2),NST2,ETERM(IMST2) &
#if KEY_WCA==1
                      ,WCA          &
#endif
#if KEY_LJPME==1
                      ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
                      )
                 SLFG1=.FALSE.
              ELSE
                 !
                 IF(ENE3) THEN
                    IF(TBHY1.OR.SLFG) THEN
                       IF(SLFG) SLFG2=.TRUE.
                       CALL EIMNBD(ETERM(IMVDW),ETERM(IMELEC),BNBNM2,BIMAG, &
                            BIMTS2,1,NATIM,CG,RSCLF,NIMGRP, &
                            IGPBS,IGPTYP,IAC,IACNB,DX,DY,DZ,X,Y,Z, &
                            QECONT,ECONT,.FALSE.,ZERO,QETERM(IMVDW), &
                            QETERM(IMELEC), &
#if KEY_FLUCQ==1
                            QFLUC,FQCFOR,    &
#endif
                            QETERM(IMST2),NST2,ETERM(IMST2) &
#if KEY_WCA==1
                            ,WCA          &
#endif
#if KEY_LJPME==1
                            ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
                            )
                       SLFG2=.FALSE.
                    ELSE
#endif
#if KEY_SCCDFTB==1
                       if(qsccb) idxnbd=1
#endif
                       !.ab.Here, we rely on the fact that IMELEC=IMVDW+1 (It is checked).
                       !.ab.MTS not supported, but tested above.
#if KEY_BLOCK==1
                       IF (QHYBH) IHYBH=IMVDW
                       !.ab.
#endif
                       CALL EIMNBD(ETERM(IMVDW),ETERM(IMELEC),BNBND,BIMAG, &
                            BIMAG,1,NATIM,CG,RSCLF,NIMGRP, &
                            IGPBS,IGPTYP,IAC,IACNB,DX,DY,DZ,X,Y,Z, &
                            QECONT,ECONT,QETERM(EWEXCL),ETERM(EWEXCL), &
                            QETERM(IMVDW),QETERM(IMELEC), &
#if KEY_FLUCQ==1
                            QFLUC,FQCFOR,    &
#endif
                            QETERM(IMST2),NST2,ETERM(IMST2) &
#if KEY_WCA==1
                            ,WCA          &
#endif
#if KEY_LJPME==1
                            ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
                            )

#if KEY_MTS==1
                    ENDIF
                 ENDIF
              ENDIF
#endif
              IF (TIMER.GT.1) CALL WRTTIM('Image energy times:')
           ENDIF
#if KEY_PBOUND==1
  endif
#endif
#if KEY_DOMDEC==1
  endif
#endif
  !
#if KEY_TSM==1
  !---  shf 1/6/92 thermodynamic simulation method for image atoms
  !---  and memory freeing.
  !---  Correct the primary image non-bond energy terms for TSM.
  !---  Gets called even if images are not being used.
  IF (QTSM) THEN
     CALL TSME(X,Y,Z,DX,DY,DZ,.FALSE.)
     IF (TIMER.GT.1) CALL WRTTIM('TSM image energy times:')
     ETERM(IMVDW)  =  ETERM(IMVDW) + TSMTRM(IMVDW)
     ETERM(IMELEC)  =  ETERM(IMELEC) + TSMTRM(IMELEC)
     ETERM(IMHBND)  =  ETERM(IMHBND) + TSMTRM(IMHBND)
  ENDIF
#endif /* TSM*/
  !
#if KEY_CHEQ==1
  !===================================================================
  !===================================================================

  IF(QCG.AND.QETERM(ELEC)) THEN
     !.ab.Would be great, but not developed. Contact A.Blondel.
#if KEY_BLOCK==1
     IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
          'HYBH and The CHarge EQuilibration &
          &Method incompatible, see code.')
#endif
     !.ab.
     CALL QBENER(NATOM,CG,X,Y,Z,DX,DY,DZ,BNBND%IBLO14, &
          BNBND%INB14,IAC,QSECD,LEWALD,LELEC)
     IF (TIMER.GT.1) CALL WRTTIM('QBENER energy times:')
  ENDIF

  !===================================================================
  !===================================================================
#endif

#if KEY_FLUCQ==1
  ! Fluctuating charge energy terms
  ! Calculates electronegativities from electrostatic potential
  ! (previously calculated by QM/MM and nonbond routines) and
  ! adds in internal energy terms
  IF (QFLUC) THEN
     IF (QETERM(FQPOL)) THEN
        !.ab.
#if KEY_BLOCK==1
        IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
             'HYBH and FLUCQ incompatible.')
#endif
        !.ab.
        CALL FQENER(ETERM(FQPOL),X,Y,Z,QETERM(FQPOL))
     ENDIF
  ENDIF
#endif /*  FLUCQ*/
  !-----------------------------------------------------------------------
#if KEY_OVERLAP==1
#if KEY_PARALLEL==1
  IF(MYNOD.EQ.0) THEN
#endif
     !.ab.Note: idem CDIHE. Contact A.Blondel.
#if KEY_BLOCK==1
     IF((QHYBH).AND.(QETERM(QOVLAP).AND.QOLAP)) &
          CALL WRNDIE(-5,'<ENERGY>', &
          'HYBH and Overlap Molecular Similarityincompatible, see code.')
#endif
     !.ab.
     IF(QETERM(QOVLAP).AND.QOLAP)CALL OLAPENER(ETERM(QOVLAP),DX,DY,DZ)
#if KEY_PARALLEL==1
  ENDIF
#endif
#endif
  !-----------------------------------------------------------------------
#if KEY_ASPMEMB==1
#if KEY_MTS==1
  IF(ENE3) THEN
#endif
     IF(SOLVMEMB_ENTRD.AND.QETERM(ASP)) THEN
        CALL ASPENERMB(ETERM(ASP),X,Y,Z,DX,DY,DZ,QECONT,ECONT)
     ENDIF
#if KEY_MTS==1
  ENDIF
#endif
#endif /*  ASPMEMB*/
  !
#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
  if (.not.q_domdec) then
#endif
     TMERI(TIMEXTE)=TMERI(TIMEXTE)+ECLOCK()-TIMMER
     TIMMER = ECLOCK()
#if KEY_DOMDEC==1
  endif
#endif
#endif
  !-----------------------------------------------------------------------

    return
  end subroutine nonbonded_image_energy

END subroutine old_ENERGY
