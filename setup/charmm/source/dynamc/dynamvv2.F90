module dynvv2
  use chm_types
  use dimens_fcm
  implicit none

  character(len=*),private,parameter :: file_name = 'dynamvv2.src'
  real(chm_real) :: T_Drude
  real(chm_real),allocatable,private :: DXSAVE(:), DYSAVE(:), DZSAVE(:)

#if KEY_DYNVV2==1 /*dynvv2_main*/

contains

  SUBROUTINE DYNAMVV2( &
       VX,VY,VZ,VK,XNEW,YNEW,ZNEW, &
       XOLD,YOLD,ZOLD, &
       AMASS_,NPRIV,NDEGF,IGVOPT, &
       IMOVE_,ISKPR,FREEAT,NFREAT,NATOMX, &
       BNBND,BIMAG, &
       ISTART,ISTOP,IPRFRQ,IDYNPR,JHTEMP, &
       GAMMA,RFD,RFT,QAVER,NAVER,XAVE,YAVE,ZAVE, &
       QKUHEAD &
#if KEY_MTS==1
       ,DXFAST,DYFAST,DZFAST,DXMEDIUM,DYMEDIUM,DZMEDIUM &
#endif
       ,Xtemp,Ytemp,Ztemp &
       )
    !
    !     Wrapper for the velocity-Verlet algorithm VV2
    !
    !     The calling syntax is the same as DYNAMVV.
    !

    use vector
    use number
    use reawri
#if KEY_BLOCK==1
    use block_fcm, only : prhybh
#endif
    use ctitla
    use cnst_fcm
    use contrl
    use coord
    use cvio
    use deriv
    use dynio
    use energym
    use averfluc
    use avfl_ucell
    use consta ! for TIMFAC
    use shake
    use stream
    use icfix
    use icpert
    use nose_mod ! for QNOSE,SNH,SNHV,SNHF,RTMPR,SQM,NOBL,INLCKP,NDGN,IUNOS,NSNOS
#if KEY_PERT==1
    use pert
    use pshake
#endif
#if KEY_AXD==1
    use axd_module
#endif
#if KEY_ADUMB==1
    use umb
#endif
#if KEY_GAMUS==1
    use gamusmodule
#endif
    use parallel ! for JPBLOCK,MYNOD
    ! FBS MOD ********************************************
    use dmcons ! for DSTEP
    use rgym ! for DSTEPRF
    ! END FBS MOD ****************************************
    use tbmts ! for NMTS1,NMTS2,NMTS,IMTS
    use image
    use fourdm
    use psf ! for IMOVE,AMASS
    use holonom,only:holonoma
    use clcg_mod, only: bmgaus
    use machutil,only:eclock,die
    use heurist,only:updeci
    use prssre
!QC_Ito_UW0616
#if KEY_RXNCOR==1
  use rxncom, only: UMBMDSTEP
#endif

#if KEY_SGLD==1
    use sgld,only:sgfavg,sgldg,sgmdg,prntsg,qsgld,qsgmd, sgld_ave_params
#endif
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,natoml,atoml
#endif
#if KEY_DIMS==1
    use new_timer, only: timer_start, timer_stop, T_DIMS,T_DIMNM,T_DIMCOMB
    use dims
#endif
#if KEY_DOMDEC==1
    use domdec_d2d_comm,only:transfer_coord,copy_to_all
    use domdec_local,only:update_local_coord
#endif
#if KEY_DIMS==1
    use dims
    use coord, only: WMAIN
    use param_store, only: set_param
#endif
#if KEY_REPDSTR==1
    use repdstr,only:qrepdstr
#endif
    !
    !     Arguments
    !
    !
    real(chm_real) VX(*),VY(*),VZ(*)  ! I/O  Velocities of atoms
    real(chm_real) VK(*)              ! I/O  Accumulated kinetic energies of atoms
    real(chm_real) XNEW(*), YNEW(*), ZNEW(*)
    real(chm_real) XOLD(*), YOLD(*), ZOLD(*)
    real(chm_real) AMASS_(*),ISKPR(*)          ! INPUT  Masses of atoms
    INTEGER NPRIV             ! Restart file was written at step NPRIV
    INTEGER NDEGF
    INTEGER IGVOPT            ! 2 for start or restart, 3 for a continuation
    INTEGER IMOVE_(*)
    INTEGER FREEAT(*),NFREAT
    INTEGER NATOMX            ! Total number of atoms (including images?)
    !!      INTEGER BNBND(*)
    !!      INTEGER BIMAG(*)
    type(nonbondDataStructure) BNBND
    type(imageDataStructure) BIMAG

    INTEGER ISTART,ISTOP,IPRFRQ
    INTEGER IDYNPR            ! ???
    real(chm_real) GAMMA(*),RFD(*),RFT(*)
    LOGICAL QAVER             ! Flag for requesting average positions
    LOGICAL QKUHEAD
    INTEGER NAVER
    real(chm_real) XAVE(*),YAVE(*),ZAVE(*)
    ! Average positions (at OUTPUT)
#if KEY_MTS==1
    real(chm_real) DXFAST(*),DYFAST(*),DZFAST(*)
    ! Fast forces (originally: XMI,YMI,ZMI)
    real(chm_real) DXMEDIUM(*),DYMEDIUM(*),DZMEDIUM(*)
    ! Medium forces (originally: XMM,YMM,ZMM)
#endif
    real(chm_real) Xtemp(*),Ytemp(*),Ztemp(*)
    ! Workspace (originally: XLD,YLD,ZLD)
    real(chm_real) JHTEMP
    !
    !     Global variables
    !
    !
    !     Local variables
    !
    integer ATFRST, ATLAST
    integer I, TI
    integer ia

    real(chm_real)  kin2              ! Two times the kinetic energy (mv2)
    SAVE    kin2
    real(chm_real)  kin2_nh(MAXNOS)   ! "kin2" for each thermostat
    SAVE    kin2_nh
    real(chm_real)  temp_nh(MAXNOS)   ! temperature for each thermostat

    real(chm_real)  KINE_CP           ! Kinetic energy associated to the barostat
    real(chm_real)  POT_CP            ! Potential energy associated to the barostat
    SAVE    KINE_CP, POT_CP
    real(chm_real)  RVAL, GNORM       ! Gradient of the crystal parameters
    real(chm_real)  QK1               ! Kinetic energy of the thermostats
    real(chm_real)  QK2               ! Potential energy of the thermostats

    real(chm_real),allocatable,dimension(:) :: DXc,DYc,DZc
    real(chm_real),allocatable,dimension(:) :: Xsave,Ysave,Zsave
    real(chm_real),allocatable,dimension(:) :: VXsave,VYsave,VZsave
    real(chm_real),allocatable,dimension(:) :: lagrange, dmass
    integer kcorr,kcount

    integer NDXlang           ! stack size for Langevin forces
    real(chm_real),allocatable,dimension(:) :: DXlang,DYlang,DZlang
    real(chm_real),allocatable,dimension(:) :: VXlang,VYlang,VZlang
    real(chm_real)  NHGAMMA_TEMP,NHGAMMAD_TEMP
    !     Functions

    real(chm_real)  RMSDrattle        ! to monitor the convergence of RATTLE/ROLL

    real(chm_real)  SNHroll(MAXNOS),SNHVroll(MAXNOS)
    SAVE    SNHroll,SNHVroll
    real(chm_real)  ETA_CProll(3,3),ZETA_CProll(3,3)
    SAVE    ETA_CProll,ZETA_CProll
    real(chm_real)  sf(6),XTLroll(6)
    SAVE    XTLroll
    real(chm_real)  VENEroll(9),TR_VENE_ROLL
    SAVE    VENEroll,TR_VENE_ROLL
    real(chm_real)  prod(3,3),trZETA2

    !     SCF dipoles
    logical SCFdipoles
    !CC      temporary bugfix... need to pass array for variable dimension.
    !CC      logical useThermostat(NOBL) !(MAXNOS)
    logical useThermostat(MAXNOS)
    !CC
    !      integer dmass             ! Stack for temporary Drude mass storage

    !     Printing, monitoring, and averaging
    integer ISTEP
    SAVE    ISTEP
    integer IST1
    real(chm_real)  TIME
    real(chm_real)  TOTEPR
    data    TOTEPR /-9999.0D0/
    real(chm_real)  DNUM,DNM1,DNP1,DNPM1
    integer IPRFRQcounter
    integer NUMSTP
    real(chm_real)  MT_               ! Total mass of drude pair
#if KEY_PARALLEL==1
    real(chm_real)  TIMMER
#endif
    logical, parameter :: debug = .false.
    logical QOK
#if KEY_DIMS==1
    LOGICAL DIMSCARTON, QPRINT
    real(chm_real) OMSC, FACT
    integer counter

    dimscarton = .false. ! make sure this is always initialized
#endif

    !     Issue warnings for missing code
    !
    !C      CALL WRNDIE(-3,'<DYNAMVV2>','No parallel implementation')

#if KEY_TSM==1
    IF (QCFTI.or.QCFTM) then
       CALL WRNDIE(-3,'<DYNAMVV2>','TSM not implemented')
    ENDIF
#endif 

#if KEY_DOMDEC==1
  if (q_domdec) then
!     call wrndie(-5,'<dynamvv2>','DOMDEC not implemented on dynamvv2')
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
     ATLAST=NATOMX
#endif /* (parfmain)*/
#else /* (paramain)*/
     ATFRST=1
     ATLAST=NATOMX
#endif /* (paramain)*/
#if KEY_DOMDEC==1
  endif
#endif

#if KEY_PARALLEL==1
    timmer=eclock()
#endif
    !     Allocate memory

    call chmalloc('dynamvv2.src','DYNAMVV2','DXc',NATOMX,crl=DXc)
    call chmalloc('dynamvv2.src','DYNAMVV2','DYc',NATOMX,crl=DYc)
    call chmalloc('dynamvv2.src','DYNAMVV2','DZc',NATOMX,crl=DZc)
    DXc(1:NATOMX) = zero
    DYc(1:NATOMX) = zero
    DZc(1:NATOMX) = zero

    call chmalloc('dynamvv2.src','DYNAMVV2','Xsave',NATOMX,crl=Xsave)
    call chmalloc('dynamvv2.src','DYNAMVV2','Ysave',NATOMX,crl=Ysave)
    call chmalloc('dynamvv2.src','DYNAMVV2','Zsave',NATOMX,crl=Zsave)
    Xsave(1:NATOMX) = zero
    Ysave(1:NATOMX) = zero
    Zsave(1:NATOMX) = zero

    call chmalloc('dynamvv2.src','DYNAMVV2','VXsave',NATOMX,crl=VXsave)
    call chmalloc('dynamvv2.src','DYNAMVV2','VYsave',NATOMX,crl=VYsave)
    call chmalloc('dynamvv2.src','DYNAMVV2','VZsave',NATOMX,crl=VZsave)
    VXsave(1:NATOMX) = zero
    VYsave(1:NATOMX) = zero
    VZsave(1:NATOMX) = zero

    call chmalloc('dynamvv2.src','DYNAMVV2','lagrange',NCONST,crl=lagrange)
    lagrange(1:NCONST) = ZERO

    NDXlang = 0
    if (QNOSE) then
       do i = ATFRST,ATLAST
          ti = INLCKP(I)
          if (QNHLANG(ti)) NDXlang = NDXlang + 1
       enddo
       call chmalloc('dynamvv2.src','DYNAMVV2','DXlang',NATOMX,crl=DXlang)
       call chmalloc('dynamvv2.src','DYNAMVV2','DYlang',NATOMX,crl=DYlang)
       call chmalloc('dynamvv2.src','DYNAMVV2','DZlang',NATOMX,crl=DZlang)
       DXlang(1:NATOMX) = zero
       DYlang(1:NATOMX) = zero
       DZlang(1:NATOMX) = zero
       call chmalloc('dynamvv2.src','DYNAMVV2','VXlang',NATOMX,crl=VXlang)
       call chmalloc('dynamvv2.src','DYNAMVV2','VYlang',NATOMX,crl=VYlang)
       call chmalloc('dynamvv2.src','DYNAMVV2','VZlang',NATOMX,crl=VZlang)
       VXlang(1:NATOMX) = zero
       VYlang(1:NATOMX) = zero
       VZlang(1:NATOMX) = zero
    endif

    !
    !     Compute inertia parameters from timescales
    !
    if (QNOSE) then
       do ti = 1,NOBL
          if (SQM(ti) == ZERO) then
             SQM(ti) = NDGN(ti) * KBOLTZ*RTMPR(ti) &
                  * (NHTAU(ti)/TIMFAC)**2
          endif
          if (PRNLEV > 5) then
             write(outu,*) 'VV2 --- Q(',ti,') =',SQM(ti)
          endif
       enddo
    endif
    if (QPCON) then
       if (W_CP == ZERO) then
          do ti = 1,NOBL
             W_CP = W_CP + NDGN(ti) * KBOLTZ*RTMPR(ti) &
                  * (TAU_CP/TIMFAC)**2
          enddo
          if (QZONLY) W_CP = THIRD*W_CP ! because only z coordinates are scaled
          if (PRNLEV > 5) then
             write(outu,*) 'VV2 --- W =',W_CP
          endif
       endif
    endif

    !
    !     Decide if we compute SCF dipoles or not.
    !
    if (QNOSE) then
       do ti = 1,NOBL
          useThermostat(ti) = .true.
       enddo
    endif
    if (NDRUDE == 0) then
       SCFdipoles = .false.
    else
       if (.not.QNOSE) then
          SCFdipoles = .true.
       else
          !           We need SCF dipoles if all Drudes are associated
          !           to a thermostat at zero temperature.
          SCFdipoles = .false.
          do i = ATFRST,ATLAST
             if (ISDRUDE(i)) then
                ti = INLCKP(I)
                if (RTMPR(ti) < TENM8) then
                   SCFdipoles = .true.
                   useThermostat(ti) = .false.
                   RTMPR(ti) = ZERO
                endif
             endif
          enddo
       endif
    endif

    !     Don't use absolute thermostat if CM Langevin damping
    if (QCMLANGEVIN) then
       useThermostat(IABST) = .false.
    endif
    !     Some printing...
    if (PRNLEV > 5) then
       if (SCFdipoles) then
          write(outu,*) 'VV2 --- ', &
               'SCF Dipoles: TOLSCF =',TOLSCF,' MAXSCF =',MAXSCF
       endif
       do ti = 1,NOBL
          if (.not.useThermostat(ti)) then
             write(outu,*) 'VV2 --- ', &
                  'Thermostat',ti,' is deactivated.'
          endif
       enddo
    endif

    !     Give back nonzero mass of SCF particles
    if (SCFdipoles) then
       call chmalloc('dynamvv2.src','DYNAMVV2','dmass',NATOMX,crl=dmass)
       dmass(1:NATOMX) = amass_(1:natomx)
       do i = ATFRST,ATLAST
          if (ISDRUDE(i) .and. amass_(i) > ZERO) then
             if (PRNLEV > 5) then
                write(outu,*) 'VV2 --- ', &
                     'Mass',amass_(i),' :',i,' ->',i-1
             endif
             amass_(i-1) = amass_(i-1) + amass_(i)
             amass_(i) = ZERO
             if (IMOVE_(i-1)  ==  -1) amass_(i-1) = ZERO
          endif
          if (ISDRUDE(i)) IMOVE_(i) = -1
       enddo
    endif

    !
    !     Initialization
    !
    IF (ISTOP < ISTART) CALL DIE

    IST1 = ISTART-1

    IF (QAVER) THEN
       IF (NAVER == 0) THEN
          call ZeroArrays(xave, yave, zave, atfrst, atlast)
       ENDIF
    ENDIF
    if (RX1_CP(1,1) == ZERO) CALL IDENT33(RX1_CP)
    if (RX2_CP(1,1) == ZERO) CALL IDENT33(RX2_CP)
    if (RV1_CP(1,1) == ZERO) CALL IDENT33(RV1_CP)
    if (RV2_CP(1,1) == ZERO) CALL IDENT33(RV2_CP)

    !     Retreiving constraint forces from XNEW
    if (QHOLO) then
       ! For RESTART only?
       CALL CopyArrays( XNEW,YNEW,ZNEW,DXc,DYc,DZc,ATFRST,ATLAST )
    endif

    !     Start
    IF (IGVOPT == 2) THEN
       IF (QHOLO .OR. SCFdipoles) THEN
          CALL CopyArrays( X,Y,Z, XOLD,YOLD,ZOLD, &
               ATFRST,ATLAST )
       ENDIF
       !        SHAKE coordinates in case they don't fit constraints
       IF (QHOLO) THEN
          IF (debug) WRITE(OUTU,*) 'DEBUG -- SHAKE (start)'
          ! APH NOTE: For domdec, there's a danger here that SHAKE will not have all coordinates
          ! up-to-date after the copyarrays -call
          CALL HOLONOMA(X,Y,Z,XOLD,YOLD,ZOLD,.TRUE.,.TRUE.,QOK)
#if KEY_DOMDEC==1
          if (q_domdec) then
             call transfer_coord(x, y, z, .false.)
             call update_local_coord(x, y, z)
          else
#endif
#if KEY_PARALLEL==1
             TMERI(TIMDCNTRL)=TMERI(TIMDCNTRL)+ECLOCK()-TIMMER
             CALL PSYNC()
             TIMMER=ECLOCK()
             CALL VDGBR(X,Y,Z,0)
             TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
             CALL PSYNC()
             TIMMER=ECLOCK()
#endif
#if KEY_DOMDEC==1
          endif
#endif
       ENDIF

       !        Readjust SCF dipoles
       IF (SCFdipoles) THEN
          IF (debug) WRITE(OUTU,*) 'DEBUG -- SCF dipoles (start)'
          CALL OptimizeDrude( TOLSCF,MAXSCF, &
               XOLD,YOLD,ZOLD, VX,VY,VZ, BNBND,BIMAG, &
               DX,DY,DZ, .false. )
#if KEY_DOMDEC==1
          if (q_domdec) then
             call transfer_coord(x, y, z, .false.)
             call update_local_coord(x, y, z)
          else
#endif
#if KEY_PARALLEL==1
             TMERI(TIMDCNTRL)=TMERI(TIMDCNTRL)+ECLOCK()-TIMMER
             CALL PSYNC()
             TIMMER=ECLOCK()
             CALL VDGBR(X,Y,Z,0)
             TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
             CALL PSYNC()
             TIMMER=ECLOCK()
#endif
#if KEY_DOMDEC==1
          endif
#endif
       ENDIF
    ENDIF

#if KEY_PARALLEL==1
    TMERI(TIMDCNTRL)=TMERI(TIMDCNTRL)+ECLOCK()-TIMMER
#endif

    !     Start or Restart
    IF ((IGVOPT == 2 .AND. (IDYNPR.EQ.0 .OR. JHSTRT.EQ.0)) .OR. &
         (IGVOPT >= 3 .AND. (IDYNPR == 0 .OR. JHSTRT.EQ.0))) THEN
       IF (debug) WRITE(OUTU,*) &
            'DEBUG -- Compute forces (start/restart)'
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
       !     some Langevin forces should be calculated  (Not sure this is needed)
       if (NDXlang > 0) then
          do i = ATFRST,ATLAST
             ti = INLCKP(I)
             if (QNHLANG(ti)) then
!                varlang = SQRT( TWO*AMASS_(i) * KBOLTZ*RTMPR(ti) &
!                     * TIMFAC*FBETA(i)/DELTA )
                !DXsave(i) = BMGAUS(varlang,ISEED)
                !DYsave(i) = BMGAUS(varlang,ISEED)
                !DZsave(i) = BMGAUS(varlang,ISEED)
             endif
          enddo
       endif

       CALL GRAM( &
#if KEY_MTS==1
            DXFAST,DYFAST,DZFAST,DXMEDIUM,DYMEDIUM,DZMEDIUM,  &
#endif
            DX,DY,DZ)
    ENDIF

#if KEY_PARALLEL==1
    TIMMER=ECLOCK()
#endif
#if KEY_SGLD==1
     !  WXW    perform force average
     IF(QSGLD.OR.QSGMD)THEN
        CALL SGFAVG(ATFRST,ATLAST,EPROP(EPOT),IMOVE, &
             DX,DY,DZ)
     ENDIF
#endif

    !     Start
    IF (IGVOPT == 2 .AND. IDYNPR.EQ.0 .AND. JHSTRT.EQ.0 &
         .AND. NPRIV == 0) THEN
       !        Rattle velocities

       IF (QHOLO) THEN

          !           Compute ~V(0) = Rattle[V(0)]
          CALL RattleRoll( &
               ATFRST,ATLAST, &
               X,Y,Z, VX,VY,VZ, &
               AMASS_,IMOVE_, DELTA, .TRUE., &
               lagrange, ISKPR, &
               QPCON,QZONLY,QPCONFULL,ZETA_CP,W_CP, RV2_CP, &
               XTLTYP,XTLREF )
          !           Compute V(-1/2) = V(0) - dt/2m F(0)
          CALL PropagateVelo( VX,VY,VZ, DX,DY,DZ, &
               ATFRST,ATLAST, &
               -HALF*DELTA,AMASS_,IMOVE_ )
          !           Compute ~V(-1/2) = Rattle[V(-1/2)]
          CALL RattleRoll( &
               ATFRST,ATLAST, &
               X,Y,Z, VX,VY,VZ, &
               AMASS_,IMOVE_, DELTA, .TRUE., &
               lagrange, ISKPR, &
               QPCON,QZONLY,QPCONFULL,ZETA_CP,W_CP, RV2_CP, &
               XTLTYP,XTLREF )
          !           Compute V(0) = ~V(-1/2) + dt/2m F(0)
          CALL PropagateVelo( VX,VY,VZ, DX,DY,DZ, &
               ATFRST,ATLAST, &
               HALF*DELTA,AMASS_,IMOVE_ )

          if (debug) write(OUTU,*) 'DEBUG -- ', &
               'Keep unRATTLED velocities'
          CALL CopyArrays( VX,VY,VZ,VXsave,VYsave,VZsave, &
               ATFRST,ATLAST )

          if (debug) write(OUTU,*) 'DEBUG -- ', &
               'RATTLE...'
          CALL RattleRoll( &
               ATFRST,ATLAST, &
               X,Y,Z, VX,VY,VZ, &
               AMASS_,IMOVE_, DELTA, .TRUE., &
               lagrange, ISKPR, &
               QPCON,QZONLY,QPCONFULL,ZETA_CP,W_CP, RV2_CP, &
               XTLTYP,XTLREF )

          CALL RattleForces( VX,VY,VZ, &
               VXsave,VYsave,VZsave,DXc,DYc,DZc, &
               ATFRST,ATLAST, DELTA,AMASS_, &
               RMSDrattle )
          IF (PRNLEV > 5 .OR. debug) WRITE(OUTU,*) 'DEBUG -- ', &
               'RMSDrattle',RMSDrattle
       ENDIF
    ENDIF

    !     Restart
    IF (IGVOPT == 2 .AND. IDYNPR.EQ.0 .AND. JHSTRT.EQ.0 &
         .AND. NPRIV > 0) THEN
       !        Nothing to do...

    ENDIF

    IF (IGVOPT == 2 .AND. QNOSE) THEN
       if (debug) write(OUTU,*) 'DEBUG -- ', &
            'Save pre-ROLL state (including virial)'
       CALL CopyArrays( X,Y,Z, XOLD,YOLD,ZOLD, &
            ATFRST,ATLAST )
       CALL CopyArrays( VX,VY,VZ, Xtemp,Ytemp,Ztemp, &
            ATFRST,ATLAST )
       do ti = 1,NOBL
          SNHroll(ti) = SNH(ti)
          SNHVroll(ti) = SNHV(ti)
       enddo
       CALL COPY33(ETA_CProll,ETA_CP)
       CALL COPY33(ZETA_CProll,ZETA_CP)
       IF (QPCON) CALL GETXTL(XTLroll,XTLABC,XTLTYP,XTLREF)
       VENEroll(1) = EPRESS(VIXX)
       VENEroll(2) = EPRESS(VIXY)
       VENEroll(3) = EPRESS(VIXZ)
       VENEroll(4) = EPRESS(VIYX)
       VENEroll(5) = EPRESS(VIYY)
       VENEroll(6) = EPRESS(VIYZ)
       VENEroll(7) = EPRESS(VIZX)
       VENEroll(8) = EPRESS(VIZY)
       VENEroll(9) = EPRESS(VIZZ)
       TR_VENE_ROLL = EPROP(VIRI)
    ENDIF

    !     Write out initial conditions
    IF (IDYNPR == 0 .OR. JHSTRT.EQ.0) THEN
       ISTEP = ISTART - 1
       TIME = TIMFAC * NPRIV * DELTA

       CALL KineticEFNH( &
            QNOSE,NOBL,INLCKP, &
            QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
            kin2,kin2_nh, ATFRST,ATLAST, IMOVE_,AMASS_, &
            VX,VY,VZ, &
#if KEY_PARALLEL==1
                                !    $        kin2_for_GCOMB,
#endif
            ISDRUDE )

       !        if (QHOLO.and.NPRIV > 0) then
       if (QHOLO) then
          if (debug) write(OUTU,*) 'DEBUG -- ', &
               'Correct virial for SHAKE (initial) 2'
          !           We have to extrapolate back the constraint forces
          !     NOTE for parallel: VIRSHK has to be after ENERGY call
          CALL VIRSHK( EPRESS(VIXX:VIZZ), NATOMX, &
               X,Y,Z,DXc,DYc,DZc )
       endif

       if (QPCON .and. NDRUDE > 0 .and. .not.QDSCALE_CP) then
          CALL CorrectVirial( 1, EPRESS(VIXX:VIZZ), &
               X,Y,Z, DX,DY,DZ, &
               DXc,DYc,DZc, ATFRST,ATLAST, &
               AMASS_,ISDRUDE,IMOVE_ )
       endif
       CALL PRSEXT
       CALL PRSINT(NATOMX,AMASS_,VX,VY,VZ,ONE,VX,VX,VX,ZERO)
       ! APH: Note VX, VX, VX here -------------|
       !      are just dummies
       if (QPCON .and. NDRUDE > 0 .and. .not.QDSCALE_CP) then
          CALL CorrectVirial( 0, EPRESS(VIXX:VIZZ), &
               X,Y,Z, DX,DY,DZ, &
               DXc,DYc,DZc, ATFRST,ATLAST, &
               AMASS_,ISDRUDE,IMOVE_ )
       endif

       QK1 = ZERO
       QK2 = ZERO
       KINE_CP = ZERO
       POT_CP = ZERO
       IF (QNOSE) THEN
          DO ti = 1,NOBL
             IF (.NOT.useThermostat(ti)) GOTO 702
             SNHF(ti) = kin2_nh(ti) - NDGN(ti)*KBOLTZ*RTMPR(ti)
             IF (QPCON .AND. ti == TI_CP) THEN
                IF (QPCONFULL) THEN
                   CALL TRANSPS(prod,ZETA_CP,3,3)
                   CALL MULT33(prod,prod,ZETA_CP)
                   trZETA2 = prod(1,1) + prod(2,2) + prod(3,3)
                   SNHF(ti) = SNHF(ti) + W_CP*trZETA2 &
                        - XDIM*KBOLTZ*RTMPR(ti)
                ELSE
                   SNHF(ti) = SNHF(ti) + W_CP*ZETA_CP(3,3)**2 &
                        - KBOLTZ*RTMPR(ti)
                ENDIF
             ENDIF

!!! Compute Nose-Hoover energy terms
             QK1 = QK1 + HALF*SQM(ti)*SNHV(ti)**2
             QK2 = QK2 + NDGN(ti)*KBOLTZ*RTMPR(ti)*SNH(ti)
702          CONTINUE
          ENDDO
          IF (QPCON) THEN
!!! Compute Andersen-Hoover energy terms
             if (QPCONFULL) then
                CALL TRANSPS(prod,ZETA_CP,3,3)
                CALL MULT33(prod,prod,ZETA_CP)
                trZETA2 = prod(1,1) + prod(2,2) + prod(3,3)
                KINE_CP = HALF * W_CP * trZETA2
                POT_CP = ATMOSP*P_CP * EPROP(VOLUME)
             else
                KINE_CP = HALF * W_CP * ZETA_CP(3,3)**2
                POT_CP = ATMOSP*P_CP * EPROP(VOLUME)
             endif
          ENDIF
       ENDIF

       EPROP(TOTKE) = kin2 / TWO
       EPROP(TOTE) = EPROP(EPOT) + EPROP(TOTKE) + QK1 + QK2 &
            + KINE_CP + POT_CP

       IF (QNOSE) THEN
          EPROP(HFCTE) = EPROP(EPOT) + EPROP(TOTKE)
          EPROP(EHFC) = QK1 + QK2
          IF (QPCON) THEN
             EPROP(EHFC) = EPROP(EHFC) + KINE_CP + POT_CP
          ENDIF
       ELSE
          EPROP(HFCTE) = ZERO
          EPROP(EHFC) = ZERO
       ENDIF
       EPROP(TEMPS) = kin2 / (KBOLTZ*NDEGF)
       IF (QNOSE) THEN        ! What T do we want to print?
          DO ti = 1,NOBL
             temp_nh(ti) = kin2_nh(ti) / (KBOLTZ*NDGN(ti))
          ENDDO
          EPROP(TEMPS) = temp_nh(1)
       ENDIF

       IF (PRNLEV > 2) THEN
          CALL PRINTE(OUTU, EPROP, ETERM, 'DYNA', 'DYN', .TRUE., &
               ISTEP, TIME, ZERO, .TRUE.)
#if KEY_SGLD==1 /*sgldprint*/
          IF(QSGLD.OR.QSGMD)CALL  PRNTSG(OUTU,.TRUE.,'DYNA',.TRUE.)
#endif /* (sgldprint)*/
          IF (QPCON) THEN
             RVAL = DOTVEC(DXTL,DXTL,XDIM) / XDIM
             GNORM = ZERO
             IF (RVAL > ZERO) GNORM = SQRT(RVAL)
             CALL PRNXTLD(OUTU,'DYNA',XTLTYP,XUCELL,.TRUE.,GNORM, &
                  .TRUE.,EPRESS)
          ENDIF
       ENDIF

    ENDIF


    !     Initialize averages and fluctuations
    IF (JHSTRT == 0) THEN
       DO I = ATFRST,ATLAST
#if KEY_PARASCAL==1
          IF (JPBLOCK(I) /= MYNOD) GOTO 703
#endif
          VK(I) = ZERO
#if KEY_PARASCAL==1
703       CONTINUE
#endif
       ENDDO
       JHTEMP = ZERO
       FITP = ZERO
       call avfl_reset_lt()
       if (QPCON) call avfl_ucell_reset_lt()
    ENDIF


    IF (MOD(ISTART-1,IPRFRQ) == 0) THEN
       call avfl_reset()
       FITA = ZERO
       if (QPCON) call avfl_ucell_reset()
    ENDIF
#if KEY_DIMS==1
  OMSC=0.0
  counter=0
#endif

    DO ISTEP = ISTART,ISTOP
!QC_Ito_UW0616
#if KEY_RXNCOR==1
  UMBMDSTEP=ISTEP
#endif

       !  Get forces for first portion of integrator loop (E.Harder 2004)
       if (ISTEP == 1) then
          if (QNHLANGEVIN) then
             NHGAMMA_TEMP = NHGAMMA
             NHGAMMAD_TEMP = NHGAMMAD
             do ia = ATFRST,ATLAST
#if KEY_DOMDEC==1
                if (q_domdec) then
                   i = atoml(ia)
                else
#endif
                   i = ia
#if KEY_DOMDEC==1
                endif
#endif
                if (ISDRUDE(i+1)) then
                   VX(i+1) = VX(i)
                   VY(i+1) = VY(i)
                   VZ(i+1) = VZ(i)
                   MT_ = AMASS(i) + AMASS(i+1)
                   DXsave(i) = DX(i) + DX(i+1)
                   DYsave(i) = DY(i) + DY(i+1)
                   DZsave(i) = DZ(i) + DZ(i+1)
                   DXsave(i+1) = DX(i+1)*AMASS(i)/MT_ -  &
                        DX(i)*AMASS(i+1)/MT_
                   DYsave(i+1) = DY(i+1)*AMASS(i)/MT_ -  &
                        DY(i)*AMASS(i+1)/MT_
                   DZsave(i+1) = DZ(i+1)*AMASS(i)/MT_ -  &
                        DZ(i)*AMASS(i+1)/MT_
                endif
                continue
             enddo
          endif
       else
          !NHGAMMA = NHGAMMA_TEMP
       endif

       JHSTRT = JHSTRT+1
       NPRIV = NPRIV+1

#if KEY_DMCONS==1
       DSTEP = NPRIV
#endif
#if KEY_RGYCONS==1
       DSTEPRG = NPRIV
#endif


       IF (QAVER) THEN        !!! Compute average coordinates
          NAVER = NAVER + 1
#if KEY_DOMDEC==1
          if (q_domdec) then
             do ia = atfrst,atlast
                i = atoml(ia)
                xave(i) = xave(i) + x(i)
                yave(i) = yave(i) + y(i)
                zave(i) = zave(i) + z(i)
             enddo
          else
#endif
             DO I = ATFRST,ATLAST
#if KEY_PARASCAL==1
                IF (JPBLOCK(I) /= MYNOD) GOTO 704
#endif
                XAVE(I) = XAVE(I) + X(I)
                YAVE(I) = YAVE(I) + Y(I)
                ZAVE(I) = ZAVE(I) + Z(I)
#if KEY_PARASCAL==1
704             CONTINUE
#endif
             ENDDO
#if KEY_DOMDEC==1
          endif
#endif
       ENDIF


       CALL CopyArrays( X,Y,Z, XOLD,YOLD,ZOLD, ATFRST,ATLAST )

#if KEY_PARALLEL==1
       tmeri(timdcntrl)=tmeri(timdcntrl)+eclock()-timmer
#endif
       if (debug) write(OUTU,*) 'DEBUG -- ', &
            'Update lists'
       CALL UPDECI(ISTEP-1,X,Y,Z,WMAIN,1,XOLD,YOLD,ZOLD,VX,VY,VZ)

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
           !   exit mainloop
           GOTO 1111
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

     IF(DIMSON) THEN
        !     Biased Energy
        CALL UPDATEDIMSED(EPROP(EPOT))
        CALL DIMSSCORE(PRNLEV >= 3)
     ENDIF
     IF(DIMSCARTON) THEN
       CALL DIMSDLNGV_VV2(GAMMA,ISEED, &
            !XCOMP,YCOMP,ZCOMP, XOLD, YOLD, ZOLD, &
            X,Y,Z, XOLD, YOLD, ZOLD, &
            DIMTARGXA, DIMTARGYA, DIMTARGZA, &
            XScratch,YSCratch,ZScratch, &
            DXDIMS,DYDIMS,DZDIMS, &
            XNEW,YNEW,ZNEW, &
            DELTA, VX,VY,VZ, &
            WMAIN,NATOM*2,NATOM*3,DWIDTH,QPRINT, &
            ATFRST,ATLAST,IDIMS)
     ENDIF

#endif

       CALL VV2( &
            NATOMX, NDEGF, AMASS_, IMOVE_, X, Y, Z, VX, VY, VZ, &
            XNEW, YNEW, ZNEW, XOLD, YOLD, ZOLD, Xtemp, Ytemp, Ztemp, &
            Xsave, Ysave, Zsave, &
            VXsave, VYsave, VZsave, &
            kcorr,kcount, &
            DXsave, DYsave, DZsave, &
            DXlang, DYlang, DZlang, &
            VXlang, VYlang, VZlang, &
            DX, DY, DZ, NPRIV, &
                                !
            NDRUDE, ISDRUDE,ISTEP, &
            QHOLO, DXc,DYc,DZc, lagrange, &
            ISKPR, BNBND, BIMAG, &
            SNHroll,SNHVroll, &
            ETA_CProll,ZETA_CProll,sf,XTLroll,VENEroll,TR_VENE_ROLL, &
                                !
            QNOSE, NOBL, MAXNOS, NDGN, RTMPR, SQM, SNH, SNHV, SNHF, &
            INLCKP, &
            QNHLANG, NDXlang, FBETA, &
            QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M,QNHLANGEVIN,NHGAMMA, &
            NHGAMMAD,QCMLANGEVIN,CMGAMMA, &
            SCFdipoles, useThermostat, TOLSCF, MAXSCF, &
                                !
            kin2, kin2_nh, &
                                !
            QPCON, QDSCALE_CP, W_CP, P_CP, TI_CP, ETA_CP, ZETA_CP, &
            RX1_CP, RX2_CP, RV1_CP, RV2_CP, &
            QZONLY, QPCONFULL, &
                                !
            DELTA, YSNC, &
                                !
            OUTU, debug &
            )

#if KEY_PARALLEL==1
       timmer=eclock()
#endif
#if KEY_PERT==1
       IF(QPSHAKE) CALL WRNDIE(-2,'<DYNAMVV2>', &
            'PERT-SHAKE INTERACTIONS BUT EPCONS NOT CALLED')
       IF(QPERT) THEN
          IF(QACCUM) CALL EPSUM
          CALL EWORKT(VX,VY,VZ,DX,DY,DZ)
       ENDIF
#endif


       QK1 = ZERO
       QK2 = ZERO
       KINE_CP = ZERO
       POT_CP = ZERO
       IF (QNOSE) THEN
!!! Compute Nose-Hoover energy terms
          DO ti = 1,NOBL
             IF (.NOT.useThermostat(ti)) GOTO 705
             QK1 = QK1 + HALF*SQM(ti)*SNHV(ti)**2
             QK2 = QK2 + NDGN(ti)*KBOLTZ*RTMPR(ti)*SNH(ti)
705          CONTINUE
          ENDDO
          IF (QPCON) THEN
!!! Compute Andersen-Hoover energy terms
             if (QPCONFULL) then
                CALL TRANSPS(prod,ZETA_CP,3,3)
                CALL MULT33(prod,prod,ZETA_CP)
                trZETA2 = prod(1,1) + prod(2,2) + prod(3,3)
                KINE_CP = HALF * W_CP * trZETA2
                POT_CP = ATMOSP*P_CP * EPROP(VOLUME)
             else
                KINE_CP = HALF * W_CP * ZETA_CP(3,3)**2
                POT_CP = ATMOSP*P_CP * EPROP(VOLUME)
             endif
          ENDIF
       ENDIF

       EPROP(TOTKE) = kin2 / TWO
       EPROP(TOTE) = EPROP(EPOT) + EPROP(TOTKE) + QK1 + QK2 &
            + KINE_CP + POT_CP

       IF (QNOSE) THEN
          EPROP(HFCTE) = EPROP(EPOT) + EPROP(TOTKE)
          EPROP(EHFC) = QK1 + QK2
          IF (QPCON) THEN
             EPROP(EHFC) = EPROP(EHFC) + KINE_CP + POT_CP
          ENDIF
       ENDIF

       !
       !     CHECK THE CHANGE IN TOTAL ENERGY TO BE SURE EVERYTHING IS OK.
       !
       IF ((.NOT.QNOSE).AND.(JHSTRT > 2)) THEN
          IF (ABS(TOTEPR-EPROP(TOTE)) > MAX(ECHECK,0.1D0*EPROP &
               (TOTKE))) THEN
             IF (TOTEPR  /=  -9999.0) THEN
                WRITE (OUTU,2000)  &
                     ECHECK,TOTEPR,EPROP(TOTE),EPROP(TOTKE)
2000            FORMAT(' TOTAL ENERGY CHANGE EXCEDED'/G12.2, &
                     ' KCAL AND 10% OF THE TOTAL KINETIC ENERGY', &
                     ' IN THE LAST STEP'/ &
                     ' PREVIOUS EPROP(EPOT)  = ',G14.4, &
                     ' CURRENT EPROP(EPOT)  = ',G14.4, &
                     ' KINETIC  = ',G14.4)

#if KEY_DOMDEC==1
                if (q_domdec) then
                   call copy_to_all(x,y,z)
                else
#endif
#if KEY_PARALLEL==1
                   CALL VDGBR(X,Y,Z,1)
#endif
#if KEY_DOMDEC==1
                endif
#endif

                CALL WRNDIE(-2,'<DYNAMVV2>', &
                     'ENERGY CHANGE TOLERANCE EXCEEDED')
             ENDIF
          ENDIF
       ENDIF

       !
       TOTEPR = EPROP(TOTE)

       TIME = TIMFAC*NPRIV*DELTA
       EPROP(TEMPS) = kin2 / (KBOLTZ*NDEGF) ! What T do we want?
       IF (QNOSE) THEN
          DO ti = 1,NOBL
             temp_nh(ti) = kin2_nh(ti) / (KBOLTZ*NDGN(ti))
          ENDDO
          EPROP(TEMPS) = temp_nh(1)
       ENDIF

       !
       JHTEMP = JHTEMP + EPROP(TEMPS)
       call avfl_update(EPROP, ETERM, EPRESS)

       !        Unit cell and gradient terms
       if (QPCON) call avfl_ucell_update()
       IPRFRQcounter = MOD(IST1,IPRFRQ) + ISTEP - IST1
       FITA = FITA + IPRFRQcounter * EPROP(TOTE)
       FITP = FITP + JHSTRT * EPROP(TOTE)


       !
       !        Write trajectories
       !
       IF (NSAVC > 0) THEN
          IF (MOD(ISTEP,NSAVC) == 0) THEN
             IF (QAVER) THEN
                !                 Write average coordinates
                DNUM = ONE
                DNUM = DNUM/NAVER
#if KEY_DOMDEC==1
                if (q_domdec) then
                   do ia = atfrst,atlast
                      i = atoml(ia)
                      xave(i) = xave(i)*dnum
                      yave(i) = yave(i)*dnum
                      zave(i) = zave(i)*dnum
                   enddo
                else
#endif
                   DO I = ATFRST,ATLAST
#if KEY_PARASCAL==1
                      IF (JPBLOCK(I) /= MYNOD) GOTO 706
#endif
                      XAVE(I) = XAVE(I)*DNUM
                      YAVE(I) = YAVE(I)*DNUM
                      ZAVE(I) = ZAVE(I)*DNUM
#if KEY_PARASCAL==1
706                   CONTINUE
#endif
                   ENDDO
#if KEY_DOMDEC==1
                endif
#endif
#if KEY_DOMDEC==1
                if (q_domdec) then
                   call copy_to_all(xave, yave, zave)
                else
#endif
#if KEY_PARALLEL==1
                   CALL VDGBR(XAVE,YAVE,ZAVE,1)
#endif
#if KEY_DOMDEC==1
                endif
#endif
                CALL WRITCV(XAVE,YAVE,ZAVE, &
#if KEY_CHEQ==1
                     (/ ZERO /), .FALSE., &
#endif
                     NATOMX,FREEAT,NFREAT,NPRIV, &
                     ISTEP,NDEGF,DELTA,NSAVC,NSTEP, &
                     TITLEA,NTITLA,IUNCRD, &
                     .FALSE.,.FALSE., (/ 0 /), .FALSE., (/ ZERO /))
                NAVER = 0
                call ZeroArrays(xave, yave, zave, atfrst, atlast)
             ELSE
                !                 Write coordinates
#if KEY_DOMDEC==1
                if (q_domdec) then
                   call copy_to_all(x, y, z)
                else
#endif
#if KEY_PARALLEL==1
                   CALL VDGBR(X,Y,Z,1)
#endif
#if KEY_DOMDEC==1
                endif
#endif
                CALL WRITCV(X,Y,Z, &
#if KEY_CHEQ==1
                     (/ ZERO /), .FALSE., &
#endif
                     NATOMX,FREEAT,NFREAT,NPRIV,ISTEP, &
                     NDEGF,DELTA, &
                     NSAVC,NSTEP,TITLEA,NTITLA,IUNCRD,.FALSE., &
                     .FALSE., (/ 0 /), .FALSE., (/ ZERO /))
             ENDIF
#if KEY_SGLD==1 /*sgldprint*/
            IF(NSAVC > 0.AND.MOD(ISTEP,NSAVC) == 0.AND.PRNLEV > 0) THEN
              IF(QSGLD.OR.QSGMD)THEN
                 CALL PRINTE(OUTU, EPROP, ETERM, 'TRAJ', 'DYN', .FALSE., &
                ISTEP, TIME, ZERO, .TRUE.)
                 CALL PRNTSG(OUTU,.FALSE.,'TRAJ',.FALSE.)
              ENDIF
            ENDIF
#endif /* (sgldprint)*/
          ENDIF
       ENDIF
       !        Write velocities
       IF (NSAVV > 0) THEN
          IF (MOD(ISTEP,NSAVV) == 0) THEN
#if KEY_DOMDEC==1
             if (q_domdec) then
                call copy_to_all(vx, vy, vz)
             else
#endif
#if KEY_PARALLEL==1
                CALL VDGBR(VX,VY,VZ,1)
#endif
#if KEY_DOMDEC==1
             endif
#endif
             CALL WRITCV(VX,VY,VZ, &
#if KEY_CHEQ==1
                  (/ ZERO /), .FALSE., &
#endif
                  NATOMX,FREEAT,NFREAT,NPRIV,ISTEP, &
                  NDEGF,DELTA, &
                  NSAVV,NSTEP,TITLEA,NTITLA,IUNVEL,.TRUE., &
                  .FALSE., (/ 0 /), .FALSE., (/ ZERO /))
          ENDIF
       ENDIF


       !
       !        Write Nose-Hoover/Andersen-Hoover coordinates
       !
       IF ((IUNOS >= 0).AND.QNOSE) THEN
          IF (MOD(ISTEP,NSNOS) == 0) THEN
166          FORMAT(I16,2F16.6)
167          FORMAT(2F16.6)
             WRITE(IUNOS,166) NPRIV,TIME,EPROP(TOTE)
             WRITE(IUNOS,167) (SNH(ti),temp_nh(ti),ti = 1,NOBL)
             IF (QPCON) THEN
                IF (QPCONFULL) THEN
                   WRITE(IUNOS,167) ETA_CP(1,1),ZETA_CP(1,1)
                   WRITE(IUNOS,167) ETA_CP(1,2),ZETA_CP(1,2)
                   WRITE(IUNOS,167) ETA_CP(1,3),ZETA_CP(1,3)
                   WRITE(IUNOS,167) ETA_CP(2,1),ZETA_CP(2,1)
                   WRITE(IUNOS,167) ETA_CP(2,2),ZETA_CP(2,2)
                   WRITE(IUNOS,167) ETA_CP(2,3),ZETA_CP(2,3)
                   WRITE(IUNOS,167) ETA_CP(3,1),ZETA_CP(3,1)
                   WRITE(IUNOS,167) ETA_CP(3,2),ZETA_CP(3,2)
                   WRITE(IUNOS,167) ETA_CP(3,3),ZETA_CP(3,3)
                ELSE
                   WRITE(IUNOS,167) ETA_CP(3,3),ZETA_CP(3,3)
                ENDIF
             ENDIF
             CALL GFLUSH(IUNOS)
          ENDIF
       ENDIF


       !
       !        Adaptive umbrella sampling, write out umbrella coordinate
       !
#if KEY_ADUMB==1
       IF ((NSAVC > 0).AND.(WCUNUM.GT.0)) THEN
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

       !        Compute pressure
       if (QPCON .and. NDRUDE > 0 .and. .not.QDSCALE_CP) then
          CALL CorrectVirial( 1, EPRESS(VIXX:VIZZ), &
               X,Y,Z, DX,DY,DZ, &
               DXc,DYc,DZc, &
               ATFRST,ATLAST, AMASS_,ISDRUDE,IMOVE_ )
       endif
       CALL PRSEXT
       CALL PRSINT(NATOMX,AMASS_,VX,VY,VZ,ONE,VX,VX,VX,ZERO)
       ! APH: Note VX, VX, VX here -------------|
       !      are just dummies
       if (QPCON .and. NDRUDE > 0 .and. .not.QDSCALE_CP) then
          CALL CorrectVirial( 0, EPRESS(VIXX:VIZZ), &
               X,Y,Z, DX,DY,DZ, &
               DXc,DYc,DZc, &
               ATFRST,ATLAST, AMASS_,ISDRUDE,IMOVE_ )
       endif
       !
       ! output pressure tensor when IUPTEN postive; rmv july 2016
       !
       ! hwm: unprotected mod() fix
       !IF(IUPTEN > 0.AND.IOLEV > 0.AND.MOD(NPRIV,IPTFRQ) == 0)  &
       IF(IUPTEN > 0.AND.IOLEV > 0.AND.IPTFRQ>0) then
          if (MOD(NPRIV,IPTFRQ) == 0) &
               WRITE(IUPTEN,955) TIME,(EPRESS(I),I=PIXX,PIZZ)
       ENDIF
 955   FORMAT(F13.3,1X,9F16.4)
       !
       !        Print energy terms
       !
       IF (MOD(ISTEP,NPRINT) == 0) THEN
          IF (PRNLEV >= 2) THEN
             CALL PRINTE(OUTU, EPROP, ETERM, 'DYNA', 'DYN', .FALSE., &
                  ISTEP, TIME, ZERO, .TRUE.)
#if KEY_SGLD==1 /*sgldprint*/
              IF((QSGLD.OR.QSGMD).AND.PRNLEV>=2)CALL PRNTSG(OUTU,.FALSE.,'DYNA',.FALSE.)
#endif /* (sgldprint)*/
             IF (QPCON) THEN
                RVAL = DOTVEC(DXTL,DXTL,XDIM) / XDIM
                GNORM = ZERO
                IF (RVAL > ZERO) GNORM = SQRT(RVAL)
                CALL PRNXTLD(OUTU,'DYNA',XTLTYP,XUCELL,.TRUE.,GNORM, &
                     .TRUE.,EPRESS)
             ENDIF
          ENDIF
          !
          CALL WRETERM(NPRIV,TIME,QKUHEAD)
          !
       ENDIF

#if KEY_BLOCK==1
       !.ab.Print out HybridH. Use the #BLOCK#
       IF(IOLEV >= 0) CALL PRHYBH()
       !.ab.
#endif

#if KEY_AXD==1
       IF (LAXD) THEN
         CALL VVAXD(X,Y,Z,XOLD,YOLD,ZOLD,VX,VY,VZ,DX,DY,DZ,AMASS,NATOMX,ISTEP,NSTEP)
       ENDIF
#endif /* AXD*/
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
        CALL SET_PARAM('OMSCORE',OMSC)
     ENDIF
#endif

    ENDDO
#if KEY_DIMS==1
1111  counter = counter+1
#endif
#if KEY_DOMDEC==1
    if (q_domdec) then
       call copy_to_all(vx, vy, vz)
       call copy_to_all(xold, yold, zold)
    else
#endif
#if KEY_PARALLEL==1
       CALL VDGBR(VX,VY,VZ,1)
#endif
#if KEY_PARALLEL==1
       CALL VDGBR(XOLD,YOLD,ZOLD,1)
#endif
#if KEY_DOMDEC==1
    endif
#endif

    IF (IPRFRQcounter < IPRFRQ) goto 999 !RETURN


    !
    !     Now do long time fluctuations
    !
    call avfl_update_lt()

    !     Unit cell and gradient long terms
    if (QPCON) call avfl_ucell_update_lt()
    NUMSTP = IPRFRQcounter
    DNUM   = NUMSTP
    DNM1   = DNUM - ONE
    DNP1   = DNUM + ONE
    DNPM1  = DNM1 * DNP1
    IF (NUMSTP > 1) THEN
       DRIFTA = (TWO*FITA-DNP1*EPRPA(TOTE))*SIX/(DNPM1*DNUM)
       EAT0A  = EPRPA(TOTE)/DNUM-DRIFTA*DNP1/TWO
       IF ((EPRP2A(TOTE)/DNUM-(EPRPA(TOTE)/DNUM)**2) > ZERO) THEN
          CORRA = DNPM1/(EPRP2A(TOTE)/DNUM &
               -(EPRPA(TOTE)/DNUM)**2)/TWELVE
          CORRA = DRIFTA*SQRT(CORRA)
       ELSE
          CORRA = ZERO
       ENDIF
       !
       !     . Compute statistics.
       call avfl_compute(DNUM)

       !        Calculate average unit cell and gradient terms
       if (QPCON) call avfl_ucell_compute(DNUM)
       !
    ENDIF

#if KEY_SGLD==1 /* sgld store params */
       if (qsgld .or. qsgmd) call sgld_ave_params()
#endif /* sgld store params */

       !     . Print out the results
    IF (PRNLEV  >=  2) THEN
       call avfl_print_aver(NUMSTP, TIME)
#if KEY_SGLD==1 /*sgldprint*/
        IF(QSGLD.OR.QSGMD)CALL PRNTSG(OUTU,.FALSE.,'AVER',.TRUE.)
#endif /* (sgldprint)*/
       if (QPCON) call avfl_ucell_print_aver(NUMSTP)
       call avfl_print_fluc(NUMSTP, TIME)
#if KEY_SGLD==1 /*sgldprint*/
        IF(QSGLD.OR.QSGMD)CALL PRNTSG(OUTU,.FALSE.,'FLUC',.TRUE.)
#endif /* (sgldprint)*/
       if (QPCON) call avfl_ucell_print_fluc(NUMSTP)
    ENDIF
    !
    DRIFTP = DRIFTA
    EAT0P = EAT0A
    CORRP = CORRA
    IF (JHSTRT <= NUMSTP) GOTO 190
    NUMSTP = JHSTRT
    DNUM = NUMSTP
    DNM1 = DNUM - ONE
    DNP1 = DNUM + ONE
    DNPM1 = DNM1 * DNP1
    IF (NUMSTP > 1) THEN
       DRIFTP = (TWO*FITP - DNP1*EPRPP(TOTE))*SIX/(DNPM1*DNUM)
       EAT0P  = EPRPP(TOTE)/DNUM-DRIFTP*DNP1/TWO
       IF ((EPRP2P(TOTE)/DNUM-(EPRPP(TOTE)/DNUM)**2) > ZERO) THEN
          CORRP = DNPM1/(EPRP2P(TOTE)/DNUM &
               -(EPRPP(TOTE)/DNUM)**2)/TWELVE
          CORRP = DRIFTP*SQRT(CORRP)
       ELSE
          CORRP = ZERO
       ENDIF
    ENDIF
    call avfl_compute_lt(DNUM)

    !     Calculate average unit cell and gradient terms
    if (QPCON) call avfl_ucell_compute_lt(DNUM)
    !
    !     . Print out results
    IF (PRNLEV >= 2) THEN
       call avfl_print_aver_lt(NUMSTP, TIME)
       if (QPCON) call avfl_ucell_print_aver_lt(NUMSTP)
       call avfl_print_fluc_lt(NUMSTP, TIME)
       if (QPCON) call avfl_ucell_print_fluc_lt(NUMSTP)
    ENDIF
    !
190 CONTINUE
    IF (NUMSTP <= 1 .OR. JHSTRT.LE.1 .OR. PRNLEV < 3) goto 999 !RETURN
    !
    WRITE(OUTU,195)DRIFTA,DRIFTP,EAT0A,EAT0P,CORRA,CORRP
195 FORMAT(/5X,'DRIFT/STEP (LAST-TOTAL): ',1P,2G17.8, &
         /5X,'EPROP(EPOT) AT STEP 0  : ',1P,2G17.8, &
         /5X,'CORR. COEFFICIENT      : ',1P,2G17.8)


999 continue

    if (IGVOPT == 2) IGVOPT = 3

    if (QHOLO) then ! Saving constraint forces to XNEW
       CALL CopyArrays( &
            DXc,DYc,DZc, &
            XNEW,YNEW,ZNEW, ATFRST,ATLAST )
    endif
    if (SCFdipoles) then
       amass_(1:natomx) = dmass(1:natomx)
       do i = ATFRST,ATLAST
          if (ISDRUDE(i)) then
             if (IMOVE_(i) == -1) IMOVE_(i) = 0
          endif
       enddo
    endif

    call chmdealloc('dynamvv2.src','DYNAMVV2','DXc',NATOMX,crl=DXc)
    call chmdealloc('dynamvv2.src','DYNAMVV2','DYc',NATOMX,crl=DYc)
    call chmdealloc('dynamvv2.src','DYNAMVV2','DZc',NATOMX,crl=DZc)
    call chmdealloc('dynamvv2.src','DYNAMVV2','Xsave',NATOMX,crl=Xsave)
    call chmdealloc('dynamvv2.src','DYNAMVV2','Ysave',NATOMX,crl=Ysave)
    call chmdealloc('dynamvv2.src','DYNAMVV2','Zsave',NATOMX,crl=Zsave)
    call chmdealloc('dynamvv2.src','DYNAMVV2','VXsave',NATOMX,crl=VXsave)
    call chmdealloc('dynamvv2.src','DYNAMVV2','VYsave',NATOMX,crl=VYsave)
    call chmdealloc('dynamvv2.src','DYNAMVV2','VZsave',NATOMX,crl=VZsave)
    call chmdealloc('dynamvv2.src','DYNAMVV2','lagrange',NCONST,crl=lagrange)
    if(qnose)then
       call chmdealloc('dynamvv2.src','DYNAMVV2','DXlang',NATOMX,crl=DXlang)
       call chmdealloc('dynamvv2.src','DYNAMVV2','DYlang',NATOMX,crl=DYlang)
       call chmdealloc('dynamvv2.src','DYNAMVV2','DZlang',NATOMX,crl=DZlang)
       call chmdealloc('dynamvv2.src','DYNAMVV2','VXlang',NATOMX,crl=VXlang)
       call chmdealloc('dynamvv2.src','DYNAMVV2','VYlang',NATOMX,crl=VYlang)
       call chmdealloc('dynamvv2.src','DYNAMVV2','VZlang',NATOMX,crl=VZlang)
    endif
    if (SCFdipoles) then
       call chmdealloc('dynamvv2.src','DYNAMVV2','dmass',NATOMX,crl=dmass)
    endif

#if KEY_PARALLEL==1
    tmeri(timdcntrl)=tmeri(timdcntrl)+eclock()-timmer
#endif

    RETURN
  END SUBROUTINE DYNAMVV2


  SUBROUTINE VV2( &
       nAtoms, NDEGF, amass, imove, X, Y, Z, VX, VY, VZ, &
       XNEW, YNEW, ZNEW, XOLD, YOLD, ZOLD, Xtemp, Ytemp, Ztemp, &
       Xsave, Ysave, Zsave, VXsave, VYsave, VZsave, &
       kcorr,kcount, &
       DXsave, DYsave, DZsave, &
       DXlang, DYlang, DZlang, &
       VXlang, VYlang, VZlang, &
       DX, DY, DZ, NPRIV, &
                                !
       nDrudes, isDrude,ISTEP, &
       QHOLO, DXc, DYc, DZc, lagrange, &
       ISKPR, BNBND, BIMAG, &
       SNHroll,SNHVroll, &
       ETA_CProll,ZETA_CProll,sf,XTLroll,VENEroll,TR_VENE_ROLL, &
                                !
       QNOSE, NOBL, MAXNOS, NDGN, RTMPR, SQM, SNH, SNHV, SNHF, &
       INLCKP, &
       QNHLANG, NDXlang, FBETA, &
       QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
       IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M,QNHLANGEVIN,NHGAMMA, &
       NHGAMMAD,QCMLANGEVIN,CMGAMMA, &
       SCFdipoles, useThermostat, TOLSCF, MAXSCF, &
                                !
       kin2, kin2_nh, &
                                !
       QPCON, QDSCALE_CP, W_CP, P_CP, TI_CP, ETA_CP, ZETA_CP, &
       RX1_CP, RX2_CP, RV1_CP, RV2_CP, &
       QZONLY, QPCONFULL, &
                                !
       DELTA_, nc, &
                                !
       outu, verbose &
       )
    !
    !     Velocity-Verlet Algorithm
    !     by Guillaume Lamoureux
    !     edited by Ed Harder to include Langevin thermostats
    !
    !     Nose-Hoover thermostats
    !     Andersen-Hoover barostat
    !     SCF solution for induced dipoles
    !
    use energym
    use image
    use consta ! for ATMOSP
    use number ! for ZERO,ONE
    use reawri ! for ISEED
    use parallel
    use clcg_mod
    use machutil,only:eclock
    use tbmts
    use prssre
#if KEY_SGLD==1
  use sgld,only:sgfavg,sgldg,sgmdg,qsgld,qsgmd
#endif
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,natoml,atoml
#endif
#if KEY_DOMDEC==1
    use domdec_d2d_comm,only:transfer_coord
    use domdec_local,only:update_local_coord
#endif
    !
    !     Arguments
    !
    integer nAtoms            ! IN  Number of atoms (including the images)
    integer NDEGF
    real(chm_real)  amass(*)          ! IN  Atomic masses
    integer imove(*)          ! IN  Moving code (1..nAtoms)

    real(chm_real) :: X(:), Y(:), Z(:)
    real(chm_real)  VX(*), VY(*), VZ(*)
    real(chm_real)  XNEW(*), YNEW(*), ZNEW(*)
    real(chm_real)  XOLD(*), YOLD(*), ZOLD(*)
    real(chm_real)  Xtemp(*), Ytemp(*), Ztemp(*)
    real(chm_real)  Xsave(*), Ysave(*), Zsave(*)
    real(chm_real)  VXsave(*), VYsave(*), VZsave(*)
    integer kcorr,kcount
    real(chm_real)  DXsave(*), DYsave(*), DZsave(*)
    real(chm_real)  DXlang(*), DYlang(*), DZlang(*)
    real(chm_real)  VXlang(*), VYlang(*), VZlang(*)
    real(chm_real) :: DX(:), DY(:), DZ(:)

    integer nDrudes           ! IN  Number of Drude particles
    logical isDrude(*)        ! IN  (1..nAtoms)
    integer ISTEP

    logical QHOLO
    real(chm_real)  DXc(*)            ! I/O Constraint forces (1..nAtoms)
    real(chm_real)  DYc(*)
    real(chm_real)  DZc(*)
    real(chm_real)  lagrange(*),ISKPR(*)
    !integer ISKP(*)
    !!      integer BNBND(*)
    !!      integer BIMAG(*)
    type(nonbondDataStructure) BNBND
    type(imageDataStructure) BIMAG

    INTEGER NPRIV             ! Restart file was written at step NPRIV
    logical QNOSE             ! IN  .true. to use thermostats
    integer NOBL              ! IN  # of thermostats
    integer MAXNOS            ! IN
    real(chm_real)  NDGN(MAXNOS)      ! IN
    real(chm_real)  RTMPR(MAXNOS)     ! IN  Temperature of each thermostat (1..NOBL)
    real(chm_real)  SQM(MAXNOS)       ! IN
    real(chm_real)  SNH(MAXNOS), SNHV(MAXNOS), SNHF(MAXNOS)
    integer,dimension(:) :: INLCKP    ! IN

    logical QNHLANG(MAXNOS)   ! IN
    integer NDXlang           ! IN
    real(chm_real)  FBETA(*)          ! IN  FBETA parameter (1..MAXA)

    logical QRELT(MAXNOS)     ! IN  Relative thermostatting flags
    logical QQRELT            ! IN
    real(chm_real)  RELT_VX(MAXNOS)   ! IN
    real(chm_real)  RELT_VY(MAXNOS)   ! IN
    real(chm_real)  RELT_VZ(MAXNOS)   ! IN
    real(chm_real)  RELT_M(MAXNOS)    ! IN
    integer IABST             ! IN  Absolute thermostatting
    real(chm_real)  ABST_VX           ! IN
    real(chm_real)  ABST_VY           ! IN
    real(chm_real)  ABST_VZ           ! IN
    real(chm_real)  ABST_M            ! IN
    logical QNHLANGEVIN       ! INPUT
    real(chm_real)  NHGAMMA,NHGAMMAD           ! INPUT
    logical QCMLANGEVIN       ! INPUT
    real(chm_real)  CMGAMMA           ! INPUT

    logical QPCON, QDSCALE_CP
    real(chm_real)  W_CP, P_CP
    integer TI_CP
    real(chm_real)  ETA_CP(3,3), ZETA_CP(3,3), G_CP(3,3)
    real(chm_real)  RX1_CP(3,3), RX2_CP(3,3), RV1_CP(3,3), RV2_CP(3,3)

    logical QZONLY            ! IN  Flag for Z-scaling only
    logical QPCONFULL         ! IN  Flag for fully flexible cell
    real(chm_real)  DELTA_            ! IN  Timestep
    integer nc                ! IN  Thermostat timestep

    integer outu              ! IN  Output unit
    logical verbose           ! IN  .true. to get a more detailed output

    !
    !     Global variables
    !
    !
    !     Local variables
    !
    integer i, j
    integer ia
    integer ATFRST, ATLAST
    real(chm_real)  halfDelta
    real(chm_real)  halfDelta2

#if KEY_PARALLEL==1
    real(chm_real)  timmer
#endif

    integer gDrudes           ! Number of degrees of freedom associated to
    ! the Drudes
    !     Thermostats
    integer ti,tj                     ! thermostat index
    real(chm_real)  kin2_nh(MAXNOS)   ! "kin2" for each thermostat



    real(chm_real)  mcm,mcmi,mR_,md_
    real(chm_real)  Rx_,Ry_,Rz_, VRx,VRy,VRz, Rx_NEW,Ry_NEW,Rz_NEW
    real(chm_real)  dx_,dy_,dz_, Vdx,Vdy,Vdz, dx_NEW,dy_NEW,dz_NEW


    real(chm_real)  kin2              ! Two times the kinetic energy (mv2)
    real(chm_real)  fact,volp,kinn


    !     ROLL
    logical USE_ROLL, rolling
    integer iRoll, MAX_ROLL   ! maximum number of ROLL iterations
    real(chm_real)  RMSDshake         ! to monitor the convergence of SHAKE/ROLL
    real(chm_real)  RMSDrattle        ! to monitor the convergence of RATTLE/ROLL
    real(chm_real)  SNHroll(MAXNOS),SNHVroll(MAXNOS)
    real(chm_real)  ETA_CProll(3,3),ZETA_CProll(3,3)
    real(chm_real)  sf(6),XTLroll(6)
    real(chm_real)  VENEroll(9),TR_VENE_ROLL

    !     SCF dipoles
    logical SCFdipoles
    logical useThermostat(NOBL) !(MAXNOS)
    real(chm_real)  TOLSCF
    integer MAXSCF

    !     Langevin random force
    real(chm_real)  varlang,varlangD
    real(chm_real)  DXlangCM, DYlangCM, DZlangCM ! for the center of mass
    real(chm_real)  MT ! total mass of drude pair
    real(chm_real)  MU ! reduced mass of drude pair
    real(chm_real)  Randomx, Randomy, Randomz,RandomxD,RandomyD,RandomzD
    real(chm_real)  factA, factAD
    SAVE    DXlangCM, DYlangCM, DZlangCM

    real(chm_real), parameter :: OVERFOURTY2 = ONE/(THIRTY+TWELVE) ! 1/42
    real(chm_real), parameter :: OVERSVNTY2 = ONE/SVNTY2 ! 1/72
    !     Functions

#if KEY_DOMDEC==1 /*domdec*/
    if (q_domdec) then
       atfrst = 1
       atlast = natoml
    else
#endif /* (domdec)*/
#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
       ATFRST=1+IPARPT(MYNOD)
       ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
       ATFRST=1
       ATLAST=NATOMS
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
#else /* (paramain)*/
       ATFRST=1
       ATLAST=NATOMS
#endif /* (paramain)*/
#if KEY_DOMDEC==1
    endif
#endif

#if KEY_PARALLEL==1
    timmer=eclock()
#endif


    !
    !     Timesteps
    !
    halfDelta = HALF * DELTA_
    halfDelta2 = HALF * DELTA_*DELTA_

    !     Thermostat timestep
    if (.not.QNOSE) then
       nc = 1
    endif

    gDrudes = 3 * nDrudes
    if (QDSCALE_CP) gDrudes = 0

    !
    !     Save "ROLL" state
    !
    if (QHOLO) then
       if (QPCON) then
          USE_ROLL = .true.
          MAX_ROLL = 20       ! 5 is enough
       else
          USE_ROLL = .true.   ! .false.
          MAX_ROLL = 5        ! 1
       endif
    else
       USE_ROLL = .false.
    endif
    rolling = .false.
    iRoll = 0


    !==============================================================
    !
    !     Velocity-Verlet algorithm
    !
    !==============================================================

    !     Propagate Nose-Hoover variables 1
    if (QNOSE) then
       if (verbose) write(outu,*) 'DEBUG --- ', &
            'PropagateThermostatForParticles 1'
       CALL PropagateTFP( &
            nc,DELTA_, &
            QNOSE,NOBL,MAXNOS,INLCKP,useThermostat, &
            QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M,QCMLANGEVIN,CMGAMMA, &
            QNHLANG,FBETA,ISTEP, &
            SQM,SNH,SNHV,SNHF,NDGN,RTMPR, &
            TI_CP, &
            kin2,kin2_nh, &
            NDEGF,ATFRST,ATLAST, IMOVE,AMASS, &
            X,Y,Z, VX,VY,VZ, isDrude,nDrudes )

       if (QPCON .and. TI_CP > 0) THEN
          if (verbose) write(outu,*) 'DEBUG --- ', &
               'PropagateThermostatForBarostat 1'
          CALL PropagateTFB( &
               nc,DELTA_, &
               QNOSE,NOBL,MAXNOS,INLCKP,useThermostat, &
               QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
               IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
               QNHLANG,FBETA, &
               SQM,SNH,SNHV,SNHF,NDGN,RTMPR, &
               QPCON,QPCONFULL,W_CP,P_CP,TI_CP, &
               ETA_CP,ZETA_CP, &
               kin2,kin2_nh, &
               NDEGF,ATFRST,ATLAST, IMOVE,AMASS, &
               X,Y,Z, VX,VY,VZ, isDrude,nDrudes )
       endif
    endif


    if (verbose) write(OUTU,*) 'DEBUG -- Zero SHAKE forces'
    CALL ZeroArrays( DXc,DYc,DZc, ATFRST,ATLAST )


    !     Save pre-ROLL state
    if (USE_ROLL) then
       if (verbose) write(outu,*) 'DEBUG -- ', &
            'Save pre-ROLL state (except virial)'
       CALL CopyArrays( X,Y,Z, XOLD,YOLD,ZOLD, &
            ATFRST,ATLAST )
       CALL CopyArrays( VX,VY,VZ, Xtemp,Ytemp,Ztemp, &
            ATFRST,ATLAST )
       do ti = 1,NOBL
          SNHroll(ti) = SNH(ti)
          SNHVroll(ti) = SNHV(ti)
       enddo
       IF (QPCON) THEN
          CALL COPY33(ETA_CProll,ETA_CP)
          CALL COPY33(ZETA_CProll,ZETA_CP)
          CALL GETXTL(XTLroll,XTLABC,XTLTYP,XTLREF)
       ENDIF
    endif


    iRoll = 1

    !     SHAKE/ROLL restart point
484 continue

    if (verbose) write(outu,*) 'DEBUG -- ', &
         'SHAKE/ROLL iteration',iRoll,' ****'

    !     Retrieve pre-ROLL state
    if (USE_ROLL) then
       if (verbose) write(OUTU,*) 'DEBUG -- ', &
            'Retrieve pre-ROLL state (including virial)'
       CALL CopyArrays( XOLD,YOLD,ZOLD, &
            X,Y,Z, ATFRST,ATLAST )
       CALL CopyArrays( Xtemp,Ytemp,Ztemp, &
            VX,VY,VZ, ATFRST,ATLAST )
       do ti = 1,NOBL
          SNH(ti) = SNHroll(ti)
          SNHV(ti) = SNHVroll(ti)
       enddo
       IF (QPCON) THEN
          CALL COPY33(ETA_CP,ETA_CProll)
          CALL COPY33(ZETA_CP,ZETA_CProll)
          CALL PUTXTL(XTLroll,XTLABC,XTLTYP,XTLREF)
       ENDIF
       CALL GETVOL(EPROP(VOLUME))
       EPRESS(VIXX) = VENEroll(1)
       EPRESS(VIXY) = VENEroll(2)
       EPRESS(VIXZ) = VENEroll(3)
       EPRESS(VIYX) = VENEroll(4)
       EPRESS(VIYY) = VENEroll(5)
       EPRESS(VIYZ) = VENEroll(6)
       EPRESS(VIZX) = VENEroll(7)
       EPRESS(VIZY) = VENEroll(8)
       EPRESS(VIZZ) = VENEroll(9)
       EPROP(VIRI) = TR_VENE_ROLL

       !        Correct virial for SHAKE
       if (QHOLO .or. SCFdipoles) then
          if (verbose) write(OUTU,*) 'DEBUG -- ', &
               'Correct virial for SHAKE'
          CALL VIRSHK( EPRESS(VIXX:VIZZ), nAtoms, &
               X,Y,Z, DXc,DYc,DZc )
       endif
    endif


    !     Propagate barostat velocity 1
    if (QPCON) then
       if (verbose) write(outu,*) 'DEBUG -- ', &
            'Propagate barostat velocity 1'
       if (nDrudes > 0 .and. .not.QDSCALE_CP) then
          CALL CorrectVirial( 1, EPRESS(VIXX:VIZZ), &
               X,Y,Z, DX,DY,DZ, DXc,DYc,DZc, &
               ATFRST,ATLAST, AMASS,isDrude,IMOVE )
       endif
    endif
    if (QQRELT) then
       CALL RelativeTStat( &
            .TRUE.,.TRUE., QNOSE,NOBL,INLCKP, &
            QRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
            ATFRST,ATLAST,IMOVE,AMASS, &
            VX,VY,VZ, isDrude )
    endif
    CALL PRSEXT
    CALL PRSINT(nAtoms,AMASS,VX,VY,VZ,ONE,VX,VX,VX,ZERO)
    ! APH: Note VX, VX, VX here ------------|
    !      are just dummies
    if (QQRELT) then
       CALL RelativeTstat( &
            .FALSE.,.FALSE., QNOSE,NOBL,INLCKP, &
            QRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
            ATFRST,ATLAST,IMOVE,AMASS, &
            VX,VY,VZ, isDrude )
    endif
    if (QPCON) then
       CALL KineticEFNH( &
            QNOSE,NOBL,INLCKP, &
            QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
            kin2,kin2_nh, ATFRST,ATLAST, IMOVE,AMASS, &
            VX,VY,VZ, isDrude )
       volp = EPROP(VOLUME)*ATMOSP
       kinn = kin2/(NDEGF-gDrudes)
       fact = halfDelta/W_CP
       CALL IDENT33(G_CP)
       IF (QPCONFULL) THEN
          ! To eliminate cell rotations, we use (Pab + Pba)/2 instead of Pab.
          G_CP(1,1) = volp* EPRESS(PIXX)
          G_CP(1,2) = volp*(EPRESS(PIXY)+EPRESS(PIYX))*HALF
          G_CP(1,3) = volp*(EPRESS(PIXZ)+EPRESS(PIZX))*HALF
          G_CP(2,1) = volp*(EPRESS(PIYX)+EPRESS(PIXY))*HALF
          G_CP(2,2) = volp* EPRESS(PIYY)
          G_CP(2,3) = volp*(EPRESS(PIYZ)+EPRESS(PIZY))*HALF
          G_CP(3,1) = volp*(EPRESS(PIZX)+EPRESS(PIXZ))*HALF
          G_CP(3,2) = volp*(EPRESS(PIZY)+EPRESS(PIYZ))*HALF
          G_CP(3,3) = volp* EPRESS(PIZZ)
          CALL MatchXTL(G_CP,XTLTYP,XTLREF)
          G_CP(1,1) = G_CP(1,1) - volp*P_CP + kinn
          G_CP(2,2) = G_CP(2,2) - volp*P_CP + kinn
          G_CP(3,3) = G_CP(3,3) - volp*P_CP + kinn
       ELSE
          IF (QZONLY) THEN
             G_CP(1,1) = ZERO
             G_CP(2,2) = ZERO
             G_CP(3,3) = volp*(EPRESS(PIZZ)-P_CP) + kinn
          ELSE
             G_CP(1,1) = THREE*(volp*(EPROP(PRESSI)-P_CP) + kinn)
             G_CP(2,2) = G_CP(1,1)
             G_CP(3,3) = G_CP(1,1)
          ENDIF
       ENDIF
       CALL AFFINE33(ZETA_CP,ZETA_CP,fact,G_CP)

       if (nDrudes > 0 .and. .not.QDSCALE_CP) then
          CALL CorrectVirial( 0, EPRESS(VIXX:VIZZ), &
               X,Y,Z, DX,DY,DZ, DXc,DYc,DZc, &
               ATFRST,ATLAST, AMASS,isDrude,IMOVE )
       endif
    endif

    !     Add -ve Langevin friction to -ve force prime (NHGAMMA is KT/mD)
    !     (E. Harder 2004)
    if (QNHLANGEVIN) then
       do ia = ATFRST,ATLAST
#if KEY_DOMDEC==1
          if (q_domdec) then
             i = atoml(ia)
          else
#endif
             i = ia
#if KEY_DOMDEC==1
          endif
#endif
          if (isDrude(i)) goto 602
          ti = INLCKP(i)
          j = i + 1
          tj = INLCKP(j)
          if (QNHLANG(ti).and.isDrude(j)) then
             MT = amass(i) + amass(j)
             MU = amass(i)*amass(j)/MT
             VXlang(i) = amass(i)*VX(i)/MT + amass(j)*VX(j)/MT
             VYlang(i) = amass(i)*VY(i)/MT + amass(j)*VY(j)/MT
             VZlang(i) = amass(i)*VZ(i)/MT + amass(j)*VZ(j)/MT
             VXlang(j) = VX(j) - VX(i)
             VYlang(j) = VY(j) - VY(i)
             VZlang(j) = VZ(j) - VZ(i)

             DXlang(i) = DXsave(i) + NHGAMMA*VXlang(i)*MT
             DYlang(i) = DYsave(i) + NHGAMMA*VYlang(i)*MT
             DZlang(i) = DZsave(i) + NHGAMMA*VZlang(i)*MT
             IF (QNHLANG(tj)) THEN
                DXlang(j) = DXsave(j) + NHGAMMAD*VXlang(j)*MU
                DYlang(j) = DYsave(j) + NHGAMMAD*VYlang(j)*MU
                DZlang(j) = DZsave(j) + NHGAMMAD*VZlang(j)*MU
             ELSE
                DXlang(j) = DXsave(j)
                DYlang(j) = DYsave(j)
                DZlang(j) = DZsave(j)
             ENDIF
          elseif (QNHLANG(ti)) then
             DXlang(i) = DXsave(i) + NHGAMMA*VX(i)*amass(i)
             DYlang(i) = DYsave(i) + NHGAMMA*VY(i)*amass(i)
             DZlang(i) = DZsave(i) + NHGAMMA*VZ(i)*amass(i)
             VXlang(i) = VX(i)
             VYlang(i) = VY(i)
             VZlang(i) = VZ(i)
          else
             DXlang(i) = DX(i)
             DYlang(i) = DY(i)
             DZlang(i) = DZ(i)
             VXlang(i) = VX(i)
             VYlang(i) = VY(i)
             VZlang(i) = VZ(i)
          endif
602       CONTINUE
       enddo
    endif

    !     Convert from R,d to ri and rD and convert forces
    if (QNHLANGEVIN) then
       do ia = ATFRST,ATLAST
#if KEY_DOMDEC==1
          if (q_domdec) then
             i = atoml(ia)
          else
#endif
             i = ia
#if KEY_DOMDEC==1
          endif
#endif
          if (isDrude(i)) goto 510
          j = i + 1
          ti = INLCKP(i)
          if (QNHLANG(ti).and.isDrude(j)) then
             MT = amass(i) + amass(j)
             VX(i) = VXlang(i) - amass(j)*VXlang(j)/MT
             VY(i) = VYlang(i) - amass(j)*VYlang(j)/MT
             VZ(i) = VZlang(i) - amass(j)*VZlang(j)/MT
             VX(j) = VXlang(i) + amass(i)*VXlang(j)/MT
             VY(j) = VYlang(i) + amass(i)*VYlang(j)/MT
             VZ(j) = VZlang(i) + amass(i)*VZlang(j)/MT
             DX(i) = DXlang(i)*amass(i)/MT - DXlang(j)
             DY(i) = DYlang(i)*amass(i)/MT - DYlang(j)
             DZ(i) = DZlang(i)*amass(i)/MT - DZlang(j)
             DX(j) = DXlang(i)*amass(j)/MT + DXlang(j)
             DY(j) = DYlang(i)*amass(j)/MT + DYlang(j)
             DZ(j) = DZlang(i)*amass(j)/MT + DZlang(j)
          else
             DX(i) = DXlang(i)
             DY(i) = DYlang(i)
             DZ(i) = DZlang(i)
             VX(i) = VXlang(i)
             VY(i) = VYlang(i)
             VZ(i) = VZlang(i)
          endif
510       continue
       enddo
    endif

    !     Propagate velocities 1
    if (verbose) write(outu,*) 'DEBUG -- Velocities 1'

    if (QQRELT) then
       CALL RelativeTstat( &
            .TRUE.,.TRUE., QNOSE,NOBL,INLCKP, &
            QRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
            ATFRST,ATLAST,IMOVE,AMASS, &
            VX,VY,VZ, isDrude )
    endif

    if (QPCON) then
       CALL ComputeRR(ZETA_CP,-HALF*DELTA_,NDEGF-gDrudes,QPCONFULL, &
            RV1_CP,RV2_CP,xtlabc)
       CALL PropagateVCP( VX,VY,VZ, DX,DY,DZ, &
            ATFRST,ATLAST, isDrude, &
            halfDelta,AMASS,IMOVE, &
                                !    $        QNOSE,NOBL,INLCKP,
                                !    $        NDXlang, DXlang, DYlang, DZlang, FBETA,
                                !    $        QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M,
                                !    $        IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M,
            QZONLY,QPCONFULL, &
            RV1_CP,RV2_CP )
    else
       CALL PropagateVelo( VX,VY,VZ, DX,DY,DZ, &
            ATFRST,ATLAST, &
            halfDelta,AMASS,IMOVE )
    endif

    !     Propagate CM velocity
    if (IABST > 0) then
       if (QCMLANGEVIN) then
          fact = halfDelta/ABST_M
          ABST_VX = ABST_VX + fact*DXlangCM
          ABST_VY = ABST_VY + fact*DYlangCM
          ABST_VZ = ABST_VZ + fact*DZlangCM
       endif
    endif

    if (QQRELT) then
       CALL RelativeTstat( &
            .FALSE.,.FALSE., QNOSE,NOBL,INLCKP, &
            QRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
            ATFRST,ATLAST,IMOVE,AMASS, &
            VX,VY,VZ, &
            isDrude )
    endif


    !     Keep unpropagated positions in XOLD,YOLD,ZOLD
    !     We need this to compute the SCF velocities.

    if (SCFdipoles) then
       CALL CopyArrays( X,Y,Z, XOLD,YOLD,ZOLD, ATFRST,ATLAST )
    endif

    !     Propagate positions
    if (verbose) write(OUTU,*) 'DEBUG -- Positions'
    CALL CopyArrays( X,Y,Z, XNEW,YNEW,ZNEW,  & ! For non-moving atoms
         ATFRST,ATLAST )

    if (QPCON) then
       CALL ComputeRR(ZETA_CP,DELTA_,0,QPCONFULL, &
            RX1_CP,RX2_CP,XTLABC)

       do ia = atfrst,atlast
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
          if (JPBLOCK(I) /= MYNOD) goto 708
#endif
          if (IMOVE(I) /= 0) goto 708
          if (ia /= ATLAST .and. isDrude(I+1)) then ! New scaling...
             !              (r,rD) -> (R,d)
             mcm = AMASS(I) + AMASS(I+1)
             mcmi = ONE / mcm
             mR_ = AMASS(I) * mcmi
             md_ = AMASS(I+1) * mcmi
             Rx_ = mR_*X(I) + md_*X(I+1)
             Ry_ = mR_*Y(I) + md_*Y(I+1)
             Rz_ = mR_*Z(I) + md_*Z(I+1)
             VRx = mR_*VX(I) + md_*VX(I+1)
             VRy = mR_*VY(I) + md_*VY(I+1)
             VRz = mR_*VZ(I) + md_*VZ(I+1)
             dx_ = X(I+1) - X(I)
             dy_ = Y(I+1) - Y(I)
             dz_ = Z(I+1) - Z(I)
             Vdx = VX(I+1) - VX(I)
             Vdy = VY(I+1) - VY(I)
             Vdz = VZ(I+1) - VZ(I)
             !              Propagate/Scale R
             CALL ApplyRR( Rx_NEW,Ry_NEW,Rz_NEW, &
                  RX1_CP,Rx_,Ry_,Rz_, &
                  DELTA_,RX2_CP,VRx,VRy,VRz )
             !              Propagate/Scale d
             if (QDSCALE_CP) then
                CALL ApplyRR( dx_NEW,dy_NEW,dz_NEW, &
                     RX1_CP,dx_,dy_,dz_, &
                     DELTA_,RX2_CP,Vdx,Vdy,Vdz )
             else
                dx_NEW = dx_ + DELTA_*Vdx
                dy_NEW = dy_ + DELTA_*Vdy
                dz_NEW = dz_ + DELTA_*Vdz
             endif
             !              (R,d) -> (r,rD)
             XNEW(I) = Rx_NEW - md_ * dx_NEW
             YNEW(I) = Ry_NEW - md_ * dy_NEW
             ZNEW(I) = Rz_NEW - md_ * dz_NEW
          elseif (isDrude(I)) then ! New scaling...
             XNEW(I) = Rx_NEW + (ONE - md_) * dx_NEW
             YNEW(I) = Ry_NEW + (ONE - md_) * dy_NEW
             ZNEW(I) = Rz_NEW + (ONE - md_) * dz_NEW
          else                ! Original scaling...
             CALL ApplyRR( XNEW(I),YNEW(I),ZNEW(I), &
                  RX1_CP,X(I),Y(I),Z(I), &
                  DELTA_,RX2_CP,VX(I),VY(I),VZ(I) )
          endif
708       CONTINUE
       enddo

       !        Warnings before trying to transform the crystal
       IF (QZONLY) THEN
          IF (XTLTYP == 'CUBI') THEN
             CALL WRNDIE(0,'<VV2>','Cannot use ZONLY with CUBI crystal')
          ELSEIF (XTLTYP == 'RECT') THEN
             CALL WRNDIE(0,'<VV2>','Cannot use ZONLY with RECT crystal')
          ELSEIF (XTLTYP == 'RHDO') THEN
             CALL WRNDIE(0,'<VV2>','Cannot use ZONLY with RHDO crystal')
          ELSEIF (XTLTYP == 'OCTA') THEN
             CALL WRNDIE(0,'<VV2>','Cannot use ZONLY with OCTA crystal')
          ELSEIF (XTLTYP == 'RHOM') THEN
             CALL WRNDIE(0,'<VV2>','Cannot use ZONLY with RHOM crystal')
          ENDIF
       ENDIF
       CALL GETVOL(EPROP(VOLUME))
       CALL TransfXTL(RX1_CP,XTLABC,XTLTYP,XTLREF,XUCELL,XDIM)
       CALL GETXTL(sf,XTLABC,XTLTYP,XTLREF)
       CALL PUTXTL(sf,XTLABC,XTLTYP,XTLREF)
       CALL GETVOL(EPROP(VOLUME))
    else
#if KEY_DOMDEC==1
       if (q_domdec) then
          do ia = atfrst,atlast
             i = atoml(ia)
             if (imove(i) /= 0) cycle
             xnew(i) = x(i) + delta_*vx(i)
             ynew(i) = y(i) + delta_*vy(i)
             znew(i) = z(i) + delta_*vz(i)
          enddo
       else
#endif
          do I = ATFRST,ATLAST
#if KEY_PARASCAL==1
             if (JPBLOCK(I) /= MYNOD) goto 709
#endif
             if (IMOVE(I) /= 0) goto 709
             XNEW(I) = X(I) + DELTA_*VX(I)
             YNEW(I) = Y(I) + DELTA_*VY(I)
             ZNEW(I) = Z(I) + DELTA_*VZ(I)
709          CONTINUE
          enddo
#if KEY_DOMDEC==1
       endif
#endif
    endif

    ! impose hard wall constraint on drude bond length
    CALL HardWallDrude(XNEW,YNEW,ZNEW,VX,VY,VZ,ATFRST,ATLAST,AMASS,  &
      isDrude,IMOVE,EPRESS(VIXX:VIZZ),DELTA_)

    !     Keep an un-SHAKEd (and/or un-SCFed) copy of the positions
    if (QHOLO .or. SCFdipoles) then
       if (verbose) write(OUTU,*) 'DEBUG -- ', &
            'Keep unSHAKED coordinates'
       CALL CopyArrays( XNEW,YNEW,ZNEW, Xsave,Ysave,Zsave, &
            ATFRST,ATLAST )
    endif

    if (QHOLO) then
       if (verbose) write(outu,*) 'DEBUG -- SHAKE'

       CALL ShakeRoll( &
            ATFRST,ATLAST, &
            X,Y,Z, XNEW,YNEW,ZNEW, VX,VY,VZ, &
            AMASS,IMOVE, DELTA_, .TRUE., &
            lagrange, DXc,DYc,DZc, ISKPR, &
            QPCON,QZONLY,QPCONFULL,ZETA_CP,W_CP,RX1_CP,RX2_CP,RV2_CP)

       CALL ShakeVelo( XNEW,YNEW,ZNEW, Xsave,Ysave,Zsave, &
            VX,VY,VZ, &
            ATFRST,ATLAST, DELTA_ )

       CALL ShakeForces( XNEW,YNEW,ZNEW, Xsave,Ysave,Zsave, &
            DXc,DYc,DZc, &
            ATFRST,ATLAST, DELTA_,AMASS, &
            RMSDshake )

       if (verbose) write(outu,*) 'DEBUG -- RMSDshake',RMSDshake
    endif

    CALL CopyArrays( XNEW,YNEW,ZNEW, X,Y,Z, ATFRST,ATLAST )

#if KEY_DOMDEC==1
    if (q_domdec) then
       call transfer_coord(x, y, z, .false.)
       call update_local_coord(x, y, z)
    else
#endif
#if KEY_PARALLEL==1
       CALL VDGBR(X,Y,Z,0)
#endif
#if KEY_DOMDEC==1
    endif
#endif

    !     Go back for a SHAKE/ROLL iteration
    if (QHOLO) then
       if (iRoll <= MAX_ROLL .and. RMSDshake > TENM8) then
          iRoll = iRoll + 1
          rolling = .true.
          goto 484
       elseif (iRoll > MAX_ROLL) then
          write(outu,*) 'VV2 -- ', &
               'SHAKE/ROLL did NOT converge!'
          write(outu,*) 'VV2 -- RMSDshake',RMSDshake
          write(outu,*) 'VV2 -- (Try increasing BTAU)'
          rolling = .false.
          iRoll = 0
       else
          if (verbose) write(outu,*) 'DEBUG -- ', &
               'SHAKE/ROLL converged ********'
          rolling = .false.
          iRoll = 0
       endif
    endif

    !     Propagate barostat
    if (QPCON) then
       if (verbose) write(outu,*) 'DEBUG -- Propagate barostat'
       CALL AFFINE33(ETA_CP,ETA_CP,DELTA_,ZETA_CP)
    endif

    !     (E. Harder 2005)
    !     Compute SCF dipoles
    if (SCFdipoles) then
       if (verbose) write(outu,*) 'DEBUG -- SCF positions'
       CALL OptimizeDrude( TOLSCF,MAXSCF, &
            XOLD,YOLD,ZOLD, VX,VY,VZ, BNBND,BIMAG, &
            DX,DY,DZ, .TRUE. )
#if KEY_DOMDEC==1
       if (q_domdec) then
          call transfer_coord(x, y, z, .false.)
          call update_local_coord(x, y, z)
       else
#endif
#if KEY_PARALLEL==1
          CALL VDGBR(X,Y,Z,0)
#endif
#if KEY_DOMDEC==1
       endif
#endif
    endif
    !      langstep = ISTEP/1000
    !      langstep = langstep*1000
    !      if (langstep == ISTEP) then
    !         CALL CopyArrays( X,Y,Z, Xsave,Ysave,Zsave,
    !     $            ATFRST,ATLAST )
    !         CALL CopyArrays( VX,VY,VZ, VXsave,VYsave,VZsave,
    !     $            ATFRST,ATLAST )
    !         CALL OptimizeDrude( TOLSCF,MAXSCF,
    !     $        XOLD,YOLD,ZOLD, VX,VY,VZ, BNBND,BIMAG,
    !     $        DX,DY,DZ, .TRUE. )
    !      MT = ZERO
    !      do i=ATFRST,ATLAST
    !      if (isDrude(i)) then
    !      MT = MT + SQRT((X(i) - Xsave(i))*(X(i) - Xsave(i)) +
    !     $             (Y(i) - Ysave(i))*(Y(i) - Ysave(i)) +
    !     $             (Z(i) - Zsave(i))*(Z(i) - Zsave(i)))/nDrudes
    !      endif
    !      continue
    !      enddo
    !      PRINT *,"rMSD=",ISTEP,MT
    !         CALL CopyArrays( VXsave,VYsave,VZsave, VX,VY,VZ,
    !     $            ATFRST,ATLAST )
    !         CALL CopyArrays( Xsave,Ysave,Zsave, X,Y,Z,
    !     $            ATFRST,ATLAST )
    !      endif

#if KEY_PARALLEL==1
    tmeri(timdcntrl)=tmeri(timdcntrl)+eclock()-timmer
#endif
    !     Compute forces
    if (verbose) write(outu,*) 'DEBUG -- Compute forces'
    QDYNCALL=.TRUE.
    CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
    QDYNCALL=.FALSE.
    CALL GRAM( &
#if KEY_MTS==1
         DX,DY,DZ,DX,DY,DZ,    &
#endif
         DX,DY,DZ)
#if KEY_PARALLEL==1
    timmer=eclock()
#endif
#if KEY_SGLD==1
     !  WXW    perform force average
     IF(QSGLD.OR.QSGMD)THEN
        CALL SGFAVG(ATFRST,ATLAST,EPROP(EPOT),IMOVE, &
             DX,DY,DZ)
     ENDIF
#endif


    !     Propagate velocities 2
    if (verbose) write(outu,*) 'DEBUG -- Velocities 2'

    if (QQRELT) then
       CALL RelativeTstat( &
            .TRUE.,.TRUE., QNOSE,NOBL,INLCKP, &
            QRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
            ATFRST,ATLAST,IMOVE,AMASS, &
            VX,VY,VZ, isDrude )
    endif


    !     Add -ve Langevin stochastic force to get -force_prime:
    !     (E. Harder 2004)
    if (QNHLANGEVIN) then
       do ia = ATFRST,ATLAST
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
          if (JPBLOCK(i) /= MYNOD) goto 603
#endif
          if (IMOVE(i) /= 0) goto 603
          if (isDrude(i)) goto 603
          ti = INLCKP(i)
          j = i + 1
          tj = INLCKP(j)
          if (QNHLANG(ti).and.isDrude(j)) then
             MT = amass(i) + amass(j)
             MU = amass(i)*amass(j)/MT
             varlang = SQRT( TWO*MT* &
                  NHGAMMA*KBOLTZ*RTMPR(ti)/(DELTA_) )
             IF (QNHLANG(tj)) THEN
                varlangD = SQRT(TWO*MU*NHGAMMAD*KBOLTZ*RTMPR(tj)/(DELTA_))
             ENDIF
             Randomx = -BMGAUS(varlang,ISEED)
             Randomy = -BMGAUS(varlang,ISEED)
             Randomz = -BMGAUS(varlang,ISEED)
             IF (QNHLANG(tj)) THEN
                RandomxD = -BMGAUS(varlangD,ISEED)
                RandomyD = -BMGAUS(varlangD,ISEED)
                RandomzD = -BMGAUS(varlangD,ISEED)
             ENDIF
             DXlang(i) =  DX(i) + DX(j) + Randomx
             DYlang(i) =  DY(i) + DY(j) + Randomy
             DZlang(i) =  DZ(i) + DZ(j) + Randomz
             IF (QNHLANG(tj)) THEN
                DXlang(j) = DX(j)*amass(i)/MT - DX(i)*amass(j)/MT + RandomxD
                DYlang(j) = DY(j)*amass(i)/MT - DY(i)*amass(j)/MT + RandomyD
                DZlang(j) = DZ(j)*amass(i)/MT - DZ(i)*amass(j)/MT + RandomzD
             ELSE
                DXlang(j) = DX(j)*amass(i)/MT - DX(i)*amass(j)/MT
                DYlang(j) = DY(j)*amass(i)/MT - DY(i)*amass(j)/MT
                DZlang(j) = DZ(j)*amass(i)/MT - DZ(i)*amass(j)/MT
             ENDIF
          elseif (QNHLANG(ti)) then
             varlang = SQRT( TWO*amass(i)* &
                  NHGAMMA*KBOLTZ*RTMPR(ti)/(DELTA_) )
             DXlang(i) = DX(i) - BMGAUS(varlang,ISEED)
             DYlang(i) = DY(i) - BMGAUS(varlang,ISEED)
             DZlang(i) = DZ(i) - BMGAUS(varlang,ISEED)
          endif
603       continue
       enddo
#if KEY_SGLD==1
     !WXW    considering the guiding effect
     !write(OUTU,'(A,1x,L4,1x,20E10.2)')"DEBUGF1 ",QSGNONET,(DXlang(ia),ia=1,10)
     
     IF(QSGLD)THEN
           CALL SGLDG(ATFRST,ATLAST,DELTA_,NDEGF, &
               INLCKP,QNHLANG,isDrude,NHGAMMA,NHGAMMAD,  &
                IMOVE,AMASS,X,Y,Z,VX,VY,VZ,DXlang,DYlang,DZlang)
     ENDIF
     !write(OUTU,'(A,1x,L4,1x,20E10.2)')"DEBUGF2 ",QSGNONET,(DXlang(ia),ia=1,10)
#endif

       !        SAVE COPY OF FORCE PRIME
       CALL CopyArrays( DXlang,DYlang,DZlang, DXsave,DYsave,DZsave, &
            ATFRST,ATLAST )

       !        CONVERT TO R,d AND SCALE FORCE AND VELOCITY
       !        (E. Harder 2004)
       do ia = ATFRST,ATLAST
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
          if (JPBLOCK(i) /= MYNOD) goto 604
#endif
          if (IMOVE(i) /= 0) goto 604
          if (isDrude(i)) goto 604
          ti = INLCKP(i)
          j = i + 1
          tj = INLCKP(j)
          if (QNHLANG(ti).and.isDrude(j)) then
             MT = amass(i) + amass(j)
             MU = amass(i)*amass(j)/MT
             factA = ONE + NHGAMMA*halfDelta
             factAD = ONE + NHGAMMAD*halfDelta
             VXlang(i) = amass(i)*VX(i)/MT/factA +  &
                  amass(j)*VX(j)/MT/factA
             VYlang(i) = amass(i)*VY(i)/MT/factA +  &
                  amass(j)*VY(j)/MT/factA
             VZlang(i) = amass(i)*VZ(i)/MT/factA +  &
                  amass(j)*VZ(j)/MT/factA
             IF (QNHLANG(tj)) THEN
                VXlang(j) = (VX(j) - VX(i))/factAD
                VYlang(j) = (VY(j) - VY(i))/factAD
                VZlang(j) = (VZ(j) - VZ(i))/factAD
             ELSE
                VXlang(j) = VX(j) - VX(i)
                VYlang(j) = VY(j) - VY(i)
                VZlang(j) = VZ(j) - VZ(i)
             ENDIF
             DXlang(i) = DXsave(i)/factA
             DYlang(i) = DYsave(i)/factA
             DZlang(i) = DZsave(i)/factA
             IF (QNHLANG(tj)) THEN
                DXlang(j) = DXsave(j)/factAD
                DYlang(j) = DYsave(j)/factAD
                DZlang(j) = DZsave(j)/factAD
             ELSE
                DXlang(j) = DXsave(j)
                DYlang(j) = DYsave(j)
                DZlang(j) = DZsave(j)
             ENDIF
          elseif (QNHLANG(ti)) then
             factA = ONE + NHGAMMA*halfDelta
             VXlang(i) = VX(i)/factA
             VYlang(i) = VY(i)/factA
             VZlang(i) = VZ(i)/factA
             DXlang(i) = DXsave(i)/factA
             DYlang(i) = DYsave(i)/factA
             DZlang(i) = DZsave(i)/factA
          else
             VXlang(i) = VX(i)
             VYlang(i) = VY(i)
             VZlang(i) = VZ(i)
             DXlang(i) = DXsave(i)
             DYlang(i) = DYsave(i)
             DZlang(i) = DZsave(i)
          endif
604       continue
       enddo
    endif
    !     Convert from R,d to ri and rD
    if (QNHLANGEVIN) then
       do ia = ATFRST,ATLAST
#if KEY_DOMDEC==1
          if (q_domdec) then
             i = atoml(ia)
          else
#endif
             i = ia
#if KEY_DOMDEC==1
          endif
#endif
          if (isDrude(i)) goto 511
          j = i + 1
          ti = INLCKP(i)
          if (QNHLANG(ti).and.isDrude(j)) then
             MT = amass(i) + amass(j)
             VX(i) = VXlang(i) - amass(j)*VXlang(j)/MT
             VY(i) = VYlang(i) - amass(j)*VYlang(j)/MT
             VZ(i) = VZlang(i) - amass(j)*VZlang(j)/MT
             VX(j) = VXlang(i) + amass(i)*VXlang(j)/MT
             VY(j) = VYlang(i) + amass(i)*VYlang(j)/MT
             VZ(j) = VZlang(i) + amass(i)*VZlang(j)/MT
             DX(i) = DXlang(i)*amass(i)/MT - DXlang(j)
             DY(i) = DYlang(i)*amass(i)/MT - DYlang(j)
             DZ(i) = DZlang(i)*amass(i)/MT - DZlang(j)
             DX(j) = DXlang(i)*amass(j)/MT + DXlang(j)
             DY(j) = DYlang(i)*amass(j)/MT + DYlang(j)
             DZ(j) = DZlang(i)*amass(j)/MT + DZlang(j)
          else
             DX(i) = DXlang(i)
             DY(i) = DYlang(i)
             DZ(i) = DZlang(i)
             VX(i) = VXlang(i)
             VY(i) = VYlang(i)
             VZ(i) = VZlang(i)
          endif
511       continue
       enddo
    endif
#if KEY_SGLD==1
     !WXW    considering the guiding effect
     IF(QSGMD)THEN
           CALL SGMDG(ATFRST,ATLAST,DELTA_,NDEGF,MAXNOS,INLCKP,RTMPR,SNHV,&
                IMOVE,AMASS,X,Y,Z,VX,VY,VZ,DX,DY,DZ)
     ENDIF
#endif

    if (QPCON) then
       CALL PropagateVCP( VX,VY,VZ, DX,DY,DZ, &
            ATFRST,ATLAST, isDrude, &
            halfDelta,AMASS,IMOVE, &
                                !    $        QNOSE,NOBL,INLCKP,
                                !    $        NDXlang, DXlang, DYlang, DZlang, FBETA,
                                !    $        QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M,
                                !    $        IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M,
            QZONLY,QPCONFULL, &
            RV1_CP,RV2_CP )
    else
       CALL PropagateVelo( VX,VY,VZ, DX,DY,DZ, &
            ATFRST,ATLAST, &
            halfDelta,AMASS,IMOVE )
    endif


    if (QHOLO) then
       if (verbose) write(outu,*) 'DEBUG -- Zero RATTLE forces'
       CALL ZeroArrays( DXc,DYc,DZc, ATFRST,ATLAST )
    endif


    !     Save pre-ROLL state
    if (USE_ROLL) then
       if (verbose) write(outu,*) 'DEBUG -- ', &
            'Save pre-ROLL state (including virial)'
       CALL CopyArrays( X,Y,Z, XOLD,YOLD,ZOLD, &
            ATFRST,ATLAST )
       CALL CopyArrays( VX,VY,VZ, Xtemp,Ytemp,Ztemp, &
            ATFRST,ATLAST )
       CALL COPY33(ETA_CProll,ETA_CP)
       CALL COPY33(ZETA_CProll,ZETA_CP)
       IF (QPCON) CALL GETXTL(XTLroll,XTLABC,XTLTYP,XTLREF)
       VENEroll(1) = EPRESS(VIXX)
       VENEroll(2) = EPRESS(VIXY)
       VENEroll(3) = EPRESS(VIXZ)
       VENEroll(4) = EPRESS(VIYX)
       VENEroll(5) = EPRESS(VIYY)
       VENEroll(6) = EPRESS(VIYZ)
       VENEroll(7) = EPRESS(VIZX)
       VENEroll(8) = EPRESS(VIZY)
       VENEroll(9) = EPRESS(VIZZ)
       TR_VENE_ROLL = EPROP(VIRI)
    endif

    iRoll = 1

    !     RATTLE/ROLL restart point
848 continue

    if (verbose) WRITE(OUTU,*) 'DEBUG -- ', &
         'RATTLE/ROLL iteration',iRoll,' ****'


    !     Retrieve pre-ROLL state
    if (USE_ROLL) then
       if (verbose) write(outu,*) 'DEBUG -- ', &
            'Retrieve pre-ROLL state (including virial)'
       CALL CopyArrays( XOLD,YOLD,ZOLD, &
            X,Y,Z, ATFRST,ATLAST )
       CALL CopyArrays( Xtemp,Ytemp,Ztemp, &
            VX,VY,VZ, ATFRST,ATLAST )
       CALL COPY33(ETA_CP,ETA_CProll)
       CALL COPY33(ZETA_CP,ZETA_CProll)
       IF (QPCON) CALL PUTXTL(XTLroll,XTLABC,XTLTYP,XTLREF)
       CALL GETVOL(EPROP(VOLUME))
       EPRESS(VIXX) = VENEroll(1)
       EPRESS(VIXY) = VENEroll(2)
       EPRESS(VIXZ) = VENEroll(3)
       EPRESS(VIYX) = VENEroll(4)
       EPRESS(VIYY) = VENEroll(5)
       EPRESS(VIYZ) = VENEroll(6)
       EPRESS(VIZX) = VENEroll(7)
       EPRESS(VIZY) = VENEroll(8)
       EPRESS(VIZZ) = VENEroll(9)
       EPROP(VIRI) = TR_VENE_ROLL
    endif

    !     Keep an un-RATTLEd copy of the velocities
    if (QHOLO) then
       if (verbose) write(outu,*) 'DEBUG -- ', &
            'Keep unRATTLED velocities'
       CALL CopyArrays( VX,VY,VZ, VXsave,VYsave,VZsave, &
            ATFRST,ATLAST )


       if (verbose) write(outu,*) 'DEBUG -- RATTLE'

       CALL RattleRoll( &
            ATFRST,ATLAST, &
            X,Y,Z, VX,VY,VZ, &
            AMASS,IMOVE, DELTA_, .TRUE., &
            lagrange, ISKPR, &
            QPCON,QZONLY,QPCONFULL,ZETA_CP,W_CP, RV2_CP, &
            XTLTYP,XTLREF )

       CALL RattleForces( VX,VY,VZ, VXsave,VYsave,VZsave, &
            DXc,DYc,DZc, &
            ATFRST,ATLAST, DELTA_,AMASS, &
            RMSDrattle )

       if (verbose) write(outu,*) 'DEBUG -- RMSDrattle',RMSDrattle
    endif

    !     Propagate CM velocity
    if (IABST > 0) then
       if (QCMLANGEVIN) then
          fact = halfDelta/ABST_M
          varlang = SQRT( TWO*CMGAMMA*KBOLTZ*RTMPR(IABST) &
               * ABST_M/DELTA_ )
          DXlangCM = BMGAUS(varlang,ISEED)
          DYlangCM = BMGAUS(varlang,ISEED)
          DZlangCM = BMGAUS(varlang,ISEED)
          ABST_VX = ABST_VX + fact*DXlangCM
          ABST_VY = ABST_VY + fact*DYlangCM
          ABST_VZ = ABST_VZ + fact*DZlangCM
       endif
    endif

    if (QQRELT) then
       CALL RelativeTstat( &
            .FALSE.,.FALSE., QNOSE,NOBL,INLCKP, &
            QRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
            ATFRST,ATLAST,IMOVE,AMASS, &
            VX,VY,VZ, isDrude )
    endif


    !     Propagate barostat velocity 2
    if (QPCON) then
       if (verbose) write(OUTU,*) 'DEBUG -- ', &
            'Correct virial for RATTLE'
       CALL VIRSHK( EPRESS(VIXX:VIZZ), nAtoms, &
            X,Y,Z, DXc,DYc,DZc )
       CALL GETVOL(EPROP(VOLUME))
       if (verbose) write(OUTU,*) 'DEBUG -- ', &
            'Propagate barostat velocity 2'
       if (nDrudes > 0 .and. .not.QDSCALE_CP) then
          CALL CorrectVirial( 1, EPRESS(VIXX:VIZZ), &
               X,Y,Z, DX,DY,DZ, &
               DXc,DYc,DZc, &
               ATFRST,ATLAST, AMASS,isDrude,IMOVE )
       endif
    endif
    if (QQRELT) then
       CALL RelativeTstat( &
            .TRUE.,.TRUE., QNOSE,NOBL,INLCKP, &
            QRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
            ATFRST,ATLAST,IMOVE,AMASS, &
            VX,VY,VZ, isDrude )
    endif
    CALL PRSEXT
    CALL PRSINT(nAtoms,AMASS,VX,VY,VZ,ONE,VX,VX,VX,ZERO)
    ! APH: Note VX, VX, VX here ------------|
    !      are just dummies
    if (QQRELT) then
       CALL RelativeTstat( &
            .FALSE.,.FALSE., QNOSE,NOBL,INLCKP, &
            QRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
            ATFRST,ATLAST,IMOVE,AMASS, &
            VX,VY,VZ, isDrude )
    endif
    if (QPCON) then
       CALL KineticEFNH( &
            QNOSE,NOBL,INLCKP, &
            QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
            kin2,kin2_nh, ATFRST,ATLAST, IMOVE,AMASS, &
            VX,VY,VZ, isDrude )
       volp = EPROP(VOLUME)*ATMOSP
       kinn = kin2/(NDEGF-gDrudes)
       fact = halfDelta/W_CP
       CALL IDENT33(G_CP)
       IF (QPCONFULL) THEN
          ! To eliminate cell rotations, we use (Pab + Pba)/2 instead of Pab.
          G_CP(1,1) = volp* EPRESS(PIXX)
          G_CP(1,2) = volp*(EPRESS(PIXY)+EPRESS(PIYX))*HALF
          G_CP(1,3) = volp*(EPRESS(PIXZ)+EPRESS(PIZX))*HALF
          G_CP(2,1) = volp*(EPRESS(PIYX)+EPRESS(PIXY))*HALF
          G_CP(2,2) = volp* EPRESS(PIYY)
          G_CP(2,3) = volp*(EPRESS(PIYZ)+EPRESS(PIZY))*HALF
          G_CP(3,1) = volp*(EPRESS(PIZX)+EPRESS(PIXZ))*HALF
          G_CP(3,2) = volp*(EPRESS(PIZY)+EPRESS(PIYZ))*HALF
          G_CP(3,3) = volp* EPRESS(PIZZ)
          CALL MatchXTL(G_CP,XTLTYP,XTLREF)
          G_CP(1,1) = G_CP(1,1) - volp*P_CP + kinn
          G_CP(2,2) = G_CP(2,2) - volp*P_CP + kinn
          G_CP(3,3) = G_CP(3,3) - volp*P_CP + kinn
       ELSE
          IF (QZONLY) THEN
             G_CP(1,1) = ZERO
             G_CP(2,2) = ZERO
             G_CP(3,3) = volp*(EPRESS(PIZZ)-P_CP) + kinn
          ELSE
             G_CP(1,1) = THREE*(volp*(EPROP(PRESSI)-P_CP) + kinn)
             G_CP(2,2) = G_CP(1,1)
             G_CP(3,3) = G_CP(1,1)
          ENDIF
       ENDIF
       CALL AFFINE33(ZETA_CP,ZETA_CP,fact,G_CP)

       if (nDrudes > 0 .and. .not.QDSCALE_CP) then
          CALL CorrectVirial( 0, EPRESS(VIXX:VIZZ), &
               X,Y,Z, DX,DY,DZ, DXc,DYc,DZc, &
               ATFRST,ATLAST, AMASS,isDrude,IMOVE )
       endif
    endif
    !     Go back for a RATTLE/ROLL iteration
    if (QHOLO) then
       if (iRoll <= MAX_ROLL .and. RMSDrattle > TENM8) then
          iRoll = iRoll + 1
          rolling = .true.
          goto 848
       elseif (iRoll > MAX_ROLL) then
          write(OUTU,*) 'VV2 -- RATTLE/ROLL did NOT converge!'
          write(OUTU,*) 'VV2 -- RMSDrattle',RMSDrattle
          write(outu,*) 'VV2 -- (Try increasing BTAU)'
          rolling = .false.
          iRoll = 0
       else
          if (verbose) write(OUTU,*) 'DEBUG -- RATTLE/ROLL converged'
          rolling = .false.
          iRoll = 0
       endif
    endif

    !     Only for NVT
    if (.not.QPCON) then
       if (QHOLO) then
          if (verbose) write(OUTU,*) 'DEBUG -- ', &
               'Correct virial for RATTLE'
          CALL VIRSHK( EPRESS(VIXX:VIZZ), nAtoms, &
               X,Y,Z, DXc,DYc,DZc )
       endif
       CALL PRSEXT
       CALL PRSINT(nAtoms,AMASS,VX,VY,VZ,ONE,VX,VX,VX,ZERO)
       ! APH: Note VX, VX, VX here ------------|
       !      are just dummies
    endif


    !     Propagate Nose-Hoover variables 2
    if (QNOSE) then
       if (QPCON .and. TI_CP > 0) then
          if (verbose) write(OUTU,*) 'DEBUG -- ', &
               'PropagateThermostatForBarostat 2'
          CALL PropagateTFB( &
               nc,DELTA_, &
               QNOSE,NOBL,MAXNOS,INLCKP,useThermostat, &
               QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
               IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
               QNHLANG,FBETA, &
               SQM,SNH,SNHV,SNHF,NDGN,RTMPR, &
               QPCON,QPCONFULL,W_CP,P_CP,TI_CP, &
               ETA_CP,ZETA_CP, &
               kin2,kin2_nh, &
               NDEGF,ATFRST,ATLAST, IMOVE,AMASS, &
               X,Y,Z, VX,VY,VZ, isDrude,nDrudes )
       endif
       if (verbose) write(OUTU,*) 'DEBUG -- ', &
            'PropagateThermostatForParticles 2'
       CALL PropagateTFP( &
            nc,DELTA_, &
            QNOSE,NOBL,MAXNOS,INLCKP,useThermostat, &
            QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M,QCMLANGEVIN,CMGAMMA, &
            QNHLANG,FBETA,ISTEP, &
            SQM,SNH,SNHV,SNHF,NDGN,RTMPR, &
                                !    $        QPCON,W_CP,P_CP,
            TI_CP, &
                                !    $        ETA_CP,ZETA_CP,
            kin2,kin2_nh, &
            NDEGF,ATFRST,ATLAST, IMOVE,AMASS, &
            X,Y,Z, VX,VY,VZ, isDrude,nDrudes )
    else
       if (verbose) write(OUTU,*) 'DEBUG -- KineticEFNH (F)'
       CALL KineticEFNH( &
            QNOSE,NOBL,INLCKP, &
            QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
            kin2,kin2_nh, ATFRST,ATLAST, IMOVE,AMASS, &
            VX,VY,VZ, isDrude )
    endif

    if (verbose) then
       CALL CMVelocity( &
            ATFRST,ATLAST,IMOVE,AMASS,VX,VY,VZ, &
            VRx,VRy,VRz,mcm )
    endif

#if KEY_PARALLEL==1
    tmeri(timdcntrl)=tmeri(timdcntrl)+eclock()-timmer
#endif
    RETURN
  END SUBROUTINE VV2

  !     PropagateThermostatForBarostat
  SUBROUTINE PropagateTFB( &
       nc,dt, &
       QNOSE,NOBL,MAXNOS,INLCKP,useThermostat, &
       QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
       IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
       QNHLANG,FBETA, &
       SQM,SNH,SNHV,SNHF,NDGN,RTMPR, &
                                !
       QPCON,QPCONFULL,W_CP,P_CP,TI_CP, &
       ETA_CP,ZETA_CP, &
                                !
       kin2,kin2_nh, &
       NDEGF,ATFRST,ATLAST, IMOVE_,AMASS_, &
       X,Y,Z, VX,VY,VZ, ISDRUDE,NDRUDE )

    use number
    !     Global variables/functions
    use stream ! for OUTU
    use consta ! for KBOLTZ
    use energym ! for EPROP
    use image ! for XTLABC


    !     Arguments
    integer nc                ! INPUT   Thermostat timestep
    real(chm_real)  dt                ! INPUT   Timestep

    logical QNOSE             ! INPUT   Nose-Hoover logical flag
    integer NOBL              ! INPUT   Number of Nose-Hoover (NH) thermostats
    integer MAXNOS            ! INPUT   Maximum value for NOBL
    integer,dimension(:) :: INLCKP  ! INPUT
    logical useThermostat(*)  ! INPUT   Enabling of NH thermostats

    logical QRELT(MAXNOS)     ! INPUT   Relative thermostatting flags
    logical QQRELT            ! INPUT
    real(chm_real)  RELT_VX(MAXNOS)   ! INPUT
    real(chm_real)  RELT_VY(MAXNOS)   ! INPUT
    real(chm_real)  RELT_VZ(MAXNOS)   ! INPUT
    real(chm_real)  RELT_M(MAXNOS)    ! INPUT
    integer IABST             ! INPUT   Absolute thermostatting
    real(chm_real)  ABST_VX           ! INPUT
    real(chm_real)  ABST_VY           ! INPUT
    real(chm_real)  ABST_VZ           ! INPUT
    real(chm_real)  ABST_M            ! INPUT

    logical QNHLANG(MAXNOS)   ! INPUT
    real(chm_real)  FBETA(*)          ! INPUT

    real(chm_real)  SQM(MAXNOS)       ! INPUT   NH inertia parameters (Q)
    real(chm_real)  SNH(MAXNOS)       ! I/O     NH positions (eta)
    real(chm_real)  SNHV(MAXNOS)      ! I/O     NH velocities (zeta)
    real(chm_real)  SNHF(MAXNOS)      ! I/O     NH forces (G)
    real(chm_real)  NDGN(MAXNOS)      ! INPUT   Number of deg of freedom for each thermostat
    real(chm_real)  RTMPR(MAXNOS)     ! INPUT   NH target temperatures

    logical QPCON             ! INPUT   Andersen-Hoover (AH) barostat logical flag

    logical QPCONFULL         ! INPUT
    real(chm_real)  W_CP              ! INPUT   Barostat inertia parameter
    real(chm_real)  P_CP              ! INPUT   Target isotropic pressure
    integer TI_CP             ! INPUT   Index of thermostat coupled to the barostat

    real(chm_real)  ETA_CP(3,3)       ! I/O     AH position
    real(chm_real)  ZETA_CP(3,3)      ! I/O     AH velocity

    real(chm_real)  kin2              ! OUTPUT  Two times the kinetic energy (mv2)
    real(chm_real)  kin2_nh(NOBL)     ! I/O     "kin2" for each thermostat

    integer NDEGF             ! INPUT   Number of degrees of freedom
    integer ATFRST, ATLAST    ! INPUT
    integer IMOVE_(*)
    real(chm_real)  AMASS_(*)
    real(chm_real)  X(*), Y(*), Z(*)
    real(chm_real)  VX(*), VY(*), VZ(*)
    logical ISDRUDE(*)
    integer NDRUDE

    !     Local variables
    integer ic
    real(chm_real)  dtsy
    integer ti                ! thermostat index

    logical, parameter :: ddebug = .false.
    real(chm_real)  prod(3,3),trZETA2,fact

    DO ic = 1,nc
       dtsy = dt/REAL(nc)
       if (ddebug) WRITE(OUTU,*) '  DEBUG -- ', &
            'YS Iteration',ic

       !        Compute force on the thermostat on the barostat
       !        Propagate the velocity of the thermostat on the barostat
       CALL KineticEFNH( &
            QNOSE,NOBL,INLCKP, &
            QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
            kin2,kin2_nh, ATFRST,ATLAST, IMOVE_,AMASS_, &
            VX,VY,VZ, ISDRUDE )
       ti = TI_CP
       SNHF(ti) = kin2_nh(ti) - NDGN(ti)*KBOLTZ*RTMPR(ti)
       IF (QPCON .AND. ti == TI_CP) THEN
          IF (QPCONFULL) THEN ! Is this working???
             CALL TRANSPS(prod,ZETA_CP,3,3)
             CALL MULT33(prod,prod,ZETA_CP)
             trZETA2 = prod(1,1) + prod(2,2) + prod(3,3)
             SNHF(ti) = SNHF(ti) + W_CP*trZETA2 &
                  - XDIM*KBOLTZ*RTMPR(ti)
          ELSE
             SNHF(ti) = SNHF(ti) + W_CP*ZETA_CP(3,3)**2 &
                  - KBOLTZ*RTMPR(ti)
          ENDIF
       ENDIF
       SNHV(ti) = SNHV(ti) + PT25*dtsy * SNHF(ti)/SQM(ti)

       !        Scale the velocity of the barostat
       !        ZETA_CP = ZETA_CP * EXP(-SNHV(ti) * HALF*dtsy)
       fact = EXP(-SNHV(ti) * HALF*dtsy)
       CALL DAFFINE33(ZETA_CP,ZERO,fact,ZETA_CP)

       !        Propagate the thermostat
       SNH(ti) = SNH(ti) + HALF*dtsy * SNHV(ti)

       !        Compute force on the thermostat on the barostat
       !        Propagate the velocity of the thermostat on the barostat
       CALL KineticEFNH( &
            QNOSE,NOBL,INLCKP, &
            QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
            kin2,kin2_nh, ATFRST,ATLAST, IMOVE_,AMASS_, &
            VX,VY,VZ, ISDRUDE )
       ti = TI_CP
       SNHF(ti) = kin2_nh(ti) - NDGN(ti)*KBOLTZ*RTMPR(ti)
       IF (QPCON .AND. ti == TI_CP) THEN
          IF (QPCONFULL) THEN ! Is this working???
             CALL TRANSPS(prod,ZETA_CP,3,3)
             CALL MULT33(prod,prod,ZETA_CP)
             trZETA2 = prod(1,1) + prod(2,2) + prod(3,3)
             SNHF(ti) = SNHF(ti) + W_CP*trZETA2 &
                  - XDIM*KBOLTZ*RTMPR(ti)
          ELSE
             SNHF(ti) = SNHF(ti) + W_CP*ZETA_CP(3,3)**2 &
                  - KBOLTZ*RTMPR(ti)
          ENDIF
       ENDIF
       SNHV(ti) = SNHV(ti) + PT25*dtsy * SNHF(ti)/SQM(ti)

    ENDDO

    RETURN
  END SUBROUTINE PropagateTFB


  !     PropagateThermostatForParticles
  SUBROUTINE PropagateTFP( &
       nc,dt, &
       QNOSE,NOBL,MAXNOS,INLCKP,useThermostat, &
       QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
       IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M,QCMLANGEVIN,CMGAMMA, &
       QNHLANG,FBETA,ISTEP, &
       SQM,SNH,SNHV,SNHF,NDGN,RTMPR, &
                                !    $
                                !    $     QPCON,W_CP,P_CP,
       TI_CP, &
                                !    $     ETA_CP,ZETA_CP,
                                !    $
       kin2,kin2_nh, &
       NDEGF,ATFRST,ATLAST, IMOVE_,AMASS_, &
       X,Y,Z, VX,VY,VZ, ISDRUDE,NDRUDE )

    use number
    !     Global variables/functions
    use stream ! for OUTU
    use consta ! for KBOLTZ
    use energym ! for EPROP
    use image ! for XTLABC
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,atoml
#endif

    !     Arguments
    integer nc                ! INPUT   Thermostat timestep
    real(chm_real)  dt                ! INPUT   Timestep

    logical QNOSE             ! INPUT   Nose-Hoover logical flag
    integer NOBL              ! INPUT   Number of Nose-Hoover (NH) thermostats
    integer MAXNOS            ! INPUT   Maximum value for NOBL
    integer,dimension(:) :: INLCKP  ! INPUT
    logical useThermostat(*)  ! INPUT   Enabling of NH thermostats

    logical QRELT(MAXNOS)     ! INPUT   Relative thermostatting flags
    logical QQRELT            ! INPUT
    real(chm_real)  RELT_VX(MAXNOS)   ! INPUT
    real(chm_real)  RELT_VY(MAXNOS)   ! INPUT
    real(chm_real)  RELT_VZ(MAXNOS)   ! INPUT
    real(chm_real)  RELT_M(MAXNOS)    ! INPUT
    integer IABST             ! INPUT   Absolute thermostatting
    real(chm_real)  ABST_VX           ! INPUT
    real(chm_real)  ABST_VY           ! INPUT
    real(chm_real)  ABST_VZ           ! INPUT
    real(chm_real)  ABST_M            ! INPUT
    logical QCMLANGEVIN       ! INPUT
    real(chm_real)  CMGAMMA           ! INPUT

    logical QNHLANG(MAXNOS)   ! INPUT
    real(chm_real)  FBETA(*)          ! INPUT
    integer ISTEP

    real(chm_real)  SQM(MAXNOS)       ! INPUT   NH inertia parameters (Q)
    real(chm_real)  SNH(MAXNOS)       ! I/O     NH positions (eta)
    real(chm_real)  SNHV(MAXNOS)      ! I/O     NH velocities (zeta)
    real(chm_real)  SNHF(MAXNOS)      ! I/O     NH forces (G)
    real(chm_real)  NDGN(MAXNOS)      ! INPUT   Number of deg of freedom for each thermostat
    real(chm_real)  RTMPR(MAXNOS)     ! INPUT   NH target temperatures

    !     logical QPCON             ! INPUT   Andersen-Hoover (AH) barostat logical flag
    !     real(chm_real)  W_CP              ! INPUT   Barostat inertia parameter
    !     real(chm_real)  P_CP              ! INPUT   Target isotropic pressure
    integer TI_CP             ! INPUT   Index of thermostat coupled to the barostat

    !     real(chm_real)  ETA_CP(3,3)       ! I/O     AH position
    !     real(chm_real)  ZETA_CP(3,3)      ! I/O     AH velocity

    real(chm_real)  kin2              ! OUTPUT  Two times the kinetic energy (mv2)
    real(chm_real)  kin2_nh(NOBL)     ! I/O     "kin2" for each thermostat

    integer NDEGF             ! INPUT   Number of degrees of freedom
    integer ATFRST, ATLAST    ! INPUT
    integer IMOVE_(*)
    real(chm_real)  AMASS_(*)
    real(chm_real)  X(*), Y(*), Z(*)
    real(chm_real)  VX(*), VY(*), VZ(*)
    logical ISDRUDE(*)
    integer NDRUDE

    !     Local variables
    integer ic
    real(chm_real)  dtsy
    integer ti                ! thermostat index
    integer I, J              ! atom indices
    integer ia

    real(chm_real)  mcmi, vxcm, vycm, vzcm
    real(chm_real)  fact_int, fact_ext
    logical, parameter :: ddebug = .false.
    !SB: cache for often used exponentials
    real(chm_real), dimension(maxnos) :: expfac1

    dtsy = dt/REAL(nc, chm_real)
    DO ic = 1,nc
       if (ddebug) WRITE(OUTU,*) '  DEBUG -- ', &
            'YS Iteration',ic

       !        Compute force on the thermostats (except the thermostat on the barostat)
       !        Propagate the velocities of the thermostats
       CALL KineticEFNH( &
            QNOSE,NOBL,INLCKP, &
            QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
            kin2,kin2_nh, ATFRST,ATLAST, IMOVE_,AMASS_, &
            VX,VY,VZ, ISDRUDE )
       DO ti = 1,NOBL
          IF (.NOT.useThermostat(ti)) GOTO 710
          IF (ti == TI_CP) GOTO 710
          SNHF(ti) = kin2_nh(ti) - NDGN(ti)*KBOLTZ*RTMPR(ti)
          SNHV(ti) = SNHV(ti) + PT25*dtsy * SNHF(ti)/SQM(ti)
          expfac1(ti) = EXP(-SNHV(ti) * HALF*dtsy) !2*0.25
710       CONTINUE
       ENDDO

       !        Scale the velocities of the particles
       if (QQRELT) then
          CALL RelativeTstat( &
               .TRUE.,.TRUE., QNOSE,NOBL,INLCKP, &
               QRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
               IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
               ATFRST,ATLAST,IMOVE_,AMASS_, VX,VY,VZ, &
               ISDRUDE )
       endif
       do ia = atfrst,atlast
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
          IF (JPBLOCK(I) /= MYNOD) GOTO 711
#endif
          IF (IMOVE_(I) /= 0) GOTO 711

          IF (ISDRUDE(I)) THEN
!!! Scale for a Drude
             J = I - 1
             ti = INLCKP(I)
             fact_int = expfac1(ti) !2*0.25
             ti = INLCKP(J)
             fact_ext = expfac1(ti) !2*0.25

             mcmi = ONE / (AMASS_(I) + AMASS_(J))
             vxcm = (AMASS_(I)*VX(I) + AMASS_(J)*VX(J)) * mcmi
             vycm = (AMASS_(I)*VY(I) + AMASS_(J)*VY(J)) * mcmi
             vzcm = (AMASS_(I)*VZ(I) + AMASS_(J)*VZ(J)) * mcmi


             VX(I) = vxcm * fact_ext + (VX(I)-vxcm) * fact_int
             VY(I) = vycm * fact_ext + (VY(I)-vycm) * fact_int
             VZ(I) = vzcm * fact_ext + (VZ(I)-vzcm) * fact_int

          elseif (ia /= atlast .and. isdrude(i+1)) then
!!! Scale for a polarizable atom
             J = I + 1
             ti = INLCKP(I)
             fact_ext = expfac1(ti) !2*0.25

             ti = INLCKP(J)
             fact_int = expfac1(ti) !2*0.25

             mcmi = ONE / (AMASS_(I) + AMASS_(J))
             vxcm = (AMASS_(I)*VX(I) + AMASS_(J)*VX(J)) * mcmi
             vycm = (AMASS_(I)*VY(I) + AMASS_(J)*VY(J)) * mcmi
             vzcm = (AMASS_(I)*VZ(I) + AMASS_(J)*VZ(J)) * mcmi


             VX(I) = vxcm * fact_ext + (VX(I)-vxcm) * fact_int
             VY(I) = vycm * fact_ext + (VY(I)-vycm) * fact_int
             VZ(I) = vzcm * fact_ext + (VZ(I)-vzcm) * fact_int

          ELSE                !!! The original scaling
             ti = INLCKP(I)
             fact_ext = expfac1(ti) !2*0.25
             VX(I) = VX(I) * fact_ext
             VY(I) = VY(I) * fact_ext
             VZ(I) = VZ(I) * fact_ext
          ENDIF
711       CONTINUE
       ENDDO

       !        Scale CM velocity
       if (QQRELT .and. IABST > 0) then
          if (QCMLANGEVIN) then
             fact_ext = EXP(-CMGAMMA * HALF*dtsy) !2*0.25
          else
             fact_ext = EXP(-SNHV(IABST) * HALF*dtsy) !2*0.25
          endif
          ABST_VX = ABST_VX * fact_ext
          ABST_VY = ABST_VY * fact_ext
          ABST_VZ = ABST_VZ * fact_ext
       endif

       if (QQRELT) then
          CALL RelativeTstat( &
               .FALSE.,.FALSE., QNOSE,NOBL,INLCKP, &
               QRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
               IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
               ATFRST,ATLAST,IMOVE_,AMASS_, VX,VY,VZ, &
               ISDRUDE )
       endif

       !        Propagate the thermostats
       DO ti = 1,NOBL
          IF (.NOT.useThermostat(ti)) GOTO 712
          IF (ti == TI_CP) GOTO 712
          SNH(ti) = SNH(ti) + HALF*dtsy * SNHV(ti)
712       CONTINUE
       ENDDO

       !        Compute force on the thermostats (except the thermostat on the barostat)
       !        Propagate the velocities of the thermostats
       CALL KineticEFNH( &
            QNOSE,NOBL,INLCKP, &
            QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
            kin2,kin2_nh, ATFRST,ATLAST, IMOVE_,AMASS_, &
            VX,VY,VZ, ISDRUDE )
       DO ti = 1,NOBL
          IF (.NOT.useThermostat(ti)) GOTO 713
          IF (ti == TI_CP) GOTO 713
          SNHF(ti) = kin2_nh(ti) - NDGN(ti)*KBOLTZ*RTMPR(ti)
          SNHV(ti) = SNHV(ti) + PT25*dtsy * SNHF(ti)/SQM(ti)
713       CONTINUE
       ENDDO

    ENDDO

    RETURN
  END SUBROUTINE PropagateTFP


  !     KineticEnergyForNoseHoover
  SUBROUTINE KineticEFNH( &
       QNOSE,NOBL,INLCKP, &
       QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
       IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
       kin2,kin2_nh, ATFRST,ATLAST, IMOVE_,AMASS_, &
       VX,VY,VZ, ISDRUDE )
    !
    !     Compute kinetic energy components for every Nose-Hoover
    !     thermostat.
    !

    !
    !     Global variables/functions
    !
    use number ! for ZERO
    use parallel ! for JPBLOCK, MYNOD
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,atoml
#endif

    !
    !     Arguments
    !
    logical QNOSE             ! INPUT   Nose-Hoover (NH) flag
    integer NOBL              ! INPUT   Number of NH thermostats
    integer,dimension(:) :: INLCKP  ! INPUT
    logical QRELT(NOBL)       ! INPUT   Relative thermostatting flags
    logical QQRELT            ! INPUT
    real(chm_real)  RELT_VX(NOBL)     ! INPUT
    real(chm_real)  RELT_VY(NOBL)     ! INPUT
    real(chm_real)  RELT_VZ(NOBL)     ! INPUT
    real(chm_real)  RELT_M(NOBL)      ! INPUT
    integer IABST             ! INPUT   Absolute thermostatting
    real(chm_real)  ABST_VX           ! INPUT
    real(chm_real)  ABST_VY           ! INPUT
    real(chm_real)  ABST_VZ           ! INPUT
    real(chm_real)  ABST_M            ! INPUT

    real(chm_real)  kin2              ! OUTPUT  Two times the kinetic energy (mv2)
    real(chm_real)  kin2_nh(NOBL)     ! OUTPUT  "kin2" for each thermostat (1:NOBL)

    integer ATFRST, ATLAST    ! INPUT
    integer IMOVE_(*)         ! INPUT
    real(chm_real)  AMASS_(*)         ! INPUT
    real(chm_real)  VX(*), VY(*), VZ(*)

    logical ISDRUDE(*)

    !
    !     Local variables
    !
    integer ti, I, J
    integer ia
    real(chm_real)  mcm, mcmi, vxcm, vycm, vzcm
    real(chm_real)  mp, vxp, vyp, vzp
    real(chm_real)  mv2

    kin2 = ZERO
    IF (QNOSE) THEN
       DO ti = 1,NOBL
          kin2_nh(ti) = ZERO
       ENDDO
    ENDIF

    if (QQRELT) then
       CALL RelativeTstat( &
            .TRUE.,.TRUE., QNOSE,NOBL,INLCKP, &
            QRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
            ATFRST,ATLAST,IMOVE_,AMASS_, VX,VY,VZ, &
            ISDRUDE )
    endif

    do ia = atfrst,atlast
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
       IF (JPBLOCK(I) /= MYNOD) GOTO 714
#endif
       IF (IMOVE_(I) /= 0) GOTO 714
       mv2 = AMASS_(I)*(VX(I)**2+VY(I)**2+VZ(I)**2)
       kin2 = kin2 + mv2
       IF (QNOSE) THEN
          ti = INLCKP(I)
!!! We need to recompute "mv2" if the
!!! atom is a Drude oscillator or is
!!! attached to a Drude oscillator.
          IF (ISDRUDE(I)) THEN
!!! We compute the kinetic energy of the
!!! internal motion
             J = I - 1
             mp = AMASS_(I)*AMASS_(J) / (AMASS_(I) + AMASS_(J))
             vxp = VX(I) - VX(J)
             vyp = VY(I) - VY(J)
             vzp = VZ(I) - VZ(J)
             mv2 = mp * (vxp**2 + vyp**2 + vzp**2)
          elseif (ia /= atlast .and. isdrude(i+1)) then
!!! We compute the kinetic energy of the
!!! center of mass
             J = I + 1
             mcm = AMASS_(I) + AMASS_(J)
             mcmi = ONE / mcm
             vxcm = ( AMASS_(I)*VX(I) + AMASS_(J)*VX(J) ) * mcmi
             vycm = ( AMASS_(I)*VY(I) + AMASS_(J)*VY(J) ) * mcmi
             vzcm = ( AMASS_(I)*VZ(I) + AMASS_(J)*VZ(J) ) * mcmi
             mv2 = mcm * (vxcm**2 + vycm**2 + vzcm**2)
          ENDIF
          kin2_nh(ti) = kin2_nh(ti) + mv2
       ENDIF
714    CONTINUE
    ENDDO
    IF (IABST > 0) THEN
       IF (kin2_nh(IABST)  /=  ZERO) THEN
          CALL WRNDIE(-3,'<VV2>','Center-of-mass EK not ZERO')
       ENDIF
       kin2_nh(IABST) = ABST_M &
            * (ABST_VX**2 + ABST_VY**2 + ABST_VZ**2)
       kin2 = kin2 + kin2_nh(IABST)
    ENDIF

#if KEY_PARALLEL==1
    CALL GCOMB(kin2,1)
#endif
#if KEY_PARALLEL==1
    CALL GCOMB(kin2_nh,nobl)
#endif

    if (QQRELT) then
       CALL RelativeTstat( &
            .FALSE.,.FALSE., QNOSE,NOBL,INLCKP, &
            QRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
            IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
            ATFRST,ATLAST,IMOVE_,AMASS_, VX,VY,VZ, &
            ISDRUDE )
    endif

  END SUBROUTINE KineticEFNH


  SUBROUTINE OptimizeDrude( toler,ncycles, &
       oldX,oldY,oldZ, ioVX,ioVY,ioVZ, BNBND,BIMAG, &
       DX,DY,DZ, useFastForces )
    !
    !     Significantly changed from "dynamvv2.src,fev2003":
    !
    use number
    use reawri ! for DELTA, ISEED
    use contrl
    use coord ! for X, Y, Z
    use shake
    use stream
    !
    use parallel ! for JPBLOCK, MYNOD
    use psf ! for IMOVE, IB, JB
    use param ! for CBC
    use code ! for ICB
    use image ! for XTLTYP, XTLABC
    use tbmts
    use energym ! for EPROP, ETERM
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,natoml,atoml
#endif

    !     Arguments
    real(chm_real)  toler
    INTEGER ncycles
    real(chm_real)  oldX(*), oldY(*), oldZ(*)
    real(chm_real)  ioVX(*), ioVY(*), ioVZ(*)
    type(nonbondDataStructure) BNBND
    type(imageDataStructure) BIMAG

    real(chm_real) :: DX(:), DY(:), DZ(:)
    LOGICAL useFastForces

    !     Local variables
    integer iter
    INTEGER ATFRST, ATLAST
    INTEGER I, ibond
    real(chm_real)  root,kroot
    real(chm_real)  f_rms,f2,f2_max
    integer i_rms
    integer ia

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

    do iter = 1,ncycles

       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)

       !        Compute RMS force on Drudes
       f_rms = ZERO
       f2_max = ZERO
       i_rms = 0
       do ia = atfrst,atlast
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
          IF (JPBLOCK(I) /= MYNOD) GOTO 715
#endif
          IF (.NOT.ISDRUDE(I)) GOTO 715
          f2 = DX(I)**2 + DY(I)**2 + DZ(I)**2
          if (f2 > f2_max) f2_max = f2
          f_rms = f_rms + f2
          i_rms = i_rms + 1
715       CONTINUE
       ENDDO
       f_rms = SQRT(f_rms/i_rms)
       if (f_rms < toler) goto 901

       do ia = atfrst,atlast
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
          IF (JPBLOCK(I) /= MYNOD) GOTO 716
#endif
          IF (.NOT.ISDRUDE(I)) GOTO 716

          !           Direct solution of quadratic or cubic equation

          !           Find the atom-Drude bond force constant
          DO ibond = 1,NBOND
             IF ( (IB(ibond) == I .AND. JB(ibond).EQ.(I-1)) &
                  .OR. &
                  (JB(ibond) == I .AND. IB(ibond).EQ.(I-1)) ) THEN
                kroot = TWO * CBC(ICB(ibond))
                GOTO 111
             ENDIF
          ENDDO
111       CONTINUE

          root = (ONE-PTONE)/kroot ! best for water
          X(I) = X(I) - DX(I) * root
          Y(I) = Y(I) - DY(I) * root
          Z(I) = Z(I) - DZ(I) * root
716       CONTINUE
       ENDDO
    enddo
901 continue

    !     Compute self-consistent velocities from position differences
    do ia = atfrst,atlast
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
       IF (JPBLOCK(I) /= MYNOD) GOTO 717
#endif
       IF (.NOT.ISDRUDE(I)) GOTO 717
       ioVX(I) = ioVX(I-1)
       ioVY(I) = ioVY(I-1)
       ioVZ(I) = ioVZ(I-1)
717    CONTINUE
    ENDDO


    CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
    if (PRNLEV > 6) then
       CALL PRINTE(OUTU, EPROP, ETERM, 'SCF ', 'ENR', .TRUE., &
            iter, ZERO, ZERO, .TRUE.)
    endif

    if (iter <= ncycles) then
       if (PRNLEV > 6) then
          write(OUTU,501) iter,f_rms,sqrt(f2_max)
501       format(' SCF converged in',I3,' iterations'/ &
               ' RMS force:',F14.10,'   MAX force:',F14.10)
       endif
    else
       if (PRNLEV > 6) then
          write(OUTU,502) iter,f_rms,sqrt(f2_max)
502       format(' SCF did NOT converge in',I3,' iterations'/ &
               ' RMS force:',F14.10,'   MAX force:',F14.10)
       endif
    endif

    RETURN
  END SUBROUTINE OptimizeDrude


  SUBROUTINE ShakeVelo( &
       inXNEW, inYNEW, inZNEW, &
       inXOLD, inYOLD, inZOLD, &
       ioVX, ioVY, ioVZ, &
       inFirstAtom, inLastAtom, &
       inDeltaT )
    !
    !     Correct SHAKE velocities:
    !     V_i <-- V_i + (x'_i-x_i)/dt
    !
    use number
    use nose_mod ! for QPCON, RX2_CP
    use parallel ! for JPBLOCK, MYNOD
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,atoml
#endif

    !     Arguments
    real(chm_real)  inXNEW(*), inYNEW(*), inZNEW(*)
    real(chm_real)  inXOLD(*), inYOLD(*), inZOLD(*)
    ! Input: Positions
    real(chm_real)  ioVX(*), ioVY(*), ioVZ(*)
    ! Input/Output: Velocities

    integer inFirstAtom       ! Input: Index of the first atom
    integer inLastAtom        ! Input: Index of the last atom

    real(chm_real)  inDeltaT          ! Input: Time step (dt)

    !     Global variables

    !     Local variables
    integer i
    integer ia
    real(chm_real)  fact1
    real(chm_real)  factX,factY,factZ
    real(chm_real)  fact(3,3),t1,t2,t3
    logical ok


    fact1 = ONE / inDeltaT
    if (QPCON) then
       if (QPCONFULL) then
          CALL INVT33(fact,RX2_CP,ok)
          if (.not.ok) then
             CALL WRNDIE(-2,'<VV2>','Transformation Rx is singular')
          endif
          CALL DAFFINE33(fact,ZERO,fact1,fact)
       elseif (QZONLY) then
          factX = fact1
          factY = fact1
          factZ = fact1 / RX2_CP(3,3)
       else
          factX = fact1 / RX2_CP(1,1)
          factY = fact1 / RX2_CP(2,2)
          factZ = fact1 / RX2_CP(3,3)
       endif
    else
       factX = fact1
       factY = fact1
       factZ = fact1
    endif

    do ia = inFirstAtom,inLastAtom
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
       if (JPBLOCK(i) /= MYNOD) goto 718
#endif
       if (QPCONFULL) then
          t1 = inXNEW(i)-inXOLD(i)
          t2 = inYNEW(i)-inYOLD(i)
          t3 = inZNEW(i)-inZOLD(i)
          ioVX(i) = ioVX(i) + fact(1,1)*t1+fact(1,2)*t2+fact(1,3)*t3
          ioVY(i) = ioVY(i) + fact(2,1)*t1+fact(2,2)*t2+fact(2,3)*t3
          ioVZ(i) = ioVZ(i) + fact(3,1)*t1+fact(3,2)*t2+fact(3,3)*t3
       else
          ioVX(i) = ioVX(i) + factX*(inXNEW(i)-inXOLD(i))
          ioVY(i) = ioVY(i) + factY*(inYNEW(i)-inYOLD(i))
          ioVZ(i) = ioVZ(i) + factZ*(inZNEW(i)-inZOLD(i))
       endif
#if KEY_PARASCAL==1
718    CONTINUE
#endif
    enddo

  END SUBROUTINE ShakeVelo


  SUBROUTINE ShakeForces( &
       inXNEW, inYNEW, inZNEW, &
       inXOLD, inYOLD, inZOLD, &
       ioDX, ioDY, ioDZ, &
       inFirstAtom, inLastAtom, &
       inDeltaT, inMass, outRMSD )
    !
    !     Compute SHAKE forces:
    !     F_i <-- 2 m_i/dt^2 (x'_i-x_i)
    !
    use number
    !     Global variables
    use nose_mod ! for QPCON, RX2_CP, RV2_CP
    use parallel ! for JPBLOCK, MYNOD
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,atoml
#endif

    !     Arguments
    real(chm_real)  inXNEW(*), inYNEW(*), inZNEW(*)
    real(chm_real)  inXOLD(*), inYOLD(*), inZOLD(*)
    ! Input: Positions
    real(chm_real)  ioDX(*), ioDY(*), ioDZ(*)
    ! Input: Old forces
    ! Output: New forces

    integer inFirstAtom       ! Input: Index of the first atom
    integer inLastAtom        ! Input: Index of the last atom

    real(chm_real)  inDeltaT          ! Input: Time step (dt)
    real(chm_real)  inMass(*)         ! Input: Atomic masses (m_i)

    real(chm_real)  outRMSD           ! Output: RMS difference between new and old forces

    !     Local variables
    integer i
    integer ia
    real(chm_real)  fact1
    real(chm_real)  factX,factY,factZ  ! for QZONLY
    integer nRMSD
    real(chm_real)  newx,newy,newz
    real(chm_real)  prod(3,3),fact(3,3),t1,t2,t3
    logical ok

    fact1 = -TWO / inDeltaT**2
    if (QPCON) then
       if (QPCONFULL) then
          !           f = RX2^-1 RV2^-1 r
          !           (see subroutine ShakeRoll)
          CALL MULT33(prod,RX2_CP,RV2_CP)
          CALL INVT33(fact,prod,ok)
          if (.not.ok) then
             CALL WRNDIE(-2,'<VV2>','Transformation RxRv is singular')
          endif
          CALL DAFFINE33(fact,ZERO,fact1,fact)
       else
          if (QZONLY) then
             factX = fact1
             factY = fact1
             factZ = fact1 / RX2_CP(3,3) / RV2_CP(3,3)
          else
             factX = fact1 / RX2_CP(1,1) / RV2_CP(1,1)
             factY = fact1 / RX2_CP(2,2) / RV2_CP(2,2)
             factZ = fact1 / RX2_CP(3,3) / RV2_CP(3,3)
          endif
       endif
    else
       factX = fact1
       factY = fact1
       factZ = fact1
    endif

    outRMSD = ZERO
    nRMSD = 0
    do ia = inFirstAtom,inLastAtom
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
       if (JPBLOCK(i) /= MYNOD) goto 719
#endif
       if (QPCONFULL) then
          t1 = inXNEW(i)-inXOLD(i)
          t2 = inYNEW(i)-inYOLD(i)
          t3 = inZNEW(i)-inZOLD(i)
          newx = inMass(i)*(fact(1,1)*t1+fact(1,2)*t2+fact(1,3)*t3)
          newy = inMass(i)*(fact(2,1)*t1+fact(2,2)*t2+fact(2,3)*t3)
          newz = inMass(i)*(fact(3,1)*t1+fact(3,2)*t2+fact(3,3)*t3)
       else
          newx = inMass(i)*factX * (inXNEW(i)-inXOLD(i))
          newy = inMass(i)*factY * (inYNEW(i)-inYOLD(i))
          newz = inMass(i)*factZ * (inZNEW(i)-inZOLD(i))
       endif
       outRMSD = outRMSD + (ioDX(i)-newx)**2 &
            + (ioDY(i)-newy)**2 + (ioDZ(i)-newz)**2
       ioDX(i) = newx
       ioDY(i) = newy
       ioDZ(i) = newz
       nRMSD = nRMSD + 1
#if KEY_PARASCAL==1
719    CONTINUE
#endif
    enddo
#if KEY_PARALLEL==1
    CALL GCOMB(outrmsd,1)
    outrmsd = sqrt(outrmsd/iparpt(numnod))
#else /**/
    outRMSD = sqrt(outRMSD/nRMSD)
#endif

  END SUBROUTINE ShakeForces


  SUBROUTINE RattleForces( &
       inVXNEW, inVYNEW, inVZNEW, &
       inVXOLD, inVYOLD, inVZOLD, &
       ioDX, ioDY, ioDZ, &
       inFirstAtom, inLastAtom, &
       inDeltaT, inMass, outRMSD )
    !
    !     Compute RATTLE forces:
    !     F_i <-- 2 m_i/dt (v'_i-v_i)
    !
    use number

    !     Global variables
    use nose_mod ! for QPCON, RV2_CP
    use parallel ! for JPBLOCK, MYNOD
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,atoml
#endif

    !     Arguments
    real(chm_real)  inVXNEW(*), inVYNEW(*), inVZNEW(*)
    real(chm_real)  inVXOLD(*), inVYOLD(*), inVZOLD(*)
    ! Input: Velocities
    real(chm_real)  ioDX(*), ioDY(*), ioDZ(*)
    ! Input: Old forces
    ! Output: New forces

    integer inFirstAtom       ! Input: Index of the first atom
    integer inLastAtom        ! Input: Index of the last atom

    real(chm_real)  inDeltaT          ! Input: Time step (dt)
    real(chm_real)  inMass(*)         ! Input: Atomic masses (m_i)

    real(chm_real)  outRMSD           ! Output: RMS difference between new and old forces

    !     Local variables
    integer i
    integer ia
    real(chm_real)  fact1
    real(chm_real)  factX,factY,factZ
    integer nRMSD
    real(chm_real)  newx,newy,newz
    real(chm_real)  fact(3,3),t1,t2,t3
    logical ok

    fact1 = -TWO / inDeltaT
    if (QPCON) then
       if (QPCONFULL) then
          CALL INVT33(fact,RV2_CP,ok)
          if (.not.ok) then
             CALL WRNDIE(-2,'<VV2>','Transformation Rv is singular')
          endif
          CALL DAFFINE33(fact,ZERO,fact1,fact)
       elseif (QZONLY) then
          factX = fact1
          factY = fact1
          factZ = fact1 / RV2_CP(3,3)
       else
          factX = fact1 / RV2_CP(1,1)
          factY = fact1 / RV2_CP(2,2)
          factZ = fact1 / RV2_CP(3,3)
       endif
    else
       factX = fact1
       factY = fact1
       factZ = fact1
    endif
    outRMSD = ZERO
    nRMSD = 0
    do ia = inFirstAtom,inLastAtom
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
       if (JPBLOCK(i) /= MYNOD) goto 720
#endif
       if (QPCONFULL) then
          t1 = inVXNEW(i)-inVXOLD(i)
          t2 = inVYNEW(i)-inVYOLD(i)
          t3 = inVZNEW(i)-inVZOLD(i)
          newx = inMass(i)*(fact(1,1)*t1+fact(1,2)*t2+fact(1,3)*t3)
          newy = inMass(i)*(fact(2,1)*t1+fact(2,2)*t2+fact(2,3)*t3)
          newz = inMass(i)*(fact(3,1)*t1+fact(3,2)*t2+fact(3,3)*t3)
       else
          newx = inMass(i)*factX * (inVXNEW(i)-inVXOLD(i))
          newy = inMass(i)*factY * (inVYNEW(i)-inVYOLD(i))
          newz = inMass(i)*factZ * (inVZNEW(i)-inVZOLD(i))
       endif
       outRMSD = outRMSD + (ioDX(i)-newx)**2 &
            + (ioDY(i)-newy)**2 + (ioDZ(i)-newz)**2
       ioDX(i) = newx
       ioDY(i) = newy
       ioDZ(i) = newz
       nRMSD = nRMSD + 1
#if KEY_PARASCAL==1
720    CONTINUE
#endif
    enddo
#if KEY_PARALLEL==1
    CALL GCOMB(outrmsd,1)
    outrmsd = sqrt(outrmsd/IPARPT(NUMNOD))
#else /**/
    outRMSD = sqrt(outRMSD/nRMSD)
#endif
  END SUBROUTINE RattleForces


  SUBROUTINE PropagateVelo( &
       ioVX, ioVY, ioVZ, &
       inputDX, inputDY, inputDZ, &
       inFirstAtom, inLastAtom, &
       inDeltaT, inMass, inMovingCode )
    !
    !     Propagates velocities according to the formula:
    !     v_i <-- v_i + dt F_i/m_i
    !
    !     Global variables
    use parallel ! for JPBLOCK, MYNOD
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,atoml
#endif

    !     Arguments
    real(chm_real)  ioVX(*), ioVY(*), ioVZ(*)
    ! Input: Velocities
    ! Output: Propagated velocities
    real(chm_real)  inputDX(*), inputDY(*), inputDZ(*)
    ! Input: Negative forces (-F_i)

    integer inFirstAtom       ! Input: Index of the first atom to propagate
    integer inLastAtom        ! Input: Index of the last atom to propagate

    !     integer NDXlang           ! IN
    !     real(chm_real)  DXlang(*)         ! IN/OUT  Langevin forces (1..NDXlang)
    !     real(chm_real)  DYlang(*)         ! IN/OUT  Langevin forces (1..NDXlang)
    !     real(chm_real)  DZlang(*)         ! IN/OUT  Langevin forces (1..NDXlang)
    !     real(chm_real)  FBETA(*)          ! IN  FBETA parameter (1..MAXA)

    real(chm_real)  inDeltaT          ! Input: Time step (dt)
    real(chm_real)  inMass(*)         ! Input: Atomic masses (m_i)
    integer inMovingCode(*)   ! Input: 0 to allow the propagation

    !     Local variables
    integer i
    integer ia
    real(chm_real) fact

    do ia = inFirstAtom,inLastAtom
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
       if (JPBLOCK(i) /= MYNOD) goto 721
#endif
       if (inMovingCode(i) /= 0) goto 721
       fact = inDeltaT/inMass(i)
       ioVX(i) = ioVX(i) - fact*inputDX(i)
       ioVY(i) = ioVY(i) - fact*inputDY(i)
       ioVZ(i) = ioVZ(i) - fact*inputDZ(i)
721    CONTINUE
    enddo

  END SUBROUTINE PropagateVelo


  !     PropagateVelocitiesCP
  SUBROUTINE PropagateVCP( &
       ioVX, ioVY, ioVZ, &
       inDX, inDY, inDZ, &
       inFirstAtom, inLastAtom, inIsDrude, &
       inDeltaT, inMass, inMovingCode, &
                                !    $     QNOSE,NOBL,INLCKP,
                                !    $     NDXlang, DXlang, DYlang, DZlang, FBETA,
                                !    $     QRELT,QQRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M,
                                !    $     IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M,
       QZONLY,QPCONFULL, &
       RV1_CP, RV2_CP )
    !
    !     Propagates velocities according to the formula:
    !     v_i <-- v_i*RV1 + dt F_i/m_i*RV2
    !
    use number
    !     Global variables
    use parallel ! for JPBLOCK, MYNOD
    use reawri ! for ISEED
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,atoml
#endif

    !     Arguments
    real(chm_real)  ioVX(*), ioVY(*), ioVZ(*)
    ! Input: Velocities
    ! Output: Propagated velocities
    real(chm_real)  inDX(*), inDY(*), inDZ(*)
    ! Input: Negative forces (-F_i)

    integer inFirstAtom       ! Input: Index of the first atom to propagate
    integer inLastAtom        ! Input: Index of the last atom to propagate
    logical inIsDrude(*)      ! Input:

    real(chm_real)  inDeltaT          ! Input: Time step (dt)
    real(chm_real)  inMass(*)         ! Input: Atomic masses (m_i)
    integer inMovingCode(*)   ! Input: 0 to allow the propagation

    !     logical QNOSE             ! INPUT
    !     integer NOBL              ! INPUT
    !     integer INLCKP            ! INPUT

    !     integer NDXlang           ! IN
    !     real(chm_real)  DXlang(*)         ! IN/OUT  Langevin forces (1..NDXlang)
    !     real(chm_real)  DYlang(*)         ! IN/OUT  Langevin forces (1..NDXlang)
    !     real(chm_real)  DZlang(*)         ! IN/OUT  Langevin forces (1..NDXlang)
    !     real(chm_real)  FBETA(*)          ! IN  FBETA parameter (1..MAXA)

    !     logical QRELT(NOBL)       ! INPUT   Relative thermostatting flags
    !     logical QQRELT            ! INPUT
    !     real(chm_real)  RELT_VX(NOBL)     ! INPUT
    !     real(chm_real)  RELT_VY(NOBL)     ! INPUT
    !     real(chm_real)  RELT_VZ(NOBL)     ! INPUT
    !     real(chm_real)  RELT_M(NOBL)      ! INPUT
    !     integer IABST             ! INPUT   Absolute thermostatting
    !     real(chm_real)  ABST_VX           ! INPUT
    !     real(chm_real)  ABST_VY           ! INPUT
    !     real(chm_real)  ABST_VZ           ! INPUT
    !     real(chm_real)  ABST_M            ! INPUT
    logical QZONLY            ! INPUT
    logical QPCONFULL         ! INPUT

    real(chm_real)  RV1_CP(3,3)       ! Input
    real(chm_real)  RV2_CP(3,3)       ! Input


    !     Local variables
    integer i
    integer ia
    real(chm_real)  fact
    real(chm_real)  mcm,mcmi,mredi,m1,m2
    real(chm_real)  VRx,VRy,VRz, DRx,DRy,DRz, new_VRx,new_VRy,new_VRz
    real(chm_real)  Vdx,Vdy,Vdz, Ddx,Ddy,Ddz, new_Vdx,new_Vdy,new_Vdz

    do ia = inFirstAtom,inLastAtom
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
       if (JPBLOCK(i) /= MYNOD) goto 722
#endif
       if (inMovingCode(i) /= 0) goto 722

       if (ia /= inLastAtom .and. inIsDrude(i+1) &
            .and. inMass(i+1) /= ZERO) then ! We have zero mass for SCF
          mcm = inMass(i) + inMass(i+1)
          mcmi = ONE / mcm
          mredi = mcm / (inMass(i)*inMass(i+1))
          m1 = inMass(i) * mcmi
          m2 = inMass(i+1) * mcmi
          !           (v,vD) -> (vR,vd)
          VRx = m1 * ioVX(i) + m2 * ioVX(i+1)
          VRy = m1 * ioVY(i) + m2 * ioVY(i+1)
          VRz = m1 * ioVZ(i) + m2 * ioVZ(i+1)
          Vdx = ioVX(i+1) - ioVX(i)
          Vdy = ioVY(i+1) - ioVY(i)
          Vdz = ioVZ(i+1) - ioVZ(i)
          !           Propagate/Scale vR
          fact = inDeltaT*mcmi
          DRx = inDX(i) + inDX(i+1)
          DRy = inDY(i) + inDY(i+1)
          DRz = inDZ(i) + inDZ(i+1)
          !           if (QPCONFULL) then
          CALL ApplyRR( new_VRx,new_VRy,new_VRz, &
               RV1_CP,VRx,VRy,VRz, &
               -fact,RV2_CP,DRx,DRy,DRz )
          !           elseif (QZONLY) then
          !              new_VRx =               VRx - fact             * DRx
          !              new_VRy =               VRy - fact             * DRy
          !              new_VRz = RV1_CP(1,1) * VRz - fact*RV2_CP(1,1) * DRz
          !           else
          !              new_VRx = RV1_CP(1,1) * VRx - fact*RV2_CP(1,1) * DRx
          !              new_VRy = RV1_CP(1,1) * VRy - fact*RV2_CP(1,1) * DRy
          !              new_VRz = RV1_CP(1,1) * VRz - fact*RV2_CP(1,1) * DRz
          !           endif
          !           Propagate vd
          fact = inDeltaT*mredi
          Ddx = (ONE - m2)*inDX(i+1) - m2*inDX(i)
          Ddy = (ONE - m2)*inDY(i+1) - m2*inDY(i)
          Ddz = (ONE - m2)*inDZ(i+1) - m2*inDZ(i)
          new_Vdx = Vdx - fact*Ddx
          new_Vdy = Vdy - fact*Ddy
          new_Vdz = Vdz - fact*Ddz
          !           (vR,vd) -> (v,vD)
          ioVX(i) = new_VRx - m2*new_Vdx
          ioVY(i) = new_VRy - m2*new_Vdy
          ioVZ(i) = new_VRz - m2*new_Vdz
          ioVX(i+1) = new_VRx + (ONE - m2)*new_Vdx
          ioVY(i+1) = new_VRy + (ONE - m2)*new_Vdy
          ioVZ(i+1) = new_VRz + (ONE - m2)*new_Vdz
       elseif (inIsDrude(i)) then
          !           Nothing to do...
       else
          fact = inDeltaT/inMass(i)
          !           if (QPCONFULL) then
          CALL ApplyRR( ioVX(i),ioVY(i),ioVZ(i), &
               RV1_CP,ioVX(i),ioVY(i),ioVZ(i), &
               -fact,RV2_CP,inDX(i),inDY(i),inDZ(i) )
          !           elseif (QZONLY) then
          !              ioVX(i) = ioVX(i)             - fact*inDX(i)
          !              ioVY(i) = ioVY(i)             - fact*inDY(i)
          !              ioVZ(i) = ioVZ(i)*RV1_CP(1,1) - fact*inDZ(i)*RV2_CP(1,1)
          !           else
          !              ioVX(i) = ioVX(i)*RV1_CP(1,1) - fact*inDX(i)*RV2_CP(1,1)
          !              ioVY(i) = ioVY(i)*RV1_CP(1,1) - fact*inDY(i)*RV2_CP(1,1)
          !              ioVZ(i) = ioVZ(i)*RV1_CP(1,1) - fact*inDZ(i)*RV2_CP(1,1)
          !           endif
       endif
722    CONTINUE
    enddo

  END SUBROUTINE PropagateVCP


  SUBROUTINE PropagatePos( &
       inX, inY, inZ, &
       outX, outY, outZ, &
       inVX, inVY, inVZ, &
       inFirstAtom, inLastAtom, &
       inDeltaT, inMovingCode )
    !
    !     Propagates positions according to the formula:
    !     r_i <-- r_i + dt v_i
    !
    !     Global variables
    use parallel ! for JPBLOCK, MYNOD
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,atoml
#endif

    !     Arguments
    real(chm_real)  inX(*), inY(*), inZ(*)
    ! Input: Positions
    real(chm_real)  outX(*), outY(*), outZ(*)
    ! Output: Propagated ositions
    real(chm_real)  inVX(*), inVY(*), inVZ(*)
    ! Input: Velocities

    integer inFirstAtom       ! Input: Index of the first atom to propagate
    integer inLastAtom        ! Input: Index of the last atom to propagate

    real(chm_real)  inDeltaT          ! Input: Time step (dt)
    integer inMovingCode(*)   ! Input: 0 to allow the propagation

    !     Local variables
    integer i
    integer ia

    do ia = inFirstAtom,inLastAtom
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
       if (JPBLOCK(i) /= MYNOD) goto 723
#endif
       if (inMovingCode(i) /= 0) goto 723
       outX(i) = inX(i) + inDeltaT*inVX(i)
       outY(i) = inY(i) + inDeltaT*inVY(i)
       outZ(i) = inZ(i) + inDeltaT*inVZ(i)
723    CONTINUE
    enddo

  END SUBROUTINE PropagatePos


  SUBROUTINE AddArrays( &
       ioX, ioY, ioZ, &
       inX, inY, inZ, &
       inFirstAtom, inLastAtom, &
       inFactor )
    !
    !     Add the "inX" arrays to the "io" arrays
    !
    !     Global variables
    use parallel ! for JPBLOCK, MYNOD
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,atoml
#endif

    !     Arguments
    real(chm_real)  ioX(*), ioY(*), ioZ(*)
    real(chm_real)  inX(*), inY(*), inZ(*)
    integer inFirstAtom       ! Input: Index of the first to copy
    integer inLastAtom        ! Input: Index of the last to copy
    real(chm_real)  inFactor

    !     Local variable
    integer i
    integer ia

#if KEY_DOMDEC==1
    if (q_domdec) then
       do ia = inFirstAtom,inLastAtom
          i = atoml(ia)
          ioX(i) = ioX(i) + inFactor*inX(i)
          ioY(i) = ioY(i) + inFactor*inY(i)
          ioZ(i) = ioZ(i) + inFactor*inZ(i)
       enddo
    else
#endif
       do i = inFirstAtom,inLastAtom
#if KEY_PARASCAL==1
          if (JPBLOCK(i) /= MYNOD) goto 724
#endif
          ioX(i) = ioX(i) + inFactor*inX(i)
          ioY(i) = ioY(i) + inFactor*inY(i)
          ioZ(i) = ioZ(i) + inFactor*inZ(i)
#if KEY_PARASCAL==1
724       CONTINUE
#endif
       enddo
#if KEY_DOMDEC==1
    endif
#endif

  END SUBROUTINE AddArrays


  SUBROUTINE CopyArrays( &
       inX, inY, inZ, &
       outX, outY, outZ, &
       inFirstAtom, inLastAtom )
    !
    !     Copy the "in" arrays to the "out" arrays
    !
    !     Global variables
    use parallel ! for JPBLOCK, MYNOD
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,atoml
#endif

    !     Arguments
    real(chm_real)  inX(*), inY(*), inZ(*)
    real(chm_real)  outX(*), outY(*), outZ(*)
    integer inFirstAtom       ! Input: Index of the first to copy
    integer inLastAtom        ! Input: Index of the last to copy


    !     Local variable
    integer i
    integer ia

#if KEY_DOMDEC==1
    if (q_domdec) then
       do ia = inFirstAtom,inLastAtom
          i = atoml(ia)
          outX(i) = inX(i)
          outY(i) = inY(i)
          outZ(i) = inZ(i)
       enddo
    else
#endif
       do i = inFirstAtom,inLastAtom
#if KEY_PARASCAL==1
          if (JPBLOCK(i) /= MYNOD) goto 725
#endif
          outX(i) = inX(i)
          outY(i) = inY(i)
          outZ(i) = inZ(i)
#if KEY_PARASCAL==1
725       CONTINUE
#endif
       enddo
#if KEY_DOMDEC==1
    endif
#endif

  END SUBROUTINE CopyArrays


  SUBROUTINE ZeroArrays( &
       outX, outY, outZ, &
       inFirstAtom, inLastAtom )
    !
    !     Set all arrays to ZERO.
    !
    !     Global variables
    use parallel ! for JPBLOCK, MYNOD
    use number ! for ZERO
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,atoml
#endif

    !     Arguments
    real(chm_real)  outX(*), outY(*), outZ(*)
    integer inFirstAtom       ! Input: Index of the first to copy
    integer inLastAtom        ! Input: Index of the last to copy

    !     Local variable
    integer i
    integer ia

#if KEY_DOMDEC==1
    if (q_domdec) then
       do ia = inFirstAtom,inLastAtom
          i = atoml(ia)
          outX(i) = ZERO
          outY(i) = ZERO
          outZ(i) = ZERO
       enddo
    else
#endif
       do i = inFirstAtom,inLastAtom
#if KEY_PARASCAL==1
          if (JPBLOCK(i) /= MYNOD) goto 726
#endif
          outX(i) = ZERO
          outY(i) = ZERO
          outZ(i) = ZERO
#if KEY_PARASCAL==1
726       CONTINUE
#endif
       enddo
#if KEY_DOMDEC==1
    endif
#endif

  END SUBROUTINE ZeroArrays

  SUBROUTINE ComputeRR( &
       ZETA_CP, dt, nf, QPCONFULL, &
       R1_CP, R2_CP, XTLABC )
    !
    !     Compute RX1_CP and RX2_CP (by using nf=0 and dt)
    !     Compute RV1_CP and RV2_CP (by using nf and -dt/2 instead of dt)
    !

    !     Global variables
    use dimens_fcm
    use number

    !     Arguments
    real(chm_real)  ZETA_CP(3,3)      ! INPUT
    real(chm_real)  dt                ! INPUT
    integer nf                ! INPUT
    logical QPCONFULL         ! INPUT
    real(chm_real)  R1_CP(3,3)        ! OUTPUT
    real(chm_real)  R2_CP(3,3)        ! OUTPUT
    real(chm_real)  XTLABC(6)         ! INPUT (for nf=0)
    !     Local variables
    real(chm_real)  overnf,temp,fact,poly,tr
    real(chm_real), parameter :: OVERFOURTY2 = ONE/(THIRTY+TWELVE) ! 1/42
    real(chm_real), parameter :: OVERSVNTY2 = ONE/SVNTY2 ! 1/72
    integer i
    real(chm_real)  z(3,3),zeval(3),zevec(3,3),zevect(3,3)
    integer errcode
    !     Locals for removing the rotation
    real(chm_real)  h(3,3),ht(3,3),hth(3,3),uvec(3,3),uvect(3,3), &
         rot(3,3)

    if (nf == 0) then
       overnf = ZERO
    else
       overnf = ONE/nf
    endif

    if (QPCONFULL) then
       CALL COPY33(Z,ZETA_CP)
       CALL EIGRS(Z,3,11,zeval,zevec,3,errcode)
    else
       zeval(1) = ZETA_CP(1,1)
       zeval(2) = ZETA_CP(2,2)
       zeval(3) = ZETA_CP(3,3)
       CALL IDENT33(zevec)
    endif
    !     CALL INVT33(zevect,zevec,ok)
    CALL TRANSPS(zevect,zevec,3,3) ! The inverse is simply the transposed

    !     Compute R1 and R2 from zeval and zevec
    tr = overnf * (zeval(1)+zeval(2)+zeval(3))
    CALL IDENT33(R1_CP)
    CALL IDENT33(R2_CP)
    do i = 1,3
       temp = (zeval(i)+tr)*dt*HALF
       fact = EXP(temp)
       R1_CP(i,i) = fact**2
       temp = temp**2
       poly = ONE + temp*SIXTH*(ONE + temp*PT05*(ONE &
            + temp*OVERFOURTY2*(ONE + temp*OVERSVNTY2))) ! Taylor expansion of sinh(x)/x
       R2_CP(i,i) = fact*poly
    enddo
    CALL MULT33(R1_CP,R1_CP,zevect)
    CALL MULT33(R2_CP,R2_CP,zevect)
    CALL MULT33(R1_CP,zevec ,R1_CP)
    CALL MULT33(R2_CP,zevec ,R2_CP)

    !     Symmetrize R1 and R2, just in case
    R1_CP(1,2) = (R1_CP(1,2)+R1_CP(2,1))*HALF
    R1_CP(1,3) = (R1_CP(1,3)+R1_CP(3,1))*HALF
    R1_CP(2,1) =  R1_CP(1,2)
    R1_CP(2,3) = (R1_CP(2,3)+R1_CP(3,2))*HALF
    R1_CP(3,1) =  R1_CP(1,3)
    R1_CP(3,2) =  R1_CP(2,3)

    R2_CP(1,2) = (R2_CP(1,2)+R2_CP(2,1))*HALF
    R2_CP(1,3) = (R2_CP(1,3)+R2_CP(3,1))*HALF
    R2_CP(2,1) =  R2_CP(1,2)
    R2_CP(2,3) = (R2_CP(2,3)+R2_CP(3,2))*HALF
    R2_CP(3,1) =  R2_CP(1,3)
    R2_CP(3,2) =  R2_CP(2,3)

    !     Remove the rotation from RX1
    if (nf == 0) then
       h(1,1) = XTLABC(1)
       h(1,2) = XTLABC(2)
       h(1,3) = XTLABC(4)
       h(2,1) = XTLABC(2)
       h(2,2) = XTLABC(3)
       h(2,3) = XTLABC(5)
       h(3,1) = XTLABC(4)
       h(3,2) = XTLABC(5)
       h(3,3) = XTLABC(6)
       CALL MULT33(h,R1_CP,h)
       CALL TRANSPS(ht,h,3,3)
       CALL MULT33(hth,ht,h)
       CALL EIGRS(hth,3,11,zeval,zevec,3,errcode)
       zeval(1) = SQRT(zeval(1))
       zeval(2) = SQRT(zeval(2))
       zeval(3) = SQRT(zeval(3))
       CALL MULT33(uvec,h,zevec)
       uvec(1,1) = uvec(1,1) / zeval(1)
       uvec(2,1) = uvec(2,1) / zeval(1)
       uvec(3,1) = uvec(3,1) / zeval(1)
       uvec(1,2) = uvec(1,2) / zeval(2)
       uvec(2,2) = uvec(2,2) / zeval(2)
       uvec(3,2) = uvec(3,2) / zeval(2)
       uvec(1,3) = uvec(1,3) / zeval(3)
       uvec(2,3) = uvec(2,3) / zeval(3)
       uvec(3,3) = uvec(3,3) / zeval(3)
       CALL TRANSPS(uvect,uvec,3,3)
       CALL MULT33(rot,zevec,uvect)
       CALL MULT33(R1_CP,rot,R1_CP)

       !        trace = 1 + 2 cos(theta)
       !        a = (rot(1,1)+rot(2,2)+rot(3,3)-ONE)*HALF
       !        if (ABS(a-ONE) > TENM14) then
       !           write(*,*)'TransfXTL: Finite rotation removed'
       !           write(*,*)'TransfXTL: rot',rot(1,1),rot(1,2),rot(1,3)
       !           write(*,*)'TransfXTL: rot',rot(2,1),rot(2,2),rot(2,3)
       !           write(*,*)'TransfXTL: rot',rot(3,1),rot(3,2),rot(3,3)
       !           write(*,*)'TransfXTL: Rotation angle',a
       !        endif
    endif

    return
  END SUBROUTINE ComputeRR


  SUBROUTINE ApplyRR( &
       xnew,ynew,znew, RX1_CP,x,y,z, dt,RX2_CP,vx,vy,vz )
    !
    !     Rnew = RX1 R + dt RX2 V
    !     Note: ApplyRR(R,RX1,R,...) is allowed.
    !
    !     Arguments
    real(chm_real)  xnew,ynew,znew    ! OUTPUT
    real(chm_real)  RX1_CP(3,3)       ! INPUT
    real(chm_real)  x,y,z             ! INPUT
    real(chm_real)  dt,RX2_CP(3,3)    ! INPUT
    real(chm_real)  vx,vy,vz          ! INPUT
    !     Local variables
    real(chm_real)  xn,yn,zn

    xn = RX1_CP(1,1)*x + RX1_CP(1,2)*y + RX1_CP(1,3)*z &
         + dt*(RX2_CP(1,1)*vx + RX2_CP(1,2)*vy + RX2_CP(1,3)*vz)
    yn = RX1_CP(2,1)*x + RX1_CP(2,2)*y + RX1_CP(2,3)*z &
         + dt*(RX2_CP(2,1)*vx + RX2_CP(2,2)*vy + RX2_CP(2,3)*vz)
    zn = RX1_CP(3,1)*x + RX1_CP(3,2)*y + RX1_CP(3,3)*z &
         + dt*(RX2_CP(3,1)*vx + RX2_CP(3,2)*vy + RX2_CP(3,3)*vz)
    xnew = xn
    ynew = yn
    znew = zn
    return
  END SUBROUTINE ApplyRR


  SUBROUTINE TransfXTL(RX1_CP,XTLABC,XTLTYP,XTLREF,XUCELL,XDIM)
    !
    !     Apply transformation RX1 to XTLABC
    !
    use number
    use stream
    !     Arguments
    real(chm_real)      RX1_CP(3,3)   ! INPUT
    real(chm_real)      XTLABC(6)     ! INPUT/OUTPUT
    character(len=4) XTLTYP           ! INPUT
    real(chm_real)      XTLREF(6)     ! INPUT (OUTPUT if the crystal symmetry is not preserved)
    real(chm_real)      XUCELL(6)     ! INPUT (OUTPUT if the crystal symmetry is not preserved)
    integer     XDIM          ! INPUT
    !     Local variables
    real(chm_real)      h(3,3),ht(3,3),hth(3,3)
    real(chm_real)      eval(3),evec(3,3),a,b,c
    real(chm_real)      uvec(3,3),uvect(3,3),rot(3,3)
    integer     errorcode

    h(1,1) = XTLABC(1)
    h(1,2) = XTLABC(2)
    h(1,3) = XTLABC(4)
    h(2,1) = XTLABC(2)
    h(2,2) = XTLABC(3)
    h(2,3) = XTLABC(5)
    h(3,1) = XTLABC(4)
    h(3,2) = XTLABC(5)
    h(3,3) = XTLABC(6)
    CALL MULT33(h,RX1_CP,h)

    !     Apply a rotation so that h is symmetric
    !     (for debugging purposes only; there shouldn't be any rotation)
    CALL TRANSPS(ht,h,3,3)
    CALL MULT33(hth,ht,h)
    CALL EIGRS(hth,3,11,eval,evec,3,errorcode)
    a = SQRT(eval(1))
    b = SQRT(eval(2))
    c = SQRT(eval(3))
    CALL MULT33(uvec,h,evec)
    uvec(1,1) = uvec(1,1) / a
    uvec(2,1) = uvec(2,1) / a
    uvec(3,1) = uvec(3,1) / a
    uvec(1,2) = uvec(1,2) / b
    uvec(2,2) = uvec(2,2) / b
    uvec(3,2) = uvec(3,2) / b
    uvec(1,3) = uvec(1,3) / c
    uvec(2,3) = uvec(2,3) / c
    uvec(3,3) = uvec(3,3) / c
    CALL TRANSPS(uvect,uvec,3,3)
    CALL MULT33(rot,evec,uvect)
    CALL MULT33(h,rot,h)

    !     trace = 1 + 2 cos(theta)
    a = (rot(1,1)+rot(2,2)+rot(3,3)-ONE)*HALF
    if (ABS(a-ONE) > TENM14) then
       write(OUTU,'(a)')'TransfXTL: Finite rotation!'
       write(OUTU,'(a)')'TransfXTL: This should not happen!'
       write(OUTU,'(a,3f12.4)') &
            'TransfXTL: rot',rot(1,1),rot(1,2),rot(1,3)
       write(OUTU,'(a,3f12.4)') &
            'TransfXTL: rot',rot(2,1),rot(2,2),rot(2,3)
       write(OUTU,'(a,3f12.4)') &
            'TransfXTL: rot',rot(3,1),rot(3,2),rot(3,3)
       write(OUTU,'(a,f12.4)')'TransfXTL: Rotation angle',a
    endif

    XTLABC(1) =  h(1,1)
    XTLABC(2) = (h(1,2) + h(2,1))*HALF
    XTLABC(3) =  h(2,2)
    XTLABC(4) = (h(1,3) + h(3,1))*HALF
    XTLABC(5) = (h(2,3) + h(3,2))*HALF
    XTLABC(6) =  h(3,3)

    !     Check if the crystal symmetry is preserved
    CALL XTLSYM(XTLABC,XUCELL,XTLTYP,XDIM,XTLREF)

    return
  END SUBROUTINE TransfXTL


  SUBROUTINE MULT33(R, A, B)
    !     Compute R = A B
    !     Note: MULT33(R,R,B) and MULT33(R,A,R) are allowed.
    use number
    !     Arguments
    real(chm_real)  R(3,3)            ! OUTPUT
    real(chm_real)  A(3,3),B(3,3)     ! INPUT
    !     Local variables
    real(chm_real)  R2(3,3)
    integer i,j,k

    do i = 1,3
       do j = 1,3
          R2(j,i) = ZERO
          do k = 1,3
             R2(j,i) = R2(j,i) + A(j,k)*B(k,i)
          enddo
       enddo
    enddo
    CALL COPY33(R,R2)
    return
  END SUBROUTINE MULT33


  SUBROUTINE COPY33(COPY,A)
    !     Copy 3x3 matrix A into COPY
    !     Arguments
    real(chm_real)  COPY(3,3)         ! OUTPUT
    real(chm_real)  A(3,3)            ! INPUT

    COPY(1,1) = A(1,1)
    COPY(1,2) = A(1,2)
    COPY(1,3) = A(1,3)
    COPY(2,1) = A(2,1)
    COPY(2,2) = A(2,2)
    COPY(2,3) = A(2,3)
    COPY(3,1) = A(3,1)
    COPY(3,2) = A(3,2)
    COPY(3,3) = A(3,3)
    return
  END SUBROUTINE COPY33


  SUBROUTINE IDENT33(A)
    !     Sets A to the 3x3 identity matrix
    use number
    !     Arguments
    real(chm_real)  A(3,3)            ! OUTPUT

    A(1,1) = ONE
    A(1,2) = ZERO
    A(1,3) = ZERO
    A(2,1) = ZERO
    A(2,2) = ONE
    A(2,3) = ZERO
    A(3,1) = ZERO
    A(3,2) = ZERO
    A(3,3) = ONE
    return
  END SUBROUTINE IDENT33


  SUBROUTINE DAFFINE33(R, c, f, A)
    !     Compute R = c I + f A
    !     (where I is the identity matrix)
    !     Arguments
    real(chm_real)  R(3,3)            ! OUTPUT
    real(chm_real)  c                 ! INPUT
    real(chm_real)  f                 ! INPUT
    real(chm_real)  A(3,3)            ! INPUT

    R(1,1) = c + f*A(1,1)
    R(1,2) =     f*A(1,2)
    R(1,3) =     f*A(1,3)
    R(2,1) =     f*A(2,1)
    R(2,2) = c + f*A(2,2)
    R(2,3) =     f*A(2,3)
    R(3,1) =     f*A(3,1)
    R(3,2) =     f*A(3,2)
    R(3,3) = c + f*A(3,3)
    return
  END SUBROUTINE DAFFINE33


  SUBROUTINE AFFINE33(R, C, f, A)
    !     Compute R = C + f A
    !     Arguments
    real(chm_real)  R(3,3)            ! OUTPUT
    real(chm_real)  C(3,3)            ! INPUT
    real(chm_real)  f                 ! INPUT
    real(chm_real)  A(3,3)            ! INPUT

    R(1,1) = C(1,1) + f*A(1,1)
    R(1,2) = C(1,2) + f*A(1,2)
    R(1,3) = C(1,3) + f*A(1,3)
    R(2,1) = C(2,1) + f*A(2,1)
    R(2,2) = C(2,2) + f*A(2,2)
    R(2,3) = C(2,3) + f*A(2,3)
    R(3,1) = C(3,1) + f*A(3,1)
    R(3,2) = C(3,2) + f*A(3,2)
    R(3,3) = C(3,3) + f*A(3,3)
    return
  END SUBROUTINE AFFINE33


  SUBROUTINE ShakeRoll( &
       ATFRST,ATLAST, &
       X0,Y0,Z0, Xt,Yt,Zt, VXt2,VYt2,VZt2, &
       AMASS,IMOVE, dt, mass_weighting, &
       lambda, DXconst,DYconst,DZconst, ISKPR, &
       QPCON,QZONLY,QPCONFULL, ZETA_CP, W_CP, RX1_CP,RX2_CP,RV2_CP )
    !
    !     It is a generalization of the SHAKE algorithm applicable to systems
    !     with fluctuating volume, i.e. for which the coordinate propagation is
    !     due also to a scale transformation.
    !     [See SHAKEA2, the original subroutine.]
    !
    !     Reference:
    !     G.J. Martyna, M.E. Tuckerman, D.J. Tobias, M.L. Klein
    !     Mol.Phys. 87 (5), 1117-1157 (1996) [Appendix D]
    !

    !     Global variables
    use number
    use shake ! for MXITER,NCONST,MAXSHK,SHKAPR,CONSTR,SHKTOL
    use parallel ! for JPBLOCK, MYNOD
    use stream ! for OUTU
    use lonepr
    use machutil,only:die


    !     Arguments
    integer ATFRST            ! INPUT  Index of first atom
    integer ATLAST            ! INPUT  Index of last atom

    real(chm_real)  X0(*),Y0(*),Z0(*) ! INPUT  Atomic coordinates at t  (ATFRST,ATLAST)

    real(chm_real)  Xt(*),Yt(*),Zt(*) ! I/O    Atomic coordinates at t+dt  (ATFRST,ATLAST)
    !        IN:  Coordinates at t+dt
    !        OUT: Coordinates verifying the constraints

    real(chm_real)  VXt2(*),VYt2(*),VZt2(*)
    ! I/O    Atomic velocities at t+dt/2  (ATFRST,ATLAST)
    !        IN:  Velocities at t+dt/2
    !        OUT: Velocities corrected for the SHAKE forces

    real(chm_real)  AMASS(*)          ! INPUT  Atomic masses  (ATFRST,ATLAST)
    integer IMOVE(*)          ! INPUT  -1,0,1  (ATFRST,ATLAST)
    real(chm_real)  dt                ! INPUT  Time step
    logical mass_weighting    ! INPUT  .true. to use mass weighting

    real(chm_real)  lambda(*)         ! OUTPUT Lagrange multipliers  (1,NCONST)
    real(chm_real)  DXconst(*)        ! INPUT  Constraint forces  (ATFRST,ATLAST)
    real(chm_real)  DYconst(*)
    real(chm_real)  DZconst(*)

    real(chm_real) ISKPR(*)           ! I/O    "Skip" tags  (ATFRST,ATLAST)

    logical QPCON             ! INPUT
    logical QZONLY            ! INPUT
    logical QPCONFULL         ! INPUT
    real(chm_real)  ZETA_CP(3,3)      ! INPUT !??? not used...
    real(chm_real)  W_CP              ! INPUT !??? not used...
    real(chm_real)  RX1_CP(3,3)       ! INPUT
    real(chm_real)  RX2_CP(3,3)       ! INPUT
    real(chm_real)  RV2_CP(3,3)       ! INPUT


    !     Local variables
    integer iconst
    integer nconst2
    integer iconst2
#if KEY_PARALLEL==1
    integer MAP(MAXSHK)       ! Constraints list
    logical error
#endif
    logical done
    integer iter,niter
    integer I,J,K             ! Atom indices

    real(chm_real)  bX0,bY0,bZ0       ! Bond vector at time 0
    real(chm_real)  bXt,bYt,bZt       ! Bond vector at time dt
    real(chm_real)  FcX0,FcY0,FcZ0    ! Constraint force at time 0
    real(chm_real)  FcXt,FcYt,FcZt    ! Constraint force at time dt
    real(chm_real)  bond2             ! Bond length
    real(chm_real)  bond2d            ! Bond length dictated by SHAKE
    real(chm_real)  sigma             ! Bond length constraint (should be zero)

    real(chm_real)  RXV(3,3)
    real(chm_real)  rf1,rf2,rf3
    real(chm_real)  frf, dlambda
    real(chm_real)  tolerance
    real(chm_real)  proj,acor,amsi,amsj
    real(chm_real)  fact,virial


    niter = 0
    tolerance = TWO * SHKTOL

    !     Prepare parallel constraints list
#if KEY_PARALLEL==1
    nconst2 = 0
    error = .false.
    loop90: DO iconst = 1,NCONST
       I = SHKAPR(1,iconst)
       IF (I == 0) cycle loop90
       J = SHKAPR(2,iconst)
#if KEY_PARAFULL==1
       IF (I < ATFRST .OR. I > ATLAST) THEN
          IF (J < ATFRST .OR. J > ATLAST) cycle loop90
          error = .true.
          cycle loop90
       ENDIF
       IF (J < ATFRST .OR. J > ATLAST) THEN
          error = .true.
          cycle loop90
       ENDIF
#elif KEY_PARASCAL==1
       IF (JPBLOCK(I) /= MYNOD) THEN
          IF (JPBLOCK(J) /= MYNOD) cycle loop90
          error = .true.
          cycle loop90
       ENDIF
       IF(JPBLOCK(J) /= MYNOD) THEN
          error = .true.
          cycle loop90
       ENDIF
#endif
       IF (error) THEN
          CALL WRNDIE(0,'<ShakeRoll>', &
               'Some contraints across parallel partitions')
       ENDIF
       nconst2 = nconst2 + 1
       MAP(nconst2) = iconst
    enddo loop90
#endif

    do I = ATFRST,ATLAST
       ISKPR(I) = 0.0
    enddo

    !
    !     SHAKE/ROLL iteration
    !
    loop902: do iter = 1,MXITER

       virial = ZERO
       done = .true.

       !        Loop over constraints
#if KEY_PARALLEL==0
       nconst2=nconst
#endif
       loop100: DO iconst2 = 1,nconst2
#if KEY_PARALLEL==1
          iconst = MAP(iconst2)
#endif
#if KEY_PARALLEL==0
          iconst = iconst2
#endif

          !           Get atom indices of the pair
          I = SHKAPR(1,iconst)
          if (I == 0) cycle loop100
          J = SHKAPR(2,iconst)

          !           "Cycle" if the pair is "done" or if it is fixed
          IF (ISKPR(I)+ISKPR(J) <= -1.9)cycle loop100
          IF (IMOVE(I) > 0 .AND. IMOVE(J).GT.0) cycle loop100

          !           Compute the bond vector.
          bXt = Xt(I)-Xt(J)
          bYt = Yt(I)-Yt(J)
          bZt = Zt(I)-Zt(J)
          FcXt = -TWO*bXt
          FcYt = -TWO*bYt
          FcZt = -TWO*bZt

          !           "Cycle" if sigma = 0 (to a given tolerance)
          bond2 = bXt*bXt + bYt*bYt + bZt*bZt
          bond2d = CONSTR(iconst)
          sigma = bond2 - bond2d

          IF (ABS(sigma)  <  bond2d*tolerance) cycle loop100

          !           Compute the "reference" bond vector (transformed).
          if (QPCON) then
             bX0 = (X0(I)-X0(J))
             bY0 = (Y0(I)-Y0(J))
             bZ0 = (Z0(I)-Z0(J))
          else
             bX0 = X0(I)-X0(J)
             bY0 = Y0(I)-Y0(J)
             bZ0 = Z0(I)-Z0(J)
          endif
          FcX0 = -TWO*bX0
          FcY0 = -TWO*bY0
          FcZ0 = -TWO*bZ0

          !           Project the new bond to the "reference" bond.
          proj = bX0*bXt + bY0*bYt + bZ0*bZt
          !           If the projection is too small:
          if (proj  <  bond2d*0.000001D0) then
             if (WRNLEV >= 2) then
321             FORMAT(' ** ERROR IN ShakeRoll **', &
                     ' DEVIATION IN SHAKE TOO LARGE'/ &
                     ' NITER=',I5,' LL=',I5,' I=',I5,' J=',I5,/ &
                     ' TOLER=',F18.10,' DIFF=',F18.10,/ &
                     ' RRPR=',F18.10)
                WRITE(OUTU,321) iter,iconst,I,J,bond2d,sigma,proj

323             FORMAT(2X,A,3F20.10)
                WRITE(OUTU,323) 'X(I)   ',Xt(I),Yt(I),Zt(I)
                WRITE(OUTU,323) 'XREF(I)',X0(I),Y0(I),Z0(I)
                WRITE(OUTU,323) 'X(J)   ',Xt(J),Yt(J),Zt(J)
                WRITE(OUTU,323) 'XREF(J)',X0(J),Y0(J),Z0(J)
                CALL DIE
             endif
             !              Instead of dying, use current displacement instead of reference
             !              vector.  This won't conserve energy, but should only be needed in
             !              high energy situations.  - BRB (from RCZ)
             acor = SQRT(bond2d/bond2)
             bX0 = acor * bXt
             bY0 = acor * bYt
             bZ0 = acor * bZt
             proj = bX0*bXt + bY0*bYt + bZ0*bZt
          endif

          !           Compute the Lagrange multiplier:
          !           1) Compute the inertia factors
          if (IMOVE(I) == 0) then
             if (mass_weighting) then
                amsi = ONE/AMASS(I)
             else
                amsi = ONE
             endif
          else
             amsi = ZERO
          endif
          if (IMOVE(J) == 0) then
             if (mass_weighting) then
                amsj = ONE/AMASS(J)
             else
                amsj = ONE
             endif
          else
             amsj = ZERO
          endif
          !           2) Compute the increment
          if (QPCONFULL) then
             !              r = RV2 RX2 Fc0
             !              (see subroutine ShakeForces)
             CALL MULT33(RXV,RV2_CP,RX2_CP)
             rf1 = RXV(1,1)*FcX0 + RXV(1,2)*FcY0 + RXV(1,3)*FcZ0
             rf2 = RXV(2,1)*FcX0 + RXV(2,2)*FcY0 + RXV(2,3)*FcZ0
             rf3 = RXV(3,1)*FcX0 + RXV(3,2)*FcY0 + RXV(3,3)*FcZ0
          elseif (QZONLY) then
             rf1 = FcX0
             rf2 = FcY0
             rf3 = RX2_CP(3,3)*RV2_CP(3,3)*FcZ0
          else
             rf1 = RX2_CP(1,1)*RV2_CP(1,1)*FcX0
             rf2 = RX2_CP(2,2)*RV2_CP(2,2)*FcY0
             rf3 = RX2_CP(3,3)*RV2_CP(3,3)*FcZ0
          endif
          if (iter == 1) then
             dlambda = lambda(iconst)
          else
             if (PRNLEV >= 2 .AND. SHKSCA /= ONE) &
                  WRITE(OUTU,*) 'ERROR: SHKSCA ne 1.0'
             frf = rf1*FcXt + rf2*FcYt + rf3*FcZt
             dlambda = TWO*sigma / (dt*dt*(amsi+amsj)*frf)
             lambda(iconst) = lambda(iconst) + dlambda
          endif

          !           Compute new coordinates
          fact = HALF*dt*dt*amsi*dlambda
          Xt(I) = Xt(I) + fact*rf1
          Yt(I) = Yt(I) + fact*rf2
          Zt(I) = Zt(I) + fact*rf3
          fact = HALF*dt*dt*amsj*dlambda
          Xt(J) = Xt(J) - fact*rf1
          Yt(J) = Yt(J) - fact*rf2
          Zt(J) = Zt(J) - fact*rf3
          !           Set ISKPR flags
          ISKPR(I) = 1.0
          ISKPR(J) = 1.0


          bXt = Xt(I)-Xt(J)
          bYt = Yt(I)-Yt(J)
          bZt = Zt(I)-Zt(J)
          bond2 = bXt*bXt + bYt*bYt + bZt*bZt
          sigma = bond2 - bond2d


       ENDDO loop100
       !        End of loop over constraints


       !        Get ISKPR flags ready for next iteration
       !        Exit if all constraints are "done"
       done = .true.
       do K = ATFRST,ATLAST
          if (ISKPR(K) >= -0.1) ISKPR(K) = ISKPR(K) - 1.0
          done = done .and. (ISKPR(K) <= -0.9)
       enddo
       if (done) then
          if (PRNLEV > 5) write(OUTU,*) &
               'SHAKE converged in',ITER,' iterations'
          exit loop902
       endif

    enddo loop902


    !     Signal if SHAKE has not converged
    if (iter >= MXITER) then
311    FORMAT(' ***** ERROR IN ShakeRoll ***** COORDINATE RESETTING', &
            ' WAS NOT ACCOMPLISHED IN',I6,' ITERATIONS',/)
       if (WRNLEV >= 2) WRITE(OUTU,311) MXITER
       CALL DIEWRN(-2)
       return
    endif


#if KEY_LONEPAIR==1
    !     LONEPRC() is not yet parallel -> broadcast the coordinates
#if KEY_PARALLEL==1
    !C      call vdgbr(xt,yt,zt,0)
#endif
    IF(NUMLP > 0) THEN
       CALL LONEPRC(ATFRST,ATLAST, Xt,Yt,Zt, X0,Y0,Z0, AMASS, &
            NUMLP,LPNHOST,LPHPTR,LPHOST,LPVALUE,LPWGHT )
    ENDIF
#endif

    RETURN
  END subroutine ShakeRoll


  SUBROUTINE RattleRoll( &
       ATFRST,ATLAST, &
       Xt,Yt,Zt, VXt,VYt,VZt, &
       AMASS,IMOVE, dt, mass_weighting, &
       lambda, ISKPR, &
       QPCON,QZONLY,QPCONFULL, ZETA_CP, W_CP, RV2_CP, &
       XTLTYP,XTLREF )

    !     Global variables
    use number
    use shake ! for MXITER,NCONST,MAXSHK,SHKAPR,CONSTR,SHKTOL
    use parallel ! for JPBLOCK, MYNOD
    use stream ! for OUTU

    !     Arguments
    integer ATFRST            ! INPUT  Index of first atom
    integer ATLAST            ! INPUT  Index of last atom

    real(chm_real)  Xt(*),Yt(*),Zt(*) ! INPUT  Atomic coordinates  (ATFRST,ATLAST)

    real(chm_real)  VXt(*),VYt(*),VZt(*)
    ! I/O    Atomic velocities at t+dt  (ATFRST,ATLAST)
    !        IN:  Velocities at t+dt
    !        OUT: Velocities verifying the constraints

    real(chm_real)  AMASS(*)          ! INPUT  Atomic masses  (ATFRST,ATLAST)
    integer IMOVE(*)          ! INPUT  -1,0,1  (ATFRST,ATLAST)
    real(chm_real)  dt                ! INPUT  Time step
    logical mass_weighting    ! INPUT  .true. to use mass weighting

    real(chm_real)  lambda(*)         ! OUTPUT Lagrange multipliers  (1,NCONST)

    real(chm_real) ISKPR(*)           ! I/O    "Skip" tags  (ATFRST,ATLAST)

    logical QPCON             ! INPUT
    logical QZONLY            ! INPUT
    logical QPCONFULL         ! INPUT
    real(chm_real)  ZETA_CP(3,3)      ! INPUT
    real(chm_real)  W_CP              ! INPUT
    real(chm_real)  RV2_CP(3,3)       ! INPUT
    character(len=4) XTLTYP           ! INPUT
    real(chm_real)  XTLREF(6)         ! INPUT

    !     Local variables
    integer iconst,nconst2,iconst2,kstart,kstop
#if KEY_PARALLEL==1
    integer MAP(MAXSHK)       ! Constraints list
    logical error
#endif
    logical done
    integer iter,niter
    integer I,J,K         ! Atom indices

    real(chm_real)  bXt,bYt,bZt       ! Bond vector at time dt
    real(chm_real)  bVXt,bVYt,bVZt    ! Bond velocity at time dt
    real(chm_real)  FcXt,FcYt,FcZt    ! Constraint force at time dt
    real(chm_real)  bond2             ! Bond length
    real(chm_real)  bond2d            ! Bond length dictated by SHAKE
    real(chm_real)  dsigma            ! Bond velocity constraint (should be zero)

    real(chm_real)  frf, dlambda
    real(chm_real)  tolerance
    real(chm_real)  amsi,amsj
    real(chm_real)  fact,dvir(3,3),vir(9)
    real(chm_real)  g1,g2,g3,f1,f2,f3

    niter = 0
    tolerance = TWO * SHKTOL


    !     Prepare parallel constraints list
#if KEY_PARALLEL==1
    nconst2 = 0
    error = .false.
    loop90: DO iconst = 1,NCONST
       I = SHKAPR(1,iconst)
       IF (I == 0) cycle loop90
       J = SHKAPR(2,iconst)
#if KEY_PARAFULL==1
       IF (I < ATFRST .OR. I > ATLAST) THEN
          IF (J < ATFRST .OR. J > ATLAST) cycle loop90
          error = .true.
          cycle loop90
       ENDIF
       IF (J < ATFRST .OR. J > ATLAST) THEN
          error = .true.
          cycle loop90
       ENDIF
#elif KEY_PARASCAL==1
       IF (JPBLOCK(I) /= MYNOD) THEN
          IF (JPBLOCK(J) /= MYNOD) cycle loop90
          error = .true.
          cycle loop90
       ENDIF
       IF(JPBLOCK(J) /= MYNOD) THEN
          error = .true.
          cycle loop90
       ENDIF
#endif
       IF (error) THEN
          CALL WRNDIE(0,'<RattleRoll>', &
               'Some contraints across parallel partitions')
       ENDIF
       nconst2 = nconst2 + 1
       MAP(nconst2) = iconst
    enddo loop90
#endif

    do I = ATFRST,ATLAST
       ISKPR(I) = zero
    enddo

    !
    !     RATTLE/ROLL iteration
    !
    do iter = 1,MXITER

       vir(1) = ZERO
       vir(2) = ZERO
       vir(3) = ZERO
       vir(4) = ZERO
       vir(5) = ZERO
       vir(6) = ZERO
       vir(7) = ZERO
       vir(8) = ZERO
       vir(9) = ZERO
       done = .true.
       !        Loop over constraints
#if KEY_PARALLEL==0
       nconst2 = nconst
#endif
       loop100: DO iconst2 = 1,nconst2
#if KEY_PARALLEL==1
          iconst = MAP(iconst2)
#endif
#if KEY_PARALLEL==0
          iconst = iconst2
#endif

          !           Get atom indices of the pair
          I = SHKAPR(1,iconst)
          if (I == 0) cycle loop100
          J = SHKAPR(2,iconst)

          !           "Cycle" if the pair is "done" or if it is fixed
          IF (ISKPR(I)+ISKPR(J) <= -1.9) cycle loop100
          IF (IMOVE(I) > 0 .AND. IMOVE(J).GT.0) cycle loop100

          !           Compute the bond vector and its time derivative.
          bXt = Xt(I)-Xt(J)
          bYt = Yt(I)-Yt(J)
          bZt = Zt(I)-Zt(J)
          FcXt = -TWO*bXt
          FcYt = -TWO*bYt
          FcZt = -TWO*bZt
          if (QPCON) then
             if (QPCONFULL) then
                g1=bXt*ZETA_CP(1,1)+bYt*ZETA_CP(1,2)+bZt*ZETA_CP(1,3)
                g2=bXt*ZETA_CP(2,1)+bYt*ZETA_CP(2,2)+bZt*ZETA_CP(2,3)
                g3=bXt*ZETA_CP(3,1)+bYt*ZETA_CP(3,2)+bZt*ZETA_CP(3,3)
                f1=RV2_CP(1,1)*FcXt+RV2_CP(1,2)*FcYt+RV2_CP(1,3)*FcZt
                f2=RV2_CP(2,1)*FcXt+RV2_CP(2,2)*FcYt+RV2_CP(2,3)*FcZt
                f3=RV2_CP(3,1)*FcXt+RV2_CP(3,2)*FcYt+RV2_CP(3,3)*FcZt
             elseif (QZONLY) then
                g1 = ZERO
                g2 = ZERO
                g3 = bZt * ZETA_CP(3,3)
                f1 = FcXt
                f2 = FcYt
                f3 = FcZt * RV2_CP(3,3)
             else
                g1 = bXt * ZETA_CP(1,1)
                g2 = bYt * ZETA_CP(2,2)
                g3 = bZt * ZETA_CP(3,3)
                f1 = FcXt * RV2_CP(1,1)
                f2 = FcYt * RV2_CP(2,2)
                f3 = FcZt * RV2_CP(3,3)
             endif
             bVXt = VXt(I)-VXt(J) + g1
             bVYt = VYt(I)-VYt(J) + g2
             bVZt = VZt(I)-VZt(J) + g3
          else
             !adm               f1 = FcXt
             !adm               f2 = FcYt
             !adm               f3 = FcZt
             ! Haibo Yu bugfix Sept 5 2008
             f1 = FcXt * RV2_CP(1,1)
             f2 = FcYt * RV2_CP(1,1)
             f3 = FcZt * RV2_CP(1,1)
             bVXt = VXt(I)-VXt(J)
             bVYt = VYt(I)-VYt(J)
             bVZt = VZt(I)-VZt(J)
          endif

          !           "Cycle" if dsigma = 0 (to a given tolerance)
          bond2 = bXt*bXt + bYt*bYt + bZt*bZt
          bond2d = CONSTR(iconst)
          dsigma = TWO*( bXt*bVXt + bYt*bVYt + bZt*bVZt )
          IF (ABS(dsigma)*dt  <  bond2d*tolerance) cycle loop100

          !           Compute the Lagrange multiplier:
          !           1) Compute the inertia factors
          if (IMOVE(I) == 0) then
             if (mass_weighting) then
                amsi = ONE/AMASS(I)
             else
                amsi = ONE
             endif
          else
             amsi = ZERO
          endif
          if (IMOVE(J) == 0) then
             if (mass_weighting) then
                amsj = ONE/AMASS(J)
             else
                amsj = ONE
             endif
          else
             amsj = ZERO
          endif
          !           2) Compute the increment
          if (iter == 1) then
             dlambda = lambda(iconst)
          else
             frf = f1*FcXt + f2*FcYt + f3*FcZt
             dlambda = TWO*dsigma / (dt*(amsi+amsj)*frf)
             lambda(iconst) = lambda(iconst) + dlambda
             if (QPCONFULL) then
                dvir(1,1) = dlambda* FcXt*bXt
                dvir(1,2) = dlambda*(FcXt*bYt+FcYt*bXt)*HALF
                dvir(1,3) = dlambda*(FcXt*bZt+FcZt*bXt)*HALF
                dvir(2,1) = dlambda*(FcYt*bXt+FcXt*bYt)*HALF
                dvir(2,2) = dlambda* FcYt*bYt
                dvir(2,3) = dlambda*(FcYt*bZt+FcZt*bYt)*HALF
                dvir(3,1) = dlambda*(FcZt*bXt+FcXt*bZt)*HALF
                dvir(3,2) = dlambda*(FcZt*bYt+FcYt*bZt)*HALF
                dvir(3,3) = dlambda* FcZt*bZt
                CALL MatchXTL(dvir,XTLTYP,XTLREF)
                vir(1) = vir(1) + dvir(1,1)
                vir(2) = vir(2) + dvir(1,2)
                vir(3) = vir(3) + dvir(1,3)
                vir(4) = vir(4) + dvir(2,1)
                vir(5) = vir(5) + dvir(2,2)
                vir(6) = vir(6) + dvir(2,3)
                vir(7) = vir(7) + dvir(3,1)
                vir(8) = vir(8) + dvir(3,2)
                vir(9) = vir(9) + dvir(3,3)
             elseif (QZONLY) then
                vir(1) = vir(1) + dlambda*(FcZt*bZt) ! ??? guess...
             else
                vir(1) = vir(1) &
                     + dlambda*(FcXt*bXt+FcYt*bYt+FcZt*bZt)
             endif
          endif

          !           Compute new velocities

          fact = HALF*dt*amsi*dlambda
          VXt(I) = VXt(I) + fact*f1
          VYt(I) = VYt(I) + fact*f2
          VZt(I) = VZt(I) + fact*f3
          fact = HALF*dt*amsj*dlambda
          VXt(J) = VXt(J) - fact*f1
          VYt(J) = VYt(J) - fact*f2
          VZt(J) = VZt(J) - fact*f3

          !           Set ISKP flags to moving=true
          ISKPR(I) = 1.0
          ISKPR(J) = 1.0


          if (QPCON) then
             bVXt = VXt(I)-VXt(J) + g1
             bVYt = VYt(I)-VYt(J) + g2
             bVZt = VZt(I)-VZt(J) + g3
          else
             bVXt = VXt(I)-VXt(J)
             bVYt = VYt(I)-VYt(J)
             bVZt = VZt(I)-VZt(J)
          endif
          dsigma = TWO*( bXt*bVXt + bYt*bVYt + bZt*bVZt )


       ENDDO loop100
       !        End of loop over constraints
       !
#if KEY_PARALLEL==1
       CALL RI1VDGBR(ISKPR)
#endif
#if KEY_PARALLEL==1
       CALL GCOMB(VIR,9)
#endif
       !
       if (QPCON) then
          fact = HALF*dt/W_CP
          if (QPCONFULL) then
             ZETA_CP(1,1) = ZETA_CP(1,1) + fact*vir(1)
             ZETA_CP(1,2) = ZETA_CP(1,2) + fact*vir(2)
             ZETA_CP(1,3) = ZETA_CP(1,3) + fact*vir(3)
             ZETA_CP(2,1) = ZETA_CP(2,1) + fact*vir(4)
             ZETA_CP(2,2) = ZETA_CP(2,2) + fact*vir(5)
             ZETA_CP(2,3) = ZETA_CP(2,3) + fact*vir(6)
             ZETA_CP(3,1) = ZETA_CP(3,1) + fact*vir(7)
             ZETA_CP(3,2) = ZETA_CP(3,2) + fact*vir(8)
             ZETA_CP(3,3) = ZETA_CP(3,3) + fact*vir(9)
          elseif (QZONLY) then
             ZETA_CP(3,3) = ZETA_CP(3,3) + fact*vir(1)
          else
             ZETA_CP(1,1) = ZETA_CP(1,1) + fact*vir(1)
             ZETA_CP(2,2) = ZETA_CP(2,2) + fact*vir(1)
             ZETA_CP(3,3) = ZETA_CP(3,3) + fact*vir(1)
          endif
       endif

       !        Get ISKPR flags ready for next iteration
       !        Exit if all constraints are "done"
       done = .true.
#if KEY_PARALLEL==1
       kstart=1
       kstop = iparpt(numnod)
#else /**/
       kstart = atfrst
       kstop = atlast
#endif
       !
       !     In parallel we have to check all constraints to
       !     keep the number of above gcomb calls the same for everybody
       !
       do K = kstart,kstop

          if (ISKPR(K) >= -0.1) ISKPR(K) = ISKPR(K) - 1.0
          done = done .and. (ISKPR(K) <= -0.9)
       enddo
       if (done) then
          if (PRNLEV > 5) write(OUTU,*) &
               'RATTLE converged in',ITER,' iterations'
          exit
       endif

    enddo

    !     Signal if RATTLE has not converged
    if (iter >= MXITER) then
311    FORMAT(' ***** ERROR IN RattleRoll ***** VELOCITY RESETTING', &
            ' WAS NOT ACCOMPLISHED IN',I6,' ITERATIONS',/)
       if (WRNLEV >= 2) WRITE(OUTU,311) MXITER
       CALL DIEWRN(-2)
       return
    endif


    RETURN
  END subroutine RattleRoll

  SUBROUTINE MatchXTL(p,XTLTYP,XTLREF)
    !
    !     Transform the "p" matrix to match the symmetry of XTLTYP
    !
    use stream ! for OUTU

    !     Global variables
    use number
    use machutil,only:eclock,die

    !     Arguments
    real(chm_real)      p(3,3)
    character(len=4)    XTLTYP
    real(chm_real)      XTLREF(6)

    !     Local variables
    real(chm_real) tr,px,py,pz,pxy,pxz


    IF (XTLTYP == 'CUBI' .OR. XTLTYP.EQ.'RECT' &
         .OR. XTLTYP == 'OCTA' .OR. XTLTYP.EQ.'RHDO') THEN ! XDIM=1
       tr = (p(1,1)+p(2,2)+p(3,3))*THIRD
       CALL IDENT33(p)
       p(1,1) = tr
       p(2,2) = tr
       p(3,3) = tr
    ELSEIF (XTLTYP == 'TETR' .OR. XTLTYP.EQ.'HEXA') THEN ! XDIM=2
       ! warning message removed after simulation tests
       ! may 2023 venabler@nhlbi.nih.gov
       pxy = (p(1,1)+p(2,2))*HALF
       pz  =  p(3,3)
       CALL IDENT33(p)
       p(1,1) = pxy
       p(2,2) = pxy
       p(3,3) = pz
    ELSEIF (XTLTYP == 'ORTH') THEN ! XDIM=3
       px = p(1,1)
       py = p(2,2)
       pz = p(3,3)
       CALL IDENT33(p)
       p(1,1) = px
       p(2,2) = py
       p(3,3) = pz
    ELSEIF (XTLTYP == 'MONO') THEN ! XDIM=4
       px  =  p(1,1)
       py  =  p(2,2)
       pz  =  p(3,3)
       pxz = (p(1,3)+p(3,1))*HALF
       CALL IDENT33(p)
       p(1,1) = px
       p(2,2) = py
       p(3,3) = pz
       p(1,3) = pxz
       p(3,1) = pxz
    ELSEIF (XTLTYP == 'RHOM') THEN ! XDIM=2
       write(OUTU,'(a)') &
            'WARNING -- RHOM crystal type is untested !'
       write(OUTU,'(a)') &
            '  Dear user, please do the following:'
       write(OUTU,'(a)') &
            '  1. Produce a benchmark with DYNA CPT LEAP'
       write(OUTU,'(a)') &
            '  2. Comment out this message from the code'
       write(OUTU,'(a)')'     and try DYNA VV2 again'
       write(OUTU,'(a)') &
            '  3. Make sure it reproduces the average lattice'
       write(OUTU,'(a)') &
            '     parameters (A,B,C,ALPHA,BETA,GAMMA) and their'
       write(OUTU,'(a)')'     amplitudes of fluctuation'
       write(OUTU,'(a)') &
            '  4. If everything is OK, remove this warning'
       write(OUTU,'(a)')'     (If not, fix the code...)'
    ! disable untested code
       CALL DIE
       tr = (p(1,1)+p(2,2)+p(3,3))*THIRD
       px = (p(1,2)+p(2,1)+p(1,3)+p(3,1)+p(2,3)+p(3,2))*SIXTH
       CALL IDENT33(p)
       p(1,1) = tr
       p(2,2) = tr
       p(3,3) = tr
       p(1,2) = px
       p(2,1) = px
       p(1,3) = px
       p(3,1) = px
       p(2,3) = px
       p(3,2) = px
    ELSEIF (XTLTYP == 'TRIC') THEN ! XDIM=6
       ! nothing to do...
    ELSE
       CALL WRNDIE(-5,'<VV2>','Unknown crystal type.')
    ENDIF

    RETURN
  END SUBROUTINE MatchXTL


  SUBROUTINE RelativeTstat( &
       switch,compute, QNOSE,NOBL,INLCKP, &
       QRELT,RELT_VX,RELT_VY,RELT_VZ,RELT_M, &
       IABST,ABST_VX,ABST_VY,ABST_VZ,ABST_M, &
       ATFRST,ATLAST,IMOVE_,AMASS_, VX,VY,VZ, &
       ISDRUDE )
    !
    !     Convert velocities back and forth for relative thermostatting
    !

    !     Global variables
    use number ! for ZERO
    use parallel ! for JPBLOCK, MYNOD

    !     Arguments
    logical switch            ! INPUT   .TRUE.=on, .FALSE.=off
    logical compute           ! INPUT

    logical QNOSE             ! INPUT   Nose-Hoover (NH) flag
    integer NOBL              ! INPUT   Number of NH thermostats
    integer,dimension(:) :: INLCKP  ! INPUT
    logical QRELT(NOBL)       ! INPUT   Relative thermostatting flags
    real(chm_real)  RELT_VX(NOBL)     ! I/O
    real(chm_real)  RELT_VY(NOBL)     ! I/O
    real(chm_real)  RELT_VZ(NOBL)     ! I/O
    real(chm_real)  RELT_M(NOBL)      ! I/O
    integer IABST             ! INPUT   Absolute thermostatting
    real(chm_real)  ABST_VX           ! I/O
    real(chm_real)  ABST_VY           ! I/O
    real(chm_real)  ABST_VZ           ! I/O
    real(chm_real)  ABST_M            ! I/O

    integer ATFRST, ATLAST    ! INPUT
    integer IMOVE_(*)         ! INPUT
    real(chm_real)  AMASS_(*)         ! INPUT
    real(chm_real)  VX(*),VY(*),VZ(*) ! I/O     Atomic velocities

    logical ISDRUDE(*)        ! INPUT

    !     Local variables
    integer ti,I

    IF (.NOT.QNOSE) RETURN

    !     Compute center of mass velocities for relative thermostatting
    IF (compute) THEN
       DO ti = 1,NOBL
          IF (QRELT(ti)) THEN
             RELT_VX(ti) = ZERO
             RELT_VY(ti) = ZERO
             RELT_VZ(ti) = ZERO
          ENDIF
       ENDDO
       ABST_VX = ZERO
       ABST_VY = ZERO
       ABST_VZ = ZERO
       DO I = ATFRST,ATLAST
#if KEY_PARASCAL==1
          IF (JPBLOCK(I) /= MYNOD) GOTO 727
#endif
          IF (IMOVE_(I) /= 0) GOTO 727
          IF (ISDRUDE(I)) THEN
             ti = INLCKP(I-1)
          ELSE
             ti = INLCKP(I)
          ENDIF
          IF (QRELT(ti)) THEN
             RELT_VX(ti) = RELT_VX(ti) + AMASS_(I)*VX(I)
             RELT_VY(ti) = RELT_VY(ti) + AMASS_(I)*VY(I)
             RELT_VZ(ti) = RELT_VZ(ti) + AMASS_(I)*VZ(I)
          ENDIF
727       CONTINUE
       ENDDO
#if KEY_PARALLEL==1
       CALL GCOMB(RELT_VX,NOBL)
#endif
#if KEY_PARALLEL==1
       CALL GCOMB(RELT_VY,NOBL)
#endif
#if KEY_PARALLEL==1
       CALL GCOMB(RELT_VZ,NOBL)
#endif
       DO ti = 1,NOBL
          IF (QRELT(ti)) THEN
             RELT_VX(ti) = RELT_VX(ti) / RELT_M(ti)
             RELT_VY(ti) = RELT_VY(ti) / RELT_M(ti)
             RELT_VZ(ti) = RELT_VZ(ti) / RELT_M(ti)
             IF (IABST > 0) THEN
                ABST_VX = ABST_VX + RELT_M(ti)*RELT_VX(ti)
                ABST_VY = ABST_VY + RELT_M(ti)*RELT_VY(ti)
                ABST_VZ = ABST_VZ + RELT_M(ti)*RELT_VZ(ti)
             ENDIF
          ENDIF
       ENDDO
       ABST_VX = ABST_VX / ABST_M
       ABST_VY = ABST_VY / ABST_M
       ABST_VZ = ABST_VZ / ABST_M
    ENDIF

    DO I = ATFRST,ATLAST
#if KEY_PARASCAL==1
       IF (JPBLOCK(I) /= MYNOD) GOTO 728
#endif
       IF (IMOVE_(I) /= 0) GOTO 728
       IF (ISDRUDE(I)) THEN
!!! We remove the drift associated with
!!! the thermostat of the core particle
          ti = INLCKP(I-1)
       ELSE
!!! We remove the drift associated with
!!! the thermostat of the particle itself
          ti = INLCKP(I)
       ENDIF
       IF (QRELT(ti)) THEN
          IF (switch) THEN
             VX(I) = VX(I) - ABST_VX
             VY(I) = VY(I) - ABST_VY
             VZ(I) = VZ(I) - ABST_VZ
          ELSE
             VX(I) = VX(I) + ABST_VX
             VY(I) = VY(I) + ABST_VY
             VZ(I) = VZ(I) + ABST_VZ
          ENDIF
       ENDIF
728    CONTINUE
    ENDDO
    IF (.NOT.switch) THEN
       ABST_VX = ZERO
       ABST_VY = ZERO
       ABST_VZ = ZERO
    ENDIF

    RETURN
  END SUBROUTINE RelativeTstat


  SUBROUTINE CMVelocity( &
       ATFRST,ATLAST,IMOVE,AMASS,VX,VY,VZ, &
       CMVX,CMVY,CMVZ,CMM )
    !
    !     Compute center-of-mass velocity
    !

    !     Global variables
    use number ! for ZERO
    use parallel ! for JPBLOCK, MYNOD

    !     Arguments
    integer ATFRST, ATLAST    ! INPUT
    integer IMOVE(*)          ! INPUT
    real(chm_real)  AMASS(*)          ! INPUT
    real(chm_real)  VX(*),VY(*),VZ(*) ! INPUT
    real(chm_real)  CMVX,CMVY,CMVZ    ! OUTPUT  Velocity
    real(chm_real)  CMM               ! OUTPUT  Mass
#if KEY_PARALLEL==1
    real(chm_real)  GCARR(4)
#endif
    !     Variables
    integer I

    CMVX = ZERO
    CMVY = ZERO
    CMVZ = ZERO
    CMM = ZERO
    DO I = ATFRST,ATLAST
       IF (IMOVE(I) /= 0) GOTO 729
       CMVX = CMVX + AMASS(I)*VX(I)
       CMVY = CMVY + AMASS(I)*VY(I)
       CMVZ = CMVZ + AMASS(I)*VZ(I)
       CMM = CMM + AMASS(I)
729    CONTINUE
    ENDDO
#if KEY_PARALLEL==1
    GCARR(1)=CMVX
    GCARR(2)=CMVY
    GCARR(3)=CMVZ
    GCARR(4)=CMM
    CALL GCOMB(GCARR,4)
    CMVX=GCARR(1)
    CMVY=GCARR(2)
    CMVZ=GCARR(3)
    CMM=GCARR(4)
#endif
    CMVX = CMVX / CMM
    CMVY = CMVY / CMM
    CMVZ = CMVZ / CMM

    RETURN
  END SUBROUTINE CMVelocity


  SUBROUTINE CorrectVirial( &
       mode, vir, X,Y,Z, DX,DY,DZ, DXc,DYc,DZc, &
       inFirstAtom,inLastAtom, inMass,inIsDrude,inMovingCode &
       )
    !
    !     Correct/Uncorrect the virial for when the center-of-mass positions depend
    !     on the volume, but the relative positions don't.
    !
    use number

    !     Global variables
    use parallel ! for JPBLOCK, MYNOD

    !     Arguments
    integer mode              ! Input: 1 to correct, 0 to uncorrect
    real(chm_real)  vir(9)            ! Input/Output: Internal virial
    real(chm_real)  X(*),Y(*),Z(*)    ! Input: Positions (r_i)
    real(chm_real)  DX(*),DY(*),DZ(*) ! Input: Negative forces (-F_i)
    real(chm_real)  DXc(*),DYc(*),DZc(*) ! Input: Negative constraint forces (-Fc_i)
    integer inFirstAtom       ! Input: Index of the first atom
    integer inLastAtom        ! Input: Index of the last atom
    real(chm_real)  inMass(*)         ! Input: Masses (m_i)
    logical inIsDrude(*)      ! Input:
    integer inMovingCode(*)   ! Input: 0 to include in correction

    !     Local variables
    integer i,iv
    real(chm_real)  mi,m1,m2, Rx,Ry,Rz, ddcx,ddcy,ddcz
    real(chm_real)  old_vir(9), virtmp(9)
    save    old_vir


    if (mode == 1) then
       !        Keep uncorrected virial
       do iv = 1,9
          old_vir(iv) = vir(iv)
          virtmp(iv) = zero
       enddo
       !        Correct virial
       do i = inFirstAtom,inLastAtom
#if KEY_PARASCAL==1
          if (JPBLOCK(i) /= MYNOD) goto 730
#endif
          if (inMovingCode(i) /= 0) goto 730

          if (inIsDrude(i)) then
             mi = 1.0 / (inMass(i-1) + inMass(i))
             m1 = inMass(i-1) * mi
             m2 = inMass(i) * mi
             Rx = m1 * X(i-1) + m2 * X(i)
             Ry = m1 * Y(i-1) + m2 * Y(i)
             Rz = m1 * Z(i-1) + m2 * Z(i)

             !              corrected virial = virial - (d-R)Fd
             ddcx = m1*DX(i) - m2*DX(i-1)
             ddcy = m1*DY(i) - m2*DY(i-1)
             ddcz = m1*DZ(i) - m2*DZ(i-1)
             virtmp(1) = virtmp(1) + ((X(i)-X(i-1))-Rx)*ddcx
             virtmp(2) = virtmp(2) + ((X(i)-X(i-1))-Rx)*ddcy
             virtmp(3) = virtmp(3) + ((X(i)-X(i-1))-Rx)*ddcz
             virtmp(4) = virtmp(4) + ((Y(i)-Y(i-1))-Ry)*ddcx
             virtmp(5) = virtmp(5) + ((Y(i)-Y(i-1))-Ry)*ddcy
             virtmp(6) = virtmp(6) + ((Y(i)-Y(i-1))-Ry)*ddcz
             virtmp(7) = virtmp(7) + ((Z(i)-Z(i-1))-Rz)*ddcx
             virtmp(8) = virtmp(8) + ((Z(i)-Z(i-1))-Rz)*ddcy
             virtmp(9) = virtmp(9) + ((Z(i)-Z(i-1))-Rz)*ddcz
          endif
730       CONTINUE
       enddo

#if KEY_PARALLEL==1
       CALL GCOMB(virtmp,9)
#endif
       do iv = 1, 9
          vir(iv) = vir(iv) + virtmp(iv)
       enddo

    elseif (mode == 0) then
       !        Get back uncorrected virial
       do iv = 1,9
          vir(iv) = old_vir(iv)
       enddo
    endif
    !
    RETURN
  END SUBROUTINE CorrectVirial


!  subroutine allocate_dynamvv2()
!    use memory
!    character(len=*),parameter :: routine_name="allocate_dynamvv2"
!    call chmalloc(file_name,routine_name,'dxsave ',maxaim,crl=dxsave)
!    call chmalloc(file_name,routine_name,'dysave ',maxaim,crl=dysave)
!    call chmalloc(file_name,routine_name,'dzsave ',maxaim,crl=dzsave)

!    return
!  end subroutine allocate_dynamvv2

SUBROUTINE HardWallDrude(X,Y,Z,VX,VY,VZ,   &
  inFirstAtom,inLastAtom,inMass,inIsDrude,inMovingCode,vir,DELTA_)
  !
  !  To impose hard wall constraint on drude bond length
  !  When the length of drude bond is longer than L_WALL,
  !  the length will be constrained at L_WALL and the velocities
  !  along the bond vector are flipped and scaled down according to
  !  drude temperature
  !
  use number
  use parallel
  use stream,only:OUTU
  use machutil,only:die
  use psf,only:L_WALL,QHARDWALL
  use consta, only:KBOLTZ
  implicit none

  !     Arguments
  real(chm_real)  vir(9)            ! Input/Output: Internal virial
  real(chm_real)  X(*),Y(*),Z(*)    ! Input: Positions (r_i)
  real(chm_real)  VX(*),VY(*),VZ(*) ! Input: velocities
  integer inFirstAtom       ! Input: Index of the first atom
  integer inLastAtom        ! Input: Index of the last atom
  real(chm_real) inMass(*),DELTA_   ! Input: Masses (m_i)
  logical inIsDrude(*)      ! Input: is a drude or not
  integer inMovingCode(*)   ! Input: 0 to include in correction

  !     Local variables
  integer i, ia, ib
  real(chm_real) x_ab, y_ab, z_ab, rab_SQ, rab, mass_a, mass_b, mass_sum
  real(chm_real) vb_x_1, vb_y_1, vb_z_1, vp_x_1, vp_y_1, vp_z_1, dot_v_r_1
  real(chm_real) vb_x_2, vb_y_2, vb_z_2, vp_x_2, vp_y_2, vp_z_2, dot_v_r_2
  real(chm_real) vb_cm, v_bond, dr, dr_a, dr_b, L_WALL_SQ
  real(chm_real) new_pos_a_x, new_pos_a_y, new_pos_a_z
  real(chm_real) new_pos_b_x, new_pos_b_y, new_pos_b_z
  real(chm_real) new_vel_a_x, new_vel_a_y, new_vel_a_z
  real(chm_real) new_vel_b_x, new_vel_b_y, new_vel_b_z
  real(chm_real) kbt,virtmp(9),invdt, df_x, df_y, df_z
  real(chm_real) delta_T

  IF (.not. QHARDWALL) return

  kbt = T_Drude * KBOLTZ
  L_WALL_SQ = L_WALL * L_WALL

  invdt = 1.0 / (DELTA_ * 0.5)
  DO i = 1,9
    virtmp(i) = 0.0
  ENDDO

  DO i = inFirstAtom+1,inLastAtom
    IF (inIsDrude(I)) THEN
      ia = I - 1
      ib = I

      x_ab = x(ib) - x(ia)
      y_ab = y(ib) - y(ia)
      z_ab = z(ib) - z(ia)
      rab_SQ = x_ab*x_ab + y_ab*y_ab + z_ab*z_ab

      IF ( rab_SQ .gt. L_WALL_SQ ) THEN
        rab = dsqrt(rab_SQ)
        IF ( rab .gt. (2.0*L_WALL) ) THEN
600       FORMAT(' ** ERROR IN HardWallDrude **', &
               ' The drude is too far away from atom ', &
               I5,'   d = ',F8.4)
          WRITE(OUTU,600) ia,rab
          CALL DIE
        ENDIF

        x_ab = x_ab / rab
        y_ab = y_ab / rab
        z_ab = z_ab / rab

        IF ( inMovingCode(ia) /= 0 ) THEN
          IF ( inMovingCode(ib) /= 0 ) THEN
            CYCLE
          ELSE
            dot_v_r_2 = vx(ib)*x_ab + vy(ib)*y_ab + vz(ib)*z_ab
            vb_x_2 = dot_v_r_2 * x_ab
            vb_y_2 = dot_v_r_2 * y_ab
            vb_z_2 = dot_v_r_2 * z_ab

            vp_x_2 = vx(ib) - vb_x_2
            vp_y_2 = vy(ib) - vb_y_2
            vp_z_2 = vz(ib) - vb_z_2

            dr = rab - L_WALL
            IF (dot_v_r_2 .eq. 0.0) THEN
              delta_T = DELTA_
            ELSE
              delta_T = dr/dabs(dot_v_r_2)
              if(delta_T .gt. DELTA_ ) delta_T = DELTA_
            ENDIF

            dot_v_r_2 = -dot_v_r_2*dsqrt(kbt/inMass(ib))/dabs(dot_v_r_2)

            vb_x_2 = dot_v_r_2 * x_ab
            vb_y_2 = dot_v_r_2 * y_ab
            vb_z_2 = dot_v_r_2 * z_ab

            new_vel_a_x = vx(ia)
            new_vel_a_y = vy(ia)
            new_vel_a_z = vz(ia)

            new_vel_b_x = vp_x_2 + vb_x_2
            new_vel_b_y = vp_y_2 + vb_y_2
            new_vel_b_z = vp_z_2 + vb_z_2

            dr_b = -dr + delta_T*dot_v_r_2

            new_pos_a_x = x(ia)
            new_pos_a_y = y(ia)
            new_pos_a_z = z(ia)

            new_pos_b_x = x(ib) + dr_b * x_ab
            new_pos_b_y = y(ib) + dr_b * y_ab
            new_pos_b_z = z(ib) + dr_b * z_ab
          ENDIF
        ELSE
          mass_a = inMass(ia)
          mass_b = inMass(ib)
          mass_sum = mass_a+mass_b

          dot_v_r_1 = vx(ia)*x_ab + vy(ia)*y_ab + vz(ia)*z_ab
          vb_x_1 = dot_v_r_1 * x_ab
          vb_y_1 = dot_v_r_1 * y_ab
          vb_z_1 = dot_v_r_1 * z_ab

          vp_x_1 = vx(ia) - vb_x_1
          vp_y_1 = vy(ia) - vb_y_1
          vp_z_1 = vz(ia) - vb_z_1

          dot_v_r_2 = vx(ib)*x_ab + vy(ib)*y_ab + vz(ib)*z_ab
          vb_x_2 = dot_v_r_2 * x_ab
          vb_y_2 = dot_v_r_2 * y_ab
          vb_z_2 = dot_v_r_2 * z_ab

          vp_x_2 = vx(ib) - vb_x_2
          vp_y_2 = vy(ib) - vb_y_2
          vp_z_2 = vz(ib) - vb_z_2

          vb_cm = (mass_a*dot_v_r_1 + mass_b*dot_v_r_2)/mass_sum
          dot_v_r_1 = dot_v_r_1 - vb_cm
          dot_v_r_2 = dot_v_r_2 - vb_cm

          dr = rab - L_WALL

          IF (dot_v_r_2 .eq. dot_v_r_1) THEN
            delta_T = DELTA_
          ELSE
            delta_T = dr/dabs(dot_v_r_2 - dot_v_r_1)
            if(delta_T .gt. DELTA_) delta_T = DELTA_
          ENDIF

          ! the relative velocity between ia and ib
          v_bond = dsqrt(kbt/mass_b)

          dot_v_r_1 = -dot_v_r_1*v_bond*mass_b/(dabs(dot_v_r_1)*mass_sum)
          dot_v_r_2 = -dot_v_r_2*v_bond*mass_a/(dabs(dot_v_r_2)*mass_sum)

          dr_a = dr*mass_b/mass_sum + delta_T*dot_v_r_1;
          dr_b = -dr*mass_a/mass_sum + delta_T*dot_v_r_2;

          new_pos_a_x = x(ia) + dr_a * x_ab
          new_pos_a_y = y(ia) + dr_a * y_ab
          new_pos_a_z = z(ia) + dr_a * z_ab

          new_pos_b_x = x(ib) + dr_b * x_ab
          new_pos_b_y = y(ib) + dr_b * y_ab
          new_pos_b_z = z(ib) + dr_b * z_ab

          dot_v_r_1 = dot_v_r_1 + vb_cm
          dot_v_r_2 = dot_v_r_2 + vb_cm

          vb_x_1 = dot_v_r_1 * x_ab
          vb_y_1 = dot_v_r_1 * y_ab
          vb_z_1 = dot_v_r_1 * z_ab

          vb_x_2 = dot_v_r_2 * x_ab
          vb_y_2 = dot_v_r_2 * y_ab
          vb_z_2 = dot_v_r_2 * z_ab

          new_vel_a_x = vp_x_1 + vb_x_1
          new_vel_a_y = vp_y_1 + vb_y_1
          new_vel_a_z = vp_z_1 + vb_z_1

          new_vel_b_x = vp_x_2 + vb_x_2
          new_vel_b_y = vp_y_2 + vb_y_2
          new_vel_b_z = vp_z_2 + vb_z_2
        ENDIF

        df_x = (new_vel_a_x - vx(ia)) * mass_a * invdt
        df_y = (new_vel_a_y - vy(ia)) * mass_a * invdt
        df_z = (new_vel_a_z - vz(ia)) * mass_a * invdt

        virtmp(1) = virtmp(1) + x(ia)*df_x
        virtmp(2) = virtmp(2) + x(ia)*df_y
        virtmp(3) = virtmp(3) + x(ia)*df_z
        virtmp(4) = virtmp(4) + y(ia)*df_x
        virtmp(5) = virtmp(5) + y(ia)*df_y
        virtmp(6) = virtmp(6) + y(ia)*df_z
        virtmp(7) = virtmp(7) + z(ia)*df_x
        virtmp(8) = virtmp(8) + z(ia)*df_y
        virtmp(9) = virtmp(9) + z(ia)*df_z

        df_x = (new_vel_b_x - vx(ib)) * mass_b * invdt
        df_y = (new_vel_b_y - vy(ib)) * mass_b * invdt
        df_z = (new_vel_b_z - vz(ib)) * mass_b * invdt

        virtmp(1) = virtmp(1) + x(ib)*df_x
        virtmp(2) = virtmp(2) + x(ib)*df_y
        virtmp(3) = virtmp(3) + x(ib)*df_z
        virtmp(4) = virtmp(4) + y(ib)*df_x
        virtmp(5) = virtmp(5) + y(ib)*df_y
        virtmp(6) = virtmp(6) + y(ib)*df_z
        virtmp(7) = virtmp(7) + z(ib)*df_x
        virtmp(8) = virtmp(8) + z(ib)*df_y
        virtmp(9) = virtmp(9) + z(ib)*df_z

        x(ia) = new_pos_a_x
        y(ia) = new_pos_a_y
        z(ia) = new_pos_a_z

        x(ib) = new_pos_b_x
        y(ib) = new_pos_b_y
        z(ib) = new_pos_b_z

        vx(ia) = new_vel_a_x
        vy(ia) = new_vel_a_y
        vz(ia) = new_vel_a_z

        vx(ib) = new_vel_b_x
        vy(ib) = new_vel_b_y
        vz(ib) = new_vel_b_z

      ENDIF
    ENDIF
  ENDDO

#if KEY_PARALLEL==1
  CALL GCOMB(virtmp,9)
#endif
  DO i = 1, 9
    vir(i) = vir(i) + virtmp(i)
  ENDDO

  RETURN
END SUBROUTINE HardWallDrude

subroutine vv2_alloc(natom)
  use number
  use memory
  integer, intent(in) :: natom
  character(len=*), parameter :: proc_name = 'vv2_alloc'

  if (allocated(DXsave)) return
  call chmalloc(file_name,proc_name,'DXsave',natom,crl=DXsave)
  call chmalloc(file_name,proc_name,'DYsave',natom,crl=DYsave)
  call chmalloc(file_name,proc_name,'DZsave',natom,crl=DZsave)
  DXsave = ZERO
  DYsave = ZERO
  DZsave = ZERO
end subroutine vv2_alloc

subroutine vv2_dealloc()
  use memory
  integer :: natom
  character(len=*), parameter :: proc_name = 'vv2_dealloc'

  if (.not. allocated(DXsave)) return
  natom = size(DXsave)
  call chmdealloc(file_name,proc_name,'DXsave',natom,crl=DXsave)
  call chmdealloc(file_name,proc_name,'DYsave',natom,crl=DYsave)
  call chmdealloc(file_name,proc_name,'DZsave',natom,crl=DZsave)
end subroutine vv2_dealloc

#endif /* (dynvv2_main)*/

end module dynvv2
