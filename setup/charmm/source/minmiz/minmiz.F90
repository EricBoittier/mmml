module minmiz_module
  use steepd_module, only: steepd
  implicit none
contains

  SUBROUTINE MINMIZ(COMLYN,COMLEN &
       , min_opts &
       , abnr_opts &
       , sd_opts &
       )
  !-----------------------------------------------------------------------
  !     MINMIZ controls the minimization options.
  !
  !     The input arguments are:
  !
  !     COMLYN           - The command line.
  !     COMLEN           - The length of the command line.
  !
  use abnerm,only:abner

  use api_types, only: min_settings, min_abnr_settings, min_sd_settings

#if KEY_CHEQ==1
  use cheq,only:qcg,cgmodel,qcginv,qcginvf,qpbeq1,   &
#endif
#if KEY_CHEQ==1
     minnorm,qcgmine,qnoco,qpolar1,ipolar1,     &
#endif
#if KEY_CHEQ==1
     checketa,checkqnorm,qpartbin,allocate_cheq
#endif

  use chm_kinds
  use dimens_fcm
  use number, only: zero
  use bases_fcm
  use contrl
  use coord
  use euler
  use eutil
  use hbondm
  use image
  use nraph_m
  use stream
  use string
  use parallel, only: numnod
  use pert
  use pert_mod
  use reawri
  use pathm
  use powell_mod, only: powell
  use psf, only: natom, ngrp
#if KEY_TMD==1
  use tmd,only:inrt
#endif
#if KEY_DOMDEC==1
  use domdec_common, only: q_domdec
#endif
#if KEY_DHDGB==1
!AP/MF
  use dhdgb
#endif
  use omm_ctrl, only: omm_requested
  use blade_ctrl_module, only: blade_requested
#if KEY_OPENMM == 1
  use omm_main, only: omm_minimize
#endif /* KEY_OPENMM */
#if KEY_BLADE == 1
  use blade_main, only: blade_minimize
#endif /* KEY_BLADE */

  use steepd_module, only: &
#if KEY_REPLICA == 1
       steepdneb, &
#endif
       steepd

  implicit none

  ! passed variables
  character(len=*) :: comlyn
  integer :: comlen

  ! api library settings
  type(min_settings), optional :: min_opts
  type(min_abnr_settings), optional :: abnr_opts
  type(min_sd_settings), optional :: sd_opts

  logical :: want_openmm = .false.
  logical :: want_blade = .false.
  
  ! local variables
  character(len=4) MINOPT
  LOGICAL QIMCEN

#if KEY_CHEQ==1
  LOGICAL QCHEQRDPRM,QCHEQNORM
#endif

#if KEY_OPENMM==1
  integer n_omm_minsteps
  real(chm_real) tolgrd
#endif
#if KEY_BLADE==1
  integer n_blade_minsteps
  integer n_blade_mintype
  integer n_blade_status
  logical q_blade_fallback
  character(len=len(comlyn)) blade_comlyn
  integer blade_comlen
  ! real(chm_real) n_blade_tolgrad
  real(chm_real) n_blade_steplen
#endif /* KEY_BLADE */

#if KEY_TNPACK==1
  !yw...TNPACK, 28-Jul-95
  LOGICAL QSTART

  QLOC1  = .FALSE.
#endif

  MINXYZ = .TRUE.
  LMINUC = .FALSE.
  QIMCEN = LIMCEN
#if KEY_BLADE==1
  q_blade_fallback = .false.
#endif
  ! omm_requested aborts via WRNDIE if OMM is requested on a non-OpenMM build.
  want_openmm = omm_requested(COMLYN, COMLEN, 'MINI')
#if KEY_OPENMM==1
  if (want_openmm) then
     if (present(min_opts)) then
        n_omm_minsteps   = min_opts%nstep
        TOLGRD = min_opts%tolgrd
     else
        n_omm_minsteps  = GTRMI(COMLYN,COMLEN,'NSTE',0)
        TOLgrd = GTRMF(COMLYN,COMLEN,'TOLG',2.39) ! 1.0 kJ/mol/nm in kcal/mol/A
     end if
     if(prnlev > 1) write(outu,'(a,2x,i6,2x,e11.4)') &
          'Minimization requested on OpenMM: nsteps, tolgrd', &
          n_omm_minsteps,tolgrd
     call omm_minimize(x,y,z,n_omm_minsteps,tolgrd)
     return
  endif
#endif /* KEY_OPENMM */
  ! blade_requested aborts via WRNDIE if BLaDE is requested on a non-BLaDE build.
  want_blade = blade_requested(COMLYN, COMLEN, 'MINI')
#if KEY_BLADE==1
  if (want_blade) then
     if (present(min_opts)) then
        n_blade_minsteps   = min_opts%nstep
        ! n_blade_tolgrad = min_opts%tolgrd
        n_blade_steplen = min_opts%step
        if (present(abnr_opts)) then
           minopt = 'ABNR'
        elseif (present(sd_opts)) then
           minopt = 'SD  '
        else
           minopt = 'SD  '
        endif
     else
        blade_comlyn = comlyn
        blade_comlen = comlen
        n_blade_minsteps  = GTRMI(blade_comlyn,blade_comlen,'NSTE',0)
        ! n_blade_tolgrad = GTRMF(COMLYN,COMLEN,'TOLG',1.0) ! kcal/mol/A
        n_blade_steplen = GTRMF(blade_comlyn,blade_comlen,'STEP',0.1)
        minopt = NEXTA4(COMLYN,COMLEN)
     endif
     if (minopt == 'LBFG') then
        n_blade_mintype=0
     elseif (minopt == 'SD  ') then
        n_blade_mintype=1
     elseif (minopt == 'SDFD') then
        n_blade_mintype=2
     elseif (minopt == 'SDMD') then
        n_blade_mintype=3
     elseif (minopt == 'ABNR') then
        call wrndie(-3,'<minmiz>','BLaDE cannot use ABNR minimization as requested. Falling back on steepest descent')
        n_blade_mintype=1
     else
        n_blade_mintype=1
     endif
     if(prnlev > 1) write(outu,'(a,2x,i6,2x,i6,2x,e11.4)') &
          'Minimization requested on BLaDE: nsteps, mintype, step', &
          n_blade_minsteps,n_blade_mintype,n_blade_steplen
     call blade_minimize(x,y,z,n_blade_minsteps,n_blade_mintype, &
          n_blade_steplen,n_blade_status)
     if (n_blade_status == 0) return
     call wrndie(1,'<MINMIZ>', &
          'BLaDE SDMD could not find an acceptable trial step. Falling back on CPU steepest descent')
     q_blade_fallback = .true.
  endif
#endif /* KEY_BLADE */
#if KEY_CHEQ==1
  QCGMIN=(INDXA(COMLYN,COMLEN,'CHEQ') > 0)
  IF (.not.QCG .and. QCGMIN) CALL WRNDIE(-3,'<MINMIZ>', &
       'Fluctuating charges not set-up, use CHEQ ON command first!')
  IF (QCG.AND.QCGMIN)  THEN
     if(.not.allocated(qpartbin)) then
        call wrndie(-1,'<minmiz>','CHEQ not set-up')
     elseif(natim>natom) then
        call allocate_cheq(natim,ngrp)
     endif
     CALL CHECKETA(QCHEQRDPRM)
     IF (.not.QCHEQRDPRM) THEN
        if(prnlev > 1)write(outu,'(2a)') &
             'CHEQ DYNAMICS HAS BEEN REQUESTED BUT', &
             ' CORRECT PARAMETERS HAVE NOT BEEN READ'
        CALL WRNDIE(-1,'<MINMIZ>', &
             'CHEQ PARAMETERS HAVE NOT BEEN READ')
     ENDIF
     CALL CHECKQNORM(QCHEQNORM)
     IF (.not.QCHEQNORM) THEN
        if(prnlev > 1)write(outu,'(2a)') &
             'CHEQ MINIMIZATION HAS BEEN REQUESTED BUT', &
             ' CORRECT NORMALIZATION SPECIFICATIONS LACKING'
        CALL WRNDIE(-1,'<MINMIZ>', &
             'CHEQ NORMALIZATION ASSIGNMENT INCORRECT')
     ENDIF
     !  ----   SINCE USER WANTS CHEQ, MAKE SURE NORMALIZATION IS SET UP AND CORRECT

     CGMODEL=GTRMI(COMLYN,COMLEN,'CGMD',CGMODEL)
     QCGINV = INDXA(COMLYN, COMLEN, 'CGIN')  >  0
     QCGINVF = INDXA(COMLYN, COMLEN, 'CGFC')  >  0
     QPBEQ1 = INDXA(COMLYN, COMLEN, 'PBEQ')  >  0

     !   FOR CHARGE NORMALIZATION
     MINNORM=.TRUE.   ! NORMALIZE w/out using different masses for minimization
     !
     IF (PRNLEV > 0) THEN
        IF (QCGMIN) WRITE(OUTU,'(a)')"CHEQ HAS BEEN READ"
        IF (QPBEQ1) WRITE(OUTU,'(a)')"PBEQ is requested"
        IF (QCGINV) WRITE(OUTU,'(a)')"<MINI>: CG INVERSION REQUESTED"
        IF (QCGINVF) WRITE(OUTU,'(a)')"<MINI>: CG FORCE REQUESTED"
     ENDIF
     QCGMINE=QCGMIN
     MINXYZ=INDXA(COMLYN, COMLEN, 'NOCO') <= 0
     QNOCO=.NOT.MINXYZ
     QPOLAR1=INDXA(COMLYN, COMLEN, 'QPOL')  >  0
     IPOLAR1=GTRMI(COMLYN,COMLEN,'IPOL',IPOLAR1)
  ENDIF
#endif

  ! pick the minimization algorithm
#if KEY_BLADE==1
  if (q_blade_fallback) then
     minopt = 'SD  '
  else if (present(abnr_opts)) then
#else
  if (present(abnr_opts)) then
#endif
     minopt = 'ABNR'
  else if (present(sd_opts)) then
     minopt = 'SD  '
  else
     MINOPT = NEXTA4(COMLYN,COMLEN)
  end if

#if KEY_DOMDEC==1
  if (q_domdec .and. .not. ( minopt == 'ABNR' .or. minopt == 'SD  ' ) ) &
       call wrndie(-1, '<MINMIZ>', &
           'Cannot minimize after enabling DOMDEC.')

  if (q_domdec .and. (numnod .gt. 1)) &
       call wrndie(-1, '<MINMIZ>', &
           'Cannot minimize with more than one MPI process after enabling DOMDEC.')
#endif

  if (present(min_opts)) then
     inbfrq = min_opts%inbfrq
     ihbfrq = min_opts%ihbfrq
  end if

  ! process update commands
  CALL UPDATE(COMLYN,COMLEN,X,Y,Z,WMAIN,.TRUE., &
       .TRUE.,.TRUE.,.TRUE.,.TRUE., &
       0,[zero],[zero],[zero],[zero],[zero],[zero])
  CALL FINCYC(NUPFRQ,0,0,0,0,INBFRQ,IHBFRQ,0,IMGFRQ,0,0,0 &
#if KEY_TMD==1
       ,inrt &
#endif
       )
  !
  ! turn off image centering during minimization (but restore it later)
  LIMCEN = .FALSE.
  !
  !-----------------------------------------------------------------------
  ! Parse general minimization options.
  !
  if (present(min_opts)) then
     !     Parse the saddle code option (i.e. minimize the gradient**2)
     QSADLE = min_opts%gradient > 0
     IF(QSADLE .AND. PRNLEV >= 2) WRITE(OUTU,255)

     !     Parse the numerical derivatives option.
     QNUMER = min_opts%numerical > 0
     IF(QNUMER .AND. PRNLEV >= 2) WRITE(OUTU,256)

     ! Parse trajectory options
     IUNCRD = min_opts%iuncrd
     NSAVC  = min_opts%nsavc
     IUNXYZ = min_opts%iunxyz
     NSAVX  = min_opts%nsavx
     MXYZ   = min_opts%mxyz
  else
     !     Parse the saddle code option (i.e. minimize the gradient**2)
     QSADLE=(INDXA(COMLYN,COMLEN,'GRAD') > 0)
     IF(QSADLE .AND. PRNLEV >= 2) WRITE(OUTU,255)

     !     Parse the numerical derivatives option.
     QNUMER = INDXA(COMLYN,COMLEN,'NUME')  >  0
     IF(QNUMER .AND. PRNLEV >= 2) WRITE(OUTU,256)

     ! Parse trajectory options
     IUNCRD = GTRMI(COMLYN,COMLEN,'IUNC',-1)
     NSAVC  = GTRMI(COMLYN,COMLEN,'NSAVC',1)
     IUNXYZ = GTRMI(COMLYN,COMLEN,'IUNX',-1)
     NSAVX  = GTRMI(COMLYN,COMLEN,'NSAVX',1)
     MXYZ   = GTRMI(COMLYN,COMLEN,'MXYZ',1)

     iunwri = gtrmi(comlyn, comlen, 'IUNW', -1)  ! biovia addition
  end if

255 FORMAT(' CHARMM> Energy will be the mean squared gradient', &
         ' during minimizations.')
256 FORMAT(' CHARMM> Forces will be determined by finite differences', &
         ' during minimizations.')

  ! SAPATEL
#if KEY_CHEQ==1
  !      IF ( .not. (MINOPT  ==  'CGSD' .OR. MINOPT .EQ. 'CONJ'
  !     $    .or. MINOPT  ==  'SD  ') .and. QCGMIN )
  !     $   CALL WRNDIE(-3,'<MINMIZ>',
  !     $   'CHEQ only supported for CONJ and SD Minimizers')
  IF ( QSADLE .AND. QCG ) CALL WRNDIE(-3,'<MINMIZ>', &
       'SADLE option not currently supported with CHEQ.')
#endif
  ! SAPATEL
  !-----------------------------------------------------------------------
#if KEY_PERT==1
  IF(QPERT) CALL PERTDF
#endif
  !
  !     Branch on the minimization option.
  !
  IF (MINOPT  ==  'ABNR') THEN
     if (present(abnr_opts)) then
        call abner(comlyn, comlen, min_opts, abnr_opts)
     else
        call abner(comlyn, comlen)
     end if
  ELSE IF (MINOPT  ==  'TN  ') THEN
     !yw...TNPACK: updated 28-Jul-95 and 12-Aug-95
#if KEY_TNPACK==1
     QSTART=.TRUE.
     CALL TNDRIV(COMLYN,COMLEN,QSTART)
#else /**/
     CALL WRNDIE(-3,'<MINMIZ>','TN minmizer NOT compiled.')
#endif
     !
  ELSE IF (MINOPT  ==  'CGSD' .OR. MINOPT .EQ. 'CONJ') THEN
     CALL CONJUG(COMLYN,COMLEN)
     !
  ELSE IF (MINOPT  ==  'NRAP') THEN
     !CC      IF (NTRANS  >  0) CALL XNBLST
     CALL NRAPH(COMLYN,COMLEN,X,Y,Z)
     !
  ELSE IF (MINOPT  ==  'POWE') THEN
     ! GAMESS has it too
!--mfc-- ##IF GAMESS
!--mfc--      CALL POWELLC(COMLYN,COMLEN,X,Y,Z)
!--mfc-- ##ELSE
     CALL POWELL(COMLYN,COMLEN,X,Y,Z)
!--mfc-- ##ENDIF
     !
  ELSE IF (MINOPT  ==  'SD  ') THEN
#if KEY_RPATH==1 /*rpath*/
     IF (QPNEB) THEN
        CALL STEEPDNEB(COMLYN,COMLEN)
        !         write(*,*)'IM IN STEEPDNEB'
     ELSE
#endif /*    (rpath) */

        if (present(sd_opts)) then
           CALL STEEPD(COMLYN,COMLEN, min_opts, sd_opts)
        else
           CALL STEEPD(COMLYN,COMLEN)
        end if

#if KEY_RPATH==1
     ENDIF
#endif
     !
  ELSE
     CALL WRNDIE(-3,'<MINMIZ>', &
          'Unrecognised minimization procedure.')
  ENDIF
  !
  MINXYZ = .TRUE.
  LMINUC = .FALSE.
  LIMCEN = QIMCEN
  !
#if KEY_PERT==1
  IF(QPERT) CALL PERTAN(.FALSE.)
#endif
  ! SAPATEL
#if KEY_CHEQ==1
  QCGINV=.FALSE.
  QCGINVF=.FALSE.
  QCGMINE=.FALSE.
  QPBEQ1 =.FALSE.
  QNOCO =.FALSE.
  CGMODEL=0
  QPOLAR1=.FALSE.
  IPOLAR1=-1
  MINNORM=.FALSE.
#endif
  ! SAPATEL
  !
  RETURN
END SUBROUTINE MINMIZ

SUBROUTINE MINTRJ(NCALLS,NSTEPS,STEP)
  !-----------------------------------------------------------------------
  !     Writes trajectory frame of minimization steps
  !
#if KEY_CHEQ==1
  use cheq,only:qcg
#endif

  use chm_kinds
  use dimens_fcm
  use reawri
  use psf
  use coord
  use cvio
  use deriv
  use dynio
  use number
  use ctitla
  !
  implicit none
  !
  INTEGER NCALLS,NSTEPS
  real(chm_real) STEP,T(1)
  !
  INTEGER NAT3
#if KEY_MIDSINR == 1
    ! dummy variables and memory allocations
    logical :: q_midsinr=.false.
    integer :: L_val=1
    real(chm_real),pointer :: v_1ij(:,:,:)=>Null(), &
                              v_2ij(:,:,:)=>Null()
#endif
  !
  T(1)=ZERO
  IF((NSAVC > 0).AND.(IUNCRD >= 0) &
       .AND.(MOD(NCALLS,NSAVC) == 0)) THEN
     NAT3=3*NATOM
     CALL WRITCV(X,Y,Z, &
#if KEY_CHEQ==1
          CG,QCG,                                &
#endif
          NATOM, (/ 0 /), NATOM,NCALLS,NCALLS,NAT3,STEP,NSAVC, &
          NSTEPS,TITLEA,NTITLA,IUNCRD,.FALSE.,.FALSE., (/ 0 /), .FALSE., (/ ZERO /))
  ENDIF

!=====================================================================
! BIOVIA CODE START
!=====================================================================
  IF((NSAVX > 0).AND.(IUNXYZ >= 0).AND.(MOD(NCALLS,NSAVX) == 0)) &
       CALL WRXYZ(IUNXYZ,X,Y,Z,T,T,T,DX,DY,DZ,NCALLS)
  !
  ! If IUNWRI is specified, save out a "restart" file every NSAVX steps.
  ! This is not really a restart file, but contains valid coordinates,
  ! energies and progress information for feeding back to a client
  !
  IF ((NSAVX.GT.0).AND.(IUNWRI.GE.0).AND.(MOD(NCALLS,NSAVX).EQ.0)) THEN
#if KEY_MIDSINR == 1
     ! dummy memory allocation
     if(associated(v_1ij)) deallocate(v_1ij)
     if(associated(v_2ij)) deallocate(v_2ij)
     allocate(v_1ij(3,L_val,natom))
     allocate(v_2ij(3,L_val,natom))
#endif
     CALL WRIDYN(IUNWRI,NATOM,X,Y,Z,X,Y,Z,X,Y,Z,        &
#if KEY_CHEQ==1
          (/ ZERO /), (/ ZERO /), (/ ZERO /), .FALSE.,  &
#endif
#if KEY_PIPF==1
          (/ ZERO /), (/ ZERO /), (/ ZERO /), .FALSE.,  &
          0, (/ ZERO /), (/ ZERO /),                    &
#endif
#if KEY_DYNVV2==1
          .FALSE., (/ ZERO /), (/ ZERO /), (/ ZERO /),  &
#endif
          NCALLS, 0, 0, NSTEPS,                         &
          0, 0, ZERO, ZERO, 0, 0                        &
#if KEY_MIDSINR == 1
          ,q_midsinr,v_1ij,v_2ij,L_val                  &
#endif
#if KEY_BLOCK==1
          ,.FALSE.                                      &
          ,.FALSE., 0, (/ ZERO /), (/ ZERO /)           &
          ,(/ ZERO /), 1                                &
#endif
#if KEY_FOURD==1
          ,(/ ZERO /), (/ ZERO /)                       &
#endif
#if KEY_SCCDFTB==1
          ,.FALSE., .FALSE., 0, 0, ZERO, ZERO, ZERO     &
#endif
          )
#if KEY_MIDSINR == 1
     if(associated(v_1ij)) deallocate(v_1ij)
     if(associated(v_2ij)) deallocate(v_2ij)
#endif
  ENDIF
!=====================================================================
! BIOVIA CODE END
!=====================================================================
  RETURN
END SUBROUTINE MINTRJ

end module minmiz_module
