module minmiz_module
  use steepd_module, only: steepd
  implicit none
contains

  SUBROUTINE MINMIZ(COMLYN,COMLEN &
#if KEY_LIBRARY == 1
       , min_opts &
       , abnr_opts &
       , sd_opts &
#endif
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

#if KEY_LIBRARY == 1
  use api_types, only: min_settings, min_abnr_settings, min_sd_settings
#endif

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
#if KEY_OPENMM == 1
  use omm_ctrl, only: omm_requested
  use omm_main, only: omm_minimize
#endif /* KEY_OPENMM */

  use steepd_module, only: &
#if KEY_REPLICA == 1
       steepdneb, &
#endif
       steepd

  implicit none

  ! passed variables
  character(len=*) :: comlyn
  integer :: comlen
#if KEY_LIBRARY == 1
  type(min_settings), optional :: min_opts
  type(min_abnr_settings), optional :: abnr_opts
  type(min_sd_settings), optional :: sd_opts
#endif
#if KEY_OPENMM == 1
  logical :: want_openmm = .false.
#endif /* KEY_OPENMM */
  
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

#if KEY_TNPACK==1
  !yw...TNPACK, 28-Jul-95
  LOGICAL QSTART

  QLOC1  = .FALSE.
#endif 

  MINXYZ = .TRUE.
  LMINUC = .FALSE.
  QIMCEN = LIMCEN
#if KEY_OPENMM==1
  want_openmm = omm_requested(COMLYN, COMLEN, 'MINI')
  if (want_openmm) then
#if KEY_LIBRARY == 1
     if (present(min_opts)) then
        n_omm_minsteps   = min_opts%nstep
        TOLGRD = min_opts%tolgrd
     else
#endif /* KEY_LIBRARY */
        n_omm_minsteps  = GTRMI(COMLYN,COMLEN,'NSTE',0)
        TOLgrd = GTRMF(COMLYN,COMLEN,'TOLG',2.39) ! 1.0 kJ/mol/nm in kcal/mol/A
#if KEY_LIBRARY == 1
     end if
#endif /* KEY_LIBRARY */
     if(prnlev > 1) write(outu,'(a,2x,i6,2x,e11.4)') &
          'Minimization requested on OpenMM: nsteps, tolgrd', &
          n_omm_minsteps,tolgrd
     call omm_minimize(x,y,z,n_omm_minsteps,tolgrd)
     return
  endif
#endif /* KEY_OPENMM */
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
#if KEY_LIBRARY == 1
  if (present(abnr_opts)) then
     minopt = 'ABNR'
  else if (present(sd_opts)) then
     minopt = 'SD  '
  else
#endif
     MINOPT = NEXTA4(COMLYN,COMLEN)
#if KEY_LIBRARY == 1
  end if
#endif

#if KEY_DOMDEC==1
  if (q_domdec .and. .not. ( minopt == 'ABNR' .or. minopt == 'SD  ' ) ) &
       call wrndie(-1, '<MINMIZ>', &
           'Cannot minimize after enabling DOMDEC.')

  if (q_domdec .and. (numnod .gt. 1)) &
       call wrndie(-1, '<MINMIZ>', &
           'Cannot minimize with more than one MPI process after enabling DOMDEC.')
#endif 
  
#if KEY_LIBRARY == 1
  if (present(min_opts)) then
     inbfrq = min_opts%inbfrq
     ihbfrq = min_opts%ihbfrq
  end if
#endif /* KEY_LIBRARY */
  
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
#if KEY_LIBRARY == 1
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
#endif /* KEY_LIBRARY */
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
#if KEY_LIBRARY == 1
  end if
#endif /* KEY_LIBRARY */

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
#if KEY_LIBRARY == 1
     if (present(abnr_opts)) then
        call abner(comlyn, comlen, min_opts, abnr_opts)
     else
#endif /* KEY_LIBRARY */
        call abner(comlyn, comlen)
#if KEY_LIBRARY == 1
     end if
#endif /* KEY_LIBRARY */
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

#if KEY_LIBRARY == 1
        if (present(sd_opts)) then
           CALL STEEPD(COMLYN,COMLEN, min_opts, sd_opts)
        else
#endif /* KEY_LIBRARY */
           CALL STEEPD(COMLYN,COMLEN)
#if KEY_LIBRARY == 1
        end if
#endif /* KEY_LIBRARY */

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
  ENDIF
!=====================================================================
! BIOVIA CODE END
!=====================================================================
  RETURN
END SUBROUTINE MINTRJ

end module minmiz_module
