module steepd_module
  implicit none
contains

  SUBROUTINE STEEPD(COMLYN,COMLEN &
#if KEY_LIBRARY == 1
       , min_opts, sd_opts &
#endif /* KEY_LIBRARY */
       )
  !-----------------------------------------------------------------------
  !     STEEPD performs a steepest descent minimization.
  !
#if KEY_STRINGM==1 /*  VO stringm */
  use sm_config, only: repa_on, repa_freq                 
#endif
#if KEY_CHEQ==1
  use cheq,only:qmolc,cgfix,cgtmp,qnpart,molcgt           
#endif
#if KEY_FLUCQ==1
  use flucqm, only: fqseln                                
#endif
  use chm_kinds
  use dimens_fcm
  use number
  use contrl
  use coord
  use egrad
  use energym
  use fourdm
  use image
  use psf
  use stream
  use string
#if KEY_TSM==1
  use tsms_mod
  use tsmh
#endif 
#if KEY_FLUCQ==1
  use flucq
#endif
#if KEY_DHDGB==1
!AP/MF
  use dhdgb,only:sdef,qfhdgb,totals
#endif
 
  use pathm
  use memory

#if KEY_LIBRARY == 1
  use api_types, only: min_settings, min_sd_settings
#endif /* KEY_LIBRARY */
  
  implicit none
  real(chm_real),allocatable,dimension(:) :: VARB
  real(chm_real),allocatable,dimension(:) :: GRAD
  real(chm_real),allocatable,dimension(:) :: VREF
  real(chm_real),allocatable,dimension(:) :: VBEST
  real(chm_real),allocatable,dimension(:) :: GBEST
  !
  !     Passed variables.
  !
  INTEGER       COMLEN
  CHARACTER(len=*) COMLYN

#if KEY_LIBRARY == 1
  type(min_settings), optional :: min_opts
  type(min_sd_settings), optional :: sd_opts
#endif /* KEY_LIBRARY */

  !
  !     Local variables.
  !
  INTEGER  CONVRG, I, NSTEP, NVAR, NCALLS
  real(chm_real)   STEP, TOLGRD, TOLFUN, TOLSTP
  LOGICAL QNOENER  ! jwchu
  !
  !     Do some initialisation and parse the command line.
  !
  CONVRG = 0

#if KEY_LIBRARY == 1
  if (present(min_opts)) then
     NPRINT = min_opts%nprint
     NSTEP  = min_opts%nstep
  else
#endif
     NPRINT = GTRMI(COMLYN,COMLEN,'NPRI',NPRINT)
     NSTEP  = GTRMI(COMLYN,COMLEN,'NSTE',100)
#if KEY_LIBRARY == 1
  end if
#endif

#if KEY_STRINGM==1 /*  VO stringm v */
  if (repa_on.and.nstep.lt.repa_freq) then ! this should not be permitted within the 0K string method, so warn & abort
   call wrndie(0,' STEEPD> ',&
  &'NUMBER OF STEPS SMALLER THAN REPARAMETRIZATION FREQUENCY. ABORT.'&
  & )
    return
  endif
#endif /* VO ^ */

#if KEY_LIBRARY == 1
  if (present(min_opts)) then
     STEP   = min_opts%step
     TOLFUN = min_opts%tolenr
  ! VO begin
  ! the criteria below are inapplicable to the 0K string method
#if KEY_STRINGM==1
     if (.not.repa_on) then                     
#endif
        TOLGRD = min_opts%tolgrd
        TOLSTP = min_opts%tolstp
#if KEY_STRINGM==1
     endif
#endif
  else
#endif /* KEY_LIBRARY */
     STEP   = GTRMF(COMLYN,COMLEN,'STEP',PT02)
     TOLFUN = GTRMF(COMLYN,COMLEN,'TOLE',ZERO)
  ! VO begin
  ! the criteria below are inapplicable to the 0K string method
#if KEY_STRINGM==1
     if (.not.repa_on) then                     
#endif
        TOLGRD = GTRMF(COMLYN,COMLEN,'TOLG',ZERO)
        TOLSTP = GTRMF(COMLYN,COMLEN,'TOLS',ZERO)
#if KEY_STRINGM==1
     endif
#endif
#if KEY_LIBRARY == 1
  end if
#endif /* KEY_LIBRARY */

  IF (NPRINT  <  0 .OR. NPRINT  >  NSTEP) NPRINT = 0

#if KEY_LIBRARY == 1
  if (present(sd_opts)) then
     LMINUC = sd_opts%lattice > 0
     MINXYZ = sd_opts%nocoords <= 0
     QNOENER = sd_opts%noenergy > 0
  else
#endif /* KEY_LIBRARY */
     LMINUC = (INDXA(COMLYN,COMLEN,'LATT')  >  0)
     MINXYZ = (INDXA(COMLYN,COMLEN,'NOCO')  <=  0)
     QNOENER = (INDXA(COMLYN,COMLEN,'NOEN')  >  0) ! jwchu
#if KEY_LIBRARY == 1
  end if
#endif /* KEY_LIBRARY */
  
  IF(QNOENER) WRITE(OUTU,'(A)')  &
       ' STEEPD> SD minimization based on force is activated.'
  !
  CALL XTRANE(COMLYN,COMLEN,'STEEPD')
  IF (COMLEN  >  0) CALL DIEWRN(-2)
  !
  !     Determine the number of variables and do some printing.
  !
#if KEY_TSM==1
  CALL CALCNVAR(QTSM,BACKLS,NVAR)
#else /**/
  CALL CALCNVAR(.FALSE.,(/0/),NVAR)
#endif
#if KEY_DHDGB==1
!AP/MF
  IF (QFHDGB) THEN
     NVAR = NVAR + TOTALS
  ENDIF
#endif
 
#if KEY_CHEQ==1
  ! if some charges are to be fixed this statement needs to be modified
  ! -1 for total charge constraint
  IF (QCGMIN) THEN
     !        write(6,*) 'CONJUG: QCGMIN true'
     !  NVAR is number of variables being minimzed, increment by number of
     !  fluctuating charges.  NOTE:  If not all atoms are polarizable then
     !  this should not be NATOM!
     IF (.NOT.MINXYZ) NVAR=0
     NVAR=NVAR+NATOM
     IF (QMOLC) THEN
        allocate(cgfix(qnpart),cgtmp(qnpart))
     ELSE
        allocate(cgfix(qnpart),cgtmp(qnpart))
     ENDIF
     CALL MOLCGT(NATOM,CG,cgfix)
  ENDIF
#endif 
  !
  IF (NPRINT  /=  0 .AND. PRNLEV >= 2) THEN
     WRITE (OUTU,'(/)')
     WRITE (OUTU,'(A)') &
          ' STEEPD> An energy minimization has been requested.'
     WRITE (OUTU,'(/,A,I12,A,I12,/,2(A,F12.7,A,F12.7,/))') &
          ' NSTEP  = ',NSTEP, '   NPRINT = ',NPRINT, &
          ' STEP   = ',STEP,  '   TOLFUN = ',TOLFUN, &
          ' TOLGRD = ',TOLGRD,'   TOLSTP = ',TOLSTP
  ENDIF
  !
  !     Allocate storage.
  !
  call chmalloc('steepd.src','STEEPD','VARB',NVAR,crl=VARB)
  call chmalloc('steepd.src','STEEPD','GRAD',NVAR,crl=GRAD)
  call chmalloc('steepd.src','STEEPD','VREF',NVAR,crl=VREF)
  call chmalloc('steepd.src','STEEPD','VBEST',NVAR,crl=VBEST)
  call chmalloc('steepd.src','STEEPD','GBEST',NVAR,crl=GBEST)
  !
  !     Fill the variable array with the coordinates to be optimised.
  !
  CALL GETVR1(.TRUE.,NATOM,VARB,IMOVE,X,Y,Z,LMINUC, &
#if KEY_CHEQ==1
       QCGMIN,CG, &              
#endif
       XTLTYP,XTLABC,XTLREF, &
#if KEY_FLUCQ==1
       QFLUC,CG,FQSELN, &        
#endif
#if KEY_FOURD==0
       .FALSE.,(/ZERO/),(/0/) &  
#endif
#if KEY_FOURD==1
       DIM4,FDIM,IMOVE4 &        
#endif
#if KEY_DHDGB==1
       ,QFHDGB,SDEF,TOTALS &
#endif
       )
  CALL STEEP2(0,NVAR,VARB,GRAD,VREF,NSTEP, &
       STEP,TOLFUN,TOLGRD,TOLSTP,CONVRG,NCALLS, &
       VBEST,GBEST,NPRINT,QNOENER &
#if KEY_DHDGB==1
!AP/MF
       ,QFHDGB &
#endif
      )
  !
  !     Fill the coordinate arrays with the optimised variables.
  !
  CALL PUTVR1(.TRUE.,NATOM,VBEST,IMOVE,X,Y,Z, &
#if KEY_CHEQ==1
       QCGMIN,CG,CGFIX,CGTMP, &  
#endif
       LMINUC,XTLTYP,XTLABC,XTLREF,.TRUE., &
#if KEY_FLUCQ==1
       QFLUC,CG,FQSELN, &        
#endif
#if KEY_FOURD==0
       .FALSE.,(/ZERO/),(/0/) &  
#endif
#if KEY_FOURD==1
       DIM4,FDIM,IMOVE4 &        
#endif
#if KEY_DHDGB==1
!AP/MF
      ,QFHDGB,SDEF,TOTALS &
#endif

       )
  !
  !     Print some results.
  !
  IF (NPRINT  /=  0 .AND. PRNLEV >= 2) THEN
     !
     !       Test for convergence.
     !
     IF (CONVRG  ==  1) THEN
        WRITE (OUTU,'(/,A,F10.7,A,/)') &
             ' STEEPD> Minimization exiting with step tolerance (', &
             TOLSTP,') satisfied.'
     ELSE IF (CONVRG  ==  2) THEN
        WRITE (OUTU,'(/,A,F10.7,A,/)') &
             ' STEEPD> Minimization exiting with gradient tolerance (', &
             TOLGRD,') satisfied.'
     ELSE IF (CONVRG  ==  3) THEN
        WRITE (OUTU,'(/,A,F10.7,A,/)') &
             ' STEEPD> Minimization exiting with function tolerance (', &
             TOLFUN,') satisfied.'
     ELSE IF (CONVRG  ==  4) THEN
        WRITE (OUTU,'(/,A,I10,A,/)') &
             ' STEEPD> Minimization exiting with number of steps limit (', &
             NSTEP,') exceeded.'
     ELSE
        WRITE (OUTU,'(/,A,I5,A,/)') &
             ' STEEPD> Unknown convergence status (',CONVRG,').'
     ENDIF
     !yw   11-May-91 Save the last step characteristics
     !       EPROP(PJNK1)=NCALLS
     !       EPROP(PJNK2)=STEP
     !
     ! -- mikem ->
     !       when minimization exits because of the number of steps limit reached,
     !       it updates the coordinates AFTER the last step, so the final energy
     !       may be different. So, let's calculate it
     !RCZ    CALL GETE(X, Y, Z, X, Y, Z, 0)
  ENDIF
  CALL PRINTE(OUTU, EPROP, ETERM, 'STPD', 'MIN', .TRUE., &
       NCALLS, ZERO, STEP, .TRUE.)
  ! <- mikem --
  !
  !     Clear up any temporary storage space.
  !
  call chmdealloc('steepd.src','STEEPD','GRAD',NVAR,crl=GRAD)
  call chmdealloc('steepd.src','STEEPD','VARB',NVAR,crl=VARB)
  call chmdealloc('steepd.src','STEEPD','VREF',NVAR,crl=VREF)
  call chmdealloc('steepd.src','STEEPD','VBEST',NVAR,crl=VBEST)
  call chmdealloc('steepd.src','STEEPD','GBEST',NVAR,crl=GBEST)
  !
#if KEY_CHEQ==1
  IF (allocated(cgfix)) deallocate(cgfix,cgtmp)
#endif 
  RETURN
END SUBROUTINE STEEPD

SUBROUTINE STEEP2(MMODE,NVAR,VARB,GRAD,VREF,NSTEP, &
     STEP,TOLFUN,TOLGRD,TOLSTP,CONVRG,NCALLS, &
     VBEST,GBEST,NPRINT,QNOENER &
#if KEY_DHDGB==1
!AP/MF
     ,QFHDGB &
#endif
    )
  !-----------------------------------------------------------------------
  !     The SD minimization is performed here.
  !
#if KEY_STRINGM==1
  use sm_config, only: repa_on, repa_freq, stat_on, stat_freq, confcons_on, confcons_freq, chirality_on, chirality_freq
  use sm0k, only: sm0k_repa, sm0k_stat, sm0k_confcons, sm0k_chirality
  use parallel
  use mpi
  use multicom_aux
#endif
  use chm_kinds
  use dimens_fcm
  use vector
  use number
  use stream
  use timerm
  use egrad
  use energym
  use replica_mod
#if KEY_RPATH==1
  use epathmod,only: PJDX,PJDY,PJDZ,PSDX,PSDY,PSDZ,PTANX,PTANY,PTANZ 
#endif
  use neb       ! jwchuneb
  use pathm
  use minmiz_util_module, only: mintrj

  implicit none
  !     Passed variables.
  !
  INTEGER MMODE
  INTEGER CONVRG, NSTEP, NVAR, NCALLS
  real(chm_real)  FUNC, GRAD(NVAR), STEP, TOLFUN, TOLGRD, TOLSTP, &
       VARB(NVAR), VREF(NVAR), VBEST(NVAR), GBEST(NVAR)
#if KEY_REPLICA==1 && KEY_RPATH==1
  real(chm_real)  ERMS,ERMSO                              
#endif
  LOGICAL QNOENER ! jwchu
  INTEGER NPRINT  ! jwchu
  !
  INTEGER I,II,ERSTAT
  INTEGER IPT, J                  
  real(chm_real)  FBEST,FOLD,GNORM,S,GOLD,GNOW,SOLD,GODTGN
  real(chm_real)  GNOLD,FACT,FACT1,GORTH,S1
  real(chm_real)  U,DELE,GRMR0,GRMOVDF,test,GTOT,GN,GF,PSTEP,PSOLD
#if KEY_DHDGB==1
!AP/MF
  LOGICAL QFHDGB
  integer count_i
#endif 
  !
  ! VO stringm v
  !
#if KEY_STRINGM==1
  !   need this to reset the minimization
  real(chm_real) :: CONVRGG ! global convergence code
  logical :: qrepa ! flag that tells whether reparametrization should be done
  logical :: qstat ! flag that tells whether or not to call statistics
  logical :: qconfcons ! flag that tells whether to call conformational consistency
  logical :: qchir ! flag that tells whether to call chirality checker
  integer :: bug
  real(chm_real) :: step0 ! store initial assigned step
  real(chm_real) :: fold_repa ! energy after previous reparaetrization step (for convergence test)
  step0=step
#endif
  !
  ! VO stringm ^
  !
  GRAD(1:NVAR) = ZERO
  NCALLS=-1
  loop10: do while(.true.)
     NCALLS=NCALLS+1
#if KEY_STRINGM==1 /*  VO */
     !cccccccccccc string method ccccccccccccccccccc
     !    warning: reparameterization may damage structure
     !             if holonomic constraints are used
     qrepa=repa_on.and.repa_freq.gt.0
     if (qrepa) qrepa=mod(ncalls,repa_freq).eq.0
     ! compute statistics at prescribed intervals, and skip this step if no minimization iterations have been done
     if (qrepa) then
      qstat=stat_on.and.stat_freq.gt.0
      if (qstat) qstat=(ncalls.gt.0.and.mod(ncalls,stat_freq).eq.0)
      if (qstat) then 
        if (prnlev.ge.3) write(outu,'(A,I3)') &
     &      ' STEEPD> COMPUTING STRING STATISTICS.'
        call sm0k_stat(nvar,varb)
      endif
     endif
     !
     qconfcons=confcons_on.and.confcons_freq.gt.0
     if (qconfcons) qconfcons=mod(ncalls,confcons_freq).eq.0
     if (qconfcons) then
      if (prnlev.ge.3) write(outu,'(A,I3)') ' STEEPD> CHECKING CONSISTENCY.'
      call sm0k_confcons(.true.,varb)
     endif
     !
     qchir=chirality_on.and.chirality_freq.gt.0
     if (qchir) qchir=mod(ncalls,chirality_freq).eq.0
     if (qchir) then
      if (prnlev.ge.3) write(outu,'(A,I3)') ' STEEPD> CHECKING CHIRALITIES.'
      call sm0k_chirality(.true.,varb)
     endif
     !
     if (qrepa) then 
      if (prnlev.ge.3) write(outu,'(A,I3)') &
     &      ' STEEPD> PERFORMING STRING REPARAMETRIZATION.'
      call sm0k_repa(nvar, varb)
     !          step=step0 ! restore initial step (to avoid slow down, and to conform to intuitive use of step)
     endif ! qrepa
     !ccccccccccccccccccccccccccccccccccccccccccccc
#endif /* VO stringm ^ */
     !
     !     Save VARB in VREF.
     !
     ! Note: moved VREF copy before the EGRAD1 call
     !  - unintuitive, but it seems to work better - BRB 8/3/98
     vref(1:nvar) = VARB(1:NVAR)

     CALL EGRAD1(NVAR,VARB,VREF,FUNC,GRAD,NCALLS,1,ERSTAT)
#if KEY_STRINGM==1
     if (repa_on.and.ncalls.eq.0) fold_repa=func              
#endif
     !
     ! If neb is used, fill projected gradients into pgrad
     !
     IF(QNOENER)U=ZERO

     if (mmode  ==  0) CALL ENEOUT(OUTU,NCALLS,STEP,ZERO)
     !     Write out minimization "trajectory" frame
     if (mmode  ==  0 .AND. NCALLS > 0) CALL MINTRJ(NCALLS,NSTEP,STEP)
#if KEY_DHDGB==1
!AP/MF
   if (mmode  ==  0) THEN
     IF (QFHDGB)THEN
         IF (NPRINT .GT. 0 .AND. NCALLS .GE. 0) THEN
              IF (MOD(NCALLS,NPRINT) .EQ. 0) THEN
                  WRITE(120,120)NCALLS,VARB(NVAR-10+1),VARB(NVAR-10+2), &
                  VARB(NVAR-10+3),VARB(NVAR-10+4),VARB(NVAR-10+5), &
                  VARB(NVAR-10+6),VARB(NVAR-10+7),VARB(NVAR-10+8), &
                  VARB(NVAR-10+9),VARB(NVAR)
 120  FORMAT(I5,10(F7.3))
              ENDIF
        ENDIF
     ENDIF
  endif
#endif

     GNOW  = sqrt(dot_product(grad,grad))
     GNORM  = sqrt(dot_product(grad,grad)/nvar)
     ! VO stringm mods
     IF(NCALLS == 0 &
#if KEY_STRINGM==1 /*  reset after each sm0K iteration */
     & .or.qrepa    &                       
#endif
     &    ) THEN ! at first iteration
        FOLD=FUNC
        FBEST=FUNC
        GOLD=GNOW
        GNOLD=GNORM
!        SOLD=STEP ! not used
#if KEY_STRINGM==1
        if (.not.qrepa) then                 
#endif
         vbest(1:nvar) = VARB(1:NVAR) 
         gbest(1:nvar) = GRAD(1:NVAR)
#if KEY_STRINGM==1
        endif                                
#endif
     ENDIF
     ! VO stringm ^
     !        S = STEP / MAX(GNORM,RSMALL)
     !MAYGR         if(mod(ncalls,50) == 0) then
     !MAYGR           write(outu,*) 'n= ',ncalls,' step= ', step,' s= ', s,
     !MAYGR     &  ' GODTGN = ',GODTGN,' GNOW= ',GNOW,' GOLD= ',GOLD,
     !MAYGR     &  ' sold= ',sold    
     !MAYGR         endif           
     !
     IF(.NOT.QNOENER) THEN
        STEP = HALF * STEP
        IF(FUNC  <  FOLD) STEP = TWOPT4 * STEP
        ! VO: if the 0K string method is used, do not let step increase beyond initial step
#if KEY_STRINGM==1
        if (repa_on.and.step.gt.step0) step=step0  
#endif
     ENDIF
     IF (FUNC  <  FBEST) THEN
        FBEST=FUNC
        vbest(1:nvar) = VARB(1:NVAR) 
     ENDIF

     IF (NCALLS /= 0) THEN
     !    the two criteria below are not applicable in the zero-temperature string method
#if KEY_STRINGM==1 /*  VO */
        if (.not.repa_on) then                           
#endif
         IF (STEP            <  TOLSTP) CONVRG = 1
         IF (GNORM           <  TOLGRD) CONVRG = 2
#if KEY_STRINGM==1 /*  VO */
        endif
!
        if (repa_on) then
         IF (ABS(FOLD_repa-FUNC) < TOLFUN) then
          CONVRG = 3
          vbest=varb
         ENDIF
        else
#endif /* Vo stringm */
          IF (ABS(FOLD-FUNC)  <  TOLFUN) CONVRG = 3
#if KEY_STRINGM==1
        endif                                            
#endif
        IF (NCALLS          >=  NSTEP)  CONVRG = 4
#if KEY_STRINGM==1 /*  VO */
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       !    in the case of 0K string, CONVRG must hold for ALL replicas
       !    note that, with the approach below, convergence is checked only after
       !    reparametrization; this is to avoid slowdown, but also makes logical sense,
       !    since the object to minimize is a string of states, not just one state
        if (qrepa) then
            if (MPI_COMM_STRNG.ne.MPI_COMM_NULL& ! sum up all codes on string roots
     &                     .and.SIZE_STRNG.gt.1) then
             call mpi_allreduce(1d0*CONVRG,CONVRGG,1,&
     &          MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_STRNG, bug)
             CONVRGG=CONVRGG/SIZE_STRNG ! VO 3/11 assume that this is a sufficicent test
            endif
            if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1)&
     &       call PSND8(CONVRGG,1) ! broadcast to slaves
       !           check criterion:
             if (ABS(1d0*CONVRG-CONVRGG).gt.RSMALL) CONVRG=0
        elseif (repa_on) then ! if reparametrization is on, but not performed at this iteration, continue
            CONVRG=0
        endif ! qrepa (i.e. check performed after reparametrization)
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#endif /* Vo stringm */
        IF (ATLIM)                      CONVRG =-1
#if KEY_DHDGB==1
!AP/MF
        IF (QFHDGB) THEN
            IF ((CONVRG .NE. 0) .AND. (CONVRG .NE.4)) THEN
               IF (NPRINT .GT. 0 .AND. NCALLS .GE. 0) THEN
                   WRITE(120,130)NCALLS,VARB(NVAR-10+1),VARB(NVAR-10+2), &
                   VARB(NVAR-10+3),VARB(NVAR-10+4),VARB(NVAR-10+5), &
                   VARB(NVAR-10+6),VARB(NVAR-10+7),VARB(NVAR-10+8), &
                   VARB(NVAR-10+9),VARB(NVAR)
130  FORMAT(I5,10(F7.3))
               ENDIF
            ENDIF
        ENDIF
#endif

        IF (CONVRG  /=  0) RETURN
     ENDIF

     S=STEP/MAX(GNORM,RSMALL)
#if KEY_REPLICA==1
#if KEY_RPATH==1
     S1=STEP/MAX(GOLD,RSMALL)

#endif 
#endif 
     !
#if KEY_STRINGM==1 /*  VO update vbest only if convergence test fails    */
     if (qrepa) vbest=varb       
#endif
     !
     VARB(1:nvar) = VARB(1:nvar) - S * GRAD(1:nvar)
     gbest(1:nvar) = GRAD(1:NVAR)

     GOLD=GNOW
     GNOLD=GNORM
     FOLD  = FUNC
!     SOLD  = STEP
#if KEY_STRINGM==1 /*  execute only after reparametrization */
     if (qrepa) fold_repa=func       
#endif
  enddo loop10   !  GOTO 10
  return 
end SUBROUTINE STEEP2
#if KEY_REPLICA==1 /*replica_main*/

!----- ADD WORKING NEB SUBROUTINES -----

!CHARMM Element source/minmiz/steepd.src 1.1
SUBROUTINE STEEPDNEB(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     STEEPD performs a steepest descent minimization.
  !
#if KEY_CHEQ==1
  use cheq,only:cgfix,cgtmp,qmolc,qnpart,molcgt    
#endif
#if KEY_FLUCQ==1
  use flucqm, only: fqseln                         
#endif
  use chm_kinds
  use dimens_fcm
  use number
  use contrl
  use coord
  use egrad
  use energym
  use fourdm
  use image
  use psf
  use stream
  use string
#if KEY_TSM==1
  use tsms_mod
  use tsmh
#endif 
#if KEY_FLUCQ==1
  use flucq
#endif 
  use pathm
  use memory
#if KEY_DHDGB==1
  use dhdgb,only:totals
#endif
  !
 implicit none
 real(chm_real),allocatable,dimension(:) :: VARB
 real(chm_real),allocatable,dimension(:) :: GRAD
 real(chm_real),allocatable,dimension(:) :: VREF
 real(chm_real),allocatable,dimension(:) :: VBEST
 real(chm_real),allocatable,dimension(:) :: GBEST
 real(chm_real),allocatable,dimension(:) :: PTOTO
 real(chm_real),allocatable,dimension(:) :: PGRADO
 real(chm_real),allocatable,dimension(:) :: PGRAD
 real(chm_real),allocatable,dimension(:) :: PSGRAD
 real(chm_real),allocatable,dimension(:) :: PTHTAN
 real(chm_real),allocatable,dimension(:) :: RMASS
 real(chm_real),allocatable,dimension(:) :: GPREV
 real(chm_real),allocatable,dimension(:) :: UOFT
 !     Passed variables.
 !
 INTEGER       COMLEN
 CHARACTER(len=*) COMLYN
 !
 !     Local variables.
 !
 INTEGER  CONVRG, I, NSTEP, NVAR, NCALLS
 real(chm_real)   STEP, TOLGRD, TOLFUN, TOLSTP
 !
 !     Do some initialisation and parse the command line.
 !
#if KEY_DHDGB==1
  real(chm_real) DUM_FHDGB(TOTALS)
#endif
 CONVRG = 0
 !
 NPRINT = GTRMI(COMLYN,COMLEN,'NPRI',NPRINT)
 NSTEP  = GTRMI(COMLYN,COMLEN,'NSTE',100)
 STEP   = GTRMF(COMLYN,COMLEN,'STEP',PT02)
 TOLFUN = GTRMF(COMLYN,COMLEN,'TOLE',ZERO)
 TOLGRD = GTRMF(COMLYN,COMLEN,'TOLG',ZERO)
 TOLSTP = GTRMF(COMLYN,COMLEN,'TOLS',ZERO)
 !
 IF (NPRINT  <  0 .OR. NPRINT  >  NSTEP) NPRINT = 0
 !
 LMINUC = (INDXA(COMLYN,COMLEN,'LATT')  >  0)
 MINXYZ = (INDXA(COMLYN,COMLEN,'NOCO')  <=  0)
 !
 CALL XTRANE(COMLYN,COMLEN,'STEEPD')
 IF (COMLEN  >  0) CALL DIEWRN(-2)
 !
 !     Determine the number of variables and do some printing.
 !
#if KEY_TSM==1
 CALL CALCNVAR(QTSM,BACKLS,NVAR)
#else /**/
 CALL CALCNVAR(.FALSE.,(/0/),NVAR)
#endif 
#if KEY_CHEQ==1
 ! if some charges are to be fixed this statement needs to be modified
 ! -1 for total charge constraint
 IF (QCGMIN) THEN
    !        write(6,*) 'CONJUG: QCGMIN true'
    !  NVAR is number of variables being minimzed, increment by number of
    !  fluctuating charges.  NOTE:  If not all atoms are polarizable then
    !  this should not be NATOM!
    IF (.NOT.MINXYZ) NVAR=0
    NVAR=NVAR+NATOM
    IF (QMOLC) THEN
       allocate(cgtmp(qnpart),cgfix(qnpart))
    ELSE
       allocate(cgtmp(1),cgfix(1))
    ENDIF
    CALL MOLCGT(NATOM,CG,CGFIX)
 ENDIF
#endif 
 !
 IF (NPRINT  /=  0 .AND. PRNLEV >= 2) THEN
    WRITE (OUTU,'(/)')
    WRITE (OUTU,'(A)') &
         ' STEEPD> An energy minimization has been requested.'
    WRITE (OUTU,'(/,A,I12,A,I12,/,2(A,F12.7,A,F12.7,/))') &
         ' NSTEP  = ',NSTEP, '   NPRINT = ',NPRINT, &
         ' STEP   = ',STEP,  '   TOLFUN = ',TOLFUN, &
         ' TOLGRD = ',TOLGRD,'   TOLSTP = ',TOLSTP
 ENDIF
 !
 !     Allocate storage.
 !
 call chmalloc('steepd.src','STEEPDNEB','VARB_hv',NVAR,crl=VARB)
 call chmalloc('steepd.src','STEEPDNEB','GRAD_hv',NVAR,crl=GRAD)
 call chmalloc('steepd.src','STEEPDNEB','VREF_hv',NVAR,crl=VREF)
 call chmalloc('steepd.src','STEEPDNEB','VBEST_hv',NVAR,crl=VBEST)
 call chmalloc('steepd.src','STEEPDNEB','GBEST_hv',NVAR,crl=GBEST)
 call chmalloc('steepd.src','STEEPDNEB','PTOTO_hv',NVAR,crl=PTOTO)
 call chmalloc('steepd.src','STEEPDNEB','PGRADO_hv',NVAR,crl=PGRADO)
 call chmalloc('steepd.src','STEEPDNEB','PGRAD_hv',NVAR,crl=PGRAD)
 call chmalloc('steepd.src','STEEPDNEB','PSGRAD_hv',NVAR,crl=PSGRAD)
 call chmalloc('steepd.src','STEEPDNEB','PTHTAN_hv',NVAR,crl=PTHTAN)
 call chmalloc('steepd.src','STEEPDNEB','RMASS_hv',NVAR,crl=RMASS)
 call chmalloc('steepd.src','STEEPDNEB','GPREV_hv',NVAR,crl=GPREV)
 call chmalloc('steepd.src','STEEPDNEB','UOFT_hv',NVAR,crl=UOFT)
 !
 !     Fill the variable array with the coordinates to be optimised.
 !
 CALL GETVR1(.TRUE.,NATOM,VARB,IMOVE,X,Y,Z,LMINUC, &
#if KEY_CHEQ==1
      QCGMIN,CG, &             
#endif
      XTLTYP,XTLABC,XTLREF, &
#if KEY_FLUCQ==1
      QFLUC,CG,FQSELN, &       
#endif
#if KEY_FOURD==0
      .FALSE.,(/ZERO/),(/0/) & 
#endif
#if KEY_FOURD==1
      DIM4,FDIM,IMOVE4 &       
#endif
#if KEY_DHDGB==1
!AP/MF
     ,.FALSE.,(/ZERO/),TOTALS &
#endif
      )
 !
 CALL GETVR1(.TRUE.,NATOM,RMASS,IMOVE,AMASS, &
      AMASS,AMASS,LMINUC, &
#if KEY_CHEQ==1
      QCGMIN,CG, &             
#endif
      XTLTYP,XTLABC,XTLREF, &
#if KEY_FLUCQ==1
      QFLUC,CG,FQSELN, &       
#endif
#if KEY_FOURD==0
      .FALSE.,(/ZERO/),(/0/) & 
#endif
#if KEY_FOURD==1
      DIM4,FDIM,IMOVE4 &       
#endif
#if KEY_DHDGB==1
!AP/MF
     ,.FALSE.,(/ZERO/),TOTALS &
#endif
      )
 !
 CALL STEEP2NEB(0,NVAR,VARB,GRAD,VREF, NSTEP,  &
      STEP,TOLFUN,TOLGRD,TOLSTP,CONVRG,NCALLS,VBEST,GBEST, &
      PGRAD,PSGRAD,PTHTAN,RMASS,PGRADO,  &
      GPREV,PTOTO)
 !
 !     Fill the coordinate arrays with the optimised variables.
 !
 CALL PUTVR1(.TRUE.,NATOM,VBEST,IMOVE,X,Y,Z, &
#if KEY_CHEQ==1
      QCGMIN,CG,CGFIX,CGTMP,  & 
#endif
      LMINUC,XTLTYP,XTLABC,XTLREF,.TRUE., &
#if KEY_FLUCQ==1
      QFLUC,CG,FQSELN, &        
#endif
#if KEY_FOURD==0
      .FALSE.,(/ZERO/),(/0/) &  
#endif
#if KEY_FOURD==1
      DIM4,FDIM,IMOVE4 &        
#endif
#if KEY_DHDGB==1
       ,QFHDGB=.FALSE.,SDEF=DUM_FHDGB, TOTALS=TOTALS &
#endif
      )
 !
 !     Print some results.
 !
 IF (NPRINT  /=  0 .AND. PRNLEV >= 2) THEN
    !
    !       Test for convergence.
    !
    IF (CONVRG  ==  1) THEN
       WRITE (OUTU,'(/,A,F10.7,A,/)') &
            ' STEEPD> Minimization exiting with step tolerance (', &
            TOLSTP,') satisfied.'
    ELSE IF (CONVRG  ==  2) THEN
       WRITE (OUTU,'(/,A,F10.7,A,/)') &
            ' STEEPD> Minimization exiting with gradient tolerance (', &
            TOLGRD,') satisfied.'
    ELSE IF (CONVRG  ==  3) THEN
       WRITE (OUTU,'(/,A,F10.7,A,/)') &
            ' STEEPD> Minimization exiting with function tolerance (', &
            TOLFUN,') satisfied.'
    ELSE IF (CONVRG  ==  4) THEN
       WRITE (OUTU,'(/,A,I10,A,/)') &
            ' STEEPD> Minimization exiting with number of steps limit (', &
            NSTEP,') exceeded.'
    ELSE
       WRITE (OUTU,'(/,A,I5,A,/)') &
            ' STEEPD> Unknown convergence status (',CONVRG,').'
    ENDIF
    !yw   11-May-91 Save the last step characteristics
    !       EPROP(PJNK1)=NCALLS
    !       EPROP(PJNK2)=STEP
    !
    ! -- mikem ->
    !       when minimization exits because of the number of steps limit reached,
    !       it updates the coordinates AFTER the last step, so the final energy
    !       may be different. So, let's calculate it
    !RCZ    CALL GETE(X, Y, Z, X, Y, Z, 0)
 ENDIF
 CALL PRINTE(OUTU, EPROP, ETERM, 'STPD', 'MIN', .TRUE., &
      NCALLS, ZERO, STEP, .TRUE.)
 ! <- mikem --
 !
 !     Clear up any temporary storage space.
 !
 call chmdealloc('steepd.src','STEEPDNEB','GRAD',NVAR,crl=GRAD)
 call chmdealloc('steepd.src','STEEPDNEB','VARB',NVAR,crl=VARB)
 call chmdealloc('steepd.src','STEEPDNEB','VREF',NVAR,crl=VREF)
 call chmdealloc('steepd.src','STEEPDNEB','VBEST',NVAR,crl=VBEST)
 call chmdealloc('steepd.src','STEEPDNEB','GBEST',NVAR,crl=GBEST)
 call chmdealloc('steepd.src','STEEPDNEB','PTOTO',NVAR,crl=PTOTO)
 call chmdealloc('steepd.src','STEEPDNEB','PGRADO',NVAR,crl=PGRADO)
 call chmdealloc('steepd.src','STEEPDNEB','PGRAD',NVAR,crl=PGRAD)
 call chmdealloc('steepd.src','STEEPDNEB','PSGRAD',NVAR,crl=PSGRAD)
 call chmdealloc('steepd.src','STEEPDNEB','PTHTAN',NVAR,crl=PTHTAN)
 call chmdealloc('steepd.src','STEEPDNEB','RMASS',NVAR,crl=RMASS)
 call chmdealloc('steepd.src','STEEPDNEB','GPREV',NVAR,crl=GPREV)
 call chmdealloc('steepd.src','STEEPDNEB','UOFT',NVAR,crl=UOFT)
 !
#if KEY_CHEQ==1
 IF (allocated(cgfix)) deallocate(cgfix,cgtmp)
#endif 
 RETURN
END SUBROUTINE STEEPDNEB

SUBROUTINE STEEP2NEB(MMODE,NVAR,VARB,GRAD,VREF,NSTEP,STEP, &
     TOLFUN,TOLGRD,TOLSTP,CONVRG,NCALLS,VBEST,PTOT, &
     PGRAD,PSGRAD,PTHTAN,RMASS,PGRADO,PSGRADO,PTOTO)
  !-----------------------------------------------------------------------
  !     The SD minimization is performed here.
  !
  use abnerm,only:tforce

  use chm_kinds
  use dimens_fcm
  use vector
  use number
  use stream
  use timerm
  use egrad
  use energym
  use replica_mod
#if KEY_RPATH==1
  use epathmod,only: PJDX,PJDY,PJDZ,PSDX,PSDY,PSDZ,PTANX,PTANY,PTANZ 
#endif

  use neb       ! jwchuneb
  use pathm
  use deriv
  use psf
  use image
  use contrl
  use parallel
#if KEY_FLUCQ==1
  use flucq
#endif 
# if KEY_DHDGB==1
 use dhdgb,only:totals 
#endif
 use minmiz_util_module, only: mintrj

 implicit none
  !
  !     Passed variables.
  !
  INTEGER MMODE
  INTEGER CONVRG, NSTEP, NVAR, NCALLS
  real(chm_real)  FUNC, GRAD(NVAR), STEP, TOLFUN, TOLGRD, TOLSTP, &
       VARB(NVAR), VREF(NVAR), PGRADO(NVAR), PTOT(NVAR)
  real(chm_real)  PGRAD(NVAR),PSGRAD(NVAR),ERMS,ERMSO,STEP1, &
       PTHTAN(NVAR),RMASS(NVAR),PSGRADO(NVAR),PTOTO(NVAR), &
       VBEST(NVAR)
  !
  INTEGER I,ERSTAT
  INTEGER IPT, J  
  real(chm_real)  FBEST,FOLD,GNORM,S,S1,GOLD,GNOW,SOLD,GODTGN
  real(chm_real)  GNOLD,FACT,GORTH
  real(chm_real)  DELE,GRMR0,GRMOVDF,test,GTOT,PSTEP,PSOLD
  real(chm_real)  NORMGRAD, NORMPGRAD, NORMPGRADOLD
  real(chm_real)  STEPGRAD,STEPSGRAD,NORMPSGRAD,NORMPSGRADOLD
  real(chm_real)  NORMPTOT,normptotold,DOTPTOTWOLD
  real(chm_real)  dotpgradwo,dotpsgradwo
#if KEY_DHDGB==1
  real(chm_real) DUM_FHDGB(TOTALS)
#endif

  !
  NCALLS=-1
10 NCALLS=NCALLS+1
  !
  !     Save VARB in VREF.
  !
  ! Note: moved VREF copy before the EGRAD1 call
  !  - unintuitive, but it seems to work better - BRB 8/3/98
  vref(1:nvar) = VARB(1:NVAR)
  !
  IF(NCALLS == 0) THEN
     pgrado(1:nvar) = VARB(1:NVAR)
     GRAD(1:NVAR) = ZERO
     PGRAD(1:NVAR) = ZERO
     PSGRAD(1:NVAR) = ZERO
     PSGRADO(1:NVAR) = ZERO
     PTOT(1:NVAR) = ZERO
     ERMS=ZERO
     ERMSO=ZERO
     STEP1=STEP
     STEPGRAD=STEP
     STEPSGRAD=STEP
  ENDIF

  !
  CALL EGRAD1(NVAR,VARB,VREF,FUNC,GRAD,NCALLS,1,ERSTAT)
  !
  ! neb is used, fill projected gradients into pgrad
  !
  !       U=ZERO
  CALL EGRADNEB(NVAR,VARB,VREF,GRAD,PGRAD,PSGRAD,ERSTAT, &
       PTHTAN)
  !
  ! at this point...
  ! GRAD  : Total internal gradients ( no spring gradients )
  ! PGRAD : projected GRAD, i.e., gradiens along perpendicular directions
  ! PSGRAD: Total spring gradients, projected in TFORCE
  !
  !        CALL PFORCE(QPCYCL,PGRAD,PTHTAN,NVAR,NREPL,NPATOMS,IREPS,
  !     $              PJDX,PJDY,PJDZ,PTANX,PTANY,PTANZ,QPCIMG,ICLREP)
#if KEY_REPLICA==1
#if KEY_RPATH==1
  CALL TFORCE(QPCYCL,PSGRAD,PTHTAN,NVAR,NREPL,NPATOMS,IREPS, &
       PJDX,PJDY,PJDZ,PTANX,PTANY,PTANZ,QPCIMG,ICLREP)

  ERMS=ETERM(PRMS)
#endif 
#endif 

  call addvec(PSGRAD,PGRAD,PTOT,NVAR)

  CALL PUTVR1(.TRUE.,NATOM,PTOT,IMOVE,DX,DY,DZ, &
#if KEY_CHEQ==1
       .false.,(/ZERO/),(/ZERO/),(/ZERO/), & 
#endif
       LMINUC,XTLTYP,XTLABC,XTLREF,.TRUE., &
#if KEY_FLUCQ==1
       .false.,(/ZERO/),(/0/), & 
#endif
       .false.,(/ZERO/),(/0/) &
#if KEY_DHDGB==1
       ,QFHDGB=.FALSE.,SDEF=DUM_FHDGB, TOTALS=TOTALS &
#endif
    )


  if (mmode  ==  0) CALL ENEOUT(OUTU,NCALLS,STEP,ZERO)
  !     Write out minimization "trajectory" frame
  if (mmode  ==  0 .AND. NCALLS > 0) CALL MINTRJ(NCALLS,NSTEP,STEP)

  normpgrad = SQRT( DOTVEC(PGRAD,PGRAD,NVAR) / NVAR)
  normgrad = SQRT( DOTVEC(GRAD,GRAD,NVAR) / NVAR)
  normpsgrad = SQRT( DOTVEC(PSGRAD,PSGRAD,NVAR) / NVAR)
  normptot = SQRT( DOTVEC(PTOT,PTOT,NVAR) / NVAR)
  !
  IF(NCALLS == 0) THEN
     FOLD=FUNC
     FBEST=FUNC
     ERMSO=ERMS     
     normpgradold=normpgrad
     normpsgradold=normpsgrad
     normptotold=normptot
     SOLD=STEP
     pgrado(1:nvar)  = PGRAD (1:NVAR)
     psgrado(1:nvar) = PSGRAD(1:NVAR)
     ptoto(1:nvar)   = PTOT  (1:NVAR)
  ENDIF
  !         if (mmode  ==  0) CALL ENEOUT(OUTU,NCALLS,STEP,ZERO)

  dotpgradwo=(DOTVEC(PGRAD,PGRADO,NVAR)/NVAR)/ &
       (normpgrad*normpgradold)
  dotpsgradwo=(DOTVEC(PSGRAD,PSGRADO,NVAR)/NVAR)/ &
       (normpsgrad*normpsgradold)
  dotptotwold=(DOTVEC(PTOT,PTOTO,NVAR)/NVAR)/ &
       (normptot*normptotold)
  !
  !       STEP = HALF * STEP
  !       IF(dotptotwold  >=  0.0) STEP = TWOPT4 * STEP

  stepgrad = HALF * stepgrad
  IF(dotpgradwo  >=  0.0) stepgrad = TWOPT4 * stepgrad

  stepsgrad = HALF * stepsgrad
  IF(dotpsgradwo  >=  0.0) stepsgrad = TWOPT4 * stepsgrad

  step=max(stepgrad,stepsgrad)
  !
  vbest(1:nvar) = VARB(1:NVAR)

  IF (NCALLS /= 0) THEN
     IF (NCALLS          >=  NSTEP)  CONVRG = 4
     IF (normptot        <  TOLGRD) CONVRG = 2
     IF (STEP            <   TOLSTP) CONVRG = 1
     IF (ATLIM)                      CONVRG =-1
     IF (CONVRG  /=  0) RETURN
  ENDIF

  S=stepgrad/MAX(normpgrad,RSMALL)
  S1=stepsgrad/MAX(normpsgrad,RSMALL)
  !       stepgrad=stepgrad/100
  !       stepsgrad=stepsgrad/100
  DO I = 1,NVAR
     VARB(I) = VARB(I)-S*PGRAD(I)-S1*PSGRAD(I)

  enddo
    ! 
    pgrado(1:nvar)  = PGRAD(1:NVAR)
    psgrado(1:nvar) = PSGRAD(1:NVAR)
    ptoto(1:nvar)   = PTOT(1:NVAR)

    normpgradold=normpgrad
    normpsgradold=normpsgrad
    normptotold=normptot
    FOLD  = FUNC
    SOLD  = STEP
    GOTO 10
  END SUBROUTINE STEEP2NEB
#endif /* (replica_main)*/

subroutine steepd_dummy()
  return
end subroutine steepd_dummy

end module steepd_module
