SUBROUTINE CONJUG(COMLYN,COMLEN)
  !
  !     ---- CONJUG performs a conjugate gradient minimization.
  !

#if KEY_CHEQ==1
  use cheq,only:qmolc,cgfix,cgtmp,qnpart,molcgt     
#endif
#if KEY_FLUCQ==1
  use flucqm, only: fqseln                          
#endif

  use chm_kinds
  use dimens_fcm
  use memory
  use exfunc
  use number
  !
  use bases_fcm
  use contrl
  use coord
  use egrad
  use energym
  use fourdm
  use image
  use psf
  use shake
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
  use dhdgb,only:totals
#endif
  implicit none
  !
  !     ---- Passed variables.
  !
  INTEGER       COMLEN
  CHARACTER(len=*) COMLYN
  !
  !     ---- Local variables.
  !
  INTEGER  CONVRG, I, NCGCYC, NSTEP, NVAR, & 
       PRTMIN, TOLITR, SAVITR, NCALLS
#if KEY_CHEQ==1
  INTEGER  NQNPART                                           
#endif
  real(chm_real)   PCUT, STEP, TOLGRD, TOLFUN, TOLSTP
  !
  !     ---- Pointers.
  !
      real(chm_real),allocatable,dimension(:) :: VARB
      real(chm_real),allocatable,dimension(:) :: NEWV
      real(chm_real),allocatable,dimension(:) :: GRAD
      real(chm_real),allocatable,dimension(:) :: P
#if KEY_DHDGB==1
      real(chm_real) DUM_FHDGB(TOTALS)
#endif
  !
  !     ---- Do some initialisation and parse the command line.
  !
  CONVRG = 0
  !
  NCGCYC = GTRMI(COMLYN,COMLEN,'NCGC',100)
  NSTEP  = GTRMI(COMLYN,COMLEN,'NSTE',100)
  PCUT   = GTRMF(COMLYN,COMLEN,'PCUT',PT9999)
  PRTMIN = GTRMI(COMLYN,COMLEN,'PRTM',1)
  NPRINT = GTRMI(COMLYN,COMLEN,'NPRI',10)
  STEP   = GTRMF(COMLYN,COMLEN,'STEP',PT02)
  TOLFUN = GTRMF(COMLYN,COMLEN,'TOLE',ZERO)
  TOLGRD = GTRMF(COMLYN,COMLEN,'TOLG',ZERO)
  TOLITR = GTRMI(COMLYN,COMLEN,'TOLI',100)
  TOLSTP = GTRMF(COMLYN,COMLEN,'TOLS',ZERO)
  !
  IF(PRNLEV < 2) PRTMIN=0
  !
  LMINUC = (INDXA(COMLYN,COMLEN,'LATT')  >  0)
  MINXYZ = (INDXA(COMLYN,COMLEN,'NOCO')  <=  0)
  !
  CALL XTRANE(COMLYN,COMLEN,'CONJUG')
  IF (COMLEN  >  0) CALL DIEWRN(-2)
  !
  !     ---- Determine the number of variables and do some printing.
  !
#if KEY_TSM==1
  CALL CALCNVAR(QTSM,BACKLS,NVAR)
#else /**/
  CALL CALCNVAR(.FALSE.,(/0/),NVAR)
#endif 
  !
  ! SAPATEL
#if KEY_CHEQ==1
  ! if some charges are to be fixed this statement needs to be modified
  ! -1 for total charge constraint
  NQNPART=0
  IF (QCGMIN) THEN
     !  NVAR is number of variables being minimzed, increment by number of
     !  fluctuating charges.  NOTE:  If not all atoms are polarizable then
     !  this should not be NATOM!
     IF (.NOT.MINXYZ) NVAR=0
     NVAR=NVAR+NATOM
     IF (QMOLC) THEN
       NQNPART=QNPART
       call chmalloc('conjug.src','CONJUG','CGFIX',NQNPART,crl=CGFIX)
       call chmalloc('conjug.src','CONJUG','CGTMP',NQNPART,crl=CGTMP)
     ELSE
       call chmalloc('conjug.src','CONJUG','CGFIX',1,crl=CGFIX)
       call chmalloc('conjug.src','CONJUG','CGTMP',1,crl=CGTMP)
     ENDIF
     CALL MOLCGT(NATOM,CG,CGFIX)
  ENDIF
#endif 
  ! SAPATEL
  IF (PRTMIN  >=  1) THEN
     WRITE (OUTU,'(/)')
     WRITE (OUTU,'(A)') &
          ' CONJUG> An energy minimization has been requested.'
     WRITE (OUTU,112) &
          ' NCGCYC = ',NCGCYC,'   NSTEP  = ',NSTEP, &
          ' PCUT   = ',PCUT,'   PRTMIN = ',PRTMIN, &
          ' STEP   = ',STEP,'   TOLFUN = ',TOLFUN, &
          ' TOLGRD = ',TOLGRD,'   TOLITR = ',TOLITR, &
          ' TOLSTP = ',TOLSTP
  ENDIF
112 FORMAT(/,A,I12,A,I12,/,A,F12.7,A,I12,/, &
       A,F12.7,A,F12.7,/,A,F12.7,A,I12,/,A,F12.7,/)
  !
  !     ---- Allocate storage.
  !
       call chmalloc('conjug.src','CONJUG','VARB',NVAR,crl=VARB)
       call chmalloc('conjug.src','CONJUG','NEWV',NVAR,crl=NEWV)
       call chmalloc('conjug.src','CONJUG','GRAD',NVAR,crl=GRAD)
       call chmalloc('conjug.src','CONJUG','P',NVAR,crl=P)
  !
  !     ---- Collect coordinates to be minimized into VARB
  !
       call GETVR1(MINXYZ,NATOM,VARB,IMOVE,X,Y,Z,LMINUC, &
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
                  ,QFHDGB=.FALSE.,SDEF=(/ZERO/), TOTALS=TOTALS &
#endif
                   )
  !
  !     ---- Perform the conjugate gradient minimization on VARB
  !
  SAVITR = MXITER
  MXITER = 60
      call CONJG2(NVAR,VARB,NEWV,GRAD, &
#if KEY_CHEQ==1
                  NATOM,QCGMIN,EPROP(SDMIN), &         
#endif
                  0,NSTEP,NCGCYC,PCUT,PRTMIN,NPRINT, &
                  STEP,P,OUTU,TOLFUN,TOLGRD, &
                  TOLITR,TOLSTP,CONVRG,NCALLS)

  MXITER = SAVITR
  !
  !     ---- Save the last step characteristics
  !
  !      EPROP(PJNK1)=NCALLS
  !      EPROP(PJNK2)=STEP
  !
  !     ---- Retrieve the coordinates from VABR
  !
      call PUTVR1(MINXYZ,NATOM,VARB,IMOVE,X,Y,Z, &
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
                  ,QFHDGB=.FALSE.,SDEF=DUM_FHDGB, TOTALS=TOTALS &
#endif
                  )
  !
  !     ---- Test for convergence and print.
  !
  IF (NPRINT /= 0 .AND. PRNLEV >= 0) THEN
     IF (CONVRG == 0) THEN
        WRITE (OUTU,'(/,A,I5,A,/)') &
             ' CONJUG> Minimization exiting with number of steps limit (', &
             NSTEP,') exceeded.'
     ELSE IF (CONVRG == 1) THEN
        WRITE (OUTU,'(/,A,F10.7,A,/)') &
             ' CONJUG> Minimization exiting with step tolerance (', &
             TOLSTP, ') satisfied.'
     ELSE IF (CONVRG == 2) THEN
        WRITE (OUTU,'(/,A,F10.7,A,/)') &
             ' CONJUG> Minimization exiting with gradient tolerance (', &
             TOLGRD,') satisfied.'
     ELSE IF (CONVRG == 3) THEN
        WRITE (OUTU,'(/,A,F10.7,A,/)') &
             ' CONJUG> Minimization exiting with function tolerance (', &
             TOLFUN,') satisfied.'
        ! -- mikem -- 07/24/92 -> making this consistenet with other methods
     ELSE IF (CONVRG == 4) THEN
        WRITE (OUTU,'(/,A,I5,A,/)') &
             ' CONJUG> Minimization exiting with number of steps limit (', &
             NSTEP,') exceeded.'
        !jb...Jay Banks 04-Dec-95: "error" exits:
        !       copied from R. Czerminski's MSI code.
     ELSE IF (CONVRG==5 .or. CONVRG==6) THEN
        WRITE (OUTU,'(/A)') ' CONJUG> Minimization exiting'// &
             ' due to an error in minimized function'
        WRITE (OUTU,'(A,I5/)') ' CONJUG> CONVRG=',CONVRG
        !jb...
     ELSE IF (CONVRG == -1) THEN
        WRITE (OUTU,'(/,A,/)') &
             ' CONJUG> Minimization exiting due to time limit'
     ELSE
        WRITE (OUTU,'(/,A,I5,A,/)') &
             ' CONJUG> Unknown convergence status (',CONVRG,').'
     ENDIF
     ! -- mikem ->
     !       when minimization exits because of the number of steps limit reached,
     !       it updates the coordinates AFTER the last step, so the final energy
     !       may be different. So, let's calculate it
     !RCZ    CALL GETE(X, Y, Z, X, Y, Z, 0)
     ! <- mikem -- 07/24/92 --
  ENDIF
  CALL PRINTE(OUTU, EPROP, ETERM, 'CONJ', 'MIN', .TRUE., &
       NCALLS, ZERO, STEP, .TRUE.)
  !
  !     ---- Clear up temporary storage space.
  !
       call chmdealloc('conjug.src','CONJUG','VARB',NVAR,crl=VARB)
       call chmdealloc('conjug.src','CONJUG','NEWV',NVAR,crl=NEWV)
       call chmdealloc('conjug.src','CONJUG','GRAD',NVAR,crl=GRAD)
       call chmdealloc('conjug.src','CONJUG','P',NVAR,crl=P)

#if KEY_CHEQ==1
  IF (QCGMIN) THEN
     IF (QMOLC) THEN
       NQNPART=QNPART
       call chmdealloc('conjug.src','CONJUG','CGFIX',NQNPART,crl=CGFIX)
       call chmdealloc('conjug.src','CONJUG','CGTMP',NQNPART,crl=CGTMP)
     ELSE
       call chmdealloc('conjug.src','CONJUG','CGFIX',1,crl=CGFIX)
       call chmdealloc('conjug.src','CONJUG','CGTMP',1,crl=CGTMP)
     ENDIF
  ENDIF
#endif 
  !
  RETURN
END SUBROUTINE CONJUG

SUBROUTINE CONJG2(NVAR,VARB,NEWV,GRAD, &
#if KEY_CHEQ==1
     NATOM,QCGMIN,SD,                & 
#endif
     ISTART,ISTOP,NCGCYC,PCUT, &
     PRTMIN,NPRINT,STEP,P,OUTU,TOLFUN,TOLGRD, &
     TOLITR,TOLSTP,CONVRG,NCALLS)
  !
  !     ----  to perform a conjugate gradient minimization.
  !
  !          NVAR       - dimension of the minimization problem
  !          VARB(NVAR) - On input, initial guess for a minimum
  !                    On output, most recent approximation to the minimum
  !          NEWV(NVAR) - A work array for storing VARB temporarily
  !          GRAD(NVAR) - A work array for storing gradients temporarily
  !          ISTOP      - Maximum number of cycles to minimize.
  !          NCGCYC     - Once every NCGCYC cycles, the minimization
  !                         direction will be reset to steepest descent
  !                         direction
  !          PCUT       - If the minimization direction is not changing
  !                         more than a cutoff angle, the direction will
  !                         be reset to steepest descent direction.  PCUT
  !                         spesifies the cosine of the cutoff angle
  !          PRTMIN     - Print level. 0- none.  1- summary 2- commentary
  !          NPRINT     - Printing interval in number of cycles
  !          STEP       - Initial step length for extrapolation to be used
  !                       whenever resetting to steepest descent
  !          P          - Array to store the search direction
  !          OUTU       - Unit number to write the output records.
  !  Minimization will terminate if one of the following criteria is met
  !          TOLFUN     - Convergence requirement on change in function
  !                         value
  !          TOLGRD     - Convergence requirement on length of gradient
  !                         vector
  !          TOLITR     - Limit on number of function evaluation for one
  !                         cycle
  !          TOLSTP     - Convergence requirement on step size
  !  On output, CONVRG indicates which criterion is satisfied,
  !   1 - TOLSTP,  2 - TOLGRD,  3 - TOLFUN,  4 - TOLITR
  !  -1 time limit or QUANTA interrupt
  !
  !
  use chm_kinds
  use dimens_fcm
  use vector
  use number
  use egrad
  use timerm
  use minmiz_util_module, only: mintrj
  use param_store, only: set_param

  implicit none

  ! SAPATEL
#if KEY_CHEQ==1
  INTEGER NATOM
  LOGICAL QCGMIN
  real(chm_real) DIFFE
#endif 
  ! SAPATEL
  INTEGER NVAR,OUTU
  real(chm_real) VARB(NVAR),NEWV(NVAR),GRAD(NVAR),P(NVAR)
  INTEGER ISTART,ISTOP,NCGCYC
  real(chm_real) PCUT
  INTEGER PRTMIN,NPRINT
  real(chm_real) STEP
  real(chm_real) FUNC
  INTEGER NCYCLE
  real(chm_real) TOLFUN,TOLGRD
  INTEGER TOLITR
  real(chm_real) TOLSTP
  INTEGER CONVRG,NCALLS
  !
  real(chm_real) OSTEP,SQT3,SD,GO2,RTINY
  real(chm_real) FNORM,BETA,PNORM,OLDNRM,COSP,FOLD
  real(chm_real) YPA,A,EYA,B,YPB,EYB,ZZ,WW,W,TM,ETM,YPTM,GN2
  INTEGER NEXTRP,NINTRP,I,ICALL
  INTEGER ERSTAT
  INTEGER CNTCG,PRTMNX
  LOGICAL NEEDSD, DONE
  !
  !     ---- Begin
  !
  !     Bug in using RSMALL, now RTINY! (mh01)
  RTINY=TENM20
  ERSTAT = 0
  SQT3 = NVAR
  SQT3 = SQRT(SQT3)
  CNTCG=0
  OSTEP = STEP
  FOLD = ZERO
  !
  newv(1:nvar) = VARB(1:NVAR)
  !
  !     ---- Minimize for NCYCLE = ISTART to ISTOP
  !
  NCYCLE = ISTART
  ICALL=0
  IF(ISTART > 0) ICALL = 1
10 CONTINUE
  !
  PRTMNX = 0
  IF (NPRINT > 0) THEN
     IF (MOD(NCYCLE,NPRINT)==0 .OR. NCYCLE==1) PRTMNX = PRTMIN
  END IF
  !
  IF (MOD(CNTCG,NCGCYC)==0) THEN
     !
     !         ---- Initialize for conjugate gradient minimization
     !
     STEP=OSTEP
     CALL EGRAD1(NVAR,VARB,NEWV,FUNC,GRAD,NCYCLE,ICALL,ERSTAT)
     ICALL = 1
     CALL ENEOUT(OUTU,NCYCLE,ZERO,FUNC)
     FNORM = LENVEC(GRAD,NVAR)
     SD=FNORM/SQT3
     GO2 = SD*SD
     !
     BETA=0.0
     PNORM = 0.0
     P(1:NVAR)=zero
  ENDIF
  !
  !       ---- Make new p vector and check for constraints to be imposed on it
  !
  newv(1:nvar) = p(1:nvar) 
  P(1:nvar) = BETA*P(1:nvar) - GRAD(1:nvar)


  ! SAPATEL
#if KEY_CHEQ==1
  ! Q variables are independent of xyzs.
  !  NOTE:  NATOM May note be correct, depending on particular system
  IF (QCGMIN) THEN
     I=NVAR-NATOM
     CALL CHECKP(I,P,VARB)
  ELSE
#endif 
     CALL CHECKP(NVAR,P,VARB)
#if KEY_CHEQ==1
  ENDIF
#endif 
  ! SAPATEL

  OLDNRM=PNORM
  PNORM = LENVEC(P,NVAR)
  CALL DOTPR(P,NEWV,NVAR,COSP)
  !
  !       ---- If direction is not changing enough, use steepd
  !
  IF (OLDNRM*PNORM /= 0.0) THEN
     COSP=COSP/(OLDNRM*PNORM)
     NEEDSD=COSP >= PCUT.AND.BETA /= 0.0.AND.MOD(CNTCG,NCGCYC) /= 0
  ELSE
     NEEDSD=.TRUE.
  ENDIF
  !
  IF (NEEDSD) THEN
     !
     !         ---- Set P vector along gradient
     !
     IF (PRTMNX > 1) WRITE (OUTU,1) COSP
1    FORMAT(/1X,'COSINE OF TWO SUCCESSIVE DIRECTIONS ',1PG14.7, &
          ' IS GREATER THAN PCUT.'/' NEXT DIRECTION WILL BE', &
          ' ALONG STEEPEST DESCENT.')
     newv(1:nvar) = p(1:nvar)
     P(1:nvar) = -GRAD(1:nvar)


     ! SAPATEL
#if KEY_CHEQ==1
     ! Q variables are independent of xyzs.
     !  NOTE:  NATOM May note be correct, depending on particular system
     IF (QCGMIN) THEN
        I=NVAR-NATOM
        CALL CHECKP(I,P,VARB)
     ELSE
#endif 
        CALL CHECKP(NVAR,P,VARB)
#if KEY_CHEQ==1
     ENDIF
#endif 
     ! SAPATEL


     !          OLDNRM=PNORM
     PNORM = LENVEC(P,NVAR)
     CALL DOTPR(P,NEWV,NVAR,COSP)
     STEP=OSTEP
     CNTCG=0
     !
  ENDIF
  !
  !       ---- Line minimization along P.  Extrapolation step
  !
  A = 0.0
  EYA=FUNC
  CALL DOTPR(P,GRAD,NVAR,YPA)
  !RCZ... 16-Feb-93 Bugfix
  B=STEP
  IF (PNORM > RSMALL) B=STEP/PNORM
  !RCZ...
  NEXTRP=0
  done = .false.
  do while ( .not. done)
     NEWV(1:nvar) = VARB(1:nvar) + B*P(1:nvar)

     CALL EGRAD1(NVAR,NEWV,VARB,FUNC,GRAD,0,0,ERSTAT)
     !          DF = LENVEC(GRAD,NVAR)
     !          SD=DF/SQT3
     IF(PRTMNX > 1) WRITE(OUTU,2)B,BETA
2    FORMAT(/1X,'EXTRAPOLATING ALONG P - B IS ',1PG14.7,3X, &
          'BETA IS',1PG14.7)
     CALL DOTPR(P,GRAD,NVAR,YPB)
     EYB=FUNC
     DONE = YPB >= 0.0 .OR.EYB >= EYA
     DONE = DONE .OR. ERSTAT /= 0
     IF (.NOT.DONE) THEN
        !           ---- Prepare for the next extrapolation
        A = B
        EYA = EYB
        YPA = YPB
        B = 2.0 * B
        NEXTRP = NEXTRP + 1
     ENDIF
     DONE = DONE .OR. NEXTRP > TOLITR
  enddo

  !       ---- Line minimization along P.  Interpolation step
  !
  NINTRP=0
  done=.false.
  loop60: do while(.not. done) 
     !          ZZ=3.0*(EYA-EYB)/(B-A)+YPA+YPB
     !     Check for zero divide.  Code by R. Czerminski, implemented in this
     !     version by Jay Banks, 04 Dec 95.
     !
     if(abs(B-A) > RTINY) then
        ZZ=3.0*(EYA-EYB)/(B-A)+YPA+YPB
     else
        ! SAPATEL
#if KEY_CHEQ==1
        IF (SD < TOLGRD)THEN
           CONVRG=2
        ELSE
#endif 
           ! SAPATEL
           CONVRG=5 ! error in function ? bug#94010rcze01
           IF(PRTMIN >= 1) then
              write(outu,'(a,4E12.4)') ' CONJUG ERROR: A,B,B-A,TM=', &
                   A,B,B-A,TM
           endif
           ! SAPATEL
#if KEY_CHEQ==1
        ENDIF
#endif 
        return
     endif
     !jb...
     WW=ZZ**2-YPA*YPB
     IF(WW < 0.0) WW=0.0
     W=SQRT(WW)
     !RCZ... 16-FEB-93 Bugfix
     !       Fix TM=B-(YPB+W-ZZ)/(YPB-YPA+2.0*W)*(B-A) to
     TM=YPB-YPA+2.0*W
     !          IF(ABS(TM) > RSMALL) TM=B-(YPB+W-ZZ)/TM*(B-A)
     !RCZ...
     !     ELSE (error) portion also by R. Czerminski, implemented in this
     !     version by Jay Banks, 04 Dec 95.
     !
     IF(ABS(TM) > RTINY) then
        TM=B-(YPB+W-ZZ)/TM*(B-A)
     else
        ! SAPATEL
#if KEY_CHEQ==1
        IF (SD < TOLGRD)THEN
           CONVRG=2
        ELSE
#endif 
           ! SAPATEL
           CONVRG=6 ! error in function ? bug#94010rcze01
           IF(PRTMIN >= 1) then
              write(outu,'(a,4E12.4)') ' CONJUG ERROR: A,B,B-A,TM=', &
                   A,B,B-A,TM
           endif
           ! SAPATEL
#if KEY_CHEQ==1
        ENDIF
#endif 
        ! SAPATEL
        return
     endif
     !jb...
     NEWV(1:nvar) = VARB(1:nvar) + TM*P(1:nvar)
     CALL EGRAD1(NVAR,NEWV,VARB,FUNC,GRAD,0,0,ERSTAT)
     ETM=FUNC
     SD = LENVEC(GRAD,NVAR)
     SD=SD/SQT3
     IF(PRTMNX > 1) WRITE(OUTU,3)TM
3    FORMAT(/1X,'INTERPOLATING - TM = ',1PG14.7)
     DONE = EYA >= ETM.AND.EYB >= ETM
     IF (.NOT.DONE) THEN
        !           ---- Prepare for the next interpolation
        CALL DOTPR(P,GRAD,NVAR,YPTM)
        IF (PRTMNX > 1) WRITE (OUTU,4) EYA,YPA,A,ETM,YPTM,TM,EYB, &
             YPB,B
4       FORMAT(/11X,'E',13X,'E''',9X,'DIST'/1X,'A:  ',2F16.8,F16.8 &
             /1X,'TM: ',2F16.8,F16.8/1X,'B:  ',2F16.8,F16.8)
        IF (YPTM < 0.0 .AND. (YPB > 0.0 .OR. ETM < EYA)) THEN
           A=TM
           EYA=ETM
           YPA=YPTM
           IF (PRTMNX > 1) WRITE (OUTU,5)
5          FORMAT(' A SET TO TM.')
        ELSE
           B=TM
           EYB=ETM
           YPB=YPTM
           IF (PRTMNX > 1) WRITE (OUTU,6)
6          FORMAT(' B SET TO TM.')
        ENDIF
     ENDIF
     NINTRP = NINTRP + 1
     DONE = DONE .OR. NINTRP+NEXTRP > TOLITR
  enddo loop60
  !
  !       ---- The new step has been found, save the coordinates and
  !       ---- call the output routine.
  !
  varb(1:nvar) = NEWV(1:NVAR)
  NCYCLE = NCYCLE + 1
  CALL ENEOUT(OUTU,NCYCLE,STEP,FUNC)
  !
  !       ---- Save the current step characteristics
  !
  NCALLS=NCYCLE
  !
  !     Write out minimization "trajectory" frame
  CALL MINTRJ(NCALLS,ISTOP,STEP)
  !
  !       ---- See if converged
  !
  ! SAPATEL
#if KEY_CHEQ==1
  DIFFE=ABS(FUNC-FOLD)
  call set_param('TOLE',DIFFE)
#endif 
  ! SAPATEL
  IF (NINTRP+NEXTRP > TOLITR)  CONVRG=4
  IF (ABS(FUNC-FOLD) < TOLFUN) CONVRG=3
  IF (SD < TOLGRD)             CONVRG=2
  IF (ABS(STEP) < TOLSTP)      CONVRG=1
  IF(ATLIM)                     CONVRG=-1
  !
  !       ---- set the next step length
  !
  GN2 = SD*SD
  !RCZ... 16-FEB-93 Bugfix
  BETA=GN2
  IF(GO2 > RSMALL) BETA=GN2/GO2
  !RCZ...
  GO2=GN2
  CNTCG=CNTCG+1
  FOLD=FUNC
  STEP=TM*PNORM
  IF (STEP==0.0) STEP=OSTEP
  !
  IF (CONVRG==0.AND.NCYCLE < ISTOP) GOTO 10
  STEP=OSTEP
  !
  RETURN
END SUBROUTINE CONJG2

