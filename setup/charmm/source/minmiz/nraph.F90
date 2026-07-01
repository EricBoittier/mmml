module nraph_m
  implicit none
contains

SUBROUTINE NRAPH(COMLYN,COMLEN,X,Y,Z)
  !-----------------------------------------------------------------------
  !     NRAPH performs a Newton-Raphson minimization. Updating is done
  !     externally. However, if necessary it could be done internally
  !     in much the same way as the printing. The fixing of atoms or
  !     the application of SHAKE constraints are not permitted by this
  !     routine.
  !
  !     The arguments are:
  !
  !     COMLYN  - the command line.
  !     COMLEN  - the length of the command line.
  !     X,Y,Z   - the starting (before) and minimized (after) coordinates.
  !
#if KEY_FLUCQ==1
  use flucqm, only: fqseln       
#endif
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use consta
  use contrl
  use egrad
  use energym
  use image
  use psf
  use shake
  use stream
  use string
#if KEY_TSM==1
  use tsms_mod
#endif 
#if KEY_FLUCQ==1
  use flucq
#endif
#if KEY_DHDGB==1
  use dhdgb,only:totals  
#endif 
  use memory
  implicit none
  real(chm_real) :: func
  real(chm_real),allocatable,dimension(:) :: grad, varb, snorm, gproj, work
  real(chm_real),allocatable,dimension(:) :: grdnew, varnew, eval, evec, dproj, ddf, varnrm

  character(len=*) COMLYN
  INTEGER       COMLEN
  real(chm_real)        X(*), Y(*), Z(*)
  !
  !     Local variables.
  !
  integer :: work_size, ddf_size
  INTEGER CONVRG, NSTEP, I, NVAR, NCALLS, NSADD
  LOGICAL QDEBUG, QERROR
  real(chm_real)  STEP, TEVAL, TOLGRD, TOLSTP
#if KEY_DHDGB==1
  real(chm_real) DUM_FHDGB(TOTALS)  
#endif
  !
  !     Storage pointers. All storage could be allocated on the stack.
  !
  INTEGER IR8,NEEDED
  !
#if KEY_TSM==1
  IF(QTSM) THEN
     CALL WRNDIE(-5,'<NRAPH>', &
          'TSM cannot be used with this minimizer.')
     RETURN
  ENDIF
#endif 
  !
  !     Parse the input command line.
  !
  NPRINT = GTRMI(COMLYN,COMLEN,'NPRI',NPRINT)
  NSADD  = GTRMI(COMLYN,COMLEN,'SADD',0)
  NSTEP  = GTRMI(COMLYN,COMLEN,'NSTE',100)
  QDEBUG = (INDXA(COMLYN,COMLEN,'DEBU')  >  0)
  IF(NPRINT < 0) NPRINT=0
  IF(PRNLEV < 2) QDEBUG=.FALSE.
  !
  STEP   = GTRMF(COMLYN,COMLEN,'STEP',PT02)
  TEVAL  = GTRMF(COMLYN,COMLEN,'TFRE',FIFTY)
  TOLGRD = GTRMF(COMLYN,COMLEN,'TOLG',ZERO)
  TOLSTP = GTRMF(COMLYN,COMLEN,'TOLS',ZERO)
  !
  IF(NTRANS > 0.OR.XNSYMM > 0) THEN
     CALL WRNDIE(-3,'<NRAPH>','NRAPH does not work with IMAGES.')
  ENDIF
  !
  !     Check that there are no unparsed commands.
  !
  CALL XTRANE(COMLYN,COMLEN,'NRAPH')
  IF (COMLEN > 0) CALL DIEWRN(-2)
  !
  !     Compute the number of variables to be minimized.
  !
  NVAR = 0
  IF (MINXYZ) THEN
     DO I = 1,NATOM
        IF (IMOVE(I)  ==  0) NVAR = NVAR + 3
     enddo
     IF (.NOT.(NVAR  ==  3 * NATOM)) THEN
        CALL WRNDIE(-3,'<NRAPH>', &
             'NRAPH does not work with fixed atoms.')
     ENDIF
  ENDIF
  !
  IF (NVAR  ==  0) THEN
     CALL WRNDIE(-1,'<NRAPH>', &
          'There are no variables to be optimised.')
     RETURN
  ENDIF
  !
  !     Check on various limitations of the implementation.
  !
  IF (QHOLO) CALL WRNDIE(-1,'<NRAPH>', &
       'SHAKE or constraints not allowed.')
  !
  IF (NTRANS  >  0) THEN
     QERROR = (NIMBON  >  0) .OR. (NIMANG  >  0) .OR. &
          (NIMDIH  >  0) .OR. (NIMIMP  >  0) &
#if KEY_CMAP==1
          .OR. (NIMCRT > 0) &     
#endif
          .OR. (NIMHB   >  0) 
     IF (QERROR) CALL WRNDIE(-5,'<NRAPH>','IMPATCH not allowed.')
  ENDIF
  !
  !     Allocate storage.
  !
  !=======================
  !
  ! !! this is not a proper use of the space allocation routines !!
  ! !! The variable IR8 could be much large than needed, and too !!
  ! !! much space would be allocated.    - BRB                   !!
  !
  IR8    = 1
  NEEDED = IR8*(8+16*NVAR+NVAR*(NVAR+1)/2 + NVAR*NVAR)
  ddf_size = nvar*(nvar + 1)/2
  work_size = 7*(nvar+1)
  call chmalloc('nraph.src','NRAPH','grad',nvar,crl=grad)
  call chmalloc('nraph.src','NRAPH','varb',nvar,crl=varb)
  call chmalloc('nraph.src','NRAPH','ddf',ddf_size,crl=ddf)
  call chmalloc('nraph.src','NRAPH','dproj',nvar,crl=dproj)
  call chmalloc('nraph.src','NRAPH','eval',nvar,crl=eval)
  call chmalloc('nraph.src','NRAPH','evec',nvar*nvar,crl=evec)
  call chmalloc('nraph.src','NRAPH','gproj',nvar,crl=gproj)
  call chmalloc('nraph.src','NRAPH','grdnew',nvar,crl=grdnew)
  call chmalloc('nraph.src','NRAPH','snorm',nvar,crl=snorm)
  call chmalloc('nraph.src','NRAPH','varnew',nvar,crl=varnew)
  call chmalloc('nraph.src','NRAPH','varnrm',nvar,crl=varnrm)
  call chmalloc('nraph.src','NRAPH','work',work_size,crl=work)

!--mfc--   GRAD   = FUNC   + IR8
!--mfc--   VARB   = GRAD   + NVAR*IR8
!--mfc--   DDF    = VARB   + NVAR*IR8
!--mfc--   DPROJ  = DDF    + (NVAR*(NVAR+1)/2)*IR8
!--mfc--   EVAL   = DPROJ  + NVAR*IR8
!--mfc--   EVEC   = EVAL   + NVAR*IR8
!--mfc--   GPROJ  = EVEC   + NVAR*NVAR*IR8
!--mfc--   GRDNEW = GPROJ  + NVAR*IR8
!--mfc--   SNORM  = GRDNEW + NVAR*IR8
!--mfc--   VARNEW = SNORM  + NVAR*IR8
!--mfc--   VARNRM = VARNEW + NVAR*IR8
!--mfc--   WORK   = VARNRM + NVAR*IR8
  !
  !     Fill the variable array.
  !
  !23456789 123456789 123456789 123456789 123456789 123456789 123456789 12
  IF (MINXYZ) CALL GETVR1(MINXYZ,NATOM,VARB,IMOVE,X,Y,Z, &
       .FALSE., &
#if KEY_CHEQ==1
       .FALSE.,CG,              & 
#endif
       'NONE',(/ZERO/),(/ZERO/), &
#if KEY_FLUCQ==1
       QFLUC,CG,FQSELN,         & 
#endif
       .FALSE.,(/ZERO/),(/0/)   &
#if KEY_DHDGB==1
!AP/MF
     ,.FALSE.,(/ZERO/),TOTALS &
#endif
       )
  !
  !     Do some initial printing.
  !
  IF (NPRINT > 0 .AND. PRNLEV >= 2) THEN
     WRITE (OUTU,'(/)')
     IF (MINXYZ) THEN
        WRITE (OUTU,'(A)') &
             ' NRAPH> An energy minimization has been requested.'
     ENDIF
     WRITE (OUTU,'(/,A,I12,A,I12,/,2(A,F12.7,A,F12.7,/))') &
          ' NSTEP  = ',NSTEP,'   NPRINT = ',NPRINT, &
          ' STEP   = ',STEP,'   TFREQ  = ',TEVAL, &
          ' TOLGRD = ',TOLGRD,'   TOLSTP = ',TOLSTP
  ENDIF
  !
  !     Start the minimization.
  !
  TEVAL  = (TEVAL/CNVFRQ)**2
  !
  CALL NRAPH2(NVAR,VARB,FUNC,GRAD,DDF, &
       NSTEP,NCALLS,STEP,TEVAL,TOLGRD,TOLSTP,CONVRG,DPROJ, &
       EVAL,EVEC,GPROJ,GRDNEW, &
       SNORM,VARNEW,VARNRM, &
       WORK,QDEBUG,OUTU,NSADD)
  !
  !     Fill the coordinate arrays with the optimised variables.
  !
  !23456789 123456789 123456789 123456789 123456789 123456789 123456789 12
  IF (MINXYZ) CALL PUTVR1(MINXYZ,NATOM,VARB,IMOVE,X,Y,Z, &
#if KEY_CHEQ==1
       .FALSE.,CG,CG,CG,        & 
#endif
       .FALSE.,'NONE',(/ZERO/),(/ZERO/),.TRUE., &
#if KEY_FLUCQ==1
       QFLUC,CG,FQSELN,         & 
#endif
       .FALSE.,(/ZERO/),(/0/)   &
#if KEY_DHDGB==1
       ,QFHDGB=.FALSE.,SDEF=DUM_FHDGB, TOTALS=TOTALS &
#endif
     )
  !
  !     Print some results.
  !
  IF (NPRINT > 0 .AND. PRNLEV >= 2) THEN
     !
     !     Test for convergence.
     !
     ! -- mikem -- 07/24/92 -- making this print consistent with other methods
     IF (CONVRG  ==  0) THEN
        WRITE (OUTU,'(/,A,I5,A,/)') &
             ' NRAPH> Minimization exiting with number of steps limit (', &
             NSTEP,') exceeded.'
     ELSE IF (CONVRG  ==  1) THEN
        WRITE (OUTU,'(/,A,F10.7,A,/)') &
             ' NRAPH> Minimization exiting with step tolerance (', &
             TOLSTP,') satisfied.'
     ELSE IF (CONVRG  ==  2) THEN
        WRITE (OUTU,'(/,A,F10.7,A,/)') &
             ' NRAPH> Minimization exiting with gradient tolerance (', &
             TOLGRD,') satisfied.'
     ELSE IF (CONVRG  == -1) THEN
        WRITE(OUTU,'(/,A,/)') &
             ' NRAPH> Minimization exiting due to time limit.'
     ELSE
        WRITE (OUTU,'(/,A,I5,A,/)') &
             ' NRAPH> Unknown convergence status (',CONVRG,').'
     ENDIF
     ! <- mikem -- 07/24/92 --
     !
     !       when minimization exits because of the number of steps limit reached,
     !       it updates the coordinates AFTER the last step, so the final energy
     !       may be different. So, let's calculate it
     !RCZ    CALL GETE(X, Y, Z, X, Y, Z, 0)
  ENDIF
  CALL PRINTE(OUTU, EPROP, ETERM, 'NRAP', 'MIN', .TRUE., &
       NCALLS, ZERO, STEP, .TRUE.)
  ! <- mikem --
  !
  !     Clear up any temporary storage space.
  !
  call chmdealloc('nraph.src','NRAPH','grad',nvar,crl=grad)
  call chmdealloc('nraph.src','NRAPH','varb',nvar,crl=varb)
  call chmdealloc('nraph.src','NRAPH','ddf',ddf_size,crl=ddf)
  call chmdealloc('nraph.src','NRAPH','dproj',nvar,crl=dproj)
  call chmdealloc('nraph.src','NRAPH','eval',nvar,crl=eval)
  call chmdealloc('nraph.src','NRAPH','evec',nvar*nvar,crl=evec)
  call chmdealloc('nraph.src','NRAPH','gproj',nvar,crl=gproj)
  call chmdealloc('nraph.src','NRAPH','grdnew',nvar,crl=grdnew)
  call chmdealloc('nraph.src','NRAPH','snorm',nvar,crl=snorm)
  call chmdealloc('nraph.src','NRAPH','varnew',nvar,crl=varnew)
  call chmdealloc('nraph.src','NRAPH','varnrm',nvar,crl=varnrm)
  call chmdealloc('nraph.src','NRAPH','work',work_size,crl=work)
  !
  !     Reset control variables.
  !
  MINXYZ = .TRUE.
  !
  RETURN
end SUBROUTINE NRAPH

SUBROUTINE NRAPH2(NVAR,VARB,FUNC,GRAD,DDF,NSTEP,NCALLS,STEP,TEVAL, &
     TOLGRD,TOLSTP,CONVRG,DPROJ,EVAL,EVEC,GPROJ, &
     GRDNEW,SNORM,VARNEW,VARNRM,WORK,QDEBUG,OUTU,NSADD)
  !-----------------------------------------------------------------------
  !     A Newton-Raphson minimization is performed by this routine.
  !
  use chm_kinds
  use vector
  use minmiz_util_module, only: mintrj
  use number
  use timerm
  implicit none
  !
  INTEGER CONVRG, NCALLS, NSTEP, NVAR, OUTU, NSADD
  LOGICAL QDEBUG
  real(chm_real) :: DDF(:), DPROJ(NVAR), EVAL(NVAR), EVEC(NVAR,NVAR)
  real(chm_real)  FUNC, GPROJ(NVAR), GRAD(NVAR), GRDNEW(NVAR)
  real(chm_real)  SNORM(NVAR), STEP, TEVAL, TOLGRD, TOLSTP
  real(chm_real)  VARB(NVAR), VARNEW(NVAR), VARNRM(NVAR), WORK(*)
  !
  INTEGER I, IVAR, J, NDAT, NMODE
  LOGICAL DONE, QTEST
  real(chm_real)  AFACT, BFACT, CFACT, COSF, FNEW, FORCE
  real(chm_real)  GCENT, GNORM,PDR, PDRMX, PDZ, RMAX, RNEW, SD, SDO
  real(chm_real)  TOLA
  !
  INTEGER    MAXDAT
  PARAMETER (MAXDAT = 13)
  INTEGER IDTP(MAXDAT), ISS(4,2)
  real(chm_real)  ASS(4,5), RSS(MAXDAT), WSS(MAXDAT), XSS(4),  &
       YSS(MAXDAT)
  !
  !     Initialise some variables.
  !
  IF (NSTEP <= 0) RETURN
  IF (NVAR <= 0) RETURN
  CONVRG = 0
  PDRMX  = STEP
  SD     = TOLSTP
  GNORM  = TOLGRD
  TOLA   = TENM5

  VARNRM(1:NVAR)=zero
  !
  NCALLS = -1
  !100 NCALLS = NCALLS + 1
  loop100: do while(.true.)
     NCALLS = NCALLS + 1
     !
     !     Calculate the energy and derivatives.
     !
     CALL NRCALC(NVAR,VARB,FUNC,GRAD,DDF,.TRUE.,NCALLS,SD)
     !
     !     Write out minimization "trajectory" frame
     !     Needs to be before convergence check!
     IF (NCALLS>0) CALL MINTRJ(NCALLS,NSTEP,SD)
     !
     !     Check convergence.
     !
     IF (SD     < TOLSTP) CONVRG = 1
     IF (GNORM  < TOLGRD) CONVRG = 2
     IF (ATLIM)            CONVRG = -1
     IF (CONVRG /= 0 .OR. NCALLS >= NSTEP) exit loop100 ! GOTO 900
     !
     !     Diagonalise the second derivative matrix.
     CALL DIAGQ(NVAR,NVAR,DDF,EVEC,WORK,WORK(NVAR+2),WORK(2*NVAR+3), &
          WORK(3*NVAR+4),EVAL,WORK(4*NVAR+5),WORK(5*NVAR+6), &
          WORK(6*NVAR+7),0)
     !
     !     Project the eigenvectors onto VARNRM.
     !
     DO I = 1,NVAR
        DPROJ(I) = DOTVEC(VARNRM,EVEC(1,I),NVAR)
     enddo
     !
     !     Low eigenvalue directions are treated now.
     !
     NMODE = 0
     DO I = 1,NVAR
        IF(EVAL(I)  <  TEVAL) NMODE = NMODE + 1
     enddo
     IF (NMODE  >  0 .AND. QDEBUG) WRITE (OUTU, &
          '(1X,''There are '',I4,'' eigenvalues less than '', &
          &''TEVAL for this Newton-Raphson step.'')') NMODE
     !
     VARNRM(1:nvar) = VARB(1:nvar)
     !
     IVAR = 0
200  CONTINUE
     IVAR = IVAR + 1
     !
     GCENT = DOTVEC(GRAD,EVEC(1,IVAR),NVAR)
     GPROJ(IVAR) = GCENT
     !
     !
     !---------------------------------------------------------------------
     !     DOWNHILL NRAPH OPTIMIZATION... 
     !---------------------------------------------------------------------
     !

     !     HLW -- check to see if the mode is going to minimized or maximized... 
     IF (IVAR  >  NSADD) THEN
        !        HLW -- test if the eigenvalue is less then the test value... 
        IF ((EVAL(IVAR)  <=  TEVAL) .AND. (IVAR+6  <  NVAR)) THEN

           !     
           !---------------------------------------------------------------------
           !     Sample points along the low eigenvalue direction for the best
           !     step size. Fit the potential to :
           !     
           !     AFACT * X**3 + BFACT * X**2 + CFACT * X + DFACT
           !     
           !     Fill arrays for a least squares solution.
           !---------------------------------------------------------------------
           !     
           NDAT = 3
           DO I = 1,NDAT
              IDTP(I) = I - 1
              RSS(I)  = ZERO
              WSS(I)  = TEN
           enddo
           YSS(1) = FUNC
           YSS(2) = GCENT
           YSS(3) = EVAL(IVAR)
           !     
           IF (QDEBUG) WRITE(OUTU,236) FUNC,GCENT,EVAL(IVAR)*HALF
236        FORMAT('  energ=',F14.6,'  fcent=',F14.6,' ffcent/2=',F14.6)
           !     
           !     Find appropriate step size, PDZ, to explore surface.
           !     
           PDZ  = PDRMX / THREE
           RNEW = -SIGN(PDZ,GCENT)
           !     
           CALL NRAPH3(NVAR,IVAR,NDAT,IDTP,FUNC,DDF,EVAL,EVEC,GRDNEW, &
                VARNEW,VARNRM,RNEW,FNEW,FORCE,RSS,YSS,WSS,QDEBUG,OUTU)
           !     
           RMAX = RNEW
           DONE = .TRUE.
           IF (FNEW  >  FUNC) THEN
              RNEW = HALF * RNEW
           ELSE IF (GCENT * FORCE  <  ZERO) THEN
              RNEW = PTSIX * RNEW
           ELSE
              RNEW = TWO * RNEW
              DONE = .FALSE.
           ENDIF
           !     
           CALL NRAPH3(NVAR,IVAR,NDAT,IDTP,FUNC,DDF,EVAL,EVEC,GRDNEW, &
                VARNEW,VARNRM,RNEW,FNEW,FORCE,RSS,YSS,WSS,QDEBUG,OUTU)
           !     
251        CONTINUE
           IF (DONE .OR. ((NDAT + 2)  >  MAXDAT)) GOTO 291
           !     
           !     Energy still dropping so continue search along this line.
           !     
           RMAX = RNEW
           DONE = (GCENT * FORCE  <  ZERO) .OR. (FNEW  >  FUNC)
           IF (.NOT.DONE) THEN
              RNEW = TWO * RNEW
              RMAX = RNEW
              CALL NRAPH3(NVAR,IVAR,NDAT,IDTP,FUNC,DDF,EVAL,EVEC, &
                   GRDNEW,VARNEW,VARNRM,RNEW,FNEW,FORCE,RSS,YSS,WSS, &
                   QDEBUG,OUTU)
           ENDIF
           GOTO 251
291        CONTINUE
           !     
           CALL LSSOLV(RSS,IDTP,YSS,WSS,NDAT,XSS,ASS,ISS,4)
           AFACT = XSS(1)
           BFACT = XSS(2)
           CFACT = XSS(3)
           !     

           PDR = BFACT * BFACT - THREE * AFACT * CFACT

           IF (PDR  >  ZERO) THEN
              PDR = SQRT(PDR)
              PDR = (PDR - BFACT) / (THREE * AFACT)
              IF (PDR * GCENT  >  ZERO) THEN
                 IF (AFACT * GCENT  >  ZERO) THEN
                    PDR = RMAX
                 ELSE
                    PDR = ZERO
                 ENDIF
              ENDIF
           ELSE
              IF(ABS(BFACT) < TOLA) THEN 
                 PDR = ZERO
              ELSE 
                 PDR = RMAX
              ENDIF
           ENDIF

           !     
           PDZ = -GCENT / EVAL(IVAR)
           !     
           IF (QDEBUG) WRITE(OUTU,336) PDZ,PDR
336        FORMAT(' Normal step size=',F14.8,'  Improved step=',F14.8)
           !     
           VARNRM(1:nvar) = VARNRM(1:nvar) + PDR * EVEC(1:nvar,IVAR)
           CALL NRCALC(NVAR,VARNRM,FUNC,GRAD,DDF,.FALSE.,-1,ZERO)
           !     

        ELSE
           !     
           !     For an eigenvector with eigenvalue greater than TEVAL.
           !     HLW -- ONLY FOR THE DOWNHILL (OPTIMIZATION) CASE.... 
           !     
           PDR  = -GCENT / EVAL(IVAR)
           RMAX = PDRMX
        ENDIF

     ELSE 

        !
        !---------------------------------------------------------------------
        !     UPHILL NRAPH OPTIMIZATION...
        !---------------------------------------------------------------------
        !

        IF (EVAL(IVAR)  >=  -TEVAL) THEN

           !     HLW -- THIS IS THE UPHILL (TRANSITION STATE) CASE... 
           !     HLW -- ONLY WHEN THE EIGENVALUE IS VERY SMALL... 

           !  
           !---------------------------------------------------------------------
           !     Sample points along the low eigenvalue direction for the best
           !     step size. Fit the potential to :
           !     
           !     AFACT * X**3 + BFACT * X**2 + CFACT * X + DFACT
           !     
           !     Fill arrays for a least squares solution.
           !---------------------------------------------------------------------
           !     
           NDAT = 3
           DO I = 1,NDAT
              IDTP(I) = I - 1
              RSS(I)  = ZERO
              WSS(I)  = TEN
           enddo
           YSS(1) = FUNC
           YSS(2) = GCENT
           YSS(3) = EVAL(IVAR)
           !     
           IF (QDEBUG) WRITE(OUTU,235) FUNC,GCENT,EVAL(IVAR)*HALF
235        FORMAT('  energ=',F14.6,'  fcent=',F14.6,' ffcent/2=',F14.6)
           !     
           !     Find appropriate step size, PDZ, to explore surface.
           !     
           PDZ  = PDRMX / THREE
           RNEW = SIGN(PDZ,GCENT)
           !     
           CALL NRAPH3(NVAR,IVAR,NDAT,IDTP,FUNC,DDF,EVAL,EVEC,GRDNEW, &
                VARNEW,VARNRM,RNEW,FNEW,FORCE,RSS,YSS,WSS,QDEBUG,OUTU)
           !     
           RMAX = RNEW
           DONE = .TRUE.
           IF (FNEW  <  FUNC) THEN
              RNEW = HALF * RNEW
           ELSE IF (GCENT * FORCE  <  ZERO) THEN
              RNEW = PTSIX * RNEW
           ELSE
              RNEW = TWO * RNEW
              DONE = .FALSE.
           ENDIF
           !     
           CALL NRAPH3(NVAR,IVAR,NDAT,IDTP,FUNC,DDF,EVAL,EVEC,GRDNEW, &
                VARNEW,VARNRM,RNEW,FNEW,FORCE,RSS,YSS,WSS,QDEBUG,OUTU)
           !     
           RMAX=RNEW

           CALL LSSOLV(RSS,IDTP,YSS,WSS,NDAT,XSS,ASS,ISS,4)
           AFACT = XSS(1)
           BFACT = XSS(2)
           CFACT = XSS(3)
           !     
           PDR = BFACT * BFACT - THREE * AFACT * CFACT

           IF (PDR  >  ZERO) THEN
              PDR = SQRT(PDR)
              PDR = (-PDR - BFACT) / (THREE * AFACT)
              IF(PDR/RMAX > TWO) PDR=TWO*RMAX
              IF (PDR * GCENT  <  ZERO) THEN
                 PDR = RMAX
              ENDIF
           ELSE
              PDR = RMAX
           ENDIF
           !

           PDZ = -GCENT / EVAL(IVAR)
           !     
           IF (QDEBUG) WRITE(OUTU,335) PDZ,PDR
335        FORMAT(' Normal step size=',F14.8,'  Improved step=',F14.8)
           !     
           VARNRM(1:nvar) = VARNRM(1:nvar) + PDR * EVEC(1:nvar,IVAR)
           CALL NRCALC(NVAR,VARNRM,FUNC,GRAD,DDF,.FALSE.,-1,ZERO)
           !     
        ELSE
           !     
           !     For an eigenvector with eigenvalue greater than TEVAL.
           !     HLW -- ONLY FOR THE UPHILL (TRANSITION STATE) CASE... 
           !     
           PDR  = -GCENT / EVAL(IVAR)
           RMAX = PDRMX

        ENDIF
     ENDIF

     IF (ABS(PDR) > ABS(RMAX)) PDR = SIGN(RMAX,PDR)
     SNORM(IVAR) = PDR
     IF (IVAR < NVAR) GOTO 200
     !
     !     Check for plodding along and adjust steps as appropriate.
     !
     PDR = LENVEC(SNORM,NVAR)
     SDO = SD
     !
     !     If new step if a significant fraction of the previous step.
     !
     QTEST = (PDR  >  SDO*HALF) .AND. (SDO  >  PT001) .AND. &
          (PDR  <  SDO*ONEPT5)
     IF (QTEST) THEN
        AFACT = ONE + ONE * (ONE - FOUR * ((PDR - SDO) / SDO)**2)
        DO I = 1,NVAR
           COSF = SNORM(I) * DPROJ(I) / (PDR * SDO)
           !
           !     Plodding in forward direction so give it a push.
           !
           IF (COSF  >  HALF) THEN
              SNORM(I) = SNORM(I) * AFACT
              IF (QDEBUG) WRITE (OUTU,'(1X,''Mode '',I4, &
                   &'' is plodding. Step scaled by '',G17.10)') I,AFACT
           ENDIF
           !
           !     Oscillating along this direction so damp it out a bit.
           !
           IF (COSF  <  -HALF) THEN
              SNORM(I) = SNORM(I) / AFACT
              IF (QDEBUG) WRITE (OUTU,'(1X,''Mode '',I4, &
                   &''is oscillating. Step scaled by '',G17.10)') &
                   I,ONE/AFACT
           ENDIF
        enddo
     ENDIF
     !
     VARNRM(1:nvar) = ZERO
     SD    = LENVEC(SNORM,NVAR)
     GNORM = SQRT(DOTVEC(GPROJ,GPROJ,NVAR) / NVAR)
     DO I = 1,NVAR
        PDR   = SNORM(I)
        VARNRM(1:nvar) = VARNRM(1:nvar) + PDR * EVEC(1:nvar,I)
        IF (QDEBUG) WRITE(OUTU,455) I,EVAL(I),GPROJ(I),SNORM(I)
     enddo
455  FORMAT(I5,'  Eigenvalue=',F12.6,'  Derivative=',F12.6, &
          '  Step=',F12.6)
     !
     IF (PDRMX  >  SD) PDRMX = PTFOUR * PDRMX
     IF (PDRMX  <  SD) PDRMX = TWO * PDRMX
     !
     !     Modify variables along accepted step.
     !
     VARB(1:nvar) = VARB(1:nvar) + VARNRM(1:nvar)
     !
     !  GOTO 100
  enddo loop100
  !
900 CONTINUE
  RETURN
  !
end SUBROUTINE NRAPH2

SUBROUTINE NRAPH3(NVAR,IVAR,NDAT,IDTP,FUNC,DDF,EVAL,EVEC,GRDNEW, &
     VARNEW,VARNRM,RNEW,FNEW,FORCE,RSS,YSS,WSS,QDEBUG,OUTU)
  !-----------------------------------------------------------------------
  !     Procedure TO SET-UP-GUESS-VARIABLES-AND-DETERMINE-FUNCTION-VALUE
  !
  use chm_kinds
  use vector
  use number
  implicit none
  !
  INTEGER NVAR, IVAR, NDAT, IDTP(*), OUTU
  LOGICAL QDEBUG
  real(chm_real) :: DDF(:), EVAL(NVAR), EVEC(NVAR,NVAR)
  real(chm_real)  FUNC, VARNEW(NVAR), VARNRM(NVAR), GRDNEW(NVAR)
  real(chm_real)  RNEW, FNEW, FORCE
  real(chm_real)  RSS(*), YSS(*), WSS(*)
  !
  INTEGER I
  !
  VARNEW(1:nvar) = VARNRM(1:nvar) + RNEW * EVEC(1:nvar,IVAR)
  CALL NRCALC(NVAR,VARNEW,FNEW,GRDNEW,DDF,.FALSE.,-1,ZERO)
  !
  !     Find the force on this mode.
  !
  FORCE = DOTVEC(GRDNEW,EVEC(1,IVAR),NVAR)
  !
  NDAT = NDAT + 1
  IDTP(NDAT) = 0
  RSS(NDAT)  = RNEW
  YSS(NDAT)  = FNEW
  WSS(NDAT)  = ONE
  NDAT = NDAT + 1
  IDTP(NDAT) = 1
  RSS(NDAT)  = RNEW
  YSS(NDAT)  = FORCE
  WSS(NDAT)  = ONE

  !
  IF (QDEBUG) WRITE(OUTU,325) RNEW,FNEW-FUNC,FORCE
325 FORMAT(' NRAPH3>   rnew=',F14.6,'   enew=',F14.6,'   fnew=',F14.6)
  !
  RETURN
end SUBROUTINE NRAPH3

SUBROUTINE NRCALC(NVAR,VARB,FUNC,GRAD,DDF,QSECD,NCALLS,STEP)
  !-----------------------------------------------------------------------
  !     The energy, forces and (if requested) the second derivatives
  !     are calculated and returned by this routine.
  !
#if KEY_FLUCQ==1
  use flucqm, only: fqseln       
#endif
  use chm_kinds
  use chm_types
  use dimens_fcm
  use exfunc
  use number
  !
  use bases_fcm
  use cnst_fcm
  use contrl
  use coord
  use deriv
  use egrad
  use energym
  use hbondm
  use image
  use inbnd
  use parallel,only: gcomb
  use psf
  use stream
#if KEY_FLUCQ==1
  use flucq
#endif 
#if KEY_DHDGB==1
  use dhdgb,only:totals
#endif
  use memory
  use heurist,only:updeci
 implicit none
 !
 !     Passed variables.
 real(chm_real),allocatable,dimension(:) :: DDM, ddtrot
 INTEGER NCALLS, NVAR
 LOGICAL QSECD
 real(chm_real) :: DDF(:), FUNC, GRAD(*), STEP, VARB(*)
 !
 !     Local variables.
 INTEGER ISPACE, NATOM3, XAT, NTOT
 LOGICAL QHDR
 real(chm_real), save :: EDIF = ZERO
#if KEY_DHDGB==1
 real(chm_real) DUM_FHDGB(TOTALS)
#endif
 !
 !     Put variables into scratch coordinate arrays.
 !
 XAT = NATOM
 CALL PUTVR1(MINXYZ,NATOM,VARB,IMOVE,X,Y,Z, &
#if KEY_CHEQ==1
      .FALSE.,CG,CG,CG,        & 
#endif
      .FALSE.,'NONE',(/ZERO/),(/ZERO/),.TRUE., &
#if KEY_FLUCQ==1
      QFLUC,CG,FQSELN,         & 
#endif
      .FALSE.,(/ZERO/),(/0/)   &
#if KEY_DHDGB==1
      ,QFHDGB=.FALSE.,SDEF=DUM_FHDGB, TOTALS=TOTALS &
#endif
      )
 !
 !     Calculate the energy, forces and second derivatives.
 !
 IF (QSECD) THEN
    !
    !     Allocate storage space.
    !
    NATOM3 = 3 * NATOM 
    ISPACE = 7 * NATOM3
    !
    call chmalloc('nraph.src','NRCALC','DDM',natom,crl=DDM)
    !
    NTOT=(NATOM3*(NATOM3+1))/2
    call chmalloc('nraph.src','NRCALC','ddtrot',ispace,crl=ddtrot)

    ddf(1:ntot) = zero
    ddm = one

    DDF(1:NTOT ) = ZERO
    DDM(1:NATOM) = ONE
    !
    !     Do list updates if appropriate
    CALL UPDECI(NCALLS,X,Y,Z,WMAIN,0,(/zero/),(/zero/),(/zero/),(/zero/),(/zero/),(/zero/))
    !
    !     Calculate energy and construct the second derivative matrix.
    !
    !23456789 123456789 123456789 123456789 123456789 123456789 123456789 12
    CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1,NATOM3,DDF)
    !
#if KEY_PARALLEL==1
    CALL VDGBR(DX,DY,DZ,1)
    CALL GCOMB(DDF,NTOT)
#endif 
    !
    IF(.NOT.QCNSTR) CALL RAISE(.TRUE.,.TRUE.,NATOM3,NATOM, &
         .FALSE.,0,DDF,DDM,DDTROT, &
         X,Y,Z,DX,DY,DZ,.FALSE.,.FALSE.)
    !
    IF (NPRINT  >  0 .AND. NCALLS  >=  0) THEN
       IF (MOD(NCALLS,NPRINT)  ==  0) THEN
          QHDR = MOD(NCALLS/NPRINT,10)  ==  1
          IF (NCALLS == 0) QHDR=.TRUE.
          IF (NCALLS == 1) QHDR=.FALSE.
          !           CALL PRINTE(OUTU, EPROP, ETERM, 'NRAP', 'MIN', QHDR,
          !    $           NCALLS, ZERO, STEP, .TRUE.)
          CALL ENEOUT(OUTU,NCALLS,STEP,ZERO)
       ENDIF
    ENDIF
    call chmdealloc('nraph.src','NRCALC','DDM',natom,crl=DDM)
    call chmdealloc('nraph.src','NRCALC','ddtrot',ispace,crl=ddtrot)
 ELSE
    !
    !     Calculate the energy and forces only.
    !
    !23456789 123456789 123456789 123456789 123456789 123456789 123456789 12
    CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
#if KEY_PARALLEL==1
    CALL VDGBR(DX,DY,DZ,1)
#endif 
 ENDIF
 !
 FUNC = EPROP(EPOT)
 CALL GETVR1(MINXYZ,NATOM,GRAD,IMOVE,DX,DY,DZ,.FALSE., &
#if KEY_CHEQ==1
      .FALSE.,CG,              & 
#endif
      'NONE',(/ZERO/),(/ZERO/), &
#if KEY_FLUCQ==1
      QFLUC,CG,FQSELN,         & 
#endif
      .FALSE.,(/ZERO/),(/0/)   &
#if KEY_DHDGB==1
      ,QFHDGB=.FALSE.,SDEF=(/ZERO/), TOTALS=TOTALS &
#endif
     )
 !
 RETURN
END SUBROUTINE NRCALC

end module nraph_m

