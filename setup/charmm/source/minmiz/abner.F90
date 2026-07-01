module abnerm
  !
contains
  !
  subroutine abner(comlyn,comlen &
#if KEY_LIBRARY == 1
       , min_opts, abnr_opts &
#endif /* KEY_LIBRARY */
       )
    !-----------------------------------------------------------------------
    !     ABNER performs an ABNR minimization.
    !
#if KEY_CHEQ==1
  use cheq,only:qmolc,cgfix,cgtmp,qnpart,molcgt     
#endif
#if KEY_FLUCQ==1
  use flucqm,only: fqseln                           
#endif
    !
  use chm_kinds
  use dimens_fcm
  use exfunc
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
#endif 
#if KEY_FLUCQ==1
  use flucq
#endif 
  use parallel
  use repdstr
#if KEY_DHDGB==1
!AP/MF
   use dhdgb
#endif

#if KEY_LIBRARY == 1
  use api_types, only: min_settings, min_abnr_settings
#endif /* KEY_LIBRARY */

    implicit none
    !
    !     Passed variables.
    !
    INTEGER       COMLEN
    CHARACTER(len=*) COMLYN

#if KEY_LIBRARY == 1
  type(min_settings), optional :: min_opts
  type(min_abnr_settings), optional :: abnr_opts
#endif /* KEY_LIBRARY */

    !
    !     Local variables.
    !
    INTEGER :: &
         i, ncalls, &
         CONVRG, MINDIM, NSTEP, NVAR, TOLITR
    
    LOGICAL :: QDEBUG, QLOCAL
    
    REAL(chm_real) :: &
         EIGRNG, PSTRCT, SDSTP, STPLIM, &
         STRICT, TOLGRD, TOLFUN, TOLSTP, &
         FMEM
    !
    !     Pointers.
    !
    !C      INTEGER  EVAL, EVEC, FUNC, GNRM, GRAD, LPREV, PARM, SECDER,
    !C     $         VARB, VRED, VRSDL, VWRK, WRK1, WRK2, WRK3, XRED,
    !C     $         VWRK1, PGRAD, PSGRAD, PFUNC, PTHTAN, PTHTAN1, ! jwchuneb
    !C     $         WRK4, XREDO, XREDDF, VSVD, USVD, SECDER1, VRED1, PGNRM
    !
    INTEGER,ALLOCATABLE,DIMENSION(:),SAVE :: LPREV
    REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:),SAVE :: EVAL, &
         EVEC,FUNC,GNRM,GRAD,PARM,SECDER,SECDER1,VARB,VRED,VRED1, &
         VRSDL,VWRK,WRK1,WRK2,WRK3,XRED,PFUNC,PSGRAD,PGRAD,PTHTAN, &
         PTHTAN1,VWRK1,WRK4,XREDO,XREDDF,USVD,VSVD,PGNRM

    !  initialize local vars
    CONVRG = 0
    MINDIM = 5
    NSTEP = 100
    NVAR = 0
    TOLITR = 100

    EIGRNG = PT0005
    PSTRCT = ZERO
    SDSTP = PT02
    STPLIM = ONE
    STRICT = PTONE
    TOLGRD = ZERO
    TOLFUN = ZERO
    TOLSTP = ZERO
    FMEM = ZERO

#if KEY_TSM==1
    IF(QTSM) THEN
       IF(PIGSET) THEN
          CALL WRNDIE(-5,'<ABNER>', &
               'TSM PIGGyBACK support has not been implemented yet.')
          RETURN
       ENDIF
    ENDIF
#endif 

    QLOCAL = LMINUC

#if KEY_LIBRARY == 1
    if (present(min_opts)) then
       nprint = min_opts%nprint
       nstep = min_opts%nstep
       sdstp = min_opts%step
       tolfun = min_opts%tolenr
       tolgrd = min_opts%tolgrd
       tolstp = min_opts%tolstp
    end if  ! present(min_opts)

    if (present(abnr_opts)) then
       eigrng = abnr_opts%eigrng
       mindim = abnr_opts%mindim
       fmem = abnr_opts%fmem
       stplim = abnr_opts%stplim
       strict = abnr_opts%strict
       tolitr = abnr_opts%tolitr
    end if  ! present(abnr_opts)
#endif /* KEY_LIBRARY */

    ! parse command line
    EIGRNG = GTRMF(COMLYN,COMLEN,'EIGR', EIGRNG)
    MINDIM = GTRMI(COMLYN,COMLEN,'MIND', MINDIM)
    NPRINT = GTRMI(COMLYN,COMLEN,'NPRI', NPRINT)
    NSTEP  = GTRMI(COMLYN,COMLEN,'NSTE', NSTEP)
    !C      PSTRCT = GTRMF(COMLYN,COMLEN,'PSTR',PT0001)
    PSTRCT = GTRMF(COMLYN,COMLEN,'PSTR', PSTRCT)
    !RCZ 91/11/19 FMEM - memory factor introduced to calculate GRADAV & STEPAV
    FMEM   = GTRMF(COMLYN,COMLEN,'FMEM', FMEM)
    QDEBUG = INDXA(COMLYN,COMLEN,'DEBU')  >  0
    SDSTP  = GTRMF(COMLYN,COMLEN,'STEP', SDSTP)
    STPLIM = GTRMF(COMLYN,COMLEN,'STPL', STPLIM)
    STRICT = GTRMF(COMLYN,COMLEN,'STRI', STRICT)
    TOLFUN = GTRMF(COMLYN,COMLEN,'TOLE', TOLFUN)
    TOLGRD = GTRMF(COMLYN,COMLEN,'TOLG', TOLGRD)
    TOLITR = GTRMI(COMLYN,COMLEN,'TOLI', TOLITR)
    TOLSTP = GTRMF(COMLYN,COMLEN,'TOLS', TOLSTP)

    IF (NPRINT  <  0 .OR. NPRINT  >  NSTEP) NPRINT = 0
    IF (PRNLEV < 2) QDEBUG=.FALSE.
    !
    LMINUC = (INDXA(COMLYN,COMLEN,'LATT')  >  0)
    MINXYZ = (INDXA(COMLYN,COMLEN,'NOCO')  <=  0)
    !
    CALL XTRANE(COMLYN,COMLEN,'ABNER')
    IF (COMLEN  >  0) CALL DIEWRN(-2)
    !
    !     Determine the number of variables and do some printing.
    !
    CALL CALCNVAR(.FALSE.,(/0/),NVAR)
#if KEY_DHDGB==1
!AP/MF
    IF (QFHDGB) THEN
       NVAR=NVAR+TOTALS
    ENDIF
#endif
#if KEY_CHEQ==1
    !  GET CHEQ Variables --- charges
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
          allocate(cgfix(1),cgtmp(1))
       ENDIF
       CALL MOLCGT(NATOM,CG,CGFIX)
    ENDIF
#endif 
    !
    !     Do some initial printing.
    !
    IF (NPRINT  /=  0 .AND. PRNLEV >= 2) THEN
       WRITE (OUTU,'(/)')
       IF (MINXYZ .AND. LMINUC) THEN
          WRITE (OUTU,'(A)') &
               ' ABNER> A full crystal minimization has been requested.'
       ELSE IF (LMINUC) THEN
          WRITE (OUTU,'(A)') &
               ' ABNER> A lattice parameter minimization has been ' &
               //'requested.'
       ELSE IF (MINXYZ) THEN
          WRITE (OUTU,'(A)') &
               ' ABNER> An energy minimization has been requested.'
       ENDIF
       WRITE(OUTU,112) &
            ' EIGRNG = ',EIGRNG,'   MINDIM = ',MINDIM, &
            ' NPRINT = ',NPRINT,'   NSTEP  = ',NSTEP, &
            ' PSTRCT = ',PSTRCT,'   SDSTP  = ',SDSTP, &
            ' STPLIM = ',STPLIM,'   STRICT = ',STRICT, &
            ' TOLFUN = ',TOLFUN,'   TOLGRD = ',TOLGRD, &
            ' TOLITR = ',TOLITR,'   TOLSTP = ',TOLSTP, &
            ' FMEM   = ',FMEM
112    FORMAT(/,A,F12.7,A,I12,/,A,I12,A,I12,/, &
            3(A,F12.7,A,F12.7,/),A,I12,A,F12.7,/,A,F12.7)
    ENDIF
    !
    !     Allocate storage.
    !
    !
    !---NEEDS-WORK---------------------------------------------
    !   these need to be chmalloc and 
    !    need to be checked for 2 dimensional arrays

    ALLOCATE(EVAL(MINDIM),EVEC(MINDIM*MINDIM),FUNC(MINDIM+2))
    ALLOCATE(GNRM(MINDIM+2),GRAD(NVAR*(MINDIM+2)),LPREV(MINDIM))
    ALLOCATE(PARM(NVAR*(MINDIM+2)),SECDER(MINDIM*MINDIM))
    ALLOCATE(SECDER1(2*MINDIM*MINDIM),VARB(NVAR),VRED(MINDIM))
    ALLOCATE(VRED1(2*MINDIM),VRSDL(NVAR),VWRK(NVAR))
    ALLOCATE(WRK1(NVAR*MINDIM),WRK2(NVAR*MINDIM),WRK3(MINDIM))
    ALLOCATE(XRED(MINDIM),PFUNC(MINDIM+2),PSGRAD(NVAR*(MINDIM+2)))
    ALLOCATE(PGRAD(NVAR),PTHTAN(NVAR),PTHTAN1(NVAR),VWRK1(NVAR))
    ALLOCATE(WRK4(MINDIM*NVAR),XREDO(MINDIM),XREDDF(MINDIM))
    ALLOCATE(USVD(2*MINDIM*MINDIM),VSVD(2*MINDIM*MINDIM))
    ALLOCATE(PGNRM(MINDIM+2))
    FUNC = ZERO
    XRED = ZERO
    !
    !     Fill the variable array with the parameters to be optimised.
    !
    CALL GETVR1(MINXYZ,NATOM,VARB,IMOVE,X,Y,Z,LMINUC, &
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
        ,QFHDGB,SDEF,TOTALS &
#endif
         )
    call abner2(varb,nvar,eigrng,mindim,nstep,pstrct, &
         sdstp,stplim,strict,tolfun,tolgrd,tolitr,tolstp,fmem, &
         eval,evec,func,gnrm,grad,lprev,parm,secder, &
         wrk1,wrk2,wrk3,vred,vrsdl,vwrk,xred,convrg, &
         qdebug,ncalls,pgrad,psgrad,pfunc,pthtan, &
         vwrk1,pthtan1,wrk4,xredo,xreddf,vsvd,usvd,secder1, &
         vred1,pgnrm &
#if KEY_DHDGB==1
!AP/MF
         ,QFHDGB &
#endif
         )
    !
    !     Fill the coordinate arrays with the optimised variables.
    !
    CALL PUTVR1(MINXYZ,NATOM,VARB,IMOVE,X,Y,Z, &
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
       IF (CONVRG  ==  0) THEN
          !RCZ 91/12/02 : printout for number of steps limits condition added
          WRITE (OUTU,'(/,A,I5,A,/)') &
               ' ABNER> Minimization exiting with number of steps limit (', &
               NSTEP,') exceeded.'
       ELSE IF (CONVRG  ==  1) THEN
          WRITE (OUTU,'(/,A,F10.7,A,/)') &
               ' ABNER> Minimization exiting with gradient tolerance (', &
               TOLGRD,') satisfied.'
       ELSE IF (CONVRG  ==  2) THEN
          WRITE (OUTU,'(/,A,F10.7,A,/)') &
               ' ABNER> Minimization exiting with step tolerance (', &
               TOLSTP,') satisfied.'
       ELSE IF (CONVRG  ==  3) THEN
          WRITE (OUTU,'(/,A,F10.7,A,/)') &
               ' ABNER> Minimization exiting with function tolerance (', &
               TOLFUN,') satisfied.'
       ELSE IF (CONVRG  ==  4) THEN
          WRITE (OUTU,'(/,A,I5,A,/)') &
               ' ABNER> Minimization exiting with iteration limit (', &
               TOLITR,') exceeded.'
       ELSE IF (CONVRG  ==  5) THEN
          WRITE (OUTU,'(/,A,A,G12.4,/)') &
               ' ABNER> Minimization exiting with step-size underflow. ', &
               'Last step was ',SDSTP
       ELSE IF (CONVRG  ==  6) THEN
          WRITE (OUTU,'(/,A,F10.7,A,/)') &
               ' ABNER> Minimization exiting with variable tolerance (', &
               PSTRCT,') satisfied.'
       ELSE IF (CONVRG  ==  -1) THEN
          WRITE (OUTU,'(/,A,A,/)') &
               ' ABNER> Minimization exiting due time limit or Quanta', &
               ' interrupt.'
       ELSE
          WRITE (OUTU,'(/,A,I5,A,/)') &
               ' ABNER> Unknown convergence status (',CONVRG,').'
       ENDIF
       !
       !     Save the last step characteristics and exit printing
       !
       !     EPROP(PJNK1)=NCALLS
       !     EPROP(PJNK2)=SDSTP
       !     when minimization exits because of the number of steps limit reached,
       !     it updates the coordinates AFTER the last step, so the final energy
       !     may be different. So, let's calculate it
       !RCZ     CALL GETE(X, Y, Z, X, Y, Z, 0)
       ! -- mikem ->
    ENDIF
#if KEY_REPDSTR==1
    IF(QREPDSTR) THEN
       CALL PRINTE(OUTU, EPROP, ETERM, 'ABNR', 'MIN', &
            .TRUE., NCALLS, ZERO, SDSTP, .FALSE.)
    ELSE
#endif 
       CALL PRINTE(OUTU, EPROP, ETERM, 'ABNR', 'MIN', &
            .TRUE., NCALLS, ZERO, SDSTP, .TRUE.)
#if KEY_REPDSTR==1
    ENDIF                                       
#endif
    !
    !     Clear up any temporary storage space.
    deallocate(eval,evec,func,gnrm,lprev,secder,secder1)
    deallocate(vred,vred1,vrsdl,vwrk,wrk3,xred,pfunc,vwrk1)
    deallocate(xredo,xreddf,usvd,vsvd,pgnrm)
    !     end stack
#if KEY_CHEQ==1
    if(allocated(cgfix))deallocate(cgfix,cgtmp)    
#endif
    deallocate(grad,parm,varb,wrk1,wrk2,psgrad,pgrad)
    deallocate(pthtan,pthtan1,wrk4)
    !
    !     Reset control variables.
    !
    IF(LMINUC)  CALL XTLMSR(XUCELL)
    LMINUC = QLOCAL
    RETURN
  END SUBROUTINE ABNER

  SUBROUTINE ABNER2(VARB,NP,EIGRNG,MINDIM,NSTEP,PSTRCT,SDSTP, &
       STPLIM,STRICT,TOLFUN,TOLGRD,TOLITR,TOLSTP,FMEM, &
       EVAL,EVEC,FUNC,GNRM,GRAD,LPREV,PARM,SECDER, &
       WRK1,WRK2,WRK3,VRED,VRSDL,VWRK,XRED,CONVRG, &
       QDEBUG,NCALLS, &
       PGRAD,PSGRAD,PFUNC,PTHTAN,VWRK1,PTHTAN1,WRK4, &
       XREDO,XREDDF,VSVD,USVD,SECDER1,VRED1,PGNRM &
#if KEY_DHDGB==1
!AP/MF
      ,QFHDGB &
#endif
      )
    !-----------------------------------------------------------------------
    !     This subroutine is a generic version of the CHARMM energy
    !     minimization ABNR algorithm. It minimizes a function FUNC
    !     with respect to the NP variables PARM. The gradients are stored
    !     in GRAD.
    !
    !     Martin J. Field from the original routine by David J. States.
    !
    !     The routine minimizes a function by applying a Newton-Raphson
    !     algorithm to a subspace of the complete variable space. The
    !     subspace is determined from variable difference vectors obtained
    !     from the last MINDIM iterations. The second derivative matrix is
    !     constructed numerically from the change in the gradient vectors
    !     and is inverted by an eigenvector analysis allowing the routine
    !     to recognise and avoid saddle points on the functions surface.
    !     At each step the residual gradient vector is calculated and used
    !     to add a steepest descent step onto the Newton-Raphson step
    !     thereby incorporating a new direction into the basis set.
    !
    !     The input parameters are (with defaults):
    !
    !     EIGRNG   0.0005   The smallest eigenvalue that will be considered
    !     non-singular.
    !
    !     MINDIM   5        The dimension of the reduced basis set. It must
    !     not exceed the number of variables.
    !
    !     NSTEP    100      The number of steps of minimization.
    !
    !     PSTRCT   0.001    The maximum change in the variables must be
    !     greater than this otherwise the optimisation
    !     stops.
    !
    !     FMEM     0.       Memory factor to calculate GRADAV and STEPAV
    !                       0 <= FMEM <= ONE
    !                       0 - means no memory; 1 - infinitely long memory
    !
    !     QDEBUG   .FALSE.  A flag indicating whether debug printing is done.
    !
    !     SDSTP    0.02     The initial size of the steepest descent step.
    !     It is altered during the run.
    !
    !     STPLIM   1.0      The maximum allowed Newton-Raphson step.
    !
    !     STRICT   0.1      Strictness of descent. The new function-value must
    !     be less than or equal to the old function-value
    !     + strict for a step to be accepted.
    !
    !     TOLITR   100      Quit if this number of function evaluations is
    !     exceeded in any one cycle of the minimization.
    !
    !     The following convergence tests are applied:
    !
    !     TOLFUN   0.0001   Quit if the function-value does not drop by this
    !     amount.
    !     TOLGRD   0.0001   Quit if the average gradient is smaller than this.
    !     TOLSTP   0.0001   Quit if the average step-size is smaller than this.
    !
    !     Storage requirements :
    !
    !     EVAL                Mindim                   real(chm_real)
    !     EVEC                Mindim * mindim          real(chm_real)
    !     FUNC                Mindim + 2               real(chm_real)
    !     GNRM                Mindim + 2               real(chm_real)
    !     GRAD                Np * (mindim + 2)        real(chm_real)
    !     LPREV               Mindim                   Integer*4
    !     PARM                Np * (mindim + 2)        real(chm_real)
    !     SECDER              Mindim * mindim          real(chm_real)
    !     VARB                Np                       real(chm_real)
    !     VRED                Mindim                   real(chm_real)
    !     VRSDL,VWRK          Np                       real(chm_real)
    !     WRK1,WRK2           Np * mindim              real(chm_real)
    !     WRK3                Mindim                   real(chm_real)
    !     XRED                Mindim                   real(chm_real)
    !
    !     Storage is managed internally with a linked list that points to
    !     the gradient, variable, function and gradient-norm arrays. LNEW is
    !     the new set, LBEST is the current best and LPREV() is a list of the
    !     previous best sets. This avoids unnecessary shuffling of the stored
    !     variable and gradient vectors as new points are placed in the list.
    !
#if KEY_REPDSTR==1
  use repdstr,only:psetloc,psetglob              
#endif

  use chm_kinds
  use dimens_fcm    ! jwchuneb
  use number
  use vector
  use stream
  use timerm
  use egrad
  use psf       ! jwchuneb
  use pathm      ! jwchuneb
  use minmiz_util_module, only: mintrj
  use replica_mod   ! jwchuneb
#if KEY_RPATH==1
  use epathmod,only: PJDX,PJDY,PJDZ,PSDX,PSDY,PSDZ,PTANX,PTANY,PTANZ 
#endif
  use neb       ! jwchuneb
  use energym    ! jwchuneb
#if KEY_DHDGB==1
!AP/MF
  use contrl
#endif
  use parallel
  use repdstr
    implicit none

    !
#if KEY_DHDGB==1
!AP/MF
    LOGICAL QFHDGB
#endif
    INTEGER  NP, CONVRG, LPREV(*), MINDIM, NSTEP, TOLITR
    LOGICAL  QDEBUG
    real(chm_real)   EIGRNG, PSTRCT, STPLIM, STRICT, TOLFUN,  &
         TOLGRD, TOLSTP
    real(chm_real)   EVAL(*), EVEC(*), FUNC(*), GNRM(*), GRAD(NP,*), &
         PARM(NP,*), SDSTP, SECDER(*), WRK1(NP,*), WRK2(NP,*), &
         WRK3(*), VARB(*), VRED(*), VRSDL(*), VWRK(*), XRED(*), &
         FMEM
    !
    CHARACTER(len=4) STPTYP
    INTEGER  DIMSUB, I, IRPLC, J, LBEST, LNEW, LTEMP, ERSTAT, &
         NCALLS, NEVAL, NEVAL0, NRPLC, NTRIES
    LOGICAL  DONE, NEWSTP, RESET, SDFLAG, PFLAG
    real(chm_real)   CGSCL, EIGMAX, EIGVAL, FDIFF, GRADAV, PDIFF,  &
         GRADAV2, &
         PDMAX, RSDLEN, RSDSCL, RSDSLP, SDSCL, SQRTNP, STEPAV, &
         STPLEN, STPMIN, TEMP, NPS, STPLEN2
    !
    ! NEB variables
    real(chm_real) PGRAD(*), PSGRAD(NP,*), PFUNC(*), PTHTAN(*),  &
         VWRK1(*)
    real(chm_real) PTHTAN1(*), WRK4(NP,*), XREDO(*), XREDDF(*)
    real(chm_real) VSVD(2*MINDIM,MINDIM), USVD(2*MINDIM,MINDIM),  &
         PGNRM(*)
    real(chm_real) GPGRAD, GPSGRAD, ERMS, ERMSO, PSTEP, STPLEN1, XDIF
    real(chm_real) test, test1, S_TAN, S_OTH, S1_TAN, S1_OTH, GN,  &
         TEMP1
    real(chm_real) SDSCL1, NTMV, RSDLEN1, RSDSCL1, WMAX, WMIN, RSDSLP1
    real(chm_real) FDOTDX, FDOTDX1
    real(chm_real) SECDER1(2*MINDIM,MINDIM),VRED1(*),FDIFF1,FDIFF2
    INTEGER IPT,ISCF
    real(chm_real) WORK(2000)
    REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:),SAVE :: WEIGHT,PDMAXS
    !
    !     Index is an arithmetic statement function for addressing the
    !     reduced second derivative matrix that is fully stored.
    !
    INTEGER INDEX
    integer prtest




    INDEX(I,J) = I + (J - 1) * DIMSUB
    !
    !     ABNER SHOULD NOT BE USED WITH CI-NEB
    !
#if KEY_RPATH==1
#if KEY_REPLICA==1
    IF(QPNEB .AND. QPCIMG) THEN
       WRITE(OUTU,'(A)') &
            ' ABNER2> WARNING CI-NEB SHOULD NOT BE USED WITH ABNER'
    ENDIF
#endif 
#endif 
    !
    !     Do some initialisation.
    !
    IF (MINDIM  >  NP) MINDIM = NP
    !
    !     Initialize local variables
    CGSCL  = ZERO
    RSDSCL = ZERO
    DONE   = .FALSE.
    CONVRG = 0
    DIMSUB = 0
    LBEST  = MINDIM + 1
    LNEW   = MINDIM + 2
    DO I = 1,MINDIM
       LPREV(I) = I
    enddo
    NEVAL  = 1
    NEVAL0 = NEVAL
    NEWSTP = .FALSE.
    NTRIES = 0
    RESET  = .TRUE.
    RSDLEN = ZERO
    SQRTNP = NP
    SQRTNP = SQRT(SQRTNP)
    NCALLS = 0
    STPMIN = ZERO
    !
    PARM(1:np,1:mindim+2) = ZERO

    PARM(1:np,LBEST) = VARB(1:np)

    !
#if KEY_PARALLEL==1
    prtest=plnod0
#else /**/
    prtest=prnlev
#endif 
    !write(50+mynodg,'(a,2i5,3l6)')'ABNR-x-0>mindim,dimsub,reset,sdflag,newstp=', &
    !     mindim,dimsub,reset,sdflag,newstp

    IF (SDSTP  <  10.0 * TOLSTP .AND. prtest >= 2) THEN
       if(prnlev >= 2) &
            WRITE (OUTU,'(A,A,G11.4,A,G11.4)') &
            ' ABNER2> The SD step is being increased to 10.0 * TOLSTP.', &
            ' The old SD step = ',SDSTP,' and TOLSTP = ',TOLSTP
       SDSTP = 10.0 * TOLSTP
    ENDIF
    !
    STPTYP = 'INIT'
    CALL EGRAD1(NP,PARM(1,LBEST),PARM(1,LBEST),FUNC(LBEST), &
         GRAD(1,LBEST),NCALLS,0,ERSTAT)
    CALL ENEOUT(OUTU,0,ZERO,ZERO)
#if KEY_DHDGB==1
!AP/MF
         IF (QFHDGB) THEN
            IF (NPRINT .GT. 0 .AND. NCALLS .GE. 0) THEN
               IF (MOD(NCALLS,NPRINT) .EQ. 0) THEN
                  WRITE(121,131)NCALLS,PARM(NP-10+1,LBEST), &
                  PARM(NP-10+2,LBEST), &
                  PARM(NP-10+3,LBEST),PARM(NP-10+4,LBEST), &
                  PARM(NP-10+5,LBEST), &
                  PARM(NP-10+6,LBEST),PARM(NP-10+7,LBEST), &
                  PARM(NP-10+8,LBEST), &
                  PARM(NP-10+9,LBEST),PARM(NP,LBEST)
 131  FORMAT(I5,10(F7.3))
               ENDIF
            ENDIF
         ENDIF
#endif
    !
    !     REPDSTR note:
    !..................
    !     NOT sure about this???
    !     We dont want to sum individual energies etc in energy.src
    !     So we do FUNC only here:
#if KEY_REPDSTR==1
#if KEY_REPLICA==1
#if KEY_RPATH==1
    IF(QREPDSTR.AND.QPATH)THEN
       !write(50+mynodg,'(a,i5,f20.10)')'abnr>lbest,func(lbest)=',lbest,func(lbest)
       if(mynod > 0) func(lbest)=zero   ! no double counting
       CALL PSETGLOB
       CALL GCOMB(FUNC(LBEST),1)
       CALL PSETLOC
       !write(50+mynodg,'(a,i5,f20.10)')'abnr>lbest,func(lbest)=',lbest,func(lbest)
    ENDIF
#endif 
#endif 
#endif 
    !
    !write(50+mynodg,'(a,2i5,3l6)')'ABNR-x-1>mindim,dimsub,reset,sdflag,newstp=', &
    !     mindim,dimsub,reset,sdflag,newstp

#if KEY_RPATH==1
#if KEY_REPLICA==1
    IF(QPNEB) THEN ! jwchu
       CALL EGRADNEB(NP,PARM(1,LBEST),PARM(1,LBEST), &
            GRAD(1,LBEST),PGRAD,PSGRAD(1,LBEST),ERSTAT,PTHTAN)
       vrsdl(1:np) = psgrad(1:np,lbest)
       CALL TFORCE(QPCYCL,VRSDL,PTHTAN,NP,NREPL,NPATOMS,IREPS, &
            PJDX,PJDY,PJDZ,PTANX,PTANY,PTANZ,QPCIMG,ICLREP)
       ERMS=ETERM(PRMS)
       FUNC(LBEST)=FUNC(LBEST)-ERMS
       PFUNC(LBEST)=ERMS
    ENDIF    ! QPNEB
#endif 
#endif 

    !RCZ 91/11/20 GRADAV initialization moved here
    GRADAV = LENVEC(GRAD(1,LBEST),NP)
    ! Start modifications for distributed replica
    PFLAG=.TRUE.
    NPS=NP
#if KEY_REPDSTR==1
#if KEY_RPATH==1
#if KEY_REPLICA==1
    IF(QREPDSTR.AND.QPATH) THEN
       PFLAG=.FALSE.
       if(mynod > 0) gradav = zero
       !write(50+mynodg,'(a,2i4,2f20.10)')'ABNER-0a>me,np,gradav,nps=',mynod,np,gradav,nps
       CALL PSETGLOB
       !     We can only sum square
       GRADAV2=GRADAV*GRADAV
       NPS=NP*NREPDSTR
       CALL GCOMB(GRADAV2,1)
       NPS=SQRT(NPS)
       GRADAV=SQRT(GRADAV2)
       CALL PSETLOC
    ENDIF
    !write(50+mynodg,'(a,2i4,2f20.10)')'ABNER-0b>me,np,gradav,nps=',mynod,np,gradav,nps
    !write(50+mynodg,*)'ABNER-0b>gradav,np,nps=',gradav,np,nps
#endif 
#endif 
#endif 
    IF(PFLAG)THEN
       GRADAV = GRADAV/SIGN(MAX(ABS(SQRTNP),RSMALL),SQRTNP)
    ELSE
       GRADAV = GRADAV/SIGN(MAX(ABS(NPS),RSMALL),NPS)
    ENDIF
    !write(50+mynodg,*)'ABNER-0c>gradav,np,nps=',gradav,np,nps
    ! End REPDSTR
    GNRM(LBEST) = GRADAV
#if KEY_REPLICA==1
#if KEY_RPATH==1
    IF(QPNEB) THEN
       GNRM(LBEST) = LENVEC(PGRAD,NP) / &
            SIGN(MAX(ABS(SQRTNP),RSMALL),SQRTNP)
       PGNRM(LBEST) = LENVEC(PSGRAD(1,LBEST),NP) / &
            SIGN(MAX(ABS(SQRTNP),RSMALL),SQRTNP)
       IF(QDEBUG .OR. (PRNLEV >= 5)) THEN
          write(outu,'(A,1E10.5,A,1E10.5)') &
               ' OFF-PATH GRAD = ',  GNRM(LBEST), &
               ' TANGENT GRAD = ', PGNRM(LBEST)
       ENDIF
    ENDIF
#endif 
#endif 

    STEPAV = GRADAV
    !
1000 CONTINUE
    !
    !write(50+mynodg,*)'ABNR-x>entering the loop: ncalls=',ncalls
    !write(50+mynodg,'(a,2i5,3l6)')'ABNR-x>mindim,dimsub,reset,sdflag,newstp=', &
    !     mindim,dimsub,reset,sdflag,newstp
    !write(50+mynodg,'(a,10i5)')'ABNR-x>lbest,lnew,lprev=',lbest,lnew,(lprev(i),i=1,dimsub)
    !
    IF (RESET) THEN
       DIMSUB = 0
       IRPLC  = 0
       NRPLC  = 0
       RESET  = .FALSE.
       SDFLAG = .TRUE.
       GOTO 100
    ENDIF
    !
    !     Rearrange pointers.
    !
#if KEY_REPLICA==1 && KEY_RPATH==1
    IF(.NOT.QPNEB) THEN   
#endif
       IF (NEWSTP) THEN
          IF (DIMSUB  <  MINDIM) DIMSUB = DIMSUB + 1
          LTEMP = LPREV(DIMSUB)
          DO I = (DIMSUB-1),1,-1
             LPREV(I + 1) = LPREV(I)
          enddo
          LPREV(1) = LBEST
          LBEST    = LNEW
          LNEW     = LTEMP
          NRPLC    = 0
          SDFLAG   = .FALSE.
       ELSE !NEWSTP
          IRPLC = MINDIM + 1
          DO I = DIMSUB,1,-1
             IF (FUNC(LNEW)  <  FUNC(LPREV(I))) IRPLC = I
          enddo
          IF (DIMSUB  <  MINDIM) THEN
             IF (IRPLC  >  MINDIM) IRPLC = DIMSUB + 1
             DIMSUB = DIMSUB + 1
             NRPLC  = 0
             SDFLAG = .FALSE.
          ENDIF
          !
          NRPLC = NRPLC + 1
          IF (NRPLC  >  1) SDFLAG = .TRUE.
          IF (IRPLC  <=  MINDIM) THEN
             LTEMP = LPREV(DIMSUB)
             DO I = (DIMSUB-1),IRPLC,-1
                LPREV(I + 1) = LPREV(I)
             enddo
             LPREV(IRPLC) = LNEW
             LNEW = LTEMP
          ELSE
             IF (.NOT. SDFLAG) DIMSUB = DIMSUB / 2
             SDFLAG = .TRUE.
          ENDIF
       ENDIF !NEWSTP
#if KEY_RPATH==1
#if KEY_REPLICA==1
    ELSE !QPNEB
       IF (NEWSTP) THEN
          IF (DIMSUB  <  MINDIM) DIMSUB = DIMSUB + 1
          LTEMP = LPREV(DIMSUB)
          DO I = (DIMSUB-1),1,-1
             LPREV(I + 1) = LPREV(I)
          ENDDO
          LPREV(1) = LBEST
          LBEST    = LNEW
          LNEW     = LTEMP
          NRPLC    = 0
          SDFLAG   = .FALSE.
       ELSE !NEWSTP
          IRPLC = MINDIM + 1
          DO I = DIMSUB,1,-1
             IF ((FUNC(LNEW)+PFUNC(LNEW)) <  &
                  (FUNC(LPREV(I))+PFUNC(LPREV(I)))) IRPLC = I
          ENDDO
          IF (DIMSUB  <  MINDIM) THEN
             IF (IRPLC  >  MINDIM) IRPLC = DIMSUB + 1
             DIMSUB = DIMSUB + 1
             NRPLC  = 0
             SDFLAG = .FALSE.
          ENDIF
          !
          NRPLC = NRPLC + 1
          IF (NRPLC  >  1) SDFLAG = .TRUE.
          IF (IRPLC  <=  MINDIM) THEN
             LTEMP = LPREV(DIMSUB)
             DO I = (DIMSUB-1),IRPLC,-1
                LPREV(I + 1) = LPREV(I)
             ENDDO
             LPREV(IRPLC) = LNEW
             LNEW = LTEMP
          ELSE
             IF (.NOT. SDFLAG) DIMSUB = DIMSUB / 2
             SDFLAG = .TRUE.
          ENDIF
       ENDIF !NEWSTP
    ENDIF !QPNEB
#endif 
#endif 
    !
    !     Determine the new move for the NR step. Calculate the reduced
    !     derivative and second derivative matrix. Diagonalise the matrix
    !     and find the reduced move vector. The eigenvectors are normalised
    !     automatically by EIGRS.
    !
    IF (.NOT. SDFLAG) THEN
#if KEY_REPLICA==1 && KEY_RPATH==1
       IF(.NOT.QPNEB) THEN      
#endif

          DO I = 1,DIMSUB
             CALL SUBVEC(PARM(1,LPREV(I)),PARM(1,LBEST),WRK2(1,I),NP)
             CALL SUBVEC(GRAD(1,LPREV(I)),GRAD(1,LBEST),WRK1(1,I),NP)
          enddo
          
          !write(50+mynodg,*)'ABNER-1>ncalls,lprev1,lbest=',ncalls,lprev(1),lbest
          !write(50+mynodg,*)'ABNER-1a>parm(curr)='
          !write(50+mynodg,'(9f10.5)')(parm(i,lprev(1)),i=1,np)
          !write(50+mynodg,*)'ABNER-1b>parm(best)='
          !write(50+mynodg,'(9f10.5)')(parm(i,lbest),i=1,np)
          !write(50+mynodg,*)'ABNER-1a>G(curr)='
          !write(50+mynodg,'(9f10.5)')(grad(i,lprev(1)),i=1,np)
          !write(50+mynodg,*)'ABNER-1b>G(best)='
          !write(50+mynodg,'(9f10.5)')(grad(i,lbest),i=1,np)
          !write(50+mynodg,*)'ABNER-1a>wrk1='
          !write(50+mynodg,'(9f10.5)')(wrk1(i,1),i=1,np)
          !write(50+mynodg,*)'ABNER-1b>wrk2='
          !write(50+mynodg,'(9f10.5)')(wrk2(i,1),i=1,np)
          !
          DO I = 1,DIMSUB
             VRED(I) = DOTVEC(WRK2(1,I),GRAD(1,LBEST),NP)
             SECDER(INDEX(I,I)) = DOTVEC(WRK1(1,I),WRK2(1,I),NP)
          enddo
          DO I = 1,DIMSUB
             DO J = (I+1),DIMSUB
                TEMP = HALF *( DOTVEC(WRK1(1,I),WRK2(1,J),NP) &
                     + DOTVEC(WRK1(1,J),WRK2(1,I),NP) )
                SECDER(INDEX(I,J)) = TEMP
                SECDER(INDEX(J,I)) = TEMP
             enddo
          enddo
          !
#if KEY_REPDSTR==1
#if KEY_REPLICA==1
#if KEY_RPATH==1
          !write(50+mynodg,*)'ABNER-1c>vred=',(vred(i),i=1,dimsub)
          !write(50+mynodg,*)'ABNER-1d>secder=',(secder(i),i=1,dimsub*dimsub)
          IF(QREPDSTR.AND.QPATH) THEN
             if(mynod > 0) secder(1:dimsub*dimsub) = zero
             if(mynod > 0) vred(1:dimsub) = zero
             CALL PSETGLOB
             CALL GCOMB(VRED,DIMSUB)
             CALL GCOMB(SECDER,DIMSUB*DIMSUB)
             CALL PSETLOC
          ENDIF
          !write(50+mynodg,*)'ABNER-1c-a>vred=',(vred(i),i=1,dimsub)
          !write(50+mynodg,*)'ABNER-1d-a>secder=',(secder(i),i=1,dimsub*dimsub)
#endif 
#endif 
#endif 

          I = 0
          CALL EIGRS(SECDER,DIMSUB,11,EVAL,EVEC,DIMSUB,I)
          IF (I /= 0 .AND. WRNLEV >= 2) THEN
             WRITE (OUTU,'(/,A,I5,/)') &
                  ' ABNER2> Possible error in diagonalisation. IER = ',I
             CALL DIEWRN(0)
          ENDIF
          !
          XRED(1:DIMSUB)=zero
          !
          IF (QDEBUG) WRITE (OUTU,'(A)') &
               ' Eigen-value   Gradient    Move'
          !
          EIGMAX = EVAL(DIMSUB)
          IF (EIGMAX  <=  ZERO) THEN
             SDFLAG = .TRUE.
             GOTO 100
          ENDIF
          !
          DO I = 1,DIMSUB
             EIGVAL = EVAL(I)
             TEMP = SIGN(MAX(ABS(EIGMAX),RSMALL),EIGMAX)
             IF (EIGVAL / TEMP  >  EIGRNG) THEN
                TEMP = DOTVEC(VRED,EVEC(INDEX(1,I)),DIMSUB)
                DO J = 1,DIMSUB
                   XRED(J) = XRED(J) + (EVEC(INDEX(J,I)) * TEMP / &
                        SIGN(MAX(ABS(EIGVAL),RSMALL),EIGVAL))
                enddo
                IF (QDEBUG) WRITE (OUTU,'(3F12.5)') &
                     EIGVAL, TEMP, TEMP/SIGN(MAX(ABS(EIGVAL),RSMALL),EIGVAL)
             ENDIF
          enddo
          !
          !       Expand move vector to the full parameter space.
          !

          VRSDL(1:np) = GRAD(1:np,LBEST)

          DO J = 1,DIMSUB
             DO I = 1,NP
                VRSDL(I) = VRSDL(I) - XRED(J) * WRK1(I,J)
             enddo
          enddo
          !
          RSDLEN = LENVEC(VRSDL,NP)
          !write(50+mynodg,*)'ABNER-3a>rsdlen,np=',rsdlen,np
          ! Start modifications for distributed replica
#if KEY_REPDSTR==1
#if KEY_RPATH==1
#if KEY_REPLICA==1
          IF(QREPDSTR.AND.QPATH)THEN
             if(mynod > 0) rsdlen = zero
             CALL PSETGLOB
             GRADAV2=RSDLEN*RSDLEN
             CALL GCOMB(GRADAV2,1)
             RSDLEN=SQRT(GRADAV2)
             CALL PSETLOC
          ENDIF
#endif 
#endif 
#endif 
          !write(50+mynodg,*)'ABNER-3b>rsdlen,np=',rsdlen,np
          ! End repdstr
          RSDSCL = SDSTP / SIGN(MAX(ABS(RSDLEN),RSMALL),RSDLEN)
          !write(50+mynodg,*)'ABNER-3c>rsdlen,rsdscl=',rsdlen,rsdscl

             VWRK(1:np) = RSDSCL * VRSDL(1:np)

          DO J = 1,DIMSUB
             TEMP = XRED(J)
             VWRK(1:np) = VWRK(1:np) + TEMP * WRK2(1:np,J)
          enddo

#if KEY_RPATH==1
#if KEY_REPLICA==1
       ELSE ! QPNEB

          pthtan1(1:np) = pthtan(1:np)
          VWRK(1:NP)=zero
          VWRK1(1:NP)=zero
          XREDO(1:MINDIM)=zero
          ISCF=0

333       CONTINUE

          DO I = 1,DIMSUB
             CALL SUBVEC(PARM(1,LPREV(I)),PARM(1,LBEST),WRK2(1,I),NP)
             CALL SUBVEC(GRAD(1,LPREV(I)),GRAD(1,LBEST),WRK1(1,I),NP)
             CALL SUBVEC(PSGRAD(1,LPREV(I)),PSGRAD(1,LBEST),WRK4(1,I),NP)
          ENDDO

          DO I=1,DIMSUB
             CALL PFORCE(QPCYCL,WRK1(1,I),PTHTAN1,NP,NREPL,NPATOMS,IREPS, &
                  PJDX,PJDY,PJDZ,PTANX,PTANY,PTANZ,QPCIMG,ICLREP)
             CALL TFORCE(QPCYCL,WRK4(1,I),PTHTAN1,NP,NREPL,NPATOMS,IREPS, &
                  PJDX,PJDY,PJDZ,PTANX,PTANY,PTANZ,QPCIMG,ICLREP)
          ENDDO

          pgrad(1:np) = GRAD(1:np,LBEST)
          vrsdl(1:np) = PSGRAD(1:np,LBEST)
          CALL PFORCE(QPCYCL,PGRAD,PTHTAN1,NP,NREPL,NPATOMS,IREPS, &
               PJDX,PJDY,PJDZ,PTANX,PTANY,PTANZ,QPCIMG,ICLREP)
          CALL TFORCE(QPCYCL,VRSDL,PTHTAN1,NP,NREPL,NPATOMS,IREPS, &
               PJDX,PJDY,PJDZ,PTANX,PTANY,PTANZ,QPCIMG,ICLREP)
          !
          DO I = 1,DIMSUB
             VRED1(I) = DOTVEC(WRK2(1,I),PGRAD,NP)
             VRED1(I+DIMSUB) = DOTVEC(WRK2(1,I),VRSDL,NP)
             SECDER1(I,I) = DOTVEC(WRK1(1,I),WRK2(1,I),NP)
             SECDER1(I+DIMSUB,I)= DOTVEC(WRK4(1,I),WRK2(1,I),NP)
          ENDDO
          DO I = 1,DIMSUB
             DO J = (I+1),DIMSUB
                TEMP =  DOTVEC(WRK1(1,I),WRK2(1,J),NP)
                TEMP1=  DOTVEC(WRK1(1,J),WRK2(1,I),NP)
                SECDER1(I,J) = TEMP
                SECDER1(J,I) = TEMP1
             ENDDO
          ENDDO

          DO I = 1,DIMSUB
             DO J = (I+1),DIMSUB
                TEMP =  DOTVEC(WRK4(1,I),WRK2(1,J),NP)
                TEMP1=  DOTVEC(WRK4(1,J),WRK2(1,I),NP)
                SECDER1(I+DIMSUB,J) = TEMP
                SECDER1(J+DIMSUB,I) = TEMP1
             ENDDO
          ENDDO
          !
          J=0

          WRK3(1:DIMSUB) = zero

          !c      SUBROUTINE SVD (NM, M, N, A, W, MATU, U, MATV, V, IERR, RV1)

          CALL SVD(2*MINDIM,2*DIMSUB,DIMSUB,SECDER1,WRK3,.TRUE.,USVD, &
               .TRUE.,VSVD,J,XRED) 
          XRED(1:DIMSUB)=zero

          IF (QDEBUG) THEN
             WRITE (OUTU,'(A,I5)') 'SVD EIGENVALUES',J
             DO I=1,DIMSUB
                WRITE(OUTU,'(1E10.5)') WRK3(I)
             ENDDO
          ENDIF

          WMAX=ZERO

          DO I=1,DIMSUB
             IF(WRK3(I) > WMAX) WMAX=WRK3(I)
          ENDDO

          WMIN=WMAX*0.0005

          DO I=1,DIMSUB
             IF(WRK3(I) < WMIN) WRK3(I)=ZERO
          ENDDO

          IF (QDEBUG) THEN
             WRITE (OUTU,'(A)') 'SVD EIGENVALUES'
             DO I=1,DIMSUB
                WRITE(OUTU,'(1E10.5)') WRK3(I)
             ENDDO
          ENDIF

          CALL SVDBSB(USVD,WRK3,VSVD,2*DIMSUB,DIMSUB, &
               2*MINDIM,MINDIM,VRED1,XRED)
          !
          IF (QDEBUG) WRITE (OUTU,'(A)') &
               ' XRED '
          IF (QDEBUG) THEN
             DO I=1,DIMSUB
                WRITE(OUTU,'(1F10.5)') XRED(I)
             ENDDO
          ENDIF
          !
          CALL SUBVEC(XRED,XREDO,XREDDF,MINDIM)
          XDIF=SQRT(DOTVEC(XREDDF,XREDDF,MINDIM)/MINDIM)

          VWRK1(1:NP)=zero

          DO J=1,DIMSUB
             TEMP=XRED(J)
             VWRK1(1:np)=VWRK1(1:np)+TEMP*WRK2(1:np,J)
          ENDDO
          VWRK(1:np)=PARM(1:np,LBEST)-VWRK1(1:np)

          ALLOCATE(WEIGHT(NPATOMS))
          CALL PTANG(VWRK,PTHTAN1,NP,NREPL,NPATOMS,IREPS,EREPS, &
               WEIGHT,PTANX,PTANY,PTANZ)
          ISCF=ISCF+1
          DEALLOCATE(WEIGHT)

          IF(QDEBUG) THEN
             write(outu,'(A)')'---NEB SCF ITERATIONS------'
             write(outu,'(A,1F10.5,A,1F10.5)')' ISCF= ',iscf,' XDIF= ',xdif
             do i=1,dimsub
                write(outu,'(I5,3F10.5)') i,xredo(i),xred(i),xreddf(i)
             enddo
          ENDIF

          IF(XDIF > 0.00005) THEN
             IF(ISCF < 50) THEN
                XREDO(1:mindim) = xred(1:MINDIM)
                GOTO 333
             ENDIF
             NTMV=sqrt(dotvec(xred,xred,dimsub)/dimsub)
             IF(NTMV > 10.0) THEN
                SDFLAG=.TRUE.
                IF(NTMV > 100.0) THEN
                   DIMSUB=0
                ENDIF
                GOTO 100
             ENDIF
             IF(QDEBUG) WRITE(OUTU,'(A,1F10.5)')'COEFF VECTOR SIZE = ',NTMV
          ENDIF

          NTMV=sqrt(dotvec(xred,xred,dimsub)/dimsub)
          IF(NTMV > 10.0) THEN
             SDFLAG=.TRUE.
             IF(NTMV > 100.0) THEN
                DIMSUB=0
             ENDIF
             GOTO 100
          ENDIF
          IF(QDEBUG) WRITE(OUTU,'(A,1F10.5)')'COEFF VECTOR SIZE = ',NTMV
          !
          !       Expand move vector to the full parameter space.
          !

          VRSDL(1:np) = PSGRAD(1:np,LBEST)
          PGRAD(1:np) = GRAD(1:np,LBEST)

          DO I = 1,DIMSUB
             CALL SUBVEC(GRAD(1,LPREV(I)),GRAD(1,LBEST),WRK1(1,I),NP)
             CALL SUBVEC(PSGRAD(1,LPREV(I)),PSGRAD(1,LBEST),WRK4(1,I),NP)
          ENDDO
          DO J = 1,DIMSUB
             DO I = 1,NP
                PGRAD(I) = PGRAD(I) - XRED(J) * WRK1(I,J)
                VRSDL(I) = VRSDL(I) - XRED(J) * WRK4(I,J)
             ENDDO
          ENDDO
          CALL PFORCE(QPCYCL,PGRAD,PTHTAN1,NP,NREPL,NPATOMS,IREPS, &
               PJDX,PJDY,PJDZ,PTANX,PTANY,PTANZ,QPCIMG,ICLREP)
          CALL TFORCE(QPCYCL,VRSDL,PTHTAN1,NP,NREPL,NPATOMS,IREPS, &
               PJDX,PJDY,PJDZ,PTANX,PTANY,PTANZ,QPCIMG,ICLREP)
          !
          RSDLEN = LENVEC(PGRAD,NP)
          RSDLEN1 = LENVEC(VRSDL,NP)
          RSDSCL = SDSTP / SIGN(MAX(ABS(RSDLEN),RSMALL),RSDLEN)
          RSDSCL1 = SDSTP / SIGN(MAX(ABS(RSDLEN1),RSMALL),RSDLEN1)
          VWRK(1:np)=RSDSCL*PGRAD(1:np)+RSDSCL1*VRSDL(1:np)
          !          VWRK(I) =RSDSCL1 * VRSDL(I)

          DO J = 1,DIMSUB
             TEMP = XRED(J)
             VWRK(1:np) = VWRK(1:np) + TEMP * WRK2(1:np,J)
          ENDDO

       ENDIF ! QPNEB
#endif 
#endif 
    ENDIF ! SDFLAG
    CGSCL = ONE
    !
    !     For steepest descents calculate the displacement vector. Then
    !     determine the new parameters and test to see if the step length
    !     (NR or SD) is valid.
    !
100 CONTINUE
    IF (SDFLAG) THEN

#if KEY_REPLICA==1 && KEY_RPATH==1
       IF(.NOT.QPNEB) THEN    
#endif

          VRSDL(1:np) = GRAD(1:np,LBEST)
          TEMP = SQRTNP * GNRM(LBEST)

#if KEY_REPDSTR==1 && KEY_REPLICA==1 && KEY_RPATH==1
          IF(QREPDSTR.AND.QPATH)TEMP=NPS*GNRM(LBEST)  
#endif
          !write(50+mynodg,*)'ABNER-z>sqrtnp,nps,gnrm,temp=',sqrtnp,nps,gnrm(lbest),temp

          SDSCL = SDSTP / SIGN(MAX(ABS(TEMP),RSMALL),TEMP)
          VWRK(1:np) = SDSCL * VRSDL(1:np)
#if KEY_REPLICA==1
#if KEY_RPATH==1
       ELSE ! QPNEB

          pgrad(1:np) = GRAD(1:np,LBEST)
          vrsdl(1:np) = PSGRAD(1:np,LBEST)
          CALL PFORCE(QPCYCL,PGRAD,PTHTAN,NP,NREPL,NPATOMS,IREPS, &
               PJDX,PJDY,PJDZ,PTANX,PTANY,PTANZ,QPCIMG,ICLREP)
          CALL TFORCE(QPCYCL,VRSDL,PTHTAN,NP,NREPL,NPATOMS,IREPS, &
               PJDX,PJDY,PJDZ,PTANX,PTANY,PTANZ,QPCIMG,ICLREP)

          TEMP = SQRT(DOTVEC(PGRAD,PGRAD,NP))
          TEMP1= SQRT(DOTVEC(VRSDL,VRSDL,NP))
          SDSCL = SDSTP / SIGN(MAX(ABS(TEMP),RSMALL),TEMP)
          SDSCL1= SDSTP / SIGN(MAX(ABS(TEMP1),RSMALL),TEMP1)
          DO I = 1,NP
             VWRK(I) = SDSCL * PGRAD(I)+SDSCL1 * VRSDL(I)
          ENDDO

       ENDIF ! QPNEB
#endif 
#endif 
    ENDIF ! SDFLAG
    !
    CALL SUBVEC(PARM(1,LBEST),VWRK,PARM(1,LNEW),NP)
    STPLEN = LENVEC(VWRK,NP)

    !     get the partial lengths from other replicas:
    !write(50+mynodg,*)'ABNER-zz>stplen=',stplen
#if KEY_REPDSTR==1 && KEY_REPLICA==1 && KEY_RPATH==1
    IF(QREPDSTR.AND.QPATH)THEN     
#endif
#if KEY_REPDSTR==1 && KEY_REPLICA==1 && KEY_RPATH==1
       STPLEN2=STPLEN*STPLEN       
#endif
#if KEY_REPDSTR==1 && KEY_REPLICA==1 && KEY_RPATH==1
       if(mynod >0) stplen2=zero   
#endif
#if KEY_REPDSTR==1 && KEY_REPLICA==1 && KEY_RPATH==1
       CALL PSETGLOB               
#endif
#if KEY_REPDSTR==1 && KEY_REPLICA==1 && KEY_RPATH==1
       CALL GCOMB(STPLEN2,1)       
#endif
#if KEY_REPDSTR==1 && KEY_REPLICA==1 && KEY_RPATH==1
       CALL PSETLOC                
#endif
#if KEY_REPDSTR==1 && KEY_REPLICA==1 && KEY_RPATH==1
       STPLEN=SQRT(STPLEN2)        
#endif
#if KEY_REPDSTR==1 && KEY_REPLICA==1 && KEY_RPATH==1
    ENDIF                          
#endif
    !
    IF (STPLEN  >  STPLIM .AND. .NOT. SDFLAG) THEN
       CGSCL = 0.99 * STPLIM / SIGN(MAX(ABS(STPLEN),RSMALL),STPLEN)
       VWRK(1:np) = CGSCL * VWRK(1:np)
       GOTO 100
    ENDIF
    IF (STPLEN  <=  ZERO) THEN
       IF (SDFLAG) THEN
          CONVRG = 5
          DONE   = .TRUE.
          GOTO 200
       ENDIF
       SDFLAG = .TRUE.
       GOTO 100
    ENDIF
    !
    IF (QDEBUG) THEN
       WRITE (OUTU,'(/,2(3(A,F10.4),/))') &
            ' STPLIM = ',STPLIM,'  CGSCL  = ',CGSCL ,'  SDSTP  = ',SDSTP, &
            ' RSDLEN = ',RSDLEN,'  RSDSCL = ',RSDSCL,'  SDSCL  = ',SDSCL
       WRITE (OUTU,'(2(A,I5,5X))') &
            ' NRPLC  = ',NRPLC ,'  IRPLC  = ',IRPLC
       WRITE (OUTU,'(2(A,F12.5))') ' Best function value = ', &
            FUNC(LBEST),'  and its gradient = ',GNRM(LBEST)
       WRITE (OUTU,'(A)') &
            '     I Link    Eprev        Grad        Vred        Xred'
       WRITE (OUTU,'(1X,2I5,4F12.5)') (I,LPREV(I),FUNC(LPREV(I)), &
            GNRM(LPREV(I)),VRED(I),XRED(I),I = 1,DIMSUB)
    ENDIF

    !
    !     Copy PGRAD to VWRK1 to form the total residual force vector
    !
#if KEY_RPATH==1
#if KEY_REPLICA==1
    IF(QPNEB) THEN
       VWRK1(1:np) = PGRAD(1:np) + VRSDL(1:np)
    ENDIF
#endif 
#endif 
    !
    !     Calculate new values for the funtion and its gradients.
    !
    IF (SDFLAG) THEN
       STPTYP = 'SD'
    ELSE
       STPTYP = 'NR'
    ENDIF
    !
    !CCC      WHEN (NEWSTP) NCALLS = STPCNT
    !CCC      ELSE NCALLS = -1
    NCALLS = NCALLS + 1
    CALL EGRAD1(NP,PARM(1,LNEW),PARM(1,LBEST),FUNC(LNEW), &
         GRAD(1,LNEW),NCALLS,1,ERSTAT)
#if KEY_DHDGB==1
!AP/MF
    IF (QFHDGB)THEN
       IF (NPRINT .GT. 0 .AND. NCALLS .GE. 0) THEN
          IF (MOD(NCALLS,NPRINT) .EQ. 0) THEN
              WRITE(121,121)NCALLS,PARM(NP-10+1,LNEW), &
              PARM(NP-10+2,LNEW), &
              PARM(NP-10+3,LNEW),PARM(NP-10+4,LNEW), &
              PARM(NP-10+5,LNEW), &
              PARM(NP-10+6,LNEW),PARM(NP-10+7,LNEW), &
              PARM(NP-10+8,LNEW), &
              PARM(NP-10+9,LNEW),PARM(NP,LNEW)
121  FORMAT(I5,10(F7.3))
          ENDIF
       ENDIF
    ENDIF
#endif
#if KEY_REPDSTR==1
#if KEY_REPLICA==1
#if KEY_RPATH==1
    !write(50+mynodg,*)'ABNER-zz1>lnew,func(lnew)=',lnew,func(lnew)
    IF(QREPDSTR.AND.QPATH)THEN
       EPROP(GRMS) = LENVEC(GRAD(1,LNEW),NP)
       EPROP(GRMS) = EPROP(GRMS) * EPROP(GRMS)
       if(mynod > 0) func(lnew) = zero
       if(mynod > 0) eprop(grms) = zero
       !write(50+mynodg,*)'ABNER-zz2>eprop(grms)=',eprop(grms)
       CALL PSETGLOB
       CALL GCOMB(FUNC(LNEW),1)
       CALL GCOMB(EPROP(GRMS),1)
       CALL PSETLOC
       EPROP(GRMS)=SQRT(EPROP(GRMS))/SIGN(MAX(ABS(NPS),RSMALL),NPS)
    ENDIF
    !write(50+mynodg,*)'ABNER-zz1a>lnew,func(lnew)=',lnew,func(lnew)
    !write(50+mynodg,*)'ABNER-zz2a>eprop(grms)=',eprop(grms)
#endif 
#endif 
#endif 
    !
    CALL ENEOUT(OUTU,NCALLS,SDSTP,ZERO)
#if KEY_RPATH==1
#if KEY_REPLICA==1
    IF(QPNEB) THEN ! jwchu
       CALL EGRADNEB(NP,PARM(1,LNEW),PARM(1,LBEST), &
            GRAD(1,LNEW),PGRAD,PSGRAD(1,LNEW),ERSTAT,PTHTAN)
       vrsdl(1:np) = PSGRAD(1:np,LNEW)
       CALL TFORCE(QPCYCL,VRSDL,PTHTAN,NP,NREPL,NPATOMS,IREPS, &
            PJDX,PJDY,PJDZ,PTANX,PTANY,PTANZ,QPCIMG,ICLREP)
       ERMS=ETERM(PRMS)
       FUNC(LNEW)=FUNC(LNEW)-ERMS
       PFUNC(LNEW)=ERMS
    ENDIF
#endif /*                     */
#endif /*                     */
    !
    !     Write out minimization "trajectory" frame
    CALL MINTRJ(NCALLS,NSTEP,SDSTP)
    !
    FDIFF = FUNC(LNEW) - FUNC(LBEST)
    !write(50+mynodg,'(a,2i3,3f20.10)')'ABNR-9>lnew,lbest,fdiff,func(lnew),func(lbest)=', &
    !     lnew,lbest,fdiff,func(lnew),func(lbest)

    !      GNRM(LNEW) = LENVEC(GRAD(1,LNEW),NP) /
    !     $             SIGN(MAX(ABS(SQRTNP),RSMALL),SQRTNP)
    GRADAV2 = LENVEC(GRAD(1,LNEW),NP)
    PFLAG=.TRUE.
#if KEY_REPDSTR==1
#if KEY_RPATH==1
#if KEY_REPLICA==1
    !write(50+mynodg,*)'ABNER-10a>gradav2,np,nps=',gradav2,np,nps
    IF(QREPDSTR.AND.QPATH) THEN
       PFLAG=.FALSE.
       if(mynod > 0) gradav2 = zero
       CALL PSETGLOB
       !     We can only sum square
       GRADAV2=GRADAV2*GRADAV2
       CALL GCOMB(GRADAV2,1)
       GRADAV2=SQRT(GRADAV2)
       CALL PSETLOC
    ENDIF
    !write(50+mynodg,*)'ABNER-10b>gradav2,np,nps=',gradav2,np,nps
#endif 
#endif 
#endif 
    IF(PFLAG)THEN
       GRADAV2=GRADAV2/SIGN(MAX(ABS(SQRTNP),RSMALL),SQRTNP)
    ELSE
       GRADAV2=GRADAV2/SIGN(MAX(ABS(NPS),RSMALL),NPS)
    ENDIF
    !write(50+mynodg,*)'ABNER-10c>gradav,np,nps=',gradav2,np,nps
    ! End REPDSTR
    GNRM(LNEW) = GRADAV2
    !
#if KEY_REPLICA==1
#if KEY_RPATH==1
    IF(QPNEB) THEN
       FDIFF1 = PFUNC(LNEW) - PFUNC(LBEST)
       GNRM(LNEW) = LENVEC(PGRAD,NP) / &
            SIGN(MAX(ABS(SQRTNP),RSMALL),SQRTNP)
       PGNRM(LNEW) = LENVEC(PSGRAD(1,LNEW),NP) / &
            SIGN(MAX(ABS(SQRTNP),RSMALL),SQRTNP)
       IF(QDEBUG .OR. (PRNLEV >= 5)) THEN
          write(outu,'(A,1E10.5,A,1E10.5)') &
               ' OFF-PATH GRAD = ',  GNRM(LBEST), &
               ' TANGENT GRAD = ', PGNRM(LBEST)
       ENDIF
    ENDIF

    FDIFF2 = GNRM(LNEW) - GNRM(LBEST)
#endif 
#endif 

    NEVAL = NEVAL + 1
    IF ((NEVAL-NEVAL0)  >  TOLITR) THEN
       CONVRG = 4
       DONE   = .TRUE.
    ENDIF
    PDMAX  = ZERO
    DO I = 1,NP
       IF (ABS(PARM(I,LBEST))  >  1.0D-6) THEN
          PDIFF  = ABS((PARM(I,LNEW) - PARM(I,LBEST)) / &
               SIGN(MAX(ABS(PARM(I,LBEST)),RSMALL),PARM(I,LBEST)))
          IF (PDIFF  >  PDMAX) PDMAX = PDIFF
       ENDIF
    enddo
    !
    !     Get the maximum from replicas
    !
#if KEY_REPDSTR==1
#if KEY_REPLICA==1
#if KEY_RPATH==1
    !write(50+mynodg,*)'ABNER-max0>pdmax=',pdmax
    IF(QREPDSTR.AND.QPATH)THEN
       ALLOCATE(PDMAXS(NUMNODG))
       PDMAXS(1:numnodg)=ZERO
       if(mynod > 0) pdmax = zero
       !write(50+mynodg,*)'ABNER-max0b>pdmax=',pdmax
       PDMAXS(MYNODG+1)=PDMAX
       !write(50+mynodg,*)'ABNER-max0c>pdmaxs=',pdmaxs(1:numnodg)
       CALL PSETGLOB
       CALL GCOMB(PDMAXS,NUMNODG)
       CALL PSETLOC
       PDMAX=MAXVAL(PDMAXS)
       !write(50+mynodg,*)'ABNER-max1>pdmax=',pdmax
       DEALLOCATE(PDMAXS)
    ENDIF
#endif 
#endif 
#endif 
    !
    IF (PDMAX  <  PSTRCT) THEN
       CONVRG = 6
       DONE   = .TRUE.
    ENDIF
    GRADAV = FMEM*GRADAV + (ONE-FMEM)*GNRM(LNEW)
    STEPAV = FMEM*STEPAV + (ONE-FMEM)*STPLEN
#if KEY_REPLICA==1 && KEY_RPATH==1
    IF(.NOT.QPNEB) THEN   
#endif
       NEWSTP = (FDIFF  <=  STRICT)
       !write(50+mynodg,'(a,2f20.10,l6)')'ABNR-x-n>fdiff,strict,newstp=',fdiff,strict,newstp
#if KEY_RPATH==1
#if KEY_REPLICA==1
    ELSE
       NEWSTP = (((FDIFF <= STRICT).AND.(FDIFF1 <= STRICT))).OR. &
            (((FDIFF <= STRICT).AND.(ABS(FDIFF) > ABS(FDIFF1))))
    ENDIF
#endif /*       */
#endif /*       */
    IF (NEWSTP) NEVAL0  = NEVAL
    !
    !     Readjust the step length.
    !
#if KEY_REPLICA==1 && KEY_RPATH==1
    IF(.NOT.QPNEB) THEN  
#endif
       RSDSLP = DOTVEC(VRSDL,GRAD(1,LNEW),NP)
       !
#if KEY_REPDSTR==1
#if KEY_REPLICA==1
#if KEY_RPATH==1
       IF(QREPDSTR.AND.QPATH) THEN
          if(mynod > 0) rsdslp = zero   ! not tested !!!! ????
          CALL PSETGLOB
          CALL GCOMB(RSDSLP,1)
          CALL PSETLOC
       ENDIF
#endif 
#endif 
#endif 
       !
       IF (QDEBUG) WRITE (OUTU,'(A,1PE12.4)') &
            ' Dot of residual and new gradient = ',RSDSLP
       IF (SDFLAG) THEN
          IF (NEWSTP) THEN
             SDSTP  = SDSTP * 1.2 * (1.6 ** NTRIES)
             NTRIES = 0
          ELSE
             NTRIES = NTRIES + 1
             SDSTP  = SDSTP * HALF
          ENDIF
       ELSE
          IF (NEWSTP) THEN
             IF (RSDSLP  >  ZERO) THEN
                SDSTP = SDSTP * 1.5
             ELSE
                SDSTP = SDSTP * 0.9
             ENDIF
          ELSE
             IF (SDSTP  >  HALF*STPLEN) SDSTP = SDSTP * 0.8
          ENDIF
       ENDIF
       IF (SDSTP  <  STPMIN) SDSTP = STPMIN
#if KEY_REPLICA==1
#if KEY_RPATH==1
    ELSE ! QPNEB
       RSDSLP=DOTVEC(VWRK1,PGRAD,NP)
       RSDSLP1=DOTVEC(VWRK1,VRSDL,NP)
       FDOTDX=DOTVEC(VWRK,PGRAD,NP)
       FDOTDX1=DOTVEC(VWRK,VRSDL,NP)
       IF (QDEBUG) WRITE (OUTU,'(A,1PE12.4)') &
            ' Dot of off-path residual and new gradient = ',RSDSLP
       IF (QDEBUG) WRITE (OUTU,'(A,1PE12.4)') &
            ' Dot of tangent  residual and new gradient = ',RSDSLP1
       IF (QDEBUG) WRITE (OUTU,'(A,1PE12.4)') &
            ' Dot of DX and new off-path gradient = ',FDOTDX
       IF (QDEBUG) WRITE (OUTU,'(A,1PE12.4)') &
            ' Dot of DX and new tangent gradient = ',FDOTDX1
       IF (SDFLAG) THEN
          IF (NEWSTP) THEN
             IF(FDOTDX > ZERO.AND.FDOTDX1 > ZERO) THEN
                SDSTP  = SDSTP * 1.2 * (1.6 ** NTRIES)
                NTRIES = 0
             ELSEIF(FDOTDX > ZERO.OR.FDOTDX1 > ZERO) THEN !FDOTDX
                SDSTP  = SDSTP * 0.9
                NTRIES = 0
             ELSE !FDOTDX
                SDSTP  = SDSTP * 0.8
                NTRIES = 0
             ENDIF !FDOTDX
          ELSE
             NTRIES = NTRIES + 1
             IF(FDOTDX > ZERO.AND.FDOTDX1 > ZERO) THEN
                SDSTP  = SDSTP * 1.1
             ELSEIF(FDOTDX > ZERO.OR.FDOTDX1 > ZERO) THEN !FDOTDX
                SDSTP  = SDSTP * 0.4
             ELSE !FDOTDX
                SDSTP  = SDSTP * 0.3
             ENDIF !FDOTDX
          ENDIF
       ELSE
          IF (NEWSTP) THEN
             IF(FDOTDX > ZERO.AND.FDOTDX1 > ZERO) THEN
                IF (RSDSLP > ZERO.AND.RSDSLP1 > ZERO) THEN
                   SDSTP = SDSTP * 1.7
                ELSEIF (RSDSLP > ZERO.OR.RSDSLP1 > ZERO) THEN
                   SDSTP = SDSTP * 1.5
                ELSE
                   SDSTP = SDSTP * 1.2
                ENDIF
             ELSEIF(FDOTDX > ZERO.OR.FDOTDX1 > ZERO) THEN !FDOTDX
                IF (RSDSLP > ZERO.AND.RSDSLP1 > ZERO) THEN
                   SDSTP = SDSTP * 0.9
                ELSEIF (RSDSLP > ZERO.OR.RSDSLP1 > ZERO) THEN
                   SDSTP = SDSTP * 0.8
                ELSE
                   SDSTP = SDSTP * 0.7
                ENDIF
             ELSE !FDOTDX
                IF (RSDSLP > ZERO.AND.RSDSLP1 > ZERO) THEN
                   SDSTP = SDSTP * 0.8
                ELSEIF (RSDSLP > ZERO.OR.RSDSLP1 > ZERO) THEN
                   SDSTP = SDSTP * 0.7
                ELSE
                   SDSTP = SDSTP * 0.6
                ENDIF
             ENDIF !FDOTDX
          ELSE
             IF(FDOTDX > ZERO.AND.FDOTDX1 > ZERO) THEN
                IF (SDSTP  >  HALF*STPLEN) SDSTP = SDSTP * 0.8
             ELSEIF(FDOTDX > ZERO.OR.FDOTDX1 > ZERO) THEN !FDOTDX
                IF (RSDSLP > ZERO.AND.RSDSLP1 > ZERO) THEN
                   SDSTP = SDSTP * 0.8
                ELSEIF (RSDSLP > ZERO.OR.RSDSLP1 > ZERO) THEN
                   SDSTP = SDSTP * 0.7
                ELSE
                   SDSTP = SDSTP * 0.6
                ENDIF
             ELSE !FDOTDX
                IF (RSDSLP > ZERO.AND.RSDSLP1 > ZERO) THEN
                   SDSTP = SDSTP * 0.6
                ELSEIF (RSDSLP > ZERO.OR.RSDSLP1 > ZERO) THEN
                   SDSTP = SDSTP * 0.5
                ELSE
                   SDSTP = SDSTP * 0.4
                ENDIF
             ENDIF !FDOTDX
          ENDIF
       ENDIF
       IF (SDSTP  <  STPMIN) SDSTP = STPMIN
    ENDIF ! QPNEB
#endif 
#endif 
    !
    !     Convergence status.
    !
    IF (ABS(FUNC(LNEW) - FUNC(LBEST))  <=  TOLFUN) CONVRG = 3
    IF (STEPAV  <=  TOLSTP) CONVRG = 2
    IF (GRADAV  <=  TOLGRD) CONVRG = 1
    IF (ATLIM)              CONVRG = -1
    IF (CONVRG  /=  0) GOTO 200
    !
    IF (NCALLS  <  NSTEP .AND. .NOT.DONE) GOTO 1000
    !
    !     Exit from the minimization. Copy best variables to the VARB array.
    !
200 CONTINUE
    IF (FUNC(LNEW)  <  FUNC(LBEST)) LBEST = LNEW

    VARB(1:np) = PARM(1:np,LBEST)
#if KEY_DHDGB==1
!AP/MF
    IF (QFHDGB) THEN
        IF ((CONVRG .NE. 0) .AND. (NCALLS .LT. NSTEP )) THEN
            IF (NPRINT .GT. 0 .AND. NCALLS .GE. 0) THEN
                WRITE(121,141)NCALLS,PARM(NP-10+1,LBEST), &
                PARM(NP-10+2,LBEST), &
                PARM(NP-10+3,LBEST),PARM(NP-10+4,LBEST), &
                PARM(NP-10+5,LBEST), &
                PARM(NP-10+6,LBEST),PARM(NP-10+7,LBEST), &
                PARM(NP-10+8,LBEST), &
                PARM(NP-10+9,LBEST),PARM(NP,LBEST)
141  FORMAT(I5,10(F7.3))
            ENDIF
        ENDIF
    ENDIF
#endif
    FUNC(1) = FUNC(LBEST)
    GNRM(1) = GNRM(LBEST)
    !
#if KEY_REPDSTR==1
    IF(QREPDSTR.AND.QPATH) EPROP(GRMS) = GNRM(1)  
#endif
    !
    RETURN
  END SUBROUTINE ABNER2

  SUBROUTINE DSVEC(A,B,NSIZE,NI,N,GN)
    !-----------------------------------------------------------------------
    !     Compute the double precision dot product of two arrays
    !     of size NSIZE.  In subdomain from NI to NI+N
    !
  use chm_kinds
  use stream
    implicit none
    INTEGER I,N,NSIZE,NI
    real(chm_real)  A(NSIZE),B(NSIZE),SUM,GN
    !
    IF(NSIZE < NI+N-1) THEN
       write(outu,*)' sth wrong ', NSIZE, NI, N
       RETURN
    ENDIF
    SUM = 0.0
    DO I = NI,NI+N-1
       SUM = SUM + A(I) * B(I)
    enddo
    GN = SUM
    RETURN
  END SUBROUTINE DSVEC

  SUBROUTINE PFORCE(QPCYCL,F,TAN,NSIZE,NREPL,NPATOMS,IREP, &
       PFX,PFY,PFZ,PX,PY,PZ,LPCIMG,ICIMG)
    !-----------------------------------------------------------------------
    !     nudge the tangent part of F and left the perpendicular part
    !     it is done for each image I
    !
  use chm_kinds
  use stream
  use number
  use dimens_fcm
  use psf
  use coord
  use contrl
  use egrad
#if KEY_FLUCQ==1
  use flucq
#endif
#if KEY_DHDGB==1
  use dhdgb,only:totals
#endif 
    implicit none
    LOGICAL QPCYCL,LPCIMG
    INTEGER I,IPT,NI,NSIZE,NREPL,NPATOMS,IREP(NREPL),J
    INTEGER IBREP,ISREP,ICIMG
    real(chm_real)  F(*),TAN(*),GNX,GNY,GNZ,GN
    real(chm_real) PFX(*),PFY(*),PFZ(*),PX(*),PY(*),PZ(*)
#if KEY_DHDGB==1
  real(chm_real) DUM_FHDGB(TOTALS)
#endif
    PFX(1:NATOM) = zero
    PFY(1:NATOM) = zero
    PFZ(1:NATOM) = zero
    px(1:natom) = X(1:NATOM)
    py(1:natom) = Y(1:NATOM)
    pz(1:natom) = Z(1:NATOM)

    CALL PUTVR1(MINXYZ,NATOM,F,IMOVE,PFX,PFY,PFZ, &
#if KEY_CHEQ==1
         .FALSE.,(/ZERO/),(/ZERO/),(/ZERO/), & 
#endif
         .FALSE.,'NONE',(/ZERO/),(/ZERO/),.FALSE., &
#if KEY_FLUCQ==1
         .FALSE.,(/ZERO/),(/0/), & 
#endif
         .FALSE.,(/ZERO/),(/0/)  & 
#if KEY_DHDGB==1
       ,QFHDGB=.FALSE.,SDEF=DUM_FHDGB, TOTALS=TOTALS &
#endif
    )

    CALL PUTVR1(MINXYZ,NATOM,TAN,IMOVE,PX,PY,PZ, &
#if KEY_CHEQ==1
         .FALSE.,(/ZERO/),(/ZERO/),(/ZERO/), & 
#endif
         .FALSE.,'NONE',(/ZERO/),(/ZERO/),.TRUE., &
#if KEY_FLUCQ==1
         .FALSE.,(/ZERO/),(/0/), & 
#endif
         .FALSE.,(/ZERO/),(/0/) &
#if KEY_DHDGB==1
       ,QFHDGB=.FALSE.,SDEF=DUM_FHDGB, TOTALS=TOTALS &
#endif
    )

    IBREP=2
    ISREP=NREPL-2
    IF(QPCYCL) IBREP=1
    IF(QPCYCL) ISREP=NREPL-1
    DO I=IBREP,ISREP
       NI=IREP(I)+1
       CALL DSVEC(PFX,PX,NATOM,NI,NPATOMS,GNX)
       CALL DSVEC(PFY,PY,NATOM,NI,NPATOMS,GNY)
       CALL DSVEC(PFZ,PZ,NATOM,NI,NPATOMS,GNZ)
       GN=GNX+GNY+GNZ
       IPT=IREP(I)
       IF(LPCIMG.AND.(I == ICIMG)) THEN
          DO J=1,NPATOMS
             IPT=IPT+1
             PFX(IPT)=PFX(IPT)-TWO*GN*PX(IPT)
             PFY(IPT)=PFY(IPT)-TWO*GN*PY(IPT)
             PFZ(IPT)=PFZ(IPT)-TWO*GN*PZ(IPT)
          ENDDO
       ELSE
          DO J=1,NPATOMS
             IPT=IPT+1
             PFX(IPT)=PFX(IPT)-GN*PX(IPT)
             PFY(IPT)=PFY(IPT)-GN*PY(IPT)
             PFZ(IPT)=PFZ(IPT)-GN*PZ(IPT)
          ENDDO
       ENDIF
    ENDDO

    CALL GETVR1(MINXYZ,NATOM,F,IMOVE,PFX,PFY,PFZ,.FALSE., &
#if KEY_CHEQ==1
         .FALSE.,(/ZERO/), & 
#endif
         'NONE',(/ZERO/),(/ZERO/), &
#if KEY_FLUCQ==1
         .FALSE.,(/ZERO/),(/0/), & 
#endif
         .FALSE.,(/ZERO/),(/0/)  &
#if KEY_DHDGB==1
!AP/MF
     ,.FALSE.,(/ZERO/),TOTALS &
#endif
    ) 
    !
    RETURN
  END SUBROUTINE PFORCE

  SUBROUTINE TFORCE(QPCYCL,F,TAN,NSIZE,NREPL,NPATOMS,IREP, &
       PFX,PFY,PFZ,PX,PY,PZ,LPCIMG,ICIMG)
    !-----------------------------------------------------------------------
    !     nudge the tangent part of F and left the perpendicular part
    !     it is done for each image I
    !
  use chm_kinds
  use stream
  use number
  use dimens_fcm
  use psf
  use coord
  use contrl
  use egrad
#if KEY_FLUCQ==1
  use flucq
#endif
#if KEY_DHDGB==1
!AP/MF
  use dhdgb,only:totals
#endif
    implicit none
    LOGICAL QPCYCL,LPCIMG
    INTEGER I,IPT,NI,NSIZE,NREPL,NPATOMS,IREP(NREPL),J
    INTEGER IBREP,ISREP,ICIMG
    real(chm_real)  F(*),TAN(*),GNX,GNY,GNZ,GN
    real(chm_real) PFX(*),PFY(*),PFZ(*),PX(*),PY(*),PZ(*)
#if KEY_DHDGB==1
  real(chm_real) DUM_FHDGB(TOTALS)
#endif
    PFX(1:NATOM) = zero
    PFY(1:NATOM) = zero
    PFZ(1:NATOM) = zero
    px(1:natom) = X(1:NATOM)
    py(1:natom) = Y(1:NATOM)
    pz(1:natom) = Z(1:NATOM)

    CALL PUTVR1(MINXYZ,NATOM,F,IMOVE,PFX,PFY,PFZ, &
#if KEY_CHEQ==1
         .FALSE.,(/ZERO/),(/ZERO/),(/ZERO/), & 
#endif
         .FALSE.,'NONE',(/ZERO/),(/ZERO/),.FALSE., &
#if KEY_FLUCQ==1
         .FALSE.,(/ZERO/),(/0/), & 
#endif
         .FALSE.,(/ZERO/),(/0/) &
#if KEY_DHDGB==1
       ,QFHDGB=.FALSE.,SDEF=DUM_FHDGB, TOTALS=TOTALS &
#endif
      )

    CALL PUTVR1(MINXYZ,NATOM,TAN,IMOVE,PX,PY,PZ, &
#if KEY_CHEQ==1
         .FALSE.,(/ZERO/),(/ZERO/),(/ZERO/), & 
#endif
         .FALSE.,'NONE',(/ZERO/),(/ZERO/),.TRUE., &
#if KEY_FLUCQ==1
         .FALSE.,(/ZERO/),(/0/), & 
#endif
         .FALSE.,(/ZERO/),(/0/)  &
#if KEY_DHDGB==1
       ,QFHDGB=.FALSE.,SDEF=DUM_FHDGB, TOTALS=TOTALS &
#endif
     )

    IBREP=2
    ISREP=NREPL-2
    IF(QPCYCL) IBREP=1
    IF(QPCYCL) ISREP=NREPL-1
    DO I=IBREP,ISREP
       NI=IREP(I)+1
       CALL DSVEC(PFX,PX,NATOM,NI,NPATOMS,GNX)
       CALL DSVEC(PFY,PY,NATOM,NI,NPATOMS,GNY)
       CALL DSVEC(PFZ,PZ,NATOM,NI,NPATOMS,GNZ)
       GN=GNX+GNY+GNZ
       IPT=IREP(I)
       IF(LPCIMG.AND.(I == ICIMG)) THEN
          DO J=1,NPATOMS
             IPT=IPT+1
             PFX(IPT)=ZERO
             PFY(IPT)=ZERO
             PFZ(IPT)=ZERO
          ENDDO
       ELSE
          DO J=1,NPATOMS
             IPT=IPT+1
             PFX(IPT)=GN*PX(IPT)
             PFY(IPT)=GN*PY(IPT)
             PFZ(IPT)=GN*PZ(IPT)
          ENDDO
       ENDIF
    ENDDO

    CALL GETVR1(MINXYZ,NATOM,F,IMOVE,PFX,PFY,PFZ,.FALSE., &
#if KEY_CHEQ==1
         .FALSE.,(/ZERO/), & 
#endif
         'NONE',(/ZERO/),(/ZERO/), &
#if KEY_FLUCQ==1
         .FALSE.,(/ZERO/),(/0/), & 
#endif
         .FALSE.,(/ZERO/),(/0/) &
#if KEY_DHDGB==1
!AP/MF
        ,.FALSE., (/ZERO/),TOTALS &
#endif
       )
    !
    RETURN
  END SUBROUTINE TFORCE

#if KEY_REPLICA==1
#if KEY_RPATH==1
  SUBROUTINE PTANG(VARB,PTHTAN1,NP,NREPL,NPATOMS,IREP,EREP, &
       WEIGHT,PX,PY,PZ)
    !-----------------------------------------------------------------------
    !    calculate the tangent vector PTHTAN1 from VARB
    !
    !
  use epathmod
#if KEY_FLUCQ==1
  use flucqm,only: fqseln,fqcfor          
#endif

  use chm_kinds
  use exfunc
  use dimens_fcm
  use psf
  use coord
  use contrl
  use egrad
  use fourdm
  use stream
  use pathm
  use number
#if KEY_FLUCQ==1
  use flucq
#endif
#if KEY_DHDGB==1
!AP/MF
  use dhdgb,only:totals
#endif
    implicit none
    INTEGER I,J,NP,NREPL,NPATOMS,IREP(NREPL)
    real(chm_real)  VARB(NP),PTHTAN1(NP),EREP(NREPL)
    real(chm_real)  PX(*),PY(*),PZ(*)

    real(chm_real) WEIGHT(NPATOMS),WTOT
    INTEGER NREP
    real(chm_real),ALLOCATABLE,DIMENSION(:) :: RPLEN,RA,RB,DRA,DRB
    real(chm_real),ALLOCATABLE,DIMENSION(:) :: DRTEMP,PATANG
    INTEGER,ALLOCATABLE,DIMENSION(:) :: ATOMPR
    INTEGER IPT,JPT,JREP
#if KEY_DHDGB==1
  real(chm_real) DUM_FHDGB(TOTALS)
#endif
    !
    NREP=NREPL-1
    !
    !mfc --- NEEDS WORK --- change to chmalloc ------
    ALLOCATE(RPLEN(NREPL),ATOMPR(2*NPATOMS),RA(3*NPATOMS))
    ALLOCATE(RB(3*NPATOMS),DRA(3*NPATOMS),DRB(3*NPATOMS))
    ALLOCATE(DRTEMP(3*NPATOMS),PATANG(3*NREPL*NPATOMS))
    px(1:natom) = X(1:NATOM)
    py(1:natom) = Y(1:NATOM)
    pz(1:natom) = Z(1:NATOM)
    !
    CALL PUTVR1(MINXYZ,NATOM,VARB,IMOVE,PX,PY,PZ, &
#if KEY_CHEQ==1
         .FALSE.,(/ZERO/),(/ZERO/),(/ZERO/), & 
#endif
         .FALSE.,'NONE',(/ZERO/),(/ZERO/),.TRUE., &
#if KEY_FLUCQ==1
         .FALSE.,FQCFOR,FQSELN, & 
#endif
         .FALSE.,(/ZERO/),(/0/) &
#if KEY_DHDGB==1
       ,QFHDGB=.FALSE.,SDEF=DUM_FHDGB, TOTALS=TOTALS &
#endif
    )

    ! Fill the weight array
    WTOT=ZERO
    DO I=1,NPATOMS
       J=I+IREP(1)
       WEIGHT(I)=ONE
       IF(QPMASS) WEIGHT(I)=AMASS(J)
       IF(QPWEIG) WEIGHT(I)=WEIGHT(I)*WMAIN(J)
       WTOT=WTOT+WEIGHT(I)
       IF(WEIGHT(I) < ZERO) THEN
          WRITE(OUTU,*) 'In PTANG , st wrong '
       ENDIF
    ENDDO
    !
    ! Calculate Path tangent of each image
    !
    IF(.NOT.QRFIX) THEN
       CALL PATHTANG(PX,PY,PZ,PX,PY,PZ,NREPL,NREP, &
            RPLEN,NPATOMS,IREP,EREP,WEIGHT,WTOT, &
            .FALSE.,QPNOTR,QPNORT,QPCYCL,ATOMPR,RA,RB,DRA,DRB, &
            DRTEMP,PATANG,ZERO,QPETAN)
    ELSE
       CALL PATHTANG(PX,PY,PZ,REFPX,REFPY,REFPZ,NREPL,NREP, &
            RPLEN,NPATOMS,IREP,EREP,WEIGHT,WTOT, &
            .FALSE.,QPNOTR,QPNORT,QPCYCL,ATOMPR, &
            RA,RB,DRA,DRB,DRTEMP,PATANG,ZERO,QPETAN)
    ENDIF


    CALL PUTTAN(PX,PY,PZ,IREP,NREPL,NREP,NPATOMS,PATANG)

    PTHTAN1(1:NP)=zero
    CALL GETVR1(MINXYZ,NATOM,PTHTAN1,IMOVE,PX,PY,PZ,.FALSE., &
#if KEY_CHEQ==1
         .FALSE.,(/ZERO/), & 
#endif
         'NONE',(/ZERO/),(/ZERO/), &
#if KEY_FLUCQ==1
         .FALSE.,FQCFOR,FQSELN, & 
#endif
         .FALSE.,(/ZERO/),(/0/) &
#if KEY_DHDGB==1
!AP/MF
        ,.FALSE., (/ZERO/),TOTALS &
#endif
       )

    DEALLOCATE(ATOMPR,RA,RB,DRA,DRB,DRTEMP,PATANG,RPLEN)

    RETURN
  END SUBROUTINE PTANG

  SUBROUTINE PUTTAN(PX,PY,PZ,IREP,NREPL,NREP,NPATOMS, &
       PATANG)
    !-----------------------------------------------------------------------
    !     nudge the perpendicular part of F and left the tangent part
    !     it is done for each image I
    !
  use chm_kinds
  use stream
  use dimens_fcm
  use psf
  use number,only:zero
    implicit none
    INTEGER I,J,NP,NREPL,NREP,NPATOMS,IREP(NREPL)
    real(chm_real)  PX(*),PY(*),PZ(*)
    real(chm_real) PATANG(3,NREPL,NPATOMS)
    INTEGER JREP,IPT

    !CC transfer pathtang to ptanx, ptany, ptanz
    PX(1:NATOM)=zero
    PY(1:NATOM)=zero
    PZ(1:NATOM)=zero
    DO JREP=1,NREP
       IPT=IREP(JREP)
       DO J=1,NPATOMS
          IPT=IPT+1
          PX(IPT)=PATANG(1,JREP,J)
          PY(IPT)=PATANG(2,JREP,J)
          PZ(IPT)=PATANG(3,JREP,J)
       ENDDO
    ENDDO
    RETURN
  END SUBROUTINE PUTTAN
#endif 
#endif 
end module abnerm

