SUBROUTINE SHKCOM(COMLYN, COMLEN)
  !-----------------------------------------------------------------------
  !
  !     Process the SHAKE commands.
  !

  use new_timer,only:timer_start,timer_stop,T_shakesetup   

  use chm_kinds
  use dimens_fcm
  use number
  !
  use bases_fcm
  use coord
  use param
  use psf
  use select
  use shake
  use stream
  use string
#if KEY_FSSHK==1
  use fstshk,only: fsrscshk 
#endif
  use memory
  
  implicit none
  !
  !     . Passed variables.
  CHARACTER(len=*) :: COMLYN
  INTEGER   COMLEN
  !     . Local variables.
  INTEGER   ICONB, ires
  LOGICAL   QCOMP, QPARM, ERR, QFSHK
  CHARACTER(len=8) :: RENWAT
  integer, allocatable, dimension(:) :: ISLCT, JSLCT
  !
  !-----------------------------------------------------------------------

  call timer_start(T_shakesetup)
  !
  !-----------------------------------------------------------------------
#if KEY_FSSHK==1
  ! free memory used by fsshake (if allocated)
  IF(QSHAKE .AND. QFSHAKE) CALL FSRSCSHK
#endif 
  !-----------------------------------------------------------------------
  !     . Process OFF Keyword
  IF(INDXA(COMLYN,COMLEN,'OFF') > 0) THEN
     QSHAKE = .FALSE.
     QFSHAKE= .FALSE.
     ICONB=0
     NCONST=0
     CALL XTRANE(COMLYN,COMLEN,'SHKSET')
     IF(PRNLEV >= 2) WRITE (OUTU,'(A)') &
          ' SHKCOM> SHAKE constraints removed.'
     !
     call timer_stop(T_shakesetup)
     call deallocate_shake(natom)
     RETURN
  ENDIF
  !
  !-----------------------------------------------------------------------
  ! Initialise some counters.
  ICONB  = 0
  !
  call allocate_shake(natom)

  ! Parse the double atom selection
  call chmalloc('shake.src','SHKCOM','ISLCT',NATOM,intg=ISLCT)
  call chmalloc('shake.src','SHKCOM','JSLCT',NATOM,intg=JSLCT)
  CALL SELCTD(COMLYN,COMLEN,ISLCT,JSLCT,X,Y,Z,WMAIN,.TRUE.,ERR)
  !
  ! Parse the command line.
  IF (INDXA(COMLYN, COMLEN, 'BONH')  >  0) ICONB = 1
  IF (INDXA(COMLYN, COMLEN, 'BOND')  >  0) ICONB = 2
  IF (INDXA(COMLYN, COMLEN, 'ANGH')  >  0) ICONB = 3
  IF (INDXA(COMLYN, COMLEN, 'ANGL')  >  0) ICONB = 4
  !
  QPARM = INDXA(COMLYN, COMLEN, 'PARA')  >  0
  QCOMP = .NOT.(INDXA(COMLYN, COMLEN, 'MAIN')  >  0)
  QCOMP = INDXA(COMLYN, COMLEN, 'COMP')  >  0
  !
  ! Update the shake tolerance parameters.
  MXITER = GTRMI(COMLYN, COMLEN, 'MXIT', MXITER)
  SHKTOL = GTRMF(COMLYN, COMLEN, 'TOL',  SHKTOL)
  SHKSCA = GTRMF(COMLYN, COMLEN, 'SHKS', ONE)
  IF(PRNLEV >= 2) WRITE (OUTU,'(A,D12.4,A,I6)') &
       ' SHKCOM> SHAKE parameters: TOL = ', SHKTOL, &
       ' MXITer = ', MXITER

  QFSHK=.FALSE.
  IF(INDXA(COMLYN, COMLEN, 'FAST')  >  0) QFSHK=.TRUE.
  IF(INDXA(COMLYN, COMLEN, 'NOFA')  >  0) QFSHK=.FALSE.
  RENWAT=GTRMA(COMLYN,COMLEN,'WATE')
  IF(RENWAT == ' ') THEN
     RENWAT='TIP3'  ! default water resn='TIP3'
  ELSE
     ! Check to see if the resn is OK.
     DO IRES=1,NRES
        IF(RES(IRES) == RENWAT) GOTO 120
     ENDDO
     CALL WRNDIE(-2,'<SHKSET>', &
          'No atoms match specified water residue name')
120  CONTINUE
  ENDIF
  QFSHAKE=.FALSE.

  IF (INDXA(COMLYN, COMLEN, 'NORE')  >  0)  ICONB = -ICONB

  ! Done parsing parameters, now do the actual initializing
  call init_shake(iconb, qparm, qcomp, qfshk, renwat, islct, jslct)
  !
  call chmdealloc('shake.src','SHKCOM','ISLCT',NATOM,intg=ISLCT)
  call chmdealloc('shake.src','SHKCOM','JSLCT',NATOM,intg=JSLCT)

  CALL XTRANE(COMLYN,COMLEN,'SHKSET')
  call timer_stop(T_shakesetup)

  RETURN
END SUBROUTINE SHKCOM

! *
! * Initializes shake, called from shkcom
! *
subroutine init_shake(iconb, qparm, qcomp, qfshk, renwat, islct, jslct)
  use chm_kinds
  use memory,only:chmalloc,chmdealloc
  use shake
#if KEY_FSSHK==1
  use fstshk,only: fsrscshk,fsshkini    
#endif
#if KEY_TSM==1
  use tsmh,only:backls  
#endif
#if KEY_TSM==1
  use tsms_mod  
#endif
  use psf
  use code
  use pert
  use coord
  use coordc
  implicit none
  ! Input
  integer, intent(in) :: iconb
  logical, intent(in) :: qparm, qcomp, qfshk
  character(len=8), intent(in) :: renwat
  integer, intent(in) :: ISLCT(*), JSLCT(*)
  ! Variables
  integer, allocatable, dimension(:) :: IBEND, ITEND, ITMID, ICEND
  integer, allocatable, dimension(:) :: IBTMP, JBTMP, ITTMP, JTTMP, KTTMP, BACKPTR
  real(chm_real), allocatable, dimension(:) :: AMTMP

  !
  ! get distances for PARAM option
  IF(QPARM) THEN
     CALL CODES(ICB,ICT,[0],[0],NATOM,IMOVE,IAC,NBOND,IB,JB, &
          NTHETA,IT,JT,KT,0,[0],[0],[0],[0],0,[0],[0],[0],[0], &
          QDRUDE,NBDRUDE,                          & ! DRUDE
#if KEY_CMAP==1
          [0],0,[0],[0],[0],[0],[0],[0],[0],[0],                     & 
#endif
          .FALSE.,.FALSE.)
  ENDIF
  !
  call chmalloc('shake.src','SHKCOM','IBEND',NATOM,intg=IBEND)
  call chmalloc('shake.src','SHKCOM','ITEND',NATOM,intg=ITEND)
  call chmalloc('shake.src','SHKCOM','ITMID',NATOM,intg=ITMID)
  call chmalloc('shake.src','SHKCOM','ICEND',NATOM,intg=ICEND)
  !
#if KEY_TSM==1
  IF(QTSM.AND.PIGSET) THEN
     call chmalloc('shake.src','SHKCOM','IBTMP',NBOND,intg=IBTMP)
     call chmalloc('shake.src','SHKCOM','JBTMP',NBOND,intg=JBTMP)
     call chmalloc('shake.src','SHKCOM','ITTMP',NTHETA,intg=ITTMP)
     call chmalloc('shake.src','SHKCOM','JTTMP',NTHETA,intg=JTTMP)
     call chmalloc('shake.src','SHKCOM','KTTMP',NTHETA,intg=KTTMP)
     call chmalloc('shake.src','SHKCOM','AMTMP',NATOM,crl=AMTMP)
     call chmalloc('shake.src','SHKCOM','BACKPTR',NATOM,intg=BACKPTR)
     CALL PIGGSHK1(abs(ICONB),IBTMP,JBTMP,ITTMP, &
          JTTMP,KTTMP,AMTMP,BACKLS, &
          BACKPTR)
  ENDIF
#endif 
  !
  !-----------------------------------------------------------------------
  !ln...adds the following three lines for NORESET option, 27-JUL-95
  !sb ln+, [NO]RESET counters to ZERO, signal by ICONB <0
  !                  SHKSET immediately reverts sign of ICONB
!  IF (INDXA(COMLYN, COMLEN, 'NORE')  >  0)  ICONB = -ICONB
  !ln...
  !
  ! Set the SHAKE constraints 
  IF (QCOMP) THEN
     CALL SHKSET(XCOMP, YCOMP, ZCOMP, ICONB, QPARM, &
          ISLCT, JSLCT, IBEND,ITEND, ITMID, ICEND, &
          qfshk, &
#if KEY_PERT==1
          PERTIP,PERTCONST,PERTSHAUX &              
#endif
#if KEY_PERT==0
          0,0,0 &                                   
#endif
          )
  ELSE
     CALL SHKSET(X, Y, Z, ICONB, QPARM, &
          ISLCT, JSLCT, IBEND, ITEND, ITMID, ICEND, &
          qfshk, &
#if KEY_PERT==1
          PERTIP,PERTCONST,PERTSHAUX &              
#endif
#if KEY_PERT==0
          0,0,0 &                                   
#endif
          )
  ENDIF
  !
  !-----------------------------------------------------------------------
#if KEY_TSM==1
  IF(QTSM.AND.PIGSET) THEN
     CALL PIGGSHK2(ICONB,IBTMP,JBTMP,ITTMP,JTTMP,KTTMP,AMTMP)
  ENDIF
#endif 

#if KEY_FSSHK==1
  nconst_pll=nconst
  IF(QSHAKE .AND. QFSHK) THEN
     CALL fsSHKINI(ICONB,RENWAT)
     QFSHAKE=.TRUE.
  ENDIF
#if KEY_PARALLEL==1
  if(qfshk)then
     call collct_fstshk()
  endif
#endif 
#endif 
  !
  IF(QSHAKE) QHOLO = .TRUE.
  
  call chmdealloc('shake.src','SHKCOM','IBEND',NATOM,intg=IBEND)
  call chmdealloc('shake.src','SHKCOM','ITEND',NATOM,intg=ITEND)
  call chmdealloc('shake.src','SHKCOM','ITMID',NATOM,intg=ITMID)
  call chmdealloc('shake.src','SHKCOM','ICEND',NATOM,intg=ICEND)

#if KEY_TSM==1
  IF(QTSM.AND.PIGSET) THEN
     call chmdealloc('shake.src','SHKCOM','IBTMP',NBOND,intg=IBTMP)
     call chmdealloc('shake.src','SHKCOM','JBTMP',NBOND,intg=JBTMP)
     call chmdealloc('shake.src','SHKCOM','ITTMP',NTHETA,intg=ITTMP)
     call chmdealloc('shake.src','SHKCOM','JTTMP',NTHETA,intg=JTTMP)
     call chmdealloc('shake.src','SHKCOM','KTTMP',NTHETA,intg=KTTMP)
     call chmdealloc('shake.src','SHKCOM','AMTMP',NATOM,crl=AMTMP)
     call chmdealloc('shake.src','SHKCOM','BACKPTR',NATOM,intg=BACKPTR)
  endif
#endif 

  return
end subroutine init_shake

!
#if KEY_FSSHK==1 /*fsshk_on*/
#if KEY_PARALLEL==1 /*pll_fsshk*/
!---------------------------------------------------
!        COLLCT_FSTSHK
!---------------------------------------------------

subroutine collct_fstshk()
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use shake
  use fstshk,only:nother, nwatgr, nsh1, nsh2, nsh3, nsh4
  use stream
  use parallel
!!$#if KEY_DOMDEC==1
!!$  use domdec_common,only:q_domdec  
!!$#endif
  implicit none
  ! nconst and nconst_tot are already declared in shake.f90. LNI
  !      integer nconst,nconst_tot
  real(chm_real) tot_nshake(7)
  integer itot_nshake(7),itot,nc

  if(prnlev > 2)write(outu,'(a)') "   ==== COLLCT_FSTSHK ===="
  if(nconst == 0)then
     nother=0
     nother=0
     nwatgr=0
     nsh1  =0
     nsh2  =0
     nsh3  =0
     nsh4  =0
  endif
  tot_nshake(1)=nother
  tot_nshake(2)=nwatgr
  tot_nshake(3)=nsh1
  tot_nshake(4)=nsh2
  tot_nshake(5)=nsh3
  tot_nshake(6)=nsh4
  tot_nshake(7)=nconst
!!$#if KEY_DOMDEC==1
!!$  if (.not.q_domdec) then  
!!$#endif
     call gcomb(tot_nshake,7)
!!$#if KEY_DOMDEC==1
!!$  endif  
!!$#endif
  do itot=1,7
     itot_nshake(itot)=tot_nshake(itot)+.5
  enddo
  nconst_tot=itot_nshake(7)
  IF(PRNLEV > 2) THEN
     write(outu,48) itot_nshake(7)
     WRITE(OUTU,45) itot_nshake(1),itot_nshake(2)
     WRITE(OUTU,47) (itot_nshake(itot),itot=3,6)
  endif
45 FORMAT(' FSSHKINI: Fast shake initialized with',I8, &
       ' bond contraints and ',I8,' water constraints (3-center).')
47 FORMAT(' FSSHKINI: ',I6,' 2-body (CH), ', &
       I6,' 3-body(CH2), ',I6,' 4-body(CH3), ',I6,' >4-body. ')
48 format(" FSSHKINI: =========== Totals for all nodes:", &
       i8," total constraints")
  !
  !
  if(nconst_tot > maxshk)then
     IF(WRNLEV >= 2) WRITE(OUTU,122) NCONST_TOT,MAXSHK
122  FORMAT(' SHKSET: *** ERROR *** NO. OF CONSTRAINTS ', &
          I6,' IS LARGER THAN MAXSHK ',I6)
     call wrndie(-3,'<collct_fsshk>','Too many shake constraints')
  endif
  if(numnod > 1)then
!!$#if KEY_DOMDEC==1
!!$     if (.not.q_domdec) then  
!!$#endif
        call fill_fstprs(nconst,nconst_tot,shkapr,constr)
!!$#if KEY_DOMDEC==1
!!$     endif  
!!$#endif
     !     
     !     
     nconst_pll=nconst
     nconst=nconst_tot
  endif
  return
end subroutine collct_fstshk

!  ---------------------------------------------------
!        FILL_FSTPRS
!
subroutine fill_fstprs(nc,nct,shkapr,constr)
  ! Turn off iff not MPI enabled., LNI
#if KEY_MPI==1
  use chm_kinds
  use number
  use dimens_fcm
  use exfunc
  use parallel
  use memory
  ! Following may need specail handling on some machines
  use mpi
  implicit none
  integer nc,nct
  integer,allocatable,dimension(:),target :: itmpnc
  integer,pointer,dimension(:) :: tmpnc,tmpnc2
  integer,allocatable,dimension(:,:),target :: itmppr
  integer,pointer,dimension(:,:) :: tmppr,tmppr2
  integer ::shkapr(2,maxshk)
  integer i,ierr,start,ipr
  real(chm_real),allocatable,dimension(:) :: tmpc,tmpc2
  real(chm_real) :: constr(nct)

  call chmalloc('shake.src','collct_fstshk','itmppr',2*numnod,intg=itmpnc)
  tmpnc=> itmpnc(1:numnod)
  tmpnc2=> itmpnc(numnod+1:2*numnod)
  call chmalloc('shake.src','collct_fstshk','itmppr',2,2*nct,intg=itmppr)
  tmppr=> itmppr(1:2,1:nct)
  tmppr2=> itmppr(1:2,nct+1:2*nct)
  call chmalloc('shake.src','collct_fstshk','tmpc',nct,crl=tmpc)
  call chmalloc('shake.src','collct_fstshk','tmpc2',nct,crl=tmpc2)

  do i=1,numnod
     tmpnc(i)=0
  enddo
  tmpnc(mynodp)=nc
  call mpi_allreduce(tmpnc,tmpnc2,numnod,MPI_INTEGER,MPI_SUM, &
       COMM_CHARMM,ierr)
  do i=1,nct
     tmppr(1,i)=0
     tmppr(2,i)=0
     tmpc(i)=zero
  enddo
  start=0
  do i=1,mynod
     start=start+tmpnc2(i)
  enddo
  do i=1,nc
     tmppr(1,start+i)=shkapr(1,i)
     tmppr(2,start+i)=shkapr(2,i)
     tmpc(start+i)=constr(i)
  enddo
  !      return

  call mpi_allreduce(tmppr,tmppr2,nct*2, &
       MPI_INTEGER,MPI_SUM, &
       COMM_CHARMM,ierr)
  call mpi_allreduce(tmpc,tmpc2,nct, &
       MPI_DOUBLE_PRECISION,MPI_SUM, &
       COMM_CHARMM,ierr)
  ipr=nc+1
  do i=1,start
     shkapr(1,ipr)=tmppr2(1,i)
     shkapr(2,ipr)=tmppr2(2,i)
     constr(ipr)  =tmpc2 (i)
     ipr=ipr+1
  enddo
  do i=start+nc+1,nct
     shkapr(1,ipr)=tmppr2(1,i)
     shkapr(2,ipr)=tmppr2(2,i)
     constr(ipr)  =tmpc2 (i)
     ipr=ipr+1
  enddo
#else /**/
  call wrndie(-3,'<FSTPRS>','FSSHK needs compilation with MPI')
#endif 
  call chmdealloc('shake.src','collct_fstshk','itmppr',2*numnod,intg=itmpnc)
  call chmdealloc('shake.src','collct_fstshk','itmppr',2,2*nct,intg=itmppr)
  call chmdealloc('shake.src','collct_fstshk','tmpc',nct,crl=tmpc)
  call chmdealloc('shake.src','collct_fstshk','tmpc2',nct,crl=tmpc2)

  return
end subroutine fill_fstprs
#endif /*    (pll_fsshk)*/
#endif /*    (fsshk_on)*/

SUBROUTINE SHKSET(X,Y,Z,ICONB,QPARM,ISLCT,JSLCT,IBEND,ITEND, &
     ITMID,ICEND,qfshk,IPERT,PCONSTR,PSHAUX)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE DETERMINES THE CONSTRAINED
  !     DEGREES OF FREEDOM ( SHKAPR ARRAY ) AND ALSO
  !     DETERMINES THE CONSTRAINT VALUE FOR THE PRESENT
  !     SET OF COORDINATES X, Y AND Z.
  !
  !     ICONB  =0 - TURN OFF ALL SHAKE CONSTRAINTS
  !            =1 - ALL BONDS INVOLVING HYDROGENS
  !            =2 - ALL BONDS
  !            =3 - ALL ANGLES INVOLVING HYDROGENS
  !            =4 - ALL ANGLES
  !
  !     All ST2 bonds and angles are excluded.
  !     (See FIXST2 for rigid ST2 algorithm)
  !
  !      Authors: S. Swaminathan
  !      Robert Bruccoleri
  !      Overhauled by B. Brooks 10/20/90
  !
  !     Stefan Boresch, April 1995: Add communication between 
  !     SHAKE and the PERT free energy module.  The actual SHAKE
  !     routine(s) are modified to calculate a constraint correction
  !     to the free energy where necessary
  !
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use bases_fcm
  use code
  use param
  use psf
  use shake
  use stream
  use pert
  use pshake
  use parallel
  use chutil,only:lone,hydrog,atomid
!!$#if KEY_DOMDEC==1
!!$  use domdec_common,only:q_domdec  
!!$#endif
  !
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*)
  LOGICAL QPARM
  INTEGER ICONB
  INTEGER ISLCT(*),JSLCT(*)
  INTEGER IBEND(*),ITEND(*),ITMID(*),ICEND(*)
  INTEGER IPERT(*),PCONSTR(*)
  real(chm_real) PSHAUX(2,*)
  !
  LOGICAL ALLBND,ALLANG,OK,qperror,qfshk
  real(chm_real) XI,YI,ZI
  INTEGER I,IX,JX,KX
  CHARACTER(len=8) SID, RID, REN, AC
#if KEY_PERT==1
  real(chm_real) DERI
  real(chm_real) PTARGET
  EXTERNAL PTARGET
#endif 
#if KEY_FSSHK==1
#if KEY_PARALLEL==1
  integer iconst
  integer atfrst,atlast
  !

!!$#if KEY_DOMDEC==1
!!$  if (.not.q_domdec) then  
!!$#endif
     call parptupdate()
     atfrst=1+iparpt(mynod)
     atlast=iparpt(mynodp)
     qperror=.false.
!!$#if KEY_DOMDEC==1
!!$  endif  
!!$#endif
#endif 
#endif 
  !
  QSHAKE=.FALSE.
  IDGFEX=0
  !yw...Need to return when ICONB == 0 when incorporating LN changes, 27-Jul-95
  IF(ICONB == 0)THEN
     NCONST=0
     RETURN
     !ln...add the following five lines to enable the NORESET option, 27-JUL-95
  ELSE IF(ICONB  <  0)THEN
     ICONB=-ICONB
  ELSE
     NCONST=0
  ENDIF
  !ln...
  ALLBND=.FALSE.
  IF(ICONB > 1) ALLBND=.TRUE.
  !
  !sb   For PERT / constraint correct
  !
#if KEY_PERT==1
  IF (QPERT) THEN
     QPSHAKE=.FALSE.
     NPCONST=0
     DO I=1,MAXSHK
        PCONSTR(I)=0
     ENDDO
  ENDIF
#endif 
  !
  !     Count degrees of freedom for each atom.
  !
  DO I=1,NATOM
     IBEND(I)=0
     ITEND(I)=0
     ITMID(I)=0
     ICEND(I)=0
  ENDDO
  !
  !     Determine the bonds to be constrained
  !
  loop100: DO I=1,NBOND
     IX=IB(I)
     IF(IX <= 0) cycle loop100
     CALL ATOMID(IX,SID,RID,REN,AC)
#if KEY_NOST2==0
     IF(REN == 'ST2') cycle loop100
#endif 
     JX=JB(I)
     IF(LONE(IX).OR.LONE(JX)) cycle loop100
     IF(IMOVE(IX) > 0.AND.IMOVE(JX) > 0) cycle loop100
     IF(IMOVE(IX) < 0.OR.IMOVE(JX) < 0) cycle loop100
     ! Modified rules for selecting shake pairs.
     ! Now must have one atom in each selection set. - BRB
     OK=(ISLCT(IX) == 1.AND.JSLCT(JX) == 1)
     OK=OK.OR.(JSLCT(IX) == 1.AND.ISLCT(JX) == 1)
     !
     IF(OK) then
!!$#if KEY_DOMDEC==1
!!$        if (.not.q_domdec) then  
!!$#endif
#if KEY_FSSHK==1 /*fs0*/
#if KEY_PARALLEL==1 /*pll0*/
           if(qfshk)then
#if KEY_PARAFULL==1 /*pf0*/
              IF((IX < ATFRST .OR. IX > ATLAST) ) THEN
                 IF(JX < ATFRST .OR. JX > ATLAST) cycle loop100
                 GOTO 99
              elseIF((JX < ATFRST .OR. JX > ATLAST) ) THEN
                 GOTO 99
              endif
#elif KEY_PARASCAL==1 /*pf0*/
              IF(JPBLOCK(IX) /= MYNOD) THEN
                 IF(JPBLOCK(JX) /= MYNOD) cycle loop100
                 GOTO 99
              ElseIF(JPBLOCK(JX) /= MYNOD) THEN
                 GOTO 99
              endif
#elif KEY_SPACDEC==1 /*pf0*/
              IF(MYNOD /= ICPUMAP(IX))THEN
                 IF(MYNOD /= ICPUMAP(JX)) cycle loop100
                 GOTO 99
              ELSEIF(MYNOD /= ICPUMAP(JX)) THEN
                 !C                 CALL WRNDIE(-5,'<SHAKE>','SPACDEC not supported.')
                 GOTO 99
              endif
#endif /* (pf0)*/
           endif
#endif /* (pll0)*/
#endif /* (fs0)*/
!!$#if KEY_DOMDEC==1
!!$        endif  
!!$#endif

        IF(ALLBND .OR. HYDROG(IX).OR.HYDROG(JX)) THEN
           NCONST=NCONST+1
           IF (NCONST > MAXSHK) GOTO 22
           SHKAPR(1,NCONST)=IX
           SHKAPR(2,NCONST)=JX
           IBEND(IX)=IBEND(IX)+1
           IBEND(JX)=IBEND(JX)+1
           IF(IMOVE(IX) > 0) ICEND(JX)=ICEND(JX)+1
           IF(IMOVE(JX) > 0) ICEND(IX)=ICEND(IX)+1
#if KEY_PERT==1
           !sb       check whether one of the constituing atoms is affected
           !         by PERT
           IF (QPERT) THEN
              IF ((IPERT(IX) /= 0).OR.(IPERT(JX) /= 0)) THEN
                 NPCONST=NPCONST+1
                 PCONSTR(NCONST)=1
              ENDIF
              !
              IF (PCONSTR(NCONST) == 0) THEN
                 IF (QPARM) THEN
                    CONSTR(NCONST)=CBB(ICB(I))**2
                 ELSE
                    XI=X(IX)-X(JX)
                    YI=Y(IX)-Y(JX)
                    ZI=Z(IX)-Z(JX)
                    CONSTR(NCONST)=XI*XI+YI*YI+ZI*ZI
                 ENDIF
              ELSE
                 !SB          else  ... set constraint according to lambda etc...
                 !            also deposit derivative with respect to lambda there...
                 IF (.NOT.QPARM) CALL WRNDIE(-1,'<SHKSET>', &
                      'Using PARA option because of PERT for some bonds')
                 CONSTR(NCONST)=PTARGET(IX,JX,DERI, &
                      PPIB,PPJB, &
                      PPICB,PCONSTR)
                 PSHAUX(2,NCONST)=DERI
              ENDIF
           ELSE
#endif 
              IF (QPARM) THEN
                 CONSTR(NCONST)=CBB(ICB(I))**2
              ELSE
                 XI=X(IX)-X(JX)
                 YI=Y(IX)-Y(JX)
                 ZI=Z(IX)-Z(JX)
                 CONSTR(NCONST)=XI*XI+YI*YI+ZI*ZI
              ENDIF
#if KEY_PERT==1
           ENDIF
#endif 
           !
        ENDIF
     ENDIF
#if KEY_FSSHK==1
#if KEY_PARALLEL==1
     cycle loop100
99   IF(ALLBND .OR. HYDROG(IX).OR.HYDROG(JX)) THEN
        QPERROR=.TRUE.
     endif
#endif 
#endif 
  enddo loop100
  
!!$#if KEY_DOMDEC==1
!!$  if (.not.q_domdec) then  
!!$#endif
#if KEY_PARALLEL==1
#if KEY_FSSHK==1
  IF(QPERROR) THEN
     WRITE(OUTU,579)mynod
579  FORMAT(' ERROR in SHKSET(a) node',i4, &
          '. Contraints spanning parallel partition ignored.')
  ENDIF
#endif 
#endif 
!!$#if KEY_DOMDEC==1
!!$  endif  
!!$#endif

  IF(ICONB > 2) then   !GOTO 300
     ALLANG=.FALSE.
     IF(ICONB == 4) ALLANG=.TRUE.
     !
     !      Determine the angles to be constrained
     !sb    NOTE: At this moment I won't support angles in connection
     !            with PERT.  I'll only give a -1 warning; so people
     !            can override at their own peril.
     !
     loop200: DO I=1,NTHETA
        IX=IT(I)
        IF(IX <= 0) cycle loop200
        CALL ATOMID(IX,SID,RID,REN,AC)
#if KEY_NOST2==0
        IF(REN == 'ST2') cycle loop200
#endif 
        JX=JT(I)
        KX=KT(I)
        IF(LONE(IX).OR.LONE(KX)) cycle loop200
        IF(IMOVE(IX) > 0.AND.IMOVE(KX) > 0) cycle loop200
        IF(IMOVE(IX) < 0.OR.IMOVE(KX) < 0) cycle loop200
        OK=(ISLCT(IX) == 1.AND.ISLCT(KX) == 1)
        OK=OK.OR.(JSLCT(KX) == 1.AND.JSLCT(IX) == 1)
        IF(.NOT.OK) cycle loop200
        IF(ALLANG .OR. HYDROG(IX).OR.HYDROG(KX)) THEN
           NCONST=NCONST+1
           IF (NCONST > MAXSHK) GOTO 22
           SHKAPR(1,NCONST)=IX
           SHKAPR(2,NCONST)=KX
           ITEND(IX)=ITEND(IX)+1
           ITMID(JX)=ITMID(JX)+1
           ITEND(KX)=ITEND(KX)+1
           IF(IMOVE(IX) > 0) ICEND(KX)=ICEND(KX)+1
           IF(IMOVE(KX) > 0) ICEND(IX)=ICEND(IX)+1
#if KEY_PERT==1
           !sb       check whether one of the constituing atoms is affected
           !         by PERT
           IF (QPERT) THEN
              IF ((IPERT(IX) /= 0).OR.(IPERT(JX) /= 0).OR. &
                   (IPERT(KX) /= 0)) THEN
                 CALL WRNDIE(-1,'<SHKSET>', &
                      'Angle constraint affected by PERT')
              ENDIF
           ENDIF
#endif 
           IF (QPARM) THEN
              IF(CTUB(ICT(I)) <= 0.0) THEN
                 CALL WRNDIE(-2,'<SHKSET>', &
                      'Bad Urey-Bradley dist (PARM option).')
              ENDIF
              CONSTR(NCONST)=CTUB(ICT(I))**2
           ELSE
              XI=X(IX)-X(KX)
              YI=Y(IX)-Y(KX)
              ZI=Z(IX)-Z(KX)
              CONSTR(NCONST)=XI*XI+YI*YI+ZI*ZI
           ENDIF
        ENDIF
     enddo loop200
     !
     !     SOME OF THE OVER DEFINED CONSTRAINTS OF BOND ANGLE
     !     ARE REMOVED HERE.
     !     AS OF NOW THIS IS NOT DONE
     !
  endif

  !
  !     Count degrees of freedom for each atom.
  !
  DO I=1,NATOM
     IDGF2(I)=6-IBEND(I)-ITEND(I)-ICEND(I)
     IF(ITMID(I) >= 6) THEN
        IDGFEX=IDGFEX+1
        IDGF2(I)=IDGF2(I)+2
     ENDIF
     IF(IDGF2(I) <= 0) THEN
        IF(WRNLEV >= 2) WRITE(OUTU,421) I,IBEND(I),ITMID(I),ITEND(I)
        IDGF2(I)=0
     ENDIF
     IF(IMOVE(I) /= 0) IDGF2(I)=0
  ENDDO
421 FORMAT(' SHKSET:  * WARNING *  Atom',I6,' is involved in:',/ &
       '  ',I3,' bond end,',I3,' angle mid, and ', &
       I3,' angle end constraints.'/, &
       ' The number of degrees of freedom will probably be wrong.'/)
  !
  IF(PRNLEV >= 2) WRITE(OUTU,430) NCONST
430 FORMAT(/10X,I6,' constraints will held by SHAKE.')
  QSHAKE=(NCONST > 0)
#if KEY_PERT==1
  !sb   Is PERT interacting with SHAKE? Initialize PREVLA (have to do
  !     this somewhere)
  IF (QPERT) THEN
     QPSHAKE=(NPCONST > 0)
     IF (QPSHAKE) PREVLA=LAMDA
  ENDIF
#endif 
  RETURN
  !
  !     ERROR
  !
22 CONTINUE
  IF(WRNLEV >= 2) WRITE(OUTU,122) NCONST,MAXSHK
122 FORMAT(' SHKSET: *** ERROR *** NO. OF CONSTRAINTS ', &
       I6,' IS LARGER THAN MAXSHK ',I6)
  CALL DIEWRN(-2)
  NCONST=0
  IDGFEX=0
  RETURN
END subroutine shkset

#if KEY_PERT==1
SUBROUTINE PSHKSET(PCONSTR,PSHAUX)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE GOES THROUGH THE CONSTRAINT LIST AND
  !     UPDATES WHERE APPROPRIATE.  IT ASSUMES THAT EVERYTHING
  !     IS SET CORRECTLY, I.E. ONLY BOND CONSTRAINTS,
  !     PARAMETER OPTION IS USED
  !
  !     Author: S. Boresch, Dec 1994
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use bases_fcm
  use shake
  use pert
  use pshake
  implicit none
  !
  INTEGER PCONSTR(*)
  real(chm_real) PSHAUX(2,*)
  !
  !
  INTEGER I
  real(chm_real) DERI
  real(chm_real) PTARGET
  EXTERNAL PTARGET

  !     check whether update is necessary

  IF (ABS(LAMDA-PREVLA) < 1.D-10) RETURN

  !     go through list of constraint

  DO I=1,NCONST
     IF (PCONSTR(I) > 0) THEN
        !           this constraint needs to be updated...   
        CONSTR(I)=PTARGET(SHKAPR(1,I),SHKAPR(2,I),DERI, &
             PPIB,PPJB,PPICB,PCONSTR)
        PSHAUX(2,I)=DERI
     ENDIF
  enddo
  !sb fix
  PREVLA=LAMDA

  RETURN
END SUBROUTINE PSHKSET

FUNCTION PTARGET(IX,JX,DERI,IBP,JBP,ICBP,PCONSTR) result(ptarg)
  !-----------------------------------------------------------------------
  !     THIS FUNCTION RETURNS THE APPROPRIATE CONSTRAINT DISTANCE
  !     BASED ON THE INFORMATION IN THE LAMB=0 AND LAMB=1 PSF.  
  !     THE ATOM NUMBERING MUST NOT CHANGE, i.e. each reactant atom
  !     MUST have a product counterpart, but in principle
  !     THE BOND LIST CAN CHANGE.
  !
  !     Author: S. Boresch, Dec 1994
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use code
  use param
  use psf
  use shake
  use pert
  use pshake
  !
  implicit none
  !
  real(chm_real) :: ptarg
  INTEGER IX,JX
  real(chm_real) DERI
  INTEGER IBP(*),JBP(*),ICBP(*)
  INTEGER PCONSTR(*)
  !
  INTEGER BOND,BONDP
  LOGICAL QFOUND,QFOUND1
  real(chm_real) KOLD,ROLD,KNEW,RNEW

  !     NOW, WHAT DO WE HAVE
  !         PSF LAMB=1  IB, JB, ICB,CBB,CBC
  !         PSF LAMB=0  IBP,JBP,ICB,CBB,CBC

  !     FIND WHICH BOND IX,JX CONSTITUTE IN BOND-LIST (LAMB=1)
  QFOUND=.FALSE.
  DO BOND=1,NBOND    
     IF (((IX == IB(BOND)).AND.(JX == JB(BOND))).OR. &
          ((JX == IB(BOND)).AND.(IX == JB(BOND)))) THEN

        QFOUND=.TRUE.
        exit
     ENDIF
  ENDDO


  !     FIND WHICH BOND IX,JX CONSTITUTE IN BOND-LIST (LAMB=0)
  QFOUND1=.FALSE.
  DO BONDP=1,NBONDP
     IF (((IX == IBP(BONDP)).AND.(JX == JBP(BONDP))).OR. &
          ((JX == IBP(BONDP)).AND.(IX == JBP(BONDP)))) THEN

        QFOUND1=.TRUE.
        exit
     ENDIF
  ENDDO

  IF (.NOT.(QFOUND.AND.QFOUND1)) THEN
     CALL WRNDIE(-2,'<PTARG>', &
          'ERROR IN BOND LOOKUP.')
  ENDIF

  !     BOND CONTAINS BOND IN LAMB=1 PSF, BONDP IN LAMB=0 PSF
  !     LOOKUP VIA ICB AND ICBP

  KOLD=CBC(ICBP(BONDP))
  ROLD=CBB(ICBP(BONDP))
  KNEW=CBC(ICB(BOND))
  RNEW=CBB(ICB(BOND))

  !sb   at this I'm trying to be smart.  The constraint is only 
  !     affected by pert if rold != rnew.  if they are equal,
  !     remove that constraint from the list...
  !     This code is written in a scary fashion.  If my program logic
  !     is correct, the IF block below should be entered only
  !     if called from SHKSET.  The
  !     if block should never evaluate as true if routine was called from
  !     pshkset!!                                                        
  !sb   If others find this too confusing, the IF block can be removed:
  !     Everything should work as before

  ! Modification by Benoit Roux to fix the case when the force constant is zero
  ! as in TIP3 (KOLD=KNEW=0)
  IF (ABS(RNEW-ROLD) < 1.D-10) THEN
     NPCONST=NPCONST-1
     PCONSTR(NCONST)=0
     PTARG=ROLD
     DERI=ZERO
     !AvdV 10.2.03 Added this elseif statement for the subtle case
     !AvdV in which a group consisting of a real atom + 2 dummies are
     !AvdV converted into TIP3 water, using SHAKE for the initial DUMMY-DUMMY bond, 
     !AvdV intermediate DUMMY-DUMMY + HT-HT, and final HT-HT bond. This bond
     !AvdV would always have a force constant of zero, which would yield infinity in
     !AvdV the else statement below the elseif. Since the force constants are zero
     !AvdV for this unphysical bond, deri is zero by default.
  else if ((kold < 1.D-10).and.(knew < 1.D-10))then
     ptarg=(lamdam*rold+lamda*rnew)
     deri=zero
  ELSE
     PTARG=(LAMDAM*KOLD*ROLD+LAMDA*KNEW*RNEW)/ &
          (LAMDAM*KOLD     +LAMDA*KNEW)
     DERI=(KOLD*KNEW*(RNEW-ROLD))/ &
          ((LAMDAM*KOLD+LAMDA*KNEW)**2)
  ENDIF

  !     FINALLY, SQUARE IT SO THAT IT CAN BE USED DIRECTLY
  PTARG=PTARG*PTARG

  RETURN
END FUNCTION PTARGET
#endif 

SUBROUTINE PRNSHK(FOUND)
  !-----------------------------------------------------------------------
  !     Print out the shake constraints and show quality of fit
  !     to coordinate set.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use stream
  use coord
  use shake
  use chutil,only:atomid
  implicit none
  !
  LOGICAL FOUND
  !
  !
  INTEGER I,IX,JX
  real(chm_real) R2,RSHK,RCUR,RMSQ
  CHARACTER(len=8) SIDDNI,RIDDNI,RESDNI,ACDNI
  CHARACTER(len=8) SIDDNJ,RIDDNJ,RESDNJ,ACDNJ
  !
25 FORMAT(' SHAKE:',/, &
       '     Number of shake constraints:',I7,/, &
       '     Excess number of SHAKE restraints (estimate):',I5,/, &
       '     SHAKE tolerance:',D12.4)
27 FORMAT('     Set PRNLev>5 to print elements.')
  !
  IF(QSHAKE) THEN
     FOUND=.TRUE.
     IF(WRNLEV >= 2) WRITE(OUTU,25) NCONST, IDGFEX, SHKTOL
     IF(PRNLEV >= 2 .AND. PRNLEV <= 5) WRITE(OUTU,27)
     !
     RMSQ=0.0
     DO I=1,NCONST
        IX=SHKAPR(1,I)
        JX=SHKAPR(2,I)
        RSHK=SQRT(CONSTR(I))
        R2=(X(IX)-X(JX))**2+(Y(IX)-Y(JX))**2+(Z(IX)-Z(JX))**2
        RCUR=SQRT(R2)
        RMSQ=RMSQ+(RCUR-RSHK)**2
        !
        IF(PRNLEV > 5) THEN
           CALL ATOMID(IX,SIDDNI,RIDDNI,RESDNI,ACDNI)
           CALL ATOMID(JX,SIDDNJ,RIDDNJ,RESDNJ,ACDNJ)
           WRITE(OUTU,35) I,IX,SIDDNI(1:idleng),RIDDNI(1:idleng), &
                RESDNI(1:idleng),ACDNI(1:idleng), &
                JX,SIDDNJ(1:idleng),RIDDNJ(1:idleng), &
                RESDNJ(1:idleng),ACDNJ(1:idleng), &
                RSHK,RCUR,RCUR-RSHK
        ENDIF
     ENDDO
35   FORMAT(I5,' Atom',I5,4(1X,A),' With',I5,4(1X,A),/,5X, &
          ' R(cons)=',F12.5,' R(current)=',F12.5,' Diff=',F17.10)
     !
     RMSQ=SQRT(RMSQ/NCONST)
     IF(PRNLEV >= 2) WRITE(OUTU,55) RMSQ
55   FORMAT(' RMS deviation from SHAKE constraints is:',F17.10)
     !
  ENDIF
  !
  RETURN
END SUBROUTINE PRNSHK

SUBROUTINE SHAKEA2(X,Y,Z,XREF,YREF,ZREF,AMASS,LMASS,LDYNA,NATOM, &
     NITER,IMOVE,ISKP,PCONSTR,PSHAUX)
  !-----------------------------------------------------------------------
  !     Recoded from van Gunsteren routine by S. Swaminathan
  !     Modified by Doug Tobias (6/90) to accomodate simultaneous
  !     iterative solution of shake and internal coordinate constraint
  !     equations.
  !
  !     Stefan Boresch, April 1995: The routine is modified to
  !     explicitly return the magnitude of the Lagrangian multiplier
  !     for a given bond constraint.  This quantity is needed to
  !     calculate the constraint correction to a free energy in
  !     combination with PERT.
  !
  use chm_kinds
  use dimens_fcm
  use number
  use shake
  use stream
  use icfix
  use parallel
  use pshake
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*),XREF(*),YREF(*),ZREF(*)
  real(chm_real) AMASS(*)
  INTEGER NATOM,NITER
  INTEGER IMOVE(*),ISKP(*)
  LOGICAL LMASS,LDYNA
  INTEGER PCONSTR(*)
  real(chm_real) PSHAUX(2,*)
  !
#if KEY_PARALLEL==1
  INTEGER LLP
  LOGICAL QPERROR
  INTEGER MAP(MAXSHK), NCONP
#endif 
  !
  real(chm_real) AMSI,AMSJ,XIJ,YIJ,ZIJ,XPIJ,YPIJ,ZPIJ,DIFF, &
       RRPR,ACOR,TOL2
  LOGICAL READY
  INTEGER I,J,K,LL
  INTEGER ATFRST,ATLAST
  real(chm_real) TOLER
  !
  NITER=0
  !
  ATFRST=1
  ATLAST=NATOM
#if KEY_PARALLEL==1
  QPERROR=.FALSE.
#if KEY_PARAFULL==1
  !     Define the atom bounds for this processor.
  IF(LDYNA) THEN
     ATFRST=1+IPARPT(MYNOD)
     ATLAST=IPARPT(MYNODP)
  ENDIF
#endif 
#endif 
  !
  !     initialise skip variables
  !
  DO I=ATFRST,ATLAST
     ISKP(I)=0
  ENDDO
  READY=.FALSE.
  TOL2=2.0*SHKTOL
  !
#if KEY_PERT==1
  !sb   initialize array to average Lagrangian multiplier...
  IF (QPSHAKE) THEN
     DO I=1,NCONST
        PSHAUX(1,I)=ZERO
     ENDDO
  ENDIF
#endif 
  !     prepare parallel constraint list
  !     might be moved out into the routine that generates SHKAPP
#if KEY_PARALLEL==1
  NCONP=0
  loop90: DO LL=1,NCONST
     I=SHKAPR(1,LL)
     IF(I == 0) cycle loop90
     J=SHKAPR(2,LL)
#if KEY_PARAFULL==1
     IF(I < ATFRST .OR. I > ATLAST) THEN
        IF(J < ATFRST .OR. J > ATLAST) cycle loop90
        QPERROR=.TRUE.
        cycle loop90
     ENDIF
     IF(J < ATFRST .OR. J > ATLAST) THEN
        QPERROR=.TRUE.
        cycle loop90
     ENDIF
#elif KEY_PARASCAL==1
     IF(JPBLOCK(I) /= MYNOD) THEN
        IF(JPBLOCK(J) /= MYNOD) cycle loop90
        QPERROR=.TRUE.
        cycle loop90
     ENDIF
     IF(JPBLOCK(J) /= MYNOD) THEN
        QPERROR=.TRUE.
        cycle loop90
     ENDIF
#endif 
     NCONP=NCONP+1
     MAP(NCONP)=LL
  enddo loop90
  IF(QPERROR) THEN
     if (prnlev > 2) WRITE(OUTU,79)
79   FORMAT(' ERROR in SHAKEA.', &
          ' Contraints spanning parallel partition ignored.')
  ENDIF
#endif 
  !
  !     loop over shake iterations
  !
311 FORMAT(' ***** ERROR IN SHAKEA ***** COORDINATE RESETTING', &
       ' WAS NOT ACCOMPLISHED IN',I6,' ITERATIONS',/)
321 FORMAT(' ** ERROR IN SHAKEA ** DEVIATION IN SHAKE TOO LARGE'/ &
       ' NITER=',I5,' LL=',I5,' I=',I5,' J=',I5,/ &
       ' TOLER=',F18.10,' DIFF=',F18.10,' RRPR=',F18.10)
323 FORMAT(2X,A,3F20.10)
  !
  loop50: do while(.NOT.(READY))
     IF (NITER > MXITER) THEN
        !     Too many iterations
        IF(WRNLEV >= 2) WRITE(OUTU,311) MXITER
        CALL DIEWRN(-2)
        RETURN
     ENDIF
     READY=.TRUE.
#if KEY_PARALLEL==1
     loop100: DO LLP=1,NCONP
        LL=MAP(LLP)
#else /**/
     loop100: DO LL=1,NCONST
#endif 
        !
        I=SHKAPR(1,LL)
        IF(I == 0) cycle loop100
        J=SHKAPR(2,LL)
        !
        IF(ISKP(I)+ISKP(J) == -2) cycle loop100
        IF (IMOVE(I) > 0 .AND. IMOVE(J) > 0) cycle loop100
        TOLER=CONSTR(LL)
        XPIJ=X(I)-X(J)
        YPIJ=Y(I)-Y(J)
        ZPIJ=Z(I)-Z(J)
        DIFF=TOLER-XPIJ*XPIJ-YPIJ*YPIJ-ZPIJ*ZPIJ
        !
        !     compare difference in r**2 with tolerance
        !
        IF(ABS(DIFF) < TOLER*TOL2) cycle loop100
        !
        !     determine old ( or reference ) bond direction
        !
        XIJ=XREF(I)-XREF(J)
        YIJ=YREF(I)-YREF(J)
        ZIJ=ZREF(I)-ZREF(J)

        RRPR=XIJ*XPIJ+YIJ*YPIJ+ZIJ*ZPIJ
        IF(RRPR < TOLER*0.000001) THEN
           !     Deviation too large
           ! BIOVIA Code Start : Bug fix
           IF(WRNLEV >= 0 .AND. PRNLEV >= 2) THEN
           ! BIOVIA Code End
              WRITE(OUTU,321) NITER,LL,I,J,TOLER,DIFF,RRPR
              WRITE(OUTU,323) 'X(I)   ',X(I),Y(I),Z(I)
              WRITE(OUTU,323) 'XREF(I)',XREF(I),YREF(I),ZREF(I)
              WRITE(OUTU,323) 'X(J)   ',X(J),Y(J),Z(J)
              WRITE(OUTU,323) 'XREF(J)',XREF(J),YREF(J),ZREF(J)
           ENDIF
           call diewrn(1)   ! non-fatal optional exit...
           !
           !     Instead of dying, use current displacement instead of reference
           !     vector.  This won't conserve energy, but should only be needed in
           !     high energy situations.  - BRB (from RCZ)
           !
           ACOR=SQRT(TOLER/(TOLER-DIFF))
           XIJ=(X(I)-X(J))*ACOR
           YIJ=(Y(I)-Y(J))*ACOR
           ZIJ=(Z(I)-Z(J))*ACOR
           RRPR=XIJ*XPIJ+YIJ*YPIJ+ZIJ*ZPIJ
        ENDIF
        !
        IF (IMOVE(I) == 0) THEN
           IF (LMASS) THEN
              AMSI=ONE/AMASS(I)
           ELSE
              AMSI=1.0
           ENDIF
        ELSE
           AMSI=0.0
        ENDIF
        IF (IMOVE(J) == 0) THEN
           IF (LMASS) THEN
              AMSJ=ONE/AMASS(J)
           ELSE
              AMSJ=1.0
           ENDIF
        ELSE
           AMSJ=0.0
        ENDIF
        ACOR=DIFF/(RRPR*(AMSI+AMSJ)*TWO)
        ACOR=ACOR*SHKSCA
#if KEY_PERT==1
        !sb       accumulate acor (= essentially the Lagrangian mult.)
        IF (QPSHAKE) THEN
           IF (PCONSTR(LL) > 0) THEN
              PSHAUX(1,LL)=PSHAUX(1,LL)+ACOR
           ENDIF
        ENDIF
#endif 
        !
        !     shift new coordinates along old bond direction
        !
        XIJ=XIJ*ACOR
        X(I)=X(I)+XIJ*AMSI
        X(J)=X(J)-XIJ*AMSJ
        YIJ=YIJ*ACOR
        Y(I)=Y(I)+YIJ*AMSI
        Y(J)=Y(J)-YIJ*AMSJ
        ZIJ=ZIJ*ACOR
        Z(I)=Z(I)+ZIJ*AMSI
        Z(J)=Z(J)-ZIJ*AMSJ
        !
        !     set flags
        !
        ISKP(I)=1
        ISKP(J)=1
     enddo loop100
     NITER=NITER+1
     !
     !     iteration complete
     !     get flags ready for next iteration
     !
     DO K=ATFRST,ATLAST
        IF(ISKP(K) >= 0) ISKP(K)=ISKP(K)-1
        READY=READY.AND.(ISKP(K) == -1)
     ENDDO
     !
#if KEY_TSM==1
     !     Doug Tobias added the next three lines:
     ANYADJ=.FALSE.
     IF (NICF > 0) CALL ICFCNS(XREF,YREF,ZREF,X,Y,Z,AMASS,NATOM)
     READY=(READY.AND..NOT.ANYADJ)
#endif 
     !
  ENDDO loop50
  !
  RETURN
END SUBROUTINE SHAKEA2

SUBROUTINE SHAKEF(DX,DY,DZ,XREF,YREF,ZREF,AMASS,LMASS,NATOM, &
     IMOVE,NITER,ISKP)
  !
  ! This routine removes all forces related to holonomic constraints.
  !
  !-----------------------------------------------------------------------
  !     This routine shakes DX,DY,DZ with respect to XREF,YREF,ZREF.
  !     This removes any forces along a bond where SHAKE is applied.
  !
  !      BRB - 3/6/83
  !
  use chm_kinds
  use dimens_fcm
  use shake
  use stream
  implicit none
  !
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) XREF(*),YREF(*),ZREF(*)
  real(chm_real) AMASS(*)
  LOGICAL LMASS            ! use mass weighting
  INTEGER NATOM,NITER
  INTEGER IMOVE(*),ISKP(*)
  real(chm_real) AMSI,AMSJ,XIJ,YIJ,ZIJ,XPIJ,YPIJ,ZPIJ,DIFF,RRPR,ACOR
  INTEGER I,J,K,LL
  real(chm_real) TOLGRD
  LOGICAL READY
  !
  NITER=0
  !
  !     initialise skip variables

  ISKP(1:natom)=0

  !
  READY=.FALSE.
  TOLGRD=SHKTOL
  IF(TOLGRD < 0.00001) TOLGRD=0.00001
  !
  !     loop over shakef iterations
  !

  do while(.NOT.(READY))
     IF (NITER > MXITER) THEN
        IF(WRNLEV >= 2) WRITE(OUTU,311) MXITER
311     FORMAT(' ***** ERROR IN SHAKEF ***** FORCE RESETTING', &
             ' WAS NOT ACCOMPLISHED IN',I6,' ITERATIONS',/)
        CALL DIEWRN(-2)
        RETURN
     ENDIF
     READY=.TRUE.
     loop100: DO LL=1,NCONST
        I=SHKAPR(1,LL)
        IF(I == 0) cycle loop100
        J=SHKAPR(2,LL)
        !
        IF(ISKP(I)+ISKP(J) == -2) cycle loop100
        !
        IF (IMOVE(I) == 0) THEN
           IF (LMASS) THEN
              AMSI=1.E0/AMASS(I)
           ELSE
              AMSI=1.0
           ENDIF
        ELSE
           AMSI=0.0
        ENDIF
        IF (IMOVE(J) == 0) THEN
           IF (LMASS) THEN
              AMSJ=1.E0/AMASS(J)
           ELSE
              AMSJ=1.0
           ENDIF
        ELSE
           AMSJ=0.0
        ENDIF
        !
        XPIJ=DX(I)*AMSI-DX(J)*AMSJ
        YPIJ=DY(I)*AMSI-DY(J)*AMSJ
        ZPIJ=DZ(I)*AMSI-DZ(J)*AMSJ
        !
        ACOR=XPIJ*XPIJ+YPIJ*YPIJ+ZPIJ*ZPIJ
        IF(ACOR < 1.0) ACOR=1.0
        !
        !     determine old ( or reference ) bond direction
        !
        XIJ=XREF(I)-XREF(J)
        YIJ=YREF(I)-YREF(J)
        ZIJ=ZREF(I)-ZREF(J)
        DIFF=XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ
        RRPR=XIJ*XPIJ+YIJ*YPIJ+ZIJ*ZPIJ
        IF(ABS(RRPR) < SQRT(DIFF*ACOR)*TOLGRD) cycle loop100
        !
        ACOR=-RRPR/(DIFF*(AMSI+AMSJ))
        !
        !      WRITE(OUTU,88) NITER,I,J,XIJ,YIJ,ZIJ,XPIJ,YPIJ,ZPIJ
        !  88  FORMAT('ITER=',I3,' I=',I3,' J=',I3/,'      DELR=',3F20.6/
        !     1     '      DELF=',3F20.6)
        !      WRITE(OUTU,89) BCOR,SQRT(DIFF),CST,FDIFF,ACOR
        !  89  FORMAT('      DELF=',F20.6,' DELR=',F10.6,' COS(T)=',F10.6/
        !     1     '      FDIFF=',F20.6,' ACOR=',F20.6)
        !
        !     shift forces along bond direction
        !
        DX(I)=DX(I)+XIJ*ACOR
        DX(J)=DX(J)-XIJ*ACOR
        DY(I)=DY(I)+YIJ*ACOR
        DY(J)=DY(J)-YIJ*ACOR
        DZ(I)=DZ(I)+ZIJ*ACOR
        DZ(J)=DZ(J)-ZIJ*ACOR
        !
        !     SET FLAGS
        !
        ISKP(I)=1
        ISKP(J)=1
     enddo loop100
     NITER=NITER+1
     !
     !     iteration complete
     !     get flags ready for next iteration
     !
     DO K=1,NATOM
        IF(ISKP(K) >= 0) ISKP(K)=ISKP(K)-1
        READY=READY.AND.(ISKP(K) == -1)
     ENDDO

  enddo
  !
  RETURN
END SUBROUTINE SHAKEF

