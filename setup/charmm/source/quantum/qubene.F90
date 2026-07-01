#if KEY_QUANTUM==1

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
SUBROUTINE QUBEAMPAC(EQUANT,X,Y,Z,FFOCK,CLAS,IQMPIATM)
  !------------------------------------------------------------------------
  !     Calculate the energy for the quantum mechanical atoms and their
  !     electrostatic interactions with the MM atoms using the PM3, AM1 or
  !     MNDO semi-empirical approximations.
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  use memory
  !
  use inbnd
  use nbndqm_mod
  !
  use psf
  use quantm
  use sizes, only : N2ELEC
  implicit none
  !
  real(chm_real)   EQUANT,X(*),Y(*),Z(*)
  ! PI variables
  LOGICAL  FFOCK,CLAS
  INTEGER  IQMPIATM(*)
  !
  !
  !========================================================================
  !     CALCULATE THE QUANTUM MECHANICAL ENERGY.
  !========================================================================

  ! allocate and check necessary arrays.
  if(natqm_check.ne.natqm) then
     if(allocated(coorqm)) call chmdealloc('qubene.src','QUBEAMPAC','coorqm',3,size(coorqm,2),crl=coorqm)
     natqm_check=natqm
  end if
  if(natom_check.ne.natom) then
     if(allocated(xim)) call chmdealloc('qubene.src','QUBEAMPAC','xim',size(xim),crl=xim)
     if(allocated(yim)) call chmdealloc('qubene.src','QUBEAMPAC','yim',size(yim),crl=yim)
     if(allocated(zim)) call chmdealloc('qubene.src','QUBEAMPAC','zim',size(zim),crl=zim)
     natom_check=natom
  end if
  if(.not.allocated(coorqm)) call chmalloc('qubene.src','QUBEAMPAC','coorqm',3,NATQM,crl=coorqm)
  if(.not.allocated(xim)) call chmalloc('qubene.src','QUBEAMPAC','xim',natom,crl=xim)
  if(.not.allocated(yim)) call chmalloc('qubene.src','QUBEAMPAC','yim',natom,crl=yim)
  if(.not.allocated(zim)) call chmalloc('qubene.src','QUBEAMPAC','zim',natom,crl=zim)

  ! allocate memory for WJ_anal and WK_anal
  if(N2ELEC .gt. size(WJ_anal) .or. N2ELEC .gt. size(WK_anal)) then
     if(allocated(WJ_anal)) call chmdealloc('qubene.src','QUBEAMPAC','WJ_anal',size(WJ_anal),crl=WJ_anal)
     if(allocated(WK_anal)) call chmdealloc('qubene.src','QUBEAMPAC','WK_anal',size(WK_anal),crl=WK_anal)
  end if
  if(.not.allocated(WJ_anal)) call chmalloc('qubene.src','QUBEAMPAC','WJ_anal',N2ELEC+50,crl=WJ_anal)
  if(.not.allocated(WK_anal)) call chmalloc('qubene.src','QUBEAMPAC','WK_anal',N2ELEC+50,crl=WK_anal)

  !=======================================================================
  !     IMAGE SWAP FOR IMAGE CASE.
  !=======================================================================
  CALL SWAPXYZ_IMAGE(NATOM,X,Y,Z,xim,yim,zim,IMATTQ)
  !=======================================================================
  !
  CALL QUBEAMPC2(EQUANT,X,Y,Z,xim,yim,zim,coorqm,FFOCK,CLAS,IQMPIATM)
  !
  RETURN
END SUBROUTINE QUBEAMPAC

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
SUBROUTINE QUBEAMPC2(EQUANT,X,Y,Z,XI,YI,ZI,COORD,FFOCK,CLAS,IQMPIATM)
  !------------------------------------------------------------------------
  !     DOES THE WORK OF QUBEAMPAC.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use memory
  !
  use number
  use consta, only : EV_TO_KCAL
  use contrl

  use inbnd
  use nbndqm_mod
  use quantm
  use scfblk
  use psf
  use sizes
  use qmlinkm
  use leps
  use memory
  implicit none
  !
  real(chm_real)  EQUANT, X(*),Y(*),Z(*)
  real(chm_real)  COORD(3,*),XI(*),YI(*),ZI(*)
  INTEGER IQMPIATM(*)
  LOGICAL FFOCK,CLAS
  !
  real(chm_real)  E_LEPS,DA_LEPS_local(3),DB_LEPS_local(3),DC_LEPS_local(3), &
       DD_LEPS_local(3),DE_LEPS_local(3),DF_LEPS_local(3)
  !
  INTEGER INDEX,INDX1E
  !
  real(chm_real)  ENUCLR2,ENUCPE1,ENUCPE2
  !
  INTEGER I, IATOM, J, NDIM, NLEN, NLENB, NLENX, NQM, num_qm_grp
  INTEGER NQMPR, NMAX,JATOM,norbl_local
  !
  real(chm_real),allocatable,dimension(:) :: istore
  !
  !========================================================================
  real(chm_real) ECLASS
  real(chm_real) ETPER0,ETPER1,ETPER2,ELECP1,ELECP2,ELEGAS
  real(chm_real) EVER,ESTA,EDIS,EPOL
  INTEGER LENPAB
  !
  !      LOGICAL DEBUG1

  !========================================================================
  !  
  EQUANT    = ZERO
  ENUCLR_qm = ZERO
  ENUCLR2   = ZERO
  ELECT_qm  = ZERO
  INDX1E = 0
  NQM = 0

  !
  ! Transfer coordinates to QM array
  do IATOM = 1,NATOM
     IF (QATLAB(IATOM) .GE. 0) THEN
        NQM = NQM + 1
        COORD(1,NQM) = X(IATOM)
        COORD(2,NQM) = Y(IATOM)
        COORD(3,NQM) = Z(IATOM)
        INDX1E = INDX1E+1
        IF (QATLAB(IATOM) .GT. 1) INDX1E = INDX1E+9
     ENDIF
  end do
  !
  !------------------------------------
  !     CHECK WHETHER WE NEEDS TO REGENERATE INITIAL GUESS AGAIN.
  !
  !      IF(DYNAMQ.AND.(NEWGS.GT.0)) THEN
  !         NSTPMD=NSTPMD+1
  !         IF(MOD(NSTPMD,NEWGS).EQ.0) THEN
  !            CALL GESDEN(.FALSE.)
  !     IF THERE IS ANY GHO ATOMS? ===> ASK PU. HOW TO REDO THE INITIAL GUESS.
  !            NSTPMD=0
  !         END IF
  !      END IF
  !
  !
  ! NAMKH 08/08/04
  ! QM/MM-EWALD
  IF(LQMEWD) THEN
     ! ALLOCATE ARRAY FOR KSPACE GRADIENT..TRICKY.
     IF(QFIRSTD .and. allocated(PDXYZ)) THEN
        ! DTM remove??
        call chmdealloc('qubene.src','QUBEAMPC2','PDXYZ',3,size(PDXYZ,2),crl=PDXYZ)
     END IF
     ! DO THE INITIAL SETUP
     CALL SETUP_QMEWD
     ! PREPARE EWALD SUMMATION
     if( allocated(PKTABXCQ) .and. NATOM*MAXKVQ.ne.size(PKTABXCQ) ) then
        if(allocated(PKTABXCQ)) call chmdealloc('qubene.src','QUBEAMPC2','PKTABXCQ',size(PKTABXCQ),crl=PKTABXCQ)
        if(allocated(PKTABXSQ)) call chmdealloc('qubene.src','QUBEAMPC2','PKTABXSQ',size(PKTABXSQ),crl=PKTABXSQ)
        if(allocated(PKTABYCQ)) call chmdealloc('qubene.src','QUBEAMPC2','PKTABYCQ',size(PKTABYCQ),crl=PKTABYCQ)
        if(allocated(PKTABYSQ)) call chmdealloc('qubene.src','QUBEAMPC2','PKTABYSQ',size(PKTABYSQ),crl=PKTABYSQ)
        if(allocated(PKTABZCQ)) call chmdealloc('qubene.src','QUBEAMPC2','PKTABZCQ',size(PKTABZCQ),crl=PKTABZCQ)
        if(allocated(PKTABZSQ)) call chmdealloc('qubene.src','QUBEAMPC2','PKTABZSQ',size(PKTABZSQ),crl=PKTABZSQ)
     end if
     if(.not.allocated(PKTABXCQ)) call chmalloc('qubene.src','QUBEAMPC2','PKTABXCQ',NATOM*MAXKVQ,crl=PKTABXCQ)
     if(.not.allocated(PKTABXSQ)) call chmalloc('qubene.src','QUBEAMPC2','PKTABXSQ',NATOM*MAXKVQ,crl=PKTABXSQ)
     if(.not.allocated(PKTABYCQ)) call chmalloc('qubene.src','QUBEAMPC2','PKTABYCQ',NATOM*MAXKVQ,crl=PKTABYCQ)
     if(.not.allocated(PKTABYSQ)) call chmalloc('qubene.src','QUBEAMPC2','PKTABYSQ',NATOM*MAXKVQ,crl=PKTABYSQ)
     if(.not.allocated(PKTABZCQ)) call chmalloc('qubene.src','QUBEAMPC2','PKTABZCQ',NATOM*MAXKVQ,crl=PKTABZCQ)
     if(.not.allocated(PKTABZSQ)) call chmalloc('qubene.src','QUBEAMPC2','PKTABZSQ',NATOM*MAXKVQ,crl=PKTABZSQ)
     ! DTM remove??
     if(.not.allocated(PDXYZ))    call chmalloc('qubene.src','QUBEAMPC2','PDXYZ',3,NATOM,crl=PDXYZ)
     !
     CALL SETUP_KTABLE(NATOM,X,Y,Z,MAXKVQ,KMAXXQ,KMAXYQ,KMAXZQ, &
          PKTABXCQ,PKTABXSQ,PKTABYCQ,PKTABYSQ,PKTABZCQ,PKTABZSQ)
     ! RESET MMINB ARRAY FOR QM/MM INDEXING
     DO I = 1, NATOM
        IF(QATLAB(I) .GE. 0) THEN
           MMINB(I) = I
        ELSE
           MMINB(I) = -I
        END IF
     END DO
  END IF
  !
  IF(QMLINK) CALL MNHBDEF4(X,Y,Z,MBT,MBTM,MDBTMMM,MQMLNK,IMQLINK,JMQLINK,KMQLINK)
  !
  !========================================================================
  !     CALCULATE THE QM INTEGRALS.
  !========================================================================
  NDIM   = 22 * (NATQM * (NATQM - 1))
  call chmalloc('qubene.src','QUBEAMPC2','ISTORE',NDIM,crl=ISTORE)

  norbl_local=(norbs*(norbs+1))/2
  H_matrix(1:norbl_local) = ZERO     ! mpack
  WJ_anal(1:N2ELEC) = ZERO

  ! GENERATE 1-ELEC MATRIX AND 2-ELEC INTEGRALS
  ! DTM FFOCK arguments added
  IF (FFOCK) THEN
     CALL QUBHCORE(coord,H_matrix,WJ_anal,WJ_anal,WK_anal,ENUCLR_qm,ISTORE,FFOCK,CLAS,IQMPIATM)
  ELSE
     CALL MNHCORE(coord,H_matrix,WJ_anal,WJ_anal,WK_anal,ENUCLR_qm,ISTORE)
  ENDIF
  !-------------------------------------
  !   QM-MM ELECTROSTATIC FREE ENERGY PERTURBATION CALCULATIONS
  !   JG 7/9/99
  !
  IF(QMPERT) THEN
     h0gas(1:linqlk) = H_matrix (1:LINQLK)
     h1pert(1:linqlk) = H_matrix(1:LINQLK)
     h2pert(1:linqlk) = H_matrix(1:LINQLK)
     ENUGAS = ENUCLR_qm
     ENUCP1 = ENUCLR_qm
     ENUCP2 = ENUCLR_qm
  ENDIF
  IF(QDECOM) THEN
     h0gas(1:linqlk) = H_matrix(1:LINQLK)
     ENUGAS = ENUCLR_qm
  ENDIF
  !--------------------------------------
  !DCC-
  !
  ! IF DO NOT INCLUDE QM CORE-CORE REPULSION ENERGY.
  !
  IF (.NOT.QEQTRM(QQCC)) ENUCLR_qm = ZERO
  !DCC+
  !========================================================================
  !     CALCULATE THE ONE-ELECTRON INTERACTION INTEGRALS.
  !========================================================================
  IF (NQMGPE .GT. 0 .OR. NQMGEX .GT. 0) THEN
     ! . ALLOCATE SPACE.
     if(ngrp_old.ne.ngrp) then
        if(allocated(xyzg))  call chmdealloc('qubene.src','QUBEAMPC2','xyzg',size(xyzg,1),3,crl=xyzg)
        if(allocated(igrpg)) call chmdealloc('qubene.src','QUBEAMPC2','igrpg',size(igrpg),intg=igrpg)
        if(allocated(qgrpmm))call chmdealloc('qubene.src','QUBEAMPC2','qgrpmm',size(qgrpmm),log=qgrpmm)
        ngrp_old=ngrp
     end if
     if(.not.allocated(xyzg))  call chmalloc('qubene.src','QUBEAMPC2','xyzg',ngrp,3,crl=xyzg)
     if(.not.allocated(igrpg)) call chmalloc('qubene.src','QUBEAMPC2','igrpg',ngrp,intg=igrpg)
     if(.not.allocated(qgrpmm))call chmalloc('qubene.src','QUBEAMPC2','qgrpmm',ngrp,log=qgrpmm)

     ! . CALCULATE THE GROUP CENTRES OF GEOMETRY.
     CALL GRPCEN(NGRP,IGPBS,X,Y,Z,xyzg(1:ngrp,1),xyzg(1:ngrp,2),xyzg(1:ngrp,3))

     ! . FILL THE QM GROUP ARRAY.
     CALL GRPMM (NGRP,NATOM,IGPBS,NATQM,num_qm_grp,QATLAB,QGRPMM)
     ! . CALCULATE THE QM/MM INTERACTION TERMS.
     IF (NQMGPE .GT. 0) THEN
        CALL NAMPE(IQMGPE(1),JQMGPE(1),NLENB,NMAX,NQMPR)
        NLEN = NLENB * 10
        !JG
        CALL QUBEAMPCE(COORD,ENUCLR_qm,XI,YI,ZI,H_matrix,IQMGPE,JQMGPE, &
             E14FAC,NLENB,xyzg(1:ngrp,1),xyzg(1:ngrp,2),xyzg(1:ngrp,3), &
             QGRPMM,INDX1E,NQMPR,FFOCK,CLAS,IQMPIATM)

     ENDIF
     ! . CALCULATE THE QM/MM EXCLUSION TERMS.
     IF (NQMGEX .GT. 0) THEN
        CALL NAMPE(IQMGEX(1),JQMGEX(1),NLENX,NMAX,NQMPR)
        NLEN = NLENX * 10

        CALL QUBEAMPCE(COORD,ENUCLR_qm,XI,YI,ZI,H_matrix,IQMGEX,JQMGEX, &
             SFT123,NLENX,xyzg(1:ngrp,1),xyzg(1:ngrp,2),xyzg(1:ngrp,3), &
             QGRPMM,INDX1E,NQMPR,FFOCK,.FALSE.,IQMPIATM)

     ENDIF
  ENDIF
  ! NAMKH 08/08/04
  ! QM/MM-EWALD
  ! COMPUTE THE EWALD POTENTIAL ON QM ATOMS FROM ALL THE MM ATOMS
  IF(LQMEWD) CALL QEWALDP(X,Y,Z,XI,YI,ZI,.FALSE.)
  !
  !========================================================================
  !     DO THE SCF.
  !========================================================================
  !------------------------------------------------------------------------
  !   QM-MM ELECTROSTATIC FREE ENERGY PERTURBATION CALCULATIONS
  !   JG 7/9/99
  !
  ! DTM test
  !cc      CALL MNVECPRT(H_matrix,NORBS)
  !cc      J=900
  !cc      WRITE(6,'(//10X,''TWO-ELECTRON MATRIX IN HCORE''/)')
  !cc      WRITE(6,120)(WJ_anal(I),I=1,J)
  !cc  120 FORMAT(10F8.4)

  ! Run SCF
  IF(QMPERT) THEN
     IF(LQMEWD) THEN
        RLAMBF = RLAMB1
        CALL ITER(H1PERT,WJ_anal,WJ_anal,WK_anal,X,Y,Z,XI,YI,ZI,ENUCP1,ELECP1,.TRUE.)
        RLAMBF = RLAMB2
        CALL ITER(H2PERT,WJ_anal,WJ_anal,WK_anal,X,Y,Z,XI,YI,ZI,ENUCP2,ELECP2,.TRUE.)
        RLAMBF = RLAMB0
        CALL ITER(H_matrix,WJ_anal,WJ_anal,WK_anal,X,Y,Z,XI,YI,ZI,ENUCLR_qm,ELECT_qm,.TRUE.)
     ELSE
        CALL ITER(H1PERT,WJ_anal,WJ_anal,WK_anal,X,Y,Z,XI,YI,ZI,ENUCP1,ELECP1,.TRUE.)
        CALL ITER(H2PERT,WJ_anal,WJ_anal,WK_anal,X,Y,Z,XI,YI,ZI,ENUCP2,ELECP2,.TRUE.)
        CALL ITER(H_matrix,WJ_anal,WJ_anal,WK_anal,X,Y,Z,XI,YI,ZI,ENUCLR_qm,ELECT_qm,.TRUE.)
     END IF
  ELSEIF(QDECOM) THEN
     CALL ITER(H0GAS,WJ_anal,WJ_anal,WK_anal,X,Y,Z,XI,YI,ZI,ENUGAS,ELEGAS,.TRUE.)
     pgas(1:linqlk) = PDENS(1:LINQLK)
     CALL ITER(H_matrix,WJ_anal,WJ_anal,WK_anal,X,Y,Z,XI,YI,ZI,ENUCLR_qm,ELECT_qm,.TRUE.)
     penzym(1:linqlk) = PDENS(1:LINQLK)
  ELSE
     CALL ITER(H_matrix,WJ_anal,WJ_anal,WK_anal,X,Y,Z,XI,YI,ZI,ENUCLR_qm,ELECT_qm,.TRUE.)
  ENDIF
  !------------------------------------------------------------------------
  !
  ! NAMKH 08/08/04
  ! QM/MM-EWALD
  ENUCLR2=ZERO
  IF(LQMEWD) THEN
     IF(QMPERT) THEN
        ENUCPE1=ZERO
        RLAMBF = RLAMB1
        CALL QEWALD_CORE(ENUCPE1)
        ENUCP1 = ENUCP1 + ENUCPE1
        !
        ENUCPE2=ZERO
        RLAMBF = RLAMB2
        CALL QEWALD_CORE(ENUCPE2)
        ENUCP2 = ENUCP2 + ENUCPE2
        !
        RLAMBF = RLAMB0 
        CALL QEWALD_CORE(ENUCLR2)
     ELSE
        CALL QEWALD_CORE(ENUCLR2)
     END IF
  END IF
  ENUCLR_qm=ENUCLR_qm+ENUCLR2
  ! 
  !
#if KEY_FLUCQ==1
  CALL FQQMMM(PDENS,COORD,IQMGPE(1),JQMGPE(1),E14FAC,xyzg(1:ngrp,1),xyzg(1:ngrp,2),xyzg(1:ngrp,3),QGRPMM)
#endif 
  EQUANT = ZERO
  !
  ! NAMKH 08/08/04
  ! QM/MM-EWALD
  IF (QEQTRM(QQEL).OR.QEQTRM(QMEE)) EQUANT = ELECT_qm
  !
  !     IF INCLUDE QM CORE-CORE REPULSION.
  !
  !     THE FOLLOWING LINE WAS CHANGED TO ELIMINATE THE IF DETERMINATION
  !     BECAUSE THE QQCC DETERMINATION WAS MOVED TO SUBROUTINE ROTATE.
  !     ALSO, QMCH HAS TO BE EVALUATED ELSEWHERE.
  !
  EQUANT = EQUANT + ENUCLR_qm
  EQUANT = EQUANT * EV_TO_KCAL       ! 23.061D0
  !
  ! IF INCLUDE THE ATOMIC  HEAT OF FORMATION.
  !
  IF (QEQTRM(QATH)) EQUANT = EQUANT + ATHEAT
  !
  !-----------------------------------------------------------------------
  ! START: LEPS POTENTIAL CORRECTION
  !
  IF(QLEPS) THEN
     ! Switch of derivatives calculation
     NDER = ZERO
     E_LEPS = ZERO
     !
     ! ASSIGN COORDINATES
     XLA(1) = X(NTA)
     XLA(2) = Y(NTA)
     XLA(3) = Z(NTA)
     XLB(1) = X(NTB)
     XLB(2) = Y(NTB)
     XLB(3) = Z(NTB)
     XLC(1) = X(NTC)
     XLC(2) = Y(NTC)
     XLC(3) = Z(NTC)
     IF (SVB_DIM .EQ. 2) THEN
        XLD(1) = X(NTD)
        XLD(2) = Y(NTD)
        XLD(3) = Z(NTD)
        XLE(1) = X(NTE)
        XLE(2) = Y(NTE)
        XLE(3) = Z(NTE)
        XLF(1) = X(NTF)
        XLF(2) = Y(NTF)
        XLF(3) = Z(NTF)
     ENDIF
     !
     ! JG 5/02
     ! START: SVB ENERGY TERM
     IF(QSVB) THEN
        IF (SVB_DIM .EQ. 1) THEN
           CALL QM_SVB1D(E_LEPS,DA_LEPS_local,DB_LEPS_local,DC_LEPS_local)   
        ENDIF
        IF (SVB_DIM .EQ. 2) THEN
           CALL QM_SVB2D(E_LEPS,DA_LEPS_local,DB_LEPS_local,DC_LEPS_local,DD_LEPS_local,DE_LEPS_local,DF_LEPS_local)
        ENDIF
     ELSE
        CALL CORRECT_LEPS(E_LEPS,DA_LEPS_local,DB_LEPS_local,DC_LEPS_local)
     END IF
     ! ENERGY CORRECTION
     !
     EQUANT = EQUANT + E_LEPS
  END IF
  !
  ! END: LEPS POTENTIAL CORRECTION
  !------------------------------------------------------------------------
  !
  !      DETERMINE THE Q-LINK ATOM DERIVATIVES
  !
  IF(QMLINK) THEN
     ECLASS = ZERO
     !
     ! TREAT RHF AND UHF DIFFERENTLY ... PJ 12/2002
     !
     IF (.NOT. UHF) THEN
        CALL QUBDQLINK4(X,Y,Z,FA,UHF,ECLASS)
        EQUANT = EQUANT + ECLASS
     END IF
     !
     ! INCLUDE THE GRADIENT CORRECTION TERM FOR ALPHA AND BETA FOCK
     ! MATRICES SEPERATELY. THE REPULSION ENERGY BETWEEN MM ATOMS
     ! LINKED TO THE GHO BOUNDARY IS INCLUDED WHEN THE ALPHA
     ! CORRECTION TERM IS INCLUDED ... PJ 12/2002
     !
     IF (UHF) THEN
        CALL QUBDQLINK4(X,Y,Z,FA,.FALSE.,ECLASS)
        EQUANT = EQUANT + ECLASS
        CALL QUBDQLINK4(X,Y,Z,FB,UHF,ECLASS)
     END IF
  ENDIF
  !------------------------------------------------------------------------
  IF(QMPERT) THEN
     EEQTRM(QPT1) = (ELECP1+ENUCP1)*EV_TO_KCAL+ECLASS*FACTP1+ATHEAT
     EEQTRM(QPT2) = (ELECP2+ENUCP2)*EV_TO_KCAL+ECLASS*FACTP2+ATHEAT
     IF(LQMEWD) THEN
        EEQTRM(QPT1) = EEQTRM(QPT1)+(EBKCHG(1)-EBKCHG(2))
        EEQTRM(QPT2) = EEQTRM(QPT2)+(EBKCHG(1)-EBKCHG(3))
     END IF
  ENDIF
  IF(QDECOM) THEN
     CALL QMDCOM(PGAS,PENZYM,H0GAS,H_matrix,H1PERT, &
          EQUANT,ELECT_qm,ENUCLR_qm,ELEGAS,ENUGAS,ECLASS,ATHEAT, &
          EEQTRM(QVER),EEQTRM(QSTA),EEQTRM(QDIS),EEQTRM(QPOL), &
          NORBS,LINQLK)
     EEQTRM(QGAS)= (ELEGAS+ENUGAS)*EV_TO_KCAL+ATHEAT+ECLASS
  ENDIF
  !
  !========================================================================
  !!      IF(LQMEWD) THEN
  !!
  !!      Compute and put the derivative from Ewald potential
  !!         CALL QEWALDD(NATOM,NATQM,X,Y,Z,XI,YI,ZI,DXM,GRAD,PDXYZ)
  !!      END IF

  !========================================================================
  !
  call chmdealloc('qubene.src','QUBEAMPC2','istore',size(istore),crl=istore)
  !
  RETURN
END SUBROUTINE QUBEAMPC2

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
SUBROUTINE QUBEAMPCE(COORD,ENUCLR,X,Y,Z,H,INBX,JNBX,EXFACT,NDIM1, &
     XG,YG,ZG,QGRPMM,INDX1E,NQMPR,FFOCK,CLAS, &
     IQMPIATM)
  !DCC+
  !------------------------------------------------------------------------
  !
  !     Calculate the one-electron integrals and core-atom energies for
  !     the QM/MM interactions.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use memory
  !
  use quantm
  use scfblk, only : H1PERT,H2PERT
  use psf
  implicit none
  !
  ! . Passed variables.
  INTEGER  INBX(*),JNBX(*),NDIM1,IQMPIATM(*)
  LOGICAL  QGRPMM(*),FFOCK,CLAS
  real(chm_real)   COORD(3,*),EXFACT,ENUCLR,H(*),XG(*),YG(*),ZG(*) 
  real(chm_real)   X(*),Y(*),Z(*)
  !
  !     THE FOLLOWING ARRAYS ARE USED TO PASS THE PRODUCT OF THE ONE-ELECTRON
  !     INTEGRALS FOR QM/MM CHARGE/VALENCE INTERATIONS AND THE DERIVATIVE OF
  !     THE SWITCHING FUNCTION TO EAMPCG.  DCC 2.22.93
  !
  INTEGER  INDX1E,NQMPR,NLENM
  !
  ! . LOCAL VARIABLES.
  INTEGER   NDIM2,NLEN,isize_needed,isize_needed_2,space_2
  LOGICAL   AM1
  ! . POINTERS.
  integer, dimension(2) :: CGL,E1B,ENUC,FACT0,FACT1M,FACT1P,FACT20,FACT2M,FACT2P,FACT4P, &
       PPPP,PZPZ,RDIST,RQM,RQM2,RQM2B,RQMB,RQMI,SCALE,SFCT1M,SFCT1P,SFCT20, &
       SFCT2M,SFCT2P,SFCT4P,SPZ,SS,WORK1,WORK2,WORK3,WORK4,WORK5,XN,YN,ZN, &
       XN2,YN2,ZN2,XQM,YQM,ZQM
  !
  AM1   = (INDEX(KEYWRD,'AM1') .GT. 0) .OR. (INDEX(KEYWRD,'PM3') .GT. 0)

  ! . ALLOCATE SPACE FOR THE ARRAYS.
  NDIM2 = NATOM - NATQM
  IF (NDIM2 .GT. 0) THEN
     NLEN   = NDIM2
     NLENM  = NQMPR

     space_2       = 2*NQMPR            ! for integer array
     isize_needed  = 50*NQMPR + ndim2   ! for real array
     isize_needed_2= isize_needed*2     ! allocate twice of isize_needed and space

     !   allocate necessary memory/arrays.
     !   allocate more than necessary memory, so that do not have to allocate everytime.
     if(allocated(local_scratch).and.size(local_scratch).lt.isize_needed) then
        call chmdealloc('qubene.src','QUBEAMPCE','local_scratch',size(local_scratch),crl=local_scratch)
        call chmdealloc('qubene.src','QUBEAMPCE','local_iscratch',size(local_iscratch),intg=local_iscratch)
     end if
     if(.not.allocated(local_scratch)) then
        call chmalloc('qubene.src','QUBEAMPCE','local_scratch',isize_needed_2,crl=local_scratch)
        call chmalloc('qubene.src','QUBEAMPCE','local_iscratch',space_2,intg=local_iscratch)
     end if

     CGL(1)   =1           ; CGL(2)   =NQMPR            
     E1B(1)   =CGL(2)+   1 ; E1B(2)   =CGL(2)+10*NQMPR  
     ENUC(1)  =E1B(2)+   1 ; ENUC(2)  =E1B(2)   +NQMPR  
     FACT0(1) =ENUC(2)+  1 ; FACT0(2) =ENUC(2)  +NQMPR  
     FACT1M(1)=FACT0(2)+ 1 ; FACT1M(2)=FACT0(2) +NQMPR  
     FACT1P(1)=FACT1M(2)+1 ; FACT1P(2)=FACT1M(2)+NQMPR  
     FACT20(1)=FACT1P(2)+1 ; FACT20(2)=FACT1P(2)+NQMPR  
     FACT2M(1)=FACT20(2)+1 ; FACT2M(2)=FACT20(2)+NQMPR  
     FACT2P(1)=FACT2M(2)+1 ; FACT2P(2)=FACT2M(2)+NQMPR  
     FACT4P(1)=FACT2P(2)+1 ; FACT4P(2)=FACT2P(2)+NQMPR  
     PPPP(1)  =FACT4P(2)+1 ; PPPP(2)  =FACT4P(2)+NQMPR  
     PZPZ(1)  =PPPP(2)+  1 ; PZPZ(2)  =PPPP(2)  +NQMPR  
     RQM2(1)  =PZPZ(2)+  1 ; RQM2(2)  =PZPZ(2)  +NQMPR  
     RQM2B(1) =RQM2(2)+  1 ; RQM2B(2) =RQM2(2)  +NQMPR  
     RQM(1)   =RQM2B(2)+ 1 ; RQM(2)   =RQM2B(2) +NQMPR  
     RQMB(1)  =RQM(2)+   1 ; RQMB(2)  =RQM(2)   +NQMPR  
     RQMI(1)  =RQMB(2)+  1 ; RQMI(2)  =RQMB(2)  +NQMPR  
     SCALE(1) =RQMI(2)+  1 ; SCALE(2) =RQMI(2)  +NQMPR  
     SFCT1M(1)=SCALE(2)+ 1 ; SFCT1M(2)=SCALE(2) +NQMPR  
     SFCT1P(1)=SFCT1M(2)+1 ; SFCT1P(2)=SFCT1M(2)+NQMPR  
     SFCT20(1)=SFCT1P(2)+1 ; SFCT20(2)=SFCT1P(2)+NQMPR  
     SFCT2M(1)=SFCT20(2)+1 ; SFCT2M(2)=SFCT20(2)+NQMPR  
     SFCT2P(1)=SFCT2M(2)+1 ; SFCT2P(2)=SFCT2M(2)+NQMPR  
     SFCT4P(1)=SFCT2P(2)+1 ; SFCT4P(2)=SFCT2P(2)+NQMPR  
     SPZ(1)   =SFCT4P(2)+1 ; SPZ(2)   =SFCT4P(2)+NQMPR  
     SS(1)    =SPZ(2)+   1 ; SS(2)    =SPZ(2)   +NQMPR  
     WORK1(1) =SS(2)+    1 ; WORK1(2) =SS(2)    +NQMPR  
     WORK2(1) =WORK1(2)+ 1 ; WORK2(2) =WORK1(2) +NQMPR  
     WORK3(1) =WORK2(2)+ 1 ; WORK3(2) =WORK2(2) +NQMPR  
     WORK4(1) =WORK3(2)+ 1 ; WORK4(2) =WORK3(2) +NQMPR  
     WORK5(1) =WORK4(2)+ 1 ; WORK5(2) =WORK4(2) +NQMPR  
     XN(1)    =WORK5(2)+ 1 ; XN(2)    =WORK5(2) +NQMPR  
     YN(1)    =XN(2)+    1 ; YN(2)    =XN(2)    +NQMPR  
     ZN(1)    =YN(2)+    1 ; ZN(2)    =YN(2)    +NQMPR  
     XN2(1)   =ZN(2)+    1 ; XN2(2)   =ZN(2)    +NQMPR  
     YN2(1)   =XN2(2)+   1 ; YN2(2)   =XN2(2)   +NQMPR  
     ZN2(1)   =YN2(2)+   1 ; ZN2(2)   =YN2(2)   +NQMPR  
     XQM(1)   =ZN2(2)+   1 ; XQM(2)   =ZN2(2)   +NQMPR  
     YQM(1)   =XQM(2)+   1 ; YQM(2)   =XQM(2)   +NQMPR  
     ZQM(1)   =YQM(2)+   1 ; ZQM(2)   =YQM(2)   +NQMPR  

     CALL QUBEAMPE2(COORD,X,Y,Z,INBX,JNBX,EXFACT,AM1,ENUCLR,H,NDIM1,XG,YG,ZG,QGRPMM,NDIM2,local_iscratch, &
          local_scratch(cgl(1):cgl(2)),local_scratch(e1b(1):e1b(2)),local_scratch(enuc(1):enuc(2)), &
          local_scratch(FACT0(1):FACT0(2)),local_scratch(FACT1M(1):FACT1M(2)),local_scratch(FACT1P(1):FACT1P(2)),&
          local_scratch(FACT20(1):FACT20(2)),local_scratch(FACT2M(1):FACT2M(2)), &
          local_scratch(FACT2P(1):FACT2P(2)),local_scratch(FACT4P(1):FACT4P(2)), &
          local_scratch(PPPP(1):PPPP(2)),local_scratch(PZPZ(1):PZPZ(2)),local_scratch(RQM2(1):RQM2(2)), &
          local_scratch(RQM2B(1):RQM2B(2)),local_scratch(RQM(1):RQM(2)),local_scratch(RQMB(1):RQMB(2)), &
          local_scratch(RQMI(1):RQMI(2)),local_scratch(SCALE(1):SCALE(2)), &
          local_scratch(SFCT1M(1):SFCT1M(2)),local_scratch(SFCT1P(1):SFCT1P(2)), &
          local_scratch(SFCT20(1):SFCT20(2)),local_scratch(SFCT2M(1):SFCT2M(2)), &
          local_scratch(SFCT2P(1):SFCT2P(2)),local_scratch(SFCT4P(1):SFCT4P(2)), &
          local_scratch(SPZ(1):SPZ(2)),local_scratch(SS(1):SS(2)), &
          local_scratch(WORK1(1):WORK1(2)),local_scratch(WORK2(1):WORK2(2)),local_scratch(WORK3(1):WORK3(2)), &
          local_scratch(WORK4(1):WORK4(2)),local_scratch(WORK5(1):WORK5(2)), & 
          local_scratch(XN(1):XN(2)),local_scratch(YN(1):YN(2)),local_scratch(ZN(1):ZN(2)), &
          local_scratch(XN2(1):XN2(2)),local_scratch(YN2(1):YN2(2)),local_scratch(ZN2(1):ZN2(2)), &
          local_scratch(XQM(1):XQM(2)),local_scratch(YQM(1):YQM(2)),local_scratch(ZQM(1):ZQM(2)), & 
          H1PERT,H2PERT,INDX1E,NQMPR,FFOCK,CLAS,IQMPIATM)
     !DCC+
  ENDIF
  !
  RETURN
END SUBROUTINE QUBEAMPCE

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
SUBROUTINE QUBEAMPE2(COORD,X,Y,Z,INBX,JNBX,EXFACT,AM1,ENUCLR,H,NDIM1,XG,YG,ZG,QGRPMM_local,NDIM2,JNBL, &
     CGL,E1B,ENUC,FACT0,FACT1M,FACT1P,FACT20,FACT2M,FACT2P,FACT4P, &
     PPPP,PZPZ,RQM2,RQM2B,RQM,RQMB,RQMI,SCALE,SFCT1M,SFCT1P,SFCT20, &
     SFCT2M,SFCT2P,SFCT4P,SPZ,SS,WORK1,WORK2,WORK3,WORK4,WORK5, &
     XN,YN,ZN,XN2,YN2,ZN2,XQM,YQM,ZQM, &
     HPRT1,HPRT2,NDIM1E,NQMPR,FFOCK,CLAS,IQMPIATM)
  !DCC+
  !------------------------------------------------------------------------
  !     Does the work of EAMPCE.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use consta
  use number
  !
  !...  use coord
  !...  use deriv
  use inbnd
  use psf
  use quantm
  use am1parm
  use nbndqm_mod
  use sizes
  use qmlinkm
#if KEY_PBOUND==1
  use pbound  
#endif
#if KEY_PBOUND==1
  use coordc  
#endif
  !
  implicit none
#if KEY_PBOUND==1 /*pbound*/
  logical:: pbc_chk(3)
  real(chm_real):: xyz_pbc(3),xyz_size(3),xyz_size2(3)
#endif /*     (pbound)*/
  !
  ! . Passed variables.
  INTEGER   INBX(*),JNBX(*),JNBL(*),NDIM1,NDIM2,IQMPIATM(*)
  LOGICAL   AM1,QGRPMM_local(*),FFOCK,CLAS
  real(chm_real)  COORD(3,*),EXFACT,ENUCLR,H(*),X(*),Y(*),Z(*)
  !DCC-
  !     THE FOLLOWING ARRAYS ARE USED TO PASS THE PRODUCT OF THE ONE-ELECTRON
  !     INTEGRALS FOR QM/MM CHARGE/VALENCE INTERATIONS AND THE DERIVATIVE OF
  !     THE SWITCHING FUNCTION TO EAMPCG.  DCC 2.22.93
  !
  INTEGER   NDIM1E,NQMPR
  !DCC+
  !
  real(chm_real) CGL(*), E1B(NQMPR,*), ENUC(*), FACT0(*), FACT1M(*), &
       FACT1P(*), FACT20(*), FACT2M(*), FACT2P(*), &
       FACT4P(*), PPPP(*), PZPZ(*), RQM2(*), RQM2B(*), &
       RQM(*), RQMB(*), RQMI(*), SCALE(*), SFCT1M(*), &
       SFCT1P(*), SFCT20(*), SFCT2M(*), SFCT2P(*), &
       SFCT4P(*), SPZ(*), SS(*), WORK1(*), WORK2(*), &
       WORK3(*), WORK4(*), WORK5(*), XN(*), YN(*), ZN(*), &
       XN2(*), YN2(*), ZN2(*), XQM(*), YQM(*), ZQM(*), &
       XG(*), YG(*), ZG(*), HPRT1(*),HPRT2(*)
  ! . LOCAL VARIABLES.
  INTEGER  I, II, IJ, IPAIR, J, JJ, N, N1, N2, N3, MATM, MGRP, MNUM, &
       MSTRT, MSTOP, NBG, NPR, NQM, NTERM, NTOT, NUMQM, QATOM, &
       QFIRST, QGRP, QNUM, QSTOP, QSTRT, IQATOM
  LOGICAL  QSWITR
  real(chm_real)   ALPHA, C2OFNB, C2ONNB, CGQM, D1, D2, D4, &  !====DELTX, DELTY, DELTZ, &
       E1BA(10), EN, F1, F2, F3, FUNCT, &
       RHO0, RHO1, RHO2, RIJL, RIJU, RQM2L, RQM2U, RUL3, RUL12, &
       SCENT, SFACT, TEMP1, TEMP2, TEMP3, TEMP4, TPPPP, TPZPZ, &
       TRQM2B, XQ, YQ, ZQ
  real(chm_real):: delta_xyz(3)
  !
  real(chm_real),parameter :: QUARTR = PT25,    &         ! 0.25D0
       ATOBHR = 1.D0 / BOHRR, &
       ATOBH2 = ATOBHR * ATOBHR, &
       CONFAC = EV_TO_KCAL, &      ! 23.061D0
       CTOEV =  AU_TO_EV           ! 27.21D0
  real(chm_real) ENUCTMP
  !DCC-
  INTEGER  ITEMP1, MSTOP1(MAXGRP), NMGRP,MATM2
  INTEGER  QSTRTGRP,QSTOPGRP,QQSTOP,QQSTRT,INDX1E,NQMGROUP
  !
  real(chm_real)  HRCUT2
  ! DTM PI specific variables
  real(chm_real), save::  QUBENUCLR,QUBE1BA(MPACK)       ! COMMON  /QUBEAMR/ QUBENUCLR,QUBE1BA
  INTEGER K
  LOGICAL SKIPGRP,QUBFIRST
  !DCC+
  ! . DEFINE SOME CONSTANTS.
  !
  C2OFNB = CTOFNB * CTOFNB
  C2ONNB = CTONNB * CTONNB
  RUL3   = ONE / (C2OFNB - C2ONNB)**3
  RUL12  = TWELVE * RUL3
  !
#if KEY_PBOUND==1 /*PBOUND*/
  xyz_size(1)=XSIZE
  xyz_size(2)=YSIZE
  xyz_size(3)=ZSIZE
  XYZ_SIZE2(1:3) = half*xyz_size(1:3)
  If(qBoun) then
     IF(.not.qCUBoun) CALL WRNDIE(-5,'<QUBEAMPE2>', 'Other than CUBoundary not supported')
  End if
#endif /*     (PBOUND)*/
  ! DTM PI Initialize nuclear energy term for Fast Fock updating. Only done
  ! for centroid on first call
  IF (FFOCK .AND. CLAS) THEN
     QUBENUCLR = ZERO
     QUBE1BA(1:MPACK) = ZERO
  ENDIF
  !
  !
  ! . LOOP OVER THE GROUP NON-BOND LISTS - THE QM GROUPS ARE FIRST.
  NBG    = 0
  NTOT   = 0
  NUMQM  = 0
  QFIRST = 0
  QSTRTGRP = 0
  INDX1E = 0
  NQMGROUP = 0
  loopGRP: do QGRP = 1,NGRP
     NPR    = INBX(QGRP) - QFIRST
     QFIRST = INBX(QGRP)
     QSTRT  = IGPBS(QGRP) + 1
     QSTOP  = IGPBS(QGRP + 1)
     QNUM   = QSTOP - QSTRT + 1
     !
     ! . LOOP OVER THE MM GROUPS.
     N = 0
     IF (NPR .LE. 0) THEN
        IF (.NOT. QGRPMM_local(QGRP)) NUMQM = NUMQM + QNUM
        cycle loopGRP
     ENDIF
     !
     ! JG 3/21/01
     IF(QSTRTGRP.EQ.0) THEN
        QSTRTGRP=QSTRT
        QSTOPGRP=QSTRT+NATQM-1
     ENDIF
     !CC DTM
     ! If no PI atom in this group skip, and cycle loopGRP
     !         SKIPGRP = .TRUE.
     !         IF (FFOCK .AND. .NOT. CLAS) THEN
     !            DO K = QSTRT,QSTOP
     !               IF (IQMPIATM(K).EQ.1) SKIPGRP = .FALSE.
     !            ENDDO
     !            IF (SKIPGRP) THEN
     !              NUMQM = NUMQM + QNUM
     !              cycle loopGRP
     !            ENDIF
     !         ENDIF
     !-JG
     NMGRP = 0
     ITEMP1 = 0
     !
     loopPAIR: do IPAIR = 1,NPR
        NBG = NBG + 1
        MGRP = JNBX(NBG)
        IF (MGRP .LT. 0) THEN
           SFACT =  EXFACT
           MGRP  = -MGRP
        ELSE
           SFACT =  ONE
        ENDIF
        MSTRT = IGPBS(MGRP) + 1
        MSTOP = IGPBS(MGRP + 1)
        MNUM  = MSTOP - MSTRT + 1
        FUNCT = ONE
        !
        delta_xyz(1) = xg(qgrp) - xg(mgrp)
        delta_xyz(2) = yg(qgrp) - yg(mgrp)
        delta_xyz(3) = zg(qgrp) - zg(mgrp)
        ! JG 6/1/01
        ! ADD SIMPLE PERIODIC BOUNDARY CONDITIONS FOR QM/MM INTERACTION
#if KEY_PBOUND==1 /*PBOUND*/
        IF(QBOUN .and. QCUBOUN) THEN
           pbc_chk(1:3)=.false.
           do ij=1,3
              if(abs(delta_xyz(i)).gt.xyz_size2(i)) then
                 pbc_chk(i)  =.true.
                 xyz_pbc(i)  = sign(xyz_size(i),delta_xyz(i))
                 delta_xyz(i)= delta_xyz(i)-xyz_pbc(i)
              end if
           end do
        ENDIF
#endif /*  (PBOUND)*/
        SCENT = delta_xyz(1)*delta_xyz(1)+delta_xyz(2)*delta_xyz(2)+delta_xyz(3)*delta_xyz(3)
        !
        !--------------------------------------------------------------------------
        IF ((.NOT.QMCUTF) .OR. (SCENT .LT. C2OFNB)) THEN
           NMGRP = NMGRP + 1
           MSTOP1(NMGRP) = ITEMP1 + MSTOP - MSTRT + 1
           ITEMP1 = MSTOP1(NMGRP)
           !
           ! NAMKH 08/08/04
           ! QM/MM-EWALD
           IF(.NOT.LQMEWD) THEN
              !
              QSWITR = QMCUTF .AND. (SCENT .GT. C2ONNB)
              IF (QSWITR) THEN
                 RIJL  = C2ONNB - SCENT
                 RIJU  = C2OFNB - SCENT
                 FUNCT = RIJU * RIJU * (RIJU - THREE * RIJL) * RUL3
              ENDIF
              !
           ENDIF
           ! FILL THE INTERMEDIATE ATOM ARRAYS.
           loopMATM: do MATM = MSTRT,MSTOP
#if KEY_PBOUND==1 /*PBOUND*/
              IF(QBOUN .and. QCUBOUN) THEN
                 IF (pbc_chk(1)) THEN
                    XCOMP(MATM) = X(MATM)+xyz_pbc(1)
                 ELSE
                    XCOMP(MATM) = X(MATM)
                 ENDIF
                 IF (pbc_chk(2)) THEN
                    YCOMP(MATM) = Y(MATM)+xyz_pbc(2)
                 ELSE
                    YCOMP(MATM) = Y(MATM)
                 ENDIF
                 IF (pbc_chk(3)) THEN
                    ZCOMP(MATM) = Z(MATM)+xyz_pbc(3)
                 ELSE
                    ZCOMP(MATM) = Z(MATM)
                 ENDIF
              ENDIF
#endif /*  (PBOUND)*/
              !
              !      QM-LINK ATOM CASE: EXCLUDE THE LINK ATOM FROM THE MM LIST
              IF(QATLAB(MATM).GT.90) cycle loopMATM
              !
              ! NAMKH 08/08/04
              ! QM/MM-EWALD
              IF(LQMEWD.AND.(MMINB(MATM).LT.0)) MMINB(MATM) = MATM
              !
              N = N + 1
              JNBL(N) = MATM
              IF(QMPERT) THEN
                 CGL(N) = RLAMB0* SFACT*CG(MATM)
              ELSE
                 CGL(N)  = SFACT * CG(MATM)
              ENDIF
              SCALE(N) = FUNCT
           End do loopMATM
        ENDIF      ! IF ((.NOT.QMCUTF) .OR. (SCENT .LT. C2OFNB)) THEN
        !--------------------------------------------------------------------------
     end do loopPAIR

     ! . LOOP OVER THE QM ATOMS.
     !       INCLUDING THE QM-LINK ATOMS, NQMLNK
     !
     ! JG 3/21/01
     QQSTRT = QSTRT
     QQSTOP = QSTOP
     IF(QNBGRP) THEN
        QSTRT = QSTRTGRP
        QSTOP = QSTOPGRP
        NUMQM = 0
        INDX1E = 0
     ENDIF
     NQMGROUP = NQMGROUP+1
     !
     loopQATOM: do QATOM = QSTRT,QSTOP
        NUMQM = NUMQM + 1
        !--JG
        !CC DTM PI atom condition included to original condition
        IF (N.GT.0 .AND. QATLAB(QATOM).GT.0 .AND.(IQMPIATM(NUMQM).EQ.1 .OR. CLAS .OR. .NOT.FFOCK)) THEN
           !--------------------------------------------------------------------------
           !   ONE SHOULD RATHER DELETE MORE MM INTERACTION THAN WITHOUT
           !   INCLUDING QM ATOMS (LINK ATOMS) IN QM CALCULATIONS.  IT'S
           !   NOT GENERATING "BALANCED" CHARGE DISTRIBUTIONS AT THE H(LINK)
           !   ATOM.   JG 12/96
           !   NOT CHANGED IN STANDARD CHARMM....JG 1/6/99
           !           IF (N .GT. 0 .AND. QATLAB(QATOM) .GE. 0) THEN
           !--------------------------------------------------------------------------
           ! . DEFINE CONSTANTS FOR THE QUANTUM MECHANICAL ATOM.
           NQM    = NAT(NUMQM)
           ALPHA  = ALFA(NQM)
           CGQM   = CORE(NQM)
           N1 = N1ST(NUMQM)
           N2 = NMIDLE(NUMQM)
           N3 = NLAST(NUMQM)
           XQ = COORD(1,NUMQM)
           YQ = COORD(2,NUMQM)
           ZQ = COORD(3,NUMQM)
           !
           RHO0 = (HALF / bdd(NQM,1) + RHO0MM) **2
           !
           IF (AM1) THEN
              NTERM = 0
              do I = 1,10
                 IF (ABS(FN1(NQM,I)).GT.ZERO) NTERM = NTERM + 1
              end do
           ENDIF
           ! . DETERMINE SOME INTERMEDIATE ARRAYS.
           do J = 1, N
#if KEY_PBOUND==1 /*PBOUND*/
              IF(QBOUN .and. QCUBOUN) THEN
                 XQM(J) = XQ - XCOMP(JNBL(J))
                 YQM(J) = YQ - YCOMP(JNBL(J))
                 ZQM(J) = ZQ - ZCOMP(JNBL(J))
              ELSE
#endif /*     (PBOUND)*/
                 XQM(J) = XQ - X(JNBL(J))
                 YQM(J) = YQ - Y(JNBL(J))
                 ZQM(J) = ZQ - Z(JNBL(J))
#if KEY_PBOUND==1 /*PBOUND*/
              ENDIF
#endif /*     (PBOUND)*/
              RQM2(J) = XQM(J)*XQM(J)+YQM(J)*YQM(J)+ZQM(J)*ZQM(J)
              RQM(J)  = SQRT(RQM2(J))
              TEMP1   = ONE / RQM(J)
              XN(J)   = TEMP1 * XQM(J)
              YN(J)   = TEMP1 * YQM(J)
              ZN(J)   = TEMP1 * ZQM(J)
              RQMI(J) = TEMP1
           end do
           ! . CALCULATE THE INTEGRALS FOR ATOMS WITH ONE ORBITAL.
           IF (NATORB(NQM) .EQ. 1) THEN
              do J = 1, N
                 TRQM2B  = ATOBH2 * RQM2(J)
                 FACT0(J)= ONE / (TRQM2B + RHO0)
                 E1B(J,1)= CTOEV * SQRT(FACT0(J))
              end do
              E1B(1:N,2:10) = ZERO
              !
              ! . DO THE INTEGRALS FOR ATOMS WITH MORE THAN ONE ORBITAL.
           ELSE
              RHO1 = (HALF / bdd(NQM,2) + RHO0MM) **2
              RHO2 = (HALF / bdd(NQM,3) + RHO0MM) **2
              !
              D1 = DD(NQM)
              D2 = TWO * QQ(NQM)
              D4 = D2 * D2
              ! . LOOP OVER THE MOLECULAR MECHANICS ATOMS.
              do J = 1, N
                 RQM2B(J)  = ATOBH2 * RQM2(J)
                 RQMB(J)   = ATOBHR * RQM(J)
                 WORK1(J)  = RQMB(J) - D1
                 WORK2(J)  = RQMB(J) + D1
                 WORK3(J)  = RQMB(J) - D2
                 WORK4(J)  = RQMB(J) + D2
              end do
              !
              do J = 1, N
                 FACT0(J)  = ONE / (RQM2B(J) + RHO0)
                 FACT1M(J) = ONE / (WORK1(J) * WORK1(J) + RHO1)
                 FACT1P(J) = ONE / (WORK2(J) * WORK2(J) + RHO1)
                 FACT20(J) = ONE / (RQM2B(J) + RHO2)
                 FACT2M(J) = ONE / (WORK3(J) * WORK3(J) + RHO2)
                 FACT2P(J) = ONE / (WORK4(J) * WORK4(J) + RHO2)
                 FACT4P(J) = ONE / (RQM2B(J) + D4 + RHO2)
                 SFCT1M(J) = SQRT(FACT1M(J))
                 SFCT1P(J) = SQRT(FACT1P(J))
                 SFCT20(J) = SQRT(FACT20(J))
                 SFCT2M(J) = SQRT(FACT2M(J))
                 SFCT2P(J) = SQRT(FACT2P(J))
                 SFCT4P(J) = SQRT(FACT4P(J))
              end do
              !
              do J = 1, N
                 SS(J)   = SQRT(FACT0(J))
                 SPZ(J)  = HALF*(SFCT1P(J)-SFCT1M(J))
                 PZPZ(J) = SS(J) + QUARTR*(SFCT2P(J)+SFCT2M(J)) - HALF*SFCT20(J)
                 PPPP(J) = SS(J) + HALF * (SFCT4P(J)-SFCT20(J))
              end do
              ! . DEFINE SOME VARIABLES NEEDED FOR THE TRANSFORMATION.
              do J = 1, N
                 XN2(J) = XN(J)*XN(J)
                 YN2(J) = YN(J)*YN(J)
                 ZN2(J) = ZN(J)*ZN(J)
              end do
              ! . FILL ARRAYS WITH THE CALCULATED INTEGRALS.
              do J = 1, N
                 E1B(J,1)  = SS(J)
                 E1B(J,2)  = XN(J) * SPZ(J)
                 E1B(J,3)  = XN2(J) * PZPZ(J) + (YN2(J) + ZN2(J)) * PPPP(J)
                 E1B(J,4)  = YN(J) * SPZ(J)
                 E1B(J,5)  = XN(J) * YN(J) * (PZPZ(J) - PPPP(J))
                 E1B(J,6)  = YN2(J) * PZPZ(J) + (XN2(J) + ZN2(J)) * PPPP(J)
                 E1B(J,7)  = ZN(J) * SPZ(J)
                 E1B(J,8)  = XN(J) * ZN(J) * (PZPZ(J) - PPPP(J))
                 E1B(J,9)  = YN(J) * ZN(J) * (PZPZ(J) - PPPP(J))
                 E1B(J,10) = ZN2(J) * PZPZ(J) + (XN2(J) + YN2(J)) * PPPP(J)
              end do
              !
              E1B(1:N,1) = CTOEV * E1B(1:N,1)
              do I = 2,10
                 E1B(1:N,I) = CTOEV * CGL(1:N) * E1B(1:N,I)
              end do
           ENDIF           ! Integrals for atoms with more than one orbital

           ! . Calculate the nuclear/nuclear terms.
           do J = 1, N
              TEMP1 = E1B(J,1)
              TEMP2 = CGQM * CGL(J)
              TEMP3 = ABS(TEMP2) * EXP(-ALPHA * RQM(J))
              TEMP4 = ABS(TEMP2) * EXP(-ALPMM * RQM(J))
              ENUC(J)  = TEMP1 * (TEMP2 + TEMP3 + TEMP4)
           end do
           IF (AM1) THEN
              do I = 1,NTERM
                 F1 = FN1(NQM,I)
                 F2 = FN2(NQM,I)
                 F3 = FN3(NQM,I)
                 do J = 1, N
                    TEMP1 = EXP(MAX(-THIRTY,-F2*(RQM(J)-F3)**2))
                    ENUC(J)  = ENUC(J)+CGQM*CGL(J)*RQMI(J)*F1*TEMP1
                 end do
              end do
           ENDIF
           ! . SCALE THE E1B(*,1) INTEGRALS BY CGL NOW ENUC IS FINISHED.
           E1B(1:N,1) = CGL(1:N) * E1B(1:N,1)
           ! . CALCULATE THE SWITCHING CORRECTIONS AND THEIR DERIVATIVES.
           E1BA(1) = ZERO
           DO J = 1, N
              E1BA(1) = E1BA(1)+SCALE(J)*E1B(J,1)
           END DO
           IF (NATORB(NQM) .GT. 1) THEN
              do I = 2,10
                 !                    E1BA(I) = DOTVEC(SCALE,E1B(1,I),N)
                 E1BA(I) = ZERO
                 DO J = 1, N
                    E1BA(I) = E1BA(I) + SCALE(J)*E1B(J,I)
                 END DO
              end do
           ELSE
              E1BA(2:10) = ZERO
           ENDIF
#if KEY_FLUCQ==1
           ! LINES ADDED FOR FLUCQ, BEN WEBB, 2000
           ! ADD CORE TERM TO FLUCTUATING CHARGE FORCE
           DO I= 1, N
              CALL FQQCOR(JNBL(I),ENUC(I)*SCALE(I))
           ENDDO
#endif 
           !
           ! . IF INCLUDE MM-CHARGE/QM-CORE ENERGY.
           !
           IF (QEQTRM(QMCH)) THEN
              !                 ENUCLR = ENUCLR + DOTVEC(SCALE,ENUC,N)
              !                 ENUCTMP = DOTVEC(SCALE,ENUC,N)
              ENUCTMP = ZERO
              DO J = 1, N
                 ENUCTMP = ENUCTMP + SCALE(J)*ENUC(J)
              END DO
              ENUCLR = ENUCLR + ENUCTMP
              IF(QMPERT) THEN
                 ENUCP1 = ENUCP1+ENUCTMP*FACTP1
                 ENUCP2 = ENUCP2+ENUCTMP*FACTP2
              ENDIF
              ! DTM PI. Store nuclear energy for each non-PI atom, so that can be corrected for later
              IF (IQMPIATM(NUMQM) .NE. 1)  QUBENUCLR = QUBENUCLR + ENUCTMP
           ENDIF
           ! . ADD THE INTEGRALS INTO THE ONE-ELECTRON MATRIX.
           !
           ! . IF INCLUDE MM-CHARGE/QM-VALENCE OR QM ELECTRONIC ENERGY.
           !
           IF (QEQTRM(QMEE).AND..NOT.QEQTRM(QQEL)) THEN
              CALL WRNDIE(-1,'<EAMPE2>','CANNOT EVALUATE QMEE WITHOUT QQEL')
           ENDIF
           IF (QEQTRM(QMEE)) THEN
              JJ = 0
              do I = N1,N2
                 II = (I * (I - 1)) / 2 + N1 - 1
                 do J = N1,I
                    II = II + 1
                    JJ = JJ + 1
                    H(II) = H(II) - E1BA(JJ)
                    ! DTM PI save QM-MM interaction integrals
                    QUBE1BA(II) = QUBE1BA(II) - E1BA(JJ)
                    IF(QMPERT) THEN
                       HPRT1(II) = HPRT1(II)-E1BA(JJ)*FACTP1
                       HPRT2(II) = HPRT2(II)-E1BA(JJ)*FACTP2
                    ENDIF
                    !                        write(*,*)'QUBE1BA @1',II,QUBE1BA(II)
                 end do
              end do
              do I = (N2+1),N3
                 II = (I * (I + 1)) / 2
                 H(II) = H(II) - E1BA(1)
                 ! DTM PI save QM-MM interaction integrals
                 QUBE1BA(II) = QUBE1BA(II) - E1BA(1)
                 IF(QMPERT) THEN
                    HPRT1(II) = HPRT1(II)-E1BA(1)*FACTP1
                    HPRT2(II) = HPRT2(II)-E1BA(1)*FACTP2
                 ENDIF
              end do
           ENDIF
           !
        ENDIF     ! If N > 0 and QATLAB > 0
     end do loopQATOM ! QATOM = QSTRT,QSTOP
  End do loopGRP      ! QGRP = 1,NGRP

  ! DTM PI Add QM-MM terms to H for non-PI atoms that are not computed when FFOCK
  IF (FFOCK .AND. .NOT. CLAS) THEN
     IF (QEQTRM(QMEE)) THEN
        DO QATOM = 1,NATQM
           IF (IQMPIATM(QATOM) .NE. 1) THEN
              N1 = N1ST(QATOM)
              N2 = NMIDLE(QATOM)
              N3 = NLAST(QATOM)
              DO I = N1,N2
                 II = (I*(I-1))/2+N1-1
                 DO J = N1,I
                    II = II + 1
                    H(II) = H(II) + QUBE1BA(II)
                 ENDDO
              ENDDO
              DO I = (N2+1),N3
                 II    = (I*(I+1))/2
                 H(II) = H(II) + QUBE1BA(II)
              ENDDO
           ENDIF
        ENDDO
     ENDIF
     ! Add nuclear energy as well
     IF (QEQTRM(QMCH)) ENUCLR = ENUCLR + QUBENUCLR
  ENDIF

  RETURN
END SUBROUTINE QUBEAMPE2

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
SUBROUTINE QUBHCORE(COORD,H,W,WJ,WK,ENUCLR,ISTORE,FFOCK,CLAS,IQMPIATM)
  !
  use chm_kinds
  use dimens_fcm
  use number, only : zero, one, half
  !
  use stream
  use sizes
  use quantm
  implicit none
  !
  ! DTM IQMPIATM,FFOCK,CLAS variables added
  real(chm_real)  COORD(3,*),H(*), WJ(N2ELEC), WK(N2ELEC), W(N2ELEC)
  real(chm_real)  ISTORE(*)
  INTEGER IQMPIATM(*)
  LOGICAL FFOCK,CLAS
  !************************************************************************
  !
  !   HCORE GENERATES THE ONE-ELECTRON MATRIX AND TWO ELECTRON INTEGRALS
  !         FOR A GIVEN MOLECULE WHOSE GEOMETRY IS DEFINED IN CARTESIAN
  !         COORDINATES.
  !
  !  ON INPUT  COORD   = COORDINATES OF THE MOLECULE.
  !
  !  ON OUTPUT  H      = ONE-ELECTRON MATRIX.
  !             W      = TWO-ELECTRON INTEGRALS.
  !             ENUCLR = NUCLEAR ENERGY
  !***********************************************************************
  !
  real(chm_real) HALF_local,ENUC,CUTOFF,ENUCLR
  INTEGER IA,IB,JA,IC,JB,JC,II,JJ,NI,NJ,KR,IONE,IM1
  INTEGER I,J,I1,I2,IREAD,J1,J2
  !
  LOGICAL, save :: DEBUG1
  logical, save :: FIRST=.true.
  real(chm_real) E1B(10),E2A(10),DI(9,9), WJD(100), WKD(100)

  ! DTM Save Hamiltonian, 2e integrals, and nuclear energies
  INTEGER N1,N2,N3,M1,M2,M3,KK,QATOM,QATOM2
  integer, parameter :: NUMATM2=NUMATM**2, &
       NUMDIAG=MAXHEV*10+MAXLIT       ! N(N+1)/2=10

  real(chm_real) ,save :: QUBENUC,QUBH(MPACK),QUBE12AB(MPACK),E12AB(MPACK),QUBW(N2ELEC) ! COMMON /QUBHCOR/
  integer,        save :: QUBKR(NUMATM2)                                                ! COMMON /QUBHCOI/ 
  !
  !
  IONE=1
  CUTOFF=1.0D10
  !
  IF (FIRST) THEN
     FIRST=.FALSE.
     DEBUG1=(INDEX(KEYWRD,'HCORE') .NE. 0)
  ENDIF
  !
  ! This been done in qmene.src
  !     H1(1:(NORBS*(NORBS+1))/2) = zero
  !
  IREAD = 0
  ENUCLR= zero
  ! DTM PI Store 2e integral index
  KR=1
  KK=1
  ! DTM PI Initialize H and W for Fast Fock updating. Only done
  ! for centroid on first call
  IF (FFOCK) THEN
     IF (CLAS) THEN
        QUBW(1:N2ELEC)  =zero
        QUBH(1:MPACK)   =zero
        E12AB(1:MPACK)  =zero      ! 1:NUMDIAG
        QUBKR(1:NUMATM2)=zero

        QUBKR(KK) = KR
        QUBENUC   = zero
     ELSE
        QUBE12AB(1:MPACK) = zero   ! 1:NUMDIAG
     ENDIF
  ENDIF
  !
  loopII: do I = 1,NATQM
     IA=N1ST(I)
     IB=NLAST(I)
     IC=NMIDLE(I)
     NI=NAT(I)
     ! 
     ! FIRST WE FILL THE DIAGONALS, AND OFF-DIAGONALS ON THE SAME ATOM
     !
     ! DTM FFOCK
     IF (IQMPIATM(I).EQ.1 .OR. CLAS .OR. .NOT. FFOCK) THEN
        do I1=IA,IB
           I2=I1*(I1-1)/2+IA-1
           do J1=IA,I1
              I2=I2+1
              H(I2)=H(I2)+zero
           end do
           H(I2)=USPD(I1)
        end do
     ENDIF
     IM1=I-IONE

     loopJJ: do J=1,IM1

        if(I.EQ.J) then
           HALF_local=half
        else
           HALF_local=one
        end if
        JA=N1ST(J)
        JB=NLAST(J)
        JC=NMIDLE(J)
        NJ=NAT(J)
        ! DTM moved
        IREAD = IREAD + 1
        ! DTM FFOCK
        IF (.NOT.FFOCK .OR. CLAS .OR. IQMPIATM(I).EQ.1 .OR. IQMPIATM(J).EQ.1) THEN
           CALL MNH1ELEC(NI,NJ,COORD(1,I),COORD(1,J),DI)
           !
           !   FILL THE ATOM-OTHER ATOM ONE-ELECTRON MATRIX<PSI(LAMBDA)|PSI(SIGMA)>
           !
           I2=0
           do I1=IA,IB
              II=I1*(I1-1)/2+JA-1
              I2=I2+1
              J2=0
              JJ=MIN(I1,JB)
              do J1=JA,JJ
                 II=II+1
                 J2=J2+1
                 H(II)=H(II)+DI(I2,J2)
              end do
           end do
           !
           !   CALCULATE THE TWO-ELECTRON INTEGRALS, W; THE ELECTRON NUCLEAR TERMS
           !   E1B AND E2A; AND THE NUCLEAR-NUCLEAR TERM ENUC.
           !
           ! DTM moved up
           !              IREAD = IREAD + 1
           ! Update KR index of 2e array in case MNROTATE was not called on previous iteration
           IF (FFOCK .AND. .NOT.CLAS) KR = QUBKR(KK)
           CALL MNROTATE(NI,NJ,COORD(1,I),COORD(1,J),W(KR),KR,E1B,E2A,ENUC,CUTOFF,ISTORE,IREAD)
           ENUCLR = ENUCLR + ENUC
           ! DTM FFOCK Store nuclear energy
           IF (FFOCK .AND. CLAS) THEN
              IF (IQMPIATM(I).EQ.0 .AND. IQMPIATM(J).EQ.0) QUBENUC = QUBENUC + ENUC
           ENDIF
           !
           !   ADD ON THE ELECTRON-NUCLEAR ATTRACTION TERM FOR ATOM I.
           !   DTM. These terms also enter diagonal elements
           !
           ! DTM FFOCK
           I2=0
           do I1=IA,IC
              II=I1*(I1-1)/2+IA-1
              do J1=IA,I1
                 II=II+1
                 I2=I2+1
                 IF (IQMPIATM(I).EQ.0.AND.IQMPIATM(J).EQ.1) THEN
                    IF (CLAS) E12AB(II) = E12AB(II) + E1B(I2)*HALF_local
                    QUBE12AB(II) = QUBE12AB(II) + E1B(I2)*HALF_local
                 ENDIF
                 H(II)=H(II)+E1B(I2)*HALF_local
              end do
           end do
           ! d-orbital part not tested!
           do I1=IC+1,IB
              II=(I1*(I1+1))/2
              IF (IQMPIATM(I).EQ.0.AND.IQMPIATM(J).EQ.1) THEN
                 IF (CLAS) E12AB(II) = E12AB(II) + E1B(1)*HALF_local
                 QUBE12AB(II) = QUBE12AB(II) + E1B(1)*HALF_local
              ENDIF
              H(II)=H(II)+E1B(1)*HALF_local
           end do
           !
           !   ADD ON THE ELECTRON-NUCLEAR ATTRACTION TERM FOR ATOM J.
           !
           ! DTM FFOCK
           I2=0
           do I1=JA,JC
              II=I1*(I1-1)/2+JA-1
              do J1=JA,I1
                 II=II+1
                 I2=I2+1
                 IF (IQMPIATM(I).EQ.1.AND.IQMPIATM(J).EQ.0) THEN
                    IF (CLAS) E12AB(II) = E12AB(II) + E2A(I2)*HALF_local
                    QUBE12AB(II) = QUBE12AB(II) + E2A(I2)*HALF_local
                 ENDIF
                 H(II)=H(II)+E2A(I2)*HALF_local
              end do
           end do
           ! d-orbital part not tested!
           do I1=JC+1,JB
              II=(I1*(I1+1))/2
              IF (IQMPIATM(I).EQ.1.AND.IQMPIATM(J).EQ.0) THEN
                 IF (CLAS) E12AB(II) = E12AB(II) + E2A(1)*HALF_local
                 QUBE12AB(II) = QUBE12AB(II) + E2A(1)*HALF_local
              ENDIF
              H(II)=H(II)+E2A(1)*HALF_local
           end do
        ENDIF
        ! DTM PI Store 2e integral index
        KK = KK + 1
        IF (FFOCK .AND. CLAS) QUBKR(KK) = KR
     end do loopJJ
  end do loopII


  ! DTM PI Save H and W for Fast Fock updating. Only done
  ! for centroid during first call.
  IF (FFOCK .AND. CLAS) THEN
     QUBW(1:N2ELEC) = WJ(1:N2ELEC)
     QUBH(1:(NORBS*(NORBS+1))/2) = H(1:(NORBS*(NORBS+1))/2)
  ENDIF
  ! DTM PI Transfer saved H and W 
  KK = 0
  IF (FFOCK .AND. .NOT. CLAS) THEN
     DO QATOM = 1,NATQM
        ! Transfer diagonal terms
        ! Must account for E1B and E2A diagonal terms, which depend on cross terms
        N1 = N1ST(QATOM)
        N2 = NMIDLE(QATOM)
        N3 = NLAST(QATOM)
        IF (IQMPIATM(QATOM).EQ.0) THEN
           DO I = N1,N2
              II = (I * (I - 1)) / 2 + N1 - 1
              DO J = N1,I
                 II = II + 1
                 H(II) = QUBH(II) + QUBE12AB(II) - E12AB(II)
              ENDDO
           ENDDO
           ! d-orbital part not tested!
           DO I = (N2+1),N3
              II = (I * (I + 1)) / 2
              H(II) = QUBH(II) + QUBE12AB(II) - E12AB(II)
           ENDDO
        ENDIF
        DO QATOM2 = 1,QATOM - 1
           KK = KK + 1
           ! Transfer off-diagonal terms
           IF (IQMPIATM(QATOM).EQ.0.AND.IQMPIATM(QATOM2).EQ.0) THEN
              M1 = N1ST(QATOM2)
              M2 = NMIDLE(QATOM2)
              M3 = NLAST(QATOM2)
              DO I = N1,N2
                 II = (I * (I - 1)) / 2 + M1 - 1
                 JJ = MIN(I,M1)
                 DO J = M1,M2 !JJ
                    II = II + 1
                    H(II) = QUBH(II)
                 ENDDO
              ENDDO
              ! d-orbital part not tested!
              DO I = (N2+1),N3
                 II = (I * (I + 1)) / 2
                 H(II) = QUBH(II)
              ENDDO
              ! Transfer 2e integrals from interactions between non-PI atoms
              WJ(QUBKR(KK):QUBKR(KK+1)-1) = QUBW(QUBKR(KK):QUBKR(KK+1)-1)
           ENDIF
        ENDDO
     ENDDO
     ! DTM PI Add up nuclear-nuclear terms from interactions between non-PI atoms
     ENUCLR = ENUCLR + QUBENUC
  ENDIF

  IF( .NOT. DEBUG1) RETURN
  IF(PRNLEV.GE.2) WRITE(6,'(//10X,''ONE-ELECTRON MATRIX FROM HCORE'')')
  CALL MNVECPRT(H,NORBS)
  J=MIN(400,KR)
  IF(PRNLEV.GE.2) THEN
     WRITE(6,'(//10X,''TWO-ELECTRON MATRIX IN HCORE''/)')
     WRITE(6,120)(W(I),I=1,J)
  END IF
120 FORMAT(10F8.4)
  RETURN
END SUBROUTINE QUBHCORE

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
SUBROUTINE QUBDQLINK4(X,Y,Z,F,UHF,ECLASS)
  !------------------------------------------------------------------------
  !   Computes the derivatives of the density matrix as a result of
  !   the transformation from hybrid orbitals to atomic orbitals basis.
  !   Updates nucleus force arrays, and computes a classical energy
  !   contribution from MM atoms directly connected to the QM boundary atom.
  !
  !   J. Gao, P. Amara, C. Alhambra, M. J. Field
  !   J. Phys. Chem. A, 102, 4714 (1998).
  !
  use chm_kinds
  use dimens_fcm
  use sizes
  use qmlinkm
  use quantm
  use consta
  use psf
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*),F(*),ECLASS
  LOGICAL UHF
  !  Local variables
  INTEGER I,M,M1,N,JJ,JJ2
  real(chm_real) DELXYZ(3),RR2,RR1,FACT1

  !
  !      INCLUDE NUCLEAR INTERACTIONS BETWEEN MM-BOUNDARY ATOMS
  !
  !
  !      LOOP OVER QM-LINK ATOMS
  !
  loopII: DO I = 1,MQMLNK
     IF (UHF) cycle loopII  ! avoid double count in beta correction UHF case 
     DO M = 1,2
        JJ = JMQLINK(M,I)
        M1 = M+1
        DO N = M1,3
           JJ2 = JMQLINK(N,I)
           DELXYZ(1) = X(JJ2)-X(JJ)
           DELXYZ(2) = Y(JJ2)-Y(JJ)
           DELXYZ(3) = Z(JJ2)-Z(JJ)
           RR2 = DELXYZ(1)*DELXYZ(1)+DELXYZ(2)*DELXYZ(2)+DELXYZ(3)*DELXYZ(3)
           RR1 = SQRT(RR2)
           !------------------------------------
           IF(QMPERT) THEN
              FACT1 = CCELEC*CG(JJ)*RLAMB0*CG(JJ2)*RLAMB0/RR1
           ELSE
              FACT1 = CCELEC*CG(JJ)*CG(JJ2)/RR1
           ENDIF
           !------------------------------------
           ECLASS = ECLASS+FACT1
           FACT1  = FACT1/RR2
        ENDDO
     ENDDO
  ENDDO loopII
  
  return
END SUBROUTINE QUBDQLINK4
  

#endif 
SUBROUTINE NULL_QUBPAC
  RETURN
END SUBROUTINE NULL_QUBPAC

