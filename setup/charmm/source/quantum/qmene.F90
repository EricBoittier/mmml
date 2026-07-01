#if KEY_QUANTUM==1
SUBROUTINE EAMPAC (EQUANT,X,Y,Z,DX,DY,DZ)
  !------------------------------------------------------------------------
  !     Calculate the energy for the quantum mechanical atoms and their
  !     electrostatic interactions with the MM atoms using the AM1 or
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
  !
  ! namkh 09/25/04
  !     Adjust nonbonded group list for IMAGES.
  use image
  !     Adjust nonbonded group list for simple pbc.
#if KEY_PBOUND==1
  use pbound  
#endif
  implicit none
  !
  real(chm_real)   equant,x(*),y(*),z(*),dx(*),dy(*),dz(*)
  !
  !========================================================================
  !     Calculate the quantum mechanical energy.
  !========================================================================
  ! BIOVIA Code Start : Bug fix
  ! Exit if NULL Hamiltonian was specified in quantum command
  ! (leave gradients untouched then).
  if (norbs.eq.0) then
     equant    = 0.0D0
     return
  endif
  ! BIOVIA Code End
  !
  if(natqm_check.ne.natqm) then
     if(allocated(coorqm)) call chmdealloc('qmene.src','EAMPAC','coorqm',3,size(coorqm,2),crl=coorqm)
     if(allocated(gradqm)) call chmdealloc('qmene.src','EAMPAC','gradqm',3,size(gradqm,2),crl=gradqm)
     natqm_check=natqm
  end if
  if(natom_check.ne.natom) then
     if(allocated(dxm_qmmm)) call chmdealloc('qmene.src','EAMPAC','dxm_qmmm',size(dxm_qmmm),crl=dxm_qmmm)
     if(allocated(xim)) call chmdealloc('qmene.src','EAMPAC','xim',size(xim),crl=xim)
     if(allocated(yim)) call chmdealloc('qmene.src','EAMPAC','yim',size(yim),crl=yim)
     if(allocated(zim)) call chmdealloc('qmene.src','EAMPAC','zim',size(zim),crl=zim)
     natom_check=natom
  end if
  if(.not.allocated(coorqm))  call chmalloc('qmene.src','EAMPAC','coorqm',3,natqm,crl=coorqm)
  if(.not.allocated(gradqm))  call chmalloc('qmene.src','EAMPAC','gradqm',3,natqm,crl=gradqm)
  if(.not.allocated(dxm_qmmm))call chmalloc('qmene.src','EAMPAC','dxm_qmmm',3*natom,crl=dxm_qmmm)
  if(.not.allocated(xim)) call chmalloc('qmene.src','EAMPAC','xim',natom,crl=xim)
  if(.not.allocated(yim)) call chmalloc('qmene.src','EAMPAC','yim',natom,crl=yim)
  if(.not.allocated(zim)) call chmalloc('qmene.src','EAMPAC','zim',natom,crl=zim)

  ! allocate memory for WJ_anal and WK_anal
  if(N2ELEC .gt. size(WJ_anal) .or. N2ELEC .gt. size(WK_anal)) then
     if(allocated(WJ_anal)) call chmdealloc('qmene.src','EAMPAC','WJ_anal',size(WJ_anal),crl=WJ_anal)
     if(allocated(WK_anal)) call chmdealloc('qmene.src','EAMPAC','WK_anal',size(WK_anal),crl=WK_anal)
  end if
  if(.not.allocated(WJ_anal)) call chmalloc('qmene.src','EAMPAC','WJ_anal',N2ELEC+50,crl=WJ_anal)
  if(.not.allocated(WK_anal)) call chmalloc('qmene.src','EAMPAC','WK_anal',N2ELEC+50,crl=WK_anal)
  !

  !
  !=======================================================================
  !     Image Swap for IMAGE case.
  !=======================================================================
  Call SwapXYZ_image(natom,x,y,z,xim,yim,zim,IMATTQ)
  !=======================================================================
  !
  Call Eampc2 (equant,x,y,z,dx,dy,dz,xim,yim,zim,coorqm,gradqm,dxm_qmmm)
  !
  Return
END SUBROUTINE EAMPAC

SUBROUTINE EAMPC2 (EQUANT,X,Y,Z,DX,DY,DZ,XI,YI,ZI,COORD,GRAD,DXM)
  !------------------------------------------------------------------------
  !     Does the work of EAMPAC.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use memory
  !
  use number
  use consta, only : EV_TO_KCAL
  use contrl
  !...  use coord
  !...  use deriv
  use inbnd
  use nbndqm_mod
  use quantm
  use scfblk
  use psf
  use sizes
  use qmlinkm
  use leps
#if KEY_PARALLEL==1
  use parallel       
#endif
#if KEY_PARALLEL==1 && KEY_MPI==1
  use mpi            
#endif
  use memory
  !
  ! for parallel run
  implicit none
#if KEY_PARALLEL==1
#if KEY_MPI==1
  integer ierr
#endif 
#endif 
  !
  real(chm_real)  equant, x(*),y(*),z(*), dx(*),dy(*),dz(*)
  real(chm_real)  coord(3,*), grad(3,*), xi(*),yi(*),zi(*),dxm(*)
  !
  real(chm_real) E_LEPS,DA_LEPS_local(3),DB_LEPS_local(3),DC_LEPS_local(3), &
       DD_LEPS_local(3),DE_LEPS_local(3),DF_LEPS_local(3)
  !
  Integer Index,INDX1E
  !
  !CC   real(chm_real)  dxm(3*natom)
  real(chm_real)  enuclr2,ENUCPE1,ENUCPE2
  !
  Integer i, iatom, j, ndim, nlen, nlenb, nlenx, nqm, num_qm_grp
  Integer NQMPR, NMAX,jatom
  !
  real(chm_real),allocatable,dimension(:) :: istore
  !
  ! go parallel
  INTEGER ATFRST,ATLAST,NORBL,natom2,ncount
  real(chm_real)  equant2
  !
  !DCC-
  !     The following arrays allow the product of the one-electron integrals
  !     for QM/MM charge/valence interations and the derivative of the
  !     switching function, calculated in EAMPE2, to be passed to
  !     EAMPCG.  DCC 2.22.93
  !     
  !     The following variables are defined in scfblk_ltm.src
!!!!      Integer e1bdxs, e1bdys, e1bdzs, e1xdxs, e1xdys, e1xdzs
  !DCC+
  !
  !========================================================================
  real(chm_real) ECLASS
  real(chm_real) ETPER0,ETPER1,ETPER2,ELECP1,ELECP2,ELEGAS
  real(chm_real) EVER,ESTA,EDIS,EPOL
  INTEGER LENPAB
  !
  !      Logical debug1
#if KEY_PARALLEL==1
  real(chm_real)  gradx(natqm*3)       
#endif
  !========================================================================
  !  
  ! Parallel initial check
#if KEY_PARALLEL==1
  IF((NUMNOD+1).GT.NATQM .AND. .NOT. QMPI) THEN
     CALL WRNDIE(-5,'<EAMPC2>', &
          'Number of QM atoms should be less than number of CPU.')
  END IF
#endif 
  !
  equant    = ZERO
  enuclr_qm = ZERO
  equant2   = ZERO
  enuclr2   = ZERO
  elect_qm  = ZERO
  INDX1E = 0
  nqm = 0
  natom2 = 2*natom
  ncount = 1
  !
  Do iatom = 1,natom
     If (qatlab(iatom) .ge. 0) then
        nqm = nqm + 1
        coord(1,nqm) = x(iatom)
        coord(2,nqm) = y(iatom)
        coord(3,nqm) = z(iatom)
        INDX1E = INDX1E+1
        If (qatlab(iatom) .gt. 1) INDX1E = INDX1E+9
     Endif
  End do

  call zerotm(dxm,3*natom)
  !
  !------------------------------------
  !     Check whether we needs to regenerate initial guess again.
  !
  IF(DYNAMQ.AND.(NEWGS.GT.0)) THEN
     NSTPMD=NSTPMD+1
     IF(MOD(NSTPMD,NEWGS).EQ.0) THEN
        CALL GESDEN(.FALSE.)
        !     If there is any GHO atoms? ===> Ask Pu. How to redo the initial guess.
        !     Currently, it will be turned off with GHO method.
        NSTPMD=0
     END IF
  END IF
  !
  ! namkh 08/08/04
  ! QM/MM-Ewald
  IF(LQMEWD) THEN
     ! Allocate array for kspace gradient..tricky.
     If(QFIRSTD .and. allocated(PDXYZ)) call chmdealloc('qmene.src','EAMPC2','PDXYZ',3,size(PDXYZ,2),crl=PDXYZ)
     ! Do the initial setup
     CALL SETUP_QMEWD
     ! Prepare Ewald summation
     if( allocated(PKTABXCQ) .and. NATOM*MAXKVQ.ne.size(PKTABXCQ) ) then
        if(allocated(PKTABXCQ)) call chmdealloc('qmene.src','EAMPC2','PKTABXCQ',size(PKTABXCQ),crl=PKTABXCQ)
        if(allocated(PKTABXSQ)) call chmdealloc('qmene.src','EAMPC2','PKTABXSQ',size(PKTABXSQ),crl=PKTABXSQ)
        if(allocated(PKTABYCQ)) call chmdealloc('qmene.src','EAMPC2','PKTABYCQ',size(PKTABYCQ),crl=PKTABYCQ)
        if(allocated(PKTABYSQ)) call chmdealloc('qmene.src','EAMPC2','PKTABYSQ',size(PKTABYSQ),crl=PKTABYSQ)
        if(allocated(PKTABZCQ)) call chmdealloc('qmene.src','EAMPC2','PKTABZCQ',size(PKTABZCQ),crl=PKTABZCQ)
        if(allocated(PKTABZSQ)) call chmdealloc('qmene.src','EAMPC2','PKTABZSQ',size(PKTABZSQ),crl=PKTABZSQ)
     end if
     if(.not.allocated(PKTABXCQ)) call chmalloc('qmene.src','EAMPC2','PKTABXCQ',NATOM*MAXKVQ,crl=PKTABXCQ)
     if(.not.allocated(PKTABXSQ)) call chmalloc('qmene.src','EAMPC2','PKTABXSQ',NATOM*MAXKVQ,crl=PKTABXSQ)
     if(.not.allocated(PKTABYCQ)) call chmalloc('qmene.src','EAMPC2','PKTABYCQ',NATOM*MAXKVQ,crl=PKTABYCQ)
     if(.not.allocated(PKTABYSQ)) call chmalloc('qmene.src','EAMPC2','PKTABYSQ',NATOM*MAXKVQ,crl=PKTABYSQ)
     if(.not.allocated(PKTABZCQ)) call chmalloc('qmene.src','EAMPC2','PKTABZCQ',NATOM*MAXKVQ,crl=PKTABZCQ)
     if(.not.allocated(PKTABZSQ)) call chmalloc('qmene.src','EAMPC2','PKTABZSQ',NATOM*MAXKVQ,crl=PKTABZSQ)

     if(.not.allocated(PDXYZ))    call chmalloc('qmene.src','EAMPC2','PDXYZ',3,NATOM,crl=PDXYZ)
     !
     CALL SETUP_KTABLE(NATOM,X,Y,Z,MAXKVQ,KMAXXQ,KMAXYQ,KMAXZQ, &
          PKTABXCQ,PKTABXSQ,PKTABYCQ,PKTABYSQ,PKTABZCQ,PKTABZSQ)
     ! Reset MMINB array for QM/MM indexing
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
  grad(1:3,1:natqm) = zero
  !
  ! go parallel
  NORBL = (NORBS*(NORBS+1))/2
  !
  !========================================================================
  !     Determine some options.
  !========================================================================
  !     debug1 = Index(keywrd,'DEBUG') .ne. 0
  !========================================================================
  !     Calculate the QM integrals.
  !========================================================================
  ndim   = 22 * (natqm * (natqm - 1))
  call chmalloc('qmene.src','EAMPC2','istore',ndim,crl=istore)
  ! go parallel
  call zerotm(h_matrix,NORBL) ! mpack)
  call zerotm(wj_anal,n2elec)

  IF(QMPERT) THEN
     call zerotm(H0GAS, LINQLK)
     call zerotm(H1PERT,LINQLK)
     call zerotm(H2PERT,LINQLK)
  END IF
  IF(QDECOM) THEN
     call zerotm(H0GAS,LINQLK)
  END IF

  Call MNHcore (coord,H_matrix,wj_anal,wj_anal,wk_anal,enuclr_qm,istore)
  !-------------------------------------
  !   QM-MM electrostatic free energy perturbation calculations
  !   JG 7/9/99
  !
  IF(QMPERT) THEN
     h0gas (1:linqlk) = H_matrix(1:LINQLK)
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
  ! If do not include QM core-core repulsion energy.
  !
  If (.not.qeqtrm(qqcc)) enuclr_qm = ZERO
  !DCC+
  !========================================================================
  !     Calculate the one-electron interaction integrals.
  !========================================================================
  If (nqmgpe .gt. 0 .or. nqmgex .gt. 0) then
     ! . Allocate space.
     if(ngrp_old.ne.ngrp) then
        if(allocated(xyzg))  call chmdealloc('qmene.src','EAMPC2','xyzg',size(xyzg,1),3,crl=xyzg)
        if(allocated(igrpg)) call chmdealloc('qmene.src','EAMPC2','igrpg',size(igrpg),intg=igrpg)
        if(allocated(qgrpmm))call chmdealloc('qmene.src','EAMPC2','qgrpmm',size(qgrpmm),log=qgrpmm)
        ngrp_old=ngrp
     end if
     if(.not.allocated(xyzg))  call chmalloc('qmene.src','EAMPC2','xyzg',ngrp,3,crl=xyzg)
     if(.not.allocated(igrpg)) call chmalloc('qmene.src','EAMPC2','igrpg',ngrp,intg=igrpg)
     if(.not.allocated(qgrpmm))call chmalloc('qmene.src','EAMPC2','qgrpmm',ngrp,log=qgrpmm)

     ! . Calculate the group centres of geometry.
     Call Grpcen(ngrp,igpbs,xi,yi,zi,xyzg(1:ngrp,1),xyzg(1:ngrp,2),xyzg(1:ngrp,3))

     ! . Fill the QM group array.
     Call Grpmm (ngrp,natom,igpbs,natqm,num_qm_grp,qatlab,qgrpmm)
     ! . Calculate the QM/MM interaction terms.
     If (nqmgpe .gt. 0) then
        call nampe(iqmgpe,jqmgpe,nlenb,nmax,NQMPR)
        nlen = nlenb * 10
        call setup_scf_arrays(nlen,natom-natqm,INDX1E,NMAX,.false.) ! allocate memory

        CALL ZEROTM(dxe1bq,INDX1E)
        CALL ZEROTM(dye1bq,INDX1E)
        CALL ZEROTM(dze1bq,INDX1E)
        CALL ZEROTM(e1bdxs,INDX1E*NMAX)
        CALL ZEROTM(e1bdys,INDX1E*NMAX)
        CALL ZEROTM(e1bdzs,INDX1E*NMAX)
        !JG
        Call Eampce(coord,dxm,enuclr_qm, xi,yi,zi, H_matrix, &
             iqmgpe, jqmgpe, e14fac, nlenb, &
             dxe1bm,dye1bm,dze1bm,dxe1bq,dye1bq,dze1bq,e1bdxs,e1bdys,e1bdzs, &
             xyzg(1:ngrp,1),xyzg(1:ngrp,2),xyzg(1:ngrp,3), &
             qgrpmm,indx1e,NQMPR)
        !
     Endif
     ! . Calculate the QM/MM exclusion terms.
     If (nqmgex .gt. 0) then
        call nampe(iqmgex,jqmgex,nlenx,nmax,NQMPR)
        nlen = nlenx * 10
        call setup_scf_arrays(nlen,natom-natqm,q_exclusion=.true.)    ! allocate memory
        !
        Call Eampce (coord,dxm,enuclr_qm,xi,yi,zi, H_matrix, iqmgex, &
             jqmgex, sft123, nlenx, &
             dxe1xm,dye1xm,dze1xm,dxe1xq,dye1xq,dze1xq,e1xdxs,e1xdys,e1xdzs, &
             xyzg(1:ngrp,1),xyzg(1:ngrp,2),xyzg(1:ngrp,3), &
             qgrpmm,indx1e,NQMPR)
     Endif
  Endif
  ! namkh 08/08/04
  ! QM/MM-Ewald
  ! Compute the Ewald potential on QM atoms from all the MM atoms
  IF(LQMEWD) CALL QEWALDP(X,Y,Z,XI,YI,ZI,.FALSE.)
  !
  ! go parallel
  ! . now synchronize H array
#if KEY_PARALLEL==1 /*paramain*/
  IF(NUMNOD.NE.1 .AND. .NOT. QMPI) THEN
     call psync()
     call GCOMB(H_matrix,norbl)
     call GCOMB(WJ_anal,n2elec)

     IF(QMPERT) THEN
        call GCOMB(H0GAS,norbl)
        call GCOMB(H1PERT,norbl)
        call GCOMB(H2PERT,norbl)
     END IF
     IF(QDECOM) THEN
        call GCOMB(H0GAS,norbl)
     END IF
  END IF
#endif /* (paramain)*/
  !
  !
  !========================================================================
  !     Do the SCF.
  !========================================================================
  !------------------------------------------------------------------------
  !   QM-MM electrostatic free energy perturbation calculations
  !   JG 7/9/99
  !
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
     CALL ITER(H_matrix,WJ_anal,WJ_anal,WK_anal,X,Y,Z,XI,YI,ZI,ENUCLR_qm,ELECT_qm, .TRUE.)
     penzym(1:linqlk) = PDENS(1:LINQLK)
  ELSE
     Call Iter(H_matrix,wj_anal,wj_anal,wk_anal,X,Y,Z,XI,YI,ZI,enuclr_qm,elect_qm,.true.)
  ENDIF
  !------------------------------------------------------------------------
  !
  ! namkh 08/08/04
  ! QM/MM-Ewald
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
  ! go parallel
#if KEY_PARALLEL==1
  IF(NUMNOD.NE.1 .AND. .NOT. QMPI) THEN
     call GCOMB(ENUCLR_qm,1)
     IF(QMPERT) THEN
        call GCOMB(ENUGAS,1)
        call GCOMB(ENUCP1,1)
        call GCOMB(ENUCP2,1)
     END IF
     IF(QDECOM) THEN
        call GCOMB(ENUGAS,1)
     END IF
  END IF
#endif 
  !
#if KEY_FLUCQ==1
  CALL FQQMMM (PDENS, coord, iqmgpe(1),jqmgpe(1), e14fac, &
       xyzg(1:ngrp,1),xyzg(1:ngrp,2),xyzg(1:ngrp,3),qgrpmm)
#endif 
  equant = ZERO
  !
  ! namkh 08/08/04
  ! QM/MM-Ewald
#if KEY_PARALLEL==1
  ! For PI, every node must do this
  IF(MYNOD.EQ.0 .OR. QMPI) THEN
#endif 
     !        enuclr_qm = enuclr_qm
     !
     !
     !     If include QM/QM electronic energy (and MM-charge/QM-valence energy).
     !
     !DCC#      If (qeqtrm(qqel)) equant = elect_qm
     !DCC#      equant = (equant + enuclr_qm) * EV_TO_KCAL + atheat  ! 23.061D0
     !DCC-
     If (qeqtrm(qqel).or.qeqtrm(qmee)) equant = elect_qm
     !
     !     If include QM core-core repulsion.
     !
     !     The following line was changed to eliminate the IF determination
     !     because the QQCC determination was moved to subroutine ROTATE.
     !     Also, QMCH has to be evaluated elsewhere.
     !
     !      If (qeqtrm(qqcc)) equant = equant + enuclr_qm
     !       write(*,*) 'Ene debug',ELECT_qm,ENUCLR_qm,ATHEAT
     equant = equant + enuclr_qm
     equant = equant * 23.061D0   ! 23.061D0
     !
     ! If include the atomic  heat of formation.
     !
     If (qeqtrm(qath)) equant = equant + atheat
#if KEY_PARALLEL==1
  END IF
#endif 
  !DCC+
  !
  !========================================================================
  !     Calculate the QM first derivatives.
  !========================================================================
  Call MNDcart (coord,grad,pdens,pdensa,pdensb,uhf,istore)
  !
  grad(1:3,1:natqm)=-half*grad(1:3,1:natqm)
  !
  !-----------------------------------------------------------------------
  ! start: leps potential correction
  !
  IF(QLEPS) THEN
     !
     ! assign coordinates
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
     ! start: SVB energy term
     IF(QSVB) THEN
        IF (SVB_DIM .EQ. 1) THEN
           CALL QM_SVB1D(E_LEPS,DA_LEPS_local,DB_LEPS_local,DC_LEPS_local)   
        ENDIF
        IF (SVB_DIM .EQ. 2) THEN
           CALL QM_SVB2D(E_LEPS,DA_LEPS_local,DB_LEPS_local,DC_LEPS_local, &
                DD_LEPS_local,DE_LEPS_local,DF_LEPS_local)
        ENDIF
     ELSE
        CALL CORRECT_LEPS(E_LEPS,DA_LEPS_local,DB_LEPS_local,DC_LEPS_local)
     END IF
     ! energy correction
     !
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.0 .OR. QMPI) THEN
#endif 
        EQUANT = EQUANT + E_LEPS
        ! gradient correction
        ncount = 3*(nta-1)+1
        dxm(ncount:ncount+2)      = dxm(ncount:ncount+2)     +DA_LEPS_local(1:3)
        !
        ncount = 3*(ntb-1)+1
        dxm(ncount:ncount+2)      = dxm(ncount:ncount+2)     +DB_LEPS_local(1:3)
        !
        ncount = 3*(ntc-1)+1
        dxm(ncount:ncount+2)      = dxm(ncount:ncount+2)     +DC_LEPS_local(1:3)

        IF (SVB_DIM .EQ. 2) THEN
           ncount = 3*(ntd-1)+1
           dxm(ncount:ncount+2)   = dxm(ncount:ncount+2)     +DD_LEPS_local(1)
           ! 
           ncount = 3*(nte-1)+1
           dxm(ncount:ncount+2)   = dxm(ncount:ncount+2)     +DE_LEPS_local(1:3)
           !
           if(.not.SVB_DE) then
              ncount = 3*(ntf-1)+1
              dxm(ncount:ncount+2)= dxm(ncount:ncount+2)     +DF_LEPS_local(1:3)
           end if
        ENDIF
#if KEY_PARALLEL==1
     END IF
#endif 
  END IF
  !
  ! end: leps potential correction
  !------------------------------------------------------------------------
  !
  !      DETERMINE THE Q-LINK ATOM DERIVATIVES
  !
  IF(QMLINK) THEN
     ECLASS = ZERO
     !
     ! Treat RHF and UHF differently ... PJ 12/2002
     !
     IF (.NOT. UHF) THEN
        CALL DQLINK4(DXM,X,Y,Z,FA,UHF,ECLASS)
#if KEY_PARALLEL==1
        IF(MYNOD.EQ.0 .OR. QMPI) THEN
#endif 
           EQUANT = EQUANT + ECLASS
#if KEY_PARALLEL==1
        END IF
#endif 
     END IF
     !
     ! Include the gradient correction term for alpha and beta FOCK
     ! matrices seperately. The repulsion energy between MM atoms
     ! linked to the GHO boundary is included when the alpha
     ! correction term is included ... PJ 12/2002
     !
     IF (UHF) THEN
        CALL DQLINK4(DXM,X,Y,Z,FA,.false.,ECLASS)
#if KEY_PARALLEL==1
        IF(MYNOD.EQ.0 .OR. QMPI) THEN
#endif 
           EQUANT = EQUANT + ECLASS
#if KEY_PARALLEL==1
        END IF
#endif 
        CALL DQLINK4(DXM,X,Y,Z,FB,UHF,ECLASS)
     END IF
     !
  ENDIF
  !------------------------------------------------------------------------
  IF(QMPERT) THEN
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.0 .OR. QMPI) THEN
#endif 
        EEQTRM(QPT1) = (ELECP1+ENUCP1)*EV_TO_KCAL +ECLASS*FACTP1 +ATHEAT
        EEQTRM(QPT2) = (ELECP2+ENUCP2)*EV_TO_KCAL +ECLASS*FACTP2 +ATHEAT
        IF(LQMEWD) THEN
           EEQTRM(QPT1) = EEQTRM(QPT1)+(EBKCHG(1)-EBKCHG(2))
           EEQTRM(QPT2) = EEQTRM(QPT2)+(EBKCHG(1)-EBKCHG(3))
        END IF
#if KEY_PARALLEL==1
     END IF
#endif 
  ENDIF
  IF(QDECOM) THEN
     CALL QMDCOM(PGAS,PENZYM,H0GAS,H_matrix,H1PERT, &
          EQUANT,ELECT_qm,ENUCLR_qm,ELEGAS,ENUGAS,ECLASS,ATHEAT, &
          EEQTRM(QVER),EEQTRM(QSTA),EEQTRM(QDIS),EEQTRM(QPOL), &
          NORBS,LINQLK)
     EEQTRM(QGAS) = (ELEGAS+ENUGAS)*EV_TO_KCAL+ATHEAT+ECLASS
  ENDIF
  !========================================================================
  !     Calculate the QM/MM first derivatives.
  !========================================================================
  If (nqmgpe .gt. 0 .or. nqmgex .gt. 0) then
     !
     !   for later expansion.. -namkh 09/25/04
     !   however, currently, it only support minimum image QM/MM interactions.
     !   refer. /quantum/qmnbnd.src
     !
     ! . Calculate the QM/MM interaction terms.
     If (nqmgpe .gt. 0) then
        Call Eampcg(dxm, iqmgpe, jqmgpe, jnbl_qmmm, &
             nlenb,uhf,grad,pdens,pdensa,pdensb, &
             dxe1bm,dye1bm,dze1bm,dxe1bq,dye1bq,dze1bq,e1bdxs,e1bdys,e1bdzs, &
             xyzg(1:ngrp,1),xyzg(1:ngrp,2),xyzg(1:ngrp,3),qgrpmm,INDX1E)
     Endif
     ! . Calculate the QM/MM exclusion terms.
     If (nqmgex .gt. 0) then
        Call Eampcg (dxm, iqmgex, &
             jqmgex, jnbl_qmmm, &
             nlenx, uhf, grad, pdens, pdensa, pdensb, &
             dxe1xm,dye1xm,dze1xm,dxe1xq,dye1xq,dze1xq,e1xdxs,e1xdys,e1xdzs, &
             xyzg(1:ngrp,1),xyzg(1:ngrp,2),xyzg(1:ngrp,3),qgrpmm,INDX1E)
     Endif
  Endif
  !
  !========================================================================
  ! namkh 08/08/04
  ! QM/MM-Ewald
  IF(LQMEWD) THEN
     !
     !     Compute and put the derivative from Ewald potential
     call QEWALDD(NATOM,NATQM,X,Y,Z,XI,YI,ZI,DXM,GRAD,PDXYZ)
     !
     ! For charge of ions..This needs to be done in case of QMPERT for consistency
     ! Copying back to original scaled charges
     ! no need...
     !        IF(QMPERT.AND.QCGSC) THEN
     !           DO I=1,NUMCGS
     !              CG(NATMCGS(I))= CGSCLE(I)*RLAMB0
     !           END DO
     !        END IF
  END IF
  !
  ! Now copying the grad into gradx array
  ! For QM derivatives...
  !
  !========================================================================
  !     Put the QM derivatives into the full derivative arrays.
  !========================================================================
  !
  ! go parallel
#if KEY_PARALLEL==1 /*paramain*/
  IF(NUMNOD.NE.1 .AND. .NOT. QMPI) THEN
     ncount   = 1
     do i=1,natqm
        gradx(ncount)   = grad(1,i)
        gradx(ncount+1) = grad(2,i)
        gradx(ncount+2) = grad(3,i)
        ncount          = ncount + 3
     end do
     call psync()
     call GCOMB(gradx,3*natqm)
     call GCOMB(dxm,3*natom)
  END IF
  !
#if KEY_PARAFULL==1 /*parfmain*/
  ATFRST=1+IPARPT(MYNOD)
  ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
  ATFRST=1
  ATLAST=NATOM
#endif /* (parfmain)*/
  IF (QMPI) THEN
     ATFRST=1
     ATLAST=NATOM
  ENDIF
#else /* (paramain)*/
  ATFRST=1
  ATLAST=NATOM
#endif /*  (paramain)*/
  !
  !DCC#     If include QM electronic energy.
  !
  !DCC#      If (qeqtrm(qqel)) then
#if KEY_PARALLEL==1
  IF(NUMNOD.NE.1 .AND. .NOT. QMPI) THEN
     ncount   = 1
     Do iatom=1,natom
        If(qatlab(iatom).ge.0) then
           If(iatom.ge.atfrst.and.iatom.le.atlast) then
              dx(iatom) = dx(iatom) + gradx(ncount)   
              dy(iatom) = dy(iatom) + gradx(ncount+1) 
              dz(iatom) = dz(iatom) + gradx(ncount+2) 
           End if
           ncount       = ncount + 3
        End if
     End do
  ELSE
#endif 
     nqm = 0
     Do iatom = 1,natom
        If (qatlab(iatom) .ge. 0) then
           nqm = nqm + 1
           dx(iatom) = dx(iatom) + grad(1,nqm)
           dy(iatom) = dy(iatom) + grad(2,nqm)
           dz(iatom) = dz(iatom) + grad(3,nqm)
        Endif
     enddo
#if KEY_PARALLEL==1
  END IF
#endif 

  ! debug
  !     ncount   = 1
  !
  ncount   = 3*(atfrst-1) + 1
  Do iatom = atfrst,atlast
     dx(iatom) = dx(iatom) + dxm(ncount)
     dy(iatom) = dy(iatom) + dxm(ncount+1)
     dz(iatom) = dz(iatom) + dxm(ncount+2)
     ncount    = ncount + 3
  End do
  !DCC#      Endif
  !
  !========================================================================
  !
  call chmdealloc('qmene.src','EAMPC2','istore',size(istore),crl=istore)

#if KEY_PARALLEL==1
  IF (.NOT. QMPI) call psync()
#endif 
  !
  Return
END SUBROUTINE EAMPC2


SUBROUTINE EAMPCE (COORD, DXM, ENUCLR, X, Y, Z, &
     H, INBX, JNBX, EXFACT, NDIM1, &
     DXE1BM, DYE1BM, DZE1BM, DXE1BQ, DYE1BQ, &
     DZE1BQ, E1BDXS, E1BDYS, E1BDZS, XG, YG, ZG, &
     QGRPMM,INDX1E,NQMPR)
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
  Integer  inbx(*), jnbx(*), ndim1
  Logical  qgrpmm(*)
  real(chm_real)   coord(3,*), dxe1bm(*), dye1bm(*),  &
       dze1bm(*), dxe1bq(*), &
       dye1bq(*), dze1bq(*), exfact, enuclr, h(*), xg(*), &
       yg(*), zg(*), dxm(3*natom)
  real(chm_real)   x(*), y(*), z(*)
  !DCC-
  !     The following arrays are used to pass the product of the one-electron
  !     integrals for QM/MM charge/valence interations and the derivative of
  !     the switching function to EAMPCG.  DCC 2.22.93
  !
  Integer  INDX1E,NQMPR,NLENM
  real(chm_real)   e1bdxs(*), e1bdys(*), e1bdzs(*)
  !DCC+
  ! . Local variables.
  Integer:: ndim2, nlen,isize_needed,isize_needed_2,space_2
  Logical   am1
  ! . Pointers.
  integer,dimension(2) :: cgl,e1b,enuc,fact0,fact1m,fact1p,fact20,fact2m,fact2p,fact4p, &
       pppp,pzpz,rqm2,rqm2b,rqm,rqmb,rqmi,scale,sfct1m,sfct1p,sfct20, &
       sfct2m,sfct2p,sfct4p,spz,ss,work1,work2,work3,work4,work5, &
       xn,yn,zn,xn2,yn2,zn2,xqm,yqm,zqm,dxenuc,dyenuc,dzenuc, &
       dxint,dyint,dzint,dxpppp,dypppp,dzpppp,dxpzpz,dypzpz,dzpzpz, &
       dxsfm,dysfm,dzsfm,dxsfq,dysfq,dzsfq,dxspz,dyspz,dzspz,dxss,dyss,dzss
  !
  am1   = (Index(keywrd,'AM1') .gt. 0) .or. (Index(keywrd,'PM3') .gt. 0)

  ! . Allocate space for the arrays.
  ndim2 = natom - natqm
  If (ndim2 .gt. 0) then
     nlen   = ndim2
     nlenM  = NQMPR

     space_2       = 2*NQMPR            ! for integer array
     isize_needed  = 100*NQMPR + ndim2  ! for real array
     isize_needed_2= isize_needed*2     ! allocate twice of isize_needed and space
     !   allocate necessary memory/arrays.
     !   allocate more than necessary memory, so that do not have to allocate everytime.
     if(allocated(local_scratch).and.size(local_scratch).lt.isize_needed) then
        call chmdealloc('qmene.src','EAMPCE','local_scratch',size(local_scratch),crl=local_scratch)
        call chmdealloc('qmene.src','EAMPCE','local_iscratch',size(local_iscratch),intg=local_iscratch)
     end if
     if(.not.allocated(local_scratch)) then
        call chmalloc('qmene.src','EAMPCE','local_scratch',isize_needed_2,crl=local_scratch)
        call chmalloc('qmene.src','EAMPCE','local_iscratch',space_2,intg=local_iscratch)
     end if
     cgl(1)   =1           ; cgl(2)   =NQMPR              
     e1b(1)   =cgl(2)   +1 ; e1b(2)   =cgl(2)   +10*NQMPR 
     enuc(1)  =e1b(2)   +1 ; enuc(2)  =e1b(2)   +NQMPR    
     fact0(1) =enuc(2)  +1 ; fact0(2) =enuc(2)  +NQMPR    
     fact1m(1)=fact0(2) +1 ; fact1m(2)=fact0(2) +NQMPR    
     fact1p(1)=fact1m(2)+1 ; fact1p(2)=fact1m(2)+NQMPR    
     fact20(1)=fact1p(2)+1 ; fact20(2)=fact1p(2)+NQMPR    
     fact2m(1)=fact20(2)+1 ; fact2m(2)=fact20(2)+NQMPR    
     fact2p(1)=fact2m(2)+1 ; fact2p(2)=fact2m(2)+NQMPR    
     fact4p(1)=fact2p(2)+1 ; fact4p(2)=fact2p(2)+NQMPR    
     pppp(1)  =fact4p(2)+1 ; pppp(2)  =fact4p(2)+NQMPR    
     pzpz(1)  =pppp(2)  +1 ; pzpz(2)  =pppp(2)  +NQMPR    
     rqm2(1)  =pzpz(2)  +1 ; rqm2(2)  =pzpz(2)  +NQMPR    
     rqm2b(1) =rqm2(2)  +1 ; rqm2b(2) =rqm2(2)  +NQMPR    
     rqm(1)   =rqm2b(2) +1 ; rqm(2)   =rqm2b(2) +NQMPR    
     rqmb(1)  =rqm(2)   +1 ; rqmb(2)  =rqm(2)   +NQMPR    
     rqmi(1)  =rqmb(2)  +1 ; rqmi(2)  =rqmb(2)  +NQMPR    
     scale(1) =rqmi(2)  +1 ; scale(2) =rqmi(2)  +NQMPR    
     sfct1m(1)=scale(2) +1 ; sfct1m(2)=scale(2) +NQMPR    
     sfct1p(1)=sfct1m(2)+1 ; sfct1p(2)=sfct1m(2)+NQMPR    
     sfct20(1)=sfct1p(2)+1 ; sfct20(2)=sfct1p(2)+NQMPR    
     sfct2m(1)=sfct20(2)+1 ; sfct2m(2)=sfct20(2)+NQMPR    
     sfct2p(1)=sfct2m(2)+1 ; sfct2p(2)=sfct2m(2)+NQMPR    
     sfct4p(1)=sfct2p(2)+1 ; sfct4p(2)=sfct2p(2)+NQMPR    
     spz(1)   =sfct4p(2)+1 ; spz(2)   =sfct4p(2)+NQMPR    
     ss(1)    =spz(2)   +1 ; ss(2)    =spz(2)   +NQMPR    
     work1(1) =ss(2)    +1 ; work1(2) =ss(2)    +NQMPR    
     work2(1) =work1(2) +1 ; work2(2) =work1(2) +NQMPR    
     work3(1) =work2(2) +1 ; work3(2) =work2(2) +NQMPR    
     work4(1) =work3(2) +1 ; work4(2) =work3(2) +NQMPR    
     work5(1) =work4(2) +1 ; work5(2) =work4(2) +NQMPR    
     xn(1)    =work5(2) +1 ; xn(2)    =work5(2) +NQMPR    
     yn(1)    =xn(2)    +1 ; yn(2)    =xn(2)    +NQMPR    
     zn(1)    =yn(2)    +1 ; zn(2)    =yn(2)    +NQMPR    
     xn2(1)   =zn(2)    +1 ; xn2(2)   =zn(2)    +NQMPR    
     yn2(1)   =xn2(2)   +1 ; yn2(2)   =xn2(2)   +NQMPR    
     zn2(1)   =yn2(2)   +1 ; zn2(2)   =yn2(2)   +NQMPR    
     xqm(1)   =zn2(2)   +1 ; xqm(2)   =zn2(2)   +NQMPR    
     yqm(1)   =xqm(2)   +1 ; yqm(2)   =xqm(2)   +NQMPR    
     zqm(1)   =yqm(2)   +1 ; zqm(2)   =yqm(2)   +NQMPR    
     dxenuc(1)=zqm(2)   +1 ; dxenuc(2)=zqm(2)   +NQMPR    
     dyenuc(1)=dxenuc(2)+1 ; dyenuc(2)=dxenuc(2)+NQMPR    
     dzenuc(1)=dyenuc(2)+1 ; dzenuc(2)=dyenuc(2)+NQMPR    
     dxint(1) =dzenuc(2)+1 ; dxint(2) =dzenuc(2)+10*NQMPR 
     dyint(1) =dxint(2) +1 ; dyint(2) =dxint(2) +10*NQMPR 
     dzint(1) =dyint(2) +1 ; dzint(2) =dyint(2) +10*NQMPR 
     dxpppp(1)=dzint(2) +1 ; dxpppp(2)=dzint(2) +NQMPR    
     dypppp(1)=dxpppp(2)+1 ; dypppp(2)=dxpppp(2)+NQMPR    
     dzpppp(1)=dypppp(2)+1 ; dzpppp(2)=dypppp(2)+NQMPR    
     dxpzpz(1)=dzpppp(2)+1 ; dxpzpz(2)=dzpppp(2)+NQMPR    
     dypzpz(1)=dxpzpz(2)+1 ; dypzpz(2)=dxpzpz(2)+NQMPR    
     dzpzpz(1)=dypzpz(2)+1 ; dzpzpz(2)=dypzpz(2)+NQMPR    
     dxsfm(1) =dzpzpz(2)+1 ; dxsfm(2) =dzpzpz(2)+NQMPR    
     dysfm(1) =dxsfm(2) +1 ; dysfm(2) =dxsfm(2) +NQMPR    
     dzsfm(1) =dysfm(2) +1 ; dzsfm(2) =dysfm(2) +NQMPR    
     dxsfq(1) =dzsfm(2) +1 ; dxsfq(2) =dzsfm(2) +NQMPR    
     dysfq(1) =dxsfq(2) +1 ; dysfq(2) =dxsfq(2) +NQMPR    
     dzsfq(1) =dysfq(2) +1 ; dzsfq(2) =dysfq(2) +NQMPR    
     dxspz(1) =dzsfq(2) +1 ; dxspz(2) =dzsfq(2) +NQMPR    
     dyspz(1) =dxspz(2) +1 ; dyspz(2) =dxspz(2) +NQMPR    
     dzspz(1) =dyspz(2) +1 ; dzspz(2) =dyspz(2) +NQMPR    
     dxss(1)  =dzspz(2) +1 ; dxss(2)  =dzspz(2) +NQMPR    
     dyss(1)  =dxss(2)  +1 ; dyss(2)  =dxss(2)  +NQMPR    
     dzss(1)  =dyss(2)  +1 ; dzss(2)  =dyss(2)  +NQMPR    
     ! . Calculate the integrals and their derivatives.
     Call Eampe2 (coord, dxm, x, y, z, inbx, jnbx, exfact, am1, enuclr, h, ndim1, &
          dxe1bm, dye1bm, dze1bm, dxe1bq, dye1bq, dze1bq, e1bdxs, e1bdys, e1bdzs, &
          xg, yg, zg, qgrpmm,ndim2,local_iscratch, &
          local_scratch(cgl(1):cgl(2)),local_scratch(e1b(1):e1b(2)), local_scratch(enuc(1):enuc(2)), &
          local_scratch(fact0(1):fact0(2)), local_scratch(fact1m(1):fact1m(2)), &
          local_scratch(fact1p(1):fact1p(2)), local_scratch(fact20(1):fact20(2)), &
          local_scratch(fact2m(1):fact2m(2)), local_scratch(fact2p(1):fact2p(2)), &
          local_scratch(fact4p(1):fact4p(2)), local_scratch(pppp(1):pppp(2)), &
          local_scratch(pzpz(1):pzpz(2)), local_scratch(rqm2(1):rqm2(2)), &
          local_scratch(rqm2b(1):rqm2b(2)), local_scratch(rqm(1):rqm(2)), &
          local_scratch(rqmb(1):rqmb(2)), local_scratch(rqmi(1):rqmi(2)), local_scratch(scale(1):scale(2)),&
          local_scratch(sfct1m(1):sfct1m(2)), local_scratch(sfct1p(1):sfct1p(2)), &
          local_scratch(sfct20(1):sfct20(2)), local_scratch(sfct2m(1):sfct2m(2)), &
          local_scratch(sfct2p(1):sfct2p(2)), local_scratch(sfct4p(1):sfct4p(2)), &
          local_scratch(spz(1):spz(2)), local_scratch(ss(1):ss(2)), &
          local_scratch(work1(1):work1(2)), local_scratch(work2(1):work2(2)), &
          local_scratch(work3(1):work3(2)), local_scratch(work4(1):work4(2)), &
          local_scratch(work5(1):work5(2)), local_scratch(xn(1):xn(2)), &
          local_scratch(yn(1):yn(2)), local_scratch(zn(1):zn(2)), &
          local_scratch(xn2(1):xn2(2)), local_scratch(yn2(1):yn2(2)), local_scratch(zn2(1):zn2(2)), &
          local_scratch(xqm(1):xqm(2)), local_scratch(yqm(1):yqm(2)), local_scratch(zqm(1):zqm(2)), &
          local_scratch(dxenuc(1):dxenuc(2)), local_scratch(dyenuc(1):dyenuc(2)), &
          local_scratch(dzenuc(1):dzenuc(2)), local_scratch(dxint(1):dxint(2)), &
          local_scratch(dyint(1):dyint(2)), local_scratch(dzint(1):dzint(2)), &
          local_scratch(dxpppp(1):dxpppp(2)), local_scratch(dypppp(1):dypppp(2)), &
          local_scratch(dzpppp(1):dzpppp(2)), local_scratch(dxpzpz(1):dxpzpz(2)), &
          local_scratch(dypzpz(1):dypzpz(2)), local_scratch(dzpzpz(1):dzpzpz(2)), &
          local_scratch(dxsfm(1):dxsfm(2)), local_scratch(dysfm(1):dysfm(2)), &
          local_scratch(dzsfm(1):dzsfm(2)), local_scratch(dxsfq(1):dxsfq(2)), &
          local_scratch(dysfq(1):dysfq(2)), local_scratch(dzsfq(1):dzsfq(2)), &
          local_scratch(dxspz(1):dxspz(2)), local_scratch(dyspz(1):dyspz(2)), &
          local_scratch(dzspz(1):dzspz(2)), local_scratch(dxss(1):dxss(2)), &
          local_scratch(dyss(1):dyss(2)), local_scratch(dzss(1):dzss(2)), &
          h1pert,h2pert, INDX1E,NQMPR)
  Endif
  !
  Return
END SUBROUTINE EAMPCE

!DCC#      SUBROUTINE EAMPE2 (COORD, INBX, JNBX, EXFACT, AM1, ENUCLR, H,     &
!DCC#                         NDIM1, DXE1BM, DYE1BM, DZE1BM, DXE1BQ,         &
!DCC#                         DYE1BQ, DZE1BQ, XG, YG, ZG, QGRPMM_local,      &
!DCC#                         NDIM2, JNBL, CGL, E1B,                         &
!DCC#                         ENUC, FACT0, FACT1M, FACT1P, FACT20, FACT2M,   &
!DCC#                         FACT2P, FACT4P, PPPP, PZPZ, RQM2, RQM2B, RQM,  &
!DCC#                         RQMB, RQMI, SCALE, SFCT1M, SFCT1P, SFCT20,     &
!DCC#                         SFCT2M, SFCT2P, SFCT4P, SPZ, SS, WORK1, WORK2, &
!DCC#                         WORK3, WORK4, WORK5, XN, YN, ZN, XN2, YN2,     &
!DCC#                         ZN2, XQM, YQM, ZQM, DXENUC, DYENUC, DZENUC,    &
!DCC#                         DXINT, DYINT, DZINT, DXPPPP, DYPPPP, DZPPPP,   &
!DCC#                         DXPZPZ, DYPZPZ, DZPZPZ, DXSFM, DYSFM, DZSFM,   &
!DCC#                         DXSFQ, DYSFQ, DZSFQ, DXSPZ, DYSPZ, DZSPZ,      &
!DCC#                         DXSS, DYSS, DZSS)
!DCC-
SUBROUTINE EAMPE2 (COORD, DXM, X, Y, Z, INBX, JNBX, EXFACT, AM1, ENUCLR, H, &
     NDIM1, DXE1BM, DYE1BM, DZE1BM, DXE1BQ, DYE1BQ, DZE1BQ,   &
     E1BDXS, E1BDYS, E1BDZS, XG, YG, ZG, QGRPMM_local, &
     NDIM2, JNBL, CGL, E1B, ENUC, &
     FACT0, FACT1M, FACT1P, FACT20, FACT2M, FACT2P, FACT4P, &
     PPPP, PZPZ, RQM2, RQM2B, RQM, RQMB, RQMI, &
     SCALE, SFCT1M, SFCT1P, SFCT20, SFCT2M, SFCT2P, SFCT4P, &
     SPZ, SS, WORK1, WORK2, WORK3, WORK4, WORK5, &
     XN, YN, ZN, XN2, YN2, ZN2, XQM, YQM, ZQM, &
     DXENUC, DYENUC, DZENUC, DXINT, DYINT, DZINT, &
     DXPPPP, DYPPPP, DZPPPP, DXPZPZ, DYPZPZ, DZPZPZ, &
     DXSFM, DYSFM, DZSFM, DXSFQ, DYSFQ, DZSFQ, DXSPZ, DYSPZ, DZSPZ, &
     DXSS, DYSS, DZSS, HPRT1,HPRT2, NDIM1E,NQMPR)
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
  !
#if KEY_PBOUND==1
  use pbound  
#endif
#if KEY_PBOUND==1
  use coordc  
#endif
  !
#if KEY_PARALLEL==1
  use parallel  
#endif
  !
  !
  implicit none

  ! . Passed variables.
  Integer   inbx(*), jnbx(*), jnbl(*), ndim1, ndim2
  Logical   am1, qgrpmm_local(*)
  real(chm_real):: coord(3,*), exfact, enuclr, h(*), &
       dxe1bm(ndim1,10), dye1bm(ndim1,10), dze1bm(ndim1,10), &
       dxe1bq(*),dye1bq(*),dze1bq(*), &
       dxm(3*natom), x(*), y(*), z(*)
  !DCC-
  !     The following arrays are used to pass the product of the one-electron
  !     integrals for QM/MM charge/valence interations and the derivative of
  !     the switching function to EAMPCG.  DCC 2.22.93
  !
  !     real(chm_real):: dxe1bq(ndim1,10), dye1bq(ndim1,10), dze1bq(ndim1,10)
  !     real(chm_real):: e1bdxs(ndim1,10), e1bdys(ndim1,10), e1bdzs(ndim1,10)
  !     real(chm_real):: cgl(*), e1b(ndim2,*), enuc(*), fact0(*), fact1m(*), &
  !                      dyint(ndim2,*), dzint(ndim2,*), dxpppp(*), dypppp(*)

  integer   NDIM1E,NQMPR
  real(chm_real)  e1bdxs(NDIM1E,*),e1bdys(NDIM1E,*),e1bdzs(NDIM1E,*)
  real(chm_real) cgl(*), e1b(NQMPR,*), enuc(*), fact0(*), fact1m(*), &
       fact1p(*), fact20(*), fact2m(*), fact2p(*), &
       fact4p(*), pppp(*), pzpz(*), rqm2(*), rqm2b(*), &
       rqm(*), rqmb(*), rqmi(*), scale(*), sfct1m(*), &
       sfct1p(*), sfct20(*), sfct2m(*), sfct2p(*), &
       sfct4p(*), spz(*), ss(*), work1(*), work2(*), &
       work3(*), work4(*), work5(*), xn(*), yn(*), zn(*), &
       xn2(*), yn2(*), zn2(*), xqm(*), yqm(*), zqm(*), &
       dxenuc(*), dyenuc(*), dzenuc(*), dxint(NQMPR,*), &
       dyint(NQMPR,*), dzint(NQMPR,*), dxpppp(*), dypppp(*), &
       dzpppp(*), dxpzpz(*), dypzpz(*), dzpzpz(*), dxsfm(*), &
       dysfm(*), dzsfm(*), dxsfq(*), dysfq(*), dzsfq(*), &
       dxspz(*), dyspz(*), dzspz(*), dxss(*), dyss(*), dzss(*), &
       xg(*), yg(*), zg(*), HPRT1(*),HPRT2(*)
  ! . Local variables.
  Integer  i, ii, ij, ipair, j, jj, n, n1, n2, n3, matm, mgrp, mnum, &
       mstrt, mstop, nbg, npr, nqm, nterm, ntot, numqm, qatom, &
       qfirst, qgrp, qnum, qstop, qstrt, IQATOM
  Logical  qswitr
  real(chm_real):: alpha, c2ofnb, c2onnb, cgqm, d1, d2, d4,  &
       dfm, dfn, dfq, e1ba(10), en, f1, f2, f3, funct, &
       rho0, rho1, rho2, rijl, riju, rqm2l, rqm2u, rul3, rul12, &
       scent, sfact, tpppp, tpzpz, &
       trqm2b, xq, yq, zq
  real(chm_real):: temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp0
  real(chm_real):: delta_xyz(3)
#if KEY_PBOUND==1 /*pbound*/
  logical:: pbc_chk(3)
  real(chm_real):: xyz_pbc(3),xyz_size(3),xyz_size2(3)
#endif /*     (pbound)*/
  !
  real(chm_real), parameter :: quartr = PT25, &              ! 0.25D0
       atobhr = 1.D0 / BOHRR, &
       atobh2 = atobhr * atobhr, &
       confac = EV_TO_KCAL, &        ! 23.061D0
       ctoev  = AU_TO_EV             ! 27.21D0 
  real(chm_real) ENUCTMP
  real(chm_real) RFLAMB
  !DCC-
  integer  itemp1, mstop1(maxgrp), nmgrp,natom2,matm2
  INTEGER  QSTRTGRP,QSTOPGRP,QQSTOP,QQSTRT,INDX1E,NQMGROUP
  !
  ! go parallel
  INTEGER ATFRST,ATLAST
  INTEGER MGFRST,MGLAST,NODNGM,ncount
  real(chm_real)  DXTMP(3)
  !
  !
  !DCC+
  ! . Define some constants.
  !
  c2ofnb = ctofnb * ctofnb
  c2onnb = ctonnb * ctonnb
  rul3   = one / (c2ofnb - c2onnb)**3
  rul12  = twelve * rul3
  natom2 = 2*natom
  !
  ! for QMPERT case
  RFLAMB=ONE
  IF(QMPERT) RFLAMB=RLAMB0
  !
#if KEY_PBOUND==1 /*pbound*/
  xyz_size(1)=XSIZE
  xyz_size(2)=YSIZE
  xyz_size(3)=ZSIZE
  XYZ_SIZE2(1:3) = half*xyz_size(1:3)
  If(qBoun) then
     IF(.not.qCUBoun) CALL WRNDIE(-5,'<EAMPE2>', 'Other than CUBoundary not supported')
  End if
#endif /*     (pbound)*/
  !
  !
  ! . Loop over the group non-bond lists - the QM groups are first.
  nbg    = 0
  ntot   = 0
  numqm  = 0
  qfirst = 0
  QSTRTGRP = 0
  INDX1E = 0
  NQMGROUP = 0
  Do qgrp = 1,ngrp
     npr    = inbx(qgrp) - qfirst
     qfirst = inbx(qgrp)
     qstrt  = igpbs(qgrp) + 1
     qstop  = igpbs(qgrp + 1)
     qnum   = qstop - qstrt + 1
     !
     ! . Loop over the MM groups.
     n = 0
     If (npr .le. 0) then
        If (.not. qgrpmm_local(qgrp)) numqm = numqm + qnum
     Else
        !
        ! JG 3/21/01
        IF(QSTRTGRP.EQ.0) THEN
           QSTRTGRP=QSTRT
           QSTOPGRP=QSTRT+NATQM-1
        ENDIF
        !-jg
        !DCC-
        nmgrp = 0
        itemp1 = 0
        !DCC+
        do ipair = 1,npr
           nbg = nbg + 1
           mgrp = jnbx(nbg)
           !CC
           If (mgrp .lt. 0) then
              sfact =  exfact
              mgrp  = -mgrp
           Else
              sfact =  one
           Endif
           mstrt = igpbs(mgrp) + 1
           mstop = igpbs(mgrp + 1)
           mnum  = mstop - mstrt + 1
           funct = one
           dfm   = zero
           dfq   = zero
           !
           delta_xyz(1) = xg(qgrp) - xg(mgrp)
           delta_xyz(2) = yg(qgrp) - yg(mgrp)
           delta_xyz(3) = zg(qgrp) - zg(mgrp)
           ! JG 6/1/01
           ! ADD SIMPLE PERIODIC BOUNDARY CONDITIONS FOR QM/MM INTERACTION
#if KEY_PBOUND==1 /*pbound*/
           IF(qBoun .and. qCUBoun) THEN
              pbc_chk(1:3)=.false.
              do ij=1,3
                 if(abs(delta_xyz(i)).gt.xyz_size2(i)) then
                    pbc_chk(i)  =.true.
                    xyz_pbc(i)  = sign(xyz_size(i),delta_xyz(i))
                    delta_xyz(i)= delta_xyz(i)-xyz_pbc(i)
                 end if
              end do
           ENDIF
#endif /*  (pbound)*/
           scent = delta_xyz(1)*delta_xyz(1)+delta_xyz(2)*delta_xyz(2)+delta_xyz(3)*delta_xyz(3)
           !
           If ((.NOT.QMCUTF) .OR. (scent .lt. c2ofnb)) then
              nmgrp = nmgrp + 1
              mstop1(nmgrp) = itemp1 + mstop - mstrt + 1
              itemp1 = mstop1(nmgrp)
              !
              ! namkh 08/08/04
              ! QM/MM-Ewald
              IF(.NOT.LQMEWD) THEN
                 qswitr = QMCUTF .AND. (scent .gt. c2onnb)
                 If (qswitr) then
                    rijl  = c2onnb - scent
                    riju  = c2ofnb - scent
                    funct = riju*riju*(riju-three*rijl)*rul3
                    dfn   = rijl*riju*rul12
                    dfm   = dfn / mnum
                    dfq   = dfn / qnum
                 Endif
              END IF
              !
              ! . Fill the intermediate atom arrays.
              do matm = mstrt,mstop
                 ! namkh 10/26/05
#if KEY_PBOUND==1 /*pbound*/
                 IF(qBoun .and. qCUBoun) THEN 
                    IF (pbc_chk(1)) THEN
                       XCOMP(MATM) = X(matm)+xyz_pbc(1)
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
#endif /*  (pbound)*/
                 !
                 !--------------------------------------------------------------------------
                 !      QM-Link atom case: exclude the link atom from the MM list
                 IF(QATLAB(MATM).LE.90) then 
                    !--------------------------------------------------------------------------
                    ! QM/MM-Ewald; namkh 08/08/04
                    IF(LQMEWD.AND.(MMINB(MATM).LT.0)) MMINB(MATM) = MATM

                    n = n + 1
                    jnbl(n)  = matm
                    cgl(n)   = RFLAMB * sfact * cg(matm)
                    scale(n) = funct
                    dxsfm(n) = delta_xyz(1) * dfm
                    dysfm(n) = delta_xyz(2) * dfm
                    dzsfm(n) = delta_xyz(3) * dfm
                    dxsfq(n) = delta_xyz(1) * dfq
                    dysfq(n) = delta_xyz(2) * dfq
                    dzsfq(n) = delta_xyz(3) * dfq
                 end if
              end do     ! matm = mstrt,mstop
           Endif         ! (.NOT.QMCUTF) .OR. (scent .lt. c2ofnb)
        end do           ! ipair = 1,npr
        ! . Loop over the QM atoms, including the QM-Link atoms, nqmlnk
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
        ! go parallel
#if KEY_PARALLEL==1 /*paramain*/
        IF(NMGRP.GT.0 .AND. NMGRP.LT.NUMNOD .AND. .NOT. QMPI)  &
             CALL WRNDIE(-5,'<EAMPE2>', 'Number of MM groups should be greater than number of CPUs.')
        NODNGM = NMGRP / NUMNOD
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
        MGFRST=MYNOD*NODNGM + 1
        MGLAST=(MYNOD+1)*NODNGM
        IF(MYNOD.EQ.(NUMNOD-1)) MGLAST=NMGRP
        !
        IF(MYNOD.EQ.0) THEN
           ATFRST=1
        ELSE
           ATFRST=mstop1(mgfrst-1)+1
        END IF
        ATLAST=mstop1(mglast)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
        ATFRST=1
        ATLAST=N
        !
        MGFRST=1
        MGLAST=NMGRP
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
        IF (QMPI) THEN
           ATFRST=1
           ATLAST=N
           !
           MGFRST=1
           MGLAST=NMGRP
        ENDIF
#else /* (paramain)*/
        ATFRST=1
        ATLAST=N
        !
        MGFRST=1
        MGLAST=NMGRP
#endif /* (paramain)*/
        !
        Do qatom = QSTRT,QSTOP
           numqm = numqm + 1
           !--jg
           If (n .gt. 0 .and. qatlab(qatom) .gt. 0) then
              !--------------------------------------------------------------------------
              !   One should rather delete more mm interaction than without
              !   including qm atoms (link atoms) in qm calculations.  It's
              !   not generating "balanced" charge distributions at the H(link)
              !   atom.   JG 12/96
              !   Not changed in standard CHARMM....JG 1/6/99
              !              If (n .gt. 0 .and. qatlab(qatom) .ge. 0) then
              !--------------------------------------------------------------------------
              ! . Define constants for the quantum mechanical atom.
              nqm    = nat(numqm)
              alpha  = alfa(nqm)
              cgqm   = core(nqm)
              n1 = n1st(numqm)
              n2 = nmidle(numqm)
              n3 = nlast(numqm)
              xq = coord(1,numqm)
              yq = coord(2,numqm)
              zq = coord(3,numqm)
              !
              rho0 = (half / bdd(nqm,1) + rho0mm) **2
              !
              If (am1) then
                 nterm = 0
                 do i = 1,10
                    If (abs(fn1(nqm,i)) .gt. zero) nterm = nterm + 1
                 end do
              Endif
              ! . Determine some intermediate arrays.
              do j = atfrst,atlast   ! j = 1, n
#if KEY_PBOUND==1 /*pbound*/
                 IF(qBoun) THEN
                    if (qCUBoun) then
                       xqm(j) = xq - xCOMP(jnbl(j))
                       yqm(j) = yq - yCOMP(jnbl(j))
                       zqm(j) = zq - zCOMP(jnbl(j))
                    end if
                 ELSE
#endif /*     (pbound)*/
                    xqm(j) = xq - x(jnbl(j))
                    yqm(j) = yq - y(jnbl(j))
                    zqm(j) = zq - z(jnbl(j))
#if KEY_PBOUND==1 /*pbound*/
                 ENDIF
#endif /*     (pbound)*/
                 rqm2(j) = xqm(j)*xqm(j)+yqm(j)*yqm(j)+zqm(j)*zqm(j)
                 rqm(j)  = sqrt(rqm2(j))
                 temp1   = one / rqm(j)
                 xn(j)   = temp1 * xqm(j)
                 yn(j)   = temp1 * yqm(j)
                 zn(j)   = temp1 * zqm(j)
                 rqmi(j) = temp1
                 !CC            
              end do
              ! . Calculate the integrals for atoms with one orbital.
              If (natorb(nqm) .eq. 1) then
                 do j = atfrst,atlast    ! j = 1, n
                    trqm2b = atobh2 * rqm2(j)
                    fact0(j) = one / (trqm2b + rho0)
                    e1b(j,1) = ctoev * sqrt(fact0(j))
                 end do
                 e1b(atfrst:atlast,2:10)=zero
                 !
                 do j = atfrst,atlast     ! j = 1, n
                    temp1 = atobh2 * fact0(j) * e1b(j,1)
                    dxint(j,1) = - xqm(j) * temp1
                    dyint(j,1) = - yqm(j) * temp1
                    dzint(j,1) = - zqm(j) * temp1
                 end do
                 dxint(atfrst:atlast,2:10) =zero
                 ! . Do the integrals for atoms with more than one orbital.
              Else
                 rho1 = (half / bdd(nqm,2) + rho0mm) **2
                 rho2 = (half / bdd(nqm,3) + rho0mm) **2
                 !
                 d1 = dd(nqm)
                 d2 = two * qq(nqm)
                 d4 = d2 * d2
                 ! . Loop over the molecular mechanics atoms.
                 do j = atfrst,atlast     ! j = 1, n
                    rqm2b(j)  = atobh2 * rqm2(j)
                    rqmb(j)   = atobhr * rqm(j)
                    work1(j)  = rqmb(j) - d1
                    work2(j)  = rqmb(j) + d1
                    work3(j)  = rqmb(j) - d2
                    work4(j)  = rqmb(j) + d2
                 end do
                 !
                 do j = atfrst,atlast    ! j = 1, n
                    fact0(j)  = one / (rqm2b(j) + rho0)
                    fact1m(j) = one / (work1(j) * work1(j) + rho1)
                    fact1p(j) = one / (work2(j) * work2(j) + rho1)
                    fact20(j) = one / (rqm2b(j) + rho2)
                    fact2m(j) = one / (work3(j) * work3(j) + rho2)
                    fact2p(j) = one / (work4(j) * work4(j) + rho2)
                    fact4p(j) = one / (rqm2b(j) + d4 + rho2)
                    sfct1m(j) = sqrt(fact1m(j))
                    sfct1p(j) = sqrt(fact1p(j))
                    sfct20(j) = sqrt(fact20(j))
                    sfct2m(j) = sqrt(fact2m(j))
                    sfct2p(j) = sqrt(fact2p(j))
                    sfct4p(j) = sqrt(fact4p(j))
                 end do
                 !
                 do j = atfrst,atlast     ! j = 1, n
                    ss(j)  = sqrt(fact0(j))
                    spz(j) = half * (sfct1p(j) - sfct1m(j))
                    pzpz(j)= ss(j) + quartr*(sfct2p(j)+sfct2m(j)) - half*sfct20(j)
                    pppp(j)= ss(j) + half * (sfct4p(j)-sfct20(j))
                 end do
                 ! . Define some variables needed for the transformation.
                 do j = atfrst,atlast    ! j = 1, n
                    xn2(j) = xn(j) * xn(j)
                    yn2(j) = yn(j) * yn(j)
                    zn2(j) = zn(j) * zn(j)
                 end do
                 ! . Fill arrays with the calculated integrals. : atfrst:atlast  ! j = 1, n
                 e1b(atfrst:atlast,1) =ss(atfrst:atlast)
                 e1b(atfrst:atlast,2) =xn(atfrst:atlast) *spz(atfrst:atlast)
                 e1b(atfrst:atlast,3) =xn2(atfrst:atlast)*pzpz(atfrst:atlast)+(yn2(atfrst:atlast) &
                      +zn2(atfrst:atlast))*pppp(atfrst:atlast)
                 e1b(atfrst:atlast,4) =yn(atfrst:atlast) *spz(atfrst:atlast)
                 e1b(atfrst:atlast,5) =xn(atfrst:atlast) *yn(atfrst:atlast)* &
                      (pzpz(atfrst:atlast)-pppp(atfrst:atlast))
                 e1b(atfrst:atlast,6) =yn2(atfrst:atlast)*pzpz(atfrst:atlast)+(xn2(atfrst:atlast) &
                      +zn2(atfrst:atlast))*pppp(atfrst:atlast)
                 e1b(atfrst:atlast,7) =zn(atfrst:atlast) *spz(atfrst:atlast)
                 e1b(atfrst:atlast,8) =xn(atfrst:atlast)*zn(atfrst:atlast)*(pzpz(atfrst:atlast)-pppp(atfrst:atlast))
                 e1b(atfrst:atlast,9) =yn(atfrst:atlast)*zn(atfrst:atlast)*(pzpz(atfrst:atlast)-pppp(atfrst:atlast))
                 e1b(atfrst:atlast,10)=zn2(atfrst:atlast)*pzpz(atfrst:atlast)+(xn2(atfrst:atlast) &
                      +yn2(atfrst:atlast))*pppp(atfrst:atlast)
                 !
                 e1b(atfrst:atlast,1) = ctoev * e1b(atfrst:atlast,1)
                 do i = 2,10
                    e1b(atfrst:atlast,i) = ctoev * cgl(atfrst:atlast) * e1b(atfrst:atlast,i)
                 end do
                 ! . Calculate some intermediate arrays.
                 do j = atfrst,atlast      ! j = 1, n
                    temp1  = fact0(j) * atobh2 * ss(j)
                    dxss(j)=-xqm(j) * temp1
                    dyss(j)=-yqm(j) * temp1
                    dzss(j)=-zqm(j) * temp1
                 end do
                 !
                 do j = atfrst,atlast      ! j = 1, n
                    temp1   = atobhr*half*rqmi(j)*( work1(j)*sfct1m(j)*fact1m(j)-work2(j)*sfct1p(j)*fact1p(j) )
                    dxspz(j)= xqm(j)*temp1
                    dyspz(j)= yqm(j)*temp1
                    dzspz(j)= zqm(j)*temp1
                 end do
                 !
                 do j = atfrst,atlast      ! j = 1, n
                    temp1 = work4(j) * sfct2p(j) * fact2p(j) / rqmb(j)
                    temp2 = work3(j) * sfct2m(j) * fact2m(j) / rqmb(j)
                    temp3 = atobh2 * (half * sfct20(j) * fact20(j) - quartr * (temp1 + temp2))
                    dxpzpz(j) = dxss(j) + xqm(j) * temp3
                    dypzpz(j) = dyss(j) + yqm(j) * temp3
                    dzpzpz(j) = dzss(j) + zqm(j) * temp3
                 end do
                 !
                 do j = atfrst,atlast      ! j = 1, n
                    temp1     = atobh2 *half   *( sfct20(j)*fact20(j) - sfct4p(j)*fact4p(j) )
                    dxpppp(j) = dxss(j)+xqm(j) *  temp1
                    dypppp(j) = dyss(j)+yqm(j) *  temp1
                    dzpppp(j) = dzss(j)+zqm(j) *  temp1
                 end do
                 ! . Calculate the derivatives.
                 do j = atfrst,atlast      ! j = 1, n
                    work1(j) = pzpz(j)- pppp(j)
                    work2(j) = yn2(j) + zn2(j)
                    work3(j) = xn2(j) + zn2(j)
                    work4(j) = rqmi(j)* spz(j)
                    work5(j) = xn2(j) + yn2(j)
                 end do
                 !
                 do j = atfrst,atlast      ! j = 1, n
                    dxint(j,1) = dxss(j)
                    dyint(j,1) = dyss(j)
                    dzint(j,1) = dzss(j)
                 end do
                 !
                 do j = atfrst,atlast      ! j = 1, n
                    dxint(j,2) = work2(j) * work4(j) + xn(j) * dxspz(j)
                    dyint(j,2) =-xn(j) * (work4(j) * yn(j) - dyspz(j))
                    dzint(j,2) =-xn(j) * (work4(j) * zn(j) - dzspz(j))
                 end do
                 !
                 do j = atfrst,atlast      ! j = 1, n
                    temp1      = two*rqmi(j)*work1(j)
                    dxint(j,3) = xn2(j)*dxpzpz(j)+work2(j)*dxpppp(j)+temp1*xn(j)*work2(j)
                    dyint(j,3) = xn2(j)*dypzpz(j)+work2(j)*dypppp(j)-temp1*yn(j)*xn2(j) 
                    dzint(j,3) = xn2(j)*dzpzpz(j)+work2(j)*dzpppp(j)-temp1*zn(j)*xn2(j) 
                 end do
                 !
                 do j = atfrst,atlast      ! j = 1, n
                    dxint(j,4) =-yn(j) * (xn(j) * work4(j) - dxspz(j))
                    dyint(j,4) = work4(j) * work3(j) + yn(j) * dyspz(j)
                    dzint(j,4) =-yn(j) * (zn(j) * work4(j) - dzspz(j))
                 end do
                 !
                 do j = atfrst,atlast      ! j = 1, n
                    dxint(j,5) = yn(j)*( rqmi(j)*work1(j)*(work2(j)-xn2(j)) + xn(j)*(dxpzpz(j)-dxpppp(j)) )
                    dyint(j,5) = xn(j)*( rqmi(j)*work1(j)*(work3(j)-yn2(j)) + yn(j)*(dypzpz(j)-dypppp(j)) )
                    dzint(j,5) = xn(j)*yn(j)*( (dzpzpz(j)-dzpppp(j))-two*zn(j)*rqmi(j)*work1(j) )
                 end do
                 !
                 do j = atfrst,atlast       ! j = 1, n
                    temp1      = two*rqmi(j)*work1(j)
                    dxint(j,6) = yn2(j)*dxpzpz(j)+work3(j)* dxpppp(j)-temp1*xn(j)*yn2(j)
                    dyint(j,6) = yn2(j)*dypzpz(j)+work3(j)*(dypppp(j)+temp1*yn(j))
                    dzint(j,6) = yn2(j)*dzpzpz(j)+work3(j)* dzpppp(j)-temp1*zn(j)*yn2(j)
                 end do
                 !
                 do j = atfrst,atlast       ! j = 1, n
                    dxint(j,7) =-zn(j) * (xn(j) * work4(j) - dxspz(j))
                    dyint(j,7) =-zn(j) * (yn(j) * work4(j) - dyspz(j))
                    dzint(j,7) = work5(j) * work4(j) + zn(j) * dzspz(j)
                 end do
                 !
                 do j = atfrst,atlast       ! j = 1, n
                    temp1      = rqmi(j)*work1(j)
                    dxint(j,8) = zn(j)*( temp1*(work2(j)-xn2(j))+xn(j)*(dxpzpz(j)-dxpppp(j)) )
                    dyint(j,8) = xn(j)*  zn(j)*( (dypzpz(j)-dypppp(j))-two*yn(j)*temp1 )
                    dzint(j,8) = xn(j)*( temp1*(work5(j)-zn2(j))+zn(j)*(dzpzpz(j)-dzpppp(j)) )
                 end do
                 !
                 do j = atfrst,atlast       ! j = 1, n
                    temp1      = rqmi(j)*work1(j)
                    dxint(j,9) = yn(j)*zn(j)*((dxpzpz(j)-dxpppp(j))-two*xn(j)*temp1)
                    dyint(j,9) = zn(j)*(temp1*(work3(j)-yn2(j))+yn(j)*(dypzpz(j)-dypppp(j)))
                    dzint(j,9) = yn(j)*(temp1*(work5(j)-zn2(j))+zn(j)*(dzpzpz(j)-dzpppp(j)))
                 end do
                 !
                 do j = atfrst,atlast        ! j = 1, n
                    temp1       = two*rqmi(j)*work1(j)
                    dxint(j,10) = zn2(j)*dxpzpz(j) + work5(j)* dxpppp(j) - temp1*xn(j)*zn2(j)
                    dyint(j,10) = zn2(j)*dypzpz(j) + work5(j)* dypppp(j) - temp1*yn(j)*zn2(j)
                    dzint(j,10) = zn2(j)*dzpzpz(j) + work5(j)*(dzpppp(j) + temp1*zn(j))
                 end do
                 ! . Multiply the derivative integrals by ctoev.
                 do j = atfrst,atlast        ! j = 1, n
                    dxint(j,1) = ctoev * dxint(j,1)
                    dyint(j,1) = ctoev * dyint(j,1)
                    dzint(j,1) = ctoev * dzint(j,1)
                 end do
                 !
                 do i = 2,10
                    do j=atfrst,atlast           ! j = 1, n
                       temp1 = ctoev * cgl(j)
                       dxint(j,i) = temp1 * dxint(j,i)
                       dyint(j,i) = temp1 * dyint(j,i)
                       dzint(j,i) = temp1 * dzint(j,i)
                    end do
                 end do
              Endif               ! (natorb(nqm) .eq. 1)

              ! . Calculate the nuclear/nuclear terms.
              do j = atfrst,atlast           ! j = 1, n
                 temp1 = e1b(j,1)
                 temp2 = cgqm * cgl(j)
                 temp3 = abs(temp2) * exp(-alpha * rqm(j))
                 temp4 = abs(temp2) * exp(-alpmm * rqm(j))
                 enuc(j)  = temp1 * (temp2 + temp3 + temp4)
                 work1(j) = temp1
                 work2(j) = temp2
                 work3(j) = temp3
                 work4(j) = temp4
              end do
              Do j = atfrst,atlast           ! j = 1, n
                 temp1 = work1(j)
                 temp2 = work2(j)
                 temp3 = work3(j)
                 temp4 = work4(j)
                 temp5 =(temp2+temp3+temp4)
                 temp6 = temp1*(alpha*temp3+alpmm*temp4)
                 dxenuc(j) = temp5*dxint(j,1) - xn(j)*temp6
                 dyenuc(j) = temp5*dyint(j,1) - yn(j)*temp6
                 dzenuc(j) = temp5*dzint(j,1) - zn(j)*temp6
              end do
              !
              If (am1) then
                 do i = 1,nterm
                    f1 = fn1(nqm,i)
                    f2 = fn2(nqm,i)
                    f3 = fn3(nqm,i)
                    do j = atfrst,atlast     ! j = 1, n
                       temp1    = exp(Max(-thirty, -f2 * (rqm(j)-f3)**2))
                       enuc(j)  = enuc(j) + cgqm * cgl(j) * rqmi(j) * f1 * temp1
                       work1(j) = temp1
                    end do
                    do j = atfrst,atlast     ! j = 1, n
                       temp1 = work1(j)
                       temp2 = cgqm * cgl(j) / rqm2(j)
                       temp3 = f1*temp1*temp2
                       temp4 = (one + two * f2 * rqm(j) * (rqm(j) - f3))
                       dxenuc(j) = dxenuc(j) - temp3 * xn(j) * temp4
                       dyenuc(j) = dyenuc(j) - temp3 * yn(j) * temp4
                       dzenuc(j) = dzenuc(j) - temp3 * zn(j) * temp4
                    end do
                 end do
              Endif
              ! . Scale the e1b(*,1) integrals by cgl now enuc is finished.
              e1b(atfrst:atlast,1) = cgl(atfrst:atlast) * e1b(atfrst:atlast,1)
              do j = atfrst,atlast
                 dxint(j,1) = cgl(j) * dxint(j,1)
                 dyint(j,1) = cgl(j) * dyint(j,1)
                 dzint(j,1) = cgl(j) * dzint(j,1)
              end do
              ! . Calculate the switching corrections and their derivatives.
              !                 e1ba(1) = dotvec(scale,e1b(1,1),n)
              e1ba(1) = zero
              do j = atfrst,atlast              ! j = 1, n
                 e1ba(1) = e1ba(1)+scale(j)*e1b(j,1)
              end do
              If (natorb(nqm) .gt. 1) then
                 do i = 2,10
                    !                       e1ba(i) = dotvec(scale,e1b(1,i),n)
                    e1ba(i) = zero
                    do j = atfrst,atlast        ! j = 1, n
                       e1ba(i) = e1ba(i) + scale(j)*e1b(j,i)
                    end do
                 end do
              Else
                 e1ba(2:10)=zero
              Endif
#if KEY_FLUCQ==1
              ! Lines added for FLUCQ, Ben Webb, 2000
              ! Add core term to fluctuating charge force
              DO I= atfrst,atlast               ! j = 1, n
                 CALL FQQCOR(JNBL(I),ENUC(I)*SCALE(I))
              ENDDO
#endif 
              !DCC-
              !
              ! . If include MM-charge/QM-core energy.
              !
              If (qeqtrm(qmch)) then
                 !                    enuclr = enuclr + dotvec(scale,enuc,n)
                 !                    ENUCTMP = dotvec(scale,enuc,n)
                 ENUCTMP = zero
                 do j = atfrst,atlast           ! j = 1, n
                    ENUCTMP = ENUCTMP + scale(j)*enuc(j)
                 end do
                 enuclr = enuclr + ENUCTMP
                 IF(QMPERT) THEN
                    ENUCP1 = ENUCP1+ENUCTMP*FACTP1
                    ENUCP2 = ENUCP2+ENUCTMP*FACTP2
                 ENDIF
              Endif
              !DCC+
              ! . Add the integrals into the one-electron matrix.
              !
              !DCC-
              ! . If include MM-charge/QM-valence or QM electronic energy.
              !
              If (qeqtrm(qmee).and..not.qeqtrm(qqel)) then
                 Call wrndie(-1,'<EAMPE2>', 'CANNOT EVALUATE QMEE WITHOUT QQEL')
              Endif
              If (qeqtrm(qmee)) then
                 jj = 0
                 do i = n1,n2
                    ii = (i * (i - 1)) / 2 + n1 - 1
                    do j = n1,i
                       ii = ii + 1
                       jj = jj + 1
                       h(ii) = h(ii) - e1ba(jj)
                       IF(QMPERT) THEN
                          HPRT1(II) = HPRT1(II)-E1BA(JJ)*FACTP1
                          HPRT2(II) = HPRT2(II)-E1BA(JJ)*FACTP2
                       ENDIF
                    end do
                 end do
                 do i = (n2+1),n3
                    ii = (i * (i + 1)) / 2
                    h(ii) = h(ii) - e1ba(1)
                    IF(QMPERT) THEN
                       HPRT1(II) = HPRT1(II)-E1BA(1)*FACTP1
                       HPRT2(II) = HPRT2(II)-E1BA(1)*FACTP2
                    ENDIF
                 end do
              Endif
              !DCC+
              ! . Store the calculated derivatives in the dxe1b/dye1b/dze1b arrays.
              !DCC#               do j = 1,n
              !DCC#                  dxe1bq(j + ntot,1) = dxint(j,1) * scale(j) + e1b(j,1) * dxsfq(j)
              !DCC#                  dye1bq(j + ntot,1) = dyint(j,1) * scale(j) + e1b(j,1) * dysfq(j)
              !DCC#                  dze1bq(j + ntot,1) = dzint(j,1) * scale(j) + e1b(j,1) * dzsfq(j)
              !DCC#                  dxe1bm(j + ntot,1) = dxint(j,1) * scale(j) + e1b(j,1) * dxsfm(j)
              !DCC#                  dye1bm(j + ntot,1) = dyint(j,1) * scale(j) + e1b(j,1) * dysfm(j)
              !DCC#                  dze1bm(j + ntot,1) = dzint(j,1) * scale(j) + e1b(j,1) * dzsfm(j)
              !DCC#               end do
              !DCC#C
              !DCC#               If (natorb(nqm) .gt. 1) then
              !DCC#                  do i = 2,10
              !DCC#                     do j = 1,n
              !DCC#                        dxe1bq(j + ntot,i) = dxint(j,i) * scale(j) + e1b(j,i) * dxsfq(j)
              !DCC#                        dye1bq(j + ntot,i) = dyint(j,i) * scale(j) + e1b(j,i) * dysfq(j)
              !DCC#                        dze1bq(j + ntot,i) = dzint(j,i) * scale(j) + e1b(j,i) * dzsfq(j)
              !DCC#                        dxe1bm(j + ntot,i) = dxint(j,i) * scale(j) + e1b(j,i) * dxsfm(j)
              !DCC#                        dye1bm(j + ntot,i) = dyint(j,i) * scale(j) + e1b(j,i) * dysfm(j)
              !DCC#                        dze1bm(j + ntot,i) = dzint(j,i) * scale(j) + e1b(j,i) * dzsfm(j)
              !DCC#                     end do
              !DCC#                  end do
              !DCC#               Else
              !DCC#                  do i = 2,10
              !DCC#                     do j = 1,n
              !DCC#                        dxe1bq(j + ntot,i) = zero
              !DCC#                        dye1bq(j + ntot,i) = zero
              !DCC#                        dze1bq(j + ntot,i) = zero
              !DCC#                        dxe1bm(j + ntot,i) = zero
              !DCC#                        dye1bm(j + ntot,i) = zero
              !DCC#                        dze1bm(j + ntot,i) = zero
              !DCC#                     end do
              !DCC#                  end do
              !DCC#               Endif
              !DCC-
              II = INDX1E + 1
              do j = atfrst,atlast             ! j = 1, n
                 jj = j + ntot
                 dxe1bm(jj,1) = dxint(j,1) * scale(j)
                 dye1bm(jj,1) = dyint(j,1) * scale(j)
                 dze1bm(jj,1) = dzint(j,1) * scale(j)
                 !JG
                 dxe1bq (ii) = dxe1bq(ii) + dxe1bm(jj,1)     ! dxint(j,1) * scale(j)
                 dye1bq (ii) = dye1bq(ii) + dye1bm(jj,1)     ! dyint(j,1) * scale(j)
                 dze1bq (ii) = dze1bq(ii) + dze1bm(jj,1)     ! dzint(j,1) * scale(j)
                 e1bdxs(ii,NQMGROUP) = e1bdxs(ii,NQMGROUP) + e1b(j,1) * dxsfq(j)
                 e1bdys(ii,NQMGROUP) = e1bdys(ii,NQMGROUP) + e1b(j,1) * dysfq(j)
                 e1bdzs(ii,NQMGROUP) = e1bdzs(ii,NQMGROUP) + e1b(j,1) * dzsfq(j)
              end do
              !JG
              mstrt = 1
              !
              ! go parallel
#if KEY_PARALLEL==1
              IF (.NOT. QMPI) THEN
                 if(mgfrst.gt.1) then
                    mstrt = mstop1(mgfrst-1)+1
                 else
                    mstrt = 1
                 end if
              ENDIF
#endif 
              !
              do i = mgfrst,mglast           ! i = 1,nmgrp
                 do j = mstrt,mstop1(i)
                    temp1 = e1b(j,1)*dxsfm(j)
                    temp2 = e1b(j,1)*dysfm(j)
                    temp3 = e1b(j,1)*dzsfm(j)
                    do jj = mstrt,mstop1(i)
                       dxe1bm(jj+ntot,1) = dxe1bm(jj+ntot,1) + temp1
                       dye1bm(jj+ntot,1) = dye1bm(jj+ntot,1) + temp2
                       dze1bm(jj+ntot,1) = dze1bm(jj+ntot,1) + temp3 
                    end do
                 end do
                 mstrt = mstop1(i)+1
              end do
              !
              If (natorb(nqm) .gt. 1) then
                 do i = 2,10
                    II = INDX1E + I
                    do j = atfrst,atlast     ! j = 1, n
                       jj = j + ntot
                       dxe1bm(jj,i) = dxint(j,i)*scale(j)
                       dye1bm(jj,i) = dyint(j,i)*scale(j)
                       dze1bm(jj,i) = dzint(j,i)*scale(j)
                       !JG
                       dxe1bq(ii) = dxe1bq(ii) + dxe1bm(jj,i)      ! dxint(j,i)*scale(j)
                       dye1bq(ii) = dye1bq(ii) + dye1bm(jj,i)      ! dyint(j,i)*scale(j)
                       dze1bq(ii) = dze1bq(ii) + dze1bm(jj,i)      ! dzint(j,i)*scale(j)
                       e1bdxs(ii,NQMGROUP) = e1bdxs(ii,NQMGROUP)+ e1b(j,i)*dxsfq(j)
                       e1bdys(ii,NQMGROUP) = e1bdys(ii,NQMGROUP)+ e1b(j,i)*dysfq(j)
                       e1bdzs(ii,NQMGROUP) = e1bdzs(ii,NQMGROUP)+ e1b(j,i)*dzsfq(j)
                    end do
                    !JG
                    mstrt = 1
                    ! go parallel
#if KEY_PARALLEL==1
                    IF (.NOT. QMPI) THEN
                       if(mgfrst.gt.1) then
                          mstrt = mstop1(mgfrst-1)+1
                       else
                          mstrt = 1
                       end if
                    ENDIF
#endif 
                    do ii = mgfrst,mglast           ! ii = 1,nmgrp
                       do j = mstrt,mstop1(ii)
                          temp1 = e1b(j,i)*dxsfm(j)
                          temp2 = e1b(j,i)*dysfm(j)
                          temp3 = e1b(j,i)*dzsfm(j)
                          do jj = mstrt,mstop1(ii)
                             dxe1bm(jj+ntot,i) = dxe1bm(jj+ntot,i) + temp1
                             dye1bm(jj+ntot,i) = dye1bm(jj+ntot,i) + temp2
                             dze1bm(jj+ntot,i) = dze1bm(jj+ntot,i) + temp3
                          end do
                       end do
                       mstrt = mstop1(ii)+1
                    end do
                 end do         ! i = 2,10
                 INDX1E = INDX1E + 10
              Else
                 INDX1E = INDX1E + 1
                 do i = 2,10
                    do j = atfrst,atlast       ! j = 1, n
                       jj = j + ntot
                       dxe1bm(jj,i) = zero
                       dye1bm(jj,i) = zero
                       dze1bm(jj,i) = zero
                       !
                       !                           dxe1bq(jj,i) = zero
                       !                           dye1bq(jj,i) = zero
                       !                           dze1bq(jj,i) = zero
                       !                           e1bdxs(jj,i) = zero
                       !                           e1bdys(jj,i) = zero
                       !                           e1bdzs(jj,i) = zero
                    end do
                 end do
              Endif
              !DCC+
              ! . Put the nuclear/nuclear terms into the derivative arrays.
              ! . If include QM-core/QM-core interactions.
              !
              If (qeqtrm(qmch)) then
                 !----------------------------------------------------------------------------
                 !   QM-Link ATOM: if qatom.gt.qstop, qatom = qm-link atom
                 !                 assign the appropriate atom number for the QM-Link atom
                 !
                 IQATOM = QATOM
                 IF(QATOM.GT.QSTOP) IQATOM = IMQLINK(QATOM-QSTOP)
                 !----------------------------------------------------------------------------

                 dxtmp(1:3) = zero
                 do j = atfrst,atlast             ! j = 1, n
                    temp1    = confac*scale(j)
                    dxtmp(1) = dxtmp(1)+temp1*dxenuc(j)
                    dxtmp(2) = dxtmp(2)+temp1*dyenuc(j)
                    dxtmp(3) = dxtmp(3)+temp1*dzenuc(j)
                 end do
                 ncount              =3*(IQATOM-1)+1
                 dxm(ncount:ncount+2)=dxm(ncount:ncount+2)+dxtmp(1:3)

                 !  Scale is a function of all coordinates of the qm group
                 !                       do i = qstrt,qstop
                 !  JG 3/21/01 To include QM-MM group for entire QM-MM interactions
                 Do i = QQSTRT,QQSTOP
                    ncount     =3*(i-1)+1
                    dxtmp(1:3) = zero
                    do j = atfrst,atlast           ! j = 1, n
                       temp1    = confac*enuc(j)
                       dxtmp(1) = dxtmp(1) + temp1*dxsfq(j)
                       dxtmp(2) = dxtmp(2) + temp1*dysfq(j)
                       dxtmp(3) = dxtmp(3) + temp1*dzsfq(j)
                    enddo
                    dxm(ncount:ncount+2)=dxm(ncount:ncount+2)+dxtmp(1:3)
                 End do
                 !DCC+
                 Do j = atfrst,atlast             ! j = 1, n
                    ncount       = 3*(jnbl(j)-1)+1
                    temp1        = confac*scale(j)
                    dxm(ncount)  =dxm(ncount)  -temp1*dxenuc(j)
                    dxm(ncount+1)=dxm(ncount+1)-temp1*dyenuc(j)
                    dxm(ncount+2)=dxm(ncount+2)-temp1*dzenuc(j)
                 End do
                 !DCC-
                 mstrt = 1
                 ! go parallel
#if KEY_PARALLEL==1
                 IF (.NOT. QMPI) THEN
                    if(mgfrst.gt.1) then
                       mstrt = mstop1(mgfrst-1)+1
                    else
                       mstrt = 1
                    end if
                 ENDIF
#endif 
                 !
                 do i = mgfrst,mglast               ! i = 1,nmgrp
                    do j = mstrt,mstop1(i)
                       temp1   =confac*enuc(j)
                       dxtmp(1)=temp1*dxsfm(j)
                       dxtmp(2)=temp1*dysfm(j)
                       dxtmp(3)=temp1*dzsfm(j)
                       do jj = mstrt,mstop1(i)
                          ncount       =3*(jnbl(jj)-1)+1
                          dxm(ncount:ncount+2)=dxm(ncount:ncount+2)-dxtmp(1:3)
                       end do
                    end do
                    mstrt=mstop1(i) + 1
                 end do
              Endif                ! (qeqtrm(qmch))
              !DCC+
              !
              do i=1,10
                 do j=atfrst,atlast
                    jj=j+ntot
                    dxe1bm(jj,i) = confac*dxe1bm(jj,i)
                    dye1bm(jj,i) = confac*dye1bm(jj,i)
                    dze1bm(jj,i) = confac*dze1bm(jj,i)
                 end do
              end do
              !
              ntot = ntot + n
           Endif              ! n .gt. 0 .and. qatlab(qatom) .gt. 0
        End do                ! qatom = QSTRT,QSTOP
        !
     End if                ! (npr .le. 0)
  End do                   ! qgrp = 1,ngrp
  ! . Scale the derivatives correctly.
  !
  ! this part moved upward for parallelization 
  !     do i = 1,10
  !        do j = 1,ntot
  !           dxe1bm(j,i) = confac * dxe1bm(j,i)
  !           dye1bm(j,i) = confac * dye1bm(j,i)
  !           dze1bm(j,i) = confac * dze1bm(j,i)
  !        end do
  !     end do
  !
  DO I = 1,NDIM1E
     DO J = 1,NQMGROUP
        e1bdxs(i,J) = confac * e1bdxs(i,J)
        e1bdys(i,J) = confac * e1bdys(i,J)
        e1bdzs(i,J) = confac * e1bdzs(i,J)
     ENDDO
     dxe1bq(i) = confac * dxe1bq(i)
     dye1bq(i) = confac * dye1bq(i)
     dze1bq(i) = confac * dze1bq(i)
  ENDDO
  !
  !----------------------------------------------------------------------------
  !
  Return
END SUBROUTINE EAMPE2

!DCC#      SUBROUTINE EAMPCG (INBX, JNBX, JNBL, NDIM, UHF, GRAD, P, PA, PB,   &
!DCC#                         DXE1BM, DYE1BM, DZE1BM, DXE1BQ, DYE1BQ, DZE1BQ, &
!DCC#                         QGRPMM_local)
!DCC-
SUBROUTINE EAMPCG (DXM, INBX, JNBX, JNBL, NDIM, UHF, GRAD, P, PA, PB, &
     DXE1BM, DYE1BM, DZE1BM, DXE1BQ, DYE1BQ, DZE1BQ, &
     E1BDXS, E1BDYS, E1BDZS, XG, YG, ZG, QGRPMM_local,NDIM1E)
  !DCC+
  !------------------------------------------------------------------------
  !
  !     The derivatives arising from the QM/MM interactions are calculated
  !     here. The Dxe1bm, Dye1bm, Dze1bm, Dxe1bq, Dye1bq and Dze1bq arrays
  !     contain the derivative integrals that were calculated in EAMPCE.
  !
  use chm_kinds
  use dimens_fcm
  use number
  !
  !...  use deriv
  use quantm
  use psf
  use sizes
  use qmlinkm
  use nbndqm_mod
  !DCC-
  use inbnd
  !DCC+
#if KEY_PBOUND==1
  use pbound
  use coordc
#endif 
  ! for parallel run
#if KEY_PARALLEL==1
  use parallel
#endif 
  implicit none
#if KEY_PBOUND==1 /*pbound*/
  real(chm_real) XPBC,YPBC,ZPBC,XSIZE2,YSIZE2,ZSIZE2
#endif /*     (pbound)*/
  !
  !
  Integer inbx(*), jnbx(*), jnbl(*), ndim
  Logical qgrpmm_local(*), uhf
  real(chm_real)  dxe1bm(ndim,10), dye1bm(ndim,10), dze1bm(ndim,10), &
       dxe1bq(*), dye1bq(*), dze1bq(*), &
       grad(3,*), p(*), pa(*), pb(*), &
       dxm(3*natom)
  !DCC-
  !     The arrays e1bdxs etc. contain the products of the one-electron
  !     integrals for QM/MM charge/valence interations and the derivative of
  !     the switching function calculated in EAMPE2.  The arrays xg etc.
  !     contain the coordinates of the group centers used to determine
  !     cutoffs.  DCC 2.22.93
  !
  !     real(chm_real) e1bdxs(ndim,10), e1bdys(ndim,10), e1bdzs(ndim,10),
  integer NDIM1E
  real(chm_real) e1bdxs(NDIM1E,*),e1bdys(NDIM1E,*),e1bdzs(NDIM1E,*), &
       xg(*), yg(*), zg(*)
  !JG
  real(chm_real)  c2ofnb, scent
  Integer kk, nstrt
  real(chm_real):: delta_xyz(3)
#if KEY_PBOUND==1 /*pbound*/
  logical:: pbc_chk(3)
  real(chm_real):: xyz_pbc(3),xyz_size(3),xyz_size2(3)
#endif /*     (pbound)*/

  !DCC+
  INTEGER  QSTRTGRP,QSTOPGRP,QQSTRT,QQSTOP,INDX1E,NQMGROUP
  !
  Integer i, ii, ij, ipair, j, jj, k, matm, mgrp, mstop, mstrt, n, &
       n1, n2, n3, nbg, npr, ntot, numqm, qatom, qfirst, &
       qgrp, qstop, qstrt
  real(chm_real)  denfac(10), dxint, dyint, dzint, pden
  real(chm_real) :: temp1,temp2,temp3,temp4,temp5,temp_xyz(3)
  !
  ! go parallel
  INTEGER ATFRST,ATLAST,natom2,qnum
  INTEGER MGFRST,MGLAST,NODNGM
  integer itemp1, mstop1(maxgrp), nmgrp, matm2, ncount
  !
  !
  ! . Do some initialisation.
#if KEY_PBOUND==1 /*pbound*/
  xyz_size(1)=XSIZE
  xyz_size(2)=YSIZE
  xyz_size(3)=ZSIZE
  XYZ_SIZE2(1:3) = half*xyz_size(1:3)
  If(qBoun) then
     IF(.not.qCUBoun) CALL WRNDIE(-5,'<EAMPCG>', 'Other than CUBoundary not supported')
  End if
#endif /*     (pbound)*/
  !DCC-
  c2ofnb = ctofnb * ctofnb
  natom2 = 2*natom
  !
  !DCC+
  n = (norbs * (norbs + 1)) / 2
  If (uhf) then
     p(1:n)  = pa(1:n)+pb(1:n)
  Else
     p(1:n)  = two*pa(1:n)
  Endif
  !
  ! . Fill the array of density factors (off-diagonal elements * 2).
  denfac(1:10)= two
  denfac(1)   = one
  denfac(3)   = one
  denfac(6)   = one
  denfac(10)  = one
  ! . Loop over the QM groups.
  nbg    = 0
  ntot   = 0
  numqm  = 0
  qfirst = 0
  !
  ! JG 3/21/01 To include full QM/MM interaction for QM group partition
  QSTRTGRP = 0
  INDX1E = 0
  NQMGROUP = 0
  !--jg
  Do qgrp = 1,ngrp
     npr    = inbx(qgrp) - qfirst
     qfirst = inbx(qgrp)
     qstrt  = igpbs(qgrp) + 1
     qstop  = igpbs(qgrp + 1)
     ! . Loop over the MM groups.
     n = 0
     If (npr .le. 0) then
        If (.not. qgrpmm_local(qgrp)) numqm = numqm + qstop - qstrt + 1
     Else
        !
        ! JG 3/21/01
        IF(QSTRTGRP.EQ.0) THEN
           QSTRTGRP=QSTRT
           QSTOPGRP=QSTRT+NATQM-1
        ENDIF
        !
        ! go parallel
        nmgrp = 0
        itemp1 = 0
        !--jg
        do ipair = 1,npr
           nbg = nbg + 1
           mgrp = Abs(jnbx(nbg))
           mstrt = igpbs(mgrp) + 1
           mstop = igpbs(mgrp + 1)
           ! . Fill the atom interaction array.
           delta_xyz(1) = xg(qgrp) - xg(mgrp)
           delta_xyz(2) = yg(qgrp) - yg(mgrp)
           delta_xyz(3) = zg(qgrp) - zg(mgrp)
           ! JG 6/1/01
           ! ADD SIMPLE PERIODIC BOUNDARY CONDITIONS FOR QM/MM INTERACTION
#if KEY_PBOUND==1 /*pbound*/
           If(qBoun .and. qCUBoun) then
              do ij=1,3
                 if(abs(delta_xyz(i)).gt.xyz_size2(i)) then
                    delta_xyz(i) = delta_xyz(i)-sign(xyz_size(i),delta_xyz(i))
                 end if
              end do
           End if
#endif /*  (pbound)*/
           scent = delta_xyz(1)*delta_xyz(1)+delta_xyz(2)*delta_xyz(2)+delta_xyz(3)*delta_xyz(3)

           If ((.NOT.QMCUTF) .OR. (scent .lt. c2ofnb)) then
              nmgrp = nmgrp + 1
              mstop1(nmgrp) = itemp1 + mstop - mstrt + 1
              itemp1 = mstop1(nmgrp)
              !
              do matm = mstrt,mstop
                 if(QATLAB(matm).le.90) then 
                    n = n + 1
                    jnbl(n) = matm
                 end if
              end do
           Endif
        end do          ! ipair = 1,npr
        ! . Loop over the QM atoms.
        !DCC-
        ! . If include MM-charge/QM-valence energy.
        If (qeqtrm(qmee)) then
           !
           !      QM-Link Atom case:  add the qm-link atoms, nqmlnk
           !
           ! JG 3/21/01
           QQSTRT = QSTRT
           QQSTOP = QSTOP
           IF(QNBGRP) THEN
              QSTRT = QSTRTGRP
              QSTOP = QSTOPGRP
              NSTRT = QQSTRT - QSTRT + 1
              NUMQM = 0
              INDX1E = 0
           ELSE
              nstrt = numqm + 1
           ENDIF
           NQMGROUP = NQMGROUP + 1
           !
           ! go parallel
#if KEY_PARALLEL==1 /*paramain*/
           NODNGM = NMGRP / NUMNOD
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
           MGFRST=MYNOD*NODNGM + 1
           MGLAST=(MYNOD+1)*NODNGM
           IF(MYNOD.EQ.(NUMNOD-1)) MGLAST=NMGRP
           !
           IF(MYNOD.EQ.0) THEN
              ATFRST=1
           ELSE
              ATFRST=mstop1(mgfrst-1)+1
           END IF
           ATLAST=mstop1(mglast)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
           ATFRST=1
           ATLAST=N
           !
           MGFRST=1
           MGLAST=NMGRP
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
           IF (QMPI) THEN
              ATFRST=1
              ATLAST=N
              !
              MGFRST=1
              MGLAST=NMGRP
           ENDIF
#else /* (paramain)*/
           ATFRST=1
           ATLAST=N
           !
           MGFRST=1
           MGLAST=NMGRP
#endif /* (paramain)*/
           !
           !
           do qatom = QSTRT,QSTOP
              numqm = numqm + 1
              !--jg
              !DCC#               If (qatom .eq. qstrt) nstrt = numqm
              !
              !   This was NOT changed in the standard CHARMM....JG, 1/7/99
              !                 If (n .gt. 0 .and. qatlab(qatom) .ge. 0) then
              If (n .gt. 0 .and. qatlab(qatom) .gt. 0) then
                 n1 = n1st(numqm)
                 n2 = nmidle(numqm)
                 n3 = nlast(numqm)
                 ! . Evaluate the derivatives involving the first e1b integral.
                 dxint = zero
                 dyint = zero
                 dzint = zero
                 !JG
                 ii   = (n1 * (n1 + 1)) / 2
                 pden = denfac(1) * p(ii)
                 !DCC-
                 INDX1E = INDX1E+1
                 II = INDX1E
                 temp_xyz(1)= pden * e1bdxs(II,NQMGROUP)
                 temp_xyz(2)= pden * e1bdys(II,NQMGROUP)
                 temp_xyz(3)= pden * e1bdzs(II,NQMGROUP)
                 DO K = 0, (QQSTOP-QQSTRT)  
                    grad(1:3,nstrt+k) = grad(1:3,nstrt+k) - temp_xyz(1:3)
                 ENDDO
                 !DCC+
                 !
                 Do j = atfrst,atlast   ! j = 1,n
                    ncount       =3*(jnbl(j)-1)+1
                    dxm(ncount)  =dxm(ncount)  +pden * dxe1bm(ntot + j,1)
                    dxm(ncount+1)=dxm(ncount+1)+pden * dye1bm(ntot + j,1)
                    dxm(ncount+2)=dxm(ncount+2)+pden * dze1bm(ntot + j,1)
                 End do
                 !
                 Do i = (n2 + 1),n3
                    ii   = (i * (i + 1)) / 2
                    pden = denfac(1) * p(ii)
                    II = INDX1E
                    temp_xyz(1)= pden * e1bdxs(II,NQMGROUP)
                    temp_xyz(2)= pden * e1bdys(II,NQMGROUP)
                    temp_xyz(3)= pden * e1bdzs(II,NQMGROUP)
                    DO K = 0, (QQSTOP-QQSTRT)
                       grad(1:3,nstrt+k) = grad(1:3,nstrt+k) - temp_xyz(1:3)
                    END DO
                    Do j = atfrst,atlast   ! j = 1,n
                       ncount       =3*(jnbl(j)-1)+1
                       dxm(ncount)  =dxm(ncount)  +pden*dxe1bm(ntot+j,1)
                       dxm(ncount+1)=dxm(ncount+1)+pden*dye1bm(ntot+j,1)
                       dxm(ncount+2)=dxm(ncount+2)+pden*dze1bm(ntot+j,1)
                    End do
                 End do
                 ! . Evaluate the remaining integrals.
                 jj = 1
                 Do i = (n1 + 1),n2
                    ii = (i * (i - 1)) / 2 + n1 - 1
                    Do j = n1,i
                       ii = ii + 1
                       jj = jj + 1
                       pden = denfac(jj) * p(ii)
                       INDX1E = INDX1E+1
                       temp_xyz(1)= pden * e1bdxs(INDX1E,NQMGROUP)
                       temp_xyz(2)= pden * e1bdys(INDX1E,NQMGROUP)
                       temp_xyz(3)= pden * e1bdzs(INDX1E,NQMGROUP)
                       Do KK = 0,(QQSTOP-QQSTRT)
                          grad(1:3,nstrt+kk) = grad(1:3,nstrt+kk) - temp_xyz(1:3)
                       End do
                       !JG
                       !
                       Do k = atfrst,atlast   ! k = 1,n
                          ncount       =3*(jnbl(k)-1)+1
                          dxm(ncount)  =dxm(ncount)  +pden*dxe1bm(ntot+k,jj)
                          dxm(ncount+1)=dxm(ncount+1)+pden*dye1bm(ntot+k,jj)
                          dxm(ncount+2)=dxm(ncount+2)+pden*dze1bm(ntot+k,jj)
                       End do
                    End do
                 End do
                 ntot = ntot + n
              Endif     ! (n .gt. 0 .and. qatlab(qatom) .gt. 0)
           end do       ! qatom = QSTRT,QSTOP
           !DCC-
        Endif       ! (qeqtrm(qmee))
     End if         ! (npr .le. 0)
  End do            ! qgrp = 1,ngrp
  !JG
  ! Collect all integral derivatives, already summed over mm atoms.
  INDX1E = 0
  !
  DO NUMQM = 1,NATQM
     N1 = N1ST(NUMQM)
     N2 = NMIDLE(NUMQM)
     N3 = NLAST(NUMQM)
     JJ = 0
     DO I = N1,N2
        II = (I*(I-1))/2 + N1 - 1
        DO J = N1,I
           II = II + 1
           JJ = JJ + 1
           INDX1E = INDX1E + 1
           PDEN = DENFAC(JJ)*P(II)
           GRAD(1,NUMQM) = GRAD(1,NUMQM)-PDEN*DXE1BQ(INDX1E)
           GRAD(2,NUMQM) = GRAD(2,NUMQM)-PDEN*DYE1BQ(INDX1E)
           GRAD(3,NUMQM) = GRAD(3,NUMQM)-PDEN*DZE1BQ(INDX1E)
        ENDDO
     ENDDO
  ENDDO
  !JG
  !
  !------------------------------------------------------------------------
  RETURN
END SUBROUTINE EAMPCG
!
SUBROUTINE NAMPE(INBX,JNBX,NTOTN,NATMAX_GRP,NQMPR)

  !DCC+
  !------------------------------------------------------------------------
  ! Calculate space needed for the QM/MM atom pair interation list.
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use quantm
  use nbndqm_mod
  implicit none
  !
  ! . Passed variables.
  Integer   inbx(*), jnbx(*), NTOTN
  !
  INTEGER NTOT,QFIRST,NPR,QSTRT,QSTOP,QNUM,QGRP
  INTEGER N,NBG,MSTRT,MSTOP,MGRP,IPAIR,NATMAX_GRP,NQMPR
  !
  ! . Loop over the group non-bond lists - the QM groups are first.
  nbg    = 0
  ntot   = 0
  qfirst = 0
  NATMAX_GRP = 0
  NQMPR = 0
  Do qgrp = 1,ngrp
     npr    = inbx(qgrp) - qfirst
     qfirst = inbx(qgrp)
     qstrt  = igpbs(qgrp) + 1
     qstop  = igpbs(qgrp + 1)
     qnum   = qstop - qstrt + 1
     IF(QNBGRP) QNUM = NATQM
     ! . Loop over the MM groups.
     n = 0
     Do ipair = 1,npr
        nbg = nbg + 1
        mgrp  = jnbx(nbg)
        IF(MGRP.LT.0) MGRP=-MGRP
        mstrt = igpbs(mgrp) + 1
        mstop = igpbs(mgrp + 1)
        N = N + MSTOP - MSTRT +1
     end do
     ! . Loop over the QM atoms.
     NTOT = NTOT + N * QNUM
     !
     IF(NPR.NE.0) NATMAX_GRP = NATMAX_GRP + 1
     IF(NQMPR.LT.N) NQMPR = N
     !
  End do
  NTOTN=NTOT
  !
  !
  Return
END SUBROUTINE NAMPE


SUBROUTINE DQLINK4(DX,X,Y,Z,F,UHF,ECLASS)
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
  !
  ! for parallel run
#if KEY_PARALLEL==1
  use parallel
#endif 
  implicit none
  !
  real(chm_real) DX(*),X(*),Y(*),Z(*),F(*),ECLASS
  LOGICAL UHF
  !
  real(chm_real), parameter:: CONFAC=EV_TO_KCAL  ! 23.061D+00
  !  Local variables
  INTEGER I, J, K, L, M, N, II, JJ, LL, MM, I1, L1, M1, IJ
  INTEGER KK1,KK2,KK3,II2,JJ2,N16,N16I1,N16J
  INTEGER I1DIAG(4),NAOS,LINAO
  real(chm_real) XDTA(4,4),YDTA(4,4),ZDTA(4,4),XDTB(4,4),YDTB(4,4)
  real(chm_real) XDTC(4,4),YDTC(4,4),ZDTC(4,4),ZDTB(4,4),TINV(4,4)
  real(chm_real) DMMXA,DMMYA,DMMZA,DMMXB,DMMYB,DMMZB, &
       DMMXC,DMMYC,DMMZC
  real(chm_real) DL2XA,DL2YA,DL2ZA,DL2XB,DL2YB,DL2ZB, &
       DL2XC,DL2YC,DL2ZC
  real(chm_real) DMMXA1,DMMYA1,DMMZA1,DMMXB1,DMMYB1,DMMZB1, &
       DMMXC1,DMMYC1,DMMZC1
  real(chm_real) PF,PF1,PF2,XFAC,DELX,DELY,DELZ,RR2,RR1, &
       FACT1,XTMP(3)
  !
  ! go parallel
  real(chm_real) PAFCT
  INTEGER ncount
  !
  !
  NAOS   = NORBS-4*MQMLNK
  IJ = NAOS*(NAOS+1)/2
  N16 = 0
  PAFCT = 1.0D0
#if KEY_PARALLEL==1
  IF(MYNOD.NE.0 .AND. .NOT. QMPI) PAFCT=0.0D0
#endif 
  !
  !      LOOP OVER QM-LINK ATOMS
  !
  DO I = 1,MQMLNK
     L1 = 4*(I-1)
     II = IMQLINK(I)
     K = NAOS+4*(I-1)+1       !N1ST(II)
     L = K+3
     LINAO = K*(K-1)/2
     !      LOCATE POSITION OF DIAGONAL ELEMENTS
     DO J = K,L
        I1DIAG(J-K+1) = J*(J+1)/2
     ENDDO
     !      ZERO TEMPORARY VARIABLES
     DMMXA = 0.0D0
     DMMYA = 0.0D0
     DMMZA = 0.0D0
     DMMXB = 0.0D0
     DMMYB = 0.0D0
     DMMZB = 0.0D0
     DMMXC = 0.0D0
     DMMYC = 0.0D0
     DMMZC = 0.0D0

     N16J = N16
     DO J = 1,4
        DO K = 1,4
           N16J = N16J+1
           TINV(K,J) = MBTM(N16J)
           XDTA(K,J) = MDBTMMM(1,1,N16J)
           YDTA(K,J) = MDBTMMM(2,1,N16J)
           ZDTA(K,J) = MDBTMMM(3,1,N16J)
           XDTB(K,J) = MDBTMMM(1,2,N16J)
           YDTB(K,J) = MDBTMMM(2,2,N16J)
           ZDTB(K,J) = MDBTMMM(3,2,N16J)
           XDTC(K,J) = MDBTMMM(1,3,N16J)
           YDTC(K,J) = MDBTMMM(2,3,N16J)
           ZDTC(K,J) = MDBTMMM(3,3,N16J)
        ENDDO
     ENDDO
     !
     !      AO-HB BLOCKS
     DO J = 1,4
        DO K = 1,NAOS
           IJ = IJ+1
           PF = 2.0D0*PHO(LINAO+K)*F(IJ)

           ! For UHF-GHO ... PJ 12/2002
           IF (UHF) THEN
              PF = 2.0D0*PBHO(LINAO+K)*F(IJ)
           ENDIF

           DMMXA = DMMXA + XDTA(1,J)*PF
           DMMXB = DMMXB + XDTB(1,J)*PF
           DMMXC = DMMXC + XDTC(1,J)*PF

           DMMYA = DMMYA + YDTA(1,J)*PF
           DMMYB = DMMYB + YDTB(1,J)*PF
           DMMYC = DMMYC + YDTC(1,J)*PF

           DMMZA = DMMZA + ZDTA(1,J)*PF
           DMMZB = DMMZB + ZDTB(1,J)*PF
           DMMZC = DMMZC + ZDTC(1,J)*PF

        ENDDO
        !
        !      HB-other HB blocks
        !
        M1 = LINAO+NAOS
        DO I1 = 1,I-1
           N16I1 = 16*(I1-1)
           DL2XA = 0.0D0
           DL2YA = 0.0D0
           DL2ZA = 0.0D0
           DL2XB = 0.0D0
           DL2YB = 0.0D0
           DL2ZB = 0.0D0
           DL2XC = 0.0D0
           DL2YC = 0.0D0
           DL2ZC = 0.0D0

           KK3 = M1+4*(I1-1)+1
           DO L1 = 1,4
              IJ = IJ+1
              PF = 2.0D0*PHO(KK3)*F(IJ)

              ! For UHF-GHO ... PJ 12/2002
              IF (UHF) THEN
                 PF = 2.0D0*PBHO(KK3)*F(IJ)
              END IF

              PF1 = MBT (N16+J)*PF
              PF2 = MBT (N16I1+L1)*PF
              DMMXA = DMMXA+PF2*XDTA(1,J)
              DMMXB = DMMXB+PF2*XDTB(1,J)
              DMMXC = DMMXC+PF2*XDTC(1,J)
              DMMYA = DMMYA+PF2*YDTA(1,J)
              DMMYB = DMMYB+PF2*YDTB(1,J)
              DMMYC = DMMYC+PF2*YDTC(1,J)
              DMMZA = DMMZA+PF2*ZDTA(1,J)
              DMMZB = DMMZB+PF2*ZDTB(1,J)
              DMMZC = DMMZC+PF2*ZDTC(1,J)

              KK1 = N16I1+4*(L1-1)+1
              DL2XA = DL2XA+PF1*MDBTMMM(1,1,KK1)
              DL2YA = DL2YA+PF1*MDBTMMM(2,1,KK1)
              DL2ZA = DL2ZA+PF1*MDBTMMM(3,1,KK1)
              DL2XB = DL2XB+PF1*MDBTMMM(1,2,KK1)
              DL2YB = DL2YB+PF1*MDBTMMM(2,2,KK1)
              DL2ZB = DL2ZB+PF1*MDBTMMM(3,2,KK1)
              DL2XC = DL2XC+PF1*MDBTMMM(1,3,KK1)
              DL2YC = DL2YC+PF1*MDBTMMM(2,3,KK1)
              DL2ZC = DL2ZC+PF1*MDBTMMM(3,3,KK1)

           ENDDO

           II2 = IMQLINK(I1)
           JJ = JMQLINK(1,I1)
           LL = JMQLINK(2,I1)
           MM = JMQLINK(3,I1)
           ncount         = 3*(jj-1)+1
           dx(ncount)     = dx(ncount)     -DL2XA*CONFAC*PAFCT
           dx(ncount+1)   = dx(ncount+1)   -DL2YA*CONFAC*PAFCT
           dx(ncount+2)   = dx(ncount+2)   -DL2ZA*CONFAC*PAFCT

           ncount         = 3*(ll-1)+1
           dx(ncount)     = dx(ncount)     -DL2XB*CONFAC*PAFCT
           dx(ncount+1)   = dx(ncount+1)   -DL2YB*CONFAC*PAFCT
           dx(ncount+2)   = dx(ncount+2)   -DL2ZB*CONFAC*PAFCT

           ncount         = 3*(mm-1)+1
           dx(ncount)     = dx(ncount)     -DL2XC*CONFAC*PAFCT
           dx(ncount+1)   = dx(ncount+1)   -DL2YC*CONFAC*PAFCT
           dx(ncount+2)   = dx(ncount+2)   -DL2ZC*CONFAC*PAFCT

           ncount         = 3*(ii2-1)+1
           dx(ncount)     = dx(ncount)     + &
                (DL2XA+DL2XB+DL2XC)*CONFAC*PAFCT
           dx(ncount+1)   = dx(ncount+1)   + &
                (DL2YA+DL2YB+DL2YC)*CONFAC*PAFCT
           dx(ncount+2)   = dx(ncount+2)   + &
                (DL2ZA+DL2ZB+DL2ZC)*CONFAC*PAFCT
        ENDDO
        !
        !      HB-HB BLOCKS
        !
        DO K = 1,J
           IJ = IJ+1
           XFAC = 2.0D0
           IF(K.EQ.J) XFAC = 1.0D0
           DO L = 1,4
              PF = XFAC*PHO(I1DIAG(L))*F(IJ)

              ! For UHF-GHO ... PJ 12/2002
              IF (UHF) THEN
                 PF = XFAC*PBHO(I1DIAG(L))*F(IJ)
              END IF

              DMMXA = DMMXA+PF*(XDTA(L,J)*TINV(L,K)+ &
                   TINV(L,J)*XDTA(L,K))
              DMMXB = DMMXB+PF*(XDTB(L,J)*TINV(L,K)+ &
                   TINV(L,J)*XDTB(L,K))
              DMMXC = DMMXC+PF*(XDTC(L,J)*TINV(L,K)+ &
                   TINV(L,J)*XDTC(L,K))

              DMMYA = DMMYA+PF*(YDTA(L,J)*TINV(L,K)+ &
                   TINV(L,J)*YDTA(L,K))
              DMMYB = DMMYB+PF*(YDTB(L,J)*TINV(L,K)+ &
                   TINV(L,J)*YDTB(L,K))
              DMMYC = DMMYC+PF*(YDTC(L,J)*TINV(L,K)+ &
                   TINV(L,J)*YDTC(L,K))

              DMMZA = DMMZA+PF*(ZDTA(L,J)*TINV(L,K)+ &
                   TINV(L,J)*ZDTA(L,K))
              DMMZB = DMMZB+PF*(ZDTB(L,J)*TINV(L,K)+ &
                   TINV(L,J)*ZDTB(L,K))
              DMMZC = DMMZC+PF*(ZDTC(L,J)*TINV(L,K)+ &
                   TINV(L,J)*ZDTC(L,K))
           ENDDO
        ENDDO
     ENDDO

     DMMXA1 = CONFAC*DMMXA*PAFCT
     DMMYA1 = CONFAC*DMMYA*PAFCT
     DMMZA1 = CONFAC*DMMZA*PAFCT
     DMMXB1 = CONFAC*DMMXB*PAFCT
     DMMYB1 = CONFAC*DMMYB*PAFCT
     DMMZB1 = CONFAC*DMMZB*PAFCT
     DMMXC1 = CONFAC*DMMXC*PAFCT
     DMMYC1 = CONFAC*DMMYC*PAFCT
     DMMZC1 = CONFAC*DMMZC*PAFCT
     !
     JJ = JMQLINK(1,I)
     LL = JMQLINK(2,I)
     MM = JMQLINK(3,I)

     ncount         = 3*(jj-1)+1
     dx(ncount)     = dx(ncount)    -DMMXA1
     dx(ncount+1)   = dx(ncount+1)  -DMMYA1
     dx(ncount+2)   = dx(ncount+2)  -DMMZA1

     ncount         = 3*(ll-1)+1
     dx(ncount)     = dx(ncount)    -DMMXB1
     dx(ncount+1)   = dx(ncount+1)  -DMMYB1
     dx(ncount+2)   = dx(ncount+2)  -DMMZB1

     ncount         = 3*(mm-1)+1
     dx(ncount)     = dx(ncount)    -DMMXC1
     dx(ncount+1)   = dx(ncount+1)  -DMMYC1
     dx(ncount+2)   = dx(ncount+2)  -DMMZC1

     ncount         = 3*(ii-1)+1
     dx(ncount)     = dx(ncount)    +DMMXA1+DMMXB1+DMMXC1
     dx(ncount+1)   = dx(ncount+1)  +DMMYA1+DMMYB1+DMMYC1
     dx(ncount+2)   = dx(ncount+2)  +DMMZA1+DMMZB1+DMMZC1
     N16 = N16+16
     !
     !      INCLUDE NUCLEAR INTERACTIONS BETWEEN MM-BOUNDARY ATOMS
     !

     !
     ! avoid double count in beta correction UHF case ... PJ 12/2002
     !
     IF (UHF) GOTO 999
     !
     ! commented, bug fixed for multiple boundary atoms ... PJ 12/2002
     !        ECLASS = 0.0D0
     !
     DO M = 1,2
        JJ = JMQLINK(M,I)
        M1 = M+1
        DO N = M1,3
           JJ2 = JMQLINK(N,I)
           DELX = X(JJ2)-X(JJ)
           DELY = Y(JJ2)-Y(JJ)
           DELZ = Z(JJ2)-Z(JJ)
           RR2 = DELX*DELX+DELY*DELY+DELZ*DELZ
           RR1 = SQRT(RR2)
           !------------------------------------
           IF(QMPERT) THEN
              FACT1 = CCELEC*CG(JJ)*RLAMB0*CG(JJ2)*RLAMB0/RR1
           ELSE
              FACT1 = CCELEC*CG(JJ)*CG(JJ2)/RR1
           ENDIF
           !------------------------------------
           ECLASS  = ECLASS+FACT1
           FACT1   = FACT1/RR2
           XTMP(1) = DELX*FACT1*PAFCT
           XTMP(2) = DELY*FACT1*PAFCT
           XTMP(3) = DELZ*FACT1*PAFCT

           ncount  = 3*(jj-1)+1

           dx(ncount)      = dx(ncount)  +XTMP(1)
           dx(ncount+1)    = dx(ncount+1)+XTMP(2)
           dx(ncount+2)    = dx(ncount+2)+XTMP(3)

           ncount  = 3*(jj2-1)+1
           dx(ncount)      = dx(ncount)  -XTMP(1)
           dx(ncount+1)    = dx(ncount+1)-XTMP(2)
           dx(ncount+2)    = dx(ncount+2)-XTMP(3)
        ENDDO
     ENDDO
     !
     ! jump tag for UHF-GHO ... PJ 12/2002
     !
999  CONTINUE

  ENDDO
  RETURN
END SUBROUTINE DQLINK4

SUBROUTINE ZEROTM(A,N)
  use chm_kinds
  implicit none
  real(chm_real) A(*)
  INTEGER N

  A(1:N) = 0.0D0

  RETURN
END SUBROUTINE ZEROTM

SUBROUTINE QMDCOM(PGAS,PENZYM,HGAS,HENZYM,H1ES, &
     EQUANT,ELECT,ENUCLR,ELEGAS,ENUGAS,ECLASS,ATHEAT, &
     EVER,ESTA,EDIS,EPOL,NORBS,LINQLK)
  !
  !   QM/MM energy decompostion calculations as described in
  !   J. Gao, X. Xia, Science, 258, 631-635 (1992)
  !   "A priori evaluation of aqueous polarization effects through
  !    Monte Carlo QM-MM simulations."
  !
  !     EXS  = total qm/mm interaction energy
  !     Ever = vertical interaction energy
  !     Epol = polarization energy
  !     Esta = qm/mm stabilization energy
  !     Edis = qm distortion energy
  !
  !     EXS  = Ever + Epol
  !     Epol = Esta + Edis
  !
  use chm_kinds
  use consta, only : BOHRR,EV_TO_KCAL,AU_TO_EV
  use number, only : zero, one, two
  use vector
  implicit none
  real(chm_real) PGAS(*),PENZYM(*),HGAS(*),HENZYM(*),H1ES(*)
  real(chm_real) EQUANT,ELECT,ENUCLR,ELEGAS,ENUGAS,ECLASS,ATHEAT, &
       EVER,ESTA,EDIS,EPOL
  INTEGER LINQLK,NORBS
  !
  INTEGER I,J,IJ
  real(chm_real) EXS
  !
  CALL SUBVEC(HENZYM,HGAS,H1ES,LINQLK)
  EVER = zero
  ESTA = zero
  IJ = 0
  DO I = 1,NORBS
     DO J = 1,I-1
        IJ = IJ+1
        EVER = EVER+two*PGAS(IJ)*H1ES(IJ)
        ESTA = ESTA+two*PENZYM(IJ)*H1ES(IJ)
     ENDDO
     IJ = IJ+1
     EVER = EVER+PGAS(IJ)*H1ES(IJ)
     ESTA = ESTA+PENZYM(IJ)*H1ES(IJ)
  ENDDO
  EXS = EQUANT-((ELEGAS+ENUGAS)*EV_TO_KCAL+ATHEAT)-ECLASS
  ESTA = (ESTA-EVER)*EV_TO_KCAL
  EVER = (EVER+ENUCLR-ENUGAS)*EV_TO_KCAL      ! 23.061D0
  EPOL = EXS-EVER
  EDIS = EPOL-ESTA
  !
  RETURN
END SUBROUTINE QMDCOM

SUBROUTINE PRQFEP(OUTU2,NUMSTP,TIME)
  !
  !      PRINTS OUT THE QM-MM ELECTROSTATIC FREE ENERGY PERTURBATION
  !      RESULTS.
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  use stream
  !
  use sizes
  use quantm
  implicit none
  INTEGER NUMSTP,OUTU2,I
  real(chm_real) TIME,EXQMMM
  !
  IF(QMPERT) THEN
     IF(PRNLEV.GE.2) THEN
        WRITE(OUTU,'(1X,A,I8,A)') 'QM-MM Electrostatic Free Energy Perturbation for',NUMSTP,' Steps '
        WRITE(OUTU,'(2A)') '    L(ref)  ->  L(pert)     Delta G(ref->pert)    RMS deviation'
        WRITE(OUTU,15) RLAMB0,RLAMB1,EQPRA(QPT1)      ! ,EQPR2A(QPT1)
        WRITE(OUTU,15) RLAMB0,RLAMB2,EQPRA(QPT2)      ! ,EQPR2A(QPT2)
     END IF
  ENDIF
  IF(QDECOM) THEN
     IF(PRNLEV.GE.2) THEN
        WRITE(OUTU,'(1X,A,I8,A,F8.1,A)') 'QM-MM Electrostatic Energy Decomposition for',NUMSTP,' Steps '
        WRITE(OUTU,25)
     END IF
     EXQMMM = EQPRA(QVER)+EQPRA(QPOL)
     IF(PRNLEV.GE.2) WRITE(OUTU,26) 'AVER ',EXQMMM,EQPRA(QVER),EQPRA(QPOL),EQPRA(QSTA),EQPRA(QDIS)
     EXQMMM = SQRT(EQPR2A(QVER)**2+EQPR2A(QPOL)**2)
     IF(PRNLEV.GE.2) THEN
        WRITE(OUTU,26) 'FLUC ',EXQMMM,EQPR2A(QVER),EQPR2A(QPOL),EQPR2A(QSTA),EQPR2A(QDIS)
        WRITE(OUTU,'(A,F11.3,A,F7.3)') '     Egas = ',EQPRA(QGAS),' +/- ',EQPR2A(QGAS)
     END IF
  ENDIF
15 FORMAT(4X,F5.3,7X,F5.3,10X,F7.3,10X,F7.3)
25 FORMAT(10X,'E(XS)',9X,'E(vert)',9X,'E(pol)',9X,'E(stab)',8X,'E(dist)')
26 FORMAT(A,5(F11.3,4X))

  RETURN
END SUBROUTINE PRQFEP
!
! namkh 08/08/04
! QM/MM-Ewald
!----------------------------------------------------------------------
SUBROUTINE SETUP_KTABLE(NATM,X,Y,Z,MAXKK, &
     KMX,KMY,KMZ, &
     KTABXC,KTABXS, &
     KTABYC,KTABYS, &
     KTABZC,KTABZS)
  !
  !    Setup k-table array for K-space ewald summation.
  !    Author:
  !
  !
  use chm_kinds
  use number
  use exfunc
  use dimens_fcm
  use image
  use consta
  use quantm
  !
  ! for parallel run
#if KEY_PARALLEL==1
  use parallel
#endif 
  implicit none
  !
  INTEGER NATM, MAXKK, KMX, KMY, KMZ
  real(chm_real)  X(*),Y(*),Z(*)
  real(chm_real) KTABXC(NATM,MAXKK),KTABXS(NATM,MAXKK)
  real(chm_real) KTABYC(NATM,MAXKK),KTABYS(NATM,MAXKK)
  real(chm_real) KTABZC(NATM,MAXKK),KTABZS(NATM,MAXKK)
  !
  !
  LOGICAL OK
  real(chm_real)  XTLINV(6)
  real(chm_real)  BOXL(3)
  !
  INTEGER I, IPT, KX, KY, KZ, KSQ, J, KSY,KSZ
  real(chm_real)  RKX(3), RKY(3), RKZ(3), MKV(3)
  ! go parallel
  INTEGER ATFRST,ATLAST
  !
  !
  !     Calculate Volume of System and Reciprocal Space Lattice Vector
  CALL INVT33S(XTLINV, XTLABC, OK)
  !
  ! go parallel
#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
  ATFRST=1+IPARPT(MYNOD)
  ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
  ATFRST=1
  ATLAST=NATM
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
  IF (QMPI) THEN
     ATFRST=1
     ATLAST=NATM
  ENDIF
#else /* (paramain)*/
  ATFRST=1
  ATLAST=NATM
#endif /* (paramain)*/
  !
  ! construct ktable for each atom
  ! initialization
  !...##IF PARALLEL
  !...##IF PARAFULL
  !     DO I=1,NATM
  !        IF(I.LT.ATFRST.OR.I.GT.ATLAST) THEN
  !           DO J=1,MAXKK
  !              KTABXC(I,J)=ZERO
  !              KTABYC(I,J)=ZERO
  !              KTABZC(I,J)=ZERO
  !              KTABXS(I,J)=ZERO
  !              KTABYS(I,J)=ZERO
  !              KTABZS(I,J)=ZERO
  !           END DO
  !        END IF
  !     END DO
  !...##ENDIF
  !...##ENDIF
  !
  IPT = 0
  DO KX = 0, KMX
     IF(KX.EQ.0) THEN
        KSY = 0
     ELSE
        KSY = -KMY
     ENDIF
     RKX(1) = TWOPI*KX*XTLINV(1)
     RKX(2) = TWOPI*KX*XTLINV(2)
     RKX(3) = TWOPI*KX*XTLINV(4)
     DO KY = KSY, KMY
        IF(KX.EQ.0.AND.KY.EQ.0) THEN
           KSZ = 1
        ELSE
           KSZ = -KMZ
        ENDIF
        RKY(1) = TWOPI*KY*XTLINV(2)
        RKY(2) = TWOPI*KY*XTLINV(3)
        RKY(3) = TWOPI*KY*XTLINV(5)
        DO KZ = KSZ, KMZ
           RKZ(1) = TWOPI*KZ*XTLINV(4)
           RKZ(2) = TWOPI*KZ*XTLINV(5)
           RKZ(3) = TWOPI*KZ*XTLINV(6)
           KSQ = KX*KX + KY*KY + KZ*KZ
           IF (KSQ.LE.KSQMAXQ .AND. KSQ.NE.0) THEN
              IPT = IPT + 1
              MKV(1:3)  = RKX(1:3) + RKY(1:3) + RKZ(1:3)
              DO I = ATFRST, ATLAST        ! I = 1, NATM
                 IF(qatlab(I).lt.0) THEN
                    !  For X coordinates
                    KTABXC(I,IPT) = COS(MKV(1)*X(I))
                    KTABXS(I,IPT) = SIN(MKV(1)*X(I))
                    !  For Y coordinates
                    KTABYC(I,IPT) = COS(MKV(2)*Y(I))
                    KTABYS(I,IPT) = SIN(MKV(2)*Y(I))
                    !  For Z coodinates
                    KTABZC(I,IPT) = COS(MKV(3)*Z(I))
                    KTABZS(I,IPT) = SIN(MKV(3)*Z(I))
                 END IF
              END DO
              DO J=1,NATQM
                 I=IABS(QMINB(J))
                 !  For X coordinates
                 KTABXC(I,IPT) = COS(MKV(1)*X(I))
                 KTABXS(I,IPT) = SIN(MKV(1)*X(I))
                 !  For Y coordinates
                 KTABYC(I,IPT) = COS(MKV(2)*Y(I))
                 KTABYS(I,IPT) = SIN(MKV(2)*Y(I))
                 !  For Z coodinates
                 KTABZC(I,IPT) = COS(MKV(3)*Z(I))
                 KTABZS(I,IPT) = SIN(MKV(3)*Z(I))
              END DO
           END IF
        END DO
     END DO
  END DO
  !
  RETURN
END SUBROUTINE SETUP_KTABLE
!
SUBROUTINE QEWALD_CORE(ENUCLR2)
  !
  !     Compute the interaction of Ewald potential with CORE in QM atoms
  !     Author:
  !
  !     ENUCLR2 : Sum (Core(I)*V(I))
  !
  use am1parm,only:core
  use chm_kinds
  use dimens_fcm
  use exfunc
  use consta, only : BOHRR, AU_TO_EV
  !
  use quantm
  implicit none
  !
  real(chm_real) ENUCLR2
  !
  INTEGER I, NI
  real(chm_real) EVC, A0C, EWDPOT, RFLAMB
  !
  ENUCLR2 = 0.0D0
  EVC     = AU_TO_EV          ! 27.21D0
  A0C     = BOHRR             ! 0.529167D0

  RFLAMB = 1.0D0
  IF(QMPERT) RFLAMB = RLAMBF
  !
  DO I = 1, NATQM
     NI = NAT(I)
     EWDPOT = RFLAMB*(EMPOT(I)+0.5D0*ESLF(I))*EVC*A0C
     ENUCLR2 = ENUCLR2 + CORE(NI)*EWDPOT
  END DO
  !
  RETURN
END SUBROUTINE QEWALD_CORE
!
SUBROUTINE QEWALDP(X,Y,Z,XI,YI,ZI,QDONE)
  !
  !    Compute the potential at QM atom position from Ewald summation
  !    Author:
  !
  !
  use ewald,only: kappa
  use chm_kinds
  use number
  use dimens_fcm
  use consta
  !...  use coord
  use psf
  use quantm
  !
  ! for parallel run
#if KEY_PARALLEL==1
  use parallel
#endif 
  !
  implicit none
  !
  INTEGER NATM
  LOGICAL QDONE
  !
  real(chm_real)  X(*), Y(*), Z(*), XI(*), YI(*), ZI(*)
  real(chm_real)  EPOT(NATQM),ESFACT, ESLF2(NATQM),PotBGRND
  INTEGER I
  !
  INTEGER ATFRST,ATLAST,NODELE
  !
  real(chm_real),parameter :: EVCALC=EV_TO_KCAL, &   ! EVCALC=23.061D0
       EVC   =AU_TO_EV  , &   ! EVC=27.21D0
       A0C   =BOHRR           ! A0C=0.529167D0
  !
  NATM       = NATOM
  ESFACT     = -KAPPA/SQRT(PI)
  !
  ! go parallel
#if KEY_PARALLEL==1 /*paramain*/
  NODELE = NATQM / NUMNOD
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
  ATFRST=MYNOD*NODELE + 1
  ATLAST=(MYNOD+1)*NODELE
  IF(MYNOD.EQ.(NUMNOD-1)) ATLAST=NATQM
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
  ATFRST=1
  ATLAST=NATQM
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
  IF (QMPI) THEN
     ATFRST=1
     ATLAST=NATQM
  ENDIF
#else /* (paramain)*/
  ATFRST=1
  ATLAST=NATQM
#endif /* (paramain)*/
  !
  !
  IF(QDONE) THEN
     ESLF(1:NATQM)    = ZERO
     ESLF2(1:NATQM)   = ZERO

     !    ignoring QM images, then Eslf term should be zero
     !
     IF(NOQMIM) RETURN
     ESLF(ATFRST:ATLAST)    = TWO*ESFACT*CHAG(ATFRST:ATLAST)    ! I = 1, NATQM
  ELSE
     !
     EPOT(1:NATQM) = zero
     EMPOT(1:NATQM)= zero
     ! for the background charge correction
     IF(QMPERT) THEN
        PotBGRND = -PI / (KAPPA*KAPPA*VOLME)
        PotBGRND = PotBGRND*EVCALC*EVC*A0C
        EBKCHG(1:3)= half*BCKCHG(1:3)*PotBGRND
     END IF
  END IF
  !
  CALL RSPACE_POT(QDONE,EPOT,ESLF2,NATM,X,Y,Z,XI,YI,ZI,CG)
  !
  CALL KSPACE_POT(QDONE,EPOT,ESLF2,NATM,X,Y,Z,CG,MAXKVQ,KMAXXQ,KMAXYQ,KMAXZQ,  &
       PKVECQ,PKTABXCQ,PKTABXSQ,PKTABYCQ,PKTABYSQ,PKTABZCQ,PKTABZSQ &
       )
  !
  IF(QDONE) THEN
     ESLF(1:NATQM) = ESLF(1:NATQM) + ESLF2(1:NATQM)
  ELSE
     EMPOT(1:NATQM) = EMPOT(1:NATQM) + EPOT(1:NATQM)
  END IF
  !
  RETURN
END SUBROUTINE QEWALDP
!
SUBROUTINE QEWALDD(NATM,NAQM,X,Y,Z,XI,YI,ZI,DXM,GRAD,DXYZM)
  !
  !    Compute the gradient at QM atom for Ewald summation by all atoms
  !
  !    The gradient from reciprocal sum is divided from real space 
  !    contribution, since the virial needs to be taken cared for
  !    reciprocal contribution.
  !
  !    The gradient needs to be added after CALL VIRAL (in energy.src).
  !
  !    Author:
  !
  !
  use chm_kinds
  use number
  use dimens_fcm
  use consta, only : EV_TO_KCAL,AU_TO_EV,BOHRR
  !...  use coord
  !...  use deriv
  use psf
  use quantm
  ! for parallel run
#if KEY_PARALLEL==1
  use parallel
#endif 
  implicit none
  !
  INTEGER NATM, NAQM
  real(chm_real)  GRAD(3,NAQM),DXM(3*NATM),DXYZM(3,NATM)
  !
  !
  INTEGER I, J, IQ, JQ,ncount
  real(chm_real) X(*),Y(*),Z(*), XI(*),YI(*),ZI(*)
  real(chm_real) DEX(NATOM),DEY(NATOM),DEZ(NATOM)
  !
  !
  !
  real(chm_real) :: CFACT,DERQE(NATQM)
  real(chm_real),parameter :: EVCALC=EV_TO_KCAL, &  ! EVCALC=23.061D0
       EVC   =AU_TO_EV,   &  ! EVC=27.21D0
       A0C   =BOHRR          ! A0C=0.529167D0

  !
  CFACT = EVCALC*EVC*A0C
  ! QMPERT
  IF(QMPERT) RLAMBF = RLAMB0
  !
  DO I = 1, NATQM
     IQ       = IABS(QMINB(I))
     DERQE(I) = CHAG(I)*CFACT
  END DO

  DEX(1:NATOM)=zero
  DEY(1:NATOM)=zero
  DEZ(1:NATOM)=zero
  !
  CALL RSPACE_GRAD(NATQM,NATOM,X,Y,Z,XI,YI,ZI, &
       DEX,DEY,DEZ,CG,DERQE)
  !
  CALL SUM_GRAD(NATQM,NATOM,DXM,DEX,DEY,DEZ,GRAD)
  !
  !
  !
  ! Reciprocal contribution computed here, and saved into DXYZM
  ! do virial setup: initialization
  !C    EWVIRIAL(1:9)=ZERO

  DEX(1:NATOM)      = zero
  DEY(1:NATOM)      = zero
  DEZ(1:NATOM)      = zero
  DXYZM(1:3,1:NATOM)= zero
  !
  IQ = NATM
  JQ = 2*NATM
  !
  CALL KSPACE_GRAD(NATQM,NATOM,DEX,DEY,DEZ,CG,DERQE,MAXKVQ,KMAXXQ,KMAXYQ,KMAXZQ, &
       PKVECQ,PKTABXCQ,PKTABXSQ,PKTABYCQ,PKTABYSQ,PKTABZCQ,PKTABZSQ  &
       )
  !
  DO J = 1, NATM
     dxyzm(1,j) = dex(j)
     dxyzm(2,j) = dey(j)
     dxyzm(3,j) = dez(j)
  END DO
  !
  ! collect force
  ! go parallel
#if KEY_PARALLEL==1 /*paramain*/
  IF(NUMNOD.NE.1 .AND. .NOT. QMPI) THEN
     call psync()
     call GCOMBr(DXYZM,3*NATM) ! MH2023: extend the gcomb interface for fix
  END IF
#endif /* (paramain)*/
  !
  RETURN
END SUBROUTINE QEWALDD
!
SUBROUTINE RSPACE_POT(QDONE,EPOT,ESLF2,NATM,X,Y,Z,XI,YI,ZI,CG &
     )
  !
  !    Compute the potential at QM atom position for Real space ewald summation
  !    Author:
  !
  !
  use ewald,only:kappa,erfmod
  use erfcd_mod,only: erfcd
  use chm_kinds
  use number
  use dimens_fcm
  use image
  use consta
  use quantm
#if KEY_PBOUND==1
  use pbound 
#endif
  !
#if KEY_PARALLEL==1
  use parallel  
#endif
  !
  implicit none
  !
  LOGICAL QDONE
  real(chm_real)  EPOT(*), ESLF2(*)
  INTEGER NATM
  real(chm_real)  X(*), Y(*), Z(*), XI(*), YI(*), ZI(*), CG(*)
  !
  INTEGER I,J,IP,JP,QMEWALD
  real(chm_real)  EELPR
  real(chm_real)  CGT, DXI, DYI, DZI, S, R2, RS, R1S
  real(chm_real)  ERFCX,DRFC
  !
  ! go parallel
  INTEGER ATFRST,ATLAST
  INTEGER ATFRS2,ATLAS2,NODNQM
  !
  !
  QMEWALD = EWMODE
  !     IF(EWMODE.NE.1.OR.EWMODE.NE.2) QMEWALD = 1
  !
  !
  ! go parallel
#if KEY_PARALLEL==1 /*paramain*/
  NODNQM = NATQM / NUMNOD
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
  ATFRST=1+IPARPT(MYNOD)
  ATLAST=IPARPT(MYNODP)
  !
  ATFRS2=MYNOD*NODNQM+1
  ATLAS2=(MYNOD+1)*NODNQM
  IF(MYNOD.EQ.(NUMNOD-1)) ATLAS2=NATQM    
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
  ATFRST=1
  ATLAST=NATM
  !
  ATFRS2=1
  ATLAS2=NATQM
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
  IF (QMPI) THEN
     ATFRST=1
     ATLAST=NATM
     !
     ATFRS2=1
     ATLAS2=NATQM
  ENDIF
#else /* (paramain)*/
  ATFRST=1
  ATLAST=NATM
  !
  ATFRS2=1
  ATLAS2=NATQM
#endif /* (paramain)*/
  !
  If(.not.QDONE) then
     !
     !   Between QM and MM atoms :
     !           Do this only when it called in EAMPC2 subroutine (QDONE=.FALSE.)
     !           It is pre-calculation that doesn't need to be updated on SCF.
     ! go parallel
     DO J = ATFRST, ATLAST      ! J = 1, NATM
        !           skip QM or MM atoms out of cutoff
        IF ((QATLAB(J).LT.0).AND.(MMINB(J).GT.0)) THEN
           CGT = CG(J)
           DO I = 1, NATQM
              EELPR = ZERO
              IP  = IABS(QMINB(I))
              DXI = XI(J) - X(IP)
              DYI = YI(J) - Y(IP)
              DZI = ZI(J) - Z(IP)

#if KEY_PBOUND==1
              IF(qBoun) CALL PBCHECK(DXI,DYI,DZI)  
#endif

              S  = DXI*DXI+DYI*DYI+DZI*DZI
              R2 = ONE/S
              RS = SQRT(S)
              R1S= ONE/RS
              !
              CALL ERFCD(RS,KAPPA,ERFCX,DRFC,ERFMOD)
              !      MM atoms :
              IF(QMEWALD.EQ.1) THEN
                 !      Normal erf function (QM in cutoff use general QM/MM interaction of MJF
                 !      erf function and direct coulomb interaction is used for i,j pair
                 EELPR=-CGT*(ONE-ERFCX)*RS*R2

              ELSE IF(QMEWALD.EQ.2) THEN
                 !      MM (inlcluding ewald potential) only interact with diagonal elements
                 !      in the FOCK matrix
                 EELPR=CGT*ERFCX*RS*R2
              END IF
              EPOT(I) = EPOT(I) + EELPR
           END DO
        END IF
     END DO
  Else       ! (.not.QDONE)

     !
     !    Between QM atoms :
     !    Do this only when it called within MNDO SCF part (QDONE=.TRUE.)
     DO J = ATFRS2, ATLAS2         ! J = 1, NATQM
        JP  = IABS(QMINB(J))
        CGT = CHAG(J)
        DO I = 1, NATQM
           EELPR = ZERO
           IP = IABS(QMINB(I))
           IF(IP.NE.JP) THEN    
              DXI = X(JP) - X(IP)
              DYI = Y(JP) - Y(IP)
              DZI = Z(JP) - Z(IP)
              !    Between QM atoms, it doesn't need to check the PBOUND
              S  = DXI*DXI+DYI*DYI+DZI*DZI
              R2 = ONE/S
              RS=SQRT(S)
              R1S=ONE/RS
              !
              CALL ERFCD(RS,KAPPA,ERFCX,DRFC,ERFMOD)
              !
              !    QM atoms :
              !              only need to correct the erf function part
              EELPR=-CGT*(ONE-ERFCX)*RS*R2
              ESLF2(I) = ESLF2(I) + EELPR
           END IF
        END DO
     END DO
  End if       ! (.not.QDONE)
  !
  RETURN
END SUBROUTINE RSPACE_POT
!
!
SUBROUTINE RSPACE_GRAD(NAQM,NATM,X,Y,Z,XI,YI,ZI, &
     DX,DY,DZ,CG,DERQE &
     )
  !
  !    Compute the gradient at QM atom position for Real spave ewald summation
  !    Author:
  !
  !
  use ewald,only:kappa,erfmod
  use erfcd_mod,only: erfcd
  use chm_kinds
  use number
  use dimens_fcm
  use consta
  use quantm
#if KEY_PBOUND==1
  use pbound
#endif 
  !
  ! for parallel run
#if KEY_PARALLEL==1
  use parallel
#endif 
  !
  implicit none
  !
  INTEGER NAQM,NATM
  real(chm_real)  X(*), Y(*), Z(*), XI(*), YI(*), ZI(*)
  real(chm_real)  DX(NATM),DY(NATM),DZ(NATM), CG(*),DERQE(*)
  !
  INTEGER J,IP,JP,QMEWALD, II,NQMA
  real(chm_real)  CGT, DXI, DYI, DZI, S, R2, RS, R1S
  real(chm_real)  ERFCX,DRFC,DF,RFLAMB 
  real(chm_real):: temp1, temp2, temp3
  LOGICAL TODOQM
  !
  ! go parallel
  INTEGER ATFRST,ATLAST
  !
  !
  QMEWALD = EWMODE
  !
  ! go parallel
#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
  ATFRST=1+IPARPT(MYNOD)
  ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
  ATFRST=1
  ATLAST=NATM
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
  IF (QMPI) THEN
     ATFRST=1
     ATLAST=NATM
  ENDIF
#else /* (paramain)*/
  ATFRST=1
  ATLAST=NATM
#endif /* (paramain)*/
  !
  ! in case of ignoring QM images
  TODOQM = .TRUE.
  IF(NOQMIM) TODOQM=.FALSE.
  RFLAMB = ONE
  IF(QMPERT) RFLAMB = RLAMBF
  !
  nqma_loop: DO NQMA=1,NAQM
     II = 0
     IP = IABS(QMINB(NQMA))
     !
     j_loop: DO J = ATFRST, ATLAST            ! J = 1, NATM
        JP  = IABS(MMINB(J))
        ! check cutoff distance
        IF(MMINB(J).GT.0) THEN
           IF(QATLAB(J).GE.0) THEN
              ! QM-QM interaction
              II  = II + 1
              IF(TODOQM) THEN
                 CGT = RFLAMB*CHAG(II)

                 !        skip i=j case between QM atoms    
                 IF(IP.EQ.JP) cycle j_loop   !     GOTO 10
                 !
                 DXI = X(IP) - X(JP)
                 DYI = Y(IP) - Y(JP)
                 DZI = Z(IP) - Z(JP)
                 !
                 S  = DXI*DXI+DYI*DYI+DZI*DZI
                 R2 = ONE/S
                 RS=SQRT(S)
                 R1S=ONE/RS
                 !
                 CALL ERFCD(RS,KAPPA,ERFCX,DRFC,ERFMOD)
                 !
                 IF(QATLAB(J).GE.0) THEN
                    !     QM atoms :
                    !              only need to correct the erf function part
                    DF=CGT*(-DRFC*R1S + (ONE-ERFCX)*R2)*R1S
                    !
                    temp1  = half*DF*DERQE(NQMA)
                    DX(IP) = DX(IP) + temp1*DXI
                    DY(IP) = DY(IP) + temp1*DYI
                    DZ(IP) = DZ(IP) + temp1*DZI
                    !
                    DX(JP) = DX(JP) - temp1*DXI
                    DY(JP) = DY(JP) - temp1*DYI
                    DZ(JP) = DZ(JP) - temp1*DZI
                 END IF
              END IF
              ! 
           ELSE
              ! QM-MM interaction
              CGT = RFLAMB*CG(JP)
              !
              DXI = X(IP) - XI(JP)
              DYI = Y(IP) - YI(JP)
              DZI = Z(IP) - ZI(JP)
              !
#if KEY_PBOUND==1
              IF(qBoun) CALL PBCHECK(DXI,DYI,DZI)  
#endif

              S  = DXI*DXI+DYI*DYI+DZI*DZI
              R2 = ONE/S
              RS=SQRT(S)
              R1S=ONE/RS
              !
              CALL ERFCD(RS,KAPPA,ERFCX,DRFC,ERFMOD)
              !
              IF(QATLAB(J).LT.0) THEN
                 !     MM atoms:
                 IF(QMEWALD.EQ.1) THEN
                    DF=CGT*(-DRFC*R1S + (ONE-ERFCX)*R2)*R1S
                    temp1  = DF*DERQE(NQMA)
                    DX(IP) = DX(IP) + DXI*temp1
                    DY(IP) = DY(IP) + DYI*temp1
                    DZ(IP) = DZ(IP) + DZI*temp1
                    !
                    DX(JP) = DX(JP) - DXI*temp1
                    DY(JP) = DY(JP) - DYI*temp1
                    DZ(JP) = DZ(JP) - DZI*temp1
                 ELSE IF(QMEWALD.EQ.2) THEN
                    DF=-CGT*(DRFC*R1S + ERFCX*R2)*R1S
                    temp1  = DF*DERQE(NQMA)
                    DX(IP) = DX(IP) + DXI*temp1
                    DY(IP) = DY(IP) + DYI*temp1
                    DZ(IP) = DZ(IP) + DZI*temp1
                    !
                    DX(JP) = DX(JP) - DXI*temp1
                    DY(JP) = DY(JP) - DYI*temp1
                    DZ(JP) = DZ(JP) - DZI*temp1
                 END IF
              END IF
           END IF
        END IF
     END DO j_loop
  END DO nqma_loop
  !
  RETURN
END SUBROUTINE RSPACE_GRAD
!
SUBROUTINE KSPACE_POT(QDONE,EPOT,ESLF2,NATM,X,Y,Z,CG,MAXKK,KMX,KMY,KMZ, &
     KVEC,KTABXC,KTABXS,KTABYC,KTABYS,KTABZC,KTABZS    &
     )
  !
  !    Compute the potential at QM atom position for K-space ewald summation
  !    Author:
  !
  !
  use chm_kinds
  use number
  use dimens_fcm
  use image
  use consta
  use quantm
  !
  ! for parallel run
#if KEY_PARALLEL==1
  use parallel
#endif 
  !
  implicit none
  !
  LOGICAL QDONE
  real(chm_real)  EPOT(*),ESLF2(*)
  INTEGER NATM,MAXKK,KMX,KMY,KMZ
  real(chm_real)  KTABXC(NATM,MAXKK),KTABXS(NATM,MAXKK)
  real(chm_real)  KTABYC(NATM,MAXKK),KTABYS(NATM,MAXKK)
  real(chm_real)  KTABZC(NATM,MAXKK),KTABZS(NATM,MAXKK)
  real(chm_real)  X(*), Y(*), Z(*), KVEC(*), CG(*)
  !
  INTEGER I,J, KX, KY, KZ, KSQ, IPT, JP, II, KSY, KSZ
  real(chm_real)  ZFACT
  real(chm_real)  KSUM, KTGS(8)
  !     real(chm_real)  KTG1, KTG2, KTG3
  ! go parallel
  INTEGER ATFRST,ATLAST
  INTEGER ATFRS2,ATLAS2,NODNQM
  !
  !
  !
  IPT = 0
  !
  !
  ! go parallel
#if KEY_PARALLEL==1 /*paramain*/
  NODNQM = NATQM / NUMNOD
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
  ATFRST=1+IPARPT(MYNOD)
  ATLAST=IPARPT(MYNODP)
  !
  ATFRS2=MYNOD*NODNQM + 1
  ATLAS2=(MYNOD+1)*NODNQM
  IF(MYNOD.EQ.(NUMNOD-1)) ATLAS2=NATQM
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
  ATFRST=1
  ATLAST=NATM
  !
  ATFRS2=1
  ATLAS2=NATQM
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
  IF (QMPI) THEN
     ATFRST=1
     ATLAST=NATM
     !
     ATFRS2=1
     ATLAS2=NATQM
  ENDIF
#else /* (paramain)*/
  ATFRST=1
  ATLAST=NATM
  !
  ATFRS2=1
  ATLAS2=NATQM
#endif /* (paramain)*/
  !
  If(.not.QDONE) then
     !
     !
     ! MM atoms :
     DO KX = 0, KMX
        IF(KX.EQ.0) THEN
           KSY = 0
        ELSE
           KSY = -KMY
        ENDIF
        !
        DO KY = KSY, KMY
           IF(KX.EQ.0.AND.KY.EQ.0) THEN
              KSZ = 1
           ELSE
              KSZ = -KMZ
           ENDIF
           !
           DO KZ = KSZ, KMZ
              ZFACT = TWO
              KSQ = KX*KX + KY*KY + KZ*KZ
              IF(KSQ.LE.KSQMAXQ .AND. KSQ.NE.0) THEN
                 IPT      = IPT + 1
                 KTGS(1:8)= ZERO
                 DO J = ATFRST, ATLAST    ! J = 1, NATM
                    IF(QATLAB(J).LT.0) THEN
                       JP   = IABS(MMINB(J))
                       KTGS(1)=KTGS(1)+CG(JP)*KTABXC(JP,IPT)*KTABYC(JP,IPT)*KTABZC(JP,IPT)
                       KTGS(2)=KTGS(2)+CG(JP)*KTABXS(JP,IPT)*KTABYS(JP,IPT)*KTABZS(JP,IPT)
                       KTGS(3)=KTGS(3)+CG(JP)*KTABXC(JP,IPT)*KTABYC(JP,IPT)*KTABZS(JP,IPT)
                       KTGS(4)=KTGS(4)+CG(JP)*KTABXC(JP,IPT)*KTABYS(JP,IPT)*KTABZC(JP,IPT)
                       KTGS(5)=KTGS(5)+CG(JP)*KTABXC(JP,IPT)*KTABYS(JP,IPT)*KTABZS(JP,IPT)
                       KTGS(6)=KTGS(6)+CG(JP)*KTABXS(JP,IPT)*KTABYS(JP,IPT)*KTABZC(JP,IPT)
                       KTGS(7)=KTGS(7)+CG(JP)*KTABXS(JP,IPT)*KTABYC(JP,IPT)*KTABZS(JP,IPT)
                       KTGS(8)=KTGS(8)+CG(JP)*KTABXS(JP,IPT)*KTABYC(JP,IPT)*KTABZC(JP,IPT)
                    END IF
                 END DO
                 !
                 DO II = 1, NATQM
                    I    = IABS(QMINB(II))
                    KSUM = KTABXC(I,IPT)*KTABYC(I,IPT)*KTABZC(I,IPT)*KTGS(1) &
                         +KTABXS(I,IPT)*KTABYS(I,IPT)*KTABZS(I,IPT)*KTGS(2) &
                         +KTABXC(I,IPT)*KTABYC(I,IPT)*KTABZS(I,IPT)*KTGS(3) &
                         +KTABXC(I,IPT)*KTABYS(I,IPT)*KTABZC(I,IPT)*KTGS(4) &
                         +KTABXC(I,IPT)*KTABYS(I,IPT)*KTABZS(I,IPT)*KTGS(5) &
                         +KTABXS(I,IPT)*KTABYS(I,IPT)*KTABZC(I,IPT)*KTGS(6) &
                         +KTABXS(I,IPT)*KTABYC(I,IPT)*KTABZS(I,IPT)*KTGS(7) &
                         +KTABXS(I,IPT)*KTABYC(I,IPT)*KTABZC(I,IPT)*KTGS(8) 
                    EPOT(II) = EPOT(II) + KSUM*ZFACT*KVEC(IPT)
                 END DO
              END IF
           END DO
        END DO
     END DO
     !
  Else           ! (.not.QDONE)
     !
     !
     ! QM atoms :
     DO KX = 0, KMX
        IF(KX.EQ.0) THEN
           KSY = 0
        ELSE
           KSY = -KMY
        ENDIF
        !
        DO KY = KSY, KMY
           IF(KX.EQ.0.AND.KY.EQ.0) THEN
              KSZ = 1
           ELSE
              KSZ = -KMZ
           ENDIF
           !
           DO KZ = KSZ, KMZ
              ZFACT = TWO
              KSQ = KX*KX + KY*KY + KZ*KZ
              IF(KSQ.LE.KSQMAXQ .AND. KSQ.NE.0) THEN
                 IPT      = IPT + 1
                 KTGS(1:8)= zero
                 DO J = ATFRS2, ATLAS2     ! J = 1, NATQM
                    JP   = IABS(QMINB(J))
                    KTGS(1)=KTGS(1)+CHAG(J)*KTABXC(JP,IPT)*KTABYC(JP,IPT)*KTABZC(JP,IPT)
                    KTGS(2)=KTGS(2)+CHAG(J)*KTABXS(JP,IPT)*KTABYS(JP,IPT)*KTABZS(JP,IPT)
                    KTGS(3)=KTGS(3)+CHAG(J)*KTABXC(JP,IPT)*KTABYC(JP,IPT)*KTABZS(JP,IPT)
                    KTGS(4)=KTGS(4)+CHAG(J)*KTABXC(JP,IPT)*KTABYS(JP,IPT)*KTABZC(JP,IPT)
                    KTGS(5)=KTGS(5)+CHAG(J)*KTABXC(JP,IPT)*KTABYS(JP,IPT)*KTABZS(JP,IPT)
                    KTGS(6)=KTGS(6)+CHAG(J)*KTABXS(JP,IPT)*KTABYS(JP,IPT)*KTABZC(JP,IPT)
                    KTGS(7)=KTGS(7)+CHAG(J)*KTABXS(JP,IPT)*KTABYC(JP,IPT)*KTABZS(JP,IPT)
                    KTGS(8)=KTGS(8)+CHAG(J)*KTABXS(JP,IPT)*KTABYC(JP,IPT)*KTABZC(JP,IPT)
                 END DO
                 DO II = 1, NATQM
                    KSUM = ZERO
                    I    = IABS(QMINB(II))
                    KSUM = KTABXC(I,IPT)*KTABYC(I,IPT)*KTABZC(I,IPT)*KTGS(1)  &
                         +KTABXS(I,IPT)*KTABYS(I,IPT)*KTABZS(I,IPT)*KTGS(2)  &
                         +KTABXC(I,IPT)*KTABYC(I,IPT)*KTABZS(I,IPT)*KTGS(3)  &
                         +KTABXC(I,IPT)*KTABYS(I,IPT)*KTABZC(I,IPT)*KTGS(4)  &
                         +KTABXC(I,IPT)*KTABYS(I,IPT)*KTABZS(I,IPT)*KTGS(5)  &
                         +KTABXS(I,IPT)*KTABYS(I,IPT)*KTABZC(I,IPT)*KTGS(6)  &
                         +KTABXS(I,IPT)*KTABYC(I,IPT)*KTABZS(I,IPT)*KTGS(7)  &
                         +KTABXS(I,IPT)*KTABYC(I,IPT)*KTABZC(I,IPT)*KTGS(8)
                    ! 
                    ESLF2(II) = ESLF2(II) + KSUM*ZFACT*KVEC(IPT)
                 END DO
              END IF
           END DO
        END DO
     END DO
  End if      ! (.not.QDONE)
  !
  !
  !
  RETURN
END SUBROUTINE KSPACE_POT
!
SUBROUTINE KSPACE_GRAD(NAQM,NATM,DX,DY,DZ,CG,DERQE, &
     MAXKK,KMX,KMY,KMZ,KVEC, &
     KTABXC,KTABXS, &
     KTABYC,KTABYS, &
     KTABZC,KTABZS &
     )
  !
  !    Compute the potential at QM atom position for K-space ewald summation
  !    Author:
  !
  !
  use ewald,only:ewvirial,kappa
  use chm_kinds
  use number
  use dimens_fcm
  use image
  use consta
  use quantm
  !
  ! for parallel run
#if KEY_PARALLEL==1
  use parallel
#endif 
  implicit none
  !
  INTEGER NAQM,NATM,MAXKK,KMX,KMY,KMZ
  real(chm_real)  KTABXC(NATM,MAXKK),KTABXS(NATM,MAXKK)
  real(chm_real)  KTABYC(NATM,MAXKK),KTABYS(NATM,MAXKK)
  real(chm_real)  KTABZC(NATM,MAXKK),KTABZS(NATM,MAXKK)
  real(chm_real)  KVEC(*)
  real(chm_real)  DX(NATM),DY(NATM),DZ(NATM),CG(*),DERQE(*)
  !
  !
  INTEGER J, KX, KY, KZ, KSQ, IPT, NQM, JP, II, NQMA
  INTEGER KSY, KSZ
  real(chm_real)  CGI, CGJ
  real(chm_real)  ZFACT, KSUM(NATQM)
  real(chm_real)  CCFKX, CCFKY, CCFKZ
  real(chm_real)  KTG1, KTG2, KTG3, FDX, FDY, FDZ
  real(chm_real)  RKX(3), RKY(3), RKZ(3), XTLINV(6), MKV(3)
  real(chm_real)  KTGS(8), FDA(8), KXR, KYR, KZR
  real(chm_real)  EWPR, KLEN,EWEN, CEELECF,PIKAPA, RFLAMB
  real(chm_real):: temp0,temp1,temp2,temp3
  LOGICAL OK, TODOQM
  real(chm_real),parameter :: EVCALC=EV_TO_KCAL, &  ! EVCALC=23.061D0
       EVC   =AU_TO_EV  , &  ! EVC=27.21D0
       A0C   =BOHRR          ! A0C=0.529167D0
  !

  !
  ! go parallel
  INTEGER ATFRST,ATLAST
  !
  !     Calculate Reciprocal Space Lattice Vector
  CALL INVT33S(XTLINV, XTLABC, OK)
  !
  ! do virial setup
  CEELECF=EVCALC*EVC*A0C
  PIKAPA=TWO/(FOUR*KAPPA*KAPPA)
  ! 
  ! do initialize here..(check ewaldf.src and pme.src)
  EWVIRIAL(1:9) = ZERO
  ! end
  ! in case of ignoring QM images
  TODOQM = .TRUE.
  IF(NOQMIM) TODOQM=.FALSE.
  RFLAMB = ONE
  IF(QMPERT) RFLAMB = RLAMBF
  !
  ! go parallel
#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
  ATFRST=1+IPARPT(MYNOD)
  ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
  ATFRST=1
  ATLAST=NATM
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
  IF (QMPI) THEN
     ATFRST=1
     ATLAST=NATM
  ENDIF
#else /* (paramain)*/
  ATFRST=1
  ATLAST=NATM
#endif /* (paramain)*/
  !
  IPT = 0
  DO KX = 0, KMX
     IF(KX.EQ.0) THEN
        KSY = 0
     ELSE
        KSY = -KMY
     ENDIF
     temp1  = TWOPI*KX
     RKX(1) = temp1*XTLINV(1)
     RKX(2) = temp1*XTLINV(2)
     RKX(3) = temp1*XTLINV(4)
     !
     DO KY = KSY, KMY
        IF(KX.EQ.0.AND.KY.EQ.0) THEN
           KSZ = 1
        ELSE
           KSZ = -KMZ
        ENDIF
        temp2  = TWOPI*KY
        RKY(1) = temp2*XTLINV(2)
        RKY(2) = temp2*XTLINV(3)
        RKY(3) = temp2*XTLINV(5)
        !
        DO KZ = KSZ, KMZ
           ZFACT = TWO

           temp3  = TWOPI*KZ
           RKZ(1) = temp3*XTLINV(4)
           RKZ(2) = temp3*XTLINV(5)
           RKZ(3) = temp3*XTLINV(6)
           !
           KSQ = KX*KX + KY*KY + KZ*KZ
           IF(KSQ.LE.KSQMAXQ .AND. KSQ.NE.0) THEN
              IPT = IPT + 1
              II  = 0
              !
              MKV(1:3)  = RKX(1:3) + RKY(1:3) + RKZ(1:3)
              !
              temp0 =  KVEC(IPT)*ZFACT
              CCFKX =  MKV(1)*temp0
              CCFKY =  MKV(2)*temp0
              CCFKZ =  MKV(3)*temp0
              !
              ! do virial part
              KXR  = MKV(1)
              KYR  = MKV(2)
              KZR  = MKV(3)
              KLEN = KXR*KXR + KYR*KYR + KZR*KZR
              EWPR = TWO/KLEN + PIKAPA
              EWEN = ZERO

              KSUM(1:NAQM) = ZERO
              ! end
              !
              ! 1) QM-MM interaction.
              KTGS(1:8) = ZERO
              !
              !    sum over all MM atoms (except QM atoms)
              DO J = ATFRST, ATLAST       ! J = 1, NATM
                 JP   = IABS(MMINB(J))
                 CGJ = RFLAMB * CG(JP)
                 IF(QATLAB(JP).LT.0) THEN
                    KTGS(1)=KTGS(1)+CGJ*KTABXC(JP,IPT)*KTABYC(JP,IPT)*KTABZC(JP,IPT)
                    KTGS(2)=KTGS(2)+CGJ*KTABXS(JP,IPT)*KTABYC(JP,IPT)*KTABZC(JP,IPT)
                    KTGS(3)=KTGS(3)+CGJ*KTABXC(JP,IPT)*KTABYS(JP,IPT)*KTABZC(JP,IPT)
                    KTGS(4)=KTGS(4)+CGJ*KTABXS(JP,IPT)*KTABYS(JP,IPT)*KTABZC(JP,IPT)
                    KTGS(5)=KTGS(5)+CGJ*KTABXC(JP,IPT)*KTABYC(JP,IPT)*KTABZS(JP,IPT)
                    KTGS(6)=KTGS(6)+CGJ*KTABXS(JP,IPT)*KTABYC(JP,IPT)*KTABZS(JP,IPT)
                    KTGS(7)=KTGS(7)+CGJ*KTABXC(JP,IPT)*KTABYS(JP,IPT)*KTABZS(JP,IPT)
                    KTGS(8)=KTGS(8)+CGJ*KTABXS(JP,IPT)*KTABYS(JP,IPT)*KTABZS(JP,IPT)
                 END IF
              END DO
              !
              DO NQMA = 1, NAQM
                 NQM  = IABS(QMINB(NQMA))
                 CGI  = DERQE(NQMA)
                 FDX  = ZERO
                 FDY  = ZERO
                 FDZ  = ZERO
                 !     Temporary arrays
                 FDA(1)=KTABXS(NQM,IPT)*KTABYC(NQM,IPT)*KTABZC(NQM,IPT)
                 FDA(2)=KTABXC(NQM,IPT)*KTABYC(NQM,IPT)*KTABZC(NQM,IPT)
                 FDA(3)=KTABXS(NQM,IPT)*KTABYS(NQM,IPT)*KTABZC(NQM,IPT)
                 FDA(4)=KTABXC(NQM,IPT)*KTABYS(NQM,IPT)*KTABZC(NQM,IPT)
                 FDA(5)=KTABXS(NQM,IPT)*KTABYC(NQM,IPT)*KTABZS(NQM,IPT)
                 FDA(6)=KTABXC(NQM,IPT)*KTABYC(NQM,IPT)*KTABZS(NQM,IPT)
                 FDA(7)=KTABXS(NQM,IPT)*KTABYS(NQM,IPT)*KTABZS(NQM,IPT)
                 FDA(8)=KTABXC(NQM,IPT)*KTABYS(NQM,IPT)*KTABZS(NQM,IPT)
                 !
                 !     Force on x-axis
                 FDX  =-KTGS(1)*FDA(1)+KTGS(2)*FDA(2)-KTGS(3)*FDA(3)+KTGS(4)*FDA(4) &
                      -KTGS(5)*FDA(5)+KTGS(6)*FDA(6)-KTGS(7)*FDA(7)+KTGS(8)*FDA(8)
                 !     Force on y-axis
                 FDY  =-KTGS(1)*FDA(4)-KTGS(2)*FDA(3)+KTGS(3)*FDA(2)+KTGS(4)*FDA(1) &
                      -KTGS(5)*FDA(8)-KTGS(6)*FDA(7)+KTGS(7)*FDA(6)+KTGS(8)*FDA(5)
                 !     Force on z-axis
                 FDZ  =-KTGS(1)*FDA(6)-KTGS(2)*FDA(5)-KTGS(3)*FDA(8)-KTGS(4)*FDA(7) &
                      +KTGS(5)*FDA(2)+KTGS(6)*FDA(1)+KTGS(7)*FDA(4)+KTGS(8)*FDA(3)
                 !
                 DX(NQM)=DX(NQM)+CCFKX*FDX*CGI
                 DY(NQM)=DY(NQM)+CCFKY*FDY*CGI
                 DZ(NQM)=DZ(NQM)+CCFKZ*FDZ*CGI
                 ! do virial part
                 KSUM(NQMA)= FDA(1)*KTGS(8)+FDA(2)*KTGS(1)+FDA(3)*KTGS(6)+FDA(4)*KTGS(4) &
                      +FDA(5)*KTGS(7)+FDA(6)*KTGS(3)+FDA(7)*KTGS(2)+FDA(8)*KTGS(5)
              END DO
              ! original..and it's been modified above..to make it more efficient
              ! by expanding all the multiplication...
              ! namkh 10/09/04
              !                 DO J = ATFRST, ATLAST       ! J = 1, NATM
              !                    JP   = IABS(MMINB(J))
              !                    KTG1 = KTABXC(JP,IPT)*KTABXC(NQM,IPT) + KTABXS(JP,IPT)*KTABXS(NQM,IPT)
              !                    KTG2 = KTABYC(JP,IPT)*KTABYC(NQM,IPT) + KTABYS(JP,IPT)*KTABYS(NQM,IPT)
              !                    KTG3 = KTABZC(JP,IPT)*KTABZC(NQM,IPT) + KTABZS(JP,IPT)*KTABZS(NQM,IPT)
              !                    FDX  =-KTABXC(JP,IPT)*KTABXS(NQM,IPT) + KTABXS(JP,IPT)*KTABXC(NQM,IPT)
              !                    FDY  =-KTABYC(JP,IPT)*KTABYS(NQM,IPT) + KTABYS(JP,IPT)*KTABYC(NQM,IPT)
              !                    FDZ  =-KTABZC(JP,IPT)*KTABZS(NQM,IPT) + KTABZS(JP,IPT)*KTABZC(NQM,IPT)
              !
              !                    IF(QATLAB(J).GE.0) THEN
              !                       II    = II + 1
              !                       DX(JP)= DX(JP) + CHAG(II)*CCFKX*FDX*KTG2*KTG3
              !                       DY(JP)= DY(JP) + CHAG(II)*CCFKY*KTG1*FDY*KTG3
              !                       DZ(JP)= DZ(JP) + CHAG(II)*CCFKZ*KTG1*KTG2*FDZ
              !                    ELSE
              !                       DX(JP)= DX(JP) + CG(JP)*CCFKX*FDX*KTG2*KTG3
              !                       DY(JP)= DY(JP) + CG(JP)*CCFKY*KTG1*FDY*KTG3
              !                       DZ(JP)= DZ(JP) + CG(JP)*CCFKZ*KTG1*KTG2*FDZ
              !                    END IF
              !                 END DO
              !
              ! 2) MM-QM interaction.
              FDA(1:8) = ZERO
              DO NQMA = 1, NAQM
                 NQM  = IABS(QMINB(NQMA))
                 CGI  = DERQE(NQMA)
                 FDA(1)=FDA(1)+CGI*KTABXS(NQM,IPT)*KTABYC(NQM,IPT)*KTABZC(NQM,IPT)
                 FDA(2)=FDA(2)+CGI*KTABXC(NQM,IPT)*KTABYC(NQM,IPT)*KTABZC(NQM,IPT)
                 FDA(3)=FDA(3)+CGI*KTABXS(NQM,IPT)*KTABYS(NQM,IPT)*KTABZC(NQM,IPT)
                 FDA(4)=FDA(4)+CGI*KTABXC(NQM,IPT)*KTABYS(NQM,IPT)*KTABZC(NQM,IPT)
                 FDA(5)=FDA(5)+CGI*KTABXS(NQM,IPT)*KTABYC(NQM,IPT)*KTABZS(NQM,IPT)
                 FDA(6)=FDA(6)+CGI*KTABXC(NQM,IPT)*KTABYC(NQM,IPT)*KTABZS(NQM,IPT)
                 FDA(7)=FDA(7)+CGI*KTABXS(NQM,IPT)*KTABYS(NQM,IPT)*KTABZS(NQM,IPT)
                 FDA(8)=FDA(8)+CGI*KTABXC(NQM,IPT)*KTABYS(NQM,IPT)*KTABZS(NQM,IPT)
              END DO
              !
              DO J = ATFRST, ATLAST       ! J = 1, NATM
                 JP   = IABS(MMINB(J))
                 FDX  = ZERO
                 FDY  = ZERO
                 FDZ  = ZERO
                 !     Temporary arrays
                 KTGS(1)=KTABXC(JP,IPT)*KTABYC(JP,IPT)*KTABZC(JP,IPT)
                 KTGS(2)=KTABXS(JP,IPT)*KTABYC(JP,IPT)*KTABZC(JP,IPT)
                 KTGS(3)=KTABXC(JP,IPT)*KTABYS(JP,IPT)*KTABZC(JP,IPT)
                 KTGS(4)=KTABXS(JP,IPT)*KTABYS(JP,IPT)*KTABZC(JP,IPT)
                 KTGS(5)=KTABXC(JP,IPT)*KTABYC(JP,IPT)*KTABZS(JP,IPT)
                 KTGS(6)=KTABXS(JP,IPT)*KTABYC(JP,IPT)*KTABZS(JP,IPT)
                 KTGS(7)=KTABXC(JP,IPT)*KTABYS(JP,IPT)*KTABZS(JP,IPT)
                 KTGS(8)=KTABXS(JP,IPT)*KTABYS(JP,IPT)*KTABZS(JP,IPT)
                 !
                 !     Force on x-axis
                 FDX  =-KTGS(1)*FDA(1)+KTGS(2)*FDA(2)-KTGS(3)*FDA(3)+KTGS(4)*FDA(4) &
                      -KTGS(5)*FDA(5)+KTGS(6)*FDA(6)-KTGS(7)*FDA(7)+KTGS(8)*FDA(8)
                 !     Force on y-axis
                 FDY  =-KTGS(1)*FDA(4)-KTGS(2)*FDA(3)+KTGS(3)*FDA(2)+KTGS(4)*FDA(1) &
                      -KTGS(5)*FDA(8)-KTGS(6)*FDA(7)+KTGS(7)*FDA(6)+KTGS(8)*FDA(5)
                 !     Force on z-axis
                 FDZ  =-KTGS(1)*FDA(6)-KTGS(2)*FDA(5)-KTGS(3)*FDA(8)-KTGS(4)*FDA(7) &
                      +KTGS(5)*FDA(2)+KTGS(6)*FDA(1)+KTGS(7)*FDA(4)+KTGS(8)*FDA(3)
                 !
                 !      DERQE has been multiplied already
                 CGJ   =RFLAMB * CG(JP)
                 DX(JP)=DX(JP)-CCFKX*FDX*CGJ
                 DY(JP)=DY(JP)-CCFKY*FDY*CGJ
                 DZ(JP)=DZ(JP)-CCFKZ*FDZ*CGJ
              END DO
              !
              ! 3) QM-QM interaction.
              DO NQMA = 1, NAQM
                 NQM  = IABS(QMINB(NQMA))
                 CGI  = DERQE(NQMA)

                 IF(TODOQM) THEN
                    DO J = 1, NAQM
                       JP= IABS(QMINB(J))
                       IF(JP.GE.ATFRST.AND.JP.LE.ATLAST) THEN
                          KTG1 = KTABXC(JP,IPT)*KTABXC(NQM,IPT) + KTABXS(JP,IPT)*KTABXS(NQM,IPT)
                          KTG2 = KTABYC(JP,IPT)*KTABYC(NQM,IPT) + KTABYS(JP,IPT)*KTABYS(NQM,IPT)
                          KTG3 = KTABZC(JP,IPT)*KTABZC(NQM,IPT) + KTABZS(JP,IPT)*KTABZS(NQM,IPT)
                          FDX  =-KTABXC(JP,IPT)*KTABXS(NQM,IPT) + KTABXS(JP,IPT)*KTABXC(NQM,IPT)
                          FDY  =-KTABYC(JP,IPT)*KTABYS(NQM,IPT) + KTABYS(JP,IPT)*KTABYC(NQM,IPT)
                          FDZ  =-KTABZC(JP,IPT)*KTABZS(NQM,IPT) + KTABZS(JP,IPT)*KTABZC(NQM,IPT)
                          !      Force on x,y,z-axis (temporary use of FDA array)
                          CGJ   =RFLAMB * CHAG(J)
                          temp0 =CGI*CGJ 
                          FDA(1)=temp0*CCFKX*FDX*KTG2*KTG3
                          FDA(2)=temp0*CCFKY*KTG1*FDY*KTG3
                          FDA(3)=temp0*CCFKZ*KTG1*KTG2*FDZ
                          IF(NQM.EQ.JP) THEN
                             DX(NQM)= DX(NQM) + FDA(1)
                             DY(NQM)= DY(NQM) + FDA(2)
                             DZ(NQM)= DZ(NQM) + FDA(3)
                          ELSE
                             DX(NQM)= DX(NQM) + HALF*FDA(1)
                             DY(NQM)= DY(NQM) + HALF*FDA(2)
                             DZ(NQM)= DZ(NQM) + HALF*FDA(3)
                             !
                             DX(JP) = DX(JP)  - HALF*FDA(1)
                             DY(JP) = DY(JP)  - HALF*FDA(2)
                             DZ(JP) = DZ(JP)  - HALF*FDA(3)
                          END IF
                          ! do virial part
                          KSUM(NQMA)=KSUM(NQMA)+HALF*CGJ*KTG1*KTG2*KTG3
                       END IF
                    END DO
                 END IF
                 ! do virial part
                 KSUM(NQMA)=RFLAMB * CHAG(NQMA) * KSUM(NQMA)
                 EWEN = EWEN + KSUM(NQMA)                     
              END DO

              ! do virial part
              ! 1:xx, 2:xy, 3:xz | 4:yx, 5:yy, 6:zx | 7:zx, 8:zy, 9:zz
              ! 
              EWEN       =KVEC(IPT)*EWEN*ZFACT*CEELECF
              EWVIRIAL(1)=EWVIRIAL(1)+EWEN*(ONE-EWPR*KXR*KXR)
              EWVIRIAL(2)=EWVIRIAL(2)-EWEN*EWPR*KXR*KYR
              EWVIRIAL(3)=EWVIRIAL(3)-EWEN*EWPR*KXR*KZR
              EWVIRIAL(4)=EWVIRIAL(4)-EWEN*EWPR*KXR*KYR
              EWVIRIAL(5)=EWVIRIAL(5)+EWEN*(ONE-EWPR*KYR*KYR)
              EWVIRIAL(6)=EWVIRIAL(6)-EWEN*EWPR*KYR*KZR
              EWVIRIAL(7)=EWVIRIAL(7)-EWEN*EWPR*KXR*KZR
              EWVIRIAL(8)=EWVIRIAL(8)-EWEN*EWPR*KYR*KZR
              EWVIRIAL(9)=EWVIRIAL(9)+EWEN*(ONE-EWPR*KZR*KZR)
           END IF
        END DO
     END DO
  END DO
  !
#if KEY_PARALLEL==1
  IF(NUMNOD.GT.1 .AND. .NOT. QMPI) THEN
     call psync()
     call GCOMB(EWVIRIAL,9)
  END IF
  IF(NUMNOD.GT.1 .AND. MYNOD.NE.0 .AND. .NOT. QMPI) EWVIRIAL(1:9) = ZERO
#endif 
  !
  RETURN
END SUBROUTINE KSPACE_GRAD
!
SUBROUTINE SUM_GRAD(NAQM,NATM,DX,DEX,DEY,DEZ,GRAD)
  !
  !     Add computed gradient to the QM grad and total gradient
  !     Author:
  !
  !
  use chm_kinds
  use number
  use dimens_fcm
  !
  use quantm
  !
  ! for parallel run
#if KEY_PARALLEL==1
  use parallel
#endif 
  !
  implicit none
  INTEGER NAQM,NATM
  real(chm_real)  DX(*),DEX(*),DEY(*),DEZ(*)
  real(chm_real)  GRAD(3,NAQM) 
  !
  !
  INTEGER I,J,IQ, ATFRST, ATLAST,NATM2,ncount
  !
  NATM2 = 2*NATM
  !
  ! go parallel
#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
  ATFRST=1+IPARPT(MYNOD)
  ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
  ATFRST=1
  ATLAST=NATM
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
  IF (QMPI) THEN
     ATFRST=1
     ATLAST=NATM
  ENDIF
#else /* (paramain)*/
  ATFRST=1
  ATLAST=NATM
#endif /* (paramain)*/
  !
  ! For QM atoms
  DO I = 1, NAQM
     IQ= IABS(QMINB(I))
     GRAD(1,I) = GRAD(1,I) + DEX(IQ)
     GRAD(2,I) = GRAD(2,I) + DEY(IQ)
     GRAD(3,I) = GRAD(3,I) + DEZ(IQ)
  END DO
  ! For MM atoms
  ncount      = 3*(ATFRST-1)+1
  DO J = ATFRST, ATLAST      ! J = 1, NATM
     IF(QATLAB(J).LT.0) THEN
        dx(ncount)  = dx(ncount)  +DEX(J)
        dx(ncount+1)= dx(ncount+1)+DEY(J)
        dx(ncount+2)= dx(ncount+2)+DEZ(J)
     END IF
     ncount      = ncount+3
  END DO

  RETURN
END SUBROUTINE SUM_GRAD
!
!
SUBROUTINE GETGRDQ(NATOM,DX,DY,DZ,QLOCAL)
  !
  !     Add kspace gradient contribution of QM/MM into gradient array.
  !     Author:
  !
  !
  use chm_kinds
  use number
  use dimens_fcm
  use quantm,only : PDXYZ
  ! for parallel run
#if KEY_PARALLEL==1
  use parallel
#endif 
  implicit none
  !
  INTEGER NATOM
  real(chm_real)  DX(*),DY(*),DZ(*)
  logical :: QLOCAL
  !
  !
  INTEGER I
  ! go parallel
  INTEGER ATFRST, ATLAST
  !
  ! go parallel
#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
  ATFRST=1+IPARPT(MYNOD)
  ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
  ATFRST=1
  ATLAST=NATOM
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
  IF (QMPI) THEN
     ATFRST=1
     ATLAST=NATOM
  ENDIF
#else /* (paramain)*/
  ATFRST=1
  ATLAST=NATOM
#endif /* (paramain)*/
  !
  DO I=ATFRST, ATLAST
     DX(I)=DX(I)+PDXYZ(1,I)
     DY(I)=DY(I)+PDXYZ(2,I)
     DZ(I)=DZ(I)+PDXYZ(3,I)
  END DO
  !
  return
end SUBROUTINE GETGRDQ

#endif 
SUBROUTINE NULL_QE
  RETURN
END SUBROUTINE NULL_QE

