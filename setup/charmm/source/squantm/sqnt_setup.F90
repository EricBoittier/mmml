#if KEY_SQUANTM==1 /*mainsquatn*/
SUBROUTINE SQMINI(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     This is the interface for running new modified MOPAC code
  !     from Scripps AMBER group with CHARMM for
  !     QM/MM calculations
  !
  !     It closely follows the interface developed for GAMESS by
  !     Milan Hodoscek.
  !
  !     Kwangho Nam, Ross Walker, and Mike Crowley
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use number
  use memory
  use bases_fcm
  use datstr
  use code
  use coord
  use energym
  use ewald_1m, only : lewald, kappa, erfmod
  !use erfcd_mod,only : erfmod
  use pme_module, only : QPME
  use gamess_fcm
  use inbnd
  ! New quantum code part
  use squantm
  use quantm, only : natom_check,xim,yim,zim
  use nbndqm_mod
  !
  use param
  use psf
  use select
  use stream
  use string
  !
  ! Adjust nonbonded group list for IMAGES.
  use image
  !     Adjust nonbonded group list for simple pbc.
#if KEY_PBOUND==1
  use pbound  
#endif
  !
  ! Check for FLUCQ
#if KEY_FLUCQ==1
  use flucq  
#endif
  !
  implicit none

  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  !
  !new
  ! allocatable arrays
  integer,allocatable,dimension(:) :: ISLCT
  integer,allocatable,dimension(:) :: JSLCT
  integer,allocatable,dimension(:) :: LSLCT
  integer,allocatable,dimension(:) :: ISLCT2  ! for dual quantum region
  integer,allocatable,dimension(:) :: JLSLCT  ! for dual quantum region

  ! regualr variables
  integer :: I, J, ii, MNUM, PRNLEVQM2, &
       natqm_1,natqm_2,nqmcharge_1,nqmcharge_2, &
       nspin_1,nspin_2,Indx_mminb
  logical :: QDONE, QIMAGE, ISNBDS, QPRINT, &
       qlink_1,qlink_2,Qdual_check(2), &
       qmswitch_local,qmewd_local,qmlay_local, &
       qget_put,qfirst,qget_h,qsetup_h, &
       QPKNOE,QPKNOANG,QPKNODIH,QPKNOIMP, &
       NOPMEwald,qmnopmewd_local
  ! for new mopac qm/mm setup 
  LOGICAL:: QFRST, QCHECK, QMMM_NoGauss, QMMM_NoDiis, LFIRST
  INTEGER:: LPERIOD
  real(chm_real)::  ESCF, CGGHO, CGGHO_2, tot_cg_local(2)
  !
  ! for local atomic numbers
  INTEGER:: NZUNC_1(NQMAX),NZUNC_2(NQMAX)

  !
  !
#if KEY_FLUCQ==1
  IF(QFLUC) CALL WRNDIE(-2,'<SQMINI>', &
       'FLUCQ is not implmented with SQUANTM.')
#endif 
  !
  !
  ! Initial check up
  QIMAGE =.FALSE.
  IF(.NOT.USEDDT_nbond(BNBND)) CALL WRNDIE(-3,'<SQMINI>', &
       'Nonbond data structure is not defined.')
#if KEY_PBOUND==1
  IF(.NOT.qBoun) THEN   
#endif
     IF(NTRANS.GT.0) THEN
        IF(LGROUP) THEN
           QIMAGE =.TRUE.
           IF(.NOT.USEDDT_image(BIMAG)) CALL WRNDIE(-3,'<SQMINI>', &
                'Image nonbond data structure is not defined.')
        ELSE
           CALL WRNDIE(-2,'<SQMINI>', &
                'QM/MM do not interact with Images under Atom Based Cutoff.')
        END IF
     END IF
#if KEY_PBOUND==1
  END IF                
#endif
  !
  ! DTM - debug
  qmused = .true.      ! safe here since next are the allocations, ...
  call allocate_gamess ! try to reduce from MAXA to NATOM
  call allocate_squantm
  !
  !new
  call chmalloc('sqnt_setup.src','SQMINI','ISLCT',NATOM,intg=ISLCT)
  call SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
  !new

  ! Connection atom (Currenlty only C-connection atom)
  CLINK  = (INDXA(COMLYN,COMLEN,'CLNK') .GT. 0)
  NCATM  = 0
  IF(CLINK) THEN
     IF(PRNLEV.GE.2) WRITE(OUTU,22) 'CLNK: Connection atoms are used'
     !new
     call chmalloc('sqnt_setup.src','SQMINI','JSLCT',NATOM,intg=JSLCT)
     call SELCTA(COMLYN,COMLEN,JSLCT,X,Y,Z,WMAIN,.TRUE.)
     !new


     CALL WRNDIE(-2,'<SQMINI>','CLNK are not supported yet.')

  END IF

  !pppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp
  !
  QLINK(1)=(INDXA(COMLYN, COMLEN, 'GLNK') .GT. 0)
  QLINK(2)=.FALSE.
  CGGHO   = ZERO
  CGGHO_2 = ZERO
  QMFEP   =.false.                      ! for dual quantum region
  QMLAY   =.false.                      ! for dual quantum region
  QMSOLV  =.false.                      ! for dual quantum region
  IF(QLINK(1)) THEN
     IF(CLINK) CALL WRNDIE(-5,'<SQMINI>','GLNK and CLNK are not compatable.')
     IF(PRNLEV.GE.2) WRITE(OUTU,22) 'GLNK: GHO atoms are used'
     !new
     call chmalloc('sqnt_setup.src','SQMINI','LSLCT',NATOM,intg=LSLCT)
     call SELCTA(COMLYN,COMLEN,LSLCT,X,Y,Z,WMAIN,.TRUE.)
     !new

     QCHECK=.TRUE.
     Qdual_check(1)= (QMFEP .or. QMLAY)  !
     Qdual_check(2)=.false.              ! copy to 1st qm region
     qfirst        =.true.
     CALL GHOHYB(NATOM,ISLCT,LSLCT,NBOND,IB,JB,CGGHO,X,Y,Z,CG,QCHECK,Qdual_check,qfirst)
     IF(.NOT.QCHECK) CALL WRNDIE(-5,'<SQMINI>','The CHARMM will stop at GHOHYB.')
  END IF
  !pppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp
  !
  !
  !***********************************************************************
  ! Determine QM method/QM charge/Spin State/SCF convergence
  !
  ! Here's default values
  !     QMTHEORY: AM1 (2)/PM3 (1)/MNDO (3)/PDDG-PM3 (4)/PDDG-MNDO(5)
  !               PDP3 = PDDG-PM3
  !               PDMN = PDDG-MNDO
  nqmtheory=2
  nqmcharge(1:2)=0
  scfconv=TENM8
  nspin(1:2)=1
  ! QM method
  IF(INDEX(COMLYN,'AM1')  .NE. 0) nqmtheory=2
  IF(INDEX(COMLYN,'PM3')  .NE. 0) nqmtheory=1
  IF(INDEX(COMLYN,'MNDO') .NE. 0) nqmtheory=3
  IF(INDEX(COMLYN,'PDP3') .NE. 0) nqmtheory=4
  IF(INDEX(COMLYN,'PDMN') .NE. 0) nqmtheory=5
  ! QM charge
  nqmcharge(1)=GTRMI(COMLYN,COMLEN,'CHAR',0)
  qmcharge(1)=REAL(nqmcharge(1))
  qmcharge(2)=zero              ! for dual quantum region
  ! QM scf convergence
  scfconv=GTRMF(COMLYN,COMLEN,'SCFC',TENM8)
  ! QM spin
  IF((INDEX(KEYWRD,'TRIPLET').NE.0) .OR.  &
       (INDEX(KEYWRD,'TRIP').NE.0)) nspin(1)=3
  !     Doublet spin state not work with Restricted QM methods
  IF((INDEX(KEYWRD,'DOUBLET').NE.0) .OR. &
       (INDEX(KEYWRD,'DOUB').NE.0)) nspin(1)=2
  IF(nspin(1).EQ.2) CALL WRNDIE(-5,'<SQMINI>','Double is not supported yet.')
  !***********************************************************************
  !
  ! Other setup
  QGMREM=(INDXA(COMLYN,COMLEN,'REMO').GT.0)
  QGMEXG=(INDXA(COMLYN,COMLEN,'EXGR').GT.0)
  If(Prnlev.ge.2) then
     if(QGMREM) then
        WRITE(OUTU,22) 'REMOve: Classical energies within QM atoms are removed.'
     else
        WRITE(OUTU,22) 'No REMOve: Classical energies within QM atoms are retained.'
     end if
     if (QGMEXG) then
        WRITE(OUTU,22) 'EXGRoup: QM/MM Electrostatics for link host groups removed.'
     else
        WRITE(OUTU,22) 'No EXGRoup: QM/MM Elec. for link atom host only is removed.'
     end if
  End if
22 FORMAT('SQMINI> ',A)

  ! for include/excluse QM-MM Gaussian-Gaussian core interactions
  ! in AM1/PM3, but PDDG-PM3 should be handled differently.
  QMMM_NoGauss =.true.
  if(nqmtheory.eq.1 .or. nqmtheory.eq.2 ) then   ! .or. nqmtheory.eq.4) then
     QMMM_NoGauss =.false.
     QMMM_NoGauss = (INDXA(COMLYN,COMLEN,'NOGA').GT.0)
  end if
  If(Prnlev.ge.2.and.QMMM_NoGauss) WRITE(OUTU,22) &
       'No GAussian: Gaussian core-core QM-MM interaction is removed.'

  ! for DIIS converger.
  QMMM_NoDiis =(INDXA(COMLYN,COMLEN,'NDIS').GT.0)
  If(Prnlev.ge.2.and.QMMM_NoDiis) WRITE(OUTU,22) &
       'No Diis: DIIS converger will be turned off.'

! for Time-reversible Born-Oppenheimer MD simulations (TR-BOMD):
      LTRBOMD=(INDXA(COMLYN,COMLEN,'TRBO').GT.0)    ! TRBOmd
      N_scf_cycle=GTRMI(COMLYN,COMLEN,'NSCF',100)

!     initialize.
      q_apply_tr_bomd=.false.
      i_md_step      =0
      i_md_step_old  =0

      if(Prnlev.ge.2 .and. LTRBOMD) WRITE(OUTU,23) &
            'Time-reversible BOMD will be used and number of scf cyle will be ', &
             N_scf_cycle
  23  FORMAT('SQMINI> ',A,I4)

  ! for dual quantum region
  ! QUANTum ...
  !     DUAL PERT [pert-specific] [atom-sele] GLN2 [atom-sele] 
  !          MLAY [mlay-specific] [atom-sele] LINK [atom-sele]
  !          SOLV [solv-specific]  <=== Not done at all!!!
  !
  ! pert-specific / solv-specific:
  !     ... REF0 [real] PEF1 [real] PEF2 [real] TEMPerature [real] 
  !
  ! mlay-specific:
  !     
  !
  QDUALQM=(INDXA(COMLYN,COMLEN,'DUAL').GT.0)
  QMFEP  =.false.
  QMLAY  =.false.
  QMSOLV =.false.
  QH_Link=.false.
  QpKa_on=.false.
  PreFact_dual(1)=one
  PreFact_dual(2)=zero
  do i=1,MAXCHL
     R_H_ref(i)=-1000.0d0
  end do
  If(QDUALQM) then
     If(Prnlev.ge.2) WRITE(OUTU,22) 'Dual QM region: Option for dual qm region is Activated.'
     QMFEP= (INDXA(COMLYN,COMLEN,'PERT').GT.0)
     QMLAY= (INDXA(COMLYN,COMLEN,'MLAY').GT.0)
     QMSOLV=(INDXA(COMLYN,COMLEN,'SOLV').GT.0)
     QCHECK=((QMFEP .and. QMLAY) .or. (QMFEP .and. QMSOLV) .or. (QMLAY .and. QMSOLV))
     if(QCHECK) call wrndie(-5,'<QMINI>', &
          'QM/MM-FEP(PERT)/MLAYer-QM/MM/SOLV are exclusive each other.')

     !        assume two atom selections.
     !        1-st one: qm selection.
     !        2-nd one: 
     !             for QMFEP: it can be gho atoms.
     !                        but, assume GHO atoms are same for 1st qm.
     !             for QMLAY: it can "ONLY" be qm atoms to put h-link atoms.
     if(QMFEP .or. QMLAY) then
        !new
        call chmalloc('sqnt_setup.src','SQMINI','ISLCT2',NATOM,intg=ISLCT2)
        call SELCTA(COMLYN,COMLEN,ISLCT2,X,Y,Z,WMAIN,.TRUE.)
        !new

        nqmcharge(2)=GTRMI(COMLYN,COMLEN,'KHAR',0)
        qmcharge(2) =REAL(nqmcharge(2))

        If(QMLAY) then
           PreFact_dual(1)=one
           PreFact_dual(2)=MINONE
           if(Prnlev.ge.2) WRITE(OUTU,22) &
                'MLAYer-QM/MM: multi-layered QM/MM is activated.'
           QH_Link =(INDXA(COMLYN,COMLEN,'LINK').GT.0)
           if(QH_Link) then
              !new
              call chmalloc('sqnt_setup.src','SQMINI','JLSLCT',NATOM,intg=JLSLCT)
              call SELCTA(COMLYN,COMLEN,JLSLCT,X,Y,Z,WMAIN,.TRUE.)
              !new
           end if

           !        for high level qm methods.
#if KEY_GAMESS==1 || KEY_GAMESSUK==1
           QMLAY_high =.true.                     
#endif
        End if
        If(QMFEP) then
           QLINK(2)=(INDXA(COMLYN,COMLEN,'GLN2').GT.0)
           if(QLINK(2)) then
              IF(PRNLEV.GE.2) WRITE(OUTU,22) 'GLNK: GHO atoms are used for (DUAL) 2nd-qm region'
              !new
              call chmalloc('sqnt_setup.src','SQMINI','JLSLCT',NATOM,intg=JLSLCT)
              call SELCTA(COMLYN,COMLEN,JLSLCT,X,Y,Z,WMAIN,.TRUE.)
              !new

              QCHECK=.TRUE.
              Qdual_check(1)=.true.
              Qdual_check(2)=.true.         ! copy to 2nd qm region
              qfirst        =.true.
              CALL GHOHYB(NATOM,ISLCT2,JLSLCT, NBOND,IB,JB, &
                   CGGHO_2,X,Y,Z,CG,QCHECK,Qdual_check,qfirst)
              IF(.NOT.QCHECK) CALL WRNDIE(-5,'<SQMINI>', &
                   'The CHARMM will stop at GHOHYB for 2nd QM.')
           end if
        End if
     end if
     QCHECK=((CLINK.and.QH_Link) .or. (CLINK.and.QLINK(2)) .or. (QH_Link.and.QLINK(2)))
     if(QCHECK) call wrndie(-5,'<QMINI>', &
          'In Dual QM calc., CLNK/LINK/GLN2 are exclusive each other.')

     If(QMFEP .or. QMSOLV) then
        lambda_qm(1)=gtrmf(COMLYN,COMLEN,'REF0',ONE)
        lambda_qm(2)=gtrmf(COMLYN,COMLEN,'PEF1',ONE)
        lambda_qm(3)=gtrmf(COMLYN,COMLEN,'PEF2',ONE)
        if( (lambda_qm(1).eq.lambda_qm(2)) .or. (lambda_qm(1).eq.lambda_qm(3)) )  &
             call wrndie(-1,'<SQMINI>','Perturbation ill defined.')
        if(Prnlev.ge.2) then
           if(QMFEP) WRITE(OUTU,22) &
                'QM/MM-FEP: PERT option is activate for lambda 0->1 direction.'
           if(QMSOLV) WRITE(OUTU,22) &
                'QM/MM-SOLV: SOLV option is activate for lambda 0->1 direction.'
           WRITE(OUTU,'(A,A,1X,3F6.3,A)') &
                'SQMINI> ','QM/MM electrostatic FEP will be performed for', &
                lambda_qm(1),lambda_qm(2),lambda_qm(3),'.'
        end if
        temperature_fep= gtrmf(COMLYN,COMLEN,'TEMP',ZERO)
        beta_qmmm_fep  = one/(KBOLTZ*temperature_fep)
        PreFact_dual(1)= one-lambda_qm(1)            ! 1-lambda for react
        PreFact_dual(2)= lambda_qm(1)                ! lambda for prodct
     End if

     !     for pKa FEP-QM/MM. (pKa calculations.)
     If(QMFEP) then
        QpKa_on=(INDXA(COMLYN,COMLEN,'PKAP').GT.0)
        if(QpKa_on) then
           If(Prnlev.ge.2) WRITE(OUTU,22) 'pKa FEP calculations will be performed.'
           !        setup for internal terms relavant for H atom to removed.
           !        (Bond should be turned on always!)
           QPKNOANG=(INDXA(COMLYN,COMLEN,'NOAN').GT.0)
           QPKNODIH=(INDXA(COMLYN,COMLEN,'NODI').GT.0)
           QPKNOIMP=(INDXA(COMLYN,COMLEN,'NOIM').GT.0)
           QPKNOE  =(INDXA(COMLYN,COMLEN,'NOMM').GT.0)
           If (QPKNOE) then
              QPKNOANG=.TRUE.
              QPKNODIH=.TRUE.
              QPKNOIMP=.TRUE.
           End if
           If(PRNLEV.GE.2) then
              If (QPKNOANG) WRITE(OUTU,22) &
                   'MM angle    term ignored for H-atom in pKa FEP.'
              If (QPKNODIH) WRITE(OUTU,22) &
                   'MM dihedral term ignored for H-atom in pKa FEP.'
              If (QPKNOIMP) WRITE(OUTU,22) &
                   'MM improper term ignored for H-atom in pKa FEP.'
           End if
        end if
     End if
  End if

  ! namkh 01/21/04
  ! Activation of LQMEWD when LEWALD and QGMREM..only this case
  ! 
  ! Possible option.
  ! EWMODE     : 0 No Ewald QM/MM-SCF, thus Ewald sum is incldued as
  !                post-scf procedure as handled in regular MM Ewald routine.
  !              1 Ewald QM/MM-SCF.
  !                The MM within cutoff do interact with QM atoms as regular 
  !                QM/MM interaction, and apply Ewald correction potential
  !                into diagonal in FOCK matrix.
  !              2 Ewald QM/MM-SCF.
  !                The total Ewald potential only applied into diagonal in
  !                FOCK matrix.
  ! NQMEWD     : 0 Use Mulliken charges on QM atoms to represent charges on
  !                image atoms.
  !              1 Use input MM charges on QM atoms to represent charges on
  !                image atoms.
  IF(LEWALD.AND.(.NOT.QGMREM)) CALL WRNDIE(-1,'<SQMINI>', &
       'QM/MM-Ewald is not compatable without REMO.')
  !
  ! Initialization
  LQMEWD=.FALSE.
  CGMM  = ZERO
  IF(LEWALD.AND.QGMREM) LQMEWD=.TRUE.
  !
  IF(LQMEWD) THEN
     If(Prnlev.ge.2) WRITE(outu,22) 'Ewald with QM/MM Option has been Activated.'
     EWMODE = 1
     EWMODE = GTRMI(COMLYN,COMLEN,'NEWD',1)
     NOPMEwald=(INDXA(COMLYN,COMLEN,'NOPM').GT.0)  ! don't use PMEwald option.

     ! sanity check: only use this when QPME==.true.
     if(.not.QPME) NOPMEwald=.true.

     !
     ! current option check
     IF(EWMODE.NE.1) THEN
        If(Prnlev.ge.2) WRITE(outu,22) &
             'Currenlty, default QM/MM-Ewald option is only available.'
        EWMODE = 1
     END IF
     !
     If(Prnlev.ge.2) then
        WRITE(outu,22) 'Default Ewald with QM/MM Option uses Mulliken Charges'
        WRITE(outu,22) 'MM within cutoff interact with regular way with QM'
        WRITE(outu,22) 'MM from Ewald Sum interact with diagonal elements in QM'
        if(NOPMEwald) then
           WRITE(outu,22) 'Regular Ewald summation will be carried out for QM/MM-Ewald.'
        else
           WRITE(outu,22) 'PMEwald option will be used for QM/MM-Ewald (QM/MM-PMEwald).'
        end if
     End if
     !
     ! Now setup for Ewald in QM/MM SCF
     ! default values
     KMAXQ  = 5
     KSQMAXQ= 27
     !
     KMAXQ  =GTRMI(COMLYN,COMLEN,'KMAX',KMAXQ)
     !                   added by Scott Feller 5/24/95, NIH
     KMAXXQ =GTRMI(COMLYN,COMLEN,'KMXX',KMAXQ)
     KMAXYQ =GTRMI(COMLYN,COMLEN,'KMXY',KMAXQ)
     KMAXZQ =GTRMI(COMLYN,COMLEN,'KMXZ',KMAXQ)
     KSQMAXQ=GTRMI(COMLYN,COMLEN,'KSQM',KSQMAXQ)

     ! Check the total charge for QM/MM-Ewald 
     DO I=1,NATOM
        IF(IGMSEL(I).EQ.5.OR.IGMSEL(I).EQ.0) CGMM=CGMM+CG(I)
     END DO
     IF(QLINK(1)) CGMM = CGMM + CGGHO

  ELSE
     ! in the case of no Ewald with QM/MM
     EWMODE  = 0
     NOPMEwald=.true.
  END IF
  !
  ! cutoff switching
  QMSWTCH=(INDXA(COMLYN,COMLEN,'SWIT').GT.0)
  IF(QMSWTCH .AND. LQMEWD) THEN
     If(Prnlev.ge.2) WRITE(OUTU,22) &
          'SWITch: QM/MM electrostatic Switching ignored in QM/MM-Ewald.'
     QMSWTCH=.FALSE.
  END IF
  IF(QMSWTCH .and. Prnlev.ge.2) WRITE(OUTU,22) &
       'SWITch: QM/MM electrostatic Switching function is used.'


  ! for dual quantum region: multi-layered QM/MM
  !     do left over, but limited, setup for GAMESS-UK/GAMESS-US.
  qmlay_local=QMLAY
  If(QMLAY) call Setup_option_Mlayer(COMLYN,COMLEN)
  !         
  ! 

  ! Turn on Logical flag that says that the PSF has been modified.
  ! (Refer code.f90)
  MUSTUP=.TRUE.

  ! Initialization for setup
  IGMSEL(1:MAXAIM) = 0

  ! First fill QMINB array, and GHO should be last atoms  in COPSEL
  !     ISLCT: QM atoms
  !     JSLCT: Connection atoms
  !     LSLCT: GHO atoms
  IF(QLINK(1)) THEN
     CALL COPSEL_sq(ISLCT,ISLCT,LSLCT)
  ELSE IF(CLINK) THEN
     CALL COPSEL_sq(ISLCT,JSLCT,ISLCT)
  ELSE
     CALL COPSEL_sq(ISLCT,ISLCT,ISLCT)
  END IF

  natqm_1 = natqm(1)

  !
  ! for dual quantum region.
  QCG_corr(1:2)       =.false.
  tot_cg_fep(1:2)     = zero
  ener_bg_cg_corr(1:2)= zero
  If(QDUALQM) then
     if(QMFEP .or. QMLAY) then
        if(QLINK(2) .or. QH_Link) then
           call copsel_dual(ISLCT2,JLSLCT)
        else
           call copsel_dual(ISLCT2,ISLCT2)
        end if
     else if(QMSOLV) then
        call copsel_dual(ISLCT,ISLCT)
     end if

     ! for checkping charges.
     ! Here, in QM/MM-Ewald, the correction term is added only using
     ! QMFEP, otherwise the correction will be done in MM part.
     ! The equations used are based on background charge correction
     ! from Figuerirido et al. J. Chem. Phys. 1995 v103 p6133-6142,
     ! which is
     !
     ! E_background = half * q_tot^2 * (- pi / kappa*2 / volume)
     !
     ! Thus, in QM/MM-Ewald for QMFEP, the energy term should be
     ! added are
     !
     ! delE_background = E_1_background - E_2_background
     !                 = half*(q_tot_1^2-q_tot_2^2)*(- pi / kappa*2 / volume)
     !
     ! Since the background charge correction only affects in total
     ! electrostatic energy and virial, this will return same results
     ! as far as user uses "PMEPLSMA".
     !
     if(QMFEP .and. LQMEWD) then        ! later implement for "QMSOLV"
        tot_cg_local(1:2)=zero
        do i=1,natom
           if(igmsel(i).eq.0 .or. igmsel(i).eq.5) then
              tot_cg_local(1)=tot_cg_local(1)+cg(i)
           end if
           if(igmsel_dual(i).eq.0 .or. igmsel_dual(i).eq.5) then
              tot_cg_local(2)=tot_cg_local(2)+cg(i)
           end if
        end do
        do i=1,natqm(1)
           tot_cg_local(2)=tot_cg_local(2)+cginb(i)
        end do

        if(QLINK(1)) tot_cg_local(1)=tot_cg_local(1)+CGGHO 
        if(QLINK(2)) tot_cg_local(2)=tot_cg_local(2)+CGGHO_2

        tot_cg_local(1:2)=tot_cg_local(1:2)+qmcharge(1:2)

        if(abs(tot_cg_local(1)).ge.(zero+0.00001d0)) then
           QCG_corr(1)  =.true.
           tot_cg_fep(1)= tot_cg_local(1)*tot_cg_local(1)  ! squared here!
           if(Prnlev.ge.2) then
              WRITE(OUTU,34)'The total charge of 1st qm system is ',tot_cg_local(1)
              WRITE(OUTU,32)'Background charge correction will be used for 1st qm system.'
           end if
        end if
        if(abs(tot_cg_local(2)).ge.(zero+0.00001d0)) then
           QCG_corr(2)  =.true.
           tot_cg_fep(2)= tot_cg_local(2)*tot_cg_local(2)  ! squared here!
           if(Prnlev.ge.2) then
              WRITE(OUTU,34)'The total charge of 2nd qm system is ',tot_cg_local(2)
              WRITE(OUTU,32)'Background charge correction will be used for 2nd qm system.'
           end if
        end if
     end if

     ! for pKa FEP-QM/MM. (internal bonded MM terms for H-dummy atom.)
     if(QMFEP .and. QpKa_on) call ChkInternal_pKa(IpKa_Hatm,QPKNOANG,QPKNODIH,QPKNOIMP)
  End if

  natqm_2 = natqm(2)
  !
  ! Non-bond list and prepare for QM/MM-interaction list
  ! and setup QM/MM-non-bonded list
  !     If nonbond data structure has been set up, generate QM/MM
  !     nonbond lists here
  ISNBDS=.TRUE.
  IF (USEDDT_nbond(BNBND).AND.natqm_1.GT.0) THEN
!!!         IF (size(BNBND%INBLO ).LT.0) ISNBDS=.FALSE.
!!!         IF (size(BNBND%IBLO14).LT.0) ISNBDS=.FALSE.
!!!         IF (size(BNBND%INBLOG).LT.0) ISNBDS=.FALSE.
!!!         IF (size(BNBND%IGLO14).LT.0) ISNBDS=.FALSE.
!!!
!!!#if KEY_PBOUND==1
!!!         if(.not.qboun) then   
!!!#endif
!!!            IF(NTRANS.GT.0) THEN
!!!               IF(size(BIMAG%IMBLO ).LT.0) ISNBDS=.FALSE.
!!!               IF(size(BIMAG%IMBLOS).LT.0) ISNBDS=.FALSE.
!!!               IF(size(BIMAG%IMBLOG).LT.0) ISNBDS=.FALSE.
!!!               IF(size(BIMAG%IMBLOX).LT.0) ISNBDS=.FALSE.
!!!            END IF
!!!#if KEY_PBOUND==1
!!!         endif                 
!!!#endif
!!!
     !        update the non-bonded list and generate for QM/MM pairs
!!!         IF (.not. ISNBDS) CALL WRNDIE(1,'<SQMINI>','BNBND or BIMAG memory point negative integer.')
     CALL NBNDQM(X,Y,Z)
  ENDIF

  ! Swap coordinates, and make corresponding adjustments on
  ! MMINB array after Swap coordinates.
  natom_check= NATOM
  call chmalloc('sqnt_setup.src','SQMINI','XIM',NATOM,crl=XIM)
  call chmalloc('sqnt_setup.src','SQMINI','YIM',NATOM,crl=YIM)
  call chmalloc('sqnt_setup.src','SQMINI','ZIM',NATOM,crl=ZIM)
  CALL SwapXYZ_image(NATOM,X,Y,Z,XIM,YIM,ZIM,IMATTQ)

  !
  !     Get them ready for QM/MM calculations and SQUANTM call
  LFIRST = .true.
  CALL CH2SQM(natqm_1,IGMSEL,XIM,YIM,ZIM,LFIRST) 

  !
  IF(QLINK(1)) THEN
     CALL CMNQUA(NZUNC_1,ISLCT,ISLCT,LSLCT)
  ELSE IF(CLINK) THEN
     CALL CMNQUA(NZUNC_1,ISLCT,JSLCT,ISLCT)
  ELSE
     CALL CMNQUA(NZUNC_1,ISLCT,ISLCT,ISLCT)
  END IF

  !
  ! for dual quantum region.
  If(QDUALQM) call CMNQUA_dual(NZUNC_2)

  !
  !***************************************************************************
  !***************************************************************************
  ! Here's the part doing Initial Setup for NewMopac code from Amber.
  ! Everything should be done here...
  !
  !
  !***Note from namkh.
  !   Currenly, the atoms passed into New Mopac part only include
  !   atoms in primary cell (in case of Periodic boundary conditions.).
  !   Thus, all the wrapping should be done this interface before
  !   call any routine in New Mopac, thus it make it easier to manage
  !   the program.
  !***End note

  ! 1) Call routine to initialize setup on qmmm_module
  ! let the program ready for QM/MM part
  QCHECK        =.TRUE.
  natqm_1       = natqm(1)
  nqmcharge_1   = nqmcharge(1)
  nspin_1       = nspin(1)
  qlink_1       = qlink(1)
  Qdual_check(1)=(QMFEP .or. QMLAY)   !
  Qdual_check(2)=.FALSE.              ! 1st qm region
  CALL QMMM_INITIALIZE(NATOM,natqm_1,nqmtheory,nqmcharge_1, &
       nspin_1, &
       NZUNC_1,igmsel,qminb1_dual, &
       scfconv,CTOFNB,qlink_1,LQMEWD,NOPMEwald, &
       QMSWTCH, &
       QMMM_NoGauss,QMMM_NoDiis,QCHECK, &
       qmlay_local,Qdual_check, &
       LTRBOMD,N_scf_cycle)

  ! some check
  IF(.NOT.QCHECK) CALL WRNDIE(-5,'<SQMINI>', &
       'The CHARMM will stop at QMMM_INITIALIZE.')

  ! for dual quantum region.
  if(QMFEP .or. QMLAY) then
     QCHECK        =.TRUE.
     natqm_2       = natqm(2)
     nqmcharge_2   = nqmcharge(2)
     nspin_2       = nspin(2)
     qlink_2       = qlink(2)
     Qdual_check(2)=.TRUE.
     qmswitch_local= QMSWTCH
     qmewd_local   = LQMEWD
     qmnopmewd_local=NOPMEwald
     if(QMLAY) then
        qmswitch_local=.false.
        qmewd_local   =.false.
        qmnopmewd_local =.false.
     end if
     CALL QMMM_INITIALIZE(NATOM,natqm_2,nqmtheory,nqmcharge_2, &
          nspin_2, &
          NZUNC_2,igmsel_dual,qminb2_dual, &
          scfconv,CTOFNB,qlink_2,qmewd_local,qmnopmewd_local, &
          qmswitch_local, &
          QMMM_NoGauss,QMMM_NoDiis,QCHECK, &
          qmlay_local,Qdual_check, &
          LTRBOMD,N_scf_cycle)
     if(.not.QCHECK) CALL WRNDIE(-5,'<SQMINI>', &
          'The CHARMM will stop at 2nd QMMM_INITIALIZE.')
  end if

  ! 2) Allocates and setup for QM and QM-MM arrays, pair lists, and etc
  QCHECK=.TRUE.
  QFRST =.TRUE.
  !
  !    temporary use QIMAGE, but should be changed near future.
  If(QIMAGE) Then
     LPERIOD = 1
  Else 
     LPERIOD = 0
  End if
  !
  ! currently, I should assume MMINB array keep the Masking job,
  ! such that MM atoms MMINB(i) > 0 are within cutoff distance. 
  !
  Qdual_check(1)= (QMFEP .or. QMLAY)  !
  Qdual_check(2)=.false.              ! 1st qm region

  ! wrap before and after qm/mm setup...
  if(Qdual_check(1)) then
     qget_put      =.true.
     call PushPull_all_array(Qdual_check,qget_put)
  end if

  Indx_mminb = 1                      ! for 1st qm region 
  CALL QMMM_SETUP(MAXA,NATOM,Indx_mminb,mminb1_dual,mminb2_dual, &
       LPERIOD,PRNLEV,XIM,YIM,ZIM,CG,QCHECK,QFRST)
  ! some check
  IF(.NOT.QCHECK) CALL WRNDIE(-5,'<SQMINI>','The CHARMM will stop at QMMM_SETUP.')
  !
  if(Qdual_check(1)) then
     qget_put      =.false.
     call PushPull_all_array(Qdual_check,qget_put)
  end if

  ! for dual quantum region.
  if(QMFEP .or. QMLAY) then
     QCHECK=.TRUE.
     Qdual_check(2)=.true.
     qget_put      =.true.

     ! wrap before and after energy/gradient call.
     call PushPull_all_array(Qdual_check,qget_put)

     ! handle h-link coords here.
     if(QMLAY .and. QH_Link) then
        qget_h=.true.
        qsetup_h=.true.
        call get_hlink_coords(NHLink,MAXCHL,IHOSTGUEST,R_H_ref, &
             x,y,z,xyz_hlink,qget_h,qsetup_h)
     end if

     Indx_mminb = 2                   ! for 2nd qm region
     CALL QMMM_SETUP(MAXA,NATOM,Indx_mminb,mminb1_dual,mminb2_dual, &
          LPERIOD,PRNLEV,XIM,YIM,ZIM,CG,QCHECK,QFRST)

     IF(.NOT.QCHECK) CALL WRNDIE(-5,'<SQMINI>','The CHARMM will stop at 2nd QMMM_SETUP.')

     qget_put      =.false.
     call PushPull_all_array(Qdual_check,qget_put)

     !  for H atom to be annihiliated in pKa FEP-QM/MM.
     !cc        if(QMFEP .and. QpKa_on) cg(IpKa_Hatm)=zero

     ! for dual quantum region: handle h-link coords here.
     if(QMLAY .and. QH_Link) then
        qget_h=.false.
        qsetup_h=.false.
        call get_hlink_coords(NHLink,MAXCHL,IHOSTGUEST,R_H_ref, &
             x,y,z,xyz_hlink,qget_h,qsetup_h)
     end if

  end if

  ! 3) Load parameters and allocating memoried
  ! The first '.FALSE.' is for parallel case, when it is mynod==0
  If(Prnlev.ge.2) then
     QPRINT=.TRUE.
  Else
     QPRINT=.FALSE.
  End if
  QCHECK=.TRUE.
  Qdual_check(1)= (QMFEP .or. QMLAY)  !
  Qdual_check(2)=.false.              ! 1st qm region
  CALL QMMM_PARM_LOAD(QPRINT,QCHECK,Qdual_check)
  ! some check
  IF(.NOT.QCHECK) CALL WRNDIE(-5,'<SQMINI>','The CHARMM will stop at QMMM_PARM_LOAD.')

  ! for dual quantum region.
  If(QMFEP .or. QMLAY) then
     QCHECK=.TRUE.
     Qdual_check(2)=.true.
     CALL QMMM_PARM_LOAD(QPRINT,QCHECK,Qdual_check)

     IF(.NOT.QCHECK) CALL WRNDIE(-5,'<SQMINI>','The CHARMM will stop at 2nd QMMM_PARM_LOAD.')
  End if

  ! 4) QM/MM-Ewald setup
  QCHECK=.TRUE.
  Qdual_check(1)= (QMFEP .or. QMLAY)  !
  Qdual_check(2)=.false.              ! 1st qm region
  IF(LQMEWD) THEN
     CALL QMMM_EWALD_INITIALIZE(NATOM,NATQM(1),EWMODE,ERFMOD, &
          IGMSEL, &
          KMAXXQ,KMAXYQ,KMAXZQ,KSQMAXQ, &
          KAPPA,cginb_pka(1:NQMAX,1:1), &
          QpKa_on,QCHECK,Qdual_check)

     IF(.NOT.QCHECK) CALL WRNDIE(-5,'<SQMINI>', &
          'The CHARMM will stop at QMMM_EWALD_SETUP.')

     ! for dual quantum region.
     if(QMFEP .or. QMLAY) then
        QCHECK        =.TRUE.
        Qdual_check(2)=.true.
        CALL QMMM_EWALD_INITIALIZE(NATOM,NATQM(2),EWMODE,ERFMOD, &
             igmsel_dual, &
             KMAXXQ,KMAXYQ,KMAXZQ,KSQMAXQ, &
             KAPPA,cginb_pka(1:NQMAX,2:2), &
             QpKa_on,QCHECK,Qdual_check)

        IF(.NOT.QCHECK) CALL WRNDIE(-5,'<SQMINI>', &
             'The CHARMM will stop at 2nd QMMM_EWALD_SETUP.')
     end if
  END IF


  ! 5) Print memory usage
  QCHECK=.TRUE.
  !     If(Prnlev.ge.2) then
  !        QPRINT=.TRUE.
  !     Else
  !        QPRINT=.FALSE.
  !     End if
  Qdual_check(1)= (QMFEP .or. QMLAY)  !
  Qdual_check(2)=.false.              ! 1st qm region
  if(Qdual_check(1).and.QPRINT) WRITE(OUTU,32) 'DUALqm: 1st QM region memory and coords.'
32 FORMAT(/,'SQMINI> ',A)
34 FORMAT(/,'SQMINI> ',A,F12.5)

  CALL QMMM_PRNT_MEM(NATOM,QPRINT,QCHECK,Qdual_check)
  ! some check
  IF(.NOT.QCHECK) CALL WRNDIE(-5,'<SQMINI>','The CHARMM will stop at QMMM_PRNT_MEM.')

  ! for dual quantum region
  if(QMFEP .or. QMLAY) then
     QCHECK        =.TRUE.
     Qdual_check(2)=.true.
     if(QPRINT) WRITE(OUTU,32)'DUALqm: 2nd QM region memory and coords.'

     CALL QMMM_PRNT_MEM(NATOM,QPRINT,QCHECK,Qdual_check)
     IF(.NOT.QCHECK) CALL WRNDIE(-5,'<SQMINI>','The CHARMM will stop at 2nd QMMM_PRNT_MEM.')
  end if


  ! for dual quantum region.
  ! Wrap before and after energy/gradient calculations.
  Qdual_check(1)= (QMFEP .or. QMLAY)  !
  Qdual_check(2)=.false.              ! 1st qm region
  if(Qdual_check(1)) then
     qget_put      =.true.
     call PushPull_all_array(Qdual_check,qget_put)
  end if


  ! setup distance arrays for QM-QM and QM-MM here.
  ! This array is used in QMMM_Ewald_Setup_And_Pot when "incore" used.
  QFRST =.TRUE.
  CALL QMMM_Distance_setup(NATOM,QFRST,Qdual_check)

  ! 6) Do initiali allocation for qm-mm energy evaluation.
  !    It is not calculating energy here. Just do First call stuffs.
  QCHECK=.TRUE.
  QFRST =.TRUE.
  ESCF  = ZERO

  CALL QMMM_Energy(ESCF,NATOM,QFRST,QCHECK,Qdual_check, &
                   q_apply_tr_bomd,i_md_step)
  ! some check
  IF(.NOT.QCHECK) CALL WRNDIE(-5,'<SQMINI>','The CHARMM will stop at QMMM_Energy.') 
  ! 
  if(Qdual_check(1)) then
     qget_put      =.false.
     call PushPull_all_array(Qdual_check,qget_put)
  end if


  ! for dual quantum region
  if(QMFEP .or. QMLAY) then
     QCHECK=.TRUE.
     QFRST =.TRUE.
     ESCF  = ZERO
     Qdual_check(2)=.true.
     qget_put      =.true.

     ! wrap before and after energy/gradient call.
     call PushPull_all_array(Qdual_check,qget_put)

     ! handle h-link coords here. (This should be moved upward at QMMM_Setup.)
     if(QMLAY .and. QH_Link) then
        qget_h=.true.
        qsetup_h=.false.
        call get_hlink_coords(NHLink,MAXCHL,IHOSTGUEST,R_H_ref, &
             x,y,z,xyz_hlink,qget_h,qsetup_h)
     end if

     ! setup distance arrays for QM-QM and QM-MM here.
     ! This array is used in QMMM_Ewald_Setup_And_Pot when "incore" used.   
     CALL QMMM_Distance_setup(NATOM,QFRST,Qdual_check)

     CALL QMMM_Energy(ESCF,NATOM,QFRST,QCHECK,Qdual_check, &
                      q_apply_tr_bomd,i_md_step)

     IF(.NOT.QCHECK) CALL WRNDIE(-5,'<SQMINI>','The CHARMM will stop at 2nd QMMM_Energy.')

     ! for MLAYer
     if(QMLAY) then
        call Setup_high_qm

        !     handle h-link coords here.
        if(QH_Link) then
           qget_h=.false.
           qsetup_h=.false.
           call get_hlink_coords(NHLink,MAXCHL,IHOSTGUEST,R_H_ref, &
                x,y,z,xyz_hlink,qget_h,qsetup_h)
        end if
     end if

     qget_put      =.false.
     call PushPull_all_array(Qdual_check,qget_put)
  end if

  !
  !***************************************************************************

  ! free memory allocations.
  if(allocated(ISLCT)) call chmdealloc('sqnt_setup.src','SQMINI','ISLCT',NATOM,intg=ISLCT)
  if(allocated(JSLCT)) call chmdealloc('sqnt_setup.src','SQMINI','JSLCT',NATOM,intg=JSLCT)
  if(allocated(LSLCT)) call chmdealloc('sqnt_setup.src','SQMINI','LSLCT',NATOM,intg=LSLCT)
  if(allocated(ISLCT2)) call chmdealloc('sqnt_setup.src','SQMINI','ISLCT2',NATOM,intg=ISLCT2)
  if(allocated(JLSLCT)) call chmdealloc('sqnt_setup.src','SQMINI','JLSLCT',NATOM,intg=JLSLCT)
  !***************************************************************************

  COMLEN = 0
  !
  RETURN
END SUBROUTINE SQMINI
!
SUBROUTINE CMNQUA(NZUNC,ISLCT,JSLCT,LSLCT)
  !----------------------------------------------------------------------
  !     Find the atoms defined as QM atoms and get them ready
  !     for SQUANTM.
  !
  !     Paul D Lyne, September 1995
  !
  use chm_kinds
  use dimens_fcm
  use number
  use exfunc
  !
  !...  use coord
  use param
  use psf
  use stream
  use gamess_fcm
! DTM
  use linkatom
  use rtf, only: atct
  use squantm
  !
  !
  implicit none

  INTEGER NZUNC(*),ISLCT(*),JSLCT(*),LSLCT(*)
  CHARACTER(len=10) AATOM(NQMAX)
  real(chm_real):: AZNUC(NQMAX)
  !
  INTEGER:: I,N,NSLCT,NATMM,NATLNK,NACA,NACG
  CHARACTER(len=6) ELE
  LOGICAL:: QPRT
  !
  QPRT=.TRUE.
  !
  N=0
  DO I = 1, natqm(1)
     N          = iabs(qminb1_dual(I))
     AATOM(I)   = ATYPE(N)
     IF(ATYPE(N)(1:3).EQ.'QQH') AATOM(I)=' H'
     !
     CALL FINDEL(ATCT(IAC(N)),AMASS(N),N,ELE,AZNUC(I),QPRT)

     !     assign nuclear charges
     !     GHO atom: 85
     IF(QLINK(1).AND.LSLCT(N).EQ.1) THEN
        NZUNC(I) = 85
        !
        !     ACA connection atom: 86
     ELSE IF(CLINK.AND.JSLCT(N).EQ.1) THEN 
        NZUNC(I) = 86
     ELSE
        NZUNC(I) = AZNUC(I)
     END IF
  END DO
  !
  NATMM = NATOM -natqm(1)
  !
  ! number of H-link atoms
  NSLCT = 0
  DO I = 1,NATOM
     IF (IGMSEL(I).EQ.2) NSLCT = NSLCT + 1
  END DO
  NATLNK = NSLCT
  ! number of Adjusted connecion atoms
  NSLCT = 0
  IF(CLINK) THEN
     DO I = 1,NATOM
        IF(JSLCT(I).EQ.1) NSLCT = NSLCT + 1
     END DO
  END IF
  NACA = NSLCT
  NCATM= NACA
  ! number of GHO atoms
  NSLCT = 0
  IF(QLINK(1)) THEN
     DO I = 1, NATOM
        IF(LSLCT(I).EQ.1) NSLCT = NSLCT + 1
     END DO
  END IF
  NACG = NSLCT
  !
  !     Write out atomic information
  !
  IF (PRNLEV.GT.2) WRITE (OUTU,'(/,1x,A,/)') &
       ' SQMINI> Some atoms will be treated quantum mechanically.'
  IF (PRNLEV.GT.2) THEN
     WRITE (OUTU,'(8X,A,I5,/,8X,A,I5)') &
          ' The number of quantum mechanical atoms   = ',natqm(1), &
          ' The number of QM/MM H-link atoms         = ',NATLNK

     IF(CLINK) WRITE (OUTU,'(8X,A,I5)') &
          ' The number of Adjusted Connection atoms  = ',NACA
     IF(QLINK(1)) WRITE (OUTU,'(8X,A,I5)') &
          ' The number of GHO atoms                  = ',NACG

     WRITE (OUTU,'(2(8X,A,I5,/),/)') &
          ' The number of molecular mechanical atoms = ',NATMM, &
          ' The number of MM atoms excluded from QM  = ',NATMM-NCUTOFF(1)
  END IF

  RETURN
END SUBROUTINE CMNQUA
!
SUBROUTINE COPSEL_sq(ISLCT,JSLCT,LSLCT)
  !-----------------------------------------------------------------------
  !     Copies selection vector to common block 
  !     so it may be used by GAMESS/CADPAC/SQUANTM interfaces
  !     Call this routine only once and retain definition
  !     of QM, MM, and link atoms throughout the calculation.
  !     We call this from GAMINI/CADINI/SQMINI which is called 
  !     from charmm/charmm.src
  !
  !     IGMSEL(I) = 5  MM atom to be excluded from QM/MM non-bonded
  !                    interaction
  !     IGMSEL(I) = 2  Hydrogen Link atom (QQH)
  !     IGMSEL(I) = 1  QM atom
  !     IGMSEL(I) = 0  MM atom
  !
  !     Not yet supported, but it will come soon (How soon?)
  !     IGMSEL(I) = -1 QM atom  (other replica)
  !     IGMSEL(I) = -2 Link atom (other replica)
  !     IGMSEL(I) = -5 MM atom to be excluded from its QM/MM
  !                    non-bonded interaction (other replica)
  !     IGMSEL(I) = -6 MM atom (other replica)
  !
  !     MM atom in position close to link atom is excluded from interaction
  !     of external charges to QM region. Instead of this atom is already
  !     a link atom so no need for two atoms in one place!
  !
  use chm_kinds
  use exfunc
  use chutil, only : getres,atomid
  use dimens_fcm
  !
  !...  use coord
  use gamess_fcm
  use stream
  use psf
  use number
  use squantm
  !
  implicit none
  INTEGER ISLCT(*), JSLCT(*), LSLCT(*)

  INTEGER I,J,I1,I2,N,IS,IQ,NLATQ
  CHARACTER(len=4) SID, RID, REN, AC
  LOGICAL LNFLAG, Qdual_check(2)
  INTEGER LN,Ncnt
  !
  ! initialize number of GHO atoms.
  NumGHO(1:2) = 0
  NHLink      = 0                            ! for H-link atom (2nd qm)

  DO I=1, NATOM
     IGMSEL(I)=ISLCT(I)
     !  for link H atom
     IF (ATYPE(I)(1:2).EQ.'QQ')    IGMSEL(I)=2
     !  for C-connection atom
     IF (CLINK.AND.JSLCT(I).EQ.1) IGMSEL(I)=1
     !  for GHO atom
     IF (QLINK(1).AND.LSLCT(I).EQ.1) IGMSEL(I)=1
  END DO

  !  initialize: mminb arrays, qminb arrays, and cginb array.
  cginb(1:nqmax)         =zero
  qminb1_dual(1:nqmax)   =0
  qminb2_dual(1:nqmax)   =0
  mminb1_dual(1:maxa,1:2)=0
  mminb2_dual(1:maxa,1:2)=0
  !
  ! namkh
  !     First, fill QMINB array, in which GHO atoms are last
  NLATQ = 0
  Ncnt  = 0
  IF(QLINK(1)) THEN
     DO I=1, NATOM
        !        pure QM atoms first
        IF((IGMSEL(I).EQ.1 .OR.  IGMSEL(I).EQ.2) .AND. LSLCT(I).EQ.0) THEN
           NLATQ        = NLATQ+1
           qminb1_dual(NLATQ) = I
        END IF
     END DO
     DO I=1, NATOM
        !        GHO atom
        IF(LSLCT(I).EQ.1) THEN
           NLATQ        = NLATQ+1
           qminb1_dual(NLATQ) = I
           Ncnt         = Ncnt + 1
        END IF
     END DO
  ELSE
     DO I=1, NATOM
        IF(IGMSEL(I).EQ.1.OR.IGMSEL(I).EQ.2) THEN
           NLATQ       = NLATQ+1
           qminb1_dual(NLATQ) = I
        END IF
     END DO
  END IF
  natqm(1) = NLATQ
  NumGHO(1)= Ncnt                  ! no. of gho atoms.
  !
  IF (natqm(1).LE.0) CALL WRNDIE(0,'<COPSEL>','No quantum mechanical atoms selected.')
  !
  !     Check if link atom is connected to any of its neighbors. If
  !     yes then that atom will not be included in QM/MM interaction.
  !     This is sometimes necessary to prevent opposite charge collision,
  !     since QM cannot prevent this to happen.
  !
  !
  DO I=1,NBOND
     I1=IB(I)
     I2=JB(I)
     !
     ! For connection atom approach, link host atom or group should be
     ! removed from QM/MM SCF procedure
     IF(IGMSEL(I1).EQ.1.AND.IGMSEL(I2).EQ.0) THEN

        IF(QLINK(1).AND.LSLCT(I1).EQ.1) GOTO 131

        IF(QGMEXG) THEN
           !              remove the entire group
           J=GETRES(I2,IGPBS,NGRP)
           IS=IGPBS(J)+1
           IQ=IGPBS(J+1)
           DO J=IS,IQ
              IF(IGMSEL(J).EQ.0) IGMSEL(J)=5
           ENDDO
        ELSE
           !              remove the link host atom
           IF(IGMSEL(I2).EQ.0) IGMSEL(I2)=5
        ENDIF

131     CONTINUE
     END IF
     IF(IGMSEL(I1).EQ.0.AND.IGMSEL(I2).EQ.1) THEN

        IF(QLINK(1).AND.LSLCT(I2).EQ.1) GOTO 132

        IF(QGMEXG) THEN
           !              remove the entire group
           J=GETRES(I1,IGPBS,NGRP)
           IS=IGPBS(J)+1
           IQ=IGPBS(J+1)
           DO J=IS,IQ
              IF(IGMSEL(J).EQ.0) IGMSEL(J)=5
           ENDDO
        ELSE
           !              remove the link host atom
           IF(IGMSEL(I1).EQ.0) IGMSEL(I1)=5
        ENDIF
132     CONTINUE
     END IF
     !
     ! For the link hydrogen atom
     IF (IGMSEL(I1).EQ.2) THEN
        !           Don't change QM atoms
        IF(QGMEXG) THEN
           !              remove the entire group
           J=GETRES(I2,IGPBS,NGRP)
           IS=IGPBS(J)+1
           IQ=IGPBS(J+1)
           DO J=IS,IQ
              IF(IGMSEL(J).EQ.0) IGMSEL(J)=5
           ENDDO
        ELSE
           !              remove the link host atom
           IF(IGMSEL(I2).EQ.0) IGMSEL(I2)=5
        ENDIF
     ENDIF
     IF (IGMSEL(I2).EQ.2) THEN
        IF(QGMEXG) THEN
           !              remove the entire group
           J=GETRES(I1,IGPBS,NGRP)
           IS=IGPBS(J)+1
           IQ=IGPBS(J+1)
           DO J=IS,IQ
              IF(IGMSEL(J).EQ.0) IGMSEL(J)=5
           ENDDO
        ELSE
           !              remove the link host atom
           IF(IGMSEL(I1).EQ.0) IGMSEL(I1)=5
        ENDIF
     ENDIF
  ENDDO
  !
  IF(PRNLEV.GE.2) THEN
     WRITE(OUTU,118)
     WRITE(OUTU,120)  &
          'Classical atoms excluded from the QM calculation'
  ENDIF
118 FORMAT('------------------------------------------------')
120 FORMAT('SQMINI: ',A,':')
122 FORMAT(10X,I5,4(1X,A4))
123 FORMAT(10X,I5,4(1X,A4),1X,'*')
124 FORMAT(10X,'NONE.')
  N=0
  DO I=1,NATOM
     IF(IGMSEL(I).EQ.5) THEN
        CALL ATOMID(I,SID,RID,REN,AC)
        IF(PRNLEV.GE.2) WRITE(OUTU,122) I,SID,RID,REN,AC
        N=N+1
     ENDIF
  ENDDO
  IF(PRNLEV.GE.2) THEN
     IF(N.EQ.0) WRITE(OUTU,124)
     WRITE(OUTU,120)  &
          'Quantum mechanical atoms, (* is Connection atom)'
  ENDIF
  N=0
  IF(CLINK) THEN
     DO I=1,NATOM
        IF(IGMSEL(I).EQ.1) THEN
           CALL ATOMID(I,SID,RID,REN,AC)
           IF(JSLCT(I).EQ.1) THEN
              IF(PRNLEV.GE.2) WRITE(OUTU,123) I,SID,RID,REN,AC
           ELSE
              IF(PRNLEV.GE.2) WRITE(OUTU,122) I,SID,RID,REN,AC
           END IF
           N=N+1
        END IF
     END DO
  ELSE
     DO I=1,NATOM
        IF(IGMSEL(I).EQ.1) THEN
           CALL ATOMID(I,SID,RID,REN,AC)
           IF(PRNLEV.GE.2) WRITE(OUTU,122) I,SID,RID,REN,AC
           N=N+1
        ENDIF
     ENDDO
  END IF
  IF(PRNLEV.GE.2) THEN
     IF(N.EQ.0) WRITE(OUTU,124)
     WRITE(OUTU,120) 'Quantum mechanical Hydrogen link atoms'
  ENDIF
  N=0
  DO I=1,NATOM
     IF(IGMSEL(I).EQ.2) THEN
        CALL ATOMID(I,SID,RID,REN,AC)
        IF(PRNLEV.GE.2) WRITE(OUTU,122) I,SID,RID,REN,AC
        N=N+1
     ENDIF
  ENDDO
  IF(QLINK(1)) THEN
     IF(PRNLEV.GE.2) THEN
        IF(N.EQ.0) WRITE(OUTU,124)
        WRITE(OUTU,120) 'Quantum mechanical GHO atoms'
     END IF
     N=0
     DO I=1,NATOM
        IF(LSLCT(I).EQ.1) THEN
           CALL ATOMID(I,SID,RID,REN,AC)
           IF(PRNLEV.GE.2) WRITE(OUTU,122) I,SID,RID,REN,AC
           N=N+1
        ENDIF
     END DO
  END IF
  IF(PRNLEV.GE.2) THEN
     IF(N.EQ.0) WRITE(OUTU,124)
     WRITE(OUTU,118)
  ENDIF
  !
  ! for dual quantum region: for 2nd qm region, refer COPSEL_Dual,
  ! in which the cginb for 2nd qm atoms are zeroed.
  if(QDUALQM) then
     do i=1,natqm(1)
        cginb(i) = cg(iabs(qminb1_dual(i)))
     end do
     If(QLINK(1)) then                     ! for charge on gho atoms.
        Qdual_check(1)=(QMFEP.or.QMLAY)
        Qdual_check(2)=.false.
        call Get_Charge_GHO(natqm(1),qminb1_dual,cginb,Qdual_check)
     end if
     !  for pKa FEP-QM/MM. 
     do i=1,natqm(1)
        cginb_pka(i,1)=cginb(i)
     end do
  end if
  !
  ! Zero charges on QM atoms to remove from MM term.
  IF(QGMREM) THEN
     DO I = 1, NATOM
        IF(IGMSEL(I).EQ.1.OR.IGMSEL(I).EQ.2) CG(I) = ZERO
     END DO
  END IF

  RETURN
END SUBROUTINE COPSEL_sq
!
!
SUBROUTINE PBCHECK(DXI,DYI,DZI)
  !
  use chm_kinds
  use dimens_fcm
  use number
  use pbound
#if KEY_PBOUND==1 /*pbound*/

  implicit none
  real(chm_real) CORR
  real(chm_real) DXI,DYI,DZI

  If(qBoun) then                                    
     If(qCUBoun.or.qTOBoun) then
        DXI = BOXINV * DXI
        DYI = BOYINV * DYI
        DZI = BOZINV * DZI
        IF(DXI.GT.  HALF) DXI = DXI - ONE
        IF(DXI.LT. -HALF) DXI = DXI + ONE
        IF(DYI.GT.  HALF) DYI = DYI - ONE
        IF(DYI.LT. -HALF) DYI = DYI + ONE
        IF(DZI.GT.  HALF) DZI = DZI - ONE
        IF(DZI.LT. -HALF) DZI = DZI + ONE
        If (qTOBoun) Then
           CORR = HALF * AINT ( R75 * (ABS(DXI) + &
                ABS(DYI) + &
                ABS(DZI)))
           DXI = DXI - SIGN( CORR,  DXI  )
           DYI = DYI - SIGN( CORR,  DYI  )
           DZI = DZI - SIGN( CORR,  DZI  )
        Endif
        DXI = XSIZE * DXI
        DYI = YSIZE * DYI
        DZI = ZSIZE * DZI
     Else
        Call PBMove(DXI, DYI, DZI)
     Endif
  Endif                                             
#endif /* (pbound)*/
  RETURN
END SUBROUTINE PBCHECK
!
#else /*          (mainsquatn)*/
SUBROUTINE SQMINI(COMLYN,COMLEN)
  implicit none
  character(len=*) :: comlyn
  integer comlen
  CALL WRNDIE(-1,'<SQMINI>','SQUANTM code not compiled.')
  RETURN
END SUBROUTINE SQMINI
#endif /*     (mainsquatn)*/
!

