SUBROUTINE QMDEFN(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     Define the set of quantum mechanical atoms and set up the data
  !     structures.
  !
  !
  use chm_kinds
  use dimens_fcm
  use memory
  use coord
  use psf
  use stream
  use am1parm
  use sizes
  use qmlinkm
  use leps
  use nbndqm_mod
  use quantm
  use select
  use string
  use gamess_fcm
  implicit none
  !
  !
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  !
  integer,allocatable,dimension(:) :: ISLCT
  integer,allocatable,dimension(:) :: LSLCT
  !
#if KEY_QUANTUM==0 /*qmsetmain*/
  CALL WRNDIE(-1,'<QMDEFN>','QUANTUM code not compiled.')
  RETURN
END SUBROUTINE QMDEFN
#else /* (qmsetmain)*/
  !
  !
  qmused = .true.      ! safe here since next are the allocations, ...
  ! added by clb to turn off quantum
  if (indxa(comlyn,comlen,'CLEA') .gt. 0) then
     qmused = .false.
     qmused_quantum = .false.
     natqm = 0
     natqm_check = 0
     return
  endif

  call allocate_gamess ! try to reduce from MAXA to NATOM
  call allocate_quantum
  !
  call chmalloc('qmset.src','QMDEFN','ISLCT',NATOM,intg=ISLCT)
  call chmalloc('qmset.src','QMDEFN','LSLCT',NATOM,intg=LSLCT)    ! for GHO atoms
  CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)

  !
  ! QM GROUP OPTION ... JG 5/02
  !
  QNBGRP = (INDXA(COMLYN,COMLEN,'GROU').GT.0)
  IF(PRNLEV.GE.2 .and. QNBGRP) THEN
     WRITE(OUTU,*) 'QMDEFN> QM/MM GROUP INTERACTION OPTION ACTIVATED.'
  ENDIF
  !
  !      LINK ATOM SELECTION
  !
  QMLINK = (INDXA(COMLYN,COMLEN,'GLNK').GT.0)
  IF(QMLINK) CALL SELCTA(COMLYN,COMLEN,LSLCT,X,Y,Z,WMAIN,.TRUE.)
  ! MG 5/2002
  ! Average charges during dynamics
  CHDYN = (INDXA(COMLYN, COMLEN, 'CHDY') .GT. 0)
  IF(PRNLEV.GE.2 .and. CHDYN) WRITE(OUTU,*)' QMSET> CHDYN = ', CHDYN
  !
  !------------------------------------------------------------------
  ! start modifications for LEPS
  !     Cristobal, LADH project, Amhers 1999
  !
  QLEPS = (INDXA(COMLYN,COMLEN,'LEPS').GT.0)
  !
  IF(QLEPS) CALL SETLEPS(COMLYN,COMLEN)
  !
  ! end modifications for leps
  !------------------------------------------------------------------
  !
  !
  ! namkh 08/08/04
  ! QM/MM-Ewald
  CALL QMDFN2(COMLYN,COMLEN,ISLCT,LSLCT, &
       NATOM,IAC,AMASS,NBOND,IB,JB,CGTOT,CG)
  !
  if(allocated(ISLCT)) call chmdealloc('qmset.src','QMDEFN','ISLCT',NATOM,intg=ISLCT)
  if(allocated(LSLCT)) call chmdealloc('qmset.src','QMDEFN','LSLCT',NATOM,intg=LSLCT)
  COMLEN = 0
  !
  !
  RETURN
END SUBROUTINE QMDEFN
!
! namkh 08/08/04
! QM/MM-Ewald
SUBROUTINE QMDFN2(COMLYN,COMLEN,ISLCT,LSLCT, &
     NATOM,IAC,AMASS,NBOND,IB,JB,CGTOT,CG)
  !-----------------------------------------------------------------------
  !     The set of atoms whose energies are to calculated quantum
  !     mechanically are defined here and the appropriate data structures
  !     set up.
  !
  use ewald,only: lewald
  use chm_kinds
  use dimens_fcm
  use consta
  use number
  use coord
  use rtf,only:atct
  use scfblk
  use linkatom
  use am1parm
  use gamess_fcm
  use stream
  use string
  use sizes
  use qmlinkm
  use quantm
  !
  ! for parallel run
#if KEY_PARALLEL==1
  use parallel
#endif 
#if KEY_FLUCQ==1
  use flucq
#endif 
  !
  implicit none
  !
  ! This is added for QM/MM-Ewald compatability check
  CHARACTER(len=*) COMLYN
  INTEGER       COMLEN, IAC(*), IB(*), ISLCT(*), LSLCT(*), JB(*), &
       NATOM, NBOND
  real(chm_real)        AMASS(*)
  ! BIOVIA Code Start : Bug fix
  LOGICAL QMNULL
  ! BIOVIA Code End
  !
  ! QM/MM-Ewald
  real(chm_real) CGTOT, CG(*)
  !
  CHARACTER(len=6) ELE
  INTEGER       I, IATOM, J, JATOM, NATLNK, NSLCT
  INTEGER       NATMM, ILINK
  real(chm_real)        ZNUM

  !  Default uses Mopac parallel code. Path-integral will raise flag.
#if KEY_PARALLEL==1
  QMPI = .FALSE.
#endif 

  !
  ! Check for keywords REMOve or EXGR
  !
  QGMREM=(INDXA(COMLYN,COMLEN,'REMO').GT.0)
  IF(PRNLEV.GE.2) THEN
     IF(QGMREM) THEN
        WRITE(OUTU,22) 'REMOve: Classical energies within QM atoms are removed.'
     ELSE
        WRITE(OUTU,22) 'Default quantum code used for QM/MM removal .'
     ENDIF
  End if
  !
  QGMEXG=(INDXA(COMLYN,COMLEN,'EXGR').GT.0)
  IF(PRNLEV.GE.2) THEN
     IF(QGMEXG) THEN
        WRITE(OUTU,22) 'EXGRoup: QM/MM Electrostatics for link host groups removed.'
     ELSE
        WRITE(OUTU,22) 'No EXGRoup: QM/MM Elec. for link atom host only is removed.'
     ENDIF
  ENDIF
22 FORMAT('GAMINT> ',A)
  !
  !-------------------------------------
  !     Step for a new generation of Initial guess
  !     It only applied when using Dynamics.
  !     It's been introduced to avoid a propagation of density error
  !     during dynamics over the time.
  !     ---namkh 05/12/05---
  NEWGS = GTRMI(COMLYN,COMLEN,'NGUE',-100)
  NSTPMD= 0
  IF(NEWGS.GT.0.AND.QMLINK) THEN 
     CALL WRNDIE(-1,'<QMDFN2>','NGUE and GLINK are not compatable.')
     NEWGS = -NEWGS
     IF(PRNLEV.GE.2) WRITE(OUTU,'(A)')'NGUE option will be ignored.'
  END IF
  IF(NEWGS.GT.0.AND.PRNLEV.GE.2) WRITE(OUTU,'(A,1X,I6,A)') &
       ' QMDFN2> Initial guess will be generated every',NEWGS,' steps during dynamics.' 

  !-------------------------------------
  !     Initialization of QM/MM electrostatic free energy perturbation:
  !     QM-MM Free energy perturbation calculations using the method described
  !     in J. Gao, J. Phys. Chem. 1992, 96, 537.
  !
  QMPERT = (INDXA(COMLYN,COMLEN,'PERT').GT.0)
  IF(QMPERT) THEN
     RLAMB0 = GTRMF(COMLYN,COMLEN,'REF0',ONE)
     RLAMB1 = GTRMF(COMLYN,COMLEN,'PER1',ONE)
     RLAMB2 = GTRMF(COMLYN,COMLEN,'PER2',ONE)
     FACTP1 = RLAMB1/RLAMB0
     FACTP2 = RLAMB2/RLAMB0
     IF (RLAMB0.EQ.RLAMB1 .OR. RLAMB0.EQ.RLAMB2) CALL &
          WRNDIE(0,'<QMDFN2>','Perturbation ill defined.')
     IF(PRNLEV.GE.2) WRITE(OUTU,'(A,1X,3F6.3)') &
          ' QMDFN2> QM/MM electrostatic FEP will be performed for',RLAMB0,RLAMB1,RLAMB2 
     TEMPERATURE = GTRMF(COMLYN,COMLEN,'TEMP',ZEROC)
     BETA_QMMM = ONE/(KBOLTZ*TEMPERATURE)
     ! For Ewald part
     QCGSC  = .FALSE.
  ENDIF
  !
  !  Initialization of QM/MM energy decomposition:
  !  J. Gao, X. Xia, Science, 1992, 258, 631.
  !
  QDECOM = (INDXA(COMLYN,COMLEN,'DECO').GT.0)
  IF(QDECOM) THEN
     IF(PRNLEV.GE.2) WRITE(OUTU,'(A)') &
          ' QMDFN2> QM/MM energy decomposition will be performed'
     !  JG 5/2002
     IF (QMLINK) CALL WRNDIE(-5,'<QMDFN2>','DECO and GHO not implemented')
  ENDIF
  !-------------------------------------
  !**********************************************************************
  ! namkh 08/08/04
  ! QM/MM-Ewald
  LQMEWD=.FALSE.
  IF(LEWALD) THEN
     LQMEWD=.TRUE.
     !        LQMEWD= (INDXA(COMLYN,COMLEN,'NEWD').GT.0)
  END IF
  IF(LQMEWD) THEN
     ! Checking
     IF(QGMREM.OR.QGMEXG) CALL WRNDIE(-5,'<QMDEFN>', &
          'Current QM/MM-Ewald is not compatible with REMO or EXGR.')
     !     On the process of implementing QM/MM-Ewald
     !        IF(QMPERT) CALL WRNDIE(-1,'<QMDEFN>','Current QM/MM-Ewald is not tested with PERT.')
     IF(QDECOM) CALL WRNDIE(-1,'<QMDEFN>','Current QM/MM-Ewald is not tested with DECO.') 
     IF(CHDYN)  CALL WRNDIE(-1,'<QMDEFN>','Current QM/MM-Ewald is not tested with CHDY.')
     !
#if KEY_FLUCQ==1
     IF(QFLUC) CALL WRNDIE(-2,'<QMDEFN>','Current QM/MM-Ewald is not tested with FLUCQ.')
#endif 
     !
     IF(PRNLEV.GE.2) WRITE(outu,22) 'Ewald with QM/MM Option has been Activated.'
     EWMODE = 1
     EWMODE = GTRMI(COMLYN,COMLEN,'NEWD',1)
     IF(EWMODE.EQ.0) THEN
        IF(PRNLEV.GE.2) WRITE(outu,22) 'Ewald with QM/MM Option use Input MM charges on QM atoms'
        CALL WRNDIE(-1,'<QMDEFN>','This mode is not available yet. Use default')
        EWMODE = 1
        !
     ELSE IF(EWMODE.EQ.-1) THEN
        IF(PRNLEV.GE.2) WRITE(outu,22) 'MM in Ewald Sum do not polarize QM atoms'
        CALL WRNDIE(-1,'<QMDEFN>','This mode is not available yet. Use default')
        EWMODE = 1
        !
     ELSE IF(EWMODE.EQ.1) THEN
        continue
     ELSE
        CALL WRNDIE(-1,'<QMDEFN>','Not supported option.')
        EWMODE = 1
     END IF
     !
     IF(PRNLEV.GE.2) then
        if(EWMODE.EQ.1) WRITE(outu,22) 'Default Ewald with QM/MM Option uses Mulliken Charges'
        WRITE(outu,22) 'MM within cutoff interact with regular way with QM'
        IF(EWMODE.EQ.1.OR.EWMODE.EQ.0) WRITE(outu,22) &
             'MM from Ewald Sum interact with diagonal elements in QM'
     END IF
     !
     ! Now setup for Ewald in QM/MM SCF
     ! default values
     KMAXQ  = 5
     KSQMAXQ= 27
     !
     KMAXQ  =GTRMI(COMLYN,COMLEN,'KMAX',KMAXQ)
     !                     added by Scott Feller 5/24/95, NIH
     KMAXXQ =GTRMI(COMLYN,COMLEN,'KMXX',KMAXQ)
     KMAXYQ =GTRMI(COMLYN,COMLEN,'KMXY',KMAXQ)
     KMAXZQ =GTRMI(COMLYN,COMLEN,'KMXZ',KMAXQ)
     KSQMAXQ=GTRMI(COMLYN,COMLEN,'KSQM',KSQMAXQ)
     ! charge handling on QM regions.
     NOQMIM=.FALSE.
     !           in case of ignoring QM images: Ignore QM images (IGNO
     NOQMIM=(INDXA(COMLYN,COMLEN,'IGNO') .GT. 0)
     IF(PRNLEV.GE.2.AND.NOQMIM) WRITE(outu,22) 'QM atoms in image cell will be ignored.'

     ! For QMPERT (charge scaling of counter-ions)
     IF(QMPERT.AND. .NOT.NOQMIM) THEN
        NUMCGS=GTRMI(COMLYN,COMLEN,'NUMC',0)
        IF(NUMCGS.GT.0) THEN
           QCGSC=.TRUE.
           if(NUMCGS.ge.1)  NATMCGS(1) =GTRMI(COMLYN,COMLEN,'NUC0',0)
           if(NUMCGS.ge.2)  NATMCGS(2) =GTRMI(COMLYN,COMLEN,'NUC1',0)
           if(NUMCGS.ge.3)  NATMCGS(3) =GTRMI(COMLYN,COMLEN,'NUC2',0)
           if(NUMCGS.ge.4)  NATMCGS(4) =GTRMI(COMLYN,COMLEN,'NUC3',0)
           if(NUMCGS.ge.5)  NATMCGS(5) =GTRMI(COMLYN,COMLEN,'NUC4',0)
           if(NUMCGS.ge.6)  NATMCGS(6) =GTRMI(COMLYN,COMLEN,'NUC5',0)
           if(NUMCGS.ge.7)  NATMCGS(7) =GTRMI(COMLYN,COMLEN,'NUC6',0)
           if(NUMCGS.ge.8)  NATMCGS(8) =GTRMI(COMLYN,COMLEN,'NUC7',0)
           if(NUMCGS.ge.8)  NATMCGS(9) =GTRMI(COMLYN,COMLEN,'NUC8',0)
           if(NUMCGS.eq.10) NATMCGS(10)=GTRMI(COMLYN,COMLEN,'NUC9',0)
           IF(NUMCGS.GT.10) CALL WRNDIE(-1,'<QMDFN2>','NUMC should be less than 10.')
        END IF
     END IF
  ELSE
     NOQMIM=.FALSE.
  END IF
  !**********************************************************************
  !
  !     Determine the number of quantum mechanical atoms.
  !
  NSLCT = 0
  Do I = 1,NATOM
     IF (ISLCT(I) .EQ. 1) THEN
        NSLCT = NSLCT + 1
        QMINB(NSLCT) = I
     END IF
  End do
  IF (NSLCT .LE. 0) CALL WRNDIE(0,'<QMDFN2>', 'No quantum mechanical atoms selected.')
  NATMM = NATOM - NSLCT
  NATQM = NSLCT
  !----------------------------------------------------------------------------
  !     FIND THE QM LINK ATOMS TO BE TREATED BY HYBRID ORBITALS
  !
  MQMLNK = 0
  IF(QMLINK) THEN
     NSLCT = 0
     DO I = 1,NATOM
        IF(LSLCT(I).EQ.1) THEN
           NSLCT = NSLCT+1
           IMQLINK(NSLCT) = I
        ENDIF
     ENDDO
     IF (NSLCT .LE. 0) CALL WRNDIE(0,'<QMDFN2>','There are no GHO QM boundary atoms selected.')
     MQMLNK = NSLCT
     !        USE ATOMIC NUMBERS 91-95 to HOLD QM LINK ATOMS
     ILINK = 90
     CALL WRNDIE(2,'<QMDFN2>','Atomic number 91-95 are used for GHO atoms')
     IF(MQMLNK.GT.5) CALL WRNDIE(-5,'<QMDFN2>','Too many QM LINK atoms. Reduce to < 5')
  ENDIF
  !----------------------------------------------------------------------------
  !
  !     Fill the QATLAB array for the MOPAC routines.
  !
  Do I = 1,NATOM
     IF (ISLCT(I) .EQ. 1) THEN
        ELE = ATCT(IAC(I))
        !
        ! Assign atomic number based on masses and do a name check just
        ! to be sure. - DCC
        !
        IF(PRNLEV.GT.4) WRITE(6,*) 'QMSET ',ELE,AMASS(I),I,IAC(I)
        CALL FINDEL(ATCT(IAC(I)),AMASS(I),I,ELE,ZNUM,.TRUE.)
        !
        do J = 0,100
           IF (ELE(1:2) .EQ. ELEMNT(J)) THEN
              QATLAB(I) = J
              !----------------------------------------------------------------------------
              !
              ! Assign QM-link atom psuedo atomic number (MJF)
              !
              IF(QMLINK) THEN
                 IF(LSLCT(I).EQ.1) THEN
                    ILINK = ILINK+1
                    QATLAB(I) = ILINK
                 ENDIF
              ENDIF
              IF(PRNLEV.GT.4) WRITE(OUTU,'(A,A2,5X,A2,5X,I5)')'QMSET>',ELE,ELEMNT(J),QATLAB(I) 
              !----------------------------------------------------------------------------
              GOTO 34 
           ENDIF
        end do
        !
        CALL WRNDIE(-3,'<QMDFN2>','QATLAB undefined.')
        QATLAB(I) = 0
        !
34      CONTINUE
     ELSE
        QATLAB(I) = -1
     ENDIF
     !
  End do
  !----------------------------------------------------------------------------
  !     Determine whether there are any link atoms.
  !
  NSLCT = 0
  Do I = 1,NATOM
     IF (QATLAB(I) .EQ. 0) NSLCT = NSLCT + 1
  End do
  !
  NATLNK = NSLCT
  !
  !     Write out some information and options requested.
  !
  IF(PRNLEV.GE.2) THEN
     WRITE (OUTU,'(/,1X,A,/)') &
          ' QMDEFN> Some atoms will be treated quantum mechanically.'
     WRITE (OUTU,'(4(8X,A,I5,/),/)') &
          ' The number of quantum   mechanical atoms = ',NATQM, &
          ' The number of molecular mechanical atoms = ',NATMM, &
          ' The number of QM/MM link atoms           = ',NATLNK, &
          ' The number of GHO boundary atoms         = ',MQMLNK
  END IF
  !
  ! Copy the remaining command line characters to the KEYWRD variable.
  ! Add the PRECISE keyword unless "NOPRecise" is explicitly requested.
  CALL TRIMA(COMLYN,COMLEN)
  IF(COMLEN.EQ.0) THEN
     KEYWRD=' PRECISE '
  ELSE IF(COMLEN.LE.72) THEN
     IF(INDXA(COMLYN,COMLEN,'NOPR').GT.0) THEN
        KEYWRD = COMLYN(1:COMLEN)
     ELSE
        KEYWRD = COMLYN(1:COMLEN)//' PRECISE'
     ENDIF
  ELSE
     CALL WRNDIE(-1,'<QMDFN2>','Too many characters in QM define, TRUNCATED.') 
     KEYWRD = COMLYN(1:80)
  ENDIF
  !
  !     Fill the quantum mechanical data structures.
  !
  ! BIOVIA Code Start : Enhancement for QM/MM
  QMNULL=(INDXA(COMLYN,COMLEN,'NULH').GT.0)
  IF (.NOT.QMNULL) THEN
     CALL SCFINI(COMLYN,COMLEN)
     CALL MNMOLDAT(COMLYN,COMLEN,NATOM)
     CALL GESDEN(.TRUE.)
  ELSE
     NELECS = 0
     NORBS = 0
     WRITE(OUTU,'(/,2X,A,/)') &
                'QMDFN2> NULH option: no QM calc., QMELec energy is zero'
     IF (QMLINK) THEN
        CALL WRNDIE(-5,'QMDFN2>','NULL option and link atoms do not work.')
     ENDIF
  ENDIF
  ! BIOVIA Code End
  !
  !      DEFINE THE HYBRID ORBITAL BASIS FUNCTIONS FOR THE QM LINK ATOMS
  !
  IF (QMLINK) CALL HBFDEF
  !
  !     Modify the molecular mechanical data structures and generate the
  !     QM/MM exclusion lists.
  !
  CALL SDFQM
  !
  !----------------------------------------------------------------------------
  ! namkh 08/08/04
  ! QM/MM-Ewald
  IF(LQMEWD) THEN
     ! First handle QMPERT case
     IF(QMPERT) THEN
        IF(QCGSC) THEN
           DO I=1,NUMCGS
              NSLCT=NATMCGS(I)
              CGSCLE(I)=CG(NSLCT)
              CG(NSLCT)=RLAMB0*CG(NSLCT)
           END DO
        END IF
     END IF
     !
     NSLCT = 0
     CHAGTOT=zero
     DO I = 1, NATOM
        IF(QATLAB(I) .GE. 0) THEN
           NSLCT = NSLCT + 1
           MMINB(I) = I
        ELSE
           MMINB(I) = -I
           CHAGTOT=CHAGTOT+CG(I)
        END IF
     END DO
     !   For GHO atoms
     IF(QMLINK) THEN
        DO I=1,MQMLNK
           CHAGTOT=CHAGTOT+QMATMQM(I)
        END DO
     END IF
     !
     IF(.NOT.NOQMIM) THEN
        IF(QMPERT) THEN
           CHAGTOT=CHAGTOT+RLAMB0*CHARGE
        ELSE
           CHAGTOT=CHAGTOT+CHARGE
        END IF
     END IF

     ! Check the Total charge of system..
     IF(ABS(CHAGTOT).GE.(ZERO+0.001D0)) THEN
        IF(PRNLEV.GE.2) WRITE(outu,*) '<QMDEFN> The total charge of system is ',CHAGTOT
        CALL WRNDIE(1,'<QMDEFN>','Check Total charge of system.')

        ! Background charge correction scheme. The equations refer
        ! Figuerirido et al. J. Chem. Phys. 1995 v103 p6133-6142
        ! This term will be included in nbonds/ewaldf.src 
        ! Same equation is used in PME with PMEPLSMA
        IF(PRNLEV.GE.2) WRITE(outu,*) '<QMDEFN> Background charge correction scheme will be used.'
     END IF

     IF(QMPERT) THEN
        !
        !
        !
        ! Note for Back ground charge correction:
        ! -namkh
        ! Here, in QM/MM-Ewald, the correction term is added only using
        ! QMPERT, otherwise the correction will be done in MM part.
        ! The equations used are based on background charge correction
        ! from Figuerirido et al. J. Chem. Phys. 1995 v103 p6133-6142,
        ! which is
        !
        ! E_background = half * q_tot^2 * (- pi / kappa*2 / volume)
        !
        ! Thus, in QM/MM-Ewald for QMPERT, the energy term should be
        ! added are
        !
        ! delE_background = E_1_background - E_2_background
        !                 = half*(q_tot_1^2-q_tot_2^2)*(- pi / kappa*2 / volume)
        !
        ! Since the background charge correction only affects in total
        ! electrostatic energy and virial, this will return same results
        ! as far as user uses "PMEPLSMA".
        IF(.NOT.NOQMIM) THEN
           BCKCHG(1) = CHAGTOT+RLAMB0*CHARGE
           BCKCHG(2) = CHAGTOT+RLAMB1*CHARGE
           BCKCHG(3) = CHAGTOT+RLAMB2*CHARGE
        ELSE
           BCKCHG(1) = CHAGTOT
           BCKCHG(2) = CHAGTOT
           BCKCHG(3) = CHAGTOT
        END IF
        BCKCHG(1:3) = BCKCHG(1:3)*BCKCHG(1:3)
     END IF
     !
     ! now do the initial setup for QM/MM-Ewald
     CALL SETUP_QMEWD
  END IF
  !----------------------------------------------------------------------------
  !
  !     Setup the analytic derivative data structures.
  ! namkh
  !     initialize NZTYPE_anal, MTYPE_anal
  !     NZTYPE_anal(1:100) = 0
  !     MTYPE_anal(1:10) = -1
  ! end
  !
  !    . LTYPE_anal = NUMBER OF TYPES OF REAL ATOM PRESENT
  !    . MTYPE_anal = TYPES OF REAL ATOMS PRESENT
  LTYPE_anal = 0
  loopII: Do I = 1,NATQM
     if (NAT(I) .GT. -1) then
        loopJJ: do J = 1,LTYPE_anal
           if(NAT(I) .EQ. MTYPE_anal(J)) cycle loopII  
        end do loopJJ
        LTYPE_anal            = LTYPE_anal + 1
        MTYPE_anal(LTYPE_anal)= NAT(I)
        NZTYPE_anal(NAT(I))   = LTYPE_anal
        J = LTYPE_anal
     end if
  End do loopII

  ! BIOVIA Code Start : Bug fix
  IF (.NOT.QMNULL) CALL MNSETUPG
  ! BIOVIA Code End
  !
  ! Process for REMove or EXGR
  !
  IF(QGMREM.OR.QGMEXG) CALL QUAJGM(ISLCT)

  ! Just make sure charges on QM atoms are removed.
  NSLCT = 0
  DO I=1,NATOM
     IF(QATLAB(I).GE.0) THEN
        NSLCT=NSLCT+1
        CG(I)=ZERO
     END IF
  END DO
  !
  !
  ! UHF-GHO density damping factor ... PJ 12/2002
  !
  DDAMP = (INDX(COMLYN, COMLEN, 'DAMP',4) .GT. 0)
  IF (.NOT. (UHF .AND. QMLINK)) THEN
     IF (DDAMP) CALL WRNDIE(-5, '<QMDFN2>','DAMP is only implmented for UHF-GHO.')
  ELSE
     IF (DDAMP) THEN
        IF(PRNLEV.GE.2) WRITE(OUTU,*) 'QMSET > DAMP: density damping for UHF-GHO calculation:'
        PALPHA = GTRMF(COMLYN,COMLEN,'DAMP',ZERO)
        IF (PALPHA .LT. ZERO .OR. PALPHA .GT. ONE) THEN
           CALL WRNDIE(-5, '<QMDFN2>','DAMP: density damping factor should be between 0 and 1.')
        ELSE
           IF(PRNLEV.GE.2) WRITE(OUTU,99) PALPHA
        ENDIF
     ENDIF
  ENDIF
99 FORMAT(9X, 'The density damping factor is : ', f8.3)
  !
  !
  RETURN
END SUBROUTINE QMDFN2
!
SUBROUTINE QUAJGM(ISLCT)
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use number
  use gamess_fcm
  use stream
  use chutil,only:atomid
  !
  ! for parallel run
#if KEY_PARALLEL==1
  use parallel     
#endif
  use chutil,only:getres
  implicit none
  !
  INTEGER ISLCT(*)
  !
  CHARACTER(len=8) SID,RID,REN,AC
  INTEGER I,I1,I2,N,J,IS,IQ
  logical:: qprint

  qprint=.false.
  if(PRNLEV.GE.2) qprint=.true.
  !
  DO I=1, NATOM
     IGMSEL(I)=ISLCT(I)
     IF (ATYPE(I)(1:2).EQ.'QQ') IGMSEL(I)=2
  ENDDO
  !
  !     Check if link atom is connected to any of its neighbors. If
  !     yes then that atom will not be included in QM/MM interaction.
  !     This is sometimes necessary to prevent oposite charge collision,
  !     since QM cannot prevent this to happen.
  !
  !
  DO I=1,NBOND
     I1=IB(I)
     I2=JB(I)
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
  IF(qprint) THEN
     WRITE(OUTU,118)
     WRITE(OUTU,120) 'Classical atoms excluded from the QM calculation'
  END IF
118 FORMAT('------------------------------------------------')
120 FORMAT('GAMINT: ',A,':')
122 FORMAT(10X,I5,4(1X,A))
124 FORMAT(10X,'NONE.')
  N=0
  DO I=1,NATOM
     IF(IGMSEL(I).EQ.5) THEN
        CALL ATOMID(I,SID,RID,REN,AC)
        IF(qprint) WRITE(OUTU,122) I,SID(1:idleng),RID(1:idleng),REN(1:idleng),AC(1:idleng)
        N=N+1
     ENDIF
  ENDDO
  IF(qprint) THEN
     IF(N.EQ.0) WRITE(OUTU,124)
     WRITE(OUTU,120) 'Quantum mechanical atoms'
  END IF
  N=0
  DO I=1,NATOM
     IF(IGMSEL(I).EQ.1) THEN
        CALL ATOMID(I,SID,RID,REN,AC)
        IF(qprint) WRITE(OUTU,122) I,SID(1:idleng),RID(1:idleng),REN(1:idleng),AC(1:idleng)
        N=N+1
     ENDIF
  ENDDO
  IF(qprint) THEN
     IF(N.EQ.0) WRITE(OUTU,124)
     WRITE(OUTU,120) 'Quantum mechanical link atoms'
  END IF
  N=0
  DO I=1,NATOM
     IF(IGMSEL(I).EQ.2) THEN
        CALL ATOMID(I,SID,RID,REN,AC)
        IF(qprint) WRITE(OUTU,122) I,SID(1:idleng),RID(1:idleng),REN(1:idleng),AC(1:idleng)
        N=N+1
     ENDIF
  ENDDO
  IF(qprint) THEN
     IF(N.EQ.0) WRITE(OUTU,124)
     WRITE(OUTU,118)
  END IF
  !
  ! Zero charges on quantum atoms to remove from MM term.
  IF(QGMREM) THEN
     DO I=1, NATOM
        IF((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2))CG(I)=ZERO
     ENDDO
  ENDIF
  !
  !----------------------------------------------------------------------------
  RETURN
END SUBROUTINE QUAJGM
!
!
SUBROUTINE SCFINI(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     Parse the SCF related commands and set up the SCF data structures.
  !
  use chm_kinds
  use number
  use dimens_fcm
  use quantm
  use scfblk
  use am1parm
  use stream
  use string
  !
  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  !
  real(chm_real),parameter :: TENM9 = 1.D-9, TENM7 = 1.D-7, TENM12 = 1.D-12
  !
  !     Do some initialisation.
  !
  NEWDG  = .FALSE.
  QFIRST = .TRUE.
  IFILL  = 0
  ITRMAX = 200
  CAMKIN = .FALSE.
  OKPULY = .FALSE.
  UHF    = .FALSE.
  BSHIFT = ZERO
  SCFCRT = TENM7
  ALLCON = .FALSE.
  OKNEWD = .FALSE.
  CI     = .FALSE.
  NMOS   = 0
  NCIS   = 0
  !
  PL     = ONE
  PLB    = ONE
  PLTEST = TENM4
  !
  NCLOSE = 0
  NOPEN  = 0
  NALPHA = 0
  NBETA  = 0
  !
  !     Parse SCF commands.
  !
  IFILL  = GTRMI(COMLYN,COMLEN,'IFIL',0)
  ITRMAX = GTRMI(COMLYN,COMLEN,'ITRM',200)
  !
  CAMKIN = (INDXA(COMLYN,COMLEN,'KING') .GT. 0) .OR. (INDXA(COMLYN,COMLEN,'CAMP') .GT. 0)
  OKPULY = INDXA(COMLYN,COMLEN,'PULA') .GT. 0
  UHF    = INDXA(COMLYN,COMLEN,'UHF') .GT. 0
  !
  BSHIFT = - GTRMF(COMLYN,COMLEN,'SHIF',ZERO)
  !DCC#      SCFCRT = MAX(GTRMF(COMLYN,COMLEN,'SCFC',TENM7),TENM9)
  !DCC-
  SCFCRT = MAX(GTRMF(COMLYN,COMLEN,'SCFC',TENM7),TENM12)
  !DCC+
  !
  ALLCON = (BSHIFT .NE. ZERO) .OR. CAMKIN .OR. OKPULY
  OKNEWD = ABS(BSHIFT) .LT. TENM3
  !
  !     Parse CI commands.
  !
  CI = INDXA(COMLYN,COMLEN,'C.I.') .GT. 0
  !
  LROOT = 1
  IF (INDEX(COMLYN,'EXCI') .NE. 0) LROOT = 2
  LROOT = GTRMI(COMLYN,COMLEN,'ROOT',LROOT)
  !
  NMOS  = GTRMI(COMLYN,COMLEN,'NMOS',0)
  NCIS  = GTRMI(COMLYN,COMLEN,'MICR',0)
  !
  IF (NMOS.EQ.0) NMOS = NOPEN - NCLOSE
  IF (NCIS.EQ.0) THEN
     IF (INDXA(COMLYN,COMLEN,'TRIPLET') .GT. 0) THEN
        NCIS = 1
     ELSE IF (INDXA(COMLYN,COMLEN,'QUARTET') .GT. 0) THEN
        NCIS = 1
     ELSE IF (INDXA(COMLYN,COMLEN,'QUINTET') .GT. 0) THEN
        NCIS = 2
     ELSE IF (INDXA(COMLYN,COMLEN,'SEXTET')  .GT. 0) THEN
        NCIS = 2
     ENDIF
  ENDIF
  !
  RETURN
END SUBROUTINE SCFINI
!
!
SUBROUTINE MNMOLDAT(COMLYN,COMLEN,NATOM)
  !-----------------------------------------------------------------------
  !     Fill the data structures necessary for a semi-empirical
  !     quantum mechanical calculation on the system of interest.
  !
  use chm_kinds
  use number, only : ZERO,HALF,ONE,FIVE,TENM4
  use consta, only : EV_TO_KCAL
  use dimens_fcm
  use stream
  use string
  use sizes
  use quantm
  use scfblk
  use am1parm
  use qmlinkm
#if KEY_PARALLEL==1
  use parallel        
#endif
  implicit none
  CHARACTER(len=*) COMLYN
  INTEGER       COMLEN, NATOM
  !
  INTEGER  I, IATOM, IATQM, IELEC, IFIRST, ILAST, ILEVEL, IORBS
  INTEGER  J, N1, N2, N2EL, NDORBS, NHEAVY, NI, NLIGHT, II, JJ
  LOGICAL  AM1, PM3, BIRAD, EXCI, MNDO, OPEN, TRIP
  real(chm_real)   EAT, ELECS
  !
  real(chm_real), parameter :: SFACT = EV_TO_KCAL ! 23.061D0
  ! DTM Added definition
  INTEGER  PUNIT
  LOGICAL  QPAR
  ! END DTM
  !
  !     Determine some logical variables.
  !
  AM1   = INDEX(COMLYN,'AM1') .NE. 0
  PM3   = INDEX(COMLYN,'PM3') .NE. 0
  MNDO  = INDEX(COMLYN,'MNDO') .NE. 0
  ! Did the user request that we disable cutoffs?
  QMCUTF= INDEX(COMLYN,'NOCU') .LE. 0
  !
  IF (PM3) THEN
     !
     !       Transfer PM3 parameters to the AM1 arrays.
     !
     Do I = 0,100
        ALFA(I)   = ALPP(I)
        EISOL(I) = EISOLP(I)
        BETAS(I) = BETASP(I)
        BETAP(I) = BETAPP(I)
        BETAD(I) = BETADP(I)
        ZS(I)    = ZSP(I)
        ZP(I)    = ZPP(I)
        ZD(I)    = ZDP(I)
        DD(I)    = DDP(I)
        QQ(I)    = QQP(I)
        bdd(I,1)    = AMP(I)
        bdd(I,2)    = ADP(I)
        bdd(I,3)    = AQP(I)
        USS(I)   = USSP(I)
        UPP(I)   = UPPP(I)
        UDD(I)   = UDDP(I)
        GSS(I)   = GSSP(I)
        GPP(I)   = GPPP(I)
        GSP(I)   = GSPP(I)
        GP2(I)   = GP2P(I)
        HSP(I)   = HSPP(I)
        VS(I)    = VSP(I)
        VP(I)    = VPP(I)

        FN1(I,1:10) = FN1P(I,1:10)
        FN2(I,1:10) = FN2P(I,1:10)
        FN3(I,1:10) = FN3P(I,1:10)

        AM1PAR(I) = PM3PAR(I)
     End do
     !
  ELSE IF (MNDO) THEN
     !
     !       Transfer MNDO parameters to the AM1 arrays.
     !
     Do I = 0,100
        ALFA(I)   = ALPM(I)
        EISOL(I) = EISOLM(I)
        BETAS(I) = BETASM(I)
        BETAP(I) = BETAPM(I)
        BETAD(I) = BETADM(I)
        ZS(I)    = ZSM(I)
        ZP(I)    = ZPM(I)
        ZD(I)    = ZDM(I)
        DD(I)    = DDM(I)
        QQ(I)    = QQM(I)
        bdd(I,1)    = AMM(I)
        bdd(I,2)    = ADM(I)
        bdd(I,3)    = AQM(I)
        USS(I)   = USSM(I)
        UPP(I)   = UPPM(I)
        UDD(I)   = UDDM(I)
        GSS(I)   = GSSM(I)
        GPP(I)   = GPPM(I)
        GSP(I)   = GSPM(I)
        GP2(I)   = GP2M(I)
        HSP(I)   = HSPM(I)

        AM1PAR(I) = MNDPAR(I)
     End do
     !
  ELSE IF (AM1) THEN
  ELSE
  ! BIOVIA Code Start : Bug fix
     CALL WRNDIE(-5,'<MOLDAT>','AM1, PM3, MNDO or NULH must be specified.')
  ! BIOVIA Code End
  ENDIF
  ! DTM
  ! Atoms using EXTErnal parameters
  PUNIT = 0
  QPAR = (INDXA(COMLYN,COMLEN,'EXTE').GT.0)
  IF(QPAR) THEN
     PUNIT = GTRMI(COMLYN,COMLEN,'PUNI',0)
     IF(PUNIT.LE.0) CALL WRNDIE(0,'MOLDAT',' provide a parameter file unit')
     IF(PRNLEV.GE.2) WRITE(OUTU,'(A,I4)') ' MOLDAT> Reading parameters from unit ',PUNIT

     !   Only node 0 gets the parameters, then scatters 
#if KEY_PARALLEL==1
     IF (MYNOD.EQ.0) THEN       
#endif

        CALL READPAR(PUNIT)
        CALL CALCPAR

#if KEY_PARALLEL==1
     ENDIF                      
#endif
#if KEY_PARALLEL==1
     CALL SCATPAR               
#endif
  ENDIF
  ! END DTM
  !
  !      DEFINE PARAMETERS (C ONLY) FOR THE QM LINK ATOMS...JG 5/17/97
  !
  IF(PRNLEV.GE.2) WRITE(OUTU,'(A,I5)') ' MOLDAT> MQMLNK =',MQMLNK
  IF(QMLINK) THEN
     II = 90+MQMLNK
     !   Specifically wired for carbon-like GHO boundary atoms
     !   To use other atoms, the following line must be modified.
     CORE(91) = 1.0D+00
     AM1PAR(91) = .TRUE.
     EISOL(91) = EISOL(6)
     EHEAT(91) = EHEAT(6)
     NATORB(91) = NATORB(6)
     J = 91
     DO I = 92,II
        ALFA(I)  = ALFA(J)
        BETAS(I) = BETAS(J)
        BETAP(I) = BETAP(J)
        BETAD(I) = BETAD(J)
        ZS(I)    = ZS(J)
        ZP(I)    = ZP(J)
        ZD(I)    = ZD(J)
        DD(I)    = DD(J)
        QQ(I)    = QQ(J)
        bdd(I,1) = bdd(J,1)
        bdd(I,2) = bdd(J,2)
        bdd(I,3) = bdd(J,3)
        USS(I)   = USS(J)
        UPP(I)   = UPP(J)
        UDD(I)   = UDD(J)
        GSS(I)   = GSS(J)
        GPP(I)   = GPP(J)
        GSP(I)   = GSP(J)
        GP2(I)   = GP2(J)
        HSP(I)   = HSP(J)
        CORE(I)  = 1.0D+00     ! CORE(I) = CORE(J)

        FN1(I,1:10) = FN1(J,1:10)
        FN2(I,1:10) = FN2(J,1:10)
        FN3(I,1:10) = FN3(J,1:10)

        !      JUST TO SHIFT THE TOTAL ENERGY, NO PARTICULAR JUSTIFICATION...JG 7/17/97
        !
        EISOL(I) = EISOL(J)
        EHEAT(I) = EHEAT(J)
        NATORB(I) = NATORB(J)

        AM1PAR(I) = .TRUE.
     ENDDO
  ENDIF
  !
  !     Get the number of electrons and orbitals.
  !
  !xl.. partial charge possible
  !xl   CHARGE = GTRMI(COMLYN,COMLEN,'CHAR',0)
  CHARGE = GTRMF(COMLYN,COMLEN,'CHAR',ZERO)
  !     . Get the FIXMM parameters.
  ALPMM  = GTRMF(COMLYN,COMLEN,'ALPM',FIVE)
  RHO0MM = GTRMF(COMLYN,COMLEN,'RHO0',ZERO)
  SFT123 = GTRMF(COMLYN,COMLEN,'SFT1',ONE)
  !
  !     Perform some initialisation.
  !
  ATHEAT = ZERO
  EAT    = ZERO
  ELECS  = - CHARGE
  IFIRST = 1
  ILAST  = 0
  NDORBS = 0
  NHEAVY = 0
  IATQM  = 0
  !
  Do IATOM = 1,NATOM
     IF (QATLAB(IATOM) .GE. 0 .AND. QATLAB(IATOM).LT.91) THEN
        IATQM = IATQM + 1
        NI    = QATLAB(IATOM)
        IORBS = NATORB(NI)
        ILAST = IFIRST + IORBS - 1
        !
        NAT(IATQM)    = NI
        N1ST(IATQM) = IFIRST
        NMIDLE(IATQM) = ILAST
        NLAST(IATQM)  = ILAST
        !
        IF (IORBS .EQ. 9) THEN
           NDORBS        = NDORBS + 5
           NMIDLE(IATQM) = IFIRST + 3
        ENDIF
        !
        ATHEAT = ATHEAT + EHEAT(NI)
        EAT    = EAT + EISOL(NI)
        ELECS  = ELECS + CORE(NI)
        !
        USPD(IFIRST) = USS(NI)
        IF (IFIRST .LT. ILAST) THEN
           NHEAVY = NHEAVY + 1
           N1 = IFIRST + 1
           N2 = IFIRST + 3
           USPD(N1:N2) = UPP(NI)
           IF (N2 .LT. ILAST) USPD(N2+1:ILAST) = UDD(NI)
        ENDIF
        IFIRST = ILAST + 1
        IF (PRNLEV.GE.4) WRITE (OUTU,"(A,7I6)") "ATOM DATA> ", IATOM, IATQM, NAT(IATQM), &
             N1ST(IATQM), NMIDLE(IATQM), NLAST(IATQM),QATLAB(IATQM)
     ENDIF
  End do
  !
  !      ADD QM LINK ATOMS...JG 5/17/97
  !
  IF(QMLINK) THEN
     DO IATOM = 1,NATOM
        IF (QATLAB(IATOM).GE.91.AND.QATLAB(IATOM).LT.96) THEN
           IATQM = IATQM + 1
           NI    = QATLAB(IATOM)
           IORBS = NATORB(NI)
           ILAST = IFIRST + IORBS - 1
           !
           NAT(IATQM)    = NI
           N1ST(IATQM) = IFIRST
           NMIDLE(IATQM) = ILAST
           NLAST(IATQM)  = ILAST
           !
           IF (IORBS .EQ. 9) THEN
              NDORBS = NDORBS + 5
              NMIDLE(IATQM) = IFIRST + 3
           ENDIF
           !
           ATHEAT = ATHEAT + EHEAT(NI)
           EAT    = EAT + EISOL(NI)
           ELECS  = ELECS + CORE(NI)
           !
           USPD(IFIRST) = USS(NI)
           IF (IFIRST .LT. ILAST) THEN
              NHEAVY = NHEAVY + 1
              N1 = IFIRST + 1
              N2 = IFIRST + 3
              USPD(N1:N2) = UPP(NI)
              IF (N2 .LT. ILAST) USPD(N2+1:ILAST)=UDD(NI)
           ENDIF
           IFIRST = ILAST + 1
           IF (PRNLEV.GE.4) WRITE (OUTU,"(A,7I6)") "LINK DATA> ", IATOM, IATQM, NAT(IATQM), &
                N1ST(IATQM), NMIDLE(IATQM), NLAST(IATQM), QATLAB(IATQM)
        ENDIF
     ENDDO
  ENDIF
  !
  ATHEAT = ATHEAT - SFACT * EAT
  NORBS  = NLAST(NATQM)
  !
  !     Check that parameters are available for each atom in the system.
  Do IATQM = 1,NATQM
     NI = NAT(IATQM)
     IF (.NOT.(AM1PAR(NI))) THEN
        IF(PRNLEV.GE.2) WRITE (OUTU,'(A,A2,A)') ' MOLDAT> Parameters unavailable for ',ELEMNT(NI),'.'
        CALL WRNDIE(-5,'<MOLDAT>','Parameters unavailable.')
     ENDIF
  End do
  !
  !     Check some program dimensions.
  !
  IF (NORBS .GT. MAXORB) CALL WRNDIE(-5,'<MOLDAT>','Maximum number of orbitals exceeded.')
  !
  NLIGHT = NATQM - NHEAVY
  !
  N2EL = 50*NHEAVY*(NHEAVY-1) + 10*NHEAVY*NLIGHT + (NLIGHT*(NLIGHT-1))/2
  IF (N2EL.GT.N2ELEC) CALL WRNDIE(-5,'<MOLDAT>', 'Maximum number of two-electron integrals exceeded.')
  !
  !     Calculate the number of occupied levels.
  !
  EXCI  = INDEX(KEYWRD,'EXCITED') .NE. 0
  TRIP  = INDEX(KEYWRD,'TRIPLET') .NE. 0
  BIRAD = EXCI .OR. INDEX(KEYWRD,'BIRAD') .NE. 0
  IF (INDEX(KEYWRD,'C.I.').NE.0 .AND. UHF) CALL WRNDIE (0,'<MOLDAT>','C.I. not allowed with UHF.')
  !
  !     Determine the number of electrons in each shell.
  !
  NALPHA = 0
  NBETA  = 0
  NCLOSE = 0
  !xl...NELECS = INT(ELECS + HALF)
  NELECS = INT(ELECS)
  I=NELECS/2
  I=I*2
  FRACT=ELECS-I
  !CC      FRACT=ELECS-(2*(NELECS/2))
  !CC      write(*,'(''fract = '',2f10.3,i5)')fract,elecs,nelecs
  IF(PRNLEV .GT. 4) write(OUTU,'(a,2f10.3,i5)') 'fract',fract,elecs,nelecs
  IF(FRACT .GT. 0.0D0) THEN
     ! UHF calculations MG 10/02
     !         NELECS=NELECS+1
     IFRAC=1
  ELSE
     FRACT=0.0D0
     IFRAC=0
  ENDIF
  !xl...
  NOPEN  = 0
  !
  IF (UHF) THEN
     FRACT = ONE
     NBETA = NELECS / 2
     IF (TRIP) THEN
        IF (2 * NBETA .NE. NELECS) THEN
           CALL WRNDIE(0,'<MOLDAT>','Triplet specified with odd number of electrons.')
        ELSE
           IF(PRNLEV.GE.2) WRITE (OUTU,'(A)') ' MOLDAT> Triplet state calculation.'
           NBETA = NBETA - 1
        ENDIF
     ENDIF
     NALPHA = NELECS - NBETA
     IF(PRNLEV.GE.2) THEN
        WRITE (OUTU,'(A)') ' MOLDAT> UHF calculation.'
        WRITE (OUTU,'(8X,A,I3,A,I3)') ' Number of alpha electrons = ',NALPHA, &
             ' and number of beta electrons = ',NBETA
     END IF
     !
     !       Determine open and closed shells.
     !
  ELSE
     IELEC  = 0
     ILEVEL = 0
     OPEN   = .FALSE.
     !
     IF (TRIP .OR. EXCI .OR. BIRAD) THEN
        IF (2*(NELECS/2).NE.NELECS) CALL WRNDIE(0,'<MOLDAT>','System specified has an odd number of electrons.')
        IF(PRNLEV.GE.2) THEN 
           IF (BIRAD) WRITE(OUTU,'(A)') ' MOLDAT> Biradical calculation.'
           IF (TRIP ) WRITE(OUTU,'(A)') ' MOLDAT> Triplet state calculation.'
           IF (EXCI ) WRITE(OUTU,'(A)') ' MOLDAT> Excited state calculation.'
        END IF
        IELEC  = 2
        ILEVEL = 2
     ELSE
        IF (2*(NELECS/2).NE.NELECS) THEN
           IELEC  = 1
           ILEVEL = 1
        ELSE
           IF (IFRAC.GT.0) then
              IELEC  = 2
              ILEVEL = 1
           ENDIF
        ENDIF
     ENDIF
     !
     IF (INDEX(KEYWRD,'QUART') .NE. 0) THEN
        IF(PRNLEV.GE.2) WRITE(OUTU,'(A)') ' MOLDAT> Quartet state calculation.'
        IELEC  = 3
        ILEVEL = 3
     ENDIF
     IF (INDEX(KEYWRD,'QUINT') .NE. 0) THEN
        IF(PRNLEV.GE.2) WRITE(OUTU,'(A)') ' MOLDAT> Quintet state calculation.'
        IELEC  = 4
        ILEVEL = 4
     ENDIF
     IF (INDEX(KEYWRD,'SEXT') .NE. 0) THEN
        IF(PRNLEV.GE.2) WRITE(OUTU,'(A)') ' MOLDAT> Sextet state calculation.'
        IELEC  = 5
        ILEVEL = 5
     ENDIF
     !
     !
     NCLOSE = NELECS / 2
     NOPEN  = NELECS - 2 * NCLOSE
     IF (IELEC .NE. 0) THEN
        IF ((2*(NELECS/2).EQ.NELECS).NEQV.(2*(IELEC/2).EQ.IELEC)) &
             CALL WRNDIE(0,'<MOLDAT>','Impossible number of open shell electrons.')
        !
        IF(PRNLEV.GT.4) WRITE(6,*) 'MOLDAT ',IELEC,ILEVEL,FRACT
        !
        NCLOSE = NCLOSE - (IELEC / 2)
        NOPEN  = ILEVEL
        !xl       FRACT  = IELEC
        FRACT  = FRACT / ILEVEL
     ENDIF
     !
     IF(PRNLEV.GE.2) THEN
        WRITE (OUTU,'(A,I3,A)') ' MOLDAT> There are ',NCLOSE,' doubly occupied levels.' 
        IF (NOPEN .NE. 0 .AND. ABS(FRACT - ONE) .LT. TENM4) &
             WRITE (OUTU,'(A,I3,A)') ' MOLDAT> There are ',NOPEN,' singly occupied levels.'
        IF (NOPEN .NE. 0 .AND. ABS(FRACT - ONE) .GT. TENM4) &
             WRITE (OUTU,'(A,I3,A,F6.3,A)') ' MOLDAT> There are ',NOPEN,' levels with an occupancy of ',FRACT,'.'
     END IF
     NOPEN = NOPEN + NCLOSE
  ENDIF
  !
  RETURN
END SUBROUTINE MNMOLDAT
!
SUBROUTINE GESDEN(QGUESFST)
  !-----------------------------------------------------------------------
  !     Space is allocated here for the SCF/CI density matrices and
  !     orbitals and an initial density matrix is generated.
  !
  use memory
  use scfblk
  use dimens_fcm
  use sizes
  use quantm
  use memory
  use number, only : zero
  use chm_kinds
  implicit none
  !
  !     INTEGER NORBS
  !
  real(chm_real),allocatable,dimension(:) :: PDIAG
  LOGICAL QGUESFST
  INTEGER LENMAT, LENORB, LENPAB, LINEAR
  !
  !     Define the array lengths.
  !
  LINEAR = (NORBS * (NORBS + 1)) / 2
  !
  LENMAT = NORBS * NORBS    
  LENORB = NORBS            
  LENPAB = LINEAR           
  !
  !     Allocate bunch of arrays:
  !     H_mtraix/
  !     CALPHA/CBETA/EIGSA/EIGSB/FA/FB/PDENS/PDENSA/PDENSB/PAOLD/PBOLD
  !     H1PERT/H2PERT/H0GAS/PGAS/PENZYM

  IF (QGUESFST) THEN
     call setup_qm_arrays(LINEAR,LENMAT,LENORB,LENPAB,QMPERT,QDECOM) 
     !
     !----------------------------------
     IF(QMPERT .or. QDECOM) LINQLK = LINEAR
  END IF
  !----------------------------------
  !     Get the density matrix.
  !
  call chmalloc('qmset.src','GESDEN','PDIAG',LENORB,crl=PDIAG)
  CALL GESDN2(LINEAR,UHF,PDENS,PDENSA,PDENSB,PAOLD,PBOLD,PDIAG)
  call chmdealloc('qmset.src','GESDEN','PDIAG',size(PDIAG),crl=PDIAG)
  !
  !----------------------------------
  IF(QDECOM) THEN
     PGAS(1:LINQLK)  =PAOLD(1:LINQLK)    
     PENZYM(1:LINQLK)=PAOLD(1:LINQLK)    
  ENDIF
  !----------------------------------
  RETURN
END SUBROUTINE GESDEN
!
SUBROUTINE GESDN2(LINEAR, UHF, P, PA, PB, PAOLD, PBOLD, PDIAG)
  !-----------------------------------------------------------------------
  !     Guess an initial density matrix.
  !
  use chm_kinds
  use number
  use dimens_fcm
  !
  use quantm
  use am1parm
  implicit none
  !
  INTEGER LINEAR
  LOGICAL UHF
  real(chm_real)  P(*), PA(*), PAOLD(*), PB(*), PBOLD(*), PDIAG(*)
  !
  INTEGER I, IFIRST, ILAST, IMIDLE, IORB, IPT, J, NA1EL, NB1EL, NI
  real(chm_real)  FACT1, FACT2, RANDOM, W, W1, W2
  real(chm_real)  SUMP,SUMPA,SUMPB
  !
  !
  !
  FACT1 = CHARGE / NORBS
  IPT   = 0
  !
  DO I = 1,NATQM
     NI   = NAT(I)
     IORB = NLAST(I) - N1ST(I) + 1
     IF (.NOT.(IORB .EQ. 0)) THEN
        FACT2  = ONE / IORB
        W      = CORE(NI) * FACT2 - FACT1
        IFIRST = N1ST(I)
        IMIDLE = NMIDLE(I)
        ILAST  = NLAST(I)
        DO J = IFIRST,IMIDLE
           IPT = IPT + 1
           PDIAG(IPT) = W
        end do
        DO J = (IMIDLE + 1),ILAST
           IPT = IPT + 1
           PDIAG(IPT) = ZERO
        end do
     ENDIF
  end do
  !
  NA1EL = NALPHA + NOPEN
  NB1EL = NBETA  + NOPEN
  !
  P(1:LINEAR)  = ZERO
  PA(1:LINEAR) = ZERO
  PB(1:LINEAR) = ZERO
  !
  W1 = NA1EL
  W1 = W1 / (NA1EL + NB1EL + TENM6)
  W2 = ONE - W1
  IF (W1 .LT. TENM6) W1 = HALF
  IF (W2 .LT. TENM6) W2 = HALF
  RANDOM = ONE
  IF (UHF .AND. (NA1EL .EQ. NB1EL)) RANDOM = 1.1D0
  do I = 1,NORBS
     J = (I*(I+1))/2
     P(J)   = PDIAG(I)
     PA(J)  = P(J) * W1 * RANDOM
     RANDOM = ONE / RANDOM
     PB(J)  = P(J) * W2 * RANDOM
  end do
  !
  SUMP =ZERO
  SUMPA=ZERO
  SUMPB=ZERO
  do I = 1,LINEAR
     PAOLD(I) = PA(I)
     PBOLD(I) = PB(I)

     SUMP =SUMP +P(I)
     SUMPA=SUMPA+PA(I)
     SUMPB=SUMPB+PB(I)
  end do
  SUMPSV =SUMP
  SUMPSVA=SUMPA
  SUMPSVB=SUMPB
  !
  RETURN
END SUBROUTINE GESDN2
!
SUBROUTINE SDFQM
  !-----------------------------------------------------------------------
  !     Modify the SDF if quantum mechanical atoms have been defined.
  !
  use chm_kinds
  use dimens_fcm
  use bases_fcm
  use code
  use inbnd
  ! namkh 09/25/04
  use coord
  use image
#if KEY_PBOUND==1 /*pbound*/
  use pbound
#endif /*     (pbound)*/
  !
  use param
  use psf
  use quantm
  use gamess_fcm
  use sizes
  use qmlinkm
  implicit none
  !
  LOGICAL   ISNBDS
  INTEGER   NTBOND,NTTHETA,NTPHI,NTIMPHI
#if KEY_CMAP==1
  INTEGER   NTCRTERM
#endif 
  !
  ! This part should be maked sured to support also IMAGES...
  ! namkh 09/25/04
  NTBOND = NBOND
  NTTHETA= NTHETA
  NTPHI  = NPHI
  NTIMPHI= NIMPHI
#if KEY_CMAP==1
  NTCRTERM = NCRTERM
#endif 
  !
#if KEY_PBOUND==1
  if(.not.qboun) then   
#endif
     IF(NTRANS.GT.0 .AND. NATIM.GT.0) THEN
        NTBOND = NBOND  + NIMBON
        NTTHETA= NTHETA + NIMANG
        NTPHI  = NPHI   + NIMDIH
        NTIMPHI= NIMPHI + NIMIMP
#if KEY_CMAP==1
        NTCRTERM = NCRTERM + NIMCRT
#endif 
     ENDIF
#if KEY_PBOUND==1
  endif                 
#endif
  !
  IF (.NOT.QGMREM) THEN
     CALL SDFQM2(NTBOND,NBOND,IB,JB, &
          NTTHETA,NTHETA,IT,JT,KT, &
          NTPHI,NPHI,IP,JP,KP,LP, &
          NTIMPHI,NIMPHI,IM,JM,KM,LM &
#if KEY_CMAP==1
          ,NTCRTERM,NCRTERM,I1CT,J1CT,K1CT,L1CT,I2CT,J2CT,K2CT,L2CT & 
#endif
          )
     !
#if KEY_PBOUND==1
     if(.not.qboun) then   
#endif
        IF(NTRANS.GT.0 .AND. NATIM.GT.0) THEN
           NIMBON = NTBOND   - NBOND
           NIMANG = NTTHETA  - NTHETA
           NIMDIH = NTPHI    - NPHI
           NIMIMP = NTIMPHI  - NIMPHI
#if KEY_CMAP==1
           NIMCRT = NTCRTERM - NCRTERM
#endif 
        ENDIF
#if KEY_PBOUND==1
     endif                 
#endif
  END IF
  !
  !     If nonbond data structure has been set up, generate QM/MM
  !     nonbond lists here
  ISNBDS=.TRUE.
  IF(NATQM.GT.0) CALL NBNDQM(X,Y,Z)
  !
  IF (.NOT.QGMREM) THEN
     CALL CODES(ICB,ICT,ICP,ICI,NATOM,IMOVE,IAC,NBOND,IB,JB, &
          NTHETA,IT,JT,KT,NPHI,IP,JP,KP,LP,NIMPHI,IM,JM,KM,LM, &
          QDRUDE,NBDRUDE,                             & ! DRUDE
#if KEY_CMAP==1
          ICCT,NCRTERM,I1CT,J1CT,K1CT,L1CT,I2CT,J2CT,K2CT,L2CT, &   
#endif
          .FALSE.,.FALSE.)
     !
#if KEY_PBOUND==1
     if(.not.qboun) then   
#endif
        IF(NTRANS.GT.0 .AND. NATIM.GT.0) THEN
           !           process codes for image energy terms
           !
           CALL CODES(ICB(NBOND+1),ICT(NTHETA+1), &
                ICP(NPHI+1),ICI(NIMPHI+1), &
                NATOM,IMOVE,IAC,NIMBON,IB(NBOND+1),JB(NBOND+1), &
                NIMANG,IT(NTHETA+1),JT(NTHETA+1),KT(NTHETA+1), &
                NIMDIH,IP(NPHI+1),JP(NPHI+1),KP(NPHI+1), &
                LP(NPHI+1),NIMIMP,IM(NIMPHI+1),JM(NIMPHI+1), &
                KM(NIMPHI+1),LM(NIMPHI+1), &
                .FALSE.,0,                             & ! DRUDE
#if KEY_CMAP==1
                ICCT(NCRTERM+1),NIMCRT,I1CT(NCRTERM+1),J1CT(NCRTERM+1), & 
                K1CT(NCRTERM+1),L1CT(NCRTERM+1), &  
                I2CT(NCRTERM+1),J2CT(NCRTERM+1), & 
                K2CT(NCRTERM+1),L2CT(NCRTERM+1), &     
#endif
                .FALSE.,.FALSE.)
           !
        ENDIF
#if KEY_PBOUND==1
     endif                 
#endif
  ENDIF
  !
  RETURN
END SUBROUTINE SDFQM

SUBROUTINE SDFQM2(NBOND,NBOND2,IB,JB, &
     NTHETA,NTHETA2,IT,JT,KT, &
     NPHI,NPHI2,IP,JP,KP,LP, &
     NIMPHI,NIMPHI2,IM,JM,KM,LM &
#if KEY_CMAP==1
     ,NCRTERM,NCRTERM2,I1CT,J1CT,K1CT,L1CT,I2CT,J2CT,K2CT,L2CT & 
#endif
     )
  !-----------------------------------------------------------------------
  !     Remove all QM/QM energy interactions from the molecular
  !     mechanical SDF.
  !
  use chm_kinds
  use dimens_fcm
  !
  use quantm
  use sizes
  use qmlinkm
  !
  implicit none
  INTEGER  NBOND, IB(*), JB(*), NTHETA, IT(*), JT(*), KT(*), &
       NPHI, IP(*), JP(*), KP(*), LP(*), &
       NIMPHI, IM(*), JM(*), KM(*), LM(*), &
       NBOND2,NTHETA2,NPHI2,NIMPHI2  
#if KEY_CMAP==1
  integer NCRTERM, I1CT(*), J1CT(*), K1CT(*), L1CT(*), &
       I2CT(*), J2CT(*), K2CT(*), L2CT(*),NCRTERM2

  INTEGER II1,II2,JJ1,JJ2,KK1,KK2,LL1,LL2
#endif 

  !
  INTEGER I,II,ITERM,JJ,KK,LL,K,ITERM2
  LOGICAL QOK
  !
  ITERM = 0
  ITERM2= 0
  DO I = 1,NBOND
     II = IB(I)
     JJ = JB(I)
     IF(NBOND.LE.NBOND2) THEN
        QOK = (QATLAB(II).LT.0) .OR. (QATLAB(JJ).LT.0)
        !
        !      REMOVE TERM OF QM ATOM and QM LINK ATOM BOND.
        !
        IF (QMLINK) THEN
           do K = 1,MQMLNK
              if((IMQLINK(K).EQ.II.AND.QATLAB(JJ).GT.0) .or. (IMQLINK(K).EQ.JJ.AND.QATLAB(II).GT.0)) QOK=.FALSE.
           end do
        ENDIF
        !
        IF (QOK) THEN
           ITERM = ITERM + 1
           ITERM2= ITERM2+ 1
           IB(ITERM) = II
           JB(ITERM) = JJ
        ENDIF
     ELSE IF(NBOND.GT.NBOND2) THEN
        ITERM = ITERM + 1
        IB(ITERM) = II
        JB(ITERM) = JJ
     END IF
  END DO
  NBOND = ITERM
  NBOND2= ITERM2
  !
  ITERM = 0
  ITERM2= 0
  DO I = 1,NTHETA
     II = IT(I)
     JJ = JT(I)
     KK = KT(I)
     IF(NTHETA.LE.NTHETA2) THEN
        QOK = (QATLAB(II).LT.0) .OR. (QATLAB(JJ).LT.0) .OR. (QATLAB(KK).LT.0)
        !
        !      REMOVE TERM OF QM-QM-QLINK ANGLE.
        IF (QMLINK) THEN
           DO K = 1,MQMLNK
              if((IMQLINK(K).EQ.II.AND.QATLAB(KK).GT.0) .or. (IMQLINK(K).EQ.KK.AND.QATLAB(II).GT.0)) QOK=.FALSE.
           ENDDO
        ENDIF
        !
        IF (QOK) THEN
           ITERM = ITERM + 1
           ITERM2= ITERM2+ 1
           IT(ITERM) = II
           JT(ITERM) = JJ
           KT(ITERM) = KK
        ENDIF
     ELSE IF(NTHETA.GT.NTHETA2) THEN
        ITERM = ITERM + 1
        IT(ITERM) = II
        JT(ITERM) = JJ
        KT(ITERM) = KK
     END IF
  END DO
  NTHETA = ITERM
  NTHETA2= ITERM2
  !
  ITERM = 0
  ITERM2= 0
  DO I = 1,NPHI
     II = IP(I)
     JJ = JP(I)
     KK = KP(I)
     LL = LP(I)
     IF(NPHI.LE.NPHI2) THEN
        QOK = (QATLAB(II).LT.0) .OR. (QATLAB(JJ).LT.0) .OR. (QATLAB(KK).LT.0) .OR. (QATLAB(LL).LT.0)
        !
        !      REMOVE TERM OF QM-QM-QM-QLINK DIHEDRAL.
        !
        IF (QMLINK) THEN
           DO K = 1,MQMLNK
              if((IMQLINK(K).EQ.II.AND.QATLAB(LL).GT.0) .or. (IMQLINK(K).EQ.LL.AND.QATLAB(II).GT.0)) QOK=.FALSE.
           ENDDO
        ENDIF
        !
        IF (QOK) THEN
           ITERM = ITERM + 1
           ITERM2= ITERM2+ 1
           IP(ITERM) = II
           JP(ITERM) = JJ
           KP(ITERM) = KK
           LP(ITERM) = LL
        ENDIF
     ELSE IF(NPHI.GT.NPHI2) THEN
        ITERM = ITERM + 1
        IP(ITERM) = II
        JP(ITERM) = JJ
        KP(ITERM) = KK
        LP(ITERM) = LL
     END IF
  END DO
  NPHI = ITERM
  NPHI2= ITERM2
  !
  ITERM = 0
  ITERM2= 0
  DO I = 1,NIMPHI
     II = IM(I)
     JJ = JM(I)
     KK = KM(I)
     LL = LM(I)
     IF(NIMPHI.LE.NIMPHI2) THEN
        QOK = (QATLAB(II).LT.0) .OR. (QATLAB(JJ).LT.0) .OR. (QATLAB(KK).LT.0) .OR. (QATLAB(LL).LT.0)
        IF (QOK) THEN
           ITERM = ITERM + 1
           ITERM2= ITERM2+ 1
           IM(ITERM) = II
           JM(ITERM) = JJ
           KM(ITERM) = KK
           LM(ITERM) = LL
        ENDIF
     ELSE IF(NIMPHI.GT.NIMPHI2) THEN
        ITERM = ITERM + 1
        IM(ITERM) = II
        JM(ITERM) = JJ
        KM(ITERM) = KK
        LM(ITERM) = LL
     END IF
  END DO
  NIMPHI = ITERM
  NIMPHI2= ITERM2

#if KEY_CMAP==1
  ITERM = 0
  ITERM2= 0
  DO I = 1,NCRTERM
     II1 = I1CT(I)
     JJ1 = J1CT(I)
     KK1 = K1CT(I)
     LL1 = L1CT(I)
     II2 = I2CT(I)
     JJ2 = J2CT(I)
     KK2 = K2CT(I)
     LL2 = L2CT(I)
     IF(NCRTERM.LE.NCRTERM2) THEN
        QOK = (QATLAB(II1).LT.0).OR.(QATLAB(JJ1).LT.0) .OR. (QATLAB(KK1).LT.0).OR.(QATLAB(LL1).LT.0) .OR. &
             (QATLAB(II2).LT.0).OR.(QATLAB(JJ2).LT.0) .OR. (QATLAB(KK2).LT.0).OR.(QATLAB(LL2).LT.0)
        IF (QOK) THEN
           ITERM = ITERM + 1
           ITERM2= ITERM2+ 1
           I1CT(ITERM) = II1
           J1CT(ITERM) = JJ1
           K1CT(ITERM) = KK1
           L1CT(ITERM) = LL1
           I2CT(ITERM) = II2
           J2CT(ITERM) = JJ2
           K2CT(ITERM) = KK2
           L2CT(ITERM) = LL2
        ENDIF
     ELSE IF(NCRTERM.GT.NCRTERM2) THEN
        ITERM = ITERM + 1
        I1CT(ITERM) = II1
        J1CT(ITERM) = JJ1
        K1CT(ITERM) = KK1
        L1CT(ITERM) = LL1
        I2CT(ITERM) = II2
        J2CT(ITERM) = JJ2
        K2CT(ITERM) = KK2
        L2CT(ITERM) = LL2
     END IF
  END DO
  NCRTERM = ITERM
  NCRTERM2= ITERM2
#endif 
  !
  RETURN
END SUBROUTINE SDFQM2
!
SUBROUTINE MULLIK(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     Carry out a population analysis of the QM density matrix.
  !
  use chm_kinds
  use dimens_fcm
  use number, only : zero
  use memory
  use sizes
  use quantm
  use scfblk
  use stream
  use qmlinkm
  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  !
  real(chm_real),allocatable,dimension(:) :: WORK1,WORK2,WORK3,Work_h
  real(chm_real),allocatable,dimension(:) :: QTOT
  integer:: ndim1,ndim2,ndim3,ndim4
  !
  ! BIOVIA Code Start : Bug fix
  IF (NORBS .EQ. 0 .OR. (.not.allocated(CALPHA)) .or.  &
     (.not.allocated(CBETA)) .or. (.not.allocated(PDENS)) ) THEN
     CALL WRNDIE(0,'<MULLIK>','Density undefined.')
     !
  ! BIOVIA Code End
  ELSE
     call chmalloc('qmset.src','MULLIK','QTOT',NATQM,crl=QTOT)
     call chmalloc('qmset.src','MULLIK','WORK1',NORBS*NORBS,crl=WORK1)
     call chmalloc('qmset.src','MULLIK','WORK2',(NORBS*(NORBS+1)/2),crl=WORK2)
     call chmalloc('qmset.src','MULLIK','WORK_h',NORBS*NORBS,crl=WORK_h)
     !
     ! transform MOs from HBO N+1 basis to AO N+4 basis for GHO
     ! including the UHF case ... PJ 12/2002
     !
     IF(QMLINK) THEN
        call chmalloc('qmset.src','MULLIK','WORK3',NORBS*NORBS,crl=WORK3)
        CALL MNCTRASF4(Calpha,work3)
        calpha(1:norbs*norbs) = WORK3(1:NORBS * NORBS)
        IF (UHF) THEN
           CALL MNCTRASF4(CBETA,WORK3)
           cbeta(1:norbs*norbs) = WORK3(1:NORBS * NORBS)
        ENDIF
     ENDIF
     !
     ndim1=size(calpha)
     ndim2=size(h_matrix)
     ndim3=NORBS*NORBS
     ndim4=(NORBS*(NORBS+1)/2)
     WORK_h(1:ndim2)=H_matrix(1:ndim2)
     WORK_h(ndim2+1:ndim3)=zero
     CALL MULIK2(CALPHA(1:ndim1),CBETA(1:ndim1),UHF,WORK_h(1:ndim3), &             ! H_matrix(1:ndim2)
          WORK1(1:ndim3),WORK2(1:ndim4),PDENS(1:ndim2),QTOT(1:natqm),OUTU)
     !
     call chmdealloc('qmset.src','MULLIK','QTOT',size(QTOT),crl=QTOT)
     call chmdealloc('qmset.src','MULLIK','WORK1',size(WORK1),crl=WORK1)
     call chmdealloc('qmset.src','MULLIK','WORK2',size(WORK2),crl=WORK2)
     call chmdealloc('qmset.src','MULLIK','WORK_h',size(WORK_h),crl=WORK_h)
     IF(QMLINK) call chmdealloc('qmset.src','MULLIK','WORK3',size(WORK3),crl=WORK3)
  ENDIF
  !
  RETURN
END SUBROUTINE MULLIK
!
SUBROUTINE MNCTRASF4(CHB,C)
  use chm_kinds
  use dimens_fcm
  use sizes
  use qmlinkm
  use quantm
  use scfblk
  use psf
  implicit none
  real(chm_real) CHB(*),C(*)
  INTEGER I,J,I1,J1,IJ,NORBHB,NAOS
  !
  NORBHB = NORBS - 3*MQMLNK
  NAOS = NORBHB-MQMLNK

  DO I = 1,NORBHB
     I1 = NORBS*(I-1)
     J1 = NORBHB*(I-1)
     C(I1+1:I1+NAOS)=CHB(J1+1:J1+NAOS)
     I1 = I1+NAOS
     J1 = J1+NAOS
     DO J = NAOS+1,NORBHB
        J1 = J1+1
        IJ = 16*(J-NAOS)-15
        C(I1+1:I1+4)=CHB(J1)*MBT(IJ:IJ+3)
        I1 = I1+4
     ENDDO
  ENDDO
  !
  !  append and tranform auxiliary hybrid orbitals ... PJ 12/2002
  !
  DO I = 1, MQMLNK
     DO J = 1, 3
        ! bug fix for multi-boundary cases ... PJ 7/2004
        !            I1 = NORBS*(NORBHB+I*J-1)
        I1 = NORBS*(NORBHB+(I-1)*3+J-1)
        IJ = 16*(I-1)+4*J
        DO J1 = 1, NORBS
           I1 = I1 + 1
           IF (J1 .GT. NAOS+(I-1)*4 .AND. J1 .LE. NAOS+I*4) THEN
              IJ = IJ + 1
              C(I1) = MBT(IJ)
           ELSE
              C(I1) = 0.0D0
           END IF
        END DO
     END DO
  END DO

  RETURN
END SUBROUTINE MNCTRASF4
!
SUBROUTINE MULIK2(C,CBETA,UHF,H,VECS,STORE,P,Q,OUTU2)
  !*********************************************************************
  !
  !     MULLIK DOES A MULLIKEN POPULATION ANALYSIS
  !     ON INPUT  C     =  SQUARE ARRAY OF EIGENVECTORS.
  !     H     =  PACKED ARRAY OF ONE-ELECTRON MATRIX
  !     VECS  =  WORKSTORE OF SIZE AT LEAST NORBS*NORBS
  !     STORE =  WORKSTORE OF SIZE AT LEAST (NORBS*(NORBS+1))/2
  !
  !*********************************************************************
  !
  use chm_kinds
  use dimens_fcm
  use stream
  use number,only : zero, one, two, tenm14
  !
  use am1parm
  use sizes
  use quantm
  !
  ! add for GHO case ... PJ 12/2002
  !
  use qmlinkm
  implicit none
  !
  INTEGER:: OUTU2
  LOGICAL:: UHF
  real(chm_real):: C(*),CBETA(*),H(*),VECS(*),STORE(*),P(*),Q(*)
  !C
  INTEGER:: JF,II,IJ,JJ,JL,IM1,LINEAR,I,J,K
  real(chm_real)::  BI,BJ,SUMM,SUM
  !C
  !
  !     FIRST, RE-CALCULATE THE OVERLAP MATRIX
  !
  real(chm_real)::  EIGS(MAXORB)   ! ,XYZ(3,MAXQM1)
  INTEGER:: I1ST,ILST,IFACT(MAXORB)
  !
  ! added for UHF density matrix calculation ... PJ 12/2002
  !
  INTEGER NA1EL,NA2EL,NB1EL,NB2EL
  NA1EL  = NALPHA + NOPEN
  NA2EL  = NCLOSE
  NB1EL  = NBETA + NOPEN
  NB2EL  = 0
  !
  do I=1,NORBS
     IFACT(I)=(I*(I-1))/2
  end do
  IFACT(NORBS+1)=(NORBS*(NORBS+1))/2
  Do I=1,NATQM
     I1ST=N1ST(I)
     ILST=NLAST(I)
     IM1=I-1
     BI=BETAS(NAT(I))
     do K=I1ST,ILST
        II=(K*(K-1))/2
        do J=1,IM1
           JF=N1ST(J)
           JL=NLAST(J)
           BJ=BETAS(NAT(J))
           do JJ=JF,JL
              IJ=II+JJ
              H(IJ)=two*H(IJ)/(BI+BJ)+tenm14      ! 1.D-14, which +1.D-14 IS TO PREVENT ERRORS IN THE DIAGONALISATION.
              STORE(IJ)=H(IJ)
              BJ=BETAP(NAT(J))
           end do
        end do
        STORE(II+I1ST:II+K)=zero
        H(II+I1ST:II+K)    =zero
        BI=BETAP(NAT(I))
     end do
  End do
  do i=1,norbs
     STORE(IFACT(i+1))=one
     H(IFACT(i+1))    =one
  end do
  CALL RSPQM(H,NORBS,NORBS,EIGS,VECS)
  do i=1,norbs
     EIGS(i)=one/SQRT(EIGS(i))
  end do
  IJ=0
  do I=1,NORBS
     do J=1,I
        IJ=IJ+1
        SUM=zero
        do K=1,NORBS
           jj=(K-1)*NORBS
           SUM=SUM + VECS(I+jj) * EIGS(K) * VECS(J+jj)
        end do
        H(I+(J-1)*NORBS)=SUM
        H(J+(I-1)*NORBS)=SUM
     end do
  end do
  !
  ! OTH$  RWISE PERFORM MULLIKEN ANALYSIS
  !
  CALL MULT(C,H,VECS,NORBS)
  I=-1
  !
  ! Modified to include both RHF case and the alpha density
  ! for UHF case. The old call is commented:
  !      CALL CALDENS(VECS,NORBS,NORBS,NCLOSE,NOPEN,FRACT,C)
  !                                           ... PJ 12/2002
  !
  CALL CALDENS(VECS,NORBS,NORBS,NA2EL,NA1EL,FRACT,C)

  !
  ! use a special one to calculate GHO density ... PJ 12/2002
  !
  IF (QMLINK) CALL CALDGHO(VECS,NORBS,NORBS,NA2EL,NA1EL,FRACT,C,MQMLNK,QMATMQM,UHF)

  LINEAR=(NORBS*(NORBS+1))/2
  C(1:LINEAR)=C(1:LINEAR)*STORE(1:LINEAR)
  SUMM=zero
  Do I=1,NORBS
     SUM=0
     do J=1,I
        SUM=SUM+C(IFACT(I)+J)
     end do
     do J=I+1,NORBS
        SUM=SUM+C(IFACT(J)+I)
     end do
     SUMM=SUMM+SUM
     C(IFACT(I+1))=SUM
  End do

  !
  ! For beta spin electrons in UHF case... PJ 12/2002
  !
  IF (UHF) THEN
     CALL MULT(CBETA,H,VECS,NORBS)
     I=-1
     CALL CALDENS(VECS,NORBS,NORBS,NB2EL,NB1EL,FRACT,CBETA)
     !
     ! Use a special one to calculate GHO density ... PJ 12/2002
     !
     IF (QMLINK) CALL CALDGHO(VECS,NORBS,NORBS,NB2EL,NB1EL,FRACT,CBETA,MQMLNK,QMATMQM,UHF)

     LINEAR=(NORBS*(NORBS+1))/2
     CBETA(1:LINEAR)=CBETA(1:LINEAR)*STORE(1:LINEAR)
     SUMM=zero
     do I=1,NORBS
        SUM=0
        do J=1,I
           SUM=SUM+CBETA(IFACT(I)+J)
        end do
        do J=I+1,NORBS
           SUM=SUM+CBETA(IFACT(J)+I)
        end do
        SUMM=SUMM+SUM
        CBETA(IFACT(I+1))=SUM
     end do
     !
     ! For UHF case, the total Mulliken density matrix is the sum
     ! of the alpha and the beta Mulliken density matrix ... PJ 12/2002
     !
     J=(NORBS*NORBS+NORBS)/2
     C(1:J) = C(1:J)+CBETA(1:J)
  END IF
  !
  !     Do charge analysis and print out.
  !
  IF(PRNLEV.GE.2) THEN
     WRITE (OUTU,'(A)') ' MULIK2> Total atomic charges :'
     WRITE (OUTU,'(8X,A)') 'Atom      Name      Charge'
  END IF
  CALL MNCHRGE(P,Q)
  SUM=zero
  Do I = 1,NATQM
     Q(I) = CORE(NAT(I)) - Q(I)
     IF(PRNLEV.GE.2) WRITE (OUTU,'(7X,I5,8X,A2,4X,F8.5)') I,ELEMNT(NAT(I)),Q(I)
     SUM=SUM+Q(I)
  End do
  IF(PRNLEV.GE.2) WRITE (OUTU,'(A,F8.5)') ' MULIK2> Net QM Charge :', SUM
  !
  !     Print out the mulliken analysis.
  !
  IF(PRNLEV.GE.2) WRITE (OUTU,'(/,A)') ' MULIK2> Mulliken Population Analysis :'
  CALL MNVECPRT(C,NORBS)
  !
  RETURN
END SUBROUTINE MULIK2
!
!      SUBROUTINE DYNDEN(WORD,Q,QTOT,QTOT2,NSTEP)
SUBROUTINE DYNDEN(WORD,DYNSTP)
  !-----------------------------------------------------------------------
  !     Perform a charge analysis during a dynamics run.   JG 5/2002
  !
  use chm_kinds
  use number
  use dimens_fcm
  !
  use stream
  use quantm
  use am1parm
  use scfblk
  implicit none
  !
  CHARACTER(len=4) WORD
  INTEGER   DYNSTP
  !     real(chm_real)    Q(*), QTOT(*), QTOT2(*)
  real(chm_real)    Q(MAXQM1),QTPR(MAXQM1),QTPR2(MAXQM1)
  real(chm_real)    CHGAS(MAXQM1),QTPRG(MAXQM1),QTPR2G(MAXQM1)
  !
  INTEGER   I
  real(chm_real):: rDYNSTP
  !
  IF (WORD .EQ. 'ACCU') THEN
     IF (QDECOM) THEN
        CALL MNCHRGE(PENZYM,Q)
     ELSE
        CALL MNCHRGE(PDENS,Q)
     END IF
     Q(1:NATQM)     = CORE(NAT(1:NATQM))- Q(1:NATQM)
     QTOTAL(1:NATQM)= QTOTAL(1:NATQM)   + Q(1:NATQM)
     QTOT2(1:NATQM) = QTOT2(1:NATQM)    + Q(1:NATQM)*Q(1:NATQM)
     !
     ! GAS PHASE CHARGES
     IF (QDECOM) THEN
        CALL MNCHRGE(PGAS,CHGAS)
        CHGAS(1:NATQM)  = CORE(NAT(1:NATQM))- CHGAS(1:NATQM)
        QTOTALG(1:NATQM)= QTOTALG(1:NATQM)  + CHGAS(1:NATQM)
        QTOT2G(1:NATQM) = QTOT2G(1:NATQM)   + CHGAS(1:NATQM)*CHGAS(1:NATQM)
     END IF
     !
  ELSE IF (WORD .EQ. 'INIT') THEN
     !     write(6,*) 'within DYNDEN init'
     Q(1:NATQM)     = ZERO
     QTOTAL(1:NATQM)= ZERO
     QTOT2(1:NATQM) = ZERO
     IF (QDECOM) THEN
        CHGAS(1:NATQM)  = ZERO
        QTOTALG(1:NATQM)= ZERO
        QTOT2G(1:NATQM) = ZERO
     END IF
     !
  ELSE IF (WORD .EQ. 'PRIN') THEN
     DYNSTP = DYNSTP -1
     IF(PRNLEV.GE.2) THEN
        WRITE (OUTU,'(A,I5,A)')' DYNDEN> Total atomic charges and their fluctuations for',' the last:',DYNSTP, 'steps'
        WRITE (OUTU,'(8X,A)')  'Atom      Name      Charge      Fluctuation'
     END IF
     rdynstp = one/DYNSTP
     QTPR(1:NATQM) =QTOTAL(1:NATQM)*RDYNSTP
     QTPR2(1:NATQM)=SQRT(QTOT2(1:NATQM)*RDYNSTP - QTPR(1:NATQM)*QTPR(1:NATQM))
     IF(PRNLEV.GE.2) WRITE (OUTU,'(7X,I5,8X,A2,4X,F8.5,9X,F8.5)') &
          (I,ELEMNT(NAT(I)),QTPR(I),QTPR2(I),I = 1,NATQM)
     !
     IF (QDECOM) THEN
        IF(PRNLEV.GE.2) THEN
           WRITE (OUTU,'(A,I5,A)') ' DYNDEN> Total atomic gas phase charges and their', &
                ' fluctuations for the last:',DYNSTP,' steps'
           WRITE (OUTU,'(8X,A)')   'Atom      Name      Charge      Fluctuation'
        END IF
        QTPRG(1:NATQM) =QTOTALG(1:NATQM)*RDYNSTP
        QTPR2G(1:NATQM)=SQRT(QTOT2G(1:NATQM)*RDYNSTP - QTPRG(1:NATQM)*QTPRG(1:NATQM))
        IF(PRNLEV.GE.2) WRITE (OUTU,'(7X,I5,8X,A2,4X,F8.5,9X,F8.5)') &
             (I,ELEMNT(NAT(I)),QTPRG(I),QTPR2G(I),I = 1,NATQM)
        !
     END IF
     DYNSTP = DYNSTP +1
  END IF
  !
  RETURN
END SUBROUTINE DYNDEN
#endif /* (qmsetmain)*/


SUBROUTINE HBFDEF
  !-----------------------------------------------------------------------
  !      SETUPS TRANSFORMATION MATRIX FOR HBO ON LINK ATOMS
  !      and EVALUATES THE CORE ENERGIES
  !
  use chm_kinds
  use dimens_fcm
  use number, only : zero,one,two,three,four
  use coord
  use psf
  use am1parm
  use quantm
  use sizes
  use qmlinkm
  implicit none
  !
  INTEGER I, J, II, JJ, IBND, ITST
  !
#if KEY_QUANTUM==1 /*q_hbfdef*/
  !       LOCATE QM and MM ATOMS CONNECTED TO THE LINK ATOM
  DO I = 1,MQMLNK
     IBND = 0
     ITST = 0
     DO J = 1,NBOND
        II = IB(J)
        JJ = JB(J)
        IF(II.EQ.IMQLINK(I)) THEN
           IF(QATLAB(JJ).GT.0) THEN
              ITST = ITST+1
              IF(ITST.GT.1) CALL WRNDIE(-5,'HBFDEF>','TOO MANY QM ATOMS CONNECTED TO THE LINK ATOM')
              KMQLINK(I) = JJ
           ELSE
              IBND = IBND+1
              IF(IBND.GT.3 ) CALL WRNDIE(-5, 'HBFDEF>','TOO MANY MM BONDS CONNECTING THE LINK ATOM')
              JMQLINK(IBND,I) = JJ
           ENDIF
        ELSEIF(JJ.EQ.IMQLINK(I)) THEN
           IF(QATLAB(II).GT.0) THEN
              ITST = ITST+1
              IF(ITST.GT.1) CALL WRNDIE(-5,'HBFDEF>','TOO MANY QM ATOMS CONNECTED TO THE LINK ATOM')
              KMQLINK(I) = II
           ELSE
              IBND = IBND+1
              IF(IBND.GT.3 ) CALL WRNDIE(-5, 'HBFDEF>','TOO MANY MM BONDS CONNECTING THE LINK ATOM')
              JMQLINK(IBND,I) = II
           ENDIF
        ENDIF
     ENDDO
  ENDDO
  !
  !
  CALL MNHBDEF4(X,Y,Z,MBT,MBTM,MDBTMMM,MQMLNK,IMQLINK,JMQLINK,KMQLINK)
  !
  !      DETERMINE CORE POTENTIALS
  !
  DO I = 1,MQMLNK
     II = NATQM-MQMLNK+I
     !
     !     core charge on QM-Link atom, and zero MM charge on QM-link atom.
     CORE(NAT(II)) = four        ! 4.0D+00
     QMATMQM(I)    = CG(IMQLINK(I))
     CG(IMQLINK(I))= zero        ! 0.0D+00
  ENDDO
  !
#endif /* (q_hbfdef)*/
  RETURN
END SUBROUTINE HBFDEF
#if KEY_QUANTUM==1 /*q3*/
SUBROUTINE MNHBDEF4(X,Y,Z,MBT,MBTM,MDBTMMM,NATVB,IATVB,JATVB,KATVB)
  !-----------------------------------------------------------------------
  !      DEFINES TRANSFORMATION MATRIX
  !
  use chm_kinds
  use consta
  use number,only : zero,one,two,three
  implicit none
  real(chm_real) X(*),Y(*),Z(*),MBT(*),MBTM(*),MDBTMMM(3,3,*)
  INTEGER NATVB,IATVB(*),JATVB(3,*),KATVB(*)
  !
  !   local variables
  INTEGER I,II,JJ,NI
  real(chm_real) A(3),B(3),C(3),AB(3),AC(3),P(3),T(3)
  !
  NI = 1
  DO I = 1,NATVB
     !
     !      EACH QM-LINK ATOM IS CONNECTED TO 1 QM ATOM and 3 MM ATOMS
     II = IATVB(I)
     !      MM ATOMS
     JJ = JATVB(1,I)
     A(1) = X(JJ)-X(II)
     A(2) = Y(JJ)-Y(II)
     A(3) = Z(JJ)-Z(II)
     JJ = JATVB(2,I)
     B(1) = X(JJ)-X(II)
     B(2) = Y(JJ)-Y(II)
     B(3) = Z(JJ)-Z(II)
     JJ = JATVB(3,I)
     C(1) = X(JJ)-X(II)
     C(2) = Y(JJ)-Y(II)
     C(3) = Z(JJ)-Z(II)
     !      QM ATOM
     JJ = KATVB(I)
     T(1) = X(JJ)-X(II)
     T(2) = Y(JJ)-Y(II)
     T(3) = Z(JJ)-Z(II)
     P(1) = ZERO
     P(2) = ZERO
     P(3) = ZERO
     CALL MNHBDRIV4(A,B,C,T,P,MBT(NI),MBTM(NI),MDBTMMM(1,1,NI))

     IF(P(1).NE.ZERO) CALL WRNDIE(-5,'HBDEF>','HYBRID ORBITAL ILLDEFINED.')
     NI = NI+16
  ENDDO
  !
  RETURN
END SUBROUTINE MNHBDEF4
SUBROUTINE MNACB4(A,B,C,S)
  !
  use chm_kinds
  implicit none
  !
  real(chm_real) A(3),B(3),C(3),S
  !
  c(1)= A(2)*B(3)-B(2)*A(3)
  c(2)= B(1)*A(3)-A(1)*B(3)
  c(3)= A(1)*B(2)-B(1)*A(2)
  S   = SQRT(c(1)*c(1)+c(2)*c(2)+c(3)*c(3))
  RETURN
END SUBROUTINE MNACB4
SUBROUTINE MNVBFTN4(F1,F1VB,X,NDIM)
  !
  use chm_kinds
  use number, only: zero
  implicit none
  !
  INTEGER NDIM,L1,I,J,K,L,L2
  real(chm_real) F1(*),F1VB(*),X(NDIM,NDIM)
  !
  real(chm_real) FAC
  !
  L1 = 0
  DO I = 1,NDIM
     DO J = 1,I
        L1 = L1+1
        L2 = 0
        F1VB(L1) = zero
        DO K = 1,NDIM
           DO L = 1,K-1
              L2  = L2+1
              FAC = X(K,I)*X(L,J)+X(K,J)*X(L,I)
              F1VB(L1) = F1VB(L1)+F1(L2)*FAC
           ENDDO
           L2  = L2+1
           FAC = X(K,I)*X(K,J)
           F1VB(L1) = F1VB(L1)+F1(L2)*FAC
        ENDDO
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE MNVBFTN4
SUBROUTINE MNHBDRIV4(A,B,C,D,O,MBT,MBTM,MDBTMMM)
  !
  use chm_kinds
  use number, only: zero,half,one,two,three
  implicit none
  !
  INTEGER I,J,K,N,IJ
  real(chm_real) A(3),B(3),C(3),D(3),O(3),MBT(4,4),MBTM(4,4),MDBTMMM(3,3,16)
  !
  !   Local variables.
  real(chm_real) AA(3),BB(3),CC(3),U(3),V(3),T(4,4)
  real(chm_real) X(3),Y(3),Z(3),TETH(4,4)
  real(chm_real) dthma(3,4,4),dthmb(3,4,4),dthmc(3,4,4)
  real(chm_real) dxa(3,3),dxb(3,3),dxc(3,3), &
       dya(3,3),dyb(3,3),dyc(3,3), &
       dza(3,3),dzb(3,3),dzc(3,3), &
       daa(3,3),dd1a(3),dd1b(3),dd1c(3)
  real(chm_real) DBTMM(4,4),DBTMMB(4,4),DBTMMC(4,4)
  real(chm_real) GRADA(4,4,3),GRADB(4,4,3),GRADC(4,4,3)
  real(chm_real) drxa(3),drxb(3),drxc(3),drza(3),drzb(3),drzc(3)
  real(chm_real) xx(3),ab(3),bc(3),ca(3)
  real(chm_real) dcsa(3),dcsb(3),dcsc(3),dcpa(3),dcpb(3),dcpc(3)

  real(chm_real), save :: AF(3,3)= reshape &
       ( (/0.0D0,1.0D0,-1.0D0,-1.0D0,0.0D0,1.0D0,1.0D0,-1.0D0,0.0D0/), &
         (/3,3/) )
  INTEGER,        save :: IR(3,3)= reshape &
       ( (/0,3,2,3,0,1,2,1,0/), (/3,3/) )
  !
  real(chm_real) DD,RA,RB,RC,CS,PFAC,RA2,RB2,RC2,CS2 ! RX,RY,RZ
  real(chm_real) THRFAC,RBA,RAC,RBC,d0
  real(chm_real):: rabc(3),r_rabc(3),rabc2(3),rxyz(3),r_rxyz(3),temp_r(3)
  real(chm_real),parameter :: r_three=one/three
  !
  !     do some initialiaztion
  grada=zero
  gradb=zero
  gradc=zero

  rabc2(1) = a(1)**2+a(2)**2+a(3)**2
  rabc2(2) = b(1)**2+b(2)**2+b(3)**2
  rabc2(3) = c(1)**2+c(2)**2+c(3)**2
  rabc(1:3)  = sqrt(rabc2(1:3))
  r_rabc(1:3)= one/rabc(1:3)
  do i = 1,3
     aa(i) = a(i)*r_rabc(1)  ! /ra
     bb(i) = b(i)*r_rabc(2)  ! /rb
     cc(i) = c(i)*r_rabc(3)  ! /rc
     u(i) = bb(i) - aa(i)
     v(i) = cc(i) - aa(i)
  enddo

  CALL MNACB4(u,v,x,rxyz(1))

  d0 = (aa(1)*x(1)+aa(2)*x(2)+aa(3)*x(3))/rxyz(1)
  dd  = abs(d0)
  if(d0.GT.zero) then
     pfac = -one
  else
     pfac = one
  end if
  !
  !     tetrahedarl hybrid orbitals:
  cs = sqrt(dd/(one+dd))

  cs2 = sqrt((one-cs**2)*r_three)   ! /3.0D0)

  do i = 1,4
     teth(1,i)  = cs2
     teth(2:4,i)= zero
  enddo
  teth(1,1) = cs

  teth(2,1) = sqrt(one-teth(1,1)**2)

  teth(2,2) =-teth(1,1)*teth(1,2)/teth(2,1)
  teth(3,2) = sqrt(one-teth(1,2)**2-teth(2,2)**2)

  teth(2,3) =-teth(1,1)*teth(1,3)/teth(2,1)
  teth(3,3) =-(teth(1,2)*teth(1,3)+teth(2,2)*teth(2,3))/teth(3,2)
  teth(4,3) = sqrt(one-teth(1,3)**2-teth(2,3)**2-teth(3,3)**2)

  teth(2,4) =-teth(1,1)*teth(1,4)/teth(2,1)
  teth(3,4) =-(teth(1,2)*teth(1,4)+teth(2,2)*teth(2,4))/teth(3,2)
  teth(4,4) =-(teth(1,3)*teth(1,4)+teth(2,3)*teth(2,4)+teth(3,3)*teth(3,4))/teth(4,3)
  !
  !
  x(1:3) = pfac*x(1:3)
  call MNacb4(x,aa,z,rxyz(3))
  call MNacb4(z,x,y,rxyz(2))

  r_rxyz(1:3)=one/rxyz(1:3)

  t(1,1:4) = zero
  t(1:4,1) = zero
  t(1,1)   = one
  t(2:4,2) = x(1:3)*r_rxyz(1)  !/rx
  t(2:4,3) = y(1:3)*r_rxyz(2)  !/ry
  t(2:4,4) = z(1:3)*r_rxyz(3)  !/rz

  call MNacb4(bb,cc,bc,rbc)
  call MNacb4(cc,aa,ca,rac)
  call MNacb4(aa,bb,ab,rba)

  do i = 1,3             !  dai
     do j = 1,3          !  dxj
        !           dxj/dai
        dxa(j,i) = -aa(i)*(ab(j)+ca(j))*r_rabc(1)  ! /ra
        dxb(j,i) = -bb(i)*(bc(j)+ab(j))*r_rabc(2)  ! /rb
        dxc(j,i) = -cc(i)*(ca(j)+bc(j))*r_rabc(3)  ! /rc
        if(j.ne.i) then
           dxa(j,i) = dxa(j,i)+af(j,i)*(cc(ir(j,i))-bb(ir(j,i)))*r_rabc(1)  ! /ra
           dxb(j,i) = dxb(j,i)+af(j,i)*(aa(ir(j,i))-cc(ir(j,i)))*r_rabc(2)  ! /rb
           dxc(j,i) = dxc(j,i)+af(j,i)*(bb(ir(j,i))-aa(ir(j,i)))*r_rabc(3)  ! /rc
        endif
     enddo
  enddo
  if(pfac.eq.one) then
     dxa(1:3,1:3) = -dxa(1:3,1:3)
     dxb(1:3,1:3) = -dxb(1:3,1:3)
     dxc(1:3,1:3) = -dxc(1:3,1:3)
  endif
  !
  !     (Rx^2)'
  do i=1,3
     drxa(i) = two*(x(1)*dxa(1,i)+x(2)*dxa(2,i)+x(3)*dxa(3,i))
     drxb(i) = two*(x(1)*dxb(1,i)+x(2)*dxb(2,i)+x(3)*dxb(3,i))
     drxc(i) = two*(x(1)*dxc(1,i)+x(2)*dxc(2,i)+x(3)*dxc(3,i))
  end do

  !     dxj/dmi
  do i = 1,3
     grada(2:4,2,i) = dxa(1:3,i)*r_rxyz(1)-half*x(1:3)*drxa(i)*(r_rxyz(1)**3)   ! /rx, /(rx**3)
     gradb(2:4,2,i) = dxb(1:3,i)*r_rxyz(1)-half*x(1:3)*drxb(i)*(r_rxyz(1)**3)
     gradc(2:4,2,i) = dxc(1:3,i)*r_rxyz(1)-half*x(1:3)*drxc(i)*(r_rxyz(1)**3)
  enddo
  !
  !      daaj/dai
  do i = 1,3
     daa(1:3,i)=-aa(1:3)*aa(i)*r_rabc(1)  ! /ra
     daa(i,i)  = daa(i,i)+r_rabc(1)       ! one/ra
  enddo

  do i = 1,3
     dza(1,i) = dxa(2,i)*aa(3)+x(2)*daa(3,i)-daa(2,i)*x(3)-aa(2)*dxa(3,i)
     dza(2,i) = dxa(3,i)*aa(1)+x(3)*daa(1,i)-daa(3,i)*x(1)-aa(3)*dxa(1,i)
     dza(3,i) = dxa(1,i)*aa(2)+x(1)*daa(2,i)-daa(1,i)*x(2)-aa(1)*dxa(2,i)

     dzb(1,i) = dxb(2,i)*aa(3)-aa(2)*dxb(3,i)
     dzb(2,i) = dxb(3,i)*aa(1)-aa(3)*dxb(1,i)
     dzb(3,i) = dxb(1,i)*aa(2)-aa(1)*dxb(2,i)

     dzc(1,i) = dxc(2,i)*aa(3)-aa(2)*dxc(3,i)
     dzc(2,i) = dxc(3,i)*aa(1)-aa(3)*dxc(1,i)
     dzc(3,i) = dxc(1,i)*aa(2)-aa(1)*dxc(2,i)
  enddo

  !     (Rz^2)'
  do i=1,3
     drza(i) = two*(z(1)*dza(1,i)+z(2)*dza(2,i)+z(3)*dza(3,i))
     drzb(i) = two*(z(1)*dzb(1,i)+z(2)*dzb(2,i)+z(3)*dzb(3,i))
     drzc(i) = two*(z(1)*dzc(1,i)+z(2)*dzc(2,i)+z(3)*dzc(3,i))
  end do
  !
  !      dzj/dmi
  !
  do i = 1,3
     grada(2:4,4,i) = dza(1:3,i)*r_rxyz(3)-half*z(1:3)*drza(i)*(r_rxyz(3)**3)   ! /rz, /(rz**3)
     gradb(2:4,4,i) = dzb(1:3,i)*r_rxyz(3)-half*z(1:3)*drzb(i)*(r_rxyz(3)**3)
     gradc(2:4,4,i) = dzc(1:3,i)*r_rxyz(3)-half*z(1:3)*drzc(i)*(r_rxyz(3)**3)
  enddo
  !
  do i = 1,3
     dya(1,i) = dza(2,i)*x(3)+z(2)*dxa(3,i)-dxa(2,i)*z(3)-x(2)*dza(3,i)
     dya(2,i) = dza(3,i)*x(1)+z(3)*dxa(1,i)-dxa(3,i)*z(1)-x(3)*dza(1,i)
     dya(3,i) = dza(1,i)*x(2)+z(1)*dxa(2,i)-dxa(1,i)*z(2)-x(1)*dza(2,i)
     dyb(1,i) = dzb(2,i)*x(3)+z(2)*dxb(3,i)-dxb(2,i)*z(3)-x(2)*dzb(3,i)
     dyb(2,i) = dzb(3,i)*x(1)+z(3)*dxb(1,i)-dxb(3,i)*z(1)-x(3)*dzb(1,i)
     dyb(3,i) = dzb(1,i)*x(2)+z(1)*dxb(2,i)-dxb(1,i)*z(2)-x(1)*dzb(2,i)
     dyc(1,i) = dzc(2,i)*x(3)+z(2)*dxc(3,i)-dxc(2,i)*z(3)-x(2)*dzc(3,i)
     dyc(2,i) = dzc(3,i)*x(1)+z(3)*dxc(1,i)-dxc(3,i)*z(1)-x(3)*dzc(1,i)
     dyc(3,i) = dzc(1,i)*x(2)+z(1)*dxc(2,i)-dxc(1,i)*z(2)-x(1)*dzc(2,i)
  enddo
  !
  !      dyj/dmi
  !
  do i = 1,3
     grada(2:4,3,i) = (dya(1:3,i)-half*y(1:3)*(drza(i)*(r_rxyz(3)**2)+drxa(i)*(r_rxyz(1)**2)))*r_rxyz(2)   ! /(rz**2), /(rx**2),/ry
     gradb(2:4,3,i) = (dyb(1:3,i)-half*y(1:3)*(drzb(i)*(r_rxyz(3)**2)+drxb(i)*(r_rxyz(1)**2)))*r_rxyz(2)
     gradc(2:4,3,i) = (dyc(1:3,i)-half*y(1:3)*(drzc(i)*(r_rxyz(3)**2)+drxc(i)*(r_rxyz(1)**2)))*r_rxyz(2)
  enddo
  !
  !     d d/dmi
  !
  dthma(1:3,1:4,1:4) = zero
  dthmb(1:3,1:4,1:4) = zero
  dthmc(1:3,1:4,1:4) = zero
  do i = 1,3
     !MJF . Signs have been changed here!
     dd1a(i)= ((dxa(1,i)*aa(1)+dxa(2,i)*aa(2)+dxa(3,i)*aa(3)- daa(1,i)*x(1)-daa(2,i)*x(2)-daa(3,i)*x(3))*r_rxyz(1) &
          -half*(x(1)*aa(1)+x(2)*aa(2)+x(3)*aa(3))*drxa(i)*(r_rxyz(1)**3))       ! /rx, /(rx**3)
     dd1b(i)= ((dxb(1,i)*aa(1)+dxb(2,i)*aa(2)+dxb(3,i)*aa(3))*r_rxyz(1) &
          -half*(x(1)*aa(1)+x(2)*aa(2)+x(3)*aa(3))*drxb(i)*(r_rxyz(1)**3))
     dd1c(i)= ((dxc(1,i)*aa(1)+dxc(2,i)*aa(2)+dxc(3,i)*aa(3))*r_rxyz(1) &
          -half*(x(1)*aa(1)+x(2)*aa(2)+x(3)*aa(3))*drxc(i)*(r_rxyz(1)**3))
  enddo

  temp_r(1)=half*(one/teth(1,1)-teth(1,1))*(teth(2,1)**2)
  temp_r(2)=half*teth(2,1)**3
  dcsa(1:3) =-temp_r(1)*dd1a(1:3)
  dcsb(1:3) =-temp_r(1)*dd1b(1:3)
  dcsc(1:3) =-temp_r(1)*dd1c(1:3)
  dcpa(1:3) = temp_r(2)*dd1a(1:3)
  dcpb(1:3) = temp_r(2)*dd1b(1:3)
  dcpc(1:3) = temp_r(2)*dd1c(1:3)
  !
  thrfac = one/sqrt(three)
  dthma(1:3,1,1) = dcsa(1:3)
  dthma(1:3,2,1) = dcpa(1:3)
  temp_r(1:3)    = dcpa(1:3)*thrfac
  dthma(1:3,1,2) = temp_r(1:3)
  dthma(1:3,1,3) = temp_r(1:3)
  dthma(1:3,1,4) = temp_r(1:3)
  temp_r(1:3)    =-dcsa(1:3)*thrfac
  dthma(1:3,2,2) = temp_r(1:3)
  dthma(1:3,2,3) = temp_r(1:3)
  dthma(1:3,2,4) = temp_r(1:3)
  !
  dthmb(1:3,1,1) = dcsb(1:3)
  dthmb(1:3,2,1) = dcpb(1:3)
  temp_r(1:3)    = dcpb(1:3)*thrfac
  dthmb(1:3,1,2) = temp_r(1:3)
  dthmb(1:3,1,3) = temp_r(1:3)
  dthmb(1:3,1,4) = temp_r(1:3)
  temp_r(1:3)    =-dcsb(1:3)*thrfac
  dthmb(1:3,2,2) = temp_r(1:3)
  dthmb(1:3,2,3) = temp_r(1:3)
  dthmb(1:3,2,4) = temp_r(1:3)
  !C
  dthmc(1:3,1,1) = dcsc(1:3)
  dthmc(1:3,2,1) = dcpc(1:3)
  temp_r(1:3)    = dcpc(1:3)*thrfac
  dthmc(1:3,1,2) = temp_r(1:3)
  dthmc(1:3,1,3) = temp_r(1:3)
  dthmc(1:3,1,4) = temp_r(1:3)
  temp_r(1:3)    =-dcsc(1:3)*thrfac
  dthmc(1:3,2,2) = temp_r(1:3)
  dthmc(1:3,2,3) = temp_r(1:3)
  dthmc(1:3,2,4) = temp_r(1:3)

  !
  !      COLLECT
  !
  DO J = 1,4
     DO I = 1,4
        MBT(I,J) = zero
        DO K = 1,4
           MBT(I,J) = MBT(I,J)+T(I,K)*TETH(K,J)
        ENDDO
        MBTM(J,I) = MBT(I,J)
     ENDDO
  ENDDO
  !
  !     Derivatives
  !
  DO N = 1,3
     do J = 1,4
        do I = 1,4
           dbtmm(i,j)  = zero
           dbtmmb(i,j) = zero
           dbtmmc(i,j) = zero
           do K = 1,4
              !MJF . Lines have been uncommented!
              dbtmm(i,j) = dbtmm(i,j)+grada(i,k,n)*teth(k,j)+t(i,k)*dthma(n,k,j)
              dbtmmb(i,j)= dbtmmb(i,j)+gradb(i,k,n)*teth(k,j)+t(i,k)*dthmb(n,k,j)
              dbtmmc(i,j)= dbtmmc(i,j)+gradc(i,k,n)*teth(k,j)+t(i,k)*dthmc(n,k,j)
           end do
        end do
     end do
     ij = 0
     do i = 1,4
        do j = 1,4
           ij = ij+1
           Mdbtmmm(n,1,ij) = dbtmm(i,j)
           Mdbtmmm(n,2,ij) = dbtmmb(i,j)
           Mdbtmmm(n,3,ij) = dbtmmc(i,j)
        enddo
     enddo
  ENDDO
  !
  return
END SUBROUTINE MNHBDRIV4

!-------------------------------------------------------------------
SUBROUTINE CALDGHO(C,MDIM,NORBS,NDUBL,NSINGL,FRACT,P,MQMLNK,QMATMQM,UHF)
  !-------------------------------------------------------------------
  !
  !     Modified density matrix construction routine, to include GHO
  !     case. Assume we already obtained MOs in AO basis, the GHO
  !     density matrix in AO basis can be directly caculated from these
  !     MOs, however, we have to treat auxiliary orbital differently
  !     because auxiliary oribtals are fractional occupied by 1-qB/3.0d0
  !     (qB is the MM charge on the GHO boundary atom B). Both the
  !     RHF-GHO and UHF-GHO case are included ... PJ 12/2002
  !
  use chm_kinds
  use number
  use quantm, only : IFRAC 
  implicit none
  !
  INTEGER MDIM
  real(chm_real) P(*), C(MDIM,*)
  !***********************************************************************
  !
  !  DENSIT COMPUTES THE DENSITY MATRIX GIVEN THE EIGENVECTOR MATRIX, AND
  !          INFORMATION ABOUT THE M.O. OCCUPANCY.
  !
  !  INPUT:  C     = SQUARE EIGENVECTOR MATRIX, C IS OF SIZE MDIM BY MDIM
  !                  AND THE EIGENVECTORS ARE STORED IN THE TOP LEFT-HAND
  !                  CORNER.
  !          NORBS = NUMBER OF ORBITALS
  !          NDUBL = NUMBER OF DOUBLY-OCCUPIED M.O.S ( =0 IF UHF)
  !          NSINGL= NUMBER OF SINGLY OR FRACTIONALLY OCCUPIED M.O.S.
  !
  !   ON EXIT: P   = DENSITY MATRIX
  !
  !   MOPAC routine DENSIT
  !   Change to CALDENS in order to resolve a name conflect with DENSIT
  !      in SOLANA of CHARMM 22. 02.15.91 YDW
  !
  !***********************************************************************
  !
  INTEGER NORBS,NORBS2,NL1,NL2,NU1,NU2,I,J,K,L,NSINGL,NDUBL
  real(chm_real)  CONST,SUM1,SUM2,FRAC,SIGN,FRACT
  !
  ! For auxiliary orbitals in GHO
  !
  INTEGER MQMLNK
  real(chm_real) QMATMQM(*)
  LOGICAL UHF
  !
  ! local varibles
  !
  INTEGER KK, KKK
  real(chm_real) SUMAUX, FACTAUX
  real(chm_real),parameter :: r_three=one/three
  !
  ! SET UP LIMITS FOR SUMS
  !  NL1 = BEGINING OF ONE ELECTRON SUM
  !  NU1 = END OF SAME
  !  NL2 = BEGINING OF TWO ELECTRON SUM
  !  NU2 = END OF SAME
  !
  NORBS2=NORBS/2
  NSINGL=MAX(NDUBL,NSINGL)
  IF((IFRAC.LE.0) .OR. (NSINGL.GT.NORBS2)) THEN
     !
     !    TAKE POSITRON EQUIVALENT
     !
     SIGN=-ONE
     FRAC=TWO-FRACT
     IF(NDUBL.EQ.0)THEN
        CONST=ONE
        NL2=2
        NU2=0
        NL1=NSINGL+1
        NU1=NORBS
     ELSE
        CONST=TWO
        NL2=NSINGL+1
        NU2=NORBS
        NL1=NDUBL+1
        NU1=NSINGL
     ENDIF
  ELSE
     !
     !    TAKE ELECTRON EQUIVALENT
     !
     SIGN=ONE
     FRAC=FRACT
     CONST=ZERO
     NL2=1
     NU2=NDUBL
     NL1=NDUBL+1
     NU1=NSINGL
  ENDIF
  L=0
  do I=1,NORBS
     do J=1,I
        L=L+1
        SUM2=ZERO
        SUM1=ZERO
        do K=NL2,NU2
           SUM2=SUM2+C(I,K)*C(J,K)
        end do
        SUM2=SUM2*TWO
        do K=NL1,NU1
           SUM1=SUM1+C(I,K)*C(J,K)
        end do
        !
        ! auxiliary orbitals fractionally occupied for GHO ... PJ 12/2002
        !
        SUMAUX=ZERO
        K = NORBS-3*MQMLNK
        DO KK=1, MQMLNK
           IF (UHF) THEN
              FACTAUX = (one-QMATMQM(KK)*r_three)*half  ! /3.0d0
           ELSE
              FACTAUX = one-QMATMQM(KK)*r_three
           END IF
           DO KKK=1, 3
              K = K + 1
              SUMAUX=SUMAUX+FACTAUX*C(I,K)*C(J,K)
           END DO
        END DO
        !
        ! for GHO, plus the contributation of auxiliary orbitals ... PJ 12/2002
        !
        P(L)=(SUM2+SUM1*FRAC)*SIGN + SUMAUX
     end do
     P(L)=CONST+P(L)
  end do
  !
  !
  RETURN
END SUBROUTINE CALDGHO
#endif /* (q3)*/

!
! namkh 08/08/04
! QM/MM-Ewald
#if KEY_QUANTUM==1 /*q2*/
!----------------------------------------------------------------------
SUBROUTINE SETUP_QMEWD
  !
  !   Setup and ready for Ewald summation calculation
  !   Author:
  !
  !     This routine calculates non bonded interaction energies and
  !     forces via the Ewald Summation
  !     NATOM  - number of atoms
  !
  !     This has been copied from file nbonds/ewaldf.src and modified
  !     namkh
  !
  use chm_kinds
  use dimens_fcm
  use number
  use consta
  use memory
  use inbnd
  use image
  use quantm
  implicit none
  !
  LOGICAL QDONE
  !
  INTEGER KSQ, KX, KY, KZ, KSY, KSZ
  !
  !     SETUPK1
  !
  ! Do we have to do this?
  IF(QSETUPKQ) THEN
     QDONE =OKMAXQ.EQ.KMAXQ     .AND. &
          OKSQMAXQ.EQ.KSQMAXQ .AND. &
          OKMAXXQ.EQ.KMAXXQ   .AND. &
          OKMAXYQ.EQ.KMAXYQ   .AND. &
          OKMAXZQ.EQ.KMAXZQ  
     QFIRSTD = .TRUE.
  ELSE
     QSETUPKQ = .TRUE.
     QDONE=.FALSE.
     QFIRSTD = .FALSE.
  ENDIF
  !
  ! do SETUPK1
  if(.not.QDONE) then
     TOTKQ = 0
     DO KX = 0, KMAXXQ
        IF(KX.EQ.0) THEN
           KSY = 0
        ELSE
           KSY = -KMAXYQ
        ENDIF
        DO KY = KSY, KMAXYQ
           IF(KX.EQ.0.AND.KY.EQ.0) THEN
              KSZ = 1
           ELSE
              KSZ = -KMAXZQ
           ENDIF
           DO KZ = KSZ, KMAXZQ
              KSQ = KX*KX + KY*KY + KZ*KZ
              IF (KSQ.LE.KSQMAXQ.AND. KSQ.NE.0) TOTKQ = TOTKQ + 1
           END DO
        END DO
     END DO
     !
     OKMAXQ   = KMAXQ
     OKMAXXQ  = KMAXXQ
     OKMAXYQ  = KMAXYQ
     OKMAXZQ  = KMAXZQ
     OKSQMAXQ = KSQMAXQ
  end if
  ! end of SETUPK1
  !
  IF(.NOT.QDONE) THEN
     IF(MAXKVQ.GT.0 .and. allocated(PKVECQ)) THEN
        call chmdealloc('qmset.src','SETUP_QMEWD','PKVECQ',size(PKVECQ),crl=PKVECQ)
     END IF
     MAXKVQ = TOTKQ
     call chmalloc('qmset.src','SETUP_QMEWD','PKVECQ',MAXKVQ,crl=PKVECQ)
  END IF

  CALL SETUP_KVEC(PKVECQ)
  !
  RETURN
END SUBROUTINE SETUP_QMEWD
!
SUBROUTINE SETUP_KVEC(KVEC)
  !
  !    Setup wave-vector arrays for K-space ewald summation.
  !    Authors:
  !           Stephen H. Fleischman
  !           Roland Stote
  !           11/91
  !
  use ewald,only:kappa
  use chm_kinds
  use dimens_fcm
  use number
  use consta
  use inbnd
  use stream
  use image
  use quantm
  use prssre
  !
  implicit none
  !
  real(chm_real) KVEC(*)
  !
  real(chm_real)  VFACT
  !
  INTEGER KSQ, KX, KY, KZ, II
  INTEGER IPT, KSY, KSZ
  real(chm_real)  B, RKX(3), RKY(3), RKZ(3), RKSQ
  real(chm_real)  VOLUME, XTLINV(6), MKV(3)
  LOGICAL OK
  !
  !
  ! Now do the SETUPK2 work here...(see ewaldf.src file)
  !     Calculate Volume of System and Reciprocal Space Lattice Vector
  CALL GETVOL(VOLUME)
  CALL INVT33S(XTLINV, XTLABC, OK)
  !
  B        = PI*PI/KAPPA/KAPPA
  VFACT    = ONE/PI/VOLUME
  ! save for late usage
  VOLME    = VOLUME
  !
  IPT = 0
  DO KX = 0, KMAXXQ
     IF(KX.EQ.0) THEN
        KSY = 0
     ELSE
        KSY = -KMAXYQ
     ENDIF
     RKX(1) = KX * XTLINV(1)
     RKX(2) = KX * XTLINV(2)
     RKX(3) = KX * XTLINV(4)
     DO KY = KSY, KMAXYQ
        IF(KX.EQ.0.AND.KY.EQ.0) THEN
           KSZ = 1
        ELSE
           KSZ = -KMAXZQ
        ENDIF
        RKY(1) = KY * XTLINV(2)
        RKY(2) = KY * XTLINV(3)
        RKY(3) = KY * XTLINV(5)
        DO KZ = KSZ, KMAXZQ
           RKZ(1) = KZ * XTLINV(4)
           RKZ(2) = KZ * XTLINV(5)
           RKZ(3) = KZ * XTLINV(6)
           KSQ = KX*KX + KY*KY + KZ*KZ
           IF (KSQ.LE.KSQMAXQ .AND. KSQ.NE.0) THEN
              IPT = IPT + 1
              MKV(1:3) = RKX(1:3) + RKY(1:3) + RKZ(1:3)
              RKSQ     = MKV(1)*MKV(1) + MKV(2)*MKV(2) + MKV(3)*MKV(3)
              KVEC(IPT)= VFACT*EXP((-B*RKSQ))/RKSQ
           END IF
        END DO
     END DO
  END DO
  !
  IF(IPT.NE.TOTKQ) CALL WRNDIE(-4,'<SETUPK_KVEC>','Bad TOTKQ count')
  !
  RETURN
END SUBROUTINE SETUP_KVEC
!
#if KEY_PBOUND==1 /*pbound*/
SUBROUTINE PBCHECK(DXI,DYI,DZI)
  !
  use chm_kinds
  use dimens_fcm
  use number
  use pbound
  implicit none
  real(chm_real) CORR
  real(chm_real) DXI,DYI,DZI
  real(chm_real) :: dxyz(3), box_inv(1:3)
  integer :: i

  box_inv(1)=BOXINV
  box_inv(2)=BOYINV
  box_inv(3)=BOZINV
  dxyz(1)   = DXI
  dxyz(2)   = DYI
  dxyz(3)   = DZI
  If(qBoun) then
     If(qCUBoun.or.qTOBoun) then
        dxyz(1:3)=box_inv(1:3)*dxyz(1:3)
        do i=1,3
           if(dxyz(i).gt. half) dxyz(i)=dxyz(i)-one
           if(dxyz(i).lt.-half) dxyz(i)=dxyz(i)+one
        end do
        if (qTOBoun) Then
           corr = HALF*AINT(R75*(abs(dxyz(1))+abs(dxyz(2))+abs(dxyz(3))))
           do i=1,3
              dxyz(i)=dxyz(i)-SIGN(corr,dxyz(i))
           end do
        end if
        DXI = XSIZE*dxyz(1)
        DYI = YSIZE*dxyz(2)
        DZI = ZSIZE*dxyz(3)
     Else
        Call PBMove(DXI, DYI, DZI)
     Endif
  Endif

  RETURN
END SUBROUTINE PBCHECK
#endif /* (pbound)*/
!
!
!  DTM Routine added to read EXTErnal parameters
SUBROUTINE READPAR(PUNIT)

  use chm_kinds
  use dimens_fcm
  use am1parm
  use string
  use stream
  implicit none

  INTEGER  PUNIT, I, J
  CHARACTER(len=8) PARNAME, ATOMNAME, ELESCR
  real(chm_real) PARVAL 
  !  String variables
  INTEGER LINELEN
  CHARACTER(len=80) LINE

  !
  !     Format of file is
  !     PARNAME1 ATOMTYPE1 PARVALUE1
  !     PARNAME2 ATOMTYPE2 PARVALUE2
  !     ...
  !     END
  !



  !  Read data file
  Do              ! outer while
10   continue
     do           ! inner while
        READ(PUNIT,'(A)',ERR=100,END=40) LINE
        LINELEN = LEN(LINE)
        if (LINE.NE.' ') EXIT     ! exit inner while
     end do          ! inner while

     IF(WRNLEV.GE.2) WRITE(OUTU,'(2A)') ' READPAR> ',LINE

     CALL NEXTWD(LINE,LINELEN,SWDTCH,SWDMAX,SWDLEN)
     ! if(prnlev.ge.2) WRITE(*,*) LINE,LINELEN,SWDTCH,SWDMAX,SWDLEN
     IF (SWDTCH.EQ.'END') EXIT   ! exit outer while  GOTO 40
     PARNAME = SWDTCH


     CALL NEXTWD(LINE,LINELEN,SWDTCH,SWDMAX,SWDLEN)
     ATOMNAME = SWDTCH

     CALL NEXTWD(LINE,LINELEN,SWDTCH,SWDMAX,SWDLEN)
     PARVAL = DECODF(SWDTCH,SWDLEN)

     do J = 0,100 
        ELESCR = ELEMNT(J)
        IF (ELESCR(1:1).EQ.' ') ELESCR = ELESCR(2:2) 
        IF (ATOMNAME.EQ.ELESCR) I = J
     end do

     !         WRITE(*,*) ATOMNAME,ELEMNT(I),I
     ! Assign parameters
     IF (PARNAME.EQ.'ALFA'.OR.PARNAME.EQ.'ALP') THEN
        ALFA(I)  = PARVAL 
     ELSEIF (PARNAME.EQ.'BETAS') THEN
        BETAS(I) = PARVAL
     ELSEIF (PARNAME.EQ.'BETAP') THEN
        BETAP(I) = PARVAL
     ELSEIF (PARNAME.EQ.'BETAD') THEN
        BETAD(I) = PARVAL
     ELSEIF (PARNAME.EQ.'ZS') THEN
        ZS(I)    = PARVAL
     ELSEIF (PARNAME.EQ.'ZP') THEN
        ZP(I)    = PARVAL
     ELSEIF (PARNAME.EQ.'ZD') THEN
        ZD(I)    = PARVAL
     ELSEIF (PARNAME.EQ.'USS') THEN
        USS(I)   = PARVAL
     ELSEIF (PARNAME.EQ.'UPP') THEN
        UPP(I)   = PARVAL
     ELSEIF (PARNAME.EQ.'UDD') THEN
        UDD(I)   = PARVAL
     ELSEIF (PARNAME.EQ.'GSS') THEN
        GSS(I)   = PARVAL
     ELSEIF (PARNAME.EQ.'GPP') THEN
        GPP(I)   = PARVAL
     ELSEIF (PARNAME.EQ.'GSP') THEN
        GSP(I)   = PARVAL
     ELSEIF (PARNAME.EQ.'GP2') THEN
        GP2(I)   = PARVAL
     ELSEIF (PARNAME.EQ.'HSP') THEN
        HSP(I)   = PARVAL
     ELSEIF (PARNAME.EQ.'VS') THEN
        VS(I)    = PARVAL
     ELSEIF (PARNAME.EQ.'VP') THEN
        VP(I)    = PARVAL
     ELSEIF (PARNAME.EQ.'K1'.OR.PARNAME.EQ.'FN11') THEN
        FN1(I,1) = PARVAL
     ELSEIF (PARNAME.EQ.'L1'.OR.PARNAME.EQ.'FN21') THEN
        FN2(I,1) = PARVAL
     ELSEIF (PARNAME.EQ.'M1'.OR.PARNAME.EQ.'FN31') THEN
        FN3(I,1) = PARVAL
     ELSEIF (PARNAME.EQ.'K2'.OR.PARNAME.EQ.'FN12') THEN
        FN1(I,2) = PARVAL
     ELSEIF (PARNAME.EQ.'L2'.OR.PARNAME.EQ.'FN22') THEN
        FN2(I,2) = PARVAL
     ELSEIF (PARNAME.EQ.'M2'.OR.PARNAME.EQ.'FN32') THEN
        FN3(I,2) = PARVAL
     ELSEIF (PARNAME.EQ.'K3'.OR.PARNAME.EQ.'FN13') THEN
        FN1(I,3) = PARVAL
     ELSEIF (PARNAME.EQ.'L3'.OR.PARNAME.EQ.'FN23') THEN
        FN2(I,3) = PARVAL
     ELSEIF (PARNAME.EQ.'M3'.OR.PARNAME.EQ.'FN33') THEN
        FN3(I,3) = PARVAL
     ELSEIF (PARNAME.EQ.'K4'.OR.PARNAME.EQ.'FN14') THEN
        FN1(I,4) = PARVAL
     ELSEIF (PARNAME.EQ.'L4'.OR.PARNAME.EQ.'FN24') THEN
        FN2(I,4) = PARVAL
     ELSEIF (PARNAME.EQ.'M4'.OR.PARNAME.EQ.'FN34') THEN
        FN3(I,4) = PARVAL
     ELSE
        CALL WRNDIE(0,'<READPAR>','Unknown parameter')
     ENDIF
  End do           ! outer while
40 continue
  WRITE(OUTU,*)' '
  CLOSE(PUNIT)
  return

100 CONTINUE
  CALL WRNDIE(0,'<READPAR>','Error reading parameters from file')
  if(PRNLEV.ge.2) WRITE(OUTU,*)' '
  CLOSE(PUNIT)

  RETURN
END SUBROUTINE READPAR

!  DTM Routine added to compute derived parameters (J.Comp.Aid.Mol.Des. Vol.4 1990 p.18-21)
!  Based on mopac509mn
SUBROUTINE CALCPAR

  use chm_kinds
  use dimens_fcm
  use consta, only : BOHRR,EV_TO_KCAL,AU_TO_EV
  use number
  use am1parm
  use stream
  implicit none

  INTEGER I,J,JMAX
  real(chm_real)  P,P2,P4,D1,D2,D3,DF,Q1,Q2,Q3,QF,QN,GDD1,GQQ,HSP1,HSP2,HPP,HPP1,HPP2
  real(chm_real),parameter :: r_AU_TO_EV=one/AU_TO_EV, &
       r_twenty=one/TWENTY
  real(chm_real) :: temp1,temp2,temp3,temp4,r_p2,r_p4,r_sq_three
  integer, parameter :: crl = chm_real

  !
  ! THE CONTINUATION LINES INDICATE THE PRINCIPAL QUANTUM NUMBER.
  ! Link atom added at position 0 for all arrays
  !
  INTEGER :: NSPQN(0:100)=(/1,1,1,(2,i=1,8),(3,i=1,8),(4,i=1,18),(5,i=1,18), &
       (6,i=1,32),(0,i=1,14)/)
  ! Number of valence S electrons 
  real(chm_real) :: USSC(0:100)=(/ &
       one, &
       one,                                                           zero, &
       one,                                               (two,i=1,6),zero, &
       one,                                               (two,i=1,6),zero, &
       one,two,two,two,two,one,two,two,two,two,one,                       (two,i=1,6),zero, &
       one,two,two,two,one,one,two,one,one,zero,one,           (two,i=1,6),zero, &
       one,(two,i=1,22),one,one,                             (two,i=1,6),zero, &
       (zero,i=1,14)/)
  ! Number of valence P electrons
  real(chm_real) :: UPPC(0:100)=(/ &
       (zero ,i=1,3), &
       (zero ,i=1,2),one,two,three,four,five,six, &
       (zero ,i=1,2),one,two,three,four,five,six, &
       (zero,i=1,12),one,two,three,four,five,six, &
       (zero,i=1,12),one,two,three,four,five,six, &
       (zero,i=1,26),one,two,three,four,five,six, &
       (zero,i=1,14)/)
  ! Number of valence D electrons
  real(chm_real) :: UDDC(0:100)=(/(zero,i=1,19), &
       (zero,i=1,2),one,two,three,five,five,six,seven,eight,ten,ten,(zero,i=1,6), &
       (zero,i=1,2),one,two,four,five,five,seven,eight,ten,ten,ten,(zero,i=1,6), &
       (zero,i=1,2),one,(zero,i=1,6),one,(zero,i=1,6),one,two,three,four, &
       five,six,seven,nine,ten,ten,(zero,i=1,6), &
       (zero,i=1,14)/)
  ! Number of SS repulsion integrals
  real(chm_real) :: GSSC(0:100)=(/zero,zero,zero, &
       zero,(one,i=1,6),zero, &
       zero,(one,i=1,6),zero, &
       zero,(one,i=1,4),zero,(one,i=1,4),zero,(one,i=1,6),zero, &
       zero,(one,i=1,3),(zero,i=1,7),(one,i=1,6),zero, &
       zero,(one,i=1,22),one,one,(one,i=1,6),zero, &
       (zero,i=1,14)/)
  ! Number of SP repulsion integrals
  real(chm_real) :: GSPC(0:100)=(/zero,zero,zero, &
       (zero,i=1,2 ),two,four,six,eight,one,zero, &
       (zero,i=1,2 ),two,four,six,eight,one,zero, &
       (zero,i=1,12),two,four,six,eight,one,zero, &
       (zero,i=1,12),two,four,six,eight,one,zero, &
       (zero,i=1,26),two,four,six,eight,one,zero, &
       (zero,i=1,14)/)
  ! Number of SP exchange integrals
  real(chm_real) :: HSPC(0:100)=(/zero,zero,zero, &
       (zero,i=1,2 ),minone,mintwo,-3._crl,-4._crl,-5._crl,zero, &
       (zero,i=1,2 ),minone,mintwo,-3._crl,-4._crl,-5._crl,zero, &
       (zero,i=1,12),minone,mintwo,-3._crl,-4._crl,-5._crl,zero, &
       (zero,i=1,12),minone,mintwo,-3._crl,-4._crl,-5._crl,zero, &
       (zero,i=1,26),minone,mintwo,-3._crl,-4._crl,-5._crl,zero, &
       (zero,i=1,14)/)
  ! Number of PP' repulsion integrals
  real(chm_real) :: GP2C(0:100)=(/zero,zero,zero, &
       (zero,i=1,3 ),1.5_crl,4.5_crl,6.5_crl,one,zero, &
       (zero,i=1,3 ),1.5_crl,4.5_crl,6.5_crl,one,zero, &
       (zero,i=1,13),1.5_crl,4.5_crl,6.5_crl,one,zero, &
       (zero,i=1,13),1.5_crl,4.5_crl,6.5_crl,one,zero, &
       (zero,i=1,27),1.5_crl,4.5_crl,6.5_crl,one,zero, &
       (zero,i=1,14)/)
  ! Number of PP repulsion integrals
  real(chm_real) :: GPPC(0:100)=(/zero,zero,zero, &
       (zero,i=1,3 ),-0.5_crl,-1.5_crl,-0.5_crl,zero,zero, &
       (zero,i=1,3 ),-0.5_crl,-1.5_crl,-0.5_crl,zero,zero, &
       (zero,i=1,13),-0.5_crl,-1.5_crl,-0.5_crl,zero,zero, &
       (zero,i=1,13),-0.5_crl,-1.5_crl,-0.5_crl,zero,zero, &
       (zero,i=1,27),-0.5_crl,-1.5_crl,-0.5_crl,zero,zero, &
       (zero,i=1,14)/)
  ! Number of SD repulsion integrals
  real(chm_real) :: GSDC(0:100)=(/ (zero,i=1,19), &
       zero,zero,two,four,six,five,one,12._crl,14._crl,16._crl,one,(zero,i=1,7), &
       zero,zero,two,four,four,five,six,seven,eight,zero,ten, (zero,i=1,7), &
       zero,zero,two,(zero,i=1,6),two,(zero,i=1,6),two,four,six,eight,one,12._crl, &
       14._crl,nine,one,(zero,i=1,7), &
       (zero,i=1,14)/)
  ! Number of DD repulsion integrals
  real(chm_real) :: GDDC(0:100)=(/(zero,i=1,19), &
       (zero,i=1,3 ),one,three,one,one,15._crl,21._crl,28._crl,   (zero,i=1,8), &
       (zero,i=1,3 ),one,six,one,15._crl,21._crl,28._crl,45._crl, (zero,i=1,8), &
       (zero,i=1,17),one,three, six,one,15._crl,21._crl,36._crl,(zero,i=1,8), &
       (zero,i=1,14)/)
  !
  !  The DATA block shown above is derived from the ground-state atomic
  !  configuration of the elements.  In checking it, pay careful attention
  !  to the actual ground-state configuration. Note also that there are no
  !  configurations which have both p and d electrons in the valence shell
  !
  if(prnlev.ge.2) then
     WRITE(OUTU,'(A)') ' CALCPAR> Calculating EISOL,DD,QQ,AM,AD,AQ ...' 
     WRITE(OUTU,*)' '
  end if
  !     SET SCALING PARAMETER.
  r_sq_three=one/SQRT(three)
  P=two
  P2=P*P
  P4=P**4
  r_p2=one/P2
  r_p4=one/P4
  loopII: do I=2,100
     IF(ZP(I).LT.1.D-4.AND.ZS(I).LT.1.D-4) cycle loopII  !====GOTO 30
     !**********************************************************************
     !*
     !*   CONSTRAINTS ON THE POSSIBLE VALUES OF PARAMETERS
     !*
     !**********************************************************************
     IF(ZP(I).LT.0.3D0) ZP(I)=0.3D0
     !  PUT IN ANY CONSTRAINTS AT THIS POINT
     !**********************************************************************
     HPP=half*(GPP(I)-GP2(I))
     HPP=MAX(PTONE,HPP)
     HSP(I)=MAX(1.D-7,HSP(I))
     EISOL(I)=USS(I)*USSC(I)+UPP(I)*UPPC(I)+UDD(I)*UDDC(I)+ &
          GSS(I)*GSSC(I)+GPP(I)*GPPC(I)+GSP(I)*GSPC(I)+ &
          GP2(I)*GP2C(I)+HSP(I)*HSPC(I)+GSD(I)*GSDC(I)+ &
          GDD(I)*GDDC(I)
     QN=NSPQN(I)
     DD(I)=(two*QN+1)*(four*ZS(I)*ZP(I))**(QN+half)/(ZS(I)+ZP(I))**(two*QN+2)*r_sq_three ! /SQRT(three)
     QQ(I)=SQRT((four*QN*QN+six*QN+two)*r_twenty)/ZP(I)   ! /TWENTY
     !     CALCULATE ADDITIVE TERMS, IN ATOMIC UNITS.
     JMAX=5
     GDD1= (P2*HSP(I)/(AU_TO_EV*four*DD(I)**2))**(THIRD)   ! thrid=one/three
     GQQ = (P4*HPP/(AU_TO_EV*48.D0*QQ(I)**4))**0.2D0
     D1=GDD1
     D2=GDD1+0.04D0
     Q1=GQQ
     Q2=GQQ+0.04D0
     do J=1,JMAX
        DF=D2-D1
        HSP1= (two*D1 - two/SQRT(four*DD(I)**2+one/D1**2))*r_p2      ! /P2
        HSP2= (two*D2 - two/SQRT(four*DD(I)**2+one/D2**2))*r_p2      ! /P2
        D3= D1 + DF*(HSP(I)*r_AU_TO_EV-HSP1)/(HSP2-HSP1)    ! /AU_TO_EV
        D1= D2
        D2= D3
     end do
     do J=1,JMAX
        QF=Q2-Q1
        HPP1=(four*Q1-eight/SQRT(four*QQ(I)**2+one/Q1**2)+four/SQRT(eight*QQ(I)**2+one/Q1**2))*r_p4    ! /P4
        HPP2=(four*Q2-eight/SQRT(four*QQ(I)**2+one/Q2**2)+four/SQRT(eight*QQ(I)**2+one/Q2**2))*r_p4    ! /P4
        Q3= Q1 + QF*(HPP*r_AU_TO_EV-HPP1)/(HPP2-HPP1)         ! /AU_TO_EV
        Q1= Q2
        Q2= Q3
     end do
     bdd(I,1)= GSS(I)*r_AU_TO_EV       ! /AU_TO_EV
     bdd(I,2)= D2
     bdd(I,3)= Q2
  end do loopII
  EISOL(0)=USS(0)
  bdd(0,1)=GSS(0)*r_AU_TO_EV       ! /AU_TO_EV
  bdd(0,2)=bdd(0,1)
  bdd(0,3)=bdd(0,1)
  EISOL(1)=USS(1)
  bdd(1,1)=GSS(1)*r_AU_TO_EV       ! /AU_TO_EV
  bdd(1,2)=bdd(1,1)
  bdd(1,3)=bdd(1,1)

  RETURN
END SUBROUTINE CALCPAR

!  DTM Routine added to transfer parameters from node 0 to N others
!  MH05 replaced direct calls to MPI with PSND8() to fix
!       compilation problems
#if KEY_PARALLEL==1 /*scatpar*/
SUBROUTINE SCATPAR

  use chm_kinds
  use dimens_fcm
  use am1parm
  use stream
  use parallel
  implicit none

  INTEGER I,IERROR,STATUS

  integer, parameter :: IHND=100, ITHD=1000
  !
  CALL PSND8(ALFA ,IHND)
  CALL PSND8(EISOL,IHND)
  CALL PSND8(BETAS,IHND)
  CALL PSND8(BETAP,IHND)
  CALL PSND8(BETAD,IHND)
  CALL PSND8(ZS   ,IHND)
  CALL PSND8(ZP   ,IHND)
  CALL PSND8(ZD   ,IHND)
  CALL PSND8(DD   ,IHND)
  CALL PSND8(QQ   ,IHND)
  CALL PSND8m(bdd  ,3*IHND)   ! CALL PSND8(AM   ,IHND) / CALL PSND8(AD   ,IHND) / CALL PSND8(AQ   ,IHND)
  CALL PSND8(USS  ,IHND)
  CALL PSND8(UPP  ,IHND)
  CALL PSND8(UDD  ,IHND)
  CALL PSND8(GSS  ,IHND)
  CALL PSND8(GPP  ,IHND)
  CALL PSND8(GSP  ,IHND)
  CALL PSND8(GP2  ,IHND)
  CALL PSND8(HSP  ,IHND)
  CALL PSND8(VS   ,IHND)
  CALL PSND8(VP   ,IHND)
  CALL PSND8m(FN1  ,ITHD) ! double array - ITHD maybe wrong number ??
  CALL PSND8m(FN2  ,ITHD)
  CALL PSND8m(FN3  ,ITHD)
  !
  RETURN
END SUBROUTINE SCATPAR
#endif /* (scatpar)*/
!----------------------------------------------------------------------
#endif /* (q2)*/

