module gukini_mod
  use chm_kinds
  implicit none

  integer gukini_dummy_integer

  ! we should bring some variables from ltm/gamess_ltm.src
  ! some are also needed by other than ab initio interfaces
  ! so they may need to stay there ???
  !
  ! for ewald, see FIXME_EWALD
  !

#if KEY_NWCHEM==1
  integer nwchem_nat_mm, nwchem_nat_qm
  real(chm_real),allocatable,dimension(:) :: nwchem_znuc
#endif

contains

  SUBROUTINE GUKINI(COMLYN,COMLEN)
    !-----------------------------------------------------------------------
    !     Define the set of quantum mechanical atoms and set up the data
    !     structures.
    !
    !     April 1993,   Milan Hodoscek,  Pure QM part
    !     October 1993,   -- " --        QM/MM interface
    !     July 2001,      -- " --     ,
    !                and Paul Sherwood,  QM/MM Replica PATH
    !                                    Combine gamini + gukini -> gukini
    !
    !     July 2003,    Milan Hodoscek,  support non-interacting QM regions
    !     May  2004,    H. Lee Woodcock, Finshed Q-Chem interface and added 
    !                                    support for replica path and NEB 
    !                                    with Q-Chem. 
    !
    !     March 2006,   Milan Hodoscek   include ewald (Q-Chem, GAMESS, GAMESSUK)
    !
    !     May   2012,   Chris Rowley     Interface to TURBOMOLE (crowley@mun.ca)
    !
    !     Currently there is no cutoff method incorporated, but this should
    !     be straightforward. No big loss since Natom(QM) is usually small so 
    !     Natom(QM)*Natom(MM)/2 is also small. Also there shouldn't be any 
    !     inconsistency with MM cutoff methods, since these are different
    !     terms anyway. 
    !
    !     Please beware that the nature of the charmm/gamess-uk
    !     Interface has changed frequently
    !     This code implements version 4 of that interface
    !
    !     This interface now works for GAMESS, GAMESS-UK and Q-Chem
    !     Other QM packages could be supported, too.
    !
    !
    use dimens_fcm
    use memory
    use code
    use coord
    use psf
    use select
    use storage
    use stream
    use string
    use energym
    use replica_mod
    use pathm
    use gamess_fcm
    use block_fcm
    use number
    use parallel
    use scalar_module
    use repdstr
    use ewald_1m,only:lewald
    ! include for ewald stuff - it is properly protected for ab initio
    use mndo97
#if KEY_QCHEM==1 || KEY_QTURBO==1
use pert
INTEGER SCRATCHLEN,K
CHARACTER(len=1), parameter :: QUOTE = ''
CHARACTER(len=255) QCSDIR
CHARACTER(len=4) QCSDIRNUM
LOGICAL dir_e,QRESTART2
#endif
!
integer,allocatable,dimension(:) :: islct
!
#if KEY_SQUANTM==1
    LOGICAL LQMEWD
#endif 
    !
    character(len=240)   name
    integer kstrm,koutu
    COMMON /CHGMIO/  KSTRM,KOUTU,name
    !
    CHARACTER COMLYN*(*)
    INTEGER   COMLEN
    !
    LOGICAL   QQINP
    INTEGER KFRAG,JFRAG
    !CCC  test code...
    logical mopac
    common /mpctst/mopac
    !
    character(len=200) tmpqms
#if KEY_REPLICA==1 /*replica*/
#if KEY_PARALLEL==1
    INTEGER IDSTNC(MAXNODE)         
#endif
    INTEGER J,IREPL
#endif /* (replica)*/
    INTEGER I,IPT
#if KEY_GAMESSUK==1
    INTEGER INIT, ICODE,IVER     
#endif
#if KEY_GAMESSUK==1
    LOGICAL STARTUP              
#endif
!
CHARACTER CFN*10,CRN*6
CHARACTER(LEN=10) WRD
INTEGER NQMPMEt1,NQMPMEt2,XI,YI,ZI
INTEGER BASLEN1,BASLEN2
LOGICAL QDONE,LDUM
!
#if KEY_CADPAC==1 /*cadpac*/
    RETURN
  END SUBROUTINE GUKINI
#else /* (cadpac)*/
#if KEY_GAMESS==0 && KEY_GAMESSUK==0 && KEY_QCHEM==0 && KEY_QTURBO==0 && KEY_G09==0 && KEY_NWCHEM==0 /*guk_main*/
    CALL WRNDIE(-1,'<GUKINI>','Ab initio QM/MM code not compiled.')
    RETURN
  END SUBROUTINE GUKINI

#else /* (guk_main)*/
    !
    CFN='gukini.src'
    CRN='GUKINI'

    !  to disable QM/MM-Ewald for Multi-layered qm/mm compilation.
#if KEY_SQUANTM==1
    LQMEWD = .false.
    LEWMODE= .false.
#endif 

    !
    !  Disable interface to switch back to MM only
    !  Does not restore MM charges to QM atoms
    !
    IF(INDXA(COMLYN,COMLEN,'OFF').GT.0)THEN
       IF(PRNLEV.GE.2) THEN
          WRITE(OUTU,22)'OFF:  GAMESS Interface Disabled'
       ENDIF
       !         CALL RSTCOPSEL
       QGMREM=.FALSE.
       NGAMES=0
#if KEY_GAMESS==1 || KEY_GAMESSUK==1
#if KEY_SQUANTM==1
       QMLAY_high=.FALSE.            
#endif
#endif 
       RETURN
    ENDIF

#if KEY_QCHEM==1
    QRESTART2=.false.
#endif 

    qmused = .true.      ! safe here since next are the allocations, ...
    call allocate_gamess ! try to reduce from MAXA to NATOM

    call chmalloc('gukini.src','GUKINI','ISLCT',NATOM,intg=ISLCT)
    CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
    WRITE(OUTU,'(A)')' ' 

#if KEY_QCHEM==1
    QMIXED=(INDXA(COMLYN,COMLEN,'MIX').GT.0)
    IF(QMIXED) THEN 
!      BASIS1=(INDXA(COMLYN,COMLEN,'BAS1').GT.0)
       CALL GTRMWD(COMLYN,COMLEN,'BAS1',4,BASIS1,10,BASLEN1)
       CALL GTRMWD(COMLYN,COMLEN,'BAS2',4,BASIS2,10,BASLEN2)
!      write(*,*)'Basis sets... ', BASIS1, BASIS2 
       WRITE(OUTU,22)'Q-Chem will use a mixed basis set; basis1 and basis2 must be defined.'
       WRITE(OUTU,'(A,A,A)')' QCHEM> The following basis sets will be used: ', BASIS1, BASIS2
    ENDIF 
#endif 

    QGMREM=(INDXA(COMLYN,COMLEN,'REMO').GT.0)
    IF(PRNLEV.GE.2) THEN
       IF(QGMREM) THEN
          WRITE(OUTU,22) &
               'REMOve: Classical energies within QM atoms are removed.'
       ELSE
          WRITE(OUTU,22) &
               'No REMOve: Classical energies within QM atoms are retained.'
       ENDIF
    ENDIF
    !
    QGMEXG=(INDXA(COMLYN,COMLEN,'EXGR').GT.0)
    QGMDIV=(INDXA(COMLYN,COMLEN,'DIV').GT.0) ! for DIV - guanhua_puja_QC_UW1212 
    IF(PRNLEV.GE.2) THEN
       IF(QGMEXG) THEN
          WRITE(OUTU,22) &
               'EXGRoup: QM/MM Electrostatics for link host groups removed.'
       ELSE IF (QGMDIV) THEN
          WRITE(OUTU,22) &
               'Using DIV scheme: using standard SLA scheme for exclusions.'
       ELSE
          WRITE(OUTU,22) &
               'No EXGRoup: QM/MM Elec. for link atom host only is removed.'
       ENDIF
    ENDIF
    !
    QBLUCH=(INDXA(COMLYN,COMLEN,'BLUR').GT.0)
    IF(QBLUCH) THEN
       IF(PRNLEV.GE.2) THEN
          WRITE(OUTU,22) &
               'BLUR: Blurred charges will be used on some atoms.'

#if KEY_GAMESSUK==0
          RECALLINT=GTRMI(COMLYN,COMLEN,'RECA',-1)
          WRITE(OUTU,22)'You are using the RECALL option.'

          if(recallint.ne.-1)then
             IF(ptrsto(RECALLINT)%len.NE.NATOM)THEN
                WRITE(OUTU,22)'Blur array is not equal to the number of, &
                     atoms'
             ENDIF
          endif
#endif 

       ENDIF
    ENDIF
    !
#if KEY_QCHEM==1
    QRESET=(INDXA(COMLYN,COMLEN,'RESE').GT.0)
    IF(QRESET) THEN 
       IF(PRNLEV.GE.2) THEN 
          WRITE(OUTU,22) &
               'RESET: Resetting Q-Chem options' 
       ENDIF
       QQCOORD=.FALSE.      ! Pass Coordinates from Q-Chem back to CHARMM
       QRESTART=.FALSE.     ! Determine if Q-Chem restarts first step from saved orbitals
       QSAVORB=.FALSE.      ! Save orbitals from Q-Chem calculation
       QQCLJ=.FALSE.        ! Use Lennard-Jones parameters in Q-Chem calculation
       QMICRO=.FALSE.       ! Controls Micro-iteration procedure
       QFRSTMICIT=.false.   ! First micro-iteration step?
       OPTCNT=0             ! Counts QM/MM optimization steps
       QCPARA=-1            ! Reset the Parallel options 
       QSAVEHESS=.false.    ! Reset Hessian saving option
       QREADHESS=.false.    ! Reset Hessian reading option 
       QSAVEGRAD=.false.    ! Reset Gradient saving option
       QREADGRAD=.false.    ! Reset Gradient reading option
       QSAVEINP=.false.     ! Save Q-Chem input files
       QSAVEOUT=.false.     ! Save Q-Chem output files
       QPCM=.false.         ! Turn off QM/MM/PCMa
       QWRITeinp=.false.    ! Force Q-Chem to write input file and then quit
       NREStart=0           ! Number of times to restart Q-Chem if a job fails 
       QQEWALD=.false.      ! Not using John Herbert's QM/MM Ewald
       QMESS=.false.        ! Not using MESS QM/MM energy estimation
       NRoots=0             ! Number of roots for MESS-H
       QMALpha=ZERO         ! alpha / kappa for QM part of QM/MM ewald 
       MMALpha=ZERO         ! alpha / kappa for MM part of QM/MM ewald
       QQNOLKATMS=.false.   ! Check if we explicitly exclude link atoms from Q-Chem
    ENDIF
#endif 

#if KEY_QCHEM==1 || KEY_QTURBO==1

#if KEY_PERT == 1
    QQMPERT=(INDXA(COMLYN,COMLEN,'PERT').GT.0)
#endif /* KEY_PERT */
    
#if KEY_REPLICA==1 && KEY_RPATH==1
    QQCRP=(INDXA(COMLYN,COMLEN,'RPATH').GT.0)
    IF(QQCRP) THEN
       IF(PRNLEV.GE.2) THEN
          WRITE(OUTU,22) &
               'QCHEM: Using Q-Chem to do Replica Path Calculation.'
       ENDIF
       QRPMINP=(INDXA(COMLYN,COMLEN,'MINP').GT.0)
        IF(PRNLEV.GE.2) THEN
            WRITE(OUTU,22) &
                 'QCHEM: Using Different Q-Chem Control Files for Replicas.'
        ENDIF
    ENDIF
#endif /* KEY_REPLICA==1 && KEY_RPATH==1 */
    
#endif /* KEY_QCHEM==1 || KEY_QTURBO==1 */

#if KEY_QCHEM==1
    QQCHARG=(INDXA(COMLYN,COMLEN,'CHAR').GT.0)
    IF(QQCHARG) THEN 
       !write(*,*)'QFRSTMICIT = ',QFRSTMICIT
       IF(PRNLEV.GE.2) THEN
          WRITE(OUTU,22) &
               'CHARGES: Reading QM Charges from charges.dat file' 
       ENDIF
    ENDIF
#endif /* */
    !
#if KEY_QCHEM==1
    QNRAP=(INDXA(COMLYN,COMLEN,'NRAP').GT.0)
    !IF(QNRAP) QSECD=.TRUE.
#endif /* */
    !
#if KEY_QCHEM==1
    QCPARA=GTRMI(COMLYN,COMLEN,'PARA',-1)
#if KEY_PARALLEL==1
    CALL PSND4(QCPARA,1)                    
#endif
    IF(QCPARA /= -1) THEN 
       IF(QCPARA .GT. NUMNOD) THEN 
#if KEY_PARALLEL==1
          IF(.not.QQCRP) THEN 
            IF(NUMNOD .GT. 1) QCPARA=NUMNOD   
          ENDIF 
#endif
       ENDIF
       IF(PRNLEV.GE.2) THEN
          WRITE(OUTU,'(A,I3,A)') & 
               ' QCHEM> PARALLEL: Q-Chem will be run on ',QCPARA, &
               ' processors.'
       ENDIF
    ENDIF
    NREStart=GTRMI(COMLYN,COMLEN,'NRES',0)
    IF(PRNLEV.GE.2) THEN 
       WRITE(OUTU,'(A,I3,A)') & 
            ' QCHEM> Q-Chem will restart jobs ',NREStart,' times.'
    ENDIF
    NROOts=GTRMI(COMLYN,COMLEN,'NROO',0)
    IF(PRNLEV.GE.2) THEN
       WRITE(OUTU,'(A,I3,A)') & 
            ' QCHEM> Q-Chem will solve ',NROOts,' roots.'
!      WRITE(OUTU,'(A38,I3,A13)') &
!           'QCHEM> Q-Chem will solve ',NROOts,' roots.'
    ENDIF
#endif /*  */
    !
#if KEY_QCHEM==1
    QQCOORD=(INDXA(COMLYN,COMLEN,'COOR').GT.0)
    IF(QQCOORD) THEN 
       IF(prnlev.ge.2) then
          write(outu,22) &
               'Coordinates from Q-Chem will be passed back to CHARMM'
       ENDIF
    ENDIF
    QQRESD=(INDXA(COMLYN,COMLEN,'RESD').GT.0)
    IF(QQRESD) THEN 
       IF(prnlev.ge.2) then
          write(outu,22) &
               'RESDistance information from CHARMM will be sent to Q-Chem'
       ENDIF
       QQCONS=(INDXA(COMLYN,COMLEN,'CONS').GT.0)
       IF(prnlev.ge.2) then
          write(outu,22) &
               'Only use single CONStraint for distance'
       ENDIF
    ENDIF
#endif 
    ! 
#if KEY_QCHEM==1
    ! Test to see if Hessian is gonna be read in 
    QREADHESS=(INDXA(COMLYN,COMLEN,'RHES').GT.0)
    QSAVEHESS=(INDXA(COMLYN,COMLEN,'SHES').GT.0)
    ! Test to see if Gradient is going to be saved
    QSAVEGRAD=(INDXA(COMLYN,COMLEN,'SGRA').GT.0)
    QREADGRAD=(INDXA(COMLYN,COMLEN,'RGRA').GT.0)
    ! Test to see if input/output files will be saved
    QSAVEINP=(INDXA(COMLYN,COMLEN,'SINP').GT.0)
    QSAVEOUT=(INDXA(COMLYN,COMLEN,'SOUT').GT.0)
    ! Test to see if PCM will be used with QM/MM
    QPCM=(INDXA(COMLYN,COMLEN,'PCM').GT.0)
    QWRITeinp=(INDXA(COMLYN,COMLEN,'WQIN').GT.0)
    QOPENMP=(INDXA(COMLYN,COMLEN,'OMP').GT.0)
    QQEWALD=(INDXA(COMLYN,COMLEN,'EWAL').GT.0)
    QMESS=(INDXA(COMLYN,COMLEN,'MESS').GT.0)
    ! Test to check for link atom exclusions
    QQNOLKATMS=(INDXA(COMLYN,COMLEN,'NOLI').GT.0)

!   write(*,*)'QQEWALD =',QQEWALD
    IF(QQEWALD) THEN 
      QMALpha=gtrmf_crl(COMLYN,COMLEN,'QMAL',ZERO)
      MMALpha=gtrmf_crl(COMLYN,COMLEN,'MMAL',ZERO)
      IF(PRNLEV.GE.2) THEN
        WRITE(OUTU,'(A,F6.3,A,F6.3)')' QCHEM> Q-Chem use, ',QMALpha,' and ',MMALpha, & 
        ' for the QM and MM part of EWALD respectively'
      ENDIF
    ENDIF
!   write(*,*)'QQEWALD =',QMALpha,MMALpha
#endif  
    !
#if KEY_QCHEM==1
    QMICRO=(INDXA(COMLYN,COMLEN,'MICR').GT.0)
    IF(QMICRO) THEN 
       QFRSTMICIT=.true.
       OPTCNT=0
       IF(prnlev.ge.2) THEN
          WRITE(OUTU,22) &
               'MICRO: Micro-iteration Procedure Activated'
          !write(*,*)'QFRSTMICIT = ',QFRSTMICIT
       ENDIF
    ENDIF
#endif 
    !
    QCUTFLAG=(INDXA(COMLYN,COMLEN,'CUTO').GT.0)
    IF(QCUTFLAG) THEN
       IF(PRNLEV.GE.2) THEN
          WRITE(OUTU,22) &
               'CUTOFF: the cutoff method from NBOND will be used.'
       ENDIF
    ENDIF
    !

#if KEY_QCHEM==1
!     IF(.not. QRESTART) THEN 
      CALL get_environment_variable("QCSCRATCH", QCSDIR)
      !write(*,*)'QCSDIR =',QCSDIR
      do i=1,10000
        write (QCSDIRNUM,'(I3)') i
        QCSDIRNUM = adjustl(QCSDIRNUM)
        j = LEN(TRIM(QCSDIR))
        QCSCRATCH='SAVE'
        QCSCRATCH = QCSCRATCH(1:4)//trim(QCSDIRNUM)
        !write(*,*)QCSCRATCH,QCSDIRNUM
        k = len(trim(QCSCRATCH))
        QCSCRATCH = QCSDIR(1:j)//QCSCRATCH(1:k)
        !write(*,*)QCSCRATCH(1:k),LEN(QCSCRATCH)
        inquire(file=QCSCRATCH, exist=dir_e)
        if(.not. dir_e) then 
          QCSCRATCH='SAVE'//trim(QCSDIRNUM)
          QRESTART=(INDXA(COMLYN,COMLEN,'REST').GT.0)
          IF(QRESET .or. QRESTART) THEN 
            QCSCRATCH=QCSCRATCHOLD
            QRESTART2=.true. 
            exit 
          ENDIF 
          QCSCRATCHOLD=QCSCRATCH
          !k = len(trim(QCSCRATCH))
          !QQCSCRATCH=.true.
          !IF(PRNLEV.GE.2) THEN 
          !  WRITE(OUTU,'(A45,A)')'QCHEM> Q-Chem scratch files will be saved in:',QCSDIR(1:j)//QCSCRATCH(1:k)
          !ENDIF
          exit 
        endif  
      enddo 
!     ENDIF 
      !write(*,*)QCSCRATCH(1:k),LEN(QCSCRATCH),'SAVE'//trim(QCSDIRNUM),QCSCRATCHOLD(1:k)
#endif 

#if KEY_QCHEM==1
    SCRATCHLEN=255
    CALL GTRMWA(COMLYN,COMLEN,'SCRA',4,QCSCRATCH,255,SCRATCHLEN)
    IF(SCRATCHLEN.ne.0) QQCSCRATCH=.true.
    IF(QQCSCRATCH) THEN 
      IF(PRNLEV.GE.2) THEN 
        WRITE(OUTU,22)'User defined scratch directory will be used to save Q-Chem files' 
      ENDIF
    ENDIF 
#endif 

!IF(.not.QQCSCRATCH) THEN 
!QSAVORB=(INDXA(COMLYN,COMLEN,'SAVE').GT.0)
!ENDIF 
#if KEY_QCHEM==1
    QSAVORB=.FALSE.
    QRESTART=.FALSE.
    QRESTART=(INDXA(COMLYN,COMLEN,'REST').GT.0)
    QRESTART=QRESTART2
    QSAVORB=(INDXA(COMLYN,COMLEN,'SAVE').GT.0)
    IF(QRESTART) THEN
       IF(PRNLEV.GE.2) THEN 
          WRITE(OUTU,22)'Using previously saved orbitals to restart'
          WRITE(OUTU,'(A41,A)')'QCHEM> Q-Chem orbitals will be read from:',QCSDIR(1:j)//QCSCRATCH(1:k)
       ENDIF
       QSAVORB=.FALSE.
       QQCSCRATCH=.true.
       OPTCNT=0
       QRESTART2=.false. 
       QRESTART=.true.
    ENDIF
    IF(QSAVORB) THEN 
       IF(PRNLEV.GE.2) THEN 
          WRITE(OUTU,22) &
               'Orbitals from current QM/MM calculation will be saved'
          WRITE(OUTU,'(A41,A)')'QCHEM> Q-Chem orbitals will be saved in:',QCSDIR(1:j)//QCSCRATCH(1:k)
       ENDIF
       QRESTART=.FALSE.
       QQCSCRATCH=.true.
    ENDIF

    QQCLJ=.false.
    QQCLJ=(INDXA(COMLYN,COMLEN,'QCLJ').GT.0)
    IF(QQCLJ) THEN 
       IF(PRNLEV.GE.2) THEN
          WRITE(OUTU,22) &
               'Lennard-Jones parameters will be applied to QM calculation'
       ENDIF
    ENDIF
#endif /* */

#if KEY_GAMESS==1 || KEY_QCHEM==1
    QNOGU=(INDXA(COMLYN,COMLEN,'NOGU').GT.0)
    IF(QNOGU) THEN
       KGUES=0
       KHFDFT=0
       IF(PRNLEV.GE.2) THEN
          WRITE(OUTU,22) &
               'NOGUess: Initial guess obtained from previous step.'
       ENDIF
#if KEY_QCHEM==1
       k = len(trim(QCSCRATCH))
       QQCSCRATCH=.true.
       IF(PRNLEV.GE.2) THEN 
         WRITE(OUTU,'(A45,A)')' QCHEM> Q-Chem scratch files will be saved in:',QCSDIR(1:j)//QCSCRATCH(1:k)
    ENDIF
#endif 
    ENDIF
#endif 

    !
#if KEY_GAMESS==1
    QFMO=(INDXA(COMLYN,COMLEN,'FMO').GT.0)
    IF(QFMO) THEN
       IF(PRNLEV.GE.2) THEN
          WRITE(OUTU,22) &
               'FMO: Fragment MO method specified.'
       ENDIF
    ENDIF
    KDIESL=GTRMI(COMLYN,COMLEN,'DIES',-1)
    IF(KDIESL.GE.0) THEN
       IF(PRNLEV.GE.2) THEN
          WRITE(OUTU,22) &
               'DIESel: Multi Reference CI energy calculation.'
       ENDIF
    ENDIF
#endif
    !
    QFRAG=.FALSE.
    NFRAG=GTRMI(COMLYN,COMLEN,'FRAG',0)
    IF(NFRAG.NE.0)THEN
       QXFRAG=.FALSE.
       IF(NFRAG.LT.0)THEN
          QXFRAG=.TRUE.
          NFRAG=-NFRAG
       ENDIF
       QFRAG=.TRUE.
       IFRAG=IFRAG+1
       IF(IFRAG.GT.NFRAG)CALL WRNDIE(-1,'<GUKINI>', &
            'Too many fragments.')
    ENDIF

    QQINP=(INDXA(COMLYN,COMLEN,'QINP').GT.0)
    IF(PRNLEV.GE.2) THEN
       IF(QQINP) THEN
          WRITE(OUTU,22) &
               'QINP: Charges will input for QM atoms.'
       ELSE
          WRITE(OUTU,22) &
               'No QINP: Charges will be based on atomic numbers.'
       ENDIF
    ENDIF
    MOPAC=(INDXA(COMLYN,COMLEN,'MOPAC').GT.0)
    IF(PRNLEV.GE.2) THEN
       IF(MOPAC) THEN
          WRITE(OUTU,22) &
               'MOPAC: A MOPAC calculation will be performed.'
       ENDIF
    ENDIF
    WRITE(OUTU,'(A)')' ' 
#if KEY_G09==1
22  FORMAT('G09> ',A)
#elif KEY_NWCHEM==1
22  FORMAT(' NWChem> ',A)
#elif KEY_QCHEM==1
22  FORMAT(' QCHEM> ',A)
#elif KEY_QTURBO==1
22  FORMAT('QTURBO> ', A)
#else /* */
22  FORMAT('GUKINT> ',A)
#endif 
    !
#if KEY_REPDSTR==1
#if KEY_GAMESS==1
    IF (QREPDSTR) CALL ENVIGRP(IREPDSTR+1)
    !50+      write(50+mynodg,*)'GUKINI-0>me,meg,np,npg=',
    !50+     $     mynod,mynodg,numnod,numnodg
#endif 
#endif 

    ! we prefer doing PME with SQUANTUM, but no SQUANTM flag specified (yet)
!!ccc##IFN SQUANTM (nosquantm)
#if KEY_MNDO97_old==1 /*nosquantm*/
    !
    !
    ! This is taken from mndint/mndini.src
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
    ! NQMPME     : 0 Use Mulliken charges on QM atoms to represent charges on
    !                image atoms.
    !              1 Use input MM charges on QM atoms to represent charges on
    !                image atoms.
    IF(LEWALD.AND.(.NOT.QGMREM)) CALL WRNDIE(-1,'<MNDINI>', &
         'QM/MM-Ewald is not compatible without REMO.')

    LQMEWD=.FALSE.
    IF(LEWALD.AND.QGMREM) LQMEWD=.TRUE.

    IF(LQMEWD) THEN
       WRITE(outu,22) &
            'Ewald with QM/MM Option has been Activated.'
       LQMEWD2=LQMEWD
       EWMODE = 0
       NQMPMEt1=9999
       NQMPMEt2=9999
       NQMPME  =9999
       NQMPMEt1=GTRMI(COMLYN,COMLEN,'NPME',9999)
       NQMPMEt2=GTRMI(COMLYN,COMLEN,'NEWD',9999)
       IF(NQMPMEt1.NE.9999.OR.NQMPMEt2.NE.9999) THEN
          IF(NQMPMEt1.NE.9999) NQMPME = NQMPMEt1
          IF(NQMPMEt2.NE.9999) NQMPME = NQMPMEt2
       END IF
       !CCC          NQMPME=GTRMI(COMLYN,COMLEN,'NPME',9999)
       IF(NQMPME.EQ.1) THEN
          NQMPME = 1
          EWMODE = 0
          WRITE(outu,22) &
               'Ewald with QM/MM Option use Input MM charges on QM atoms'
       ELSE IF(NQMPME.EQ.9999.OR.NQMPME.EQ.2) THEN
          NQMPME = 2
          EWMODE = 0
          WRITE(outu,22) &
               'Default Ewald with QM/MM Option uses Mulliken Charges'
       ELSE IF(NQMPME.EQ.-2) THEN
          NQMPME = 2
          EWMODE = 2
          WRITE(outu,22) &
               'Default Ewald with QM/MM Option uses Mulliken Charges'
          WRITE(outu,22) &
               'MM atoms (with Ewald) polarize only diagonal elements in QM'
       ELSE IF(NQMPME.EQ.-1) THEN
          NQMPME = 2
          EWMODE = 1
          WRITE(outu,22) &
               'Default Ewald with QM/MM Option uses Mulliken Charges'
          WRITE(outu,22) &
               'MM within cutoff interact with regular way with QM'
          WRITE(outu,22) &
               'MM from Ewald Sum interact with diagonal elements in QM'
       END IF
       IF (EWMODE.EQ.0) WRITE(outu,22) &
            'MM in Ewald Sum do not polarize QM atoms'
       !
       !        Now setup for Ewald in QM/MM SCF
       LEWMODE = .FALSE.
       IF(LQMEWD.AND.EWMODE.GT.0) LEWMODE = .TRUE.

       IF(LEWMODE) THEN
          !           default values
          KMAXQ  = 5
          KSQMAXQ= 27
          !
          KMAXQ  =GTRMI(COMLYN,COMLEN,'KMAX',KMAXQ)
          KMAXXQ =GTRMI(COMLYN,COMLEN,'KMXX',KMAXQ)
          KMAXYQ =GTRMI(COMLYN,COMLEN,'KMXY',KMAXQ)
          KMAXZQ =GTRMI(COMLYN,COMLEN,'KMXZ',KMAXQ)
          KSQMAXQ=GTRMI(COMLYN,COMLEN,'KSQM',KSQMAXQ)
       END IF
    ELSE
       !     in the case of no Ewald with QM/MM
       EWMODE = -1
       NQMPME = -1
       LEWMODE = .FALSE.
    END IF
#endif /*  (nosquantm)*/
    !     end ewald
    !
    MUSTUP=.TRUE.
    !
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QCHEM==1 || KEY_G09==1 || KEY_NWCHEM==1 /*rpath*/
#if KEY_REPLICA==1
#if KEY_RPATH==1
    ! defaults for QM replica loop structure
    NPGRP      = 1                 ! number of parallel groups
    NREPQMNOD  = 1                 ! number of replicas on this node

#if KEY_PARALLEL==1 /*parallel*/
    !     save global node values
    !
    !CC this was not protected it is now in paral1.src
    !      NUMNODG=NUMNOD
    !      MYNODG=MYNOD
    !
    QQMPAR=.FALSE.
    IF(NUMNOD.GT.1)QQMPAR=.TRUE.
    !
    IF(QFRAG)THEN
       KFRAG=NUMNOD/NFRAG
       NUMNOD=KFRAG
       MYNOD=MOD(MYNODG,KFRAG)
       MYNODP=MYNOD+1
       NODDIM=NPTWO()
       JFRAG=MYNODG/KFRAG+1
       DO I=0,NUMNOD
          IPPMAP(I)=I+KFRAG*(JFRAG-1)
       ENDDO
       NUMNOD_save=NUMNOD
       MYNOD_save=MYNOD
    ENDIF
#endif /* (parallel)*/
#endif 
#endif 
    !
#if KEY_REPLICA==1 /*replica*/
#if KEY_RPATH==1
    IF(QREP) THEN
       !CC   WRITE(OUTU,*)'GAMINI>nsub,nrepl=',nsub,nrepl
       !CC   write(outu,*)'GAMINI>repnoa=',(repnoa(i),i=1,natom)
       !$$$  C     Count the number of replicas (We rely just on REPNOA array)
       !$$$  C     This code does not work OK, but for now we assume that
       !$$$  C     user deletes the original!
       !$$$  IPT=1
       !$$$  IDSTNC(IPT)=REPNOA(1)
       !$$$  NREPQM=0
       !$$$  IF(REPNOA(1).GT.1)NREPQM=1
       !$$$  DO I=2,NATOM
       !$$$  IF(REPNOA(I).EQ.1)GOTO 111
       !$$$  DO J=1,IPT
       !$$$  IF(IDSTNC(J).EQ.REPNOA(I))GOTO 111
       !$$$  ENDDO
       !$$$  IPT=IPT+1
       !$$$  IDSTNC(IPT)=REPNOA(I)
       !$$$  NREPQM=NREPQM+1
       !$$$  111        CONTINUE
       !$$$  ENDDO

       NREPQM=NREPL-1
       !     
       IF(NREPQM.GT.0)THEN
#if KEY_PARALLEL==1

          !     Determine parallel strategy
          !
          !       KREPQM    - number of nodes available per replica
          !       IREPQM    - first replica to compute in this node
          !       NREPQMNOD - number of replicas on this node
          !
          KREPQM=NUMNOD/NREPQM

          !     Some of this will be later replaced with
          !     the dynamic loadbalance for parallel/parallel

          if (KREPQM*NREPQM .EQ. NUMNOD) then
             !     
             !     exact divisibility - try parallel/parallel
             !     
             NPGRP     = NREPQM  ! number of parallel process groups
             NREPQMNOD = 1       ! number of gamess replicas for this node

             if(PRNLEV.GT.2 .AND. NUMNOD .GT. 1) THEN
                WRITE(OUTU,20) KREPQM
#if KEY_G09==1
20              FORMAT('G09> Parallel replicas activated, use ', &
                     i3,' procs per replica')
#elif KEY_QCHEM==1
20              FORMAT('QCHEM> Parallel replicas activated, use ', &
                     i3,' procs per replica')
#else /**/
20              FORMAT('GUKINI> Parallel replicas activated, use ', &
                     i3,' procs per replica')
#endif 
             ENDIF

             IREPQM=MYNODG/KREPQM+1

             ! Revised processor numbering
             NUMNOD=KREPQM
             MYNOD=MOD(MYNODG,KREPQM)
             MYNODP=MYNOD+1
             NODDIM=NPTWO()
             !     Setup new IPPMAP
             DO I=0,NUMNOD
                IPPMAP(I)=I+KREPQM*(IREPQM-1)
             ENDDO
          else
#endif 
             !     
             !     Serial and conventional parallel version
             !     
             IREPQM    = 1
             NPGRP     = 1   
             NREPQMNOD = NREPQM   
#if KEY_PARALLEL==1
             if(PRNLEV.GT.2 .AND. NUMNOD .GT. 1) THEN
                WRITE(OUTU,21) NUMNOD
21              FORMAT('GUKINT> All replicas will run parallel ', &
                     'over all ',i5,' processors')
             ENDIF
          endif
#endif 
       ENDIF
    ENDIF

#if KEY_PARALLEL==1
    ! Save to be used
    NUMNOD_save=NUMNOD
    MYNOD_save=MYNOD

    if(qrep .and. PRNLEV.GT.5) THEN
       write(OUTU,*)'GUKINI> NODE 0 to COMPUTE ',NREPQMNOD, &
            ' QM REPLICAS'
       write(OUTU,*)
    endif
#endif 

#endif 
#endif /* (replica)*/
#if KEY_REPDSTR==1 /*repdstr*/
    ! it needs also protection against qrpath??!!
    if(qrepdstr)then
       nrepqm=nrepdstr
       krepqm=numnod
       npgrp=nrepdstr
       nrepqmnod=1
       irepqm=irepdstr+1
       numnod_save=numnod    ! we never need to change this in repdstr
       mynod_save=mynod
    endif
#endif /* (repdstr)*/
#endif /* (rpath)*/
    !

    !write(50+mynodg,'(a,10i4)')'gukini>afterrepinit:nrepqm,krepqm,npgrp,irepqm=',nrepqm,krepqm,npgrp,irepqm
    !write(50+mynodg,'(a,10i4)')'gukini>afterrepinit:mynod,numnod,mynod_save,numnod_save=',mynod,numnod,mynod_save,numnod_save

#if KEY_QCHEM==1 || KEY_QTURBO==1
      if(qmused_qchem)then
      MAPFF(1)=-1
      DO I=1, NATOM
         DO J=1,I-1
            IF(CG(I).EQ.CG(J))THEN
               MAPFF(I)=MAPFF(J)
               EXIT
            ELSE
               MAPFF(I)=MAPFF(J)-1
            ENDIF
         ENDDO
         !write(*,*)'MAPFF= ','I= ',I, MAPFF(I),cg(i)
      ENDDO
      endif
#endif 

    !
    CALL COPSEL(ISLCT,QQINP)
    !     
    !     This initialize gamess data 
    QINIGM=.TRUE.
    !
    !     Modify QChem input and output filenames for replica/path and neb
#if KEY_QCHEM==1 || KEY_G09==1 /*qcini*/
    if(qmused_qchem.or.qmused_g09) then
#if KEY_REPLICA==1
#if KEY_RPATH==1
    !write(50+mynodg,*)'qrep,qrepd,irepqm=',qrep,qrepdstr,irepqm
#if KEY_REPDSTR==0
    IF(QREP) THEN                 
#endif
#if KEY_REPDSTR==1
    IF(QREP.or.qrepdstr) THEN     
#endif
       CALL RECOPSEL(IREPQM)
       IF(QINIQC) THEN
          CALL ENVIGRP(IREPQM)   
          QINIQC=.false.
       ENDIF
    ENDIF
#endif 
#endif 
    !     Get the info from the Q-chem input file
    CALL QCHEMINI
    endif
#endif /* (qcini)*/
#if KEY_QTURBO==1 /*qtini*/
    CALL QTURBOINI
#endif /* (qtini)*/
    !write(50+mynodg,'(a,10i4)')'numnod,mynod,irepqm=',numnod,mynod,irepqm
    !
    ! begin ewald
    !      write(*,*)'GUKINI>lewmode=',lewmode
    ! QC_UW1017: need to remove G09 as well
!#if KEY_SQUANTM==0 && KEY_SCCDFTB==0 && KEY_MNDO97==0  && KEY_QCHEM==0 && KEY_GAMESS==0 && KEY_QTURBO==0 && KEY_NWCHEM==0 /*nosquantm2*/
#if KEY_SQUANTM==0 && KEY_SCCDFTB==0 && KEY_MNDO97==0  && KEY_QCHEM==0 && KEY_GAMESS==0 && KEY_QTURBO==0 && KEY_NWCHEM==0 && KEY_G09==0/*nosquantm2*/
    IF(LEWMODE) THEN
       !     Do the initial setup
       !         COPSEL calculates the number of QM atoms, but it is later
       !                redifined in CHMDAT; there is no problem for QCHEM!
       !
       !     ****************************BE CAREFULL ********
       !
       !     These are the things which have not been taken care of yet:
       !
       !     - We might need a call to NBNDQM in nbonds.src (line 1901)
       !
       !     - ewaldf.src (line 64 and 132 - modify total charge with the QM!!!)
       !
       !     -
       !
       !     ****************************BE CAREFULL ********
       !
       !
       !     Fill the QMINB, MMINB arrays, RPATH needs checking here
       !
       IPT=0
       DO I=1,NATOM
          IF((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2))THEN
             IPT=IPT+1
             QMINB(IPT)=I
             MMINB(IPT)=I
          ENDIF
       ENDDO
       !
       !CC this was problem before...         NUMAT=NGAMES
       NUMAT=IPT
       !
       IPT=0
       DO I=1,NATOM
          IF(IGMSEL(I).EQ.0)THEN
             IPT=IPT+1
             MMINB(NUMAT+IPT)=I
          ENDIF
       ENDDO
       !
       !     Not sure if we need to zero the QM atoms here ???
       DO I=1,NATOM
          CGQMMM(I)=CG(I)
       ENDDO
       !
       !     qdone is output here, should be used somehwere?
#if KEY_QTURBO == 0
       CALL SETUP_QMEWD(QDONE)
#endif

       !     Prepare Ewald summation

       !        Compute the Ewald potential on QM atoms from all the MM atoms
       !C         use xi,yi,zi but you need to call swapxyz_image and define imattq!!
       !C         CALL QEWALDP(NATOM,XI),YI),ZI),.FALSE.)
#if KEY_QTURBO == 0
       CALL QEWALDP(NATOM,X,Y,Z,.FALSE.)
#endif
       !
       write(*,*)'GUKINI>numat,empot(1)=',numat,empot(1)
       !        Check the total charge for QM/MM-Ewald (PME usage)
       CGMM=ZERO
       DO I=1,NATOM
          IF(IGMSEL(I).EQ.5.OR.IGMSEL(I).EQ.0) CGMM=CGMM+CG(I)
       END DO
#if KEY_GHO==1
       IF(QLINK) THEN
          DO I = 1,NQMLNK
             CGMM=CGMM+QMATMQ(I)
          END DO
       END IF
#endif 
    END IF
#endif /*  (nosquantm2)*/
    ! end ewald
    !
#if KEY_GAMESS==1
    if(qmused_gamess) then
#if KEY_REPLICA==1
#if KEY_RPATH==1
    !     TODO: (FIXME) 
    !     Make some checks for number of replicas and number of processes
    !
    IF(QREP)THEN
       CALL RECOPSEL(IREPQM)
       CALL ENVIGRP(IREPQM)
    ENDIF
#endif 
#endif 
    IF(QFRAG)THEN
       IF(IFRAG.EQ.JFRAG)CALL ENVIGRP(JFRAG)
       !     This has some problems
       IF(IFRAG.EQ.JFRAG)CALL FCOPSEL(JFRAG)
    ENDIF

    IF(QFRAG)THEN
       IF(IFRAG.EQ.JFRAG)THEN
          CALL CH2GMS(.TRUE.)
          CALL GAMESS
       ENDIF
    ELSE
       CALL CH2GMS(.TRUE.)
       CALL GAMESS
    ENDIF
    endif
#endif
    !
#if KEY_NWCHEM==1
       ! NWChem intialization
       ! using KEY_NWCHEM + qmused_nwchem flags
       !write(*,*)'init:before nwchem> qmused_nwchem=',qmused_nwchem
       if(qmused_nwchem) then
          ! perform all the parsing for the NWChem program
          ! by caling the program with the NWChem input script specified
          ! with the environment variable NWCHEM_INPUT
          ! the input file must be for the target system,
          ! except the geometry can be anything:
          ! for example a single atom, but correct charge and multiplicity
          ! this routine is from the NWChem program
          call nwchem_charmm
          ! setup also the NWChem/CHARMM interface variables
          call nwchem_qm_atoms(x,y,z)
          call nwchem_mm_atoms(x,y,z)
          call nwchem_rtdb_clr     ! force recomputing forces
       endif
#endif
    !
#if KEY_GAMESSUK==1 /*guk*/
    INIT=1
    STARTUP = .true.
#if KEY_REPLICA==1
#if KEY_RPATH==1
    !
    do IREPL = IREPQM, IREPQM + NREPQMNOD - 1
       !
       !     update igmsel array for this replica
       !
       CALL RECOPSEL(IREPL)
       !
       !     Setup environment variables to separate group's I/O
       CALL ENVIGRP(IREPL)
#endif 
#endif 
       iver = 5
       CALL GAMESS(INIT,ICODE,STARTUP,LQMEWD,IVER)
       if (IVER.NE.-5) then
          write(6,*)iver
          CALL WRNDIE(-5,'<GAMESS-UK>','Code Version Mismatch')
       endif
#if KEY_REPLICA==1
#if KEY_RPATH==1
    enddo
#endif 
#endif 

#endif /* (guk)*/
    !     
    !     Report QM/MM repulsion energy, also when no derivatives involved
#if KEY_GAMESS==1
    IF((PRNLEV>=2).and.qmused_gamess) CALL CGREPE(NATOM) 
#endif
    !     
    !     Now we have to put pointer for input file at the end of gamess input
    !CC   CALL GAMEND(99)
    !
    ! begin ewald
    !     Ewald stuff allocates and frees memory each time qewaldp is called
    !     Maybe change this??
    !     Free memory allocation
#if KEY_SQUANTM==0 && KEY_MNDO97==0
    IF(LEWMODE) THEN
    END IF
#endif 
    ! end ewald
    !
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QCHEM==1 || KEY_G09==1
#if KEY_REPLICA==1 /*replica*/
#if KEY_RPATH==1
#if KEY_PARALLEL==1 /*parallel*/
    !     Restore the parallel info; but not for repd!
    !
#if KEY_REPDSTR==1
    if(.not.qrepdstr) then          
#endif
    IF(npgrp.ne.1)THEN
       MYNOD=MYNODG
       MYNODP=MYNOD+1
       NUMNOD=NUMNODG
       NODDIM=NPTWO()
       CALL CUBE(MYNOD,NUMNOD,IPPMAP)
    ENDIF
#if KEY_REPDSTR==1
    endif                           
#endif
    !     
#endif /* (parallel)*/
#endif 
#endif /* (replica)*/
#endif 
    !
#if KEY_PARALLEL==1
    IF(QFRAG)THEN
       MYNOD=MYNODG
       MYNODP=MYNOD+1
       NUMNOD=NUMNODG
       NODDIM=NPTWO()
       CALL CUBE(MYNOD,NUMNOD,IPPMAP)
    ENDIF
#endif 
    !
    call chmdealloc('gukini.src','GUKINI','ISLCT',NATOM,intg=ISLCT)

    !write(50+mynodg,'(a,10i4)')'gukini>end:numnod,mynod,irepqm=',numnod,mynod,irepqm

    COMLEN = 0
    !     
    RETURN
  END SUBROUTINE GUKINI
  !     
#if KEY_QCHEM==1 || KEY_G09==1
  ! The following routine is a hack that makes Q-Chem run with CHARMM
  ! when compiled with new MPI libraries (openmpi, mpich2). It should
  ! be removed, once the libraries get fixed for this problem.  The
  ! problem is that in parallel QM/MM jobs, CHARMM eats a lot of CPU
  ! time, while waiting for Q-Chem to finish. This makes it use under
  ! 1% of CPU time.
  ! not sure if ##.not.CMPI really works???
  ! This would make it general for ##ENSEMBLE, too???
#if KEY_PARALLEL==1
#if KEY_CMPI==1 || KEY_MPI==1
  subroutine break_busy_wait_mpi()
    use iso_c_binding
    use mpi
    use parallel
#if KEY_REPLICA==1
    use replica_mod,only:QQCRP
#endif 
    implicit none

    interface
       subroutine usleep(useconds) bind(C)
         use iso_c_binding
         implicit none
         integer(c_int32_t), value :: useconds
       end subroutine usleep
    end interface

    integer :: tag,req,ier,status(mpi_status_size),i,to,from
    logical :: flag
    integer(c_int32_t) :: micro_seconds = 10000
    integer,dimension(1) :: x = [1]
    tag=1
    if (numnod == 1) return
    if (mynod == 0) then
       do i=1,numnod-1
#if KEY_CMPI==1
          to=ippmap(i)
#else
          to=i
#endif
          call mpi_isend(x,1,mpi_integer,to,tag,comm_charmm,req,ier)
          call mpi_wait(req,status,ier)
       enddo
    else
#if KEY_CMPI==1
       from=ippmap(0)
#else
       from=0
#endif
       call mpi_irecv(x,1,mpi_integer,from,tag,comm_charmm,req,ier)
!      IF(.not.QQCRP) THEN 
       mpiw: do
          call mpi_test(req,flag,status,ier)
          if (flag) exit mpiw
          call usleep(micro_seconds)
       enddo mpiw
!      ENDIF 
       call mpi_wait(req,status,ier)
    endif
    return
  end subroutine break_busy_wait_mpi
#endif
#endif

  SUBROUTINE QCHEMINI
    !-----------------------------------------------------------------------
    !     Determine the number of QM atoms to pass to Q-Chem
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use gamess_fcm
    use psf
    !
    INTEGER I

    !
    qmused_qchem=.true.
    NGAMES = 0
    DO I=1, NATOM
       IF((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2))THEN
          NGAMES=NGAMES+1
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE QCHEMINI
#endif 
  !
#if KEY_QCHEM==1 /*qchem*/
  SUBROUTINE QCHEM(E,DX,DY,DZ,CGX,AMASSX,IACX,NDD1,DD1,QSECD,IUPT,JUPT)
    !-----------------------------------------------------------------------
    !     Run Q-chem and parse the output
    !-----------------------------------------------------------------------
    use chm_kinds
    use number
    use dimens_fcm
    use gamess_fcm
    use memory
    use psf
    use param
    use coord
    use parallel
    use consta
    use eutil
    use storage
    use stream
    use string
    use replica_mod
    use rtf,only:atct
    use block_fcm
    use pathm
    use pert
    use code
    use comand
    use selctam
    use scalar_module
    use lonepr,only: LKATMS
    use ewald_1m,only:lewald
    use linkatom, only: findel ! JZ_UW12
    use image,only:xucell
    use resdist_ltm
#if KEY_SMBP==1
    use pbeq,only:qsmbp,numsurf,qsmbp_qc_grad  /* JZ_UW12: For SMBP*/
#endif
#if KEY_PHMD == 1
    use phmd,only:QPHMD ! PME-PHMD -- Y Huang 2017
#endif

    implicit none
    
    !real(chm_real) E,DX(*),DY(*),DZ(*),CGX(:),AMASSX(*) ! JZ_UW12
    real(chm_real) E,DX(*),DY(*),DZ(*),CGX(*),AMASSX(*)
    INTEGER IACX(*),NDD1,NDDX
    real(chm_real) DD1(*)
    INTEGER IUPT(*),JUPT(*)
    LOGICAL QSECD
    LOGICAL LPRT,LEXIST,LQINIGM,LMIXBAS
    !
    INTEGER LC,LI,LO,LX,LQCH,IMO,OMO,OUT,ix,LNP,PROCS,FMO,LCR
    INTEGER ILEN,IFLAG,JFLAG,MMATOMS,QMATOMS,TMPVAR,IOSTAT
    INTEGER I,J,K,L,M,N,O,P,R,LPWD,LPA,LPB,STAT,LRPNDS
    INTEGER MAPQM(MAXA),MAPMM(MAXA),IPT,JPT,IERR       ! lw050728
    INTEGER MAPQM2(MAXA),MAPMM2(MAXA)                  !hlw_080705
    INTEGER NUMFF,MINFF,loc,numqcatmtyp
    CHARACTER(len=255) FILCNT,FILIN,FILOUT,FILEXE,PWDQC,SGROUP,PWDESP
    CHARACTER(len=255) RPNODEFILE,RPNODES,FILCNTA,FILCNTB,REP1INFO,REP2INFO
    CHARACTER(len=255) TMPA,PRTINA,PRTINB,PRTOUTA,PRTOUTB,PWDQCH,PWD,CWD
    CHARACTER(len=80) LINE,CRAP,CRAP1,CRAP2,LINEA,LINEB
    CHARACTER(len=6) ELE,NP,NCIS,NPOPT
    CHARACTER(len=6) tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9
    CHARACTER(len=10) corr,filecount
    CHARACTER(len=6) bas1,bas2
    CHARACTER(len=32) junkene
    CHARACTER(len=80) rem(25)
    CHARACTER(len=4) :: WRD
    !     LOGICAL QMP2,QLMP2,QCCSD,QCIS
    real(chm_real) T,NC,CC,DUMX,DUMY,DUMZ,LJE,Eewald,EE2,EE21,EE22
    real(chm_real) TMPX,TMPY,TMPZ,RX,RY,RZ,S,S2
    !     real(chm_real) tempdd1((3*natom*(3*natom+1))/2)
    !     real(chm_real) tempdd1(1000)
    real(chm_real) tmpxx(natom),tmpyy(natom),tmpzz(natom)
    INTEGER QMDIM,IQ,IIQ,IC,II,IADD,JADD,JQ,JJQ,JC,JJ,I1,LJCOUNT
    real(chm_real),allocatable, dimension(:) :: tempdd1
    real(chm_real),allocatable, dimension(:) :: mmpotqm
    integer(chm_int4),allocatable, dimension(:) :: mixbas,qcbonds,tmpqcb
    real(chm_real),allocatable, dimension(:,:) :: qmpotqm

    !hlw_080705
    !    real(chm_real) LKATMS
    !    COMMON /LKATM/  LKATMS(3,MAXAIM)
    INTEGER NLK, ILK, IQMH, IMMH, ios, count 
    !hlw_080705
#if KEY_QTURBO==1
    CHARACTER(len=255)  TURBODATADIR, TURBOEXE, TURBOPWD
    INTEGER LTURBODATADIR, LTURBOEXE, LTURBOPWD
    INTEGER readstatus
#endif 

    ! JZ_UW12
    INTEGER NSURF
    CHARACTER(len=10) jobtyp

    IF(QSECD .or. QNRAP) THEN 
       allocate(tempdd1((3*natom*(3*natom+1))/2),stat=ierr)
    ENDIF
    !
    LPRT=.false.

    !write(50+mynodg,'(a,10i4)')'qchem>start:numnod,mynod,irepqm=',numnod,mynod,irepqm

#if KEY_PARALLEL==1
    !The following call should be after the call system() line, but
    !that one is not executed on CPUs other than 0. So we call it
    !here, because this routine must be called by everyone!
#if KEY_CMPI==1 || KEY_MPI==1
!   IF(MYNOD > 0)  write(*,*)'CALLING BREAK_BUSY_WAIT',MYNOD
    IF(.not.QQCRP) THEN
!      write(*,*)'calling break_busy_wait: L1260' 
       IF(MYNOD > 0) CALL break_busy_wait_mpi()
    ENDIF 
#endif
    IF(MYNOD.GT.0) THEN
       E=ZERO
       RETURN
    ENDIF
#endif 
    IF (QQEWALD) THEN
      call chmalloc('gukini.src','QCHEM','QCBONDS',nbond,intg=QCBONDS)
      call chmalloc('gukini.src','QCHEM','TMPQCB',nbond,intg=TMPQCB)
   ENDIF
   
#if KEY_PERT == 1
    IF(QPERT) THEN
       IF (QMSTATE .EQ. 0) THEN
          call get_environment_variable('SAINP', filcnt, lc)
          IF(LC.EQ.0)CALL WRNDIE(-5,'<QCHEM>','No input specified.')
       ELSE IF (QMSTATE .EQ. 1) THEN
          CALL get_environment_variable('SBINP', FILCNT, LC)
          IF(LC.EQ.0)CALL WRNDIE(-5,'<QCHEM>','No input specified.')
       ENDIF
    ELSE
#endif /* KEY_PERT */
       
       CALL get_environment_variable("QCHEMCNT", FILCNT, LC)
       IF(LC.EQ.0)CALL WRNDIE(-5,'<QCHEM>','No input specified.')

#if KEY_PERT == 1
    ENDIF

    IF(QPERT) THEN
       IF(QMSTATE .EQ. 0) THEN
          CALL GET_ENVIRONMENT_VARIABLE("STATEAINP", FILIN, LI)
          IF(LI.EQ.0)CALL WRNDIE(-5,'<QCHEM>','No input specified.')
       ELSE IF(QMSTATE .EQ. 1) THEN
          CALL GET_ENVIRONMENT_VARIABLE("STATEBINP", FILIN, LI)
          IF(LI.EQ.0)CALL WRNDIE(-5,'<QCHEM>','No input specified.')
       ENDIF
    ELSE
#endif /* KEY_PERT */
       
       FILIN=''
       CALL GET_ENVIRONMENT_VARIABLE("QCHEMINP", FILIN, LI)
       IF(LI.EQ.0)CALL WRNDIE(-5,'<QCHEM>','No input specified.')

#if KEY_PERT == 1
    ENDIF


    IF (QPERT) THEN
       IF (QMSTATE .EQ. 0) THEN
          CALL GET_ENVIRONMENT_VARIABLE("STATEAOUT", FILOUT, LO)
          IF(LO.EQ.0)CALL WRNDIE(-5,'<QCHEM>','No output specified.')
       ELSE IF (QMSTATE .EQ. 1) THEN
          CALL GET_ENVIRONMENT_VARIABLE("STATEBOUT", FILOUT, LO)
          IF(LO.EQ.0)CALL WRNDIE(-5,'<QCHEM>','No output specified.')
       ENDIF
    ELSE
#endif /* KEY_PERT */
       
       CALL GET_ENVIRONMENT_VARIABLE("QCHEMOUT", FILOUT, LO)
       IF(LO.EQ.0)CALL WRNDIE(-5,'<QCHEM>','No output specified.')

#if KEY_PERT == 1
    ENDIF
#endif
 
    CALL GET_ENVIRONMENT_VARIABLE("QCHEMEXE", FILEXE, LX)
    IF(LX.EQ.0)CALL WRNDIE(-5,'<QCHEM>','No Q-chem specified.')
    !

    IMO=90
    OMO=91

    IF(QQCRP) THEN 
       !     Need this for replica path with Q-Chem
       CALL GET_ENVIRONMENT_VARIABLE("RPATHNODES", RPNODES, LRPNDS)
       IF(LRPNDS.EQ.0)CALL WRNDIE(-5,'<QCHEM>','No Nodefile for Q-chem specified.')
       !
       CALL GET_ENVIRONMENT_VARIABLE("QCHEMPWD", PWDQC, LPWD)
       IF(LPWD.EQ.0)  &
            CALL WRNDIE(-5,'<QCHEM>','No working directory set')

       WRITE(SGROUP,'(I5)')IREPQM
       M=255
       CALL TRIMA(SGROUP,M)

       N=LPWD+16+M
       PWDESP(1:N)=PWDQC(1:LPWD)//'/rpath'//SGROUP(1:M)// &
            '/ESPGrid'//CHAR(0)

       O=LPWD+16+M
       PWDQCH(1:O)=PWDQC(1:LPWD)//'/rpath'//SGROUP(1:M)// &
            '/qchosts'//CHAR(0)

       LCR=LPWD+2+LC
       FILCNT(1:LCR)=PWDQC(1:LPWD)//'/'//FILCNT(1:LC)//CHAR(0)

       ! create new variable that sets path to charges.dat file for rpath 
       LCH=LPWD+20
       FILCHRG(1:LCH)=PWDQC(1:LPWD)//'/rpath'//SGROUP(1:M)// &
            '/charges.dat'//CHAR(0)

       K=LPWD+1+LRPNDS
       RPNODEFILE(1:K)=PWDQC(1:LPWD)//'/'//RPNODES(1:LRPNDS)//CHAR(0) 
       !write(*,*)'RPNODEFILE = ',K,RPNODEFILE(1:K)

       IF(QRPMINP) THEN 
          R=LPWD+13 
          REP1INFO(1:R)=PWDQC(1:LPWD)//'/rep1info.dat'//CHAR(0) 
          REP2INFO(1:R)=PWDQC(1:LPWD)//'/rep2info.dat'//CHAR(0) 
       ENDIF 

       !write(*,*)'filech = ',FILCHRG(1:LCH)
       !write(50+mynodg,*)'filcnt = ',filcnt(1:Lcr)
       !write(50+mynodg,*)'filin = ',FILin(1:Li)
       !write(50+mynodg,*)'pwdesp = ',pwdesp(1:n)
       !write(50+mynodg,*)'pwdqch = ',pwdqch(1:o)
       OPEN(UNIT=90,FILE=FILCNT(1:LCR),STATUS='OLD')
       OPEN(UNIT=91,FILE=FILIN(1:LI),STATUS='REPLACE')
!      OPEN(UNIT=10,FILE=PWDESP(1:N),STATUS='REPLACE')

       OPEN(UNIT=12,FILE=PWDQCH(1:O),STATUS='REPLACE')
       OPEN(UNIT=14,FILE=RPNODEFILE(1:K),STATUS='OLD')

       IF(QRPMINP) THEN 
          OPEN(UNIT=70,FILE=REP1INFO(1:R),STATUS='OLD')
          OPEN(UNIT=71,FILE=REP2INFO(1:R),STATUS='OLD')
       ENDIF 

    ELSE
       OPEN(UNIT=90,FILE=FILCNT(1:LC),STATUS='OLD')
       OPEN(UNIT=91,FILE=FILIN(1:LI),STATUS='REPLACE')
!      OPEN(UNIT=10,FILE='ESPGrid',STATUS='REPLACE')
    ENDIF

    IF(QQCRP) THEN
       IF(OPTCNT.EQ.0) QCHEMCNTRL(1:LCR)=FILCNT(1:LCR)
       IF(QCHEMCNTRL(1:LCR).NE.FILCNT(1:LCR)) THEN
          QCHEMCNTRL(1:LCR)=FILCNT(1:LCR)
          WRITE(OUTU,22)'Control File Changed'
          QPRNTINP=.TRUE.
       ENDIF
    ELSE
       IF(OPTCNT.EQ.0) QCHEMCNTRL(1:LC)=FILCNT(1:LC)
       IF(QCHEMCNTRL(1:LC).NE.FILCNT(1:LC)) THEN
          QCHEMCNTRL(1:LC)=FILCNT(1:LC)
          WRITE(OUTU,22)'Control File Changed'
          QPRNTINP=.TRUE.
       ENDIF
    ENDIF

    !11   continue  ! loop to continue in case of multiple input sections 
    jflag=0
23  continue  ! loop to restart failed Q-Chem jobs...


    IF (QMESS.and.OptCnt.eq.0) THEN 
!      Need to compute Inverse Hessian at the first step 
       IFLAG=0
       DO
          READ(IMO,'(A)',END=100)LINE
          ILEN=80
          write(*,*) '1 line=', LINE
          CALL TRIMA(LINE,ILEN)
          CALL CNVTLC(LINE,ILEN)
          write(*,*) '2 line=', LINE
          WRITE(OMO,'(A)')LINE(1:ILEN)
!         1. get the QM geometry
          IF(STRFIN(LINE,'$molecule').GT.0)THEN
             READ(IMO,'(A)',END=100)LINE
             WRITE(OMO,'(A)')LINE(1:ILEN)
             !READ(IMO,'(A)',END=100)LINE
             DO I=1, NATOM
                IF((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2))THEN
                   ELE='      '    
                   LQINIGM=QINIGM
                   CALL FINDEL(ATCT(IACX(I)),AMASSX(I),I,ELE,T,LQINIGM)
                   WRITE(OMO,'(A6,3F20.10)')ELE,X(I),Y(I),Z(I)
                ENDIF
             ENDDO
          ENDIF
          IF(STRFIN(LINE,'$end').GT.0) THEN 
             IFLAG = IFLAG + 1
             IF (IFLAG.eq.2) GOTO 1150
          ENDIF

!         2. set rem section
          IF(STRFIN(LINE,'$rem').GT.0)THEN
             write(omo,'(A)')'STABILITY_ANALYSIS 1'
             write(omo,'(A)')'ESTIMATE_QMMM_ENERGY -2'
             write(omo,'(A,I5)')'CIS_N_ROOTS',NROOts
             write(omo,'(A)')'sym_ignore true'
             write(omo,'(A)')'symmetry false'
          ENDIF
       ENDDO
1150     CONTINUE
!
       WRITE(OMO,'(A)') '   '
       WRITE(OMO,'(A)') '@@@'
       WRITE(OMO,'(A)') '   '
       REWIND(IMO)

    ENDIF

    IF(.not.QPCM.and..not.QQEWALD) write(omo,'(A)')'$external_charges'

    MMATOMS=0
    JPT=1
    ! JZ_UW12: Add SMBP
    ! Need nogu option because for (separate) gradient calculation
    ! a converged density is needed
    NSURF=0
#if KEY_SMBP==1
      IF (QSMBP .and. .not. QNOGU) THEN
        CALL WRNDIE(-5,'<QCHEM>','NOGU option must be set for SMBP/Q-Chem')
      ENDIF
      IF (QSMBP) NSURF = NUMSURF
#endif 

!##IF PARALLEL
     IF(QOPENMP) THEN 
        WRITE(NPOPT,'(A3,1X)')"-nt"
     ELSE 
        WRITE(NPOPT,'(A3,1X)')"-np"
     ENDIF 
     !write(*,*)NPOPT
!##ENDIF 

    DO I=1, NATOM+NSURF
       !hlw_080705        IF(IGMSEL(I).EQ.0)THEN
       IF((IGMSEL(I).EQ.0).OR.(IGMSEL(I).EQ.5))THEN
          MMATOMS = MMATOMS+1
          MAPQM(MMATOMS)=I                            ! lw050728
          MAPQM2(I) = MMATOMS                         ! ys080320

          IF(QBLUCH) THEN
             IF(RECALLINT.NE.-1) THEN
                !WRITE(*,*)'RECALLINT 1 ',RECALLINT
                ! 0.56005704 = (sqrt(2)*sqrt(2)*.529*.529)
                IF (ABS(PTRSTO(recallint)%a(I)).LE.RSMALL) &
                     PTRSTO(recallint)%a(I)=PT005
                IF (ABS(PTRSTO(recallint)%a(I)).GE.FTHSND) &
                     PTRSTO(recallint)%a(I)=FTHSND
                
                IF(.not.QPCM) THEN
                  WRITE(OMO,'(5F16.8)')X(I),Y(I),Z(I),CGX(I), &
                  (BOHRR*BOHRR/(PTRSTO(recallint)%a(I)*PTRSTO(recallint)%a(I)))
                ENDIF 
                !write(*,*)PTRSTO(I)
             ELSE
                !WRITE(*,*)'RECALLINT 2 ',RECALLINT
                IF (ABS(WMAIN(I)).LE.RSMALL) WMAIN(I)=PT005
                IF (ABS(WMAIN(I)).GE.FTHSND) WMAIN(I)=FTHSND
                !write(*,*)WMAIN(I)

                WRITE(OMO,'(5F16.8)')X(I),Y(I),Z(I),CGX(I), &
                     (BOHRR*BOHRR/(WMAIN(I)*WMAIN(I)))            
             ENDIF
          ELSEIF(QQCLJ) THEN 
             I1=ITC(IAC(I))
             IC = I1*(I1+1)/2
             IF(.not.QPCM) THEN
             WRITE(OMO,'(6F16.8)')X(I),Y(I),Z(I),CGX(I), &
                  CNBB(IC)/TOKCAL,(TWO*VDWR(ITC(IAC(I))))/(2**(ONE/SIX))
             ENDIF
          ELSE 
             !hlw_080705            WRITE(OMO,'(4F16.8)')X(I),Y(I),Z(I),CGX(I)
             IF(IGMSEL(I).EQ.5) THEN
               IF(.not.QPCM) WRITE(OMO,'(4F16.8)')X(I),Y(I),Z(I),ZERO
! for DIV - guanhua_puja_QC_UW1212
               ELSE IF (NDIV(I).EQ.1) THEN
                 IF(.not.QPCM) WRITE(OMO,'(4F16.8)')X(I),Y(I),Z(I),ZLHOST(I)
               ELSE 
                 IF(.not.QPCM.and..not.QQEWALD) WRITE(OMO,'(4F16.8)')X(I),Y(I),Z(I),CGX(I)
               ENDIF
          ENDIF

          ! JZ_UW12: Add surface charges for SMBP
#if KEY_SMBP==1
       ELSEIF (QSMBP .and. (IGMSEL(I) .eq. 6)) THEN
            WRITE(OMO,'(4F16.8)')X(I),Y(I),Z(I),CGX(I)
#endif 

       ELSE 
          IPT=MMATOMS+JPT
          IF((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2)) THEN 
             MAPMM(JPT)=IPT
          ENDIF
          JPT=JPT+1
       ENDIF
    ENDDO

    IF(.not.QPCM.and..not.QQEWALD) write(omo,'(A)')'$end'
    !close(10)

    !hlw_080705
    IF (QCNBLK.gt.0) THEN
!      WRITE(OUTU,*) 'in QChem interface, QCNBLK=', QCNBLK
       write(omo,'(A)')''
       write(omo,'(A)')'$mm_blocks'
       K = 1
       DO I = 1, QCNBLK
          WRITE(OMO,'(40I3)') &
               (MAPQM2(QCBLKREORDER(J)),J=K,K+QCBLKSIZES(I)-1)
          K = K + QCBLKSIZES(I)
       ENDDO
       DO M = K, NATOM
          N = QCBLKREORDER(M)
          IF((IGMSEL(N).EQ.0).or.(igmsel(n).eq.5)) &
               WRITE(OMO,'(1I3)') MAPQM2(N)
       ENDDO
       write(omo,'(A)')'$end'
       write(omo,'(A)')''
    ENDIF
    !hlw_080705

    IF(QQCLJ) THEN
       write(omo,'(A)')' ' 
       write(omo,'(A)')'$lj_parameters'
       LJCOUNT=1
       DO I=1,NATOM
          IF((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2)) THEN
             I1=ITC(IAC(I))
             IC = I1*(I1+1)/2

             write(OMO,'(I3,2F16.8)')LJCOUNT,CNBB(IC)/TOKCAL, &
                  (TWO*VDWR(ITC(IAC(I))))/(2**(ONE/SIX))
             LJCOUNT=LJCOUNT+1
          ENDIF
       ENDDO
       write(omo,'(A)')'$end'
       write(omo,'(A)')' ' 
    ENDIF

    IF(QPCM)THEN
       NUMFF=MINVAL(MAPFF)
       !write(*,*)'number of ff= ',numff
    ENDIF

    IFLAG=0
    DO 
!10  READ(IMO,'(A)',END=100)LINE
    READ(IMO,'(A)',END=100)LINE
    ILEN=80
    CALL TRIMA(LINE,ILEN)
    CALL CNVTLC(LINE,ILEN)

    ! JZ_UW12
    ! For SMBP, Change JOBTYPE if necessary
#if KEY_SMBP==1
      J=INDEX(LINE,'jobtyp')
      IF (QSMBP .and. J.gt.0) THEN
         READ(LINE,*)tmp1,jobtyp
         IF (QSMBP_QC_GRAD) THEN
           WRITE(OMO,'(A,I8)') 'jobtype               force'
         ELSE
           WRITE(OMO,'(A,I8)') 'jobtype               sp'
         ENDIF
      ELSE
         WRITE(OMO,'(A)')LINE(1:ILEN)
      ENDIF
#else /**/
    WRITE(OMO,'(A)')LINE(1:ILEN)
#endif 

    IF(QPRNTINP) THEN
#if KEY_PARALLEL==1
       IF(mynodg.eq.0) THEN     
#endif
          IF(STRFIN(LINE,'$end').GT.0)IFLAG=0
          IF(IFLAG.eq.1) WRITE(OUTU,22) LINE(1:ILEN)
          IF(STRFIN(LINE,'$rem').GT.0) THEN
             WRITE(OUTU,22)'Q-Chem Job Parameters'
             WRITE(OUTU,22)'---------------------'
             IFLAG=1
          ENDIF
#if KEY_PARALLEL==1
       ENDIF                    
#endif
    ENDIF

    !IF(IFLAG.EQ.1)GOTO 100
    !IF(STRFIN(LINE,'$molecule').GT.0)IFLAG=1

    IF(QPCM)THEN
       IF(STRFIN(LINE,'$molecule').GT.0)THEN
          READ(IMO,'(A)',END=100)LINE
          WRITE(OMO,'(A)')LINE(1:ILEN)
          DO I=1,NATOM
             ELE='      '
             LQINIGM=QINIGM
             CALL FINDEL(ATCT(IACX(I)),AMASSX(I),I,ELE,T,LQINIGM)
             IF(ELE(1:3).EQ.'QQH')ELE(1:6)=' H    '
             IF(ELE(1:3).EQ.'qqh')ELE(1:6)=' H    '
             WRITE(OMO,'(A6,3F20.10,I8,I2,I2,I2,I2)')ELE,X(I),Y(I), &
             Z(I),MAPFF(I),0,0,0,0
          ENDDO
       ENDIF

       QMATOMS=0
       IPT=MMATOMS
       DO I=1, NATOM
          IF((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2))THEN
             IPT=IPT+1
             MAPQM(IPT)=I
             QMATOMS=QMATOMS+1
          ELSE
             IF(IGMSEL(I).EQ.0 .OR. IGMSEL(I).EQ.5)THEN
                MAPMM(JPT)=I
                JPT=JPT+1
             ENDIF
          ENDIF
       ENDDO
    ELSE

!    IF(QQEWALD) THEN 
!       J=0
!       do I=1,NATC
!!        write(*,*)'ATCCNT = ',ATCCNT(I)
!         if(ATCCNT(I).gt.0) then
!           J=J+1
!         endif
!       enddo
!    ENDIF 

    IF(STRFIN(LINE,'$molecule').GT.0)THEN
      READ(IMO,'(A)',END=100)LINE
      !WRITE(OMO,'(A)')LINE(1:ILEN)
!     WRITE(*,*)'testing molecule - ',QRPMINP,LINE
      IF(QRPMINP) THEN 
        IF(SGROUP(1:M).EQ."1") THEN 
          READ(70,'(A)',END=100)LINE
        ENDIF 
        IF(SGROUP(1:M).EQ."2") THEN 
          READ(71,'(A)',END=100)LINE
        ENDIF 
        WRITE(OMO,'(A)')LINE(1:ILEN)
      ELSE 
        WRITE(OMO,'(A)')LINE(1:ILEN)
      ENDIF 
      !WRITE(*,'(A)')LINE(1:ILEN)
      QMATOMS=0
      IPT=MMATOMS
      DO I=1, NATOM
        IF((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2))THEN
          ELE='      '
          LQINIGM=QINIGM
          CALL FINDEL(ATCT(IACX(I)),AMASSX(I),I,ELE,T,LQINIGM)
          IF(ELE(1:3).EQ.'QQH')ELE(1:6)=' H    '
          IF(ELE(1:3).EQ.'qqh')ELE(1:6)=' H    '
          IF (QQEWALD) THEN
          numqcatmtyp=size(qcffatmtyp)
!         write(*,*)'size = ',loc 
!         qcffatmtyp(0)=4
!         qcffatmtyp(1)=75
!         qcffatmtyp(2)=79
!         qcffatmtyp(3)=163
            do j=1,nbond
              qcbonds(j)=0
              tmpqcb(j)=0
            enddo 
            do j=1,nbond
              if(I .eq. IB(J)) then 
                qcbonds(j)=JB(J)
              elseif(I .eq. JB(J)) then
                qcbonds(j)=IB(J)
              else
                qcbonds(j)=0
              endif
            enddo
            l=0
            do k=1,nbond
              if(qcbonds(k).gt.0) then
                l=l+1
                tmpqcb(l)=qcbonds(k)
                qcbonds(k)=0
              endif
            enddo
            locloop1: do l=0,numqcatmtyp-1
              loc=abs(qcffatmtyp(l)-IAC(I)) 
              if(loc .eq. 0) then
                loc=l
                exit locloop1
              endif 
            enddo locloop1
            if(ELE(2:2) .eq. 'H') then 
              WRITE(OMO,'(A6,3F20.10,5I6)')ELE,X(I),Y(I),Z(I),-loc-1,tmpqcb(1),0,0,0
            else 
              WRITE(OMO,'(A6,3F20.10,5I6)')ELE,X(I),Y(I),Z(I),-loc-1,tmpqcb(1),tmpqcb(2),tmpqcb(3),tmpqcb(4)
            endif 
          ELSE
            WRITE(OMO,'(A6,3F20.10)')ELE,X(I),Y(I),Z(I)
          ENDIF
          IPT=IPT+1
          MAPQM(IPT)=I
          QMATOMS=QMATOMS+1
        ELSE
          IF(IGMSEL(I).EQ.0 .OR. IGMSEL(I).EQ.5)THEN
            IF (QQEWALD) THEN
              ELE='      '
              LQINIGM=QINIGM
              CALL FINDEL(ATCT(IACX(I)),AMASSX(I),I,ELE,T,LQINIGM)
              numqcatmtyp=size(qcffatmtyp)
              IF (QQEWALD) THEN
                do j=1,nbond
                  qcbonds(j)=0
                  tmpqcb(j)=0
                enddo
                do j=1,nbond
                  if(I .eq. IB(J)) then
                    qcbonds(j)=JB(J)
                    elseif(I .eq. JB(J)) then
                    qcbonds(j)=IB(J)
                  else
                    qcbonds(j)=0
                  endif
                enddo
                l=0
                do k=1,nbond
                  if(qcbonds(k).gt.0) then
                    l=l+1
                    tmpqcb(l)=qcbonds(k)
                    qcbonds(k)=0
                  endif
                enddo
              endif
              locloop2: do l=0,numqcatmtyp-1
                loc=abs(qcffatmtyp(l)-IAC(I)) 
                if(loc .eq. 0) then
                  loc=l
                  exit locloop2
                endif 
              enddo locloop2
              if(ELE(2:2) .eq. 'H') then 
                WRITE(OMO,'(A6,3F20.10,5I6)')ELE,X(I),Y(I),Z(I),-loc-1,tmpqcb(1),0,0,0
              else 
                WRITE(OMO,'(A6,3F20.10,5I6)')ELE,X(I),Y(I),Z(I),-loc-1,tmpqcb(1),tmpqcb(2),tmpqcb(3),tmpqcb(4)
              endif 
            ENDIF
            MAPMM(JPT)=I
            JPT=JPT+1
          ENDIF
        ENDIF
      ENDDO
    QINIGM=.FALSE.
    ENDIF
    ENDIF

    IF(STRFIN(LINE,'$rem').GT.0) THEN
       IF(QBLUCH) THEN
          !           DO NOTHING!!! 
          WRITE(OMO,'(A)') 'symmetry                off'
          WRITE(OMO,'(A)') 'sym_ignore             true'
       ELSE
          WRITE(OMO,'(A,I8)') 'symmetry                off'
          WRITE(OMO,'(A,I8)') 'sym_ignore             true'
       ENDIF

       IF(QSECD) THEN 
          IF (MMATOMS .GT. 0) THEN 
             WRITE(OMO,'(A)') 'QMMM_FULL_HESSIAN      TRUE' 
          ENDIF
       ENDIF

       IF(QRESTART) THEN 
          IF(OPTCNT.EQ.0) THEN
             WRITE(OMO,'(A)')'scf_guess      read'
          ENDIF
       ENDIF

       IF(QNOGU) THEN
          IF(OPTCNT.GT.0.or.QMESS) THEN
             WRITE(OMO,'(A,I8)')'scf_guess      read'
          ENDIF
       ENDIF

       IF (QMESS) THEN
          WRITE(OMO,'(A)')'ESTIMATE_QMMM_ENERGY 1002'
       ENDIF

       IF(QPCM.or.QQEWALD) THEN
           WRITE(OMO,'(A)')'user_connect          true'
           WRITE(OMO,'(A)')'qm_mm_interface       janus'
           WRITE(OMO,'(A)')'force_field           charmm27'
!          WRITE(OMO,'(A)')'force_field           read'
!          do i=1,nbond
!            write(*,*)'bonds in qchem: ',IB(I),JB(I)
!          enddo
       ENDIF

       IF (QQEWALD) THEN
           WRITE(OMO,'(A)')'mm_subtractive        true'
           WRITE(OMO,'(A)')'qm_mm                 true'
           WRITE(OMO,'(A)')'ewald_on              true'
       ENDIF

       ! JZ_UW12: Add options for SMBP
#if KEY_SMBP==1
       IF(QSMBP) THEN
             WRITE(OMO,'(A,I8)')'qmmm_charges          true'
             IF (.not. QSMBP_QC_GRAD) THEN
!              WRITE(OMO,'(A,I8)')'mm_charges            true'
!              WRITE(OMO,'(A,I8)')'mm_charges_write      true'
!              WRITE(OMO,'(A,I8)')'mm_charges_high_mom   true'
!              WRITE(OMO,'(A,I8)')'mm_charges_qconstalg  1'
!              WRITE(OMO,'(A,I8)')'mm_charges_qmax       1'
             ELSE
               WRITE(OMO,'(A,I8)')'maxscf                0'
!              WRITE(OMO,'(A,I8)')'scf_convergence       6'
             ENDIF
       ENDIF
#endif 
    ENDIF


!   !     Echo Q-Chem job info into CHARMM output file
!   IF(PRNLEV.GE.2) THEN
!      IF(OPTCNT.EQ.0) THEN
!         IF(STRFIN(LINE,'correlation').GT. 0) THEN
!            WRITE(OUTU,22) LINE(1:ILEN)
!            !               WRITE(OUTU,22)''
!            LPRT=.false.
!         ENDIF
!         IF(STRFIN(LINE,'basis').GT. 0) THEN
!            WRITE(OUTU,22) LINE(1:ILEN)
!            WRITE(OUTU,22)''
!            LPRT=.false.
!         ENDIF
!         IF(LPRT) THEN
!            WRITE(OUTU,22)''
!            LPRT=.false.
!         ENDIF
!         IF(STRFIN(LINE,'exchange').GT. 0) THEN
!            WRITE(OUTU,22)''
!            WRITE(OUTU,22)'Q-Chem Job Parameters'
!            WRITE(OUTU,22)'---------------------'
!            WRITE(OUTU,22) LINE(1:ILEN)
!            LPRT=.true.
!         ENDIF
!      ENDIF
!   ENDIF

    !-----------------------------------------------------
    !---------------Determine QM Method ------------------
    !-----------------------------------------------------

    IF(OPTCNT.EQ.0) THEN
       corr='hf/dft'
    ENDIF

    J=INDEX(LINE,'correlation')
    if (j.gt.0) then
       read(line,*)tmp1,corr

       K=INDEX(LINE,'!')
       IF (K.gt.0) THEN
          corr='hf/dft'
       ENDIF

       if (corr.eq.'mp2') then
          QMP2=.true.
       else if (corr.eq.'local_mp2') then
          QLMP2=.true.
       else if (corr.eq.'ccsd') then
          QCCSD=.true.
       else if (corr.eq.'ccsd(t)') then
          QCCSD=.true.
       else if (corr.eq.'rimp2') then
          QRIMP2=.true.
       else if (corr.eq.'sosmp2') then
          QSOSMP2=.true.
       else if (corr.eq.'mosmp2') then
          QMOSMP2=.true.
       else if (corr.eq.'scsmp2') then
          QSCSMP2=.true.
       else
          QMP2=.false.
          QLMP2=.false.
          QCCSD=.false.
          QCIS=.false.
          QRIMP2=.false.
          QSOSMP2=.false.
          QMOSMP2=.false.
          QSCSMP2=.false.
          corr='hf/dft'
       endif
    endif


    L=index(line,'cis_state_deriv')
    if (l.gt.0) then
       read(line,*)tmp1,ncis
       K=INDEX(LINE,'!')
       IF (K.gt.0) THEN
          corr='hf/dft'
          QCIS=.false.
       ELSE
          QCIS=.true.
       ENDIF
    endif

22  FORMAT(' QCHEM> ',A)

    !-----------------------------------------------------

!   GOTO 10
    ENDDO
100 CONTINUE
!   !

    IF(QMIXED) THEN 
       call chmalloc('gukini.src','QCHEM','MIXBAS',NATOM,intg=MIXBAS)
       mixbas=0
       lmixbas=.false.
       write(OMO,'(A)')'$basis'
       K=1
       DO I=1, NUMSKY
          bas1='basis1'
          bas2='basis2'
!         write(*,*)bas1,len(bas1),bas2,len(bas2)
          call cnvtuc(bas1,len(bas1))
          call cnvtuc(bas2,len(bas2))
          DO J=1, NATOM
             IF((IGMSEL(J).EQ.1).OR.(IGMSEL(J).EQ.2))THEN
                IF(EQST(NAMSKY(I),LNAMSK(I),bas1,len(bas1))) THEN
                   IF(PTRSKY(I)%a(J) .EQ. 1) THEN
                      mixbas(J)=1
                      lmixbas=.true. 
                   ENDIF
                ELSE IF(EQST(NAMSKY(I),LNAMSK(I),bas2,len(bas2))) THEN
                   IF(PTRSKY(I)%a(J) .EQ. 1) THEN
                      mixbas(J)=2
                      lmixbas=.true. 
                   ENDIF
                ENDIF
             ENDIF
          ENDDO
       ENDDO
       K=1
       DO J=1,NATOM
          IF(MIXBAS(J) .EQ. 0) THEN 
             ! Do nothing for MM atoms 
          ELSE IF(MIXBAS(J) .EQ. 1) THEN 
             LQINIGM=QINIGM
             CALL FINDEL(ATCT(IACX(J)),AMASSX(J),J,ELE,T,LQINIGM)
             IF(ELE(1:3).EQ.'QQH')ELE(1:6)=' H    '
             IF(ELE(1:3).EQ.'qqh')ELE(1:6)=' H    '
             write(OMO, '(A,I4)') ELE, K
             write(OMO,'(A)')BASIS1
             write(OMO,'(A)')'****' 
             K=K+1
          ELSE IF(MIXBAS(J) .EQ. 2) THEN 
             LQINIGM=QINIGM
             CALL FINDEL(ATCT(IACX(J)),AMASSX(J),J,ELE,T,LQINIGM)
             IF(ELE(1:3).EQ.'QQH')ELE(1:6)=' H    '
             IF(ELE(1:3).EQ.'qqh')ELE(1:6)=' H    '
             write(OMO, '(A,I4)') ELE, K
             write(OMO,'(A)')BASIS2
             write(OMO,'(A)')'****' 
             K=K+1
          ENDIF
       ENDDO 
       write(OMO,'(A)')'$end'
       write(OMO,'(A)')''

       if (.not. lmixbas) then 
          CALL WRNDIE(-5,'<QCHEM>','No mixed basis set defined.')
       endif 
       call chmdealloc('gukini.src','QCHEM','MIXBAS',NATOM,intg=MIXBAS)
       IF(QQEWALD) then 
          call chmdealloc('gukini.src','QCHEM','QCBONDS',4,intg=QCBONDS)
          call chmdealloc('gukini.src','QCHEM','TMPQCB',4,intg=TMPQCB)
       endif 
    ENDIF


    IF(QQNOLKATMS) THEN
      IGMSEL=1
    ENDIF 
    !hlw_080705
    !     yihan, tell qchem which are link atoms, 
    !     this should only be done for full hessian evaluations
    IF(QSECD) THEN 
       NLK = 0
       DO I=1, NATOM
          IF(IGMSEL(I).EQ.2)  THEN
             NLK = NLK + 1
             IF (NLK.eq.1) THEN
                write(omo,'(A)')' ' 
                write(omo,'(A)')'$link_atoms'
             ENDIF
             !write(*,*) I, LKATMS(1,I), LKATMS(2,I), LKATMS(3,I)
             DO J = MMATOMS+1, MMATOMS+QMATOMS, 1
                IF (MAPQM(J).EQ.INT(LKATMS(1,I))) THEN
                   !WRITE(*,*) 'QM host:', J-MMATOMS
                   IQMH = J-MMATOMS
                ENDIF
                IF (MAPQM(J).EQ.I) THEN
                   !WRITE(*,*) 'QM Link', J-MMATOMS
                   ILK  = J-MMATOMS
                ENDIF
             ENDDO
             DO J = 1, MMATOMS
                IF (MAPQM(J).EQ.INT(LKATMS(2,I))) THEN
                   !WRITE(*,*) 'MM host:', J
                   IMMH = J
                ENDIF
             ENDDO
             write(omo, '(3I5,F12.7)') ILK, IQMH, IMMH, -LKATMS(3,I)
          ENDIF
       ENDDO
       IF (NLK.GE.1) write(omo,'(A)')'$end'
    ENDIF
    !hlw_080705

    IF(QPCM) THEN
       write(omo,'(A)')' '
       write(omo,'(A)')'$qm_atoms'
       DO I=1,NATOM
          IF((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2)) write(omo,'(I8)')I
       ENDDO
       write(omo,'(A)')'$end'
    ENDIF

    IF(QPCM) THEN
       NUMFF=ABS(NUMFF)
       write(omo,'(A)')' '
       write(omo,'(A)')'$force_field_params'
       write(omo,'(A,I8)')'NumAtomTypes',NUMFF
       minff=0
       DO I=1,NATOM
          IF((mapff(i).lt.minff).and.(CGX(I).NE.ZERO)) THEN
             minff=mapff(i)
             write(omo,'(A,I8,2F14.8,A)')'AtomType',MAPFF(I),CGX(I), &
             VDWR(ITC(IAC(I))), '   0.000'
             !write(*,*)'mapff= ',mapff(i) VDWR(ITC(IAC(I)))
          ENDIF
       ENDDO
       write(omo,'(A)')'$end'
    ENDIF


    ! Add MM potential vector for QM atoms and a matrix for QM-QM ewald
    IF (QQEWALD) THEN
       write(omo,'(A)') ' '
       write(omo,'(A)')'$qm_atoms'
       write(omo,'(40I6)') (MAPQM(I),I=MMATOMS+1,MMATOMS+QMATOMS)
       write(omo,'(A)')'$end'
       write(omo,'(A)') ' '
       write(omo,'(A)')'$forceman'
       write(omo,'(A)')'noQM-QMorQM-MMvdw'
       write(omo,'(A)')'ewald'
       write(omo,'(A,F6.3,1X,F6.3)')'alpha ',QMALpha,MMALpha
       write(omo,'(A,F12.7)')'box_length',XUCELL(1)
       write(omo,'(A)')'$end'
       write(omo,'(A)')' '
! Instead of doign this... echo back MyForceField.prm file 
! to the qchem input file...
       INQUIRE(file='MyForceField.prm',exist=lexist) 
       IF (lexist) THEN
         IF(MYNODG.EQ.0) THEN
          !write(*,*)'READING MyForceField.prm'
          OPEN (unit=11, file='MyForceField.prm', status='old')
          REWIND(11)
          count=0
            do
              read(11,'(A255)',iostat=ios) TMPA
              if (ios/=0) exit     
              write(omo,'(A255)') TMPA
              !write(*,*) ios 
              count=count+1
            enddo 
            close(11)
         ENDIF 
       ELSE 
          write(omo,'(A)')'$force_field_params'
          write(omo,'(A)')'read MyForceField.prm'
          write(omo,'(A)')'$end'
       ENDIF 
    ENDIF

    IF(QQRESD) THEN 
      IF(QQCOORD) THEN 
! r12mr34 1 2 1 6 -2.0 1000.0
        IF(QQCONS) THEN 
          write(omo,'(A)')'$opt'  
          write(omo,'(A)')'CONSTRAINT' 
        ELSE 
          write(omo,'(A)')'$opt2'  
        ENDIF 
        j=1
        do i=1,natom
          IF((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2))THEN
            mapmm2(i)=j
            j=j+1
          else 
            mapmm2(i)=0
          endif
        enddo 
        j=1
        do i=1,REDNUM
          IF(QQCONS) THEN 
            if( (mapmm2(REDILIS(1,j)).gt.0) .and. (mapmm2(REDILIS(2,j)).gt.0) ) then 
              write(omo,'(A,2I5,F12.4)')'stre ',mapmm2(REDILIS(1,j)),mapmm2(REDILIS(2,j)),REDRVAL(i)
            endif 
            j=j+1
          ELSE 
            if( (mapmm2(REDILIS(1,j)).gt.0) .and. (mapmm2(REDILIS(2,j)).gt.0) .and. & 
                (mapmm2(REDILIS(1,j+1)).gt.0) .and. (mapmm2(REDILIS(2,j+1)).gt.0.) ) then  
                  write(omo,'(A,4I5,2F12.4)')'r12mr34 ',mapmm2(REDILIS(1,j)), & 
                        mapmm2(REDILIS(2,j)),mapmm2(REDILIS(1,j+1)), & 
                        mapmm2(REDILIS(2,j+1)),REDRVAL(i),REDKVAL(i)
            endif 
            j=j+2
          ENDIF 
        enddo 
        IF(QQCONS) THEN 
          write(omo,'(A)')'ENDCONSTRAINT'
          write(omo,'(A)')'$end'
        ELSE 
          write(omo,'(A)')'$end'
        ENDIF 
      ELSE 
        CALL WRNDIE(-5,'<QCHEM>','Coordindate reading must be enabled.')
      ENDIF
    ENDIF

    IF(LEWALD.and..not.QQEWALD)THEN
       call chmalloc('gukini.src','QCHEM','MMPOTQM',QMATOMS,crl=MMPOTQM)
       call chmalloc('gukini.src','QCHEM','QMPOTQM',QMATOMS,QMATOMS,crl=QMPOTQM)
       write(omo,'(A)')'$ewald_potential'

       call mmpotqmew(qmatoms,natom,mmpotqm,x,y,z,cgx)

       do i = 1, qmatoms
          write(omo,'(f20.10)')MMPOTQM(i)
       enddo

       call qmpotqmew(qmatoms,natom,qmpotqm)

       do i = 1, qmatoms
          do j = i, qmatoms
             write(omo,'(f20.10)')QMPOTQM(i,j)
          enddo
       enddo

       write(omo,'(A)')'$end'
       call chmdealloc('gukini.src','QCHEM','MMPOTQM',QMATOMS,crl=MMPOTQM)
       call chmdealloc('gukini.src','QCHEM','QMPOTQM',QMATOMS,QMATOMS,crl=QMPOTQM)
    ENDIF

    close(omo)
    close(imo)
    QPRNTINP=.FALSE.

    IF(QWRITeinp) THEN 
!     QWRITeinp=.false.
      RETURN 
    ENDIF 

    IF(.NOT.QREADHESS) THEN 
    call system('sleep 2') 

#if KEY_PARALLEL==1
       !write(50+mynodg,'(a,l3,10i4)')'qchem:qqcrp,mynod,numnod=',qqcrp,mynod,numnod
       IF(QQCRP) THEN

          !     Original Scheme: Do not use unless you talk to Lee Woodcock
          !     CALL SYSTEM('cd '//PWDQC(1:LPWD)//'/rpath'//SGROUP(1:M)//
          !     $        '; $QCHEMEXE '//FILIN(1:LI)//' '//FILOUT(1:LO))

          !     Call the executable directly
          !     CALL SYSTEM('$QC/exe/qcprog.exe '//FILIN(1:LI)//
          !     $        ' /tmp/qchem'//SGROUP(1:M)// '> '//FILOUT(1:LO))

          !     Generic use of qchem script to do rpath: should be the default
          P=LPWD+8+M
          IF(QNOGU) THEN
             !            CALL SYSTEM('cd '//PWDQC(1:LPWD)//'/rpath'//SGROUP(1:M)//
             !     $           '; $QCHEMEXE '//FILIN(P:LI)//' '//FILOUT(P:LO)// 
             !     $           ' '//'SAVE_'//SGROUP(1:M))


             !     Do Parallel or Parallel/Parallel Replica Path/NEB/Off-Path Simulation
             ! Commented this out in c36a3 to get to work... not sure if its needed in c36a4 yet. 
             IF(QCPARA /= -1) THEN
                WRITE(NP,'(I5,1X)')QCPARA
             ELSE 
               WRITE(NP,'(I5,1X)')NUMNOD
             ENDIF

#if KEY_PARCMD==0 /* mc070104 fix*/
             !...##ERROR This code needs PARCMD in pref.dat
             !Break the compiler here It appears that parhost is only filled for gnu and iris
#endif /*        mc070104 fix*/

             !     WRITE OUT NODES NAMES TO QCHOSTS FILE (THANKS TO MILAN FOR HOSTNAME INFO)
            DO I=0,NUMNODG-1
              IF (I.EQ.MYNODG) THEN
                  IF(MYNODG.EQ.0) THEN
                     DO J=1,KREPQM
                       read(14,*)line
!                      write(*,*)NUMNODG,MYNODG,NUMNOD,line
                       write(12,'(A80)')line
                     ENDDO
                  ELSE
                     DO J=1,MYNODG
                       read(14,*)line
                     ENDDO
                     DO J=1,KREPQM
                       read(14,*)line
!                      write(*,*)NUMNODG,MYNODG,NUMNOD,line
                       write(12,'(A80)')line
                     ENDDO
                  ENDIF
              ENDIF
            ENDDO
!           DO I=0,NUMNODG-1
!             IF (I.EQ.MYNODG) THEN
!                DO J=1,KREPQM
!                  WRITE(12,'(A80)')PARHOST(MYNODG+J)
!                ENDDO
!             ENDIF
!           ENDDO
            CLOSE(12)
            CLOSE(14)

            IF(QQCSCRATCH) THEN 
            ! Still need to fix this: 9/19/2011
            CALL SYSTEM('cd '//PWDQC(1:LPWD)//'/rpath'//SGROUP(1:M)// & 
                 '; $QCHEMEXE '//NPOPT//' '//NP//FILIN(P:LI)//' '//    & 
                 FILOUT(P:LO)//' '//QCSCRATCH//'_'//SGROUP(1:M))
            ELSE 
            CALL SYSTEM('cd '//PWDQC(1:LPWD)//'/rpath'//SGROUP(1:M)// & 
                 '; $QCHEMEXE '//NPOPT//' '//NP//FILIN(P:LI)//' '//    & 
                 FILOUT(P:LO)//' SAVE_'//SGROUP(1:M))
            ENDIF 
          ELSE
             !     Do Parallel or Parallel/Parallel Replica Path/NEB/Off-Path Simulation
            IF(QCPARA /= -1) THEN
               WRITE(NP,'(I5,1X)')QCPARA
            ELSE 
               WRITE(NP,'(I5,1X)')NUMNOD
            ENDIF

             !     WRITE OUT NODES NAMES TO QCHOSTS FILE (THANKS TO MILAN FOR HOSTNAME INFO)
!           write(*,*)'numnodg, mynodg, numnod, krepqm: ',NUMNODG, MYNODG,NUMNOD,krepqm
            DO I=0,NUMNODG-1
              IF (I.EQ.MYNODG) THEN
                  IF(MYNODG.EQ.0) THEN 
                     DO J=1,KREPQM
                       read(14,*)line 
!                      write(*,*)NUMNODG,MYNODG,NUMNOD,line 
                       write(12,'(A80)')line
                     ENDDO
                  ELSE 
                     DO J=1,MYNODG
                       read(14,*)line 
                     ENDDO
                     DO J=1,KREPQM
                       read(14,*)line 
!                      write(*,*)NUMNODG,MYNODG,NUMNOD,line 
                       write(12,'(A80)')line
                     ENDDO
                  ENDIF
              ENDIF
            ENDDO

!           DO I=0,NUMNODG-1
!             IF (I.EQ.MYNODG) THEN
!                DO J=1,KREPQM
!                  WRITE(12,'(A80)')PARHOST(MYNODG+J)
!                ENDDO
!             ENDIF
!           ENDDO
            CLOSE(12)
            CLOSE(14)

            IF(QQCSCRATCH) THEN                ! QSCRATCH TEST 
            IF(QRESTART) THEN 
               CALL SYSTEM('cd '//PWDQC(1:LPWD)//'/rpath'//SGROUP(1:M)// &
                    '; $QCHEMEXE '//NPOPT//' '//NP//FILIN(P:LI)//' '//    & 
                    FILOUT(P:LO)//' '//QCSCRATCH//'_'//SGROUP(1:M))
               QRESTART=.FALSE.
            ELSEIF(QSAVORB) THEN 
               CALL SYSTEM('cd '//PWDQC(1:LPWD)//'/rpath'//SGROUP(1:M)// &
                    '; $QCHEMEXE '//NPOPT//' '//NP//FILIN(P:LI)//' '//    & 
                    FILOUT(P:LO)//' '//QCSCRATCH//'_'//SGROUP(1:M))
               QSAVORB=.FALSE.
            ELSE 
               CALL SYSTEM('cd '//PWDQC(1:LPWD)//'/rpath'//SGROUP(1:M)// &
                    '; $QCHEMEXE '//NPOPT//' '//NP//FILIN(P:LI)//' '//    & 
                    FILOUT(P:LO))
            ENDIF
            ELSE                               ! QSCRATCH TEST 
            IF(QRESTART) THEN 
               CALL SYSTEM('cd '//PWDQC(1:LPWD)//'/rpath'//SGROUP(1:M)// &
                    '; $QCHEMEXE '//NPOPT//' '//NP//FILIN(P:LI)//' '// &
                    FILOUT(P:LO)//' SAVE_'//SGROUP(1:M))
               QRESTART=.FALSE.
            ELSEIF(QSAVORB) THEN 
               CALL SYSTEM('cd '//PWDQC(1:LPWD)//'/rpath'//SGROUP(1:M)// &
                    '; $QCHEMEXE '//NPOPT//' '//NP//FILIN(P:LI)//' '// &
                    FILOUT(P:LO)//' SAVE_'//SGROUP(1:M))
               QSAVORB=.FALSE.
            ELSE 
               CALL SYSTEM('cd '//PWDQC(1:LPWD)//'/rpath'//SGROUP(1:M)// &
                    '; $QCHEMEXE '//NPOPT//' '//NP//FILIN(P:LI)//' '// &
                    FILOUT(P:LO))
            ENDIF
            ENDIF 
          ENDIF
       ELSE
          IF(QCPARA /= -1) THEN
             WRITE(NP,'(I5,1X)')QCPARA
          ELSE 
             WRITE(NP,'(I5,1X)')NUMNOD
          ENDIF

          !----------------------------------------------------------------------------------
          ! Fix this to work with PARCMD and PARHOST.... 
          ! If not PARCMD then set this up to work with PBS_NODEFILE and print a warning... 
          !----------------------------------------------------------------------------------
          IF( (QCPARA /= -1) .or. (NUMNOD .GT. 1) ) THEN 
             DO I=0,NUMNODG-1
                CALL SYSTEM('hostname -s >> qchosts') 
             ENDDO
             !TMPA(1:20) = 'MP_HOSTFILE=qchosts'//CHAR(0) 
             !CALL FPUTENV(TMPA,20) 
             !CALL SYSTEM('env') 
          ENDIF

          IF(QQCSCRATCH) THEN                ! QSCRATCH TEST 
          IF(QNOGU) THEN
             CALL SYSTEM('$QCHEMEXE '//NPOPT//' '//NP//FILIN(1:LI)//' ' &
                  //FILOUT(1:LO)//' '//QCSCRATCH)
          ELSE         
             IF(QRESTART) THEN 
                CALL SYSTEM('$QCHEMEXE '//NPOPT//' '//NP//FILIN(1:LI)//' ' &
                     //FILOUT(1:LO)//' '//QCSCRATCH)
                QRESTART=.FALSE.
             ELSEIF (QSAVORB) THEN 
                CALL SYSTEM('$QCHEMEXE '//NPOPT//' '//NP//FILIN(1:LI)//' ' &
                     //FILOUT(1:LO)//' '//QCSCRATCH)
                QSAVORB=.FALSE.
             ELSE 
                CALL SYSTEM('$QCHEMEXE '//NPOPT//' '//NP//FILIN(1:LI)//' ' &
                     //FILOUT(1:LO))
             ENDIF
          ENDIF
          ELSE                             ! QSCRATCH TEST 
          IF(QNOGU) THEN
             CALL SYSTEM('$QCHEMEXE '//NPOPT//' '//NP//FILIN(1:LI)//' ' &
                  //FILOUT(1:LO)//' '//'SAVE')
          ELSE         
             IF(QRESTART) THEN 
                CALL SYSTEM('$QCHEMEXE '//NPOPT//' '//NP//FILIN(1:LI)//' ' & 
                     //FILOUT(1:LO)//' '//'SAVE')
                QRESTART=.FALSE.
             ELSEIF (QSAVORB) THEN 
                CALL SYSTEM('$QCHEMEXE '//NPOPT//' '//NP//FILIN(1:LI)//' ' & 
                     //FILOUT(1:LO)//' '//'SAVE')
                QSAVORB=.FALSE.
             ELSE 
                CALL SYSTEM( &
                     '$QCHEMEXE '//NPOPT//' '//NP//FILIN(1:LI)//' '//FILOUT(1:LO))
             ENDIF
          ENDIF
          ENDIF                            ! QSCRATCH TEST 
       ENDIF
#elif KEY_PARALLEL==0
       ! HLW - Dec. 2007
       ! Fix the serial CHARMM version to work with QCPARA and allow 
       ! parallel execution of Q-Chem
       IF(QQCSCRATCH) THEN                ! QSCRATCH TEST 
       IF(QNOGU) THEN
          CALL SYSTEM('$QCHEMEXE '//FILIN(1:LI)//' '//FILOUT(1:LO)// &
               ' '//QCSCRATCH)
       ELSE
          IF(QRESTART) THEN 
             CALL SYSTEM('$QCHEMEXE '//FILIN(1:LI)//' '//FILOUT(1:LO)// &
                  ' '//QCSCRATCH)
             QRESTART=.FALSE.
          ELSEIF (QSAVORB) THEN 
             CALL SYSTEM('$QCHEMEXE '//FILIN(1:LI)//' '//FILOUT(1:LO)// &
                  ' '//QCSCRATCH)
             QSAVORB=.FALSE.
          ELSE 
             CALL SYSTEM('$QCHEMEXE '//FILIN(1:LI)//' '//FILOUT(1:LO))
          ENDIF
       ENDIF
       ELSE                             ! QSCRATCH TEST 
       IF(QNOGU) THEN
          CALL SYSTEM('$QCHEMEXE '//FILIN(1:LI)//' '//FILOUT(1:LO)// & 
               ' '//'SAVE')
       ELSE
          IF(QRESTART) THEN 
             CALL SYSTEM('$QCHEMEXE '//FILIN(1:LI)//' '//FILOUT(1:LO)// &
                  ' '//'SAVE')
             QRESTART=.FALSE.
          ELSEIF (QSAVORB) THEN 
             CALL SYSTEM('$QCHEMEXE '//FILIN(1:LI)//' '//FILOUT(1:LO)// &
                  ' '//'SAVE')
             QSAVORB=.FALSE.
          ELSE 
             CALL SYSTEM('$QCHEMEXE '//FILIN(1:LI)//' '//FILOUT(1:LO))
          ENDIF
       ENDIF
       ENDIF                            ! QSCRATCH TEST 
#endif 

       ! The following routine must be called after call
       ! system('qchem...')  This is executed only on processor 0
       ! here, so it must have corresponding call elsewhere. Currently
       ! in the beginning of this routine!

#if KEY_PARALLEL==1
#if KEY_CMPI==1 || KEY_MPI==1
     IF(.not.QQCRP) THEN
!       write(*,*)'calling break_busy_wait: L2393' 
        call break_busy_wait_mpi()
     ENDIF 
#endif
#endif

       !------------------------------------------------------
       !---- Get the energy and forces from Q-chem output ----
       !------------------------------------------------------

       E=ZERO
       T=ZERO
       CC=ZERO

       open(omo,FILE=filout(1:lo),status='old')
20     read(omo,'(a)',end=200) line

       IF(QMP2.or.QCCSD.or.QSOSMP2.or.QMOSMP2.or.QSCSMP2) THEN 
          i=index(line,'The QM part of the Energy is')
          if (i.gt.0) then
             read(line,*)tmp1,tmp2,tmp2,tmp4,tmp5,tmp6,tmp7,e
          endif
       ELSE If(QLMP2) THEN
          i=index(line,'TRIM MP2           total energy')
          if (i.gt.0) then
             read(line,*)tmp1,tmp2,tmp3,tmp4,tmp5,e
          endif
       ELSE IF(QRIMP2) THEN 
          i=index(line,'RI-MP2 TOTAL ENERGY')
          if (i.gt.0) then
             read(line,*)tmp1,tmp2,tmp3,tmp4,e
          endif
       ELSE IF(QCIS) THEN
          i=index(line,'Total energy for state   ')
          if (i.gt.0) then
             tmp1=line(29:30)
             if (tmp1(2:3) .eq. ncis) then
                read(line,*)tmp1,tmp2,tmp3,tmp4,tmp5,e
             endif
          endif
       ELSE
          i=index(line,'criterion')
          if (i.gt.0) then
             read(line,*)ix, e
          endif
       ENDIF

      IPT=1
      if(QQCOORD) then
!     write(*,*)'QQCOORD = ',QQCOORD
         i=index(line,'Coordinates (Angstroms)')
         if(i.gt.0) then
            read(omo,'(a)',end=200)line
!        write(*,*)line
            do j=1,natom
!            write(*,*)'j= ',j,'igmsel= ',igmsel(j), ',mapmm= ',mapmm(j)
               if((igmsel(j).eq.1).or.(igmsel(j).eq.2)) then
                  jpt=mapmm(ipt)
                  ipt=ipt+1
                  read(omo,'(a)',end=200)line
!                  write(*,*)'jpt= ',jpt, ',mapmm= ',mapmm(j), 'mapqm= ',   & 
!                            mapqm(j), line
                  !read(line,'(18x,f9.6,3x,f9.6,3x,f9.6)')x(jpt),y(jpt),z(jpt)
                  !read(line,'(18x,f9.6,3x,f9.6,3x,f9.6)')x(jpt),y(jpt),z(jpt)
                  read(line,'(17x,f12.9,3x,f12.9,3x,f12.9)')x(jpt),y(jpt),z(jpt)
!                 write(*,*)x(jpt),' ',y(jpt),' ',z(jpt)
              endif
           enddo
        endif
      endif

       i=index(line,'Convergence failure')
       if(i.gt.0) THEN
          WRITE(OUTU,22)'SCF Convergence Failure!!!'
          WRITE(OUTU,22)'Try adding "SCF_ALGORITHM DIIS_GDM" to QM file'
          WRITE(OUTU,22)'Or increase the number of scf iterations....'
          WRITE(OUTU,22)'MAX_SCF_CYCLES 100'
          STOP
       ENDIF

       i=index(line,'Thank you very much for using Q-Chem.')
       if(i.gt.0) THEN
!         write(outu,22)'Q-Chem Run successful... '
          jflag=-1
       endif

       i=index(line,'Nucleus-charge')
       if (i.gt.0) then
          read(line,*)tmp1,tmp2,tmp3,NC
       endif
       i=index(line,'Charge-charge')
       if (i.gt.0) then
          read(line,*)tmp1,tmp2,tmp3,CC
       endif
       i=index(line,'Eewald:')
       if (i.gt.0) then
          read(line,*)tmp1,Eewald
       endif

       if (QMESS) then
          i=index(line,'E2:')
          if (i.gt.0) then
!             E2:  -0.0008001 model I:  -0.0007817 model II:  -0.0008099
!             read(line,*)tmp1,EE2,tmp2,EE21,tmp3,EE22
             read(line,*)tmp1,EE2,tmp2,tmp3,EE21,tmp4,tmp5,EE22
             write(outu,'(A8, F14.10, A8, F14.10, A8, F14.10)') 'ENER E2>', EE2, 'MESS-E:', EE21, 'MESS-H:', EE22
          endif
       endif

       !JZ_UW12: For SMBP: Crash if we didn't get MDC charges
       !                   Warn if the MDC's are of lower quality
#if KEY_SMBP==1
       IF(QSMBP) THEN
         I=INDEX(line,'Unable to obtain Stewart charges')
         IF(I.gt.0) THEN
            CALL WRNDIE(-5,'<QCHEM>','Obtaining MDC charges failed')
         ENDIF
         I=INDEX(line,'Lower quality Stewart charges')
         IF(I.gt.0) THEN
            CALL WRNDIE(0,'<QCHEM>','MDC charges are of low quality')
         ENDIF
       ENDIF
#endif 

       if(QQCLJ) then 
          i=index(line,'Lennard-Jones')
          if(i.gt.0) then 
             read(line,'(31x,F13.10)')LJE
          endif
       endif

       if(QBLUCH) then
          i=index(line,'EGChgGChg:')
          if (i.gt.0) then
             read(line,*)tmp1,CC
          endif
       endif

50     FORMAT(3F12.7)
51     FORMAT(3F15.10)
52     FORMAT(F25.18)

       goto 20
200    continue
    ENDIF ! end test of QREADHESS 

!   write(*,*)'jflag= ',jflag
!   write(*,*)'NREStart = ',NREStart
    if(jflag.ne.-1) then
!      Make "5" a user setable option.... 
       if(jflag.lt.NREStart) then
          write(outu,22)'Q-Chem Job Failed... '
          write(outu,22)'Restarting Q-Chem Job'
          jflag=jflag+1
          goto 23
       endif
    endif

    !-----------------------------------------------------------
    !    GET QM AND EXTERNAL FIELD DERIVATIVES and CHARGES
    !-----------------------------------------------------------
    TMPVAR=MMATOMS
    IF(QQCRP) THEN
       open(imo,FILE=PWDQC(1:LPWD)//'/rpath'//SGROUP(1:M)// &
            '/efield.dat',status='unknown')
    ELSE
       open(imo,FILE='efield.dat',status='unknown')
    ENDIF
    !------------------------------------------------------

    !     CONVERT FROM A.U. TO KCAL/MOL
    DO I=1,MMATOMS
       read(imo,'(a)',end=300)line
       read(line,*)DUMX,DUMY,DUMZ
       IPT=MAPQM(I)
       !hlw_080705
       IF (IGMSEL(IPT).EQ.0) THEN
! for DIV - guanhua_puja_QC_UW1212
        IF (NDIV(IPT).EQ.0) THEN
          DX(IPT) = DX(IPT) + (DUMX*TOKCAL/BOHRR)*(-CGX(IPT))
          DY(IPT) = DY(IPT) + (DUMY*TOKCAL/BOHRR)*(-CGX(IPT))
          DZ(IPT) = DZ(IPT) + (DUMZ*TOKCAL/BOHRR)*(-CGX(IPT))
        ELSE IF (NDIV(IPT).EQ.1) THEN
          DX(IPT) = DX(IPT) + (DUMX*TOKCAL/BOHRR)*(-ZLHOST(IPT))
          DY(IPT) = DY(IPT) + (DUMY*TOKCAL/BOHRR)*(-ZLHOST(IPT))
          DZ(IPT) = DZ(IPT) + (DUMZ*TOKCAL/BOHRR)*(-ZLHOST(IPT))
        ENDIF
!
       ENDIF
       !hlw_080705
    ENDDO
    DO I=MMATOMS+1,MMATOMS+QMATOMS
       read(imo,'(a)',end=300)line
       read(line,*,iostat=iostat)DUMX,DUMY,DUMZ
! LNI format fix, in an attempt to cope with lines such as this:
!  13.97408441879924545503-116.04812193725601332517 -31.62032204259966761128
       if(iostat > 0) read(line,'(3F25.0)')DUMX,DUMY,DUMZ
       IPT=MAPQM(I)
       DX(IPT) = DX(IPT) + DUMX*TOKCAL/BOHRR
       DY(IPT) = DY(IPT) + DUMY*TOKCAL/BOHRR
       DZ(IPT) = DZ(IPT) + DUMZ*TOKCAL/BOHRR
    ENDDO

300 continue

    !    -----------------------------------   
    !      Start QM/MM Hessian Computation
    !    -----------------------------------   
    IF(QSECD .or. QNRAP) THEN
       IF(NDD1.LE.0)CALL WRNDIE(-5,'<QCHEM>','No second derivatives available.')
       QMDIM=3*(MMATOMS+QMATOMS)
       !write(*,*)'qmdim=',qmdim
       !         JUPT is still using old mem allocations???
       CALL FILUPT(JUPT,QMDIM)
       NDDX=(QMDIM*(QMDIM+1))/2
       OPEN(UNIT=99,FILE='hessian.dat',STATUS='UNKNOWN')
       !OPEN(UNIT=99,FILE='testhess.dat',STATUS='UNKNOWN')

       do i=1,NDDX 
          READ(99,52)tempdd1(i)
          !write(*,52)tempdd1(i)
       enddo
       !write(*,*)nddx
       !write(*,*)'hessian after read in from qchem'
       !do j=1,1000
       !   write(*,*)'DD1 = ',j,dd1(j)
       !enddo 

       DO IQ=1,MMATOMS+QMATOMS
          IIQ=IQ*3-2
          IC=MAPMM(IQ)
          II=IC*3-2

          !           Copy Diagonal Elements 
          IADD=IUPT(II)+II
          JADD=JUPT(IIQ)+IIQ
          DD1(IADD)=DD1(IADD)+tempdd1(JADD)        ! xx element
          !           write(*,*)DD1(IADD),tempdd1(JADD), IADD, JADD 
          DD1(IADD+1)=DD1(IADD+1)+tempdd1(JADD+1)  ! xy element
          DD1(IADD+2)=DD1(IADD+2)+tempdd1(JADD+2)  ! xz element 

          IADD=IUPT(II+1)+II+1
          JADD=IUPT(IIQ+1)+IIQ+1
          DD1(IADD)=DD1(IADD)+tempdd1(JADD)        ! yy element
          DD1(IADD+1)=DD1(IADD+1)+tempdd1(JADD+1)  ! yz element
          !           write(*,*)DD1(IADD),tempdd1(JADD), IADD, JADD 

          IADD=IUPT(II+2)+II+2
          JADD=IUPT(IIQ+2)+IIQ+2
          DD1(IADD)=DD1(IADD)+tempdd1(JADD)        ! zz element
          !           write(*,*)DD1(IADD),tempdd1(JADD), IADD, JADD 

          !           Copy Off-Diagonal Elements
          DO JQ=IQ+1,MMATOMS+QMATOMS
             JJQ=JQ*3-2
             JC=MAPMM(JQ)
             IF(JC.GT.IC) THEN
                JJ=JC*3-2
                II=IC*3-2

                IADD=IUPT(II)+JJ
                JADD=JUPT(IIQ)+JJQ
                !write(*,*)DD1(IADD),DD1(JADD),tempdd1(IADD),tempdd1(JADD), IADD, JADD 
                DD1(IADD)=DD1(IADD)+TEMPDD1(JADD)
                DD1(IADD+1)=DD1(IADD+1)+TEMPDD1(JADD+1)
                DD1(IADD+2)=DD1(IADD+2)+TEMPDD1(JADD+2)
                !           write(*,*)DD1(IADD),tempdd1(JADD), IADD, JADD 

                IADD=IUPT(II+1)+JJ
                JADD=JUPT(IIQ+1)+JJQ
                !write(*,*)DD1(IADD),DD1(JADD),tempdd1(IADD),tempdd1(JADD), IADD, JADD 
                DD1(IADD)=DD1(IADD)+TEMPDD1(JADD)
                DD1(IADD+1)=DD1(IADD+1)+TEMPDD1(JADD+1)
                DD1(IADD+2)=DD1(IADD+2)+TEMPDD1(JADD+2)
                !           write(*,*)DD1(IADD),tempdd1(JADD), IADD, JADD 

                IADD=IUPT(II+2)+JJ
                JADD=JUPT(IIQ+2)+JJQ
                !write(*,52)tempdd1(IADD),tempdd1(JADD)
                !write(*,*)DD1(IADD),DD1(JADD),tempdd1(IADD),tempdd1(JADD), IADD, JADD 
                DD1(IADD)=DD1(IADD)+TEMPDD1(JADD)
                DD1(IADD+1)=DD1(IADD+1)+TEMPDD1(JADD+1)
                DD1(IADD+2)=DD1(IADD+2)+TEMPDD1(JADD+2)

             ELSE 
                JJ=IC*3-2
                II=JC*3-2

                IADD=IUPT(II)+JJ
                JADD=JUPT(IIQ)+JJQ

                DD1(IADD)=DD1(IADD)+TEMPDD1(JADD) 
                !           write(*,*)DD1(IADD),tempdd1(JADD), IADD, JADD 
                IADD=IUPT(II+1)+JJ
                DD1(IADD)=DD1(IADD)+TEMPDD1(JADD+1) 
                !           write(*,*)DD1(IADD),tempdd1(JADD), IADD, JADD 
                IADD=IUPT(II+2)+JJ
                DD1(IADD)=DD1(IADD)+TEMPDD1(JADD+2) 
                !           write(*,*)DD1(IADD),tempdd1(JADD), IADD, JADD 

                IADD=IUPT(II)+JJ+1
                JADD=JUPT(IIQ+1)+JJQ

                DD1(IADD)=DD1(IADD)+TEMPDD1(JADD) 
                !           write(*,*)DD1(IADD),tempdd1(JADD), IADD, JADD 
                IADD=IUPT(II+1)+JJ+1
                DD1(IADD)=DD1(IADD)+TEMPDD1(JADD+1) 
                !           write(*,*)DD1(IADD),tempdd1(JADD), IADD, JADD 
                IADD=IUPT(II+2)+JJ+1
                DD1(IADD)=DD1(IADD)+TEMPDD1(JADD+2) 
                !           write(*,*)DD1(IADD),tempdd1(JADD), IADD, JADD 

                IADD=IUPT(II)+JJ+2
                JADD=JUPT(IIQ+2)+JJQ

                DD1(IADD)=DD1(IADD)+TEMPDD1(JADD) 
                !           write(*,*)DD1(IADD),tempdd1(JADD), IADD, JADD 
                IADD=IUPT(II+1)+JJ+2
                DD1(IADD)=DD1(IADD)+TEMPDD1(JADD+1) 
                !           write(*,*)DD1(IADD),tempdd1(JADD), IADD, JADD 
                IADD=IUPT(II+2)+JJ+2
                DD1(IADD)=DD1(IADD)+TEMPDD1(JADD+2) 
                !           write(*,*)DD1(IADD),tempdd1(JADD), IADD, JADD 

             ENDIF
          ENDDO
       ENDDO

       !    write(*,*)dd1(127),tempdd1(1)
       !    write(*,*)'hessian after read in from qchem'
       !    do i=1,100
       !       write(*,*)dd1(i)
       !    enddo

       !write(*,*)'QSAVEHESS = ; QREADHESS = ',QSAVEHESS, QREADHESS
       IF(.NOT.QSAVEHESS .and. .not.QREADHESS) THEN 
          call unlink('hessian.dat')
       ENDIF
    ENDIF

    IF(QREADGRAD) THEN
      do i=1,natom
         dx(i)=xtmpgrad(i)
         dy(i)=ytmpgrad(i)
         dz(i)=ztmpgrad(i)
      enddo
    ENDIF

    close(imo)
    close(omo)
    close(99)

    IF(QSAVEGRAD) THEN
       do i=1,natom
          xtmpgrad(i)=dx(i)
          ytmpgrad(i)=dy(i)
          ztmpgrad(i)=dz(i)
       enddo
    ENDIF

    !     SCALE FORCES BY BLOCK FACTOR FOR RPATH AND NEB
    IF(QQCRP) THEN
       DO I=1,NATOM
          DX(I) = DX(I) * blfactor
          DY(I) = DY(I) * blfactor
          DZ(I) = DZ(I) * blfactor
       ENDDO
    ENDIF
    
!    write(*,*) 'NC=', NC, 'CC=', CC, 'Eewald=', Eewald, TOKCAL
    IF(QQEWALD) THEN 
        WRITE(OUTU, '(A,F20.7)') 'QCHEM> MM Ewald Energy:', Eewald
    ENDIF 

    IF(QQCRP) THEN
       IF(QQCLJ) THEN
          !           Remove Charge-Charge and Lennard-Jones Interaction Energies
          E=(E-CC-LJE)*blfactor
       ELSE 
          !           Remove Charge-Charge Interaction and scale by block factor
          E=(E-CC)*blfactor
       ENDIF
    ELSE
       IF(QQCLJ) THEN
          !           Remove Charge-Charge and Lennard-Jones Interaction Energies
          E=(E-CC-LJE)
       ELSE
          !           Remove Charge-Charge Interaction
          E=E-CC                 
       ENDIF
    ENDIF

    IF(QSAVEINP) THEN
       CALL GET_ENVIRONMENT_VARIABLE(NAME="PWD",VALUE=PWD,STATUS=Stat)
       IF (Stat.ne.0) THEN
          WRITE(OUTU,22)'Current directory unknown, using /tmp'
          write(filecount,'(i10)')OPTCNT
          M=10
          CALL TRIMA(FILECOUNT,M)
          IF(QQCRP) THEN
             call system ('mkdir -p /tmp/saved_inputs'//SGROUP(1:M))
             call system ('cp '//TRIM(PWD)//'/rpath'//SGROUP(1:M)//   & 
             '/'//FILIN(P:LI)//' /tmp/saved_inputs'//SGROUP(1:M)//    & 
             '/qchem.inp_'//FILECOUNT)
          ELSE
             call system ('mkdir -p /tmp/saved_inputs')
             call system ('cp '//FILIN(P:LI)//                        & 
             ' /tmp/saved_inputs/qchem.inp_'//FILECOUNT)
          ENDIF
       ELSE
          write(filecount,'(i10)')OPTCNT
          M=10
          CALL TRIMA(FILECOUNT,M)
          IF(QQCRP) THEN
             call system ('mkdir -p '//TRIM(PWD)//'/rpath'//          & 
             SGROUP(1:M)//'/saved_inputs')
             call system ('cp '//TRIM(PWD)//'/rpath'//SGROUP(1:M)//      & 
             '/'//FILIN(P:LI)//' '//TRIM(PWD)//'/rpath'//SGROUP(1:M)//   & 
             '/saved_inputs/qchem.inp_'//FILECOUNT)
          ELSE
             call system ('mkdir -p '//TRIM(PWD)//'/saved_inputs')
             call system ('cp '//TRIM(PWD)//'/'//FILIN(1:LI)//        & 
             ' '//TRIM(PWD)//'/saved_inputs/qchem.inp_'//FILECOUNT)
          ENDIF
       ENDIF
    ENDIF

    IF(QSAVEOUT) THEN
       CALL GET_ENVIRONMENT_VARIABLE(NAME="PWD",VALUE=PWD,STATUS=Stat)
       IF (Stat.ne.0) THEN
          WRITE(OUTU,22)'Current directory unknown, using /tmp'
          write(filecount,'(i10)')OPTCNT
          M=10
          CALL TRIMA(FILECOUNT,M)
          IF(QQCRP) THEN
             call system ('mkdir -p /tmp/saved_outputs'//SGROUP(1:M))
             call system ('cp '//TRIM(PWD)//'/rpath'//SGROUP(1:M)//   & 
             '/'//FILOUT(P:LO)//' /tmp/saved_outputs'//SGROUP(1:M)//  & 
             '/qchem.inp_'//FILECOUNT)
          ELSE
             call system ('mkdir -p /tmp/saved_outputs')
             call system ('cp '//FILOUT(P:LO)//                       & 
             ' /tmp/saved_outputs/qchem.inp_'//FILECOUNT)
          ENDIF
       ELSE
          write(filecount,'(i10)')OPTCNT
          M=10
          CALL TRIMA(FILECOUNT,M)
          IF(QQCRP) THEN
             call system ('mkdir -p '//TRIM(PWD)//'/rpath'//          & 
             SGROUP(1:M)//'/saved_outputs')
             call system ('cp '//TRIM(PWD)//'/rpath'//SGROUP(1:M)//   & 
             '/'//FILOUT(P:LO)//' '//TRIM(PWD)//'/rpath'//            & 
             SGROUP(1:M)//'/saved_outputs/qchem.out_'//FILECOUNT)
          ELSE
             call system ('mkdir -p '//TRIM(PWD)//'/saved_outputs')
             call system ('cp '//TRIM(PWD)//'/'//FILOUT(1:LO)//' '//  & 
             TRIM(PWD)//'/saved_outputs/qchem.out_'//FILECOUNT)
          ENDIF
       ENDIF
    ENDIF

    !write(*,*)'Modifing OPTCNT'
    OPTCNT = OPTCNT + 1
    !     Remove old efield.dat file
    IF(QQCRP) THEN
        call unlink( PWDQC(1:LPWD)//'/rpath' &
             //SGROUP(1:M)//'/efield.dat' )      
    ELSE
!      call system ('cp efield.dat efield.cur')
       call unlink('efield.dat')
       call unlink('qchosts')
    ENDIF

    !write(50+mynodg,'(a,l3,10i4)')'qchem>end::qqcrp,mynod,numnod=',qqcrp,mynod,numnod
    !write(*,*)'Leaving Q-Chem Subroutine....'

    RETURN
  END SUBROUTINE QCHEM
#endif /* (qchem)*/

    ! Guanhua_QC_UW1111: this subroutine takes care of G09 calculations
#if KEY_G09==1 /*g09*/
  SUBROUTINE qg09(E,DX,DY,DZ,CGX,AMASSX,IACX)
  !--------------------------------------
  !  Run Gaussian09 and parse the output
  !--------------------------------------
    use chm_kinds
    use number
    use dimens_fcm
    use gamess_fcm
    use memory
    use psf
    use rtf,only:atct
    use param
    use coord
    use parallel
    use consta
    use stream
    use string
    use code
    use scalar_module
    use linkatom, only: findel ! JZ_UW12
#if KEY_SMBP==1
    use pbeq,only:qsmbp,numsurf,qsmbp_qc_grad  /* JZ_UW12: For SMBP*/
#endif
    implicit none
    !
    real(chm_real) E,DX(*),DY(*),DZ(*),CGX(*),AMASSX(*)
    INTEGER IACX(*)
! local variables
    real(chm_real) cc
    INTEGER LC,LI,IMO,OMO,lr,lk,lpr,le,lf,ls,lgs,tmplen
    INTEGER ILEN,MMATOMS,QMATOMS
    INTEGER I,J,K,L,M,N,O,P,R
    INTEGER MAPQM(MAXA),MAPMM(MAXA),IPT,IERR       
    CHARACTER(len=255) FILCMD,FILIN,FILOUT,filchk,filfchk,filtmp
    character(len=255) g09profile,g09exe,g09fchk,g09scr
    CHARACTER(len=80) LINE
    CHARACTER(len=6) ELE
    CHARACTER(len=6) tmp1,tmp2
    real(chm_real) DUMX,DUMY,DUMZ
    real(chm_real) TMPX,TMPY,TMPZ,RX,RY,RZ,S,t
    logical qgmethod,qgbasis,qgcharge,qgspin,qgtitle,qgextra
    logical qgnproc,qgmem,qggrad
    character(len=100) gtitle
    character(len=40) gmethod,gbasis,gcharge,gspin,gextra
    character(len=40) gnproc,gmem,grestart
    character(len=20) zlhost_char,cgx_char
    character(len=1) k_char
    real(chm_real), dimension(5) ::  ggrad
    ! JZ_UW12
    real(chm_real), dimension(5) ::  gesp 
    integer nsurf,jmo,kmo,lmo
    logical qgforce,qgpop,qgsmbp
    character(len=40) gforce,gsmbp,gpop
    ! Puja_QC_UW1212
    logical qgextend,lqinigm
    character(len=80) gextend
    integer gmo,nwritext

#if KEY_PARALLEL==1
    IF(MYNOD.GT.0) THEN
       E=ZERO
       RETURN
    ENDIF
#endif 


! Guanhua_QC_UW1111: read in the name of G09 keyword file 
    call get_environment_variable("G09PROFILE", g09profile, lpr)
    if (lpr.eq.0) call wrndie(-5,'<G09>','No G09 profile specified')

    call get_environment_variable("G09EXE", g09exe, le)
    if (le.eq.0) call wrndie(-5,'<G09>','No G09 executable specified')

    call get_environment_variable("G09FCHK", g09fchk, lf)
    if (lf.eq.0) call wrndie(-5,'<G09>','No G09 formchk specified')

    CALL GET_ENVIRONMENT_VARIABLE("G09CMD", FILCMD, LC)
    IF(LC.EQ.0)CALL WRNDIE(-5,'<G09>','No keyword file specified.')

    FILIN=''
    CALL GET_ENVIRONMENT_VARIABLE("G09INP", filtmp, li)
    IF(LI.EQ.0)CALL WRNDIE(-5,'<G09>','No input specified.')

! JZ_UW12: Check for GAUSS_SCRDIR and put checkpoint file there
    CALL GET_ENVIRONMENT_VARIABLE("GAUSS_SCRDIR", g09scr, lgs)
    IF(LGS.EQ.0)CALL WRNDIE(-5,'<G09>','GAUSS_SCRDIR not specified.')


    filout=filtmp(1:li)//'.log'
    filchk=g09scr(1:lgs)//'/'//filtmp(1:li)//'.chk' ! JZ
    filfchk=filtmp(1:li)//'.fchk'
    filin=filtmp(1:li)//'.inp'

    lr=strlng(filin)
    lk=strlng(filfchk)
    ls=strlng(filchk) ! JZ

    IMO=90
    OMO=91
    JMO=92 ! JZ
    KMO=93 ! JZ
    GMO=94 ! Puja_QC_UW1212

    open(gmo,file='extend.dat',status='replace') ! Puja_QC_UW1212
    OPEN(UNIT=90,FILE=FILCMD(1:LC),STATUS='OLD')
    OPEN(UNIT=91,FILE=FILIN(1:lr),STATUS='REPLACE')

! Guanhua_QC_UW1111: setup some flags for reading in different cmds 
    qgmethod=.false.
    qgbasis=.false.
    qgcharge=.false.
    qgspin=.false.
    qgtitle=.false.
    qgextra=.false.
    qgnproc=.false.
    qgmem=.false.
    qgpop=.false.  ! JZ
    qgforce=.true. ! JZ
    qgextend=.false. ! Puja_QC_UW1212
    nwritext=0 ! Puja_QC_UW1212

    gmethod=''
    gbasis=''
    gcharge=''
    gspin=''
    gtitle='dummy title'  !JZ: This is important --> g09 needs a title!
    gextra=''
    gnproc=''
    gmem=''
    grestart=''
    gsmbp=''        ! JZ
    gpop=''         ! JZ
    gforce='force iop(1/10=10,7/29=2)'  ! JZ: this iop tells g09 not to generate
                                        !     the approximate hessian (i.e., set to unit matrix)
    gextend='' ! Puja_QC_UW1212
 
! JZ: Default for SMBP
#if KEY_SMBP==1
    if (qsmbp) gpop='pop=mk'  
#endif

 10 READ(IMO,'(A)',END=100)LINE
    ILEN=80
! Guanhua: rm redundant spaces
    CALL TRIMA(LINE,ILEN)
! Guanhua: tranform to lower case
    CALL CNVTLC(LINE,ILEN)

    if (strfin(line,'$method') .gt. 0) then
      qgmethod=.true.
      goto 10
    endif

    if (strfin(line,'$basis') .gt. 0) then
      qgbasis=.true.
      goto 10
    endif

    if (strfin(line,'$title') .gt. 0) then
      qgtitle=.true.
      goto 10
    endif

    if (strfin(line,'$charge') .gt. 0) then
      qgcharge=.true.
      goto 10
    endif

    if (strfin(line,'$spin') .gt. 0) then
      qgspin=.true.
      goto 10
    endif

    if (strfin(line,'$pop') .gt. 0) then
      qgpop=.true.
      goto 10
    endif

    if (strfin(line,'$extra') .gt. 0) then
      qgextra=.true.
      goto 10
    endif

    if (strfin(line,'$nproc') .gt. 0) then
      qgnproc=.true.
      goto 10
    endif

    if (strfin(line,'$mem') .gt. 0) then
      qgmem=.true.
      goto 10
    endif

    if (strfin(line,'$extend') .gt. 0) then ! Puja_QC_UW1212
      qgextend=.true.
      goto 10
    endif

    if (qgmethod) then
      gmethod=line
      qgmethod=.false.
    endif

    if (qgbasis) then
      gbasis=line
      qgbasis=.false.
    endif

    if (qgtitle) then
      tmplen=strlng(line)
      if (tmplen .eq. 0) then
        gtitle='dummy title' ! JZ: g09 needs a title!
      else
        gtitle=line
      endif
      qgtitle=.false.
    endif

    if (qgcharge) then
      gcharge=line
      qgcharge=.false.
    endif

    if (qgspin) then
      gspin=line
      qgspin=.false.
    endif

    if (qgpop) then
      tmplen=strlng(line)
      if (qsmbp .and. (tmplen .eq. 0)) then
        gpop='pop=mk'
      else
        gpop=line
      endif
      qgpop=.false.
    endif

    if (qgextra) then
      gextra=line
      qgextra=.false.
    endif

    if (qgnproc) then
      gnproc=line
      qgnproc=.false.
    endif

    if (qgmem) then
      gmem=line
      qgmem=.false.
    endif
  
    if (qgextend) then ! Puja_QC_UW1212
      gextend=line

! If the first line is non-empty,read the rest of the lines
! till end-of-file and save them in 'extend.dat'

! Remember that the first line has been stored in 'gextend'

      tmplen=strlng(gextend)
      if (tmplen.gt.0) then
277      READ(IMO,'(A)',END=100)LINE 
         ILEN=80
         CALL TRIMA(LINE,ILEN)
         CALL CNVTLC(LINE,ILEN)
         nwritext=nwritext+1
         write(gmo,'(A)')LINE
         goto 277
      endif

      qgextend=.false.
    endif  ! end of 'extend' section

    IF(OPTCNT.GT.0) THEN
       grestart='guess=read'
    ENDIF

    GOTO 10
100 CONTINUE

    if(qgextend)qgextend=.false.  ! Puja_QC_UW1212

! Guanhua_QC_UW1111: setup the input and record things in charmm output
    write(omo,'(A)') '%chk='//filchk(1:ls) ! JZ: lr --> ls
    write(omo,'(A)') '%nproc='//gnproc
    write(omo,'(A)') '%mem='//gmem
! JZ_UW12: For SMBP, change jobtype if necessary
#if KEY_SMBP==1
    IF (QSMBP) THEN
      IF (QSMBP_QC_GRAD) THEN
        gsmbp='scf=(maxcycle=1,conver=1)'
      ELSE 
        qgforce=.false.
        gforce=''
        gsmbp=''
      ENDIF
    ENDIF
#endif 
!   The final iop option prevents g09 from aborting in case of small atom dist
    write(omo,'(A)') '#P '//trim(gmethod)//'/'//trim(gbasis) &
                     //' nosymm '//trim(gforce)//' '  &
                     //trim(grestart)//' '//trim(gextra)//' ' &
                     //trim(gpop)//' '//trim(gsmbp)//' iop(2/12=1)' 
    write(omo,'(A)') ''
    write(omo,'(A)') gtitle
    write(omo,'(A)') ''
    write(omo,'(A)') trim(gcharge)//' '//gspin
! Guanhua: now write out the coordinate and charges
! first loop to wirte out qm atoms
    QMATOMS=0
    DO I=1, NATOM
      IF((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2))THEN
         QMATOMS=QMATOMS+1
         mapqm(qmatoms)=i
         ELE='      '
         lqinigm=qinigm
         CALL FINDEL(ATCT(IACX(I)),AMASSX(I),I,ELE,T,LQINIGM)
         IF(ELE(1:3).EQ.'QQH')ELE(1:6)=' H    '
         IF(ELE(1:3).EQ.'qqh')ELE(1:6)=' H    '
         WRITE(OMO,'(A6,3F20.10)')ELE,X(I),Y(I),Z(I)
      ENDIF
    ENDDO
    QINIGM=.FALSE.
    lqinigm=qinigm

! JZ_QC_UW_1111: SMBP
    NSURF=0
#if KEY_SMBP==1
    IF (QSMBP) NSURF=NUMSURF  
#endif

! loop over MM point charges
    MMATOMS=0
    DO I=1, NATOM + NSURF
      IF(((IGMSEL(I).EQ.0).OR.(IGMSEL(I).EQ.5))) THEN
        MMATOMS = MMATOMS+1
        mapmm(mmatoms)=i
        ele='      '
        lqinigm=qinigm
        CALL FINDEL(ATCT(IACX(I)),AMASSX(I),I,ELE,T,LQINIGM)
        IF(IGMSEL(I).EQ.5) THEN
          WRITE(OMO,'(a,3F16.8)') trim(ele)//'-#6-' &
                  //'0.0',X(I),Y(I),Z(I)
! for DIV - guanhua_puja_QC_UW1212
        ELSE IF(NDIV(I).eq.1) then
          write(zlhost_char,'(f16.8)') zlhost(i)
          write(OMO,'(a,3F16.8)') trim(ele)//'-#6-' &
                  //adjustl(zlhost_char),x(i),y(i),z(i)
!
        ELSE
          write(cgx_char,'(f16.8)') cgx(i)
          WRITE(OMO,'(a,3F16.8)') trim(ele)//'-#6-' &
                  //adjustl(cgx_char),X(I),Y(I),Z(I)
        ENDIF
! JZ_UW12: Add surface charges for SMBP
#if KEY_SMBP==1
      ELSEIF (QSMBP .and. (IGMSEL(I) .eq. 6)) THEN
        write(cgx_char,'(f16.8)') cgx(i)
        WRITE(OMO,'(a,3F16.8)') trim(ele)//'-#6-' &
               //adjustl(cgx_char),X(I),Y(I),Z(I)
#endif 
      ENDIF
    ENDDO

    tmplen=strlng(gextend) ! Puja_QC_UW1212
    if(tmplen.gt.0)then
     write(omo,'(A)') ''
     write(omo,'(A)') gextend
     if(nwritext.gt.0)then
       rewind(gmo)
       do i=1,nwritext
         read(gmo,'(A)')line
         write(omo,'(A)')line
       enddo
     endif  
    endif

! last line of G09 input needs to be blank
    write(omo,'(A)') ''

22  FORMAT('G09> ',A)

!   Echo G09 job info into CHARMM output file
    IF(PRNLEV.GE.2) THEN
      IF(OPTCNT.EQ.0) THEN
          WRITE(OUTU,22)'G09 Job Parameters'
          WRITE(OUTU,22)'---------------------'
          WRITE(OUTU,22) 'Method: '//gmethod
          WRITE(OUTU,22) 'Basis: '//gbasis
      ENDIF
    ENDIF

    close(omo)
    close(imo)
    close(gmo) !Puja_QC_UW1212
! done with the input setup


! Guanhua_QC_UW1111: run Gaussian
    CALL SYSTEM('source '//g09profile(1:lpr)//';' &
                //g09exe(1:le)//' '//FILIN(1:Lr)//';' &
                //g09fchk(1:lf)//' '//filchk(1:ls)//' ' & ! JZ: lr --> ls
                //'>&/dev/null')

! JZ_UW12: move filfchk to current WD
    CALL SYSTEM('mv -f '//g09scr(1:lgs)//'/'//filfchk(1:lk)//' .')

!   write(*,*) "QG09> Returned"
!   write(*,*) "QG09> Processing output"

! Guanhua_QC_UW1111: readout G09 output
    E=ZERO
    cc=zero
    T=ZERO
    qggrad=.false.
    qgsmbp=.false.

    open(omo,FILE=filfchk(1:lk),status='old')
    open(imo,file='gradient.dat',status='unknown')
#if KEY_SMBP==1
    if (qsmbp) then
      open(jmo,file='esp_charges.dat',status='unknown') 
      open(kmo,file='charges.dat',status='unknown')     
    endif
#endif 

20  read(omo,'(a)',end=200) line

!   write(*,'(a)') line
    i=index(line,'Total Energy')
    if (i.gt.0) then
      read(line,'(t45,g27.16)') e
    endif

!   write(*,*) "Energy found:",e

! Guanhua: do a little bit formating for gradient
    i=index(line,'Cartesian Gradient')
    if (i.gt.0) then
      qggrad=.true.
      read (line,'(t50,i20)') n
      m=n/5
      k=mod(n,5)
      write (k_char,'(i1)') k
!     write(*,*) "Gradient found:",n,k
      goto 20
    endif

    if (qggrad) then
      m=m-1
!     write(*,*) "Getting gradient row", m
      if (m.ge.0) then
        read (line,'(5g16.9)') (ggrad(i),i=1,5)
        do i=1,5
          write (imo,'(g16.9)') ggrad(i)
        enddo
      else
!     QC: have to deal with the case when k=0
!       write(*,*) "Potentially more rows?",k
        if (k.ne.0) then 
         read (line,'('//k_char//'g16.9)') (ggrad(i),i=1,k)
         do i=1,k
           write (imo,'(g16.9)') ggrad(i)
         enddo
        endif
        qggrad=.false.
!       write(*,*) "Done gradinet",m,qggrad
        goto 200
      endif
    endif

! Guanhua: need to take care of error messages later
!      i=index(line,'Convergence failure')
!      if(i.gt.0) THEN
!         WRITE(OUTU,22)'SCF Convergence Failure!!!'
!         WRITE(OUTU,22)'Try adding "SCF_ALGORITHM DIIS_GDM" to QM file'
!         WRITE(OUTU,22)'Or increase the number of scf iterations....'
!         WRITE(OUTU,22)'MAX_SCF_CYCLES 100'
!         STOP
!      ENDIF

! JZ_UW12: Read ESP charges for SMBP
#if KEY_SMBP==1
    if (qsmbp) then
      i=index(line,'ESP Charges')
      j=index(line,'Mulliken Charges')
      if (i.gt.0 .or. j.gt.0) then
        qgsmbp=.true.
        if (i.gt.0) lmo=jmo 
        if (j.gt.0) lmo=kmo
        read (line,'(t50,i20)') n
        m=n/5
        k=mod(n,5)
        write (k_char,'(i1)') k
        goto 20
      endif
    
      if (qgsmbp) then
        m=m-1
        if (m.ge.0) then
          read (line,'(5g16.9)') (gesp(i),i=1,5)
          do i=1,5
            write (lmo,'(e16.9)') gesp(i)
          enddo
        else
          read (line,'('//k_char//'g16.9)') (gesp(i),i=1,k)
          do i=1,k
            write (lmo,'(e16.9)') gesp(i)
          enddo
          qgsmbp=.false.
        endif
      endif
    endif
#endif 

    goto 20
200 continue
!   write(*,*) "G09> Processed gradient"

    close(omo)
#if KEY_SMBP==1
    if (qsmbp) then
      close(jmo) 
      close(kmo) 
    endif
#endif 

    open(omo,FILE=filout(1:lr),status='old')

30  read(omo,'(a)',end=300) line

    i=index(line,'Nuclear repulsion from inactive atom pairs')
    if (i.gt.0) then
      read(line,'(t45,g20.10)') cc
      goto 300
    endif

#if KEY_SMBP==1
!     JZ: Check for g09 problems with close proximities of surface
!         charges and QM atoms
    if (qsmbp) then
      i=index(line,'Problem with the distance matrix')
      if (i.gt.0) then
        call wrndie(-5,'<QG09>','surface charges too close to QM region')
      endif
    endif
#endif 

    goto 30
300 continue

    close(omo)
!   write(*,*) "G09> Processed nuclear repulsion inactive",cc

! remove the contribution from charge-charge interactions
    e=e-cc

!   write(*,*) "G09> Adding force",qgforce
    if (qgforce) then
      rewind(imo)
! read in force for QM first
      do i=1,qmatoms
        read (imo,'(g16.9)') dumx
        read (imo,'(g16.9)') dumy
        read (imo,'(g16.9)') dumz
        IPT=MAPQM(I)
        dx(ipt)=dx(ipt)+ DUMX*TOKCAL/BOHRR
        dy(ipt)=dy(ipt)+ DUMY*TOKCAL/BOHRR
        dz(ipt)=dz(ipt)+ DUMZ*TOKCAL/BOHRR
      enddo

! readin force for MM then
      do i=1,mmatoms
        read (imo,'(g16.9)') dumx
        read (imo,'(g16.9)') dumy
        read (imo,'(g16.9)') dumz
        IPT=mapmm(I)
        IF (IGMSEL(IPT).EQ.0) THEN
          DX(IPT) = DX(IPT) + (DUMX*TOKCAL/BOHRR)
          DY(IPT) = DY(IPT) + (DUMY*TOKCAL/BOHRR)
          DZ(IPT) = DZ(IPT) + (DUMZ*TOKCAL/BOHRR)
        ENDIF
      enddo
    endif 

    close(imo)
!   write(*,*) "QG09> Completed processing data"

! delete scratch files
    call unlink('gradient.dat')
    call system('rm ./Gau-* >& /dev/null')
 
    OPTCNT = OPTCNT + 1

    RETURN
    END SUBROUTINE qg09
#endif /* (g09)*/
      
#if KEY_QCHEM==1 || KEY_G09==1
  subroutine mmpotqmew(qmatoms,natom,mmpotqm,x,y,z,cg)
    use number
    use gamess_fcm
    use ewald_1m, only: ewvirial,kappa
    use pme_module
    use memory
    use image
    use prssre

    ! Box 6 IN Case's paper
 
    integer qmatoms,natom
    !real(chm_real) :: mmpotqm(:),x(:),y(:),z(:),cg(:) ! JZ_UW12
    real(chm_real) :: mmpotqm(:),x(:),y(:),z(:),cg(*)

    real(chm_real),allocatable, dimension (:) :: dx,dy,dz,cgq,fqcfor,mcgq
    real(chm_real) EWKSUM,EWSELF,EWQCOR,EWUTIL,cgtot
    logical :: qewksum=.true.,qewself=.false.,qewqcor=.false.,qewutil=.false.
    integer iqmatom,iatom
    ! to be replaced....
    real(chm_real) :: volume,recip(6)
    logical :: ok

    ! We don't need the forces here??
    call chmalloc('gukini.src','MMPOTQMEW','DX',natom,crl=dx)
    call chmalloc('gukini.src','MMPOTQMEW','DY',natom,crl=dy)
    call chmalloc('gukini.src','MMPOTQMEW','DZ',natom,crl=dz)
    call chmalloc('gukini.src','MMPOTQMEW','CGQ',natom,crl=cgq)
    call chmalloc('gukini.src','MMPOTQMEW','MCGQ',natom,crl=mcgq)

    call getvol(volume)
    call invt33s(recip,xtlabc,ok)

    mmpotqm(1:qmatoms)=zero
    mcgq(1:qmatoms)=zero
    !cgtot=sum(cg) ! we need to add QM charge here - de we have one??? ! JZ_UW12
    cgtot=sum(cg(1:natom)) ! we need to add QM charge here - de we have one???

    cgq(1:natom)=cg(1:natom)

    do iatom=1,natom
       if(igmsel(iatom) == 1) cgq(iatom)=zero
    enddo

    ! get the potential on QM atom postions from MM atoms
    
!!    CALL PME(EWKSUM,EWSELF,EWQCOR,EWUTIL, &
!!         QEWKSUM,QEWSELF,QEWQCOR,QEWUTIL, &
!!         X,Y,Z,DX,DY,DZ,NATOM,CGQ,CGTOT,ewvirial,kappa,QPHMD) ! Yandong Huang, Jul 2016
    ! we might start by using routines from squantum:

    !scf_mchg_2 is mulliken charge => mcgq
    ! recip,volume,kappa analogous to gpupme
    
    ! See what we are sending to SQUANTM:
    !write(*,'(a,2(i0,1x),f0.3)')'mmpotqmew>qmatoms,natom,kappa=',qmatoms,natom,kappa
    !write(*,'(a,100(1x,f0.3))')'mmpotqmew>cgq=',cgq(1:natom)
    !write(*,'(a,100(1x,f0.8))')'mmpotqmew>mull=',mcgq(1:qmatoms)
    !write(*,'(a,100(1x,f0.8))')'mmpotqmew>volume=',volume
    !write(*,'(a,100(1x,f0.8))')'mmpotqmew>recip=',recip

    !This is not cmopleted yet, so temporarily disabled for QCHEM default compile
    !call qm_pme_mm_pot(natom,qmatoms,x,y,z,cgq,mcgq,  &
    !     recip,volume, mmpotqm, kappa)

    call chmdealloc('gukini.src','MMPOTQMEW','DX',natom,crl=dx)
    call chmdealloc('gukini.src','MMPOTQMEW','DY',natom,crl=dy)
    call chmdealloc('gukini.src','MMPOTQMEW','DZ',natom,crl=dz)
    call chmdealloc('gukini.src','MMPOTQMEW','CGQ',natom,crl=cgq)
    call chmdealloc('gukini.src','MMPOTQMEW','MCGQ',natom,crl=mcgq)

    return
  end subroutine mmpotqmew
  
  subroutine qmpotqmew(qmatoms,natom,qmpotqm)
    use number
    use gamess_fcm

    integer qmatoms,natom
    real(chm_real) qmpotqm(:,:)

    qmpotqm(1:qmatoms,1:qmatoms)=zero

    return
  end subroutine qmpotqmew

#endif /* */
  !     
    ! implementation of interface to TURBOMOLE
    ! details in Riahi, S., Rowley C.N. The CHARMM-TURBOMOLE Interface for Efficient and 
    ! Accurate QM/MM Molecular Dynamics, Free Energies, and Excited State Properties. 
    ! J. Comput. Chem. 2014, DOI: 10.1002/jcc.23716
#if KEY_QTURBO==1 /*qturbo*/
  SUBROUTINE QTURBOINI
    !-----------------------------------------------------------------------
    !     Determine the number of QM atoms to pass to Turbomole
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use gamess_fcm
    use psf
    !
    INTEGER I

    !
    NGAMES = 0
    DO I=1, NATOM
       IF((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2))THEN
          NGAMES=NGAMES+1
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE QTURBOINI
  
  SUBROUTINE QTURBO(E,DX,DY,DZ,CGX,AMASSX,IACX,NDD1,DD1,QSECD,IUPT,JUPT)
    !-----------------------------------------------------------------------
    !     Run TURBOMOLE and parse the output
    !-----------------------------------------------------------------------
    use chm_kinds
    use number
    use dimens_fcm
    use gamess_fcm
    use memory
    use psf
    use param
    use coord
    use parallel
    use consta
    use eutil
    use storage
    use stream
    use string
    use replica_mod
    use block_fcm
    use pathm
    use pert
    use code
    use scalar_module
    use lonepr,only: LKATMS
    use ewald_1m,only:lewald
    use chm_kinds
    use rtf,only:atct
    use linkatom, only: findel
    !
    real(chm_real) E,DX(*),DY(*),DZ(*),CGX(*),AMASSX(*)
    INTEGER IACX(*),NDD1,NDDX
    real(chm_real) DD1(*)
    INTEGER IUPT(*),JUPT(*)
    LOGICAL QSECD
    !
    INTEGER LC,LI,LO,LX,IMO,OMO,OUT,ix,LNP,PROCS,FMO,LCR
    INTEGER ILEN,IFLAG,JFLAG,MMATOMS,QMATOMS,TMPVAR
    INTEGER I,J,K,L,M,N,O,P,R,LPWD,LPA,LPB,STAT
    INTEGER MAPQM(MAXA),MAPMM(MAXA),IPT,JPT,IERR       ! lw050728
    INTEGER MAPQM2(MAXA)                               !hlw_080705
    INTEGER NUMFF,MINFF
    CHARACTER(len=255) FILCNT,FILIN,FILOUT,FILEXE,PWDQC,SGROUP,PWDESP
    CHARACTER(len=255) TMPA,PRTINA,PRTINB,PRTOUTA,PRTOUTB,PWDQCH,PWD
    CHARACTER(len=6) ELE,NP,NCIS
    CHARACTER(len=6) tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9
    CHARACTER(len=10) corr,filecount
    CHARACTER(len=32) junkene
    CHARACTER(len=80) rem(25)
    LOGICAL LPRT,LEXIST
    real(chm_real) T,NC,CC,DUMX,DUMY,DUMZ,LJE
    real(chm_real) TMPX,TMPY,TMPZ,RX,RY,RZ,S,S2
    real(chm_real) tmpxx(natom),tmpyy(natom),tmpzz(natom)
    INTEGER QMDIM,IQ,IIQ,IC,II,IADD,JADD,JQ,JJQ,JC,JJ,I1,LJCOUNT
    real(chm_real),allocatable, dimension(:) :: tempdd1
    real(chm_real),allocatable, dimension(:) :: mmpotqm
    real(chm_real),allocatable, dimension(:,:) :: qmpotqm
    INTEGER NLK, ILK, IQMH, IMMH
    CHARACTER(len=255) CMDLINE, LINE
    CHARACTER(len=255) TURBOPWD, TURBOINP, TURBOEXE
    INTEGER LTURBOPWD, LTURBOINP, LTURBOEXE
    INTEGER readstatus

    IF(QSECD .or. QNRAP) THEN 
       allocate(tempdd1((3*natom*(3*natom+1))/2),stat=ierr)
    ENDIF
    !
    LPRT=.false.


    CALL GET_ENVIRONMENT_VARIABLE("QTURBOOUTPATH", TURBOPWD, LTURBOPWD)
    IF(LTURBOPWD.EQ.0)CALL WRNDIE(-5,'<QTURBO>','No output path specified.')
    CALL GET_ENVIRONMENT_VARIABLE("QTURBOINPATH", TURBOINP, LTURBOINP)
    IF(LTURBOINP.EQ.0)CALL WRNDIE(-5,'<QTURBO>','No input path specified.')
    !
    CALL GET_ENVIRONMENT_VARIABLE("QTURBOEXE", TURBOEXE, LTURBOEXE)
    IF(LTURBOEXE.EQ.0)CALL WRNDIE(-5,'<QTURBO>','No TURBOMOLE execution script specified.')
    !

    OPEN(UNIT=91,FILE=TURBOPWD(1:LTURBOPWD)//'/point_charges',STATUS='REPLACE')

    write(91,'(A)')'$point_charges'
    MMATOMS=0
    JPT=1
    DO I=1, NATOM
        IF((IGMSEL(I).EQ.0).OR.(IGMSEL(I).EQ.5))THEN
          MMATOMS = MMATOMS+1
          MAPMM(MMATOMS)=I
          TMPX = X(I)/BOHRR
          TMPY = Y(I)/BOHRR
          TMPZ = Z(I)/BOHRR

          IF(IGMSEL(I).EQ.0) THEN
            WRITE(91,'(4F16.8)') TMPX,TMPY,TMPZ,CGX(I)
          ENDIF
        ENDIF
    ENDDO
    write(91,'(A)')'$end'
    close(91)

!   turbomole start (write coord file)
    QMATOMS=0
    MMATOMS=0
    IPT=0
    JPT=0
    OPEN(UNIT=91,FILE=TURBOPWD(1:LTURBOPWD)//'/coord',STATUS='REPLACE')
    write(91,'(A)')'$coord'
    DO I=1, NATOM
       IF((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2))THEN
          ELE='      '
          CALL FINDEL(ATCT(IACX(I)),AMASSX(I),I,ELE,T,.TRUE.)
          IF(ELE(1:3).EQ.'QQH')ELE(1:6)=' H    '
          IF(ELE(1:3).EQ.'qqh')ELE(1:6)=' H    '
          TMPX = X(I)/BOHRR
          TMPY = Y(I)/BOHRR
          TMPZ = Z(I)/BOHRR
          WRITE(91,'(3F20.10,X,A5)')TMPX,TMPY,TMPZ,ELE
          IPT=IPT+1
          MAPQM(IPT)=I
          QMATOMS=QMATOMS+1
       ELSE
          IF((IGMSEL(I).EQ.0).or.(igmsel(i).eq.5))THEN
             JPT=JPT+1
             MAPMM(JPT)=I
             MMATOMS=MMATOMS+1
          ENDIF
       ENDIF
    ENDDO
    write(91,'(A)')'$'
    close(91)

!       turbomole (start execute turbomole )
!       the script executed here should run ridft and rdgrad or dscf and ricc2
!       to generate a gradient file
!       an FEP calculation is being performed, an additional argument describing
!       the state is passed to TURBOEXE

        IF(QPERT) THEN
          write(CMDLINE,'(A,X,A,X,A,X,I1)') TURBOEXE(1:LTURBOEXE), TURBOINP(1:LTURBOINP),&
       &        TURBOPWD(1:LTURBOPWD), QMSTATE
        ELSE
          write(CMDLINE,'(A,X,A,X,A)') TURBOEXE(1:LTURBOEXE), TURBOINP(1:LTURBOINP), TURBOPWD(1:LTURBOPWD)
        ENDIF
        CALL SYSTEM(CMDLINE)

#if KEY_PARALLEL==1
    IF(MYNOD.GT.0) THEN
       E=ZERO
       RETURN
    ENDIF
#endif 

! turbomole start (read energy and gradients)

    OPEN(UNIT=90,FILE=TURBOPWD(1:LTURBOPWD)//'/gradient',STATUS='UNKNOWN')

    read(90,'(a)') line
    read(90,'(a)') line
    read(line,*) tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,e

    do j=1,QMATOMS
       ipt=mapqm(j)
       read(90,'(A)') line
       if(PRNLEV.gt.4) THEN
          write(OUTU,'(A)') line
       ENDIF
       read(line, '(3D22.14,A)') DUMX,DUMY,DUMZ,tmp1
       if(QQCOORD) then
         X(IPT)=DUMX
         Y(IPT)=DUMY
         Z(IPT)=DUMZ
       endif
    enddo

    do I=1,QMATOMS
       ipt=mapqm(I)
       read(90,'(A)') line
       IF(PRNLEV.ge.4) THEN
         write(OUTU,*) line
       ENDIF

       read(LINE, '(3D22.14)') DUMX,DUMY,DUMZ

       DX(IPT) = DX(IPT) + (DUMX*TOKCAL/BOHRR)
       DY(IPT) = DY(IPT) + (DUMY*TOKCAL/BOHRR)
       DZ(IPT) = DZ(IPT) + (DUMZ*TOKCAL/BOHRR)
       IF(PRNLEV .gt. 4) THEN
         write(OUTU,'(A,X,I4,X,I4,X,3D22.14)') 'G QM ',I,IPT,DUMX,DUMY,DUMZ
       ENDIF
    enddo
    
    close(90)

! read point charge gradients

    IF(MMATOMS.GT.0) THEN
       OPEN(UNIT=90,FILE=TURBOPWD(1:LTURBOPWD)//'/pc_gradient',STATUS='UNKNOWN')
       read(90,'(A)') line
       
       DO I=1,MMATOMS
           read(90,'(A)') line
           read(LINE, '(3D22.14)') DUMX,DUMY,DUMZ
           IF(PRNLEV.ge.4) THEN
              write(OUTU,'(A)') LINE
              write(OUTU,'(3D22.14)') DUMX,DUMY,DUMZ
           ENDIF     
           IPT=MAPMM(I)

           DUMX=DUMX*TOKCAL/BOHRR
           DUMY=DUMY*TOKCAL/BOHRR
           DUMZ=DUMZ*TOKCAL/BOHRR

           DX(IPT) = DX(IPT) + DUMX
           DY(IPT) = DY(IPT) + DUMY
           DZ(IPT) = DZ(IPT) + DUMZ
           IF(PRNLEV.ge.4) THEN
              write(OUTU,'(A,X,I4,X,I4,X,3D22.14)') 'G MM ',I,IPT,DUMX,DUMY,DUMZ
              write(OUTU,'(A,X,I4,X,I4,X,3D22.14)') 'G MM FORCE',I,IPT,DX(IPT),DY(IPT),DZ(IPT)
           ENDIF
      enddo
   
      close(90)
    ENDIF
    RETURN
  END SUBROUTINE QTURBO

#endif /* (qturbo)*/

#if KEY_GAMESSUK==1 /*gamessuk*/
  SUBROUTINE ENVIGRP(GROUP)
    !-----------------------------------------------------------------------
    !     Define new environment variables for the GAMESS files
    !     so they are independent from each other
    !     GAMESS-UK version
    !     
    use chm_kinds
    use dimens_fcm
    use gamess_fcm
    use cstuff, only: fputenv

    implicit none
    
    INTEGER GROUP
    !     
    CALL ENVIAPP('gamess.in',GROUP)
    CALL ENVIAPP('gamess.out',GROUP)
    CALL ENVIAPP('ed3',GROUP)
    !
    ! Only want separate integral and ed7 files
    ! when jobs run at the same time

    if(npgrp.gt.1)then
       CALL ENVIAPP('ed7',GROUP)
       CALL ENVIAPP('ed2',GROUP)
    endif
    !     
    RETURN
  END SUBROUTINE ENVIGRP
  !     
  SUBROUTINE ENVIAPP(ENV,GROUP)
    !-----------------------------------------------------------------------
    !     Define new environment variables for the GAMESS files
    !     so they are independent from each other
    !     
    use chm_kinds
    use dimens_fcm
    use replica_mod
    use string
#if KEY_UNIX==1
    use cstuff, only: setenv
    use, intrinsic :: iso_c_binding, only: C_NULL_CHAR
#endif /* UNIX */

    CHARACTER(len=*) ENV
    INTEGER GROUP
    !     
    CHARACTER(len=255) FILENAME,SGROUP,TMPENV,valfil
    INTEGER L,K,M,N,II,nf
    !     
    M=LEN(ENV)
    TMPENV(1:M+1)=ENV(1:M)//CHAR(0)
    CALL GET_ENVIRONMENT_VARIABLE(TMPENV, FILENAME, L)
    IF(L.EQ.0)RETURN
    ! Remove any existing replica number
    TMPENV = FILENAME
    DO ii = 1,3
       IF(FILENAME(l-ii:l-ii).eq.'_')THEN
          TMPENV= FILENAME(1:l-ii-1)
          l = l-ii-1
       ENDIF
    ENDDO
    FILENAME=TMPENV
    WRITE(SGROUP,'(I5)')GROUP
    K=255
    CALL TRIMA(SGROUP,K)
#if KEY_REPLICA==1
#if KEY_RPATH==1
    if (qrep) then
       N=L+K+M+3
       nf=l+k+2
       valfil(1:nf)='//FILENAME(1:L)//'_'//SGROUP(1:K)//CHAR(0)
       TMPENV(1:N)= &
            ENV(1:M)//'='//FILENAME(1:L)//'_'//SGROUP(1:K)//CHAR(0)
    else
#endif 
#endif 
       n = l + m + 2
       nf=l+k+2
       valfil=FILENAME(1:L)//'_'//SGROUP(1:K)//CHAR(0)
       tmpenv(1:n) = env(1:m)//'='//filename(1:l)//CHAR(0)
#if KEY_REPLICA==1
#if KEY_RPATH==1
    endif
#endif 
#endif 

    !not wroking anymore ?? :CALL FPUTENV(TMPENV,N)
    ii = setenv(env(1:m),valfil(1:nf),1)
    
    if (ii .ne. 0) then
       call wrndie(0, '<ENVIAPP>', &
            'failed to change environment variable')
    end if
    
    !     
    RETURN
  END SUBROUTINE ENVIAPP
  !     
#endif /* (gamessuk)*/
#if KEY_GAMESS==1 || KEY_QCHEM==1 || KEY_G09==1 /*gamess*/
  SUBROUTINE ENVIGRP(GROUP)
    !-----------------------------------------------------------------------
    !     Define new environment variables for the GAMESS files
    !     so they are independent from each other
    !     
    use chm_kinds
    INTEGER GROUP
    !     
    !     Two ways of doing it:
    !     1. put everything into separate directories (we choose this for Q-Chem)
    !     2. append group number to the name (we choose this for Gamess)
    !     
    !     This is the complete list of files used in GAMESS
    !     as of January 2003 version
    !
    !     You only need input and output file for Q-Chem as of Jun, 2004.
    !     

#if KEY_QCHEM==1 || KEY_G09==1
    CALL ENVIAPP('QCHEMINP',GROUP)
    CALL ENVIAPP('QCHEMOUT',GROUP)
#endif 

#if KEY_GAMESS==1
    CALL ENVIAPP('INPUT',GROUP)
    CALL ENVIAPP('OUTPUT',GROUP)
    CALL ENVIAPP('PUNCH',GROUP)
    CALL ENVIAPP('DICTNRY',GROUP)
    CALL ENVIAPP('WORK15',GROUP)
    CALL ENVIAPP('DASORT',GROUP)
    !     
    !     Above is enough for HF
    !     
    CALL ENVIAPP('IRCDATA',GROUP)
    CALL ENVIAPP('AOINTS',GROUP)
    CALL ENVIAPP('MOINTS',GROUP)
    CALL ENVIAPP('DRTFILE',GROUP)
    CALL ENVIAPP('CIVECTR',GROUP)
    CALL ENVIAPP('CASINTS',GROUP)
    CALL ENVIAPP('CIINTS',GROUP)
    CALL ENVIAPP('WORK16',GROUP)
    CALL ENVIAPP('CSFSAVE',GROUP)
    CALL ENVIAPP('FOCKDER',GROUP)
    CALL ENVIAPP('OVLPDER',GROUP)
    CALL ENVIAPP('DFTINTS',GROUP)
    CALL ENVIAPP('DFTGRID',GROUP)
    CALL ENVIAPP('JKFILE',GROUP)
    CALL ENVIAPP('ORDINT',GROUP)
    CALL ENVIAPP('EFPIND',GROUP)
    CALL ENVIAPP('PCMDATA',GROUP)
    CALL ENVIAPP('PCMINTS',GROUP)
    CALL ENVIAPP('MLTPL',GROUP)
    CALL ENVIAPP('MLTPLT',GROUP)
    CALL ENVIAPP('DAFL30',GROUP)
    CALL ENVIAPP('SOINTX',GROUP)
    CALL ENVIAPP('SOINTY',GROUP)
    CALL ENVIAPP('SOINTZ',GROUP)
    CALL ENVIAPP('SORESC',GROUP)
    CALL ENVIAPP('SIMEN',GROUP)
    CALL ENVIAPP('SIMCOR',GROUP)
    CALL ENVIAPP('GCILIST',GROUP)
    CALL ENVIAPP('CIMOHSS',GROUP)
    CALL ENVIAPP('SOCCDAT',GROUP)
    CALL ENVIAPP('AABB41',GROUP)
    CALL ENVIAPP('BBAA42',GROUP)
    CALL ENVIAPP('BBBB43',GROUP)
    CALL ENVIAPP('MCQD50',GROUP)
    CALL ENVIAPP('MCQD51',GROUP)
    CALL ENVIAPP('MCQD52',GROUP)
    CALL ENVIAPP('MCQD53',GROUP)
    CALL ENVIAPP('MCQD54',GROUP)
    CALL ENVIAPP('MCQD55',GROUP)
    CALL ENVIAPP('MCQD56',GROUP)
    CALL ENVIAPP('MCQD57',GROUP)
    CALL ENVIAPP('MCQD58',GROUP)
    CALL ENVIAPP('MCQD59',GROUP)
    CALL ENVIAPP('MCQD60',GROUP)
    CALL ENVIAPP('MCQD61',GROUP)
    CALL ENVIAPP('MCQD62',GROUP)
    CALL ENVIAPP('MCQD63',GROUP)
    CALL ENVIAPP('MCQD64',GROUP)
    CALL ENVIAPP('GVVPT',GROUP)
    CALL ENVIAPP('CCREST',GROUP)
    CALL ENVIAPP('CCDIIS',GROUP)
    CALL ENVIAPP('CCINTS',GROUP)
    CALL ENVIAPP('CCT1AMP',GROUP)
    CALL ENVIAPP('CCT2AMP',GROUP)
    CALL ENVIAPP('CCT3AMP',GROUP)
    CALL ENVIAPP('CCVM',GROUP)
    CALL ENVIAPP('CCVE',GROUP)
#endif 

    RETURN
  END SUBROUTINE ENVIGRP
  !     
  SUBROUTINE ENVIAPP(ENV,GROUP)
    !-----------------------------------------------------------------------
    !     Define new environment variables for the GAMESS files
    !     so they are independent from each other
    !     FIXME: (TODO) Check if it can be merged with the GAMESS-UK version
    !     
    use chm_kinds
    use dimens_fcm
    use replica_mod
    use string
    use cstuff, only: setenv

    implicit none

    CHARACTER(len=*) ENV
    INTEGER GROUP
    !     
    CHARACTER(len=255) FILENAME,SGROUP,TMPENV,TMPA,tmpb,PWDQC
    INTEGER L,K,M,N,O,P,I,ii,jj
    !     
    M=LEN(ENV)
    TMPENV(1:M+1)=ENV(1:M)//CHAR(0)
    CALL GET_ENVIRONMENT_VARIABLE(TMPENV, FILENAME, L)
    IF(L.EQ.0)RETURN
    WRITE(SGROUP,'(I5)')GROUP
    K=255
    CALL TRIMA(SGROUP,K)
#if KEY_REPLICA==1
#if KEY_RPATH==1
    IF(QQCRP) THEN
       CALL GET_ENVIRONMENT_VARIABLE("QCHEMPWD", PWDQC, I)
       IF(I.EQ.0)  &
            CALL WRNDIE(-5,'<QCHEM>','No working directory set.')

       !     Need this to set Q-Chem's input files with rpath
       N=M+1+I+6+K+1+L+1+K+1
       TMPENV(1:N)= &
            ENV(1:M)//'='//PWDQC(1:I)//'/rpath'//SGROUP(1:K)//'/' &
            //FILENAME(1:L)//'_'//SGROUP(1:K)//CHAR(0)

       CALL SYSTEM('mkdir -p '//PWDQC(1:I)//'/rpath'//SGROUP(1:K))
    ELSE
       N=L+K+M+3
       ii=m+1
       jj=l+k+2
       tmpa(1:ii)=env(1:m)//char(0)
       tmpb(1:jj)=filename(1:l)//'_'//sgroup(1:k)//char(0)
       TMPENV(1:N)= &
            ENV(1:M)//'='//FILENAME(1:L)//'_'//SGROUP(1:K)//CHAR(0)
    ENDIF
#endif 
#endif 
    !
    !NOT working anymore?    CALL FPUTENV(TMPENV,N)
    i=setenv(tmpa(1:ii),tmpb(1:jj),1)
    !
    RETURN
  END SUBROUTINE ENVIAPP

#if KEY_GAMESS==1
  SUBROUTINE CH2GMS(BFIRST)
    !-----------------------------------------------------------------------
    !     Define CHARMM atoms as point charges and copy to COMMON/CHMGMS/
    !
    !     (Not used in GAMESS-UK case)
    !     
    !     To simplify changes in GAMESS we build contiguous arrays:
    !     (XCHM,YCHM,ZCHM,QCHM) for .not.QM atoms.
    !     [NOTE: Similar code is present also in CGREP for
    !     nuclear repulsion derivatives]
    !     
    !     On QM atoms we don't care for cutoff, we just take all of the
    !     atoms in the system. This is not inconsistent with MM since QMs
    !     are special anyway. Also calculation time is linear with number
    !     of MM atoms!
    !     
    !     [NOTE: It should be straightforward to implement usage of
    !     variety of cutoff methods implemented in the CHARMM:
    !     Just use nonbond array here and disable CGREP routine;
    !     but you need to specify charges on QM atoms as their
    !     atomic number in the RTF file!!!
    !     
    !     
    use chm_kinds
    use dimens_fcm
    use coord
    use psf
    use storage
    use stream
    use gamess_fcm
    use consta
    use number
    use block_fcm
    use scalar_module
    use replica_mod
    !     
    LOGICAL BFIRST
    !     
    INTEGER J,I,N,K,NBLUR,IBL,IBLQM
    real(chm_real) TPIPOH,SIGM1,SIGM2,SIGM3
    real(chm_real) RBR,FAC
    real(chm_real) tmpblur(MAXA)


    ! this is currently broken (FIXME) - maybe not needed ???
    ! multi-layered qm/mm and then return after that.
#if KEY_GAMESS==1
#if KEY_SQUANTM==1 /*squantm*/
    If(QMLAY_high) then
       call CH2GMS_mlayer(natom,nchmat,nbluch,ibluch, &
            xchm,ychm,zchm,qchm, &
            tmpblur,ebluch,cgblch,sgblch,cbluch)
       return
    End if
#endif /* (squantm)*/
#endif 

    !     
    !     
    !     All atoms which are not quantum contribute to electrostatic
    !     interaction in the QM part. See MJF reference:
    !     J. Comp. Chem., Vol. 11, No. 6, 700-733 (1990)
    !     
    N=0
    NBLUR=0
    !     tpipoh = 2*sqrt(PI**3)
    TPIPOH=11.136655993663416_chm_real
    RBR=ONE/BOHRR
    IBLUCH(1:NATOM) = 0
    !
#if KEY_BLOCK==1 /*block*/
    IF(QBLOCK)THEN
       IBLQM=0
       DO I=1,NATOM
          ibl=iblckp(i)
          IF((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2))THEN
             IF(IBLQM.EQ.0)IBLQM=IBL
             IF(IBLQM.NE.IBL)THEN
                CALL WRNDIE(-5,'<GUKINI>', &
                     'QM region must be within the same block.')
             ENDIF
          ENDIF
       ENDDO
    ENDIF
#endif /*  (block)*/

    DO I = 1,NATOM
       IF(QBLUCH) THEN
          IF((IGMSEL(I).EQ.0).OR.(IGMSEL(I).EQ.5))THEN
             N = N + 1
             XCHM(N)=X(I)*RBR
             YCHM(N)=Y(I)*RBR
             ZCHM(N)=Z(I)*RBR
             QCHM(N)=CG(I)
             !     
             !     Fill in blurred charges arrays
             !     
             if(recallint.ne.-1)then
                do k=1,natom
                   tmpblur(k)=PTRSTO(recallint)%a(k)
                enddo
             else
                do k=1,natom
                   tmpblur(k)=WMAIN(k)
                enddo
             endif


             IF (ABS(tmpblur(I)).GE.RSMALL) THEN
                IF(tmpblur(I).GT.NINE99) THEN

                   QCHM(N)=ZERO
                ELSE
                   NBLUR=NBLUR+1
                   IBLUCH(NBLUR)=N
                   SIGM1=BOHRR/tmpblur(I)
                   SIGM2=SIGM1*SIGM1
                   SIGM3=SIGM2*SIGM1
                   EBLUCH(NBLUR)=SIGM2
                   IF(BFIRST) THEN
                      CGBLCH(NBLUR)=CG(I)
                   ENDIF
                   SGBLCH(NBLUR)=tmpblur(I)*RBR
                   !CC   CBLUCH(NBLUR)=CGBLCH(NBLUR)*SIGM3/TPIPOH
                   CBLUCH(NBLUR)=CGBLCH(NBLUR)* &
                        SIGM3*0.5079490874739
                   !     
                   !     Put to zero both MM charges
                   !     a) the one which goes to GAMESS (QCHM)
                   !     b) the one which goes to CHARMM (CG)
                   !     
                   CG(I)=ZERO
                   QCHM(N)=ZERO
                   !     
                   !     For now put back MM charges
                   !     They are dealt in CHARMM, and we only have
                   !     Blurred charges interacting with QM atoms.
                   !     No blur-blur! This is done in standard CHARMM way!
                   !     
                   CG(I)=CGBLCH(NBLUR)
                ENDIF
             ENDIF
          ENDIF
       ELSE
          IF (IGMSEL(I) .EQ. 0) THEN
             !
             ! Ewald: put here the image stuff
             !
             N = N + 1
             XCHM(N)=X(I)/BOHRR
             YCHM(N)=Y(I)/BOHRR
             ZCHM(N)=Z(I)/BOHRR
             !
             !     BLOCK implemented for GAMESS only (also NO blur yet)
             !
             FAC=ONE
#if KEY_BLOCK==1
             !         preform this only when no replicas present !!!
#if KEY_REPLICA==1
             IF(.NOT.QREP)THEN      
#endif
                IF(QBLOCK)THEN
                   j=iblckp(i)
!!!  POSSIBLE BUG:  IBL here is maybe wrong!!!
                   !!                   FAC=GTRR8(oldmem(BLCOEP),IBL+(J*(J-1))/2)
                   fac=blcoep((j*(j+1))/2)
                ENDIF
#if KEY_REPLICA==1
             ENDIF                 
#endif
#endif 
             QCHM(N)=CG(I)*FAC

          ENDIF
       ENDIF
    ENDDO
    !
    NCHMAT=N
    NBLUCH=NBLUR
    !
    RETURN
  END SUBROUTINE CH2GMS
#endif
  !
#endif /* (gamess)*/
#if KEY_QTURBO==0 && KEY_G09==0 /*noqchem*/
#if KEY_GAMESS==1 || KEY_GAMESSUK==1
  SUBROUTINE CHMDAT(AATOM,AZNUC,CORD,NAT &
#if KEY_GAMESSUK==1
       ,nel,expo,wght,maxatg                & 
#endif
       )
    !-----------------------------------------------------------------------
    !     Define the set of quantum mechanical atoms and set up the data
    !     structures.
    !     
    !     NB - for the GAMESS-UK interface, this deals with the 
    !     classical atoms as well, since all interactions are 
    !     handled within GAMESS-UK
    !     
    use chm_kinds
    use dimens_fcm
    use linkatom
    use number
    use coord
    use consta
    use psf
    use param
    use rtf,only:atct
    use stream
    use gamess_fcm
    use parallel
#if KEY_GAMESSUK==1
    use scalar_module       
#endif
#if KEY_SQUANTM==1
    use squantm      
#endif
    !     
    !     COMMON from GAMESS
    !     
    !     The following is defined in GAMESS after call to CHMDAT
    !     real(chm_real) HMLTN
    !     LOGICAL MPCWFN
    !     COMMON /MPCLNK/ HMLTN,MPCWFN
    !CCC  test code...
    logical mopac
    common /mpctst/mopac
    !CCC  end test code...
    !     
#if KEY_GAMESS==1 /*gamess*/
    CHARACTER(len=10) AATOM(MAXGMS)
    real(chm_real) AZNUC(MAXGMS), CORD(MAXGMS,3)
    INTEGER(gms_int) NAT
#endif /* (gamess)*/
#if KEY_GAMESSUK==1 /*gamessuk*/
    CHARACTER(len=10) AATOM(*)
    real(chm_real) AZNUC(*), CORD(3,*)
    real(chm_real) expo(*),wght(*),SIGM1,SIGM2,totnuc
    INTEGER NEL, IATOM, MAXATG, K
    real(chm_real) AZN, EXP1, WGH, TESTNE
    logical qm,obq
    INTEGER NAT
#endif /* (gamessuk)*/
    INTEGER I,NSLCT,NATMM_local,NATQM_local,NATLNK
    CHARACTER(len=6) ELE
    logical lqinigm
    integer iclean,jclean
    character(len=1), parameter, dimension(1:10) :: &
         num_clean = (/'0','1','2','3','4','5','6','7','8','9'/)
    real(chm_real) tmpblur(maxa)

    !  Counter for true QM atoms    
    NATQM_local=0

    ! multi-layered qm/mm and then return after that.
#if KEY_GAMESS==1 || KEY_GAMESSUK==1
#if KEY_SQUANTM==1 /*squantm*/
    If(QMLAY_high) then
       call CHMDAT_mlayer(AATOM,AZNUC,CORD,NAT &
#if KEY_GAMESSUK==1
            ,nel,expo,wght,maxatg    & 
#endif
            ,PXIM,PYIM,PZIM &
            ,wmain &
            )
       return
    End if
#endif /* (squantm)*/
#endif 

    !     
#if KEY_GAMESSUK==1 /*gamessuk*/

    !
    ! for consistency with GAMESS(US) this is 
    ! a counter for BQ-class centres and Blurred centres
    ! (not useful to the QM code)
    !
    NCHMAT=0
    !
    totnuc = zero

    !
    ! to obtain position in the charmm list of
    ! atom igms in the GAMESS-UK list use
    !
    ! ichm = GMSMAP(igms) 
    !
    ! 
    DO I = 1, NATOM
       GMSMAP(I) = -1
    ENDDO
    !
    !  First loop Quantum and blurred atoms
    !
    IATOM = 0

    DO I = 1,NATOM

       IF (IGMSEL(I) .EQ. 1 .OR. IGMSEL(I) .EQ. 2) THEN

          iatom = iatom + 1

          if (IATOM .GT. MAXATG)  &
               CALL WRNDIE(0,'<CHMDAT>', &
               'Too many atoms, redimension GAMESS-UK')

          NATQM_local = NATQM_local + 1
          gmsmap(iatom) = i

          CORD(1,IATOM)=X(I)/BOHRR
          CORD(2,IATOM)=Y(I)/BOHRR
          CORD(3,IATOM)=Z(I)/BOHRR

          expo(IATOM) = -1.0d0
          wght(IATOM) =  0.0d0
          AZNUC(IATOM)=  0.0d0


          AATOM(IATOM) = ATYPE(I)
          IF (ATYPE(I)(1:3) == 'QQH') AATOM(IATOM) = 'H         '
          IF (ATYPE(I)(1:3) == 'qqh') AATOM(IATOM) = 'h         '
          !
          ! ensure mass vector is only referenced for initialisation
          ! step (helps use lone pair assignments)
          !
          IF(QINIGM)A2MASS(I) = AMASS(I)
          LQINIGM=QINIGM
          CALL FINDEL(ATCT(IAC(I)),A2MASS(I),I,ELE,AZNUC(IATOM), &
               LQINIGM)
          aatom(IATOM) = ele
          UZNUC(IATOM)=AZNUC(IATOM)
          IF(ELE(1:3).EQ.'QQH')AATOM(IATOM)='H         '
          IF(ELE(1:3).EQ.'qqh')AATOM(IATOM)='h         '
          !     
          ! Store sum of nuclear charges
          !     
          totnuc = totnuc + aznuc(IATOM)

       elseif (IGMSEL(I) .eq. 0) then

          IF(QBLUCH)then

             ! it is not set in gamess.f90. (namkh)
             !              if(recallint.ne.-1)then
             !                 do k=1,natom
             !                    tmpblur(k)=PTRSTO(k)
             !                 enddo
             !              else
             do k=1,natom
                tmpblur(k)=WMAIN(k)
             enddo
             !              endif

             IF (ABS(tmpblur(I)) .GE. RSMALL) THEN
                IF(tmpblur(I).GT.NINE99) THEN
                   !
                   ! explicitly excluded atom
                   !
                ELSE
                   !
                   ! add a blurred centre
                   !
                   IATOM = IATOM + 1
                   if (IATOM .GT. MAXATG)  &
                        CALL WRNDIE(0,'<CHMDAT>', &
                        'Too many atoms, redimension GAMESS-UK')

                   gmsmap(iatom) = i
                   NCHMAT = NCHMAT + 1
                   AZNUC(IATOM)=ZERO
                   AATOM(IATOM)='BQ        '
                   CORD(1,IATOM)=X(I)/BOHRR
                   CORD(2,IATOM)=Y(I)/BOHRR
                   CORD(3,IATOM)=Z(I)/BOHRR
                   NBLUCH=NBLUCH+1
                   SIGM1=BOHRR/tmpblur(I)
                   SIGM2=SIGM1*SIGM1
                   expo(IATOM) = SIGM2
                   wght(IATOM) = CG(I)
                ENDIF
             ENDIF
          ENDIF
       ENDIF
    ENDDO
    !
    !  Second loop to assign BQ atoms
    !
    DO I = 1,NATOM

       ! The earlier codes included -1 (now 5) here
       ! now we leave them out

       IF (IGMSEL(I) .EQ. 0) THEN
          IF(QBLUCH)then

             !      skip blurred centres (already included above)
             !      this statement also skips explicitly excluded atoms,
             !      these have WMAIN(I).GT.NINE99

             OBQ = .NOT. (ABS(tmpblur(I)) .GE. RSMALL)
          ELSE
             OBQ = .TRUE.
          ENDIF

          if (OBQ) then
             IATOM = IATOM + 1
             if (IATOM .GT. MAXATG)  &
                  CALL WRNDIE(0,'<CHMDAT>', &
                  'Too many atoms, redimension GAMESS-UK')
             GMSMAP(IATOM) = I
             NCHMAT=NCHMAT + 1
             AATOM(IATOM)='BQ        '
             CORD(1,IATOM)=X(I)/BOHRR
             CORD(2,IATOM)=Y(I)/BOHRR
             CORD(3,IATOM)=Z(I)/BOHRR
             EXPO(IATOM) = -1.0d0
             WGHT(IATOM) =  0.0d0
             AZNUC(IATOM)=  CG(I)
          ENDIF

       ELSEIF (IGMSEL(I) .EQ. 5 .OR. IGMSEL(I) .LT. 0) THEN
          ! excluded centre, or other replica
       ENDIF
    ENDDO
    !
    NAT    = IATOM
    NEL    = NINT(totnuc)
    TESTNE = NEL
    TESTNE = TESTNE - TOTNUC
    if ( dabs(testne) .gt. PT0001) then
       write (6,9568) totnuc,1.0d0*nel,testne
9568   format(1x,'non-integral charge found',1x,3e15.8)
       CALL WRNDIE(0,'<CHMDAT>','non-integral QM charge')
    endif
    !
#endif /*  (gamessuk)*/
    !     
#if KEY_GAMESS==1 /*gamess*/
    !
    DO I = 1,NATOM
       IF ((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2).OR.MOPAC) THEN
          NATQM_local = NATQM_local + 1
          CORD(NATQM_local,1)=X(I)
          CORD(NATQM_local,2)=Y(I)
          CORD(NATQM_local,3)=Z(I)
          CUNIQ(NATQM_local,1)=CORD(NATQM_local,1)
          CUNIQ(NATQM_local,2)=CORD(NATQM_local,2)
          CUNIQ(NATQM_local,3)=CORD(NATQM_local,3)
          !     
          !     Added this condition for lone pairs.
          !     This prevents to change QM region within one run
          !     
          IF (QINIGM) THEN
             AATOM(NATQM_local) = ATYPE(I)
             ! MCP basis set does not deal well with atom names with numbers
             ! we need to clean up aatom() array:
             cleanup: do iclean=1,len(aatom(natqm_local))
                do jclean=1,10
                   if (aatom(natqm_local)(iclean:iclean)==num_clean(jclean)) &
                        aatom(natqm_local)(iclean:iclean)=' '
                enddo
             enddo cleanup
             IF (ATYPE(I)(1:3) == 'QQH') AATOM(NATQM_local) = 'H         '
             !
             LQINIGM=QINIGM
             CALL FINDEL(ATCT(IAC(I)),AMASS(I),I,ELE, &
                  AZNUC(NATQM_local), &
                  LQINIGM)
             !
             UATOM(NATQM_local)=AATOM(NATQM_local)
             UZNUC(NATQM_local)=AZNUC(NATQM_local)
          ENDIF
       ENDIF
    ENDDO
    !     
    NAT=NATQM_local
    NATREL=NAT
    UATOM(NAT+1)='$END      '
    !     
#endif /* (gamess)*/
    !
    IF (NATQM_local .LE. 0) CALL WRNDIE(0,'<CHMDAT>', &
         'No quantum mechanical atoms selected.')
    NATMM_local = NATOM - NATQM_local
    NGAMES = NATQM_local
    !     
    NSLCT = 0
    DO I = 1,NATOM
       IF (IGMSEL(I).EQ.2) NSLCT = NSLCT + 1
    ENDDO
    NATLNK = NSLCT
    !     
    !     Write out some information and options requested.
    !     
    IF(QBLUCH.AND.PRNLEV.GE.2.AND.QINIGM) WRITE (OUTU,'(/,8X,A,I10)') &
         ' The number of blurred MM charges         = ',NBLUCH
    IF ((PRNLEV.GE.2).AND.QINIGM) WRITE (OUTU,'(/,1X,A,/)') &
         ' GAMDFN> Some atoms will be treated quantum mechanically.'
    IF ((PRNLEV.GE.2).AND.QINIGM) WRITE (OUTU,'(4(8X,A,I10,/),/)') &
         ' The number of quantum   mechanical atoms = ',NATQM_local, &
         ' Of which the number of QM/MM link atoms  = ',NATLNK, &
         ' The number of molecular mechanical atoms = ',NATMM_local, &
         ' The number of MM atoms excluded from QM  = ', &
         NATMM_local-NCHMAT
    !     
    RETURN
  END SUBROUTINE CHMDAT
#endif
#endif /* (noqchem)*/

#if KEY_NWCHEM==1
!!!
  subroutine nwchem_rtdb_clr
    use gamess_fcm,only:rtdb_handle
    logical rtdb_delete,ldum
!!!
    ! this routine makes NWChem perform the necessary calculations
    ! by forgetting some flags.
    ! Each method seems to have a different thingy here
    ! ask for a better way to do it ??
    ! currently tested for HF, DFT, MP2 
!!!
    ldum = rtdb_delete(rtdb_handle,'dft:converged')
    !write(*,*)'after delete dft:converged ...',ldum
    ldum = rtdb_delete(rtdb_handle,'scf:converged')
    !write(*,*)'after delete scf:converged ...',ldum

    return
  end subroutine nwchem_rtdb_clr
!!!
  SUBROUTINE nwchem_qm_atoms(x,y,z)
    !-----------------------------------------------------------------------
    !     Define the set of quantum mechanical atoms and set up the data
    !     structures, using RTDB
    !     
    use chm_kinds
    use dimens_fcm
    use linkatom
    use number
    use consta
    use psf
    use param
    use rtf,only:atct
    use stream
    use gamess_fcm
    use parallel
    use memory
    !
    real(chm_real) x(*),y(*),z(*)
    character(len=256) rtdb_filename
    CHARACTER(len=16),allocatable,dimension(:) :: AATOM
    double precision,allocatable,dimension(:) :: AZNUC
    double precision,allocatable,dimension(:,:) :: CORD
    INTEGER*8 NAT_QM
    !
    logical ldum
    integer*8 ii_ncent,ncent
    character(len=16),allocatable,dimension(:):: atom_tag
    double precision,allocatable,dimension(:,:):: atom_c
    double precision,allocatable,dimension(:):: atom_q
    logical geom_create,geom_check_handle
    logical geom_ncent,geom_cart_get,geom_cart_set
    logical rtdb_open,rtdb_parallel,rtdb_print,geom_rtdb_store

    INTEGER I,NSLCT,NATMM_local,NATQM_local,NATLNK
    CHARACTER(len=6) ELE
    logical lqinigm

    !  Counter for true QM atoms    
    NATQM_local=0

    ! count the QM atoms
    DO I = 1,NATOM
       IF ((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2)) &
            NATQM_local = NATQM_local + 1
    enddo
    NAT_QM=NATQM_local
    !write(*,*)'qinigm, NAT_QM: ',qinigm, nat_qm
!    call chmalloc('gukini.src','nwchem_qm_atoms','aznuc',NAT,crl=aznuc)
!    call chmalloc('gukini.src','nwchem_qm_atoms','cord',3,NAT,crl=cord)
    if (qinigm) then
       !write(*,*)'In qinigm'
       allocate(nwchem_znuc(nat_qm))
    endif

    allocate(aatom(nat_qm),cord(3,nat_qm),aznuc(nat_qm))

    NATQM_local=0
    DO I = 1,NATOM
       IF ((IGMSEL(I)==1).OR.(IGMSEL(I)==2)) THEN
          NATQM_local = NATQM_local + 1
          !write(*,*)'natqm_local=',natqm_local
          CORD(1,NATQM_local)=X(I)/bohrr
          CORD(2,NATQM_local)=Y(I)/bohrr
          CORD(3,NATQM_local)=Z(I)/bohrr
          !     
          !??     Added this condition for lone pairs.
          !??     This prevents to change QM region within one run
          !     
!!!          IF (QINIGM) THEN
!!             IF (ATYPE(I)(1:3) == 'QQH') AATOM(NATQM_local) = 'H         '
             !
             LQINIGM=QINIGM
             CALL FINDEL(ATCT(IAC(I)),AMASS(I),I,ELE, &
                  AZNUC(NATQM_local), &
                  LQINIGM)
             AATOM(NATQM_local) = ELE
             IF (ATYPE(I)(1:3) == 'QQH') AATOM(NATQM_local) = 'H         '
             nwchem_znuc(natqm_local)=AZNUC(NATQM_local)
             !
!!!          ENDIF
       ENDIF
    ENDDO
    !     
    IF (NATQM_local .LE. 0) CALL WRNDIE(0,'<CHMDAT>', &
         'No quantum mechanical atoms selected.')
    NATMM_local = NATOM - NATQM_local
    NGAMES = NATQM_local
    nwchem_nat_qm = NATQM_local
    !     
    NSLCT = 0
    DO I = 1,NATOM
       IF (IGMSEL(I).EQ.2) NSLCT = NSLCT + 1
    ENDDO
    NATLNK = NSLCT
    !     
    !     Write out some information and options requested.
    !     
!!! Not everyhting is defined yet... LATER!!
!!! for now commented out!!!
!    IF(QBLUCH.AND.PRNLEV.GE.2.AND.QINIGM) WRITE (OUTU,'(/,8X,A,I10)') &
!         ' The number of blurred MM charges         = ',NBLUCH
    IF ((PRNLEV.GE.2).AND.QINIGM) WRITE (OUTU,'(/,1X,A,/)') &
         ' GAMDFN> Some atoms will be treated quantum mechanically.'
    IF ((PRNLEV.GE.2).AND.QINIGM) WRITE (OUTU,'(4(8X,A,I10,/),/)') &
         ' The number of quantum   mechanical atoms = ',NATQM_local, &
         ' Of which the number of QM/MM link atoms  = ',NATLNK
!         ' Of which the number of QM/MM link atoms  = ',NATLNK, &
!         ' The number of molecular mechanical atoms = ',NATMM_local, &
!         ' The number of MM atoms excluded from QM  = ', &
!         NATMM_local-NCHMAT
    !
    ! fill the RTDB
    ! define handles only once!
    !write(*,*)'nwchem_qm_atoms>qinigm=',qinigm
    if(qinigm) then
       ldum=geom_create(geom_handle,'geometry')
       !write(*,*)'after geom_create>',ldum,' geom_handle=',geom_handle
       ldum=geom_check_handle(geom_handle,'seems to be not OK?')
       !write(*,*)'after geom_check_handle>',ldum
       ! the following line may not be needed ??
       call rtdb_init()
       !          ldum=rtdb_parallel(.true.)
       !          write(*,*)'after rtdb_parallel>',ldum
       call get_environment_variable('NWCHEM_RTDB',rtdb_filename)
       !write(*,*)'rtdb_filename=',trim(rtdb_filename)
       ldum=rtdb_open(trim(rtdb_filename),'unknown',rtdb_handle)
       !write(*,*)'after rtdb_open>',ldum,' rtdb=',rtdb_handle
       !ldum=rtdb_print(rtdb_handle,.true.)
       !write(*,*)'after rtdb_print>output probably in _nw.out',ldum
       !ldum=geom_ncent(geom_handle,ncent)
       !write(*,*)'after geom_cart_get>',ldum,' ncent=',ncent
       !write(*,*)'Number of qm atoms...',nat_qm
!!!       allocate(atom_c(3,nat_qm),atom_q(nat_qm))
!!!       do ii_ncent=1,nat_qm
!!!          write(*,*)ii_ncent,aatom(ii_ncent),cord(1:3,ii_ncent),&
!!!               aznuc(ii_ncent)
!!!       enddo
       qinigm=.false.
    endif

    ldum=geom_cart_set(geom_handle,nat_qm,aatom,cord,aznuc)
    !write(*,*)'after geom_cart_set>',ldum
!    ldum=geom_cart_get(geom_handle,ncent,atom_tag,atom_c,atom_q)
!    write(*,*)'after geom_cart_get>',ldum
    ldum=geom_rtdb_store(rtdb_handle,geom_handle,'geometry')
    !write(*,*)'after geom_rtdb_store>',ldum
    !ldum=rtdb_print(rtdb_handle,.true.)
    !write(*,*)'after rtdb_print>output seems to be to 0',ldum
!    do ii_ncent=1,ncent
!       write(*,*)ii_ncent,atom_tag(ii_ncent),atom_c(1:3,ii_ncent),&
!            atom_q(ii_ncent)
!    enddo
    ! moved above: call task(rtdb_handle)
    deallocate(aatom,cord,aznuc)
    RETURN
  END SUBROUTINE NWCHEM_QM_ATOMS
!!!
  SUBROUTINE nwchem_mm_atoms(x,y,z)
    !-----------------------------------------------------------------------
    !     Define the set of classical mechanical atoms and set up the data
    !     for hnd_stvint.F integral code in NWChem
    !     
    use chm_kinds
    use dimens_fcm
    use number
    use consta
    use psf
    use stream
    use gamess_fcm,only: igmsel
    use parallel
    !
    real(chm_real) x(*),y(*),z(*)
    ! these 3/4 lines go into hnd_stvint.F program:
    INTEGER*8 NAT_MM,nchm
    double precision dxyzchm,xyzchm,qchm
    common/nwc_chm/xyzchm(3*360720),qchm(360720),nchm
    common/nwc_chmd/dxyzchm(3*360720)
    !
    logical ldum

    INTEGER I,NSLCT,NATMM_local,NATLNK
    logical lqinigm

    ! count the MM atoms
    natmm_local=0
    !write(*,'(a,20i2)')'igmsel:',(igmsel(i),i=1,natom)
    DO I = 1,NATOM
       IF (IGMSEL(I)==0) then
          NATMM_local = NATMM_local + 1
          xyzchm(3*NATMM_local-2)=X(I)/bohrr
          xyzchm(3*NATMM_local-1)=Y(I)/bohrr
          xyzchm(3*NATMM_local)=Z(I)/bohrr
          qchm(NATMM_local)=CG(I)
       endif
    enddo
    NAT_MM=NATMM_local
    nchm=NATMM_local
    !write(*,*)'NAT_MM: ',nat_mm
    !     
    RETURN
  END SUBROUTINE NWCHEM_MM_ATOMS
!!!
  SUBROUTINE nwchem_qmmm_results(energy,dx,dy,dz,x,y,z)
    use chm_kinds
    use dimens_fcm
    use stream
    use consta
    use number
    use psf,only:natom,cg
    use gamess_fcm, only: rtdb_handle,igmsel
    use parallel
!!!
    real(chm_real) energy, dx(*), dy(*), dz(*), x(*), y(*), z(*)
    ! these 2 lines are in the hnd_ij.F program:
    double precision dxyzchm
    common/nwc_chmd/dxyzchm(3*360720)
    !
!!! stuff from NWChem:
    logical rtdb_get,ldum
    ! the following mt_dbl "magic number" is from
    ! nwchem/src/tools/install/include/{macommon.h,mafdecls.fh}
    integer*8 :: mt_dbl=1013
    integer*8 nwc_one,nwc_grad
    double precision,allocatable,dimension(:) :: energy_nwchem, grad_nwchem
    integer iii_grad,local_qm,i,l,n,j
    real(chm_real) blfactor,q1,x1,y1,z1,x2,y2,z2,x12,y12,z12
    real(chm_real) rr12,r12,q2,el,elr,repuls,convert_u
!!!
    nwc_one=1
    nwc_grad=3*nwchem_nat_qm
    ! rtdb calls are maybe universal:
    ! so the whole routine cannot be protected for mynod !!
    allocate(energy_nwchem(nwc_one),grad_nwchem(nwc_grad))
!!! hopefully this always works: so far tested for dft + hf
    ldum=rtdb_get(rtdb_handle,'task:energy',mt_dbl,&
         nwc_one,energy_nwchem)
!    write(*,*)'Energy for this task is:',energy_nwchem(1)
    if(mynod==0) energy = energy + energy_nwchem(1)*tokcal
    ldum=rtdb_get(rtdb_handle,'task:gradient',mt_dbl,&
         nwc_grad,grad_nwchem)
    convert_u=tokcal/bohrr
    local_qm=0
    ! for parallel this needs to be done only on node zero...
    if (mynod==0) then
       do iii_grad=1,natom
          if ((igmsel(iii_grad).eq.1).or.(igmsel(iii_grad).eq.2)) then
             local_qm=local_qm+1
             dx(iii_grad)=dx(iii_grad)+grad_nwchem(3*local_qm-2)*convert_u
             dy(iii_grad)=dy(iii_grad)+grad_nwchem(3*local_qm-1)*convert_u
             dz(iii_grad)=dz(iii_grad)+grad_nwchem(3*local_qm)*convert_u
             IF(PRNLEV.GT.6) THEN
                WRITE(OUTU,334) Iii_grad,local_qm &
                     , grad_nwchem(3*local_qm-2)*convert_u &
                     , grad_nwchem(3*local_qm-1)*convert_u &
                     , grad_nwchem(3*local_qm)*convert_u
334             FORMAT(2I10,3F14.6,3F14.9)
             endif
          endif
       enddo
    endif
    ! end for parallel protection
    !write(*,*)'gradient for this task:',grad_nwchem
!!!    ldum=rtdb_print(rtdb_handle,.true.)
    deallocate(energy_nwchem,grad_nwchem) ! every body does this (as above)!

!!!  add also the MM repulsion here....
!!!  repulsion is also just node 0 business
!!!  look at c42a2q/nwchem/{xx,yy,zz}.f90 for hints....

    !if(mynod>0) return ! could be easily parallelized
    blfactor=one       ! for block (not yet)
    L=0
    repuls = zero
    DO I = 1,NATOM
       IF ((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2)) THEN
          !
          !        QM atoms
          !
          L=L+1
          Q1 = nwchem_znuc(L)
          ! NOT YET: IF(FQQCHG(L).GT.-NINE99) Q1 = FQQCHG(L)
          X1 = X(I)
          Y1 = Y(I)
          Z1 = Z(I)
          N = 0
          DO J = 1, NATOM
             IF(IGMSEL(J)==0) then
                !
                !           MM atoms
                !
                N = N + 1
                X2 = X(J)
                Y2 = Y(J)
                Z2 = Z(J)
                X12 = X1-X2
                Y12 = Y1-Y2
                Z12 = Z1-Z2
                RR12 = X12**2+Y12**2+Z12**2
                R12 = SQRT(RR12)
                Q2 = CG(J)
                EL = Q1*Q2/R12 !*BLFACTOR
                ELR = EL/RR12*convert_u
                REPULS = REPULS + EL
                DX(I) = DX(I) - X12*ELR
                DX(J) = DX(J) + X12*ELR
                DY(I) = DY(I) - Y12*ELR
                DY(J) = DY(J) + Y12*ELR
                DZ(I) = DZ(I) - Z12*ELR
                DZ(J) = DZ(J) + Z12*ELR
                ! very probably a good place to get also the
                ! derivatives from 1e integrals to MM atoms
                ! everything in place already here: index N !
             ENDIF
          ENDDO
       ENDIF
    ENDDO
    energy=energy+repuls*tokcal*bohrr
    !write(*,*)'nwchem: repuls=',repuls,repuls*tokcal*bohrr
    if(mynod == 0) then  ! not tested yet...
       N = 0
       DO I = 1,NATOM
          IF(IGMSEL(I).EQ.0) then
             N = N + 1
             DX(I) = DX(I) - dxyzchm(3*N-2)*convert_u*BLFACTOR
             DY(I) = DY(I) - dxyzchm(3*N-1)*convert_u*BLFACTOR
             DZ(I) = DZ(I) - dxyzchm(3*N)*convert_u*BLFACTOR
             !     
             IF(PRNLEV.GT.6) THEN
                WRITE(OUTU,334) I,N &
                     , -dxyzchm(3*N-2)*convert_u*BLFACTOR &
                     , -dxyzchm(3*N-1)*convert_u*BLFACTOR &
                     , -dxyzchm(3*N)*convert_u*BLFACTOR
             ENDIF
             !     
          ENDIF
       ENDDO
    endif  !mynod==0 ?? (test it!!)
    return
  end SUBROUTINE nwchem_qmmm_results
!!!
#endif

  !     
  SUBROUTINE GUKENE(GTOT,X,Y,Z,DX,DY,DZ,CGX,AMASSX,IACX,NATOM,NDD1, &
       DD1,QSECD,IUPT,JUPT)
    !-----------------------------------------------------------------------
    !     
    !     Get energy and forces from GAMESS,GAMESS-UK
    !     
    use chm_kinds
    use dimens_fcm
    use consta
    use number
    use gamess_fcm
    use stream
    use replica_mod
    use pathm
    use repdstr
    use block_fcm
    use mndo97
    use parallel
    use scalar_module
    ! Add quantum energy to econt array for NEB usage ! jwchu
    use econtmod                ! jwchu 
    use pbeq,only:QSMBP
    !     
    real(chm_real) GTOT,X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
    !real(chm_real) CGX(:),AMASSX(*) ! JZ_UW12
    real(chm_real) CGX(*),AMASSX(*)
    INTEGER NATOM,IACX(*),NDD1
    real(chm_real) GTOTOLD,ECURRENT             ! jwchu
    real(chm_real) DD1(*)
    INTEGER IUPT(*),JUPT(*)
    LOGICAL lexist,QSECD
    !     
    !
    !     
#if KEY_GAMESSUK==1
    INTEGER INIT,ICODE,IVER,N1   
#endif
#if KEY_QCHEM==1 || KEY_GAMESS==1 || KEY_QTURBO==1 || KEY_G09==1
    INTEGER N1                   
#endif
#if KEY_GAMESSUK==1
    LOGICAL STARTUP              
#endif
#if KEY_GAMESS==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1 /*gamess*/
    INTEGER ICHARM
    !     
    real(chm_real) E, EG
    COMMON /FUNCT/ E, EG(3*MAXGMS)
    !     
    INTEGER(gms_int) NAT,ICH,MUL,NUM,NX,NE,NA,NB,IAN
    real(chm_real) ZAN, C
    COMMON /INFOA / NAT,ICH,MUL,NUM,NX,NE,NA,NB, &
         ZAN(MAXGMS),C(3,MAXGMS),IAN(MAXGMS)
    !
    real(chm_real) HMLTN
    LOGICAL MPCWFN
    COMMON /MPCLNK/ HMLTN,MPCWFN
    !     
    real(chm_real) QMMMRP, RBR, EDIESL
    INTEGER IPT,N
    logical mopac
#if KEY_REPLICA==0
    real(chm_real) BLFACTOR    
#endif
#endif /*  (gamess)*/
    !     
#if KEY_BLOCK==1
    INTEGER IBL,IBLQM,KK                    
#endif

    LOGICAL QDONE
    CHARACTER CFN*10,CRN*6
    INTEGER I
#if KEY_PARALLEL==1
    INTEGER MYNODL            
#endif
#if KEY_REPLICA==1 && KEY_RPATH==1
    INTEGER IREPL             
#endif
#if KEY_GAMESSUK==1
    CHARACTER(len=3) TMPSTR        
#endif
#if KEY_GAMESSUK==1
    CHARACTER(len=40) MSG          
#endif
#if KEY_QCHEM==1 || KEY_G09==1
    CHARACTER(len=80) QCTMP        
#endif

#if KEY_NWCHEM==1
    logical ldum
#endif
#if KEY_SQUANTM==1
    LOGICAL LQMEWD
#endif 
    !     
    !     Are there any QM atoms?
    !     
    IF(NGAMES.EQ.0) RETURN
#if KEY_GAMESS==1 || KEY_GAMESSUK==1
#if KEY_SQUANTM==1
    IF(QMLAY_high) RETURN    
#endif
#endif 
    !
    CFN='gukini.src'
    CRN='GUKENE'

    !
#if KEY_SQUANTM==1
    LQMEWD =.false.
#endif 
    !     
    !     Separate to parallel/parallel
    !     
    BLFACTOR=ONE
    !
    GTOT = ZERO
    GTOTOLD=ZERO
    !
    !     Zero the QM charges, in case they are not calculated
    QMMUL(1:NGAMES) = ZERO
    QMLOW(1:NGAMES) = ZERO
    QMKOL(1:NGAMES) = ZERO
    !

#if KEY_REPLICA==1 /*replica*/
#if KEY_RPATH==1
#if KEY_PARALLEL==1 /*parallel*/
    !write(50+mynodg,'(a,5i5)')'gukene>start:mynod,numnod,irepqm,npgrp=', &
    !     mynod,numnod,irepqm,npgrp
    IF(NPGRP .NE. 1)then

       MYNOD=mynod_save
       MYNODP=MYNOD+1
       NUMNOD=numnod_save
       NODDIM=NPTWO()

       DO I=0,NUMNOD
          IPPMAP(I)=I+KREPQM*(IREPQM-1)
       ENDDO

#if KEY_G09==1
       IF(PRNLEV.GT.2) THEN
          IF(OPTCNT.EQ.0) THEN
             WRITE(OUTU,*)'G09> USE ',npgrp, &
                  ' GROUPS OF PARALLEL PROCESSES'
             WRITE(OUTU,*)
          ENDIF
#elif KEY_QCHEM==1
       IF(PRNLEV.GT.2) THEN
          IF(OPTCNT.EQ.0) THEN
             WRITE(OUTU,*)'QCHEM> USE ',npgrp, &
                  ' GROUPS OF PARALLEL PROCESSES'
             WRITE(OUTU,*)
          ENDIF
#else /**/
       IF(PRNLEV.GT.2) THEN
          WRITE(OUTU,*)'GUKENE> USE ',npgrp, &
               ' GROUPS OF PARALLEL PROCESSES'
          WRITE(OUTU,*)
#endif 
       ENDIF
    ENDIF
#endif /* (parallel)*/
#endif 
#endif /* (replica)*/
       !     
#if KEY_PARALLEL==1
#if KEY_REPLICA==1
#if KEY_RPATH==1
    IF(QFRAG)THEN
       MYNOD=MYNOD_SAVE
       MYNODP=MYNOD+1
       NUMNOD=NUMNOD_SAVE
       NODDIM=NPTWO()
       DO I=0,NUMNOD
          IPPMAP(I)=I+NUMNOD*(MYNODG/NUMNOD)
       ENDDO
    ENDIF
#endif 
#endif 
#endif 
    !
    ! Main loop over replicas
    !
    !
    ! This forces a restart for all components of this job
    !
#if KEY_GAMESSUK==1
    STARTUP = QINIGM
#endif 

#if KEY_GAMESSUK==1
#if KEY_REPLICA==1
#if KEY_RPATH==1
    !
    do IREPL = IREPQM, IREPQM + NREPQMNOD -1 
       !
       !     update igmsel array for this replica
       !
       call RECOPSEL(IREPL)
       !
       !     Setup environment variables to separate group's I/O
       CALL ENVIGRP(IREPL)
#endif 
#endif 
#endif 
       !

       !     determine block factor (for every QM package)

       BLFACTOR=ONE
#if KEY_BLOCK==1 /*block*/
       IF(QBLOCK)THEN
          IBLQM=0
          DO I=1,NATOM
             ibl=iblckp(i)
             IF((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2))THEN
                IF(IBLQM.EQ.0)IBLQM=IBL
                IF(IBLQM.NE.IBL)THEN
                   CALL WRNDIE(-5,'<GUKINI>', &
                        'QM region must be within the same block.')
                ELSE
                   KK=IBL*(IBL+1)/2
                   BLFACTOR=BLCOEB(KK)
                ENDIF
             ENDIF
          ENDDO
          !         WRITE(OUTU,*)'GAMINI>meg,me,coef=',mynodg,mynod,blfactor
       ENDIF
#endif /*  (block)*/
       ! begin ewald
       !      write(*,*)'GUKINI>lewmode=',lewmode
       ! QC: UW_1017
       ! Need to remove G09 as well
!#if KEY_SQUANTM==0 && KEY_SCCDFTB==0 && KEY_MNDO97==0 && KEY_QCHEM==0 && KEY_GAMESS==0 && KEY_QTURBO==0 && KEY_NWCHEM==0
#if KEY_SQUANTM==0 && KEY_SCCDFTB==0 && KEY_MNDO97==0 && KEY_QCHEM==0 && KEY_GAMESS==0 && KEY_QTURBO==0 && KEY_NWCHEM==0 && KEY_G09==0 
       IF(LEWMODE) THEN
          ! array allocation for kspace gradient..tricky.
          !        Because it is using standard ewald code from CHARMM, which is run before...
          IF(QFIRSTD) THEN
          END IF
          !
          ! Do the initial setup
#if KEY_QTURBO == 0
          CALL SETUP_QMEWD(QDONE)
#endif
          !     Prepare Ewald summation
          !        Compute the Ewald potential on QM atoms from all the MM atoms
          !C         use xi,yi,zi but you need to call swapxyz_image and define imattq!!
          !C         CALL QEWALDP(NATOM,XI),YI),ZI),.FALSE.)
#if KEY_SCCDFTB==0 && KEY_QTURBO == 0
          CALL QEWALDP(NATOM,X,Y,Z,.FALSE.)
#endif
          !
          write(*,*)'GUKINI>numat,empot(1)=',numat,empot(1)
       END IF
#endif 
       ! end ewald
       !
#if KEY_GAMESS==1 /*gamess*/
       RBR=ONE/BOHRR
       !     Update coordinates
       N=0
       DO I = 1,NATOM
          IF ((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2)) THEN
             N = N + 1
             C(1,N)=X(I)*RBR
             C(2,N)=Y(I)*RBR
             C(3,N)=Z(I)*RBR
          ENDIF
       ENDDO
       !     
       CALL CH2GMS(.FALSE.)
       CALL CGREP(NATOM,QMMMRP,DX,DY,DZ)
       IF(PRNLEV.GE.6) WRITE(OUTU,'(A,2F17.6)') &
            'QM/MM repulsion is (au,kcal) ',QMMMRP,QMMMRP*TOKCAL
       !     
       !     From gamess/grd1.src in case of other WFns.
       !     
       IF(NCHMAT.NE.0) THEN
          DO ICHARM=1,NCHMAT
             DXELMM(ICHARM)=ZERO
             DYELMM(ICHARM)=ZERO
             DZELMM(ICHARM)=ZERO
          ENDDO
       ENDIF
       !     
       !CC   do i=1,3*ngames
       !CC   eg(i)=zero
       !CC   enddo
       !     
       !     C      CALL START
       !     C      irest=0
       !     C      CALL GRADX
       QINIGM=.FALSE.
       CALL GAMESS
#endif /*  (gamess)*/
#if KEY_NWCHEM==1
       ! using KEY_NWCHEM + qmused_nwchem flags
       !write(*,*)'before nwchem> qmused_nwchem=',qmused_nwchem
       if(qmused_nwchem) then
          call nwchem_qm_atoms(x,y,z)
          call nwchem_mm_atoms(x,y,z)
          call nwchem_rtdb_clr
          call task(rtdb_handle)
          call nwchem_qmmm_results(gtot,dx,dy,dz,x,y,z)
          !write(*,*)'after nwchem>'
       endif
#endif
#if KEY_GAMESSUK==1 /*gamessuk*/
       !     
#if KEY_FLUCQ==1 /*flucq*/
       !     Tell FlucQ to start collecting the nuclear repulsion term
       CALL FQQCST
#endif /* (flucq)*/
       !     
       QINIGM=.FALSE.
       INIT=0
       IVER=5
       CALL GAMESS(INIT,ICODE,STARTUP, LQMEWD, IVER)

       if(icode .ne. 0)then
          if(icode .eq. 1)then
#if KEY_REPLICA==1
#if KEY_RPATH==1
             IF(QREP) THEN
                write(tmpstr,'(i3)')irepl
                msg =  &
                     'SCF convergence failure for replica '//tmpstr
             ELSE
                write(tmpstr,'(i3)')0
                msg = 'SCF convergence failure '//tmpstr
             ENDIF
#endif 
#else /**/
             msg = 'SCF convergence failure '//tmpstr
#endif 
          else if(icode .ne.0)then

             write(tmpstr,'(i3)')icode
             write(OUTU,*)'Return code '//tmpstr
          endif

          CALL WRNDIE(-1,'<GAMESS-UK>',msg)

       endif

       !     
       !     Recover Energy and gradient here
       !     
       GTOTOLD=GTOT              ! jwchu
       !c         CALL GMS2CHM(GTOT,DX,DY,DZ,GMSMAP,BLFACTOR,NATOM)
       ! quantum charges written into first (QM) section of CGQMMM 
       CALL GMS2CHM(GTOT,DX,DY,DZ,CGQMMM,GMSMAP,BLFACTOR,NATOM)
       ECURRENT=GTOT-GTOTOLD     ! jwchu
       IF (QECONT) THEN          ! jwchu
          N1=0
          DO I = 1,NATOM
             IF ((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2)) THEN
                N1=N1+1  ! jwchu
             ENDIF
          ENDDO
          DO I = 1,NATOM
             IF ((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2)) THEN
                ECONT(I)=ECONT(I)+ECURRENT/DBLE(N1)  ! jwchu
             ENDIF
          ENDDO
       ENDIF                     ! jwchu  
       !             
#endif /* (gamessuk)*/
       !     
#if KEY_GAMESS==1 /*gamess*/
       !     
       !     If this is DIESEL calculation then get the energy
       CALL DIESELE(E)
       !     
       !CC   call gamend(99)
       !     
       !     MOPAC energies are already in kcal/mole.
       !     
       !     This is NOT TRUE ANYMORE!!!!!
       !     
       !     IF(MPCWFN) THEN
       !     GTOT=E+QMMMRP*TOKCAL
       !     ELSE
       !     GTOT=(E+QMMMRP)*TOKCAL
       !     ENDIF
       !     
       !     NOTE: QMMMRP is already scaled by BLFACTOR
       GTOTOLD=GTOT
#if KEY_PARALLEL==1 /*parallel*/
       IF(MYNOD.EQ.0) THEN
          GTOT= GTOT + E*TOKCAL*BLFACTOR+QMMMRP*TOKCAL
       ELSE
          GTOT=ZERO
       ENDIF
#else /* (parallel)*/
       GTOT=GTOT + E*TOKCAL*BLFACTOR+QMMMRP*TOKCAL
       !     C      GTOT=(E+QMMMRP)*TOKCAL*BLFACTOR
#endif /* (parallel)*/
       ECURRENT=GTOT-GTOTOLD
       IF(QECONT) THEN
          N1=0
          DO I = 1,NATOM
             IF ((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2)) THEN
                N1=N1+1          
             ENDIF
          ENDDO
          DO I = 1,NATOM
             IF ((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2)) THEN
                ECONT(I)=ECONT(I)+ECURRENT/DBLE(N1) 
             ENDIF
          ENDDO
       ENDIF
       !
       !     
       !     MM atoms, without igmsel(i)=5, unless BLUR !!
       !     
       N = 0
       DO I = 1,NATOM
          IF((IGMSEL(I).EQ.0).OR.((IGMSEL(I).EQ.5).AND.QBLUCH))THEN
             N = N + 1
             DX(I) = DX(I) - DXELMM(N)*TOKCAL*RBR*BLFACTOR
             DY(I) = DY(I) - DYELMM(N)*TOKCAL*RBR*BLFACTOR
             DZ(I) = DZ(I) - DZELMM(N)*TOKCAL*RBR*BLFACTOR
             !     
             IF(PRNLEV.GT.6) THEN
                WRITE(OUTU,334) I,N, - DXELMM(N)*TOKCAL*RBR &
                     , - DYELMM(N)*TOKCAL*RBR &
                     , - DZELMM(N)*TOKCAL*RBR
             ENDIF
             !     
          ENDIF
       ENDDO
       !     
#endif /* (gamess)*/
       !
       ! begin ewald
#if KEY_SQUANTM==0 && KEY_MNDO97==0
       IF(LEWMODE) THEN
          ! Compute Kspace gradient contribution, but not added here.
          ! Added in subroutine KSPACE (ewalf.src)
          ! FIXME_EWALD         CALL QKSPACD(NATOM,NUMAT,(PDXYZ))

          ! Free the memory allocation
       END IF
#endif 
       ! end ewald
       !     
       !     QM atoms (in parallel they are already summed up!)
       !     
#if KEY_PARALLEL==1
       IF(MYNOD.GT.0) GOTO 999
#endif 

#if KEY_GAMESS==1
       N = 0
       DO I = 1,NATOM
          IF ((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2)) THEN
             N = N + 1
             IPT=3*(N-1)+1
             DX(I) = DX(I) + EG(IPT)*TOKCAL*RBR*BLFACTOR
             DY(I) = DY(I) + EG(IPT+1)*TOKCAL*RBR*BLFACTOR
             DZ(I) = DZ(I) + EG(IPT+2)*TOKCAL*RBR*BLFACTOR
             !     
             IF(PRNLEV.GT.6) THEN
                WRITE(OUTU,334) I,N, EG(IPT)*TOKCAL*RBR     &
                     , EG(IPT+1)*TOKCAL*RBR &
                     , EG(IPT+2)*TOKCAL*RBR
334             FORMAT(2I10,3F14.6,3F14.9)
             ENDIF
             !     
          ENDIF
       ENDDO
#endif 
       !     
999    continue
#if KEY_GAMESSUK==1
#if KEY_REPLICA==1
#if KEY_RPATH==1
    enddo                     ! End of loop over QM calculations
#endif 
#endif 
#endif 

#if KEY_QCHEM==1 && KEY_GAMESS==0 && KEY_NWCHEM==0
    GTOTOLD=GTOT
    IF(QMICRO) THEN 
       !CALL PSND4(QFRSTMICIT,1)
       !write(*,*)'QFRSTMICIT = ',QFRSTMICIT
       IF(QFRSTMICIT) THEN 
          !write(*,*)'CALLING QCHEM'
          CALL QCHEM(E,DX,DY,DZ,CGX,AMASSX,IACX,NDD1,DD1,QSECD,IUPT,JUPT)
       ENDIF
    ELSE 
       !write(*,*)'CALLING QCHEM'
       CALL QCHEM(E,DX,DY,DZ,CGX,AMASSX,IACX,NDD1,DD1,QSECD,IUPT,JUPT)
    ENDIF
    GTOT=E*TOKCAL

    !     We obtain E,DX,DY,DZ from Q-chem output
    !     We put X,Y,Z to input for Q-chem

    !     Add loops to do QM/MM NEB with Q-Chem 
    ECURRENT=GTOT-GTOTOLD

    IF(QECONT) THEN
       N1=0
       DO I = 1,NATOM
          IF ((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2)) THEN
             N1=N1+1          
          ENDIF
       ENDDO
       DO I = 1,NATOM
          IF ((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2)) THEN
             ECONT(I)=ECONT(I)+ECURRENT/DBLE(N1) 
          ENDIF
       ENDDO
    ENDIF
#endif 

#if KEY_G09==1
    GTOTOLD=GTOT
    call qg09(E,DX,DY,DZ,CGX,AMASSX,IACX)
    GTOT=E*TOKCAL
#endif 

#if KEY_QTURBO==1
    GTOTOLD=GTOT
    CALL QTURBO(E,DX,DY,DZ,CGX,AMASSX,IACX,NDD1,DD1,QSECD,IUPT,JUPT)

    GTOT=E*TOKCAL

    !     We obtain E,DX,DY,DZ from TURBOMOLE output
    !     We put X,Y,Z to input for TURBOMOLE

    !     Add loops to do QM/MM NEB with TURBOMOLE
    ECURRENT=GTOT-GTOTOLD

    IF(QECONT) THEN
       N1=0
       DO I = 1,NATOM
          IF ((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2)) THEN
             N1=N1+1          
          ENDIF
       ENDDO
       DO I = 1,NATOM
          IF ((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2)) THEN
             ECONT(I)=ECONT(I)+ECURRENT/DBLE(N1) 
          ENDIF
       ENDDO
    ENDIF
#endif 

#if KEY_PARALLEL==1
#if KEY_REPLICA==1
#if KEY_RPATH==1
    !     
    !     Leave parallel/parallel mode
    !     
#if KEY_REPDSTR==1
    if(.not.qrepdstr)then                 
#endif
    IF(npgrp .ne. 1)THEN
       NUMNOD=NUMNODG
       MYNOD=MYNODG
       MYNODP=MYNOD+1
       NODDIM=NPTWO()
       CALL CUBE(MYNOD,NUMNOD,IPPMAP)
    ENDIF
#if KEY_REPDSTR==1
    ENDIF                                 
#endif
#endif 
#endif 
    IF(QFRAG)THEN
       NUMNOD=NUMNODG
       MYNOD=MYNODG
       MYNODP=MYNOD+1
       NODDIM=NPTWO()
       CALL CUBE(MYNOD,NUMNOD,IPPMAP)
    ENDIF
#endif 
    !
#if KEY_QCHEM==1 && KEY_GAMESS==0
    !write(*,*)'filech = ',FILCHRG(1:LCH)
    !WRITE(*,*)'TESTING FOR CHARGE READING',QQCHARG,QFRSTMICIT,QQCRP
    IF(QQCHARG) THEN 
!      IF(QFRSTMICIT) THEN 
!         !DO NOTHING
!      ELSE
       IF(QQCRP) THEN
          INQUIRE(file=FILCHRG(1:LCH),exist=lexist)
       ELSE
          INQUIRE(file='charges.dat',exist=lexist)
       ENDIF
       IF (lexist) THEN
          !write(*,*)'READING MULIKEN CHARGES'
          IF(QQCRP) THEN
             OPEN (unit=11, file=FILCHRG(1:LCH), status='old')
          ELSE
             OPEN (unit=11, file='charges.dat', status='old')
          ENDIF
          REWIND(11)   
          IF(MYNODG.EQ.0) THEN
            DO I=1, NATOM
               IF((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2))THEN
                  READ(11,'(3X,F19.16)')QMMUL(I)
                  CGX(I)=QMMUL(I)
                  !write(*,*)NATOM,I,IGMSEL(I),QMMUL(I),CGX(I)
               ELSE 
#if KEY_PARALLEL==1
                  IF(MYNODG.GT.0) CGX(I)=ZERO    
#endif
               ENDIF
            ENDDO
          ENDIF
#if KEY_PARALLEL==1
          CALL GCOMB(CGX,NATOM) 
#endif
       ENDIF
       IF (.not. QSMBP) THEN
          call unlink('charges.dat')
       ENDIF 
!      ENDIF
    ENDIF
#endif 

    CALL GETQMCHG
    !
    !write(50+mynodg,'(a,5i5)')'gukene>end:mynod,numnod,irepqm=',mynod,numnod,irepqm
    RETURN
  END SUBROUTINE GUKENE
  !
  SUBROUTINE GETQMCHG
    !-----------------------------------------------------------------------
    !     Get the charges from ab initio program
    !     Currently this works only for GAMESS but others
    !     can do the same (use COMMON /QMCHG/ in gamess_ltm.src)
    !
    use chm_kinds
    use dimens_fcm
    use psf
    use gamess_fcm
    !
    INTEGER IPT,I
    !
    IPT=0
    DO I=1,NATOM
       QMCMUL(I)=CG(I)
       QMCLOW(I)=CG(I)
       QMCKOL(I)=CG(I)
       IF((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2))THEN
          IPT=IPT+1
          QMCMUL(I)=QMMUL(IPT)
          QMCLOW(I)=QMLOW(IPT)
          QMCKOL(I)=QMKOL(IPT)
       ENDIF
    ENDDO
    !
    RETURN
  END SUBROUTINE GETQMCHG
  !
  SUBROUTINE SCFITCHM(NBASIS,D,S,FAO,ZAN,NSHELL,KATOM,KMIN,KMAX)
    !-----------------------------------------------------------------------
    !
    !     This routine calculates:
    !
    !     - Mulliken charges
    !     - adds to fock matrix from QM/MM Ewald
    !
    use chm_kinds
    use number
    use consta
    use dimens_fcm
    use coord
    use psf
    use gamess_fcm
    use mndo97
    !
    INTEGER(gms_int) NBASIS,NSHELL,KATOM(*),KMIN(*),KMAX(*)
    real(chm_real) D(*),S(*),FAO(*),ZAN(*)
    !
#if KEY_GAMESSUK==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
#if KEY_GAMESS==0
    integer,PARAMETER :: MXAO=1024
#endif
#endif

    INTEGER I,J,K,L,IPT,IBFAT(MXAO)
    real(chm_real) SUM,SUM1,KC

    !QC: UW_1017 Remove G09 as well
!#if KEY_SQUANTM==0 && KEY_MNDO97==0 && KEY_QCHEM==0 && KEY_GAMESS==0 && KEY_QTURBO==0 /*nosquantm*/
#if KEY_SQUANTM==0 && KEY_MNDO97==0 && KEY_QCHEM==0 && KEY_GAMESS==0 && KEY_QTURBO==0 && KEY_G09==0 /*nosquantm*/
    !
    kc=ccelec
    !     If there is no Ewald to calculate return immediately
    !C      write(*,*)'SCFITCHM>lqmewd=',lqmewd

    IF(.NOT.LQMEWD)RETURN
    !
    L=NBASIS*(NBASIS+1)/2
    !
    !C      write(6,*)'nbasis',nbasis
    !C      write(6,*)'katom',(katom(j),j=1,10)
    !C      write(6,*)'kmin',(kmin(j),j=1,10)
    !C      write(6,*)'kmax',(kmax(j),j=1,10)
    !C      write(6,*)'s',(s(j),j=1,10)
    !C      write(6,*)'d',(d(j),j=1,10)
    !C      write(6,*)'fao',(fao(j),j=1,10)
    !C      write(6,*)'zan',(zan(j),j=1,10)
    !C
    !C      WRITE(6,*)'charges-before=',(chag(i),i=1,ngames)
    !
    !     fill IBFAT array from /NSHEL/ data
    !     IBFAT(I): to which atom I-th basis function belongs
    IPT=1
    DO I = 1, NSHELL
       DO J=IPT,IPT+KMAX(I)-KMIN(I)
          IBFAT(J)=KATOM(I)
       ENDDO
       IPT=IPT+KMAX(I)-KMIN(I)+1
    ENDDO
    !
    !C      write(6,*)'ibfat',(ibfat(j),j=1,nbasis)

    !     Initialize Mulliken charges array

    !C      write(6,*)'ngames',ngames

    DO I=1,NGAMES
       CHAG(I)=ZAN(I)
    ENDDO
    !     Calculate charges
    DO I = 1, NBASIS
       SUM = ZERO
       DO J = 1, I
          SUM=SUM+D(I*(I-1)/2+J)*S(I*(I-1)/2+J)
       ENDDO
       DO J = I + 1, NBASIS
          SUM=SUM+D(J*(J-1)/2+I)*S(J*(J-1)/2+I)
       ENDDO
       CHAG(IBFAT(I)) = CHAG(IBFAT(I)) - SUM
    ENDDO
    !
    !C      WRITE(6,*)'charges-after=',(chag(i),i=1,ngames)
    !
#if KEY_SCCDFTB==0 && KEY_QCHEM==0 && KEY_QTURBO==0
    CALL QEWALDP(NATOM,X,Y,Z,.TRUE.)
#endif
    !
    write(*,*)'SCFITCHM>eslf=',(eslf(i)*kc,i=1,ngames)
    write(*,*)'SCFITCHM>empot=',(empot(i)*kc,i=1,ngames)
    sum=zero
    sum1=zero
    do i =1, ngames
       sum=sum-chag(i)*eslf(i)
       sum1=sum1-chag(i)*empot(i)
    enddo
    !     0.5 according to paper!!
    write(*,*)'SCFITCHM>eself,empot=',0.5*sum*kc,kc*sum1
    !
    !     Check for units !??
    !      DO I = 1, NGAMES
    !         ESLF(I)  = ESLF(I)
    !      END DO
    DO I = 1, NBASIS
       !CC         write(*,*)'SCFITCHM>i,ibfat,eslf,empot,fao=',
       !CC     $        i,ibfat(i),eslf(ibfat(i)),empot(ibfat(i)),fao(i*(i+1)/2)
       FAO(I*(I+1)/2)=FAO(I*(I+1)/2) &
            -HALF*ESLF(IBFAT(I))-EMPOT(IBFAT(I))
    ENDDO

#endif /*  (nosquantm)*/
    !
    RETURN
  END SUBROUTINE SCFITCHM
  !
#if KEY_GAMESS==1 /*gamess_utils*/
  SUBROUTINE CGREP(NATOM,REPULS,DX,DY,DZ)
    !-----------------------------------------------------------------------
    !
    !     This routine calculates nuclear repulsion between
    !     CHARMM atoms and GAMESS atoms + derivatives
    !
    use chm_kinds
    use dimens_fcm
    use stream
    use coord
    use gamess_fcm
    use consta
    use number
    use parallel
    !
    real(chm_real) REPULS,DX(*),DY(*),DZ(*)
    INTEGER NATOM
    !
    INTEGER(gms_int) NAT,ICH,MUL,NUM,NX,NE,NA,NB,IAN
    real(chm_real) ZAN, C
    COMMON /INFOA / NAT,ICH,MUL,NUM,NX,NE,NA,NB, &
         ZAN(MAXGMS),C(3,MAXGMS),IAN(MAXGMS)
    !
    INTEGER I,J,N,L,KBLUCH,KKBLCH,NN
    real(chm_real) ERRF,ETMP,SIGM1,SIGM2,TSQP
    real(chm_real) Q1,Q2,X1,X2,Y1,Y2,Z1,Z2,R12,RR12,EL,ELR
    real(chm_real) X12,Y12,Z12,RBR
#if KEY_REPLICA==0
    real(chm_real) BLFACTOR          
#endif
    !
    !
    !     Put derivatives in the right places in DX,DY,DZ arrays
    !     and take Newton's 3rd law into account. Also 
    !     here we don't care for cutoff for the same reason as
    !     above.
    !
#if KEY_REPLICA==0
    BLFACTOR=ONE     
#endif
    RBR=ONE/BOHRR
    L = 0
    TSQP = TWO/SQRT(PI)
    REPULS = ZERO
    ! DIRTY ...
    CALL ERRT
    ! DIRTY ...
    IF(QBLUCH) CALL ERRT

    ! multi-layered qm/mm and then return after that.
#if KEY_SQUANTM==1 /*squantm*/
    If(QMLAY_high) then
       call CGREP_mlayer(natom,repuls,dx,dy,dz,C,ZAN)
       return
    End if
#endif /* (squantm)*/

#if KEY_PARALLEL==1
    IF (MYNOD.GT.0) RETURN
#endif 
    !
    !     This loop is for QM nuclei - MM atoms electrostatic interaction
    !     It deals also with QM nuclei - Blurred charge interaction
    !
    DO I = 1,NATOM
       IF ((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2)) THEN
          !
          !        QM atoms
          !
          L=L+1
          Q1 = ZAN(L)
          IF(FQQCHG(L).GT.-NINE99) Q1 = FQQCHG(L)
          X1 = C(1,L)
          Y1 = C(2,L)
          Z1 = C(3,L)
          N = 0
          KBLUCH=1
          DO J = 1, NATOM
             IF((IGMSEL(J).EQ.0).OR.((IGMSEL(J).EQ.5).AND.QBLUCH))THEN
                !
                !           MM atoms
                !
                N = N + 1
                X2 = XCHM(N)
                Y2 = YCHM(N)
                Z2 = ZCHM(N)
                X12 = X1-X2
                Y12 = Y1-Y2
                Z12 = Z1-Z2
                RR12 = X12**2+Y12**2+Z12**2
                R12 = SQRT(RR12)
                IF(QBLUCH.AND.(N.EQ.IBLUCH(KBLUCH))) THEN
                   !
                   !                    QM nuclei - Blurred charge interaction
                   !
                   Q2=CGBLCH(KBLUCH)
                   ETMP=R12/SGBLCH(KBLUCH)
                   EL = Q1*Q2/R12*BLFACTOR
                   ELR=EL/RR12 &
                        *(ERRF(ETMP)-TSQP*ETMP*EXP(-ETMP*ETMP)) &
                        *TOKCAL/BOHRR
                   EL = EL*ERRF(ETMP)
                   KBLUCH=KBLUCH+1
                ELSE
                   Q2 = QCHM(N)
                   EL = Q1*Q2/R12*BLFACTOR
                   ELR = EL/RR12*TOKCAL/BOHRR
                ENDIF
                REPULS = REPULS + EL
#if KEY_FLUCQ==1
                ! broken: CALL FQQCOR(J,EL)
#endif 
                DX(I) = DX(I) - X12*ELR
                DX(J) = DX(J) + X12*ELR
                DY(I) = DY(I) - Y12*ELR
                DY(J) = DY(J) + Y12*ELR
                DZ(I) = DZ(I) - Z12*ELR
                DZ(J) = DZ(J) + Z12*ELR
             ENDIF
          ENDDO
       ENDIF
    ENDDO
    !
    !     This loop is for Blurred charges - MM atoms electrostatic interaction
    !     It deals also with the Blurred charge - Blurred charge interaction
    !
    !============================
    !
    !     SIMPLIFICATION: (????)
    !     For now we deal with blurred charges not interacting
    !     with QM region in the CHARMM as classical atoms.
    !     The following code is not good for MM - Blur and Blur - Blur,
    !     because it doesn't deal correctly with the bonded atoms!!
    !
    !     USE: bnbnd%inblo,bnbnd%jnb
    !
    !
    IF(QBLUCH) RETURN
    N = 0
    KBLUCH = 1
    DO I = 1,NATOM
       IF(QBLUCH.AND.((IGMSEL(I).EQ.0).OR.(IGMSEL(I).EQ.5)))THEN
          N = N + 1
          IF(N.EQ.IBLUCH(KBLUCH)) THEN
             !
             !              Found KBLUCH-th blurred charge.
             !
             X1 = XCHM(N)
             Y1 = YCHM(N)
             Z1 = ZCHM(N)
             Q1 = CGBLCH(KBLUCH)
             SIGM1 = SGBLCH(KBLUCH)
             KBLUCH = KBLUCH + 1
             NN = 0
             KKBLCH = 1
             DO J = 1,NATOM
                IF((IGMSEL(J).EQ.0).OR.(IGMSEL(J).EQ.5))THEN
                   !
                   !                 Either MM or Blurred atom
                   !
                   NN = NN + 1
                   X2 = XCHM(NN)
                   Y2 = YCHM(NN)
                   Z2 = ZCHM(NN)
                   X12 = X1-X2
                   Y12 = Y1-Y2
                   Z12 = Z1-Z2
                   RR12 = X12**2+Y12**2+Z12**2
                   R12 = SQRT(RR12)
                   ELR=ZERO
                   EL=ZERO
                   IF (NN.EQ.IBLUCH(KKBLCH)) THEN
                      IF (KKBLCH.GE.KBLUCH) THEN
                         !
                         !                          This is for blurred charge - blurred charge
                         !                          Blurred charges with index less than KBLUCH
                         !                          are ignored (EL,ELR is put to zero)
                         !
                         SIGM2 = ONE/SQRT(EBLUCH(KKBLCH))
                         ETMP=R12/SQRT(SIGM1*SIGM1+SIGM2*SIGM2)
                         EL=Q1*CGBLCH(KKBLCH)/R12
                         ELR=EL/R12 &
                              *(ERRF(ETMP)-TSQP*ETMP*EXP(-ETMP*ETMP)) &
                              *TOKCAL/BOHRR
                         EL=EL*ERRF(ETMP)
                      ENDIF
                      KKBLCH = KKBLCH + 1
                   ELSE
                      !                       This is for Blurred charge - MM charge
                      ETMP=R12/SIGM1
                      EL=Q1*QCHM(NN)/R12
                      ELR=EL/R12 &
                           *(ERRF(ETMP)-TSQP*ETMP*EXP(-ETMP*ETMP)) &
                           *TOKCAL/BOHRR
                      EL=EL*ERRF(ETMP)
                   ENDIF
                   REPULS=REPULS+EL
#if KEY_FLUCQ==1
                   ! broken: CALL FQQCOR(J,EL)
#endif 
                   DX(I) = DX(I) - X12*ELR
                   DX(J) = DX(J) + X12*ELR
                   DY(I) = DY(I) - Y12*ELR
                   DY(J) = DY(J) + Y12*ELR
                   DZ(I) = DZ(I) - Z12*ELR
                   DZ(J) = DZ(J) + Z12*ELR
                ENDIF
             ENDDO
          ENDIF
       ENDIF
    ENDDO
    !
    RETURN
  END SUBROUTINE CGREP
  !
  SUBROUTINE CGREPE(NATOM)
    !-----------------------------------------------------------------------
    !
    !     This is here for the debugging purpose!
    !     This routine calculates nuclear repulsion between
    !     CHARMM atoms and GAMESS atoms - IT DOES NOT WORK WITH BLUR!!!
    !                                     Try to get rid of it
    !                                     by filling ETERM array with the
    !                                     terms calculated in the CGREP
    !                                     subroutine
    !
    use chm_kinds
    use dimens_fcm
    use stream
    use gamess_fcm
    use consta
    use number
    !
    INTEGER NATOM
    !
    INTEGER(gms_int) NAT,ICH,MUL,NUM,NX,NE,NA,NB,IAN
    real(chm_real) ZAN, C
    COMMON /INFOA / NAT,ICH,MUL,NUM,NX,NE,NA,NB, &
         ZAN(MAXGMS),C(3,MAXGMS),IAN(MAXGMS)
    !
    real(chm_real) E, EG
    COMMON /FUNCT/ E, EG(3*MAXGMS)
    !
    INTEGER I,J,N,L,KBLUCH
    !CC      real(chm_real) ERRF
    real(chm_real) Q1,Q2,X1,X2,Y1,Y2,Z1,Z2,R12,RR12,EL
    real(chm_real) X12,Y12,Z12
    real(chm_real) REPULS
    !

    ! multi-layered qm/mm and then return after that.
#if KEY_SQUANTM==1 /*squantm*/
    If(QMLAY_high) then
       call CGREPE_mlayer(natom,Prnlev,ZAN,C,E,EG)
       return
    End if
#endif /* (squantm)*/

    L = 0
    REPULS = ZERO
    DO I = 1,NATOM
       IF((IGMSEL(I).EQ.1).OR.(IGMSEL(I).EQ.2)) THEN
          !
          !        QM atoms
          !
          L=L+1
          Q1 = ZAN(L)
          IF(FQQCHG(L).GT.-NINE99) Q1 = FQQCHG(L)
          X1 = C(1,L)
          Y1 = C(2,L)
          Z1 = C(3,L)
          N = 0
          KBLUCH=1
          DO J = 1, NATOM
             IF((IGMSEL(J).EQ.0).OR.((IGMSEL(J).EQ.5).AND.QBLUCH))THEN
                !
                !        MM atoms
                !
                N = N + 1
                X2 = XCHM(N)
                Y2 = YCHM(N)
                Z2 = ZCHM(N)
                X12 = X1-X2
                Y12 = Y1-Y2
                Z12 = Z1-Z2
                RR12 = X12**2+Y12**2+Z12**2
                R12 = SQRT(RR12)
                !CC                  IF(QBLUCH.AND.(N.EQ.IBLUCH(KBLUCH))) THEN
                !CC                     Q2=CBLUCH(KBLUCH)*ERRF(R12/SQRT(EBLUCH(KBLUCH)))
                !CC                     KBLUCH=KBLUCH+1
                !CC                  ELSE
                Q2 = QCHM(N)
                !CC                  ENDIF
                EL = Q1*Q2/R12
                REPULS = REPULS + EL
             ENDIF
          ENDDO
       ENDIF
    ENDDO
    !
    IF(PRNLEV.GE.2) THEN
       WRITE(OUTU,'(A,2F20.8)')'QM/MM repulsion (a.u.,kcal/mole) = ', &
            REPULS,REPULS*TOKCAL
       WRITE(OUTU,'(A,2F20.8)')'QM/MM total en. (a.u.,kcal/mole) = ', &
            E+REPULS,(E+REPULS)*TOKCAL
    ENDIF
    !
    RETURN
  END SUBROUTINE CGREPE
  !
  SUBROUTINE CHGMIU(IR,IW,inize)
    !-----------------------------------------------------------------------
    !
    !     Routine to define unit numbers for input/output of GAMESS
    !     The rest of files are opened by gamess with names defined
    !     by environment variables set by shell or ENVIron CHARMM command.
    !
    use chm_kinds
    use stream
    !
    character(len=240)   name
    integer kstrm,koutu
    COMMON /CHGMIO/  KSTRM,KOUTU,name
    logical         qopen,qform,qwrite
    !
    INTEGER IR,IW,inize,ip
    !
    !      The following is the complete table of I/O
    !      used in GAMESS program
    !      Wich ones are used depend on the type of GAMESS job.
    !
    !      IRCDATA $SCR/$JOB.irc
    !        INPUT $SCR/$JOB.F05
    !        PUNCH $SCR/$JOB.dat
    !      INTGRLS $SCR/$JOB.F08
    !       ORDINT $SCR/$JOB.F09
    !       JKFILE $SCR/$JOB.F09
    !      DICTNRY $SCR/$JOB.F10  ! ENVI must be always present
    !      DRTFILE $SCR/$JOB.F11
    !      CIVECTR $SCR/$JOB.F12
    !      NTNFMLA $SCR/$JOB.F13
    !       CIINTS $SCR/$JOB.F14
    !       WORK15 $SCR/$JOB.F15  ! ENVI must be always present
    !       WORK16 $SCR/$JOB.F16
    !      CSFSAVE $SCR/$JOB.F17
    !      FOCKDER $SCR/$JOB.F18
    !       DASORT $SCR/$JOB.F20  ! ENVI must be always present
    !
    IR=ISTRM
    IW=OUTU
    ip=7
    !
    KSTRM=IR
    KOUTU=IW
    if (inize.ne.-1) return
    if(ir.eq.5)ir=99
    if(inize.ge.0) then
       write(outu,*)'CHGMIU-0>ir,ip',ir,ip
       CALL SEQOPN(IR,'INPUT', 'OLD',.TRUE., 'FORMATTED')
       CALL SEQOPN(IP,'PUNCH', 'NEW',.FALSE.,'FORMATTED')
       !CC   call vinqre('unit',name,240,length,qopen,qform,qwrite,ir)
       inquire(unit=ir,name=name,opened=qopen)
#if KEY_PARALLEL==1
       CALL DDI_BCAST(1,'I',IR,1,0)
       CALL DDI_BCAST(1,'I',IW,1,0)
       call DDI_BCAST(1,'I',name,80,0)
#endif 
       write(outu,*)'CHGMIU-a>ir,iw,q,n=',ir,iw,qopen,name(1:40)
    else
       IR=KSTRM
       IW=KOUTU
       CALL SEQOPN(IR,'INPUT', 'OLD',.TRUE., 'FORMATTED')
       CALL SEQOPN(IP,'PUNCH', 'NEW',.FALSE.,'FORMATTED')
       call getenv('INPUT',name)
       rewind(unit=ir)
       write(outu,*)'CHGMIU-b>ir,iw,q,n=',ir,iw,qopen,name(1:40)
    endif
    !
    RETURN
  END SUBROUTINE CHGMIU
  !
  SUBROUTINE GAMEND(kstrm)
    !-----------------------------------------------------------------------
    !     Routine to put pointer in input file after all gamess input,
    !     i.e. after last ' $end' string! 
    !
    use chm_kinds
    use dimens_fcm
    use stream
    use string
    !
    integer kstrm
    CHARACTER(len=80) COMLYN
    INTEGER COMLEN,NENDS,IENDS
    !
    !CC      IF(IOLEV.GT.0) THEN
    REWIND KSTRM
    NENDS=0
100 READ(KSTRM,45,ERR=945,END=955) COMLYN(1:80)
45  FORMAT(A)
    COMLEN=80
    CALL CNVTUC(COMLYN,COMLEN)
    IF (INDX(COMLYN,COMLEN,'$END',4).GT.0) NENDS=NENDS+1
    GOTO 100
955 CONTINUE
    WRITE(outu,*)'NUMBER OF ENDs',NENDS
    REWIND KSTRM
    IENDS=0
101 READ(KSTRM,45,ERR=945,END=945) COMLYN(1:80)
    COMLEN=80
    CALL CNVTUC(COMLYN,COMLEN)
    IF (INDX(COMLYN,COMLEN,'$END',4).GT.0) IENDS=IENDS+1
    IF(IENDS.EQ.NENDS) RETURN
    GOTO 101
    RETURN
945 CALL WRNDIE(-2,'<GAMEND>','Unable to find $END')
    !CC      ENDIF
    !
    RETURN
  END SUBROUTINE GAMEND
  !
  !
#endif /* (gamess_utils)*/
  !
#if KEY_FLUCQ==0
#if KEY_GAMESS==1
  SUBROUTINE FQQMMM(DMAT)
    ! Dummy subroutine to ensure GAMESS-US links properly with CHARMM when
    ! FlucQ is not being used
    REAL(CHM_REAL) DMAT(*)
    RETURN
  END SUBROUTINE FQQMMM
#endif 
#if KEY_GAMESSUK==1
  ! Dummy subroutines to ensure GAMESS-UK links properly with CHARMM when
  ! FlucQ is not being used
  SUBROUTINE FQQCDN
    RETURN
  END SUBROUTINE FQQCDN
  SUBROUTINE FQQMMM(Q,ISO,NSHELS)
    INTEGER Q,ISO,NSHELS
    RETURN
  END SUBROUTINE FQQMMM
  SUBROUTINE FQQCOR(I,J,VALUE)
    INTEGER I,J
    real(chm_real) VALUE
    RETURN
  END SUBROUTINE FQQCOR
#endif 
#endif 
  !
#if KEY_GAMESS==1 || KEY_QCHEM==1 || KEY_G09==1 /*gamess_diesel*/
  INTEGER FUNCTION STRFIN(LIN,KEY)
    !-----------------------------------------------------------------------
    !     No such routine seems to be available in string.src
    !     Finds the position of the string KEY in the LIN string
    !
    use chm_kinds
    !
    CHARACTER(len=*) LIN,KEY
    INTEGER L,I,J
    !
    J=LEN(KEY)
    L=LEN(LIN)
    !
    IF(L.LT.J)THEN
       STRFIN=0
       RETURN
    ELSE IF (L.EQ.J) THEN
       IF(LIN(1:L).EQ.KEY(1:J))THEN
          STRFIN=1
          RETURN
       ENDIF
       STRFIN=0
    ELSE
       DO I=1,L-J
          IF(LIN(I:I+J-1).EQ.KEY(1:J))THEN
             STRFIN=I
             RETURN
          ENDIF
       ENDDO
       STRFIN=0
    ENDIF
    !
    RETURN
  END FUNCTION STRFIN
  !
  SUBROUTINE DIESELE(EDIESL)
    !-----------------------------------------------------------------------
    !     This subroutine runs DIESEL program and gets the energy to EDIESL
    !
    use chm_kinds
    use number
    use dimens_fcm
    use stream
    use gamess_fcm
    !=====================================
    !     This is from GAMESS!!!
    !
    INTEGER, PARAMETER :: MXRT=100
    real(chm_real) ENUCR,EELCT,ETOT,SZ,SZZ,ECORE,ESCF,EERD,E1,E2, &
         VEN,VEE,EPOT,EKIN,ESTATE,STATN,EDFT,EDISP
    INTEGER(gms_int) NFZC,MCORBS,NCI,MORBS,MORB,NBF1
    !
    COMMON /ENRGYS/ ENUCR,EELCT,ETOT,SZ,SZZ,ECORE,ESCF,EERD,E1,E2, &
         VEN,VEE,EPOT,EKIN,ESTATE(MXRT),STATN,EDFT(3),EDISP
    COMMON /MCPAR / NFZC,MCORBS,NCI,MORBS,MORB,NBF1
    !
    !     End of COMMONs from GAMESS
    !=====================================
    real(chm_real) EDIESL
    REAL(chm_real4) RECORD(45001)
    !
    external lunass
    INTEGER L,IU,I,N,IMO,OMO,lunass
    CHARACTER FILENM*256,LIN*80
    !
    IF(KDIESL.LT.0) RETURN
    !
#if KEY_GNU==1
    !
    !     Prepare a file for Diesel calculations from MOINTS (mccas.src)
    !
    IMO=LUNASS(90)
    OMO=LUNASS(91)
    write(outu,*)'DIESELE>imo,omo=',imo,omo
    OPEN(IMO,FILE='test.f9',form='unformatted',status='old')
    OPEN(OMO,FILE='moints.f9',form='unformatted',status='unknown')
    write(outu,*)'DIESELE>nci,enucr,ecore,+=',nci,enucr,ecore
    WRITE(OMO)NCI
    WRITE(OMO)ENUCR+EFCORE
    READ(IMO)(RECORD(I),I=1,NCI*(NCI+1))
    WRITE(OMO)(RECORD(I),I=1,NCI*(NCI+1))
50  CONTINUE
    READ(IMO,END=60)RECORD
    WRITE(OMO)RECORD
    GOTO 50
60  CONTINUE
    CLOSE(IMO)
    CLOSE(OMO)
    !
    !     Run diesel....
    CALL SYSTEM('$DIESELSCRIPT')
    !
    call get_environment_variable('DIESELOUT', filenm, l)
    IU=LUNASS(90)
    OPEN(IU,FILE=FILENM,STATUS='UNKNOWN',ERR=20)
    !
10  READ(IU,'(A80)',END=20)LIN
    IF(KDIESL.EQ.1) THEN
       IF(STRFIN(LIN,'full MRCI').GT. 0)THEN
          DO I=1,5
             READ(IU,*)
          ENDDO
          READ(IU,'(A)')LIN
          READ(LIN,'(36X,F12.5)')EDIESL
          GOTO 30
       ENDIF
    ELSE IF (KDIESL.EQ.2) THEN
       IF(STRFIN(LIN,'full CI').GT. 0)THEN
          DO I=1,6
             READ(IU,*)
          ENDDO
          READ(IU,'(A)')LIN
          READ(LIN,'(36X,F12.5)')EDIESL
          GOTO 30
       ENDIF
    ELSE
       CALL WRNDIE(-5,'<DIESELE>', &
            'DIESEL this energy is not available.')
    ENDIF
    GOTO 10
20  CONTINUE
    CALL WRNDIE(-5,'<DIESELE>', &
         'DIESEL didn''t found the energy.')
    RETURN
30  CONTINUE
#else /**/
    CALL WRNDIE(-1,'<DIESELE>', &
         'DIESEL/CHARMM code doesn''t work on this platform.')
#endif 
    !
    RETURN
  END SUBROUTINE DIESELE
#endif /* (gamess_diesel)*/
  !
#if KEY_GAMESSUK==1
  !
  ! Called by GAMESS-UK to determine input and output streams
  !

  subroutine getfilenames(infile,outfile,length)

    use chm_kinds
    use dimens_fcm
    use replica_mod
    use gamess_fcm

    character infile*(*), outfile*(*)
    integer length

    outfile = ' '
    infile = ' '

    call gtnv("gamess.out",outfile)
    call gtnv("gamess.in",infile)

    if(outfile.eq.' ')then
       outfile = "gamess.out"
    endif

    if(infile.eq.' ')then
       infile = "gamess.in"
    endif

  end subroutine getfilenames
  !
#endif 
  SUBROUTINE COPSEL(ISLCT,QQINP)
    !-----------------------------------------------------------------------
    !     Copies selection vector to common block 
    !     so it may be used by GAMESS interface
    !     Call this routine only once and retain definition
    !     of QM, MM, and link atoms throughout the calculation.
    !     We call this from GUKINI which is called from charmm/charmm.src
    !
    !     IGMSEL(I) = 5  MM atom to be excluded from QM/MM interaction
    !     IGMSEL(I) = 2  Link atom
    !     IGMSEL(I) = 1  QM atom
    !     IGMSEL(I) = 0  MM atom
    !     IGMSEL(I) = -1 QM atom  (other replica)
    !     IGMSEL(I) = -2 Link atom (other replica)
    !     IGMSEL(I) = -5 MM atom to be excluded from its QM/MM 
    !                    interaction (other replica)
    !     IGMSEL(I) = -6 MM atom (other replica)
    !
    !     MM atom in position close to link atom is excluded from interaction
    !     of external charges to QM region. Instead of this atom is already
    !     a link atom so no need for two atoms in one place!
    !
    use chm_kinds
    use dimens_fcm
    use coord
    use gamess_fcm
    use stream
    use psf
    use number
    use parallel
    use replica_mod
    use pathm
    use chutil, only: getres,atomid
    !
    INTEGER ISLCT(*)
    LOGICAL QQINP
    !
    INTEGER I,J,I1,I2,N,LN,IS,IQ
    CHARACTER(len=8) SID, RID, REN, AC
    LOGICAL LNFLAG
    !
    ! for DIV - guanhua_puja_QC_UW1212 
    INTEGER K,NE,NQ,L
    REAL(chm_real) DQ
    !
    !     Assign 0,1, and 2
    DO I=1, NATOM
       IGMSEL(I)=ISLCT(I)
       IF (ATYPE(I)(1:2) == 'QQ') IGMSEL(I)=2
    ENDDO
    !
    !     Check if link atom is connected to any of its neighbors. If
    !     yes then that atom will not be included in QM/MM interaction.
    !     This is sometimes necessary to prevent oposite charge collision,
    !     since QM cannot prevent this to happen.
    !
    !     Assign 5
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
! for DIV - guanhua_puja_QC_UW1212
    DO I=1,NATOM
      NDIV(I)=0
    ENDDO
    IF (QGMDIV) then
        do i=1, natom
          zlhost(i)=cg(i)
          if (igmsel(i).eq.0) then
            k=getres(i,igpbs,ngrp)
            is=igpbs(k)+1
            iq=igpbs(k+1)
            dq=0.0d0
            nq=0
            ne=0
            do l=is,iq
              if (igmsel(l).eq.5) then
                dq=dq+cg(l)
                ne=ne+1
              endif
              if (igmsel(l).eq.0) then
                nq=nq+1
              endif
            enddo
            if (ne.gt.0) then
              ndiv(i)=1
              zlhost(i)=zlhost(i)+dq/nq
            endif
          endif
        enddo
    ENDIF
! 
    !
#if KEY_GAMESSUK==1
    !
    ! Flag members of all replicas except the first
    ! This call will be repeated when working on 
    ! a different replica.
    !
    call RECOPSEL(1)
#endif 
    !
    IF(PRNLEV.GE.2) THEN
       WRITE(OUTU,118)
       WRITE(OUTU,120)  &
            'Classical atoms excluded from the QM calculation'
       IF (QBLUCH) then
#if KEY_GAMESSUK==0
          if(recallint.ne.-1) then
             WRITE(OUTU,120)  &
                  'When BLUR option is used, it is controlled by RECALL'
          else
#endif 
             WRITE(OUTU,120)  &
                  'When BLUR option is used, it is controlled by WMAIN'
#if KEY_GAMESSUK==0
          endif
#endif 
       ENDIF
    ENDIF
118 FORMAT('------------------------------------------------')
#if KEY_G09==1
120 FORMAT('G09: ',A,':')
#elif KEY_QCHEM==1
120 FORMAT('QCHEM: ',A,':')
#elif KEY_NWCHEM==1
120 FORMAT('NWCHEM: ',A,':')
#else /* */
120 FORMAT('GUKINT: ',A,':')
#endif 
122 FORMAT(10X,I10,4(1X,A))
124 FORMAT(10X,'NONE.')
    N=0
    DO I=1,NATOM
       IF(IGMSEL(I).EQ.5) THEN
          CALL ATOMID(I,SID,RID,REN,AC)
          IF(PRNLEV.GE.2) WRITE(OUTU,122) I, &
               SID(1:idleng),RID(1:idleng),REN(1:idleng),AC(1:idleng)
          N=N+1
       ENDIF
    ENDDO
    IF(PRNLEV.GE.2) THEN
       IF(N.EQ.0) WRITE(OUTU,124)
       WRITE(OUTU,120) 'Quantum mechanical atoms'
    ENDIF
    N=0
    DO I=1,NATOM
       IF(IGMSEL(I).EQ.1) THEN
          CALL ATOMID(I,SID,RID,REN,AC)
          IF(PRNLEV.GE.2) WRITE(OUTU,122) I, &
               SID(1:idleng),RID(1:idleng),REN(1:idleng),AC(1:idleng)
          N=N+1
       ENDIF
    ENDDO
#if KEY_GAMESS==1 || KEY_GAMESSUK==1
    NGAMES=N                             
#endif
    IF(PRNLEV.GE.2) THEN
       IF(N.EQ.0) WRITE(OUTU,124)
       WRITE(OUTU,120) 'Quantum mechanical link atoms'
    ENDIF
    N=0
    DO I=1,NATOM
       IF(IGMSEL(I).EQ.2) THEN
          CALL ATOMID(I,SID,RID,REN,AC)
          IF(PRNLEV.GE.2) WRITE(OUTU,122) I, &
               SID(1:idleng),RID(1:idleng),REN(1:idleng),AC(1:idleng)
          N=N+1
       ENDIF
    ENDDO
#if KEY_GAMESS==1 || KEY_GAMESSUK==1
    NGAMES=N+NGAMES                      
#endif
    IF(PRNLEV.GE.2) THEN
       IF(N.EQ.0) WRITE(OUTU,124)
       WRITE(OUTU,118)
    ENDIF
    !
    !     Allow for partial charges on any QM atom
    !
    N=0
    LN=0
    DO I = 1,NATOM
       !
       !     ******** BUG *******************
       !     Problem with the replica and MAXGMS. Here it does for
       !     MAXGMS * NREPLICA times FQQCHG stuff. But FQQCHG arrays
       !     are only MAXGMS. Right now it works only for
       !     NATOM (=nrepl*natom-per-repl)< MAXGMS
       !     For now we just skip this problem with the warning, and it
       !     doesn't overwrite the memory - so in most cases this is OK!!
       !     But make some better fix !!!!
       !     ********* BUG **********************************
       !
       IF ((abs(IGMSEL(I)).EQ.1).OR. &
            (abs(IGMSEL(I)).EQ.2)) THEN
          N = N + 1
          !
          !     Non integer charges for link atoms can be specified separately
          !     Also allow to change them subsequently with SCALar command 
          !     using QINP keyword in GAMEss command. 
          !
          LNFLAG=.FALSE.

          IF (ATYPE(I)(1:2) == 'QQ' .AND. NQQCHG /= 0) THEN
             LN=LN+1
             IF(QQCHG(LN).GT.-NINE99) THEN
                FQQCHG(N) = QQCHG(LN)
             ELSE
                FQQCHG(N) = -THOSND
             ENDIF
             !
             !     Don't have any more link atoms, put NQQCHG to 0
             !     for possible subsequent changes with SCALar command
             !     or on restarts.
             !
             IF(LN.EQ.NQQCHG) NQQCHG=0
             LNFLAG=.TRUE.
          ENDIF
          !
          !
          !     First check the flag QINP for all atoms.
          !
          IF(QQINP) THEN
             !
             !     All QM charges (accept ones specified in ADDLink
             !     are taken from PSF
             !
             IF(.NOT.LNFLAG) FQQCHG(N)=CG(I)
          ELSE
             IF(N.LE.MAXGMS) THEN
                FQQCHG(N)=-THOSND
             ELSE
                IF(PRNLEV.GT.2)WRITE(OUTU,'(A,I8)') &
                     'Number of QM atoms exceeded (REPLICA maybe OK?)',N
                GOTO 999
             ENDIF
          ENDIF
          !
          ! only output for the current replica
          !
          if(igmsel(i) .gt. 0) then
             IF(PRNLEV.GE.6) WRITE(OUTU,'(A,2I10,A,F15.5)') &
                  'GUKINT: ATOM(',I,N,') has QNUC: ',FQQCHG(N)
          endif
999       CONTINUE
       ENDIF
    ENDDO
    !
    ! Zero charges on quantum atoms to remove from MM term.
    !     This is maybe not needed ?? CHECK in nbonda.src !!
    !
    IF(QGMREM) THEN
       DO I=1, NATOM
          !           Replica ! use ABS()
          IF((ABS(IGMSEL(I)).EQ.1).OR.(ABS(IGMSEL(I)).EQ.2)) &
               CG(I)=ZERO
       ENDDO
    ENDIF
    !
    RETURN
  END SUBROUTINE COPSEL
  !
  SUBROUTINE FCOPSEL(IGRP)
    !-----------------------------------------------------------------------
    !     update copsel to reflect current fragment assignment
    !
    !     Enhancement: This should be different for QXFRAG true or false!
    !
    use chm_kinds
    use dimens_fcm
    use gamess_fcm
    use stream
    use psf
    use parallel
    !
    INTEGER IGRP
    INTEGER I
    !
    DO I=1,NATOM
       IF(IFRAG.NE.IGRP)THEN
          ! Other fragment
          IF(ABS(IGMSEL(I)).EQ.1) IGMSEL(I) = -1 
          IF(ABS(IGMSEL(I)).EQ.2) IGMSEL(I) = -2
          IF(ABS(IGMSEL(I)).EQ.5) IGMSEL(I) = -5
       ELSE
          ! This fragment                
          IF(ABS(IGMSEL(I)).EQ.1) IGMSEL(I) = 1 
          IF(ABS(IGMSEL(I)).EQ.2) IGMSEL(I) = 2
          IF(ABS(IGMSEL(I)).EQ.5) IGMSEL(I) = 5
       ENDIF
    ENDDO
    !
    RETURN
  END SUBROUTINE FCOPSEL
#endif /* (guk_main)*/
#endif /* (cadpac)*/
  !
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QUANTUM==1 || KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_G09==1 /*allqm*/
  !
  SUBROUTINE RECOPSEL(IREPL)
    !-----------------------------------------------------------------------
    !     update copsel to reflect current replica assignment
    !
    use chm_kinds
    use dimens_fcm
    use coord
    use gamess_fcm
    use stream
    use psf
    use number
    use parallel
    use replica_mod
    use pathm
    !
    INTEGER IREPL
    INTEGER I,J,I1,I2,N,LN,IS,IQ
    LOGICAL LNFLAG
    !
    !     Check if the atom belongs to some other replica and ignore (-1,-2,-5)
    !     TODO: Make a test here if QM region is within one replica
    !
    !      write(6,*)'recopsel',qrep, nrepqm, irepqm,
    !     &     (repnoa(i),i=1,natom)

#if KEY_REPLICA==1 /*replica*/
#if KEY_RPATH==1 /*rpath*/
    IF(QREP.AND.(NREPQM.GT.0)) THEN
       DO I=1,NATOM
          IF((REPNOA(I)-1.NE.IREPL).AND.(REPNOA(I).GT.1))THEN
             ! Other replica
             if(abs(igmsel(i)) .eq. 1) &
                  igmsel(i) = -1
             if(abs(igmsel(i)) .eq. 2) &
                  igmsel(i) = -2
             if(abs(igmsel(i)) .eq. 5) &
                                ! SUSPECT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                ! changed from -1 to -5 Nov 2002 by PS
                                ! replica path has yet to be tested with link atoms
                                ! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                  igmsel(i) = -5
             if(igmsel(i) .eq. 0) &
                  igmsel(i) = -6
          ELSE
             ! This replica
             if(abs(igmsel(i)) .eq. 1) &
                  igmsel(i) = 1
             if(abs(igmsel(i)) .eq. 2) &
                  igmsel(i) = 2
             if(abs(igmsel(i)) .eq. 5) &
                  igmsel(i) = 5
             if(igmsel(i) .eq. -6) &
                  igmsel(i) = 0
          ENDIF
       ENDDO
    ENDIF
#endif /* (rpath)*/
#endif /* (replica)*/
    RETURN
  END SUBROUTINE RECOPSEL
  !
  SUBROUTINE RSTCOPSEL
    use chm_kinds
    use dimens_fcm
    use gamess_fcm
    use psf
    INTEGER I
    DO I=1,NATOM
       if(abs(igmsel(i)) .eq. 2)then
          ! leave link atoms unchanged
          igmsel(i) = 2
       else
          ! every atom MM
          igmsel(i) = 0
       endif
    ENDDO
  END SUBROUTINE RSTCOPSEL
  !
  SUBROUTINE MICROCORR(ETERM,LENENT,DX,DY,DZ)
    use chm_kinds
    use dimens_fcm
    use number
    use psf
    use coordc
    use gamess_fcm
    use parallel

    real(chm_real) :: DX(*),DY(*),DZ(*),ETERM(*)
    integer :: i,ierr,LENENT
    real(chm_real),allocatable, dimension(:) :: tmpx
    real(chm_real),allocatable, dimension(:) :: tmpy
    real(chm_real),allocatable, dimension(:) :: tmpz
    real(chm_real),allocatable, dimension(:) :: tmpe

    allocate(tmpx(natom),stat=ierr)
    allocate(tmpy(natom),stat=ierr)
    allocate(tmpz(natom),stat=ierr)
    allocate(tmpe(LENENT),stat=ierr)

    IF(QFRSTMICIT) THEN 
       !write(*,*)'SAVING QM/MM FORCES' 
       !write(*,*)'OPTCNT = ',OPTCNT
#if KEY_COMP2==1 /*comp2*/
       ! Save current forces in comp2 array 
       do i=1,natom 
          xcomp2(i)=dx(i)
          ycomp2(i)=dy(i)
          zcomp2(i)=dz(i)
       enddo
       ! Save current energy terms 
       do i=1,LENENT
          ecorr(i)=eterm(i)
       enddo
#else /* (comp2)*/
       ! Save current forces in comp array 
       do i=1,natom 
          xcomp(i)=dx(i)
          ycomp(i)=dy(i)
          zcomp(i)=dz(i)
       enddo
       ! Save current energy terms 
       do i=1,LENENT
          ecorr(i)=eterm(i)
       enddo
#endif /*  (comp2)*/
    ELSE 
       !write(*,*)'CORRECTING CLASSICAL FORCES'
       !write(*,*)'OPTCNT = ',OPTCNT
#if KEY_COMP2==1 /*comp2*/
       if(optcnt.eq.1) then 
          do i=1,natom 
             ! copy saved QM gradient in tmp
             tmpx(i)=xcomp2(i)
             tmpy(i)=ycomp2(i)
             tmpz(i)=zcomp2(i)
             ! Compute Gcorr and save in comp2
             xcomp2(i)=tmpx(i)-dx(i)
             ycomp2(i)=tmpy(i)-dy(i)
             zcomp2(i)=tmpz(i)-dz(i)
             ! Augment current classical gradient with Gcorr
             dx(i)=dx(i)+xcomp2(i)
             dy(i)=dy(i)+ycomp2(i)
             dz(i)=dz(i)+zcomp2(i)
          enddo
          optcnt=-9999
          ! Compute and correct energy terms 
          do i=1,LENENT
             tmpe(i)=ecorr(i)
             ecorr(i)=eterm(i)-tmpe(i)
             eterm(i)=eterm(i)-ecorr(i)
          enddo
       else 
          ! Correct current forces 
          do i=1,natom 
             dx(i)=dx(i)+xcomp2(i)
             dy(i)=dy(i)+ycomp2(i)
             dz(i)=dz(i)+zcomp2(i)
          enddo
          ! Compute and correct energy terms 
          do i=1,LENENT
             eterm(i)=eterm(i)-ecorr(i)
          enddo
       endif
#else /* (comp2)*/
       if(optcnt.eq.1) then 
          do i=1,natom 
             ! copy saved QM gradient in tmp
             tmpx(i)=xcomp(i)
             tmpy(i)=ycomp(i)
             tmpz(i)=zcomp(i)
             ! Compute Gcorr and save in comp
             xcomp(i)=tmpx(i)-dx(i)
             ycomp(i)=tmpy(i)-dy(i)
             zcomp(i)=tmpz(i)-dz(i)
             ! Augment current classical gradient with Gcorr
             dx(i)=dx(i)+xcomp(i)
             dy(i)=dy(i)+ycomp(i)
             dz(i)=dz(i)+zcomp(i)
          enddo
          optcnt=-9999
          ! Compute and correct energy terms 
          do i=1,LENENT
             tmpe(i)=ecorr(i)
             ecorr(i)=eterm(i)-tmpe(i)
             eterm(i)=eterm(i)-ecorr(i)
          enddo
       else 
          ! Correct current forces 
          do i=1,natom 
             dx(i)=dx(i)+xcomp(i)
             dy(i)=dy(i)+ycomp(i)
             dz(i)=dz(i)+zcomp(i)
          enddo
          ! Compute and correct energy terms 
          do i=1,LENENT
             eterm(i)=eterm(i)-ecorr(i)
          enddo
       endif
#endif /* (comp2)*/
    ENDIF
    QFRSTMICIT=.FALSE.
    RETURN 
  END SUBROUTINE MICROCORR
  ! 
  SUBROUTINE QMCUTS(QX,QY,QZ,PX,PY,PZ,QMM,SQMM,SQMMG)
    !-----------------------------------------------------------------------
    !     Implement the cutoff method: The best is force shift
    !
    !     This routine gets called in the inner loop of one electron integrals for 1/r
    !
    !     QX,QY,QZ  - coordinates of the external (MM) charge
    !     PX,PY,PZ  - weighted center of the gaussian product: px=(a1*x1+a2*x2)/(a1+a2)
    !                 a1,a2 gaussian exponents, x1,x2 - coordinates of each gaussian in the product
    !     QMM       - original MM charge
    !     SQMM      - scaled MM charge
    !     SQMMG     - derivative for SQMM
    !
    use chm_kinds
    use number
    use dimens_fcm
    use inbnd
    use gamess_fcm
    !
    real(chm_real) QX,QY,QZ,PX,PY,PZ,QMM,SQMM,SQMMG
    !
    real(chm_real) RR,XX,YY,ZZ,C2OFNB,R,T
    !
    IF(.NOT.QCUTFLAG)THEN
       SQMM=QMM
       RETURN
    ELSE
       !
       !     Just start with force shift. Later make the code jump to individual
       !     cutoff method similar to what is in enbaexp.src
       !
       XX=QX-PX
       YY=QY-PY
       ZZ=QZ-PZ
       RR=XX*XX+YY*YY+ZZ*ZZ
       C2OFNB=CTOFNB*CTOFNB
       SQMM=ZERO
       IF(RR.LT.C2OFNB)THEN
          R=SQRT(RR)
          T=ONE-R/CTOFNB
          SQMM=QMM*T*T
          !     The following is not enough!
          !     it needs to be multiplied by XX,YY,ZZ!
          SQMMG=MINTWO*QMM*T/CTOFNB 
       ENDIF
       !     
    ENDIF
    !
    RETURN
  END SUBROUTINE QMCUTS
  !

#endif /* (allqm)*/
end module gukini_mod

!

#if KEY_GAMESS==1 /*gamess_util2*/
#if KEY_PARALLEL==0
!     The following is to enable compilation of CHARMM/GAMESS
!     without PARALLEL preflx keyword
!
!  WARNING: This has to be outside the module since it is
!           called from the legacy code
!  XXX these belong in machdep/paral*
!
!-----------------------------------------------------------------------
!CC      SUBROUTINE DDI_BCAST(MSGID,MSGTYP,BUFF,MSGLEN,IFROM)
!CC      REAL(CHM_REAL) BUFF(*)
!CC      CHARACTER*1 MSGTYP
!CC      INTEGER MSGID,MSGLEN,IFROM
!CC      RETURN
!CC      END

#if KEY_ENSEMBLE==0
SUBROUTINE GCOMB(X,N)
  use chm_kinds
  real(chm_real) X(*)
  INTEGER N
  RETURN
END SUBROUTINE GCOMB
#endif 

SUBROUTINE PSYNC()
  RETURN
END SUBROUTINE PSYNC

#endif 
!
#endif /* (gamess_util2)*/

subroutine guk_dummy()
  return
end subroutine guk_dummy

